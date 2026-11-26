"""Causal ConvNeXt-TCN + Unidirectional GRU streaming lip-sync student model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipsync_2d.visemes import OCULUS_VISEMES, RHUBARB_SHAPES


@dataclass(frozen=True)
class ModelConfig:
    n_mels: int = 80
    channels: int = 288
    mlp_ratio: int = 2
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
    gru_hidden: int = 288
    gru_layers: int = 0
    lookahead_frames: int = 2
    n_viseme15: int = len(OCULUS_VISEMES)
    n_rhubarb9: int = len(RHUBARB_SHAPES)
    n_curves: int = 3
    dropout: float = 0.1

    @property
    def conv_cache_frames(self) -> int:
        return sum((self.kernel_size - 1) * d for d in self.dilations)


class CausalConvNeXtBlock(nn.Module):
    """Causal dilated 1D ConvNeXt block with explicit streaming cache support."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation

        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            padding=0,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        hidden = channels * mlp_ratio
        self.pwconv1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, channels)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for [B, C, T] tensor using causal zero left-padding."""
        residual = x
        x_pad = F.pad(x, (self.left_pad, 0))
        x = self.dwconv(x_pad)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.drop(x * self.gamma)
        return residual + x.transpose(1, 2)

    def forward_stream(
        self, x: torch.Tensor, cache_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward for [B, C, T] and cache_in [B, C, left_pad]."""
        residual = x
        x_cat = torch.cat([cache_in, x], dim=-1)
        cache_out = x_cat[:, :, -self.left_pad :]
        x = self.dwconv(x_cat)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x * self.gamma
        return residual + x.transpose(1, 2), cache_out


class LipSyncStudent(nn.Module):
    """Causal ConvNeXt-TCN + 2-layer UniGRU student model (100 Hz Mel -> 50 Hz outputs)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels

        # Strictly non-overlapping 20 ms patch stem (100 Hz -> 50 Hz) so stem is chunk-stateless.
        self.stem_conv = nn.Conv1d(self.config.n_mels, c, kernel_size=2, stride=2, padding=0)
        self.stem_norm = nn.LayerNorm(c, eps=1e-6)
        self.stem_act = nn.GELU()

        self.blocks = nn.ModuleList(
            [
                CausalConvNeXtBlock(
                    channels=c,
                    kernel_size=self.config.kernel_size,
                    dilation=d,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                )
                for d in self.config.dilations
            ]
        )

        if self.config.gru_layers > 0:
            self.gru: nn.GRU | None = nn.GRU(
                input_size=c,
                hidden_size=self.config.gru_hidden,
                num_layers=self.config.gru_layers,
                batch_first=True,
                bidirectional=False,
            )
            out_dim = self.config.gru_hidden
        else:
            self.gru = None
            out_dim = c
        self.out_norm = nn.LayerNorm(out_dim, eps=1e-6)
        self.viseme15_head = nn.Linear(out_dim, self.config.n_viseme15)
        self.rhubarb9_head = nn.Linear(out_dim, self.config.n_rhubarb9)
        self.mouth_curves_head = nn.Linear(out_dim, self.config.n_curves)

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h_in, conv_cache_in) for streaming inference."""
        h_layers = max(1, self.config.gru_layers)
        h_dim = self.config.gru_hidden if self.config.gru_layers > 0 else 1
        h_in = torch.zeros(h_layers, batch_size, h_dim, device=device, dtype=dtype)
        conv_cache_in = torch.zeros(
            batch_size, self.config.channels, self.config.conv_cache_frames, device=device, dtype=dtype
        )
        return h_in, conv_cache_in

    def _encode_stem(self, mel: torch.Tensor) -> torch.Tensor:
        # Truncate trailing odd frame if T_10ms is odd to match 20 ms 50 Hz framing cleanly.
        t_even = (mel.shape[1] // 2) * 2
        x = mel[:, :t_even, :].transpose(1, 2)  # [B, 80, T_even]
        x = self.stem_conv(x)  # [B, C, T_50hz]
        x = self.stem_act(self.stem_norm(x.transpose(1, 2))).transpose(1, 2)
        return x

    def _decode_heads(
        self, gru_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.out_norm(gru_out)
        viseme15_logits = self.viseme15_head(feat)
        rhubarb9_logits = self.rhubarb9_head(feat)
        mouth_curves = torch.sigmoid(self.mouth_curves_head(feat))
        return viseme15_logits, rhubarb9_logits, mouth_curves

    def forward(
        self, mel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Offline forward pass with bounded 40 ms (2-frame) lookahead: mel [B, T_10ms, 80] -> 50 Hz."""
        la = self.config.lookahead_frames
        if la > 0:
            mel_in = F.pad(mel, (0, 0, 0, la * 2))
        else:
            mel_in = mel
        x = self._encode_stem(mel_in)
        for block in self.blocks:
            x = block(x)
        seq = x.transpose(1, 2)
        if self.gru is not None:
            seq, _ = self.gru(seq)
        if la > 0:
            seq = seq[:, la:, :]
        return self._decode_heads(seq)

    def forward_stream(
        self,
        mel_chunk: torch.Tensor,
        h_in: torch.Tensor,
        conv_cache_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly causal streaming pass with explicit ConvNeXt (and optional GRU) caches."""
        x = self._encode_stem(mel_chunk)
        offset = 0
        next_caches: list[torch.Tensor] = []
        for block in self.blocks:
            pad = block.left_pad
            block_cache = conv_cache_in[:, :, offset : offset + pad]
            x, new_cache = block.forward_stream(x, block_cache)
            next_caches.append(new_cache)
            offset += pad
        conv_cache_out = torch.cat(next_caches, dim=-1)
        seq = x.transpose(1, 2)
        if self.gru is not None:
            seq, h_out = self.gru(seq, h_in)
        else:
            h_out = h_in
        viseme15_logits, rhubarb9_logits, mouth_curves = self._decode_heads(seq)
        return viseme15_logits, rhubarb9_logits, mouth_curves, h_out, conv_cache_out


class StreamingStudentWrapper(nn.Module):
    """Wrapper exposing forward_stream as forward() for clean ONNX export."""

    def __init__(self, student: LipSyncStudent) -> None:
        super().__init__()
        self.student = student

    def forward(
        self,
        mel_chunk: torch.Tensor,
        h_in: torch.Tensor,
        conv_cache_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.student.forward_stream(mel_chunk, h_in, conv_cache_in)


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
