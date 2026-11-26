"""Shared 16 kHz mono -> 80-bin Log-Mel Spectrogram frontend (NumPy & PyTorch)."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

SAMPLE_RATE: int = 16000
N_FFT: int = 512
WIN_LENGTH: int = 400
HOP_LENGTH: int = 160
N_MELS: int = 80
F_MIN: float = 64.0
F_MAX: float = 7800.0
LOG_EPS: float = 1e-6
CAUSAL_LEFT_PAD: int = WIN_LENGTH - HOP_LENGTH  # 240 samples

DEFAULT_MEL_MEAN: float = -6.0
DEFAULT_MEL_STD: float = 4.0
DEFAULT_MEL_VAR: float = 4.0
DEFAULT_EMA_DECAY: float = 0.02
AGC_TARGET_RMS: float = 0.125
AGC_MIN_SPEECH_RMS: float = 0.005
AGC_MAX_GAIN: float = 10.0

NormMode = Literal["fixed", "causal", "utterance", "none"]


def normalize_speech_rms(wav: np.ndarray) -> np.ndarray:
    """Apply gentle speech-gated RMS normalization (-18 dBFS target, -46 dBFS gate)."""
    x = np.asarray(wav, dtype=np.float32)
    if x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(x * x, dtype=np.float64)))
    if rms <= AGC_MIN_SPEECH_RMS:
        return x
    gain = min(AGC_MAX_GAIN, AGC_TARGET_RMS / rms)
    if abs(gain - 1.0) < 1e-3:
        return x
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


def _hz_to_mel_slaney(hz: np.ndarray | float) -> np.ndarray | float:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    hz_arr = np.asarray(hz, dtype=np.float64)
    mels = hz_arr / f_sp
    log_mask = hz_arr >= min_log_hz
    mels = np.where(log_mask, min_log_mel + np.log(np.maximum(hz_arr, min_log_hz) / min_log_hz) / logstep, mels)
    return float(mels) if np.ndim(hz) == 0 else mels


def _mel_to_hz_slaney(mels: np.ndarray | float) -> np.ndarray | float:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    mel_arr = np.asarray(mels, dtype=np.float64)
    hz = mel_arr * f_sp
    log_mask = mel_arr >= min_log_mel
    hz = np.where(log_mask, min_log_hz * np.exp(logstep * (mel_arr - min_log_mel)), hz)
    return float(hz) if np.ndim(mels) == 0 else hz


def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Return Slaney-normalized Mel filterbank matrix of shape [n_mels, n_fft // 2 + 1]."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)
    mel_min = _hz_to_mel_slaney(f_min)
    mel_max = _hz_to_mel_slaney(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = _mel_to_hz_slaney(mel_points)

    f_diff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / f_diff[i]
        upper = ramps[i + 2] / f_diff[i + 1]
        enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
        fb[i] = np.maximum(0.0, np.minimum(lower, upper)) * enorm
    return fb.astype(np.float32)


def build_hann_window(win_length: int = WIN_LENGTH) -> np.ndarray:
    """Return periodic Hann window of shape [win_length] matching torch.hann_window(..., periodic=True)."""
    n = np.arange(win_length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)).astype(np.float32)


def export_mel_filterbank(dtype: str = "float32") -> dict[str, object]:
    """Export precomputed Mel filterbank and Hann window for WebAudio/TS parity."""
    fb = build_mel_filterbank().astype(dtype)
    win = build_hann_window().astype(dtype)
    return {
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "log_eps": LOG_EPS,
        "causal_left_pad": CAUSAL_LEFT_PAD,
        "mel_filterbank": fb.tolist(),
        "hann_window": win.tolist(),
    }


def _causal_ema_normalize_np(
    log_mel: np.ndarray,
    alpha: float = DEFAULT_EMA_DECAY,
    init_mean: float = DEFAULT_MEL_MEAN,
    init_var: float = DEFAULT_MEL_VAR,
) -> np.ndarray:
    """Apply strictly causal per-bin EMA normalization along time [T, M]."""
    t_len, n_mels = log_mel.shape
    out = np.empty_like(log_mel, dtype=np.float32)
    mean = np.full(n_mels, init_mean, dtype=np.float32)
    var = np.full(n_mels, init_var, dtype=np.float32)
    one_minus = np.float32(1.0 - alpha)
    alpha_f = np.float32(alpha)
    for t in range(t_len):
        x = log_mel[t]
        delta = x - mean
        mean = one_minus * mean + alpha_f * x
        var = one_minus * var + alpha_f * (delta * (x - mean))
        out[t] = (x - mean) / np.sqrt(np.maximum(var, 1e-4))
    return out


def extract_log_mel_np(
    audio: np.ndarray,
    norm: NormMode = "fixed",
    causal: bool = True,
) -> np.ndarray:
    """Compute [T_10ms, 80] Log-Mel spectrogram from 16 kHz mono float waveform in NumPy."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)

    if causal:
        padded = np.pad(wav, (CAUSAL_LEFT_PAD, 0), mode="constant")
    else:
        padded = np.pad(wav, (N_FFT // 2, N_FFT // 2), mode="constant")

    if padded.shape[0] < WIN_LENGTH:
        return np.zeros((0, N_MELS), dtype=np.float32)

    n_frames = 1 + (padded.shape[0] - WIN_LENGTH) // HOP_LENGTH
    if n_frames <= 0:
        return np.zeros((0, N_MELS), dtype=np.float32)

    stride = padded.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, WIN_LENGTH),
        strides=(HOP_LENGTH * stride, stride),
    ).copy()

    window = build_hann_window(WIN_LENGTH)
    frames *= window[None, :]

    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel_fb = build_mel_filterbank()
    mel = power @ mel_fb.T
    log_mel = np.log(np.maximum(mel, 0.0) + LOG_EPS).astype(np.float32)

    if norm == "none":
        return log_mel
    if norm == "fixed":
        return ((log_mel - DEFAULT_MEL_MEAN) / DEFAULT_MEL_STD).astype(np.float32)
    if norm == "utterance":
        mean = log_mel.mean(axis=0, keepdims=True)
        std = np.maximum(log_mel.std(axis=0, keepdims=True), 1e-4)
        return ((log_mel - mean) / std).astype(np.float32)
    if norm == "causal":
        return _causal_ema_normalize_np(log_mel)
    raise ValueError(f"Unsupported norm mode: {norm!r}")


def extract_log_mel_torch(
    audio: "torch.Tensor",
    norm: NormMode = "fixed",
    causal: bool = True,
):
    """Compute [B, T_10ms, 80] or [T_10ms, 80] Log-Mel spectrogram in PyTorch matching NumPy."""
    import torch
    import torch.nn.functional as F

    squeeze_batch = audio.ndim == 1
    wav = audio.unsqueeze(0) if squeeze_batch else audio
    wav = wav.to(dtype=torch.float32)
    device = wav.device

    if causal:
        padded = F.pad(wav, (CAUSAL_LEFT_PAD, 0))
    else:
        padded = F.pad(wav, (N_FFT // 2, N_FFT // 2))

    if padded.shape[-1] < WIN_LENGTH:
        empty = torch.zeros((wav.shape[0], 0, N_MELS), dtype=torch.float32, device=device)
        return empty.squeeze(0) if squeeze_batch else empty

    frames = padded.unfold(dimension=-1, size=WIN_LENGTH, step=HOP_LENGTH)  # [B, T, WIN_LENGTH]
    window = torch.from_numpy(build_hann_window(WIN_LENGTH)).to(device=device, dtype=torch.float32)
    frames = frames * window.view(1, 1, WIN_LENGTH)

    spec = torch.fft.rfft(frames, n=N_FFT, dim=-1)
    power = spec.real.square() + spec.imag.square()
    mel_fb = torch.from_numpy(build_mel_filterbank()).to(device=device, dtype=torch.float32)
    mel = torch.matmul(power, mel_fb.t())
    log_mel = torch.log(torch.clamp(mel, min=0.0) + LOG_EPS)

    if norm == "fixed":
        out = (log_mel - DEFAULT_MEL_MEAN) / DEFAULT_MEL_STD
    elif norm == "utterance":
        mean = log_mel.mean(dim=1, keepdim=True)
        std = torch.clamp(log_mel.std(dim=1, unbiased=False, keepdim=True), min=1e-4)
        out = (log_mel - mean) / std
    elif norm == "causal":
        bsz, t_len, n_mels = log_mel.shape
        out = torch.empty_like(log_mel)
        mean = torch.full((bsz, n_mels), DEFAULT_MEL_MEAN, dtype=torch.float32, device=device)
        var = torch.full((bsz, n_mels), DEFAULT_MEL_VAR, dtype=torch.float32, device=device)
        alpha = DEFAULT_EMA_DECAY
        one_minus = 1.0 - alpha
        for t in range(t_len):
            x = log_mel[:, t, :]
            delta = x - mean
            mean = one_minus * mean + alpha * x
            var = one_minus * var + alpha * (delta * (x - mean))
            out[:, t, :] = (x - mean) / torch.sqrt(torch.clamp(var, min=1e-4))
    elif norm == "none":
        out = log_mel
    else:
        raise ValueError(f"Unsupported norm mode: {norm!r}")

    return out.squeeze(0) if squeeze_batch else out


def compute_frame_energy_db(audio: np.ndarray, frame_ms: int = 20) -> np.ndarray:
    """Compute RMS energy in dB aligned to 50 Hz (20 ms) frames."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    hop = (SAMPLE_RATE * frame_ms) // 1000
    n_frames = wav.shape[0] // hop
    if n_frames == 0:
        return np.zeros((0,), dtype=np.float32)
    trimmed = wav[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    return (20.0 * np.log10(rms)).astype(np.float32)
