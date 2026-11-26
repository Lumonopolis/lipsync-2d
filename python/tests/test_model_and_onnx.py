"""Tests for causal student model, ONNX export, quantization, and streaming equivalence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from lipsync_2d.export_onnx import export_all_variants
from lipsync_2d.features import SAMPLE_RATE, extract_log_mel_torch
from lipsync_2d.model import LipSyncStudent, count_parameters


def test_student_param_count_and_shapes() -> None:
    model = LipSyncStudent().eval()
    n_params = count_parameters(model)
    assert 2_000_000 <= n_params <= 6_000_000

    mel = torch.randn(2, 50, 80)
    with torch.no_grad():
        v15, r9, curves = model(mel)
    assert v15.shape == (2, 25, 15)
    assert r9.shape == (2, 25, 9)
    assert curves.shape == (2, 25, 3)
    assert torch.all((curves >= 0.0) & (curves <= 1.0))


def test_end_to_end_causal_invariance_at_t_plus_lookahead() -> None:
    torch.manual_seed(0)
    model = LipSyncStudent().eval()
    rng = np.random.default_rng(99)
    audio_a = (rng.standard_normal(SAMPLE_RATE) * 0.25).astype(np.float32)
    audio_b = audio_a.copy()

    # Bounded lookahead = 2 frames (40 ms): frame t=9 sees up to (t + 1 + 2) * 20ms = 240 ms.
    t_frame = 9
    cutoff_sample = (t_frame + 1 + model.config.lookahead_frames) * 320
    audio_b[cutoff_sample:] = rng.standard_normal(SAMPLE_RATE - cutoff_sample).astype(np.float32)

    with torch.no_grad():
        mel_a = extract_log_mel_torch(torch.from_numpy(audio_a), norm="fixed", causal=True).unsqueeze(0)
        mel_b = extract_log_mel_torch(torch.from_numpy(audio_b), norm="fixed", causal=True).unsqueeze(0)
        v15_a, r9_a, c_a = model(mel_a)
        v15_b, r9_b, c_b = model(mel_b)

    assert torch.max(torch.abs(v15_a[:, : t_frame + 1] - v15_b[:, : t_frame + 1])).item() == 0.0
    assert torch.max(torch.abs(r9_a[:, : t_frame + 1] - r9_b[:, : t_frame + 1])).item() == 0.0
    assert torch.max(torch.abs(c_a[:, : t_frame + 1] - c_b[:, : t_frame + 1])).item() == 0.0


def test_onnx_export_streaming_equivalence_and_size(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = LipSyncStudent().eval()
    paths = export_all_variants(model, tmp_path)

    assert paths["offline_fp16"].stat().st_size < 12 * 1024 * 1024
    assert paths["streaming_fp16"].stat().st_size < 12 * 1024 * 1024
    assert paths["offline_int8"].stat().st_size < 10 * 1024 * 1024

    mel = torch.randn(1, 30, 80, dtype=torch.float32)
    with torch.no_grad():
        pt_v15, pt_r9, pt_c = model(mel)

    sess_off = ort.InferenceSession(str(paths["offline_fp32"]), providers=["CPUExecutionProvider"])
    off_v15, off_r9, off_c = sess_off.run(None, {"mel": mel.numpy()})

    assert np.max(np.abs(pt_v15.numpy() - off_v15)) < 1e-4
    assert np.max(np.abs(pt_r9.numpy() - off_r9)) < 1e-4
    assert np.max(np.abs(pt_c.numpy() - off_c)) < 1e-4

    # Streaming emits identical frames shifted by lookahead_frames (zero-padded tail matches offline).
    la = model.config.lookahead_frames
    mel_stream_in = torch.cat([mel, torch.zeros(1, la * 2, 80, dtype=torch.float32)], dim=1)
    total_10ms = mel_stream_in.shape[1]
    sess_str = ort.InferenceSession(str(paths["streaming_fp32"]), providers=["CPUExecutionProvider"])
    for chunk_10ms in (2, 6, 10):
        h_in, cc_in = model.init_state(batch_size=1)
        h_np, cc_np = h_in.numpy(), cc_in.numpy()
        v15_chunks, r9_chunks, c_chunks = [], [], []
        for start in range(0, total_10ms, chunk_10ms):
            chunk = mel_stream_in[:, start : start + chunk_10ms].numpy()
            if chunk.shape[1] < 2:
                continue
            sv15, sr9, sc, h_np, cc_np = sess_str.run(
                None,
                {
                    "mel_chunk": chunk,
                    "h_in": h_np,
                    "conv_cache_in": cc_np,
                },
            )
            v15_chunks.append(sv15)
            r9_chunks.append(sr9)
            c_chunks.append(sc)
        str_v15 = np.concatenate(v15_chunks, axis=1)[:, la:]
        str_r9 = np.concatenate(r9_chunks, axis=1)[:, la:]
        str_c = np.concatenate(c_chunks, axis=1)[:, la:]
        n_cmp = min(off_v15.shape[1], str_v15.shape[1])
        assert np.max(np.abs(off_v15[:, :n_cmp] - str_v15[:, :n_cmp])) < 1e-4
        assert np.max(np.abs(off_r9[:, :n_cmp] - str_r9[:, :n_cmp])) < 1e-4
        assert np.max(np.abs(off_c[:, :n_cmp] - str_c[:, :n_cmp])) < 1e-4

    # Verify FP16 and INT8 models execute cleanly
    for key in ("offline_fp16", "offline_int8"):
        sess = ort.InferenceSession(str(paths[key]), providers=["CPUExecutionProvider"])
        out_v15, out_r9, out_c = sess.run(None, {"mel": mel.numpy()})
        assert out_v15.shape == (1, 15, 15)
        assert out_r9.shape == (1, 15, 9)
        assert out_c.shape == (1, 15, 3)
