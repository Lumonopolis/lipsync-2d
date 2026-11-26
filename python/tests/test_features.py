"""Tests for shared 16 kHz -> 80-bin Log-Mel frontend and causal invariance."""

from __future__ import annotations

import numpy as np
import torch

from lipsync_2d.features import (
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    build_mel_filterbank,
    export_mel_filterbank,
    extract_log_mel_np,
    extract_log_mel_torch,
)


def test_mel_filterbank_shape_and_export() -> None:
    fb = build_mel_filterbank()
    assert fb.shape == (N_MELS, N_FFT // 2 + 1)
    assert np.all(fb >= 0.0)
    assert np.all(fb.sum(axis=1) > 0.0)

    exported = export_mel_filterbank()
    assert exported["sample_rate"] == SAMPLE_RATE
    assert exported["n_mels"] == N_MELS
    assert len(exported["mel_filterbank"]) == N_MELS
    assert len(exported["hann_window"]) == 400


def test_numpy_pytorch_log_mel_parity() -> None:
    rng = np.random.default_rng(123)
    audio = (rng.standard_normal(SAMPLE_RATE * 2) * 0.25).astype(np.float32)

    for norm in ("causal", "utterance", "none"):
        mel_np = extract_log_mel_np(audio, norm=norm, causal=True)
        mel_th = extract_log_mel_torch(torch.from_numpy(audio), norm=norm, causal=True).numpy()
        assert mel_np.shape == mel_th.shape
        max_diff = float(np.max(np.abs(mel_np - mel_th)))
        assert max_diff < 1e-4, f"norm={norm} max_diff={max_diff}"


def test_feature_causal_invariance() -> None:
    rng = np.random.default_rng(7)
    audio_a = (rng.standard_normal(SAMPLE_RATE) * 0.2).astype(np.float32)
    audio_b = audio_a.copy()
    # Mutate audio strictly after t = 200 ms (+20 ms = 220 ms -> sample 3520)
    cutoff_ms = 220
    cutoff_sample = (SAMPLE_RATE * cutoff_ms) // 1000
    audio_b[cutoff_sample:] = rng.standard_normal(SAMPLE_RATE - cutoff_sample).astype(np.float32)

    mel_a = extract_log_mel_np(audio_a, norm="causal", causal=True)
    mel_b = extract_log_mel_np(audio_b, norm="causal", causal=True)

    # 10 ms frames 0..21 end at or before 220 ms.
    n_frames_before = cutoff_ms // 10
    assert np.max(np.abs(mel_a[:n_frames_before] - mel_b[:n_frames_before])) == 0.0
