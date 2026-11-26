"""Tests for evaluation metrics and MFCC baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.metrics import (
    benchmark_onnx_model,
    compute_boundary_onset_mae_ms,
    compute_classification_f1,
    compute_high_pitch_relative_drop_pct,
    compute_silence_false_positive_rate_pct,
    evaluate_predictions,
)
from eval.mfcc_baseline import MFCCBaseline, log_mel_to_mfcc_50hz
from lipsync_2d.export_onnx import export_offline_onnx
from lipsync_2d.model import LipSyncStudent
from lipsync_2d.visemes import RHUBARB_SHAPES


def test_metrics_exact_calculations() -> None:
    y_true = np.array([0, 0, 1, 1, 4, 4, 7, 7, 0, 0], dtype=np.int64)
    # Shift one transition by 1 frame (20 ms) and add 1 silence FP frame out of 4 silence frames
    y_pred = np.array([0, 1, 1, 1, 4, 4, 7, 7, 0, 0], dtype=np.int64)

    f1_res = compute_classification_f1(y_true, y_pred, RHUBARB_SHAPES)
    assert f1_res["per_class_f1"]["G"] == 1.0
    assert f1_res["per_class_f1"]["D"] == 1.0
    assert 0.7 < f1_res["macro_f1"] < 1.0

    # 4 silence frames in y_true: frame 1 predicted as 1 -> 1/4 = 25%
    sil_fp = compute_silence_false_positive_rate_pct(y_true, y_pred, silence_idx=0)
    assert sil_fp == 25.0

    # Transitions in y_true at indices 2, 4, 6, 8; y_pred has transitions at 1, 4, 6, 8 -> errors [20, 0, 0, 0] -> MAE 5.0 ms
    mae_ms = compute_boundary_onset_mae_ms(y_true, y_pred, frame_dt=0.02)
    assert mae_ms == 5.0

    drop = compute_high_pitch_relative_drop_pct(0.80, 0.72)
    assert abs(drop - 10.0) < 1e-4

    full = evaluate_predictions(
        r9_true=y_true,
        r9_pred=y_pred,
        r9_high_pitch_true=y_true,
        r9_high_pitch_pred=y_pred,
    )
    assert full["bilabial_A_f1"] == f1_res["per_class_f1"]["A"]
    assert full["labiodental_G_f1"] == 1.0
    assert full["high_pitch_relative_f1_drop_pct"] == 0.0


def test_mfcc_baseline_fit_and_predict() -> None:
    rng = np.random.default_rng(11)
    mel_c0 = rng.normal(-2.0, 0.1, size=(40, 80)).astype(np.float32)
    mel_c4 = rng.normal(1.5, 0.1, size=(40, 80)).astype(np.float32)
    mel_c4[:, :20] += 3.0

    x0 = log_mel_to_mfcc_50hz(mel_c0)
    x4 = log_mel_to_mfcc_50hz(mel_c4)
    x = np.concatenate([x0, x4], axis=0)
    y = np.array([0] * len(x0) + [4] * len(x4), dtype=np.int64)

    clf = MFCCBaseline(n_classes=9).fit(x, y)
    probs = clf.predict_proba(x)
    assert probs.shape == (len(x), 9)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    assert float(np.mean(clf.predict(x) == y)) > 0.95


def test_benchmark_onnx_model(tmp_path: Path) -> None:
    onnx_path = export_offline_onnx(LipSyncStudent().eval(), tmp_path / "bench.onnx")
    bench = benchmark_onnx_model(onnx_path, streaming=False, n_warmup=1, n_iters=3, n_50hz_frames=10)
    assert bench["onnx_size_mb"] > 0.0
    assert bench["ms_per_20ms_frame"] > 0.0
