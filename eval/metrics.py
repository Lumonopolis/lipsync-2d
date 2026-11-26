"""Objective evaluation metrics for 2D lip-sync models and baselines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from lipsync_2d.visemes import OCULUS_VISEMES, RHUBARB_SHAPES


def compute_classification_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute per-class F1, Macro-F1, and Weighted-F1 across present classes."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    n_classes = len(class_names)

    per_class: dict[str, float] = {}
    supports = np.zeros(n_classes, dtype=np.float64)
    f1s = np.zeros(n_classes, dtype=np.float64)

    for idx, name in enumerate(class_names):
        tp = int(np.sum((yt == idx) & (yp == idx)))
        fp = int(np.sum((yt != idx) & (yp == idx)))
        fn = int(np.sum((yt == idx) & (yp != idx)))
        supports[idx] = float(tp + fn)
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        f1s[idx] = f1
        per_class[name] = round(float(f1), 6)

    present = supports > 0
    macro_f1 = float(np.mean(f1s[present])) if np.any(present) else 0.0
    total_support = float(np.sum(supports))
    weighted_f1 = float(np.sum(f1s * supports) / total_support) if total_support > 0 else 0.0
    return {
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class_f1": per_class,
    }


def compute_boundary_onset_mae_ms(
    y_true_seq: np.ndarray | list[np.ndarray],
    y_pred_seq: np.ndarray | list[np.ndarray],
    frame_dt: float = 0.02,
    max_match_ms: float = 200.0,
) -> float:
    """Compute Mean Absolute Error (ms) of predicted transition onsets against true transitions."""
    seqs_true = [y_true_seq] if isinstance(y_true_seq, np.ndarray) and y_true_seq.ndim == 1 else list(y_true_seq)
    seqs_pred = [y_pred_seq] if isinstance(y_pred_seq, np.ndarray) and y_pred_seq.ndim == 1 else list(y_pred_seq)

    max_frames = int(round((max_match_ms / 1000.0) / frame_dt))
    errors_ms: list[float] = []

    for yt, yp in zip(seqs_true, seqs_pred):
        yt_arr = np.asarray(yt, dtype=np.int64)
        yp_arr = np.asarray(yp, dtype=np.int64)
        if yt_arr.size < 2 or yp_arr.size < 2:
            continue
        true_onsets = np.where(yt_arr[1:] != yt_arr[:-1])[0] + 1
        pred_onsets = np.where(yp_arr[1:] != yp_arr[:-1])[0] + 1
        if true_onsets.size == 0:
            continue
        if pred_onsets.size == 0:
            errors_ms.extend([max_match_ms] * int(true_onsets.size))
            continue

        for t_idx in true_onsets:
            dist = int(np.min(np.abs(pred_onsets - t_idx)))
            err_frames = min(dist, max_frames)
            errors_ms.append(err_frames * frame_dt * 1000.0)

    if not errors_ms:
        return 0.0
    return round(float(np.mean(errors_ms)), 4)


def compute_silence_false_positive_rate_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    silence_idx: int = 0,
) -> float:
    """Return percentage of true silence frames predicted as non-silence (jaw flap rate)."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    sil_mask = yt == silence_idx
    total_sil = int(np.sum(sil_mask))
    if total_sil == 0:
        return 0.0
    fp = int(np.sum(sil_mask & (yp != silence_idx)))
    return round(100.0 * fp / total_sil, 4)


def compute_high_pitch_relative_drop_pct(
    base_macro_f1: float,
    high_pitch_macro_f1: float,
) -> float:
    """Compute relative macro-F1 drop (%) at +4 semitones."""
    if base_macro_f1 <= 1e-8:
        return 0.0
    drop = (base_macro_f1 - high_pitch_macro_f1) / base_macro_f1 * 100.0
    return round(float(drop), 4)


def evaluate_predictions(
    r9_true: np.ndarray,
    r9_pred: np.ndarray,
    v15_true: np.ndarray | None = None,
    v15_pred: np.ndarray | None = None,
    r9_true_seqs: list[np.ndarray] | None = None,
    r9_pred_seqs: list[np.ndarray] | None = None,
    r9_high_pitch_true: np.ndarray | None = None,
    r9_high_pitch_pred: np.ndarray | None = None,
    frame_dt: float = 0.02,
) -> dict[str, Any]:
    """Compute complete evaluation dictionary for Rhubarb-9 and Oculus-15 predictions."""
    r9_stats = compute_classification_f1(r9_true, r9_pred, RHUBARB_SHAPES)
    report: dict[str, Any] = {
        "rhubarb9_macro_f1": r9_stats["macro_f1"],
        "rhubarb9_weighted_f1": r9_stats["weighted_f1"],
        "rhubarb9_per_class_f1": r9_stats["per_class_f1"],
        "bilabial_A_f1": r9_stats["per_class_f1"]["A"],
        "labiodental_G_f1": r9_stats["per_class_f1"]["G"],
        "silence_fp_jaw_flap_pct": compute_silence_false_positive_rate_pct(r9_true, r9_pred, silence_idx=0),
    }

    if v15_true is not None and v15_pred is not None:
        v15_stats = compute_classification_f1(v15_true, v15_pred, OCULUS_VISEMES)
        report["viseme15_macro_f1"] = v15_stats["macro_f1"]
        report["viseme15_weighted_f1"] = v15_stats["weighted_f1"]
        report["viseme15_per_class_f1"] = v15_stats["per_class_f1"]
        report["bilabial_PP_f1"] = v15_stats["per_class_f1"]["PP"]
        report["labiodental_FF_f1"] = v15_stats["per_class_f1"]["FF"]

    if r9_true_seqs is not None and r9_pred_seqs is not None:
        report["boundary_onset_mae_ms"] = compute_boundary_onset_mae_ms(
            r9_true_seqs, r9_pred_seqs, frame_dt=frame_dt
        )
    else:
        report["boundary_onset_mae_ms"] = compute_boundary_onset_mae_ms(
            r9_true, r9_pred, frame_dt=frame_dt
        )

    if r9_high_pitch_true is not None and r9_high_pitch_pred is not None:
        hp_stats = compute_classification_f1(r9_high_pitch_true, r9_high_pitch_pred, RHUBARB_SHAPES)
        report["high_pitch_rhubarb9_macro_f1"] = hp_stats["macro_f1"]
        report["high_pitch_relative_f1_drop_pct"] = compute_high_pitch_relative_drop_pct(
            r9_stats["macro_f1"], hp_stats["macro_f1"]
        )

    return report


def benchmark_onnx_model(
    onnx_path: str | Path,
    streaming: bool = False,
    n_warmup: int = 5,
    n_iters: int = 50,
    n_50hz_frames: int = 50,
) -> dict[str, float]:
    """Measure ONNX file size (MB) and inference latency (ms per 20 ms frame)."""
    import onnxruntime as ort

    path = Path(onnx_path)
    size_mb = round(path.stat().st_size / (1024.0 * 1024.0), 4)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    if streaming:
        inputs_by_name = {inp.name: inp.shape for inp in session.get_inputs()}
        mel_chunk = np.random.randn(1, 2, 80).astype(np.float32)
        h_shape = tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in inputs_by_name.get("h_in", [1, 1, 1]))
        c_shape = tuple(
            int(d) if isinstance(d, int) and d > 0 else 1 for d in inputs_by_name.get("conv_cache_in", [1, 288, 248])
        )
        h_in = np.zeros(h_shape, dtype=np.float32)
        conv_cache = np.zeros(c_shape, dtype=np.float32)
        feed = {"mel_chunk": mel_chunk, "h_in": h_in, "conv_cache_in": conv_cache}
        frames_per_call = 1
    else:
        mel = np.random.randn(1, n_50hz_frames * 2, 80).astype(np.float32)
        feed = {"mel": mel}
        frames_per_call = n_50hz_frames

    for _ in range(n_warmup):
        session.run(None, feed)

    start = time.perf_counter()
    for _ in range(n_iters):
        session.run(None, feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    ms_per_frame = elapsed_ms / (n_iters * frames_per_call)

    return {
        "onnx_size_mb": size_mb,
        "ms_per_20ms_frame": round(float(ms_per_frame), 4),
    }
