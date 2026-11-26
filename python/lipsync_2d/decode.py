"""Viterbi + min-duration + silence-gate hysteresis post-filter for 50 Hz visemes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lipsync_2d.visemes import RHUBARB_SHAPES


@dataclass(frozen=True)
class MouthCue:
    start: float
    end: float
    value: str


def _apply_silence_hysteresis(
    probs: np.ndarray,
    silence_idx: int,
    sil_enter_prob: float,
    sil_exit_prob: float,
    audio_rms_db: np.ndarray | None,
    rms_silence_db: float,
) -> np.ndarray:
    """Return hysteresis-gated boolean mask [T] where True forces silence."""
    t_len = probs.shape[0]
    sil_mask = np.zeros(t_len, dtype=bool)
    in_silence = True
    for t in range(t_len):
        p_sil = float(probs[t, silence_idx])
        p_speech = 1.0 - p_sil
        rms_quiet = audio_rms_db is not None and float(audio_rms_db[t]) < rms_silence_db
        if in_silence:
            if not rms_quiet and p_speech >= sil_exit_prob:
                in_silence = False
        else:
            if rms_quiet or p_sil >= sil_enter_prob:
                in_silence = True
        sil_mask[t] = in_silence
    return sil_mask


def _viterbi_smooth(
    log_probs: np.ndarray,
    switch_penalty: float,
) -> np.ndarray:
    """Exact Viterbi decoding with uniform transition penalty for switching states."""
    t_len, n_classes = log_probs.shape
    if t_len == 0:
        return np.zeros((0,), dtype=np.int64)

    dp = np.empty((t_len, n_classes), dtype=np.float64)
    backptr = np.empty((t_len, n_classes), dtype=np.int64)
    dp[0] = log_probs[0]
    backptr[0] = 0

    for t in range(1, t_len):
        prev = dp[t - 1]
        best_prev_idx = int(np.argmax(prev))
        best_prev_val = prev[best_prev_idx]
        # Candidate from switching from best previous state
        switch_val = best_prev_val - switch_penalty
        stay_better = prev >= switch_val
        dp[t] = np.where(stay_better, prev, switch_val) + log_probs[t]
        backptr[t] = np.where(stay_better, np.arange(n_classes), best_prev_idx)

    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]
    return path


def _merge_short_segments(
    path: np.ndarray,
    probs: np.ndarray,
    min_frames: int,
    protected_classes: set[int] | None = None,
    protected_min_frames: int = 1,
) -> np.ndarray:
    """Merge runs shorter than min_frames into the higher-confidence adjacent neighbor."""
    if path.size == 0 or min_frames <= 1:
        return path.copy()

    protected = protected_classes or set()
    segments: list[list[int]] = []
    start = 0
    for i in range(1, len(path)):
        if path[i] != path[start]:
            segments.append([start, i, int(path[start])])
            start = i
    segments.append([start, len(path), int(path[start])])

    while len(segments) > 1:
        shortest_idx = -1
        shortest_len = 10**9
        for idx, (s, e, cls) in enumerate(segments):
            req = protected_min_frames if cls in protected else min_frames
            seg_len = e - s
            if seg_len < req and seg_len < shortest_len:
                shortest_len = seg_len
                shortest_idx = idx

        if shortest_idx == -1:
            break

        s, e, _ = segments[shortest_idx]
        if shortest_idx == 0:
            segments[1][0] = s
        elif shortest_idx == len(segments) - 1:
            segments[-2][1] = e
        else:
            left_cls = segments[shortest_idx - 1][2]
            right_cls = segments[shortest_idx + 1][2]
            left_score = float(probs[s:e, left_cls].mean())
            right_score = float(probs[s:e, right_cls].mean())
            if left_score >= right_score:
                segments[shortest_idx - 1][1] = e
            else:
                segments[shortest_idx + 1][0] = s
        segments.pop(shortest_idx)

        # Coalesce adjacent identical segments created by merge.
        merged: list[list[int]] = []
        for seg in segments:
            if merged and merged[-1][2] == seg[2]:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        segments = merged

    out = np.empty_like(path)
    for s, e, cls in segments:
        out[s:e] = cls
    return out


def decode_cues(
    probs: np.ndarray,
    labels: tuple[str, ...] = RHUBARB_SHAPES,
    frame_dt: float = 0.02,
    silence_idx: int = 0,
    switch_penalty: float = 1.2,
    min_duration_ms: float = 40.0,
    plosive_min_duration_ms: float = 20.0,
    plosive_indices: tuple[int, ...] = (1,),
    sil_enter_prob: float = 0.65,
    sil_exit_prob: float = 0.45,
    audio_rms_db: np.ndarray | None = None,
    rms_silence_db: float = -52.0,
) -> list[MouthCue]:
    """Decode 50 Hz class probabilities into jitter-free MouthCue intervals."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return []
    if p.shape[1] != len(labels):
        raise ValueError(f"Expected {len(labels)} classes, got {p.shape[1]}")

    sil_mask = _apply_silence_hysteresis(
        p,
        silence_idx=silence_idx,
        sil_enter_prob=sil_enter_prob,
        sil_exit_prob=sil_exit_prob,
        audio_rms_db=audio_rms_db,
        rms_silence_db=rms_silence_db,
    )

    adjusted = np.clip(p, 1e-6, 1.0)
    log_probs = np.log(adjusted)
    for t in range(p.shape[0]):
        if sil_mask[t]:
            log_probs[t, :] = -12.0
            log_probs[t, silence_idx] = 0.0
        else:
            log_probs[t, silence_idx] = min(log_probs[t, silence_idx], -4.0)

    path = _viterbi_smooth(log_probs, switch_penalty=switch_penalty)
    min_frames = max(1, int(round((min_duration_ms / 1000.0) / frame_dt)))
    prot_frames = max(1, int(round((plosive_min_duration_ms / 1000.0) / frame_dt)))
    path = _merge_short_segments(
        path,
        adjusted,
        min_frames=min_frames,
        protected_classes=set(plosive_indices),
        protected_min_frames=prot_frames,
    )

    cues: list[MouthCue] = []
    start = 0
    for i in range(1, len(path)):
        if path[i] != path[start]:
            cues.append(
                MouthCue(
                    start=round(start * frame_dt, 4),
                    end=round(i * frame_dt, 4),
                    value=labels[int(path[start])],
                )
            )
            start = i
    cues.append(
        MouthCue(
            start=round(start * frame_dt, 4),
            end=round(len(path) * frame_dt, 4),
            value=labels[int(path[start])],
        )
    )
    return cues
