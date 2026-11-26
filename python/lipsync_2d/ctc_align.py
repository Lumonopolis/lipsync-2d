"""CTC blank-to-acoustic-span alignment and energy-based speech activity detection."""

from __future__ import annotations

import numpy as np


def detect_speech_activity(
    energy_db: np.ndarray,
    abs_floor_db: float = -45.0,
    rel_range_db: float = 32.0,
    median_window: int = 3,
    bridge_gap_frames: int = 2,
) -> np.ndarray:
    """Detect active speech frames at 50 Hz without dropping brief stop-closure dips."""
    e = np.asarray(energy_db, dtype=np.float32).reshape(-1)
    if e.size == 0:
        return np.zeros((0,), dtype=bool)

    peak_db = float(np.max(e))
    thresh = max(abs_floor_db, peak_db - rel_range_db)
    raw = e >= thresh

    # 3-frame median removes isolated 1-frame noise bursts inside true silence.
    if median_window >= 3 and e.size >= median_window:
        pad = median_window // 2
        padded = np.pad(raw.astype(np.int32), (pad, pad), mode="edge")
        smoothed = np.empty_like(raw)
        for i in range(e.size):
            smoothed[i] = padded[i : i + median_window].sum() > (median_window // 2)
    else:
        smoothed = raw.copy()

    # Morphological closing bridges brief stop closures (<= 2 frames = 40 ms, up to 60 ms across edges).
    if bridge_gap_frames > 0 and e.size >= 3:
        out = smoothed.copy()
        true_idx = np.where(smoothed)[0]
        if true_idx.size >= 2:
            for a, b in zip(true_idx[:-1], true_idx[1:]):
                gap = b - a - 1
                if 0 < gap <= bridge_gap_frames:
                    out[a + 1 : b] = True
        return out
    return smoothed


def ctc_posteriors_to_spans(
    p_ipa: np.ndarray,
    energy_db: np.ndarray,
    blank_idx: int = 0,
    special_count: int = 4,
    nonblank_thresh: float = 0.15,
    max_gap_frames: int = 8,
    abs_floor_db: float = -45.0,
    rel_range_db: float = 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand CTC phoneme spikes across acoustic spans while gating genuine silence.

    Args:
        p_ipa: [T_50hz, V_ipa] teacher softmax posteriors.
        energy_db: [T_50hz] RMS frame energy in dBFS.
        blank_idx: CTC blank token id (default 0 for <pad>).
        special_count: Number of leading special tokens (<pad>, <s>, </s>, <unk>).
        nonblank_thresh: Minimum non-blank probability mass to treat a frame as a spike.
        max_gap_frames: Maximum gap (in 20 ms frames; 8 = 160 ms) to bridge from a spike.

    Returns:
        clean_ipa: [T_50hz, V_ipa] dense span-aligned distributions summing to 1.0.
        is_speech: [T_50hz] boolean speech-activity mask.
    """
    probs = np.asarray(p_ipa, dtype=np.float32)
    t_len, vocab_size = probs.shape
    if t_len == 0:
        return np.zeros((0, vocab_size), dtype=np.float32), np.zeros((0,), dtype=bool)

    e_db = np.asarray(energy_db, dtype=np.float32).reshape(-1)
    if e_db.shape[0] < t_len:
        e_db = np.pad(e_db, (0, t_len - e_db.shape[0]), mode="edge")
    else:
        e_db = e_db[:t_len]

    is_speech = detect_speech_activity(
        e_db,
        abs_floor_db=abs_floor_db,
        rel_range_db=rel_range_db,
    )

    p_phone = probs.copy()
    p_phone[:, :special_count] = 0.0
    phone_mass = p_phone.sum(axis=-1)
    p_phone_norm = p_phone / np.maximum(phone_mass[:, None], 1e-8)

    argmax_ids = np.argmax(probs, axis=-1)
    is_spike = is_speech & ((argmax_ids >= special_count) | (phone_mass > nonblank_thresh))
    spike_indices = np.where(is_spike)[0]

    clean_ipa = np.zeros((t_len, vocab_size), dtype=np.float32)
    for t in range(t_len):
        if not is_speech[t]:
            clean_ipa[t, blank_idx] = 1.0
            continue

        if spike_indices.size == 0:
            if phone_mass[t] > 1e-6:
                clean_ipa[t] = p_phone_norm[t]
            else:
                clean_ipa[t, blank_idx] = 1.0
            continue

        # Nearest spike naturally partitions adjacent spikes at their exact midpoint.
        dists = np.abs(spike_indices - t)
        nearest_pos = int(np.argmin(dists))
        nearest_dist = int(dists[nearest_pos])
        nearest_t = int(spike_indices[nearest_pos])

        if nearest_dist > max_gap_frames:
            clean_ipa[t, blank_idx] = 1.0
            continue

        span_dist = p_phone_norm[nearest_t]
        if phone_mass[t] > nonblank_thresh:
            blend = float(np.clip((phone_mass[t] - nonblank_thresh) / (1.0 - nonblank_thresh), 0.0, 1.0))
            clean_ipa[t] = blend * p_phone_norm[t] + (1.0 - blend) * span_dist
        else:
            clean_ipa[t] = span_dist

    clean_ipa /= np.maximum(clean_ipa.sum(axis=-1, keepdims=True), 1e-8)
    return clean_ipa, is_speech
