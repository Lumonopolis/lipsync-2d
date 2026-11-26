"""Tests for CTC blank-to-acoustic-span alignment and stop-closure retention."""

from __future__ import annotations

import numpy as np

from lipsync_2d.ctc_align import ctc_posteriors_to_spans, detect_speech_activity
from lipsync_2d.visemes import build_vocab_projection_matrices, load_xlsr_espeak_vocab


def test_speech_activity_bridges_brief_stop_closures_and_rejects_noise() -> None:
    # 30 frames (600 ms): silence [0..5], vowel [6..12], 40ms stop dip [13..14], vowel [15..22], silence [23..29]
    energy = np.full(30, -65.0, dtype=np.float32)
    energy[2] = -20.0  # isolated 1-frame click inside silence -> rejected by 3-frame median
    energy[6:13] = -15.0
    energy[13:15] = -55.0  # 40 ms bilabial stop closure dip -> bridged
    energy[15:23] = -14.0

    active = detect_speech_activity(energy)
    assert not np.any(active[:6])
    assert np.all(active[6:23])
    assert not np.any(active[23:])


def test_ctc_blank_expansion_avoids_80pct_silence_and_keeps_bilabial_span() -> None:
    vocab = load_xlsr_espeak_vocab()
    m15, m9 = build_vocab_projection_matrices(vocab)
    v_size = len(vocab)

    idx_a = vocab["a"]
    idx_p = vocab["p"]
    idx_i = vocab["i"]

    # 30 frames (600 ms) total:
    # 0..4: true silence
    # 5..19: spoken /a p i/ where CTC emits <pad> (idx 0) on ~80% of speech frames
    # 20..29: true silence (> 160 ms after last spike)
    t_len = 30
    p_ipa = np.zeros((t_len, v_size), dtype=np.float32)
    p_ipa[:, 0] = 0.96
    p_ipa[:, idx_a] = 0.04

    # Phoneme spikes at frames 6 (/a/), 11 (/p/ burst), 16 (/i/)
    p_ipa[6] = 0.0
    p_ipa[6, idx_a] = 0.90
    p_ipa[6, 0] = 0.10

    p_ipa[11] = 0.0
    p_ipa[11, idx_p] = 0.85
    p_ipa[11, 0] = 0.15

    p_ipa[16] = 0.0
    p_ipa[16, idx_i] = 0.90
    p_ipa[16, 0] = 0.10

    energy = np.full(t_len, -65.0, dtype=np.float32)
    energy[5:20] = -16.0
    energy[9:11] = -52.0  # 40 ms /p/ bilabial closure dip right before burst at frame 11

    clean_ipa, is_speech = ctc_posteriors_to_spans(p_ipa, energy)
    v15_probs = clean_ipa @ m15
    r9_probs = clean_ipa @ m9
    v15_labels = np.argmax(v15_probs, axis=-1)
    r9_labels = np.argmax(r9_probs, axis=-1)

    # Active speech frames 5..19 must NOT collapse to sil (0) / X (0)
    assert np.all(is_speech[5:20])
    assert np.all(v15_labels[5:20] != 0)
    assert np.all(r9_labels[5:20] != 0)

    # Bilabial /p/ (PP=1, A=1) must retain >= 3 frames (60 ms) across its closure/burst span (frames 9..13)
    pp_frames = int(np.sum(v15_labels[5:20] == 1))
    a_frames = int(np.sum(r9_labels[5:20] == 1))
    assert pp_frames >= 3
    assert a_frames >= 3

    # True silence regions remain sil/X
    assert np.all(r9_labels[:5] == 0)
    assert np.all(r9_labels[20:] == 0)
