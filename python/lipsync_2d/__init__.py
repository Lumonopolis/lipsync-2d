"""Real-time multilingual 2D lip-sync and viseme prediction."""

from lipsync_2d.ctc_align import ctc_posteriors_to_spans, detect_speech_activity
from lipsync_2d.visemes import (
    OCULUS_VISEMES,
    RHUBARB_SHAPES,
    VISEME15_TO_RHUBARB9_IDX,
    build_vocab_projection_matrices,
    ipa_to_rhubarb9,
    ipa_to_viseme15,
    load_xlsr_espeak_vocab,
    viseme15_to_rhubarb9,
)

__all__ = [
    "OCULUS_VISEMES",
    "RHUBARB_SHAPES",
    "VISEME15_TO_RHUBARB9_IDX",
    "build_vocab_projection_matrices",
    "ctc_posteriors_to_spans",
    "detect_speech_activity",
    "ipa_to_rhubarb9",
    "ipa_to_viseme15",
    "load_xlsr_espeak_vocab",
    "viseme15_to_rhubarb9",
]
