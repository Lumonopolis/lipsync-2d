"""Tests for 100% IPA-to-Viseme15 and Rhubarb-9 coverage."""

from __future__ import annotations

import pytest

from lipsync_2d.visemes import (
    OCULUS_VISEMES,
    RHUBARB_SHAPES,
    SPECIAL_IPA_TOKENS,
    build_vocab_projection_matrices,
    ipa_to_distributions,
    ipa_to_rhubarb9,
    ipa_to_viseme15,
    load_xlsr_espeak_vocab,
    viseme15_to_rhubarb9,
)


def test_entire_xlsr_espeak_vocab_mapped() -> None:
    vocab = load_xlsr_espeak_vocab()
    assert len(vocab) == 392

    non_special = [tok for tok in vocab if tok not in SPECIAL_IPA_TOKENS]
    assert len(non_special) > 380

    for tok in vocab:
        v15, r9 = ipa_to_distributions(tok)
        assert v15.shape == (15,)
        assert r9.shape == (9,)
        assert abs(float(v15.sum()) - 1.0) < 1e-5
        assert abs(float(r9.sum()) - 1.0) < 1e-5
        assert (v15 >= 0.0).all()
        assert (r9 >= 0.0).all()

    m15, m9 = build_vocab_projection_matrices(vocab)
    assert m15.shape == (len(vocab), 15)
    assert m9.shape == (len(vocab), 9)
    assert abs(m15.sum(axis=1) - 1.0).max() < 1e-5
    assert abs(m9.sum(axis=1) - 1.0).max() < 1e-5


def test_multilingual_sample_ipa_tokens_en_ja_es_fr_de_zh() -> None:
    samples_by_lang = {
        "en": ["p", "b", "m", "f", "v", "θ", "ð", "tʃ", "dʒ", "ɹ", "æ", "ɑː", "ə", "ʊ", "aɪ", "əl"],
        "ja": ["p", "m", "tɕ", "dʑ", "ɕ", "ɾ", "ɯᵝ", "N", "o", "a", "i", "e", "ɸ", "ts"],
        "es": ["p", "b", "β", "f", "t̪", "d̪", "r", "ɾ", "ɲ", "ʎ", "x", "tʃ", "a", "e", "i", "o", "u"],
        "fr": ["p", "b", "m", "f", "v", "ʃ", "ʒ", "ʁ", "y", "ø", "œ", "ɑ̃", "ɔ̃", "ɛ̃", "œ̃", "ɥ"],
        "de": ["pf", "ts", "ç", "x", "ʁ", "ʏ" if False else "yː", "øː", "œ", "aʊ", "ɔɪ", "ɛː", "ʔ"],
        "zh": ["pʰ", "p", "m", "f", "tɕ", "tɕh", "ts.", "ts.h", "s.", "ɻ", "ʐ", "i̪5", "onɡ5", "uaiɜ", "yɛ5ʲ"],
    }
    for lang, tokens in samples_by_lang.items():
        for tok in tokens:
            v15 = ipa_to_viseme15(tok)
            r9 = ipa_to_rhubarb9(tok)
            assert v15 in OCULUS_VISEMES, (lang, tok)
            assert r9 in RHUBARB_SHAPES, (lang, tok)


def test_critical_viseme_and_lateral_h_mappings() -> None:
    for bilabial in ("p", "b", "m", "pʰ", "bː", "β", "ɸ"):
        assert ipa_to_viseme15(bilabial) == "PP"
        assert ipa_to_rhubarb9(bilabial) == "A"

    for labiodental in ("f", "v", "ʋ", "fʲ", "vʲ"):
        assert ipa_to_viseme15(labiodental) == "FF"
        assert ipa_to_rhubarb9(labiodental) == "G"

    for lateral in ("l", "lː", "l̩", "ɫ", "ʎ", "ɭ", "ɬ"):
        assert ipa_to_viseme15(lateral) == "nn"
        assert ipa_to_rhubarb9(lateral) == "H"

    for sil_tok in ("<pad>", "<s>", "</s>", "sil"):
        assert ipa_to_viseme15(sil_tok) == "sil"
        assert ipa_to_rhubarb9(sil_tok) == "X"

    assert viseme15_to_rhubarb9("sil") == "X"
    assert viseme15_to_rhubarb9("PP") == "A"
    assert viseme15_to_rhubarb9("FF") == "G"
    assert viseme15_to_rhubarb9("aa") == "D"
    assert viseme15_to_rhubarb9("ou") == "F"


def test_unmapped_non_ipa_raises() -> None:
    with pytest.raises(ValueError):
        ipa_to_distributions("🚀")
