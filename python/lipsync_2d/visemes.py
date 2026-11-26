"""Audited IPA to 15 Oculus/VRChat visemes and 9 Rhubarb (X, A-H) mappings."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np

OCULUS_VISEMES: tuple[str, ...] = (
    "sil",
    "PP",
    "FF",
    "TH",
    "DD",
    "kk",
    "CH",
    "SS",
    "nn",
    "RR",
    "aa",
    "E",
    "ih",
    "oh",
    "ou",
)

RHUBARB_SHAPES: tuple[str, ...] = ("X", "A", "B", "C", "D", "E", "F", "G", "H")

VISEME15_TO_ID: dict[str, int] = {name: i for i, name in enumerate(OCULUS_VISEMES)}
RHUBARB9_TO_ID: dict[str, int] = {name: i for i, name in enumerate(RHUBARB_SHAPES)}

# Deterministic 15 -> 9 projection. Laterals map directly to H in IPA_TO_RHUBARB9
# so the direct 9-class head preserves H even though Oculus collapses l into nn.
VISEME15_TO_RHUBARB9_IDX: tuple[int, ...] = (
    0,  # 0: sil -> X
    1,  # 1: PP  -> A
    7,  # 2: FF  -> G
    2,  # 3: TH  -> B
    2,  # 4: DD  -> B
    2,  # 5: kk  -> B
    2,  # 6: CH  -> B
    2,  # 7: SS  -> B
    2,  # 8: nn  -> B
    5,  # 9: RR  -> E
    4,  # 10: aa -> D
    3,  # 11: E  -> C
    2,  # 12: ih -> B
    5,  # 13: oh -> E
    6,  # 14: ou -> F
)

VISEME15_TO_RHUBARB9_MATRIX: np.ndarray = np.eye(len(RHUBARB_SHAPES), dtype=np.float32)[
    list(VISEME15_TO_RHUBARB9_IDX)
]

# Canonical mouth-curve targets [jaw_open, lip_round, lip_close] per Oculus viseme.
VISEME15_MOUTH_CURVES: np.ndarray = np.array(
    [
        [0.00, 0.00, 1.00],  # 0: sil
        [0.00, 0.00, 1.00],  # 1: PP (p/b/m)
        [0.10, 0.00, 0.85],  # 2: FF (f/v)
        [0.20, 0.00, 0.35],  # 3: TH
        [0.22, 0.00, 0.30],  # 4: DD
        [0.28, 0.00, 0.25],  # 5: kk
        [0.25, 0.35, 0.30],  # 6: CH
        [0.16, 0.00, 0.45],  # 7: SS
        [0.24, 0.00, 0.25],  # 8: nn
        [0.32, 0.45, 0.20],  # 9: RR
        [0.95, 0.05, 0.00],  # 10: aa
        [0.62, 0.05, 0.05],  # 11: E
        [0.35, 0.00, 0.15],  # 12: ih
        [0.65, 0.75, 0.05],  # 13: oh
        [0.32, 0.95, 0.15],  # 14: ou
    ],
    dtype=np.float32,
)

SPECIAL_IPA_TOKENS: frozenset[str] = frozenset(
    {"<pad>", "<s>", "</s>", "<unk>", "??", "1", "sil", "sp", "spn", "|", "_", "#"}
)

# Multi-character tokens matched before single-codepoint fallback.
# Values are tuples of (viseme15_idx, rhubarb9_idx, weight) pairs.
MULTI_CHAR_IPA_RULES: dict[str, tuple[tuple[int, int, float], ...]] = {
    "ts.h": ((6, 2, 1.0),),
    "ts.": ((6, 2, 1.0),),
    "tɕh": ((6, 2, 1.0),),
    "tɕʲ": ((6, 2, 1.0),),
    "tɕ": ((6, 2, 1.0),),
    "dʑʲ": ((6, 2, 1.0),),
    "dʑ": ((6, 2, 1.0),),
    "tʃʰ": ((6, 2, 1.0),),
    "tʃʲ": ((6, 2, 1.0),),
    "tʃː": ((6, 2, 1.0),),
    "tʃ": ((6, 2, 1.0),),
    "dʒʲ": ((6, 2, 1.0),),
    "dʒː": ((6, 2, 1.0),),
    "dʒ": ((6, 2, 1.0),),
    "tS": ((6, 2, 1.0),),
    "dZ": ((6, 2, 1.0),),
    "tsh": ((7, 2, 1.0),),
    "tsʲ": ((7, 2, 1.0),),
    "tsː": ((7, 2, 1.0),),
    "ts": ((7, 2, 1.0),),
    "dzː": ((7, 2, 1.0),),
    "dz": ((7, 2, 1.0),),
    "onɡ": ((13, 5, 0.6), (5, 2, 0.4)),
    "pf": ((1, 1, 0.5), (2, 7, 0.5)),
    "sx": ((7, 2, 0.5), (5, 2, 0.5)),
    "əl": ((11, 3, 0.4), (8, 8, 0.6)),
    "ɯᵝ": ((14, 6, 1.0),),
    "r̝̊": ((9, 5, 1.0),),
    "r̝": ((9, 5, 1.0),),
    "s.": ((6, 2, 1.0),),
    "i̪": ((12, 2, 1.0),),
    "t[": ((4, 2, 1.0),),
    "d[": ((4, 2, 1.0),),
    "t^": ((4, 2, 1.0),),
    "d^": ((4, 2, 1.0),),
    "s^": ((7, 2, 1.0),),
    "ɪ^": ((12, 2, 1.0),),
    'u"': ((14, 6, 1.0),),
    "a.": ((10, 4, 1.0),),
    "i.": ((12, 2, 1.0),),
    "u.": ((14, 6, 1.0),),
    "r.": ((9, 5, 1.0),),
    "ph": ((1, 1, 1.0),),
    "pʰ": ((1, 1, 1.0),),
    "bh": ((1, 1, 1.0),),
    "bʰ": ((1, 1, 1.0),),
    "th": ((4, 2, 1.0),),
    "tʰ": ((4, 2, 1.0),),
    "dh": ((4, 2, 1.0),),
    "dʰ": ((4, 2, 1.0),),
    "kh": ((5, 2, 1.0),),
    "kʰ": ((5, 2, 1.0),),
    "ɡh": ((5, 2, 1.0),),
    "ɡʰ": ((5, 2, 1.0),),
    "gh": ((5, 2, 1.0),),
    "gʰ": ((5, 2, 1.0),),
    "ch": ((5, 2, 1.0),),
    "cʰ": ((5, 2, 1.0),),
    "ɟʰ": ((5, 2, 1.0),),
    "ʈʰ": ((4, 2, 1.0),),
    "ɖʰ": ((4, 2, 1.0),),
}

# Single base IPA symbol -> (viseme15_idx, rhubarb9_idx).
BASE_IPA_MAP: dict[str, tuple[int, int]] = {
    # 1: PP / A (bilabials)
    "p": (1, 1),
    "b": (1, 1),
    "m": (1, 1),
    "β": (1, 1),
    "ɸ": (1, 1),
    "ʙ": (1, 1),
    # 2: FF / G (labiodentals)
    "f": (2, 7),
    "v": (2, 7),
    "ʋ": (2, 7),
    "ⱱ": (2, 7),
    # 3: TH / B (interdentals)
    "θ": (3, 2),
    "ð": (3, 2),
    # 4: DD / B (alveolar & retroflex stops)
    "t": (4, 2),
    "d": (4, 2),
    "ʈ": (4, 2),
    "ɖ": (4, 2),
    # 5: kk / B (velar, palatal, uvular, pharyngeal, glottal)
    "k": (5, 2),
    "ɡ": (5, 2),
    "g": (5, 2),
    "ŋ": (5, 2),
    "x": (5, 2),
    "h": (5, 2),
    "ɦ": (5, 2),
    "ɣ": (5, 2),
    "q": (5, 2),
    "ɢ": (5, 2),
    "χ": (5, 2),
    "c": (5, 2),
    "ɟ": (5, 2),
    "ç": (5, 2),
    "ʝ": (5, 2),
    "ħ": (5, 2),
    "ʕ": (5, 2),
    "ʔ": (5, 2),
    "ɴ": (5, 2),
    "N": (5, 2),
    "X": (5, 2),
    # 6: CH / B (postalveolar, alveolo-palatal, retroflex sibilants)
    "ʃ": (6, 2),
    "ʒ": (6, 2),
    "ɕ": (6, 2),
    "ʑ": (6, 2),
    "ʂ": (6, 2),
    "ʐ": (6, 2),
    "S": (6, 2),
    # 7: SS / B (alveolar sibilants)
    "s": (7, 2),
    "z": (7, 2),
    # 8: nn / B (nasals & taps) and nn / H (laterals!)
    "n": (8, 2),
    "ɲ": (8, 2),
    "ɳ": (8, 2),
    "ɾ": (8, 2),
    "ɽ": (8, 2),
    "l": (8, 8),
    "ɫ": (8, 8),
    "ʎ": (8, 8),
    "ɭ": (8, 8),
    "ɬ": (8, 8),
    "ɮ": (8, 8),
    # 9: RR / E (rhotics)
    "ɹ": (9, 5),
    "ɻ": (9, 5),
    "r": (9, 5),
    "ʁ": (9, 5),
    "ʀ": (9, 5),
    # 10: aa / D (open / low vowels)
    "a": (10, 4),
    "ɑ": (10, 4),
    "æ": (10, 4),
    "ɐ": (10, 4),
    "ä": (10, 4),
    # 11: E / C (mid & central vowels)
    "e": (11, 3),
    "ɛ": (11, 3),
    "ə": (11, 3),
    "ʌ": (11, 3),
    "ɜ": (11, 3),
    "ɚ": (11, 3),
    "ɝ": (11, 3),
    "ɵ": (11, 3),
    "ø": (11, 3),
    "œ": (11, 3),
    "ɞ": (11, 3),
    # 12: ih / B (close front & palatal glide)
    "i": (12, 2),
    "ɪ": (12, 2),
    "j": (12, 2),
    "ɨ": (12, 2),
    "ᵻ": (12, 2),
    "ʲ": (12, 2),
    # 13: oh / E (rounded back mid/open vowels)
    "o": (13, 5),
    "ɔ": (13, 5),
    "ɒ": (13, 5),
    # 14: ou / F (close rounded / labiovelar)
    "u": (14, 6),
    "ʊ": (14, 6),
    "w": (14, 6),
    "y": (14, 6),
    "ʉ": (14, 6),
    "ɯ": (14, 6),
    "ɥ": (14, 6),
    "ʍ": (14, 6),
}

_STRIPPABLE_IPA_CHARS: frozenset[str] = frozenset(
    {
        "ː",
        ":",
        "ʰ",
        "ʲ",
        "ʷ",
        "ˤ",
        "ˠ",
        "ˈ",
        "ˌ",
        "͡",
        "͜",
        "1",
        "2",
        "3",
        "4",
        "5",
        "˥",
        "˦",
        "˧",
        "˨",
        "˩",
        ".",
        " ",
    }
)

_SORTED_MULTI_KEYS: tuple[str, ...] = tuple(
    sorted(MULTI_CHAR_IPA_RULES.keys(), key=len, reverse=True)
)


def default_vocab_path() -> Path:
    """Return the bundled XLSR-53 eSpeak vocabulary path."""
    return Path(__file__).resolve().parents[2] / "assets" / "xlsr_espeak_vocab.json"


def load_xlsr_espeak_vocab(path: str | Path | None = None) -> dict[str, int]:
    """Load the XLSR-53 eSpeak vocab.json token-to-id dictionary."""
    vocab_path = Path(path) if path is not None else default_vocab_path()
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def _strip_combining_marks(token: str) -> str:
    # Keep precomposed ä and eSpeak/IPA symbols intact while removing combining
    # diacritics such as nasal tilde, dental bridge, syllabic mark, and ring.
    out: list[str] = []
    for ch in token:
        if ch == "ä":
            out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        for sub in decomposed:
            if unicodedata.category(sub) != "Mn":
                out.append(sub)
    return "".join(out)


def ipa_to_distributions(token: str) -> tuple[np.ndarray, np.ndarray]:
    """Map an IPA token to normalized (viseme15[15], rhubarb9[9]) distributions.

    Handles diacritics, tone digits, length marks, affricates, diphthongs,
    and eSpeak token conventions. Raises ValueError on unmapped non-special IPA.
    """
    if token in SPECIAL_IPA_TOKENS:
        v15 = np.zeros(len(OCULUS_VISEMES), dtype=np.float32)
        r9 = np.zeros(len(RHUBARB_SHAPES), dtype=np.float32)
        v15[0] = 1.0
        r9[0] = 1.0
        return v15, r9

    if token == "ʲ":
        v15 = np.zeros(len(OCULUS_VISEMES), dtype=np.float32)
        r9 = np.zeros(len(RHUBARB_SHAPES), dtype=np.float32)
        v15[12] = 1.0
        r9[2] = 1.0
        return v15, r9

    cleaned = _strip_combining_marks(token.strip())
    v15 = np.zeros(len(OCULUS_VISEMES), dtype=np.float32)
    r9 = np.zeros(len(RHUBARB_SHAPES), dtype=np.float32)
    matched_segments = 0

    pos = 0
    while pos < len(cleaned):
        if cleaned[pos] in _STRIPPABLE_IPA_CHARS:
            pos += 1
            continue

        matched_multi = False
        for key in _SORTED_MULTI_KEYS:
            if cleaned.startswith(key, pos):
                for v_idx, r_idx, weight in MULTI_CHAR_IPA_RULES[key]:
                    v15[v_idx] += weight
                    r9[r_idx] += weight
                matched_segments += 1
                pos += len(key)
                matched_multi = True
                break
        if matched_multi:
            continue

        ch = cleaned[pos]
        if ch in BASE_IPA_MAP:
            v_idx, r_idx = BASE_IPA_MAP[ch]
            v15[v_idx] += 1.0
            r9[r_idx] += 1.0
            matched_segments += 1
            pos += 1
            continue

        raise ValueError(f"Unmapped IPA segment {ch!r} in token {token!r}")

    if matched_segments == 0:
        raise ValueError(f"Token {token!r} produced zero mapped IPA segments")

    v15 /= float(v15.sum())
    r9 /= float(r9.sum())
    return v15, r9


def ipa_to_viseme15(token: str) -> str:
    """Map an IPA token deterministically to its primary Oculus viseme name."""
    v15, _ = ipa_to_distributions(token)
    return OCULUS_VISEMES[int(np.argmax(v15))]


def ipa_to_rhubarb9(token: str) -> str:
    """Map an IPA token deterministically to its primary Rhubarb shape (X, A-H)."""
    _, r9 = ipa_to_distributions(token)
    return RHUBARB_SHAPES[int(np.argmax(r9))]


def viseme15_to_rhubarb9(viseme: int | str) -> str:
    """Deterministically project a 15-class Oculus viseme to Rhubarb 9."""
    idx = VISEME15_TO_ID[viseme] if isinstance(viseme, str) else int(viseme)
    return RHUBARB_SHAPES[VISEME15_TO_RHUBARB9_IDX[idx]]


def build_vocab_projection_matrices(
    vocab: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [V_ipa, 15] and [V_ipa, 9] soft projection matrices for P_ipa @ M."""
    if vocab is None:
        vocab = load_xlsr_espeak_vocab()
    vocab_size = max(vocab.values()) + 1
    m15 = np.zeros((vocab_size, len(OCULUS_VISEMES)), dtype=np.float32)
    m9 = np.zeros((vocab_size, len(RHUBARB_SHAPES)), dtype=np.float32)
    for tok, idx in vocab.items():
        v15, r9 = ipa_to_distributions(tok)
        m15[idx] = v15
        m9[idx] = r9
    return m15, m9
