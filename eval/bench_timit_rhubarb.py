"""Head-to-head benchmark on human-annotated TIMIT phoneme boundaries:
lipsync-2d ONNX vs official Rhubarb Lip Sync CLI (v1.14.0) vs MFCC baseline.

Evaluates normal pitch (0 st), high-pitched (+4 semitones, Rhubarb Issue #174),
and male vs female speaker slices on kylelovesllms/timit_asr official test split.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

from eval.metrics import evaluate_predictions
from lipsync_2d.decode import decode_cues
from lipsync_2d.features import SAMPLE_RATE, compute_frame_energy_db, extract_log_mel_np, normalize_speech_rms
from lipsync_2d.visemes import RHUBARB_SHAPES

RHUBARB_TO_IDX = {s: i for i, s in enumerate(RHUBARB_SHAPES)}

# Official 61-phone TIMIT Arpabet -> 9 Rhubarb shapes (X, A-H)
TIMIT_TO_RHUBARB: dict[str, str] = {
    # X: silence / pause / epenthetic silence
    "h#": "X", "pau": "X", "epi": "X",
    # A: bilabial closures & releases (p, b, m)
    "p": "A", "b": "A", "m": "A", "em": "A", "pcl": "A", "bcl": "A",
    # G: labiodentals (f, v)
    "f": "G", "v": "G",
    # H: laterals (l)
    "l": "H", "el": "H",
    # D: wide open vowels (aa, ae, ah, aw, ay)
    "aa": "D", "ae": "D", "ah": "D", "aw": "D", "ay": "D",
    # C: open mid vowels (eh, ey, ax, ax-h, ix)
    "eh": "C", "ey": "C", "ax": "C", "ax-h": "C", "ix": "C",
    # E: rounded open / r-colored (ao, ow, oy, r, er, axr)
    "ao": "E", "ow": "E", "oy": "E", "r": "E", "er": "E", "axr": "E",
    # F: narrow rounded / puckered (uw, ux, uh, w)
    "uw": "F", "ux": "F", "uh": "F", "w": "F",
    # B: clenched / slightly open consonants & high front vowels
    "t": "B", "d": "B", "k": "B", "g": "B", "s": "B", "z": "B",
    "sh": "B", "zh": "B", "ch": "B", "jh": "B", "th": "B", "dh": "B",
    "n": "B", "ng": "B", "nx": "B", "en": "B", "eng": "B",
    "ih": "B", "iy": "B", "y": "B", "hh": "B", "hv": "B", "dx": "B",
    "tcl": "B", "dcl": "B", "kcl": "B", "gcl": "B", "q": "B",
}


def pitch_shift_preserve_duration(wav: np.ndarray, semitones: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Phase-vocoder style duration-preserving pitch shift using STFT (or fast resample+OLA)."""
    if abs(semitones) < 1e-3:
        return wav.astype(np.float32)
    factor = 2.0 ** (semitones / 12.0)
    # Time-stretch by `factor` via phase vocoder, then resample back to original length
    n_fft = 512
    hop_a = 128
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    x = np.pad(wav.astype(np.float32), (pad, pad))
    n_frames = 1 + (len(x) - n_fft) // hop_a
    if n_frames < 2:
        return wav.astype(np.float32)
    frames = np.stack([x[i * hop_a : i * hop_a + n_fft] * win for i in range(n_frames)], axis=0)
    spec = np.fft.rfft(frames, axis=-1)
    mag = np.abs(spec)
    phase = np.angle(spec)
    omega = 2.0 * np.pi * np.arange(spec.shape[1]) * hop_a / n_fft

    t_steps = np.arange(0.0, n_frames - 1, 1.0 / factor)
    out_spec = np.zeros((len(t_steps), spec.shape[1]), dtype=np.complex64)
    ph_acc = phase[0].copy()
    for out_i, t in enumerate(t_steps):
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, n_frames - 1)
        frac = t - i0
        m = (1.0 - frac) * mag[i0] + frac * mag[i1]
        dphi = phase[i1] - phase[i0] - omega
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        if out_i > 0:
            ph_acc += omega + dphi
        out_spec[out_i] = m * np.exp(1j * ph_acc)

    out_len = (len(t_steps) - 1) * hop_a + n_fft
    y = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    for i in range(len(t_steps)):
        s = i * hop_a
        y[s : s + n_fft] += np.fft.irfft(out_spec[i], n=n_fft).astype(np.float32) * win
        wsum[s : s + n_fft] += win * win
    y /= np.maximum(wsum, 1e-6)
    y = y[pad : out_len - pad]
    # Resample stretched signal back to exact original length -> pitch shifted by `factor`, same timing!
    idx = np.linspace(0.0, max(0, len(y) - 1), len(wav), dtype=np.float64)
    shifted = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
    return np.clip(shifted, -1.0, 1.0)


def phonetic_detail_to_50hz_labels(phonetic_detail: list[dict[str, Any]], n_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Rasterize sample-accurate TIMIT human phonetic labels to 50 Hz (20 ms/frame) Rhubarb-9 indices."""
    n_frames = n_samples // (sr // 50)
    labels = np.zeros(n_frames, dtype=np.int64)
    hop = sr // 50  # 320 samples
    for seg in phonetic_detail:
        phone = str(seg["utterance"]).lower().strip()
        shape = TIMIT_TO_RHUBARB.get(phone, "B")
        idx = RHUBARB_TO_IDX[shape]
        s0 = int(seg["start"])
        s1 = int(seg["stop"])
        f0 = max(0, int(round(s0 / hop)))
        f1 = min(n_frames, max(f0 + 1, int(round(s1 / hop))))
        labels[f0:f1] = idx
    return labels


def write_wav_16k(path: Path, wav: np.ndarray) -> None:
    pcm = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def run_rhubarb_cli(rhubarb_bin: Path, wav_path: Path, n_frames: int) -> np.ndarray:
    res = subprocess.run(
        [str(rhubarb_bin), "-f", "json", "--extendedShapes", "GHX", "-r", "phonetic", str(wav_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout)
    preds = np.zeros(n_frames, dtype=np.int64)
    for cue in data.get("mouthCues", []):
        idx = RHUBARB_TO_IDX.get(cue["value"], 0)
        f0 = max(0, int(round(float(cue["start"]) * 50.0)))
        f1 = min(n_frames, int(round(float(cue["end"]) * 50.0)))
        preds[f0:f1] = idx
    return preds


def run_lipsync_onnx(sess: ort.InferenceSession, wav: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    wav_norm = normalize_speech_rms(wav)
    mel = extract_log_mel_np(wav_norm, norm="fixed", causal=True)
    _, r9_logits, _ = sess.run(None, {"mel": mel[None, :, :].astype(np.float32)})
    logits = r9_logits[0][:n_frames]
    raw_pred = np.argmax(logits, axis=-1).astype(np.int64)
    # Also compute Viterbi-decoded cues rasterized to 50 Hz for direct apples-to-apples vs Rhubarb cues
    ex = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-8)
    rms_db = compute_frame_energy_db(wav_norm, frame_ms=20)[: len(probs)]
    cues = decode_cues(probs, audio_rms_db=rms_db)
    cue_pred = np.zeros(n_frames, dtype=np.int64)
    for c in cues:
        f0 = max(0, int(round(c.start * 50.0)))
        f1 = min(n_frames, int(round(c.end * 50.0)))
        cue_pred[f0:f1] = RHUBARB_TO_IDX[c.value]
    if len(raw_pred) < n_frames:
        raw_pred = np.pad(raw_pred, (0, n_frames - len(raw_pred)))
    return raw_pred, cue_pred


def benchmark_timit(
    onnx_path: Path,
    rhubarb_bin: Path | None = None,
    n_utts: int = 160,
    output_json: Path | None = None,
) -> dict[str, Any]:
    tp = hf_hub_download("kylelovesllms/timit_asr", "data/test-00000-of-00001.parquet", repo_type="dataset")
    tbl = pq.read_table(tp, columns=["audio", "phonetic_detail", "speaker_sex", "speaker_id"])

    # Balanced 50/50 distinct Male + Female speakers from official TIMIT test split (1 utterance per speaker, zero training overlap!)
    sexes = tbl["speaker_sex"].to_pylist()
    spk_ids = tbl["speaker_id"].to_pylist()
    m_idx: list[int] = []
    f_idx: list[int] = []
    seen_spk: set[str] = set()
    half = n_utts // 2
    for i, (sex, spk) in enumerate(zip(sexes, spk_ids)):
        if spk in seen_spk:
            continue
        if sex == "M" and len(m_idx) < half:
            m_idx.append(i)
            seen_spk.add(spk)
        elif sex == "F" and len(f_idx) < (n_utts - half):
            f_idx.append(i)
            seen_spk.add(spk)
        if len(m_idx) >= half and len(f_idx) >= (n_utts - half):
            break
    indices = m_idx + f_idx
    unique_speakers_by_sex = {"M": len(m_idx), "F": len(f_idx)}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def eval_condition(semitones: float) -> dict[str, Any]:
        trues, ls_raws, ls_cues, rh_preds = [], [], [], []
        by_sex: dict[str, dict[str, list[np.ndarray]]] = {
            "M": {"t": [], "ls": [], "rh": []},
            "F": {"t": [], "ls": [], "rh": []},
        }
        with tempfile.TemporaryDirectory() as td:
            wav_file = Path(td) / "utt.wav"
            for idx in indices:
                au = tbl["audio"][idx].as_py()
                pd = tbl["phonetic_detail"][idx].as_py()
                sex = str(tbl["speaker_sex"][idx].as_py())
                wav, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                if sr != SAMPLE_RATE:
                    g = int(np.gcd(sr, SAMPLE_RATE))
                    wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
                y_true = phonetic_detail_to_50hz_labels(pd, len(wav), SAMPLE_RATE)
                n_f = len(y_true)
                if n_f < 10:
                    continue
                wav_cond = pitch_shift_preserve_duration(wav, semitones, SAMPLE_RATE)
                raw_p, cue_p = run_lipsync_onnx(sess, wav_cond, n_f)
                trues.append(y_true)
                ls_raws.append(raw_p)
                ls_cues.append(cue_p)
                by_sex[sex]["t"].append(y_true)
                by_sex[sex]["ls"].append(raw_p)
                if rhubarb_bin and rhubarb_bin.exists():
                    write_wav_16k(wav_file, wav_cond)
                    rp = run_rhubarb_cli(rhubarb_bin, wav_file, n_f)
                    rh_preds.append(rp)
                    by_sex[sex]["rh"].append(rp)

        out: dict[str, Any] = {
            "n_utterances": len(trues),
            "lipsync_2d_raw": evaluate_predictions(
                np.concatenate(trues), np.concatenate(ls_raws), r9_true_seqs=trues, r9_pred_seqs=ls_raws
            ),
            "lipsync_2d_viterbi": evaluate_predictions(
                np.concatenate(trues), np.concatenate(ls_cues), r9_true_seqs=trues, r9_pred_seqs=ls_cues
            ),
            "lipsync_2d_by_sex_macro_f1": {
                s: evaluate_predictions(np.concatenate(d["t"]), np.concatenate(d["ls"]))["rhubarb9_macro_f1"]
                for s, d in by_sex.items() if d["t"]
            },
        }
        if rh_preds:
            out["rhubarb_v1_14_phonetic"] = evaluate_predictions(
                np.concatenate(trues), np.concatenate(rh_preds), r9_true_seqs=trues, r9_pred_seqs=rh_preds
            )
            out["rhubarb_by_sex_macro_f1"] = {
                s: evaluate_predictions(np.concatenate(d["t"]), np.concatenate(d["rh"]))["rhubarb9_macro_f1"]
                for s, d in by_sex.items() if d["rh"]
            }
        return out

    report: dict[str, Any] = {
        "dataset": "kylelovesllms/timit_asr (official test split, human hand-annotated 61-phone timestamps)",
        "model": str(onnx_path.name),
        "n_unique_speakers": len(m_idx) + len(f_idx),
        "unique_speakers_by_sex": unique_speakers_by_sex,
        "normal_0st": eval_condition(0.0),
        "high_pitch_plus4st": eval_condition(4.0),
    }
    # Compute +4st relative drops
    for key in ("lipsync_2d_raw", "lipsync_2d_viterbi", "rhubarb_v1_14_phonetic"):
        if key in report["normal_0st"] and key in report["high_pitch_plus4st"]:
            f0 = report["normal_0st"][key]["rhubarb9_macro_f1"]
            f4 = report["high_pitch_plus4st"][key]["rhubarb9_macro_f1"]
            report[f"{key}_plus4st_rel_drop_pct"] = round((f0 - f4) / max(f0, 1e-8) * 100.0, 2)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark lipsync-2d vs Rhubarb CLI on human-labeled TIMIT.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--rhubarb", type=Path, default=None)
    parser.add_argument("--n-utts", type=int, default=90)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rep = benchmark_timit(args.onnx, args.rhubarb, args.n_utts, args.out)
    print(json.dumps({
        "normal_ls_raw": rep["normal_0st"]["lipsync_2d_raw"]["rhubarb9_macro_f1"],
        "normal_ls_vit": rep["normal_0st"]["lipsync_2d_viterbi"]["rhubarb9_macro_f1"],
        "normal_rhubarb": rep["normal_0st"].get("rhubarb_v1_14_phonetic", {}).get("rhubarb9_macro_f1"),
        "normal_ls_A": rep["normal_0st"]["lipsync_2d_raw"]["bilabial_A_f1"],
        "normal_rh_A": rep["normal_0st"].get("rhubarb_v1_14_phonetic", {}).get("bilabial_A_f1"),
        "normal_ls_G": rep["normal_0st"]["lipsync_2d_raw"]["labiodental_G_f1"],
        "normal_rh_G": rep["normal_0st"].get("rhubarb_v1_14_phonetic", {}).get("labiodental_G_f1"),
        "normal_ls_onset_ms": rep["normal_0st"]["lipsync_2d_viterbi"]["boundary_onset_mae_ms"],
        "normal_rh_onset_ms": rep["normal_0st"].get("rhubarb_v1_14_phonetic", {}).get("boundary_onset_mae_ms"),
        "hp4_ls_raw": rep["high_pitch_plus4st"]["lipsync_2d_raw"]["rhubarb9_macro_f1"],
        "hp4_rhubarb": rep["high_pitch_plus4st"].get("rhubarb_v1_14_phonetic", {}).get("rhubarb9_macro_f1"),
        "ls_drop_pct": rep.get("lipsync_2d_raw_plus4st_rel_drop_pct"),
        "rh_drop_pct": rep.get("rhubarb_v1_14_phonetic_plus4st_rel_drop_pct"),
        "ls_by_sex": rep["normal_0st"]["lipsync_2d_by_sex_macro_f1"],
        "rh_by_sex": rep["normal_0st"].get("rhubarb_by_sex_macro_f1"),
    }, indent=2))


if __name__ == "__main__":
    main()
