"""Extract 50 Hz multilingual IPA teacher posteriors and 80-bin Log-Mel training shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from scipy.signal import resample_poly

from lipsync_2d.ctc_align import ctc_posteriors_to_spans
from lipsync_2d.features import SAMPLE_RATE, compute_frame_energy_db, extract_log_mel_np
from lipsync_2d.visemes import (
    OCULUS_VISEMES,
    RHUBARB_SHAPES,
    VISEME15_MOUTH_CURVES,
    build_vocab_projection_matrices,
)

TEACHER_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

FLEURS_LANGS: dict[str, str] = {
    "en_us": "en",
    "ja_jp": "ja",
    "es_419": "es",
    "fr_fr": "fr",
    "de_de": "de",
    "cmn_hans_cn": "zh",
    "ko_kr": "ko",
}


def _resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if orig_sr == SAMPLE_RATE:
        return wav
    g = int(np.gcd(orig_sr, SAMPLE_RATE))
    return resample_poly(wav, SAMPLE_RATE // g, orig_sr // g).astype(np.float32)


def _iter_hf_streams(
    max_per_lang: int,
    include_librispeech: bool,
    include_common_voice: bool,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> Iterator[dict[str, object]]:
    from datasets import load_dataset

    if include_librispeech:
        ls_split_map = {"train": "train.100", "validation": "validation", "test": "test"}
        for split_name in splits:
            ds = load_dataset(
                "openslr/librispeech_asr",
                "clean",
                split=ls_split_map[split_name],
                streaming=True,
            )
            count = 0
            for ex in ds:
                if count >= max_per_lang:
                    break
                audio = ex["audio"]
                spk = f"ls_{ex.get('speaker_id', count)}"
                yield {
                    "audio": _resample_to_16k(audio["array"], int(audio["sampling_rate"])),
                    "lang": "en",
                    "split": split_name,
                    "speaker_id": spk,
                }
                count += 1

    for fleurs_cfg, lang_code in FLEURS_LANGS.items():
        for split_name in splits:
            try:
                ds = load_dataset(
                    "google/fleurs",
                    fleurs_cfg,
                    split=split_name,
                    streaming=True,
                    revision="refs/convert/parquet",
                )
            except Exception:
                try:
                    ds = load_dataset(
                        "google/fleurs",
                        fleurs_cfg,
                        split=split_name,
                        streaming=True,
                        trust_remote_code=True,
                    )
                except Exception as exc:
                    print(f"[warn] skipping fleurs {fleurs_cfg}/{split_name}: {exc}")
                    continue
            count = 0
            limit = max_per_lang if split_name == "train" else max(100, max_per_lang // 5)
            for ex in ds:
                if count >= limit:
                    break
                audio = ex["audio"]
                raw_spk = ex.get("speaker_id") or ex.get("id") or count
                spk = f"fleurs_{lang_code}_{raw_spk}"
                yield {
                    "audio": _resample_to_16k(audio["array"], int(audio["sampling_rate"])),
                    "lang": lang_code,
                    "split": split_name,
                    "speaker_id": hashlib.sha1(str(spk).encode()).hexdigest()[:12],
                }
                count += 1

    if include_common_voice:
        cv_langs = ("en", "ja", "es", "fr", "de", "zh-CN")
        for cv_lang in cv_langs:
            lang_norm = "zh" if cv_lang == "zh-CN" else cv_lang
            for split_name in splits:
                cv_split = split_name
                try:
                    ds = load_dataset(
                        "fsicoli/common_voice_19_0", cv_lang, split=cv_split, streaming=True
                    )
                except Exception:
                    continue
                count = 0
                limit = max_per_lang if split_name == "train" else max(100, max_per_lang // 5)
                for ex in ds:
                    if count >= limit:
                        break
                    audio = ex["audio"]
                    spk = f"cv_{lang_norm}_{ex.get('client_id', count)}"
                    yield {
                        "audio": _resample_to_16k(audio["array"], int(audio["sampling_rate"])),
                        "lang": lang_norm,
                        "split": split_name,
                        "speaker_id": hashlib.sha1(str(spk).encode()).hexdigest()[:12],
                    }
                    count += 1


def _save_shard(
    out_dir: Path,
    shard_idx: int,
    records: list[dict[str, object]],
) -> Path:
    mels = np.concatenate([r["mel"] for r in records], axis=0).astype(np.float16)
    v15_probs = np.concatenate([r["v15_probs"] for r in records], axis=0).astype(np.float16)
    r9_probs = np.concatenate([r["r9_probs"] for r in records], axis=0).astype(np.float16)
    curves = np.concatenate([r["curves"] for r in records], axis=0).astype(np.float16)

    mel_lens = np.array([r["mel"].shape[0] for r in records], dtype=np.int32)
    lbl_lens = np.array([r["r9_probs"].shape[0] for r in records], dtype=np.int32)
    langs = np.array([r["lang"] for r in records])
    splits = np.array([r["split"] for r in records])
    speakers = np.array([r["speaker_id"] for r in records])

    shard_path = out_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(
        shard_path,
        mels=mels,
        v15_probs=v15_probs,
        r9_probs=r9_probs,
        curves=curves,
        mel_lens=mel_lens,
        lbl_lens=lbl_lens,
        langs=langs,
        splits=splits,
        speakers=speakers,
    )
    return shard_path


def extract_teacher_shards(
    output_dir: Path,
    max_per_lang: int = 2000,
    shard_size: int = 256,
    include_librispeech: bool = True,
    include_common_voice: bool = False,
    device: str | None = None,
    hf_repo: str | None = None,
) -> list[Path]:
    """Run XLSR-53-espeak teacher inference and write compressed feature+target shards."""
    from tqdm import tqdm
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    feature_extractor = AutoFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    teacher = AutoModelForCTC.from_pretrained(TEACHER_MODEL_ID).to(device=dev, dtype=dtype).eval()

    m_v15, m_r9 = build_vocab_projection_matrices()

    records: list[dict[str, object]] = []
    shard_paths: list[Path] = []
    shard_idx = 0

    for item in tqdm(_iter_hf_streams(max_per_lang, include_librispeech, include_common_voice)):
        wav: np.ndarray = item["audio"]  # type: ignore[assignment]
        if wav.size < SAMPLE_RATE // 4:
            continue
        mel = extract_log_mel_np(wav, norm="fixed", causal=True)
        n_50hz = mel.shape[0] // 2
        if n_50hz < 4:
            continue
        mel = mel[: n_50hz * 2]

        inputs = feature_extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device=dev, dtype=dtype)
        with torch.no_grad():
            logits = teacher(input_values).logits[0].float().cpu()
            p_ipa = torch.softmax(logits, dim=-1).numpy()

        # Align teacher CTC length to exact 50 Hz mel frame count (T_50hz = mel_frames // 2).
        if p_ipa.shape[0] < n_50hz:
            pad_n = n_50hz - p_ipa.shape[0]
            pad_rows = np.repeat(p_ipa[-1:], pad_n, axis=0)
            p_ipa = np.concatenate([p_ipa, pad_rows], axis=0)
        elif p_ipa.shape[0] > n_50hz:
            p_ipa = p_ipa[:n_50hz]

        energy_db = compute_frame_energy_db(wav, frame_ms=20)
        if energy_db.shape[0] < n_50hz:
            energy_db = np.pad(energy_db, (0, n_50hz - energy_db.shape[0]), mode="edge")
        else:
            energy_db = energy_db[:n_50hz]

        clean_ipa, is_speech = ctc_posteriors_to_spans(p_ipa, energy_db)
        v15_probs = clean_ipa @ m_v15
        r9_probs = clean_ipa @ m_r9

        # Enforce hard silence probability on non-speech frames.
        sil_mask = ~is_speech
        if np.any(sil_mask):
            v15_probs[sil_mask] = 0.0
            v15_probs[sil_mask, 0] = 1.0
            r9_probs[sil_mask] = 0.0
            r9_probs[sil_mask, 0] = 1.0

        v15_probs /= np.maximum(v15_probs.sum(axis=-1, keepdims=True), 1e-6)
        r9_probs /= np.maximum(r9_probs.sum(axis=-1, keepdims=True), 1e-6)
        curves = v15_probs @ VISEME15_MOUTH_CURVES

        records.append(
            {
                "mel": mel,
                "v15_probs": v15_probs,
                "r9_probs": r9_probs,
                "curves": curves,
                "lang": item["lang"],
                "split": item["split"],
                "speaker_id": item["speaker_id"],
            }
        )
        if len(records) >= shard_size:
            shard_paths.append(_save_shard(output_dir, shard_idx, records))
            shard_idx += 1
            records.clear()

    if records:
        shard_paths.append(_save_shard(output_dir, shard_idx, records))

    meta = {
        "teacher_model": TEACHER_MODEL_ID,
        "oculus_visemes": list(OCULUS_VISEMES),
        "rhubarb_shapes": list(RHUBARB_SHAPES),
        "num_shards": len(shard_paths),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=hf_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=hf_repo,
            repo_type="dataset",
            path_in_repo="shards",
        )

    return shard_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract XLSR-53-espeak teacher shards.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/shards"))
    parser.add_argument("--max-per-lang", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--no-librispeech", action="store_true")
    parser.add_argument("--include-common-voice", action="store_true")
    parser.add_argument("--hf-repo", type=str, default=None)
    args = parser.parse_args()

    extract_teacher_shards(
        output_dir=args.output_dir,
        max_per_lang=args.max_per_lang,
        shard_size=args.shard_size,
        include_librispeech=not args.no_librispeech,
        include_common_voice=args.include_common_voice,
        hf_repo=args.hf_repo,
    )


if __name__ == "__main__":
    main()
