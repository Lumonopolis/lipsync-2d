"""Distill causal ConvNeXt-TCN + UniGRU lip-sync student from XLSR teacher shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from eval.metrics import benchmark_onnx_model, evaluate_predictions
from eval.mfcc_baseline import MFCCBaseline, log_mel_to_mfcc_50hz
from lipsync_2d.export_onnx import export_all_variants
from lipsync_2d.model import LipSyncStudent, ModelConfig, count_parameters
from lipsync_2d.visemes import OCULUS_VISEMES, RHUBARB_SHAPES


def pitch_warp_mel(mel: torch.Tensor, semitones: float) -> torch.Tensor:
    """Warp 80-bin Mel frequency axis to simulate pitch shift without changing frame timing."""
    if abs(semitones) < 1e-3:
        return mel
    scale = 2.0 ** (semitones / 12.0)
    n_mels = mel.shape[-1]
    bins = torch.arange(n_mels, device=mel.device, dtype=mel.dtype)
    src = torch.clamp(bins / scale, 0.0, float(n_mels - 1))
    lo = torch.floor(src).long()
    hi = torch.clamp(lo + 1, max=n_mels - 1)
    frac = (src - lo.to(mel.dtype)).view(*([1] * (mel.ndim - 1)), n_mels)
    return (1.0 - frac) * mel[..., lo] + frac * mel[..., hi]


def apply_spec_augment(
    mel: torch.Tensor,
    rng: np.random.Generator,
    max_freq_mask: int = 12,
    max_time_mask: int = 12,
) -> torch.Tensor:
    """Apply SpecAugment time and frequency masking on [T_10ms, 80] Mel tensor."""
    out = mel.clone()
    t_len, n_mels = out.shape
    if n_mels > max_freq_mask:
        f_w = int(rng.integers(0, max_freq_mask + 1))
        f_0 = int(rng.integers(0, n_mels - f_w + 1))
        out[:, f_0 : f_0 + f_w] = 0.0
    if t_len > max_time_mask:
        t_w = int(rng.integers(0, max_time_mask + 1))
        t_0 = int(rng.integers(0, t_len - t_w + 1))
        out[t_0 : t_0 + t_w, :] = 0.0
    return out


class UtteranceShardDataset(Dataset):
    """In-memory utterance dataset loaded from extracted teacher .npz shards."""

    def __init__(
        self,
        utterances: list[dict[str, Any]],
        augment: bool = False,
        seed: int = 42,
        silence_neg_prob: float = 0.1,
        max_frames_50hz: int = 256,
    ) -> None:
        self.utterances = utterances
        self.augment = augment
        self.silence_neg_prob = silence_neg_prob
        self.max_frames_50hz = max_frames_50hz
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.utterances[idx]
        mel_np = item["mel"].astype(np.float32)
        v15_np = item["v15_probs"].astype(np.float32)
        r9_np = item["r9_probs"].astype(np.float32)
        curves_np = item["curves"].astype(np.float32)

        t50 = r9_np.shape[0]
        if self.augment and self.max_frames_50hz > 0 and t50 > self.max_frames_50hz:
            start = int(self.rng.integers(0, t50 - self.max_frames_50hz + 1))
            end = start + self.max_frames_50hz
            mel_np = mel_np[start * 2 : end * 2]
            v15_np = v15_np[start:end]
            r9_np = r9_np[start:end]
            curves_np = curves_np[start:end]

        mel = torch.from_numpy(mel_np)
        v15 = torch.from_numpy(v15_np)
        r9 = torch.from_numpy(r9_np)
        curves = torch.from_numpy(curves_np)

        if self.augment:
            # Explicit silence / room-noise / breathing negative segment augmentation (matches fixed norm floor)
            if self.rng.random() < self.silence_neg_prob:
                floor_val = float(self.rng.uniform(-1.98, -1.30))
                mel = torch.randn_like(mel) * float(self.rng.uniform(0.01, 0.12)) + floor_val
                v15 = torch.zeros_like(v15)
                v15[:, 0] = 1.0
                r9 = torch.zeros_like(r9)
                r9[:, 0] = 1.0
                curves = torch.zeros_like(curves)
                curves[:, 2] = 1.0
            else:
                if self.rng.random() < 0.45:
                    semitones = float(self.rng.uniform(-4.5, 4.5))
                    mel = pitch_warp_mel(mel, semitones)
                if self.rng.random() < 0.60:
                    gain_shift = float(self.rng.uniform(-0.35, 0.35))
                    mel = mel + gain_shift
                if self.rng.random() < 0.35:
                    noise_std = float(self.rng.uniform(0.015, 0.14))
                    mel = mel + torch.randn_like(mel) * noise_std
                if self.rng.random() < 0.45:
                    mel = apply_spec_augment(mel, self.rng, max_freq_mask=8, max_time_mask=10)

        return {
            "mel": mel,
            "v15_probs": v15,
            "r9_probs": r9,
            "curves": curves,
            "lang": item["lang"],
        }


def collate_utterances(batch: list[dict[str, Any]]) -> dict[str, Any]:
    raw_max_t50 = max(b["r9_probs"].shape[0] for b in batch)
    max_t50 = 256 if raw_max_t50 <= 256 else ((raw_max_t50 + 63) // 64) * 64
    max_t10 = max_t50 * 2
    bsz = len(batch)

    mel_pad = torch.zeros(bsz, max_t10, 80, dtype=torch.float32)
    v15_pad = torch.zeros(bsz, max_t50, len(OCULUS_VISEMES), dtype=torch.float32)
    r9_pad = torch.zeros(bsz, max_t50, len(RHUBARB_SHAPES), dtype=torch.float32)
    curves_pad = torch.zeros(bsz, max_t50, 3, dtype=torch.float32)
    mask = torch.zeros(bsz, max_t50, dtype=torch.bool)

    for i, b in enumerate(batch):
        t50 = b["r9_probs"].shape[0]
        t10 = t50 * 2
        mel_pad[i, :t10] = b["mel"][:t10]
        v15_pad[i, :t50] = b["v15_probs"][:t50]
        r9_pad[i, :t50] = b["r9_probs"][:t50]
        curves_pad[i, :t50] = b["curves"][:t50]
        mask[i, :t50] = True

    return {
        "mel": mel_pad,
        "v15_probs": v15_pad,
        "r9_probs": r9_pad,
        "curves": curves_pad,
        "mask": mask,
        "langs": [b["lang"] for b in batch],
    }


def _canonical_split(raw_split: str) -> str:
    if raw_split in ("validation", "val"):
        return "val"
    if raw_split == "test":
        return "test"
    return "train"


def load_shards(
    shards_dirs: Path | list[Path],
    exclude_speaker_prefixes: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load shards preserving strictly speaker-disjoint official train/val/test partitions."""
    dirs = [shards_dirs] if isinstance(shards_dirs, Path) else list(shards_dirs)
    train_utts: list[dict[str, Any]] = []
    val_utts: list[dict[str, Any]] = []
    test_utts: list[dict[str, Any]] = []

    shard_files: list[Path] = []
    for d in dirs:
        shard_files.extend(sorted(d.glob("shard_*.npz")))

    for shard_file in shard_files:
        data = np.load(shard_file, allow_pickle=False)
        mels = data["mels"]
        v15 = data["v15_probs"]
        r9 = data["r9_probs"]
        curves = data["curves"]
        mel_lens = data["mel_lens"]
        lbl_lens = data["lbl_lens"]
        langs = data["langs"]
        splits = data["splits"]
        speakers = data["speakers"]

        m_off = 0
        l_off = 0
        for i in range(len(mel_lens)):
            ml = int(mel_lens[i])
            ll = int(lbl_lens[i])
            spk = str(speakers[i])
            if exclude_speaker_prefixes and spk.startswith(exclude_speaker_prefixes):
                m_off += ml
                l_off += ll
                continue
            assigned = _canonical_split(str(splits[i]))
            rec = {
                "mel": mels[m_off : m_off + ml],
                "v15_probs": v15[l_off : l_off + ll],
                "r9_probs": r9[l_off : l_off + ll],
                "curves": curves[l_off : l_off + ll],
                "lang": str(langs[i]),
                "speaker_id": spk,
            }
            m_off += ml
            l_off += ll
            if assigned == "train":
                train_utts.append(rec)
            elif assigned == "val":
                val_utts.append(rec)
            else:
                test_utts.append(rec)

    return train_utts, val_utts, test_utts


def _compute_class_weights(
    utterances: list[dict[str, Any]],
    key: str,
    n_classes: int,
    boost_indices: tuple[int, ...],
) -> torch.Tensor:
    counts = np.ones(n_classes, dtype=np.float64)
    for u in utterances:
        labels = np.argmax(u[key], axis=-1)
        counts += np.bincount(labels, minlength=n_classes)
    weights = np.power(counts.max() / counts, 0.35)
    weights = np.clip(weights, 1.0, 4.0)
    for idx in boost_indices:
        weights[idx] *= 1.35
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def _focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=0.01,
    )
    if gamma > 0.0:
        pt = torch.exp(-ce)
        ce = ((1.0 - pt) ** gamma) * ce
    mf = mask.to(ce.dtype)
    return (ce * mf).sum() / mf.sum().clamp_min(1.0)


def _kl_distill_loss(logits: torch.Tensor, teacher_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    t_p = torch.clamp(teacher_probs, min=1e-6)
    t_p = t_p / t_p.sum(dim=-1, keepdim=True)
    kl_per_frame = (t_p * (torch.log(t_p) - log_p)).sum(dim=-1)
    mf = mask.to(kl_per_frame.dtype)
    return (kl_per_frame * mf).sum() / mf.sum().clamp_min(1.0)


@torch.no_grad()
def _predict_dataset(
    model: LipSyncStudent,
    utterances: list[dict[str, Any]],
    device: torch.device,
    pitch_semitones: float = 0.0,
    batch_size: int = 64,
) -> dict[str, Any]:
    model.eval()
    use_amp = (device.type == "cuda" and torch.cuda.is_bf16_supported()) or device.type == "mps"
    amp_device = device.type if device.type in ("cuda", "mps") else "cpu"
    order = sorted(range(len(utterances)), key=lambda i: utterances[i]["r9_probs"].shape[0])
    v15_trues, v15_preds = [], []
    r9_trues, r9_preds = [], []
    r9_true_seqs, r9_pred_seqs = [], []
    per_lang: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}

    for start in range(0, len(order), batch_size):
        chunk_idx = order[start : start + batch_size]
        chunk = [utterances[i] for i in chunk_idx]
        raw_max_t50 = max(u["r9_probs"].shape[0] for u in chunk)
        max_t50 = ((raw_max_t50 + 127) // 128) * 128
        mel_pad = torch.zeros(len(chunk), max_t50 * 2, 80, dtype=torch.float32)
        for bi, u in enumerate(chunk):
            t10 = u["r9_probs"].shape[0] * 2
            mel_pad[bi, :t10] = torch.from_numpy(u["mel"][:t10].astype(np.float32))
        mel_pad = mel_pad.to(device, non_blocking=True)
        if abs(pitch_semitones) > 1e-3:
            mel_pad = pitch_warp_mel(mel_pad, pitch_semitones)
        with torch.amp.autocast(amp_device, dtype=torch.bfloat16, enabled=use_amp):
            v15_logits, r9_logits, _ = model(mel_pad)
        v15_batch = torch.argmax(v15_logits, dim=-1).cpu().numpy()
        r9_batch = torch.argmax(r9_logits, dim=-1).cpu().numpy()

        for bi, u in enumerate(chunk):
            v15_t = np.argmax(u["v15_probs"], axis=-1)
            r9_t = np.argmax(u["r9_probs"], axis=-1)
            n = min(len(r9_t), r9_batch.shape[1])
            v15_p = v15_batch[bi, :n]
            r9_p = r9_batch[bi, :n]

            v15_trues.append(v15_t[:n])
            v15_preds.append(v15_p)
            r9_trues.append(r9_t[:n])
            r9_preds.append(r9_p)
            r9_true_seqs.append(r9_t[:n])
            r9_pred_seqs.append(r9_p)

            lang = u["lang"]
            if lang not in per_lang:
                per_lang[lang] = ([], [])
            per_lang[lang][0].append(r9_t[:n])
            per_lang[lang][1].append(r9_p)

    return {
        "v15_true": np.concatenate(v15_trues),
        "v15_pred": np.concatenate(v15_preds),
        "r9_true": np.concatenate(r9_trues),
        "r9_pred": np.concatenate(r9_preds),
        "r9_true_seqs": r9_true_seqs,
        "r9_pred_seqs": r9_pred_seqs,
        "per_lang": per_lang,
    }


def train_and_evaluate(
    shards_dir: Path | list[Path],
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    hf_repo: str | None = None,
    hf_artifact_prefix: str = "artifacts_v3",
    balance_langs: bool = True,
    exclude_speaker_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Train student, run MFCC baseline comparison, export ONNX, and save eval_report.json."""
    first_dir = shards_dir[0] if isinstance(shards_dir, list) else shards_dir
    if hf_repo and not any(first_dir.glob("shard_*.npz")):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            allow_patterns="shards/*",
            local_dir=str(first_dir.parent),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_utts, val_utts, test_utts = load_shards(shards_dir, exclude_speaker_prefixes=exclude_speaker_prefixes)
    if not train_utts or not test_utts:
        raise RuntimeError(f"Insufficient utterances in {shards_dir}: train={len(train_utts)}, test={len(test_utts)}")

    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    lang_counts: dict[str, int] = {}
    for u in train_utts:
        lang_counts[u["lang"]] = lang_counts.get(u["lang"], 0) + 1
    print(f"[train] device={dev} train={len(train_utts)} val={len(val_utts)} test={len(test_utts)} langs={lang_counts}", flush=True)
    cfg = ModelConfig()
    model = LipSyncStudent(cfg).to(dev)

    w15 = _compute_class_weights(train_utts, "v15_probs", len(OCULUS_VISEMES), boost_indices=(1, 2)).to(dev)
    w9 = _compute_class_weights(train_utts, "r9_probs", len(RHUBARB_SHAPES), boost_indices=(1, 7, 8)).to(dev)

    train_ds = UtteranceShardDataset(train_utts, augment=True, silence_neg_prob=0.12)
    sampler = None
    if balance_langs and len(lang_counts) > 1:
        # Soft inverse-sqrt balancing lifts lower-resource languages (JA/FR/DE/ZH/KO) without starving EN.
        lang_w = {lg: (max(lang_counts.values()) / max(1, cnt)) ** 0.55 for lg, cnt in lang_counts.items()}
        sample_weights = torch.tensor([lang_w[u["lang"]] for u in train_utts], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_utts), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_utterances,
        num_workers=4 if dev.type == "cuda" else 0,
        pin_memory=(dev.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_bf16 = (dev.type == "cuda" and torch.cuda.is_bf16_supported()) or dev.type == "mps"
    use_fp16 = dev.type == "cuda" and not use_bf16
    use_amp = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_device = dev.type if dev.type in ("cuda", "mps") else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    best_val_f1 = -1.0
    best_ckpt = output_dir / "best_student.pt"

    import time

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            mel = batch["mel"].to(dev, non_blocking=True)
            v15_t = batch["v15_probs"].to(dev, non_blocking=True)
            r9_t = batch["r9_probs"].to(dev, non_blocking=True)
            curves_t = batch["curves"].to(dev, non_blocking=True)
            mask = batch["mask"].to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                v15_logits, r9_logits, curves_pred = model(mel)
                loss_kl = _kl_distill_loss(v15_logits.float(), v15_t, mask) + _kl_distill_loss(
                    r9_logits.float(), r9_t, mask
                )
                loss_ce = _focal_ce_loss(v15_logits.float(), torch.argmax(v15_t, dim=-1), mask, w15) + _focal_ce_loss(
                    r9_logits.float(), torch.argmax(r9_t, dim=-1), mask, w9
                )
                mf = mask.float()
                loss_curves = (
                    F.smooth_l1_loss(curves_pred.float(), curves_t, reduction="none").mean(dim=-1) * mf
                ).sum() / mf.sum().clamp_min(1.0)
                loss = loss_kl + 0.55 * loss_ce + 0.5 * loss_curves

            if use_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        run_val = epoch == 1 or epoch % 2 == 0 or epoch >= epochs - 4
        if run_val:
            eval_utts = val_utts if val_utts else test_utts
            val_out = _predict_dataset(model, eval_utts, dev)
            val_metrics = evaluate_predictions(val_out["r9_true"], val_out["r9_pred"])
            val_f1 = float(val_metrics["rhubarb9_macro_f1"])
            dt = time.monotonic() - t0
            print(
                f"Epoch {epoch:02d}/{epochs} ({dt:.1f}s) loss={total_loss/max(1,len(train_loader)):.4f} val_r9_f1={val_f1:.4f}",
                flush=True,
            )
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                torch.save({"config": vars(cfg), "model_state_dict": model.state_dict()}, best_ckpt)
                if hf_repo and epoch % 10 == 0:
                    from huggingface_hub import HfApi

                    HfApi().upload_file(
                        path_or_fileobj=str(best_ckpt),
                        path_in_repo=f"{hf_artifact_prefix}/best_student.pt",
                        repo_id=hf_repo,
                        repo_type="dataset",
                    )
        else:
            dt = time.monotonic() - t0
            print(
                f"Epoch {epoch:02d}/{epochs} ({dt:.1f}s) loss={total_loss/max(1,len(train_loader)):.4f}",
                flush=True,
            )

    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_out = _predict_dataset(model, test_utts, dev, pitch_semitones=0.0)
    hp_out = _predict_dataset(model, test_utts, dev, pitch_semitones=4.0)

    # Realistic pure-silence + low-level room-noise evaluation through exact frontend
    from lipsync_2d.features import extract_log_mel_np

    sil_rng = np.random.default_rng(1234)
    sil_mels = []
    for amp in (0.0, 1e-5, 5e-5, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3):
        wav_sil = (sil_rng.standard_normal(16000) * amp).astype(np.float32)
        sil_mels.append(extract_log_mel_np(wav_sil, norm="fixed", causal=True)[:100])
    sil_mel = torch.from_numpy(np.stack(sil_mels, axis=0)).to(dev)
    with torch.no_grad():
        _, sil_r9_logits, _ = model(sil_mel)
    sil_r9_pred = torch.argmax(sil_r9_logits, dim=-1).cpu().numpy().reshape(-1)
    sil_r9_true = np.zeros_like(sil_r9_pred)

    student_metrics = evaluate_predictions(
        r9_true=test_out["r9_true"],
        r9_pred=test_out["r9_pred"],
        v15_true=test_out["v15_true"],
        v15_pred=test_out["v15_pred"],
        r9_true_seqs=test_out["r9_true_seqs"],
        r9_pred_seqs=test_out["r9_pred_seqs"],
        r9_high_pitch_true=hp_out["r9_true"],
        r9_high_pitch_pred=hp_out["r9_pred"],
    )
    student_metrics["dedicated_silence_fp_pct"] = evaluate_predictions(sil_r9_true, sil_r9_pred)[
        "silence_fp_jaw_flap_pct"
    ]
    student_metrics["per_language_r9_macro_f1"] = {
        lang: evaluate_predictions(np.concatenate(ts), np.concatenate(ps))["rhubarb9_macro_f1"]
        for lang, (ts, ps) in test_out["per_lang"].items()
    }

    # Fit & evaluate uLipSync-style MFCC baseline on exact same train/test split (preserving utterance order)
    mfcc_train_x = np.concatenate(
        [log_mel_to_mfcc_50hz(u["mel"])[: len(u["r9_probs"])] for u in train_utts], axis=0
    )
    mfcc_train_y = np.concatenate([np.argmax(u["r9_probs"], axis=-1) for u in train_utts], axis=0)

    mfcc_clf = MFCCBaseline(n_classes=len(RHUBARB_SHAPES)).fit(mfcc_train_x, mfcc_train_y)
    mfcc_true_seqs = [np.argmax(u["r9_probs"], axis=-1) for u in test_utts]
    mfcc_pred_seqs = [mfcc_clf.predict(log_mel_to_mfcc_50hz(u["mel"]))[: len(u["r9_probs"])] for u in test_utts]
    mfcc_metrics = evaluate_predictions(
        r9_true=np.concatenate(mfcc_true_seqs, axis=0),
        r9_pred=np.concatenate(mfcc_pred_seqs, axis=0),
        r9_true_seqs=mfcc_true_seqs,
        r9_pred_seqs=mfcc_pred_seqs,
    )

    model_cpu = model.cpu().eval()
    onnx_paths = export_all_variants(model_cpu, output_dir / "onnx")
    benchmarks = {
        key: benchmark_onnx_model(path, streaming=("streaming" in key))
        for key, path in onnx_paths.items()
    }

    report = {
        "params": count_parameters(model_cpu),
        "split_counts": {"train": len(train_utts), "val": len(val_utts), "test": len(test_utts)},
        "student_metrics": student_metrics,
        "mfcc_baseline_metrics": mfcc_metrics,
        "onnx_benchmarks": benchmarks,
    }
    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=hf_repo,
            repo_type="dataset",
            path_in_repo=hf_artifact_prefix,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train causal lip-sync student model.")
    parser.add_argument("--shards-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-artifact-prefix", type=str, default="artifacts_v3")
    parser.add_argument("--no-balance-langs", action="store_true")
    parser.add_argument("--exclude-speaker-prefix", action="append", default=[])
    args = parser.parse_args()

    shards_dirs = args.shards_dir if args.shards_dir else [Path("data/shards")]
    train_and_evaluate(
        shards_dir=shards_dirs,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hf_repo=args.hf_repo,
        hf_artifact_prefix=args.hf_artifact_prefix,
        balance_langs=not args.no_balance_langs,
        exclude_speaker_prefixes=tuple(args.exclude_speaker_prefix),
    )


if __name__ == "__main__":
    main()
