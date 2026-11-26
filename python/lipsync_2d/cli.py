"""Rhubarb-compatible CLI entrypoint for lipsync-2d."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from lipsync_2d.decode import decode_cues
from lipsync_2d.exporters import (
    export_blender_py,
    export_godot_tres,
    export_live2d_json,
    export_rhubarb_json,
    export_rhubarb_tsv,
    export_rhubarb_xml,
)
from lipsync_2d.features import SAMPLE_RATE, compute_frame_energy_db, extract_log_mel_np, normalize_speech_rms

SUPPORTED_FORMATS: tuple[str, ...] = ("json", "tsv", "xml", "live2d", "godot", "blender")


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def load_wav_mono_16k(path: str | Path) -> tuple[np.ndarray, float]:
    """Load a WAV file as float32 mono 16 kHz waveform and return (audio, duration_s)."""
    sr, data = wavfile.read(str(path))
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        wav = arr.astype(np.float32) / scale
    else:
        wav = arr.astype(np.float32)

    duration = float(wav.shape[0]) / float(sr)
    if sr != SAMPLE_RATE:
        g = int(np.gcd(sr, SAMPLE_RATE))
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wav, duration


def run_inference(
    model_path: str | Path,
    mel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run offline ONNX or PyTorch checkpoint inference on [T_10ms, 80] Mel frames."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    mel_batch = mel[None, :, :].astype(np.float32)
    if path.suffix.lower() == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        v15_logits, r9_logits, curves = session.run(None, {"mel": mel_batch})
        return _softmax(v15_logits[0]), _softmax(r9_logits[0]), curves[0]

    if path.suffix.lower() in (".pt", ".pth"):
        import torch
        from lipsync_2d.model import LipSyncStudent, ModelConfig

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["config"]) if isinstance(ckpt, dict) and "config" in ckpt else ModelConfig()
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model = LipSyncStudent(cfg).eval()
        model.load_state_dict(state)
        with torch.no_grad():
            v15_logits, r9_logits, curves = model(torch.from_numpy(mel_batch))
        return (
            _softmax(v15_logits[0].numpy()),
            _softmax(r9_logits[0].numpy()),
            curves[0].numpy(),
        )

    raise ValueError(f"Unsupported model extension {path.suffix!r}; expected .onnx, .pt, or .pth")


def resolve_model_path(model_arg: Path | None, repo_id: str, variant: str = "fp16") -> Path:
    """Resolve explicit --model path, LIPSYNC_2D_MODEL env var, or auto-download from HF Hub."""
    import os

    if model_arg is not None:
        return model_arg
    env_path = os.environ.get("LIPSYNC_2D_MODEL")
    if env_path:
        return Path(env_path)
    filename = f"onnx/lipsync_student_offline_{variant}.onnx"
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        return Path(downloaded)
    except Exception as exc:
        raise FileNotFoundError(
            f"No --model specified and auto-download of {repo_id}/{filename} failed ({exc}). "
            "Pass --model path/to/lipsync_student_offline_fp16.onnx or set LIPSYNC_2D_MODEL."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    import os

    parser = argparse.ArgumentParser(
        prog="lipsync-2d",
        description="Real-time multilingual 2D lip-sync and viseme generator.",
    )
    parser.add_argument("input_wav", type=Path, help="Input WAV audio file")
    parser.add_argument(
        "-f",
        "--exportFormat",
        dest="export_format",
        choices=SUPPORTED_FORMATS,
        default="json",
        help="Output format (json, tsv, xml, live2d, godot, blender)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file path (default: stdout)")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=None,
        help="Path to .onnx or .pt student model (omit to auto-download from Hugging Face Hub)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.environ.get("LIPSYNC_2D_HF_REPO", "Lumonopolis/lipsync-2d"),
        help="Hugging Face model repo ID for auto-download when --model is omitted",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "int8", "fp32"),
        default="fp16",
        help="Model precision to auto-download when --model is omitted",
    )
    parser.add_argument("--switch-penalty", type=float, default=1.2, help="Viterbi transition switch penalty")
    parser.add_argument("--min-duration-ms", type=float, default=40.0, help="Minimum non-plosive cue duration (ms)")
    parser.add_argument("--fps", type=float, default=24.0, help="Target scene FPS for Blender export")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input_wav.exists():
        print(f"Error: input audio not found: {args.input_wav}", file=sys.stderr)
        return 2

    try:
        model_path = resolve_model_path(args.model, args.repo_id, args.precision)
        wav, duration = load_wav_mono_16k(args.input_wav)
        wav_norm = normalize_speech_rms(wav)
        mel = extract_log_mel_np(wav_norm, norm="fixed", causal=True)
        rms_db = compute_frame_energy_db(wav_norm, frame_ms=20)
        v15_probs, r9_probs, curves = run_inference(model_path, mel)
        n_frames = min(r9_probs.shape[0], rms_db.shape[0]) if rms_db.size > 0 else r9_probs.shape[0]
        rms_aligned = rms_db[:n_frames] if rms_db.size >= n_frames else None
        cues = decode_cues(
            r9_probs[:n_frames],
            switch_penalty=args.switch_penalty,
            min_duration_ms=args.min_duration_ms,
            audio_rms_db=rms_aligned,
        )

        fmt = args.export_format
        if fmt == "json":
            text = export_rhubarb_json(cues, args.input_wav.name, duration)
        elif fmt == "tsv":
            text = export_rhubarb_tsv(cues)
        elif fmt == "xml":
            text = export_rhubarb_xml(cues, args.input_wav.name, duration)
        elif fmt == "live2d":
            text = export_live2d_json(v15_probs[:n_frames], r9_probs[:n_frames], curves[:n_frames])
        elif fmt == "godot":
            text = export_godot_tres(cues, mouth_curves=curves[:n_frames])
        elif fmt == "blender":
            text = export_blender_py(
                cues,
                viseme15_probs=v15_probs[:n_frames],
                mouth_curves=curves[:n_frames],
                fps=args.fps,
            )
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
