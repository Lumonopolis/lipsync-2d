"""End-to-end integration tests for the lipsync-2d CLI."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from lipsync_2d.cli import main
from lipsync_2d.export_onnx import export_offline_onnx
from lipsync_2d.model import LipSyncStudent


def _write_test_wav(path: Path, sr: int = 22050, duration_s: float = 0.5) -> None:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    sig = 0.3 * np.sin(2.0 * np.pi * 220.0 * t)
    pcm = (sig * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, pcm)


def test_cli_all_formats_and_error_paths(tmp_path: Path) -> None:
    wav_path = tmp_path / "speech.wav"
    _write_test_wav(wav_path)
    onnx_path = export_offline_onnx(LipSyncStudent().eval(), tmp_path / "student.onnx")

    outputs = {
        "json": tmp_path / "out.json",
        "tsv": tmp_path / "out.tsv",
        "xml": tmp_path / "out.xml",
        "live2d": tmp_path / "out_live2d.json",
        "godot": tmp_path / "out.tres",
        "blender": tmp_path / "out.py",
    }
    for fmt, out_file in outputs.items():
        rc = main([str(wav_path), "-f", fmt, "-o", str(out_file), "--model", str(onnx_path)])
        assert rc == 0
        assert out_file.exists() and out_file.stat().st_size > 0

    assert "mouthCues" in json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert ET.fromstring(outputs["xml"].read_text(encoding="utf-8")).tag == "rhubarbResult"
    l2d = json.loads(outputs["live2d"].read_text(encoding="utf-8"))
    assert "parameters" in l2d and "visemes" in l2d
    assert '[gd_resource type="Animation" format=3]' in outputs["godot"].read_text(encoding="utf-8")
    compile(outputs["blender"].read_text(encoding="utf-8"), "cli_blender.py", "exec")

    # Failure paths: missing audio and missing model
    assert main([str(tmp_path / "missing.wav"), "--model", str(onnx_path)]) != 0
    assert main([str(wav_path), "--model", str(tmp_path / "missing.onnx")]) != 0


def test_cli_uses_fixed_norm_and_suppresses_room_noise(tmp_path: Path) -> None:
    from lipsync_2d.cli import decode_cues, load_wav_mono_16k, run_inference
    from lipsync_2d.features import compute_frame_energy_db, extract_log_mel_np, normalize_speech_rms

    wav_path = tmp_path / "speech.wav"
    _write_test_wav(wav_path)
    onnx_path = export_offline_onnx(LipSyncStudent().eval(), tmp_path / "student.onnx")
    out_json = tmp_path / "cli_out.json"
    assert main([str(wav_path), "-f", "json", "-o", str(out_json), "--model", str(onnx_path)]) == 0
    cli_cues = json.loads(out_json.read_text(encoding="utf-8"))["mouthCues"]

    wav, _ = load_wav_mono_16k(wav_path)
    wav_norm = normalize_speech_rms(wav)
    mel_fixed = extract_log_mel_np(wav_norm, norm="fixed", causal=True)
    rms_db = compute_frame_energy_db(wav_norm, frame_ms=20)
    _, r9_probs, _ = run_inference(onnx_path, mel_fixed)
    n_frames = min(r9_probs.shape[0], rms_db.shape[0])
    expected = decode_cues(r9_probs[:n_frames], audio_rms_db=rms_db[:n_frames])
    assert [c["value"] for c in cli_cues] == [c.value for c in expected]

    # If trained FP16 model is staged locally, verify -50.5 dBFS room noise yields X-only (no jaw-flap)
    staged_fp16 = Path(__file__).resolve().parents[2] / "web" / "public" / "models" / "lipsync2d.onnx"
    if staged_fp16.exists():
        rng = np.random.default_rng(1234)
        noise = (rng.standard_normal(16000) * 3e-3 * 32767.0).astype(np.int16)
        noise_wav = tmp_path / "room_noise.wav"
        wavfile.write(str(noise_wav), 16000, noise)
        noise_json = tmp_path / "noise_out.json"
        assert main([str(noise_wav), "-f", "json", "-o", str(noise_json), "--model", str(staged_fp16)]) == 0
        cues = json.loads(noise_json.read_text(encoding="utf-8"))["mouthCues"]
        assert [c["value"] for c in cues] == ["X"], f"Room noise jaw-flapped: {cues}"
