"""Verify end-to-end Python <-> TypeScript Mel + ONNX inference parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from lipsync_2d.features import extract_log_mel_np, normalize_speech_rms


def load_wav_mono_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sw}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n_out = int(round(len(data) * 16000 / sr))
        x_old = np.linspace(0.0, 1.0, len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return data.astype(np.float32)


def run_parity_check(wav_path: Path, offline_onnx: Path, streaming_onnx: Path | None = None) -> dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    pkg_dir = root / "packages" / "lipsync-2d-web"
    pcm = load_wav_mono_16k(wav_path)
    py_pcm_norm = normalize_speech_rms(pcm)
    py_quiet_norm = normalize_speech_rms((pcm * 0.05).astype(np.float32))
    py_mel = extract_log_mel_np(py_pcm_norm, norm="fixed", causal=True)

    sess_off = ort.InferenceSession(str(offline_onnx), providers=["CPUExecutionProvider"])
    py_v15, py_r9, py_curves = sess_off.run(None, {"mel": py_mel[None, :, :].astype(np.float32)})

    py_stream_r9 = None
    if streaming_onnx is not None:
        sess_st = ort.InferenceSession(str(streaming_onnx), providers=["CPUExecutionProvider"])
        in_shapes = {i.name: i.shape for i in sess_st.get_inputs()}
        h_shape = tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in in_shapes["h_in"])
        c_shape = tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in in_shapes["conv_cache_in"])
        h = np.zeros(h_shape, dtype=np.float32)
        c = np.zeros(c_shape, dtype=np.float32)
        chunks = []
        t_even = (py_mel.shape[0] // 2) * 2
        for pos in range(0, t_even, 4):
            chunk = py_mel[pos : pos + 4]
            if chunk.shape[0] < 2:
                break
            _, r9_c, _, h, c = sess_st.run(
                None,
                {"mel_chunk": chunk[None, :, :].astype(np.float32), "h_in": h, "conv_cache_in": c},
            )
            chunks.append(r9_c[0])
        py_stream_r9 = np.concatenate(chunks, axis=0)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pcm_bin = td_path / "pcm.f32"
        pcm.tofile(pcm_bin)
        out_json = td_path / "ts_out.json"
        runner = td_path / "run_ts.mjs"
        runner.write_text(
            f"""
import fs from "node:fs";
import * as ort from "{(pkg_dir / 'node_modules' / 'onnxruntime-web' / 'dist' / 'ort.min.mjs').as_posix()}";
import {{ LogMelExtractor, normalizeSpeechRms, N_MELS, HIDDEN_STATE_SHAPE, HIDDEN_STATE_SIZE, CONV_CACHE_SHAPE, CONV_CACHE_SIZE }} from "{(pkg_dir / 'dist' / 'index.js').as_posix()}";

ort.env.wasm.numThreads = 1;
const raw = fs.readFileSync({json.dumps(str(pcm_bin))});
const pcm = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const pcmNorm = normalizeSpeechRms(pcm);
const quietPcm = new Float32Array(pcm.length);
for (let i = 0; i < pcm.length; i++) quietPcm[i] = pcm[i] * 0.05;
const quietNorm = normalizeSpeechRms(quietPcm);
const extractor = new LogMelExtractor();
const mel = extractor.extract(pcmNorm, {{ causal: true, standardize: "fixed" }});

const offBytes = fs.readFileSync({json.dumps(str(offline_onnx.resolve()))});
const offSess = await ort.InferenceSession.create(offBytes, {{ executionProviders: ["wasm"] }});
const offOut = await offSess.run({{ mel: new ort.Tensor("float32", mel.data, [1, mel.frames, N_MELS]) }});

let streamR9 = null;
const streamPath = {json.dumps(str(streaming_onnx.resolve()) if streaming_onnx else "")};
if (streamPath) {{
  const stBytes = fs.readFileSync(streamPath);
  const stSess = await ort.InferenceSession.create(stBytes, {{ executionProviders: ["wasm"] }});
  let h = new Float32Array(HIDDEN_STATE_SIZE);
  let c = new Float32Array(CONV_CACHE_SIZE);
  const tEven = Math.floor(mel.frames / 2) * 2;
  const parts = [];
  for (let pos = 0; pos < tEven; pos += 4) {{
    const len = Math.min(4, tEven - pos);
    if (len < 2) break;
    const sub = mel.data.subarray(pos * N_MELS, (pos + len) * N_MELS);
    const res = await stSess.run({{
      mel_chunk: new ort.Tensor("float32", sub, [1, len, N_MELS]),
      h_in: new ort.Tensor("float32", h, [...HIDDEN_STATE_SHAPE]),
      conv_cache_in: new ort.Tensor("float32", c, [...CONV_CACHE_SHAPE]),
    }});
    parts.push(...Array.from(res.rhubarb9_logits.data));
    h = new Float32Array(res.h_out.data);
    c = new Float32Array(res.conv_cache_out.data);
  }}
  streamR9 = parts;
}}

fs.writeFileSync({json.dumps(str(out_json))}, JSON.stringify({{
  quiet_norm: Array.from(quietNorm),
  mel_frames: mel.frames,
  mel_data: Array.from(mel.data),
  v15: Array.from(offOut.viseme15_logits.data),
  r9: Array.from(offOut.rhubarb9_logits.data),
  curves: Array.from(offOut.mouth_curves.data),
  stream_r9: streamR9,
}}));
""",
            encoding="utf-8",
        )
        subprocess.run(["node", str(runner)], check=True, cwd=str(pkg_dir))
        ts = json.loads(out_json.read_text(encoding="utf-8"))

    ts_quiet_norm = np.asarray(ts["quiet_norm"], dtype=np.float32)
    ts_mel = np.asarray(ts["mel_data"], dtype=np.float32).reshape(ts["mel_frames"], 80)
    ts_v15 = np.asarray(ts["v15"], dtype=np.float32).reshape(py_v15.shape)
    ts_r9 = np.asarray(ts["r9"], dtype=np.float32).reshape(py_r9.shape)
    ts_curves = np.asarray(ts["curves"], dtype=np.float32).reshape(py_curves.shape)

    metrics = {
        "max_abs_rms_norm_diff": float(np.max(np.abs(py_quiet_norm - ts_quiet_norm))),
        "max_abs_mel_diff": float(np.max(np.abs(py_mel - ts_mel))),
        "max_abs_v15_logit_diff": float(np.max(np.abs(py_v15 - ts_v15))),
        "max_abs_r9_logit_diff": float(np.max(np.abs(py_r9 - ts_r9))),
        "max_abs_curves_diff": float(np.max(np.abs(py_curves - ts_curves))),
        "r9_argmax_match_pct": float(
            100.0 * np.mean(np.argmax(py_r9[0], axis=-1) == np.argmax(ts_r9[0], axis=-1))
        ),
    }
    if py_stream_r9 is not None and ts["stream_r9"] is not None:
        ts_st = np.asarray(ts["stream_r9"], dtype=np.float32).reshape(py_stream_r9.shape)
        metrics["max_abs_stream_r9_diff"] = float(np.max(np.abs(py_stream_r9 - ts_st)))
        metrics["stream_r9_argmax_match_pct"] = float(
            100.0 * np.mean(np.argmax(py_stream_r9, axis=-1) == np.argmax(ts_st, axis=-1))
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Python <-> TypeScript ONNX inference parity.")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--offline-onnx", type=Path, required=True)
    parser.add_argument("--streaming-onnx", type=Path, default=None)
    parser.add_argument("--tol", type=float, default=1e-3, help="Max abs logit diff tolerance (default 1e-3 for FP32; use 5e-3 for FP16).")
    args = parser.parse_args()

    res = run_parity_check(args.wav, args.offline_onnx, args.streaming_onnx)
    print(json.dumps(res, indent=2))
    assert res["max_abs_rms_norm_diff"] < 1e-5, f"RMS norm parity failed: {res}"
    assert res["max_abs_mel_diff"] < 1e-4, f"Mel parity failed: {res}"
    assert res["max_abs_r9_logit_diff"] < args.tol, f"Offline r9 parity failed: {res}"
    assert res["max_abs_v15_logit_diff"] < args.tol, f"Offline v15 parity failed: {res}"
    assert res["max_abs_curves_diff"] < args.tol, f"Offline curves parity failed: {res}"
    assert res["r9_argmax_match_pct"] == 100.0, f"Offline argmax mismatch: {res}"
    if "max_abs_stream_r9_diff" in res:
        assert res["max_abs_stream_r9_diff"] < args.tol, f"Streaming r9 parity failed: {res}"
        assert res["stream_r9_argmax_match_pct"] == 100.0, f"Streaming argmax mismatch: {res}"


if __name__ == "__main__":
    main()
