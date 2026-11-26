import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import * as ort from "../packages/lipsync-2d-web/node_modules/onnxruntime-web/dist/ort.min.mjs";
import {
  CONV_CACHE_SHAPE,
  CONV_CACHE_SIZE,
  HIDDEN_STATE_SHAPE,
  HIDDEN_STATE_SIZE,
  N_MELS,
  SAMPLE_RATE,
  StreamingMelExtractor,
  StreamingResampler,
} from "../packages/lipsync-2d-web/dist/index.js";

const modelPath = process.argv[2] || "web/public/models/lipsync2d_stream.onnx";
ort.env.wasm.numThreads = 1;
const bytes = fs.readFileSync(modelPath);
const sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });

// Synthesize 4.0s of 48 kHz AudioWorklet 128-sample quanta (1500 blocks)
const sr = 48000;
const totalSamples = sr * 4;
const audio48k = new Float32Array(totalSamples);
for (let i = 0; i < totalSamples; i++) {
  const t = i / sr;
  audio48k[i] = 0.25 * Math.sin(2 * Math.PI * 180 * t) + 0.15 * Math.sin(2 * Math.PI * 620 * t);
}

const resampler = new StreamingResampler(sr, SAMPLE_RATE);
const melExt = new StreamingMelExtractor("fixed");
let h = new Float32Array(HIDDEN_STATE_SIZE);
let c = new Float32Array(CONV_CACHE_SIZE);
let pending = new Float32Array(0);
let stepTimesMs = [];
let totalFramesOut = 0;

for (let pos = 0; pos < totalSamples; pos += 128) {
  const block = audio48k.subarray(pos, Math.min(pos + 128, totalSamples));
  const pcm16k = resampler.push(block);
  const m = melExt.push(pcm16k);
  if (m.frames > 0) {
    const merged = new Float32Array(pending.length + m.data.length);
    merged.set(pending, 0);
    merged.set(m.data, pending.length);
    const totalMelFrames = Math.floor(merged.length / N_MELS);
    const runFrames = totalMelFrames - (totalMelFrames % 2);
    if (runFrames >= 2) {
      const batch = merged.subarray(0, runFrames * N_MELS);
      const t0 = performance.now();
      const out = await sess.run({
        mel_chunk: new ort.Tensor("float32", batch, [1, runFrames, N_MELS]),
        h_in: new ort.Tensor("float32", h, [...HIDDEN_STATE_SHAPE]),
        conv_cache_in: new ort.Tensor("float32", c, [...CONV_CACHE_SHAPE]),
      });
      const dt = performance.now() - t0;
      const nOut = Number(out.rhubarb9_logits.dims[1]);
      stepTimesMs.push(dt / Math.max(1, nOut));
      totalFramesOut += nOut;
      h = new Float32Array(out.h_out.data);
      c = new Float32Array(out.conv_cache_out.data);
      pending = merged.slice(runFrames * N_MELS);
    } else {
      pending = merged;
    }
  }
}

// Drop first 5 warmup steps
const steady = stepTimesMs.slice(5).sort((a, b) => a - b);
const meanMs = steady.reduce((a, b) => a + b, 0) / Math.max(1, steady.length);
const p95Ms = steady[Math.floor(steady.length * 0.95)] ?? 0;
console.log(JSON.stringify({
  model: path.basename(modelPath),
  backend: "wasm",
  inputRateHz: sr,
  quantumSamples: 128,
  totalFrames50Hz: totalFramesOut,
  meanMsPer20msFrame: Number(meanMs.toFixed(3)),
  p95MsPer20msFrame: Number(p95Ms.toFixed(3)),
  realtimeFactor: Number((20.0 / Math.max(meanMs, 1e-6)).toFixed(1)),
}, null, 2));
