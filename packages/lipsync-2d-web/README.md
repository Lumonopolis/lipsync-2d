# lipsync-2d-web

Real-time 2D lip-sync inference (`onnxruntime-web` WASM/WebGPU), causal 80-bin Log-Mel frontend, Viterbi cue decoder, and Rhubarb / Live2D / Godot 4 / Blender 5 exporters.

- **Live Browser Studio (HF Space):** https://huggingface.co/spaces/Lumonopolis/lipsync-2d-studio
- **ONNX Weights (Apache-2.0):** https://huggingface.co/Lumonopolis/lipsync-2d
- **Repository & Benchmarks:** https://github.com/Lumonopolis/lipsync-2d

## Usage

```ts
import { LipSyncEngine, toRhubarbJson, toGodotTres, toLive2DJson } from "lipsync-2d-web";

const engine = new LipSyncEngine({ wasmPaths: "/ort/" });
await engine.loadModel(
  "https://huggingface.co/Lumonopolis/lipsync-2d/resolve/main/onnx/lipsync_student_offline_fp16.onnx",
  "https://huggingface.co/Lumonopolis/lipsync-2d/resolve/main/onnx/lipsync_student_streaming_fp16.onnx",
);

const result = await engine.analyzeAudio(audioBuffer);
const rhubarbJson = toRhubarbJson(result.rhubarbCues, result.duration, "voice.wav");
const godotTres = toGodotTres(result.rhubarbCues, result.duration);
const live2dJson = toLive2DJson(result.frames, result.duration);
```

Runs a `6.55 MB` FP16 ONNX model across 7 languages (`en, es, fr, de, zh, ko, ja`) at `~1.5 ms`/frame (`11–16×` real-time) in browser WASM.
