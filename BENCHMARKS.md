# lipsync-2d Benchmarks, Architecture & Verification

Full evaluation reports, architecture specs, latency/parity measurements, and DCC import verification for `lipsync-2d`.

## 1. Head-to-Head on Human-Annotated TIMIT Phonemes (Zero Teacher Overlap)

Evaluated zero-shot across `90` distinct balanced speakers (`45` Male / `45` Female, 1 utterance per speaker, covering every female speaker in the official TIMIT test split `kylelovesllms/timit_asr`, human hand-annotated 61-phone timestamps mapped to the 9 Rhubarb shapes `X, A–H`), comparing `lipsync-2d` (`6.55 MB` FP16 ONNX) directly against **Official Rhubarb Lip Sync `v1.14.0`** (`-r phonetic --extendedShapes GHX`, see [eval/bench_timit_rhubarb.py](eval/bench_timit_rhubarb.py) and [eval/results_timit_rhubarb.json](eval/results_timit_rhubarb.json)):

| Metric (Human-Annotated TIMIT Test, 90 distinct speakers) | `lipsync-2d` (Raw / Viterbi) | Official `Rhubarb v1.14.0` | Delta (Raw / Viterbi) |
| --- | ---: | ---: | ---: |
| **9-Rhubarb Macro-F1** ↑ | **0.615 / 0.616** | 0.432 | **+18.3 / +18.4 pts (1.42×)** |
| **9-Rhubarb Weighted-F1** ↑ | **0.666 / 0.663** | 0.493 | **+17.3 / +17.0 pts** |
| **Bilabial `A` (`p, b, m`) F1** ↑ | **0.557 / 0.555** | 0.323 | **+23.4 / +23.2 pts (1.73×)** |
| **Labiodental `G` (`f, v`) F1** ↑ | 0.565 / 0.568 | **0.583** | -1.8 / -1.5 pts |
| **Lateral `H` (`l`) F1** ↑ | **0.589 / 0.615** | 0.283 | **+30.6 / +33.2 pts (2.08×)** |
| **Boundary Onset MAE** ↓ | **19.07 ms / 25.22 ms** | 44.55 ms | **-57.2% / -43.4%** |
| **Silence FP (Jaw-Flap)** ↓ | **5.36% / 5.70%** | 32.53% | **-83.5% / -82.5%** |
| **Female / Male Macro-F1 (0 st)** ↑ | **0.611 (F) / 0.619 (M)** | 0.427 (F) / 0.438 (M) | **+18.5 pts (F) / +18.1 pts (M)** |
| **High-Pitch `+4` Semitones Macro-F1 (rel. drop)** ↑ | **0.496 (`-19.36%`)** | 0.344 (`-20.40%`) | **+15.2 pts** |
| **High-Pitch `+4` Semitones Female Macro-F1** ↑ | **0.466** | 0.277 | **+18.9 pts (1.68×)** |

## 2. Speaker-Disjoint 7-Language Official Test Set (3,600 Utterances)

Trained exclusively on `27,427` strictly `train`-split utterances from 100% commercially permissive corpora (LibriSpeech `train-clean-100` CC-BY-4.0, Google FLEURS 7 languages CC-BY-4.0, Mozilla Common Voice 19 `ja/fr/de/es/zh-CN` CC0-1.0, Zeroth-Korean CC-BY-4.0) and evaluated on `3,600` strictly speaker-disjoint official test utterances (`1,500` LibriSpeech `test-clean` + `2,100` Google FLEURS `test`, aligned via `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, see [eval/results.json](eval/results.json)):

| Metric | `lipsync-2d` (3.41M Pure ConvNeXt-TCN) | uLipSync-style MFCC Baseline | Delta |
| --- | ---: | ---: | ---: |
| **9-Rhubarb Macro-F1** ↑ | **0.736** | 0.271 | **+46.6 pts (2.72×)** |
| **9-Rhubarb Weighted-F1** ↑ | **0.859** | 0.542 | **+31.7 pts** |
| **15-Oculus Macro-F1 / Weighted-F1** ↑ | **0.718 / 0.829** | — | — |
| **Bilabial `A` (`p, b, m`) F1** ↑ | **0.733** | 0.131 | **+60.2 pts (5.60×)** |
| **Labiodental `G` (`f, v`) F1** ↑ | **0.656** | 0.054 | **+60.2 pts (12.09×)** |
| **Lateral `H` (`l`) F1** ↑ | **0.685** | 0.064 | **+62.1 pts (10.72×)** |
| **Boundary Onset MAE** ↓ | **11.49 ms** | 21.63 ms | **-46.9%** |
| **High-Pitch (`+4` semitones) Relative Drop** ↓ | **4.59%** | — | `<5%` target |
| **Silence FP (overall / dedicated room-noise)** ↓ | **1.93% / 0.00%** | 8.25% / — | **-76.6%** |

Per-language 9-Rhubarb Macro-F1 (official speaker-disjoint `test` splits):

| Language | `es` | `en` | `fr` | `de` | `zh` | `ko` | `ja` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Macro-F1** | **0.791** | **0.756** | **0.702** | **0.683** | **0.672** | **0.668** | **0.611** |

## 3. ONNX Footprint, Latency & Python ↔ TypeScript Parity

| Variant | Size | CPU Latency (ms / 20 ms frame) | Py ↔ TS Argmax Match (Max Logit Diff) |
| --- | ---: | ---: | ---: |
| `offline_fp32` | 13.05 MB | 0.164 ms | 100.0% (`5.25e-6`) |
| `streaming_fp32` | 13.03 MB | 2.318 ms | 100.0% (`5.60e-6`) |
| **`offline_fp16`** | **6.55 MB** | **0.121 ms** | **100.0%** (`5.13e-3` FP16 ULP) |
| **`streaming_fp16`** | **6.54 MB** | **1.851 ms** | **100.0%** (`7.03e-3` FP16 ULP) |
| `offline_int8` | 3.57 MB | 0.088 ms | — |
| `streaming_int8` | 3.55 MB | 1.099 ms | — |

- **48 kHz WASM Streaming Latency (`eval/bench_web_streaming.mjs`):** `1.21–1.82 ms` mean / `1.53–2.70 ms` p95 per 20 ms frame (**`11–16.6×` real-time**) on continuous 128-sample AudioWorklet chunks using `streaming_fp16`.
- **Frontend Parity (`features.py` ↔ `mel.ts`):** Clip-level speech-gated RMS normalization **`2.98e-8`** max abs diff; 80-bin causal Slaney Log-Mel **`4.53e-6`** max abs diff.

## 4. Architecture & DCC Verification

- **Frontend (`python/lipsync_2d/features.py` & `packages/lipsync-2d-web/src/mel.ts`):** Clip-level speech-gated RMS normalization (`-18 dBFS` target above `-46 dBFS` gate, `+20 dB` max cap) → causal 16 kHz mono 80-bin Slaney Log-Mel spectrogram (25 ms periodic Hann window, 10 ms hop, 512 FFT, 64–7800 Hz, fixed scaling `(logMel + 6.0) / 4.0`).
- **Student (`python/lipsync_2d/model.py`):** 10-block Pure Causal ConvNeXt-TCN (`3.41M` params, `channels=288`, `kernel_size=5`, `dilations=(1,2,4,8,16,1,2,4,8,16)`, 5.0 s receptive field, `40 ms` bounded lookahead, zero GRU ops). Outputs 50 Hz predictions for 15 Oculus visemes, 9 Rhubarb shapes (`X, A–H`), and 3 continuous mouth curves (`jawOpen`, `lipRound`, `lipClose`).
- **Real DCC Headless Import Verification (`eval/verify_dcc_imports.py`):** Both Python and TypeScript Blender `.py` exporters (including Blender 5.x Slotted Actions) and Godot 4 `.tres` exporters pass headless import execution in **Blender 5.2.1 LTS** and **Godot 4.7.2**.

## Reproducing All Checks

```bash
pytest -q
cd packages/lipsync-2d-web && npm test && cd ../..
PYTHONPATH=.:python python eval/verify_py_ts_parity.py \
  --wav web/public/samples/hello_studio.wav \
  --offline-onnx path/to/lipsync_student_offline_fp32.onnx \
  --streaming-onnx path/to/lipsync_student_streaming_fp32.onnx
PYTHONPATH=.:python python eval/verify_dcc_imports.py
node eval/bench_web_streaming.mjs
```
