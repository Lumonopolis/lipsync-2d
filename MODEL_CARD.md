---
license: apache-2.0
language:
  - en
  - es
  - fr
  - de
  - ko
  - zh
  - ja
tags:
  - audio
  - lip-sync
  - viseme
  - rhubarb
  - onnx
  - webgpu
  - 2d-animation
---

# lipsync-2d

Real-time causal streaming 2D lip-sync and viseme prediction model (`3.41M` parameters, `6.55 MB` FP16 / `3.57 MB` INT8 ONNX, Apache-2.0).

## Model Description

- **Architecture**: 10-block Pure Causal ConvNeXt-TCN (`channels=288`, `kernel_size=5`, `dilations=(1,2,4,8,16,1,2,4,8,16)`, 5.0 s receptive field, `lookahead_frames=2` = 40 ms bounded lookahead, zero recurrent/GRU ops for WebGPU/WASM speed).
- **Input**: 16 kHz mono audio → speech-gated RMS normalization (`-18 dBFS` target above `-46 dBFS` gate, `+20 dB` max cap) → 80-bin causal Slaney Log-Mel spectrogram (25 ms periodic Hann window, 10 ms hop, 512 FFT, 64–7800 Hz, fixed scaling `(logMel + 6.0) / 4.0`).
- **Outputs (50 Hz / 20 ms step)**:
  - `viseme15_logits`: 15 canonical Oculus visemes (`sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, aa, E, ih, oh, ou`)
  - `rhubarb9_logits`: 9 Rhubarb mouth shapes (`X, A, B, C, D, E, F, G, H`) with lateral `/l/` mapped directly to `H`
  - `mouth_curves`: 3 continuous sigmoid curves in `[0, 1]` (`jawOpen, lipRound, lipClose`)

## Training Data & Distillation

Trained exclusively on official `train` partitions (`27,427` train / `3,481` validation utterances, inverse-sqrt language-balanced sampling) and evaluated on strictly speaker-disjoint official `test` partitions (`3,600` utterances: `1,500` LibriSpeech `test-clean` + `2,100` FLEURS `test`) from 100% commercially permissive (CC0-1.0 / CC-BY-4.0) corpora:
- LibriSpeech `train-clean-100` / `dev-clean` / `test-clean` (CC-BY-4.0)
- Google FLEURS (`en_us, es_419, cmn_hans_cn, ko_kr, fr_fr, de_de, ja_jp`, CC-BY-4.0)
- Mozilla Common Voice 19 (`ja, fr, de, es, zh-CN`, CC0-1.0)
- Zeroth-Korean (`ko`, CC-BY-4.0)

Teacher supervision was extracted using `facebook/wav2vec2-xlsr-53-espeak-cv-ft` with 100% audited 392-token IPA → 15-viseme / 9-Rhubarb mapping, energy-VAD midpoint-span CTC blank expansion, `±4.5` semitone Mel pitch-warping, gain/noise/SpecAugment perturbations, and 12% synthetic silence/room-noise negative segments.

## Evaluation Results

### 1. Human-Annotated TIMIT Phonemes (Zero-Shot vs Official Rhubarb v1.14.0)

`90` distinct balanced Male/Female speakers (`45` Male / `45` Female, 1 utterance per speaker) from `kylelovesllms/timit_asr` official `test` split (hand-annotated 61-phone timestamps, zero training or teacher overlap):
- **9-Rhubarb Macro-F1 (raw / Viterbi)**: **`0.615 / 0.616`** vs Official `Rhubarb v1.14.0` (`-r phonetic`): `0.432` (**`+18.3 / +18.4 pts`**)
- **Bilabial `A` (`p/b/m`) F1**: **`0.557 / 0.555`** vs Rhubarb `0.323`
- **Labiodental `G` (`f/v`) F1**: `0.565 / 0.568` vs Rhubarb `0.583`
- **Lateral `H` (`l`) F1**: **`0.589 / 0.615`** vs Rhubarb `0.283`
- **Boundary Onset MAE**: **`19.07 ms / 25.22 ms`** vs Rhubarb `44.55 ms`
- **Silence FP (jaw-flap)**: **`5.36% / 5.70%`** vs Rhubarb `32.53%`
- **Female / Male Macro-F1 (0 st)**: **`0.611 (F) / 0.619 (M)`** vs Rhubarb `0.427 (F) / 0.438 (M)`
- **High-Pitch `+4` Semitones Macro-F1**: **`0.496`** (`-19.36%` rel. drop; Female `0.466`) vs Rhubarb `0.344` (`-20.40%` rel. drop; Female `0.277`)

### 2. Speaker-Disjoint 7-Language Official Test Split (3,600 Utterances)

- **9-Rhubarb Macro-F1**: **`0.736`** (Weighted-F1 **`0.859`**) vs uLipSync-style MFCC baseline `0.271` (Weighted-F1 `0.542`)
- **15-Oculus Macro-F1**: **`0.718`** (Weighted-F1 **`0.829`**)
- **Bilabial `A` (`p/b/m`) F1**: **`0.733`** (vs MFCC `0.131`)
- **Labiodental `G` (`f/v`) F1**: **`0.656`** (vs MFCC `0.054`)
- **Lateral `H` (`l`) F1**: **`0.685`** (vs MFCC `0.064`)
- **Boundary Onset MAE**: **`11.49 ms`** (vs MFCC `21.63 ms`)
- **High-Pitch (`+4` semitones) Relative Macro-F1 Drop**: **`4.59%`**
- **Dedicated Silence / Room-Noise False Positive Rate**: **`0.00%`** (overall test silence FP **`1.93%`** vs MFCC `8.25%`)
- **Per-Language Macro-F1**: `es=0.791`, `en=0.756`, `fr=0.702`, `de=0.683`, `zh=0.672`, `ko=0.668`, `ja=0.611`

## Provided ONNX Files

- `onnx/lipsync_student_offline_fp16.onnx` (`6.55 MB`, recommended for browser & CLI)
- `onnx/lipsync_student_streaming_fp16.onnx` (`6.54 MB`, stateful streaming with `h_in=[1,1,1]`, `conv_cache_in=[1,288,248]`, `1.82 ms`/frame = `11.0×` real-time in 48 kHz WASM)
- `onnx/lipsync_student_offline_int8.onnx` (`3.57 MB`) & `onnx/lipsync_student_streaming_int8.onnx` (`3.55 MB`)
- `onnx/lipsync_student_offline_fp32.onnx` (`13.05 MB`) & `onnx/lipsync_student_streaming_fp32.onnx` (`13.03 MB`)

## Limitations & Intended Use

Intended for 2D character animation, VTuber/Live2D rigs, Godot/Unity/Blender pipelines, and client-side browser lip-sync. Multilingual `3,600`-utterance test scores are evaluated against `wav2vec2-xlsr-53-espeak-cv-ft` IPA alignments and supplemented by zero-shot human hand-annotated TIMIT evaluation. Performance is strongest on Spanish, English, and French and lower on Japanese (`0.611`) and heavily overlapping/musical audio.
