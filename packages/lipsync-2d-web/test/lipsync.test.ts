import { describe, expect, it } from "vitest";
import {
  F_MAX,
  F_MIN,
  HOP_LENGTH,
  N_FFT,
  N_FREQ_BINS,
  N_MELS,
  OCULUS_VISEMES,
  RHUBARB_SHAPES,
  SAMPLE_RATE,
  WIN_LENGTH,
  LogMelExtractor,
  Radix2PowerFFT,
  StreamingMelExtractor,
  StreamingResampler,
  createHannWindow,
  createSlaneyMelFilterbank,
  decodeRhubarbCues,
  decodeVisemeCues,
  hzToMelSlaney,
  melToHzSlaney,
  resampleTo16kMono,
  toBlenderPython,
  toGodotTres,
  toLive2DJson,
  toRhubarbJson,
  toRhubarbTsv,
  toRhubarbXml,
  type FramePrediction,
} from "../src/index.js";

describe("Slaney Mel filterbank & STFT frontend", () => {
  it("inverts Slaney Hz <-> Mel cleanly across linear and log regions", () => {
    for (const hz of [64, 250, 500, 999, 1000, 1001, 2400, 4800, 7800]) {
      const m = hzToMelSlaney(hz);
      const back = melToHzSlaney(m);
      expect(Math.abs(back - hz)).toBeLessThan(1e-6);
    }
  });

  it("constructs a non-negative 80x257 Slaney Mel filterbank bounded to 64-7800 Hz", () => {
    const bank = createSlaneyMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS, F_MIN, F_MAX);
    expect(bank.length).toBe(N_MELS * N_FREQ_BINS);

    for (let m = 0; m < N_MELS; m++) {
      let rowSum = 0;
      let nonZeroCount = 0;
      for (let k = 0; k < N_FREQ_BINS; k++) {
        const w = bank[m * N_FREQ_BINS + k]!;
        expect(w).toBeGreaterThanOrEqual(0);
        if (w > 0) {
          nonZeroCount++;
          const freq = (k * SAMPLE_RATE) / N_FFT;
          expect(freq).toBeGreaterThanOrEqual(F_MIN);
          expect(freq).toBeLessThanOrEqual(F_MAX);
        }
        rowSum += w;
      }
      expect(nonZeroCount).toBeGreaterThanOrEqual(1);
      expect(rowSum).toBeGreaterThan(0);
    }
  });

  it("computes exact Parseval energy in Radix2PowerFFT", () => {
    const fft = new Radix2PowerFFT(N_FFT);
    const frame = new Float32Array(N_FFT);
    let timeEnergy = 0;
    for (let n = 0; n < WIN_LENGTH; n++) {
      const val = Math.sin((2 * Math.PI * 440 * n) / SAMPLE_RATE) * 0.5;
      frame[n] = val;
      timeEnergy += val * val;
    }
    const power = new Float32Array(N_FREQ_BINS);
    fft.computePowerSpectrum(frame, power);

    // Sum two-sided energy: DC + Nyquist once, interior bins twice.
    let freqEnergy = power[0]! + power[N_FREQ_BINS - 1]!;
    for (let k = 1; k < N_FREQ_BINS - 1; k++) {
      freqEnergy += 2 * power[k]!;
    }
    freqEnergy /= N_FFT;

    expect(Math.abs(freqEnergy - timeEnergy)).toBeLessThan(1e-4);
  });

  it("peaks at the expected Mel bin for a 1 kHz sine wave and matches periodic Hann window", () => {
    const win = createHannWindow(WIN_LENGTH);
    expect(win.length).toBe(WIN_LENGTH);
    expect(win[0]).toBeCloseTo(0, 6);
    expect(win[WIN_LENGTH / 2]).toBeCloseTo(1, 6);

    const pcm = new Float32Array(SAMPLE_RATE);
    for (let i = 0; i < pcm.length; i++) {
      pcm[i] = Math.sin((2 * Math.PI * 1000 * i) / SAMPLE_RATE);
    }

    const extractor = new LogMelExtractor();
    const res = extractor.extract(pcm, { causal: true, standardize: "none" });
    expect(res.bins).toBe(N_MELS);
    expect(res.frames).toBe(Math.floor(SAMPLE_RATE / HOP_LENGTH));

    // Check center frame peak bin corresponds to ~1000 Hz.
    const midFrame = Math.floor(res.frames / 2);
    let peakBin = 0;
    let peakVal = -Infinity;
    for (let m = 0; m < N_MELS; m++) {
      const v = res.data[midFrame * N_MELS + m]!;
      if (v > peakVal) {
        peakVal = v;
        peakBin = m;
      }
    }
    const melMin = hzToMelSlaney(F_MIN);
    const melMax = hzToMelSlaney(F_MAX);
    const expectedBin = Math.round(((hzToMelSlaney(1000) - melMin) / (melMax - melMin)) * (N_MELS + 1)) - 1;
    expect(Math.abs(peakBin - expectedBin)).toBeLessThanOrEqual(1);
  });

  it("downmixes stereo 48 kHz audio to 16 kHz mono and matches offline causal Mel frame-for-frame when streaming", () => {
    const sr = 48000;
    const len = sr;
    const left = new Float32Array(len);
    const right = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      left[i] = Math.sin((2 * Math.PI * 500 * i) / sr);
      right[i] = left[i]!;
    }
    const mono16k = resampleTo16kMono([left, right], sr, SAMPLE_RATE);
    expect(mono16k.length).toBe(16000);

    const offline = new LogMelExtractor().extract(mono16k, { causal: true, standardize: "fixed" });
    const streaming = new StreamingMelExtractor("fixed");
    const c1 = streaming.push(mono16k.subarray(0, 4000));
    const c2 = streaming.push(mono16k.subarray(4000, 16000));
    expect(c1.frames + c2.frames).toBe(offline.frames);

    const combined = new Float32Array((c1.frames + c2.frames) * N_MELS);
    combined.set(c1.data, 0);
    combined.set(c2.data, c1.data.length);
    for (let i = 0; i < combined.length; i++) {
      expect(Math.abs(combined[i]! - offline.data[i]!)).toBeLessThan(1e-5);
    }

    // Also verify causal EMA mode matches offline causal EMA.
    const offlineCausal = new LogMelExtractor().extract(mono16k, { causal: true, standardize: "causal" });
    const streamingCausal = new StreamingMelExtractor("causal");
    const e1 = streamingCausal.push(mono16k.subarray(0, 4000));
    const e2 = streamingCausal.push(mono16k.subarray(4000, 16000));
    const combinedCausal = new Float32Array((e1.frames + e2.frames) * N_MELS);
    combinedCausal.set(e1.data, 0);
    combinedCausal.set(e2.data, e1.data.length);
    for (let i = 0; i < combinedCausal.length; i++) {
      expect(Math.abs(combinedCausal[i]! - offlineCausal.data[i]!)).toBeLessThan(1e-5);
    }

    // Verify stateful StreamingResampler produces exact sample count without drift across 128-sample 48 kHz quanta.
    const resampler = new StreamingResampler(48000, SAMPLE_RATE);
    let totalOut = 0;
    for (let pos = 0; pos < left.length; pos += 128) {
      totalOut += resampler.push(left.subarray(pos, Math.min(pos + 128, left.length))).length;
    }
    expect(Math.abs(totalOut - SAMPLE_RATE)).toBeLessThanOrEqual(1);
  });
});

describe("Viterbi cue decoder", () => {
  function makeProbs(sequence: Array<[number, number]>, numClasses: number): Float32Array {
    const out = new Float32Array(sequence.length * numClasses);
    for (let t = 0; t < sequence.length; t++) {
      const [dominant, conf] = sequence[t]!;
      const rem = (1 - conf) / (numClasses - 1);
      for (let c = 0; c < numClasses; c++) {
        out[t * numClasses + c] = c === dominant ? conf : rem;
      }
    }
    return out;
  }

  it("eliminates 1-frame jitter, enforces min duration, and guarantees contiguous non-overlapping timestamps", () => {
    // Pattern: X (5 frames), D (4 frames), 1-frame B spike, D (4 frames), X (5 frames)
    const seq: Array<[number, number]> = [
      [0, 0.92], [0, 0.92], [0, 0.92], [0, 0.92], [0, 0.92],
      [4, 0.88], [4, 0.88], [4, 0.88], [4, 0.88],
      [2, 0.70], // 1-frame jitter inside D!
      [4, 0.88], [4, 0.88], [4, 0.88], [4, 0.88],
      [0, 0.94], [0, 0.94], [0, 0.94], [0, 0.94], [0, 0.94],
    ];
    const probs = makeProbs(seq, RHUBARB_SHAPES.length);
    const cues = decodeRhubarbCues(probs, { minDurationFrames: 2, switchPenalty: 1.0 });

    expect(cues.map((c) => c.value)).toEqual(["X", "D", "X"]);
    expect(cues[0]!.start).toBe(0);
    expect(cues[cues.length - 1]!.end).toBeCloseTo(seq.length * 0.02, 4);

    for (let i = 0; i < cues.length; i++) {
      expect(cues[i]!.end).toBeGreaterThan(cues[i]!.start);
      if (i > 0) {
        expect(cues[i]!.start).toBe(cues[i - 1]!.end);
      }
    }
  });

  it("applies silence-gate hysteresis against weak breath noise during silence", () => {
    const seq: Array<[number, number]> = [
      [0, 0.9], [0, 0.9], [0, 0.9],
      // Weak non-silence frame where sil still has 0.56 prob (below exit speech threshold)
      [0, 0.56], [0, 0.58],
      // Strong speech D
      [4, 0.92], [4, 0.92], [4, 0.92], [4, 0.92],
    ];
    const probs = makeProbs(seq, OCULUS_VISEMES.length);
    const cues = decodeVisemeCues(probs);
    expect(cues[0]!.value).toBe("sil");
    expect(cues[cues.length - 1]!.value).toBe("DD");
  });
});

describe("Format exporters", () => {
  const sampleCues = [
    { start: 0.0, end: 0.12, value: "X" as const, confidence: 0.95 },
    { start: 0.12, end: 0.28, value: "D" as const, confidence: 0.88 },
    { start: 0.28, end: 0.44, value: "A" as const, confidence: 0.91 },
    { start: 0.44, end: 0.6, value: "X" as const, confidence: 0.96 },
  ];

  const sampleFrames: FramePrediction[] = Array.from({ length: 30 }, (_, i) => {
    const vProbs = new Float32Array(OCULUS_VISEMES.length);
    const rProbs = new Float32Array(RHUBARB_SHAPES.length);
    const active = i >= 6 && i < 14;
    vProbs[active ? 10 : 0] = 0.9;
    rProbs[active ? 4 : 0] = 0.9;
    return {
      time: i * 0.02,
      viseme15Probs: vProbs,
      rhubarb9Probs: rProbs,
      viseme15: active ? "aa" : "sil",
      rhubarb9: active ? "D" : "X",
      mouthCurves: {
        jawOpen: active ? 0.92 : 0.0,
        lipRound: active ? 0.05 : 0.0,
        lipClose: active ? 0.0 : 1.0,
      },
    };
  });

  it("produces 100% Rhubarb schema valid JSON, TSV, and XML", () => {
    const jsonStr = toRhubarbJson(sampleCues, 0.6, "voice & test.wav");
    const parsed = JSON.parse(jsonStr);
    expect(parsed).toEqual({
      metadata: { soundFile: "voice & test.wav", duration: 0.6 },
      mouthCues: [
        { start: 0, end: 0.12, value: "X" },
        { start: 0.12, end: 0.28, value: "D" },
        { start: 0.28, end: 0.44, value: "A" },
        { start: 0.44, end: 0.6, value: "X" },
      ],
    });

    const tsv = toRhubarbTsv(sampleCues);
    expect(tsv).toBe("0.00\tX\n0.12\tD\n0.28\tA\n0.44\tX\n");

    const xml = toRhubarbXml(sampleCues, 0.6, "voice & test.wav");
    expect(xml).toContain('<?xml version="1.0" encoding="utf-8"?>');
    expect(xml).toContain("<soundFile>voice &amp; test.wav</soundFile>");
    expect(xml).toContain("<duration>0.60</duration>");
    expect(xml).toContain('<mouthCue start="0.12" end="0.28">D</mouthCue>');
  });

  it("produces valid Live2D JSON with ParamMouthOpenY, ParamMouthForm, and 15 Oculus visemes", () => {
    const live2d = JSON.parse(toLive2DJson(sampleFrames, 50));
    expect(live2d.fps).toBe(50);
    expect(live2d.frameCount).toBe(30);
    expect(live2d.parameters.ParamMouthOpenY).toHaveLength(30);
    expect(live2d.parameters.ParamMouthForm).toHaveLength(30);
    expect(Object.keys(live2d.visemes)).toEqual([...OCULUS_VISEMES]);
    expect(live2d.parameters.ParamMouthOpenY[8]).toBeCloseTo(0.92, 2);
  });

  it("produces valid Godot 4 Animation .tres resource and runnable Blender Python script", () => {
    const tres = toGodotTres(sampleCues, 0.6);
    expect(tres).toContain('[gd_resource type="Animation" format=3]');
    expect(tres).toContain("length = 0.60");
    expect(tres).toContain('"times": PackedFloat32Array(0, 0.12, 0.28, 0.44)');
    expect(tres).toContain('"values": ["X", "D", "A", "X"]');

    const py = toBlenderPython(sampleCues, 24);
    expect(py).toContain("import bpy");
    expect(py).toContain("FPS = 24");
    expect(py).toContain('obj.keyframe_insert(data_path=data_path, frame=frame)');
    expect(py).toContain("kp.interpolation = 'CONSTANT'");
  });
});
