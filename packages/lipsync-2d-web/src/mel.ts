import {
  CAUSAL_LEFT_PAD,
  DEFAULT_EMA_DECAY,
  DEFAULT_MEL_MEAN,
  DEFAULT_MEL_STD,
  DEFAULT_MEL_VAR,
  F_MAX,
  F_MIN,
  HOP_LENGTH,
  MEL_EPS,
  N_FFT,
  N_FREQ_BINS,
  N_MELS,
  SAMPLE_RATE,
  WIN_LENGTH,
} from "./constants.js";

export type MelStandardization = "fixed" | "causal" | "utterance" | "none";

export interface MelExtractionOptions {
  causal?: boolean;
  standardize?: MelStandardization;
  eps?: number;
}

export interface MelSpectrogramResult {
  data: Float32Array;
  frames: number;
  bins: number;
}

const SLANEY_F_SP = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ = 1000.0;
const SLANEY_MIN_LOG_MEL = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
const SLANEY_LOGSTEP = Math.log(6.4) / 27.0;

export function hzToMelSlaney(hz: number): number {
  if (hz < SLANEY_MIN_LOG_HZ) {
    return hz / SLANEY_F_SP;
  }
  return SLANEY_MIN_LOG_MEL + Math.log(hz / SLANEY_MIN_LOG_HZ) / SLANEY_LOGSTEP;
}

export function melToHzSlaney(mel: number): number {
  if (mel < SLANEY_MIN_LOG_MEL) {
    return mel * SLANEY_F_SP;
  }
  return SLANEY_MIN_LOG_HZ * Math.exp(SLANEY_LOGSTEP * (mel - SLANEY_MIN_LOG_MEL));
}

function sinc(x: number): number {
  if (Math.abs(x) < 1e-12) {
    return 1.0;
  }
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

function lanczosKernel(x: number, lobes = 3): number {
  const ax = Math.abs(x);
  if (ax >= lobes) {
    return 0.0;
  }
  return sinc(x) * sinc(x / lobes);
}

/**
 * Downmix multi-channel audio to mono and resample to 16,000 Hz using a
 * Lanczos-3 windowed-sinc filter with anti-aliasing cutoff scaling.
 */
export function resampleTo16kMono(
  input: Float32Array | readonly Float32Array[],
  sourceSampleRate: number,
  targetSampleRate = SAMPLE_RATE,
): Float32Array {
  if (sourceSampleRate <= 0 || !Number.isFinite(sourceSampleRate)) {
    throw new RangeError(`Invalid sourceSampleRate: ${sourceSampleRate}`);
  }

  let mono: Float32Array;
  if (input instanceof Float32Array) {
    mono = input;
  } else {
    if (input.length === 0) {
      return new Float32Array(0);
    }
    const len = input[0]?.length ?? 0;
    mono = new Float32Array(len);
    const chCount = input.length;
    for (let c = 0; c < chCount; c++) {
      const ch = input[c];
      if (!ch) continue;
      for (let i = 0; i < len; i++) {
        mono[i] = (mono[i] ?? 0) + (ch[i] ?? 0);
      }
    }
    const inv = 1 / chCount;
    for (let i = 0; i < len; i++) {
      mono[i] = (mono[i] ?? 0) * inv;
    }
  }

  if (mono.length === 0) {
    return new Float32Array(0);
  }
  if (sourceSampleRate === targetSampleRate) {
    return new Float32Array(mono);
  }

  const ratio = sourceSampleRate / targetSampleRate;
  const outLength = Math.max(1, Math.round(mono.length / ratio));
  const out = new Float32Array(outLength);
  const lobes = 3;
  const filterScale = ratio > 1 ? 1 / ratio : 1;
  const radius = Math.ceil(lobes / filterScale);

  for (let i = 0; i < outLength; i++) {
    const center = i * ratio;
    const start = Math.max(0, Math.floor(center) - radius);
    const end = Math.min(mono.length - 1, Math.ceil(center) + radius);
    let acc = 0;
    let weightSum = 0;
    for (let j = start; j <= end; j++) {
      const w = lanczosKernel((center - j) * filterScale, lobes);
      if (w !== 0) {
        acc += (mono[j] ?? 0) * w;
        weightSum += w;
      }
    }
    out[i] = weightSum !== 0 ? acc / weightSum : 0;
  }

  return out;
}

/**
 * Periodic Hann window matching torch.hann_window(length, periodic=True).
 */
export function createHannWindow(length = WIN_LENGTH): Float32Array {
  const win = new Float32Array(length);
  for (let n = 0; n < length; n++) {
    win[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / length));
  }
  return win;
}

/**
 * Build an [nMels, nFft / 2 + 1] row-major Slaney Mel filterbank matching
 * librosa.filters.mel(..., norm="slaney", htk=False) and Python features.py.
 */
export function createSlaneyMelFilterbank(
  sampleRate = SAMPLE_RATE,
  nFft = N_FFT,
  nMels = N_MELS,
  fMin = F_MIN,
  fMax = F_MAX,
): Float32Array {
  const nBins = Math.floor(nFft / 2) + 1;
  const weights = new Float32Array(nMels * nBins);

  const melMin = hzToMelSlaney(fMin);
  const melMax = hzToMelSlaney(fMax);
  const melPoints = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    const mel = melMin + ((melMax - melMin) * i) / (nMels + 1);
    melPoints[i] = melToHzSlaney(mel);
  }

  const fftFreqs = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    fftFreqs[k] = (k * sampleRate) / nFft;
  }

  for (let m = 0; m < nMels; m++) {
    const fLower = melPoints[m]!;
    const fCenter = melPoints[m + 1]!;
    const fUpper = melPoints[m + 2]!;
    const denomLower = fCenter - fLower;
    const denomUpper = fUpper - fCenter;
    const enorm = 2.0 / (fUpper - fLower);
    const rowOffset = m * nBins;

    for (let k = 0; k < nBins; k++) {
      const f = fftFreqs[k]!;
      const lowerSlope = denomLower > 0 ? (f - fLower) / denomLower : 0;
      const upperSlope = denomUpper > 0 ? (fUpper - f) / denomUpper : 0;
      const val = Math.max(0, Math.min(lowerSlope, upperSlope)) * enorm;
      weights[rowOffset + k] = val;
    }
  }

  return weights;
}

/**
 * Precomputed radix-2 Cooley-Tukey 512-point real-to-power-spectrum FFT.
 */
export class Radix2PowerFFT {
  readonly size: number;
  readonly nBins: number;
  private readonly bitRev: Uint16Array;
  private readonly cosTable: Float64Array;
  private readonly sinTable: Float64Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;

  constructor(size = N_FFT) {
    if ((size & (size - 1)) !== 0 || size < 2) {
      throw new RangeError(`FFT size must be a power of 2, got ${size}`);
    }
    this.size = size;
    this.nBins = size / 2 + 1;
    this.bitRev = new Uint16Array(size);
    const log2 = Math.round(Math.log2(size));
    for (let i = 0; i < size; i++) {
      let rev = 0;
      for (let b = 0; b < log2; b++) {
        rev = (rev << 1) | ((i >> b) & 1);
      }
      this.bitRev[i] = rev;
    }

    const half = size / 2;
    this.cosTable = new Float64Array(half);
    this.sinTable = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      const angle = (-2 * Math.PI * k) / size;
      this.cosTable[k] = Math.cos(angle);
      this.sinTable[k] = Math.sin(angle);
    }

    this.re = new Float64Array(size);
    this.im = new Float64Array(size);
  }

  computePowerSpectrum(frame: Float32Array, outPower: Float32Array): void {
    const n = this.size;
    const re = this.re;
    const im = this.im;
    const bitRev = this.bitRev;

    for (let i = 0; i < n; i++) {
      const r = bitRev[i]!;
      re[i] = r < frame.length ? frame[r]! : 0;
      im[i] = 0;
    }

    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let start = 0; start < n; start += len) {
        for (let k = 0; k < half; k++) {
          const twIdx = k * step;
          const wr = this.cosTable[twIdx]!;
          const wi = this.sinTable[twIdx]!;
          const evenIdx = start + k;
          const oddIdx = evenIdx + half;
          const oddRe = re[oddIdx]!;
          const oddIm = im[oddIdx]!;
          const tr = wr * oddRe - wi * oddIm;
          const ti = wr * oddIm + wi * oddRe;
          re[oddIdx] = re[evenIdx]! - tr;
          im[oddIdx] = im[evenIdx]! - ti;
          re[evenIdx] = re[evenIdx]! + tr;
          im[evenIdx] = im[evenIdx]! + ti;
        }
      }
    }

    for (let k = 0; k < this.nBins; k++) {
      const rk = re[k]!;
      const ik = im[k]!;
      outPower[k] = rk * rk + ik * ik;
    }
  }
}

export class LogMelExtractor {
  readonly sampleRate = SAMPLE_RATE;
  readonly winLength = WIN_LENGTH;
  readonly hopLength = HOP_LENGTH;
  readonly nFft = N_FFT;
  readonly nMels = N_MELS;
  readonly nBins = N_FREQ_BINS;

  private readonly window: Float32Array;
  private readonly melBank: Float32Array;
  private readonly fft: Radix2PowerFFT;
  private readonly paddedFrame: Float32Array;
  private readonly powerBuf: Float32Array;

  constructor() {
    this.window = createHannWindow(WIN_LENGTH);
    this.melBank = createSlaneyMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS, F_MIN, F_MAX);
    this.fft = new Radix2PowerFFT(N_FFT);
    this.paddedFrame = new Float32Array(N_FFT);
    this.powerBuf = new Float32Array(N_FREQ_BINS);
  }

  /**
   * Compute log(mel + eps) for a single 400-sample frame (unstandardized).
   */
  computeFrameLogMel(
    samples: Float32Array,
    offset: number,
    outMel: Float32Array,
    outOffset = 0,
    eps = MEL_EPS,
  ): void {
    this.paddedFrame.fill(0);
    for (let i = 0; i < WIN_LENGTH; i++) {
      this.paddedFrame[i] = (samples[offset + i] ?? 0) * this.window[i]!;
    }
    this.fft.computePowerSpectrum(this.paddedFrame, this.powerBuf);

    for (let m = 0; m < N_MELS; m++) {
      const bankRow = m * N_FREQ_BINS;
      let energy = 0;
      for (let k = 0; k < N_FREQ_BINS; k++) {
        energy += this.melBank[bankRow + k]! * this.powerBuf[k]!;
      }
      outMel[outOffset + m] = Math.log(Math.max(0, energy) + eps);
    }
  }

  /**
   * Extract [T_10ms, 80] Log-Mel spectrogram from 16 kHz mono PCM matching
   * Python's `extract_log_mel_np` bit-for-bit.
   */
  extract(pcm16k: Float32Array, options: MelExtractionOptions = {}): MelSpectrogramResult {
    const causal = options.causal ?? true;
    const standardize = options.standardize ?? "fixed";
    const eps = options.eps ?? MEL_EPS;

    if (pcm16k.length === 0) {
      return { data: new Float32Array(0), frames: 0, bins: N_MELS };
    }

    let padded: Float32Array;
    if (causal) {
      padded = new Float32Array(CAUSAL_LEFT_PAD + pcm16k.length);
      padded.set(pcm16k, CAUSAL_LEFT_PAD);
    } else {
      const half = Math.floor(N_FFT / 2);
      padded = new Float32Array(pcm16k.length + 2 * half);
      padded.set(pcm16k, half);
    }

    if (padded.length < WIN_LENGTH) {
      return { data: new Float32Array(0), frames: 0, bins: N_MELS };
    }

    const numFrames = 1 + Math.floor((padded.length - WIN_LENGTH) / HOP_LENGTH);
    if (numFrames <= 0) {
      return { data: new Float32Array(0), frames: 0, bins: N_MELS };
    }

    const out = new Float32Array(numFrames * N_MELS);
    for (let t = 0; t < numFrames; t++) {
      this.computeFrameLogMel(padded, t * HOP_LENGTH, out, t * N_MELS, eps);
    }

    if (standardize === "fixed") {
      standardizeFixedInPlace(out);
    } else if (standardize === "causal") {
      standardizeCausalEmaInPlace(out, numFrames, N_MELS);
    } else if (standardize === "utterance") {
      standardizeUtteranceInPlace(out, numFrames, N_MELS);
    }

    return { data: out, frames: numFrames, bins: N_MELS };
  }
}

/**
 * Stateless fixed normalization matching Python norm="fixed": (logMel + 6.0) / 4.0.
 */
export function standardizeFixedInPlace(
  data: Float32Array,
  mean = DEFAULT_MEL_MEAN,
  std = DEFAULT_MEL_STD,
): void {
  const invStd = 1.0 / std;
  for (let i = 0; i < data.length; i++) {
    data[i] = (data[i]! - mean) * invStd;
  }
}

export function standardizeUtteranceInPlace(data: Float32Array, frames: number, bins = N_MELS): void {
  if (frames === 0) return;
  for (let m = 0; m < bins; m++) {
    let mean = 0;
    for (let t = 0; t < frames; t++) {
      mean += data[t * bins + m]!;
    }
    mean /= frames;

    let varSum = 0;
    for (let t = 0; t < frames; t++) {
      const d = data[t * bins + m]! - mean;
      varSum += d * d;
    }
    const std = Math.max(Math.sqrt(varSum / frames), 1e-4);
    const invStd = 1 / std;
    for (let t = 0; t < frames; t++) {
      const idx = t * bins + m;
      data[idx] = (data[idx]! - mean) * invStd;
    }
  }
}

/**
 * Strictly causal per-bin EMA normalization matching Python `_causal_ema_normalize_np`.
 */
export function standardizeCausalEmaInPlace(
  data: Float32Array,
  frames: number,
  bins = N_MELS,
  alpha = DEFAULT_EMA_DECAY,
  initMean = DEFAULT_MEL_MEAN,
  initVar = DEFAULT_MEL_VAR,
): void {
  if (frames === 0) return;
  const mean = new Float32Array(bins).fill(initMean);
  const variance = new Float32Array(bins).fill(initVar);
  const oneMinus = 1.0 - alpha;

  for (let t = 0; t < frames; t++) {
    const base = t * bins;
    for (let m = 0; m < bins; m++) {
      const x = data[base + m]!;
      const delta = x - mean[m]!;
      mean[m] = oneMinus * mean[m]! + alpha * x;
      variance[m] = oneMinus * variance[m]! + alpha * (delta * (x - mean[m]!));
      data[base + m] = (x - mean[m]!) / Math.sqrt(Math.max(variance[m]!, 1e-4));
    }
  }
}

/**
 * Compute 50 Hz (20 ms) frame RMS energy in dB matching Python `compute_frame_energy_db`.
 */
export function computeFrameEnergyDb(pcm16k: Float32Array, frameMs = 20): Float32Array {
  const hop = Math.floor((SAMPLE_RATE * frameMs) / 1000);
  const nFrames = Math.floor(pcm16k.length / hop);
  const out = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    let sumSq = 0;
    const base = f * hop;
    for (let i = 0; i < hop; i++) {
      const s = pcm16k[base + i]!;
      sumSq += s * s;
    }
    const rms = Math.sqrt(sumSq / hop + 1e-12);
    out[f] = 20 * Math.log10(rms);
  }
  return out;
}

/**
 * Stateful streaming 10 ms Log-Mel extractor with causal 240-sample left-padding
 * matching offline `extract(..., { causal: true, standardize })` bit-for-bit.
 */
export class StreamingMelExtractor {
  private readonly extractor = new LogMelExtractor();
  private readonly standardize: MelStandardization;
  private buffer = new Float32Array(CAUSAL_LEFT_PAD);
  private readonly runningMean = new Float32Array(N_MELS).fill(DEFAULT_MEL_MEAN);
  private readonly runningVar = new Float32Array(N_MELS).fill(DEFAULT_MEL_VAR);
  private readonly alpha = DEFAULT_EMA_DECAY;

  constructor(standardize: MelStandardization = "fixed") {
    this.standardize = standardize;
  }

  reset(): void {
    this.buffer = new Float32Array(CAUSAL_LEFT_PAD);
    this.runningMean.fill(DEFAULT_MEL_MEAN);
    this.runningVar.fill(DEFAULT_MEL_VAR);
  }

  /**
   * Push 16 kHz mono samples and emit newly available standardized [T_new, 80] frames.
   */
  push(pcm16k: Float32Array): MelSpectrogramResult {
    if (pcm16k.length === 0) {
      return { data: new Float32Array(0), frames: 0, bins: N_MELS };
    }

    const merged = new Float32Array(this.buffer.length + pcm16k.length);
    merged.set(this.buffer, 0);
    merged.set(pcm16k, this.buffer.length);

    if (merged.length < WIN_LENGTH) {
      this.buffer = merged;
      return { data: new Float32Array(0), frames: 0, bins: N_MELS };
    }

    const numFrames = 1 + Math.floor((merged.length - WIN_LENGTH) / HOP_LENGTH);
    const out = new Float32Array(numFrames * N_MELS);
    const oneMinus = 1.0 - this.alpha;

    for (let t = 0; t < numFrames; t++) {
      const offset = t * HOP_LENGTH;
      this.extractor.computeFrameLogMel(merged, offset, out, t * N_MELS);

      const base = t * N_MELS;
      if (this.standardize === "fixed") {
        for (let m = 0; m < N_MELS; m++) {
          out[base + m] = (out[base + m]! - DEFAULT_MEL_MEAN) / DEFAULT_MEL_STD;
        }
      } else if (this.standardize === "causal") {
        for (let m = 0; m < N_MELS; m++) {
          const x = out[base + m]!;
          const delta = x - this.runningMean[m]!;
          this.runningMean[m] = oneMinus * this.runningMean[m]! + this.alpha * x;
          this.runningVar[m] =
            oneMinus * this.runningVar[m]! + this.alpha * (delta * (x - this.runningMean[m]!));
          out[base + m] =
            (x - this.runningMean[m]!) / Math.sqrt(Math.max(this.runningVar[m]!, 1e-4));
        }
      }
    }

    const consumed = numFrames * HOP_LENGTH;
    this.buffer = merged.slice(consumed);
    return { data: out, frames: numFrames, bins: N_MELS };
  }
}

/**
 * Speech-gated RMS normalization matching Python normalize_speech_rms (-18 dBFS target,
 * -46 dBFS noise gate so silence/room-noise is never amplified).
 */
export function normalizeSpeechRms(pcm: Float32Array): Float32Array {
  if (pcm.length === 0) return pcm;
  let sumSq = 0;
  for (let i = 0; i < pcm.length; i++) {
    const v = pcm[i]!;
    sumSq += v * v;
  }
  const rms = Math.sqrt(sumSq / pcm.length);
  if (rms <= 0.005) return pcm;
  const gain = Math.min(10.0, 0.125 / rms);
  if (Math.abs(gain - 1.0) < 1e-3) return pcm;
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    const scaled = pcm[i]! * gain;
    out[i] = scaled < -1 ? -1 : scaled > 1 ? 1 : scaled;
  }
  return out;
}

/**
 * Stateful fractional-phase streaming resampler preventing clock drift and
 * block-boundary discontinuities across 128-sample AudioWorklet quanta.
 */
export class StreamingResampler {
  private readonly ratio: number;
  private carry: Float32Array = new Float32Array(0);
  private phase = 0;

  constructor(sourceRate: number, targetRate = SAMPLE_RATE) {
    this.ratio = sourceRate / targetRate;
  }

  reset(): void {
    this.carry = new Float32Array(0);
    this.phase = 0;
  }

  push(chunk: Float32Array): Float32Array {
    if (chunk.length === 0) return new Float32Array(0);
    if (Math.abs(this.ratio - 1) < 1e-6) {
      return chunk;
    }

    const input = new Float32Array(this.carry.length + chunk.length);
    input.set(this.carry, 0);
    input.set(chunk, this.carry.length);

    if (input.length < 2) {
      this.carry = input;
      return new Float32Array(0);
    }

    const maxPos = input.length - 1;
    const count = this.phase < maxPos ? Math.floor((maxPos - this.phase) / this.ratio) + 1 : 0;
    const out = new Float32Array(count);
    let pos = this.phase;
    for (let i = 0; i < count; i++) {
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = input[idx]!;
      const b = input[idx + 1] ?? a;
      out[i] = a + (b - a) * frac;
      pos += this.ratio;
    }

    const drop = Math.min(input.length - 1, Math.floor(pos));
    this.phase = pos - drop;
    this.carry = input.slice(drop);
    return out;
  }
}
