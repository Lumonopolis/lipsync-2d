import * as ort from "onnxruntime-web";
import {
  CONV_CACHE_SHAPE,
  CONV_CACHE_SIZE,
  FRAME_DURATION_SEC,
  HIDDEN_STATE_SHAPE,
  HIDDEN_STATE_SIZE,
  N_MELS,
  OCULUS_VISEMES,
  RHUBARB_SHAPES,
  SAMPLE_RATE,
  type FramePrediction,
  type LipSyncCue,
  type MouthCurves,
  type OculusViseme,
  type RhubarbShape,
} from "./constants.js";
import { decodeRhubarbCues, decodeVisemeCues, type ViterbiDecodeOptions } from "./decoder.js";
import { computeFrameEnergyDb, LogMelExtractor, normalizeSpeechRms, resampleTo16kMono, StreamingMelExtractor, StreamingResampler } from "./mel.js";

export type InferenceBackend = "webgpu" | "wasm";

export interface LipSyncEngineOptions {
  wasmPaths?: string;
  preferWebGpu?: boolean;
  numThreads?: number;
}

export interface AnalysisResult {
  frames: FramePrediction[];
  rhubarbCues: LipSyncCue<RhubarbShape>[];
  visemeCues: LipSyncCue<OculusViseme>[];
  duration: number;
  backend: InferenceBackend;
  inferenceMs: number;
}

export interface StreamingMicHandle {
  stop: () => Promise<void>;
  getBackend: () => InferenceBackend;
}

export class ModelNotLoadedError extends Error {
  constructor(message = "LipSync ONNX model is not loaded. Provide a valid .onnx model URL or file.") {
    super(message);
    this.name = "ModelNotLoadedError";
  }
}

export class ModelLoadError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, cause !== undefined ? { cause } : undefined);
    this.name = "ModelLoadError";
  }
}

function softmaxRow(logits: Float32Array, offset: number, count: number, out: Float32Array): void {
  let maxVal = -Infinity;
  for (let i = 0; i < count; i++) {
    const v = logits[offset + i]!;
    if (v > maxVal) maxVal = v;
  }
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const e = Math.exp(logits[offset + i]! - maxVal);
    out[i] = e;
    sum += e;
  }
  const inv = sum > 0 ? 1 / sum : 1 / count;
  for (let i = 0; i < count; i++) {
    out[i] = out[i]! * inv;
  }
}

function argmax(arr: Float32Array): number {
  let best = 0;
  let bestVal = arr[0] ?? -Infinity;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i]! > bestVal) {
      bestVal = arr[i]!;
      best = i;
    }
  }
  return best;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function buildFramesFromOutputs(
  visemeLogits: Float32Array,
  rhubarbLogits: Float32Array,
  mouthCurves: Float32Array,
  numFrames: number,
  startTimeSec = 0,
): { frames: FramePrediction[]; flatRhubarbProbs: Float32Array; flatVisemeProbs: Float32Array } {
  const frames: FramePrediction[] = [];
  const flatRhubarbProbs = new Float32Array(numFrames * RHUBARB_SHAPES.length);
  const flatVisemeProbs = new Float32Array(numFrames * OCULUS_VISEMES.length);

  for (let t = 0; t < numFrames; t++) {
    const vProbs = new Float32Array(OCULUS_VISEMES.length);
    const rProbs = new Float32Array(RHUBARB_SHAPES.length);

    softmaxRow(visemeLogits, t * OCULUS_VISEMES.length, OCULUS_VISEMES.length, vProbs);
    softmaxRow(rhubarbLogits, t * RHUBARB_SHAPES.length, RHUBARB_SHAPES.length, rProbs);

    flatVisemeProbs.set(vProbs, t * OCULUS_VISEMES.length);
    flatRhubarbProbs.set(rProbs, t * RHUBARB_SHAPES.length);

    const curveBase = t * 3;
    const curves: MouthCurves = {
      jawOpen: clamp01(mouthCurves[curveBase] ?? 0),
      lipRound: clamp01(mouthCurves[curveBase + 1] ?? 0),
      lipClose: clamp01(mouthCurves[curveBase + 2] ?? 0),
    };

    frames.push({
      time: startTimeSec + t * FRAME_DURATION_SEC,
      viseme15Probs: vProbs,
      rhubarb9Probs: rProbs,
      viseme15: OCULUS_VISEMES[argmax(vProbs)]!,
      rhubarb9: RHUBARB_SHAPES[argmax(rProbs)]!,
      mouthCurves: curves,
    });
  }

  return { frames, flatRhubarbProbs, flatVisemeProbs };
}

const STREAMING_WORKLET_SOURCE = `
class LipSyncCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0 && input[0].length > 0) {
      const chCount = input.length;
      const len = input[0].length;
      const mono = new Float32Array(len);
      for (let c = 0; c < chCount; c++) {
        const ch = input[c];
        for (let i = 0; i < len; i++) {
          mono[i] += ch[i];
        }
      }
      if (chCount > 1) {
        const inv = 1 / chCount;
        for (let i = 0; i < len; i++) mono[i] *= inv;
      }
      this.port.postMessage(mono, [mono.buffer]);
    }
    return true;
  }
}
registerProcessor("lipsync-capture-processor", LipSyncCaptureProcessor);
`;

async function fetchModelBytes(source: string | ArrayBuffer | Uint8Array): Promise<{ bytes: Uint8Array; label: string }> {
  if (typeof source !== "string") {
    return {
      bytes: source instanceof Uint8Array ? source : new Uint8Array(source),
      label: "buffer",
    };
  }

  let response: Response;
  try {
    response = await fetch(source);
  } catch (err) {
    throw new ModelLoadError(`Failed to fetch ONNX model from ${source}`, err);
  }
  if (!response.ok) {
    throw new ModelLoadError(
      `ONNX model not found at ${source} (HTTP ${response.status} ${response.statusText})`,
    );
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("text/html")) {
    throw new ModelLoadError(
      `Expected ONNX binary at ${source}, received HTML fallback page instead`,
    );
  }
  const buf = await response.arrayBuffer();
  if (buf.byteLength < 64) {
    throw new ModelLoadError(`ONNX model at ${source} is empty or truncated (${buf.byteLength} bytes)`);
  }
  return { bytes: new Uint8Array(buf), label: source };
}

export class LipSyncEngine {
  private session: ort.InferenceSession | null = null;
  private streamingSession: ort.InferenceSession | null = null;
  private backend: InferenceBackend = "wasm";
  private modelSource: string | null = null;
  private readonly melExtractor = new LogMelExtractor();
  private readonly options: LipSyncEngineOptions;

  constructor(options: LipSyncEngineOptions = {}) {
    this.options = options;
    if (options.wasmPaths) {
      ort.env.wasm.wasmPaths = options.wasmPaths;
    }
    if (options.numThreads !== undefined) {
      ort.env.wasm.numThreads = options.numThreads;
    }
  }

  isLoaded(): boolean {
    return this.session !== null;
  }

  getBackend(): InferenceBackend {
    return this.backend;
  }

  getModelSource(): string | null {
    return this.modelSource;
  }

  hasStreamingHiddenState(): boolean {
    const target = this.streamingSession ?? this.session;
    return target !== null && target.inputNames.includes("h_in") && target.outputNames.includes("h_out");
  }

  hasStreamingModel(): boolean {
    return this.streamingSession !== null || this.hasStreamingHiddenState();
  }

  /**
   * Load or replace only the dedicated stateful streaming ONNX model
   * (inputs: mel_chunk, h_in, conv_cache_in).
   */
  async loadStreamingModel(source: string | ArrayBuffer | Uint8Array): Promise<InferenceBackend> {
    const { bytes } = await fetchModelBytes(source);
    const { session, backend } = await this.createSession(bytes);
    if (this.streamingSession) {
      await this.streamingSession.release();
    }
    this.streamingSession = session;
    return backend;
  }

  private async createSession(bytes: Uint8Array): Promise<{ session: ort.InferenceSession; backend: InferenceBackend }> {
    const preferWebGpu = this.options.preferWebGpu ?? true;
    const hasWebGpu = typeof navigator !== "undefined" && "gpu" in navigator;
    const candidates: InferenceBackend[] = preferWebGpu && hasWebGpu ? ["webgpu", "wasm"] : ["wasm"];

    let lastError: unknown;
    for (const candidate of candidates) {
      try {
        const session = await ort.InferenceSession.create(bytes, {
          executionProviders: [candidate],
        });
        return { session, backend: candidate };
      } catch (err) {
        lastError = err;
      }
    }

    throw new ModelLoadError(
      `Failed to initialize ONNX Runtime session (${candidates.join(" -> ")})`,
      lastError,
    );
  }

  /**
   * Load the primary ONNX model (offline or unified). Optionally pass a dedicated streaming model URL/buffer.
   */
  async loadModel(
    source: string | ArrayBuffer | Uint8Array,
    streamingSource?: string | ArrayBuffer | Uint8Array,
  ): Promise<InferenceBackend> {
    const { bytes, label } = await fetchModelBytes(source);
    const { session, backend } = await this.createSession(bytes);

    let nextStreaming: ort.InferenceSession | null = null;
    if (streamingSource !== undefined) {
      const s = await fetchModelBytes(streamingSource);
      nextStreaming = (await this.createSession(s.bytes)).session;
    }

    await this.disposeSession();
    this.session = session;
    this.streamingSession = nextStreaming;
    this.backend = backend;
    this.modelSource = label;
    return backend;
  }

  async disposeSession(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    if (this.streamingSession) {
      await this.streamingSession.release();
      this.streamingSession = null;
    }
  }

  async analyzeAudio(
    input: AudioBuffer | Float32Array | readonly Float32Array[],
    sampleRate = SAMPLE_RATE,
    decodeOptions?: ViterbiDecodeOptions,
  ): Promise<AnalysisResult> {
    if (!this.session) {
      throw new ModelNotLoadedError();
    }

    let pcm16k: Float32Array;
    if (typeof AudioBuffer !== "undefined" && input instanceof AudioBuffer) {
      const channels: Float32Array[] = [];
      for (let c = 0; c < input.numberOfChannels; c++) {
        channels.push(input.getChannelData(c));
      }
      pcm16k = resampleTo16kMono(channels, input.sampleRate, SAMPLE_RATE);
    } else {
      pcm16k = resampleTo16kMono(input as Float32Array | readonly Float32Array[], sampleRate, SAMPLE_RATE);
    }

    const duration = pcm16k.length / SAMPLE_RATE;
    const pcmNorm = normalizeSpeechRms(pcm16k);
    const mel = this.melExtractor.extract(pcmNorm, { causal: true, standardize: "fixed" });
    if (mel.frames < 2) {
      return {
        frames: [],
        rhubarbCues: [],
        visemeCues: [],
        duration,
        backend: this.backend,
        inferenceMs: 0,
      };
    }

    const inputName = this.session.inputNames.includes("mel")
      ? "mel"
      : (this.session.inputNames[0] ?? "mel");
    const feeds: Record<string, ort.Tensor> = {
      [inputName]: new ort.Tensor("float32", mel.data, [1, mel.frames, N_MELS]),
    };
    if (this.session.inputNames.includes("h_in")) {
      feeds.h_in = new ort.Tensor("float32", new Float32Array(HIDDEN_STATE_SIZE), [...HIDDEN_STATE_SHAPE]);
    }
    if (this.session.inputNames.includes("conv_cache_in")) {
      feeds.conv_cache_in = new ort.Tensor("float32", new Float32Array(CONV_CACHE_SIZE), [...CONV_CACHE_SHAPE]);
    }

    const t0 = performance.now();
    const outputs = await this.session.run(feeds);
    const inferenceMs = performance.now() - t0;

    const vTensor = outputs.viseme15_logits;
    const rTensor = outputs.rhubarb9_logits;
    const cTensor = outputs.mouth_curves;

    if (!vTensor || !rTensor || !cTensor) {
      throw new ModelLoadError(
        `ONNX model outputs missing expected tensors (got: ${Object.keys(outputs).join(", ")})`,
      );
    }

    const outFrames = Number(rTensor.dims[1] ?? 0);
    const { frames, flatRhubarbProbs, flatVisemeProbs } = buildFramesFromOutputs(
      vTensor.data as Float32Array,
      rTensor.data as Float32Array,
      cTensor.data as Float32Array,
      outFrames,
    );

    const energyDb = computeFrameEnergyDb(pcmNorm);
    const rhubarbCues = decodeRhubarbCues(flatRhubarbProbs, {
      audioRmsDb: energyDb,
      ...decodeOptions,
      totalDurationSec: duration,
    });
    const visemeCues = decodeVisemeCues(flatVisemeProbs, {
      audioRmsDb: energyDb,
      ...decodeOptions,
      totalDurationSec: duration,
    });

    return {
      frames,
      rhubarbCues,
      visemeCues,
      duration,
      backend: this.backend,
      inferenceMs,
    };
  }

  /**
   * Run one streaming inference chunk on [1, T_chunk, 80] mel frames while threading
   * h_in -> h_out (and conv_cache_in -> conv_cache_out when present).
   */
  async stepStreamingMel(
    melData: Float32Array,
    melFrames: number,
    hiddenState: Float32Array,
    convCache: Float32Array,
    startTimeSec: number,
  ): Promise<{ frames: FramePrediction[]; nextHiddenState: Float32Array; nextConvCache: Float32Array }> {
    const activeSession = this.streamingSession ?? this.session;
    if (!activeSession) {
      throw new ModelNotLoadedError();
    }

    const melName = activeSession.inputNames.includes("mel_chunk")
      ? "mel_chunk"
      : activeSession.inputNames.includes("mel")
        ? "mel"
        : (activeSession.inputNames[0] ?? "mel");

    const feeds: Record<string, ort.Tensor> = {
      [melName]: new ort.Tensor("float32", melData, [1, melFrames, N_MELS]),
    };
    if (activeSession.inputNames.includes("h_in")) {
      feeds.h_in = new ort.Tensor("float32", hiddenState, [...HIDDEN_STATE_SHAPE]);
    }
    if (activeSession.inputNames.includes("conv_cache_in")) {
      feeds.conv_cache_in = new ort.Tensor("float32", convCache, [...CONV_CACHE_SHAPE]);
    }

    const outputs = await activeSession.run(feeds);
    const vTensor = outputs.viseme15_logits;
    const rTensor = outputs.rhubarb9_logits;
    const cTensor = outputs.mouth_curves;
    if (!vTensor || !rTensor || !cTensor) {
      throw new ModelLoadError("Streaming model outputs missing required tensors");
    }

    const outFrames = Number(rTensor.dims[1] ?? 0);
    const { frames } = buildFramesFromOutputs(
      vTensor.data as Float32Array,
      rTensor.data as Float32Array,
      cTensor.data as Float32Array,
      outFrames,
      startTimeSec,
    );

    const nextHidden =
      outputs.h_out instanceof ort.Tensor
        ? new Float32Array(outputs.h_out.data as Float32Array)
        : hiddenState;
    const nextCache =
      outputs.conv_cache_out instanceof ort.Tensor
        ? new Float32Array(outputs.conv_cache_out.data as Float32Array)
        : convCache;

    return { frames, nextHiddenState: nextHidden, nextConvCache: nextCache };
  }

  /**
   * Capture live microphone via AudioWorklet, resample to 16 kHz, extract causal
   * log-mel frames, and stream ONNX predictions in real time.
   */
  async startMicrophone(
    onFrames: (frames: FramePrediction[]) => void,
    onError?: (err: Error) => void,
  ): Promise<StreamingMicHandle> {
    const activeSession = this.streamingSession ?? this.session;
    if (!activeSession) {
      throw new ModelNotLoadedError();
    }
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone capture is unavailable in this browser context.");
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const audioCtx = new AudioContext();
    await audioCtx.resume();

    const blob = new Blob([STREAMING_WORKLET_SOURCE], { type: "application/javascript" });
    const workletUrl = URL.createObjectURL(blob);

    try {
      await audioCtx.audioWorklet.addModule(workletUrl);
    } finally {
      URL.revokeObjectURL(workletUrl);
    }

    const source = audioCtx.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(audioCtx, "lipsync-capture-processor");
    const silentGain = audioCtx.createGain();
    silentGain.gain.value = 0;

    source.connect(workletNode);
    workletNode.connect(silentGain);
    silentGain.connect(audioCtx.destination);

    const streamingMel = new StreamingMelExtractor();
    const streamingResampler = new StreamingResampler(audioCtx.sampleRate, SAMPLE_RATE);
    const stateful = activeSession.inputNames.includes("h_in");
    let hiddenState: Float32Array = new Float32Array(HIDDEN_STATE_SIZE);
    let convCache: Float32Array = new Float32Array(CONV_CACHE_SIZE);
    let rollingMel: Float32Array = new Float32Array(0);
    let pendingMelFrames = 0;
    let pendingMelChunks: Float32Array[] = [];
    let emittedFrames = 0;
    let inFlight = false;
    let stopped = false;

    const flushChunks = async () => {
      if (inFlight || stopped || pendingMelFrames < 2) return;
      inFlight = true;
      try {
        const totalFrames = pendingMelFrames - (pendingMelFrames % 2);
        if (totalFrames >= 2) {
          const batch = new Float32Array(totalFrames * N_MELS);
          let writePos = 0;
          let remaining = totalFrames * N_MELS;
          const nextChunks: Float32Array[] = [];

          for (const chunk of pendingMelChunks) {
            if (remaining <= 0) {
              nextChunks.push(chunk);
              continue;
            }
            if (chunk.length <= remaining) {
              batch.set(chunk, writePos);
              writePos += chunk.length;
              remaining -= chunk.length;
            } else {
              batch.set(chunk.subarray(0, remaining), writePos);
              nextChunks.push(chunk.subarray(remaining));
              writePos += remaining;
              remaining = 0;
            }
          }

          pendingMelChunks = nextChunks;
          pendingMelFrames -= totalFrames;

          const startTime = emittedFrames * FRAME_DURATION_SEC;
          if (stateful) {
            const res = await this.stepStreamingMel(
              batch,
              totalFrames,
              hiddenState,
              convCache,
              startTime,
            );
            hiddenState = res.nextHiddenState;
            convCache = res.nextConvCache;
            if (!stopped && res.frames.length > 0) {
              emittedFrames += res.frames.length;
              onFrames(res.frames);
            }
          } else {
            // Sliding causal window (up to 100 10 ms frames = 1.0 s) when using offline-only model.
            const maxContext = 100 * N_MELS;
            const merged = new Float32Array(rollingMel.length + batch.length);
            merged.set(rollingMel, 0);
            merged.set(batch, rollingMel.length);
            rollingMel = merged.length > maxContext ? merged.slice(merged.length - maxContext) : merged;
            const ctxFrames = Math.floor(rollingMel.length / N_MELS);
            const res = await this.stepStreamingMel(
              rollingMel,
              ctxFrames,
              hiddenState,
              convCache,
              startTime,
            );
            const newCount = Math.min(res.frames.length, totalFrames / 2);
            const tail = res.frames.slice(res.frames.length - newCount).map((f, idx) => ({
              ...f,
              time: startTime + idx * FRAME_DURATION_SEC,
            }));
            if (!stopped && tail.length > 0) {
              emittedFrames += tail.length;
              onFrames(tail);
            }
          }
        }
      } catch (err) {
        onError?.(err instanceof Error ? err : new Error(String(err)));
      } finally {
        inFlight = false;
        if (!stopped && pendingMelFrames >= 2) {
          void flushChunks();
        }
      }
    };

    workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
      if (stopped) return;
      const pcm16k = streamingResampler.push(event.data);
      const res = streamingMel.push(pcm16k);
      if (res.frames > 0) {
        pendingMelChunks.push(res.data);
        pendingMelFrames += res.frames;
        void flushChunks();
      }
    };

    return {
      getBackend: () => this.backend,
      stop: async () => {
        if (stopped) return;
        stopped = true;
        workletNode.port.onmessage = null;
        workletNode.disconnect();
        source.disconnect();
        silentGain.disconnect();
        for (const track of stream.getTracks()) {
          track.stop();
        }
        await audioCtx.close();
      },
    };
  }
}
