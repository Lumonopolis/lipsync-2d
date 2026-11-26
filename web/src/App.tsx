import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  FRAME_DURATION_SEC,
  LipSyncEngine,
  type AnalysisResult,
  type FramePrediction,
  type InferenceBackend,
  type LipSyncCue,
  type RhubarbShape,
  type StreamingMicHandle,
} from "lipsync-2d-web";
import { CharacterStage, type VectorStyle } from "./components/CharacterStage.tsx";
import { ExportBar } from "./components/ExportBar.tsx";
import { Timeline } from "./components/Timeline.tsx";
import { VisemePanel } from "./components/VisemePanel.tsx";

const BASE_URL = import.meta.env.BASE_URL ?? "/";
function assetUrl(relPath: string): string {
  return `${BASE_URL.replace(/\/$/, "")}/${relPath.replace(/^\//, "")}`;
}

const DEMO_SAMPLES = [
  { label: "EN · Hello Studio (1.62s)", url: assetUrl("samples/hello_studio.wav"), file: "hello_studio.wav" },
  { label: "EN · Phoneme Sweep (1.64s)", url: assetUrl("samples/phoneme_sweep.wav"), file: "phoneme_sweep.wav" },
  { label: "JA · Japanese Demo (3.9s)", url: assetUrl("samples/ja_demo.wav"), file: "ja_demo.wav" },
  { label: "ES · Spanish Demo (2.9s)", url: assetUrl("samples/es_demo.wav"), file: "es_demo.wav" },
] as const;

const DEFAULT_MODEL_URL = assetUrl("models/lipsync2d.onnx");
const DEFAULT_STREAM_MODEL_URL = assetUrl("models/lipsync2d_stream.onnx");
const FALLBACK_HF_MODEL_URL =
  "https://huggingface.co/Lumonopolis/lipsync-2d/resolve/main/onnx/lipsync_student_offline_fp16.onnx";
const FALLBACK_HF_STREAM_URL =
  "https://huggingface.co/Lumonopolis/lipsync-2d/resolve/main/onnx/lipsync_student_streaming_fp16.onnx";
const ORT_WASM_BASE = `${BASE_URL.replace(/\/$/, "")}/ort/`;

type ModelStatus = "loading" | "ready" | "missing";

function computeWaveformPeaks(buffer: AudioBuffer, bucketCount = 360): Float32Array {
  const ch = buffer.getChannelData(0);
  const len = ch.length;
  const count = Math.min(bucketCount, Math.max(1, len));
  const peaks = new Float32Array(count);
  const blockSize = len / count;

  let globalMax = 0;
  for (let b = 0; b < count; b++) {
    const start = Math.floor(b * blockSize);
    const end = Math.min(len, Math.floor((b + 1) * blockSize));
    let peak = 0;
    for (let i = start; i < end; i++) {
      const v = Math.abs(ch[i] ?? 0);
      if (v > peak) peak = v;
    }
    peaks[b] = peak;
    if (peak > globalMax) globalMax = peak;
  }

  if (globalMax > 0) {
    const norm = 1 / globalMax;
    for (let b = 0; b < count; b++) {
      peaks[b] = (peaks[b] ?? 0) * norm;
    }
  }
  return peaks;
}

function findCueAtTime(cues: readonly LipSyncCue<RhubarbShape>[], time: number): RhubarbShape | null {
  for (let i = 0; i < cues.length; i++) {
    const c = cues[i]!;
    if (time >= c.start && time < c.end) {
      return c.value;
    }
  }
  return cues.length > 0 && time >= cues[cues.length - 1]!.start ? cues[cues.length - 1]!.value : null;
}

function formatTime(sec: number): string {
  return `${sec.toFixed(2)}s`;
}

export function App() {
  const engineRef = useRef<LipSyncEngine | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const micHandleRef = useRef<StreamingMicHandle | null>(null);
  const rafRef = useRef<number | null>(null);
  const playStartCtxTimeRef = useRef(0);
  const playStartOffsetRef = useRef(0);

  const audioFileInputRef = useRef<HTMLInputElement | null>(null);
  const modelFileInputRef = useRef<HTMLInputElement | null>(null);

  const [theme, setTheme] = useState<"light" | "dark">(() =>
    window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light",
  );
  const [modelStatus, setModelStatus] = useState<ModelStatus>("loading");
  const [backend, setBackend] = useState<InferenceBackend | null>(null);
  const [modelMessage, setModelMessage] = useState("Checking /models/lipsync2d.onnx...");
  void modelMessage;

  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [audioFileName, setAudioFileName] = useState<string>("hello_studio.wav");
  const [selectedSampleUrl, setSelectedSampleUrl] = useState<string>("");
  const [waveformPeaks, setWaveformPeaks] = useState<Float32Array | null>(null);

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const [isMicActive, setIsMicActive] = useState(false);
  const [liveFrame, setLiveFrame] = useState<FramePrediction | null>(null);

  const [vectorStyle, setVectorStyle] = useState<VectorStyle>("editorial");
  const [lockedShape, setLockedShape] = useState<RhubarbShape | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const getAudioContext = useCallback(() => {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext();
    }
    return audioCtxRef.current;
  }, []);

  const stopPlayback = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.onended = null;
        sourceNodeRef.current.stop();
        sourceNodeRef.current.disconnect();
      } catch {
        // Ignore already-stopped source errors.
      }
      sourceNodeRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const runAnalysis = useCallback(async (buf: AudioBuffer) => {
    const engine = engineRef.current;
    if (!engine || !engine.isLoaded()) {
      setAnalysis(null);
      return;
    }
    setIsAnalyzing(true);
    setErrorMsg(null);
    try {
      const res = await engine.analyzeAudio(buf);
      setAnalysis(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
      setAnalysis(null);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const loadDefaultModel = useCallback(async () => {
    const engine = engineRef.current ?? new LipSyncEngine({ wasmPaths: ORT_WASM_BASE });
    engineRef.current = engine;
    setModelStatus("loading");
    setModelMessage(`Loading ${DEFAULT_MODEL_URL}...`);
    try {
      let activeBackend: InferenceBackend;
      let streamUrl = DEFAULT_STREAM_MODEL_URL;
      try {
        activeBackend = await engine.loadModel(DEFAULT_MODEL_URL);
      } catch {
        activeBackend = await engine.loadModel(FALLBACK_HF_MODEL_URL);
        streamUrl = FALLBACK_HF_STREAM_URL;
      }
      try {
        await engine.loadStreamingModel(streamUrl);
      } catch {
        // Dedicated streaming model is optional; engine handles single-model cleanly.
      }
      setBackend(activeBackend);
      setModelStatus("ready");
      setModelMessage(`Ready (${activeBackend.toUpperCase()})`);
      setErrorMsg(null);
    } catch (err) {
      setBackend(null);
      setModelStatus("missing");
      setModelMessage(err instanceof Error ? err.message : "ONNX model not found.");
    }
  }, []);

  useEffect(() => {
    void loadDefaultModel();
    return () => {
      stopPlayback();
      void micHandleRef.current?.stop();
      void engineRef.current?.disposeSession();
    };
  }, [loadDefaultModel, stopPlayback]);

  // Re-run analysis automatically if model becomes ready while audio is already loaded.
  useEffect(() => {
    if (modelStatus === "ready" && audioBuffer && !analysis && !isAnalyzing) {
      void runAnalysis(audioBuffer);
    }
  }, [modelStatus, audioBuffer, analysis, isAnalyzing, runAnalysis]);

  const loadAudioArrayBuffer = useCallback(
    async (arrayBuf: ArrayBuffer, name: string) => {
      stopPlayback();
      setErrorMsg(null);
      try {
        const ctx = getAudioContext();
        const decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
        setAudioBuffer(decoded);
        setAudioFileName(name);
        setWaveformPeaks(computeWaveformPeaks(decoded));
        setCurrentTime(0);
        setAnalysis(null);
        setLockedShape(null);
        if (engineRef.current?.isLoaded()) {
          await runAnalysis(decoded);
        }
      } catch (err) {
        setErrorMsg(`Failed to decode audio (${name}): ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [getAudioContext, runAnalysis, stopPlayback],
  );

  const handleLoadSample = useCallback(
    async (url: string) => {
      setSelectedSampleUrl(url);
      if (!url) return;
      const sample = DEMO_SAMPLES.find((s) => s.url === url);
      const name = sample?.file ?? "sample.wav";
      try {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status} ${res.statusText}`);
        }
        const buf = await res.arrayBuffer();
        await loadAudioArrayBuffer(buf, name);
      } catch (err) {
        setErrorMsg(`Failed to load demo audio: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [loadAudioArrayBuffer],
  );

  // Load initial sample waveform cleanly on startup so studio opens populated.
  useEffect(() => {
    void handleLoadSample(DEMO_SAMPLES[0].url);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleModelFiles = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      if (list.length === 0) return;
      const engine = engineRef.current ?? new LipSyncEngine({ wasmPaths: "/ort/" });
      engineRef.current = engine;
      setModelStatus("loading");
      setErrorMsg(null);
      try {
        const streamFile = list.find((f) => f.name.toLowerCase().includes("stream"));
        const primaryFile = list.find((f) => f !== streamFile) ?? list[0]!;
        const primaryBuf = await primaryFile.arrayBuffer();
        const streamBuf = streamFile ? await streamFile.arrayBuffer() : undefined;
        const activeBackend = await engine.loadModel(primaryBuf, streamBuf);
        setBackend(activeBackend);
        setModelStatus("ready");
        setModelMessage(`${primaryFile.name} (${activeBackend.toUpperCase()})`);
        if (audioBuffer) {
          await runAnalysis(audioBuffer);
        }
      } catch (err) {
        setModelStatus("missing");
        setBackend(null);
        setModelMessage(err instanceof Error ? err.message : String(err));
      }
    },
    [audioBuffer, runAnalysis],
  );

  const startPlaybackFrom = useCallback(
    async (offsetSec: number) => {
      if (!audioBuffer) return;
      stopPlayback();
      const ctx = getAudioContext();
      if (ctx.state === "suspended") {
        await ctx.resume();
      }

      const clampedOffset = Math.min(Math.max(0, offsetSec), Math.max(0, audioBuffer.duration - 0.01));
      const src = ctx.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(ctx.destination);

      playStartCtxTimeRef.current = ctx.currentTime;
      playStartOffsetRef.current = clampedOffset;
      sourceNodeRef.current = src;

      src.onended = () => {
        if (sourceNodeRef.current === src) {
          stopPlayback();
          setCurrentTime(0);
        }
      };

      src.start(0, clampedOffset);
      setIsPlaying(true);

      const tick = () => {
        if (!sourceNodeRef.current) return;
        const elapsed = ctx.currentTime - playStartCtxTimeRef.current;
        const nextTime = playStartOffsetRef.current + elapsed;
        if (nextTime >= audioBuffer.duration) {
          stopPlayback();
          setCurrentTime(audioBuffer.duration);
          return;
        }
        setCurrentTime(nextTime);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    },
    [audioBuffer, getAudioContext, stopPlayback],
  );

  const togglePlayback = useCallback(() => {
    if (!audioBuffer) return;
    if (isPlaying) {
      const ctx = audioCtxRef.current;
      const pauseAt = ctx
        ? Math.min(audioBuffer.duration, playStartOffsetRef.current + (ctx.currentTime - playStartCtxTimeRef.current))
        : currentTime;
      stopPlayback();
      setCurrentTime(pauseAt);
    } else {
      const startAt = currentTime >= audioBuffer.duration - 0.03 ? 0 : currentTime;
      void startPlaybackFrom(startAt);
    }
  }, [audioBuffer, currentTime, isPlaying, startPlaybackFrom, stopPlayback]);

  const handleSeek = useCallback(
    (timeSec: number) => {
      if (!audioBuffer) return;
      const clamped = Math.min(Math.max(0, timeSec), audioBuffer.duration);
      setCurrentTime(clamped);
      if (isPlaying) {
        void startPlaybackFrom(clamped);
      }
    },
    [audioBuffer, isPlaying, startPlaybackFrom],
  );

  const stepFrame = useCallback(
    (deltaFrames: number) => {
      if (!audioBuffer) return;
      stopPlayback();
      setCurrentTime((prev) =>
        Math.min(audioBuffer.duration, Math.max(0, Math.round((prev + deltaFrames * FRAME_DURATION_SEC) * 50) / 50)),
      );
    },
    [audioBuffer, stopPlayback],
  );

  const toggleMic = useCallback(async () => {
    if (isMicActive) {
      await micHandleRef.current?.stop();
      micHandleRef.current = null;
      setIsMicActive(false);
      setLiveFrame(null);
      return;
    }

    const engine = engineRef.current;
    if (!engine || !engine.isLoaded()) {
      setErrorMsg("Load an ONNX lip-sync model before starting live microphone inference.");
      return;
    }

    stopPlayback();
    setErrorMsg(null);
    setLockedShape(null);
    try {
      const handle = await engine.startMicrophone(
        (frames) => {
          const last = frames[frames.length - 1];
          if (last) setLiveFrame(last);
        },
        (err) => {
          setErrorMsg(`Microphone inference error: ${err.message}`);
          setIsMicActive(false);
        },
      );
      micHandleRef.current = handle;
      setIsMicActive(true);
    } catch (err) {
      setErrorMsg(`Microphone unavailable: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [isMicActive, stopPlayback]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlayback();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        stepFrame(-1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        stepFrame(1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [stepFrame, togglePlayback]);

  const activePlaybackFrame = useMemo(() => {
    if (!analysis || analysis.frames.length === 0) return null;
    const idx = Math.min(
      analysis.frames.length - 1,
      Math.max(0, Math.floor(currentTime / FRAME_DURATION_SEC)),
    );
    return analysis.frames[idx] ?? null;
  }, [analysis, currentTime]);

  const activeFrame = isMicActive ? liveFrame : activePlaybackFrame;

  const activeShape: RhubarbShape = useMemo(() => {
    if (lockedShape) return lockedShape;
    if (isMicActive && liveFrame) return liveFrame.rhubarb9;
    if (analysis) {
      const cueShape = findCueAtTime(analysis.rhubarbCues, currentTime);
      if (cueShape) return cueShape;
    }
    return activeFrame?.rhubarb9 ?? "X";
  }, [activeFrame, analysis, currentTime, isMicActive, liveFrame, lockedShape]);

  const duration = audioBuffer?.duration ?? analysis?.duration ?? 0;

  return (
    <div
      className={`studio-shell ${isDragging ? "is-dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setIsDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;
        const onnxFiles = files.filter((f) => f.name.toLowerCase().endsWith(".onnx"));
        if (onnxFiles.length > 0) {
          void handleModelFiles(onnxFiles);
          return;
        }
        const audioFile = files[0]!;
        setSelectedSampleUrl("");
        void audioFile.arrayBuffer().then((buf) => loadAudioArrayBuffer(buf, audioFile.name));
      }}
    >
      <header className="studio-header">
        <div className="header-brand">
          <h1 className="brand-title">LipSync 2D</h1>
          <span
            className={`status-pill ${
              modelStatus === "ready"
                ? "is-ready"
                : modelStatus === "loading"
                  ? "is-loading"
                  : "is-error"
            }`}
          >
            <span className="status-dot" aria-hidden="true" />
            {modelStatus === "ready"
              ? `${backend?.toUpperCase() ?? "WASM"} Ready`
              : modelStatus === "loading"
                ? "Loading Model"
                : "Model Missing"}
          </span>
        </div>

        <div className="header-controls">
          <input
            ref={modelFileInputRef}
            type="file"
            accept=".onnx"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) void handleModelFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            className="btn"
            onClick={() => modelFileInputRef.current?.click()}
          >
            Load .onnx
          </button>
          <button
            type="button"
            className="btn"
            aria-label="Toggle color theme"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </header>

      {modelStatus === "missing" && (
        <div className="model-notice" role="status">
          <span>
            No ONNX weights at <code>/models/lipsync2d.onnx</code> — load a local <code>.onnx</code> file to run inference.
          </span>
          <div className="model-notice-actions">
            <button type="button" className="btn" onClick={() => void loadDefaultModel()}>
              Retry Default
            </button>
            <button type="button" className="btn btn-primary" onClick={() => modelFileInputRef.current?.click()}>
              Select .onnx
            </button>
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="error-banner" role="alert">
          <span>{errorMsg}</span>
          <button type="button" className="btn" onClick={() => setErrorMsg(null)}>
            Dismiss
          </button>
        </div>
      )}

      <main className="studio-main">
        <div className="workspace-col">
          <section className="panel transport-bar" aria-label="Audio Source and Transport">
            <div className="transport-group">
              <input
                ref={audioFileInputRef}
                type="file"
                accept="audio/*,.wav,.mp3,.ogg,.m4a"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    setSelectedSampleUrl("");
                    void f.arrayBuffer().then((buf) => loadAudioArrayBuffer(buf, f.name));
                  }
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                className="btn"
                onClick={() => audioFileInputRef.current?.click()}
              >
                Upload Audio
              </button>

              <select
                className="select"
                aria-label="Demo Audio Sample"
                value={selectedSampleUrl}
                onChange={(e) => void handleLoadSample(e.target.value)}
              >
                <option value="">Demo audio...</option>
                {DEMO_SAMPLES.map((s) => (
                  <option key={s.url} value={s.url}>
                    {s.label}
                  </option>
                ))}
              </select>

              <button
                type="button"
                className={`btn ${isMicActive ? "btn-danger" : ""}`}
                aria-pressed={isMicActive}
                onClick={() => void toggleMic()}
              >
                {isMicActive ? "Stop Mic" : "Live Mic"}
              </button>
            </div>

            <div className="transport-group">
              <button
                type="button"
                className="btn btn-icon"
                disabled={!audioBuffer}
                aria-label="Previous 20ms frame"
                title="Step Back 20ms (Left Arrow)"
                onClick={() => stepFrame(-1)}
              >
                ◀
              </button>
              <button
                type="button"
                className="btn btn-primary"
                disabled={!audioBuffer}
                onClick={togglePlayback}
              >
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                className="btn btn-icon"
                disabled={!audioBuffer}
                aria-label="Next 20ms frame"
                title="Step Forward 20ms (Right Arrow)"
                onClick={() => stepFrame(1)}
              >
                ▶
              </button>
              <span className="timecode">
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>
          </section>

          <CharacterStage
            shape={activeShape}
            style={vectorStyle}
            onStyleChange={setVectorStyle}
            curves={lockedShape ? undefined : activeFrame?.mouthCurves}
            lockedShape={lockedShape}
            onSelectPreviewShape={setLockedShape}
            isLiveActive={isMicActive}
          />

          <Timeline
            peaks={waveformPeaks}
            cues={analysis?.rhubarbCues ?? []}
            duration={duration}
            currentTime={currentTime}
            onSeek={handleSeek}
          />
        </div>

        <aside className="inspector-col">
          <VisemePanel frame={lockedShape ? null : activeFrame} />
          <ExportBar
            cues={analysis?.rhubarbCues ?? []}
            frames={analysis?.frames ?? []}
            duration={duration}
            soundFile={audioFileName}
          />
          {analysis && (
            <section className="panel">
              <p className="muted-note">
                {analysis.frames.length} frames · {analysis.rhubarbCues.length} cues ·{" "}
                {analysis.inferenceMs.toFixed(1)} ms ({analysis.backend.toUpperCase()})
              </p>
            </section>
          )}
          {isAnalyzing && (
            <section className="panel">
              <p className="muted-note">Running ONNX inference...</p>
            </section>
          )}
        </aside>
      </main>
    </div>
  );
}
