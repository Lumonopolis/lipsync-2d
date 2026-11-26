import { useEffect, useRef } from "react";
import type { LipSyncCue, RhubarbShape } from "lipsync-2d-web";

interface TimelineProps {
  peaks: Float32Array | null;
  cues: readonly LipSyncCue<RhubarbShape>[];
  duration: number;
  currentTime: number;
  onSeek: (timeSec: number) => void;
}

const SHAPE_COLORS: Record<RhubarbShape, string> = {
  X: "#73706a",
  A: "#c74b50",
  B: "#437fb5",
  C: "#4e9a62",
  D: "#d27d2d",
  E: "#9b59b6",
  F: "#2b9eb3",
  G: "#c29224",
  H: "#d65a8a",
};

export function Timeline({ peaks, cues, duration, currentTime, onSeek }: TimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef(false);

  const seekFromClientX = (clientX: number) => {
    const canvas = canvasRef.current;
    if (!canvas || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / Math.max(1, rect.width)));
    onSeek(ratio * duration);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;

    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const styles = getComputedStyle(document.documentElement);
    const waveColor = styles.getPropertyValue("--accent").trim() || "#2e6fa7";
    const gridColor = styles.getPropertyValue("--border-default").trim() || "#d6cfc3";
    const textMuted = styles.getPropertyValue("--text-secondary").trim() || "#625a50";
    const playheadColor = styles.getPropertyValue("--error").trim() || "#c73737";

    ctx.clearRect(0, 0, width, height);

    const waveHeight = height - 34;
    const waveCenterY = waveHeight / 2;

    // Center axis
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, waveCenterY);
    ctx.lineTo(width, waveCenterY);
    ctx.stroke();

    // Waveform peaks
    if (peaks && peaks.length > 0) {
      ctx.fillStyle = waveColor;
      ctx.globalAlpha = 0.72;
      const barCount = peaks.length;
      const step = width / barCount;
      for (let i = 0; i < barCount; i++) {
        const amp = Math.max(0.03, peaks[i] ?? 0);
        const h = amp * (waveHeight - 12);
        const x = i * step;
        ctx.fillRect(x, waveCenterY - h / 2, Math.max(1, step - 0.5), h);
      }
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = textMuted;
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Load or drop an audio file to inspect waveform & mouth cues", width / 2, waveCenterY + 4);
    }

    // Cue strip along bottom 30px
    const cueTop = height - 30;
    const cueH = 24;
    if (duration > 0 && cues.length > 0) {
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      for (const cue of cues) {
        const x0 = (cue.start / duration) * width;
        const x1 = (cue.end / duration) * width;
        const w = Math.max(1, x1 - x0);
        ctx.fillStyle = SHAPE_COLORS[cue.value] ?? "#73706a";
        ctx.fillRect(x0, cueTop, Math.max(1, w - 1), cueH);
        if (w >= 14) {
          ctx.fillStyle = "#ffffff";
          ctx.fillText(cue.value, x0 + w / 2, cueTop + cueH / 2);
        }
      }
    }

    // Playhead
    if (duration > 0) {
      const px = Math.min(width - 1, Math.max(1, (currentTime / duration) * width));
      ctx.strokeStyle = playheadColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, height);
      ctx.stroke();

      ctx.fillStyle = playheadColor;
      ctx.beginPath();
      ctx.moveTo(px - 5, 0);
      ctx.lineTo(px + 5, 0);
      ctx.lineTo(px, 7);
      ctx.closePath();
      ctx.fill();
    }
  }, [peaks, cues, duration, currentTime]);

  return (
    <section className="panel" aria-label="Waveform and Cue Timeline">
      <div className="panel-head">
        <h2 className="panel-title">Timeline &amp; Rhubarb Cues</h2>
        <span className="muted-note">
          {cues.length > 0 ? `${cues.length} cues · 50 Hz` : "Space play/pause · ←/→ step frame"}
        </span>
      </div>

      <div
        className="timeline-canvas-wrap"
        onPointerDown={(e) => {
          draggingRef.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          seekFromClientX(e.clientX);
        }}
        onPointerMove={(e) => {
          if (draggingRef.current) {
            seekFromClientX(e.clientX);
          }
        }}
        onPointerUp={(e) => {
          draggingRef.current = false;
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId);
          }
        }}
        onPointerCancel={() => {
          draggingRef.current = false;
        }}
      >
        <canvas ref={canvasRef} className="timeline-canvas" />
      </div>
    </section>
  );
}
