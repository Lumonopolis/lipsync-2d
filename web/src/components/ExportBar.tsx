import { useState } from "react";
import {
  toBlenderPython,
  toGodotTres,
  toLive2DJson,
  toRhubarbJson,
  toRhubarbTsv,
  toRhubarbXml,
  type FramePrediction,
  type LipSyncCue,
  type RhubarbShape,
} from "lipsync-2d-web";

interface ExportBarProps {
  cues: readonly LipSyncCue<RhubarbShape>[];
  frames: readonly FramePrediction[];
  duration: number;
  soundFile: string;
}

function downloadText(filename: string, content: string, mimeType = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function ExportBar({ cues, frames, duration, soundFile }: ExportBarProps) {
  const [copiedBlender, setCopiedBlender] = useState(false);
  const hasCues = cues.length > 0;
  const baseName = soundFile.replace(/\.[^/.]+$/, "") || "lipsync";

  const handleCopyBlender = async () => {
    const script = toBlenderPython(cues, 24);
    try {
      await navigator.clipboard.writeText(script);
      setCopiedBlender(true);
      window.setTimeout(() => setCopiedBlender(false), 1800);
    } catch {
      downloadText(`${baseName}_blender.py`, script, "text/x-python;charset=utf-8");
    }
  };

  return (
    <section className="panel" aria-label="Export Lip-Sync Data">
      <div className="panel-head">
        <h2 className="panel-title">Export</h2>
        <span className="muted-note">{hasCues ? `${baseName}` : "No cues yet"}</span>
      </div>

      <div className="export-grid">
        <button
          type="button"
          className="btn"
          disabled={!hasCues}
          onClick={() => downloadText(`${baseName}.json`, toRhubarbJson(cues, duration, soundFile), "application/json")}
        >
          Rhubarb JSON
        </button>
        <button
          type="button"
          className="btn"
          disabled={!hasCues}
          onClick={() => downloadText(`${baseName}.tsv`, toRhubarbTsv(cues), "text/tab-separated-values")}
        >
          Rhubarb TSV
        </button>
        <button
          type="button"
          className="btn"
          disabled={!hasCues}
          onClick={() => downloadText(`${baseName}.xml`, toRhubarbXml(cues, duration, soundFile), "application/xml")}
        >
          Rhubarb XML
        </button>
        <button
          type="button"
          className="btn"
          disabled={frames.length === 0}
          onClick={() => downloadText(`${baseName}_live2d.json`, toLive2DJson(frames, 50), "application/json")}
        >
          Live2D JSON
        </button>
        <button
          type="button"
          className="btn"
          disabled={!hasCues}
          onClick={() => downloadText(`${baseName}.tres`, toGodotTres(cues, duration), "text/plain")}
        >
          Godot .tres
        </button>
        <button
          type="button"
          className="btn"
          disabled={!hasCues}
          onClick={handleCopyBlender}
        >
          {copiedBlender ? "Copied Script" : "Copy Blender .py"}
        </button>
      </div>
    </section>
  );
}
