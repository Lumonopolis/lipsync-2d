import { OCULUS_VISEMES, type FramePrediction } from "lipsync-2d-web";

interface VisemePanelProps {
  frame: FramePrediction | null;
}

export function VisemePanel({ frame }: VisemePanelProps) {
  const curves = frame?.mouthCurves ?? { jawOpen: 0, lipRound: 0, lipClose: 1 };

  return (
    <section className="panel" aria-label="15 Oculus Viseme Probabilities">
      <div className="panel-head">
        <h2 className="panel-title">Oculus 15 Visemes</h2>
        <span className="muted-note">{frame ? frame.viseme15 : "idle"}</span>
      </div>

      <div className="viseme-grid">
        {OCULUS_VISEMES.map((name, idx) => {
          const prob = frame ? (frame.viseme15Probs[idx] ?? 0) : idx === 0 ? 1 : 0;
          const pct = Math.round(prob * 100);
          const isTop = frame ? frame.viseme15 === name : idx === 0;
          return (
            <div key={name} className={`viseme-row ${isTop ? "is-top" : ""}`}>
              <span className="viseme-name">{name}</span>
              <div className="bar-track" aria-hidden="true">
                <div className="bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <span className="viseme-val">{pct}%</span>
            </div>
          );
        })}
      </div>

      <div className="curve-row" aria-label="Continuous Mouth Curves">
        <div className="curve-cell">
          <span className="curve-label">jawOpen</span>
          <span className="curve-num">{curves.jawOpen.toFixed(2)}</span>
        </div>
        <div className="curve-cell">
          <span className="curve-label">lipRound</span>
          <span className="curve-num">{curves.lipRound.toFixed(2)}</span>
        </div>
        <div className="curve-cell">
          <span className="curve-label">lipClose</span>
          <span className="curve-num">{curves.lipClose.toFixed(2)}</span>
        </div>
      </div>
    </section>
  );
}
