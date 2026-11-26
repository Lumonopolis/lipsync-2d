import { useEffect, useState } from "react";
import {
  RHUBARB_DESCRIPTIONS,
  RHUBARB_SHAPES,
  type MouthCurves,
  type RhubarbShape,
} from "lipsync-2d-web";

export type VectorStyle = "editorial" | "indie";

interface CharacterStageProps {
  shape: RhubarbShape;
  style: VectorStyle;
  onStyleChange: (style: VectorStyle) => void;
  curves?: MouthCurves;
  lockedShape: RhubarbShape | null;
  onSelectPreviewShape: (shape: RhubarbShape | null) => void;
  isLiveActive: boolean;
}

interface MouthGeom {
  width: number;
  openHeight: number;
  roundness: number;
  showUpperTeeth: boolean;
  showLowerTeeth: boolean;
  teethBiteLip: boolean;
  showTongue: boolean;
  tongueTipUp: boolean;
  pressedLip: boolean;
}

const BASE_MOUTH_GEOM: Record<RhubarbShape, MouthGeom> = {
  X: {
    width: 24,
    openHeight: 0,
    roundness: 0,
    showUpperTeeth: false,
    showLowerTeeth: false,
    teethBiteLip: false,
    showTongue: false,
    tongueTipUp: false,
    pressedLip: false,
  },
  A: {
    width: 28,
    openHeight: 0,
    roundness: 0,
    showUpperTeeth: false,
    showLowerTeeth: false,
    teethBiteLip: false,
    showTongue: false,
    tongueTipUp: false,
    pressedLip: true,
  },
  B: {
    width: 30,
    openHeight: 7,
    roundness: 0.15,
    showUpperTeeth: true,
    showLowerTeeth: true,
    teethBiteLip: false,
    showTongue: false,
    tongueTipUp: false,
    pressedLip: false,
  },
  C: {
    width: 30,
    openHeight: 15,
    roundness: 0.35,
    showUpperTeeth: true,
    showLowerTeeth: true,
    teethBiteLip: false,
    showTongue: true,
    tongueTipUp: false,
    pressedLip: false,
  },
  D: {
    width: 30,
    openHeight: 25,
    roundness: 0.45,
    showUpperTeeth: true,
    showLowerTeeth: true,
    teethBiteLip: false,
    showTongue: true,
    tongueTipUp: false,
    pressedLip: false,
  },
  E: {
    width: 21,
    openHeight: 19,
    roundness: 0.85,
    showUpperTeeth: true,
    showLowerTeeth: false,
    teethBiteLip: false,
    showTongue: true,
    tongueTipUp: false,
    pressedLip: false,
  },
  F: {
    width: 12,
    openHeight: 11,
    roundness: 1.0,
    showUpperTeeth: false,
    showLowerTeeth: false,
    teethBiteLip: false,
    showTongue: false,
    tongueTipUp: false,
    pressedLip: false,
  },
  G: {
    width: 28,
    openHeight: 7,
    roundness: 0.1,
    showUpperTeeth: true,
    showLowerTeeth: false,
    teethBiteLip: true,
    showTongue: false,
    tongueTipUp: false,
    pressedLip: false,
  },
  H: {
    width: 27,
    openHeight: 14,
    roundness: 0.3,
    showUpperTeeth: true,
    showLowerTeeth: false,
    teethBiteLip: false,
    showTongue: true,
    tongueTipUp: true,
    pressedLip: false,
  },
};

function MouthRenderer({
  shape,
  curves,
  cx,
  cy,
  inkColor,
  cavityColor,
  tongueColor,
  lipFill,
}: {
  shape: RhubarbShape;
  curves?: MouthCurves;
  cx: number;
  cy: number;
  inkColor: string;
  cavityColor: string;
  tongueColor: string;
  lipFill: string;
}) {
  const base = BASE_MOUTH_GEOM[shape];
  const hasCurves = curves !== undefined;
  const openBoost = hasCurves && base.openHeight > 0 ? (curves.jawOpen - 0.5) * 5 : 0;
  const roundAdjust = hasCurves && base.openHeight > 0 ? (curves.lipRound - 0.3) * 4 : 0;

  const halfW = Math.max(5.5, base.width / 2 - roundAdjust * 0.45);
  const openH = Math.max(0, base.openHeight + openBoost);

  if (shape === "X") {
    return (
      <g>
        <path
          d={`M ${cx - halfW} ${cy} Q ${cx} ${cy + 2.5} ${cx + halfW} ${cy}`}
          fill="none"
          stroke={inkColor}
          strokeWidth={2.5}
          strokeLinecap="round"
        />
      </g>
    );
  }

  if (shape === "A" || base.pressedLip) {
    return (
      <g>
        <path
          d={`M ${cx - halfW} ${cy + 0.5} Q ${cx} ${cy - 1.2} ${cx + halfW} ${cy + 0.5}`}
          fill="none"
          stroke={inkColor}
          strokeWidth={3.1}
          strokeLinecap="round"
        />
        <path
          d={`M ${cx - halfW - 2.5} ${cy - 2} Q ${cx - halfW - 4} ${cy + 0.5} ${cx - halfW - 2} ${cy + 3}`}
          fill="none"
          stroke={inkColor}
          strokeWidth={1.6}
          strokeLinecap="round"
        />
        <path
          d={`M ${cx + halfW + 2.5} ${cy - 2} Q ${cx + halfW + 4} ${cy + 0.5} ${cx + halfW + 2} ${cy + 3}`}
          fill="none"
          stroke={inkColor}
          strokeWidth={1.6}
          strokeLinecap="round"
        />
      </g>
    );
  }

  if (shape === "G" || base.teethBiteLip) {
    return (
      <g>
        {/* Lower lip tucked up under upper teeth */}
        <path
          d={`M ${cx - halfW} ${cy + 1} Q ${cx} ${cy + 8} ${cx + halfW} ${cy + 1} Q ${cx} ${cy + 3.5} ${cx - halfW} ${cy + 1} Z`}
          fill={lipFill}
          stroke={inkColor}
          strokeWidth={2}
          strokeLinejoin="round"
        />
        {/* Upper teeth biting onto lower lip */}
        <path
          d={`M ${cx - halfW * 0.8} ${cy - 3.5} L ${cx + halfW * 0.8} ${cy - 3.5} L ${cx + halfW * 0.72} ${cy + 2.2} Q ${cx} ${cy + 3.2} ${cx - halfW * 0.72} ${cy + 2.2} Z`}
          fill="#ffffff"
          stroke={inkColor}
          strokeWidth={1.8}
          strokeLinejoin="round"
        />
        <path
          d={`M ${cx - halfW - 1} ${cy - 2} Q ${cx} ${cy - 5} ${cx + halfW + 1} ${cy - 2}`}
          fill="none"
          stroke={inkColor}
          strokeWidth={2.2}
          strokeLinecap="round"
        />
      </g>
    );
  }

  if (shape === "F") {
    const rX = Math.max(5.5, halfW * 0.88);
    const rY = Math.max(5.5, openH * 0.52);
    return (
      <g>
        <ellipse
          cx={cx}
          cy={cy + 2}
          rx={rX + 2.8}
          ry={rY + 2.4}
          fill={lipFill}
          opacity={0.7}
        />
        <ellipse
          cx={cx}
          cy={cy + 2}
          rx={rX}
          ry={rY}
          fill={cavityColor}
          stroke={inkColor}
          strokeWidth={2.4}
        />
      </g>
    );
  }

  // Open shapes: B, C, D, E, H
  const topY = cy - openH * 0.34;
  const botY = cy + openH * 0.66;
  const cornerY = cy + openH * 0.04;
  const ctrlCurve = base.roundness * 0.55;
  const topCtrlX = halfW * (1 - ctrlCurve);
  const clipId = `mouth-clip-${shape}`;

  const pathD =
    shape === "E"
      ? `M ${cx - halfW} ${cornerY} C ${cx - halfW} ${topY - 3}, ${cx + halfW} ${topY - 3}, ${cx + halfW} ${cornerY} C ${cx + halfW} ${botY + 3}, ${cx - halfW} ${botY + 3}, ${cx - halfW} ${cornerY} Z`
      : `M ${cx - halfW} ${cornerY} Q ${cx - topCtrlX} ${topY} ${cx} ${topY} Q ${cx + topCtrlX} ${topY} ${cx + halfW} ${cornerY} Q ${cx + halfW * 0.72} ${botY} ${cx} ${botY} Q ${cx - halfW * 0.72} ${botY} ${cx - halfW} ${cornerY} Z`;

  return (
    <g>
      <defs>
        <clipPath id={clipId}>
          <path d={pathD} />
        </clipPath>
      </defs>

      <path d={pathD} fill={cavityColor} />

      <g clipPath={`url(#${clipId})`}>
        {base.showUpperTeeth && (
          <rect
            x={cx - halfW}
            y={topY - 1}
            width={halfW * 2}
            height={shape === "B" ? openH * 0.55 : Math.min(6.2, openH * 0.34)}
            rx={2}
            fill="#ffffff"
            stroke={inkColor}
            strokeWidth={1.2}
          />
        )}

        {base.showLowerTeeth && (
          <rect
            x={cx - halfW * 0.85}
            y={botY - (shape === "B" ? openH * 0.45 : 4.2)}
            width={halfW * 1.7}
            height={shape === "B" ? openH * 0.5 : 5.2}
            rx={2}
            fill="#f7f7f7"
            stroke={inkColor}
            strokeWidth={1.1}
          />
        )}

        {base.showTongue && !base.tongueTipUp && (
          <ellipse
            cx={cx}
            cy={botY - 1}
            rx={halfW * 0.72}
            ry={Math.max(4.5, openH * 0.34)}
            fill={tongueColor}
          />
        )}

        {base.tongueTipUp && (
          <path
            d={`M ${cx - halfW * 0.65} ${botY + 2} Q ${cx - 4} ${topY + 4.5} ${cx} ${topY + 3.5} Q ${cx + 5} ${topY + 4.5} ${cx + halfW * 0.65} ${botY + 2} Z`}
            fill={tongueColor}
            stroke={inkColor}
            strokeWidth={1.3}
          />
        )}
      </g>

      <path
        d={pathD}
        fill="none"
        stroke={inkColor}
        strokeWidth={2.4}
        strokeLinejoin="round"
      />
    </g>
  );
}

export function CharacterStage({
  shape,
  style,
  onStyleChange,
  curves,
  lockedShape,
  onSelectPreviewShape,
  isLiveActive,
}: CharacterStageProps) {
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    let timeoutId: number;
    let blinkEndId: number;
    const scheduleBlink = () => {
      const delay = 2600 + Math.random() * 2600;
      timeoutId = window.setTimeout(() => {
        setIsBlinking(true);
        blinkEndId = window.setTimeout(() => {
          setIsBlinking(false);
          scheduleBlink();
        }, 130);
      }, delay);
    };
    scheduleBlink();
    return () => {
      window.clearTimeout(timeoutId);
      window.clearTimeout(blinkEndId);
    };
  }, []);

  const jawOffset = curves ? curves.jawOpen * 4.5 : shape === "D" ? 4 : shape === "C" || shape === "E" ? 2.5 : 0;
  const info = RHUBARB_DESCRIPTIONS[shape];

  return (
    <section className="panel stage-panel" aria-label="2D Character Stage">
      <div className="stage-toolbar">
        <div className="segmented" role="group" aria-label="Character Vector Style">
          <button
            type="button"
            className={`segmented-btn ${style === "editorial" ? "is-active" : ""}`}
            onClick={() => onStyleChange("editorial")}
          >
            Editorial Vector
          </button>
          <button
            type="button"
            className={`segmented-btn ${style === "indie" ? "is-active" : ""}`}
            onClick={() => onStyleChange("indie")}
          >
            Indie Rig
          </button>
        </div>

        {lockedShape ? (
          <button
            type="button"
            className="btn"
            onClick={() => onSelectPreviewShape(null)}
            title="Return character to audio / mic lip-sync"
          >
            Sync to Audio
          </button>
        ) : (
          <span className="muted-note">
            {isLiveActive ? "Live Mic Sync" : "Click a shape below to preview"}
          </span>
        )}
      </div>

      <div className="stage-canvas-wrap">
        <div className="stage-hud">
          <span className="shape-chip">
            {shape} · {info.label}
          </span>
          <span className="shape-chip">{info.phonemes}</span>
        </div>

        <svg
          className="stage-svg"
          viewBox="0 0 280 260"
          role="img"
          aria-label={`2D character mouth shape ${shape}: ${info.label}`}
        >
          {style === "editorial" ? (
            <g>
              {/* Soft backdrop disk */}
              <circle cx="140" cy="136" r="104" fill="#e6dfd1" stroke="#b9b09f" strokeWidth="1.5" />
              {/* Neck & collar */}
              <path d="M 118 194 L 118 228 L 162 228 L 162 194 Z" fill="#efc8ab" stroke="#221e1a" strokeWidth="2.4" />
              <path d="M 76 250 C 82 218, 108 214, 140 214 C 172 214, 198 218, 204 250 Z" fill="#2e6fa7" stroke="#221e1a" strokeWidth="2.4" />
              {/* Ears */}
              <ellipse cx="76" cy="136" rx="11" ry="16" fill="#efc8ab" stroke="#221e1a" strokeWidth="2.4" />
              <ellipse cx="204" cy="136" rx="11" ry="16" fill="#efc8ab" stroke="#221e1a" strokeWidth="2.4" />
              {/* Head */}
              <path
                d={`M 82 114 C 82 64, 198 64, 198 114 L 196 ${154 + jawOffset * 0.4} C 194 ${194 + jawOffset}, 166 ${206 + jawOffset}, 140 ${206 + jawOffset} C 114 ${206 + jawOffset}, 86 ${194 + jawOffset}, 84 ${154 + jawOffset * 0.4} Z`}
                fill="#f7d7be"
                stroke="#221e1a"
                strokeWidth="2.6"
              />
              {/* Editorial Hair */}
              <path
                d="M 78 112 C 74 56, 128 38, 168 48 C 198 56, 208 82, 202 118 C 192 94, 174 86, 148 86 C 118 86, 94 94, 78 112 Z"
                fill="#2c2724"
                stroke="#221e1a"
                strokeWidth="2.4"
                strokeLinejoin="round"
              />
              {/* Eyebrows */}
              <path d="M 101 107 Q 114 102 126 107" fill="none" stroke="#2c2724" strokeWidth="2.8" strokeLinecap="round" />
              <path d="M 154 107 Q 166 102 179 107" fill="none" stroke="#2c2724" strokeWidth="2.8" strokeLinecap="round" />
              {/* Eyes */}
              {isBlinking ? (
                <g stroke="#221e1a" strokeWidth="2.6" strokeLinecap="round">
                  <path d="M 104 126 Q 114 130 124 126" fill="none" />
                  <path d="M 156 126 Q 166 130 176 126" fill="none" />
                </g>
              ) : (
                <g>
                  <circle cx="114" cy="126" r="5.5" fill="#221e1a" />
                  <circle cx="116" cy="124" r="1.8" fill="#ffffff" />
                  <circle cx="166" cy="126" r="5.5" fill="#221e1a" />
                  <circle cx="168" cy="124" r="1.8" fill="#ffffff" />
                </g>
              )}
              {/* Nose */}
              <path d="M 140 126 L 136 144 L 145 145" fill="none" stroke="#221e1a" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
              {/* Cheek subtle tint */}
              <ellipse cx="99" cy="146" rx="8" ry="4" fill="#e58a77" opacity="0.32" />
              <ellipse cx="181" cy="146" rx="8" ry="4" fill="#e58a77" opacity="0.32" />

              <MouthRenderer
                shape={shape}
                curves={curves}
                cx={140}
                cy={169 + jawOffset * 0.25}
                inkColor="#221e1a"
                cavityColor="#4a1c20"
                tongueColor="#dc6a74"
                lipFill="#df8c84"
              />
            </g>
          ) : (
            <g>
              {/* Indie-game / anime stylized rig */}
              <rect x="28" y="24" width="224" height="224" rx="28" fill="#dce7f2" stroke="#9db6cd" strokeWidth="1.5" />
              {/* Back hair silhouette */}
              <path
                d="M 66 126 C 62 66, 102 36, 140 36 C 178 36, 218 66, 214 126 L 210 184 L 182 174 L 98 174 L 70 184 Z"
                fill="#29324b"
                stroke="#181d2c"
                strokeWidth="2.6"
                strokeLinejoin="round"
              />
              {/* Neck & hoodie */}
              <rect x="122" y="190" width="36" height="34" fill="#f3cbb1" stroke="#181d2c" strokeWidth="2.4" />
              <path d="M 74 248 C 82 214, 110 212, 140 212 C 170 212, 198 214, 206 248 Z" fill="#465c96" stroke="#181d2c" strokeWidth="2.6" />
              {/* Ears */}
              <ellipse cx="78" cy="140" rx="9" ry="14" fill="#f6d5be" stroke="#181d2c" strokeWidth="2.2" />
              <ellipse cx="202" cy="140" rx="9" ry="14" fill="#f6d5be" stroke="#181d2c" strokeWidth="2.2" />
              {/* Tapered chin anime face */}
              <path
                d={`M 82 110 C 82 66, 198 66, 198 110 L 195 ${152 + jawOffset * 0.35} C 190 ${184 + jawOffset * 0.7}, 158 ${204 + jawOffset}, 140 ${205 + jawOffset} C 122 ${204 + jawOffset}, 90 ${184 + jawOffset * 0.7}, 85 ${152 + jawOffset * 0.35} Z`}
                fill="#fce1ce"
                stroke="#181d2c"
                strokeWidth="2.6"
              />
              {/* Cel shadow under chin/hair */}
              <path d="M 88 106 Q 140 120 192 106 L 192 94 Q 140 84 88 94 Z" fill="#e6bca2" opacity="0.55" />
              {/* Expressive anime eyes */}
              {isBlinking ? (
                <g stroke="#181d2c" strokeWidth="3" strokeLinecap="round">
                  <path d="M 97 132 Q 112 138 126 132" fill="none" />
                  <path d="M 154 132 Q 168 138 183 132" fill="none" />
                </g>
              ) : (
                <g>
                  {/* Left eye */}
                  <ellipse cx="112" cy="130" rx="14" ry="13" fill="#ffffff" stroke="#181d2c" strokeWidth="2.2" />
                  <ellipse cx="114" cy="131" rx="8.5" ry="10.5" fill="#3a6bc4" />
                  <ellipse cx="114" cy="132" rx="4.5" ry="5.8" fill="#181d2c" />
                  <circle cx="110" cy="126" r="3" fill="#ffffff" />
                  <circle cx="118" cy="136" r="1.5" fill="#ffffff" />
                  <path d="M 96 125 Q 112 114 127 124" fill="none" stroke="#181d2c" strokeWidth="3.4" strokeLinecap="round" />

                  {/* Right eye */}
                  <ellipse cx="168" cy="130" rx="14" ry="13" fill="#ffffff" stroke="#181d2c" strokeWidth="2.2" />
                  <ellipse cx="166" cy="131" rx="8.5" ry="10.5" fill="#3a6bc4" />
                  <ellipse cx="166" cy="132" rx="4.5" ry="5.8" fill="#181d2c" />
                  <circle cx="162" cy="126" r="3" fill="#ffffff" />
                  <circle cx="170" cy="136" r="1.5" fill="#ffffff" />
                  <path d="M 153 124 Q 168 114 184 125" fill="none" stroke="#181d2c" strokeWidth="3.4" strokeLinecap="round" />
                </g>
              )}
              {/* Brows */}
              <path d="M 98 110 Q 112 104 125 110" fill="none" stroke="#181d2c" strokeWidth="2.3" strokeLinecap="round" />
              <path d="M 155 110 Q 168 104 182 110" fill="none" stroke="#181d2c" strokeWidth="2.3" strokeLinecap="round" />
              {/* Front bangs */}
              <path
                d="M 74 116 C 72 62, 114 42, 140 42 C 166 42, 208 62, 206 116 L 194 90 L 182 114 L 166 84 L 144 116 L 130 82 L 106 112 L 90 88 Z"
                fill="#354162"
                stroke="#181d2c"
                strokeWidth="2.4"
                strokeLinejoin="round"
              />
              {/* Nose & blush */}
              <path d="M 139 144 L 141 148" fill="none" stroke="#181d2c" strokeWidth="2" strokeLinecap="round" />
              <ellipse cx="97" cy="150" rx="9" ry="4.5" fill="#f07e78" opacity="0.34" />
              <ellipse cx="183" cy="150" rx="9" ry="4.5" fill="#f07e78" opacity="0.34" />

              <MouthRenderer
                shape={shape}
                curves={curves}
                cx={140}
                cy={171 + jawOffset * 0.28}
                inkColor="#181d2c"
                cavityColor="#5c1b28"
                tongueColor="#ee7483"
                lipFill="#eb9490"
              />
            </g>
          )}
        </svg>
      </div>

      <div className="shape-strip" role="group" aria-label="Rhubarb Mouth Shapes">
        {RHUBARB_SHAPES.map((code) => {
          const desc = RHUBARB_DESCRIPTIONS[code];
          const active = shape === code;
          return (
            <button
              key={code}
              type="button"
              className={`shape-btn ${active ? "is-active" : ""}`}
              aria-pressed={lockedShape === code}
              title={`${code}: ${desc.label} (${desc.phonemes})`}
              onClick={() => onSelectPreviewShape(lockedShape === code ? null : code)}
            >
              <span className="shape-btn-code">{code}</span>
              <span className="shape-btn-hint">{desc.phonemes.split(",")[0]}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
