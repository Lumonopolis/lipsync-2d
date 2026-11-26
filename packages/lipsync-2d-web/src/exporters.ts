import {
  OCULUS_VISEMES,
  RHUBARB_SHAPES,
  type FramePrediction,
  type LipSyncCue,
  type RhubarbJsonOutput,
  type RhubarbShape,
} from "./constants.js";

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function escapeXml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function clamp11(v: number): number {
  return Math.min(1, Math.max(-1, v));
}

/**
 * Export cues in 100% Rhubarb Lip Sync compatible JSON schema.
 */
export function toRhubarbJson(
  cues: readonly LipSyncCue<RhubarbShape>[],
  duration: number,
  soundFile: string,
): string {
  const payload: RhubarbJsonOutput = {
    metadata: {
      soundFile,
      duration: round2(duration),
    },
    mouthCues: cues.map((cue) => ({
      start: round2(cue.start),
      end: round2(cue.end),
      value: cue.value,
    })),
  };
  return JSON.stringify(payload, null, 2);
}

/**
 * Export cues in official Rhubarb Lip Sync TSV format (start<TAB>value).
 */
export function toRhubarbTsv(cues: readonly LipSyncCue<RhubarbShape>[]): string {
  if (cues.length === 0) return "";
  return cues.map((cue) => `${cue.start.toFixed(2)}\t${cue.value}`).join("\n") + "\n";
}

/**
 * Export cues in Rhubarb Lip Sync XML format.
 */
export function toRhubarbXml(
  cues: readonly LipSyncCue<RhubarbShape>[],
  duration: number,
  soundFile: string,
): string {
  const lines: string[] = [
    '<?xml version="1.0" encoding="utf-8"?>',
    "<rhubarbResult>",
    "  <metadata>",
    `    <soundFile>${escapeXml(soundFile)}</soundFile>`,
    `    <duration>${duration.toFixed(2)}</duration>`,
    "  </metadata>",
    "  <mouthCues>",
  ];

  for (const cue of cues) {
    lines.push(
      `    <mouthCue start="${cue.start.toFixed(2)}" end="${cue.end.toFixed(2)}">${escapeXml(cue.value)}</mouthCue>`,
    );
  }

  lines.push("  </mouthCues>", "</rhubarbResult>");
  return lines.join("\n") + "\n";
}

export interface Live2DJsonOutput {
  version: number;
  fps: number;
  duration: number;
  frameCount: number;
  parameters: {
    ParamMouthOpenY: number[];
    ParamMouthForm: number[];
    jawOpen: number[];
    lipRound: number[];
    lipClose: number[];
  };
  visemes: Record<string, number[]>;
}

/**
 * Export 50 Hz frame predictions (resampled to target fps) for Live2D Cubism parameters.
 */
export function toLive2DJson(frames: readonly FramePrediction[], fps = 50): string {
  if (fps <= 0 || !Number.isFinite(fps)) {
    throw new RangeError(`Invalid fps: ${fps}`);
  }

  const duration =
    frames.length === 0
      ? 0
      : Math.max(
          (frames[frames.length - 1]?.time ?? 0) + 0.02,
          frames.length * 0.02,
        );
  const outCount = frames.length === 0 ? 0 : Math.max(1, Math.round(duration * fps));

  const paramMouthOpenY: number[] = [];
  const paramMouthForm: number[] = [];
  const jawOpen: number[] = [];
  const lipRound: number[] = [];
  const lipClose: number[] = [];
  const visemes: Record<string, number[]> = {};
  for (const v of OCULUS_VISEMES) {
    visemes[v] = [];
  }

  for (let i = 0; i < outCount; i++) {
    const t = i / fps;
    const srcIdx = Math.min(frames.length - 1, Math.max(0, Math.round(t * 50)));
    const f = frames[srcIdx]!;

    const jaw = clamp01(f.mouthCurves.jawOpen);
    const round = clamp01(f.mouthCurves.lipRound);
    const close = clamp01(f.mouthCurves.lipClose);
    const spread = (f.viseme15Probs[11] ?? 0) + (f.viseme15Probs[12] ?? 0);
    const form = clamp11(spread - round * 0.9);

    paramMouthOpenY.push(Math.round(jaw * 10000) / 10000);
    paramMouthForm.push(Math.round(form * 10000) / 10000);
    jawOpen.push(Math.round(jaw * 10000) / 10000);
    lipRound.push(Math.round(round * 10000) / 10000);
    lipClose.push(Math.round(close * 10000) / 10000);

    for (let vIdx = 0; vIdx < OCULUS_VISEMES.length; vIdx++) {
      const name = OCULUS_VISEMES[vIdx]!;
      visemes[name]!.push(Math.round((f.viseme15Probs[vIdx] ?? 0) * 10000) / 10000);
    }
  }

  const payload: Live2DJsonOutput = {
    version: 1,
    fps,
    duration: round2(duration),
    frameCount: outCount,
    parameters: {
      ParamMouthOpenY: paramMouthOpenY,
      ParamMouthForm: paramMouthForm,
      jawOpen,
      lipRound,
      lipClose,
    },
    visemes,
  };

  return JSON.stringify(payload, null, 2);
}

/**
 * Export cues as a valid Godot 4 Animation `.tres` resource.
 */
export function toGodotTres(
  cues: readonly LipSyncCue<RhubarbShape>[],
  duration: number,
  nodePath = "MouthSprite:animation",
): string {
  const safeDuration = Math.max(0.02, round2(duration));
  const times = cues.map((c) => round2(c.start).toString());
  const transitions = cues.map(() => "1");
  const values = cues.map((c) => JSON.stringify(c.value));

  return [
    '[gd_resource type="Animation" format=3]',
    "",
    "[resource]",
    'resource_name = "lipsync"',
    `length = ${safeDuration.toFixed(2)}`,
    "loop_mode = 0",
    "step = 0.02",
    'tracks/0/type = "value"',
    "tracks/0/imported = false",
    "tracks/0/enabled = true",
    `tracks/0/path = NodePath(${JSON.stringify(nodePath)})`,
    "tracks/0/interp = 0",
    "tracks/0/loop_wrap = true",
    "tracks/0/keys = {",
    `"times": PackedFloat32Array(${times.join(", ")}),`,
    `"transitions": PackedFloat32Array(${transitions.join(", ")}),`,
    '"update": 1,',
    `"values": [${values.join(", ")}]`,
    "}",
    "",
  ].join("\n");
}

/**
 * Export cues as a runnable Blender Python keyframe script.
 */
export function toBlenderPython(
  cues: readonly LipSyncCue<RhubarbShape>[],
  fps = 24,
  objectName = "MouthRig",
  propertyName = "mouth_shape",
): string {
  const keyframes = cues.map((c) => {
    const frame = Math.round(c.start * fps) + 1;
    const shapeIdx = RHUBARB_SHAPES.indexOf(c.value);
    return `    (${frame}, ${shapeIdx >= 0 ? shapeIdx : 0}, ${JSON.stringify(c.value)}),`;
  });

  return [
    "import bpy",
    "",
    `FPS = ${Math.round(fps)}`,
    `OBJECT_NAME = ${JSON.stringify(objectName)}`,
    `PROP_NAME = ${JSON.stringify(propertyName)}`,
    `SHAPES = ${JSON.stringify([...RHUBARB_SHAPES])}`,
    "CUES = [",
    ...keyframes,
    "]",
    "",
    "def _iter_action_fcurves(action):",
    "    if hasattr(action, 'fcurves'):",
    "        return list(action.fcurves)",
    "    out = []",
    "    for layer in getattr(action, 'layers', ()):",
    "        for strip in getattr(layer, 'strips', ()):",
    "            for bag in getattr(strip, 'channelbags', ()):",
    "                out.extend(getattr(bag, 'fcurves', ()))",
    "    return out",
    "",
    "def apply_lipsync():",
    "    scene = bpy.context.scene",
    "    scene.render.fps = FPS",
    "    obj = bpy.data.objects.get(OBJECT_NAME)",
    "    if obj is None:",
    '        raise LookupError(f"Blender object {OBJECT_NAME!r} not found")',
    "    if PROP_NAME not in obj:",
    "        obj[PROP_NAME] = 0",
    "    data_path = f'[\"{PROP_NAME}\"]'",
    "    for frame, shape_idx, _label in CUES:",
    "        obj[PROP_NAME] = shape_idx",
    "        obj.keyframe_insert(data_path=data_path, frame=frame)",
    "    if obj.animation_data and obj.animation_data.action:",
    "        for fcurve in _iter_action_fcurves(obj.animation_data.action):",
    "            if fcurve.data_path == data_path:",
    "                for kp in fcurve.keyframe_points:",
    "                    kp.interpolation = 'CONSTANT'",
    "",
    'if __name__ == "__main__":',
    "    apply_lipsync()",
    "",
  ].join("\n");
}
