"""Exporters for Rhubarb (JSON/TSV/XML), Live2D/VTube/ARKit, Godot 4, and Blender."""

from __future__ import annotations

import json
from xml.sax.saxutils import escape

import numpy as np

from lipsync_2d.decode import MouthCue
from lipsync_2d.visemes import OCULUS_VISEMES, RHUBARB_SHAPES


def export_rhubarb_json(
    cues: list[MouthCue],
    sound_file: str,
    duration: float,
) -> str:
    """Format cues as Rhubarb-compatible JSON."""
    payload = {
        "metadata": {
            "soundFile": str(sound_file),
            "duration": round(float(duration), 4),
        },
        "mouthCues": [
            {
                "start": round(float(c.start), 2),
                "end": round(float(c.end), 2),
                "value": c.value,
            }
            for c in cues
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def export_rhubarb_tsv(cues: list[MouthCue]) -> str:
    """Format cues as Rhubarb-compatible TSV (start timestamp tab shape)."""
    lines = [f"{c.start:.2f}\t{c.value}" for c in cues]
    return "\n".join(lines) + ("\n" if lines else "")


def export_rhubarb_xml(
    cues: list[MouthCue],
    sound_file: str,
    duration: float,
) -> str:
    """Format cues as Rhubarb-compatible XML."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<rhubarbResult>",
        "  <metadata>",
        f"    <soundFile>{escape(str(sound_file))}</soundFile>",
        f"    <duration>{float(duration):.4f}</duration>",
        "  </metadata>",
        "  <mouthCues>",
    ]
    for c in cues:
        lines.append(
            f'    <mouthCue start="{c.start:.2f}" end="{c.end:.2f}">{escape(c.value)}</mouthCue>'
        )
    lines.extend(["  </mouthCues>", "</rhubarbResult>"])
    return "\n".join(lines) + "\n"


def export_live2d_json(
    viseme15_probs: np.ndarray,
    rhubarb9_probs: np.ndarray,
    mouth_curves: np.ndarray,
    frame_dt: float = 0.02,
) -> str:
    """Format 50 Hz predictions as canonical Live2D / VTube Studio JSON matching lipsync-2d-web."""
    v15 = np.asarray(viseme15_probs, dtype=np.float32)
    curves = np.asarray(mouth_curves, dtype=np.float32)
    n_frames = int(v15.shape[0])
    fps = round(1.0 / frame_dt, 3)
    duration = round(n_frames * frame_dt, 2)

    jaw_open: list[float] = []
    lip_round: list[float] = []
    lip_close: list[float] = []
    param_mouth_open_y: list[float] = []
    param_mouth_form: list[float] = []
    visemes: dict[str, list[float]] = {name: [] for name in OCULUS_VISEMES}

    for t in range(n_frames):
        jaw = float(np.clip(curves[t, 0], 0.0, 1.0))
        rnd = float(np.clip(curves[t, 1], 0.0, 1.0))
        cls = float(np.clip(curves[t, 2], 0.0, 1.0))
        spread = float(v15[t, 11]) + float(v15[t, 12])
        form = float(np.clip(spread - rnd * 0.9, -1.0, 1.0))

        param_mouth_open_y.append(round(jaw, 4))
        param_mouth_form.append(round(form, 4))
        jaw_open.append(round(jaw, 4))
        lip_round.append(round(rnd, 4))
        lip_close.append(round(cls, 4))
        for i, name in enumerate(OCULUS_VISEMES):
            visemes[name].append(round(float(v15[t, i]), 4))

    payload = {
        "version": 1,
        "fps": fps,
        "duration": duration,
        "frameCount": n_frames,
        "parameters": {
            "ParamMouthOpenY": param_mouth_open_y,
            "ParamMouthForm": param_mouth_form,
            "jawOpen": jaw_open,
            "lipRound": lip_round,
            "lipClose": lip_close,
        },
        "visemes": visemes,
    }
    return json.dumps(payload, indent=2) + "\n"


def export_godot_tres(
    cues: list[MouthCue],
    mouth_curves: np.ndarray | None = None,
    frame_dt: float = 0.02,
    anim_name: str = "lipsync",
    node_path: str = "Mouth",
) -> str:
    """Format cues and optional curves as a Godot 4 Animation .tres resource."""
    duration = cues[-1].end if cues else 0.0
    if mouth_curves is not None and len(mouth_curves) > 0:
        duration = max(duration, len(mouth_curves) * frame_dt)

    cue_times = ", ".join(f"{c.start:.4f}" for c in cues)
    cue_trans = ", ".join("1" for _ in cues)
    cue_vals = ", ".join(f'"{c.value}"' for c in cues)

    tracks = [
        f'[gd_resource type="Animation" format=3]',
        "",
        "[resource]",
        f'resource_name = "{anim_name}"',
        f"length = {duration:.4f}",
        "step = 0.02",
        "tracks/0/type = \"value\"",
        "tracks/0/imported = false",
        "tracks/0/enabled = true",
        f'tracks/0/path = NodePath("{node_path}:rhubarb_viseme")',
        "tracks/0/interp = 0",
        "tracks/0/loop_wrap = true",
        "tracks/0/keys = {",
        f'"times": PackedFloat32Array({cue_times}),',
        f'"transitions": PackedFloat32Array({cue_trans}),',
        '"update": 1,',
        f'"values": [{cue_vals}]',
        "}",
    ]

    if mouth_curves is not None and len(mouth_curves) > 0:
        curve_names = ("jaw_open", "lip_round", "lip_close")
        times = ", ".join(f"{i * frame_dt:.4f}" for i in range(len(mouth_curves)))
        trans = ", ".join("1" for _ in range(len(mouth_curves)))
        for idx, prop in enumerate(curve_names):
            t_idx = idx + 1
            vals = ", ".join(f"{float(mouth_curves[i, idx]):.4f}" for i in range(len(mouth_curves)))
            tracks.extend(
                [
                    f'tracks/{t_idx}/type = "value"',
                    f"tracks/{t_idx}/imported = false",
                    f"tracks/{t_idx}/enabled = true",
                    f'tracks/{t_idx}/path = NodePath("{node_path}:{prop}")',
                    f"tracks/{t_idx}/interp = 1",
                    f"tracks/{t_idx}/loop_wrap = true",
                    f"tracks/{t_idx}/keys = {{",
                    f'"times": PackedFloat32Array({times}),',
                    f'"transitions": PackedFloat32Array({trans}),',
                    '"update": 0,',
                    f'"values": [{vals}]',
                    "}",
                ]
            )

    return "\n".join(tracks) + "\n"


def export_blender_py(
    cues: list[MouthCue],
    viseme15_probs: np.ndarray | None = None,
    mouth_curves: np.ndarray | None = None,
    frame_dt: float = 0.02,
    fps: float = 24.0,
    object_name: str = "CharacterHead",
) -> str:
    """Generate a self-contained Blender Python keyframe script."""
    cue_data = [(round(c.start, 4), c.value) for c in cues]
    lines = [
        "import bpy",
        "",
        f"OBJ_NAME = {object_name!r}",
        f"FPS = {float(fps)}",
        f"FRAME_DT = {float(frame_dt)}",
        f"MOUTH_CUES = {cue_data!r}",
    ]
    if mouth_curves is not None:
        curves_list = [[round(float(x), 4) for x in row] for row in mouth_curves]
        lines.append(f"MOUTH_CURVES = {curves_list!r}")
    else:
        lines.append("MOUTH_CURVES = []")

    if viseme15_probs is not None:
        v15_names = list(OCULUS_VISEMES)
        v15_rows = [[round(float(x), 4) for x in row] for row in viseme15_probs]
        lines.append(f"VISEME15_NAMES = {v15_names!r}")
        lines.append(f"VISEME15_PROBS = {v15_rows!r}")
    else:
        lines.append("VISEME15_NAMES = []")
        lines.append("VISEME15_PROBS = []")

    lines.extend(
        [
            f"SHAPES = {list(RHUBARB_SHAPES)!r}",
            "SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}",
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
            "obj = bpy.data.objects.get(OBJ_NAME)",
            "if obj is None:",
            "    raise RuntimeError(f'Blender object {OBJ_NAME!r} not found')",
            "",
            "if 'rhubarb_viseme' not in obj:",
            "    obj['rhubarb_viseme'] = 0",
            "shape_keys = obj.data.shape_keys.key_blocks if obj.data and getattr(obj.data, 'shape_keys', None) else None",
            "for t_sec, shape in MOUTH_CUES:",
            "    frame = round(t_sec * FPS) + 1",
            "    obj['rhubarb_viseme'] = SHAPE_TO_IDX.get(shape, 0)",
            "    obj.keyframe_insert(data_path='[\"rhubarb_viseme\"]', frame=frame)",
            "    if shape_keys is not None:",
            "        for kb in shape_keys:",
            "            if kb.name.startswith('viseme_'):",
            "                kb.value = 1.0 if kb.name == f'viseme_{shape}' else 0.0",
            "                kb.keyframe_insert(data_path='value', frame=frame)",
            "",
            "if obj.animation_data and obj.animation_data.action:",
            "    for fc in _iter_action_fcurves(obj.animation_data.action):",
            "        if fc.data_path == '[\"rhubarb_viseme\"]':",
            "            for kp in fc.keyframe_points:",
            "                kp.interpolation = 'CONSTANT'",
            "",
            "for idx, (jaw_open, lip_round, lip_close) in enumerate(MOUTH_CURVES):",
            "    frame = round(idx * FRAME_DT * FPS) + 1",
            "    obj['jaw_open'] = jaw_open",
            "    obj['lip_round'] = lip_round",
            "    obj['lip_close'] = lip_close",
            "    obj.keyframe_insert(data_path='[\"jaw_open\"]', frame=frame)",
            "    obj.keyframe_insert(data_path='[\"lip_round\"]', frame=frame)",
            "    obj.keyframe_insert(data_path='[\"lip_close\"]', frame=frame)",
            "",
            "if shape_keys is not None and VISEME15_NAMES:",
            "    for idx, weights in enumerate(VISEME15_PROBS):",
            "        frame = round(idx * FRAME_DT * FPS) + 1",
            "        for name, w in zip(VISEME15_NAMES, weights):",
            "            if name in shape_keys:",
            "                shape_keys[name].value = w",
            "                shape_keys[name].keyframe_insert(data_path='value', frame=frame)",
        ]
    )
    return "\n".join(lines) + "\n"
