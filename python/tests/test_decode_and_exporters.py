"""Tests for Viterbi/hysteresis cue decoding and all 6 exporter formats."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

from lipsync_2d.decode import MouthCue, decode_cues
from lipsync_2d.exporters import (
    export_blender_py,
    export_godot_tres,
    export_live2d_json,
    export_rhubarb_json,
    export_rhubarb_tsv,
    export_rhubarb_xml,
)


def test_decode_cues_hysteresis_and_min_duration() -> None:
    # 20 frames (400 ms): silence [0..4], D [5..11] with 1-frame glitch C at 8, plosive A [12], E [13..15], silence [16..19]
    probs = np.full((20, 9), 0.01, dtype=np.float32)
    probs[0:5, 0] = 0.90  # X
    probs[5:12, 4] = 0.85  # D
    probs[8, 4] = 0.30
    probs[8, 3] = 0.60  # 1-frame spike -> merged out by min_duration_ms=40
    probs[12, 1] = 0.88  # 20 ms plosive A -> allowed via plosive_min_duration_ms=20
    probs[13:16, 5] = 0.86  # E
    probs[16:20, 0] = 0.90  # X

    cues = decode_cues(probs, switch_penalty=1.0, min_duration_ms=40.0, plosive_min_duration_ms=20.0)
    values = [c.value for c in cues]
    assert values == ["X", "D", "A", "E", "X"]
    assert cues[0].start == 0.0
    assert cues[-1].end == 0.4


def test_exporters_formats_validity() -> None:
    cues = [
        MouthCue(0.0, 0.1, "X"),
        MouthCue(0.1, 0.24, "D"),
        MouthCue(0.24, 0.4, "X"),
    ]
    v15_probs = np.full((20, 15), 1.0 / 15.0, dtype=np.float32)
    r9_probs = np.full((20, 9), 1.0 / 9.0, dtype=np.float32)
    curves = np.clip(np.linspace(0.0, 1.0, 60, dtype=np.float32).reshape(20, 3), 0.0, 1.0)

    # Rhubarb JSON
    j_str = export_rhubarb_json(cues, "sample.wav", 0.4)
    j_obj = json.loads(j_str)
    assert j_obj["metadata"]["soundFile"] == "sample.wav"
    assert j_obj["metadata"]["duration"] == 0.4
    assert len(j_obj["mouthCues"]) == 3
    assert j_obj["mouthCues"][1] == {"start": 0.1, "end": 0.24, "value": "D"}

    # Rhubarb TSV
    tsv_lines = export_rhubarb_tsv(cues).strip().splitlines()
    assert tsv_lines == ["0.00\tX", "0.10\tD", "0.24\tX"]

    # Rhubarb XML
    xml_str = export_rhubarb_xml(cues, "sample.wav", 0.4)
    root = ET.fromstring(xml_str)
    assert root.tag == "rhubarbResult"
    assert root.findtext("metadata/soundFile") == "sample.wav"
    xml_cues = root.findall("mouthCues/mouthCue")
    assert len(xml_cues) == 3
    assert xml_cues[1].attrib["start"] == "0.10"
    assert xml_cues[1].text == "D"

    # Live2D JSON (canonical schema matching lipsync-2d-web)
    l2d_obj = json.loads(export_live2d_json(v15_probs, r9_probs, curves))
    assert l2d_obj["version"] == 1
    assert l2d_obj["fps"] == 50.0
    assert l2d_obj["frameCount"] == 20
    assert len(l2d_obj["parameters"]["ParamMouthOpenY"]) == 20
    assert len(l2d_obj["parameters"]["ParamMouthForm"]) == 20
    assert len(l2d_obj["visemes"]) == 15

    # Godot 4 .tres
    tres = export_godot_tres(cues, mouth_curves=curves)
    assert '[gd_resource type="Animation" format=3]' in tres
    assert 'NodePath("Mouth:rhubarb_viseme")' in tres
    assert 'NodePath("Mouth:jaw_open")' in tres

    # Blender Python script
    bpy_src = export_blender_py(cues, viseme15_probs=v15_probs, mouth_curves=curves)
    compile(bpy_src, "blender_export.py", "exec")
    assert "import bpy" in bpy_src
