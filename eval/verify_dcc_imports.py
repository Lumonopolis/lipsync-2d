"""Verify exported Blender .py scripts in real headless Blender and Godot .tres in real headless Godot 4."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from lipsync_2d.cli import main as cli_main


def verify_blender_and_godot(wav_path: Path, onnx_path: Path) -> dict[str, object]:
    blender_bin = shutil.which("blender")
    godot_bin = shutil.which("godot")
    if not blender_bin or not godot_bin:
        raise RuntimeError(f"Missing DCC binary: blender={blender_bin}, godot={godot_bin}")

    root = Path(__file__).resolve().parents[1]
    pkg_dir = root / "packages" / "lipsync-2d-web"

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        py_blender = td_p / "py_mouth.py"
        py_godot = td_p / "py_mouth.tres"
        py_json = td_p / "py_cues.json"
        assert cli_main([str(wav_path), "-m", str(onnx_path), "-f", "blender", "-o", str(py_blender)]) == 0
        assert cli_main([str(wav_path), "-m", str(onnx_path), "-f", "godot", "-o", str(py_godot)]) == 0
        assert cli_main([str(wav_path), "-m", str(onnx_path), "-f", "json", "-o", str(py_json)]) == 0

        # Also generate TypeScript Blender & Godot exports from the exact same cues
        ts_blender = td_p / "ts_mouth.py"
        ts_godot = td_p / "ts_mouth.tres"
        ts_gen = td_p / "gen_ts.mjs"
        ts_gen.write_text(
            f"""
import fs from "node:fs";
import {{ toBlenderPython, toGodotTres }} from "{(pkg_dir / 'dist' / 'index.js').as_posix()}";
const data = JSON.parse(fs.readFileSync({json.dumps(str(py_json))}, "utf8"));
const cues = data.mouthCues.map(c => ({{ ...c, confidence: 1.0 }}));
fs.writeFileSync({json.dumps(str(ts_blender))}, toBlenderPython(cues, 24));
fs.writeFileSync({json.dumps(str(ts_godot))}, toGodotTres(cues, data.metadata.duration));
""",
            encoding="utf-8",
        )
        subprocess.run(["node", str(ts_gen)], check=True, cwd=str(pkg_dir))

        # 1. Execute BOTH Python and TS Blender scripts in real headless Blender
        blender_harness = td_p / "run_blender_check.py"
        blender_harness.write_text(
            f"""
import bpy, runpy

def iter_fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for bag in getattr(strip, "channelbags", ()):
                out.extend(getattr(bag, "fcurves", ()))
    return out

# Test Python CLI export on CharacterHead
bpy.ops.mesh.primitive_cube_add()
head = bpy.context.active_object
head.name = "CharacterHead"
runpy.run_path({json.dumps(str(py_blender))}, run_name="__main__")
fcs_py = iter_fcurves(head.animation_data.action)
paths_py = {{fc.data_path: len(fc.keyframe_points) for fc in fcs_py}}
assert '["rhubarb_viseme"]' in paths_py and paths_py['["rhubarb_viseme"]'] >= 2, f"Missing py keyframes: {{paths_py}}"
assert '["jaw_open"]' in paths_py, f"Missing jaw_open keyframes: {{paths_py}}"
for fc in fcs_py:
    if fc.data_path == '["rhubarb_viseme"]':
        assert all(kp.interpolation == "CONSTANT" for kp in fc.keyframe_points)

# Test TypeScript export on MouthRig
bpy.ops.mesh.primitive_cube_add()
rig = bpy.context.active_object
rig.name = "MouthRig"
runpy.run_path({json.dumps(str(ts_blender))}, run_name="__main__")
fcs_ts = iter_fcurves(rig.animation_data.action)
paths_ts = {{fc.data_path: len(fc.keyframe_points) for fc in fcs_ts}}
assert '["mouth_shape"]' in paths_ts and paths_ts['["mouth_shape"]'] >= 2, f"Missing ts keyframes: {{paths_ts}}"
for fc in fcs_ts:
    if fc.data_path == '["mouth_shape"]':
        assert all(kp.interpolation == "CONSTANT" for kp in fc.keyframe_points)

print("BLENDER_OK", paths_py, paths_ts)
""",
            encoding="utf-8",
        )
        b_res = subprocess.run(
            [blender_bin, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(blender_harness)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "BLENDER_OK" in b_res.stdout, b_res.stdout + b_res.stderr

        # 2. Load BOTH Python and TS Godot .tres animations in real headless Godot 4
        gd_dir = td_p / "godot_proj"
        gd_dir.mkdir()
        (gd_dir / "project.godot").write_text(
            'config_version=5\n\n[application]\nconfig/name="LipSyncCheck"\n',
            encoding="utf-8",
        )
        shutil.copy(py_godot, gd_dir / "py_mouth.tres")
        shutil.copy(ts_godot, gd_dir / "ts_mouth.tres")
        (gd_dir / "check.gd").write_text(
            """extends SceneTree
func _init():
    var a1 = ResourceLoader.load("res://py_mouth.tres")
    var a2 = ResourceLoader.load("res://ts_mouth.tres")
    if not (a1 is Animation) or not (a2 is Animation):
        push_error("Failed to load Animation .tres")
        quit(1)
        return
    if a1.get_track_count() < 4 or a1.track_get_key_count(0) < 2:
        push_error("Unexpected py Animation track/key counts")
        quit(1)
        return
    if a2.get_track_count() < 1 or a2.track_get_key_count(0) < 2:
        push_error("Unexpected ts Animation track/key counts")
        quit(1)
        return
    print("GODOT_OK py_tracks=", a1.get_track_count(), " py_keys=", a1.track_get_key_count(0), " ts_tracks=", a2.get_track_count(), " ts_keys=", a2.track_get_key_count(0))
    quit(0)
""",
            encoding="utf-8",
        )
        g_res = subprocess.run(
            [godot_bin, "--headless", "--path", str(gd_dir), "--script", "res://check.gd"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "GODOT_OK" in g_res.stdout, g_res.stdout + g_res.stderr

        return {
            "blender": [ln for ln in b_res.stdout.splitlines() if "BLENDER_OK" in ln][-1],
            "godot": [ln for ln in g_res.stdout.splitlines() if "GODOT_OK" in ln][-1],
        }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = verify_blender_and_godot(
        root / "web" / "public" / "samples" / "hello_studio.wav",
        root / "web" / "public" / "models" / "lipsync2d.onnx",
    )
    print(json.dumps(out, indent=2))
