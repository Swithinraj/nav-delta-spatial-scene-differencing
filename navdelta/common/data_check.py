"""Check that a data folder holds two recordings in the expected layout (docs/data-format.md)."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_POSE_KEYS = ("t_00", "t_01", "t_02", "t_03", "t_10", "t_11", "t_12", "t_13",
                      "t_20", "t_21", "t_22", "t_23", "fx", "fy", "cx", "cy", "width", "height")


def check_recording(data: Path, name: str) -> tuple[list[str], dict]:
    """Problems found for one recording (empty list = usable) and a short summary."""
    problems: list[str] = []
    keyframes = data / f"{name}_imgs" / "keyframes"
    raw = data / f"{name}_rawdata"
    ply = data / f"{name}_ply" / f"{name}.ply"
    folders = {"corrected_cameras": keyframes / "corrected_cameras",
               "corrected_images": keyframes / "corrected_images",
               "corrected_depth": keyframes / "corrected_depth",
               "raw cameras": raw / "keyframes" / "cameras"}
    for label, folder in folders.items():
        if not folder.is_dir():
            problems.append(f"missing folder {folder}")
    for required in (raw / "mesh_info.json", ply):
        if not required.is_file():
            problems.append(f"missing file {required}")
    if problems:
        return problems, {}

    stems = sorted(p.stem for p in folders["corrected_cameras"].glob("*.json"))
    if not stems:
        problems.append(f"no camera JSON files in {folders['corrected_cameras']}")
    for stem in stems:
        if not stem.isdigit():
            problems.append(f"frame name {stem!r} is not an integer timestamp (microseconds)")
            break
    missing_rgb = [s for s in stems if not (folders["corrected_images"] / f"{s}.jpg").is_file()]
    missing_depth = [s for s in stems if not (folders["corrected_depth"] / f"{s}.Clean.png").is_file()]
    missing_raw = [s for s in stems if not (folders["raw cameras"] / f"{s}.json").is_file()]
    for label, missing in (("corrected_images/<frame>.jpg", missing_rgb),
                           ("corrected_depth/<frame>.Clean.png", missing_depth),
                           ("rawdata/keyframes/cameras/<frame>.json", missing_raw)):
        if missing:
            problems.append(f"{len(missing)} frames without {label} (first: {missing[0]})")
    if stems:
        camera = json.loads((folders["corrected_cameras"] / f"{stems[0]}.json").read_text())
        absent = [key for key in REQUIRED_POSE_KEYS if key not in camera]
        if absent:
            problems.append(f"camera JSON lacks keys {absent}")
    mesh_info = json.loads((raw / "mesh_info.json").read_text())
    transform = mesh_info.get("alignmentTransform")
    if not isinstance(transform, list) or len(transform) != 16:
        problems.append("mesh_info.json needs 'alignmentTransform': 16 numbers (4x4, column-major)")
    return problems, {"frames": len(stems), "ply_bytes": ply.stat().st_size}


def check_data(data: Path, names: list[str]) -> bool:
    ok = True
    print(f"Data folder: {data}")
    if len(names) != 2:
        print(f"  expected two recordings, found {names or 'none'} "
              "(a recording is a <name>_imgs folder with a keyframes subfolder)")
        return False
    for role, name in zip(("before", "after"), names):
        problems, summary = check_recording(data, name)
        if problems:
            ok = False
            print(f"  {role:<6} {name}: NOT USABLE")
            for problem in problems:
                print(f"           - {problem}")
        else:
            print(f"  {role:<6} {name}: ok, {summary['frames']} keyframes, "
                  f"PLY {summary['ply_bytes'] / 1e6:.1f} MB")
    return ok
