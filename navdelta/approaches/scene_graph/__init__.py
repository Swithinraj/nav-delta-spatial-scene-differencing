"""Approach 2 - 3D scene graphs from RGB, compared object by object.

Uses the shared reconstruction (navdelta.recon: MASt3R depth + the phone's tracking + a cross-video
pose graph), so it needs no LiDAR. Stages:

  1  objects    Florence-2 boxes + SAM 2.1 masks (+ a SAM point grid for things Florence-2 has no
                name for), lifted to 3D and merged into object nodes; edges "on" / "next to"
  2  diff       every node checked against the other video's depth: still there / gone / unseen;
                gone and appeared nodes (touching pieces grouped) are matched by SigLIP 2
                appearance -> moved, otherwise removed / added
  3  describe   sentences from the graphs: name, relations before / after, landmark distances
  4  package    changes.json, before/after photos, 3D viewer
"""
from __future__ import annotations

import json
from pathlib import Path

NAME = "scene_graph"
TITLE = "3D scene graphs from RGB (MASt3R + phone tracking), compared object by object"
REQUIRES = "recon"
STAGES = [
    ("objects", "Step 1  objects: detections -> 3D nodes -> scene graphs"),
    ("diff", "Step 2  graph diff, every claim verified by visibility"),
    ("describe", "Step 3  descriptions"),
    ("package", "Step 4  changes.json, photos, 3D viewer"),
]


def result_hashes(output: Path) -> dict[str, str | None]:
    def field(name: str) -> str | None:
        path = output / name
        return json.loads(path.read_text(encoding="utf-8")).get("result_sha256") if path.is_file() else None
    return {"step1_scene_graphs": field("scene_graphs.json"), "step3_changes_json": field("changes.json")}


def summary(output: Path) -> str | None:
    path = output / "changes.json"
    return json.loads(path.read_text(encoding="utf-8"))["summary"] if path.is_file() else None
