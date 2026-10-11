"""Approach 4 - RGB-D fusion: scene graphs + loop-closure outliers on measured depth.

Uses the RGB-D variant of the shared reconstruction (navdelta.recon.reconstruct_rgbd: the same
pose graph with the recording's own depth maps instead of MASt3R depth). It runs approach 2 and
approach 3's detectors on that measured depth -- with tighter, range-scaled tolerances, as RGB-D
SLAM does -- and fuses them, as semantic RGB-D SLAM fuses instance masks with multi-view geometry:

  1  objects    scene graphs: Florence-2 + SAM 2.1 + SigLIP 2 objects lifted with sensor depth
  2  diff       each object node checked against the other video's depth (still there / seen through)
  3  mine       cross-video feature matches: inliers vs outliers; both ends vacated -> moved (one
                3D-3D RANSAC motion per object), one end vacated -> vanished / appeared
  4  segment    SAM 2.1 outlines and SigLIP 2 names for the outlier changes
  5  fuse       end-by-end association of the two: confirmed by both, or kept alone when strong
                (outliers alone are vetoed where the graph sees an unchanged object)
  6  package    changes.json, before/after photos, 3D viewer
"""
from __future__ import annotations

import json
from pathlib import Path

NAME = "rgbd_fusion"
TITLE = "RGB-D fusion: scene graphs + loop-closure outliers on the recording's depth maps"
REQUIRES = "recon_rgbd"
STAGES = [
    ("objects", "Step 1  objects: detections -> 3D nodes (sensor depth) -> scene graphs"),
    ("diff", "Step 2  graph diff, every node verified against the other video's depth"),
    ("mine", "Step 3  loop-closure outliers: moved / vanished / appeared"),
    ("segment", "Step 4  outline and name the outlier changes"),
    ("fuse", "Step 5  fuse both detectors: confirmed, or strong alone"),
    ("package", "Step 6  changes.json, photos, 3D viewer"),
]


def result_hashes(output: Path) -> dict[str, str | None]:
    def field(name: str) -> str | None:
        path = output / name
        return json.loads(path.read_text(encoding="utf-8")).get("result_sha256") if path.is_file() else None
    return {"step1_scene_graphs": field("scene_graphs.json"), "step5_changes_json": field("changes.json")}


def summary(output: Path) -> str | None:
    path = output / "changes.json"
    return json.loads(path.read_text(encoding="utf-8"))["summary"] if path.is_file() else None
