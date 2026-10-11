"""Shared reconstruction for the RGB approaches (scene_graph, loop_outliers).

  reconstruct  MASt3R depth for every keyframe + the phone's tracking + MASt3R matches between the
               videos, solved in one pose graph: both recordings in one drift-corrected frame

Needs only RGB keyframes and the phone's tracking poses (no LiDAR depth, no mesh); writes
<outputs>/recon/. voxel_diff uses navdelta.alignment instead.
"""
from __future__ import annotations

import json
from pathlib import Path

STAGES = [("reconstruct", "Reconstruction  MASt3R depth + cross-video pose graph")]
RGBD_STAGES = [("reconstruct_rgbd", "RGB-D reconstruction  sensor depth + cross-video pose graph")]


def result_hashes(recon: Path) -> dict[str, str | None]:
    report = recon / "recon_report.json"
    return {f"{recon.name}/scene": json.loads(report.read_text()).get("result_sha256") if report.is_file() else None}
