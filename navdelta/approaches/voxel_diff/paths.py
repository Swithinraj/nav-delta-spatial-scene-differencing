"""Folders and recording names used by every stage of the voxel_diff approach."""

from __future__ import annotations

from pathlib import Path

from navdelta import context
from navdelta.context import ALIGNMENT, DATA, inside

APPROACH = "voxel_diff"
OUTPUT = context.approach_dir(APPROACH)
SEMANTICS = OUTPUT / "semantics"
VIEWER = OUTPUT / "viewer"
RESOURCES = Path(__file__).resolve().parent / "resources"

BEFORE, AFTER = context.SESSIONS
SESSIONS = context.SESSIONS
# Dense coloured clouds of each recording in the common frame (from the alignment stage).
SCAN_CLOUDS = {BEFORE: f"{BEFORE}_optimized_rgbd_cloud.ply",
               AFTER: f"{AFTER}_aligned_to_{BEFORE}.ply"}

inside_workspace = inside


def out_rel(path: Path) -> str:
    """Path of a generated file relative to this approach's output folder ('/' separators)."""
    return Path(path).resolve().relative_to(OUTPUT).as_posix()


__all__ = ["ALIGNMENT", "AFTER", "APPROACH", "BEFORE", "DATA", "OUTPUT", "RESOURCES", "SCAN_CLOUDS",
           "SEMANTICS", "SESSIONS", "VIEWER", "inside_workspace", "out_rel"]
