"""Approach 3 - changes are the outliers of the cross-video loop closure.

Uses the shared reconstruction (navdelta.recon), whose pose graph locks both videos together with
MASt3R matches. Robust SLAM discards the matches that disagree with that alignment; here they are
the signal. Stages:

  1  mine      lift every cross-video match into the common frame; check both ends of each outlier
               against the other video's depth: both vacated -> the surface moved (one rigid motion
               per object by sequential RANSAC: distance and turn from its own features); one end
               vacated -> it vanished / appeared
  2  segment   SAM 2.1 prompted with the outlier pixels outlines each changed object; SigLIP 2 names
               it; vanished + appeared look-alikes are one moved object; reflections and weak,
               uncorroborated evidence are dropped
  3  context   where: Florence-2 + SAM 2.1 on the same 1-2 photos per change -> the unchanged objects
               around it ("next to the desk") and how its distance to them changed
  4  package   changes.json, before/after photos, 3D viewer
Only the changed objects are segmented, so the change step takes seconds to minutes.
"""
from __future__ import annotations

import json
from pathlib import Path

NAME = "loop_outliers"
TITLE = "Changes as the outliers of the cross-video loop closure (MASt3R + pose graph)"
REQUIRES = "recon"
STAGES = [
    ("mine", "Step 1  mine the loop-closure outliers: moved / vanished / appeared"),
    ("segment", "Step 2  segment and name the changed objects"),
    ("context", "Step 3  where: relations to unchanged objects around each change"),
    ("package", "Step 4  changes.json, photos, 3D viewer"),
]


def result_hashes(output: Path) -> dict[str, str | None]:
    path = output / "changes.json"
    return {"step2_changes_json": json.loads(path.read_text(encoding="utf-8")).get("result_sha256")
            if path.is_file() else None}


def summary(output: Path) -> str | None:
    path = output / "changes.json"
    return json.loads(path.read_text(encoding="utf-8"))["summary"] if path.is_file() else None
