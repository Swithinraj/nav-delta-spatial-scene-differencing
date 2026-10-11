"""Approach 1 - visibility-aware voxel differencing + open-vocabulary labelling.

Stages run in this order (each is a module with a ``main()``):

  1  change_map   ray-cast both recordings into one 5 cm voxel grid; a voxel is
                  "added"/"removed" only if one recording saw it occupied and the
                  other saw it empty (unobserved space stays unknown)
     qa_regions   every change region in real before/after photos
  2  objects      group regions into objects; pair removed with added objects
                  (shape, height, colour; Hungarian); move vs. small push
     qa_objects   every object change in real photos
  3  labels       SAM 2.1 masks, Florence-2 names, SigLIP 2 cross-check, and a
                  photo check of the same spot in the other recording
  4  landmarks    static furniture (Florence-2 <OD> + SAM 2.1, voted in 3D)
     colours      one colour word per object (SmolVLM2-500M), optional
     describe     template sentences filled from the measurements
  5  package      changes.json, before/after photo pairs, 3D viewer
"""

from __future__ import annotations

import json
from pathlib import Path

NAME = "voxel_diff"
REQUIRES = "alignment"
TITLE = "Visibility-aware voxel differencing + open-vocabulary labelling"

STAGES = [
    ("change_map", "Step 1  voxel change map"),
    ("qa_regions", "Step 1  QA photos of change regions"),
    ("objects", "Step 2  objects and moves"),
    ("qa_objects", "Step 2  QA photos of object changes"),
    ("labels", "Step 3  open-vocabulary labels"),
    ("landmarks", "Step 4a static landmarks"),
    ("colours", "Step 4c colour words (small VLM)"),
    ("describe", "Step 4b natural-language descriptions"),
    ("package", "Step 5  changes.json, photo gallery, 3D viewer"),
]


def result_hashes(output: Path) -> dict[str, str]:
    """The result hash each stage records, for reproducibility checks."""
    def field(relative: str, *keys: str) -> str | None:
        path = output / relative
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            value = value[key]
        return value

    return {
        "step1_change_mask": field("change_detection_report.json", "change_voxels", "change_mask_sha256"),
        "step2_objects_and_moves": field("object_changes.json", "result_sha256"),
        "step3_labels": field("semantics/object_labels.json", "result_sha256"),
        "step4a_landmarks": field("semantics/landmarks.json", "result_sha256"),
        "step4c_colours": field("semantics/appearance.json", "result_sha256"),
        "step4b_descriptions": field("change_descriptions.json", "result_sha256"),
        "step5_changes_json": field("changes.json", "result_sha256"),
    }


def summary(output: Path) -> str | None:
    path = output / "change_descriptions.json"
    return json.loads(path.read_text(encoding="utf-8"))["summary"] if path.is_file() else None
