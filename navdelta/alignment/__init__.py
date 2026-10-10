"""Shared alignment of the two recordings (used by every approach).

  align        per-recording RGB-D + ARKit/VIO pose graph, then gravity-locked
               registration of the second recording into the first one's frame
  qa_overlays  top-down and side overlays of both aligned rooms and trajectories
  validate     transform invariants, pose exports, floor agreement, artifact manifest
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

STAGES = [
    ("align", "Alignment  pose graphs + gravity-locked registration"),
    ("qa_overlays", "Alignment  QA overlays"),
    ("validate", "Alignment  validation + artifact manifest"),
]

# Multithreaded ICP and pose-graph optimisation sum in a varying order, so poses
# can differ between runs in the last bit (~1e-17 m). Results are compared at
# 1e-9 (a nanometre), far below anything downstream can resolve.
ROUND_DECIMALS = 9


def _digest(values) -> str:
    import numpy as np
    rounded = (np.round(np.asarray(values, dtype=np.float64), ROUND_DECIMALS) + 0.0).tolist()
    return hashlib.sha256(json.dumps(rounded).encode()).hexdigest()


def result_hashes(alignment: Path) -> dict[str, str]:
    """Hashes of the transform and of every camera pose, rounded to 1e-9."""
    hashes = {}
    for path in sorted(alignment.glob("*_to_*_transform.json")):
        hashes[f"alignment/{path.name}"] = _digest(json.loads(path.read_text())["T_source_to_target"])
    for path in sorted(alignment.glob("*_aligned_poses.jsonl")):
        poses = [json.loads(line)["T_camera_to_common"] for line in path.read_text().splitlines() if line.strip()]
        hashes[f"alignment/{path.name}"] = _digest(poses)
    return hashes
