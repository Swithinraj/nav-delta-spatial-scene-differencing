"""rgbd_fusion, stage 1: scene graphs (approach 2's objects stage) on the RGB-D reconstruction.

Florence-2 + SAM 2.1 (+ point grid) + SigLIP 2 detections, lifted with the recording's measured
depth. The 2D detections of approach 2 are reused when it already ran on the same keyframes.
"""
from __future__ import annotations

import os
import shutil

from navdelta import config, context
from navdelta.approaches.scene_graph import objects as base

OVERRIDES: dict = {}              # same detectors and graph rules; only the depth differs
vars(base).update(OVERRIDES)
base.RECON = context.OUTPUTS / "recon_rgbd"
base.OUTPUT = context.approach_dir("rgbd_fusion")
PROFILE_SETTINGS = config.apply(vars(base), "rgbd_fusion.objects")


def main() -> None:
    base.OUTPUT.mkdir(parents=True, exist_ok=True)
    cached = context.approach_dir("scene_graph") / "detections_2d.npz"
    target = base.OUTPUT / "detections_2d.npz"
    if cached.is_file() and not target.exists() and os.environ.get("NAVDELTA_RECOMPUTE") != "1":
        shutil.copyfile(cached, target)          # same keyframes: the 2D detections are the same
    base.main()


if __name__ == "__main__":
    main()
