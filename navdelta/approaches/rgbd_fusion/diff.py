"""rgbd_fusion, stage 2: the scene-graph diff (approach 2's diff stage) with measured depth.

Each node's space is tested against the other recording's sensor depth (still there / seen
through / unseen). Measured depth has centimetre noise that grows with range (sigma(z) = a + b z),
so the gates are tighter than with MASt3R depth and scale with range.
"""
from __future__ import annotations

from navdelta import config, context
from navdelta.approaches.scene_graph import diff as base

OVERRIDES = {
    "SEE_PAST_M": 0.08,           # seen through: the other depth lies 8 cm + 1 cm/m behind the point
    "SEE_PAST_PER_M": 0.01,
    "SURFACE_M": 0.08,            # still there: within 8 cm + 1 cm/m
    "SURFACE_PER_M": 0.01,
}
vars(base).update(OVERRIDES)
base.RECON = context.OUTPUTS / "recon_rgbd"
base.OUTPUT = context.approach_dir("rgbd_fusion")
PROFILE_SETTINGS = config.apply(vars(base), "rgbd_fusion.diff")


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
