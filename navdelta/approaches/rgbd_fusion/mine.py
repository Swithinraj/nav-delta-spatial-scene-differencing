"""rgbd_fusion, stage 3: loop-closure outliers (approach 3's mine stage) with measured depth.

As in RGB-D SLAM, every feature match takes its depth from the sensor, so:
  inlier             a depth-scaled gate on the 3D distance of the two ends (ORB-SLAM's chi-square
                     test with sigma(z) = a + b z), tighter than with MASt3R depth
  outlier            ends more than 25 cm apart, as with MASt3R depth: phone LiDAR is low-resolution
                     and smoothed at object borders, and a 15 cm gate floods a room with false
                     "appeared" surfaces (our room recordings: 15 candidates instead of 6)
  vacated            DynaSLAM-style depth consistency: project each end into the other video's
                     depth maps; seen through = gone from there
  motion             per-object 3D-3D RANSAC (yaw about gravity + translation) at sensor precision
"""
from __future__ import annotations

from navdelta import config, context
from navdelta.approaches.loop_outliers import mine as base

OVERRIDES = {
    "INLIER_M": 0.06,             # static: ends agree within max(6 cm, 2 % of depth)
    "INLIER_PER_M": 0.02,
    "OUTLIER_M": 0.25,            # an outlier: ends more than 25 cm apart (15 cm: false outliers at borders)
    "SEE_PAST_M": 0.08,
    "SEE_PAST_PER_M": 0.01,
    "SURFACE_M": 0.08,
    "SURFACE_PER_M": 0.01,
    "RANSAC_M": 0.05,             # one object's matches agree on its motion to 5 cm
}
vars(base).update(OVERRIDES)
base.RECON = context.OUTPUTS / "recon_rgbd"
base.OUTPUT = context.approach_dir("rgbd_fusion")
PROFILE_SETTINGS = config.apply(vars(base), "rgbd_fusion.mine")


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
