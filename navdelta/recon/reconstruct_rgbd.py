"""RGB-D variant of the shared reconstruction (used by rgbd_fusion): the same pose graph, with the
recording's own depth maps (LiDAR / ToF / simulator) instead of MASt3R depth.

  depth    measured, metric; pixels on depth edges (mixed foreground / background, "flying pixels")
           are dropped, as RGB-D SLAM front ends do; no scale estimation (the scale is held fixed)
  matches  MASt3R's cross-video and loop-closure matches, reused from <outputs>/recon when the RGB
           reconstruction already ran (same keyframes), otherwise computed
  graph    tracking odometry + match reprojections + gravity, as in reconstruct.py; with measured
           depth the pair verification (3D-3D RANSAC) can be tighter
Output: <outputs>/recon_rgbd/ (same files as <outputs>/recon/).
"""
from __future__ import annotations

from navdelta import config, context
from navdelta.recon import reconstruct as base

OVERRIDES = {
    "SIGMA_SCALE": 0.005,          # measured depth: keep its scale
    "RANSAC_M": 0.05,              # measured depth: tighter 3D-3D pair verification
    "STEREO_GATE_M": 0.06,
}
vars(base).update(OVERRIDES)
base.DEPTH_SOURCE = "sensor"
base.OUT = context.OUTPUTS / "recon_rgbd"
base.MATCHES_FROM = context.OUTPUTS / "recon"
PROFILE_SETTINGS = config.apply(vars(base), "recon.reconstruct_rgbd")


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
