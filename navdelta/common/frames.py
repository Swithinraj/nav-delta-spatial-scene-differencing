"""Aligned keyframes of a recording, and projection of 3D points into them.

Shared by every approach: the alignment stage writes one JSON per keyframe
(<outputs>/alignment/aligned_frame_poses/<recording>/<frame>.json) with the
camera pose in the common frame, the intrinsics and the RGB/depth paths
(relative to the data folder).
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from navdelta.context import ALIGNMENT, DATA, inside

SURFACE_AGREEMENT_M = 0.10      # measured depth within this of a point: the camera sees that surface
SEE_PAST_MARGIN_M = 0.10        # measured depth beyond a point by this: the camera sees through its spot
DEPTH_RANGE_M = (0.25, 5.0)


def load_session(session: str) -> list[dict]:
    """All aligned keyframes of a recording, with depth in metres and the 4x4 pose."""
    pose_dir = inside(ALIGNMENT / "aligned_frame_poses" / session)
    frames = []
    for path in sorted(pose_dir.glob("*.json")):
        record = json.loads(path.read_text())
        record["depth_m"] = cv2.imread(str(inside(DATA / record["depth"])),
                                       cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
        record["pose"] = np.asarray(record["T_camera_to_common"], dtype=np.float64)
        frames.append(record)
    if not frames:
        raise RuntimeError(f"No aligned keyframes for {session} in {pose_dir}; run the alignment first.")
    return frames


def project(frame: dict, points: np.ndarray):
    """Pixel coordinates of world points and what the frame's depth says about them.

    Returns u, v, in_view, surface (depth agrees with the point) and see_past
    (depth lands clearly beyond the point, i.e. the spot is seen to be empty)."""
    pose = frame["pose"]
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    intr = frame["intrinsics"]
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr["fx"] * camera[:, 0] / z + intr["cx"]
        v = intr["fy"] * camera[:, 1] / z + intr["cy"]
    in_view = (z > 0.2) & (u >= 0) & (u < intr["width"]) & (v >= 0) & (v < intr["height"])
    depth = frame["depth_m"]
    measured = np.zeros(len(points), dtype=np.float32)
    if in_view.any():
        scale = depth.shape[1] / intr["width"]
        du = np.clip(np.round((u[in_view] + 0.5) * scale - 0.5).astype(int), 0, depth.shape[1] - 1)
        dv = np.clip(np.round((v[in_view] + 0.5) * scale - 0.5).astype(int), 0, depth.shape[0] - 1)
        measured[in_view] = depth[dv, du]
    valid = in_view & (measured >= DEPTH_RANGE_M[0]) & (measured <= DEPTH_RANGE_M[1])
    surface = valid & (np.abs(measured - z) < SURFACE_AGREEMENT_M)
    see_past = valid & (measured > z + SEE_PAST_MARGIN_M)
    return u, v, in_view, surface, see_past


def upright_turns(pose: np.ndarray) -> int:
    """np.rot90 turns (counter-clockwise) that make world +Z point to image-up."""
    up = pose[:3, :3].T @ np.array([0.0, 0.0, 1.0])     # world up in camera axes (x right, y down)
    if abs(up[1]) >= abs(up[0]):
        return 0 if up[1] < 0 else 2
    return 1 if up[0] > 0 else 3
