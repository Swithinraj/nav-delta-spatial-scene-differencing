"""Viewer data for the RGB approaches (scene_graph, loop_outliers), in the format voxel_diff writes.

<output>/viewer/: index.html (the shared template), data/room_before.js, data/room_after.js,
data/changes.js, gallery/<id>_before.jpg, _after.jpg -- see docs/changes-schema.md ("Viewer data").
Rooms come from the shared reconstruction (MASt3R depth of every keyframe, back-projected).
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

TEMPLATE = Path(__file__).resolve().parents[1] / "viewer" / "index.html"
CONF_MIN = 1.5
DEPTH_MAX_M = 6.0


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def encode_points(points: np.ndarray, colors: np.ndarray | None = None) -> dict:
    """uint16-quantised positions (+ uint8 colours) as base64, decoded in the viewer."""
    low, high = points.min(axis=0), points.max(axis=0)
    scale = np.maximum(high - low, 1e-6) / 65535.0
    out = {"count": int(len(points)), "origin": low.round(5).tolist(), "scale": scale.tolist(),
           "positions": b64(np.round((points - low) / scale).astype("<u2"))}
    if colors is not None:
        out["colors"] = b64(np.clip(np.round(colors * 255), 0, 255).astype(np.uint8))
    return out


def downsample(points: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean point and colour per voxel (deterministic, numpy only)."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    mean = lambda v: np.stack([np.bincount(inverse, v[:, k]) / counts for k in range(v.shape[1])], 1)  # noqa: E731
    return mean(points), mean(colors)


def room_clouds(scene: dict, voxel: float, stride: int = 4) -> dict[str, dict]:
    """Each recording's coloured cloud from its keyframes' depth (common frame)."""
    out = {}
    for rec in dict.fromkeys(scene["owner"]):
        pts, cols = [], []
        for k in np.flatnonzero(scene["owner"] == rec):
            depth, K, T = scene["depth"][k], scene["K"][k], scene["cam2w"][k]
            mask = np.zeros(depth.shape, bool)
            mask[::stride, ::stride] = True
            mask &= (scene["conf"][k] > CONF_MIN) & (depth > 0.1) & (depth < DEPTH_MAX_M)
            v, u = np.nonzero(mask)
            z = depth[v, u]
            p = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
            pts.append(p @ T[:3, :3].T + T[:3, 3])
            cols.append(scene["rgb"][k][v, u] / 255.0)
        p, c = downsample(np.concatenate(pts), np.concatenate(cols), voxel)
        out[rec] = encode_points(p, c)
    return out


def photo(scene: dict, rec: str, points: np.ndarray, path: Path, colour: tuple[int, int, int]) -> str | None:
    """Crop of the keyframe that sees most of `points`, the object outlined; returns the gallery name."""
    best = None
    for k in np.flatnonzero(scene["owner"] == rec):
        w2c = np.linalg.inv(scene["cam2w"][k])
        cam = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        K = scene["K"][k]
        h, w = scene["depth"][k].shape
        ok = z > 0.2
        u = K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]
        v = K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if inside.sum() < 0.6 * len(points):
            continue
        # visible, not just in the picture: where the frame has depth, it reaches the point (nothing in
        # front); pixels without a depth measurement (0: sensor holes, filtered edges) are unknown
        ui, vi = u[inside].astype(int), v[inside].astype(int)
        measured = scene["depth"][k][vi, ui]
        valid = measured > 0
        seen = valid & (np.abs(measured - z[ok][inside]) < 0.15 + 0.03 * z[ok][inside])
        if valid.sum() < 0.3 * len(points) or seen.sum() < 0.5 * valid.sum():
            continue
        # prefer views that see it whole, near the image centre
        score = seen.mean() - 0.5 * np.hypot(u[inside].mean() / w - 0.5, v[inside].mean() / h - 0.5)
        if best is None or score > best[0]:
            best = (score, k, u[inside], v[inside])
    if best is None:
        return None
    _, k, u, v = best
    img = cv2.cvtColor(scene["rgb"][k], cv2.COLOR_RGB2BGR).copy()
    hull = cv2.convexHull(np.column_stack([u, v]).astype(np.int32))
    cv2.polylines(img, [hull], True, colour, 2)
    x0, y0, x1, y1 = u.min(), v.min(), u.max(), v.max()
    pad = 0.6 * max(x1 - x0, y1 - y0, 60)
    h, w = img.shape[:2]
    crop = img[int(max(0, y0 - pad)):int(min(h, y1 + pad)), int(max(0, x0 - pad)):int(min(w, x1 + pad))]
    up = scene["cam2w"][k][:3, :3].T @ np.array([0.0, 0.0, 1.0])     # world up in the camera: rotate upright
    turns = (0 if up[1] < 0 else 2) if abs(up[1]) >= abs(up[0]) else (1 if up[0] > 0 else 3)
    crop = np.ascontiguousarray(np.rot90(crop, turns))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path.name


def write(out: Path, scene: dict, summary: str, changes: list[dict], landmarks: list[dict], voxel: float,
          extra: dict | None = None) -> None:
    """changes: viewer records with "parts" holding {"role", "points" (N x 3), "colors" (N x 3)}."""
    data = out / "viewer" / "data"
    data.mkdir(parents=True, exist_ok=True)
    recs = list(dict.fromkeys(scene["owner"]))
    rooms = room_clouds(scene, voxel)
    for rec, key in zip(recs, ("before", "after")):
        (data / f"room_{key}.js").write_text(f"window.ROOM_CLOUDS = window.ROOM_CLOUDS || {{}};\n"
                                             f"window.ROOM_CLOUDS[{json.dumps(key)}] = {json.dumps(rooms[rec])};\n",
                                             encoding="utf-8")
    referenced = {Path(v).name for c in changes for v in (c.get("photos") or {}).values() if v}
    gallery = out / "viewer" / "gallery"
    if gallery.is_dir():                     # photos of earlier runs that this run did not make
        for f in gallery.iterdir():
            if f.is_file() and f.name not in referenced:
                f.unlink()
    viewer_changes = []
    for c in changes:
        parts = [{"role": p["role"], "object": p.get("object", c["id"]),
                  **encode_points(p["points"], p.get("colors"))} for p in c["parts"] if len(p["points"])]
        viewer_changes.append({**{k: v for k, v in c.items() if k != "parts"}, "parts": parts})
    lows = np.array([r["origin"] for r in rooms.values()])
    highs = np.array([np.array(r["origin"]) + np.array(r["scale"]) * 65535 for r in rooms.values()])
    viewer = {"summary": summary, "recordings": {"before": recs[0], "after": recs[1]}, "floor_z": 0.0,
              "bounds": [lows.min(0).round(3).tolist(), highs.max(0).round(3).tolist()],
              "landmarks": landmarks, "changes": viewer_changes, **(extra or {})}
    (data / "changes.js").write_text(f"window.CHANGE_DATA = {json.dumps(viewer, ensure_ascii=False)};\n",
                                     encoding="utf-8")
    shutil.copyfile(TEMPLATE, out / "viewer" / "index.html")
