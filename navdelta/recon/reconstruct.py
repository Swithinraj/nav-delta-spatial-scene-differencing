"""Shared reconstruction for approaches 2 and 3: one metric 3D scene of both recordings from
RGB keyframes + the phone's own tracking (no LiDAR).

  poses   the phone's visual-inertial tracking (ARKit / ARCore), saved with every keyframe: metric
          and gravity-aligned, but each recording drifts on its own (10-30 cm across a room), so
          no single rigid transform lines the two recordings up everywhere
  depth   MASt3R on each keyframe paired with a neighbour ~0.3 m away: the dense point map gives
          the depth shape; its matches, triangulated with the tracked poses, give the metric
          scale; a multi-view consistency check removes floaters
  graph   MASt3R matches on SigLIP-chosen pairs -- across the recordings and loop closures within
          each -- become constraints; with the tracking as odometry they go into one robust pose
          graph that also refines each frame's depth scale. Pairs are verified first (4-DOF RANSAC:
          both worlds share gravity), so matches on moved objects do not pull the recordings
          together. Every 5th cross pair is held out to grade the result.

Inputs  <data>/<name>_imgs/keyframes/images/<frame>.jpg  and  .../cameras/<frame>.json (raw tracking)
Outputs <outputs>/recon/: scene.npz (both recordings in the first one's frame, Z up, floor at z = 0),
        cross_matches.npz (every cross-video MASt3R match: approach 3 mines its outliers),
        depth / loop caches, recon_report.json
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
import torch
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from navdelta import config, context
from navdelta.recon import mast3r_src

DATA = context.DATA
OUT = context.OUTPUTS / "recon"
DEPTH_SOURCE = "mast3r"         # "sensor": the recording's own depth maps (RGB-D mode, navdelta.recon.reconstruct_rgbd)
MATCHES_FROM: Path | None = None  # reuse another reconstruction's MASt3R matches and pairs (same keyframes)
SENSOR_DEPTH_RANGE_M = (0.25, 5.0)
SENSOR_EDGE_M, SENSOR_EDGE_PER_M = 0.05, 0.02   # depth edges: a 3x3 patch spanning more is mixed ("flying") pixels
RECOMPUTE = os.environ.get("NAVDELTA_RECOMPUTE") == "1"     # `navdelta verify`: ignore every cache
SIGLIP = ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f")
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
YUP_TO_ZUP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
SEED = 0
KEYFRAME_EVERY = 2
MIN_SHARPNESS = 100.0       # variance of the Laplacian at 512x384: blank walls / ceiling score far below
CONF_MIN = 1.5
# depth
BASELINE_M = 0.30           # preferred stereo baseline for the depth partner
MIN_BASELINE_M = 0.10
MAX_VIEW_ANGLE = 35.0
PARTNER_WINDOW = 8
MIN_PARALLAX_DEG = 2.0
REPROJ_PX = 2.0
MIN_TRIANGULATED = 30
MV_REL, MV_ABS, MV_NEIGHBOURS = 0.05, 0.05, 3
# pairs
CROSS_K = 2                 # SigLIP nearest frames of the other recording, both ways
LOOP_K = 1                  # SigLIP nearest frame of the same recording, outside the time gap
LOOP_GAP = 20               # keyframes
RANSAC_M = 0.08
RANSAC_ITERS = 1000
PAIR_MIN_INLIERS = 40
PAIR_MIN_RATIO = 0.25
PAIR_MAX_YAW_DEG = 10.0     # w.r.t. the initial alignment
PAIR_GATE_M = 0.75          # drift is tens of cm, not metres
STEREO_GATE_M = 0.10
POINTS_PER_PAIR = 60
HOLDOUT_EVERY = 5
# graph (residuals in units of their sigma)
SIGMA_ODOM_T, SIGMA_ODOM_T_PER_M = 0.01, 0.03
SIGMA_ODOM_R = np.radians(0.3)
SIGMA_PX = 3.0              # match reprojection, pixels at 512 x 384
SIGMA_GRAVITY = np.radians(0.5)
SIGMA_SCALE = 0.05
ROBUST_SCALE = 2.0

# Scale profile (navdelta/configs/*.toml): the values above are the room profile.
PROFILE_SETTINGS = config.apply(globals(), "recon.reconstruct")


# ---------------------------------------------------------------- inputs
def sharpness(path: Path) -> float:
    gray = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (512, 384), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def keyframe_folder(rec: str, kind: str) -> Path:
    keyframes = DATA / f"{rec}_imgs" / "keyframes"
    for name in {"images": ("images", "corrected_images"), "cameras": ("cameras",)}[kind]:
        if (keyframes / name).is_dir():
            return keyframes / name
    if kind == "cameras" and (DATA / f"{rec}_rawdata" / "keyframes" / "cameras").is_dir():
        return DATA / f"{rec}_rawdata" / "keyframes" / "cameras"
    raise SystemExit(f"{rec}: no keyframes/{kind} folder (see docs/data-format.md)")


def load_recording(rec: str, every: int) -> dict:
    images, cameras = keyframe_folder(rec, "images"), keyframe_folder(rec, "cameras")
    frames = sorted(images.glob("*.jpg"))
    kept = [p for p in frames if sharpness(p) >= MIN_SHARPNESS][::every]
    print(f"{rec}: {len(kept)} of {len(frames)} keyframes (texture filter, every {every})", flush=True)
    cams = [json.loads((cameras / f"{p.stem}.json").read_text()) for p in kept]
    K = []
    for c in cams:
        s = 512.0 / c["width"]
        K.append(np.array([[c["fx"] * s, 0, (c["cx"] + 0.5) * s - 0.5],
                           [0, c["fy"] * s, (c["cy"] + 0.5) * s - 0.5], [0, 0, 1]]))
    return {"name": rec, "paths": kept, "cam2w": np.stack([camera_pose(c) for c in cams]), "K": np.stack(K)}


@torch.no_grad()
def global_descriptors(paths: list[Path]) -> np.ndarray:
    """SigLIP 2 image embeddings: which frames show the same place (pair candidates)."""
    from transformers import AutoModel, AutoProcessor
    from navdelta.common.device import DEVICE, DTYPE
    processor = AutoProcessor.from_pretrained(SIGLIP[0], revision=SIGLIP[1])
    model = AutoModel.from_pretrained(SIGLIP[0], revision=SIGLIP[1], dtype=DTYPE).to(DEVICE).eval()
    feats = []
    for start in range(0, len(paths), 8):
        batch = [Image.open(p).convert("RGB") for p in paths[start:start + 8]]
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        f = model.get_image_features(pixel_values=inputs["pixel_values"].to(DTYPE))
        f = getattr(f, "pooler_output", f)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


# ---------------------------------------------------------------- MASt3R
class Mast3r:
    """MASt3R (fp16 with fp32 heads on CUDA; fits a 4 GB GPU), loaded on first use so cached runs need no model."""

    def __init__(self) -> None:
        self.model = None

    @torch.no_grad()
    def __call__(self, path1: Path, path2: Path, both: bool = False) -> list[dict]:
        from navdelta.common.device import DEVICE
        AsymmetricMASt3R, inference, load_images, fast_reciprocal_NNs = mast3r_src.load()
        if self.model is None:
            model = AsymmetricMASt3R.from_pretrained(mast3r_src.WEIGHTS)
            if DEVICE == "cuda":
                model = model.half()
                model.downstream_head1.float()
                model.downstream_head2.float()
            self.model = model.to(DEVICE).eval()
        imgs = load_images([str(path1), str(path2)], size=512, verbose=False)
        if DEVICE == "cuda":
            for img in imgs:
                img["img"] = img["img"].half()
        pairs = [(imgs[0], imgs[1])] + ([(imgs[1], imgs[0])] if both else [])
        out = inference(pairs, self.model, DEVICE, batch_size=1, verbose=False)
        results = []
        for b in range(len(pairs)):
            pts = out["pred1"]["pts3d"][b].float().cpu().numpy()
            conf = out["pred1"]["conf"][b].float().cpu().numpy()
            d1, d2 = out["pred1"]["desc"][b].float(), out["pred2"]["desc"][b].float()
            xy1, xy2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=8, device=DEVICE, dist="dot",
                                           block_size=2 ** 13)
            h, w = pts.shape[:2]
            ok = (xy1[:, 0] >= 3) & (xy1[:, 0] < w - 3) & (xy1[:, 1] >= 3) & (xy1[:, 1] < h - 3) & \
                 (xy2[:, 0] >= 3) & (xy2[:, 0] < w - 3) & (xy2[:, 1] >= 3) & (xy2[:, 1] < h - 3)
            rgb = (imgs[b]["img"][0].float().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            results.append({"depth": pts[..., 2], "conf": conf,
                            "xy": np.hstack([xy1[ok], xy2[ok]]).astype(np.int16),
                            "rgb": (rgb * 255).astype(np.uint8)})
        return results

    def release(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def camera_pose(c: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = [[c[f"t_{r}{k}"] for k in range(3)] for r in range(3)]
    T[:3, 3] = [c["t_03"], c["t_13"], c["t_23"]]
    return YUP_TO_ZUP @ T @ CV_TO_ARKIT


def view_angle(Ta: np.ndarray, Tb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(Ta[:3, 2] @ Tb[:3, 2], -1, 1))))


def partners(cam2w: np.ndarray) -> list[tuple[int, int]]:
    """For every frame, the best forward and backward neighbour for stereo: ~0.3 m apart, similar view."""
    pairs = set()
    n = len(cam2w)
    for i in range(n):
        for direction in (1, -1):
            best, best_cost = None, None
            for k in range(1, PARTNER_WINDOW + 1):
                j = i + direction * k
                if not 0 <= j < n:
                    break
                b = float(np.linalg.norm(cam2w[i][:3, 3] - cam2w[j][:3, 3]))
                if view_angle(cam2w[i], cam2w[j]) > MAX_VIEW_ANGLE:
                    continue
                cost = abs(b - BASELINE_M) + (1.0 if b < MIN_BASELINE_M else 0.0)
                if best_cost is None or cost < best_cost:
                    best, best_cost = j, cost
            if best is not None:
                pairs.add((min(i, best), max(i, best)))
        if not any(i in p for p in pairs) and n > 1:      # no partner in range: the nearest neighbour
            pairs.add((i, i + 1) if i + 1 < n else (i - 1, i))
    return sorted(pairs)


def triangulate_scale(depth, conf, xy, Ki, Kj, Ti, Tj) -> tuple[float | None, int]:
    """Metric scale of MASt3R's depth for view 1: triangulate its matches with the tracked poses."""
    rel = np.linalg.inv(Tj) @ Ti                         # camera i -> camera j
    x1, x2 = xy[:, :2].astype(np.float64), xy[:, 2:].astype(np.float64)
    if len(x1) < MIN_TRIANGULATED:
        return None, 0
    X = cv2.triangulatePoints(Ki @ np.hstack([np.eye(3), np.zeros((3, 1))]), Kj @ rel[:3], x1.T, x2.T)
    X = (X[:3] / X[3]).T
    Xj = X @ rel[:3, :3].T + rel[:3, 3]
    ok = (X[:, 2] > 0.1) & (Xj[:, 2] > 0.1)
    r1 = np.linalg.norm((X @ Ki.T)[:, :2] / np.maximum(X[:, 2:], 1e-9) - x1, axis=1)
    r2 = np.linalg.norm((Xj @ Kj.T)[:, :2] / np.maximum(Xj[:, 2:], 1e-9) - x2, axis=1)
    centre_j = -rel[:3, :3].T @ rel[:3, 3]               # camera j centre in camera i
    a = X / np.linalg.norm(X, axis=1, keepdims=True)
    b = (X - centre_j) / np.linalg.norm(X - centre_j, axis=1, keepdims=True)
    parallax = np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))
    u, v = xy[:, 0].astype(np.int64), xy[:, 1].astype(np.int64)
    ok &= (r1 < REPROJ_PX) & (r2 < REPROJ_PX) & (parallax > MIN_PARALLAX_DEG) & (conf[v, u] > CONF_MIN)
    if ok.sum() < MIN_TRIANGULATED:
        return None, int(ok.sum())
    return float(np.median(X[ok, 2] / depth[v[ok], u[ok]])), int(ok.sum())


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.dstack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth])


def multiview_filter(depth, conf, K, cam2w) -> np.ndarray:
    """Keep pixels whose depth agrees with at least one neighbouring keyframe's depth."""
    n = len(depth)
    keep = np.zeros(depth.shape, bool)
    for i in range(n):
        world = backproject(depth[i], K[i]).reshape(-1, 3) @ cam2w[i][:3, :3].T + cam2w[i][:3, 3]
        agree = np.zeros(world.shape[0], bool)
        for j in range(max(0, i - MV_NEIGHBOURS), min(n, i + MV_NEIGHBOURS + 1)):
            if j == i:
                continue
            w2c = np.linalg.inv(cam2w[j])
            cam = world @ w2c[:3, :3].T + w2c[:3, 3]
            z = cam[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.round(K[j][0, 0] * cam[:, 0] / z + K[j][0, 2])
                v = np.round(K[j][1, 1] * cam[:, 1] / z + K[j][1, 2])
            h, w = depth[j].shape
            ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = np.where(ok, u, 0).astype(np.int64)
            v = np.where(ok, v, 0).astype(np.int64)
            dj = depth[j][v, u]
            ok &= conf[j][v, u] > CONF_MIN
            agree |= ok & (np.abs(dj - z) < np.maximum(MV_ABS, MV_REL * z))
        keep[i] = agree.reshape(depth[i].shape)
    return keep


def recording_depth(net: Mast3r, rec: dict, cache: Path) -> dict:
    stems = np.array([p.stem for p in rec["paths"]])
    cached = cache / f"depth_{rec['name']}.npz"
    if cached.is_file() and not RECOMPUTE:
        c = dict(np.load(cached, allow_pickle=True))
        if np.array_equal(c.pop("stems"), stems):
            print(f"{rec['name']}: depth from cache", flush=True)
            c["stereo"] = c["stereo"].item()
            return c
    n = len(rec["paths"])
    pairs = partners(rec["cam2w"])
    print(f"{rec['name']}: {len(pairs)} stereo pairs", flush=True)
    maps = [[] for _ in range(n)]
    rgb = [None] * n
    stereo = {}
    started = time.time()
    for count, (i, j) in enumerate(pairs):
        for (a, b), res in zip(((i, j), (j, i)), net(rec["paths"][i], rec["paths"][j], both=True)):
            s, used = triangulate_scale(res["depth"], res["conf"], res["xy"], rec["K"][a], rec["K"][b],
                                        rec["cam2w"][a], rec["cam2w"][b])
            maps[a].append((s, used, res["depth"], res["conf"]))
            rgb[a] = res["rgb"]
            if a == i:
                stereo[(i, j)] = res["xy"]
        if count % 25 == 0:
            print(f"  {count + 1}/{len(pairs)} stereo pairs ({time.time() - started:.0f}s)", flush=True)
    scales = [s for m in maps for s, *_ in m if s is not None]
    fallback = float(np.median(scales))
    depth, conf = [], []
    for m in maps:
        weights = np.array([used if s is not None else 1 for s, used, *_ in m], float)[:, None, None]
        scaled = np.stack([d * (s if s is not None else fallback) for s, _, d, _ in m])
        depth.append((scaled * weights).sum(0) / weights.sum())
        conf.append(np.min(np.stack([c for *_, c in m]), 0))
    depth, conf = np.stack(depth).astype(np.float32), np.stack(conf).astype(np.float32)
    keep = multiview_filter(depth, conf, rec["K"], rec["cam2w"])
    print(f"{rec['name']}: scale median {fallback:.3f} (IQR {np.percentile(scales, 25):.3f}.."
          f"{np.percentile(scales, 75):.3f}, {len(scales)} of {sum(len(m) for m in maps)} views triangulated); "
          f"multi-view consistent {(keep & (conf > CONF_MIN)).sum() / max(1, (conf > CONF_MIN).sum()):.0%} "
          f"({time.time() - started:.0f}s)", flush=True)
    result = {"depth": depth, "conf": np.where(keep, conf, 0.0).astype(np.float32), "rgb": np.stack(rgb)}
    np.savez(cached, stems=stems, stereo=np.array(stereo, dtype=object), **result)
    result["stereo"] = stereo
    return result


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def yaw_of(T: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def yaw_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation about Z + translation taking src to dst."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src[:, :2] - ms[:2], dst[:, :2] - md[:2]
    theta = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = md - T[:3, :3] @ ms
    return T


def ransac_yaw(src: np.ndarray, dst: np.ndarray, thr: float, rng) -> tuple[np.ndarray | None, np.ndarray]:
    best = np.zeros(len(src), bool)
    if len(src) < 3:
        return None, best
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(src), 2, replace=False)
        if np.linalg.norm(src[idx[0], :2] - src[idx[1], :2]) < 0.1:
            continue
        T = yaw_fit(src[idx], dst[idx])
        inl = np.linalg.norm(apply(T, src) - dst, axis=1) < thr
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        return None, best
    for _ in range(3):
        T = yaw_fit(src[best], dst[best])
        best = np.linalg.norm(apply(T, src) - dst, axis=1) < thr
    return T, best


def cam_points(xy: np.ndarray, depth, conf, K) -> tuple[np.ndarray, np.ndarray]:
    u, v = xy[:, 0], xy[:, 1]
    z = depth[v, u]
    return np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]), conf[v, u] > CONF_MIN


def cached_pairs(A: dict, B: dict, folder: Path) -> tuple[list, dict] | None:
    """The pairs an earlier run of this reconstruction matched (read from its caches), or None."""
    files = [folder / "cross_matches.npz"] + [folder / f"loop_matches_{r['name']}.npz" for r in (A, B)]
    if RECOMPUTE or not all(f.is_file() for f in files):
        return None
    index = {r["name"]: {p.stem: k for k, p in enumerate(r["paths"])} for r in (A, B)}

    def pairs(path: Path, first: dict, second: dict) -> list[tuple[int, int]]:
        out = []
        for key in np.load(path, allow_pickle=True)["key"]:
            a, b = str(key).split("|")
            if a in index[first["name"]] and b in index[second["name"]]:
                out.append((index[first["name"]][a], index[second["name"]][b]))
        return sorted(out)

    cross = pairs(files[0], A, B)
    loops = {r["name"]: pairs(f, r, r) for r, f in zip((A, B), files[1:])}
    return (cross, loops) if cross else None


def sensor_depth(rec: dict) -> dict:
    """RGB-D mode: the recording's own depth maps (millimetre PNGs), at the 512-px working size."""
    keyframes = DATA / f"{rec['name']}_imgs" / "keyframes"
    depth, conf, rgb = [], [], []
    for path in rec["paths"]:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h = int(round(image.shape[0] * 512 / image.shape[1]))
        rgb.append(cv2.resize(image, (512, h), interpolation=cv2.INTER_AREA))
        d = cv2.imread(str(keyframes / "corrected_depth" / f"{path.stem}.Clean.png"), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise SystemExit(f"{rec['name']}: no depth map for keyframe {path.stem} (RGB-D mode needs "
                             f"keyframes/corrected_depth/<frame>.Clean.png)")
        d = cv2.resize(d.astype(np.float32) / 1000.0, (512, h), interpolation=cv2.INTER_NEAREST)
        ok = (d > SENSOR_DEPTH_RANGE_M[0]) & (d < SENSOR_DEPTH_RANGE_M[1])
        # as in RGB-D SLAM front ends: pixels on a depth discontinuity mix foreground and background
        kernel = np.ones((3, 3), np.uint8)
        spread = cv2.dilate(np.where(ok, d, 0.0), kernel) - cv2.erode(np.where(ok, d, 1.0e3), kernel)
        ok &= spread <= SENSOR_EDGE_M + SENSOR_EDGE_PER_M * d
        depth.append(np.where(ok, d, 0.0).astype(np.float32))
        conf.append(np.where(ok, 10.0, 0.0).astype(np.float32))          # measured: fully trusted
    stereo = {}
    source = (MATCHES_FROM / f"depth_{rec['name']}.npz") if MATCHES_FROM else None
    if source is not None and source.is_file() and not RECOMPUTE:
        c = np.load(source, allow_pickle=True)
        if np.array_equal(c["stems"], np.array([p.stem for p in rec["paths"]])):
            stereo = c["stereo"].item()                                   # neighbour matches, if MASt3R ran
    print(f"{rec['name']}: sensor depth for {len(depth)} keyframes, {len(stereo)} stereo pairs reused", flush=True)
    return {"depth": np.stack(depth), "conf": np.stack(conf), "rgb": np.stack(rgb), "stereo": stereo}


def candidate_pairs(A: dict, B: dict) -> tuple[list, dict]:
    desc = global_descriptors(A["paths"] + B["paths"])
    na = len(A["paths"])
    full = desc @ desc.T
    sim = full[:na, na:]
    cross = set()
    for i in range(na):
        cross |= {(i, int(j)) for j in np.argsort(-sim[i])[:CROSS_K]}
    for j in range(sim.shape[1]):
        cross |= {(int(i), j) for i in np.argsort(-sim[:, j])[:CROSS_K]}
    loops = {}
    for name, offset, count in ((A["name"], 0, na), (B["name"], na, len(B["paths"]))):
        block = full[offset:offset + count, offset:offset + count]
        found = set()
        for i in range(count):
            row = block[i].copy()
            row[max(0, i - LOOP_GAP):i + LOOP_GAP + 1] = -1
            found |= {tuple(sorted((i, int(j)))) for j in np.argsort(-row)[:LOOP_K] if row[j] > 0}
        loops[name] = sorted(found)
    return sorted(cross), loops


def initial_alignment(cross: dict, A: dict, B: dict, rng) -> np.ndarray:
    """One rigid 4-DOF transform B -> A: the per-pair estimate that most other pairs agree with."""
    estimates = []
    for (i, j), xy in cross.items():
        pa, oka = cam_points(xy[:, :2], A["depth"][i], A["conf"][i], A["K"][i])
        pb, okb = cam_points(xy[:, 2:], B["depth"][j], B["conf"][j], B["K"][j])
        ok = oka & okb
        src, dst = apply(B["cam2w"][j], pb[ok]), apply(A["cam2w"][i], pa[ok])
        T, inl = ransac_yaw(src, dst, RANSAC_M, rng)
        if T is not None and inl.sum() >= PAIR_MIN_INLIERS and inl.sum() >= PAIR_MIN_RATIO * ok.sum():
            estimates.append((T, src[inl], dst[inl]))
    votes = [sum(np.median(np.linalg.norm(apply(T, s) - d, axis=1)) < 2 * RANSAC_M for _, s, d in estimates)
             for T, _, _ in estimates]
    best = int(np.argmax(votes))
    print(f"initial alignment: {len(estimates)} of {len(cross)} cross pairs register on their own; "
          f"the best estimate is shared by {votes[best]} (yaw {yaw_of(estimates[best][0]):.1f} deg)", flush=True)
    return estimates[best][0]


def verify(kind: str, pairs: dict, frames: list, idx_a: dict, idx_b: dict, cam2w0: np.ndarray, rng) -> list[dict]:
    """Inlier 3D correspondences per pair, under the initial poses. Cross/loop pairs: 4-DOF RANSAC
    (rejects moved objects and mismatches); stereo pairs: residual gate."""
    out = []
    for (i, j), xy in pairs.items():
        a, b = idx_a[i], idx_b[j]
        fa, fb = frames[a], frames[b]
        pa, oka = cam_points(xy[:, :2].astype(np.int64), fa["depth"], fa["conf"], fa["K"])
        pb, okb = cam_points(xy[:, 2:].astype(np.int64), fb["depth"], fb["conf"], fb["K"])
        ok = oka & okb
        pa, pb = pa[ok], pb[ok]
        wa, wb = apply(cam2w0[a], pa), apply(cam2w0[b], pb)
        if kind == "stereo":
            inl = np.linalg.norm(wa - wb, axis=1) < STEREO_GATE_M
            good = inl.sum() >= MIN_TRIANGULATED
        else:
            T, inl = ransac_yaw(wb, wa, RANSAC_M, rng)
            good = (T is not None and inl.sum() >= PAIR_MIN_INLIERS and inl.sum() >= PAIR_MIN_RATIO * len(pa)
                    and abs(yaw_of(T)) < PAIR_MAX_YAW_DEG
                    and np.median(np.linalg.norm(wa[inl] - wb[inl], axis=1)) < PAIR_GATE_M)
        if not good:
            continue
        keep = np.flatnonzero(inl)
        if len(keep) > POINTS_PER_PAIR:
            keep = rng.choice(keep, POINTS_PER_PAIR, replace=False)
        xa, xb = xy[:, :2][ok][keep], xy[:, 2:][ok][keep]
        out.append({"kind": kind, "a": a, "b": b, "pa": pa[keep], "pb": pb[keep], "xa": xa, "xb": xb})
    return out


class PoseGraph:
    """Per frame: rotation about its own centre (3), centre shift (3), log depth scale (1).
    Matches enter as reprojection errors both ways (a's point seen by b, b's point seen by a): unlike
    3D point distances, these cannot be lowered by shrinking the depth maps."""

    def __init__(self, cam2w0: np.ndarray, K: np.ndarray, odometry: list[tuple[int, int]], constraints: list[dict],
                 anchor: int):
        self.K = K
        self.R0, self.c0 = cam2w0[:, :3, :3], cam2w0[:, :3, 3]
        self.n = len(cam2w0)
        self.odom = np.array(odometry)
        i, j = self.odom[:, 0], self.odom[:, 1]
        self.odom_R = np.einsum("nji,njk->nik", self.R0[i], self.R0[j])
        self.odom_t = np.einsum("nji,nj->ni", self.R0[i], self.c0[j] - self.c0[i])
        steps = np.linalg.norm(self.odom_t, axis=1)
        self.odom_sigma_t = (SIGMA_ODOM_T + SIGMA_ODOM_T_PER_M * steps)[:, None]
        self.ca = np.concatenate([np.full(len(c["pa"]), c["a"]) for c in constraints])
        self.cb = np.concatenate([np.full(len(c["pb"]), c["b"]) for c in constraints])
        self.pa = np.concatenate([c["pa"] for c in constraints])
        self.pb = np.concatenate([c["pb"] for c in constraints])
        self.xa = np.concatenate([c["xa"] for c in constraints]).astype(np.float64)
        self.xb = np.concatenate([c["xb"] for c in constraints]).astype(np.float64)
        self.anchor = anchor

    def reprojection(self, R, c, s, src, p, dst, x) -> np.ndarray:
        """Pixel error of frame src's point p (camera coords, scaled by s) as seen by frame dst."""
        w = np.einsum("nij,nj->ni", R[src], p * s[src, None]) + c[src]
        q = np.einsum("nji,nj->ni", R[dst], w - c[dst])
        K = self.K[dst]
        z = np.maximum(q[:, 2], 0.05)
        u = K[:, 0, 0] * q[:, 0] / z + K[:, 0, 2]
        v = K[:, 1, 1] * q[:, 1] / z + K[:, 1, 2]
        return np.column_stack([u - x[:, 0], v - x[:, 1]]) / SIGMA_PX

    def unpack(self, x: np.ndarray):
        x = x.reshape(self.n, 7)
        R = Rotation.from_rotvec(x[:, :3]).as_matrix() @ self.R0
        return R, self.c0 + x[:, 3:6], np.exp(x[:, 6])

    def residuals(self, x: np.ndarray) -> np.ndarray:
        R, c, s = self.unpack(x)
        i, j = self.odom[:, 0], self.odom[:, 1]
        rel_R = np.einsum("nji,njk->nik", R[i], R[j])
        rel_t = np.einsum("nji,nj->ni", R[i], c[j] - c[i])
        err_R = Rotation.from_matrix(np.einsum("nji,njk->nik", self.odom_R, rel_R)).as_rotvec() / SIGMA_ODOM_R
        err_t = (rel_t - self.odom_t) / self.odom_sigma_t
        err_p = np.hstack([self.reprojection(R, c, s, self.cb, self.pb, self.ca, self.xa),
                           self.reprojection(R, c, s, self.ca, self.pa, self.cb, self.xb)])
        xr = x.reshape(self.n, 7)
        gravity = xr[:, :2] / SIGMA_GRAVITY                  # roll/pitch corrections: the IMU knows "down"
        scale = xr[:, 6] / SIGMA_SCALE
        anchor = xr[self.anchor, :6] / 1e-3                  # gauge: first frame of the first recording
        return np.concatenate([err_R.ravel(), err_t.ravel(), err_p.ravel(), gravity.ravel(), scale, anchor])

    def sparsity(self) -> sp.csr_matrix:
        rows, cols = [], []

        def two_frame_block(offset: int, fa: np.ndarray, fb: np.ndarray, per: int) -> int:
            m = len(fa)
            r = offset + per * np.arange(m)[:, None] + np.arange(per)[None, :]                       # (m, per)
            c = np.concatenate([7 * fa[:, None] + np.arange(7), 7 * fb[:, None] + np.arange(7)], 1)  # (m, 14)
            rows.append(np.broadcast_to(r[:, :, None], (m, per, 14)).ravel())
            cols.append(np.broadcast_to(c[:, None, :], (m, per, 14)).ravel())
            return offset + per * m

        r = two_frame_block(0, self.odom[:, 0], self.odom[:, 1], 3)       # odometry rotation
        r = two_frame_block(r, self.odom[:, 0], self.odom[:, 1], 3)       # odometry translation
        r = two_frame_block(r, self.ca, self.cb, 4)                       # match reprojections, both ways
        k = np.arange(self.n)
        rows += [r + 2 * k, r + 2 * k + 1]                                # gravity
        cols += [7 * k, 7 * k + 1]
        r += 2 * self.n
        rows.append(r + k)                                                # depth scale
        cols.append(7 * k + 6)
        r += self.n
        rows.append(r + np.arange(6))                                     # anchor
        cols.append(7 * self.anchor + np.arange(6))
        r += 6
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, 7 * self.n))

    def solve(self) -> np.ndarray:
        x0 = np.zeros(7 * self.n)
        started = time.time()
        res = least_squares(self.residuals, x0, jac_sparsity=self.sparsity(), loss="soft_l1", f_scale=ROBUST_SCALE,
                            x_scale="jac", method="trf", tr_solver="lsmr", max_nfev=60, verbose=0)
        print(f"pose graph: {self.n} frames, {len(self.odom)} odometry edges, {len(self.ca)} point constraints; "
              f"cost {0.5 * np.sum(self.residuals(x0) ** 2):.0f} -> {0.5 * np.sum(self.residuals(res.x) ** 2):.0f} "
              f"({res.nfev} evaluations, {time.time() - started:.0f}s)", flush=True)
        return res.x


def residual_stats(constraints: list[dict], cam2w: np.ndarray, scale: np.ndarray) -> dict:
    if not constraints:
        return {}
    d = []
    for c in constraints:
        wa = apply(cam2w[c["a"]], c["pa"] * scale[c["a"]])
        wb = apply(cam2w[c["b"]], c["pb"] * scale[c["b"]])
        d.append(np.linalg.norm(wa - wb, axis=1))
    d = np.concatenate(d)
    return {"pairs": len(constraints), "points": int(len(d)), "median_cm": round(float(np.median(d)) * 100, 1),
            "p90_cm": round(float(np.percentile(d, 90)) * 100, 1), "within_5cm": round(float((d < 0.05).mean()), 3)}


def floor_height(depth, conf, K, cam2w) -> float:
    """Lowest dense horizontal layer of points below the cameras (z histogram, 1 cm bins)."""
    zs = []
    for k in range(len(depth)):
        m = np.zeros(depth[k].shape, bool)
        m[::4, ::4] = True
        m &= conf[k] > CONF_MIN
        zs.append(apply(cam2w[k], backproject(depth[k], K[k])[m])[:, 2])
    z = np.concatenate(zs)
    below = z[z < np.median(cam2w[:, 2, 3]) - 0.5]
    hist, edges = np.histogram(below, bins=np.arange(below.min(), below.max() + 0.01, 0.01))
    return float(edges[np.argmax(hist)] + 0.005)


def depth_vs_lidar(frames, owner, depth, conf) -> dict:
    errs, rels, within = [], [], []
    for k, (stem, rec) in enumerate(zip(frames, owner)):
        path = DATA / f"{rec}_imgs" / "keyframes" / "corrected_depth" / f"{stem}.Clean.png"
        if not path.is_file():
            continue
        sensor = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        sensor = cv2.resize(sensor, depth[k].shape[::-1], interpolation=cv2.INTER_NEAREST)
        ok = (sensor > 0.25) & (sensor < 5.0) & (conf[k] > CONF_MIN)
        if ok.sum() < 500:
            continue
        e = np.abs(depth[k][ok] - sensor[ok])
        errs.append(np.median(e))
        rels.append(np.median(e / sensor[ok]))
        within.append((e < 0.05).mean())
    return {"frames": len(errs), "abs_err_median_cm": round(float(np.median(errs) * 100), 2),
            "rel_err_median": round(float(np.median(rels)), 4), "within_5cm": round(float(np.mean(within)), 3)}


def match_pairs(net: Mast3r, A: dict, B: dict, pairs: list[tuple[int, int]], cache: Path, label: str) -> dict:
    """MASt3R matches (x1, y1, x2, y2 at 512x384) for frame pairs, cached by keyframe names."""
    store = {}
    if cache.is_file() and not RECOMPUTE:
        c = np.load(cache, allow_pickle=True)
        store = dict(zip(c["key"], c["xy"]))
    out, started, fresh = {}, time.time(), 0
    for count, (i, j) in enumerate(pairs):
        key = f"{A['paths'][i].stem}|{B['paths'][j].stem}"
        if key not in store:
            store[key] = net(A["paths"][i], B["paths"][j])[0]["xy"]
            fresh += 1
            if fresh % 50 == 1:
                print(f"  {label}: {count + 1}/{len(pairs)} pairs ({time.time() - started:.0f}s)", flush=True)
        out[(i, j)] = store[key].astype(np.int64)
    if fresh:
        keys = sorted(store)
        xy = np.empty(len(keys), object)
        xy[:] = [store[k] for k in keys]
        np.savez(cache, key=np.array(keys), xy=xy)
    return out

# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    A, B = (load_recording(r, KEYFRAME_EVERY) for r in context.SESSIONS)
    net = Mast3r()
    if MATCHES_FROM is not None and not RECOMPUTE:         # reuse the matches of the RGB reconstruction
        for name in ["cross_matches.npz"] + [f"loop_matches_{r['name']}.npz" for r in (A, B)]:
            if (MATCHES_FROM / name).is_file() and not (OUT / name).exists():
                shutil.copyfile(MATCHES_FROM / name, OUT / name)
    for rec in (A, B):
        rec.update(sensor_depth(rec) if DEPTH_SOURCE == "sensor" else recording_depth(net, rec, OUT))
    cross_ids, loop_ids = cached_pairs(A, B, OUT) or candidate_pairs(A, B)
    print(f"candidate pairs: {len(cross_ids)} across, loops {[len(v) for v in loop_ids.values()]}", flush=True)
    cross = match_pairs(net, A, B, cross_ids, OUT / "cross_matches.npz", "cross")
    loops = {rec["name"]: match_pairs(net, rec, rec, loop_ids[rec["name"]], OUT / f"loop_matches_{rec['name']}.npz",
                                      f"loops {rec['name']}") for rec in (A, B)}
    net.release()

    T0 = initial_alignment(cross, A, B, rng)
    na, nb = len(A["paths"]), len(B["paths"])
    cam2w0 = np.concatenate([A["cam2w"], np.einsum("ij,njk->nik", T0, B["cam2w"])])
    frames = [{"depth": r["depth"][k], "conf": r["conf"][k], "K": r["K"][k]} for r in (A, B) for k in range(len(r["paths"]))]
    ia = {k: k for k in range(na)}
    ib = {k: na + k for k in range(nb)}
    cons = verify("cross", cross, frames, ia, ib, cam2w0, rng)
    test = cons[HOLDOUT_EVERY - 1::HOLDOUT_EVERY]
    train = [c for k, c in enumerate(cons) if (k + 1) % HOLDOUT_EVERY]
    loop_cons = verify("loop", loops[A["name"]], frames, ia, ia, cam2w0, rng) + \
        verify("loop", loops[B["name"]], frames, ib, ib, cam2w0, rng)
    stereo_cons = verify("stereo", A["stereo"], frames, ia, ia, cam2w0, rng) + \
        verify("stereo", B["stereo"], frames, ib, ib, cam2w0, rng)
    print(f"verified pairs: cross {len(cons)} of {len(cross)} ({len(test)} held out), "
          f"loops {len(loop_cons)} of {sum(len(v) for v in loops.values())}, stereo {len(stereo_cons)}", flush=True)
    odometry = [(k, k + 1) for k in range(na - 1)] + [(na + k, na + k + 1) for k in range(nb - 1)]
    ones = np.ones(na + nb)
    before = {"held_out_cross": residual_stats(test, cam2w0, ones), "cross": residual_stats(train, cam2w0, ones),
              "loops": residual_stats(loop_cons, cam2w0, ones)}
    graph = PoseGraph(cam2w0, np.concatenate([A["K"], B["K"]]), odometry, train + loop_cons + stereo_cons, anchor=0)
    R, c, scale = graph.unpack(graph.solve())
    cam2w = np.tile(np.eye(4), (na + nb, 1, 1))
    cam2w[:, :3, :3], cam2w[:, :3, 3] = R, c
    after = {"held_out_cross": residual_stats(test, cam2w, scale), "cross": residual_stats(train, cam2w, scale),
             "loops": residual_stats(loop_cons, cam2w, scale)}
    shift = np.linalg.norm(c - cam2w0[:, :3, 3], axis=1)
    print("residuals before:", json.dumps(before), "\nresiduals after: ", json.dumps(after), flush=True)

    depth = np.concatenate([A["depth"], B["depth"]]) * scale[:, None, None].astype(np.float32)
    conf = np.concatenate([A["conf"], B["conf"]])
    K = np.concatenate([A["K"], B["K"]])
    floor = floor_height(depth[:na], conf[:na], K[:na], cam2w[:na])
    cam2w[:, 2, 3] -= floor
    names = np.array([p.stem for r in (A, B) for p in r["paths"]])
    owner = np.array([r["name"] for r in (A, B) for _ in r["paths"]])
    report = {"profile": config.summary(), "depth_source": DEPTH_SOURCE, "keyframe_every": KEYFRAME_EVERY,
              "frames": {A["name"]: na, B["name"]: nb},
              "pairs": {"cross_candidates": len(cross), "cross_verified": len(cons), "held_out": len(test),
                        "loops_verified": len(loop_cons), "stereo_verified": len(stereo_cons)},
              "residuals_before_graph": before, "residuals_after_graph": after,
              "graph_correction_m": {"median": round(float(np.median(shift)), 3), "max": round(float(shift.max()), 3)},
              "depth_scale_correction": {"median": round(float(np.median(scale)), 4),
                                         "p5": round(float(np.percentile(scale, 5)), 4),
                                         "p95": round(float(np.percentile(scale, 95)), 4)},
              "floor_z": floor, "runtime_s": round(time.time() - started, 1)}
    if DEPTH_SOURCE != "sensor":
        try:      # informative only: the phone's own depth sensor, if the export has it
            report["depth_vs_sensor"] = depth_vs_lidar(names, owner, depth, conf)
        except (ValueError, ZeroDivisionError):
            pass
    np.savez_compressed(OUT / "scene.npz", frames=names, owner=owner, cam2w=cam2w, K=K, depth=depth, conf=conf,
                        rgb=np.concatenate([A["rgb"], B["rgb"]]))
    report["result_sha256"] = hashlib.sha256(np.round(cam2w, 6).tobytes()).hexdigest()
    (OUT / "recon_report.json").write_text(json.dumps(report, indent=2))
    print(f"recon: {na} + {nb} keyframes, held-out cross-video pairs agree to "
          f"{after['held_out_cross'].get('median_cm')} cm (median) in {report['runtime_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
