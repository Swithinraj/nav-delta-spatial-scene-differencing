"""Convert one ChangeSim warehouse (reference run + query run) into the navdelta data layout.

  python benchmarks/changesim/convert.py --warehouse Warehouse_6 --seq Seq_0 --out benchmarks/work/changesim/data/W6_S0

Before = the reference run: raw frames at RTAB-Map graph nodes (raw/poses.g2o; node id == raw frame
         number), 16-bit depth in millimetres.
After  = the query run (normal light): rgb/, 8-bit depth (value * 50 / 255 m), pose/<frame>.txt.
Both runs are in the same RTAB-Map map frame. So that the approaches have to register them, the
query run is moved into its own frame by a known rigid transform (yaw + shift); the ground-truth
query -> reference transform is stored for grading.
Camera axes (OpenCV vs NED) are chosen by depth consistency between neighbouring frames.

Ground truth (gt.json, evaluation only), from the query run's change masks (0 static, 1 new,
2 missing, 3 rotated, 4 replaced): changed pixels are lifted to 3D -- with the query depth, or for
"missing" objects with the matched reference frame's depth (t0/), since the query depth shows what
is behind the gap -- and grouped into objects by connected 10 cm voxels.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from remote_zip import RangeFile  # noqa: E402
import paths  # noqa: E402

K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1]])
W, H = 640, 480
Q_YUP_TO_ZUP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
NED_FROM_CV = np.eye(4)
NED_FROM_CV[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]     # cv (x right, y down, z fwd) -> ned (x fwd, y right, z down)
QUERY_SHIFT = (25.0, np.array([1.5, -0.8, 0.0]))           # yaw (deg), translation (m) applied to the query run
CHANGE_COLOURS = {(81, 38, 0): 1, (41, 36, 132): 2, (25, 48, 16): 3, (131, 192, 13): 4}
CHANGE_NAMES = {1: "added", 2: "removed", 3: "rotated", 4: "replaced"}
GT_VOXEL_M = 0.10
GT_MIN_POINTS = 200
GT_MIN_FRAMES = 3
GT_MAX_SIZE_M = 4.0         # longer clusters are depth smears along an aisle, not one object
DEPTH_MAX_M = 20.0


def pose_matrix(t, q) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


class Source:
    """Local copy first, otherwise one entry from the remote archive."""

    def __init__(self, raw: Path):
        self.raw = raw
        self.zips: dict[str, zipfile.ZipFile] = {}

    def read(self, name: str) -> bytes:
        local = self.raw / name
        if local.is_file():
            return local.read_bytes()
        part = name.split("/")[0]
        if part not in self.zips:
            self.zips[part] = zipfile.ZipFile(RangeFile(part))
        data = self.zips[part].read(name)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        return data

    def image(self, name: str, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.read(name), np.uint8), flags)


def backproject(depth: np.ndarray, T: np.ndarray, stride: int = 4, mask: np.ndarray | None = None,
                with_pixels: bool = False):
    ok = (depth > 0.2) & (depth < DEPTH_MAX_M)
    if mask is not None:
        ok &= mask
    sub = np.zeros_like(ok)
    sub[::stride, ::stride] = True
    v, u = np.nonzero(ok & sub)
    z = depth[v, u]
    p = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    p = p @ T[:3, :3].T + T[:3, 3]
    return (p, v, u) if with_pixels else p


def render(depth: np.ndarray, T_src: np.ndarray, T_dst: np.ndarray) -> np.ndarray:
    """Depth image of frame src seen from camera dst (z-buffer)."""
    world = backproject(depth, T_src, 1)
    w2c = np.linalg.inv(T_dst)
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    ok = cam[:, 2] > 0.2
    cam = cam[ok]
    u = np.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.full((H, W), np.inf)
    np.minimum.at(out, (v[ok], u[ok]), cam[ok, 2])
    out[np.isinf(out)] = 0
    filled = cv2.morphologyEx(out.astype(np.float32), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return np.where(out > 0, out, filled)


def choose_axes(depths: list[np.ndarray], poses: list[np.ndarray]) -> tuple[np.ndarray, str, dict]:
    scores = {}
    for name, fix in (("opencv", np.eye(4)), ("ned", NED_FROM_CV)):
        d = []
        for k in range(0, len(depths) - 1, max(1, len(depths) // 12)):
            a = backproject(depths[k], poses[k] @ fix, 8)
            b = backproject(depths[k + 1], poses[k + 1] @ fix, 8)
            if len(a) and len(b):
                d.append(cKDTree(b).query(a)[0])
        scores[name] = float(np.median(np.concatenate(d)))
    name = min(scores, key=scores.get)
    return (np.eye(4) if name == "opencv" else NED_FROM_CV), name, scores


def write_recording(out: Path, name: str, frames: list[dict], fix: np.ndarray, shift: np.ndarray) -> None:
    kf = out / f"{name}_imgs" / "keyframes"
    for sub in ("images", "corrected_images", "cameras", "corrected_cameras", "corrected_depth"):
        (kf / sub).mkdir(parents=True, exist_ok=True)
    raw_cams = out / f"{name}_rawdata" / "keyframes" / "cameras"
    raw_cams.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_rawdata" / "mesh_info.json").write_text(json.dumps({"alignmentTransform": np.eye(4).ravel().tolist()}))
    xyz, rgb = [], []
    for f in frames:
        T_cv = shift @ f["pose"] @ fix
        stem = str(1_000_000 + f["index"])
        T_arkit = Q_YUP_TO_ZUP.T @ T_cv @ CV_TO_ARKIT
        cam = {f"t_{r}{c}": float(T_arkit[r, c]) for r in range(3) for c in range(4)}
        cam.update(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=W, height=H, timestamp=int(stem),
                   source_frame=f["index"])
        for folder in (kf / "cameras", kf / "corrected_cameras", raw_cams):
            (folder / f"{stem}.json").write_text(json.dumps(cam))
        jpg = cv2.imencode(".jpg", f["rgb"], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        for folder in (kf / "images", kf / "corrected_images"):
            (folder / f"{stem}.jpg").write_bytes(jpg)
        cv2.imwrite(str(kf / "corrected_depth" / f"{stem}.Clean.png"),
                    np.round(np.clip(f["depth"], 0, 65.0) * 1000).astype(np.uint16))
        pts, v, u = backproject(f["depth"], T_cv, 4, f["depth"] < 8.0, with_pixels=True)
        xyz.append(pts)
        rgb.append(f["rgb"][v, u, ::-1])
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    keys = np.floor(xyz / 0.02).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse)
    mxyz = np.stack([np.bincount(inverse, xyz[:, k]) / counts for k in range(3)], 1)
    mrgb = np.stack([np.bincount(inverse, rgb[:, k].astype(float)) / counts for k in range(3)], 1)
    data = np.empty(len(mxyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = mxyz.T
    data["red"], data["green"], data["blue"] = np.round(mrgb).astype(np.uint8).T
    (out / f"{name}_ply").mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_ply" / f"{name}.ply", "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        fh.write(data.tobytes())
    print(f"  {name}: {len(frames)} keyframes, cloud {len(data)} points", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(paths.RAW))
    ap.add_argument("--warehouse", default="Warehouse_6")
    ap.add_argument("--seq", default="Seq_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keyframes", type=int, default=200)
    ap.add_argument("--gt-frames", type=int, default=300, help="about this many query frames build the ground truth")
    args = ap.parse_args()
    src = Source(Path(args.raw))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ref = f"Ref_Seq_Test/{args.warehouse}/{args.seq}"
    qry = f"Query_Seq_Test/{args.warehouse}/{args.seq}"

    # reference run: graph nodes
    nodes = {}
    for line in src.read(f"{ref}/raw/poses.g2o").decode().splitlines():
        p = line.split()
        if p and p[0] == "VERTEX_SE3:QUAT":
            nodes[int(p[1])] = pose_matrix([float(x) for x in p[2:5]], [float(x) for x in p[5:9]])
    ids = sorted(nodes)
    pick = ids[:: max(1, len(ids) // args.keyframes)]
    print(f"reference: {len(ids)} graph nodes -> {len(pick)} keyframes", flush=True)
    before = [{"index": i, "pose": nodes[i], "rgb": src.image(f"{ref}/raw/rgb/{i}.png", cv2.IMREAD_COLOR),
               "depth": src.image(f"{ref}/raw/depth/{i}.png").astype(np.float64) / 1000.0} for i in pick]

    # query run
    q_ids = sorted(int(n.split(".")[0]) for n in
                   (p.name for p in (Path(args.raw) / qry / "pose").glob("*.txt")))
    q_pick = q_ids[:: max(1, len(q_ids) // args.keyframes)]
    print(f"query: {len(q_ids)} frames -> {len(q_pick)} keyframes", flush=True)

    def q_pose(i: int) -> np.ndarray:
        v = [float(x) for x in src.read(f"{qry}/pose/{i}.txt").decode().split()]
        return pose_matrix(v[:3], v[3:7])

    def q_depth(i: int) -> np.ndarray:
        return src.image(f"{qry}/depth/{i}.png").astype(np.float64) * 50.0 / 255.0

    after = [{"index": i, "pose": q_pose(i), "rgb": src.image(f"{qry}/rgb/{i}.png", cv2.IMREAD_COLOR),
              "depth": q_depth(i)} for i in q_pick]

    fix, axes, scores = choose_axes([f["depth"] for f in before[:40]], [f["pose"] for f in before[:40]])
    print(f"camera axes: {axes} (neighbour depth consistency, median cm: "
          f"{ {k: round(v * 100, 1) for k, v in scores.items()} })", flush=True)
    yaw, t = QUERY_SHIFT
    shift = np.eye(4)
    shift[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
    shift[:3, 3] = t
    names = {"before": f"A_{args.warehouse[-1]}{args.seq[-1]}ref", "after": f"B_{args.warehouse[-1]}{args.seq[-1]}qry"}
    write_recording(out, names["before"], before, fix, np.eye(4))
    write_recording(out, names["after"], after, fix, shift)

    # ground truth from the change masks (evaluation only)
    pts = {c: [] for c in CHANGE_NAMES}
    frames_of = {c: [] for c in CHANGE_NAMES}
    sem_of = {c: [] for c in CHANGE_NAMES}
    gt_ids = q_ids[:: max(1, len(q_ids) // args.gt_frames)]
    for i in gt_ids:
        cm = src.image(f"{qry}/change_segmentation/{i}.png", cv2.IMREAD_COLOR)[..., ::-1]
        sem = src.image(f"{qry}/semantic_segmentation/{i}.png", cv2.IMREAD_COLOR)[..., ::-1]
        code = np.zeros(cm.shape[:2], np.int64)
        for colour, c in CHANGE_COLOURS.items():
            code[(cm == colour).all(-1)] = c
        if not code.any():
            continue
        depth = q_depth(i)
        pose = q_pose(i) @ fix
        for c in CHANGE_NAMES:
            m = code == c
            if m.sum() < 50:
                continue
            if c == 2:      # missing: the object was there in the reference run -> reference depth (t0),
                #               rendered into the query camera so it lines up with the query mask
                node = int(src.read(f"{qry}/t0/idx/{i}.txt").decode().split()[0])
                d = src.image(f"{qry}/t0/depth/{i}.png").astype(np.float64) / 1000.0     # 16-bit mm, like raw/
                p = backproject(render(d, nodes[node] @ fix, pose), pose, 2, m)
            else:
                p = backproject(depth, pose, 2, m)
            pts[c].append(p)
            frames_of[c].append(np.full(len(p), i))
            colours = sem[m][:: max(1, m.sum() // 200)]
            sem_of[c].append(colours)
    changes = []
    dropped = 0
    for c in CHANGE_NAMES:
        if not pts[c]:
            continue
        P, F = np.concatenate(pts[c]), np.concatenate(frames_of[c])
        if len(P) == 0:
            continue
        keys = np.floor(P / GT_VOXEL_M).astype(np.int64)
        low = keys.min(0)
        grid = np.zeros(tuple(keys.max(0) - low + 3), bool)
        grid[tuple((keys - low + 1).T)] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
        idx = lab[tuple((keys - low + 1).T)]
        for k in range(1, n + 1):
            m = idx == k
            if m.sum() < GT_MIN_POINTS or len(np.unique(F[m])) < GT_MIN_FRAMES:
                continue
            q = P[m]
            lo, hi = np.percentile(q, 2, axis=0), np.percentile(q, 98, axis=0)
            if max(hi - lo) > GT_MAX_SIZE_M:
                dropped += 1
                continue
            box = {"label": "object", "centroid": ((lo + hi) / 2).tolist(), "axes": np.eye(3).tolist(),
                   "lengths": np.maximum(hi - lo, 0.05).tolist()}
            kind = CHANGE_NAMES[c]
            changes.append({"id": f"{kind}{len(changes)}", "type": kind,
                            "changesim_type": kind, "label": "object", "frames": int(len(np.unique(F[m]))),
                            "points": int(m.sum()), "size": float(max(hi - lo)),
                            "before": box if kind != "added" else None, "after": box if kind != "removed" else None})
    print(f"ground truth: {len(changes)} changed objects ({dropped} clusters > {GT_MAX_SIZE_M} m dropped) "
          f"{ {k: sum(c['changesim_type'] == k for c in changes) for k in CHANGE_NAMES.values()} }", flush=True)
    gt = {"dataset": "ChangeSim", "reference": f"{ref}", "rescan": f"{qry}", "recordings": names,
          "frame": "reference run map frame (RTAB-Map, Z up), metres", "camera_axes": axes,
          "rescan_to_reference": np.linalg.inv(shift).tolist(), "changes": changes,
          "dropped_smears": dropped,
          "note": "rotated / replaced: an object changed in place (turned / swapped for another)"}
    (out / "gt.json").write_text(json.dumps(gt, indent=1))
    (out / "convert_report.json").write_text(json.dumps({"before_keyframes": len(before), "after_keyframes": len(after),
                                                         "camera_axes": axes, "axis_scores_m": scores,
                                                         "gt_changes": len(changes)}, indent=2))


if __name__ == "__main__":
    main()
