"""loop_outliers, stage 1: changes are the outliers of the cross-video loop closure.

The pose graph (shared with approach 2: phone tracking + MASt3R loop closures) puts both videos in
one drift-corrected frame. Every MASt3R match between a frame of video A and a frame of video B then
claims "this surface in A is that surface in B". For the static room both ends land on the same 3D
point: the loop closure's inliers. Robust SLAM throws the outliers away; here they are the signal.

An outlier match is explained by checking each of its two ends against the OTHER video's depth:
  both ends vacated   (A's end is empty space in B, B's end was empty space in A): the same surface
                      sits somewhere else now -> MOVED; sequential RANSAC on these matches finds one
                      rigid motion (yaw about gravity + translation) per object, measured from its
                      own features (distance and turn, so an object turned in place is caught too)
  only A's end vacated  that surface is gone -> evidence for REMOVED (MASt3R matched it to the
                      background now visible where it stood)
  only B's end vacated  -> evidence for ADDED
  neither             a mismatch: discarded
Removed / added evidence is grouped into objects (connected 10 cm cells, seen in >= 2 view pairs).

"""
from __future__ import annotations

import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from navdelta import config, context
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"                  # shared reconstruction (navdelta.recon)
OUTPUT = context.approach_dir("loop_outliers")

CONF_MIN = 1.5
INLIER_M, INLIER_PER_M = 0.10, 0.03      # static: ends agree within max(10 cm, 3 % of depth)
OUTLIER_M = 0.25                          # an outlier: ends further apart than this ...
OUTLIER_PER_M = 0.0                       # ... plus this much per metre of depth (large spaces)
SEE_PAST_M, SEE_PAST_PER_M = 0.10, 0.0    # the other video's depth lies this far behind a point: empty
SURFACE_M, SURFACE_PER_M = 0.10, 0.0      # ... within this of the point: a surface is still there
RANSAC_M = 0.08                           # an object's own matches agree on its motion to 8 cm
RANSAC_ITERS = 500
MIN_MATCHES = 25                          # matches behind one moved object ...
MIN_PAIRS = 2                             # ... from at least this many view pairs
MAX_EXTENT_M = 2.5                        # a moved thing is at most this big
CELL_M = 0.10                             # grouping of removed / added evidence
MIN_EVIDENCE = 20                         # evidence points behind one removed / added object
EXPLAINED_M = 0.20                        # evidence this close to a moved object belongs to it
ABSORB_M = 0.50                           # a removed / added group this close to a move is part of it
SEED = 0


# Scale profile (navdelta/configs/*.toml): the values above are the room profile.
PROFILE_SETTINGS = config.apply(globals(), "loop_outliers.mine")


def lift(xy, depth, conf, K, cam2w):
    u, v = xy[:, 0], xy[:, 1]
    z = depth[v, u]
    p = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    return p @ cam2w[:3, :3].T + cam2w[:3, 3], z, conf[v, u] > CONF_MIN


def vacated(points: np.ndarray, scene: dict, frames: np.ndarray) -> np.ndarray:
    """True where the given frames see through the point (empty) at least as often as they see a
    surface at it (a thin object in front of furniture keeps some furniture hits)."""
    empty = np.zeros(len(points), int)
    surface = np.zeros(len(points), int)
    for k in frames:
        w2c = np.linalg.inv(scene["cam2w"][k])
        cam = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        K = scene["K"][k]
        h, w = scene["depth"][k].shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2])
            v = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2])
        ok = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not ok.any():
            continue
        u = np.where(ok, u, 0).astype(int)
        v = np.where(ok, v, 0).astype(int)
        d = scene["depth"][k][v, u]
        ok &= scene["conf"][k][v, u] > CONF_MIN
        empty += ok & (d > z + SEE_PAST_M + SEE_PAST_PER_M * z)
        surface += ok & (np.abs(d - z) < SURFACE_M + SURFACE_PER_M * z)
    return (empty > 0) & (empty >= surface)


def yaw_fit(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src[:, :2] - ms[:2], dst[:, :2] - md[:2]
    theta = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = md - T[:3, :3] @ ms
    return T


def apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def motions(src, dst, pair, rng) -> list[dict]:
    """Sequential RANSAC: rigid motions that enough matches, from enough view pairs, agree on."""
    left = np.arange(len(src))
    found = []
    while len(left) >= MIN_MATCHES:
        i = rng.choice(left, RANSAC_ITERS)
        near = np.argsort(np.linalg.norm(src[left][None] - src[i][:, None], axis=2), axis=1)[:, 1:20]
        j = left[near[np.arange(RANSAC_ITERS), rng.integers(0, near.shape[1], RANSAC_ITERS)]]
        best = None
        for a, b in zip(i, j):
            if np.linalg.norm(src[a, :2] - src[b, :2]) < 0.05:
                continue
            T = yaw_fit(src[[a, b]], dst[[a, b]])
            inl = left[np.linalg.norm(apply(T, src[left]) - dst[left], axis=1) < RANSAC_M]
            if best is None or len(inl) > len(best[1]):
                best = (T, inl)
        if best is None or len(best[1]) < MIN_MATCHES:
            break
        T, inl = best
        for _ in range(3):
            T = yaw_fit(src[inl], dst[inl])
            inl = left[np.linalg.norm(apply(T, src[left]) - dst[left], axis=1) < RANSAC_M]
        centre = np.median(src[inl], axis=0)
        inl = inl[np.linalg.norm(src[inl] - centre, axis=1) < MAX_EXTENT_M / 2]
        left = np.setdiff1d(left, inl)
        if len(inl) < MIN_MATCHES or len(np.unique(pair[inl])) < MIN_PAIRS:
            continue
        T = yaw_fit(src[inl], dst[inl])
        shift = np.median(dst[inl] - src[inl], axis=0)
        found.append({"T": T, "idx": inl, "pairs": int(len(np.unique(pair[inl]))),
                      "from": np.median(src[inl], axis=0), "to": np.median(dst[inl], axis=0),
                      "distance_m": float(np.linalg.norm(shift[:2])),
                      "turn_deg": float(abs(np.degrees(np.arctan2(T[1, 0], T[0, 0]))))})
    return found


def merge_motions(found: list[dict], src: np.ndarray) -> list[dict]:
    """Motions of one object found twice (parts of it): same place, same motion -> one."""
    out = []
    for m in sorted(found, key=lambda m: -len(m["idx"])):
        for o in out:
            if (np.linalg.norm(m["from"][:2] - o["from"][:2]) < 0.6 and np.linalg.norm(m["to"][:2] - o["to"][:2]) < 0.6
                    and abs(m["turn_deg"] - o["turn_deg"]) < 30):
                o["idx"] = np.union1d(o["idx"], m["idx"])
                o["pairs"] = max(o["pairs"], m["pairs"])
                break
        else:
            out.append(dict(m))
    return out


def groups(points: np.ndarray, pair: np.ndarray) -> list[np.ndarray]:
    """Evidence points grouped by connected CELL_M cells; enough points from enough view pairs."""
    if len(points) == 0:
        return []
    keys = np.floor(points / CELL_M).astype(np.int64)
    low = keys.min(0)
    grid = np.zeros(tuple(keys.max(0) - low + 3), bool)
    grid[tuple((keys - low + 1).T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    lab = labels[tuple((keys - low + 1).T)]
    out = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(lab == k)
        if len(idx) >= MIN_EVIDENCE and len(np.unique(pair[idx])) >= MIN_PAIRS:
            out.append(idx)
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    work = OUTPUT
    started = time.time()
    scene = {k: v for k, v in np.load(inside(RECON / "scene.npz")).items() if k != "rgb"}      # decompress once
    index = {(o, f): k for k, (o, f) in enumerate(zip(scene["owner"], scene["frames"]))}
    rec_a, rec_b = list(dict.fromkeys(scene["owner"]))
    frames_a, frames_b = np.flatnonzero(scene["owner"] == rec_a), np.flatnonzero(scene["owner"] == rec_b)
    cache = np.load(inside(RECON / "cross_matches.npz"), allow_pickle=True)
    src, dst, pair, frame_a, frame_b, pix_a, pix_b, residuals = [], [], [], [], [], [], [], []
    n_static = n_matches = 0
    for p, (key, xy) in enumerate(zip(cache["key"], cache["xy"])):
        sa, sb = key.split("|")
        if (rec_a, sa) not in index or (rec_b, sb) not in index:
            continue
        a, b = index[(rec_a, sa)], index[(rec_b, sb)]
        xy = xy.astype(np.int64)
        pa, za, oka = lift(xy[:, :2], scene["depth"][a], scene["conf"][a], scene["K"][a], scene["cam2w"][a])
        pb, zb, okb = lift(xy[:, 2:], scene["depth"][b], scene["conf"][b], scene["K"][b], scene["cam2w"][b])
        ok = oka & okb
        r = np.linalg.norm(pa - pb, axis=1)
        residuals.append(r[ok])
        n_matches += int(ok.sum())
        n_static += int((ok & (r < np.maximum(INLIER_M, INLIER_PER_M * np.maximum(za, zb)))).sum())
        out = ok & (r > OUTLIER_M + OUTLIER_PER_M * np.maximum(za, zb))
        src.append(pa[out]); dst.append(pb[out]); pair.append(np.full(int(out.sum()), p))
        frame_a.append(np.full(int(out.sum()), a)); frame_b.append(np.full(int(out.sum()), b))
        pix_a.append(xy[out, :2]); pix_b.append(xy[out, 2:])
    src, dst, pair = np.concatenate(src), np.concatenate(dst), np.concatenate(pair)
    frame_a, frame_b = np.concatenate(frame_a), np.concatenate(frame_b)
    pix_a, pix_b = np.concatenate(pix_a), np.concatenate(pix_b)
    res = np.concatenate(residuals)
    gone_a = vacated(src, scene, frames_b)          # A's end: empty space in B now
    new_b = vacated(dst, scene, frames_a)           # B's end: was empty space in A
    both, only_a, only_b = gone_a & new_b, gone_a & ~new_b, new_b & ~gone_a
    print(f"{n_matches} cross-video matches: {n_static / n_matches:.0%} static (median residual "
          f"{np.median(res) * 100:.1f} cm); {len(src)} outliers: {both.sum()} moved surfaces, {only_a.sum()} vanished, "
          f"{only_b.sum()} appeared, {(~gone_a & ~new_b).sum()} mismatches ({time.time() - started:.1f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    mv = np.flatnonzero(both)
    found = merge_motions(motions(src[mv], dst[mv], pair[mv], rng), src[mv])
    for m in found:
        m["idx"] = mv[m["idx"]]
    explained_a = np.zeros(len(src), bool)
    explained_b = np.zeros(len(src), bool)
    for m in found:
        sa, sb = src[m["idx"]], dst[m["idx"]]
        explained_a |= np.min(np.linalg.norm(src[:, None, :2] - sa[None, ::5, :2], axis=2), axis=1) < EXPLAINED_M
        explained_b |= np.min(np.linalg.norm(dst[:, None, :2] - sb[None, ::5, :2], axis=2), axis=1) < EXPLAINED_M
    rem = np.flatnonzero(only_a & ~explained_a)
    add = np.flatnonzero(only_b & ~explained_b)
    removed = [rem[g] for g in groups(src[rem], pair[rem])]
    added = [add[g] for g in groups(dst[add], pair[add])]
    # leftover pieces of a moved object (a chair's seat when its back carried the matches)
    for m in found:
        m["absorbed_before"], m["absorbed_after"] = [], []
    for kind, pieces, ends in (("before", removed, src), ("after", added, dst)):
        keep = []
        for g in pieces:
            pts = ends[g][:: max(1, len(g) // 200), :2]
            near = [m for m in found if np.min(np.linalg.norm(
                (src if kind == "before" else dst)[m["idx"]][None, ::3, :2] - pts[:, None], axis=2)) < ABSORB_M]
            if near:
                near[0][f"absorbed_{kind}"].append(g)
            else:
                keep.append(g)
        pieces[:] = keep

    def box(points):
        lo, hi = np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)
        return {"centroid": ((lo + hi) / 2).round(3).tolist(), "lengths": np.maximum(hi - lo, CELL_M).round(3).tolist()}

    changes = []
    for k, m in enumerate(found):
        before = np.concatenate([m["idx"], *m["absorbed_before"]]) if m["absorbed_before"] else m["idx"]
        after = np.concatenate([m["idx"], *m["absorbed_after"]]) if m["absorbed_after"] else m["idx"]
        changes.append({"id": f"M{k + 1}", "type": "moved", "evidence_matches": int(len(m["idx"])),
                        "view_pairs": m["pairs"], "distance_m": round(m["distance_m"], 3),
                        "turn_deg": round(m["turn_deg"], 1), "transform": m["T"].round(5).tolist(),
                        "before": box(src[before]), "after": box(dst[after]), "idx": m["idx"].tolist(),
                        "idx_before": before.tolist(), "idx_after": after.tolist()})
    for k, g in enumerate(removed):
        changes.append({"id": f"OR{k + 1}", "type": "removed", "evidence_matches": int(len(g)),
                        "view_pairs": int(len(np.unique(pair[g]))), "object": box(src[g]), "idx": g.tolist()})
    for k, g in enumerate(added):
        changes.append({"id": f"OA{k + 1}", "type": "added", "evidence_matches": int(len(g)),
                        "view_pairs": int(len(np.unique(pair[g]))), "object": box(dst[g]), "idx": g.tolist()})
    for c in changes:
        where = c["before"]["centroid"] if c["type"] == "moved" else c["object"]["centroid"]
        extra = (f"-> ({c['after']['centroid'][0]:+.2f}, {c['after']['centroid'][1]:+.2f}) {c['distance_m']:.2f} m, "
                 f"turned {c['turn_deg']:.0f} deg" if c["type"] == "moved" else f"size {max(c['object']['lengths']):.2f} m")
        print(f"  {c['id']:<5} {c['type']:<8} {c['evidence_matches']:5d} matches / {c['view_pairs']:2d} view pairs  at "
              f"({where[0]:+.2f}, {where[1]:+.2f}, {where[2]:.2f}) {extra}")
    np.savez_compressed(work / "outliers.npz", src=src, dst=dst, pair=pair, frame_a=frame_a, frame_b=frame_b,
                        pix_a=pix_a, pix_b=pix_b, gone_a=gone_a, new_b=new_b)
    (work / "outlier_changes.json").write_text(json.dumps({
        "matches": n_matches, "static_share": round(n_static / n_matches, 3), "outliers": int(len(src)),
        "moved_surfaces": int(both.sum()), "vanished": int(only_a.sum()), "appeared": int(only_b.sum()),
        "runtime_s": round(time.time() - started, 1), "changes": changes}, indent=1))
    print(f"{len(changes)} changes ({time.time() - started:.1f}s)")

    fig, ax = plt.subplots(figsize=(8, 10), dpi=110)
    ax.scatter(src[~(gone_a | new_b), 0], src[~(gone_a | new_b), 1], s=1, c="#dddddd")
    for c in changes:
        idx = np.array(c["idx"])
        if c["type"] == "moved":
            ax.scatter(src[idx, 0], src[idx, 1], s=4, c="#d7263d")
            ax.scatter(dst[idx, 0], dst[idx, 1], s=4, c="#1b9e4b")
            ax.annotate(c["id"], xy=c["after"]["centroid"][:2], xytext=c["before"]["centroid"][:2],
                        arrowprops=dict(arrowstyle="-|>", color="#2f5bd3", lw=2))
        else:
            pts = src[idx] if c["type"] == "removed" else dst[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c="#d7263d" if c["type"] == "removed" else "#1b9e4b", marker="x")
            ax.annotate(c["id"], c["object"]["centroid"][:2], fontsize=8)
    ax.set_aspect("equal")
    ax.set_title("Loop-closure outliers: moved (arrows), vanished (red x), appeared (green x)")
    fig.tight_layout()
    (work / "qa").mkdir(exist_ok=True)
    fig.savefig(work / "qa" / "outliers.png")


if __name__ == "__main__":
    main()
