"""scene_graph, stage 2: compare the two scene graphs, every claim verified by visibility.

Every object node of one recording is first checked against the OTHER recording's depth maps:
  present   the other recording sees a surface where the node is
  gone      it sees through the node's space (its depth lies clearly behind it)
  unknown   it never looked there (occluded / out of view): no claim either way
Graph matching then only has to explain the nodes that are gone (before) or appeared (after):
  moved     a gone node and an appeared node that look alike (SigLIP 2) and have similar size;
            touching gone (or appeared) nodes are first grouped into one object -- segmentation
            often splits one thing (a chair with a jacket on it) into several nodes
  rotated   a node matched in place whose elongated footprint turned, with part of its old
            footprint now seen empty (a turned object leaves space behind)
  removed / added   gone / appeared nodes with no look-alike on the other side
Requiring the space to be seen empty is what keeps fragmented or missed nodes from turning
into fake changes: a node that is merely unmatched, but still visibly there, is unchanged.
Relations from the graphs (on / next to) describe where a moved object was and is.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from navdelta import config, context
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"                  # shared reconstruction (navdelta.recon)
OUTPUT = context.approach_dir("scene_graph")

VOXEL_M = 0.05
CONF_MIN = 1.5
SEE_PAST_M = 0.10           # the other depth must lie this far behind a point to call it empty ...
SEE_PAST_PER_M = 0.0        # ... plus this much per metre of range (large spaces)
SURFACE_M = 0.10
SURFACE_PER_M = 0.0
OBSERVED_MIN = 0.30         # share of a node's points the other recording saw at all
GONE_MIN = 0.50             # share of the observed points seen empty (a thin object on the floor keeps ~half floor hits)
PRESENT_MIN = 0.40
STATIC_OVERLAP = 0.30
STATIC_SEM = 0.60
MOVE_SEM = 0.70
SIZE_RATIO_MAX = 2.2
MOVE_MIN_M = 0.20
MOVE_MAX_M = 1.0e9          # farther look-alike pairs are reported as removed + added
MIN_CHANGE_OBS = 3          # keyframes that saw a node before it can be reported as changed
MIN_CHANGE_SIZE_M = 0.15
ELONGATION_MIN = 1.5        # footprint length / width needed for a meaningful heading
ROTATION_MIN_DEG = 30.0
ROTATION_EMPTY_MIN = 0.25   # share of the old footprint seen empty after a turn
GROUP_GAP_M = 0.25          # gone (or appeared) nodes closer than this are one object (a jacket on a chair)
LABEL_BONUS = 0.10          # added to the look-alike score when both ends carry the same name


# Scale profile (navdelta/configs/*.toml): the values above are the room profile.
PROFILE_SETTINGS = config.apply(globals(), "scene_graph.diff")


def keys(points: np.ndarray) -> set:
    return set(map(tuple, np.floor(points / VOXEL_M).astype(np.int64)))


def dilate(cells: set) -> set:
    out = set()
    for x, y, z in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.add((x + dx, y + dy, z + dz))
    return out


def size(node: dict) -> np.ndarray:
    return np.sort(np.maximum(node["bbox_max"] - node["bbox_min"], VOXEL_M))


def heading(node: dict) -> tuple[float, float]:
    """Principal axis of the node's floor footprint (degrees, mod 180) and its elongation."""
    xy = node["voxels"][:, :2]
    if len(xy) < 4:
        return 0.0, 1.0
    evals, evecs = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
    axis = evecs[:, -1]
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0), float(np.sqrt(evals[-1] / max(evals[0], 1e-6)))


def turn(a: dict, b: dict) -> float | None:
    (ha, ea), (hb, eb) = heading(a), heading(b)
    if min(ea, eb) < ELONGATION_MIN:
        return None
    d = abs(ha - hb) % 180.0
    return float(min(d, 180.0 - d))


def visibility(points: np.ndarray, data: dict, rec: str) -> dict:
    """Per point: did the other recording see through it (empty) or see a surface at it?"""
    empty = np.zeros(len(points), bool)
    surface = np.zeros(len(points), bool)
    for k in np.flatnonzero(data["owner"] == rec):
        w2c = np.linalg.inv(data["cam2w"][k])
        cam = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        K = data["K"][k]
        h, w = data["depth"][k].shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2])
            v = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2])
        ok = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not ok.any():
            continue
        u = np.where(ok, u, 0).astype(int)
        v = np.where(ok, v, 0).astype(int)
        measured = data["depth"][k][v, u]
        good = ok & (data["conf"][k][v, u] > CONF_MIN)
        empty |= good & (measured > z + SEE_PAST_M + SEE_PAST_PER_M * z)
        surface |= good & (np.abs(measured - z) < SURFACE_M + SURFACE_PER_M * z)
    observed = empty | surface
    n_obs = max(1, int(observed.sum()))
    return {"observed": round(float(observed.mean()), 3),
            "empty": round(float((empty & ~surface).sum() / n_obs), 3),
            "surface": round(float(surface.sum() / n_obs), 3)}


def state(vis: dict) -> str:
    if vis["observed"] < OBSERVED_MIN:
        return "unknown"
    if vis["empty"] >= GONE_MIN:
        return "gone"
    if vis["surface"] >= PRESENT_MIN:
        return "present"
    return "partial"


def neighbours(graph: dict, index: int) -> list[str]:
    out = []
    for e in graph["edges"]:
        if e["source"] == index:
            out.append(f"{e['relation']} {graph['nodes'][e['target']]['label']}")
        elif e["target"] == index and e["relation"] == "next to":
            out.append(f"next to {graph['nodes'][e['source']]['label']}")
    return sorted(set(out))


def joint_label(a: dict, b: dict) -> str:
    """One name for an object seen at two places: the label votes of both ends together."""
    votes: dict[str, int] = {}
    for node in (a, b):
        for label, n in node["label_votes"].items():
            votes[label] = votes.get(label, 0) + n
    return max(sorted(votes), key=votes.get)


def group(nodes: list[dict], candidates: list[int]) -> list[list[int]]:
    """Groups of candidate nodes whose voxels come within GROUP_GAP_M of each other."""
    parent = {i: i for i in candidates}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    from scipy.spatial import cKDTree
    trees = {i: cKDTree(nodes[i]["voxels"]) for i in candidates}
    for x, i in enumerate(candidates):
        for j in candidates[x + 1:]:
            if find(i) != find(j) and np.isfinite(
                    trees[i].query(nodes[j]["voxels"], distance_upper_bound=GROUP_GAP_M + VOXEL_M)[0]).any():
                parent[find(j)] = find(i)
    out: dict[int, list[int]] = {}
    for i in candidates:
        out.setdefault(find(i), []).append(i)
    return sorted(out.values())


def combine(nodes: list[dict], members: list[int]) -> dict:
    """One node from several touching nodes of the same recording."""
    if len(members) == 1:
        return {**nodes[members[0]], "members": members}
    vox = np.concatenate([nodes[m]["voxels"] for m in members])
    frames = set().union(*(set(nodes[m]["frames"]) for m in members))
    emb = sum(nodes[m]["embedding"] * nodes[m]["observations"] for m in members)
    votes: dict[str, int] = {}
    for m in members:
        for label, n in nodes[m]["label_votes"].items():
            votes[label] = votes.get(label, 0) + n
    return {"label": max(sorted(votes), key=votes.get), "label_votes": votes, "observations": len(frames),
            "frames": sorted(frames), "embedding": emb / np.linalg.norm(emb), "voxels": vox,
            "centroid": vox.mean(0), "bbox_min": vox.min(0) - VOXEL_M / 2, "bbox_max": vox.max(0) + VOXEL_M / 2,
            "members": members}


def neighbours_of(graph: dict, members: list[int]) -> list[str]:
    names = {graph["nodes"][m]["label"] for m in members}
    return sorted({r for m in members for r in neighbours(graph, m)} - {f"next to {n}" for n in names}
                  - {f"on {n}" for n in names})


def reportable(node: dict) -> bool:
    return node["observations"] >= MIN_CHANGE_OBS and float(size(node)[-1]) >= MIN_CHANGE_SIZE_M


def main() -> None:
    started = time.time()
    WORK = OUTPUT
    graphs = np.load(WORK / "graphs.npy", allow_pickle=True).item()
    data = dict(np.load(inside(RECON / "scene.npz")))
    rec_a, rec_b = list(dict.fromkeys(data["owner"]))
    A, B = graphs[rec_a]["nodes"], graphs[rec_b]["nodes"]
    vis_a = [visibility(n["voxels"], data, rec_b) for n in A]     # A's nodes, seen by B
    vis_b = [visibility(n["voxels"], data, rec_a) for n in B]     # B's nodes, seen by A
    st_a, st_b = [state(v) for v in vis_a], [state(v) for v in vis_b]

    ka = [keys(n["voxels"]) for n in A]
    kb = [keys(n["voxels"]) for n in B]
    sem = np.array([[float(a["embedding"] @ b["embedding"]) for b in B] for a in A]).reshape(len(A), len(B))
    overlap = np.array([[len(ka[i] & dilate(kb[j])) / max(1, min(len(ka[i]), len(kb[j]))) for j in range(len(B))]
                        for i in range(len(A))]).reshape(len(A), len(B))
    dist = np.array([[float(np.linalg.norm(a["centroid"][:2] - b["centroid"][:2])) for b in B] for a in A]
                    ).reshape(len(A), len(B))
    ratio = np.array([[float(max(size(a)[-1], size(b)[-1]) / min(size(a)[-1], size(b)[-1])) for b in B] for a in A]
                     ).reshape(len(A), len(B))

    # 1 in place: nodes still there that overlap and look alike (landmarks, relations, turns)
    score = np.where((overlap >= STATIC_OVERLAP) & (sem >= STATIC_SEM), overlap * sem, 0.0)
    for i, s in enumerate(st_a):
        if s == "gone":
            score[i] = 0.0
    for j, s in enumerate(st_b):
        if s == "gone":
            score[:, j] = 0.0
    static, rotated = [], []
    used_a, used_b = set(), set()
    if score.size:
        for i, j in zip(*linear_sum_assignment(-score)):
            if score[i, j] <= 0:
                continue
            used_a.add(int(i))
            used_b.add(int(j))
            rot = turn(A[i], B[j])
            if (rot is not None and rot >= ROTATION_MIN_DEG and vis_a[i]["empty"] >= ROTATION_EMPTY_MIN
                    and reportable(A[i]) and reportable(B[j])):
                rotated.append((int(i), int(j), rot))
            else:
                static.append((int(i), int(j)))

    # 2 moves: something left one place (gone) and a look-alike appeared elsewhere. Touching gone
    #   nodes are one object (and touching appeared nodes), so fragments do not become extra changes.
    groups_a = group(A, [i for i in range(len(A)) if st_a[i] == "gone" and i not in used_a])
    groups_b = group(B, [j for j in range(len(B)) if st_b[j] == "gone" and j not in used_b])
    gone = [g for g in (combine(A, m) for m in groups_a) if reportable(g)]
    new = [g for g in (combine(B, m) for m in groups_b) if reportable(g)]
    moved = []
    if gone and new:
        sub = np.array([[float(a["embedding"] @ b["embedding"]) for b in new] for a in gone])
        same_name = np.array([[a["label"] == b["label"] for b in new] for a in gone])
        gdist = np.array([[float(np.linalg.norm(a["centroid"][:2] - b["centroid"][:2])) for b in new] for a in gone])
        gratio = np.array([[float(max(size(a)[-1], size(b)[-1]) / min(size(a)[-1], size(b)[-1])) for b in new]
                           for a in gone])
        sub[(gratio > SIZE_RATIO_MAX) | (sub < MOVE_SEM) | (gdist > MOVE_MAX_M)] = 0.0
        sub = np.where(sub > 0, sub + LABEL_BONUS * same_name, 0.0)
        for r, c in zip(*linear_sum_assignment(-sub)):
            if sub[r, c] > 0:
                moved.append((r, c, float(gdist[r, c]), float(gone[r]["embedding"] @ new[c]["embedding"])))
    paired_a = {r for r, *_ in moved}
    paired_b = {c for _, c, *_ in moved}

    changes = []
    for r, c, d, sim in moved:
        a, b = gone[r], new[c]
        kind = "moved" if d >= MOVE_MIN_M else "rotated"
        changes.append({"type": kind, "label": joint_label(a, b), "label_before": a["label"], "label_after": b["label"],
                        "from": a["centroid"].round(3).tolist(), "to": b["centroid"].round(3).tolist(),
                        "at": b["centroid"].round(3).tolist(),
                        "distance_m": round(d, 3), "similarity": round(sim, 3),
                        "rotation_deg": turn(a, b), "before": neighbours_of(graphs[rec_a], a["members"]),
                        "after": neighbours_of(graphs[rec_b], b["members"]),
                        "near": neighbours_of(graphs[rec_b], b["members"]),
                        "node_before": a["members"][0], "node_after": b["members"][0],
                        "members_before": a["members"], "members_after": b["members"]})
    for i, j, rot in rotated:
        changes.append({"type": "rotated", "label": A[i]["label"], "label_after": B[j]["label"],
                        "from": A[i]["centroid"].round(3).tolist(), "to": B[j]["centroid"].round(3).tolist(),
                        "at": B[j]["centroid"].round(3).tolist(), "rotation_deg": round(rot, 1),
                        "distance_m": round(float(dist[i, j]), 3), "similarity": round(float(sem[i, j]), 3),
                        "near": neighbours(graphs[rec_b], j), "visibility_before_in_after": vis_a[i],
                        "node_before": i, "node_after": j})
    for r, a in enumerate(gone):
        if r not in paired_a:
            changes.append({"type": "removed", "label": a["label"], "at": a["centroid"].round(3).tolist(),
                            "visibility_in_other": vis_a[a["members"][0]], "near": neighbours_of(graphs[rec_a], a["members"]),
                            "node_before": a["members"][0], "members_before": a["members"]})
    for c, b in enumerate(new):
        if c not in paired_b:
            changes.append({"type": "added", "label": b["label"], "at": b["centroid"].round(3).tolist(),
                            "visibility_in_other": vis_b[b["members"][0]], "near": neighbours_of(graphs[rec_b], b["members"]),
                            "node_after": b["members"][0], "members_after": b["members"]})
    unknown = ([{"recording": rec_a, "label": A[i]["label"], "at": A[i]["centroid"].round(3).tolist(), "node": i}
                for i in range(len(A)) if st_a[i] == "unknown"] +
               [{"recording": rec_b, "label": B[j]["label"], "at": B[j]["centroid"].round(3).tolist(), "node": j}
                for j in range(len(B)) if st_b[j] == "unknown"])

    tally = lambda st: {s: st.count(s) for s in ("present", "partial", "gone", "unknown")}  # noqa: E731
    print(f"{rec_a}: {len(A)} nodes {tally(st_a)}; {rec_b}: {len(B)} nodes {tally(st_b)}; in place {len(static)}")
    for c in changes:
        extra = (f"{c['distance_m']:.2f} m" + (f", turned {c['rotation_deg']:.0f} deg" if c.get("rotation_deg") else "")
                 if c["type"] in ("moved", "rotated") else f"empty {c['visibility_in_other']['empty']:.2f}")
        print(f"  {c['type'].upper():<8} {c['label']:<22} at ({c['at'][0]:+.2f}, {c['at'][1]:+.2f})  {extra}  "
              f"near: {', '.join(c['near']) or '-'}")
    (WORK / "graph_diff.json").write_text(json.dumps({
        "unchanged": static, "changes": changes, "unknown": unknown,
        "node_states": {rec_a: [{"label": n["label"], "state": s, **v} for n, s, v in zip(A, st_a, vis_a)],
                        rec_b: [{"label": n["label"], "state": s, **v} for n, s, v in zip(B, st_b, vis_b)]}},
        indent=2))

    fig, ax = plt.subplots(figsize=(9, 11), dpi=110)
    for nodes, colour, marker in ((A, "#d7263d", "o"), (B, "#1b9e4b", "s")):
        for n in nodes:
            ax.scatter(n["voxels"][:, 0], n["voxels"][:, 1], s=2, c=colour, alpha=0.25, linewidths=0)
            ax.scatter(*n["centroid"][:2], s=30, c=colour, marker=marker)
            ax.annotate(n["label"], n["centroid"][:2], fontsize=6, color=colour)
    for i, j in static:
        ax.plot([A[i]["centroid"][0], B[j]["centroid"][0]], [A[i]["centroid"][1], B[j]["centroid"][1]], "-",
                color="#888888", lw=0.8)
    for c in changes:
        if c["type"] == "moved":
            ax.annotate("", xy=c["to"][:2], xytext=c["from"][:2], arrowprops=dict(arrowstyle="-|>", color="#2f5bd3", lw=2))
        elif c["type"] == "rotated":
            ax.scatter(*c["at"][:2], s=220, facecolors="none", edgecolors="#9b30d9", linewidths=2)
        else:
            ax.scatter(*c["at"][:2], s=220, marker="x", c="#d7263d" if c["type"] == "removed" else "#1b9e4b")
    ax.set_aspect("equal")
    ax.set_title(f"{rec_a} (red) vs {rec_b} (green): grey = in place, blue = moved, purple = turned, x = removed/added",
                 fontsize=8)
    fig.tight_layout()
    (WORK / "qa").mkdir(exist_ok=True)
    fig.savefig(WORK / "qa" / "graph_diff.png")


if __name__ == "__main__":
    main()
