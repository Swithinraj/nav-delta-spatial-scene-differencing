"""rgbd_fusion, stage 5: fuse the scene-graph diff with the loop-closure outliers.

Two independent detectors on one RGB-D reconstruction:
  scene graph   objects (Florence-2 + SAM 2.1 + SigLIP 2) whose space the other recording's depth
                sees through: semantic, whole outlines, names and relations; weak on motion
  outliers      cross-video feature matches whose two ends both moved (one rigid motion per object,
                by RANSAC) or that vanished / appeared: geometric, exact motion; partial outlines
As in semantic RGB-D SLAM (DynaSLAM, MaskFusion: instance masks + multi-view geometry), each covers
the other's blind spot. Changes are associated end by end: a "before" end (where something was)
of an outlier change belongs to the scene-graph object whose before end holds most of its outline
(ASSOC_SHARE of its points within ASSOC_M), the same for "after" ends; a move has both ends, so
one outlier move can join a graph "removed" and a graph "added" into one move. A look-alike pairing
(vanished here, a similar thing appeared there; no RANSAC motion) with only one end on a graph
object is split, and its other end is judged on its own. Each cluster is then decided:
  both detectors     confirmed: the graph's object (outline, name, relations) with the outliers'
                     measured motion (distance and turn from the object's own features)
  scene graph only   kept if its own evidence is strong enough (share seen through, views, look-alike)
  outliers only      kept if strong (matches from enough view pairs, as in approach 3) and the graph
                     does not see an unchanged object in the same place (veto)
  next to a mirror   one detector alone is not trusted beside a mirror, window or glass: depth
                     sensors see through them, so their depth shows space that is not there
Writes changes.json (shared schema) + objects.npz + CHANGE_REPORT.md.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from navdelta import config, context
from navdelta.approaches.scene_graph import describe as sg_describe
from navdelta.approaches.scene_graph import diff as sg_diff
from navdelta.common.wording import It, is_, it, stands, was, without_self
from navdelta.context import inside

RECON = context.OUTPUTS / "recon_rgbd"
OUTPUT = context.approach_dir("rgbd_fusion")

VOXEL_M = 0.05                # node voxels (as the objects stage)
ASSOC_M = 0.15                # an outline point this close to a graph object's voxels lies on it ...
ASSOC_SHARE = 0.30            # ... and an outlier change whose outline lies this much on it is that object
MOVE_MIN_M = 0.20             # a move shorter than this, with a turn, is "turned in place"
TURN_MIN_DEG = 30.0
GRAPH_ALONE_EMPTY = 0.0       # graph-only removed / added: share of the object seen through ...
GRAPH_ALONE_OBS = 0           # ... and keyframes that saw it
GRAPH_ALONE_SEM = 0.0         # graph-only move: look-alike score of its two ends
GRAPH_ALONE_ROTATED = True    # graph-only "turned in place"
ALONE_MIN_MATCHES = 60        # outliers-only removed / added: matches ...
ALONE_MIN_PAIRS = 4           # ... from this many view pairs (approach 3's corroboration rule)
VETO_PRESENT = 0.5            # outliers-only: vetoed when unchanged graph objects cover this share of it
NEXT_TO_M = 0.30              # relations of an object the graph has no node for
ON_GAP_M = 0.15
REFLECTIVE = ["mirror", "window", "glass", "glass door", "window frame"]   # depth sees through them
LANDMARK_MIN_SIZE_M = 0.5     # landmark distances in the descriptions (as scene_graph.describe)
LANDMARK_RANGE_M = 1.5
LANDMARK_CHANGE_M = 0.25

# Scale profile (navdelta/configs/*.toml): the values above are the room profile.
PROFILE_SETTINGS = config.apply(globals(), "rgbd_fusion.fuse")
vars(sg_describe).update(VOXEL_M=VOXEL_M, LANDMARK_MIN_SIZE_M=LANDMARK_MIN_SIZE_M,
                         LANDMARK_RANGE_M=LANDMARK_RANGE_M, LANDMARK_CHANGE_M=LANDMARK_CHANGE_M)
ORDER = ("moved", "rotated", "removed", "added")


# ---------------------------------------------------------------- the two detectors' changes
def graph_candidates(diff: dict, graphs: dict, rec_a: str, rec_b: str) -> list[dict]:
    A, B = graphs[rec_a]["nodes"], graphs[rec_b]["nodes"]
    out = []
    for c in diff["changes"]:
        ends = {}
        for side, nodes in (("before", A), ("after", B)):
            members = c.get(f"members_{side}") or ([c[f"node_{side}"]] if f"node_{side}" in c else None)
            if members:
                n = sg_diff.combine(nodes, members)
                ends[side] = {"min": n["bbox_min"], "max": n["bbox_max"], "centroid": n["centroid"],
                              "points": n["voxels"], "node": n, "members": members}
        out.append({"source": "scene_graph", "type": c["type"], "label": c["label"], "ends": ends, "change": c})
    return out


def outlier_candidates(seg: dict, shapes: dict) -> list[dict]:
    out = []
    for r in seg["changes"]:
        ends = {}
        for side in ("before", "after"):
            geo = r.get(side) or (r["object"] if r.get("object") and side == (
                "before" if r["type"] == "removed" else "after") else None)
            if not geo:
                continue
            pts = shapes.get(f"{r['id']}_{side}")
            if pts is None or len(pts) < 4:
                pts = np.array([geo["bbox_min_m"], geo["bbox_max_m"]], np.float32)
            ends[side] = {"min": np.array(geo["bbox_min_m"]), "max": np.array(geo["bbox_max_m"]),
                          "centroid": np.array(geo["centroid_m"]), "points": pts.astype(np.float32)}
        out.append({"source": "outliers", "type": r["type"], "label": r["label"], "ends": ends, "change": r})
    return out


def box_gap(a: dict, b: dict) -> float:
    sep = np.maximum(0.0, np.maximum(a["min"] - b["max"], b["min"] - a["max"]))
    return float(np.linalg.norm(sep))


def on_object(outline: dict, obj: dict) -> float:
    """Share of an outlier change's outline lying on a graph object (same recording, same side)."""
    if box_gap(outline, obj) > ASSOC_M:
        return 0.0
    d = cKDTree(obj["points"]).query(outline["points"], distance_upper_bound=ASSOC_M)[0]
    return float(np.isfinite(d).mean())


def clusters(graph: list[dict], outliers: list[dict]) -> list[tuple[list[int], list[int]]]:
    """(graph changes, outlier changes) that are one object. Every outlier end goes to the graph
    object holding most of its outline; an outlier move joins two graph changes only when they are
    a "removed" (its before end) and an "added" (its after end). An outlier "move" that is only a
    look-alike pairing (vanished here, similar thing appeared there; not measured by RANSAC) and
    lies on a graph object with one end only is split: its other end becomes its own candidate
    (identical boxes make such pairings unreliable, but each end may still be a real change); the
    new candidate is appended to `outliers`."""
    owner = list(range(len(graph)))               # graph change -> its cluster

    def find(i: int) -> int:
        while owner[i] != i:
            i = owner[i]
        return i
    attached: dict[int, list[int]] = {}
    alone = []
    for oi, o in enumerate(outliers):
        best = {}
        for side, end in o["ends"].items():
            scores = [(on_object(end, g["ends"][side]), gi) for gi, g in enumerate(graph) if side in g["ends"]]
            share, gi = max(scores, default=(0.0, None))
            if share >= ASSOC_SHARE:
                best[side] = (share, gi)
        if not best:
            alone.append(oi)
            continue
        targets = {gi for _, gi in best.values()}
        if len(targets) == 2 and graph[best["before"][1]]["type"] == "removed" \
                and graph[best["after"][1]]["type"] == "added":
            owner[find(best["after"][1])] = find(best["before"][1])      # one object moved
            target = best["before"][1]
        else:
            target = max(best.values())[1]
            linked = {side for side, (_, gi) in best.items() if gi == target}
            if (o["type"] == "moved" and o["change"]["evidence"].get("paired_by_appearance") is not None
                    and len(linked) == 1):
                (other,) = set(o["ends"]) - linked
                outliers.append({**o, "type": "added" if other == "after" else "removed",
                                 "ends": {other: o["ends"][other]}, "split": True})
                o.update(type="removed" if other == "after" else "added",
                         ends={side: o["ends"][side] for side in linked}, split=True)
        attached.setdefault(target, []).append(oi)
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for gi in range(len(graph)):
        groups.setdefault(find(gi), ([], []))[0].append(gi)
    for gi, ois in attached.items():
        groups[find(gi)][1].extend(ois)
    return list(groups.values()) + [([], [oi]) for oi in alone]


# ---------------------------------------------------------------- decisions
def in_place(graphs: dict, states: dict) -> dict:
    """Per recording, a KD-tree over the voxels of the graph objects the other recording still sees."""
    out = {}
    for rec, graph in graphs.items():
        present = [n["voxels"] for n, s in zip(graph["nodes"], states[rec]) if s["state"] == "present"]
        out[rec] = cKDTree(np.concatenate(present)) if present else None
    return out


def present_share(points: np.ndarray, tree: cKDTree | None) -> float:
    """Share of an outline lying on graph objects the other recording still sees in place."""
    if tree is None or not len(points):
        return 0.0
    return float((tree.query(points)[0] <= VOXEL_M).mean())


def beside_reflection(c: dict, graphs: dict, rec_a: str, rec_b: str) -> str | None:
    """A reflective object (mirror, window, glass) the change is on or next to, if any."""
    if c["label"] in REFLECTIVE:
        return c["label"]
    for side, rec in (("before", rec_a), ("after", rec_b)):
        if side in c["ends"]:
            e = c["ends"][side]
            relations = (sg_diff.neighbours_of(graphs[rec], e["members"]) if "members" in e
                         else relations_of(e["points"], graphs[rec]))
            for r in relations:
                name = r.split(" ", 2)[-1] if r.startswith("next to ") else r[3:]
                if name in REFLECTIVE:
                    return name
    return None


def graph_alone_ok(c: dict) -> tuple[bool, str]:
    ch = c["change"]
    if c["type"] in ("removed", "added"):
        node = next(iter(c["ends"].values()))["node"]
        empty = ch["visibility_in_other"]["empty"]
        if empty < GRAPH_ALONE_EMPTY or node["observations"] < GRAPH_ALONE_OBS:
            return False, f"scene graph only, weak ({empty:.0%} seen through, {node['observations']} views)"
    elif c["type"] == "moved" and ch.get("similarity", 1.0) < GRAPH_ALONE_SEM:
        return False, f"scene graph only, ends look alike only {ch['similarity']:.2f}"
    elif c["type"] == "rotated" and not GRAPH_ALONE_ROTATED:
        return False, "scene graph only turn"
    return True, ""


def outliers_alone_ok(c: dict, unchanged: dict, rec_a: str, rec_b: str) -> tuple[bool, str]:
    ev = c["change"]["evidence"]
    if c["type"] != "moved" and (not ev.get("paired_by_appearance") or c.get("split")) and (
            ev["outlier_matches"] < ALONE_MIN_MATCHES or ev["view_pairs"] < ALONE_MIN_PAIRS):
        return False, f"outliers only, weak ({ev['outlier_matches']} matches, {ev['view_pairs']} view pairs)"
    shares = [present_share(e["points"], unchanged[rec])
              for side, rec in (("before", rec_a), ("after", rec_b)) if (e := c["ends"].get(side))]
    if shares and min(shares) >= VETO_PRESENT:
        return False, f"outliers only; the scene graph sees an unchanged object there ({min(shares):.0%})"
    return True, ""


# ---------------------------------------------------------------- description
def relations_of(points: np.ndarray, graph: dict) -> list[str]:
    """'on X' / 'next to Y' for an outline the graph has no node for, nearest first."""
    if not len(points):
        return []
    tree = cKDTree(points)
    lo, hi = points.min(0), points.max(0)
    rows = []
    for node in graph["nodes"]:
        d = tree.query(node["voxels"])[0]
        if (d <= VOXEL_M).mean() > 0.5:           # the object itself
            continue
        gap = max(0.0, float(d.min()) - VOXEL_M)
        if gap > NEXT_TO_M:
            continue
        overlap = (np.minimum(hi[:2], node["bbox_max"][:2]) - np.maximum(lo[:2], node["bbox_min"][:2])).clip(0).prod()
        on = (0 <= lo[2] - node["bbox_max"][2] + 0.1 <= ON_GAP_M + 0.1
              and overlap >= 0.3 * max((hi[:2] - lo[:2]).prod(), 1e-6) and lo[2] > node["bbox_min"][2] + 0.1)
        rows.append((gap, f"{'on' if on else 'next to'} {node['label']}"))
    return list(dict.fromkeys(r for _, r in sorted(rows)))


def geometry(end: dict) -> dict:
    out = {"centroid_m": np.asarray(end["centroid"]).round(3).tolist(),
           "bbox_min_m": np.asarray(end["min"]).round(3).tolist(), "bbox_max_m": np.asarray(end["max"]).round(3).tolist()}
    if "node" in end:
        out["observations"] = end["node"]["observations"]
    return out


def fuse_cluster(members: list[dict], graphs: dict, rec_a: str, rec_b: str) -> dict:
    g = [m for m in members if m["source"] == "scene_graph"]
    o = [m for m in members if m["source"] == "outliers"]
    ends = {}
    for side in ("before", "after"):                 # the graph's whole object first, else the outline
        for group in (g, o):
            have = [m["ends"][side] for m in group if side in m["ends"]]
            if have:
                ends[side] = max(have, key=lambda e: len(e["points"]))
                break
    moved_o = sorted((m for m in o if m["type"] == "moved"), key=lambda m: -m["change"]["evidence"]["outlier_matches"])
    turned_g = [m["change"].get("rotation_deg") for m in g if m["change"].get("rotation_deg") is not None]
    motion = None
    if "before" in ends and "after" in ends:
        shift = ends["after"]["centroid"] - ends["before"]["centroid"]
        distance = float(np.linalg.norm(shift[:2]))
        turn = None
        # measured from the object's own features (RANSAC), not from a look-alike pairing of outlines
        measured = [m for m in moved_o if m["change"]["evidence"].get("paired_by_appearance") is None]
        if measured:
            distance = measured[0]["change"]["motion"]["distance_m"]
            turn = measured[0]["change"]["motion"].get("turn_deg")
        elif turned_g:
            turn = turned_g[0]
        kind = "moved" if distance >= MOVE_MIN_M or not (turn and turn >= TURN_MIN_DEG) else "rotated"
        motion = {"translation_m": shift.round(3).tolist(), "distance_m": round(distance, 3),
                  "rotation_deg": round(turn, 1) if turn is not None else None,
                  "measured_by": "its own matched features (RANSAC)" if measured else
                  "object centres (scene graph)" if g else "object outlines"}
    else:
        kind = "removed" if "before" in ends else "added"
    # name: the graph's votes over all its observations; else SigLIP on the outline
    nodes = [e["node"] for e in ends.values() if "node" in e]
    if nodes:
        votes: dict[str, int] = {}
        for n in nodes:
            for label, k in n["label_votes"].items():
                votes[label] = votes.get(label, 0) + k
        label = max(sorted(votes), key=votes.get)
    else:
        label = (moved_o or o)[0]["label"]
    relations = {}
    for side, rec in (("before", rec_a), ("after", rec_b)):
        if side in ends:
            e = ends[side]
            relations[side] = without_self(sg_diff.neighbours_of(graphs[rec], e["members"]) if "members" in e
                                           else relations_of(e["points"], graphs[rec]), label)   # not "chair next to office chair"
    return {"type": kind, "label": label, "ends": ends, "motion": motion, "relations": relations,
            "sources": sorted({m["source"] for m in members}), "members": members}


def evidence(f: dict) -> tuple[str, dict]:
    parts, data = [], {}
    for m in f["members"]:
        c = m["change"]
        if m["source"] == "scene_graph":
            vis = c.get("visibility_in_other") or c.get("visibility_before_in_after")
            text = f"the scene graphs ({c['type']}" + (f", {vis['empty']:.0%} of it seen through" if vis else "") + \
                   (f", ends alike {c['similarity']:.2f}" if c.get("similarity") is not None else "") + ")"
            data.setdefault("scene_graph", []).append({"type": c["type"], "label": c["label"],
                                                       "visibility_in_other": vis, "similarity": c.get("similarity")})
        else:
            ev = c["evidence"]
            text = f"{ev['outlier_matches']} cross-video features from {ev['view_pairs']} view pairs ({m['type']})"
            data.setdefault("outliers", []).append({"id": c["id"], "type": m["type"], **ev})
        parts.append(text)
    lead = "Both detectors agree: " if len(f["sources"]) == 2 else "Found by "
    return lead + sg_describe.join(parts) + ".", data


def describe(f: dict, marks: list[dict]) -> tuple[str, str]:
    label, rel = f["label"], f["relations"]
    if f["type"] == "moved":
        m = f["motion"]
        turned = m["rotation_deg"] if m["rotation_deg"] and m["rotation_deg"] >= TURN_MIN_DEG else None
        title = f"{label} moved {m['distance_m']:.2f} m" + (f", turned ~{turned:.0f}°" if turned else "")
        text = [f"The {label} {was(label)} moved {m['distance_m']:.2f} m"
                + (f" and turned by about {turned:.0f}°" if turned else "") + ".",
                f"Before, {it(label)} {was(label)} {sg_describe.place(rel.get('before', []))}; now {it(label)} "
                f"{is_(label)} {sg_describe.place(rel.get('after', []))}."]
        text += sg_describe.distance_story(f["ends"]["before"]["points"], f["ends"]["after"]["points"], marks, label)
    elif f["type"] == "rotated":
        m = f["motion"]
        title = f"{label} turned ~{m['rotation_deg']:.0f}° in place"
        text = [f"The {label} {was(label)} turned by about {m['rotation_deg']:.0f}° where {it(label)} "
                f"{stands(label)} (the centre moved only {m['distance_m']:.2f} m).",
                f"{It(label)} {is_(label)} {sg_describe.place(rel.get('after', []))}."]
    elif f["type"] == "removed":
        title = f"{label} removed"
        text = [f"The {label} that {was(label)} {sg_describe.place(rel.get('before', []))} {is_(label)} no longer "
                f"there."]
    else:
        title = f"{label} added"
        text = [sg_describe.added_sentence(label, rel.get("after", []))]
    return title, " ".join(text)


# ---------------------------------------------------------------- main
def main() -> None:
    started = time.time()
    work = OUTPUT
    graphs = np.load(work / "graphs.npy", allow_pickle=True).item()
    diff = json.loads((work / "graph_diff.json").read_text(encoding="utf-8"))
    seg = json.loads((work / "outlier_objects.json").read_text(encoding="utf-8"))
    shapes = dict(np.load(work / "outlier_objects.npz"))
    report = json.loads((inside(RECON / "recon_report.json")).read_text(encoding="utf-8"))
    rec_a, rec_b = list(graphs)
    unchanged = in_place(graphs, diff["node_states"])
    graph_cands, outlier_cands = graph_candidates(diff, graphs, rec_a, rec_b), outlier_candidates(seg, shapes)
    marks = sg_describe.landmarks(graphs, diff["unchanged"], rec_a, rec_b)

    fused, dropped = [], []
    for gs, os_ in clusters(graph_cands, outlier_cands):
        members = [graph_cands[i] for i in gs] + [outlier_cands[i] for i in os_]
        if len({m["source"] for m in members}) == 1:
            c = members[0]
            ok, why = (graph_alone_ok(c) if c["source"] == "scene_graph"
                       else outliers_alone_ok(c, unchanged, rec_a, rec_b))
            mirror = beside_reflection(c, graphs, rec_a, rec_b) if ok else None
            if mirror:
                ok, why = False, f"{c['source'].replace('_', ' ')} only, beside a {mirror} (depth sees through it)"
            if not ok:
                dropped.append({"source": c["source"], "type": c["type"], "label": c["label"], "why": why,
                                "at_m": np.asarray(next(iter(c["ends"].values()))["centroid"]).round(3).tolist()})
                continue
        fused.append(fuse_cluster(members, graphs, rec_a, rec_b))

    records, shapes_out, counts = [], {}, {}
    for f in sorted(fused, key=lambda f: (ORDER.index(f["type"]), -len(f["sources"]))):
        counts[f["type"]] = counts.get(f["type"], 0) + 1
        rid = {"moved": "M", "rotated": "R", "removed": "OR", "added": "OA"}[f["type"]] + str(counts[f["type"]])
        title, text = describe(f, marks)
        why, ev = evidence(f)
        rec = {"id": rid, "type": f["type"], "title": title, "label": f["label"], "description": f"{text} {why}",
               "sentence": text, "evidence_text": why, "found_by": f["sources"],
               "confirmed": len(f["sources"]) == 2, "evidence": ev}
        if f["type"] in ("moved", "rotated"):
            rec.update(before=geometry(f["ends"]["before"]), after=geometry(f["ends"]["after"]), motion=f["motion"],
                       relations=f["relations"])
            rec["arrow"] = {"from": rec["before"]["centroid_m"], "to": rec["after"]["centroid_m"]}
        else:
            side = "before" if f["type"] == "removed" else "after"
            rec.update(object={**geometry(f["ends"][side]), "present_in": rec_a if side == "before" else rec_b},
                       place={"near": f["relations"].get(side, [])})
        for side, e in f["ends"].items():
            shapes_out[f"{rid}_{side}"] = np.asarray(e["points"], np.float32)
        records.append(rec)

    parts = []
    for kind, word in (("moved", "moved"), ("rotated", "turned in place"), ("removed", "removed"), ("added", "added")):
        items = [r["label"] for r in records if r["type"] == kind]
        if items:
            parts.append(f"{len(items)} object{'s' if len(items) > 1 else ''} {word} ({', '.join(items)})")
    confirmed = sum(r["confirmed"] for r in records)
    summary = ("Between the two recordings, " + sg_describe.join(parts) + "." if parts else "No changes were found.") + (
        f" {confirmed} of {len(records)} found by both detectors." if records else "")
    out = {"schema_version": "1.0.0", "approach": "rgbd_fusion",
           "title": "Changes between two phone recordings of the same room",
           "recordings": {"before": rec_a, "after": rec_b},
           "common_frame": f"{rec_a} tracking frame after the joint pose graph (RGB-D); metres; Z up; floor at z = 0",
           "summary": summary, "counts": counts, "changes": records, "dropped": dropped,
           "detectors": {"scene_graph": len(diff["changes"]), "outliers": len(seg["changes"]),
                         "loop_closure": seg.get("loop_closure")},
           "reconstruction": {k: report.get(k) for k in ("depth_source", "runtime_s") if k in report},
           "models": sg_describe.MODELS, "profile": config.summary(),
           "learned_models_trained_or_finetuned": False, "manual_labels_used": False}
    out["result_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()
    (work / "changes.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(work / "objects.npz", **shapes_out)
    lines = ["# What changed between the two recordings", "", summary, ""] + \
            [f"- **{r['id']}** ({r['type']}): {r['description']}" for r in records]
    if dropped:
        lines += ["", "Not reported:"] + [f"- {d['source']} {d['type']} {d['label']}: {d['why']}" for d in dropped]
    (work / "CHANGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"fused {len(graph_cands) + len(outlier_cands)} candidates ({len(graph_cands)} scene graph, "
          f"{len(outlier_cands)} outliers) into "
          f"{len(records)} changes, {len(dropped)} dropped ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
