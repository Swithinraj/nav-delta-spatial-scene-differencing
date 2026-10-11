"""scene_graph, stage 3: graph diff -> plain-English change descriptions + changes.json (shared schema).

Every sentence is built from the two scene graphs: the object's label (Florence-2 vote across
its observations), its relations ("on", "next to") before and after, and how its distance to
unchanged landmarks (nodes matched in place in both graphs) changed.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from navdelta import config, context
from navdelta.common.wording import It, article, is_, it, stands, was, without_self
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"                  # shared reconstruction (navdelta.recon)
OUTPUT = context.approach_dir("scene_graph")

VOXEL_M = 0.05
MODELS = [
    {"id": "florence-community/Florence-2-large", "revision": "4271c66b88cdbc05735372ec13b2360108de5317",
     "license": "MIT", "use": "object proposals with names"},
    {"id": "facebook/sam2.1-hiera-base-plus", "revision": "b7320756a13354e7530a63935656d35b2f91a290",
     "license": "Apache-2.0", "use": "masks (boxes and point grid)"},
    {"id": "google/siglip2-so400m-patch14-384", "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
     "license": "Apache-2.0", "use": "appearance embeddings, zero-shot names, pair selection"},
    {"id": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", "revision": "pinned source f5209afc",
     "license": "CC BY-NC-SA 4.0", "use": "depth and cross-video matches (shared reconstruction)"},
]
LANDMARK_MIN_SIZE_M = 0.5     # a landmark is a large unchanged object (table, bed, wardrobe)
LANDMARK_RANGE_M = 1.5
LANDMARK_CHANGE_M = 0.25


# Scale profile (navdelta/configs/*.toml): the values above are the room profile.
PROFILE_SETTINGS = config.apply(globals(), "scene_graph.describe")


def gap(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, float(cKDTree(b).query(a)[0].min()) - VOXEL_M)


def join(items: list[str]) -> str:
    items = list(dict.fromkeys(items))
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def place(relations: list[str]) -> str:
    on = [r[3:] for r in relations if r.startswith("on ")]
    near = [r[8:] for r in relations if r.startswith("next to ")]
    parts = []
    if on:
        parts.append(f"on the {on[0]}")
    if near:
        parts.append("next to the " + join(near[:3]).replace(" and ", " and the ").replace(", ", ", the "))
    return " and ".join(parts) if parts else "free-standing"


def landmarks(graphs: dict, unchanged: list, rec_a: str, rec_b: str) -> list[dict]:
    out = []
    for i, j in unchanged:
        a, b = graphs[rec_a]["nodes"][i], graphs[rec_b]["nodes"][j]
        if float(np.max(a["bbox_max"][:2] - a["bbox_min"][:2])) >= LANDMARK_MIN_SIZE_M:
            out.append({"label": a["label"], "before": a["voxels"], "after": b["voxels"]})
    return out


def distance_story(before: np.ndarray, after: np.ndarray, marks: list[dict], exclude: str) -> list[str]:
    """Sentences for the landmarks the object got clearly closer to / further from."""
    rows = []
    for m in marks:
        if m["label"] == exclude:
            continue
        g0, g1 = gap(before, m["before"]), gap(after, m["after"])
        if min(g0, g1) <= LANDMARK_RANGE_M and abs(g1 - g0) >= LANDMARK_CHANGE_M:
            rows.append((abs(g1 - g0), m["label"], g0, g1))
    seen, out = set(), []
    for _, label, g0, g1 in sorted(rows, reverse=True):
        if label in seen:
            continue
        seen.add(label)
        word = "closer to" if g1 < g0 else "further away from"
        out.append(f"{It(exclude)} {is_(exclude)} now {word} the {label}: {g1:.2f} m from it, compared with "
                   f"{g0:.2f} m before.")
        if len(out) == 2:
            break
    return out


def added_sentence(label: str, relations: list[str]) -> str:
    where = place(relations)
    if where == "free-standing":
        return f"{article(label)} appeared, away from other furniture; {it(label)} {was(label)} not there before."
    return f"{article(label)} now {stands(label)} {where}; {it(label)} {was(label)} not there before."


def node_geometry(node: dict) -> dict:
    return {"centroid_m": node["centroid"].round(3).tolist(), "bbox_min_m": node["bbox_min"].round(3).tolist(),
            "bbox_max_m": node["bbox_max"].round(3).tolist(), "observations": node["observations"]}


def main() -> None:
    started = time.time()
    work = OUTPUT
    graphs = np.load(work / "graphs.npy", allow_pickle=True).item()
    diff = json.loads((work / "graph_diff.json").read_text())
    rec_a, rec_b = list(graphs)
    A, B = graphs[rec_a]["nodes"], graphs[rec_b]["nodes"]
    from navdelta.approaches.scene_graph.diff import combine

    def node(c: dict, side: str) -> dict:
        nodes = A if side == "before" else B
        members = c.get(f"members_{side}") or [c[f"node_{side}"]]
        return combine(nodes, members)
    marks = landmarks(graphs, diff["unchanged"], rec_a, rec_b)
    records, unconfirmed = [], []
    counters = {"moved": 0, "rotated": 0, "removed": 0, "added": 0}
    prefix = {"moved": "M", "rotated": "R", "removed": "OR", "added": "OA"}
    for c in diff["changes"]:
        kind = c["type"]
        if kind == "not re-observed":
            unconfirmed.append(c)
            continue
        counters[kind] += 1
        rid = f"{prefix[kind]}{counters[kind]}"
        label = c["label"]
        for key in ("before", "after", "near"):          # relations to the object itself under another name
            if key in c:
                c[key] = without_self(c[key], label)
        if kind == "moved":
            a, b = node(c, "before"), node(c, "after")
            turned = c.get("rotation_deg")
            title = f"{label} moved {c['distance_m']:.2f} m" + (f", turned ~{turned:.0f}°" if turned and turned >= 30 else "")
            text = [f"The {label} {was(label)} moved {c['distance_m']:.2f} m"
                    + (f" and turned by about {turned:.0f}°" if turned and turned >= 30 else "") + "."]
            text.append(f"Before, {it(label)} {was(label)} {place(c['before'])}; now {it(label)} {is_(label)} "
                        f"{place(c['after'])}.")
            text += distance_story(a["voxels"], b["voxels"], marks, label)
            geo = {"before": node_geometry(a), "after": node_geometry(b),
                   "motion": {"translation_m": (b["centroid"] - a["centroid"]).round(3).tolist(),
                              "distance_m": c["distance_m"], "rotation_deg": turned},
                   "relations": {"before": c["before"], "after": c["after"]},
                   "arrow": {"from": c["from"], "to": c["to"]}}
        elif kind == "rotated":
            a, b = node(c, "before"), node(c, "after")
            title = f"{label} turned ~{c['rotation_deg']:.0f}° in place"
            text = [f"The {label} {was(label)} turned by about {c['rotation_deg']:.0f}° where {it(label)} "
                    f"{stands(label)} (the centre moved only {c['distance_m']:.2f} m).",
                    f"{It(label)} {is_(label)} {place(c['near'])}."]
            geo = {"before": node_geometry(a), "after": node_geometry(b),
                   "motion": {"translation_m": (b["centroid"] - a["centroid"]).round(3).tolist(),
                              "distance_m": c["distance_m"], "rotation_deg": c["rotation_deg"]},
                   "relations": {"after": c["near"]}}
        else:
            obj = node(c, "before") if kind == "removed" else node(c, "after")
            other = "second" if kind == "removed" else "first"
            vis = c["visibility_in_other"]
            title = f"{label} {'removed' if kind == 'removed' else 'added'}"
            text = [(f"The {label} that {was(label)} {place(c['near'])} {is_(label)} no longer there."
                     if kind == "removed" else added_sentence(label, c["near"])),
                    f"The {other} recording looked through that spot and saw past it ({vis['empty']:.0%} of its "
                    f"points), and no similar object appeared elsewhere."]
            geo = {"object": {**node_geometry(obj), "present_in": rec_a if kind == "removed" else rec_b},
                   "place": {"near": c["near"]}}
        records.append({"id": rid, "type": kind, "title": title, "label": label,
                        "label_votes": node(c, "before" if "node_before" in c else "after")["label_votes"],
                        "description": " ".join(text), **geo})

    parts = []
    for kind, word in (("moved", "moved"), ("rotated", "turned in place"), ("removed", "removed"), ("added", "added")):
        items = [r["label"] for r in records if r["type"] == kind]
        if items:
            parts.append(f"{len(items)} object{'s' if len(items) > 1 else ''} {word} ({', '.join(items)})")
    summary = ("Between the two recordings, " + join(parts) + "." if parts else "No changes were found.") + (
        f" {len(unconfirmed)} further objects were seen by one recording only, so whether they changed is unknown."
        if unconfirmed else "")
    out = {"schema_version": "1.0.0", "title": "Changes between two phone recordings of the same room",
           "recordings": {"before": rec_a, "after": rec_b},
           "common_frame": f"{rec_a} phone-tracking frame after the joint pose graph; metres; Z up; floor at z = 0",
           "summary": summary, "counts": {k: v for k, v in counters.items() if v},
           "changes": records,
           "unconfirmed": [{"label": c["label"], "at_m": c["at"], "recording": rec_a if "node_before" in c else rec_b,
                            "visibility_in_other": c["visibility_in_other"]} for c in unconfirmed],
           "scene_graphs": {rec: {"nodes": len(g["nodes"]), "edges": len(g["edges"])} for rec, g in graphs.items()},
           "models": MODELS, "profile": config.summary(),
           "learned_models_trained_or_finetuned": False, "manual_labels_used": False}
    out["result_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()
    (work / "changes.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    lines = ["# What changed between the two recordings", "", summary, ""]
    lines += [f"- **{r['id']}** ({r['type']}): {r['description']}" for r in records]
    (work / "CHANGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
