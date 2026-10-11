"""scene_graph, stage 4: the 3D viewer and before/after photos (same viewer as voxel_diff)."""
from __future__ import annotations

import json
import time

import numpy as np

from navdelta import config, context
from navdelta.approaches.scene_graph.diff import combine
from navdelta.common import viewer_data
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"
OUTPUT = context.approach_dir("scene_graph")
ROOM_VOXEL_M = 0.02
OLD, NEW = (61, 38, 215), (75, 158, 27)         # BGR outlines: red before, green after (as voxel_diff)

PROFILE_SETTINGS = config.apply(globals(), "scene_graph.package")


def facts(c: dict) -> list[list[str]]:
    out = []
    if c["type"] in ("moved", "rotated"):
        out.append(["Moved", f"{c['motion']['distance_m']:.2f} m"])
        if c["motion"].get("rotation_deg"):
            out.append(["Turned", f"{c['motion']['rotation_deg']:.0f}°"])
    for side in ("before", "after", "object"):
        if side in c:
            out.append([f"Seen in ({side})" if side != "object" else "Seen in", f"{c[side]['observations']} keyframes"])
    votes = c.get("label_votes") or {}
    if votes:
        out.append(["Names voted", ", ".join(f"{k} {v}" for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[:3])])
    return out


def main() -> None:
    started = time.time()
    scene = dict(np.load(inside(RECON / "scene.npz")))
    graphs = np.load(OUTPUT / "graphs.npy", allow_pickle=True).item()
    diff = json.loads((OUTPUT / "graph_diff.json").read_text())
    changes = json.loads((OUTPUT / "changes.json").read_text(encoding="utf-8"))
    rec_a, rec_b = list(graphs)
    A, B = graphs[rec_a]["nodes"], graphs[rec_b]["nodes"]
    by_id = {c["id"]: c for c in changes["changes"]}
    records = []
    for d, c in zip([x for x in diff["changes"] if x["type"] not in ("not re-observed",)], changes["changes"]):
        parts, photos = [], {}
        if "members_before" in d or "node_before" in d:
            node = combine(A, d.get("members_before") or [d["node_before"]])
            parts.append({"role": "old", "points": node["voxels"], "colors": np.tile([0.84, 0.15, 0.24], (len(node["voxels"]), 1))})
            photos["before"] = viewer_data.photo(scene, rec_a, node["voxels"], OUTPUT / "viewer" / "gallery" / f"{c['id']}_before.jpg", OLD)
        if "members_after" in d or "node_after" in d:
            node = combine(B, d.get("members_after") or [d["node_after"]])
            parts.append({"role": "new", "points": node["voxels"], "colors": np.tile([0.11, 0.62, 0.29], (len(node["voxels"]), 1))})
            photos["after"] = viewer_data.photo(scene, rec_b, node["voxels"], OUTPUT / "viewer" / "gallery" / f"{c['id']}_after.jpg", NEW)
        record = {k: c.get(k) for k in ("id", "type", "title", "label", "description")}
        record.update(display_name=c["label"], label_confident=None, colour=None,
                      arrow={"from": c["before"]["centroid_m"], "to": c["after"]["centroid_m"]} if "after" in c and "before" in c else None,
                      photos={k: f"gallery/{v}" for k, v in photos.items() if v} or None, facts=facts(c), parts=parts)
        records.append(record)
    landmarks = []
    for i, _ in diff["unchanged"]:
        n = A[i]
        if float(np.max(n["bbox_max"][:2] - n["bbox_min"][:2])) >= 0.5:
            landmarks.append({"name": n["label"], "centroid": n["centroid"].round(3).tolist(), "top": float(n["bbox_max"][2])})
    viewer_data.write(OUTPUT, scene, changes["summary"], records, landmarks, ROOM_VOXEL_M,
                      {"models": changes.get("models", []), "approach": "scene_graph"})
    print(f"viewer: {len(records)} changes, {len(landmarks)} landmarks ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
