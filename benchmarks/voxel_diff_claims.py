"""Approach 1 results -> the changes.json fields evaluate.py reads, straight from its Step 2 + Step 3
outputs (object_changes.json, semantics/object_labels.json). Used when the description/package
stages did not run; the detected changes and their labels are the same either way.

  python benchmarks/voxel_diff_claims.py <outputs>/voxel_diff
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    step2 = json.loads((out / "object_changes.json").read_text(encoding="utf-8"))
    objects = {}
    for value in step2["objects"].values():
        for o in value if isinstance(value, list) else [value]:
            objects[o["id"]] = o
    labels = {}
    labels_file = out / "semantics" / "object_labels.json"
    if labels_file.is_file():
        for o in json.loads(labels_file.read_text(encoding="utf-8"))["objects"]:
            labels[o["object"]] = o["label"]

    def geo(oid: str) -> dict:
        o = objects[oid]
        return {"object": oid, "centroid_m": o["centroid_m"], "bbox_min_m": o["bbox_min_m"],
                "bbox_max_m": o["bbox_max_m"], "present_in": o["present_in"]}

    changes = []
    for e in step2["events"]:
        if e["event"] in ("moved", "shifted"):
            changes.append({"id": e["id"], "type": e["event"], "label": labels.get(e["from"]) or labels.get(e["to"]),
                            "before": geo(e["from"]), "after": geo(e["to"])})
        elif e["event"] in ("removed", "added"):
            changes.append({"id": e["object"], "type": e["event"], "label": labels.get(e["object"]),
                            "object": geo(e["object"])})
    target = out / "changes_from_step2.json"
    target.write_text(json.dumps({"source": "object_changes.json + semantics/object_labels.json",
                                  "changes": changes}, indent=2), encoding="utf-8")
    print(f"{len(changes)} changes -> {target}")
    for c in changes:
        print(f"  {c['id']:<5} {c['type']:<8} {c['label']}")


if __name__ == "__main__":
    main()
