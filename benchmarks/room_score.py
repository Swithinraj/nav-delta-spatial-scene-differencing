"""Score the RGB approaches on our room recordings against the three staged changes.

  python benchmarks/room_score.py [outputs]  ->  benchmarks/results/room_results.json

The staged changes (luggage turned, office chair moved, tower fan moved) are given by where they
were and are in the shared reconstruction's frame (Myroomesh1 tracking frame after the joint pose
graph; the RGB and RGB-D reconstructions share it). A staged change is found when a reported change
has an end within 0.5 m of its old or new place; it is found exactly when the report says "moved"
(or "rotated") with both ends right. Everything else reported is an extra claim. voxel_diff works in
its own LiDAR frame; its room result is the hand-checked one recorded in runs.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from paths import RESULTS, ROOT
STAGED = {
    "luggage (turned ~80°, moved ~0.6 m)": ((-3.59, -0.90), (-4.04, -1.52)),
    "office chair (moved ~0.5 m)": ((-2.75, -1.87), (-3.27, -1.60)),
    "tower fan (moved ~0.7 m)": ((-2.24, -0.84), (-1.95, -1.52)),
}
RADIUS_M = 0.5


def ends(c: dict) -> dict:
    out = {k: np.array(c[k]["centroid_m"][:2]) for k in ("before", "after") if c.get(k)}
    if c.get("object"):
        out["before" if c["type"] == "removed" else "after"] = np.array(c["object"]["centroid_m"][:2])
    return out


def score(changes: dict) -> dict:
    found, exact, used = {}, {}, set()
    for name, (old, new) in STAGED.items():
        for i, c in enumerate(changes["changes"]):
            e = ends(c)
            near = [np.linalg.norm(p - np.array(old)) < RADIUS_M or np.linalg.norm(p - np.array(new)) < RADIUS_M
                    for p in e.values()]
            if any(near):
                found[name] = c["id"]
                used.add(i)
                if (c["type"] in ("moved", "rotated") and "before" in e and "after" in e
                        and np.linalg.norm(e["before"] - np.array(old)) < RADIUS_M
                        and np.linalg.norm(e["after"] - np.array(new)) < RADIUS_M):
                    exact[name] = c["id"]
    extras = [{"id": c["id"], "type": c["type"], "label": c["label"]}
              for i, c in enumerate(changes["changes"]) if i not in used]
    return {"staged": len(STAGED), "staged_found": len(found), "staged_exact": len(exact), "found": found,
            "other_claims": len(extras), "extras": extras}


def main() -> None:
    outputs = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs"
    runs = {}
    for line in (outputs / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("approach"):
            runs[r["approach"]] = r                      # latest run per approach
    shared = {}
    for line in (outputs / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for s in r["stages"]:
            if s["stage"].startswith(("recon.", "recon_rgbd.")):
                shared[s["stage"].split(".")[0]] = s["seconds"]
    results = {}
    for approach in ("scene_graph", "loop_outliers", "rgbd_fusion"):
        path = outputs / approach / "changes.json"
        if not path.is_file():
            continue
        changes = json.loads(path.read_text(encoding="utf-8"))
        run = runs.get(approach, {})
        stages = {s["stage"]: s["seconds"] for s in run.get("stages", [])}
        recon = "recon_rgbd" if approach == "rgbd_fusion" else "recon"
        results[approach] = {**score(changes), "summary": changes["summary"],
                             "stages_seconds": stages, "shared_seconds": shared.get(recon)}
        r = results[approach]
        print(f"{approach:<14} staged {r['staged_found']}/3 (exact {r['staged_exact']}), extras {r['other_claims']}: "
              f"{changes['summary']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "room_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
