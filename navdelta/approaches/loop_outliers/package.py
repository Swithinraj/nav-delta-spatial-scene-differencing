"""loop_outliers, stage 3: the 3D viewer and before/after photos (same viewer as voxel_diff).

Each change shows its segmented object at its old (red) and new (green) place, plus the outlier
matches that found it: the evidence, drawn as small points.
"""
from __future__ import annotations

import json
import time

import numpy as np

from navdelta import config, context
from navdelta.common import viewer_data
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"
OUTPUT = context.approach_dir("loop_outliers")
ROOM_VOXEL_M = 0.02
OLD, NEW = (61, 38, 215), (75, 158, 27)         # BGR outlines: red before, green after (as voxel_diff)

PROFILE_SETTINGS = config.apply(globals(), "loop_outliers.package")


def facts(c: dict) -> list[list[str]]:
    out = []
    if c["type"] == "moved":
        out.append(["Moved", f"{c['motion']['distance_m']:.2f} m"])
        if c["motion"].get("turn_deg") is not None:
            out.append(["Turned", f"{c['motion']['turn_deg']:.0f}° (from its own matches)"])
    ev = c.get("evidence", {})
    out.append(["Evidence", f"{ev.get('outlier_matches')} outlier matches, {ev.get('view_pairs')} view pairs"])
    if ev.get("paired_by_appearance"):
        out.append(["Paired by look", f"similarity {ev['paired_by_appearance']:.2f}"])
    return out


def main() -> None:
    started = time.time()
    scene = dict(np.load(inside(RECON / "scene.npz")))
    changes = json.loads((OUTPUT / "changes.json").read_text(encoding="utf-8"))
    shapes = dict(np.load(OUTPUT / "objects.npz"))
    rec_a, rec_b = changes["recordings"]["before"], changes["recordings"]["after"]
    records = []
    for c in changes["changes"]:
        parts, photos = [], {}
        for side, rec, role, rgb, colour in (("before", rec_a, "old", [0.84, 0.15, 0.24], OLD),
                                            ("after", rec_b, "new", [0.11, 0.62, 0.29], NEW)):
            pts = shapes.get(f"{c['id']}_{side}")
            if pts is None or not len(pts):
                continue
            parts.append({"role": role, "points": pts, "colors": np.tile(rgb, (len(pts), 1))})
            photos[side] = viewer_data.photo(scene, rec, pts, OUTPUT / "viewer" / "gallery" / f"{c['id']}_{side}.jpg",
                                             colour)
        record = {k: c.get(k) for k in ("id", "type", "title", "label", "description")}
        record.update(display_name=c["label"], label_confident=None, colour=None,
                      arrow={"from": c["before"]["centroid_m"], "to": c["after"]["centroid_m"]} if c["type"] == "moved" else None,
                      photos={k: f"gallery/{v}" for k, v in photos.items() if v} or None, facts=facts(c), parts=parts)
        records.append(record)
    viewer_data.write(OUTPUT, scene, changes["summary"], records, [], ROOM_VOXEL_M,
                      {"models": changes.get("models", []), "approach": "loop_outliers",
                       "loop_closure": changes.get("loop_closure")})
    print(f"viewer: {len(records)} changes ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
