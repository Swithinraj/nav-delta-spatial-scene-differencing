"""Visual verification of Step 2: every object change event in real photos.

For a move, four views are shown (voxels projected with the aligned poses,
frames chosen by depth evidence exactly as in qa_regions.py):

  scan 1 at the old position  (object surface seen)
  scan 2 at the old position  (camera sees through the now-empty spot)
  scan 1 at the new position  (camera sees through the then-empty spot)
  scan 2 at the new position  (object surface seen)

Unmatched removed/added objects get the two views of their single position.
No learned model is used.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navdelta.approaches.voxel_diff.paths import AFTER, BEFORE, OUTPUT, inside_workspace
from navdelta.approaches.voxel_diff.qa_regions import COLORS, best_frame, render_view
from navdelta.common.frames import load_session

ROWS_PER_PAGE = 5
MOVE_COLOR = (31, 58, 147)


def main() -> None:
    result = json.loads(inside_workspace(OUTPUT / "object_changes.json").read_text())
    voxels = np.load(str(inside_workspace(OUTPUT / "objects" / "object_voxels.npz")))
    grid = np.load(str(inside_workspace(OUTPUT / "voxel_change_grid.npz")))
    origin, dims, voxel = grid["origin"], grid["dims"], float(grid["voxel_m"])

    def centres(object_id: str) -> np.ndarray:
        ijk = np.column_stack(np.unravel_index(voxels[object_id], tuple(int(d) for d in dims)))
        return origin + (ijk + 0.5) * voxel

    sessions = {name: load_session(name) for name in (BEFORE, AFTER)}

    def pair(points: np.ndarray, present_session: str, color, label: str):
        absent_session = AFTER if present_session == BEFORE else BEFORE
        present, p_score = best_frame(sessions[present_session], points, "present")
        absent, a_score = best_frame(sessions[absent_session], points, "absent")
        present_img, box = render_view(present, points, color, None,
                                       f"{label} {present_session}: surface ({p_score})")
        absent_img, _ = render_view(absent, points, color, box,
                                    f"{label} {absent_session}: see-through ({a_score})")
        return (present_img, absent_img) if present_session == BEFORE else (absent_img, present_img)

    rows = []
    for event in result["events"]:
        if event["event"] in ("moved", "shifted"):
            old_scan1, old_scan2 = pair(centres(event["from"]), BEFORE, COLORS["removed"], "old")
            new_scan1, new_scan2 = pair(centres(event["to"]), AFTER, COLORS["added"], "new")
            rotation = event["rotation"]["rotation_deg"]
            distance = (f"~{event['shift_estimate_m']:.2f} m slide" if event["event"] == "shifted"
                        else f"{event['horizontal_distance_m']:.2f} m")
            caption = (f"{event['id']} {event['event']}\n{event['from']} -> {event['to']}\n"
                       f"{distance}\n"
                       f"rot {'n/a' if rotation is None else f'{rotation:.0f} deg'}\n"
                       f"score {event['cues']['score']:.2f}")
            rows.append((caption, [("scan 1 @ old", old_scan1), ("scan 2 @ old", old_scan2),
                                   ("scan 1 @ new", new_scan1), ("scan 2 @ new", new_scan2)], MOVE_COLOR))
        elif event["event"] in ("removed", "added", "surface_changed"):
            object_id = event["object"]
            side = next(o["side"] for group in result["objects"].values() for o in group if o["id"] == object_id)
            present_session = BEFORE if side == "removed" else AFTER
            color = COLORS["added_surface"] if event["event"] == "surface_changed" else COLORS[side]
            scan1, scan2 = pair(centres(object_id), present_session, color, "")
            caption = f"{object_id} {event['event']}\n{'+'.join(event['regions'])}"
            rows.append((caption, [("scan 1", scan1), ("scan 2", scan2)], color))

    pages = [rows[i:i + ROWS_PER_PAGE] for i in range(0, len(rows), ROWS_PER_PAGE)]
    for page_number, page in enumerate(pages, start=1):
        figure, axes = plt.subplots(len(page), 4, figsize=(18, 3.6 * len(page)), dpi=100,
                                    facecolor="white", squeeze=False)
        for row_axes, (caption, panels, color) in zip(axes, page):
            for axis in row_axes:
                axis.axis("off")
            for axis, (title, image) in zip(row_axes, panels):
                axis.imshow(image)
                axis.set_title(title, fontsize=9)
            row_axes[0].text(-0.03, 0.5, caption, transform=row_axes[0].transAxes, ha="right",
                             va="center", fontsize=9, fontweight="bold", color=np.array(color) / 255.0)
        figure.suptitle(f"Step 2 object changes in real photos (page {page_number}/{len(pages)})",
                        fontsize=12, fontweight="bold")
        figure.tight_layout(rect=(0.07, 0, 1, 0.98))
        path = inside_workspace(OUTPUT / "qa" / f"object_views_p{page_number}.png")
        figure.savefig(str(path))
        plt.close(figure)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
