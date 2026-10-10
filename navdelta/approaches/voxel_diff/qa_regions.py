"""Visual verification of Step 1: show every change region in real before/after photos.

For each change region, the voxel centres are projected into every keyframe of
both recordings using the aligned poses, and depth decides which frames are
real evidence:

  present view  the recording where the region is occupied (scan 2 for added,
                scan 1 for removed). Best frame = most region voxels whose
                projected depth agrees with the measured depth (surface seen).
  absent view   the other recording. Best frame = most region voxels the camera
                demonstrably sees *past* (measured depth well beyond the voxel),
                i.e. direct photographic evidence the space is empty.

Output is a before/after contact sheet per region: scan 1 on the left, scan 2 on
the right, region voxels overlaid. No learned model is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navdelta.approaches.voxel_diff.paths import AFTER, BEFORE, DATA, OUTPUT, SESSIONS, inside_workspace
from navdelta.common.frames import load_session, project

MIN_REGION_VOXELS_SHOWN = 15
ROWS_PER_PAGE = 6
COLORS = {"added": (27, 158, 75), "removed": (215, 38, 61),
          "added_surface": (230, 161, 0), "removed_surface": (230, 161, 0)}


def best_frame(frames: list[dict], points: np.ndarray, mode: str):
    best, best_score = None, 0
    for frame in frames:
        _, _, in_view, surface, see_past = project(frame, points)
        score = int((surface if mode == "present" else see_past).sum())
        if score > best_score:
            best, best_score = frame, score
    return best, best_score


def render_view(frame: dict | None, points: np.ndarray, color: tuple[int, int, int],
                anchor_box: tuple[float, float, float, float] | None, caption: str) -> tuple[np.ndarray, tuple]:
    if frame is None:
        canvas = np.full((360, 480, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "no frame observes this region", (40, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (90, 90, 90), 1, cv2.LINE_AA)
        return canvas, anchor_box
    image = cv2.cvtColor(cv2.imread(str(inside_workspace(DATA / frame["rgb"]))), cv2.COLOR_BGR2RGB)
    u, v, in_view, _, _ = project(frame, points)
    overlay = image.copy()
    for x, y in zip(u[in_view], v[in_view]):
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
    if in_view.any():
        x0, x1 = u[in_view].min(), u[in_view].max()
        y0, y1 = v[in_view].min(), v[in_view].max()
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color, 3, cv2.LINE_AA)
        box = (x0, y0, x1, y1)
    else:
        box = anchor_box or (0, 0, image.shape[1], image.shape[0])
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * 0.85 + 140
    left, right = int(max(0, cx - half * 1.33)), int(min(image.shape[1], cx + half * 1.33))
    top, bottom = int(max(0, cy - half)), int(min(image.shape[0], cy + half))
    crop = image[top:bottom, left:right]
    crop = cv2.resize(crop, (480, int(480 * crop.shape[0] / max(crop.shape[1], 1))))
    cv2.putText(crop, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(crop, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return crop, box


def main() -> None:
    grid = np.load(str(inside_workspace(OUTPUT / "voxel_change_grid.npz")))
    report = json.loads((inside_workspace(OUTPUT / "change_detection_report.json")).read_text())
    origin, dims, voxel = grid["origin"], grid["dims"], float(grid["voxel_m"])

    def centres(flat: np.ndarray) -> np.ndarray:
        k = flat % dims[2]
        j = (flat // dims[2]) % dims[1]
        i = flat // (dims[1] * dims[2])
        return origin + (np.column_stack([i, j, k]) + 0.5) * voxel

    sessions = {name: load_session(name) for name in SESSIONS}
    rows = []
    for group, label_key in (("added", "added_region_label"), ("removed", "removed_region_label"),
                             ("surface", "surface_region_label")):
        labels = grid[label_key]
        for number, region in enumerate(report["regions"][group], start=1):
            if region["voxels"] < MIN_REGION_VOXELS_SHOWN:
                continue
            kind = region["type"]
            points = centres(np.flatnonzero(labels == number))
            present_session = AFTER if kind.startswith("added") else BEFORE
            absent_session = BEFORE if kind.startswith("added") else AFTER
            present, present_score = best_frame(sessions[present_session], points, "present")
            absent, absent_score = best_frame(sessions[absent_session], points, "absent")
            rows.append((region, kind, points, present_session, present, present_score,
                         absent_session, absent, absent_score))
            print(f"{region['id']:>4} {kind:<7} {region['voxels']:4d} vox | "
                  f"{present_session} frame {present['frame'] if present else '-'} sees surface on {present_score} vox | "
                  f"{absent_session} frame {absent['frame'] if absent else '-'} sees past {absent_score} vox", flush=True)

    pages = [rows[i:i + ROWS_PER_PAGE] for i in range(0, len(rows), ROWS_PER_PAGE)]
    for page_number, page in enumerate(pages, start=1):
        figure, axes = plt.subplots(len(page), 2, figsize=(11, 3.9 * len(page)), dpi=110,
                                    facecolor="white", squeeze=False)
        for row_axes, (region, kind, points, ps, present, pscore, as_, absent, ascore) in zip(axes, page):
            color = COLORS[kind]
            present_img, box = render_view(present, points, color, None,
                                           f"{ps}: surface seen ({pscore} vox)")
            absent_img, _ = render_view(absent, points, color, box,
                                        f"{as_}: seen through ({ascore} vox)")
            scan1_img, scan2_img = ((absent_img, present_img) if kind.startswith("added")
                                    else (present_img, absent_img))
            height = region["bbox_max_m"][2] - region["bbox_min_m"][2]
            for axis, img, title in ((row_axes[0], scan1_img, "scan 1 (before)"),
                                     (row_axes[1], scan2_img, "scan 2 (after)")):
                axis.imshow(img)
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            row_axes[0].text(-0.02, 0.5, f"{region['id']}\n{kind}\n{region['voxels']} vox\n"
                                         f"h {height:.2f} m\n({region['centroid_m'][0]:.2f}, "
                                         f"{region['centroid_m'][1]:.2f})",
                             transform=row_axes[0].transAxes, ha="right", va="center", fontsize=9,
                             fontweight="bold", color=np.array(color) / 255.0)
        figure.suptitle(f"Step 1 change regions in real photos (page {page_number}/{len(pages)})",
                        fontsize=12, fontweight="bold")
        figure.tight_layout(rect=(0.08, 0, 1, 0.98))
        path = inside_workspace(OUTPUT / "qa" / f"region_views_p{page_number}.png")
        figure.savefig(str(path))
        plt.close(figure)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
