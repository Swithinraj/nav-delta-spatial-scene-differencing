"""Step 1: visibility-aware voxel change detection between the two aligned recordings.

Each recording's RGB-D keyframes are ray-cast into one shared voxel grid in the
common frame (poses from <outputs>/alignment/aligned_frame_poses). For
every voxel and every recording this accumulates OctoMap-style log-odds from
two kinds of per-frame evidence:

  hit   a depth ray ended in the voxel (a surface was seen there)
  free  a depth ray passed through the voxel and hit something farther away

That yields a per-recording state of occupied / free / unknown. A change is
only declared where both recordings actually observed the voxel:

  added    occupied in the second recording, free in the first
  removed  occupied in the first recording, free in the second

A voxel one recording simply never looked at stays "unknown"; it is never
reported as removed. Alignment and depth noise are absorbed by a carving margin
(free space stops short of each measured surface) and a 10 cm surface
tolerance (no change is declared within two voxels of the other recording's
occupied surface), set from the alignment stage's measured cross-view residuals. Large structural planes (floor, walls, ceiling) are masked
out, since they are not objects and attract most residual misalignment.

No learned model is used anywhere in this step.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy
from scipy import ndimage


from navdelta.approaches.voxel_diff.paths import ALIGNMENT, BEFORE, DATA, OUTPUT, SESSIONS, inside_workspace
PIPELINE_VERSION = "1.0.0"

VOXEL_M = 0.05
DEPTH_STRIDE = 2                 # every 2nd pixel of the 512x384 depth map
DEPTH_MIN_M, DEPTH_MAX_M = 0.25, 5.0   # same validated range as the alignment stage
CARVE_START_M = 0.20
CARVE_MARGIN_M = 0.10            # free space stops this far short of each surface ...
CARVE_MARGIN_PER_M = 0.01        # ... plus 1 cm per metre of range
SAMPLE_STEP_M = VOXEL_M * 0.5
RAY_CHUNK = 4096

LOGODDS_HIT, LOGODDS_MISS = 0.85, -0.40     # OctoMap defaults
OCCUPIED_THRESHOLD, FREE_THRESHOLD = 1.5, -1.2
SUPPORT_WEIGHT = {"high": 1.0, "medium": 0.6}
DEFAULT_SUPPORT_WEIGHT = 0.25    # VIO-only / low RGB-D support frames

# 2 voxels = 10 cm: the alignment report's held-out cross-view check has 99.43% of
# correspondences within 10 cm (p90 69 mm), so a 5 cm tolerance sits below the
# known registration noise and leaks misaligned surfaces as false changes.
SURFACE_TOLERANCE_VOXELS = 2
# Structural planes are found by a deterministic exhaustive search, not RANSAC.
# The alignment locks gravity exactly, so every wall is vertical with a
# horizontal normal: each horizontal direction (1 degree steps) plus the vertical
# axis is scanned, the densest +/-5 cm slab is taken, its points removed, and the
# search repeats. RANSAC was tried first and dropped: its random sampling made
# the result depend on the seed on this room's multi-layer window wall (wall,
# window frame and curtain within ~25 cm of each other).
PLANE_ANGLE_STEP_DEG = 1.0
PLANE_OFFSET_BIN_M = 0.01
MAX_PLANES = 24
PLANE_DISTANCE_M = 0.05
MIN_PLANE_VOXELS = 2000          # ~5 m^2 at 5 cm: walls/floor, never furniture faces
MIN_PLANE_COMPONENT_VOXELS = 500
# A wall spans most of the floor-to-ceiling height *at every point along it*.
# Vertical planes are therefore tested column by column (5 cm wide): only columns
# containing a continuous vertical run of at least WALL_MIN_HEIGHT_M count as
# wall. Without this, a row of furniture faces lined up along a wall (bed side,
# cabinet, a suitcase standing parallel to the wall) forms one big coplanar
# plane and gets masked as "structure" -- which hid a real moved object in early
# testing. A whole-component height test was not enough: a furniture row that
# touched any tall element still passed as wall.
WALL_MIN_HEIGHT_M = 1.5
WALL_RUN_GAP_VOXELS = 2
HORIZONTAL_PLANE_Z_TOLERANCE_M = 0.20
# Room-surface state changes (curtains/blinds drawn, etc.) appear in the change map
# as thin vertical sheets taller than any movable object here; they are split out
# into their own "surface" category so they never merge with an object region
# next to them (the drawn window curtains otherwise fused with the moved chair).
MIN_SHEET_VOXELS = 150
# A change "sheet" column must hold a continuous vertical run of at least 1.0 m.
# Measured on this scene: the tallest column of any moved object is 0.90 m (the
# office chair; trolley bag 0.65 m, tower fan 0.75 m), while the drawn window
# curtains run 1.11 m at their 10th percentile and 1.45 m at the median. Window
# coverings and doors cover >1 m continuously; chair/fan/bag-sized objects do not.
SHEET_MIN_HEIGHT_M = 1.0
# Once a plane qualifies as a sheet (enough tall columns), its shorter, partly
# observed columns belong to the same physical surface: every change voxel in the
# slab within the tall columns' horizontal extent (+ this margin) joins the sheet.
SHEET_EXTENT_MARGIN_M = 0.20
MIN_REGION_VOXELS = 10

CUBE = np.ones((3, 3, 3), dtype=bool)


class VoxelGrid:
    def __init__(self, origin: np.ndarray, dims: np.ndarray, voxel: float):
        self.origin = origin.astype(np.float64)
        self.dims = dims.astype(np.int64)
        self.voxel = voxel
        self.size = int(np.prod(self.dims))

    def flat_index(self, points: np.ndarray) -> np.ndarray:
        ijk = np.floor((points - self.origin) / self.voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        ijk = ijk[inside]
        return (ijk[:, 0] * self.dims[1] + ijk[:, 1]) * self.dims[2] + ijk[:, 2]

    def centers(self, flat: np.ndarray) -> np.ndarray:
        k = flat % self.dims[2]
        j = (flat // self.dims[2]) % self.dims[1]
        i = flat // (self.dims[1] * self.dims[2])
        return self.origin + (np.column_stack([i, j, k]) + 0.5) * self.voxel

    def as_volume(self, flat_values: np.ndarray) -> np.ndarray:
        return flat_values.reshape(tuple(self.dims))


def load_frames(session: str) -> list[dict]:
    pose_dir = inside_workspace(ALIGNMENT / "aligned_frame_poses" / session)
    records = [json.loads(path.read_text()) for path in sorted(pose_dir.glob("*.json"))]
    if not records:
        raise RuntimeError(f"No aligned poses found for {session}")
    return records


def build_grid() -> tuple[VoxelGrid, o3d.geometry.PointCloud]:
    shared = o3d.io.read_point_cloud(str(inside_workspace(ALIGNMENT / "shared_aligned_room_pointcloud.ply")))
    points = np.asarray(shared.points)
    lower = np.quantile(points, 0.002, axis=0) - 0.25
    upper = np.quantile(points, 0.998, axis=0) + 0.25
    dims = np.ceil((upper - lower) / VOXEL_M).astype(np.int64)
    return VoxelGrid(lower, dims, VOXEL_M), shared


def backproject(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return camera centre, world endpoints, world unit directions, and ray lengths."""
    depth_raw = cv2.imread(str(inside_workspace(DATA / record["depth"])), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"Cannot read depth {record['depth']}")
    depth = depth_raw[::DEPTH_STRIDE, ::DEPTH_STRIDE].astype(np.float32) * 0.001
    rows, cols = np.nonzero((depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M))
    z = depth[rows, cols].astype(np.float64)

    intr = record["intrinsics"]
    scale_x = intr["width"] / depth_raw.shape[1]
    scale_y = intr["height"] / depth_raw.shape[0]
    # Pixel centres of the strided depth samples, mapped into the 1024x768 colour image.
    u_rgb = (cols * DEPTH_STRIDE + 0.5) * scale_x - 0.5
    v_rgb = (rows * DEPTH_STRIDE + 0.5) * scale_y - 0.5
    directions = np.column_stack([(u_rgb - intr["cx"]) / intr["fx"],
                                  (v_rgb - intr["cy"]) / intr["fy"],
                                  np.ones_like(z)])
    camera_points = directions * z[:, None]

    pose = np.asarray(record["T_camera_to_common"], dtype=np.float64)
    rotation, centre = pose[:3, :3], pose[:3, 3]
    endpoints = camera_points @ rotation.T + centre
    ray_length = np.linalg.norm(camera_points, axis=1)
    unit = (directions / np.linalg.norm(directions, axis=1, keepdims=True)) @ rotation.T
    return centre, endpoints, unit, ray_length


def integrate_session(session: str, grid: VoxelGrid) -> dict:
    records = load_frames(session)
    weighted_hit = np.zeros(grid.size, dtype=np.float32)
    weighted_free = np.zeros(grid.size, dtype=np.float32)
    count_hit = np.zeros(grid.size, dtype=np.uint16)
    count_free = np.zeros(grid.size, dtype=np.uint16)
    frame_hit = np.zeros(grid.size, dtype=bool)
    frame_free = np.zeros(grid.size, dtype=bool)
    support_counts: dict[str, int] = {}
    total_rays = 0
    started = time.time()

    for index, record in enumerate(records):
        label = record["pose_support"]["label"]
        support_counts[label] = support_counts.get(label, 0) + 1
        weight = SUPPORT_WEIGHT.get(label, DEFAULT_SUPPORT_WEIGHT)

        centre, endpoints, unit, ray_length = backproject(record)
        total_rays += len(ray_length)
        frame_hit[grid.flat_index(endpoints)] = True

        carve_length = ray_length - (CARVE_MARGIN_M + CARVE_MARGIN_PER_M * ray_length)
        carvable = carve_length > CARVE_START_M
        unit_c, length_c = unit[carvable], carve_length[carvable]
        for start in range(0, len(length_c), RAY_CHUNK):
            u = unit_c[start:start + RAY_CHUNK].astype(np.float32)
            length = length_c[start:start + RAY_CHUNK].astype(np.float32)
            steps = int(np.ceil((length.max() - CARVE_START_M) / SAMPLE_STEP_M)) + 1
            distances = CARVE_START_M + SAMPLE_STEP_M * np.arange(steps, dtype=np.float32)
            keep = distances[None, :] <= length[:, None]
            samples = centre[None, None, :] + u[:, None, :] * distances[None, :, None]
            frame_free[grid.flat_index(samples[keep])] = True

        # Within a single frame, a seen surface beats a grazing ray through the same voxel.
        frame_free &= ~frame_hit
        weighted_hit += weight * frame_hit
        weighted_free += weight * frame_free
        count_hit += frame_hit
        count_free += frame_free
        frame_hit[:] = False
        frame_free[:] = False

        if (index + 1) % 50 == 0 or index + 1 == len(records):
            print(f"[{session}] integrated {index + 1}/{len(records)} frames "
                  f"({time.time() - started:.0f}s)", flush=True)

    logodds = LOGODDS_HIT * weighted_hit + LOGODDS_MISS * weighted_free
    state = np.zeros(grid.size, dtype=np.int8)
    state[logodds >= OCCUPIED_THRESHOLD] = 1
    state[logodds <= FREE_THRESHOLD] = -1
    return {
        "session": session,
        "frames": len(records),
        "rays": int(total_rays),
        "support_labels": support_counts,
        "logodds": logodds,
        "count_hit": count_hit,
        "count_free": count_free,
        "state": state,
    }


_PLANE_ANGLES = np.radians(np.arange(0.0, 180.0, PLANE_ANGLE_STEP_DEG))
PLANE_DIRECTIONS = np.vstack([
    np.column_stack([np.cos(_PLANE_ANGLES), np.sin(_PLANE_ANGLES), np.zeros_like(_PLANE_ANGLES)]),
    [[0.0, 0.0, 1.0]],
])


def densest_plane(points: np.ndarray, directions: np.ndarray = PLANE_DIRECTIONS) -> tuple[np.ndarray, float, np.ndarray]:
    """Exhaustive, deterministic search for the densest +/-PLANE_DISTANCE_M slab
    among gravity-aligned planes (any horizontal normal, or the vertical axis).
    Returns unit normal n, offset c with n.p = c, and inlier indices."""
    half_window = int(round(PLANE_DISTANCE_M / PLANE_OFFSET_BIN_M))
    window = np.ones(2 * half_window + 1, dtype=np.int64)
    best_count, best = -1, None
    for normal in directions:
        offsets = points @ normal
        low = offsets.min()
        histogram = np.bincount(np.floor((offsets - low) / PLANE_OFFSET_BIN_M).astype(np.int64))
        slab_counts = np.convolve(histogram, window, mode="same")
        index = int(slab_counts.argmax())
        if slab_counts[index] > best_count:
            best_count = int(slab_counts[index])
            best = (normal, low + (index + 0.5) * PLANE_OFFSET_BIN_M)
    normal, offset = best[0], float(best[1])
    inliers = np.flatnonzero(np.abs(points @ normal - offset) < PLANE_DISTANCE_M)
    return normal, offset, inliers


def wall_columns(points: np.ndarray, normal: np.ndarray,
                 min_height_m: float = WALL_MIN_HEIGHT_M) -> np.ndarray:
    """Per inlier point: does its 5 cm column along the plane hold a tall continuous run?"""
    along = np.cross(normal, [0.0, 0.0, 1.0])
    along /= np.linalg.norm(along)
    column = np.floor(points @ along / VOXEL_M).astype(np.int64)
    level = np.floor(points[:, 2] / VOXEL_M).astype(np.int64)
    keep = np.zeros(len(points), dtype=bool)
    for value in np.unique(column):
        # Neighbouring columns are pooled so sparsely observed wall strips still connect.
        levels = np.unique(level[np.abs(column - value) <= 1])
        breaks = np.flatnonzero(np.diff(levels) > WALL_RUN_GAP_VOXELS + 1)
        run_starts = np.concatenate([[0], breaks + 1])
        run_ends = np.concatenate([breaks, [len(levels) - 1]])
        longest = (levels[run_ends] - levels[run_starts] + 1).max() * VOXEL_M
        if longest >= min_height_m:
            keep[column == value] = True
    return keep


def structural_planes(grid: VoxelGrid, occupied_any: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Mask large floor/wall/ceiling planes, restricted to their actual extent."""
    flat = np.flatnonzero(occupied_any)
    points = grid.centers(flat)
    z_low, z_high = np.quantile(points[:, 2], [0.01, 0.99])
    remaining = np.arange(len(points))
    mask = np.zeros(grid.size, dtype=bool)
    planes = []
    for _ in range(MAX_PLANES):
        if len(remaining) < MIN_PLANE_VOXELS:
            break
        normal, offset, inliers = densest_plane(points[remaining])
        # The search is exhaustive, so the densest remaining slab being too small
        # means no wall-sized plane is left.
        if len(inliers) < MIN_PLANE_VOXELS:
            break
        inlier_points = points[remaining[inliers]]
        mean_z = float(inlier_points[:, 2].mean())
        if normal[2] > 0.5:
            kind = "horizontal"
            structural = (abs(mean_z - z_low) < HORIZONTAL_PLANE_Z_TOLERANCE_M
                          or abs(mean_z - z_high) < HORIZONTAL_PLANE_Z_TOLERANCE_M)
        else:
            kind = "vertical"
            structural = True

        kept = 0
        if structural and kind == "vertical":
            wall = wall_columns(inlier_points, normal)
            mask[flat[remaining[inliers[wall]]]] = True
            kept = int(wall.sum())
        elif structural:
            # Floor/ceiling: keep only large connected pieces of the inlier set, so
            # the plane's infinite extension never masks unrelated coplanar faces.
            # Connected components on the voxel grid (deterministic, unlike Open3D's
            # multithreaded DBSCAN); a 1-voxel dilation bridges sparse coverage gaps.
            inlier_flat = flat[remaining[inliers]]
            inlier_volume = np.zeros(grid.size, dtype=bool)
            inlier_volume[inlier_flat] = True
            bridged = ndimage.binary_dilation(grid.as_volume(inlier_volume), structure=CUBE)
            labelled, _ = ndimage.label(bridged, structure=CUBE)
            component = labelled.ravel()[inlier_flat]
            sizes = np.bincount(component)
            large = sizes[component] >= MIN_PLANE_COMPONENT_VOXELS
            mask[inlier_flat[large]] = True
            kept = int(large.sum())
        planes.append({
            "kind": kind,
            "normal": normal.round(4).tolist(),
            "offset_m": round(offset, 4),
            "mean_z_m": round(mean_z, 3),
            "inlier_voxels": int(len(inliers)),
            "masked_voxels": kept,
            "treated_as_structural": bool(structural and kept > 0),
        })
        remaining = np.delete(remaining, inliers)

    mask = ndimage.binary_dilation(grid.as_volume(mask), structure=CUBE).ravel()
    return mask, planes


def split_surface_sheets(grid: VoxelGrid, change: np.ndarray) -> np.ndarray:
    """Return the subset of change voxels lying on tall thin vertical sheets."""
    sheet = np.zeros(grid.size, dtype=bool)
    flat = np.flatnonzero(change)
    if len(flat) < MIN_SHEET_VOXELS:
        return sheet
    points = grid.centers(flat)
    remaining = np.arange(len(points))
    vertical_directions = PLANE_DIRECTIONS[:-1]
    for _ in range(MAX_PLANES):
        if len(remaining) < MIN_SHEET_VOXELS:
            break
        normal, _, inliers = densest_plane(points[remaining], vertical_directions)
        if len(inliers) < MIN_SHEET_VOXELS:
            break
        slab = points[remaining[inliers]]
        tall = wall_columns(slab, normal, SHEET_MIN_HEIGHT_M)
        if tall.sum() >= MIN_SHEET_VOXELS:
            along = np.cross(normal, [0.0, 0.0, 1.0])
            position = slab @ (along / np.linalg.norm(along))
            within = ((position >= position[tall].min() - SHEET_EXTENT_MARGIN_M)
                      & (position <= position[tall].max() + SHEET_EXTENT_MARGIN_M))
            sheet[flat[remaining[inliers[within]]]] = True
        remaining = np.delete(remaining, inliers)
    return sheet


def label_regions(grid: VoxelGrid, change: np.ndarray, kind: str) -> list[dict]:
    labelled, count = ndimage.label(grid.as_volume(change), structure=CUBE)
    flat_labels = labelled.ravel()
    regions = []
    for region_id in range(1, count + 1):
        flat = np.flatnonzero(flat_labels == region_id)
        if len(flat) < MIN_REGION_VOXELS:
            continue
        centres = grid.centers(flat)
        regions.append({
            "type": kind,
            "voxels": int(len(flat)),
            "surface_area_m2_approx": round(len(flat) * VOXEL_M * VOXEL_M, 3),
            "centroid_m": centres.mean(axis=0).round(3).tolist(),
            "bbox_min_m": (centres.min(axis=0) - VOXEL_M / 2).round(3).tolist(),
            "bbox_max_m": (centres.max(axis=0) + VOXEL_M / 2).round(3).tolist(),
            "flat_indices": flat,
        })
    regions.sort(key=lambda region: -region["voxels"])
    prefix = {"added": "A", "removed": "R", "added_surface": "SA", "removed_surface": "SR"}[kind]
    for number, region in enumerate(regions, start=1):
        region["id"] = f"{prefix}{number}"
        region["height_m"] = round(region["bbox_max_m"][2] - region["bbox_min_m"][2], 3)
    return regions


def save_voxel_cloud(path: Path, centres: np.ndarray, color: tuple[float, float, float]) -> None:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres))
    cloud.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(centres), 1)))
    o3d.io.write_point_cloud(str(inside_workspace(path)), cloud)


def render_overview(grid: VoxelGrid, shared: o3d.geometry.PointCloud, added: list[dict],
                    removed: list[dict], surface: list[dict], path: Path) -> None:
    background = np.asarray(shared.points)
    rng = np.random.default_rng(0)
    if len(background) > 160_000:
        background = background[rng.choice(len(background), 160_000, replace=False)]
    views = [("Front (X-Z)", 0, 2, "X [m]", "Z [m]"),
             ("Top (X-Y)", 0, 1, "X [m]", "Y [m]"),
             ("Side (Y-Z)", 1, 2, "Y [m]", "Z [m]")]
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.6), dpi=170, facecolor="white")
    for axis, (title, xi, yi, xlabel, ylabel) in zip(axes, views):
        axis.scatter(background[:, xi], background[:, yi], s=0.15, c="#b9bec7", linewidths=0,
                     rasterized=True)
        for regions, color in ((surface, "#e6a100"), (removed, "#d7263d"), (added, "#1b9e4b")):
            for region in regions:
                centres = grid.centers(region["flat_indices"])
                axis.scatter(centres[:, xi], centres[:, yi], s=2.2, c=color, linewidths=0,
                             rasterized=True)
                if xi == 0 and yi == 1:
                    cx, cy = region["centroid_m"][0], region["centroid_m"][1]
                    axis.annotate(region["id"], (cx, cy), fontsize=7.5, fontweight="bold",
                                  color=color, xytext=(4, 4), textcoords="offset points")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_aspect("equal", adjustable="box")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color="#1b9e4b", label="added object (in scan 2 only)"),
               plt.Line2D([0], [0], marker="s", linestyle="", color="#d7263d", label="removed object (in scan 1 only)"),
               plt.Line2D([0], [0], marker="s", linestyle="", color="#e6a100",
                          label="surface change (tall thin sheet, e.g. curtain/blinds)"),
               plt.Line2D([0], [0], marker="o", linestyle="", color="#b9bec7", label="aligned room (both scans)")]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    figure.suptitle("Step 1 - visibility-aware voxel change map (5 cm voxels)", fontsize=13,
                    fontweight="bold")
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(str(inside_workspace(path)))
    plt.close(figure)


def main() -> None:
    started = time.time()
    output = inside_workspace(OUTPUT)
    (output / "qa").mkdir(parents=True, exist_ok=True)

    grid, shared = build_grid()
    print(f"Voxel grid {grid.dims.tolist()} at {VOXEL_M} m ({grid.size:,} voxels), "
          f"origin {grid.origin.round(3).tolist()}", flush=True)

    integrated = [integrate_session(session, grid) for session in SESSIONS]
    state1, state2 = integrated[0]["state"], integrated[1]["state"]
    occupied1, occupied2 = state1 == 1, state2 == 1
    free1, free2 = state1 == -1, state2 == -1

    near_occupied1 = ndimage.binary_dilation(grid.as_volume(occupied1), structure=CUBE,
                                             iterations=SURFACE_TOLERANCE_VOXELS).ravel()
    near_occupied2 = ndimage.binary_dilation(grid.as_volume(occupied2), structure=CUBE,
                                             iterations=SURFACE_TOLERANCE_VOXELS).ravel()
    added_raw = occupied2 & free1 & ~near_occupied1
    removed_raw = occupied1 & free2 & ~near_occupied2

    structural_mask, planes = structural_planes(grid, occupied1 | occupied2)
    added = added_raw & ~structural_mask
    removed = removed_raw & ~structural_mask

    added_sheet = split_surface_sheets(grid, added)
    removed_sheet = split_surface_sheets(grid, removed)
    added_objects = added & ~added_sheet
    removed_objects = removed & ~removed_sheet

    added_regions = label_regions(grid, added_objects, "added")
    removed_regions = label_regions(grid, removed_objects, "removed")
    surface_regions = (label_regions(grid, added_sheet, "added_surface")
                       + label_regions(grid, removed_sheet, "removed_surface"))

    occupied_any = occupied1 | occupied2
    observed_by_other = ((occupied1 & (state2 != 0)) | (occupied2 & (state1 != 0)))
    coverage = {
        "occupied_voxels_scan1": int(occupied1.sum()),
        "occupied_voxels_scan2": int(occupied2.sum()),
        "free_voxels_scan1": int(free1.sum()),
        "free_voxels_scan2": int(free2.sum()),
        "occupied_in_either": int(occupied_any.sum()),
        "occupied_in_either_and_observed_by_other_scan": int(observed_by_other.sum()),
        "cross_observed_fraction": round(float(observed_by_other.sum() / max(occupied_any.sum(), 1)), 4),
        "occupied_in_one_scan_but_unknown_in_other": int((occupied_any & ~observed_by_other).sum()),
    }

    # Region membership per voxel (0 = none, n = n-th region in the report list),
    # so later stages reuse exactly these regions instead of re-labelling.
    def region_labels(regions: list[dict]) -> np.ndarray:
        labels = np.zeros(grid.size, dtype=np.int16)
        for number, region in enumerate(regions, start=1):
            labels[region["flat_indices"]] = number
        return labels

    np.savez_compressed(
        str(inside_workspace(output / "voxel_change_grid.npz")),
        origin=grid.origin, dims=grid.dims, voxel_m=np.float64(VOXEL_M),
        logodds_scan1=integrated[0]["logodds"], logodds_scan2=integrated[1]["logodds"],
        state_scan1=state1, state_scan2=state2,
        added_raw=added_raw, removed_raw=removed_raw, structural_mask=structural_mask,
        added_objects=added_objects, removed_objects=removed_objects,
        added_sheet=added_sheet, removed_sheet=removed_sheet,
        added_region_label=region_labels(added_regions),
        removed_region_label=region_labels(removed_regions),
        surface_region_label=region_labels(surface_regions),
    )
    added_color, removed_color, surface_color = (0.11, 0.62, 0.29), (0.84, 0.15, 0.24), (0.90, 0.63, 0.0)
    save_voxel_cloud(output / "added_voxels.ply", grid.centers(np.flatnonzero(added_objects)), added_color)
    save_voxel_cloud(output / "removed_voxels.ply", grid.centers(np.flatnonzero(removed_objects)), removed_color)
    save_voxel_cloud(output / "surface_change_voxels.ply",
                     grid.centers(np.flatnonzero(added_sheet | removed_sheet)), surface_color)

    overlay = o3d.geometry.PointCloud(shared)
    overlay.paint_uniform_color((0.72, 0.74, 0.78))
    changes = o3d.geometry.PointCloud()
    for flat, color in ((np.flatnonzero(added_objects), added_color),
                        (np.flatnonzero(removed_objects), removed_color),
                        (np.flatnonzero(added_sheet | removed_sheet), surface_color)):
        part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid.centers(flat)))
        part.paint_uniform_color(color)
        changes += part
    o3d.io.write_point_cloud(str(inside_workspace(output / "change_overlay.ply")), overlay + changes)

    render_overview(grid, shared, added_regions, removed_regions, surface_regions,
                    output / "qa" / "change_overview.png")

    def public(region: dict) -> dict:
        return {key: value for key, value in region.items() if key != "flat_indices"}

    mask_hash = hashlib.sha256(b"".join(np.packbits(mask).tobytes() for mask in
                                        (added_objects, removed_objects, added_sheet, removed_sheet))).hexdigest()

    report = {
        "schema_version": "1.0.0",
        "pipeline_version": PIPELINE_VERSION,
        "stage": "step1_visibility_aware_voxel_change_map",
        "common_frame": f"{BEFORE} PLY frame; metres; Z-up",
        "learned_models_used": [],
        "parameters": {
            "voxel_m": VOXEL_M, "depth_source": "corrected_depth/*.Clean.png",
            "depth_stride": DEPTH_STRIDE, "depth_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
            "carve_start_m": CARVE_START_M, "carve_margin_m": CARVE_MARGIN_M,
            "carve_margin_per_m": CARVE_MARGIN_PER_M, "sample_step_m": SAMPLE_STEP_M,
            "logodds_hit": LOGODDS_HIT, "logodds_miss": LOGODDS_MISS,
            "occupied_threshold": OCCUPIED_THRESHOLD, "free_threshold": FREE_THRESHOLD,
            "support_weight": {**SUPPORT_WEIGHT, "other": DEFAULT_SUPPORT_WEIGHT},
            "surface_tolerance_voxels": SURFACE_TOLERANCE_VOXELS,
            "plane_search": "deterministic exhaustive gravity-aligned slab search",
            "plane_angle_step_deg": PLANE_ANGLE_STEP_DEG, "plane_offset_bin_m": PLANE_OFFSET_BIN_M,
            "max_planes": MAX_PLANES,
            "plane_distance_m": PLANE_DISTANCE_M, "min_plane_voxels": MIN_PLANE_VOXELS,
            "min_plane_component_voxels": MIN_PLANE_COMPONENT_VOXELS,
            "wall_min_height_m": WALL_MIN_HEIGHT_M, "wall_run_gap_voxels": WALL_RUN_GAP_VOXELS,
            "min_sheet_voxels": MIN_SHEET_VOXELS, "sheet_min_height_m": SHEET_MIN_HEIGHT_M,
            "sheet_extent_margin_m": SHEET_EXTENT_MARGIN_M,
            "min_region_voxels": MIN_REGION_VOXELS,
        },
        "grid": {"origin_m": grid.origin.round(4).tolist(), "dims": grid.dims.tolist(),
                 "voxels": grid.size},
        "sessions": [{key: value for key, value in item.items()
                      if key in ("session", "frames", "rays", "support_labels")} for item in integrated],
        "coverage": coverage,
        "structural_planes": planes,
        "change_voxels": {
            "added_raw": int(added_raw.sum()), "removed_raw": int(removed_raw.sum()),
            "added_after_structural_mask": int(added.sum()),
            "removed_after_structural_mask": int(removed.sum()),
            "added_surface_sheet": int(added_sheet.sum()),
            "removed_surface_sheet": int(removed_sheet.sum()),
            "added_objects": int(added_objects.sum()),
            "removed_objects": int(removed_objects.sum()),
            "added_in_regions": int(sum(r["voxels"] for r in added_regions)),
            "removed_in_regions": int(sum(r["voxels"] for r in removed_regions)),
            "change_mask_sha256": mask_hash,
        },
        "regions": {"added": [public(r) for r in added_regions],
                    "removed": [public(r) for r in removed_regions],
                    "surface": [public(r) for r in surface_regions]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__},
        "runtime_s": round(time.time() - started, 1),
    }
    (output / "change_detection_report.json").write_text(json.dumps(report, indent=2))

    print("\nCoverage:", json.dumps(coverage, indent=2))
    print(f"Structural planes masked: {sum(p['treated_as_structural'] for p in planes)} of {len(planes)} found")
    print(f"Change voxels  raw: +{int(added_raw.sum())} / -{int(removed_raw.sum())}   "
          f"after plane mask: +{int(added.sum())} / -{int(removed.sum())}   "
          f"surface sheets: +{int(added_sheet.sum())} / -{int(removed_sheet.sum())}")
    print(f"Change mask sha256: {mask_hash}")
    for region in added_regions[:12] + removed_regions[:12] + surface_regions:
        lo, hi = region["bbox_min_m"], region["bbox_max_m"]
        print(f"  {region['id']:>4} {region['type']:<15} {region['voxels']:5d} vox  "
              f"centroid {region['centroid_m']}  z {lo[2]:.2f}..{hi[2]:.2f}")
    print(f"\nDone in {time.time() - started:.0f}s -> {output}")


if __name__ == "__main__":
    main()
