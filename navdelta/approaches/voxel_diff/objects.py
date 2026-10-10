"""Step 2: group Step 1 change regions into objects and pair removed with added objects.

Step 1 produces voxel *regions*, which are often fragments of one physical
object (a chair's backrest and seat, a door's top and bottom) or of one changed
surface (a window covering split by a radiator). This stage turns regions into
objects and change events using geometry and colour only:

  1. Grouping (per side: removed = scan 1, added = scan 2)
     - same object:   regions lying on one connected, object-sized piece of the
                      scan's own occupied (non-structural) surface
     - same panel:    thin, coplanar regions that overlap along the plane
                      (e.g. a door seen above and below an occluder)
     - fragments:     a region smaller than a 15 cm cube joins the nearest
                      region within the 10 cm surface tolerance
     - surfaces:      Step 1's tall sheets absorb the coplanar pieces of the
                      same surface, including thin layers just in front of them
  2. Sliver trimming: 1-voxel-wide slices at the top or bottom of an object
     (misalignment residue along wall edges) are excluded from its descriptors.
  3. Move matching: every removed object is compared with every added object by
     a rotation-aware shape fit (exhaustive yaw and offset search), top height
     and colour; a global one-to-one assignment (Hungarian) keeps pairs above a
     fixed score. Unpaired objects are reported as removed or added.

No learned model is used. Every step is deterministic.
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


from navdelta.approaches.voxel_diff.paths import AFTER, ALIGNMENT, BEFORE, OUTPUT, SCAN_CLOUDS, inside_workspace, out_rel
PIPELINE_VERSION = "1.0.0"

# Grouping. A movable object here is at most chair-sized: a connected piece of a
# scan's surface larger than this is furniture/room clutter fused together, so it
# is not used as evidence that two regions are one object.
OBJECT_MAX_FOOTPRINT_M = 1.2
OBJECT_MAX_HEIGHT_M = 1.5          # = Step 1 wall run height
PANEL_MAX_THICKNESS_M = 0.15       # 3 voxels
PANEL_MIN_ELONGATION = 2.0
PANEL_COPLANAR_M = 0.10            # = Step 1 surface tolerance
PANEL_MAX_ANGLE_DEG = 10.0
# A 3x3x3 block (15 cm cube) is the smallest thing reported as an object of its
# own; this matches the ~10-15 cm detection floor of the 5 cm grid with a 10 cm
# surface tolerance.
FRAGMENT_MAX_VOXELS = 27
FRAGMENT_ATTACH_GAP_VOXELS = 2.0   # voxel-centre distance = 10 cm surface tolerance
# A piece belongs to a sheet if every one of its floor-plan columns lies within
# the 10 cm surface tolerance of the sheet's columns, inside the sheet's extent.
SHEET_ATTACH_TOLERANCE_M = 0.10
SHEET_EXTENT_MARGIN_M = 0.20       # = Step 1 sheet extent margin
# Slivers: a top/bottom slice holding <10% of the object's widest slice.
SLIVER_FRACTION = 0.10
# Footprint extents come from columns at least this fraction as tall as the tallest.
COLUMN_SUPPORT_FRACTION = 0.25

# Matching.
YAW_STEP_DEG = 3.0
SHIFT_RANGE_M = 0.10               # centroid of a partial view is biased by up to ~2 voxels
SHIFT_STEP_M = 0.05
FIT_TOLERANCE_M = 0.075            # within one voxel (face or in-plane diagonal neighbour)
HEIGHT_SCALE_M = 0.30              # top-height difference at which height similarity reaches 0
COLOUR_SCALE_DE = 40.0             # CIE76 difference at which colour similarity reaches 0
# Weighted geometric mean: every cue has to agree, a single cue at zero vetoes the
# pair (an additive score let shape + colour carry a pair whose heights differ).
WEIGHTS = {"shape": 0.5, "height": 0.25, "colour": 0.25}
MATCH_MIN_SCORE = 0.5
ROTATION_MIN_FIT = 0.6             # below this, >40% of voxels do not correspond: no rotation claim
YAW_AMBIGUITY = 0.03               # fit within this of the best at +180 deg -> axis only
YAW_UNDETERMINED = 0.10            # fit varies less than this over all yaws -> no rotation
# Move vs. shift: a real move crosses space that one of the scans saw empty. When
# a large object slides a few cm, its trailing face shows up as "removed" and
# its leading face as "added" -- two changed faces with the object's persistent
# surface between them. Measured here: 55% of the path between the drawer
# unit's two faces is occupied in both scans, 0-3% for the moved bag/fan/chair.
SHIFT_PERSISTENT_FRACTION = 0.25
PATH_SAMPLE_M = 0.025
WALL_PARALLEL_DEG = 20.0
MIN_ELONGATION_FOR_AXIS = 1.3

CUBE = np.ones((3, 3, 3), dtype=bool)
EPS = 1e-6                          # extents are multiples of the voxel size; compare them float-safely


class Grid:
    def __init__(self, data):
        self.origin = data["origin"].astype(np.float64)
        self.dims = data["dims"].astype(np.int64)
        self.voxel = float(data["voxel_m"])
        self.shape = tuple(int(d) for d in self.dims)
        self.size = int(np.prod(self.dims))

    def centers(self, flat: np.ndarray) -> np.ndarray:
        ijk = np.column_stack(np.unravel_index(flat, self.shape))
        return self.origin + (ijk + 0.5) * self.voxel

    def flat_index(self, points: np.ndarray) -> np.ndarray:
        ijk = np.floor((points - self.origin) / self.voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        flat = np.full(len(points), -1, dtype=np.int64)
        flat[inside] = (ijk[inside, 0] * self.dims[1] + ijk[inside, 1]) * self.dims[2] + ijk[inside, 2]
        return flat


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def footprint(points: np.ndarray, voxel: float) -> dict:
    """Principal axes of the occupied floor-plan columns.

    Each column is weighted by how many voxels it holds, and only columns at
    least COLUMN_SUPPORT_FRACTION as tall as the tallest one set the extents, so
    a one-voxel residue strip along a wall edge cannot tilt or stretch the
    footprint of the object standing next to it."""
    xy, counts = np.unique(points[:, :2].round(4), axis=0, return_counts=True)
    weights = counts.astype(np.float64)
    centre = (xy * weights[:, None]).sum(axis=0) / weights.sum()
    if len(xy) < 3:
        return {"length_m": voxel, "width_m": voxel, "axis_deg": 0.0, "elongation": 1.0,
                "centre": centre, "major": np.array([1.0, 0.0]), "normal": np.array([0.0, 1.0]),
                "offset_across": float(centre[1]), "along_range": (float(centre[0]), float(centre[0]))}
    offsets = xy - centre
    covariance = (offsets * weights[:, None]).T @ offsets / weights.sum()
    _, eigvecs = np.linalg.eigh(covariance)
    major = eigvecs[:, 1]
    angle = float(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    supported = counts >= COLUMN_SUPPORT_FRACTION * counts.max()
    along = offsets[supported] @ major
    across = offsets[supported] @ eigvecs[:, 0]
    length = float(along.max() - along.min() + voxel)
    width = float(across.max() - across.min() + voxel)
    return {"length_m": length, "width_m": width, "axis_deg": angle,
            "elongation": length / max(width, 1e-6), "centre": centre,
            "major": major, "offset_across": float(np.median(xy @ eigvecs[:, 0])),
            "normal": eigvecs[:, 0], "along_range": (float((xy @ major).min()), float((xy @ major).max()))}


def trim_slivers(points: np.ndarray, voxel: float) -> np.ndarray:
    """Mask of voxels kept after dropping thin slices at the top and bottom ends."""
    levels = np.round(points[:, 2] / voxel).astype(np.int64)
    low = levels.min()
    counts = np.bincount(levels - low)
    threshold = SLIVER_FRACTION * counts.max()
    top, bottom = len(counts) - 1, 0
    while top > bottom and counts[top] < threshold:
        top -= 1
    while bottom < top and counts[bottom] < threshold:
        bottom += 1
    index = levels - low
    return (index >= bottom) & (index <= top)


def lab_of(rgb01: np.ndarray) -> np.ndarray:
    pixel = np.clip(np.round(rgb01 * 255), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float64)
    return np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0])


def load_regions(grid: Grid, data, report: dict) -> list[dict]:
    regions = []
    for group, key in (("added", "added_region_label"), ("removed", "removed_region_label"),
                       ("surface", "surface_region_label")):
        labels = data[key]
        for number, info in enumerate(report["regions"][group], start=1):
            flat = np.flatnonzero(labels == number)
            kind = info["type"]
            regions.append({
                "id": info["id"],
                "side": "added" if kind.startswith("added") else "removed",
                "surface": kind.endswith("surface"),
                "flat": flat,
                "points": grid.centers(flat),
            })
    return regions


def group_side(grid: Grid, regions: list[dict],
               present_occupied: np.ndarray) -> tuple[list[list[int]], dict[tuple[int, int], str]]:
    """Union-find grouping of one side's non-surface regions: index groups and the reason for each link."""
    n = len(regions)
    uf = UnionFind(n)
    evidence: dict[tuple[int, int], str] = {}

    # Same connected, object-sized piece of the scan's own occupied surface.
    components, _ = ndimage.label(grid_volume(grid, present_occupied), structure=CUBE)
    components = components.ravel()
    component_ok: dict[int, bool] = {}
    for label in sorted({int(c) for r in regions for c in np.unique(components[r["flat"]]) if c > 0}):
        pts = grid.centers(np.flatnonzero(components == label))
        span = pts.max(axis=0) - pts.min(axis=0) + grid.voxel
        component_ok[label] = bool(span[0] <= OBJECT_MAX_FOOTPRINT_M and span[1] <= OBJECT_MAX_FOOTPRINT_M
                                   and span[2] <= OBJECT_MAX_HEIGHT_M)
    owners: dict[int, list[int]] = {}
    for i, region in enumerate(regions):
        for label in np.unique(components[region["flat"]]):
            if label > 0 and component_ok[int(label)]:
                owners.setdefault(int(label), []).append(i)
    for members in owners.values():
        for j in members[1:]:
            uf.union(members[0], j)
            evidence[(members[0], j)] = "same object-sized surface component"

    # Thin coplanar panels overlapping along the plane.
    prints = [footprint(r["points"], grid.voxel) for r in regions]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = prints[i], prints[j]
            if not all(p["width_m"] <= PANEL_MAX_THICKNESS_M + EPS and p["elongation"] >= PANEL_MIN_ELONGATION
                       for p in (a, b)):
                continue
            angle = abs((a["axis_deg"] - b["axis_deg"] + 90.0) % 180.0 - 90.0)
            if angle > PANEL_MAX_ANGLE_DEG:
                continue
            normal = a["normal"]
            if abs(np.median(regions[j]["points"][:, :2] @ normal) - a["offset_across"]) > PANEL_COPLANAR_M:
                continue
            along_b = regions[j]["points"][:, :2] @ a["major"]
            if along_b.max() < a["along_range"][0] or along_b.min() > a["along_range"][1]:
                continue
            uf.union(i, j)
            evidence[(i, j)] = "thin coplanar panel"

    # Fragments join their nearest neighbour within the surface tolerance.
    distance_maps = {}
    for i, region in enumerate(regions):
        if len(region["flat"]) >= FRAGMENT_MAX_VOXELS:
            continue
        best, best_gap = None, None
        for j, other in enumerate(regions):
            if j == i:
                continue
            if j not in distance_maps:
                mask = np.zeros(grid.size, dtype=bool)
                mask[other["flat"]] = True
                distance_maps[j] = ndimage.distance_transform_edt(~grid_volume(grid, mask)).ravel()
            gap = float(distance_maps[j][region["flat"]].min())
            if gap <= FRAGMENT_ATTACH_GAP_VOXELS and (best_gap is None or gap < best_gap):
                best, best_gap = j, gap
        if best is not None:
            uf.union(i, best)
            evidence[(min(i, best), max(i, best))] = f"fragment within {best_gap * grid.voxel:.2f} m"

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: -sum(len(regions[i]["flat"]) for i in g)), evidence


def grid_volume(grid: Grid, flat_mask: np.ndarray) -> np.ndarray:
    return flat_mask.reshape(grid.shape)


def build_surfaces(grid: Grid, surface_regions: list[dict], object_regions: list[dict]) -> tuple[list[dict], set[str]]:
    """Merge surface sheets with their coplanar pieces; return surfaces and absorbed object region ids."""
    surfaces, used, absorbed = [], set(), set()
    for main in sorted(surface_regions, key=lambda r: -len(r["flat"])):
        if main["id"] in used:
            continue
        fp = footprint(main["points"], grid.voxel)
        low, high = fp["along_range"]
        sheet_columns = cKDTree(np.unique(main["points"][:, :2].round(4), axis=0))

        def belongs(region: dict) -> bool:
            if region["side"] != main["side"]:
                return False
            columns = np.unique(region["points"][:, :2].round(4), axis=0)
            gap = sheet_columns.query(columns)[0]
            along = columns @ fp["major"]
            return bool(gap.max() <= SHEET_ATTACH_TOLERANCE_M + EPS
                        and along.min() >= low - SHEET_EXTENT_MARGIN_M - EPS
                        and along.max() <= high + SHEET_EXTENT_MARGIN_M + EPS)

        members = [main] + [r for r in surface_regions if r["id"] not in used and r is not main and belongs(r)]
        extra = [r for r in object_regions if r["id"] not in absorbed and belongs(r)]
        for r in members:
            used.add(r["id"])
        for r in extra:
            absorbed.add(r["id"])
        points = np.vstack([r["points"] for r in members + extra])
        surfaces.append({"members": [r["id"] for r in members + extra], "side": main["side"],
                         "points": points, "flat": np.concatenate([r["flat"] for r in members + extra]),
                         "absorbed_object_regions": [r["id"] for r in extra]})
    return surfaces, absorbed


def describe(grid: Grid, obj: dict, floor_z: float, cloud: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    points = obj["points"]
    keep = trim_slivers(points, grid.voxel)
    body = points[keep]
    obj["body_points"] = body
    obj["body_flat"] = obj["flat"][keep]
    obj["trimmed_voxels"] = int((~keep).sum())
    fp = footprint(body, grid.voxel)
    obj["footprint"] = fp
    obj["top_above_floor_m"] = float(body[:, 2].max() + grid.voxel / 2 - floor_z)
    obj["bottom_above_floor_m"] = float(body[:, 2].min() - grid.voxel / 2 - floor_z)
    obj["centroid"] = np.array([fp["centre"][0], fp["centre"][1], float(body[:, 2].mean())])
    xyz, rgb, flat = cloud
    inside = np.isin(flat, obj["body_flat"])
    obj["cloud_points"] = xyz[inside]
    obj["cloud_colors"] = rgb[inside]
    median = np.median(rgb[inside], axis=0) if inside.any() else np.array([0.5, 0.5, 0.5])
    obj["median_rgb"] = median
    obj["lab"] = lab_of(median)
    panel = fp["width_m"] <= PANEL_MAX_THICKNESS_M + EPS and fp["elongation"] >= PANEL_MIN_ELONGATION
    obj["shape_class"] = "thin panel" if panel else "volume"


YAWS = np.radians(np.arange(0.0, 360.0, YAW_STEP_DEG))
_SHIFT_AXIS = np.arange(-SHIFT_RANGE_M, SHIFT_RANGE_M + 1e-9, SHIFT_STEP_M)
SHIFTS = np.array([(dx, dy) for dx in _SHIFT_AXIS for dy in _SHIFT_AXIS])


def shape_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Best symmetric overlap of source onto target over yaw and small xy offsets.

    Both are body voxel centres; the source is rotated about its floor-plan
    centroid, placed on the target's centroid (bottom-aligned in z) and nudged.
    Returns the best score and the best score per yaw."""
    src = source - np.array([*source[:, :2].mean(axis=0), source[:, 2].min()])
    anchor = np.array([*target[:, :2].mean(axis=0), target[:, 2].min()])
    target_tree = cKDTree(target)
    per_yaw = np.zeros(len(YAWS))
    for index, yaw in enumerate(YAWS):
        c, s = np.cos(yaw), np.sin(yaw)
        rotated = src @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T + anchor
        best = 0.0
        for dx, dy in SHIFTS:
            moved = rotated + np.array([dx, dy, 0.0])
            forward = target_tree.query(moved, distance_upper_bound=FIT_TOLERANCE_M)[0]
            backward = cKDTree(moved).query(target, distance_upper_bound=FIT_TOLERANCE_M)[0]
            score = 0.5 * (np.isfinite(forward).mean() + np.isfinite(backward).mean())
            best = max(best, score)
        per_yaw[index] = best
    return float(per_yaw.max()), per_yaw


def path_between(grid: Grid, old: dict, new: dict, state1: np.ndarray, state2: np.ndarray) -> dict:
    """Occupancy along the straight path between two objects (both objects +1 voxel excluded),
    sampled at every voxel level the two bodies share."""
    zone = np.zeros(grid.size, dtype=bool)
    zone[old["body_flat"]] = True
    zone[new["body_flat"]] = True
    zone = ndimage.binary_dilation(zone.reshape(grid.shape), structure=CUBE).ravel()
    a, b = old["centroid"][:2], new["centroid"][:2]
    z_low = max(old["body_points"][:, 2].min(), new["body_points"][:, 2].min())
    z_high = min(old["body_points"][:, 2].max(), new["body_points"][:, 2].max())
    if z_high < z_low:
        return {"samples": 0, "occupied_in_both": 0.0, "free_in_either": 0.0, "unknown_in_both": 0.0}
    steps = np.linspace(0.0, 1.0, int(np.linalg.norm(b - a) / PATH_SAMPLE_M) + 1)
    levels = np.arange(z_low, z_high + EPS, grid.voxel)
    samples = np.array([[*(a + t * (b - a)), z] for t in steps for z in levels])
    flat = grid.flat_index(samples)
    flat = flat[(flat >= 0)]
    flat = flat[~zone[flat]]
    if len(flat) == 0:
        return {"samples": 0, "occupied_in_both": 0.0, "free_in_either": 0.0, "unknown_in_both": 0.0}
    return {"samples": int(len(flat)),
            "occupied_in_both": round(float(((state1[flat] == 1) & (state2[flat] == 1)).mean()), 3),
            "free_in_either": round(float(((state1[flat] == -1) | (state2[flat] == -1)).mean()), 3),
            "unknown_in_both": round(float(((state1[flat] == 0) & (state2[flat] == 0)).mean()), 3)}


def wall_relation(obj: dict, walls: list[dict]) -> dict | None:
    fp = obj["footprint"]
    centre = obj["centroid"]
    nearest = min(walls, key=lambda w: abs(centre @ w["normal"] - w["offset"]))
    distance = abs(centre @ nearest["normal"] - nearest["offset"])
    result = {"wall": nearest["name"], "distance_to_wall_m": round(float(distance), 2)}
    if fp["elongation"] >= MIN_ELONGATION_FOR_AXIS:
        along_wall = np.degrees(np.arctan2(-nearest["normal"][0], nearest["normal"][1])) % 180.0
        angle = abs((fp["axis_deg"] - along_wall + 90.0) % 180.0 - 90.0)
        result["long_axis_vs_wall_deg"] = round(float(angle), 1)
        result["relation"] = ("parallel" if angle <= WALL_PARALLEL_DEG
                              else "perpendicular" if angle >= 90.0 - WALL_PARALLEL_DEG else "oblique")
    else:
        result["relation"] = "no dominant axis"
    return result


def axis_relation(axis_deg: float, wall: dict) -> dict:
    along_wall = np.degrees(np.arctan2(-wall["normal"][0], wall["normal"][1])) % 180.0
    angle = abs((axis_deg - along_wall + 90.0) % 180.0 - 90.0)
    return {"wall": wall["name"], "long_axis_vs_wall_deg": round(float(angle), 1),
            "relation": ("parallel" if angle <= WALL_PARALLEL_DEG
                         else "perpendicular" if angle >= 90.0 - WALL_PARALLEL_DEG else "oblique")}


def relation_to(obj: dict, wall: dict) -> dict:
    fp = obj["footprint"]
    out = {"wall": wall["name"],
           "distance_to_wall_m": round(float(abs(obj["centroid"] @ wall["normal"] - wall["offset"])), 2)}
    if fp["elongation"] >= MIN_ELONGATION_FOR_AXIS:
        along_wall = np.degrees(np.arctan2(-wall["normal"][0], wall["normal"][1])) % 180.0
        angle = abs((fp["axis_deg"] - along_wall + 90.0) % 180.0 - 90.0)
        out["long_axis_vs_wall_deg"] = round(float(angle), 1)
        out["relation"] = ("parallel" if angle <= WALL_PARALLEL_DEG
                           else "perpendicular" if angle >= 90.0 - WALL_PARALLEL_DEG else "oblique")
    else:
        out["relation"] = "no dominant axis"
    return out


def save_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(inside_workspace(path)), cloud)


def render(background: np.ndarray, removed: list[dict], added: list[dict], surfaces: list[dict],
           small: list[dict], moves: list[dict], matrix: np.ndarray, path: Path) -> None:
    figure = plt.figure(figsize=(17, 8.6), dpi=150, facecolor="white")
    top = figure.add_axes((0.03, 0.08, 0.42, 0.84))
    heat = figure.add_axes((0.55, 0.20, 0.40, 0.64))
    rng = np.random.default_rng(0)
    if len(background) > 150_000:
        background = background[rng.choice(len(background), 150_000, replace=False)]
    top.scatter(background[:, 0], background[:, 1], s=0.12, c="#c3c7cf", linewidths=0, rasterized=True)
    for objs, color in ((surfaces, "#e6a100"), (small, "#8a8f99"), (removed, "#d7263d"), (added, "#1b9e4b")):
        for obj in objs:
            pts = obj.get("body_points", obj["points"])
            top.scatter(pts[:, 0], pts[:, 1], s=4, c=color, linewidths=0, alpha=0.75, rasterized=True)
            top.annotate(obj["id"], obj["centroid"][:2] if "centroid" in obj else pts[:, :2].mean(axis=0),
                         fontsize=7.5, fontweight="bold", color=color, xytext=(5, 5),
                         textcoords="offset points")
    for move in moves:
        a, b = move["from_xy"], move["to_xy"]
        style = dict(arrowstyle="-|>", lw=2.0, color="#1f3a93")
        if move["event"] == "shifted":
            style.update(arrowstyle="-", linestyle="--", lw=1.4, color="#7a4fb3")
        top.annotate("", xy=b, xytext=a, arrowprops=style)
        mid = (np.asarray(a) + np.asarray(b)) / 2
        top.text(mid[0], mid[1], f" {move['id']}", fontsize=9, fontweight="bold", color="#1f3a93")
    top.set_aspect("equal", adjustable="box")
    top.set_xlabel("X [m]")
    top.set_ylabel("Y [m]")
    top.set_title("Top view: objects and matched moves")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color="#d7263d", label="removed object (scan 1 position)"),
               plt.Line2D([0], [0], marker="s", linestyle="", color="#1b9e4b", label="added object (scan 2 position)"),
               plt.Line2D([0], [0], marker="s", linestyle="", color="#e6a100", label="surface change"),
               plt.Line2D([0], [0], marker="s", linestyle="", color="#8a8f99", label="small change (< 15 cm cube)"),
               plt.Line2D([0], [0], color="#1f3a93", lw=2, label="matched move"),
               plt.Line2D([0], [0], color="#7a4fb3", lw=1.4, linestyle="--",
                          label="small shift (two faces of one object)")]
    top.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)

    image = heat.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    heat.set_xticks(range(len(added)), [o["id"] for o in added], rotation=45, ha="right", fontsize=8)
    heat.set_yticks(range(len(removed)), [o["id"] for o in removed], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            heat.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if matrix[i, j] > 0.6 else "#222222")
    for move in moves:
        i, j = move["matrix_index"]
        heat.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.2, ec="#d7263d"))
    heat.set_xlabel("added objects (scan 2)")
    heat.set_ylabel("removed objects (scan 1)")
    heat.set_title(f"Match score = shape^0.5 x height^0.25 x colour^0.25; boxed = accepted (>= {MATCH_MIN_SCORE})",
                   fontsize=9)
    figure.colorbar(image, ax=heat, fraction=0.04)
    figure.suptitle("Step 2 - object grouping and move matching", fontsize=13, fontweight="bold")
    figure.savefig(str(inside_workspace(path)))
    plt.close(figure)


def main() -> None:
    started = time.time()
    output = inside_workspace(OUTPUT)
    (output / "objects").mkdir(parents=True, exist_ok=True)
    data = np.load(str(inside_workspace(output / "voxel_change_grid.npz")))
    report = json.loads(inside_workspace(output / "change_detection_report.json").read_text())
    grid = Grid(data)

    structural = [p for p in report["structural_planes"] if p["treated_as_structural"]]
    surface_tolerance = report["parameters"]["surface_tolerance_voxels"] * grid.voxel
    floor_z = min(p["offset_m"] for p in structural if p["kind"] == "horizontal")
    walls = []
    for p in structural:
        if p["kind"] == "vertical":
            normal = np.array(p["normal"], dtype=np.float64)
            axis = "x" if abs(normal[0]) > abs(normal[1]) else "y"
            walls.append({"name": f"wall {axis} = {p['offset_m']:+.2f} m", "normal": normal,
                          "offset": float(p["offset_m"])})

    clouds = {}
    for session, name in SCAN_CLOUDS.items():
        cloud = o3d.io.read_point_cloud(str(inside_workspace(ALIGNMENT / name)))
        xyz = np.asarray(cloud.points)
        clouds[session] = (xyz, np.asarray(cloud.colors), grid.flat_index(xyz))

    structural_mask = data["structural_mask"]
    occupied = {"removed": (data["state_scan1"] == 1) & ~structural_mask,
                "added": (data["state_scan2"] == 1) & ~structural_mask}
    present_session = {"removed": BEFORE, "added": AFTER}

    regions = load_regions(grid, data, report)
    surface_regions = [r for r in regions if r["surface"]]
    object_regions = [r for r in regions if not r["surface"]]
    surfaces, absorbed = build_surfaces(grid, surface_regions, object_regions)

    objects: dict[str, list[dict]] = {}
    grouping_evidence = []
    for side in ("removed", "added"):
        side_regions = [r for r in object_regions if r["side"] == side and r["id"] not in absorbed]
        groups, evidence = group_side(grid, side_regions, occupied[side])
        for (i, j), reason in sorted(evidence.items()):
            grouping_evidence.append({"side": side, "regions": [side_regions[i]["id"], side_regions[j]["id"]],
                                      "reason": reason})
        side_objects = []
        for group in groups:
            members = sorted((side_regions[i] for i in group), key=lambda r: -len(r["flat"]))
            obj = {"side": side, "members": [r["id"] for r in members],
                   "flat": np.concatenate([r["flat"] for r in members]),
                   "points": np.vstack([r["points"] for r in members])}
            describe(grid, obj, floor_z, clouds[present_session[side]])
            side_objects.append(obj)
        objects[side] = side_objects

    for number, surface in enumerate(surfaces, start=1):
        surface["id"] = f"S{number}"
        describe(grid, surface, floor_z, clouds[present_session[surface["side"]]])

    # Objects below the 15 cm cube floor are reported but never interpreted.
    small, candidates = [], {"removed": [], "added": []}
    for side in ("removed", "added"):
        for obj in objects[side]:
            (candidates[side] if len(obj["flat"]) >= FRAGMENT_MAX_VOXELS else small).append(obj)
    for side, prefix in (("removed", "OR"), ("added", "OA")):
        for number, obj in enumerate(candidates[side], start=1):
            obj["id"] = f"{prefix}{number}"
    for number, obj in enumerate(small, start=1):
        obj["id"] = f"X{number}"

    removed, added = candidates["removed"], candidates["added"]
    print(f"Objects: {len(removed)} removed, {len(added)} added, {len(surfaces)} surface, "
          f"{len(small)} small (< {FRAGMENT_MAX_VOXELS} voxels)", flush=True)
    for obj in removed + added + small + surfaces:
        fp = obj["footprint"]
        print(f"  {obj['id']:>4} {obj['side']:<7} {'+'.join(obj['members']):<16} {len(obj['flat']):4d} vox "
              f"(trim {obj['trimmed_voxels']:2d})  top {obj['top_above_floor_m']:.2f} m  "
              f"footprint {fp['length_m']:.2f} x {fp['width_m']:.2f} @ {fp['axis_deg']:5.1f} deg  "
              f"{obj['shape_class']:<10} rgb {(obj['median_rgb'] * 255).round().astype(int).tolist()}", flush=True)

    # Pairwise similarity.
    cues = {}
    matrix = np.zeros((len(removed), len(added)))
    for i, old in enumerate(removed):
        for j, new in enumerate(added):
            fit, per_yaw = shape_fit(old["body_points"], new["body_points"])
            height = max(0.0, 1.0 - abs(old["top_above_floor_m"] - new["top_above_floor_m"]) / HEIGHT_SCALE_M)
            delta_e = float(np.linalg.norm(old["lab"] - new["lab"]))
            colour = max(0.0, 1.0 - delta_e / COLOUR_SCALE_DE)
            score = (fit ** WEIGHTS["shape"]) * (height ** WEIGHTS["height"]) * (colour ** WEIGHTS["colour"])
            matrix[i, j] = score
            cues[(i, j)] = {"shape_fit": round(fit, 3), "height_similarity": round(height, 3),
                            "colour_similarity": round(colour, 3), "colour_delta_e": round(delta_e, 1),
                            "score": round(score, 3), "per_yaw": per_yaw}
        print(f"  {old['id']} vs added: " + "  ".join(
            f"{a['id']} {matrix[i, j]:.2f} (s{cues[(i, j)]['shape_fit']:.2f} h{cues[(i, j)]['height_similarity']:.2f} "
            f"c{cues[(i, j)]['colour_similarity']:.2f})" for j, a in enumerate(added)), flush=True)

    rows, cols = linear_sum_assignment(-matrix) if matrix.size else (np.array([], int), np.array([], int))
    moves, matched_removed, matched_added = [], set(), set()
    for i, j in sorted(zip(rows.tolist(), cols.tolist()), key=lambda ij: -matrix[ij[0], ij[1]]):
        if matrix[i, j] < MATCH_MIN_SCORE:
            continue
        old, new = removed[i], added[j]
        cue = cues[(i, j)]
        per_yaw = cue.pop("per_yaw")
        best_yaw = float(np.degrees(YAWS[int(per_yaw.argmax())]))
        opposite = per_yaw[(int(per_yaw.argmax()) + len(YAWS) // 2) % len(YAWS)]
        if per_yaw.max() < ROTATION_MIN_FIT:
            rotation = {"rotation_deg": None,
                        "note": f"not determinable: best shape fit {per_yaw.max():.2f} < {ROTATION_MIN_FIT}"}
        elif per_yaw.max() - per_yaw.min() < YAW_UNDETERMINED:
            rotation = {"rotation_deg": None, "note": "not determinable: shape fits equally at every yaw"}
        else:
            signed = (best_yaw + 180.0) % 360.0 - 180.0
            if per_yaw.max() - opposite <= YAW_AMBIGUITY:
                folded = (signed + 90.0) % 180.0 - 90.0
                rotation = {"rotation_deg": round(folded, 1),
                            "note": "axis rotation; front/back (180 deg) not distinguishable"}
            else:
                rotation = {"rotation_deg": round(signed, 1), "note": "from best shape fit"}
        axis_change = None
        if min(old["footprint"]["elongation"], new["footprint"]["elongation"]) >= MIN_ELONGATION_FOR_AXIS:
            axis_change = round(abs((new["footprint"]["axis_deg"] - old["footprint"]["axis_deg"] + 90.0)
                                    % 180.0 - 90.0), 1)
        # Orientation after the move implied by the fitted rotation: more reliable than the
        # new footprint when the new view is partial or carries edge residue.
        implied = None
        if rotation["rotation_deg"] is not None and old["footprint"]["elongation"] >= MIN_ELONGATION_FOR_AXIS:
            implied_axis = (old["footprint"]["axis_deg"] + rotation["rotation_deg"]) % 180.0
            implied = {"long_axis_deg": round(implied_axis, 1)}
        alternatives_row = sorted(matrix[i, :].tolist(), reverse=True)
        alternatives_col = sorted(matrix[:, j].tolist(), reverse=True)
        second = max(alternatives_row[1] if len(alternatives_row) > 1 else 0.0,
                     alternatives_col[1] if len(alternatives_col) > 1 else 0.0)
        old_wall = wall_relation(old, walls)
        old_wall_obj = next(w for w in walls if w["name"] == old_wall["wall"])
        translation = new["centroid"] - old["centroid"]
        path = path_between(grid, old, new, data["state_scan1"], data["state_scan2"])
        shifted = path["occupied_in_both"] >= SHIFT_PERSISTENT_FRACTION
        direction = translation[:2] / max(np.linalg.norm(translation[:2]), 1e-9)
        layer = max(float(np.ptp(o["body_points"][:, :2] @ direction)) + grid.voxel for o in (old, new))
        kind = "shifted" if shifted else "moved"
        number = sum(m["event"] == kind for m in moves) + 1
        moves.append({
            "event": kind,
            "id": f"{'SH' if shifted else 'M'}{number}",
            "path_between": path,
            "from": old["id"], "to": new["id"],
            "from_regions": old["members"], "to_regions": new["members"],
            "from_centroid_m": old["centroid"].round(3).tolist(),
            "to_centroid_m": new["centroid"].round(3).tolist(),
            "translation_m": translation.round(3).tolist(),
            "horizontal_distance_m": round(float(np.linalg.norm(translation[:2])), 3),
            "rotation": rotation,
            "footprint_axis_change_deg": axis_change,
            "before_vs_nearest_wall": old_wall,
            "after_vs_nearest_wall": wall_relation(new, walls),
            "after_vs_before_wall": relation_to(new, old_wall_obj),
            "after_axis_from_rotation": (None if implied is None else {
                **implied, **axis_relation(implied["long_axis_deg"], old_wall_obj)}),
            "cues": cue,
            "margin_to_next_best": round(float(matrix[i, j] - second), 3),
            **({"interpretation": "the removed and added parts are the trailing and leading faces of one "
                                  "larger object that stayed in place between them; it slid by about the "
                                  "thickness of the changed layers plus the surface tolerance, not by the "
                                  "face-to-face distance",
                "face_to_face_distance_m": round(float(np.linalg.norm(translation[:2])), 3),
                "changed_layer_m": round(layer, 3),
                # Step 1 hides the first SURFACE_TOLERANCE in front of the other scan's surface,
                # so each changed face is that much thinner than the real slide.
                "shift_estimate_m": round(layer + surface_tolerance, 3),
                "shift_uncertainty_m": round(grid.voxel, 3),
                "shift_direction_xy": direction.round(3).tolist()} if shifted else {}),
            "matrix_index": (i, j),
            "from_xy": old["centroid"][:2].tolist(), "to_xy": new["centroid"][:2].tolist(),
        })
        matched_removed.add(i)
        matched_added.add(j)

    # Object ids are re-numbered on every run: remove clouds left by an earlier run.
    for stale in (output / "objects").glob("*.ply"):
        inside_workspace(stale).unlink()
    np.savez_compressed(str(inside_workspace(output / "objects" / "object_voxels.npz")),
                        **{obj["id"]: obj["body_flat"] for obj in removed + added + small + surfaces})
    for obj in removed + added + small + surfaces:
        path = output / "objects" / f"{obj['id']}.ply"
        save_cloud(path, obj["cloud_points"], obj["cloud_colors"])
        obj["ply"] = out_rel(path)

    def public(obj: dict) -> dict:
        fp = obj["footprint"]
        body = obj["body_points"]
        return {
            "id": obj["id"], "side": obj["side"],
            "present_in": present_session[obj["side"]],
            "member_regions": obj["members"],
            "voxels": int(len(obj["flat"])), "sliver_voxels_trimmed": obj["trimmed_voxels"],
            "centroid_m": obj["centroid"].round(3).tolist(),
            "bbox_min_m": (body.min(axis=0) - grid.voxel / 2).round(3).tolist(),
            "bbox_max_m": (body.max(axis=0) + grid.voxel / 2).round(3).tolist(),
            "top_above_floor_m": round(obj["top_above_floor_m"], 3),
            "bottom_above_floor_m": round(obj["bottom_above_floor_m"], 3),
            "footprint": {"length_m": round(fp["length_m"], 3), "width_m": round(fp["width_m"], 3),
                          "long_axis_deg": round(fp["axis_deg"], 1), "elongation": round(fp["elongation"], 2)},
            "shape_class": obj["shape_class"],
            "median_rgb": (obj["median_rgb"] * 255).round().astype(int).tolist(),
            "points": int(len(obj["cloud_points"])),
            "ply": obj["ply"],
        }

    events = []
    for move in moves:
        events.append({k: v for k, v in move.items() if k not in ("matrix_index", "from_xy", "to_xy")})
    for i, obj in enumerate(removed):
        if i not in matched_removed:
            events.append({"event": "removed", "object": obj["id"], "regions": obj["members"],
                           "best_candidate": (added[int(matrix[i].argmax())]["id"] if len(added) else None),
                           "best_score": round(float(matrix[i].max()), 3) if len(added) else None,
                           "note": "no added object matched; it disappeared from here or moved somewhere "
                                   "only one recording observed"})
    for j, obj in enumerate(added):
        if j not in matched_added:
            events.append({"event": "added", "object": obj["id"], "regions": obj["members"],
                           "best_candidate": (removed[int(matrix[:, j].argmax())]["id"] if len(removed) else None),
                           "best_score": round(float(matrix[:, j].max()), 3) if len(removed) else None,
                           "note": "no removed object matched; it appeared here or came from somewhere "
                                   "only one recording observed"})
    for surface in surfaces:
        events.append({"event": "surface_changed", "object": surface["id"], "regions": surface["members"],
                       "absorbed_object_regions": surface["absorbed_object_regions"],
                       "note": f"tall thin sheet present only in {present_session[surface['side']]}"})
    for obj in small:
        events.append({"event": "small_change", "object": obj["id"], "side": obj["side"],
                       "regions": obj["members"],
                       "note": f"under {FRAGMENT_MAX_VOXELS} voxels (smaller than a 15 cm cube); "
                               "reported, not interpreted"})

    result = {
        "schema_version": "1.0.0",
        "pipeline_version": PIPELINE_VERSION,
        "stage": "step2_object_grouping_and_move_matching",
        "common_frame": f"{BEFORE} PLY frame; metres; Z-up",
        "learned_models_used": [],
        "inputs": ["voxel_diff/voxel_change_grid.npz", "voxel_diff/change_detection_report.json",
                   *[f"alignment/{name}" for name in SCAN_CLOUDS.values()]],
        "step1_change_mask_sha256": report["change_voxels"]["change_mask_sha256"],
        "parameters": {
            "object_max_footprint_m": OBJECT_MAX_FOOTPRINT_M, "object_max_height_m": OBJECT_MAX_HEIGHT_M,
            "panel_max_thickness_m": PANEL_MAX_THICKNESS_M, "panel_min_elongation": PANEL_MIN_ELONGATION,
            "panel_coplanar_m": PANEL_COPLANAR_M, "panel_max_angle_deg": PANEL_MAX_ANGLE_DEG,
            "fragment_max_voxels": FRAGMENT_MAX_VOXELS,
            "fragment_attach_gap_m": FRAGMENT_ATTACH_GAP_VOXELS * grid.voxel,
            "sheet_attach_tolerance_m": SHEET_ATTACH_TOLERANCE_M,
            "sheet_extent_margin_m": SHEET_EXTENT_MARGIN_M, "sliver_fraction": SLIVER_FRACTION,
            "column_support_fraction": COLUMN_SUPPORT_FRACTION,
            "yaw_step_deg": YAW_STEP_DEG, "shift_range_m": SHIFT_RANGE_M, "shift_step_m": SHIFT_STEP_M,
            "fit_tolerance_m": FIT_TOLERANCE_M, "height_scale_m": HEIGHT_SCALE_M,
            "colour_scale_delta_e": COLOUR_SCALE_DE, "weights": WEIGHTS,
            "score": "shape^w_shape * height^w_height * colour^w_colour",
            "match_min_score": MATCH_MIN_SCORE, "rotation_min_fit": ROTATION_MIN_FIT,
            "yaw_ambiguity": YAW_AMBIGUITY,
            "yaw_undetermined": YAW_UNDETERMINED, "wall_parallel_deg": WALL_PARALLEL_DEG,
            "shift_persistent_fraction": SHIFT_PERSISTENT_FRACTION, "path_sample_m": PATH_SAMPLE_M,
        },
        "floor_z_m": round(floor_z, 4),
        "walls": [{"name": w["name"], "normal": w["normal"].round(4).tolist(), "offset_m": w["offset"]}
                  for w in walls],
        "grouping_evidence": grouping_evidence,
        "objects": {"removed": [public(o) for o in removed], "added": [public(o) for o in added],
                    "surface": [public(o) for o in surfaces], "small": [public(o) for o in small]},
        "match_matrix": {"rows_removed": [o["id"] for o in removed], "cols_added": [o["id"] for o in added],
                         "score": matrix.round(3).tolist()},
        "events": events,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__},
    }
    digest = hashlib.sha256(json.dumps({"objects": result["objects"], "events": events,
                                        "match_matrix": result["match_matrix"]},
                                       sort_keys=True).encode()).hexdigest()
    result["result_sha256"] = digest
    result["runtime_s"] = round(time.time() - started, 1)
    (output / "object_changes.json").write_text(json.dumps(result, indent=2))

    background = np.asarray(o3d.io.read_point_cloud(
        str(inside_workspace(ALIGNMENT / "shared_aligned_room_pointcloud.ply"))).points)
    render(background, removed, added, surfaces, small, moves, matrix,
           output / "qa" / "step2_objects_and_moves.png")

    print("\nEvents:")
    for event in events:
        if event["event"] in ("moved", "shifted"):
            print(f"  {event['id']} {event['event']} path {event['path_between']}")
            print(f"  {event['id']} {event['event']} {event['from']} {event['from_regions']} -> {event['to']} "
                  f"{event['to_regions']}: {event['horizontal_distance_m']:.2f} m, rotation "
                  f"{event['rotation']}, axis change {event['footprint_axis_change_deg']}, "
                  f"score {event['cues']['score']} (margin {event['margin_to_next_best']})")
            print(f"       before {event['before_vs_nearest_wall']}  after {event['after_vs_nearest_wall']}  "
                  f"after-vs-before-wall {event['after_vs_before_wall']}")
        else:
            print(f"  {event['event']:<15} {event['object']:<4} {event['regions']}  "
                  f"{event.get('best_candidate', '')} {event.get('best_score', '')}")
    print(f"Result sha256: {digest}")
    print(f"Done in {time.time() - started:.0f}s -> {output / 'object_changes.json'}")


if __name__ == "__main__":
    main()
