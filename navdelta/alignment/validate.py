"""Deterministic structural and artifact validation for the final alignment.

Checks the exported poses, the transform invariants, that the aligned PLY is the
input PLY under the reported transform, and that both floors agree; writes
<outputs>/alignment/validation_report.json and a SHA-256 artifact manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


from navdelta import context
from navdelta.context import ALIGNMENT as OUTPUT, DATA, OUTPUTS, inside

BEFORE, AFTER = context.SESSIONS
PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def ply_layout(path: Path) -> tuple[int, int, np.dtype]:
    lines = []
    with inside(path).open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Missing PLY end_header: {path}")
            line = raw.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                offset = stream.tell()
                break
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"Expected binary little-endian PLY: {path}")
    count = None
    current = None
    properties = []
    for line in lines:
        fields = line.split()
        if len(fields) == 3 and fields[0] == "element":
            current = fields[1]
            if current == "vertex":
                count = int(fields[2])
        elif len(fields) == 3 and fields[0] == "property" and current == "vertex":
            properties.append((fields[2], PLY_TYPES[fields[1]]))
    if count is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    return offset, count, np.dtype(properties)


def ply_points(path: Path, maximum: int = 800_000) -> tuple[np.ndarray, int]:
    offset, count, dtype = ply_layout(path)
    vertices = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))
    indices = np.linspace(0, count - 1, min(count, maximum), dtype=np.int64)
    selected = vertices[indices]
    xyz = np.column_stack((selected["x"], selected["y"], selected["z"])).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)], count


def dominant_horizontal_plane(points: np.ndarray, upper: bool = False) -> dict:
    """Fit z=ax+by+c at the dominant low/high horizontal surface."""
    low, high = np.quantile(points[:, 2], [0.01, 0.99])
    span = high - low
    if upper:
        candidates = points[points[:, 2] > high - 0.24 * span]
    else:
        candidates = points[points[:, 2] < low + 0.24 * span]
    bins = max(20, int(math.ceil(span / 0.01)))
    histogram, edges = np.histogram(candidates[:, 2], bins=bins)
    peak = int(np.argmax(histogram))
    peak_z = 0.5 * (edges[peak] + edges[peak + 1])
    selected = candidates[np.abs(candidates[:, 2] - peak_z) < 0.035]
    for threshold in (0.030, 0.020, 0.015):
        design = np.column_stack((selected[:, 0], selected[:, 1], np.ones(len(selected))))
        coefficients, *_ = np.linalg.lstsq(design, selected[:, 2], rcond=None)
        residual = selected[:, 2] - design @ coefficients
        mask = np.abs(residual) < threshold
        if int(mask.sum()) < 500:
            break
        selected = selected[mask]
    a, b, c = coefficients
    design = np.column_stack((selected[:, 0], selected[:, 1], np.ones(len(selected))))
    residual = selected[:, 2] - design @ coefficients
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    return {
        "equation": "z = a*x + b*y + c",
        "a": float(a), "b": float(b), "c": float(c),
        "normal": normal.tolist(),
        "inlier_count": int(len(selected)),
        "reliable_room_plane": bool(len(selected) >= 20_000),
        "residual_median_abs_m": float(np.median(np.abs(residual))),
        "residual_rms_m": float(np.sqrt(np.mean(np.square(residual)))),
        "xy_center_m": np.median(selected[:, :2], axis=0).tolist(),
    }


def compare_planes(first: dict, second: dict) -> dict:
    center = 0.5 * (np.asarray(first["xy_center_m"]) + np.asarray(second["xy_center_m"]))
    height1 = first["a"] * center[0] + first["b"] * center[1] + first["c"]
    height2 = second["a"] * center[0] + second["b"] * center[1] + second["c"]
    dot = abs(float(np.dot(first["normal"], second["normal"])))
    return {
        "evaluation_xy_m": center.tolist(),
        "height_difference_m": float(height2 - height1),
        "normal_angle_deg": float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))),
    }


def validate_pose_directory(name: str, expected: int) -> dict:
    directory = inside(OUTPUT / "aligned_frame_poses" / name)
    files = sorted(directory.glob("*.json"), key=lambda path: int(path.stem))
    orthogonality, determinants, bottom_rows = [], [], []
    support: dict[str, int] = {}
    referenced_paths = 0
    for path in files:
        record = json.loads(path.read_text())
        transform = np.asarray(record["T_camera_to_common"], dtype=np.float64)
        rotation = transform[:3, :3]
        orthogonality.append(float(np.linalg.norm(rotation.T @ rotation - np.eye(3))))
        determinants.append(float(np.linalg.det(rotation)))
        bottom_rows.append(float(np.linalg.norm(transform[3] - np.array([0.0, 0.0, 0.0, 1.0]))))
        label = record["pose_support"]["label"]
        support[label] = support.get(label, 0) + 1
        for key in ("rgb", "depth"):
            if not inside(DATA / record[key]).is_file():
                raise FileNotFoundError(record[key])
            referenced_paths += 1
    timestamps = [int(path.stem) for path in files]
    tum_rows = sum(1 for line in (OUTPUT / f"{name}_aligned_trajectory_tum.txt").read_text().splitlines() if line.strip())
    return {
        "expected_frames": expected,
        "pose_json_files": len(files),
        "tum_rows": tum_rows,
        "strictly_increasing_timestamps": bool(all(b > a for a, b in zip(timestamps, timestamps[1:]))),
        "referenced_rgb_depth_files_found": referenced_paths,
        "max_rotation_orthogonality_error": max(orthogonality, default=float("inf")),
        "determinant_range": [min(determinants, default=0.0), max(determinants, default=0.0)],
        "max_homogeneous_bottom_row_error": max(bottom_rows, default=float("inf")),
        "support_counts": support,
        "passed": bool(
            len(files) == expected == tum_rows
            and referenced_paths == 2 * expected
            and all(b > a for a, b in zip(timestamps, timestamps[1:]))
            and max(orthogonality, default=float("inf")) < 1e-6
            and min(determinants, default=0.0) > 0.999999
            and max(determinants, default=2.0) < 1.000001
            and max(bottom_rows, default=float("inf")) < 1e-9
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with inside(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest() -> Path:
    paths = [
        DATA / f"{BEFORE}_ply" / f"{BEFORE}.ply",
        DATA / f"{AFTER}_ply" / f"{AFTER}.ply",
        OUTPUT / "alignment_report.json",
        OUTPUT / f"{AFTER}_to_{BEFORE}_transform.json",
        *[OUTPUT / f"{name}_{suffix}" for name in (BEFORE, AFTER)
          for suffix in ("aligned_poses.jsonl", "pose_graph_edges.json", "optimized_rgbd_cloud.ply",
                         "aligned_trajectory_tum.txt")],
        OUTPUT / f"{AFTER}_aligned_to_{BEFORE}.ply",
        OUTPUT / "shared_aligned_room_pointcloud.ply",
        OUTPUT / "qa" / "alignment_overview.png",
        OUTPUT / "qa" / "alignment_topdown_xy.png",
        OUTPUT / "qa" / "alignment_elevation_xz.png",
        OUTPUT / "qa" / "alignment_elevation_yz.png",
        OUTPUT / "qa" / "alignment_qa_metadata.json",
        OUTPUT / "validation_report.json",
    ]
    entries = []
    for path in paths:
        if path.is_file():
            entries.append({
                "path": (f"data/{context.data_relative(path)}" if path.resolve().is_relative_to(DATA)
                         else f"outputs/{path.resolve().relative_to(OUTPUTS).as_posix()}"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "schema_version": "1.0.0",
        "hash": "SHA-256",
        "note": "The manifest intentionally excludes itself.",
        "files": entries,
    }
    path = inside(OUTPUT / "artifact_manifest.json")
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    report = json.loads(inside(OUTPUT / "alignment_report.json").read_text())
    transform_record = json.loads(inside(OUTPUT / f"{AFTER}_to_{BEFORE}_transform.json").read_text())
    transform = np.asarray(transform_record["T_source_to_target"], dtype=np.float64)
    rotation = transform[:3, :3]

    scan1_path = inside(DATA / f"{BEFORE}_ply" / f"{BEFORE}.ply")
    scan2_raw_path = inside(DATA / f"{AFTER}_ply" / f"{AFTER}.ply")
    scan2_aligned_path = inside(OUTPUT / f"{AFTER}_aligned_to_{BEFORE}.ply")
    scan1, scan1_count = ply_points(scan1_path)
    scan2_raw, scan2_count = ply_points(scan2_raw_path)
    scan2_aligned, scan2_aligned_count = ply_points(scan2_aligned_path)

    # Open3D preserves point order when applying and writing this rigid transform.
    common_count = min(len(scan2_raw), len(scan2_aligned))
    predicted = scan2_raw[:common_count] @ rotation.T + transform[:3, 3]
    export_error = np.linalg.norm(predicted - scan2_aligned[:common_count], axis=1)

    floor1 = dominant_horizontal_plane(scan1, upper=False)
    floor2 = dominant_horizontal_plane(scan2_aligned, upper=False)
    ceiling1 = dominant_horizontal_plane(scan1, upper=True)
    ceiling2 = dominant_horizontal_plane(scan2_aligned, upper=True)
    floor_comparison = compare_planes(floor1, floor2)
    ceiling_comparison = compare_planes(ceiling1, ceiling2)
    ceiling_available = bool(
        ceiling1["reliable_room_plane"] and ceiling2["reliable_room_plane"])

    pose_validation = {
        name: validate_pose_directory(name, int(report["input_frames"][name]))
        for name in (BEFORE, AFTER)
    }
    checks = {
        "report_internal_quality_gate": bool(report["internal_quality_gate"]["passed"]),
        "pose_exports": bool(all(row["passed"] for row in pose_validation.values())),
        "aligned_ply_point_count": bool(scan2_count == scan2_aligned_count),
        "aligned_ply_transform_application": bool(np.percentile(export_error, 99) < 1e-5),
        "floor_height_consistency_within_3cm": bool(abs(floor_comparison["height_difference_m"]) < 0.03),
        "floor_normal_consistency_within_1deg": bool(floor_comparison["normal_angle_deg"] < 1.0),
    }
    validation = {
        "schema_version": "1.0.0",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "transform_invariants": {
            "determinant": float(np.linalg.det(rotation)),
            "rotation_orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
            "gravity_tilt_deg": float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))),
        },
        "pose_exports": pose_validation,
        "point_cloud_exports": {
            "scan1_input_points": scan1_count,
            "scan2_input_points": scan2_count,
            "scan2_aligned_points": scan2_aligned_count,
            "transform_application_sample_count": common_count,
            "transform_application_error_median_m": float(np.median(export_error)),
            "transform_application_error_p99_m": float(np.percentile(export_error, 99)),
        },
        "structural_planes": {
            "method": (
                "dominant lower/upper z-band plane on a deterministic PLY sample; "
                "robust iterative z=ax+by+c fit"
            ),
            "scan1_floor": floor1,
            "aligned_scan2_floor": floor2,
            "floor_comparison": floor_comparison,
            "scan1_ceiling": ceiling1,
            "aligned_scan2_ceiling": ceiling2,
            "ceiling_comparison": ceiling_comparison,
            "ceiling_comparison_available": ceiling_available,
            "ceiling_note": (
                "Scan 2 does not contain a sufficiently dense ceiling plane for a reliable "
                "ceiling-height check. The fitted upper band is diagnostic only."
                if not ceiling_available else "Both scans contain a dense ceiling plane."
            ),
        },
        "claim_boundary": (
            "This is deterministic internal validation of supplied RGB-D/PLY data. "
            "It is not an external metrology or surveyed-ground-truth certificate."
        ),
    }
    path = inside(OUTPUT / "validation_report.json")
    path.write_text(json.dumps(validation, indent=2))
    manifest_path = write_manifest()
    summary = {
        "status": validation["status"],
        "checks": checks,
        "floor_comparison": floor_comparison,
        "ceiling_comparison_available": ceiling_available,
        "validation_report": path.relative_to(OUTPUTS).as_posix(),
        "manifest": manifest_path.relative_to(OUTPUTS).as_posix(),
    }
    if ceiling_available:
        summary["ceiling_comparison"] = ceiling_comparison
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
