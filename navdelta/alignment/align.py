"""Offline RGB-D/VIO alignment of two recordings into one shared frame.

The pipeline keeps the continuous metric ARKit/VIO trajectory of each recording
as a prior, adds depth-verified SIFT relative factors and loop closures, runs
robust pose-graph optimisation, registers the two independently reconstructed
rooms (gravity-locked: yaw + translation only), and exports every camera pose in
the common frame of the first ("before") recording's PLY.

Inputs:  <data>/<name>_imgs, <data>/<name>_rawdata, <data>/<name>_ply  (see docs/data-format.md)
Outputs: <outputs>/alignment/
"""

from __future__ import annotations

import csv
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from scipy.spatial import cKDTree

from navdelta import context
from navdelta.context import ALIGNMENT as DEFAULT_OUTPUT, DATA, OUTPUTS, inside

SESSION_NAMES = context.SESSIONS
PIPELINE_VERSION = "1.2.0"
REPORT_SCHEMA_VERSION = "1.0.0"

# OpenCV camera (right, down, forward) -> ARKit camera (right, up, backward).
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
# Polycam's aligned Y-up room coordinates -> exported Z-up PLY coordinates.
YUP_TO_ZUP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


inside_workspace = inside


def outputs_relative(path: Path) -> str:
    return str(path.resolve().relative_to(OUTPUTS)).replace("\\", "/")


def read_json(path: Path):
    return json.loads(inside_workspace(path).read_text())


def rotation_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))))


def transform_delta(measured: np.ndarray, prior: np.ndarray) -> tuple[float, float]:
    delta = measured @ np.linalg.inv(prior)
    return float(np.linalg.norm(delta[:3, 3])), rotation_deg(delta[:3, :3])


def rigid_fit(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(source), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= max(weights.sum(), 1e-12)
    source_mean = (source * weights[:, None]).sum(axis=0)
    target_mean = (target * weights[:, None]).sum(axis=0)
    covariance = (source - source_mean).T @ ((target - target_mean) * weights[:, None])
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def robust_fit(source: np.ndarray, target: np.ndarray, initial: np.ndarray,
               thresholds: Iterable[float] = (0.15, 0.10, 0.07, 0.05)) -> tuple[np.ndarray, np.ndarray]:
    transform = initial.copy()
    mask = np.ones(len(source), dtype=bool)
    for threshold in thresholds:
        transformed = source @ transform[:3, :3].T + transform[:3, 3]
        residual = np.linalg.norm(transformed - target, axis=1)
        mask = residual < threshold
        if mask.sum() < 6:
            break
        # Tukey-style weights within the current consensus.
        normalized = residual[mask] / threshold
        weights = np.square(np.clip(1.0 - normalized * normalized, 0.0, 1.0))
        transform = rigid_fit(source[mask], target[mask], weights)
    transformed = source @ transform[:3, :3].T + transform[:3, 3]
    residual = np.linalg.norm(transformed - target, axis=1)
    mask = residual < list(thresholds)[-1]
    return transform, mask


def camera_matrix(camera: dict) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = [[camera[f"t_{row}{col}"] for col in range(3)] for row in range(3)]
    pose[:3, 3] = [camera["t_03"], camera["t_13"], camera["t_23"]]
    return pose


def matrix_to_tum(timestamp_us: int, pose: np.ndarray) -> str:
    quaternion = Quaternion(matrix=pose[:3, :3])
    return (f"{timestamp_us / 1e6:.6f} {pose[0,3]:.9f} {pose[1,3]:.9f} {pose[2,3]:.9f} "
            f"{quaternion.x:.9f} {quaternion.y:.9f} {quaternion.z:.9f} {quaternion.w:.9f}")


@dataclass
class Frame:
    stem: str
    timestamp_us: int
    raw_camera: dict
    corrected_camera: dict
    raw_pose: np.ndarray
    corrected_pose: np.ndarray
    image_path: Path
    depth_path: Path
    keypoints: list | None = None
    descriptors: np.ndarray | None = None
    depth: np.ndarray | None = None


@dataclass
class Session:
    name: str
    root: Path
    ply_path: Path
    alignment: np.ndarray
    frames: list[Frame]
    initial_poses: list[np.ndarray]
    optimized_poses: list[np.ndarray] | None = None
    graph_stats: dict | None = None
    frame_quality: list[dict] | None = None


def load_session(name: str) -> Session:
    root = inside_workspace(DATA / f"{name}_imgs")
    keyframes = root / "keyframes"
    raw_root = inside_workspace(DATA / f"{name}_rawdata")
    mesh_info = read_json(raw_root / "mesh_info.json")
    alignment = np.asarray(mesh_info["alignmentTransform"], dtype=np.float64).reshape(4, 4, order="F")
    raw_cameras = raw_root / "keyframes" / "cameras"
    corrected_cameras = keyframes / "corrected_cameras"
    stems = sorted(path.stem for path in corrected_cameras.glob("*.json"))
    frames: list[Frame] = []
    poses = []
    for stem in stems:
        raw_camera = read_json(raw_cameras / f"{stem}.json")
        corrected_camera = read_json(corrected_cameras / f"{stem}.json")
        raw_pose = YUP_TO_ZUP @ alignment @ camera_matrix(raw_camera) @ CV_TO_ARKIT
        corrected_pose = YUP_TO_ZUP @ alignment @ camera_matrix(corrected_camera) @ CV_TO_ARKIT
        frame = Frame(
            stem=stem,
            timestamp_us=int(stem),
            raw_camera=raw_camera,
            corrected_camera=corrected_camera,
            raw_pose=raw_pose,
            corrected_pose=corrected_pose,
            image_path=inside_workspace(keyframes / "corrected_images" / f"{stem}.jpg"),
            depth_path=inside_workspace(keyframes / "corrected_depth" / f"{stem}.Clean.png"),
        )
        frames.append(frame)
        poses.append(raw_pose)
    return Session(
        name=name,
        root=root,
        ply_path=inside_workspace(DATA / f"{name}_ply" / f"{name}.ply"),
        alignment=alignment,
        frames=frames,
        initial_poses=poses,
    )


def extract_features(session: Session, nfeatures: int = 2200) -> None:
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02, edgeThreshold=12)
    for index, frame in enumerate(session.frames):
        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Cannot read {frame.image_path}")
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        frame.keypoints = keypoints
        frame.descriptors = descriptors
        frame.depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if frame.depth is None:
            raise RuntimeError(f"Cannot read {frame.depth_path}")
        if (index + 1) % 40 == 0 or index + 1 == len(session.frames):
            print(f"[{session.name}] features {index + 1}/{len(session.frames)}", flush=True)


def depth_point(frame: Frame, image_point: tuple[float, float]) -> np.ndarray | None:
    assert frame.depth is not None
    image_width = float(frame.corrected_camera["width"])
    image_height = float(frame.corrected_camera["height"])
    scale_x = frame.depth.shape[1] / image_width
    scale_y = frame.depth.shape[0] / image_height
    u_rgb, v_rgb = image_point
    # Map pixel centres between the 1024x768 colour image and 512x384 depth.
    u = int(round((u_rgb + 0.5) * scale_x - 0.5))
    v = int(round((v_rgb + 0.5) * scale_y - 0.5))
    if not (0 <= u < frame.depth.shape[1] and 0 <= v < frame.depth.shape[0]):
        return None
    y0, y1 = max(0, v - 1), min(frame.depth.shape[0], v + 2)
    x0, x1 = max(0, u - 1), min(frame.depth.shape[1], u + 2)
    values = frame.depth[y0:y1, x0:x1]
    valid = values[(values >= 250) & (values <= 5000)]
    if not len(valid):
        return None
    depth_m = float(np.median(valid)) * 0.001
    camera = frame.corrected_camera
    return np.array([
        (u_rgb - camera["cx"]) * depth_m / camera["fx"],
        (v_rgb - camera["cy"]) * depth_m / camera["fy"],
        depth_m,
    ], dtype=np.float64)


def matched_depth_points(frame_i: Frame, frame_j: Frame, ratio: float = 0.72) -> tuple[np.ndarray, np.ndarray, int]:
    if frame_i.descriptors is None or frame_j.descriptors is None:
        return np.empty((0, 3)), np.empty((0, 3)), 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(frame_i.descriptors, frame_j.descriptors, k=2)
    reverse = matcher.knnMatch(frame_j.descriptors, frame_i.descriptors, k=2)
    reverse_best = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    matches = []
    for pair in forward:
        if len(pair) != 2 or pair[0].distance >= ratio * pair[1].distance:
            continue
        match = pair[0]
        if reverse_best.get(match.trainIdx) == match.queryIdx:
            matches.append(match)
    points_i, points_j = [], []
    assert frame_i.keypoints is not None and frame_j.keypoints is not None
    for match in matches:
        point_i = depth_point(frame_i, frame_i.keypoints[match.queryIdx].pt)
        point_j = depth_point(frame_j, frame_j.keypoints[match.trainIdx].pt)
        if point_i is not None and point_j is not None:
            points_i.append(point_i)
            points_j.append(point_j)
    return np.asarray(points_i), np.asarray(points_j), len(matches)


def estimate_rgbd_edge(frame_i: Frame, frame_j: Frame, prior: np.ndarray,
                       loop: bool = False) -> tuple[np.ndarray | None, dict]:
    source, target, descriptor_matches = matched_depth_points(frame_i, frame_j, ratio=0.70 if loop else 0.73)
    if len(source) < 12:
        return None, {"descriptor_matches": descriptor_matches, "depth_pairs": len(source), "inliers": 0}
    transformed = source @ prior[:3, :3].T + prior[:3, 3]
    prior_residual = np.linalg.norm(transformed - target, axis=1)
    initial_mask = prior_residual < (0.30 if loop else 0.20)
    if initial_mask.sum() < 10:
        return None, {"descriptor_matches": descriptor_matches, "depth_pairs": len(source), "inliers": 0,
                      "prior_median_m": float(np.median(prior_residual))}
    measured, _ = robust_fit(source[initial_mask], target[initial_mask], prior,
                             (0.12, 0.08, 0.055, 0.04))
    residual = np.linalg.norm(source @ measured[:3, :3].T + measured[:3, 3] - target, axis=1)
    inlier_mask = residual < 0.04
    inlier_residual = residual[inlier_mask]
    translation_delta, rotation_delta = transform_delta(measured, prior)
    stats = {
        "descriptor_matches": descriptor_matches,
        "depth_pairs": len(source),
        "inliers": int(inlier_mask.sum()),
        "inlier_fraction": float(inlier_mask.mean()),
        "rmse_m": float(np.sqrt(np.mean(inlier_residual ** 2))) if len(inlier_residual) else 999.0,
        "median_m": float(np.median(inlier_residual)) if len(inlier_residual) else 999.0,
        "p90_m": float(np.percentile(inlier_residual, 90)) if len(inlier_residual) else 999.0,
        "prior_delta_translation_m": translation_delta,
        "prior_delta_rotation_deg": rotation_delta,
    }
    minimum = 24 if loop else 18
    max_translation_delta = 0.30 if loop else 0.16
    max_rotation_delta = 15.0 if loop else 9.0
    accepted = (
        stats["inliers"] >= minimum
        and stats["inlier_fraction"] >= 0.28
        and stats["rmse_m"] <= 0.030
        and stats["p90_m"] <= 0.040
        and translation_delta <= max_translation_delta
        and rotation_delta <= max_rotation_delta
    )
    return (measured if accepted else None), stats


def information_scale(stats: dict, minimum: float, maximum: float) -> float:
    count = float(stats.get("inliers", 0))
    rmse = max(float(stats.get("rmse_m", 0.03)), 0.008)
    return float(np.clip(2.5 * count * 0.02 / rmse, minimum, maximum))


def optimize_session(session: Session, output_dir: Path) -> None:
    n = len(session.frames)
    poses = session.initial_poses
    graph = o3d.pipelines.registration.PoseGraph()
    for pose in poses:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose.copy()))
    reference_node = n
    graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    edge_records = []
    rgbd_support = np.zeros(n, dtype=np.int32)
    loop_support = np.zeros(n, dtype=np.int32)
    sfm_support = np.zeros(n, dtype=bool)

    def add_edge(source: int, target: int, transform: np.ndarray, scale: float,
                 uncertain: bool, kind: str, stats: dict | None = None):
        information = np.eye(6) * scale
        graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            source, target, transform, information, uncertain=uncertain))
        edge_records.append({
            "source": source, "target": target, "kind": kind,
            "information_scale": scale, "uncertain": uncertain,
            "measurement_T_source_to_target": transform.tolist(),
            "information_matrix": information.tolist(),
            **(stats or {}),
        })
        if kind.startswith("rgbd_"):
            rgbd_support[source] += 1
            rgbd_support[target] += 1
            if kind == "rgbd_loop":
                loop_support[source] += 1
                loop_support[target] += 1
        if kind == "strong_polycam_sfm_anchor":
            sfm_support[source] = True

    # Unary raw-VIO priors preserve gravity and global metric continuity.
    for index, pose in enumerate(poses):
        add_edge(index, reference_node, pose, 18.0, False, "raw_vio_absolute_prior")
        corrected = session.frames[index].corrected_pose
        weak = bool(session.frames[index].corrected_camera.get("weakly_connected", True))
        if not weak:
            delta_t, delta_r = transform_delta(corrected, pose)
            if delta_t < 0.12 and delta_r < 5.0:
                add_edge(index, reference_node, corrected, 35.0, False,
                         "strong_polycam_sfm_anchor",
                         {"raw_delta_translation_m": delta_t, "raw_delta_rotation_deg": delta_r})

    accepted_sequential = 0
    failed_sequential = 0
    for index in range(n - 1):
        prior = np.linalg.inv(poses[index + 1]) @ poses[index]
        add_edge(index, index + 1, prior, 120.0, False, "raw_vio_sequential")
        measured, stats = estimate_rgbd_edge(session.frames[index], session.frames[index + 1], prior)
        if measured is not None:
            add_edge(index, index + 1, measured, information_scale(stats, 180.0, 900.0),
                     False, "rgbd_sequential", stats)
            accepted_sequential += 1
        else:
            failed_sequential += 1
        if (index + 1) % 40 == 0:
            print(f"[{session.name}] sequential factors {index + 1}/{n - 1}", flush=True)

    accepted_skip = 0
    for gap in (2, 4):
        for index in range(0, n - gap):
            target = index + gap
            prior = np.linalg.inv(poses[target]) @ poses[index]
            measured, stats = estimate_rgbd_edge(session.frames[index], session.frames[target], prior)
            if measured is None:
                continue
            add_edge(index, target, measured, information_scale(stats, 140.0, 700.0),
                     True, f"rgbd_skip_{gap}", stats)
            accepted_skip += 1

    # VIO only proposes revisits; independent RGB-D geometry admits loops.
    centers = np.asarray([pose[:3, 3] for pose in poses])
    accepted_loops = 0
    tested_loops: set[tuple[int, int]] = set()
    for index in range(0, n, 2):
        candidates = []
        for target in range(index + 15, n):
            distance = float(np.linalg.norm(centers[index] - centers[target]))
            if distance > 0.75:
                continue
            view_dot = float(poses[index][:3, 2] @ poses[target][:3, 2])
            if view_dot < 0.15:
                continue
            candidates.append((distance - 0.10 * view_dot, target))
        for _, target in sorted(candidates)[:2]:
            if (index, target) in tested_loops:
                continue
            tested_loops.add((index, target))
            prior = np.linalg.inv(poses[target]) @ poses[index]
            measured, stats = estimate_rgbd_edge(session.frames[index], session.frames[target], prior, loop=True)
            if measured is None:
                continue
            add_edge(index, target, measured, information_scale(stats, 180.0, 700.0),
                     True, "rgbd_loop", stats)
            accepted_loops += 1

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.5,
        reference_node=reference_node,
    )
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        criteria,
        option,
    )
    optimized = []
    for index in range(n):
        pose = np.asarray(graph.nodes[index].pose).copy()
        u, _, vt = np.linalg.svd(pose[:3, :3])
        pose[:3, :3] = u @ vt
        optimized.append(pose)
    session.optimized_poses = optimized

    optimized_with_reference = optimized + [np.eye(4)]
    residual_translation = []
    residual_rotation = []
    # Open3D physically removes pruned uncertain edges.  Match surviving edges
    # back to their saved records by endpoints and measurement rather than by
    # list position so every record receives an unambiguous retained flag.
    surviving_by_pair: dict[tuple[int, int], list] = {}
    for graph_edge in graph.edges:
        surviving_by_pair.setdefault(
            (int(graph_edge.source_node_id), int(graph_edge.target_node_id)), []).append(graph_edge)
    rgbd_support.fill(0)
    loop_support.fill(0)
    for record in edge_records:
        source = record["source"]
        target = record["target"]
        predicted = (np.linalg.inv(optimized_with_reference[target])
                     @ optimized_with_reference[source])
        measurement = np.asarray(record["measurement_T_source_to_target"], dtype=np.float64)
        residual_t, residual_r = transform_delta(predicted, measurement)
        record["post_optimization_residual_translation_m"] = residual_t
        record["post_optimization_residual_rotation_deg"] = residual_r
        candidates = surviving_by_pair.get((source, target), [])
        graph_edge = None
        if candidates:
            best_index = int(np.argmin([
                np.linalg.norm(np.asarray(edge.transformation) - measurement)
                for edge in candidates
            ]))
            graph_edge = candidates.pop(best_index)
        retained = graph_edge is not None
        record["retained_after_pruning"] = retained
        record["post_optimization_confidence"] = (
            float(graph_edge.confidence) if graph_edge is not None else 0.0)
        if retained and record["kind"].startswith("rgbd_"):
            rgbd_support[source] += 1
            rgbd_support[target] += 1
            if record["kind"] == "rgbd_loop":
                loop_support[source] += 1
                loop_support[target] += 1
        residual_translation.append(residual_t)
        residual_rotation.append(residual_r)

    corrections_t, corrections_r = [], []
    for initial, result in zip(poses, optimized):
        dt, dr = transform_delta(result, initial)
        corrections_t.append(dt)
        corrections_r.append(dr)
    xyz = np.asarray([pose[:3, 3] for pose in optimized])
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    frame_quality = []
    for index, (translation_correction, rotation_correction) in enumerate(
            zip(corrections_t, corrections_r)):
        support = int(rgbd_support[index])
        if support >= 2 and translation_correction <= 0.10 and rotation_correction <= 3.0:
            label = "high"
        elif support >= 1 and translation_correction <= 0.15 and rotation_correction <= 5.0:
            label = "medium"
        else:
            label = "vio_only_or_low_rgbd_support"
        frame_quality.append({
            "label": label,
            "rgbd_factor_degree": support,
            "loop_factor_degree": int(loop_support[index]),
            "strong_polycam_anchor": bool(sfm_support[index]),
            "translation_correction_m": float(translation_correction),
            "rotation_correction_deg": float(rotation_correction),
        })
    session.frame_quality = frame_quality
    quality_counts = {
        label: sum(row["label"] == label for row in frame_quality)
        for label in ("high", "medium", "vio_only_or_low_rgbd_support")
    }
    session.graph_stats = {
        "frames": n,
        "rgbd_sequential_accepted": accepted_sequential,
        "rgbd_sequential_rejected": failed_sequential,
        "rgbd_skip_accepted": accepted_skip,
        "rgbd_loop_accepted": accepted_loops,
        "rgbd_skip_retained": sum(
            row["retained_after_pruning"] and row["kind"].startswith("rgbd_skip_")
            for row in edge_records),
        "rgbd_loop_retained": sum(
            row["retained_after_pruning"] and row["kind"] == "rgbd_loop"
            for row in edge_records),
        "total_graph_edges": len(edge_records),
        "correction_translation_m": {
            "median": float(np.median(corrections_t)), "p95": float(np.percentile(corrections_t, 95)),
            "max": float(np.max(corrections_t)),
        },
        "correction_rotation_deg": {
            "median": float(np.median(corrections_r)), "p95": float(np.percentile(corrections_r, 95)),
            "max": float(np.max(corrections_r)),
        },
        "trajectory_path_length_m": float(steps.sum()),
        "trajectory_max_step_m": float(steps.max(initial=0.0)),
        "trajectory_extent_m": np.ptp(xyz, axis=0).tolist(),
        "frame_quality_counts": quality_counts,
        "post_optimization_edge_residual_translation_m": {
            "median": float(np.median(residual_translation)),
            "p95": float(np.percentile(residual_translation, 95)),
            "max": float(np.max(residual_translation)),
        },
        "post_optimization_edge_residual_rotation_deg": {
            "median": float(np.median(residual_rotation)),
            "p95": float(np.percentile(residual_rotation, 95)),
            "max": float(np.max(residual_rotation)),
        },
    }
    (output_dir / f"{session.name}_pose_graph_edges.json").write_text(json.dumps(edge_records, indent=2))
    print(f"[{session.name}] optimized: seq {accepted_sequential}/{n-1}, "
          f"skip {accepted_skip}, loops {accepted_loops}", flush=True)


def backproject_frame(frame: Frame, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    assert frame.depth is not None
    depth = frame.depth
    rows = np.arange(0, depth.shape[0], stride)
    cols = np.arange(0, depth.shape[1], stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    z = depth[vv, uu].astype(np.float64) * 0.001
    valid = (z >= 0.25) & (z <= 5.0)
    z = z[valid]
    u_depth = uu[valid]
    v_depth = vv[valid]
    camera = frame.corrected_camera
    scale_x = camera["width"] / depth.shape[1]
    scale_y = camera["height"] / depth.shape[0]
    u_rgb = u_depth * scale_x
    v_rgb = v_depth * scale_y
    points = np.column_stack([
        (u_rgb - camera["cx"]) * z / camera["fx"],
        (v_rgb - camera["cy"]) * z / camera["fy"],
        z,
    ])
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    rgb_u = np.clip(np.round(u_rgb).astype(int), 0, image.shape[1] - 1)
    rgb_v = np.clip(np.round(v_rgb).astype(int), 0, image.shape[0] - 1)
    colors = image[rgb_v, rgb_u][:, ::-1].astype(np.float64) / 255.0
    return points, colors


def reconstruct_session_cloud(session: Session, voxel: float = 0.02) -> o3d.geometry.PointCloud:
    assert session.optimized_poses is not None
    chunks_points, chunks_colors = [], []
    for index, (frame, pose) in enumerate(zip(session.frames, session.optimized_poses)):
        points, colors = backproject_frame(frame, stride=4)
        points = points @ pose[:3, :3].T + pose[:3, 3]
        chunks_points.append(points)
        chunks_colors.append(colors)
        if (index + 1) % 50 == 0:
            print(f"[{session.name}] reconstructed {index + 1}/{len(session.frames)}", flush=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(chunks_points, axis=0))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(chunks_colors, axis=0))
    return cloud.voxel_down_sample(voxel)


def cloud_distance_metrics(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> dict:
    distances = np.asarray(source.compute_point_cloud_distance(target))
    return {
        "count": len(distances),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "mean_m": float(np.mean(distances)),
        "fraction_within_5cm": float(np.mean(distances < 0.05)),
        "fraction_within_10cm": float(np.mean(distances < 0.10)),
    }


def validate_local_cloud(session: Session, reconstructed: o3d.geometry.PointCloud) -> dict:
    reference = o3d.io.read_point_cloud(str(session.ply_path)).voxel_down_sample(0.025)
    forward = cloud_distance_metrics(reconstructed, reference)
    reverse = cloud_distance_metrics(reference, reconstructed)
    return {"reconstruction_to_export": forward, "export_to_reconstruction": reverse}


def z_rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return transform


def principal_xy_angle(points: np.ndarray) -> float:
    covariance = np.cov(points[:, :2].T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, np.argmax(values)]
    return math.atan2(float(axis[1]), float(axis[0]))


def robust_center(points: np.ndarray) -> np.ndarray:
    return (np.quantile(points, 0.01, axis=0) + np.quantile(points, 0.99, axis=0)) * 0.5


def prepare_cloud(cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    result = cloud.voxel_down_sample(voxel)
    result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=40))
    return result


def refine_icp(source_full: o3d.geometry.PointCloud, target_full: o3d.geometry.PointCloud,
               initial: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    transform = initial.copy()
    levels = [(0.10, 0.30, 60), (0.06, 0.16, 50), (0.035, 0.09, 40), (0.02, 0.055, 30)]
    history = []
    for voxel, distance, iterations in levels:
        source = prepare_cloud(source_full, voxel)
        target = prepare_cloud(target_full, voxel)
        kernel = o3d.pipelines.registration.TukeyLoss(k=distance * 0.5)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel)
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance, transform, estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations))
        transform = result.transformation
        history.append({"voxel_m": voxel, "max_correspondence_m": distance,
                        "fitness": float(result.fitness), "rmse_m": float(result.inlier_rmse)})
    return transform, history


def gravity_locked_transform(transform: np.ndarray) -> np.ndarray:
    """Project an SE(3) transform onto yaw plus XYZ translation."""
    yaw = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
    result = z_rotation(yaw)
    result[:3, 3] = transform[:3, 3]
    return result


def refine_gravity_locked_icp(source_full: o3d.geometry.PointCloud,
                              target_full: o3d.geometry.PointCloud,
                              initial: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Robust multiscale point-to-plane ICP with exactly four degrees of freedom.

    Both Polycam exports inherit measured gravity from ARKit.  Letting dense ICP
    alter roll and pitch can fit moved furniture at the expense of the room
    frame, so this refinement estimates only yaw and XYZ translation.
    """
    projected = gravity_locked_transform(initial)
    theta = math.atan2(float(projected[1, 0]), float(projected[0, 0]))
    translation = projected[:3, 3].copy()
    levels = [(0.10, 0.30, 50), (0.06, 0.16, 45), (0.035, 0.09, 40), (0.02, 0.055, 35)]
    history = []
    for voxel, max_distance, max_iterations in levels:
        source = prepare_cloud(source_full, voxel)
        target = prepare_cloud(target_full, voxel)
        source_points = np.asarray(source.points)
        source_normals = np.asarray(source.normals)
        target_points = np.asarray(target.points)
        target_normals = np.asarray(target.normals)
        tree = cKDTree(target_points)
        accepted = 0
        robust_rms = float("inf")
        iteration = -1
        for iteration in range(max_iterations):
            rotation = z_rotation(theta)[:3, :3]
            rotated = source_points @ rotation.T
            transformed = rotated + translation
            distances, indices = tree.query(transformed, workers=-1)
            valid = distances < max_distance
            if int(valid.sum()) < 200:
                break

            p = transformed[valid]
            q = target_points[indices[valid]]
            normals = target_normals[indices[valid]]
            rotated_normals = source_normals[valid] @ rotation.T
            normal_agreement = np.abs(np.sum(rotated_normals * normals, axis=1))
            compatible = normal_agreement > math.cos(math.radians(50.0))
            p = p[compatible]
            q = q[compatible]
            normals = normals[compatible]
            distance = distances[valid][compatible]
            rotated_valid = rotated[valid][compatible]
            normal_agreement = normal_agreement[compatible]
            if len(p) < 200:
                break

            residual = np.sum(normals * (p - q), axis=1)
            cutoff = max(0.35 * max_distance, 0.018)
            plane_scale = residual / cutoff
            distance_scale = distance / max_distance
            weights = np.square(np.clip(1.0 - plane_scale * plane_scale, 0.0, 1.0))
            weights *= np.square(np.clip(1.0 - distance_scale * distance_scale, 0.0, 1.0))
            weights *= np.square(normal_agreement)
            useful = weights > 1e-5
            if int(useful.sum()) < 200:
                break

            normals = normals[useful]
            residual = residual[useful]
            weights = weights[useful]
            rotated_valid = rotated_valid[useful]
            yaw_derivative = np.column_stack((
                -rotated_valid[:, 1], rotated_valid[:, 0], np.zeros(len(rotated_valid))))
            jacobian = np.column_stack((np.sum(normals * yaw_derivative, axis=1), normals))
            root_weight = np.sqrt(weights)
            delta, *_ = np.linalg.lstsq(
                jacobian * root_weight[:, None], -residual * root_weight, rcond=None)
            delta[0] = float(np.clip(delta[0], -math.radians(1.5), math.radians(1.5)))
            delta[1:] = np.clip(delta[1:], -0.04, 0.04)
            theta += float(delta[0])
            translation += delta[1:]
            accepted = len(residual)
            robust_rms = float(np.sqrt(np.average(np.square(residual), weights=weights)))
            if abs(delta[0]) < math.radians(0.001) and np.linalg.norm(delta[1:]) < 2e-5:
                break

        result = z_rotation(theta)
        result[:3, 3] = translation
        transformed_cloud = o3d.geometry.PointCloud(source)
        transformed_cloud.transform(result)
        evaluation = o3d.pipelines.registration.evaluate_registration(
            transformed_cloud, target, max_distance, np.eye(4))
        history.append({
            "voxel_m": voxel,
            "max_correspondence_m": max_distance,
            "iterations": iteration + 1,
            "accepted_point_to_plane_rows": accepted,
            "robust_point_to_plane_rms_m": robust_rms,
            "fitness": float(evaluation.fitness),
            "rmse_m": float(evaluation.inlier_rmse),
            "yaw_deg": float(np.degrees(theta)),
            "translation_m": translation.tolist(),
        })
    result = z_rotation(theta)
    result[:3, 3] = translation
    return result, history


def registration_score(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                       transform: np.ndarray) -> dict:
    source_eval = source.voxel_down_sample(0.04)
    target_eval = target.voxel_down_sample(0.04)
    transformed = o3d.geometry.PointCloud(source_eval)
    transformed.transform(transform)
    forward = np.asarray(transformed.compute_point_cloud_distance(target_eval))
    reverse = np.asarray(target_eval.compute_point_cloud_distance(transformed))
    symmetric_overlap_5 = 0.5 * (np.mean(forward < 0.05) + np.mean(reverse < 0.05))
    symmetric_overlap_10 = 0.5 * (np.mean(forward < 0.10) + np.mean(reverse < 0.10))
    trimmed = np.concatenate([
        np.sort(forward)[:int(0.8 * len(forward))],
        np.sort(reverse)[:int(0.8 * len(reverse))],
    ])
    score = float(symmetric_overlap_5 + 0.5 * symmetric_overlap_10 - 2.0 * np.mean(trimmed))
    return {
        "score": score,
        "symmetric_overlap_5cm": float(symmetric_overlap_5),
        "symmetric_overlap_10cm": float(symmetric_overlap_10),
        "trimmed_mean_m": float(np.mean(trimmed)),
        "forward_median_m": float(np.median(forward)),
        "forward_p90_m": float(np.percentile(forward, 90)),
        "reverse_median_m": float(np.median(reverse)),
        "reverse_p90_m": float(np.percentile(reverse, 90)),
    }


def register_room_clouds(session1: Session, session2: Session) -> tuple[np.ndarray, dict]:
    cloud1 = o3d.io.read_point_cloud(str(session1.ply_path))
    cloud2 = o3d.io.read_point_cloud(str(session2.ply_path))
    sample1 = np.asarray(cloud1.voxel_down_sample(0.08).points)
    sample2 = np.asarray(cloud2.voxel_down_sample(0.08).points)
    angle1 = principal_xy_angle(sample1)
    angle2 = principal_xy_angle(sample2)
    base_yaw = angle1 - angle2
    candidates = []
    candidate_id = 0
    for ambiguity in (0.0, math.pi):
        for perturbation_deg in (-8.0, -4.0, 0.0, 4.0, 8.0):
            yaw = base_yaw + ambiguity + math.radians(perturbation_deg)
            initial = z_rotation(yaw)
            rotated_center = initial[:3, :3] @ robust_center(sample2)
            initial[:3, 3] = robust_center(sample1) - rotated_center
            refined, history = refine_icp(cloud2, cloud1, initial)
            # Gravity must remain aligned; reject any accidental tilt.
            gravity_dot = float(refined[:3, 2] @ np.array([0.0, 0.0, 1.0]))
            if gravity_dot < 0.97:
                continue
            metrics = registration_score(cloud2, cloud1, refined)
            candidates.append({
                "id": candidate_id,
                "initial_yaw_deg": float(np.degrees(yaw)),
                "transform": refined,
                "icp_levels": history,
                **metrics,
            })
            print(f"[cross] candidate {candidate_id}: yaw={np.degrees(yaw):.1f} "
                  f"overlap5={metrics['symmetric_overlap_5cm']:.3f} score={metrics['score']:.3f}", flush=True)
            candidate_id += 1
    if not candidates:
        raise RuntimeError("No gravity-consistent inter-session registration candidate")
    best = max(candidates, key=lambda row: row["score"])
    report_candidates = []
    for row in candidates:
        report_candidates.append({key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                  for key, value in row.items()})
    return best["transform"], {
        "pca_angle_scan1_deg": float(np.degrees(angle1)),
        "pca_angle_scan2_deg": float(np.degrees(angle2)),
        "candidates": report_candidates,
        "selected_candidate_id": best["id"],
        "selected_geometric_metrics": {key: best[key] for key in (
            "score", "symmetric_overlap_5cm", "symmetric_overlap_10cm", "trimmed_mean_m",
            "forward_median_m", "forward_p90_m", "reverse_median_m", "reverse_p90_m")},
    }


def cross_visual_refinement(session1: Session, session2: Session,
                            geometric_transform: np.ndarray
                            ) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    assert session1.optimized_poses is not None and session2.optimized_poses is not None
    poses1 = session1.optimized_poses
    poses2 = session2.optimized_poses
    centers1 = np.asarray([pose[:3, 3] for pose in poses1])
    candidate_pairs = []
    for index2, pose2 in enumerate(poses2):
        center2 = (geometric_transform @ np.r_[pose2[:3, 3], 1.0])[:3]
        distances = np.linalg.norm(centers1 - center2, axis=1)
        choices = []
        forward2 = geometric_transform[:3, :3] @ pose2[:3, 2]
        for index1 in np.argsort(distances)[:12]:
            view_dot = float(poses1[index1][:3, 2] @ forward2)
            if distances[index1] < 1.0 and view_dot > 0.0:
                choices.append((float(distances[index1] - 0.15 * view_dot), int(index1)))
        for _, index1 in sorted(choices)[:2]:
            candidate_pairs.append((index1, index2))

    verified = []
    all_source, all_target, all_pair_ids = [], [], []
    for pair_id, (index1, index2) in enumerate(candidate_pairs):
        frame1, frame2 = session1.frames[index1], session2.frames[index2]
        points1_camera, points2_camera, descriptor_matches = matched_depth_points(frame1, frame2, ratio=0.70)
        if len(points1_camera) < 12:
            continue
        world1 = points1_camera @ poses1[index1][:3, :3].T + poses1[index1][:3, 3]
        world2 = points2_camera @ poses2[index2][:3, :3].T + poses2[index2][:3, 3]
        transformed2 = world2 @ geometric_transform[:3, :3].T + geometric_transform[:3, 3]
        residual = np.linalg.norm(transformed2 - world1, axis=1)
        mask = residual < 0.12
        if mask.sum() < 12 or float(np.median(residual[mask])) > 0.065:
            continue
        # Limit any one image pair from dominating the rigid fit.
        accepted_indices = np.flatnonzero(mask)
        if len(accepted_indices) > 100:
            accepted_indices = accepted_indices[np.linspace(0, len(accepted_indices) - 1, 100).astype(int)]
        all_source.append(world2[accepted_indices])
        all_target.append(world1[accepted_indices])
        all_pair_ids.extend([len(verified)] * len(accepted_indices))
        verified.append({
            "frame1": frame1.stem, "frame2": frame2.stem,
            "descriptor_matches": descriptor_matches,
            "depth_pairs": len(points1_camera), "geometric_inliers_12cm": int(mask.sum()),
            "median_residual_m": float(np.median(residual[mask])),
        })
    if not all_source:
        empty = np.empty((0, 3), dtype=np.float64)
        return geometric_transform, {
            "candidate_pairs": len(candidate_pairs), "verified_pairs": 0,
            "accepted": False, "reason": "no independently verified RGB-D frame pairs",
            "scope": "transform-conditioned cross-view consistency check; not external ground truth",
        }, {"source": empty, "target": empty, "holdout": np.empty(0, dtype=bool)}
    source = np.concatenate(all_source, axis=0)
    target = np.concatenate(all_target, axis=0)
    pair_ids = np.asarray(all_pair_ids)
    train = (pair_ids % 2) == 0
    holdout = ~train
    visual_transform, train_mask = robust_fit(source[train], target[train], geometric_transform,
                                               (0.12, 0.09, 0.065, 0.045))
    visual_delta_t, visual_delta_r = transform_delta(visual_transform, geometric_transform)
    transformed_all = source @ visual_transform[:3, :3].T + visual_transform[:3, 3]
    residual_all = np.linalg.norm(transformed_all - target, axis=1)
    holdout_residual = residual_all[holdout] if holdout.any() else residual_all
    accepted = (
        len(verified) >= 4
        and train_mask.sum() >= 60
        and visual_delta_t < 0.20
        and visual_delta_r < 7.0
        and float(np.median(holdout_residual)) < 0.08
    )
    chosen = visual_transform if accepted else geometric_transform
    return chosen, {
        "scope": (
            "transform-conditioned RGB-D cross-view consistency check; candidate pairs and "
            "the 12 cm admission gate use the geometric transform, so this is not external ground truth"
        ),
        "candidate_pairs": len(candidate_pairs),
        "verified_pairs": len(verified),
        "verified_pair_details": verified,
        "correspondences": len(source),
        "train_inliers_4_5cm": int(train_mask.sum()),
        "holdout_correspondences": int(holdout.sum()),
        "holdout_median_m": float(np.median(holdout_residual)),
        "holdout_p90_m": float(np.percentile(holdout_residual, 90)),
        "visual_vs_geometry_translation_m": visual_delta_t,
        "visual_vs_geometry_rotation_deg": visual_delta_r,
        "accepted": accepted,
        "visual_transform": visual_transform.tolist(),
    }, {"source": source, "target": target, "holdout": holdout}


def correspondence_residuals(evidence: dict[str, np.ndarray], transform: np.ndarray) -> dict:
    """Summarize independent RGB-D correspondences under a supplied transform."""
    source = evidence["source"]
    target = evidence["target"]
    holdout = evidence["holdout"]
    if len(source) == 0:
        return {"count": 0}
    transformed = source @ transform[:3, :3].T + transform[:3, 3]
    residual = np.linalg.norm(transformed - target, axis=1)
    selected = residual[holdout] if holdout.any() else residual
    return {
        "count": int(len(selected)),
        "median_m": float(np.median(selected)),
        "p90_m": float(np.percentile(selected, 90)),
        "fraction_within_5cm": float(np.mean(selected < 0.05)),
        "fraction_within_10cm": float(np.mean(selected < 0.10)),
    }


def export_aligned_poses(session: Session, common_from_local: np.ndarray, output_dir: Path) -> None:
    assert session.optimized_poses is not None
    assert session.frame_quality is not None
    pose_dir = output_dir / "aligned_frame_poses" / session.name
    pose_dir.mkdir(parents=True, exist_ok=True)
    tum_lines = []
    jsonl = []
    for frame, local_pose, quality in zip(
            session.frames, session.optimized_poses, session.frame_quality):
        pose = common_from_local @ local_pose
        record = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "session": session.name,
            "frame": frame.stem,
            "timestamp_us": frame.timestamp_us,
            "coordinate_frame": f"{SESSION_NAMES[0]} PLY frame; Z-up; OpenCV camera axes",
            "T_camera_to_common": pose.tolist(),
            "pose_support": quality,
            "intrinsics": {key: frame.corrected_camera[key] for key in ("fx", "fy", "cx", "cy", "width", "height")},
            "rgb": context.data_relative(frame.image_path),
            "depth": context.data_relative(frame.depth_path),
        }
        (pose_dir / f"{frame.stem}.json").write_text(json.dumps(record, indent=2))
        jsonl.append(json.dumps(record, separators=(",", ":")))
        tum_lines.append(matrix_to_tum(frame.timestamp_us, pose))
    (output_dir / f"{session.name}_aligned_poses.jsonl").write_text("\n".join(jsonl) + "\n")
    (output_dir / f"{session.name}_aligned_trajectory_tum.txt").write_text("\n".join(tum_lines) + "\n")


def save_shared_cloud(session1: Session, session2: Session, transform21: np.ndarray,
                      output_dir: Path) -> dict:
    cloud1 = o3d.io.read_point_cloud(str(session1.ply_path))
    cloud2 = o3d.io.read_point_cloud(str(session2.ply_path))
    aligned2 = o3d.geometry.PointCloud(cloud2)
    aligned2.transform(transform21)
    aligned2_path = output_dir / f"{session2.name}_aligned_to_{session1.name}.ply"
    o3d.io.write_point_cloud(str(aligned2_path), aligned2, write_ascii=False, compressed=True)
    merged = cloud1.voxel_down_sample(0.015) + aligned2.voxel_down_sample(0.015)
    merged = merged.voxel_down_sample(0.012)
    shared_path = output_dir / "shared_aligned_room_pointcloud.ply"
    o3d.io.write_point_cloud(str(shared_path), merged, write_ascii=False, compressed=True)
    return {
        "aligned_scan2_points": len(aligned2.points),
        "shared_downsampled_points": len(merged.points),
        "aligned_scan2_ply": outputs_relative(aligned2_path),
        "shared_pointcloud_ply": outputs_relative(shared_path),
    }


def main() -> None:
    output_dir = inside_workspace(DEFAULT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = [load_session(name) for name in SESSION_NAMES]
    for session in sessions:
        print(f"Loading and optimizing {session.name} ({len(session.frames)} frames)", flush=True)
        extract_features(session)
        optimize_session(session, output_dir)

    local_clouds = []
    local_validation = {}
    for session in sessions:
        cloud = reconstruct_session_cloud(session)
        local_clouds.append(cloud)
        cloud_path = output_dir / f"{session.name}_optimized_rgbd_cloud.ply"
        o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False, compressed=True)
        local_validation[session.name] = validate_local_cloud(session, cloud)

    geometric_transform, registration_report = register_room_clouds(sessions[0], sessions[1])
    visual_transform, visual_report, visual_evidence = cross_visual_refinement(
        sessions[0], sessions[1], geometric_transform)

    # Final geometry refinement starts from the independently supported visual
    # transform when available, then checks that it improves the symmetric room score.
    cloud1_full = o3d.io.read_point_cloud(str(sessions[0].ply_path))
    cloud2_full = o3d.io.read_point_cloud(str(sessions[1].ply_path))
    unconstrained_transform, final_icp_history = refine_icp(
        cloud2_full, cloud1_full, visual_transform)
    score_before = registration_score(cloud2_full, cloud1_full, geometric_transform)
    unconstrained_score = registration_score(cloud2_full, cloud1_full, unconstrained_transform)
    if unconstrained_score["score"] + 0.01 < score_before["score"]:
        unconstrained_transform = geometric_transform
        unconstrained_score = score_before
        unconstrained_selection = "geometric candidate retained; unconstrained refinement degraded score"
    else:
        geometry_delta_t, geometry_delta_r = transform_delta(
            unconstrained_transform, geometric_transform)
        if geometry_delta_t < 1e-4 and geometry_delta_r < 0.01:
            unconstrained_selection = (
                "geometric multiscale ICP fixed point, cross-checked by RGB-D correspondences")
        else:
            unconstrained_selection = (
                "visual-initialized multiscale robust ICP, cross-checked by dense geometry")

    # ARKit provides gravity independently in both recordings.  Refine only
    # yaw and translation so movable objects cannot pull roll/pitch away from
    # that measurement.  Prefer the constrained estimate whenever its dense
    # score is within 0.003 of the unconstrained result.
    gravity_transform, gravity_history = refine_gravity_locked_icp(
        cloud2_full, cloud1_full, unconstrained_transform)
    gravity_score = registration_score(cloud2_full, cloud1_full, gravity_transform)
    if gravity_score["score"] + 0.003 >= unconstrained_score["score"]:
        final_transform = gravity_transform
        score_after = gravity_score
        final_selection = (
            "gravity-locked (yaw + XYZ) robust multiscale ICP; "
            f"seed: {unconstrained_selection}")
    else:
        final_transform = unconstrained_transform
        score_after = unconstrained_score
        final_selection = (
            "six-DoF robust multiscale ICP retained because gravity lock "
            "reduced the dense registration score by more than 0.003")

    visual_report["holdout_under_geometric_transform"] = correspondence_residuals(
        visual_evidence, geometric_transform)
    visual_report["holdout_under_visual_fit"] = correspondence_residuals(
        visual_evidence, visual_transform)
    visual_report["holdout_under_unconstrained_transform"] = correspondence_residuals(
        visual_evidence, unconstrained_transform)
    visual_report["holdout_under_final_transform"] = correspondence_residuals(
        visual_evidence, final_transform)

    export_aligned_poses(sessions[0], np.eye(4), output_dir)
    export_aligned_poses(sessions[1], final_transform, output_dir)
    cloud_outputs = save_shared_cloud(sessions[0], sessions[1], final_transform, output_dir)

    transform_path = output_dir / f"{sessions[1].name}_to_{sessions[0].name}_transform.json"
    transform_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": f"{sessions[1].name} PLY frame",
        "target": f"{sessions[0].name} PLY frame",
        "coordinate_convention": "Z-up room; OpenCV camera axes; metres",
        "T_source_to_target": final_transform.tolist(),
        "inverse_T_target_to_source": np.linalg.inv(final_transform).tolist(),
        "selection": final_selection,
    }
    transform_path.write_text(json.dumps(transform_payload, indent=2))

    final_rotation = final_transform[:3, :3]
    visual_final = visual_report["holdout_under_final_transform"]
    quality_checks = [
        {
            "name": "proper_rigid_transform",
            "value": float(np.linalg.norm(final_rotation.T @ final_rotation - np.eye(3))),
            "requirement": "orthogonality_error_frobenius <= 1e-6 and det(R) > 0",
            "passed": bool(
                np.linalg.norm(final_rotation.T @ final_rotation - np.eye(3)) <= 1e-6
                and np.linalg.det(final_rotation) > 0.999999),
        },
        {
            "name": "gravity_locked",
            "value_deg": float(np.degrees(np.arccos(np.clip(final_rotation[2, 2], -1.0, 1.0)))),
            "requirement": "roll_pitch_tilt <= 0.05 deg",
            "passed": bool(final_rotation[2, 2] >= math.cos(math.radians(0.05))),
        },
        {
            "name": "same_source_local_rgbd_consistency",
            "worst_forward_median_m": float(max(
                row["reconstruction_to_export"]["median_m"] for row in local_validation.values())),
            "requirement": "worst session forward median <= 0.025 m",
            "passed": bool(max(
                row["reconstruction_to_export"]["median_m"]
                for row in local_validation.values()) <= 0.025),
        },
        {
            "name": "cross_view_consistency",
            "holdout_count": int(visual_final["count"]),
            "holdout_p90_m": float(visual_final["p90_m"]),
            "requirement": "at least 500 admitted holdout correspondences and p90 <= 0.10 m",
            "passed": bool(visual_final["count"] >= 500 and visual_final["p90_m"] <= 0.10),
        },
        {
            "name": "cross_cloud_overlap",
            "symmetric_overlap_10cm": float(score_after["symmetric_overlap_10cm"]),
            "requirement": "symmetric overlap within 10 cm >= 0.65",
            "passed": bool(score_after["symmetric_overlap_10cm"] >= 0.65),
        },
    ]
    pose_directories = [
        outputs_relative(output_dir / "aligned_frame_poses" / session.name)
        for session in sessions
    ]

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pipeline": "offline RGB-D + ARKit/VIO pose graph and gravity-locked room registration",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "data_folder": str(DATA),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "open3d": o3d.__version__,
        },
        "reproduction_command": "navdelta align",
        "recordings": {"before": sessions[0].name, "after": sessions[1].name},
        "common_coordinate_frame": f"{sessions[0].name} exported PLY frame (Z-up, metres)",
        "pose_convention": "T_camera_to_common; OpenCV camera axes (right, down, forward)",
        "input_frames": {session.name: len(session.frames) for session in sessions},
        "per_session_pose_graph": {session.name: session.graph_stats for session in sessions},
        "local_rgbd_to_export_validation": local_validation,
        "inter_session_geometric_registration": registration_report,
        "inter_session_visual_validation": visual_report,
        "final_icp_history": final_icp_history,
        "unconstrained_registration_metrics": unconstrained_score,
        "gravity_locked_icp_history": gravity_history,
        "gravity_locked_registration_metrics": gravity_score,
        "final_selection": final_selection,
        "final_registration_metrics": score_after,
        "final_transform": final_transform.tolist(),
        "internal_quality_gate": {
            "passed": bool(all(check["passed"] for check in quality_checks)),
            "checks": quality_checks,
            "claim_boundary": (
                "These checks establish internal consistency on the supplied data. "
                "Absolute accuracy is not certified because no surveyed ground truth was supplied."
            ),
        },
        "outputs": {
            **cloud_outputs,
            "transform_json": outputs_relative(transform_path),
            "aligned_pose_directories": pose_directories,
        },
    }
    (output_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "status": report["status"],
        "transform": final_transform.tolist(),
        "final_registration_metrics": score_after,
        "visual_pairs": visual_report.get("verified_pairs", 0),
        "outputs": report["outputs"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
