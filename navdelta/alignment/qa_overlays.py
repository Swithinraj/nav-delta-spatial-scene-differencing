#!/usr/bin/env python3
"""Create non-destructive QA overlays for the two aligned room scans.

The script reads the original "before" PLY, the already transformed "after"
PLY, and the two aligned TUM trajectories. It only writes visualization
artifacts under ``<outputs>/alignment/qa``; it never changes alignment inputs or
results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from navdelta import context
from navdelta.context import ALIGNMENT, DATA

BEFORE, AFTER = context.SESSIONS

PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}

SCAN1_COLOR = "#1976B9"
SCAN2_COLOR = "#E6862A"
TRAJ1_COLOR = "#062F5B"
TRAJ2_COLOR = "#8E2C14"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    output_root = ALIGNMENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan1-ply",
        type=Path,
        default=DATA / f"{BEFORE}_ply" / f"{BEFORE}.ply",
    )
    parser.add_argument(
        "--scan2-ply",
        type=Path,
        default=output_root / f"{AFTER}_aligned_to_{BEFORE}.ply",
    )
    parser.add_argument(
        "--trajectory1",
        type=Path,
        default=output_root / f"{BEFORE}_aligned_trajectory_tum.txt",
    )
    parser.add_argument(
        "--trajectory2",
        type=Path,
        default=output_root / f"{AFTER}_aligned_trajectory_tum.txt",
    )
    parser.add_argument(
        "--alignment-report",
        type=Path,
        default=output_root / "alignment_report.json",
    )
    parser.add_argument("--output-dir", type=Path, default=output_root / "qa")
    parser.add_argument(
        "--max-points",
        type=int,
        default=350_000,
        help="Maximum deterministic sample per point cloud for each plot.",
    )
    parser.add_argument("--seed", type=int, default=20261009)
    return parser.parse_args(argv)


def _read_ply_vertex_layout(path: Path) -> tuple[int, int, np.dtype]:
    """Return (header bytes, vertex count, structured vertex dtype)."""
    lines: list[str] = []
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Missing end_header in {path}")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"Non-ASCII PLY header in {path}") from exc
            lines.append(line)
            if line == "end_header":
                header_bytes = stream.tell()
                break

    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"Only binary_little_endian PLY is supported: {path}")

    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    current_element: str | None = None
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "element" and len(fields) == 3:
            current_element = fields[1]
            if current_element == "vertex":
                vertex_count = int(fields[2])
        elif fields[0] == "property" and current_element == "vertex":
            if len(fields) != 3 or fields[1] == "list":
                raise ValueError(f"Unsupported vertex property in {path}: {line}")
            scalar_type, name = fields[1], fields[2]
            if scalar_type not in PLY_SCALAR_DTYPES:
                raise ValueError(f"Unsupported PLY scalar type {scalar_type!r} in {path}")
            properties.append((name, PLY_SCALAR_DTYPES[scalar_type]))

    if vertex_count is None:
        raise ValueError(f"No vertex element in {path}")
    names = {name for name, _ in properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY lacks xyz properties: {path}")
    return header_bytes, vertex_count, np.dtype(properties)


def sample_ply_xyz(path: Path, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    header_bytes, count, dtype = _read_ply_vertex_layout(path)
    vertices = np.memmap(path, mode="r", dtype=dtype, offset=header_bytes, shape=(count,))
    if count > max_points:
        indices = np.sort(rng.choice(count, size=max_points, replace=False))
        selected = vertices[indices]
    else:
        selected = vertices
    xyz = np.column_stack((selected["x"], selected["y"], selected["z"])).astype(
        np.float64, copy=False
    )
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], count


def load_tum_positions(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] < 4:
        raise ValueError(f"Expected TUM rows with timestamp and xyz: {path}")
    xyz = values[:, 1:4]
    return xyz[np.isfinite(xyz).all(axis=1)]


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("final_registration_metrics", {})
    return {
        key: float(metrics[key])
        for key in (
            "symmetric_overlap_5cm",
            "symmetric_overlap_10cm",
            "trimmed_mean_m",
            "forward_median_m",
            "reverse_median_m",
        )
        if key in metrics
    }


def metric_subtitle(metrics: dict[str, float]) -> str:
    if not metrics:
        return f"Shared frame: {BEFORE} PLY coordinates (Z-up, metres)"
    return (
        f"Reported symmetric overlap: {100 * metrics.get('symmetric_overlap_5cm', float('nan')):.1f}% @ 5 cm, "
        f"{100 * metrics.get('symmetric_overlap_10cm', float('nan')):.1f}% @ 10 cm; "
        f"trimmed mean distance: {100 * metrics.get('trimmed_mean_m', float('nan')):.1f} cm"
    )


def padded_limits(arrays: Iterable[np.ndarray], columns: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
    valid = [a[:, columns] for a in arrays if a.size]
    values = np.vstack(valid)
    lower = np.nanpercentile(values, 0.05, axis=0)
    upper = np.nanpercentile(values, 99.95, axis=0)
    span = np.maximum(upper - lower, 0.1)
    margin = 0.045 * span
    return (float(lower[0] - margin[0]), float(upper[0] + margin[0])), (
        float(lower[1] - margin[1]),
        float(upper[1] + margin[1]),
    )


def legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=SCAN1_COLOR, label="Scan 1 geometry"),
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=SCAN2_COLOR, label="Aligned scan 2 geometry"),
        Line2D([], [], linewidth=2.2, color=TRAJ1_COLOR, label="Scan 1 camera path"),
        Line2D([], [], linewidth=2.2, color=TRAJ2_COLOR, label="Aligned scan 2 camera path"),
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor="white", markeredgecolor="#202020", label="Trajectory start"),
        Line2D([], [], marker="X", linestyle="none", markersize=7, color="#202020", label="Trajectory end"),
    ]


def draw_projection(
    ax: plt.Axes,
    scan1: np.ndarray,
    scan2: np.ndarray,
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    columns: tuple[int, int],
    labels: tuple[str, str],
    title: str,
    point_size: float = 0.24,
) -> None:
    i, j = columns
    # Low-opacity dense points make coincident structure visibly blend while
    # preserving scan-specific colour along unmatched surfaces.
    ax.scatter(
        scan1[:, i], scan1[:, j], s=point_size, c=SCAN1_COLOR, alpha=0.13,
        linewidths=0, rasterized=True, zorder=1,
    )
    ax.scatter(
        scan2[:, i], scan2[:, j], s=point_size, c=SCAN2_COLOR, alpha=0.13,
        linewidths=0, rasterized=True, zorder=2,
    )
    ax.plot(
        trajectory1[:, i], trajectory1[:, j], color=TRAJ1_COLOR, linewidth=1.55,
        alpha=0.96, zorder=5,
    )
    ax.plot(
        trajectory2[:, i], trajectory2[:, j], color=TRAJ2_COLOR, linewidth=1.55,
        alpha=0.96, zorder=6,
    )
    for trajectory, color in ((trajectory1, TRAJ1_COLOR), (trajectory2, TRAJ2_COLOR)):
        ax.scatter(
            trajectory[0, i], trajectory[0, j], s=50, marker="o",
            facecolor="white", edgecolor=color, linewidth=1.8, zorder=8,
        )
        ax.scatter(
            trajectory[-1, i], trajectory[-1, j], s=58, marker="X",
            c=color, edgecolor="white", linewidth=0.8, zorder=8,
        )
    xlim, ylim = padded_limits((scan1, scan2, trajectory1, trajectory2), columns)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title, fontweight="bold", pad=9)
    ax.grid(True, color="#D7DEE5", linewidth=0.65, alpha=0.72)
    ax.set_facecolor("#FBFCFD")


def save_single_projection(
    output: Path,
    scan1: np.ndarray,
    scan2: np.ndarray,
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    columns: tuple[int, int],
    labels: tuple[str, str],
    title: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 9.2), constrained_layout=False)
    draw_projection(ax, scan1, scan2, trajectory1, trajectory2, columns, labels, title)
    fig.suptitle(subtitle, fontsize=10.5, y=0.965, color="#36454F")
    ax.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.085),
        ncol=3, frameon=False, fontsize=9.5,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.17)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def save_overview(
    output: Path,
    scan1: np.ndarray,
    scan2: np.ndarray,
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    subtitle: str,
    counts: tuple[int, int],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.3))
    specs = (
        ((0, 1), ("X (m)", "Y (m)"), "Top-down XY"),
        ((0, 2), ("X (m)", "Z elevation (m)"), "Elevation XZ"),
        ((1, 2), ("Y (m)", "Z elevation (m)"), "Elevation YZ"),
    )
    for ax, (columns, labels, title) in zip(axes, specs):
        draw_projection(
            ax, scan1, scan2, trajectory1, trajectory2, columns, labels, title,
            point_size=0.18,
        )
    fig.suptitle("Aligned room scans and optimized camera trajectories", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.915, subtitle, ha="center", fontsize=10.5, color="#36454F")
    fig.text(
        0.5, 0.037,
        f"Full point counts: scan 1 = {counts[0]:,}; aligned scan 2 = {counts[1]:,}. "
        "Plots use deterministic samples; axes use metres.",
        ha="center", fontsize=9.3, color="#4B5963",
    )
    fig.legend(
        handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.065),
        ncol=6, frameon=False, fontsize=9.2,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.17, wspace=0.25)
    fig.savefig(output, dpi=210, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args([] if argv is None else argv)
    for source in (args.scan1_ply, args.scan2_ply, args.trajectory1, args.trajectory2):
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.max_points < 1:
        raise ValueError("--max-points must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng1 = np.random.default_rng(args.seed)
    rng2 = np.random.default_rng(args.seed + 1)
    scan1, count1 = sample_ply_xyz(args.scan1_ply, args.max_points, rng1)
    scan2, count2 = sample_ply_xyz(args.scan2_ply, args.max_points, rng2)
    trajectory1 = load_tum_positions(args.trajectory1)
    trajectory2 = load_tum_positions(args.trajectory2)
    metrics = load_metrics(args.alignment_report)
    subtitle = metric_subtitle(metrics)

    products = {
        "topdown_xy": args.output_dir / "alignment_topdown_xy.png",
        "elevation_xz": args.output_dir / "alignment_elevation_xz.png",
        "elevation_yz": args.output_dir / "alignment_elevation_yz.png",
        "overview": args.output_dir / "alignment_overview.png",
    }
    save_single_projection(
        products["topdown_xy"], scan1, scan2, trajectory1, trajectory2,
        (0, 1), ("X (m)", "Y (m)"), "Top-down alignment overlay (XY)", subtitle,
    )
    save_single_projection(
        products["elevation_xz"], scan1, scan2, trajectory1, trajectory2,
        (0, 2), ("X (m)", "Z elevation (m)"), "Side alignment overlay (XZ)", subtitle,
    )
    save_single_projection(
        products["elevation_yz"], scan1, scan2, trajectory1, trajectory2,
        (1, 2), ("Y (m)", "Z elevation (m)"), "Side alignment overlay (YZ)", subtitle,
    )
    save_overview(
        products["overview"], scan1, scan2, trajectory1, trajectory2,
        subtitle, (count1, count2),
    )

    metadata = {
        "purpose": "read-only QA visualization of the current alignment",
        "coordinate_frame": f"{BEFORE} exported PLY frame; Z-up; metres",
        "inputs": {
            "scan1_ply": str(args.scan1_ply.resolve()),
            "aligned_scan2_ply": str(args.scan2_ply.resolve()),
            "scan1_trajectory": str(args.trajectory1.resolve()),
            "aligned_scan2_trajectory": str(args.trajectory2.resolve()),
        },
        "full_point_counts": {"scan1": count1, "scan2": count2},
        "plotted_point_counts": {"scan1": len(scan1), "scan2": len(scan2)},
        "trajectory_pose_counts": {"scan1": len(trajectory1), "scan2": len(trajectory2)},
        "reported_registration_metrics": metrics,
        "outputs": {name: str(path.resolve()) for name, path in products.items()},
    }
    metadata_path = args.output_dir / "alignment_qa_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": metadata["outputs"], "metadata": str(metadata_path.resolve())}, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
