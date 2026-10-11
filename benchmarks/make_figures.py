"""Figures for the README and docs/benchmark.md, from the tracked results.

  python benchmarks/make_figures.py  ->  docs/images/benchmark_f1.png, benchmark_pr.png, room_changes.png

  benchmark_f1   F1 per ChangeSim pair and held-out micro F1, the four approaches (tuned profile)
  benchmark_pr   held-out precision vs recall, with F1 iso-lines
  room_changes   top view of our room per approach (from results/<approach>/), with every
                 reported change and, in the shared RGB frame, the three staged changes
"""
from __future__ import annotations

import base64
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import RESULTS, ROOT

OUT = ROOT / "docs" / "images"
APPROACHES = {"voxel_diff": "1 · voxel_diff", "scene_graph": "2 · scene_graph",
              "loop_outliers": "3 · loop_outliers", "rgbd_fusion": "4 · rgbd_fusion"}
COLOURS = {"voxel_diff": "#8c8c8c", "scene_graph": "#d9a441", "loop_outliers": "#4a90c2", "rgbd_fusion": "#c2453a"}
PAIRS = ["W6_S0", "W6_S1", "W7_S0", "W8_S0", "W9_S0"]
LABEL_OFFSET = {"voxel_diff": (12, -6), "scene_graph": (6, -20), "loop_outliers": (10, 0), "rgbd_fusion": (10, 0)}
HELD_OUT = ["W7_S0", "W8_S0", "W9_S0"]
STAGED = {   # old -> new place in the shared RGB reconstruction's frame (see room_score.py)
    "luggage": ((-3.59, -0.90), (-4.04, -1.52)),
    "office chair": ((-2.75, -1.87), (-3.27, -1.60)),
    "tower fan": ((-2.24, -0.84), (-1.95, -1.52)),
}


def tuned_runs() -> dict:
    runs = {}
    for line in (RESULTS / "final.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["profile"] == "tuned":
            runs[(r["pair"], r["approach"])] = r
    return runs


def micro(records: list[dict]) -> tuple[float, float, float]:
    claims = sum(r["metrics"]["claims"] for r in records)
    tp = sum(r["metrics"]["true_positive_claims"] for r in records)
    gt = sum(r["metrics"]["ground_truth"]["changed"] for r in records)
    found = sum(round(r["metrics"]["recall"] * r["metrics"]["ground_truth"]["changed"]) for r in records)
    p, rc = tp / max(1, claims), found / max(1, gt)
    return p, rc, 2 * p * rc / max(1e-9, p + rc)


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#333333", labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def benchmark_f1(runs: dict) -> None:
    groups = PAIRS + ["held-out\n(micro)"]
    x = np.arange(len(groups)) + np.array([0, 0, 0.35, 0.35, 0.35, 0.8])
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=150)
    ax.axvspan(-0.5, 1.5, color="#f3f3f3", zorder=0)
    ax.text(0.5, 0.79, "tuning pairs", ha="center", fontsize=9, color="#666666")
    ax.text(x[3], 0.79, "held-out pairs", ha="center", fontsize=9, color="#666666")
    for k, (a, name) in enumerate(APPROACHES.items()):
        values = [runs[(p, a)]["metrics"]["f1"] for p in PAIRS]
        values.append(micro([runs[(p, a)] for p in HELD_OUT])[2])
        bars = ax.bar(x + (k - 1.5) * width, values, width, color=COLOURS[a], label=name, zorder=2)
        ax.text(bars[-1].get_x() + width / 2, values[-1] + 0.012, f"{values[-1]:.2f}", ha="center", fontsize=8,
                color=COLOURS[a], fontweight="bold")
    ax.set_xticks(x, [g.replace("_", " ") for g in groups])
    ax.set_ylim(0, 0.84)
    ax.set_ylabel("F1 (claim within 30 cm of a changed object)", fontsize=9)
    ax.set_title("ChangeSim warehouses: F1 per pair, tuned profile", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_f1.png")
    plt.close(fig)


def benchmark_pr(runs: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150)
    r = np.linspace(0.01, 1, 200)
    for f in (0.2, 0.4, 0.6):
        p = f * r / (2 * r - f)
        ok = (p > 0) & (p <= 1)
        ax.plot(r[ok], p[ok], color="#dddddd", linewidth=0.9, zorder=1)
        ax.text(f * 0.97 / (2 * 0.97 - f) + 0.01, 0.97, f"F1 {f}", fontsize=7, color="#999999", va="top")
    for a, name in APPROACHES.items():
        pts = [runs[(p, a)]["metrics"] for p in HELD_OUT]
        ax.scatter([m["recall"] for m in pts], [m["precision"] for m in pts], s=18, color=COLOURS[a], alpha=0.45,
                   zorder=2, linewidths=0)
        p, rc, f1 = micro([runs[(q, a)] for q in HELD_OUT])
        ax.scatter([rc], [p], s=90, color=COLOURS[a], zorder=3, edgecolors="white", linewidths=1.2)
        dx, dy = LABEL_OFFSET[a]
        ax.annotate(f"{name}  (F1 {f1:.2f})", (rc, p), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=COLOURS[a], fontweight="bold", ha="left" if dx > 0 else "right", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("recall", fontsize=9)
    ax.set_ylabel("precision", fontsize=9)
    ax.set_title("Held-out warehouses: precision vs recall\n(large dots: W7–W9 together; small: per pair)",
                 fontsize=10, loc="left")
    style(ax)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_pr.png")
    plt.close(fig)


def room_cloud(approach: str) -> np.ndarray:
    """The "before" room cloud stored in the approach's viewer (uint16 positions, base64)."""
    text = (ROOT / "results" / approach / "viewer" / "data" / "room_before.js").read_text(encoding="utf-8")
    data = json.loads(re.search(r'= (\{.*\});?\s*$', text, re.S).group(1))
    q = np.frombuffer(base64.b64decode(data["positions"]), dtype="<u2").reshape(-1, 3).astype(np.float64)
    return q * np.array(data["scale"]) + np.array(data["origin"])


def change_ends(c: dict) -> dict:
    out = {k: np.array(c[k]["centroid_m"][:2]) for k in ("before", "after") if isinstance(c.get(k), dict)
           and "centroid_m" in c[k]}
    if isinstance(c.get("object"), dict) and "centroid_m" in c["object"]:
        out["before" if c["type"] in ("removed", "depth_only", "changed_at_landmark") else "after"] = \
            np.array(c["object"]["centroid_m"][:2])
    return out


def room_changes() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), dpi=150)
    for ax, (a, name) in zip(axes, APPROACHES.items()):
        cloud = room_cloud(a)
        z0 = np.percentile(cloud[:, 2], 2)
        keep = cloud[:, 2] < z0 + 1.6                       # cut the ceiling away
        ax.scatter(cloud[keep, 0], cloud[keep, 1], s=0.05, color="#c8c8c8", linewidths=0, rasterized=True)
        changes = json.loads((ROOT / "results" / a / "changes.json").read_text(encoding="utf-8"))["changes"]
        shown = 0
        for n, c in enumerate(changes):
            e = change_ends(c)
            if c["type"] in ("moved", "shifted", "rotated") and "before" in e and "after" in e:
                ax.annotate("", xy=e["after"], xytext=e["before"],
                            arrowprops=dict(arrowstyle="-|>", color=COLOURS[a], lw=2.2, shrinkA=0, shrinkB=0))
                ax.text(*e["after"], f" {c['label']}", fontsize=7, color="#222222",
                        va="bottom" if n % 2 else "top")             # alternate: close labels do not overlap
                shown += 1
            elif c["type"] in ("removed", "added", "changed_at_landmark", "surface_changed") and e:
                p = next(iter(e.values()))
                ax.scatter(*p, marker="x" if c["type"] == "removed" else "+", s=70, color=COLOURS[a], linewidths=2)
                ax.text(*p, f" {c['label']} ({c['type']})", fontsize=6.5, color="#555555",
                        va="bottom" if n % 2 else "top")          # alternate: close neighbours do not overlap
                shown += 1
        if a != "voxel_diff":                                 # the three staged changes, same frame
            for label, (old, new) in STAGED.items():
                ax.annotate("", xy=new, xytext=old, arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9,
                                                                    linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_gid(f"{name}: {shown} change{'s' if shown != 1 else ''} drawn")
    fig.text(0.01, 0.02, "Top view of the 'before' recording. Coloured arrows: reported moves; x / +: reported "
             "removed / added. Dashed black (approaches 2–4, which share a frame): the three staged moves.\n"
             "Approach 1 works in its own LiDAR frame; its 2 depth-only differences are not drawn.", fontsize=8,
             color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    for ax, a in zip(axes, APPROACHES):                     # titles on one line, above all panels
        fig.text(ax.get_position().x0, 0.95, ax.get_gid(), fontsize=10, ha="left",
                 color=COLOURS[a] if a != "voxel_diff" else "#555555")
    fig.savefig(OUT / "room_changes.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = tuned_runs()
    benchmark_f1(runs)
    benchmark_pr(runs)
    room_changes()
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
