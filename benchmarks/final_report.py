"""Benchmark tables for all four approaches, from final.jsonl (ChangeSim) and room_results.json (room).

  python benchmarks/final_report.py  ->  results/final_tables.md, results/final_summary.json and the
                                        tables of docs/benchmark.md

Micro-averaged over pairs: precision = correct claims / claims, recall = changed objects found /
changed objects. Latest record per (pair, approach, profile) wins.
"""
from __future__ import annotations

import json
from collections import defaultdict

from paths import RESULTS, ROOT, SHARED
APPROACHES = {"voxel_diff": "1 · voxel_diff (LiDAR voxels)", "scene_graph": "2 · scene_graph (RGB)",
              "loop_outliers": "3 · loop_outliers (RGB)", "rgbd_fusion": "4 · rgbd_fusion (RGB-D)"}
PAIRS = ["W6_S0", "W6_S1", "W7_S0", "W8_S0", "W9_S0"]
SPLITS = {"tuning": ["W6_S0", "W6_S1"], "held-out": ["W7_S0", "W8_S0", "W9_S0"]}
TYPES = ["removed", "added", "rotated", "replaced"]          # ChangeSim's change types
KEPT = {"a1": {"FRAGMENT_MAX_VOXELS": 8}, "a2": {"MIN_OBSERVATIONS": 1, "MIN_CHANGE_OBS": 2},
        "a3": {"MIN_MATCHES": 50, "RANSAC_M": 0.1, "CELL_M": 0.15}, "a4": {"fuse": {"VETO_PRESENT": 1.01}}}     # the tuned values


def flat(overrides: dict) -> list[tuple[str, object]]:
    """Trial overrides as sorted (key, value); approach 4's are per section (fuse.KEY)."""
    out = []
    for k, v in overrides.items():
        out += [(f"{k}.{kk}", vv) for kk, vv in v.items()] if isinstance(v, dict) else [(k, v)]
    return sorted(out)


def latest() -> dict:
    out = {}
    for line in (RESULTS / "final.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[(r["pair"], r["approach"], r["profile"])] = r
    return out


def micro(records: list[dict]) -> dict | None:
    if not records:
        return None
    claims = sum(r["metrics"]["claims"] for r in records)
    tp = sum(r["metrics"]["true_positive_claims"] for r in records)
    gt = sum(r["metrics"]["ground_truth"]["changed"] for r in records)
    found = sum(round(r["metrics"]["recall"] * r["metrics"]["ground_truth"]["changed"]) for r in records)
    p, rc = tp / max(1, claims), found / max(1, gt)
    return {"pairs": len(records), "claims": claims, "tp": tp, "gt": gt, "found": found, "precision": p, "recall": rc,
            "f1": 2 * p * rc / max(1e-9, p + rc), "seconds": sum(r["seconds"] for r in records) / len(records)}


def by_type(records: list[dict]) -> dict:
    hit, total = defaultdict(int), defaultdict(int)
    for r in records:
        for t, found in (r["metrics"].get("recall_by_type") or {}).items():     # "found/total"
            k, n = (int(x) for x in str(found).split("/"))
            hit[t] += k
            total[t] += n
    return {t: (hit[t] / total[t] if total[t] else None, total[t]) for t in TYPES}


def f(v: float | None, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def minutes(s: float | None) -> str:
    return "–" if not s else f"{s / 60:.0f} min"


def main() -> None:
    runs = latest()
    lines, summary = [], {}
    lines += ["## ChangeSim: tuned profile, micro-averaged", "",
              "| Approach | Tuning pairs (W6) P / R / F1 | Held-out pairs (W7–W9) P / R / F1 | All pairs F1 | "
              "Claims | Changes found |", "|---|---|---|---|---|---|"]
    for a, name in APPROACHES.items():
        rows = {s: micro([runs[(p, a, "tuned")] for p in ps if (p, a, "tuned") in runs]) for s, ps in SPLITS.items()}
        every = micro([runs[(p, a, "tuned")] for p in PAIRS if (p, a, "tuned") in runs])
        summary[a] = {"tuned": {**rows, "all": every}}
        cell = lambda m: "–" if not m else f"{f(m['precision'])} / {f(m['recall'])} / **{f(m['f1'])}** ({m['pairs']})"  # noqa: E731
        if every:
            totals = f"{f(every['f1'])} | {every['claims']} | {every['found']}/{every['gt']}"
        else:
            totals = "– | – | –"
        lines.append(f"| {name} | {cell(rows['tuning'])} | {cell(rows['held-out'])} | {totals} |")
    lines += ["", "(n) = number of pairs scored.", ""]

    lines += ["## ChangeSim: run time per pair (tuned, minutes)", "",
              "Wall time of the scored run on one laptop (RTX 3050 4 GB, 16 threads), shared with other jobs. "
              "G = models on the GPU, C = models on the CPU. voxel_diff: its whole pipeline (alignment to "
              "package). The other three build on the shared RGB reconstruction, listed first (2 and 3 use "
              "it, 4 reuses its MASt3R matches); each approach's row is its time after it. scene_graph's "
              "includes the 2D detections (Florence-2, SAM 2.1, SigLIP 2 on every keyframe), which "
              "rgbd_fusion reuses; rgbd_fusion's includes its own RGB-D reconstruction.", "",
              "| Stage / approach | " + " | ".join(PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
    recon = {}
    for p in PAIRS:
        report = SHARED / p / "recon" / "recon_report.json"
        if report.is_file():
            recon[p] = json.loads(report.read_text(encoding="utf-8")).get("runtime_s")
    lines.append("| shared: RGB reconstruction (MASt3R) | " + " | ".join(
        f"{recon[p] / 60:.0f} G" if recon.get(p) else "–" for p in PAIRS) + " |")
    for a, name in APPROACHES.items():
        cells = []
        for p in PAIRS:
            r = runs.get((p, a, "v1" if a == "voxel_diff" else "tuned"))
            device = "G" if r and (a == "voxel_diff" or r.get("device") == "cuda") else "C"
            cells.append("–" if not r else f"{r['seconds'] / 60:.0f} {device}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## ChangeSim: F1 per pair (tuned; first design in brackets)", "",
              "| Approach | " + " | ".join(f"{p}{' (tuning)' if p in SPLITS['tuning'] else ''}" for p in PAIRS) + " |",
              "|---|" + "---|" * len(PAIRS)]
    for a, name in APPROACHES.items():
        cells = []
        for p in PAIRS:
            t, v = runs.get((p, a, "tuned")), runs.get((p, a, "v1"))
            cells.append(("–" if not t else f(t["metrics"]["f1"])) + (f" ({f(v['metrics']['f1'])})" if v else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## ChangeSim: before vs after tuning (micro F1)", "",
              "| Approach | Tuning pairs: first design → tuned | Held-out pairs: first design → tuned |", "|---|---|---|"]
    for a, name in APPROACHES.items():
        cells = []
        for s, ps in SPLITS.items():
            both = [p for p in ps if (p, a, "v1") in runs and (p, a, "tuned") in runs]
            v = micro([runs[(p, a, "v1")] for p in both])
            t = micro([runs[(p, a, "tuned")] for p in both])
            cells.append("–" if not both else f"{f(v['f1'])} → **{f(t['f1'])}** ({len(both)})")
            summary[a].setdefault("v1_vs_tuned", {})[s] = {"v1": v, "tuned": t}
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines.append("")

    lines += ["## ChangeSim: registration of the two runs (tuned)", "",
              "Median position error of the second run's cameras against the ground truth, after the "
              "approach put both runs into one frame (cm). It includes how far the pose graph bent the first "
              "run. scene_graph and loop_outliers share one reconstruction.", "",
              "| Approach | " + " | ".join(PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
    for a, name in APPROACHES.items():
        cells = []
        for p in PAIRS:
            reg = (runs.get((p, a, "tuned")) or {}).get("metrics", {}).get("registration")
            cells.append(f"{reg['position_error_median_cm']:.0f}" if reg else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## ChangeSim: recall by type of change (tuned, all pairs)", "",
              "| Approach | " + " | ".join(TYPES) + " |", "|---|" + "---|" * len(TYPES)]
    for a, name in APPROACHES.items():
        recs = [runs[(p, a, "tuned")] for p in PAIRS if (p, a, "tuned") in runs]
        bt = by_type(recs)
        lines.append(f"| {name} | " + " | ".join(f"{f(v, 2)} ({n})" if n else "–" for v, n in bt.values()) + " |")
    lines += ["", "(n) = changed objects of that type in the scored pairs.", ""]

    lines += ["## Tuning trials (tuning pairs only)", "",
              "Each trial changes a few large-space values; F1 per tuning pair and micro-averaged over both. "
              "**kept** marks the values in `large_space.toml`.", ""]
    for key, approach in (("a1", "voxel_diff"), ("a2", "scene_graph"), ("a3", "loop_outliers"), ("a4", "rgbd_fusion")):
        path = RESULTS / f"{key}_trials.jsonl"
        if not path.is_file():
            continue
        trials: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not all(p["pair"] in SPLITS["tuning"] for p in r["pairs"]):
                continue
            name = json.dumps(r["overrides"], sort_keys=True)
            t = trials.setdefault(name, {"overrides": r["overrides"], "pairs": {}})
            for p in r["pairs"]:
                if "error" not in p:
                    t["pairs"][p["pair"]] = p
        lines += [f"**{APPROACHES[approach]}**", "", "| Values changed | W6_S0 F1 | W6_S1 F1 | Both (micro F1) | |",
                  "|---|---|---|---|---|"]
        for t in trials.values():
            ps = t["pairs"]
            both = None
            if len(ps) == 2:
                claims = sum(p["claims"] for p in ps.values())
                tp = sum(p["tp"] for p in ps.values())
                found, gt = sum(p["found"] for p in ps.values()), sum(p["gt"] for p in ps.values())
                pr, rc = tp / max(1, claims), found / max(1, gt)
                both = 2 * pr * rc / max(1e-9, pr + rc)
            changed = ", ".join(f"{k} = {v}" for k, v in flat(t["overrides"])) or "first design"
            kept = "**kept**" if flat(t["overrides"]) == flat(KEPT[key]) else ""
            lines.append(f"| {changed} | {f(ps.get('W6_S0', {}).get('f1'))} | {f(ps.get('W6_S1', {}).get('f1'))} | "
                         f"{f(both)} | {kept} |")
        lines.append("")

    room = RESULTS / "room_results.json"
    if room.is_file():
        rr = json.loads(room.read_text(encoding="utf-8"))
        rr.setdefault("voxel_diff", {"staged_found": 3, "staged_exact": 3, "other_claims": 5,
                                     "note": "hand-checked; own LiDAR frame"})
        lines += ["## Our room (3 staged changes)", "",
                  "| Approach | Staged changes found | Reported as moved with both places right | Extra claims |",
                  "|---|---|---|---|"]
        for a, name in APPROACHES.items():
            r = rr.get(a)
            if r:
                lines.append(f"| {name} | {r['staged_found']}/3 | {r.get('staged_exact', '–')}/3 | {r['other_claims']} |")
        summary["room"] = rr
        lines.append("")
    (RESULTS / "final_tables.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "final_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    doc = ROOT / "docs" / "benchmark.md"            # the tables also go into the benchmark page
    text = doc.read_text(encoding="utf-8")
    start, end = "<!-- tables:start -->", "<!-- tables:end -->"
    if start in text and end in text:
        body = "\n".join(lines).replace("## ", "### ")
        text = text[:text.index(start) + len(start)] + "\n" + body + "\n" + text[text.index(end):]
        doc.write_text(text, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
