"""Tune approach 2 on the tuning pairs (never on the held-out ones), from the shared expensive stages.

  python benchmarks/tune_a2.py --pairs W6_S0,W6_S1 --trial NAME [--base large_space] KEY=VALUE ...
  python benchmarks/tune_a2.py --diagnose --pairs W6_S0 --trial NAME   (also explain every miss)

A trial = the base profile + overrides (KEY=VALUE; the scene_graph stage is found from the module
that defines KEY). Per pair it hard-links the shared reconstruction and 2D detections, re-runs
objects (lifting and merging; the detections are reused), diff and describe, scores the result and
appends it to results/a2_trials.jsonl.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import time
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

from paths import DATA, PY, RESULTS, ROOT, RUNS, SHARED

HERE = Path(__file__).resolve().parent
CODE = ROOT / "navdelta" / "approaches" / "scene_graph"
STAGES = ("objects", "diff", "describe")


def stage_of(key: str) -> str:
    for stage in STAGES:
        tree = ast.parse((CODE / f"{stage}.py").read_text(encoding="utf-8"))
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in n.targets):
                return stage
    raise SystemExit(f"{key} is not a constant of {STAGES}")


def base_file(base: str) -> Path:
    return Path(base) if base.endswith(".toml") else ROOT / f"navdelta/configs/{base}.toml"


def write_profile(path: Path, base: str, overrides: dict) -> None:
    values = tomllib.loads(base_file(base).read_text(encoding="utf-8"))
    for key, value in overrides.items():
        values.setdefault("scene_graph", {}).setdefault(stage_of(key), {})[key] = value
    lines = []
    for approach, stages in values.items():
        for stage, vals in stages.items():
            lines.append(f"[{approach}.{stage}]")
            lines += [f"{k} = {json.dumps(v)}" for k, v in vals.items()]
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def diagnose(pair_dir: Path, work: Path) -> Counter:
    """Per ground-truth object: is there a node there, what state, why not reported."""
    gt = json.loads((pair_dir / "gt.json").read_text())
    floor = json.loads((work / "recon" / "recon_report.json").read_text())["floor_z"]
    out = work / "scene_graph"
    graphs = np.load(out / "graphs.npy", allow_pickle=True).item()
    diff = json.loads((out / "graph_diff.json").read_text())
    ev = json.loads((out / "eval_scene_graph.json").read_text())
    found = {o["object"] for o in ev["objects"] if o["detected_by"]}
    recs = list(graphs)
    verdicts = Counter()
    for c in gt["changes"]:
        if c["id"] in found:
            verdicts["reported"] += 1
            continue
        side = 0 if c["type"] == "removed" else 1
        box = c["before"] or c["after"]
        lo = np.array(box["centroid"]) - np.array(box["lengths"]) / 2 - 0.3
        hi = np.array(box["centroid"]) + np.array(box["lengths"]) / 2 + 0.3
        best = None
        for k, n in enumerate(graphs[recs[side]]["nodes"]):
            v = n["voxels"] + np.array([0, 0, floor])
            share = float(np.all((v >= lo) & (v <= hi), axis=1).mean())
            if share > 0.3 and (best is None or share > best[1]):
                best = (k, share, n)
        if best is None:
            verdicts["no node there"] += 1
            continue
        k, _, n = best
        st = diff["node_states"][recs[side]][k]
        size = float(np.max(n["bbox_max"] - n["bbox_min"]))
        why = (f"state {st['state']}" if st["state"] != "gone" else
               "seen in < MIN_CHANGE_OBS frames" if n["observations"] < 3 else
               "smaller than MIN_CHANGE_SIZE_M" if size < 0.3 else "gone but paired / filtered")
        verdicts[why] += 1
    return verdicts


def run_pair(pair: str, trial_dir: Path, profile: Path, do_diagnose: bool) -> dict:
    data = DATA / pair
    shared = SHARED / pair
    work = trial_dir / pair
    for f in (shared / "recon").iterdir():
        link(f, work / "recon" / f.name)
    link(shared / "detections_2d.npz", work / "scene_graph" / "detections_2d.npz")
    started = time.time()
    with open(work / "run.log", "w", encoding="utf-8") as fh:
        if subprocess.run([PY, "-X", "utf8", "-m", "navdelta", "detect", "--approach", "scene_graph", "--from",
                           "objects", "--to", "describe", "--data", str(data), "--outputs", str(work),
                           "--profile", str(profile)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode:
            return {"pair": pair, "error": f"see {work / 'run.log'}"}
    floor = json.loads((work / "recon" / "recon_report.json").read_text())["floor_z"]
    changes = work / "scene_graph" / "changes.json"
    subprocess.run([PY, "-X", "utf8", str(HERE / "evaluate.py"), "--pair", str(data), "--changes", str(changes),
                    "--name", "scene_graph", "--z-offset", str(floor)], check=True, capture_output=True)
    ev = json.loads(changes.with_name("eval_scene_graph.json").read_text())
    out = {"pair": pair, "claims": ev["claims"], "tp": ev["true_positive_claims"], "precision": ev["precision"],
           "recall": ev["recall"], "found": round(ev["recall"] * ev["ground_truth"]["changed"]),
           "gt": ev["ground_truth"]["changed"], "f1": ev["f1"], "seconds": round(time.time() - started)}
    if do_diagnose:
        out["misses"] = dict(diagnose(data, work))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="W6_S0")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--base", default="large_space")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    overrides = {k: json.loads(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    trial_dir = RUNS / "tuning" / "a2" / args.trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    profile = trial_dir / "profile.toml"
    write_profile(profile, args.base, overrides)
    results = [run_pair(p, trial_dir, profile, args.diagnose) for p in args.pairs.split(",")]
    ok = [r for r in results if "error" not in r]
    tot = {k: sum(r[k] for r in ok) for k in ("claims", "tp", "found", "gt")}
    p = tot["tp"] / max(1, tot["claims"])
    r = tot["found"] / max(1, tot["gt"])
    record = {"trial": args.trial, "base": args.base, "overrides": overrides, "pairs": results,
              "micro": {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3)}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "a2_trials.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
