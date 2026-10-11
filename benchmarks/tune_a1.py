"""Tune approach 1's large-space profile on the tuning pairs (never on the held-out ones).

  python benchmarks/tune_a1.py --pairs W6_S0,W6_S1 --trial NAME KEY=VALUE [KEY=VALUE ...]

A trial = large_space.toml + overrides ([voxel_diff.<stage>] KEY = VALUE; the stage is found from
the module that defines KEY). On each pair it re-runs change_map + objects on that pair's alignment
(from approach 1's first-design run), scores the reported changes and appends the result to
results/a1_trials.jsonl. The geometry stages need Open3D in the running Python.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from paths import DATA, PY, RESULTS, ROOT, RUNS, SHARED

HERE = Path(__file__).resolve().parent
STAGES = ("change_map", "objects")


def stage_of(key: str) -> str:
    for stage in STAGES:
        src = (ROOT / "navdelta/approaches/voxel_diff" / f"{stage}.py").read_text(encoding="utf-8")
        if any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key or
                                                  isinstance(t, ast.Tuple) and any(getattr(e, "id", None) == key
                                                                                   for e in t.elts)
                                                  for t in n.targets) for n in ast.parse(src).body):
            return stage
    raise SystemExit(f"{key} is not a constant of {STAGES}")


def base_file(base: str) -> Path:
    return Path(base) if base.endswith(".toml") else ROOT / f"navdelta/configs/{base}.toml"


def write_profile(path: Path, overrides: dict, base_name: str = "large_space") -> None:
    base = tomllib.loads(base_file(base_name).read_text(encoding="utf-8"))
    for key, value in overrides.items():
        base.setdefault("voxel_diff", {}).setdefault(stage_of(key), {})[key] = value
    lines = []
    for approach, stages in base.items():
        for stage, values in stages.items():
            lines.append(f"[{approach}.{stage}]")
            lines += [f"{k} = {json.dumps(v)}" for k, v in values.items()]
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def link_tree(src: Path, dst: Path) -> None:
    """Hard links of every file: the trial reads the shared alignment, it never writes to it."""
    for f in src.rglob("*"):
        if f.is_file():
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(f, target)


def run_pair(pair: str, trial_dir: Path, profile: Path) -> dict:
    data = DATA / pair
    base = SHARED / pair / "voxel_diff_v1"
    out = trial_dir / pair
    out.mkdir(parents=True, exist_ok=True)
    link_tree(base / "alignment", out / "alignment")
    started = time.time()
    log = out / "run.log"
    with open(log, "w") as fh:
        code = subprocess.run([PY, "-X", "utf8", "-m", "navdelta", "detect", "--approach", "voxel_diff",
                               "--from", "change_map", "--to", "objects", "--data", str(data), "--outputs", str(out),
                               "--profile", str(profile)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if code:
        return {"pair": pair, "error": f"exit {code}, see {log}"}
    subprocess.run([PY, "-X", "utf8", str(HERE / "voxel_diff_claims.py"), str(out / "voxel_diff")],
                   check=True, capture_output=True)
    changes = out / "voxel_diff" / "changes_from_step2.json"
    subprocess.run([PY, "-X", "utf8", str(HERE / "evaluate.py"), "--pair", str(data), "--changes", str(changes),
                    "--name", "voxel_diff"], check=True, capture_output=True)
    ev = json.loads(changes.with_name("eval_voxel_diff.json").read_text())
    return {"pair": pair, "claims": ev["claims"], "tp": ev["true_positive_claims"], "precision": ev["precision"],
            "recall": ev["recall"], "found": round(ev["recall"] * ev["ground_truth"]["changed"]),
            "gt": ev["ground_truth"]["changed"], "f1": ev["f1"], "seconds": round(time.time() - started)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="W6_S0")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--base", default="large_space", help="profile name or .toml file the overrides apply to")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    overrides = {k: json.loads(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    trial_dir = RUNS / "tuning" / "a1" / args.trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    profile = trial_dir / "profile.toml"
    write_profile(profile, overrides, args.base)
    results = [run_pair(p, trial_dir, profile) for p in args.pairs.split(",")]
    ok = [r for r in results if "error" not in r]
    total = {"claims": sum(r["claims"] for r in ok), "tp": sum(r["tp"] for r in ok),
             "found": sum(r["found"] for r in ok), "gt": sum(r["gt"] for r in ok)}
    p = total["tp"] / max(1, total["claims"])
    r = total["found"] / max(1, total["gt"])
    record = {"trial": args.trial, "base": args.base, "overrides": overrides, "pairs": results,
              "micro": {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3)}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "a1_trials.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    sys.exit(main())
