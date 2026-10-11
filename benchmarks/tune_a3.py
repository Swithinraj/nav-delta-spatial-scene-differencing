"""Tune approach 3's large-space mining on the tuning pairs (seconds per trial, no segmentation).

  python benchmarks/tune_a3.py --pairs W6_S0,W6_S1 --trial NAME KEY=VALUE ...

Runs the mine stage with large_space + overrides ([loop_outliers.mine]) on each pair's shared
reconstruction, turns the mined changes into claims by position -- removed / added changes with
weak evidence are dropped as the segment stage drops uncorroborated ones -- scores them, and appends
the result to results/a3_trials.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import tomllib
from pathlib import Path

from paths import DATA, PY, RESULTS, ROOT, RUNS, SHARED

HERE = Path(__file__).resolve().parent
ALONE_MIN_MATCHES, ALONE_MIN_PAIRS = 60, 4


def write_profile(path: Path, base: str, overrides: dict) -> None:
    base_path = Path(base) if base.endswith(".toml") else ROOT / f"navdelta/configs/{base}.toml"
    values = tomllib.loads(base_path.read_text(encoding="utf-8"))
    values.setdefault("loop_outliers", {}).setdefault("mine", {}).update(overrides)
    lines = []
    for approach, stages in values.items():
        for stage, vals in stages.items():
            lines += [f"[{approach}.{stage}]", *[f"{k} = {json.dumps(v)}" for k, v in vals.items()], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pair(pair: str, trial_dir: Path, profile: Path) -> dict:
    data = DATA / pair
    work = trial_dir / pair
    (work / "recon").mkdir(parents=True, exist_ok=True)
    for f in (SHARED / pair / "recon").iterdir():
        if not (work / "recon" / f.name).exists():
            os.link(f, work / "recon" / f.name)
    started = time.time()
    with open(work / "mine.log", "w", encoding="utf-8") as fh:
        if subprocess.run([PY, "-X", "utf8", "-m", "navdelta", "detect", "--approach", "loop_outliers", "--from",
                           "mine", "--to", "mine", "--data", str(data), "--outputs", str(work), "--profile",
                           str(profile)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode:
            return {"pair": pair, "error": f"see {work / 'mine.log'}"}
    mined = json.loads((work / "loop_outliers" / "outlier_changes.json").read_text())
    report = json.loads((work / "recon" / "recon_report.json").read_text())
    rec_a, rec_b = report["frames"].keys()
    claims = []
    for c in mined["changes"]:
        if c["type"] == "moved":
            claims.append({"id": c["id"], "type": "moved", "label": "",
                           "before": {"centroid_m": c["before"]["centroid"]}, "after": {"centroid_m": c["after"]["centroid"]}})
        elif c["evidence_matches"] >= ALONE_MIN_MATCHES and c["view_pairs"] >= ALONE_MIN_PAIRS:
            claims.append({"id": c["id"], "type": c["type"], "label": "",
                           "object": {"centroid_m": c["object"]["centroid"],
                                      "present_in": rec_a if c["type"] == "removed" else rec_b}})
    changes = work / "loop_outliers" / "claims.json"
    changes.write_text(json.dumps({"changes": claims}))
    subprocess.run([PY, "-X", "utf8", str(HERE / "evaluate.py"), "--pair", str(data), "--changes", str(changes),
                    "--name", "loop_outliers", "--z-offset", str(report["floor_z"])], check=True, capture_output=True)
    ev = json.loads(changes.with_name("eval_loop_outliers.json").read_text())
    return {"pair": pair, "claims": ev["claims"], "tp": ev["true_positive_claims"], "precision": ev["precision"],
            "recall": ev["recall"], "found": round(ev["recall"] * ev["ground_truth"]["changed"]),
            "gt": ev["ground_truth"]["changed"], "f1": ev["f1"], "seconds": round(time.time() - started),
            "static_share": mined["static_share"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="W6_S0,W6_S1")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--base", default="large_space")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    overrides = {k: json.loads(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    trial_dir = RUNS / "tuning" / "a3" / args.trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    profile = trial_dir / "profile.toml"
    write_profile(profile, args.base, overrides)
    results = [run_pair(p, trial_dir, profile) for p in args.pairs.split(",")]
    ok = [r for r in results if "error" not in r]
    tot = {k: sum(r[k] for r in ok) for k in ("claims", "tp", "found", "gt")}
    p = tot["tp"] / max(1, tot["claims"])
    r = tot["found"] / max(1, tot["gt"])
    record = {"trial": args.trial, "base": args.base, "overrides": overrides, "pairs": results,
              "micro": {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3)}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "a3_trials.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"{args.trial:<16} P {p:.3f} R {r:.3f} F1 {record['micro']['f1']:.3f}  " +
          "  ".join(f"{x['pair']}: {x.get('claims')} claims, {x.get('found')}/{x.get('gt')}" for x in results))


if __name__ == "__main__":
    main()
