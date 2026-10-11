"""Tune approach 4 (rgbd_fusion) on the tuning pairs, from a finished v1 run of each pair.

  python benchmarks/tune_a4.py --pairs W6_S0,W6_S1 --trial NAME [--base FILE] SECTION.KEY=VALUE ...

SECTION is diff, mine, segment or fuse (the [rgbd_fusion.<section>] keys of the profile). A trial
re-runs the stages it changes, the stages that read their output (mine -> segment) and fuse --
fuse alone takes seconds, diff minutes, segment (SAM 2.1 + SigLIP 2) tens of minutes on the CPU --
in its own folder (work/runs/tuning/a4/<trial>/<pair>), scores the result and appends it to
results/a4_trials.jsonl. It starts from each pair's first-design run (final_eval.py --profile v1).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

from paths import DATA, PY, RESULTS, ROOT, RUNS, V1_RGBD

HERE = Path(__file__).resolve().parent
ORDER = ["diff", "mine", "segment", "fuse"]
READ_ONLY = {"detections_2d.npz", "graphs.npy", "scene_graphs.json"}     # written by objects, never re-run here


def write_profile(path: Path, base: Path, overrides: dict) -> None:
    values = tomllib.loads(base.read_text(encoding="utf-8"))
    for section, settings in overrides.items():
        values.setdefault("rgbd_fusion", {}).setdefault(section, {}).update(settings)
    lines = []
    for approach, stages in values.items():
        for stage, vals in stages.items():
            lines += [f"[{approach}.{stage}]", *[f"{k} = {json.dumps(v)}" for k, v in vals.items()], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def run_pair(pair: str, trial_dir: Path, profile: Path, stages: list[str]) -> dict:
    data = DATA / pair
    base = RUNS / "final" / pair / "v1" / "rgbd_fusion"
    work = trial_dir / pair
    for f in (base / "recon_rgbd").iterdir():
        if f.is_file():
            link(f, work / "recon_rgbd" / f.name)
    (work / "rgbd_fusion").mkdir(parents=True, exist_ok=True)
    for f in (base / "rgbd_fusion").iterdir():      # what a stage writes is copied: a hard link would change v1
        if f.is_file():
            if f.name in READ_ONLY:
                link(f, work / "rgbd_fusion" / f.name)
            else:
                shutil.copy(f, work / "rgbd_fusion" / f.name)
    started = time.time()
    env = {**os.environ, "NAVDELTA_DEVICE": os.environ.get("NAVDELTA_DEVICE", "cpu")}
    with open(work / "trial.log", "w", encoding="utf-8") as fh:
        for stage in stages:
            if subprocess.run([PY, "-u", "-X", "utf8", "-m", "navdelta", "detect", "--approach", "rgbd_fusion",
                               "--from", stage, "--to", stage, "--data", str(data), "--outputs", str(work),
                               "--profile", str(profile)], cwd=ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT).returncode:
                return {"pair": pair, "error": f"{stage}: see {work / 'trial.log'}"}
    floor = json.loads((work / "recon_rgbd" / "recon_report.json").read_text())["floor_z"]
    changes = work / "rgbd_fusion" / "changes.json"
    subprocess.run([PY, "-X", "utf8", str(HERE / "evaluate.py"), "--pair", str(data), "--changes",
                    str(changes), "--name", "rgbd_fusion", "--scene", str(work / "recon_rgbd" / "scene.npz"),
                    "--z-offset", str(floor)], check=True, capture_output=True)
    ev = json.loads(changes.with_name("eval_rgbd_fusion.json").read_text())
    out = json.loads(changes.read_text(encoding="utf-8"))
    return {"pair": pair, "claims": ev["claims"], "tp": ev["true_positive_claims"], "precision": ev["precision"],
            "recall": ev["recall"], "found": round(ev["recall"] * ev["ground_truth"]["changed"]),
            "gt": ev["ground_truth"]["changed"], "f1": ev["f1"], "seconds": round(time.time() - started),
            "confirmed": sum(c["confirmed"] for c in out["changes"]), "dropped": len(out["dropped"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="W6_S0,W6_S1")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--base", default=str(V1_RGBD))
    ap.add_argument("overrides", nargs="*", help="SECTION.KEY=VALUE, e.g. fuse.GRAPH_ALONE_EMPTY=0.7")
    args = ap.parse_args()
    overrides: dict[str, dict] = {}
    for item in args.overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        if section not in ORDER:
            raise SystemExit(f"unknown section {section!r}: one of {ORDER}")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        overrides.setdefault(section, {})[name] = value
    stages = [s for s in ORDER if s in overrides or (s == "segment" and "mine" in overrides) or s == "fuse"]
    trial_dir = RUNS / "tuning" / "a4" / args.trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    profile = trial_dir / "profile.toml"
    write_profile(profile, Path(args.base), overrides)
    pairs = [run_pair(p, trial_dir, profile, stages) for p in args.pairs.split(",")]
    ok = [p for p in pairs if "error" not in p]
    claims, tp = sum(p["claims"] for p in ok), sum(p["tp"] for p in ok)
    found, gt = sum(p["found"] for p in ok), sum(p["gt"] for p in ok)
    precision, recall = tp / max(1, claims), found / max(1, gt)
    micro = {"precision": round(precision, 3), "recall": round(recall, 3),
             "f1": round(2 * precision * recall / max(1e-9, precision + recall), 3)}
    record = {"trial": args.trial, "base": Path(args.base).name, "overrides": overrides, "stages": stages,
              "pairs": pairs, "micro": micro}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "a4_trials.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
