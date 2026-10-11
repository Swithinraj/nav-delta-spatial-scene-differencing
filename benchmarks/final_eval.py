"""Final benchmark: every ChangeSim pair x every approach x {first design, tuned}, scored the same way.

  python benchmarks/final_eval.py --approach scene_graph --profile v1|tuned [--pairs ...] [--device cpu|cuda]
  python benchmarks/final_eval.py --approach rgbd_fusion --profile tuned --reuse v1   (re-run only what changed)

Needs each pair's shared inputs (benchmarks/prepare.py): the converted pair, the RGB reconstruction
and approach 1's first-design run.
  voxel_diff     v1: approach 1's first-design run, scored from its Step 2 objects; tuned: change map +
                 objects re-run with the tuned profile (tune_a1.run_pair)
  scene_graph    objects (2D detections on the first run of a pair, then shared), diff, describe,
                 package with the profile, on the shared reconstruction
  loop_outliers  mine, segment, context, package with the profile, on the shared reconstruction
  rgbd_fusion    navdelta run: its RGB-D reconstruction (reusing the shared MASt3R matches), objects
                 (the shared 2D detections), diff, mine, segment, fuse, package
Each approach runs in its own folder (work/runs/final/<pair>/<profile>/<approach>/), so runs can
go in parallel. Every result (scores, stages, run times) is appended to results/final.jsonl;
final_report.py builds the tables.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import paths
from paths import DATA, PROFILES, PY, RESULTS, ROOT, RUNS, SHARED, TUNING_PAIRS, V1_RGBD

HERE = Path(__file__).resolve().parent


def evaluate(pair: str, changes: Path, name: str, extra: list[str]) -> dict:
    subprocess.run([PY, "-X", "utf8", str(HERE / "evaluate.py"), "--pair", str(DATA / pair), "--changes",
                    str(changes), "--name", name, *extra], check=True, capture_output=True)
    ev = json.loads(changes.with_name(f"eval_{name}.json").read_text())
    keys = ("claims", "true_positive_claims", "false_positive_claims", "precision", "recall", "recall_by_type",
            "f1", "type_accuracy", "registration")
    return {"ground_truth": ev["ground_truth"], **{k: ev[k] for k in keys}}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def run_voxel_diff(pair: str, profile: str) -> dict:
    v1 = SHARED / pair / "voxel_diff_v1"
    if profile == "v1":
        subprocess.run([PY, "-X", "utf8", str(HERE / "voxel_diff_claims.py"), str(v1 / "voxel_diff")], check=True,
                       capture_output=True)
        metrics = evaluate(pair, v1 / "voxel_diff" / "changes_from_step2.json", "voxel_diff",
                           ["--poses-dir", str(v1 / "alignment" / "aligned_frame_poses")])
        stages = []
        for line in (v1 / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            stages += [s for s in json.loads(line)["stages"] if s["stage"] in
                       ("alignment.align", "change_map", "objects", "labels", "landmarks", "colours", "describe", "package")]
        return {"metrics": metrics, "stages": stages, "seconds": round(sum(s["seconds"] for s in stages), 1)}
    import tune_a1
    trial = RUNS / "tuning" / "a1" / "final_tuned"
    trial.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROFILES["tuned"], trial / "profile.toml")
    started = time.time()
    r = tune_a1.run_pair(pair, trial, trial / "profile.toml")
    if "error" in r:
        raise SystemExit(r["error"])
    out = trial / pair / "voxel_diff"
    metrics = evaluate(pair, out / "changes_from_step2.json", "voxel_diff",
                       ["--poses-dir", str(v1 / "alignment" / "aligned_frame_poses")])
    return {"metrics": metrics, "stages": [{"stage": "change_map + objects (tuned)", "seconds": round(time.time() - started, 1)}],
            "seconds": round(time.time() - started, 1)}


def run_rgb(pair: str, approach: str, profile: str, device: str) -> dict:
    shared = SHARED / pair
    out = RUNS / "final" / pair / profile / approach
    for f in (shared / "recon").iterdir():             # the shared RGB reconstruction (read only)
        if f.suffix == ".npz":
            link(f, out / "recon" / f.name)
    (out / "recon").mkdir(parents=True, exist_ok=True)
    shutil.copy(shared / "recon" / "recon_report.json", out / "recon" / "recon_report.json")
    detections = shared / "detections_2d.npz"          # 2D detections of the scene graphs, once per pair
    if detections.is_file():
        link(detections, out / "scene_graph" / "detections_2d.npz")
    env = {**os.environ, "NAVDELTA_DEVICE": "cpu"} if device == "cpu" else dict(os.environ)
    rgbd = approach == "rgbd_fusion"
    profile_file = V1_RGBD if rgbd and profile == "v1" else PROFILES[profile]
    started = time.time()
    with open(out / f"{approach}.log", "w", encoding="utf-8") as fh:
        command = ["run"] if rgbd else ["detect"]          # rgbd_fusion: its own RGB-D reconstruction first
        code = subprocess.run([PY, "-u", "-X", "utf8", "-m", "navdelta", *command, "--approach", approach,
                               "--data", str(DATA / pair), "--outputs", str(out), "--profile", str(profile_file)],
                              cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"{approach} failed on {pair}, see {out / f'{approach}.log'}")
    made = out / "scene_graph" / "detections_2d.npz"
    if approach == "scene_graph" and made.is_file() and not detections.exists():
        os.link(made, detections)                      # computed by this run: shared with rgbd_fusion
    recon = out / ("recon_rgbd" if rgbd else "recon")
    floor = json.loads((recon / "recon_report.json").read_text())["floor_z"]
    metrics = evaluate(pair, out / approach / "changes.json", approach,
                       ["--scene", str(recon / "scene.npz"), "--z-offset", str(floor)])
    run = json.loads((out / "run_log.json").read_text())
    stages = run["stages"]
    recon_seconds = json.loads((shared / "recon" / "recon_report.json").read_text()).get("runtime_s")
    if not rgbd:            # the shared RGB reconstruction: its time counts too
        stages = [{"stage": "shared reconstruction", "seconds": recon_seconds}] + stages
    else:                   # the RGB-D reconstruction reuses its MASt3R matches: count their time
        stages = [{"stage": "RGB reconstruction whose MASt3R matches are reused (upper bound)",
                   "seconds": recon_seconds}] + stages
    return {"metrics": metrics, "stages": stages, "seconds": round(time.time() - started, 1)}


STAGE_SECTIONS = {"objects": ("scene_graph", "objects"), "diff": ("scene_graph", "diff"),
                  "mine": ("loop_outliers", "mine"), "segment": ("loop_outliers", "segment")}
DEPENDS = {"diff": "objects", "segment": "mine"}      # fuse and package always re-run


def rerun_rgbd(pair: str, profile: str, source: str) -> dict:
    """rgbd_fusion with `profile`, starting from the `source` profile's finished run: only the stages
    whose settings differ (and the stages that read their output) are re-run, then fuse and package.
    The shared RGB-D reconstruction must have the same settings."""
    import tomllib
    files = {name: V1_RGBD if name == "v1" else PROFILES[name] for name in (profile, source)}
    values = {name: tomllib.loads(f.read_text(encoding="utf-8")) for name, f in files.items()}

    def settings(name: str, stage: str) -> tuple:
        v = values[name]
        base, own = STAGE_SECTIONS[stage]
        return v.get(base, {}).get(own), v.get("rgbd_fusion", {}).get(stage)
    if values[profile].get("recon") != values[source].get("recon"):
        raise SystemExit(f"{profile} and {source} differ in the reconstruction: run the pipeline instead")
    rerun = []
    for stage in ("objects", "diff", "mine", "segment"):
        if settings(profile, stage) != settings(source, stage) or DEPENDS.get(stage) in rerun:
            rerun.append(stage)
    rerun += ["fuse", "package"]
    src = RUNS / "final" / pair / source / "rgbd_fusion"
    out = RUNS / "final" / pair / profile / "rgbd_fusion"
    if src != out:                                     # else: the same run, re-run in place
        for f in (src / "recon_rgbd").iterdir():
            if f.is_file():
                link(f, out / "recon_rgbd" / f.name)
        (out / "rgbd_fusion").mkdir(parents=True, exist_ok=True)
        for f in (src / "rgbd_fusion").iterdir():      # copies: the re-run stages rewrite some of them
            if f.is_file():
                shutil.copy(f, out / "rgbd_fusion" / f.name) if f.suffix != ".npy" else link(f, out / "rgbd_fusion" / f.name)
    timings = {}
    with open(out / "rgbd_fusion.log", "w", encoding="utf-8") as fh:
        for stage in rerun:
            code = subprocess.run([PY, "-u", "-X", "utf8", "-m", "navdelta", "detect", "--approach", "rgbd_fusion",
                                   "--from", stage, "--to", stage, "--data", str(DATA / pair), "--outputs", str(out),
                                   "--profile", str(files[profile])], cwd=ROOT,
                                  env={**os.environ, "NAVDELTA_DEVICE": "cpu"}, stdout=fh,
                                  stderr=subprocess.STDOUT).returncode
            if code:
                raise SystemExit(f"{stage} failed on {pair}, see {out / 'rgbd_fusion.log'}")
            timings.update({s["stage"]: s["seconds"] for s in json.loads((out / "run_log.json").read_text())["stages"]})
    floor = json.loads((out / "recon_rgbd" / "recon_report.json").read_text())["floor_z"]
    metrics = evaluate(pair, out / "rgbd_fusion" / "changes.json", "rgbd_fusion",
                       ["--scene", str(out / "recon_rgbd" / "scene.npz"), "--z-offset", str(floor)])
    previous = [json.loads(line) for line in (RESULTS / "final.jsonl").read_text(encoding="utf-8").splitlines()]
    base = [r for r in previous if r["pair"] == pair and r["approach"] == "rgbd_fusion" and r["profile"] == source][-1]
    stages = [{**st, "seconds": timings.get(st["stage"], st["seconds"])} for st in base["stages"]]
    delta = sum(timings[st["stage"]] - st["seconds"] for st in base["stages"] if st["stage"] in timings)
    return {"metrics": metrics, "stages": stages, "seconds": round(base["seconds"] + delta, 1),
            "reused_stages_from": source, "rerun_stages": rerun}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True, choices=["voxel_diff", "scene_graph", "loop_outliers", "rgbd_fusion"])
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--pairs", default=",".join(paths.PAIRS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reuse", choices=list(PROFILES),
                    help="rgbd_fusion: start from this profile's finished run and re-run only the stages whose "
                         "settings differ (then fuse and package)")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for pair in args.pairs.split(","):
        if args.reuse:
            result = rerun_rgbd(pair, args.profile, args.reuse)
        else:
            result = (run_voxel_diff(pair, args.profile) if args.approach == "voxel_diff"
                      else run_rgb(pair, args.approach, args.profile, args.device))
        record = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": f"ChangeSim {pair}", "pair": pair,
                  "split": "tuning" if pair in TUNING_PAIRS else "held-out", "approach": args.approach,
                  "profile": args.profile, "device": "cpu" if args.approach == "voxel_diff" else args.device, **result}
        with open(RESULTS / "final.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        m = result["metrics"]
        print(f"{pair} {args.approach} {args.profile}: P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} "
              f"({m['claims']} claims) {result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
