"""Prepare ChangeSim pairs: download, convert, and run the stages several approaches share.

  python benchmarks/prepare.py --pairs W6_S0,W6_S1 [--steps index,download,convert,voxel_diff,reconstruct]

  index        list the two ChangeSim archives once (names + sizes, a few MB)
  download     fetch only the archive entries a pair needs (keyframes, depth, poses, change masks)
  convert      convert the pair to the phone-export layout, with the ground truth (gt.json)
  voxel_diff   approach 1's whole pipeline with its first-design profile -> shared/<pair>/voxel_diff_v1
  reconstruct  the RGB reconstruction shared by approaches 2-4 -> shared/<pair>/recon

Approach 1 needs Open3D for its geometry stages (Linux / WSL). The Docker images hold the navdelta
package but not this folder: use a local environment, or mount the repository into a container. The 2D detections
of the scene graphs are made by the first scene_graph run of final_eval.py and then shared.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import paths
from paths import CHANGESIM, DATA, PROFILES, PY, ROOT, SHARED

HERE = Path(__file__).resolve().parent
STEPS = ["index", "download", "convert", "voxel_diff", "reconstruct"]


def run(label: str, command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {label} (log: {log})", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        if subprocess.run(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode:
            raise SystemExit(f"failed, see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(paths.PAIRS))
    ap.add_argument("--steps", default=",".join(STEPS))
    args = ap.parse_args()
    steps = args.steps.split(",")
    if "index" in steps:
        for part in ("Ref_Seq_Test", "Query_Seq_Test"):
            if not (CHANGESIM / f"index_{part}.txt").is_file():
                run(f"index {part}", [PY, "-X", "utf8", str(HERE / "changesim" / "remote_zip.py"), "list", part],
                    CHANGESIM / f"index_{part}.log")
    for pair in args.pairs.split(","):
        warehouse, seq = paths.PAIRS[pair]
        data, shared = DATA / pair, SHARED / pair
        scripts = HERE / "changesim"
        jobs = {
            "download": [str(scripts / "prefetch.py"), "--warehouse", warehouse, "--seq", seq],
            "convert": [str(scripts / "convert.py"), "--warehouse", warehouse, "--seq", seq, "--out", str(data)],
            "voxel_diff": ["-m", "navdelta", "run", "--approach", "voxel_diff", "--data", str(data),
                           "--outputs", str(shared / "voxel_diff_v1"), "--profile", str(PROFILES["v1"])],
            "reconstruct": ["-m", "navdelta", "reconstruct", "--data", str(data), "--outputs", str(shared),
                            "--profile", str(PROFILES["tuned"])],
        }
        print(pair, flush=True)
        for step, arguments in jobs.items():
            if step in steps:
                run(step, [PY, "-X", "utf8", *arguments], shared / "logs" / f"{step}.log")

if __name__ == "__main__":
    main()
