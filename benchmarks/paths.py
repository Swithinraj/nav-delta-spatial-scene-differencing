"""Where the benchmark keeps things.

Everything large lives under WORK (default benchmarks/work/, set NAVDELTA_BENCH_WORK to move it;
not in git):

  WORK/changesim/raw/            ChangeSim archive entries, downloaded on demand
  WORK/changesim/index_*.txt     the archive indexes (names + sizes)
  WORK/changesim/data/<pair>/    a pair converted to the phone-export layout, with gt.json
  WORK/shared/<pair>/            what several approaches build on:
      recon/                       the RGB reconstruction (`navdelta reconstruct`)
      detections_2d.npz            the scene graphs' 2D detections (made by the scene_graph run)
      voxel_diff_v1/               approach 1's whole first-design run (`navdelta run`)
  WORK/runs/final/<pair>/<profile>/<approach>/   the scored runs
  WORK/runs/tuning/<a1..a4>/<trial>/             the tuning trials

RESULTS (default benchmarks/results/, in git; NAVDELTA_BENCH_RESULTS moves it): every scored run
(final.jsonl) and tuning trial (a1..a4_trials.jsonl) as JSON lines, and the generated tables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = Path(os.environ.get("NAVDELTA_BENCH_WORK", HERE / "work"))
RESULTS = Path(os.environ.get("NAVDELTA_BENCH_RESULTS", HERE / "results"))
CHANGESIM = WORK / "changesim"
RAW = CHANGESIM / "raw"
DATA = CHANGESIM / "data"
SHARED = WORK / "shared"
RUNS = WORK / "runs"
PY = sys.executable

PAIRS = {"W6_S0": ("Warehouse_6", "Seq_0"), "W6_S1": ("Warehouse_6", "Seq_1"), "W7_S0": ("Warehouse_7", "Seq_0"),
         "W8_S0": ("Warehouse_8", "Seq_0"), "W9_S0": ("Warehouse_9", "Seq_0")}
TUNING_PAIRS = {"W6_S0", "W6_S1"}                  # the large-space values were tuned on these only
PROFILES = {"v1": HERE / "profiles" / "large_space_v1.toml",           # first design, before tuning
            "tuned": ROOT / "navdelta" / "configs" / "large_space.toml"}
V1_RGBD = HERE / "profiles" / "large_space_rgbd_v1.toml"              # approach 4 before its own tuning


def recordings(pair: str) -> tuple[str, str]:
    """The before / after recording names convert.py gives a pair (e.g. A_60ref, B_60qry)."""
    w, s = pair[1], pair.split("_S")[1]
    return f"A_{w}{s}ref", f"B_{w}{s}qry"
