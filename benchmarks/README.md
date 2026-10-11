# Benchmark tooling

Scripts that produced the numbers in [`docs/benchmark.md`](../docs/benchmark.md): the four
approaches on five ChangeSim warehouse pairs and on our room recordings. Results are in
[`results/`](results/). Everything large (downloads, converted pairs, runs) goes to
`benchmarks/work/`, which is not in git. Set `NAVDELTA_BENCH_WORK` to put it elsewhere.

| Script | What it does |
|---|---|
| [`paths.py`](paths.py) | where everything lives; the five pairs; which two are for tuning |
| [`prepare.py`](prepare.py) | per pair: download the needed ChangeSim files, convert, run approach 1's first design, run the shared RGB reconstruction |
| [`changesim/`](changesim/) | range-request reads of ChangeSim's public archives (`remote_zip.py`), download (`prefetch.py`) and conversion to the phone-export layout with a ground truth (`convert.py`) |
| [`evaluate.py`](evaluate.py) | scores one `changes.json` against a pair's ground truth: precision, recall, F1, recall per type, registration |
| [`final_eval.py`](final_eval.py) | one approach × profile (`v1` = first design, `tuned`) × pairs, scored; appends to `results/final.jsonl` |
| [`tune_a1.py`](tune_a1.py) … [`tune_a4.py`](tune_a4.py) | tuning trials on the tuning pairs; append to `results/a*_trials.jsonl` |
| [`room_score.py`](room_score.py) | the room: staged changes found, extra claims → `results/room_results.json` |
| [`final_report.py`](final_report.py) | all tables → `results/final_tables.md` and the tables in `docs/benchmark.md` |
| [`voxel_diff_claims.py`](voxel_diff_claims.py) | approach 1's Step 2 objects as claims (used for its first-design and tuning scores) |
| [`profiles/`](profiles/) | the first-design large-space profiles: `large_space_v1.toml` (approaches 1–3), `large_space_rgbd_v1.toml` (approach 4) |

## Reproduce

Run from the repository root, in an environment with the pinned dependencies
(`pip install -r requirements.txt && pip install -e .`, see the main README). The Docker images hold
the `navdelta` package but not this folder: either mount the repository into a container, or use
a local environment. Approach 1's geometry stages need Open3D (Linux / WSL), and the models need a
GPU to finish in reasonable time.

```bash
python benchmarks/prepare.py                                   # all five pairs (downloads ~2.5 GB)
for a in voxel_diff scene_graph loop_outliers rgbd_fusion; do
  python benchmarks/final_eval.py --approach $a --profile tuned --device cuda
done
python benchmarks/final_eval.py --approach voxel_diff --profile v1
python benchmarks/final_eval.py --approach rgbd_fusion --profile v1 --device cuda
python benchmarks/room_score.py                                # after `navdelta run` of approaches 2-4 on the room
python benchmarks/final_report.py
```

Run `scene_graph` before `rgbd_fusion`: its first run on a pair makes the 2D detections that
`rgbd_fusion` reuses (`work/shared/<pair>/detections_2d.npz`). For approach 4, `--reuse v1` turns a
first-design run into the tuned one by re-running only the stages whose settings differ (here:
fuse and package).

## How the published numbers were made

On one Windows laptop (RTX 3050, 4 GB): the model stages ran in a Windows Python, and approach 1's
geometry stages in WSL (Open3D). The shared stages were made by a queue running the development
versions of the same code, from which the packaged stages were taken unchanged. The scored runs
used the packaged CLI exactly as `final_eval.py` runs it. The ported scripts were checked against
the original numbers:
- `evaluate.py` re-scores a stored result identically.
- `tune_a3.py` reproduces its original trial (W6_S0: 45 claims, F1 0.325).
- `prepare.py --steps download,convert` rebuilds W7_S0's `gt.json` byte for byte.

`results/final.jsonl` keeps the latest run per pair, approach and profile. The scene-graph-based
approaches (2 and 4) were re-run after one fix: SAM-grid segments are now named over a fixed
vocabulary (ScanNet200). Before, the vocabulary depended on which frames were detected together.
Their tuning trials (`a2_trials.jsonl`, `a4_trials.jsonl`) were run before that fix.
