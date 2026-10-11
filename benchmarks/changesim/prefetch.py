"""Download every archive entry convert.py needs for one ChangeSim pair, 8 at a time, into raw/.

  python benchmarks/changesim/prefetch.py --warehouse Warehouse_7 --seq Seq_0

File names come from the archive indexes (index_<part>.txt, written by remote_zip.py list).
The frame choice mirrors convert.py: ~--keyframes keyframes per run, ~--gt-frames query frames
for the ground truth.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import RangeFile  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402
local = threading.local()


def fetch(part: str, name: str, raw: Path) -> int:
    target = raw / name
    if target.is_file():
        return 0
    if not hasattr(local, "zips"):
        local.zips = {}
    if part not in local.zips:
        local.zips[part] = zipfile.ZipFile(RangeFile(part))
    data = local.zips[part].read(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return len(data)


def run(jobs: list[tuple[str, str]], raw: Path, label: str) -> None:
    todo = [j for j in jobs if not (raw / j[1]).is_file()]
    print(f"{label}: {len(jobs)} files needed, {len(todo)} to download", flush=True)
    started = time.time()
    with ThreadPoolExecutor(8) as pool:
        for done, _ in enumerate(pool.map(lambda j: fetch(j[0], j[1], raw), todo), 1):
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)


def frame_ids(index: Path, prefix: str, suffix: str) -> list[int]:
    out = []
    for line in index.read_text().splitlines():
        name = line.split("\t", 1)[1]
        if name.startswith(prefix) and name.endswith(suffix):
            out.append(int(name[len(prefix):-len(suffix)]))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(paths.RAW))
    ap.add_argument("--warehouse", default="Warehouse_6")
    ap.add_argument("--seq", default="Seq_0")
    ap.add_argument("--keyframes", type=int, default=200)
    ap.add_argument("--gt-frames", type=int, default=300)
    args = ap.parse_args()
    raw = Path(args.raw)
    ref = f"Ref_Seq_Test/{args.warehouse}/{args.seq}"
    qry = f"Query_Seq_Test/{args.warehouse}/{args.seq}"
    q_ids = frame_ids(paths.CHANGESIM / "index_Query_Seq_Test.txt", f"{qry}/pose/", ".txt")
    run([("Ref_Seq_Test", f"{ref}/raw/poses.g2o")] + [("Query_Seq_Test", f"{qry}/pose/{i}.txt") for i in q_ids],
        raw, "poses")
    ids = sorted(int(l.split()[1]) for l in (raw / ref / "raw" / "poses.g2o").read_text().splitlines()
                 if l.startswith("VERTEX_SE3:QUAT"))
    pick = ids[:: max(1, len(ids) // args.keyframes)]
    q_pick = q_ids[:: max(1, len(q_ids) // args.keyframes)]
    gt = q_ids[:: max(1, len(q_ids) // args.gt_frames)]
    jobs = [("Ref_Seq_Test", f"{ref}/raw/{kind}/{i}.png") for i in pick for kind in ("rgb", "depth")]
    jobs += [("Query_Seq_Test", f"{qry}/{kind}/{i}.png") for i in q_pick for kind in ("rgb", "depth")]
    jobs += [("Query_Seq_Test", f"{qry}/{kind}") for i in gt
             for kind in (f"change_segmentation/{i}.png", f"semantic_segmentation/{i}.png", f"depth/{i}.png",
                          f"t0/idx/{i}.txt", f"t0/depth/{i}.png")]
    run(sorted(set(jobs)), raw, f"{args.warehouse}/{args.seq} frames")
    print("done", flush=True)


if __name__ == "__main__":
    main()
