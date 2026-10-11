"""navdelta command line: one deterministic entry point for the whole pipeline.

  navdelta run          the shared stage the approach needs + change detection + descriptions + viewer
  navdelta align        LiDAR + tracking alignment only (shared stage of voxel_diff)
  navdelta reconstruct  RGB + tracking reconstruction only (shared stage of scene_graph, loop_outliers)
  navdelta reconstruct-rgbd  the same with the recording's depth maps (shared stage of rgbd_fusion)
  navdelta detect       an approach's own stages only (needs its shared stage's outputs)
  navdelta verify      run twice and require identical result hashes for every stage
  navdelta check-data  check that a data folder holds two usable recordings
  navdelta stages      list the stages of the shared stages and of each approach
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _configure(args: argparse.Namespace) -> None:
    """Export the run configuration before any stage module (or navdelta.context) is imported."""
    os.environ["NAVDELTA_DATA"] = str(Path(args.data).resolve())
    os.environ["NAVDELTA_OUTPUTS"] = str(Path(args.outputs).resolve())
    if args.model_cache:
        os.environ["NAVDELTA_MODEL_CACHE"] = str(Path(args.model_cache).resolve())
    if args.before or args.after:
        if not (args.before and args.after):
            raise SystemExit("Pass both --before and --after, or neither.")
        os.environ["NAVDELTA_BEFORE"], os.environ["NAVDELTA_AFTER"] = args.before, args.after
    if getattr(args, "cpu", False):
        os.environ["NAVDELTA_DEVICE"] = "cpu"
    if getattr(args, "profile", None):
        os.environ["NAVDELTA_PROFILE"] = args.profile


def _shared(args: argparse.Namespace) -> str:
    """The shared stage this command runs or uses: "alignment" (voxel_diff), "recon" (scene_graph,
    loop_outliers) or "recon_rgbd" (rgbd_fusion)."""
    from navdelta import approaches
    if args.command in ("align", "reconstruct", "reconstruct-rgbd"):
        return {"align": "alignment", "reconstruct": "recon", "reconstruct-rgbd": "recon_rgbd"}[args.command]
    return getattr(approaches.load(args.approach), "REQUIRES", "alignment")


def _stage_plan(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """[(module path, stage name, title)] for the requested command."""
    from navdelta import alignment, approaches, recon
    plan = []
    shared = _shared(args)
    if args.command in ("run", "align", "reconstruct", "reconstruct-rgbd", "verify") \
            and not getattr(args, "skip_alignment", False):
        module, stages = {"alignment": (alignment, alignment.STAGES), "recon": (recon, recon.STAGES),
                          "recon_rgbd": (recon, recon.RGBD_STAGES)}[shared]
        plan += [(f"{module.__name__}.{name}", f"{shared}.{name}", title) for name, title in stages]
    if args.command in ("run", "detect", "verify"):
        approach = approaches.load(args.approach)
        plan += [(f"{approaches.REGISTRY[args.approach]}.{name}", name, title) for name, title in approach.STAGES]
    names = [name for _, name, _ in plan]

    def position(stage: str) -> int:
        matches = [i for i, name in enumerate(names) if name == stage or name.endswith("." + stage)]
        if not matches:
            raise SystemExit(f"Unknown stage {stage!r}; stages: {', '.join(names)}")
        return matches[0]

    first = position(args.from_stage) if getattr(args, "from_stage", None) else 0
    last = position(args.to_stage) if getattr(args, "to_stage", None) else len(plan) - 1
    return plan[first:last + 1]


def _gpu_peak_gb() -> float | None:
    torch = sys.modules.get("torch")             # only if a stage already loaded it
    if torch is not None and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        return round(peak, 2)
    return None


def _run_plan(plan: list[tuple[str, str, str]]) -> list[dict]:
    from navdelta import config
    chosen = config.active()
    print(f"Profile: {chosen['name']} ({chosen['reason']})", flush=True)
    timings = []
    for index, (module_path, name, title) in enumerate(plan, start=1):
        print(f"\n=== [{index}/{len(plan)}] {title}  ({name})", flush=True)
        started = time.time()
        module = importlib.import_module(module_path)
        module.main()
        timings.append({"stage": name, "seconds": round(time.time() - started, 1), "gpu_peak_gb": _gpu_peak_gb()})
    return timings


def _hashes(args: argparse.Namespace) -> dict[str, str | None]:
    from navdelta import alignment, approaches, context, recon
    shared = _shared(args)
    hashes = {}
    if not (args.command == "verify" and getattr(args, "skip_alignment", False)):   # verify: only what ran
        hashes = (alignment.result_hashes(context.ALIGNMENT) if shared == "alignment"
                  else recon.result_hashes(context.OUTPUTS / shared))
    if args.command in ("run", "detect", "verify"):
        approach = approaches.load(args.approach)
        hashes.update({f"{approach.NAME}/{k}": v for k, v in approach.result_hashes(
            context.approach_dir(approach.NAME)).items()})
    return hashes


def _write_log(args: argparse.Namespace, timings: list[dict], hashes: dict, extra: dict | None = None) -> Path:
    from navdelta import __version__, config, context
    log = {
        "navdelta_version": __version__,
        "command": " ".join(["navdelta", *sys.argv[1:]]),
        "data": str(context.DATA), "outputs": str(context.OUTPUTS),
        "recordings": {"before": context.BEFORE, "after": context.AFTER},
        "approach": getattr(args, "approach", None),
        "python": platform.python_version(), "platform": platform.platform(),
        "profile": config.summary(), "stages": timings,
        "total_seconds": round(sum(t["seconds"] for t in timings), 1),
        "result_hashes": hashes, **(extra or {}),
    }
    context.OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = context.OUTPUTS / "run_log.json"
    path.write_text(json.dumps(log, indent=2))
    # every run is also kept: one line per run, so speed and results can be compared over time
    with open(context.OUTPUTS / "runs.jsonl", "a", encoding="utf-8") as history:
        history.write(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **log}) + "\n")
    return path


def _check(args: argparse.Namespace) -> bool:
    from navdelta import context
    from navdelta.common.data_check import check_data, check_rgb_data
    shared = _shared(args)
    if shared == "alignment":
        return check_data(context.DATA, list(context.SESSIONS))
    return check_rgb_data(context.DATA, list(context.SESSIONS), depth=shared == "recon_rgbd")


def cmd_pipeline(args: argparse.Namespace) -> None:
    _configure(args)
    from navdelta import approaches, context
    if args.command != "detect":
        if context.RECORDING_ERROR or not _check(args):
            raise SystemExit(context.RECORDING_ERROR or "The data folder is not usable; see docs/data-format.md.")
    plan = _stage_plan(args)
    started = time.time()
    timings = _run_plan(plan)
    hashes = _hashes(args)
    log = _write_log(args, timings, hashes)
    print(f"\nDone in {time.time() - started:.0f} s. Run log: {log}")
    if args.command in ("run", "detect"):
        approach = approaches.load(args.approach)
        output = context.approach_dir(approach.NAME)
        print(f"\n{approach.summary(output) or ''}\n")
        print(f"  Report:  {output / 'CHANGE_REPORT.md'}")
        print(f"  Data:    {output / 'changes.json'}")
        print(f"  Viewer:  {output / 'viewer' / 'index.html'}  (open in a browser)")


def cmd_verify(args: argparse.Namespace) -> None:
    os.environ["NAVDELTA_RECOMPUTE"] = "1"          # both runs compute everything: no cache is reused
    _configure(args)
    from navdelta import context
    if context.RECORDING_ERROR or not _check(args):
        raise SystemExit(context.RECORDING_ERROR or "The data folder is not usable; see docs/data-format.md.")
    plan = _stage_plan(args)
    runs = []
    for attempt in (1, 2):
        print(f"\n######## verification run {attempt}/2 ########", flush=True)
        timings = _run_plan(plan)
        runs.append({"timings": timings, "hashes": _hashes(args)})
    first, second = runs[0]["hashes"], runs[1]["hashes"]
    rows = [(key, first.get(key), second.get(key)) for key in sorted(set(first) | set(second))]
    same = all(a is not None and a == b for _, a, b in rows)
    print("\nStage result hashes (run 1 vs run 2):")
    for key, a, b in rows:
        print(f"  {'same' if a == b and a else 'DIFF':<4}  {key:<52} {str(a)[:16]}  {str(b)[:16]}")
    _write_log(args, runs[1]["timings"], second,
               {"verification": {"reproducible": same, "run_1": first, "run_2": second}})
    print("\nREPRODUCIBLE: every stage produced identical results twice." if same
          else "\nNOT REPRODUCIBLE: see the DIFF rows above.")
    if not same:
        raise SystemExit(1)


def cmd_check_data(args: argparse.Namespace) -> None:
    _configure(args)
    from navdelta.common.data_check import check_data, check_rgb_data
    from navdelta.context import discover_recordings
    data = Path(args.data).resolve()
    names = [args.before, args.after] if args.before else discover_recordings(data)
    shared = _shared(args)
    ok = check_data(data, names) if shared == "alignment" else check_rgb_data(data, names, depth=shared == "recon_rgbd")
    raise SystemExit(0 if ok else 1)


def cmd_stages(args: argparse.Namespace) -> None:
    from navdelta import alignment, approaches, recon
    for label, stages in (("alignment (shared stage of voxel_diff; LiDAR + tracking)", alignment.STAGES),
                          ("recon (shared stage of scene_graph and loop_outliers; RGB + tracking)", recon.STAGES),
                          ("recon_rgbd (shared stage of rgbd_fusion; RGB-D + tracking)", recon.RGBD_STAGES)):
        print(f"{label}:")
        for name, title in stages:
            print(f"  {name:<12} {title}")
        print()
    for key in approaches.REGISTRY:
        approach = approaches.load(key)
        print(f"approach {approach.NAME} (needs {approach.REQUIRES}): {approach.TITLE}")
        for name, title in approach.STAGES:
            print(f"  {name:<12} {title}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navdelta", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=os.environ.get("NAVDELTA_DATA", REPO / "data"),
                        help="folder with the two recordings (default: ./data)")
    common.add_argument("--outputs", default=os.environ.get("NAVDELTA_OUTPUTS", REPO / "outputs"),
                        help="folder for all generated files (default: ./outputs)")
    common.add_argument("--before", help="name of the first recording (default: first of the two found)")
    common.add_argument("--after", help="name of the second recording (default: second of the two found)")
    common.add_argument("--model-cache", default=os.environ.get("NAVDELTA_MODEL_CACHE"),
                        help="Hugging Face cache for model weights (default: ./.hf_cache)")
    common.add_argument("--profile", default=os.environ.get("NAVDELTA_PROFILE"),
                        help="scale profile: room | large_space | auto (default: auto, from the camera path) "
                             "or a .toml file; see navdelta/configs/")
    pipeline = argparse.ArgumentParser(add_help=False)
    pipeline.add_argument("--approach", default=os.environ.get("NAVDELTA_APPROACH", "voxel_diff"),
                          help="voxel_diff | scene_graph | loop_outliers | rgbd_fusion (default: voxel_diff, or "
                               "$NAVDELTA_APPROACH)")
    pipeline.add_argument("--from", dest="from_stage", help="start at this stage (see `navdelta stages`)")
    pipeline.add_argument("--to", dest="to_stage", help="stop after this stage")
    pipeline.add_argument("--cpu", action="store_true", help="run the model stages on the CPU (slower)")

    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", parents=[common, pipeline], help="the whole pipeline")
    run.add_argument("--skip-alignment", "--skip-shared", dest="skip_alignment", action="store_true",
                     help="reuse the shared stage's outputs (<outputs>/alignment, recon or recon_rgbd)")
    commands.add_parser("align", parents=[common], help="LiDAR + tracking alignment only (voxel_diff)")
    commands.add_parser("reconstruct", parents=[common], help="RGB + tracking reconstruction only (scene_graph, loop_outliers)")
    commands.add_parser("reconstruct-rgbd", parents=[common], help="RGB-D + tracking reconstruction only (rgbd_fusion)")
    commands.add_parser("detect", parents=[common, pipeline],
                        help="an approach's own stages only (after its shared stage: align / reconstruct / "
                             "reconstruct-rgbd)")
    verify = commands.add_parser("verify", parents=[common, pipeline],
                                 help="run twice and require identical results")
    verify.add_argument("--skip-alignment", "--skip-shared", dest="skip_alignment", action="store_true",
                        help="verify the approach's own stages only, on the existing shared stage")
    check = commands.add_parser("check-data", parents=[common], help="check the data folder")
    check.add_argument("--approach", default=os.environ.get("NAVDELTA_APPROACH", "voxel_diff"),
                       help="check for this approach's inputs")
    commands.add_parser("stages", help="list all stages")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {"run": cmd_pipeline, "align": cmd_pipeline, "reconstruct": cmd_pipeline,
                "reconstruct-rgbd": cmd_pipeline, "detect": cmd_pipeline,
                "verify": cmd_verify, "check-data": cmd_check_data, "stages": cmd_stages}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
