"""navdelta command line: one deterministic entry point for the whole pipeline.

  navdelta run         alignment + change detection + labels + descriptions + viewer
  navdelta align       alignment only
  navdelta detect      change pipeline only (needs an existing alignment)
  navdelta verify      run twice and require identical result hashes for every stage
  navdelta check-data  check that a data folder holds two usable recordings
  navdelta stages      list the stages of the alignment and of each approach
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


def _stage_plan(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """[(module path, stage name, title)] for the requested command."""
    from navdelta import alignment, approaches
    plan = []
    if args.command in ("run", "align", "verify") and not getattr(args, "skip_alignment", False):
        plan += [(f"navdelta.alignment.{name}", f"alignment.{name}", title) for name, title in alignment.STAGES]
    if args.command in ("run", "detect", "verify"):
        approach = approaches.load(args.approach)
        plan += [(f"{approaches.REGISTRY[args.approach]}.{name}", name, title) for name, title in approach.STAGES]
    if getattr(args, "from_stage", None):
        names = [name for _, name, _ in plan]
        matches = [i for i, name in enumerate(names) if name == args.from_stage or name.endswith("." + args.from_stage)]
        if not matches:
            raise SystemExit(f"Unknown stage {args.from_stage!r}; stages: {', '.join(names)}")
        plan = plan[matches[0]:]
    return plan


def _run_plan(plan: list[tuple[str, str, str]]) -> list[dict]:
    timings = []
    for index, (module_path, name, title) in enumerate(plan, start=1):
        print(f"\n=== [{index}/{len(plan)}] {title}  ({name})", flush=True)
        started = time.time()
        module = importlib.import_module(module_path)
        module.main()
        timings.append({"stage": name, "seconds": round(time.time() - started, 1)})
    return timings


def _hashes(args: argparse.Namespace) -> dict[str, str | None]:
    from navdelta import alignment, approaches, context
    hashes = alignment.result_hashes(context.ALIGNMENT)
    if args.command in ("run", "detect", "verify"):
        approach = approaches.load(args.approach)
        hashes.update({f"{approach.NAME}/{k}": v for k, v in approach.result_hashes(
            context.approach_dir(approach.NAME)).items()})
    return hashes


def _write_log(args: argparse.Namespace, timings: list[dict], hashes: dict, extra: dict | None = None) -> Path:
    from navdelta import __version__, context
    log = {
        "navdelta_version": __version__,
        "command": " ".join(["navdelta", *sys.argv[1:]]),
        "data": str(context.DATA), "outputs": str(context.OUTPUTS),
        "recordings": {"before": context.BEFORE, "after": context.AFTER},
        "approach": getattr(args, "approach", None),
        "python": platform.python_version(), "platform": platform.platform(),
        "stages": timings, "result_hashes": hashes, **(extra or {}),
    }
    context.OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = context.OUTPUTS / "run_log.json"
    path.write_text(json.dumps(log, indent=2))
    return path


def cmd_pipeline(args: argparse.Namespace) -> None:
    _configure(args)
    from navdelta import approaches, context
    from navdelta.common.data_check import check_data
    if args.command != "detect":
        if context.RECORDING_ERROR or not check_data(context.DATA, list(context.SESSIONS)):
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
    _configure(args)
    from navdelta import context
    from navdelta.common.data_check import check_data
    if context.RECORDING_ERROR or not check_data(context.DATA, list(context.SESSIONS)):
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
    from navdelta.common.data_check import check_data
    from navdelta.context import discover_recordings
    data = Path(args.data).resolve()
    names = [args.before, args.after] if args.before else discover_recordings(data)
    raise SystemExit(0 if check_data(data, names) else 1)


def cmd_stages(args: argparse.Namespace) -> None:
    from navdelta import alignment, approaches
    print("alignment (shared by every approach):")
    for name, title in alignment.STAGES:
        print(f"  {name:<12} {title}")
    for key in approaches.REGISTRY:
        approach = approaches.load(key)
        print(f"\napproach {approach.NAME}: {approach.TITLE}")
        for name, title in approach.STAGES:
            print(f"  {name:<12} {title}")


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
    pipeline = argparse.ArgumentParser(add_help=False)
    pipeline.add_argument("--approach", default="voxel_diff", help="change-detection approach (default: voxel_diff)")
    pipeline.add_argument("--from", dest="from_stage", help="start at this stage (see `navdelta stages`)")
    pipeline.add_argument("--cpu", action="store_true", help="run the model stages on the CPU (slower)")

    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", parents=[common, pipeline], help="the whole pipeline")
    run.add_argument("--skip-alignment", action="store_true", help="reuse <outputs>/alignment")
    commands.add_parser("align", parents=[common], help="alignment only")
    commands.add_parser("detect", parents=[common, pipeline], help="change pipeline only (after `align`)")
    verify = commands.add_parser("verify", parents=[common, pipeline],
                                 help="run twice and require identical results")
    verify.add_argument("--skip-alignment", action="store_true", help="verify the change pipeline only")
    commands.add_parser("check-data", parents=[common], help="check the data folder")
    commands.add_parser("stages", help="list all stages")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {"run": cmd_pipeline, "align": cmd_pipeline, "detect": cmd_pipeline,
                "verify": cmd_verify, "check-data": cmd_check_data, "stages": cmd_stages}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
