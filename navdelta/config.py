"""Scale profiles: one set of tuning values for a room, one for a large space (warehouse, hall).

  NAVDELTA_PROFILE = auto (default) | room | large_space | path/to/custom.toml

Every stage keeps its room values as documented module constants. A profile (navdelta/configs/
<name>.toml) lists, per stage, the constants it sets; apply() writes them over the module's
constants before the stage runs. room.toml repeats today's values (`python -m navdelta.config
--check` proves it), so the room results are unchanged; large_space.toml changes only what
depends on the size of the space: voxel size, depth range, object sizes, distances, tolerances.

"auto" measures the space from the "before" recording's camera path: a path that spans more
than LARGE_SPACE_SPAN_M is a large space, otherwise a room.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

PROFILES = Path(__file__).resolve().parent / "configs"
REPO = Path(__file__).resolve().parents[1]
LARGE_SPACE_SPAN_M = 10.0
CAMERA_FOLDERS = ("corrected_cameras", "cameras")


def camera_span(data: Path, recording: str) -> float | None:
    """Diagonal of the horizontal bounding box of a recording's camera positions (metres)."""
    for folder in CAMERA_FOLDERS:
        files = sorted((Path(data) / f"{recording}_imgs" / "keyframes" / folder).glob("*.json"))
        if files:
            break
    else:
        return None
    xs, zs = [], []
    for f in files[:: max(1, len(files) // 400)]:
        c = json.loads(f.read_text())
        xs.append(c["t_03"])
        zs.append(c["t_23"])                     # ARKit / Polycam: Y is up, X-Z is the floor plane
    return float(((max(xs) - min(xs)) ** 2 + (max(zs) - min(zs)) ** 2) ** 0.5)


def select(name: str | None = None, data: Path | None = None, recording: str | None = None) -> dict:
    """The profile to use: name, file, why, and its values."""
    name = name or os.environ.get("NAVDELTA_PROFILE") or "auto"
    reason = "chosen explicitly"
    if name == "auto":
        span = camera_span(data, recording) if data is not None and recording else None
        if span is None:
            name, reason = "room", "auto: no camera poses found, room values"
        else:
            name = "large_space" if span > LARGE_SPACE_SPAN_M else "room"
            reason = f"auto: camera path spans {span:.1f} m ({'>' if span > LARGE_SPACE_SPAN_M else '<='} " \
                     f"{LARGE_SPACE_SPAN_M:.0f} m)"
    path = Path(name) if name.endswith(".toml") else PROFILES / f"{name}.toml"
    if not path.is_file():
        raise SystemExit(f"Unknown profile {name!r}: no {path}. Profiles: "
                         f"{', '.join(p.stem for p in sorted(PROFILES.glob('*.toml')))}")
    raw = path.read_bytes()
    return {"name": path.stem, "file": str(path), "reason": reason, "sha256": hashlib.sha256(raw).hexdigest(),
            "values": tomllib.loads(raw.decode("utf-8"))}


_ACTIVE: dict | None = None


def active() -> dict:
    """The profile for this run (resolved once; the CLI and stage modules share it)."""
    global _ACTIVE
    if _ACTIVE is None:
        from navdelta import context
        _ACTIVE = select(None, context.DATA, context.BEFORE)
    return _ACTIVE


def section(values: dict, dotted: str) -> dict:
    for key in dotted.split("."):
        values = values.get(key, {})
    return values


def apply(namespace: dict, dotted: str, chosen: dict | None = None) -> dict:
    """Overwrite a module's constants with the profile's values for `dotted` (e.g.
    "voxel_diff.change_map"). Unknown names are an error, so a typo cannot be silently ignored."""
    chosen = chosen or active()
    applied = {}
    for key, value in section(chosen["values"], dotted).items():
        if key not in namespace:
            raise KeyError(f"profile {chosen['name']}: [{dotted}] sets {key}, which the stage does not have")
        if isinstance(namespace[key], tuple):
            value = tuple(value)
        namespace[key] = value
        applied[key] = value
    return applied


def summary(chosen: dict | None = None) -> dict:
    """Name, file (relative to the repository, so results carry no local paths), reason, sha256."""
    chosen = chosen or active()
    out = {k: chosen[k] for k in ("name", "file", "reason", "sha256")}
    try:
        out["file"] = Path(out["file"]).resolve().relative_to(REPO).as_posix()
    except ValueError:
        out["file"] = Path(out["file"]).name
    return out


# ---------------------------------------------------------------- check: room.toml == code
MODULE_FILES = {"recon": REPO / "navdelta" / "recon",
                "voxel_diff": REPO / "navdelta" / "approaches" / "voxel_diff",
                "scene_graph": REPO / "navdelta" / "approaches" / "scene_graph",
                "loop_outliers": REPO / "navdelta" / "approaches" / "loop_outliers",
                "rgbd_fusion": REPO / "navdelta" / "approaches" / "rgbd_fusion"}


def module_defaults(path: Path) -> dict:
    """A stage's constants. A wrapper stage (`from navdelta... import <stage> as base`) has the
    wrapped stage's constants, with its own OVERRIDES on top."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("navdelta"):
            for alias in node.names:
                if alias.asname == "base":
                    out.update(module_defaults(REPO.joinpath(*node.module.split(".")) / f"{alias.name}.py"))
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for t in targets:
            names = [t] if isinstance(t, ast.Name) else list(t.elts) if isinstance(t, ast.Tuple) else []
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            values = [value] if len(names) == 1 else list(value) if isinstance(value, tuple) else []
            for n, v in zip(names, values):
                if isinstance(n, ast.Name):
                    out[n.id] = v
    out.update(out.pop("OVERRIDES", {}))
    return out


def check() -> int:
    """room.toml must equal the module defaults; large_space.toml may only set known constants."""
    problems = 0
    for name in ("room", "large_space"):
        values = select(name)["values"]
        for approach, stages in values.items():
            if not isinstance(stages, dict) or approach not in MODULE_FILES:
                continue
            for stage, settings in stages.items():
                path = MODULE_FILES[approach] / f"{stage}.py"
                if not path.is_file():
                    print(f"[{name}] [{approach}.{stage}]: no stage {path.relative_to(REPO)}")
                    problems += 1
                    continue
                defaults = module_defaults(path)
                for key, value in settings.items():
                    if key not in defaults:
                        print(f"[{name}] {approach}.{stage}.{key}: not a constant of {path.name}")
                        problems += 1
                    elif name == "room" and (tuple(value) if isinstance(value, list) else value) != defaults[key]:
                        print(f"[room] {approach}.{stage}.{key} = {value!r}, code has {defaults[key]!r}")
                        problems += 1
    print("profiles OK: room.toml equals the code defaults" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(check())
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps({k: v for k, v in select(None, data, sys.argv[2] if len(sys.argv) > 2 else None).items()
                      if k != "values"}, indent=2))
