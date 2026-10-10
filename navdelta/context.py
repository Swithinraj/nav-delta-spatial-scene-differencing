"""Run configuration shared by every stage.

The CLI (``navdelta``) sets these through environment variables before any stage
module is imported, so each stage reads one consistent configuration:

  NAVDELTA_DATA         folder holding the two recordings      (default: <repo>/data)
  NAVDELTA_OUTPUTS      folder for every generated file        (default: <repo>/outputs)
  NAVDELTA_BEFORE       name of the first recording  (reference frame)
  NAVDELTA_AFTER        name of the second recording
  NAVDELTA_MODEL_CACHE  Hugging Face cache for model weights   (default: <repo>/.hf_cache)

If the recording names are not given, the two recordings found in the data
folder are used in sorted order (the first is "before").
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _folder(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable) or default).expanduser().resolve()


DATA = _folder("NAVDELTA_DATA", REPO / "data")
OUTPUTS = _folder("NAVDELTA_OUTPUTS", REPO / "outputs")
MODEL_CACHE = _folder("NAVDELTA_MODEL_CACHE", REPO / ".hf_cache")
ALIGNMENT = OUTPUTS / "alignment"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def discover_recordings(data: Path = DATA) -> list[str]:
    """Recording names in a data folder: every <name>_imgs folder with keyframes."""
    return sorted(path.name[:-len("_imgs")] for path in data.glob("*_imgs") if (path / "keyframes").is_dir())


def recordings() -> tuple[str, str]:
    before, after = os.environ.get("NAVDELTA_BEFORE"), os.environ.get("NAVDELTA_AFTER")
    if before and after:
        return before, after
    found = discover_recordings()
    if len(found) != 2:
        raise RuntimeError(f"Expected exactly two recordings in {DATA} (found {found}); "
                           "pass --before and --after to choose them.")
    return found[0], found[1]


RECORDING_ERROR = None
try:
    BEFORE, AFTER = recordings()
except RuntimeError as error:   # reported by the CLI before any stage runs
    BEFORE, AFTER, RECORDING_ERROR = None, None, str(error)
SESSIONS = (BEFORE, AFTER)


def inside(path: Path) -> Path:
    """Resolve a path and refuse anything outside the repository, data, outputs or model cache."""
    resolved = Path(path).resolve()
    for root in (REPO, DATA, OUTPUTS, MODEL_CACHE):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise RuntimeError(f"Path outside the configured folders: {resolved}")


def approach_dir(name: str) -> Path:
    return OUTPUTS / name


def data_relative(path: Path) -> str:
    """Path of a data file as stored in outputs: relative to the data folder, '/' separators."""
    return str(Path(path).resolve().relative_to(DATA)).replace("\\", "/")
