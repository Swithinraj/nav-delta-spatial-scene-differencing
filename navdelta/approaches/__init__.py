"""Change-detection approaches.

Every approach is a sub-package with:
  NAME, TITLE                 identifiers used by the CLI (--approach NAME)
  STAGES                      [(module name, title), ...] run in order; each module has main()
  result_hashes(output_dir)   {stage: sha256} for `navdelta verify`
  summary(output_dir)         one-paragraph summary printed after a run

Approaches share the alignment (<outputs>/alignment), the frame utilities in
navdelta.common, the viewer template in navdelta/viewer, and write their results
to <outputs>/<NAME>/, ending with changes.json in the common schema
(docs/changes-schema.md). Add a new approach as a new sub-package and register it below.
"""

from __future__ import annotations

import importlib

REGISTRY = {
    "voxel_diff": "navdelta.approaches.voxel_diff",
}
DEFAULT = "voxel_diff"


def load(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"Unknown approach {name!r}; available: {', '.join(REGISTRY)}")
    return importlib.import_module(REGISTRY[name])
