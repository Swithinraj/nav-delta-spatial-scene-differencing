"""rgbd_fusion, stage 4: outline and name the outlier changes (approach 3's segment stage).

SAM 2.1 prompted with the outlier pixels, lifted with sensor depth; SigLIP 2 names. Weak evidence
is not dropped here: the fuse stage keeps it when the scene graph corroborates it. Writes
outlier_objects.json / .npz.
"""
from __future__ import annotations

from navdelta import config, context
from navdelta.approaches.loop_outliers import segment as base

OVERRIDES = {
    "ALONE_MIN_MATCHES": 0,       # the corroboration rule moves to the fuse stage
    "ALONE_MIN_PAIRS": 0,
}
vars(base).update(OVERRIDES)
base.RECON = context.OUTPUTS / "recon_rgbd"
base.OUTPUT = context.approach_dir("rgbd_fusion")
base.CHANGES_FILE, base.OBJECTS_FILE, base.REPORT_FILE = "outlier_objects.json", "outlier_objects.npz", None
PROFILE_SETTINGS = config.apply(vars(base), "rgbd_fusion.segment")


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
