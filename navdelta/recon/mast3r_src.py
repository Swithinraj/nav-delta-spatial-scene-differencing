"""MASt3R source code at a pinned commit, fetched on first use (no git needed).

MASt3R (https://github.com/naver/mast3r) is not a pip package: its code and its two nested
repositories (dust3r, croco) are downloaded once as GitHub archives at fixed commits into the model
cache and put on sys.path. Its weights come from the Hugging Face Hub like every other model.

Licence: MASt3R, DUSt3R and CroCo are CC BY-NC-SA 4.0 (non-commercial). See docs/licences.md.
"""
from __future__ import annotations

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from navdelta import context

REPOS = {   # name: (GitHub repository, commit, folder inside the assembled tree)
    "mast3r": ("naver/mast3r", "f5209afc300cec36239a7ac992263f36847bbba0", ""),
    "dust3r": ("naver/dust3r", "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12", "dust3r"),
    "croco": ("naver/croco", "d7de0705845239092414480bd829228723bf20de", "dust3r/croco"),
}
WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def source_dir() -> Path:
    return context.MODEL_CACHE / f"mast3r-src-{REPOS['mast3r'][1][:12]}"


def fetch() -> Path:
    target = source_dir()
    if (target / "mast3r" / "model.py").is_file():
        return target
    tmp = target.with_name(target.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    for name, (repo, commit, inside) in REPOS.items():
        url = f"https://github.com/{repo}/archive/{commit}.zip"
        print(f"  fetching {repo} @ {commit[:12]}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        top = archive.namelist()[0].split("/")[0]
        dest = tmp / inside
        for member in archive.infolist():
            rel = member.filename[len(top) + 1:]
            if not rel or member.is_dir():
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(member))
    tmp.rename(target)
    return target


def load():
    """Import MASt3R (fetching its pinned source on first use); returns the modules this package uses."""
    root = fetch()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import mast3r.utils.path_to_dust3r  # noqa: F401  (puts dust3r on the path)
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from mast3r.fast_nn import fast_reciprocal_NNs
    from mast3r.model import AsymmetricMASt3R
    return AsymmetricMASt3R, inference, load_images, fast_reciprocal_NNs
