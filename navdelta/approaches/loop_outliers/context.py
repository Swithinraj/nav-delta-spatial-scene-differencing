"""loop_outliers, stage 3: where each change is -- relations to unchanged objects around it.

Only the 1-2 photos already used to outline each end of a change are looked at (no detection over
the whole scene, so this stays fast):
  Florence-2 + SAM 2.1   the objects around it, lifted to 3D with the photo's depth
  landmark check         a neighbour counts only if the OTHER video still sees a surface where it is
                         (a thing that moved too is no reference)
  relations              "on" (resting on it), "next to" (gap below NEXT_TO_M), and for a moved
                         object the change of its distance to a landmark seen at both ends
The sentences in changes.json and CHANGE_REPORT.md are rewritten with them.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from navdelta import config, context
from navdelta.common.device import DEVICE, DTYPE
from navdelta.common.wording import It, is_, it, was, without_self
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"
OUTPUT = context.approach_dir("loop_outliers")
FLORENCE = ("florence-community/Florence-2-large", "4271c66b88cdbc05735372ec13b2360108de5317")
SAM = ("facebook/sam2.1-hiera-base-plus", "b7320756a13354e7530a63935656d35b2f91a290")
NOT_OBJECTS = {"house", "room", "building", "person", "human face", "man", "woman", "wall", "floor", "ceiling",
               "window", "door handle", "power plugs and sockets", "doorframe"}
CONF_MIN = 1.5
NEXT_TO_M = 0.30                # gap to a neighbour for "next to"
ON_GAP_M = 0.15                 # object bottom this close above a neighbour's top (with overlap): "on"
STATIC_SURFACE_MIN = 0.5        # share of a neighbour's points the other video still sees as surface
SURFACE_M, SURFACE_PER_M = 0.10, 0.0
SELF_OVERLAP = 0.3              # a box this much inside the changed object's outline is the object itself
DISTANCE_CHANGE_M = 0.25        # a landmark distance change worth a sentence
MAX_BOX_AREA = 0.6

PROFILE_SETTINGS = config.apply(globals(), "loop_outliers.context")


def upright_turns(cam2w: np.ndarray) -> int:
    up = cam2w[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    if abs(up[1]) >= abs(up[0]):
        return 0 if up[1] < 0 else 2
    return 1 if up[0] > 0 else 3


def rotate_box(box, turns, width, height):
    x0, y0, x1, y1 = box
    for _ in range(turns % 4):
        x0, y0, x1, y1 = y0, width - x1, y1, width - x0
        width, height = height, width
    return x0, y0, x1, y1


class Detector:
    def __init__(self) -> None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration, Sam2Model, Sam2Processor
        self.fp = AutoProcessor.from_pretrained(FLORENCE[0], revision=FLORENCE[1])
        self.fm = Florence2ForConditionalGeneration.from_pretrained(FLORENCE[0], revision=FLORENCE[1],
                                                                    dtype=DTYPE).to(DEVICE).eval()
        self.sp = Sam2Processor.from_pretrained(SAM[0], revision=SAM[1])
        self.sm = Sam2Model.from_pretrained(SAM[0], revision=SAM[1]).to(DEVICE).eval()

    @torch.no_grad()
    def objects(self, image: np.ndarray, cam2w: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """(name, mask) of every object Florence-2 names in the image, masks by SAM 2.1."""
        H, W = image.shape[:2]
        turns = upright_turns(cam2w)
        upright = Image.fromarray(np.ascontiguousarray(np.rot90(image, turns)))
        inputs = self.fp(text="<OD>", images=upright, return_tensors="pt").to(DEVICE, DTYPE)
        out = self.fm.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                               max_new_tokens=256, num_beams=3, do_sample=False)
        result = self.fp.post_process_generation(self.fp.batch_decode(out, skip_special_tokens=False)[0],
                                                 task="<OD>", image_size=upright.size)["<OD>"]
        boxes, names = [], []
        for box, name in zip(result["bboxes"], result["labels"]):
            name = " ".join(name.lower().split())
            box = rotate_box(tuple(box), (4 - turns) % 4, upright.size[0], upright.size[1])
            if name in NOT_OBJECTS or (box[2] - box[0]) * (box[3] - box[1]) > MAX_BOX_AREA * W * H:
                continue
            boxes.append([float(v) for v in box])
            names.append(name)
        if not boxes:
            return []
        sam_in = self.sp(images=Image.fromarray(image), input_boxes=[boxes], return_tensors="pt").to(DEVICE)
        sam_out = self.sm(**sam_in, multimask_output=False)
        masks = self.sp.post_process_masks(sam_out.pred_masks.cpu(), sam_in["original_sizes"].cpu())[0][:, 0].numpy()
        return list(zip(names, masks.astype(bool)))


def lift(mask: np.ndarray, scene: dict, k: int) -> np.ndarray:
    mask = ndimage.binary_erosion(mask, iterations=2) & (scene["conf"][k] > CONF_MIN)
    v, u = np.nonzero(mask)
    if len(u) < 30:
        return np.zeros((0, 3))
    z = scene["depth"][k][v, u]
    med = np.median(z)
    keep = np.abs(z - med) < 3 * 1.4826 * (np.median(np.abs(z - med)) + 1e-6) + 0.05
    u, v, z = u[keep], v[keep], z[keep]
    K, T = scene["K"][k], scene["cam2w"][k]
    p = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    return p @ T[:3, :3].T + T[:3, 3]


def surface_share(points: np.ndarray, scene: dict, frames: np.ndarray) -> float:
    """Share of the points another recording's depth still finds a surface at (among those it sees)."""
    seen = np.zeros(len(points), bool)
    surface = np.zeros(len(points), bool)
    for k in frames:
        w2c = np.linalg.inv(scene["cam2w"][k])
        cam = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        K = scene["K"][k]
        h, w = scene["depth"][k].shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2])
            v = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2])
        ok = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = np.where(ok, u, 0).astype(int)
        v = np.where(ok, v, 0).astype(int)
        d = scene["depth"][k][v, u]
        ok &= scene["conf"][k][v, u] > CONF_MIN
        seen |= ok & (d > z - SURFACE_M - SURFACE_PER_M * z)          # not hidden behind something closer
        surface |= ok & (np.abs(d - z) < SURFACE_M + SURFACE_PER_M * z)
    return float(surface.sum() / max(1, seen.sum()))


def gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(cKDTree(b).query(a[:: max(1, len(a) // 500)])[0].min())


def relations(obj: np.ndarray, neighbours: list[dict]) -> tuple[list[str], dict[str, float]]:
    """("on X" / "next to X") and the gap to every landmark, for one end of a change."""
    out, gaps = [], {}
    lo, hi = obj.min(0), obj.max(0)
    for n in neighbours:
        d = gap(obj, n["points"])
        gaps[n["name"]] = min(d, gaps.get(n["name"], np.inf))
        nlo, nhi = n["points"].min(0), n["points"].max(0)
        overlap = np.clip(np.minimum(hi[:2], nhi[:2]) - np.maximum(lo[:2], nlo[:2]), 0, None).prod()
        if 0 <= lo[2] - nhi[2] + 0.05 <= ON_GAP_M + 0.05 and overlap >= 0.3 * np.prod(np.maximum(hi[:2] - lo[:2], 0.05)):
            out.append(f"on the {n['name']}")
        elif d < NEXT_TO_M:
            out.append(f"next to the {n['name']}")
    return sorted(dict.fromkeys(out)), gaps


def join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def main() -> None:
    started = time.time()
    scene = dict(np.load(inside(RECON / "scene.npz")))
    changes = json.loads((OUTPUT / "changes.json").read_text(encoding="utf-8"))
    shapes = dict(np.load(OUTPUT / "objects.npz"))
    index = {str(f): k for k, f in enumerate(scene["frames"])}
    rec_a, rec_b = changes["recordings"]["before"], changes["recordings"]["after"]
    frames_of = {rec_a: np.flatnonzero(scene["owner"] == rec_a), rec_b: np.flatnonzero(scene["owner"] == rec_b)}
    other = {rec_a: rec_b, rec_b: rec_a}
    detector = Detector()
    changed_points = [p for p in shapes.values() if len(p)]
    for c in changes["changes"]:
        ends = {}
        for side, rec in (("before", rec_a), ("after", rec_b)):
            geo = c.get(side) or (c["object"] if c.get("object", {}).get("present_in") == rec else None)
            obj = shapes.get(f"{c['id']}_{side}")
            if geo is None or obj is None or not len(obj):
                continue
            neighbours = []
            for view in geo.get("views", []):
                k = index[view["frame"]]
                for name, mask in detector.objects(scene["rgb"][k], scene["cam2w"][k]):
                    pts = lift(mask, scene, k)
                    if len(pts) < 30 or name == c["label"]:
                        continue
                    if any(gap(pts, q) < 0.05 and len(pts) and np.mean(cKDTree(q).query(pts[::10])[0] < 0.1) > SELF_OVERLAP
                           for q in changed_points):
                        continue                                   # the changed object itself, or another change
                    if surface_share(pts, scene, frames_of[other[rec]]) < STATIC_SURFACE_MIN:
                        continue                                   # not there in the other video: no landmark
                    neighbours.append({"name": name, "points": pts})
            rel, gaps = relations(obj, neighbours)
            ends[side] = (without_self(rel, c["label"]), gaps)          # not "chair next to office chair"
        c["context"] = {side: {"relations": r, "landmark_gaps_m": {k: round(v, 2) for k, v in g.items()}}
                        for side, (r, g) in ends.items()}
        sentences = []
        if c["type"] == "moved" and "before" in ends and "after" in ends:
            before, after = ends["before"][0], ends["after"][0]
            if before or after:
                label = c["label"]
                sentences.append(f"Before, {it(label)} {was(label)} "
                                 f"{join(before) if before else 'away from other furniture'}; now {it(label)} "
                                 f"{is_(label)} {join(after) if after else 'away from other furniture'}.")
            shared = set(ends["before"][1]) & set(ends["after"][1])
            moves = sorted(((ends["after"][1][n] - ends["before"][1][n], n) for n in shared), key=lambda t: -abs(t[0]))
            for delta, name in moves[:2]:
                if abs(delta) >= DISTANCE_CHANGE_M:
                    sentences.append(f"{It(c['label'])} {is_(c['label'])} now "
                                     f"{'closer to' if delta < 0 else 'further from'} the {name}: "
                                     f"{ends['after'][1][name]:.2f} m, compared with {ends['before'][1][name]:.2f} m before.")
        else:
            for side in ends:
                if ends[side][0]:
                    verb = "stood" if side == "before" else ("stand" if it(c["label"]) == "they" else "stands")
                    sentences.append(f"{It(c['label'])} {verb} {join(ends[side][0])}.")
        c["description"] = " ".join([c["sentence"], *sentences, c["evidence_text"]])
        print(f"  {c['id']:<5} {c['description']}", flush=True)
    import hashlib
    changes.pop("result_sha256", None)
    changes["result_sha256"] = hashlib.sha256(json.dumps(changes, sort_keys=True, default=str).encode()).hexdigest()
    (OUTPUT / "changes.json").write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# What changed between the two recordings", "", changes["summary"], ""] + \
            [f"- **{c['id']}** ({c['type']}): {c['description']}" for c in changes["changes"]]
    (OUTPUT / "CHANGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"context for {len(changes['changes'])} changes ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
