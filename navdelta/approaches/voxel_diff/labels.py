"""Step 3: open-vocabulary labels for every changed object, with off-the-shelf open models.

No model is trained or fine-tuned and no data is labelled by hand. For each
object end of a Step 2 event (old and new position of a move, removed, added,
surface change):

  1. Views: the object's voxels are projected into every keyframe of the
     recording it exists in; frames where depth confirms the surface are ranked,
     and up to VIEWS_PER_OBJECT diverse viewpoints are kept.
  2. Mask: SAM 2.1 (Apache-2.0) segments the object in each view from a box +
     point prompt built from the projected, depth-confirmed voxels.
  3. Upright: each crop is rotated by a multiple of 90 deg so world "up" (known
     from the aligned pose) points up -- the phone video is mostly sideways.
  4. Labels: Florence-2-large (MIT) names the masked region in context
     (<REGION_TO_CATEGORY>) and describes it (<REGION_TO_DESCRIPTION>), per view.
  5. Cross-check: SigLIP 2 (Apache-2.0) scores the tight crops against the 200
     class names of the public ScanNet200 benchmark plus every Florence
     proposal; scores are pooled over views.
  6. Decision: a Florence-2 label backed by a strict majority of the views is
     confirmed when one of SigLIP's top 3 names the same object (identical, or
     same head noun such as "chair" / "office chair"; the more specific name is
     kept). Otherwise SigLIP's best candidate is used; it counts as confident
     only if Florence-2 proposed a compatible name in at least one view.

Models run one at a time in fp16 to fit a 4 GB GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

from navdelta.approaches.voxel_diff.paths import AFTER, BEFORE, DATA, OUTPUT, RESOURCES, SEMANTICS, SESSIONS, inside_workspace, out_rel

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from PIL import Image

from navdelta.common.device import DEVICE, DTYPE, PRECISION  # noqa: E402
from navdelta.common.frames import load_session, project  # noqa: E402
PIPELINE_VERSION = "1.0.0"

SAM_MODEL = "facebook/sam2.1-hiera-base-plus"
FLORENCE_MODEL = "florence-community/Florence-2-large"
SIGLIP_MODEL = "google/siglip2-so400m-patch14-384"
# Exact model snapshots (Hugging Face commit hashes): weights are downloaded on
# first use and every later run loads the same files.
SAM_REVISION = "b7320756a13354e7530a63935656d35b2f91a290"
FLORENCE_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
SIGLIP_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"

VIEWS_PER_OBJECT = 5
MIN_VIEW_FRACTION = 0.3          # a view must show >= 30% as many surface voxels as the best one
MIN_SURFACE_POINTS = 5
VIEW_MIN_CAMERA_DISTANCE_M = 0.30
BORDER_PX = 6
SAM_POINTS = 3
MIN_MASK_SCORE = 0.3
CONTEXT_PAD = 0.30               # Florence sees the object plus 30% context on each side
TIGHT_PAD = 0.08                 # SigLIP sees the object itself
MIN_CROP_PX = 64
SIGLIP_TOP_K_SUPPORT = 3
# Photo cross-check: the same spot is also labelled in the OTHER recording's photos
# (frames whose depth sees through the spot). If both recordings name the same
# object there, the change is depth-only (thin/see-through material, a slight
# shift or local misalignment), not a removal or an addition.
PHOTO_CHECK_VIEWS = 3
SIGLIP_TEMPLATE = "a photo of a {}."

# Vocabulary for the SigLIP cross-check: the 200 class names of the public
# ScanNet200 indoor benchmark, read as data (never executed) from the file
# downloaded from the official repository. Using a published list keeps the
# vocabulary independent of this scene; it is extended at run time with
# whatever Florence-2 proposes (open vocabulary).
VOCABULARY_FILE = RESOURCES / "scannet200_classes.json"


def load_vocabulary() -> tuple[list[str], dict]:
    record = json.loads(inside_workspace(VOCABULARY_FILE).read_text(encoding="utf-8"))
    return record["classes"], {"source": record["name"], "url": record["source_url"],
                               "source_sha256": record["source_sha256"]}


CROP_COLOR = {"removed": (215, 38, 61), "added": (27, 158, 75), "surface": (230, 161, 0)}


def normalise_label(text: str) -> str:
    text = text.split("<loc_")[0].strip().lower()
    for prefix in ("a ", "an ", "the "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return " ".join(text.replace("_", " ").split())


def upright_turns(pose: np.ndarray) -> int:
    """np.rot90 turns (counter-clockwise) that make world +Z point to image-up."""
    up = pose[:3, :3].T @ np.array([0.0, 0.0, 1.0])     # world up in camera axes (x right, y down)
    x, y = up[0], up[1]
    if abs(y) >= abs(x):
        return 0 if y < 0 else 2
    return 1 if x > 0 else 3


def rotate_box(box: tuple[float, float, float, float], turns: int, width: int, height: int):
    """Box (x0, y0, x1, y1) after np.rot90(image, turns)."""
    x0, y0, x1, y1 = box
    for _ in range(turns % 4):
        # one CCW turn: (x, y) -> (y, width - x); new size (height, width)
        x0, y0, x1, y1 = y0, width - x1, y1, width - x0
        width, height = height, width
    return x0, y0, x1, y1


def farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    start = int(np.argmin(np.linalg.norm(points - np.median(points, axis=0), axis=1)))
    chosen = [start]
    distance = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < min(count, len(points)):
        nxt = int(distance.argmax())
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(points - points[nxt], axis=1))
    return points[chosen]


def object_ends(result: dict) -> list[dict]:
    """Every object position that needs a label, tagged with its event."""
    ends = []
    for event in result["events"]:
        kind = event["event"]
        if kind in ("moved", "shifted"):
            ends.append({"object": event["from"], "event": event["id"], "role": "old position",
                         "side": "removed", "session": BEFORE})
            ends.append({"object": event["to"], "event": event["id"], "role": "new position",
                         "side": "added", "session": AFTER})
        elif kind in ("removed", "added"):
            ends.append({"object": event["object"], "event": event["object"], "role": kind,
                         "side": kind, "session": BEFORE if kind == "removed" else AFTER})
        elif kind == "surface_changed":
            obj = next(o for o in result["objects"]["surface"] if o["id"] == event["object"])
            ends.append({"object": event["object"], "event": event["object"], "role": "surface change",
                         "side": "surface", "session": obj["present_in"]})
    return ends


def select_views(frames: list[dict], points: np.ndarray, mode: str = "present",
                 limit: int = VIEWS_PER_OBJECT) -> list[dict]:
    """Diverse frames that see the object ('present': depth hits its surface) or
    see through its spot ('absent': depth lands beyond it)."""
    candidates = []
    for frame in frames:
        u, v, in_view, surface, see_past = project(frame, points)
        if mode == "absent":
            surface = see_past
        count = int(surface.sum())
        if count < MIN_SURFACE_POINTS:
            continue
        su, sv = u[surface], v[surface]
        intr = frame["intrinsics"]
        touches = (su.min() < BORDER_PX or sv.min() < BORDER_PX
                   or su.max() > intr["width"] - BORDER_PX or sv.max() > intr["height"] - BORDER_PX)
        candidates.append({"frame": frame, "count": count, "score": count * (0.5 if touches else 1.0),
                           "uv": np.column_stack([su, sv])})
    if not candidates:
        return []
    best = max(c["count"] for c in candidates)
    candidates = [c for c in candidates if c["count"] >= MIN_VIEW_FRACTION * best]
    candidates.sort(key=lambda c: (-c["score"], c["frame"]["frame"]))
    chosen = []
    for candidate in candidates:
        centre = candidate["frame"]["pose"][:3, 3]
        if all(np.linalg.norm(centre - c["frame"]["pose"][:3, 3]) >= VIEW_MIN_CAMERA_DISTANCE_M for c in chosen):
            chosen.append(candidate)
        if len(chosen) == limit:
            break
    return chosen


def prepare_check_view(view: dict, color) -> None:
    """Absent-side view: there is no object to segment, so the region is the projected spot."""
    image = np.asarray(Image.open(inside_workspace(DATA / view["frame"]["rgb"])).convert("RGB"))
    view["image"] = image
    uv = view["uv"]
    pad = 10.0
    view["box"] = (max(0.0, uv[:, 0].min() - pad), max(0.0, uv[:, 1].min() - pad),
                   min(image.shape[1] - 1.0, uv[:, 0].max() + pad), min(image.shape[0] - 1.0, uv[:, 1].max() + pad))
    view["mask"], view["mask_score"], view["color"] = None, None, color


def expand(box, pad, width, height, min_size=MIN_CROP_PX):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    grow_x = max(pad * w, (min_size - w) / 2, 0)
    grow_y = max(pad * h, (min_size - h) / 2, 0)
    return (int(max(0, np.floor(x0 - grow_x))), int(max(0, np.floor(y0 - grow_y))),
            int(min(width, np.ceil(x1 + grow_x))), int(min(height, np.ceil(y1 + grow_y))))


def run_sam(views_by_end: list[list[dict]], device: str) -> dict:
    from transformers import Sam2Model, Sam2Processor
    processor = Sam2Processor.from_pretrained(SAM_MODEL, revision=SAM_REVISION)
    model = Sam2Model.from_pretrained(SAM_MODEL, revision=SAM_REVISION).to(device).eval()
    for views in views_by_end:
        for view in views:
            frame = view["frame"]
            image = Image.open(inside_workspace(DATA / frame["rgb"])).convert("RGB")
            view["image"] = np.asarray(image)
            uv = view["uv"]
            depth = float(np.median(view["depth"]))
            pad = frame["intrinsics"]["fx"] * 0.025 / max(depth, 0.3)
            box = (uv[:, 0].min() - pad, uv[:, 1].min() - pad, uv[:, 0].max() + pad, uv[:, 1].max() + pad)
            box = (max(0.0, box[0]), max(0.0, box[1]), min(image.width - 1.0, box[2]), min(image.height - 1.0, box[3]))
            prompts = farthest_points(uv, SAM_POINTS)
            inputs = processor(images=image, input_boxes=[[list(box)]],
                               input_points=[[[p.tolist() for p in prompts]]],
                               input_labels=[[[1] * len(prompts)]], return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs, multimask_output=True)
            masks = processor.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
            best, best_score = None, 0.0
            ui = np.clip(np.round(uv[:, 0]).astype(int), 0, image.width - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int), 0, image.height - 1)
            for mask in masks:
                if not mask.any():
                    continue
                inside = float(mask[vi, ui].mean())
                ys, xs = np.nonzero(mask)
                mbox = (xs.min(), ys.min(), xs.max(), ys.max())
                ix = max(0, min(mbox[2], box[2]) - max(mbox[0], box[0]))
                iy = max(0, min(mbox[3], box[3]) - max(mbox[1], box[1]))
                union = ((mbox[2] - mbox[0]) * (mbox[3] - mbox[1]) + (box[2] - box[0]) * (box[3] - box[1])
                         - ix * iy)
                score = inside * (ix * iy / max(union, 1.0))
                if score > best_score:
                    best, best_score = mask, score
            view["prompt_box"] = box
            view["mask_score"] = round(best_score, 3)
            if best is not None and best_score >= MIN_MASK_SCORE:
                ys, xs = np.nonzero(best)
                view["mask"] = best
                view["box"] = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            else:
                view["mask"] = None
                view["box"] = box
    del model
    torch.cuda.empty_cache()
    return {"id": SAM_MODEL, "revision": SAM_REVISION}


def prepare_crops(view: dict) -> None:
    image = view["image"]
    height, width = image.shape[:2]
    turns = upright_turns(view["frame"]["pose"])
    view["turns"] = turns
    context = expand(view["box"], CONTEXT_PAD, width, height)
    tight = expand(view["box"], TIGHT_PAD, width, height)
    x0, y0, x1, y1 = context
    crop = image[y0:y1, x0:x1].copy()
    region = (view["box"][0] - x0, view["box"][1] - y0, view["box"][2] - x0, view["box"][3] - y0)
    view["context_crop"] = np.ascontiguousarray(np.rot90(crop, turns))
    view["region"] = rotate_box(region, turns, x1 - x0, y1 - y0)
    tx0, ty0, tx1, ty1 = tight
    view["tight_crop"] = np.ascontiguousarray(np.rot90(image[ty0:ty1, tx0:tx1], turns))
    overlay = crop.copy()
    if view["mask"] is not None:
        sub = view["mask"][y0:y1, x0:x1].astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, view["color"], 3)
    else:
        bx0, by0, bx1, by1 = (int(v) for v in region)
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), view["color"], 3)
    view["display_crop"] = np.ascontiguousarray(np.rot90(overlay, turns))


def run_florence(views_by_end: list[list[dict]], device: str) -> dict:
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, revision=FLORENCE_REVISION)
    model = Florence2ForConditionalGeneration.from_pretrained(
        FLORENCE_MODEL, revision=FLORENCE_REVISION, dtype=DTYPE).to(device).eval()

    def ask(image: Image.Image, task: str, region) -> str:
        w, h = image.size
        loc = "".join(f"<loc_{int(np.clip(value / size * 1000, 0, 999))}>"
                      for value, size in zip(region, (w, h, w, h)))
        inputs = processor(text=task + loc, images=image, return_tensors="pt").to(device, DTYPE)
        with torch.no_grad():
            generated = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                       max_new_tokens=40, num_beams=3, do_sample=False)
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(text, task=task, image_size=image.size)[task]
        return parsed if isinstance(parsed, str) else str(parsed)

    for views in views_by_end:
        for view in views:
            image = Image.fromarray(view["context_crop"])
            view["florence_category"] = normalise_label(ask(image, "<REGION_TO_CATEGORY>", view["region"]))
            view["florence_description"] = normalise_label(ask(image, "<REGION_TO_DESCRIPTION>", view["region"]))
    del model
    torch.cuda.empty_cache()
    return {"id": FLORENCE_MODEL, "revision": FLORENCE_REVISION}


def run_siglip(views_by_end: list[list[dict]], candidates: list[str], device: str) -> dict:
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, revision=SIGLIP_REVISION)
    model = AutoModel.from_pretrained(SIGLIP_MODEL, revision=SIGLIP_REVISION, dtype=DTYPE).to(device).eval()
    texts = [SIGLIP_TEMPLATE.format(c) for c in candidates]
    text_inputs = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = getattr(text_features, "pooler_output", text_features)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
        scale, bias = model.logit_scale.exp().float(), model.logit_bias.float()
        for views in views_by_end:
            for view in views:
                image_inputs = processor(images=Image.fromarray(view["tight_crop"]), return_tensors="pt").to(device)
                image_features = model.get_image_features(pixel_values=image_inputs["pixel_values"].to(DTYPE))
                image_features = getattr(image_features, "pooler_output", image_features)
                image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
                logits = (image_features @ text_features.T) * scale + bias
                view["siglip_probs"] = torch.softmax(logits[0], dim=0).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return {"id": SIGLIP_MODEL, "revision": SIGLIP_REVISION}


def compatible(a: str, b: str) -> bool:
    """Same object at different specificity: one label's words contain the other's
    ("chair" / "office chair"), or both share the head noun (last word)."""
    wa, wb = a.split(), b.split()
    return a == b or set(wa) <= set(wb) or set(wb) <= set(wa) or wa[-1] == wb[-1]


def decide(views: list[dict], candidates: list[str]) -> dict:
    votes = Counter(v["florence_category"] for v in views if v["florence_category"])
    pooled = np.mean([v["siglip_probs"] for v in views], axis=0)
    order = np.argsort(-pooled)
    ranking = [(candidates[i], float(pooled[i])) for i in order]
    rank_of = {name: rank for rank, (name, _) in enumerate(ranking)}
    # Florence-2 speaks for the object only with a strict majority of the views
    # (or the only view); a split vote is reported but does not decide.
    florence_label = None
    if votes:
        top_count = max(votes.values())
        if top_count > len(views) / 2 or len(views) == 1:
            tied = [label for label, count in votes.items() if count == top_count]
            florence_label = min(tied, key=lambda label: rank_of.get(label, len(ranking)))
    siglip_top = [name for name, _ in ranking[:SIGLIP_TOP_K_SUPPORT]]
    support = next((name for name in siglip_top if florence_label and compatible(florence_label, name)), None)
    if support is not None:
        # Keep the more specific of the two names for the same object.
        final = max((florence_label, support), key=lambda label: (len(label.split()), label == florence_label))
        confident = True
        agreement = ("both models agree" if florence_label == support
                     else f"compatible: Florence-2 '{florence_label}', SigLIP 2 '{support}' "
                          f"(rank {rank_of[support] + 1})")
    else:
        final = ranking[0][0]
        confident = any(compatible(final, label) for label in votes)
        if florence_label is None:
            agreement = (f"Florence-2 split {dict(votes)}; SigLIP 2 choice used"
                         + (" (Florence-2 proposed it in some views)" if confident else ""))
        else:
            agreement = (f"models disagree: Florence-2 '{florence_label}' not in SigLIP 2 top "
                         f"{SIGLIP_TOP_K_SUPPORT}; SigLIP 2 choice used")
    return {
        "label": final,
        "agreement": agreement,
        "confident": confident,
        "florence_votes": dict(votes.most_common()),
        "florence_vote_fraction": round(votes.get(florence_label, 0) / max(len(views), 1), 2) if florence_label else 0.0,
        "siglip_top5": [{"label": name, "prob": round(prob, 3)} for name, prob in ranking[:5]],
        "descriptions": [v["florence_description"] for v in views],
    }


def majority_names(decision: dict, n_views: int) -> set[str]:
    names = {decision["label"]}
    votes = decision["florence_votes"]
    if votes:
        label, count = max(votes.items(), key=lambda kv: kv[1])
        if count > n_views / 2 or n_views == 1:
            names.add(label)
    return names


def same_object(present: dict, absent: dict, n_present: int, n_absent: int) -> bool:
    """Both recordings name the same object at this spot (final label or Florence-2 majority)."""
    return any(compatible(a, b) for a in majority_names(present, n_present)
               for b in majority_names(absent, n_absent))


def render_sheet(ends: list[dict], path_pattern: str) -> list[str]:
    rows_per_page = 6
    paths = []
    pages = [ends[i:i + rows_per_page] for i in range(0, len(ends), rows_per_page)]
    for number, page in enumerate(pages, start=1):
        figure, axes = plt.subplots(len(page), VIEWS_PER_OBJECT, figsize=(3.2 * VIEWS_PER_OBJECT, 3.3 * len(page)),
                                    dpi=100, facecolor="white", squeeze=False)
        for row_axes, end in zip(axes, page):
            for axis in row_axes:
                axis.axis("off")
            for axis, view in zip(row_axes, end["views"]):
                axis.imshow(view["display_crop"])
                axis.set_title(f"{view['frame']['frame']}: {view['florence_category']}", fontsize=8)
            decision = end["decision"]
            top = ", ".join(f"{d['label']} {d['prob']:.2f}" for d in decision["siglip_top5"][:3])
            row_axes[0].text(-0.06, 0.5, f"{end['object']} ({end['event']}, {end['role']})\n"
                                         f"LABEL: {decision['label']}\n{decision['agreement']}\nSigLIP: {top}",
                             transform=row_axes[0].transAxes, ha="right", va="center", fontsize=8.5,
                             fontweight="bold", color=np.array(CROP_COLOR[end["side"]]) / 255.0)
        figure.suptitle(f"Step 3 open-vocabulary labels (page {number}/{len(pages)})", fontsize=12, fontweight="bold")
        figure.tight_layout(rect=(0.2, 0, 1, 0.98))
        path = inside_workspace(OUTPUT / "qa" / path_pattern.format(number))
        figure.savefig(str(path))
        plt.close(figure)
        paths.append(str(path))
    return paths


def main() -> None:
    started = time.time()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = DEVICE
    (inside_workspace(SEMANTICS) / "crops").mkdir(parents=True, exist_ok=True)

    result = json.loads(inside_workspace(OUTPUT / "object_changes.json").read_text())
    voxels = np.load(str(inside_workspace(OUTPUT / "objects" / "object_voxels.npz")))
    grid = np.load(str(inside_workspace(OUTPUT / "voxel_change_grid.npz")))
    origin, dims, voxel = grid["origin"], tuple(int(d) for d in grid["dims"]), float(grid["voxel_m"])
    sessions = {name: load_session(name) for name in SESSIONS}

    ends = object_ends(result)
    for end in ends:
        ijk = np.column_stack(np.unravel_index(voxels[end["object"]], dims))
        points = origin + (ijk + 0.5) * voxel
        end["views"] = select_views(sessions[end["session"]], points)
        for view in end["views"]:
            u, v, _, surface, _ = project(view["frame"], points)
            pose = view["frame"]["pose"]
            view["depth"] = ((points[surface] - pose[:3, 3]) @ pose[:3, :3])[:, 2]
            view["color"] = CROP_COLOR[end["side"]]
        print(f"{end['object']:>4} {end['role']:<15} {end['session']}: {len(end['views'])} views "
              f"{[v['frame']['frame'] for v in end['views']]}", flush=True)
    ends = [end for end in ends if end["views"]]
    views_by_end = [end["views"] for end in ends]
    for end in ends:
        other = AFTER if end["session"] == BEFORE else BEFORE
        ijk = np.column_stack(np.unravel_index(voxels[end["object"]], dims))
        end["check_session"] = other
        end["check_views"] = select_views(sessions[other], origin + (ijk + 0.5) * voxel, "absent", PHOTO_CHECK_VIEWS)
        for view in end["check_views"]:
            prepare_check_view(view, (110, 110, 110))
    check_by_end = [end["check_views"] for end in ends if end["check_views"]]

    models = [run_sam(views_by_end, device)]
    print(f"SAM 2.1 masks done ({time.time() - started:.0f}s)", flush=True)
    for views in views_by_end + check_by_end:
        for view in views:
            prepare_crops(view)
    models.append(run_florence(views_by_end + check_by_end, device))
    print(f"Florence-2 labels done ({time.time() - started:.0f}s)", flush=True)

    vocabulary, vocabulary_source = load_vocabulary()
    proposals = sorted({v["florence_category"] for views in views_by_end for v in views if v["florence_category"]})
    candidates = list(dict.fromkeys(vocabulary + proposals))
    models.append(run_siglip(views_by_end + check_by_end, candidates, device))
    print(f"SigLIP 2 scoring done ({time.time() - started:.0f}s)", flush=True)

    records = []
    for end in ends:
        end["decision"] = decide(end["views"], candidates)
        view_records = []
        for index, view in enumerate(end["views"], start=1):
            crop_path = SEMANTICS / "crops" / f"{end['object']}_v{index}.jpg"
            cv2.imwrite(str(inside_workspace(crop_path)), cv2.cvtColor(view["display_crop"], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            top = np.argsort(-view["siglip_probs"])[:3]
            view_records.append({
                "session": end["session"], "frame": view["frame"]["frame"], "rgb": view["frame"]["rgb"],
                "surface_voxels_seen": view["count"], "mask_score": view["mask_score"],
                "mask_used": view["mask"] is not None,
                "box_xyxy": [round(float(b), 1) for b in view["box"]],
                "upright_turns_ccw": view["turns"],
                "florence_category": view["florence_category"],
                "florence_description": view["florence_description"],
                "siglip_top3": [{"label": candidates[i], "prob": round(float(view["siglip_probs"][i]), 3)}
                                for i in top],
                "crop": out_rel(crop_path),
            })
        check = None
        if end["check_views"]:
            end["check_decision"] = decide(end["check_views"], candidates)
            check = {"session": end["check_session"], **end["check_decision"],
                     "same_object_visible": same_object(end["decision"], end["check_decision"],
                                                        len(end["views"]), len(end["check_views"])),
                     "views": [{"frame": v["frame"]["frame"], "florence_category": v["florence_category"],
                                "box_xyxy": [round(float(b), 1) for b in v["box"]],
                                "siglip_top3": [{"label": candidates[i], "prob": round(float(v["siglip_probs"][i]), 3)}
                                                for i in np.argsort(-v["siglip_probs"])[:3]]}
                               for v in end["check_views"]]}
        records.append({"object": end["object"], "event": end["event"], "role": end["role"],
                        "side": end["side"], "session": end["session"], **end["decision"],
                        "views": view_records, "photo_check": check})
        print(f"{end['object']:>4} {end['role']:<15} -> {end['decision']['label']:<20} "
              f"{'confident' if end['decision']['confident'] else 'UNCERTAIN':<9} "
              f"({end['decision']['agreement']}; Florence {end['decision']['florence_votes']}; "
              f"SigLIP {[d['label'] for d in end['decision']['siglip_top5'][:3]]})", flush=True)
        if check:
            print(f"      photo check in {check['session']}: {check['label']} (Florence {check['florence_votes']}; "
                  f"SigLIP {[d['label'] for d in check['siglip_top5'][:3]]}) -> same object visible: "
                  f"{check['same_object_visible']}", flush=True)

    # Event level: a move/shift is labelled once, from both of its ends.
    event_labels = []
    by_event: dict[str, list[dict]] = {}
    for record, end in zip(records, ends):
        by_event.setdefault(record["event"], []).append((record, end))
    for event_id, members in by_event.items():
        if len(members) == 2:
            all_views = members[0][1]["views"] + members[1][1]["views"]
            joint = decide(all_views, candidates)
            ends_agree = members[0][0]["label"] == members[1][0]["label"]
            event_labels.append({"event": event_id, "label": joint["label"], "agreement": joint["agreement"],
                                 "confident": joint["confident"],
                                 "old_position_still_shows_it": (members[0][0]["photo_check"] or {}).get(
                                     "same_object_visible"),
                                 "new_position_showed_it_before": (members[1][0]["photo_check"] or {}).get(
                                     "same_object_visible"),
                                 "old_position_label": members[0][0]["label"],
                                 "new_position_label": members[1][0]["label"],
                                 "ends_agree": ends_agree,
                                 "florence_votes": joint["florence_votes"],
                                 "siglip_top5": joint["siglip_top5"]})
        else:
            record = members[0][0]
            event_labels.append({"event": event_id, "label": record["label"], "agreement": record["agreement"],
                                 "confident": record["confident"], "florence_votes": record["florence_votes"],
                                 "siglip_top5": record["siglip_top5"],
                                 "same_object_visible_in_other_recording": (record["photo_check"] or {}).get(
                                     "same_object_visible"),
                                 "other_recording_label": (record["photo_check"] or {}).get("label")})

    sheets = render_sheet(ends, "step3_labels_p{}.png")
    check_ends = [{"object": end["object"], "event": end["event"], "role": f"spot in {end['check_session']}",
                   "side": end["side"], "views": end["check_views"], "decision": end["check_decision"]}
                  for end in ends if end["check_views"]]
    for view in (v for end in check_ends for v in end["views"]):
        view["display_crop"] = view["context_crop"]
    sheets += render_sheet(check_ends, "step3_photo_check_p{}.png")
    output = {
        "schema_version": "1.0.0",
        "pipeline_version": PIPELINE_VERSION,
        "stage": "step3_open_vocabulary_labels",
        "learned_models_used": [{**m, "use": use, "license": lic, "trained_or_finetuned_here": False}
                                for m, use, lic in zip(models, ("object masks", "region category + description",
                                                                "zero-shot cross-check"),
                                                       ("Apache-2.0", "MIT", "Apache-2.0"))],
        "manual_labels_used": False,
        "parameters": {"views_per_object": VIEWS_PER_OBJECT, "min_view_fraction": MIN_VIEW_FRACTION,
                       "view_min_camera_distance_m": VIEW_MIN_CAMERA_DISTANCE_M, "sam_points": SAM_POINTS,
                       "min_mask_score": MIN_MASK_SCORE, "context_pad": CONTEXT_PAD, "tight_pad": TIGHT_PAD,
                       "siglip_template": SIGLIP_TEMPLATE, "siglip_top_k_support": SIGLIP_TOP_K_SUPPORT,
                       "florence_generation": {"num_beams": 3, "do_sample": False, "max_new_tokens": 40},
                       "photo_check_views": PHOTO_CHECK_VIEWS,
                       "precision": f"Florence-2 and SigLIP 2: {PRECISION}; SAM 2.1: fp32"},
        "vocabulary": {**vocabulary_source, "file": "navdelta/approaches/voxel_diff/resources/"
                       "scannet200_classes.json", "classes": len(vocabulary)},
        "florence_proposals_added": [p for p in proposals if p not in vocabulary],
        "step2_result_sha256": result["result_sha256"],
        "events": event_labels,
        "objects": records,
        "qa_sheets": [out_rel(Path(p)) for p in sheets],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    digest = hashlib.sha256(json.dumps({"events": event_labels, "objects": [
        {k: v for k, v in r.items() if k != "views"} | {"views": [{k: v for k, v in x.items() if k != "siglip_top3"}
                                                                  for x in r["views"]]}
        for r in records]}, sort_keys=True).encode()).hexdigest()
    output["result_sha256"] = digest
    output["runtime_s"] = round(time.time() - started, 1)
    (inside_workspace(SEMANTICS) / "object_labels.json").write_text(json.dumps(output, indent=2))

    print("\nEvent labels:")
    for item in event_labels:
        extra = (f" (old: {item['old_position_label']}, new: {item['new_position_label']}, "
                 f"ends agree: {item['ends_agree']})" if "ends_agree" in item else "")
        print(f"  {item['event']:<5} {item['label']:<20} {'confident' if item['confident'] else 'UNCERTAIN':<9} "
              f"{item['agreement']}{extra}")
    print(f"Result sha256: {digest}")
    print(f"Done in {time.time() - started:.0f}s -> {SEMANTICS / 'object_labels.json'}")


if __name__ == "__main__":
    main()
