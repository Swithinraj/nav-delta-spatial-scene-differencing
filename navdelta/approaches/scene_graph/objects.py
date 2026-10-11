"""scene_graph, stage 1: 2D detections -> 3D object nodes -> one scene graph per recording.

Per keyframe (raw RGB, rotated upright using the gravity-aligned pose):
  Florence-2 <OD> proposes labelled boxes; SAM 2.1 turns each box into a mask;
  objects outside Florence-2's fixed categories (a tower fan) come from SAM 2.1 prompted with a
  point grid: segments no Florence-2 mask covers are named by SigLIP 2 against the ScanNet200
  vocabulary (ConceptGraphs-style); SigLIP 2 embeds every crop (2D results cached per keyframe);
  the mask is lifted to 3D with the MASt3R depth of that frame (eroded edges,
  confident depth only, depth outliers cut, largest connected piece kept).
Per recording, detections are merged frame by frame into object nodes when they
overlap in 3D and look alike (ConceptGraphs-style). Nodes seen in one frame
only are dropped. Edges: "on" (resting on another node) and "next to".
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from navdelta import config, context
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"                  # shared reconstruction (navdelta.recon)
OUTPUT = context.approach_dir("scene_graph")
from navdelta.recon.reconstruct import keyframe_folder

FLORENCE = ("florence-community/Florence-2-large", "4271c66b88cdbc05735372ec13b2360108de5317")
SAM = ("facebook/sam2.1-hiera-base-plus", "b7320756a13354e7530a63935656d35b2f91a290")
SIGLIP = ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f")

NOT_OBJECTS = {"house", "room", "building", "person", "human face", "man", "woman", "wall", "floor",
               "ceiling", "window", "door handle", "power plugs and sockets", "doorframe", "windowsill",
               "column", "pillar", "rail", "stair rail", "ledge", "structure", "object"}
VOCABULARY = Path(__file__).resolve().parents[1] / "voxel_diff" / "resources" / "scannet200_classes.json"
GRID = (12, 9)              # SAM point prompts across the image
GRID_IOU_MIN = 0.80         # SAM's own mask-quality estimate
GRID_AREA = (0.004, 0.30)   # share of the image
GRID_NMS_IOU = 0.5
GRID_CHUNK = 36             # point prompts per SAM call (4 GB GPU)
DETECT_CHUNK_FRAMES = 50    # the 2D cache is saved after every chunk of frames
CACHE_VERSION = 2           # 2: grid segments named over a fixed vocabulary; older caches are recomputed
COVERED_BY_FLORENCE = 0.5
GRID_LABEL_MIN = 0.05
SIGLIP_BATCH = 16
TIMES: dict[str, float] = {}     # seconds per detector part, for the speed log


def clock(part: str, since: float) -> float:
    now = time.time()
    TIMES[part] = TIMES.get(part, 0.0) + now - since
    return now       # SigLIP probability of the best name; lower: floor/wall patches
MAX_BOX_AREA = 0.6          # a box covering most of the image is the scene, not an object
CONF_MIN = 1.5              # MASt3R confidence for usable depth
ERODE_PX = 2                # at the 512x384 depth resolution
VOXEL_M = 0.05
MIN_POINTS = 40
OVERLAP_MIN = 0.25          # share of a detection's voxels already in a node
SEM_MIN = 0.70              # SigLIP cosine between a detection and a node (same object across views: 0.8-0.9)
MIN_OBSERVATIONS = 2
MERGE_SIZE_RATIO = 1.0e9    # a detection joins a node only if their extents are within this ratio
                            # (large spaces: rows of identical boxes must not fuse into one node)
NEXT_TO_M = 0.30
ON_GAP_M = 0.15


# Scale profile (navdelta/configs/*.toml): the values above are the room profile.
PROFILE_SETTINGS = config.apply(globals(), "scene_graph.objects")


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


def voxel_keys(points: np.ndarray) -> set:
    return set(map(tuple, np.floor(points / VOXEL_M).astype(np.int64)))


def largest_component(points: np.ndarray) -> np.ndarray:
    keys = np.floor(points / VOXEL_M).astype(np.int64)
    low = keys.min(0)
    grid = np.zeros(tuple(keys.max(0) - low + 3), bool)
    idx = keys - low + 1
    grid[tuple(idx.T)] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    if count <= 1:
        return np.ones(len(points), bool)
    lab = labels[tuple(idx.T)]
    best = np.bincount(lab)[1:].argmax() + 1
    return lab == best


def detect_2d(data: dict, cache_path: Path) -> list[dict]:
    """Boxes, labels, masks (at depth resolution) and SigLIP embeddings per keyframe, cached by keyframe."""
    cache = {}
    if cache_path.is_file() and os.environ.get("NAVDELTA_RECOMPUTE") != "1":     # `navdelta verify`: recompute
        stored = np.load(cache_path, allow_pickle=True)
        if "version" in stored and int(stored["version"]) == CACHE_VERSION:
            cache = stored["detections"].item()
        else:
            print(f"  {cache_path.name} was made by an earlier version of this stage: detecting again", flush=True)
    todo = [k for k, stem in enumerate(data["frames"]) if f"{data['owner'][k]}/{stem}" not in cache]
    for start in range(0, len(todo), DETECT_CHUNK_FRAMES):     # saved as it goes: a crash loses one chunk
        cache.update(run_detectors(data, todo[start:start + DETECT_CHUNK_FRAMES]))
        if cache_path.exists():
            cache_path.unlink()                # never write through a hard link to another run's cache
        np.savez(cache_path, detections=np.array(cache, dtype=object), version=CACHE_VERSION)
        print(f"  2D detections cached for {min(start + DETECT_CHUNK_FRAMES, len(todo))}/{len(todo)} frames",
              flush=True)
    detections = []
    for k, (stem, rec) in enumerate(zip(data["frames"], data["owner"])):
        for det in cache[f"{rec}/{stem}"]:
            detections.append({**det, "frame": int(k), "rec": str(rec)})
    return detections


def grid_masks(sp, sm, image: np.ndarray, taken: np.ndarray, out_size: tuple[int, int]) -> list[np.ndarray]:
    """Class-agnostic SAM 2.1 masks from a point grid, minus what Florence-2 already covers.
    Masks come back at out_size (h, w) -- the depth resolution -- which keeps this fast."""
    H, W = image.shape[:2]
    xs = (np.arange(GRID[0]) + 0.5) * W / GRID[0]
    ys = (np.arange(GRID[1]) + 0.5) * H / GRID[1]
    points = [[[float(x), float(y)]] for y in ys for x in xs]
    inp = sp(images=Image.fromarray(image), return_tensors="pt").to("cuda")
    with torch.no_grad():
        embeddings = sm.get_image_embeddings(inp["pixel_values"])
    scores, lows = [], []
    for start in range(0, len(points), GRID_CHUNK):     # prompts in chunks: a lower memory peak, same masks
        chunk = points[start:start + GRID_CHUNK]
        prompt = sp(images=Image.fromarray(image), input_points=[chunk], input_labels=[[[1]] * len(chunk)],
                    return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = sm(image_embeddings=embeddings, input_points=prompt["input_points"],
                     input_labels=prompt["input_labels"], multimask_output=True)
        sc = out.iou_scores[0].float().cpu().numpy()                     # (chunk, 3)
        scores.append(sc)
        lows.append(out.pred_masks[0][torch.arange(len(chunk)), torch.as_tensor(sc.argmax(1))].cpu())
        del out
    scores = np.concatenate(scores)
    low = torch.cat(lows)                                                # (points, h, w)
    best = scores.argmax(1)
    keep = np.flatnonzero(scores[np.arange(len(points)), best] >= GRID_IOU_MIN)
    if not len(keep):
        return []
    masks = sp.post_process_masks([low[torch.as_tensor(keep)][:, None]], [list(out_size)])[0][:, 0].numpy()
    taken = cv2.resize(taken.astype(np.uint8), out_size[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    order = np.argsort(-scores[keep, best[keep]])
    # overlap tests on 4x-subsampled masks: same decisions to within a few pixels, ~16x cheaper
    small = masks[:, ::4, ::4].reshape(len(masks), -1)
    small_taken = taken[::4, ::4].ravel()
    kept: list[int] = []
    for k in order:
        area = masks[k].mean()
        if not GRID_AREA[0] <= area <= GRID_AREA[1]:
            continue
        m = small[k]
        n = max(1, int(m.sum()))
        if (m & small_taken).sum() >= COVERED_BY_FLORENCE * n:
            continue
        if kept:
            q = small[kept]
            inter = (q & m).sum(1)
            union = (q | m).sum(1)
            if (inter / np.maximum(union, 1) > GRID_NMS_IOU).any():
                continue
        kept.append(int(k))
    return [masks[k] for k in kept]


def run_detectors(data: dict, todo: list[int]) -> dict:
    from transformers import AutoModel, AutoProcessor, Florence2ForConditionalGeneration, Sam2Model, Sam2Processor
    fp, fm = AutoProcessor.from_pretrained(FLORENCE[0], revision=FLORENCE[1]), None
    fm = Florence2ForConditionalGeneration.from_pretrained(FLORENCE[0], revision=FLORENCE[1],
                                                           dtype=torch.float16).to("cuda").eval()
    sp = Sam2Processor.from_pretrained(SAM[0], revision=SAM[1])
    sm = Sam2Model.from_pretrained(SAM[0], revision=SAM[1]).to("cuda").eval()
    found: dict[str, list[dict]] = {}
    started = time.time()
    h, w = data["depth"].shape[1:]
    for count, k in enumerate(todo):
        stem, rec = data["frames"][k], data["owner"][k]
        found[f"{rec}/{stem}"] = []
        path = keyframe_folder(rec, "images") / f"{stem}.jpg"
        t = time.time()
        image = np.asarray(Image.open(path).convert("RGB"))
        H, W = image.shape[:2]
        turns = upright_turns(data["cam2w"][k])
        upright = Image.fromarray(np.ascontiguousarray(np.rot90(image, turns)))
        inputs = fp(text="<OD>", images=upright, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            out = fm.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                              max_new_tokens=256, num_beams=3, do_sample=False)
        result = fp.post_process_generation(fp.batch_decode(out, skip_special_tokens=False)[0], task="<OD>",
                                            image_size=upright.size)["<OD>"]
        t = clock("florence2", t)
        boxes, labels = [], []
        for box, label in zip(result["bboxes"], result["labels"]):
            label = " ".join(label.lower().split())
            box = rotate_box(tuple(box), (4 - turns) % 4, upright.size[0], upright.size[1])
            area = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
            if label in NOT_OBJECTS or area > MAX_BOX_AREA:
                continue
            boxes.append([float(v) for v in box])
            labels.append(label)
        masks = np.zeros((0, H, W), bool)
        if boxes:
            sam_in = sp(images=Image.fromarray(image), input_boxes=[boxes], return_tensors="pt").to("cuda")
            with torch.no_grad():
                sam_out = sm(**sam_in, multimask_output=False)
            masks = sp.post_process_masks(sam_out.pred_masks.cpu(), sam_in["original_sizes"])[0][:, 0].numpy()
        t = clock("sam_boxes", t)
        extra = grid_masks(sp, sm, image, masks.any(0) if len(masks) else np.zeros((H, W), bool), (h, w))
        t = clock("sam_grid", t)
        sx, sy = W / w, H / h
        small_masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                       for m in masks]
        for m in extra:
            v, u = np.nonzero(m)
            boxes.append([float(u.min() * sx), float(v.min() * sy), float((u.max() + 1) * sx), float((v.max() + 1) * sy)])
            labels.append(None)                                   # named by SigLIP below
            small_masks.append(m)
        for small, box, label in zip(small_masks, boxes, labels):
            small = ndimage.binary_erosion(small, iterations=ERODE_PX)
            if small.sum() < MIN_POINTS:
                continue
            found[f"{rec}/{stem}"].append({"label": label, "box": box, "mask": np.packbits(small),
                                           "image_path": str(path), "turns": turns,
                                           "source": "florence2" if label else "sam_grid"})
        if (count + 1) % 20 == 0:
            print(f"  detected {count + 1}/{len(todo)} frames, "
                  f"{sum(len(v) for v in found.values())} detections ({time.time() - started:.0f}s)", flush=True)
    del fm, sm
    torch.cuda.empty_cache()
    embed_and_name([d for v in found.values() for d in v], (h, w))
    for key in found:
        found[key] = [d for d in found[key] if keep_detection(d)]
    return found


def keep_detection(det: dict) -> bool:
    return det["label"] not in NOT_OBJECTS and det.get("label_score", 1.0) >= GRID_LABEL_MIN


def crop_for(det: dict, image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Upright crop of a detection. Grid segments are masked (background grey), so a floor patch
    next to a fan is not named "fan" because the fan is in its bounding box."""
    x0, y0, x1, y1 = (int(round(c)) for c in det["box"])
    x0, y0 = max(0, x0), max(0, y0)
    crop = image[y0:y1, x0:x1].copy()
    if det.get("source") == "sam_grid":
        h, w = size
        mask = np.unpackbits(det["mask"])[:h * w].reshape(h, w)
        full = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        full = ndimage.binary_dilation(full, iterations=4)            # undo the erosion done for lifting
        crop[~full[y0:y1, x0:x1]] = 128
        det["masked_crop"] = True
    return np.ascontiguousarray(np.rot90(crop, det["turns"]))


def embed_and_name(detections: list[dict], size: tuple[int, int]) -> None:
    """SigLIP 2 embedding of each detection's crop; grid segments are named zero-shot over the
    ScanNet200 classes. The vocabulary is fixed: a segment's name does not depend on which other
    frames were detected with it."""
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP[0], revision=SIGLIP[1])
    model = AutoModel.from_pretrained(SIGLIP[0], revision=SIGLIP[1], dtype=torch.float16).to("cuda").eval()
    cache: dict[str, np.ndarray] = {}
    t = time.time()
    batch: list[tuple[dict, np.ndarray]] = []

    def flush() -> None:
        with torch.no_grad():
            inp = proc(images=[Image.fromarray(c) for _, c in batch], return_tensors="pt").to("cuda")
            f = model.get_image_features(pixel_values=inp["pixel_values"].half())
            f = getattr(f, "pooler_output", f)
            f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
        for (det, _), e in zip(batch, f):
            det["embedding"] = e
        batch.clear()

    for det in detections:                       # in frame order: one image decode per frame
        image = cache.get(det["image_path"])
        if image is None:
            if len(cache) > 8:
                cache.clear()
            image = cache.setdefault(det["image_path"], np.asarray(Image.open(det["image_path"]).convert("RGB")))
        batch.append((det, crop_for(det, image, size)))
        if len(batch) == SIGLIP_BATCH:
            flush()
    if batch:
        flush()
    names = sorted(set(json.loads(VOCABULARY.read_text(encoding="utf-8"))["classes"]))
    with torch.no_grad():
        tok = proc(text=[f"a photo of a {n}." for n in names], padding="max_length", max_length=64,
                   return_tensors="pt").to("cuda")
        tf = model.get_text_features(**tok)
        tf = getattr(tf, "pooler_output", tf)
        text = torch.nn.functional.normalize(tf.float(), dim=-1).cpu().numpy()
        scale, bias = float(model.logit_scale.exp()), float(model.logit_bias)
    for det in detections:
        if det.get("source") == "sam_grid":
            logits = scale * (text @ det["embedding"]) + bias
            k = int(np.argmax(logits))
            det["label"] = names[k]
            det["label_score"] = float(1 / (1 + np.exp(-logits[k])))
    clock("siglip", t)
    del model
    torch.cuda.empty_cache()


def lift(detections: list[dict], data: dict) -> list[dict]:
    """Each 2D mask -> 3D points with that frame's depth and pose."""
    h, w = data["depth"].shape[1:]
    out = []
    for det in detections:
        k = det["frame"]
        depth, conf, K, cam2w = data["depth"][k], data["conf"][k], data["K"][k], data["cam2w"][k]
        mask = np.unpackbits(det["mask"])[:h * w].reshape(h, w).astype(bool) & (conf > CONF_MIN)
        v, u = np.nonzero(mask)
        if len(u) < MIN_POINTS:
            continue
        z = depth[v, u]
        med = np.median(z)
        mad = np.median(np.abs(z - med)) + 1e-6
        keep = np.abs(z - med) < 3.0 * 1.4826 * mad + 0.05
        u, v, z = u[keep], v[keep], z[keep]
        cam = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        pts = cam @ cam2w[:3, :3].T + cam2w[:3, 3]
        pts = pts[largest_component(pts)]
        if len(pts) < MIN_POINTS:
            continue
        out.append({**det, "points": pts.astype(np.float32)})
    return out


def merge(detections: list[dict]) -> list[dict]:
    nodes: list[dict] = []
    for det in sorted(detections, key=lambda d: d["frame"]):
        keys = voxel_keys(det["points"])
        extent = float(np.max(det["points"].max(0) - det["points"].min(0))) + VOXEL_M
        best, best_overlap = None, 0.0
        for node in nodes:
            sem = float(det["embedding"] @ node["embedding"] / np.linalg.norm(node["embedding"]))
            if sem < SEM_MIN:
                continue
            if max(extent / node["extent"], node["extent"] / extent) > MERGE_SIZE_RATIO:
                continue
            overlap = len(keys & node["dilated"]) / len(keys)
            if overlap >= OVERLAP_MIN and overlap > best_overlap:
                best, best_overlap = node, overlap
        if best is None:
            best = {"voxels": Counter(), "dilated": set(), "embedding": np.zeros_like(det["embedding"]),
                    "labels": Counter(), "frames": set(), "points": [], "lo": det["points"].min(0),
                    "hi": det["points"].max(0)}
            nodes.append(best)
        best["lo"] = np.minimum(best["lo"], det["points"].min(0))
        best["hi"] = np.maximum(best["hi"], det["points"].max(0))
        best["extent"] = float(np.max(best["hi"] - best["lo"])) + VOXEL_M
        best["voxels"].update(keys)
        for key in keys:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        best["dilated"].add((key[0] + dx, key[1] + dy, key[2] + dz))
        best["embedding"] = best["embedding"] + det["embedding"]
        best["labels"][det["label"]] += 1
        best["frames"].add(det["frame"])
        best["points"].append(det["points"][:: max(1, len(det["points"]) // 800)])
    out = []
    for node in nodes:
        if len(node["frames"]) < MIN_OBSERVATIONS:
            continue
        # keep voxels confirmed by at least two detections when the node was seen often enough
        min_count = 2 if len(node["frames"]) >= 4 else 1
        cells = np.array([k for k, c in node["voxels"].items() if c >= min_count])
        if len(cells) < 4:
            continue
        centres = (cells + 0.5) * VOXEL_M
        out.append({"label": node["labels"].most_common(1)[0][0], "label_votes": dict(node["labels"]),
                    "observations": len(node["frames"]), "frames": sorted(node["frames"]),
                    "embedding": node["embedding"] / np.linalg.norm(node["embedding"]),
                    "voxels": centres.astype(np.float32),
                    "centroid": centres.mean(0), "bbox_min": centres.min(0) - VOXEL_M / 2,
                    "bbox_max": centres.max(0) + VOXEL_M / 2})
    return out


def relations(nodes: list[dict]) -> list[dict]:
    from scipy.spatial import cKDTree
    edges = []
    trees = [cKDTree(n["voxels"]) for n in nodes]
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            gap = max(0.0, float(trees[i].query(b["voxels"])[0].min()) - VOXEL_M)
            if gap > NEXT_TO_M:
                continue
            rel = "next to"
            for top, bottom, x, y in ((a, b, i, j), (b, a, j, i)):
                overlap_xy = (np.minimum(top["bbox_max"][:2], bottom["bbox_max"][:2])
                              - np.maximum(top["bbox_min"][:2], bottom["bbox_min"][:2])).clip(0).prod()
                area = (top["bbox_max"][:2] - top["bbox_min"][:2]).prod()
                if (0 <= top["bbox_min"][2] - bottom["bbox_max"][2] + 0.1 <= ON_GAP_M + 0.1
                        and overlap_xy >= 0.3 * area and top["bbox_min"][2] > bottom["bbox_min"][2] + 0.1):
                    rel, i_, j_ = "on", x, y
                    break
            else:
                i_, j_ = i, j
            edges.append({"source": i_, "target": j_, "relation": rel, "gap_m": round(gap, 3)})
    return edges


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(inside(RECON / "scene.npz")))
    started = time.time()
    detections = lift(detect_2d(data, OUTPUT / "detections_2d.npz"), data)
    print(f"{len(detections)} detections lifted to 3D ({time.time() - started:.0f}s)", flush=True)
    graphs = {}
    for rec in list(dict.fromkeys(data["owner"])):
        nodes = merge([d for d in detections if d["rec"] == rec])
        edges = relations(nodes)
        graphs[rec] = {"nodes": nodes, "edges": edges}
        print(f"{rec}: {len(nodes)} nodes, {len(edges)} edges", flush=True)
    np.save(OUTPUT / "graphs.npy", graphs, allow_pickle=True)
    readable = {rec: {"nodes": [{"id": i, "label": n["label"], "label_votes": n["label_votes"],
                                 "observations": n["observations"], "centroid_m": n["centroid"].round(3).tolist(),
                                 "bbox_min_m": n["bbox_min"].round(3).tolist(), "bbox_max_m": n["bbox_max"].round(3).tolist()}
                                for i, n in enumerate(g["nodes"])], "edges": g["edges"]} for rec, g in graphs.items()}
    import hashlib
    digest = hashlib.sha256(json.dumps(readable, sort_keys=True).encode()).hexdigest()
    (OUTPUT / "scene_graphs.json").write_text(json.dumps({
        "stage": "objects", "profile": config.summary(), "detections_3d": len(detections),
        "detector_seconds": {k: round(v, 1) for k, v in TIMES.items()}, "graphs": readable,
        "result_sha256": digest}, indent=1), encoding="utf-8")
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
