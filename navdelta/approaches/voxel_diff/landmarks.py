"""Step 4a: static landmarks (desk, bed, shelves, door, ...) for spatial language.

Descriptions such as "moved next to the desk" need to know where the static
furniture is. Landmarks are found with the same off-the-shelf models as Step 3,
without any fixed object list:

  1. Florence-2 <OD> (its own open label space) runs on every
     FRAME_STRIDE-th keyframe of both recordings, rotated upright.
  2. Each detection box is segmented by SAM 2.1; the masked depth pixels are
     back-projected into the shared voxel grid, one vote per frame and label.
  3. A voxel belongs to a label when at least MIN_FRAME_VOTES frames agree;
     connected pieces become landmark instances.
  4. Only static, furniture-sized instances are kept: most of their voxels are
     occupied in BOTH recordings (moved objects drop out by construction), they
     are mostly not wall/floor/ceiling (windows and doors excepted: they are
     wall openings), and they fit in the room.
  5. Instances of different labels that overlap (e.g. "desk" / "table") are
     merged; the better supported label names the landmark.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from navdelta.approaches.voxel_diff.paths import DATA, OUTPUT, SEMANTICS, SESSIONS, inside_workspace

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from navdelta.common.device import DEVICE, DTYPE  # noqa: E402
from navdelta.common.frames import load_session  # noqa: E402
from navdelta.approaches.voxel_diff.labels import (FLORENCE_MODEL, FLORENCE_REVISION, SAM_MODEL,  # noqa: E402
                                                   SAM_REVISION,
                           normalise_label, rotate_box, upright_turns)


FRAME_STRIDE = 3
DEPTH_STRIDE = 2
DEPTH_MIN_M, DEPTH_MAX_M = 0.25, 5.0
MIN_FRAME_VOTES = 3
MIN_LANDMARK_VOXELS = 40           # ~0.1 m^2 of surface at 5 cm
MAX_LANDMARK_EXTENT_M = 3.0
MIN_STATIC_FRACTION = 0.5          # voxels occupied in both recordings (+-1 voxel)
MAX_STRUCTURAL_FRACTION = 0.6      # mostly wall/floor/ceiling -> room structure, not a landmark
MERGE_OVERLAP = 0.3
# Windows and doors are openings in a wall, so lying on a wall plane does not
# disqualify them (they still have to be static).
WALL_FEATURES = {"window", "door"}
# Words that name the scene or a person rather than a piece of furniture.
NOT_LANDMARKS = {"house", "room", "building", "person", "human face", "man", "woman", "wall", "floor", "ceiling"}
CUBE = np.ones((3, 3, 3), dtype=bool)

# Scale profile (navdelta/configs/*.toml): the values above are the room profile; a large space
# overrides some of them. Applied before anything below derives from them.
from navdelta import config  # noqa: E402
PROFILE_SETTINGS = config.apply(globals(), "voxel_diff.landmarks")


def back_project(frame: dict, mask: np.ndarray) -> np.ndarray:
    depth = frame["depth_m"][::DEPTH_STRIDE, ::DEPTH_STRIDE]
    intr = frame["intrinsics"]
    scale_x = intr["width"] / frame["depth_m"].shape[1]
    scale_y = intr["height"] / frame["depth_m"].shape[0]
    rows, cols = np.nonzero((depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M))
    u = (cols * DEPTH_STRIDE + 0.5) * scale_x - 0.5
    v = (rows * DEPTH_STRIDE + 0.5) * scale_y - 0.5
    inside = mask[np.clip(np.round(v).astype(int), 0, mask.shape[0] - 1),
                  np.clip(np.round(u).astype(int), 0, mask.shape[1] - 1)]
    z = depth[rows, cols][inside].astype(np.float64)
    camera = np.column_stack([(u[inside] - intr["cx"]) / intr["fx"] * z,
                              (v[inside] - intr["cy"]) / intr["fy"] * z, z])
    pose = frame["pose"]
    return camera @ pose[:3, :3].T + pose[:3, 3]


def main() -> None:
    started = time.time()
    device = DEVICE
    grid = np.load(str(inside_workspace(OUTPUT / "voxel_change_grid.npz")))
    origin, voxel = grid["origin"], float(grid["voxel_m"])
    dims = np.array([int(d) for d in grid["dims"]])
    shape, size = tuple(dims), int(np.prod(dims))

    def flat_index(points: np.ndarray) -> np.ndarray:
        ijk = np.floor((points - origin) / voxel).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
        return np.ravel_multi_index(ijk[ok].T, shape)

    from transformers import AutoProcessor, Florence2ForConditionalGeneration, Sam2Model, Sam2Processor
    florence_processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, revision=FLORENCE_REVISION)
    florence = Florence2ForConditionalGeneration.from_pretrained(FLORENCE_MODEL, revision=FLORENCE_REVISION,
                                                                 dtype=DTYPE).to(device).eval()
    sam_processor = Sam2Processor.from_pretrained(SAM_MODEL, revision=SAM_REVISION)
    sam = Sam2Model.from_pretrained(SAM_MODEL, revision=SAM_REVISION).to(device).eval()

    votes: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(size, dtype=np.uint16))
    detections_per_label: dict[str, int] = defaultdict(int)
    frames_used = 0
    for session in SESSIONS:
        frames = load_session(session)[::FRAME_STRIDE]
        for frame in frames:
            image = np.asarray(Image.open(inside_workspace(DATA / frame["rgb"])).convert("RGB"))
            height, width = image.shape[:2]
            turns = upright_turns(frame["pose"])
            upright = Image.fromarray(np.ascontiguousarray(np.rot90(image, turns)))
            inputs = florence_processor(text="<OD>", images=upright, return_tensors="pt").to(device, DTYPE)
            with torch.no_grad():
                generated = florence.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                              max_new_tokens=256, num_beams=3, do_sample=False)
            text = florence_processor.batch_decode(generated, skip_special_tokens=False)[0]
            result = florence_processor.post_process_generation(text, task="<OD>", image_size=upright.size)["<OD>"]
            boxes, labels = [], []
            for box, label in zip(result["bboxes"], result["labels"]):
                label = normalise_label(label)
                if not label or label in NOT_LANDMARKS:
                    continue
                # Back from the upright image to the stored image.
                boxes.append(rotate_box(tuple(box), (4 - turns) % 4, upright.size[0], upright.size[1]))
                labels.append(label)
            frames_used += 1
            if not boxes:
                continue
            sam_inputs = sam_processor(images=Image.fromarray(image), input_boxes=[[list(map(float, b)) for b in boxes]],
                                       return_tensors="pt").to(device)
            with torch.no_grad():
                output = sam(**sam_inputs, multimask_output=False)
            masks = sam_processor.post_process_masks(output.pred_masks.cpu(), sam_inputs["original_sizes"])[0]
            seen_this_frame: dict[str, set] = defaultdict(set)
            for mask, label in zip(masks[:, 0].numpy(), labels):
                points = back_project(frame, mask)
                if len(points):
                    seen_this_frame[label].update(np.unique(flat_index(points)).tolist())
                    detections_per_label[label] += 1
            for label, cells in seen_this_frame.items():
                votes[label][np.fromiter(cells, dtype=np.int64)] += 1
        print(f"[{session}] {len(frames)} frames processed ({time.time() - started:.0f}s)", flush=True)
    del florence, sam
    torch.cuda.empty_cache()

    occupied1 = ndimage.binary_dilation((grid["state_scan1"] == 1).reshape(shape), structure=CUBE).ravel()
    occupied2 = ndimage.binary_dilation((grid["state_scan2"] == 1).reshape(shape), structure=CUBE).ravel()
    static = occupied1 & occupied2
    structural = grid["structural_mask"]

    instances = []
    for label in sorted(votes):
        cells = votes[label] >= MIN_FRAME_VOTES
        if cells.sum() < MIN_LANDMARK_VOXELS:
            continue
        bridged = ndimage.binary_dilation(cells.reshape(shape), structure=CUBE)
        components, count = ndimage.label(bridged, structure=CUBE)
        components = components.ravel()
        for component in range(1, count + 1):
            flat = np.flatnonzero(cells & (components == component))
            if len(flat) < MIN_LANDMARK_VOXELS:
                continue
            ijk = np.column_stack(np.unravel_index(flat, shape))
            centres = origin + (ijk + 0.5) * voxel
            extent = centres.max(axis=0) - centres.min(axis=0) + voxel
            static_fraction = float(static[flat].mean())
            structural_fraction = float(structural[flat].mean())
            keep = (extent[:2].max() <= MAX_LANDMARK_EXTENT_M and static_fraction >= MIN_STATIC_FRACTION
                    and (structural_fraction <= MAX_STRUCTURAL_FRACTION or label in WALL_FEATURES))
            instances.append({"label": label, "flat": flat, "votes": int(votes[label][flat].sum()),
                              "max_frames": int(votes[label][flat].max()), "extent": extent,
                              "static_fraction": static_fraction, "structural_fraction": structural_fraction,
                              "kept": keep})
    rejected = [i for i in instances if not i["kept"]]
    instances = sorted((i for i in instances if i["kept"]), key=lambda i: -i["votes"])

    # Merge overlapping instances of different labels into one landmark.
    landmarks = []
    for instance in instances:
        cells = set(instance["flat"].tolist())
        for landmark in landmarks:
            overlap = len(cells & landmark["cells"]) / min(len(cells), len(landmark["cells"]))
            if overlap >= MERGE_OVERLAP:
                landmark["cells"] |= cells
                landmark["aliases"].append(instance["label"])
                landmark["votes"] += instance["votes"]
                break
        else:
            landmarks.append({"label": instance["label"], "cells": cells, "aliases": [], "votes": instance["votes"]})

    counts = defaultdict(int)
    records, arrays = [], {}
    for landmark in sorted(landmarks, key=lambda l: -l["votes"]):
        counts[landmark["label"]] += 1
        landmark_id = f"L{len(records) + 1}"
        flat = np.array(sorted(landmark["cells"]), dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(flat, shape))
        centres = origin + (ijk + 0.5) * voxel
        arrays[landmark_id] = flat
        records.append({"id": landmark_id, "label": landmark["label"],
                        "aliases": sorted(set(landmark["aliases"]) - {landmark["label"]}),
                        "voxels": int(len(flat)), "frame_votes": landmark["votes"],
                        "centroid_m": centres.mean(axis=0).round(3).tolist(),
                        "bbox_min_m": (centres.min(axis=0) - voxel / 2).round(3).tolist(),
                        "bbox_max_m": (centres.max(axis=0) + voxel / 2).round(3).tolist()})
    for record in records:
        if counts[record["label"]] > 1:
            record["name"] = f"{record['label']} at ({record['centroid_m'][0]:+.1f}, {record['centroid_m'][1]:+.1f})"
        else:
            record["name"] = record["label"]

    np.savez_compressed(str(inside_workspace(SEMANTICS / "landmark_voxels.npz")), **arrays)
    output = {
        "schema_version": "1.0.0",
        "stage": "step4a_static_landmarks",
        "learned_models_used": [{"id": FLORENCE_MODEL, "revision": FLORENCE_REVISION, "use": "<OD>"},
                                {"id": SAM_MODEL, "revision": SAM_REVISION, "use": "box -> mask"}],
        "parameters": {"frame_stride": FRAME_STRIDE, "min_frame_votes": MIN_FRAME_VOTES,
                       "min_landmark_voxels": MIN_LANDMARK_VOXELS, "max_landmark_extent_m": MAX_LANDMARK_EXTENT_M,
                       "min_static_fraction": MIN_STATIC_FRACTION,
                       "max_structural_fraction": MAX_STRUCTURAL_FRACTION, "merge_overlap": MERGE_OVERLAP,
                       "not_landmarks": sorted(NOT_LANDMARKS), "wall_features": sorted(WALL_FEATURES)},
        "frames_used": frames_used,
        "detections_per_label": dict(sorted(detections_per_label.items(), key=lambda kv: -kv[1])),
        "landmarks": records,
        "rejected_instances": [{"label": i["label"], "voxels": int(len(i["flat"])),
                                "static_fraction": round(i["static_fraction"], 2),
                                "structural_fraction": round(i["structural_fraction"], 2),
                                "extent_m": i["extent"].round(2).tolist()} for i in rejected],
    }
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode())
    for key in sorted(arrays):
        digest.update(arrays[key].tobytes())
    output["result_sha256"] = digest.hexdigest()
    output["runtime_s"] = round(time.time() - started, 1)
    (inside_workspace(SEMANTICS) / "landmarks.json").write_text(json.dumps(output, indent=2))

    background = origin + (np.column_stack(np.unravel_index(
        np.flatnonzero((grid["state_scan1"] == 1) | (grid["state_scan2"] == 1)), shape)) + 0.5) * voxel
    figure, axis = plt.subplots(figsize=(9, 10), dpi=130, facecolor="white")
    axis.scatter(background[:, 0], background[:, 1], s=0.5, c="#c9ccd3", linewidths=0, rasterized=True)
    palette = plt.get_cmap("tab10")
    for index, record in enumerate(records):
        pts = origin + (np.column_stack(np.unravel_index(arrays[record["id"]], shape)) + 0.5) * voxel
        axis.scatter(pts[:, 0], pts[:, 1], s=3, color=palette(index % 10), linewidths=0, alpha=0.8)
        axis.annotate(f"{record['id']} {record['name']}", record["centroid_m"][:2], fontsize=8, fontweight="bold",
                      color=palette(index % 10))
    axis.set_aspect("equal")
    axis.set_xlim(background[:, 0].min() - 0.2, background[:, 0].max() + 1.2)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_title("Step 4a - static landmarks\n(Florence-2 <OD> + SAM 2.1, voted in 3D, present in both scans)")
    figure.tight_layout()
    figure.savefig(str(inside_workspace(OUTPUT / "qa" / "step4_landmarks.png")))
    plt.close(figure)

    print("Detections per label:", output["detections_per_label"])
    for record in records:
        print(f"  {record['id']:>3} {record['name']:<28} {record['voxels']:5d} vox  votes {record['frame_votes']:6d}  "
              f"aliases {record['aliases']}  bbox {record['bbox_min_m']} .. {record['bbox_max_m']}")
    for item in output["rejected_instances"]:
        print(f"  rejected {item}")
    print(f"Result sha256: {output['result_sha256']}")
    print(f"Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
