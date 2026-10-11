"""loop_outliers, stage 2: segment, name and describe only the changes the loop-closure outliers found.

For every change from mine.py, in the 1-2 views holding most of its outlier matches:
  SAM 2.1     prompted with up to 6 well-spread evidence pixels -> the object's outline
  3D extent   the outline lifted with that view's depth
  SigLIP 2    the masked crop, named zero-shot over ScanNet200
Reflections (mirror, window, glass) are not objects: those changes are dropped. A removed and an
added change that look alike (SigLIP) and have a similar size are one object that MOVED between
places no single view pair showed together. Writes changes.json (shared schema) + CHANGE_REPORT.md.

"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from navdelta import config, context
from navdelta.common.wording import article, is_, was
from navdelta.context import inside

RECON = context.OUTPUTS / "recon"                  # shared reconstruction (navdelta.recon)
OUTPUT = context.approach_dir("loop_outliers")
CHANGES_FILE, OBJECTS_FILE, REPORT_FILE = "changes.json", "objects.npz", "CHANGE_REPORT.md"   # rgbd_fusion renames them
from navdelta.common.device import DEVICE

SAM = ("facebook/sam2.1-hiera-base-plus", "b7320756a13354e7530a63935656d35b2f91a290")
SIGLIP = ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f")
VOCABULARY = Path(__file__).resolve().parents[1] / "voxel_diff" / "resources" / "scannet200_classes.json"
MODELS = [
    {"id": "facebook/sam2.1-hiera-base-plus", "revision": "b7320756a13354e7530a63935656d35b2f91a290",
     "license": "Apache-2.0", "use": "outline of each changed object, prompted with its outlier pixels"},
    {"id": "google/siglip2-so400m-patch14-384", "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
     "license": "Apache-2.0", "use": "zero-shot names, removed/added pairing"},
    {"id": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", "revision": "pinned source f5209afc",
     "license": "CC BY-NC-SA 4.0", "use": "depth and cross-video matches (shared reconstruction)"},
]
VIEWS = 2                      # views segmented per end of a change
PROMPTS = 6                    # SAM point prompts per view
CONF_MIN = 1.5
PAIR_SEM = 0.75                # a removed and an added change this alike are one moved object ...
PAIR_SIZE_RATIO = 2.0          # ... if their sizes are within this ratio
ALONE_MIN_MATCHES = 60         # a removed / added change nobody corroborates needs this much evidence ...
ALONE_MIN_PAIRS = 4            # ... from this many view pairs (a paired one only needs mine.py's minimum)
REFLECTIONS = {"mirror", "window", "glass", "glass door", "window frame", "picture", "poster", "tv", "television",
               "monitor", "computer monitor"}
STRUCTURE = {"wall", "floor", "ceiling", "doorframe"}


# Scale profile (navdelta/configs/*.toml): the values above are the room profile.
PROFILE_SETTINGS = config.apply(globals(), "loop_outliers.segment")


def spread(points: np.ndarray, count: int) -> np.ndarray:
    """Farthest-point sample: prompts that cover the object, not one corner of it."""
    chosen = [int(np.argmin(np.linalg.norm(points - np.median(points, 0), axis=1)))]
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < min(count, len(points)):
        chosen.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return points[chosen]


class Models:
    def __init__(self, device: str):
        from transformers import AutoModel, AutoProcessor, Sam2Model, Sam2Processor
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.sp = Sam2Processor.from_pretrained(SAM[0], revision=SAM[1])
        self.sm = Sam2Model.from_pretrained(SAM[0], revision=SAM[1]).to(device).eval()
        self.cp = AutoProcessor.from_pretrained(SIGLIP[0], revision=SIGLIP[1])
        self.cm = AutoModel.from_pretrained(SIGLIP[0], revision=SIGLIP[1], dtype=dtype).to(device).eval()
        self.dtype = dtype
        names = sorted(set(json.loads(VOCABULARY.read_text(encoding="utf-8"))["classes"]))
        with torch.no_grad():
            tok = self.cp(text=[f"a photo of a {n}." for n in names], padding="max_length", max_length=64,
                          return_tensors="pt").to(device)
            tf = self.cm.get_text_features(**tok)
            tf = getattr(tf, "pooler_output", tf)
            self.text = torch.nn.functional.normalize(tf.float(), dim=-1).cpu().numpy()
        self.names = names
        self.scale, self.bias = float(self.cm.logit_scale.exp()), float(self.cm.logit_bias)

    @torch.no_grad()
    def mask(self, image: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, float]:
        inp = self.sp(images=Image.fromarray(image), input_points=[[pixels.astype(float).tolist()]],
                      input_labels=[[[1] * len(pixels)]], return_tensors="pt").to(self.device)
        out = self.sm(**inp, multimask_output=True)
        scores = out.iou_scores[0, 0].float().cpu().numpy()
        masks = self.sp.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0].numpy()
        k = int(scores.argmax())
        return masks[k].astype(bool), float(scores[k])

    @torch.no_grad()
    def embed(self, crop: np.ndarray) -> np.ndarray:
        inp = self.cp(images=Image.fromarray(crop), return_tensors="pt").to(self.device)
        f = self.cm.get_image_features(pixel_values=inp["pixel_values"].to(self.dtype))
        f = getattr(f, "pooler_output", f)
        return torch.nn.functional.normalize(f.float(), dim=-1)[0].cpu().numpy()

    def name(self, embedding: np.ndarray) -> tuple[str, float]:
        logits = self.scale * (self.text @ embedding) + self.bias
        k = int(np.argmax(logits))
        return self.names[k], float(1 / (1 + np.exp(-logits[k])))


def upright_turns(cam2w: np.ndarray) -> int:
    up = cam2w[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    if abs(up[1]) >= abs(up[0]):
        return 0 if up[1] < 0 else 2
    return 1 if up[0] > 0 else 3


def segment_end(models: Models, scene: dict, frames: np.ndarray, pixels: np.ndarray) -> dict | None:
    """Outline, 3D extent and appearance of one end of a change, from its best views."""
    points3d, embeddings, views = [], [], []
    for k in [f for f, _ in sorted(((f, int((frames == f).sum())) for f in np.unique(frames)),
                                   key=lambda t: -t[1])][:VIEWS]:
        px = pixels[frames == k]
        if len(px) < 3:
            continue
        mask, score = models.mask(scene["rgb"][k], spread(px.astype(float), PROMPTS))
        if mask.mean() > 0.5 or mask.sum() < 50:            # a whole wall / floor, or nothing
            continue
        depth, conf, K, cam2w = scene["depth"][k], scene["conf"][k], scene["K"][k], scene["cam2w"][k]
        v, u = np.nonzero(mask & (conf > CONF_MIN))
        if len(u) < 30:
            continue
        z = depth[v, u]
        med = np.median(z)
        keep = np.abs(z - med) < 3 * 1.4826 * (np.median(np.abs(z - med)) + 1e-6) + 0.05
        u, v, z = u[keep], v[keep], z[keep]
        cam = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        points3d.append(cam @ cam2w[:3, :3].T + cam2w[:3, 3])
        ys, xs = np.nonzero(mask)
        crop = scene["rgb"][k][ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        crop[~mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]] = 128
        embeddings.append(models.embed(np.ascontiguousarray(np.rot90(crop, upright_turns(cam2w)))))
        views.append({"frame": str(scene["frames"][k]), "sam_score": round(score, 3), "mask_px": int(mask.sum())})
    if not points3d:
        return None
    pts = np.concatenate(points3d)
    lo, hi = np.percentile(pts, 3, axis=0), np.percentile(pts, 97, axis=0)
    emb = np.mean(embeddings, axis=0)
    emb /= np.linalg.norm(emb)
    label, score = models.name(emb)
    return {"centroid": (lo + hi) / 2, "bbox_min": lo, "bbox_max": hi, "embedding": emb, "label": label,
            "label_score": score, "views": views, "points": pts[:: max(1, len(pts) // 4000)].astype(np.float32)}


def feature_end(points: np.ndarray, other: dict) -> dict:
    """An end of a move SAM found no outline for: the box of its own matched features, named and
    embedded as the other end (the RANSAC motion says they are one object)."""
    lo, hi = np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)
    return {"centroid": (lo + hi) / 2, "bbox_min": lo, "bbox_max": hi, "embedding": other["embedding"],
            "label": other["label"], "label_score": other["label_score"], "views": [],
            "points": points[:: max(1, len(points) // 4000)].astype(np.float32)}


def geometry(end: dict) -> dict:
    return {"centroid_m": end["centroid"].round(3).tolist(), "bbox_min_m": end["bbox_min"].round(3).tolist(),
            "bbox_max_m": end["bbox_max"].round(3).tolist(), "views": end["views"]}


def size(end: dict) -> float:
    return float(np.max(end["bbox_max"] - end["bbox_min"]))


def main() -> None:
    work = OUTPUT
    started = time.time()
    scene = dict(np.load(inside(RECON / "scene.npz")))
    o = dict(np.load(work / "outliers.npz"))
    mined = json.loads((work / "outlier_changes.json").read_text())
    rec_a, rec_b = list(dict.fromkeys(scene["owner"]))
    models = Models(DEVICE)
    print(f"models on {DEVICE} ({time.time() - started:.0f}s)", flush=True)

    ends, dropped = [], []
    for c in mined["changes"]:
        rec = {"id": c["id"], "type": c["type"], "evidence_matches": c["evidence_matches"], "view_pairs": c["view_pairs"]}
        if c["type"] == "moved":
            ia, ib = np.array(c["idx_before"]), np.array(c["idx_after"])
            rec["before"] = segment_end(models, scene, o["frame_a"][ia], o["pix_a"][ia])
            rec["after"] = segment_end(models, scene, o["frame_b"][ib], o["pix_b"][ib])
            rec.update(distance_m=c["distance_m"], turn_deg=c["turn_deg"])
            if rec["before"] is None and rec["after"] is not None:
                rec["before"] = feature_end(o["src"][ia], rec["after"])
            elif rec["after"] is None and rec["before"] is not None:
                rec["after"] = feature_end(o["dst"][ib], rec["before"])
        elif c["type"] == "removed":
            idx = np.array(c["idx"])
            rec["before"] = segment_end(models, scene, o["frame_a"][idx], o["pix_a"][idx])
        else:
            idx = np.array(c["idx"])
            rec["after"] = segment_end(models, scene, o["frame_b"][idx], o["pix_b"][idx])
        parts = [rec[k] for k in ("before", "after") if rec.get(k)]
        if not parts:
            dropped.append({**c, "why": "no outline found"})
            continue
        if any(p["label"] in REFLECTIONS | STRUCTURE for p in parts):
            dropped.append({"id": c["id"], "type": c["type"], "why": f"{parts[0]['label']}: not a movable object"})
            continue
        ends.append(rec)
        print(f"  {c['id']:<5} {c['type']:<8} " + " -> ".join(f"{p['label']} ({p['label_score']:.2f})" for p in parts),
              flush=True)

    # removed + added that look alike: one object moved where no view pair showed both places
    rem = [e for e in ends if e["type"] == "removed"]
    add = [e for e in ends if e["type"] == "added"]
    if rem and add:
        sim = np.array([[float(r["before"]["embedding"] @ a["after"]["embedding"]) for a in add] for r in rem])
        ratio = np.array([[max(size(r["before"]), size(a["after"])) / max(1e-6, min(size(r["before"]), size(a["after"])))
                           for a in add] for r in rem])
        sim[(sim < PAIR_SEM) | (ratio > PAIR_SIZE_RATIO)] = 0.0
        for i, j in zip(*linear_sum_assignment(-sim)):
            if sim[i, j] > 0:
                r, a = rem[i], add[j]
                d = float(np.linalg.norm(a["after"]["centroid"][:2] - r["before"]["centroid"][:2]))
                ends.append({"id": f"{r['id']}+{a['id']}", "type": "moved", "before": r["before"], "after": a["after"],
                             "distance_m": round(d, 3), "turn_deg": None, "paired_by_appearance": round(sim[i, j], 3),
                             "evidence_matches": r["evidence_matches"] + a["evidence_matches"],
                             "view_pairs": r["view_pairs"] + a["view_pairs"]})
                r["merged"] = a["merged"] = True
    ends = [e for e in ends if not e.get("merged")]
    weak = [e for e in ends if e["type"] != "moved" and
            (e["evidence_matches"] < ALONE_MIN_MATCHES or e["view_pairs"] < ALONE_MIN_PAIRS)]
    dropped += [{"id": e["id"], "type": e["type"], "why": f"weak evidence alone ({e['evidence_matches']} matches, "
                 f"{e['view_pairs']} view pairs)"} for e in weak]
    ends = [e for e in ends if e not in weak]

    records, counts = [], {}
    for e in sorted(ends, key=lambda e: ("moved", "removed", "added").index(e["type"])):
        counts[e["type"]] = counts.get(e["type"], 0) + 1
        rid = {"moved": "M", "removed": "OR", "added": "OA"}[e["type"]] + str(counts[e["type"]])
        if e["type"] == "moved":
            emb = e["before"]["embedding"] + e["after"]["embedding"]
            label = models.name(emb / np.linalg.norm(emb))[0]
            turn = e.get("turn_deg")
            how = (f"measured from {e['evidence_matches']} of its own features matched across both videos"
                   if e.get("paired_by_appearance") is None else
                   f"it vanished from one place and an object that looks the same (similarity "
                   f"{e['paired_by_appearance']:.2f}) appeared at the other")
            sentence = (f"The {label} {was(label)} moved {e['distance_m']:.2f} m"
                        + (f" and turned by about {turn:.0f}°" if turn is not None and turn >= 20 else "") + ".")
            evidence = f"Evidence: {how}; seen in {e['view_pairs']} view pairs."
            text = f"{sentence} {evidence}"
            rec = {"id": rid, "type": "moved", "label": label, "title": f"{label} moved {e['distance_m']:.2f} m",
                   "description": text, "sentence": sentence, "evidence_text": evidence, "before": geometry(e["before"]), "after": geometry(e["after"]),
                   "motion": {"distance_m": e["distance_m"], "turn_deg": turn},
                   "evidence": {"outlier_matches": e["evidence_matches"], "view_pairs": e["view_pairs"],
                                "paired_by_appearance": e.get("paired_by_appearance")}}
        else:
            end = e["before"] if e["type"] == "removed" else e["after"]
            verb = f"{is_(end['label'])} no longer there" if e["type"] == "removed" else "appeared"
            sentence = f"{article(end['label'])} ({size(end):.2f} m) {verb}."
            evidence = (f"Evidence: {e['evidence_matches']} of its features, seen in {e['view_pairs']} view pairs, "
                        f"now match empty space in the other video.")
            text = f"{sentence} {evidence}"
            rec = {"id": rid, "type": e["type"], "label": end["label"], "title": f"{end['label']} {e['type']}",
                   "description": text, "sentence": sentence, "evidence_text": evidence,
                   "object": {**geometry(end), "present_in": rec_a if e["type"] == "removed" else rec_b},
                   "evidence": {"outlier_matches": e["evidence_matches"], "view_pairs": e["view_pairs"]}}
        records.append(rec)
    summary = ("Between the two recordings, " + ", ".join(
        f"{n} object{'s' if n > 1 else ''} {t} ({', '.join(r['label'] for r in records if r['type'] == t)})"
        for t, n in counts.items()) + "." if counts else "No changes found.")
    out = {"schema_version": "1.0.0", "approach": "loop_outliers", "recordings": {"before": rec_a, "after": rec_b},
           "common_frame": f"{rec_a} tracking frame after the joint pose graph; metres; Z up; floor at z = 0",
           "summary": summary, "counts": counts, "changes": records, "dropped": dropped,
           "loop_closure": {k: mined[k] for k in ("matches", "static_share", "outliers", "moved_surfaces",
                                                  "vanished", "appeared")},
           "models": MODELS, "profile": config.summary(),
           "learned_models_trained_or_finetuned": False, "manual_labels_used": False}
    out["result_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()
    (work / CHANGES_FILE).write_text(json.dumps(out, indent=2), encoding="utf-8")
    shapes = {}
    for r, e in zip(records, sorted(ends, key=lambda e: ("moved", "removed", "added").index(e["type"]))):
        for side in ("before", "after"):
            if e.get(side):
                shapes[f"{r['id']}_{side}"] = e[side]["points"]
    np.savez_compressed(work / OBJECTS_FILE, **shapes)
    lines = ["# What changed between the two recordings", "", summary, ""] + \
            [f"- **{r['id']}** ({r['type']}): {r['description']}" for r in records]
    if REPORT_FILE:
        (work / REPORT_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
