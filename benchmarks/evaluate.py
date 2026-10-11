"""Score an approach's changes.json against the ground truth of one pair.

  python benchmarks/evaluate.py --pair benchmarks/work/changesim/data/<pair> --changes <changes.json>          --name <approach> [--poses-dir <aligned_frame_poses>] [--scene <scene.npz>] [--z-offset <m>]

Ground truth (the pair's gt.json), in the "before" recording's frame:
  ChangeSim  gt["changes"]: every changed object (removed / added / rotated / replaced) with its box
             in the run(s) where it exists, from changesim/convert.py
  3RScan     instance annotations (rigid / removed / nonrigid); moved = box centre moved more than
             MOVE_MIN_M or turned more than TURN_MIN_DEG, nonrigid and barely moved ones are neutral
Structural classes (wall, floor, ceiling) are never changes.

A claim (a reported change other than depth-only / small) "hits" a ground-truth object when one
of its positions -- before and/or after -- lies inside that object's box in the matching recording,
grown by MARGIN_M on every side.
  precision  claims that hit a changed object / claims that hit a changed or no object
  recall     changed objects hit by at least one claim / changed objects
  type       hit objects whose hitting claim has a compatible type (moved <-> moved/shifted/rotated)
  move error for hit moved objects reported as moved: |predicted - true centre displacement|
  label      hit objects whose predicted label shares a word with the ground-truth label
Registration: the second recording's camera centres as placed by the approach vs. the ground truth.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

MOVE_MIN_M = 0.10
TURN_MIN_DEG = 15.0
MARGIN_M = 0.30
DETECTABLE_M = 0.30           # largest box side; the approaches' stated detection floor is ~15 cm
STRUCTURE = {"wall", "floor", "ceiling"}
CLAIM_TYPES = {"moved", "shifted", "rotated", "removed", "added", "changed_at_landmark", "surface_changed"}
MOVE_TYPES = {"moved", "shifted", "rotated"}
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
Q_YUP_TO_ZUP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def inside(point: np.ndarray, box: dict, margin: float) -> bool:
    local = np.array(box["axes"]) @ (np.asarray(point) - np.array(box["centroid"]))
    return bool(np.all(np.abs(local) <= np.array(box["lengths"]) / 2 + margin))


def heading(box: dict) -> float | None:
    """Heading (deg, mod 180) of the longer horizontal box axis; None for a near-square footprint."""
    axes, lengths = np.array(box["axes"]), np.array(box["lengths"])
    horizontal = [k for k in range(3) if abs(axes[k][2]) < 0.5]
    if len(horizontal) < 2:
        return None
    a, b = sorted(horizontal, key=lambda k: -lengths[k])[:2]
    if lengths[a] < 1.2 * lengths[b]:
        return None
    return float(np.degrees(np.arctan2(axes[a][1], axes[a][0])) % 180.0)


def ground_truth(gt: dict) -> tuple[dict, set]:
    if "changes" in gt:                       # datasets that list their changed objects directly (ChangeSim)
        changed = {}
        for c in gt["changes"]:
            g = {"type": c["type"], "label": c["label"], "size": c["size"]}
            if c.get("before"):
                g["before"] = c["before"]
            if c.get("after"):
                g["after"] = c["after"]
            if c["type"] == "moved":
                g["shift"] = np.array(c["after"]["centroid"]) - np.array(c["before"]["centroid"])
            changed[c["id"]] = g
        return changed, set()
    before, after = gt["objects_before"], gt["objects_after"]
    changed, neutral = {}, {str(i) for i in gt["nonrigid"]}
    for r in gt["rigid"]:
        b, a = before.get(str(r["before"])), after.get(str(r["after"]))
        if b is None or a is None or b["label"] in STRUCTURE:
            continue
        shift = np.array(a["centroid"]) - np.array(b["centroid"])
        hb, ha = heading(b), heading(a)
        turn = None if hb is None or ha is None or r.get("symmetry", 0) else min(abs(hb - ha), 180 - abs(hb - ha))
        if np.linalg.norm(shift) > MOVE_MIN_M or (turn is not None and turn > TURN_MIN_DEG):
            changed[str(r["before"])] = {"type": "moved", "label": b["label"], "before": b, "after": a,
                                         "shift": shift, "turn": turn, "size": float(max(b["lengths"]))}
        else:
            neutral.add(str(r["before"]))
    for i in gt["removed"]:
        b = before.get(str(i))
        if b is not None and b["label"] not in STRUCTURE:
            changed[str(i)] = {"type": "removed", "label": b["label"], "before": b, "size": float(max(b["lengths"]))}
    ambiguous = {str(x.get("instance_target")) for x in gt.get("ambiguity", []) if isinstance(x, dict)}
    for i, a in after.items():
        if i not in before and a["label"] not in STRUCTURE and i not in ambiguous:
            changed[f"+{i}"] = {"type": "added", "label": a["label"], "after": a, "size": float(max(a["lengths"]))}
    return changed, neutral


def claims(changes: dict, before_name: str, z_offset: float) -> list[dict]:
    off = np.array([0.0, 0.0, z_offset])
    out = []
    for c in changes["changes"]:
        if c["type"] not in CLAIM_TYPES:
            continue
        pos = []
        if "before" in c and "after" in c:
            pos = [("before", np.array(c["before"]["centroid_m"]) + off), ("after", np.array(c["after"]["centroid_m"]) + off)]
        elif "object" in c:
            state = c["object"].get("present_in")
            state = ("before" if state == before_name else "after") if state else \
                ("before" if c["type"] == "removed" else "after")
            pos = [(state, np.array(c["object"]["centroid_m"]) + off)]
        out.append({"id": c["id"], "type": c["type"], "label": c.get("label") or "", "positions": pos,
                    "shift": (pos[1][1] - pos[0][1]) if len(pos) == 2 else None})
    return out


def words(label: str) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", label.lower()) if len(w) > 2}


def registration(gt: dict, pair: Path, poses_dir: Path | None, scene: Path | None, z_offset: float) -> dict | None:
    """Rescan camera centres as placed by the approach vs. ground truth (reference frame)."""
    name = gt["recordings"]["after"]
    T = np.array(gt["rescan_to_reference"])
    cams = pair / f"{name}_imgs" / "keyframes" / "cameras"
    truth = {}
    for f in cams.glob("*.json"):
        c = json.loads(f.read_text())
        A = np.eye(4)
        A[:3, :3] = [[c[f"t_{r}{k}"] for k in range(3)] for r in range(3)]
        A[:3, 3] = [c["t_03"], c["t_13"], c["t_23"]]
        truth[f.stem] = T @ (Q_YUP_TO_ZUP @ A @ CV_TO_ARKIT)
    est = {}
    if poses_dir is not None:
        for f in (poses_dir / name).glob("*.json"):
            est[f.stem] = np.array(json.loads(f.read_text())["T_camera_to_common"])
    if scene is not None:
        s = np.load(scene)
        for stem, owner, P in zip(s["frames"], s["owner"], s["cam2w"]):
            if owner == name:
                P = P.copy()
                P[2, 3] += z_offset
                est[str(stem)] = P
    common = sorted(set(est) & set(truth))
    if not common:
        return None
    pos = np.array([np.linalg.norm(est[k][:3, 3] - truth[k][:3, 3]) for k in common])
    rot = np.array([np.degrees(np.arccos(np.clip((np.trace(est[k][:3, :3].T @ truth[k][:3, :3]) - 1) / 2, -1, 1)))
                    for k in common])
    return {"frames": len(common), "position_error_median_cm": round(float(np.median(pos) * 100), 1),
            "position_error_p90_cm": round(float(np.percentile(pos, 90) * 100), 1),
            "rotation_error_median_deg": round(float(np.median(rot)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--changes", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--poses-dir")
    ap.add_argument("--scene")
    ap.add_argument("--z-offset", type=float, default=0.0, help="add to predicted z to reach the reference frame")
    args = ap.parse_args()
    pair = Path(args.pair)
    gt = json.loads((pair / "gt.json").read_text())
    changes = json.loads(Path(args.changes).read_text(encoding="utf-8"))
    changed, neutral = ground_truth(gt)
    preds = claims(changes, gt["recordings"]["before"], args.z_offset)

    def hits(claim: dict, pool: dict) -> list[str]:
        found = []
        for key, g in pool.items():
            for state, p in claim["positions"]:
                box = g.get(state)
                if box is not None and inside(p, box, MARGIN_M):
                    found.append(key)
                    break
        return found

    neutral_boxes = {i: {"before": gt["objects_before"].get(i), "after": gt["objects_after"].get(i)} for i in neutral}
    label_scored = "objects_before" in gt     # ChangeSim's change masks carry no object names
    tp = fp = neu = 0
    hit_by: dict[str, list[dict]] = {}
    rows = []
    for c in preds:
        h = hits(c, changed)
        if h:
            tp += 1
            for k in h:
                hit_by.setdefault(k, []).append(c)
        elif hits(c, neutral_boxes):
            neu += 1
        else:
            fp += 1
        rows.append({"claim": c["id"], "type": c["type"], "label": c["label"], "hits": [changed[k]["label"] + f" ({k})" for k in h]})
    in_place = MOVE_TYPES | {"changed_at_landmark", "surface_changed"}
    compatible = {"moved": MOVE_TYPES, "removed": {"removed", "changed_at_landmark", "moved"},
                  "added": {"added", "changed_at_landmark", "moved"},
                  "rotated": in_place, "replaced": in_place | {"removed", "added"}}
    type_ok = label_ok = 0
    move_err = []
    per_object = []
    for k, g in changed.items():
        cs = hit_by.get(k, [])
        best = next((c for c in cs if c["type"] in compatible[g["type"]]), cs[0] if cs else None)
        ok_type = best is not None and best["type"] in compatible[g["type"]]
        ok_label = best is not None and bool(words(best["label"]) & words(g["label"]))
        type_ok += ok_type
        label_ok += ok_label
        err = None
        if best is not None and g["type"] == "moved" and best["shift"] is not None:
            err = float(np.linalg.norm(best["shift"] - g["shift"]))
            move_err.append(err)
        per_object.append({"object": k, "label": g["label"], "type": g["type"], "size_m": round(g["size"], 2),
                           "detected_by": best["id"] if best else None, "predicted_type": best["type"] if best else None,
                           "predicted_label": best["label"] if best else None,
                           "true_shift_m": round(float(np.linalg.norm(g["shift"])), 2) if "shift" in g else None,
                           "move_error_m": round(err, 2) if err is not None else None})
    n = len(changed)
    det = [o for o in per_object if o["size_m"] >= DETECTABLE_M]
    hit = sum(o["detected_by"] is not None for o in per_object)
    hit_det = sum(o["detected_by"] is not None for o in det)
    types = [t for t in ("moved", "rotated", "replaced", "removed", "added") if any(o["type"] == t for o in per_object)]
    by_type = {t: f"{sum(o['detected_by'] is not None for o in per_object if o['type'] == t)}/"
                  f"{sum(o['type'] == t for o in per_object)}" for t in types}
    by_type_gt = {t: sum(o["type"] == t for o in per_object) for t in ("moved", "removed", "added")}
    result = {"approach": args.name, "pair": f"{gt['reference']} -> {gt['rescan']}",
              "ground_truth": {"changed": n, "detectable_size": len(det), "neutral": len(neutral), "by_type": {
                  t: sum(o["type"] == t for o in per_object) for t in types}},
              "claims": len(preds), "true_positive_claims": tp, "false_positive_claims": fp, "neutral_claims": neu,
              "precision": round(tp / max(1, tp + fp), 3),
              "recall": round(hit / max(1, n), 3), "recall_detectable": round(hit_det / max(1, len(det)), 3),
              "recall_by_type": by_type,
              "f1": round(2 * tp / max(1, tp + fp) * hit / max(1, n) / max(1e-9, tp / max(1, tp + fp) + hit / max(1, n)), 3),
              "type_accuracy": round(type_ok / max(1, hit), 3),
              "label_accuracy": round(label_ok / max(1, hit), 3) if label_scored else None,
              "move_error_median_m": round(float(np.median(move_err)), 3) if move_err else None,
              "registration": registration(gt, pair, Path(args.poses_dir) if args.poses_dir else None,
                                           Path(args.scene) if args.scene else None, args.z_offset),
              "objects": per_object, "claims_detail": rows}
    out = Path(args.changes).with_name(f"eval_{args.name}.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("objects", "claims_detail")}, indent=2))


if __name__ == "__main__":
    main()
