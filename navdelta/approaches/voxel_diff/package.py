"""Step 5: changes.json, the before/after photo gallery, and the 3D viewer's data.

Everything here is read from the outputs of Steps 1-4; no learned model runs.
All paths below are inside <outputs>/voxel_diff/.

  changes.json                 one record per change: what (label + confidence),
                               where (before/after geometry, places), how (motion),
                               the evidence behind it, and links to media
  viewer/gallery/*.jpg         a before and an after photo per change, from the
                               most similar pair of viewpoints in the two
                               recordings; the object is filled where it is and
                               outlined (dashed) where it was / will be
  viewer/data/*.js             point clouds and change records for viewer/index.html
                               (plain <script> data files, so the viewer also
                               works when opened straight from disk)
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from navdelta.approaches.voxel_diff.paths import (AFTER, ALIGNMENT, BEFORE, DATA as RECORDINGS, OUTPUT,
                                                  SCAN_CLOUDS, SEMANTICS, SESSIONS, VIEWER, inside_workspace,
                                                  out_rel)
from navdelta.common.frames import load_session, project

GALLERY = VIEWER / "gallery"
VIEWER_DATA = VIEWER / "data"
VIEWER_TEMPLATE = Path(__file__).resolve().parents[2] / "viewer" / "index.html"

ROOM_VOXEL_M = 0.02
OBJECT_VOXEL_M = 0.015
CANDIDATE_FRAMES = 25
VIEW_DISTANCE_SCALE_M = 0.6
VIEW_ANGLE_SCALE_DEG = 30.0
MIN_PAIR_VISIBILITY = 0.10
CROP_PAD = 0.40
MIN_CROP_PX = 240
PHOTO_SIZE = (480, 640)            # upright width x height
JPEG_QUALITY = 85

BGR = {"old": (61, 38, 215), "new": (75, 158, 27), "surface": (0, 161, 230), "shift": (179, 79, 122),
       "depth": (150, 128, 110)}


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def encode_points(points: np.ndarray, colors: np.ndarray | None = None) -> dict:
    """uint16-quantised positions (+ uint8 colours) as base64, decoded in the viewer."""
    low, high = points.min(axis=0), points.max(axis=0)
    scale = np.maximum(high - low, 1e-6) / 65535.0
    quantised = np.round((points - low) / scale).astype("<u2")
    out = {"count": int(len(points)), "origin": low.round(5).tolist(), "scale": scale.tolist(),
           "positions": b64(quantised)}
    if colors is not None:
        out["colors"] = b64(np.clip(np.round(colors * 255), 0, 255).astype(np.uint8))
    return out


def downsample(points: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud = cloud.voxel_down_sample(voxel)
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def read_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cloud = o3d.io.read_point_cloud(str(inside_workspace(path)))
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def upright_turns(pose: np.ndarray) -> int:
    up = pose[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    if abs(up[1]) >= abs(up[0]):
        return 0 if up[1] < 0 else 2
    return 1 if up[0] > 0 else 3


def dashed_contours(image: np.ndarray, contours, color, thickness=3, dash=14, gap=9) -> None:
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        points = np.vstack([points, points[:1]])
        carry, drawing = 0.0, True
        for a, b in zip(points[:-1], points[1:]):
            length = float(np.linalg.norm(b - a))
            position = 0.0
            while position < length:
                step = min((dash if drawing else gap) - carry, length - position)
                if drawing:
                    p0 = a + (b - a) * (position / max(length, 1e-9))
                    p1 = a + (b - a) * ((position + step) / max(length, 1e-9))
                    cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color,
                             thickness, cv2.LINE_AA)
                position += step
                carry += step
                if carry >= (dash if drawing else gap) - 1e-9:
                    carry, drawing = 0.0, not drawing


def footprint_mask(frame: dict, points: np.ndarray, shape) -> np.ndarray:
    u, v, in_view, _, _ = project(frame, points)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not in_view.any():
        return mask
    pose = frame["pose"]
    depth = ((points[in_view] - pose[:3, 3]) @ pose[:3, :3])[:, 2]
    radius = np.clip(frame["intrinsics"]["fx"] * 0.03 / np.maximum(depth, 0.3), 3, 40).astype(int)
    for x, y, r in zip(u[in_view], v[in_view], radius):
        cv2.circle(mask, (int(round(x)), int(round(y))), int(r), 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask


def visibility(frame: dict, points: np.ndarray, role: str) -> float:
    """Share of the points the frame shows as occupied ('present') or seen-through ('absent')."""
    _, _, _, surface, see_past = project(frame, points)
    return float((surface if role == "present" else see_past).mean())


def choose_pair(sessions: dict, sets: dict) -> dict:
    """sets: {session: [(points, role), ...]}; returns the best (before, after) frames."""
    scored = {}
    for session in SESSIONS:
        rows = []
        for frame in sessions[session]:
            score = float(np.mean([visibility(frame, pts, role) for pts, role in sets[session]]))
            rows.append((score, frame))
        rows.sort(key=lambda r: (-r[0], r[1]["frame"]))
        scored[session] = rows[:CANDIDATE_FRAMES]
    best, best_score = None, -1.0
    for s1, f1 in scored[BEFORE]:
        for s2, f2 in scored[AFTER]:
            distance = float(np.linalg.norm(f1["pose"][:3, 3] - f2["pose"][:3, 3]))
            forward = [f["pose"][:3, :3] @ np.array([0.0, 0.0, 1.0]) for f in (f1, f2)]
            angle = float(np.degrees(np.arccos(np.clip(forward[0] @ forward[1], -1, 1))))
            similarity = np.exp(-distance / VIEW_DISTANCE_SCALE_M) * np.exp(-angle / VIEW_ANGLE_SCALE_DEG)
            score = s1 * s2 * similarity
            if score > best_score:
                best, best_score = (f1, f2, s1, s2, distance, angle), score
    f1, f2, s1, s2, distance, angle = best
    matched = min(s1, s2) >= MIN_PAIR_VISIBILITY
    if not matched:
        f1, s1 = scored[BEFORE][0][1], scored[BEFORE][0][0]
        f2, s2 = scored[AFTER][0][1], scored[AFTER][0][0]
        distance = float(np.linalg.norm(f1["pose"][:3, 3] - f2["pose"][:3, 3]))
        forward = [f["pose"][:3, :3] @ np.array([0.0, 0.0, 1.0]) for f in (f1, f2)]
        angle = float(np.degrees(np.arccos(np.clip(forward[0] @ forward[1], -1, 1))))
    return {"before": f1, "after": f2, "before_visibility": round(s1, 3), "after_visibility": round(s2, 3),
            "viewpoint_distance_m": round(distance, 2), "viewpoint_angle_deg": round(angle, 1),
            "matched_viewpoint": matched}


def render(frame: dict, layers: list[tuple[np.ndarray, str, str]], path: Path) -> None:
    """layers: (points, colour key, 'fill' | 'dashed'). Saves an upright, cropped JPEG."""
    image = cv2.imread(str(inside_workspace(RECORDINGS / frame["rgb"])))
    height, width = image.shape[:2]
    union = np.zeros((height, width), dtype=np.uint8)
    for points, color, style in layers:
        mask = footprint_mask(frame, points, image.shape)
        union |= mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if style == "fill":
            overlay = image.copy()
            overlay[mask > 0] = BGR[color]
            image = cv2.addWeighted(overlay, 0.32, image, 0.68, 0)
            cv2.drawContours(image, contours, -1, BGR[color], 3, cv2.LINE_AA)
        else:
            dashed_contours(image, contours, BGR[color])
    turns = upright_turns(frame["pose"])
    ys, xs = np.nonzero(union)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        x0, x1, y0, y1 = width * 0.25, width * 0.75, height * 0.25, height * 0.75
    # Crop with context at the final upright aspect ratio (3:4).
    target = PHOTO_SIZE[0] / PHOTO_SIZE[1] if turns % 2 == 0 else PHOTO_SIZE[1] / PHOTO_SIZE[0]
    w = max((x1 - x0) * (1 + 2 * CROP_PAD), MIN_CROP_PX)
    h = max((y1 - y0) * (1 + 2 * CROP_PAD), MIN_CROP_PX)
    if w / h > target:
        h = w / target
    else:
        w = h * target
    w, h = min(w, width), min(h, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - w / 2, 0, width - w))
    top = int(np.clip(cy - h / 2, 0, height - h))
    crop = image[top:top + int(h), left:left + int(w)]
    crop = np.ascontiguousarray(np.rot90(crop, turns))
    crop = cv2.resize(crop, PHOTO_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(inside_workspace(path)), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def side_by_side(before: Path, after: Path, path: Path, title: str) -> None:
    a, b = cv2.imread(str(before)), cv2.imread(str(after))
    gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
    body = np.hstack([a, gap, b])
    header = np.full((54, body.shape[1], 3), 255, np.uint8)
    cv2.putText(header, "BEFORE (scan 1)", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(header, "AFTER (scan 2)", (a.shape[1] + 22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2,
                cv2.LINE_AA)
    footer = np.full((46, body.shape[1], 3), 255, np.uint8)
    cv2.putText(footer, title[:80], (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(inside_workspace(path)), np.vstack([header, body, footer]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def short_title(change: dict, step2_event: dict | None) -> str:
    kind, label = change["type"], change["display_name"]
    if kind == "moved":
        rotation = change["facts"].get("rotation_deg")
        turn = f", turned ~{int(5 * round(abs(rotation) / 5))}°" if rotation is not None else ""
        return f"{label} moved {change['facts']['distance_m']:.2f} m{turn}"
    if kind == "shifted":
        return f"{label} nudged ~{change['facts']['shift_estimate_m']:.2f} m"
    if kind == "surface_changed":
        return f"{label}s drawn closed" if "drawn closed" in change["description"] else f"{label} changed"
    if kind == "removed":
        return f"{label} left its spot"
    if kind == "added":
        return f"{label} appeared"
    if kind == "changed_at_landmark":
        return f"change at the {change['facts']['landmark']}"
    if kind == "depth_only":
        return f"{label}: depth-only difference"
    return "small changes"


def main() -> None:
    started = time.time()
    for folder in (GALLERY, VIEWER_DATA):
        inside_workspace(folder).mkdir(parents=True, exist_ok=True)
    for stale in GALLERY.glob("*.jpg"):
        inside_workspace(stale).unlink()

    report = json.loads(inside_workspace(OUTPUT / "change_detection_report.json").read_text())
    steps = json.loads(inside_workspace(OUTPUT / "object_changes.json").read_text())
    labels = json.loads(inside_workspace(SEMANTICS / "object_labels.json").read_text())
    landmarks = json.loads(inside_workspace(SEMANTICS / "landmarks.json").read_text())
    appearance = json.loads(inside_workspace(SEMANTICS / "appearance.json").read_text())
    described = json.loads(inside_workspace(OUTPUT / "change_descriptions.json").read_text(encoding="utf-8"))
    grid = np.load(str(inside_workspace(OUTPUT / "voxel_change_grid.npz")))
    origin, voxel = grid["origin"], float(grid["voxel_m"])
    shape = tuple(int(d) for d in grid["dims"])
    object_voxels = np.load(str(inside_workspace(OUTPUT / "objects" / "object_voxels.npz")))

    def centres(key: str) -> np.ndarray:
        return origin + (np.column_stack(np.unravel_index(object_voxels[key], shape)) + 0.5) * voxel

    objects = {o["id"]: o for group in steps["objects"].values() for o in group}
    step2_events = {e.get("id", e.get("object")): e for e in steps["events"]}
    label_of_event = {e["event"]: e for e in labels["events"]}
    sessions = {name: load_session(name) for name in SESSIONS}
    print(f"Loaded inputs and {sum(len(v) for v in sessions.values())} frames ({time.time() - started:.0f}s)",
          flush=True)

    changes, viewer_changes = [], []
    for change in described["changes"]:
        cid, kind = change["id"], change["type"]
        members = change["objects"]
        step2 = step2_events.get(cid)
        record = {"id": cid, "type": kind, "title": short_title(change, step2),
                  "label": change["label"], "label_confident": change["label_confident"],
                  "colour": (appearance["events"].get(cid, {}) or {}).get("colour"),
                  "description": change["description"]}
        parts = []
        if kind in ("moved", "shifted"):
            old, new = objects[step2["from"]], objects[step2["to"]]
            record["before"] = {"object": old["id"], "regions": old["member_regions"],
                                "centroid_m": old["centroid_m"], "bbox_min_m": old["bbox_min_m"],
                                "bbox_max_m": old["bbox_max_m"]}
            record["after"] = {"object": new["id"], "regions": new["member_regions"],
                               "centroid_m": new["centroid_m"], "bbox_min_m": new["bbox_min_m"],
                               "bbox_max_m": new["bbox_max_m"]}
            record["motion"] = {key: step2[key] for key in ("translation_m", "horizontal_distance_m", "rotation",
                                                            "footprint_axis_change_deg", "after_axis_from_rotation",
                                                            "before_vs_nearest_wall") if key in step2}
            if kind == "shifted":
                record["motion"].update({key: step2[key] for key in ("shift_estimate_m", "shift_uncertainty_m",
                                                                     "changed_layer_m", "shift_direction_xy",
                                                                     "face_to_face_distance_m", "interpretation")})
            record["places"] = {"before": change["facts"].get("from") or change["facts"].get("place"),
                                "after": change["facts"].get("to") or change["facts"].get("place")}
            record["relations"] = change["facts"].get("landmark_relations")
            record["evidence"] = {"match_score": step2["cues"], "margin_to_next_best": step2["margin_to_next_best"],
                                  "path_between": step2["path_between"]}
            sets = {BEFORE: [(centres(old["id"]), "present"), (centres(new["id"]), "absent")],
                    AFTER: [(centres(new["id"]), "present"), (centres(old["id"]), "absent")]}
            color = "shift" if kind == "shifted" else None
            layers = {"before": [(centres(old["id"]), color or "old", "fill"), (centres(new["id"]), color or "new", "dashed")],
                      "after": [(centres(new["id"]), color or "new", "fill"), (centres(old["id"]), color or "old", "dashed")]}
            parts = [("shift_old" if kind == "shifted" else "old", old["id"]),
                     ("shift_new" if kind == "shifted" else "new", new["id"])]
            record["arrow"] = {"from": old["centroid_m"], "to": new["centroid_m"]}
        elif kind == "small_changes":
            record["objects"] = [{"object": key, "side": objects[key]["side"], "regions": objects[key]["member_regions"],
                                  "centroid_m": objects[key]["centroid_m"], "voxels": objects[key]["voxels"]}
                                 for key in members]
            parts = [("small", key) for key in members]
            sets = None
        else:
            main_object = objects[members[0]]
            side = main_object["side"]
            role_color = ("surface" if kind == "surface_changed" else "depth" if kind == "depth_only"
                          else ("old" if side == "removed" else "new"))
            pts = np.vstack([centres(m) for m in members])
            record["object"] = {"objects": members,
                                "regions": [r for m in members for r in objects[m]["member_regions"]],
                                "present_in": main_object["present_in"], "centroid_m": main_object["centroid_m"],
                                "bbox_min_m": main_object["bbox_min_m"], "bbox_max_m": main_object["bbox_max_m"]}
            record["place"] = change["facts"].get("place")
            record["facts"] = {k: v for k, v in change["facts"].items() if k != "place"}
            present_session = BEFORE if side == "removed" else AFTER
            absent_session = AFTER if side == "removed" else BEFORE
            sets = {present_session: [(pts, "present")], absent_session: [(pts, "absent")]}
            layers = {"before": [(pts, role_color, "fill" if present_session == BEFORE else "dashed")],
                      "after": [(pts, role_color, "fill" if present_session == AFTER else "dashed")]}
            parts = [("surface" if kind == "surface_changed" else ("old" if side == "removed" else "new"), m)
                     for m in members]
            if kind == "changed_at_landmark":
                parts = [("landmark_change", m) for m in members]
            if kind == "depth_only":
                parts = [("depth_only", m) for m in members]

        label_info = label_of_event.get(cid)
        if label_info:
            record["label_evidence"] = {"agreement": label_info["agreement"],
                                        "florence2_votes": label_info["florence_votes"],
                                        "siglip2_top5": label_info["siglip_top5"]}
        if sets is not None:
            pair = choose_pair(sessions, sets)
            before_path, after_path = GALLERY / f"{cid}_before.jpg", GALLERY / f"{cid}_after.jpg"
            render(pair["before"], layers["before"], before_path)
            render(pair["after"], layers["after"], after_path)
            side_by_side(before_path, after_path, GALLERY / f"{cid}_pair.jpg", f"{cid}: {record['title']}")
            record["media"] = {
                "before_photo": out_rel(before_path),
                "after_photo": out_rel(after_path),
                "pair_photo": out_rel(GALLERY / f"{cid}_pair.jpg"),
                "before_frame": pair["before"]["frame"], "after_frame": pair["after"]["frame"],
                "viewpoint_distance_m": pair["viewpoint_distance_m"],
                "viewpoint_angle_deg": pair["viewpoint_angle_deg"],
                "matched_viewpoint": pair["matched_viewpoint"],
                "point_clouds": [objects[m]["ply"] for _, m in parts]}
            print(f"{cid:>5} {record['title']:<42} frames {pair['before']['frame']} / {pair['after']['frame']} "
                  f"(view {pair['viewpoint_distance_m']} m, {pair['viewpoint_angle_deg']} deg, "
                  f"vis {pair['before_visibility']}/{pair['after_visibility']}, "
                  f"{'matched' if pair['matched_viewpoint'] else 'separate views'})", flush=True)
        changes.append(record)

        viewer_parts = []
        for role, key in parts:
            points, colors = read_cloud(OUTPUT / "objects" / f"{key}.ply")
            if len(points) == 0:
                points, colors = centres(key), np.full((len(object_voxels[key]), 3), 0.5)
            points, colors = downsample(points, colors, OBJECT_VOXEL_M)
            viewer_parts.append({"role": role, "object": key, **encode_points(points, colors)})
        viewer_changes.append({**{k: record[k] for k in ("id", "type", "title", "label", "label_confident",
                                                         "colour", "description")},
                               "display_name": change["display_name"],
                               "arrow": record.get("arrow"),
                               "photos": ({"before": "gallery/" + Path(record["media"]["before_photo"]).name,
                                           "after": "gallery/" + Path(record["media"]["after_photo"]).name,
                                           "matched": record["media"]["matched_viewpoint"]}
                                          if "media" in record else None),
                               "facts": viewer_facts(record),
                               "parts": viewer_parts})

    # Room clouds for the viewer.
    low = origin - 0.1
    high = origin + np.array(shape) * voxel + 0.1
    rooms = {}
    for session, name in SCAN_CLOUDS.items():
        points, colors = read_cloud(ALIGNMENT / name)
        inside = np.all((points >= low) & (points <= high), axis=1)
        points, colors = downsample(points[inside], colors[inside], ROOM_VOXEL_M)
        rooms[session] = encode_points(points, colors)
        print(f"{session}: {len(points)} room points for the viewer", flush=True)

    floor_z = steps["floor_z_m"]
    walls = described["walls"]
    stage_hashes = {"step1_change_mask_sha256": report["change_voxels"]["change_mask_sha256"],
                    "step2_result_sha256": steps["result_sha256"],
                    "step3_result_sha256": labels["result_sha256"],
                    "step4a_result_sha256": landmarks["result_sha256"],
                    "step4b_result_sha256": described["result_sha256"],
                    "step4c_result_sha256": appearance["result_sha256"]}
    models = []
    for source in (labels["learned_models_used"], appearance["learned_models_used"]):
        for model in source:
            models.append({k: model[k] for k in ("id", "revision", "license", "use") if k in model})
    output = {
        "schema_version": "1.0.0",
        "title": "Changes between two phone recordings of the same room",
        "recordings": {"before": BEFORE, "after": AFTER},
        "common_frame": f"{BEFORE} PLY frame; metres; Z up; floor at z = {floor_z:.3f} m",
        "conventions": {"left_right": "as seen when facing the window",
                        "distances": "gaps between voxel surfaces (5 cm voxels), about +-5 cm",
                        "rotation": "about the vertical axis, about +-10 deg",
                        "label_confident": "true when Florence-2 and SigLIP 2 name the same object"},
        "summary": described["summary"],
        "counts": {kind: sum(c["type"] == kind for c in changes) for kind in sorted({c["type"] for c in changes})},
        "changes": changes,
        "room": {"floor_z_m": floor_z, "walls": walls,
                 "landmarks": [{k: l[k] for k in ("id", "name", "centroid_m", "bbox_min_m", "bbox_max_m")}
                               for l in landmarks["landmarks"]]},
        "coverage": report["coverage"],
        "models": models,
        "learned_models_trained_or_finetuned": False,
        "manual_labels_used": False,
        "pipeline": stage_hashes,
        "limitations": [
            "Only space observed by both recordings can be compared; changes elsewhere are reported as unknown, "
            f"not as removed (cross-observed fraction of occupied space: {report['coverage']['cross_observed_fraction']}).",
            "Objects smaller than about a 15 cm cube are listed but not interpreted.",
            "'Gone' means the space is empty in the second recording and no matching object was found in space "
            "both recordings observed; the object may have moved somewhere only one recording saw.",
            "Labels marked label_confident=false come from one model only and can be wrong (the door is labelled "
            "'closet wall').",
        ],
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    (inside_workspace(OUTPUT) / "changes.json").write_text(json.dumps(output, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")

    bounds_low = (origin - 0.0).round(3).tolist()
    bounds_high = (origin + np.array(shape) * voxel).round(3).tolist()
    viewer = {"summary": described["summary"], "recordings": {"before": BEFORE, "after": AFTER},
              "floor_z": floor_z, "bounds": [bounds_low, bounds_high],
              "conventions": output["conventions"],
              "landmarks": [{"name": l["name"], "centroid": l["centroid_m"], "top": l["bbox_max_m"][2]}
                            for l in landmarks["landmarks"]],
              "changes": viewer_changes, "pipeline": stage_hashes, "models": models}
    for session, key in ((BEFORE, "before"), (AFTER, "after")):
        (inside_workspace(VIEWER_DATA) / f"room_{key}.js").write_text(
            f"window.ROOM_CLOUDS = window.ROOM_CLOUDS || {{}};\nwindow.ROOM_CLOUDS[{json.dumps(key)}] = "
            f"{json.dumps(rooms[session])};\n", encoding="utf-8")
    (inside_workspace(VIEWER_DATA) / "changes.js").write_text(
        f"window.CHANGE_DATA = {json.dumps(viewer, ensure_ascii=False)};\n", encoding="utf-8")
    shutil.copyfile(VIEWER_TEMPLATE, inside_workspace(VIEWER / "index.html"))
    sizes = {p.name: p.stat().st_size for p in list(VIEWER_DATA.glob("*.js")) + list(GALLERY.glob("*.jpg"))}
    print(f"Viewer data {sum(v for k, v in sizes.items() if k.endswith('.js')) / 1e6:.1f} MB, "
          f"gallery {sum(v for k, v in sizes.items() if k.endswith('.jpg')) / 1e6:.1f} MB")
    print(f"changes.json sha256 {output['result_sha256']}")
    print(f"Done in {time.time() - started:.0f}s")


def viewer_facts(record: dict) -> list[list[str]]:
    facts = []
    motion = record.get("motion", {})
    if "horizontal_distance_m" in motion and record["type"] == "moved":
        facts.append(["Moved", f"{motion['horizontal_distance_m']:.2f} m"])
    rotation = (motion.get("rotation") or {}).get("rotation_deg")
    if rotation is not None:
        facts.append(["Turned", f"{abs(rotation):.0f}° ±10°"])
    if record["type"] == "shifted":
        facts.append(["Pushed", f"~{motion['shift_estimate_m']:.2f} ± {motion['shift_uncertainty_m']:.2f} m"])
    for relation in record.get("relations") or []:
        facts.append([f"To {relation['landmark']}", f"{relation['gap_before_m']:.2f} → {relation['gap_after_m']:.2f} m"])
    evidence = record.get("evidence")
    if evidence:
        facts.append(["Match score", f"{evidence['match_score']['score']:.2f}"])
        path = evidence["path_between"]
        facts.append(["Path between", f"{path['free_in_either'] * 100:.0f}% seen empty"])
    if record.get("label_evidence"):
        votes = record["label_evidence"]["florence2_votes"]
        top = record["label_evidence"]["siglip2_top5"][:2]
        facts.append(["Florence-2", ", ".join(f"{k} ×{v}" for k, v in list(votes.items())[:3]) or "—"])
        facts.append(["SigLIP 2", ", ".join(f"{t['label']} {t['prob']:.2f}" for t in top)])
    if record.get("facts", {}).get("changed_part"):
        facts.append(["Changed part", record["facts"]["changed_part"]])
    check = record.get("facts", {}).get("photo_check")
    if check:
        facts.append(["Other recording", f"{check['label']} ({'same object' if check['same_object_visible'] else 'different'})"])
    return facts


if __name__ == "__main__":
    main()
