"""Step 4b: a natural-language description of every change, grounded in measurements.

Every sentence is filled from measured quantities -- no language model writes
them, so each word can be traced back to a number:

  what   the Step 3 label; an uncertain label says so and names what the other
         model said
  where  support (floor / furniture top below), nearest static landmark
         (Step 4a) and the room walls; left/right are as seen when facing the
         window (checked against a photo: the desk is on the left)
  how    Step 2 geometry: distance moved, rotation, orientation against the
         wall, the move-vs-shift path test, and the change in distance to the
         piece of furniture the object moved towards or away from

Two consolidations happen here, both on measured evidence:
  - an added/removed object touching a surface change with the same confident
    label is part of that surface (a fold of the curtain joins the curtain);
  - an added/removed object touching a static landmark with the same label is
    a change *at* that landmark, not a new or vanished piece of furniture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from navdelta.approaches.voxel_diff.paths import AFTER, OUTPUT, SEMANTICS, inside_workspace

NEXT_TO_M = 0.25                 # gap to a landmark for "next to"
NEAR_M = 0.60                    # gap for "near"
RELATION_CHANGE_M = 0.25         # a landmark distance change worth mentioning
RELATION_RANGE_M = 1.0           # only landmarks this close to the old or new spot
ON_FLOOR_M = 0.20                # bottom of the object within this of the floor
ELEVATED_M = 0.60                # bottom above this and nothing below: give its height
SUPPORT_BELOW_M = 0.25           # furniture top this far below the object's bottom
SUPPORT_MIN_OVERLAP = 0.30       # share of the object's floor-plan columns over that top
TOUCH_M = 0.10                   # = Step 1 surface tolerance
AGAINST_WALL_M = 0.50
DIRECTION_MAX_DEG = 30.0         # a landmark counts as "in the direction of" a shift within this
ROTATION_ROUND_DEG = 5
ITEM_COLUMN_SHARE = 0.5          # landmark is an item on furniture: half its floor plan inside it ...
ITEM_RAISED_M = 0.30             # ... and its bottom at least this far above the furniture's bottom


def singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def compatible(a: str, b: str) -> bool:
    wa, wb = [singular(w) for w in a.split()], [singular(w) for w in b.split()]
    return wa == wb or set(wa) <= set(wb) or set(wb) <= set(wa) or wa[-1] == wb[-1]


def article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


class Room:
    """Voxel geometry, walls named relative to the window, and landmarks."""

    def __init__(self, grid, steps: dict, landmarks: dict, landmark_voxels):
        self.origin, self.voxel = grid["origin"], float(grid["voxel_m"])
        self.shape = tuple(int(d) for d in grid["dims"])
        self.floor_z = steps["floor_z_m"]
        occupied = np.flatnonzero((grid["state_scan1"] == 1) | (grid["state_scan2"] == 1))
        self.centre = self.centers(occupied).mean(axis=0)
        self.landmarks = []
        for record in landmarks["landmarks"]:
            points = self.centers(landmark_voxels[record["id"]])
            self.landmarks.append({**record, "points": points, "tree": cKDTree(points),
                                   "tops": self._column_tops(points)})
        # A landmark standing on a larger one (laptop on the desk, fan on the drawers)
        # is an item, not furniture, and a poor reference for "moved towards". The
        # furniture's top is hidden exactly where items stand on it, so the test is
        # containment: most of the item's floor plan lies inside the larger
        # landmark's floor-plan outline (convex hull) and the item starts clearly higher.
        outlines = {l["id"]: Delaunay(np.unique(l["points"][:, :2].round(4), axis=0)) for l in self.landmarks}
        for landmark in self.landmarks:
            columns = np.unique(landmark["points"][:, :2].round(4), axis=0)
            bottom = landmark["points"][:, 2].min()
            hosts = [other for other in self.landmarks
                     if other["voxels"] > landmark["voxels"]
                     and (outlines[other["id"]].find_simplex(columns) >= 0).mean() >= ITEM_COLUMN_SHARE
                     and bottom - other["points"][:, 2].min() >= ITEM_RAISED_M
                     and self.bottom_above_floor(landmark["points"]) > ON_FLOOR_M]
            landmark["rests_on"] = max(hosts, key=lambda h: h["voxels"])["name"] if hosts else None
        self.walls = self._name_walls(steps["walls"])

    def centers(self, flat: np.ndarray) -> np.ndarray:
        return self.origin + (np.column_stack(np.unravel_index(flat, self.shape)) + 0.5) * self.voxel

    def bottom_above_floor(self, points: np.ndarray) -> float:
        return float(points[:, 2].min() - self.voxel / 2 - self.floor_z)

    def _columns(self, points: np.ndarray) -> set:
        return {tuple(c) for c in np.round(points[:, :2] / self.voxel).astype(int)}

    def _column_tops(self, points: np.ndarray) -> dict:
        tops: dict = {}
        for key, z in zip(map(tuple, np.round(points[:, :2] / self.voxel).astype(int)), points[:, 2]):
            tops[key] = max(tops.get(key, -np.inf), z)
        return tops

    def _name_walls(self, walls: list[dict]) -> list[dict]:
        named = []
        for wall in walls:
            normal = np.array(wall["normal"], dtype=np.float64)
            inward = normal if self.centre @ normal > wall["offset_m"] else -normal
            named.append({**wall, "normal": normal, "inward": inward})
        windows = [l for l in self.landmarks if compatible(l["label"], "window")]
        self.facing = None
        reference = None
        if windows:
            window = max(windows, key=lambda l: l["voxels"])
            reference = min(named, key=lambda w: abs(window["points"].mean(axis=0) @ w["normal"] - w["offset_m"]))
            self.facing = -reference["inward"]
        for wall in named:
            if reference is None:
                axis = "x" if abs(wall["normal"][0]) > abs(wall["normal"][1]) else "y"
                wall["label"] = f"wall at {axis} = {wall['offset_m']:+.2f} m"
            elif wall is reference:
                wall["label"] = "window wall"
            elif abs(wall["normal"] @ reference["normal"]) > 0.9:
                wall["label"] = "wall opposite the window"
            else:
                right = np.cross(self.facing, [0.0, 0.0, 1.0])
                wall["label"] = "right-hand wall" if (-wall["inward"]) @ right > 0 else "left-hand wall"
        return named

    def wall_distance(self, point: np.ndarray, wall: dict) -> float:
        return float(abs(point @ wall["normal"] - wall["offset_m"]))

    def gaps(self, points: np.ndarray) -> list[tuple[dict, float]]:
        out = []
        for landmark in self.landmarks:
            distance = float(landmark["tree"].query(points)[0].min())
            out.append((landmark, max(0.0, distance - self.voxel)))
        return sorted(out, key=lambda item: item[1])

    def support(self, points: np.ndarray, bottom_above_floor: float, exclude: str | None = None) -> str | None:
        """'floor', the name of the furniture whose top is just below the object, or None."""
        if bottom_above_floor <= ON_FLOOR_M:
            return "floor"
        bottom = points[:, 2].min()
        columns = {tuple(c) for c in np.round(points[:, :2] / self.voxel).astype(int)}
        best = None
        for landmark in self.landmarks:
            if landmark["id"] == exclude:
                continue
            under = [c for c in columns if c in landmark["tops"]
                     and bottom - SUPPORT_BELOW_M - self.voxel <= landmark["tops"][c] < bottom]
            share = len(under) / len(columns)
            if share >= SUPPORT_MIN_OVERLAP and (best is None or share > best[1]):
                best = (landmark, share)
        return best[0]["name"] if best else None

    def place(self, points: np.ndarray, bottom_above_floor: float, thin_panel: bool = False,
              exclude: str | None = None) -> dict:
        support = None if thin_panel else self.support(points, bottom_above_floor, exclude=exclude)
        centre = points.mean(axis=0)
        candidates = [(l, g) for l, g in self.gaps(points) if l["name"] != support and l["id"] != exclude]
        near = [(l, g) for l, g in candidates if g <= NEAR_M]
        furniture = [(l, g) for l, g in near if l["rests_on"] is None]
        wall = min(self.walls, key=lambda w: self.wall_distance(centre, w))
        parts = []
        if support == "floor":
            parts.append("on the floor")
        elif support:
            parts.append(f"on the {support}")
        elif bottom_above_floor >= ELEVATED_M:
            parts.append(f"about {bottom_above_floor:.1f} m above the floor")
        chosen = (furniture or near)[:1]
        if chosen:
            landmark, gap = chosen[0]
            parts.append(("next to" if gap <= NEXT_TO_M else "near") + f" the {landmark['name']}")
        elif self.wall_distance(centre, wall) <= AGAINST_WALL_M:
            parts.append(f"by the {wall['label']}")
        if not parts:
            parts.append("in the middle of the room")
        return {"text": " ".join(parts), "support": support,
                "nearest_landmarks": [{"landmark": l["name"], "gap_m": round(g, 2)} for l, g in candidates[:3]],
                "nearest_wall": {"wall": wall["label"], "distance_m": round(self.wall_distance(centre, wall), 2)}}

    def direction(self, origin_points: np.ndarray, direction: np.ndarray) -> str:
        """Name a horizontal direction by the landmark ahead, else by the room frame."""
        start = origin_points.mean(axis=0)
        best = None
        for landmark in self.landmarks:
            if landmark["rests_on"] is not None:
                continue
            offset = landmark["points"][:, :2] - start[:2]
            distance = np.linalg.norm(offset, axis=1)
            nearest = int(distance.argmin())
            if distance[nearest] < 1e-6 or distance[nearest] > 1.5:
                continue
            angle = np.degrees(np.arccos(np.clip(offset[nearest] @ direction[:2] / distance[nearest], -1, 1)))
            if angle <= DIRECTION_MAX_DEG and (best is None or distance[nearest] < best[1]):
                best = (landmark, distance[nearest])
        if best:
            return f"towards the {best[0]['name']}"
        if self.facing is None:
            return f"along ({direction[0]:+.2f}, {direction[1]:+.2f})"
        right = np.cross(self.facing, [0.0, 0.0, 1.0])
        forward, side = float(direction @ self.facing), float(direction @ right)
        if abs(forward) >= abs(side):
            return "towards the window" if forward > 0 else "away from the window"
        return "towards the right-hand wall" if side > 0 else "towards the left-hand wall"


def label_phrase(info: dict | None, fallback: str = "object") -> tuple[str, str, str]:
    """(noun, uncertainty note, short display name) for a Step 3 label record."""
    if info is None:
        return fallback, "", fallback
    noun = info["label"]
    if info["confident"]:
        return noun, "", noun
    alternatives = [label for label in info.get("florence_votes", {}) if not compatible(label, noun)]
    if alternatives:
        return noun, f" (uncertain: the other model said “{alternatives[0]}”)", f"{noun}?"
    return noun, " (uncertain label)", f"{noun}?"


def main() -> None:
    grid = np.load(str(inside_workspace(OUTPUT / "voxel_change_grid.npz")))
    steps = json.loads(inside_workspace(OUTPUT / "object_changes.json").read_text())
    labels = json.loads(inside_workspace(SEMANTICS / "object_labels.json").read_text())
    landmarks = json.loads(inside_workspace(SEMANTICS / "landmarks.json").read_text())
    landmark_voxels = np.load(str(inside_workspace(SEMANTICS / "landmark_voxels.npz")))
    object_voxels = np.load(str(inside_workspace(OUTPUT / "objects" / "object_voxels.npz")))
    room = Room(grid, steps, landmarks, landmark_voxels)
    appearance_path = SEMANTICS / "appearance.json"
    appearance = (json.loads(inside_workspace(appearance_path).read_text()) if appearance_path.exists() else None)

    def colour(key: str, level: str) -> str:
        """Optional colour adjective from Step 4c (with a trailing space), or ''."""
        if appearance is None:
            return ""
        value = appearance[level].get(key, {}).get("colour")
        return f"{value} " if value else ""

    objects = {o["id"]: o for group in steps["objects"].values() for o in group}
    label_of_event = {e["event"]: e for e in labels["events"]}
    label_of_object = {o["object"]: o for o in labels["objects"]}
    points_of = {key: room.centers(object_voxels[key]) for key in object_voxels.files}

    def gap_between(a: np.ndarray, b: np.ndarray) -> float:
        return max(0.0, float(cKDTree(b).query(a)[0].min()) - room.voxel)

    # Consolidation 1: object pieces of a surface change with the same confident label.
    merged_into: dict[str, str] = {}
    for event in steps["events"]:
        if event["event"] != "surface_changed":
            continue
        surface_label = label_of_object.get(event["object"])
        for other in steps["events"]:
            if other["event"] not in ("added", "removed"):
                continue
            other_label = label_of_object.get(other["object"])
            if (surface_label and other_label and surface_label["confident"] and other_label["confident"]
                    and compatible(surface_label["label"], other_label["label"])
                    and objects[other["object"]]["side"] == objects[event["object"]]["side"]
                    and gap_between(points_of[other["object"]], points_of[event["object"]]) <= TOUCH_M):
                merged_into[other["object"]] = event["object"]

    descriptions = []

    def record(event_id: str, kind: str, info: dict | None, display: str, text: str, facts: dict,
               members: list[str]) -> None:
        descriptions.append({"id": event_id, "type": kind,
                             "label": info["label"] if info else None,
                             "label_confident": bool(info and info["confident"]),
                             "display_name": display, "description": text, "facts": facts, "objects": members})

    for event in steps["events"]:
        kind = event["event"]
        if kind in ("moved", "shifted"):
            old, new = objects[event["from"]], objects[event["to"]]
            info = label_of_event.get(event["id"])
            noun, note, display = label_phrase(info)
            if kind == "shifted":
                landmark = next((l for l, g in room.gaps(points_of[old["id"]])
                                 if g <= TOUCH_M and compatible(l["label"], noun)), None)
                base = landmark["name"] if landmark else noun
                name = colour(event["id"], "events") + base
                direction = np.array([*event["shift_direction_xy"], 0.0])
                towards = room.direction(points_of[old["id"]], direction)
                where = room.place(np.vstack([points_of[old["id"]], points_of[new["id"]]]),
                                   min(old["bottom_above_floor_m"], new["bottom_above_floor_m"]),
                                   exclude=landmark["id"] if landmark else None)
                text = (f"The {name}{note} was pushed {towards} by about {event['shift_estimate_m']:.2f} m "
                        f"(\u00b1{event['shift_uncertainty_m']:.2f} m). Only its back and front faces changed "
                        f"({event['face_to_face_distance_m']:.2f} m apart); the rest of it stayed in place, so it "
                        f"was nudged, not carried somewhere else. It stands {where['text']}.")
                record(event["id"], "shifted", info, (base if info and info["confident"] else f"{base}?"), text,
                       {"shift_estimate_m": event["shift_estimate_m"],
                        "shift_uncertainty_m": event["shift_uncertainty_m"], "direction": towards,
                        "face_to_face_distance_m": event["face_to_face_distance_m"],
                        "path_between": event["path_between"], "place": where},
                       [old["id"], new["id"]])
                continue
            p_old = room.place(points_of[old["id"]], old["bottom_above_floor_m"])
            p_new = room.place(points_of[new["id"]], new["bottom_above_floor_m"])
            before = {l["id"]: g for l, g in room.gaps(points_of[old["id"]])}
            after = {l["id"]: g for l, g in room.gaps(points_of[new["id"]])}
            changes = [(l, before[l["id"]], after[l["id"]]) for l in room.landmarks
                       if min(before[l["id"]], after[l["id"]]) <= RELATION_RANGE_M
                       and abs(after[l["id"]] - before[l["id"]]) >= RELATION_CHANGE_M]
            furniture = [c for c in changes if c[0]["rests_on"] is None] or changes
            # The strongest "further away from" and "closer to" furniture, each if present.
            picks = [max((c for c in furniture if c[2] > c[1]), key=lambda c: c[2] - c[1], default=None),
                     max((c for c in furniture if c[2] < c[1]), key=lambda c: c[1] - c[2], default=None)]
            relation, relation_fact = "", []
            for pick in sorted((p for p in picks if p), key=lambda c: -abs(c[2] - c[1])):
                landmark, b, a = pick
                verb = "closer to" if a < b else "further away from"
                relation += (f" It is now {verb} the {landmark['name']}: {a:.2f} m from it, compared with "
                             f"{b:.2f} m before.")
                relation_fact.append({"landmark": landmark["name"], "gap_before_m": round(b, 2),
                                      "gap_after_m": round(a, 2), "relation": verb})
            rotation = event["rotation"]["rotation_deg"]
            turn, orientation = "", ""
            if rotation is not None:
                rounded = int(ROTATION_ROUND_DEG * round(abs(rotation) / ROTATION_ROUND_DEG))
                turn = f" and turned by roughly {rounded}° (±10°)"
                implied = event.get("after_axis_from_rotation")
                before_wall = event["before_vs_nearest_wall"]
                if implied and before_wall.get("relation") in ("parallel", "perpendicular"):
                    wall = min(room.walls, key=lambda w: abs(w["offset_m"] - float(before_wall["wall"].split("=")[1]
                                                                                    .replace("m", ""))))
                    orientation = (f" Its long side used to be {before_wall['relation']} to the {wall['label']} "
                                   f"({before_wall['long_axis_vs_wall_deg']:.0f}° off); now it is "
                                   f"{implied['relation']} to it ({implied['long_axis_vs_wall_deg']:.0f}°).")
            text = (f"The {colour(event['id'], 'events')}{noun}{note} was moved {event['horizontal_distance_m']:.2f} m{turn}. It was "
                    f"{p_old['text']}; it is now {p_new['text']}.{relation}{orientation}")
            record(event["id"], "moved", info, display, text,
                   {"distance_m": event["horizontal_distance_m"], "rotation_deg": rotation,
                    "rotation_note": event["rotation"]["note"], "from": p_old, "to": p_new,
                    "landmark_relations": relation_fact, "orientation": orientation.strip() or None,
                    "match_score": event["cues"]["score"], "path_between": event["path_between"]},
                   [old["id"], new["id"]])
        elif kind in ("removed", "added"):
            if event["object"] in merged_into:
                continue
            obj = objects[event["object"]]
            info = label_of_object.get(event["object"])
            noun, note, display = label_phrase(info)
            pts = points_of[obj["id"]]
            thin = obj["shape_class"] == "thin panel"
            at_landmark = next((l for l, g in room.gaps(pts) if g <= TOUCH_M and info and info["confident"]
                                and compatible(l["label"], noun)), None)
            place = room.place(pts, obj["bottom_above_floor_m"], thin_panel=thin,
                               exclude=at_landmark["id"] if at_landmark else None)
            size = (f"{obj['footprint']['length_m']:.2f} × {obj['footprint']['width_m']:.2f} m, "
                    f"{obj['top_above_floor_m'] - obj['bottom_above_floor_m']:.2f} m tall")
            check = (info or {}).get("photo_check") or {}
            other = "second" if kind == "removed" else "first"
            if at_landmark:
                text = (f"Part of the {at_landmark['name']} {'now fills' if kind == 'added' else 'no longer fills'} "
                        f"space that was {'empty' if kind == 'added' else 'occupied'} in the first recording "
                        f"(changed part: {size}). Either the {at_landmark['name']} was nudged or something at its "
                        f"edge changed; the labels only say it is the {at_landmark['name']}.")
                if check.get("same_object_visible"):
                    text += (f" The {other} recording's photos show the {at_landmark['name']} at this spot too, "
                             f"which fits a small nudge.")
                record(event["object"], "changed_at_landmark", info, at_landmark["name"], text,
                       {"landmark": at_landmark["name"], "side": kind, "changed_part": size, "place": place},
                       [obj["id"]])
                continue
            adjective = colour(obj["id"], "objects")
            subject = (f"A thin {adjective}vertical panel labelled “{noun}”{note}" if thin
                       else f"{capital(article(adjective + noun))} {adjective}{noun}{note}")
            facts = {"changed_part": size, "place": place, "photo_check": check or None}
            if check.get("same_object_visible"):
                text = (f"{subject} {place['text']}: the depth surfaces differ by more than 10 cm here (changed "
                        f"part: {size}), but the {other} recording's photos still show "
                        f"{article(check['label'])} {check['label']} at this spot, so nothing was "
                        f"{'removed' if kind == 'removed' else 'added'}. Most likely the {noun} sits slightly "
                        f"differently, a thin or see-through surface read differently in depth, or the two "
                        f"recordings are locally misaligned here.")
                record(event["object"], "depth_only", info, display, text, facts, [obj["id"]])
                continue
            if kind == "removed":
                text = (f"{subject} that stood {place['text']} no longer fills that space (changed part: {size}). "
                        f"No matching object appeared anywhere both recordings observed, so it was taken away or "
                        f"moved somewhere only one recording saw.")
            else:
                text = (f"{subject} now fills space {place['text']} that was empty in the first recording "
                        f"(changed part: {size}); no matching object disappeared elsewhere.")
            record(event["object"], kind, info, display, text, facts, [obj["id"]])
        elif kind == "surface_changed":
            obj = objects[event["object"]]
            info = label_of_object.get(event["object"])
            noun, note, display = label_phrase(info, "surface")
            parts = [event["object"]] + [k for k, v in merged_into.items() if v == event["object"]]
            pts = np.vstack([points_of[p] for p in parts])
            width = float(obj["footprint"]["length_m"])
            height = float(pts[:, 2].max() - pts[:, 2].min() + room.voxel)
            windows = [(l, g) for l, g in room.gaps(pts) if compatible(l["label"], "window") and g <= NEAR_M]
            location = f"across the {windows[0][0]['name']}" if windows else room.place(pts, 0.0)["text"]
            present = obj["present_in"]
            recording = "second" if present == AFTER else "first"
            other = "first" if present == AFTER else "second"
            check = (info or {}).get("photo_check") or {}
            adjective = colour(event["object"], "objects")
            if check.get("same_object_visible"):
                text = (f"The {adjective}{singular(noun)}{note} {location} appears in the photos of both "
                        f"recordings (the {other} recording's photos are labelled \u201c{check['label']}\u201d), "
                        f"but only the {recording} recording's depth registers it as a solid surface: a "
                        f"{width:.1f} m wide, {height:.1f} m tall sheet. So it was not opened or closed; most likely "
                        f"the depth sensor saw through the thin material in the {other} recording, or it hangs "
                        f"slightly differently.")
                if len(parts) > 1:
                    text += (f" A fold that hangs further into the room ({', '.join(parts[1:])}) is included, "
                             f"merged because it has the same label and touches it.")
                record(event["object"], "depth_only", info, display, text,
                       {"width_m": round(width, 2), "height_m": round(height, 2), "present_in": present,
                        "merged_objects": parts[1:], "photo_check": check}, parts)
                continue
            if compatible(noun, "curtain") or compatible(noun, "blind"):
                state = "drawn closed" if present == AFTER else "opened"
                text = (f"The {adjective}{singular(noun)}s{note} were {state} {location}: a {width:.1f} m wide, "
                        f"{height:.1f} m tall sheet that exists only in the {recording} recording.")
            else:
                text = (f"A {width:.1f} m wide, {height:.1f} m tall {noun}{note} surface {location} exists only in "
                        f"the {recording} recording.")
            if len(parts) > 1:
                text += (f" A fold of it that hangs further into the room ({', '.join(parts[1:])}) is included, "
                         f"merged because it has the same label and touches it.")
            record(event["object"], "surface_changed", info, display, text,
                   {"width_m": round(width, 2), "height_m": round(height, 2), "present_in": present,
                    "merged_objects": parts[1:]}, parts)

    small = [e for e in steps["events"] if e["event"] == "small_change"]
    if small:
        record("SMALL", "small_changes", None, "small changes",
               f"{plural(len(small), 'further change is', 'further changes are')} smaller than a 15 cm cube; "
               f"listed, but not interpreted.",
               {"objects": [e["object"] for e in small]}, [e["object"] for e in small])

    def group(kind):
        return [d for d in descriptions if d["type"] == kind]

    def names(items):
        return ", ".join(d["display_name"] for d in items)

    clauses = []
    for kind, one, many in (("moved", "object was moved", "objects were moved"),
                            ("shifted", "piece of furniture was nudged", "pieces of furniture were nudged"),
                            ("surface_changed", "surface changed", "surfaces changed"),
                            ("removed", "item no longer fills its spot", "items no longer fill their spots"),
                            ("added", "item appeared", "items appeared"),
                            ("changed_at_landmark", "piece of furniture changed at its edge",
                             "pieces of furniture changed at their edge"),
                            ("depth_only", "difference is depth-only", "differences are depth-only")):
        items = group(kind)
        if items:
            clauses.append(f"{plural(len(items), one, many)} ({names(items)})")
    summary = ("Between the two recordings, " + ", ".join(clauses[:-1])
               + (f" and {clauses[-1]}" if len(clauses) > 1 else clauses[0]) + "."
               + (" Depth-only means the object appears at that spot in both recordings' photos; only the "
                  "measured depth differs." if group("depth_only") else "")
               + " A question mark marks a label the two models did not agree on.")

    output = {
        "schema_version": "1.0.0",
        "stage": "step4b_natural_language_descriptions",
        "method": "templates filled from measured geometry (Step 2), open-vocabulary labels (Step 3) and "
                  "static landmarks (Step 4a); no language model writes the sentences. Colour adjectives, when "
                  "present, come from the optional Step 4c VLM question (majority of views).",
        "colour_source": (appearance["learned_models_used"][0]["id"] if appearance else None),
        "conventions": {"left_right": "as seen when facing the window",
                        "distances": "gap between the nearest voxel surfaces, metres"},
        "parameters": {"next_to_m": NEXT_TO_M, "near_m": NEAR_M, "relation_change_m": RELATION_CHANGE_M,
                       "relation_range_m": RELATION_RANGE_M, "on_floor_m": ON_FLOOR_M, "elevated_m": ELEVATED_M,
                       "support_below_m": SUPPORT_BELOW_M, "support_min_overlap": SUPPORT_MIN_OVERLAP,
                       "touch_m": TOUCH_M, "against_wall_m": AGAINST_WALL_M,
                       "direction_max_deg": DIRECTION_MAX_DEG, "item_column_share": ITEM_COLUMN_SHARE,
                       "item_raised_m": ITEM_RAISED_M},
        "walls": [{"label": w["label"], "normal": w["normal"].round(3).tolist(), "offset_m": w["offset_m"]}
                  for w in room.walls],
        "landmarks": [{"id": l["id"], "name": l["name"], "rests_on": l["rests_on"]} for l in room.landmarks],
        "merged": merged_into,
        "summary": summary,
        "changes": descriptions,
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    (inside_workspace(OUTPUT) / "change_descriptions.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# What changed between the two recordings", "", summary, "",
             "Left and right are as seen when facing the window. Distances are gaps between surfaces.", ""]
    for item in descriptions:
        lines.append(f"- **{item['id']}** ({item['type'].replace('_', ' ')}): {item['description']}")
    lines += ["", "Generated by `navdelta/approaches/voxel_diff/describe.py` from `object_changes.json` (Step 2), "
                  "`semantics/object_labels.json` (Step 3), `semantics/landmarks.json` (Step 4a) and, for colour "
                  "words only, `semantics/appearance.json` (Step 4c), all in `outputs/voxel_diff/`."]
    (inside_workspace(OUTPUT) / "CHANGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Walls:", [(w["label"], w["offset_m"]) for w in room.walls])
    print("Items resting on furniture:", [(l["name"], l["rests_on"]) for l in room.landmarks if l["rests_on"]])
    print(summary, "\n")
    for item in descriptions:
        print(f"{item['id']:>5} [{item['type']}] {item['description']}\n")
    print("Result sha256:", output["result_sha256"])


if __name__ == "__main__":
    main()
