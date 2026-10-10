# `changes.json`: the common result format

Every approach writes `outputs/<approach>/changes.json` in this format, so
results from different approaches can be compared and shown in the same
viewer. All coordinates are in the shared frame of the "before" recording:
metres, Z up.

## Top level

| Key | Meaning |
|---|---|
| `schema_version` | format version (`1.0.0`) |
| `recordings` | `{"before": name, "after": name}` |
| `common_frame` | description of the coordinate frame, including the floor height |
| `conventions` | units, the left/right convention, precision notes |
| `summary` | one paragraph in plain English |
| `counts` | number of changes per `type` |
| `changes` | list of change records (below) |
| `room` | floor height, walls, static landmarks (approach-dependent) |
| `coverage` | how much of the space both recordings observed |
| `models` | every learned model used: id, revision, licence, use |
| `learned_models_trained_or_finetuned`, `manual_labels_used` | both `false` for `voxel_diff` |
| `pipeline` | result hash of every stage |
| `limitations` | known limitations, in plain English |
| `result_sha256` | hash of this file's content |

## A change record

Common fields:

| Key | Meaning |
|---|---|
| `id` | stable id within a run: `M1` move, `SH1` small push, `OR2` / `OA5` / `S1` single-position changes, `SMALL` |
| `type` | one of the types below |
| `title` | short title, e.g. `suitcase moved 0.54 m, turned ~100°` |
| `label`, `label_confident` | object name; `false` when the two labelling models disagree |
| `colour` | optional colour word |
| `description` | the full sentence(s) |
| `media` | `before_photo`, `after_photo`, `pair_photo` (paths relative to `outputs/<approach>/`), the frames used, and the object point clouds |

Types:

| `type` | Meaning | Geometry fields |
|---|---|---|
| `moved` | object left one place and appears at another | `before`, `after` (centroid, bbox, source regions), `motion` (translation, distance, rotation), `relations`, `arrow` |
| `shifted` | object pushed slightly; only its faces changed | as `moved`, plus `motion.shift_estimate_m` |
| `removed` | space is empty in the second recording, no match elsewhere | `object`, `place` |
| `added` | space was empty in the first recording, no match elsewhere | `object`, `place` |
| `surface_changed` | a thin, tall sheet (curtain, door leaf) changed | `object` |
| `changed_at_landmark` | part of a static piece of furniture changed | `object`, `place` |
| `depth_only` | depth differs, but both recordings' photos show the same object | `object`, `place` |
| `small_changes` | changes smaller than the detection floor, listed only | `objects` |

Approaches may add their own keys (`voxel_diff` adds `evidence` and
`label_evidence`). Readers should ignore keys they don't know.

## Viewer data

`outputs/<approach>/viewer/` contains `index.html` (shared template), `gallery/`
(photos) and `data/`:

- `room_before.js`, `room_after.js` set `window.ROOM_CLOUDS.before / .after`.
- `changes.js` sets `window.CHANGE_DATA`: the summary, floor height, bounds,
  landmarks, and per change its `parts` (point sets with a `role`: `old`, `new`,
  `shift_old`, `shift_new`, `surface`, `landmark_change`, `depth_only`,
  `small`), an optional `arrow`, `photos` and `facts`.

Point sets store uint16 positions quantised to their bounding box, plus
optional uint8 colours, base64-encoded (`origin`, `scale`, `count`,
`positions`, `colors`). An approach that writes these files gets the 3D viewer
for free.
