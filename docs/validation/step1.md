# Step 1 validation against the known changes

> Written while developing the pipeline on the provided recordings, in order. Later steps corrected some earlier readings; the corrections are marked in place rather than hidden. Paths such as `qa/…`, `objects/…`, `semantics/…` and `viewer/…` are inside `outputs/voxel_diff/` after `navdelta run`.

Ground truth (`room_changes.txt`, written by the person who staged the scene):

1. The tower ventilator was moved near the table.
2. The office chair was moved further away from the table.
3. The travel trolley bag was moved perpendicular to the adjacent wall.

Each region below was checked by projecting its voxels into the real keyframes
of both recordings that observe it best (`qa/region_views_p*.png`, regions of
15 voxels or more). The "present" view is the frame where measured depth agrees
with the region's surface. The "absent" view is the frame where the camera
clearly sees *past* the region's location. Region IDs are from the final
deterministic run, change-mask hash `0c1c5a6c…3267570e`.

## Listed changes: 3 / 3 detected

| Known change | Old position (removed) | New position (added) | Photo evidence |
|---|---|---|---|
| Tower fan moved near the table | R5, 54 vox at (−0.26, −0.50) | A4, 80 vox at (0.43, −0.64) | Scan 1 shows the black tower fan beside the bed and drawers, and scan 2 sees bare floor there. Scan 2 shows the fan beside the desk, and scan 1 sees empty space there. |
| Office chair moved away from the table | R3 (backrest, 111) + R4 (seat, 110) + R11 (17), around (−0.16…0.26, −1.44…−1.83) | A1, 330 vox at (−0.48, −1.69), z −0.96…−0.06, + A7 (base, 17) | Scan 1 shows the chair's seat and backrest next to the desk. Scan 2 shows the chair at A1, and its old spot is empty. It moved about 0.55 m away from the desk (x ≈ 0.6–1.3). |
| Trolley bag turned perpendicular to the wall | R1, 266 vox at (−1.15, −1.45) | A2, 223 vox at (−1.14, −2.01) | Scan 1 shows the navy trolley bag standing parallel to the left wall. Scan 2 shows it standing out from the wall, and its old spot is empty. |

## Surface changes (tall, thin sheets, reported separately)

| Regions | What the photos show |
|---|---|
| SA1 (1088 vox, 1.50 m tall) | The brown sheer window curtains. Only scan 2's depth registers them as a surface; scan 1's depth sees through them to the window. *Corrected twice:* this was first read as the blinds being lowered, then (Step 3, from scan-2 crops only) as the curtains being drawn closed. Step 5's before/after photos from matching viewpoints show the curtains across the window in **both** recordings, so this is a depth-only difference, not a state change (`viewer/gallery/S1_pair.jpg`). |
| A3 (175 vox at (0.38, −1.99)) | The lower part of the same curtains. It falls under the 1.0 m sheet height because the desk occludes the rest, so it stays in the object list. Step 2 should merge it with SA1. |
| SA2–SA5 (11–43 vox) | Small pieces along the window and radiator line (y ≈ −2.1), from the same curtains. |

## Additional regions (not in the list)

The photos show these are mostly real differences between the recordings, not
geometric artifacts. None of them were staged.

| Regions | What the photos show |
|---|---|
| R2, R8 | The door in the far alcove (y ≈ 2.4). *Corrected in Step 5:* the before/after photos from matching viewpoints show the door closed and unchanged in both recordings, so this is a depth-only difference (local misalignment or depth noise at the far end of the room), not a door movement (`viewer/gallery/OR2_pair.jpg`). |
| R6, R9 | *Corrected after Steps 2–3:* both were first read as a white body pillow. R6 is the edge of a white floor lamp that shifted (labelled in Step 3). R9 is the back face of the white drawer unit, which slid about 0.2 m (Step 2's shift test). |
| R7, R10 | Small clutter on a shelf top and in a shelf cubby. |
| R12, R13, A8, A9 | Small items under and beside the desk (a backpack and things near the drawer unit). Each is 10–17 voxels. A9 and R13 are under the 15-voxel threshold for the photo sheets, so they were not checked one by one. |
| A5 (68 vox) | A patch on the desk side next to the fan's new spot. Step 2 will decide whether it belongs to the fan or is a separate item. |
| A6 (27 vox) | An item on top of the drawer unit by the bed. |

## Counts

| Stage | Added | Removed |
|---|---|---|
| Raw (occupied in one scan, observed free in the other) | 2996 | 1111 |
| After masking walls, floor and ceiling | 2234 | 1062 |
| Surface sheets (curtains) | 1197 | 0 |
| Object voxels | 1037 | 1062 |
| In regions of 10 voxels or more | 947 | 1003 |

Of the 56,482 voxels occupied in either scan, 25,881 (45.8%) were also observed
by the other recording. The remaining 30,601 were seen by only one recording, so
they are treated as *unknown*, not reported as changes.

## Lessons carried into Step 2

- **Fragments need grouping.** The chair appears as separate backrest, seat and
  base pieces in both recordings, and the curtains appear as SA1 plus A3 and
  SA2–SA5. Pieces have to be grouped into objects before old and new positions
  are matched.
- **Size alone cannot separate objects from clutter.** The tower fan is thin
  (about 25 cm across). Its regions (54 and 80 voxels) are the same size as
  clutter such as the lamp edge R6 (46). Step 2 therefore matches removed regions to
  added regions by height, footprint and appearance instead of using a size
  threshold.
- **Moves include rotations.** The trolley bag was turned 90°, so matching has
  to compare footprints regardless of orientation.
- **Some changes are unlisted.** The floor lamp and the drawer unit really
  moved, so the final README reports them as detected but not staged rather
  than hiding them. The door and the curtains turned out to be depth-only
  differences (see Steps 3 and 5); the README reports them as such.

## Robustness checks performed

- **No randomness anywhere.** Ray casting, occupancy, wall/floor/ceiling
  detection and region labelling are all deterministic.
  `navdelta verify` runs the full pipeline twice and
  compares the SHA-256 of the packed change masks. The two runs are
  bit-identical.
- **Structural planes come from an exhaustive search, not RANSAC.** The search
  tries every gravity-aligned direction (vertical walls in 1° steps, plus
  horizontal), uses a ±5 cm slab and 1 cm offset bins, and peels planes off
  greedily. An earlier attempt with Open3D RANSAC gave different walls on
  identical runs (A1 jumped from 330 to 1602 voxels). Seeded RANSAC still
  depended on the seed, so it was replaced.
- **Walls are tested column by column.** A 5 cm column counts as wall only if it
  has a continuous vertical run of at least 1.5 m. A whole-plane test had
  treated a row of furniture faces as wall and hidden the trolley bag's old
  position. Floor and ceiling count as structural only near the vertical
  extremes (±0.20 m) and in components of at least 500 voxels.
- **The curtain threshold was measured, not guessed.** Sheet columns must have a
  continuous run of at least 1.0 m. All object regions have runs of 0.90 m or
  less, and the curtains' 10th percentile is 1.11 m, so 1.0 m sits in the gap.
- **The surface tolerance was not tuned against the known answer.** The 10 cm
  value comes from the alignment report's held-out cross-view residuals (99.43%
  within 10 cm, p90 69 mm).
