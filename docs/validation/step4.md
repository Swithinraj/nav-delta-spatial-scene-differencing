# Step 4 validation: natural-language descriptions

> Written while developing the pipeline on the provided recordings, in order. Later steps corrected some earlier readings; the corrections are marked in place rather than hidden. Paths such as `qa/…`, `objects/…`, `semantics/…` and `viewer/…` are inside `outputs/voxel_diff/` after `navdelta run`.

Step 4 turns the measured changes into sentences in
`outputs/voxel_diff/CHANGE_REPORT.md`, with structured data in
`change_descriptions.json`. It runs as part of `navdelta run` and has
three parts:

| Part | Script | What it does | Model (open, not trained here) |
|---|---|---|---|
| 4a | `navdelta/approaches/voxel_diff/landmarks.py` | finds static furniture (desk, bed, …) in 3D, to use as spatial references | Florence-2-large `<OD>` (MIT, `4271c66b`) + SAM 2.1 base-plus (Apache-2.0, `b7320756`) |
| 4b | `navdelta/approaches/voxel_diff/describe.py` | fills sentence templates from measured numbers only | none |
| 4c | `navdelta/approaches/voxel_diff/colours.py` | optional: one colour word per object | SmolVLM2-500M-Video-Instruct (Apache-2.0, `7b375e1b`) |

No language model writes the sentences. Every distance, angle and relation in
them comes from Step 2 geometry, Step 3 labels or the Step 4a landmarks. The
VLM contributes only a colour adjective.

All four stage hashes were identical over repeated full runs of Steps 3–4:
labels `f6dd5b19…`, landmarks `61036162…`, colours `c732a5c2…`, descriptions
`6e6d900c…`. One full run takes about 6½ minutes on the RTX 3050 (4 GB).

## 4a: Landmarks

Florence-2 `<OD>` used its own open label space on 143 upright keyframes from
both recordings. Each detection was masked with SAM 2.1, and its depth pixels
were voted into the shared voxel grid.

**Kept (17 landmarks):** each needed at least 3 agreeing frames and was
present in both scans:
- bed, desk and chest of drawers;
- the window and the door;
- items on the desk: two monitors, a laptop, a keyboard and a television;
- shelves, a sink, a mirror, the shoes by the entrance, a small fan on the
  drawers, and a chair at the desk's end.

The map is in `qa/step4_landmarks.png`. Items standing on larger furniture
(laptop, monitors and the small fan) are marked as items, so they are not used
as "moved towards" references.

**Rejected by the static test:** these detections are exactly the things that
changed.

| Detection | Present in both scans | Why it was dropped |
|---|---|---|
| curtain | 24% | its depth surface differs between the recordings (depth-only, see Step 3) |
| door (far alcove) | 41% | its depth surface differs between the recordings (depth-only, see Step 3) |
| suitcase | 0% | it moved |
| backpack | 37% | it changed position |
| carpet | 100% | it is static, but it lies on the floor plane (structural, not a landmark) |

**Left and right:** these are as seen when facing the window. I checked this
on frame `217154594659`, which faces the window. The desk's centre projects to
the left of the upright image, which agrees with the computed "left-hand wall"
on the desk side. The poses are right-handed (det R = +1 for every frame).

## 4b: The staged changes, as generated

| `room_changes.txt` | Generated sentence (abridged) | Agrees? |
|---|---|---|
| The tower ventilator was moved near the table. | "The black fan … was moved 0.71 m. It was on the floor near the chest of drawers; it is now on the floor next to the desk. It is now closer to the desk: 0.02 m from it, compared with 0.74 m before." | ✔ The label is flagged uncertain (Florence-2 said "remote control"). |
| The office chair was moved further away from the table. | "The black office chair was moved 0.57 m. It was next to the desk; it is now near the window. It is now further away from the desk: 0.84 m from it, compared with 0.20 m before." | ✔ |
| The travel trolley bag was moved perpendicular to the adjacent wall. | "The blue suitcase was moved 0.54 m and turned by roughly 100° (±10°) … Its long side used to be parallel to the right-hand wall (10° off); now it is perpendicular to it (71°)." | ✔ (71° is just inside the 70° "perpendicular" band) |

The other events are described in `CHANGE_REPORT.md`:
- the chest of drawers pushed towards the bed by about 0.21 m;
- the curtains and the far door: depth-only differences, because the photos of
  both recordings show the same object (the curtain fold OA3 is merged in by
  label and contact);
- a lamp edge, a shelf item and a plastic bag that no longer fill their spots;
- a change at the desk's edge;
- four tiny changes, not interpreted.

## 4c: Why only colour words from the VLM

Both SmolVLM2 sizes were tested on six crops, asked for colour and type:

| Object | SmolVLM2-2.2B | SmolVLM2-500M |
|---|---|---|
| tower fan | "Black, tall, cylindrical" | "black, portable" |
| suitcase | "Black suitcase" | "dark blue, suitcase" |
| office chair | "Black, swivel" | "Black, office chair" |
| floor lamp | "White, paper lantern" | "white, spherical" |
| door | "White, door" | "white, metal" |
| curtain | "Brown sheer curtain" | "brown" |

**Which model:** the 2.2B model needed 4.5 GB and only ran by spilling into
shared system memory, so it does not fit the 4 GB GPU. The 500M model uses
1.1 GB.

**What it is asked:** its colour words were right on every crop (it even saw
the suitcase as dark blue), but its shape words were not (a "spherical" lamp,
a "metal" door). So only one question is asked: "What is its main colour?"

**How the answer is used:**
- The question is asked on up to 3 crops per object, with no mask outline, so
  the red or green outline can't bias the answer.
- An answer counts only if it is a basic colour word.
- A colour is used only when it wins a strict majority of the views.

**Results:** black fan, blue suitcase, black chair, white chest of drawers,
brown curtains, and white door, lamp, plastic bag and desk edge. All of these
are consistent with the photos.

## Rules changed after seeing the output

These changes were made while reading this scene's sentences. Each one was
triggered by a visibly wrong phrase:

- **Support.** The door was "on the footwear". The furniture below an object
  must now have its top just under the object's bottom, across at least 30% of
  the object's floor plan. Thin panels get no support phrase.
- **Height.** The chair was "about 0.4 m above the floor" because its legs
  were not captured. A height is now given only when the object's bottom is at
  least 0.6 m up and nothing is under it.
- **Items on furniture.** The drawer unit was "pushed towards the mechanical
  fan" (a fan standing on it), and the chair "further away from the laptop".
  Landmarks standing on larger ones are now marked as items, using the larger
  landmark's floor-plan outline, and are skipped as references.
  **This changed the chair's sentence from "laptop" to "desk"**, which is the
  wording of `room_changes.txt`. The laptop stands on that desk, so the
  measured relation is the same; only the reference object changed.
- **Two relations per move.** Each move now reports both its strongest
  "closer to" and its strongest "further away from" relation, instead of only
  one.
- **Windows and doors exempt.** The window was first rejected as a landmark
  for lying on a wall plane. Windows and doors are now exempt from that filter;
  they still have to be static.

## Limitations

- **Uncertain labels carry through.** A sentence inherits its Step 3 label.
  The door is described as "a thin white vertical panel labelled 'closet wall'
  (uncertain: the other model said 'door')", which is honest but awkward.
- **"No longer fills its spot" can mean "moved a little".** For the lamp
  edge, the photos show the lamp still in the room, slightly shifted. The
  sentence says only what was measured: the spot is empty now and no match was
  found. (An earlier version said "is gone", which overstated it.)
- **No explanation for the desk change.** OA5 is a change at the desk's edge,
  and the pipeline cannot say what changed there.
- **Precision.** Distances are gaps between voxel surfaces, good to about
  ±5 cm. Rotations are good to about ±10°.
- **Wall names need a window.** Left and right are defined by the detected
  window. A room without a detected window falls back to coordinate names
  ("wall at x = +1.31 m").
