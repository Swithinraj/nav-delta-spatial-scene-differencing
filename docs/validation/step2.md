# Step 2 validation: objects and moves

> Written while developing the pipeline on the provided recordings, in order. Later steps corrected some earlier readings; the corrections are marked in place rather than hidden. Paths such as `qa/…`, `objects/…`, `semantics/…` and `viewer/…` are inside `outputs/voxel_diff/` after `navdelta run`.

Step 2 (`navdelta/approaches/voxel_diff/objects.py`) groups Step 1's change regions
into objects. It then pairs removed objects with added objects to find moves.
It uses only geometry and colour, with no learned model, and is fully
deterministic. Two runs give the same result hash, `b3b88e3d…7ee5`, on top of
Step 1's hash `0c1c5a6c…3267570e`.

Outputs:
- `object_changes.json`: objects, events, score matrix and the evidence behind each decision.
- `objects/*.ply`: dense coloured points of each object, taken from the scan it appears in.
- `objects/object_voxels.npz`: the voxels of each object.
- `qa/step2_objects_and_moves.png`: map and score matrix.
- `qa/object_views_p*.png`: real photos of every event.

## Listed changes: 3 / 3 recovered as moves

| Known change | Event | Old → new | Distance | Orientation | Score (margin) | Path between |
|---|---|---|---|---|---|---|
| Trolley bag turned perpendicular to the wall | M1 | R1 → A2 | 0.54 m | Rotated 99° (shape fit 0.95). Long axis was 10° from the side wall x = −1.44, so parallel to it. After rotating, it is 71° from that wall, so perpendicular. | 0.88 (0.12) | 100% seen empty |
| Tower fan moved near the table | M2 | R5 → A4 | 0.71 m | Not determinable: the fan is a round column and fits equally well at every angle. | 0.88 (0.12) | 100% seen empty |
| Office chair moved away from the table | M3 | R3+R4+R11 → A1+A7 | 0.57 m | Not claimed: the two partial views share only 45% of their voxels. | 0.63 (0.26) | 77% seen empty, 3% occupied in both |

"Path between" checks the straight line from the old position to the new one.
In all three cases at least one recording saw that space empty, which is the
signature of a real move. In the photos (`object_views_p1.png`), every move
shows the object at its old spot in scan 1 and that spot empty in scan 2. The
new spot is empty in scan 1 and shows the object in scan 2.

On the bag's orientation, two estimates disagree slightly. The angle above (71°)
comes from the fitted rotation. The new footprint on its own gives 63°, because
a strip of wall-corner residue next to the bag tilts it. That puts the bag at
63–71° from the side wall: perpendicular within 20–27°. The rotation itself is
99° rather than 90°, because the yaw search uses 3° steps over partial views.

## Other events (real, but not staged)

| Event | Regions | What the photos show |
|---|---|---|
| SH1 shifted, about 0.21 m (±0.05) | R9 → A6 | The white drawer unit by the bed slid. R9 is its old trailing face and A6 its new leading face, both 0.72 m tall (the unit's height) and 0.58 m apart (its length). This was first accepted as a 0.58 m "move" with score 0.92. The path test exposed it: 55% of the path between the faces is surface present in both scans. *Corrected in Step 5:* the slide was first reported as "≤ 0.11 m" (the thickness of the changed layer). Step 1 hides the first 10 cm in front of the other scan's surface, so the real slide is the layer plus that tolerance: about 0.21 m (±0.05). The matched before/after photos show a gap of that order between the unit and the lamp. |
| S1 surface changed | SA1–SA5 | The brown sheer window curtains. Only scan 2's depth registers them. *Corrected in Steps 3–5:* the photos show the curtains across the window in both recordings, so the pipeline now reports this as a depth-only difference, not "drawn closed". |
| OA3 added | A3 | The lower part of the same curtains, 5–20 cm in front of the main sheet. That is beyond the 10 cm surface tolerance, so geometry keeps it separate. Step 3 labels it "curtain" as well, so Step 4 merges it with S1. |
| OR2 removed | R2+R8 | The door in the far alcove. The top and bottom parts were joined as one thin coplanar panel. *Corrected in Steps 3–5:* the photos show the door unchanged in both recordings, so this is a depth-only difference. |
| OR5 removed | R6 | The edge of the large white floor lamp next to the bag (labelled in Step 3). The before/after photos show the lamp still in the room, but its edge has left this spot, so the lamp shifted slightly. Its new edge was too thin to become a counterpart. |
| OR6 removed | R7 | An item on a shelf top. |
| OR8 removed | R10 | A plastic bag that is gone from a shelf cubby. |
| OA5 added | A5 | Something at the side of the desk, near the fan's new spot. Not identified yet. |
| X1–X4 small | R12, R13, A8, A9 | Each is smaller than a 15 cm cube. They are reported but not interpreted. |

## How the decisions are made

**Grouping, in order:**
1. **One object-sized piece of surface.** Regions lying on one connected piece of the scan's own surface are one object, if that piece fits in 1.2 × 1.2 × 1.5 m. This joined the chair's backrest and seat.
2. **Thin coplanar panels.** Thin panels in the same plane that overlap along it are one object. This joined the door's top and bottom.
3. **Fragments.** A fragment smaller than a 15 cm cube joins its nearest neighbour within the 10 cm surface tolerance. This attached the chair's base pieces, R11 and A7.

**Surfaces:** a Step 1 sheet absorbs pieces whose floor-plan columns all lie within 10 cm of it. This rule doesn't absorb A3, as described above.

**Descriptors:**
- Slices at the top or bottom of an object holding less than 10% of its widest slice are trimmed. This removed a 1-voxel wall-corner strip from the bag's new position, which had made it look 1.2 m tall instead of 0.72 m.
- Footprints are weighted by column height for the same reason.

**Score:** the weighted geometric mean shape^0.5 × height^0.25 × colour^0.25.
- **Shape:** the best overlap over yaw in 3° steps and offsets up to ±10 cm, with a 7.5 cm tolerance.
- **Height:** top height above the floor; the score reaches 0 at a 30 cm difference.
- **Colour:** CIE76 difference of the median colour; the score reaches 0 at ΔE 40.

The Hungarian algorithm then makes a one-to-one assignment, keeping only pairs that score at least 0.5.

**Move or shift:** an accepted pair counts as a shift when at least 25% of the path between the two positions is surface that exists in both scans.

## Rules changed after seeing results

These three changes were made while looking at this scene, where the answer
is known. Each was triggered by a visible failure and is physically motivated:

- **Additive score → geometric mean.** The additive score let shape + colour
  carry a pair with a height difference of 0.55 m: a shelf item matched to
  the curtain 3 m away. With the geometric mean, one cue at zero vetoes the pair. As
  a side effect, the chair's margin grew from 0.02 to 0.26.
- **Shift test.** This was added after the photos showed SH1 was the drawer
  unit's two faces. The threshold sits between the measured values: 0.55 for
  the drawer unit against 0–0.03 for the three real moves.
- **Rotation claims only at shape fit ≥ 0.6.** The chair's 45% fit had
  produced an unsupported 168° rotation.

None of the three changes is needed to recover the three staged moves. All
earlier versions found them too. The changes remove false or unsupported claims.

## Limitations

- **Partial views.** Matching works on the voxels that changed, which are
  partial views. The chair's two views share only 45% of their voxels, so its
  match relies on height and colour, plus the global assignment.
- **Few, mostly dark objects.** The fan, bag and chair are all dark, so colour
  barely tells them apart. The fan's old position also scores 0.76 against the
  bag's new position. The assignment resolves this only because the bag's own
  match is stronger (0.88).
- **Rotation resolution.** Rotation comes from a 3° search over 5 cm voxels.
  Expect errors of about ±10° on elongated objects. Round objects have no
  measurable rotation.
- **No semantics yet.** Step 2 knows positions and sizes, not what objects
  are, so "near the table" can't be checked yet. Labels come from the
  open-source models in Step 3.
