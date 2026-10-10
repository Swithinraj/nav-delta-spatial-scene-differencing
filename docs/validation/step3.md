# Step 3 validation: open-vocabulary labels

> Written while developing the pipeline on the provided recordings, in order. Later steps corrected some earlier readings; the corrections are marked in place rather than hidden. Paths such as `qa/…`, `objects/…`, `semantics/…` and `viewer/…` are inside `outputs/voxel_diff/` after `navdelta run`.

Step 3 (`navdelta/approaches/voxel_diff/labels.py`, run by `navdelta run`) names every
changed object. It uses only off-the-shelf open models: nothing is trained or
fine-tuned, and no data is labelled by hand.

| Model | Role | License | Revision |
|---|---|---|---|
| `facebook/sam2.1-hiera-base-plus` | object mask in each view, from a box + point prompt built from the projected voxels | Apache-2.0 | `b7320756` |
| `florence-community/Florence-2-large` | open-vocabulary name and description of the masked region | MIT | `4271c66b` |
| `google/siglip2-so400m-patch14-384` | zero-shot cross-check against a fixed vocabulary | Apache-2.0 | `e8e48729` |

The SigLIP 2 vocabulary is the 200 class names of the public ScanNet200 indoor
benchmark. The file comes from the official repository and is parsed as data,
never executed (sha256 `28a36582…`). Every name Florence-2 proposes is added to
it at run time.

How a label is decided:
- **Views:** each object end uses up to 5 diverse views, chosen where depth confirms the surface.
- **Upright crops:** each crop is turned upright using the aligned camera pose. The phone video is mostly sideways.
- **Florence-2's vote:** it only counts with a strict majority of the views.
- **Confirmed label:** a majority name is confirmed when one of SigLIP 2's top 3 names the same object, either identically or with the same head noun ("chair" / "office chair"). The more specific name is kept.
- **Otherwise:** SigLIP 2's choice is used, and it is flagged *uncertain* unless Florence-2 proposed a compatible name in some view.

Run time: 92 s on the RTX 3050 Laptop GPU (4 GB). The models are loaded one at
a time, in fp16 where possible. With the photo cross-check described below,
one run takes 150 s, and repeated runs gave the same result hash,
`f6dd5b19…63ed`.

## Results, checked against the crops (`qa/step3_labels_p*.png`)

| Event | Label | Flag | Florence-2 votes | SigLIP 2 top 3 | Crops show |
|---|---|---|---|---|---|
| M1 (bag) | suitcase | confident | suitcase 7/7 (4 old, 3 new) | luggage, suitcase, bag | ✔ navy suitcase at both ends |
| M2 (fan) | fan | **uncertain** | split: remote control 3, loudspeaker 2, vase 1 | fan, speaker, column | ✔ black tower fan. SigLIP 2 is right; Florence-2 never said "fan". |
| M3 (chair) | office chair | confident | chair 4, person 2, coat/jacket/dog 1 each | office chair, coat, jacket | ✔ black office chair with a black coat over the backrest, which explains the coat/person votes |
| SH1 (shift) | drawer | confident | drawer 4, chest of drawers 2, toilet 1 | drawer, file cabinet, dresser | ✔ white drawer unit at both ends, which confirms Step 2's "shift" reading |
| S1 (surface) | curtain | confident | curtain 4/4 | curtain, office chair, radiator | ✔ brown sheer curtains. The label is right, but "drawn closed" was wrong; see the photo cross-check below. |
| OA3 (added) | curtain | confident | curtain 4/4 | curtain, shower curtain, windowsill | ✔ lower part of the same curtains, so Step 4 merges it with S1 |
| OR8 (removed) | plastic bag | confident | plastic bag 1/1 | plastic bag, closet, bag | ✔ plastic bag in a shelf cubby |
| OA5 (added) | desk | confident | desk 3/3 | desk, keyboard, computer tower | ✔ the region is the desk's front edge with papers on it. *What* changed there is not resolved: the desk may have moved slightly, or the items on it changed. |
| OR5 (removed) | lamp | **uncertain** | split: vase, toilet paper | lamp, vase, ceiling | ✔ white ribbed floor lamp with a round base |
| OR2 (removed) | closet wall | **uncertain** | door 2/2 | closet wall, telephone, doorframe | ✘ it is a door with a wall intercom. Florence-2 was right; SigLIP 2 was distracted by the intercom. |
| OR6 (removed) | bathroom cabinet | **uncertain** | refrigerator 1/1 | bathroom cabinet, cabinet, kitchen cabinet | ✘ the crop shows the shelf, and the removed item itself can't be identified |

**Calibration:**
- **Confident labels:** all 7 are correct names for what is in the region.
- **Uncertain labels:** 2 of the 4 are correct (fan, lamp) and 2 are wrong (door, shelf item).
- **Staged moves:** all three objects get the right names: suitcase, fan and office chair.
- **No ground truth:** there are no ground-truth labels for this scene, so these checks are my own reading of the crops, not a measured accuracy.

## Rules changed after seeing results

These changes were made while looking at this scene's output:

- **Vocabulary.** The first run used a 100-word list I wrote myself. It
  contained "tower fan", "suitcase", "office chair" and "window blind", which I
  knew were in the scene. I replaced it with the published ScanNet200 list.
  Effect: the fan's label went from "tower fan" to the more generic "fan". No
  other staged label changed.
- **Compatible names.** "chair" vs "office chair" had been reported as a
  disagreement. The head-noun rule now treats them as one object at two levels
  of detail.
- **Strict Florence-2 majority.** At the chair's new position, Florence-2's four
  views said person, coat, jacket and dog, one vote each. The tie-break picked
  "coat" over SigLIP 2's top "office chair". A split vote now no longer decides.

## Photo cross-check (added after Step 5 exposed two false changes)

Step 5's before/after photos from matching viewpoints showed that two
"changes" were not real. The sheer curtains hang across the window in
both recordings, and the far door is closed and unchanged in both. Only the
depth differed. The labelling step now also labels **the other recording's**
photos at the same spot: up to 3 frames whose depth sees through the spot,
using the same models and the same decision rule. If both recordings name the
same object there (final label or Florence-2 majority, same head noun), the
change is reported as **depth-only**.

| Change | Labelled where it is | Other recording at the same spot | Verdict |
|---|---|---|---|
| S1 curtain (+ OA3) | curtain | curtain | depth-only ✔ (matches the photos) |
| OR2 far door | closet wall / door | door / closet door | depth-only ✔ (matches the photos) |
| OA5 desk edge | desk | desk | change at the desk, consistent with a small nudge |
| OR5 lamp edge | lamp | closet wall | real: the lamp's edge left this spot ✔ |
| OR8 plastic bag | plastic bag | bathroom stall | real: the spot is empty now |
| OR6 shelf item | bathroom cabinet | closet door | not resolved; the two viewpoints differ by 70° |

**Not used for moves:** near a moved object, the other recording's crop can
contain the object at its other position. The suitcase moved only 0.54 m, so
both of its spots "still show a suitcase". For moves the check is stored as
evidence but never used in the sentences.

## Corrections this step made to earlier steps

- **Window change (Steps 1–2).** It was first read as the blinds being
  lowered, then, from scan-2 crops only, as the curtains being drawn closed.
  Both readings were wrong: it is a depth-only difference on the sheer
  curtains, which appear in both recordings' photos.
- **Far door (Steps 1–2).** It was read as the door having moved. The
  photos show it unchanged, so it is a depth-only difference.
- **"Pillow" (Step 1).** The regions first read as a "white body pillow" are
  two different things: the edge of a floor lamp (R6) and the back face of the
  drawer unit (R9).

## Limitations

- **Few views for some objects.** Some objects have only 1–2 usable views (the
  door, the fan's old position, shelf items), so a single bad crop can decide
  their label.
- **Florence-2's region naming is weak on thin, uncommon objects.** It never
  recognised the tower fan.
- **Labels are about what is in the region.** A label names what is visible in
  the recording where the region is occupied. It cannot say *why* a region
  changed (OA5: "desk", but what about the desk changed?).
- **The vocabulary includes catch-all classes.** ScanNet200 contains entries
  such as "object", "structure" and "closet wall", which can win on plain
  surfaces.
