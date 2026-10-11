# Step 5 validation: changes.json, before/after gallery and 3D viewer

> Written while developing the pipeline on our room recordings, in order. Later steps corrected some earlier readings; the corrections are marked in place rather than hidden. Paths such as `qa/…`, `objects/…`, `semantics/…` and `viewer/…` are inside `outputs/voxel_diff/` after `navdelta run`.

Step 5 (`navdelta/approaches/voxel_diff/package.py`, run by `navdelta run`) packages Steps 1–4.
No learned model runs here.

| Output | What it is |
|---|---|
| `outputs/voxel_diff/changes.json` | One record per change, with sections: what (label, confidence, evidence from both models), where (before/after centroid, bounding box, place), how (distance, rotation, orientation, path test), media (photos, point clouds), and provenance (every stage hash, model ids and revisions). sha256 `8ac7e18e…c7d52`, identical over repeated builds. |
| `viewer/gallery/*.jpg` | A before and an after photo per change, plus a side-by-side `_pair.jpg` (20 + 10 images, 2.0 MB). |
| `viewer/index.html` + `viewer/data/*.js` | Interactive 3D viewer (5.7 MB of data). Open the file in a browser; no server is needed. It needs internet only for three.js 0.160.0 (jsDelivr) and the IBM Plex fonts. |
| `outputs/voxel_diff/CHANGE_REPORT.md` | The plain-language list of changes from Step 4, unchanged. |

Final counts:
- 3 moves (suitcase, fan, office chair);
- 1 nudge (chest of drawers, about 0.21 m);
- 3 items that no longer fill their spots (lamp edge, shelf item, plastic bag);
- 1 change at the desk's edge;
- 2 depth-only differences (curtains, far door);
- 4 tiny changes, not interpreted.

## How the photo pairs are chosen

For each change, every keyframe of each recording gets a score: the share of
the object's voxels its depth shows as **present** (surface hit) where the
object is, and as **seen through** where it is not. For moves, both the old
and new spots are scored.

The 25 best frames of each recording are paired, and the pair with the highest
score × viewpoint similarity wins. Viewpoint similarity is
exp(−distance / 0.6 m) × exp(−angle / 30°).

In each photo, the object is *filled* where it is in that recording, and
*dashed* where it is in the other recording. The crops are rotated upright
with the camera pose.

| Change | Viewpoint difference | Comment |
|---|---|---|
| M1 suitcase | 0.42 m, 5° | clear: bag against the wall → bag in the corner, turned |
| M2 fan | 0.37 m, 18° | clear: tower fan beside the bed → its old spot is empty floor |
| M3 office chair | 0.53 m, 7° | clear: chair by the desk → chair by the radiator |
| SH1 chest of drawers | 0.25 m, 16° | a gap between the lamp and the unit opens up |
| S1 curtain | 0.54 m, 6° | **the curtains are across the window in both photos**, see below |
| OR2 far door | 0.06 m, 12° | **the door is identical in both photos**, see below |
| OR5 lamp edge | 0.42 m, 5° | the lamp is still there; its edge has left the marked spot |
| OA5 desk edge | 0.57 m, 2° | the front table sits differently; what exactly changed is not resolved |
| OR8 plastic bag | 1.71 m, 38° | weak pair: the viewpoints differ a lot |
| OR6 shelf item | 1.70 m, 71° | weak pair: the viewpoints differ a lot |

## What the gallery caught, and what was changed because of it

These photos are the first place where both recordings are compared side by
side from (nearly) the same viewpoint. They exposed four errors in my earlier
reports, and the pipeline was changed to handle each one automatically:

1. **The curtains were not drawn.** Step 3 had confirmed "curtain" from scan-2
   crops only, and Step 4 wrote "drawn closed". The pair shows the same sheer
   curtains across the window in both recordings; only the depth differs.
   **Fix:** Step 3 now also labels the other recording's photos at the same
   spot. If both recordings name the same object, Step 4 reports a
   *depth-only difference*.
2. **The far door did not move.** It is closed and unchanged in both photos.
   The same photo cross-check now reports it as depth-only.
3. **The drawer unit slid about 0.2 m, not "≤ 0.11 m".** The changed slab is
   thinner than the real slide by Step 1's 10 cm surface tolerance. **Fix:**
   Step 2 now reports the layer plus that tolerance: 0.21 m (±0.05).
4. **"Lamp gone" overstated it.** The lamp is still in the room, slightly
   moved. **Fix:** "removed" changes now say "no longer fills its spot … taken
   away or moved somewhere only one recording saw".

The three staged moves were unaffected by all four fixes: their photo pairs are
unambiguous.

## Viewer

**Recording:** before, after or both point clouds, at 2 cm (245k and 222k points).

**Cut-away height:** a slider (default 1.4 m above the floor) removes the
ceiling and upper walls, so you can see into the room.

**Fade room:** dims the room so the changes stand out.

**Furniture labels:** the Step 4a landmarks.

**Change list:** clicking a change, or its tag in 3D, flies the camera to it,
highlights it, and shows its before/after photos and measured facts.

**Colours:** the same as all QA images:

| Colour | Meaning |
|---|---|
| red | position in scan 1 |
| green | position in scan 2 |
| purple | nudge |
| teal | furniture edge |
| slate | depth-only |
| blue arrow | move |

**Themes:** light and dark both follow the system or host setting.

**Testing:** checked once in headless Edge (SwiftShader WebGL) from a
`file://` URL. It renders the room, the change points, the arrows, the tags,
the change list and the legend. Clicking and the camera flights were not
tested headless.

## Limitations

- **Weak photo pairs.** Two pairs (OR6, OR8) come from very different
  viewpoints, because no frame pair sees those shelf spots well from the same
  side. Treat them as weak evidence.
- **Viewer colours.** Change points use category colours, not the objects'
  true colours (true colours are in `objects/*.ply`).
- **Offline use.** The viewer needs internet access for three.js and the
  fonts. Everything else (data, photos) is local.
