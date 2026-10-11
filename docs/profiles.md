# Scale profiles: one room, one large space

Every stage has tuning values that depend on the size of the space: voxel size, depth range,
how far apart two things may be and still be "next to" each other, how much depth error to
tolerate at 10 m. NavDelta keeps them in two profiles:

| profile | for | file |
|---|---|---|
| `room` | a room or small apartment filmed with a phone (up to ~10 m across) | [`navdelta/configs/room.toml`](../navdelta/configs/room.toml) |
| `large_space` | warehouses, halls, open-plan floors (tens of metres) | [`navdelta/configs/large_space.toml`](../navdelta/configs/large_space.toml) |

```bash
navdelta run --approach scene_graph --profile auto          # default: chosen from the camera path
navdelta run --approach scene_graph --profile large_space
navdelta run --approach scene_graph --profile my_site.toml  # your own file, same keys
NAVDELTA_PROFILE=room ./run.sh voxel_diff                   # same, through the environment
```

**auto** measures the "before" recording's camera path: if its horizontal extent spans more
than 10 m, the space is large; otherwise it is a room. The choice and the reason are printed and
stored in `run_log.json` and every `changes.json` (`profile`: name, file, reason, sha256).

## How a profile is applied

Every stage keeps its room values as documented constants at the top of its module. A profile
lists, per stage, the constants it sets — `[scene_graph.diff]` sets constants of
`navdelta/approaches/scene_graph/diff.py` — and `config.apply()` writes them over the module's
constants before the stage runs. A key the stage does not have is an error, so a typo cannot be
silently ignored.

- `room.toml` repeats the code's values exactly. `python -m navdelta.config --check` proves it,
  so the room results do not depend on the profile mechanism.
- `large_space.toml` sets the same keys with large-space values. Anything it does not list keeps
  its room value.
- Approach 4's stages wrap approach 2's and 3's stages. Each `[rgbd_fusion.<stage>]` section
  starts from the wrapped stage's values for the same profile and lists only what measured depth
  changes, such as tighter, range-scaled depth gates.

## What changes for a large space

| | room | large space | why |
|---|---|---|---|
| voxel / cell size | 5 cm | 10 cm | a 40 × 20 × 10 m hall stays at ~8 M cells |
| depth used up to | 5 m (phone LiDAR) | 12 m | aisles are long |
| depth tolerances | fixed (10 cm) | + 1.5–2 cm per metre of range | depth error grows with range |
| stereo baseline (MASt3R depth) | 30 cm | 60 cm | keeps parallax at 10–20 m |
| object size limits | chair-sized | pallets, carts (up to 4 m) | |
| "next to" / landmark distances | 0.3 m / 1.5 m | 0.6 m / 5 m | |
| look-alike pairing | 0.70–0.75 | 0.80 | many identical boxes and racks |

The large-space values were tuned on two ChangeSim warehouse pairs only (W6_S0, W6_S1). Three more
warehouses (W7, W8, W9) are held out to check that the tuning generalises, as described in
[`benchmark.md`](benchmark.md). Each value tuned there has a comment in `large_space.toml` with
the trial result that justified it.

## What the tuning changed

Starting from the first design, a value was kept only if it helped on both tuning pairs. Every
trial is listed in [`benchmark.md`](benchmark.md).

| Approach | Tuned large-space values | Tuning pairs F1 | Held-out F1 |
|---|---|---|---|
| 1 `voxel_diff` | `FRAGMENT_MAX_VOXELS = 8`: regions from a 20 cm cube up are matched as objects, not set aside as fragments | 0.26 → 0.31 | 0.35 → 0.38 |
| 2 `scene_graph` | `MIN_OBSERVATIONS = 1`, `MIN_CHANGE_OBS = 2`: a node seen once is kept, but a change needs 2 views | 0.15 → 0.21 | (final: 0.16) |
| 3 `loop_outliers` | `MIN_MATCHES = 50`, `RANSAC_M = 0.10`, `CELL_M = 0.15`: fewer, larger, more certain moved objects | 0.25 → 0.30 | (final: 0.40) |
| 4 `rgbd_fusion` | `VETO_PRESENT = 1.01` (veto off): boxes removed from racks were vetoed by the rack's node | 0.41 → 0.43 | 0.50 → 0.55 |

Approaches 2 and 3 were tuned before the final runs, by scoring the tuning pairs directly
(approach 3: its mined changes, without segmentation), so their first design has no final
held-out run. The room profile was never changed by this tuning.
