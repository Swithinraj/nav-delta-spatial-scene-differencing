## ChangeSim: tuned profile, micro-averaged

| Approach | Tuning pairs (W6) P / R / F1 | Held-out pairs (W7–W9) P / R / F1 | All pairs F1 | Claims | Changes found |
|---|---|---|---|---|---|
| 1 · voxel_diff (LiDAR voxels) | 0.39 / 0.26 / **0.31** (2) | 0.39 / 0.36 / **0.38** (3) | 0.35 | 415 | 128/396 |
| 2 · scene_graph (RGB) | 0.38 / 0.14 / **0.21** (2) | 0.39 / 0.10 / **0.16** (3) | 0.18 | 103 | 46/396 |
| 3 · loop_outliers (RGB) | 0.55 / 0.25 / **0.35** (2) | 0.46 / 0.35 / **0.40** (3) | 0.38 | 259 | 124/396 |
| 4 · rgbd_fusion (RGB-D) | 0.66 / 0.32 / **0.43** (2) | 0.76 / 0.43 / **0.55** (3) | 0.50 | 230 | 153/396 |

(n) = number of pairs scored.

## ChangeSim: run time per pair (tuned, minutes)

Wall time of the scored run on one laptop (RTX 3050 4 GB, 16 threads), shared with other jobs. G = models on the GPU, C = models on the CPU. voxel_diff: its whole pipeline (alignment to package). The other three build on the shared RGB reconstruction, listed first (2 and 3 use it, 4 reuses its MASt3R matches); each approach's row is its time after it. scene_graph's includes the 2D detections (Florence-2, SAM 2.1, SigLIP 2 on every keyframe), which rgbd_fusion reuses; rgbd_fusion's includes its own RGB-D reconstruction.

| Stage / approach | W6_S0 | W6_S1 | W7_S0 | W8_S0 | W9_S0 |
|---|---|---|---|---|---|
| shared: RGB reconstruction (MASt3R) | 28 G | 29 G | 25 G | 27 G | 33 G |
| 1 · voxel_diff (LiDAR voxels) | 51 G | 108 G | 116 G | 93 G | 47 G |
| 2 · scene_graph (RGB) | 48 G | 46 G | 35 G | 39 G | 41 G |
| 3 · loop_outliers (RGB) | 26 C | 78 C | 6 G | 113 C | 25 G |
| 4 · rgbd_fusion (RGB-D) | 27 C | 25 C | 33 C | 34 C | 27 C |

## ChangeSim: F1 per pair (tuned; first design in brackets)

| Approach | W6_S0 (tuning) | W6_S1 (tuning) | W7_S0 | W8_S0 | W9_S0 |
|---|---|---|---|---|---|
| 1 · voxel_diff (LiDAR voxels) | 0.38 (0.30) | 0.25 (0.24) | 0.39 (0.36) | 0.25 (0.26) | 0.55 (0.44) |
| 2 · scene_graph (RGB) | 0.20 | 0.21 | 0.12 | 0.17 | 0.17 |
| 3 · loop_outliers (RGB) | 0.33 | 0.37 | 0.35 | 0.35 | 0.46 |
| 4 · rgbd_fusion (RGB-D) | 0.42 (0.40) | 0.45 (0.42) | 0.40 (0.27) | 0.45 (0.43) | 0.70 (0.66) |

## ChangeSim: before vs after tuning (micro F1)

| Approach | Tuning pairs: first design → tuned | Held-out pairs: first design → tuned |
|---|---|---|
| 1 · voxel_diff (LiDAR voxels) | 0.26 → **0.31** (2) | 0.35 → **0.38** (3) |
| 2 · scene_graph (RGB) | – | – |
| 3 · loop_outliers (RGB) | – | – |
| 4 · rgbd_fusion (RGB-D) | 0.41 → **0.43** (2) | 0.50 → **0.55** (3) |

## ChangeSim: registration of the two runs (tuned)

Median position error of the second run's cameras against the ground truth, after the approach put both runs into one frame (cm). It includes how far the pose graph bent the first run. scene_graph and loop_outliers share one reconstruction.

| Approach | W6_S0 | W6_S1 | W7_S0 | W8_S0 | W9_S0 |
|---|---|---|---|---|---|
| 1 · voxel_diff (LiDAR voxels) | 18 | 70 | 75 | 82 | 17 |
| 2 · scene_graph (RGB) | 12 | 32 | 60 | 44 | 23 |
| 3 · loop_outliers (RGB) | 12 | 32 | 60 | 44 | 23 |
| 4 · rgbd_fusion (RGB-D) | 31 | 50 | 56 | 32 | 36 |

## ChangeSim: recall by type of change (tuned, all pairs)

| Approach | removed | added | rotated | replaced |
|---|---|---|---|---|
| 1 · voxel_diff (LiDAR voxels) | 0.30 (210) | 0.33 (72) | 0.37 (54) | 0.37 (60) |
| 2 · scene_graph (RGB) | 0.08 (210) | 0.28 (72) | 0.09 (54) | 0.08 (60) |
| 3 · loop_outliers (RGB) | 0.20 (210) | 0.46 (72) | 0.35 (54) | 0.48 (60) |
| 4 · rgbd_fusion (RGB-D) | 0.25 (210) | 0.58 (72) | 0.46 (54) | 0.57 (60) |

(n) = changed objects of that type in the scored pairs.

## Tuning trials (tuning pairs only)

Each trial changes a few large-space values; F1 per tuning pair and micro-averaged over both. **kept** marks the values in `large_space.toml`.

**1 · voxel_diff (LiDAR voxels)**

| Values changed | W6_S0 F1 | W6_S1 F1 | Both (micro F1) | |
|---|---|---|---|---|
| first design | 0.30 | 0.24 | 0.26 |  |
| MIN_REGION_VOXELS = 5 | 0.32 | – | – |  |
| FRAGMENT_MAX_VOXELS = 8 | 0.38 | 0.25 | 0.31 | **kept** |
| CARVE_MARGIN_M = 0.15, CARVE_MARGIN_PER_M = 0.015 | 0.30 | – | – |  |
| DEPTH_MAX_M = 16.0 | 0.32 | – | – |  |
| DEPTH_MAX_M = 16.0, FRAGMENT_MAX_VOXELS = 8, MIN_REGION_VOXELS = 5 | 0.38 | – | – |  |
| DEPTH_MAX_M = 16.0, FRAGMENT_MAX_VOXELS = 4, MIN_REGION_VOXELS = 5 | 0.39 | 0.26 | 0.32 |  |
| DEPTH_MAX_M = 20.0, FRAGMENT_MAX_VOXELS = 8, MIN_REGION_VOXELS = 5 | 0.37 | – | – |  |
| DEPTH_MAX_M = 16.0, FRAGMENT_MAX_VOXELS = 8 | – | 0.25 | – |  |

**2 · scene_graph (RGB)**

| Values changed | W6_S0 F1 | W6_S1 F1 | Both (micro F1) | |
|---|---|---|---|---|
| first design | 0.18 | 0.13 | 0.15 |  |
| GONE_MIN = 0.5 | 0.18 | – | – |  |
| SEE_PAST_M = 0.1, SEE_PAST_PER_M = 0.01, SURFACE_M = 0.1, SURFACE_PER_M = 0.01 | 0.19 | – | – |  |
| MIN_CHANGE_OBS = 2 | 0.20 | 0.19 | 0.20 |  |
| MIN_CHANGE_SIZE_M = 0.2 | 0.18 | – | – |  |
| MIN_CHANGE_OBS = 2, MIN_OBSERVATIONS = 1 | 0.21 | 0.21 | 0.21 | **kept** |
| MERGE_SIZE_RATIO = 1.5 | 0.12 | – | – |  |

**3 · loop_outliers (RGB)**

| Values changed | W6_S0 F1 | W6_S1 F1 | Both (micro F1) | |
|---|---|---|---|---|
| OUTLIER_M = 0.4, OUTLIER_PER_M = 0.0 | 0.22 | 0.24 | 0.23 |  |
| first design | 0.27 | 0.22 | 0.25 |  |
| OUTLIER_PER_M = 0.08 | 0.26 | 0.23 | 0.24 |  |
| MIN_EVIDENCE = 40 | 0.27 | 0.22 | 0.25 |  |
| MIN_MATCHES = 50 | 0.28 | 0.24 | 0.26 |  |
| RANSAC_M = 0.1 | 0.28 | 0.25 | 0.26 |  |
| MIN_MATCHES = 50, RANSAC_M = 0.1 | 0.28 | 0.26 | 0.27 |  |
| MIN_MATCHES = 50, RANSAC_M = 0.12 | 0.28 | 0.23 | 0.26 |  |
| MIN_MATCHES = 50, OUTLIER_PER_M = 0.06, RANSAC_M = 0.1 | 0.31 | 0.25 | 0.28 |  |
| MIN_MATCHES = 40, RANSAC_M = 0.1 | 0.28 | 0.24 | 0.26 |  |
| CELL_M = 0.15, MIN_MATCHES = 50, RANSAC_M = 0.1 | 0.33 | 0.28 | 0.30 | **kept** |
| CELL_M = 0.15, MIN_MATCHES = 50, OUTLIER_PER_M = 0.06, RANSAC_M = 0.1 | 0.31 | 0.25 | 0.28 |  |
| CELL_M = 0.2, MIN_MATCHES = 50, RANSAC_M = 0.1 | 0.28 | 0.26 | 0.27 |  |

**4 · rgbd_fusion (RGB-D)**

| Values changed | W6_S0 F1 | W6_S1 F1 | Both (micro F1) | |
|---|---|---|---|---|
| first design | 0.40 | 0.41 | 0.41 |  |
| fuse.GRAPH_ALONE_EMPTY = 0.7 | 0.40 | – | – |  |
| fuse.GRAPH_ALONE_EMPTY = 0.8 | 0.40 | – | – |  |
| fuse.GRAPH_ALONE_OBS = 3 | 0.40 | – | – |  |
| fuse.GRAPH_ALONE_ROTATED = False | 0.41 | 0.38 | 0.39 |  |
| fuse.GRAPH_ALONE_SEM = 0.85 | 0.40 | – | – |  |
| fuse.ALONE_MIN_MATCHES = 30, fuse.ALONE_MIN_PAIRS = 2 | 0.39 | – | – |  |
| fuse.ALONE_MIN_MATCHES = 100 | 0.41 | 0.41 | 0.41 |  |
| fuse.VETO_PRESENT = 0.3 | 0.39 | – | – |  |
| fuse.VETO_PRESENT = 1.01 | 0.42 | 0.45 | 0.43 | **kept** |
| fuse.ASSOC_M = 0.5 | 0.40 | – | – |  |
| fuse.ASSOC_SHARE = 0.15 | 0.40 | – | – |  |
| fuse.ALONE_MIN_MATCHES = 1000000, fuse.GRAPH_ALONE_EMPTY = 1.01 | 0.35 | – | – |  |
| diff.GONE_MIN = 0.5 | 0.41 | – | – |  |
| fuse.ASSOC_M = 0.3 | 0.40 | 0.41 | 0.41 |  |
| fuse.VETO_PRESENT = 0.8 | 0.42 | 0.40 | 0.41 |  |
| fuse.ALONE_MIN_MATCHES = 80 | 0.41 | 0.41 | 0.41 |  |
| fuse.ALONE_MIN_PAIRS = 6 | 0.40 | 0.41 | 0.41 |  |
| fuse.ALONE_MIN_MATCHES = 100, fuse.VETO_PRESENT = 1.01 | 0.43 | 0.44 | 0.43 |  |
| fuse.ALONE_MIN_MATCHES = 100, fuse.GRAPH_ALONE_ROTATED = False, fuse.VETO_PRESENT = 1.01 | 0.44 | 0.41 | 0.42 |  |
| fuse.ALONE_MIN_MATCHES = 100, fuse.VETO_PRESENT = 0.8 | 0.42 | 0.41 | 0.42 |  |
| diff.MIN_CHANGE_OBS = 1 | 0.41 | – | – |  |

## Our room (3 staged changes)

| Approach | Staged changes found | Reported as moved with both places right | Extra claims |
|---|---|---|---|
| 1 · voxel_diff (LiDAR voxels) | 3/3 | 3/3 | 5 |
| 2 · scene_graph (RGB) | 3/3 | 3/3 | 1 |
| 3 · loop_outliers (RGB) | 3/3 | 3/3 | 0 |
| 4 · rgbd_fusion (RGB-D) | 3/3 | 3/3 | 1 |
