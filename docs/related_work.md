# How approaches 3 and 4 differ from existing change detection

Finding what changed between two visits to a space is studied in two communities. Long-term SLAM
keeps a robot's map current. Scene change detection finds the objects that were added, removed or
moved between two captures. Most methods in both share one recipe:

1. **Align** the two captures: a SLAM back end, SfM, or a feed-forward geometry model.
2. **Throw away what does not fit the alignment.** Correspondences that disagree with it are
   outliers and are rejected.
3. **Compare** the aligned captures in a separate step: depth or occupancy differences, appearance
   or semantic correspondences, object association.

Approaches 3 (`loop_outliers`) and 4 (`rgbd_fusion`) change step 2. The correspondences the
alignment rejects are kept: **they are the change candidates.** This document says how that differs
from the closest work, and what it gains on our data. We present it as a different design, not as
a new principle. Every ingredient (robust outlier rejection, depth-consistency tests, multi-model
fitting, semantic and geometric fusion) is established.

## The closest work

| Method | Aligns by | Change signal | What happens to the alignment's outliers | Motion of a moved object |
|---|---|---|---|---|
| Robust loop closing: RRR [1], Kimera / PCM [2], DARE-SLAM [3] | SLAM loop closures | none (not a change detector) | rejected as errors, mostly attributed to perceptual aliasing | – |
| DynaVINS [4], DYNEMO-SLAM [5] | visual(-inertial) SLAM | objects that moved between loop-closure visits are detected | grouped / filtered so the loop closure stays robust; not reported as changes | DYNEMO-SLAM estimates dynamic entity poses inside its graph |
| Dynamic Pose Graph SLAM [6] | LiDAR scan matching | scans of one place at two times compared after alignment | – | – |
| Persistence filter [7], temporary maps [8], ROLL [9] | feature / LiDAR localization | a feature's (non-)re-detection; the localization inlier ratio as an anomaly trigger | used as statistics: feature survival, or a switch to a temporary map | – |
| Khronos [10] | metric-semantic SLAM | change detection over its long-term factor graph, separate from tracking | – | – |
| OASIS-Map [11] | multi-session LiDAR SLAM (given poses) | dense semantic patch correspondences (DINOv3, SAM 2) | – | centroid displacement of associated objects |
| SceneDiff [12] | feed-forward joint geometry (π³) | depth-reprojection difference + DINOv3 appearance + SAM region matching | – | moved = before / after objects matched by appearance |
| 3DGS-CD [13] | Gaussian-splatting model of the first capture | rendered-vs-observed difference, object templates | – | RANSAC-PnP of the object template |
| VGGT-CD [14] | VGGT + robust Sim(3) on the static background | change maps after the robust alignment | static-background filtering for the alignment | not stated |
| Multi-model 3D registration [15] | – (one pair of point clouds) | – (registration, not change detection) | correspondences assigned to several rigid motions by EM | one rigid motion per object |
| **3 `loop_outliers`** (ours) | **cross-video MASt3R loop closure + pose graph** [16] | **the loop closure's own outlier matches** | **kept, and each is typed by checking both ends against the other video's depth** | **sequential RANSAC on the object's own matches: distance and turn** |
| **4 `rgbd_fusion`** (ours) | the same pose graph on the recording's depth maps | the outliers of approach 3 **and** an object scene graph (approach 2), fused per object | as approach 3, with sensor depth | from the outliers when they measured it, else from the graph |

## What is different in approach 3

1. **One signal, two uses.** The cross-video matches that lock the two recordings into one frame
   are the same matches that reveal the changes. Robust SLAM rejects them as outliers [1–3]. Work on
   objects that moved between visits removes them to protect the loop closure [4, 5]. Approach 3
   reports them. Long-term mapping uses mismatch statistics as triggers or survival estimates
   [7–9]. Here, every outlier is a located piece of evidence about one object.
2. **Changes are typed before any object is segmented.** Each outlier match has two ends, and each
   end is projected into the other video's depth, a DynaSLAM-style depth-consistency test [17]
   applied across sessions:

   | A's end is now empty space | B's end was empty space | type |
   |---|---|---|
   | yes | yes | moved |
   | yes | no | removed |
   | no | yes | added |
   | no | no | discarded: mismatch or aliasing |

   This separates real changes from the perceptual aliasing that SLAM papers name as the main
   cause of loop-closure outliers [2]. The change detectors that compare after alignment do not
   test both ends of a correspondence against the other capture's depth. They compare depth,
   appearance or semantic features region by region [11–13].
3. **A moved object's motion comes from its own features.** The moved matches are split into rigid
   motions by sequential RANSAC, the multi-model fitting idea of [15], here run on the residue of a
   loop closure. That gives each object a distance and a turn, which is how the trolley bag's 80°
   turn was measured. Object-association methods report a moved object from the displacement of
   matched instances [11, 12]. 3DGS-CD estimates it per object template [13].
4. **Only what changed is segmented.** SAM 2.1 is prompted with the outlier pixels of each change,
   in 1–2 views. Whole-scene methods segment every frame of both captures [11, 12], as our approach
   2 does.

## What is different in approach 4

1. **Two detectors that fail differently, fused per object.** The scene graph knows which object
   it is: its whole outline, its name and its relations. The outliers know how it moved. Semantic
   RGB-D SLAM combines instance masks with multi-view geometry to find what is dynamic [17, 18].
   Approach 4 does the same between two sessions, and its geometric side is the loop-closure
   outliers of approach 3. SceneDiff also combines geometry, appearance and region cues, but as a
   weighted score per region in one pass [12]. Approach 4 keeps two complete detectors and decides
   per object. Both agree: confirmed. One alone: kept only if its own evidence is strong.
2. **Association by the ends of a change.** A change has a before end and an after end, and the
   detectors are matched end by end. So one outlier move can join a graph "removed" here and a
   graph "added" there into one move. A look-alike pairing of identical boxes is split instead of
   trusted.
3. **Measured depth with RGB-D SLAM's habits.** Approach 4 uses sensor depth instead of predicted
   depth, drops depth-edge (flying) pixels, uses range-scaled noise gates and runs 3D–3D RANSAC
   verification [19].

## Why it is better on our data

All four approaches were run with the same reconstruction, scorer and profiles, so the comparison
between them is fair. We did not run the published methods above. Their numbers use other
protocols and metrics and are not comparable, so "better" here means better than the other
designs we built ([`benchmark.md`](benchmark.md)).

| | 1 voxel differencing | 2 scene graphs | 3 loop-closure outliers | 4 fusion of 2 + 3 |
|---|---|---|---|---|
| ChangeSim, held-out warehouses: F1 | 0.38 | 0.16 | 0.40 | **0.55** |
| precision / recall | 0.39 / 0.36 | 0.39 / 0.10 | 0.46 / 0.35 | **0.76 / 0.43** |
| the room: staged changes found, extra claims | 3/3, 5 | 3/3, 1 | **3/3, 0** | 3/3, 1 |
| the room: run time after the shared reconstruction | (11 min in all) | 15 min | **1 min** | 18.5 min |

- **The outliers beat the object comparison built on the same reconstruction.** Approach 3 reaches
  F1 0.40 against approach 2's 0.16, on the same MASt3R reconstruction. It segments only the
  changed objects and runs in about a fifteenth of the time.
- **The fusion beats either part.** Approach 4 reaches 0.55 against 0.40 for the outliers alone
  and 0.16 for the graph alone. It was best on each of the five warehouse pairs.
- **Its advantage is precision.** 0.76 of approach 4's claims are real changes.

## Limits of the comparison

- A change is found only where the two recordings were matched (approach 3), or where the scene
  graph has an object (approach 4). A small object seen in one or two view pairs is missed.
- Benchmark: one simulated dataset (five warehouse pairs) and one real room. The held-out pairs
  were never used for tuning.
- This is not a claim that no one has done this. It summarises what a literature search in October
  2026 found: robust SLAM papers, long-term mapping and two-capture change detection.

## References

1. Latif, Cadena, Neira. *Robust Loop Closing Over Time for Pose Graph SLAM.* IJRR 2013. [pdf](http://webdiis.unizar.es/~ylatif/papers/IJRR.pdf)
2. Rosinol et al. *Kimera: from SLAM to Spatial Perception with 3D Dynamic Scene Graphs.* IJRR 2021. [arXiv:2101.06894](https://arxiv.org/abs/2101.06894)
3. Ebadi et al. *DARE-SLAM: Degeneracy-Aware and Resilient Loop Closing in Perceptually-Degraded Environments.* [arXiv:2102.05117](https://arxiv.org/abs/2102.05117)
4. Song, Lim, Lee, Myung. *DynaVINS: A Visual-Inertial SLAM for Dynamic Environments.* RA-L 2022. [arXiv:2208.11500](https://arxiv.org/abs/2208.11500)
5. *DYNEMO-SLAM: Dynamic Entity and Motion-Aware 3D Scene Graph SLAM.* [arXiv:2503.02050](https://arxiv.org/abs/2503.02050)
6. Walcott-Bryant, Kaess, Johannsson, Leonard. *Dynamic Pose Graph SLAM: Long-term Mapping in Low Dynamic Environments.* IROS 2012. [page](https://people.csail.mit.edu/kaess/pub/WalcottBryant12iros.html)
7. Rosen, Mason, Leonard. *Towards Lifelong Feature-Based Mapping in Semi-Static Environments.* ICRA 2016. [MIT DSpace](https://dspace.mit.edu/handle/1721.1/107620)
8. *Temporary maps for robust localization in semi-static environments.* [ResearchGate](https://www.researchgate.net/publication/221065605_Temporary_maps_for_robust_localization_in_semi-static_environments)
9. *ROLL: Long-Term Robust LiDAR-based Localization With Temporary Mapping in Changing Environments.* [arXiv:2203.03923](https://arxiv.org/abs/2203.03923)
10. Schmid et al. *Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments.* RSS 2024. [arXiv:2402.13817](https://arxiv.org/abs/2402.13817)
11. *OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching.* [arXiv:2607.14899](https://arxiv.org/abs/2607.14899)
12. *SceneDiff: A Benchmark and Method for Multiview Object Change Detection.* [arXiv:2512.16908](https://arxiv.org/abs/2512.16908)
13. *3DGS-CD: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement.* RA-L 2024. [arXiv:2411.03706](https://arxiv.org/abs/2411.03706)
14. *VGGT-CD: Training-Free Robust Registration for 3D Change Detection.* [arXiv:2605.16859](https://arxiv.org/abs/2605.16859)
15. Jin, Karmalkar, Zhang, Carlone. *Multi-Model 3D Registration: Finding Multiple Moving Objects in Cluttered Point Clouds.* ICRA 2024. [arXiv:2402.10865](https://arxiv.org/abs/2402.10865)
16. Leroy, Cabon, Revaud. *Grounding Image Matching in 3D with MASt3R.* ECCV 2024. [arXiv:2406.09756](https://arxiv.org/abs/2406.09756)
17. Bescos, Fácil, Civera, Neira. *DynaSLAM: Tracking, Mapping and Inpainting in Dynamic Scenes.* RA-L 2018. [arXiv:1806.05620](https://arxiv.org/abs/1806.05620)
18. Rünz, Buffier, Agapito. *MaskFusion: Real-Time Recognition, Tracking and Reconstruction of Multiple Moving Objects.* ISMAR 2018. [arXiv:1804.09194](https://arxiv.org/abs/1804.09194)
19. Campos et al. *ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM.* T-RO 2021. [arXiv:2007.11898](https://arxiv.org/abs/2007.11898)

Datasets: ChangeSim (Park et al., IROS 2021, [github.com/SAMMiCA/ChangeSim](https://github.com/SAMMiCA/ChangeSim)); for
related indoor rescans, 3RScan and Living Scenes ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Living_Scenes_Multi-object_Relocalization_and_Reconstruction_in_Changing_3D_Environments_CVPR_2024_paper.pdf)).
