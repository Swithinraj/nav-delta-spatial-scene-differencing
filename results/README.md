# Reference results

Output of each approach on our room recordings `Myroomesh1` (before) and `Myroomesh2`
(after), with the `room` profile. It is committed so the results can be inspected without
downloading the data or running anything.

[`room_changes.txt`](room_changes.txt) lists the changes that were staged between the two
recordings (the ground truth).

| Approach | Report | Data | 3D viewer | Run record |
|---|---|---|---|---|
| 1 `voxel_diff` | [CHANGE_REPORT.md](voxel_diff/CHANGE_REPORT.md) | [changes.json](voxel_diff/changes.json) | [viewer](voxel_diff/viewer/index.html) | `voxel_diff/run_log.json` |
| 2 `scene_graph` | [CHANGE_REPORT.md](scene_graph/CHANGE_REPORT.md) | [changes.json](scene_graph/changes.json) | [viewer](scene_graph/viewer/index.html) | `scene_graph/run_log.json` |
| 3 `loop_outliers` | [CHANGE_REPORT.md](loop_outliers/CHANGE_REPORT.md) | [changes.json](loop_outliers/changes.json) | [viewer](loop_outliers/viewer/index.html) | `loop_outliers/run_log.json` |
| 4 `rgbd_fusion` | [CHANGE_REPORT.md](rgbd_fusion/CHANGE_REPORT.md) | [changes.json](rgbd_fusion/changes.json) | [viewer](rgbd_fusion/viewer/index.html) | `rgbd_fusion/run_log.json` |

- `CHANGE_REPORT.md`: every detected change in plain English.
- `changes.json`: the same, machine-readable ([format](../docs/changes-schema.md)).
- `viewer/index.html`: the interactive 3D viewer with before/after photos. Download the folder
  and open the file in a browser.
- `run_log.json`: command, versions, profile, per-stage run time, peak GPU memory and result
  hashes of the run that produced the folder: `navdelta verify`, which ran every stage twice and
  found identical results. Approach 1: every stage, alignment included. Approaches 2–4: their own
  stages, 2D detections included, on one shared reconstruction (which separately reproduced bit
  for bit from scratch).

A fresh run writes the same files to `outputs/<approach>/`. `navdelta verify --approach <name>`
checks that two runs give identical results.
