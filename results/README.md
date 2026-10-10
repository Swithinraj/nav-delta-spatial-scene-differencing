# Reference results

Output of `navdelta run` (approach `voxel_diff`) on the provided recordings
`Myroomesh1` (before) and `Myroomesh2` (after). It is committed so the results
can be inspected without downloading the data or running anything.

| File | Content |
|---|---|
| [`room_changes.txt`](room_changes.txt) | the changes that were staged between the two recordings (the ground truth) |
| [`voxel_diff/CHANGE_REPORT.md`](voxel_diff/CHANGE_REPORT.md) | every detected change in plain English |
| [`voxel_diff/changes.json`](voxel_diff/changes.json) | the same, machine-readable ([format](../docs/changes-schema.md)) |
| [`voxel_diff/viewer/index.html`](voxel_diff/viewer/index.html) | interactive 3D viewer with before/after photos: download the folder and open the file in a browser |
| `voxel_diff/run_log.json` | command, versions, per-stage run time and result hashes |

A fresh run writes the same files to `outputs/voxel_diff/`. `navdelta verify`
checks that two runs give identical results.
