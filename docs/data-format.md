# Data format

The pipeline compares **two recordings of the same space**. Each recording is an
RGB-D phone capture exported from [Polycam](https://poly.cam) ("raw data" export
of a LiDAR scan, e.g. on an iPhone/iPad Pro). Any data converted to the same
layout works.

## Folder layout

Put both recordings in one folder (default `./data`, or pass `--data PATH`):

```text
data/
├── <name>_imgs/
│   └── keyframes/
│       ├── corrected_cameras/<frame>.json     camera pose + intrinsics per keyframe
│       ├── corrected_images/<frame>.jpg       RGB keyframe
│       └── corrected_depth/<frame>.Clean.png  depth keyframe
├── <name>_rawdata/
│   ├── mesh_info.json                         room alignment transform
│   └── keyframes/cameras/<frame>.json         raw ARKit pose per keyframe (same frame names)
└── <name>_ply/
    └── <name>.ply                             coloured point cloud of the recording
```

`<name>` is the recording's name (here `Myroomesh1`, `Myroomesh2`), and
`<frame>` is the keyframe timestamp in microseconds (an integer, e.g.
`216255589148`). The two recordings are taken in sorted order: the first is
"before" and defines the shared coordinate frame. To choose explicitly, use
`--before NAME --after NAME`.

Other files in a Polycam export (`images/`, `depth/`, `confidence/`, `polycam.mp4`,
`roomplan.json`, …) may be present. They are ignored.

## File contents

| File | Content | Used for |
|---|---|---|
| `corrected_cameras/<frame>.json` | `t_00 … t_23`: 3×4 camera-to-world pose (ARKit camera axes, Polycam Y-up room); `fx, fy, cx, cy, width, height`: pinhole intrinsics of the RGB image | initial poses, projection |
| `rawdata/keyframes/cameras/<frame>.json` | same keys, from ARKit before Polycam's correction | the visual-inertial prior of the pose graph |
| `corrected_images/<frame>.jpg` | RGB, 1024 × 768 in this dataset (any size matching the intrinsics) | SIFT features, labels, photos |
| `corrected_depth/<frame>.Clean.png` | 16-bit PNG, **millimetres**, 512 × 384 in this dataset (a lower resolution than RGB is fine; pixel centres are mapped by scale) | ray casting, back-projection |
| `mesh_info.json` | `alignmentTransform`: 16 numbers, a 4×4 matrix in **column-major** order (Polycam's room alignment) | camera poses in the PLY frame |
| `<name>.ply` | binary PLY with `x y z` (+ `red green blue`), **Z-up, metres**, in the same room frame | registration of the two recordings |

Valid depth is 0.25–5.0 m; zero means no measurement.

A camera pose in the PLY frame is

```text
T_ply_camera = Q_yup_to_zup · A_alignment · T_arkit_camera · diag(1, −1, −1, 1)
```

where `Q_yup_to_zup` maps (x, y, z) → (x, −z, y). The last factor converts
OpenCV camera axes (right, down, forward) to ARKit's (right, up, backward). The
alignment stage checks this formula by back-projecting depth into each
recording's own PLY.

## Check your data

```bash
navdelta check-data --data path/to/data          # or ./run.sh check-data
```

It lists the two recordings it found, their keyframe counts, and anything
missing: folders, per-frame files, camera keys, the 16-number transform.

## Recording tips

- Record the same area twice, starting from roughly the same place, and cover
  the parts that changed from more than one direction. Only space that **both**
  recordings observed can be compared.
- Smallest change the pipeline can detect: about a 15 cm object, set by the
  5 cm voxels and the 10 cm alignment tolerance.
- Thin or see-through materials (sheer curtains, glass) can read differently
  in depth between recordings. The pipeline then reports a **depth-only**
  difference if the photos show the same object.
