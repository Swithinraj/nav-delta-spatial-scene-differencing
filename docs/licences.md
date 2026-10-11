# Models, code and licences

Every learned model is used as published: nothing is trained, fine-tuned or labelled by hand.
Weights download automatically on first use, at pinned revisions, into the model cache
(default `./.hf_cache`, `/models` in Docker).

| Model / code | Used by | Licence | Pinned |
|---|---|---|---|
| [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-base-plus) (hiera-base-plus) | all four | Apache-2.0 | `b7320756` |
| [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | all four | MIT | `4271c66b` |
| [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384) (so400m) | all four | Apache-2.0 | `e8e48729` |
| [SmolVLM2-500M](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | voxel_diff (colour words) | Apache-2.0 | `7b375e1b` |
| [MASt3R](https://github.com/naver/mast3r) code + [weights](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric), with [DUSt3R](https://github.com/naver/dust3r) and [CroCo](https://github.com/naver/croco) | scene_graph, loop_outliers, rgbd_fusion (shared reconstruction; rgbd_fusion uses only its matches) | **CC BY-NC-SA 4.0 (non-commercial)** | code `f5209afc` / `3cc8c88c` / `d7de0705` |
| ScanNet200 class names (a list of 200 words) | all four | from the [ScanNet200 benchmark](https://github.com/ScanNet/ScanNet) | file in `navdelta/approaches/voxel_diff/resources/` |

**Consequence:** approach 1 (`voxel_diff`) uses only permissively licensed models. Approaches 2,
3 and 4 depend on MASt3R and are therefore usable for **non-commercial purposes only**. MASt3R's code
is not vendored in this repository: it is downloaded from GitHub at the pinned commits on first use
([`navdelta/recon/mast3r_src.py`](../navdelta/recon/mast3r_src.py)).

Python libraries (PyTorch, Open3D, OpenCV, SciPy, NumPy, transformers, ...) are open source; exact
versions are in [`requirements.txt`](../requirements.txt).

## Data

| Data | Used for | Licence / terms |
|---|---|---|
| our two room recordings (Polycam raw-data exports) | development and the room results | captured for this project; downloadable from the link in the README, not stored in the repository |
| [ChangeSim](https://github.com/SAMMiCA/ChangeSim) (Park et al., IROS 2021) | the warehouse benchmark only | repository under MIT; the data is read from the official archives by `benchmarks/prepare.py` and not redistributed |
