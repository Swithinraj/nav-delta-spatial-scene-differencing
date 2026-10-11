#!/usr/bin/env bash
# One entry point (Linux / WSL): builds the image if needed and runs navdelta in Docker.
#   ./run.sh                     approach 1 (voxel_diff): full pipeline on ./data -> ./outputs
#   ./run.sh scene_graph         approach 2; ./run.sh loop_outliers: approach 3; ./run.sh rgbd_fusion: approach 4
#   ./run.sh verify              run twice, require identical results
#   ./run.sh check-data          check the recordings
#   ./run.sh run --cpu           force the CPU      (an approach name may come first: ./run.sh loop_outliers verify)
# Folders: NAVDELTA_DATA_DIR (default ./data), NAVDELTA_OUT_DIR (./outputs), NAVDELTA_MODEL_DIR (./.hf_cache)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "${NAVDELTA_OUT_DIR:-./outputs}" "${NAVDELTA_MODEL_DIR:-./.hf_cache}"

approach=voxel_diff
case "${1:-}" in voxel_diff|scene_graph|loop_outliers|rgbd_fusion) approach=$1; shift ;; esac
service="${approach}-cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && [[ " $* " != *" --cpu "* ]]; then
  service="${approach}"
fi
docker compose build "${service}"
if [[ $# -eq 0 ]]; then set -- run; fi
exec docker compose run --rm "${service}" "$@"
