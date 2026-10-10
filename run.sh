#!/usr/bin/env bash
# One entry point (Linux / WSL): builds the image if needed and runs navdelta in Docker.
#   ./run.sh                     full pipeline on ./data -> ./outputs
#   ./run.sh verify              run twice, require identical results
#   ./run.sh check-data          check the recordings
#   ./run.sh run --cpu           force the CPU
# Folders: NAVDELTA_DATA_DIR (default ./data), NAVDELTA_OUT_DIR (./outputs), NAVDELTA_MODEL_DIR (./.hf_cache)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "${NAVDELTA_OUT_DIR:-./outputs}" "${NAVDELTA_MODEL_DIR:-./.hf_cache}"

service=navdelta-cpu
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && [[ " $* " != *" --cpu "* ]]; then
  service=navdelta
fi
docker compose build "${service}"
if [[ $# -eq 0 ]]; then set -- run; fi
exec docker compose run --rm "${service}" "$@"
