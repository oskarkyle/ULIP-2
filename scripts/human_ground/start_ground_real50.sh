#!/usr/bin/env bash
set -euo pipefail
base=/media/robosense/DataCenterNew/Kyle/ULIP-2
docker run --rm --name ulip-ground-real50 --gpus device=1 \
  --memory=32g --memory-swap=32g --cpus=4 --shm-size=1g \
  --user "$(id -u):$(id -g)" -v "$base:/workspace/ULIP-2" -w /workspace/ULIP-2 \
  -e HOME=/workspace/ULIP-2/cache/home -e OMP_NUM_THREADS=4 \
  --entrypoint bash nvcr.io/nvidia/pytorch:22.06-py3 -c \
  'export LD_LIBRARY_PATH="/workspace/ULIP-2/env/sysroot/usr/lib/x86_64-linux-gnu:/workspace/ULIP-2/env/sysroot/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"; source env/ulip/bin/activate; exec python outputs/ulip_human_ground_v1/evaluate_ground_real50.py'
