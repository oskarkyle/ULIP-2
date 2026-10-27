#!/usr/bin/env bash
set -euo pipefail
base=/media/robosense/DataCenterNew/Kyle/ULIP-2
docker run --rm --gpus "device=${ULIP_GPU:-0}" --network host --shm-size=2g --tmpfs /tmp:rw,size=256m,mode=1777 --user "$(id -u):$(id -g)" \
 -v "$base:/workspace/ULIP-2" -w /workspace/ULIP-2 \
 -e HOME=/workspace/ULIP-2/cache/home \
 -e PIP_CONFIG_FILE=/dev/null -e PIP_EXTRA_INDEX_URL= -e PIP_INDEX_URL=https://pypi.org/simple \
 -e PIP_CACHE_DIR=/workspace/ULIP-2/cache/pip \
 -e TMPDIR=/workspace/ULIP-2/cache/tmp \
 -e HF_HOME=/workspace/ULIP-2/cache/huggingface \
 -e TORCH_HOME=/workspace/ULIP-2/cache/torch \
 -e TORCH_EXTENSIONS_DIR=/workspace/ULIP-2/cache/torch_extensions \
 -e TORCH_CUDA_ARCH_LIST=8.6 -e MAX_JOBS=4 -e PYTHONNOUSERSITE=1 \
 --entrypoint bash nvcr.io/nvidia/pytorch:22.06-py3 -c 'export LD_LIBRARY_PATH="/workspace/ULIP-2/env/sysroot/usr/lib/x86_64-linux-gnu:/workspace/ULIP-2/env/sysroot/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"; exec bash "$@"' bash "$@"
