#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ULIP-2
source env/ulip/bin/activate
export ULIP_OPENCLIP_CHECKPOINT=/workspace/ULIP-2/checkpoints/open_clip_pytorch_model.bin
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
cd repo
python -u main.py \
 --model ULIP2_PointBERT_Colored --npoints 10000 \
 --output-dir /workspace/ULIP-2/outputs/modelnet40 \
 --evaluate_3d_ulip2 --validate_dataset_name modelnet40 \
 --batch-size 8 --workers 4 \
 --test_ckpt_addr /workspace/ULIP-2/checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
