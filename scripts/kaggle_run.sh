#!/usr/bin/env bash
# Kaggle 2×T4 DDP one-liner — launch via torchrun (WORLD_SIZE=2 RANK=0/1 LOCAL_RANK=0/1 auto)
# Single-GPU fallback (keep compat): python scripts/train_kaggle_2xT4.py --dim 256 --layers 4 --synthetic --batch_size 16 --seq_len 512 --max_steps 200
set -e
torchrun --nproc_per_node=2 scripts/train_kaggle_2xT4.py --dim 1024 --layers 12 --batch_size 48 --seq_len 512 --max_steps 15000 --use_lpc --dtype fp16 "$@"
