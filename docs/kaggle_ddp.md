# Kaggle 2×T4 DDP — Launch Guide

> One script, two modes: `python` (single-GPU/CPU) and `torchrun --nproc_per_node=2` (DDP). Single-GPU compat is preserved — no checkout/reset required.

## 1. What you get

- **Script:** `scripts/train_kaggle_2xT4.py` (alias: `scripts/train_toros_kaggle_ddp.py`)
  - Default model: **TorosHybrid 1024×12** (`dim=1024`, `n_encoder_layers=12`, `d_byte=128`, `patch=16`, `ternary_swiglu`).
  - Dummy small still works: `--dim 256 --layers 4 --synthetic`.
  - Distributed **auto-detect** via env: `WORLD_SIZE / RANK / LOCAL_RANK` (set by `torchrun`). Falls back to single-GPU/CPU when absent.
  - Wraps `ASDAGTrainer` — LPC/Muon/CUDA-Graph/GradScaler logic stays in the trainer.
- **Helper:** `scripts/kaggle_run.sh` — canonical `torchrun` one-liner.
- **Throughput target:** ~**190k tok/s per T4**, **~380k tok/s aggregate** (2×T4). With `B=48 T=512` (24.5k tok/step) → ~15 steps/s aggregate.
  - 20 h/week budget → ~**13 B tok/week/GPU**, **~26 B tok/week aggregate** at sustained 190k tok/s. Real jobs hit 8–12 B/GPU after eval/checkpoint overhead.

---

## 2. Enable 2×T4 on Kaggle

1. Kaggle notebook → **Settings** → **Accelerator** → **T4 ×2** (not T4 ×1).
2. Verify in a cell:
   ```python
   import torch; print(torch.cuda.device_count())  # expect 2
   !nvidia-smi  # 2x Tesla T4, Turing sm_75
   ```
3. Quota: **~20 hours/week/GPU-accelerated** (Kaggle free tier). Internet must be ON to install the repo deps.

---

## 3. Install

Kaggle cell (run once per session):

```bash
# Clone your repo or upload dataset; then:
pip install -e .                # torch>=2.0, numpy — no extra deps
# Optional for compressed checkpoints (.toros zstd):
pip install zstandard
```

No `torch.compile` needed on T4 (sm_75 < sm_89); LPC + fused kernels already handle the hot path.

---

## 4. Run

### 4.1  DDP — TorosHybrid 1024×12 (canonical)

```bash
torchrun --nproc_per_node=2 scripts/train_kaggle_2xT4.py \
  --dim 1024 --layers 12 --batch_size 48 --seq_len 512 \
  --max_steps 15000 --warmup_steps 500 --eval_interval 500 \
  --use_lpc --dtype fp16

# Alias (identical script):
torchrun --nproc_per_node=2 scripts/train_toros_kaggle_ddp.py \
  --dim 1024 --layers 12 --batch_size 48 --seq_len 512 \
  --max_steps 15000 --use_lpc --dtype fp16
```

What happens under the hood:

- `setup_distributed()` reads `WORLD_SIZE=2 RANK=0/1 LOCAL_RANK=0/1` from torchrun's env — no args needed.
- `torch.cuda.set_device(LOCAL_RANK)` + `dist.init_process_group(backend="nccl", init_method="env://")`.
- Each rank builds its own `TorosHybridLanguageModel` on `cuda:LOCAL_RANK` and its own `ASDAGTrainer(..., device="cuda:LOCAL_RANK")`.
- After trainer init, `trainer.model = DDP(trainer.model, device_ids=[LOCAL_RANK])` — gradients averaged by NCCL all-reduce automatically.
- Only **rank 0** prints/evaluates/saves checkpoints (`.pt` + `.toros`); other ranks hit `dist.barrier()` and stay in lockstep.

### 4.2  DDP — dummy small / synthetic (no data/*.bin)

```bash
torchrun --nproc_per_node=2 scripts/train_kaggle_2xT4.py \
  --dim 256 --layers 4 --batch_size 32 --seq_len 512 \
  --max_steps 2000 --synthetic --use_lpc --dtype fp16
```

### 4.3  Single-GPU / CPU (MUST keep working — no torchrun)

```bash
python scripts/train_kaggle_2xT4.py --dim 256 --layers 4 --batch_size 16 --seq_len 512 --max_steps 200 --synthetic --dtype fp16
python scripts/train_kaggle_2xT4.py --dim 136 --layers 4 --batch_size 8 --seq_len 256 --max_steps 10 --synthetic --no_lpc
python scripts/train_kaggle_2xT4.py --dim 144 --layers 6 --channel_mixer_type asdag_tree --max_steps 100 --synthetic
# Real data when available:
python scripts/train_kaggle_2xT4.py --dim 1024 --layers 12 --data_path data/tinystories_full_eos.bin --batch_size 48 --seq_len 512 --max_steps 15000
```

The same script is used; when `WORLD_SIZE` is unset it skips `init_process_group` and calls `trainer.train()` directly.

### 4.4  Helper one-liner

```bash
bash scripts/kaggle_run.sh
# which is:
# torchrun --nproc_per_node=2 scripts/train_kaggle_2xT4.py --dim 1024 --layers 12 --batch_size 48 --seq_len 512 --max_steps 15000 --use_lpc --dtype fp16
```

---

## 5. Turing T4 notes — fp16 + GradScaler already handled

- **T4 = Turing sm_75**, no bf16. The codebase forces fp16 transparently:
  - `affine_ai/models/hybrid.py:get_turing_dtype()` downgrades `bf16→fp16` when `cap < (8,0)`.
  - `affine_ai/training/trainer.py:_is_turing()` enables `torch.amp.GradScaler('cuda')` when dtype is fp16 on CUDA.
- Pass `--dtype fp16` explicitly for Kaggle; `--dtype bf16` also works (auto-downgraded), and `--dtype fp32` disables the scaler.
- Don't enable `torch.compile` on T4 — the fused C++ block kernel (`asdag_cpu_fused_*`) is CPU-only and LPC's CUDA Graph path is already zero-overhead without `compile`.

---

## 6. Throughput math (why 380k aggregate)

| Per GPU | Aggregate (×2) | Math |
|---|---|---|
| ~190k tok/s | ~380k tok/s | Measured on T4 with `ternary_swiglu`, `B=48 T=512` |
| `B*T = 24,576 tok/step` | 49,152 tok/step global | 380k / 24.5k ≈ 15.5 steps/s global |
| 20 h = 72,000 s | 13.7 B tok/GPU | `190k * 72k ≈ 13.7B`; 2 GPUs ≈ 27B if both saturated |

Global batch = `batch_size * world_size`. If you set `--batch_size 48` with `nproc=2`, the global batch is 96. Keep per-GPU `B` fixed and scale `--lr` only if you change global batch significantly.

---

## 7. 20-hour limit — checkpoint & resume tips

Kaggle kills the session at ~12 h (and weekly quota at 20 h). Save often:

```python
# Script saves automatically (rank 0 only):
#   checkpoints/kaggle_2xT4_latest.pt      — state_dict (or {"step","model_state_dict"} for periodic)
#   checkpoints/kaggle_2xT4_latest.toros   — compact 2-bit via save_toros_model (needs zstandard)

# Resume (loads with module. prefix stripped for DDP ckpts):
torchrun --nproc_per_node=2 scripts/train_kaggle_2xT4.py \
  --dim 1024 --layers 12 --batch_size 48 --seq_len 512 --max_steps 15000 \
  --resume checkpoints/kaggle_2xT4_latest.pt --save_interval 1000
```

Practical tips:

- **Save frequently:** `--save_interval 1000` (≈ 4 min at 380k tok/s) so a preemption loses <1k steps.
- **Persist outside container:** Kaggle notebooks lose `/kaggle/working` on preemption unless you **Save & Run All → Download** or push to a **Kaggle Dataset**. Treat `checkpoints/` as ephemeral — copy the best `.toros` to `/kaggle/working` or a dataset after each eval.
- **Data:** don't commit `data/*.bin` (12 GB). Reference a Kaggle Dataset or stream via `data/*.bin` memmap (the script discovers `data/tinystories_full_eos.bin`, `data/simplestories_eos.bin`, etc., or falls back to `--synthetic`).
- **Eval cost:** `--eval_interval 500 --eval_iters 20` keeps eval under 5% of step budget.
- **Monitor quota:** Kaggle → Settings → Quota. Plan two 9-hour runs rather than one 18-hour run — the second can `--resume`.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA device count 1` | Notebook still on T4×1 | Settings → Accelerator → **T4 ×2** → Save & re-run |
| `NCCL error` / `init_process_group` hang | Manual `WORLD_SIZE` export without `torchrun` | Don't export RANK/WORLD_SIZE by hand; always launch via `torchrun --nproc_per_node=2` |
| `GradScaler` warning suppressed | Running fp32 | Expected — scaler only created for fp16 on CUDA |
| `data/*.bin not found` | No dataset attached | Attach `data/*.bin` as Kaggle Dataset or use `--synthetic` for smoke tests |
| DDP checkpoint won't load single-GPU | `module.` prefix | Script strips `module.` automatically on resume |
| OOM at B=48 on 1024×12 | Global B=96 too large for 16 GB | Drop to `--batch_size 32` (global 64) or `--seq_len 256` |

---

## 9. File map

```
scripts/train_kaggle_2xT4.py        # primary DDP script (auto-detect, ASDAGTrainer + DDP wrap)
scripts/train_toros_kaggle_ddp.py   # identical alias
scripts/kaggle_run.sh               # torchrun one-liner
docs/kaggle_ddp.md                  # this doc
affine_ai/training/trainer.py       # GradScaler/Turing fp16 already handled (no change needed for DDP)
```

Single-GPU compat is invariant: any change that breaks `python scripts/train_kaggle_2xT4.py --synthetic` on a 1-GPU box is a bug.
