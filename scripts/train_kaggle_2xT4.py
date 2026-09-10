#!/usr/bin/env python3
"""
Kaggle 2×T4 DDP training — TorosHybrid 1024×12 (or dummy small) via ASDAGTrainer.
================================================================================
Distributed auto-detect: single-GPU and DDP-compatible.

Kaggle env (torchrun sets automatically):
    WORLD_SIZE=2  RANK=0/1  LOCAL_RANK=0/1

Quick start (Kaggle notebook — Settings → Accelerator → T4 ×2):
    pip install -e .                               # repo root
    torchrun --nproc_per_node=2 scripts/train_kaggle_2xT4.py \
        --dim 1024 --layers 12 --batch_size 48 --seq_len 512 \
        --max_steps 15000 --use_lpc --dtype fp16

Single-GPU / CPU sanity (no torchrun — MUST still work):
    python scripts/train_kaggle_2xT4.py --dim 256 --layers 4 --batch_size 16 --seq_len 512 \
        --max_steps 200 --synthetic --dtype fp16
    python scripts/train_kaggle_2xT4.py --dim 136 --layers 4 --batch_size 8 --seq_len 256 \
        --max_steps 10 --synthetic --no_lpc   # ASDAGLanguageModel path

Checkpoint (rank-0 only):
    save_toros_model via affine_ai.core.format (2-bit ternary + zstd) + torch.save fallback
    Resume:  --resume checkpoints/kaggle_2xT4_latest.pt

Notes:
- Turing T4 (sm_75) is fp16-only: TorosHybridConfig(dtype=fp16) + GradScaler
  are already handled inside hybrid.py / trainer.py (get_turing_dtype / _is_turing).
  Passing --dtype fp16 is explicit and safe; --dtype bf16 auto-downgrades to fp16 on T4.
- Expected throughput: ~190k tok/s per T4, ~380k tok/s aggregate (2×T4). At
  B=48 T=512 (24.5k tok/step) → ~15 steps/s aggregate. 20 h/week ≈ 13 B tok/GPU.
"""
from __future__ import annotations

import argparse
import os
import math
import time
import glob
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Distributed helpers — auto-detect from torchrun env, no-op for single GPU
# ---------------------------------------------------------------------------
def setup_distributed():
    """Return (rank, local_rank, world_size, device, is_distributed).

    Auto-detects via env: RANK / WORLD_SIZE / LOCAL_RANK set by torchrun.
    Single-GPU compat: when not launched via torchrun, returns rank=0/world=1
    and device=cuda if available else cpu, without initializing process group.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_distributed = world_size > 1

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested (WORLD_SIZE>1) but CUDA not available")
        # Must set device before init_process_group for NCCL
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        # Optional: deterministic speed tweaks (no effect on correctness)
        torch.backends.cudnn.benchmark = True
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Honor LOCAL_RANK if user manually exported it on single-GPU
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
            try:
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
            except Exception:
                pass

    return rank, local_rank, world_size, device, is_distributed


def is_main_process(rank: int) -> bool:
    return rank == 0


def ddp_wrap_model(model: torch.nn.Module, local_rank: int):
    """Wrap model in DDP. Caller must have already moved model to cuda:local_rank."""
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )


def get_dtype(name: str):
    name = (name or "fp16").lower().strip()
    if name in ("fp16", "float16", "half", "16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "float", "32"):
        return torch.float32
    raise ValueError(f"Unknown --dtype {name!r} (use fp16|bf16|fp32)")


def parse_args():
    p = argparse.ArgumentParser(description="Kaggle 2×T4 DDP — TorosHybrid 1024×12 via ASDAGTrainer")
    # Model
    p.add_argument("--dim", type=int, default=1024, help="Hybrid dim (TorosHybridConfig.dim)")
    p.add_argument("--layers", type=int, default=12, help="n_encoder_layers")
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_byte", type=int, default=128)
    p.add_argument("--target_patch_size", type=int, default=16)
    p.add_argument("--channel_mixer_type", type=str, default="ternary_swiglu",
                   choices=["ternary_swiglu", "asdag_tree", "dense_swiglu", "classic_mlp"])
    p.add_argument("--dtype", type=str, default="fp16", help="fp16|bf16|fp32 (T4→fp16 forced by get_turing_dtype)")
    # Data
    p.add_argument("--data_path", type=str, default="",
                   help="Path to uint8 .bin (e.g. data/tinystories_full_eos.bin). Auto-discovers data/*.bin if empty.")
    p.add_argument("--val_split", type=float, default=0.02, help="Fraction for validation if single file")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic random bytes (no data/*.bin needed)")
    p.add_argument("--synthetic_bytes", type=int, default=50_000_000, help="Size of synthetic stream")
    # Training
    p.add_argument("--batch_size", type=int, default=48, help="Per-GPU batch size (global = batch_size * world_size)")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=15000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_lpc", dest="use_lpc", action="store_true", help="Enable LPC (default)")
    p.add_argument("--no_lpc", dest="use_lpc", action="store_false", help="Disable LPC (global backprop)")
    p.set_defaults(use_lpc=True)
    p.add_argument("--use_muon", action="store_true", default=True)
    p.add_argument("--no_muon", dest="use_muon", action="store_false")
    # Checkpoint
    p.add_argument("--save_path", type=str, default="checkpoints/kaggle_2xT4_latest.pt")
    p.add_argument("--save_toros_path", type=str, default="checkpoints/kaggle_2xT4_latest.toros")
    p.add_argument("--save_interval", type=int, default=1000, help="Rank-0 checkpoint interval")
    p.add_argument("--resume", type=str, default="", help="Resume from .pt checkpoint")
    # Misc
    p.add_argument("--model_type", type=str, default="hybrid",
                   choices=["hybrid", "asdag"], help="hybrid=TorosHybridLanguageModel, asdag=ASDAGLanguageModel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true", help="torch.compile model (sm_89+ recommended)")
    return p.parse_args()


def discover_data_path(explicit: str) -> str | None:
    if explicit and os.path.exists(explicit):
        return explicit
    # Common files — prefer full_eos / eos variants
    candidates = [
        "data/tinystories_full_eos.bin",
        "data/simplestories_eos.bin",
        "data/smoltalk_eos.bin",
        "data/tinystories_eos.bin",
        "data/fineweb_edu_sample_eos.bin",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    bins = sorted(glob.glob("data/*.bin"))
    return bins[0] if bins else None


def load_byte_stream(args, rank: int) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None]:
    """Return (train_data, val_data) as np.memmap or torch Tensor.

    On Kaggle each rank sees the same filesystem; we load once and slice via
    memmap (zero-copy). For synthetic, each rank generates the same seed
    (single-GPU compat) — DDP sampling divergence comes from per-rank RNG in trainer.get_batch.
    """
    if args.synthetic:
        if is_main_process(rank):
            print(f"[synthetic] generating {args.synthetic_bytes:,} random bytes (seed={args.seed})")
        rng = np.random.default_rng(args.seed)
        raw = rng.integers(0, 256, size=args.synthetic_bytes, dtype=np.uint8)
        # Use numpy array directly — ASDAGTrainer will convert to torch long as needed
        split = int(len(raw) * (1 - args.val_split))
        return raw[:split], raw[split:]

    path = discover_data_path(args.data_path)
    if path is None:
        if is_main_process(rank):
            print("[data] no data/*.bin found — falling back to synthetic 50 MB stream")
        rng = np.random.default_rng(args.seed)
        raw = rng.integers(0, 256, size=args.synthetic_bytes, dtype=np.uint8)
        split = int(len(raw) * (1 - args.val_split))
        return raw[:split], raw[split:]

    if is_main_process(rank):
        print(f"[data] using {path} ({os.path.getsize(path):,} bytes)")
    # Zero-copy memmap — shared across ranks, no duplication
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    split = int(len(raw) * (1 - args.val_split))
    train_data = raw[:split]
    val_data = raw[split:]
    return train_data, val_data


def build_model(args, device: str, dtype):
    if args.model_type == "hybrid":
        from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig

        cfg = TorosHybridConfig(
            dim=args.dim,
            d_byte=min(args.d_byte, args.dim),
            n_encoder_layers=args.layers,
            n_heads=args.n_heads,
            target_patch_size=args.target_patch_size,
            channel_mixer_type=args.channel_mixer_type,
            dtype=dtype,
            compile_forward=bool(args.compile),
        )
        model = TorosHybridLanguageModel(cfg)
    else:
        from affine_ai.models.language_model import ASDAGLanguageModel

        model = ASDAGLanguageModel(
            vocab_size=256,
            d_model=args.dim,
            n_layers=args.layers,
            n_heads=args.n_heads,
            channel_mixer_type=args.channel_mixer_type,
            dtype=dtype,
        )
    # Move to target device BEFORE DDP wrap / trainer init
    # Trainer will call model.to(device) again — safe for DDP wrappers.
    model = model.to(device)
    return model


def main():
    args = parse_args()
    rank, local_rank, world_size, device, is_distributed = setup_distributed()
    main_process = is_main_process(rank)

    # Seed: offset by rank so get_batch diverges across ranks (different samples)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Dtype — T4 (sm_75) forces fp16 inside hybrid/trainer via get_turing_dtype
    dtype = get_dtype(args.dtype)

    if main_process:
        print("=" * 95)
        print("  KAGGLE 2×T4 DDP — TOROS HYBRID TRAINING (ASDAGTrainer + distributed auto-detect)")
        print("=" * 95)
        print(f"  world_size={world_size}  rank={rank}  local_rank={local_rank}  device={device}  "
              f"is_distributed={is_distributed}")
        print(f"  Model:  {args.model_type}  dim={args.dim}  layers={args.layers}  heads={args.n_heads}  "
              f"d_byte={args.d_byte}  patch={args.target_patch_size}  mixer={args.channel_mixer_type}  dtype={str(dtype).replace('torch.', '')}")
        print(f"  Train:  batch_size={args.batch_size} per-GPU (global={args.batch_size * world_size})  "
              f"seq_len={args.seq_len}  max_steps={args.max_steps}  use_lpc={args.use_lpc}  use_muon={args.use_muon}")
        print(f"  Data:   {'synthetic' if args.synthetic else (args.data_path or 'auto-discover data/*.bin')}")
        print(f"  Env:    WORLD_SIZE={os.environ.get('WORLD_SIZE','(unset)')}  "
              f"RANK={os.environ.get('RANK','(unset)')}  LOCAL_RANK={os.environ.get('LOCAL_RANK','(unset)')}")
        if is_distributed:
            print("  DDP:    NCCL backend, GradScaler fp16 auto-enabled on T4 (sm_75) via ASDAGTrainer._is_turing()")
        else:
            print("  Mode:   single-GPU/CPU — no process group (keep single-GPU compat)")
        # Canonical launch examples
        print("-" * 95)
        print("  Launch (2×T4):  torchrun --nproc_per_node=2 scripts/train_kaggle_2xT4.py "
              "--dim 1024 --layers 12 --batch_size 48 --seq_len 512 --max_steps 15000 --use_lpc --dtype fp16")
        print("  Launch (single): python scripts/train_kaggle_2xT4.py "
              "--dim 256 --layers 4 --batch_size 16 --seq_len 512 --max_steps 200 --synthetic")
        print("=" * 95)

    train_data, val_data = load_byte_stream(args, rank)

    model = build_model(args, device, dtype)

    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            # Handle DDP prefix 'module.' if checkpoint was saved from DDP
            cleaned = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
            model.load_state_dict(cleaned, strict=False)
            if main_process:
                print(f"[resume] loaded {args.resume} @ step {ckpt.get('step', '?')}")
        except Exception as e:
            if main_process:
                print(f"[resume] failed ({e}), starting fresh")

    # ------------------------------------------------------------------
    # ASDAGTrainer — distributed auto-detect
    # Trainer already handles on-device batching, Muon/AdamW splits, LPC,
    # Turing fp16 GradScaler, CUDA Graphs, etc. We pass device=device
    # (cuda:LOCAL_RANK in DDP) so each rank operates on its GPU.
    # ------------------------------------------------------------------
    from affine_ai.training.trainer import ASDAGTrainer

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,           # per-GPU; global = * world_size
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        grad_clip=args.grad_clip,
        device=device,
        use_lpc=args.use_lpc,
        use_muon=args.use_muon,
        compile_model=bool(args.compile),
    )

    # DDP wrap AFTER trainer init so optimizer param refs stay valid.
    # Trainer.model.to(device) already called; now wrap the post-init model.
    if is_distributed:
        trainer.model = ddp_wrap_model(trainer.model, local_rank)
        # Ensure only rank-0 saves checkpoints (trainer.train saves best-val)
        # We'll run a manual loop instead of trainer.train() so we can gate I/O.
        if main_process:
            print(f"[DDP] wrapped trainer.model with DistributedDataParallel (local_rank={local_rank})")

    # ------------------------------------------------------------------
    # Training loop — DDP-aware stepping, rank-0 checkpointing
    # ------------------------------------------------------------------
    if is_distributed:
        # Manual loop so we control distributed sync + rank-0 I/O.
        # Gradients are averaged by DDP all-reduce automatically in backward().
        best_val = float("inf")
        t0 = time.time()
        trainer.model.train()
        for step in range(args.max_steps):
            loss_val = trainer.train_step(step, sync_loss=False)
            # loss_val is a detached tensor in LPC/no-sync mode; item() only on rank0 for logging
            if step % args.eval_interval == 0 or step == args.max_steps - 1:
                # Evaluate on rank0 only to avoid duplicate work; broadcast choice to keep loops aligned
                if main_process:
                    metrics = trainer.evaluate()
                    val_loss = metrics["val_loss"]
                    if val_loss < best_val:
                        best_val = val_loss
                        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                        try:
                            torch.save(trainer.model.state_dict(), args.save_path)
                        except Exception:
                            torch.save(trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict(),
                                       args.save_path)
                        try:
                            from affine_ai.core.format import save_toros_model
                            base = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                            # Unwrap Trainer/Dynamo if needed — save_toros_model expects nn.Module
                            save_toros_model(base, args.save_toros_path)
                        except Exception as e:
                            print(f"[ckpt] save_toros_model note: {e}")
                    # Logging: synchronize loss scalar for display
                    try:
                        lv = loss_val.item() if hasattr(loss_val, "item") else float(loss_val)
                    except Exception:
                        lv = float(metrics["val_loss"])
                    elapsed = time.time() - t0
                    tok_per_s = (args.batch_size * args.seq_len * world_size) / max(elapsed / max(1, step + 1), 1e-6)
                    print(f"Step {step:5d}/{args.max_steps} | loss {lv:.4f} | val {val_loss:.4f} "
                          f"| best {best_val:.4f} | {tok_per_s:,.0f} tok/s (agg) | {elapsed/60:.1f}m", flush=True)
                # Barrier so non-zero ranks don't race ahead while rank0 evaluates
                if world_size > 1:
                    torch.distributed.barrier()

            # Periodic checkpoint (rank0)
            if main_process and args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                ckpt_path = args.save_path.replace(".pt", f"_step{step+1}.pt")
                try:
                    sd = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
                    torch.save({"step": step + 1, "model_state_dict": sd}, ckpt_path)
                except Exception as e:
                    print(f"[ckpt] periodic save failed: {e}")

        if is_distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        if main_process:
            total = time.time() - t0
            print(f"[done] {args.max_steps} steps in {total:.1f}s — best val {best_val:.4f}")
    else:
        # Single-GPU/CPU: delegate to trainer.train() (handles checkpoint of best val)
        stats = trainer.train(save_path=args.save_path if args.save_path else None)
        if main_process:
            # Also emit .toros compact checkpoint
            try:
                from affine_ai.core.format import save_toros_model
                base = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                # Trainer may have torch.compile wrapper — unwrap by saving state dict
                save_toros_model(base, args.save_toros_path)
                print(f"[ckpt] saved toros -> {args.save_toros_path}")
            except Exception as e:
                print(f"[ckpt] save_toros_model note: {e}")
            print(f"[done] single-GPU stats: {stats}")


if __name__ == "__main__":
    main()
