import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.language_model import ASDAGLanguageModel
from affine_ai.optim.muon import HybridMuonAdamW


def update_lr(optimizer, step, warmup_steps=50, max_steps=500, min_lr_ratio=0.1, base_muon_lr=0.03, base_adamw_lr=3e-3):
    if step < warmup_steps:
        ratio = float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        ratio = max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    if optimizer.muon_opt is not None:
        for g in optimizer.muon_opt.param_groups:
            g['lr'] = base_muon_lr * ratio
    if optimizer.adamw_opt is not None:
        for g in optimizer.adamw_opt.param_groups:
            g['lr'] = base_adamw_lr * ratio


def train():
    parser = argparse.ArgumentParser(description='Train ASDAG BLT on 100MB TinyStories on CPU')
    parser.add_argument('--data_path', type=str, default='data/tinystories_extracted.bin')
    parser.add_argument('--save_path', type=str, default='checkpoints/asdag_blt_tinystories_100mb.pt')
    parser.add_argument('--d_model', type=int, default=160)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_byte', type=int, default=64)
    parser.add_argument('--target_patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--eval_iters', type=int, default=10)
    parser.add_argument('--muon_lr', type=float, default=0.03)
    parser.add_argument('--adamw_lr', type=float, default=3e-3)
    args = parser.parse_args()

    print('=' * 64)
    print('  STARTING ASDAG BYTE LATENT TRANSFORMER (BLT) TRAINING')
    print('=' * 64)
    print(f'Dataset:               {args.data_path}')
    print(f'Model Architecture:    BLT (Byte Latent Transformer) + ASDAG')
    print(f'Dimensions:            d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}')
    print(f'Byte Encoder/Decoder:  d_byte={args.d_byte}, target_patch_size={args.target_patch_size}')
    print(f'Channel Mixer:         Ternary BitLinear SwiGLU (MatMul-Free)')
    print(f'Time Mixer:            Monarch GLA Associative Memory (Native C++ SIMD)')
    print(f'Batch Size & Context:  batch_size={args.batch_size}, seq_len={args.seq_len} ({args.batch_size * args.seq_len:,} bytes/step)')
    print(f'Total Steps:           {args.max_steps}')
    print(f'Device:                CPU (Native AVX2 / AVX-512 C++ Engine)')
    print('=' * 64)

    # 1. Load Data with 90% Train / 10% Val Split
    raw_data = np.fromfile(args.data_path, dtype=np.uint8)
    total_bytes = len(raw_data)
    split_idx = int(0.9 * total_bytes)
    train_data = torch.from_numpy(raw_data[:split_idx].astype(np.int64))
    val_data = torch.from_numpy(raw_data[split_idx:].astype(np.int64))
    print(f'Dataset Loaded: {total_bytes:,} total bytes')
    print(f'  Train Set (90%): {len(train_data):,} bytes')
    print(f'  Val Set   (10%): {len(val_data):,} bytes')

    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(0, len(d) - args.seq_len - 1, (args.batch_size,))
        x = torch.stack([d[i:i + args.seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + args.seq_len + 1] for i in ix])
        return x, y

    # 2. Build Model
    torch.manual_seed(42)
    model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_byte=args.d_byte,
        target_patch_size=args.target_patch_size,
        channel_mixer_type="ternary_swiglu",
        use_blt=True
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model Parameters:      {num_params:,}')

    # 3. Build Optimizer
    optimizer = HybridMuonAdamW(
        model=model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        muon_momentum=0.95,
        adamw_weight_decay=0.01
    )

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0.0
        for _ in range(args.eval_iters):
            x, y = get_batch("val")
            logits, loss, _ = model.blt(x, targets=y)
            total_loss += loss.item()
        val_loss = total_loss / args.eval_iters
        val_ppl = math.exp(min(val_loss, 20.0))
        model.train()
        return val_loss, val_ppl

    best_val_loss = float("inf")
    start_time = time.perf_counter()
    tokens_processed = 0

    print("\n--- Training Loop Started ---")
    for step in range(1, args.max_steps + 1):
        update_lr(optimizer, step, warmup_steps=50, max_steps=args.max_steps, base_muon_lr=args.muon_lr, base_adamw_lr=args.adamw_lr)
        step_t0 = time.perf_counter()
        xb, yb = get_batch("train")

        optimizer.zero_grad()
        logits, loss, stats = model.blt(xb, targets=yb, return_logits=False)
        loss.backward()
        optimizer.step()

        step_time = time.perf_counter() - step_t0
        tokens_processed += args.batch_size * args.seq_len
        current_speed = (args.batch_size * args.seq_len) / max(step_time, 1e-5)

        if step % 25 == 0 or step == 1:
            val_loss, val_ppl = evaluate()
            print(f"Step {step:4d}/{args.max_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | Speed: {current_speed:,.0f} bytes/s | Time: {time.perf_counter() - start_time:5.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "config": {
                        "d_model": args.d_model,
                        "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                        "d_byte": args.d_byte,
                        "target_patch_size": args.target_patch_size,
                        "channel_mixer_type": "ternary_swiglu",
                        "use_blt": True
                    }
                }, args.save_path)

        if step % args.eval_interval == 0 or step == args.max_steps:
            prompt_str = "Once upon a time, there was a little girl named Lily."
            prompt_bytes = torch.tensor([[ord(c) for c in prompt_str]], dtype=torch.long)
            gen_bytes = model.generate(prompt_bytes, max_new_tokens=80, temperature=0.7)
            gen_text = bytes(gen_bytes[0].tolist()).decode("utf-8", errors="ignore")
            print("-" * 64)
            print(f"🌟 GENERATION @ Step {step}: '{gen_text}'")
            print("-" * 64)

    total_training_time = time.perf_counter() - start_time
    avg_speed = tokens_processed / total_training_time
    print("\n" + "=" * 64)
    print("  TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 64)
    print(f"Total Time:            {total_training_time:.1f} seconds")
    print(f"Average Speed:         {avg_speed:,.0f} training bytes / second 🚀")
    print(f"Best Val Loss:         {best_val_loss:.4f}")
    print(f"Saved Checkpoint:      {args.save_path}")
    print("=" * 64)


if __name__ == "__main__":
    train()
