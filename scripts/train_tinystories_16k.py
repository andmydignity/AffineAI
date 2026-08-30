import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from affine_ai.models.language_model import ASDAGLanguageModel
from affine_ai.optim.muon import HybridMuonAdamW


def parse_args():
    parser = argparse.ArgumentParser(description="ASDAG 16K Context BLT Training on TinyStories")
    parser.add_argument('--data_path', type=str, default='data/tinystories_eos.bin')
    parser.add_argument('--save_path', type=str, default='checkpoints/asdag_blt_tinystories_16k.pt')
    parser.add_argument('--d_model', type=int, default=160)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_byte', type=int, default=64)
    parser.add_argument('--target_patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=16384)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=20)
    parser.add_argument('--eval_iters', type=int, default=5)
    parser.add_argument('--muon_lr', type=float, default=0.03)
    parser.add_argument('--adamw_lr', type=float, default=3e-3)
    return parser.parse_args()


def update_lr(optimizer, step, warmup_steps=20, max_steps=400, min_lr_ratio=0.1, base_muon_lr=0.03, base_adamw_lr=3e-3):
    if step < warmup_steps:
        ratio = float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        ratio = max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    if hasattr(optimizer, 'muon_opt') and optimizer.muon_opt is not None:
        for g in optimizer.muon_opt.param_groups:
            g['lr'] = base_muon_lr * ratio
    if hasattr(optimizer, 'adamw_opt') and optimizer.adamw_opt is not None:
        for g in optimizer.adamw_opt.param_groups:
            g['lr'] = base_adamw_lr * ratio


def main():
    args = parse_args()

    print('=' * 75)
    print('   STARTING 16K CONTEXT ASDAG BYTE LATENT TRANSFORMER (BLT) TRAINING')
    print('=' * 75)
    print(f'Dataset:               {args.data_path}')
    print(f'Model Architecture:    BLT + ASDAG (Ternary SwiGLU + Monarch GLA)')
    print(f'Dimensions:            d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}')
    print(f'Super-Patch Size:      P = {args.target_patch_size} ({args.seq_len // args.target_patch_size:,} patches/sequence)')
    print(f'Context Window:        seq_len = {args.seq_len:,} bytes (16K Context)')
    print(f'Batch Size:            batch_size = {args.batch_size} ({args.batch_size * args.seq_len:,} bytes/step)')
    print(f'Total Steps:           {args.max_steps}')
    print(f'EOS Terminator:        ASCII \0 (Byte 0)')
    print(f'Device:                CPU (Native AVX2 / AVX-512 Engine, Core-Pinned)')
    print('=' * 75)

    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found!")
        return

    raw_data = np.fromfile(args.data_path, dtype=np.uint8)
    n_total = len(raw_data)
    n_train = int(0.9 * n_total)
    train_data = torch.from_numpy(raw_data[:n_train].astype(np.int64))
    val_data = torch.from_numpy(raw_data[n_train:].astype(np.int64))

    print(f'Dataset Loaded:        {n_total:,} total bytes')
    print(f'  Train Set (90%):     {n_train:,} bytes')
    print(f'  Val Set   (10%):     {n_total - n_train:,} bytes')

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        ix = torch.randint(0, len(data) - args.seq_len - 1, (args.batch_size,))
        x = torch.stack([data[i:i + args.seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + args.seq_len] for i in ix])
        return x, y

    model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_byte=args.d_byte,
        target_patch_size=args.target_patch_size,
        channel_mixer_type="ternary_swiglu",
        use_blt=True
    ).to("cpu")

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model Parameters:      {num_params:,}')

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
            logits, loss, _ = model.blt(x, targets=y, return_logits=False)
            total_loss += loss.item()
        val_loss = total_loss / args.eval_iters
        val_ppl = math.exp(min(val_loss, 20.0))
        model.train()
        return val_loss, val_ppl

    best_val_loss = float("inf")
    start_step = 1
    if os.path.exists(args.save_path):
        try:
            ckpt = torch.load(args.save_path, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Resumed from checkpoint: {args.save_path} @ Step {start_step - 1} (Best Val Loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"Could not load checkpoint ({e}), starting fresh.")

    start_time = time.perf_counter()

    print("\n--- Training Loop Started ---")
    for step in range(start_step, args.max_steps + 1):
        update_lr(optimizer, step, warmup_steps=10, max_steps=args.max_steps, base_muon_lr=args.muon_lr, base_adamw_lr=args.adamw_lr)
        step_t0 = time.perf_counter()
        xb, yb = get_batch("train")

        optimizer.zero_grad()
        logits, loss, stats = model.blt(xb, targets=yb, return_logits=False)
        loss.backward()
        optimizer.step()

        step_time = time.perf_counter() - step_t0
        current_speed = (args.batch_size * args.seq_len) / max(step_time, 1e-5)

        if step % args.eval_interval == 0 or step == 1:
            val_loss, val_ppl = evaluate()
            print(f"Step {step:4d}/{args.max_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | Speed: {current_speed:,.0f} B/s | Time: {time.perf_counter() - start_time:5.1f}s", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                }, args.save_path)

            # Fast Sample Generation with EOS stopping
            prompt = "Once upon a time, there was a little girl named Lily."
            prompt_bytes = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long)
            gen = model.generate(prompt_bytes, max_new_tokens=40, temperature=0.7, eos_byte=0)
            gen_bytes = gen[0].tolist()
            gen_text = bytes([b for b in gen_bytes if 0 < b < 256]).decode('utf-8', errors='ignore')
            hit_eos = 0 in gen_bytes
            print(f'  [Sample @ Step {step} (EOS Hit: {hit_eos})]: \"{gen_text}\"\n', flush=True)

    total_time = time.perf_counter() - start_time
    executed_steps = max(args.max_steps - start_step + 1, 1)
    total_tokens = executed_steps * args.batch_size * args.seq_len
    avg_speed = total_tokens / max(total_time, 1e-5)

    print('=' * 75)
    print('  16K CONTEXT TRAINING COMPLETED SUCCESSFULLY!')
    print('=' * 75)
    print(f'Total Time:            {total_time:.1f} seconds')
    print(f'Total Tokens Processed:{total_tokens:,} bytes')
    print(f'Average Speed:         {avg_speed:,.0f} training bytes/second 🚀')
    print(f'Best Val Loss:         {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 20.0)):.2f})')
    print(f'Saved Checkpoint:      {args.save_path}')
    print('=' * 75)


if __name__ == '__main__':
    main()
