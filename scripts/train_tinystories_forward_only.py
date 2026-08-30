import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.language_model import ASDAGLanguageModel


def train():
    parser = argparse.ArgumentParser(description='Train 700k ASDAG BLT using Forward-Only Local Plasticity on 100MB TinyStories')
    parser.add_argument('--data_path', type=str, default='data/tinystories_extracted.bin')
    parser.add_argument('--d_model', type=int, default=160)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_byte', type=int, default=64)
    parser.add_argument('--target_patch_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--eval_iters', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    args = parser.parse_args()

    print('=' * 72)
    print('  TRAINING 700K ASDAG BLT VIA FORWARD-ONLY PREDICTIVE PLASTICITY')
    print('=' * 72)
    print(f'Dataset:               {args.data_path}')
    print(f'Model Parameters:      701,105')
    print(f'Training Mode:         Forward-Only (0 Global Autograd Backward Pass)')
    print(f'Batch Size & Context:  batch_size={args.batch_size}, seq_len={args.seq_len} ({args.batch_size * args.seq_len:,} bytes/step)')
    print(f'Max Steps:             {args.max_steps}')
    print('=' * 72)

    # 1. Load Data with 90% Train / 10% Val Split
    raw_data = np.fromfile(args.data_path, dtype=np.uint8)
    total_bytes = len(raw_data)
    split_idx = int(0.9 * total_bytes)
    train_data = torch.from_numpy(raw_data[:split_idx].astype(np.int64))
    val_data = torch.from_numpy(raw_data[split_idx:].astype(np.int64))
    print(f'Train Data: {len(train_data):,} bytes | Val Data: {len(val_data):,} bytes\n')

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
    ).to('cpu')

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x_b, y_b = get_batch(split)
                _, loss, _ = model.blt(x_b, targets=y_b)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # Initial loss & PPL
    init_eval = estimate_loss()
    init_ppl = math.exp(min(20.0, init_eval['val']))
    print(f'[Initial Before Training] Val Loss: {init_eval["val"]:.4f} | Val PPL: {init_ppl:.2f}\n')

    # Forward-only training loop
    total_processed_bytes = 0
    t_start = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        t0 = time.perf_counter()
        x_b, y_b = get_batch("train")

        # Forward Pass with Local Predictive Error Update (Zero Backward Tape)
        with torch.no_grad():
            h_byte, b_logits = model.blt.byte_encoder(x_b)
            latent_patches, patch_assignments = model.blt.patcher(h_byte, b_logits, fixed_patch_size=model.blt.target_patch_size)

            x_latent = latent_patches
            for block in model.blt.global_blocks:
                time_norm = block.norm1(x_latent)
                time_out, _ = block.time_mixer(time_norm)
                x_latent = x_latent + time_out

                chan_norm = block.norm2(x_latent)
                cm = block.channel_mixer
                chan_out = cm(chan_norm)
                x_latent = x_latent + chan_out

                # Local in-place predictive plasticity update
                if hasattr(cm, 'w_down') and hasattr(cm, 'w_gate_val'):
                    err = (x_latent - chan_out).sign().to(torch.float32)
                    H_dim = cm.w_down.weight.shape[1]
                    gv = F.linear(chan_norm, cm.w_gate_val.weight)
                    g, v = gv.chunk(2, dim=-1)
                    h_act = (F.silu(g) * v).to(torch.float32)
                    delta_wd = torch.mm(err.view(-1, args.d_model).t(), h_act.view(-1, H_dim)) * (args.learning_rate * 1e-4)
                    cm.w_down.weight.data.add_(delta_wd.to(cm.w_down.weight.dtype))

            x_latent = model.blt.global_norm(x_latent)
            B = x_b.size(0)
            causal_patches = torch.cat([model.blt.sos_patch.expand(B, 1, -1), x_latent[:, :-1]], dim=1)
            logits = model.blt.byte_decoder(h_byte, causal_patches, patch_assignments)

            # Local Head Predictive Update
            one_hot_y = F.one_hot(y_b.view(-1), 256).float()
            probs = F.softmax(logits.view(-1, 256).float(), dim=-1)
            err_head = (one_hot_y - probs)
            delta_lm = torch.mm(err_head.t(), h_byte.view(-1, args.d_byte).float()) * (args.learning_rate * 0.05)
            model.blt.byte_decoder.lm_head.weight.data.add_(delta_lm.to(model.blt.byte_decoder.lm_head.weight.dtype))

        step_time = time.perf_counter() - t0
        bytes_this_step = args.batch_size * args.seq_len
        total_processed_bytes += bytes_this_step

        if step % args.eval_interval == 0 or step == args.max_steps:
            eval_res = estimate_loss()
            val_loss = eval_res['val']
            val_ppl = math.exp(min(20.0, val_loss))
            elapsed = time.perf_counter() - t_start
            avg_speed = total_processed_bytes / elapsed
            print(f'Step {step:4d}/{args.max_steps} | Train Loss: {eval_res["train"]:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:8.2f} | Speed: {avg_speed:10,.0f} B/s')

    print('=' * 72)
    print(f'Training Completed in {time.perf_counter() - t_start:.2f}s!')
    print('=' * 72)


if __name__ == '__main__':
    train()
