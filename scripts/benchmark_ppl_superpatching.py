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


def test_ppl_convergence(target_patch_size=16, max_steps=300):
    print('=' * 75)
    print(f'  EVALUATING PPL ON 100MB TINYSTORIES (Target Patch Size P={target_patch_size})')
    print('=' * 75)

    data_path = 'data/tinystories_extracted.bin'
    raw_data = np.fromfile(data_path, dtype=np.uint8)
    total_bytes = len(raw_data)
    split_idx = int(0.9 * total_bytes)
    train_data = torch.from_numpy(raw_data[:split_idx].astype(np.int64))
    val_data = torch.from_numpy(raw_data[split_idx:].astype(np.int64))
    
    batch_size = 32
    seq_len = 128
    
    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(0, len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i:i + seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + seq_len + 1] for i in ix])
        return x, y

    torch.manual_seed(42)
    model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=160,
        n_layers=4,
        n_heads=4,
        d_byte=64,
        target_patch_size=target_patch_size,
        channel_mixer_type="ternary_swiglu",
        use_blt=True
    ).to('cpu')

    optimizer = HybridMuonAdamW(model, muon_lr=0.03, adamw_lr=3e-3)

    @torch.no_grad()
    def estimate_val_ppl(eval_iters=20):
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_b, y_b = get_batch("val")
            _, loss, _ = model.blt(x_b, targets=y_b)
            losses[k] = loss.item()
        model.train()
        mean_loss = losses.mean().item()
        return mean_loss, math.exp(min(20.0, mean_loss))

    init_loss, init_ppl = estimate_val_ppl()
    print(f'Initial Untrained:  Val Loss: {init_loss:.4f} | Val PPL: {init_ppl:8.2f}')

    t_start = time.perf_counter()
    for step in range(1, max_steps + 1):
        x_b, y_b = get_batch("train")
        optimizer.zero_grad()
        _, loss, _ = model.blt(x_b, targets=y_b)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == max_steps:
            val_loss, val_ppl = estimate_val_ppl()
            elapsed = time.perf_counter() - t_start
            speed = (step * batch_size * seq_len) / elapsed
            print(f'Step {step:4d}/{max_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:8.2f} | Speed: {speed:9,.0f} B/s')

    val_loss, val_ppl = estimate_val_ppl(eval_iters=30)
    print(f'FINAL EVALUATION:   Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:8.2f}')
    print('=' * 75 + '\n')
    return val_loss, val_ppl


if __name__ == '__main__':
    print('Comparing PPL convergence between P=4 and P=16...\n')
    loss_p4, ppl_p4 = test_ppl_convergence(target_patch_size=4, max_steps=200)
    loss_p16, ppl_p16 = test_ppl_convergence(target_patch_size=16, max_steps=200)

    print('=' * 75)
    print('                      SUMMARY PPL COMPARISON')
    print('=' * 75)
    print(f'Baseline P=4:   Val Loss = {loss_p4:.4f} | Val PPL = {ppl_p4:.2f}')
    print(f'Chained P=16:    Val Loss = {loss_p16:.4f} | Val PPL = {ppl_p16:.2f}')
    print('=' * 75)
