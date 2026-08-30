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


def run_sweep():
    print('=' * 80)
    print('       PATCH SIZE PPL & SPEED SWEEP ON 100MB TINYSTORIES (P=4 TO 256)')
    print('=' * 80)

    data_path = 'data/tinystories_extracted.bin'
    raw_data = np.fromfile(data_path, dtype=np.uint8)
    total_bytes = len(raw_data)
    split_idx = int(0.9 * total_bytes)
    train_data = torch.from_numpy(raw_data[:split_idx].astype(np.int64))
    val_data = torch.from_numpy(raw_data[split_idx:].astype(np.int64))

    batch_size = 32
    seq_len = 256
    max_steps = 150

    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(0, len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i:i + seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + seq_len + 1] for i in ix])
        return x, y

    patch_sizes = [4, 8, 16, 32, 64, 128, 256]
    results = []

    for P in patch_sizes:
        print(f'\n>>> Training ASDAG BLT with Target Patch Size P={P}...')
        torch.manual_seed(42)
        model = ASDAGLanguageModel(
            vocab_size=256,
            d_model=160,
            n_layers=4,
            n_heads=4,
            d_byte=64,
            target_patch_size=P,
            channel_mixer_type="ternary_swiglu",
            use_blt=True
        ).to('cpu')

        opt = HybridMuonAdamW(model, muon_lr=0.03, adamw_lr=3e-3)

        t_start = time.perf_counter()
        for step in range(1, max_steps + 1):
            x_b, y_b = get_batch("train")
            opt.zero_grad()
            _, loss, _ = model.blt(x_b, targets=y_b)
            loss.backward()
            opt.step()
        
        train_time = time.perf_counter() - t_start
        speed = (max_steps * batch_size * seq_len) / train_time

        # Evaluate Validation Loss & PPL across 30 batches
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(30):
                x_v, y_v = get_batch("val")
                _, v_loss, _ = model.blt(x_v, targets=y_v)
                val_losses.append(v_loss.item())
        
        mean_val_loss = float(np.mean(val_losses))
        val_ppl = math.exp(min(20.0, mean_val_loss))

        results.append({
            'P': P,
            'patches_per_seq': seq_len // P,
            'speed': speed,
            'val_loss': mean_val_loss,
            'val_ppl': val_ppl
        })

        print(f'P={P:3d} (Patches/Seq: {seq_len//P:3d}) | Val Loss: {mean_val_loss:.4f} | Val PPL: {val_ppl:7.2f} | Speed: {speed:8,.0f} B/s')

    print('\n' + '=' * 80)
    print('                           FINAL SWEEP SUMMARY TABLE')
    print('=' * 80)
    print(f'Patch Size (P)   Patches/Seq (T=256)   Training Speed   Val Loss   Val Perplexity')
    print('-' * 80)
    for r in results:
        delta_str = ""
        if r['P'] > 4:
            delta_ppl = r['val_ppl'] - results[0]['val_ppl']
            delta_str = f' (+{delta_ppl:.2f})' if delta_ppl >= 0 else f' ({delta_ppl:.2f})'
        print(f"P = {r['P']:<3d}          {r['patches_per_seq']:<4d} patches         {r['speed']:7,.0f} B/s     {r['val_loss']:.4f}     {r['val_ppl']:6.2f}{delta_str}")
    print('=' * 80)


if __name__ == '__main__':
    run_sweep()
