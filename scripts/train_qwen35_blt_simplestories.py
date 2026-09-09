#!/usr/bin/env python3
"""
Train Qwen3.5-4B Byte Latent Transformer (BLT) on SimpleStories.
Zero token embedding table: embeds raw UTF-8 bytes [0..255] directly into latent patches (P=16).
Runs on RTX 3050 (4 GB VRAM) using 3-bitplane POT5 frozen backbone (< 3.6 GB peak VRAM).

Usage:
  python scripts/train_qwen35_blt_simplestories.py --max-steps 100 --data-path data/simplestories_eos.bin
"""

import os
import gc
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_pot5_gpu import FastPOT5Block
from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35RMSNorm
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher, ByteLocalDecoder
from affine_ai.training.distill_bytes import DistillBytesAligner


class SimpleStoriesByteDataset:
    """Memory-mapped raw uint8 byte stream dataset."""
    def __init__(self, bin_path: str, seq_len: int = 256):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.total_bytes = len(self.data)
        print(f"Loaded {bin_path}: {self.total_bytes:,} bytes ({self.total_bytes / (1024*1024*1024):.2f} GB)")

    def get_batch(self, batch_size: int = 1, device: str = "cuda"):
        max_idx = self.total_bytes - (self.seq_len + 1)
        indices = np.random.randint(0, max_idx, size=batch_size)
        inputs = []
        targets = []
        for idx in indices:
            chunk = self.data[idx : idx + self.seq_len + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
        x = torch.tensor(np.stack(inputs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(targets), dtype=torch.long, device=device)
        return x, y


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3.5 BLT on SimpleStories with P=16")
    parser.add_argument("--data-path", type=str, default="data/simplestories_eos.bin")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/qwen35_pot5")
    parser.add_argument("--output-dir", type=str, default="checkpoints/qwen35_blt")
    parser.add_argument("--seq-len", type=int, default=256, help="Byte sequence length (default: 256 -> 16 patches)")
    parser.add_argument("--patch-size", type=int, default=16, help="Target patch size (default: 16)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1 for RTX 3050 VRAM)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for local byte interface (default: 5e-4)")
    parser.add_argument("--max-steps", type=int, default=100, help="Max training steps (default: 100)")
    parser.add_argument("--eval-interval", type=int, default=25, help="Eval and sample text interval (default: 25)")
    parser.add_argument("--save-interval", type=int, default=50, help="Checkpoint save interval (default: 50)")
    parser.add_argument("--use-distillbytes", action="store_true", default=True, help="Enable DistillBytes embedding alignment distillation")
    parser.add_argument("--align-weight", type=float, default=2.0, help="DistillBytes alignment loss weight (default: 2.0)")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from BLT interface checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Training Qwen3.5 BLT (Byte-In, Byte-Out, P={args.patch_size}) ===")
    print(f"Device:             {device}")
    print(f"Sequence length:    {args.seq_len} bytes -> {args.seq_len // args.patch_size} latent patches")
    print(f"Batch size (accum): {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Learning rate:      {args.lr}")
    print(f"Dataset:            {args.data_path}")

    dataset = SimpleStoriesByteDataset(args.data_path, seq_len=args.seq_len)

    # 1. Load 32 Qwen ASDAG Backbone Blocks (Frozen in GPU VRAM: 3.32 GB)
    config = Qwen35ASDAGConfig()
    print("\nLoading 32 Qwen3.5 ASDAG Blocks into GPU VRAM...")
    t0 = time.time()
    blocks = []
    for i in range(config.num_layers):
        bp = os.path.join(args.checkpoint_dir, f"block_{i:02d}.pt")
        st = torch.load(bp, map_location="cpu", weights_only=True)
        b = FastPOT5Block(config, i, st)
        del st
        b.eval()
        for p in b.parameters():
            p.requires_grad = False
        blocks.append(b)

    norm = Qwen35RMSNorm(config.dim, eps=config.rms_norm_eps).to(device)
    norm_st = torch.load(os.path.join(args.checkpoint_dir, "output_norm.pt"), map_location="cpu", weights_only=True)
    norm.weight.data.copy_(norm_st.to(device))
    norm.weight.requires_grad = False
    del norm_st
    gc.collect()
    torch.cuda.empty_cache()

    load_time = time.time() - t0
    vram_used = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"32 backbone blocks resident in VRAM ({load_time:.1f}s | {vram_used:.1f} MB allocated)")

    # 2. Local Byte Interface (Trainable: ~822k parameters, ~1.6 MB)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    d_byte = 128

    byte_encoder = ByteLocalEncoder(vocab_size=256, d_byte=d_byte, kernel_size=4, dtype=dtype).to(device)
    patcher = EntropyPatcher(d_byte=d_byte, d_model=config.dim, target_patch_size=args.patch_size, dtype=dtype).to(device)
    sos_patch = nn.Parameter(torch.randn(1, 1, config.dim, dtype=dtype, device=device) * 0.02)
    byte_decoder = ByteLocalDecoder(vocab_size=256, d_byte=d_byte, d_model=config.dim, dtype=dtype).to(device)

    trainable_params = (
        list(byte_encoder.parameters()) +
        list(patcher.parameters()) +
        [sos_patch] +
        list(byte_decoder.parameters())
    )
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable Byte Interface Parameters: {total_trainable:,} ({total_trainable * 2 / (1024*1024):.2f} MB)")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming BLT interface weights from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        byte_encoder.load_state_dict(ckpt["byte_encoder"])
        patcher.load_state_dict(ckpt["patcher"])
        sos_patch.data.copy_(ckpt["sos_patch"].to(device))
        byte_decoder.load_state_dict(ckpt["byte_decoder"])
        print("  Successfully restored weights into ByteLocalEncoder, EntropyPatcher, sos_patch, and ByteLocalDecoder!")

    distill_aligner = DistillBytesAligner(dim=config.dim, device=device) if args.use_distillbytes else None
    if distill_aligner is not None:
        print("DistillBytes Enabled: Supervising byte patch latents with teacher token embedding alignment.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    print("\nStarting alignment training loop...")
    P = args.patch_size
    step = 0
    t_start = time.time()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        step_t0 = time.time()

        # Gradient accumulation
        accum_loss = 0.0
        accum_align_cos = 0.0
        for accum_idx in range(args.grad_accum):
            x, y = dataset.get_batch(batch_size=args.batch_size, device=device)
            B, T = x.shape
            M = T // P

            # 1. Local Byte Encoder
            h_byte, boundary = byte_encoder(x)

            # 2. Entropy Patcher
            latent_patches, patch_assignments = patcher(h_byte, torch.zeros_like(boundary), fixed_patch_size=P)

            # 3. Frozen 32-layer ASDAG Backbone (torch.no_grad saves 3.5 GB of backward activations!)
            with torch.no_grad():
                curr = latent_patches.detach()
                for b in blocks:
                    curr, _ = b(curr)
                final_h = norm(curr)

            # 4. Causal patch shift
            causal_patches = torch.cat([sos_patch.expand(B, 1, -1), final_h[:, :-1]], dim=1)

            # 5. Local Byte Decoder
            logits = byte_decoder(h_byte, causal_patches, patch_assignments)
            ce_loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))

            # 6. Auxiliary patch autoencoder loss (aligns patcher to byte decoder space)
            reconstructed_patches = byte_decoder.patch_to_byte(latent_patches)
            target_pooled = h_byte.view(B, M, P, d_byte).mean(dim=2)
            patch_loss = F.mse_loss(reconstructed_patches, target_pooled)

            # 7. DistillBytes Embedding Alignment Loss
            if distill_aligner is not None:
                align_loss, align_cos = distill_aligner.compute_alignment_loss(latent_patches, x)
                loss = (ce_loss + 0.1 * patch_loss + args.align_weight * align_loss) / args.grad_accum
                accum_align_cos += align_cos / args.grad_accum
            else:
                loss = (ce_loss + 0.1 * patch_loss) / args.grad_accum

            loss.backward()
            accum_loss += ce_loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        step_elapsed = time.time() - step_t0
        running_loss += accum_loss
        bpb = accum_loss / math.log(2)  # Bits per byte

        if step % 5 == 0 or step == 1:
            bytes_per_sec = (args.batch_size * args.grad_accum * args.seq_len) / step_elapsed
            cur_lr = scheduler.get_last_lr()[0]
            align_str = f" | Align Cos: {accum_align_cos:.4f}" if distill_aligner is not None else ""
            print(f"Step {step:4d}/{args.max_steps:4d} | Loss: {accum_loss:.4f} | BPB: {bpb:.2f}{align_str} | {bytes_per_sec:.0f} byte/s | lr: {cur_lr:.2e} | {step_elapsed:.2f}s/step")

        # Evaluate and sample generation
        if step % args.eval_interval == 0 or step == args.max_steps:
            print(f"\n--- [Eval Step {step}] Generating from prompt ---")
            sample_prompt = "Once upon a time, there was a little "
            byte_encoder.eval()
            byte_decoder.eval()
            patcher.eval()

            gen_bytes = list(sample_prompt.encode("utf-8"))
            with torch.no_grad():
                for _ in range(48):
                    cur_in = torch.tensor([gen_bytes[-args.seq_len:]], dtype=torch.long, device=device)
                    cur_T = cur_in.shape[1]
                    rem = cur_T % P
                    if rem != 0:
                        cur_in_padded = F.pad(cur_in, (0, P - rem), value=0)
                    else:
                        cur_in_padded = cur_in

                    h_b, bnd = byte_encoder(cur_in_padded)
                    lp, p_ass = patcher(h_b, torch.zeros_like(bnd), fixed_patch_size=P)
                    curr = lp
                    for b in blocks:
                        curr, _ = b(curr)
                    fh = norm(curr)
                    cp = torch.cat([sos_patch.expand(1, 1, -1), fh[:, :-1]], dim=1)
                    lgt = byte_decoder(h_b, cp, p_ass)
                    next_byte = int(lgt[0, cur_T - 1].argmax().item())
                    gen_bytes.append(next_byte)
                    if next_byte == 0 or next_byte == 10:  # null byte or newline
                        break

            generated_text = bytes(gen_bytes).decode("utf-8", errors="replace")
            print(f"Prompt:     \"{sample_prompt}\"")
            print(f"Generated:  \"{generated_text}\"\n")

            byte_encoder.train()
            byte_decoder.train()
            patcher.train()

        # Save checkpoint
        if step % args.save_interval == 0 or step == args.max_steps:
            ckpt_path = os.path.join(args.output_dir, f"blt_interface_step_{step:04d}.pt")
            torch.save({
                "step": step,
                "loss": accum_loss,
                "byte_encoder": byte_encoder.state_dict(),
                "patcher": patcher.state_dict(),
                "sos_patch": sos_patch.data.cpu(),
                "byte_decoder": byte_decoder.state_dict(),
            }, ckpt_path)
            print(f">>> Saved BLT interface checkpoint to {ckpt_path}")

    total_time = time.time() - t_start
    print(f"\nTraining Complete! {args.max_steps} steps in {total_time:.1f}s ({total_time / args.max_steps:.2f}s/step)")


if __name__ == "__main__":
    main()
