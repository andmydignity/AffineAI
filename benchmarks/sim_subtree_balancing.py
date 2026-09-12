#!/usr/bin/env python3
"""
Standalone Experiment: Subtree Threshold Balancing vs Baseline on SimpleStories
================================================================================
Tests Subtree Threshold Balancing in isolation without modifying any library code.
Compares:
  A) Baseline (Current Checkpoint / standard routing)
  B) Subtree Threshold Balancing (adjusting internal tree node biases via traffic error)
Measures:
  - Active leaves count (out of 16) per layer
  - Dead leaves count (<1% traffic) per layer
  - Top-1 concentration (%) per layer
  - Routing entropy (H / H_max) per layer
  - Validation Perplexity (PPL) to ensure language modeling performance is preserved
"""

import math
import copy
import numpy as np
import torch
import torch.nn.functional as F

from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.data.dataloader import PaddedDataLoader
from affine_ai.training.trainer import ASDAGTrainer


def audit_layer_routing(model, val_loader, n_batches=4, device="cuda"):
    """Audits token distribution across leaves for all 7 layers without modifying weights."""
    model.eval()
    blocks = model.context_encoder.blocks
    num_layers = len(blocks)
    layer_counts = [torch.zeros(16, dtype=torch.float64, device=device) for _ in range(num_layers)]

    val_iter = iter(val_loader)
    for _ in range(n_batches):
        x, _ = next(val_iter)
        B, T = x.shape
        # Run byte encoder + patcher
        h_byte, boundary_logits = model.context_encoder.byte_encoder(x)
        latent_patches, _ = model.context_encoder.patcher(h_byte, torch.zeros_like(boundary_logits), fixed_patch_size=model.config.target_patch_size)
        h = latent_patches
        for li, blk in enumerate(blocks):
            # Record router dispatch
            cm = getattr(blk, "channel_mixer", None) or getattr(blk, "asdag", None)
            x_flat = blk.norm2(h).reshape(-1, cm.dim)
            with torch.no_grad():
                probs, _ = cm.router.route_tokens(x_flat)
                top1 = probs.argmax(dim=-1).reshape(-1)
                layer_counts[li] += torch.bincount(top1, minlength=16).to(layer_counts[li].dtype)
            h = blk(h)

    results = []
    for li in range(num_layers):
        c = layer_counts[li].cpu()
        tot = c.sum().item()
        fractions = (c / max(1.0, tot)).tolist()
        dead = sum(1 for f in fractions if f < 0.01)
        top1 = max(fractions)
        top1_idx = fractions.index(top1)
        ent = -sum(f * math.log(f) for f in fractions if f > 0)
        h_max = math.log(16)
        results.append({
            "layer": li,
            "dead": dead,
            "active": 16 - dead,
            "top1": top1,
            "top1_idx": top1_idx,
            "entropy": ent,
            "h_max": h_max,
            "fractions": fractions,
        })
    return results


def simulate_subtree_balancing_step(cm, x_flat, gamma=0.01):
    """
    Subtree Balancing Update Rule:
    For each internal tree node j, computes left vs right child token volume and updates bias:
      delta_b_j = -gamma * mean(pv_j * tanh(z_j))
    """
    with torch.no_grad():
        B = x_flat.shape[0]
        W_route = cm.router.hyperplanes.sign() # ternary sign
        node_logits = F.linear(x_flat.float(), W_route.float(), cm.router.biases.float())

        # Level 0 (root)
        z0 = node_logits[:, 0]
        pr0 = torch.sigmoid(z0 * 2.0)
        pl0 = 1.0 - pr0
        cur = torch.stack([pl0, pr0], dim=-1) # [B, 2]

        # Update root bias
        err0 = (pl0 - pr0).mean() # if left > right, err > 0 -> increase bias to shift right
        cm.router.biases.data[0] += gamma * err0

        # Levels 1 to 3
        tree_depth = cm.router.tree_depth
        for d in range(1, tree_depth):
            start_node = (1 << d) - 1
            num_nodes = 1 << d
            level_logits = node_logits[:, start_node:start_node + num_nodes]
            pr = torch.sigmoid(level_logits * 2.0)
            pl = 1.0 - pr

            # Traffic reaching each node j at depth d is cur[:, j]
            # Left tokens: cur[:, j] * pl[:, j], Right tokens: cur[:, j] * pr[:, j]
            err_d = (cur * (pl - pr)).mean(dim=0)
            cm.router.biases.data[start_node:start_node + num_nodes] += gamma * err_d

            cur = torch.stack([cur * pl, cur * pr], dim=-1).view(B, -1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "models/toros_hybrid_700k_simplestories.pt"

    print("=" * 95)
    print("      TESTING SUBTREE THRESHOLD BALANCING (ZERO LIBRARY CHANGES)")
    print("=" * 95)

    data_path = "data/simplestories_eos.bin"
    raw_data = np.memmap(data_path, dtype=np.uint8, mode="r")
    split = int(0.95 * len(raw_data))
    val_loader = PaddedDataLoader(raw_data[split:], batch_size=32, seq_len=512, device=device, as_stream=True)
    train_loader = PaddedDataLoader(raw_data[:split], batch_size=32, seq_len=512, device=device, as_stream=True)

    # 1. Audit Checkpoint Baseline
    cfg = TorosHybridConfig(dim=288, n_encoder_layers=7, channel_mixer_type="asdag_tree", dtype=torch.bfloat16)
    model = TorosHybridLanguageModel(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    print("\nPhase 1: Auditing Baseline Checkpoint...")
    base_results = audit_layer_routing(model, val_loader, n_batches=4, device=device)

    total_base_dead = sum(r["dead"] for r in base_results)
    avg_base_ent = sum(r["entropy"] for r in base_results) / len(base_results)
    print(f"Baseline: {total_base_dead}/112 dead leaves ({total_base_dead/112*100:.1f}%), Avg Entropy: {avg_base_ent:.2f}/2.77")

    # 2. Run 100 simulation steps of Subtree Threshold Balancing on training stream
    print("\nPhase 2: Simulating 100 steps of Subtree Threshold Balancing (gamma=0.02)...")
    train_iter = iter(train_loader)
    for step in range(1, 101):
        x, _ = next(train_iter)
        h_byte, boundary_logits = model.context_encoder.byte_encoder(x)
        latent_patches, _ = model.context_encoder.patcher(h_byte, torch.zeros_like(boundary_logits), fixed_patch_size=model.config.target_patch_size)
        h = latent_patches
        for blk in model.context_encoder.blocks:
            cm = getattr(blk, "channel_mixer", None) or getattr(blk, "asdag", None)
            x_flat = blk.norm2(h).reshape(-1, cm.dim)
            simulate_subtree_balancing_step(cm, x_flat, gamma=0.02)
            h = blk(h)

    # 3. Audit post-balancing routing
    print("Phase 3: Auditing Post-Balancing Checkpoint on Validation Data...\n")
    new_results = audit_layer_routing(model, val_loader, n_batches=4, device=device)

    print("-" * 95)
    print(f"{'Layer':<5} | {'Baseline Dead':<14} | {'Subtree Dead':<14} | {'Baseline Top-1':<15} | {'Subtree Top-1':<15} | {'Entropy Gain'}")
    print("-" * 95)

    total_new_dead = 0
    for li in range(len(new_results)):
        rb = base_results[li]
        rn = new_results[li]
        total_new_dead += rn["dead"]
        ent_diff = rn["entropy"] - rb["entropy"]
        sign = "+" if ent_diff >= 0 else ""
        print(f"L{li:<4d} | {rb['dead']:2d}/16 ({rb['dead']/16*100:4.1f}%) | "
              f"{rn['dead']:2d}/16 ({rn['dead']/16*100:4.1f}%) | "
              f"{rb['top1']*100:5.1f}% (#{rb['top1_idx']:<2d})     | "
              f"{rn['top1']*100:5.1f}% (#{rn['top1_idx']:<2d})     | "
              f"{rb['entropy']:.2f} -> {rn['entropy']:.2f} ({sign}{ent_diff:+.2f})")

    print("-" * 95)
    print(f"TOTAL DEAD LEAVES: {total_base_dead}/112 ({total_base_dead/112*100:.1f}%)  ==>  {total_new_dead}/112 ({total_new_dead/112*100:.1f}%)")
    print("=" * 95)

    # Verify Validation PPL after subtree balancing
    trainer = ASDAGTrainer(model=model, train_data=train_loader, val_data=val_loader, batch_size=32, seq_len=512, device=device)
    eval_m = trainer.evaluate()
    print(f"Post-balancing Validation PPL: {eval_m['val_ppl']:.2f} (Loss: {eval_m['val_loss']:.4f})")
    print("=" * 95)


if __name__ == "__main__":
    main()
