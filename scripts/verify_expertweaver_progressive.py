#!/usr/bin/env python3
"""
Progressive Verification Gates for ExpertWeaver ASDAG Upcycling.
Evaluates the mathematical and representational fidelity of each transformation gate:
  - Gate 0: Full Pretrained Dense SwiGLU (Baseline Teacher)
  - Gate 1: ExpertWeaver MoE Split (1 Shared Leaf + 7 Routed Leaves, Gate-Centroid Router, BF16)
  - Gate 2: Routed Leaves Ternarized (Shared Leaf kept BF16)
  - Gate 3: Shared Leaf Ternarized (Unpruned, Protected from 1:16)
  - Gate 4: Full ASDAG (Shared Leaf Ternary Unpruned + Routed Leaves Ternary 1:16 Sparse)
  - Comparison: Old Naive Slicing (Uniform 8 leaves, uninitialized router, all 1:16 sparse)
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import math
import torch
import torch.nn.functional as F

from affine_ai.models.qwen35_asdag import (
    Qwen35ASDAGConfig,
    Qwen35ASDAGLeaf,
    Qwen35ASDAGFFN,
    apply_nm_sparsity
)
from affine_ai.core.ast_dag import quantize_shift4, ternarize, dual_ternarize

try:
    import gguf
    from gguf.quants import dequantize
except ImportError:
    gguf = None


def dequantize_tensor(tensor) -> torch.Tensor:
    arr = dequantize(tensor.data, tensor.tensor_type)
    return torch.from_numpy(arr.copy()).float()


def evaluate_gate(y_pred: torch.Tensor, y_true: torch.Tensor, name: str, x_input: torch.Tensor = None) -> dict:
    diff = (y_pred - y_true).float()
    rel_l2_err = (diff.norm() / (y_true.float().norm() + 1e-8)).item()
    cos_sim = F.cosine_similarity(y_pred.float().reshape(-1), y_true.float().reshape(-1), dim=0).item()
    snr_db = 10.0 * math.log10((y_true.float().norm().item() ** 2) / (diff.norm().item() ** 2 + 1e-12))
    
    res_str = ""
    if x_input is not None:
        r_pred = (x_input + y_pred).float().reshape(-1)
        r_true = (x_input + y_true).float().reshape(-1)
        r_cos = F.cosine_similarity(r_pred, r_true, dim=0).item()
        r_diff = (r_pred - r_true).norm().item()
        r_snr = 10.0 * math.log10((r_true.norm().item() ** 2) / (r_diff ** 2 + 1e-12))
        res_str = f" | ResCos: {r_cos:7.4f} | ResSNR: {r_snr:5.2f} dB"

    print(f"| {name:<46} | L2Err: {rel_l2_err*100:5.2f}% | CosSim: {cos_sim:7.4f} | SNR: {snr_db:5.2f} dB{res_str} |")
    return {"rel_l2_err": rel_l2_err, "cos_sim": cos_sim, "snr_db": snr_db}


def main():
    gguf_path = "/home/semih/Modeller/Qwen3.5-4B-Q4_K_M.gguf"
    if not os.path.exists(gguf_path):
        print(f"ERROR: GGUF model not found at {gguf_path}")
        sys.exit(1)

    print("=" * 95)
    print(" AffineAI Progressive Verification Gates: ExpertWeaver vs Naive Upcycling")
    print("=" * 95)

    reader = gguf.GGUFReader(gguf_path)
    layer_idx = 0
    print(f"Loading Layer {layer_idx} weights from {gguf_path}...")

    layer_tensors = {t.name: t for t in reader.tensors if t.name.startswith(f"blk.{layer_idx}.")}
    W_gate = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ffn_gate.weight"])  # [9216, 2560]
    W_up = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ffn_up.weight"])      # [9216, 2560]
    W_down = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ffn_down.weight"])  # [2560, 9216]

    dim = 2560
    intermediate_dim = 9216
    num_leaves = 8
    num_shared = 1
    num_routed = 7
    leaf_dim = intermediate_dim // num_leaves  # 1152

    # Synthetic realistic calibration activations
    torch.manual_seed(42)
    B, T = 4, 128
    x_input = torch.randn(B, T, dim, dtype=torch.bfloat16)
    x_2d = x_input.reshape(-1, dim).float()

    # --- GATE 0: Dense Teacher Baseline ---
    y_dense = (F.silu(x_2d @ W_gate.T) * (x_2d @ W_up.T)) @ W_down.T
    print("\n[Baseline] Pretrained Full Precision Dense SwiGLU computed.")
    print("-" * 95)

    # 1. Compute Neuron CV for ExpertWeaver Universal Partitioning
    num_probe = 2048
    probe = torch.randn(num_probe, dim) / math.sqrt(dim)
    neuron_acts = F.silu(probe @ W_gate.T) * (probe @ W_up.T)
    cv = neuron_acts.std(dim=0) / (neuron_acts.mean(dim=0).abs() + 1e-6)
    sorted_cv = torch.argsort(cv)

    shared_idx = sorted_cv[:leaf_dim]
    spec_idx = sorted_cv[leaf_dim:]

    # Partition specialized neurons into 7 balanced leaves
    from scripts.upcycle_strat_c import balanced_spherical_kmeans, apply_wanda_1_to_16_sparsity
    rand_k = torch.randn(dim, 128) / math.sqrt(dim)
    spec_feats = (F.silu(W_gate[spec_idx] @ rand_k) * (W_up[spec_idx] @ rand_k))
    spec_assignments, _ = balanced_spherical_kmeans(spec_feats, k=num_routed, num_iters=15)

    # Compute ExpertWeaver Gate Centroids for the 7 routed leaves
    router_weights = []
    routed_indices_list = []
    for r in range(num_routed):
        mask = (spec_assignments == r)
        indices = spec_idx[mask]
        routed_indices_list.append(indices)
        centroid = W_gate[indices].mean(dim=0)
        centroid = centroid / (centroid.norm() + 1e-8)
        router_weights.append(centroid)

    W_router = torch.stack(router_weights, dim=0).to(torch.bfloat16)  # [7, dim]
    W_router_shift4 = quantize_shift4(W_router).float()

    # Pre-compute routing decisions (Top-1 routed)
    x_route_shift4 = quantize_shift4(x_2d.to(torch.bfloat16)).float()
    logits = x_route_shift4 @ W_router_shift4.T  # [N, 7]
    routing_w, top_routed = torch.topk(logits, 1, dim=-1)
    routing_w = F.softmax(routing_w, dim=-1)

    # Shared Leaf forward helper
    def forward_shared(x, ternary=False):
        wg = ternarize(W_gate[shared_idx].to(torch.bfloat16)).float() if ternary else W_gate[shared_idx]
        wu = ternarize(W_up[shared_idx].to(torch.bfloat16)).float() if ternary else W_up[shared_idx]
        wd = ternarize(W_down[:, shared_idx].to(torch.bfloat16)).float() if ternary else W_down[:, shared_idx]
        return (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T

    # Routed Leaf forward helper with configurable top_k
    Hessian_diag = (W_gate ** 2).sum(dim=0) + 1e-4
    def forward_routed_k(x, k=1, ternary=False, sparse_1_16=False):
        routing_w_k, top_routed_k = torch.topk(logits, k, dim=-1)
        routing_w_k = F.softmax(routing_w_k, dim=-1)
        out = torch.zeros_like(x)
        for r_idx, indices in enumerate(routed_indices_list):
            mask = (top_routed_k == r_idx)
            token_mask = mask.any(dim=-1)
            if not token_mask.any():
                continue
            sub_x = x[token_mask]

            wg = W_gate[indices].clone().to(torch.bfloat16)
            wu = W_up[indices].clone().to(torch.bfloat16)
            wd = W_down[:, indices].clone().to(torch.bfloat16)

            if sparse_1_16:
                wg, _ = apply_wanda_1_to_16_sparsity(wg, Hessian_diag)
                wu, _ = apply_wanda_1_to_16_sparsity(wu, Hessian_diag)
                wd, _ = apply_wanda_1_to_16_sparsity(wd, (wd**2).sum(dim=0))

            if ternary:
                wg = ternarize(wg)
                wu = ternarize(wu)
                wd = ternarize(wd)

            sub_out = (F.silu(sub_x @ wg.float().T) * (sub_x @ wu.float().T)) @ wd.float().T
            w_sum = (routing_w_k * mask.float()).sum(dim=-1, keepdim=True)
            out[token_mask] += sub_out * w_sum[token_mask]
        return out

    # --- GATE 1: MoE Split Only (BF16, No Sparsity) ---
    y_gate1_top1 = forward_shared(x_2d, ternary=False) + forward_routed_k(x_2d, k=1, ternary=False, sparse_1_16=False)
    evaluate_gate(y_gate1_top1, y_dense, "Gate 1: ExpertWeaver MoE (BF16, Top-1)", x_input=x_2d)

    y_gate1_top2 = forward_shared(x_2d, ternary=False) + forward_routed_k(x_2d, k=2, ternary=False, sparse_1_16=False)
    evaluate_gate(y_gate1_top2, y_dense, "Gate 1: ExpertWeaver MoE (BF16, Top-2)", x_input=x_2d)

    # --- GATE 2: Routed Leaves Ternarized (Shared Leaf BF16) ---
    y_gate2 = forward_shared(x_2d, ternary=False) + forward_routed_k(x_2d, k=1, ternary=True, sparse_1_16=False)
    evaluate_gate(y_gate2, y_dense, "Gate 2: Routed Ternary (Shared BF16)", x_input=x_2d)

    # --- GATE 3: Shared Leaf Ternarized (Unpruned) ---
    y_gate3 = forward_shared(x_2d, ternary=True) + forward_routed_k(x_2d, k=1, ternary=True, sparse_1_16=False)
    evaluate_gate(y_gate3, y_dense, "Gate 3: Both Ternary (Shared Unpruned)", x_input=x_2d)

    # --- GATE 4: Full ASDAG (Routed 1:16 Sparse + Shared Protected) ---
    y_gate4_top1 = forward_shared(x_2d, ternary=True) + forward_routed_k(x_2d, k=1, ternary=True, sparse_1_16=True)
    evaluate_gate(y_gate4_top1, y_dense, "Gate 4: Full ASDAG (Top-1, 1:16 Sparse)", x_input=x_2d)

    y_gate4_top2 = forward_shared(x_2d, ternary=True) + forward_routed_k(x_2d, k=2, ternary=True, sparse_1_16=True)
    evaluate_gate(y_gate4_top2, y_dense, "Gate 4: Full ASDAG (Top-2, 1:16 Sparse)", x_input=x_2d)

    # --- NAIVE BASELINE: Old Slicing (All 8 Leaves 1:16 Sparse, Random Router) ---
    def forward_naive(x):
        torch.manual_seed(999)
        W_router_rand = torch.randn(8, dim, dtype=torch.bfloat16)
        logits_rand = quantize_shift4(x.to(torch.bfloat16)).float() @ quantize_shift4(W_router_rand).float().T
        rw, top_leaves = torch.topk(logits_rand, 2, dim=-1)
        rw = F.softmax(rw, dim=-1)
        out = torch.zeros_like(x)
        for leaf_k in range(8):
            mask = (top_leaves == leaf_k).any(dim=-1)
            if not mask.any():
                continue
            idx = slice(leaf_k * leaf_dim, (leaf_k + 1) * leaf_dim)
            wg, _ = apply_wanda_1_to_16_sparsity(W_gate[idx, :].to(torch.bfloat16), Hessian_diag)
            wu, _ = apply_wanda_1_to_16_sparsity(W_up[idx, :].to(torch.bfloat16), Hessian_diag)
            wd, _ = apply_wanda_1_to_16_sparsity(W_down[:, idx].to(torch.bfloat16), (W_down[:, idx]**2).sum(dim=0))
            sub_x = x[mask]
            sub_out = (F.silu(sub_x @ ternarize(wg).float().T) * (sub_x @ ternarize(wu).float().T)) @ ternarize(wd).float().T
            w_sum = (rw * (top_leaves == leaf_k).float()).sum(dim=-1, keepdim=True)
            out[mask] += sub_out * w_sum[mask]
        return out

    y_naive = forward_naive(x_2d)
    evaluate_gate(y_naive, y_dense, "Naive Upcycling (Old Slicing, Random)", x_input=x_2d)

    # --- GATE 5: Candidate 4 Dual-Ternary Dense (100% Compute, Pure Integer Add/Sub) ---
    W_g_dt = dual_ternarize(W_gate)
    W_u_dt = dual_ternarize(W_up)
    W_d_dt = dual_ternarize(W_down)
    y_gate5_dense = (F.silu(x_2d @ W_g_dt.T) * (x_2d @ W_u_dt.T)) @ W_d_dt.T
    evaluate_gate(y_gate5_dense, y_dense, "Gate 5: Dual-Ternary Dense (100% Compute)", x_input=x_2d)

    # --- GATE 6: Candidate 4 AbsTopK Dual-Ternary (25% Compute, 75% Savings) ---
    config_c4 = Qwen35ASDAGConfig(
        dim=dim,
        intermediate_dim=intermediate_dim,
        sparsity_mode="abstopk",
        use_dual_ternary=True,
        retain_ratio=0.25,
        dtype=torch.float32
    )
    ffn_c4 = Qwen35ASDAGFFN(config_c4)
    ffn_c4.gate_proj.weight.data = W_gate.clone()
    ffn_c4.up_proj.weight.data = W_up.clone()
    ffn_c4.down_proj.weight.data = W_down.clone()
    ffn_c4.eval()

    y_gate6_c4 = ffn_c4(x_2d, retain_ratio=0.25)
    evaluate_gate(y_gate6_c4, y_dense, "Gate 6: Candidate 4 AbsTopK-GLU (25% Compute)", x_input=x_2d)

    print("-" * 95)
    print("Progressive Verification Complete!\n")


if __name__ == "__main__":
    main()
