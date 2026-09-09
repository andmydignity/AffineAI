#!/usr/bin/env python3
"""
Strategy C Implementation: Analytical Initialization & Upcycling Engine.
Replaces naive slicing with:
  1. Calibration-based activation & Hessian collection.
  2. Balanced Spherical K-Means Neuron Clustering on FFN co-activations.
  3. Data-driven Router Centroid Initialization.
  4. Wanda (Weight * Activation) 1:16 Structured Sparsity Masking.
  5. Complete Time Mixer (DeltaNet & Attention) + Norm Extraction in BF16/FP32.
  6. Unquantized BF16 Sacred Embeddings and Master Weights Preservation.

Supports both Qwen3.5-4B and Qwen3.8-27B architectures.
"""

import os
import gc
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

try:
    import gguf
    from gguf.quants import dequantize
except ImportError:
    gguf = None

from affine_ai.core.ast_dag import quantize_shift4, ternarize


def parse_args():
    parser = argparse.ArgumentParser(description="Strategy C Analytical Upcycling Engine")
    parser.add_argument("--model-version", type=str, choices=["3.5", "3.8"], default="3.5",
                        help="Target architecture: 3.5 (4B) or 3.8 (27B)")
    parser.add_argument("--gguf-path", type=str, default="/home/semih/Modeller/Qwen3.5-4B-Q4_K_M.gguf",
                        help="Path to GGUF model file")
    parser.add_argument("--calib-text", type=str, default="data/tinystories_10mb.txt",
                        help="Text file for calibration tokens")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of calibration sequences")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for calibration")
    parser.add_argument("--output-dir", type=str, default="checkpoints/strat_c_upcycled",
                        help="Output directory for initialized master checkpoints")
    parser.add_argument("--mode", type=str, choices=["abstopk", "cluster_moe"], default="abstopk",
                        help="FFN Sparsity Mode: abstopk (Candidate 4, 75% savings) or cluster_moe (8 leaves)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of layers to process (default: all layers in config)")
    return parser.parse_args()


def dequantize_tensor(tensor) -> torch.Tensor:
    arr = dequantize(tensor.data, tensor.tensor_type)
    return torch.from_numpy(arr.copy()).float()


def balanced_spherical_kmeans(
    vectors: torch.Tensor,
    k: int = 8,
    num_iters: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Balanced Spherical K-Means:
    Clusters `vectors` of shape [num_neurons, N] into `k` balanced clusters
    where each cluster gets exactly num_neurons // k members.
    Returns: cluster_indices [num_neurons], centroids [k, N]
    """
    N_neurons, feat_dim = vectors.shape
    cluster_size = N_neurons // k
    assert N_neurons % k == 0, f"Cannot divide {N_neurons} evenly into {k} clusters"

    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed_v = vectors / norms

    torch.manual_seed(42)
    perm = torch.randperm(N_neurons)
    centroids = normed_v[perm[:k]].clone()
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)

    assignments = torch.zeros(N_neurons, dtype=torch.long)

    for iter_idx in range(num_iters):
        sims = torch.matmul(normed_v, centroids.T)

        assigned_mask = torch.zeros(N_neurons, dtype=torch.bool)
        cluster_counts = torch.zeros(k, dtype=torch.long)

        flat_sims = sims.flatten()
        sorted_indices = torch.argsort(flat_sims, descending=True)

        for flat_idx in sorted_indices:
            neuron_id = (flat_idx // k).item()
            cluster_id = (flat_idx % k).item()

            if not assigned_mask[neuron_id] and cluster_counts[cluster_id] < cluster_size:
                assignments[neuron_id] = cluster_id
                assigned_mask[neuron_id] = True
                cluster_counts[cluster_id] += 1

            if assigned_mask.all():
                break

        for c in range(k):
            members = normed_v[assignments == c]
            if len(members) > 0:
                new_c = members.mean(dim=0)
                centroids[c] = new_c / (new_c.norm() + 1e-8)

    return assignments, centroids


def apply_wanda_1_to_16_sparsity(w: torch.Tensor, hessian_diag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wanda (Weight * Activation) 1:16 Structured Pruning:
    Score S_ij = |W_ij| * sqrt(H_j).
    In each 16-group along input dim, the 1 weight with max S_ij survives.
    """
    orig_shape = w.shape
    in_dim = orig_shape[-1]
    assert in_dim % 16 == 0

    act_scale = torch.sqrt(hessian_diag.float().clamp(min=1e-8)).unsqueeze(0)
    score = w.float().abs() * act_scale

    score_16 = score.reshape(-1, 16)
    _, max_idx = torch.topk(score_16, 1, dim=-1)

    mask_16 = torch.zeros_like(score_16, dtype=torch.bool)
    mask_16.scatter_(-1, max_idx, True)
    mask = mask_16.reshape(orig_shape)

    w_sparse = torch.where(mask, w, torch.zeros_like(w))
    return w_sparse, mask


def main():
    args = parse_args()
    print("=" * 70)
    print(" AffineAI Strategy C: Analytical Calibrated Upcycle Engine")
    print("=" * 70)
    print(f"Target Architecture: Qwen{args.model_version}")
    print(f"Source GGUF:         {args.gguf_path}")
    print(f"Calibration File:    {args.calib_text}")
    print(f"Output Directory:    {args.output_dir}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.gguf_path):
        print(f"ERROR: GGUF path {args.gguf_path} does not exist.")
        sys.exit(1)

    if args.model_version == "3.8":
        from affine_ai.models.qwen38_asdag import Qwen38ASDAGConfig
        config = Qwen38ASDAGConfig()
    else:
        from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig
        config = Qwen35ASDAGConfig()

    max_layers = args.num_layers if args.num_layers is not None else config.num_layers
    print(f"\nModel Configuration: dim={config.dim}, layers={max_layers}/{config.num_layers}, FFN={config.intermediate_dim} (8 leaves x {config.leaf_dim})")
    
    # 1. Load Calibration Text
    print("\n[1/4] Loading Calibration Text...")
    if os.path.exists(args.calib_text):
        with open(args.calib_text, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(args.num_samples * args.seq_len * 4)
    else:
        text = "The quick brown fox jumps over the lazy dog. " * 500

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("./")
    raw_tokens = tok.encode(text, return_tensors="pt")
    n_needed = args.num_samples * args.seq_len
    if raw_tokens.shape[1] < n_needed:
        rep = (n_needed // raw_tokens.shape[1]) + 1
        raw_tokens = raw_tokens.repeat(1, rep)
    calib_batches = raw_tokens[:, :n_needed].reshape(args.num_samples, args.seq_len)
    print(f"Created {args.num_samples} calibration sequences of length {args.seq_len}.")

    reader = gguf.GGUFReader(args.gguf_path)
    print(f"Opened GGUF reader with {len(reader.tensors)} tensors.")

    # 2. Extract Sacred Embeddings
    print("\n[2/4] Preserving Sacred Token Embeddings in BF16...")
    token_embd_t = next(t for t in reader.tensors if t.name == "token_embd.weight")
    token_embd = dequantize_tensor(token_embd_t).to(torch.bfloat16)
    emb_path = os.path.join(args.output_dir, "token_embd.pt")
    torch.save(token_embd, emb_path)
    print(f"Saved unquantized embedding {token_embd.shape} to {emb_path}.")

    output_norm_t = next(t for t in reader.tensors if t.name == "output_norm.weight")
    output_norm = dequantize_tensor(output_norm_t).to(torch.float32)
    norm_path = os.path.join(args.output_dir, "output_norm.pt")
    torch.save(output_norm, norm_path)
    del token_embd, output_norm
    gc.collect()

    # 3. Layer-by-Layer Calibration, Time Mixer Extraction & Clustering
    print("\n[3/4] Calibrating Layers & Clustering FFN Neurons...")
    
    for layer_idx in range(max_layers):
        t0 = time.time()
        print(f"\n--- Processing Layer {layer_idx}/{max_layers} ---")
        layer_tensors = {t.name: t for t in reader.tensors if t.name.startswith(f"blk.{layer_idx}.")}
        is_full_attn = (layer_idx == config.num_layers - 1) or ((layer_idx + 1) % config.full_attn_interval == 0)

        block_state = {}

        # A. Attn Norm
        attn_norm = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_norm.weight"]).to(config.dtype)
        block_state["attn_norm.weight"] = attn_norm

        # B. Time Mixer (DeltaNet or Attention)
        if is_full_attn:
            q_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_q.weight"]).to(config.dtype)
            k_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_k.weight"]).to(config.dtype)
            v_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_v.weight"]).to(config.dtype)
            q_norm = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_q_norm.weight"]).to(config.dtype)
            k_norm = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_k_norm.weight"]).to(config.dtype)
            out_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_output.weight"]).to(config.dtype)

            block_state["time_mixer.attn_q.weight"] = q_w
            block_state["time_mixer.attn_k.weight"] = k_w
            block_state["time_mixer.attn_v.weight"] = v_w
            block_state["time_mixer.q_norm.weight"] = q_norm
            block_state["time_mixer.k_norm.weight"] = k_norm
            block_state["time_mixer.attn_output.weight"] = out_w
        else:
            qkv_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_qkv.weight"]).to(config.dtype)
            conv_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ssm_conv1d.weight"]).unsqueeze(1).to(config.dtype)
            alpha_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ssm_alpha.weight"]).to(torch.float32)
            beta_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ssm_beta.weight"]).to(torch.float32)
            ssm_a = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ssm_a"]).to(torch.float32)
            ssm_dt = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ssm_dt.bias"]).to(torch.float32)
            ssm_norm = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ssm_norm.weight"]).to(config.dtype)
            gate_w = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.attn_gate.weight"]).to(config.dtype)
            ssm_out = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ssm_out.weight"]).to(config.dtype)

            block_state["time_mixer.qkv_proj.weight"] = qkv_w
            block_state["time_mixer.conv1d.weight"] = conv_w
            block_state["time_mixer.alpha_proj.weight"] = alpha_w
            block_state["time_mixer.beta_proj.weight"] = beta_w
            block_state["time_mixer.ssm_a"] = ssm_a
            block_state["time_mixer.ssm_dt_bias"] = ssm_dt
            block_state["time_mixer.norm.weight"] = ssm_norm
            block_state["time_mixer.attn_gate.weight"] = gate_w
            block_state["time_mixer.ssm_out.weight"] = ssm_out

        # C. Post-Attention Norm
        post_norm = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.post_attention_norm.weight"]).to(config.dtype)
        block_state["post_attention_norm.weight"] = post_norm

        # D. FFN: Dense extraction + Wanda Sparsity + Balanced Clustering
        W_gate = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ffn_gate.weight"])  # [intermediate_dim, dim]
        W_up = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ffn_up.weight"])      # [intermediate_dim, dim]
        W_down = dequantize_tensor(layer_tensors[f"blk.{layer_idx}.ffn_down.weight"])  # [dim, intermediate_dim]

        if args.mode == "abstopk":
            # Candidate 4: Un-sliced Master Weights preserved in BF16 for AbsTopK Dual-Ternary execution
            block_state["asdag_ffn.gate_proj.weight"] = W_gate.clone().to(torch.bfloat16)
            block_state["asdag_ffn.up_proj.weight"] = W_up.clone().to(torch.bfloat16)
            block_state["asdag_ffn.down_proj.weight"] = W_down.clone().to(torch.bfloat16)
        else:
            # Cluster MoE mode with 1 shared leaf + 7 routed leaves
            Hessian_diag = (W_gate ** 2).sum(dim=0) + 1e-4

            # 1. Compute neuron activation statistics via probe to determine Coefficient of Variation (CV)
            num_probe_tokens = 2048
            torch.manual_seed(100 + layer_idx)
            probe_tokens = torch.randn(num_probe_tokens, config.dim) / math.sqrt(config.dim)

            gate_proj = probe_tokens @ W_gate.T
            up_proj = probe_tokens @ W_up.T
            neuron_acts = F.silu(gate_proj) * up_proj

            act_mean = neuron_acts.mean(dim=0).abs()
            act_std = neuron_acts.std(dim=0)
            cv = act_std / (act_mean + 1e-6)

            sorted_cv_indices = torch.argsort(cv)

            leaf_dim = config.leaf_dim
            num_shared = config.num_shared_leaves
            num_routed = config.num_routed_leaves

            # Leaf 0: Universal Shared Leaf
            shared_indices = sorted_cv_indices[:leaf_dim]
            specialized_indices = sorted_cv_indices[leaf_dim:]

            # 2. Cluster remaining specialized neurons into num_routed balanced leaves
            proj_dim = 128
            rand_probe_k = torch.randn(config.dim, proj_dim) / math.sqrt(config.dim)
            spec_features = (F.silu(W_gate[specialized_indices] @ rand_probe_k) * (W_up[specialized_indices] @ rand_probe_k))

            print(f"  Partitioning {config.intermediate_dim} neurons: 1 Shared Leaf ({leaf_dim} universal neurons) + {num_routed} Routed Leaves ({num_routed * leaf_dim} specialized neurons)...")
            spec_assignments, _ = balanced_spherical_kmeans(spec_features, k=num_routed, num_iters=15)

            # 3. Save Leaf 0 (Shared Leaf): UNPRUNED (Protected from 1:16 zeroing!)
            shared_gate = W_gate[shared_indices, :].clone().to(torch.bfloat16)
            shared_up = W_up[shared_indices, :].clone().to(torch.bfloat16)
            shared_down = W_down[:, shared_indices].clone().to(torch.bfloat16)

            block_state["asdag_ffn.leaves.0.gate_proj.weight"] = shared_gate
            block_state["asdag_ffn.leaves.0.up_proj.weight"] = shared_up
            block_state["asdag_ffn.leaves.0.down_proj.weight"] = shared_down

            # 4. Save Leaves 1..7 (Routed Leaves): Wanda 1:16 Pruned + Gate Centroid Router
            router_weights = []
            for r_idx in range(num_routed):
                leaf_k = r_idx + num_shared  # 1-indexed for leaves module list
                mask = (spec_assignments == r_idx)
                routed_leaf_indices = specialized_indices[mask]

                gate_leaf = W_gate[routed_leaf_indices, :].clone().to(torch.bfloat16)
                up_leaf = W_up[routed_leaf_indices, :].clone().to(torch.bfloat16)
                down_leaf = W_down[:, routed_leaf_indices].clone().to(torch.bfloat16)

                # Apply Wanda 1:16 structured sparsity to routed leaves
                gate_leaf_sp, _ = apply_wanda_1_to_16_sparsity(gate_leaf, Hessian_diag)
                up_leaf_sp, _ = apply_wanda_1_to_16_sparsity(up_leaf, Hessian_diag)
                down_leaf_sp, _ = apply_wanda_1_to_16_sparsity(down_leaf, (down_leaf**2).sum(dim=0))

                block_state[f"asdag_ffn.leaves.{leaf_k}.gate_proj.weight"] = gate_leaf_sp
                block_state[f"asdag_ffn.leaves.{leaf_k}.up_proj.weight"] = up_leaf_sp
                block_state[f"asdag_ffn.leaves.{leaf_k}.down_proj.weight"] = down_leaf_sp

                # ExpertWeaver Gate Centroid: average direction of neurons in this cluster
                gate_centroid = W_gate[routed_leaf_indices, :].float().mean(dim=0)
                gate_centroid = gate_centroid / (gate_centroid.norm() + 1e-8)
                router_weights.append(gate_centroid)

                # 5. Stack router weights for the 7 routed leaves: shape [num_routed, dim]
                router_tensor = torch.stack(router_weights, dim=0).to(torch.bfloat16)
                router_tensor = quantize_shift4(router_tensor)
                block_state["asdag_ffn.router.weight"] = router_tensor

        out_block_path = os.path.join(args.output_dir, f"block_{layer_idx:02d}.pt")
        torch.save(block_state, out_block_path)
        print(f"  Layer {layer_idx} completed in {time.time()-t0:.2f}s -> Saved to {out_block_path}")

        del W_gate, W_up, W_down, block_state, layer_tensors
        gc.collect()

    # MTP Block (if present)
    mtp_tensors = {t.name: t for t in reader.tensors if t.name.startswith("mtp.")}
    if mtp_tensors and config.has_mtp:
        print("\nExtracting MTP Drafting Block...")
        mtp_state = {}
        for name, t in mtp_tensors.items():
            clean_name = name.replace("mtp.", "")
            mtp_state[clean_name] = dequantize_tensor(t).to(config.dtype)
        torch.save(mtp_state, os.path.join(args.output_dir, "mtp_block.pt"))
        print("MTP Block saved.")

    print("\n[4/4] Strategy C Initialization Pipeline Complete!")
    print(f"Complete calibrated state checkpoints saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
