#!/usr/bin/env python3
"""
Upcycle Qwen3.5-4B GGUF to AffineAI ASDAG Architecture.

Usage:
  python scripts/upcycle_qwen35.py --gguf-path /home/semih/Modeller/Qwen3.5-4B-Q4_K_M.gguf --max-layers 2
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

try:
    import gguf
    from gguf.quants import dequantize
except ImportError:
    raise ImportError("gguf package is required. Install via: pip install gguf")

from affine_ai.models.qwen35_asdag import (
    Qwen35ASDAGConfig,
    Qwen35ASDAGModel,
    Qwen35Block,
    Qwen35MTPBlock,
)


def dequantize_tensor(tensor) -> torch.Tensor:
    """Dequantize a GGUF tensor to a PyTorch float32 tensor."""
    arr = dequantize(tensor.data, tensor.tensor_type)
    return torch.from_numpy(arr.copy()).float()


def extract_block_state_dict(reader, layer_idx: int, config: Qwen35ASDAGConfig) -> dict:
    """
    Extracts and upcycles a single block from GGUF into ASDAG format.
    Slices the 9216 SwiGLU FFN into K ASDAG leaves.
    """
    tensors = {t.name: t for t in reader.tensors if t.name.startswith(f"blk.{layer_idx}.")}
    is_full_attn = ((layer_idx + 1) % config.full_attn_interval == 0)

    state_dict = {}

    # 1. Attn Norm
    attn_norm = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_norm.weight"])
    state_dict["attn_norm.weight"] = attn_norm.to(config.dtype)

    # 2. Time Mixer
    if is_full_attn:
        # Gated Attention
        q_w = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_q.weight"])
        k_w = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_k.weight"])
        v_w = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_v.weight"])
        q_norm = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_q_norm.weight"])
        k_norm = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_k_norm.weight"])
        out_w = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_output.weight"])

        state_dict["time_mixer.attn_q.weight"] = q_w.to(config.dtype)
        state_dict["time_mixer.attn_k.weight"] = k_w.to(config.dtype)
        state_dict["time_mixer.attn_v.weight"] = v_w.to(config.dtype)
        state_dict["time_mixer.q_norm.weight"] = q_norm.to(config.dtype)
        state_dict["time_mixer.k_norm.weight"] = k_norm.to(config.dtype)
        state_dict["time_mixer.attn_output.weight"] = out_w.to(config.dtype)
    else:
        # Gated DeltaNet
        qkv_w = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_qkv.weight"])
        conv_w = dequantize_tensor(tensors[f"blk.{layer_idx}.ssm_conv1d.weight"])  # [8192, 4]
        alpha_w = dequantize_tensor(tensors[f"blk.{layer_idx}.ssm_alpha.weight"]) # [32, 2560]
        beta_w = dequantize_tensor(tensors[f"blk.{layer_idx}.ssm_beta.weight"])   # [32, 2560]
        ssm_a = dequantize_tensor(tensors[f"blk.{layer_idx}.ssm_a"])             # [32]
        ssm_dt = dequantize_tensor(tensors[f"blk.{layer_idx}.ssm_dt.bias"])      # [32]
        ssm_norm = dequantize_tensor(tensors[f"blk.{layer_idx}.ssm_norm.weight"]) # [128]
        gate_w = dequantize_tensor(tensors[f"blk.{layer_idx}.attn_gate.weight"]) # [4096, 2560]
        ssm_out = dequantize_tensor(tensors[f"blk.{layer_idx}.ssm_out.weight"])   # [2560, 4096]

        state_dict["time_mixer.qkv_proj.weight"] = qkv_w.to(config.dtype)
        # Conv1d weights in PyTorch have shape [out_channels, in_channels/groups, kernel_size] -> [8192, 1, 4]
        state_dict["time_mixer.conv1d.weight"] = conv_w.unsqueeze(1).to(config.dtype)
        state_dict["time_mixer.alpha_proj.weight"] = alpha_w.to(torch.float32)
        state_dict["time_mixer.beta_proj.weight"] = beta_w.to(torch.float32)
        state_dict["time_mixer.ssm_a"] = ssm_a.to(torch.float32)
        state_dict["time_mixer.ssm_dt_bias"] = ssm_dt.to(torch.float32)
        state_dict["time_mixer.norm.weight"] = ssm_norm.to(config.dtype)
        state_dict["time_mixer.attn_gate.weight"] = gate_w.to(config.dtype)
        state_dict["time_mixer.ssm_out.weight"] = ssm_out.to(config.dtype)

    # 3. Post-Attention Norm
    post_norm = dequantize_tensor(tensors[f"blk.{layer_idx}.post_attention_norm.weight"])
    state_dict["post_attention_norm.weight"] = post_norm.to(config.dtype)

    # 4. ASDAG Tree FFN: Slice 9216 intermediate channels into K leaves
    ffn_gate = dequantize_tensor(tensors[f"blk.{layer_idx}.ffn_gate.weight"])  # [9216, 2560]
    ffn_up = dequantize_tensor(tensors[f"blk.{layer_idx}.ffn_up.weight"])      # [9216, 2560]
    ffn_down = dequantize_tensor(tensors[f"blk.{layer_idx}.ffn_down.weight"])  # [2560, 9216]

    num_leaves = config.num_leaves
    leaf_dim = config.leaf_dim

    router_weights = []
    for k in range(num_leaves):
        start = k * leaf_dim
        end = (k + 1) * leaf_dim

        gate_slice = ffn_gate[start:end, :].to(config.dtype)
        up_slice = ffn_up[start:end, :].to(config.dtype)
        down_slice = ffn_down[:, start:end].to(config.dtype)

        state_dict[f"asdag_ffn.leaves.{k}.gate_proj.weight"] = gate_slice
        state_dict[f"asdag_ffn.leaves.{k}.up_proj.weight"] = up_slice
        state_dict[f"asdag_ffn.leaves.{k}.down_proj.weight"] = down_slice

        # Initialize router centroid as normalized mean gate direction for this leaf
        leaf_centroid = gate_slice.float().mean(dim=0)
        leaf_centroid = leaf_centroid / (leaf_centroid.norm() + 1e-8)
        router_weights.append(leaf_centroid)

    # Router weight: [num_leaves, dim]
    state_dict["asdag_ffn.router.weight"] = torch.stack(router_weights, dim=0).to(config.dtype)

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Upcycle Qwen3.5-4B GGUF to AffineAI ASDAG")
    parser.add_argument("--gguf-path", type=str, default="/home/semih/Modeller/Qwen3.5-4B-Q4_K_M.gguf")
    parser.add_argument("--output-dir", type=str, default="checkpoints/qwen35_asdag")
    parser.add_argument("--num-leaves", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=None, help="Limit layers for fast trial conversion")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Opening GGUF file: {args.gguf_path}")
    reader = gguf.GGUFReader(args.gguf_path)
    print(f"Successfully loaded GGUF with {len(reader.tensors)} tensors.")

    config = Qwen35ASDAGConfig(
        num_leaves=args.num_leaves,
        top_k=args.top_k,
        leaf_dim=9216 // args.num_leaves,
        dtype=torch.bfloat16
    )

    os.makedirs(args.output_dir, exist_ok=True)
    num_layers = args.max_layers if args.max_layers is not None else config.num_layers
    print(f"\nUpcycling {num_layers} layers into ASDAG (leaves={config.num_leaves}, top_k={config.top_k}, leaf_dim={config.leaf_dim})...")

    t0 = time.time()
    for layer_idx in range(num_layers):
        layer_t0 = time.time()
        block_state = extract_block_state_dict(reader, layer_idx, config)
        out_file = os.path.join(args.output_dir, f"block_{layer_idx:02d}.pt")
        torch.save(block_state, out_file)
        dt = time.time() - layer_t0
        is_attn = ((layer_idx + 1) % config.full_attn_interval == 0)
        layer_type = "Gated Attention" if is_attn else "Gated DeltaNet"
        print(f"  Layer {layer_idx:02d} ({layer_type}): converted & saved in {dt:.2f}s -> {out_file}")

    # Extract global weights if doing full model or first trial
    print("\nExtracting global weights (Token Embedding & Output Norm)...")
    token_embd_t = [t for t in reader.tensors if t.name == "token_embd.weight"][0]
    token_embd = dequantize_tensor(token_embd_t).to(config.dtype)
    torch.save(token_embd, os.path.join(args.output_dir, "token_embd.pt"))
    print(f"  Token embedding saved: shape={token_embd.shape}")

    output_norm_t = [t for t in reader.tensors if t.name == "output_norm.weight"][0]
    output_norm = dequantize_tensor(output_norm_t).to(config.dtype)
    torch.save(output_norm, os.path.join(args.output_dir, "output_norm.pt"))
    print(f"  Output norm saved: shape={output_norm.shape}")

    total_time = time.time() - t0
    print(f"\nUpcycling complete in {total_time:.2f}s! Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
