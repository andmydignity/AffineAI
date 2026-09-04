#!/usr/bin/env python3
"""
Upcycle Qwen3.5-4B GGUF to AffineAI ASDAG Architecture.
Implements native defaults:
  - Ternary Leaves {-1, 0, +1} * gamma
  - 4-bit Log-Shift Routing (Log4/Shift4)
  - BF16 Master Weights (preserved in memory during training, stripped in .toros)
  - Streaming .toros packing

Usage:
  python scripts/upcycle_qwen35.py --gguf-path /home/semih/Modeller/Qwen3.5-4B-Q4_K_M.gguf --force
"""

import os
import io
import sys
import gc
import json
import struct
import argparse
import time
import torch
import numpy as np

try:
    import gguf
    from gguf.quants import dequantize
except ImportError:
    raise ImportError("gguf package is required. Install via: pip install gguf")

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from affine_ai.core.ast_dag import ternarize, quantize_shift4
from affine_ai.core.format import (
    MAGIC_HEADER,
    FORMAT_VERSION,
    FLAG_RAW_FP16,
    FLAG_TERNARY_2BIT,
    FLAG_SPARSE_TERNARY,
    pack_ternary_tensor,
)
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
    Extracts and upcycles a single block from GGUF into native ASDAG format:
      1. Slices 9216 SwiGLU FFN into K ASDAG leaves.
      2. Ternarizes leaves to {-1, 0, +1} * gamma.
      3. Quantizes router to 4-bit log-shift representation.
    """
    tensors = {t.name: t for t in reader.tensors if t.name.startswith(f"blk.{layer_idx}.")}
    is_full_attn = (layer_idx == 32) or ((layer_idx + 1) % config.full_attn_interval == 0)

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

    # 4. Native ASDAG Tree FFN: Slice into K leaves + Ternarize
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

        # Apply native ternary quantization {-1, 0, 1} * gamma to leaves
        if config.ternary_leaves:
            gate_slice = ternarize(gate_slice)
            up_slice = ternarize(up_slice)
            down_slice = ternarize(down_slice)

        state_dict[f"asdag_ffn.leaves.{k}.gate_proj.weight"] = gate_slice
        state_dict[f"asdag_ffn.leaves.{k}.up_proj.weight"] = up_slice
        state_dict[f"asdag_ffn.leaves.{k}.down_proj.weight"] = down_slice

        # Compute leaf centroid
        leaf_centroid = gate_slice.float().mean(dim=0)
        leaf_centroid = leaf_centroid / (leaf_centroid.norm() + 1e-8)
        router_weights.append(leaf_centroid)

    router_tensor = torch.stack(router_weights, dim=0).to(config.dtype)
    # Apply 4-bit Log-Shift Quantization to router weights
    if config.use_shift4_routing:
        router_tensor = quantize_shift4(router_tensor)
    state_dict["asdag_ffn.router.weight"] = router_tensor

    return state_dict


def pack_all_to_toros(input_dir: str, output_file: str, config: Qwen35ASDAGConfig, compression_level: int = 3):
    """Streams all blocks into a single unified .toros binary checkpoint."""
    print(f"\n--- Packing to Unified .toros Checkpoint: {output_file} ---")
    cctx = zstd.ZstdCompressor(level=compression_level)
    temp_comp_path = output_file + ".tmp"

    total_tensors = 0
    total_params = 0

    token_embd_path = os.path.join(input_dir, "token_embd.pt")
    if os.path.exists(token_embd_path):
        t = torch.load(token_embd_path, weights_only=True)
        total_tensors += 1
        total_params += t.numel()
        del t

    output_norm_path = os.path.join(input_dir, "output_norm.pt")
    if os.path.exists(output_norm_path):
        t = torch.load(output_norm_path, weights_only=True)
        total_tensors += 1
        total_params += t.numel()
        del t

    for i in range(config.num_layers):
        bp = os.path.join(input_dir, f"block_{i:02d}.pt")
        if os.path.exists(bp):
            st = torch.load(bp, weights_only=True)
            total_tensors += len(st)
            total_params += sum(p.numel() for p in st.values())
            del st

    mtp_path = os.path.join(input_dir, "mtp_block.pt")
    if os.path.exists(mtp_path):
        st = torch.load(mtp_path, weights_only=True)
        total_tensors += len(st)
        total_params += sum(p.numel() for p in st.values())
        del st

    gc.collect()
    uncompressed_bytes = 0
    ternary_count = 0
    fp16_count = 0

    t0 = time.time()
    with open(temp_comp_path, "wb") as f_out:
        with cctx.stream_writer(f_out, closefd=False) as compressor:
            count_hdr = struct.pack("<I", total_tensors)
            compressor.write(count_hdr)
            uncompressed_bytes += len(count_hdr)

            def pack_and_write(name: str, tensor: torch.Tensor):
                nonlocal uncompressed_bytes, ternary_count, fp16_count
                name_bytes = name.encode("utf-8")

                is_ternary = (
                    tensor.dim() >= 2 and
                    ("leaves" in name or "asdag_ffn" in name) and
                    ("weight" in name) and
                    ("router" not in name) and
                    ("norm" not in name) and
                    ("conv" not in name)
                )

                if is_ternary:
                    packed_bytes, gamma, shape, flag = pack_ternary_tensor(tensor)
                    hdr = (
                        struct.pack("<H", len(name_bytes)) +
                        name_bytes +
                        struct.pack("<B", flag) +
                        struct.pack("<e", gamma) +
                        struct.pack("<B", len(shape)) +
                        b"".join(struct.pack("<I", d) for d in shape) +
                        struct.pack("<I", len(packed_bytes))
                    )
                    compressor.write(hdr)
                    compressor.write(packed_bytes)
                    uncompressed_bytes += len(hdr) + len(packed_bytes)
                    ternary_count += 1
                else:
                    t_fp16 = tensor.detach().cpu().to(torch.float16).contiguous()
                    t_bytes = t_fp16.numpy().tobytes()
                    shape = list(tensor.shape)
                    flag = FLAG_RAW_FP16

                    hdr = (
                        struct.pack("<H", len(name_bytes)) +
                        name_bytes +
                        struct.pack("<B", flag) +
                        struct.pack("<e", 1.0) +
                        struct.pack("<B", len(shape)) +
                        b"".join(struct.pack("<I", d) for d in shape) +
                        struct.pack("<I", len(t_bytes))
                    )
                    compressor.write(hdr)
                    compressor.write(t_bytes)
                    uncompressed_bytes += len(hdr) + len(t_bytes)
                    fp16_count += 1

            if os.path.exists(token_embd_path):
                t_emb = torch.load(token_embd_path, weights_only=True)
                pack_and_write("token_embd.weight", t_emb)
                del t_emb
                gc.collect()

            if os.path.exists(output_norm_path):
                t_norm = torch.load(output_norm_path, weights_only=True)
                pack_and_write("output_norm.weight", t_norm)
                del t_norm
                gc.collect()

            for i in range(config.num_layers):
                bp = os.path.join(input_dir, f"block_{i:02d}.pt")
                if not os.path.exists(bp):
                    continue
                block_st = torch.load(bp, weights_only=True)
                for k, v in block_st.items():
                    pack_and_write(f"blocks.{i}.{k}", v)
                del block_st
                gc.collect()

            if os.path.exists(mtp_path):
                mtp_st = torch.load(mtp_path, weights_only=True)
                for k, v in mtp_st.items():
                    if k.startswith(("eh_proj", "enorm", "hnorm", "shared_head_norm")):
                        pack_and_write(f"mtp_block.{k}", v)
                    else:
                        pack_and_write(f"mtp_block.block.{k}", v)
                del mtp_st
                gc.collect()

    compressed_bytes = os.path.getsize(temp_comp_path)
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None
    }
    meta = {
        "format": "TOROS",
        "version": FORMAT_VERSION,
        "model_type": "Qwen35ASDAGModel",
        "config": config_dict,
        "sparsity": {
            "num_leaves": config.num_leaves,
            "top_k": config.top_k,
            "leaf_dim": config.leaf_dim,
            "active_sparsity": f"{config.top_k}/{config.num_leaves} (75% zero compute)"
        },
        "statistics": {
            "total_params": total_params,
            "ternary_tensors": ternary_count,
            "fp16_tensors": fp16_count,
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": compressed_bytes
        }
    }
    meta_json = json.dumps(meta, indent=2).encode("utf-8")

    with open(output_file, "wb") as f_final:
        f_final.write(MAGIC_HEADER)
        f_final.write(struct.pack("<B", FORMAT_VERSION))
        f_final.write(struct.pack("<B", 1))
        f_final.write(struct.pack("<I", len(meta_json)))
        f_final.write(meta_json)
        f_final.write(struct.pack("<Q", uncompressed_bytes))
        f_final.write(struct.pack("<Q", compressed_bytes))

        with open(temp_comp_path, "rb") as f_tmp:
            while True:
                chunk = f_tmp.read(32 * 1024 * 1024)
                if not chunk:
                    break
                f_final.write(chunk)

    if os.path.exists(temp_comp_path):
        os.remove(temp_comp_path)

    final_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Packed .toros generated: {output_file} ({final_size_mb:,.1f} MB) in {time.time() - t0:.2f}s!")


def main():
    parser = argparse.ArgumentParser(description="Upcycle Qwen3.5-4B GGUF to AffineAI ASDAG")
    parser.add_argument("--gguf-path", type=str, default="/home/semih/Modeller/Qwen3.5-4B-Q4_K_M.gguf")
    parser.add_argument("--output-dir", type=str, default="checkpoints/qwen35_asdag")
    parser.add_argument("--toros-file", type=str, default="checkpoints/qwen35_asdag.toros")
    parser.add_argument("--num-leaves", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=None, help="Limit layers for fast trial conversion")
    parser.add_argument("--force", action="store_true", help="Force regenerate existing block files")
    args = parser.parse_args()

    print(f"Opening GGUF file: {args.gguf_path}")
    reader = gguf.GGUFReader(args.gguf_path)
    print(f"Successfully loaded GGUF with {len(reader.tensors)} tensors.")

    config = Qwen35ASDAGConfig(
        num_leaves=args.num_leaves,
        top_k=args.top_k,
        leaf_dim=9216 // args.num_leaves,
        dtype=torch.bfloat16,
        ternary_leaves=True,
        use_shift4_routing=True,
        use_fp8_hybrid=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    num_layers = args.max_layers if args.max_layers is not None else config.num_layers
    print(f"\nUpcycling {num_layers} layers into Native ASDAG:")
    print(f"  - Leaves: Ternary {{-1, 0, 1}} * gamma (2-bit packed)")
    print(f"  - Router: 4-bit Log-Shift (Log4/Shift4)")
    print(f"  - Master Weights: BF16 (not saved in .toros)")
    print(f"  - Slicing: {config.num_leaves} leaves x {config.leaf_dim} channels, Top-{config.top_k} active\n")

    t0 = time.time()
    for layer_idx in range(num_layers):
        out_file = os.path.join(args.output_dir, f"block_{layer_idx:02d}.pt")
        if os.path.exists(out_file) and not args.force:
            print(f"  Layer {layer_idx:02d}: already converted at {out_file}, skipping...")
            continue
        layer_t0 = time.time()
        block_state = extract_block_state_dict(reader, layer_idx, config)
        torch.save(block_state, out_file)
        dt = time.time() - layer_t0
        is_attn = (layer_idx == 32) or ((layer_idx + 1) % config.full_attn_interval == 0)
        layer_type = "Gated Attention" if is_attn else "Gated DeltaNet"
        print(f"  Layer {layer_idx:02d} ({layer_type}): converted & saved in {dt:.2f}s -> {out_file}")

    # Extract MTP block if present
    has_mtp = any(t.name.startswith("blk.32.") for t in reader.tensors)
    if has_mtp and (args.max_layers is None or args.max_layers >= 32):
        mtp_out = os.path.join(args.output_dir, "mtp_block.pt")
        if not os.path.exists(mtp_out) or args.force:
            print("\nExtracting MTP (Multi-Token Prediction) Block 32...")
            mtp_t0 = time.time()
            mtp_state = extract_block_state_dict(reader, 32, config)
            
            tensors_32 = {t.name: t for t in reader.tensors if t.name.startswith("blk.32.")}
            mtp_state["eh_proj.weight"] = dequantize_tensor(tensors_32["blk.32.nextn.eh_proj.weight"]).to(config.dtype)
            mtp_state["enorm.weight"] = dequantize_tensor(tensors_32["blk.32.nextn.enorm.weight"]).to(config.dtype)
            mtp_state["hnorm.weight"] = dequantize_tensor(tensors_32["blk.32.nextn.hnorm.weight"]).to(config.dtype)
            mtp_state["shared_head_norm.weight"] = dequantize_tensor(tensors_32["blk.32.nextn.shared_head_norm.weight"]).to(config.dtype)
            
            torch.save(mtp_state, mtp_out)
            print(f"  MTP block saved in {time.time() - mtp_t0:.2f}s -> {mtp_out}")

    # Extract global weights
    print("\nExtracting global weights...")
    token_embd_path = os.path.join(args.output_dir, "token_embd.pt")
    if not os.path.exists(token_embd_path) or args.force:
        token_embd_t = [t for t in reader.tensors if t.name == "token_embd.weight"][0]
        token_embd = dequantize_tensor(token_embd_t).to(config.dtype)
        torch.save(token_embd, token_embd_path)
        print(f"  Token embedding saved: shape={token_embd.shape}")

    output_norm_path = os.path.join(args.output_dir, "output_norm.pt")
    if not os.path.exists(output_norm_path) or args.force:
        output_norm_t = [t for t in reader.tensors if t.name == "output_norm.weight"][0]
        output_norm = dequantize_tensor(output_norm_t).to(config.dtype)
        torch.save(output_norm, output_norm_path)
        print(f"  Output norm saved: shape={output_norm.shape}")

    total_time = time.time() - t0
    print(f"\nUpcycling extraction complete in {total_time:.2f}s!")

    # Pack to unified .toros
    if args.max_layers is None or args.max_layers >= config.num_layers:
        pack_all_to_toros(args.output_dir, args.toros_file, config)


if __name__ == "__main__":
    main()
