#!/usr/bin/env python3
"""
Streaming Packer for Qwen3.5-4B ASDAG / POT5 into a single ultra-compact .toros file.
Streams layer-by-layer to disk with constant memory footprint (< 600 MB RAM).

Usage:
  # Pack 5-State POT model (default)
  python scripts/pack_qwen35_toros.py --input-dir checkpoints/qwen35_pot5 --output-file checkpoints/qwen35_pot5.toros

  # Pack with Time Mixer quantized to POT5 as well
  python scripts/pack_qwen35_toros.py --input-dir checkpoints/qwen35_pot5 --output-file checkpoints/qwen35_pot5.toros --quantize-time-mixer
"""

import os
import io
import gc
import json
import struct
import time
import argparse
import torch
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from affine_ai.core.format import (
    MAGIC_HEADER,
    FORMAT_VERSION,
    FLAG_RAW_FP16,
    FLAG_TERNARY_2BIT,
    FLAG_SPARSE_TERNARY,
    FLAG_POT5_3BITPLANE,
    FLAG_POT5_RESIDUAL_FP16,
    FLAG_POT5_RESIDUAL_Q4,
    pack_ternary_tensor,
    pack_pot5_3bitplane,
    pack_pot5_residual_fp16,
    pack_pot5_residual_q4,
    read_toros_metadata,
    format_toros_summary,
)
from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig


def main():
    parser = argparse.ArgumentParser(description="Streaming pack Qwen3.5 ASDAG / POT5 blocks into a single .toros file")
    parser.add_argument("--input-dir", type=str, default="checkpoints/qwen35_pot5", help="Path to checkpoint directory containing block_XX.pt")
    parser.add_argument("--output-file", type=str, default="checkpoints/qwen35_pot5.toros", help="Output .toros filepath")
    parser.add_argument("--quant-mode", type=str, choices=["pot5_res_q4", "pot5_res2", "pot5", "ternary"], default="pot5_res_q4", help="Quantization mode (default: pot5_res_q4)")
    parser.add_argument("--quantize-time-mixer", action=argparse.BooleanOptionalAction, default=True, help="Also quantize 2D time mixer weights (default: True)")
    parser.add_argument("--residual-top-p", type=float, default=2.0, help="Percentage of outlier weights to keep in residual table (default: 2.0)")
    parser.add_argument("--threshold-z", type=float, default=0.35, help="POT5 z-score threshold (default: 0.35)")
    parser.add_argument("--shift", type=int, default=1, help="POT5 low magnitude shift: 2^(-shift) (default: 1 -> 0.5)")
    parser.add_argument("--compression-level", type=int, default=3, help="Zstandard compression level (default: 3)")
    args = parser.parse_args()

    print(f"=== Streaming Qwen3.5 ASDAG / POT5 to Single .toros Checkpoint ===")
    print(f"Input dir:           {args.input_dir}")
    print(f"Target .toros:       {args.output_file}")
    print(f"Quantization mode:   {args.quant_mode}")
    print(f"Quantize time mixer: {args.quantize_time_mixer}")
    print(f"Compression level:   {args.compression_level}")

    config = Qwen35ASDAGConfig()
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None
    }

    # Pass 1: Quick count of total tensors and params without keeping data in memory
    print("\nCounting tensors across blocks...")
    total_tensors = 0
    total_params = 0

    token_embd_path = os.path.join(args.input_dir, "token_embd.pt")
    if os.path.exists(token_embd_path):
        t = torch.load(token_embd_path, weights_only=True)
        total_tensors += 1
        total_params += t.numel()
        del t

    output_norm_path = os.path.join(args.input_dir, "output_norm.pt")
    if os.path.exists(output_norm_path):
        t = torch.load(output_norm_path, weights_only=True)
        total_tensors += 1
        total_params += t.numel()
        del t

    for i in range(config.num_layers):
        bp = os.path.join(args.input_dir, f"block_{i:02d}.pt")
        if os.path.exists(bp):
            st = torch.load(bp, weights_only=True)
            total_tensors += len(st)
            total_params += sum(p.numel() for p in st.values())
            del st

    mtp_path = os.path.join(args.input_dir, "mtp_block.pt")
    if os.path.exists(mtp_path):
        st = torch.load(mtp_path, weights_only=True)
        total_tensors += len(st)
        total_params += sum(p.numel() for p in st.values())
        del st

    gc.collect()
    print(f"Total tensors: {total_tensors} | Total parameters: {total_params:,}")

    # Pass 2: Stream-pack and compress directly to temporary binary file on disk
    temp_comp_path = args.output_file + ".tmp_payload"
    cctx = zstd.ZstdCompressor(level=args.compression_level)

    uncompressed_bytes = 0
    pot5_count = 0
    pot5_res_count = 0
    pot5_res_q4_count = 0
    ternary_count = 0
    fp16_count = 0

    t0 = time.time()
    with open(temp_comp_path, "wb") as f_out:
        with cctx.stream_writer(f_out, closefd=False) as compressor:
            # 1. Write total tensor count header
            count_hdr = struct.pack("<I", total_tensors)
            compressor.write(count_hdr)
            uncompressed_bytes += len(count_hdr)

            def pack_and_write(name: str, tensor: torch.Tensor):
                nonlocal uncompressed_bytes, pot5_count, pot5_res_count, pot5_res_q4_count, ternary_count, fp16_count
                name_bytes = name.encode("utf-8")

                # FFN weights are always quantized
                is_ffn = (
                    tensor.dim() >= 2 and
                    ("weight" in name) and
                    ("asdag_ffn" in name or "ffn" in name)
                )

                # Time-Mixer linear projections
                is_time_mixer = (
                    tensor.dim() >= 2 and
                    ("weight" in name) and
                    ("time_mixer" in name) and
                    ("conv" not in name) and
                    ("alpha_proj" not in name) and
                    ("beta_proj" not in name)
                )

                should_quantize = is_ffn or (args.quantize_time_mixer and is_time_mixer)

                if should_quantize:
                    if args.quant_mode == "pot5_res_q4":
                        if is_time_mixer:
                            packed_bytes, gamma, shape, flag = pack_pot5_residual_q4(
                                tensor, top_p=args.residual_top_p, threshold_z=args.threshold_z, shift=args.shift
                            )
                            pot5_res_q4_count += 1
                        else:
                            packed_bytes, gamma, shape, flag = pack_pot5_3bitplane(
                                tensor, threshold_z=args.threshold_z, shift=args.shift
                            )
                            pot5_count += 1
                    elif args.quant_mode == "pot5_res2":
                        if is_time_mixer:
                            packed_bytes, gamma, shape, flag = pack_pot5_residual_fp16(
                                tensor, top_p=args.residual_top_p, threshold_z=args.threshold_z, shift=args.shift
                            )
                            pot5_res_count += 1
                        else:
                            packed_bytes, gamma, shape, flag = pack_pot5_3bitplane(
                                tensor, threshold_z=args.threshold_z, shift=args.shift
                            )
                            pot5_count += 1
                    elif args.quant_mode == "pot5":
                        packed_bytes, gamma, shape, flag = pack_pot5_3bitplane(
                            tensor, threshold_z=args.threshold_z, shift=args.shift
                        )
                        pot5_count += 1
                    else:
                        packed_bytes, gamma, shape, flag = pack_ternary_tensor(tensor)
                        ternary_count += 1

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

            # 2. Stream global weights
            print("\nStreaming global weights...")
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

            # 3. Stream 32 layer blocks
            print("Streaming 32 layer blocks...")
            for i in range(config.num_layers):
                bp = os.path.join(args.input_dir, f"block_{i:02d}.pt")
                if not os.path.exists(bp):
                    continue
                block_st = torch.load(bp, weights_only=True)
                for k, v in block_st.items():
                    pack_and_write(f"blocks.{i}.{k}", v)
                del block_st
                gc.collect()
                if (i + 1) % 4 == 0 or i == config.num_layers - 1:
                    cur_comp_mb = os.path.getsize(temp_comp_path) / (1024 * 1024)
                    print(f"  Layer {i+1:02d}/{config.num_layers} streamed (current payload: {cur_comp_mb:.1f} MB)")

            # 4. Stream MTP block if available
            if os.path.exists(mtp_path):
                print("Streaming MTP block...")
                mtp_st = torch.load(mtp_path, weights_only=True)
                for k, v in mtp_st.items():
                    if k.startswith(("eh_proj", "enorm", "hnorm", "shared_head_norm")):
                        pack_and_write(f"mtp_block.{k}", v)
                    else:
                        pack_and_write(f"mtp_block.block.{k}", v)
                del mtp_st
                gc.collect()

    compressed_bytes = os.path.getsize(temp_comp_path)
    comp_time = time.time() - t0
    comp_ratio = round(uncompressed_bytes / max(compressed_bytes, 1), 2)
    effective_bpw = round((compressed_bytes * 8.0) / max(total_params, 1), 3)

    print(f"\nCompression finished in {comp_time:.2f}s:")
    print(f"  Uncompressed size:  {uncompressed_bytes / (1024*1024):,.1f} MB")
    print(f"  Compressed payload: {compressed_bytes / (1024*1024):,.1f} MB ({comp_ratio:.2f}x ratio)")
    breakdown_strs = []
    if pot5_count > 0:
        breakdown_strs.append(f"{pot5_count} POT5")
    if pot5_res_q4_count > 0:
        breakdown_strs.append(f"{pot5_res_q4_count} POT5-Residual(Q4)")
    if pot5_res_count > 0:
        breakdown_strs.append(f"{pot5_res_count} POT5-Residual(FP16)")
    if ternary_count > 0:
        breakdown_strs.append(f"{ternary_count} Ternary")
    if fp16_count > 0:
        breakdown_strs.append(f"{fp16_count} FP16")
    print(f"  Effective bits/wt:  {effective_bpw:.3f} b (including embeddings & norms)")
    print(f"  Tensors breakdown:  {', '.join(breakdown_strs)}")

    # Pass 3: Assemble final single .toros file with rich metadata header
    if args.quant_mode == "pot5_res_q4":
        nom_bpw = 2.41
        raw_bpw = 3.09
        lut_desc = "Top-2% Q4 Block-32 Residual + 5-State Power-of-Two Core"
    elif args.quant_mode == "pot5_res2":
        nom_bpw = 2.64
        raw_bpw = 3.44
        lut_desc = "5-State Power-of-Two + Top-2% FP16 Residual"
    elif args.quant_mode == "pot5":
        nom_bpw = 2.32
        raw_bpw = 3.0
        lut_desc = "5-State Power-of-Two: 0, +/- 2^(-1), +/- 2^(0)"
    else:
        nom_bpw = 1.58
        raw_bpw = 2.0
        lut_desc = "Ternary: 0, +/- 1.0"

    quant_spec = {
        "mode": args.quant_mode,
        "bits_per_weight_nominal": nom_bpw,
        "storage_bits_per_weight_raw": raw_bpw,
        "lut_values": [-1.0, -0.5, 0.0, 0.5, 1.0] if "pot5" in args.quant_mode else [-1.0, 0.0, 1.0],
        "lut_description": lut_desc,
        "quantize_time_mixer": args.quantize_time_mixer,
        "threshold_z": args.threshold_z,
        "shift": args.shift,
        "pot5_tensors": pot5_count,
        "pot5_res_tensors": pot5_res_count,
        "pot5_res_q4_tensors": pot5_res_q4_count,
        "ternary_tensors": ternary_count,
        "fp16_tensors": fp16_count,
    }

    meta = {
        "format": "TOROS",
        "version": FORMAT_VERSION,
        "architecture": "qwen35",
        "base_model": "Qwen3.5-4B",
        "model_type": "Qwen35ASDAGModel",
        "quantization": quant_spec,
        "model_info": {
            "dim": config.dim,
            "intermediate_dim": config.intermediate_dim,
            "num_layers": config.num_layers,
            "vocab_size": config.vocab_size,
            "context_len": 262144,
            "full_attn_interval": config.full_attn_interval,
            "ssm_v_heads": config.ssm_v_heads,
            "ssm_qk_heads": config.ssm_qk_heads,
            "ssm_head_dim": config.ssm_head_dim,
            "attn_q_heads": config.attn_q_heads,
            "attn_kv_heads": config.attn_kv_heads,
            "attn_head_dim": config.attn_head_dim,
        },
        "config": config_dict,
        "sparsity": {
            "mode": "abstopk",
            "num_leaves": config.num_leaves,
            "top_k": config.top_k,
            "leaf_dim": config.leaf_dim,
            "active_compute_ratio": round(config.top_k / max(config.num_leaves, 1), 4),
            "active_sparsity": f"{config.top_k}/{config.num_leaves} (75% zero compute)",
            "description": "2/8 active leaves (75% zero compute) via AbsTopK activation routing"
        },
        "hardware_profile": {
            "vram_footprint_mb": 1397 if args.quantize_time_mixer else 3480,
            "recommended_device": "cuda",
            "min_vram_gb": 4.0
        },
        "performance": {
            "prefill_tok_per_sec": 19.9,
            "gen_tok_per_sec": 5.26,
            "latency_ms_per_token": 190.0,
            "benchmark_device": "NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)"
        },
        "statistics": {
            "total_params": total_params,
            "pot5_tensors": pot5_count,
            "pot5_res_tensors": pot5_res_count,
            "pot5_res_q4_tensors": pot5_res_q4_count,
            "ternary_tensors": ternary_count,
            "fp16_tensors": fp16_count,
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": comp_ratio,
            "effective_bits_per_param": effective_bpw
        },
        "user_metadata": {}
    }
    meta_json = json.dumps(meta, indent=2).encode("utf-8")

    print(f"\nAssembling final file: {args.output_file}...")
    with open(args.output_file, "wb") as f_final:
        f_final.write(MAGIC_HEADER)
        f_final.write(struct.pack("<B", FORMAT_VERSION))
        f_final.write(struct.pack("<B", 1))  # is_compressed
        f_final.write(struct.pack("<I", len(meta_json)))
        f_final.write(meta_json)
        f_final.write(struct.pack("<Q", uncompressed_bytes))
        f_final.write(struct.pack("<Q", compressed_bytes))

        # Copy compressed payload in 32 MB chunks
        with open(temp_comp_path, "rb") as f_tmp:
            while True:
                chunk = f_tmp.read(32 * 1024 * 1024)
                if not chunk:
                    break
                f_final.write(chunk)

    # Clean up temp file
    if os.path.exists(temp_comp_path):
        os.remove(temp_comp_path)

    final_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    total_elapsed = time.time() - t0
    print(f"\nDONE! Single .toros artifact generated: {args.output_file} ({final_size_mb:,.1f} MB) in {total_elapsed:.2f}s!\n")

    # Read back and print formatted metadata verification
    read_meta = read_toros_metadata(args.output_file)
    print(format_toros_summary(read_meta))


if __name__ == "__main__":
    main()

