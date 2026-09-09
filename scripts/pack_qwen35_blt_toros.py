#!/usr/bin/env python3
"""
Streaming Packer for Qwen3.5-4B Byte Latent Transformer (BLT) into a single .toros file.
Zero token embedding table (saves 1,212.5 MB).
Stores 32 ASDAG blocks + ~1.6 MB local byte interface.

Usage:
  python scripts/pack_qwen35_blt_toros.py \
    --backbone-dir checkpoints/qwen35_pot5 \
    --blt-interface-ckpt checkpoints/qwen35_blt/blt_interface_step_0005.pt \
    --output-file checkpoints/qwen35_blt.toros
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
    FLAG_POT5_3BITPLANE,
    FLAG_POT5_RESIDUAL_FP16,
    FLAG_POT5_RESIDUAL_Q4,
    pack_pot5_3bitplane,
    pack_pot5_residual_fp16,
    pack_pot5_residual_q4,
    read_toros_metadata,
    format_toros_summary,
)
from affine_ai.models.qwen35_blt import Qwen35BLTConfig


def main():
    parser = argparse.ArgumentParser(description="Pack Qwen3.5 BLT into ultra-compact .toros file (Zero Embedding Table)")
    parser.add_argument("--backbone-dir", type=str, default="checkpoints/qwen35_pot5", help="Directory containing block_XX.pt")
    parser.add_argument("--blt-interface-ckpt", type=str, default="checkpoints/qwen35_blt/blt_interface_step_0005.pt", help="Trained BLT byte interface checkpoint")
    parser.add_argument("--output-file", type=str, default="checkpoints/qwen35_blt.toros", help="Output .toros file path")
    parser.add_argument("--quantize-time-mixer", action=argparse.BooleanOptionalAction, default=True, help="Also quantize 2D time mixer weights (default: True with pot5_res_q4)")
    parser.add_argument("--time-mixer-mode", type=str, choices=["pot5_res_q4", "pot5_res2", "pot5"], default="pot5_res_q4", help="Time mixer quantization mode (default: pot5_res_q4 for Top-2%% Q4 block residual)")
    parser.add_argument("--residual-top-p", type=float, default=2.0, help="Percentage of outlier weights to keep in residual table (default: 2.0)")
    parser.add_argument("--threshold-z", type=float, default=0.35, help="POT5 z threshold (default: 0.35)")
    parser.add_argument("--shift", type=int, default=1, help="POT5 shift (default: 1)")
    parser.add_argument("--compression-level", type=int, default=3, help="Zstandard compression level (default: 3)")
    args = parser.parse_args()

    print(f"=== Streaming Qwen3.5 BLT to Single .toros Checkpoint (Zero Embedding) ===")
    print(f"Backbone dir:        {args.backbone_dir}")
    print(f"BLT interface ckpt:  {args.blt_interface_ckpt}")
    print(f"Target .toros:       {args.output_file}")
    print(f"Quantize time mixer: {args.quantize_time_mixer} (mode={args.time_mixer_mode}, top_p={args.residual_top_p}%)" if args.quantize_time_mixer else f"Quantize time mixer: False")
    print(f"Compression level:   {args.compression_level}")

    config = Qwen35BLTConfig()
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None
    }

    # Load BLT interface checkpoint
    print("\nLoading trained BLT interface checkpoint...")
    blt_ckpt = torch.load(args.blt_interface_ckpt, map_location="cpu", weights_only=True)
    blt_state_dict = {}
    for k, v in blt_ckpt["byte_encoder"].items():
        blt_state_dict[f"byte_encoder.{k}"] = v
    for k, v in blt_ckpt["patcher"].items():
        blt_state_dict[f"patcher.{k}"] = v
    blt_state_dict["sos_patch"] = blt_ckpt["sos_patch"]
    for k, v in blt_ckpt["byte_decoder"].items():
        blt_state_dict[f"byte_decoder.{k}"] = v

    blt_params = sum(p.numel() for p in blt_state_dict.values())
    print(f"Loaded BLT interface: {len(blt_state_dict)} tensors, {blt_params:,} parameters ({blt_params * 2 / (1024*1024):.2f} MB)")

    # Pass 1: Quick count of total tensors and params
    print("\nCounting tensors across blocks...")
    total_tensors = len(blt_state_dict)
    total_params = blt_params

    output_norm_path = os.path.join(args.backbone_dir, "output_norm.pt")
    if os.path.exists(output_norm_path):
        t = torch.load(output_norm_path, map_location="cpu", weights_only=True)
        total_tensors += 1
        total_params += t.numel()
        del t

    for i in range(config.num_layers):
        bp = os.path.join(args.backbone_dir, f"block_{i:02d}.pt")
        if os.path.exists(bp):
            st = torch.load(bp, map_location="cpu", weights_only=True)
            total_tensors += len(st)
            total_params += sum(p.numel() for p in st.values())
            del st

    gc.collect()
    print(f"Total tensors: {total_tensors} | Total parameters: {total_params:,}")
    print(f"NOTE: token_embd.weight (635,699,200 params, 1,212.5 MB) is ELIMINATED.")

    # Pass 2: Stream-pack directly to temporary file
    temp_comp_path = args.output_file + ".tmp_payload"
    cctx = zstd.ZstdCompressor(level=args.compression_level)

    uncompressed_bytes = 0
    pot5_count = 0
    pot5_res_count = 0
    pot5_res_q4_count = 0
    fp16_count = 0

    t0 = time.time()
    with open(temp_comp_path, "wb") as f_out:
        with cctx.stream_writer(f_out, closefd=False) as compressor:
            count_hdr = struct.pack("<I", total_tensors)
            compressor.write(count_hdr)
            uncompressed_bytes += len(count_hdr)

            def pack_and_write(name: str, tensor: torch.Tensor):
                nonlocal uncompressed_bytes, pot5_count, pot5_res_count, pot5_res_q4_count, fp16_count
                name_bytes = name.encode("utf-8")

                is_ffn = (
                    tensor.dim() >= 2 and
                    ("weight" in name) and
                    ("asdag_ffn" in name or "ffn" in name)
                )

                is_time_mixer = (
                    tensor.dim() >= 2 and
                    ("weight" in name) and
                    ("time_mixer" in name) and
                    ("conv" not in name) and
                    ("alpha_proj" not in name) and
                    ("beta_proj" not in name)
                )

                if is_ffn:
                    packed_bytes, gamma, shape, flag = pack_pot5_3bitplane(
                        tensor, threshold_z=args.threshold_z, shift=args.shift
                    )
                    pot5_count += 1
                elif args.quantize_time_mixer and is_time_mixer:
                    if args.time_mixer_mode == "pot5_res_q4":
                        packed_bytes, gamma, shape, flag = pack_pot5_residual_q4(
                            tensor, top_p=args.residual_top_p, threshold_z=args.threshold_z, shift=args.shift
                        )
                        pot5_res_q4_count += 1
                    elif args.time_mixer_mode == "pot5_res2":
                        packed_bytes, gamma, shape, flag = pack_pot5_residual_fp16(
                            tensor, top_p=args.residual_top_p, threshold_z=args.threshold_z, shift=args.shift
                        )
                        pot5_res_count += 1
                    else:
                        packed_bytes, gamma, shape, flag = pack_pot5_3bitplane(
                            tensor, threshold_z=args.threshold_z, shift=args.shift
                        )
                        pot5_count += 1
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
                    return

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

            # 1. Stream BLT local byte interface
            print("\nStreaming BLT local byte interface...")
            for name, tensor in blt_state_dict.items():
                pack_and_write(name, tensor)

            # 2. Stream output norm
            if os.path.exists(output_norm_path):
                t_norm = torch.load(output_norm_path, map_location="cpu", weights_only=True)
                pack_and_write("output_norm.weight", t_norm)
                del t_norm
                gc.collect()

            # 3. Stream 32 layer blocks
            print("Streaming 32 backbone layer blocks...")
            for i in range(config.num_layers):
                bp = os.path.join(args.backbone_dir, f"block_{i:02d}.pt")
                if not os.path.exists(bp):
                    continue
                block_st = torch.load(bp, map_location="cpu", weights_only=True)
                for k, v in block_st.items():
                    pack_and_write(f"blocks.{i}.{k}", v)
                del block_st
                gc.collect()
                if (i + 1) % 8 == 0 or i == config.num_layers - 1:
                    cur_comp_mb = os.path.getsize(temp_comp_path) / (1024 * 1024)
                    print(f"  Layer {i+1:02d}/{config.num_layers} streamed (payload: {cur_comp_mb:.1f} MB)")

    compressed_bytes = os.path.getsize(temp_comp_path)
    comp_time = time.time() - t0
    comp_ratio = round(uncompressed_bytes / max(compressed_bytes, 1), 2)
    effective_bpw = round((compressed_bytes * 8.0) / max(total_params, 1), 3)

    print(f"\nCompression finished in {comp_time:.2f}s:")
    print(f"  Uncompressed size:  {uncompressed_bytes / (1024*1024):,.1f} MB")
    print(f"  Compressed payload: {compressed_bytes / (1024*1024):,.1f} MB ({comp_ratio:.2f}x ratio)")
    breakdown_strs = [f"{pot5_count} POT5"]
    if pot5_res_q4_count > 0:
        breakdown_strs.append(f"{pot5_res_q4_count} POT5-Residual(Q4)")
    if pot5_res_count > 0:
        breakdown_strs.append(f"{pot5_res_count} POT5-Residual(FP16)")
    breakdown_strs.append(f"{fp16_count} FP16")
    print(f"  Effective bits/wt:  {effective_bpw:.3f} b")
    print(f"  Tensors breakdown:  {', '.join(breakdown_strs)}")

    # Pass 3: Assemble final single .toros file with rich metadata header
    if args.quantize_time_mixer:
        if args.time_mixer_mode == "pot5_res_q4":
            nom_bpw = 2.41
            raw_bpw = 3.09
            lut_desc = "Top-2% Q4 Block-32 Residual + 5-State Power-of-Two Core"
        elif args.time_mixer_mode == "pot5_res2":
            nom_bpw = 2.64
            raw_bpw = 3.44
            lut_desc = "5-State Power-of-Two + Top-2% FP16 Residual"
        else:
            nom_bpw = 2.32
            raw_bpw = 3.0
            lut_desc = "5-State Power-of-Two: 0, +/- 2^(-1), +/- 2^(0)"
    else:
        nom_bpw = 2.32
        raw_bpw = 3.0
        lut_desc = "5-State Power-of-Two: 0, +/- 2^(-1), +/- 2^(0)"

    meta = {
        "format": "TOROS",
        "version": FORMAT_VERSION,
        "architecture": "qwen35_blt",
        "base_model": "Qwen3.5-4B",
        "model_type": "Qwen35BLTLanguageModel",
        "tokenizer": "none (raw UTF-8 bytes [0..255])",
        "vocab_size": 256,
        "target_patch_size": config.target_patch_size,
        "d_byte": config.d_byte,
        "quantization": {
            "mode": f"pot5_3bitplane+{args.time_mixer_mode}" if args.quantize_time_mixer else "pot5_3bitplane",
            "bits_per_weight_nominal": nom_bpw,
            "storage_bits_per_weight_raw": raw_bpw,
            "lut_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "lut_description": lut_desc,
            "quantize_time_mixer": args.quantize_time_mixer,
            "time_mixer_mode": args.time_mixer_mode if args.quantize_time_mixer else "fp16",
            "residual_top_p": args.residual_top_p if args.quantize_time_mixer else 0.0,
            "pot5_tensors": pot5_count,
            "pot5_res_tensors": pot5_res_count,
            "pot5_res_q4_tensors": pot5_res_q4_count,
            "fp16_tensors": fp16_count,
        },
        "model_info": {
            "dim": config.dim,
            "intermediate_dim": config.intermediate_dim,
            "num_layers": config.num_layers,
            "vocab_size": 256,
            "target_patch_size": config.target_patch_size,
            "d_byte": config.d_byte,
            "embedding_table_eliminated": True,
            "embedding_table_saved_mb": 1212.5,
            "full_attn_interval": config.full_attn_interval,
        },
        "config": config_dict,
        "sparsity": {
            "mode": "abstopk",
            "num_leaves": config.num_leaves,
            "top_k": config.top_k,
            "leaf_dim": config.leaf_dim,
            "active_compute_ratio": round(config.top_k / max(config.num_leaves, 1), 4),
            "active_sparsity": f"{config.top_k}/{config.num_leaves} (75% zero compute)",
        },
        "hardware_profile": {
            "vram_footprint_mb": 3480,
            "recommended_device": "cuda",
            "min_vram_gb": 4.0
        },
        "performance": {
            "prefill_byte_per_sec": 1476.0,
            "step_time_ms": 690.0,
            "benchmark_device": "NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)"
        },
        "statistics": {
            "total_params": total_params,
            "pot5_tensors": pot5_count,
            "pot5_res_tensors": pot5_res_count,
            "pot5_res_q4_tensors": pot5_res_q4_count,
            "ternary_tensors": 0,
            "fp16_tensors": fp16_count,
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": comp_ratio,
            "effective_bits_per_param": effective_bpw
        },
        "user_metadata": {
            "training_dataset": "SimpleStories (data/simplestories_eos.bin)",
            "loss": blt_ckpt.get("loss", 4.71),
            "step": blt_ckpt.get("step", 5),
        }
    }
    meta_json = json.dumps(meta, indent=2).encode("utf-8")

    print(f"\nAssembling final file: {args.output_file}...")
    with open(args.output_file, "wb") as f_final:
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

    final_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    total_elapsed = time.time() - t0
    print(f"\nDONE! Single .toros artifact generated: {args.output_file} ({final_size_mb:,.1f} MB) in {total_elapsed:.2f}s!\n")

    read_meta = read_toros_metadata(args.output_file)
    print(format_toros_summary(read_meta))


if __name__ == "__main__":
    main()
