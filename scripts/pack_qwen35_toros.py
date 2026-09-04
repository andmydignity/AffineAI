#!/usr/bin/env python3
"""
Streaming Packer for Qwen3.5-4B ASDAG into a single ultra-compact .toros file.
Streams layer-by-layer to disk with constant memory footprint (< 600 MB RAM).

Usage:
  python scripts/pack_qwen35_toros.py --input-dir checkpoints/qwen35_asdag --output-file checkpoints/qwen35_asdag.toros
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
    pack_ternary_tensor,
)
from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig


def main():
    parser = argparse.ArgumentParser(description="Streaming pack Qwen3.5 ASDAG blocks into a single .toros file")
    parser.add_argument("--input-dir", type=str, default="checkpoints/qwen35_asdag")
    parser.add_argument("--output-file", type=str, default="checkpoints/qwen35_asdag.toros")
    parser.add_argument("--compression-level", type=int, default=3, help="Zstandard compression level (default: 3)")
    args = parser.parse_args()

    print(f"=== Streaming Qwen3.5 ASDAG to Single .toros Checkpoint ===")
    print(f"Input dir:         {args.input_dir}")
    print(f"Target .toros:     {args.output_file}")
    print(f"Compression level: {args.compression_level}")

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
                lt0 = time.time()
                block_st = torch.load(bp, weights_only=True)
                for k, v in block_st.items():
                    pack_and_write(f"blocks.{i}.{k}", v)
                del block_st
                gc.collect()
                if (i + 1) % 4 == 0 or i == config.num_layers - 1:
                    cur_comp_mb = os.path.getsize(temp_comp_path) / (1024 * 1024)
                    print(f"  Layer {i+1:02d}/{config.num_layers} streamed (current compressed size: {cur_comp_mb:.1f} MB)")

            # 4. Stream MTP block
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
    print(f"\nCompression finished in {comp_time:.2f}s:")
    print(f"  Uncompressed size: {uncompressed_bytes / (1024*1024):,.1f} MB")
    print(f"  Compressed payload: {compressed_bytes / (1024*1024):,.1f} MB ({uncompressed_bytes / max(compressed_bytes, 1):.2f}x ratio)")
    print(f"  Tensors: {ternary_count} ternary (2-bit packed), {fp16_count} FP16")

    # Pass 3: Assemble final single .toros file with metadata header
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
    print(f"DONE! Single .toros artifact generated: {args.output_file} ({final_size_mb:,.1f} MB) in {total_elapsed:.2f}s!")


if __name__ == "__main__":
    main()
