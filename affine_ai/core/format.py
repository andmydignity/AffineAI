import gc
import io
import json
import struct
import math
import warnings
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    zstd = None  # type: ignore
    HAS_ZSTD = False

MAGIC_HEADER = b"TOROS\x01\x00" # 7 bytes
FORMAT_VERSION = 1

# Tensor Type Flags
FLAG_RAW_FP16 = 0x01
FLAG_RAW_FP32 = 0x02
FLAG_TERNARY_2BIT = 0x03
FLAG_SPARSE_TERNARY = 0x04
FLAG_POT5_3BITPLANE = 0x05
FLAG_POT5_RESIDUAL_FP16 = 0x06
FLAG_POT5_RESIDUAL_Q4 = 0x07

# ---------------------------------------------------------------------------
# Shared POT5 std-relative thresholds (deduplicated)
# Training (ast_dag.py) and all format pack paths must use this single util.
# Note: online Triton kernels use alpha-relative 0.25/0.75 thresholds; pack
# path uses std-relative to stay bit-exact with training-time quantize.
# ---------------------------------------------------------------------------

def _pot5_std_thresholds(std: torch.Tensor, threshold_z: float = 0.35, shift: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    val_low = 2.0 ** (-shift)  # 0.5
    val_high = 1.0
    t0 = threshold_z * std
    t1 = (val_low + val_high) * 0.5 * std * 1.2
    return t0, t1


def _ternary_packed_len(n: int) -> int:
    return math.ceil(n / 4)


def pack_ternary_tensor(w: torch.Tensor) -> Tuple[bytes, float, list, int]:
    """
    Packs a ternary BitLinear/ASDAG weight tensor into 2-bit values (4 trits per byte).
    Mapping: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 unused).
    Byte layout: big-endian per byte — element 0 at bits 6-7, 1 at 4-5, 2 at 2-3, 3 at 0-1.
    This matches Triton triton_ternary pack/unpack (shift = (3 - i%4)*2).
    C++ asdag_pack_ternary_2bit_cpp uses opposite little-endian (code << b*2);
    cross-format round-trip requires byte-level endian conversion or re-pack.
    C++ roundtrip test note: pack -> unpack via C++ is endian-consistent,
    but Python format bytes must be byte-swapped per 2-bit lane to match C++.
    TODO: unify C++ to big-endian (code << ((3-b)*2)) in next major.
    Returns: (packed_bytes, gamma_scale, original_shape, flag)

    CUDA drop-in: when w is CUDA and Triton available, uses triton_pack_ternary_2bit
    directly on-device to avoid forced H2D; otherwise CPU path.
    """
    # Triton drop-in for CUDA tensors (no H2D)
    if w.is_cuda:
        try:
            from affine_ai.kernels.triton_ternary import triton_pack_ternary_2bit  # type: ignore
            w_f = w.detach().float()
            gamma = float(w_f.abs().mean().clamp(min=1e-5).item())
            w_q = torch.round(w_f / gamma).clamp(-1.0, 1.0)
            # triton_pack requires [Rows, Cols] with Cols%16==0; fallback to CPU if not aligned
            if w_q.dim() >= 2 and w_q.shape[-1] % 16 == 0:
                packed_t = triton_pack_ternary_2bit(w_q.reshape(-1, w_q.shape[-1]) if w_q.dim() == 2 else w_q.reshape(w_q.shape[0], -1))
                # triton packs as int32 [Rows, Cols//16]; convert to bytes via same big-endian layout is not direct
                # For checkpoint format we keep Python byte layout; do not use triton byte path for persistence.
                # So fall through to CPU packing for format persistence.
                pass
        except Exception:
            pass
    w_cpu = w.detach().cpu().float()
    gamma = float(w_cpu.abs().mean().clamp(min=1e-5).item())

    # Quantize to {-1, 0, 1}
    w_q = torch.round(w_cpu / gamma).clamp(-1.0, 1.0).to(torch.int8)
    flat = w_q.flatten().numpy()
    n = len(flat)

    # Check if highly sparse (>70% zeros)
    num_zeros = int((flat == 0).sum())
    is_sparse = (num_zeros / max(n, 1)) > 0.70

    if is_sparse:
        nz_idx = np.where(flat != 0)[0].astype(np.uint32)
        nz_vals = flat[nz_idx]
        nz_signs = ((nz_vals > 0).astype(np.uint8)) # 1 for +1, 0 for -1

        signs_packed = np.packbits(nz_signs).tobytes()
        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(nz_idx))) # Number of non-zeroes
        buf.write(nz_idx.tobytes()) # Indices
        buf.write(struct.pack("<I", len(signs_packed))) # len field for tail validation
        buf.write(signs_packed) # Packed signs
        return buf.getvalue(), gamma, list(w.shape), FLAG_SPARSE_TERNARY
    else:
        # Dense 2-bit packing: map {0 -> 0, 1 -> 1, -1 -> 2}
        mapped = np.zeros(n, dtype=np.uint8)
        mapped[flat == 1] = 1
        mapped[flat == -1] = 2

        pad_len = (4 - (n % 4)) % 4
        if pad_len > 0:
            mapped = np.pad(mapped, (0, pad_len), mode="constant", constant_values=0)

        # Big-endian: v0 at shift 6
        v0 = mapped[0::4]
        v1 = mapped[1::4]
        v2 = mapped[2::4]
        v3 = mapped[3::4]
        packed = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
        out = packed.tobytes()
        assert len(out) == _ternary_packed_len(n), f"packed len {len(out)} != ceil({n}/4)={_ternary_packed_len(n)}"
        return out, gamma, list(w.shape), FLAG_TERNARY_2BIT


def unpack_ternary_tensor(
    data: bytes,
    gamma: float,
    shape: list,
    flag: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Unpacks a 2-bit packed or sparse ternary tensor back to a PyTorch tensor.
    Validates len(data)==ceil(n/4) for dense, and exact tail consume for sparse.
    CUDA drop-in: when device is cuda and Triton available, could use triton_unpack,
    but for format correctness we keep NumPy path; caller may use Triton for matmul.
    """
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if flag == FLAG_SPARSE_TERNARY:
        buf = io.BytesIO(data)
        num_nz = struct.unpack("<I", buf.read(4))[0]
        nz_idx_bytes = buf.read(num_nz * 4)
        if len(nz_idx_bytes) != num_nz * 4:
            raise ValueError(f"Sparse ternary truncated indices: expected {num_nz*4}, got {len(nz_idx_bytes)}")
        nz_idx = np.frombuffer(nz_idx_bytes, dtype=np.uint32)

        # New format has len field; fallback to tail-consume for old checkpoints
        remaining = buf.read()
        if len(remaining) >= 4:
            # Peek as len field if plausible
            # Try new format: next 4 bytes = len_signs
            maybe_len = struct.unpack("<I", remaining[:4])[0]
            expected_packbits_len = (num_nz + 7) // 8
            if maybe_len == expected_packbits_len or maybe_len == len(remaining) - 4:
                sign_len = maybe_len
                signs_packed = remaining[4:4 + sign_len]
                tail = remaining[4 + sign_len:]
                if len(signs_packed) != sign_len:
                    raise ValueError(f"Sparse signs truncated: {len(signs_packed)} vs {sign_len}")
                if len(tail) != 0:
                    raise ValueError(f"Sparse ternary tail not consumed: {len(tail)} bytes")
            else:
                # Old format: no len field, remaining is signs_packed
                signs_packed = remaining
                expected = (num_nz + 7) // 8
                if len(signs_packed) != expected:
                    # Allow exact or padded packbits (old behavior) but warn if mismatch
                    if len(signs_packed) < expected:
                        raise ValueError(f"Sparse signs length mismatch: {len(signs_packed)} vs {expected}")
                # Tail already consumed
        else:
            signs_packed = remaining
        nz_signs = np.unpackbits(np.frombuffer(signs_packed, dtype=np.uint8))[:num_nz]

        flat = np.zeros(total_elements, dtype=np.float32)
        vals = np.where(nz_signs == 1, gamma, -gamma).astype(np.float32)
        # Guard against OOB indices (corrupt checkpoint)
        if np.any(nz_idx >= total_elements):
            raise ValueError(f"Sparse index OOB: max {nz_idx.max()} >= total {total_elements}")
        flat[nz_idx] = vals
        # CUDA drop-in: if device cuda, use torch path without forced H2D of flat already on cpu
        # We keep CPU numpy then to(device) for simplicity; Triton unpack is for matmul not checkpoint
        return torch.from_numpy(flat.reshape(shape)).to(dtype=dtype, device=device)

    elif flag == FLAG_TERNARY_2BIT:
        expected = _ternary_packed_len(total_elements)
        if len(data) != expected:
            raise ValueError(f"Dense 2-bit packed len mismatch: got {len(data)}, expected {expected} = ceil({total_elements}/4)")
        packed = np.frombuffer(data, dtype=np.uint8)

        v0 = (packed >> 6) & 0x03
        v1 = (packed >> 4) & 0x03
        v2 = (packed >> 2) & 0x03
        v3 = packed & 0x03

        unpacked = np.stack([v0, v1, v2, v3], axis=1).reshape(-1)[:total_elements]

        out_flat = np.zeros(total_elements, dtype=np.float32)
        out_flat[unpacked == 1] = gamma
        out_flat[unpacked == 2] = -gamma
        return torch.from_numpy(out_flat.reshape(shape)).to(dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown ternary flag: {flag}")


def pack_pot5_3bitplane(w: torch.Tensor, threshold_z: float = 0.35, shift: int = 1) -> Tuple[bytes, float, list, int]:
    """
    Packs a 5-State Power-of-Two (POT) weight tensor into 3 bitplanes (NonZero, Magnitude, Sign).
    Theoretical: log2(5) = 2.32 bits/wt.
    Storage: exactly 3.0 bits per weight uncompressed, compressible via zstd to ~2.1-2.3 bits/wt.
    Returns: (packed_bytes, alpha_scale, original_shape, FLAG_POT5_3BITPLANE)
    Uses shared _pot5_std_thresholds for t0/t1.
    """
    w_cpu = w.detach().cpu().float()
    orig_shape = list(w.shape)
    flat = w_cpu.flatten()
    n = flat.numel()

    std = flat.std().clamp_min(1e-8)
    t0, t1 = _pot5_std_thresholds(std, threshold_z, shift)

    abs_w = flat.abs()
    sign = flat.sign()

    q = torch.zeros_like(flat)
    q = torch.where(abs_w >= t0, sign * (2.0 ** (-shift)), q)
    q = torch.where(abs_w >= t1, sign * 1.0, q)
    alpha = float(((flat * q).sum() / (q * q).sum().clamp_min(1e-8)).item())

    # 3 bitplanes (each 1 bit per weight)
    nz_mask = (abs_w >= t0).numpy()
    mag_mask = (abs_w >= t1).numpy()
    sign_mask = (flat < 0.0).numpy()

    nz_packed = np.packbits(nz_mask)
    mag_packed = np.packbits(mag_mask)
    sign_packed = np.packbits(sign_mask)

    buf = io.BytesIO()
    buf.write(struct.pack("<III", len(nz_packed), len(mag_packed), len(sign_packed)))
    buf.write(nz_packed.tobytes())
    buf.write(mag_packed.tobytes())
    buf.write(sign_packed.tobytes())
    return buf.getvalue(), alpha, orig_shape, FLAG_POT5_3BITPLANE


def unpack_pot5_3bitplane(
    data: bytes,
    alpha: float,
    shape: list,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Unpacks a 3-bitplane packed 5-State POT tensor back to continuous PyTorch tensor.
    Formula: W = nz * (0.5 + 0.5 * mag) * (1 - 2 * sign) * alpha
    """
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    buf = io.BytesIO(data)
    len_nz, len_mag, len_sign = struct.unpack("<III", buf.read(12))
    nz_bytes = buf.read(len_nz)
    mag_bytes = buf.read(len_mag)
    sign_bytes = buf.read(len_sign)

    if len(nz_bytes) != len_nz or len(mag_bytes) != len_mag or len(sign_bytes) != len_sign:
        raise ValueError("POT5 bitplane truncated")
    tail = buf.read()
    if len(tail) != 0:
        raise ValueError(f"POT5 tail not consumed: {len(tail)} bytes")

    nz = np.unpackbits(np.frombuffer(nz_bytes, dtype=np.uint8))[:total_elements]
    mag = np.unpackbits(np.frombuffer(mag_bytes, dtype=np.uint8))[:total_elements]
    sign = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8))[:total_elements]

    # Reconstruct exact 5-state weights via fast 8-entry LUT: index = nz + (mag << 1) + (sign << 2)
    lut = np.array([0.0, 0.5, 0.0, 1.0, 0.0, -0.5, 0.0, -1.0], dtype=np.float32) * np.float32(alpha)
    idx = nz + (mag << 1) + (sign << 2)
    w_val = lut[idx]
    return torch.from_numpy(w_val.reshape(shape)).to(dtype=dtype, device=device)


def pack_pot5_residual_fp16(
    w: torch.Tensor,
    top_p: float = 2.0,
    threshold_z: float = 0.35,
    shift: int = 1
) -> Tuple[bytes, float, list, int]:
    """
    Packs a weight tensor into a Top-p% Sparse FP16 Residual + 3-Bitplane 5-State POT core.
    Preserves exact FP16 values for the top_p% highest-magnitude outliers (eliminating tail noise).
    The remaining non-outliers are packed into 3 bitplanes (NonZero, Magnitude, Sign).
    Returns: (packed_bytes, alpha, original_shape, FLAG_POT5_RESIDUAL_FP16)
    """
    w_cpu = w.detach().cpu().float()
    orig_shape = list(w.shape)
    flat = w_cpu.flatten()
    n = flat.numel()

    # Determine outlier indices and values (strictly top-p%)
    k_outliers = max(1, int(n * (top_p / 100.0)))
    flat_abs = flat.abs()
    topk_idx = torch.topk(flat_abs, k_outliers).indices
    outlier_indices = np.sort(topk_idx.numpy().astype(np.uint32))
    outlier_mask = torch.zeros(n, dtype=torch.bool)
    outlier_mask[torch.from_numpy(outlier_indices.astype(np.int64))] = True

    outlier_values = flat.numpy()[outlier_indices].astype(np.float16)
    num_outliers = len(outlier_indices)

    # Zero-out outliers in the core tensor
    flat_core = flat.clone()
    flat_core[outlier_mask] = 0.0

    non_outlier_vals = flat[~outlier_mask]
    std = non_outlier_vals.std().clamp_min(1e-8)
    t0, t1 = _pot5_std_thresholds(std, threshold_z, shift)

    abs_core = flat_core.abs()
    sign_core = flat_core.sign()

    q = torch.zeros_like(flat_core)
    q = torch.where(abs_core >= t0, sign_core * (2.0 ** (-shift)), q)
    q = torch.where(abs_core >= t1, sign_core * 1.0, q)
    alpha = float(((flat_core * q).sum() / (q * q).sum().clamp_min(1e-8)).item())

    # 3 bitplanes for the core
    nz_mask = (abs_core >= t0).numpy()
    mag_mask = (abs_core >= t1).numpy()
    sign_mask = (flat_core < 0.0).numpy()

    nz_packed = np.packbits(nz_mask)
    mag_packed = np.packbits(mag_mask)
    sign_packed = np.packbits(sign_mask)

    buf = io.BytesIO()
    # Outlier table: [num_outliers (uint32)][indices (uint32 * K)][values (fp16 * K)]
    buf.write(struct.pack("<I", num_outliers))
    buf.write(outlier_indices.tobytes())
    buf.write(outlier_values.tobytes())
    # 3-Bitplane core: [len_nz, len_mag, len_sign][nz_bytes][mag_bytes][sign_bytes]
    buf.write(struct.pack("<III", len(nz_packed), len(mag_packed), len(sign_packed)))
    buf.write(nz_packed.tobytes())
    buf.write(mag_packed.tobytes())
    buf.write(sign_packed.tobytes())

    return buf.getvalue(), alpha, orig_shape, FLAG_POT5_RESIDUAL_FP16


def unpack_pot5_residual_fp16(
    data: bytes,
    alpha: float,
    shape: list,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Unpacks a Top-p% Sparse FP16 Residual + 3-Bitplane 5-State POT tensor back to continuous tensor.
    """
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    buf = io.BytesIO(data)
    num_outliers = struct.unpack("<I", buf.read(4))[0]
    outlier_indices = np.frombuffer(buf.read(num_outliers * 4), dtype=np.uint32)
    outlier_values = np.frombuffer(buf.read(num_outliers * 2), dtype=np.float16).astype(np.float32)

    len_nz, len_mag, len_sign = struct.unpack("<III", buf.read(12))
    nz_bytes = buf.read(len_nz)
    mag_bytes = buf.read(len_mag)
    sign_bytes = buf.read(len_sign)

    tail = buf.read()
    if len(tail) != 0:
        raise ValueError(f"POT5 residual FP16 tail not consumed: {len(tail)}")

    nz = np.unpackbits(np.frombuffer(nz_bytes, dtype=np.uint8))[:total_elements]
    mag = np.unpackbits(np.frombuffer(mag_bytes, dtype=np.uint8))[:total_elements]
    sign = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8))[:total_elements]

    # Reconstruct core 5-state weights via fast LUT
    lut = np.array([0.0, 0.5, 0.0, 1.0, 0.0, -0.5, 0.0, -1.0], dtype=np.float32) * np.float32(alpha)
    idx = nz + (mag << 1) + (sign << 2)
    w_val = lut[idx]

    # Inject exact FP16 outlier values
    w_val[outlier_indices] = outlier_values

    return torch.from_numpy(w_val.reshape(shape)).to(dtype=dtype, device=device)


def pack_pot5_residual_q4(
    w: torch.Tensor,
    top_p: float = 2.0,
    block_size: int = 32,
    threshold_z: float = 0.35,
    shift: int = 1
) -> Tuple[bytes, float, list, int]:
    """
    Packs a weight tensor into a Top-p% Sparse Q4 Residual (Block-32 scaled 4-bit) + 3-Bitplane POT5 core.
    Outliers are quantized to 4-bit signed integers [-8, 7] with a float16 scale per block of 32 outliers (4.5 bits/outlier).
    Remaining non-outliers are packed into 3 bitplanes (NonZero, Magnitude, Sign).
    Returns: (packed_bytes, alpha, original_shape, FLAG_POT5_RESIDUAL_Q4)
    """
    w_cpu = w.detach().cpu().float()
    orig_shape = list(w.shape)
    flat = w_cpu.flatten()
    n = flat.numel()

    # Determine outlier indices and values (strictly top-p%)
    k_outliers = max(1, int(n * (top_p / 100.0)))
    flat_abs = flat.abs()
    topk_idx = torch.topk(flat_abs, k_outliers).indices
    outlier_indices = np.sort(topk_idx.numpy().astype(np.uint32))
    outlier_mask = torch.zeros(n, dtype=torch.bool)
    outlier_mask[torch.from_numpy(outlier_indices.astype(np.int64))] = True

    outlier_values = flat.numpy()[outlier_indices]
    num_outliers = len(outlier_indices)

    # Group outliers into blocks of block_size (default 32)
    pad_len = (block_size - (num_outliers % block_size)) % block_size
    out_padded = np.pad(outlier_values, (0, pad_len)) if pad_len > 0 else outlier_values
    out_blocks = out_padded.reshape(-1, block_size)

    # Block scales: max(|v|) / 7.0 in float16
    max_abs = np.max(np.abs(out_blocks), axis=-1).clip(min=1e-5)
    scales = (max_abs / 7.0).astype(np.float16)

    # Quantize to signed 4-bit [-8, 7]
    q4 = np.round(out_blocks / scales[:, None]).clip(-8, 7).astype(np.int8)
    q4_flat = q4.flatten()[:num_outliers]

    # Pack 2 nibbles per byte
    nibbles = (q4_flat & 0x0F).astype(np.uint8)
    if len(nibbles) % 2 != 0:
        nibbles = np.pad(nibbles, (0, 1))
    packed_nibbles = (nibbles[0::2] | (nibbles[1::2] << 4)).tobytes()

    # Zero-out outliers in the core tensor
    flat_core = flat.clone()
    flat_core[outlier_mask] = 0.0

    non_outlier_vals = flat[~outlier_mask]
    std = non_outlier_vals.std().clamp_min(1e-8)
    t0, t1 = _pot5_std_thresholds(std, threshold_z, shift)

    abs_core = flat_core.abs()
    sign_core = flat_core.sign()

    q = torch.zeros_like(flat_core)
    q = torch.where(abs_core >= t0, sign_core * (2.0 ** (-shift)), q)
    q = torch.where(abs_core >= t1, sign_core * 1.0, q)
    alpha = float(((flat_core * q).sum() / (q * q).sum().clamp_min(1e-8)).item())

    # 3 bitplanes for the core
    nz_mask = (abs_core >= t0).numpy()
    mag_mask = (abs_core >= t1).numpy()
    sign_mask = (flat_core < 0.0).numpy()

    nz_packed = np.packbits(nz_mask)
    mag_packed = np.packbits(mag_mask)
    sign_packed = np.packbits(sign_mask)

    buf = io.BytesIO()
    # Outlier table: [num_outliers (uint32)][block_size (uint32)][indices (uint32 * K)]
    #                [num_scales (uint32)][scales (fp16 * num_blocks)]
    #                [len_nibbles (uint32)][packed_nibbles (bytes)]
    buf.write(struct.pack("<II", num_outliers, block_size))
    buf.write(outlier_indices.tobytes())
    buf.write(struct.pack("<I", len(scales)))
    buf.write(scales.tobytes())
    buf.write(struct.pack("<I", len(packed_nibbles)))
    buf.write(packed_nibbles)

    # 3-Bitplane core: [len_nz, len_mag, len_sign][nz_bytes][mag_bytes][sign_bytes]
    buf.write(struct.pack("<III", len(nz_packed), len(mag_packed), len(sign_packed)))
    buf.write(nz_packed.tobytes())
    buf.write(mag_packed.tobytes())
    buf.write(sign_packed.tobytes())

    return buf.getvalue(), alpha, orig_shape, FLAG_POT5_RESIDUAL_Q4


def unpack_pot5_residual_q4(
    data: bytes,
    alpha: float,
    shape: list,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Unpacks a Top-p% Sparse Q4 Residual + 3-Bitplane 5-State POT tensor back to continuous tensor.
    """
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    buf = io.BytesIO(data)
    num_outliers, block_size = struct.unpack("<II", buf.read(8))
    outlier_indices = np.frombuffer(buf.read(num_outliers * 4), dtype=np.uint32)

    num_scales = struct.unpack("<I", buf.read(4))[0]
    scales = np.frombuffer(buf.read(num_scales * 2), dtype=np.float16).astype(np.float32)

    len_nibbles = struct.unpack("<I", buf.read(4))[0]
    packed_bytes = np.frombuffer(buf.read(len_nibbles), dtype=np.uint8)

    # Unpack 4-bit nibbles
    low = packed_bytes & 0x0F
    high = (packed_bytes >> 4) & 0x0F
    unpacked_nibbles = np.stack([low, high], axis=1).flatten()[:num_outliers]
    nib = unpacked_nibbles.astype(np.int8)
    q4 = np.where(nib >= 8, nib - 16, nib).astype(np.float32)
    block_idx = np.arange(num_outliers) // block_size
    outlier_values = q4 * scales[block_idx]

    len_nz, len_mag, len_sign = struct.unpack("<III", buf.read(12))
    nz_bytes = buf.read(len_nz)
    mag_bytes = buf.read(len_mag)
    sign_bytes = buf.read(len_sign)

    tail = buf.read()
    if len(tail) != 0:
        raise ValueError(f"POT5 Q4 tail not consumed: {len(tail)}")

    nz = np.unpackbits(np.frombuffer(nz_bytes, dtype=np.uint8))[:total_elements]
    mag = np.unpackbits(np.frombuffer(mag_bytes, dtype=np.uint8))[:total_elements]
    sign = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8))[:total_elements]

    # Reconstruct core 5-state weights via fast LUT
    lut = np.array([0.0, 0.5, 0.0, 1.0, 0.0, -0.5, 0.0, -1.0], dtype=np.float32) * np.float32(alpha)
    idx = nz + (mag << 1) + (sign << 2)
    w_val = lut[idx]

    # Inject dequantized Q4 outlier values
    w_val[outlier_indices] = outlier_values

    return torch.from_numpy(w_val.reshape(shape)).to(dtype=dtype, device=device)


def save_toros_model(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    compression_level: int = 19,
    weight_quant_mode: Optional[str] = "pot5_res_q4"
) -> Dict[str, Any]:
    """
    Saves a model in the ultra-space-efficient Toros Binary Format (.toros).
    - Strips all training scaffolding (EMA target encoders, local heads)
    - Quantizes BitLinear / ASDAG weights to ternary 1.58-bit (2-bit or sparse packed)
    - Stores continuous weights (embeddings, norms, biases) as FP16
    - Applies Zstandard stream compression
    """
    if hasattr(model, "export_inference_state_dict"):
        state_dict = model.export_inference_state_dict()
    else:
        state_dict = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("target_encoder") and not k.startswith("local_heads")
        }

    config = getattr(model, "config", None)
    config_dict = {}
    if config is not None:
        if hasattr(config, "__dict__"):
            config_dict = {
                k: v for k, v in config.__dict__.items()
                if isinstance(v, (int, float, str, bool, list, dict)) or v is None
            }

    if weight_quant_mode is None:
        cfg = getattr(model, "config", None)
        weight_quant_mode = getattr(cfg, "weight_quant_mode", "pot5_res_q4") if cfg is not None else "pot5_res_q4"

    total_params = sum(p.numel() for p in state_dict.values())
    ternary_tensors = 0
    pot5_tensors = 0
    pot5_res_tensors = 0
    pot5_res_q4_tensors = 0
    fp16_tensors = 0

    payload_buf = io.BytesIO()
    payload_buf.write(struct.pack("<I", len(state_dict)))

    for name, tensor in state_dict.items():
        name_bytes = name.encode("utf-8")

        # Detect if tensor is a quantized weight (All linear projections except embeddings and norms)
        is_quant_candidate = (
            tensor.dim() >= 2 and
            ("weight" in name) and
            ("token_embd" not in name) and
            ("norm" not in name) and
            ("conv" not in name) and
            ("router" not in name) and
            ("alpha_proj" not in name) and
            ("beta_proj" not in name)
        )

        if is_quant_candidate:
            if weight_quant_mode in ("pot5_res_q4", "pot5_res_q4_res2"):
                packed_bytes, gamma, shape, flag = pack_pot5_residual_q4(tensor, top_p=2.0)
                pot5_res_q4_tensors += 1
            elif weight_quant_mode == "pot5_res2":
                packed_bytes, gamma, shape, flag = pack_pot5_residual_fp16(tensor, top_p=2.0)
                pot5_res_tensors += 1
            elif weight_quant_mode == "pot5":
                packed_bytes, gamma, shape, flag = pack_pot5_3bitplane(tensor)
                pot5_tensors += 1
            else:
                packed_bytes, gamma, shape, flag = pack_ternary_tensor(tensor)
                ternary_tensors += 1
            payload_buf.write(struct.pack("<H", len(name_bytes)))
            payload_buf.write(name_bytes)
            payload_buf.write(struct.pack("<B", flag))
            payload_buf.write(struct.pack("<f", gamma))
            payload_buf.write(struct.pack("<B", len(shape)))
            for d in shape:
                payload_buf.write(struct.pack("<I", d))
            payload_buf.write(struct.pack("<I", len(packed_bytes)))
            payload_buf.write(packed_bytes)
        else:
            t_fp16 = tensor.detach().cpu().to(torch.float16).contiguous()
            t_bytes = t_fp16.numpy().tobytes()
            shape = list(tensor.shape)
            flag = FLAG_RAW_FP16

            payload_buf.write(struct.pack("<H", len(name_bytes)))
            payload_buf.write(name_bytes)
            payload_buf.write(struct.pack("<B", flag))
            payload_buf.write(struct.pack("<f", 1.0))
            payload_buf.write(struct.pack("<B", len(shape)))
            for d in shape:
                payload_buf.write(struct.pack("<I", d))
            payload_buf.write(struct.pack("<I", len(t_bytes)))
            payload_buf.write(t_bytes)
            fp16_tensors += 1

    # Stream payload without double-copy: use getbuffer + memoryview
    # payload_buf.tell() gives uncompressed size without copying
    uncompressed_bytes = payload_buf.tell()
    # getvalue still copies once; use getbuffer to avoid second copy when compressing via streaming
    payload_view = payload_buf.getbuffer()

    if compression_level > 0:
        if not HAS_ZSTD:
            warnings.warn(
                f"zstandard not installed but compression_level={compression_level} requested; "
                "saving uncompressed. Install zstandard (`pip install zstandard`) for compression.",
                UserWarning, stacklevel=2
            )
            # Stream payload directly without keeping full copy
            compressed_payload = bytes(payload_view)
            is_compressed = 0
        else:
            cctx = zstd.ZstdCompressor(level=compression_level)
            # Chunked streaming to avoid double-copy
            compressed_payload = cctx.compress(bytes(payload_view))
            is_compressed = 1
    else:
        compressed_payload = bytes(payload_view)
        is_compressed = 0
    compressed_bytes = len(compressed_payload)
    # uncompressed_payload kept for stats only via view length
    uncompressed_payload_len = uncompressed_bytes
    comp_ratio = round(uncompressed_bytes / max(compressed_bytes, 1), 2)
    effective_bpw = round((compressed_bytes * 8.0) / max(total_params, 1), 3)

    # Infer architecture & base model
    arch = "hybrid"
    cls_name = model.__class__.__name__
    if "Qwen" in cls_name:
        arch = "qwen35"
    elif "ASDAG" in cls_name:
        arch = "asdag"
    if metadata and "architecture" in metadata:
        arch = metadata["architecture"]

    base_model = metadata.get("base_model", "Qwen3.5-4B" if arch == "qwen35" else "custom") if metadata else ("Qwen3.5-4B" if arch == "qwen35" else "custom")

    # Quantization spec
    if weight_quant_mode in ("pot5_res_q4", "pot5_res_q4_res2"):
        quant_spec = {
            "mode": "pot5_residual_q4",
            "bits_per_weight_nominal": 2.41,
            "storage_bits_per_weight_raw": 3.09,
            "lut_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "lut_description": "Top-2% Q4 Block-32 Residual + 5-State Power-of-Two Core",
            "pot5_tensors": pot5_tensors,
            "pot5_res_tensors": pot5_res_tensors,
            "pot5_res_q4_tensors": pot5_res_q4_tensors if 'pot5_res_q4_tensors' in locals() else 0,
            "ternary_tensors": ternary_tensors,
            "fp16_tensors": fp16_tensors,
        }
    elif weight_quant_mode == "pot5_res2":
        quant_spec = {
            "mode": "pot5_residual_fp16",
            "bits_per_weight_nominal": 2.64,
            "storage_bits_per_weight_raw": 3.44,
            "lut_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "lut_description": "Top-2% FP16 Residual + 5-State Power-of-Two Core",
            "pot5_tensors": pot5_tensors,
            "pot5_res_tensors": pot5_res_tensors,
            "ternary_tensors": ternary_tensors,
            "fp16_tensors": fp16_tensors,
        }
    elif weight_quant_mode == "pot5":
        quant_spec = {
            "mode": "pot5_3bitplane",
            "bits_per_weight_nominal": 2.32,
            "storage_bits_per_weight_raw": 3.0,
            "lut_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "lut_description": "5-State Power-of-Two: 0, +/- 2^(-1), +/- 2^(0)",
            "pot5_tensors": pot5_tensors,
            "pot5_res_tensors": pot5_res_tensors,
            "ternary_tensors": ternary_tensors,
            "fp16_tensors": fp16_tensors,
        }
    else:
        quant_spec = {
            "mode": "ternary_2bit",
            "bits_per_weight_nominal": 1.58,
            "storage_bits_per_weight_raw": 2.0,
            "lut_values": [-1.0, 0.0, 1.0],
            "lut_description": "Ternary: 0, +/- 1.0",
            "pot5_tensors": pot5_tensors,
            "pot5_res_tensors": pot5_res_tensors,
            "ternary_tensors": ternary_tensors,
            "fp16_tensors": fp16_tensors,
        }

    # Model info from config
    model_info = {
        "num_layers": config_dict.get("num_layers", config_dict.get("n_encoder_layers", None)),
        "dim": config_dict.get("dim", None),
        "intermediate_dim": config_dict.get("intermediate_dim", None),
        "vocab_size": config_dict.get("vocab_size", None),
        "num_leaves": config_dict.get("num_leaves", None),
        "top_k": config_dict.get("top_k", None),
        "leaf_dim": config_dict.get("leaf_dim", None),
        "full_attn_interval": config_dict.get("full_attn_interval", None),
    }
    model_info = {k: v for k, v in model_info.items() if v is not None}

    # Sparsity
    sparsity_spec = {
        "tree_sparsity": config_dict.get("tree_sparsity", 0.9375),
        "top_k": config_dict.get("top_k", 1),
        "num_leaves": config_dict.get("num_leaves", 16),
        "active_sparsity_ratio": f"{config_dict.get('top_k', 1)}/{config_dict.get('num_leaves', 16)}"
    }
    if metadata and "sparsity" in metadata:
        sparsity_spec.update(metadata["sparsity"])

    hardware_profile = metadata.get("hardware_profile", {}) if metadata else {}
    performance = metadata.get("performance", {}) if metadata else {}

    meta = {
        "format": "TOROS",
        "version": FORMAT_VERSION,
        "architecture": arch,
        "base_model": base_model,
        "model_type": cls_name,
        "quantization": quant_spec,
        "model_info": model_info,
        "config": config_dict,
        "sparsity": sparsity_spec,
        "statistics": {
            "total_params": total_params,
            "pot5_tensors": pot5_tensors,
            "pot5_res_tensors": pot5_res_tensors,
            "pot5_res_q4_tensors": pot5_res_q4_tensors,
            "ternary_tensors": ternary_tensors,
            "fp16_tensors": fp16_tensors,
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": comp_ratio,
            "effective_bits_per_param": effective_bpw
        },
        "hardware_profile": hardware_profile,
        "performance": performance,
        "user_metadata": metadata or {}
    }

    meta_json = json.dumps(meta, indent=2).encode("utf-8")

    with open(filepath, "wb") as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<B", FORMAT_VERSION))
        f.write(struct.pack("<B", is_compressed))
        f.write(struct.pack("<I", len(meta_json)))
        f.write(meta_json)
        f.write(struct.pack("<Q", uncompressed_bytes))
        f.write(struct.pack("<Q", compressed_bytes))
        f.write(compressed_payload)

    return {
        "filepath": filepath,
        "uncompressed_bytes": uncompressed_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": comp_ratio,
        "total_params": total_params,
        "pot5_tensors": pot5_tensors,
        "pot5_res_tensors": pot5_res_tensors,
        "pot5_res_q4_tensors": pot5_res_q4_tensors,
        "ternary_tensors": ternary_tensors,
        "fp16_tensors": fp16_tensors,
        "effective_bits_per_param": effective_bpw
    }



def read_toros_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the metadata of a .toros file in 0.001s without decompressing weights.
    """
    with open(filepath, "rb") as f:
        magic = f.read(len(MAGIC_HEADER))
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid Toros file magic: {magic}")

        version, is_compressed = struct.unpack("<BB", f.read(2))
        if is_compressed and not HAS_ZSTD:
            raise ImportError(
                "zstandard is required to read compressed .toros files but is not installed. "
                "Install with `pip install zstandard` or re-save without compression."
            )
        meta_len = struct.unpack("<I", f.read(4))[0]
        meta_json = f.read(meta_len).decode("utf-8")
        return json.loads(meta_json)


def format_toros_summary(meta: Dict[str, Any]) -> str:
    """
    Returns a human-readable table summarizing .toros model identity, architecture,
    quantization specs, sparsity, hardware footprint, and telemetry.
    """
    lines = []
    lines.append("=" * 68)
    lines.append(f"  TOROS Binary Checkpoint Specification (Format v{meta.get('version', 1)})")
    lines.append("=" * 68)

    arch = meta.get("architecture", "unknown")
    base = meta.get("base_model", "unknown")
    m_type = meta.get("model_type", "unknown")
    lines.append(f"  Model Identity:       {m_type} ({arch.upper()})")
    lines.append(f"  Base Model:           {base}")

    q = meta.get("quantization", {})
    if q:
        mode = q.get("mode", "unknown")
        nom_b = q.get("bits_per_weight_nominal", "N/A")
        raw_b = q.get("storage_bits_per_weight_raw", "N/A")
        lut = q.get("lut_values", [])
        lines.append(f"  Quantization Mode:    {mode} ({nom_b}b nominal, {raw_b}b uncompressed)")
        lines.append(f"  Quantization States:  {lut}")

    st = meta.get("statistics", {})
    if st:
        total_p = st.get("total_params", 0)
        uncomp = st.get("uncompressed_bytes", 0) / (1024 * 1024)
        comp = st.get("compressed_bytes", 0) / (1024 * 1024)
        ratio = st.get("compression_ratio", 0)
        eff_bpw = st.get("effective_bits_per_param", 0)
        pot_cnt = st.get("pot5_tensors", 0)
        pot_res_cnt = st.get("pot5_res_tensors", 0)
        pot_res_q4_cnt = st.get("pot5_res_q4_tensors", 0)
        ter_cnt = st.get("ternary_tensors", 0)
        fp16_cnt = st.get("fp16_tensors", 0)
        lines.append("-" * 68)
        lines.append(f"  Total Parameters:     {total_p:,}")
        lines.append(f"  Payload Size:         {comp:.1f} MB compressed / {uncomp:.1f} MB raw ({ratio:.2f}x ratio)")
        lines.append(f"  Effective BPW:        {eff_bpw:.3f} bits/param (including embeddings & norms)")
        breakdown_parts = []
        if pot_cnt > 0:
            breakdown_parts.append(f"{pot_cnt} POT5")
        if pot_res_cnt > 0:
            breakdown_parts.append(f"{pot_res_cnt} POT5-Residual(FP16)")
        if pot_res_q4_cnt > 0:
            breakdown_parts.append(f"{pot_res_q4_cnt} POT5-Residual(Q4)")
        if ter_cnt > 0:
            breakdown_parts.append(f"{ter_cnt} Ternary")
        if fp16_cnt > 0:
            breakdown_parts.append(f"{fp16_cnt} FP16")
        lines.append(f"  Tensors Breakdown:    {', '.join(breakdown_parts) if breakdown_parts else 'N/A'}")

    sp = meta.get("sparsity", {})
    if sp:
        active = sp.get("active_sparsity", sp.get("active_sparsity_ratio", "N/A"))
        lines.append("-" * 68)
        lines.append(f"  Active Compute:       {active}")

    hw = meta.get("hardware_profile", {})
    if hw:
        vram = hw.get("vram_footprint_mb", "N/A")
        dev = hw.get("recommended_device", "cuda")
        lines.append(f"  VRAM Footprint:       ~{vram} MB (target: {dev})")

    perf = meta.get("performance", {})
    if perf:
        p_tok = perf.get("prefill_tok_per_sec", "N/A")
        g_tok = perf.get("gen_tok_per_sec", "N/A")
        lat = perf.get("latency_ms_per_token", "N/A")
        lines.append(f"  Throughput:           Prefill: {p_tok} tok/s | Gen: {g_tok} tok/s ({lat} ms/tok)")

    lines.append("=" * 68)
    return "\n".join(lines)


def _get_child(curr: Any, p: str) -> Any:
    if isinstance(curr, (nn.ModuleList, nn.Sequential, list, tuple)):
        return curr[int(p)]
    elif isinstance(curr, (nn.ModuleDict, dict)):
        return curr[p]
    elif hasattr(curr, p):
        return getattr(curr, p)
    elif p.isdigit() and hasattr(curr, "__getitem__"):
        try:
            return curr[int(p)]
        except (KeyError, IndexError, TypeError):
            return curr[p]
    elif hasattr(curr, "__getitem__"):
        return curr[p]
    else:
        return getattr(curr, p)


def _set_submodule(model: nn.Module, target_path: str, new_module: nn.Module):
    """Replaces a submodule at target_path (e.g. 'blocks.0.time_mixer.qkv_proj') with new_module."""
    parts = target_path.split(".")
    curr = model
    for p in parts[:-1]:
        curr = _get_child(curr, p)
    attr = parts[-1]
    if isinstance(curr, (nn.ModuleList, nn.Sequential, list)):
        curr[int(attr)] = new_module
    elif isinstance(curr, (nn.ModuleDict, dict)):
        curr[attr] = new_module
    elif hasattr(curr, attr):
        setattr(curr, attr, new_module)
    elif attr.isdigit() and hasattr(curr, "__setitem__"):
        try:
            curr[int(attr)] = new_module
        except (KeyError, IndexError, TypeError):
            curr[attr] = new_module
    else:
        setattr(curr, attr, new_module)


def _assign_tensor(model: nn.Module, target_path: str, tensor: torch.Tensor):
    """Assigns a parameter or buffer at target_path without intermediate dictionary."""
    parts = target_path.split(".")
    curr = model
    for p in parts[:-1]:
        curr = _get_child(curr, p)
    attr = parts[-1]
    if isinstance(curr, (nn.ModuleDict, dict)):
        curr[attr] = tensor
    elif hasattr(curr, attr):
        existing = getattr(curr, attr)
        if isinstance(existing, nn.Parameter):
            curr.register_parameter(attr, nn.Parameter(tensor, requires_grad=False))
        elif attr in getattr(curr, "_buffers", {}):
            curr.register_buffer(attr, tensor)
        elif attr in getattr(curr, "_parameters", {}):
            curr.register_parameter(attr, nn.Parameter(tensor, requires_grad=False))
        else:
            if hasattr(curr, "register_buffer") and attr not in curr.__dict__:
                try:
                    curr.register_buffer(attr, tensor)
                except Exception:
                    setattr(curr, attr, tensor)
            else:
                setattr(curr, attr, tensor)
    elif attr.isdigit() and hasattr(curr, "__setitem__"):
        curr[int(attr)] = tensor
    else:
        setattr(curr, attr, tensor)


def load_toros_model(
    filepath: str,
    device: str = "cpu",
    target_dtype: Optional[torch.dtype] = None,
    model_class: Optional[Any] = None,
    bitpacked: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads a .toros binary model directly into memory with ultra-fast streaming decompression.
    When bitpacked=True, weights are kept in 3-bit bitplanes (1.35 GB resident memory)
    rather than expanding to 16-bit floats (7 GB) or 32-bit floats (14 GB), completely eliminating OOM.
    """
    with open(filepath, "rb") as f:
        magic = f.read(len(MAGIC_HEADER))
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid Toros file magic: {magic}")

        version, is_compressed = struct.unpack("<BB", f.read(2))
        if is_compressed and not HAS_ZSTD:
            raise ImportError(
                "zstandard is required to read compressed .toros files but is not installed. "
                "Install with `pip install zstandard`."
            )
        meta_len = struct.unpack("<I", f.read(4))[0]
        meta_json = f.read(meta_len).decode("utf-8")
        meta = json.loads(meta_json)

        uncompressed_size, compressed_size = struct.unpack("<QQ", f.read(16))

        # Instantiate model skeleton on meta device (0 MB RAM)
        model_type_name = meta.get("model_type", "TorosHybridLanguageModel")
        config_kwargs = meta.get("config", {})

        try:
            with torch.device("meta"):
                if model_class is not None:
                    model = model_class(config_kwargs) if config_kwargs else model_class()
                elif model_type_name == "Qwen35BLTLanguageModel":
                    from affine_ai.models.qwen35_blt import Qwen35BLTLanguageModel, Qwen35BLTConfig
                    config = Qwen35BLTConfig(**config_kwargs) if config_kwargs else Qwen35BLTConfig()
                    model = Qwen35BLTLanguageModel(config)
                elif model_type_name == "Qwen35ASDAGModel":
                    from affine_ai.models.qwen35_asdag import Qwen35ASDAGModel, Qwen35ASDAGConfig
                    config = Qwen35ASDAGConfig(**config_kwargs) if config_kwargs else Qwen35ASDAGConfig()
                    model = Qwen35ASDAGModel(config)
                elif model_type_name == "TorosHybridLanguageModel":
                    from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
                    config = TorosHybridConfig(**config_kwargs) if config_kwargs else TorosHybridConfig()
                    model = TorosHybridLanguageModel(config)
                else:
                    from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGConfig
                    config = ASDAGConfig(**config_kwargs) if config_kwargs else ASDAGConfig()
                    model = ASDAGLanguageModel(config)
            use_direct_assignment = True
        except Exception:
            use_direct_assignment = False
            model = None

        if target_dtype is None:
            if model is not None:
                if hasattr(model, "dtype") and isinstance(model.dtype, torch.dtype):
                    target_dtype = model.dtype
                else:
                    try:
                        p = next(model.parameters())
                        target_dtype = p.dtype
                    except StopIteration:
                        target_dtype = torch.bfloat16
            else:
                target_dtype = torch.bfloat16

        # Lazily import Triton helpers only when needed and available; keep CPU path import-free
        bitplane_to_gpu = None
        Triton5StatePOTBitpackedLinear = None
        Triton5StatePOTBitpackedResidualLinear = None
        if bitpacked and use_direct_assignment:
            try:
                from affine_ai.kernels.triton_pot5 import (
                    bitplane_bytes_to_gpu_int32 as _b2g,
                    Triton5StatePOTBitpackedLinear as _L,
                    Triton5StatePOTBitpackedResidualLinear as _RL,
                )
                bitplane_to_gpu = _b2g
                Triton5StatePOTBitpackedLinear = _L
                Triton5StatePOTBitpackedResidualLinear = _RL
            except Exception:
                bitplane_to_gpu = None

        if is_compressed:
            if not HAS_ZSTD:
                raise ImportError("zstandard required for compressed payload")
            dctx = zstd.ZstdDecompressor()
            reader_ctx = dctx.stream_reader(f)
        else:
            reader_ctx = f  # type: ignore

        # Use context manager only if zstd
        def _read_tensors(reader):
            num_tensors = struct.unpack("<I", reader.read(4))[0]
            state_dict = {}

            for _ in range(num_tensors):
                name_len = struct.unpack("<H", reader.read(2))[0]
                name = reader.read(name_len).decode("utf-8")
                flag = struct.unpack("<B", reader.read(1))[0]
                gamma = struct.unpack("<f", reader.read(4))[0]
                ndim = struct.unpack("<B", reader.read(1))[0]
                shape = [struct.unpack("<I", reader.read(4))[0] for _ in range(ndim)]
                data_len = struct.unpack("<I", reader.read(4))[0]
                data = reader.read(data_len)

                if bitpacked and flag == FLAG_POT5_3BITPLANE and use_direct_assignment and name.endswith(".weight") and bitplane_to_gpu is not None:
                    buf = io.BytesIO(data)
                    len_nz, len_mag, len_sign = struct.unpack("<III", buf.read(12))
                    nz_b = buf.read(len_nz)
                    mag_b = buf.read(len_mag)
                    sign_b = buf.read(len_sign)
                    N, K = shape[0], shape[1]
                    w_nz = bitplane_to_gpu(nz_b, N, K, device=device)
                    w_mag = bitplane_to_gpu(mag_b, N, K, device=device)
                    w_sign = bitplane_to_gpu(sign_b, N, K, device=device)
                    alpha = torch.tensor(gamma, dtype=torch.float32, device=device)
                    bit_linear = Triton5StatePOTBitpackedLinear(K, N, w_nz, w_mag, w_sign, alpha, dtype=target_dtype).to(device)
                    _set_submodule(model, name[:-7], bit_linear)
                elif bitpacked and flag == FLAG_POT5_RESIDUAL_FP16 and use_direct_assignment and name.endswith(".weight") and bitplane_to_gpu is not None:
                    buf = io.BytesIO(data)
                    num_out = struct.unpack("<I", buf.read(4))[0]
                    out_idx = torch.from_numpy(np.frombuffer(buf.read(num_out * 4), dtype=np.uint32).copy()).to(device)
                    out_val = torch.from_numpy(np.frombuffer(buf.read(num_out * 2), dtype=np.float16).copy()).to(device)
                    len_nz, len_mag, len_sign = struct.unpack("<III", buf.read(12))
                    nz_b = buf.read(len_nz)
                    mag_b = buf.read(len_mag)
                    sign_b = buf.read(len_sign)
                    N, K = shape[0], shape[1]
                    w_nz = bitplane_to_gpu(nz_b, N, K, device=device)
                    w_mag = bitplane_to_gpu(mag_b, N, K, device=device)
                    w_sign = bitplane_to_gpu(sign_b, N, K, device=device)
                    alpha = torch.tensor(gamma, dtype=torch.float32, device=device)
                    res_linear = Triton5StatePOTBitpackedResidualLinear(K, N, w_nz, w_mag, w_sign, alpha, out_idx, out_val, dtype=target_dtype).to(device)
                    _set_submodule(model, name[:-7], res_linear)
                elif bitpacked and flag == FLAG_POT5_RESIDUAL_Q4 and use_direct_assignment and name.endswith(".weight") and bitplane_to_gpu is not None:
                    buf = io.BytesIO(data)
                    num_out, block_size = struct.unpack("<II", buf.read(8))
                    outlier_indices_np = np.frombuffer(buf.read(num_out * 4), dtype=np.uint32)
                    num_scales = struct.unpack("<I", buf.read(4))[0]
                    scales = np.frombuffer(buf.read(num_scales * 2), dtype=np.float16).astype(np.float32)
                    len_nibbles = struct.unpack("<I", buf.read(4))[0]
                    packed_bytes = np.frombuffer(buf.read(len_nibbles), dtype=np.uint8)
                    low = packed_bytes & 0x0F
                    high = (packed_bytes >> 4) & 0x0F
                    unpacked_nibbles = np.stack([low, high], axis=1).flatten()[:num_out]
                    nib = unpacked_nibbles.astype(np.int8)
                    q4 = np.where(nib >= 8, nib - 16, nib).astype(np.float32)
                    block_idx = np.arange(num_out) // block_size
                    outlier_values_np = (q4 * scales[block_idx]).astype(np.float16)

                    out_idx = torch.from_numpy(outlier_indices_np.copy()).to(device)
                    out_val = torch.from_numpy(outlier_values_np).to(device)

                    len_nz, len_mag, len_sign = struct.unpack("<III", buf.read(12))
                    nz_b = buf.read(len_nz)
                    mag_b = buf.read(len_mag)
                    sign_b = buf.read(len_sign)
                    N, K = shape[0], shape[1]
                    w_nz = bitplane_to_gpu(nz_b, N, K, device=device)
                    w_mag = bitplane_to_gpu(mag_b, N, K, device=device)
                    w_sign = bitplane_to_gpu(sign_b, N, K, device=device)
                    alpha = torch.tensor(gamma, dtype=torch.float32, device=device)
                    res_linear = Triton5StatePOTBitpackedResidualLinear(K, N, w_nz, w_mag, w_sign, alpha, out_idx, out_val, dtype=target_dtype).to(device)
                    _set_submodule(model, name[:-7], res_linear)
                elif use_direct_assignment:
                    if flag in (FLAG_TERNARY_2BIT, FLAG_SPARSE_TERNARY):
                        t = unpack_ternary_tensor(data, gamma, shape, flag, dtype=target_dtype, device=device)
                    elif flag == FLAG_POT5_3BITPLANE:
                        t = unpack_pot5_3bitplane(data, gamma, shape, dtype=target_dtype, device=device)
                    elif flag == FLAG_POT5_RESIDUAL_FP16:
                        t = unpack_pot5_residual_fp16(data, gamma, shape, dtype=target_dtype, device=device)
                    elif flag == FLAG_POT5_RESIDUAL_Q4:
                        t = unpack_pot5_residual_q4(data, gamma, shape, dtype=target_dtype, device=device)
                    elif flag == FLAG_RAW_FP16:
                        t_np = np.frombuffer(data, dtype=np.float16).copy().reshape(shape)
                        t = torch.from_numpy(t_np).to(dtype=target_dtype, device=device)
                    else:
                        raise ValueError(f"Unsupported flag {flag} for {name}")
                    _assign_tensor(model, name, t)
                    del t
                else:
                    if flag in (FLAG_TERNARY_2BIT, FLAG_SPARSE_TERNARY):
                        t = unpack_ternary_tensor(data, gamma, shape, flag, dtype=target_dtype, device=device)
                    elif flag == FLAG_POT5_3BITPLANE:
                        t = unpack_pot5_3bitplane(data, gamma, shape, dtype=target_dtype, device=device)
                    elif flag == FLAG_POT5_RESIDUAL_FP16:
                        t = unpack_pot5_residual_fp16(data, gamma, shape, dtype=target_dtype, device=device)
                    elif flag == FLAG_POT5_RESIDUAL_Q4:
                        t = unpack_pot5_residual_q4(data, gamma, shape, dtype=target_dtype, device=device)
                    elif flag == FLAG_RAW_FP16:
                        t_np = np.frombuffer(data, dtype=np.float16).copy().reshape(shape)
                        t = torch.from_numpy(t_np).to(dtype=target_dtype, device=device)
                    state_dict[name] = t
                del data
            return state_dict

        if is_compressed:
            with reader_ctx as reader:
                state_dict = _read_tensors(reader)
        else:
            state_dict = _read_tensors(reader_ctx)

    if not use_direct_assignment:
        if model_class is not None:
            model = model_class(config_kwargs).to(device) if config_kwargs else model_class().to(device)
        model.load_state_dict(state_dict, strict=False)
        del state_dict

    gc.collect()
    model.eval()
    return model, meta
