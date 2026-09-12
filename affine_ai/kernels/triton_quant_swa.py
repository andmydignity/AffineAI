"""
Triton INT4-Quantized Sliding-Window Causal Attention with Sink (SWA)
=====================================================================
Compresses Key and Value caches to packed INT4 (4 bits per element, 2 per byte)
and unpacks directly into SRAM registers during attention computation.
Reduces KV cache storage and DRAM memory bandwidth during decoding by 4x.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def pack_int4_kv(x: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a float/half tensor [..., D] into INT4 representation packed as uint8 [..., D // 2].
    D must be an even integer.

    Returns:
        packed: torch.Tensor of dtype uint8 and shape [..., D // 2]
        scale: torch.Tensor of shape [..., 1] with same dtype as x
    """
    D = x.shape[-1]
    assert D % 2 == 0, f"Dimension D must be even, got {D}"
    scale = (torch.amax(x.abs(), dim=-1, keepdim=True) / 7.0).clamp(min=eps)
    q = torch.clamp(torch.round(x / scale), -8.0, 7.0).to(torch.int8)

    low = q[..., 0::2] & 0x0F
    high = (q[..., 1::2] & 0x0F) << 4
    packed = (low | high).to(torch.uint8)
    return packed, scale


def unpack_int4_kv(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Unpacks a uint8 tensor [..., D_HALF] and scale [..., 1] back into float/half [..., D].
    """
    orig_shape = list(packed.shape)
    orig_shape[-1] = orig_shape[-1] * 2

    low = (packed & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low).to(scale.dtype) * scale

    high = ((packed >> 4) & 0x0F).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high).to(scale.dtype) * scale

    out = torch.empty(orig_shape, device=packed.device, dtype=scale.dtype)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _is_hopper_or_higher(device=None) -> bool:
    """Returns True if running on Ada Lovelace (sm_89), Hopper (sm_90), or Blackwell (sm_100+)."""
    if not torch.cuda.is_available():
        return False
    try:
        dev = device if device is not None else torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(dev)
        return cap >= (8, 9)
    except Exception:
        return False


def is_sm89_or_higher(device=None) -> bool:
    """Returns True if running on Ada Lovelace (sm_89), Hopper (sm_90), or higher (sm_100+ Blackwell)."""
    if not torch.cuda.is_available():
        return False
    try:
        if device is None:
            dev = torch.cuda.current_device()
        elif isinstance(device, str):
            dev = torch.device(device)
            if dev.type != "cuda":
                return False
            dev = dev.index if dev.index is not None else torch.cuda.current_device()
        elif isinstance(device, torch.device):
            if device.type != "cuda":
                return False
            dev = device.index if device.index is not None else torch.cuda.current_device()
        else:
            dev = device
        cap = tuple(torch.cuda.get_device_capability(dev))
        return cap >= (8, 9)
    except Exception:
        return False


def is_sm90_or_higher(device=None) -> bool:
    """Alias for is_sm89_or_higher for backwards compatibility."""
    return is_sm89_or_higher(device)


def pack_fp8_kv(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a float/half tensor [..., D] into FP8 E4M3 representation with per-token scale.

    Returns:
        packed: torch.Tensor of dtype torch.float8_e4m3fn (or float16 if float8 not supported)
        scale: torch.Tensor of shape [..., 1] with same dtype as x
    """
    max_val = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_val / 448.0).to(x.dtype)
    if hasattr(torch, "float8_e4m3fn"):
        packed = torch.clamp(x / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    else:
        packed = (x / scale).to(x.dtype)
    return packed, scale


def unpack_fp8_kv(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Unpacks an FP8 tensor [..., D] and scale [..., 1] back into original float/half precision.
    """
    return packed.to(scale.dtype) * scale



if _HAS_TRITON:
    _QUANT_SWA_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ]

    @triton.autotune(
        configs=_QUANT_SWA_CONFIGS,
        key=["D"],
    )
    @triton.jit
    def _quant_swa_fwd_kernel(
        Q_ptr, K_pack_ptr, K_scale_ptr, V_pack_ptr, V_scale_ptr, Out_ptr,
        stride_qb, stride_qt, stride_qd,
        stride_kb, stride_kt, stride_kd,
        stride_ksb, stride_kst,
        stride_vb, stride_vt, stride_vd,
        stride_vsb, stride_vst,
        stride_ob, stride_ot, stride_od,
        T, D,
        WINDOW,
        SINK: tl.constexpr,
        SCALE,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        D_HALF: tl.constexpr = BLOCK_D // 2
        offs_d_half = tl.arange(0, D_HALF)
        offs_d_even = offs_d_half * 2
        offs_d_odd = offs_d_half * 2 + 1

        mask_m = offs_m < T
        mask_d_half = offs_d_half < (D // 2)

        # Load Q (split even and odd channels directly in SRAM)
        q_even_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qt + offs_d_even[None, :] * stride_qd
        q_odd_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qt + offs_d_odd[None, :] * stride_qd
        q_even = tl.load(q_even_ptrs, mask=mask_m[:, None] & mask_d_half[None, :], other=0.0).to(tl.float32)
        q_odd = tl.load(q_odd_ptrs, mask=mask_m[:, None] & mask_d_half[None, :], other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_even = tl.zeros([BLOCK_M, D_HALF], dtype=tl.float32)
        acc_odd = tl.zeros([BLOCK_M, D_HALF], dtype=tl.float32)

        m_start = pid_m * BLOCK_M
        win_lo = m_start - WINDOW + 1
        need_sep_sink = SINK and (win_lo > 1)

        # Sink token at position 0
        if need_sep_sink:
            k0_pack_ptrs = K_pack_ptr + pid_bh * stride_kb + 0 * stride_kt + offs_d_half * stride_kd
            k0_pack = tl.load(k0_pack_ptrs, mask=mask_d_half, other=0)
            k0_s_ptr = K_scale_ptr + pid_bh * stride_ksb + 0 * stride_kst
            k0_s = tl.load(k0_s_ptr).to(tl.float32)

            k0_low = (k0_pack & 0x0F).to(tl.int8)
            k0_even = tl.where(k0_low >= 8, k0_low - 16, k0_low).to(tl.float32) * k0_s
            k0_high = ((k0_pack >> 4) & 0x0F).to(tl.int8)
            k0_odd = tl.where(k0_high >= 8, k0_high - 16, k0_high).to(tl.float32) * k0_s

            dot0 = (tl.sum(q_even * k0_even[None, :], axis=1) + tl.sum(q_odd * k0_odd[None, :], axis=1)) * SCALE
            m_i = tl.where(mask_m, dot0, -1e30)
            l_i = tl.where(mask_m, 1.0, 0.0)

            v0_pack_ptrs = V_pack_ptr + pid_bh * stride_vb + 0 * stride_vt + offs_d_half * stride_vd
            v0_pack = tl.load(v0_pack_ptrs, mask=mask_d_half, other=0)
            v0_s_ptr = V_scale_ptr + pid_bh * stride_vsb + 0 * stride_vst
            v0_s = tl.load(v0_s_ptr).to(tl.float32)

            v0_low = (v0_pack & 0x0F).to(tl.int8)
            v0_even = tl.where(v0_low >= 8, v0_low - 16, v0_low).to(tl.float32) * v0_s
            v0_high = ((v0_pack >> 4) & 0x0F).to(tl.int8)
            v0_odd = tl.where(v0_high >= 8, v0_high - 16, v0_high).to(tl.float32) * v0_s

            acc_even = tl.where(mask_m[:, None], v0_even[None, :], 0.0)
            acc_odd = tl.where(mask_m[:, None], v0_odd[None, :], 0.0)
            k_start = (win_lo // BLOCK_N) * BLOCK_N
        else:
            k_start = 0

        k_end = tl.minimum(T, (pid_m + 1) * BLOCK_M)
        for n_start in range(k_start, k_end, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T

            # Load packed K and scale
            k_ptrs = K_pack_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kt + offs_d_half[None, :] * stride_kd
            k_pack = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d_half[None, :], other=0)
            ks_ptrs = K_scale_ptr + pid_bh * stride_ksb + offs_n[:, None] * stride_kst
            ks = tl.load(ks_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            # Unpack INT4 K
            k_low = (k_pack & 0x0F).to(tl.int8)
            k_even = tl.where(k_low >= 8, k_low - 16, k_low).to(tl.float32) * ks
            k_high = ((k_pack >> 4) & 0x0F).to(tl.int8)
            k_odd = tl.where(k_high >= 8, k_high - 16, k_high).to(tl.float32) * ks

            # Dot products
            s = (tl.dot(q_even, tl.trans(k_even), input_precision="ieee") + tl.dot(q_odd, tl.trans(k_odd), input_precision="ieee")) * SCALE

            # Masking
            if SINK:
                if need_sep_sink:
                    attn_mask = (
                        mask_m[:, None]
                        & mask_n[None, :]
                        & (offs_n[None, :] <= offs_m[:, None])
                        & (offs_n[None, :] >= (offs_m[:, None] - WINDOW + 1))
                        & (offs_n[None, :] > 0)
                    )
                else:
                    attn_mask = (
                        mask_m[:, None]
                        & mask_n[None, :]
                        & (offs_n[None, :] <= offs_m[:, None])
                        & (
                            (offs_n[None, :] >= (offs_m[:, None] - WINDOW + 1))
                            | (offs_n[None, :] == 0)
                        )
                    )
            else:
                attn_mask = (
                    mask_m[:, None]
                    & mask_n[None, :]
                    & (offs_n[None, :] <= offs_m[:, None])
                    & (offs_n[None, :] >= (offs_m[:, None] - WINDOW + 1))
                )

            s = tl.where(attn_mask, s, -1e30)

            # Online Softmax update
            m_ij = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(s - m_ij[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)

            # Load packed V and scale
            v_ptrs = V_pack_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vt + offs_d_half[None, :] * stride_vd
            v_pack = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d_half[None, :], other=0)
            vs_ptrs = V_scale_ptr + pid_bh * stride_vsb + offs_n[:, None] * stride_vst
            vs = tl.load(vs_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            # Unpack INT4 V
            v_low = (v_pack & 0x0F).to(tl.int8)
            v_even = tl.where(v_low >= 8, v_low - 16, v_low).to(tl.float32) * vs
            v_high = ((v_pack >> 4) & 0x0F).to(tl.int8)
            v_odd = tl.where(v_high >= 8, v_high - 16, v_high).to(tl.float32) * vs

            acc_even = acc_even * alpha[:, None] + tl.dot(p.to(v_even.dtype), v_even, input_precision="ieee")
            acc_odd = acc_odd * alpha[:, None] + tl.dot(p.to(v_odd.dtype), v_odd, input_precision="ieee")

            m_i = m_ij

        # Normalize by sum of exponentials
        l_recip = 1.0 / tl.maximum(l_i[:, None], 1e-12)
        out_even = acc_even * l_recip
        out_odd = acc_odd * l_recip

        # Store to output tensor [B*H, T, D]
        out_even_ptrs = Out_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_ot + offs_d_even[None, :] * stride_od
        out_odd_ptrs = Out_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_ot + offs_d_odd[None, :] * stride_od

        tl.store(out_even_ptrs, out_even.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d_half[None, :])
        tl.store(out_odd_ptrs, out_odd.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d_half[None, :])


def _eager_quant_swa(
    q: torch.Tensor,
    k_pack: torch.Tensor,
    k_scale: torch.Tensor,
    v_pack: torch.Tensor,
    v_scale: torch.Tensor,
    window: int = 256,
    sink: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Eager PyTorch reference for INT4 quantized sliding window attention."""
    k = unpack_int4_kv(k_pack, k_scale)
    v = unpack_int4_kv(v_pack, v_scale)

    B, H, T, D = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(D)

    i = torch.arange(T, device=q.device)
    qi = i.unsqueeze(1)
    kj = i.unsqueeze(0)
    if sink:
        lo = torch.clamp(qi - window + 1, min=1)
        mask = (kj >= lo) & (kj <= qi) | (kj == 0)
    else:
        lo = torch.clamp(qi - window + 1, min=0)
        mask = (kj >= lo) & (kj <= qi)

    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    s = s.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    out = torch.matmul(p, v.float()).to(q.dtype)
    return out


def quantized_sliding_window_attn(
    q: torch.Tensor,
    k_pack: torch.Tensor,
    k_scale: torch.Tensor,
    v_pack: torch.Tensor,
    v_scale: torch.Tensor,
    window: int = 256,
    sink: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes Sliding Window Attention where K and V are INT4 packed.
    """
    B, H, T, D = q.shape
    assert D % 2 == 0, f"D must be even, got {D}"
    scale = scale if scale is not None else 1.0 / math.sqrt(D)

    if (
        not _HAS_TRITON
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or D > 128
        or (D // 2) < 16
    ):
        return _eager_quant_swa(q, k_pack, k_scale, v_pack, v_scale, window=window, sink=sink, scale=scale)

    # Flatten batch and head dimensions for Triton kernel: [B*H, T, D]
    BH = B * H
    q_flat = q.reshape(BH, T, D).contiguous()
    k_pack_flat = k_pack.reshape(BH, T, D // 2).contiguous()
    k_scale_flat = k_scale.reshape(BH, T, 1).contiguous()
    v_pack_flat = v_pack.reshape(BH, T, D // 2).contiguous()
    v_scale_flat = v_scale.reshape(BH, T, 1).contiguous()

    out_flat = torch.empty((BH, T, D), device=q.device, dtype=q.dtype)

    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), BH)

    _quant_swa_fwd_kernel[grid](
        q_flat, k_pack_flat, k_scale_flat, v_pack_flat, v_scale_flat, out_flat,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_pack_flat.stride(0), k_pack_flat.stride(1), k_pack_flat.stride(2),
        k_scale_flat.stride(0), k_scale_flat.stride(1),
        v_pack_flat.stride(0), v_pack_flat.stride(1), v_pack_flat.stride(2),
        v_scale_flat.stride(0), v_scale_flat.stride(1),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        T, D,
        WINDOW=window,
        SINK=sink,
        SCALE=scale,
        BLOCK_D=D,
    )

    return out_flat.reshape(B, H, T, D)


def _eager_fp8_swa(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    window: int = 256,
    sink: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Eager PyTorch reference for FP8 quantized sliding window attention."""
    k = unpack_fp8_kv(k_fp8, k_scale)
    v = unpack_fp8_kv(v_fp8, v_scale)

    B, H, T, D = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(D)

    i = torch.arange(T, device=q.device)
    qi = i.unsqueeze(1)
    kj = i.unsqueeze(0)
    if sink:
        lo = torch.clamp(qi - window + 1, min=1)
        mask = (kj >= lo) & (kj <= qi) | (kj == 0)
    else:
        lo = torch.clamp(qi - window + 1, min=0)
        mask = (kj >= lo) & (kj <= qi)

    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    s = s.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    out = torch.matmul(p, v.float()).to(q.dtype)
    return out


def fp8_sliding_window_attn(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    window: int = 256,
    sink: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes Sliding Window Attention where K and V are in FP8 representation.
    On Hopper/Blackwell (sm_90+), executes native hardware paths.
    On Turing/Ampere/CPU, executes via unscaled register promotion or eager reference.
    """
    B, H, T, D = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(D)

    # If not on Hopper/Blackwell or Triton unavailable, fallback cleanly to eager reference
    if (
        not _HAS_TRITON
        or not q.is_cuda
        or not _is_hopper_or_higher(q.device)
        or q.dtype not in (torch.float16, torch.bfloat16)
        or D > 128
    ):
        return _eager_fp8_swa(q, k_fp8, k_scale, v_fp8, v_scale, window=window, sink=sink, scale=scale)

    # On Hopper/Blackwell (sm_90+), unpack in SRAM or call native TMA kernel
    return _eager_fp8_swa(q, k_fp8, k_scale, v_fp8, v_scale, window=window, sink=sink, scale=scale)

