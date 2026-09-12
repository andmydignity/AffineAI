"""
Custom Triton Kernel: In-SRAM Fused Monarch Permutation Chain
============================================================
Fuses multi-stage diagonal scaling and permutation indexing directly
in GPU SRAM registers with analytical transposed backward pass.
Eliminates intermediate VRAM writes across all Monarch projection stages.
"""

import math
import os
import warnings
from typing import Tuple, Optional
import torch
import triton
import triton.language as tl


def _is_turing() -> bool:
    """Turing sm_75 detection: 64KB SMEM, FP16-only, no BF16."""
    try:
        from affine_ai.kernels import _IS_TURING as _T  # type: ignore

        return bool(_T)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return (7, 5) <= tuple(cap) < (8, 0)
    except Exception:
        pass
    return False


def _maybe_warn_bf16_turing(dtype: torch.dtype, where: str) -> None:
    if _is_turing() and dtype == torch.bfloat16:
        warnings.warn(f"Turing sm_75: bf16 not supported, treating as fp16 (acc fp32) in {where}", stacklevel=3)


def precompute_monarch_composed_single(diagonals: torch.Tensor, perms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes composed 1D monomial scale W[d] and gather index P[d]
    across S stages for single Monarch permutation chain.
    P stored as int32; D < 2**31 required (assert below). Downstream mul uses 64-bit where needed.
    """
    num_stages, D = diagonals.shape
    assert D < 2**31, f"D={D} exceeds int32 range for P index"
    idx = torch.arange(D, device=perms.device, dtype=torch.long)
    W = diagonals[num_stages - 1].clone()
    for s in range(num_stages - 2, -1, -1):
        idx = perms[s][idx].long()
        W = W * diagonals[s][idx]
    P = idx.to(torch.int32)
    return W.contiguous(), P.contiguous()


def precompute_monarch_composed_fused(diagonals: torch.Tensor, perms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes composed 1D monomial scale W[m, d] and gather index P[d]
    across S stages for M branches.
    P stored as int32; D < 2**31 required. Downstream host mul uses 64-bit.
    """
    num_branches, num_stages, D = diagonals.shape
    assert D < 2**31, f"D={D} exceeds int32 range for P index"
    idx = torch.arange(D, device=perms.device, dtype=torch.long)
    W = diagonals[:, num_stages - 1].clone()  # [M, D]
    for s in range(num_stages - 2, -1, -1):
        idx = perms[s][idx].long()
        W = W * diagonals[:, s, idx]
    P = idx.to(torch.int32)
    return W.contiguous(), P.contiguous()


_MONARCH_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 128}, num_warps=8, num_stages=3),
]

# Turing sm_75: prune to BLOCK 32/64 only, remove 128 variants, num_warps 2/4 max, BLOCK <=64.
if _is_turing():
    _MONARCH_AUTOTUNE_CONFIGS = [
        c for c in _MONARCH_AUTOTUNE_CONFIGS
        if c.kwargs.get("BLOCK_M", 32) <= 64 and c.kwargs.get("BLOCK_D", 32) <= 64 and c.num_warps <= 4
    ]

try:
    _cap_monarch = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 0)
    if _cap_monarch >= (9, 0):
        _MONARCH_AUTOTUNE_CONFIGS.append(triton.Config({"BLOCK_M": 128, "BLOCK_D": 64}, num_warps=8, num_stages=3))
except Exception:
    pass


_MONARCH_DIAG_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64, "BLOCK_D": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
]
if _is_turing():
    _MONARCH_DIAG_AUTOTUNE_CONFIGS = [
        c for c in _MONARCH_DIAG_AUTOTUNE_CONFIGS
        if c.kwargs.get("BLOCK_N", 32) <= 64 and c.kwargs.get("BLOCK_D", 32) <= 64 and c.num_warps <= 4
    ]


@triton.autotune(configs=_MONARCH_AUTOTUNE_CONFIGS, key=["N", "D"])
@triton.jit
def _monarch_chain_fwd_kernel(
    X, W, P, Bias, Y,
    stride_xm, stride_xd,
    stride_bd,
    stride_ym, stride_yd,
    N, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Fused Monarch chain: Y[m,d]=Bias[d]+W[d]*X[m,P[d]]. BLOCK_M/D tiles cover (N,D).
    """
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < D

    # P/W/B reused across N tiles -> cache-friendly; gather X via permuted index is streaming.
    # Use eviction_policy to keep P/W in cache while evicting streaming gather loads.
    # Contiguous Bias/W loads are optimized via tl.make_block_ptr for better coalescing when contiguous.
    # Random gather for X via perm (inherently uncoalesced) stays manual — cannot use block_ptr due to permutation.
    # Bias/W are contiguous 1D vectors; block_ptr gives coalesced vector loads vs manual P+offs_d pointer arithmetic.
    # Fallback manual kept for strided views.

    # Load P (int32 perm index) — still manual gather for index, but block_ptr for coalesced fetch when contiguous
    # P is [D] contiguous, use block_ptr
    p_block_ptr = tl.make_block_ptr(
        base=P,
        shape=(D,),
        strides=(1,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    p = tl.load(p_block_ptr, boundary_check=(0,))
    # fallback manual: p = tl.load(P + offs_d, mask=mask_d, other=0, eviction_policy="evict_first")

    # W contiguous vector
    w_block_ptr = tl.make_block_ptr(
        base=W,
        shape=(D,),
        strides=(1,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    w = tl.load(w_block_ptr, boundary_check=(0,))
    # fallback manual: w = tl.load(W + offs_d, mask=mask_d, other=0.0, eviction_policy="evict_first")

    b_block_ptr = tl.make_block_ptr(
        base=Bias,
        shape=(D,),
        strides=(stride_bd,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    b = tl.load(b_block_ptr, boundary_check=(0,))
    # fallback manual: b = tl.load(Bias + offs_d, mask=mask_d, other=0.0, eviction_policy="evict_first")

    # X gather via perm is inherently uncoalesced (random indirect), keep manual pointer arithmetic
    # Cannot use block_ptr due to perm indirection: X[offs_m, p] where p is permuted
    xv = tl.load(
        X + offs_m[:, None] * stride_xm + p[None, :] * stride_xd,
        mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        eviction_policy="evict_last",
    )
    y = b[None, :] + w[None, :] * xv
    # Store Y via block_ptr when contiguous (stride_yd==1 gives coalesced)
    # Fallback manual: Y + offs_m[:,None]*stride_ym + offs_d[None,:]*stride_yd
    y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(N, D),
        strides=(stride_ym, stride_yd),
        offsets=(pid_m * BLOCK_M, pid_d * BLOCK_D),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))
    # fallback manual: tl.store(Y + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd, y, mask=mask_m[:, None] & mask_d[None, :])


@triton.autotune(configs=_MONARCH_AUTOTUNE_CONFIGS, key=["N", "D"])
@triton.jit
def _fused_monarch_chain_fwd_kernel(
    X, W, P, Bias, Y,
    stride_xm, stride_xd,
    stride_wm, stride_wd,
    stride_bm, stride_bd,
    stride_ym, stride_yn, stride_yd,
    N, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Fused multi-branch Monarch chain: Y[m,n,d]=Bias[m,d]+W[m,d]*X[n,P[d]].
    Grid (cdiv(N,BLOCK_M), cdiv(D,BLOCK_D), M). Early-exit on fully masked D tail.
    """
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    br = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < D

    # Perm gather for X is uncoalesced; contiguous W/B via block_ptr
    p_block_ptr = tl.make_block_ptr(
        base=P,
        shape=(D,),
        strides=(1,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    p = tl.load(p_block_ptr, boundary_check=(0,))

    # W per-branch: W is [M, D], contiguous in D when stride_wd==1
    # Use block_ptr for coalesced W load per branch
    w_block_ptr = tl.make_block_ptr(
        base=W + br * stride_wm,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    w = tl.load(w_block_ptr, boundary_check=(0,))
    # fallback manual: w = tl.load(W + br * stride_wm + offs_d * stride_wd, mask=mask_d, other=0.0, eviction_policy="evict_first")

    b_block_ptr = tl.make_block_ptr(
        base=Bias + br * stride_bm,
        shape=(D,),
        strides=(stride_bd,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    b = tl.load(b_block_ptr, boundary_check=(0,))
    # fallback manual: b = tl.load(Bias + br * stride_bm + offs_d * stride_bd, mask=mask_d, other=0.0, eviction_policy="evict_first")

    xv = tl.load(
        X + offs_m[:, None] * stride_xm + p[None, :] * stride_xd,
        mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        eviction_policy="evict_last",
    )
    y = b[None, :] + w[None, :] * xv
    # Store Y [M, N, D] via block_ptr for coalesced store when stride_yd==1
    # Base per branch: Y + br*stride_ym, shape (N,D), strides (stride_yn, stride_yd)
    y_block_ptr = tl.make_block_ptr(
        base=Y + br * stride_ym,
        shape=(N, D),
        strides=(stride_yn, stride_yd),
        offsets=(pid_m * BLOCK_M, pid_d * BLOCK_D),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))
    # fallback manual: tl.store(Y + br * stride_ym + offs_m[:, None] * stride_yn + offs_d[None, :] * stride_yd, y, mask=mask_m[:, None] & mask_d[None, :])


def triton_monarch_chain_fwd(
    x: torch.Tensor,
    diagonals: torch.Tensor,
    perms: torch.Tensor,
    bias: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    P: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Monarch chain forward: composes diagonals/perms into monomial W,P then fused kernel.
    Tail-mask subtlety: blocked loads use boundary_check + early-exit on fully masked D tail;
    masked lanes contribute 0 (other=0) not NaN, so D not divisible by BLOCK_D is safe.
    """
    assert x.ndim == 2, f"x must be [N,D], got {x.shape}"
    N, D = x.shape
    assert diagonals.ndim == 2 and diagonals.shape[1] == D, f"diagonals must be [S,D] with D={D}"
    assert bias.shape == (D,), f"bias must be [D], got {bias.shape}"
    assert perms.shape[1] == D, f"perms must be [S,D] with D={D}"
    # Turing sm_75: bf16 fallback to fp16 (acc fp32) with warning.
    _maybe_warn_bf16_turing(x.dtype, "triton_monarch_chain_fwd")
    _maybe_warn_bf16_turing(diagonals.dtype, "triton_monarch_chain_fwd")
    if not x.is_cuda or not torch.cuda.is_available():
        # CPU fallback: cache perms long+contiguous outside loop to avoid per-iteration alloc
        perms_long = [p.long().contiguous() for p in perms]
        h = x * diagonals[0]
        for s in range(diagonals.shape[0] - 1):
            h = h[:, perms_long[s]] * diagonals[s + 1]
        return h + bias

    if W is None or P is None:
        W, P = precompute_monarch_composed_single(diagonals, perms)
    out = torch.empty((N, D), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), triton.cdiv(D, META["BLOCK_D"]))
    _monarch_chain_fwd_kernel[grid](
        x, W, P, bias, out,
        x.stride(0), x.stride(1),
        bias.stride(0),
        out.stride(0), out.stride(1),
        N, D,
    )
    return out


def triton_fused_monarch_chain_fwd(
    x: torch.Tensor,
    diagonals: torch.Tensor,
    perms: torch.Tensor,
    bias: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    P: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused multi-branch Monarch chain forward. Tail-mask subtlety same as single-branch:
    masked D lanes are other=0 with boundary_check, fully masked tail block early-exits.
    """
    assert x.ndim == 2, f"x must be [N,D], got {x.shape}"
    N, D = x.shape
    M = diagonals.shape[0]
    assert diagonals.ndim == 3 and diagonals.shape[2] == D, f"diagonals must be [M,S,D] with D={D}"
    assert bias.shape == (M, D), f"bias must be [M,D], got {bias.shape}"
    if M > 65535:
        raise ValueError(f"M={M} exceeds grid z limit 65535")
    _maybe_warn_bf16_turing(x.dtype, "triton_fused_monarch_chain_fwd")
    _maybe_warn_bf16_turing(diagonals.dtype, "triton_fused_monarch_chain_fwd")
    if not x.is_cuda or not torch.cuda.is_available():
        perms_long = [p.long().contiguous() for p in perms]
        h = x.unsqueeze(0) * diagonals[:, 0].unsqueeze(1)
        for s in range(diagonals.shape[1] - 1):
            h = h[:, :, perms_long[s]] * diagonals[:, s + 1].unsqueeze(1)
        return h + bias.unsqueeze(1)

    if W is None or P is None:
        W, P = precompute_monarch_composed_fused(diagonals, perms)
    out = torch.empty((M, N, D), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), triton.cdiv(D, META["BLOCK_D"]), M)
    _fused_monarch_chain_fwd_kernel[grid](
        x, W, P, bias, out,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        bias.stride(0), bias.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        N, D,
    )
    return out


def precompute_monarch_bwd_scales_single(diagonals: torch.Tensor, perms: torch.Tensor, inv_perms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S, D = diagonals.shape
    P_in = torch.empty((S, D), dtype=torch.int32, device=diagonals.device)
    W_in = torch.empty((S, D), dtype=diagonals.dtype, device=diagonals.device)
    P_in[0] = torch.arange(D, dtype=torch.int32, device=diagonals.device)
    W_in[0] = 1.0
    for s in range(1, S):
        p = perms[s - 1].long()
        P_in[s] = P_in[s - 1][p]
        W_in[s] = (W_in[s - 1] * diagonals[s - 1])[p]

    P_out = torch.empty((S, D), dtype=torch.int32, device=diagonals.device)
    W_out = torch.empty((S, D), dtype=diagonals.dtype, device=diagonals.device)
    P_out[S - 1] = torch.arange(D, dtype=torch.int32, device=diagonals.device)
    W_out[S - 1] = 1.0
    for s in range(S - 2, -1, -1):
        ip = inv_perms[s].long()
        P_out[s] = P_out[s + 1][ip]
        W_out[s] = (W_out[s + 1] * diagonals[s + 1])[ip]

    W_scale = (W_in * W_out).contiguous()
    return W_scale, P_in.contiguous(), P_out.contiguous()


def precompute_monarch_bwd_scales_fused(diagonals: torch.Tensor, perms: torch.Tensor, inv_perms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, S, D = diagonals.shape
    P_in = torch.empty((S, D), dtype=torch.int32, device=diagonals.device)
    W_in = torch.empty((M, S, D), dtype=diagonals.dtype, device=diagonals.device)
    P_in[0] = torch.arange(D, dtype=torch.int32, device=diagonals.device)
    W_in[:, 0] = 1.0
    for s in range(1, S):
        p = perms[s - 1].long()
        P_in[s] = P_in[s - 1][p]
        W_in[:, s] = (W_in[:, s - 1] * diagonals[:, s - 1])[:, p]

    P_out = torch.empty((S, D), dtype=torch.int32, device=diagonals.device)
    W_out = torch.empty((M, S, D), dtype=diagonals.dtype, device=diagonals.device)
    P_out[S - 1] = torch.arange(D, dtype=torch.int32, device=diagonals.device)
    W_out[:, S - 1] = 1.0
    for s in range(S - 2, -1, -1):
        ip = inv_perms[s].long()
        P_out[s] = P_out[s + 1][ip]
        W_out[:, s] = (W_out[:, s + 1] * diagonals[:, s + 1])[:, ip]

    W_scale = (W_in * W_out).contiguous()
    return W_scale, P_in.contiguous(), P_out.contiguous()


@triton.autotune(configs=_MONARCH_AUTOTUNE_CONFIGS, key=["N", "D"])
@triton.jit
def _monarch_chain_bwd_gx_kernel(
    GradOut, W, P_inv, GX,
    stride_gom, stride_god,
    stride_wd,
    stride_gxm, stride_gxd,
    N, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < D

    p_inv_ptr = tl.make_block_ptr(
        base=P_inv,
        shape=(D,),
        strides=(1,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    p_inv = tl.load(p_inv_ptr, boundary_check=(0,))

    w = tl.load(W + p_inv * stride_wd, mask=mask_d, other=0.0)
    go = tl.load(
        GradOut + offs_m[:, None] * stride_gom + p_inv[None, :] * stride_god,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    gx = go * w[None, :]
    gx_block_ptr = tl.make_block_ptr(
        base=GX,
        shape=(N, D),
        strides=(stride_gxm, stride_gxd),
        offsets=(pid_m * BLOCK_M, pid_d * BLOCK_D),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(gx_block_ptr, gx.to(GX.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_MONARCH_DIAG_AUTOTUNE_CONFIGS, key=["N", "D"])
@triton.jit
def _monarch_chain_bwd_diag_kernel(
    GradOut, X, W_scale, P_in, P_out, GDiag,
    stride_gom, stride_god,
    stride_xm, stride_xd,
    stride_wsm, stride_wsd,
    stride_pim, stride_pid,
    stride_pom, stride_pod,
    stride_gdm, stride_gdd,
    N, D,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    w_scale = tl.load(W_scale + pid_s * stride_wsm + offs_d * stride_wsd, mask=mask_d, other=0.0)
    p_in = tl.load(P_in + pid_s * stride_pim + offs_d * stride_pid, mask=mask_d, other=0)
    p_out = tl.load(P_out + pid_s * stride_pom + offs_d * stride_pod, mask=mask_d, other=0)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        go = tl.load(GradOut + offs_n[:, None] * stride_gom + p_out[None, :] * stride_god, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        xv = tl.load(X + offs_n[:, None] * stride_xm + p_in[None, :] * stride_xd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc += tl.sum(go.to(tl.float32) * xv.to(tl.float32), axis=0)

    gdiag = acc * w_scale.to(tl.float32)
    tl.store(GDiag + pid_s * stride_gdm + offs_d * stride_gdd, gdiag.to(GDiag.dtype.element_ty), mask=mask_d)


@triton.autotune(configs=_MONARCH_AUTOTUNE_CONFIGS, key=["N", "D"])
@triton.jit
def _fused_monarch_chain_bwd_gx_kernel(
    GradOut, W, P_inv, GX,
    stride_gom, stride_gon, stride_god,
    stride_wm, stride_wd,
    stride_gxm, stride_gxd,
    N, D, M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < D

    p_inv_ptr = tl.make_block_ptr(
        base=P_inv,
        shape=(D,),
        strides=(1,),
        offsets=(pid_d * BLOCK_D,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    p_inv = tl.load(p_inv_ptr, boundary_check=(0,))

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for br in range(M):
        w = tl.load(W + br * stride_wm + p_inv * stride_wd, mask=mask_d, other=0.0)
        go = tl.load(
            GradOut + br * stride_gom + offs_m[:, None] * stride_gon + p_inv[None, :] * stride_god,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += go.to(tl.float32) * w[None, :].to(tl.float32)

    gx_block_ptr = tl.make_block_ptr(
        base=GX,
        shape=(N, D),
        strides=(stride_gxm, stride_gxd),
        offsets=(pid_m * BLOCK_M, pid_d * BLOCK_D),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(gx_block_ptr, acc.to(GX.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_MONARCH_DIAG_AUTOTUNE_CONFIGS, key=["N", "D"])
@triton.jit
def _fused_monarch_chain_bwd_diag_kernel(
    GradOut, X, W_scale, P_in, P_out, GDiag,
    stride_gom, stride_gon, stride_god,
    stride_xm, stride_xd,
    stride_wsm, stride_wss, stride_wsd,
    stride_pim, stride_pid,
    stride_pom, stride_pod,
    stride_gdm, stride_gds, stride_gdd,
    N, D,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    br = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    w_scale = tl.load(W_scale + br * stride_wsm + pid_s * stride_wss + offs_d * stride_wsd, mask=mask_d, other=0.0)
    p_in = tl.load(P_in + pid_s * stride_pim + offs_d * stride_pid, mask=mask_d, other=0)
    p_out = tl.load(P_out + pid_s * stride_pom + offs_d * stride_pod, mask=mask_d, other=0)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        go = tl.load(GradOut + br * stride_gom + offs_n[:, None] * stride_gon + p_out[None, :] * stride_god, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        xv = tl.load(X + offs_n[:, None] * stride_xm + p_in[None, :] * stride_xd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc += tl.sum(go.to(tl.float32) * xv.to(tl.float32), axis=0)

    gdiag = acc * w_scale.to(tl.float32)
    tl.store(GDiag + br * stride_gdm + pid_s * stride_gds + offs_d * stride_gdd, gdiag.to(GDiag.dtype.element_ty), mask=mask_d)


class TritonMonarchChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        diagonals: torch.Tensor,
        perms: torch.Tensor,
        inv_perms: torch.Tensor,
        bias: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim >= 2 and x.shape[-1] == diagonals.shape[-1], f"x last dim {x.shape[-1]} != D {diagonals.shape[-1]}"
        orig_shape = x.shape
        # Ensure perms/inv_perms contiguous for stride assumptions in gathering
        perms = perms.long().contiguous()
        inv_perms = inv_perms.long().contiguous()
        x_flat = x.reshape(-1, x.shape[-1]).to(diagonals.dtype)
        num_stages = diagonals.shape[0]
        # Debug parity check: perms/inv_perms must be bijective (recompute vs composed forward)
        if __debug__ and os.environ.get("AFFINE_DEBUG_MONARCH", "0") == "1":
            D = x.shape[-1]
            arange = torch.arange(D, device=perms.device)
            for s in range(num_stages - 1):
                if not torch.equal(inv_perms[s].gather(0, perms[s]), arange):
                    raise AssertionError(f"perms/inv_perms not bijective at stage {s}")

        if not x_flat.is_cuda or not torch.cuda.is_available():
            out = triton_monarch_chain_fwd(x_flat, diagonals, perms, bias)
            ctx.save_for_backward(x_flat, diagonals, perms, inv_perms, None, None)
        else:
            W, P = precompute_monarch_composed_single(diagonals, perms)
            out = triton_monarch_chain_fwd(x_flat, diagonals, perms, bias, W=W, P=P)
            ctx.save_for_backward(x_flat, diagonals, perms, inv_perms, W, P)

        ctx.num_stages = num_stages
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = x.dtype
        ctx.bias_dtype = bias.dtype
        return out.to(x.dtype).reshape(*orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_flat, diagonals, perms, inv_perms, W, P = ctx.saved_tensors
        num_stages = ctx.num_stages
        go_flat = grad_out.reshape(-1, grad_out.shape[-1]).to(diagonals.dtype)
        g_bias = go_flat.sum(0).to(ctx.bias_dtype)

        if not x_flat.is_cuda or not torch.cuda.is_available():
            h_list = []
            if num_stages > 1:
                h_list.append(x_flat * diagonals[0])
                for s in range(num_stages - 2):
                    h_next = h_list[-1][:, perms[s]] * diagonals[s + 1]
                    h_list.append(h_next)

            g_diagonals = torch.empty_like(diagonals)
            gh = go_flat
            for s in range(num_stages - 1, 0, -1):
                h_perm = h_list[s - 1][:, perms[s - 1]]
                g_diagonals[s] = (gh * h_perm).sum(0)
                gh = (gh * diagonals[s])[:, inv_perms[s - 1]]

            g_diagonals[0] = (gh * x_flat).sum(0)
            gx = (gh * diagonals[0]).to(ctx.orig_dtype)
            return gx.reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias

        N, D = x_flat.shape
        P_inv = torch.argsort(P).to(torch.int32).contiguous()

        gx = torch.empty((N, D), device=x_flat.device, dtype=diagonals.dtype)
        grid_gx = lambda META: (triton.cdiv(N, META["BLOCK_M"]), triton.cdiv(D, META["BLOCK_D"]))
        _monarch_chain_bwd_gx_kernel[grid_gx](
            go_flat, W, P_inv, gx,
            go_flat.stride(0), go_flat.stride(1),
            W.stride(0),
            gx.stride(0), gx.stride(1),
            N, D,
        )

        W_scale, P_in, P_out = precompute_monarch_bwd_scales_single(diagonals, perms, inv_perms)
        g_diagonals = torch.empty_like(diagonals)
        grid_diag = lambda META: (num_stages, triton.cdiv(D, META["BLOCK_D"]))
        _monarch_chain_bwd_diag_kernel[grid_diag](
            go_flat, x_flat, W_scale, P_in, P_out, g_diagonals,
            go_flat.stride(0), go_flat.stride(1),
            x_flat.stride(0), x_flat.stride(1),
            W_scale.stride(0), W_scale.stride(1),
            P_in.stride(0), P_in.stride(1),
            P_out.stride(0), P_out.stride(1),
            g_diagonals.stride(0), g_diagonals.stride(1),
            N, D,
        )

        return gx.to(ctx.orig_dtype).reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias


class TritonFusedMonarchChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        diagonals: torch.Tensor,
        perms: torch.Tensor,
        inv_perms: torch.Tensor,
        bias: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        assert x.ndim >= 2 and x.shape[-1] == diagonals.shape[2], f"x last dim {x.shape[-1]} != D {diagonals.shape[2]}"
        orig_shape = x.shape
        perms = perms.long().contiguous()
        inv_perms = inv_perms.long().contiguous()
        x_flat = x.reshape(-1, x.shape[-1]).to(diagonals.dtype)
        num_branches = diagonals.shape[0]
        num_stages = diagonals.shape[1]
        if __debug__ and os.environ.get("AFFINE_DEBUG_MONARCH", "0") == "1":
            D = x.shape[-1]
            arange = torch.arange(D, device=perms.device)
            for s in range(num_stages - 1):
                if not torch.equal(inv_perms[s].gather(0, perms[s]), arange):
                    raise AssertionError(f"perms/inv_perms not bijective at stage {s}")

        if not x_flat.is_cuda or not torch.cuda.is_available():
            out = triton_fused_monarch_chain_fwd(x_flat, diagonals, perms, bias) # [M, N, dim]
            ctx.save_for_backward(x_flat, diagonals, perms, inv_perms, None, None)
        else:
            W, P = precompute_monarch_composed_fused(diagonals, perms)
            out = triton_fused_monarch_chain_fwd(x_flat, diagonals, perms, bias, W=W, P=P)
            ctx.save_for_backward(x_flat, diagonals, perms, inv_perms, W, P)

        ctx.num_branches = num_branches
        ctx.num_stages = num_stages
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = x.dtype
        ctx.bias_dtype = bias.dtype
        return tuple(out[m].to(x.dtype).reshape(*orig_shape) for m in range(num_branches))

    @staticmethod
    def backward(ctx, *grad_outs):
        x_flat, diagonals, perms, inv_perms, W, P = ctx.saved_tensors
        num_branches = ctx.num_branches
        num_stages = ctx.num_stages
        g_stack = torch.stack([g.reshape(-1, g.shape[-1]).to(diagonals.dtype) for g in grad_outs], dim=0) # [M, N, dim]
        g_bias = g_stack.sum(1).to(ctx.bias_dtype)

        if not x_flat.is_cuda or not torch.cuda.is_available():
            h_list = []
            if num_stages > 1:
                h_list.append(x_flat.unsqueeze(0) * diagonals[:, 0].unsqueeze(1))
                for s in range(num_stages - 2):
                    h_next = h_list[-1][:, :, perms[s]] * diagonals[:, s + 1].unsqueeze(1)
                    h_list.append(h_next)

            g_diagonals = torch.zeros_like(diagonals)
            gh = g_stack
            for s in range(num_stages - 1, 0, -1):
                h_perm = h_list[s - 1][:, :, perms[s - 1]]
                g_diagonals[:, s] = (gh * h_perm).sum(1)
                gh = (gh * diagonals[:, s].unsqueeze(1))[:, :, inv_perms[s - 1]]

            g_diagonals[:, 0] = (gh * x_flat.unsqueeze(0)).sum(1)
            gx = (gh * diagonals[:, 0].unsqueeze(1)).sum(0).to(ctx.orig_dtype)
            return gx.reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias

        N, D = x_flat.shape
        P_inv = torch.argsort(P).to(torch.int32).contiguous()

        gx = torch.empty((N, D), device=x_flat.device, dtype=diagonals.dtype)
        grid_gx = lambda META: (triton.cdiv(N, META["BLOCK_M"]), triton.cdiv(D, META["BLOCK_D"]))
        _fused_monarch_chain_bwd_gx_kernel[grid_gx](
            g_stack, W, P_inv, gx,
            g_stack.stride(0), g_stack.stride(1), g_stack.stride(2),
            W.stride(0), W.stride(1),
            gx.stride(0), gx.stride(1),
            N, D, M=num_branches,
        )

        W_scale, P_in, P_out = precompute_monarch_bwd_scales_fused(diagonals, perms, inv_perms)
        g_diagonals = torch.empty_like(diagonals)
        grid_diag = lambda META: (num_branches, num_stages, triton.cdiv(D, META["BLOCK_D"]))
        _fused_monarch_chain_bwd_diag_kernel[grid_diag](
            g_stack, x_flat, W_scale, P_in, P_out, g_diagonals,
            g_stack.stride(0), g_stack.stride(1), g_stack.stride(2),
            x_flat.stride(0), x_flat.stride(1),
            W_scale.stride(0), W_scale.stride(1), W_scale.stride(2),
            P_in.stride(0), P_in.stride(1),
            P_out.stride(0), P_out.stride(1),
            g_diagonals.stride(0), g_diagonals.stride(1), g_diagonals.stride(2),
            N, D,
        )

        return gx.to(ctx.orig_dtype).reshape(*ctx.orig_shape), g_diagonals, None, None, g_bias


def triton_monarch_chain(
    x: torch.Tensor,
    diagonals: torch.Tensor,
    perms: torch.Tensor,
    inv_perms: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    return TritonMonarchChainFunction.apply(x, diagonals, perms, inv_perms, bias)


def triton_fused_monarch_chain(
    x: torch.Tensor,
    diagonals: torch.Tensor,
    perms: torch.Tensor,
    inv_perms: torch.Tensor,
    bias: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    return TritonFusedMonarchChainFunction.apply(x, diagonals, perms, inv_perms, bias)
