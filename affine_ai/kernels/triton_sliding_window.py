"""
Triton Sliding-Window Causal Attention with Sink (SWA)
=======================================================
Per query t, attend to sink 0 plus window [max(1, t-W+1) .. t].
Scale = 1/sqrt(D). FP32 accumulation, fp16/bf16 I/O.
CUDA-graph safe: no .item(), no host sync, shapes as constexpr args.
"""

import math
import warnings
from typing import Optional

import torch
import triton
import triton.language as tl


def _is_turing() -> bool:
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


# ---------------------------------------------------------------------------
# Autotune configs: BLOCK_D covers D, num_warps tuned for occupancy.
# ---------------------------------------------------------------------------
def _prune_swa_configs(configs, named_args, **kwargs):
    D = kwargs.get("D", named_args.get("D", None))
    if D is not None:
        valid = [c for c in configs if c.kwargs.get("BLOCK_D", 0) >= D]
        if valid:
            min_block = min(c.kwargs["BLOCK_D"] for c in valid)
            configs = [c for c in configs if c.kwargs["BLOCK_D"] == min_block]
    if _is_turing():
        pruned = [
            c
            for c in configs
            if c.kwargs.get("BLOCK_D", 64) <= 64
            and c.kwargs.get("BLOCK_M", 64) <= 32
            and c.num_warps <= 4
        ]
        if pruned:
            return pruned
    return configs


_SWA_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_D": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 128}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=_SWA_CONFIGS,
    key=["D"],
    prune_configs_by={"early_config_prune": _prune_swa_configs},
)
@triton.jit
def _swa_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, LSE_ptr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_ob, stride_ot, stride_od,
    stride_lb, stride_lt,
    T, D,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)
    T_i64 = T.to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask_m = offs_m < T_i64
    mask_d = offs_d < D

    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    m_start = pid_m * BLOCK_M
    win_lo = m_start - WINDOW + 1
    zero_i64 = tl.zeros([], dtype=tl.int64)
    need_sep_sink = SINK and (win_lo > 1)

    if need_sep_sink:
        k0_ptrs = K_ptr + pid_bh * stride_kb + 0 * stride_kt + offs_d * stride_kd
        k0 = tl.load(k0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dot0 = tl.sum(q * k0[None, :], axis=1) * SCALE
        m_i = tl.where(mask_m, dot0, -1e30)
        l_i = tl.where(mask_m, 1.0, 0.0)
        v0_ptrs = V_ptr + pid_bh * stride_vb + 0 * stride_vt + offs_d * stride_vd
        v0 = tl.load(v0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc = tl.where(mask_m[:, None], v0[None, :], 0.0)
        k_start = (win_lo // BLOCK_N) * BLOCK_N
    else:
        k_start = zero_i64

    k_end = tl.minimum(T_i64, (pid_m + 1) * BLOCK_M)
    for n_start in range(k_start, k_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < T_i64

        k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k), allow_tf32=False) * SCALE

        if SINK:
            if need_sep_sink:
                attn_mask = (
                    mask_m[:, None]
                    & mask_n[None, :]
                    & (offs_n[None, :] <= offs_m[:, None])
                    & (offs_n[None, :] > offs_m[:, None] - WINDOW)
                    & (offs_n[None, :] >= 1)
                )
            else:
                attn_mask = (
                    mask_m[:, None]
                    & mask_n[None, :]
                    & (offs_n[None, :] <= offs_m[:, None])
                    & (
                        (offs_n[None, :] > offs_m[:, None] - WINDOW)
                        | (offs_n[None, :] == 0)
                    )
                )
        else:
            attn_mask = (
                mask_m[:, None]
                & mask_n[None, :]
                & (offs_n[None, :] <= offs_m[:, None])
                & (offs_n[None, :] > offs_m[:, None] - WINDOW)
            )

        s = tl.where(attn_mask, s, -1e30)
        chunk_max = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, chunk_max)

        alpha = tl.where(m_i > -1e20, tl.exp(m_i - m_new), 0.0)
        p = tl.where(attn_mask, tl.exp(s - m_new[:, None]), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_new
        l_i = l_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-12)
    out = acc * inv_l[:, None]

    out_ptrs = Out_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

    lse = tl.where(mask_m, m_i + tl.log(tl.maximum(l_i, 1e-12)), -1e30)
    lse_ptrs = LSE_ptr + pid_bh * stride_lb + offs_m * stride_lt
    tl.store(lse_ptrs, lse, mask=mask_m)


@triton.autotune(
    configs=_SWA_CONFIGS,
    key=["D"],
    prune_configs_by={"early_config_prune": _prune_swa_configs},
)
@triton.jit
def _swa_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, LSE_ptr, Delta_ptr, dQ_ptr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_dob, stride_dot, stride_dod,
    stride_lb, stride_lt,
    stride_delb, stride_delt,
    stride_dqb, stride_dqt, stride_dqd,
    T, D,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)
    T_i64 = T.to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask_m = offs_m < T_i64
    mask_d = offs_d < D

    q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    lse_ptrs = LSE_ptr + pid_bh * stride_lb + offs_m * stride_lt
    lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)

    delta_ptrs = Delta_ptr + pid_bh * stride_delb + offs_m * stride_delt
    delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    m_start = pid_m * BLOCK_M
    win_lo = m_start - WINDOW + 1
    zero_i64 = tl.zeros([], dtype=tl.int64)
    need_sep_sink = SINK and (win_lo > 1)

    if need_sep_sink:
        k0_ptrs = K_ptr + pid_bh * stride_kb + 0 * stride_kt + offs_d * stride_kd
        k0 = tl.load(k0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        v0_ptrs = V_ptr + pid_bh * stride_vb + 0 * stride_vt + offs_d * stride_vd
        v0 = tl.load(v0_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        s0 = tl.sum(q * k0[None, :], axis=1) * SCALE
        p0 = tl.where(mask_m, tl.exp(s0 - lse), 0.0)
        dp0 = tl.sum(do * v0[None, :], axis=1)
        ds0 = p0 * (dp0 - delta) * SCALE
        dq += ds0[:, None] * k0[None, :]
        k_start = (win_lo // BLOCK_N) * BLOCK_N
    else:
        k_start = zero_i64

    k_end = tl.minimum(T_i64, (pid_m + 1) * BLOCK_M)
    for n_start in range(k_start, k_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < T_i64

        k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k), allow_tf32=False) * SCALE

        if SINK:
            if need_sep_sink:
                attn_mask = (
                    mask_m[:, None]
                    & mask_n[None, :]
                    & (offs_n[None, :] <= offs_m[:, None])
                    & (offs_n[None, :] > offs_m[:, None] - WINDOW)
                    & (offs_n[None, :] >= 1)
                )
            else:
                attn_mask = (
                    mask_m[:, None]
                    & mask_n[None, :]
                    & (offs_n[None, :] <= offs_m[:, None])
                    & (
                        (offs_n[None, :] > offs_m[:, None] - WINDOW)
                        | (offs_n[None, :] == 0)
                    )
                )
        else:
            attn_mask = (
                mask_m[:, None]
                & mask_n[None, :]
                & (offs_n[None, :] <= offs_m[:, None])
                & (offs_n[None, :] > offs_m[:, None] - WINDOW)
            )

        p = tl.where(attn_mask, tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), allow_tf32=False)
        ds = p * (dp - delta[:, None]) * SCALE
        dq += tl.dot(ds, k, allow_tf32=False)

    dq_ptrs = dQ_ptr + pid_bh * stride_dqb + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


@triton.autotune(
    configs=_SWA_CONFIGS,
    key=["D"],
    prune_configs_by={"early_config_prune": _prune_swa_configs},
)
@triton.jit
def _swa_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, LSE_ptr, Delta_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_dob, stride_dot, stride_dod,
    stride_lb, stride_lt,
    stride_delb, stride_delt,
    stride_dkb, stride_dkt, stride_dkd,
    stride_dvb, stride_dvt, stride_dvd,
    T, D,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)
    T_i64 = T.to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask_n = offs_n < T_i64
    mask_d = offs_d < D

    k_ptrs = K_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptrs = V_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    n_start = pid_n * BLOCK_N
    zero_i64 = tl.zeros([], dtype=tl.int64)
    if pid_n == 0 and SINK:
        q_start = zero_i64
        q_end = T_i64
    else:
        q_start = (n_start // BLOCK_M) * BLOCK_M
        n_max = (pid_n + 1) * BLOCK_N - 1
        q_end = tl.minimum(T_i64, ((n_max + WINDOW + BLOCK_M - 1) // BLOCK_M) * BLOCK_M)

    for m_start in range(q_start, q_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = offs_m < T_i64

        q_ptrs = Q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        do_ptrs = dO_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        lse_ptrs = LSE_ptr + pid_bh * stride_lb + offs_m * stride_lt
        lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)

        delta_ptrs = Delta_ptr + pid_bh * stride_delb + offs_m * stride_delt
        delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)

        s = tl.dot(k, tl.trans(q), allow_tf32=False) * SCALE

        if SINK:
            attn_mask = (
                mask_n[:, None]
                & mask_m[None, :]
                & (offs_n[:, None] <= offs_m[None, :])
                & (
                    (offs_n[:, None] > offs_m[None, :] - WINDOW)
                    | (offs_n[:, None] == 0)
                )
            )
        else:
            attn_mask = (
                mask_n[:, None]
                & mask_m[None, :]
                & (offs_n[:, None] <= offs_m[None, :])
                & (offs_n[:, None] > offs_m[None, :] - WINDOW)
            )

        p = tl.where(attn_mask, tl.exp(s - lse[None, :]), 0.0)
        dp = tl.dot(v, tl.trans(do), allow_tf32=False)
        ds = p * (dp - delta[None, :]) * SCALE

        dk += tl.dot(ds, q, allow_tf32=False)
        dv += tl.dot(p, do, allow_tf32=False)

    dk_ptrs = dK_ptr + pid_bh * stride_dkb + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dk.to(dK_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])

    dv_ptrs = dV_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dv.to(dV_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


# ---------------------------------------------------------------------------
# Eager reference (for CPU fallback)
# ---------------------------------------------------------------------------

def _eager_swa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    sink: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    orig_is_4d = q.ndim == 4
    if q.ndim == 3:
        q4 = q.unsqueeze(0)
        k4 = k.unsqueeze(0)
        v4 = v.unsqueeze(0)
        was_3d = True
    elif q.ndim == 4:
        q4 = q
        k4 = k
        v4 = v
        was_3d = False
    else:
        raise ValueError(f"q must be 3D or 4D, got {q.shape}")

    B, H, T, D = q4.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    device = q4.device

    qi = torch.arange(T, device=device)
    kj = torch.arange(T, device=device)
    if sink:
        lo = torch.clamp(qi - window + 1, min=1).unsqueeze(1)
        window_mask = (kj.unsqueeze(0) >= lo) & (kj.unsqueeze(0) <= qi.unsqueeze(1))
        sink_mask = (kj.unsqueeze(0) == 0).expand(T, T)
        mask = window_mask | sink_mask
    else:
        mask = (kj.unsqueeze(0) <= qi.unsqueeze(1)) & (kj.unsqueeze(0) > (qi - window).unsqueeze(1))

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            y = torch.nn.functional.scaled_dot_product_attention(
                q4, k4, v4, attn_mask=mask, is_causal=False, scale=scale
            )
    except Exception:
        y = torch.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask, is_causal=False, scale=scale
        )

    if was_3d:
        return y.squeeze(0)
    return y


class _SlidingWindowAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window: int,
        sink: bool,
        scale: Optional[float],
    ):
        ctx.window = window
        ctx.sink = sink
        ctx.scale = scale

        if not q.is_cuda:
            ctx.save_for_backward(q, k, v)
            ctx.is_cuda = False
            return _eager_swa(q, k, v, window, sink, scale)

        B, H, T, D = q.shape
        BH = B * H
        q_ = q.reshape(BH, T, D).contiguous()
        k_ = k.reshape(BH, T, D).contiguous()
        v_ = v.reshape(BH, T, D).contiguous()
        out_ = torch.empty_like(q_)
        lse_ = torch.empty((BH, T), device=q.device, dtype=torch.float32)

        if scale is None:
            scale_val = 1.0 / math.sqrt(D)
        else:
            scale_val = float(scale)

        def grid(META):
            return (triton.cdiv(T, META["BLOCK_M"]), BH)

        try:
            _swa_fwd_kernel[grid](
                q_, k_, v_, out_, lse_,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out_.stride(0), out_.stride(1), out_.stride(2),
                lse_.stride(0), lse_.stride(1),
                T, D,
                WINDOW=window,
                SINK=sink,
                SCALE=scale_val,
            )
        except Exception as e:
            warnings.warn(f"Triton SWA forward failed ({e}), falling back to eager", stacklevel=2)
            ctx.save_for_backward(q, k, v)
            ctx.is_cuda = False
            return _eager_swa(q, k, v, window, sink, scale)

        ctx.save_for_backward(q_, k_, v_, out_, lse_)
        ctx.is_cuda = True
        ctx.shape = (B, H, T, D)
        return out_.view(B, H, T, D)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        window = ctx.window
        sink = ctx.sink
        scale = ctx.scale

        need_q = ctx.needs_input_grad[0]
        need_k = ctx.needs_input_grad[1]
        need_v = ctx.needs_input_grad[2]

        if not (need_q or need_k or need_v):
            return None, None, None, None, None, None

        if not getattr(ctx, "is_cuda", False):
            q, k, v = ctx.saved_tensors[:3]
            with torch.enable_grad():
                q_req = q.detach().requires_grad_(need_q)
                k_req = k.detach().requires_grad_(need_k)
                v_req = v.detach().requires_grad_(need_v)
                out = _eager_swa(q_req, k_req, v_req, window, sink, scale)
                grads = torch.autograd.grad(
                    out,
                    [t for t, need in [(q_req, need_q), (k_req, need_k), (v_req, need_v)] if need],
                    grad_output,
                    allow_unused=True,
                )
            grad_q = grad_k = grad_v = None
            idx = 0
            if need_q:
                grad_q = grads[idx]
                idx += 1
            if need_k:
                grad_k = grads[idx]
                idx += 1
            if need_v:
                grad_v = grads[idx]
                idx += 1
            return grad_q, grad_k, grad_v, None, None, None

        q_, k_, v_, out_, lse_ = ctx.saved_tensors
        BH, T, D = q_.shape
        B, H = ctx.shape[0], ctx.shape[1]

        if scale is None:
            scale_val = 1.0 / math.sqrt(D)
        else:
            scale_val = float(scale)

        do_ = grad_output.reshape(BH, T, D).contiguous()
        delta_ = (do_.float() * out_.float()).sum(dim=-1).contiguous()

        grad_q = grad_k = grad_v = None

        if need_q:
            dq_ = torch.empty_like(q_)
            def grid_dq(META):
                return (triton.cdiv(T, META["BLOCK_M"]), BH)

            _swa_bwd_dq_kernel[grid_dq](
                q_, k_, v_, do_, lse_, delta_, dq_,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                do_.stride(0), do_.stride(1), do_.stride(2),
                lse_.stride(0), lse_.stride(1),
                delta_.stride(0), delta_.stride(1),
                dq_.stride(0), dq_.stride(1), dq_.stride(2),
                T, D,
                WINDOW=window,
                SINK=sink,
                SCALE=scale_val,
            )
            grad_q = dq_.view(B, H, T, D)

        if need_k or need_v:
            dk_ = torch.empty_like(k_)
            dv_ = torch.empty_like(v_)
            def grid_dkv(META):
                return (triton.cdiv(T, META["BLOCK_N"]), BH)

            _swa_bwd_dkv_kernel[grid_dkv](
                q_, k_, v_, do_, lse_, delta_, dk_, dv_,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                do_.stride(0), do_.stride(1), do_.stride(2),
                lse_.stride(0), lse_.stride(1),
                delta_.stride(0), delta_.stride(1),
                dk_.stride(0), dk_.stride(1), dk_.stride(2),
                dv_.stride(0), dv_.stride(1), dv_.stride(2),
                T, D,
                WINDOW=window,
                SINK=sink,
                SCALE=scale_val,
            )
            if need_k:
                grad_k = dk_.view(B, H, T, D)
            if need_v:
                grad_v = dv_.view(B, H, T, D)

        return grad_q, grad_k, grad_v, None, None, None


def sliding_window_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    sink: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Sliding-window causal attention with optional sink token.

    Args:
        q, k, v: shape (B, H, T, D) with same dtype/device.
                 Also accepts (BH, T, D) 3D.
        window: window size W (int)
        sink: if True, position 0 is always visible (attend to 0 plus window [max(1,t-W+1)..t])
        scale: softmax scale, default 1/sqrt(D)

    Returns:
        Tensor same shape as q.
    """
    if q.ndim not in (3, 4):
        raise ValueError(f"q must be 3D or 4D, got {q.shape}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shape mismatch: {q.shape} vs {k.shape} vs {v.shape}")

    if not q.is_cuda:
        return _eager_swa(q, k, v, window, sink, scale)

    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        warnings.warn(
            f"SWA Triton supports fp16/bf16/fp32, got {q.dtype}, falling back to eager",
            stacklevel=2,
        )
        return _eager_swa(q, k, v, window, sink, scale)

    try:
        import triton  # noqa: F401
    except Exception:
        return _eager_swa(q, k, v, window, sink, scale)

    was_3d = q.ndim == 3
    if was_3d:
        q4 = q.unsqueeze(1)
        k4 = k.unsqueeze(1)
        v4 = v.unsqueeze(1)
    else:
        q4 = q
        k4 = k
        v4 = v

    out4 = _SlidingWindowAttnFunc.apply(q4, k4, v4, int(window), bool(sink), scale)
    if was_3d:
        return out4.squeeze(1)
    return out4


__all__ = ["sliding_window_attn", "_eager_swa"]
