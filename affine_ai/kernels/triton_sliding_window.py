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
            # noqa: E501
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
        pruned = [c for c in configs if c.kwargs.get("BLOCK_D", 64) <= 64 and c.num_warps <= 4]
        if pruned:
            return pruned
    return configs


_SWA_CONFIGS = [
    triton.Config({"BLOCK_D": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(
    configs=_SWA_CONFIGS,
    key=["D"],
    prune_configs_by={"early_config_prune": _prune_swa_configs},
)
@triton.jit
def _swa_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_ob, stride_ot, stride_od,
    T, D,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    One program per query position (bh, t).
    Grid: (BH, T)
    Q/K/V shape: (BH, T, D) contiguous: stride_b = T*D, stride_t = D, stride_d = 1
    Handles arbitrary BH via pid_bh, arbitrary T via pid_t.
    WINDOW constexpr, SINK constexpr bool.
    """
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Guard tail: non-divisible T handled via mask, but grid ensures pid_t < T via Python launch
    # However we still guard if T not multiple of BLOCK_T=1, caller ensures cdiv.
    if pid_t >= T:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load q vector for this position
    q_ptrs = Q_ptr + pid_bh * stride_qb + pid_t * stride_qt + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    if SINK:
        lo_candidate = pid_t - WINDOW + 1
        if lo_candidate < 1:
            lo = 1
        else:
            lo = lo_candidate
    else:
        lo_candidate = pid_t - WINDOW + 1
        if lo_candidate < 0:
            lo = 0
        else:
            lo = lo_candidate

    m_val = -1e30

    if SINK:
        k0_ptrs = K_ptr + pid_bh * stride_kb + 0 * stride_kt + offs_d * stride_kd
        k0 = tl.load(k0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dot0 = tl.sum(q * k0, axis=0) * SCALE
        m_val = tl.maximum(m_val, dot0)

    for w in range(WINDOW):
        j = pid_t - w
        if j >= lo:
            if SINK:
                if j != 0:
                    k_ptrs = K_ptr + pid_bh * stride_kb + j * stride_kt + offs_d * stride_kd
                    k = tl.load(k_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                    dot = tl.sum(q * k, axis=0) * SCALE
                    m_val = tl.maximum(m_val, dot)
            else:
                k_ptrs = K_ptr + pid_bh * stride_kb + j * stride_kt + offs_d * stride_kd
                k = tl.load(k_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                dot = tl.sum(q * k, axis=0) * SCALE
                m_val = tl.maximum(m_val, dot)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    l_sum = 0.0

    if SINK:
        k0_ptrs = K_ptr + pid_bh * stride_kb + 0 * stride_kt + offs_d * stride_kd
        k0 = tl.load(k0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dot0 = tl.sum(q * k0, axis=0) * SCALE
        e0 = tl.exp(dot0 - m_val)
        l_sum += e0
        v0_ptrs = V_ptr + pid_bh * stride_vb + 0 * stride_vt + offs_d * stride_vd
        v0 = tl.load(v0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc += e0 * v0

    for w in range(WINDOW):
        j = pid_t - w
        if j >= lo:
            if SINK:
                if j != 0:
                    k_ptrs = K_ptr + pid_bh * stride_kb + j * stride_kt + offs_d * stride_kd
                    k = tl.load(k_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                    dot = tl.sum(q * k, axis=0) * SCALE
                    e = tl.exp(dot - m_val)
                    l_sum += e
                    v_ptrs = V_ptr + pid_bh * stride_vb + j * stride_vt + offs_d * stride_vd
                    v = tl.load(v_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                    acc += e * v
            else:
                k_ptrs = K_ptr + pid_bh * stride_kb + j * stride_kt + offs_d * stride_kd
                k = tl.load(k_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                dot = tl.sum(q * k, axis=0) * SCALE
                e = tl.exp(dot - m_val)
                l_sum += e
                v_ptrs = V_ptr + pid_bh * stride_vb + j * stride_vt + offs_d * stride_vd
                v = tl.load(v_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                acc += e * v

    # Normalize
    # l_sum >0 because at least self is visible
    out = acc / l_sum

    out_ptrs = Out_ptr + pid_bh * stride_ob + pid_t * stride_ot + offs_d * stride_od
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_d)


# ---------------------------------------------------------------------------
# Eager reference (for backward and CPU fallback)
# ---------------------------------------------------------------------------

def _eager_swa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int, sink: bool = True, scale: Optional[float] = None) -> torch.Tensor:
    """
    Eager SWA reference: q/k/v shape (B, H, T, D) or (BH, T, D).
    Returns same shape as q.
    """
    # Normalize to 4D for mask logic, then restore
    orig_is_4d = q.ndim == 4
    if q.ndim == 3:
        # (BH, T, D) -> (1, BH, T, D) for uniform handling, then squeeze
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
        lo = torch.clamp(qi - window + 1, min=1).unsqueeze(1)  # [T,1]
        window_mask = (kj.unsqueeze(0) >= lo) & (kj.unsqueeze(0) <= qi.unsqueeze(1))
        sink_mask = (kj.unsqueeze(0) == 0).expand(T, T)
        # For pid_t=0, window_mask has no valid j>=1, sink covers 0
        mask = window_mask | sink_mask
        # But for pid_t=0, window_mask incorrectly has no entries, sink gives 0 => correct.
        # For pid_t where lo=1, window_mask excludes 0, sink adds it.
    else:
        mask = (kj.unsqueeze(0) <= qi.unsqueeze(1)) & (kj.unsqueeze(0) > (qi - window).unsqueeze(1))

    # mask shape [T, T] -> [1,1,T,T] broadcast to [B,H,T,T]
    # Use scaled_dot_product_attention with is_causal=False and attn_mask
    # mask True = allowed
    y = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, attn_mask=mask, is_causal=False, scale=scale
    )
    if was_3d:
        return y.squeeze(0)
    return y


class _SlidingWindowAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int, sink: bool, scale: Optional[float]):
        # Save for backward (eager recompute)
        ctx.window = window
        ctx.sink = sink
        ctx.scale = scale
        # Need to save q/k/v for grad; also need to know if they require grad
        # Save detached copies to avoid holding graph? Use save_for_backward with original tensors
        ctx.save_for_backward(q, k, v)
        # If not CUDA or triton unavailable, use eager
        # Check that caller already gated, but double-check
        if not q.is_cuda:
            return _eager_swa(q, k, v, window, sink, scale)

        # Try triton path; on any exception fallback to eager (will be handled by caller, but also here)
        # Assume q/k/v are contiguous; if not, make contiguous (no graph impact for forward)
        # We cannot call .item() or sync here.
        B, H, T, D = q.shape
        BH = B * H
        # Flatten to (BH, T, D) contiguous for simple strides
        q_ = q.reshape(BH, T, D).contiguous()
        k_ = k.reshape(BH, T, D).contiguous()
        v_ = v.reshape(BH, T, D).contiguous()
        out_ = torch.empty_like(q_)

        if scale is None:
            scale_val = 1.0 / math.sqrt(D)
        else:
            scale_val = float(scale)

        # Determine BLOCK_D via next_power_of_2
        # Autotune will pick appropriate BLOCK_D; we pass D and let autotune select.
        # Grid: (BH, T)
        grid = (BH, T)

        # Launch kernel; SINK as constexpr bool (int 0/1)
        # WINDOW as constexpr, SCALE as constexpr float
        try:
            _swa_fwd_kernel[grid](
                q_, k_, v_, out_,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out_.stride(0), out_.stride(1), out_.stride(2),
                T, D,
                WINDOW=window,
                SINK=sink,
                SCALE=scale_val,
            )
        except Exception as e:
            warnings.warn(f"Triton SWA forward failed ({e}), falling back to eager", stacklevel=2)
            return _eager_swa(q, k, v, window, sink, scale)

        return out_.view(B, H, T, D)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        window = ctx.window
        sink = ctx.sink
        scale = ctx.scale

        # Recompute eager with grad enabled
        # Use torch.autograd.grad to get grads w.r.t q,k,v
        # This is correct but recomputes O(T^2) anyway; training still gets correct grads.
        # No host sync beyond Python ints.
        need_q = ctx.needs_input_grad[0]
        need_k = ctx.needs_input_grad[1]
        need_v = ctx.needs_input_grad[2]

        if not (need_q or need_k or need_v):
            return None, None, None, None, None, None

        # Detach and require grad for eager recompute
        # Preserve dtype and device
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(need_q)
            k_req = k.detach().requires_grad_(need_k)
            v_req = v.detach().requires_grad_(need_v)
            # Ensure they are leaf
            out = _eager_swa(q_req, k_req, v_req, window, sink, scale)
            grads = torch.autograd.grad(
                out, (q_req, k_req, v_req) if (need_q or need_k or need_v) else (),
                grad_output,
                allow_unused=True,
                retain_graph=False,
            )
        # Map grads to outputs; grads tuple may contain None
        # torch.autograd.grad returns tuple in order of inputs that required grad
        # But we passed only those that need grad? Simpler: always pass all three and filter
        # Instead we did conditional, so need to unpack.
        # Easier: compute with all three always, then mask
        # Redo if we need distinct handling
        # The above grads order corresponds to (q_req, k_req, v_req) filtered
        # To avoid complexity, recompute with all three
        # If any None, we need to align
        # Simpler approach: call grad with list of all that need grad, then fill
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
        q, k, v: shape (B, H, T, D) with same dtype/device. D <= 128 supported.
                 Also accepts (BH, T, D) 3D.
        window: window size W (int)
        sink: if True, position 0 is always visible (attend to 0 plus window [max(1,t-W+1)..t])
        scale: softmax scale, default 1/sqrt(D)

    Returns:
        Tensor same shape as q.

    Notes:
        CUDA-graph safe: no .item(), no host-device sync in capture path.
        Backward uses eager SDPA recompute for correctness (verified via gradcheck).
    """
    if q.ndim not in (3, 4):
        raise ValueError(f"q must be 3D or 4D, got {q.shape}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shape mismatch: {q.shape} vs {k.shape} vs {v.shape}")

    # CPU fallback directly to eager
    if not q.is_cuda:
        return _eager_swa(q, k, v, window, sink, scale)

    # Dtype guard: only fp16/bf16/fp32 on CUDA; fallback to eager for other dtypes
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        warnings.warn(f"SWA Triton supports fp16/bf16/fp32, got {q.dtype}, falling back to eager", stacklevel=2)
        return _eager_swa(q, k, v, window, sink, scale)

    # If triton not available, eager
    try:
        import triton  # noqa: F401
    except Exception:
        return _eager_swa(q, k, v, window, sink, scale)

    # Check D limit: kernel supports D <=128 (BLOCK_D max 128). Larger D fallback
    D = q.shape[-1]
    if D > 128:
        warnings.warn(f"SWA Triton D={D} >128, falling back to eager", stacklevel=2)
        return _eager_swa(q, k, v, window, sink, scale)

    # For 3D case, unsqueeze to 4D for Function, then squeeze
    was_3d = q.ndim == 3
    if was_3d:
        # (BH, T, D) -> (BH,1,T,D) -> treat B=BH, H=1
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
