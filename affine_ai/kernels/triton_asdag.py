"""
Custom Triton Kernel: 2D Grid-Tiled ASDAG Forward Dispatch
===========================================================
Parallelizes grid across BOTH (Batch Tiles, Leaves) so each GPU thread block
loads its leaf weight tile ONCE into SRAM and processes all tokens in parallel.
Optimized: transposed loops for X reuse, block_ptr coalescing, autotune,
early-exit for sparse secondary context.
"""

import math
import warnings
from typing import Optional, Tuple
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


# Autotune configs capped BLOCK_M/D<=64 to avoid register/SMEM blow-up (16x128x128 would spill).
# SMEM per block ~ BLOCK_M*BLOCK_D*4 bytes for acc + X/W tiles; 64x64=16KB within 99KB sm80, 64KB sm75.
_ASDAG_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 64}, num_warps=8, num_stages=3),
]

if _is_turing():
    _ASDAG_AUTOTUNE_CONFIGS = [
        c for c in _ASDAG_AUTOTUNE_CONFIGS
        if c.kwargs.get("BLOCK_M", 32) <= 64 and c.kwargs.get("BLOCK_D", 32) <= 64 and c.num_warps <= 4
    ]


@triton.autotune(configs=_ASDAG_AUTOTUNE_CONFIGS, key=["B_SZ", "DIM"])
@triton.jit
def _fused_asdag_2d_grid_kernel(
    X_ptr,
    W_stack_ptr,
    Bias_stack_ptr,
    Routing_ptr,
    Context_ptr,
    Peer_ptr,
    Norm_Factors_ptr,
    Leaf_Outs_ptr,
    stride_xb, stride_xd,
    stride_wk, stride_wd1, stride_wd2,
    stride_bk, stride_bd,
    stride_rb, stride_rk,
    stride_ck, stride_cm, stride_cd,
    stride_pok, stride_pos, stride_pob, stride_pod,
    stride_norm,
    stride_lob, stride_lok, stride_lod,
    B_SZ,
    DIM: tl.constexpr,
    MAX_SECONDARY: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_PEER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    2D grid-tiled ASDAG forward: Y[b,k,d]=norm[k]*(bias[k,d]+ sum_e x[b,e]*W[k,e,d] + peer_ctx).
    Grid (cdiv(B,BLOCK_M), K). BLOCK_M/D <=64 enforced; SMEM ~BLOCK_M*BLOCK_D*4*2 <99KB.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < B_SZ
    norm_factor = tl.load(Norm_Factors_ptr + pid_k * stride_norm, eviction_policy="evict_first").to(tl.float32)
    num_d_blocks = (DIM + BLOCK_D - 1) // BLOCK_D
    for d_out_idx in range(16):
        if d_out_idx < num_d_blocks:
            d_out_start = d_out_idx * BLOCK_D
            offs_d = d_out_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < DIM
            if d_out_start < DIM:
                acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
                for d_in_start in range(0, DIM, BLOCK_D):
                    x_block_ptr = tl.make_block_ptr(
                        base=X_ptr,
                        shape=(B_SZ, DIM),
                        strides=(stride_xb, stride_xd),
                        offsets=(pid_m * BLOCK_M, d_in_start),
                        block_shape=(BLOCK_M, BLOCK_D),
                        order=(1, 0),
                    )
                    x = tl.load(x_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last")
                    w_block_ptr = tl.make_block_ptr(
                        base=W_stack_ptr + pid_k * stride_wk,
                        shape=(DIM, DIM),
                        strides=(stride_wd2, stride_wd1),
                        offsets=(d_in_start, d_out_start),
                        block_shape=(BLOCK_D, BLOCK_D),
                        order=(1, 0),
                    )
                    w_k = tl.load(w_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last")
                    contrib = tl.dot(x, w_k, input_precision=INPUT_PRECISION)
                    acc = acc + contrib
                b_block_ptr = tl.make_block_ptr(
                    base=Bias_stack_ptr + pid_k * stride_bk,
                    shape=(DIM,),
                    strides=(stride_bd,),
                    offsets=(d_out_start,),
                    block_shape=(BLOCK_D,),
                    order=(0,),
                )
                b_k = tl.load(b_block_ptr, boundary_check=(0,), eviction_policy="evict_first")
                b_k = b_k.to(tl.float32)
                y_prim = acc + b_k[None, :]
                if HAS_PEER:
                    h_ctx = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
                    for s in range(MAX_SECONDARY):
                        c_s = tl.load(
                            Context_ptr + pid_k * stride_ck + s * stride_cm + offs_d * stride_cd,
                            mask=mask_d, other=0.0, eviction_policy="evict_first",
                        ).to(tl.float32)
                        if tl.sum(c_s) != 0:
                            p_s = tl.load(
                            Peer_ptr + pid_k * stride_pok + s * stride_pos + offs_m[:, None] * stride_pob + offs_d[None, :] * stride_pod,
                            mask=mask_m[:, None] & mask_d[None, :], other=0.0, eviction_policy="evict_last",
                        ).to(tl.float32)
                            h_ctx += c_s[None, :] * p_s
                    y_prim = y_prim + h_ctx
                y_v = y_prim * norm_factor
                if ACTIVATION == 1:
                    y_v = tl.minimum(tl.maximum(y_v, 0.0), 6.0)
                elif ACTIVATION == 2:
                    y_v = tl.where(y_v >= 0.0, 1.0, -1.0)
                    y_v = tl.where(mask_d[None, :], y_v, 0.0)
                out_block_ptr = tl.make_block_ptr(
                    base=Leaf_Outs_ptr + pid_k * stride_lok,
                    shape=(B_SZ, DIM),
                    strides=(stride_lob, stride_lod),
                    offsets=(pid_m * BLOCK_M, d_out_start),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )
                tl.store(out_block_ptr, y_v.to(Leaf_Outs_ptr.dtype.element_ty), boundary_check=(0, 1))


class FusedASDAG2DFunction(torch.autograd.Function):
    """
    2D grid-tiled ASDAG forward with recompute-based backward.

    Forward is Triton-tiled (BLOCK_M >=32, BLOCK_D up to 128 autotuned) with masked
    correctness for non-divisible B/D.  Backward does NOT use a Triton
    kernel; it recomputes the forward analytically via PyTorch and
    triggers autograd on the recomputed graph.  That recompute currently
    materializes y_prim [B, K, D] via ``einsum('be,kde->bkd')`` -- peak
    ~ B*K*D elements (e.g. B=8192,K=8,D=64 => ~4M floats).  Set
    ``use_checkpoint=True`` to wrap the recompute in
    ``torch.utils.checkpoint`` and/or chunk B to cut peak at the cost of
    extra forward compute.  A native Triton backward that avoids the
    [B,K,D] materialization is the intended follow-up.
    """
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_stack: torch.Tensor,
        bias_stack: torch.Tensor,
        routing_probs: torch.Tensor,
        context_gates: torch.Tensor,
        norm_factors: torch.Tensor,
        activation: str = "relu6",
        peer_outputs: Optional[torch.Tensor] = None,
        input_precision: Optional[str] = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w_stack, bias_stack, routing_probs, context_gates, norm_factors, peer_outputs)
        ctx.activation = activation
        ctx.use_checkpoint = bool(use_checkpoint)

        assert x.ndim == 2 and x.shape[1] == w_stack.shape[1], f"x {x.shape} vs w_stack {w_stack.shape}"
        assert w_stack.ndim == 3 and w_stack.shape[2] == x.shape[1], f"w_stack must be [K,D,D] with D={x.shape[1]}"
        assert bias_stack.shape == (w_stack.shape[0], x.shape[1])
        assert routing_probs.shape == (x.shape[0], w_stack.shape[0])
        if not w_stack.is_contiguous():
            w_stack = w_stack.contiguous()
        assert w_stack.is_contiguous(), "w_stack must be contiguous for block_ptr coalescing; transposed in Python above"
        B, D = x.shape
        K = w_stack.shape[0]
        M_max = context_gates.shape[1] if context_gates.ndim >= 3 else 1
        if K > 65535:
            raise ValueError(f"K={K} exceeds grid limit 65535")
        leaf_outs = torch.empty((B, K, D), device=x.device, dtype=x.dtype)
        act_code = 1 if activation == "relu6" else (2 if activation == "sign" else 0)
        if _is_turing() and x.dtype == torch.bfloat16:
            warnings.warn("Turing sm_75: bf16 not supported, treating as fp16 (acc fp32) in fused_asdag_2d_triton", stacklevel=3)
        grid = lambda META: (triton.cdiv(B, META["BLOCK_M"]), K)

        has_peer = peer_outputs is not None
        # dummy_peer stride zeroing: when not has_peer strides passed as 0, kernel gates loads via HAS_PEER
        assert not has_peer or peer_outputs.shape == (w_stack.shape[0], context_gates.shape[1] if context_gates.ndim >= 3 else 1, B, D) or peer_outputs.ndim in (2, 3, 4), "peer_outputs shape mismatch"
        dummy_peer = x if not has_peer else peer_outputs

        if input_precision is None:
            if x.dtype == torch.float32:
                prec = "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"
            else:
                prec = "ieee"
        else:
            prec = input_precision

        stride_ck = context_gates.stride(0)
        stride_cm = context_gates.stride(1) if context_gates.ndim >= 3 else 0
        stride_cd = context_gates.stride(2) if context_gates.ndim >= 3 else context_gates.stride(1)

        stride_norm = norm_factors.stride(0)

        _fused_asdag_2d_grid_kernel[grid](
            x, w_stack, bias_stack, routing_probs, context_gates, dummy_peer, norm_factors, leaf_outs,
            x.stride(0), x.stride(1),
            w_stack.stride(0), w_stack.stride(1), w_stack.stride(2),
            bias_stack.stride(0), bias_stack.stride(1),
            routing_probs.stride(0), routing_probs.stride(1),
            stride_ck, stride_cm, stride_cd,
            dummy_peer.stride(0) if has_peer else 0,
            dummy_peer.stride(1) if has_peer else 0,
            dummy_peer.stride(2) if has_peer else 0,
            dummy_peer.stride(3) if has_peer else 0,
            stride_norm,
            leaf_outs.stride(0), leaf_outs.stride(1), leaf_outs.stride(2),
            B,
            DIM=D,
            MAX_SECONDARY=M_max,
            ACTIVATION=act_code,
            HAS_PEER=has_peer,
            INPUT_PRECISION=prec,
        )

        return torch.einsum('bk, bkd -> bd', routing_probs, leaf_outs)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None, None, None, None, None

        needs_any_grad = any(
            ctx.needs_input_grad[i] for i in range(len(ctx.needs_input_grad))
        )
        if not needs_any_grad:
            return None, None, None, None, None, None, None, None, None, None

        x, w_stack, bias_stack, routing_probs, context_gates, norm_factors, peer_outputs = ctx.saved_tensors
        activation = ctx.activation
        use_checkpoint = getattr(ctx, "use_checkpoint", False)

        with torch.enable_grad():
            xr = x.detach().requires_grad_(ctx.needs_input_grad[0])
            wr = w_stack.detach().requires_grad_(ctx.needs_input_grad[1])
            br = bias_stack.detach().requires_grad_(ctx.needs_input_grad[2])
            rpr = routing_probs.detach().requires_grad_(ctx.needs_input_grad[3])
            cgr = context_gates.detach().requires_grad_(ctx.needs_input_grad[4])
            nfr = norm_factors.detach().requires_grad_(ctx.needs_input_grad[5])
            por = (
                peer_outputs.detach().requires_grad_(ctx.needs_input_grad[7])
                if peer_outputs is not None
                else None
            )

            def _forward_impl(xr_, wr_, br_, rpr_, cgr_, nfr_, por_):
                y_prim = torch.einsum('be, kde -> bkd', xr_, wr_) + br_.unsqueeze(0)
                if por_ is not None:
                    if cgr_.ndim == 2:
                        if por_.ndim == 3:
                            h_ctx = (cgr_.unsqueeze(1) * por_).permute(1, 0, 2)
                        else:
                            h_ctx = (cgr_.unsqueeze(1).unsqueeze(2) * por_).sum(dim=1).permute(1, 0, 2)
                    else:
                        h_ctx = (cgr_.unsqueeze(2) * por_).sum(dim=1).permute(1, 0, 2)
                    y_prim = y_prim + h_ctx
                y_v = y_prim * nfr_.view(1, -1, 1)
                if activation == "relu6":
                    y_act = torch.nn.functional.relu6(y_v)
                elif activation == "sign":
                    y_act = y_v + (torch.where(y_v >= 0.0, 1.0, -1.0) - y_v).detach()
                else:
                    y_act = y_v
                return torch.einsum('bk, bkd -> bd', rpr_, y_act)

            if use_checkpoint:
                import torch.utils.checkpoint as _ckpt
                # Checkpoint trades compute for memory: y_prim [B,K,D] intermediates
                # are not retained until inner backward recompute; chunking further
                # bounds peak but single checkpoint already cuts retained memory.
                if por is None:
                    def _forward_impl_nopor(xr_, wr_, br_, rpr_, cgr_, nfr_):
                        return _forward_impl(xr_, wr_, br_, rpr_, cgr_, nfr_, None)
                    out = _ckpt.checkpoint(
                        _forward_impl_nopor,
                        xr, wr, br, rpr, cgr, nfr, use_reentrant=False,
                    )
                else:
                    out = _ckpt.checkpoint(_forward_impl, xr, wr, br, rpr, cgr, nfr, por, use_reentrant=False)
                torch.autograd.backward(out, grad_output.to(out.dtype))
            else:
                out = _forward_impl(xr, wr, br, rpr, cgr, nfr, por)
                torch.autograd.backward(out, grad_output.to(out.dtype))

        return (
            xr.grad if ctx.needs_input_grad[0] else None,
            wr.grad if ctx.needs_input_grad[1] else None,
            br.grad if ctx.needs_input_grad[2] else None,
            rpr.grad if ctx.needs_input_grad[3] else None,
            cgr.grad if ctx.needs_input_grad[4] else None,
            nfr.grad if ctx.needs_input_grad[5] else None,
            None,
            por.grad if (por is not None and ctx.needs_input_grad[7]) else None,
            None,
            None,
        )


def fused_asdag_2d_triton(
    x: torch.Tensor,
    w_stack: torch.Tensor,
    bias_stack: torch.Tensor,
    routing_probs: torch.Tensor,
    context_gates: torch.Tensor,
    norm_factors: torch.Tensor,
    activation: str = "relu6",
    peer_outputs: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    2D Grid-Tiled Fused ASDAG Forward Pass in Triton wrapped with autograd.

    Forward uses BLOCK_M=64/BLOCK_D~64 tiling with masks for non-divisible
    B/D.  Backward is recompute-based (no Triton backward kernel) and
    materializes y_prim [B,K,D] peak B*K*D despite O(1) per-block claim;
    see FusedASDAG2DFunction docstring for peak estimate.
    Pass ``use_checkpoint=True`` to wrap the recompute in
    ``torch.utils.checkpoint`` to reduce peak memory at the cost of an
    extra forward.  Native Triton backward avoiding the [B,K,D] buffer is
    TODO.
    """
    input_precision = kwargs.get("input_precision", None)
    use_checkpoint = kwargs.get("use_checkpoint", False)
    return FusedASDAG2DFunction.apply(
        x,
        w_stack,
        bias_stack,
        routing_probs,
        context_gates,
        norm_factors,
        activation,
        peer_outputs,
        input_precision,
        use_checkpoint,
    )


fused_asdag_forward_triton = fused_asdag_2d_triton
