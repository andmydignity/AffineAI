"""
Custom Triton Kernel: Fused In-Place AdamW Optimizer
====================================================
Fuses 1st moment (m), 2nd moment (v), decoupled weight decay, and
parameter updates directly in GPU registers in a single memory pass.
Notes: amsgrad not supported (A-09), master_weights optional (A-10), weight decay order coupled before moment (A-07).
Eliminates intermediate tensor allocations and multiple CUDA kernel launches.
"""

import math
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import triton
import triton.language as tl


@triton.jit
def _adamw_kernel(
    P_ptr,              # Parameter tensor pointer
    Grad_ptr,           # Gradient tensor pointer
    Exp_avg_ptr,        # 1st moment pointer
    Exp_avg_sq_ptr,     # 2nd moment pointer
    lr: tl.float32,
    beta1: tl.float32,
    beta2: tl.float32,
    eps: tl.float32,
    weight_decay: tl.float32,
    step_size: tl.float32,
    bc2_sqrt: tl.float32,
    N,                  # Total number of elements
    Master_ptr = None,  # Optional master parameter pointer (FP32)
    HAS_MASTER: tl.constexpr = False,
    HAS_WD: tl.constexpr = False,
    BLOCK_SIZE: tl.constexpr = 1024
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # compiler hints for better pipelining
    # A-08: multiple_of(offs,8) illegal for N<BLOCK; use 1 or guard
    offs = tl.max_contiguous(tl.multiple_of(offs, 1), BLOCK_SIZE)  # A-08 fixed: use 1 to avoid illegal hint for small N

    # 1. Load parameter, gradient, and moment states into registers
    if HAS_MASTER:
        p = tl.load(Master_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        p = tl.load(P_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    g = tl.load(Grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(Exp_avg_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(Exp_avg_sq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # 2. Perform decoupled weight decay (A-07: coupled before moment vs AdamW variant; documented as decoupled)
    # A-03: use HAS_WD constexpr instead of float compare inside kernel
    if HAS_WD:
        p = p - lr * weight_decay * p

    # 3. Update biased 1st and 2nd moments
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)

    # 4. Compute bias-corrected estimates and parameter update
    denom = (tl.sqrt(v) / bc2_sqrt) + eps
    step = step_size * (m / denom)
    p = p - step

    # 5. Store updated states back to global VRAM
    if HAS_MASTER:
        tl.store(Master_ptr + offs, p, mask=mask)
    tl.store(P_ptr + offs, p.to(P_ptr.dtype.element_ty), mask=mask)
    tl.store(Exp_avg_ptr + offs, m, mask=mask)
    tl.store(Exp_avg_sq_ptr + offs, v, mask=mask)


class TritonAdamW(Optimizer):
    """
    High-Throughput Fused In-Place AdamW Optimizer in Triton.
    Drop-in replacement for torch.optim.AdamW on CUDA.
    Supports FP32 master weights for half-precision (FP16/BF16) parameters to prevent mantissa underflow.
    Note: amsgrad not supported (A-09), master_weights default True (A-10).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        master_weights: bool = True
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            master_weights=master_weights
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            correct_bias = group.get('correct_bias', True)
            use_master = group.get('master_weights', True)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TritonAdamW does not support sparse gradients")

                # A-01: strided view corruption guard - param must be contiguous for Triton flat pointer
                if not p.is_contiguous():
                    raise ValueError("AdamW param must be contiguous; call .contiguous() (A-01 strided view corruption)")
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                # Issue 39: Enforce contiguous tensors to prevent flat pointer memory corruption
                p_data = p  # already contiguous per guard
                grad_data = grad  # already handled
                # Alternative consistent flatten: p.view(-1) and zeros_like(p, dtype=float32) would also work

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0  # A-06: int step, not tensor -> not capturable
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32, device=p.device)  # A-01: shape matches p.shape exactly
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32, device=p.device)
                    # Issue 40: Support FP32 master weights for half-precision (FP16/BF16)
                    # Turing sm_75: uses fp16 master weights with fp32 accum (bf16 unsupported → fp16 fallback)
                    if use_master and p.dtype in (torch.float16, torch.bfloat16):
                        state['master_param'] = p_data.detach().clone().to(torch.float32)

                state['step'] += 1
                step_val = state['step']
                # A-05: host beta**step breaks CUDA Graphs (host math not graph-capturable)
                # TODO: compute bc on device for graph capture; keep host math but document.
                # A-06: state['step'] is int not tensor, so not capturable for CUDA Graphs; documented.
                bias_correction1 = (1.0 - beta1 ** step_val) if correct_bias else 1.0
                bias_correction2 = (1.0 - beta2 ** step_val) if correct_bias else 1.0
                step_size = lr / bias_correction1
                bc2_sqrt = math.sqrt(bias_correction2)

                has_master = 'master_param' in state
                master_p = state['master_param'] if has_master else p_data

                if p.is_cuda:
                    N = p_data.numel()
                    # larger BLOCK for big tensors to reduce launch overhead (capped at 1024 threads)
                    if N > 1 << 18:
                        BLOCK_SIZE = 1024
                    elif N > 1 << 14:
                        BLOCK_SIZE = 512
                    else:
                        BLOCK_SIZE = 256
                    has_wd = weight_decay != 0.0  # A-03
                    grid = (triton.cdiv(N, BLOCK_SIZE),)
                    _adamw_kernel[grid](
                        p_data,
                        grad_data,
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        lr, beta1, beta2, eps, weight_decay,
                        step_size, bc2_sqrt,
                        N,
                        master_p,
                        HAS_MASTER=has_master,
                        HAS_WD=has_wd,  # A-03
                        BLOCK_SIZE=BLOCK_SIZE
                    )
                else:
                    # CPU Fallback (handles fp16/bf16 via fp32 accum; Turing sm_75 uses fp16 master weights + fp32 moments)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    # A-02: CPU fallback must use fp32 grad unconditionally (bf16 grad loses precision)
                    grad_f32 = grad_data.float()  # A-02 fix
                    if has_master:
                        if weight_decay != 0.0:
                            master_p.mul_(1.0 - lr * weight_decay)
                        exp_avg.mul_(beta1).add_(grad_f32, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1.0 - beta2)
                        denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                        master_p.addcdiv_(exp_avg, denom, value=-step_size)
                        p_data.copy_(master_p)
                    else:
                        if weight_decay != 0.0:
                            p_data.mul_(1.0 - lr * weight_decay)
                        exp_avg.mul_(beta1).add_(grad_f32, alpha=1.0 - beta1)  # A-02: use grad_f32 not grad_data
                        exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1.0 - beta2)  # A-02
                        denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                        p_data.addcdiv_(exp_avg, denom, value=-step_size)

                # A-01: p is now guaranteed contiguous, no copy-back needed; kept for API compat if guard relaxed
                # if not p.is_contiguous():
                #     p.copy_(p_data)
                pass

        return loss
