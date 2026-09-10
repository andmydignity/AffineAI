"""
Custom Triton Kernel: Fused In-Place AdamW Optimizer
====================================================
Fuses 1st moment (m), 2nd moment (v), decoupled weight decay, and
parameter updates directly in GPU registers in a single memory pass.
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
    BLOCK_SIZE: tl.constexpr = 1024
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # compiler hints for better pipelining
    offs = tl.max_contiguous(tl.multiple_of(offs, 8), BLOCK_SIZE)

    # 1. Load parameter, gradient, and moment states into registers
    if HAS_MASTER:
        p = tl.load(Master_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        p = tl.load(P_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    g = tl.load(Grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(Exp_avg_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(Exp_avg_sq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # 2. Perform decoupled weight decay
    if weight_decay != 0.0:
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

                # Issue 39: Enforce contiguous tensors to prevent flat pointer memory corruption
                p_data = p if p.is_contiguous() else p.contiguous()
                grad_data = grad if grad.is_contiguous() else grad.contiguous()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data, dtype=torch.float32, device=p.device)
                    state['exp_avg_sq'] = torch.zeros_like(p_data, dtype=torch.float32, device=p.device)
                    # Issue 40: Support FP32 master weights for half-precision (FP16/BF16)
                    if use_master and p.dtype in (torch.float16, torch.bfloat16):
                        state['master_param'] = p_data.detach().clone().to(torch.float32)

                state['step'] += 1
                step_val = state['step']
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
                        BLOCK_SIZE=BLOCK_SIZE
                    )
                else:
                    # CPU Fallback
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    if has_master:
                        if weight_decay != 0.0:
                            master_p.mul_(1.0 - lr * weight_decay)
                        grad_f32 = grad_data.float()
                        exp_avg.mul_(beta1).add_(grad_f32, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1.0 - beta2)
                        denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                        master_p.addcdiv_(exp_avg, denom, value=-step_size)
                        p_data.copy_(master_p)
                    else:
                        if weight_decay != 0.0:
                            p_data.mul_(1.0 - lr * weight_decay)
                        exp_avg.mul_(beta1).add_(grad_data, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad_data, grad_data, value=1.0 - beta2)
                        denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                        p_data.addcdiv_(exp_avg, denom, value=-step_size)

                # If p was non-contiguous, copy back the updated contiguous buffer
                if not p.is_contiguous():
                    p.copy_(p_data)

        return loss
