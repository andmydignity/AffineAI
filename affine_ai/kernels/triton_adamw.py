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
    bias_correction1: tl.float32,
    bias_correction2: tl.float32,
    N,                  # Total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # 1. Load parameter, gradient, and moment states into registers
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

    # 4. Compute bias-corrected estimates
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # 5. Compute parameter update
    denom = tl.sqrt(v_hat) + eps
    step = (m_hat / denom) * lr
    p = p - step

    # 6. Store updated states back to global VRAM
    tl.store(P_ptr + offs, p, mask=mask)
    tl.store(Exp_avg_ptr + offs, m, mask=mask)
    tl.store(Exp_avg_sq_ptr + offs, v, mask=mask)


class TritonAdamW(Optimizer):
    """
    High-Throughput Fused In-Place AdamW Optimizer in Triton.
    Drop-in replacement for torch.optim.AdamW on CUDA.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True
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
            correct_bias=correct_bias
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TritonAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32, device=p.device)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32, device=p.device)

                state['step'] += 1
                step_val = state['step']
                bias_correction1 = 1.0 - beta1 ** step_val
                bias_correction2 = 1.0 - beta2 ** step_val

                if p.is_cuda:
                    N = p.numel()
                    BLOCK_SIZE = 1024
                    grid = (triton.cdiv(N, BLOCK_SIZE),)
                    
                    orig_dtype = p.dtype
                    if orig_dtype != torch.float32:
                        p_f32 = p.float()
                        g_f32 = grad.float()
                        _adamw_kernel[grid](
                            p_f32,
                            g_f32,
                            state['exp_avg'],
                            state['exp_avg_sq'],
                            lr, beta1, beta2, eps, weight_decay,
                            bias_correction1, bias_correction2,
                            N,
                            BLOCK_SIZE=BLOCK_SIZE
                        )
                        p.copy_(p_f32.to(orig_dtype))
                    else:
                        _adamw_kernel[grid](
                            p,
                            grad,
                            state['exp_avg'],
                            state['exp_avg_sq'],
                            lr, beta1, beta2, eps, weight_decay,
                            bias_correction1, bias_correction2,
                            N,
                            BLOCK_SIZE=BLOCK_SIZE
                        )
                else:
                    # CPU Fallback
                    if weight_decay != 0.0:
                        p.data.mul_(1.0 - lr * weight_decay)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
