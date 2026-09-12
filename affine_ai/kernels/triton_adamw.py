"""
Custom Triton Kernel: Fused In-Place AdamW Optimizer
====================================================
Fuses 1st moment (m), 2nd moment (v), decoupled weight decay, and
parameter updates directly in GPU registers in a single memory pass.
Notes: amsgrad not supported (A-09), master_weights optional (A-10), weight decay order coupled before moment (A-07).
Eliminates intermediate tensor allocations and multiple CUDA kernel launches.
"""

import math
from typing import Tuple, Optional
import torch
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
    step_size,          # float32 scalar OR pointer to 1D float32 tensor
    inv_bc2_sqrt,       # float32 scalar OR pointer to 1D float32 tensor
    N,                  # Total number of elements
    Master_ptr = None,  # Optional master parameter pointer (FP32)
    HAS_MASTER: tl.constexpr = False,
    HAS_WD: tl.constexpr = False,
    BLOCK_SIZE: tl.constexpr = 1024,
    IS_ALIGNED_8: tl.constexpr = False,
    IS_CAPTURABLE: tl.constexpr = False
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    if IS_ALIGNED_8:
        offs = tl.multiple_of(offs, 8)
    offs = tl.max_contiguous(offs, BLOCK_SIZE)

    # 1. Load parameter, gradient, and moment states into registers
    if HAS_MASTER:
        p = tl.load(Master_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        p = tl.load(P_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    g = tl.load(Grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(Exp_avg_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(Exp_avg_sq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if IS_CAPTURABLE:
        step_sz = tl.load(step_size)
        inv_bc2 = tl.load(inv_bc2_sqrt)
    else:
        step_sz = step_size
        inv_bc2 = inv_bc2_sqrt

    # 2. Perform decoupled weight decay
    if HAS_WD:
        p = p - lr * weight_decay * p

    # 3. Update biased 1st and 2nd moments
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)

    # 4. Compute bias-corrected estimates and parameter update (inv_bc2_sqrt multiplication eliminates division)
    denom = (tl.sqrt(v) * inv_bc2) + eps
    step = step_sz * (m / denom)
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
    Supports capturable=True for full PyTorch CUDA Graph capture without CPU synchronization.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        master_weights: bool = True,
        capturable: bool = False
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
            master_weights=master_weights,
            capturable=capturable
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
            capturable = group.get('capturable', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TritonAdamW does not support sparse gradients")

                if not p.is_contiguous():
                    raise ValueError("AdamW param must be contiguous; call .contiguous() (A-01 strided view corruption)")
                if not grad.is_contiguous():
                    grad = grad.contiguous()

                p_data = p
                grad_data = grad

                state = self.state[p]
                if len(state) == 0:
                    moment_dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                    state['exp_avg'] = torch.zeros_like(p, dtype=moment_dtype, device=p.device)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=moment_dtype, device=p.device)
                    if capturable:
                        state['step'] = torch.zeros(1, dtype=torch.float32, device=p.device)
                    else:
                        state['step'] = 0
                    if use_master and p.dtype in (torch.float16, torch.bfloat16):
                        state['master_param'] = p_data.detach().to(torch.float32)
                        state['p_version'] = p._version

                is_tensor_step = isinstance(state['step'], torch.Tensor)
                if is_tensor_step:
                    # Capturable path: 100% on-device tensor operations, zero CPU sync / .item()
                    state['step'].add_(1)
                    bias_correction1 = (1.0 - beta1 ** state['step']) if correct_bias else torch.tensor(1.0, device=p.device)
                    bias_correction2 = (1.0 - beta2 ** state['step']) if correct_bias else torch.tensor(1.0, device=p.device)
                    step_size = (lr / bias_correction1).to(torch.float32)
                    inv_bc2_sqrt = (1.0 / torch.sqrt(bias_correction2)).to(torch.float32)
                else:
                    state['step'] += 1
                    step_val = state['step']
                    bias_correction1 = (1.0 - beta1 ** step_val) if correct_bias else 1.0
                    bias_correction2 = (1.0 - beta2 ** step_val) if correct_bias else 1.0
                    step_size = lr / bias_correction1
                    inv_bc2_sqrt = 1.0 / math.sqrt(bias_correction2)

                is_fp64_param = p.dtype == torch.float64

                if use_master and p.dtype in (torch.float16, torch.bfloat16):
                    if 'master_param' not in state:
                        state['master_param'] = p_data.detach().to(torch.float32)
                        state['p_version'] = p._version
                    elif state.get('p_version') is not None and p._version != state['p_version']:
                        state['master_param'].copy_(p_data)
                        state['p_version'] = p._version
                    has_master = True
                    master_p = state['master_param']
                else:
                    has_master = False
                    master_p = p_data

                if p.is_cuda and not is_fp64_param:
                    N = p_data.numel()
                    if N > 1 << 18:
                        BLOCK_SIZE = 1024
                    elif N > 1 << 14:
                        BLOCK_SIZE = 512
                    else:
                        BLOCK_SIZE = 256
                    has_wd = weight_decay != 0.0
                    # Strict 16-byte base pointer alignment check for safe 128-bit vectorization
                    is_aligned_8 = (
                        (N % 8 == 0) and
                        (p_data.data_ptr() % 16 == 0) and
                        (grad_data.data_ptr() % 16 == 0) and
                        (state['exp_avg'].data_ptr() % 16 == 0) and
                        (state['exp_avg_sq'].data_ptr() % 16 == 0) and
                        (not has_master or master_p.data_ptr() % 16 == 0)
                    )
                    grid = (triton.cdiv(N, BLOCK_SIZE),)
                    _adamw_kernel[grid](
                        p_data,
                        grad_data,
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        lr, beta1, beta2, eps, weight_decay,
                        step_size, inv_bc2_sqrt,
                        N,
                        master_p,
                        HAS_MASTER=has_master,
                        HAS_WD=has_wd,
                        BLOCK_SIZE=BLOCK_SIZE,
                        IS_ALIGNED_8=is_aligned_8,
                        IS_CAPTURABLE=is_tensor_step
                    )
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    grad_acc = grad_data.double() if is_fp64_param else grad_data.float()
                    step_sz_val = -step_size.item() if is_tensor_step else -step_size
                    inv_bc2_val = inv_bc2_sqrt if is_tensor_step else inv_bc2_sqrt
                    if has_master:
                        if weight_decay != 0.0:
                            master_p.mul_(1.0 - lr * weight_decay)
                        exp_avg.mul_(beta1).add_(grad_acc, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad_acc, grad_acc, value=1.0 - beta2)
                        denom = (exp_avg_sq.sqrt() * inv_bc2_val).add_(eps)
                        master_p.addcdiv_(exp_avg, denom, value=step_sz_val)
                        p_data.copy_(master_p)
                    else:
                        if weight_decay != 0.0:
                            p_data.mul_(1.0 - lr * weight_decay)
                        exp_avg.mul_(beta1).add_(grad_acc, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad_acc, grad_acc, value=1.0 - beta2)
                        denom = (exp_avg_sq.sqrt() * inv_bc2_val).add_(eps)
                        p_data.addcdiv_(exp_avg, denom, value=step_sz_val)

                if has_master:
                    state['p_version'] = p._version

        return loss

    def sync_master_weights(self):
        """Synchronize master FP32 weights from parameters."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, None)
                if state is not None and 'master_param' in state:
                    state['master_param'].copy_(p.detach().to(torch.float32))
                    state['p_version'] = p._version


def triton_adamw_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
    master_p: Optional[torch.Tensor] = None,
):
    """Functional interface for fused Triton AdamW step."""
    is_tensor_step = isinstance(step, torch.Tensor)
    if is_tensor_step:
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        step_size = (lr / bc1).to(torch.float32)
        inv_bc2_sqrt = (1.0 / torch.sqrt(bc2)).to(torch.float32)
    else:
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        step_size = lr / bc1
        inv_bc2_sqrt = 1.0 / math.sqrt(bc2)

    if not p.is_cuda or not torch.cuda.is_available():
        has_master = master_p is not None
        mp = master_p if has_master else p
        grad_acc = grad.double() if p.dtype == torch.float64 else grad.float()
        step_sz_val = -step_size.item() if is_tensor_step else -step_size
        inv_bc2_val = inv_bc2_sqrt if is_tensor_step else inv_bc2_sqrt
        if has_master:
            if weight_decay != 0.0:
                mp.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad_acc, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_acc, grad_acc, value=1.0 - beta2)
            denom = (exp_avg_sq.sqrt() * inv_bc2_val).add_(eps)
            mp.addcdiv_(exp_avg, denom, value=step_sz_val)
            p.copy_(mp)
        else:
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad_acc, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_acc, grad_acc, value=1.0 - beta2)
            denom = (exp_avg_sq.sqrt() * inv_bc2_val).add_(eps)
            p.addcdiv_(exp_avg, denom, value=step_sz_val)
        return

    is_fp64_param = p.dtype == torch.float64
    if is_fp64_param:
        grad_acc = grad.double()
        has_master = master_p is not None
        mp = master_p if has_master else p
        step_sz_val = -step_size.item() if is_tensor_step else -step_size
        inv_bc2_val = inv_bc2_sqrt if is_tensor_step else inv_bc2_sqrt
        if weight_decay != 0.0:
            mp.mul_(1.0 - lr * weight_decay)
        exp_avg.mul_(beta1).add_(grad_acc, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_acc, grad_acc, value=1.0 - beta2)
        denom = (exp_avg_sq.sqrt() * inv_bc2_val).add_(eps)
        mp.addcdiv_(exp_avg, denom, value=step_sz_val)
        if has_master:
            p.copy_(mp)
        return

    N = p.numel()
    if N > 1 << 18:
        BLOCK_SIZE = 1024
    elif N > 1 << 14:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    has_master = master_p is not None
    mp = master_p if has_master else p
    has_wd = weight_decay != 0.0
    is_aligned_8 = (
        (N % 8 == 0) and
        (p.data_ptr() % 16 == 0) and
        (grad.data_ptr() % 16 == 0) and
        (exp_avg.data_ptr() % 16 == 0) and
        (exp_avg_sq.data_ptr() % 16 == 0) and
        (not has_master or mp.data_ptr() % 16 == 0)
    )
    _adamw_kernel[grid](
        p, grad, exp_avg, exp_avg_sq,
        lr, beta1, beta2, eps, weight_decay,
        step_size, inv_bc2_sqrt, N, mp,
        HAS_MASTER=has_master, HAS_WD=has_wd, BLOCK_SIZE=BLOCK_SIZE,
        IS_ALIGNED_8=is_aligned_8,
        IS_CAPTURABLE=is_tensor_step
    )


__all__ = ["TritonAdamW", "triton_adamw_step", "_adamw_kernel"]
