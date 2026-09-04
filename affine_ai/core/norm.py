"""
Root Mean Square Normalization (RMSNorm)
========================================
Standalone normalization module with automatic hardware Triton kernel acceleration.
"""

import torch
import torch.nn as nn
import affine_ai.kernels as kernels


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and getattr(kernels, "TRITON_AVAILABLE", False) and x.dtype != torch.float64:
            return kernels.triton_rms_norm(x, self.scale, self.eps)
        
        # High numerical stability: retain variance and rsqrt in float64 for float64 inputs, and float32 otherwise
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        variance = x.to(calc_dtype).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.eps).to(x.dtype)
        return x * rsqrt * self.scale.to(x.dtype)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
    """
    Fused In-SRAM Residual Addition + RMSNorm:
    Computes res_out = x + residual, and y_norm = RMSNorm(res_out, scale) in a single pass.
    Returns (res_out, y_norm).
    """
    if x.is_cuda and getattr(kernels, "triton_fused_add_rms_norm", None) is not None and x.dtype != torch.float64:
        y_norm, res_out = kernels.triton_fused_add_rms_norm(x, residual, scale, eps)
        return res_out, y_norm

    res_out = x + residual
    calc_dtype = torch.float64 if res_out.dtype == torch.float64 else torch.float32
    variance = res_out.to(calc_dtype).pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + eps).to(res_out.dtype)
    y_norm = res_out * rsqrt * scale.to(res_out.dtype)
    return res_out, y_norm

