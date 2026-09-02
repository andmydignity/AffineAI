import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any


class BitLinear(nn.Module):
    """
    1.58-Bit Ternary BitLinear Layer (BitNet b1.58 / Scalable MatMul-Free LM):
    Weights are quantized to {-1, 0, +1} with a dynamic scale factor gamma = mean(|W|).
    Activations are quantized to 8-bit integers [-128, 127] with dynamic per-token scale.
    Training uses the Straight-Through Estimator (STE).
    Hardware execution replaces floating-point multiplications with pure additions and sign flips.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=dtype) * (1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            if not self.training:
                from affine_ai.core.cpp_ops import asdag_cpu_bitlinear_ternary_int
                gamma = self.weight.abs().mean().clamp(min=1e-5)
                w_ternary = torch.round(self.weight / gamma).clamp(-1.0, 1.0)
                out_shape = x.shape[:-1] + (self.out_features,)
                x_flat = x.reshape(-1, x.shape[-1])
                out = asdag_cpu_bitlinear_ternary_int(x_flat, w_ternary, gamma.item(), self.bias)
                return out.to(x.dtype).reshape(out_shape)
            from affine_ai.core.cpp_ops import asdag_cpu_bitlinear
            return asdag_cpu_bitlinear(x, self.weight, self.bias)

        orig_dtype = x.dtype
        x_in = x.to(self.weight.dtype)

        gamma = self.weight.abs().mean().clamp(min=1e-5)
        w_scaled = self.weight / gamma
        w_ternary = torch.round(w_scaled).clamp(-1.0, 1.0)
        w_quant = self.weight + (w_ternary * gamma - self.weight).detach()

        scale_x = 127.0 / x_in.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_quant = (torch.round(x_in * scale_x).clamp(-128.0, 127.0) / scale_x).to(self.weight.dtype)
        x_ste = x_in + (x_quant - x_in).detach()

        out = F.linear(x_ste, w_quant, self.bias)
        return out.to(orig_dtype)


class TernaryBitLinearSwiGLU(nn.Module):
    """
    MatMul-Free Gated Channel Mixer (SwiGLU):
    Computes: y = BitLinear_down( SiLU(gate) * val ) where [gate, val] = BitLinear_gate_val(x).
    Eliminates 100% of floating-point multipliers in weights using ternary {-1, 0, +1} representations.
    """
    def __init__(
        self,
        dim: int,
        expand: int = 2,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = expand * dim
        self.w_gate_val = BitLinear(dim, 2 * self.hidden_dim, bias=False, dtype=dtype)
        self.w_down = BitLinear(self.hidden_dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            from affine_ai.core.cpp_ops import asdag_cpu_bitlinear_swiglu
            return asdag_cpu_bitlinear_swiglu(x, self.w_gate_val.weight, self.w_down.weight)

        from affine_ai.kernels.triton_bitlinear import triton_bitlinear_swiglu
        return triton_bitlinear_swiglu(x, self.w_gate_val.weight, self.w_down.weight)
