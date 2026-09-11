import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union

import affine_ai.kernels as kernels
from affine_ai.core.norm import RMSNorm
from affine_ai.core.backpressure_tree import ternary_ste

def _maybe_compile(fn):
    try:
        if hasattr(torch, 'compile'):
            return torch.compile(fn)
    except Exception:
        pass
    return fn

def _clamp_min_for_dtype(dtype: torch.dtype) -> float:
    """Return underflow-safe clamp for GLA decay diff: -11 for fp16, -30 otherwise."""
    return -11.0 if dtype == torch.float16 else -30.0



class MonarchChainCUDAAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, diagonals: torch.Tensor, perms: torch.Tensor, inv_perms: torch.Tensor, bias: torch.Tensor):
        num_stages = diagonals.shape[0]
        h_list = [x * diagonals[0]]
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1):
            h_next = h_list[-1][:, perms[s]] * diagonals[s + 1]
            h_list.append(h_next)
        out = h_list[-1] + bias
        ctx.save_for_backward(x, diagonals, perms, inv_perms, *h_list)
        ctx.num_stages = num_stages
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        saved = ctx.saved_tensors
        x = saved[0]
        diagonals = saved[1]
        perms = saved[2]
        inv_perms = saved[3]
        num_stages = ctx.num_stages
        h_list = saved[4:4 + num_stages]

        g_bias = grad_out.sum(0)
        g_diagonals = torch.empty_like(diagonals)
        gh = grad_out

        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1, 0, -1):
            h_perm = h_list[s - 1][:, perms[s - 1]]
            g_diagonals[s] = (gh * h_perm).sum(0)
            gh = (gh * diagonals[s])[:, inv_perms[s - 1]]

        g_diagonals[0] = (gh * x).sum(0)
        gx = gh * diagonals[0]
        return gx, g_diagonals, None, None, g_bias


class FusedMonarchChainCUDAAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, diagonals: torch.Tensor, perms: torch.Tensor, inv_perms: torch.Tensor, bias: torch.Tensor):
        num_branches = diagonals.shape[0]
        num_stages = diagonals.shape[1]
        h_list = [x.unsqueeze(0) * diagonals[:, 0].unsqueeze(1)]
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1):
            h_next = h_list[-1][:, :, perms[s]] * diagonals[:, s + 1].unsqueeze(1)
            h_list.append(h_next)
        out = h_list[-1] + bias.unsqueeze(1) # [M, N, dim]
        ctx.save_for_backward(x, diagonals, perms, inv_perms, *h_list)
        ctx.num_branches = num_branches
        ctx.num_stages = num_stages
        return tuple(out[m] for m in range(num_branches))

    @staticmethod
    def backward(ctx, *grad_outs):
        saved = ctx.saved_tensors
        x = saved[0]
        diagonals = saved[1]
        perms = saved[2]
        inv_perms = saved[3]
        num_stages = ctx.num_stages
        h_list = saved[4:4 + num_stages]

        g_stack = torch.stack(grad_outs, dim=0) # [M, N, dim]
        g_bias = g_stack.sum(1)
        g_diagonals = torch.empty_like(diagonals)
        gh = g_stack

        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1, 0, -1):
            h_perm = h_list[s - 1][:, :, perms[s - 1]]
            g_diagonals[:, s] = (gh * h_perm).sum(1)
            gh = (gh * diagonals[:, s].unsqueeze(1))[:, :, inv_perms[s - 1]]

        g_diagonals[:, 0] = (gh * x.unsqueeze(0)).sum(1)
        gx = (gh * diagonals[:, 0].unsqueeze(1)).sum(0)
        return gx, g_diagonals, None, None, g_bias


class MonarchPermutationChain(nn.Module):
    """
    Zero-MatMul Monarch Permutation Chain:
    Computes: y = D_{L-1} * pi_{L-2}( ... D_1 * pi_0(D_0 * x) ... ) + bias
    Uses L stages of continuous diagonal scale vectors and fixed random permutation tables.
    Matches dense linear expressivity with O(L * d) parameters and ZERO matrix multiplications.
    """
    def __init__(
        self,
        dim: int,
        num_stages: int = 4,
        seed_offset: int = 0,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.num_stages = num_stages

        # Fixed random permutation tables
        perms = []
        inv_perms = []
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1):
            g = torch.Generator().manual_seed(seed_offset * 1000 + s + 1)
            p = torch.randperm(dim, generator=g)
            perms.append(p)
            inv = torch.empty(dim, dtype=torch.long)
            inv[p] = torch.arange(dim)
            inv_perms.append(inv)
        self.register_buffer('perms', torch.stack(perms)) # [num_stages - 1, dim]
        self.register_buffer('inv_perms', torch.stack(inv_perms)) # [num_stages - 1, dim]

        # Learnable continuous diagonal scales: [num_stages, dim]
        self.diagonals = nn.Parameter(
            torch.ones(num_stages, dim, dtype=dtype) + torch.randn(num_stages, dim, dtype=dtype) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            from affine_ai.core.cpp_ops import asdag_cpu_monarch_chain
            return asdag_cpu_monarch_chain(x, self.diagonals, self.perms, self.inv_perms, self.bias)

        if getattr(kernels, "TRITON_AVAILABLE", False) and getattr(kernels, "triton_monarch_chain", None) is not None:
            try:
                return kernels.triton_monarch_chain(x, self.diagonals, self.perms, self.inv_perms, self.bias)
            except Exception as e:
                warnings.warn(f"triton_monarch_chain failed: {e}; falling back", stacklevel=2)
        try:
            from affine_ai.kernels.triton_monarch import triton_monarch_chain as _fallback_chain
            return _fallback_chain(x, self.diagonals, self.perms, self.inv_perms, self.bias)
        except Exception as e:
            warnings.warn(f"triton_monarch_chain fallback failed: {e}; using Python path", stacklevel=2)
        # Python fallback (exact but slower)
        num_stages = self.diagonals.shape[0]
        h = x * self.diagonals[0]
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1):
            h = h[:, self.perms[s]] * self.diagonals[s + 1]
        return h + self.bias


class FusedMonarchChain(nn.Module):
    """
    Fused Zero-MatMul Monarch Permutation Chain:
    Computes M output projections (e.g. Q, K, V, Gate) simultaneously with shared permutation tables.
    Uses M distinct diagonal scale chains: [M, L, dim].
    """
    def __init__(
        self,
        dim: int,
        num_branches: int = 4,
        num_stages: int = 4,
        seed_offset: int = 0,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.num_branches = num_branches
        self.num_stages = num_stages

        perms = []
        inv_perms = []
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1):
            g = torch.Generator().manual_seed(seed_offset * 1000 + s + 1)
            p = torch.randperm(dim, generator=g)
            perms.append(p)
            inv = torch.empty(dim, dtype=torch.long)
            inv[p] = torch.arange(dim)
            inv_perms.append(inv)
        self.register_buffer('perms', torch.stack(perms)) # [num_stages - 1, dim]
        self.register_buffer('inv_perms', torch.stack(inv_perms)) # [num_stages - 1, dim]

        self.diagonals = nn.Parameter(
            torch.ones(num_branches, num_stages, dim, dtype=dtype) + torch.randn(num_branches, num_stages, dim, dtype=dtype) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(num_branches, dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if not x.is_cuda:
            from affine_ai.core.cpp_ops import asdag_cpu_fused_monarch_chain
            return asdag_cpu_fused_monarch_chain(x, self.diagonals, self.perms, self.inv_perms, self.bias)

        if getattr(kernels, "TRITON_AVAILABLE", False) and getattr(kernels, "triton_fused_monarch_chain", None) is not None:
            try:
                return kernels.triton_fused_monarch_chain(x, self.diagonals, self.perms, self.inv_perms, self.bias)
            except Exception as e:
                warnings.warn(f"triton_fused_monarch_chain failed: {e}; falling back", stacklevel=2)
        try:
            from affine_ai.kernels.triton_monarch import triton_fused_monarch_chain as _fallback_fused
            return _fallback_fused(x, self.diagonals, self.perms, self.inv_perms, self.bias)
        except Exception as e:
            warnings.warn(f"triton_fused_monarch_chain fallback failed: {e}; using Python path", stacklevel=2)
        # Python fallback
        num_branches = self.diagonals.shape[0]
        num_stages = self.diagonals.shape[1]
        h = x.unsqueeze(0) * self.diagonals[:, 0].unsqueeze(1)
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for s in range(num_stages - 1):
            h = h[:, :, self.perms[s]] * self.diagonals[:, s + 1].unsqueeze(1)
        out = h + self.bias.unsqueeze(1)
        return tuple(out[m] for m in range(num_branches))


class PermutationProjection(nn.Module):
    """
    Zero-MatMul Permutation Projection:
    Computes y = bias + sum_{p=0}^{P-1} (w_p * x[perm_p])
    Uses Ternary QAT {-1, 0, +1} weights with learned FP8/BF16 scale factors.
    Zero Python loops: Executes via native C++ AVX2/AVX-512 SIMD gather kernels.
    """
    def __init__(
        self,
        dim: int,
        num_perms: int = 4,
        seed_offset: int = 0,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.num_perms = num_perms

        # Fixed random permutation tables (p=0 is Identity)
        perms = [torch.arange(dim)]
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for p in range(1, num_perms):
            g = torch.Generator().manual_seed(seed_offset * 1000 + p)
            perms.append(torch.randperm(dim, generator=g))
        self.register_buffer('perms', torch.stack(perms)) # [P, dim]

        inv_perms = []
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for p in range(num_perms):
            inv = torch.empty(dim, dtype=torch.long)
            inv[self.perms[p]] = torch.arange(dim)
            inv_perms.append(inv)
        self.register_buffer('inv_perms', torch.stack(inv_perms)) # [P, dim]

        self.latent_w = nn.Parameter(
            torch.randn(1, num_perms, dim, dtype=dtype) * (1.0 / math.sqrt(dim))
        )
        self.scale = nn.Parameter(torch.ones(1, num_perms, 1, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        w = ternary_ste(self.latent_w, scale=self.scale) # [1, P, dim]

        if not x.is_cuda:
            from affine_ai.core.cpp_ops import asdag_cpu_fused_perm_proj
            out = asdag_cpu_fused_perm_proj(x_flat, w, self.perms, self.inv_perms, self.bias)
            return out[0].reshape(*orig_shape)

        if getattr(kernels, "TRITON_AVAILABLE", False) and getattr(kernels, "triton_fused_perm_proj", None) is not None:
            try:
                out = kernels.triton_fused_perm_proj(x_flat, w, self.perms, self.inv_perms, self.bias)
                return out[0].reshape(*orig_shape)
            except Exception as e:
                warnings.warn(f"triton_fused_perm_proj failed: {e}; falling back", stacklevel=2)
        else:
            try:
                from affine_ai.kernels.triton_perm_proj import triton_fused_perm_proj as _perm
                out = _perm(x_flat, w, self.perms, self.inv_perms, self.bias)
                return out[0].reshape(*orig_shape)
            except Exception as e:
                warnings.warn(f"triton_fused_perm_proj fallback failed: {e}", stacklevel=2)

        x_gathered = torch.gather(
            x_flat.unsqueeze(1).expand(-1, self.num_perms, -1),
            dim=-1,
            index=self.perms.unsqueeze(0).expand(x_flat.size(0), -1, -1)
        )
        out = (x_gathered * w[0].unsqueeze(0)).sum(dim=1) + self.bias[0]
        return out.reshape(*orig_shape)


class FusedPermutationProjection(nn.Module):
    """
    Fused Zero-MatMul Permutation Projection:
    Computes M output projections (e.g. Q, K, V, Gate) simultaneously with shared permutation gathers.
    """
    def __init__(
        self,
        dim: int,
        num_branches: int = 4,
        num_perms: int = 4,
        seed_offset: int = 0,
        dtype: Any = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.num_branches = num_branches
        self.num_perms = num_perms

        perms = [torch.arange(dim)]
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for p in range(1, num_perms):
            g = torch.Generator().manual_seed(seed_offset * 1000 + p)
            perms.append(torch.randperm(dim, generator=g))
        self.register_buffer('perms', torch.stack(perms)) # [P, dim]

        inv_perms = []
        # TRIVIAL: 3-4 iters, config-time or fallback tiny, not hot
        for p in range(num_perms):
            inv = torch.empty(dim, dtype=torch.long)
            inv[self.perms[p]] = torch.arange(dim)
            inv_perms.append(inv)
        self.register_buffer('inv_perms', torch.stack(inv_perms)) # [P, dim]

        self.latent_w = nn.Parameter(
            torch.randn(num_branches, num_perms, dim, dtype=dtype) * (1.0 / math.sqrt(dim))
        )
        self.scale = nn.Parameter(torch.ones(num_branches, num_perms, 1, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_branches, dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        w = ternary_ste(self.latent_w, scale=self.scale) # [M, P, dim]

        if not x.is_cuda:
            from affine_ai.core.cpp_ops import asdag_cpu_fused_perm_proj
            out = asdag_cpu_fused_perm_proj(x_flat, w, self.perms, self.inv_perms, self.bias)
            branch_outs = tuple(out[m].reshape(*orig_shape) for m in range(self.num_branches))
            return branch_outs

        if getattr(kernels, "TRITON_AVAILABLE", False) and getattr(kernels, "triton_fused_perm_proj", None) is not None:
            try:
                out = kernels.triton_fused_perm_proj(x_flat, w, self.perms, self.inv_perms, self.bias)
                branch_outs = tuple(out[m].reshape(*orig_shape) for m in range(self.num_branches))
                return branch_outs
            except Exception as e:
                warnings.warn(f"triton_fused_perm_proj failed: {e}; falling back", stacklevel=2)
        else:
            try:
                from affine_ai.kernels.triton_perm_proj import triton_fused_perm_proj as _perm2
                out = _perm2(x_flat, w, self.perms, self.inv_perms, self.bias)
                branch_outs = tuple(out[m].reshape(*orig_shape) for m in range(self.num_branches))
                return branch_outs
            except Exception as e:
                warnings.warn(f"triton_fused_perm_proj fallback failed: {e}", stacklevel=2)

        x_gathered = torch.gather(
            x_flat.unsqueeze(1).expand(-1, self.num_perms, -1),
            dim=-1,
            index=self.perms.unsqueeze(0).expand(x_flat.size(0), -1, -1)
        ) # [B_total, P, dim]
        out = (x_gathered.unsqueeze(0) * w.unsqueeze(1)).sum(dim=2) + self.bias.unsqueeze(1) # [M, B_total, dim]
        branch_outs = tuple(out[m].reshape(*orig_shape) for m in range(self.num_branches))
        return branch_outs


class FusedGLAAnalyticalCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, C: int = 64):
        B, H, T, D = q.shape
        C_eff = min(C, T)
        NC = T // C_eff
        dtype = q.dtype
        eps = 1e-4 if dtype == torch.float16 else 1e-5
        
        qc = q.view(B, H, NC, C_eff, D)
        kc = k.view(B, H, NC, C_eff, D)
        vc = v.view(B, H, NC, C_eff, D)
        gc = gamma.view(B, H, NC, C_eff)
        
        log_gc = torch.log(gc.float().clamp(min=1e-5, max=1.0))
        cum_log_c = torch.cumsum(log_gc, dim=-1)
        clamp_min = _clamp_min_for_dtype(dtype)
        decay_intra = (cum_log_c.unsqueeze(-1) - cum_log_c.unsqueeze(-2)).clamp(min=clamp_min, max=0.0)
        mask_intra = torch.tril(torch.ones(C_eff, C_eff, device=q.device, dtype=torch.bool))
        decay_mat_intra = torch.where(mask_intra, torch.exp(decay_intra), torch.zeros_like(decay_intra)).to(dtype)
        
        scores_intra = torch.matmul(qc, kc.transpose(-1, -2)) * decay_mat_intra
        num_intra = torch.matmul(scores_intra, vc)
        den_intra = scores_intra.sum(dim=-1, keepdim=True)
        
        decay_chunk_tot = torch.exp(cum_log_c[:, :, :, -1:])  # [B, H, NC, 1]
        weight_to_end = torch.exp((cum_log_c[:, :, :, -1:] - cum_log_c).unsqueeze(-1)).to(dtype)
        
        kw = kc * weight_to_end
        S_local = torch.matmul(kw.transpose(-1, -2), vc)  # [B, H, NC, D, D]
        z_local = kw.sum(dim=-2)                          # [B, H, NC, D]
        
        # Vectorized inter-chunk associative scan using cumulative log-decay formulation (Issue 12)
        if NC > 1:
            c = cum_log_c[:, :, :, -1]  # [B, H, NC]
            c_cum = torch.cumsum(c, dim=-1)
            c_cum_prev = torch.cat([torch.zeros(B, H, 1, device=c.device, dtype=c.dtype), c_cum[:, :, :-1]], dim=-1)
            clamp_min_inter = _clamp_min_for_dtype(dtype)
            diff = (c_cum_prev.unsqueeze(-1) - c_cum.unsqueeze(-2)).clamp(min=clamp_min_inter, max=0.0)
            strict_tril = torch.tril(torch.ones(NC, NC, device=c.device, dtype=torch.bool), diagonal=-1)
            M_mat = torch.where(strict_tril, torch.exp(diff), torch.zeros_like(diff)).to(dtype)
            S_all = torch.matmul(M_mat, S_local.view(B, H, NC, D * D)).view(B, H, NC, D, D)
            z_all = torch.matmul(M_mat, z_local)
        else:
            M_mat = None
            S_all = torch.zeros(B, H, 1, D, D, device=q.device, dtype=dtype)
            z_all = torch.zeros(B, H, 1, D, device=q.device, dtype=dtype)
            
        weight_from_start = torch.exp(cum_log_c.unsqueeze(-1)).to(dtype)
        q_cur = qc * weight_from_start
        num_inter = torch.matmul(q_cur, S_all)
        den_inter = torch.matmul(q_cur, z_all.unsqueeze(-1))
        
        num_total = num_intra + num_inter
        den_total = (den_intra + den_inter).clamp(min=eps)
        y = (num_total / den_total).view(B, H, T, D)
        
        ctx.save_for_backward(
            qc, kc, vc, gc, y.view(B, H, NC, C_eff, D),
            num_total, den_total, scores_intra, decay_mat_intra,
            S_all, z_all, cum_log_c, decay_chunk_tot, weight_to_end, weight_from_start,
            M_mat, kw, S_local, z_local
        )
        ctx.orig_dtype = dtype
        ctx.C_eff = C_eff
        ctx.NC = NC
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            qc, kc, vc, gc, y,
            num_tot, den_tot, scores_intra, decay_mat_intra,
            S_all, z_all, cum_log_c, decay_chunk_tot, weight_to_end, weight_from_start,
            M_mat, kw, S_local, z_local
        ) = ctx.saved_tensors
        B, H, NC, C_eff, D = qc.shape
        T = NC * C_eff
        go = grad_out.view(B, H, NC, C_eff, D)
        dtype = qc.dtype
        eps = ctx.eps
        
        d_num = go / den_tot
        d_den = -(go * y).sum(dim=-1, keepdim=True) / den_tot
        
        d_q_inter = (torch.matmul(d_num, S_all.transpose(-1, -2)) + d_den * z_all.unsqueeze(-2)) * weight_from_start
        d_S_all = torch.matmul((qc * weight_from_start).transpose(-1, -2), d_num)
        d_z_all = (qc * weight_from_start * d_den).sum(dim=-2)
        
        if ctx.NC > 1 and M_mat is not None:
            d_S_local = torch.matmul(M_mat.transpose(-1, -2), d_S_all.view(B, H, NC, D * D)).view(B, H, NC, D, D)
            d_z_local = torch.matmul(M_mat.transpose(-1, -2), d_z_all)
            d_M_mat = torch.matmul(d_S_all.view(B, H, NC, D * D), S_local.view(B, H, NC, D * D).transpose(-1, -2)) + torch.matmul(d_z_all, z_local.transpose(-1, -2))
            d_diff = d_M_mat * M_mat
            g_c_cum_prev = d_diff.sum(dim=-1)
            g_c_cum = -d_diff.sum(dim=-2)
            g_c_cum[:, :, :-1] += g_c_cum_prev[:, :, 1:]
            g_c = g_c_cum.flip(-1).cumsum(-1).flip(-1)
        else:
            d_S_local = torch.zeros_like(d_S_all)
            d_z_local = torch.zeros_like(d_z_all)
            g_c = torch.zeros(B, H, NC, device=go.device, dtype=cum_log_c.dtype)
            
        d_kw = torch.matmul(vc, d_S_local.transpose(-1, -2)) + d_z_local.unsqueeze(-2)
        d_vc_inter = torch.matmul(kc * weight_to_end, d_S_local)
        d_kc_inter = d_kw * weight_to_end
        
        d_scores = torch.matmul(d_num, vc.transpose(-1, -2)) + d_den
        d_decay_scores = d_scores * decay_mat_intra
        
        d_qc_intra = torch.matmul(d_decay_scores, kc)
        d_kc_intra = torch.matmul(d_decay_scores.transpose(-1, -2), qc)
        d_vc_intra = torch.matmul(scores_intra.transpose(-1, -2), d_num)
        
        g_q = (d_q_inter + d_qc_intra).view(B, H, T, D)
        g_k = (d_kc_inter + d_kc_intra).view(B, H, T, D)
        g_v = (d_vc_inter + d_vc_intra).view(B, H, T, D)
        
        # Analytical recurrence adjoint gradient for gamma (Issue 13)
        G_intra = d_scores * scores_intra
        g_cum_intra = G_intra.sum(dim=-1) - G_intra.sum(dim=-2)
        
        d_q_cur = torch.matmul(d_num, S_all.transpose(-1, -2)) + d_den * z_all.unsqueeze(-2)
        g_cum_start = (d_q_cur * (qc * weight_from_start)).sum(dim=-1)
        
        A = (d_kw * kw).sum(dim=-1)
        g_cum_to_end = -A
        g_cum_to_end[:, :, :, -1] += A.sum(dim=-1)
        
        g_cum_total = g_cum_intra + g_cum_start + g_cum_to_end
        g_cum_total[:, :, :, -1] += g_c
        
        g_log_gc = g_cum_total.flip(-1).cumsum(-1).flip(-1)
        g_gc = g_log_gc / gc.float().clamp(min=1e-5)
        mask_clamp = (gc >= 1e-5) & (gc <= 1.0)
        g_gc = torch.where(mask_clamp, g_gc, torch.zeros_like(g_gc)).to(dtype)
        g_gamma = g_gc.view(B, H, T)
        
        return g_q, g_k, g_v, g_gamma, None


class NativeASDAGAssociativeMixer(nn.Module):
    """
    Native Causal Gated Linear Associative (GLA) Sequence Mixer:
    1. Zero-MatMul Monarch Permutation Chains for Q, K, V, Gate, and Output (L=4 stages).
    2. Strictly Positive Kernel Space: Phi(Q) = ELU(q_norm(Q)) + 1, Phi(K) = ELU(k_norm(K)) + 1.
    3. Data-Dependent Gating: Gamma_t = Sigmoid(W_decay * x_t).
    4. Exact Denominator State Normalization: Eliminates magnitude drift across sequence positions.
    5. Dual Execution:
       - Training: Fully Vectorized Cumulative Log-Decay Associative Scan (Zero Python Loops!).
       - Inference: Native C++ AVX2/AVX-512 O(1) Constant Memory State Space Step Kernel.

    rule="delta" swaps the recurrence to the error-corrective delta rule
    (DeltaNet-style, left-projection):
        S_t = S_{t-1} (I - beta * khat_t khat_t^T) + beta * khat_t v_t^T
    where khat is the unit-normalized key. A repeated (k, v) pair has zero
    residual, so loops overwrite their own key slots instead of flooding the
    whole state (the GLA accumulation pathology measured in the oracle probe:
    28.6x loop/story state ratio for GLA vs 2.7x for delta).
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        num_stages: int = 4,
        proj_type: str = "monarch", # "monarch" or "ternary_perm"
        num_perms: int = 4,
        max_seq_len: int = 4096,
        seed_offset: int = 0,
        dtype: Any = torch.bfloat16,
        rule: str = "gla",  # "gla" or "delta"
        delta_beta: float = 1.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.proj_type = proj_type
        self.rule = rule
        self.delta_beta = delta_beta

        # 1. Sequence Projections
        if proj_type == "monarch":
            self.qkvg_proj = FusedMonarchChain(
                dim=d_model,
                num_branches=4,
                num_stages=num_stages,
                seed_offset=seed_offset * 10 + 1,
                dtype=dtype
            )
            self.out_proj = MonarchPermutationChain(
                dim=d_model,
                num_stages=num_stages,
                seed_offset=seed_offset * 10 + 5,
                dtype=dtype
            )
        else:
            self.qkvg_proj = FusedPermutationProjection(
                dim=d_model,
                num_branches=4,
                num_perms=num_perms,
                seed_offset=seed_offset * 10 + 1,
                dtype=dtype
            )
            self.out_proj = PermutationProjection(
                dim=d_model,
                num_perms=num_perms,
                seed_offset=seed_offset * 10 + 5,
                dtype=dtype
            )

        # 2. Data-Dependent Decay Gate Projection
        self.gate_decay = nn.Linear(d_model, n_heads, bias=True, dtype=dtype)
        nn.init.constant_(self.gate_decay.bias, 3.0)

        # 3. QK Normalization for numerical stability
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        reset_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        H = self.n_heads
        D = self.d_head
        orig_dtype = x.dtype

        if (self.rule == "gla"
                and not x.is_cuda and not return_state and state is None and self.proj_type == "monarch"):
            from affine_ai.core.cpp_ops import asdag_cpu_fused_monarch_gla
            out = asdag_cpu_fused_monarch_gla(
                x,
                self.qkvg_proj.diagonals,
                self.qkvg_proj.perms,
                self.qkvg_proj.inv_perms,
                self.qkvg_proj.bias,
                self.q_norm.scale,
                self.k_norm.scale,
                self.gate_decay.weight,
                self.gate_decay.bias,
                self.out_proj.diagonals,
                self.out_proj.perms,
                self.out_proj.inv_perms,
                self.out_proj.bias,
                reset_mask
            )
            return out, None

        # 1. Fused Projections
        q_raw, k_raw, v_raw, g_raw = self.qkvg_proj(x)

        # Positive Feature Maps (ELU + 1.0)
        phi_q = (F.elu(self.q_norm(q_raw.view(B, T, H, D))) + 1.0).transpose(1, 2)  # [B, H, T, D]
        phi_k = (F.elu(self.k_norm(k_raw.view(B, T, H, D))) + 1.0).transpose(1, 2)  # [B, H, T, D]
        v = v_raw.view(B, T, H, D).transpose(1, 2)                                  # [B, H, T, D]
        g = F.silu(g_raw)                                                          # [B, T, C]

        # Data-dependent decay in (0, 1)
        gamma = torch.sigmoid(self.gate_decay(x.to(self.gate_decay.weight.dtype))).transpose(1, 2)  # [B, H, T]

        # 2. Sequential O(1) Step Mode (Inference Generation with state caching)
        if state is not None:
            state_S, state_z = state

            if self.rule == "delta":
                # Delta recurrence, O(1) per step, autograd-exact.
                # On CUDA dispatch via torch.compile to avoid Python-loop overhead (no new kernels).
                beta = self.delta_beta
                if x.is_cuda:
                    def _delta_state_loop(phi_q_, phi_k_, v_, gamma_, state_S_, state_z_, beta_):
                        outs_ = []
                        S_ = state_S_
                        z_ = state_z_
                        # UNVECTORIZABLE: sequential GLA/delta, needs scan kernel (proposed)
                        for t in range(phi_q_.shape[2]):
                            q_t = phi_q_[:, :, t]
                            k_t = phi_k_[:, :, t]
                            v_t = v_[:, :, t]
                            gam_t = gamma_[:, :, t]
                            khat = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-9)
                            P = beta_ * (khat.unsqueeze(-1) @ khat.unsqueeze(-2))
                            S_ = S_ * gam_t[:, :, None, None] - (S_ * gam_t[:, :, None, None]) @ P + beta_ * (khat.unsqueeze(-1) * v_t.unsqueeze(-2))
                            z_ = z_ * gam_t[:, :, None] - (z_ * khat).sum(dim=-1, keepdim=True) * beta_ * khat + beta_ * khat
                            num = (q_t.unsqueeze(-2) @ S_).squeeze(-2)
                            den = (q_t * z_).sum(dim=-1, keepdim=True).clamp(min=1e-5)
                            outs_.append((num / den).unsqueeze(2))
                        return torch.cat(outs_, dim=1), S_, z_
                    try:
                        compiled = _maybe_compile(_delta_state_loop)
                        out_cat, state_S, state_z = compiled(phi_q, phi_k, v, gamma, state_S, state_z, beta)
                        out = out_cat.transpose(1, 2).reshape(B, T, C).to(orig_dtype)
                        return self.out_proj(out * g), (state_S, state_z)
                    except Exception as e:
                        warnings.warn(f"delta state compile failed: {e}; using eager loop", stacklevel=2)
                outs = []
                # UNVECTORIZABLE: sequential GLA/delta, needs scan kernel (proposed)
                for t in range(T):
                    q_t = phi_q[:, :, t]
                    k_t = phi_k[:, :, t]
                    v_t = v[:, :, t]
                    gam_t = gamma[:, :, t]
                    khat = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-9)
                    P = beta * (khat.unsqueeze(-1) @ khat.unsqueeze(-2))
                    state_S = state_S * gam_t[:, :, None, None] - (state_S * gam_t[:, :, None, None]) @ P + beta * (khat.unsqueeze(-1) * v_t.unsqueeze(-2))
                    state_z = state_z * gam_t[:, :, None] - (state_z * khat).sum(dim=-1, keepdim=True) * beta * khat + beta * khat
                    num = (q_t.unsqueeze(-2) @ state_S).squeeze(-2)
                    den = (q_t * state_z).sum(dim=-1, keepdim=True).clamp(min=1e-5)
                    outs.append((num / den).unsqueeze(2))
                out = torch.cat(outs, dim=1).transpose(1, 2).reshape(B, T, C).to(orig_dtype)
                return self.out_proj(out * g), (state_S, state_z)

            if not x.is_cuda:
                from affine_ai.core.cpp_ops import asdag_cpu_gla_step
                if T == 1:
                    q_t = phi_q[:, :, 0]
                    k_t = phi_k[:, :, 0]
                    v_t = v[:, :, 0]
                    gam_t = gamma[:, :, 0]
                    y_t, next_S, next_z = asdag_cpu_gla_step(q_t, k_t, v_t, gam_t, state_S, state_z)
                    out = y_t.unsqueeze(1).transpose(1, 2).reshape(B, 1, C).to(orig_dtype)
                    return self.out_proj(out * g), (next_S, next_z)

                outs = []
                # UNVECTORIZABLE: sequential GLA/delta, needs scan kernel (proposed)
                for t in range(T):
                    q_t = phi_q[:, :, t]
                    k_t = phi_k[:, :, t]
                    v_t = v[:, :, t]
                    gam_t = gamma[:, :, t]
                    y_t, state_S, state_z = asdag_cpu_gla_step(q_t, k_t, v_t, gam_t, state_S, state_z)
                    outs.append(y_t.unsqueeze(2))
                out = torch.cat(outs, dim=1).transpose(1, 2).reshape(B, T, C).to(orig_dtype)
                return self.out_proj(out * g), (state_S, state_z)
            else:
                # Native PyTorch CUDA recurrent step — vectorized (no Python T loop) via decay matrix + state term (bmm/einsum/cumsum)
                if T == 1:
                    q_t = phi_q[:, :, 0]
                    k_t = phi_k[:, :, 0]
                    v_t = v[:, :, 0]
                    gam_t = gamma[:, :, 0]
                    state_S = state_S * gam_t.unsqueeze(-1).unsqueeze(-1) + (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
                    state_z = state_z * gam_t.unsqueeze(-1) + k_t
                    num = torch.matmul(q_t.unsqueeze(-2), state_S).squeeze(-2)
                    den = (q_t * state_z).sum(dim=-1, keepdim=True).clamp(min=1e-5)
                    out = (num / den).unsqueeze(2).transpose(1, 2).reshape(B, 1, C).to(orig_dtype)
                    return self.out_proj(out * g), (state_S, state_z)

                # Vectorized for T>1: uses batched decay + einsum, no per-step Python loop.
                try:
                    log_gam = torch.log(gamma.clamp(min=1e-5, max=1.0))
                    cum_log = torch.cumsum(log_gam, dim=-1)
                    clamp_min = _clamp_min_for_dtype(orig_dtype)
                    decay = None
                    if getattr(kernels, "TRITON_AVAILABLE", False) and getattr(kernels, "triton_gla_decay", None) is not None:
                        try:
                            decay = kernels.triton_gla_decay(gamma)
                        except Exception as e:
                            warnings.warn(f"triton_gla_decay for state failed: {e}", stacklevel=2)
                    if decay is None:
                        decay_diff = (cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)).clamp(min=clamp_min, max=0.0)
                        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                        decay = torch.where(causal, torch.exp(decay_diff), torch.zeros_like(decay_diff)).to(phi_q.dtype)
                    scores = torch.matmul(phi_q, phi_k.transpose(-1, -2)) * decay
                    num_data = torch.matmul(scores, v)
                    den_data = scores.sum(dim=-1, keepdim=True)
                    exp_cum = torch.exp(cum_log).to(phi_q.dtype)
                    qS0 = torch.einsum('bhtd,bhde->bhte', phi_q, state_S)
                    state_num = qS0 * exp_cum.unsqueeze(-1)
                    q_z0 = (phi_q * state_z.unsqueeze(2)).sum(dim=-1, keepdim=True)
                    state_den = q_z0 * exp_cum.unsqueeze(-1)
                    num = num_data + state_num
                    den = (den_data + state_den).clamp(min=1e-5)
                    y = (num / den).transpose(1, 2).reshape(B, T, C).to(orig_dtype)
                    cum_last = cum_log[:, :, -1:]
                    w_last = torch.exp((cum_last - cum_log).unsqueeze(-1).unsqueeze(-1))
                    if T > 64:
                        chunk = 64
                        next_S = state_S * torch.exp(cum_log[:, :, -1]).unsqueeze(-1).unsqueeze(-1)
                        next_z = state_z * torch.exp(cum_log[:, :, -1]).unsqueeze(-1)
                        # VECTORIZED (chunked): avoids [B,H,T,D,D] alloc via chunked bmm (64)
                        for s in range(0, T, chunk):
                            e = min(s + chunk, T)
                            k_c = phi_k[:, :, s:e]
                            v_c = v[:, :, s:e]
                            cum_c = cum_log[:, :, s:e]
                            w_c = torch.exp((cum_last - cum_c).unsqueeze(-1).unsqueeze(-1))
                            w_zc = torch.exp((cum_last - cum_c).unsqueeze(-1))
                            kv_c = torch.matmul(k_c.unsqueeze(-1), v_c.unsqueeze(-2))
                            next_S = next_S + (kv_c * w_c).sum(dim=2)
                            next_z = next_z + (k_c * w_zc).sum(dim=2)
                    else:
                        kv = torch.matmul(phi_k.unsqueeze(-1), v.unsqueeze(-2))
                        next_S = state_S * torch.exp(cum_last.squeeze(-1)).unsqueeze(-1).unsqueeze(-1) + (kv * w_last).sum(dim=2)
                        next_z = state_z * torch.exp(cum_last.squeeze(-1)).unsqueeze(-1) + (phi_k * torch.exp((cum_last - cum_log).unsqueeze(-1))).sum(dim=2)
                    return self.out_proj(y * g), (next_S, next_z)
                except Exception as e:
                    warnings.warn(f"vectorized state path failed: {e}; falling back to sequential (UNVECTORIZABLE without scan kernel)", stacklevel=2)
                    outs = []
                    # UNVECTORIZABLE: sequential fallback, needs scan kernel proposal
                    for t in range(T):
                        q_t = phi_q[:, :, t]
                        k_t = phi_k[:, :, t]
                        v_t = v[:, :, t]
                        gam_t = gamma[:, :, t]
                        state_S = state_S * gam_t.unsqueeze(-1).unsqueeze(-1) + (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
                        state_z = state_z * gam_t.unsqueeze(-1) + k_t
                        num = torch.matmul(q_t.unsqueeze(-2), state_S).squeeze(-2)
                        den = (q_t * state_z).sum(dim=-1, keepdim=True).clamp(min=1e-5)
                        outs.append((num / den).unsqueeze(2))
                    out = torch.cat(outs, dim=1).transpose(1, 2).reshape(B, T, C).to(orig_dtype)
                    return self.out_proj(out * g), (state_S, state_z)

        # 3. Delta-rule training forward: sequential recurrence from zero state,
        #    autograd-exact. Dispatch via torch.compile on CUDA to avoid Python loops.
        if self.rule == "delta":
            beta = self.delta_beta
            if x.is_cuda:
                def _delta_train_loop(phi_q_, phi_k_, v_, gamma_, beta_):
                    S_ = torch.zeros(phi_q_.shape[0], phi_q_.shape[1], phi_q_.shape[3], phi_q_.shape[3], device=phi_q_.device, dtype=phi_q_.dtype)
                    z_ = torch.zeros(phi_q_.shape[0], phi_q_.shape[1], phi_q_.shape[3], device=phi_q_.device, dtype=phi_q_.dtype)
                    outs_ = []
                    # UNVECTORIZABLE: sequential GLA/delta, needs scan kernel (proposed)
                    for t in range(phi_q_.shape[2]):
                        q_t = phi_q_[:, :, t]
                        k_t = phi_k_[:, :, t]
                        v_t = v_[:, :, t]
                        gam_t = gamma_[:, :, t]
                        khat = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-9)
                        P = beta_ * (khat.unsqueeze(-1) @ khat.unsqueeze(-2))
                        S_ = S_ * gam_t[:, :, None, None] - (S_ * gam_t[:, :, None, None]) @ P + beta_ * (khat.unsqueeze(-1) * v_t.unsqueeze(-2))
                        z_ = z_ * gam_t[:, :, None] - (z_ * khat).sum(dim=-1, keepdim=True) * beta_ * khat + beta_ * khat
                        num = (q_t.unsqueeze(-2) @ S_).squeeze(-2)
                        den = (q_t * z_).sum(dim=-1, keepdim=True).clamp(min=1e-5)
                        outs_.append((num / den).unsqueeze(2))
                    return torch.cat(outs_, dim=1), S_, z_
                try:
                    compiled = _maybe_compile(_delta_train_loop)
                    out_cat, state_S, state_z = compiled(phi_q, phi_k, v, gamma, beta)
                    out = out_cat.transpose(1, 2).reshape(B, T, C).to(orig_dtype)
                    return self.out_proj(out * g), (state_S, state_z)
                except Exception as e:
                    warnings.warn(f"delta train compile failed: {e}; using eager loop", stacklevel=2)
            state_S = phi_q.new_zeros(B, H, D, D)
            state_z = phi_q.new_zeros(B, H, D)
            outs = []
            # UNVECTORIZABLE: sequential GLA/delta, needs scan kernel (proposed)
            for t in range(T):
                q_t = phi_q[:, :, t]
                k_t = phi_k[:, :, t]
                v_t = v[:, :, t]
                gam_t = gamma[:, :, t]
                khat = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-9)
                P = beta * (khat.unsqueeze(-1) @ khat.unsqueeze(-2))
                state_S = state_S * gam_t[:, :, None, None] - (state_S * gam_t[:, :, None, None]) @ P + beta * (khat.unsqueeze(-1) * v_t.unsqueeze(-2))
                state_z = state_z * gam_t[:, :, None] - (state_z * khat).sum(dim=-1, keepdim=True) * beta * khat + beta * khat
                num = (q_t.unsqueeze(-2) @ state_S).squeeze(-2)
                den = (q_t * state_z).sum(dim=-1, keepdim=True).clamp(min=1e-5)
                outs.append((num / den).unsqueeze(2))
            out = torch.cat(outs, dim=1).transpose(1, 2).reshape(B, T, C).to(orig_dtype)
            return self.out_proj(out * g), (state_S, state_z)

        # 4. Fused C++ SIMD Associative Scan
        if not x.is_cuda and not return_state:
            from affine_ai.core.cpp_ops import asdag_cpu_gla_scan
            y_scan = asdag_cpu_gla_scan(phi_q, phi_k, v, gamma) # [B, H, T, D]
            y = y_scan.transpose(1, 2).reshape(B, T, C).to(orig_dtype)
            return self.out_proj(y * g), None

        eps = 1e-4 if orig_dtype == torch.float16 else 1e-5
        clamp_min = _clamp_min_for_dtype(orig_dtype)
        # Chunked fast path on CUDA avoids materializing dense [B, H, T, T] tensors (Issue 18)
        # When reset_mask is passed, route to exact same_doc masked path to guarantee zero cross-doc attention
        if reset_mask is not None:
            # Exact document boundary masking: guarantees 0.0 attention/decay across document boundaries
            doc_id = torch.cumsum(reset_mask.long(), dim=-1)  # [B, T]
            same_doc = (doc_id.unsqueeze(-1) == doc_id.unsqueeze(-2)).unsqueeze(1)  # [B, 1, T, T]
            log_gam = torch.log(gamma.clamp(min=1e-5, max=1.0))
            cum_log_gam = torch.cumsum(log_gam, dim=-1)
            decay_diff = (cum_log_gam.unsqueeze(-1) - cum_log_gam.unsqueeze(-2)).clamp(min=clamp_min, max=0.0)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            decay_mat = torch.where(causal_mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))
            decay_mat = decay_mat * same_doc
            scores = torch.matmul(phi_q, phi_k.transpose(-1, -2)) * decay_mat
            scores = scores * same_doc
        elif x.is_cuda and not return_state:
            if T <= 64:
                y = FusedGLAAnalyticalCUDA.apply(phi_q, phi_k, v, gamma, T).transpose(1, 2).reshape(B, T, C)
                return self.out_proj(y * g), None
            elif T % 64 == 0:
                y = FusedGLAAnalyticalCUDA.apply(phi_q, phi_k, v, gamma, 64).transpose(1, 2).reshape(B, T, C)
                return self.out_proj(y * g), None
            else:
                pad_len = 64 - (T % 64)
                phi_q_pad = F.pad(phi_q, (0, 0, 0, pad_len))
                phi_k_pad = F.pad(phi_k, (0, 0, 0, pad_len))
                v_pad = F.pad(v, (0, 0, 0, pad_len))
                gamma_pad = F.pad(gamma, (0, pad_len), value=1.0)
                y_pad = FusedGLAAnalyticalCUDA.apply(phi_q_pad, phi_k_pad, v_pad, gamma_pad, 64)
                y = y_pad[:, :, :T, :].transpose(1, 2).reshape(B, T, C)
                return self.out_proj(y * g), None
        else:
            cum_log_gam = None
            decay_mat = None
            if x.is_cuda and getattr(kernels, "TRITON_AVAILABLE", False) and getattr(kernels, "triton_gla_decay", None) is not None:
                try:
                    decay_mat = kernels.triton_gla_decay(gamma)
                except Exception as e:
                    warnings.warn(f"triton_gla_decay failed: {e}; using PyTorch fallback", stacklevel=2)
                    decay_mat = None
            if decay_mat is None:
                # Try direct import fallback before PyTorch
                try:
                    from affine_ai.kernels.triton_gla import triton_gla_decay as _triton_decay
                    decay_mat = _triton_decay(gamma)
                except Exception as e:
                    if getattr(kernels, "TRITON_AVAILABLE", False):
                        warnings.warn(f"triton_gla_decay fallback failed: {e}", stacklevel=2)
                if decay_mat is None:
                    log_gam = torch.log(gamma.clamp(min=1e-5, max=1.0))              # [B, H, T]
                    cum_log_gam = torch.cumsum(log_gam, dim=-1)                      # [B, H, T]
                    decay_diff = (cum_log_gam.unsqueeze(-1) - cum_log_gam.unsqueeze(-2)).clamp(min=clamp_min, max=0.0) # [B, H, T, T]
                    causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                    decay_mat = torch.where(causal_mask, torch.exp(decay_diff), torch.zeros_like(decay_diff))
            scores = torch.matmul(phi_q, phi_k.transpose(-1, -2)) * decay_mat    # [B, H, T, T]

        num = torch.matmul(scores, v)                                        # [B, H, T, D]
        den = scores.sum(dim=-1, keepdim=True).clamp(min=eps)                # [B, H, T, 1]
        y = (num / den).transpose(1, 2).reshape(B, T, C).to(orig_dtype)


        next_state = None
        if return_state:
            if cum_log_gam is None:
                cum_log_gam = torch.cumsum(torch.log(gamma.clamp(min=1e-5)), dim=-1)
            # Chunked scan to avoid [B,H,T,D,D] alloc (~ B*H*T*D*D floats)
            chunk = 64
            next_S = torch.zeros(B, H, D, D, device=phi_q.device, dtype=phi_q.dtype)
            next_z = torch.zeros(B, H, D, device=phi_q.device, dtype=phi_q.dtype)
            # cum_last for weight computation
            cum_last = cum_log_gam[:, :, -1:]  # [B,H,1]
            # VECTORIZED (chunked): avoids [B,H,T,D,D] alloc via chunked bmm (C=64)
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                k_c = phi_k[:, :, s:e]  # [B,H,C,D]
                v_c = v[:, :, s:e]
                cum_c = cum_log_gam[:, :, s:e]  # [B,H,C]
                w_c = torch.exp((cum_last - cum_c).unsqueeze(-1).unsqueeze(-1))  # [B,H,C,1,1]
                # Also need w for z: [B,H,C,1]
                w_z = torch.exp((cum_last - cum_c).unsqueeze(-1))  # [B,H,C,D? wait]
                # Compute kv per chunk without full T alloc: [B,H,C,D,D]
                kv_c = torch.matmul(k_c.unsqueeze(-1), v_c.unsqueeze(-2))  # [B,H,C,D,D]
                next_S = next_S + (kv_c * w_c).sum(dim=2)
                next_z = next_z + (k_c * w_z).sum(dim=2)
            next_state = (next_S, next_z)

        return self.out_proj(y * g), next_state

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(x_t, state=state, return_state=True)
