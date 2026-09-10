"""Triton fused sparse-tree-perm forward (CUDA).

Fuses the ASTDAG perm-leaf CUDA fallback path:
  leaf_prim[b,k,d] = bias[k,d] + sum_p w_perm[k,p,d] * r_in[b, perms[k,p,d]]
  act = relu6(leaf_prim)
  out[b,d] = sum_k routing_sparse[b,k] * act[b,k,d]

...into a single kernel that evaluates ONLY the top-k active leaves per
row (the reference computes all K then masks). Elementwise FMAs only
(SIMT, no Tensor Cores). Backward computes analytical gradients
directly via fused Triton kernels and direct index addition.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import warnings


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


@triton.jit
def _tree_perm_fwd_kernel(
    R, W, B, P, TI, TW, Y,
    stride_rm, stride_rd,
    stride_wk, stride_wp, stride_wd,
    stride_bk, stride_bd,
    stride_pk, stride_pp, stride_pd,
    stride_tbt, stride_tbk,
    stride_twb, stride_twk,
    stride_ym, stride_yd,
    B_ROWS: tl.constexpr, K_LEAVES: tl.constexpr, D_DIM: tl.constexpr, P_NUM: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Loop bounds P_NUM/TOPK/B_ROWS/K_LEAVES/D_DIM are tl.constexpr for unrolling (already constexpr)
    # Grid coalescing: where K*B_rows product large, suggest coalescing blocks via 2D tiling but keep as is for now
    # (heavy K*B tail would benefit from flattened grid, but current (B_rows//BLOCK_B, D//BLOCK_D) is cache-friendly for small TOPK).
    # Block_ptr optimization: Y store can be block_ptr (contiguous D when stride_yd==1) for coalesced stores.
    # R/W contiguous loads via block_ptr where possible, but gather via perm keeps manual due to indirect indexing.
    # R is [B,D] accessible via block_ptr for gathered dimension still manual (random perm), but contiguous chunks use block_ptr.
    # Keep manual pointer arithmetic as fallback for non-contiguous/strided views (comment below).
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_b = offs_b < B_ROWS
    mask_d = offs_d < D_DIM
    mask = mask_b[:, None] & mask_d[None, :]

    acc_total = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    for tk in range(TOPK):
        ki = tl.load(TI + offs_b * stride_tbt + tk * stride_tbk, mask=mask_b, other=0, eviction_policy="evict_last")
        tw = tl.load(TW + offs_b * stride_twb + tk * stride_twk, mask=mask_b, other=0.0, eviction_policy="evict_last")
        ki = tl.where(mask_b, ki, 0)
        # tw zeroing for masked rows: make masked lanes explicit
        tw = tl.where(mask_b, tw, 0.0)

        acc = tl.load(
            B + ki[:, None] * stride_bk + offs_d[None, :] * stride_bd,
            mask=mask, other=0.0,
            eviction_policy="evict_last",
        )
        # Note: contiguous Bias/W loads could use block_ptr if ki uniform; but ki varies per row (topk per sample), so manual gather per ki is required
        # Fallback manual kept; block_ptr for uniform case would be:
        # b_block_ptr = tl.make_block_ptr(base=B, shape=(K_LEAVES, D_DIM), strides=(stride_bk, stride_bd), offsets=(0, pid_d*BLOCK_D), block_shape=(1, BLOCK_D), order=(1,0)) — not usable with per-row ki
        for p in range(P_NUM):
            w = tl.load(
                W + ki[:, None] * stride_wk + p * stride_wp + offs_d[None, :] * stride_wd,
                mask=mask, other=0.0,
                eviction_policy="evict_last",
            )
            pm = tl.load(
                P + ki[:, None] * stride_pk + p * stride_pp + offs_d[None, :] * stride_pd,
                mask=mask, other=0,
                eviction_policy="evict_last",
            )
            # Bounds clamp for gather pm without bounds check
            pm = tl.where((pm >= 0) & (pm < D_DIM), pm, 0)
            xv = tl.load(
                R + offs_b[:, None] * stride_rm + pm * stride_rd,
                mask=mask, other=0.0,
            )
            acc += w * xv
        acc = tl.minimum(tl.maximum(acc, 0.0), 6.0)
        acc_total += acc * tw[:, None]

    # Y store via block_ptr for coalesced store when stride_yd==1 (contiguous D)
    y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(B_ROWS, D_DIM),
        strides=(stride_ym, stride_yd),
        offsets=(pid_b * BLOCK_B, pid_d * BLOCK_D),
        block_shape=(BLOCK_B, BLOCK_D),
        order=(1, 0),
    )
    tl.store(y_block_ptr, acc_total, boundary_check=(0, 1))
    # Fallback manual (non-contiguous):
    # tl.store(Y + offs_b[:, None] * stride_ym + offs_d[None, :] * stride_yd, acc_total, mask=mask)


def triton_tree_perm_fwd(r_in, w_perm, bias, perms, top_idx, top_w):
    B, D = r_in.shape
    K, P, Dp = w_perm.shape
    Tk = top_idx.shape[1]
    assert Dp == D
    # Turing sm_75: bf16 -> fp16 fallback (acc fp32), warn. BLOCK <=64 (64KB SMEM), num_warps 2/4 max.
    if _is_turing() and r_in.dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 not supported, treating as fp16 (acc fp32) in triton_tree_perm_fwd", stacklevel=2)
    if _is_turing() and w_perm.dtype == torch.bfloat16:
        warnings.warn("Turing sm_75: bf16 not supported, treating as fp16 (acc fp32) in triton_tree_perm_fwd (w_perm)", stacklevel=2)
    out = torch.empty((B, D), device=r_in.device, dtype=torch.float32)
    # BLOCK_B=16 BLOCK_D=32 handles non-divisible B/D via masking in kernel (mask_b/mask_d)
    # Turing: clamp to <=64, avoid 128.
    BM, BD = 16, 32
    if _is_turing():
        BM = min(BM, 64)
        BD = min(BD, 64)
    grid = ((B + BM - 1) // BM, (D + BD - 1) // BD)
    _tree_perm_fwd_kernel[grid](
        r_in, w_perm, bias, perms, top_idx, top_w, out,
        r_in.stride(0), r_in.stride(1),
        w_perm.stride(0), w_perm.stride(1), w_perm.stride(2),
        bias.stride(0), bias.stride(1),
        perms.stride(0), perms.stride(1), perms.stride(2),
        top_idx.stride(0), top_idx.stride(1),
        top_w.stride(0), top_w.stride(1),
        out.stride(0), out.stride(1),
        B, K, D, P, Tk,
        BLOCK_B=BM, BLOCK_D=BD, num_warps=4)
    return out


@triton.jit
def _tree_perm_bwd_dprim_kernel(
    R, W, Bias, Perms, TopIdx, TopW, GO,
    D_PRIM, ACT,
    stride_rm, stride_rd,
    stride_wk, stride_wp, stride_wd,
    stride_bk, stride_bd,
    stride_pk, stride_pp, stride_pd,
    stride_tbm, stride_tbtk,
    stride_twm, stride_twtk,
    stride_gom, stride_god,
    stride_dpm, stride_dptk, stride_dpd,
    stride_actm, stride_acttk, stride_actd,
    B_ROWS: tl.constexpr, D_DIM: tl.constexpr, P_NUM: tl.constexpr, TOPK: tl.constexpr,
    STORE_ACT: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_b = offs_b < B_ROWS
    mask_d = offs_d < D_DIM
    mask = mask_b[:, None] & mask_d[None, :]

    # GO load via block_ptr when contiguous (stride_god==1)
    go_block_ptr = tl.make_block_ptr(
        base=GO,
        shape=(B_ROWS, D_DIM),
        strides=(stride_gom, stride_god),
        offsets=(pid_b * BLOCK_B, pid_d * BLOCK_D),
        block_shape=(BLOCK_B, BLOCK_D),
        order=(1, 0),
    )
    go_val = tl.load(go_block_ptr, boundary_check=(0, 1))
    # fallback manual: go_val = tl.load(GO + offs_b[:, None] * stride_gom + offs_d[None, :] * stride_god, mask=mask, other=0.0)

    for tk in range(TOPK):
        ki = tl.load(TopIdx + offs_b * stride_tbm + tk * stride_tbtk, mask=mask_b, other=0)
        tw = tl.load(TopW + offs_b * stride_twm + tk * stride_twtk, mask=mask_b, other=0.0)
        tw = tl.where(mask_b, tw, 0.0)
        ki = tl.where(mask_b, ki, 0)

        acc = tl.load(Bias + ki[:, None] * stride_bk + offs_d[None, :] * stride_bd, mask=mask, other=0.0)
        for p in range(P_NUM):
            w = tl.load(W + ki[:, None] * stride_wk + p * stride_wp + offs_d[None, :] * stride_wd, mask=mask, other=0.0)
            pm = tl.load(Perms + ki[:, None] * stride_pk + p * stride_pp + offs_d[None, :] * stride_pd, mask=mask, other=0)
            pm = tl.where((pm >= 0) & (pm < D_DIM), pm, 0)
            xv = tl.load(R + offs_b[:, None] * stride_rm + pm * stride_rd, mask=mask, other=0.0)
            acc += w * xv

        # relu6 derivative strict >/< : 0 outside (0,6), 1 inside; matches forward clamp
        mask_relu = (acc > 0.0) & (acc < 6.0)
        dp = go_val * tw[:, None] * tl.where(mask_relu, 1.0, 0.0)
        # D_PRIM store via block_ptr where possible (contiguous D)
        # D_PRIM is (B_ROWS, TOPK, D_DIM) — not 2D, keep manual for 3D, but note contiguous D could use make_block_ptr with proper strides
        tl.store(D_PRIM + offs_b[:, None] * stride_dpm + tk * stride_dptk + offs_d[None, :] * stride_dpd, dp, mask=mask)

        if STORE_ACT:
            act = tl.minimum(tl.maximum(acc, 0.0), 6.0)
            tl.store(ACT + offs_b[:, None] * stride_actm + tk * stride_acttk + offs_d[None, :] * stride_actd, act, mask=mask)


@triton.jit
def _tree_perm_bwd_dx_kernel(
    D_PRIM, W, InvPerms, TopIdx, GR,
    stride_dpm, stride_dptk, stride_dpd,
    stride_wk, stride_wp, stride_wd,
    stride_pk, stride_pp, stride_pd,
    stride_tbm, stride_tbtk,
    stride_grm, stride_grd,
    B_ROWS: tl.constexpr, D_DIM: tl.constexpr, P_NUM: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_b = offs_b < B_ROWS
    mask_d = offs_d < D_DIM
    mask = mask_b[:, None] & mask_d[None, :]
    acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    for tk in range(TOPK):
        ki = tl.load(TopIdx + offs_b * stride_tbm + tk * stride_tbtk, mask=mask_b, other=0)
        ki = tl.where(mask_b, ki, 0)
        for p in range(P_NUM):
            inv_ptrs = InvPerms + ki[:, None] * stride_pk + p * stride_pp + offs_d[None, :] * stride_pd
            src = tl.load(inv_ptrs, mask=mask, other=0)
            src = tl.where((src >= 0) & (src < D_DIM), src, 0)
            dp_ptrs = D_PRIM + offs_b[:, None] * stride_dpm + tk * stride_dptk + src * stride_dpd
            dp = tl.load(dp_ptrs, mask=mask, other=0.0)
            w_ptrs = W + ki[:, None] * stride_wk + p * stride_wp + src * stride_wd
            w = tl.load(w_ptrs, mask=mask, other=0.0)
            acc += dp * w
    # GR store via block_ptr for coalesced contiguous D (stride_grd==1)
    gr_block_ptr = tl.make_block_ptr(
        base=GR,
        shape=(B_ROWS, D_DIM),
        strides=(stride_grm, stride_grd),
        offsets=(pid_b * BLOCK_B, pid_d * BLOCK_D),
        block_shape=(BLOCK_B, BLOCK_D),
        order=(1, 0),
    )
    tl.store(gr_block_ptr, acc, boundary_check=(0, 1))
    # fallback manual: tl.store(GR + offs_b[:, None] * stride_grm + offs_d[None, :] * stride_grd, acc, mask=mask)


@triton.jit
def _tree_perm_bwd_gw_kernel(
    D_PRIM, R, Perms, TopIdx, GW,
    stride_dpm, stride_dptk, stride_dpd,
    stride_rm, stride_rd,
    stride_pk, stride_pp, stride_pd,
    stride_tbm, stride_tbtk,
    stride_gwk, stride_gwp, stride_gwd,
    B_ROWS: tl.constexpr, D_DIM: tl.constexpr, P_NUM: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_DIM
    p_idx = tl.load(Perms + pid_k * stride_pk + pid_p * stride_pp + offs_d * stride_pd, mask=mask_d, other=0)
    p_idx = tl.where((p_idx >= 0) & (p_idx < D_DIM), p_idx, 0)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for b_start in range(0, B_ROWS, BLOCK_B):
        offs_b = b_start + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B_ROWS
        # R gather via perm — contiguous R row when block_ptr with stride_rd==1, but perm gather manual
        xv_ptrs = R + offs_b[:, None] * stride_rm + p_idx[None, :] * stride_rd
        xv = tl.load(xv_ptrs, mask=mask_b[:, None] & mask_d[None, :], other=0.0)
        for tk in range(TOPK):
            ki = tl.load(TopIdx + offs_b * stride_tbm + tk * stride_tbtk, mask=mask_b, other=0)
            ki = tl.where(mask_b, ki, 0)
            dp_ptrs = D_PRIM + offs_b[:, None] * stride_dpm + tk * stride_dptk + offs_d[None, :] * stride_dpd
            dp = tl.load(dp_ptrs, mask=mask_b[:, None] & mask_d[None, :], other=0.0)
            need = ki == pid_k
            dp = tl.where(need[:, None], dp, 0.0)
            grad = dp * xv
            acc += tl.sum(grad, axis=0)
    # GW is 3D [K,P,D] not 2D block_ptr friendly due to K*P striding; keep manual store
    # Contiguous D case (stride_gwd==1) would be coalesced via make_block_ptr if reshaped, but fallback manual kept for simplicity
    gw_ptrs = GW + pid_k * stride_gwk + pid_p * stride_gwp + offs_d * stride_gwd
    tl.store(gw_ptrs, acc, mask=mask_d)


class TritonTreePermFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_in, w_perm, bias, perms, top_idx, top_w):
        ctx.save_for_backward(r_in, w_perm, bias, perms, top_idx, top_w)
        orig_shape = r_in.shape
        D = orig_shape[-1]
        r_f = r_in.reshape(-1, D).contiguous().float()
        top_idx_f = top_idx.reshape(-1, top_idx.shape[-1]).contiguous().to(torch.int32)
        top_w_f = top_w.reshape(-1, top_w.shape[-1]).contiguous().float()
        out = triton_tree_perm_fwd(
            r_f,
            w_perm.detach().float().contiguous(),
            bias.detach().float().contiguous(),
            perms.detach().to(torch.int32).contiguous(),
            top_idx_f,
            top_w_f,
        )
        return out.to(r_in.dtype).reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not any(ctx.needs_input_grad):
            return None, None, None, None, None, None

        r_in, w_perm, bias, perms, top_idx, top_w = ctx.saved_tensors
        orig_shape = r_in.shape
        D = orig_shape[-1]
        r_flat = r_in.reshape(-1, D).contiguous().float()
        B_flat = r_flat.shape[0]
        K, P, _ = w_perm.shape
        Tk = top_idx.shape[-1]
        top_idx_flat = top_idx.reshape(B_flat, Tk).contiguous().to(torch.int32)
        top_w_flat = top_w.reshape(B_flat, Tk).contiguous().float()
        go_flat = grad_output.reshape(B_flat, D).contiguous().float()

        w_perm_f = w_perm.detach().float().contiguous()
        bias_f = bias.detach().float().contiguous() if isinstance(bias, torch.Tensor) else torch.zeros((K, D), device=r_in.device, dtype=torch.float32)
        perms_f = perms.detach().to(torch.int32).contiguous()

        # d_prim reused by gr/gw/gb; fusing would 3x leaf-prim recompute (P*D gathers) for ~B*T*D memory saved
        d_prim = torch.empty((B_flat, Tk, D), device=r_flat.device, dtype=torch.float32)
        store_act = ctx.needs_input_grad[5]
        act = torch.empty((B_flat, Tk, D), device=r_flat.device, dtype=torch.float32) if store_act else torch.empty(0, device=r_flat.device, dtype=torch.float32)

        BM, BD = 16, 32
        # Turing: ensure BLOCK <=64 (64KB SMEM)
        if _is_turing():
            BM = min(BM, 64)
            BD = min(BD, 64)
        grid = ((B_flat + BM - 1) // BM, (D + BD - 1) // BD)

        _tree_perm_bwd_dprim_kernel[grid](
            r_flat, w_perm_f, bias_f, perms_f, top_idx_flat, top_w_flat, go_flat,
            d_prim, act,
            r_flat.stride(0), r_flat.stride(1),
            w_perm_f.stride(0), w_perm_f.stride(1), w_perm_f.stride(2),
            bias_f.stride(0), bias_f.stride(1),
            perms_f.stride(0), perms_f.stride(1), perms_f.stride(2),
            top_idx_flat.stride(0), top_idx_flat.stride(1),
            top_w_flat.stride(0), top_w_flat.stride(1),
            go_flat.stride(0), go_flat.stride(1),
            d_prim.stride(0), d_prim.stride(1), d_prim.stride(2),
            act.stride(0) if store_act else 0, act.stride(1) if store_act else 0, act.stride(2) if store_act else 0,
            B_flat, D, P, Tk,
            STORE_ACT=store_act,
            BLOCK_B=BM, BLOCK_D=BD, num_warps=4,
        )

        gt = None
        if store_act:
            gt_flat = (act * go_flat.unsqueeze(1)).sum(dim=-1)
            gt = gt_flat.to(top_w.dtype).reshape(top_w.shape)

        gb = None
        if ctx.needs_input_grad[2] and isinstance(bias, torch.Tensor):
            flat_k = top_idx_flat.reshape(-1).long()
            # Duplicate k in topk: index_add_ sums correctly; order nondeterministic but sum deterministic
            if __debug__ and flat_k.numel() > 0:
                # Optional debug assert for duplicate detection (no-op if unique)
                pass  # duplicates sum; to enforce uniqueness: assert flat_k.unique().numel() == flat_k.numel()
            gb = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device)
            gb.index_add_(0, flat_k, d_prim.reshape(-1, D))
            gb = gb.to(bias.dtype)

        gw = None
        if ctx.needs_input_grad[1]:
            gw = torch.zeros(w_perm.shape, dtype=torch.float32, device=w_perm.device)
            BD_GW = 32
            BM_GW = 32
            if _is_turing():
                BD_GW = min(BD_GW, 64)
                BM_GW = min(BM_GW, 64)
            grid_gw = (K, P, (D + BD_GW - 1) // BD_GW)
            _tree_perm_bwd_gw_kernel[grid_gw](
                d_prim, r_flat, perms_f, top_idx_flat, gw,
                d_prim.stride(0), d_prim.stride(1), d_prim.stride(2),
                r_flat.stride(0), r_flat.stride(1),
                perms_f.stride(0), perms_f.stride(1), perms_f.stride(2),
                top_idx_flat.stride(0), top_idx_flat.stride(1),
                gw.stride(0), gw.stride(1), gw.stride(2),
                B_flat, D, P, Tk,
                BLOCK_B=BM_GW, BLOCK_D=BD_GW, num_warps=4,
            )
            gw = gw.to(w_perm.dtype)

        gr = None
        if ctx.needs_input_grad[0]:
            gr_flat = torch.zeros((B_flat, D), dtype=torch.float32, device=r_flat.device)
            inv_perms_f = torch.argsort(perms_f, dim=-1).to(torch.int32).contiguous()
            _tree_perm_bwd_dx_kernel[grid](
                d_prim, w_perm_f, inv_perms_f, top_idx_flat, gr_flat,
                d_prim.stride(0), d_prim.stride(1), d_prim.stride(2),
                w_perm_f.stride(0), w_perm_f.stride(1), w_perm_f.stride(2),
                inv_perms_f.stride(0), inv_perms_f.stride(1), inv_perms_f.stride(2),
                top_idx_flat.stride(0), top_idx_flat.stride(1),
                gr_flat.stride(0), gr_flat.stride(1),
                B_flat, D, P, Tk,
                BLOCK_B=BM, BLOCK_D=BD, num_warps=4,
            )
            gr = gr_flat.to(r_in.dtype).reshape(orig_shape)

        return gr, gw, gb, None, None, gt


def triton_tree_perm(r_in, w_perm, bias, perms, top_idx, top_w):
    return TritonTreePermFunction.apply(r_in, w_perm, bias, perms, top_idx, top_w)
