import math
import torch
import pytest


def _eager_ref(q, k, v, window, sink=True):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    device = q.device
    qi = torch.arange(T, device=device)
    kj = torch.arange(T, device=device)
    if sink:
        lo = torch.clamp(qi - window + 1, min=1).unsqueeze(1)
        window_mask = (kj.unsqueeze(0) >= lo) & (kj.unsqueeze(0) <= qi.unsqueeze(1))
        sink_mask = (kj.unsqueeze(0) == 0).expand(T, T)
        mask = window_mask | sink_mask
    else:
        mask = (kj.unsqueeze(0) <= qi.unsqueeze(1)) & (kj.unsqueeze(0) > (qi - window).unsqueeze(1))
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=scale)
    except Exception:
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=scale)
        except Exception:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=scale)


def test_triton_swa_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")
    from affine_ai.kernels.triton_sliding_window import sliding_window_attn

    B, T, D, H, W = 2, 512, 64, 4, 129
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, T, D, device="cuda", dtype=dtype)

        eager = _eager_ref(q, k, v, W, sink=True)
        triton_out = sliding_window_attn(q, k, v, window=W, sink=True)
        diff = (triton_out.float() - eager.float()).abs().max().item()
        tol = 1e-3 if dtype == torch.float16 else 5e-3
        assert diff < tol, f"parity failed dtype={dtype} diff={diff}"

        # sink-identity: position 0 output equals v[0]
        sink_diff = (triton_out[:, :, 0, :].float() - v[:, :, 0, :].float()).abs().max().item()
        assert sink_diff < 1e-3, f"sink identity failed dtype={dtype} diff={sink_diff}"

        # non-divisible T tail
        T2 = 511
        q2 = torch.randn(B, H, T2, D, device="cuda", dtype=dtype)
        k2 = torch.randn(B, H, T2, D, device="cuda", dtype=dtype)
        v2 = torch.randn(B, H, T2, D, device="cuda", dtype=dtype)
        eager2 = _eager_ref(q2, k2, v2, W, sink=True)
        triton2 = sliding_window_attn(q2, k2, v2, window=W, sink=True)
        diff2 = (triton2.float() - eager2.float()).abs().max().item()
        assert diff2 < tol, f"tail parity failed dtype={dtype} diff={diff2}"


def test_triton_swa_grads():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")
    from affine_ai.kernels.triton_sliding_window import sliding_window_attn

    B, T, D, H, W = 2, 64, 32, 2, 17
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(1)
        q0 = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
        k0 = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
        v0 = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
        q1 = q0.detach().clone().requires_grad_(True)
        k1 = k0.detach().clone().requires_grad_(True)
        v1 = v0.detach().clone().requires_grad_(True)

        out_triton = sliding_window_attn(q0, k0, v0, window=W, sink=True)
        loss_t = out_triton.sum()
        loss_t.backward()
        gq_t, gk_t, gv_t = q0.grad.float(), k0.grad.float(), v0.grad.float()

        out_eager = _eager_ref(q1, k1, v1, W, sink=True)
        loss_e = out_eager.sum()
        loss_e.backward()
        gq_e, gk_e, gv_e = q1.grad.float(), k1.grad.float(), v1.grad.float()

        for name, gt, ge in [("dq", gq_t, gq_e), ("dk", gk_t, gk_e), ("dv", gv_t, gv_e)]:
            diff = (gt - ge).abs().max().item()
            tol = 5e-3 if dtype == torch.float16 else 4e-2
            assert diff < tol, f"grad {name} mismatch dtype={dtype} diff={diff}"


def test_swa_mixer_triton_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")
    from affine_ai.core.swa import SlidingWindowAttentionMixer

    B, T, d_model, n_heads, W = 2, 128, 256, 4, 32
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(2)
        mixer_triton = SlidingWindowAttentionMixer(d_model=d_model, n_heads=n_heads, window=W, dtype=dtype, use_triton=True).cuda().to(dtype)
        mixer_eager = SlidingWindowAttentionMixer(d_model=d_model, n_heads=n_heads, window=W, dtype=dtype, use_triton=False).cuda().to(dtype)
        # copy weights
        mixer_eager.load_state_dict(mixer_triton.state_dict())
        x = torch.randn(B, T, d_model, device="cuda", dtype=dtype)
        with torch.no_grad():
            out_t, _ = mixer_triton(x)
            out_e, _ = mixer_eager(x)
        diff = (out_t.float() - out_e.float()).abs()
        # ANCHORED METRIC (do not "fix" by raising tolerances): attention-level
        # parity is covered by test_triton_swa_parity; this test checks wiring
        # (GQA repeat, transpose/reshape, scale). Wiring bugs produce O(1e-2+)
        # mean errors; Triton-vs-SDPA rounding through out_proj produces
        # ~1e-4 mean with single-bf16-ULP max outliers (eps=2^-8 at |x|~1, and
        # max-abs grows with element count). So assert on the MEAN, with a
        # wide max-abs guard band that can never need raising.
        assert diff.mean().item() < 5e-4, f"mixer wiring diff mean={diff.mean().item()} dtype={dtype}"
        assert diff.max().item() < 1e-2, f"mixer parity diff={diff.max().item()} dtype={dtype}"
