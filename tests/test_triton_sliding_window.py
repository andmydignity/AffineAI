import math
import pytest
import torch

from affine_ai.kernels.triton_sliding_window import sliding_window_attn, _eager_swa


def test_triton_sliding_window_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")

    B, T, D, H, W = 2, 256, 64, 2, 65
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, T, D, device="cuda", dtype=dtype)

        eager = _eager_swa(q, k, v, W, sink=True)
        triton_out = sliding_window_attn(q, k, v, window=W, sink=True)
        diff = (triton_out.float() - eager.float()).abs().max().item()
        tol = 1e-3 if dtype == torch.float16 else 2e-3
        assert diff < tol, f"parity failed dtype={dtype} diff={diff}"

        # Sink token identity at t=0
        sink_diff = (triton_out[:, :, 0, :].float() - v[:, :, 0, :].float()).abs().max().item()
        assert sink_diff < 1e-3, f"sink identity failed dtype={dtype} diff={sink_diff}"


def test_triton_sliding_window_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")

    B, T, D, H, W = 2, 64, 32, 2, 17
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(42)
        q0 = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
        k0 = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
        v0 = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)

        q1 = q0.detach().clone().requires_grad_(True)
        k1 = k0.detach().clone().requires_grad_(True)
        v1 = v0.detach().clone().requires_grad_(True)

        out_tri = sliding_window_attn(q0, k0, v0, window=W, sink=True)
        out_tri.sum().backward()

        out_ref = _eager_swa(q1, k1, v1, window=W, sink=True)
        out_ref.sum().backward()

        for name, gt, ge in [("dq", q0.grad.float(), q1.grad.float()), ("dk", k0.grad.float(), k1.grad.float()), ("dv", v0.grad.float(), v1.grad.float())]:
            diff = (gt - ge).abs().max().item()
            tol = 5e-3 if dtype == torch.float16 else 4e-2
            assert diff < tol, f"backward grad {name} mismatch dtype={dtype} diff={diff}"


def test_triton_sliding_window_large_context_grid_overflow():
    """Verify context T > 65535 does not overflow gridDim.y by mapping T to dim 0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")

    # Sequence length > 65535 (CUDA gridDim.y limit)
    T = 65536 + 128
    B, H, D, W = 1, 1, 32, 32
    dtype = torch.float16

    # Test forward with small footprint
    q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, T, D, device="cuda", dtype=dtype)

    out = sliding_window_attn(q, k, v, window=W, sink=True)
    assert out.shape == (B, H, T, D)
    assert not torch.isnan(out[:1, :1, :10]).any()
    assert not torch.isnan(out[:1, :1, -10:]).any()


def test_triton_sliding_window_d_gt_128():
    """Verify D = 256 (>128) executes through Triton without error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not available")

    B, H, T, D, W = 1, 2, 128, 256, 32
    dtype = torch.float16
    q = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True)

    out = sliding_window_attn(q, k, v, window=W, sink=True)
    assert out.shape == (B, H, T, D)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
