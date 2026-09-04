import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels.triton_rms_norm import triton_rms_norm, triton_fused_add_rms_norm


def ref_rms_norm(x, scale, eps=1e-6):
    x_float = x.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(var + eps)
    return (x_float * rsqrt * scale.float()).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_rms_norm_fwd_bwd():
    torch.manual_seed(42)
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    scale = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True)

    # Test different eps values to verify eps: tl.float32 scalar arg works without recompilation issue
    for eps in [1e-5, 1e-6, 1e-4]:
        out = triton_rms_norm(x, scale, eps=eps)
        ref_out = ref_rms_norm(x, scale, eps=eps)
        assert torch.allclose(out, ref_out, atol=1e-4)

        loss = out.sum()
        loss.backward(retain_graph=True)
        assert x.grad is not None and scale.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(scale.grad).any()
        x.grad.zero_()
        scale.grad.zero_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_fused_add_rms_norm_parity():
    torch.manual_seed(42)
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    res = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    scale = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True)

    y, res_out = triton_fused_add_rms_norm(x, res, scale, eps=1e-6)
    ref_res = x + res
    ref_y = ref_rms_norm(ref_res, scale, eps=1e-6)

    assert torch.allclose(res_out, ref_res, atol=1e-4)
    assert torch.allclose(y, ref_y, atol=1e-4)

    (y.sum() + res_out.sum()).backward()

    ref_x = x.detach().clone().requires_grad_(True)
    ref_r = res.detach().clone().requires_grad_(True)
    ref_s = scale.detach().clone().requires_grad_(True)
    ref_res_val = ref_x + ref_r
    ref_y_val = ref_rms_norm(ref_res_val, ref_s, eps=1e-6)
    (ref_y_val.sum() + ref_res_val.sum()).backward()

    assert torch.allclose(x.grad, ref_x.grad, atol=1e-4)
    assert torch.allclose(res.grad, ref_r.grad, atol=1e-4)
    assert torch.allclose(scale.grad, ref_s.grad, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_fused_add_rms_norm_no_storage_aliasing():
    """Verify dx_out and dres_out_val do not share storage when both require grad."""
    torch.manual_seed(42)
    B, T, D = 2, 8, 32
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    res = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    scale = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True)

    y, res_out = triton_fused_add_rms_norm(x, res, scale, eps=1e-6)
    (y.sum() + res_out.sum()).backward()

    # Pointers must differ to prevent in-place corruption
    assert x.grad.data_ptr() != res.grad.data_ptr()

    # Modifying x.grad in-place should not affect res.grad
    res_grad_copy = res.grad.clone()
    x.grad.zero_()
    assert torch.equal(res.grad, res_grad_copy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_fused_add_rms_norm_fp16_overflow():
    """Ensure cast to float32 before x + res prevents half-precision overflow (> 65504)."""
    torch.manual_seed(42)
    B, T, D = 1, 4, 32
    # 40000 + 40000 = 80000, which overflows FP16 (max 65504)
    x = torch.full((B, T, D), 40000.0, device="cuda", dtype=torch.float16)
    res = torch.full((B, T, D), 40000.0, device="cuda", dtype=torch.float16)
    scale = torch.ones(D, device="cuda", dtype=torch.float16)

    # In fused kernel, res_acc is computed in float32 before rsqrt
    y, res_out = triton_fused_add_rms_norm(x, res, scale, eps=1e-6)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
