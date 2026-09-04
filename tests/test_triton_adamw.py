import math
import pytest
import torch
from affine_ai.kernels.triton_adamw import TritonAdamW, _adamw_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_adamw_cuda_parity():
    """Verify TritonAdamW matches PyTorch AdamW on GPU with FP32."""
    torch.manual_seed(42)
    N = 2048

    p_ref = torch.randn(N, dtype=torch.float32, device="cuda")
    p_triton = p_ref.clone()

    g = torch.randn(N, dtype=torch.float32, device="cuda")
    p_ref.grad = g.clone()
    p_triton.grad = g.clone()

    opt_ref = torch.optim.AdamW([p_ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opt_triton = TritonAdamW([p_triton], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    for step in range(5):
        opt_ref.step()
        opt_triton.step()

        # Update gradients for next step
        g = torch.randn(N, dtype=torch.float32, device="cuda")
        p_ref.grad = g.clone()
        p_triton.grad = g.clone()

    assert torch.allclose(p_triton, p_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("correct_bias", [True, False])
def test_triton_adamw_dtypes_and_bias(dtype, correct_bias):
    """Verify TritonAdamW supports multiple floating point dtypes and correct_bias settings."""
    torch.manual_seed(123)
    N = 1024

    p = torch.randn(N, dtype=dtype, device="cuda")
    p.grad = torch.randn(N, dtype=dtype, device="cuda")

    opt = TritonAdamW([p], lr=1e-3, correct_bias=correct_bias, weight_decay=0.01)
    opt.step()

    assert p.dtype == dtype
    assert not torch.isnan(p).any()
    assert not torch.isinf(p).any()


def test_triton_adamw_cpu_fallback():
    """Verify CPU fallback path for TritonAdamW."""
    torch.manual_seed(42)
    N = 512

    p_ref = torch.randn(N, dtype=torch.float32)
    p_test = p_ref.clone()

    g = torch.randn(N, dtype=torch.float32)
    p_ref.grad = g.clone()
    p_test.grad = g.clone()

    opt_ref = torch.optim.AdamW([p_ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opt_test = TritonAdamW([p_test], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    opt_ref.step()
    opt_test.step()

    assert torch.allclose(p_test, p_ref, atol=1e-6)
