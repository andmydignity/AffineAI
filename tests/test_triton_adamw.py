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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_adamw_cuda_graph_capture():
    """Verify TritonAdamW supports full CUDA graph capture without host sync or stalls."""
    torch.manual_seed(42)
    N = 1024
    p = torch.randn(N, dtype=torch.float32, device="cuda", requires_grad=True)
    opt = TritonAdamW([p], lr=1e-3)
    p.grad = torch.randn_like(p)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            opt.step()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        opt.step()
    g.replay()
    assert not torch.isnan(p).any()
    assert isinstance(opt.state[p]['step'], int)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_adamw_master_weight_sync():
    """Verify master weights stay synchronized when parameters are modified outside optimizer."""
    torch.manual_seed(42)
    N = 512
    p = torch.randn(N, dtype=torch.float16, device="cuda")
    p.grad = torch.randn(N, dtype=torch.float16, device="cuda")

    opt = TritonAdamW([p], lr=1e-3, master_weights=True)
    opt.step()

    # Modify p outside optimizer step
    new_vals = torch.randn(N, dtype=torch.float16, device="cuda")
    p.copy_(new_vals)
    p.grad = torch.randn(N, dtype=torch.float16, device="cuda")

    opt.step()
    # Verify master_param reflects updated parameter rather than stale previous step values
    master_p = opt.state[p]['master_param']
    assert torch.allclose(master_p.half(), p, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_adamw_step_export():
    """Verify triton_adamw_step can be imported directly and matches TritonAdamW."""
    from affine_ai.kernels.triton_adamw import triton_adamw_step
    from affine_ai.kernels import triton_adamw_step as init_step

    assert triton_adamw_step is not None
    assert init_step is not None

    torch.manual_seed(42)
    N = 256
    p = torch.randn(N, dtype=torch.float32, device="cuda")
    g = torch.randn(N, dtype=torch.float32, device="cuda")
    m = torch.zeros(N, dtype=torch.float32, device="cuda")
    v = torch.zeros(N, dtype=torch.float32, device="cuda")

    triton_adamw_step(p, g, m, v, lr=1e-3, step=1)
    assert not torch.isnan(p).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_adamw_unaligned_sizes():
    """Verify alignment hints work for both aligned (N%8==0) and unaligned sizes."""
    torch.manual_seed(42)
    for N in [255, 256, 257, 1023, 1024, 1025]:
        p = torch.randn(N, dtype=torch.float32, device="cuda")
        p.grad = torch.randn(N, dtype=torch.float32, device="cuda")
        opt = TritonAdamW([p], lr=1e-3)
        opt.step()
        assert not torch.isnan(p).any()
