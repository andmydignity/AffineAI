import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_ternary_linear_fwd_2d_and_3d():
    from affine_ai.kernels.triton_ternary import triton_ternary_linear_fwd

    device = torch.device("cuda")
    torch.manual_seed(42)

    # 2D test
    M, K, N = 64, 128, 96
    x2d = torch.randn(M, K, device=device)
    w_tern = torch.randint(-1, 2, (N, K), device=device).float()
    gamma = 0.5
    bias = torch.randn(N, device=device)

    out2d = triton_ternary_linear_fwd(x2d, w_tern, gamma, bias=bias)
    assert out2d.shape == (M, N)
    assert not torch.isnan(out2d).any()

    # 3D test
    B, T = 4, 16
    x3d = torch.randn(B, T, K, device=device)
    out3d = triton_ternary_linear_fwd(x3d, w_tern, gamma, bias=bias)
    assert out3d.shape == (B, T, N)
    assert not torch.isnan(out3d).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_fp32_linear_2d_and_3d():
    from affine_ai.kernels.triton_ternary import triton_fp32_linear

    device = torch.device("cuda")
    torch.manual_seed(42)

    # 2D
    M, K, N = 32, 64, 48
    a2d = torch.randn(M, K, device=device)
    b = torch.randn(N, K, device=device)
    bias = torch.randn(N, device=device)

    out2d = triton_fp32_linear(a2d, b, bias)
    assert out2d.shape == (M, N)
    ref2d = F.linear(a2d, b, bias)
    assert torch.allclose(out2d, ref2d, atol=1e-3)

    # 3D
    B, T = 2, 8
    a3d = torch.randn(B, T, K, device=device)
    out3d = triton_fp32_linear(a3d, b, bias)
    assert out3d.shape == (B, T, N)
    ref3d = F.linear(a3d, b, bias)
    assert torch.allclose(out3d, ref3d, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_ternary_linear_autograd_2d_and_3d():
    from affine_ai.kernels.triton_ternary import triton_ternary_linear

    device = torch.device("cuda")
    torch.manual_seed(42)

    # 3D autograd with bias
    B, T, K, N = 2, 16, 64, 32
    x = torch.randn(B, T, K, device=device, requires_grad=True)
    w = torch.randn(N, K, device=device, requires_grad=True)
    bias = torch.randn(N, device=device, requires_grad=True)

    out = triton_ternary_linear(x, w, bias)
    assert out.shape == (B, T, N)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()

    assert w.grad is not None
    assert w.grad.shape == w.shape
    assert not torch.isnan(w.grad).any()

    assert bias.grad is not None
    assert bias.grad.shape == bias.shape
    assert not torch.isnan(bias.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_ternary_twin_autograd():
    from affine_ai.kernels.triton_ternary import triton_ternary_twin

    device = torch.device("cuda")
    torch.manual_seed(42)

    B, T, K, O = 2, 8, 64, 32
    x = torch.randn(B, T, K, device=device, requires_grad=True)
    w1 = torch.randn(O, K, device=device, requires_grad=True)
    b1 = torch.randn(O, device=device, requires_grad=True)
    w2 = torch.randn(O, K, device=device, requires_grad=True)
    b2 = torch.randn(O, device=device, requires_grad=True)

    out = triton_ternary_twin(x, w1, b1, w2, b2)
    assert out.shape == (B, T, 2 * O)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert w1.grad is not None
    assert b1.grad is not None
    assert w2.grad is not None
    assert b2.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(w1.grad).any()
    assert not torch.isnan(w2.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_2bit_packing_roundtrip():
    from affine_ai.kernels.triton_ternary import triton_pack_ternary_2bit, triton_unpack_ternary_2bit

    device = torch.device("cuda")
    torch.manual_seed(42)

    Rows, Cols = 32, 64
    w_raw = torch.randint(-1, 2, (Rows, Cols), device=device, dtype=torch.float32)

    packed = triton_pack_ternary_2bit(w_raw)
    assert packed.shape == (Rows, Cols // 16)
    assert packed.dtype == torch.int32

    unpacked = triton_unpack_ternary_2bit(packed, (Rows, Cols), dtype=torch.float32)
    assert unpacked.shape == (Rows, Cols)
    assert torch.equal(unpacked, w_raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_popc_unpadded_parity():
    from affine_ai.kernels.triton_popc import triton_pack_sign_bits, triton_popc_sign_similarity

    device = torch.device("cuda")
    torch.manual_seed(42)

    for D in [33, 45, 70, 95]:
        M, N = 64, 32
        x = torch.randn(M, D, device=device)
        w = torch.randn(N, D, device=device)

        x_sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        w_sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        ref_dot = torch.matmul(x_sign, w_sign.t()).to(torch.int32)

        x_bits = triton_pack_sign_bits(x)
        w_bits = triton_pack_sign_bits(w)
        out_dot = triton_popc_sign_similarity(x_bits, w_bits, D=D)

        assert out_dot.shape == (M, N)
        diff = (out_dot - ref_dot).abs().max().item()
        assert diff == 0, f"Unpadded POPC dot diff for D={D}: max diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_quantize_x_and_fast_gw():
    from affine_ai.kernels.triton_ternary import (
        triton_quantize_x,
        triton_row_amax,
        triton_ternary_linear_gw,
        _ternary_gw_kernel,
        _grid,
    )

    device = torch.device("cuda")
    torch.manual_seed(42)

    M, N, K = 128, 96, 64
    go = torch.randn(M, N, device=device)
    x = torch.randn(M, K, device=device)
    amax = triton_row_amax(x)

    # 1. Test pre-quantization
    xq = triton_quantize_x(x, amax)
    assert xq.shape == (M, K)
    assert not torch.isnan(xq).any()

    # 2. Test triton_ternary_linear_gw with x_q
    gw_prequant = triton_ternary_linear_gw(go, None, None, N, K, x_q=xq)

    # 3. Test triton_ternary_linear_gw without x_q (internally quantizes)
    gw_autoquant = triton_ternary_linear_gw(go, x, amax, N, K)
    assert torch.equal(gw_prequant, gw_autoquant)

    # 4. Compare against legacy _ternary_gw_kernel
    gw_legacy = torch.empty((N, K), device=device, dtype=torch.float32)
    _ternary_gw_kernel[_grid(N, K, 32, 64)](
        go, x, amax, gw_legacy,
        go.stride(0), go.stride(1), x.stride(0), x.stride(1),
        gw_legacy.stride(0), gw_legacy.stride(1),
        M, N, K, BLOCK_M=64, BLOCK_N=32, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )
    diff = (gw_prequant - gw_legacy).abs().max().item()
    assert diff == 0.0, f"Discrepancy between fast prequant gw and legacy gw: {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_ternary_twin_gradient_scaling_issue23():
    from affine_ai.kernels.triton_ternary import triton_ternary_twin

    device = torch.device("cuda")
    torch.manual_seed(42)

    M, K, O = 32, 64, 48
    x = torch.randn(M, K, device=device, requires_grad=True)
    w1 = torch.randn(O, K, device=device) * 2.5
    w2 = torch.randn(O, K, device=device) * 3.5

    out = triton_ternary_twin(x, w1, None, w2, None)
    go = torch.randn_like(out)
    out.backward(go)

    # Reference gradient scaling check:
    g1 = w1.abs().mean().clamp(min=1e-5)
    g2 = w2.abs().mean().clamp(min=1e-5)
    w1t = torch.round(w1 / g1).clamp(-1.0, 1.0)
    w2t = torch.round(w2 / g2).clamp(-1.0, 1.0)
    go1, go2 = go.split(O, dim=-1)
    ref_gx = torch.matmul(go1, w1t * g1) + torch.matmul(go2, w2t * g2)

    # Gradient must be scaled by g, NOT g^2
    diff = (x.grad - ref_gx).abs().max().item()
    assert diff < 1e-2, f"Twin gx scale mismatch vs reference: {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_ternary_bf16_tc_enabled_issue24():
    from affine_ai.kernels.triton_ternary import triton_ternary_linear, _resolve_tc

    device = torch.device("cuda")
    torch.manual_seed(42)

    x = torch.randn(16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)

    assert _resolve_tc(None, x) is True
    out = triton_ternary_linear(x, w)
    assert out.dtype == torch.bfloat16
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ternary tests")
def test_triton_row_amax_and_quantize_x_3d_issue25_26():
    from affine_ai.kernels.triton_ternary import triton_row_amax, triton_quantize_x

    device = torch.device("cuda")
    torch.manual_seed(42)

    B, T, K = 4, 16, 64
    x_cpu = torch.randn(B, T, K)
    x_cuda = x_cpu.to(device)

    # Issue 25: Harmonize 3D shapes between CPU and CUDA
    amax_cpu = triton_row_amax(x_cpu)
    amax_cuda = triton_row_amax(x_cuda)
    assert amax_cpu.shape == (B, T)
    assert amax_cuda.shape == (B, T)
    assert torch.allclose(amax_cpu, amax_cuda.cpu(), atol=1e-5)

    # Issue 26: 3D support in triton_quantize_x
    xq_cpu = triton_quantize_x(x_cpu, amax_cpu)
    xq_cuda = triton_quantize_x(x_cuda, amax_cuda)
    assert xq_cuda.shape == (B, T, K)
    assert torch.allclose(xq_cpu, xq_cuda.cpu(), atol=1e-4)


