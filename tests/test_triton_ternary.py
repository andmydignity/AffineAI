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
