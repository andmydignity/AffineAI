import torch
import pytest
from affine_ai.core.bitlinear import BitLinear, TernaryBitLinearSwiGLU
from affine_ai.core.associative import MonarchPermutationChain, FusedMonarchChain, NativeASDAGAssociativeMixer
from affine_ai.models.language_model import ASDAGLanguageModel


def test_bitlinear_ste_quantization():
    torch.manual_seed(42)
    layer = BitLinear(in_features=32, out_features=64, bias=True)
    x = torch.randn(4, 16, 32, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 16, 64)
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_ternary_swiglu_channel_mixer():
    torch.manual_seed(42)
    mixer = TernaryBitLinearSwiGLU(dim=32, expand=2)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    out = mixer(x)
    assert out.shape == (2, 8, 32)
    loss = out.sum()
    loss.backward()
    assert mixer.w_gate_val.weight.grad is not None
    assert mixer.w_down.weight.grad is not None


def test_monarch_permutation_chain():
    torch.manual_seed(42)
    monarch = MonarchPermutationChain(dim=32, num_stages=4)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    out = monarch(x)
    assert out.shape == (2, 8, 32)
    loss = out.sum()
    loss.backward()
    assert monarch.diagonals.grad is not None
    assert monarch.bias.grad is not None


def test_fused_monarch_chain():
    torch.manual_seed(42)
    fused = FusedMonarchChain(dim=32, num_branches=4, num_stages=4)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    q, k, v, g = fused(x)
    assert q.shape == (2, 8, 32)
    assert k.shape == (2, 8, 32)
    assert v.shape == (2, 8, 32)
    assert g.shape == (2, 8, 32)


def test_full_option_b_language_model():
    torch.manual_seed(42)
    model = ASDAGLanguageModel(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        channel_mixer_type="ternary_swiglu"
    )
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)

    # Test O(1) step generation (deterministic, no EOS early stop)
    gen = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5, temperature=0.0, eos_byte=None)
    assert gen.shape == (1, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M", [1, 16, 64, 128])
def test_triton_swiglu_cuda_parity_and_backward(M):
    from affine_ai.kernels.triton_bitlinear import triton_bitlinear_swiglu

    torch.manual_seed(42)
    K, N = 64, 128
    dtype = torch.float32  # Test mathematical parity accurately
    x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
    w_gv = torch.randn(2 * N, K, device="cuda", dtype=dtype, requires_grad=True)
    w_d = torch.randn(K, N, device="cuda", dtype=dtype, requires_grad=True)

    out = triton_bitlinear_swiglu(x, w_gv, w_d)
    assert out.shape == (M, K)

    # PyTorch reference with ternary quantization
    gamma_gv = w_gv.abs().mean().clamp(min=1e-5)
    gamma_d = w_d.abs().mean().clamp(min=1e-5)
    w_gv_q = torch.round(torch.clamp(w_gv / gamma_gv, -1.0, 1.0))
    w_d_q = torch.round(torch.clamp(w_d / gamma_d, -1.0, 1.0))

    gv = torch.matmul(x, w_gv_q.t()) * gamma_gv
    g = gv[:, :N]
    v = gv[:, N:]
    h_act = (g.sigmoid() * g) * v
    ref = torch.matmul(h_act, w_d_q.t()) * gamma_d

    assert torch.allclose(out, ref, atol=0.5, rtol=1e-3), f"SwiGLU M={M} parity mismatch: max diff={(out - ref).abs().max().item()}"

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert w_gv.grad is not None
    assert w_d.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(w_gv.grad).any()
    assert not torch.isnan(w_d.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_fused_add_rms_norm():
    from affine_ai.kernels.triton_rms_norm import triton_fused_add_rms_norm

    torch.manual_seed(42)
    B, T, D = 4, 32, 64
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    res = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = torch.randn(D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y_norm, res_out = triton_fused_add_rms_norm(x, res, scale, eps=1e-6)

    assert y_norm.shape == (B, T, D)
    assert res_out.shape == (B, T, D)
    assert torch.allclose(res_out, x + res, atol=1e-3)

    loss = y_norm.sum() + res_out.sum()
    loss.backward()

    assert x.grad is not None
    assert res.grad is not None
    assert scale.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(res.grad).any()
    assert not torch.isnan(scale.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_2bit_ternary_packing_and_unpacking():
    from affine_ai.kernels.triton_ternary import triton_pack_ternary_2bit, triton_unpack_ternary_2bit

    torch.manual_seed(42)
    Rows, Cols = 32, 64
    w_raw = torch.randint(-1, 2, (Rows, Cols), device="cuda", dtype=torch.float32)

    packed = triton_pack_ternary_2bit(w_raw)
    assert packed.shape == (Rows, Cols // 16)
    assert packed.dtype == torch.int32

    unpacked = triton_unpack_ternary_2bit(packed, (Rows, Cols), dtype=torch.float32)
    assert unpacked.shape == (Rows, Cols)
    assert torch.equal(unpacked, w_raw)



