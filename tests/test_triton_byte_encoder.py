import pytest
import torch
import torch.nn.functional as F

from affine_ai.kernels.triton_byte_encoder import triton_fused_byte_encoder


def ref_byte_encoder(byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, bp_b):
    B, T = byte_ids.shape
    d_byte = embed_w.shape[1]
    K = conv_w.shape[-1]

    x = F.embedding(byte_ids.to(torch.int64), embed_w)
    x_pad = F.pad(x.transpose(1, 2), (K - 1, 0))
    x_conv = F.conv1d(x_pad, conv_w, conv_b, groups=d_byte).transpose(1, 2)

    x_res = x + x_conv
    rms = torch.rsqrt(x_res.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5).to(x_res.dtype)
    h = (x_res * rms * norm_scale).to(proj_w.dtype)

    u = torch.mm(h.reshape(B * T, d_byte), proj_w.t()).reshape(B, T, d_byte)
    sig = torch.sigmoid(u)
    h_byte = (u * sig).to(bp_w.dtype)

    b_logits = torch.mm(h_byte.reshape(B * T, d_byte), bp_w.t()).reshape(B, T)
    if bp_b is not None:
        b_logits = b_logits + bp_b

    return h_byte, b_logits


@pytest.mark.parametrize("device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
def test_triton_byte_encoder_parity(device):
    torch.manual_seed(42)
    B, T, d_byte, K = 2, 8, 16, 3
    vocab_size = 256

    byte_ids = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.int64)
    embed_w = torch.randn(vocab_size, d_byte, device=device, dtype=torch.float64, requires_grad=True)
    conv_w = torch.randn(d_byte, 1, K, device=device, dtype=torch.float64, requires_grad=True)
    conv_b = torch.randn(d_byte, device=device, dtype=torch.float64, requires_grad=True)
    norm_scale = torch.randn(d_byte, device=device, dtype=torch.float64, requires_grad=True)
    proj_w = torch.randn(d_byte, d_byte, device=device, dtype=torch.float64, requires_grad=True)
    bp_w = torch.randn(1, d_byte, device=device, dtype=torch.float64, requires_grad=True)
    bp_b = torch.randn(1, device=device, dtype=torch.float64, requires_grad=True)

    ref_embed_w = embed_w.detach().clone().requires_grad_(True)
    ref_conv_w = conv_w.detach().clone().requires_grad_(True)
    ref_conv_b = conv_b.detach().clone().requires_grad_(True)
    ref_norm_scale = norm_scale.detach().clone().requires_grad_(True)
    ref_proj_w = proj_w.detach().clone().requires_grad_(True)
    ref_bp_w = bp_w.detach().clone().requires_grad_(True)
    ref_bp_b = bp_b.detach().clone().requires_grad_(True)

    h_byte, b_logits = triton_fused_byte_encoder(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, bp_b
    )

    ref_h, ref_b = ref_byte_encoder(
        byte_ids, ref_embed_w, ref_conv_w, ref_conv_b, ref_norm_scale, ref_proj_w, ref_bp_w, ref_bp_b
    )

    # Assert forward mathematical parity
    assert torch.allclose(h_byte, ref_h, atol=1e-5), f"h_byte max diff: {(h_byte - ref_h).abs().max().item()}"
    assert torch.allclose(b_logits, ref_b, atol=1e-5), f"b_logits max diff: {(b_logits - ref_b).abs().max().item()}"

    # Assert backward mathematical parity
    loss1 = (h_byte * 1.5).sum() + (b_logits * 2.0).sum()
    loss1.backward()

    loss2 = (ref_h * 1.5).sum() + (ref_b * 2.0).sum()
    loss2.backward()

    assert torch.allclose(embed_w.grad, ref_embed_w.grad, atol=1e-5), "embed_w.grad mismatch"
    assert torch.allclose(conv_w.grad, ref_conv_w.grad, atol=1e-5), "conv_w.grad mismatch"
    assert torch.allclose(conv_b.grad, ref_conv_b.grad, atol=1e-5), "conv_b.grad mismatch"
    assert torch.allclose(norm_scale.grad, ref_norm_scale.grad, atol=1e-5), "norm_scale.grad mismatch"
    assert torch.allclose(proj_w.grad, ref_proj_w.grad, atol=1e-5), "proj_w.grad mismatch"
    assert torch.allclose(bp_w.grad, ref_bp_w.grad, atol=1e-5), "bp_w.grad mismatch"
    assert torch.allclose(bp_b.grad, ref_bp_b.grad, atol=1e-5), "bp_b.grad mismatch"


@pytest.mark.parametrize("device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
def test_triton_byte_encoder_int32_ids(device):
    """Verify backward scatter_add succeeds even when byte_ids is torch.int32."""
    torch.manual_seed(42)
    B, T, d_byte, K = 2, 8, 16, 3
    vocab_size = 256

    byte_ids = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.int32)
    embed_w = torch.randn(vocab_size, d_byte, device=device, dtype=torch.float32, requires_grad=True)
    conv_w = torch.randn(d_byte, 1, K, device=device, dtype=torch.float32, requires_grad=True)
    conv_b = torch.randn(d_byte, device=device, dtype=torch.float32, requires_grad=True)
    norm_scale = torch.randn(d_byte, device=device, dtype=torch.float32, requires_grad=True)
    proj_w = torch.randn(d_byte, d_byte, device=device, dtype=torch.float32, requires_grad=True)
    bp_w = torch.randn(1, d_byte, device=device, dtype=torch.float32, requires_grad=True)

    h_byte, b_logits = triton_fused_byte_encoder(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, None
    )

    (h_byte.sum() + b_logits.sum()).backward()
    assert embed_w.grad is not None
    assert not torch.isnan(embed_w.grad).any()


def test_triton_byte_encoder_gradcheck():
    """Verify double-precision torch.autograd.gradcheck passes."""
    from affine_ai.kernels.triton_byte_encoder import TritonByteEncoderFunction

    torch.manual_seed(42)
    B, T, d_byte, K = 2, 4, 8, 3
    vocab_size = 16

    byte_ids = torch.randint(0, vocab_size, (B, T), dtype=torch.int64)
    embed_w = torch.randn(vocab_size, d_byte, dtype=torch.float64, requires_grad=True)
    conv_w = torch.randn(d_byte, 1, K, dtype=torch.float64, requires_grad=True)
    conv_b = torch.randn(d_byte, dtype=torch.float64, requires_grad=True)
    norm_scale = torch.randn(d_byte, dtype=torch.float64, requires_grad=True)
    proj_w = torch.randn(d_byte, d_byte, dtype=torch.float64, requires_grad=True)
    bp_w = torch.randn(1, d_byte, dtype=torch.float64, requires_grad=True)
    bp_b = torch.randn(1, dtype=torch.float64, requires_grad=True)

    torch.autograd.gradcheck(
        TritonByteEncoderFunction.apply,
        (byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, bp_b),
        eps=1e-6,
        atol=1e-4,
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (2, 32), (4, 128)])
def test_triton_byte_encoder_shapes(shape):
    """Verify single-token decoding (M=1) up to batch sizes (M=128+)."""
    B, T = shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_byte, K = 16, 3
    vocab_size = 256

    byte_ids = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.int64)
    embed_w = torch.randn(vocab_size, d_byte, device=device, dtype=torch.float32, requires_grad=True)
    conv_w = torch.randn(d_byte, 1, K, device=device, dtype=torch.float32, requires_grad=True)
    conv_b = torch.randn(d_byte, device=device, dtype=torch.float32, requires_grad=True)
    norm_scale = torch.randn(d_byte, device=device, dtype=torch.float32, requires_grad=True)
    proj_w = torch.randn(d_byte, d_byte, device=device, dtype=torch.float32, requires_grad=True)
    bp_w = torch.randn(1, d_byte, device=device, dtype=torch.float32, requires_grad=True)

    h_byte, b_logits = triton_fused_byte_encoder(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, None
    )
    assert h_byte.shape == (B, T, d_byte)
    assert b_logits.shape == (B, T)
    (h_byte.sum() + b_logits.sum()).backward()
    assert embed_w.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP16 overflow test")
def test_triton_byte_encoder_fp16_rms_stability():
    """Verify float32 RMS computation prevents overflow in FP16."""
    torch.manual_seed(42)
    B, T, d_byte, K = 2, 8, 16, 3
    vocab_size = 256

    byte_ids = torch.randint(0, vocab_size, (B, T), device="cuda", dtype=torch.int64)
    # Weights where x_res is within FP16 (< 65504) but x_res^2 exceeds FP16 max (65504)
    embed_w = torch.full((vocab_size, d_byte), 200.0, device="cuda", dtype=torch.float16, requires_grad=True)
    conv_w = torch.full((d_byte, 1, K), 1.0, device="cuda", dtype=torch.float16, requires_grad=True)
    conv_b = torch.zeros(d_byte, device="cuda", dtype=torch.float16, requires_grad=True)
    norm_scale = torch.ones(d_byte, device="cuda", dtype=torch.float16, requires_grad=True)
    proj_w = torch.randn(d_byte, d_byte, device="cuda", dtype=torch.float16, requires_grad=True)
    bp_w = torch.randn(1, d_byte, device="cuda", dtype=torch.float16, requires_grad=True)

    h_byte, b_logits = triton_fused_byte_encoder(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, None
    )

    assert not torch.isnan(h_byte).any()
    assert not torch.isinf(h_byte).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton patch pooling test")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("P", [8, 16, 32])
def test_triton_patch_mean_pool(dtype, P):
    """Verify triton_patch_mean_pool forward and backward parity against PyTorch."""
    from affine_ai.kernels.triton_byte_encoder import triton_patch_mean_pool

    torch.manual_seed(42)
    B, T, D = 4, 128, 64
    M = T // P

    x_ref = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)

    out_ref = x_ref.view(B, M, P, D).mean(dim=2)
    out_tri = triton_patch_mean_pool(x_tri, P)

    assert torch.allclose(out_ref, out_tri, atol=1e-4, rtol=1e-3)

    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    out_tri.backward(g)

    assert torch.allclose(x_ref.grad, x_tri.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton patch pooling test")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("P", [8, 16])
def test_triton_patch_weighted_pool(dtype, P):
    """Verify triton_patch_weighted_pool forward and backward parity against PyTorch."""
    from affine_ai.kernels.triton_byte_encoder import triton_patch_weighted_pool

    torch.manual_seed(42)
    B, T, D = 4, 128, 64
    M = T // P

    x_ref = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    l_ref = torch.randn(B, T, device="cuda", dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    l_tri = l_ref.detach().clone().requires_grad_(True)

    weights = F.softmax(l_ref.view(B, M, P), dim=-1).unsqueeze(-1)
    out_ref = (x_ref.view(B, M, P, D) * weights).sum(dim=2)
    out_tri = triton_patch_weighted_pool(x_tri, l_tri, P)

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3

    assert torch.allclose(out_ref, out_tri, atol=atol, rtol=rtol)

    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    out_tri.backward(g)

    assert torch.allclose(x_ref.grad, x_tri.grad, atol=atol, rtol=rtol)
    assert torch.allclose(l_ref.grad, l_tri.grad, atol=atol, rtol=rtol)


