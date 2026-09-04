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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_byte_encoder_parity():
    torch.manual_seed(42)
    B, T, d_byte, K = 2, 8, 16, 3
    vocab_size = 256

    byte_ids = torch.randint(0, vocab_size, (B, T), device="cuda", dtype=torch.int64)
    embed_w = torch.randn(vocab_size, d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    conv_w = torch.randn(d_byte, 1, K, device="cuda", dtype=torch.float32, requires_grad=True)
    conv_b = torch.randn(d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    norm_scale = torch.randn(d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    proj_w = torch.randn(d_byte, d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    bp_w = torch.randn(1, d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    bp_b = torch.randn(1, device="cuda", dtype=torch.float32, requires_grad=True)

    h_byte, b_logits = triton_fused_byte_encoder(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, bp_b
    )

    ref_h, ref_b = ref_byte_encoder(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, bp_b
    )

    assert torch.allclose(h_byte, ref_h, atol=1e-4)
    assert torch.allclose(b_logits, ref_b, atol=1e-4)

    (h_byte.sum() + b_logits.sum()).backward()
    assert embed_w.grad is not None
    assert conv_w.grad is not None
    assert conv_b.grad is not None
    assert norm_scale.grad is not None
    assert proj_w.grad is not None
    assert bp_w.grad is not None
    assert bp_b.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_byte_encoder_int32_ids():
    """Verify backward scatter_add succeeds even when byte_ids is torch.int32."""
    torch.manual_seed(42)
    B, T, d_byte, K = 2, 8, 16, 3
    vocab_size = 256

    byte_ids = torch.randint(0, vocab_size, (B, T), device="cuda", dtype=torch.int32)
    embed_w = torch.randn(vocab_size, d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    conv_w = torch.randn(d_byte, 1, K, device="cuda", dtype=torch.float32, requires_grad=True)
    conv_b = torch.randn(d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    norm_scale = torch.randn(d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    proj_w = torch.randn(d_byte, d_byte, device="cuda", dtype=torch.float32, requires_grad=True)
    bp_w = torch.randn(1, d_byte, device="cuda", dtype=torch.float32, requires_grad=True)

    h_byte, b_logits = triton_fused_byte_encoder(
        byte_ids, embed_w, conv_w, conv_b, norm_scale, proj_w, bp_w, None
    )

    # If byte_ids is not cast to int64 in scatter_add_, this backward will raise RuntimeError
    (h_byte.sum() + b_logits.sum()).backward()
    assert embed_w.grad is not None
    assert not torch.isnan(embed_w.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
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
