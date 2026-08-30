import pytest
import torch
import torch.nn.functional as F
from affine_ai.models.mtp import ASDAGMTPHead, ASDAGMTPModule
from affine_ai.models.language_model import ASDAGLanguageModel


def test_mtp_head_forward():
    d_model = 64
    vocab_size = 256
    head = ASDAGMTPHead(d_model=d_model, vocab_size=vocab_size, k_offset=2)
    
    h = torch.randn(2, 16, d_model)
    logits = head(h)
    assert logits.shape == (2, 16, vocab_size)


def test_mtp_module_loss_and_gradients():
    d_model = 64
    vocab_size = 256
    mtp = ASDAGMTPModule(d_model=d_model, vocab_size=vocab_size, num_mtp_heads=2, mtp_lambda=0.3)
    
    h = torch.randn(2, 32, d_model, requires_grad=True)
    targets = torch.randint(0, vocab_size, (2, 32))
    
    all_logits, loss, stats = mtp(h, targets=targets)
    assert len(all_logits) == 2
    assert loss is not None
    assert loss.item() > 0
    assert "loss_mtp_k2" in stats
    assert "loss_mtp_k3" in stats
    
    loss.backward()
    assert mtp.heads[0].proj.weight.grad is not None
    assert mtp.heads[1].proj.weight.grad is not None


def test_asdag_blt_with_mtp():
    model = ASDAGLanguageModel(
        vocab_size=256,
        d_byte=64,
        d_model=96,
        n_layers=2,
        n_heads=2,
        target_patch_size=8,
        use_blt=True,
        use_hybrid=False,
        use_mtp=True,
        num_mtp_heads=1,
        mtp_lambda=0.3,
        dtype=torch.float32
    )
    
    x = torch.randint(0, 256, (2, 64))
    y = torch.randint(0, 256, (2, 64))
    
    logits, loss, stats = model.blt(x, targets=y)
    assert logits.shape == (2, 64, 256)
    assert loss is not None
    assert "loss_main" in stats
    assert "loss_mtp" in stats
    
    loss.backward()
    assert model.blt.mtp.heads[0].proj.weight.grad is not None
