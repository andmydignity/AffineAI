import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.associative import PermutationProjection, NativeASDAGAssociativeMixer
from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGBlock
from affine_ai.core.ast_dag import ASDAGConfig


def test_permutation_projection():
    dim = 64
    num_perms = 4
    proj = PermutationProjection(dim=dim, num_perms=num_perms, seed_offset=42, dtype=torch.bfloat16)
    
    x = torch.randn(2, 8, dim, dtype=torch.bfloat16, requires_grad=True)
    out = proj(x)
    
    assert out.shape == (2, 8, dim)
    assert out.dtype == torch.bfloat16
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert proj.latent_w.grad is not None


def test_associative_mixer_parallel_vs_step():
    torch.manual_seed(42)
    d_model = 64
    n_heads = 4
    mixer = NativeASDAGAssociativeMixer(d_model=d_model, n_heads=n_heads, num_perms=4, dtype=torch.float32)
    mixer.eval()
    
    B, T = 2, 8
    x = torch.randn(B, T, d_model)
    
    # 1. Parallel training mode
    out_parallel, _ = mixer(x)
    
    # 2. Sequential step mode
    state = None
    step_outs = []
    for t in range(T):
        x_t = x[:, t:t+1]
        out_t, state = mixer.step(x_t, state)
        step_outs.append(out_t)
    out_step = torch.cat(step_outs, dim=1)
    
    # Outputs should match closely (within floating-point numerical tolerance)
    diff = (out_parallel - out_step).abs().max().item()
    assert diff < 1e-4, f"Parallel vs Sequential step mismatch: {diff}"


def test_asdag_lm_dual_mixer_end_to_end():
    torch.manual_seed(42)
    model = ASDAGLanguageModel(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        num_leaves=4,
        num_permutations=4,
        dtype=torch.bfloat16
    )
    
    # Forward pass
    input_ids = torch.randint(0, 128, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, 128)
    assert logits.dtype == torch.bfloat16
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    
    # Generation with O(1) state cache
    gen_tokens = model.generate(input_ids[:, :4], max_new_tokens=8, use_cache=True)
    assert gen_tokens.shape == (2, 12)
