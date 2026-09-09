import pytest
import torch
import torch.nn as nn
from affine_ai.core.cuda_graph import CUDAGraphRunner
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA graph tests")
def test_cuda_graph_inference():
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(self.linear(x))

    model = SimpleModule().cuda().bfloat16()
    x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)

    # Reference output
    ref_out = model(x).clone()

    runner = CUDAGraphRunner(
        step_fn=model.forward,
        sample_inputs=(x,),
        warmup_iters=2,
    )

    out = runner.step(x)
    assert torch.allclose(ref_out, out, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA graph tests")
def test_cuda_graph_lpc_adamw():
    config = TorosHybridConfig(
        context_dim=64,
        byte_dim=32,
        num_layers=2,
        target_patch_size=4,
        vocab_size=256,
        num_experts=1,
    )
    model = TorosHybridLanguageModel(config).cuda().bfloat16()
    opts = model.get_default_lpc_optimizers(use_muon=False, capturable=True)

    bx = torch.randint(0, 256, (2, 32), device="cuda")
    by = torch.randint(0, 256, (2, 32), device="cuda")

    runner = model.capture_lpc_graph(bx, by, opts, warmup_iters=2)

    for _ in range(3):
        bx_new = torch.randint(0, 256, (2, 32), device="cuda")
        by_new = torch.randint(0, 256, (2, 32), device="cuda")
        res = runner.step(bx_new, by_new)
        loss = res["loss"]
        assert torch.isfinite(loss)
        assert loss.item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA graph tests")
def test_cuda_graph_lpc_muon():
    config = TorosHybridConfig(
        context_dim=64,
        byte_dim=32,
        num_layers=2,
        target_patch_size=4,
        vocab_size=256,
        num_experts=1,
    )
    model = TorosHybridLanguageModel(config).cuda().bfloat16()
    opts = model.get_default_lpc_optimizers(use_muon=True, capturable=True)

    bx = torch.randint(0, 256, (2, 32), device="cuda")
    by = torch.randint(0, 256, (2, 32), device="cuda")

    runner = model.capture_lpc_graph(bx, by, opts, warmup_iters=2)

    for _ in range(3):
        bx_new = torch.randint(0, 256, (2, 32), device="cuda")
        by_new = torch.randint(0, 256, (2, 32), device="cuda")
        res = runner.step(bx_new, by_new)
        loss = res["loss"]
        assert torch.isfinite(loss)
        assert loss.item() > 0.0
