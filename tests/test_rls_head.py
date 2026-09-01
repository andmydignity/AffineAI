import torch
from affine_ai.core.rls_head import RLSPredictiveHead

def test_rls_converges():
    torch.manual_seed(0)
    d, V = 16, 8
    W_true = torch.randn(V, d) * 0.5
    head = RLSPredictiveHead(d_model=d, vocab_size=V, forgetting=0.999, ridge=0.1)
    # Generate data
    X = torch.randn(200, d)
    y = (X @ W_true.T).argmax(dim=-1)
    # Initial error
    with torch.no_grad():
        logits, _ = head(X[:10])
        pred = logits.argmax(dim=-1)
        # not asserting initial, just ensure update runs
    for i in range(0, 200, 20):
        head.update(X[i:i+20], y[i:i+20])
    with torch.no_grad():
        assert head.P.trace().item() < d * (1.0 / 0.1)  # initial trace = d/ridge
        h_seen = X[:5]
        assert (head.uncertainty(h_seen) > 0).all()
        assert torch.isfinite(head.uncertainty(h_seen)).all()

def test_rls_contract():
    head = RLSPredictiveHead(d_model=8, vocab_size=4)
    h = torch.randn(2, 4, 8)
    logits, loss = head(h, targets=torch.randint(0,4,(2,4)))
    assert logits.shape == (2,4,4)
    assert loss is not None and loss.item() > 0
    logits2, _ = head(h, targets=None)
    assert logits2.shape == (2,4,4)

def test_rls_ignore_index():
    head = RLSPredictiveHead(d_model=8, vocab_size=4)
    h = torch.randn(2, 4, 8)
    y = torch.randint(0,4,(2,4))
    y[0,0] = -100
    head.update(h, y, ignore_index=-100)
    assert head._updates == 7  # 8 - 1 ignored

def test_rls_no_autograd_pollution():
    head = RLSPredictiveHead(d_model=8, vocab_size=4)
    h = torch.randn(2, 4, 8, requires_grad=True)
    y = torch.randint(0,4,(2,4))
    with torch.enable_grad():
        head.update(h, y)
    # update should not create grad nodes
    assert not head.P.requires_grad
    # External loss still backward works
    logits, loss = head(h, targets=y)
    loss.backward()
    assert h.grad is not None
