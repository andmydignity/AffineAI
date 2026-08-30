import pytest
import torch
import torch.nn.functional as F
from affine_ai.core.cpp_ops import get_asdag_cpu_ops, asdag_cpu_forward_backward

def test_asdag_cpu_ops_forward_parity():
    ops = get_asdag_cpu_ops()
    assert ops is not False and ops is not None, "asdag_cpu_ops extension failed to compile/load"

    torch.manual_seed(42)
    B = 8
    dim = 32
    K = 4

    x = torch.randn(B, dim, dtype=torch.float32)
    W = torch.randint(-1, 2, (K, dim, dim), dtype=torch.float32)
    biases = torch.randn(K, dim, dtype=torch.float32)
    routing_probs = F.softmax(torch.randn(B, K), dim=-1)

    # 1. Reference PyTorch computation
    leaf_prim = torch.einsum('bi, kdi -> bkd', x, W) + biases.unsqueeze(0)
    leaf_outs_ref = torch.clamp(leaf_prim, 0.0, 6.0)
    out_ref = torch.einsum('bk, bkd -> bd', routing_probs, leaf_outs_ref)

    # 2. Native C++ SIMD computation
    out_cpp, leaf_outs_cpp = ops.forward(x, W, biases, routing_probs)

    assert torch.allclose(out_ref, out_cpp, atol=1e-5), f"Max diff: {(out_ref - out_cpp).abs().max()}"
    assert torch.allclose(leaf_outs_ref, leaf_outs_cpp, atol=1e-5)


def test_asdag_cpu_ops_autograd_parity():
    ops = get_asdag_cpu_ops()
    assert ops is not False and ops is not None

    torch.manual_seed(42)
    B = 4
    dim = 16
    K = 4

    x1 = torch.randn(B, dim, requires_grad=True)
    W1 = torch.randint(-1, 2, (K, dim, dim), dtype=torch.float32).requires_grad_(True)
    b1 = torch.randn(K, dim, requires_grad=True)
    probs1 = F.softmax(torch.randn(B, K), dim=-1).requires_grad_(True)

    x2 = x1.detach().clone().requires_grad_(True)
    W2 = W1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)
    probs2 = probs1.detach().clone().requires_grad_(True)

    # Reference PyTorch
    leaf_prim = torch.einsum('bi, kdi -> bkd', x1, W1) + b1.unsqueeze(0)
    leaf_outs_ref = torch.clamp(leaf_prim, 0.0, 6.0)
    out_ref = torch.einsum('bk, bkd -> bd', probs1, leaf_outs_ref)
    loss_ref = (out_ref ** 2).sum()
    loss_ref.backward()

    # C++ Autograd
    out_cpp = asdag_cpu_forward_backward(x2, W2, b2, probs2)
    loss_cpp = (out_cpp ** 2).sum()
    loss_cpp.backward()

    assert torch.allclose(out_ref, out_cpp, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(W1.grad, W2.grad, atol=1e-5)


def test_asdag_cpu_sparse_tree_perm():
    from affine_ai.core.cpp_ops import asdag_cpu_sparse_tree_perm
    torch.manual_seed(42)
    B = 16
    dim = 32
    K = 8
    P = 4
    N = 2

    x1 = torch.randn(B, dim, requires_grad=True)
    w1 = torch.randn(K, P, dim, requires_grad=True)
    perms = torch.stack([torch.randperm(dim) for _ in range(K * P)]).view(K, P, dim)
    perms[:, 0] = torch.arange(dim)
    inv_perms = torch.empty_like(perms)
    for k in range(K):
        for p in range(P):
            inv_perms[k, p][perms[k, p]] = torch.arange(dim)
    b1 = torch.randn(K, dim, requires_grad=True)
    
    top_indices = torch.randint(0, K, (B, N))
    top_weights1 = torch.softmax(torch.randn(B, N), dim=-1).requires_grad_(True)

    x2 = x1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)
    top_weights2 = top_weights1.detach().clone().requires_grad_(True)

    # Reference
    w_sel = w1[top_indices]
    b_sel = b1[top_indices]
    p_sel = perms[top_indices]
    xg = torch.gather(x1.unsqueeze(1).unsqueeze(2).expand(B, N, P, -1), -1, p_sel)
    l_prim = (xg * w_sel).sum(dim=2) + b_sel
    l_act = torch.clamp(l_prim, 0.0, 6.0)
    out_ref = (l_act * top_weights1.unsqueeze(-1)).sum(dim=1)
    loss_ref = (out_ref ** 2).sum()
    loss_ref.backward()

    # C++ SIMD-Block N:M
    out_cpp = asdag_cpu_sparse_tree_perm(x2, w2, perms, inv_perms, b2, top_indices, top_weights2)
    loss_cpp = (out_cpp ** 2).sum()
    loss_cpp.backward()

    assert torch.allclose(out_ref, out_cpp, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(w1.grad, w2.grad, atol=1e-5)
    assert torch.allclose(b1.grad, b2.grad, atol=1e-5)
    assert torch.allclose(top_weights1.grad, top_weights2.grad, atol=1e-5)


def test_asdag_cpu_2bit_packing():
    from affine_ai.core.cpp_ops import asdag_cpu_pack_ternary_2bit, asdag_cpu_unpack_ternary_2bit
    torch.manual_seed(42)
    shape = (16, 64)
    w = torch.randint(-1, 2, shape).float()
    packed = asdag_cpu_pack_ternary_2bit(w)
    assert packed.numel() == (16 * 64) // 4
    unpacked = asdag_cpu_unpack_ternary_2bit(packed, shape)
    assert torch.equal(w, unpacked)


def test_asdag_cpu_monarch_reg_parity():
    from affine_ai.core.cpp_ops import get_asdag_cpu_ops, asdag_cpu_monarch_reg_forward
    ops = get_asdag_cpu_ops()
    assert ops is not False and ops is not None
    torch.manual_seed(42)
    B, dim, L = 8, 32, 4
    x = torch.randn(B, dim)
    diagonals = torch.randn(L, dim)
    perms = torch.stack([torch.randperm(dim) for _ in range(L - 1)])
    bias = torch.randn(dim)

    # Reference
    out_ref = ops.monarch_chain_forward(x, diagonals, perms, bias)
    # Register-fused
    out_reg = asdag_cpu_monarch_reg_forward(x, diagonals, perms, bias)
    assert torch.allclose(out_ref, out_reg, atol=1e-5)


def test_asdag_cpu_fused_rmsnorm_parity():
    from affine_ai.core.cpp_ops import asdag_cpu_fused_rmsnorm_proj
    torch.manual_seed(42)
    B, in_dim, out_dim = 8, 32, 64
    x = torch.randn(B, in_dim)
    w = torch.randn(out_dim, in_dim)
    eps = 1e-5

    rms_scale = 1.0 / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    ref = F.linear(x * rms_scale, w)
    cpp = asdag_cpu_fused_rmsnorm_proj(x, w, eps=eps)
    assert torch.allclose(ref, cpp, atol=1e-5)

