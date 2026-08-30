# AffineAI: Affine N-ary Trees

`affine_ai` is a PyTorch library implementing **Affine N-ary Trees**—neural network layers that substitute rigid dense matrix multiplications with multi-interval polytope spatial routing and local continuous affine leaf surfaces.

## Key Concepts

- **$N$-ary Polytope Partitioning ($N \ge 3$)**: Partitions feature spaces using multi-interval thresholds per dimension.
- **Local Affine Leaves ($y = Wx + b$)**: Continuous linear fitting at tree leaves eliminating staircase errors.
- **Batched 3D Einsum LM**: Parallel sequence modeling using vectorized leaf transformations across token batches.

## Quickstart

```python
import torch
from affine_ai import AffineNaryTree, BatchedAffineTreeLM

# Create an Affine N-ary Tree layer
# 2 dimensions, branching factor N=4, tree depth D=2 (16 leaves)
tree = AffineNaryTree(in_features=2, out_features=1, n_ary=4, depth=2)
x = torch.randn(32, 2)
y = tree(x)  # shape: (32, 1)

# Create a Batched Affine Tree LM
lm = BatchedAffineTreeLM(vocab_size=65, embed_dim=16, context_len=32, n_ary=4, depth=3)
input_ids = torch.randint(0, 65, (4, 32))
logits = lm(input_ids)  # shape: (4, 32, 65)
```
