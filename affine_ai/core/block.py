import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from affine_ai.core.norm import RMSNorm
from affine_ai.core.recurrent import TreeAssociativeMemory
from affine_ai.core.multi_head import MultiHeadAffineTree
from affine_ai.core.backpressure_tree import FusedSparseBackpressureTreeV3
from affine_ai.core.ast_dag import ASTDAGLayer


class CQAPv8Block(nn.Module):
    """
    CQAP v8 Block (Canonical Quadratic Affine Polytope v8):
    1. Logarithmic Hierarchical Polytope Router (O(depth * N) hyperplane evaluations).
    2. Direct Fused Leaf Transformation: Low-rank leaves map directly to output without
       allocating or computing a separate dense out_proj matrix.
    3. Fused SRAM Dispatch with Exact Autograd Backward.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        depth: int = 2,
        n_ary: int = 4,
        rank: int = 16,
        top_k: int = 2,
        temperature: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.depth = depth
        self.n_ary = n_ary
        self.num_leaves = n_ary ** depth
        self.rank = rank
        self.top_k = min(top_k, self.num_leaves)
        self.temperature = max(temperature, 1e-4)

        self.norm = RMSNorm(d_model)

        # 1. Logarithmic Hierarchical Router Hyperplanes: (H, d_head, depth * n_ary)
        self.router_weights = nn.Parameter(
            torch.randn(n_heads, self.d_head, depth * n_ary) * (1.0 / math.sqrt(self.d_head))
        )
        self.router_biases = nn.Parameter(torch.zeros(n_heads, depth * n_ary))

        # 2. Direct Fused Leaf Transformations (No dense out_proj matrix!)
        self.leaf_u = nn.Parameter(
            torch.randn(n_heads, self.num_leaves, self.d_head, rank) * (1.0 / math.sqrt(self.d_head))
        )
        self.leaf_v = nn.Parameter(
            torch.randn(n_heads, rank, self.d_head) * (1.0 / math.sqrt(rank))
        )
        self.leaf_biases = nn.Parameter(torch.zeros(n_heads, self.num_leaves, self.d_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_norm = self.norm(x)
        B_star = x_norm.reshape(-1, self.d_model).shape[0]
        x_heads = x_norm.reshape(B_star, self.n_heads, self.d_head)

        # 1. Logarithmic Hierarchical Router
        raw_logits = torch.einsum('bhd, hds -> bhs', x_heads, self.router_weights) + self.router_biases
        logits_reshaped = raw_logits.view(B_star, self.n_heads, self.depth, self.n_ary)
        log_p = F.log_softmax(logits_reshaped / self.temperature, dim=-1)

        # Cartesian sum across tree depth: (B*, H, N, 1) + (B*, H, 1, N) -> (B*, H, N^2)
        route_logits = (log_p[:, :, 0, :, None] + log_p[:, :, 1, None, :]).reshape(B_star, self.n_heads, self.num_leaves)

        # 2. Top-K Dispatch
        top_logits, top_indices = torch.topk(route_logits, self.top_k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)

        # 3. Direct Fused Leaf Computation (Vectorized across top-k)
        h_idx = torch.arange(self.n_heads, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, H, 1)
        u_k = self.leaf_u[h_idx, top_indices]      # (B_star, H, K, d_head, rank)
        b_k = self.leaf_biases[h_idx, top_indices]  # (B_star, H, K, d_head)
        low_rank = torch.einsum('bhd, bhkdr -> bhkr', x_heads, u_k)
        leaf_out = torch.einsum('bhkr, hrd -> bhkd', low_rank, self.leaf_v) + b_k
        head_outs = torch.einsum('bhk, bhkd -> bhd', top_weights, leaf_out)

        out = head_outs.reshape(B_star, self.d_model)
        return x + out.reshape(*orig_shape)


class AffineTreeBlock(nn.Module):
    """
    Unified Dual-Mixer CQAP v8 Block combining:
    1. Time-Mixer: Recurrent Decaying Associative Memory + Spatial Tree
    2. Channel-Mixer: Multi-Head Polytope Affine Tree (CQAP v8)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: Optional[int] = None,
        n_ary: int = 4,
        depth: int = 2,
        rank: Optional[int] = 16,
        top_k: Optional[int] = 2,
        router_type: str = "polytope",
        init_gaussian: bool = False,
        chunk_size: int = 64,
        channel_mixer: str = "multi_head",
        ternary_leaves: bool = True,
        leaf_sparsity: float = 0.0,
        nm: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else (d_model // n_heads)
        self.channel_mixer_type = channel_mixer

        self.time_norm = RMSNorm(d_model)
        self.time_mixer = TreeAssociativeMemory(
            d_model=d_model,
            n_heads=n_heads,
            d_head=self.d_head,
            n_ary=n_ary,
            depth=depth,
            rank=rank,
            top_k=top_k,
            router_type=router_type,
            init_gaussian=init_gaussian,
            chunk_size=chunk_size
        )
        self.channel_norm = RMSNorm(d_model)
        if channel_mixer == "backpressure":
            warnings.warn(
                "channel_mixer='backpressure' (FusedSparseBackpressureTreeV3) is deprecated. "
                "Use 'ast_dag' (AdaptiveSparseTreeDAGLayer) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Hydraulic tree with LOW-RANK leaves by default (rank=None here
            # coerces to 16, matching the module's own default -- full-rank
            # is available only by constructing the layer directly with
            # rank=None). Trainable by plain backprop (STE) or via
            # apply_updates() on the layer itself.
            mixer_rank = rank if rank is not None else 16
            self.channel_mixer = FusedSparseBackpressureTreeV3(
                in_features=d_model,
                out_features=d_model,
                depth=depth,
                n_ary=n_ary,
                top_k=top_k if top_k is not None else 2,
                rank=mixer_rank,
                ternary_leaves=ternary_leaves,
                leaf_sparsity=leaf_sparsity,
                nm=nm,
            )
        elif channel_mixer == "ast_dag":
            self.channel_mixer = ASTDAGLayer(
                dim=d_model,
                out_features=d_model,
                initial_branches=n_ary ** depth if depth <= 2 else n_ary,
                rank=rank,
                nm=nm,
                leaf_sparsity=leaf_sparsity,
            )
        else:
            self.channel_mixer = MultiHeadAffineTree(
                d_model=d_model,
                n_heads=n_heads,
                d_head=self.d_head,
                n_ary=n_ary,
                depth=depth,
                rank=rank,
                top_k=top_k,
                router_type=router_type,
                init_gaussian=init_gaussian
            )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm Time-Mixer with Residual
        time_out, next_state = self.time_mixer(self.time_norm(x), initial_state=state)
        x = x + time_out

        # Pre-Norm Channel-Mixer with Residual
        channel_out = self.channel_mixer(self.channel_norm(x))
        x = x + channel_out

        return x, next_state

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # O(1) Time-Mixer Step
        time_out, next_state = self.time_mixer.step(self.time_norm(x_t), state)
        x_t = x_t + time_out

        # O(1) Channel-Mixer Step
        channel_out = self.channel_mixer(self.channel_norm(x_t))
        x_t = x_t + channel_out

        return x_t, next_state

