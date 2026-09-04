"""Unit tests for Triton Fused Linear Cross Entropy kernel."""
from tests.test_triton_loss import (
    test_triton_fused_linear_cross_entropy_parity,
    test_triton_fused_linear_cross_entropy_gradcheck,
    test_triton_fused_linear_cross_entropy_multi_dim,
)

__all__ = [
    "test_triton_fused_linear_cross_entropy_parity",
    "test_triton_fused_linear_cross_entropy_gradcheck",
    "test_triton_fused_linear_cross_entropy_multi_dim",
]
