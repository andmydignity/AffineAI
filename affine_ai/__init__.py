"""
AffineAI: Pure MatMul-Free Adaptive Sparse Tree DAG (ASDAG) Neural Architecture
"""

import os

# Configure PyTorch virtual memory allocator for reduced fragmentation if not explicitly set
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from affine_ai.core.ast_dag import ASTDAGLayer, ASDAGConfig, AdaptiveSparseTreeDAGLayer
# Deprecated: Use AdaptiveSparseTreeDAGLayer or ASTDAGLayer instead of FusedSparseBackpressureTreeV3
from affine_ai.core.backpressure_tree import FusedSparseBackpressureTreeV3
from affine_ai.core.associative import NativeASDAGAssociativeMixer, PermutationProjection

AdaptiveBackpressureTreeLayer = FusedSparseBackpressureTreeV3  # Deprecated alias

from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGBlock
from affine_ai.models.blt import ASDAGByteLatentModel, ASDAGByteLatentModel as ByteLatentTransformer, ASDAGByteLatentModel as BLT
from affine_ai.core.lpc import LocalPredictiveLanguageModel, LocalPredictiveHead
from affine_ai.core.priority_replay import DynamicPriorityReplayBuffer
from affine_ai.training.trainer import ASDAGTrainer, train

# Deprecated: TorosJEPA is deprecated. Use TorosHybridLanguageModel or ASDAGLanguageModel instead.
from affine_ai.models.jepa import TorosJEPA, TorosJEPAConfig
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.core.format import (
    save_toros_model,
    load_toros_model,
    read_toros_metadata,
    format_toros_summary,
    FLAG_POT5_RESIDUAL_FP16,
    FLAG_POT5_RESIDUAL_Q4,
    pack_pot5_residual_fp16,
    pack_pot5_residual_q4,
    unpack_pot5_residual_fp16,
    unpack_pot5_residual_q4,
)

from affine_ai.models.qwen35_asdag import Qwen35ASDAGModel, Qwen35ASDAGConfig, Qwen35Block
from affine_ai.models.qwen35_blt import Qwen35BLTLanguageModel, Qwen35BLTConfig
from affine_ai.models.qwen38_asdag import Qwen38ASDAGModel, Qwen38ASDAGConfig, Qwen38Block
from affine_ai.models.attention_bridge import AttentionBridge, CrossArchitectureAttentionBridge
from affine_ai.training.distill_bytes import DistillBytesAligner
from affine_ai.core.cuda_graph import CUDAGraphRunner
from affine_ai.data.dataloader import (
    PaddedDataLoader,
    FixedShapeDataLoader,
    CUDAGraphDataLoader,
)

__all__ = [
    "ASTDAGLayer",
    "ASDAGConfig",
    "AdaptiveSparseTreeDAGLayer",
    "FusedSparseBackpressureTreeV3",
    "AdaptiveBackpressureTreeLayer",
    "NativeASDAGAssociativeMixer",
    "PermutationProjection",
    "ASDAGLanguageModel",
    "ASDAGBlock",
    "ASDAGByteLatentModel",
    "ByteLatentTransformer",
    "BLT",
    "LocalPredictiveLanguageModel",
    "LocalPredictiveHead",
    "DynamicPriorityReplayBuffer",
    "ASDAGTrainer",
    "TorosJEPA",
    "TorosJEPAConfig",
    "TorosHybridLanguageModel",
    "TorosHybridConfig",
    "save_toros_model",
    "load_toros_model",
    "read_toros_metadata",
    "format_toros_summary",
    "FLAG_POT5_RESIDUAL_FP16",
    "FLAG_POT5_RESIDUAL_Q4",
    "pack_pot5_residual_fp16",
    "pack_pot5_residual_q4",
    "unpack_pot5_residual_fp16",
    "unpack_pot5_residual_q4",
    "Qwen35ASDAGModel",
    "Qwen35ASDAGConfig",
    "Qwen35Block",
    "Qwen35BLTLanguageModel",
    "Qwen35BLTConfig",
    "Qwen38ASDAGModel",
    "Qwen38ASDAGConfig",
    "Qwen38Block",
    "AttentionBridge",
    "CrossArchitectureAttentionBridge",
    "DistillBytesAligner",
    "CUDAGraphRunner",
    "PaddedDataLoader",
    "FixedShapeDataLoader",
    "CUDAGraphDataLoader",
    "train",
]
