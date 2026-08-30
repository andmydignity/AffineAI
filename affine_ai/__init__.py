"""
AffineAI: Pure MatMul-Free Adaptive Sparse Tree DAG (ASDAG) Neural Architecture
"""

from affine_ai.core.ast_dag import ASTDAGLayer, ASDAGConfig, AdaptiveSparseTreeDAGLayer
from affine_ai.core.backpressure_tree import FusedSparseBackpressureTreeV3
from affine_ai.core.associative import NativeASDAGAssociativeMixer, PermutationProjection

AdaptiveBackpressureTreeLayer = FusedSparseBackpressureTreeV3

from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGBlock
from affine_ai.models.blt import ASDAGByteLatentModel, ASDAGByteLatentModel as ByteLatentTransformer, ASDAGByteLatentModel as BLT
from affine_ai.core.lpc import LocalPredictiveLanguageModel, LocalPredictiveHead
from affine_ai.training.trainer import ASDAGTrainer

from affine_ai.models.jepa import TorosJEPA, TorosJEPAConfig
from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
from affine_ai.core.format import save_toros_model, load_toros_model, read_toros_metadata

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
    "ASDAGTrainer",
    "TorosJEPA",
    "TorosJEPAConfig",
    "TorosHybridLanguageModel",
    "TorosHybridConfig",
    "save_toros_model",
    "load_toros_model",
    "read_toros_metadata",
]
