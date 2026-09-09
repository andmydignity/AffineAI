"""
Cross-Architecture Attention Bridge (CAB)
=========================================
Implements the Attention Bridge distillation architecture enabling quadratic
Softmax Attention layers to be replaced with 100% O(1) linear recurrent Gated DeltaNet
layers while preserving output representation fidelity.
"""

import math
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.models.qwen35_asdag import Qwen35ASDAGConfig, Qwen35GatedDeltaNet, Qwen35GatedAttention


class AttentionBridge(nn.Module):
    """
    Lightweight SwiGLU Projection Bridge:
    Maps linear recurrent state representations (Gated DeltaNet) into the quadratic
    softmax attention output space with a non-linear residual adapter.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dtype: Any = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 1.5)
        
        # SwiGLU Gate & Up projection: [dim -> hidden_dim * 2]
        self.gate_up = nn.Linear(dim, self.hidden_dim * 2, bias=False)
        # Down projection: [hidden_dim -> dim]
        self.down = nn.Linear(self.hidden_dim, dim, bias=False)
        
        # Initialize down projection near zero so initial bridge behaves as smooth identity
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.normal_(self.gate_up.weight, std=1.0 / math.sqrt(dim))
        
        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        adapted = self.down(F.silu(gate) * up)
        return x + adapted


class CrossArchitectureAttentionBridge(nn.Module):
    """
    Cross-Architecture Attention Bridge (CAB) Layer:
    Combines an O(1) linear recurrent Gated DeltaNet with a trained AttentionBridge MLP.
    
    Modes:
      - Inference / Standard: Executes student DeltaNet + Bridge in O(1) constant memory (Zero quadratic attention!).
      - Distillation: Executes both teacher attention and student DeltaNet, returning bridge loss L_CAB.
    """
    def __init__(
        self,
        config: Qwen35ASDAGConfig,
        include_teacher: bool = False
    ):
        super().__init__()
        self.config = config
        self.student_delta = Qwen35GatedDeltaNet(config)
        self.bridge = AttentionBridge(config.dim, dtype=config.dtype)
        
        self.teacher_attention: Optional[Qwen35GatedAttention] = None
        if include_teacher:
            self.teacher_attention = Qwen35GatedAttention(config)
            # Freeze teacher parameters by default
            for p in self.teacher_attention.parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        pos: int = 0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        O(1) Recurrent Forward:
        Passes input through Gated DeltaNet and applies the Attention Bridge.
        """
        delta_out, next_state = self.student_delta(x, conv_state=conv_state, ssm_state=ssm_state)
        bridged_out = self.bridge(delta_out)
        return bridged_out, next_state

    def forward_distill(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
        pos: int = 0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Distillation Forward:
        Executes teacher attention and student DeltaNet + Bridge, computing L_CAB.
        """
        if self.teacher_attention is None:
            raise RuntimeError("Teacher attention was not instantiated (include_teacher=True required for distillation).")
            
        with torch.no_grad():
            y_attn, _ = self.teacher_attention(x, pos=pos)
            
        delta_out, next_state = self.student_delta(x, conv_state=conv_state, ssm_state=ssm_state)
        y_bridged = self.bridge(delta_out)
        
        # CAB Distillation Loss: MSE + Cosine alignment
        loss_mse = F.mse_loss(y_bridged, y_attn)
        cos_sim = F.cosine_similarity(y_bridged, y_attn, dim=-1).mean()
        loss_cab = loss_mse + (1.0 - cos_sim)
        
        return y_bridged, next_state, loss_cab
