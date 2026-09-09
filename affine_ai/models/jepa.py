"""
Toros-JEPA: Joint Embedding Predictive Architecture for ASDAG
=============================================================
Implements Yann LeCun's Joint Embedding Predictive Architecture (JEPA)
adapted for the 1.58-Bit MatMul-Free ASDAG engine:

1. Context Encoder (E_theta): Encodes unmasked past context x -> s_x in latent space.
2. Target Encoder (E_xi): Encodes future target y -> s_y under torch.no_grad(), updated via EMA.
3. Latent Predictor (P_phi): Predicts future representation s_hat_y from s_x.
4. VICReg Anti-Collapse Loss: Invariance (L1/Smooth L1) + Variance (Std >= 1.0) regularization.
5. Pure Latent Rollouts: High-speed mental simulation without token decoding.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.ast_dag import ASDAGConfig
from affine_ai.core.norm import RMSNorm
from affine_ai.core.bitlinear import TernaryBitLinearSwiGLU, BitLinear
from affine_ai.models.language_model import ASDAGBlock


@dataclass
class TorosJEPAConfig:
    """
    .. deprecated:: 0.2.0
        `TorosJEPAConfig` is deprecated. Use `TorosHybridConfig` or `ASDAGConfig` instead.
    """
    dim: int = 128
    d_byte: int = 64
    n_encoder_layers: int = 4
    n_predictor_layers: int = 1
    n_heads: int = 4
    target_patch_size: int = 16
    channel_mixer_type: str = "ternary_swiglu"
    mlp_hidden_dim: int = 84
    time_mixer_rule: str = "gla"
    use_conv_prefix: bool = True
    conv_kernel_size: int = 4
    sim_loss_weight: float = 1.0
    sigreg_weight: float = 1.0
    sigreg_n_sketches: int = 4
    std_target: float = 1.0
    dtype: Any = torch.float32


from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher


class TorosEncoder(nn.Module):
    """
    1.58-Bit Ternary ASDAG Latent Encoder with BLT Dynamic Entropy Patching:
    Maps raw input bytes [0..255] into abstract patch representation trajectories.
    """
    def __init__(self, config: TorosJEPAConfig):
        super().__init__()
        self.config = config
        self.byte_encoder = ByteLocalEncoder(
            vocab_size=256,
            d_byte=config.d_byte,
            kernel_size=4,
            dtype=config.dtype
        )
        self.patcher = EntropyPatcher(
            d_byte=config.d_byte,
            d_model=config.dim,
            target_patch_size=config.target_patch_size,
            dtype=config.dtype
        )
        
        asdag_cfg = ASDAGConfig(
            dim=config.dim,
            dtype=config.dtype,
            time_mixer_rule=getattr(config, 'time_mixer_rule', 'gla'),
            use_conv_prefix=getattr(config, 'use_conv_prefix', True),
            conv_kernel_size=getattr(config, 'conv_kernel_size', 4),
            mlp_hidden_dim=getattr(config, 'mlp_hidden_dim', 84)
        )
        self.blocks = nn.ModuleList([
            ASDAGBlock(
                config=asdag_cfg,
                n_heads=config.n_heads,
                layer_idx=i,
                channel_mixer_type=config.channel_mixer_type
            )
            for i in range(config.n_encoder_layers)
        ])
        self.norm_out = RMSNorm(config.dim)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        h_byte, boundary_logits = self.byte_encoder(byte_ids)
        latent_patches, _ = self.patcher(h_byte, boundary_logits, fixed_patch_size=self.config.target_patch_size)
        h = latent_patches
        for block in self.blocks:
            h = block(h)
        return self.norm_out(h)


class TorosPredictor(nn.Module):
    """
    .. deprecated:: 0.2.0
        TorosPredictor is deprecated as part of the JEPA deprecation.

    Lightweight 1.58-Bit Latent Space Predictor:
    Predicts future state s_hat_y from current state s_x.
    """
    def __init__(self, config: TorosJEPAConfig):
        super().__init__()
        warnings.warn(
            "TorosPredictor is deprecated as part of the JEPA deprecation.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        asdag_cfg = ASDAGConfig(dim=config.dim, dtype=config.dtype, time_mixer_rule=getattr(config, 'time_mixer_rule', 'gla'))
        self.blocks = nn.ModuleList([
            ASDAGBlock(
                config=asdag_cfg,
                n_heads=config.n_heads,
                layer_idx=i,
                channel_mixer_type=config.channel_mixer_type
            )
            for i in range(config.n_predictor_layers)
        ])
        self.norm_out = RMSNorm(config.dim)
        self.pred_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, s_context: torch.Tensor) -> torch.Tensor:
        h = s_context
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.pred_proj(h)


class TorosJEPA(nn.Module):
    """
    .. deprecated:: 0.2.0
        `TorosJEPA` is deprecated. Empirical benchmarks demonstrated that latent auxiliary
        JEPA losses degrade language modeling perplexity compared to next-token prediction
        at 1.4-2.2x step cost. Use `TorosHybridLanguageModel` or `ASDAGLanguageModel` instead.

    Toros Joint Embedding Predictive Architecture (Toros-JEPA):
    - Context Encoder E_theta (Active Gradients)
    - Target Encoder E_xi (EMA-Updated, Zero Reverse-Mode Tape)
    - Predictor P_phi (Latent Trajectory Forecaster)
    """
    def __init__(self, config: Optional[TorosJEPAConfig] = None):
        super().__init__()
        warnings.warn(
            "TorosJEPA is deprecated. Empirical benchmarks demonstrated that latent auxiliary "
            "JEPA losses degrade language modeling perplexity compared to next-token prediction "
            "at 1.4-2.2x step cost. Use TorosHybridLanguageModel or ASDAGLanguageModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config or TorosJEPAConfig()

        # 1. Context Encoder
        self.context_encoder = TorosEncoder(self.config)

        # 2. Latent Predictor
        self.predictor = TorosPredictor(self.config)

        if self.config.dtype is not None and self.config.dtype != torch.float32:
            self.to(self.config.dtype)

    def _sigreg(self, z: torch.Tensor) -> torch.Tensor:
        """
        SIGReg anti-collapse regularizer: per random 1D projection, the sorted
        sketch (empirical marginal) is matched against the quantiles of a fixed
        isotropic Gaussian N(0, std_target). A collapsed embedding sorts to a
        point mass, far from the Gaussian quantiles, so collapse is punished
        directly — no EMA teacher, no separate variance hinge.
        """
        flat = z.reshape(-1, z.shape[-1]).float()
        n = flat.shape[0]
        std_target = self.config.std_target
        loss = flat.new_zeros(())
        for _ in range(self.config.sigreg_n_sketches):
            v = F.normalize(torch.randn(flat.shape[1], device=flat.device), dim=0)
            s = flat @ v
            g = torch.randn(n, device=flat.device).sort().values * std_target
            loss = loss + F.smooth_l1_loss(s.sort().values, g)
        return loss / self.config.sigreg_n_sketches

    def compute_vicreg_loss(
        self,
        s_pred: torch.Tensor,
        s_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Invariance (Smooth L1) + SIGReg collapse prevention in latent space.
        """
        sim_loss = F.smooth_l1_loss(s_pred, s_target)
        sigreg = self._sigreg(s_pred)

        total_loss = (
            self.config.sim_loss_weight * sim_loss +
            self.config.sigreg_weight * sigreg
        )

        metrics = {
            "loss_jepa": total_loss.item(),
            "loss_invariance": sim_loss.item(),
            "loss_sigreg": sigreg.item(),
            "latent_std": s_pred.float().std().item()
        }
        return total_loss, metrics

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_tgt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """
        Forward step:
        - Passes x_ctx through context encoder and predictor to get s_hat_y.
        - If x_tgt is provided, encodes it with a detached stop-grad pass and
          computes the JEPA loss.
        """
        # Context Encoder
        s_ctx = self.context_encoder(x_ctx)

        # Latent Predictor
        s_pred = self.predictor(s_ctx)

        if x_tgt is None:
            return s_pred, None, {}

        # Stop-grad target: re-encode the target through the same encoder, detached.
        with torch.no_grad():
            s_tgt = self.context_encoder(x_tgt).detach()

        # Align lengths if needed
        min_len = min(s_pred.shape[1], s_tgt.shape[1])
        s_pred_aligned = s_pred[:, :min_len]
        s_tgt_aligned = s_tgt[:, :min_len]

        loss, metrics = self.compute_vicreg_loss(s_pred_aligned, s_tgt_aligned)
        return s_pred, loss, metrics

    @torch.no_grad()
    def latent_rollout(
        self,
        initial_context: torch.Tensor,
        num_steps: int = 10
    ) -> List[torch.Tensor]:
        """
        Pure Latent Mental Simulation:
        Simulates 'num_steps' future thought trajectories in 1.58-bit latent space
        without executing vocabulary projections or token decoding.
        """
        trajectories = []
        curr_state = self.context_encoder(initial_context)
        
        for _ in range(num_steps):
            next_state = self.predictor(curr_state)
            trajectories.append(next_state)
            curr_state = next_state
            
        return trajectories
