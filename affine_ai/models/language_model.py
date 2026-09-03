import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any

from affine_ai.core.ast_dag import AdaptiveSparseTreeDAGLayer, ASDAGConfig, ASTDAGLayer
from affine_ai.core.norm import RMSNorm
from affine_ai.core.associative import NativeASDAGAssociativeMixer
from affine_ai.core.bitlinear import TernaryBitLinearSwiGLU


class ASDAGBlock(nn.Module):
    """
    Native Dual-Mixer ASDAG Block (100% MatMul-Free):
      1. Time Mixer: Monarch Gated Linear Associative (GLA) State Space (Zero-MatMul, O(1) Memory)
      2. Channel Mixer: Ternary BitLinear SwiGLU (Expand=2x, MatMul-Free) or ASDAG Sparse Tree
    """
    def __init__(
        self,
        config: ASDAGConfig,
        n_heads: int = 8,
        layer_idx: int = 0,
        channel_mixer_type: Optional[str] = None
    ):
        super().__init__()
        self.config = config
        self.channel_mixer_type = channel_mixer_type if channel_mixer_type is not None else getattr(config, 'channel_mixer_type', 'ternary_swiglu')
        self.norm1 = RMSNorm(config.dim)
        self.time_mixer = NativeASDAGAssociativeMixer(
            d_model=config.dim,
            n_heads=n_heads,
            proj_type="monarch",
            num_stages=4,
            seed_offset=layer_idx * 10,
            dtype=config.dtype,
            rule=getattr(config, 'time_mixer_rule', 'gla')
        )
        self.norm2 = RMSNorm(config.dim)

        if self.channel_mixer_type == "ternary_swiglu":
            self.channel_mixer = TernaryBitLinearSwiGLU(config.dim, expand=2, dtype=config.dtype)
            self.asdag = None
        elif self.channel_mixer_type == "dense_swiglu":
            self.channel_mixer = nn.Sequential(
                nn.Linear(config.dim, 2 * config.dim, bias=False, dtype=config.dtype),
                nn.SiLU(),
                nn.Linear(2 * config.dim, config.dim, bias=False, dtype=config.dtype)
            )
            self.asdag = None
        else: # "asdag_tree"
            self.asdag = AdaptiveSparseTreeDAGLayer(config)
            self.channel_mixer = self.asdag

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        use_quantized_gates: bool = True,
        use_shift4_act: bool = True,
        record_cache: Optional[bool] = None,
        reset_mask: Optional[torch.Tensor] = None
    ) -> Any:
        if (not x.is_cuda and not return_state and state is None
            and self.channel_mixer_type == "ternary_swiglu"
            and getattr(self.time_mixer, 'proj_type', '') == "monarch"
            and getattr(self.time_mixer, 'rule', 'gla') == "gla"):
            from affine_ai.core.cpp_ops import asdag_cpu_fused_asdag_block
            tm = self.time_mixer
            cm = self.channel_mixer
            return asdag_cpu_fused_asdag_block(
                x,
                self.norm1.scale,
                tm.qkvg_proj.diagonals,
                tm.qkvg_proj.perms,
                tm.qkvg_proj.inv_perms,
                tm.qkvg_proj.bias,
                tm.q_norm.scale,
                tm.k_norm.scale,
                tm.gate_decay.weight,
                tm.gate_decay.bias,
                tm.out_proj.diagonals,
                tm.out_proj.perms,
                tm.out_proj.inv_perms,
                tm.out_proj.bias,
                self.norm2.scale,
                cm.w_gate_val.weight,
                cm.w_down.weight,
                reset_mask
            )

        first_leaf = self.asdag.leaves[0] if self.asdag is not None and self.asdag.leaves else None
        if (not x.is_cuda and not return_state and state is None
            and self.channel_mixer_type == "asdag_tree"
            and x.shape[-1] >= 64
            and getattr(self.time_mixer, 'proj_type', '') == "monarch"
            and getattr(self.time_mixer, 'rule', 'gla') == "gla"
            and self.asdag is not None
            and self.asdag.use_hierarchical_routing
            and self.asdag.top_k == 2
            and first_leaf is not None
            and first_leaf.leaf_mode in ("permutation", "perm")
            and first_leaf.activation == "relu6"
            and not any(len(leaf.secondary_parents) > 0 for leaf in self.asdag.leaves)
            and len(self.asdag.root.secondary_parents) == 0
            and self.asdag.root.scale_perm is not None):
            from affine_ai.core.cpp_ops import asdag_cpu_fused_asdag_tree_block
            from affine_ai.core.ast_dag import ternarize
            tm = self.time_mixer
            leaves = self.asdag.leaves
            w_perm_stack = torch.stack([
                ternarize(leaf.latent_w_perm, self.asdag.threshold_frac, scale=leaf.scale_perm if self.asdag.learnable_scale else None)
                for leaf in leaves
            ], dim=0)
            b_stack = torch.stack([leaf.bias for leaf in leaves], dim=0)
            perms_stack = torch.stack([leaf.perms for leaf in leaves], dim=0)
            inv_perms_stack = torch.stack([leaf.inv_perms for leaf in leaves], dim=0)
            root = self.asdag.root
            return asdag_cpu_fused_asdag_tree_block(
                x,
                self.norm1.scale,
                tm.qkvg_proj.diagonals,
                tm.qkvg_proj.perms,
                tm.qkvg_proj.inv_perms,
                tm.qkvg_proj.bias,
                tm.q_norm.scale,
                tm.k_norm.scale,
                tm.gate_decay.weight,
                tm.gate_decay.bias,
                tm.out_proj.diagonals,
                tm.out_proj.perms,
                tm.out_proj.inv_perms,
                tm.out_proj.bias,
                self.norm2.scale,
                w_perm_stack, perms_stack, inv_perms_stack, b_stack,
                root.latent_w_perm, root.scale_perm, root.bias, root.perms,
                self.asdag.router.hyperplanes, self.asdag.router.biases,
                reset_mask
            )

        # 1. Time Mixer (Monarch GLA State Space)
        time_out, next_state = self.time_mixer(
            self.norm1(x),
            state=state,
            return_state=return_state,
            reset_mask=reset_mask
        )
        x = x + time_out

        # 2. Channel Mixer
        if self.asdag is not None:
            channel_out = self.asdag(
                self.norm2(x),
                record_cache=record_cache,
                use_quantized_gates=use_quantized_gates,
                use_shift4_act=use_shift4_act
            )
        else:
            channel_out = self.channel_mixer(self.norm2(x))
        x = x + channel_out

        if return_state or state is not None:
            return x, next_state
        return x

    def backward_backpressure(
        self,
        upstream_error: torch.Tensor,
        use_sign_backpressure: bool = False
    ) -> torch.Tensor:
        if self.asdag is not None:
            updates, _ = self.asdag.compute_backpressure_updates(
                upstream_error, loss_type="direct", use_sign_backpressure=use_sign_backpressure
            )
            self.asdag.apply_updates(updates, None)
            input_pressure = updates.get("input_pressure", upstream_error).reshape_as(upstream_error)
            return upstream_error + input_pressure
        elif not upstream_error.is_cuda and hasattr(self.channel_mixer, 'w_gate_val'):
            # In the original backward pass, upstream error was pushed through a fused C++
            # backpressure pipeline; that kernel was removed in the decluttering pass.
            # The surviving backpressure tree layer handles this path when present.
            return upstream_error
        return upstream_error


class ASDAGLanguageModel(nn.Module):
    """
    End-to-End ASDAG Language Model (Option B: 100% MatMul-Free Dual-Mixer).
    Architecture:
      Token Embeddings -> Stack of N Dual-Mixer Blocks -> RMSNorm -> LM Head.
      1. Time Mixer: Monarch Gated Linear Associative (GLA) Memory with Zero-MatMul Projections.
      2. Channel Mixer: Ternary BitLinear SwiGLU (Expand=2x) with {-1, 0, +1} Integer Additions.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        num_leaves: int = 16,
        sparsity_ratio: float = 0.9375,
        shift_bits: int = 4,
        top_k: int = 2,
        leaf_mode: str = "permutation",
        num_permutations: int = 4,
        channel_mixer_type: str = "ternary_swiglu",
        use_fp8: bool = True,
        dtype: Any = torch.bfloat16,
        tie_weights: bool = True,
        use_blt: bool = True,
        d_byte: int = 64,
        target_patch_size: int = 16,
        use_mtp: bool = True,
        num_mtp_heads: int = 1,
        mtp_lambda: float = 0.3,
        use_hybrid: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.channel_mixer_type = channel_mixer_type
        self.use_blt = use_blt
        self.use_mtp = use_mtp
        self.num_mtp_heads = num_mtp_heads
        self.mtp_lambda = mtp_lambda
        self.use_hybrid = use_hybrid

        if self.use_hybrid and self.use_blt and self.vocab_size == 256:
            from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
            cfg = TorosHybridConfig(
                dim=d_model,
                d_byte=d_byte if d_byte <= d_model else d_model,
                n_encoder_layers=n_layers,
                n_predictor_layers=max(1, n_layers // 2),
                n_heads=n_heads,
                target_patch_size=target_patch_size,
                channel_mixer_type=channel_mixer_type,
                dtype=dtype
            )
            self.hybrid = TorosHybridLanguageModel(cfg)
            self.blt = self.hybrid
            self.tok_embeddings = self.hybrid.context_encoder.byte_encoder.byte_embed
            self.lm_head = self.hybrid.byte_decoder.lm_head
            self.blocks = self.hybrid.context_encoder.blocks
            self.norm_f = self.hybrid.context_encoder.norm_out
            return

        if self.use_blt and self.vocab_size == 256:
            from affine_ai.models.blt import ASDAGByteLatentModel
            self.blt = ASDAGByteLatentModel(
                vocab_size=vocab_size,
                d_byte=d_byte if d_byte <= d_model else d_model,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                target_patch_size=target_patch_size,
                channel_mixer_type=channel_mixer_type,
                use_mtp=use_mtp,
                num_mtp_heads=num_mtp_heads,
                mtp_lambda=mtp_lambda,
                dtype=dtype
            )
            self.tok_embeddings = self.blt.byte_encoder.byte_embed
            self.lm_head = self.blt.byte_decoder.lm_head
            self.blocks = self.blt.global_blocks
            self.norm_f = self.blt.global_norm
            return

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        self.config = ASDAGConfig(
            dim=d_model,
            num_leaves=num_leaves,
            sparsity_ratio=sparsity_ratio,
            shift_bits=shift_bits,
            top_k=top_k,
            leaf_mode=leaf_mode,
            num_permutations=num_permutations,
            channel_mixer_type=channel_mixer_type,
            use_fp8=use_fp8,
            dtype=dtype
        )

        self.blocks = nn.ModuleList([
            ASDAGBlock(
                self.config,
                n_heads=n_heads,
                layer_idx=i,
                channel_mixer_type=channel_mixer_type
            ) for i in range(n_layers)
        ])

        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def get_default_optimizer(
        self,
        muon_lr: float = 0.03,
        adamw_lr: float = 3e-3,
        muon_momentum: float = 0.95,
        adamw_weight_decay: float = 0.01,
        fused: bool = True
    ) -> Any:
        from affine_ai.optim.muon import HybridMuonAdamW
        return HybridMuonAdamW(
            model=self,
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            muon_momentum=muon_momentum,
            adamw_weight_decay=adamw_weight_decay,
            fused=fused
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_quantized_gates: bool = True,
        use_shift4_act: bool = True,
        record_cache: Optional[bool] = None,
        return_states: bool = False
    ) -> Any:
        if getattr(self, 'hybrid', None) is not None:
            logits, _, _ = self.hybrid(input_ids)
            if return_states:
                return logits, []
            return logits

        if getattr(self, 'blt', None) is not None and not hasattr(self, 'hybrid'):
            logits, _, _ = self.blt(input_ids)
            if return_states:
                return logits, []
            return logits

        x = self.tok_embeddings(input_ids)

        next_states = []
        for idx, block in enumerate(self.blocks):
            layer_state = states[idx] if (states is not None and len(states) > idx) else None
            res = block(
                x,
                state=layer_state,
                return_state=return_states,
                use_quantized_gates=use_quantized_gates,
                use_shift4_act=use_shift4_act,
                record_cache=record_cache
            )
            if return_states or layer_state is not None:
                x, ns = res
                next_states.append(ns)
            else:
                x = res

        x = self.norm_f(x)
        if not x.is_cuda and self.lm_head.weight.dtype == torch.bfloat16:
            logits = F.linear(x.float(), self.lm_head.weight.float()).to(x.dtype)
        else:
            logits = self.lm_head(x)

        if return_states:
            return logits, next_states
        return logits

    def backward_backpressure(
        self,
        logits_error: torch.Tensor,
        use_sign_backpressure: bool = False
    ) -> None:
        error = F.linear(logits_error, self.lm_head.weight.t())
        for block in reversed(self.blocks):
            error = block.backward_backpressure(
                error, use_sign_backpressure=use_sign_backpressure
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        use_quantized_gates: bool = True,
        use_shift4_act: bool = True,
        eos_byte: Optional[int] = 0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if getattr(self, 'hybrid', None) is not None:
            return self.hybrid.generate_with_latent_planning(
                input_ids, max_new_bytes=max_new_tokens, plan_steps=3,
                temperature=temperature, top_k=top_k, top_p=top_p,
                eos_byte=eos_byte, generator=generator
            )

        if getattr(self, 'blt', None) is not None and not hasattr(self, 'hybrid'):
            return self.blt.generate(
                input_ids, max_new_bytes=max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, eos_byte=eos_byte, generator=generator
            )

        B = input_ids.size(0)
        curr_ids = input_ids

        def sample(logits_last):
            from affine_ai.models.hybrid import TorosHybridLanguageModel
            return TorosHybridLanguageModel._sample_next_byte(
                logits_last, temperature, top_k, top_p, generator
            )

        if not use_cache:
            for _ in range(max_new_tokens):
                logits = self.forward(
                    curr_ids,
                    use_quantized_gates=use_quantized_gates,
                    use_shift4_act=use_shift4_act
                )
                next_token = sample(logits[:, -1, :])
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
            return curr_ids

        logits, states = self.forward(
            curr_ids,
            return_states=True,
            use_quantized_gates=use_quantized_gates,
            use_shift4_act=use_shift4_act
        )
        next_token = sample(logits[:, -1, :])
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            next_logits, states = self.forward(
                next_token,
                states=states,
                return_states=True,
                use_quantized_gates=use_quantized_gates,
                use_shift4_act=use_shift4_act
            )
            next_token = sample(next_logits[:, -1, :])
            generated.append(next_token)

        return torch.cat([curr_ids] + generated, dim=1)
