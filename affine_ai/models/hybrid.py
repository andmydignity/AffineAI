"""
Toros-Hybrid: Byte Latent Language Model (System 1 BLT Decoder + Patch Latent Encoder)
====================================================================================
Stripped of JEPA/System-2 scaffolding (predictor, latent local heads, masked
JEPA, SIGReg, mask token). The JEPA auxiliary loss is documented as off by
default and losing to gen-only; the predictor is stripped from exports.
Remaining is a clean patch-latent encoder (byte -> patch -> GLA blocks) plus
causal byte decoder. LPC training (System-1, forward-only per-layer) lives in
affine_ai.core.lpc.LocalPredictiveLanguageModel and is the intended local
learning path for this model when needed.
"""

import math
from dataclasses import dataclass, fields, MISSING
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.ast_dag import ASDAGConfig
from affine_ai.core.norm import RMSNorm
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher, ByteLocalDecoder
from affine_ai.models.jepa import TorosEncoder, TorosJEPAConfig


@dataclass
class TorosHybridConfig:
    dim: int = 136
    d_byte: int = 64
    n_encoder_layers: int = 4
    n_heads: int = 4
    target_patch_size: int = 16
    channel_mixer_type: str = "asdag_tree"
    time_mixer_rule: str = "gla"
    gen_loss_weight: float = 1.0
    # Stripped JEPA/System-2 fields kept for checkpoint compat (ignored):
    # n_predictor_layers, jepa_loss_weight, sigreg_*, mask_* are deprecated.
    # Unlikelihood (objective anti-repetition, not System-2) kept behind flag:
    unlikelihood_weight: float = 0.0
    unlikelihood_n: int = 4
    unlikelihood_window: int = 64
    use_rls_heads: bool = True
    rls_weight: float = 0.1
    rls_forgetting: float = 0.999
    use_type_codebook: bool = True
    type_codebook_max_types: int = 32
    use_growth: bool = True
    growth_alpha0: float = 1.0
    growth_threshold: float = 0.15
    use_bmr: bool = True
    use_info_gain: bool = True
    dtype: Any = torch.float32

    # Back-compat: ignore unknown kwargs from old checkpoints (e.g. n_predictor_layers)
    def __init__(self, **kwargs):
        for f in fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:  # type: ignore
                setattr(self, f.name, f.default_factory())  # type: ignore
            else:
                raise TypeError(f"Missing required field: {f.name}")


# Keep a module-level alias for old imports: TorosHybridConfig fields are superset-compat.
# Old checkpoints may contain jepa_loss_weight etc.; __init__ above ignores them.

class TorosHybridLanguageModel(nn.Module):
    """
    Stripped Toros-Hybrid: patch-latent encoder + causal byte decoder.
    No predictor, no latent local heads, no masked JEPA. Use
    affine_ai.core.lpc.LocalPredictiveLanguageModel for System-1 LPC training
    if local per-layer updates are desired.
    """
    def __init__(self, config: Optional[TorosHybridConfig] = None):
        super().__init__()
        # Filter old config dicts that may contain deprecated System-2 keys
        if config is not None and isinstance(config, dict):
            allowed = {f.name for f in fields(TorosHybridConfig)}
            config = TorosHybridConfig(**{k: v for k, v in config.items() if k in allowed})
        self.config = config or TorosHybridConfig()
        
        jepa_cfg = TorosJEPAConfig(
            dim=self.config.dim,
            d_byte=self.config.d_byte,
            n_encoder_layers=self.config.n_encoder_layers,
            n_heads=self.config.n_heads,
            target_patch_size=self.config.target_patch_size,
            channel_mixer_type=self.config.channel_mixer_type,
            time_mixer_rule=getattr(self.config, 'time_mixer_rule', 'gla'),
            dtype=self.config.dtype
        )
        self.context_encoder = TorosEncoder(jepa_cfg)

        self.byte_decoder = ByteLocalDecoder(
            vocab_size=256,
            d_byte=self.config.d_byte,
            d_model=self.config.dim,
            dtype=self.config.dtype
        )
        self.sos_patch = nn.Parameter(torch.zeros(1, 1, self.config.dim))
        nn.init.normal_(self.sos_patch, mean=0.0, std=0.02)

        if getattr(self.config, 'use_rls_heads', False):
            from affine_ai.core.rls_head import RLSPredictiveHead
            self.rls_heads = nn.ModuleList([
                RLSPredictiveHead(
                    d_model=self.config.dim,
                    vocab_size=256,
                    forgetting=getattr(self.config, 'rls_forgetting', 0.999),
                )
                for _ in range(self.config.n_encoder_layers)
            ])
        if getattr(self.config, 'use_type_codebook', False):
            from affine_ai.core.type_codebook import LatentTypeCodebook
            self.type_codebook = LatentTypeCodebook(
                dim=self.config.dim,
                max_types=getattr(self.config, 'type_codebook_max_types', 32),
            )
        if getattr(self.config, 'use_growth', False):
            from affine_ai.core.growth import StickBreakingGrowthController
            self.growth_controllers = [
                StickBreakingGrowthController(
                    alpha0=getattr(self.config, 'growth_alpha0', 1.0),
                    threshold=getattr(self.config, 'growth_threshold', 0.15),
                )
                for _ in range(self.config.n_encoder_layers)
            ]
        
        if self.config.dtype is not None and self.config.dtype != torch.float32:
            self.to(self.config.dtype)

    def get_default_optimizers(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        use_muon: bool = False,
        muon_lr: float = 0.02
    ) -> List[Any]:
        optimizers = []
        if use_muon:
            from affine_ai.optim.muon import HybridMuonAdamW
            for i, block in enumerate(self.context_encoder.blocks):
                mods = [block]
                if i == 0:
                    mods += [self.context_encoder.byte_encoder, self.context_encoder.patcher]
                optimizers.append(HybridMuonAdamW(
                    nn.ModuleList(mods), muon_lr=muon_lr, adamw_lr=lr, adamw_weight_decay=weight_decay
                ))
            tail_mods = [self.byte_decoder]
            optimizers.append(HybridMuonAdamW(
                nn.ModuleList(tail_mods), muon_lr=muon_lr, adamw_lr=lr, adamw_weight_decay=weight_decay
            ))
            return optimizers

        for i, block in enumerate(self.context_encoder.blocks):
            layer_params = list(block.parameters())
            if i == 0:
                layer_params += list(self.context_encoder.byte_encoder.parameters())
                layer_params += list(self.context_encoder.patcher.parameters())
            optimizers.append(torch.optim.AdamW(layer_params, lr=lr, weight_decay=weight_decay))

        tail_params = list(self.byte_decoder.parameters()) + [self.sos_patch]
        optimizers.append(torch.optim.AdamW(tail_params, lr=lr, weight_decay=weight_decay))
        return optimizers

    def update_target_encoder(self, *args, **kwargs):
        """Deprecated stub: JEPA target encoder removed. No-op for checkpoint compat."""
        return

    def _unlikelihood_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.unlikelihood_weight <= 0.0:
            return logits.new_zeros(())

        B, T, V = logits.shape
        n = self.config.unlikelihood_n
        W = min(self.config.unlikelihood_window, T)

        prev = torch.full((B, T, n - 1), -1, dtype=torch.int64, device=logits.device)
        for k in range(1, n):
            prev[:, k:, n - 1 - k] = targets[:, :T - k]

        repeat = torch.zeros(B, T, dtype=torch.bool, device=logits.device)
        valid_head = torch.arange(T, device=logits.device) >= n - 1
        for dt in range(1, W):
            src = torch.clamp(torch.arange(T, device=logits.device) - dt, min=0)
            prev_shift = prev[:, src]
            match = (prev_shift == prev).all(dim=-1) & valid_head
            repeat |= match

        repeat &= (prev[:, :, 0] >= 0)

        logprobs = F.log_softmax(logits.view(-1, V).float(), dim=-1)
        p_tgt = logprobs.gather(1, targets.view(-1).unsqueeze(1)).squeeze(1).exp()
        one_minus = (1.0 - p_tgt).clamp(min=1e-6)
        ul = -one_minus.log().view(B, T) * repeat.float()
        n_flag = repeat.sum()
        if n_flag == 0:
            return logits.new_zeros(())
        return ul.sum() / n_flag.float()

    def forward(
        self,
        byte_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        B, T = byte_ids.shape
        h_byte, boundary_logits = self.context_encoder.byte_encoder(byte_ids)
        latent_patches, patch_assignments = self.context_encoder.patcher(
            h_byte, boundary_logits, fixed_patch_size=self.config.target_patch_size
        )
        
        hiddens = []
        h_latent = latent_patches
        for block in self.context_encoder.blocks:
            h_latent = block(h_latent)
            hiddens.append(h_latent)
        h_latent = self.context_encoder.norm_out(h_latent)
        if getattr(self.config, 'use_type_codebook', False) and hasattr(self, 'type_codebook'):
            h_latent, _ = self.type_codebook(h_latent)
            if self.training:
                self.type_codebook.update(h_latent.detach())
                if getattr(self.config, 'use_bmr', False) and int(self.type_codebook.num_types.item()) > 1:
                    self.type_codebook.bmr_merge()
        if getattr(self.config, 'use_growth', False) and hasattr(self, 'growth_controllers'):
            for i, block in enumerate(self.context_encoder.blocks):
                cm = getattr(block, 'channel_mixer', None) or getattr(block, 'asdag', None)
                if cm is not None and hasattr(cm, '_last_routing_probs') and getattr(cm, '_last_routing_probs', None) is not None:
                    try:
                        self.growth_controllers[i].update(cm._last_routing_probs)
                    except Exception:
                        pass
        
        causal_latent_patches = torch.cat([self.sos_patch.expand(B, 1, -1), h_latent[:, :-1]], dim=1)
        logits = self.byte_decoder(h_byte, causal_latent_patches, patch_assignments)
        
        loss = None
        metrics = {}

        if targets is not None:
            loss_gen = F.cross_entropy(logits.view(-1, 256), targets.view(-1))
            loss_unl = self._unlikelihood_loss(logits, targets)
            rls_loss = None
            if getattr(self.config, 'use_rls_heads', False) and hasattr(self, 'rls_heads'):
                rls_terms = []
                for h, rls_head in zip(hiddens, self.rls_heads):
                    Mh = h.shape[1]
                    pt = targets[:, ::self.config.target_patch_size][:, :Mh]
                    if pt.shape[1] < Mh:
                        pad = torch.full((pt.shape[0], Mh - pt.shape[1]), -100, device=pt.device, dtype=pt.dtype)
                        pt = torch.cat([pt, pad], dim=1)
                    elif pt.shape[1] > Mh:
                        pt = pt[:, :Mh]
                    logits_rls, _ = rls_head(h)
                    rls_terms.append(F.cross_entropy(logits_rls.view(-1, 256), pt.view(-1), ignore_index=-100))
                    if self.training:
                        rls_head.update(h.detach(), pt)
                if rls_terms:
                    rls_loss = torch.stack(rls_terms).mean()
            loss = loss_gen
            if self.config.unlikelihood_weight > 0.0:
                loss = loss + self.config.unlikelihood_weight * loss_unl
                metrics["loss_unl"] = loss_unl.detach().item()
            if rls_loss is not None:
                loss = loss + self.config.rls_weight * rls_loss
                metrics["loss_rls"] = rls_loss.detach().item()
            metrics.update({
                "loss_total": loss.item(),
                "loss_gen": loss_gen.item(),
                "ppl": math.exp(min(loss_gen.item(), 20.0))
            })

        return logits, loss, metrics

    @torch.no_grad()
    def forward_incremental(
        self,
        byte_ids: torch.Tensor,
        gen_state: Optional[Dict[str, Any]] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        B, T = byte_ids.shape
        assert T >= 1
        P = self.config.target_patch_size
        ce = self.context_encoder
        d_byte = self.config.d_byte
        enc = ce.byte_encoder

        if gen_state is None:
            ref_dtype = enc.byte_embed.weight.dtype
            gen_state = {
                "block_states": [None] * len(ce.blocks),
                "patch_h_byte": torch.zeros(B, 0, d_byte, device=byte_ids.device, dtype=ref_dtype),
                "patch_boundary": torch.zeros(B, 0, 1, device=byte_ids.device, dtype=ref_dtype),
                "conv_hist": torch.zeros(B, 0, d_byte, device=byte_ids.device, dtype=ref_dtype),
                "h_cache": None,
                "n_patches": 0,
            }

        x_new = enc.byte_embed(byte_ids)                                    # [B, T, d_byte]
        hist = gen_state["conv_hist"]
        K = enc.kernel_size
        h_pre_rows = torch.cat([hist, x_new], dim=1)
        x_conv = F.conv1d(
            h_pre_rows.transpose(1, 2), enc.conv.weight, enc.conv.bias,
            padding=K - 1, groups=enc.conv.groups
        ).transpose(1, 2)
        x_conv = x_conv[:, hist.shape[1] : hist.shape[1] + T]              # [B, T, d_byte]
        h_pre = x_new + x_conv
        rms = torch.rsqrt(h_pre.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        h_norm = (h_pre.float() * rms) * enc.norm.scale.float()
        h_byte_new = F.silu(F.linear(h_norm, enc.proj.weight.float(), enc.proj.bias))
        boundary_new = F.linear(
            h_byte_new, enc.boundary_predictor.weight.float(), enc.boundary_predictor.bias.float()
        ).squeeze(-1)
        h_byte_new = h_byte_new.to(x_new.dtype)
        keep = min(K - 1, h_pre_rows.shape[1])
        gen_state["conv_hist"] = h_pre_rows[:, -keep:].detach() if keep > 0 else gen_state["conv_hist"]

        hb = torch.cat([gen_state["patch_h_byte"], h_byte_new], dim=1)      # [B, r+T, d_byte]
        b_acc = torch.cat([gen_state["patch_boundary"], boundary_new.unsqueeze(-1)], dim=1)
        r = gen_state["patch_h_byte"].shape[1]

        j = gen_state["n_patches"] + torch.arange(r, r + T, device=byte_ids.device) // P  # [T]

        n_done = hb.shape[1] // P
        new_latents = []
        for _ in range(n_done):
            chunk_h = hb[:, :P]
            hb = hb[:, P:]
            b_acc = b_acc[:, P:]
            pooled = chunk_h.float().mean(dim=1, keepdim=True)              # [B, 1, d_byte]
            lat = ce.patcher.patch_norm(ce.patcher.patch_proj(pooled.to(ce.patcher.patch_proj.weight.dtype)))
            new_latents.append(lat)
            gen_state["n_patches"] += 1

        gen_state["patch_h_byte"] = hb
        gen_state["patch_boundary"] = b_acc

        if new_latents:
            h = torch.cat(new_latents, dim=1)                               # [B, m, dim]
            for i, block in enumerate(ce.blocks):
                h_out, st = block(h, state=gen_state["block_states"][i], return_state=True)
                gen_state["block_states"][i] = st
                h = h_out
            h = ce.norm_out(h)
            gen_state["h_cache"] = (
                h if gen_state["h_cache"] is None
                else torch.cat([gen_state["h_cache"], h], dim=1)
            )

        h_latent = gen_state["h_cache"]
        grid = (
            torch.cat([self.sos_patch.expand(B, 1, -1), h_latent], dim=1)
            if h_latent is not None
            else self.sos_patch.expand(B, 1, -1)
        )                                                                   # [B, M+1, dim]
        pa = j.clamp(max=grid.shape[1] - 1).unsqueeze(0).expand(B, T).contiguous()
        logits = self.byte_decoder(h_byte_new, grid, pa)
        if return_state:
            return logits, gen_state
        return logits, None

    @staticmethod
    def _sample_next_byte(
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scores = logits.float() / temperature

        if top_k is not None and 0 < top_k < scores.size(-1):
            kth = torch.topk(scores, top_k, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < kth, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
            probs = F.softmax(sorted_scores, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative - probs > top_p
            sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_scores)

        probs = F.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    def generate_with_latent_planning(
        self,
        prompt_bytes: torch.Tensor,
        max_new_bytes: int = 250,
        plan_steps: int = 3,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        eos_byte: Optional[int] = 0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        curr = prompt_bytes.clone()

        gen_state = None
        _, gen_state = self.forward_incremental(curr, gen_state=None, return_state=True)

        generated = 0
        finished = torch.zeros(curr.size(0), 1, dtype=torch.bool, device=curr.device)
        while generated < max_new_bytes:
            logits, gen_state = self.forward_incremental(curr[:, -1:], gen_state, return_state=True)
            last_logits = logits[:, -1, :]

            next_byte = self._sample_next_byte(last_logits, temperature, top_k, top_p, generator)
            if eos_byte is not None:
                next_byte = torch.where(finished, torch.full_like(next_byte, eos_byte), next_byte)
                finished = finished | (next_byte == eos_byte)

            curr = torch.cat([curr, next_byte], dim=1)
            generated += 1
            if eos_byte is not None and finished.all():
                break

        return curr

    def export_inference_state_dict(self) -> Dict[str, torch.Tensor]:
        raw = self.state_dict()
        clean = {}
        for k, v in raw.items():
            if k.startswith("target_encoder") or k.startswith("local_heads") or k.startswith("predictor") or k.startswith("mask_token") or k.startswith("rls_heads") or k.startswith("type_codebook") or k.startswith("growth_controllers"):
                continue
            clean[k] = v
        return clean

    def rank_candidates_by_info_gain(self, candidates: list) -> list:
        if not getattr(self.config, 'use_rls_heads', False) or not hasattr(self, 'rls_heads'):
            raise RuntimeError("use_rls_heads must be enabled for info-gain ranking")
        from affine_ai.core.type_codebook import rank_windows_by_info_gain
        return rank_windows_by_info_gain(self.rls_heads[0], candidates)

    def save_inference_checkpoint(self, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        clean_state = self.export_inference_state_dict()
        payload = {
            "model_state_dict": clean_state,
            "config": self.config,
            "metadata": metadata or {},
            "scaffolding_stripped": True
        }
        torch.save(payload, save_path)

    def save_toros(self, filepath: str, metadata: Optional[Dict[str, Any]] = None, compression_level: int = 19) -> Dict[str, Any]:
        from affine_ai.core.format import save_toros_model
        return save_toros_model(self, filepath, metadata=metadata, compression_level=compression_level)

    @classmethod
    def from_toros(cls, filepath: str, device: str = "cpu", target_dtype: torch.dtype = torch.float32) -> "TorosHybridLanguageModel":
        from affine_ai.core.format import load_toros_model
        model, _ = load_toros_model(filepath, device=device, target_dtype=target_dtype, model_class=cls)
        return model
