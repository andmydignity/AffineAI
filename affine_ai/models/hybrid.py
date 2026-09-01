"""
Toros-Hybrid: Two-Tier Language Model (System 2 JEPA Planner + System 1 BLT Decoder)
===================================================================================
Combines:
1. System 2 (The Brain - Toros-JEPA):
   Abstract latent thought trajectory planning in 1.58-bit MatMul-free representation space.
2. System 1 (The Mouth - BLT Local Byte Decoder):
   Fluent, crisp autoregressive next-byte generation conditioned on planned latent thoughts.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.ast_dag import ASDAGConfig
from affine_ai.core.norm import RMSNorm
from affine_ai.models.blt import ByteLocalEncoder, EntropyPatcher, ByteLocalDecoder
from affine_ai.models.jepa import TorosEncoder, TorosPredictor, TorosJEPAConfig


@dataclass
class TorosHybridConfig:
    dim: int = 136
    d_byte: int = 64
    n_encoder_layers: int = 4
    n_predictor_layers: int = 1
    n_heads: int = 4
    target_patch_size: int = 16
    channel_mixer_type: str = "ternary_swiglu"
    time_mixer_rule: str = "gla"
    # JEPA auxiliary loss is OFF by default: masked-patch JEPA (the best variant,
    # see benchmarks/README_session_results.md) still loses to gen-loss-only by
    # ~2% ppl at 2.2x step cost on SimpleStories/TinyStories A/Bs. Opt in via
    # jepa_loss_weight > 0 for low-data or large-dim regimes or System-2 planning.
    jepa_loss_weight: float = 0.0
    sigreg_weight: float = 1.0
    sigreg_n_sketches: int = 4
    mask_ratio: float = 0.3
    mask_span_len: int = 3
    gen_loss_weight: float = 1.0
    # Unlikelihood training (Welleck et al.): penalize P(token) at positions whose
    # preceding n-gram context already occurred in the window, adding the anti-
    # repetition term that plain CE lacks. 0.0 = off; 0.01-0.1 typical.
    unlikelihood_weight: float = 0.0
    unlikelihood_n: int = 4
    unlikelihood_window: int = 64
    dtype: Any = torch.float32


from affine_ai.core.lpc import LocalPredictiveHead


class TorosHybridLanguageModel(nn.Module):
    """
    Toros-Hybrid Two-Tier Language Model with Native Local Predictive Coding (LPC):
    - System 2 Planner: Latent Thought Predictor (1.58-bit ASDAG JEPA)
    - System 1 Generator: Causal Local Byte Decoder (Sub-Byte SwiGLU)
    - Native LPC: O(1) constant tape memory layer-by-layer forward updates.
    """
    def __init__(self, config: Optional[TorosHybridConfig] = None):
        super().__init__()
        self.config = config or TorosHybridConfig()
        
        # 1. System 2: Context Encoder & Predictor
        jepa_cfg = TorosJEPAConfig(
            dim=self.config.dim,
            d_byte=self.config.d_byte,
            n_encoder_layers=self.config.n_encoder_layers,
            n_predictor_layers=self.config.n_predictor_layers,
            n_heads=self.config.n_heads,
            target_patch_size=self.config.target_patch_size,
            channel_mixer_type=self.config.channel_mixer_type,
            time_mixer_rule=self.config.time_mixer_rule,
            dtype=self.config.dtype
        )
        self.context_encoder = TorosEncoder(jepa_cfg)

        self.predictor = TorosPredictor(jepa_cfg)
        
        # Native Local Predictive Heads for each intermediate ASDAG block
        self.local_heads = nn.ModuleList([
            LocalPredictiveHead(
                d_model=self.config.dim,
                vocab_size=self.config.dim,
                dtype=self.config.dtype
            )
            for _ in range(self.config.n_encoder_layers)
        ])
        
        # 2. System 1: Local Byte Decoder
        self.byte_decoder = ByteLocalDecoder(
            vocab_size=256,
            d_byte=self.config.d_byte,
            d_model=self.config.dim,
            dtype=self.config.dtype
        )
        self.sos_patch = nn.Parameter(torch.zeros(1, 1, self.config.dim))
        nn.init.normal_(self.sos_patch, mean=0.0, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(self.config.d_byte))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        
        if self.config.dtype is not None and self.config.dtype != torch.float32:
            self.to(self.config.dtype)

    def get_default_optimizers(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        use_muon: bool = False,
        muon_lr: float = 0.02
    ) -> List[Any]:
        """
        Creates layer-by-layer optimizers for O(1) memory LPC forward training.
        """
        optimizers = []
        if use_muon:
            from affine_ai.optim.muon import HybridMuonAdamW
            for i, block in enumerate(self.context_encoder.blocks):
                mods = [block, self.local_heads[i]]
                if i == 0:
                    mods += [self.context_encoder.byte_encoder, self.context_encoder.patcher]
                optimizers.append(HybridMuonAdamW(
                    nn.ModuleList(mods), muon_lr=muon_lr, adamw_lr=lr, adamw_weight_decay=weight_decay
                ))
            tail_mods = [self.predictor, self.byte_decoder]
            optimizers.append(HybridMuonAdamW(
                nn.ModuleList(tail_mods), muon_lr=muon_lr, adamw_lr=lr, adamw_weight_decay=weight_decay
            ))
            return optimizers

        for i, block in enumerate(self.context_encoder.blocks):
            layer_params = list(block.parameters()) + list(self.local_heads[i].parameters())
            if i == 0:
                layer_params += list(self.context_encoder.byte_encoder.parameters())
                layer_params += list(self.context_encoder.patcher.parameters())
            optimizers.append(torch.optim.AdamW(layer_params, lr=lr, weight_decay=weight_decay))

        tail_params = (
            list(self.predictor.parameters()) +
            list(self.byte_decoder.parameters()) +
            [self.sos_patch]
        )
        optimizers.append(torch.optim.AdamW(tail_params, lr=lr, weight_decay=weight_decay))
        return optimizers

    def _unlikelihood_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Segment-level unlikelihood (Welleck et al. 2020) at byte level:
        at position t, if the (n-1)-gram preceding the target already occurred in the
        trailing window, subtract log P(target_t) — teaching the model NOT to extend
        repeated n-grams. Plain CE has no such term, which is the objective-level
        root of greedy-decode repetition collapse.
        """
        if self.config.unlikelihood_weight <= 0.0:
            return logits.new_zeros(())

        B, T, V = logits.shape
        n = self.config.unlikelihood_n
        W = min(self.config.unlikelihood_window, T)

        # prev[b, t, :] = the n-1 target tokens preceding position t (-1 padded head)
        prev = torch.full((B, T, n - 1), -1, dtype=torch.int64, device=logits.device)
        for k in range(1, n):
            prev[:, k:, n - 1 - k] = targets[:, :T - k]

        # repeat[b, t] = True if prev[b, t] equals the prev-context of any earlier
        # position t-dt within the window (the n-gram context already occurred)
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

    def _sigreg(self, z: torch.Tensor) -> torch.Tensor:
        """
        SIGReg anti-collapse regularizer (LeJEPA-style): per random 1D projection,
        the sorted sketch (empirical marginal) is matched against the quantiles of
        a fixed isotropic Gaussian N(0, 1). A collapsed embedding sorts to a point
        mass, far from the Gaussian quantiles, so collapse is punished directly —
        no EMA teacher, no separate variance hinge. o(n log n) per sketch.
        """
        flat = z.reshape(-1, z.shape[-1]).float()
        n = flat.shape[0]
        loss = flat.new_zeros(())
        for _ in range(self.config.sigreg_n_sketches):
            v = F.normalize(torch.randn(flat.shape[1], device=flat.device), dim=0)
            s = flat @ v
            g = torch.randn(n, device=flat.device).sort().values
            loss = loss + F.smooth_l1_loss(s.sort().values, g)
        return loss / self.config.sigreg_n_sketches

    def forward(
        self,
        byte_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_shift: int = 16,
        compute_jepa: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """
        Standard Inference and Evaluation Forward Step.
        """
        B, T = byte_ids.shape
        h_byte, boundary_logits = self.context_encoder.byte_encoder(byte_ids)
        latent_patches, patch_assignments = self.context_encoder.patcher(
            h_byte, boundary_logits, fixed_patch_size=self.config.target_patch_size
        )

        # Global ASDAG Transformer Forward
        h_latent = latent_patches
        for block in self.context_encoder.blocks:
            h_latent = block(h_latent)
        h_latent = self.context_encoder.norm_out(h_latent)

        # Strictly Causal Patch Conditioning for Decoder
        causal_latent_patches = torch.cat([self.sos_patch.expand(B, 1, -1), h_latent[:, :-1]], dim=1)
        logits = self.byte_decoder(h_byte, causal_latent_patches, patch_assignments)

        loss = None
        metrics = {}

        if targets is not None:
            loss_gen = F.cross_entropy(logits.view(-1, 256), targets.view(-1))
            loss_unl = self._unlikelihood_loss(logits, targets)

            if compute_jepa and self.config.jepa_loss_weight > 0.0:
                # Masked-patch JEPA (I-JEPA-style, EMA-free). The clean pass above is the
                # free target; masking spans of bytes creates a real information
                # bottleneck (context and target are disjoint), which the roll-shift
                # objective it replaces lacked — there the target was nearly a copy.
                ce = self.context_encoder
                P = self.config.target_patch_size
                M = h_latent.shape[1]
                n_mask = max(1, int(round(M * self.config.mask_ratio)))
                span = self.config.mask_span_len

                mask_starts = []
                taken = torch.zeros(M, dtype=torch.bool, device=h_latent.device)
                tries = 0
                while len(mask_starts) < n_mask and tries < 10 * n_mask:
                    s = int(torch.randint(0, M, (1,)).item())
                    e = min(s + span, M)
                    if not taken[s:e].any():
                        mask_starts.append(s)
                        taken[s:e] = True
                    tries += 1
                mask_patch = taken  # [M] bool: patches held out for prediction

                mask_byte = torch.repeat_interleave(mask_patch, P)[:T]  # [T] bool

                # Masked bytes read as the learned mask token from the embed onward
                x_emb = ce.byte_encoder.byte_embed(byte_ids)                      # [B, T, d_byte]
                mt = self.mask_token.to(x_emb.dtype).expand_as(x_emb)
                x_masked_emb = torch.where(mask_byte.view(1, -1, 1), mt, x_emb)

                conv = ce.byte_encoder.conv
                x_conv = F.conv1d(
                    x_masked_emb.transpose(1, 2), conv.weight, conv.bias,
                    padding=conv.weight.shape[-1] - 1, groups=conv.groups
                ).transpose(1, 2)[:, :T]
                hs = x_masked_emb + x_conv
                rms = torch.rsqrt(hs.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
                hn = (hs.float() * rms) * ce.byte_encoder.norm.scale.float()
                h_byte_m = F.silu(F.linear(hn, ce.byte_encoder.proj.weight.float()))
                h_byte_m = h_byte_m.to(x_emb.dtype)

                # Zero logits -> softmax pooling degenerates to uniform = mean pooling,
                # matching the C++ simd patcher's eval semantics.
                zero_logits = torch.zeros_like(boundary_logits)
                pooled_m, _ = ce.patcher(h_byte_m, zero_logits, fixed_patch_size=P)
                h_m = pooled_m
                for block in ce.blocks:
                    h_m = block(h_m)
                h_m = ce.norm_out(h_m)

                s_pred = self.predictor(h_m)
                sel = mask_patch.to(h_m.dtype).view(1, M, 1)
                diff = s_pred - h_latent.detach()
                loss_inv = (diff * diff * sel).sum() / (sel.sum() * diff.shape[-1]).clamp(min=1.0)
                loss_sigreg = self._sigreg(s_pred)
                loss_jepa = loss_inv + self.config.sigreg_weight * loss_sigreg

                loss = self.config.gen_loss_weight * loss_gen + self.config.jepa_loss_weight * loss_jepa
            else:
                loss = loss_gen
                loss_jepa = torch.tensor(0.0)
                loss_sigreg = torch.tensor(0.0)

            metrics = {
                "loss_total": loss.item(),
                "loss_gen": loss_gen.item(),
                "loss_jepa": float(loss_jepa.detach()),
                "ppl": math.exp(min(loss_gen.item(), 20.0))
            }
            if compute_jepa and self.config.jepa_loss_weight > 0.0:
                metrics["loss_inv"] = loss_inv.detach().item()
                metrics["loss_sigreg"] = loss_sigreg.detach().item()
                metrics["latent_std"] = s_pred.float().std().detach().item()

            if self.config.unlikelihood_weight > 0.0:
                loss = loss + self.config.unlikelihood_weight * loss_unl
                metrics["loss_unl"] = loss_unl.detach().item()

        return logits, loss, metrics

    def forward_lpc_step(
        self,
        byte_ids: torch.Tensor,
        targets: torch.Tensor,
        optimizers: List[Any],
        target_shift: int = 16,
        grad_clip: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes a strictly local, forward-only LPC training step in O(1) constant tape memory.
        """
        B, T = byte_ids.shape
        h_byte, boundary_logits = self.context_encoder.byte_encoder(byte_ids)
        latent_patches, patch_assignments = self.context_encoder.patcher(
            h_byte, boundary_logits, fixed_patch_size=self.config.target_patch_size
        )
        
        with torch.no_grad():
            shifted_bytes = torch.roll(byte_ids, -target_shift, dims=1)
            s_tgt = self.context_encoder(shifted_bytes).detach()
            
        h = latent_patches
        layer_losses = []
        
        # Layer-by-Layer Local Forward + Local Backward
        for i, (block, head) in enumerate(zip(self.context_encoder.blocks, self.local_heads)):
            optimizers[i].zero_grad()
            h_next = block(h)
            
            min_p = min(h_next.shape[1], s_tgt.shape[1])
            local_pred, _ = head(h_next[:, :min_p])
            loss_local = F.smooth_l1_loss(local_pred, s_tgt[:, :min_p])
            loss_local.backward(retain_graph=False)
            
            if grad_clip > 0:
                params = list(block.parameters()) + list(head.parameters())
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                
            optimizers[i].step()
            layer_losses.append(loss_local.item())
            h = h_next.detach()
            
        # Final step: Predictor + Decoder Local Step
        tail_opt = optimizers[-1]
        tail_opt.zero_grad()
        
        h_norm = self.context_encoder.norm_out(h)
        causal_latent_patches = torch.cat([self.sos_patch.expand(B, 1, -1), h_norm[:, :-1]], dim=1)
        s_pred = self.predictor(h_norm)
        min_len = min(s_pred.shape[1], s_tgt.shape[1])
        loss_jepa = F.smooth_l1_loss(s_pred[:, :min_len], s_tgt[:, :min_len])

        if (
            not h_byte.is_cuda
            and h_byte.shape[0] * h_byte.shape[1] == targets.shape[0] * targets.shape[1]
        ):
            from affine_ai.core.cpp_ops import asdag_cpu_blt_2layer_decoder_loss
            bd = self.byte_decoder
            loss_gen = asdag_cpu_blt_2layer_decoder_loss(
                h_byte.detach().float(),
                causal_latent_patches.detach().float(),
                bd.patch_to_byte.weight.float(),
                bd.fusion.weight.float(),
                bd.gate_proj.weight.float(),
                bd.val_proj.weight.float(),
                bd.down_proj.weight.float(),
                bd.lm_head.weight.float(),
                patch_assignments,
                targets.reshape(-1)
            )
        else:
            logits = self.byte_decoder(h_byte.detach(), causal_latent_patches, patch_assignments)
            loss_gen = F.cross_entropy(logits.view(-1, 256), targets.view(-1))

        loss_final = self.config.gen_loss_weight * loss_gen + self.config.jepa_loss_weight * loss_jepa
        loss_final.backward()
        
        if grad_clip > 0:
            tail_params = list(self.predictor.parameters()) + list(self.byte_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(tail_params, grad_clip)
            
        tail_opt.step()
        
        ppl = math.exp(min(loss_gen.item(), 20.0))
        return {
            "loss": loss_final.item(),
            "loss_gen": loss_gen.item(),
            "loss_jepa": loss_jepa.item(),
            "layer_losses": layer_losses,
            "ppl": ppl,
            "bpc": loss_gen.item() / math.log(2)
        }

    @torch.no_grad()
    def forward_incremental(
        self,
        byte_ids: torch.Tensor,
        gen_state: Optional[Dict[str, Any]] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        O(1) incremental forward for autoregressive generation.

        Causality argument (why the current byte's latent is NOT needed):
          The decoder is conditioned on `causal_latent_patches = [sos, h_latent[:, :-1]]`
          (see forward()), i.e. byte at position t is decoded from the latent of the
          *previous* patch. Therefore the freshly appended byte only contributes to a
          not-yet-complete patch, and we can return logits for position t immediately
          while carrying partial-patch accumulators plus per-block GLA states.

        gen_state carries:
          block_states: list of (S, z) GLA states, one per encoder block
          patch_h_byte: [B, r, d_byte] h_byte rows of the partially filled patch
          patch_boundary: [B, r, 1] boundary logits of the partially filled patch
          conv_hist: [B, kernel_size-1, d_byte] conv input history (embed+residual mix)
          h_cache: [B, n_patches, dim] norm_out'ed latents of completed patches
          n_patches: number of completed patches emitted to the encoder so far
        """
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

        # 1. Local byte encoding with conv history stitched in (encoder is causal:
        # embed -> conv(kernel K) -> residual -> norm -> proj -> silu, so only the
        # last K-1 embed rows carry cross-call information).
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

        # 2. Fold new bytes into the partial-patch accumulators
        hb = torch.cat([gen_state["patch_h_byte"], h_byte_new], dim=1)      # [B, r+T, d_byte]
        b_acc = torch.cat([gen_state["patch_boundary"], boundary_new.unsqueeze(-1)], dim=1)
        r = gen_state["patch_h_byte"].shape[1]

        # 3. Byte at global position g = M*P + r + t lives in patch j = g // P and is
        #    conditioned on patch j-1's latent (sos for j == 0) — matching forward()'s
        #    decoder grid [sos, h_latent[:, :-1]] with patch_assignments = g // P.
        j = gen_state["n_patches"] + torch.arange(r, r + T, device=byte_ids.device) // P  # [T]

        # 4. Emit completed patches (mean pooling mirrors the C++ simd patcher)
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

        # 4. Advance encoder blocks over completed patches, carrying GLA state
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
        )                                                                   # [B, M+1, dim]: grid[j] = latent of patch j-1
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
        """
        Samples the next byte [B, 1] from last-position logits [B, V]:
        temperature -> top_k -> nucleus (top_p), then multinomial.
        temperature=0 falls back to greedy argmax.
        """
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
            # keep tokens until cumulative mass first exceeds top_p (always keep rank 0)
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
        """
        System 1 byte-by-byte generation with O(1) incremental state.
        Decoding: temperature scaling -> optional top-k -> optional nucleus
        (top-p) -> multinomial sampling. temperature=0 gives greedy argmax.
        Stops automatically when every sequence has produced eos_byte.
        """
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
        """
        Exports a clean state_dict stripped of all pretraining scaffolding:
        - Discards JEPA Predictor (training-time quality extraction only)
        - Discards Local LPC Auxiliary Heads
        Only keeps the active Context Encoder and Byte Decoder.
        """
        raw = self.state_dict()
        clean = {}
        for k, v in raw.items():
            if k.startswith("target_encoder") or k.startswith("local_heads") or k.startswith("predictor"):
                continue
            clean[k] = v
        return clean

    def save_inference_checkpoint(self, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves an ultra-lightweight deployment checkpoint free of all training scaffolding.
        """
        clean_state = self.export_inference_state_dict()
        payload = {
            "model_state_dict": clean_state,
            "config": self.config,
            "metadata": metadata or {},
            "scaffolding_stripped": True
        }
        torch.save(payload, save_path)

    def save_toros(self, filepath: str, metadata: Optional[Dict[str, Any]] = None, compression_level: int = 19) -> Dict[str, Any]:
        """
        Exports model directly to the ultra-space-efficient Toros Binary Format (.toros).
        """
        from affine_ai.core.format import save_toros_model
        return save_toros_model(self, filepath, metadata=metadata, compression_level=compression_level)

    @classmethod
    def from_toros(cls, filepath: str, device: str = "cpu", target_dtype: torch.dtype = torch.float32) -> "TorosHybridLanguageModel":
        """
        Loads a TorosHybridLanguageModel directly from a .toros binary file.
        """
        from affine_ai.core.format import load_toros_model
        model, _ = load_toros_model(filepath, device=device, target_dtype=target_dtype, model_class=cls)
        return model
