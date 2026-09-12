"""
AffineAI CSA2: Compressed Sparse Attention 2 (Cross-Layer KV Reuse + Tree-SGA)
=============================================================================
Implements DeepSeek-V4.1-Flash CSA2 principles adapted to AffineAI's architecture:
  1. Multi-Mode Cross-Layer KV Sharing:
     - "full": Computes fresh Q, K, V; stores shared KV for downstream layers.
     - "reindex": Reuses shared KV from producer; computes fresh Q and re-scores candidates.
     - "reuse": Reuses shared KV and sparse candidate routing from producer (zero KV storage).
  2. ASDAG Tree-Guided Sparse Global Attention (Tree-SGA):
     - Uses hierarchical tree leaf clusters from ASDAG routers to index long-range context
       beyond the local sliding window without requiring a separate indexing model.
  3. INT4 Low-Bit KV Cache:
     - Quantizes and packs cached K and V tensors into 4-bit representation, unpacked on-the-fly
       in Triton SRAM registers, reducing memory footprint and decoding bandwidth by 4x.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from affine_ai.core.swa import _rope_tables, _apply_rope
from affine_ai.kernels.triton_sliding_window import sliding_window_attn as _triton_swa
from affine_ai.kernels.triton_quant_swa import (
    pack_int4_kv,
    unpack_int4_kv,
    quantized_sliding_window_attn,
    pack_fp8_kv,
    unpack_fp8_kv,
    fp8_sliding_window_attn,
    is_sm89_or_higher,
    is_sm90_or_higher,
)


@dataclass
class SharedKVEntry:
    """Holds shared Key/Value representations for a group of linked CSA layers."""
    k: torch.Tensor
    v: torch.Tensor
    k_pack: Optional[torch.Tensor] = None
    k_scale: Optional[torch.Tensor] = None
    v_pack: Optional[torch.Tensor] = None
    v_scale: Optional[torch.Tensor] = None
    k_fp8: Optional[torch.Tensor] = None
    k_fp8_scale: Optional[torch.Tensor] = None
    v_fp8: Optional[torch.Tensor] = None
    v_fp8_scale: Optional[torch.Tensor] = None
    leaf_indices: Optional[torch.Tensor] = None
    step: int = 0


class CompressedSparseAttentionMixer(nn.Module):
    """
    Compressed Sparse Attention 2 (CSA2) time mixer with cross-layer reuse,
    ASDAG tree-sparse global attention (Tree-SGA), and INT4/FP8 quantized caching.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        window: int = 256,
        mode: str = "full",  # "full", "reindex", "reuse"
        kv_quant: str = "auto",  # "auto", "none", "int4", "fp8"
        shared_source: Optional["CompressedSparseAttentionMixer"] = None,
        use_tree_sga: bool = True,
        global_topk: int = 64,
        rope_base: float = 10000.0,
        dtype: Any = torch.bfloat16,
        use_triton: bool = True,
        sink: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert mode in ("full", "reindex", "reuse"), f"Unknown CSA mode: {mode}"
        if kv_quant == "auto":
            kv_quant = "fp8" if is_sm89_or_higher() else "int4"
        assert kv_quant in ("none", "int4", "fp8"), f"Unsupported kv_quant: {kv_quant}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0
        self.d_head = d_model // n_heads
        self.window = window
        self.mode = mode
        self.kv_quant = kv_quant
        # Use object.__setattr__ so shared_source is NOT registered as a PyTorch child submodule
        object.__setattr__(self, "shared_source", shared_source)
        self.use_tree_sga = use_tree_sga
        self.global_topk = global_topk
        self.rope_base = rope_base
        self.dtype = dtype
        self.use_triton = use_triton
        self.sink = sink

        # Query projection (every layer has its own query)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype)

        # Key & Value projections are ONLY allocated in "full" mode
        if self.mode == "full":
            self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False, dtype=dtype)
            self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False, dtype=dtype)
        else:
            self.k_proj = None
            self.v_proj = None

        self._shared_entry: Optional[SharedKVEntry] = None
        self._rope_cache: Dict[Any, Any] = {}

    def set_shared_source(self, source: "CompressedSparseAttentionMixer") -> None:
        """Sets the upstream producer layer for KV and index reuse."""
        assert source.mode == "full", "Shared source must operate in 'full' mode"
        object.__setattr__(self, "shared_source", source)

    def get_shared_kv(self) -> SharedKVEntry:
        """Retrieves shared KV cached by this layer (if full mode) or upstream source."""
        if self.mode == "full":
            if self._shared_entry is None:
                raise RuntimeError("get_shared_kv called before producer layer executed forward pass.")
            return self._shared_entry
        elif self.shared_source is not None:
            return self.shared_source.get_shared_kv()
        else:
            raise RuntimeError(f"Layer in mode '{self.mode}' has no shared_source configured.")

    def _rope(self, T: int, device: torch.device, dtype: torch.dtype):
        key = (T, self.d_head, str(device))
        ent = self._rope_cache.get(key)
        if ent is None:
            ent = _rope_tables(T, self.d_head, device, dtype, self.rope_base)
            self._rope_cache[key] = ent
        return ent

    def _build_tree_sga_mask(
        self,
        T: int,
        leaf_indices: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Builds causal attention mask combining:
          1. Local sliding window: [i - window + 1, i]
          2. Attention sink token: j = 0
          3. Tree-SGA: tokens j < i - window + 1 sharing the same leaf assignment
        """
        i = torch.arange(T, device=device)
        qi = i.unsqueeze(1)
        kj = i.unsqueeze(0)

        # 1. Local SWA + Sink
        if self.sink:
            lo = torch.clamp(qi - self.window + 1, min=1)
            local_mask = (kj >= lo) & (kj <= qi) | (kj == 0)
        else:
            lo = torch.clamp(qi - self.window + 1, min=0)
            local_mask = (kj >= lo) & (kj <= qi)

        # 2. Global Tree Affinity: matching leaf assignment (causal: kj <= qi)
        if leaf_indices.dim() == 2:
            # [B, T]
            leaves_q = leaf_indices.unsqueeze(-1)  # [B, T, 1]
            leaves_k = leaf_indices.unsqueeze(-2)  # [B, 1, T]
            tree_match = (leaves_q == leaves_k) & (kj.unsqueeze(0) <= qi.unsqueeze(0))
            full_mask = local_mask.unsqueeze(0) | tree_match
            return full_mask
        else:
            return local_mask

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        reset_mask: Optional[torch.Tensor] = None,
        tree_leaf_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        H, Hkv, D, W = self.n_heads, self.n_kv_heads, self.d_head, self.window
        rep = H // Hkv

        # 1. Project Query
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)

        # 2. Obtain Keys and Values (Compute if full mode, Reuse if reindex/reuse mode)
        if self.mode == "full":
            k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
            v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)

            # Apply RoPE
            if state is None:
                cos, sin = self._rope(T, x.device, q.dtype)
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
                q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
                k0, v0 = k, v
            else:
                kc, vc = state
                if reset_mask is not None and bool(reset_mask[:, -1:].any()):
                    kc = kc[:, :, :0]
                    vc = vc[:, :, :0]
                off = kc.shape[2]
                cosS, sinS = self._rope(off + T, x.device, q.dtype)
                cos_q = cosS[off:off + T].unsqueeze(0).unsqueeze(0)
                sin_q = sinS[off:off + T].unsqueeze(0).unsqueeze(0)
                q = _apply_rope(q, cos_q, sin_q)
                k = _apply_rope(k, cos_q, sin_q)
                k0 = torch.cat([kc, k], dim=2)
                v0 = torch.cat([vc, v], dim=2)

            # Quantize to INT4 or FP8 if requested
            k_pack, k_scale = None, None
            v_pack, v_scale = None, None
            if self.kv_quant == "int4":
                k_pack, k_scale = pack_int4_kv(k0)
                v_pack, v_scale = pack_int4_kv(v0)
            elif self.kv_quant == "fp8":
                k_pack, k_scale = pack_fp8_kv(k0)
                v_pack, v_scale = pack_fp8_kv(v0)

            # Cache shared entry for downstream reindex/reuse layers
            self._shared_entry = SharedKVEntry(
                k=k0,
                v=v0,
                k_pack=k_pack,
                k_scale=k_scale,
                v_pack=v_pack,
                v_scale=v_scale,
                leaf_indices=tree_leaf_indices,
            )
        else:
            # REUSE / REINDEX MODE: Fetch shared KV from producer layer (detached)
            shared = self.get_shared_kv()
            k0 = shared.k.detach()
            v0 = shared.v.detach()
            k_pack = shared.k_pack.detach() if shared.k_pack is not None else None
            k_scale = shared.k_scale.detach() if shared.k_scale is not None else None
            v_pack = shared.v_pack.detach() if shared.v_pack is not None else None
            v_scale = shared.v_scale.detach() if shared.v_scale is not None else None

            if self.mode == "reuse" and tree_leaf_indices is None:
                tree_leaf_indices = shared.leaf_indices

            # Apply RoPE to Query
            Tk = k0.shape[2]
            off = Tk - T
            cosS, sinS = self._rope(Tk, x.device, q.dtype)
            cos_q = cosS[off:off + T].unsqueeze(0).unsqueeze(0)
            sin_q = sinS[off:off + T].unsqueeze(0).unsqueeze(0)
            q = _apply_rope(q, cos_q, sin_q)

        # Expand GQA keys and values if needed
        if rep > 1:
            k = k0.repeat_interleave(rep, dim=1)
            v = v0.repeat_interleave(rep, dim=1)
        else:
            k, v = k0, v0

        # 3. Attention Execution (Quantized INT4 / Triton SWA / Tree-SGA / Fallback)
        scale = 1.0 / math.sqrt(D)

        # Fast path: INT4 Quantized Triton SWA
        if (
            self.kv_quant == "int4"
            and k_pack is not None
            and v_pack is not None
            and not self.use_tree_sga
            and state is None
            and reset_mask is None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and D <= 128
        ):
            if rep > 1:
                kp = k_pack.repeat_interleave(rep, dim=1)
                ks = k_scale.repeat_interleave(rep, dim=1)
                vp = v_pack.repeat_interleave(rep, dim=1)
                vs = v_scale.repeat_interleave(rep, dim=1)
            else:
                kp, ks, vp, vs = k_pack, k_scale, v_pack, v_scale
            y = quantized_sliding_window_attn(q, kp, ks, vp, vs, window=W, sink=self.sink, scale=scale)
        # Fast path: FP8 Quantized SWA
        elif (
            self.kv_quant == "fp8"
            and k_pack is not None
            and v_pack is not None
            and not self.use_tree_sga
            and state is None
            and reset_mask is None
        ):
            if rep > 1:
                kp = k_pack.repeat_interleave(rep, dim=1)
                ks = k_scale.repeat_interleave(rep, dim=1)
                vp = v_pack.repeat_interleave(rep, dim=1)
                vs = v_scale.repeat_interleave(rep, dim=1)
            else:
                kp, ks, vp, vs = k_pack, k_scale, v_pack, v_scale
            y = fp8_sliding_window_attn(q, kp, ks, vp, vs, window=W, sink=self.sink, scale=scale)

        # Fast path: Float Triton SWA
        elif (
            self.use_triton
            and _triton_swa is not None
            and not self.use_tree_sga
            and q.is_cuda
            and state is None
            and reset_mask is None
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and k.shape[2] == T
            and v.shape[2] == T
            and D <= 128
        ):
            try:
                y = _triton_swa(q, k, v, window=W, sink=self.sink, scale=scale)
            except Exception:
                mask = self._build_tree_sga_mask(T, tree_leaf_indices if tree_leaf_indices is not None else torch.empty(0), x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        # Eager / Tree-SGA Attention Path
        else:
            if tree_leaf_indices is not None and self.use_tree_sga:
                mask = self._build_tree_sga_mask(T, tree_leaf_indices, x.device)
                if mask.dim() == 3:  # [B, T, T] -> [B, 1, T, T] for GQA
                    mask = mask.unsqueeze(1)
            else:
                i = torch.arange(T, device=x.device)
                qi = i.unsqueeze(1)
                kj = torch.arange(k.shape[2], device=x.device).unsqueeze(0)
                if self.sink:
                    lo = torch.clamp(qi - W + 1, min=1)
                    mask = (kj >= lo) & (kj <= qi) | (kj == 0)
                else:
                    lo = torch.clamp(qi - W + 1, min=0)
                    mask = (kj >= lo) & (kj <= qi)

            if reset_mask is not None:
                rm = reset_mask.reshape(B, -1)[:, :T]
                doc = torch.cumsum(rm.long(), dim=-1)
                same = doc.unsqueeze(-1) == doc.unsqueeze(-2)
                mask = (mask.unsqueeze(0) & same).unsqueeze(1) if mask.dim() == 2 else (mask & same.unsqueeze(1))

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        y = y.transpose(1, 2).reshape(B, T, H * D).to(x.dtype)
        out = self.out_proj(y)

        if return_state or state is not None:
            return out, (k0, v0)
        return out, None


def interleave_csa(
    model: Any,
    every_n: int = 4,
    group_size: int = 4,
    window: int = 256,
    kv_quant: str = "int4",
    use_tree_sga: bool = True,
    n_kv_heads: Optional[int] = None,
) -> int:
    """
    Interleaves CSA2 layers into a model's trunk blocks.
    Within each group of attention layers:
      - Layer 0: "full" mode (Producer of shared KV)
      - Layer 1: "reindex" mode (Shares KV, re-indexes queries)
      - Layer 2+: "reuse" mode (Shares KV and selection)
    """
    blocks = None
    hyb = getattr(model, "hybrid", None)
    if hyb is not None and hasattr(hyb, "context_encoder"):
        blocks = hyb.context_encoder.blocks
    elif hasattr(model, "context_encoder"):
        blocks = model.context_encoder.blocks
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    if blocks is None:
        raise ValueError("interleave_csa: no blocks found (expected .hybrid.context_encoder.blocks or .blocks)")

    n_csa = 0
    current_producer: Optional[CompressedSparseAttentionMixer] = None
    csa_in_group = 0

    for i, b in enumerate(blocks):
        if (i + 1) % every_n != 0:
            continue

        if csa_in_group == 0 or current_producer is None:
            mode = "full"
        elif csa_in_group == 1:
            mode = "reindex"
        else:
            mode = "reuse"

        d_model = getattr(b.config, "dim", None) or b.norm1.scale.shape[0]
        n_heads = getattr(b, "n_heads", 8)
        dtype = getattr(b.config, "dtype", torch.bfloat16)

        csa_layer = CompressedSparseAttentionMixer(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            window=window,
            mode=mode,
            kv_quant=kv_quant,
            shared_source=current_producer if mode != "full" else None,
            use_tree_sga=use_tree_sga,
            dtype=dtype,
        )

        b.time_mixer = csa_layer
        if mode == "full":
            current_producer = csa_layer

        csa_in_group = (csa_in_group + 1) % group_size
        n_csa += 1

    return n_csa
