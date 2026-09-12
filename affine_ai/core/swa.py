"""
Sliding Window Attention (SWA) time mixer.

Interleaved with Monarch-GLA blocks (see interleave_swa), SWA restores exact
pairwise lookup inside a local window: the mechanism linear recurrences lack
(copying, associative recall, precise coreference). Matches the
NativeASDAGAssociativeMixer call convention so it drops into ASDAGBlock as
time_mixer: forward(x, state=None, return_state=False, reset_mask=None).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

try:
    from affine_ai.kernels.triton_sliding_window import sliding_window_attn as _triton_swa
except Exception:
    _triton_swa = None


def _rope_tables(T: int, D: int, device: torch.device, dtype: torch.dtype,
                 base: float = 10000.0):
    d_rot = D - (D % 2)
    if d_rot <= 0:
        return torch.ones((T, 0), device=device, dtype=dtype), torch.zeros((T, 0), device=device, dtype=dtype)
    inv = 1.0 / (base ** (torch.arange(0, d_rot, 2, device=device).float() / d_rot))
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]
    d_rot = cos.shape[-1] * 2
    if d_rot == 0:
        return x
    if d_rot == D:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return torch.stack([o1, o2], dim=-1).flatten(-2)
    x_rot = x[..., :d_rot]
    x_pass = x[..., d_rot:]
    x1 = x_rot[..., ::2]
    x2 = x_rot[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    o_rot = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([o_rot, x_pass], dim=-1)


class SlidingWindowAttentionMixer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        window: int = 256,
        rope_base: float = 10000.0,
        dtype: Any = torch.bfloat16,
        use_triton: bool = True,
        sink: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0
        self.d_head = d_model // n_heads
        self.window = window
        self.rope_base = rope_base
        self.dtype = dtype
        self.use_triton = use_triton
        self.sink = sink
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self._rope_cache: Dict[Any, Any] = {}

    def _rope(self, T: int, device: torch.device, dtype: torch.dtype):
        key = (T, self.d_head, str(device))
        ent = self._rope_cache.get(key)
        if ent is None:
            ent = _rope_tables(T, self.d_head, device, dtype, self.rope_base)
            self._rope_cache[key] = ent
        return ent

    def _sliding_mask(self, T: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(T, device=device)
        if self.sink:
            qi = i.unsqueeze(1)  # [T,1] query
            kj = i.unsqueeze(0)  # [1,T] key
            lo = torch.clamp(qi - self.window + 1, min=1)
            window_mask = (kj >= lo) & (kj <= qi)
            sink_mask = (kj == 0)
            return window_mask | sink_mask
        dist = i.unsqueeze(1) - i.unsqueeze(0)
        return (dist >= 0) & (dist < self.window)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        H, Hkv, D, W = self.n_heads, self.n_kv_heads, self.d_head, self.window
        rep = H // Hkv
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        if state is not None:
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
            Tk = k0.shape[2]
            qi = off + torch.arange(T, device=x.device)
            kj = torch.arange(Tk, device=x.device)
            if self.sink:
                lo = torch.clamp(qi.unsqueeze(1) - W + 1, min=1)
                window_mask = (kj.unsqueeze(0) >= lo) & (kj.unsqueeze(0) <= qi.unsqueeze(1))
                sink_mask = (kj.unsqueeze(0) == 0)
                mask = window_mask | sink_mask
            else:
                in_window = kj.unsqueeze(0) > (qi - W).unsqueeze(1)
                mask = (kj.unsqueeze(0) <= qi.unsqueeze(1)) & in_window
        else:
            cos, sin = self._rope(T, x.device, q.dtype)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
            k0, v0 = k, v
            mask = self._sliding_mask(T, x.device)
            if reset_mask is not None:
                rm = reset_mask.reshape(B, -1)[:, :T]
                doc = torch.cumsum(rm.long(), dim=-1)
                same = doc.unsqueeze(-1) == doc.unsqueeze(-2)
                mask = (mask.unsqueeze(0) & same).unsqueeze(1)

        if rep > 1:
            k = k0.repeat_interleave(rep, dim=1)
            v = v0.repeat_interleave(rep, dim=1)
        else:
            k, v = k0, v0
        # Triton fast path: CUDA, no state cache (equal T), no reset_mask, sink consistent, dtype supported
        use_triton_path = (
            self.use_triton
            and _triton_swa is not None
            and q.is_cuda
            and state is None
            and reset_mask is None
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and k.dtype == q.dtype
            and v.dtype == q.dtype
            and k.shape[2] == T
            and v.shape[2] == T
            and self.d_head <= 128
        )
        if use_triton_path:
            try:
                scale = 1.0 / math.sqrt(self.d_head)
                y = _triton_swa(q, k, v, window=W, sink=self.sink, scale=scale)
            except Exception:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, H * D).to(x.dtype)
        out = self.out_proj(y)
        if return_state or state is not None:
            return out, (k0, v0)
        return out, None


def interleave_swa(model: Any, every_n: int = 6, window: int = 256,
                   n_kv_heads: Optional[int] = None) -> int:
    blocks = None
    hyb = getattr(model, "hybrid", None)
    if hyb is not None and hasattr(hyb, "context_encoder"):
        blocks = hyb.context_encoder.blocks
    elif hasattr(model, "context_encoder"):
        blocks = model.context_encoder.blocks
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    if blocks is None:
        raise ValueError("interleave_swa: no blocks found (expected .hybrid.context_encoder.blocks or .blocks)")
    n = 0
    for i, b in enumerate(blocks):
        if (i + 1) % every_n != 0:
            continue
        tm = getattr(b, "time_mixer", None)
        if tm is None or isinstance(tm, SlidingWindowAttentionMixer):
            continue
        ref = next(tm.parameters(), None)
        dev = ref.device if ref is not None else torch.device("cpu")
        dt = ref.dtype if ref is not None else torch.bfloat16
        dm = getattr(tm, "d_model", None) or getattr(getattr(b, "config", None), "dim", 512)
        nh = getattr(tm, "n_heads", 8)
        new = SlidingWindowAttentionMixer(d_model=dm, n_heads=nh, n_kv_heads=n_kv_heads,
                                          window=window, dtype=dt).to(dev)
        b.time_mixer = new
        n += 1
    return n
