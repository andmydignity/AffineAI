# AffineAI: MatMul-Free Adaptive Sparse Tree DAG (ASDAG) Language Models

`affine_ai` is a PyTorch research library for **byte-level language models** built from
MatMul-free primitives: ternary/dual-ternary/POT-quantized projections, Monarch
permutation-chain mixing, GLA state-space time mixing, and sparse tree-DAG channel mixers.
Training is **Local Predictive Coding (LPC)** — layer-wise, forward-only credit assignment
with O(1) activation memory in depth — with plain global backprop still available.

```python
import torch
from affine_ai import ASDAGLanguageModel

# Byte-level LM: vocab 256, ternary SwiGLU mixer, hybrid patch encoder by default
lm = ASDAGLanguageModel(vocab_size=256, d_model=512, n_layers=8, n_heads=8)
input_ids = torch.randint(0, 256, (4, 128))
logits = lm(input_ids)  # shape: (4, 128, 256)
```

## Install

```bash
pip install -e .          # torch>=2.0, numpy; Python >=3.9
python -m pytest tests/ -x -q
```

Optional: `zstandard` (compressed `.toros` checkpoints), HuggingFace `datasets`
(prep/streaming scripts). Triton GPU kernels need CUDA + a Triton install and
Ampere (sm_80)+ hardware; everything falls back to PyTorch/CPU paths otherwise.

## Training

```python
import numpy as np
from affine_ai import ASDAGLanguageModel, ASDAGTrainer

raw = np.fromfile("data/tinystories.bin", dtype=np.uint8)  # raw byte stream
model = ASDAGLanguageModel(vocab_size=256, d_model=256, n_layers=6, n_heads=4,
                           channel_mixer_type="ternary_swiglu")
trainer = ASDAGTrainer(model=model, train_data=raw[:95_000_000], val_data=raw[95_000_000:],
                       batch_size=16, seq_len=512, max_steps=15000,
                       use_lpc=True, use_muon=True, device="cuda")
stats = trainer.train()  # {"best_val_loss", "best_val_bpc", "best_val_ppl", ...}
```

Entry-point scripts (run from repo root, they use relative `data/` paths):

```bash
python train_tinystories_700k_lpc_cpu.py     # small CPU LPC run (B16 T128 dim136 L6)
python scripts/train_toros_tinystories.py    # hybrid TinyStories, B16 T512
python scripts/train_toros_smoltalk.py
python scripts/train_toros_hybrid_250k_entire_simplestories.py
```

Live HuggingFace streams (no download) work too — the loader yields the same
static `[B, T]` contract, so CUDA Graphs keep working:

```python
from datasets import load_dataset
from affine_ai.data import HFStreamDataLoader
raw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
loader = HFStreamDataLoader(raw, batch_size=16, seq_len=512, text_field="text")
trainer = ASDAGTrainer(model=model, train_data=loader, ..., max_steps=20000, device="cuda")
```

## Key concepts

- **ASDAG layers** (`affine_ai/core/ast_dag.py`): sparse tree-DAG channel mixing with
  ternary STE weights, N:M structured sparsity, top-k leaf routing. The default LM
  instead uses `ternary_swiglu` (same quality at matched shape in our A/Bs, much faster);
  pick `channel_mixer_type="asdag_tree"` for the quality-first sparse path.
- **Time mixer**: Monarch GLA state-space (`affine_ai/core/associative.py`), O(1) per
  byte at generation via state stepping. `time_mixer_rule="delta"` exists but lost its A/B.
- **LPC** (`affine_ai/core/lpc.py`): each block predicts the next token through a local
  head and updates in place — no cross-layer autograd tape, so VRAM is ~4.25 bytes/param
  (≈3.2× more params than backprop on the same GPU in our measurements). Per-layer
  optimizers are `HybridMuonAdamW` (Muon for 2D mats, AdamW for the rest); shared
  encoder params live in a dedicated tail optimizer so async pipelining is race-free.
- **Hybrid byte model** (`affine_ai/models/hybrid.py`): byte encoder + patcher (P=16 default)
  → latent GLA blocks → byte decoder. Generation is O(1)/byte via `forward_incremental`.
- **Checkpoints** (`affine_ai/core/format.py`): `save_toros_model` / `load_toros_model`
  (2-bit ternary packing, zstd) for shippable artifacts instead of `torch.save`.
- **Hardware**: CPU is first-class (AVX2/AVX-512 C++ engine with fused block kernels,
  auto-compiled on first use); on CUDA, fused Triton kernels cover RMSNorm, GLA/Monarch,
  ternary BitLinear, the LPC/CE heads (zero logits allocation), and AdamW. Whole-step
  CUDA Graphs capture LPC steps; `torch.compile` per-block is opt-in (needs sm_89+ for
  FP8 models).

## Layout

- `affine_ai/core/` — layers and engines (`ast_dag`, `lpc`, `bitlinear`, `associative`,
  `cpp_ops`, `cuda_graph`, `format`, `growth`, `rls_head`, `type_codebook`)
- `affine_ai/models/` — `language_model` (ASDAG LM), `hybrid` (byte-patch hybrid),
  `blt`, `mtp`, `qwen35_*`/`qwen38_*` (upcycle targets)
- `affine_ai/kernels/` — Triton GPU kernels (import-safe on CPU-only boxes)
- `affine_ai/optim/muon.py`, `affine_ai/training/trainer.py`, `affine_ai/data/` (loaders)
- `benchmarks/README_session_results.md` — prior A/B findings (read before re-running ablations);
  `AXIOM.md` — salvageable ideas from AXIOM (arXiv:2505.24784) with experiment sketches
- `data/*.bin` — raw uint8 training streams (disk-only, ~12 GB, never commit)

## Notes

- **JEPA is deprecated and off**: `jepa_loss_weight=0.0` default — gen-only beat it by ~2% PPL
  at half the step cost (see benchmarks). Don't add latent auxiliary losses without an A/B.
- The decoder conditions byte *t* on the latent of patch *t//P − 1*; the in-progress tail
  patch is never emitted during incremental decoding.
- Launch CPU training with `OMP_WAIT_POLICY=active` (≈15% faster) and see `source env_cpu.sh`.
