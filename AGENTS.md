# AGENTS.md

PyTorch research library for MatMul-free Adaptive Sparse Tree DAG (ASDAG) models. Python ≥3.9; deps: `torch>=2.0`, `numpy`. No lint/typecheck/format/CI config exists; `pytest` is the only dev tool.

## Commands

```bash
pip install -e .                 # package is installed editable (imports resolve from repo root)
python -m pytest tests/ -x -q     # all tests (triton/GPU tests auto-skip without CUDA)
python -m pytest tests/test_cpp_ops.py::test_asdag_cpu_ops_forward_parity -x  # single test
```

No test runner config; tests are plain pytest functions in `tests/`.

## C++ extension (critical gotcha)

- `affine_ai/core/cpp_ops.py` JIT-compiles `affine_ai/csrc/asdag_cpu_ops.cpp` on first use via `torch.utils.cpp_extension.load` with `-O3 -march=native -fopenmp` (+AVX-512 or AVX2 flags, chosen per host CPU).
- **First test/run is slow** (one-time compile, cached by torch afterward). Don't assume it hung.
- On compile failure the module silently falls back to pure PyTorch (`get_asdag_cpu_ops()` returns `False`) — most layers still work, but fused paths (`asdag_cpu_fused_monarch_gla`, `asdag_cpu_fused_asdag_block`) raise `RuntimeError` instead.
- Importing `affine_ai.core.cpp_ops` sets OpenMP env vars (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`, ...) if unset — expected, don't override them ad hoc.
- `cpp_infer/` is a separate standalone C++ bench/CLI tree (prebuilt binaries checked in); its `Makefile` references source files that no longer exist — build those benchmarks with `g++` directly, not `make`.

## Layout

- `affine_ai/core/` — layers and engines: `ast_dag.py` (ASDAG layer), `cpp_ops.py` (C++ bindings, all have PyTorch fallbacks), `format.py` (custom `.toros` binary checkpoint format; loading compressed files requires the `zstandard` package, **not** in pyproject), `lpc.py` (Local Predictive Coding forward-only training).
- `affine_ai/models/` — `language_model.py` (ASDAG LM, default `channel_mixer_type="ternary_swiglu"`), `blt.py`, `hybrid.py` (TorosHybrid = JEPA encoder + BLT decoder), `jepa.py`, `mtp.py`.
- JEPA is EMA-free (LeJEPA-style): the target is a detached re-encode through the same context encoder, collapse is prevented by SIGReg (sorted 1D Wasserstein vs fixed N(0,1) on random projections). There is no `target_encoder`/`update_target_encoder` — old checkpoints/scripts referencing them are stale. The predictor exists only for training-time quality extraction and is stripped from `.toros`/inference exports.
- **JEPA loss is OFF by default** (`jepa_loss_weight=0.0`): A/B on TinyStories + SimpleStories showed it loses ~2% ppl vs gen-only at 1.4-2.2x step cost. See `benchmarks/README_session_results.md` before re-enabling; don't add latent auxiliary losses without the same A/B.
- Generation: `TorosHybridLanguageModel.generate_with_latent_planning` is O(1) per byte via `forward_incremental` (GLA state stepping + conv-history stitching). It matches full `forward()` logits to ~1e-2 (int8 quantization rounding inside the fused C++ block kernel vs the float state path); don't chase bit-exactness between those two paths in tests.
- The decoder conditions byte t on the latent of patch `t//P - 1` (sos shift); the in-progress tail patch is never emitted during incremental decoding.
- `affine_ai/kernels/` — Triton GPU kernels; `__init__.py` wraps imports in try/except and sets names to `None` on CPU-only boxes, so importing always succeeds.
- `affine_ai/optim/muon.py` — hybrid Muon/AdamW optimizer; `affine_ai/training/trainer.py` — `ASDAGTrainer`.
- Public API is re-exported in `affine_ai/__init__.py`; check it before importing from submodules.

## Data & artifacts

- Training scripts read raw `uint8` byte streams from `data/*.bin` (TinyStories, SmolTalk, FineWeb-EDU). `data/` is ~12 GB and lives only on disk — never commit or read it wholesale.
- Prep/streaming scripts: `scripts/prepare_smoltalk.py`, `scripts/stream_fineweb_edu.py` (require HuggingFace `datasets`); `scripts/prepare_simplestories.py` converts the local SimpleStories parquet corpus (2.24 GB byte stream).
- `AXIOM.md` catalogs salvageable ideas from the AXIOM paper (arXiv:2505.24784) — stick-breaking tree growth, RLS local heads, BMR pruning, info-gain data selection — with experiment sketches and a do-not-transfer list. Consult before designing growth/pruning/uncertainty machinery; anything adopted must pass the same A/B discipline as the JEPA ablations.
- Entry-point training scripts live at repo root (`train_tinystories_700k_lpc_cpu.py`) and in `scripts/` (`train_toros_*.py`, `train_tinystories_*.py`); run them from repo root (they use relative `data/` paths).
- `models/` and `checkpoints/` hold trained `.pt` / `.toros` artifacts. Use `save_toros_model` / `load_toros_model` (2-bit ternary packing, zstd) rather than `torch.save` for shippable checkpoints.
- `benchmarks/README_session_results.md` documents prior experiment findings (quantization bugs, ablations) — consult it before re-running ablation studies.

## Notes

- The README quickstart is stale: it imports `AffineNaryTree` / `BatchedAffineTreeLM`, which no longer exist. The real API is `ASDAGLanguageModel`, `ASDAGTrainer`, `TorosHybridLanguageModel`, etc. (see `affine_ai/__init__.py`).
- CPU is a first-class target (AVX2/AVX-512 SIMD engine); several fused kernels are CPU-only and raise on CUDA tensors — check `x.is_cuda` branches in `cpp_ops.py` before porting code to GPU.
