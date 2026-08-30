# Session experiment archive (2025-08-23)

Scripts persisted from interactive debugging; each prints its own results table.
Key findings (all on synthetic targets unless noted):

## tree_vs_dense_matched_params.py
Full-precision low-rank backpressure trees vs dense MLPs at matched STORED params:
MLPs win 2-5x on smooth AND branchy targets. At matched ACTIVE params (~5.5k),
tree 0.21 vs MLP 0.16 (branchy-big) -- 1.3x gap remains. Task saturates: wide
MLP == narrow MLP, so >64-leaf capacity was never exercised by these toys.

## sparsifier_ablation.py / ternary_factor_ablation.py / ternary_ptq_qat_probe.py
- TERNARY SCALE BUG (fixed in core): threshold computed from mean|w| but scale
  discarded -> bare {-1,0,1}; PTQ inflated MSE 400x. TWN alpha-scaling fix landed.
- Low-rank factorization: free when rank >= min(I,O) (always true in our configs);
  direct W_k beats factorized by ~7% via optimization geometry at equal capacity.
- Sparsifier fixes: threshold stats over SURVIVORS (mask no longer deflates ternary),
  sparsity applies to U only, RigL-with-hydraulic-gradient-growth best (+10% vs random).
- N:M structured masks: 1:16 matches unstructured 95% within 2%; kernel-friendly.
- Known anomaly: N:M 1:8 consistently worse than denser configs (unexplained).

## backpressure_scaling_and_growth.py
Random-init big-K trees starve (per-leaf visits ~ 1/K; K=4096 worse than mean).
grown_copy() (function-preserving root-duplication) prevents collapse:
static K=1024 0.448 vs grown-to-K=1024 0.277 (multi-regime target). Quality
plateaus rather than climbs with K on toy targets that saturate past ~64 leaves.
Growth-phase training should use sparse_dispatch=False ONLY for from-scratch
trees; inherited trees do better with hardened gradients (phase-dependent).

## Recipe conclusions carried into the module docs
- Train hard-forward + soft-relaxation routing grads (sparse_dispatch=False),
  mild temp anneal helps branchy tasks; deploy sparse_dispatch=True.
- route_mode='soft' is the exact-backprop regime + self-organization telemetry;
  pure soft training then hard deploy loses to interface mismatch.
