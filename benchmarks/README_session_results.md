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

## jepa_ab (2026-08-30, GPU A/B, harness at /tmp/opencode/jepa_ab.py)
Question: can a JEPA-family auxiliary latent loss extract extra quality from the
same bytes? Answer at TorosHybrid scale (dim=136, 6 blocks, ~1M params): NO.

Protocol: B=16, T=512, 15k steps, lr 3e-3 cosine+warmup, fixed eval batches,
SimpleStories full-dataset memmap (2.24 GB bytes; 123 MB seen = 5.5%), seed 42.
TinyStories A/B (1k steps, 100 MB cap) agreed in direction on every checkpoint.

  arm                     params      ms/step   final val ppl
  no-jepa (gen only)      1,013,009    38.3      3.317
  masked-patch JEPA       1,013,009    85.5      3.388   (+2.1% ppl, 2.2x time)
  roll-shift JEPA (old)   1,013,009    55.1      3.398   (+2.4% ppl, 1.4x time)
  (TinyStories, 1k steps: no-jepa 3.997 vs roll-shift 4.068 vs EMA-old 4.058)

Variants tested:
- roll-shift invariance (predict E(x shifted 64B) from E(x), stop-grad target):
  near-copy task, no information bottleneck; aux gradient competes with gen
  loss for the same tiny ternary encoder -> consistent small regression.
- EMA teacher + VICReg (pre-rework recipe): statistically indistinguishable
  from stop-grad + SIGReg at equal depth (4.058 vs 4.053). EMA machinery
  (extra 721k-param frozen copy + update) bought nothing.
- masked-patch JEPA (learned mask token on ~30% of patch spans, predictor
  reconstructs held-out latents, loss on masked slots vs clean pass): beat
  roll-shift at every checkpoint -> bottleneck hypothesis confirmed -- but
  still lost to no-jepa. Next-byte prediction on raw text is already a hard
  masked-modeling task; little latent-side quality left to extract at this
  scale, and the 2nd encoder pass doubles step cost.

Machinery kept in hybrid.py behind jepa_loss_weight (default 0.0): masked-
patch objective + SIGReg (4-sketch sorted 1D Wasserstein vs fixed N(0,1)) +
EMA-free stop-grad design. Worth revisiting only for: low-data regimes,
larger dim, or System-2 latent planning (predictor wired into generation).
Predictor stays stripped from .toros/inference exports.

## repetition_collapse + unlikelihood_ab (2026-08-30, GPU probes, 1k-step protocol)
Question: TorosHybrid (and siblings) collapse into greedy decode loops
("the stories and the stories..."). Where does it come from and can objective-
side unlikelihood training (Welleck et al. 2020) fix it at the 1k-step budget?

Diagnosis probes (48B real prompt, 256B greedy, SimpleStories val):
  model                 data_ppl  gen self-ppl  distinct-4gram  drift curve
  hybrid 1k             4.02      2.06 flat    0.02             zero (converged)
  hybrid 1k temp 0.7    --        --           0.95             loops masked -> word salad
  LPLM plain 1k         11.4      3.35 flat    0.019            "the the the"
  unl-trained 1k        4.01      2.15 flat    0.08             same basin
  => No drift anywhere: gen self-ppl < data_ppl and flat. The loop is an
  in-distribution attractor (GLA state contraction + marginal backoff under
  ternary capacity pressure), NOT compounding error. This kills the case for
  latent-planning rollouts at this scale (planning fixes drift; we converge).
  Note: prior "it generates fine" impressions came from temp 0.7 + top_k 50 +
  40-80B horizons + EOS truncation: sampling masks the loop, doesn't cure it.
  At 1k steps temp-sampled text is unigram salad either way (likelihood trap).

Unlikelihood A/B (w=0.05, n=4, window=64; same seed/protocol as jepa_ab):
  clean 1k:      ppl 4.02, distinct-4gram 0.02
  unlikelihood:  ppl 3.97 (CE unharmed), distinct-4gram 0.08 (4x more escapes)
  BUT greedy still re-converges to the same "the stories and" basin.
  Mechanism verified: unl trace 0.004 -> 0.70 over 1k steps (self-sharpening as
  the model gains confidence in repeats); flag positions correct on synthetic
  loops (T-7 of T). Failure mode: the penalty bites token-level repeats; the
  attractor lives at the recurrent-state level. Once GLA state saturates, the
  pre-loop context has decayed away and no readout-side token penalty can
  point back out. Penalties: JEPA attacked quality, planning attacked drift,
  unlikelihood attacked token loops -- none touch state contraction.

Next candidate (untested): readout-level neuron fatigue (activity-adaptive
gain on h_final before the LM head, fires-too-much -> suppressed, decays back).
Only proposed mechanism that acts on the representation/state interface and
stays deterministic + O(1) for CPU deployment. Evaluate on >=1k-step
checkpoints at realistic training length (16k) before concluding anything.

Machinery kept in hybrid.py behind config flags, all default-off:
  unlikelihood_weight=0.0 (loss implemented, ~5% step cost when on,
  doubles as a repetition monitor via its trace), jepa_loss_weight=0.0.
