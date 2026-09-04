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

## delta_rule_mixer_ab (2026-08-31, GPU, 1k-step protocol)
Question: does swapping the GLA accumulate-recurrence for the error-corrective
delta rule (DeltaNet-style, left-projection, unit-normalized keys) fix greedy
decode repetition collapse at the 1k-step budget?

Implementation (associative.py, rule="delta", config flag time_mixer_rule):
  S <- g*S - (g*S) @ (b k k^T) + b k v^T     (k unit-normalized, b=delta_beta=1)
  z <- g*z - b (z.k) k + b k                 (delta-aware denominator)
  y_t = (q S) / (q z + eps);  autograd-exact sequential loop (train + state-carry).
Plumbed via ASDAGConfig.time_mixer_rule -> ASDAGBlock -> TorosHybrid/JEPA configs.
Fused C++ GLA paths correctly bypass delta blocks (gates check rule == "gla").

ALSO FIXED (found during verification, PRE-EXISTING BUG in GLA):
  Sequential state paths did y_t.unsqueeze(1) on a [B,H,D] tensor producing
  [B,1,H,D] slices; cat+transpose scrambled head/position layout for ANY
  multi-token state-carry call (GLA CPU multi-step, GLA CUDA recurrent, and my
  new delta loops). H=2 short-prefix tests masked it; H=4 exposes it. Fixed to
  unsqueeze(2) at 4 sites. This means pre-fix incremental generation with T>1
  chunks silently produced head-scrambled time-mixer outputs.

A/B (same protocol as all 1k arms: B=16, T=512, seed 42, SimpleStories):
  arm        params  ms/step(GPU)  data_ppl  gen self-ppl  distinct-4gram  loop text
  gla        1.01M      ~37          3.987     2.077 flat     0.086          "the stories and..."
  delta      1.01M      ~361         4.013     2.061 flat     0.094          "the stories and..."

VERDICT: delta does NOT fix the loop at this scale/budget. distinct-4gram
0.086 -> 0.094 is noise-level; the loop content is the same corpus-mode
attractor. Both models happily stay in their own loop (self-ppl ~2.0 flat).

Why the oracle's 10x SNR improvement didn't translate: the oracle tested state
mechanics given a FORCED loop input. But the loop is not caused by state
flooding alone -- it is the argmax dynamics + objective (marginal-backoff)
choosing to enter the loop. Delta makes the loop state more survivable/escapable
AFTER entry, but the model still prefers entering it. Escape also requires the
conditional distribution to rank a non-loop continuation above the loop token,
and at 1k steps the conditionals are still corpus-marginal-dominated.

Also notable: delta costs ~10x GPU step time in its sequential-loop training
form (37 -> 361 ms). Production delta training needs the chunked-parallel form
(WYD-representation chunking, as in DeltaNet paper) to be viable; inference
state-stepping is unaffected (O(1) per byte either way). Do NOT adopt delta
based on this A/B; the machinery stays behind time_mixer_rule="delta" for
future long-horizon/longer-training experiments where the entry dynamics may
differ.

Culprit chain, final form: repetition collapse is caused by
(objective: no anti-loop term) + (argmax decode) + (undertrained conditionals
retreating to corpus marginals). State recurrence design (GLA vs delta) modulates
how STUCK a loop gets, not whether the model ENTERS it. The remaining untested
levers from the ranked menu: readout fatigue (state-level escape pressure,
cheap) and decode-side penalties (rep-penalty/n-gram block), all composable.

## depth_vs_width (2026-09-03, CPU, TinyStories, hybrid defaults)

Protocol: fixed ~0.3M budget, B8 T256, AdamW 3e-4 then Muon
(lr 3e-3 + muon_lr 0.02 + clip 1.0), same seed/data, 2k steps (~4M tok),
20-batch val. One 20k-step confirmation pair.

| dim | L | P | AdamW val | Muon val |
|---|---|---|---|---|
| 256 | 2 | 0.334M | 2.39 | ~2.0 (FRAGILE: NaN 1/2 runs) |
| 192 | 4 | 0.296M | 2.80 | 2.12 |
| 160 | 6 | 0.298M | 2.52 | 2.26 |
| 128 | 9 | 0.305M | 2.84 | 2.77 |
| 96 | 14 | 0.318M | 3.22 | 3.97 |

20k-step check (41M tok): 256x2 -> 2.01, 160x6 -> 2.54 (plateaued;
train kept falling, val froze = optimization stall, not capacity).

Findings:
- Depth loses everywhere at this scale, both optimizers. Tree mixer
  gives per-layer expressivity that dense nets need depth for.
- Muon >> AdamW (e.g. 2.12 vs 2.80 at 192x4), but 256x2+Muon is on
  the stability edge (OpenMP reduction-order nondeterminism flips one
  run to NaN). 192x4 is the robust sweet spot.
- Throughput peaks mid-range too (dim192/L4 fastest of the five).

Scaling formula (fitted, TinyStories, 0.24-0.34M only):
P ~= 2e-6 * L * dim^2. Rule: L* in [3,6] (default 4),
dim* = sqrt(P / (2e-6 * L*)), d_byte = dim/2, heads 4.
Predicts dim~285x4 at 0.7M (ran dim384x3 pre-formula instead;
dim285x4 rematch pending). Do NOT trust past ~10M without refit.

## secondary_parents_ab (2026-09-03, CPU, 700k dim384x3, Muon vs AdamW)

Q: do DAG secondary (cross-leaf context) edges matter?
A: No evidence they help; evidence they hurt.
- Growth (step_topology) is UNWIRED: called only in tests, never in
  training. use_growth=True alone changes nothing (controllers update
  stats, topology never moves). Secondary lists stay empty on defaults.
- Grafted-edge A/B (87 hand-added topo-ordered edges, 2k steps):
  no-sec val 1.909 (PPL 6.7) at 30k tok/s;
  grafted-sec NaN at 13k tok/s (fused kernels bail on has_secondary).
- Diagnosis: machinery sound under AdamW (30 steps clean) and under
  Muon@0.005 (60 steps clean). NaN is heat: secondary coupling +
  Muon@0.02 overshoots. Forward starts identical (zero-init gates).
- Verdict: keep secondary/growth OFF and unwired. The DAG is a tree
  until someone shows edges buying PPL. Merging/splitting dynamics
  untested; do not enable without an A/B.

## mixer_ab (2026-09-03, CPU, 700k, full TinyStories 95/5, Muon, 2k steps)

Same params (~0.7M), same L3, same protocol:
- asdag_tree dim384: val 2.067 (PPL 7.9), 30k tok/s
- ternary_swiglu dim176: val 2.346 (PPL 10.4), 83k tok/s

Tree wins quality (-12% loss), swiglu wins speed (2.8x). Confounder:
same params forces different shapes (tree is 2.2x wider -- sparse
efficiency buys width). Tree's quality edge likely IS the width edge.
Verdict: tree stays default (quality-first); swiglu is the documented
fast alternative when throughput matters more than PPL.

Same-shape control (dim176x3, Muon, 2k): tree 0.229M val 2.334
vs swiglu 0.698M val 2.346, both ~81-83k tok/s. Identical quality
AND speed at 1/3 the params. Mechanism cost is nil (shared GLA +
decoder dominate); tree's win is pure param efficiency. The mixer
A/B gap above is 100% width effect.

## delta_vs_gla_swiglu (2026-09-03, CPU, 700k swiglu dim176x3, Muon, 2k)

- gla+swiglu: val 2.327 (PPL 10.3), 82k tok/s
- delta+swiglu: val 2.473 (PPL 11.9), 31k tok/s
Delta loses both ways on CPU: no fused kernel (split-path sequential
scan, 2.7x slower) and worse PPL (+0.15). Matches the old GPU verdict.
Keep rule="gla" default; delta stays opt-in behind
time_mixer_rule for long-horizon experiments only.
