# AXIOM.md — Findings from AXIOM (arXiv:2505.24784) and applicable improvements

Source: Heins et al., "AXIOM: Learning to Play Games in Minutes with Expanding
Object-Centric Models" (VERSES / Friston-lineage active inference, May 2025).
Read 2026-08-31 in the context of this repo's cheapness doctrine
(matmul-free, CPU-first, LPC forward-only training, O(1) inference).

## What AXIOM is

An object-centric active-inference agent that masters 10 simple games in
10k interaction steps with no gradients, no replay buffer, and few parameters.
Four conjugate mixture modules (perception, identity, transitions, relations)
learned online with closed-form variational E/M steps. Structure grows by
truncated stick-breaking priors (posterior mass on a "new component"
pseudo-count α₀ spawns components when evidence demands) and is pruned by
Bayesian model reduction (BMR: merge components whose removal does not
decrease model evidence). Policy selection uses expected free energy
(pragmatic + epistemic/information-gain terms).

## Why it is less alien to this repo than it looks

AXIOM's thesis — tiny models, closed-form online updates, structure that
grows from evidence — is the cheapness doctrine taken to its logical extreme.
LPC is already the philosophical sibling of AXIOM's online variational
learning: local, forward-only, in-place updates, no cross-layer tape.
The mapping below is where the kinship becomes mathematically concrete.

## Salvageable improvements (ranked by fit, evidence-gated like everything else)

### 1. Stick-breaking growth for the tree mixer (highest fit)

AXIOM grows mixture components via a truncated stick-breaking prior where the
final Dirichlet pseudo-count α₀ is the "propensity to spawn a new component";
unexplained evidence (high surprisal under current components) moves posterior
mass onto a fresh component, which is born from the data. Merging is handled
by BMR (below). No growth schedule or rate hyperparameter.

Why we care — this is the principled version of a problem we already hit and
solved heuristically (see benchmarks/README_session_results.md,
"backpressure_scaling_and_growth"): random-init big-K trees starve
(per-leaf visits ~1/K), grown_copy() root-duplication prevents collapse
(static K=1024 0.448 vs grown-to-K 0.277), and growth-phase training needs
sparse_dispatch=False only for from-scratch trees. Our growth signal is
hydraulic gradients + hand-tuned schedules; AXIOM's is posterior inference.

Candidate experiment: per-leaf Dirichlet concentration as the spawn criterion
in backpressure_tree.py — spawn a leaf split when the posterior over its
assignment mass concentrates against the current leaf's ability to explain
its activations. A/B vs grown_copy() at matched budgets on the same toy
targets used for the original growth experiments.

### 2. RLS / closed-form posterior heads for LocalPredictiveHead

AXIOM's one-frame-at-a-time E/M updates replace gradient descent entirely.
Intractable for a 1M-param ternary network — but LocalPredictiveHead
(affine_ai/core/lpc.py) is a linear map h -> logits with an AdamW/Muon step.
A linear head with a Gaussian posterior over its weights is exactly recursive
least squares (RLS): O(d x V) closed-form update per step, no learning-rate
hyperparameter, and — the real prize — a posterior covariance that provides
per-token uncertainty estimates for free. This fits the LPC doctrine
(forward-only, local, cheap) better than the optimizer it would replace,
and unlocks salvage #5.

Candidate experiment: swap LocalPredictiveHead's optimizer for RLS on the
same 1k-step SimpleStories protocol; watch val ppl + step time + the noise
floor. Do NOT assume conjugate beats gradient here — the JEPA lesson applies
to Bayesian ideas exactly as much as to loss functions.

### 3. Type-vs-instance conditioning (iMM analog)

AXIOM shares dynamics across slots via discrete identity codes — "falling" is
one motif whether it is the ball or the enemy — so dynamics are learned
type-specific rather than instance-specific. The text analog: a discrete code
over patch latents so recurring narrative/syntactic patterns ("once upon a
time there was a ___") become types with slots, not thousands of memorized
instances. Relevance to the measured repetition-collapse disease: a model
with type-level structure can represent "I am in repeated-template mode",
which is a prerequisite for representing EXIT from it. Speculative — flagged
as such; no experiment designed yet.

### 4. Bayesian model reduction as the pruning criterion

Our sparsifier drops weights/leaves by magnitude + hydraulic gradients
(RigL-with-hydraulic-growth won the prior ablation). BMR prunes by evidence:
merge components whose removal does not decrease marginal likelihood. The
paper's own ablation shows BMR is load-bearing for generalization ("No BMR"
arm degrades). Candidate: evidence-based merge/split decisions for Monarch
stages, tree leaves, or any future mixture components — as the replacement
criterion in the same ablation harness, not as new machinery.

### 5. Information-gain data selection

AXIOM's exploration maximizes expected free energy (epistemic term). With
RLS posterior covariance from #2, training windows could be ranked by
predicted information gain instead of streaming bytes blindly — a curriculum
for free, from machinery #2 already provides. Cheap to try once #2 exists;
meaningless without it.

## What does NOT transfer (recorded so nobody retries it)

- The object-centric slot machinery itself. Text has no rigid bodies; the
  BLT entropy patcher is already the text-side analog of segmentation.
  Forcing a pixel-style sMM onto bytes is a category error.
- Active-inference planning rollouts: measured dead here (see
  benchmarks/README_session_results.md, repetition_collapse section) —
  our models converge to a loop basin rather than drift; rollout planning
  fixes drift, not convergence. No drift gap exists to close at this scale.
- The headline sample-efficiency numbers. They come from domain priors that
  are true of games (rigid objects, smooth trajectories, sparse collisions)
  and have no byte-level text equivalent. Byte-level language is prior-poor;
  the field's open question is what text's core priors even are. AXIOM's
  numbers promise nothing for ours.

## Standing rules for adopting any of this

Same discipline as jepa_ab and unlikelihood_ab: implement behind a flag,
A/B at matched budget on the standard 1k-step SimpleStories protocol,
record in benchmarks/README_session_results.md, keep only on evidence.
Negative results get documented with the same care.

## Implementation status (2026-09-01)

- Hybrid stripped: TorosHybrid is now gen-only (encoder + decoder + unlikelihood).
  No predictor/local_heads/mask_token/SIGReg/masked-JEPA. JEPA lives standalone
  in jepa.py. Old checkpoints load via strict=False; predictor is stripped from
  exports and trainer handles missing local_heads gracefully.

- Transfer 1 — Stick-breaking growth: `affine_ai/core/growth.py`
  (StickBreakingGrowthController, windowed surprisal -> grow_mass). Pure
  inference-side, no gradients. Behind threshold/alpha0; A/B vs grown_copy
  on toy targets is the next step.

- Transfer 2 — RLS heads: `affine_ai/core/rls_head.py` (RLSPredictiveHead,
  shared P [d,d], forgetting lambda, ridge delta, uncertainty via h^T P h).
  Drop-in parallel to LocalPredictiveHead. Tests cover convergence, ignore
  handling, no-autograd pollution.

- Transfer 3 — Type codebook: `affine_ai/core/type_codebook.py`
  (LatentTypeCodebook, zero-init scale for function-preserving enrichment,
  online mean update + stick-breaking birth). Active types tracked via
  num_types buffer.

- Transfer 4 — BMR pruning: same module (`bmr_merge()` on the codebook).
  Closed-form evidence via distance threshold; handles embedding shift and
  count merging. Tested on duplicate-type collapse.

- Transfer 5 — Info-gain selection: `rank_windows_by_info_gain(head, windows)`
  in type_codebook.py, using RLS posterior variance. Tested ranking validity.

All five are implemented behind flags/default-off where they touch training;
none are wired into the hybrid forward path by default. See tests
`test_rls_head.py` / `test_axiom_transfers.py` for contracts.
