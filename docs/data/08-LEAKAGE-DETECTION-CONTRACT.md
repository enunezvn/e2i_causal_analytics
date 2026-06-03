# 08 — Leakage Detection Contract (Tier-0 Adaptive Temporal Validity)

This is the reader-facing companion to the Tier-0 leakage detector. It explains
how the adaptive temporal-validity ensemble decides whether a feature is a leak,
how a recognized cohort's declared-safe (pre-index) features earn full immunity,
and the operational caveats an analyst must know before trusting the console
output.

The behavior is implemented in:

- `src/data/adversarial_leakage.py` — the Layer-3 statistical discriminator
  (permutation-baseline z-score, the Benjamini-Hochberg FDR confident set, the
  permutation feasibility floor).
- `src/data/manifests/__init__.py` / `src/data/manifests/resolution.py` — the
  per-cohort `FeatureContract` registry and the cohort-identity -> manifest-source
  resolver (Layer-1 / Layer-5 opt-in).
- `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py` — the
  ensemble orchestrator (the Step 1-6 ladder, declared-safe immunity, severity
  recompute).
- `src/agents/ml_foundation/data_preparer/nodes/leakage_remediation.py` — the
  LLM/rule remediation pass (and the companion declared-safe strip).
- `src/agents/ml_foundation/data_preparer/nodes/feast_registrar.py` — the Feast
  freshness gate / advisory.

When this doc and the code disagree, **the code is the source of truth** — every
value below is transcribed from it, but it is the authority. See also
[Model Success Criteria & QC Gates](../model_success_criteria.md) for how the
remediated feature set is then judged at training time.

---

## Core principle: only the temporal contract knows a leak

A correlation, structural, or FDR detector measures **how strongly a feature
predicts the target**. It **cannot** distinguish a strong, legitimate PRE-INDEX
predictor from a genuine leak — both look identical to a statistical test. Only
the **temporal contract** (a `FeatureContract` whose `knowable_at` is at or
before the index/prediction anchor) can certify a feature as temporally
admissible.

Two corollaries the design leans on heavily:

- The deterministic structural checks (`zero_variance_within_class`,
  `perfect_class_separation`) **false-fire** on sparse, rare-event columns — a
  column that is constant within the tiny minority class looks "tautological"
  even when it is a legitimate pre-index clinical predictor.
- Therefore a manifest-declared pre-index feature is governed by the **contract**,
  not the statistics. This is the basis of declared-safe immunity (below).

---

## The ensemble precedence ladder

`adaptive_validity_check` runs a per-feature ladder. Layer 1 is metadata-only and
can flag any column; Layers 3+ require a numeric AUC. The steps
(`adaptive_validity_check.py` module docstring, "Per-layer ordering"):

| Step | Layer | What it does |
|------|-------|--------------|
| 1 | Layer 1 (manifest contract) | Every column checked against its `FeatureContract`. A post-index contract short-circuits with `severity=high`, `decided_by="layer_1"`. |
| 2 | Layer 3 (permutation discriminator) | `compute_adversarial_score` per numeric feature -> `z_score`, `actual_auc`, `null_mean`, `null_std`, plus-one `p_value`. |
| 3 | HBLP severity classification | `hblp_classify`: 3a z-band -> `severity_pre_joint_check`; 3b the issue-#194 joint `|delta_AUC|` clamp may force `info`. |
| 4 | Layer 3 ablation (opt-in) | `compute_feature_ablation` combined by MAX-rule; can ESCALATE info->moderate/high, never downgrade. Off by default. |
| 5 | Layer 4 (LLM causal-role) | Fires on `severity_pre_joint_check`; auditor only, off by default. |
| 6 | EnsembleVoter | Renders the final verdict from the precedence ladder. |

The voter's precedence (module docstring Step 6):

```
Layer 1 high veto  ->  Adversarial high veto  ->  KG-contradictory abstain
  ->  LLM path  ->  Adversarial-moderate -> review  ->  no-signal abstain
```

A `None` / unrecognized manifest source means Layer 1 contributes nothing for
that run, and decisioning falls through to the Layer-3 statistical path
(`lookup_feature_contract` returns `None`).

---

## The sigma-band thresholds

The static z-band lives in `hblp_classify` (constants `HIGH_Z = 5.0`,
`MODERATE_Z = 3.0`):

| z-score band | severity | remediation |
|--------------|----------|-------------|
| `z > 5σ` | high | drop (auto-flag) |
| `3σ < z ≤ 5σ` | moderate | ambiguous -> Layer 4 / review |
| `z ≤ 3σ` | info | keep |

The z-score is computed on the **folded** AUC scale (`max(auc, 1-auc)`), so a
strongly anti-correlated feature (raw AUC near 0) is treated as suspicious as a
strongly correlated one.

### HBLP variance inflation (small-N protection)

At low positive counts the permutation null variance scales as `~1/sqrt(n_pos)`,
so a fixed 5σ over-flags. `hblp_effective_z_threshold` inflates the band by
`sqrt(reference_n / n_positives)` (reference N = 50,
`T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N`). It only ever **relaxes** (never
tightens below the base 5σ). A Layer-1 declared-safe feature additionally gets a
`1.5x` multiplier (`T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER`) — encoding the
prior that a contract-cleared feature needs **stronger** statistical evidence to
be reclassified as a leak (so 5σ -> 7.5σ at N ≥ 50).

### Issue-#194 joint `|delta_AUC|` clamp

The z-band alone over-flags benign weak predictors at large N (the null std
shrinks per the CLT). The fix is a JOINT check:

```
severity ∈ {moderate, high}  ⇔  (z > k)  AND  (|delta_AUC| > epsilon)
```

with `delta_AUC = actual_auc - null_mean` on the folded scale and
`epsilon = LAYER5_DELTA_AUC_FLOOR_DEFAULT = 0.10`. When `|delta_AUC| ≤ 0.10`,
severity is forced to `info` even if z cleared the band. The joint check fires
only when `delta_auc` is supplied and finite (legacy z-only callers see legacy
behavior). One-sided escape: a degenerate null (`null_std=0`) returns `z=+inf`;
when `z=+inf` AND `|delta_AUC| > 0.10`, severity is `high` (a real
deterministic leak); `z=-inf` / `NaN` still fall to `info`.

---

## Layer-3 FDR confident set (the auto-fire driver)

By default the static `z > 5σ` HIGH tier is **replaced** by a cohort-adaptive
Benjamini-Hochberg confident set over the per-feature plus-one permutation
p-values. A feature is a *confident* leak only when BOTH hold
(`fdr_confident_set`):

1. its plus-one p-value clears Benjamini-Hochberg at FDR `q`
   (`benjamini_hochberg`), AND
2. `|effect| > effect_floor` (the issue-#194 actionable bar — a BH-significant
   feature with a tiny effect is the "ambiguous interior" -> review, not dropped).

Defaults (`adaptive_validity_check.py`): `DEFAULT_FDR_Q = 0.10` (a screening gate,
not a confirmatory claim — the looser q halves the feasibility floor), capped at
`DEFAULT_FDR_MAX_PERMUTATIONS = 2000`.

### The permutation feasibility floor

The smallest plus-one permutation p-value is `1 / (1 + n)`. For the
most-significant feature (BH rank 1) to clear its `q/m` threshold you need
`1/(1+n) ≤ q/m`, i.e. `n ≥ m/q - 1`. `min_permutations_for_fdr(m, q)` returns the
exact integer floor:

```
min_permutations_for_fdr(m, q) = ceil(m / q) - 1
```

Worked example (from the source): `m = 40` features at `q = 0.05` needs
`ceil(40/0.05) - 1 = 799` permutations for any rejection to be possible.

`fdr_permutation_budget` sizes the budget against this floor up to the cap:

- **feasible** -> run BH at `max(floor, default)` permutations (never below the
  configured `default`, which preserves z-score quality for narrow cohorts).
- **infeasible** (floor > cap) -> FDR is impossible at this width; the node
  **falls back to the static sigma-band** for that run rather than silently
  returning an always-empty confident set.

`benjamini_hochberg` is fail-loud: p-values must be in `(0, 1]` (a `0.0`, an
out-of-range value, or `±inf` raises `ValueError`); only `NaN` is tolerated and
treated as non-significant. When `n_permutations` is passed it also refuses to
run if the budget is below the feasibility floor, and rejects any p-value below
`1/(1+n)` (which cannot have come from the plus-one estimator at that budget).

The `_apply_fdr_firing_override` step re-decides the HIGH tier from confident-set
membership: a confident, NOT-declared-safe feature fires `high/drop`; a
sigma-band-high feature that is NOT FDR-confident is demoted to `moderate/review`
(FDR is the auto-fire authority). Declared-safe features take the immunity paths
below.

---

## Declared-safe FULL manifest immunity (#648)

> **Invariant (user decision, 2026-06-03):** on a **recognized-manifest cohort**,
> a feature whose `FeatureContract` declares `knowable_at` pre-or-at-index is
> **NEVER reported as leakage** — neither by the statistical/FDR discriminator
> (which cannot tell a strong pre-index predictor from a leak) nor by the
> deterministic structural checks (which false-fire on sparse rare-event
> columns). The contract is the authoritative temporal arbiter.

This is the **only sanctioned severity DOWNGRADE** in the whole detector — a
contract-certified feature was never a real leak. It is enforced in **both** nodes:

- `adaptive_validity_check.py` — `_declared_safe_immune_features(candidates,
  manifest_source)` returns the candidates whose contract
  `knowable_at.is_pre_or_at_index()` is true. They are stripped from
  `leaked_features`, `leakage_findings`, and `adaptive_flagged_features`, then
  severity is **recomputed** from the surviving findings
  (`_severity_from_finding_dicts`). This is the only place severity is allowed to
  fall (e.g. `high -> none/moderate`).
- `leakage_remediation.py` — the Step 2.6 companion. Because the LLM remediator
  reasons over ALL columns, it can narratively add a contract-certified feature to
  its drop list; the companion strips those declared-safe features back off
  `features_to_drop` and restores them to `recommended_feature_set`. Statistical
  governance still applies to **un-contracted** features.

### Opt-in safety invariant

A `None` / unrecognized `manifest_source` grants **NO immunity**
(`_declared_safe_immune_features` returns the empty set). Synthetic and ad-hoc
runs that never registered a manifest stay fully under statistical governance —
immunity is opt-in by cohort identity, never the default.

### Distinct from the #604 synthetic-fixtures FDR carve-out

Do not confuse full manifest immunity (#648) with the **synthetic-fixtures-only**
`declared_safe_full_immunity` FDR carve-out (#604), which is a different,
narrower mechanism:

| | #648 full manifest immunity | #604 `declared_safe_full_immunity` |
|---|---|---|
| Scope | Any recognized-manifest cohort (csu / optum / synthetic) | Legacy synthetic ml_patients fixtures ONLY (manifest leak-free by construction) |
| Where | `_declared_safe_immune_features` in both nodes | `_apply_fdr_firing_override` (the FDR driver) |
| Default | Active whenever a manifest resolves | `False` — set only by `run_tier0_test._resolve_declared_safe_full_immunity` |
| Effect | Declared-safe feature is exempt from leakage entirely | A declared-safe FDR-confident feature is routed to review even if its sigma-band reached high |
| Real cohorts | Immunity applies for declared pre-index features | Carve-out OFF — the "overwhelming evidence still drops" backstop is preserved for the fallible real-cohort manifest |

---

## Manifest auto-detection (#648)

`resolve_manifest_source(data_source, override)` decides which cohort manifest a
run consults (priority: explicit `override` > path auto-detection > `None`).
`autodetect_manifest_source` matches a registered source when a path segment
**equals** it OR is an **underscore-delimited variant** (`<source>_...`):

| `--data-dir` segment | resolves to | why |
|----------------------|-------------|-----|
| `data/rwd/optum/initiation` | `optum` | exact segment match |
| `data/rwd/optum_gap_enriched/initiation` | `optum` | `<source>_` prefix match (#648) — a gap-enriched extract is still the same cohort and must consult the same `FeatureContract` |
| `data/rwd/csu` | `csu` | exact segment match |
| `data/synthetic` | `synthetic` | exact segment match |

The `<source>_` shape (not a bare `startswith`) prevents partial-word false
positives like `optumistic` matching `optum`, and preserves the strictness
contract: a path with two distinct known segments raises `ValueError` (ambiguous
-> must disambiguate with an override); an `override` that conflicts with the
auto-detected source raises; an unknown override raises (fail loud on a typo).

> **Discrepancy note (trust-the-code):** an earlier note in the backlog stated
> that `run_tier0_test.py` does NOT auto-detect `optum_gap_enriched` from
> `--data-dir`, so Layer-5 immunity there needs an explicit
> `--feature-manifest-source optum` override. On the current worktree HEAD this
> is **no longer true** — the #648 `<source>_` prefix match in `resolution.py`
> fires, and `--data-dir data/rwd/optum_gap_enriched/<cohort>` auto-detects to
> `optum`. Verified by running `resolve_manifest_source` against the gap-enriched
> path (returns `optum`). The explicit override is still accepted and is the safe
> choice when in doubt, but it is no longer required for gap-enriched Optum runs.

A non-string `data_source` (e.g. the `{"type": "file_dir", "path": ...}` dict the
pipeline accepts for file batches) yields no auto-detection — the runner passes
the raw `--data-dir` string to the resolver, so detection still works.

---

## Cosmetic console caveat: "Clean Features / Dropped"

The Tier-0 "Clean Features / Dropped" console report reflects the LLM/rule
**`recommended_feature_set`** from `leakage_remediation` — it is **NOT** the
actual `train_df` columns the model is trained on. Treat it as cosmetic.

To verify the REAL retained-feature count, use one of:

- the **data-sufficiency floor** (`sufficiency_check.py`): events-per-variable
  rule `required_n = ceil(epv_floor * n_features / minority_prevalence)` — i.e.
  with an EPV floor of ~2, roughly `2 * n_features / minority_prevalence` events
  are needed; invert it to back out the feature count the run actually
  supports; or
- the **Feast-registration count** (the features the feast_registrar actually
  registered).

Do not read feature counts off the console summary.

---

## Feast freshness: advisory for file-sourced runs (#648)

A `--data-dir` run sources features straight from parquet, not from the Feast
online store, so a stale/unreachable Feast is irrelevant to that data. In
`feast_registrar.py`:

- When `data_source` is `{"type": "file_dir", ...}` (a file-sourced batch run),
  stale features are logged as an **ADVISORY** and do not block training.
- Genuine Feast-serving runs keep the hard block: stale features set
  `feast_blocked=True` and add a blocking issue **unless** `ALLOW_STALE_FEAST=1`
  is set in the environment (an ops-only escape hatch for known Feast outages).

So a run that lacks a `feature_store.yaml` (e.g. an isolated worktree) can set
`ALLOW_STALE_FEAST=1` to proceed.

---

## Operator quick reference

| Knob | Default | Effect |
|------|---------|--------|
| `--feature-manifest-source {csu,optum,synthetic}` | auto-detected from `--data-dir` | Opts the run into a cohort manifest -> Layer-1 verdicts + declared-safe immunity. |
| `adaptive_fdr_enabled` | `True` | FDR confident set drives the HIGH tier (else static sigma-band). |
| `adaptive_fdr_q` | `0.10` | Benjamini-Hochberg false-discovery rate. |
| `adaptive_fdr_max_permutations` | `2000` | Cap on the feasibility-aware budget; over it -> sigma-band fallback. |
| `adaptive_layer3_ablation_enabled` | `False` | Opt-in joint-model ablation pass (MAX-rule escalation). |
| `adaptive_layer4_enabled` | `False` | Opt-in LLM causal-role auditor. |
| `adaptive_declared_safe_full_immunity` | `False` | #604 carve-out; set only for synthetic fixtures by the runner. |
| `ALLOW_STALE_FEAST` (env) | unset | `=1` bypasses the Feast staleness hard block. |
