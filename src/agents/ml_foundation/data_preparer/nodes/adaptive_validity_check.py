"""Adaptive validity check — Layer 5 pipeline integration node.

Runs the data-derived Layer 3 adversarial discriminator against every
feature in train_df and emits a structured ``LeakageVerdict`` per feature.
Augments (does not replace) the existing ``detect_leakage`` results, so
both the legacy hardcoded checks and the adaptive permutation-baseline
checks contribute to the leakage_remediation routing.

Decision policy (data-derived, no hardcoded AUC thresholds):

    z > 5σ above null  → severity=high,     remediation=drop      (auto-flag)
    3σ < z ≤ 5σ        → severity=moderate, remediation=ambiguous (Layer 4 review)
    z ≤ 3σ             → severity=info,     remediation=keep

Layer 4 (DSPy CausalRoleClassifier) is invoked for ``ambiguous`` verdicts
when an LM is configured; otherwise the verdict is recorded for manual
governance review. This implementation focuses on Layers 1+3 wiring; Layer
4 LM dispatch lands when the API key configuration story is finalized.

Per-layer ordering (issue #212 — Layer 4 fires on pre-joint-check z-band):

    Step 1.  Layer 1 manifest contract pass (every column). Caught features
             short-circuit with severity=high, decided_by="layer_1".
    Step 2.  Layer 3 permutation-baseline scoring per numeric feature
             (``compute_adversarial_score`` → z, actual_auc, null_mean,
             null_std). Runs once per surviving feature.
    Step 3.  HBLP severity classification (``hblp_classify``):
             3a. z-only band assignment → ``severity_pre_joint_check``.
             3b. Issue #194 joint check ``|delta_AUC| <= floor`` may clamp
                 the final ``severity`` to ``info`` (legitimate weak
                 predictor protection). ``severity_pre_joint_check`` is
                 preserved as the raw signal for downstream gating.
    Step 4.  Layer-3 ablation pass (issue #196, opt-in via
             ``adaptive_layer3_ablation_enabled``). Combined with
             permutation via MAX-rule on the post-joint-check severity:
             ablation can ESCALATE info→moderate/high if it crosses its
             own joint check, but never DOWNGRADE.
    Step 5.  Layer 4 LLM-verdict trigger (issue #193 / #212): fires on
             ``severity_pre_joint_check`` (NOT the joint-clamped final
             ``severity``) so a weak-effect Layer 3 signal that #194 has
             downgraded still surfaces an LLM verdict for the audit
             trail. Trigger rule:
                 ``severity_pre_joint_check == "moderate"`` OR
                 (``severity_pre_joint_check == "high"`` AND
                  ``layer_1_declared_safe``)
             Issue #194's downstream bar is preserved unchanged — the
             final verdict still uses the joint-clamped ``severity``;
             Layer 4 is an additive audit channel, not a relaxation.
    Step 6.  EnsembleVoter renders the final EnsembleVerdict from the
             precedence ladder (Layer 1 high veto → Adversarial high veto
             → KG-contradictory abstain → LLM path → Adversarial-moderate
             review → no-signal abstain). When an LLM verdict was
             produced in Step 5, the voter emits
             ``decided_by="llm"`` + ``layer="4"`` in the audit trail.

The pre-#212 ordering had Layer 4 fire on the joint-clamped
``severity``, which silently starved the LLM-verdict path for every
feature where the joint check had clamped to ``info``. That contradicted
the documented "Layer 4 disambiguates ambiguous z-band signals" intent.

Acceptance criterion #4 of ``adaptive_temporal_validity_redesign.md``:
every feature decision produces a structured record with layer, evidence,
confidence, and remediation.

Phase 2.9 Stage 1 wiring (2026-05-08): per-feature decisions for cases
that combine Layer 1 + Layer 3 signals route through the
``EnsembleVoter`` from ``src/data/kg/ensemble_voter.py``. This is the
single canonical decision path the redesign plan calls for. The voter
output is adapted back to the legacy dict shape so downstream consumers
(``leakage_remediation`` node + ``write_adaptive_verdicts_sidecar``)
continue to work unchanged. Three new optional fields are added to each
verdict for the Phase 2.7+ audit trail:

- ``decided_by``: ``"layer_1"`` / ``"adversarial"`` / ``"kg"`` /
  ``"llm"`` / ``"abstain"`` (where Phase 2.9 Stage 1 only emits the
  first two; KG and LLM stay ``None`` until Stage 2/3 follow-ups land).
- ``disagreements``: tuple of strings describing cross-source
  contradictions (always empty in Stage 1 since only one source is
  active per feature).
- ``kg_signal``: KG signal classification (always ``"no_signal"`` in
  Stage 1 since ``kg_edges`` is empty).

Cases the voter cannot decide are routed through bypass paths to
preserve the legacy ``severity=info, remediation=keep`` semantics for
"tested and passed" (adv=info alone) and "could not test"
(too-few-rows / scoring-error) verdicts. The voter would otherwise
abstain on these inputs, which would change the downstream contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, cast

import numpy as np
import pandas as pd

from src.data.adversarial_leakage import (
    compute_adversarial_score,
    compute_feature_ablation,
    fdr_confident_set,
    fdr_permutation_budget,
    min_permutations_for_fdr,
)
from src.data.feature_contract import FeatureContract
from src.data.manifests import (
    CSU_FEATURES,
    CSU_FORBIDDEN_AS_FEATURES,
    OPTUM_FEATURES,
    OPTUM_FORBIDDEN_AS_FEATURES,
    SYNTHETIC_FEATURES,
    SYNTHETIC_FORBIDDEN_AS_FEATURES,
    lookup_feature_contract,
)
from src.ml.causal_role_dgp.extractor import derive_structural_role

# ``EnsembleVoter`` and ``EnsembleVerdict`` are LAZY-imported below to
# avoid triggering ``src.data.kg.__init__`` at module-import time. The
# kg package transitively imports ``httpx`` (via UMLSClient /
# EuropePMCClient / CrossrefClient), and pulling ``httpx`` into modules
# that LangGraph nodes import at pytest-collection time has produced
# asyncio-loop interactions in xdist-parallelised integration tests on
# CI (``RuntimeError: Event loop is closed``). The voter + verdict
# types are pure-Python; deferring the import to helper-call time
# keeps adaptive_validity_check.py's import surface free of httpx.
if TYPE_CHECKING:
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import CausalRole, EnsembleVerdict, KGEdge, LLMVerdict


class _HblpRoutingViolationError(RuntimeError):
    """Raised when ``_compose_legacy_verdict`` receives a pre-classified
    ``adversarial_input`` lacking the ``_hblp_classified=True`` tag.

    Plan v4 §2 G3 / codex MED-5: every Layer 3 severity classification
    MUST route through ``_adversarial_input → hblp_classify``. The
    wiring-guard's AST scan only verifies static callsites; this
    runtime check rejects pre-classified dicts that bypassed the
    routing chain (e.g. a hand-rolled legacy ``if z > HIGH_Z`` ladder
    constructed in a caller and passed in as ``adversarial_input``).
    """


def _get_ensemble_voter_class() -> type:
    """Return the ``EnsembleVoter`` class via lazy import.

    Centralises the lazy import so all runtime call sites share a single
    point of side-effect. Per the docstring at the top of this module,
    ``src.data.kg.__init__`` transitively imports ``httpx``; deferring
    until first use keeps ``adaptive_validity_check.py``'s import-time
    surface free of httpx.
    """
    from src.data.kg.ensemble_voter import EnsembleVoter as _EnsembleVoter

    return _EnsembleVoter


def _try_load_layer_4_classifier() -> Optional[Any]:
    """Lazily load the persisted Phase 2.5 compiled classifier.

    Imports inside the function so the loader's ``dspy`` import doesn't
    pull DSPy into this module's import-time surface (mirroring the
    ``EnsembleVoter`` lazy-import pattern above). Returns ``None`` when:

    - no DSPy LM is configured AND no API key is present in env (the
      developer-laptop / CI-without-key path);
    - the compiled artifact is missing (compile script hasn't run yet);
    - the loader raises during load (we log + swallow so Layer 4 is
      best-effort, never blocking the pipeline).

    Stage 3 wiring (issue #193): the orchestrator calls this once per node
    invocation and reuses the returned classifier across every per-feature
    Layer 4 call to avoid reloading the artifact on every feature.

    Codex pass-1 HIGH-1 (issue #193): this helper now invokes
    ``ensure_dspy_lm_configured`` so the production runtime path (where
    only ``ANTHROPIC_API_KEY`` is set in env, with no prior
    ``dspy.configure(...)`` call anywhere upstream) actually instantiates
    a DSPy LM. Without that call the loader's no-LM short-circuit would
    silently disable every Stage 3 invocation.
    """
    try:
        from src.data.causal_role_classifier_loader import (
            ensure_dspy_lm_configured,
            load_compiled_classifier,
        )
    except Exception as exc:  # pragma: no cover - import-time defensive
        logger.warning(
            "adaptive_validity_check: loader import failed (%s); Layer 4 skipped",
            exc,
        )
        return None

    # Codex HIGH-1: configure a default DSPy LM if none is registered AND
    # an API key is present. Returns False when no key is present (the
    # documented developer-laptop / CI-without-key path) — Layer 4 still
    # silently skips, but the production path with a key is no longer
    # silently disabled.
    if not ensure_dspy_lm_configured():
        return None

    try:
        return load_compiled_classifier(strict=False)
    except Exception as exc:  # pragma: no cover - load-time defensive
        logger.warning(
            "adaptive_validity_check: loader raised (%s); Layer 4 skipped",
            exc,
        )
        return None


logger = logging.getLogger(__name__)


HIGH_Z = 5.0
MODERATE_Z = 3.0
DEFAULT_PERMUTATIONS = 200

# Plan v4 Layer-A Phase 1 — dynamic FDR confident set (firing/severity driver).
# DEFAULT_FDR_Q: the Benjamini-Hochberg false-discovery rate for the confident
#   set. 0.10 (not 0.05) because this is a SCREENING gate that routes features
#   to drop/review, not a confirmatory scientific claim — and the looser q
#   halves the feasibility floor (ceil(m/q) permutations), bounding cost.
# DEFAULT_FDR_MAX_PERMUTATIONS: the cap on the feasibility-aware budget. When a
#   cohort is so wide that ceil(m/q) exceeds this, FDR is infeasible and the
#   node falls back to the static σ-band for that run (never silently empty).
DEFAULT_FDR_Q = 0.10
DEFAULT_FDR_MAX_PERMUTATIONS = 2000

# Minimum non-null sample count to run Layer 3 scoring on a feature.
# Below this floor the permutation-baseline z-score is too noisy to be
# reliable, so the feature gets a short-circuit ``severity=info`` verdict
# and is left for downstream review. Promoted from a hardcoded `30` per
# backlog item #11.c so future tightening can change one place.
MIN_LAYER3_SAMPLES = 30

# Issue #194 — Joint (z, |delta_AUC|) threshold for Layer 5 severity.
#
# Problem reproduced 2026-05-14 at n in {1k, 5k, 10k, 50k} via
# ``scripts/calibration/run_layer5_joint_threshold_sweep.py``: the legacy
# 5σ z-threshold alone over-flags legitimate weak demographic predictors
# at large n. The permutation-null std scales as ~1/sqrt(n) per the CLT
# (measured: null_std ≈ 0.029 at n=1k → 0.010 at n=10k → 0.004 at n=50k
# on the synthetic_rwd_realistic regime). At n=10k a benign feature with
# single-feature AUC=0.54 yields z ≈ 4σ; at n=50k the same feature yields
# z ≈ 16σ — well above any z-only threshold yet domain-trivial.
#
# Decision (user, issue #194): adopt the joint check
#     ``severity ∈ {moderate, high}  ⇔  (z > k) AND (|delta_AUC| > epsilon)``
# where ``delta_AUC = actual_auc - null_mean`` on the folded AUC-ROC scale
# (the same scale ``compute_adversarial_score`` already produces). The
# absolute-effect floor is interpretable in the pharma domain: a feature
# whose single-feature AUC is less than ``epsilon`` above chance is not
# an actionable leakage signal — at clinically-relevant sample sizes
# (n ≥ 1000) any leak worth dropping has delta_AUC well above this floor.
#
# Calibration (2026-05-14, sweep grid k ∈ {3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}
# × epsilon ∈ {0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09,
# 0.10, 0.12, 0.15} across N_REP=50 cohorts at each n):
#
#   - Legitimate weak demographic features (age, eligibility_duration_days
#     under signal_scale=1.0): empirical p99 of |delta_AUC| at n=10k is
#     0.0913; max observed across all n in {1k, 5k, 10k, 50k} is 0.1334.
#   - Injected leak patterns (post_index_aggregation, post_hoc_termination,
#     treatment_leaked_code, spurious_correlation): minimum observed
#     |delta_AUC| is 0.354 (treatment_leaked_code at n=2000) — well above
#     any reasonable epsilon floor.
#   - The benign-vs-leak window is wide ([0.13, 0.35]); choice of epsilon
#     within that window is robust.
#
# Chosen: ``k = HIGH_Z (5.0)`` (no relaxation in z; preserves HBLP wiring
# and the legacy 5σ contract for callers that override) and
# ``epsilon = 0.10`` (rounded up from the empirical p99=0.0913 with a
# small safety margin — same shape as PR #153's T2.2 buffer-fitting
# rule ``floor(P5 margin * 100) / 100 - 0.01 safety``, applied to the
# upper-tail of benign delta_AUC instead of the lower-tail of legitimate
# margin).
#
# Measured FPR at (k=5.0, epsilon=0.10) on legitimate weak predictors:
# 0% at n ∈ {1k, 5k, 10k, 50k}; TPR=100% on all 4 injected leak patterns
# at n=2000. Loosening to (k=4.5, epsilon=0.09) still meets FPR ≤ 1% at
# every n; tightening (smaller epsilon) reintroduces large-n
# false-positives.
#
# Lifecycle: this is the runtime threshold (NOT an observability-only
# advisory). It is enforced in ``hblp_classify`` for severity ∈ {high,
# moderate}; severity=info is unchanged. See
# ``calibration_runs/issue_194_sweep.jsonl`` for the full sweep output.
#
# Codex pass-1 MED-1 + pass-3 LOW-2 escape: when the permutation null
# has zero variance (``null_std=0``) AND ``actual_auc > null_mean``,
# ``compute_adversarial_score`` returns ``z=+inf``. Pre-joint-check
# the legacy non-finite-z guard classified these as severity=info
# silently — a false negative on deterministic high-effect signals
# with degenerate nulls. The joint check provides a principled escape:
# when ``z=+inf`` AND ``|delta_AUC| > LAYER5_DELTA_AUC_FLOOR_DEFAULT``,
# ``hblp_classify`` returns severity=high. The same escape is mirrored
# in (a) ``EnsembleVoter.vote``'s M3 guard (so KG-active shadow-mode
# interactions don't silently downgrade), and (b) the sweep helper
# ``_decide_joint`` in ``scripts/calibration/run_layer5_joint_threshold_sweep.py``.
# ``z=-inf`` and ``z=NaN`` still fall through to severity=info — the
# escape is one-sided (positive-inf strong-effect only).
LAYER5_DELTA_AUC_FLOOR_DEFAULT: float = 0.10


# ============================================================================
# Issue #196 — Phase 3.3 Layer-3 multi-feature ablation wiring.
#
# ``src/data/adversarial_leakage.py:compute_feature_ablation`` is the
# permutation-baseline of the OTHER suspicion axis: for each feature, drop
# it, retrain the joint model, measure |delta_AUC| relative to a column-
# shuffle null. The single-feature permutation test in
# ``compute_adversarial_score`` measures the MARGINAL leak signal of a
# feature on its own; ablation measures the MARGINAL CONTRIBUTION of the
# feature to the joint model. These are different mechanisms:
#
#   - Single-feature leak (caught by permutation):  feature ALONE has high
#     AUC → suspicious z-score.
#   - Interaction-only leak (caught by ablation):   feature on its own has
#     near-chance AUC, but the joint model crashes when it's removed —
#     the leak is encoded via an interaction term with another feature.
#
# Combination rule chosen: MAX (any-source-suspicious wins).
#
# Why MAX over ensemble averaging:
#   * Ensemble averaging would DILUTE an interaction-only leak: ablation
#     z=8σ (large) averaged with permutation z=0σ (near-noise) → 4σ, which
#     might fall below the moderate band's HBLP-effective threshold at
#     n_pos > 200. Averaging weighted by null-variance has the same
#     defect — the permutation null at n=10k can have std two orders of
#     magnitude smaller than the ablation null, weighting permutation
#     ~100x and crushing the ablation signal.
#   * MAX preserves the "if either signal is strong, escalate" contract
#     that operators expect from a defense-in-depth signal. The two tests
#     measure orthogonal failure modes; treating "ablation says drop" and
#     "permutation says drop" as substitutes is the right call.
#   * Symmetric application of the issue #194 joint check
#     ``severity ∈ {moderate, high}  ⇔  (z > k) AND (|delta_AUC| > epsilon)``
#     to BOTH the permutation z AND the ablation z prevents the same
#     large-n false-positive failure mode that #194 closed for permutation.
#     A feature with ablation z=6σ but |delta_AUC|=0.005 (legitimate weak
#     contribution that the joint model can substitute around at large n)
#     is NOT escalated.
#
# The escape-clause from issue #194 codex pass-1 MED-1 (z=+inf + degenerate
# null + |delta_AUC| above floor → severity=high) ALSO applies symmetrically
# to ablation z=+inf with the ablation delta_AUC.
#
# AND a NEW escape (strong-effect): delta_AUC > LAYER5_ABLATION_STRONG_EFFECT_DEFAULT
# (default 0.30, 3x the issue #194 floor) bypasses the z-anchored ladder
# entirely. SIGNED requirement (codex pass-1 MED-2) — positive-only:
# delta_auc > 0 means dropping the feature DEGRADES joint AUC (leak-carrier);
# delta_auc < 0 means dropping the feature IMPROVES joint AUC (nuisance/
# multicollinearity), NOT a leak. Rationale for the threshold magnitude:
# ``compute_feature_ablation``'s null distribution is built by shuffling
# the feature COLUMN, not the labels. For interaction-pair leaks — features
# whose ROW-ALIGNMENT with another column is the leak vector (redundant-
# noise-cancel pairs, sign-stratified variance shifts) — the column-shuffle
# null produces ``perm_delta_auc ≈ actual_delta_auc``, so the z-score
# collapses to ~0 even when delta_AUC is huge. Without this escape the
# AND-rule would silently miss exactly the leak class ablation is supposed
# to catch beyond permutation. The 0.30 threshold is structurally robust:
# legitimate weak predictors at any cohort size cannot produce
# delta_AUC > 0.30 (dropping the feature would have to destroy 30% of the
# joint model's AUC, which is the dominance signature of a real leak).
#
# Costs: O(n_features) main retrains + O(n_features × n_permutations) shuffle
# retrains. With ``DEFAULT_ABLATION_PERMUTATIONS=50`` and the LogisticRegression
# factory baked into ``compute_feature_ablation``, runtime on a 5-feature × 300-
# row pin (the integration test fixture) is ~2 s wall-clock. A 50-feature CSU/
# Optum pipeline at production widths is ~10-30 s. Hence default OFF; opt-in
# via the ``adaptive_layer3_ablation_enabled`` flag.
#
# Wide-feature blowup guard: ``DEFAULT_ABLATION_MAX_FEATURES=50`` caps the
# active ablation set. When ``numeric_candidates`` after Layer-1 exclusion
# exceeds the cap, the ablation pass is SKIPPED entirely (not partial —
# subsetting which features to ablate would bias the joint-model AUC the
# survivors are measured against). The orchestrator logs a warning so the
# operator sees the cap fired.
# ============================================================================

# Phase 3.3 — ablation tuning constants. All overridable via scope_spec state.
DEFAULT_ABLATION_PERMUTATIONS = 50
DEFAULT_ABLATION_MAX_FEATURES = 50


# ============================================================================
# Plan v3 §3 Tier 1B step 2 — Hierarchical Bayesian Leakage Prior (HBLP)
# variance-inflation helpers + step 5 derivation-lineage audit.
#
# HBLP rationale: at low n_positives the permutation null variance scales as
# ~1/√n_positives. With ~22 train positives σ_null≈0.04 makes legitimate
# confounders look like 3-5σ leaks even when Layer 1 cleared them. HBLP
# inflates the Layer 3 z-threshold proportionally so the perm test does
# not over-drop at small N.
#
# Prior structure: when Layer 1 manifest CLEARED a feature (knowable_at <=
# index_date per the manifest), require STRONGER Layer 3 evidence to override
# (i.e., a higher z-threshold). Conversely, when Layer 1 dropped the feature,
# Layer 3 inherits the standard threshold.
#
# Plan §6 Tier 1B Gate B1 acceptance: HBLP can ship as DIAGNOSTIC with this
# helper alone (no enforcement). Gate B2 (quality uplift claim) requires
# pre-specified ΔAUC≥0.03 + ECE/2 + stability/0.7 — a separate measurement.
# ============================================================================

# Variance-inflation reference. At n_positives=50 the inflation factor is
# 1.0 (no relaxation); at n_positives=22 it's sqrt(50/22)≈1.51 (51% wider
# threshold); at n_positives=200 it's sqrt(50/200)=0.5 (tightening, but
# capped at 1.0 so we never tighten below the base 5σ).
T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N: int = 50

# Layer-1-conditional inflation: a feature that Layer 1 manifest declared
# safe (knowable_at <= index_date) gets an ADDITIONAL z-threshold buffer on
# top of variance inflation. This encodes the structural prior: declared-
# safe features need stronger statistical evidence to be reclassified as
# leaks. Default 1.5x base threshold (so 5σ → 7.5σ for declared-safe at
# any N >= 50; further inflated for low-N).
T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER: float = 1.5


def hblp_effective_z_threshold(
    n_positives: int,
    layer_1_declared_safe: bool,
    base_threshold: float = HIGH_Z,
    variance_inflation_reference_n: int = T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N,
    declared_safe_prior_multiplier: float = T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER,
) -> float:
    """Plan v3 §3 Tier 1B step 2 — HBLP variance-inflation z-threshold.

    Computes an effective Layer 3 z-threshold that:
      1. Inflates by ``sqrt(reference_n / n_positives)`` when n_positives <
         reference_n (so 5σ at n=22 becomes ~7.5σ to compensate for the
         small-N null variance).
      2. Multiplies by ``declared_safe_prior_multiplier`` (default 1.5x)
         when ``layer_1_declared_safe=True`` — encoding the prior that a
         feature whose declared knowable_at <= index_date needs stronger
         Layer 3 evidence to be reclassified as a leak.
      3. Capped at ``base_threshold`` from below (HBLP never tightens; it
         only relaxes for low-N or declared-safe features).

    Args:
        n_positives: Training-split positive-class count (the binding
            constraint for permutation-null variance).
        layer_1_declared_safe: True iff the feature's manifest contract
            declared knowable_at <= index_date (Layer 1 cleared it).
        base_threshold: Pre-HBLP Layer 3 z-threshold (default 5.0).
        variance_inflation_reference_n: N at which inflation factor = 1.0.
        declared_safe_prior_multiplier: Additional multiplier when Layer 1
            cleared the feature.

    Returns:
        Effective z-threshold to use for Layer 3 decisioning. Always
        ``>= base_threshold`` (HBLP only relaxes, never tightens).
    """
    if n_positives <= 0:
        # Degenerate: no positives → no permutation signal possible.
        # Return base threshold; caller will likely short-circuit anyway.
        return float(base_threshold)

    variance_inflation = max(
        1.0,
        (float(variance_inflation_reference_n) / float(n_positives)) ** 0.5,
    )
    layer_1_factor = float(declared_safe_prior_multiplier) if layer_1_declared_safe else 1.0
    return float(base_threshold) * variance_inflation * layer_1_factor


def hblp_classify(
    z_score: float,
    n_positives: int,
    layer_1_declared_safe: bool,
    base_threshold: float = HIGH_Z,
    moderate_base_threshold: float = MODERATE_Z,
    delta_auc: Optional[float] = None,
    delta_auc_floor: float = LAYER5_DELTA_AUC_FLOOR_DEFAULT,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Plan v3 §3 Tier 1B step 2 — HBLP-aware Layer 3 severity classifier.

    Wraps the legacy ``HIGH_Z`` / ``MODERATE_Z`` ladder with HBLP
    variance-inflation. Returns a dict matching the legacy verdict shape:

      * ``severity`` ∈ {"high", "moderate", "info"}
      * ``effective_high_threshold``: float, post-HBLP threshold used
      * ``effective_moderate_threshold``: float, post-HBLP moderate band
      * ``base_threshold``: float, original 5σ
      * ``variance_inflation_factor``: float, sqrt(50/n_pos) clipped at 1.0
      * ``layer_1_factor``: float, 1.5x for declared-safe else 1.0
      * ``hblp_relaxed``: bool, True iff effective > base
      * ``rationale``: str, explanation

    Issue #194 — joint ``(z, |delta_AUC|)`` threshold:

      When ``delta_auc`` is supplied AND ``|delta_auc| <= delta_auc_floor``,
      severity is FORCED to ``info`` even if z exceeds the HBLP-effective
      high/moderate band. This is the joint check
      ``severity ∈ {moderate, high}  ⇔  (z > k) AND (|delta_auc| > epsilon)``
      adopted in issue #194; it prevents large-n false-positives on
      legitimate weak predictors whose null variance has shrunk per CLT
      below the absolute-effect floor of pharma-actionable leakage.

      Backwards compatibility: callers that DO NOT supply ``delta_auc``
      (legacy code paths, tests that build classifications from raw z
      alone) see the legacy z-only behaviour. The joint check fires ONLY
      when ``delta_auc`` is explicitly provided AND finite. Non-finite
      ``delta_auc`` (NaN / inf) is treated as "delta_AUC unknown" and the
      classification falls through to the z-only path.

    Args:
        z_score: Permutation-anchored z-score (folded-AUC scale).
        n_positives: Train-split positive-class count (drives HBLP variance
            inflation).
        layer_1_declared_safe: True iff the feature's manifest contract
            declared knowable_at <= index_date.
        base_threshold: Pre-HBLP z-threshold for severity=high (default 5σ).
        moderate_base_threshold: Pre-HBLP z-threshold for severity=moderate
            (default 3σ).
        delta_auc: Signed ``actual_auc - null_mean`` on the folded AUC-ROC
            scale, when known. Optional; when omitted (None) or non-finite,
            the joint check does not fire and the classifier falls through
            to legacy z-only behaviour.
        delta_auc_floor: Absolute-effect floor (``epsilon``). Severity is
            forced to "info" when ``|delta_auc| <= delta_auc_floor``.
            Default 0.10 per issue #194 calibration sweep.
    """
    high_eff = hblp_effective_z_threshold(
        n_positives=n_positives,
        layer_1_declared_safe=layer_1_declared_safe,
        base_threshold=base_threshold,
        **kwargs,
    )
    # Moderate threshold scales by the same factor as high to preserve
    # the band proportions.
    relaxation_factor = high_eff / float(base_threshold)
    moderate_eff = float(moderate_base_threshold) * relaxation_factor

    # Issue #194 — joint check. Compute the |delta_AUC| floor outcome
    # BEFORE the z-only ladder so the rationale string can carry the
    # joint-check decision uniformly. ``delta_auc`` is optional; when
    # None or non-finite we treat the floor as inactive.
    # ``delta_auc_value`` is a narrowed-non-None float used everywhere
    # below; ``delta_auc_known`` is the predicate.
    delta_auc_known = delta_auc is not None and bool(np.isfinite(delta_auc))
    delta_auc_value: float = float(delta_auc) if delta_auc is not None else 0.0
    delta_auc_floor_value: float = float(delta_auc_floor)
    delta_auc_below_floor = delta_auc_known and abs(delta_auc_value) <= delta_auc_floor_value

    # Issue #194 codex pass-1 MEDIUM-1: a zero-variance permutation null
    # makes ``compute_adversarial_score`` return ``z=+inf`` when
    # ``actual_auc > null_mean`` (a deterministic high-effect signal —
    # every permutation produced the same AUC, so the actual feature's
    # AUC is infinitely many standard deviations above the null). The
    # legacy code path classified ``z=inf`` as severity=info via the
    # non-finite-z guard, which silently dropped these strong signals.
    # The joint check now provides a principled escape: when ``z=+inf``
    # AND ``|delta_AUC| > floor``, severity=high (the absolute-effect
    # floor confirms a real leak; the inf z just means the null was
    # degenerate). ``z=-inf`` or NaN still falls through to severity=info.
    z_is_positive_inf_strong_effect = (
        z_score is not None
        and isinstance(z_score, (int, float))
        and not isinstance(z_score, bool)
        and not (isinstance(z_score, float) and np.isnan(z_score))
        and not np.isfinite(z_score)
        and z_score > 0
        and delta_auc_known
        and abs(delta_auc_value) > delta_auc_floor_value
    )

    # Issue #212 — pre-joint-check severity. The z-only classification
    # before the issue #194 joint-check ``|delta_AUC| <= floor`` clamp
    # downgrades to ``info``. Used by the orchestrator's Layer 4 trigger
    # so an LLM verdict can be obtained for legitimate weak signals that
    # the joint check would otherwise silently swallow (which starves
    # the LLM-verdict audit trail). The FINAL ``severity`` field below
    # is still joint-check-clamped — issue #194's downstream bar is
    # preserved unchanged. See module docstring "Per-layer ordering"
    # section.
    if z_is_positive_inf_strong_effect:
        severity_pre_joint_check = "high"
    elif not _is_finite_z(z_score):
        severity_pre_joint_check = "info"
    elif z_score > high_eff:
        severity_pre_joint_check = "high"
    elif z_score > moderate_eff:
        severity_pre_joint_check = "moderate"
    else:
        severity_pre_joint_check = "info"

    if z_is_positive_inf_strong_effect:
        # Deterministic high-effect signal with degenerate null.
        # Route through severity=high so a real leak isn't silently
        # dropped. Audit-trail records the inf-z + strong-effect path.
        severity = "high"
        rationale = (
            f"z={z_score} (degenerate null; null_std=0 → infinite "
            f"separation) AND |delta_AUC|={abs(delta_auc_value):.4f} > floor "
            f"{delta_auc_floor_value:.4f}; joint check confirms severity=high "
            f"(issue #194 codex pass-1 MED-1)"
        )
    elif not _is_finite_z(z_score):
        severity = "info"
        rationale = "z_score is non-finite; HBLP defaults to severity=info"
    elif z_score > high_eff:
        if delta_auc_below_floor:
            # Joint check fires: z passes the high band but |delta_AUC|
            # is below the absolute-effect floor. Force severity=info
            # so a legitimate large-n weak predictor isn't dropped.
            severity = "info"
            rationale = (
                f"z={z_score:.2f}σ > HBLP-effective {high_eff:.2f}σ but "
                f"|delta_AUC|={abs(delta_auc_value):.4f} ≤ floor "
                f"{delta_auc_floor_value:.4f}; joint check forces "
                f"severity=info (issue #194)"
            )
        else:
            severity = "high"
            joint_note = (
                f", |delta_AUC|={abs(delta_auc_value):.4f} > floor {delta_auc_floor_value:.4f}"
                if delta_auc_known
                else ""
            )
            rationale = (
                f"z={z_score:.2f}σ > HBLP-effective {high_eff:.2f}σ "
                f"(base={base_threshold}σ × inflation={relaxation_factor:.2f})"
                f"{joint_note}"
            )
    elif z_score > moderate_eff:
        if delta_auc_below_floor:
            # Same joint logic for the moderate band.
            severity = "info"
            rationale = (
                f"z={z_score:.2f}σ in HBLP moderate band but "
                f"|delta_AUC|={abs(delta_auc_value):.4f} ≤ floor "
                f"{delta_auc_floor_value:.4f}; joint check forces "
                f"severity=info (issue #194)"
            )
        else:
            severity = "moderate"
            joint_note = (
                f", |delta_AUC|={abs(delta_auc_value):.4f} > floor {delta_auc_floor_value:.4f}"
                if delta_auc_known
                else ""
            )
            rationale = (
                f"z={z_score:.2f}σ between moderate {moderate_eff:.2f}σ and "
                f"high {high_eff:.2f}σ (HBLP-inflated){joint_note}"
            )
    else:
        severity = "info"
        rationale = f"z={z_score:.2f}σ ≤ HBLP-effective moderate {moderate_eff:.2f}σ"

    variance_inflation = max(
        1.0,
        (float(T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N) / max(n_positives, 1)) ** 0.5,
    )
    layer_1_factor = (
        float(T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER) if layer_1_declared_safe else 1.0
    )
    return {
        "severity": severity,
        # Issue #212 — z-only severity BEFORE the issue #194 joint-check
        # ``|delta_AUC| <= floor`` clamp. Equals ``severity`` when the
        # joint check did not fire (no clamp). When the joint check
        # downgraded to ``info``, this field preserves the band the z
        # would have landed in (``moderate`` or ``high``). The
        # orchestrator's Layer 4 trigger reads this field so an LLM
        # verdict still surfaces on weak-effect signals that joint
        # check clamps. NOTE: this does NOT relax the joint check —
        # the FINAL ``severity`` field is still clamped; Layer 4 is
        # additive (audit signal), not a bar.
        "severity_pre_joint_check": severity_pre_joint_check,
        "effective_high_threshold": high_eff,
        "effective_moderate_threshold": moderate_eff,
        "base_threshold": float(base_threshold),
        "variance_inflation_factor": variance_inflation,
        "layer_1_factor": layer_1_factor,
        "hblp_relaxed": high_eff > float(base_threshold),
        "n_positives": int(n_positives),
        "layer_1_declared_safe": layer_1_declared_safe,
        "rationale": rationale,
        # Issue #194 — joint-check audit fields. Always populated so
        # downstream readers (audit JSON sidecar, dashboards) can see
        # whether the joint check was active and what the floor was.
        "delta_auc": (delta_auc_value if delta_auc_known else None),
        "delta_auc_floor": delta_auc_floor_value,
        "delta_auc_below_floor": bool(delta_auc_below_floor),
    }


def _is_finite_z(z: Any) -> bool:
    try:
        return bool(np.isfinite(float(z)))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
# Plan v3 §3 Tier 1B step 5 — Derivation-lineage audit (declared-path only).
#
# IMPORTANT scope: this audit proves DECLARED-PATH validity only — i.e., the
# feature's manifest contract names only pre-anchor data sources. It does
# NOT detect (a) false declarations (a manifest entry that lies about its
# inputs), (b) hidden post-anchor dependencies inside the derivation
# function, or (c) leakage encoded through legitimate-looking pre-anchor
# paths. Per plan §7 risk register, broader leakage mitigation requires
# lineage + leakage-injection regression + AdversarialProbe (already shipped
# Phase 2.4 / PR #90) + conditional MI (deferred Tier 3).
# ----------------------------------------------------------------------------


def lineage_audit_declared_path(
    feature_name: str,
    data_source: Optional[str],
) -> Dict[str, Any]:
    """Plan v3 §3 Tier 1B step 5 — Declared-path lineage validity check.

    Looks up the feature's ``FeatureContract`` from ``MANIFEST_SOURCES``
    and confirms its declared ``knowable_at`` reference is pre-anchor
    (i.e., ``"index_date"`` or earlier). Returns a structured audit
    record with:

      * ``feature_name``, ``data_source`` — inputs.
      * ``contract_found`` — bool, True iff the manifest registry returned
        a contract for this feature.
      * ``knowable_at_reference`` — str, the contract's reference (e.g.
        ``"index_date"``, ``"post_index"``).
      * ``declared_path_valid`` — bool, True iff the reference is
        pre-anchor. None when contract_found is False.
      * ``rationale`` — str, human-readable explanation.

    SCOPE NOTE (per plan §7 risk register): this audit proves DECLARED-PATH
    validity only. A feature whose manifest entry lies about its inputs
    will pass this audit AND be a leak. Broader leakage mitigation needs
    lineage + leakage-injection regression + AdversarialProbe (Phase 2.4)
    + conditional MI (deferred Tier 3).
    """
    from src.data.manifests import MANIFEST_SOURCES

    if data_source is None or data_source not in MANIFEST_SOURCES:
        return {
            "feature_name": feature_name,
            "data_source": data_source,
            "contract_found": False,
            "knowable_at_reference": None,
            "declared_path_valid": None,
            "rationale": (
                f"data_source={data_source!r} not in MANIFEST_SOURCES "
                "registry; cannot audit declared-path validity"
            ),
        }

    contract = MANIFEST_SOURCES[data_source](feature_name)
    if contract is None:
        return {
            "feature_name": feature_name,
            "data_source": data_source,
            "contract_found": False,
            "knowable_at_reference": None,
            "declared_path_valid": None,
            "rationale": (
                f"manifest for source={data_source!r} returned no contract "
                f"for feature={feature_name!r}; cannot audit declared-path "
                "validity"
            ),
        }

    knowable_at_ref = getattr(contract.knowable_at, "reference", None)
    # Codex pass-1 MED-7 (PR #137 v4 G1): use ``KnowableAt.is_pre_or_at_index()``
    # API instead of a string allow-list. The string allow-list approach
    # was fragile — adding a new pre-anchor reference (e.g., a column
    # reference like ``"first_eligible_date"``) required updating the
    # set in two places. The API delegates that decision to the
    # ``KnowableAt`` dataclass, which is the single source of truth per
    # ``src/data/feature_contract.py:115`` (returns False only for
    # ``"post_index"`` or positive ``offset_days``; conservatively
    # accepts any other reference at offset_days <= 0).
    knowable_at = contract.knowable_at
    if knowable_at is None:
        declared_path_valid = False
        rationale = "contract.knowable_at is None — cannot audit declared-path validity"
    else:
        declared_path_valid = knowable_at.is_pre_or_at_index()
        rationale = (
            f"contract knowable_at={knowable_at} "
            f"{'IS pre-anchor (audit pass; is_pre_or_at_index=True)' if declared_path_valid else 'is NOT pre-anchor (audit fail; is_pre_or_at_index=False)'}"
        )

    return {
        "feature_name": feature_name,
        "data_source": data_source,
        "contract_found": True,
        "knowable_at_reference": knowable_at_ref,
        "declared_path_valid": declared_path_valid,
        "rationale": rationale,
    }


# =============================================================================
# Verdict building — Phase 2.9 Stage 1 wiring.
#
# Three layers of helpers participate in producing the legacy verdict dict
# that downstream nodes consume:
#
# 1. ``_layer_1_input`` and ``_adversarial_input`` build the per-source
#    "input verdict" dicts that ``EnsembleVoter.vote`` accepts. They are
#    pure data — no flow control.
# 2. ``EnsembleVoter.vote`` composes those inputs into a single
#    ``EnsembleVerdict`` with documented precedence rules.
# 3. ``_ensemble_to_legacy_dict`` adapts the ``EnsembleVerdict`` back to
#    the legacy ``LeakageVerdict`` shape this node has emitted since PR
#    #84, with three new optional fields for the Phase 2.7+ audit trail.
#
# The ``_legacy_*`` helpers handle bypass cases the voter would otherwise
# abstain on (info-severity adversarial alone, short-circuited adversarial
# probes). They emit the legacy dict directly so downstream consumers see
# the same ``severity=info, remediation=keep`` contract they have always
# seen.
# =============================================================================


def _layer_1_input(feature: str, contract: FeatureContract) -> dict[str, Any]:
    """Build a Layer 1 ``EnsembleVoter.vote`` input dict.

    Mirrors what ``EnsembleVoter`` documents as the Layer 1 verdict shape
    (severity + contract_source + contract_window_days). The voter uses
    contract_source as the M4 audit-integrity guard; we always populate
    it from the manifest contract so the guard's "missing or empty"
    branch never fires for our own input.
    """
    return {
        "feature": feature,
        "layer": "1",
        "severity": "high",
        "remediation": "drop",
        "evidence": (
            f"Layer 1 declarative contract: feature.knowable_at="
            f"{contract.knowable_at} (post_index); the manifest declares this "
            f"column is not knowable at prediction time → drop"
        ),
        "contract_source": contract.source,
        "contract_window_days": contract.window_days,
    }


def _layer_1_verdict(feature: str, contract: FeatureContract) -> dict[str, Any]:
    """Legacy Layer 1 verdict producer — kept for backward compatibility.

    Used by external test importers that construct verdicts directly.
    The internal decision flow routes through ``_layer_1_input`` +
    ``_compose_legacy_verdict`` + ``EnsembleVoter`` + adapter; this
    wrapper does the same so the voter's audit-integrity guards (M4
    malformed contract_source check, etc.) apply uniformly to all
    Layer 1 verdict-construction call sites.

    Codex review MEDIUM (M2, 2026-05-08): the prior implementation
    constructed the ``EnsembleVerdict`` directly and called the
    adapter without involving the voter, bypassing M4's malformed-
    contract guard. While ``FeatureContract.source`` is typed ``str``
    (always populated in production), defense in depth routes this
    helper through the voter so external callers / future code paths
    that pass synthetic contracts see consistent guard behaviour.
    """
    return _compose_legacy_verdict(
        feature,
        voter=_get_ensemble_voter_class()(),
        layer_1_input=_layer_1_input(feature, contract),
    )


def _build_layer_4_inputs(
    feature: str,
    contract: Optional[FeatureContract],
    target: str,
    manifest_source: Optional[str],
) -> tuple[str, str]:
    """Build the (derivation_pseudocode, dataset_context) pair for Layer 4.

    The compiled :class:`src.data.causal_role_classifier.CausalRoleClassifier`
    expects three input fields: ``feature_name`` (provided by caller),
    ``derivation_pseudocode``, and ``dataset_context``. This helper assembles
    the latter two from the feature's manifest contract (when available) and
    the scope_spec target metadata.

    When ``contract`` is None (feature has no manifest entry — e.g. a numeric
    column the runner pre-cleaned), the derivation pseudocode falls back to a
    "no manifest contract on file" sentinel string. The LLM is still able to
    classify based on the feature name alone, but with reduced confidence;
    the audit trail records the absence so an operator can extend the
    manifest later.
    """
    if contract is not None:
        derivation = (
            f"source={contract.source}; "
            f"derivation_inputs={list(contract.derivation_inputs)}; "
            f"aggregation={contract.aggregation}; "
            f"window_days={contract.window_days}; "
            f"knowable_at={contract.knowable_at}"
        )
    else:
        derivation = (
            f"No manifest contract on file for {feature!r} "
            "(LLM is classifying from feature name + dataset context only)"
        )

    cohort = manifest_source or "unspecified"
    dataset_context = f"cohort={cohort}; target={target}; prediction_anchor=index_date"
    return derivation, dataset_context


def _adversarial_input(
    score: dict[str, Any],
    *,
    n_train_pos: Optional[int] = None,
    layer_1_declared_safe: bool = False,
) -> dict[str, Any]:
    """Build a Layer 3 ``EnsembleVoter.vote`` input dict from a raw score.

    Maps the score dict produced by ``compute_adversarial_score`` (z_score,
    actual_auc, null_mean, null_std, p_value, n_permutations) into the
    severity-tagged shape the voter expects. Always populates ``z_score``
    so the voter's M3 audit-integrity guard never fires on our own input.

    Severity routing — Plan v4 §2 G3 wiring (post-2026-05-10):

    Routes z-score classification through ``hblp_classify``, which:

      * Inflates the high (5σ) and moderate (3σ) thresholds by
        ``sqrt(reference_n / n_train_pos)`` when ``n_train_pos <
        reference_n`` (default 50). At low N the permutation-null variance
        scales as ``~1/sqrt(n_pos)``, so a fixed 5σ over-flags legitimate
        confounders. HBLP's variance-inflation compensates.
      * Adds a ``declared_safe_prior_multiplier`` (default 1.5x) when
        ``layer_1_declared_safe=True`` — encoding the prior that a feature
        whose declared ``knowable_at <= index_date`` (per
        ``MANIFEST_SOURCES``) needs stronger Layer 3 evidence to be
        reclassified as a leak.
      * Returns ``severity ∈ {"high", "moderate", "info"}`` with the
        post-HBLP effective thresholds annotated in the evidence string.

    The legacy ``if z > HIGH_Z / elif z > MODERATE_Z / else`` ladder is
    removed — there is no parallel path. Pass ``n_train_pos=None`` and
    ``layer_1_declared_safe=False`` (the defaults) to reproduce the legacy
    fixed-threshold behaviour: ``hblp_effective_z_threshold`` falls through
    to ``base_threshold`` when ``n_train_pos`` is unset (treated as the
    reference N — no inflation) AND when ``layer_1_declared_safe`` is
    False (1.0x multiplier).

    Returns the severity-tagged input dict for the degenerate-score case
    (z is NaN / None / non-finite); callers should treat that as "no
    adversarial signal" and let the voter abstain or the bypass paths
    emit a legacy info verdict.

    The ``p_value`` propagated into the verdict dict is the plus-one
    (Phipson & Smyth) upper-tail permutation p-value from
    ``compute_adversarial_score``; its floor is ``1 / (1 + n_permutations)``
    (default 200 → floor ~0.00498) and it is therefore NEVER exactly 0.0
    (backlog #11.b). Severity routing here uses ``z_score`` only, so the
    p_value is informational for downstream consumers.

    Args:
        score: Raw output from ``compute_adversarial_score``.
        n_train_pos: Training-split positive-class count. When None,
            HBLP falls through to no variance inflation (reference-N
            behaviour). Threaded from the orchestrator at
            ``adaptive_validity_check`` so every per-feature call sees
            the same cohort positive count.
        layer_1_declared_safe: True iff the feature's manifest contract
            declared ``knowable_at <= index_date`` (Layer 1 cleared it).
            False for features without a manifest entry OR whose contract
            declared post-anchor inputs (those would have been caught by
            Layer 1 and never reach this path). Threaded from the
            orchestrator's ``lookup_feature_contract`` lookup.
    """
    z = score.get("z_score", float("nan"))
    auc = score.get("actual_auc", float("nan"))
    null_mean = score.get("null_mean", float("nan"))

    # Codex review HIGH (H3, 2026-05-08): explicit ``z_score=None`` (or
    # any non-numeric value) used to crash on the ``z > HIGH_Z``
    # comparison with TypeError. The dict.get(default=NaN) only catches
    # the *missing* case — a None VALUE bypasses the default. Treat any
    # non-finite/non-numeric z as the degenerate-score case so the
    # bypass path emits a severity=info verdict instead of crashing
    # the whole node.
    z_is_degenerate = (
        z is None
        or not isinstance(z, (int, float))
        or isinstance(z, bool)
        or (isinstance(z, float) and np.isnan(z))
    )

    if z_is_degenerate:
        # Degenerate score (e.g., constant feature → identical AUC under
        # all permutations, or malformed input from a custom scorer).
        # The voter has no signal to act on; the bypass path emits a
        # severity=info verdict matching legacy behaviour.
        severity = "info"
        remediation = "keep"
        evidence = (
            f"Adversarial score undefined (degenerate; actual_auc={auc}, null_mean={null_mean})"
        )
        z_input: Optional[float] = None
    else:
        # Plan v4 §2 G3: route through hblp_classify. The classifier
        # produces severity ∈ {"high", "moderate", "info"} using the
        # HBLP-effective thresholds; we map severity → remediation +
        # evidence string. ``n_train_pos`` falls back to the reference
        # N (no inflation) when unset — preserves legacy 5σ/3σ
        # behaviour for callers that don't thread cohort metadata.
        # Tier 1 invariant: ``hblp_classify`` accepts the additional
        # optional ``delta_auc`` kwarg for the issue #194 joint check;
        # legacy z-only callers see legacy behaviour because the floor
        # only fires when ``delta_auc`` is finite.
        effective_n_pos = (
            int(n_train_pos)
            if n_train_pos is not None and n_train_pos > 0
            else T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N
        )
        # Issue #194 — thread ``delta_AUC = actual_auc - null_mean``
        # into the classifier so the joint check ``(z > k) AND
        # (|delta_AUC| > epsilon)`` fires for severity ∈ {moderate,
        # high}. ``compute_adversarial_score`` always populates both
        # fields when the score is non-degenerate; we recompute the
        # difference here rather than carry it through the score dict
        # to keep ``compute_adversarial_score``'s output schema stable.
        # If either input is non-finite (only possible if a custom
        # scorer fills the score dict by hand), the classifier sees
        # ``delta_auc=None`` and falls through to z-only behaviour.
        if (
            isinstance(auc, (int, float))
            and isinstance(null_mean, (int, float))
            and np.isfinite(auc)
            and np.isfinite(null_mean)
        ):
            delta_auc_arg: Optional[float] = float(auc) - float(null_mean)
        else:
            delta_auc_arg = None
        classification = hblp_classify(
            z_score=float(z),
            n_positives=effective_n_pos,
            layer_1_declared_safe=bool(layer_1_declared_safe),
            delta_auc=delta_auc_arg,
        )
        severity = classification["severity"]
        high_eff = classification["effective_high_threshold"]
        moderate_eff = classification["effective_moderate_threshold"]
        hblp_relaxed = classification["hblp_relaxed"]
        relaxation_note = (
            f" [HBLP-relaxed: high_eff={high_eff:.2f}σ "
            f"(base={HIGH_Z}σ × {classification['variance_inflation_factor']:.2f} "
            f"× layer_1_factor={classification['layer_1_factor']:.2f}), "
            f"n_train_pos={effective_n_pos}, "
            f"layer_1_declared_safe={bool(layer_1_declared_safe)}]"
            if hblp_relaxed
            else ""
        )
        # Issue #194 — joint-check audit footnote. When the joint check
        # has fired (severity forced from {high, moderate} → info), the
        # evidence string must record BOTH the z-evidence and the
        # |delta_AUC|-floor decision so downstream audit readers can see
        # why a feature with z above HIGH_Z was nonetheless kept.
        joint_check_footnote = (
            f" [joint check #194: |delta_AUC|={abs(float(delta_auc_arg or 0.0)):.4f} "
            f"≤ floor {LAYER5_DELTA_AUC_FLOOR_DEFAULT:.4f}; "
            f"z above HBLP band but absolute effect below pharma-actionable threshold]"
            if classification.get("delta_auc_below_floor") and delta_auc_arg is not None
            else ""
        )
        if severity == "high":
            remediation = "drop"
            evidence = (
                f"Layer 3 adversarial discriminator: z={z:.2f}σ above null "
                f"(actual_auc={auc:.4f}, null_mean={null_mean:.4f}); "
                f"{high_eff:.2f}σ HBLP-effective threshold exceeded → drop"
                f"{relaxation_note}"
            )
        elif severity == "moderate":
            remediation = "ambiguous"
            evidence = (
                f"Layer 3 adversarial discriminator: z={z:.2f}σ "
                f"(between {moderate_eff:.2f}σ and {high_eff:.2f}σ HBLP-effective); "
                f"ambiguous → queued for Layer 4 causal-role classification"
                f"{relaxation_note}"
            )
        else:  # severity == "info"
            remediation = "keep"
            evidence = (
                f"Layer 3 adversarial discriminator: z={z:.2f}σ "
                f"(below {moderate_eff:.2f}σ HBLP-effective noise floor); "
                f"legitimate weak signal{relaxation_note}{joint_check_footnote}"
            )
        z_input = float(z)

    # Issue #194 codex pass-1 LOW-1: thread joint-check audit fields
    # through the adversarial-input dict so structured-sidecar consumers
    # (audit JSON, dashboards, regression-test fixtures) can branch on
    # the joint-check decision without parsing the human-readable
    # ``evidence`` string. Always populated:
    #   - delta_auc: signed float, or None when classifier had no input
    #   - delta_auc_floor: float, the active floor at decision time
    #   - delta_auc_below_floor: bool, True iff joint check fired
    # The degenerate-z path (z_is_degenerate above) returns the three
    # fields as None / 0.0 / False because hblp_classify wasn't called.
    if z_is_degenerate:
        ax_delta_auc: Optional[float] = None
        ax_delta_auc_floor: float = float(LAYER5_DELTA_AUC_FLOOR_DEFAULT)
        ax_delta_auc_below_floor: bool = False
        # Degenerate z → no z-only classification to recover; the
        # pre-joint-check severity matches the final ``info`` severity.
        ax_severity_pre_joint_check: str = "info"
    else:
        ax_delta_auc = classification.get("delta_auc")
        ax_delta_auc_floor = float(
            classification.get("delta_auc_floor", LAYER5_DELTA_AUC_FLOOR_DEFAULT)
        )
        ax_delta_auc_below_floor = bool(classification.get("delta_auc_below_floor", False))
        # Issue #212 — propagate the z-only severity (pre-joint-check
        # clamp) so the orchestrator's Layer 4 trigger can fire on the
        # raw signal even when issue #194 has downgraded the final
        # severity to ``info``. Default to the final severity when the
        # classifier didn't publish a pre-joint-check field (e.g.,
        # alternative classifier shims in tests). NOTE: this does NOT
        # affect the final ``severity`` field, the joint-check audit
        # bar, or any downstream consumer of ``adv_input["severity"]``;
        # it is a parallel audit channel.
        ax_severity_pre_joint_check = str(classification.get("severity_pre_joint_check", severity))

    return {
        "layer": "3",
        "severity": severity,
        # Issue #212 — z-only severity classification before issue #194
        # joint-check ``|delta_AUC| <= floor`` clamp. Used by the
        # orchestrator's Layer 4 trigger so the LLM verdict still fires
        # for legitimate weak signals (3σ < z ≤ 5σ but |delta_AUC| ≤ 0.10)
        # that #194 forces to ``info`` in the final verdict. See module
        # docstring "Per-layer ordering" section.
        "severity_pre_joint_check": ax_severity_pre_joint_check,
        "remediation": remediation,
        "evidence": evidence,
        "z_score": z_input,
        "actual_auc": float(auc) if not (isinstance(auc, float) and np.isnan(auc)) else None,
        "null_mean": float(null_mean)
        if not (isinstance(null_mean, float) and np.isnan(null_mean))
        else None,
        "null_std": score.get("null_std"),
        "p_value": score.get("p_value"),
        "n_permutations": score.get("n_permutations"),
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1).
        "delta_auc": ax_delta_auc,
        "delta_auc_floor": ax_delta_auc_floor,
        "delta_auc_below_floor": ax_delta_auc_below_floor,
        # Plan v4 §2 G3 / codex MED-5: tag the adversarial-input dict so
        # ``_compose_legacy_verdict`` can verify the severity classifi-
        # cation came from ``hblp_classify`` (not a hand-rolled legacy
        # ``if z > HIGH_Z`` ladder, which would dodge the wiring guard
        # at runtime). The tag is set unconditionally — the degenerate-
        # z path also routes severity through this function so the same
        # invariant holds.
        "_hblp_classified": True,
    }


def _marginal_effect_size(score: dict[str, Any]) -> float:
    """Signed marginal effect ``actual_auc - null_mean`` for the FDR effect axis.

    The confident set intersects BH-rejection with ``|effect| > floor``; on the
    always-on marginal path the effect is how far the feature's folded AUC sits
    above its permutation-null mean. Returns NaN when either field is
    missing/non-numeric (a degenerate score) — ``fdr_confident_set`` treats NaN
    as non-confident.
    """
    auc = score.get("actual_auc")
    null_mean = score.get("null_mean")
    if auc is None or null_mean is None:
        return float("nan")
    try:
        return float(auc) - float(null_mean)
    except (TypeError, ValueError):
        return float("nan")


def _apply_fdr_firing_override(
    adv_input: dict[str, Any],
    *,
    is_confident: bool,
    fdr_q: float,
) -> dict[str, Any]:
    """Re-decide the auto-fire (HIGH) tier from the FDR confident set.

    Plan v4 Layer-A Phase 1: when the dynamic Benjamini-Hochberg confident set
    is the active firing driver, a feature's ``severity="high"`` (auto-drop)
    decision comes from confident-set membership — NOT the static z>5σ band.
    This wraps the *marginal* σ-band verdict from ``_adversarial_input`` and
    overrides ONLY the HIGH tier:

      * ``is_confident=True``  → severity=high, remediation=drop. FDR confidently
        flags the feature (BH-rejected ∩ ``|delta_AUC|>floor``), so it fires even
        if the static z-threshold only saw moderate/info — the adaptive benefit
        of a cohort-relative FDR decision over a fixed σ-threshold.
      * ``is_confident=False`` AND the σ-band said high → DEMOTE to
        moderate/ambiguous. The feature is suspicious but NOT FDR-confident:
        route to review, do not auto-drop (FDR is the auto-fire authority).
      * otherwise (σ-band moderate/info) → unchanged. The moderate→review band
        (the "ambiguous interior") and the info/keep band stay z-based.

    Applied to the MARGINAL verdict BEFORE the opt-in ablation MAX-rule combine,
    so the joint-model ablation signal (which catches interaction-only leaks the
    marginal permutation cannot) can still escalate a not-confident feature on
    its own merits — FDR governs the marginal tier, not the orthogonal ablation
    escalation. Only the consequential auto-drop decision is FDR-controlled
    (Plan v4 N3: the statistical severity-ladder semantics are otherwise
    preserved). Returns a NEW dict; the input is not mutated. Records
    ``fdr_confident`` (bool) on the result for the audit trail.

    Args:
        adv_input: the per-feature dict from ``_adversarial_input``.
        is_confident: True iff the feature is in the FDR confident set.
        fdr_q: the active FDR level (annotated into the evidence string).
    """
    out = dict(adv_input)
    out["fdr_confident"] = bool(is_confident)
    sigma_severity = out.get("severity")
    base_evidence = out.get("evidence", "")
    if is_confident:
        out["severity"] = "high"
        out["severity_pre_joint_check"] = "high"
        out["remediation"] = "drop"
        out["evidence"] = (
            f"{base_evidence} [FDR firing driver: CONFIDENT leak at "
            f"q={fdr_q:.3g} (BH-rejected ∩ |delta_AUC|>floor) → drop]"
        )
    elif sigma_severity == "high":
        # σ-band flagged high but FDR is NOT confident → demote to review.
        out["severity"] = "moderate"
        out["severity_pre_joint_check"] = "moderate"
        out["remediation"] = "ambiguous"
        out["evidence"] = (
            f"{base_evidence} [FDR firing driver: NOT confident at "
            f"q={fdr_q:.3g} (σ-band high not BH-confirmed) → demoted to review]"
        )
    # else: σ-band moderate/info — unchanged (review / clean tiers stay z-based).
    return out


def _fdr_confident_features(
    bh_eligible: list[str],
    l3_scores: dict[str, Any],
    *,
    q: float,
    n_permutations: int,
    effect_floor: float,
) -> set[str]:
    """Confident-leak set over the FULL eligible BH family.

    A scoring exception (BaseException sentinel) or a missing/degenerate score
    for an eligible feature is kept in the Benjamini-Hochberg family as a
    NON-rejected ``NaN`` p-value — NOT dropped. Dropping it would shrink ``m``
    and LOOSEN the BH threshold ``q/m``, which could falsely promote a
    borderline feature to a confident leak (codex iter-0 HIGH). ``NaN`` p-values
    and effects are tolerated by ``fdr_confident_set`` (never rejected) and keep
    ``m`` equal to the number of eligible hypotheses the permutation budget was
    sized for — the conservative, correct denominator.
    """
    if not bh_eligible:
        return set()
    p_values: list[float] = []
    effect_sizes: list[float] = []
    for feat in bh_eligible:
        score = l3_scores.get(feat)
        if isinstance(score, dict):
            p = score.get("p_value")
            p_values.append(
                float(p)
                if isinstance(p, (int, float)) and not isinstance(p, bool)
                else float("nan")
            )
            effect_sizes.append(_marginal_effect_size(score))
        else:
            # Scoring raised, or no score → a non-rejected NaN that still COUNTS
            # toward m (keeps the BH family size honest).
            p_values.append(float("nan"))
            effect_sizes.append(float("nan"))
    mask = fdr_confident_set(
        p_values,
        effect_sizes,
        q=q,
        n_permutations=n_permutations,
        effect_floor=effect_floor,
    )
    return {feat for feat, is_conf in zip(bh_eligible, mask, strict=True) if bool(is_conf)}


# Map ``EnsembleVerdict.decided_by`` → legacy ``layer`` field for the
# audit-trail JSON sidecar. Phase 2.9 Stage 1 only emits "layer_1" and
# "adversarial"; Stage 2 will add "kg" → "2", Stage 3 will add "llm" → "4".
_DECIDED_BY_TO_LAYER: dict[str, str] = {
    "layer_1": "1",
    "adversarial": "3",
    "adversarial_ablation": "3",
    "kg": "2",
    "llm": "4",
    # Plan v4 Layer B / Phase 2 — the deterministic structural decider replaces
    # the LLM in the Layer-4 slot for attested features, so it maps to layer "4".
    "structural": "4",
    "abstain": "abstain",
}


def _ensemble_to_legacy_dict(
    verdict: EnsembleVerdict,
    *,
    adversarial_input: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Adapt a Phase 2.7 ``EnsembleVerdict`` to the legacy verdict dict.

    Preserves every field the existing downstream consumers
    (``leakage_remediation`` and ``write_adaptive_verdicts_sidecar``) read
    from a Layer 5 verdict, AND appends three new optional fields for the
    Phase 2.7+ audit trail (``decided_by``, ``disagreements``,
    ``kg_signal``).

    Numeric fields (``z_score``, ``actual_auc``, ``null_mean``,
    ``null_std``, ``p_value``, ``n_permutations``) are pulled from
    ``adversarial_input`` when present (the voter doesn't carry them
    through), so the audit JSON sidecar still records the underlying
    permutation-test numbers.

    The ``contract_source`` / ``contract_window_days`` fields are pulled
    from ``verdict.layer_1_input`` (the snapshot the voter took at
    vote-time) — if Layer 1 was the deciding source.
    """
    layer_1 = verdict.layer_1_input or {}
    adv = adversarial_input or {}

    layer_str = _DECIDED_BY_TO_LAYER.get(verdict.decided_by, "abstain")

    # ``EnsembleVerdict.evidence`` is a tuple of lines; the legacy schema
    # carries a single string. Join with "; " so the join is greppable.
    evidence_str = "; ".join(verdict.evidence) if verdict.evidence else ""

    # Codex pass-3 LOW (issue #193): when an LLM verdict was supplied
    # but the voter's deterministic veto won (Layer 1 high or
    # Adversarial high), the legacy dict previously dropped the LLM's
    # role / remediation. The disagreement was recorded in
    # ``disagreements`` (e.g. ``"adversarial=high but llm=ancestor"``),
    # but operators auditing why Layer 4 cost was spent on this feature
    # had no structured field to consume. Surface the LLM's verdict
    # fields explicitly so the audit cost is observable even when
    # adversarial / Layer 1 wins on severity.
    llm_in = verdict.llm_input
    llm_role = getattr(llm_in, "causal_role", None) if llm_in is not None else None
    llm_remediation = (
        getattr(llm_in, "recommended_remediation", None) if llm_in is not None else None
    )
    # Layer-4 evaluator audit-only sidecar (Plan
    # .claude/plans/layer4_evaluator_audit_signal.md). None when the
    # evaluator was disabled / failed / had no worker verdict to read.
    llm_audit = getattr(llm_in, "evaluator_audit", None) if llm_in is not None else None

    # Issue #240 Stage 1 (shadow mode) — iterate the promotion-rule
    # registry to compute three shadow flags from
    # ``(verdict.severity, llm_audit)``. ALL rules are pure functions
    # that read both inputs without mutation; the voter does NOT consume
    # the returned values at Stage 1. Design ref:
    # ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 1.
    #
    # Byte-identity invariant (AC1.2): when every rule returns None,
    # the legacy dict below must be byte-identical (modulo the three
    # shadow keys) to the legacy dict produced before this hook landed.
    # Enforced by tests/integration/test_audit_evaluator_shadow_byte_identity.py.
    #
    # Lazy import (matches the EnsembleVoter/Verdict pattern above) —
    # ``evaluator_promotion_rules`` itself only depends on
    # ``kg.types.LLMEvaluatorAudit`` which is already in the import
    # graph, but the lazy form keeps the top-level surface uniform.
    from src.data.evaluator_promotion_rules import PROMOTION_RULES

    _shadow_results: dict[str, object | None] = {"R1": None, "R2": None, "R3": None}
    for _rule_id, _rule_fn in PROMOTION_RULES:
        _shadow_results[_rule_id] = _rule_fn(verdict.severity, llm_audit)

    return {
        "feature": verdict.feature_name,
        "layer": layer_str,
        # Numeric fields from the adversarial probe (None when no
        # adversarial input was supplied or it was malformed).
        "z_score": adv.get("z_score"),
        "actual_auc": adv.get("actual_auc"),
        "null_mean": adv.get("null_mean"),
        "null_std": adv.get("null_std"),
        "p_value": adv.get("p_value"),
        "n_permutations": adv.get("n_permutations"),
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1).
        # Populated when ``adversarial_input`` carries them; default
        # to None/floor/False when Layer 3 didn't fire.
        "delta_auc": adv.get("delta_auc"),
        "delta_auc_floor": adv.get("delta_auc_floor", LAYER5_DELTA_AUC_FLOOR_DEFAULT),
        "delta_auc_below_floor": bool(adv.get("delta_auc_below_floor", False)),
        # Issue #212 audit field — z-only severity before issue #194
        # joint-check clamp. Equals ``severity`` when the joint check
        # did not fire (no clamp) OR when no Layer 3 signal was
        # produced (degenerate / short-circuit). Always populated so
        # downstream audit consumers can distinguish "Layer 4 fired
        # because pre-joint-check was moderate, joint-clamped to info"
        # from inconsistent layer routing. Default ``"info"`` when
        # ``adversarial_input`` did not carry the field (older callers
        # or bypass paths that don't run hblp_classify).
        "severity_pre_joint_check": adv.get("severity_pre_joint_check", "info"),
        # Issue #196 Phase 3.3 — ablation audit fields. None when the
        # ablation pass was OFF / unable to run / no row for this feature;
        # populated when ``_combine_ablation_with_permutation`` set them.
        "ablation_z_score": adv.get("ablation_z_score"),
        "ablation_delta_auc": adv.get("ablation_delta_auc"),
        "ablation_null_mean": adv.get("ablation_null_mean"),
        "ablation_null_std": adv.get("ablation_null_std"),
        "ablation_severity": adv.get("ablation_severity"),
        # Severity / remediation routed through the voter (or set
        # directly by the bypass paths for short-circuit / info-only).
        "severity": verdict.severity,
        "remediation": verdict.remediation,
        "evidence": evidence_str,
        # Layer 1 contract metadata (None when Layer 1 didn't fire).
        "contract_source": layer_1.get("contract_source"),
        "contract_window_days": layer_1.get("contract_window_days"),
        # Phase 2.7+ audit fields. Always populated.
        "decided_by": verdict.decided_by,
        "disagreements": list(verdict.disagreements),
        "kg_signal": verdict.kg_signal,
        # Phase 2.9 Stage 3 audit (codex pass-3 LOW, issue #193): LLM
        # role/remediation surfaced even when the deterministic veto
        # path wins on severity. ``None`` when no LLM verdict was
        # supplied for this feature.
        "llm_role": llm_role,
        "llm_remediation": llm_remediation,
        # Layer-4 evaluator audit-only fields (Plan
        # .claude/plans/layer4_evaluator_audit_signal.md). All five
        # keys are None when the evaluator is disabled, the evaluator
        # failed, or the worker's LLMVerdict had no evaluator_audit.
        "evaluator_satisfied": llm_audit.satisfied if llm_audit else None,
        "evaluator_rationale_complete": (llm_audit.rationale_complete if llm_audit else None),
        "evaluator_missed_considerations": (llm_audit.missed_considerations if llm_audit else None),
        "evaluator_notes": llm_audit.notes if llm_audit else None,
        "evaluator_model": llm_audit.evaluator_model if llm_audit else None,
        # Issue #241: Layer-4 evaluator telemetry. Same nullability
        # semantics as the 5 audit keys above plus an additional
        # partial-telemetry case: ``latency_ms`` may be non-None while
        # ``input_tokens`` / ``output_tokens`` / ``cost_usd`` are None
        # when the underlying LM emitted no usage block (cache hit, stub
        # LM, etc.). Consumers must treat these as audit-only metrics.
        "evaluator_latency_ms": llm_audit.latency_ms if llm_audit else None,
        "evaluator_input_tokens": llm_audit.input_tokens if llm_audit else None,
        "evaluator_output_tokens": llm_audit.output_tokens if llm_audit else None,
        "evaluator_cost_usd": llm_audit.cost_usd if llm_audit else None,
        # Issue #240 Stage 1 (shadow mode) — three nullable flags
        # populated from the promotion-rule registry above. NULL when
        # the rule's trigger did not fire. The voter does NOT read
        # these fields at Stage 1; they exist for analytics only. See
        # ``src/data/evaluator_promotion_rules.py`` for rule semantics
        # and ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3.
        "would_promote_severity": _shadow_results["R1"],
        "would_flag_for_review": _shadow_results["R2"],
        "rationale_incomplete_flag": _shadow_results["R3"],
        # Issue #240 Stage 3 (env-gated soft-gate) — audit-loop-coupling
        # mitigation (design §5 R-4). ``gate_rule_fired`` names the rule
        # that modulated ``verdict.severity`` inside the voter (only
        # ``"R1"`` today), or None when the gate was disabled / did not
        # fire. ``worker_severity_pre_gate`` recovers the un-mutated worker
        # severity so compile-set curation never trains on a gate-escalated
        # label: when R1 flipped the verdict, the worker severity was
        # "info" by R1's precondition (info→moderate is the only
        # transition the gate performs, reframed 2026-05-25); NULL when no
        # gate fired (then ``verdict.severity`` already IS the worker
        # severity, so the sentinel is None — the same nullable-shadow
        # contract as the three Stage-1 columns above). See
        # ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3/§5 +
        # ``docs/plans/240-r1-reachability-investigation.md``.
        "gate_rule_fired": verdict.gate_rule_fired,
        "worker_severity_pre_gate": ("info" if verdict.gate_rule_fired == "R1" else None),
        # Issue #501 / #240 — leakage × role cross-check (shadow mode).
        # Default None here; the per-feature loop in the node orchestrator
        # overrides this value via an in-loop assignment after
        # ``_compose_legacy_verdict`` returns. The None default ensures
        # schema uniformity: every verdict dict carries the key regardless
        # of whether the orchestrator later sets it to True.
        "would_flag_role_leak_disagreement": None,
        # Issue #501 — M-structure structural-remediation gate (shadow,
        # env-gated by ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED). All four
        # keys default None here; the per-feature loop overrides them via
        # in-loop single-key assignment (mirroring the #508 key above) when a
        # feature carries a ``FeatureContract.causal_structure`` attestation.
        # ``structural_role``: the role the extended extractor derives from the
        # authored DAG fragment (None when un-attested). ``structural_llm_dis-
        # agreement``: True iff structural_role != llm_role (both present).
        # ``structural_remediation_override``: the remediation the gate forced
        # (e.g. "drop"), or None. ``structural_gate_fired``: the rule id
        # ("R-STRUCT") when the env-gated override fired, else None.
        #
        # Plan v4 Layer B / Phase 2: when the voter's STRUCTURAL rule decided this
        # verdict (decided_by="structural"), surface the deterministic role here
        # from ``verdict.final_role`` — the structural decider does NOT run the
        # post-LLM ``_apply_structural_attestation`` telemetry path (guarded off
        # for decided_by="structural"), so this adapter is where the structural
        # role becomes observable. ``structural_unclassifiable`` is True exactly
        # when the rule fired on a malformed attestation (decided_by="structural"
        # AND final_role is None → routed to review).
        "structural_role": (verdict.final_role if verdict.decided_by == "structural" else None),
        "structural_llm_disagreement": None,
        "structural_remediation_override": None,
        "structural_gate_fired": None,
        "structural_unclassifiable": (
            verdict.decided_by == "structural" and verdict.final_role is None
        ),
    }


def _legacy_adversarial_alone_verdict(
    feature: str,
    adversarial_input: dict[str, Any],
) -> dict[str, Any]:
    """Emit a legacy verdict from adversarial-only inputs, bypassing the voter.

    Used when adversarial is the only signal (no Layer 1 contract, no
    KG/LLM). Preserves the legacy ``severity`` / ``remediation`` /
    ``evidence`` exactly as ``_adversarial_input`` produced them — for
    ``info`` severity that's ``keep``, for ``moderate`` it's
    ``ambiguous`` (codex H5 fix: the voter would have rewritten this
    to ``review``, diverging from the legacy contract downstream
    consumers branch on), for ``high`` it's ``drop``.

    Tags ``decided_by="adversarial"`` and the empty-signal KG/disagree
    audit fields. The voter's value-add (cross-source precedence,
    contradiction detection, confidence scoring) is irrelevant when
    adversarial is the only source — the verdict is purely a function
    of the z-score thresholds.
    """
    return {
        "feature": feature,
        "layer": "3",
        "z_score": adversarial_input.get("z_score"),
        "actual_auc": adversarial_input.get("actual_auc"),
        "null_mean": adversarial_input.get("null_mean"),
        "null_std": adversarial_input.get("null_std"),
        "p_value": adversarial_input.get("p_value"),
        "n_permutations": adversarial_input.get("n_permutations"),
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1).
        "delta_auc": adversarial_input.get("delta_auc"),
        "delta_auc_floor": adversarial_input.get("delta_auc_floor", LAYER5_DELTA_AUC_FLOOR_DEFAULT),
        "delta_auc_below_floor": bool(adversarial_input.get("delta_auc_below_floor", False)),
        # Issue #212 — z-only severity before joint-check clamp.
        # Schema-shape consistency with ``_ensemble_to_legacy_dict``.
        "severity_pre_joint_check": adversarial_input.get("severity_pre_joint_check", "info"),
        # Issue #196 Phase 3.3 — ablation audit fields. Same schema as
        # ``_ensemble_to_legacy_dict``; populated from ``adversarial_input``
        # when ``_combine_ablation_with_permutation`` ran, else None.
        "ablation_z_score": adversarial_input.get("ablation_z_score"),
        "ablation_delta_auc": adversarial_input.get("ablation_delta_auc"),
        "ablation_null_mean": adversarial_input.get("ablation_null_mean"),
        "ablation_null_std": adversarial_input.get("ablation_null_std"),
        "ablation_severity": adversarial_input.get("ablation_severity"),
        "severity": adversarial_input.get("severity", "info"),
        "remediation": adversarial_input.get("remediation", "keep"),
        "evidence": adversarial_input.get("evidence", ""),
        "contract_source": None,
        "contract_window_days": None,
        "decided_by": "adversarial",
        "disagreements": [],
        "kg_signal": "no_signal",
        # Codex pass-3 LOW (issue #193): schema-shape consistency with
        # _ensemble_to_legacy_dict. The bypass path never carries an
        # LLM verdict (it's adversarial-alone), so both audit fields
        # are ``None``.
        "llm_role": None,
        "llm_remediation": None,
        # Layer-4 evaluator audit-only fields (Plan
        # .claude/plans/layer4_evaluator_audit_signal.md). Adversarial-only
        # bypass has no LLM verdict, so the evaluator never runs.
        "evaluator_satisfied": None,
        "evaluator_rationale_complete": None,
        "evaluator_missed_considerations": None,
        "evaluator_notes": None,
        "evaluator_model": None,
        # Issue #241: telemetry fields. Bypass paths never invoke the
        # evaluator, so all four are explicitly None (not missing) for
        # sidecar-schema uniformity.
        "evaluator_latency_ms": None,
        "evaluator_input_tokens": None,
        "evaluator_output_tokens": None,
        "evaluator_cost_usd": None,
        # Issue #240 Stage 1 (shadow mode) — adversarial-only bypass
        # has no LLM verdict, so the evaluator never runs and no
        # promotion rule can fire. All three shadow flags are
        # explicitly None for sidecar-schema uniformity (so consumers
        # see the same key set across all four legacy-dict producers).
        "would_promote_severity": None,
        "would_flag_for_review": None,
        "rationale_incomplete_flag": None,
        # Issue #240 Stage 3 — bypass paths never route through the voter,
        # so the env-gated soft-gate cannot fire here. Both gate keys are
        # explicitly None for sidecar-schema uniformity across all four
        # legacy-dict producers.
        "gate_rule_fired": None,
        "worker_severity_pre_gate": None,
        # Issue #501 / #240 — leakage × role cross-check (shadow mode).
        # Adversarial-only bypass has no LLM verdict, so the cross-check
        # cannot fire here. None for sidecar-schema uniformity.
        "would_flag_role_leak_disagreement": None,
        # Issue #501 — M-structure structural keys. Adversarial-only bypass
        # carries no LLM role / contract attestation, so they stay None;
        # present for sidecar-schema uniformity across all producers. Plan v4
        # Layer B / Phase 2: ``structural_unclassifiable`` is None on this bypass
        # — an attested feature would route through the voter (the bypass gate
        # now checks structural inputs), never here.
        "structural_role": None,
        "structural_llm_disagreement": None,
        "structural_remediation_override": None,
        "structural_gate_fired": None,
        "structural_unclassifiable": None,
    }


def _legacy_info_verdict(
    feature: str,
    *,
    adversarial_input: Optional[dict[str, Any]],
    evidence: str,
) -> dict[str, Any]:
    """Emit a legacy info verdict — backward-compat wrapper for callers
    that still construct degenerate-score verdicts directly.

    For the adv-alone path, prefer ``_legacy_adversarial_alone_verdict``;
    that helper preserves whatever severity / remediation
    ``_adversarial_input`` computed (so moderate stays ``ambiguous``,
    not the voter's ``review``). This wrapper is kept for the
    explicit-None-z-score and degenerate-score callers that always
    want ``severity=info, remediation=keep`` regardless of the input
    severity field.
    """
    adv = adversarial_input or {}
    return {
        "feature": feature,
        "layer": "3",
        "z_score": adv.get("z_score"),
        "actual_auc": adv.get("actual_auc"),
        "null_mean": adv.get("null_mean"),
        "null_std": adv.get("null_std"),
        "p_value": adv.get("p_value"),
        "n_permutations": adv.get("n_permutations"),
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1).
        "delta_auc": adv.get("delta_auc"),
        "delta_auc_floor": adv.get("delta_auc_floor", LAYER5_DELTA_AUC_FLOOR_DEFAULT),
        "delta_auc_below_floor": bool(adv.get("delta_auc_below_floor", False)),
        # Issue #212 — schema-shape consistency.
        "severity_pre_joint_check": adv.get("severity_pre_joint_check", "info"),
        # Issue #196 Phase 3.3 — ablation audit fields.
        "ablation_z_score": adv.get("ablation_z_score"),
        "ablation_delta_auc": adv.get("ablation_delta_auc"),
        "ablation_null_mean": adv.get("ablation_null_mean"),
        "ablation_null_std": adv.get("ablation_null_std"),
        "ablation_severity": adv.get("ablation_severity"),
        "severity": "info",
        "remediation": "keep",
        "evidence": evidence,
        "contract_source": None,
        "contract_window_days": None,
        "decided_by": "adversarial",
        "disagreements": [],
        "kg_signal": "no_signal",
        # Codex pass-3 LOW (issue #193): schema-shape consistency with
        # _ensemble_to_legacy_dict.
        "llm_role": None,
        "llm_remediation": None,
        # Layer-4 evaluator audit-only fields (Plan
        # .claude/plans/layer4_evaluator_audit_signal.md). Info-only
        # bypass has no LLM verdict.
        "evaluator_satisfied": None,
        "evaluator_rationale_complete": None,
        "evaluator_missed_considerations": None,
        "evaluator_notes": None,
        "evaluator_model": None,
        # Issue #241: telemetry fields, all None on info-only bypass.
        "evaluator_latency_ms": None,
        "evaluator_input_tokens": None,
        "evaluator_output_tokens": None,
        "evaluator_cost_usd": None,
        # Issue #240 Stage 1 (shadow mode) — info-only bypass has no
        # LLM verdict; shadow flags stay None for schema uniformity.
        "would_promote_severity": None,
        "would_flag_for_review": None,
        "rationale_incomplete_flag": None,
        # Issue #240 Stage 3 — bypass paths never route through the voter,
        # so the env-gated soft-gate cannot fire here. Both gate keys are
        # explicitly None for sidecar-schema uniformity across all four
        # legacy-dict producers.
        "gate_rule_fired": None,
        "worker_severity_pre_gate": None,
        # Issue #501 / #240 — leakage × role cross-check (shadow mode).
        # Info-only bypass has no LLM verdict. None for schema uniformity.
        "would_flag_role_leak_disagreement": None,
        # Issue #501 — M-structure structural keys. Info-only bypass carries
        # no LLM role / attestation; None for schema uniformity. Plan v4 Layer B
        # / Phase 2: structural_unclassifiable is None here (attested features
        # route through the voter, not this bypass).
        "structural_role": None,
        "structural_llm_disagreement": None,
        "structural_remediation_override": None,
        "structural_gate_fired": None,
        "structural_unclassifiable": None,
    }


def _legacy_short_circuit_verdict(feature: str, *, evidence: str) -> dict[str, Any]:
    """Emit a legacy short-circuit verdict (too-few-rows / scoring-error).

    Same shape as ``_legacy_info_verdict`` but with all numeric fields
    set to None — the adversarial probe did not run. ``decided_by`` is
    still tagged ``"adversarial"`` because the *intended* path was
    Layer 3; the audit trail records that the test couldn't fire.
    """
    return {
        "feature": feature,
        "layer": "3",
        "z_score": None,
        "actual_auc": None,
        "null_mean": None,
        "null_std": None,
        "p_value": None,
        "n_permutations": None,
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1).
        # Short-circuit path: classifier never ran, so the fields
        # default to None/floor/False — the audit JSON sidecar still
        # sees the field present (schema uniformity), just unpopulated.
        "delta_auc": None,
        "delta_auc_floor": float(LAYER5_DELTA_AUC_FLOOR_DEFAULT),
        "delta_auc_below_floor": False,
        # Issue #212 — schema-shape consistency. Short-circuit never
        # invokes hblp_classify, so the pre-joint-check severity is
        # the same ``info`` placeholder as the final severity.
        "severity_pre_joint_check": "info",
        # Issue #196 Phase 3.3 — ablation audit fields. None on the
        # short-circuit path; the ablation pass also cannot run when
        # the per-feature row count was below MIN_LAYER3_SAMPLES.
        "ablation_z_score": None,
        "ablation_delta_auc": None,
        "ablation_null_mean": None,
        "ablation_null_std": None,
        "ablation_severity": None,
        "severity": "info",
        "remediation": "keep",
        "evidence": evidence,
        "contract_source": None,
        "contract_window_days": None,
        "decided_by": "adversarial",
        "disagreements": [],
        "kg_signal": "no_signal",
        # Codex pass-3 LOW (issue #193): schema-shape consistency with
        # _ensemble_to_legacy_dict.
        "llm_role": None,
        "llm_remediation": None,
        # Layer-4 evaluator audit-only fields (Plan
        # .claude/plans/layer4_evaluator_audit_signal.md). Short-circuit
        # bypass (too-few-rows / scoring-error) has no LLM verdict.
        "evaluator_satisfied": None,
        "evaluator_rationale_complete": None,
        "evaluator_missed_considerations": None,
        "evaluator_notes": None,
        "evaluator_model": None,
        # Issue #241: telemetry fields, all None on short-circuit bypass.
        "evaluator_latency_ms": None,
        "evaluator_input_tokens": None,
        "evaluator_output_tokens": None,
        "evaluator_cost_usd": None,
        # Issue #240 Stage 1 (shadow mode) — short-circuit bypass has
        # no LLM verdict; shadow flags stay None for schema uniformity.
        "would_promote_severity": None,
        "would_flag_for_review": None,
        "rationale_incomplete_flag": None,
        # Issue #240 Stage 3 — bypass paths never route through the voter,
        # so the env-gated soft-gate cannot fire here. Both gate keys are
        # explicitly None for sidecar-schema uniformity across all four
        # legacy-dict producers.
        "gate_rule_fired": None,
        "worker_severity_pre_gate": None,
        # Issue #501 / #240 — leakage × role cross-check (shadow mode).
        # Short-circuit bypass has no LLM verdict. None for schema uniformity.
        "would_flag_role_leak_disagreement": None,
        # Issue #501 — M-structure structural keys. Short-circuit bypass
        # carries no LLM role / attestation; None for schema uniformity. Plan v4
        # Layer B / Phase 2: structural_unclassifiable is None here (attested
        # features route through the voter via the extended bypass gate).
        "structural_role": None,
        "structural_llm_disagreement": None,
        "structural_remediation_override": None,
        "structural_gate_fired": None,
        "structural_unclassifiable": None,
    }


def _apply_structural_attestation(
    verdict: dict[str, Any],
    contract: Optional[FeatureContract],
) -> None:
    """Issue #501 — M-structure structural-remediation gate (shadow, env-gated).

    Mutates ``verdict`` IN PLACE via single-key assignments (mirrors the #508
    leak-crosscheck pattern). NEVER reassigns the dict, so #508's key and every
    precomputed field are preserved. No-op (all four structural keys stay None,
    remediation untouched) when:

    * the feature carries no ``contract.causal_structure`` attestation; OR
    * the extended ``extract_role`` derives a role equal to the LLM role (no
      disagreement); OR
    * the env switch ``ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED`` is OFF (then
      the *telemetry* keys ``structural_role`` / ``structural_llm_disagreement``
      are still recorded for analytics, but NO remediation override is applied
      and ``structural_gate_fired`` stays None — dark-launchable).

    When attested AND disagreeing: always records ``structural_role`` +
    ``structural_llm_disagreement`` (shadow telemetry, gate-independent). The
    remediation override + ``structural_gate_fired="R-STRUCT"`` are applied ONLY
    when the env switch is on (the gate's ACTING behaviour). Severity is NEVER
    mutated (the reachable seam is remediation, per §4.1; R1's severity path and
    the byte-identity invariant are undisturbed).
    """
    if verdict.get("decided_by") == "structural":
        # Plan v4 Layer B / Phase 2: the voter's STRUCTURAL rule already decided
        # this verdict from the SAME authored edges (decided_by="structural").
        # Re-running the post-LLM telemetry/override here would double-apply (and
        # could re-narrow remediation) — skip it (one decision, one code path).
        return
    if contract is None or contract.causal_structure is None:
        return

    # Derive the structural role from the authored DAG fragment via the shared
    # pure helper (deterministic, zero LLM cost) — the SAME code path the pre-LLM
    # decider uses, so graph-building lives in one place. ``contract`` is non-None
    # and carries ``causal_structure`` here (guarded above).
    from src.data.kg.ensemble_voter import (
        apply_structural_remediation_gate,
        structural_gate_enabled,
    )

    structural_role, structural_err = derive_structural_role(contract)
    if structural_err is not None:
        logger.warning(
            "adaptive_validity_check: structural attestation for %r could not be "
            "classified (%s); skipping structural gate",
            verdict.get("feature"),
            structural_err,
        )
        return

    llm_role = verdict.get("llm_role")
    # Shadow telemetry (gate-independent): always record the derived role and
    # whether it disagrees with the LLM role.
    verdict["structural_role"] = structural_role
    disagreement = (
        structural_role is not None and llm_role is not None and structural_role != llm_role
    )
    verdict["structural_llm_disagreement"] = disagreement if llm_role is not None else None

    if not structural_gate_enabled():
        # Dark-launch: telemetry recorded, but no remediation override.
        return

    override = apply_structural_remediation_gate(
        structural_role=structural_role,
        llm_role=llm_role if isinstance(llm_role, str) else None,
        current_remediation=verdict.get("remediation"),
        llm_remediation=(
            verdict.get("llm_remediation")
            if isinstance(verdict.get("llm_remediation"), str)
            else None
        ),
    )
    if override is not None:
        logger.info(
            "structural_gate: R-STRUCT narrowed remediation %r→%r for feature %r "
            "(structural_role=%s, llm_role=%s)",
            verdict.get("remediation"),
            override,
            verdict.get("feature"),
            structural_role,
            llm_role,
        )
        verdict["remediation"] = override
        verdict["structural_remediation_override"] = override
        verdict["structural_gate_fired"] = "R-STRUCT"


def _compose_legacy_verdict(
    feature: str,
    *,
    voter: EnsembleVoter,
    layer_1_input: Optional[dict[str, Any]] = None,
    adversarial_input: Optional[dict[str, Any]] = None,
    adversarial_score: Optional[dict[str, Any]] = None,
    short_circuit_evidence: Optional[str] = None,
    kg_edges: Iterable["KGEdge"] = (),
    feature_entity_ids: Iterable[str] = (),
    target_entity_ids: Iterable[str] = (),
    kg_mode: Optional[str] = None,
    n_train_pos: Optional[int] = None,
    layer_1_declared_safe: Optional[bool] = None,
    llm_verdict: Optional["LLMVerdict"] = None,
    structural_role: Optional["CausalRole"] = None,
    structural_unclassifiable: bool = False,
) -> dict[str, Any]:
    """Compose one legacy verdict dict from the per-source inputs.

    Routes through ``EnsembleVoter`` for cases that involve a real
    precedence decision (Layer 1 contract present, or KG signal
    available, or LLM verdict supplied, or adversarial severity
    high/moderate). Bypasses the voter for two cases the voter would
    otherwise abstain on:

    1. ``short_circuit_evidence`` is set (too-few-rows, scoring-error)
       → emit ``_legacy_short_circuit_verdict``.
    2. Only signal is adversarial=info → emit ``_legacy_info_verdict``
       so the audit trail records "tested and passed", not "abstain".

    The voter is the authority on every other case.

    Stage 2 update: ``kg_edges`` + ``feature_entity_ids`` +
    ``target_entity_ids`` are forwarded to ``voter.vote(...)``. Empty
    defaults preserve Stage 1 behavior — the voter's KG path is a
    no-op.

    Stage 3 update (Phase 2.9 Stage 3 — Layer 4 LLM wiring): an optional
    ``llm_verdict`` (an :class:`src.data.kg.types.LLMVerdict`) is forwarded
    to ``voter.vote(...)``. When present, it triggers the voter's
    LLM-with-KG-cross-check precedence rule and the resulting verdict is
    tagged ``decided_by="llm"`` (mapped to ``layer="4"`` in the legacy
    schema). Passing ``llm_verdict=None`` preserves Stage 1/2 behaviour —
    no LLM input is considered. The bypass to
    ``_legacy_adversarial_alone_verdict`` is gated on
    ``llm_verdict is None`` so a moderate adversarial signal paired with
    an LLM verdict routes through the voter (which is the only path that
    can emit ``decided_by="llm"`` for the audit trail).

    Plan v4 §2 G3 wiring (post-2026-05-10):
      * ``n_train_pos`` and ``layer_1_declared_safe`` are threaded from
        the orchestrator so the Layer 3 severity classification routes
        through ``hblp_classify`` (the HBLP-effective z-thresholds
        that compensate for low-N permutation null variance and apply
        the declared-safe prior). Both are optional: when unset, the
        underlying classifier falls through to legacy 5σ/3σ behaviour
        (no relaxation). The legacy ``if z > HIGH_Z`` branch is removed
        — there is no parallel path.
      * Per codex MED-5, ``_compose_legacy_verdict`` OWNS classification.
        Two entry points:
          (a) Pass ``adversarial_score`` (raw output of
              ``compute_adversarial_score``); this function calls
              ``_adversarial_input(score, n_train_pos=...,
              layer_1_declared_safe=...)`` itself, threading the
              cohort metadata. This is the codex MED-5 recommended path:
              `_compose_legacy_verdict` invokes hblp_classify directly.
          (b) Pass a pre-classified ``adversarial_input``. In this case
              the dict MUST carry the ``_hblp_classified=True`` tag (set
              by ``_adversarial_input`` unconditionally). Pre-classified
              inputs WITHOUT the tag are REJECTED with a
              ``_HblpRoutingViolationError`` — they would dodge the
              wiring-guard at runtime.

        Passing both ``adversarial_score`` and ``adversarial_input``
        is a programmer error and raises ``ValueError``.
    """
    # codex MED-5: enforce HBLP routing for any Layer 3 input.
    if adversarial_score is not None and adversarial_input is not None:
        raise ValueError(
            "_compose_legacy_verdict: pass exactly one of `adversarial_score` "
            "(raw — this function classifies) OR `adversarial_input` (pre-"
            "classified by `_adversarial_input` — must carry "
            "`_hblp_classified=True` tag). Got both."
        )
    if adversarial_score is not None:
        # Path (a): own the classification by calling _adversarial_input
        # ourselves. This guarantees the call chain
        #   _compose_legacy_verdict → _adversarial_input → hblp_classify
        # runs end-to-end so the wiring-guard's AST scan finds the
        # callsite (codex MED-5).
        adversarial_input = _adversarial_input(
            adversarial_score,
            n_train_pos=n_train_pos,
            layer_1_declared_safe=bool(layer_1_declared_safe),
        )
    elif adversarial_input is not None and not adversarial_input.get("_hblp_classified", False):
        # Path (b) violation: pre-classified input lacks the HBLP tag.
        # Reject so a hand-rolled legacy classifier can't dodge the wiring
        # guard at runtime (codex MED-5).
        raise _HblpRoutingViolationError(
            f"_compose_legacy_verdict: `adversarial_input` for {feature!r} "
            "lacks `_hblp_classified=True` tag — every Layer 3 severity "
            "classification MUST route through `_adversarial_input` (which "
            "calls `hblp_classify`). Pass `adversarial_score=` instead so "
            "this function classifies, OR build the input via "
            "`_adversarial_input(...)`. The wiring guard's AST scan only "
            "verifies static callsites; this runtime check rejects "
            "pre-classified dicts that bypassed the routing chain."
        )

    # Plan v4 Layer B / Phase 2: an attested feature must reach the voter even on
    # the short-circuit path — its structural role is data-INDEPENDENT, so a
    # too-few-rows / scoring-error feature still gets its structural decision.
    if (
        short_circuit_evidence is not None
        and structural_role is None
        and not structural_unclassifiable
    ):
        return _legacy_short_circuit_verdict(feature, evidence=short_circuit_evidence)

    # Materialize once so we can both check truthiness and forward without
    # re-iterating an exhausted generator.
    kg_edges_tuple = tuple(kg_edges)
    feature_ids_tuple = tuple(feature_entity_ids)
    target_ids_tuple = tuple(target_entity_ids)

    # When KG edges are present but they don't connect feature ↔ target
    # (kg_signal would be ``no_signal``), forwarding to the voter still
    # triggers rule #6 (adv-moderate-alone → remediation=review) which
    # diverges from the Stage 1 contract downstream JSON consumers branch
    # on (``ambiguous``). Treating no-signal KG as equivalent to "no KG
    # input" preserves the bypass and Stage 1 semantics.
    if kg_edges_tuple:
        from src.data.kg.ensemble_voter import (
            classify_kg_signal as _classify_kg_signal,
        )

        preview, _considered = _classify_kg_signal(
            kg_edges_tuple,
            feature_entity_ids=feature_ids_tuple,
            target_entity_ids=target_ids_tuple,
        )
        if preview == "no_signal":
            kg_edges_tuple = ()

    # Codex H5 fix (Stage 1): bypass when adversarial is the ONLY signal —
    # for ANY severity (info, moderate, high). The voter would otherwise
    # rewrite ``moderate`` remediation from the legacy ``ambiguous`` to
    # ``review``, diverging from the contract downstream JSON consumers
    # branch on. Stage 2: bypass is preserved when KG produces no_signal
    # (the kg_edges_tuple zero-out above) — the voter only adds value when
    # KG produces a real signal.
    # Stage 3 (issue #193): the bypass is ALSO skipped when an LLM
    # verdict is supplied. With an LLM verdict in hand the voter is the
    # only path that can emit ``decided_by="llm"`` for the audit trail,
    # so we route through it even when adversarial would otherwise be
    # the only structured input.
    if (
        layer_1_input is None
        and adversarial_input is not None
        and not kg_edges_tuple
        and llm_verdict is None
        and structural_role is None
        and not structural_unclassifiable
    ):
        return _legacy_adversarial_alone_verdict(feature, adversarial_input)

    # Real cross-source decision needed → route through the voter so
    # the ``EnsembleVerdict`` audit fields (decided_by, disagreements,
    # kg_signal) reflect the precedence rule that fired.
    verdict = voter.vote(
        feature,
        layer_1_verdict=layer_1_input,
        adversarial_verdict=adversarial_input,
        kg_edges=kg_edges_tuple,
        feature_entity_ids=feature_ids_tuple,
        target_entity_ids=target_ids_tuple,
        llm_verdict=llm_verdict,
        structural_role=structural_role,
        structural_unclassifiable=structural_unclassifiable,
    )
    legacy = _ensemble_to_legacy_dict(verdict, adversarial_input=adversarial_input)

    # Stage 2 PR-E shadow-mode gate: when KG decided this verdict but
    # the operator hasn't promoted KG to drop authority, cap severity
    # to "info" and remediation to "keep". The audit fields
    # (decided_by="kg", kg_signal=...) stay intact so divergence
    # measurements (compute_promotion_eligibility) can compare KG vs
    # adversarial outcomes BEFORE promotion. Codex M1: annotate the
    # ``evidence`` text so audit readers see the cap explicitly —
    # otherwise voter.evidence still says "drop" while the legacy
    # severity/remediation say "info"/"keep" (operator confusion).
    if kg_mode == "shadow" and legacy.get("decided_by") == "kg":
        legacy["severity"] = "info"
        legacy["remediation"] = "keep"
        existing_evidence = legacy.get("evidence", "") or ""
        annotation = "[shadow-mode: verdict capped to info/keep; KG not yet promoted]"
        legacy["evidence"] = (
            f"{existing_evidence} {annotation}".strip() if existing_evidence else annotation
        )

    # Issue #212 — joint-check final-severity cap on the LLM-decided
    # path. The Layer 4 trigger now fires on ``severity_pre_joint_check``
    # (pre-joint-clamp z-band), which surfaces an LLM verdict for
    # weak-effect features that issue #194's joint check has clamped
    # to ``info`` in the adversarial input. The voter's LLM path
    # (``EnsembleVoter.vote`` rule 4) maps the LLM role through
    # ``_llm_severity`` and emits the LLM-derived severity on the final
    # verdict — for a leak role that is ``high``/``drop``. Without this
    # cap, an LLM verdict misclassifying a joint-clamped weak signal
    # as a leak would PROMOTE the final verdict to drop, silently
    # relaxing issue #194's downstream bar. That contradicts the
    # documented #194 contract ("when joint check fires, the feature is
    # retained because the absolute-effect floor confirms benign weak
    # signal").
    #
    # Fix: when the adversarial input recorded ``delta_auc_below_floor``
    # (joint check fired on the Layer 3 signal) AND the voter selected
    # the LLM path, cap final ``severity``/``remediation`` to the
    # joint-clamped adversarial values. The LLM audit fields
    # (``decided_by="llm"``, ``layer="4"``, ``llm_role``,
    # ``llm_remediation``, ``disagreements``) remain on the verdict so
    # the operator can see Layer 4 was consulted but its severity was
    # capped by the joint check. ``evidence`` carries an annotation so
    # audit readers can grep the cap explicitly.
    #
    # Bar preservation: this cap is INWARD only (severity high → info,
    # remediation drop → keep). Layer 4 cannot relax info → high via
    # the joint-clamped path. When the LLM agrees the feature is
    # benign (non-leak role), the cap is a no-op (severity already
    # info on both sides).
    #
    # Pre-cap corroboration guard (issue #212 codex pass-2 MED-1
    # follow-on): the cap predicate reads the PERMUTATION pathway's
    # joint-check decision via ``delta_auc_below_floor``. If issue #196
    # ablation pass independently CORROBORATED the signal — i.e.
    # ``ablation_severity`` is in {moderate, high} which already
    # required passing its OWN joint check on ablation_delta_AUC —
    # then the joint-clamped floor was overridden by a second Layer 3
    # sub-test. Capping in that case would silently relax #196's
    # ablation contract too. Skip the cap when ablation independently
    # corroborated.
    ablation_corroborated = adversarial_input is not None and str(
        adversarial_input.get("ablation_severity") or "info"
    ) in {"moderate", "high"}
    if (
        adversarial_input is not None
        and bool(adversarial_input.get("delta_auc_below_floor", False))
        and legacy.get("decided_by") == "llm"
        and not ablation_corroborated
    ):
        # The joint-clamped adversarial severity is the contract floor.
        # adv_input["severity"] is already 'info' here because the
        # joint-clamp downgraded it; we re-read it explicitly to be
        # robust to any future intermediate severity-mutation step.
        joint_clamped_severity = str(adversarial_input.get("severity", "info"))
        joint_clamped_remediation = str(adversarial_input.get("remediation", "keep"))
        # Issue #212 codex pass-2 LOW-1: fire the cap when EITHER
        # severity OR remediation differs from the joint-clamped
        # values. The pre-pass-2 guard only checked severity, missing
        # the case where the LLM emits a non-leak role (severity stays
        # 'info' via ``_llm_severity``) but the voter computes a
        # remediation other than the joint-clamped ``keep`` (e.g.
        # ``keep_with_caveat`` for accept-role with confidence below
        # promotion thresholds). Both severity AND remediation must
        # be reset to the joint-clamped values + the annotation must
        # always appear when the cap condition is reached so audit
        # readers see Layer 4's verdict was capped consistently.
        severity_differs = joint_clamped_severity != legacy.get("severity")
        remediation_differs = joint_clamped_remediation != legacy.get("remediation")
        if severity_differs or remediation_differs:
            existing_evidence = legacy.get("evidence", "") or ""
            cap_annotation = (
                f" [issue #212 cap: LLM verdict ({legacy.get('severity')!r}/"
                f"{legacy.get('remediation')!r}) capped to joint-clamped "
                f"adversarial values ({joint_clamped_severity!r}/"
                f"{joint_clamped_remediation!r}) because "
                f"|delta_AUC| ≤ floor "
                f"{float(adversarial_input.get('delta_auc_floor', 0.10)):.4f}; "
                f"LLM audit fields preserved (decided_by=llm, layer=4)]"
            )
            legacy["severity"] = joint_clamped_severity
            legacy["remediation"] = joint_clamped_remediation
            legacy["evidence"] = (
                f"{existing_evidence}{cap_annotation}"
                if existing_evidence
                else cap_annotation.strip()
            )

    return legacy


_VALID_KG_MODES: frozenset[str] = frozenset({"off", "shadow", "promoted"})


def _resolve_kg_mode(raw: Any) -> str:
    """Coerce a raw scope_spec ``kg_mode`` value to one of the valid modes.

    None / unset → ``"off"``. An unknown but truthy value (e.g. typo
    ``"shadowmode"``) falls back to ``"off"`` with a warning so a
    misconfiguration surfaces as a log line rather than silent
    promotion bypass (codex L1, L2).
    """
    if raw is None:
        return "off"
    if raw in _VALID_KG_MODES:
        return raw  # type: ignore[no-any-return]
    if raw == "":
        # Empty-string is its own special case — treat as 'off' silently
        # (most likely an unset YAML field). No warning to avoid noise.
        return "off"
    logger.warning(
        "kg_mode=%r is not in %s; defaulting to 'off'",
        raw,
        sorted(_VALID_KG_MODES),
    )
    return "off"


def _load_kg_cache(scope_spec: dict[str, Any]) -> Optional[dict[str, list["KGEdge"]]]:
    """Read the KG cache file pointed at by ``scope_spec['kg_cache_path']``.

    Returns a dict mapping ``feature_name -> list[KGEdge]``. Returns
    ``None`` when:
    - ``kg_mode`` is ``"off"`` (or unset, the default) — the cache is
      never opened so KG verdicts never carry a signal (Stage 1 behavior)
    - no ``kg_cache_path`` is configured
    - the configured path does not exist (warn-and-skip)
    - ``kg_mode == "shadow"`` AND any record's manifest/target fingerprint
      does not match the current pipeline state (warn + return None;
      audit-only path tolerates staleness)

    Loaded otherwise (``kg_mode`` in {``"shadow"``, ``"promoted"``}). The
    severity cap that distinguishes shadow from promoted is applied
    later in ``_compose_legacy_verdict``, not here — the cache is
    identical content in both modes; the difference is what the
    pipeline does with the resulting verdict.

    Fingerprint validation policy (closes the KNOWN GAP that previously
    lived here, plan task #6):

    - ``kg_mode == "shadow"`` AND mismatch → log a warning naming the
      offending feature(s) and return ``None`` so the run proceeds
      without KG verdicts. Shadow mode is operator-audit-only; verdicts
      are advisory and a stale cache must not silently feed the
      promotion-readiness metrics.
    - ``kg_mode == "promoted"`` AND mismatch → raise ``KGCacheStaleError``.
      Promoted mode lets KG verdicts DROP features; silently mismatched
      fingerprints could drop the wrong features. The pipeline must
      halt and force an explicit rebuild.
    - Validation is GATED on ``feature_manifest_source`` being set in
      scope_spec. Legacy runs (no manifest source) bypass validation;
      Layer 1 manifest contracts also no-op on those runs, so there's
      no Layer 1 / KG mismatch to catch.
    """
    kg_mode = _resolve_kg_mode(scope_spec.get("kg_mode"))
    if kg_mode == "off":
        return None
    path_str = scope_spec.get("kg_cache_path")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        logger.warning(
            "kg_cache_path %r does not exist — KG verdicts will be skipped this run",
            path_str,
        )
        return None
    # Lazy import — keeps httpx and the manifest registry out of the
    # adaptive_validity_check.py import surface (the EnsembleVoter
    # lazy-import workaround in this module's docstring uses the same
    # rationale; CI's Event-loop-is-closed flake was tracked back to
    # eager httpx imports in the kg package's __init__).
    from src.data.kg.cache import (
        KGCacheStaleError,
        compute_manifest_fingerprint,
        compute_target_codes_fingerprint,
        load_cache,
    )

    records = load_cache(path)

    manifest_source = scope_spec.get("feature_manifest_source")
    if manifest_source and records:
        features = _resolve_manifest_features(manifest_source)
        if features is None:
            # Codex MEDIUM-3 follow-up: a typo at orchestration time
            # ("cs" instead of "csu") would silently bypass validation
            # without this warning. Surfacing the unknown source name
            # gives the operator one log line to grep for.
            logger.warning(
                "kg_cache validation: feature_manifest_source=%r is not in "
                "the registered manifests (%s) — fingerprint validation "
                "skipped for this run, cache treated as 'trusted upstream'. "
                "Verify the manifest source string if this was unexpected.",
                manifest_source,
                ("csu", "optum", "synthetic"),
            )
        else:
            # Critical writer/reader symmetry: the cache builder at
            # ``scripts/build_kg_cache.py:build_cache_for_manifest``
            # computes ``manifest_fp = compute_manifest_fingerprint(features)``
            # over the FULL FeatureContract list (not filtered to
            # ``kg_entity_codes``). It then skips non-entity features
            # at record-emission time. So a non-entity feature added or
            # mutated in the manifest STILL changes ``manifest_fp`` —
            # the reader must hash the same full list, not a filtered
            # subset, or every cache load would mismatch. (Codex review
            # of D2 PR HIGH-1 caught this: hashing only entity features
            # made the reader's fingerprint diverge from every cache
            # the writer ever emitted.)
            current_manifest_fp = compute_manifest_fingerprint(features)
            target_codes = _coerce_target_codes_for_fingerprint(
                scope_spec.get("target_entity_codes") or []
            )
            current_target_fp = compute_target_codes_fingerprint(target_codes)
            mismatches: list[str] = [
                rec.feature_name
                for rec in records
                if rec.manifest_fingerprint_sha8 != current_manifest_fp
                or rec.target_codes_fingerprint_sha8 != current_target_fp
            ]
            if mismatches:
                preview = mismatches[:5]
                more = "..." if len(mismatches) > 5 else ""
                msg = (
                    f"KG cache fingerprint mismatch in {len(mismatches)} of "
                    f"{len(records)} record(s) at {path_str!r} (features: "
                    f"{preview}{more}). Current manifest_fp="
                    f"{current_manifest_fp}, target_fp={current_target_fp}. "
                    f"Rebuild the cache via scripts/build_kg_cache.py."
                )
                if kg_mode == "promoted":
                    raise KGCacheStaleError(msg)
                logger.warning("%s — kg_mode=shadow → skipping cache for this run", msg)
                return None

    return {r.feature_name: list(r.edges) for r in records}


def _coerce_target_codes_for_fingerprint(raw: Any) -> list[tuple[str, str]]:
    """Normalize ``scope_spec['target_entity_codes']`` into the
    ``list[tuple[str, str]]`` shape the cache builder uses.

    Tolerates both list-of-list and list-of-tuple inputs (JSON config
    serializes tuples as lists). Malformed entries are skipped with a
    warning so a typo at orchestration time surfaces as a stale-
    fingerprint warning rather than a TypeError. The fingerprint
    function ignores nothing — it sorts and hashes whatever is fed in,
    so an entry missing here will produce a stable mismatch against
    the cache writer's fingerprint.
    """
    if not raw:
        return []
    out: list[tuple[str, str]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            system, code = entry
            out.append((str(system), str(code)))
        else:
            logger.warning(
                "target_entity_codes (fingerprint): malformed entry %r — "
                "skipped (expected (system, code))",
                entry,
            )
    return out


def _resolve_manifest_features(
    manifest_source: str,
) -> Optional[list["FeatureContract"]]:
    """Resolve a manifest source string ("csu", "optum", "synthetic")
    to its FeatureContract registry list.

    Returns ``None`` when ``manifest_source`` is unknown so the caller
    can bypass fingerprint validation (legacy compatibility for runs
    that point at a custom manifest module not in
    ``MANIFEST_SOURCES``). Cache-staleness in that case is impossible
    to detect from this surface; the caller treats the cache as
    "trusted upstream".

    Issue #356: ``"synthetic"`` is a registered first-class data source
    in ``src.data.manifests.MANIFEST_SOURCES`` (the v5 Gate C2 engineering
    CI manifest that declares ``borderline_genuine_feature``). Earlier
    versions of this resolver omitted the synthetic branch, so synthetic
    runs silently fell through to the legacy "unknown source" fallback —
    bypassing Layer 1 fingerprint validation and producing empty
    role-attribution dicts in ``_derive_role_attributions_safely``. The
    resolver dict must stay in lockstep with ``MANIFEST_SOURCES`` (see
    the ``test_resolve_manifest_features_lockstep_with_manifest_sources_registry``
    drift guard). ``SYNTHETIC_FEATURES`` is a ``dict[str, FeatureContract]``
    (unlike CSU/Optum's ``list[FeatureContract]``); we materialize its
    values into a list to match the resolver's declared return shape.
    """
    registries: dict[str, list[FeatureContract]] = {
        "csu": list(CSU_FEATURES),
        "optum": list(OPTUM_FEATURES),
        "synthetic": list(SYNTHETIC_FEATURES.values()),
    }
    return registries.get(manifest_source)


def compute_promotion_eligibility(
    verdicts: Iterable[dict[str, Any]],
    *,
    n_patients: int,
    min_n_patients: int = 200,
    min_non_abstain_pct: float = 0.95,
    max_disagreement_rate: float = 0.05,
    min_kg_decided_count: int = 1,
) -> dict[str, Any]:
    """Compute KG promotion-readiness metrics over a verdict list.

    Returns a dict with:
    - ``n_features``: total verdicts considered
    - ``n_patients``: cohort size passed by the caller
    - ``non_abstain_pct``: fraction with ``decided_by != "abstain"``
    - ``kg_decided_count``: count of verdicts with ``decided_by == "kg"``
      (codex H2: gate against promoting when KG never fired)
    - ``cross_source_disagreement_rate``: fraction of verdicts with a
      non-empty ``disagreements`` audit list. This counts ALL voter-
      recorded conflicts (Layer 1 vs adversarial, LLM vs adversarial,
      etc.) — NOT specifically KG-vs-adversarial. The voter does not
      track KG-vs-adversarial as a "disagreement" by design (KG cannot
      contradict a deterministic veto), so the broader cross-source
      rate is the closest available proxy. (codex H1 rename.)
    - ``patient_count_pass``: True iff ``n_patients >= min_n_patients``
    - ``passes``: True iff all of:
      * ``patient_count_pass``
      * ``non_abstain_pct >= min_non_abstain_pct``
      * ``cross_source_disagreement_rate <= max_disagreement_rate``
      * ``kg_decided_count >= min_kg_decided_count``

    This is a governance tool — it does not auto-promote. Operators
    review the metrics on a shadow-mode run, then update the cohort's
    ``scope_spec.kg_mode`` from ``"shadow"`` to ``"promoted"`` when the
    rates are satisfactory.

    The ``n_patients`` parameter is required (no default). The design spec
    requires N ≥ 200 patients in the cohort before promotion is meaningful;
    callers MUST supply the cohort size explicitly so an under-powered
    cohort cannot pass on a verdict-only signal. Earlier versions of this
    function punted this guard to the caller (codex N2 caller-responsibility);
    this version builds the guard in so it cannot be silently skipped.

    Raises ``ValueError`` if ``n_patients`` is negative.

    Empty input returns ``passes=False`` (no evidence ⇒ cannot promote).
    """
    if n_patients < 0:
        raise ValueError(
            f"n_patients must be non-negative; got {n_patients}. "
            "If the cohort size is unknown, that itself is a promotion blocker."
        )

    patient_count_pass = n_patients >= min_n_patients

    verdicts_list = list(verdicts)
    n = len(verdicts_list)
    if n == 0:
        return {
            "n_features": 0,
            "n_patients": n_patients,
            "non_abstain_pct": 0.0,
            "kg_decided_count": 0,
            "cross_source_disagreement_rate": 0.0,
            "patient_count_pass": patient_count_pass,
            "passes": False,
        }

    non_abstain = sum(1 for v in verdicts_list if v.get("decided_by") != "abstain")
    non_abstain_pct = non_abstain / n

    kg_decided_count = sum(1 for v in verdicts_list if v.get("decided_by") == "kg")

    # See ``cross_source_disagreement_rate`` field docstring above for
    # what this captures (and what it does NOT capture).
    disagreements_count = sum(1 for v in verdicts_list if (v.get("disagreements") or []))
    disagreement_rate = disagreements_count / n

    passes = (
        patient_count_pass
        and non_abstain_pct >= min_non_abstain_pct
        and disagreement_rate <= max_disagreement_rate
        and kg_decided_count >= min_kg_decided_count
    )

    return {
        "n_features": n,
        "n_patients": n_patients,
        "non_abstain_pct": non_abstain_pct,
        "kg_decided_count": kg_decided_count,
        "cross_source_disagreement_rate": disagreement_rate,
        "patient_count_pass": patient_count_pass,
        "passes": passes,
    }


def _parse_target_entity_codes(raw: Any) -> tuple[str, ...]:
    """Extract target CUI/ID strings from raw scope_spec input.

    ``scope_spec['target_entity_codes']`` arrives as a list of
    ``(system, code)`` pairs but is *not* validated by
    ``FeatureContract.__post_init__`` (it's runner-injected, not contract
    metadata). Bare ``code for _, code in target_codes`` would raise
    ``ValueError`` on malformed entries (1- or 3-element lists), crashing
    the node. This helper walks the raw input, accepts only well-formed
    2-tuples, and warns on the rest so a typo at orchestration time
    surfaces as a log warning rather than a pipeline crash.
    """
    if not raw:
        return ()
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            _system, code = entry
            out.append(str(code))
        else:
            logger.warning(
                "target_entity_codes: malformed entry %r — skipped (expected (system, code))",
                entry,
            )
    return tuple(out)


def _build_verdict(
    feature: str,
    score: dict[str, Any],
    *,
    voter: Optional["EnsembleVoter"] = None,
    n_train_pos: Optional[int] = None,
    layer_1_declared_safe: bool = False,
    structural_role: Optional["CausalRole"] = None,
    structural_unclassifiable: bool = False,
) -> dict[str, Any]:
    """Backward-compat wrapper for the legacy Layer 3 verdict builder.

    Now flows through ``_compose_legacy_verdict`` so both call sites
    (this node's main loop AND any remaining external test importers)
    produce the same shape, including the new audit fields.

    Plan v4 §2 G3 wiring (post-2026-05-10): ``n_train_pos`` and
    ``layer_1_declared_safe`` are threaded into ``_adversarial_input``
    so severity classification routes through ``hblp_classify``. Default
    values (``n_train_pos=None``, ``layer_1_declared_safe=False``)
    reproduce legacy fixed-threshold behaviour for callers that have not
    been updated to thread cohort metadata (e.g. ad-hoc tests). The
    orchestrator at ``adaptive_validity_check`` always threads both.
    """
    voter = voter or _get_ensemble_voter_class()()
    adv = _adversarial_input(
        score,
        n_train_pos=n_train_pos,
        layer_1_declared_safe=layer_1_declared_safe,
    )
    # Degenerate-score case: ``_adversarial_input`` returns severity=info
    # with z_score=None. Route via the bypass info path so the legacy
    # "Adversarial score undefined" evidence is preserved exactly.
    # Plan v4 Layer B / Phase 2: a degenerate-score feature that is ALSO attested
    # must still reach the voter for its (data-independent) structural decision —
    # gate the info-alone bypass on the absence of structural inputs (mirrors the
    # _compose_legacy_verdict bypass gates) so _build_verdict callers can't lose
    # the structural decision either.
    if (
        adv.get("z_score") is None
        and adv.get("severity") == "info"
        and structural_role is None
        and not structural_unclassifiable
    ):
        return _legacy_info_verdict(
            feature,
            adversarial_input=adv,
            evidence=adv.get("evidence", ""),
        )
    return _compose_legacy_verdict(
        feature,
        voter=voter,
        layer_1_input=None,
        adversarial_input=adv,
        n_train_pos=n_train_pos,
        layer_1_declared_safe=layer_1_declared_safe,
        structural_role=structural_role,
        structural_unclassifiable=structural_unclassifiable,
    )


def _short_circuit_verdict(feature: str, *, evidence: str) -> dict[str, Any]:
    """Backward-compat wrapper for the short-circuit emission path."""
    return _legacy_short_circuit_verdict(feature, evidence=evidence)


# ============================================================================
# Issue #196 — Phase 3.3 ablation helpers.
# ============================================================================


# Issue #196 — ablation strong-effect threshold. The ablation null
# distribution is built by SHUFFLING THE FEATURE COLUMN (not labels) inside
# ``compute_feature_ablation``. For features whose VALUE distribution is
# independent of target but whose ROW-ALIGNMENT with another feature is the
# leak vector (e.g., redundant-noise-cancel pairs, sign-stratified
# variance shifts), the column-shuffle null produces ``perm_delta_auc``
# close to the actual ``delta_auc`` — so the z-score collapses to ~0 even
# when the feature is a clear leak (|delta_AUC| huge). Without an
# absolute-effect escape, the AND-rule ``z > k AND |delta_AUC| > floor``
# would miss exactly the interaction-pair leak class ablation is
# supposed to catch.
#
# Escape: when ``delta_AUC > LAYER5_ABLATION_STRONG_EFFECT_DEFAULT`` AND
# ``z`` is non-NaN, classify severity=high regardless of z magnitude.
# This is the absolute-effect analog of the ``z=+inf`` escape in
# ``hblp_classify`` (issue #194 codex pass-1 MED-1) — when the effect
# magnitude is large enough that no reasonable null could produce it,
# the z-test's failure is the test's bug, not the signal's.
#
# Codex pass-1 MED-1: REQUIRES ``z`` non-NaN. A NaN z indicates the
# permutation null could not be built; with no statistical anchor the
# ablation sub-test has no contract and degrades to permutation-only.
#
# Codex pass-1 MED-2: SIGNED escape. ``delta_auc = full_auc -
# ablated_auc`` > 0 means the feature ADDS to joint AUC (leak-carrier).
# A NEGATIVE delta means removing the feature IMPROVES the joint model,
# typically model-instability noise from a multicollinear nuisance
# variable — NOT a leak. Using ``abs(delta_AUC)`` would false-flag those.
#
# Calibration: at this threshold the joint model's predictive power
# falls by more than 30 AUC points when the feature is removed. With
# ``full_auc ≤ 1.0`` and ``delta_AUC = full_auc - ablated_auc > 0.30``,
# algebra gives ``ablated_auc < full_auc - 0.30 ≤ 0.70`` (codex pass-2
# LOW-2 algebra fix; pre-fix the doc said ``< 0.50``, which assumed an
# additional ``ablated_auc ≥ 0.50`` bound that is not structurally
# guaranteed). The qualitative signature is still strong: a real
# predictor with delta_AUC > 0.30 has measurably dominant joint-model
# contribution. The MODERATE band lowers the bar to floor=0.10.
LAYER5_ABLATION_STRONG_EFFECT_DEFAULT: float = 0.30


def _classify_ablation_severity(
    ablation_row: dict[str, Any],
    *,
    z_threshold: float = HIGH_Z,
    delta_auc_floor: float = LAYER5_DELTA_AUC_FLOOR_DEFAULT,
    strong_effect_threshold: float = LAYER5_ABLATION_STRONG_EFFECT_DEFAULT,
) -> str:
    """Map an ablation per-feature row to a severity tag.

    Two-tier rule (issue #196, refined per codex pass-1 to address
    column-shuffle null weakness on interaction-pair leaks):

      0. Degradation contract: when ``delta_auc`` OR ``z_score`` is
         NaN / None / non-numeric, return ``"info"``. The ablation
         sub-test has no statistical anchor; the verdict falls back
         to permutation-only.

      A. Strong-effect escape (positive delta primary signal):
         when ``delta_AUC > strong_effect_threshold`` (default 0.30)
         AND ``z`` is non-NaN, classify severity=high regardless of
         z magnitude. The column-shuffle null inside
         ``compute_feature_ablation`` produces poor z-scores on
         interaction-pair leaks (the null delta ≈ actual delta), so
         |delta_AUC| is the load-bearing signal for that class.
         SIGNED escape (codex MED-2): NEGATIVE delta = "model improves
         when feature dropped" = nuisance/multicollinearity, NOT a leak.

      B. Joint-check ladder (issue #194 framing, AND-rule; pass-2 MED-1
         extended the signed-delta requirement here):
         when ``delta_AUC > floor`` AND z passes the band:
           * ``z > z_threshold`` AND ``delta_AUC > floor`` → high.
           * ``MODERATE_Z < z <= z_threshold`` AND ``delta_AUC > floor`` → moderate.
         When ``z=+inf`` (degenerate null) AND ``delta_AUC > floor``,
         severity=high (mirror of hblp_classify's MED-1 escape).
         Negative delta cannot reach any of these paths (symmetric with
         case A); the ladder gate is signed, not absolute.

      C. Default: severity=info (below-floor delta, etc.).

    Returns one of ``"high"``, ``"moderate"``, ``"info"``. The
    MODERATE/HIGH bands match the permutation Layer-3 ladder so the
    MAX-rule combination in ``_combine_ablation_with_permutation`` can
    reason on a unified severity scale.
    """
    z = ablation_row.get("z_score")
    delta_auc = ablation_row.get("delta_auc")

    if delta_auc is None:
        return "info"
    if not isinstance(delta_auc, (int, float)) or isinstance(delta_auc, bool):
        return "info"
    delta_f = float(delta_auc)
    if np.isnan(delta_f):
        return "info"

    # Codex pass-1 MED-1: require z is NOT NaN even for the strong-effect
    # escape. A NaN z indicates the permutation null could not be built
    # (e.g., every permuted retrain failed → null_std=NaN per
    # ``compute_feature_ablation`` lines 220/227 in
    # ``src/data/adversarial_leakage.py``). When the null itself is
    # undefined the ablation sub-test has no statistical anchor; per the
    # documented degradation contract ("NaN signals fall back to
    # permutation-only"), we must classify info. The +inf z escape below
    # handles the "null collapsed to zero variance" case separately —
    # that is a DEFINED null with degenerate variance, not an undefined
    # null. NaN means undefined.
    z_is_nan = (
        z is None
        or not isinstance(z, (int, float))
        or isinstance(z, bool)
        or (isinstance(z, float) and np.isnan(z))
    )
    if z_is_nan:
        return "info"

    # Strong-effect escape (case A). The z-score is irrelevant here —
    # ``delta_auc > strong_effect_threshold`` means dropping the feature
    # destroys >30% of the joint model's AUC, which is structurally
    # impossible for legitimate weak predictors at any cohort size.
    #
    # Codex pass-1 MED-2: positive-only escape. ``delta_auc = full_auc -
    # ablated_auc`` (per ``compute_feature_ablation`` line 201 in
    # ``src/data/adversarial_leakage.py``); ``delta_auc > 0`` means the
    # feature ADDS to joint AUC (the leak-carrier case). A large
    # NEGATIVE delta means removing the feature IMPROVES the joint
    # model, which is model-instability noise (often a multicollinear
    # nuisance variable causing LR coefficient explosion), NOT a leak.
    # Using ``abs(delta_f)`` here would false-flag noisy nuisance
    # features as leaks. The conservative escape is signed.
    if delta_f > float(strong_effect_threshold):
        return "high"

    # Below the strong-effect threshold, fall back to the z-anchored
    # joint-check ladder (case B). Requires POSITIVE delta_auc > floor.
    # ``z`` was already verified non-NaN above (codex MED-1 guard).
    #
    # Codex pass-2 MED-1: positive-only delta requirement EXTENDED to the
    # z-band ladder. Pre-pass-2 the ladder used ``abs(delta_f) > floor``,
    # which would classify ``delta_auc=-0.20, z=6.0`` as high via the
    # ``z > z_threshold`` branch — contradicting the MED-2 rationale
    # (negative delta = model improves when feature dropped = nuisance,
    # not leak). Symmetric signed requirement across both case A and B.
    above_floor = delta_f > float(delta_auc_floor)
    if not above_floor:
        return "info"

    z_f = float(z)  # type: ignore[arg-type]  # guarded above

    # +inf-with-strong-effect escape (issue #194 codex MED-1 mirror).
    # ``delta_auc > floor`` already checked above, so this only fires for
    # positive-delta + degenerate-null cases.
    if not np.isfinite(z_f):
        return "high" if z_f > 0 else "info"

    if z_f > float(z_threshold):
        return "high"
    if z_f > MODERATE_Z:
        return "moderate"
    return "info"


_SEVERITY_RANK: dict[str, int] = {"info": 0, "moderate": 1, "high": 2}


def _combine_ablation_with_permutation(
    perm_input: dict[str, Any],
    ablation_row: Optional[dict[str, Any]],
    *,
    z_threshold: float = HIGH_Z,
    delta_auc_floor: float = LAYER5_DELTA_AUC_FLOOR_DEFAULT,
    strong_effect_threshold: float = LAYER5_ABLATION_STRONG_EFFECT_DEFAULT,
) -> dict[str, Any]:
    """Combine permutation-anchored input with the per-feature ablation row.

    Combination rule: MAX over (permutation severity, ablation severity)
    using the rank ``high > moderate > info``.

    When ablation escalates severity strictly above the permutation's
    severity, the returned input dict is rewritten with:
      * ``severity`` / ``remediation`` updated to the escalated values
        (``high`` → drop, ``moderate`` → ambiguous).
      * ``evidence`` text extended with an audit footnote that names the
        ablation signal and the joint-check decision (so audit readers
        can see the escalation source without parsing severity alone).
      * Five audit-trail fields populated from ``ablation_row``:
          ``ablation_z_score`` / ``ablation_delta_auc`` / ``ablation_null_mean``
          / ``ablation_null_std`` / ``ablation_severity``.
      * The ``_hblp_classified=True`` invariant is PRESERVED so the
        wiring-guard in ``_compose_legacy_verdict`` does not reject the
        dict. The classification chain runs permutation first
        (``_adversarial_input → hblp_classify``); ablation is an
        ESCALATION applied on top, never a bypass.

    When ablation does NOT escalate (ablation severity ≤ permutation
    severity), the original dict is returned unchanged (perm_input is
    mutated in place — same shape) but the five ablation-audit fields
    are STILL populated (always when ablation_row is provided), so audit
    consumers can branch on "ablation ran but agreed" vs "ablation did
    not run".

    Returns the (possibly mutated) input dict.
    """
    if ablation_row is None:
        # Ablation pass didn't run for this feature — leave perm_input
        # alone but populate the audit fields as None for schema
        # uniformity.
        perm_input.setdefault("ablation_z_score", None)
        perm_input.setdefault("ablation_delta_auc", None)
        perm_input.setdefault("ablation_null_mean", None)
        perm_input.setdefault("ablation_null_std", None)
        perm_input.setdefault("ablation_severity", None)
        return perm_input

    ablation_sev = _classify_ablation_severity(
        ablation_row,
        z_threshold=z_threshold,
        delta_auc_floor=delta_auc_floor,
        strong_effect_threshold=strong_effect_threshold,
    )
    perm_sev = str(perm_input.get("severity", "info"))

    # Populate audit fields unconditionally — readers see "ran and
    # contributed" vs "ran but agreed/lost" via severity comparison.
    perm_input["ablation_z_score"] = ablation_row.get("z_score")
    perm_input["ablation_delta_auc"] = ablation_row.get("delta_auc")
    perm_input["ablation_null_mean"] = ablation_row.get("null_mean")
    perm_input["ablation_null_std"] = ablation_row.get("null_std")
    perm_input["ablation_severity"] = ablation_sev

    perm_rank = _SEVERITY_RANK.get(perm_sev, 0)
    ablation_rank = _SEVERITY_RANK.get(ablation_sev, 0)
    if ablation_rank <= perm_rank:
        # Permutation's severity already wins (or ties) — no escalation.
        # Original evidence stays intact; perm pathway already routed
        # through hblp_classify so the invariants hold.
        return perm_input

    # Ablation escalates. Map severity → remediation matching the
    # permutation ladder.
    if ablation_sev == "high":
        new_remediation = "drop"
    elif ablation_sev == "moderate":
        new_remediation = "ambiguous"
    else:
        # Cannot reach: rank only escalates from info → moderate or
        # info/moderate → high. ``ablation_sev`` cannot be ``"info"``
        # here because that would not exceed ``perm_rank``.
        new_remediation = perm_input.get("remediation", "keep")

    ablation_z = ablation_row.get("z_score")
    ablation_delta = ablation_row.get("delta_auc")
    z_str = (
        f"{float(ablation_z):.2f}σ"
        if isinstance(ablation_z, (int, float))
        and not isinstance(ablation_z, bool)
        and np.isfinite(float(ablation_z))
        else f"{ablation_z}"
    )
    delta_str = (
        f"{abs(float(ablation_delta)):.4f}"
        if isinstance(ablation_delta, (int, float))
        and not isinstance(ablation_delta, bool)
        and np.isfinite(float(ablation_delta))
        else f"{ablation_delta}"
    )
    footnote = (
        f" [Layer-3 ablation escalated severity from '{perm_sev}' to "
        f"'{ablation_sev}': ablation_z={z_str}, |ablation_delta_AUC|="
        f"{delta_str} > floor {float(delta_auc_floor):.4f} (issue #196 "
        f"MAX-rule, joint check applied symmetrically)]"
    )
    existing_evidence = perm_input.get("evidence", "") or ""
    perm_input["evidence"] = (
        f"{existing_evidence}{footnote}" if existing_evidence else footnote.strip()
    )
    perm_input["severity"] = ablation_sev
    perm_input["remediation"] = new_remediation
    # Issue #212 codex pass-2 MED-1: ``severity_pre_joint_check`` must
    # also reflect ablation escalation so the orchestrator's Layer 4
    # trigger fires on ablation-escalated interaction-only leaks.
    # Without this, a feature whose permutation z lands in info but
    # whose ablation z escalates to moderate would skip the LLM
    # review that the pre-#212 post-combine severity gate would have
    # invoked.
    #
    # Ablation runs ``_classify_ablation_severity`` which already
    # applies the issue #194 joint check symmetrically (delta_AUC
    # floor + signed-positive contract per #196 pass-1 MED-2). So
    # an ablation severity of ``moderate``/``high`` is GUARANTEED to
    # have crossed the joint check (or the strong-effect escape with
    # |delta_AUC| > 3x floor). The escalated pre-joint-check
    # severity matches the escalated final severity in that case —
    # they are the same value because the ablation classifier
    # doesn't produce a pre-joint-check / post-joint-check split.
    perm_pre_rank = _SEVERITY_RANK.get(str(perm_input.get("severity_pre_joint_check", perm_sev)), 0)
    if ablation_rank > perm_pre_rank:
        perm_input["severity_pre_joint_check"] = ablation_sev
    return perm_input


def _run_ablation_pass(
    train_df: pd.DataFrame,
    target: str,
    feature_names: list[str],
    *,
    binary_label_mask: pd.Series,
    n_permutations: int,
    seed: int,
    model_factory: Optional[Any] = None,
) -> Optional[dict[str, dict[str, Any]]]:
    """Run ``compute_feature_ablation`` once over the active feature set.

    Returns a dict mapping ``feature_name → per_feature row`` (with the
    ``delta_auc`` / ``z_score`` / ``null_mean`` / ``null_std`` fields).
    Returns ``None`` when the ablation cannot run (insufficient rows for
    a joint model, single-class target after masking, or training error
    at the full-model stage). Per-feature failures inside
    ``compute_feature_ablation`` are tolerated — the per-feature row will
    carry NaN ``delta_auc`` / ``z_score`` and ``_classify_ablation_severity``
    will return ``"info"`` for those rows (silent degradation to
    permutation-only for that feature).
    """
    mask = binary_label_mask.copy()
    # Restrict to rows where ALL active features are non-null, so the
    # joint model has a consistent design matrix. NaN-imputation would
    # change the ablation null distribution shape; dropping rows is
    # cheaper and matches what real model_trainer pipelines do.
    for feat in feature_names:
        mask = mask & train_df[feat].notna()
    if mask.sum() < MIN_LAYER3_SAMPLES:
        logger.warning(
            "Layer-3 ablation pass skipped: only %d rows survived joint-mask "
            "intersection across %d active features (need >= %d)",
            int(mask.sum()),
            len(feature_names),
            MIN_LAYER3_SAMPLES,
        )
        return None

    X = train_df.loc[mask, feature_names].astype(float)
    y = train_df.loc[mask, target].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        logger.warning(
            "Layer-3 ablation pass skipped: target has < 2 classes after "
            "joint-mask intersection (n_rows=%d, n_pos=%d)",
            int(mask.sum()),
            int((y == 1).sum()),
        )
        return None

    try:
        result = compute_feature_ablation(
            X,
            y,
            model_factory=model_factory,
            n_permutations=n_permutations,
            seed=seed,
            z_threshold=HIGH_Z,
        )
    except Exception as exc:
        logger.warning(
            "Layer-3 ablation pass failed at full-model stage (%s) — falling "
            "back to permutation-only Layer 3",
            exc,
        )
        return None

    per_feat = result.get("per_feature", []) or []
    # Re-derive ``delta_auc = full_auc - ablated_auc`` row-wise so it
    # matches the meaning ``_classify_ablation_severity`` expects
    # (the function already stores ``delta_auc`` per-row; we just
    # surface the existing field). null_mean/std/z_score already present.
    return {row["feature"]: row for row in per_feat if row.get("feature") is not None}


_MANIFEST_FORBIDDEN_BY_SOURCE: dict[str, list[str]] = {
    "csu": CSU_FORBIDDEN_AS_FEATURES,
    "optum": OPTUM_FORBIDDEN_AS_FEATURES,
    # v5 Gate C2: synthetic manifest has no forbidden columns by design.
    # Registered explicitly so ``_select_features`` does NOT log the
    # "unknown manifest_source" warning when synthetic runs opt in.
    "synthetic": SYNTHETIC_FORBIDDEN_AS_FEATURES,
}


def _select_features(
    df: pd.DataFrame,
    target: str,
    excluded: list[str],
    manifest_source: Optional[str] = None,
) -> list[str]:
    """Return the feature columns Layer 3 should evaluate.

    - Excludes the target itself.
    - Excludes columns the scope spec already declared excluded (PII, declared leakage).
    - Excludes manifest-declared post-index / target-coupled columns when
      a known ``manifest_source`` is supplied. This is the proactive
      counterpart to the Layer 1 contract audit downstream: forbidden
      columns no longer reach Layer 3 scoring at all, saving compute and
      providing defense-in-depth so a Layer 1 verdict bug cannot let a
      forbidden column through to model training. Unknown / None
      ``manifest_source`` values fall through to the legacy behaviour
      (no manifest-based exclusion) so synthetic regimes that share
      column names with CSU/Optum are not penalised.
    - Excludes non-numeric columns: Layer 3 needs a continuous score for AUC, and
      categorical handling routes through ``check_categorical_class_separation``
      in the legacy detector. Categorical adaptive scoring is a Layer 5 follow-up.
    """
    # Use pandas' is_numeric_dtype, not np.issubdtype: the latter raises
    # `TypeError: Cannot interpret 'Int64Dtype()' as a data type` on pandas
    # extension dtypes (Int64/Float64/boolean). Any DataFrame ingested from
    # Supabase/SQLAlchemy with nullable-int schema would crash the node.
    excluded_set = set(excluded or [])
    excluded_set.add(target)
    if manifest_source is not None:
        # Codex M1 (PR #92 review): a typo or future-cohort value would
        # silently fall through to legacy behaviour, defeating the
        # defense-in-depth objective with no operator signal. Warn once
        # per call so an operator who misspelt ``feature_manifest_source``
        # in scope_spec can spot the issue before the run completes.
        #
        # v5 Gate C2: distinguish "known manifest with no forbidden
        # columns by design" (e.g., the synthetic manifest) from "unknown
        # manifest source typo" by membership check, not truthiness — an
        # empty list is a valid registration.
        if manifest_source in _MANIFEST_FORBIDDEN_BY_SOURCE:
            excluded_set.update(_MANIFEST_FORBIDDEN_BY_SOURCE[manifest_source])
        else:
            logger.warning(
                "_select_features: unknown manifest_source %r — no "
                "manifest forbidden-list applied (known sources: %s). "
                "Layer 1 audit downstream will still catch contract "
                "violations, but the proactive defense-in-depth pass "
                "was skipped for this run.",
                manifest_source,
                sorted(_MANIFEST_FORBIDDEN_BY_SOURCE.keys()),
            )
    cols = []
    for c in df.columns:
        if c in excluded_set:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


async def adaptive_validity_check(state: dict[str, Any]) -> dict[str, Any]:
    """Run Layer 3 adversarial discriminator on every feature; emit verdicts.

    Args:
        state: Current DataPreparerState (dict-like).

    Returns:
        Dict with state updates:
        - ``adaptive_verdicts``: list of verdict dicts (one per evaluated feature).
        - ``adaptive_flagged_features``: features at ``severity=high`` (z > 5σ).
        - ``leaked_features``: union of pre-existing flagged set + new flags.
        - ``leakage_findings``: pre-existing list extended with adaptive verdicts.
    """
    train_df = state.get("train_df")
    scope_spec = state.get("scope_spec") or {}
    target = scope_spec.get("prediction_target")
    excluded = scope_spec.get("excluded_features", []) or []
    # Layer 1 (manifest-driven contracts) is opt-in per cohort. Scenario_a
    # and other synthetic regimes leave this unset; CSU/Optum runners set
    # ``feature_manifest_source`` in scope_spec so only the matching manifest
    # is consulted. Without this guard the manifest matches any column that
    # happens to share a name across cohorts (e.g., scenario_a's constant
    # ``brand="Kisqali"`` would hit the CSU manifest's post-index contract
    # and halt the pipeline).
    manifest_source = scope_spec.get("feature_manifest_source")

    # Graceful no-op cases
    if train_df is None or target is None or target not in getattr(train_df, "columns", []):
        logger.info("adaptive_validity_check: no target/train_df → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    # Layer 1 (manifest-driven) operates on ALL columns regardless of dtype —
    # the contract is metadata, not data. Layer 3 (statistical) requires a
    # numeric AUC, so non-numeric columns can only be caught by Layer 1.
    excluded_set = set(excluded or [])
    excluded_set.add(target)
    all_columns = [c for c in train_df.columns if c not in excluded_set]
    numeric_candidates = _select_features(train_df, target, excluded, manifest_source)

    if not all_columns:
        logger.info("adaptive_validity_check: no candidate columns → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    # Build a per-row target-validity mask. For a binary classification target
    # we accept ONLY {0, 1}; integer sentinels like -1 (unknown outcome) would
    # otherwise pass the `pd.isna` check (integers can't be NaN), reach
    # `roc_auc_score` as a 3-class input, raise ValueError, get caught, and
    # silently produce severity=info verdicts for every numeric feature —
    # turning Layer 3 into a complete blind spot.
    target_arr = train_df[target].to_numpy()
    target_notna = ~pd.isna(target_arr)
    binary_label_mask = pd.Series(
        np.isin(target_arr, [0, 1]) & target_notna,
        index=train_df.index,
    )
    n_invalid = int((~binary_label_mask).sum() - (~target_notna).sum())
    if n_invalid > 0:
        logger.warning(
            "adaptive_validity_check: target %r has %d rows with non-binary "
            "values (sentinels?); these rows are excluded from Layer 3 scoring",
            target,
            n_invalid,
        )
    valid_target_values = target_arr[binary_label_mask.to_numpy()]
    if len(np.unique(valid_target_values)) < 2:
        logger.info("adaptive_validity_check: target has < 2 classes → skipping")
        return {
            "adaptive_verdicts": [],
            "adaptive_flagged_features": [],
        }

    # Plan v4 §2 G3 wiring: compute n_train_pos once at orchestrator entry
    # so every per-feature ``_compose_legacy_verdict`` call routes through
    # ``hblp_classify`` with the same cohort positive count. Counted from
    # the binary-label mask (sentinels excluded) so HBLP's variance-
    # inflation factor reflects the actual N used in Layer 3 scoring.
    n_train_pos = int(np.sum(valid_target_values == 1))

    # Use explicit `is not None` checks: `state.get(...) or DEFAULT` silently
    # replaces a legitimate 0 with the default (Python's falsy-zero semantics).
    # `adaptive_seed=0` is a valid seed; the old form returned 7 instead.
    _n_perms = state.get("adaptive_n_permutations")
    n_perms = int(_n_perms) if _n_perms is not None else DEFAULT_PERMUTATIONS
    _seed = state.get("adaptive_seed")
    seed = int(_seed) if _seed is not None else 7

    # Issue #196 — Phase 3.3 Layer-3 multi-feature ablation config. Default
    # OFF: the joint-model retrain cost is O(n_features) × O(n_permutations)
    # so even at modest widths it adds 10-30 s to a node call. Tuning knobs:
    #   * adaptive_layer3_ablation_enabled (bool, default False): master gate.
    #   * adaptive_ablation_n_permutations (int, default DEFAULT_ABLATION_PERMUTATIONS=50):
    #     smaller than the permutation pass since each round is a full
    #     retrain.
    #   * adaptive_ablation_z_threshold (float, default HIGH_Z=5.0): match
    #     the permutation HIGH band so MAX-rule reasons on a unified scale.
    #   * adaptive_ablation_max_features (int, default DEFAULT_ABLATION_MAX_FEATURES=50):
    #     O(n²) blowup guard. When active-feature count exceeds this, the
    #     entire ablation pass is SKIPPED with a warning (subsetting which
    #     features to ablate would bias the joint-model AUC the survivors
    #     are measured against).
    ablation_enabled = bool(state.get("adaptive_layer3_ablation_enabled", False))
    _ablation_perms = state.get("adaptive_ablation_n_permutations")
    ablation_n_perms = (
        int(_ablation_perms) if _ablation_perms is not None else DEFAULT_ABLATION_PERMUTATIONS
    )
    _ablation_z = state.get("adaptive_ablation_z_threshold")
    ablation_z_threshold = float(_ablation_z) if _ablation_z is not None else HIGH_Z
    _ablation_max = state.get("adaptive_ablation_max_features")
    ablation_max_features = (
        int(_ablation_max) if _ablation_max is not None else DEFAULT_ABLATION_MAX_FEATURES
    )
    # Issue #196 — model_factory escape hatch. Callable returning a fresh
    # sklearn-compatible classifier with predict_proba. None falls through to
    # ``compute_feature_ablation``'s default LogisticRegression — the right
    # call for the linear-leak regime that dominates RWD pipelines. Pass a
    # tree-based factory (DecisionTreeClassifier, GradientBoosting, etc.) to
    # detect interaction-only leaks the linear baseline cannot learn (the
    # integration test fixture uses this path).
    ablation_model_factory = state.get("adaptive_ablation_model_factory")
    _ablation_strong = state.get("adaptive_ablation_strong_effect_threshold")
    ablation_strong_effect = (
        float(_ablation_strong)
        if _ablation_strong is not None
        else LAYER5_ABLATION_STRONG_EFFECT_DEFAULT
    )

    verdicts: list[dict[str, Any]] = []
    flagged: list[str] = []
    voter = _get_ensemble_voter_class()()

    # Issue #501 / #240 — leakage × role cross-check (shadow mode).
    # Build a per-feature lookup from the PRIOR ``leakage_findings`` list
    # (the statistical detect_leakage output already in state) BEFORE this
    # node appends its own adversarial verdicts to the cumulative stream.
    # Using this node's own verdicts would be CIRCULAR (they feed the
    # ensemble vote). The lookup maps feature → max severity among all its
    # statistical findings, so a feature flagged at both moderate and high
    # maps to "high" (the dominant signal).
    #
    # This dict is pure data: no LM call, no I/O. It is consumed once per
    # feature below (after ``_compose_legacy_verdict``) to set
    # ``would_flag_role_leak_disagreement``. Shadow-only: the value is
    # never read by the voter, leakage_severity, or any routing logic.
    #
    # Lazy import follows the existing pattern (see ``_ensemble_to_legacy_dict``
    # importing ``evaluator_promotion_rules``). Imported once here at node
    # entry so the per-feature loop doesn't repeat the import call.
    from src.data.leakage_role_crosscheck import evaluate_role_vs_statistical_leak

    _prior_findings_for_crosscheck: list[dict[str, Any]] = list(state.get("leakage_findings") or [])
    _SEVERITY_RANK = {"critical": 3, "high": 2, "moderate": 1, "info": 0}
    stat_leak_by_feature: dict[str, str] = {}
    for _f in _prior_findings_for_crosscheck:
        if not isinstance(_f, dict):
            continue
        _fname = _f.get("feature")
        _fsev = _f.get("severity")
        if not isinstance(_fname, str) or not isinstance(_fsev, str):
            continue
        if _SEVERITY_RANK.get(_fsev, -1) > _SEVERITY_RANK.get(
            stat_leak_by_feature.get(_fname, ""), -1
        ):
            stat_leak_by_feature[_fname] = _fsev

    # Stage 2 KG wiring (Phase 2.9 PR-D): load offline KG cache once at
    # node entry, reuse the per-feature lookup across both passes.
    # ``None`` means no cache configured / file missing / kg_mode='off'
    # → kg_edges default of ``()`` flows through to the voter (Stage 1
    # behavior).
    kg_cache = _load_kg_cache(scope_spec)
    kg_mode = _resolve_kg_mode(scope_spec.get("kg_mode"))
    target_ids = _parse_target_entity_codes(scope_spec.get("target_entity_codes") or [])

    # Stage 3 LLM wiring (issue #193): load the persisted compiled
    # CausalRoleClassifier once at node entry, reuse across every feature
    # whose adversarial severity comes back ``moderate`` (the "ambiguous"
    # bucket per the module docstring). Returns None when no LM endpoint
    # is configured OR the compiled artifact is missing — in which case
    # Layer 4 silently skips and the verdict goes through the legacy
    # adversarial-alone bypass.
    layer_4_classifier = _try_load_layer_4_classifier()

    def _kg_inputs(
        feat: str, contract: Optional[FeatureContract]
    ) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        """Per-feature KG edges + entity IDs for the voter."""
        edges = tuple((kg_cache or {}).get(feat, ()))
        if contract is None:
            return edges, ()
        feat_ids = tuple(code for _system, code in contract.kg_entity_codes)
        return edges, feat_ids

    # Layer 1 pass — every column, manifest-driven catch for post-index ones.
    # Skipped entirely when ``feature_manifest_source`` is unset (e.g.,
    # synthetic regimes); see scope_spec read at the top of this function.
    # Layer 1 verdicts route through ``_compose_legacy_verdict`` which
    # consults ``EnsembleVoter`` so the audit trail records ``decided_by``
    # consistently with Layer 3.
    layer_1_caught: set[str] = set()
    for feat in all_columns:
        contract = lookup_feature_contract(feat, data_source=manifest_source)
        if contract is not None and not contract.knowable_at.is_pre_or_at_index():
            edges, feat_ids = _kg_inputs(feat, contract)
            verdict = _compose_legacy_verdict(
                feat,
                voter=voter,
                layer_1_input=_layer_1_input(feat, contract),
                kg_edges=edges,
                feature_entity_ids=feat_ids,
                target_entity_ids=target_ids,
                kg_mode=kg_mode,
            )
            # Issue #501 / #240 — leakage × role cross-check (shadow mode).
            # Same as the Layer 3 loop: override the None default from the
            # producer. Layer 1 verdicts rarely carry an LLM role (Layer 4
            # fires only in the Layer 3 loop) so this will almost always
            # remain None, but schema uniformity demands the assignment.
            # ``evaluate_role_vs_statistical_leak`` is imported once at node
            # entry (see stat_leak_by_feature block above).
            verdict["would_flag_role_leak_disagreement"] = evaluate_role_vs_statistical_leak(
                verdict.get("llm_role"),
                stat_leak_by_feature.get(feat),
            )
            verdicts.append(verdict)
            flagged.append(feat)
            layer_1_caught.add(feat)

    # Issue #196 — Phase 3.3 Layer-3 ablation pass (opt-in, default OFF).
    # Run BEFORE the per-feature permutation loop so the joint-model retrain
    # happens once over the full active feature set. The per-feature loop
    # below then looks up each feature's ablation row by name and combines
    # it with the permutation result via MAX-rule.
    ablation_active_features: list[str] = [
        feat for feat in numeric_candidates if feat not in layer_1_caught
    ]
    ablation_results: Optional[dict[str, dict[str, Any]]] = None
    ablation_skipped_reason: Optional[str] = None
    if ablation_enabled:
        if len(ablation_active_features) == 0:
            ablation_skipped_reason = "no active features after Layer 1"
        elif len(ablation_active_features) > ablation_max_features:
            ablation_skipped_reason = (
                f"active-feature count {len(ablation_active_features)} > "
                f"ablation_max_features={ablation_max_features} cap; O(n²) "
                f"blowup guard fired"
            )
            logger.warning("Layer-3 ablation pass skipped: %s", ablation_skipped_reason)
        else:
            ablation_results = _run_ablation_pass(
                train_df,
                target,
                ablation_active_features,
                binary_label_mask=binary_label_mask,
                n_permutations=ablation_n_perms,
                seed=seed,
                model_factory=ablation_model_factory,
            )
            if ablation_results is None:
                ablation_skipped_reason = "ablation pass returned None (see warning above)"
            else:
                logger.info(
                    "Layer-3 ablation pass: scored %d features (n_perms=%d, "
                    "z_threshold=%.2f, |delta_AUC| floor=%.4f)",
                    len(ablation_results),
                    ablation_n_perms,
                    ablation_z_threshold,
                    LAYER5_DELTA_AUC_FLOOR_DEFAULT,
                )

    # ------------------------------------------------------------------
    # Plan v4 Layer-A Phase 1 — dynamic FDR confident set (firing driver).
    # The static z>5σ "high" tier is replaced by a Benjamini-Hochberg confident
    # set over the per-feature plus-one permutation p-values (BH-rejected ∩
    # |delta_AUC| > floor). A plus-one p-value can clear the BH rank-1 threshold
    # q/m only when n_permutations >= ceil(m/q), so we (a) size the permutation
    # budget to that feasibility floor up to a cap, (b) fall back to the static
    # σ-band when a cohort is too wide for the cap (never a silently-empty set),
    # and (c) score every Layer-3-eligible feature ONCE at that budget so the
    # confident set is known before any feature is classified. Default-ON
    # (validated faithfully on the Optum initiation cohort: caught the real
    # treatment_initiated leak, zero false positives on 39 legit features).
    # ------------------------------------------------------------------
    fdr_enabled = bool(state.get("adaptive_fdr_enabled", True))
    _fdr_q = state.get("adaptive_fdr_q")
    fdr_q = float(_fdr_q) if _fdr_q is not None else DEFAULT_FDR_Q
    _fdr_cap = state.get("adaptive_fdr_max_permutations")
    fdr_cap = int(_fdr_cap) if _fdr_cap is not None else DEFAULT_FDR_MAX_PERMUTATIONS

    # Plan v4 Phase 1: the Layer-4 LLM auditor is OFF by default. When False the
    # LLM is never invoked — the FDR confident set + the deterministic voter
    # decide. Set adaptive_layer4_enabled=True to run it as an auditor during the
    # Phase-3 attestation ramp (still audit-only in the voter by default).
    layer4_enabled = bool(state.get("adaptive_layer4_enabled", False))
    # Plan v4 Layer B / Phase 2 — dark-launch flag (default OFF). When False (the
    # production default) no structural derivation runs and the new decision
    # branch is never taken, so the first authored manifest cannot auto-activate
    # it. Declared on DataPreparerState (extra="ignore" drops undeclared keys on
    # the model path), so a future Phase-3 ramp can enable it in production.
    structural_decider_enabled = bool(state.get("adaptive_structural_decider_enabled", False))

    l3_candidates = [feat for feat in numeric_candidates if feat not in layer_1_caught]
    # Cheap pre-scan (no permutations): which candidates clear the min-samples
    # gate? Those are the BH hypotheses; their count m sizes the feasibility
    # floor and each is scored once below.
    l3_masks: dict[str, Any] = {}
    bh_eligible: list[str] = []
    for feat in l3_candidates:
        feat_mask = train_df[feat].notna() & binary_label_mask
        l3_masks[feat] = feat_mask
        if int(feat_mask.sum()) >= MIN_LAYER3_SAMPLES:
            bh_eligible.append(feat)

    fdr_active = False
    if not fdr_enabled:
        fdr_reason = "disabled"
    elif not bh_eligible:
        fdr_reason = "no_eligible_features"
    else:
        n_perms, fdr_active = fdr_permutation_budget(
            len(bh_eligible), fdr_q, default=n_perms, cap=fdr_cap
        )
        fdr_reason = (
            "active"
            if fdr_active
            else (
                f"sigma_fallback: BH floor "
                f"{min_permutations_for_fdr(len(bh_eligible), fdr_q)} > cap {fdr_cap} "
                f"at m={len(bh_eligible)} eligible features"
            )
        )

    # Score every BH-eligible feature ONCE at the chosen budget. A scoring
    # exception is stored as a sentinel so the classify loop emits the same
    # error verdict it produced when scoring was inline.
    l3_scores: dict[str, Any] = {}
    for feat in bh_eligible:
        feat_mask = l3_masks[feat]
        try:
            l3_scores[feat] = compute_adversarial_score(
                train_df[feat][feat_mask].to_numpy(dtype=float),
                train_df.loc[feat_mask, target].to_numpy(dtype=int),
                n_permutations=n_perms,
                seed=seed,
                z_threshold=HIGH_Z,
            )
        except Exception as exc:  # noqa: BLE001 — recorded; re-surfaced as a verdict below
            l3_scores[feat] = exc

    confident_features: set[str] = set()
    if fdr_active:
        # Build the confident set over the FULL eligible family (errored
        # features stay as non-rejected NaN so a scoring exception cannot shrink
        # m and loosen q/m — codex iter-0 HIGH).
        confident_features = _fdr_confident_features(
            bh_eligible,
            l3_scores,
            q=fdr_q,
            n_permutations=n_perms,
            effect_floor=LAYER5_DELTA_AUC_FLOOR_DEFAULT,
        )

    # Layer 3 pass — numeric columns only, skipping anything Layer 1 already caught.
    for feat in numeric_candidates:
        if feat in layer_1_caught:
            continue

        # Plan v4 Layer B / Phase 2: resolve the contract + derive the
        # deterministic structural role at the TOP of the loop — BEFORE the
        # short-circuit gates — because the role is data-INDEPENDENT (from
        # authored edges). A too-few-rows / scoring-error feature WITH an
        # attestation must still get its structural decision (routed through the
        # voter by _compose_legacy_verdict). Resolved ONCE here; the per-feature
        # code below reuses this ``contract`` (the redundant Layer-3 re-lookup
        # was removed).
        #
        # Scope note: this loop is ``numeric_candidates`` minus ``layer_1_caught``
        # — the SAME scope as the Layer-4 LLM, whose ONLY call site
        # (``classify_feature``, below in this loop) shares it. The structural
        # decider replaces the LLM EXACTLY where the LLM runs. Non-numeric /
        # excluded / manifest-forbidden features (filtered out by
        # ``_select_features``) and Layer-1-caught leaks (already decided by the
        # higher-precedence Layer-1 veto, which sits ABOVE structural) are out of
        # the Layer-3/4 decider's scope BY DESIGN — there is no LLM path they
        # could "fall through" to, so this is not a silent drop.
        contract = lookup_feature_contract(feat, data_source=manifest_source)
        structural_role: Optional[CausalRole] = None
        structural_unclassifiable = False
        if structural_decider_enabled:
            _role_str, _structural_err = derive_structural_role(contract)
            structural_unclassifiable = _structural_err is not None
            # extract_role returns exactly the six CausalRole members → cast is sound.
            structural_role = cast("CausalRole", _role_str) if _role_str is not None else None

        col = train_df[feat]
        mask = col.notna() & binary_label_mask
        if mask.sum() < MIN_LAYER3_SAMPLES:
            _sc_verdict = _compose_legacy_verdict(
                feat,
                voter=voter,
                short_circuit_evidence=(
                    f"Skipped: only {int(mask.sum())} non-null rows (need ≥{MIN_LAYER3_SAMPLES})"
                ),
                structural_role=structural_role,
                structural_unclassifiable=structural_unclassifiable,
            )
            # Issue #501 — short-circuit bypasses never carry an LLM role
            # so evaluate_role_vs_statistical_leak always returns None here.
            # Assignment preserves schema uniformity (the key is already None
            # from the producer; this is explicit for audit-trail clarity).
            # ``evaluate_role_vs_statistical_leak`` imported once at node entry.
            _sc_verdict["would_flag_role_leak_disagreement"] = evaluate_role_vs_statistical_leak(
                _sc_verdict.get("llm_role"),
                stat_leak_by_feature.get(feat),
            )
            verdicts.append(_sc_verdict)
            continue

        # Phase 1: read the score computed once in the FDR pre-pass above. A
        # BaseException sentinel means scoring raised — emit the same error
        # verdict the inline try/except produced.
        score = l3_scores.get(feat)
        if isinstance(score, BaseException):
            logger.warning("adaptive_validity_check: scoring failed for %s: %s", feat, score)
            _err_verdict = _compose_legacy_verdict(
                feat,
                voter=voter,
                short_circuit_evidence=f"Adversarial scoring error: {score}",
                structural_role=structural_role,
                structural_unclassifiable=structural_unclassifiable,
            )
            # Issue #501 — same schema-uniformity assignment as above.
            # ``evaluate_role_vs_statistical_leak`` imported once at node entry.
            _err_verdict["would_flag_role_leak_disagreement"] = evaluate_role_vs_statistical_leak(
                _err_verdict.get("llm_role"),
                stat_leak_by_feature.get(feat),
            )
            verdicts.append(_err_verdict)
            continue
        if score is None:
            # Defensive: any feature past the min-samples gate is in l3_scores;
            # None would mean logic drift — skip rather than crash.
            continue

        edges, feat_ids = _kg_inputs(feat, contract)
        # Plan v4 §2 G3 wiring: route severity classification through
        # ``hblp_classify`` by threading cohort metadata into both the
        # adversarial-input builder AND ``_compose_legacy_verdict``. The
        # ``layer_1_declared_safe`` boolean reflects the manifest's
        # ``knowable_at <= index_date`` predicate — True iff Layer 1
        # cleared the feature (manifest contract present AND pre-or-at
        # index). Features without a manifest entry default to False
        # (treat as not-declared-safe; legacy 5σ threshold applies).
        layer_1_declared_safe = bool(
            contract is not None and contract.knowable_at.is_pre_or_at_index()
        )
        adv_input = _adversarial_input(
            score,
            n_train_pos=n_train_pos,
            layer_1_declared_safe=layer_1_declared_safe,
        )

        # Plan v4 Layer-A Phase 1 — FDR confident set drives the HIGH tier.
        # Applied to the MARGINAL verdict BEFORE the ablation combine so the
        # orthogonal joint-model ablation signal can still escalate a
        # not-confident feature on its own merits. No-op when FDR fell back to
        # the static σ-band (fdr_active=False) — the legacy z-band then decides.
        if fdr_active:
            adv_input = _apply_fdr_firing_override(
                adv_input,
                is_confident=feat in confident_features,
                fdr_q=fdr_q,
            )

        # Issue #196 — Phase 3.3 Layer-3 ablation MAX-rule combination.
        # When the opt-in ablation pass produced a per-feature row for this
        # feature, combine its severity with the permutation severity via
        # ``_combine_ablation_with_permutation``. The combination:
        #   * Escalates severity (info → moderate / moderate → high) when
        #     ablation crosses its threshold AND |delta_AUC| > floor (issue
        #     #194 joint check, applied symmetrically).
        #   * Populates 5 audit-trail fields (ablation_z_score,
        #     ablation_delta_auc, ablation_null_mean, ablation_null_std,
        #     ablation_severity) so audit readers see "ran and agreed" vs
        #     "did not run" without needing to parse evidence text.
        #   * Preserves the ``_hblp_classified=True`` invariant — permutation
        #     ran first through ``_adversarial_input → hblp_classify``;
        #     ablation is an ESCALATION applied on top, not a bypass.
        # When ablation is OFF or the ablation pass returned None, the
        # call is a no-op except for setting the 5 audit fields to None.
        pre_combine_severity = adv_input.get("severity")
        ablation_row_for_feat = ablation_results.get(feat) if ablation_results is not None else None
        adv_input = _combine_ablation_with_permutation(
            adv_input,
            ablation_row_for_feat,
            z_threshold=ablation_z_threshold,
            delta_auc_floor=LAYER5_DELTA_AUC_FLOOR_DEFAULT,
            strong_effect_threshold=ablation_strong_effect,
        )
        ablation_escalated = (
            ablation_row_for_feat is not None and adv_input.get("severity") != pre_combine_severity
        )

        # Stage 3 Layer 4 trigger (issue #193): when adversarial severity is
        # ``moderate`` (3σ < z ≤ 5σ — the "ambiguous" bucket per the module
        # docstring) AND a compiled classifier is available, invoke the LLM
        # to disambiguate. The verdict then routes through the voter with
        # ``llm_verdict`` populated, which yields ``decided_by="llm"`` in
        # the audit trail (mapped to ``layer="4"`` via _DECIDED_BY_TO_LAYER).
        #
        # Codex pass-1 MEDIUM-2 (issue #193): ALSO invoke Layer 4 when the
        # adversarial severity is ``high`` AND Layer 1 declared the feature
        # safe (manifest contract knowable_at <= index_date). This is the
        # expensive false-positive case — the voter's deterministic
        # adversarial-high veto still wins on severity (drop), but the
        # ``EnsembleVerdict.disagreements`` audit trail records
        # ``adversarial=high but llm=<accept-role>`` when the LLM agrees
        # with Layer 1's "safe" assessment. Without this, the operator
        # has no Layer 4 signal to triage Layer-1-vs-Layer-3 disagreement.
        #
        # Issue #212 — fire Layer 4 on the PRE-joint-check severity so a
        # weak-effect Layer 3 signal (3σ < z ≤ 5σ but |delta_AUC| ≤ 0.10)
        # that issue #194 forced to ``info`` still surfaces an LLM
        # verdict for the audit trail. The FINAL ``severity`` field
        # (post-joint-check) is unchanged — issue #194's downstream bar
        # is preserved; this only widens the audit-signal channel.
        # Defense in depth: when ``severity_pre_joint_check`` is absent
        # (alt classifier shim / older fixture), fall back to the
        # post-joint-check ``severity`` so the trigger semantics match
        # the pre-#212 behaviour for those callers.
        llm_verdict: Optional[Any] = None
        adv_severity_pre = adv_input.get("severity_pre_joint_check", adv_input.get("severity"))
        # Plan v4 Phase 1: the Layer-4 LLM auditor is OFF by default
        # (adaptive_layer4_enabled). When off, the LLM is never CALLED — the FDR
        # confident set + the deterministic voter decide. When on (the Phase-3
        # ramp auditor) the verdict is STILL audit-only in the voter unless
        # ADAPTIVE_LAYER4_LLM_DECIDES=1 (ensemble_voter._llm_decides_enabled).
        # Plan v4 Layer B / Phase 2: skip the LLM entirely for attested features
        # — extract_role decides via the voter, so the LLM is never the decider
        # (nor even called) for a feature carrying a structural attestation.
        layer_4_should_fire = (
            layer4_enabled
            and structural_role is None
            and not structural_unclassifiable
            and (
                adv_severity_pre == "moderate"
                or (adv_severity_pre == "high" and layer_1_declared_safe)
            )
        )
        if layer_4_classifier is not None and layer_4_should_fire:
            try:
                from src.data.causal_role_classifier_loader import classify_feature

                derivation, ds_context = _build_layer_4_inputs(
                    feat, contract, target, manifest_source
                )
                llm_verdict = classify_feature(
                    feature_name=feat,
                    derivation_pseudocode=derivation,
                    dataset_context=ds_context,
                    classifier=layer_4_classifier,
                )
            except Exception as exc:  # pragma: no cover - defensive
                # Best-effort: log and proceed without LLM. The voter will
                # see ``llm_verdict=None`` and fall through to the
                # non-LLM precedence path.
                logger.warning(
                    "adaptive_validity_check: Layer 4 invocation failed for "
                    "%s: %s — proceeding without LLM verdict",
                    feat,
                    exc,
                )
                llm_verdict = None

        verdict = _compose_legacy_verdict(
            feat,
            voter=voter,
            adversarial_input=adv_input,
            kg_edges=edges,
            feature_entity_ids=feat_ids,
            target_entity_ids=target_ids,
            kg_mode=kg_mode,
            n_train_pos=n_train_pos,
            layer_1_declared_safe=layer_1_declared_safe,
            llm_verdict=llm_verdict,
            structural_role=structural_role,
            structural_unclassifiable=structural_unclassifiable,
        )
        # Issue #196 Phase 3.3 — tag ``decided_by="adversarial_ablation"`` so
        # audit-trail consumers can distinguish "permutation caught it" from
        # "ablation escalated it". Only tag when ablation strictly escalated
        # severity AND the voter did NOT promote a different source
        # (Layer 1 / KG / LLM) past adversarial — the deterministic-veto
        # precedence is preserved. The voter sets ``decided_by`` on the
        # verdict; we overwrite ONLY when the verdict's decided_by is the
        # adversarial path (voter rule for adv-alone or the bypass route).
        if ablation_escalated and verdict.get("decided_by") == "adversarial":
            verdict["decided_by"] = "adversarial_ablation"
        # Issue #501 / #240 — leakage × role cross-check (shadow mode).
        # Override the None default set by ``_compose_legacy_verdict`` (via
        # its producer functions) with the result of the pure cross-check
        # function. Only fires (True) when:
        #   1. The LLM assigned a benign keep-clean role (ancestor /
        #      confounder / instrument) — i.e. ``verdict["llm_role"]`` is
        #      in BENIGN_KEEP_ROLES.
        #   2. The statistical detect_leakage already flagged this feature
        #      at critical/high severity (via ``stat_leak_by_feature``
        #      built at node entry from the PRIOR leakage_findings — never
        #      from this node's own adversarial verdicts, which would be
        #      circular).
        # Shadow-only: this value is never read by the voter, leakage_severity,
        # routing, or any decision-making code. It is written to the sidecar
        # for analytics and curation only. Byte-identity invariant enforced by
        # ``tests/integration/test_leak_crosscheck_shadow_byte_identity.py``.
        # ``evaluate_role_vs_statistical_leak`` imported once at node entry.
        verdict["would_flag_role_leak_disagreement"] = evaluate_role_vs_statistical_leak(
            verdict.get("llm_role"),
            stat_leak_by_feature.get(feat),
        )
        # Issue #501 — M-structure structural-remediation gate (shadow,
        # env-gated). COEXISTS with the #508 leak-crosscheck above: this is a
        # DISJOINT failure mode (intra-LEAK-role subtype misclassification, the
        # #242 collider↔mediator↔descendant correlated failures), gated at the
        # reachable remediation seam (collider narrows remediation to {drop},
        # overriding an LLM's permissive transform/window). Computed via single-
        # key assignment on the EXISTING verdict dict — NEVER reassign
        # ``verdict = {...}`` (that wipes precomputed state, including #508's key
        # just set above). The structural role is derived from the feature's
        # authored ``FeatureContract.causal_structure`` edge list (None when
        # un-attested → no override). ``_apply_structural_attestation`` is a
        # pure helper; the env switch makes the override default-OFF.
        _apply_structural_attestation(verdict, contract)
        verdicts.append(verdict)
        if verdict["severity"] == "high":
            flagged.append(feat)

    # Merge with existing leakage state — augment, don't replace. The
    # graph re-enters this node after leakage_remediation drops columns,
    # so we extend the prior `adaptive_verdicts` and `adaptive_flagged_features`
    # rather than overwriting them; the audit trail spans every invocation.
    #
    # Asymmetry note (backlog #11.d): the legacy `leakage_findings` field
    # is CLEARED on each leakage_remediation re-entry (see leakage_remediation.py
    # — the legacy detector recomputes from scratch each pass). This node's
    # `adaptive_verdicts`, in contrast, are CUMULATIVE across re-entries (we
    # extend, dedup-by-feature-name, first-write-wins). Audit-trail readers
    # MUST account for this when correlating the two streams: a feature
    # present in `adaptive_verdicts` from invocation #1 may be absent from
    # `leakage_findings` after invocation #2 cleared the legacy stream.
    prior_leaked = list(state.get("leaked_features") or [])
    prior_findings = list(state.get("leakage_findings") or [])
    prior_severity = state.get("leakage_severity") or "none"
    prior_verdicts = list(state.get("adaptive_verdicts") or [])
    prior_flagged = list(state.get("adaptive_flagged_features") or [])

    merged_leaked = sorted(set(prior_leaked) | set(flagged))
    merged_findings = prior_findings + verdicts

    # Dedup verdicts by feature name — first verdict wins (the one from the
    # initial invocation, before columns were dropped, has the most evidence).
    seen_features = {v["feature"] for v in prior_verdicts}
    extended_verdicts = list(prior_verdicts)
    for v in verdicts:
        if v["feature"] not in seen_features:
            extended_verdicts.append(v)
            seen_features.add(v["feature"])
    extended_flagged = sorted(set(prior_flagged) | set(flagged))

    # Escalate severity if Layer 3 caught something legacy missed. Severity
    # ordering: critical > high > moderate > info > none. Adaptive only escalates
    # — never downgrades — so the legacy detector's verdict is preserved.
    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "info": 1, "none": 0}
    new_severity = prior_severity
    if flagged and severity_rank.get(prior_severity, 0) < severity_rank["high"]:
        new_severity = "high"

    logger.info(
        "adaptive_validity_check: scored=%d flagged=%d (high) prior_severity=%s new_severity=%s",
        len(verdicts),
        len(flagged),
        prior_severity,
        new_severity,
    )

    update: dict[str, Any] = {
        "adaptive_verdicts": extended_verdicts,
        "adaptive_flagged_features": extended_flagged,
        "leaked_features": merged_leaked,
        "leakage_findings": merged_findings,
        # Plan v4 Layer-A Phase 1 — auditable FDR firing-driver summary. Always
        # present so sidecar/audit consumers can see whether the dynamic
        # confident set drove this run (active) or the σ-band fallback did, the
        # q/budget used, and which features were confidently flagged.
        "leakage_fdr": {
            "active": fdr_active,
            "enabled": fdr_enabled,
            "q": fdr_q,
            "n_permutations": n_perms,
            "n_confident": len(confident_features),
            "confident_features": sorted(confident_features),
            "reason": fdr_reason,
        },
    }
    if new_severity != prior_severity:
        update["leakage_severity"] = new_severity
        update["leakage_detected"] = True
    return update
