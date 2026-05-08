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
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.data.adversarial_leakage import compute_adversarial_score
from src.data.feature_contract import FeatureContract
from src.data.manifests import (
    CSU_FORBIDDEN_AS_FEATURES,
    OPTUM_FORBIDDEN_AS_FEATURES,
    lookup_feature_contract,
)

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
    from src.data.kg.types import EnsembleVerdict, KGEdge


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


logger = logging.getLogger(__name__)


HIGH_Z = 5.0
MODERATE_Z = 3.0
DEFAULT_PERMUTATIONS = 200

# Minimum non-null sample count to run Layer 3 scoring on a feature.
# Below this floor the permutation-baseline z-score is too noisy to be
# reliable, so the feature gets a short-circuit ``severity=info`` verdict
# and is left for downstream review. Promoted from a hardcoded `30` per
# backlog item #11.c so future tightening can change one place.
MIN_LAYER3_SAMPLES = 30


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


def _adversarial_input(score: dict[str, Any]) -> dict[str, Any]:
    """Build a Layer 3 ``EnsembleVoter.vote`` input dict from a raw score.

    Maps the score dict produced by ``compute_adversarial_score`` (z_score,
    actual_auc, null_mean, null_std, p_value, n_permutations) into the
    severity-tagged shape the voter expects. Always populates ``z_score``
    so the voter's M3 audit-integrity guard never fires on our own input.

    Severity routing (matches the legacy ``_build_verdict`` thresholds):

        z > 5σ above null  → severity=high
        3σ < z ≤ 5σ        → severity=moderate
        z ≤ 3σ             → severity=info

    Returns None for the degenerate-score case (z is NaN); callers should
    treat that as "no adversarial signal" and let the voter abstain or the
    bypass paths emit a legacy info verdict.

    The ``p_value`` propagated into the verdict dict is the empirical
    upper-tail proportion from ``compute_adversarial_score``; it is bounded
    below by ``1 / n_permutations`` (default 200 → floor 0.005), so a
    persisted ``p_value=0.0`` means ``< 1/n_permutations``, NOT exact zero
    (backlog #11.b). Severity routing here uses ``z_score`` only, so this
    rounding is purely informational for downstream consumers.
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
    elif z > HIGH_Z:
        severity = "high"
        remediation = "drop"
        evidence = (
            f"Layer 3 adversarial discriminator: z={z:.2f}σ above null "
            f"(actual_auc={auc:.4f}, null_mean={null_mean:.4f}); "
            f"{HIGH_Z}σ governance threshold exceeded → drop"
        )
        z_input = float(z)
    elif z > MODERATE_Z:
        severity = "moderate"
        remediation = "ambiguous"
        evidence = (
            f"Layer 3 adversarial discriminator: z={z:.2f}σ "
            f"(between {MODERATE_Z}σ and {HIGH_Z}σ); ambiguous → "
            f"queued for Layer 4 causal-role classification"
        )
        z_input = float(z)
    else:
        severity = "info"
        remediation = "keep"
        evidence = (
            f"Layer 3 adversarial discriminator: z={z:.2f}σ "
            f"(below {MODERATE_Z}σ noise floor); legitimate weak signal"
        )
        z_input = float(z)

    return {
        "layer": "3",
        "severity": severity,
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
    }


# Map ``EnsembleVerdict.decided_by`` → legacy ``layer`` field for the
# audit-trail JSON sidecar. Phase 2.9 Stage 1 only emits "layer_1" and
# "adversarial"; Stage 2 will add "kg" → "2", Stage 3 will add "llm" → "4".
_DECIDED_BY_TO_LAYER: dict[str, str] = {
    "layer_1": "1",
    "adversarial": "3",
    "kg": "2",
    "llm": "4",
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
        "severity": adversarial_input.get("severity", "info"),
        "remediation": adversarial_input.get("remediation", "keep"),
        "evidence": adversarial_input.get("evidence", ""),
        "contract_source": None,
        "contract_window_days": None,
        "decided_by": "adversarial",
        "disagreements": [],
        "kg_signal": "no_signal",
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
        "severity": "info",
        "remediation": "keep",
        "evidence": evidence,
        "contract_source": None,
        "contract_window_days": None,
        "decided_by": "adversarial",
        "disagreements": [],
        "kg_signal": "no_signal",
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
        "severity": "info",
        "remediation": "keep",
        "evidence": evidence,
        "contract_source": None,
        "contract_window_days": None,
        "decided_by": "adversarial",
        "disagreements": [],
        "kg_signal": "no_signal",
    }


def _compose_legacy_verdict(
    feature: str,
    *,
    voter: EnsembleVoter,
    layer_1_input: Optional[dict[str, Any]] = None,
    adversarial_input: Optional[dict[str, Any]] = None,
    short_circuit_evidence: Optional[str] = None,
    kg_edges: Iterable["KGEdge"] = (),
    feature_entity_ids: Iterable[str] = (),
    target_entity_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Compose one legacy verdict dict from the per-source inputs.

    Routes through ``EnsembleVoter`` for cases that involve a real
    precedence decision (Layer 1 contract present, or KG signal
    available, or adversarial severity high/moderate). Bypasses the
    voter for two cases the voter would otherwise abstain on:

    1. ``short_circuit_evidence`` is set (too-few-rows, scoring-error)
       → emit ``_legacy_short_circuit_verdict``.
    2. Only signal is adversarial=info → emit ``_legacy_info_verdict``
       so the audit trail records "tested and passed", not "abstain".

    The voter is the authority on every other case.

    Stage 2 update: ``kg_edges`` + ``feature_entity_ids`` +
    ``target_entity_ids`` are forwarded to ``voter.vote(...)``. Empty
    defaults preserve Stage 1 behavior — the voter's KG path is a
    no-op.
    """
    if short_circuit_evidence is not None:
        return _legacy_short_circuit_verdict(feature, evidence=short_circuit_evidence)

    # Materialize kg_edges once so we can both check truthiness and
    # forward without re-iterating an exhausted generator.
    kg_edges_tuple = tuple(kg_edges)

    # Codex H5 fix (Stage 1): bypass when adversarial is the ONLY signal —
    # for ANY severity (info, moderate, high). The voter would otherwise
    # rewrite ``moderate`` remediation from the legacy ``ambiguous`` to
    # ``review``, diverging from the contract downstream JSON consumers
    # branch on. Stage 2 narrows the bypass: when KG edges are present,
    # the voter sees a real cross-source decision (KG vs adversarial)
    # so we route through the voter; the bypass only fires when the
    # adversarial verdict is genuinely the only available signal.
    if layer_1_input is None and adversarial_input is not None and not kg_edges_tuple:
        return _legacy_adversarial_alone_verdict(feature, adversarial_input)

    # Real cross-source decision needed → route through the voter so
    # the ``EnsembleVerdict`` audit fields (decided_by, disagreements,
    # kg_signal) reflect the precedence rule that fired.
    verdict = voter.vote(
        feature,
        layer_1_verdict=layer_1_input,
        adversarial_verdict=adversarial_input,
        kg_edges=kg_edges_tuple,
        feature_entity_ids=tuple(feature_entity_ids),
        target_entity_ids=tuple(target_entity_ids),
    )
    return _ensemble_to_legacy_dict(verdict, adversarial_input=adversarial_input)


def _load_kg_cache(scope_spec: dict[str, Any]) -> Optional[dict[str, list["KGEdge"]]]:
    """Read the KG cache file pointed at by ``scope_spec['kg_cache_path']``.

    Returns a dict mapping ``feature_name -> list[KGEdge]``. Returns
    ``None`` when no cache path is configured or the configured path
    doesn't exist (Stage 1 behavior preserved). PR-E adds the
    shadow-vs-promoted policy gate that distinguishes "missing cache
    is fine" from "missing cache is fatal" by mode.
    """
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
    from src.data.kg.cache import load_cache  # lazy: keep httpx out of import surface

    records = load_cache(path)
    return {r.feature_name: list(r.edges) for r in records}


def _build_verdict(
    feature: str,
    score: dict[str, Any],
    *,
    voter: Optional["EnsembleVoter"] = None,
) -> dict[str, Any]:
    """Backward-compat wrapper for the legacy Layer 3 verdict builder.

    Now flows through ``_compose_legacy_verdict`` so both call sites
    (this node's main loop AND any remaining external test importers)
    produce the same shape, including the new audit fields.
    """
    voter = voter or _get_ensemble_voter_class()()
    adv = _adversarial_input(score)
    # Degenerate-score case: ``_adversarial_input`` returns severity=info
    # with z_score=None. Route via the bypass info path so the legacy
    # "Adversarial score undefined" evidence is preserved exactly.
    if adv.get("z_score") is None and adv.get("severity") == "info":
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
    )


def _short_circuit_verdict(feature: str, *, evidence: str) -> dict[str, Any]:
    """Backward-compat wrapper for the short-circuit emission path."""
    return _legacy_short_circuit_verdict(feature, evidence=evidence)


_MANIFEST_FORBIDDEN_BY_SOURCE: dict[str, list[str]] = {
    "csu": CSU_FORBIDDEN_AS_FEATURES,
    "optum": OPTUM_FORBIDDEN_AS_FEATURES,
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
        forbidden = _MANIFEST_FORBIDDEN_BY_SOURCE.get(manifest_source)
        if forbidden:
            excluded_set.update(forbidden)
        else:
            # Codex M1 (PR #92 review): a typo or future-cohort value
            # would silently fall through to legacy behaviour, defeating
            # the defense-in-depth objective with no operator signal.
            # Warn once per call so an operator who misspelt
            # ``feature_manifest_source`` in scope_spec can spot the issue
            # before the run completes. The reactive Layer 1 audit still
            # catches forbidden columns downstream — this warning is the
            # only signal the proactive layer was bypassed.
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

    # Use explicit `is not None` checks: `state.get(...) or DEFAULT` silently
    # replaces a legitimate 0 with the default (Python's falsy-zero semantics).
    # `adaptive_seed=0` is a valid seed; the old form returned 7 instead.
    _n_perms = state.get("adaptive_n_permutations")
    n_perms = int(_n_perms) if _n_perms is not None else DEFAULT_PERMUTATIONS
    _seed = state.get("adaptive_seed")
    seed = int(_seed) if _seed is not None else 7

    verdicts: list[dict[str, Any]] = []
    flagged: list[str] = []
    voter = _get_ensemble_voter_class()()

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
            verdict = _compose_legacy_verdict(
                feat,
                voter=voter,
                layer_1_input=_layer_1_input(feat, contract),
            )
            verdicts.append(verdict)
            flagged.append(feat)
            layer_1_caught.add(feat)

    # Layer 3 pass — numeric columns only, skipping anything Layer 1 already caught.
    for feat in numeric_candidates:
        if feat in layer_1_caught:
            continue

        col = train_df[feat]
        mask = col.notna() & binary_label_mask
        if mask.sum() < MIN_LAYER3_SAMPLES:
            verdicts.append(
                _compose_legacy_verdict(
                    feat,
                    voter=voter,
                    short_circuit_evidence=(
                        f"Skipped: only {int(mask.sum())} non-null rows "
                        f"(need ≥{MIN_LAYER3_SAMPLES})"
                    ),
                )
            )
            continue

        try:
            score = compute_adversarial_score(
                col[mask].to_numpy(dtype=float),
                train_df.loc[mask, target].to_numpy(dtype=int),
                n_permutations=n_perms,
                seed=seed,
                z_threshold=HIGH_Z,
            )
        except Exception as exc:
            logger.warning("adaptive_validity_check: scoring failed for %s: %s", feat, exc)
            verdicts.append(
                _compose_legacy_verdict(
                    feat,
                    voter=voter,
                    short_circuit_evidence=f"Adversarial scoring error: {exc}",
                )
            )
            continue

        verdict = _compose_legacy_verdict(
            feat,
            voter=voter,
            adversarial_input=_adversarial_input(score),
        )
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
    }
    if new_severity != prior_severity:
        update["leakage_severity"] = new_severity
        update["leakage_detected"] = True
    return update
