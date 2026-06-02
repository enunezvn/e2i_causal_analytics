"""Single source of truth for the QC ``overall_score`` blocking threshold.

The data-quality gate blocks the whole pipeline (per cohort) when the weighted
``overall_score`` falls below a minimum bar. Historically that bar was a
HARDCODED ``0.80`` literal duplicated across THREE enforcement points
(``quality_checker.run_quality_checks`` blocking-issue append, the
``graph.finalize_output`` QC-gate decision, and — transitively, via the
already-decided ``qc_passed`` boolean — the model_trainer ``check_qc_gate``).
Duplicating the literal meant the bar could silently DRIFT between points, and
an unconfigurable literal could not be relaxed for cohorts where some QC
dimensions are inherently lower without editing source.

``resolve_qc_min_overall_score`` removes both defects:

1. **Single source of truth** — every enforcement point resolves the bar
   through this one function, so the threshold can never drift.
2. **Dynamic / overridable** — the default stays ``0.80`` (so existing
   pass/fail outcomes are PRESERVED), but a caller, per-cohort scope_spec, or
   ops/CI environment may explicitly opt into a different bar.

Resolution precedence (first match wins; default last):

1. Explicit per-run override on the agent state: ``qc_min_overall_score``
   (caller / PipelineConfig — highest precedence, mirrors the
   ``adaptive_fdr_enabled`` per-run-switch convention).
2. Per-cohort override on the scope_spec: ``scope_spec["qc_min_overall_score"]``
   (set by scope_definer / the cohort config for that cohort).
3. Ops / CI environment override: ``QC_MIN_OVERALL_SCORE`` (a numeric floor
   that mirrors the existing ``QC``-style env overrides used elsewhere in
   data_preparer, e.g. ``ALLOW_STALE_FEAST``).
4. Default ``0.80`` — UNCHANGED baseline.

Every candidate is parsed defensively and clamped to ``[0.0, 1.0]``. A
candidate that is missing, ``None``, non-numeric, or out of range is IGNORED
(resolution falls through to the next source, and ultimately to the strict
``0.80`` default) so a malformed override can never silently *lower* the gate
to an unsafe value or crash the gate.

Why NOT regime-keyed (the fuller, #641-spirit version) — deferred, with
reasoning rather than an invented formula:

    The ``adaptive_success_criteria`` engine keys its bars on
    ``regime`` ∈ {default, clean, adverse} read from the scope_definer state
    (``state.get("regime")``, sourced by ``scripts/run_tier0_test.py`` for the
    SYNTHETIC fixtures). That ``regime`` signal is a scope_definer concept; it
    is NOT a field of ``ScopeSpecSchema`` and does NOT propagate into the
    ``scope_spec`` that the data_preparer QC gate actually sees. Keying the QC
    bar on N / prevalence instead would be a formula invented without basis,
    and — worse — would risk silently MOVING the bar for small-N or
    rare-responder REAL cohorts, violating the hard "default unchanged at
    0.80" guarantee. When/if a principled regime signal is threaded into
    ``scope_spec`` (a separate, intent-backed change), this resolver is the
    single place to add a regime branch: callers already opt in explicitly
    today, so adding regime-awareness later is additive, not a behavior change.
"""

import logging
import os
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# UNCHANGED baseline blocking bar. Behavior is identical to the historical
# duplicated ``0.80`` literals when no override is supplied.
DEFAULT_QC_MIN_OVERALL_SCORE: float = 0.80

# State key (per-run / caller / PipelineConfig override) and scope_spec key
# (per-cohort override). Same name in both namespaces for a single mental model.
_STATE_KEY = "qc_min_overall_score"
_SCOPE_SPEC_KEY = "qc_min_overall_score"

# Ops / CI environment override.
_ENV_KEY = "QC_MIN_OVERALL_SCORE"


def _coerce_threshold(raw: Any, *, source: str) -> Optional[float]:
    """Parse and validate a candidate threshold; ``None`` if unusable.

    A usable threshold is a real number in the closed range ``[0.0, 1.0]``.
    Anything else (``None``, non-numeric string, bool, NaN/inf, out of range)
    is rejected with a warning so resolution falls through to the next source.
    """
    if raw is None:
        return None
    # ``bool`` is an ``int`` subclass; a stray ``True``/``False`` would coerce
    # to 1.0/0.0 and silently swing the gate. Reject it explicitly.
    if isinstance(raw, bool):
        logger.warning("Ignoring boolean QC min-overall-score override from %s: %r", source, raw)
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring non-numeric QC min-overall-score override from %s: %r", source, raw
        )
        return None
    # Reject NaN / inf (float("nan") != float("nan")).
    if value != value or value in (float("inf"), float("-inf")):
        logger.warning("Ignoring non-finite QC min-overall-score override from %s: %r", source, raw)
        return None
    if not (0.0 <= value <= 1.0):
        logger.warning(
            "Ignoring out-of-range QC min-overall-score override from %s "
            "(must be in [0.0, 1.0]): %r",
            source,
            raw,
        )
        return None
    return value


def resolve_qc_min_overall_score(
    state: Optional[Mapping[str, Any]] = None,
) -> float:
    """Resolve the effective QC ``overall_score`` blocking threshold.

    Args:
        state: The agent state (or any mapping) for the current run. May carry
            an explicit ``qc_min_overall_score`` and/or a ``scope_spec`` with
            its own ``qc_min_overall_score``. ``None`` / missing keys are fine.

    Returns:
        The effective minimum ``overall_score`` below which the QC gate blocks.
        Defaults to ``0.80`` when no valid override is supplied, preserving
        historical behavior exactly.
    """
    state = state or {}

    # 1. Explicit per-run / caller override on the state.
    candidate = _coerce_threshold(state.get(_STATE_KEY), source="state")
    if candidate is not None:
        return candidate

    # 2. Per-cohort override on the scope_spec. ``scope_spec`` may be a dict, a
    # BaseAgentSchema (dict-like ``.get`` shim), or ``None``.
    scope_spec = state.get("scope_spec") or {}
    scope_get = getattr(scope_spec, "get", None)
    if callable(scope_get):
        candidate = _coerce_threshold(scope_get(_SCOPE_SPEC_KEY), source="scope_spec")
        if candidate is not None:
            return candidate

    # 3. Ops / CI environment override.
    candidate = _coerce_threshold(os.environ.get(_ENV_KEY), source="env")
    if candidate is not None:
        return candidate

    # 4. Default — UNCHANGED baseline.
    return DEFAULT_QC_MIN_OVERALL_SCORE
