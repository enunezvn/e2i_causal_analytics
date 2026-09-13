"""Shared constants and small numeric/date helpers for the causal routes package.

Import rule: imports nothing from the package — this is the bottom layer.

Admission criterion: leaf helpers and constants read by several route modules;
nothing with an intra-package dependency. Revisit a split past ~400 lines.
"""

import math
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.api.errors import user_safe_503_detail
from src.causal.stats import z_score_for_confidence

# #931: the episodic event_type the causal_impact agent emits when an analysis
# completes. Used by both the health-check activity fields and the history
# endpoint so the KPI count and the History tab share one source of truth.
CAUSAL_COMPLETED_EVENT_TYPE = "causal_analysis_completed"

# Agent-run wall-clock budgets (orphan-fix). The async agent task wraps the
# whole graph in ``asyncio.wait_for(..., _AGENT_HARD_TIMEOUT_S)`` — a HARD cap.
# But the heavy refutation suite runs in a worker thread that wait_for CANNOT
# cancel (Python can't force-kill a thread), so hitting the hard cap would
# orphan a still-grinding refutation thread that keeps burning a CPU core and
# accumulates across runs. To prevent that we pass the graph a COOPERATIVE
# deadline (``_REFUTATION_COMPUTE_BUDGET_S`` from task start); the refutation
# node skips refuters that would run past it and fails-closed cleanly, so the
# thread returns and releases the heavy-compute slot BEFORE the hard cap fires.
# The gap between them is headroom for one in-flight refuter's overshoot plus
# the post-refutation sensitivity/interpretation nodes.
_AGENT_HARD_TIMEOUT_S = 900.0

_REFUTATION_COMPUTE_BUDGET_S = 720.0

# Generic 5xx detail. Raw exception text MUST NOT be echoed to clients: it can
# leak stack-internal paths, library/module names, table/column names, and other
# information useful to an attacker. The full exception is logged server-side
# (with exc_info) instead; the client receives only this opaque message.
_GENERIC_500_DETAIL = "Internal server error"

_ROBUSTNESS_UNVALIDATED_WARNING = (
    "robustness_validation_performed=false: this ATE was estimated but NOT "
    "refutation-tested (the sequential/parallel pipeline does not run "
    "refutation/sensitivity checks). Treat the effect as UNVALIDATED for "
    "robustness; do not present it as a validated causal claim."
)

# R6-F1 (#740): caveats for the opt-in refutation path. The refutation runs only
# on the DoWhy estimate (Owner-decision 1: a labeled proxy for the consensus),
# and on REVIEW/BLOCK the pipeline DOWNGRADES (still 200, flag=False) rather than
# 503-blocking the whole multi-library answer (Owner-decision 2).
_ROBUSTNESS_REVIEW_WARNING = (
    "robustness_validation_performed=false (gate=REVIEW): the DoWhy refutation "
    "suite returned a REVIEW band (borderline-robust) for this estimate — it is "
    "usable only with expert review and MUST NOT be presented as validated. "
    "Robustness was validated on the DoWhy estimate only; EconML/CausalML "
    "estimates in the consensus are unrefuted."
)

_ROBUSTNESS_BLOCK_WARNING = (
    "robustness_validation_performed=false (gate=BLOCK): the DoWhy refutation "
    "suite BLOCKED this estimate (a critical refutation test failed or confidence "
    "was below threshold). Treat the effect as NOT robust. Robustness was "
    "validated on the DoWhy estimate only; EconML/CausalML estimates in the "
    "consensus are unrefuted."
)

# M-fo2 (precise): a directed cycle only breaks identification when it lands on the
# (treatment, outcome) ancestral subgraph (``undefined_cyclic``). That caveat is
# un-ignorable — appended to BOTH the warnings list and robustness_warning — and it
# FORCES robustness False, sets requires_review=True, and WITHHOLDS the consensus
# effect (backdoor adjustment is mathematically undefined on such a graph).
_NON_DAG_STRUCTURAL_WARNING = (
    "Discovered causal graph contains a directed cycle ON the treatment-outcome "
    "ancestral subgraph; backdoor adjustment is undefined for this estimand. The "
    "consensus effect is WITHHELD and the result is quarantined for review "
    "(requires_review=true) — do NOT treat any per-library number as a causal claim."
)

# A cycle OFF the ancestral subgraph leaves the estimand identifiable: informational
# only, no penalty, consensus preserved.
_CYCLE_IRRELEVANT_WARNING = (
    "Discovered causal graph contains a cycle OUTSIDE the treatment-outcome "
    "ancestral subgraph; this estimand remains identifiable and no structural "
    "penalty was applied."
)

# causal_impact agent runs (POST /causal/agent-analyze submit -> GET poll). The
# agent's energy-score selection + refutation is too slow for a synchronous
# request (~minutes), so it runs as a background task and the FE polls.
# Cross-worker job store (Redis-backed; in-memory fallback). The API runs
# multiple gunicorn workers, so a module-level dict would 404 on poll when the
# GET lands on a different worker than the POST. See DurableJobStore.
# Job records live long enough to outlast a working session, not just the run.
# The ranked-effects leaderboard stays on-screen in the browser indefinitely (it
# is React state, not reloaded), but each drilled-in effect re-fetches its full
# detail (DAG + refutation) from this store. At the old 1h TTL the leaderboard
# would still be visible while its drill-downs had silently expired — clicking a
# row then 404'd with "this analysis may have expired". An 8h window (a workday)
# keeps the leaderboard and its drill-downs alive together. The discover-effects
# job store below uses the SAME TTL so the two never diverge.
_CAUSAL_JOB_TTL_SECONDS = 8 * 3600  # 8h — a full working session


def _opt_float(value: Any) -> Optional[float]:
    """Coerce a refuter field to a float, or None if absent/non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# These are curated, exception-free explanations meant for end users, so they opt
# in to the global 503 handler surfacing them verbatim (the FE Heterogeneous
# Treatment Effects card matches "no real data backend" to render an honest
# "data isn't wired yet" state). Keep the wording in sync with that FE gate.
_NO_REAL_DATA_BACKEND_DETAIL = user_safe_503_detail(
    "Causal pipeline endpoints have no real data backend wired. "
    "There is no production data source returning treatment/outcome columns by name. "
    "Pass demo_mode=true to get a clearly-labeled pinned-zero placeholder for UI demos, "
    "or wire real data and re-issue the request."
)

_NO_RESOLVABLE_DATA_DETAIL = user_safe_503_detail(
    "Sequential/parallel pipeline executed but no library produced a result: "
    "no DataFrame was resolvable from the request filters and there is no "
    "production data backend wired for arbitrary data_source identifiers. "
    "Supply inline data via filters.estimation_data_records (list of dicts with "
    "treatment / outcome / covariate columns), or pass demo_mode=true for the "
    "clearly-labeled pinned-zero placeholder used in UI demos."
)

# Libraries that REQUIRE a DataFrame to produce a real causal estimate.
# NetworkX is excluded because it is a symbolic-input graph executor (see
# C-5 design spike) — it can succeed with only variable names and an
# upstream `state['causal_graph']`. If a request includes any of these
# data-required libraries AND none of them succeed, we fail-close even
# when NetworkX succeeded, because the pipeline did not answer the
# causal-effect question the user asked.
_DATA_REQUIRED_LIBRARIES: frozenset[str] = frozenset({"dowhy", "econml", "causalml"})


def _dowhy_interval(dowhy_payload: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """95 % normal interval ``effect +/- z*SE`` from a DoWhy executor payload (#2014).

    ``None`` when the payload has no finite effect or no positive finite
    ``standard_error`` (every DoWhy method but linear regression) — never a number
    without a measured SE.
    """
    effect = _as_optional_float(dowhy_payload.get("causal_effect"))
    se = _as_optional_float(dowhy_payload.get("standard_error"))
    if effect is None or se is None or not math.isfinite(effect) or not math.isfinite(se):
        return None
    if se <= 0.0:
        return None
    z = z_score_for_confidence(0.95)
    return effect - z * se, effect + z * se


def _parse_occurred_at(value: Any) -> Optional[datetime]:
    """Coerce an episodic ``occurred_at`` (ISO string or datetime) to datetime.

    Returns ``None`` for missing/unparseable values rather than fabricating a
    timestamp.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    """Coerce a raw_content numeric field to float, or ``None`` if absent/invalid.

    Returns ``None`` (honest unknown) rather than a fabricated default so a
    missing ATE/confidence never renders as a plausible-looking number. Accepts
    native numbers and numeric strings (a JSONB round-trip or a non-canonical
    writer can encode a float as ``"0.185"``); a non-numeric value is ``None``.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _te_pvalue_from_z(ate: float, std_error: Optional[float]) -> Optional[float]:
    """Two-sided model-based z-test p-value, mirroring the agent estimation path.

    ``p = 2*(1 - Phi(|ate|/std_error))``. Returns None when std_error is missing
    or not a usable positive finite value (we never emit p=NaN). This is a
    model-based p-value, NOT a refutation p-value.
    """
    if std_error is None:
        return None
    try:
        se = float(std_error)
    except (TypeError, ValueError):
        return None
    import math as _math

    if not _math.isfinite(se) or se <= 0.0:
        return None
    from scipy import stats as _scipy_stats

    z = abs(float(ate)) / se
    return float(2.0 * (1.0 - _scipy_stats.norm.cdf(z)))


def _resolve_pipeline_dataframe(
    filters: Optional[Dict[str, Any]],
) -> Optional["pd.DataFrame"]:  # type: ignore[name-defined] # noqa: F821
    """Rehydrate an estimation DataFrame from request filters.

    Surface C accepts a DataFrame only via inline JSON-serialized records in
    ``filters.estimation_data_records``. This preserves the existing schema
    (``filters: Optional[Dict[str, Any]]``) without forcing a separate file
    upload surface. Returns ``None`` when no DataFrame can be rehydrated —
    the caller fail-closes with 503.

    Per CLAUDE.md anti-mocking discipline: this helper does NOT manufacture
    synthetic data when no DataFrame is provided. The 503 fail-close path
    is the honest response when the data backend is absent.
    """
    import pandas as pd

    if not isinstance(filters, dict):
        return None
    records = filters.get("estimation_data_records")
    if not isinstance(records, list) or not records:
        return None
    try:
        df = pd.DataFrame.from_records(records)
    except Exception:  # noqa: BLE001 - any rehydration failure → fail-close
        return None
    if df.empty:
        return None
    return df
