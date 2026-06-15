"""Central provenance enforcement (SSOT) for the synthetic causal-validation dataset.

Every PostgREST reader calls ``apply_provenance_filter`` so a forgotten table can
never leak ``is_synthetic=true`` rows into real-mode results. Every estimator drops
``PROVENANCE_DROP_COLS`` before building its covariate matrix so the tag is never an
effect modifier. See plan 07-provenance-readpath-enforcement.md.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import pandas as pd

PROVENANCE_COLUMN = "is_synthetic"
# Columns that must NEVER enter an estimator design matrix (an all-cols-except-
# treatment/outcome derivation would otherwise capture them as constant covariates).
PROVENANCE_DROP_COLS: tuple[str, ...] = (PROVENANCE_COLUMN,)

# Every table carrying the ``is_synthetic`` column, derived from the three
# migrations that added it (#894 SSOT — when a migration tags a new table, add
# it HERE; table-aware readers like MLDataLoader and the sentinel evaluators
# gate on this set so untagged tables never hit a 42703):
#   database/migrations/063_is_synthetic_provenance.sql        (M1, 12 tables)
#   database/migrations/067_kpi_view_synthetic_exclusion.sql   (3 view-backed)
#   database/migrations/069_synthetic_provenance_shard09_tables.sql (Shard 09, 11)
PROVENANCE_TAGGED_TABLES: frozenset[str] = frozenset(
    {
        # 063
        "triggers",
        "business_metrics",
        "ml_predictions",
        "agent_activities",
        "causal_paths",
        "patient_journeys",
        "treatment_events",
        "hcp_profiles",
        "user_sessions",
        "hcp_intent_surveys",
        "episodic_memories",
        "ab_experiment_assignments",
        # 067
        "data_source_tracking",
        "etl_pipeline_metrics",
        "ml_annotations",
        # 069
        "ml_experiments",
        "ml_model_registry",
        "ml_training_runs",
        "ml_deployments",
        "ab_experiment_enrollments",
        "ab_experiment_results",
        "ml_observability_spans",
        "learning_signals",
        "feature_groups",
        "features",
        "feature_values",
    }
)


def coerce_provenance_flag(value: Any) -> bool:
    """Strictly parse a provenance opt-in value (``include_synthetic``/``synthetic``).

    Only ``True`` / ``"true"`` / ``"1"`` / ``"yes"`` opt in. Everything else —
    ``False``, ``"false"``, ``"0"``, ``"no"``, ``None``, invalid types — stays
    real-mode: this flag controls provenance isolation, so an ambiguous value
    must FAIL CLOSED to the default-exclude predicate (``bool("false")`` is
    ``True`` and would silently flip an explicit opt-OUT into reading — or
    training on — synthetic rows).

    Shared SSOT (issue #883 §4, lifting the codex #874-R2 helper out of the
    orchestrator dispatcher): agents and celery tasks must parse payload
    provenance flags through THIS function rather than ``bool()`` so the strict
    contract cannot drift per boundary. It lives here, next to
    :func:`apply_provenance_filter`, because both enforce the same isolation:
    one parses the opt-in, the other applies it to reads.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


#: Truthy spellings for the ``E2I_*`` runtime flags (mirrors
#: ``src/kpi/synthetic_mode.py`` and the copilotkit ``E2I_ENABLE_*`` idiom).
_TRUTHY = ("1", "true", "yes")


def deployment_includes_synthetic() -> bool:
    """Whether THIS deployment treats synthetic-tagged rows as first-class data.

    A showcase / review instance whose only substrate is synthetic-gold (the
    2026-06-11 cleanup) sets ``E2I_INCLUDE_SYNTHETIC`` so the synthetic flag is a
    *warning/badge*, never a *gate*: every read-path chokepoint INCLUDES synthetic
    rows so the platform runs at full potential. Read fresh on every call (truthy:
    ``1`` / ``true`` / ``yes``, case-insensitive) so it can be toggled per
    deployment without a restart-coupled import-time capture.

    Defaults to ``False`` → the strict real-mode default-exclude gate is preserved
    verbatim for a true-production instance carrying real RWD alongside synthetic.
    Generalizes the KPI-only ``E2I_KPI_INCLUDE_SYNTHETIC`` flag
    (:mod:`src.kpi.synthetic_mode`) to every provenance-gated reader, reversibly.
    """
    return os.getenv("E2I_INCLUDE_SYNTHETIC", "0").strip().lower() in _TRUTHY


def apply_provenance_filter(query: Any, include_synthetic: bool = False) -> Any:
    """Append the default-exclude provenance predicate to a supabase-py query.

    Real mode (``include_synthetic=False``) appends ``.eq('is_synthetic', False)``.
    Validation mode (``True``) returns the query unchanged (caller opts in
    explicitly). The predicate is a no-op cost on tables whose every row defaults to
    ``false`` and is index-friendly (Shard 01 adds the partial index).

    On a synthetic-gold showcase instance (``E2I_INCLUDE_SYNTHETIC`` set, see
    :func:`deployment_includes_synthetic`) the predicate is skipped for EVERY
    reader so synthetic rows power the platform — reversibly: unset the env and
    the strict gate returns verbatim.
    """
    if include_synthetic or deployment_includes_synthetic():
        return query
    return query.eq(PROVENANCE_COLUMN, False)


def apply_provenance_filter_for_table(
    query: Any, table: str, include_synthetic: bool = False
) -> Any:
    """Table-aware variant of :func:`apply_provenance_filter` (#894).

    Applies the default-exclude predicate only when ``table`` is in
    :data:`PROVENANCE_TAGGED_TABLES` — readers that take an operator- or
    config-supplied table name (sentinel evaluators, MLDataLoader) would
    otherwise raise a REAL 42703 on untagged tables (e.g.
    ``executive_insights``).
    """
    if table in PROVENANCE_TAGGED_TABLES:
        return apply_provenance_filter(query, include_synthetic)
    return query


def drop_provenance_cols(frame: pd.DataFrame, extra: Iterable[str] = ()) -> pd.DataFrame:
    """Return ``frame`` without provenance/bookkeeping columns (covariate safety)."""
    to_drop = [c for c in (*PROVENANCE_DROP_COLS, *extra) if c in frame.columns]
    return frame.drop(columns=to_drop) if to_drop else frame
