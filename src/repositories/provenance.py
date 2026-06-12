"""Central provenance enforcement (SSOT) for the synthetic causal-validation dataset.

Every PostgREST reader calls ``apply_provenance_filter`` so a forgotten table can
never leak ``is_synthetic=true`` rows into real-mode results. Every estimator drops
``PROVENANCE_DROP_COLS`` before building its covariate matrix so the tag is never an
effect modifier. See plan 07-provenance-readpath-enforcement.md.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

PROVENANCE_COLUMN = "is_synthetic"
# Columns that must NEVER enter an estimator design matrix (an all-cols-except-
# treatment/outcome derivation would otherwise capture them as constant covariates).
PROVENANCE_DROP_COLS: tuple[str, ...] = (PROVENANCE_COLUMN,)


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


def apply_provenance_filter(query: Any, include_synthetic: bool = False) -> Any:
    """Append the default-exclude provenance predicate to a supabase-py query.

    Real mode (``include_synthetic=False``) appends ``.eq('is_synthetic', False)``.
    Validation mode (``True``) returns the query unchanged (caller opts in
    explicitly). The predicate is a no-op cost on tables whose every row defaults to
    ``false`` and is index-friendly (Shard 01 adds the partial index).
    """
    if include_synthetic:
        return query
    return query.eq(PROVENANCE_COLUMN, False)


def drop_provenance_cols(frame: pd.DataFrame, extra: Iterable[str] = ()) -> pd.DataFrame:
    """Return ``frame`` without provenance/bookkeeping columns (covariate safety)."""
    to_drop = [c for c in (*PROVENANCE_DROP_COLS, *extra) if c in frame.columns]
    return frame.drop(columns=to_drop) if to_drop else frame
