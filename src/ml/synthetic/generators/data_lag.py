"""Post-hoc column stampers for Shard 09 KPI substrate.

stamp_data_lag_hours (WS1-DQ-007): recent, bounded data_lag_hours on the synthetic
patient_journeys frame so v_kpi_data_lag computes non-NULL over now()-30d.

stamp_sequence_number (WS3-BI-006 NRx): per-(patient,brand) chronological index of
prescription events so the NRx KPI (sequence_number=1) counts new prescriptions.

Both columns ALREADY exist on the faithful DB (integer, nullable) -- no DDL needed;
the loader carries them via TABLE_COLUMNS (Task 1).

stamp_claim_arrival (backlog #45): the synthetic claims ARRIVAL plane on
treatment_events -- claim_available_date (= event_date + drawn adjudication lag)
and adjudication_lag_days (migration 115). Parameters are vocabulary-driven
(data_constraints.adjudication_lag_dgp). ADDITIVE-ONLY by construction: a
post-generation stamp with its OWN rng stream cannot perturb any generator
stream, and NO base KPI reads the new columns -- they feed only the
completion-factor nowcast overlay (PR-B).
"""

from typing import Any, Optional

import numpy as np
import pandas as pd


def stamp_data_lag_hours(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Return a copy of df with a recent, bounded integer data_lag_hours column.

    Right-skewed toward fresh: gamma(shape=2, scale=18) (mean ~36h), clipped to
    [1, 168] (1h..7d plausible ingest lag) so the mean stays well under the 72h
    health threshold while the tail still reaches a week.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    lag = np.clip(rng.gamma(shape=2.0, scale=18.0, size=len(out)).round().astype(int), 1, 168)
    out["data_lag_hours"] = lag
    return out


def stamp_sequence_number(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a per-(patient_id, brand) chronological
    sequence_number on prescription events (1 = the first/new prescription).

    NRx (WS3-BI-006) counts treatment_events WHERE event_type='prescription' AND
    sequence_number=1. The synthetic treatment generator does not emit this column,
    so every synthetic prescription would be missed without this stamp. Only
    prescription rows are numbered; non-prescription rows keep a NULL sequence.
    """
    out = df.copy()
    if "event_type" not in out.columns or out.empty:
        out["sequence_number"] = pd.NA
        return out
    out["sequence_number"] = pd.NA
    is_rx = out["event_type"] == "prescription"
    rx = out.loc[is_rx].copy()
    if not rx.empty:
        group_cols = [c for c in ("patient_id", "brand") if c in rx.columns]
        # Stable chronological rank within each (patient, brand): 1, 2, 3, ...
        rx = rx.sort_values([*group_cols, "event_date"], kind="mergesort")
        rx["__seq"] = rx.groupby(group_cols, dropna=False).cumcount() + 1
        out.loc[rx.index, "sequence_number"] = rx["__seq"].astype(int)
    return out


def stamp_claim_arrival(
    df: pd.DataFrame, seed: int, vocab: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Return a copy of ``df`` (treatment_events) with the claims ARRIVAL plane
    stamped: ``claim_available_date`` (= event_date + drawn adjudication lag)
    and ``adjudication_lag_days`` (backlog #45, migration 115).

    Parameters come from ``data_constraints.adjudication_lag_dgp`` in the
    domain vocabulary (per source-class gamma shape/scale + clip_days, keyed by
    event_type) -- no scalars in code; ``vocab`` overrides the registry-loaded
    ``data_constraints`` mapping (tests). Semantics:

    * mapped claims event types: lag = clip(round(gamma(shape, scale)), clip),
      claim_available_date = event_date + lag days;
    * event types outside every source class (CRM/triggers plane) when
      ``crm_zero_lag`` is set: adjudication_lag_days=0, claim_available_date
      NULL (not on the claims plane);
    * rows with missing/unparseable event_date: both columns NULL (fail-empty).

    Own ``np.random.default_rng(seed)`` stream (load path uses seed+10) -- the
    stamp cannot perturb any generator stream; deterministic given (df, seed);
    never mutates its input. Draw order is config order then frame order, both
    deterministic per run.
    """
    if vocab is None:
        from src.ontology.vocabulary_registry import VocabularyRegistry

        vocab = VocabularyRegistry.load().get_data_constraints()
    dgp = (vocab or {}).get("adjudication_lag_dgp")
    if not dgp:
        raise ValueError(
            "data_constraints.adjudication_lag_dgp is not authored -- the arrival "
            "plane cannot be stamped (fail loud; do not load NULL-only columns)"
        )
    if dgp.get("distribution") != "gamma":
        raise ValueError(f"unsupported adjudication lag distribution: {dgp.get('distribution')!r}")

    rng = np.random.default_rng(seed)
    out = df.copy()
    out["claim_available_date"] = pd.NA
    out["adjudication_lag_days"] = pd.NA
    if out.empty or "event_type" not in out.columns or "event_date" not in out.columns:
        return out

    event_dt = pd.to_datetime(out["event_date"], errors="coerce")
    valid = event_dt.notna()

    mapped_types: set[str] = set()
    for cls_name, cls in dgp.get("source_classes", {}).items():
        applies_to = list(cls["applies_to_event_types"])
        mapped_types.update(applies_to)
        mask = valid & out["event_type"].isin(applies_to)
        n = int(mask.sum())
        if n == 0:
            continue
        clip_lo, clip_hi = cls["clip_days"]
        lag = np.clip(
            np.round(rng.gamma(shape=cls["shape"], scale=cls["scale"], size=n)).astype(int),
            clip_lo,
            clip_hi,
        )
        out.loc[mask, "adjudication_lag_days"] = lag
        out.loc[mask, "claim_available_date"] = (
            event_dt[mask] + pd.to_timedelta(lag, unit="D")
        ).dt.strftime("%Y-%m-%d")

    if dgp.get("crm_zero_lag", False):
        # CRM/triggers-plane event types have lifecycle timestamps already: no
        # adjudication lag (0) and NOT on the claims plane (NULL arrival date).
        out.loc[valid & ~out["event_type"].isin(mapped_types), "adjudication_lag_days"] = 0
    return out
