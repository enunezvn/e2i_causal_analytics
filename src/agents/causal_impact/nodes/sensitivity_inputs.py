"""Benchmark inputs for the sensitivity reading, shared by both causal-impact nodes.

Spec ``docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md``
§4.3: the refutation node (which feeds the runner) and the sensitivity node (which
feeds the agent narrative) compute the SAME three inputs — baseline risk, naive
contrast and the measured covariates' bias factors — from the SAME full frame, so
the two engines cannot disagree on a run.

Lives outside ``refutation.py`` so the sensitivity node can import it without
pulling in the refutation node. Pure computation on a frame: pandas/numpy through
``src.causal_engine.evalue``, no I/O and no node imports.
"""

from typing import Any, Mapping

import pandas as pd

from src.causal_engine import evalue
from src.causal_engine.errors import RefutationError


def is_numeric_column(frame: Any, column: str) -> bool:
    """Is ``column`` a numeric column? Asked only of the TREATMENT and the OUTCOME.

    Covariates are NOT screened this way: ``evalue`` scores a categorical covariate
    level by level, and dropping one would understate the measured-confounding
    benchmark (see ``sensitivity_benchmark_inputs``).
    """
    try:
        return bool(pd.api.types.is_numeric_dtype(frame[column]))
    except Exception:  # noqa: BLE001 - an unreadable column is a missing input
        return False


def sensitivity_benchmark_inputs(
    *,
    estimation_data: Any,
    treatment: str,
    outcome: str,
    estimation_result: Mapping[str, Any],
) -> evalue.BenchmarkInputs:
    """FULL-frame inputs for the sensitivity reading (spec 2026-09-10 §4.3).

    Baseline risk, naive contrast and the backdoor set's covariate bias factors are
    computed on ``estimation_data`` (the full frame), never on the refutation
    subsample, so the benchmark describes the frame the reported effect came from.
    ``naive_ate`` from the estimation node wins; ``baseline_covariates_adjusted``
    (efficiency controls, #1188) are excluded from the factors.

    CATEGORICAL covariates are passed through, not screened out. The live #1351
    resolver binds string driver columns (``trigger_type``, ``delivery_channel``, …)
    into the adjustment set and the estimation node fits their one-hot encoding
    (#1417), so they are measured confounding; ``evalue`` scores them level by level.
    Dropping them would understate the fallback benchmark and let a run read
    ``beyond_measured_confounding`` on confounding this run actually measured.

    MISSING inputs are a legitimate fallback: empty inputs, and the reading is
    ``unbenchmarked``. Missing means no frame, an absent treatment/outcome column, or
    a NON-NUMERIC treatment or outcome. The last is not a screening choice but a
    statement about the frame: the estimator cannot have fit a string T or Y, so a
    frame carrying one is not the frame the reported effect came from, and there is
    no contrast to benchmark.

    A computation that RAISES inside ``evalue`` (e.g. a covariate — or one level of a
    categorical covariate — that perfectly separates treatment and outcome, a
    positivity violation whose bias factor diverges) is a real failure and surfaces
    as ``RefutationError`` with reason ``sensitivity_benchmark_failed``, never as a
    fabricated reading (spec §5; the runner applies the same rule to its own
    fallback, and the sensitivity node turns it into ``sensitivity_error``).
    """
    empty = evalue.BenchmarkInputs(baseline_risk=None, naive_effect=None)
    if estimation_data is None or not hasattr(estimation_data, "columns"):
        return empty
    if treatment not in estimation_data.columns or outcome not in estimation_data.columns:
        return empty
    if not is_numeric_column(estimation_data, treatment) or not is_numeric_column(
        estimation_data, outcome
    ):
        return empty
    covariates = [
        str(c)
        for c in (estimation_result.get("covariates_adjusted") or [])
        if c in estimation_data.columns
    ]
    try:
        return evalue.benchmark_inputs_from_frame(
            estimation_data,
            treatment,
            outcome,
            covariates,
            naive_effect=estimation_result.get("naive_ate"),
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with a reason code, never swallowed
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without "
            "refutation. Sensitivity benchmark inputs could not be computed on the "
            f"estimation frame: {exc}",
            details={
                "reason": "sensitivity_benchmark_failed",
                "treatment": treatment,
                "outcome": outcome,
                "covariates": covariates,
            },
            original_error=exc,
        ) from exc
