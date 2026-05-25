"""Instrument Analyzer Node for Gap Analyzer Agent (#357, P-2 producer).

This node is the LIVE producer for the instrument-availability bonus: it computes
per-feature instrument strength from a REAL IV first-stage F-test (Staiger-Stock),
using ``src.causal_engine.iv.TwoStageLSEstimator``.

For each ``feature_name -> InstrumentSpec`` in ``state["instrument_specs"]`` whose
referenced columns all exist in ``state["tier0_data"]`` and which has at least
``MIN_FIRST_STAGE_N`` complete-case rows, it slices ``Y`` (outcome), ``D`` (treatment),
``Z`` (instruments) and optional ``X`` (covariates), runs the 2SLS estimator, and
records ``result.diagnostics.to_dict()`` into ``instrument_strength_by_feature``.

Anti-mocking (AC2): the strength is the REAL ``IVDiagnostics`` output of a real
first-stage F-test; there are NO hardcoded/placeholder strength values. Features
without a spec, with missing columns, below the n-floor, or whose estimation fails
are simply ABSENT from the map -> the bonus does not fire (honest "no signal",
fail-closed exactly like the V4.4 causal path).

If ``instrument_specs`` or ``tier0_data`` is absent, the node is a no-op and returns
an empty map (AC4b).
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.causal_engine.iv import TwoStageLSEstimator

from ..state import GapAnalyzerState, InstrumentSpec

logger = logging.getLogger(__name__)

# Minimum complete-case sample size to run a first stage. Below this we treat the
# feature as having no usable instrument signal (skip, do NOT fabricate a value).
MIN_FIRST_STAGE_N = 30


class InstrumentAnalyzerNode:
    """Produce per-feature IV first-stage instrument strength for gap_analyzer (#357)."""

    def __init__(self) -> None:
        """Initialize the instrument analyzer node."""
        self._estimator = TwoStageLSEstimator()

    async def execute(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Compute instrument strength per feature from real IV first stages.

        Args:
            state: Gap analyzer state. Reads ``instrument_specs`` and ``tier0_data``.

        Returns:
            ``{"instrument_strength_by_feature": {...}}`` mapping feature_name to the
            ``IVDiagnostics.to_dict()`` output. Preserves any precomputed map already on
            state (the P-1 passthrough path) and merges freshly-computed entries on top.
        """
        start_time = time.time()
        instrument_specs = state.get("instrument_specs")
        tier0_data = state.get("tier0_data")

        # Preserve any precomputed strengths already on state (e.g. supplied via
        # GapAnalyzerAgent input / a future orchestrator P-1 passthrough). The IV step
        # must NOT clobber them when it has nothing of its own to compute; it merges
        # freshly-computed real diagnostics on top (recomputation wins for that feature).
        existing = state.get("instrument_strength_by_feature") or {}
        by_feature: Dict[str, Dict[str, Any]] = dict(existing)

        # Fail-closed no-op: no specs or no data => nothing to compute; keep what we have.
        if not instrument_specs or tier0_data is None:
            return {"instrument_strength_by_feature": by_feature}

        # tier0_data is an opaque passthrough; require a DataFrame-like API.
        if not hasattr(tier0_data, "columns"):
            logger.warning(
                "instrument_analyzer: tier0_data has no 'columns' attribute "
                "(type=%s); skipping IV first stage.",
                type(tier0_data).__name__,
            )
            return {"instrument_strength_by_feature": by_feature}

        available_columns = set(tier0_data.columns)
        warnings: List[str] = []

        for feature_name, spec in instrument_specs.items():
            try:
                diag = self._estimate_feature(feature_name, spec, tier0_data, available_columns)
            except Exception as exc:  # estimation failure => no signal, not a stub
                logger.warning(
                    "instrument_analyzer: IV first stage failed for feature '%s': %s",
                    feature_name,
                    exc,
                )
                warnings.append(
                    f"Instrument first stage failed for '{feature_name}'; "
                    f"no instrument bonus applied."
                )
                continue

            if diag is not None:
                by_feature[feature_name] = diag

        result: Dict[str, Any] = {"instrument_strength_by_feature": by_feature}
        if warnings:
            result["warnings"] = warnings
        result["instrument_latency_ms"] = int((time.time() - start_time) * 1000)
        return result

    def _estimate_feature(
        self,
        feature_name: str,
        spec: InstrumentSpec,
        tier0_data: Any,
        available_columns: set,
    ) -> Optional[Dict[str, Any]]:
        """Run a real first stage for one feature; return diagnostics dict or None.

        Returns None (feature skipped) when columns are missing, instruments are
        empty, or the complete-case sample is below the n-floor.
        """
        treatment_col = spec.get("treatment_col")
        instrument_cols = spec.get("instrument_cols") or []
        # Outcome defaults to the feature's own metric column when not specified.
        outcome_col = spec.get("outcome_col") or feature_name
        covariate_cols = spec.get("covariate_cols") or []

        if not treatment_col or not instrument_cols:
            logger.debug(
                "instrument_analyzer: feature '%s' missing treatment/instrument cols; skip.",
                feature_name,
            )
            return None

        required_cols = [outcome_col, treatment_col, *instrument_cols, *covariate_cols]
        missing = [c for c in required_cols if c not in available_columns]
        if missing:
            logger.debug(
                "instrument_analyzer: feature '%s' missing columns %s in tier0_data; skip.",
                feature_name,
                missing,
            )
            return None

        # Slice complete cases only (drop any row with NA across the used columns).
        subset = tier0_data[required_cols].dropna()
        n = len(subset)
        if n < MIN_FIRST_STAGE_N:
            logger.debug(
                "instrument_analyzer: feature '%s' has n=%d < floor %d; skip.",
                feature_name,
                n,
                MIN_FIRST_STAGE_N,
            )
            return None

        Y = np.asarray(subset[outcome_col], dtype=np.float64)
        D = np.asarray(subset[treatment_col], dtype=np.float64)
        Z = np.asarray(subset[list(instrument_cols)], dtype=np.float64)
        X = np.asarray(subset[list(covariate_cols)], dtype=np.float64) if covariate_cols else None

        result = self._estimator.fit(outcome=Y, treatment=D, instruments=Z, covariates=X)
        if not result.success:
            logger.warning(
                "instrument_analyzer: 2SLS reported failure for feature '%s': %s",
                feature_name,
                result.error_message,
            )
            return None

        # REAL diagnostics: instrument_strength, is_weak_instrument, first_stage_f_stat.
        return result.diagnostics.to_dict()
