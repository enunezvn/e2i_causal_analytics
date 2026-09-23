"""#2207 split contract: a table cohort contract + its DGP manifest clears Layer 3.

Measured 2026-09-23 on the live Kisqali goldstd contracts (full data_preparer graph walk):
``adaptive_validity_check`` (Layer-3 adversarial discriminator) flags the DGP's DESIGNED
prognostic drivers — ``disease_severity`` (z=40σ, single-feature AUC 0.71) and
``age_at_diagnosis`` (z=41σ) on initiation — as HIGH/drop and routes the retrain to LLM
remediation. With ``scope_spec.feature_manifest_source = "synthetic_csu"`` (the manifest
of the DGP that seeded ``patient_journeys``, declared-safe BY CONSTRUCTION) "Declared-safe
immunity" exempts the declared pre-index covariates and the QC gate passes. Migration 151
therefore seeds ``cohort_feature_manifest_source='synthetic_csu'`` on the 9 patient rows.

This test pins that interaction hermetically on a small synthetic frame shaped like the
contract: the CONTROL (no manifest) must flag the strong pre-index driver HIGH — proving the
plant landed — and the same state with the manifest must not. Mirrors
``test_layer_5_pipeline_integration.py`` (async, fresh loop per test).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    adaptive_validity_check,
)
from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import detect_leakage
from src.data.manifests.resolution import resolve_manifest_source

_COLUMNS = [
    "disease_severity",
    "academic_hcp",
    "geographic_region",
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "rep_detailing_high",
    "sample_dropped",
    "trigger_accepted",
    "treatment_initiated",
]
_CONTRACT: Dict[str, Any] = {
    "type": "table",
    "table": "patient_journeys",
    "filters": {"brand": "Kisqali", "is_synthetic": True},
    "columns": _COLUMNS,
}


def _frame(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """A pre-index driver (disease_severity) that predicts the label at AUC ~0.7, like
    the live DGP; every other covariate is noise."""
    rng = np.random.default_rng(seed)
    severity = rng.integers(1, 5, n)
    logit = -1.2 + 0.9 * (severity - 2.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "disease_severity": severity,
            "academic_hcp": rng.integers(0, 2, n),
            "geographic_region": rng.choice(["Northeast", "South", "West"], n),
            "insurance_type": rng.choice(["commercial", "medicare"], n),
            "age_at_diagnosis": rng.integers(30, 80, n),
            "comorbidity_burden": rng.integers(0, 4, n),
            "prior_therapy_lines": rng.integers(0, 3, n),
            "rep_detailing_high": rng.integers(0, 2, n),
            "sample_dropped": rng.integers(0, 2, n),
            "trigger_accepted": rng.integers(0, 2, n),
            "treatment_initiated": y,
            "data_split": ["train"] * n,
        }
    )


def _state(manifest: str | None) -> Dict[str, Any]:
    df = _frame()
    scope: Dict[str, Any] = {
        "prediction_target": "treatment_initiated",
        "data_source": _CONTRACT,
        "filters": {},
        "problem_type": "binary_classification",
        "required_features": [],
        "excluded_features": [],
    }
    if manifest:
        scope["feature_manifest_source"] = manifest
    return {
        "experiment_id": f"test-2207-manifest-{manifest or 'none'}",
        "data_source": _CONTRACT,
        "scope_spec": scope,
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "holdout_df": None,
        "blocking_issues": [],
    }


async def _run(state: Dict[str, Any]) -> Dict[str, Any]:
    state.update({k: v for k, v in (await detect_leakage(state)).items() if v is not None})
    state.update({k: v for k, v in (await adaptive_validity_check(state)).items() if v is not None})
    return state


def test_table_dict_resolves_the_synthetic_csu_override_without_m1_m2() -> None:
    assert resolve_manifest_source(_CONTRACT, "synthetic_csu") == "synthetic_csu"
    assert resolve_manifest_source(_CONTRACT, None) is None  # no auto-detect from a dict


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_without_manifest_flags_the_designed_driver_high() -> None:
    """The plant lands: Layer 3 sees the strong pre-index driver as a leak."""
    state = await _run(_state(None))
    verdicts = {v["feature"]: v for v in state.get("adaptive_verdicts") or []}
    assert verdicts["disease_severity"]["severity"] == "high", verdicts["disease_severity"]
    assert verdicts["disease_severity"]["remediation"] == "drop"
    assert "disease_severity" in (state.get("leaked_features") or [])
    assert state.get("leakage_severity") == "high"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_manifest_declared_safe_immunity_clears_the_designed_driver(
    caplog: pytest.LogCaptureFixture,
) -> None:
    node_logger = "src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check"
    with caplog.at_level(logging.WARNING, logger=node_logger):
        state = await _run(_state("synthetic_csu"))
    assert "disease_severity" not in (state.get("leaked_features") or []), state.get(
        "leaked_features"
    )
    assert not state.get("adaptive_flagged_features"), state.get("adaptive_flagged_features")
    assert state.get("leakage_severity") in ("none", "info"), state.get("leakage_severity")
    immunity = [
        r.getMessage() for r in caplog.records if "Declared-safe immunity" in r.getMessage()
    ]
    assert immunity and "disease_severity" in immunity[0], immunity
    # Undeclared commercial columns are still scored by Layer 3 and kept (info/keep).
    verdicts = {v["feature"]: v for v in state.get("adaptive_verdicts") or []}
    for col in ("rep_detailing_high", "sample_dropped", "trigger_accepted"):
        assert verdicts[col]["remediation"] == "keep", verdicts[col]
