"""#2335: a sweep retrain's ``required_features`` are the contract's covariates.

The drift sweep's retrain input (``_cohort_input_from_training_config``) passed no
``candidate_features``, so ``scope_builder._define_required_features`` fell back to its
generic placeholder list (hcp_specialty, patient_count, prescription_history,
brand_affinity_score, + engagement_score / channel_response_rate). Four of those columns
exist in no table in the schema, so every such retrain finished with ``is_ready=False``
and a "Missing required features" blocker although every real gate passed (live proof 3
of #2286/#2287). The same placeholder list also scoped ``detect_leakage``'s structural
checks, ``compute_baseline_metrics`` and the Feast registrar to columns the frame does
not have.

A table contract already declares the exact columns it loads (migrations 151 / 163:
``columns`` = the goldstd covariates + the label). Those covariates are what the parent
was trained on, so they are what the retrain requires. Since the follow-up PR the
pipeline's scope stage resolves them (``cohort_contract.resolve_required_features``)
and records ``required_features_source == "contract"``; the sweep's builder passes the
contract through and invents no candidates. These tests drive the real
consumers: the registry row as the migration writes it -> ``contract_from_registry_row``
-> the sweep's input builder -> the REAL pipeline scope stage and scope_definer graph ->
``scope_spec`` -> the data_preparer's ``finalize_output``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.graph import finalize_output
from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)
from src.services.cohort_contract import contract_from_registry_row
from src.tasks.drift_monitoring_tasks import _cohort_input_from_training_config

MIGRATIONS = Path(__file__).resolve().parents[3] / "database" / "migrations"

_UPDATE_RE = re.compile(
    r"UPDATE ml_model_registry\s+SET (?P<set>.*?)\s+WHERE model_name = '(?P<model>[^']+)'",
    re.S,
)


def _contract_rows() -> Dict[str, Dict[str, Any]]:
    """{model: registry row} for every goldstd row a migration gives a table contract.

    151 writes the 9 patient rows (label in its own SET); 163 writes the 3 HCP-adoption
    rows over the ``adopted`` label 151 set.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for name in ("151_*.sql", "163_*.sql"):
        (path,) = sorted(MIGRATIONS.glob(name))
        for m in _UPDATE_RE.finditer(path.read_text()):
            sets = dict(re.findall(r"(cohort_\w+) = '([^']*)'", m["set"]))
            if "cohort_data_source" not in sets:
                continue
            sets.setdefault("cohort_target_outcome", "adopted")
            rows[m["model"]] = sets
    return rows


ROWS = _contract_rows()


def _covariates(model: str) -> List[str]:
    row = ROWS[model]
    contract = contract_from_registry_row(row)
    return [c for c in contract["data_source"]["columns"] if c != row["cohort_target_outcome"]]


def test_the_migrations_give_twelve_table_contracts() -> None:
    # 9 patient (151) + 3 HCP adoption (163); guards the parser against a silent zero.
    assert len(ROWS) == 12, sorted(ROWS)
    assert sum(m.startswith("hcp_adoption_") for m in ROWS) == 3


@pytest.fixture(autouse=True)
def _no_external_writes():
    """``ScopeDefinerAgent.run`` persists unconditionally (#2300) — stub its writers.

    Same four writers + Opik connector as ``test_target_variable_hint_2284.py``; the
    pipeline's audit writes are no-ops because ``_result()`` has no audit_workflow_id.
    """
    hooks = MagicMock()
    hooks.store_experiment_pattern = AsyncMock(return_value=True)
    hooks.store_scope_definition = AsyncMock(return_value=True)
    base = "src.agents.ml_foundation.scope_definer.agent"
    with (
        patch(f"{base}._get_experiment_repository", new=AsyncMock(return_value=None)),
        patch(f"{base}.ScopeDefinerMemoryHooks", return_value=hooks),
        patch(f"{base}._get_procedural_memory", return_value=None),
        patch(f"{base}._get_opik_connector", return_value=None),
    ):
        yield


def _result() -> PipelineResult:
    return PipelineResult(
        pipeline_run_id="run-2335",
        status="running",
        current_stage=PipelineStage.SCOPE_DEFINITION,
    )


async def _scope_spec_for(model: str) -> Dict[str, Any]:
    input_data = _cohort_input_from_training_config(contract_from_registry_row(ROWS[model]))
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    result = _result()
    await pipeline._run_scope_definition(input_data, result, None)
    assert result.scope_spec, result.errors
    return dict(result.scope_spec)


def _finalize_state(scope_spec: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    frame = pd.DataFrame({c: [0, 1] for c in columns})
    return {
        "experiment_id": "exp-2335",
        "scope_spec": scope_spec,
        "qc_status": "passed",
        "overall_score": 1.0,
        "blocking_issues": [],
        "train_df": frame,
        "validation_df": frame,
        "test_df": frame,
        "holdout_df": frame,
    }


# --------------------------------------------------------------------------
# the sweep's input builder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(ROWS))
def test_the_retrain_input_hands_the_declaring_contract_to_the_pipeline(model: str) -> None:
    """The builder invents nothing: the contract's columns reach the pipeline's scope
    stage, which resolves them (``resolve_required_features``, the one place) and records
    the provenance — see the end-to-end tests below."""
    input_data = _cohort_input_from_training_config(contract_from_registry_row(ROWS[model]))

    target = ROWS[model]["cohort_target_outcome"]
    assert "candidate_features" not in input_data
    assert set(input_data["data_source"]["columns"]) == set(_covariates(model)) | {target}
    assert input_data["target_variable_hint"] == target


def test_an_explicit_candidate_list_wins_over_the_contract_columns() -> None:
    model = "hcp_adoption_kisqali_goldstd_lr_v1"
    cfg = {**contract_from_registry_row(ROWS[model]), "candidate_features": ["specialty"]}

    assert _cohort_input_from_training_config(cfg)["candidate_features"] == ["specialty"]


@pytest.mark.parametrize(
    "data_source",
    [
        "data/rwd/optum/initiation",  # file route: no declared column list
        {"type": "table", "table": "patient_journeys", "filters": {"brand": "Kisqali"}},
        {"type": "table", "table": "patient_journeys", "columns": ["adopted"]},
    ],
)
def test_no_declared_covariates_leaves_candidate_features_unset(data_source: Any) -> None:
    cfg = {"data_source": data_source, "target_outcome": "adopted"}

    assert "candidate_features" not in _cohort_input_from_training_config(cfg)


# --------------------------------------------------------------------------
# end to end: registry row -> real pipeline scope stage -> finalize_output
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("model", sorted(ROWS))
async def test_the_real_scope_stage_requires_exactly_the_contract_covariates(model: str) -> None:
    scope_spec = await _scope_spec_for(model)

    assert scope_spec["required_features"] == _covariates(model)
    assert scope_spec["required_features_source"] == "contract"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_hcp_adoption_retrain_is_ready_on_its_own_frame() -> None:
    """The exact #2335 symptom: the 163 frame (covariates + label + data_split)."""
    model = "hcp_adoption_remibrutinib_goldstd_lr_v1"
    scope_spec = await _scope_spec_for(model)
    frame_columns = contract_from_registry_row(ROWS[model])["data_source"]["columns"] + [
        "data_split"
    ]

    out = await finalize_output(_finalize_state(scope_spec, frame_columns))

    assert out["missing_required_features"] == []
    assert out["is_ready"] is True
    assert not any("Missing required features" in b for b in out["blockers"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_contracted_covariate_absent_from_the_frame_still_blocks_readiness() -> None:
    """is_ready stays meaningful: a genuinely missing contracted column is caught."""
    model = "hcp_adoption_remibrutinib_goldstd_lr_v1"
    scope_spec = await _scope_spec_for(model)
    frame_columns = [
        c
        for c in contract_from_registry_row(ROWS[model])["data_source"]["columns"]
        if c != "peer_influence_score"
    ]

    out = await finalize_output(_finalize_state(scope_spec, frame_columns))

    assert out["missing_required_features"] == ["peer_influence_score"]
    assert out["is_ready"] is False
