"""#2207 follow-up (owner decision 2026-09-22): the data-prep Feast QC gate.

Before: ``feast_registrar`` probed ``feature_analyzer_<experiment_id>`` — a view name
present in no Feast registry and in no source map — so ``last_updated`` was always
``None``, every run was "unverifiable", and the gate hard-blocked training on the
worker image AND on any box with feast installed. It was unpassable by construction.

Now: the gate measures the freshness of the Feast views SOURCED FROM the run's data
source (inverse of ``FEAST_FEATURE_VIEW_SOURCE_TABLES``) through the feast-free #559
recency probe, records the result, and hard-blocks ONLY when the run actually trains on
Feast-served features (``features_served_by_feast=True`` — nothing in the pipeline sets
it today: ``data_loader`` reads a Supabase table, files or the synthetic sample, and
``model_trainer.split_loader`` takes its Feast branch only when data_preparer handed it
no splits, which the pipeline always does). ``ALLOW_STALE_FEAST=1`` keeps its meaning
for that block branch only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes import feast_registrar
from src.agents.ml_foundation.data_preparer.nodes.feast_registrar import (
    register_features_in_feast,
)
from src.agents.ml_foundation.data_preparer.state import DataPreparerState

# Relative to the real clock: the registrar's probe ages recency against
# datetime.now() with a 24 h ceiling, so a pinned date turns "fresh" fixtures stale
# a day after it (the pinned 2026-09-22 value broke every CI run from 2026-09-23 10:00Z).
NOW = datetime.now(timezone.utc)
ADAPTER_PATH = (
    "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter"
)


def _state(data_source, **extra):
    train_df = pd.DataFrame(
        {"hcp_id": ["h1", "h2", "h3"], "feature1": np.arange(3.0), "target": [0, 1, 0]}
    )
    state = {
        "experiment_id": "exp_gate_2207",
        "train_df": train_df,
        "data_source": data_source,
        "scope_spec": {
            "required_features": ["feature1"],
            "entity_key": "hcp_id",
            "prediction_target": "target",
        },
    }
    state.update(extra)
    return state


def _adapter():
    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={"features_registered": 1, "errors": []}
    )
    # The adapter's own freshness probe must NOT be the gate any more (it needed a
    # Feast client init that fails on the worker image).
    adapter.check_feature_freshness = AsyncMock(side_effect=AssertionError("must not be called"))
    adapter._feast_client = None
    return adapter


def _recency(value):
    async def _q(table):
        return value

    return _q


@pytest.fixture(autouse=True)
def _no_escape_hatch(monkeypatch):
    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)


def _run(state, recency):
    with (
        patch(ADAPTER_PATH, return_value=_adapter()),
        patch.object(feast_registrar, "_source_recency_query", recency),
    ):
        import asyncio

        return asyncio.run(register_features_in_feast(state))


# ---------------------------------------------------------------------------
# table sources
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_table_source_with_fresh_mapped_views_registers_and_reports_fresh():
    result = _run(_state("business_metrics"), _recency(NOW - timedelta(hours=2)))
    assert result["feast_registration_status"] == "completed"
    assert result["feast_blocked"] is False
    fc = result["feast_freshness_check"]
    assert fc["fresh"] is True
    assert fc["feast_backed"] is True
    assert fc["source_table"] == "business_metrics"
    assert set(fc["feature_views"]) == {
        "hcp_conversion_features",
        "hcp_engagement_features",
        "market_dynamics_features",
    }
    assert not any(w.startswith("Freshness") for w in result["feast_warnings"])
    assert "blocking_issues" not in result


@pytest.mark.unit
def test_table_source_with_stale_views_is_advisory_not_blocked():
    result = _run(_state("triggers"), _recency(NOW - timedelta(days=40)))
    assert result["feast_blocked"] is False
    assert result["feast_registration_status"] == "advisory_stale_features"
    assert result["feast_freshness_check"]["fresh"] is False
    assert any(w.startswith("Freshness") for w in result["feast_warnings"])
    assert "blocking_issues" not in result


@pytest.mark.unit
def test_unmapped_table_is_advisory_not_feast_backed():
    calls = []

    async def _q(table):
        calls.append(table)
        return NOW

    result = _run(_state("ml_some_cohort_table"), _q)
    assert calls == []  # nothing to probe: no Feast view is sourced from it
    assert result["feast_blocked"] is False
    assert result["feast_registration_status"] == "completed"
    fc = result["feast_freshness_check"]
    assert fc["fresh"] is None and fc["feast_backed"] is False
    assert any("not Feast-backed" in w for w in result["feast_warnings"])


# ---------------------------------------------------------------------------
# file sources: both loader shapes (data_loader accepts file_dir AND files)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "data_source",
    [{"type": "file_dir", "path": "data/rwd/optum"}, {"type": "files", "paths": {"train": "a"}}],
)
def test_file_sources_are_advisory(data_source):
    result = _run(_state(data_source), _recency(None))
    assert result["feast_blocked"] is False
    assert result["feast_freshness_check"]["feast_backed"] is False
    assert result["feast_freshness_check"]["source_kind"] == data_source["type"]
    assert any("not Feast-backed" in w for w in result["feast_warnings"])
    assert "blocking_issues" not in result


# ---------------------------------------------------------------------------
# the retained guarantee: a Feast-SERVED run still hard-blocks on stale/unverifiable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_feast_served_run_with_stale_views_is_blocked():
    result = _run(
        _state("triggers", features_served_by_feast=True), _recency(NOW - timedelta(days=40))
    )
    assert result["feast_blocked"] is True
    assert result["feast_registration_status"] == "blocked_stale_features"
    assert any("Feast features stale" in b for b in result["blocking_issues"])


@pytest.mark.unit
def test_feast_served_run_with_unverifiable_views_is_blocked():
    """#556: unverifiable is not fresh — for a Feast-served run that still blocks."""
    result = _run(_state("hcp_profiles", features_served_by_feast=True), _recency(None))
    assert result["feast_blocked"] is True
    assert any("Feast features stale" in b for b in result["blocking_issues"])


@pytest.mark.unit
def test_feast_served_run_allow_stale_escape_hatch(monkeypatch):
    monkeypatch.setenv("ALLOW_STALE_FEAST", "1")
    result = _run(
        _state("triggers", features_served_by_feast=True), _recency(NOW - timedelta(days=40))
    )
    assert result["feast_blocked"] is False
    assert result["feast_registration_status"] != "blocked_stale_features"
    assert "blocking_issues" not in result


@pytest.mark.unit
def test_feast_served_run_with_fresh_views_passes():
    result = _run(
        _state("business_metrics", features_served_by_feast=True),
        _recency(NOW - timedelta(hours=1)),
    )
    assert result["feast_blocked"] is False
    assert result["feast_registration_status"] == "completed"


# ---------------------------------------------------------------------------
# state contract: the flag must be a declared channel field or LangGraph drops it
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_features_served_by_feast_is_a_declared_state_field():
    s = DataPreparerState(
        audit_workflow_id=uuid4(), experiment_id="e", features_served_by_feast=True
    )
    assert s.features_served_by_feast is True
    assert DataPreparerState(
        audit_workflow_id=uuid4(), experiment_id="e"
    ).features_served_by_feast in (
        None,
        False,
    )


@pytest.mark.unit
def test_advisory_status_is_a_declared_literal():
    s = DataPreparerState(
        audit_workflow_id=uuid4(),
        experiment_id="e",
        feast_registration_status="advisory_stale_features",
    )
    assert s.feast_registration_status == "advisory_stale_features"


# ---------------------------------------------------------------------------
# probe failure keeps the #556 exception semantics (stale unless ALLOW_STALE_FEAST=1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_probe_exception_is_unverifiable_and_only_blocks_feast_served_runs(monkeypatch):
    async def _boom(table):
        raise RuntimeError("supabase down")

    advisory = _run(_state("triggers"), _boom)
    assert advisory["feast_blocked"] is False
    assert advisory["feast_freshness_check"]["fresh"] is False
    assert "error" in advisory["feast_freshness_check"]

    served = _run(_state("triggers", features_served_by_feast=True), _boom)
    assert served["feast_blocked"] is True
