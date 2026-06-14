"""Unit tests for CohortDeployer — gold-standard-feature training + staging registration.

Two behaviors that distinguish the gold-standard eval model from the existing
60-feature serving champion (prediction_synthesizer_deploy):

1. ``train_cohort_model`` fits on the NAMED-column gold-standard DataFrame from
   FeatureBuilder, so ``model.feature_names_in_`` == the FeatureBuilder
   ``feature_columns`` (the 3-covariate-derived encoded set) — NOT the 60
   synthetic-generator features. The test builds a tiny REAL
   ``FeatureBuilder.build_from_frame`` to get (X, y) and asserts the estimator
   carries exactly those columns.

2. ``register_cohort_model`` writes the registry row at ``stage='staging'`` (a
   valid ``model_stage_enum`` value), NEVER ``stage='production'``. This is the
   mandatory collision guard: the initiation target already has a ``production``
   60-feature serving champion, and ``get_models_for_target`` filters
   ``_SERVING_STAGES=('production',)`` — so registering the gold-standard model
   at ``staging`` keeps it out of the serving ensemble while
   ``_resolve_model_id`` (matches by model_name/version regardless of stage)
   still lets the Time-Series trend endpoint resolve it.

No real DB: a fake async client captures every ``.insert``/``.delete``/``.select``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.mlops.gold_standard_eval.cohort_deployer import (
    register_cohort_model,
    serialize_model,
    train_cohort_model,
)
from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

# ---------------------------------------------------------------------------
# Fake async Supabase client — records the chained PostgREST calls.
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class _Query:
    """Captures a chain of builder calls and resolves to a programmed result."""

    def __init__(self, client: "FakeClient", table: str, op: str) -> None:
        self._client = client
        self._table = table
        self._op = op  # "select" | "insert" | "delete"
        self._filters: dict[str, Any] = {}
        self._payload: Any = None

    # builder methods all return self for chaining
    def select(self, *_args: Any, **_kw: Any) -> "_Query":
        self._op = "select"
        return self

    def insert(self, row: Any) -> "_Query":
        self._op = "insert"
        self._payload = row
        return self

    def delete(self) -> "_Query":
        self._op = "delete"
        return self

    def eq(self, col: str, val: Any) -> "_Query":
        self._filters[col] = val
        return self

    async def execute(self) -> _Result:
        return self._client._dispatch(self._table, self._op, self._filters, self._payload)


class FakeClient:
    """Minimal async Supabase double.

    Programmed so ``_get_or_create_experiment`` finds NO existing experiment
    (forcing an insert that returns a fixed id) and the registry insert's
    read-back returns the row that was inserted (so the staging read-back check
    passes). Every write is recorded for assertions.
    """

    EXPERIMENT_ID = "exp-goldstd-uuid"

    def __init__(self) -> None:
        self.inserts: list[tuple[str, dict[str, Any]]] = []
        self.deletes: list[tuple[str, dict[str, Any]]] = []
        self.selects: list[tuple[str, dict[str, Any]]] = []
        # last registry row inserted, surfaced by the read-back select
        self._last_registry_row: dict[str, Any] | None = None

    def table(self, name: str) -> _Query:
        return _Query(self, name, op="select")

    def _dispatch(self, table: str, op: str, filters: dict[str, Any], payload: Any) -> _Result:
        if op == "select":
            self.selects.append((table, dict(filters)))
            if table == "ml_experiments":
                # No pre-existing experiment → caller creates one.
                return _Result([])
            if table == "ml_model_registry":
                # Read-back: echo the row just inserted (matching the filters).
                if self._last_registry_row is not None:
                    return _Result([self._last_registry_row])
                return _Result([])
            return _Result([])

        if op == "insert":
            self.inserts.append((table, dict(payload)))
            if table == "ml_experiments":
                return _Result([{"id": self.EXPERIMENT_ID}])
            if table == "ml_model_registry":
                self._last_registry_row = dict(payload)
                return _Result([dict(payload)])
            return _Result([dict(payload)])

        if op == "delete":
            self.deletes.append((table, dict(filters)))
            return _Result([])

        raise AssertionError(f"unexpected op {op!r} on {table!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_gold_standard_xy() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build a small REAL (X, y) via FeatureBuilder so feature_names are genuine."""
    fb = FeatureBuilder(INITIATION)
    raw = pd.DataFrame(
        {
            "patient_id": [f"scvpt_{i}" for i in range(8)],
            "treatment_initiated": [1, 0, 1, 0, 1, 0, 1, 0],
            "disease_severity": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.85, 0.15],
            "academic_hcp": [1, 0, 1, 0, 1, 0, 0, 1],
            "geographic_region": [
                "west",
                "south",
                "west",
                "east",
                "south",
                "west",
                "east",
                "south",
            ],
        }
    )
    X, y = fb.build_from_frame(raw)
    return X, y, list(fb.feature_columns)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_cohort_model_feature_names_match_feature_builder_columns():
    """The estimator must carry EXACTLY the FeatureBuilder feature_columns.

    This is the core design constraint: train on the gold-standard named-column
    frame so ``feature_names_in_`` == the 3-covariate-derived encoded set (NOT
    the 60 synthetic-generator features the serving champion uses).
    """
    X, y, feature_columns = _tiny_gold_standard_xy()

    model = train_cohort_model(INITIATION, X, y)

    assert hasattr(model, "predict_proba"), "must be a probabilistic classifier"
    assert hasattr(model, "feature_names_in_"), "must carry named features for by-name mapping"
    assert list(model.feature_names_in_) == feature_columns, (
        "feature_names_in_ must equal the FeatureBuilder columns, "
        f"got {list(model.feature_names_in_)} vs {feature_columns}"
    )
    # real fit → predicts probabilities in range
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_serialize_model_writes_loadable_pickle(tmp_path):
    """serialize_model pickles the estimator and returns an absolute path."""
    import pickle

    X, y, _ = _tiny_gold_standard_xy()
    model = train_cohort_model(INITIATION, X, y)

    path = serialize_model(model, tmp_path / "artifacts", "csu_initiation_goldstd_lr_v1")

    from pathlib import Path

    assert Path(path).is_absolute(), f"artifact path must be absolute, got {path}"
    assert Path(path).is_file()
    with open(path, "rb") as fh:
        loaded = pickle.load(fh)
    assert list(loaded.feature_names_in_) == list(model.feature_names_in_)


@pytest.mark.asyncio
async def test_register_cohort_model_writes_staging_not_production(tmp_path):
    """register_cohort_model MUST write stage='staging', never 'production'.

    Collision guard: the initiation target already has a production 60-feature
    serving champion. A production-staged gold-standard row would make
    get_models_for_target return BOTH → the incompatible-feature ensemble breaks.
    """
    X, y, feature_columns = _tiny_gold_standard_xy()
    model = train_cohort_model(INITIATION, X, y)
    artifact_path = serialize_model(model, tmp_path / "artifacts", "csu_initiation_goldstd_lr_v1")

    client = FakeClient()
    returned = await register_cohort_model(
        client,
        INITIATION,
        model_name="csu_initiation_goldstd_lr_v1",
        model_version="1.0",
        artifact_path=artifact_path,
        auc=0.671,
        feature_count=len(feature_columns),
    )

    assert returned == "csu_initiation_goldstd_lr_v1"

    # exactly one registry row written
    registry_inserts = [r for (t, r) in client.inserts if t == "ml_model_registry"]
    assert len(registry_inserts) == 1, f"expected 1 registry insert, got {registry_inserts}"
    row = registry_inserts[0]

    # MANDATORY collision guard: staging, never production
    assert row["stage"] == "staging", f"must register at staging, got {row['stage']!r}"
    assert row["stage"] != "production"

    # honest provenance + loadable artifact
    assert row["is_synthetic"] is False
    assert row["artifact_path"] == artifact_path
    assert row["model_name"] == "csu_initiation_goldstd_lr_v1"
    assert row["model_version"] == "1.0"
    assert row["auc"] == 0.671
    assert row["feature_count"] == len(feature_columns)

    # NO registry row may carry stage='production' (defense across all writes)
    assert all(
        r.get("stage") != "production" for (t, r) in client.inserts if t == "ml_model_registry"
    )

    # distinct experiment was created for the gold-standard model (not the
    # serving experiment) and the row is bound to it.
    experiment_inserts = [r for (t, r) in client.inserts if t == "ml_experiments"]
    assert len(experiment_inserts) == 1
    assert experiment_inserts[0]["prediction_target"] == INITIATION.target
    assert experiment_inserts[0]["experiment_name"] != "csu_treatment_initiation_live_v1", (
        "must use a DISTINCT experiment from the serving deploy"
    )
    assert row["experiment_id"] == FakeClient.EXPERIMENT_ID


@pytest.mark.asyncio
async def test_register_cohort_model_idempotent_replace_and_readback(tmp_path):
    """Mirror the deploy module's safety: delete-by-(name,version) then read-back.

    The read-back must confirm the row landed at the INTENDED stage (staging).
    """
    X, y, feature_columns = _tiny_gold_standard_xy()
    model = train_cohort_model(INITIATION, X, y)
    artifact_path = serialize_model(model, tmp_path / "artifacts", "csu_initiation_goldstd_lr_v1")

    client = FakeClient()
    await register_cohort_model(
        client,
        INITIATION,
        model_name="csu_initiation_goldstd_lr_v1",
        model_version="1.0",
        artifact_path=artifact_path,
        auc=0.671,
        feature_count=len(feature_columns),
    )

    # idempotent replace: a delete scoped to (model_name, model_version) fired
    registry_deletes = [f for (t, f) in client.deletes if t == "ml_model_registry"]
    assert len(registry_deletes) == 1
    assert registry_deletes[0].get("model_name") == "csu_initiation_goldstd_lr_v1"
    assert registry_deletes[0].get("model_version") == "1.0"

    # read-back select on the registry happened (verification, not blind insert)
    registry_selects = [f for (t, f) in client.selects if t == "ml_model_registry"]
    assert len(registry_selects) >= 1


@pytest.mark.asyncio
async def test_register_cohort_model_refuses_missing_artifact():
    """An artifact that does not exist on disk must NOT be registered."""
    client = FakeClient()
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        await register_cohort_model(
            client,
            INITIATION,
            model_name="csu_initiation_goldstd_lr_v1",
            model_version="1.0",
            artifact_path="/nonexistent/path/to/model.pkl",
            auc=0.671,
            feature_count=4,
        )
