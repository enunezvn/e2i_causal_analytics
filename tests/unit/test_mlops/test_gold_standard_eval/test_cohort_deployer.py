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
        self._op = op  # "select" | "insert" | "delete" | "upsert"
        self._filters: dict[str, Any] = {}
        self._payload: Any = None
        self._on_conflict: Any = None

    # builder methods all return self for chaining
    def select(self, *_args: Any, **_kw: Any) -> "_Query":
        self._op = "select"
        return self

    def insert(self, row: Any) -> "_Query":
        self._op = "insert"
        self._payload = row
        return self

    def upsert(self, row: Any, on_conflict: Any = None) -> "_Query":
        self._op = "upsert"
        self._payload = row
        self._on_conflict = on_conflict
        return self

    def delete(self) -> "_Query":
        self._op = "delete"
        return self

    def eq(self, col: str, val: Any) -> "_Query":
        self._filters[col] = val
        return self

    async def execute(self) -> _Result:
        return self._client._dispatch(
            self._table, self._op, self._filters, self._payload, self._on_conflict
        )


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
        self.upserts: list[tuple[str, dict[str, Any], Any]] = []
        self.deletes: list[tuple[str, dict[str, Any]]] = []
        self.selects: list[tuple[str, dict[str, Any]]] = []
        # last registry row written, surfaced by the read-back select
        self._last_registry_row: dict[str, Any] | None = None

    def table(self, name: str) -> _Query:
        return _Query(self, name, op="select")

    def _dispatch(
        self,
        table: str,
        op: str,
        filters: dict[str, Any],
        payload: Any,
        on_conflict: Any = None,
    ) -> _Result:
        if op == "select":
            self.selects.append((table, dict(filters)))
            if table == "ml_experiments":
                # No pre-existing experiment → caller creates one.
                return _Result([])
            if table == "ml_model_registry":
                # Read-back: echo the row just written (matching the filters).
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

        if op == "upsert":
            self.upserts.append((table, dict(payload), on_conflict))
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

    # exactly one registry row written (upsert in place, not delete+insert)
    registry_writes = [r for (t, r, _oc) in client.upserts if t == "ml_model_registry"]
    assert len(registry_writes) == 1, f"expected 1 registry upsert, got {registry_writes}"
    row = registry_writes[0]

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
        r.get("stage") != "production" for (t, r, _oc) in client.upserts if t == "ml_model_registry"
    )

    # distinct experiment was created for the gold-standard model (not the
    # serving experiment) and the row is bound to it.
    experiment_inserts = [r for (t, r) in client.inserts if t == "ml_experiments"]
    assert len(experiment_inserts) == 1
    assert experiment_inserts[0]["prediction_target"] == INITIATION.target
    assert experiment_inserts[0]["experiment_name"] != "csu_treatment_initiation_live_v1", (
        "must use a DISTINCT experiment from the serving deploy"
    )
    # Lifecycle (migration 102): lineage rows are written 'completed' — the
    # DB-default 'running' is reserved for actively-enrolling A/B experiments.
    assert experiment_inserts[0]["status"] == "completed"
    assert row["experiment_id"] == FakeClient.EXPERIMENT_ID


@pytest.mark.asyncio
async def test_register_cohort_model_upserts_in_place_no_delete_fk_safe(tmp_path):
    """Re-registration MUST upsert in place (preserve the registry id), NOT
    delete+insert.

    ``ml_performance_metrics`` / ``ml_drift_history`` / ``ml_monitoring_alerts``
    carry a RESTRICT FK to ``ml_model_registry(id)``. A delete+insert churns the
    id, so once any dependent row exists the registry DELETE 23503-fails (and any
    pre-delete dependent cleanup that already ran is left as collateral damage).
    Upserting on the ``(model_name, model_version)`` unique key UPDATEs the row in
    place — the id (and every dependent FK reference) survives. The row must omit
    ``id`` so the upsert-merge preserves the existing PK rather than minting a new one.
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

    # NO delete on the registry — the id-churning delete+insert is the FK landmine.
    registry_deletes = [f for (t, f) in client.deletes if t == "ml_model_registry"]
    assert registry_deletes == [], (
        "register_model_row must NOT delete the registry row (id churn trips the "
        "ml_drift_history RESTRICT FK on re-run); it must upsert in place."
    )

    # Exactly one upsert on the (model_name, model_version) unique key.
    registry_upserts = [(r, oc) for (t, r, oc) in client.upserts if t == "ml_model_registry"]
    assert len(registry_upserts) == 1, f"expected 1 registry upsert, got {registry_upserts}"
    row, on_conflict = registry_upserts[0]
    assert on_conflict == "model_name,model_version", (
        f"upsert must target the (model_name, model_version) unique key, got {on_conflict!r}"
    )
    # id MUST be omitted so the existing PK is preserved on UPDATE.
    assert "id" not in row, "registry row must omit 'id' so upsert-merge preserves the PK"
    assert row["stage"] == "staging"
    assert row["model_name"] == "csu_initiation_goldstd_lr_v1"
    assert row["model_version"] == "1.0"

    # read-back select on the registry happened (verification, not a blind write).
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


@pytest.mark.asyncio
async def test_register_cohort_model_uses_spec_target_for_experiment(tmp_path):
    """A non-initiation spec must resolve/create its experiment under its own target.

    PERSISTENCE has target='pnh_persistence' — the experiment row inserted must
    carry that target, NOT the initiation target ('csu_treatment_initiation').
    The registry row must still land at stage='staging' and is_synthetic=False.
    """
    from src.mlops.gold_standard_eval.cohort_spec import PERSISTENCE

    X, y, feature_columns = _tiny_gold_standard_xy()
    model = train_cohort_model(INITIATION, X, y)  # estimator shape doesn't matter here
    artifact_path = serialize_model(model, tmp_path / "artifacts", "pnh_persistence_goldstd_lr_v1")

    client = FakeClient()
    returned = await register_cohort_model(
        client,
        PERSISTENCE,
        model_name="pnh_persistence_goldstd_lr_v1",
        experiment_name="persistence_goldstd_eval_v1",
        artifact_path=artifact_path,
        auc=0.77,
        feature_count=9,
        training_samples=8336,
    )

    assert returned == "pnh_persistence_goldstd_lr_v1"

    # The experiment row must be created under the PERSISTENCE target.
    experiment_inserts = [r for (t, r) in client.inserts if t == "ml_experiments"]
    assert len(experiment_inserts) == 1, f"expected 1 experiment insert, got {experiment_inserts}"
    exp_row = experiment_inserts[0]
    assert exp_row["prediction_target"] == "pnh_persistence", (
        f"experiment must be created under 'pnh_persistence', got {exp_row['prediction_target']!r}"
    )
    assert exp_row["experiment_name"] == "persistence_goldstd_eval_v1"

    # Registry row must still be at staging, is_synthetic=False.
    registry_writes = [r for (t, r, _oc) in client.upserts if t == "ml_model_registry"]
    assert len(registry_writes) == 1
    row = registry_writes[0]
    assert row["stage"] == "staging"
    assert row["is_synthetic"] is False
    assert row["model_name"] == "pnh_persistence_goldstd_lr_v1"
    assert row["auc"] == 0.77
    assert row["feature_count"] == 9
    assert row["training_samples"] == 8336
    assert row["experiment_id"] == FakeClient.EXPERIMENT_ID


@pytest.mark.asyncio
async def test_register_cohort_model_stamps_training_provenance_synthetic_gold(tmp_path):
    """#968: gold-standard rows must be self-describing as synthetic-trained.

    The models ARE real fitted estimators (``is_synthetic`` stays ``False`` so they
    remain servable/explainable — serving/explain/predictions/health all filter on
    ``.eq("is_synthetic", False)``), but they are trained ONLY on the synthetic-gold
    cohort. ``register_cohort_model`` must therefore stamp
    ``training_provenance='synthetic_gold'`` on BOTH the registry row and the
    experiment row, so a real-mode catalog consumer can distinguish them and the
    promotion gate can refuse a synthetic-gold -> production transition.
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

    registry_writes = [r for (t, r, _oc) in client.upserts if t == "ml_model_registry"]
    assert len(registry_writes) == 1
    reg_row = registry_writes[0]
    # semantics preserved: still a "real fitted model" for the serving/explain filters
    assert reg_row["is_synthetic"] is False
    # NEW (#968): self-describing training-data provenance
    assert reg_row["training_provenance"] == "synthetic_gold"

    # the experiment row is stamped too (issue names BOTH registry + experiment rows)
    experiment_inserts = [r for (t, r) in client.inserts if t == "ml_experiments"]
    assert len(experiment_inserts) == 1
    assert experiment_inserts[0]["training_provenance"] == "synthetic_gold"
