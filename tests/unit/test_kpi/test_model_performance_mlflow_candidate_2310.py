"""#2310 R3: the KPI MLflow fallback never reads a retrain candidate's run.

A retrain registers a new MLflow version of the model it retrains, tags it
``e2i.role=candidate`` and leaves its ``current_stage`` at ``None`` (Lane A, #2310). The old
fallback asked ``get_latest_versions(stages=[Production, Staging, None])`` and read
``versions[0]``: with the reference version in Staging and a newer candidate at None, what it
read depended on the order MLflow returned the per-stage latest versions, and a model with no
Production/Staging version read the candidate outright.

Real MLflow: a file-local sqlite tracking + registry store under ``tmp_path`` (never the
configured server), driven through the real ``MlflowClient``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from src.kpi.calculators.model_performance import ModelPerformanceCalculator  # noqa: E402

NAME = "lane2310_kpi_model"


@pytest.fixture
def client(tmp_path: Path):
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    c = MlflowClient(tracking_uri=uri, registry_uri=uri)
    c.create_registered_model(NAME)
    return c, c.create_experiment("lane2310", artifact_location=str(tmp_path / "art"))


def _version(client_exp, auc: float, *, stage: str | None, candidate: bool = False) -> str:
    c, exp = client_exp
    run = c.create_run(exp)
    c.log_metric(run.info.run_id, "roc_auc", auc)
    c.set_terminated(run.info.run_id)
    mv = c.create_model_version(
        NAME, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id
    )
    if stage:
        c.transition_model_version_stage(NAME, mv.version, stage)
    if candidate:
        c.set_model_version_tag(NAME, mv.version, "e2i.role", "candidate")
        c.set_model_version_tag(NAME, mv.version, "e2i.retrain_of", "4ec55d13")
    return str(mv.version)


def _read(client_exp) -> tuple[float | None, str | None]:
    calc = ModelPerformanceCalculator(db_client=None, mlflow_client=client_exp[0])
    return calc._get_metric_from_mlflow(NAME, "roc_auc")


def test_candidate_only_model_reads_as_not_found(client):
    """A name whose only version is a candidate has no canonical version to read."""
    _version(client, 0.61, stage=None, candidate=True)
    assert _read(client) == (None, f"model_not_found:{NAME}")


def test_reference_version_wins_over_a_newer_candidate(client):
    _version(client, 0.72, stage="Staging")
    _version(client, 0.61, stage=None, candidate=True)
    _version(client, 0.60, stage=None, candidate=True)
    assert _read(client) == (0.72, None)


def test_a_tagged_candidate_is_skipped_whatever_its_stage(client):
    """If an operator moves a candidate's MLflow stage without clearing the tag, it is still
    not canonical here: the DB stage / the tag carry the role, not the MLflow stage."""
    _version(client, 0.72, stage="Staging")
    _version(client, 0.99, stage="Production", candidate=True)
    assert _read(client) == (0.72, None)


def test_stage_precedence_then_newest_version(client):
    _version(client, 0.50, stage=None)
    _version(client, 0.70, stage="Staging")
    _version(client, 0.71, stage="Staging")
    assert _read(client) == (0.71, None)
    _version(client, 0.80, stage="Production")
    assert _read(client) == (0.80, None)


def test_archived_versions_are_not_read(client):
    _version(client, 0.90, stage="Archived")
    assert _read(client) == (None, f"model_not_found:{NAME}")
