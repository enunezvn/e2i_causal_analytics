"""#1452 — health_score MLflow tracker: tracking-URI resolution + artifact destination.

Red-first regression tests for the production failure

    [WARNING] src.agents.health_score.mlflow_tracker -
      Failed to log health metrics to MLflow: [Errno 30] Read-only file system: '/mlflow'

emitted on EVERY health_score run in the ``e2i_api`` container.

Measured ground truth on the droplet (2026-08-03) that these tests encode:

* ``MLFLOW_TRACKING_URI=http://mlflow:5000`` IS present in ``e2i_api``
  (``docker/docker-compose.yml`` x-common-env line 43) and ``mlflow`` resolves
  on ``e2i_mlops_network`` -> the tracking server was never the problem.
* Metrics and tags DO persist: run ``ebf7988490e845f09d351785eac6450a`` on
  experiment 9 carries all five metrics and all three tags.
* What fails is the ARTIFACT upload. Experiment
  ``e2i_causal/health_score/default`` (id 9, created 2026-01-22) predates
  b0a30f11 ("use artifact proxy protocol for all agent MLflow trackers",
  2026-01-28) and therefore still carries the legacy filesystem
  ``artifact_location='/mlflow/artifacts/9'``. Its runs get
  ``artifact_uri='/mlflow/artifacts/9/<run_id>/artifacts'`` — a *server-local*
  path the MLflow client tries to ``mkdir`` inside the read-only api rootfs.
  ``artifact_location`` is create-time only and MLflow exposes no API to
  rewrite it, so the client must detect this and stop retrying every run.

Nothing here mocks ``mlflow``: every assertion runs the real resolution /
classification logic against real config inputs and real filesystem state.
"""

from __future__ import annotations

import logging
import os

import pytest

from src.agents.health_score.mlflow_tracker import (
    DEFAULT_TRACKING_URI,
    HealthScoreMLflowTracker,
    classify_artifact_destination,
    create_tracker,
    report_artifact_write_blocked,
    reset_tracking_reports,
    resolve_tracking_uri,
)

# The exact strings observed on the running droplet.
PROD_TRACKING_URI = "http://mlflow:5000"
PROD_LEGACY_ARTIFACT_URI = "/mlflow/artifacts/9/ebf7988490e845f09d351785eac6450a/artifacts"
PROD_PROXIED_ARTIFACT_URI = "mlflow-artifacts:/a7a0403f94c745cfb257c8d5851a8374/artifacts"


@pytest.fixture(autouse=True)
def _clean_report_state():
    """The once-per-process report ledger is module state; isolate each test."""
    reset_tracking_reports()
    yield
    reset_tracking_reports()


# =============================================================================
# TRACKING URI RESOLUTION (real config inputs, no mlflow involved)
# =============================================================================


class TestResolveTrackingUri:
    def test_env_var_is_used(self):
        assert resolve_tracking_uri(env={"MLFLOW_TRACKING_URI": PROD_TRACKING_URI}) == (
            PROD_TRACKING_URI
        )

    def test_explicit_argument_wins_over_env(self):
        assert (
            resolve_tracking_uri(
                "http://explicit:5000",
                env={"MLFLOW_TRACKING_URI": PROD_TRACKING_URI},
            )
            == "http://explicit:5000"
        )

    def test_default_is_a_tracking_server_not_a_local_store(self):
        """Absent config the tracker must still aim at a server, never ./mlruns."""
        resolved = resolve_tracking_uri(env={})
        assert resolved == DEFAULT_TRACKING_URI
        assert resolved.startswith("http")

    def test_blank_env_var_falls_back_to_default(self):
        assert resolve_tracking_uri(env={"MLFLOW_TRACKING_URI": "   "}) == DEFAULT_TRACKING_URI

    def test_tracker_resolves_env_at_construction(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", PROD_TRACKING_URI)
        assert HealthScoreMLflowTracker().tracking_uri == PROD_TRACKING_URI

    def test_tracker_explicit_uri_still_honoured(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", PROD_TRACKING_URI)
        tracker = HealthScoreMLflowTracker(tracking_uri="http://other:5000")
        assert tracker.tracking_uri == "http://other:5000"

    def test_factory_resolves_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", PROD_TRACKING_URI)
        assert create_tracker().tracking_uri == PROD_TRACKING_URI


# =============================================================================
# ARTIFACT DESTINATION CLASSIFICATION (real paths, real permissions)
# =============================================================================


class TestClassifyArtifactDestination:
    def test_proxied_uri_never_touches_the_local_filesystem(self):
        dest = classify_artifact_destination(PROD_PROXIED_ARTIFACT_URI)
        assert dest.local_path is None
        assert dest.is_blocked is False

    @pytest.mark.parametrize(
        "uri",
        [
            "http://mlflow:5000/api/2.0/mlflow-artifacts/artifacts/1/x",
            "s3://bucket/prefix/1/x",
        ],
    )
    def test_remote_schemes_are_not_local(self, uri):
        assert classify_artifact_destination(uri).local_path is None

    def test_writable_local_root_is_allowed(self, tmp_path):
        uri = str(tmp_path / "exp" / "run" / "artifacts")
        dest = classify_artifact_destination(uri)
        assert dest.local_path == uri
        assert dest.is_blocked is False

    def test_file_scheme_is_treated_as_local(self, tmp_path):
        dest = classify_artifact_destination(f"file://{tmp_path}/exp/run/artifacts")
        assert dest.local_path == f"{tmp_path}/exp/run/artifacts"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses filesystem permissions")
    def test_unwritable_local_root_is_blocked(self, tmp_path):
        """The production shape: target absent, nearest existing ancestor not writable."""
        jail = tmp_path / "jail"
        jail.mkdir()
        os.chmod(jail, 0o555)
        try:
            dest = classify_artifact_destination(str(jail / "artifacts" / "9" / "run"))
            assert dest.is_blocked is True
            assert dest.blocked_root == str(jail)
        finally:
            os.chmod(jail, 0o755)

    def test_production_legacy_uri_is_blocked_under_a_read_only_root(self, monkeypatch):
        """`/mlflow/artifacts/9/<run>/artifacts` must classify as blocked, not be retried.

        The api container has a read-only rootfs and no `/mlflow` mount, so the
        nearest existing ancestor is `/`. Simulated here with real filesystem
        semantics by asserting on whichever ancestor actually exists.
        """
        dest = classify_artifact_destination(PROD_LEGACY_ARTIFACT_URI)
        assert dest.local_path == PROD_LEGACY_ARTIFACT_URI
        # On a dev box `/` is not writable by uid 1000 either.
        if not os.access("/", os.W_OK):
            assert dest.is_blocked is True


# =============================================================================
# ONCE-PER-PROCESS REPORTING (the "not a per-run warning forever" requirement)
# =============================================================================


class TestReportedOnce:
    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses filesystem permissions")
    def test_blocked_artifact_root_warns_once_then_drops_to_debug(self, tmp_path, caplog):
        jail = tmp_path / "jail"
        jail.mkdir()
        os.chmod(jail, 0o555)
        try:
            uri = str(jail / "artifacts" / "9" / "run" / "artifacts")
            with caplog.at_level(logging.DEBUG, logger="src.agents.health_score.mlflow_tracker"):
                first = report_artifact_write_blocked(uri, run_id="run-1")
                second = report_artifact_write_blocked(uri, run_id="run-2")
                third = report_artifact_write_blocked(uri, run_id="run-3")
        finally:
            os.chmod(jail, 0o755)

        assert first is True
        assert second is False
        assert third is False

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"expected exactly one WARNING, got {len(warnings)}"
        assert str(jail) in warnings[0].getMessage()

        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debugs) == 2, "repeat occurrences must still be observable at DEBUG"

    def test_writable_root_reports_nothing(self, tmp_path, caplog):
        with caplog.at_level(logging.DEBUG, logger="src.agents.health_score.mlflow_tracker"):
            assert report_artifact_write_blocked(str(tmp_path / "a" / "b"), run_id="r") is False
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_proxied_uri_reports_nothing(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="src.agents.health_score.mlflow_tracker"):
            assert report_artifact_write_blocked(PROD_PROXIED_ARTIFACT_URI, run_id="r") is False
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses filesystem permissions")
    def test_warning_names_the_remediation(self, tmp_path, caplog):
        """A surfaced-once failure is only useful if it says what to do about it."""
        jail = tmp_path / "jail"
        jail.mkdir()
        os.chmod(jail, 0o555)
        try:
            with caplog.at_level(logging.WARNING, logger="src.agents.health_score.mlflow_tracker"):
                report_artifact_write_blocked(str(jail / "9" / "run" / "artifacts"), run_id="r")
        finally:
            os.chmod(jail, 0o755)

        message = caplog.records[0].getMessage()
        assert "mlflow-artifacts:" in message
        assert "artifact_location" in message


# =============================================================================
# END-TO-END AGAINST A REAL MLFLOW FILE STORE (no mlflow mocking anywhere)
# =============================================================================


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses filesystem permissions")
def test_repeated_runs_against_an_unwritable_artifact_root_warn_once(tmp_path, monkeypatch, caplog):
    """Reproduce production experiment 9 and prove the warning stops repeating.

    Real ``mlflow`` against a local file store (the established pattern in
    tests/unit/test_digital_twin/test_training_job.py) — no network, no mocks.
    An experiment is pre-created with a filesystem ``artifact_location`` the
    process cannot write to, exactly like ``e2i_causal/health_score/default``
    whose location is ``/mlflow/artifacts/9`` inside a read-only container.
    Three health runs must produce ONE warning, and every run's metrics must
    still land on the tracking store.
    """
    mlflow = pytest.importorskip("mlflow")

    store = tmp_path / "mlruns"
    store.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{store}")
    mlflow.set_tracking_uri(f"file://{store}")

    jail = tmp_path / "readonly_artifacts"
    jail.mkdir()

    experiment_name = "issue1452"
    full_name = f"e2i_causal/health_score/{experiment_name}"
    mlflow.create_experiment(full_name, artifact_location=str(jail / "9"))

    os.chmod(jail, 0o555)
    try:
        tracker = HealthScoreMLflowTracker()
        assert tracker.tracking_uri == f"file://{store}"

        output = _FakeHealthOutput()
        state = {"component_statuses": [{"name": "db", "status": "healthy"}]}

        run_ids = []
        with caplog.at_level(logging.DEBUG, logger="src.agents.health_score.mlflow_tracker"):
            for _ in range(3):
                import asyncio

                async def one_run():
                    async with tracker.start_health_run(
                        experiment_name=experiment_name, check_scope="full"
                    ) as ctx:
                        await tracker.log_health_result(output, state)
                        return ctx.run_id

                run_ids.append(asyncio.run(one_run()))
    finally:
        os.chmod(jail, 0o755)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, (
        f"three runs must surface the artifact problem once, got {len(warnings)}: "
        f"{[r.getMessage() for r in warnings]}"
    )
    assert "Failed to log health metrics" not in warnings[0].getMessage(), (
        "metrics are NOT what failed — misattributing this sends people to the wrong place"
    )

    # Every run's metrics reached the store even though artifacts could not.
    client = mlflow.tracking.MlflowClient()
    for run_id in run_ids:
        metrics = client.get_run(run_id).data.metrics
        assert metrics["overall_health_score"] == 42.0
        assert metrics["health_grade_numeric"] == 3.0


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses filesystem permissions")
def test_unusable_tracking_store_degrades_and_warns_once(tmp_path, monkeypatch, caplog):
    """A tracking store that cannot accept experiments must not re-warn every run.

    Uses a real ``file://`` store inside an unwritable directory — deterministic
    and network-free (an unreachable HTTP URI would sit in MLflow's retry
    backoff and blow the suite timeout).
    """
    pytest.importorskip("mlflow")

    store = tmp_path / "mlruns"
    store.mkdir()
    os.chmod(store, 0o555)
    tracking_uri = f"file://{store}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    try:
        tracker = HealthScoreMLflowTracker()
        assert tracker.tracking_uri == tracking_uri

        import asyncio

        async def one_run():
            async with tracker.start_health_run(experiment_name="dead", check_scope="full") as ctx:
                return ctx.run_id

        with caplog.at_level(logging.DEBUG, logger="src.agents.health_score.mlflow_tracker"):
            run_ids = [asyncio.run(one_run()) for _ in range(3)]
    finally:
        os.chmod(store, 0o755)

    assert run_ids == ["experiment-error"] * 3, "health checks must still complete"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected one warning, got {[r.getMessage() for r in warnings]}"
    assert tracking_uri in warnings[0].getMessage()


class _FakeHealthOutput:
    """A real HealthScoreOutput-shaped value object (not a mock of the unit under test).

    The unit under test is the tracker; this is its *input* datum, kept local so
    the test does not drag in the whole agent graph.
    """

    timestamp = "2026-08-03T16:31:51+00:00"
    overall_health_score = 42.0
    health_grade = "C"
    health_summary = "degraded"
    component_health_score = 0.7
    model_health_score = None
    pipeline_health_score = None
    agent_health_score = None
    critical_issues = ["db slow"]
    warnings = []
    total_latency_ms = 1554
