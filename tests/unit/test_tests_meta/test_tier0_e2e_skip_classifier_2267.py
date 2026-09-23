"""#2267: the must-pass heavy e2e must not turn a defect into a silent skip.

``test_full_mlops_pipeline`` sat SKIPPED in the blocking Excluded Heavy E2E job
every night from 2026-09-12 to 2026-09-22 with no reason in the log, and the
MLflow model-logging defect it would have caught stayed hidden. Two guards:

1. Its MLflow skip fires only when MLflow is genuinely unavailable (connection
   refused / unresolvable / 503 / circuit breaker open), never on a defect whose
   message merely mentions MLflow (d6fcc0915 added the skip "for circuit
   breaker scenarios"; the ``"MLflow" in error_msg`` test matched far more).
2. The heavy job prints skip reasons (``-rs``), so a skip is never silent again.

The infra strings are the real messages, captured 2026-09-23 by driving the
real ``log_to_mlflow`` node against a closed port and an open circuit breaker.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from tests.integration.test_tier0_e2e import _mlflow_infra_unavailable

REPO = Path(__file__).resolve().parents[3]

_SERVER_DOWN = (
    "Training error (unknown_error): MLflow logging failed: API request to "
    "http://127.0.0.1:59999/api/2.0/mlflow/experiments/get-by-name failed with exception "
    "HTTPConnectionPool(host='127.0.0.1', port=59999): Max retries exceeded with url: "
    "/api/2.0/mlflow/experiments/get-by-name?experiment_name=e2i_model_trainer_probe "
    "(Caused by NewConnectionError(\"HTTPConnection(host='127.0.0.1', port=59999): "
    'Failed to establish a new connection: [Errno 111] Connection refused"))'
)
_BREAKER_OPEN = (
    "Training error (unknown_error): MLflow logging failed: MLflow circuit breaker is open"
)


@pytest.mark.parametrize(
    "error_msg",
    [
        _SERVER_DOWN,
        _BREAKER_OPEN,
        "MLflow logging failed: API request to http://mlflow:5000/api/2.0/mlflow/runs/create "
        "failed with exception HTTPConnectionPool(host='mlflow', port=5000): Max retries "
        "exceeded with url: /api/2.0/mlflow/runs/create (Caused by NameResolutionError("
        "\"<urllib3.connection.HTTPConnection object>: Failed to resolve 'mlflow' "
        '([Errno -2] Name or service not known)"))',
        "MLflow logging failed: API request to http://localhost:5000/api/2.0/mlflow/runs/create "
        "failed with exception HTTPConnectionPool(host='localhost', port=5000): Max retries "
        "exceeded with url: /api/2.0/mlflow/runs/create (Caused by ResponseError('too many "
        "503 error responses'))",
    ],
    ids=["connection_refused", "circuit_breaker_open", "dns_unresolvable", "http_503"],
)
def test_genuine_mlflow_unavailability_skips(error_msg: str) -> None:
    assert _mlflow_infra_unavailable(error_msg) is True


@pytest.mark.parametrize(
    "error_msg",
    [
        # The 2026-09-23 nightly: the model was never logged (NaN metrics).
        "missing_model_uri: Missing model_uri for registration",
        # The same defect, had the connector propagated the store's rejection.
        "Training error (unknown_error): MLflow logging failed: BAD_REQUEST: (raised as a "
        "result of Query-invoked autoflush; consider using a session.no_autoflush block if "
        "this flush is occurring prematurely)\n(sqlite3.IntegrityError) UNIQUE constraint "
        "failed: metrics.key, metrics.timestamp, metrics.step, metrics.run_uuid, "
        "metrics.value, metrics.is_nan",
        # A code bug inside the logging node.
        "Training error (unknown_error): MLflow logging failed: 'NoneType' object has no "
        "attribute 'run_id'",
        # A registration defect.
        "MLflow model registration failed: RESOURCE_ALREADY_EXISTS: Registered Model "
        "(name=kisqali_discontinuation) already exists.",
        "Training error (unknown_error): No trained model available",
    ],
    ids=["missing_model_uri", "metric_integrity_error", "code_bug", "registration", "no_model"],
)
def test_mlflow_defects_are_not_skipped(error_msg: str) -> None:
    assert _mlflow_infra_unavailable(error_msg) is False


def test_tier0_e2e_has_no_broad_mlflow_substring_skip() -> None:
    """Every MLflow skip goes through the classifier, not a bare substring test."""
    tree = ast.parse((REPO / "tests/integration/test_tier0_e2e.py").read_text())
    broad = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Constant)
        and node.left.value in ("MLflow", "circuit breaker")
        and any(isinstance(op, ast.In) for op in node.ops)
    ]
    assert broad == [], f"broad MLflow substring skip test(s) at lines {broad}"


def test_excluded_heavy_job_reports_skip_reasons() -> None:
    workflow = yaml.safe_load((REPO / ".github/workflows/slow-tests.yml").read_text())
    steps = workflow["jobs"]["excluded-heavy-tests"]["steps"]
    (heavy,) = [s for s in steps if s.get("id") == "heavy"]
    assert re.search(r"(?<!\S)-r[a-zA-Z]*s", heavy["run"]), (
        "the blocking heavy e2e job must print skip reasons (-rs), or a skip is silent"
    )
