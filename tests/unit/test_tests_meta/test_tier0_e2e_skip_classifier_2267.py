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
import http.server
import re
import socket
import threading
from pathlib import Path
from typing import Iterator

import pytest
import yaml

from tests.integration.test_tier0_e2e import _mlflow_infra_unavailable, _mlflow_server_reachable

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
        "MLflow logging failed: API request to http://localhost:5000/api/2.0/mlflow/runs/create "
        "failed with timeout exception HTTPConnectionPool(host='localhost', port=5000): Read "
        "timed out. (read timeout=120). To increase the timeout, set the environment variable "
        "MLFLOW_HTTP_REQUEST_TIMEOUT to a larger value.",
        "MLflow logging failed: API request to http://localhost:5000/api/2.0/mlflow/runs/create "
        "failed with exception HTTPConnectionPool(host='localhost', port=5000): Max retries "
        "exceeded with url: /api/2.0/mlflow/runs/create (Caused by ResponseError('too many "
        "502 error responses'))",
        "MLflow logging failed: API request to http://localhost:5000/api/2.0/mlflow/runs/create "
        "failed with exception ('Connection aborted.', RemoteDisconnected('Remote end closed "
        "connection without response'))",
    ],
    ids=[
        "connection_refused",
        "circuit_breaker_open_server_down",
        "dns_unresolvable",
        "http_503",
        "request_timeout",
        "http_502",
        "connection_aborted",
    ],
)
def test_genuine_mlflow_unavailability_skips(error_msg: str) -> None:
    assert _mlflow_infra_unavailable(error_msg, server_reachable=lambda: False) is True


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
    assert _mlflow_infra_unavailable(error_msg, server_reachable=lambda: False) is False
    assert _mlflow_infra_unavailable(error_msg, server_reachable=lambda: True) is False


def test_open_breaker_with_a_reachable_server_is_a_defect() -> None:
    """The breaker also opens after repeated DEFECTS in one process (every
    connector exception counts), so an open breaker only means "MLflow is
    down" when the server really does not answer."""
    assert _mlflow_infra_unavailable(_BREAKER_OPEN, server_reachable=lambda: True) is False
    assert _mlflow_infra_unavailable(_BREAKER_OPEN, server_reachable=lambda: False) is True


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def answering_server() -> Iterator[str]:
    class _Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_probe_reports_a_closed_port_unreachable() -> None:
    assert _mlflow_server_reachable(f"http://127.0.0.1:{_closed_port()}", timeout=2.0) is False


def test_probe_reports_an_answering_server_reachable(answering_server: str) -> None:
    assert _mlflow_server_reachable(answering_server, timeout=2.0) is True


def test_probe_treats_a_local_store_as_reachable(tmp_path: Path) -> None:
    assert _mlflow_server_reachable(f"sqlite:///{tmp_path / 'mlflow.db'}") is True


def test_tier0_e2e_has_no_broad_mlflow_substring_skip() -> None:
    """Every MLflow skip goes through the classifier, not a bare substring test."""
    tree = ast.parse((REPO / "tests/integration/test_tier0_e2e.py").read_text())
    broad = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
        and node.left.value.lower() in ("mlflow", "circuit breaker")
        and any(isinstance(op, ast.In) for op in node.ops)
    ]
    assert broad == [], f"broad MLflow substring skip test(s) at lines {broad}"


def test_qc_gate_failure_fails_the_pipeline_test() -> None:
    """The synthetic sample is built to pass QC (a8aa8ed68: "Data Preparer: QC gate
    passes"); a QC-gate skip is what hid the pipeline for 11 nights."""
    tree = ast.parse((REPO / "tests/integration/test_tier0_e2e.py").read_text())
    qc_skips = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skip"
        and any(
            isinstance(arg, ast.Constant) and "qc" in str(arg.value).lower() for arg in node.args
        )
    ]
    assert qc_skips == [], f"QC-gate skip(s) at lines {qc_skips}"


def test_excluded_heavy_job_reports_skip_reasons() -> None:
    workflow = yaml.safe_load((REPO / ".github/workflows/slow-tests.yml").read_text())
    steps = workflow["jobs"]["excluded-heavy-tests"]["steps"]
    (heavy,) = [s for s in steps if s.get("id") == "heavy"]
    assert "tests/integration/test_tier0_e2e.py" in heavy["run"]
    assert re.search(r"(?<!\S)-r[a-zA-Z]*s", heavy["run"]), (
        "the blocking heavy e2e job must print skip reasons (-rs), or a skip is silent"
    )
