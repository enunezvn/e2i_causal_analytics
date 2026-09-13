"""The routes split is a MOVE, not an edit: every /api/causal path, method, operationId,
parameters, request body and responses are byte-identical to the pre-split baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("E2I_TESTING_MODE", "true")

from src.api.main import app  # noqa: E402 - env must be set before this import

FIXTURE = Path(__file__).parent / "fixtures" / "causal_openapi_paths.json"


def test_causal_openapi_paths_unchanged():
    spec = app.openapi()
    now = {k: v for k, v in spec["paths"].items() if k.startswith("/api/causal")}
    baseline = json.loads(FIXTURE.read_text())
    assert sorted(now) == sorted(baseline)
    for path in baseline:
        assert json.dumps(now[path], sort_keys=True) == json.dumps(
            baseline[path], sort_keys=True
        ), path
