"""The routes split is a MOVE, not an edit: every /api/causal path, method, operationId,
parameters, request body and responses are byte-identical to the pre-split baseline.

A red here during the #1991 causal-routes split means the move changed the wire
contract - the fix is the code that moved, not this fixture. Regenerating the
fixture is legitimate ONLY when the wire change is intentional and reviewed.

To regenerate tests/unit/test_api/fixtures/causal_openapi_paths.json:

    .venv/bin/python -m scripts.export_openapi --output /tmp/openapi.json
    .venv/bin/python - <<'EOF'
    import json
    spec = json.load(open("/tmp/openapi.json"))
    paths = {k: v for k, v in spec["paths"].items() if k.startswith("/api/causal")}
    open("tests/unit/test_api/fixtures/causal_openapi_paths.json", "w").write(
        json.dumps(paths, indent=2, sort_keys=True) + "\n"
    )
    EOF
"""

from __future__ import annotations

import json
from pathlib import Path

# E2I_TESTING_MODE is set in tests/unit/test_api/conftest.py at import
# time so auth dependencies short-circuit.
from src.api.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "causal_openapi_paths.json"

# The pre-split route registration order, taken from the decorator order in
# ``git show c36eb5535:src/api/routes/causal.py`` and cross-checked against the
# path order in the tracked frontend/src/types/generated/api.ts. The two agree.
EXPECTED_ORDER = [
    "/api/causal/hierarchical/analyze",
    "/api/causal/hierarchical/{analysis_id}",
    "/api/causal/route",
    "/api/causal/brands",
    "/api/causal/variables",
    "/api/causal/propose-questions",
    "/api/causal/discover-effects/questions",
    "/api/causal/discover-effects",
    "/api/causal/discover-effects/{job_id}",
    "/api/causal/discover-effects/{job_id}/cancel",
    "/api/causal/clinical-context",
    "/api/causal/estimation-data",
    "/api/causal/agent-analyze",
    "/api/causal/agent-analyze/{analysis_id}",
    "/api/causal/pipeline/sequential",
    "/api/causal/pipeline/parallel",
    "/api/causal/pipeline/{pipeline_id}",
    "/api/causal/validate",
    "/api/causal/estimators",
    "/api/causal/health",
    "/api/causal/history",
    "/api/causal/value-chains",
    "/api/causal/treatment-effects",
]


def _first_diff(expected: object, actual: object, prefix: str) -> str | None:
    """Return a description of the first divergence between ``expected`` (the
    pinned baseline) and ``actual`` (the live OpenAPI schema), walking dicts
    and lists recursively - or ``None`` if they are identical.

    A plain string-equality assert truncates under this repo's ``--tb=short``
    addopt before the changed value appears; this names the exact
    path/method/field and both values instead.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                return f"{prefix}.{key}: in baseline, missing from current schema"
            if key not in expected:
                return f"{prefix}.{key}: in current schema, missing from baseline"
            diff = _first_diff(expected[key], actual[key], f"{prefix}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{prefix}: length {len(expected)} -> {len(actual)}"
        for index, (exp_item, act_item) in enumerate(zip(expected, actual, strict=True)):
            diff = _first_diff(exp_item, act_item, f"{prefix}[{index}]")
            if diff is not None:
                return diff
        return None
    if expected != actual:
        return f"{prefix}: {expected!r:.120} -> {actual!r:.120}"
    return None


def test_causal_openapi_paths_unchanged():
    spec = app.openapi()
    now = {k: v for k, v in spec["paths"].items() if k.startswith("/api/causal")}
    baseline = json.loads(FIXTURE.read_text())

    missing = sorted(set(baseline) - set(now))
    extra = sorted(set(now) - set(baseline))
    assert not missing and not extra, f"missing={missing} extra={extra}"

    for path in baseline:
        diff = _first_diff(baseline[path], now[path], prefix=path)
        assert diff is None, diff


def test_causal_openapi_path_order_unchanged():
    """Path ORDER is part of the wire contract, not just path content.

    The OpenAPI ``paths`` object preserves insertion order, ``openapi-typescript``
    emits frontend/src/types/generated/api.ts in that order, and CI's
    verify-types workflow diffs the regenerated file against the tracked one
    byte-for-byte. Registering a route at a different position is therefore a
    failing gate even when every path, method and schema is identical - during
    the #1991 split it showed up as a 234-line api.ts diff that was a pure block
    move. Splitting a concern across two routers to preserve this order is the
    fix; reordering the fixture is not.

    When a /api/causal endpoint is legitimately added or removed, update
    EXPECTED_ORDER from the decorator order of the owning route module (keep
    the aggregator's ``include_router`` order in
    src/api/routes/causal/__init__.py unchanged), then run
    ``make generate-types`` and commit the regenerated
    frontend/src/types/generated/api.ts - CI byte-diffs it.
    """
    spec = app.openapi()
    assert [p for p in spec["paths"] if p.startswith("/api/causal")] == EXPECTED_ORDER
