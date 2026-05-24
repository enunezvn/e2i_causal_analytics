"""Guard tests for integration-tests shard configuration in CI.

Issue #480 — PR #476 sharded integration-tests via ``pytest-split --splits 2``.
Post-merge CI showed 6:41 vs 14:00 imbalance because pytest-split distributes
by test-id hash without a ``.test_durations`` file. Resolution: bump the
matrix to ``N=4`` so finer-grained hash partitioning brings shards within
the 20% target and the slowest shard under 10 min.

These tests parse ``.github/workflows/backend-tests.yml`` and pin three
invariants:

1. The ``integration-tests`` matrix has exactly ``EXPECTED_SHARDS`` shards.
2. The pytest invocation uses ``--splits N --group ${{ matrix.shard }}``
   with ``N == EXPECTED_SHARDS`` (drift between matrix size and split count
   would silently lose tests or duplicate them across shards).
3. The ``ci-success`` aggregate job still ``needs: integration-tests`` —
   matrix aggregation means the required check is unaffected by shard count,
   and this guard catches a regression where someone wires
   ``needs: integration-tests-1`` (per-shard) instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "backend-tests.yml"
EXPECTED_SHARDS = 4


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Parse the backend-tests workflow once per test module."""
    assert WORKFLOW_PATH.exists(), f"workflow not found at {WORKFLOW_PATH}"
    with WORKFLOW_PATH.open() as fh:
        return yaml.safe_load(fh)


def test_integration_tests_matrix_has_expected_shard_count(workflow: dict) -> None:
    """The matrix.shard list size pins the parallel-job count for issue #480."""
    job = workflow["jobs"]["integration-tests"]
    shards = job["strategy"]["matrix"]["shard"]
    assert isinstance(shards, list), "matrix.shard must be a list"
    assert shards == list(range(1, EXPECTED_SHARDS + 1)), (
        f"matrix.shard expected {list(range(1, EXPECTED_SHARDS + 1))} "
        f"(N={EXPECTED_SHARDS} per issue #480 rebalance), got {shards}"
    )


def test_pytest_invocation_splits_match_matrix_size(workflow: dict) -> None:
    """``--splits N`` MUST match the matrix size; drift silently loses tests."""
    job = workflow["jobs"]["integration-tests"]
    run_step = next(step for step in job["steps"] if step.get("name") == "Run integration tests")
    cmd = run_step["run"]

    splits_match = re.search(r"--splits\s+(\d+)", cmd)
    assert splits_match is not None, "pytest invocation missing --splits flag"
    splits_n = int(splits_match.group(1))
    assert splits_n == EXPECTED_SHARDS, (
        f"--splits {splits_n} does not match matrix size {EXPECTED_SHARDS}; "
        "out-of-sync split-count and matrix size silently drops or duplicates tests"
    )

    assert "--group ${{ matrix.shard }}" in cmd, (
        "pytest invocation missing --group ${{ matrix.shard }} — each runner "
        "must select its shard by matrix coordinate"
    )


def test_ci_success_aggregates_integration_tests(workflow: dict) -> None:
    """``ci-success`` must depend on the matrix-level job name, not per-shard."""
    needs = workflow["jobs"]["ci-success"]["needs"]
    assert "integration-tests" in needs, (
        "ci-success must aggregate the integration-tests matrix as a single "
        "needs entry; per-shard wiring would break when shard count changes"
    )
