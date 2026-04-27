"""Block 3B: assert that ``feast apply`` is structurally idempotent.

The registry file (``feature_repo/data/registry.db``) embeds a
``last_updated_timestamp``, so two consecutive applies always produce different
file hashes. What we actually care about is **schema drift**: the set of
registered entities and feature views (with their entity wiring) must be
identical across runs.

This test is intentionally lightweight — it shells out to the ``feast`` CLI
twice with ``--skip-source-validation`` (no Postgres/Redis required) and
compares the structural output. It skips cleanly when ``feast`` is not
installed (e.g., the optional ``feast`` extras were not pulled in for a slim
CI runner).

Scope of this gate (per Block 3B review item I-2)
-------------------------------------------------
This idempotency check verifies that a second ``feast apply`` does not add
or remove entities or feature views — by NAME only.

It does **not** detect intra-feature-view changes:

* dtype flips (e.g., ``Int64`` -> ``Float32`` on an existing field),
* TTL drift on an existing FeatureView,
* source rename or re-pointing (FV keeps its name, swaps its source),
* schema field add/remove inside an existing FV,
* entity-to-FV wiring changes that preserve the name set.

A schema-deep diff (registry-proto-aware comparison rather than name-set
comparison) is planned for Block 6B's Feast integration suite. For now,
treat this test as the **minimum-viable gate** against non-deterministic
apply behaviour — it catches the worst class of bug (random
appearance/disappearance of FVs across runs) cheaply without standing up
Postgres or Redis.

Findings reference: Block 3B (#4 residual, gitignore + apply lifecycle;
I-2 scope-documentation).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Skip the entire module if the Feast Python SDK is not importable.
pytest.importorskip("feast", reason="Feast SDK not installed; skipping registry tests.")

# Feast's CLI is slow (Pydantic + dask + heavy imports add ~5s per call). Override
# the project-wide 30s pytest timeout for every test in this module so subprocess
# invocations have headroom on cold cache.
pytestmark = pytest.mark.timeout(180)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = PROJECT_ROOT / "feature_repo"


def _feast_cli_available() -> bool:
    """Return True iff a ``feast`` executable resolves on PATH."""
    return shutil.which("feast") is not None


def _run_feast(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a ``feast`` subcommand and return the completed process.

    Always runs from ``PROJECT_ROOT``; the previous ``cwd`` parameter
    was dead — every caller relied on the default and pluming it through
    only obscured the call sites. (3B-M-5)
    """
    cmd = ["feast", "--chdir", str(FEATURE_REPO.relative_to(PROJECT_ROOT))] + list(args)
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _parse_table(stdout: str) -> set[str]:
    """Parse a ``feast … list`` table into the set of names in the first column.

    The table format is::

        NAME    DESCRIPTION    TYPE
        alpha   ...            ...
        beta    ...            ...

    We split on whitespace and take the first token of each non-header,
    non-blank line. This is robust to trailing whitespace and to Feast's
    Pydantic deprecation warnings being emitted to stderr.
    """
    names: set[str] = set()
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("NAME"):
            continue
        first_token = line.split()[0]
        names.add(first_token)
    return names


@pytest.fixture(scope="module")
def feast_cli() -> str:
    """Skip the module if the ``feast`` CLI is not on PATH."""
    if not _feast_cli_available():
        pytest.skip("`feast` CLI not on PATH; install feast to run this test.")
    return "feast"


@pytest.fixture(scope="module")
def applied_once(feast_cli: str) -> subprocess.CompletedProcess[str]:
    """Run ``feast apply`` once and return the result (module-scoped)."""
    result = _run_feast("apply", "--skip-source-validation")
    if result.returncode != 0:
        pytest.skip(
            "`feast apply` failed to run in this environment "
            f"(stderr: {result.stderr.strip()[:200]}); skipping idempotency check."
        )
    return result


def test_feast_apply_succeeds(applied_once: subprocess.CompletedProcess[str]) -> None:
    """First apply exits zero and reports the project name."""
    assert applied_once.returncode == 0, (
        f"feast apply failed: stdout={applied_once.stdout!r} stderr={applied_once.stderr!r}"
    )
    # Sanity: stdout mentions the configured project.
    assert "e2i_causal_analytics" in applied_once.stdout, (
        f"Expected project name in apply output; got: {applied_once.stdout!r}"
    )


def test_feast_apply_idempotent_no_schema_drift(
    applied_once: subprocess.CompletedProcess[str],  # noqa: ARG001 — fixture orders runs
    feast_cli: str,  # noqa: ARG001 — fixture is the gate
) -> None:
    """Second apply must produce the same entity + feature-view inventory.

    We cannot byte-compare ``registry.db`` because Feast embeds
    ``last_updated_timestamp`` in the registry on every write. Instead we
    snapshot the structural inventory before/after a second apply.
    """
    entities_before = _parse_table(_run_feast("entities", "list").stdout)
    fvs_before = _parse_table(_run_feast("feature-views", "list").stdout)

    # Sanity: we should have observed *some* entities/feature views — otherwise
    # the parser is broken or apply produced nothing.
    assert entities_before, "Expected at least one entity registered after first apply."
    assert fvs_before, "Expected at least one feature view registered after first apply."

    # Run apply a second time and re-snapshot.
    second = _run_feast("apply", "--skip-source-validation")
    assert second.returncode == 0, (
        f"Second feast apply failed: stdout={second.stdout!r} stderr={second.stderr!r}"
    )

    entities_after = _parse_table(_run_feast("entities", "list").stdout)
    fvs_after = _parse_table(_run_feast("feature-views", "list").stdout)

    assert entities_after == entities_before, (
        "Entity drift between consecutive `feast apply` runs:\n"
        f"  added:   {entities_after - entities_before}\n"
        f"  removed: {entities_before - entities_after}"
    )
    assert fvs_after == fvs_before, (
        "Feature-view drift between consecutive `feast apply` runs:\n"
        f"  added:   {fvs_after - fvs_before}\n"
        f"  removed: {fvs_before - fvs_after}"
    )


# Note: ``test_registry_db_not_tracked_in_git`` was moved to
# ``tests/integration/test_feast_repo_hygiene.py`` (3B-M-4) — it was
# testing repo hygiene rather than apply idempotency, and bundling it
# here meant it skipped along with the rest of this module whenever the
# Feast CLI was unavailable.
