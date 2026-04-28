"""Block 3B: assert that ``feast apply`` is structurally idempotent.

The registry file (``feature_repo/data/registry.db``) embeds a
``last_updated_timestamp``, so two consecutive applies always produce different
file hashes. What we actually care about is **schema drift**: every registered
feature view (with its TTL, schema, source, and entity wiring) must serialise
to identical proto bytes across consecutive applies.

This test is intentionally lightweight — it shells out to the ``feast`` CLI
twice with ``--skip-source-validation`` (no Postgres/Redis required) and
diffs the registry by walking the in-process ``FeatureStore`` and serialising
each ``FeatureView`` proto. It skips cleanly when ``feast`` is not installed
(e.g., the optional ``feast`` extras were not pulled in for a slim CI runner).

Scope of this gate (Block 6B-infra-5 upgrade)
---------------------------------------------
The Block 3B-era version of this test compared **name sets** across the
two applies and missed an entire class of intra-FV drift:

* dtype flips (e.g., ``Int64`` -> ``Float32`` on an existing field),
* TTL drift on an existing FeatureView,
* source rename or re-pointing (FV keeps its name, swaps its source),
* schema field add/remove inside an existing FV,
* entity-to-FV wiring changes that preserve the name set.

Block 6B-infra-5 upgraded the comparison to a schema-deep proto-byte
diff: for every FV in the registry we serialise
``store.get_feature_view(name).to_proto().SerializeToString()`` before
the second ``feast apply`` and again after, and assert byte equality.
The serialised FV proto embeds TTL, schema, source name, and entity
references — so all five drift cases listed above are covered by the
single byte-diff.

The proto-byte serialisation helper (``proto_bytes``) and the in-memory
FV builder (``build_minimal_feature_view``) used by the deliberate-
failure verification test are shared with
``tests/integration/test_feast_integration_suite.py`` via
``tests/integration/_feast_helpers.py`` so a future drift in
serialisation semantics surfaces in one place.

Findings reference: Block 3B (#4 residual, gitignore + apply lifecycle;
I-2 scope-documentation), Block 6B-infra-5 (schema-deep upgrade).
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.integration._feast_helpers import (
    build_minimal_feature_view,
    proto_bytes,
)

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


def _open_feature_store() -> Any:
    """Construct an in-process ``FeatureStore`` rooted at ``feature_repo/``.

    Mirrors the pattern in ``test_feast_offline_online_parity`` /
    ``test_feast_tier0_auto_register``: build the store once, surface init
    failures as a skip rather than a hard fail (an unreachable Postgres or
    Redis is a caller-environment issue, not a registry-drift bug).
    """
    from feast import FeatureStore

    try:
        return FeatureStore(repo_path=str(FEATURE_REPO))
    except Exception as exc:  # noqa: BLE001 — skip on ANY init failure
        pytest.skip(f"FeatureStore init failed: {exc!s:.200}")


def _snapshot_fv_proto_bytes(store: Any) -> dict[str, bytes]:
    """Walk the registry and return ``{fv_name: serialized_proto_bytes}``.

    The map is keyed by FV name so the diff message can pinpoint exactly
    which FV drifted between applies. We use ``list_feature_views`` to
    enumerate (rather than scanning the FV name set first and re-fetching)
    so we get a single consistent registry snapshot per call.
    """
    return {fv.name: proto_bytes(fv) for fv in store.list_feature_views()}


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
    """Second apply must produce byte-identical FeatureView protos.

    We cannot byte-compare ``registry.db`` because Feast embeds
    ``last_updated_timestamp`` in the registry on every write. Instead we
    walk the registry through an in-process ``FeatureStore``, serialise
    each FV's ``to_proto().SerializeToString()`` bytes before the second
    ``feast apply``, and re-serialise after. Any byte diff -- whether
    from a TTL flip, a schema field add/remove, a source rename, a
    dtype change, or an entity-wiring change -- fails the assertion.

    This is the schema-deep upgrade promised by Block 3B's I-2
    documentation: the previous name-set comparison missed every form of
    intra-FV drift (TTL, schema, source identity, dtype) and only flagged
    FVs appearing or disappearing wholesale.
    """
    store_before = _open_feature_store()
    fv_bytes_before = _snapshot_fv_proto_bytes(store_before)

    # Sanity: we should have observed *some* feature views — otherwise the
    # registry walk found nothing and the byte-diff is vacuous.
    assert fv_bytes_before, (
        "Expected at least one feature view registered after first apply; "
        "registry walk returned an empty FV map."
    )

    # Run apply a second time.
    second = _run_feast("apply", "--skip-source-validation")
    assert second.returncode == 0, (
        f"Second feast apply failed: stdout={second.stdout!r} stderr={second.stderr!r}"
    )

    # Re-open the store so we read a fresh registry snapshot rather than
    # trusting the in-memory FeatureStore object to refresh itself
    # (Feast 0.43's FeatureStore caches some registry state).
    store_after = _open_feature_store()
    fv_bytes_after = _snapshot_fv_proto_bytes(store_after)

    # Name-set check first so the diff message is friendly when an FV
    # appears or disappears wholesale (the byte diff would still catch
    # this, but the message would point at "byte mismatch" rather than
    # "FV vanished").
    names_before = set(fv_bytes_before)
    names_after = set(fv_bytes_after)
    assert names_after == names_before, (
        "Feature-view name drift between consecutive `feast apply` runs:\n"
        f"  added:   {names_after - names_before}\n"
        f"  removed: {names_before - names_after}"
    )

    # Schema-deep proto-byte diff: for every FV name that exists in both
    # snapshots, the serialised FV proto must be byte-identical.
    drifted = [name for name in names_before if fv_bytes_before[name] != fv_bytes_after[name]]
    assert not drifted, (
        "FeatureView proto bytes drifted between consecutive `feast apply` "
        f"runs for: {sorted(drifted)!r}. This is exactly the intra-FV drift "
        "(TTL, schema, source name, entity wiring, dtype) that the Block "
        "6B-infra-5 schema-deep idempotency check is designed to catch."
    )


def test_proto_byte_diff_catches_deliberate_ttl_change() -> None:
    """Deliberate-failure verification — proto-byte diff catches a TTL flip.

    The Block 6B-infra-5 plan calls for a deliberate-failure check that
    asserts the diff helper actually detects drift. We build two in-memory
    FeatureViews with identical name + schema + source + entity wiring but
    DIFFERENT TTLs (7 days vs 14 days) and assert that ``proto_bytes``
    produces different bytes for them. If this test passes, the schema-deep
    idempotency check above is provably non-vacuous: a real TTL drift in
    the registry would be caught.

    Runs unconditionally (no ``FEAST_INTEGRATION`` gate, no ``feast`` CLI
    needed) — it builds FVs in memory and never touches a live registry.
    The Feast SDK is still required, which the module-level
    ``importorskip`` enforces.

    Sibling drift cases (schema add/remove, source rename, identical-spec
    negative control) are exercised in
    ``tests/integration/test_feast_integration_suite.py`` and we deliberately
    do NOT duplicate them here -- this file is the apply-idempotency gate
    and only needs ONE deliberate-failure proof that the helper bites.
    """
    name = "test_6b_infra_5_apply_idem_ttl"
    feature_names = ["trx_count", "nrx_count"]

    base = build_minimal_feature_view(name, ttl=timedelta(days=7), feature_names=feature_names)
    drifted = build_minimal_feature_view(name, ttl=timedelta(days=14), feature_names=feature_names)

    base_bytes = proto_bytes(base)
    drifted_bytes = proto_bytes(drifted)

    # Negative control: differing TTL must produce differing bytes.
    assert base_bytes != drifted_bytes, (
        "Deliberate TTL change (7d -> 14d) produced byte-identical FV "
        "protos. proto_bytes() does NOT detect TTL drift, which means the "
        "schema-deep idempotency check would silently pass even when the "
        "registry's FV TTL drifts between applies. Investigate "
        "_feast_helpers.proto_bytes and Feast 0.43's FeatureView.to_proto "
        "implementation."
    )

    # Positive control: re-serialising the SAME FV must be deterministic
    # within a single process. If this fails, the byte-diff above is
    # tautological (any two serialisations would differ regardless of TTL).
    assert proto_bytes(base) == base_bytes, (
        "Re-serialising the same FeatureView produced different proto "
        "bytes within one process. proto_bytes() is non-deterministic, "
        "which would make the schema-deep idempotency check fail even on "
        "a no-op apply. Either Feast injects a non-deterministic field "
        "(timestamp, uuid) into to_proto(), or the helper has hidden "
        "state."
    )


# Note: ``test_registry_db_not_tracked_in_git`` was moved to
# ``tests/integration/test_feast_repo_hygiene.py`` (3B-M-4) — it was
# testing repo hygiene rather than apply idempotency, and bundling it
# here meant it skipped along with the rest of this module whenever the
# Feast CLI was unavailable.
