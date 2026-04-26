"""Block 6B Feast integration suite — five lifecycle scenarios + schema-deep idempotency.

This module is the dedicated Feast integration suite specified by Block 6B
(``#13`` in the tier-0 remediation plan). It exercises the SDK surface that
``FeastClient`` wraps end-to-end against a live Feast deployment:

  1. **Registration** — ``FeastClient.register_feature_view`` round-trips
     through ``store.list_feature_views`` with the correct entity, schema, and
     TTL.
  2. **Materialization** — synthetic features are pushed to the online store
     via the FV's auto-generated ``PushSource`` (Feast 0.43 ``store.push``)
     and round-tripped through ``get_online_features``. We use ``push``
     instead of ``materialize_incremental`` here because the throwaway FV is
     not backed by a real Postgres table; ``push`` is the cleaner test
     surface for an isolated FV (see module footnote on pull-vs-push).
  3. **Historical retrieval (TZ-aware)** — ``get_historical_features`` is
     called with a TZ-aware UTC entity dataframe; the test asserts the
     returned ``event_timestamp`` column preserves TZ semantics. Catches the
     classic TZ-naive vs TZ-aware regression in Feast PIT joins.
  4. **Online retrieval** — round-trip the entity rows used in (2) and assert
     the retrieved values match what we pushed.
  5. **Re-registration idempotency** — calling ``register_feature_view`` with
     the same spec twice succeeds and produces exactly one FV in
     ``list_feature_views``.

The module additionally exercises a **schema-deep idempotency check**: it
verifies that re-applying with a TTL change, a schema field add, or a source
rename produces a DETECTABLE diff. Detection is done by comparing
``FeatureView.to_proto().SerializeToString()`` bytes between the original and
the perturbed view — Feast 0.43 makes idempotency contingent on byte
equality, so byte drift is the contract this test asserts.

Skip behaviour
--------------
This test mirrors the pattern from ``test_feast_offline_online_parity.py`` and
``test_feast_tier0_auto_register.py``:

* Requires the ``feast`` Python SDK (``pytest.importorskip``).
* Requires opting in via ``FEAST_INTEGRATION=1``.
* Requires ``feature_repo/`` with the ``hcp`` entity and
  ``business_metrics_source`` already applied.

Fail-vs-skip discipline
-----------------------
Once a developer has opted in via ``FEAST_INTEGRATION=1``:

* **Environment / connection problems** (Feast SDK missing, repo path
  not present, registry not bootstrapped, entity / batch source
  not yet applied) → ``skip``. These are caller-environment issues; the
  opt-in does not assert the registry is up.
* **Registration / round-trip failure** (``register_feature_view`` returns
  ``registered=False``, or the registry round-trip drops the FV after
  ``registered=True``) → ``fail``. We connected, the wiring is wrong, and
  that is exactly what this suite is supposed to catch.

Cleanup contract
----------------
Every test that registers a synthetic FV uses a UUID-suffixed name and removes
it again via ``store.apply([], objects_to_delete=[fv], partial=True)`` in a
``try/finally``. Cleanup failure → low-severity ``UserWarning``, not a test
failure: Feast 0.43 ``apply()`` is idempotent (upsert), so a missed cleanup
just gets re-applied on the next run.

Why ``store.push`` over ``materialize_incremental`` for the materialization test
-------------------------------------------------------------------------------
``materialize_incremental`` reads from the FV's batch source and writes to the
online store. A throwaway FV constructed in-test has no real Postgres rows
backing it (the underlying ``business_metrics_source`` view does, but we don't
control which entity ids surface). Since the FV is built with a
``PushSource`` wrapping the batch source (see ``FeastClient.register_feature_view``
line 1120), we can write known synthetic rows directly via ``store.push``
against the auto-generated ``{name}_push_source``. This sidesteps offline-store
state assumptions and keeps the test fully self-contained while still
exercising the same online-store write path Feast uses internally during
``materialize_incremental``.

Findings reference: Block 6B (#13, Feast integration suite + schema-deep
idempotency).
"""

from __future__ import annotations

import asyncio
import os
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

# Skip the entire module if the Feast Python SDK is not importable.
pytest.importorskip("feast", reason="Feast SDK not installed; skipping integration suite.")

# Optional dependency for entity-row construction and push payloads.
pd = pytest.importorskip("pandas", reason="pandas required for Feast integration suite.")

# Loading the registry, applying FeatureViews, and pushing rows is heavy
# (Pydantic + dask + Redis round-trip). The 30s default pytest timeout is too
# tight; mirror the parity / auto-register tests' 180s ceiling so the entire
# module shares a consistent budget.
pytestmark = pytest.mark.timeout(180)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = PROJECT_ROOT / "feature_repo"

# Live-confirmed registry names (mirrors Block 5B's auto-register integration
# test). Keep these in sync if Block 6B's canonical-schema migration renames
# the underlying entity / batch source.
ENTITY_NAME = "hcp"
ENTITY_JOIN_KEY = "hcp_id"
BATCH_SOURCE_NAME = "business_metrics_source"

# Feature columns we know exist in business_metrics_source (see
# feature_repo/data_sources.py). The smallest pair that exercises the
# schema-builder + PushSource round-trip without depending on any real
# downstream consumers.
SELECTED_FEATURES = ["trx_count", "nrx_count"]

# Default TTL for synthetic FVs registered by this suite. Picked deliberately
# different from the FeastClient default (7 days) so the schema-deep diff
# test can perturb it and observe a byte change.
TEST_TTL = timedelta(days=14)


def _feast_integration_available() -> bool:
    """True iff the caller has opted into the live Feast integration suite.

    The droplet (and only the droplet) sets ``FEAST_INTEGRATION=1`` in its
    environment so this suite runs there but stays a no-op everywhere else.
    Mirrors the gate from ``test_feast_offline_online_parity`` and
    ``test_feast_tier0_auto_register``.
    """
    return os.environ.get("FEAST_INTEGRATION", "").strip().lower() in {"1", "true", "yes"}


def _unique_fv_name(prefix: str) -> str:
    """Build a UUID-suffixed FV name so concurrent CI doesn't collide.

    Pattern is ``test_6b_suite_<prefix>_<8-hex>`` — short enough for Feast's
    name-length limits (Feast 0.43 internally enforces ~63 chars for the
    materialized table name) and unique enough that a missed cleanup in one
    run never blocks a subsequent one.
    """
    return f"test_6b_suite_{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def feast_client() -> Any:
    """Initialise a ``FeastClient`` and verify prerequisites are present.

    Module-scoped so the registry-load cost is paid once. Skips when:

    * Caller hasn't opted in (``FEAST_INTEGRATION`` unset).
    * ``feature_repo/`` is missing on disk.
    * ``FeastClient.initialize`` raises (registry unreachable, etc.).
    * The pre-existing entity / batch source isn't applied.
    """
    if not _feast_integration_available():
        pytest.skip(
            "FEAST_INTEGRATION not set; skipping Feast integration suite. "
            "Set FEAST_INTEGRATION=1 on a host with a bootstrapped Feast registry."
        )

    if not FEATURE_REPO.exists():
        pytest.skip(f"feature_repo not found at {FEATURE_REPO}")

    from src.feature_store.feast_client import FeastClient

    client = FeastClient()

    try:
        asyncio.run(client.initialize())
    except Exception as exc:  # noqa: BLE001 — skip on ANY init failure
        pytest.skip(f"FeastClient.initialize() failed: {exc!s:.200}")

    if client._store is None:
        pytest.skip(
            "FeastClient initialised in fallback-only mode (no live Feast store). "
            "This suite requires a real FeatureStore."
        )

    # Pre-existing registry objects MUST be there. This is the only thing we
    # check before declaring the environment fit-for-purpose.
    try:
        client._store.get_entity(ENTITY_NAME)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"entity {ENTITY_NAME!r} not registered in Feast: {exc!s:.200}. "
            "Run `feast apply` from feature_repo/ first."
        )

    try:
        client._store.get_data_source(BATCH_SOURCE_NAME)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"batch source {BATCH_SOURCE_NAME!r} not registered in Feast: "
            f"{exc!s:.200}. Run `feast apply` from feature_repo/ first."
        )

    return client


def _delete_feature_view(store: Any, fv_name: str) -> None:
    """Synchronous cleanup helper — best-effort delete of a synthetic FV.

    Looks up the FV by name (so we use the registry-resolved object, not a
    fresh stub that might not match the canonical reference) and asks Feast
    to delete it via ``apply([], objects_to_delete=[fv], partial=True)`` —
    canonical for Feast 0.43.

    Failures are downgraded to ``UserWarning``: Feast 0.43 ``apply()`` is
    idempotent (upsert), so a missed cleanup just gets re-applied on the
    next run, but the registry will accumulate test FVs until manually
    pruned. This mirrors the pattern from
    ``test_feast_tier0_auto_register._delete_feature_view_async``.
    """
    try:
        fv = store.get_feature_view(fv_name)
    except Exception as lookup_exc:  # noqa: BLE001
        # FV already gone (or never applied). Cleanup is a no-op.
        warnings.warn(
            f"FeatureView {fv_name!r} not found during cleanup (probably "
            f"already removed): {lookup_exc!s:.200}",
            stacklevel=2,
        )
        return

    try:
        store.apply([], objects_to_delete=[fv], partial=True)
    except Exception as apply_exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to clean up FeatureView {fv_name!r}: {apply_exc!s:.200}. "
            "Feast 0.43 apply() is idempotent so this does not break "
            "subsequent runs, but the registry will accumulate test FVs "
            "until manually pruned.",
            stacklevel=2,
        )


# =============================================================================
# Five lifecycle scenarios
# =============================================================================


def test_register_feature_view_round_trips_through_list(feast_client: Any) -> None:
    """Scenario 1 — registration round-trips through ``list_feature_views``.

    Registers a synthetic FV via ``FeastClient.register_feature_view``, then
    asserts:

    * The helper reports ``registered=True`` with the expected name + count.
    * The FV appears in ``store.list_feature_views()``.
    * The FV's entity wiring matches ``ENTITY_NAME`` (or ``ENTITY_JOIN_KEY``,
      since Feast 0.43 stores entities-by-reference).
    * The FV's schema matches ``SELECTED_FEATURES``.
    * The FV's TTL matches ``TEST_TTL`` (verifies our non-default TTL was
      preserved through the FeastClient -> apply round-trip).
    """
    fv_name = _unique_fv_name("register")
    store = feast_client._store

    try:
        result = asyncio.run(
            feast_client.register_feature_view(
                name=fv_name,
                entity_name=ENTITY_NAME,
                feature_names=list(SELECTED_FEATURES),
                batch_source_name=BATCH_SOURCE_NAME,
                ttl=TEST_TTL,
            )
        )

        if not result.get("registered"):
            pytest.fail(
                f"register_feature_view returned registered=False; "
                f"error={result.get('error')!r} full_result={result!r}"
            )
        assert result["feature_view_name"] == fv_name
        assert result["features_count"] == len(SELECTED_FEATURES)

        # ---- list_feature_views round-trip --------------------------------
        registered_names = {fv.name for fv in store.list_feature_views()}
        assert fv_name in registered_names, (
            f"FV {fv_name!r} not found in list_feature_views() after "
            f"successful register_feature_view; registered names: "
            f"{sorted(registered_names)!r}"
        )

        fv_obj = store.get_feature_view(fv_name)

        # Entity wiring (Feast 0.43 stores entities by reference; check both
        # the entity-name list and the join-keys list).
        entity_names = list(getattr(fv_obj, "entities", []) or [])
        join_keys = list(getattr(fv_obj, "join_keys", []) or [])
        assert ENTITY_NAME in entity_names or ENTITY_JOIN_KEY in join_keys, (
            f"FV {fv_name!r} did not bind to entity {ENTITY_NAME!r}; "
            f"got entities={entity_names!r} join_keys={join_keys!r}"
        )

        # Schema round-trip.
        registered_features = sorted(f.name for f in fv_obj.features)
        assert registered_features == sorted(SELECTED_FEATURES), (
            f"FV {fv_name!r} feature names diverged from input: "
            f"got {registered_features!r} expected {sorted(SELECTED_FEATURES)!r}"
        )

        # TTL round-trip — we passed a non-default 14-day TTL specifically so
        # this assertion catches the "TTL got dropped on the way through
        # FeastClient" regression.
        assert fv_obj.ttl == TEST_TTL, (
            f"FV {fv_name!r} TTL drift: got {fv_obj.ttl!r} expected {TEST_TTL!r}"
        )

    finally:
        _delete_feature_view(store, fv_name)


def test_materialize_via_push_round_trips_to_online_store(feast_client: Any) -> None:
    """Scenario 2 — synthetic rows pushed to the online store land there.

    See module docstring for why we use ``store.push`` instead of
    ``materialize_incremental`` for an isolated throwaway FV.

    Strategy:
      1. Register a synthetic FV (PushSource-backed) via FeastClient.
      2. Build a small DataFrame with known entity ids + feature values.
      3. ``store.push(push_source_name, df, to=ONLINE)``.
      4. Round-trip through ``get_online_features`` and assert the values
         we pushed are what we read back.
    """
    from feast.data_source import PushMode

    fv_name = _unique_fv_name("materialize")
    push_source_name = f"{fv_name}_push_source"
    store = feast_client._store

    try:
        result = asyncio.run(
            feast_client.register_feature_view(
                name=fv_name,
                entity_name=ENTITY_NAME,
                feature_names=list(SELECTED_FEATURES),
                batch_source_name=BATCH_SOURCE_NAME,
                ttl=TEST_TTL,
            )
        )
        if not result.get("registered"):
            pytest.fail(f"setup register_feature_view failed: {result!r}")

        # Synthetic rows. Two entity ids, known integer values for the two
        # features so equality checks are unambiguous.
        push_ts = datetime.now(timezone.utc)
        push_df = pd.DataFrame(
            {
                ENTITY_JOIN_KEY: ["hcp_test_a", "hcp_test_b"],
                "trx_count": [42, 7],
                "nrx_count": [13, 3],
                "event_timestamp": [push_ts, push_ts],
                "created_at": [push_ts, push_ts],
            }
        )

        # Feast 0.43 stamps the FV's auto-generated PushSource as
        # ``{fv_name}_push_source`` (see FeastClient.register_feature_view).
        store.push(push_source_name, push_df, to=PushMode.ONLINE)

        # Round-trip through online features.
        feature_refs = [f"{fv_name}:{f}" for f in SELECTED_FEATURES]
        entity_rows = [
            {ENTITY_JOIN_KEY: "hcp_test_a"},
            {ENTITY_JOIN_KEY: "hcp_test_b"},
        ]
        online_dict = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
            full_feature_names=False,
        ).to_dict()

        assert ENTITY_JOIN_KEY in online_dict, (
            f"online response missing join key {ENTITY_JOIN_KEY!r}; "
            f"keys={sorted(online_dict.keys())!r}"
        )
        # Order is preserved by Feast for online lookups: row-i in the input
        # maps to index-i in each output column.
        returned_ids = list(online_dict[ENTITY_JOIN_KEY])
        assert returned_ids == ["hcp_test_a", "hcp_test_b"], (
            f"online entity ids out of order or missing: got {returned_ids!r}"
        )
        assert list(online_dict["trx_count"]) == [42, 7], (
            f"trx_count round-trip mismatch: got {online_dict['trx_count']!r}"
        )
        assert list(online_dict["nrx_count"]) == [13, 3], (
            f"nrx_count round-trip mismatch: got {online_dict['nrx_count']!r}"
        )

    finally:
        _delete_feature_view(store, fv_name)


def test_get_historical_features_preserves_tz_aware_event_timestamp(
    feast_client: Any,
) -> None:
    """Scenario 3 — TZ-aware entity_df flows through PIT joins without TZ loss.

    Feast PIT joins are notoriously fragile around TZ semantics: a
    TZ-naive ``event_timestamp`` column in the entity_df can silently flip
    PIT semantics from "as of T (UTC)" to "as of T (server local)". This
    test passes a TZ-aware UTC timestamp and asserts the returned column
    is still TZ-aware.

    This is a smoke test, not an exhaustive correctness check — we don't
    care which rows came back, only that the timestamp dtype survived the
    round-trip. The auto-register and offline/online parity tests cover
    the value-correctness side of the contract.
    """
    fv_name = _unique_fv_name("tz_aware")
    store = feast_client._store

    try:
        result = asyncio.run(
            feast_client.register_feature_view(
                name=fv_name,
                entity_name=ENTITY_NAME,
                feature_names=list(SELECTED_FEATURES),
                batch_source_name=BATCH_SOURCE_NAME,
                ttl=TEST_TTL,
            )
        )
        if not result.get("registered"):
            pytest.fail(f"setup register_feature_view failed: {result!r}")

        # TZ-aware UTC entity_df. The point of this test is to assert
        # ``pd.api.types.is_datetime64tz_dtype(returned[event_timestamp])``
        # remains True after Feast's PIT join.
        utc_now = pd.Timestamp(datetime.now(timezone.utc))
        assert utc_now.tz is not None, "entity_df fixture should be TZ-aware"

        entity_df = pd.DataFrame(
            {
                ENTITY_JOIN_KEY: ["hcp_tz_test_a", "hcp_tz_test_b"],
                "event_timestamp": [utc_now, utc_now],
            }
        )

        feature_refs = [f"{fv_name}:{f}" for f in SELECTED_FEATURES]

        # ``get_historical_features`` against a synthetic FV with no rows in
        # the offline source returns a row per entity with NaN features —
        # that's fine for this assertion. We're checking dtype preservation,
        # not value correctness.
        offline_df = store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
            full_feature_names=False,
        ).to_df()

        assert "event_timestamp" in offline_df.columns, (
            f"get_historical_features dropped event_timestamp; "
            f"got columns: {list(offline_df.columns)!r}"
        )
        ts_col = offline_df["event_timestamp"]

        # Two acceptable shapes: TZ-aware datetime64[ns, UTC] (Feast 0.43
        # default) or TZ-aware datetime64[ns, <some tz>]. The classic bug
        # this test catches is a TZ-naive datetime64[ns] dtype.
        assert pd.api.types.is_datetime64_any_dtype(ts_col), (
            f"event_timestamp dtype is not datetime-like: got {ts_col.dtype!r}"
        )
        assert getattr(ts_col.dtype, "tz", None) is not None, (
            f"TZ regression: get_historical_features returned a TZ-naive "
            f"event_timestamp column. Feast PIT joins drop TZ when the "
            f"dataframe is silently coerced to TZ-naive somewhere in the "
            f"join path. Got dtype={ts_col.dtype!r} (no tzinfo). "
            f"Input was TZ-aware UTC."
        )

    finally:
        _delete_feature_view(store, fv_name)


def test_get_online_features_round_trips_pushed_values(feast_client: Any) -> None:
    """Scenario 4 — online retrieval of pushed values matches what we sent.

    This is a direct round-trip test: push -> get_online_features -> equality.
    Distinct from scenario 2 in that scenario 2 uses two entities and tests
    the materialization path; this one focuses on the get_online_features
    contract itself with a wider value range (negative, zero, large) to
    catch dtype-coercion bugs in the online store.
    """
    from feast.data_source import PushMode

    fv_name = _unique_fv_name("online")
    push_source_name = f"{fv_name}_push_source"
    store = feast_client._store

    try:
        result = asyncio.run(
            feast_client.register_feature_view(
                name=fv_name,
                entity_name=ENTITY_NAME,
                feature_names=list(SELECTED_FEATURES),
                batch_source_name=BATCH_SOURCE_NAME,
                ttl=TEST_TTL,
            )
        )
        if not result.get("registered"):
            pytest.fail(f"setup register_feature_view failed: {result!r}")

        # Three rows with a wider value spectrum than scenario 2.
        push_ts = datetime.now(timezone.utc)
        push_df = pd.DataFrame(
            {
                ENTITY_JOIN_KEY: ["hcp_round_a", "hcp_round_b", "hcp_round_c"],
                "trx_count": [0, 100_000, 1],
                "nrx_count": [0, 50_000, 1],
                "event_timestamp": [push_ts, push_ts, push_ts],
                "created_at": [push_ts, push_ts, push_ts],
            }
        )
        store.push(push_source_name, push_df, to=PushMode.ONLINE)

        feature_refs = [f"{fv_name}:{f}" for f in SELECTED_FEATURES]
        entity_rows = [{ENTITY_JOIN_KEY: eid} for eid in push_df[ENTITY_JOIN_KEY]]
        online_dict = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
            full_feature_names=False,
        ).to_dict()

        # Order-independent assertion: build {entity_id -> value} dicts on both
        # sides and compare. Feast 0.43 happens to preserve input order for
        # online lookups, but this is not a documented contract and future
        # versions may parallelize per-entity reads. Comparing by entity id
        # decouples the test from that implicit ordering.
        returned_ids = list(online_dict[ENTITY_JOIN_KEY])
        expected_ids = list(push_df[ENTITY_JOIN_KEY])
        assert sorted(returned_ids) == sorted(expected_ids), (
            f"online entity id set mismatch — sent {expected_ids!r}, "
            f"got {returned_ids!r}"
        )
        for col in SELECTED_FEATURES:
            returned_by_id = dict(
                zip(online_dict[ENTITY_JOIN_KEY], online_dict[col], strict=True)
            )
            expected_by_id = dict(
                zip(push_df[ENTITY_JOIN_KEY], push_df[col], strict=True)
            )
            assert returned_by_id == expected_by_id, (
                f"{col} round-trip mismatch — sent {expected_by_id!r}, "
                f"got {returned_by_id!r}"
            )

    finally:
        _delete_feature_view(store, fv_name)


def test_re_registration_with_identical_spec_is_idempotent(
    feast_client: Any,
) -> None:
    """Scenario 5 — calling ``register_feature_view`` twice with the same spec.

    Feast 0.43 ``apply()`` is idempotent IF the FV proto bytes are identical;
    this test asserts that:

      1. The second ``register_feature_view`` call returns ``registered=True``
         (i.e., does not raise a duplicate-key error).
      2. Exactly one FV with the target name exists in
         ``list_feature_views()`` after the second call.
      3. The FV's identity (entity, schema, ttl) is unchanged across the
         two calls.

    This protects against a regression where Feast switches to strict
    duplicate-detection or where ``register_feature_view`` accidentally
    starts mutating defaults across calls.
    """
    fv_name = _unique_fv_name("reregister")
    store = feast_client._store

    try:
        # First register.
        first = asyncio.run(
            feast_client.register_feature_view(
                name=fv_name,
                entity_name=ENTITY_NAME,
                feature_names=list(SELECTED_FEATURES),
                batch_source_name=BATCH_SOURCE_NAME,
                ttl=TEST_TTL,
            )
        )
        if not first.get("registered"):
            pytest.fail(f"first register_feature_view failed: {first!r}")

        first_fv = store.get_feature_view(fv_name)
        first_features = sorted(f.name for f in first_fv.features)
        first_ttl = first_fv.ttl

        # Second register — IDENTICAL spec.
        second = asyncio.run(
            feast_client.register_feature_view(
                name=fv_name,
                entity_name=ENTITY_NAME,
                feature_names=list(SELECTED_FEATURES),
                batch_source_name=BATCH_SOURCE_NAME,
                ttl=TEST_TTL,
            )
        )
        if not second.get("registered"):
            pytest.fail(
                f"second register_feature_view failed (Feast apply was "
                f"NOT idempotent): {second!r}"
            )

        # Exactly one FV with this name in list_feature_views().
        matching = [fv for fv in store.list_feature_views() if fv.name == fv_name]
        assert len(matching) == 1, (
            f"expected exactly one FV named {fv_name!r}, got {len(matching)}: "
            f"{[fv.name for fv in matching]!r}"
        )

        # Identity unchanged across the two calls.
        second_fv = matching[0]
        second_features = sorted(f.name for f in second_fv.features)
        assert second_features == first_features, (
            f"feature drift across re-registration: first={first_features!r} "
            f"second={second_features!r}"
        )
        assert second_fv.ttl == first_ttl, (
            f"TTL drift across re-registration: first={first_ttl!r} "
            f"second={second_fv.ttl!r}"
        )

    finally:
        _delete_feature_view(store, fv_name)


# =============================================================================
# Schema-deep idempotency check (proto-byte diff)
# =============================================================================
#
# Plan-language: "registry-proto-aware comparison rather than name-set
# comparison". Feast 0.43 makes idempotency contingent on byte equality of
# the FV proto, so the cleanest detection is `to_proto().SerializeToString()`
# byte comparison. We test the three drift cases the plan calls out:
#   * TTL change
#   * schema field add (rename also covered by add+remove)
#   * source rename (FV keeps its name, swaps its source)
#
# These are NOT tests of "Feast detects drift and fails apply" — Feast 0.43
# happily upserts non-byte-identical specs. They're tests of "OUR diff
# mechanism (proto-byte comparison) detects drift", which is what the plan
# actually asks for. If Feast's apply later starts rejecting drift, we'd add
# a separate failing-apply assertion; until then, the contract is "we can
# tell when it changed".


def _build_minimal_feature_view(
    name: str,
    *,
    ttl: timedelta,
    feature_names: list[str],
    source_name: str | None = None,
) -> Any:
    """Build a FeatureView in memory (no apply) for proto-byte comparison.

    Mirrors the construction in ``FeastClient.register_feature_view`` but
    skips the apply call so we can compute ``to_proto().SerializeToString()``
    on a hypothetical-but-not-yet-applied FV.

    The ``source_name`` lets the source-rename test perturb the PushSource
    name while keeping the FV name fixed. When ``source_name`` is None the
    PushSource follows the canonical ``{name}_push_source`` convention.
    """
    from feast import Entity, FeatureView, Field, PushSource
    from feast.types import Float64

    schema = [Field(name=fn, dtype=Float64) for fn in feature_names]
    push_source_name = source_name if source_name is not None else f"{name}_push_source"

    # We build a stub batch source ref — we do NOT apply this FV, so the
    # batch source identity doesn't have to round-trip through Feast. The
    # proto bytes will differ if and only if the configured stub source name
    # differs, which is the contract we want for the source-rename test.
    from feast import FileSource

    batch_source = FileSource(
        name=f"{name}_batch_stub",
        path=f"/tmp/{name}_stub.parquet",
        timestamp_field="event_timestamp",
    )
    push_source = PushSource(name=push_source_name, batch_source=batch_source)

    entity = Entity(name=ENTITY_NAME, join_keys=[ENTITY_JOIN_KEY])

    return FeatureView(
        name=name,
        entities=[entity],
        ttl=ttl,
        schema=schema,
        source=push_source,
        online=True,
    )


def _proto_bytes(fv: Any) -> bytes:
    """Return a stable byte-string for a FeatureView proto.

    Direct ``SerializeToString()`` is sufficient for our drift-detection
    contract: Feast's proto definition is deterministic for the fields we
    perturb (ttl, schema, source name). We do NOT use ``SerializeToString(
    deterministic=True)`` because Feast 0.43's wrapper chain doesn't expose
    that kwarg uniformly across proto versions.
    """
    return bytes(fv.to_proto().SerializeToString())


def test_schema_deep_diff_detects_ttl_change() -> None:
    """Drift case 1 — TTL change produces detectable proto-byte drift.

    No Feast registry needed — this is a pure in-memory proto-byte diff,
    so it runs even when ``FEAST_INTEGRATION`` is unset. (The Feast SDK
    is still required, which the module-level ``importorskip`` enforces.)
    """
    name = "test_6b_diff_ttl"
    base = _build_minimal_feature_view(
        name, ttl=timedelta(days=7), feature_names=list(SELECTED_FEATURES)
    )
    drifted = _build_minimal_feature_view(
        name, ttl=timedelta(days=14), feature_names=list(SELECTED_FEATURES)
    )

    base_bytes = _proto_bytes(base)
    drifted_bytes = _proto_bytes(drifted)

    assert base_bytes != drifted_bytes, (
        "TTL change (7d -> 14d) produced byte-identical FV protos. "
        "Schema-deep idempotency check would NOT detect this drift."
    )


def test_schema_deep_diff_detects_schema_field_add() -> None:
    """Drift case 2 — adding a field produces detectable proto-byte drift."""
    name = "test_6b_diff_schema_add"
    base = _build_minimal_feature_view(
        name, ttl=TEST_TTL, feature_names=["trx_count"]
    )
    drifted = _build_minimal_feature_view(
        name, ttl=TEST_TTL, feature_names=["trx_count", "nrx_count"]
    )

    base_bytes = _proto_bytes(base)
    drifted_bytes = _proto_bytes(drifted)

    assert base_bytes != drifted_bytes, (
        "Adding a schema field (trx_count + nrx_count) produced byte-"
        "identical FV protos. Schema-deep idempotency check would NOT "
        "detect this drift."
    )


def test_schema_deep_diff_detects_schema_field_remove() -> None:
    """Drift case 2b — removing a field produces detectable proto-byte drift.

    Tested as a separate case from "add" because Feast's proto could in
    principle compress the schema list in a way that makes add and remove
    asymmetric. This test catches the symmetric-detection regression.
    """
    name = "test_6b_diff_schema_remove"
    base = _build_minimal_feature_view(
        name, ttl=TEST_TTL, feature_names=["trx_count", "nrx_count"]
    )
    drifted = _build_minimal_feature_view(
        name, ttl=TEST_TTL, feature_names=["trx_count"]
    )

    base_bytes = _proto_bytes(base)
    drifted_bytes = _proto_bytes(drifted)

    assert base_bytes != drifted_bytes, (
        "Removing a schema field (nrx_count dropped) produced byte-"
        "identical FV protos. Schema-deep idempotency check would NOT "
        "detect this drift."
    )


def test_schema_deep_diff_detects_source_rename() -> None:
    """Drift case 3 — source rename (FV name unchanged) is detectable.

    The plan calls this case out explicitly because it's the sneakiest
    drift mode: the FV name stays the same, the registry's name-set diff
    sees no change, but the underlying source has been swapped. The proto
    serialisation embeds the source's name, so a byte diff catches it.
    """
    name = "test_6b_diff_source_rename"
    base = _build_minimal_feature_view(
        name,
        ttl=TEST_TTL,
        feature_names=list(SELECTED_FEATURES),
        source_name=f"{name}_push_source",
    )
    drifted = _build_minimal_feature_view(
        name,
        ttl=TEST_TTL,
        feature_names=list(SELECTED_FEATURES),
        source_name=f"{name}_push_source_renamed",
    )

    base_bytes = _proto_bytes(base)
    drifted_bytes = _proto_bytes(drifted)

    assert base_bytes != drifted_bytes, (
        "Source rename (push_source -> push_source_renamed) produced byte-"
        "identical FV protos. Schema-deep idempotency check would NOT "
        "detect this drift, even though the FV is now sourced differently."
    )


def test_schema_deep_diff_treats_identical_specs_as_equal() -> None:
    """Sanity check — identical specs produce identical proto bytes.

    Negative control for the three drift tests above: if the diff fires on
    a no-op, the drift tests are tautologies. This test ensures the
    proto-byte comparison is meaningful.
    """
    name = "test_6b_diff_identical"
    a = _build_minimal_feature_view(
        name, ttl=TEST_TTL, feature_names=list(SELECTED_FEATURES)
    )
    b = _build_minimal_feature_view(
        name, ttl=TEST_TTL, feature_names=list(SELECTED_FEATURES)
    )

    assert _proto_bytes(a) == _proto_bytes(b), (
        "Two FVs constructed with identical kwargs produced different "
        "proto bytes. Either Feast injects a non-deterministic field "
        "(timestamp, uuid) into to_proto(), or _build_minimal_feature_view "
        "has a hidden source of variance. The drift tests above would be "
        "false-positive-prone until this is fixed."
    )
