"""Block 3B: assert offline/online parity for each FeatureView.

For a sampled set of entity rows at timestamp ``T``, the values returned by
``FeatureStore.get_online_features(entity_rows)`` must match those returned by
``FeatureStore.get_historical_features(entity_df_at_T)`` within a numeric
tolerance. Mismatches mean the materialization step lost or rewrote data
between the offline (Postgres) and online (Redis) stores.

Skip behaviour
--------------
This test is intentionally hard to run in CI:

* Requires the ``feast`` Python SDK (skip via ``pytest.importorskip``).
* Requires a reachable Feast deployment with both stores populated. We
  detect this via the ``FEAST_INTEGRATION`` env var (set to ``"1"`` on the
  droplet) and skip cleanly otherwise.
* Sample sizes are kept small (5 entities per FV) so that — when the
  environment IS available — the test stays cheap.

Fail-vs-skip discipline
-----------------------
Once a developer has opted in via ``FEAST_INTEGRATION=1``, this module
distinguishes two failure modes when probing the offline store:

* **Environment / connection problems** (missing credentials, host
  unreachable, password rotated, no entities in source table) -> ``skip``.
  These are caller-environment issues; opting in does not assert the DB is
  up.
* **Schema / permission problems** (table missing, column missing,
  ``permission denied``, malformed SQL) -> ``fail``. We connected, the
  wiring is wrong, and that is exactly what this parity test is supposed
  to catch. A bare ``except Exception`` here would silently turn real
  parity violations into green skips on the droplet — that bug is the
  motivation for this discipline (see review item I-3).

Coverage extension (Phase 3 Task 3.2 / shard #4)
------------------------------------------------
Probes all 9 registered FeatureViews (originally 3) and adds a clock-skew
test. Key wiring notes:

* ``join_keys`` is a tuple to support multi-entity FVs (e.g. ``hcp_conversion``
  joins on both ``hcp_id`` and ``hcp_brand_id``).
* In Feast 0.43, ``fv.entities`` is a list of name *strings*
  (``feast/feature_view.py:38``). The original ``getattr(entity, "join_key",
  None)`` fallback was a dead path that silently filtered out every FV.
  We resolve via the sibling ``ENTITY_MAP``.
* Entity-id sampling uses ``PostgreSQLSource.get_table_query_string()`` as
  a subquery (the Feast source *name* is not a Postgres table).
* ``test_parity_with_clock_skew`` simulates a 5-minute offset between the
  offline join timestamp and the online lookup wall-clock; asserts both
  parity AND TTL non-violation (online cells are non-null after skew).

Findings reference: Block 3B (#4 residual, parity invariant; I-3 fix);
Phase 3 Task 3.2 (shard #4: 9/9 coverage + clock-skew).
"""

from __future__ import annotations

import math
import sys as _sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.integration._feast_helpers import feast_integration_available

# Skip the entire module if the Feast Python SDK is not importable.
pytest.importorskip("feast", reason="Feast SDK not installed; skipping parity tests.")

# Optional dependency for entity-row construction.
pd = pytest.importorskip("pandas", reason="pandas required for parity entity dataframes.")

# Feast historical-feature retrieval is heavy (Pydantic + dask). The project's
# default 30s pytest timeout is too tight for module-scoped registry loading
# plus per-test offline/online round-trips.
pytestmark = pytest.mark.timeout(180)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = PROJECT_ROOT / "feature_repo"

# Numeric tolerance for offline-vs-online comparison. Online lookups round-trip
# through Redis (string-typed); we accept tiny float drift but flag anything
# meaningful as a parity violation.
PARITY_RTOL = 1e-6
PARITY_ATOL = 1e-9

# Maximum entities per FeatureView to sample. Kept small so the test stays
# cheap when the environment IS available — see module docstring.
SAMPLE_SIZE = 5

# Clock-skew offset between the offline join timestamp and the simulated
# online lookup wall-clock for ``test_parity_with_clock_skew``. 5 minutes
# is well within every FV's TTL (smallest is 1 day) so values must still
# be served — the test asserts both parity AND non-missingness.
CLOCK_SKEW_OFFSET = timedelta(minutes=5)

# Canonical single-entity FV used by the clock-skew probe. Picked because
# its TTL (30 days) is the loosest among single-entity FVs, leaving zero
# chance the 5-minute skew alone trips an unrelated TTL miss and masks
# the parity assertion. If this FV is ever removed or renamed, swap to
# ``territory_performance_features`` (TTL 1 day, still >> 5 min).
CLOCK_SKEW_FV_KEY = "hcp_profile"


def _build_feature_view_probes() -> list[tuple[str, tuple[str, ...], str]]:
    """Derive ``[(fv_name, join_keys, source_subquery), ...]`` for every FV.

    * ``join_keys`` — tuple of entity join_key column names, in FV-entity
      order. Resolved via ``ENTITY_MAP`` because ``fv.entities`` is a
      list of name strings in Feast 0.43, not Entity objects (see module
      docstring for the dead-code-path detail).
    * ``source_subquery`` — ``PostgreSQLSource.get_table_query_string()``;
      a parenthesised subquery suitable for ``FROM (...) AS s``.

    Returns ``[]`` if the feature_repo can't be imported — pytest surfaces
    that at parametrize time as a clean skip.
    """
    feature_repo_path = str(FEATURE_REPO)
    if feature_repo_path not in _sys.path:
        _sys.path.insert(0, feature_repo_path)
    try:
        from entities import ENTITY_MAP
        from features import FEATURE_VIEW_MAP
    except ImportError:
        return []

    probes: list[tuple[str, tuple[str, ...], str]] = []
    for _key, fv in FEATURE_VIEW_MAP.items():
        # Resolve every entity *name* on the FV through ENTITY_MAP to
        # recover the join_key column. If any entity isn't in ENTITY_MAP
        # we skip the FV (rather than emitting a malformed probe) — that
        # state would indicate a feature_repo wiring bug worth catching
        # upstream, not silently masking here.
        join_keys: list[str] = []
        all_resolved = True
        for entity_ref in fv.entities:
            if isinstance(entity_ref, str):
                ent_name: str | None = entity_ref
            else:
                ent_name = getattr(entity_ref, "name", None)
            if not ent_name or ent_name not in ENTITY_MAP:
                all_resolved = False
                break
            join_keys.append(ENTITY_MAP[ent_name].join_key)
        if not all_resolved or not join_keys:
            continue

        # Pull the wrapped SQL subquery from the PostgreSQLSource. The
        # `get_table_query_string()` method returns a parenthesised
        # ``( SELECT ... FROM <table> WHERE ... )`` string ready to splice
        # into ``FROM <here> AS s``. If the source isn't a SQL source
        # (e.g. some future PushSource without a batch query) the call
        # raises and we skip — that's a structural reason the FV doesn't
        # fit the SELECT-DISTINCT probe shape.
        try:
            source_subquery = fv.source.get_table_query_string()
        except Exception:  # noqa: BLE001 — any failure means the source isn't probable this way
            continue
        if not source_subquery:
            continue

        probes.append((fv.name, tuple(join_keys), str(source_subquery)))
    return probes


FEATURE_VIEW_PROBES: list[tuple[str, tuple[str, ...], str]] = _build_feature_view_probes()


@pytest.fixture(scope="module")
def feature_store() -> Any:
    """Construct a Feast ``FeatureStore`` rooted at ``feature_repo/``.

    Skips the test if the registry can't be loaded or the env var hasn't been
    set. The fixture is module-scoped so we pay the registry-load cost once.
    """
    if not feast_integration_available():
        pytest.skip(
            "FEAST_INTEGRATION not set; skipping offline/online parity. "
            "Set FEAST_INTEGRATION=1 on a host with reachable Feast stores."
        )

    from feast import FeatureStore

    if not FEATURE_REPO.exists():
        pytest.skip(f"feature_repo not found at {FEATURE_REPO}")

    try:
        return FeatureStore(repo_path=str(FEATURE_REPO))
    except Exception as exc:  # noqa: BLE001 — we want to skip on ANY init failure
        pytest.skip(f"FeatureStore init failed: {exc!s:.200}")


def _sample_entity_tuples(
    store: Any,
    source_subquery: str,
    join_keys: tuple[str, ...],
) -> list[tuple[Any, ...]]:
    """Return up to ``SAMPLE_SIZE`` distinct join-key tuples from the offline source.

    Sampling DISTINCT *tuples* (not each key independently) is required for
    multi-entity FVs: ``(hcp_id, hcp_brand_id)`` pairs must coexist in the
    same row, otherwise the online lookup asks for a composite that never
    appears together in the offline store.

    Fail-vs-skip discipline (Block 3B I-3 fix): connection / config issues
    return ``[]`` (caller skips); schema / permission errors surface as
    ``pytest.fail`` so a real parity-violation never green-skips on the droplet.
    """
    sqlalchemy = pytest.importorskip(
        "sqlalchemy",
        reason="sqlalchemy required to probe the Feast offline store.",
    )

    cfg = store.config.offline_store
    # PostgreSQL DSN. The offline-store config exposes host/port/database/
    # user/password/sslmode; if any are missing we treat that as an env config
    # problem (skip), not a parity-test failure.
    try:
        url = sqlalchemy.URL.create(
            drivername="postgresql+psycopg2",
            username=getattr(cfg, "user", None),
            password=getattr(cfg, "password", None) or None,
            host=getattr(cfg, "host", None),
            port=getattr(cfg, "port", None),
            database=getattr(cfg, "database", None),
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        # Bad/missing offline-store config fields -> environment issue, skip.
        return []

    # Safe to interpolate join_keys: they come from ENTITY_MAP (in-repo
    # source-controlled), not user input. The source_subquery is also
    # in-repo (from feature_repo/data_sources.py via PostgreSQLSource).
    select_cols = ", ".join(join_keys)
    not_null_clause = " AND ".join(f"{k} IS NOT NULL" for k in join_keys)
    sql = (
        f"SELECT DISTINCT {select_cols} "
        f"FROM {source_subquery} AS s "
        f"WHERE {not_null_clause} "
        f"LIMIT :n"
    )

    engine = None
    try:
        engine = sqlalchemy.create_engine(
            url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
        )
        with engine.connect() as conn:
            rows = conn.execute(sqlalchemy.text(sql), {"n": SAMPLE_SIZE}).fetchall()
    except sqlalchemy.exc.OperationalError:
        # DB unreachable / connection refused / auth failed -> env issue, skip.
        return []
    except sqlalchemy.exc.ProgrammingError as exc:
        # Connected, but the table/column/permissions are wrong. This is a
        # real parity-test failure: the caller opted in via FEAST_INTEGRATION
        # but the offline store doesn't match the FeatureView definitions.
        pytest.fail(
            f"Parity test schema mismatch on {join_keys!r} "
            f"(SQL programming error from a reachable DB): {exc!s:.500}"
        )
    except sqlalchemy.exc.DatabaseError as exc:
        # Reachable DB returned a non-programming database error (data error,
        # internal error, etc.) -> still a real failure when opted in.
        pytest.fail(
            f"Parity test database error on {join_keys!r} "
            f"(reachable DB rejected the probe): {exc!s:.500}"
        )
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:  # noqa: BLE001 — dispose() is best-effort cleanup
                pass

    return [tuple(str(v) for v in row) for row in rows]


def _build_entity_df(
    join_keys: tuple[str, ...],
    id_tuples: list[tuple[Any, ...]],
    event_timestamp: datetime,
) -> Any:
    """Build the offline-join entity dataframe from sampled id tuples."""
    df_data: dict[str, list[Any]] = {k: [] for k in join_keys}
    for t in id_tuples:
        for k, v in zip(join_keys, t, strict=True):
            df_data[k].append(v)
    df_data["event_timestamp"] = [event_timestamp] * len(id_tuples)
    return pd.DataFrame(df_data)


def _build_entity_rows(
    join_keys: tuple[str, ...],
    id_tuples: list[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    """Build the online-lookup entity_rows from sampled id tuples."""
    return [dict(zip(join_keys, t, strict=True)) for t in id_tuples]


def _compose_entity_id(id_tuple: tuple[Any, ...]) -> str:
    """Render an id-tuple as a stable string key for diff messages."""
    return "|".join(str(v) for v in id_tuple)


def _compare_offline_online(
    fv_name: str,
    join_keys: tuple[str, ...],
    feature_cols: list[str],
    offline_df: Any,
    online_df: Any,
) -> list[str]:
    """Return mismatch strings between offline_df and online_df, indexed by join_keys."""
    index_arg = join_keys[0] if len(join_keys) == 1 else list(join_keys)
    offline_indexed = offline_df.set_index(index_arg)
    online_indexed = online_df.set_index(index_arg)
    common_ids = sorted(set(offline_indexed.index) & set(online_indexed.index))

    if not common_ids:
        return [
            f"NO-OVERLAP: offline ids={list(offline_indexed.index)!r} "
            f"online ids={list(online_indexed.index)!r}"
        ]

    mismatches: list[str] = []
    for eid in common_ids:
        eid_str = _compose_entity_id(eid if isinstance(eid, tuple) else (eid,))
        for col in feature_cols:
            if col not in online_indexed.columns:
                mismatches.append(f"{eid_str}/{col}: missing in online store")
                continue
            off_val = offline_indexed.at[eid, col]
            on_val = online_indexed.at[eid, col]

            # Numeric path: tolerant compare; everything else: equality.
            if isinstance(off_val, (int, float)) and isinstance(on_val, (int, float)):
                if not _floats_close(float(off_val), float(on_val)):
                    mismatches.append(
                        f"{eid_str}/{col}: offline={off_val!r} online={on_val!r} "
                        f"(rtol={PARITY_RTOL}, atol={PARITY_ATOL})"
                    )
            elif off_val != on_val:
                mismatches.append(f"{eid_str}/{col}: offline={off_val!r} online={on_val!r}")
    return mismatches


@pytest.mark.parametrize("fv_name,join_keys,source_subquery", FEATURE_VIEW_PROBES)
def test_offline_online_parity_per_feature_view(
    feature_store: Any,
    fv_name: str,
    join_keys: tuple[str, ...],
    source_subquery: str,
) -> None:
    """Online and offline lookups must agree for the same (entity, T) pair.

    Strategy:
      1. Sample up to ``SAMPLE_SIZE`` distinct entity-id tuples from the
         offline source query.
      2. Build an entity dataframe with those tuples and a single timestamp T.
      3. Call ``get_historical_features(entity_df, [fv:*])`` -> ``offline_df``.
      4. Call ``get_online_features(features=[fv:*], entity_rows=…)``
         -> ``online_dict``.
      5. For each entity, for each numeric feature, assert values match
         within ``(PARITY_RTOL, PARITY_ATOL)``. For string/categorical
         features, assert equality.
    """
    id_tuples = _sample_entity_tuples(feature_store, source_subquery, join_keys)
    if not id_tuples:
        pytest.skip(
            f"No entities available for {fv_name} via join_keys={join_keys!r}. "
            f"Probe failed or table is empty."
        )

    now = datetime.now(timezone.utc)
    entity_df = _build_entity_df(join_keys, id_tuples, now)

    # Build explicit feature refs (Feast 0.43.0 does not expand ":*" wildcards;
    # the registry projection raises KeyError: 'Feature * not found'). Pull the
    # full feature list from the FeatureView definition instead.
    fv_obj = feature_store.get_feature_view(fv_name)
    feature_refs = [f"{fv_name}:{f.name}" for f in fv_obj.features]

    # ---- Offline retrieval ------------------------------------------------
    offline_df = feature_store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()
    assert not offline_df.empty, f"Offline retrieval returned no rows for {fv_name}"

    # ---- Online retrieval -------------------------------------------------
    entity_rows = _build_entity_rows(join_keys, id_tuples)
    online_dict = feature_store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    ).to_dict()
    online_df = pd.DataFrame(online_dict)
    assert not online_df.empty, f"Online retrieval returned no rows for {fv_name}"

    # ---- Compare ----------------------------------------------------------
    feature_cols = [
        c for c in offline_df.columns if c not in set(join_keys) | {"event_timestamp"}
    ]
    assert feature_cols, f"No comparable feature columns found for {fv_name}"

    mismatches = _compare_offline_online(fv_name, join_keys, feature_cols, offline_df, online_df)
    assert not mismatches, (
        f"Offline/online parity violations in {fv_name} "
        f"({len(mismatches)} mismatch(es) of {len(id_tuples) * len(feature_cols)} cells):\n"
        + "\n".join(f"  - {m}" for m in mismatches[:20])
        + ("\n  …" if len(mismatches) > 20 else "")
    )


def test_parity_with_clock_skew(feature_store: Any) -> None:
    """Offline join at T must match online lookup at T + 5min within tolerance.

    Phase 3 Task 3.2 deliverable: simulate training-serving clock skew and
    verify (a) feature parity still holds within ``(PARITY_RTOL, PARITY_ATOL)``,
    and (b) the FV TTL is not exceeded — online cells are populated, not
    None / NaN. The vanilla parity test cannot catch silent TTL expiration
    because it never queries the online store at a different wall-clock.

    Uses ``unittest.mock.patch`` (not ``freezegun``) because adding deps is
    out of scope for shard #4. Patches ``feast.feature_store.datetime`` for
    the duration of the online call only.
    """
    probe = next(
        (p for p in FEATURE_VIEW_PROBES if p[0].startswith(CLOCK_SKEW_FV_KEY)),
        None,
    )
    if probe is None:
        pytest.skip(
            f"Clock-skew test FV '{CLOCK_SKEW_FV_KEY}' not in FEATURE_VIEW_PROBES; "
            "the canonical FV may have been removed/renamed — pick a replacement "
            "with TTL >> 5 min."
        )
    fv_name, join_keys, source_subquery = probe

    id_tuples = _sample_entity_tuples(feature_store, source_subquery, join_keys)
    if not id_tuples:
        pytest.skip(
            f"No entities available for clock-skew probe on {fv_name} "
            f"(join_keys={join_keys!r}). Probe failed or table is empty."
        )

    # Anchor the offline join at a fixed T (capture once, share with the
    # online lookup so the skew is measured against the same baseline).
    t_offline = datetime.now(timezone.utc)
    entity_df = _build_entity_df(join_keys, id_tuples, t_offline)

    fv_obj = feature_store.get_feature_view(fv_name)
    feature_refs = [f"{fv_name}:{f.name}" for f in fv_obj.features]

    # ---- Offline retrieval at T --------------------------------------------
    offline_df = feature_store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()
    assert not offline_df.empty, (
        f"Clock-skew test: offline retrieval returned no rows for {fv_name} at T={t_offline!r}"
    )

    # ---- Online retrieval at T + 5min --------------------------------------
    # Patch ``feast.feature_store.datetime`` so the online lookup observes
    # wall-clock = T + 5min while we keep the offline anchor at T.
    skewed_now = t_offline + CLOCK_SKEW_OFFSET

    class _SkewedDatetime(datetime):
        """Subclass of datetime that returns ``skewed_now`` for ``now()``."""

        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return skewed_now if tz is None else skewed_now.astimezone(tz)

        @classmethod
        def utcnow(cls) -> datetime:  # type: ignore[override]
            return skewed_now.replace(tzinfo=None)

    entity_rows = _build_entity_rows(join_keys, id_tuples)
    with patch("feast.feature_store.datetime", _SkewedDatetime):
        online_dict = feature_store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()
    online_df = pd.DataFrame(online_dict)
    assert not online_df.empty, (
        f"Clock-skew test: online retrieval returned no rows for {fv_name} "
        f"at skewed wall-clock={skewed_now!r} (TTL miss?)"
    )

    # ---- TTL assertion: every cell must be populated under the skew --------
    feature_cols = [
        c for c in offline_df.columns if c not in set(join_keys) | {"event_timestamp"}
    ]
    online_indexed = (
        online_df.set_index(join_keys[0])
        if len(join_keys) == 1
        else online_df.set_index(list(join_keys))
    )
    ttl_misses: list[str] = []
    for id_tuple in id_tuples:
        eid = id_tuple[0] if len(join_keys) == 1 else id_tuple
        if eid not in online_indexed.index:
            ttl_misses.append(f"{_compose_entity_id(id_tuple)}: entire row missing online")
            continue
        for col in feature_cols:
            if col not in online_indexed.columns:
                continue  # absent column is a parity issue, handled below
            val = online_indexed.at[eid, col]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                ttl_misses.append(
                    f"{_compose_entity_id(id_tuple)}/{col}: online value is "
                    f"{val!r} (TTL exceeded under {CLOCK_SKEW_OFFSET} skew?)"
                )
    assert not ttl_misses, (
        f"Clock-skew TTL violations in {fv_name} under {CLOCK_SKEW_OFFSET} skew "
        f"(offline T={t_offline!r}, online T={skewed_now!r}; "
        f"{len(ttl_misses)} miss(es)):\n"
        + "\n".join(f"  - {m}" for m in ttl_misses[:20])
        + ("\n  …" if len(ttl_misses) > 20 else "")
    )

    # ---- Parity assertion: values still agree within tolerance -------------
    mismatches = _compare_offline_online(fv_name, join_keys, feature_cols, offline_df, online_df)
    assert not mismatches, (
        f"Clock-skew parity violations in {fv_name} under {CLOCK_SKEW_OFFSET} skew "
        f"(offline T={t_offline!r}, online T={skewed_now!r}; "
        f"{len(mismatches)} mismatch(es) of {len(id_tuples) * len(feature_cols)} cells):\n"
        + "\n".join(f"  - {m}" for m in mismatches[:20])
        + ("\n  …" if len(mismatches) > 20 else "")
    )


def _floats_close(a: float, b: float) -> bool:
    """math.isclose with the module-level tolerance defaults."""
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=PARITY_RTOL, abs_tol=PARITY_ATOL)
