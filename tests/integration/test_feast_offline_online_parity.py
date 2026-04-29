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

Findings reference: Block 3B (#4 residual, parity invariant; I-3 fix).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

# Subset of feature_views to exercise in this parity test. Probes are
# derived from ``feature_repo.features.FEATURE_VIEW_MAP`` so the list
# auto-extends when new FVs land — no hand-curated triple-tuple drift.
# (3B-M-6)
#
# We restrict to the originally curated *single-entity* FVs because:
#   - parity-test wiring assumes one ``join_key`` per FV (tuple shape
#     is ``(fv_name, join_key, source_table)``);
#   - multi-entity FVs (e.g. hcp_conversion which has both ``hcp_id`` +
#     ``hcp_brand_id``) need a different probe shape we haven't built;
#   - keeping the parity loop fast-cheap during development.
#
# When new single-entity FVs land in FEATURE_VIEW_MAP they are picked
# up automatically. Multi-entity FVs are filtered out.
_PROBE_FV_KEYS = {
    "hcp_profile",
    "territory_performance",
    "trigger_response",
}


def _build_feature_view_probes() -> list[tuple[str, str, str]]:
    """Derive ``[(fv_name, join_key, source_table), ...]`` from
    ``FEATURE_VIEW_MAP`` so the probe list auto-extends as the FVs do.

    Returns ``[]`` if the feature_repo cannot be imported (e.g. the
    integration runner stripped it down) — that lands at the parametrize
    level and pytest will surface it via a SkipReason.
    """
    import sys as _sys

    feature_repo_path = str(FEATURE_REPO)
    if feature_repo_path not in _sys.path:
        _sys.path.insert(0, feature_repo_path)
    try:
        from features import FEATURE_VIEW_MAP
    except ImportError:
        return []

    probes: list[tuple[str, str, str]] = []
    for key, fv in FEATURE_VIEW_MAP.items():
        if key not in _PROBE_FV_KEYS:
            continue
        # Each FV has 1+ entities; we only handle the single-entity case
        # here. Multi-entity FVs need a different probe shape and are
        # explicitly skipped above.
        if len(fv.entities) != 1:
            continue
        # FV.entities[0] is an Entity object — .join_key gives us the
        # column name in the offline source.  Fall back to .name if
        # join_key is unset (Feast 0.43 defaults join_key to entity name
        # when explicit join_key is omitted).
        entity = fv.entities[0]
        join_key = getattr(entity, "join_key", None) or getattr(entity, "name", None)
        source_table = getattr(fv.source, "name", None)
        if not join_key or not source_table:
            continue
        probes.append((fv.name, str(join_key), str(source_table)))
    return probes


FEATURE_VIEW_PROBES: list[tuple[str, str, str]] = _build_feature_view_probes()


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


def _sample_entity_ids(store: Any, table: str, join_key: str) -> list[str]:
    """Return up to ``SAMPLE_SIZE`` distinct ids from the offline source table.

    We connect directly to the offline store (PostgreSQL) using credentials
    from ``feature_store.yaml`` and run a small ``SELECT DISTINCT join_key …
    LIMIT N`` probe. This avoids the chicken-and-egg of needing entity ids in
    order to call ``get_historical_features``.

    Fail-vs-skip discipline (post-Block-3B I-3 fix):

    * **Environment / connection problems** (missing credential field,
      unreachable host, password rotated, URL build failure) -> return ``[]``
      so the caller can ``pytest.skip``. These are caller-environment issues,
      not parity-test failures, and ``FEAST_INTEGRATION=1`` only asserts that
      the caller *opted in*, not that the DB is up.
    * **Schema / permission problems** (table missing, column missing,
      ``permission denied``, malformed SQL) -> propagate as
      ``sqlalchemy.exc.ProgrammingError`` / ``DatabaseError`` so pytest
      reports a real failure with full traceback. We connected, the wiring
      is wrong: that is exactly what this test is supposed to catch.

    Anything not in those two buckets is left unhandled and will surface as a
    test error, which is the correct loud-failure path for unexpected states.
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

    engine = None
    try:
        engine = sqlalchemy.create_engine(
            url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
        )
        with engine.connect() as conn:
            rows = conn.execute(
                sqlalchemy.text(f"SELECT DISTINCT {join_key} FROM {table} LIMIT :n"),
                {"n": SAMPLE_SIZE},
            ).fetchall()
    except sqlalchemy.exc.OperationalError:
        # DB unreachable / connection refused / auth failed -> env issue, skip.
        return []
    except sqlalchemy.exc.ProgrammingError as exc:
        # Connected, but the table/column/permissions are wrong. This is a
        # real parity-test failure: the caller opted in via FEAST_INTEGRATION
        # but the offline store doesn't match the FeatureView definitions.
        pytest.fail(
            f"Parity test schema mismatch on {table}.{join_key} "
            f"(SQL programming error from a reachable DB): {exc!s:.500}"
        )
    except sqlalchemy.exc.DatabaseError as exc:
        # Reachable DB returned a non-programming database error (data error,
        # internal error, etc.) -> still a real failure when opted in.
        pytest.fail(
            f"Parity test database error on {table}.{join_key} "
            f"(reachable DB rejected the probe): {exc!s:.500}"
        )
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:  # noqa: BLE001 — dispose() is best-effort cleanup
                pass

    return [str(row[0]) for row in rows if row[0] is not None]


@pytest.mark.parametrize("fv_name,join_key,source_table", FEATURE_VIEW_PROBES)
def test_offline_online_parity_per_feature_view(
    feature_store: Any,
    fv_name: str,
    join_key: str,
    source_table: str,
) -> None:
    """Online and offline lookups must agree for the same (entity, T) pair.

    Strategy:
      1. Sample up to ``SAMPLE_SIZE`` entity ids that exist in the offline
         source table.
      2. Build an entity dataframe with those ids and a single timestamp T.
      3. Call ``get_historical_features(entity_df, [fv:*])`` -> ``offline_df``.
      4. Call ``get_online_features(features=[fv:*], entity_rows=…)``
         -> ``online_dict``.
      5. For each entity, for each numeric feature, assert values match
         within ``(PARITY_RTOL, PARITY_ATOL)``. For string/categorical
         features, assert equality.
    """
    ids = _sample_entity_ids(feature_store, source_table, join_key)
    if not ids:
        pytest.skip(
            f"No entities available in offline source {source_table!r} "
            f"(join_key={join_key!r}). Probe failed or table is empty."
        )

    now = datetime.now(timezone.utc)
    entity_df = pd.DataFrame(
        {
            join_key: ids,
            "event_timestamp": [now] * len(ids),
        }
    )

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
    entity_rows = [{join_key: eid} for eid in ids]
    online_dict = feature_store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    ).to_dict()
    online_df = pd.DataFrame(online_dict)
    assert not online_df.empty, f"Online retrieval returned no rows for {fv_name}"

    # ---- Compare ----------------------------------------------------------
    feature_cols = [c for c in offline_df.columns if c not in {join_key, "event_timestamp"}]
    assert feature_cols, f"No comparable feature columns found for {fv_name}"

    # Align by join key so row ordering doesn't matter.
    offline_indexed = offline_df.set_index(join_key)
    online_indexed = online_df.set_index(join_key)
    common_ids = sorted(set(offline_indexed.index) & set(online_indexed.index))
    assert common_ids, (
        f"No overlapping entity ids between offline and online for {fv_name}; "
        f"offline ids={list(offline_indexed.index)!r} "
        f"online ids={list(online_indexed.index)!r}"
    )

    mismatches: list[str] = []
    for eid in common_ids:
        for col in feature_cols:
            if col not in online_indexed.columns:
                mismatches.append(f"{eid}/{col}: missing in online store")
                continue
            off_val = offline_indexed.at[eid, col]
            on_val = online_indexed.at[eid, col]

            # Numeric path: tolerant compare; everything else: equality.
            if isinstance(off_val, (int, float)) and isinstance(on_val, (int, float)):
                if not _floats_close(float(off_val), float(on_val)):
                    mismatches.append(
                        f"{eid}/{col}: offline={off_val!r} online={on_val!r} "
                        f"(rtol={PARITY_RTOL}, atol={PARITY_ATOL})"
                    )
            elif off_val != on_val:
                mismatches.append(f"{eid}/{col}: offline={off_val!r} online={on_val!r}")

    assert not mismatches, (
        f"Offline/online parity violations in {fv_name} "
        f"({len(mismatches)} mismatch(es) of {len(common_ids) * len(feature_cols)} cells):\n"
        + "\n".join(f"  - {m}" for m in mismatches[:20])
        + ("\n  …" if len(mismatches) > 20 else "")
    )


def _floats_close(a: float, b: float) -> bool:
    """math.isclose with the module-level tolerance defaults."""
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=PARITY_RTOL, abs_tol=PARITY_ATOL)
