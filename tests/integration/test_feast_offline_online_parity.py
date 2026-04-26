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

Findings reference: Block 3B (#4 residual, parity invariant).
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

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

# (FeatureView, join_key, source_table) — the source table name lets us probe
# the offline store directly for real entity ids (instead of guessing) before
# we round-trip through Feast. Tables come from feature_repo/data_sources.py.
FEATURE_VIEW_PROBES: list[tuple[str, str, str]] = [
    ("hcp_profile_features", "hcp_id", "hcp_profiles"),
    ("territory_performance_features", "territory_id", "territory_metrics"),
    ("trigger_response_features", "trigger_id", "triggers"),
]


def _feast_integration_available() -> bool:
    """True iff the caller has opted into the live Feast parity check.

    The droplet (and only the droplet) sets ``FEAST_INTEGRATION=1`` in its
    environment so this test runs there but stays a no-op everywhere else.
    """
    return os.environ.get("FEAST_INTEGRATION", "").strip().lower() in {"1", "true", "yes"}


@pytest.fixture(scope="module")
def feature_store() -> Any:
    """Construct a Feast ``FeatureStore`` rooted at ``feature_repo/``.

    Skips the test if the registry can't be loaded or the env var hasn't been
    set. The fixture is module-scoped so we pay the registry-load cost once.
    """
    if not _feast_integration_available():
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

    Returns an empty list if the probe fails for any reason (unreachable DB,
    missing table, permission denied, etc.); the caller decides whether to
    skip or to fail.
    """
    sqlalchemy = pytest.importorskip(
        "sqlalchemy",
        reason="sqlalchemy required to probe the Feast offline store.",
    )

    cfg = store.config.offline_store
    # PostgreSQL DSN. The offline-store config exposes host/port/database/
    # user/password/sslmode; if any are missing we let the connection error
    # surface and skip.
    try:
        url = sqlalchemy.URL.create(
            drivername="postgresql+psycopg2",
            username=getattr(cfg, "user", None),
            password=getattr(cfg, "password", None) or None,
            host=getattr(cfg, "host", None),
            port=getattr(cfg, "port", None),
            database=getattr(cfg, "database", None),
        )
    except Exception:
        return []

    try:
        engine = sqlalchemy.create_engine(
            url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
        )
        with engine.connect() as conn:
            rows = conn.execute(
                sqlalchemy.text(f"SELECT DISTINCT {join_key} FROM {table} LIMIT :n"),
                {"n": SAMPLE_SIZE},
            ).fetchall()
    except Exception:
        return []
    finally:
        try:
            engine.dispose()
        except Exception:
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

    # ---- Offline retrieval ------------------------------------------------
    offline_df = feature_store.get_historical_features(
        entity_df=entity_df,
        features=[f"{fv_name}:*"],
    ).to_df()
    assert not offline_df.empty, f"Offline retrieval returned no rows for {fv_name}"

    # ---- Online retrieval -------------------------------------------------
    entity_rows = [{join_key: eid} for eid in ids]
    online_dict = feature_store.get_online_features(
        features=[f"{fv_name}:*"],
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
