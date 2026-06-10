"""Faithful (real-DB, NO mocks) regression test for issue #852 — the feature-store
tables (feature_groups / features / feature_values) loaded 0 rows in the synthetic
load.

ROOT CAUSE (proven against the live docker supabase-db): the generic
``BatchLoader._load_batch`` upserted with NO ``on_conflict``, so PostgREST conflicted
on the PRIMARY KEY (``id``). The seeder mints fresh random UUIDs every run, but the
four canonical group names already exist in the DB with DIFFERENT ids -> the
upsert-by-id INSERTs and collides with ``feature_groups_name_key`` UNIQUE(name) ->
23505 -> whole batch fails -> 0 loaded -> features/feature_values orphaned -> 0/0.

Adding ``on_conflict=name`` alone is INSUFFICIENT: it then tries to rewrite the
existing row's PK to the fresh id, which ``features_feature_group_id_fkey`` rejects
(23503). The fix pairs a natural-key ``on_conflict`` (TABLE_ON_CONFLICT) with a
reconcile pass (``reconcile_feature_store_ids``) that remaps generated ids onto the
existing DB ids, cascading group -> feature -> value, so the load is idempotent AND
FK-coherent.

Opt-in (real docker supabase-db required), skipped in CI by default; run -n0.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_issue_852_feature_store_load_realdb.py -p no:cacheprovider -n0
"""

import os
from datetime import date

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)


def _build_feature_store_datasets(seed: int, n_values: int):
    from src.ml.synthetic.generators import (
        FeatureStoreSeeder,
        FeatureValueGenerator,
        GeneratorConfig,
    )

    cfg = GeneratorConfig(seed=seed, anchor_to_now=True, anchor_reference=date.today())
    groups_df, features_df = FeatureStoreSeeder(cfg).seed()
    fv_cfg = GeneratorConfig(
        seed=seed, n_records=n_values, anchor_to_now=True, anchor_reference=date.today()
    )
    fv_df = FeatureValueGenerator(fv_cfg, features_df=features_df).generate()
    datasets = {
        "feature_groups": groups_df,
        "features": features_df,
        "feature_values": fv_df,
    }
    for df in datasets.values():
        df["is_synthetic"] = True
    return datasets


def test_feature_store_tables_load_above_zero_idempotently():
    """All three feature-store tables must load >0 rows against the live DB, and a
    SECOND load of fresh frames (new UUIDs) must also succeed — proving idempotency
    and FK-coherence (no 23505 / 23503)."""
    from src.ml.synthetic.loaders import BatchLoader, LoaderConfig

    loader = BatchLoader(LoaderConfig(batch_size=200, max_retries=1, verbose=False))
    if loader.client is None:
        pytest.skip("supabase client unavailable (env not configured)")

    # --- First load ---
    ds1 = _build_feature_store_datasets(seed=85201, n_values=300)
    results1 = loader.load_all(ds1)

    for tbl in ("feature_groups", "features", "feature_values"):
        r = results1[tbl]
        assert r.records_loaded > 0, f"{tbl}: 0 loaded (#852 not fixed): {r.errors}"
        assert r.records_failed == 0, f"{tbl}: load failures: {r.errors}"

    assert results1["feature_groups"].records_loaded == 4
    assert results1["features"].records_loaded == 15

    # --- Second load with DIFFERENT fresh UUIDs: must NOT collide (idempotent) ---
    ds2 = _build_feature_store_datasets(seed=85202, n_values=300)
    results2 = loader.load_all(ds2)

    for tbl in ("feature_groups", "features", "feature_values"):
        r = results2[tbl]
        assert r.records_loaded > 0, f"{tbl}: second load 0 loaded: {r.errors}"
        assert r.records_failed == 0, f"{tbl}: second load failures (not idempotent): {r.errors}"


def test_reconcile_preserves_existing_group_ids():
    """The reconcile pass must NOT rewrite an existing group's PK to a fresh id — it
    remaps the generated frame onto the existing id so child FKs stay valid."""
    from src.ml.synthetic.loaders import BatchLoader, LoaderConfig

    loader = BatchLoader(LoaderConfig(max_retries=1))
    if loader.client is None:
        pytest.skip("supabase client unavailable")

    # Read the existing canonical group ids straight from the DB.
    resp = (
        loader.client.table("feature_groups")
        .select("id,name")
        .in_(
            "name", ["hcp_demographics", "patient_features", "brand_performance", "causal_features"]
        )
        .execute()
    )
    existing = {row["name"]: row["id"] for row in (resp.data or [])}
    assert existing, "expected the 4 canonical groups to pre-exist in the DB"

    ds = _build_feature_store_datasets(seed=85203, n_values=10)
    loader.reconcile_feature_store_ids(ds)

    for _, row in ds["feature_groups"].iterrows():
        if row["name"] in existing:
            assert row["id"] == existing[row["name"]], (
                f"{row['name']} id was not reconciled to the existing DB id"
            )
