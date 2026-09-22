"""#2207 follow-up (owner decision 2026-09-22): the per-model cohort contract of record
lives on ``ml_model_registry`` (migration 150) and ``src/services/cohort_contract.py`` is
the one place that encodes / decodes / loads / heals it.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from src.services.cohort_contract import (
    REGISTRY_CONTRACT_COLUMNS,
    contract_from_registry_row,
    contract_from_training_config,
    decode_data_source,
    encode_data_source,
    heal_registry_cohort_contract,
    load_registry_cohort_contract,
    merge_contracts,
)
from tests.unit._fakes.async_supabase import FakeAsyncSupabase


@pytest.mark.unit
def test_columns_are_the_migration_150_columns():
    assert REGISTRY_CONTRACT_COLUMNS == (
        "cohort_data_source",
        "cohort_target_outcome",
        "cohort_feature_manifest_source",
    )


@pytest.mark.unit
def test_data_source_round_trips_table_names_and_file_dicts():
    assert encode_data_source("patient_journeys") == "patient_journeys"
    assert decode_data_source("patient_journeys") == "patient_journeys"
    files = {"type": "file_dir", "path": "data/rwd/optum/initiation"}
    encoded = encode_data_source(files)
    assert encoded == json.dumps(files, sort_keys=True)
    assert decode_data_source(encoded) == files
    assert encode_data_source(None) is None
    assert decode_data_source(None) is None
    assert decode_data_source("") is None
    assert decode_data_source("{not json") == "{not json"  # never raises; kept verbatim


@pytest.mark.unit
def test_contract_from_registry_row_is_none_free_and_decodes():
    row = {
        "id": "x",
        "cohort_data_source": json.dumps({"type": "files", "paths": {"train": "a"}}),
        "cohort_target_outcome": "initiated_biologic_180d",
        "cohort_feature_manifest_source": None,
    }
    assert contract_from_registry_row(row) == {
        "data_source": {"type": "files", "paths": {"train": "a"}},
        "target_outcome": "initiated_biologic_180d",
    }
    assert contract_from_registry_row({"id": "x"}) == {}
    assert contract_from_registry_row(None) == {}


@pytest.mark.unit
def test_merge_explicit_wins_over_fallback():
    explicit = {"data_source": "req_table", "brand": "Kisqali"}
    fallback = {"data_source": "row_table", "target_outcome": "y", "feature_manifest_source": "csu"}
    assert merge_contracts(explicit, fallback) == {
        "data_source": "req_table",
        "target_outcome": "y",
        "feature_manifest_source": "csu",
        "brand": "Kisqali",
    }
    assert merge_contracts(None, fallback) == fallback
    assert merge_contracts({"data_source": None}, fallback)["data_source"] == "row_table"
    assert merge_contracts(None, None) == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_resolves_uuid_version_and_name_handles():
    rid = str(uuid4())
    db = FakeAsyncSupabase(
        {
            "ml_model_registry": [
                {
                    "id": rid,
                    "model_name": "initiation_kisqali_goldstd_lr_v1",
                    "model_version": "1.0",
                    "cohort_data_source": "patient_journeys",
                    "cohort_target_outcome": "treatment_initiated",
                    "cohort_feature_manifest_source": None,
                }
            ]
        }
    )
    for handle in (rid, "initiation_kisqali_goldstd_lr_v1"):
        model_id, contract = await load_registry_cohort_contract(db, handle)
        assert model_id == rid
        assert contract == {
            "data_source": "patient_journeys",
            "target_outcome": "treatment_initiated",
        }
    model_id, contract = await load_registry_cohort_contract(db, "not_a_model")
    assert model_id is None and contract == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_survives_a_pre_migration_schema():
    """Before migration 150 lands the select 42703s; the trigger must still work."""
    rid = str(uuid4())

    class _NoColumns(FakeAsyncSupabase):
        def table(self, name):
            q = super().table(name)
            real_execute = q.execute

            async def _execute():
                if any(c in str(q._filters) for c in ("cohort_",)):
                    raise RuntimeError("column does not exist (42703)")
                return await real_execute()

            q.execute = _execute  # type: ignore[method-assign]
            return q

    db = _NoColumns({"ml_model_registry": [{"id": rid, "model_version": "v1"}]})
    model_id, contract = await load_registry_cohort_contract(db, rid)
    assert model_id == rid and contract == {}


@pytest.mark.unit
def test_contract_from_training_config_is_none_free_and_drops_non_contract_keys():
    tc = {"data_source": "t", "target_outcome": "y", "feature_manifest_source": None, "notes": "n"}
    assert contract_from_training_config(tc) == {"data_source": "t", "target_outcome": "y"}
    assert contract_from_training_config(None) == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_heal_fills_only_null_columns_and_never_overwrites():
    rid = str(uuid4())
    db = FakeAsyncSupabase(
        {
            "ml_model_registry": [
                {
                    "id": rid,
                    "cohort_data_source": None,
                    "cohort_target_outcome": "initiated_biologic_180d",
                    "cohort_feature_manifest_source": None,
                }
            ]
        }
    )
    contract = {
        "data_source": {"type": "file_dir", "path": "data/rwd/optum/initiation"},
        "target_outcome": "initiated_biologic_180d",  # consistent with the row
        "feature_manifest_source": "optum",
        "brand": "Kisqali",  # not a registry column
    }
    written = await heal_registry_cohort_contract(db, rid, contract)
    assert written == {
        "cohort_data_source": json.dumps(contract["data_source"], sort_keys=True),
        "cohort_feature_manifest_source": "optum",
    }
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_target_outcome"] == "initiated_biologic_180d"
    # second call: nothing left NULL -> nothing written
    assert await heal_registry_cohort_contract(db, rid, contract) == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_heal_refuses_to_compose_a_mixed_pair_on_any_conflict():
    """codex r1 HIGH-2: row {target=initiation_kisqali} + contract {source=patient_journeys,
    target=treatment_initiated} must NOT become {patient_journeys, initiation_kisqali}."""
    rid = str(uuid4())
    db = FakeAsyncSupabase(
        {
            "ml_model_registry": [
                {
                    "id": rid,
                    "cohort_data_source": None,
                    "cohort_target_outcome": "initiation_kisqali",
                    "cohort_feature_manifest_source": None,
                }
            ]
        }
    )
    written = await heal_registry_cohort_contract(
        db, rid, {"data_source": "patient_journeys", "target_outcome": "treatment_initiated"}
    )
    assert written == {}
    (row,) = db.rows("ml_model_registry")
    assert (
        row["cohort_data_source"] is None and row["cohort_target_outcome"] == "initiation_kisqali"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_heal_is_a_compare_and_set_per_column():
    """A concurrent healer that filled the column first wins; ours writes nothing."""
    rid = str(uuid4())

    class _RacingSupabase(FakeAsyncSupabase):
        def table(self, name):
            q = super().table(name)
            real_execute = q.execute

            async def _execute():
                if q._op == "update":
                    # someone else filled it between our read and our write
                    for r in self.store["ml_model_registry"]:
                        r["cohort_data_source"] = r["cohort_data_source"] or "theirs"
                return await real_execute()

            q.execute = _execute  # type: ignore[method-assign]
            return q

    db = _RacingSupabase(
        {
            "ml_model_registry": [
                {
                    "id": rid,
                    "cohort_data_source": None,
                    "cohort_target_outcome": None,
                    "cohort_feature_manifest_source": None,
                }
            ]
        }
    )
    written = await heal_registry_cohort_contract(db, rid, {"data_source": "ours"})
    assert written == {}
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] == "theirs"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_heal_is_a_noop_without_a_row_or_a_client():
    assert await heal_registry_cohort_contract(None, "x", {"data_source": "t"}) == {}
    db = FakeAsyncSupabase({"ml_model_registry": []})
    assert await heal_registry_cohort_contract(db, "missing", {"data_source": "t"}) == {}
