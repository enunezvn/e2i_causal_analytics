"""Unit tests for src.tasks.nppes_tasks (issue #154 PR-0).

Bulk-dump ingestion is exercised via an in-memory CSV fixture; live DB
writes are exercised via a fake cursor that captures executemany() calls.
"""

from __future__ import annotations

import io
import json
from datetime import date

from scripts.rwd_common import NppesRecord, NppesTaxonomy
from src.tasks.nppes_tasks import (
    _record_to_upsert_params,
    _row_to_record,
    _row_tuple_to_record,
    ingest_bulk_dump_csv,
    refresh_npi_taxonomy_cache,
    upsert_npi_records,
)

# --------------------------------------------------------------------------- #
# Bulk-dump row → NppesRecord                                                  #
# --------------------------------------------------------------------------- #


def _bulk_row(**overrides) -> dict[str, str]:
    """Minimal NPPES bulk-dump row with just the fields the parser touches.

    The real dump has ~330 columns; missing columns default to "" in
    csv.DictReader which the parser already tolerates.
    """
    row = {
        "NPI": "1234567893",
        "Entity Type Code": "1",
        "Provider Enumeration Date": "2010-06-15",
        "Last Update Date": "2024-01-02",
        "Provider Organization Name (Legal Business Name)": "",
        "Parent Organization LBN": "Big Health System",
        "Is Sole Proprietor": "N",
        "Provider First Name": "Jane",
        "Provider Last Name (Legal Name)": "Doe",
        "Provider First Line Business Practice Location Address": "100 Clinic Way",
        "Provider Business Practice Location Address City Name": "Clinictown",
        "Provider Business Practice Location Address State Name": "CA",
        "Provider Business Practice Location Address Postal Code": "90210",
        "Healthcare Provider Taxonomy Code_1": "207K00000X",
        "Healthcare Provider Primary Taxonomy Switch_1": "Y",
        "Provider License Number_1": "MD123",
        "Provider License Number State Code_1": "CA",
        "Healthcare Provider Taxonomy Code_2": "207N00000X",
        "Healthcare Provider Primary Taxonomy Switch_2": "N",
    }
    row.update(overrides)
    return row


def test_row_to_record_parses_core_fields():
    rec = _row_to_record(_bulk_row())
    assert rec is not None
    assert rec.npi == "1234567893"
    assert rec.entity_type == "1"
    assert rec.enumeration_date == date(2010, 6, 15)
    assert rec.last_updated_nppes == date(2024, 1, 2)
    assert rec.sole_proprietor is False
    assert rec.first_name == "Jane"
    assert rec.parent_organization_legal_name == "Big Health System"
    assert rec.source == "bulk_dump"


def test_row_to_record_parses_taxonomies_with_primary_flag():
    rec = _row_to_record(_bulk_row())
    assert rec is not None
    assert len(rec.taxonomies) == 2
    primary = rec.primary_taxonomy
    assert primary is not None
    assert primary.code == "207K00000X"
    assert primary.primary is True
    assert primary.license == "MD123"
    assert primary.state == "CA"


def test_row_to_record_returns_none_on_invalid_npi():
    assert _row_to_record(_bulk_row(NPI="not-an-npi")) is None
    assert _row_to_record(_bulk_row(NPI="")) is None
    assert _row_to_record(_bulk_row(NPI="12345")) is None


def test_row_to_record_omits_practice_address_when_all_fields_blank():
    rec = _row_to_record(
        _bulk_row(
            **{
                "Provider First Line Business Practice Location Address": "",
                "Provider Business Practice Location Address City Name": "",
                "Provider Business Practice Location Address State Name": "",
                "Provider Business Practice Location Address Postal Code": "",
            }
        )
    )
    assert rec is not None
    assert rec.practice_address is None


def test_ingest_bulk_dump_csv_streams_records():
    csv_text = (
        "NPI,Entity Type Code,Provider Enumeration Date,Last Update Date,"
        "Is Sole Proprietor,Provider First Name,Provider Last Name (Legal Name),"
        "Healthcare Provider Taxonomy Code_1,"
        "Healthcare Provider Primary Taxonomy Switch_1\n"
        "1234567893,1,2010-06-15,2024-01-02,N,Jane,Doe,207K00000X,Y\n"
        "bad-npi,1,2010-06-15,2024-01-02,N,Bad,Row,207K00000X,Y\n"
        "9999999999,2,2015-01-01,2024-01-02,N,,Big Org,261QF0400X,Y\n"
    )
    import csv as _csv

    reader = _csv.DictReader(io.StringIO(csv_text))
    records = list(ingest_bulk_dump_csv(reader))
    assert len(records) == 2  # bad-npi row is dropped
    assert {r.npi for r in records} == {"1234567893", "9999999999"}


# --------------------------------------------------------------------------- #
# Upsert (fake cursor)                                                         #
# --------------------------------------------------------------------------- #


class _FakeCursor:
    def __init__(self):
        self.batches: list[list[dict]] = []

    def executemany(self, sql, params):
        self.batches.append(list(params))


def test_record_to_upsert_params_serializes_dataclasses():
    rec = NppesRecord(
        npi="1234567893",
        entity_type="1",
        enumeration_date=date(2010, 6, 15),
        taxonomies=(NppesTaxonomy(code="207K00000X", primary=True),),
    )
    params = _record_to_upsert_params(rec)
    assert params["npi"] == "1234567893"
    assert params["taxonomies"] == [
        {"code": "207K00000X", "desc": None, "primary": True, "license": None, "state": None}
    ]
    assert params["enumeration_date"] == date(2010, 6, 15)
    # practice_address absent on this record → None in params
    assert params["practice_address"] is None


def test_upsert_batches_and_serializes_json_columns():
    cur = _FakeCursor()
    records = [
        NppesRecord(npi=f"123456789{i}", taxonomies=(NppesTaxonomy(code="207K00000X"),))
        for i in range(3)
    ]
    # batch_size=2 → 2 batches (sizes 2, 1)
    total = upsert_npi_records(records, cursor=cur, batch_size=2)
    assert total == 3
    assert len(cur.batches) == 2
    assert len(cur.batches[0]) == 2
    assert len(cur.batches[1]) == 1
    # JSON columns serialised to strings
    first_row = cur.batches[0][0]
    assert isinstance(first_row["taxonomies"], str)
    decoded = json.loads(first_row["taxonomies"])
    assert decoded[0]["code"] == "207K00000X"


def test_upsert_empty_iterable_is_noop():
    cur = _FakeCursor()
    total = upsert_npi_records(iter(()), cursor=cur)
    assert total == 0
    assert cur.batches == []


# --------------------------------------------------------------------------- #
# DB → NppesRecord                                                             #
# --------------------------------------------------------------------------- #


def test_row_tuple_to_record_reconstructs_full_record():
    row = (
        "1234567893",
        "1",
        date(2010, 6, 15),
        date(2024, 1, 2),
        [{"code": "207K00000X", "primary": True, "desc": "Allergy"}],
        {"address_1": "100 Clinic Way", "city": "Clinictown", "state": "CA"},
        "Big Health System",
        None,
        False,
        "Jane",
        "Doe",
        "bulk_dump",
    )
    rec = _row_tuple_to_record(row)
    assert rec.npi == "1234567893"
    assert rec.enumeration_date == date(2010, 6, 15)
    assert rec.taxonomies[0].code == "207K00000X"
    assert rec.taxonomies[0].primary is True
    assert rec.practice_address is not None
    assert rec.practice_address.address_1 == "100 Clinic Way"


# --------------------------------------------------------------------------- #
# Celery task entry point                                                      #
# --------------------------------------------------------------------------- #


def test_refresh_task_is_noop_when_bulk_path_unset(monkeypatch):
    monkeypatch.delenv("NPPES_BULK_DUMP_PATH", raising=False)
    # Invoke the underlying function directly (Celery task is a thin wrapper).
    # Celery binds `self` at call time, but we can invoke .run() to skip
    # the bind, or pass a stand-in self via the underlying __wrapped__.
    result = refresh_npi_taxonomy_cache.run()
    assert result["status"] == "skipped"
    assert result["reason"] == "no_bulk_dump_path"
    assert result["rows_upserted"] == 0


def test_refresh_task_is_noop_when_db_url_unset(monkeypatch, tmp_path):
    fake_dump = tmp_path / "dump.csv"
    fake_dump.write_text("NPI\n1234567893\n")
    monkeypatch.setenv("NPPES_BULK_DUMP_PATH", str(fake_dump))
    monkeypatch.delenv("NPPES_DB_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = refresh_npi_taxonomy_cache.run()
    assert result["status"] == "skipped"
    assert result["reason"] == "no_db_url"


def test_refresh_task_reports_missing_bulk_dump(monkeypatch, tmp_path):
    monkeypatch.setenv("NPPES_BULK_DUMP_PATH", str(tmp_path / "does-not-exist.csv"))
    monkeypatch.setenv("NPPES_DB_URL", "postgresql://fake")
    result = refresh_npi_taxonomy_cache.run()
    assert result["status"] == "error"
    assert result["reason"] == "bulk_dump_missing"
