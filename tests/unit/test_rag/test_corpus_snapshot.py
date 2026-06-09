"""Offline guard for the latest-per-combo corpus snapshot logic (audit F3b).

The durable corpus must cover EVERY (brand, metric_name, region) combo, not just
the rows on the most-recent dates. Live data is irregular: the latest metric_date
carries only a subset of combos (e.g. 24/48 for Kisqali, 3/48 for Fabhalta), so a
naive ``limit_per_brand`` (order-by-date-desc, take N) silently omits combos
(e.g. Remibrutinib and the `west` region were entirely absent from the corpus).
``_latest_per_combo`` keeps the FIRST row per (metric_name, region) from a
date-descending stream -> the latest snapshot of every combo. Tested with real
dicts (no mock).
"""

from src.rag.corpus_ingestion import _latest_per_combo


def test_keeps_latest_row_per_metric_region_combo():
    # rows arrive ordered by metric_date DESC (as the DB query returns them)
    rows = [
        {"metric_name": "TRx", "region": "west", "metric_date": "2025-10-29", "value": 5},
        {"metric_name": "TRx", "region": "west", "metric_date": "2025-09-01", "value": 4},  # older dup
        {"metric_name": "TRx", "region": "east", "metric_date": "2025-08-01", "value": 3},
        {"metric_name": "NBRx", "region": "west", "metric_date": "2025-07-01", "value": 2},
    ]
    out = _latest_per_combo(rows)
    assert len(out) == 3  # (TRx,west), (TRx,east), (NBRx,west)
    trx_west = [r for r in out if r["metric_name"] == "TRx" and r["region"] == "west"]
    assert len(trx_west) == 1 and trx_west[0]["value"] == 5  # latest kept (date-desc first)


def test_covers_all_combos_regardless_of_date_gaps():
    # a combo whose only rows are OLD must still appear (the naive recent-N bug)
    rows = [
        {"metric_name": "TRx", "region": "northeast", "metric_date": "2025-10-29", "value": 9},
        {"metric_name": "Market_Share", "region": "west", "metric_date": "2022-01-01", "value": 1},
    ]
    out = _latest_per_combo(rows)
    combos = {(r["metric_name"], r["region"]) for r in out}
    assert ("Market_Share", "west") in combos, "stale-only combo must not be dropped"


def test_empty_input_returns_empty():
    assert _latest_per_combo([]) == []
