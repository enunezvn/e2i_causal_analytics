"""scripts/backfill_hcp_treatment_arm.py's channel re-plant path (lane T1).

Hermetic: a recording PostgREST fake holds hcp_profiles, hcp_brand_adoption and the
business_metrics per_hcp_rollup rows. Pinned:

  * the paged read of the planted rollups asks for ``count="exact"``, stops on an empty page,
    and fails LOUD when a page is short of the server's count (a silent partial read would
    plant a shift on a subset and call the rest non-joined);
  * ``derive(..., channel_rollups=...)`` keeps treatment_arm identical to the no-shift
    derive (the shift consumes no rng draws), gives every joined row
    ``consideration_date = max(metric_date) + lag`` with lag in [1, 90] and CLAMPED to the run
    date, leaves non-joined rows' dates alone (NaT -> the writer omits the column), and never
    dates a row after the run date;
  * the dry-run report carries the fields the owner reads before ``--execute``;
  * ``main()`` without ``--execute`` performs NO update on the client, and ``write_rows``
    sends consideration_date only for joined rows and ``updated_at`` on every row.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from scripts import backfill_hcp_treatment_arm as bf
from src.data.per_hcp_cohort_columns import INTERVENTION_TREATMENT_MAP

_CHANNELS = sorted(set(INTERVENTION_TREATMENT_MAP.values()))
_RUN_DATE = date(2026, 9, 23)


# --------------------------------------------------------------------------- fake client
class _Resp:
    def __init__(self, data, count):
        self.data, self.count = data, count


class _Query:
    def __init__(self, table, op, payload=None):
        self._t, self._op, self._payload = table, op, payload
        self._filters, self._in, self._count, self._order, self._range = [], [], None, [], None

    def select(self, cols, count=None):
        self._count = count
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def in_(self, col, vals):
        self._in.append((col, list(vals)))
        return self

    def order(self, col, desc=False):
        self._order.append((col, desc))
        self._t.orders.append(list(self._order))
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def update(self, payload):
        return _Query(self._t, "update", payload)

    def _matching(self):
        rows = [
            r
            for r in self._t.rows
            if all(r.get(c) == v for c, v in self._filters)
            and all(r.get(c) in vals for c, vals in self._in)
        ]
        for col, desc in reversed(self._order):
            rows = sorted(rows, key=lambda r: str(r.get(col)), reverse=desc)
        return rows

    def execute(self):
        rows = self._matching()
        if self._op == "update":
            for r in rows:
                r.update(self._payload)
            self._t.updates.append((list(self._filters), dict(self._payload)))
            return _Resp([dict(r) for r in rows], None)
        total = len(rows)
        if self._range is not None:
            s, e = self._range
            rows = rows[s : e + 1]
        if self._t.short_by and self._range is not None and self._range[0] == 0:
            rows = rows[: max(0, len(rows) - self._t.short_by)]
        return _Resp([dict(r) for r in rows], total if self._count == "exact" else None)


class _Table:
    def __init__(self, rows):
        self.rows = [dict(r) for r in rows]
        self.updates = []
        self.orders = []
        self.short_by = 0
        self.fail_select = False

    def select(self, cols, count=None):
        if self.fail_select:
            raise RuntimeError("simulated PostgREST read failure")
        return _Query(self, "select").select(cols, count=count)

    def update(self, payload):
        return _Query(self, "update", payload)


class _Client:
    def __init__(self, tables):
        self.tables = {k: _Table(v) for k, v in tables.items()}

    def table(self, name):
        return self.tables[name]


# --------------------------------------------------------------------------- fixture data
_HCPS = ["h1", "h2", "h3", "h4", "h5", "h6"]  # h6 never appears in the rollups (non-joined)
_SPECS = ["oncology", "hematology", "dermatology", "oncology", "allergy_immunology", "neurology"]


def _profiles():
    return [
        {"hcp_id": h, "peer_influence_score": p, "specialty": s, "is_synthetic": True}
        for h, p, s in zip(_HCPS, [4.1, 2.2, 3.3, 5.0, 1.7, 2.9], _SPECS, strict=True)
    ]


def _adoption(derived_dates=None):
    rows = []
    for brand in bf.BRANDS:
        for i, h in enumerate(_HCPS):
            rows.append(
                {
                    "hcp_id": h,
                    "brand": brand,
                    "adopted": i % 2,
                    "adoption_category": "ADOPTER" if i % 2 else "NON_ADOPTER",
                    "treatment_arm": None,
                    "consideration_date": "2024-03-01",
                    "is_synthetic": True,
                }
            )
    return rows


def _rollups():
    """16 rows over 3 brands: multi-row pairs (h1/Remibrutinib x3, h2/Fabhalta x2), a Sep-21
    exposure for h3/Remibrutinib so an unclamped lag would land after the run date, and 3
    Kisqali rows. Plus rows the read/derive must EXCLUDE: a non-synthetic row, an other-brand
    row, and an ORPHAN rollup (hcp_id absent from hcp_profiles) that would move a median."""
    spec = [
        ("h1", "Remibrutinib", "2026-05-01"),
        ("h1", "Remibrutinib", "2026-07-21"),
        ("h1", "Remibrutinib", "2026-08-10"),
        ("h2", "Remibrutinib", "2026-07-21"),
        ("h3", "Remibrutinib", "2026-09-21"),
        ("h4", "Remibrutinib", "2026-07-21"),
        ("h5", "Remibrutinib", "2026-06-02"),
        ("h1", "Fabhalta", "2026-07-21"),
        ("h2", "Fabhalta", "2026-07-21"),
        ("h2", "Fabhalta", "2026-09-01"),
        ("h3", "Fabhalta", "2026-07-21"),
        ("h4", "Fabhalta", "2026-05-15"),
        ("h5", "Fabhalta", "2026-07-21"),
        ("h1", "Kisqali", "2026-07-21"),
        ("h2", "Kisqali", "2026-07-21"),
        ("h4", "Kisqali", "2026-08-02"),
    ]
    rng = np.random.default_rng(1)
    rows = []
    for i, (h, b, d) in enumerate(spec):
        row = {
            "metric_id": f"m{i:03d}",
            "hcp_id": h,
            "brand": b,
            "metric_type": "per_hcp_rollup",
            "is_synthetic": True,
            "metric_date": d,
            "region": ["northeast", "west", "south", "midwest"][i % 4],
            "market_share": round(float(rng.random()), 3),
            "triggers_total_count": int(rng.integers(1, 20)),
        }
        for c in _CHANNELS:
            row[c] = round(float(rng.random() * 10), 3)
        rows.append(row)
    # One non-synthetic and one other-brand row that the read must EXCLUDE.
    rows.append({**rows[0], "metric_id": "real0", "is_synthetic": False})
    rows.append({**rows[1], "metric_id": "other0", "brand": "OtherBrand"})
    # An ORPHAN synthetic rollup: hcp_id not in hcp_profiles. Extreme values so that, if it
    # were counted, every Kisqali median would move.
    rows.append(
        {
            **rows[-3],
            "metric_id": "orphan0",
            "hcp_id": "zz_orphan",
            "brand": "Kisqali",
            **dict.fromkeys(_CHANNELS, 999.0),
        }
    )
    return rows


@pytest.fixture
def client():
    return _Client(
        {
            "hcp_profiles": _profiles(),
            "hcp_brand_adoption": _adoption(),
            "business_metrics": _rollups(),
        }
    )


# --------------------------------------------------------------------------- paged read
def test_fetch_channel_rollups_reads_only_synthetic_rows_of_the_three_brands(client):
    df = bf.fetch_channel_rollups(client, page_size=5)
    assert len(df) == 17  # 16 cohort rows + the orphan (the READ is by brand/is_synthetic only)
    assert set(df["brand"]) == {"Remibrutinib", "Fabhalta", "Kisqali"}
    assert set(df.columns) >= {
        "hcp_id",
        "brand",
        "metric_date",
        "region",
        "market_share",
        "triggers_total_count",
        *_CHANNELS,
    }
    assert pd.api.types.is_datetime64_any_dtype(df["metric_date"])


def test_fetch_channel_rollups_fails_loud_on_a_short_page(client):
    client.tables["business_metrics"].short_by = 1
    with pytest.raises(RuntimeError, match="short"):
        bf.fetch_channel_rollups(client, page_size=5)


def test_fetch_channel_rollups_stops_on_an_empty_table():
    empty = _Client({"business_metrics": []})
    df = bf.fetch_channel_rollups(empty, page_size=5)
    assert df.empty


def test_fetch_live_adoption_pages_on_a_total_order(client):
    """hcp_id repeats once per brand, so a page boundary inside an hcp_id's three rows is
    non-deterministic under ORDER BY hcp_id alone: the 2026-09-23 dry-run read one Remibrutinib
    row twice and dropped a Kisqali row (5001/5000/4999). The sort must be total."""
    bf.fetch_live_adoption(client)
    orders = client.tables["hcp_brand_adoption"].orders
    assert orders and orders[-1] == [("hcp_id", False), ("brand", False)]


# --------------------------------------------------------------------------- derive
@pytest.fixture
def derived(client):
    centrality = bf.fetch_centrality(client)
    rollups = bf.fetch_channel_rollups(client, page_size=5)
    return bf.derive(centrality, seed=427, channel_rollups=rollups, run_date=_RUN_DATE)


def test_derive_with_channels_keeps_treatment_arm_identical_to_the_no_shift_derive(client, derived):
    plain = bf.derive(bf.fetch_centrality(client), seed=427)
    assert derived["treatment_arm"].tolist() == plain["treatment_arm"].tolist()
    assert derived["hcp_id"].tolist() == plain["hcp_id"].tolist()
    assert derived["brand"].tolist() == plain["brand"].tolist()
    assert set(plain.columns) == {
        "hcp_id",
        "brand",
        "treatment_arm",
        "adopted",
        "adoption_category",
        "cate_estimate",
    }


def test_derive_dates_joined_rows_after_their_last_exposure_and_never_after_the_run_date(derived):
    joined = derived[derived["joined"]]
    non_joined = derived[~derived["joined"]]
    assert len(joined) == 13 and len(non_joined) == 5  # 6 HCPs x 3 brands
    assert set(non_joined["hcp_id"]) >= {"h6"}
    assert "zz_orphan" not in set(derived["hcp_id"])
    cd = pd.to_datetime(joined["consideration_date"])
    mx = pd.to_datetime(joined["max_metric_date"])
    assert (cd > mx).all()
    assert (cd <= pd.Timestamp(_RUN_DATE)).all()
    lag = (cd - mx).dt.days
    assert lag.min() >= 1 and lag.max() <= 90
    # The Sep-21 exposure (h3/Remibrutinib) must be clamped: only 2 days of room.
    h3 = joined[(joined["hcp_id"] == "h3") & (joined["brand"] == "Remibrutinib")].iloc[0]
    assert pd.Timestamp(h3["consideration_date"]) <= pd.Timestamp(_RUN_DATE)
    assert pd.Timestamp(h3["max_metric_date"]) == pd.Timestamp("2026-09-21")
    assert non_joined["consideration_date"].isna().all()
    assert (non_joined["channel_shift"] == 0.0).all()
    assert (joined["channel_shift"] != 0.0).any()


def test_derive_lag_stream_is_spawned_not_the_arm_stream(client):
    """The lag comes from default_rng([brand_seed, 1]); the 4 DGP draws are untouched."""
    centrality = bf.fetch_centrality(client)
    rollups = bf.fetch_channel_rollups(client, page_size=5)
    a = bf.derive(centrality, seed=427, channel_rollups=rollups, run_date=_RUN_DATE)
    b = bf.derive(centrality, seed=427, channel_rollups=rollups, run_date=date(2027, 1, 1))
    assert a["adopted"].tolist() == b["adopted"].tolist()
    assert a["treatment_arm"].tolist() == b["treatment_arm"].tolist()
    master = np.random.default_rng(427)
    brand_seed = int(master.integers(0, 2**32))  # Remibrutinib is first
    lags = np.random.default_rng([brand_seed, 1]).integers(1, 91, size=len(centrality))
    rem = b[(b["brand"] == "Remibrutinib") & b["joined"]]
    expect = pd.to_datetime(rem["max_metric_date"]) + pd.to_timedelta(
        lags[rem.index % len(centrality)], unit="D"
    )
    assert pd.to_datetime(rem["consideration_date"]).tolist() == expect.tolist()


def test_derive_excludes_orphan_rollups_from_the_planting_medians(client):
    """codex r1 (MED): the planted contrast must be the estimator's contrast, and the estimator
    only sees rows joined to the cohort. An orphan rollup (hcp_id not in hcp_profiles) must
    not move a within-brand median: with the 999-valued orphan counted, the Kisqali medians
    over 4 rows would sit between the cohort's values and flip a bit."""
    centrality = bf.fetch_centrality(client)
    rollups = bf.fetch_channel_rollups(client, page_size=5)
    with_orphan = bf.derive(centrality, seed=427, channel_rollups=rollups, run_date=_RUN_DATE)
    without = bf.derive(
        centrality,
        seed=427,
        channel_rollups=rollups[rollups["hcp_id"] != "zz_orphan"],
        run_date=_RUN_DATE,
    )
    kis = with_orphan[(with_orphan["brand"] == "Kisqali") & with_orphan["joined"]].set_index(
        "hcp_id"
    )
    kis0 = without[(without["brand"] == "Kisqali") & without["joined"]].set_index("hcp_id")
    for c in _CHANNELS:
        assert kis[f"tbin_{c}"].tolist() == kis0[f"tbin_{c}"].tolist(), c
    assert with_orphan["adopted"].tolist() == without["adopted"].tolist()


def test_derive_fails_loud_on_a_null_confounder_or_channel(client):
    """codex r1 (MED): a row null in a confounder is dropped by the estimator but would sit in
    the planting median. The plant writes every column; a null is corruption -> refuse."""
    centrality = bf.fetch_centrality(client)
    rollups = bf.fetch_channel_rollups(client, page_size=5)
    bad = rollups.copy()
    bad.loc[bad.index[0], "market_share"] = np.nan
    with pytest.raises(ValueError, match="null"):
        bf.derive(centrality, seed=427, channel_rollups=bad, run_date=_RUN_DATE)
    bad = rollups.copy()
    bad.loc[bad.index[1], "engagement_score"] = np.nan
    with pytest.raises(ValueError, match="null"):
        bf.derive(centrality, seed=427, channel_rollups=bad, run_date=_RUN_DATE)


def test_derive_refuses_an_exposure_on_or_after_the_run_date(client):
    centrality = bf.fetch_centrality(client)
    rollups = bf.fetch_channel_rollups(client, page_size=5)
    with pytest.raises(ValueError, match="run date"):
        bf.derive(centrality, seed=427, channel_rollups=rollups, run_date=date(2026, 9, 21))


# --------------------------------------------------------------------------- report + writes
def test_verify_report_carries_the_owner_fields(client, derived):
    live = bf.fetch_live_adoption(client)
    report = bf.verify(derived, live, run_date=_RUN_DATE)
    for brand in bf.BRANDS:
        r = report[brand]
        for key in (
            "n_rows",
            "n_joined",
            "n_non_joined",
            "arm_match_vs_live",
            "prevalence_live",
            "prevalence_new",
            "labels_flipped",
            "realised_rd_stratified",
            "realised_rd_dgp_true",
            "null_channel_rd",
            "n_joined_dated_after_last_exposure",
            "n_future_dates",
            "true_ate_arm",
            "naive_diff_arm",
        ):
            assert key in r, f"{brand}: {key} missing"
        assert r["n_future_dates"] == 0
        assert r["n_joined_dated_after_last_exposure"] == r["n_joined"]
        assert set(r["realised_rd_stratified"]) == set(_CHANNELS) or r["n_joined"] == 0


def test_main_dry_run_performs_no_update_and_writes_the_frame(client, tmp_path, monkeypatch):
    out = tmp_path / "frame.parquet"
    rc = bf.main(
        ["--run-date", "2026-09-23", "--frame-out", str(out), "--backup-dir", str(tmp_path)],
        client=client,
    )
    assert rc == 0
    assert client.tables["hcp_brand_adoption"].updates == []
    frame = pd.read_parquet(out)
    assert len(frame) == 18 and "consideration_date" in frame.columns and "region" in frame.columns


def test_main_refuses_when_a_brand_has_no_planted_rollups(tmp_path):
    """codex r1 (MED): a brand with no rollups would silently get zero shifts and be written."""
    rollups = [r for r in _rollups() if r["brand"] != "Kisqali"]
    c = _Client(
        {
            "hcp_profiles": _profiles(),
            "hcp_brand_adoption": _adoption(),
            "business_metrics": rollups,
        }
    )
    rc = bf.main(["--run-date", "2026-09-23", "--backup-dir", str(tmp_path)], client=c)
    assert rc == 1
    assert c.tables["hcp_brand_adoption"].updates == []


def test_execute_refuses_when_the_live_read_fails_but_dry_run_reports(tmp_path, monkeypatch):
    """codex r1 (MED): --execute was fail-open on a failed live read (no backup, no arm-match
    precondition). It must abort; the dry-run may still report without the comparison."""
    c = _Client(
        {
            "hcp_profiles": _profiles(),
            "hcp_brand_adoption": _adoption(),
            "business_metrics": _rollups(),
        }
    )
    c.tables["hcp_brand_adoption"].fail_select = True
    monkeypatch.setattr(bf, "ensure_schema", lambda client: None)
    rc = bf.main(["--execute", "--run-date", "2026-09-23", "--backup-dir", str(tmp_path)], client=c)
    assert rc == 1
    assert c.tables["hcp_brand_adoption"].updates == []
    rc = bf.main(["--run-date", "2026-09-23", "--backup-dir", str(tmp_path)], client=c)
    assert rc == 0


def test_execute_refuses_when_live_keys_do_not_match_the_derived_keys(tmp_path, monkeypatch):
    rows = [r for r in _adoption() if not (r["hcp_id"] == "h6" and r["brand"] == "Kisqali")]
    c = _Client(
        {"hcp_profiles": _profiles(), "hcp_brand_adoption": rows, "business_metrics": _rollups()}
    )
    monkeypatch.setattr(bf, "ensure_schema", lambda client: None)
    rc = bf.main(["--execute", "--run-date", "2026-09-23", "--backup-dir", str(tmp_path)], client=c)
    assert rc == 1
    assert c.tables["hcp_brand_adoption"].updates == []


def test_write_rows_requires_exactly_one_matched_row_per_key(client, derived):
    """codex r1 (MED): the count was of ATTEMPTED updates. A key with no live row must abort."""
    client.tables["hcp_brand_adoption"].rows = [
        r
        for r in client.tables["hcp_brand_adoption"].rows
        if not (r["hcp_id"] == "h3" and r["brand"] == "Fabhalta")
    ]
    with pytest.raises(RuntimeError, match="h3"):
        bf.write_rows(client, derived)


def test_write_rows_sends_dates_for_joined_rows_only_and_updated_at_for_all(client, derived):
    n = bf.write_rows(client, derived)
    updates = client.tables["hcp_brand_adoption"].updates
    assert n == len(updates) == 18
    for filters, payload in updates:
        keys = dict(filters)
        assert set(keys) == {"hcp_id", "brand", "is_synthetic"}
        assert {"treatment_arm", "adopted", "adoption_category", "updated_at"} <= set(payload)
        datetime.fromisoformat(payload["updated_at"])
        row = derived[
            (derived["hcp_id"] == keys["hcp_id"]) & (derived["brand"] == keys["brand"])
        ].iloc[0]
        if row["joined"]:
            assert (
                payload["consideration_date"]
                == pd.Timestamp(row["consideration_date"]).date().isoformat()
            )
        else:
            assert "consideration_date" not in payload
