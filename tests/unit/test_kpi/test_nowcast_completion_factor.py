"""Tests for src.kpi.nowcast.completion_factor (chain-ladder claims nowcast, #45).

Row fixtures mirror the REAL migration-116 kpi_query output shape (validated
against a rolled-back temp-table session on the live psql 2026-07-21): rows of
service_month / arrival_offset_days / n plus the global prescription range
(data_min / frontier) on every row. arrival_offset_days is NULL for unstamped
rows (pre-#45 substrate).

The estimator is pure math on these frames — NOTHING here touches the live DB,
which does not yet carry the arrival-plane columns (migration 115 is PR-A).
"""

from datetime import date

import numpy as np
import pytest

from src.kpi.nowcast.completion_factor import (
    MIN_MATURE_MONTHS,
    NOWCAST_KPI_QUERY_FAMILIES,
    NowcastConfig,
    estimate_completion_from_rows,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

#: Deterministic per-month arrival-offset histogram: N=1000, max offset 130.
#: Identical across months -> the pooled mature CDF equals every month's own
#: CDF exactly, so the nowcast must recover the mature value EXACTLY.
_EXACT_HIST = {10: 200, 40: 300, 70: 300, 100: 150, 130: 50}


def _month_range(first: str, last: str) -> list[str]:
    months = []
    d = date.fromisoformat(first)
    stop = date.fromisoformat(last)
    while d <= stop:
        months.append(d.isoformat())
        d = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return months


def _rows(month_hists: dict, frontier: str, data_min: str) -> list[dict]:
    rows = []
    for month, hist in month_hists.items():
        for offset, n in hist.items():
            rows.append(
                {
                    "service_month": month,
                    "arrival_offset_days": offset,
                    "n": n,
                    "data_min": data_min,
                    "frontier": frontier,
                }
            )
    return rows


def _exact_fixture_rows(frontier: str = "2026-06-15") -> list[dict]:
    """17 identical months 2025-01..2026-05; frontier mid-June 2026.

    Ages vs 2026-06-15: 2026-05 -> 45d (CF 0.50), 2026-04 -> 75d (CF 0.80),
    2026-03 -> 106d (CF 0.95); months <= 2026-02 (age >= 134d >= D_obs=130)
    are fully arrived -> 14 mature months.
    """
    hists = {m: dict(_EXACT_HIST) for m in _month_range("2025-01-01", "2026-05-01")}
    return _rows(hists, frontier=frontier, data_min="2025-01-01")


def _noisy_fixture_rows(seed: int = 7) -> list[dict]:
    """Same layout, per-month multinomial(1000) draws -> CF varies by month."""
    rng = np.random.default_rng(seed)
    offsets = [10, 40, 70, 100, 130]
    p = [0.2, 0.3, 0.3, 0.15, 0.05]
    hists = {}
    for m in _month_range("2025-01-01", "2026-05-01"):
        counts = rng.multinomial(1000, p)
        hists[m] = {o: int(c) for o, c in zip(offsets, counts, strict=True) if c > 0}
    return _rows(hists, frontier="2026-06-15", data_min="2025-01-01")


def _by_month(result):
    return {p.month.isoformat(): p for p in result.months}


# ---------------------------------------------------------------------------
# CF math on the synthesized triangle
# ---------------------------------------------------------------------------


class TestCompletionFactorMath:
    def test_mature_months_cf_is_one_and_nowcast_equals_mature(self):
        result = estimate_completion_from_rows(_exact_fixture_rows())
        assert result.insufficient_maturity is False
        assert result.frontier == date(2026, 6, 15)
        points = _by_month(result)
        for m in _month_range("2025-01-01", "2026-02-01"):
            pt = points[m]
            assert pt.is_mature is True
            assert pt.completion_factor == pytest.approx(1.0)
            assert pt.mature_value == 1000.0
            assert pt.provisional_value == 1000.0
            assert pt.nowcast_value == pytest.approx(1000.0)

    def test_recent_month_cf_matches_arrived_fraction_exactly(self):
        points = _by_month(estimate_completion_from_rows(_exact_fixture_rows()))
        assert points["2026-03-01"].completion_factor == pytest.approx(0.95)
        assert points["2026-04-01"].completion_factor == pytest.approx(0.80)
        assert points["2026-05-01"].completion_factor == pytest.approx(0.50)

    def test_nowcast_recovers_known_mature_value_on_partially_arrived_months(self):
        points = _by_month(estimate_completion_from_rows(_exact_fixture_rows()))
        for m in ("2026-03-01", "2026-04-01", "2026-05-01"):
            pt = points[m]
            assert pt.is_mature is False
            assert pt.nowcast_value == pytest.approx(pt.mature_value, rel=1e-9)

    def test_nowcast_within_tolerance_on_noisy_triangle(self):
        points = _by_month(estimate_completion_from_rows(_noisy_fixture_rows()))
        for m in ("2026-03-01", "2026-04-01", "2026-05-01"):
            pt = points[m]
            assert pt.nowcast_value == pytest.approx(1000.0, rel=0.10)

    def test_provisional_less_than_mature_for_recent_months(self):
        points = _by_month(estimate_completion_from_rows(_exact_fixture_rows()))
        for m in ("2026-03-01", "2026-04-01", "2026-05-01"):
            pt = points[m]
            assert pt.provisional_value < pt.mature_value

    def test_cf_monotone_nondecreasing_in_age(self):
        result = estimate_completion_from_rows(_exact_fixture_rows())
        # months ascend -> age decreases -> CF must be non-increasing.
        cfs = [p.completion_factor for p in result.months]
        assert all(a >= b for a, b in zip(cfs, cfs[1:], strict=False))

    def test_months_are_complete_calendar_months_sorted(self):
        result = estimate_completion_from_rows(_exact_fixture_rows())
        got = [p.month.isoformat() for p in result.months]
        assert got == _month_range("2025-01-01", "2026-05-01")


# ---------------------------------------------------------------------------
# Maturity guard (codex-folded finding 3)
# ---------------------------------------------------------------------------


class TestMaturityGuard:
    def test_min_mature_months_threshold_documented_value(self):
        assert MIN_MATURE_MONTHS == 6

    def test_insufficient_mature_months_is_explicit_no_nowcast(self):
        # Only 2026-01..2026-05 exist -> just Jan+Feb are mature (2 < 6).
        hists = {m: dict(_EXACT_HIST) for m in _month_range("2026-01-01", "2026-05-01")}
        rows = _rows(hists, frontier="2026-06-15", data_min="2026-01-01")
        result = estimate_completion_from_rows(rows)
        assert result.insufficient_maturity is True
        assert "insufficient_mature_months" in result.reason
        assert "2" in result.reason and "6" in result.reason
        assert result.months == []  # NO nowcast values — never a fallback CF

    def test_anchor_cap_frontier_month_excluded_from_estimation_and_output(self):
        # Reproduce the #853 pile-up: a huge frontier-month cohort whose claims
        # have not arrived (offsets beyond the frontier's day-of-month).
        base = _exact_fixture_rows(frontier="2026-06-15")
        pileup = _rows(
            {"2026-06-01": {21: 4000, 50: 2000}},
            frontier="2026-06-15",
            data_min="2025-01-01",
        )
        with_pileup = estimate_completion_from_rows(base + pileup)
        without = estimate_completion_from_rows(base)
        assert with_pileup.anchor_cap_month == date(2026, 6, 1)
        assert date(2026, 6, 1) not in with_pileup.mature_months
        assert "2026-06-01" not in {p.month.isoformat() for p in with_pileup.months}
        # The pile-up must not perturb the completion curve at all.
        assert [(p.month, p.completion_factor, p.nowcast_value) for p in with_pileup.months] == [
            (p.month, p.completion_factor, p.nowcast_value) for p in without.months
        ]

    def test_anchor_cap_month_excluded_even_when_calendar_complete(self):
        # Frontier ON the month's last day: _complete_months would keep June,
        # but the anchor-cap guard must still drop it.
        base = _exact_fixture_rows(frontier="2026-06-30")
        pileup = _rows(
            {"2026-06-01": {36: 4000, 60: 2000}},
            frontier="2026-06-30",
            data_min="2025-01-01",
        )
        result = estimate_completion_from_rows(base + pileup)
        assert result.anchor_cap_month == date(2026, 6, 1)
        assert "2026-06-01" not in {p.month.isoformat() for p in result.months}

    def test_unpopulated_arrival_plane_is_explicit_not_fabricated(self):
        # Pre-reseed live substrate: every row is unstamped (NULL offset).
        hists = {m: {None: 1000} for m in _month_range("2025-01-01", "2026-05-01")}
        rows = _rows(hists, frontier="2026-06-15", data_min="2025-01-01")
        result = estimate_completion_from_rows(rows)
        assert result.insufficient_maturity is True
        assert "arrival_plane_not_populated" in result.reason
        assert result.months == []
        assert result.arrival_plane_coverage == pytest.approx(0.0)

    def test_empty_rows_are_explicit(self):
        result = estimate_completion_from_rows([])
        assert result.insufficient_maturity is True
        assert result.reason == "no_data"
        assert result.months == []


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_covers_true_mature_value_on_noisy_fixture(self):
        result = estimate_completion_from_rows(
            _noisy_fixture_rows(), config=NowcastConfig(ci_level=0.95, rng_seed=0)
        )
        points = _by_month(result)
        for m in ("2026-03-01", "2026-04-01", "2026-05-01"):
            pt = points[m]
            lo, hi = pt.nowcast_ci
            assert lo <= 1000.0 <= hi, f"{m}: CI [{lo}, {hi}] misses true mature 1000"
            assert lo <= pt.nowcast_value <= hi
            assert hi > lo

    def test_ci_is_deterministic_given_seed(self):
        cfg = NowcastConfig(rng_seed=123)
        a = estimate_completion_from_rows(_noisy_fixture_rows(), config=cfg)
        b = estimate_completion_from_rows(_noisy_fixture_rows(), config=cfg)
        assert [p.nowcast_ci for p in a.months] == [p.nowcast_ci for p in b.months]

    def test_mature_months_carry_no_ci(self):
        points = _by_month(estimate_completion_from_rows(_exact_fixture_rows()))
        assert points["2025-06-01"].nowcast_ci is None


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestFamilies:
    def test_rx_volume_family_only(self):
        assert NOWCAST_KPI_QUERY_FAMILIES == {
            "WS3-BI-005": "business_impact_trx",
            "WS3-BI-006": "business_impact_nrx",
            "WS3-BI-007": "business_impact_nbrx",
        }
