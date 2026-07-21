"""Chain-ladder completion-factor nowcast over the synthetic claims arrival plane.

Backlog #45 PR-B. The DGP stamps every claims-derived ``treatment_events`` row
with ``claim_available_date`` (= event_date + adjudication lag; migration 115 +
PR-A stamp pass). Base KPIs never read that column — they report the MATURE
(omniscient, all-events) values. This module computes the honest as-of-frontier
view for the Rx-volume trend KPIs (TRx WS3-BI-005 / NRx WS3-BI-006 / NBRx
WS3-BI-007) and grosses it up back to a nowcast:

1. **Data**: the migration-116 lag-triangle registry queries return, per
   calendar service month, the histogram of ``arrival_offset_days`` (=
   claim_available_date − month start, day-granular; NULL = unstamped row)
   plus the global prescription ``data_min`` / ``frontier`` scalars. Query-time
   live compute, mirroring the migration-110 segmented-history pattern.
2. **Empirical completion curve** from MATURE service months only. A month is
   mature iff (a) its age ``x_m = frontier − month_start`` reaches the maximum
   OBSERVED arrived offset ``D_obs`` (so its inclusion cannot be
   selection-biased by a lucky fast tail) and (b) every one of its events has
   arrived (``A_m == N_m`` — checkable because the substrate is omniscient; a
   real-feed deployment would substitute a fixed maturity horizon). The pooled
   delay CDF over mature months gives ``CF(x) = P(offset <= x)`` — genuinely
   re-estimated from the arrival plane, never read from the generating gamma.
3. **Provisional** value for month ``m``: events with
   ``claim_available_date <= frontier`` ⇔ ``offset <= x_m`` — the under-count.
4. **Nowcast** = provisional / CF(x_m); **mature** = the all-events total, so
   the demo can display the nowcast recovering the known truth (an honest
   self-check: if nowcast ≉ mature on mature-enough months the estimator is
   wrong, not hidden).
5. **Uncertainty**: percentile bootstrap CI on the nowcast combining
   (i) estimation noise in CF — cluster resampling of mature months — and
   (ii) binomial process noise in the arrived count given CF (the Mack-style
   process-variance term; without it the interval could not honestly cover the
   mature value, because the arrived count's own sampling noise dominates once
   many mature months pin CF down).

MATURITY GUARD (codex adversarial review, finding 3 — MANDATORY):

* ``MIN_MATURE_MONTHS = 6`` mature months are required before ANY completion
  factor is estimated. Rationale: (a) the bootstrap resamples mature months as
  clusters — below ~6 clusters the percentile CI degenerates (with k months
  there are only C(2k−1, k) distinct resamples; k=6 → 462, k=3 → 10) and
  single-month idiosyncrasies dominate the pooled CDF; (b) six months is at
  least the plane's own runout horizon (pharmacy-claims lags clip at 120 d ≈ 4
  months, +1 month of within-month offset, so a curve estimated from fewer
  mature months than the runout it must extrapolate is under-identified). The
  live substrate spans ~36 months → ~30 mature months in practice; the guard
  only bites on unusually short or young substrates.
* The **anchor-cap frontier month is ALWAYS excluded** from both the mature
  set and the output series: the #853 date-collapse piles ~35-40 % of all
  treatment_events onto the single frontier reference date (measured live
  2026-07-21: 35.1 %), which makes that month unusable for curve fitting and
  its provisional/nowcast values meaningless.
* When the guard fails — or the arrival plane is not populated (pre-reseed
  substrate: NULL offsets) — the result says so EXPLICITLY
  (``insufficient_maturity=True`` + machine-readable ``reason``) and carries
  **no nowcast values whatsoever**. Never a fabricated or fallback CF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.kpi.history_backfill import _complete_months, _to_date
from src.kpi.synthetic_mode import nowcast_triangle_query_id

logger = logging.getLogger(__name__)

# KPI registry code -> base query family (the Rx-volume trend family — the only
# family with monthly history + a single-factor gross-up; ratio KPIs need
# separate numerator/denominator completion factors and are a labeled
# follow-on, per the design). Mirrors SEGMENTED_KPI_QUERY_FAMILIES.
NOWCAST_KPI_QUERY_FAMILIES: Dict[str, str] = {
    "WS3-BI-005": "business_impact_trx",
    "WS3-BI-006": "business_impact_nrx",
    "WS3-BI-007": "business_impact_nbrx",
}

#: Minimum mature service months required to estimate a completion curve.
#: See the module docstring ("MATURITY GUARD") for the derivation.
MIN_MATURE_MONTHS = 6


@dataclass(frozen=True)
class NowcastConfig:
    """Estimator knobs (all defaults documented, none tuned per-request)."""

    min_mature_months: int = MIN_MATURE_MONTHS
    ci_level: float = 0.95
    n_bootstrap: int = 500
    #: Fixed seed -> deterministic, reproducible CIs for identical inputs.
    rng_seed: int = 0
    #: Tolerated unstamped-row share before the plane counts as unpopulated.
    max_unstamped_fraction: float = 0.01


@dataclass(frozen=True)
class MonthNowcast:
    """Per-service-month nowcast point."""

    month: date
    age_days: int
    mature_value: float
    provisional_value: float
    is_mature: bool
    completion_factor: Optional[float]
    nowcast_value: Optional[float]
    nowcast_ci: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class NowcastResult:
    """Full estimator output; ``insufficient_maturity`` gates ``months``."""

    frontier: Optional[date]
    data_min: Optional[date]
    insufficient_maturity: bool
    reason: Optional[str]
    mature_months: List[date] = field(default_factory=list)
    anchor_cap_month: Optional[date] = None
    arrival_plane_coverage: Optional[float] = None
    ci_level: float = NowcastConfig.ci_level
    months: List[MonthNowcast] = field(default_factory=list)


async def fetch_nowcast_rows(kpi_id: str, *, brand: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run the migration-116 lag-triangle query for one KPI via kpi_query.

    Returns the raw (service_month, arrival_offset_days, n, data_min, frontier)
    rows; empty list on error (logged), matching fetch_segmented_rows.
    """
    import inspect

    base = NOWCAST_KPI_QUERY_FAMILIES[kpi_id]
    query_id = nowcast_triangle_query_id(base)
    try:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        result_or_coro = client.rpc(
            "kpi_query", {"query_id": query_id, "params": [brand]}
        ).execute()
        result = await result_or_coro if inspect.isawaitable(result_or_coro) else result_or_coro
        return result.data if getattr(result, "data", None) else []
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to fetch nowcast triangle via {query_id}: {e}", exc_info=True)
        return []


def _insufficient(
    reason: str,
    *,
    frontier: Optional[date] = None,
    data_min: Optional[date] = None,
    mature_months: Optional[List[date]] = None,
    anchor_cap_month: Optional[date] = None,
    coverage: Optional[float] = None,
    config: NowcastConfig,
) -> NowcastResult:
    return NowcastResult(
        frontier=frontier,
        data_min=data_min,
        insufficient_maturity=True,
        reason=reason,
        mature_months=mature_months or [],
        anchor_cap_month=anchor_cap_month,
        arrival_plane_coverage=coverage,
        ci_level=config.ci_level,
        months=[],
    )


def estimate_completion_from_rows(
    rows: List[Dict[str, Any]], config: Optional[NowcastConfig] = None
) -> NowcastResult:
    """Estimate completion factors + nowcasts from migration-116 triangle rows.

    Pure function of ``rows`` — synthesizable in tests, no DB access. See the
    module docstring for the math and the maturity guard.
    """
    cfg = config or NowcastConfig()
    if not rows:
        return _insufficient("no_data", config=cfg)

    # ---- parse ------------------------------------------------------------
    hists: Dict[date, Dict[Optional[int], int]] = {}
    frontier: Optional[date] = None
    data_min: Optional[date] = None
    for r in rows:
        m = _to_date(r.get("service_month"))
        n = r.get("n")
        if m is None or n is None:
            continue
        raw_offset = r.get("arrival_offset_days")
        offset = int(raw_offset) if raw_offset is not None else None
        hists.setdefault(m, {})
        hists[m][offset] = hists[m].get(offset, 0) + int(n)
        if frontier is None:
            frontier = _to_date(r.get("frontier"))
            data_min = _to_date(r.get("data_min"))
    if not hists or frontier is None or data_min is None:
        return _insufficient("no_data", config=cfg)

    anchor_cap_month = frontier.replace(day=1)

    # ---- arrival-plane coverage guard ------------------------------------
    total = sum(sum(h.values()) for h in hists.values())
    unstamped = sum(h.get(None, 0) for h in hists.values())
    coverage = 1.0 - (unstamped / total) if total else 0.0
    if coverage < 1.0 - cfg.max_unstamped_fraction:
        return _insufficient(
            f"arrival_plane_not_populated: coverage {coverage:.4f} < "
            f"{1.0 - cfg.max_unstamped_fraction:.4f}",
            frontier=frontier,
            data_min=data_min,
            anchor_cap_month=anchor_cap_month,
            coverage=coverage,
            config=cfg,
        )

    # ---- per-month aggregates --------------------------------------------
    def _age(m: date) -> int:
        return (frontier - m).days

    n_total: Dict[date, int] = {}
    n_arrived: Dict[date, int] = {}
    for m, h in hists.items():
        x = _age(m)
        n_total[m] = sum(h.values())
        n_arrived[m] = sum(c for o, c in h.items() if o is not None and o <= x)

    arrived_offsets = [
        o for m, h in hists.items() for o in h if o is not None and o <= _age(m) and h[o] > 0
    ]
    if not arrived_offsets:
        return _insufficient(
            "no_arrived_claims",
            frontier=frontier,
            data_min=data_min,
            anchor_cap_month=anchor_cap_month,
            coverage=coverage,
            config=cfg,
        )
    d_obs = max(arrived_offsets)

    # ---- mature set + guard ----------------------------------------------
    mature = sorted(
        m
        for m in hists
        if m != anchor_cap_month and _age(m) >= d_obs and n_arrived[m] == n_total[m]
    )
    if len(mature) < cfg.min_mature_months:
        return _insufficient(
            f"insufficient_mature_months: {len(mature)} < {cfg.min_mature_months}",
            frontier=frontier,
            data_min=data_min,
            mature_months=mature,
            anchor_cap_month=anchor_cap_month,
            coverage=coverage,
            config=cfg,
        )

    # ---- pooled empirical completion curve (point estimate) ---------------
    # counts[i, o] = arrivals of mature month i at offset o (o in 0..d_obs).
    counts = np.zeros((len(mature), d_obs + 1), dtype=np.int64)
    for i, m in enumerate(mature):
        for o, c in hists[m].items():
            if o is not None:
                counts[i, min(o, d_obs)] += c
    pooled_cum = counts.sum(axis=0).cumsum()
    pooled_total = float(pooled_cum[-1])

    def _cf(x: int) -> float:
        if x < 0:
            return 0.0
        return float(pooled_cum[min(x, d_obs)]) / pooled_total

    # ---- bootstrap machinery (estimation + process noise) -----------------
    rng = np.random.default_rng(cfg.rng_seed)
    k = len(mature)
    idx = rng.integers(0, k, size=(cfg.n_bootstrap, k))
    boot_cum = counts[idx].sum(axis=1).cumsum(axis=1)  # (B, d_obs+1)
    boot_totals = boot_cum[:, -1].astype(np.float64)
    alpha = (1.0 - cfg.ci_level) / 2.0

    def _boot_ci(x: int, arrived: int, cf_hat: float) -> Optional[Tuple[float, float]]:
        cf_b = boot_cum[:, min(max(x, 0), d_obs)] / boot_totals
        valid = cf_b > 0.0
        if valid.mean() < 0.9:
            return None
        cf_b = cf_b[valid]
        # Process noise: re-draw the arrived count from the resampled CF given
        # the point-nowcast total (parametric bootstrap; Mack process term).
        n_hat = int(round(arrived / cf_hat)) if cf_hat > 0 else 0
        arrived_b = rng.binomial(n_hat, np.clip(cf_b, 0.0, 1.0))
        nowcast_b = arrived_b / cf_b
        lo, hi = np.quantile(nowcast_b, [alpha, 1.0 - alpha])
        return (float(lo), float(hi))

    # ---- output months (complete calendar months, anchor-cap excluded) ----
    months = [m for m in _complete_months([data_min, frontier]) if m != anchor_cap_month]
    mature_set = set(mature)
    points: List[MonthNowcast] = []
    for m in months:
        x = _age(m)
        h = hists.get(m, {})
        total_m = sum(h.values())
        arrived_m = sum(c for o, c in h.items() if o is not None and o <= x)
        is_mature = m in mature_set
        cf = _cf(x)
        if cf > 0.0:
            nowcast: Optional[float] = float(arrived_m) / cf
            ci = None if is_mature else _boot_ci(x, arrived_m, cf)
        else:
            # Younger than the observed offset support: no honest gross-up.
            nowcast, ci = None, None
        points.append(
            MonthNowcast(
                month=m,
                age_days=x,
                mature_value=float(total_m),
                provisional_value=float(arrived_m),
                is_mature=is_mature,
                completion_factor=cf if cf > 0.0 else None,
                nowcast_value=nowcast,
                nowcast_ci=ci,
            )
        )

    return NowcastResult(
        frontier=frontier,
        data_min=data_min,
        insufficient_maturity=False,
        reason=None,
        mature_months=mature,
        anchor_cap_month=anchor_cap_month,
        arrival_plane_coverage=coverage,
        ci_level=cfg.ci_level,
        months=points,
    )
