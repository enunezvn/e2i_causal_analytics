"""Frontier-append synthetic supplement — grow the substrate, never rewrite it.

WHY (gold-standard supplement workstream, 2026-07-04): the weekly Mon-3AM cron
ran a FULL ``--anchor-to-now`` reseed, which rewrites every synthetic timestamp
forward each Monday. That kept NOW()-windowed KPIs fed (#1127 MAU/WAU decay)
but (a) re-fired an honest-but-noisy drift storm on the 12 gold-standard models
every week (the 2026-07-04 alert-storm investigation), (b) meant the dataset
never grew, and (c) made every row's values hostage to the anchor date.

DESIGN — freeze + epoch-append:

* The existing substrate (id namespace ``scv``, last anchored 2026-07-03) is
  FROZEN. It must never be regenerated: ``--anchor-to-now`` consumes a
  different number of RNG draws than calendar-fixed generation, so re-anchoring
  or re-pinning would rewrite — on the SAME primary keys — the attribute values
  the 12 gold-standard models were trained on. Full reseed survives only as a
  disaster-recovery path (``scripts/reseed_synthetic.sh --full``).

* From ``EPOCH`` (2026-07-06, the first Monday after the freeze) the substrate
  grows by deterministic WEEKLY COHORTS: new patients whose journeys start
  inside one ISO week, plus their downstream treatments / predictions /
  triggers / feature values. ``business_metrics`` is monthly-grain (the
  generator floors dates to month starts — measured), so it gets MONTHLY
  cohorts from ``BM_EPOCH`` instead; weekly cohorts would multiply month rows.

* A cohort is a pure function of its calendar key: ``id_prefix`` and ``seed``
  derive from the ISO week (or month), sizes are frozen constants. Generation
  is idempotent (two runs emit byte-identical frames — verified 2026-07-04)
  and PK-disjoint from the base substrate and from other cohorts (verified),
  so the loader's upsert-on-PK appends without ever clobbering history.

* Derived events legitimately overshoot their cohort week (treatment_date =
  journey + <=30d; prediction/trigger timestamps up to ~+90d — measured).
  Rather than capping dates (which piles events onto week boundaries), each
  append run REGENERATES the trailing ``TRAIL_WEEKS`` cohorts and filters
  every frame to occurrence-date <= frontier: rows already loaded upsert as
  byte-equal no-ops; rows whose dates newly crossed the frontier append.
  Stateless dribble — no continuation bookkeeping anywhere.

* ``user_sessions`` rides the existing CoverageTablesGenerator contract (rows
  keyed to seed + ABSOLUTE calendar date): re-running with the BASE config and
  ``run_date=frontier`` appends only the new frontier days, which keeps
  MAU/WAU fed — the original reason the weekly full reseed existed. The 4
  index-keyed coverage tables (hcp_intent_surveys, ...) refresh in place,
  keeping BR-002/DQ trailing-window substrates current.

* The fixed-universe Shard-09 substrate (experiments, A/B, MLOps registry,
  observability, causal_paths) is NOT part of append runs — it does not grow
  with the calendar.

WEEKLY_SIZES are FULL_SIZES / 156 (the designed weekly density of the
2022→2024 calendar span). Sizes, seeds and prefixes MUST stay constant:
``n_records`` feeds the RNG draw shape, so resizing a past week's cohort would
regenerate different values under the same PKs and silently rewrite history.

Steady-state consequence (documented, intentional): the frozen base packs ~60%
of its rows into the 30 days before 2026-07-03 (anchor bias). As that bulk
ages out of trailing KPI windows (~5 weeks), volume KPIs step down to the
honest append-rate density (~160 new patients/week). MAU/WAU are unaffected.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.ml.synthetic.config import DGPType
from src.ml.synthetic.generators import (
    AgentActivitiesGenerator,
    BusinessMetricsGenerator,
    FeatureStoreSeeder,
    FeatureValueGenerator,
    GeneratorConfig,
    HCPGenerator,
    PatientGenerator,
    PredictionGenerator,
    TreatmentGenerator,
    TriggerGenerator,
)
from src.ml.synthetic.generators.change_tracking import stamp_change_tracking
from src.ml.synthetic.generators.coverage_tables_generator import (
    CoverageTablesGenerator,
)
from src.ml.synthetic.generators.data_lag import (
    stamp_claim_arrival,
    stamp_data_lag_hours,
    stamp_sequence_number,
)
from src.ml.synthetic.generators.model_metrics import stamp_model_metrics

logger = logging.getLogger(__name__)

# --- frozen constants (see module docstring: changing any of these rewrites
# --- already-loaded cohorts under the same PKs) --------------------------------
EPOCH = date(2026, 7, 6)  # first Monday after the base-substrate freeze
BM_EPOCH = date(2026, 8, 1)  # base bm rows exist through 2026-07-01
# #1566 D1: absolute calendar origin for the monthly business_metrics trend.
# FROZEN and NOT re-derivable by re-running the base generator (its default
# date range re-anchors to the run date); 2013-01-01 is the first month of the
# frozen base actually loaded in the DB (2026-07-03 load, 163 months
# 2013-01..2026-07), giving July 2026 index 162 and the first appended month
# (2026-08) index 163. Without it, a single-date cohort run resets the
# positional month_idx to 0 and collapses to the 2013 baseline (~24% of July).
BM_TREND_ORIGIN = date(2013, 1, 1)
# #1833: the frozen base's generation identity, for regenerating it IN-MEMORY
# (reseed script, gap arbiter) — never for the cron, which only emits cohort
# months from BM_EPOCH. scripts/load_synthetic_data.py loads it with the
# DEFAULT start (re-anchored to the run month); pinning start_date to the
# 2026-07-03 load's first month reproduces that load exactly (Step 0 of #1833
# measured all 9,780 rows byte-identical to the DB on 16 columns).
BM_BASE_N = 10000
BM_BASE_START = BM_TREND_ORIGIN
TRAIL_WEEKS = 26  # covers the max measured derived-date overshoot (~+89d) with margin

# Base-substrate generation identity (scripts/load_synthetic_data.py defaults:
# --tag scv, seed 42, FULL sizes). Needed to regenerate the base HCP universe
# in-memory so weekly patients reference the EXISTING 5000 HCPs instead of
# minting a new HCP population every week.
BASE_TAG = "scv"
BASE_SEED = 42
BASE_HCP_N = 5000
BASE_DGP = DGPType.CONFOUNDED
# CoverageTablesGenerator base config (generate_datasets: seed+5, n=bm size).
BASE_COVERAGE_SEED = BASE_SEED + 5
BASE_COVERAGE_N = 10000

# FULL_SIZES / 156 ISO weeks of the 2022→2024 designed span.
WEEKLY_SIZES = {
    "patient": 160,
    "treatment": 481,
    "prediction": 128,
    "trigger": 77,
    "feature_values": 321,
    # #1355: generic agent_activities rows per week (the curated CATE/causal/
    # gap blocks are fixed additive, ~54 rows). Keeps the NOW()-30d
    # business_impact_roi_agent_activities KPI window populated week over week.
    "agent_activities": 24,
}
MONTHLY_SIZES = {
    # BusinessMetricsGenerator emits n // (combos_per_date + 1) MONTHS forward
    # from start_date (end_date is ignored), combos_per_date = 3 brands x 4
    # regions x 5 metric types = 60. n=61 -> exactly one month of 60 rows —
    # the same monthly density as the frozen base (10000 -> 163 months).
    "business_metrics": 61,
}

# Column each table's frontier filter keys on: the row's OCCURRENCE date (when
# the event happened), never deadline-ish columns (trigger expiration_date may
# legitimately sit in the future).
OCCURRENCE_COLUMNS = {
    "patient_journeys": "journey_start_date",
    "treatment_events": "event_date",
    "ml_predictions": "prediction_timestamp",
    "triggers": "trigger_timestamp",
    "business_metrics": "metric_date",
    "feature_values": "event_timestamp",
    "agent_activities": "activity_timestamp",  # #1355
}

# #1577: columns that must ALSO be filtered at INSTANT granularity. The DB
# enforces ``CHECK (event_timestamp <= now())`` on feature_values (constraint
# ``valid_event_timestamp`` — the only now()-CHECK in the schema). The Mon-3AM
# cron runs with frontier=today and iter_week_starts includes the week
# CONTAINING the frontier, so the current-week cohort holds frontier-DAY rows
# whose times fall after the run instant (measured: e.g. 18:56:17 on the
# 2026-08-10 03:00 run). Date-granularity filtering passes them; the DB then
# rejects them (23514) — 28-459 rows/week, every cron run since 2026-07-06.
# Held-back rows regenerate byte-identically on the loader's upsert key
# (feature_id, entity_values, event_timestamp) and load on a later run — the
# same self-heal contract as the date filter, extended to instant granularity.
# They must be held back, NEVER clamped: event_timestamp is part of that
# upsert key, so shifting it would make re-generated rows duplicate instead
# of dedup.
# Scope (deliberate): ONLY feature_values. triggers / ml_predictions /
# agent_activities carry intraday times too but have no DB instant constraint
# and no measured harm — add a table:column entry here to opt one in.
INSTANT_COLUMNS = {
    "feature_values": "event_timestamp",
}


def week_start_of(d: date) -> date:
    """Monday of the ISO week containing ``d``."""
    return d - timedelta(days=d.weekday())


def week_prefix(week_start: date) -> str:
    """Entity-id namespace for a weekly cohort, e.g. 2026-W28 -> ``w2628``.

    5 chars keeps the longest id (patient_journey_id, 14 chars un-prefixed)
    inside varchar(20); disjoint from the base ``scv`` namespace by the ``w``
    lead. iso-year%100 wraps in 2100 — acceptable for a showcase substrate.
    """
    iso = week_start.isocalendar()
    return f"w{iso[0] % 100:02d}{iso[1]:02d}"


def week_seed(week_start: date) -> int:
    """Deterministic per-week seed, disjoint from base seeds (42..51 — iso
    year*1000 dominates) and from month seeds (week 1..53 < month offset 500)."""
    iso = week_start.isocalendar()
    return BASE_SEED + iso[0] * 1000 + iso[1]


def month_prefix(month_start: date) -> str:
    """Namespace for a monthly business_metrics cohort, e.g. ``m2608``."""
    return f"m{month_start.year % 100:02d}{month_start.month:02d}"


def month_seed(month_start: date) -> int:
    return BASE_SEED + month_start.year * 1000 + 500 + month_start.month


def iter_week_starts(frontier: date) -> List[date]:
    """Mondays of the cohorts an append run must (re)generate: the trailing
    TRAIL_WEEKS window clamped at EPOCH, through the week containing frontier."""
    if frontier < EPOCH:
        return []
    current = week_start_of(frontier)
    first = max(EPOCH, current - timedelta(weeks=TRAIL_WEEKS - 1))
    out = []
    ws = first
    while ws <= current:
        out.append(ws)
        ws += timedelta(weeks=1)
    return out


def iter_month_starts(frontier: date) -> List[date]:
    """Month starts from BM_EPOCH through the month containing frontier.
    bm rows sit exactly on month starts (no overshoot), so no trailing
    regeneration is needed — but re-listing all epoch months is still cheap
    and keeps the run stateless; overlaps upsert as no-ops."""
    if frontier < BM_EPOCH:
        return []
    out = []
    ms = BM_EPOCH
    while ms <= frontier:
        out.append(ms)
        ms = (ms.replace(day=1) + timedelta(days=32)).replace(day=1)
    return out


def base_hcp_frame() -> pd.DataFrame:
    """Regenerate the frozen base HCP universe in-memory (ids are positional,
    so they match the loaded ``scv`` rows regardless of the anchor flag the
    original load used). NOT loaded — parent pool for weekly patients only."""
    return HCPGenerator(
        GeneratorConfig(id_prefix=BASE_TAG, seed=BASE_SEED, n_records=BASE_HCP_N)
    ).generate()


def generate_week_cohort(week_start: date, hcp_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """One deterministic weekly cohort: patients journeying this ISO week plus
    their downstream events. Mirrors generate_datasets' per-frame renames and
    stamp order so appended rows are shape-identical to base rows."""
    week_end = week_start + timedelta(days=6)
    seed = week_seed(week_start)
    prefix = week_prefix(week_start)

    def cfg(n: int, seed_offset: int = 0, **kw) -> GeneratorConfig:
        return GeneratorConfig(
            id_prefix=prefix,
            seed=seed + seed_offset,
            n_records=n,
            start_date=week_start,
            end_date=week_end,
            **kw,
        )

    patients = PatientGenerator(
        cfg(WEEKLY_SIZES["patient"], dgp_type=BASE_DGP), hcp_df=hcp_df
    ).generate()

    treatments = TreatmentGenerator(cfg(WEEKLY_SIZES["treatment"]), patient_df=patients).generate()
    treatments = treatments.rename(
        columns={
            "treatment_date": "event_date",
            "treatment_type": "event_type",
            "days_supply": "duration_days",
        }
    )

    predictions = PredictionGenerator(
        cfg(WEEKLY_SIZES["prediction"]), patient_df=patients
    ).generate()
    predictions = predictions.rename(columns={"prediction_date": "prediction_timestamp"})

    trigger_gen = TriggerGenerator(
        cfg(WEEKLY_SIZES["trigger"]),
        patient_df=patients,
        hcp_df=hcp_df,
        treatment_df=treatments,
    )
    triggers = trigger_gen.generate()
    injected_rx = trigger_gen.injected_prescriptions
    if injected_rx is not None and len(injected_rx) > 0:
        treatments = pd.concat([treatments, injected_rx], ignore_index=True)

    # Feature store: seed groups/features with the cohort identity; the loader
    # reconciles them onto the canonical DB rows by natural key (#852), which
    # also remaps feature_values.feature_id — include all three frames.
    feature_groups, features = FeatureStoreSeeder(cfg(1000)).seed()
    feature_values = FeatureValueGenerator(
        cfg(WEEKLY_SIZES["feature_values"]), features_df=features, patient_df=patients
    ).generate()

    # Stamps mirror generate_datasets' seed offsets (+6 data-lag, +8 model
    # metrics, +9 change tracking, +10 claims arrival plane). Sequence
    # numbering runs on the merged treatments frame BEFORE the frontier filter
    # so an rx keeps its sequence number as later prescriptions cross the
    # frontier in later runs. The arrival stamp draws each row's lag as a pure
    # function of (seed, treatment_event_id) — order/filter/concat cannot
    # reassign it, so byte-equal re-run upserts hold STRUCTURALLY (codex
    # diff-review 2026-07-21), not incidentally to generator ordering.
    patients = stamp_data_lag_hours(patients, seed=seed + 6)
    treatments = stamp_sequence_number(treatments)
    treatments = stamp_claim_arrival(treatments, seed=seed + 10)
    predictions = stamp_model_metrics(predictions, seed=seed + 8)
    triggers = stamp_change_tracking(triggers, seed=seed + 9)

    # #1355: weekly agent_activities cohort (seed offset +11, mirroring
    # generate_datasets). Content-addressed ids under the week prefix keep
    # re-runs idempotent and cohorts disjoint from base 'scv' rows; the
    # frontier filter keys on activity_timestamp (OCCURRENCE_COLUMNS).
    agent_activities = AgentActivitiesGenerator(
        cfg(WEEKLY_SIZES["agent_activities"], seed_offset=11)
    ).generate()

    return {
        "patient_journeys": patients,
        "treatment_events": treatments,
        "ml_predictions": predictions,
        "triggers": triggers,
        "feature_groups": feature_groups,
        "features": features,
        "feature_values": feature_values,
        "agent_activities": agent_activities,
    }


def base_business_metrics_frame() -> pd.DataFrame:
    """Regenerate the frozen business_metrics base (2013-01..2026-07, 9,780
    rows, unprefixed ``metric_<12hex>`` ids) in-memory under the CURRENT DGP.

    Same seed / size as the loader (BASE_SEED, BM_BASE_N) with the date range
    pinned to BM_BASE_START, so ids, dates, targets and every RNG-drawn column
    match the DB rows byte-for-byte; only ``value`` (and its derived
    achievement_rate / CI columns) moves when the value-only brand x region
    terms are retuned — which is what makes a reseed an in-place upsert on
    ``metric_id`` (#1833). NOT part of append runs."""
    return BusinessMetricsGenerator(
        GeneratorConfig(
            id_prefix=BASE_TAG,
            seed=BASE_SEED,
            n_records=BM_BASE_N,
            start_date=BM_BASE_START,
        )
    ).generate()


def generate_month_cohort(month_start: date) -> Dict[str, pd.DataFrame]:
    """One deterministic monthly business_metrics cohort (rows land on
    month_start — the generator floors to month grain)."""
    month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    prefix = month_prefix(month_start)
    bm = BusinessMetricsGenerator(
        GeneratorConfig(
            id_prefix=prefix,
            seed=month_seed(month_start),
            n_records=MONTHLY_SIZES["business_metrics"],
            start_date=month_start,
            end_date=month_end,
            # #1566 D1: anchor the trend index to the frozen base's first
            # month; without it this single-date run gets month_idx=0 and the
            # cohort collapses to the 2013 baseline.
            trend_origin=BM_TREND_ORIGIN,
        )
    ).generate()
    # The generator draws metric_id as unprefixed seeded hex ("metric_<12hex>")
    # — deterministic but namespace-blind. Re-key positionally under the month
    # prefix so cohort ids can NEVER collide with the frozen base's ~10k hex
    # ids (or another month's) and are purgeable by prefix.
    bm["metric_id"] = [f"{prefix}_{i:04d}" for i in range(len(bm))]
    return {"business_metrics": bm}


def filter_to_frontier(
    datasets: Dict[str, pd.DataFrame], frontier: date, as_of: Optional[datetime] = None
) -> Dict[str, pd.DataFrame]:
    """Drop rows whose OCCURRENCE date is after the frontier. Tables without a
    registered occurrence column pass through unchanged (feature_groups,
    features, coverage tables — self-bounded by run_date).

    #1577: when ``as_of`` is given, tables registered in INSTANT_COLUMNS are
    additionally filtered to instant <= as_of (inclusive, mirroring the DB
    CHECK ``event_timestamp <= now()``), so a 03:00 cron run cannot ship
    frontier-day rows the DB will reject. ``as_of=None`` preserves the
    date-only contract for direct callers."""
    out: Dict[str, pd.DataFrame] = {}
    cutoff = frontier.isoformat()
    for table, df in datasets.items():
        col = OCCURRENCE_COLUMNS.get(table)
        if col is None or col not in df.columns or df.empty:
            out[table] = df
            continue
        keep = df[col].astype(str).str[:10] <= cutoff
        dropped = int((~keep).sum())
        if dropped:
            logger.info(
                "frontier filter %s: %d rows beyond %s held back (regenerated next run)",
                table,
                dropped,
                cutoff,
            )
        instant_col = INSTANT_COLUMNS.get(table)
        if as_of is not None and instant_col is not None and instant_col in df.columns:
            # format="ISO8601": the column holds datetime.isoformat() strings
            # of MIXED precision (a row landing on a whole second prints
            # without the .%f fraction) — default inference locks onto the
            # first row's format and raises on the rest.
            instant_keep = pd.to_datetime(df[instant_col], format="ISO8601") <= pd.Timestamp(as_of)
            held_instant = int((keep & ~instant_keep).sum())
            if held_instant:
                logger.info(
                    "instant filter %s: %d rows after %s held back (regenerated next run)",
                    table,
                    held_instant,
                    as_of.isoformat(),
                )
            keep &= instant_keep
        out[table] = df[keep].reset_index(drop=True)
    return out


def build_frontier_datasets(
    frontier: Optional[date] = None,
    as_of: Optional[datetime] = None,
    include_coverage: bool = True,
    hcp_frame_factory: Callable[[], pd.DataFrame] = base_hcp_frame,
) -> Dict[str, pd.DataFrame]:
    """Assemble everything one append run loads: trailing weekly cohorts +
    epoch monthly bm cohorts, frontier-filtered, plus the coverage refresh.
    Deterministic given ``(frontier, as_of)``; safe to re-run any number of
    times.

    ``as_of`` (#1577) is the load instant: INSTANT_COLUMNS tables are held
    back to instant <= as_of so the DB's ``event_timestamp <= now()`` CHECK
    cannot reject rows. None resolves to ``datetime.now()`` (mirroring the
    ``frontier`` default) — for a backdated frontier that is a no-op, so
    historical/manual runs behave exactly as before."""
    frontier = frontier or date.today()
    as_of = as_of or datetime.now()
    week_starts = iter_week_starts(frontier)
    month_starts = iter_month_starts(frontier)
    if not week_starts:
        logger.warning("frontier %s precedes EPOCH %s — nothing to append", frontier, EPOCH)
        return {}

    logger.info(
        "frontier-append: frontier=%s, %d weekly cohorts (%s..%s), %d monthly bm cohorts",
        frontier,
        len(week_starts),
        week_starts[0],
        week_starts[-1],
        len(month_starts),
    )

    hcp_df = hcp_frame_factory()
    merged: Dict[str, List[pd.DataFrame]] = {}

    for ws in week_starts:
        for table, df in generate_week_cohort(ws, hcp_df).items():
            if table in ("feature_groups", "features"):
                # Canonical metadata — byte-identical every cohort (deterministic
                # uuid5 ids); keep a single copy. Every cohort's feature_values
                # resolve against it, so the loader's #852 reconcile covers them all.
                merged.setdefault(table, [])
                if not merged[table]:
                    merged[table].append(df)
                continue
            merged.setdefault(table, []).append(df)

    for ms in month_starts:
        for table, df in generate_month_cohort(ms).items():
            merged.setdefault(table, []).append(df)

    datasets = {t: pd.concat(frames, ignore_index=True) for t, frames in merged.items()}
    datasets = filter_to_frontier(datasets, frontier, as_of=as_of)

    if include_coverage:
        # Base-identity coverage run: user_sessions rows are keyed to absolute
        # dates, so this appends only the days since the last run; the 4
        # index-keyed tables refresh in place. Same config as the base load
        # (tag scv, seed 47, n=10000) or the overlap-day rows would differ.
        coverage = CoverageTablesGenerator(
            GeneratorConfig(
                id_prefix=BASE_TAG,
                seed=BASE_COVERAGE_SEED,
                n_records=BASE_COVERAGE_N,
            ),
            run_date=frontier,
            hcp_ids=hcp_df["hcp_id"].tolist(),
        ).generate()
        datasets.update(coverage)

    # Tables the filter emptied entirely (e.g. no trigger has occurred yet in
    # the first append week) are omitted: the loader's validation rejects
    # empty frames, and there is nothing to load — later runs pick them up.
    datasets = {t: df for t, df in datasets.items() if not df.empty}

    for table, df in datasets.items():
        df["is_synthetic"] = True
        datasets[table] = df

    total = sum(len(df) for df in datasets.values())
    logger.info("frontier-append: %d rows across %d tables ready to upsert", total, len(datasets))
    return datasets
