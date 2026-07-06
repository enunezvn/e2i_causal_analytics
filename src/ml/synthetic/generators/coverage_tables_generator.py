"""Synthetic substrate for the 5 view-backed KPI tables (Shard 09).

These tables already hold rows on prod but ALL pre-date now()-30d (max 2025-11-28),
so MAU/WAU/intent-delta/data-lag/label-quality read 0. We INSERT fresh
is_synthetic=true rows anchored to the rolling window (we do NOT delete the stale
real rows). Columns verified against information_schema.columns; data_lag_hours /
time_to_release_hours are derived from the rolling timestamps so v_kpi_data_lag /
v_kpi_time_to_release compute non-NULL.

Enum-exact (22P02 landmine): user_region (region_type), brand (brand_type).

#1115 user-population model: the old `i % 30` cap saturated distinct users at 30,
pinning WS3-BI-001 (MAU target 2000) and WS3-BI-002 (WAU target 1200) to CRITICAL
forever. user_sessions now carries a heterogeneous population of ~n_records//4
users (2500 at the production loader config) split into daily / weekly /
occasional cohorts with weekday/weekend structure and a weekly activity wobble,
so trailing-30d MAU clears 2000 and trailing-7d WAU clears 1200 at the frontier
with a realistic WAU/MAU engagement ratio (~0.6) and week-to-week variation.

IDEMPOTENT (reseed-safe): uuid.uuid4() PKs ignored the seed, so every reseed
INSERTed fresh-id rows the loader's upsert-on-PK could never match -> the table
ACCUMULATED (user_sessions reached 40k = 4 x 10k reseeds; same failure mode as
the #1105/#1106 MLOps/ab_* incident). All 5 tables now derive their PKs by uuid5
from a NATURAL KEY (cf. experiment_generator._EXP_ID_NS) so a re-run UPDATEs in
place. Session activity/attributes are additionally keyed to the ABSOLUTE
calendar date (per-date RNG streams), so a later reseed regenerates identical
rows for overlapping dates and only appends the new frontier days.
"""

import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import BaseGenerator, GeneratorConfig

_REGIONS = ["northeast", "south", "midwest", "west"]
_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]
_SOURCES = ["iqvia", "healthverity", "komodo", "veeva"]

# Fixed namespace for DETERMINISTIC ids (cf. experiment_generator._EXP_ID_NS,
# mlops_generator._MLOPS_ID_NS). uuid5(natural key) makes the loader's
# upsert-on-PK UPDATE in place instead of accumulating fresh-uuid rows.
_COVERAGE_ID_NS = uuid.UUID("3f2a7c81-95d4-4e0b-a6c9-1d8e5b7f2043")


def _det_id(*parts: object) -> str:
    """Deterministic uuid5 from a natural key (stable across runs)."""
    return str(uuid.uuid5(_COVERAGE_ID_NS, "|".join(str(p) for p in parts)))


# --- #1115 user-population model -------------------------------------------
# (cohort, population share, weekday activity probability). Shares are exact
# index slices (first 15% of users are daily, ...), so cohort membership is
# stable across runs. Weekday rates x the weekend factor / weekly wobble give:
# WAU ~= 0.53-0.56 x population, MAU ~= 0.86-0.89 x population -> at 2500 users
# MAU ~2200 (>= 2000 target) and WAU ~1400 (>= 1200 target), WAU/MAU ~0.63.
_USER_COHORTS: List[Tuple[str, float, float]] = [
    ("daily", 0.15, 0.50),
    ("weekly", 0.45, 0.20),
    ("occasional", 0.40, 0.05),
]
_WEEKEND_FACTOR = 0.25  # field tool: weekend usage collapses
_WOBBLE_SIGMA = 0.08  # week-to-week activity multiplier ~ clip(N(1, 0.08))
_WOBBLE_CLIP = (0.85, 1.15)
_HISTORY_DAYS = 90  # usage history span ending at run_date
_ROLES = ["rep", "manager", "analyst"]
_ROLE_P = [0.70, 0.15, 0.15]
_MIN_USERS = 30


class CoverageTablesGenerator(BaseGenerator[pd.DataFrame]):
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        run_date: Optional[date] = None,
        hcp_ids: Optional[Sequence[str]] = None,
        n_users: Optional[int] = None,
    ):
        super().__init__(config)
        self.run_date = run_date or date.today()
        # hcp_intent_surveys.hcp_id carries a NO-ACTION FK to hcp_profiles.hcp_id:
        # the loader passes the run's generated (namespaced) hcp ids so surveys
        # resolve against the run's own profiles instead of hardcoded hcp_NNNNN ids
        # that only exist as legacy stub rows.
        self.hcp_ids = list(hcp_ids) if hcp_ids is not None else None
        # ~4 config rows per user keeps the session volume in the same order of
        # magnitude as the pre-#1115 substrate (FULL n_records=10000 -> 2500
        # users x ~13 sessions/user/90d ~= 33k sessions vs the 40k it had
        # accumulated). Explicit n_users overrides for tests/sims.
        self.n_users = (
            n_users if n_users is not None else max(_MIN_USERS, self.config.n_records // 4)
        )

    @property
    def entity_type(self) -> str:
        return "user_sessions"

    def _ts(self, max_days_back: int, recent_frac: float = 0.7) -> datetime:
        """Timestamp in [run_date-max_days_back, run_date]; recent_frac fall in last 30d."""
        if self._rng.random() < recent_frac:
            days = int(self._rng.integers(0, 30))
        else:
            days = int(self._rng.integers(30, max_days_back + 1))
        d = self.run_date - timedelta(days=days)
        return datetime.combine(d, time(hour=int(self._rng.integers(0, 24))), tzinfo=timezone.utc)

    def _generate_user_sessions(self, now: datetime) -> pd.DataFrame:
        """user_sessions: MAU/WAU (WS3-BI-001/002) — heterogeneous user population.

        Every draw is keyed to (seed, user index) or (seed, absolute date), NOT to
        the offset from run_date: re-running with a later run_date reproduces the
        exact same rows for overlapping dates (upsert no-op) and appends only the
        new frontier days.
        """
        n_users = self.n_users
        prefix = self.config.id_prefix
        user_ids = [f"synth_user_{i:04d}" for i in range(n_users)]

        # Cohort weekday rates by exact index slice (deterministic membership).
        base_p = np.empty(n_users, dtype=float)
        lo = 0
        for _, share, p_weekday in _USER_COHORTS:
            hi = min(n_users, lo + int(round(share * n_users)))
            base_p[lo:hi] = p_weekday
            lo = hi
        base_p[lo:] = _USER_COHORTS[-1][2]  # rounding remainder -> occasional

        # Per-user stable attributes from a dedicated stream (a user keeps ONE
        # role/region across sessions and across reseeds with the same seed).
        user_rng = np.random.default_rng([self.config.seed, 0xA77])
        roles = user_rng.choice(_ROLES, size=n_users, p=_ROLE_P)
        regions = user_rng.choice(_REGIONS, size=n_users)

        days = [self.run_date - timedelta(days=b) for b in range(_HISTORY_DAYS - 1, -1, -1)]
        wobble_cache: Dict[Tuple[int, int], float] = {}
        rows: List[Dict[str, Any]] = []
        for d in days:
            iso = d.isocalendar()
            wk = (iso[0], iso[1])
            if wk not in wobble_cache:
                w_rng = np.random.default_rng([self.config.seed, 0x1115, iso[0], iso[1]])
                wobble_cache[wk] = float(np.clip(w_rng.normal(1.0, _WOBBLE_SIGMA), *_WOBBLE_CLIP))
            mult = wobble_cache[wk] * (_WEEKEND_FACTOR if d.weekday() >= 5 else 1.0)
            p = np.clip(base_p * mult, 0.0, 0.95)

            # One RNG stream per ABSOLUTE date -> activity and attributes for a
            # given (user, date) are identical in every run that covers the date.
            day_rng = np.random.default_rng([self.config.seed, 0x5E55, d.toordinal()])
            active = day_rng.random(n_users) < p
            hours = day_rng.integers(7, 20, size=n_users)
            minutes = day_rng.integers(0, 60, size=n_users)
            durations = day_rng.integers(60, 3600, size=n_users)
            pages = day_rng.integers(1, 30, size=n_users)
            queries = day_rng.integers(0, 15, size=n_users)
            actions = day_rng.integers(0, 8, size=n_users)
            engagement = day_rng.uniform(0.2, 0.95, size=n_users)

            d_iso = d.isoformat()
            for ui in np.flatnonzero(active):
                start = datetime.combine(
                    d, time(hour=int(hours[ui]), minute=int(minutes[ui])), tzinfo=timezone.utc
                )
                dur = int(durations[ui])
                rows.append(
                    {
                        # Natural key (user, date): one session per active day;
                        # reseed UPDATEs in place instead of accumulating.
                        "session_id": _det_id("session", prefix, user_ids[ui], d_iso),
                        "user_id": user_ids[ui],
                        "user_role": str(roles[ui]),
                        "user_region": str(regions[ui]),
                        "session_start": start.isoformat(),
                        "session_end": (start + timedelta(seconds=dur)).isoformat(),
                        "session_duration_seconds": dur,
                        "page_views": int(pages[ui]),
                        "queries_executed": int(queries[ui]),
                        "actions_taken": int(actions[ui]),
                        "engagement_score": round(float(engagement[ui]), 3),
                        "created_at": now.isoformat(),
                        "is_synthetic": True,
                    }
                )
        return pd.DataFrame(rows)

    def generate(self) -> Dict[str, pd.DataFrame]:  # type: ignore[override]
        n = self.config.n_records
        prefix = self.config.id_prefix
        now = datetime.now(timezone.utc)

        # --- user_sessions: MAU/WAU (WS3-BI-001/002) — #1115 population model ---
        us_df = self._generate_user_sessions(now)

        # --- hcp_intent_surveys: BR-002 intent delta (v_kpi_intent_to_prescribe) ---
        hi = []
        for i in range(n):
            sdate = self._ts(90).date()
            pre = int(self._rng.integers(2, 6))
            delta = int(self._rng.integers(0, 3))  # positive change so BR-002 target met
            hi.append(
                {
                    "survey_id": _det_id("survey", prefix, i),
                    "hcp_id": (
                        str(self._rng.choice(self.hcp_ids)) if self.hcp_ids else f"hcp_{i % 50:05d}"
                    ),
                    "survey_date": sdate.isoformat(),
                    "survey_type": "follow_up",
                    "brand": _BRANDS[i % 3],
                    "intent_to_prescribe_score": min(7, pre + delta),
                    "intent_to_prescribe_change": delta,
                    "awareness_score": int(self._rng.integers(3, 8)),
                    "favorability_score": int(self._rng.integers(3, 8)),
                    "previous_survey_id": None,
                    "days_since_last_survey": int(self._rng.integers(20, 40)),
                    "survey_source": "synthetic",
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )

        # --- data_source_tracking: WS1-DQ-003/004 (cross-source match / stacking_lift) ---
        ds = []
        for i in range(n):
            received = int(self._rng.integers(1000, 5000))
            matched = int(received * float(self._rng.uniform(0.70, 0.90)))
            elig = int(received * 0.5)
            ds.append(
                {
                    "tracking_id": _det_id("tracking", prefix, i),
                    "tracking_date": self._ts(60).date().isoformat(),
                    "source_name": _SOURCES[i % len(_SOURCES)],
                    "source_type": "claims",
                    "records_received": received,
                    "records_matched": matched,
                    "records_unique": int(matched * 0.95),
                    "match_rate_vs_iqvia": round(float(self._rng.uniform(0.70, 0.90)), 3),
                    "match_rate_vs_healthverity": round(float(self._rng.uniform(0.65, 0.85)), 3),
                    "match_rate_vs_komodo": round(float(self._rng.uniform(0.60, 0.80)), 3),
                    "match_rate_vs_veeva": round(float(self._rng.uniform(0.70, 0.88)), 3),
                    "stacking_eligible_records": elig,
                    "stacking_applied_records": int(elig * float(self._rng.uniform(0.6, 0.9))),
                    "stacking_lift_percentage": round(float(self._rng.uniform(0.10, 0.25)), 3),
                    "data_quality_score": round(float(self._rng.uniform(0.80, 0.98)), 3),
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )

        # --- etl_pipeline_metrics: WS1-DQ-009 TTR (v_kpi_time_to_release) ---
        et = []
        for i in range(n):
            src_ts = self._ts(60)
            ttr_h = float(self._rng.uniform(2, 40))
            end = src_ts + timedelta(hours=ttr_h)
            et.append(
                {
                    "pipeline_run_id": _det_id("etl_run", prefix, i),
                    "pipeline_name": "rwd_ingest",
                    "pipeline_version": "v9.1",
                    "run_start": src_ts.isoformat(),
                    "run_end": end.isoformat(),
                    "duration_seconds": int(ttr_h * 3600),
                    "source_data_date": src_ts.date().isoformat(),
                    "source_data_timestamp": src_ts.isoformat(),
                    "time_to_release_hours": round(ttr_h, 2),
                    "records_processed": int(self._rng.integers(1000, 50000)),
                    "records_failed": int(self._rng.integers(0, 50)),
                    # 'success' per the e2i_ml_complete_v3_schema.sql status
                    # contract ('success'|'partial'|'failed') — v_kpi_time_to_release
                    # and the WS1-DQ-009 registry queries filter status='success',
                    # so any other spelling zeroes the TTR KPI (migration 095
                    # backfilled the pre-fix 'completed' rows).
                    "status": "success",
                    "quality_checks_passed": int(self._rng.integers(8, 12)),
                    "quality_checks_failed": int(self._rng.integers(0, 2)),
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )

        # --- ml_annotations: WS1-DQ-008 IAA (v_kpi_label_quality) ---
        ann: list[dict[str, Any]] = []
        for i in range(n):
            ann.append(
                {
                    "annotation_id": _det_id("annotation", prefix, i),
                    "entity_type": "patient_journey",
                    "entity_id": f"pj_{i:05d}",
                    "annotation_type": "discontinuation_label",
                    "annotator_id": f"annot_{i % 5}",
                    "annotator_role": "clinician",
                    # v_kpi_label_quality / data_quality_label_quality count the
                    # categorical label strings 'positive'/'negative'/'uncertain'
                    # (NOT 0/1) -> emit those so the IAA computation is non-degenerate.
                    "annotation_value": {
                        "label": str(self._rng.choice(["positive", "negative", "uncertain"]))
                    },
                    "annotation_confidence": round(float(self._rng.uniform(0.7, 0.99)), 3),
                    "annotation_timestamp": self._ts(30).isoformat(),
                    "is_adjudicated": False,
                    # Triplets share a group (i//3): same grouping as before, but
                    # deterministic so IAA groups are stable across reseeds.
                    "iaa_group_id": _det_id("iaa_group", prefix, i // 3),
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )

        return {
            "user_sessions": us_df,
            "hcp_intent_surveys": pd.DataFrame(hi),
            "data_source_tracking": pd.DataFrame(ds),
            "etl_pipeline_metrics": pd.DataFrame(et),
            "ml_annotations": pd.DataFrame(ann),
        }
