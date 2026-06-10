"""Synthetic substrate for the 5 view-backed KPI tables (Shard 09).

These tables already hold rows on prod but ALL pre-date now()-30d (max 2025-11-28),
so MAU/WAU/intent-delta/data-lag/label-quality read 0. We INSERT fresh
is_synthetic=true rows anchored to the rolling window (we do NOT delete the stale
real rows). Columns verified against information_schema.columns; data_lag_hours /
time_to_release_hours are derived from the rolling timestamps so v_kpi_data_lag /
v_kpi_time_to_release compute non-NULL.

Enum-exact (22P02 landmine): user_region (region_type), brand (brand_type).
"""

import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Optional

import pandas as pd

from .base import BaseGenerator, GeneratorConfig

_REGIONS = ["northeast", "south", "midwest", "west"]
_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]
_SOURCES = ["iqvia", "healthverity", "komodo", "veeva"]


class CoverageTablesGenerator(BaseGenerator[pd.DataFrame]):
    def __init__(self, config: Optional[GeneratorConfig] = None, run_date: Optional[date] = None):
        super().__init__(config)
        self.run_date = run_date or date.today()

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

    def generate(self) -> Dict[str, pd.DataFrame]:  # type: ignore[override]
        n = self.config.n_records
        now = datetime.now(timezone.utc)

        # --- user_sessions: MAU/WAU (WS3-BI-001/002) ---
        us = []
        for i in range(n):
            start = self._ts(90)
            dur = int(self._rng.integers(60, 3600))
            us.append(
                {
                    "session_id": str(uuid.uuid4()),
                    "user_id": f"synth_user_{i % 30:03d}",
                    "user_role": str(self._rng.choice(["rep", "manager", "analyst"])),
                    "user_region": str(self._rng.choice(_REGIONS)),
                    "session_start": start.isoformat(),
                    "session_end": (start + timedelta(seconds=dur)).isoformat(),
                    "session_duration_seconds": dur,
                    "page_views": int(self._rng.integers(1, 30)),
                    "queries_executed": int(self._rng.integers(0, 15)),
                    "actions_taken": int(self._rng.integers(0, 8)),
                    "engagement_score": round(float(self._rng.uniform(0.2, 0.95)), 3),
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )

        # --- hcp_intent_surveys: BR-002 intent delta (v_kpi_intent_to_prescribe) ---
        hi = []
        for i in range(n):
            sdate = self._ts(90).date()
            pre = int(self._rng.integers(2, 6))
            delta = int(self._rng.integers(0, 3))  # positive change so BR-002 target met
            hi.append(
                {
                    "survey_id": str(uuid.uuid4()),
                    "hcp_id": f"hcp_{i % 50:05d}",
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
                    "tracking_id": str(uuid.uuid4()),
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
                    "pipeline_run_id": str(uuid.uuid4()),
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
                    "status": "completed",
                    "quality_checks_passed": int(self._rng.integers(8, 12)),
                    "quality_checks_failed": int(self._rng.integers(0, 2)),
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )

        # --- ml_annotations: WS1-DQ-008 IAA (v_kpi_label_quality) ---
        ann = []
        for i in range(n):
            group = (
                str(uuid.uuid4())
                if i % 3 == 0
                else (ann[-1]["iaa_group_id"] if ann else str(uuid.uuid4()))
            )
            ann.append(
                {
                    "annotation_id": str(uuid.uuid4()),
                    "entity_type": "patient_journey",
                    "entity_id": f"pj_{i:05d}",
                    "annotation_type": "discontinuation_label",
                    "annotator_id": f"annot_{i % 5}",
                    "annotator_role": "clinician",
                    "annotation_value": {"label": int(self._rng.integers(0, 2))},
                    "annotation_confidence": round(float(self._rng.uniform(0.7, 0.99)), 3),
                    "annotation_timestamp": self._ts(30).isoformat(),
                    "is_adjudicated": False,
                    "iaa_group_id": group,
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                }
            )

        return {
            "user_sessions": pd.DataFrame(us),
            "hcp_intent_surveys": pd.DataFrame(hi),
            "data_source_tracking": pd.DataFrame(ds),
            "etl_pipeline_metrics": pd.DataFrame(et),
            "ml_annotations": pd.DataFrame(ann),
        }
