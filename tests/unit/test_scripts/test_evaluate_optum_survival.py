"""Plan v3 §3 Tier 1C survival evaluation — unit tests for the
target-derivation logic.

Pins `derive_survival_target` semantics: right-censoring at
prediction_horizon_days, observed-event computation, multi-event
collapse to first-biologic, negative-day rejection (pre-index events
are NOT counted).
"""

from __future__ import annotations

import pandas as pd

from scripts.evaluate_optum_survival import (
    BIOLOGIC_DRUG_NAME_PATTERNS,
    DEFAULT_PREDICTION_HORIZON_DAYS,
    SURVIVAL_LIFT_THRESHOLD_OVER_BINARY,
    derive_survival_target,
)

# --------------------------------------------------------------------------- #
# Module constants                                                            #
# --------------------------------------------------------------------------- #


def test_constants_match_plan() -> None:
    assert DEFAULT_PREDICTION_HORIZON_DAYS == 180
    assert SURVIVAL_LIFT_THRESHOLD_OVER_BINARY == 0.04
    assert "xolair" in BIOLOGIC_DRUG_NAME_PATTERNS
    assert "dupixent" in BIOLOGIC_DRUG_NAME_PATTERNS


# --------------------------------------------------------------------------- #
# derive_survival_target                                                      #
# --------------------------------------------------------------------------- #


def _journeys(*pids: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": list(pids),
            "index_date": ["2022-01-01"] * len(pids),
        }
    )


def _events_for(pid: str, day_offset: int, drug: str = "Xolair") -> pd.DataFrame:
    """Single-event DataFrame at index_date + day_offset."""
    return pd.DataFrame(
        {
            "patient_id": [pid],
            "event_date": [pd.Timestamp("2022-01-01") + pd.Timedelta(days=day_offset)],
            "drug_name": [drug],
        }
    )


class TestDeriveSurvivalTarget:
    def test_event_within_horizon_observed(self) -> None:
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", 30)
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 1
        assert row["time_to_initiation_days"] == 30.0

    def test_event_at_horizon_boundary_observed(self) -> None:
        """Day 180 exactly — the spec says ≤ horizon → event observed."""
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", 180)
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 1
        assert row["time_to_initiation_days"] == 180.0

    def test_event_after_horizon_censored(self) -> None:
        """Day 181 — beyond horizon → censored at horizon."""
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", 181)
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 0
        assert row["time_to_initiation_days"] == 180.0  # capped

    def test_event_pre_index_NOT_observed(self) -> None:
        """Day -30 — pre-index event → censored at horizon (NOT a valid
        outcome since index defines the prediction-window start)."""
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", -30)
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 0
        assert row["time_to_initiation_days"] == 180.0

    def test_no_biologic_event_censored(self) -> None:
        """Patient with no biologic event in the events DataFrame →
        censored at horizon."""
        journeys = _journeys("PAT_1")
        events = pd.DataFrame(columns=["patient_id", "event_date", "drug_name"])
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 0
        assert row["time_to_initiation_days"] == 180.0

    def test_multiple_biologic_events_uses_first(self) -> None:
        """Two biologic events at days 30 and 90 → first (day 30) wins."""
        journeys = _journeys("PAT_1")
        events = pd.concat(
            [_events_for("PAT_1", 90), _events_for("PAT_1", 30)],
            ignore_index=True,
        )
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 1
        assert row["time_to_initiation_days"] == 30.0

    def test_dupixent_pattern_matches(self) -> None:
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", 30, drug="DUPIXENT (dupilumab)")
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 1

    def test_omalizumab_generic_pattern_matches(self) -> None:
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", 30, drug="omalizumab generic")
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 1

    def test_non_biologic_drug_does_not_match(self) -> None:
        """Aspirin is not a biologic; should NOT count as initiation."""
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", 30, drug="Aspirin")
        target = derive_survival_target(journeys, events)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 0

    def test_returns_one_row_per_journey_patient(self) -> None:
        journeys = _journeys("PAT_1", "PAT_2", "PAT_3")
        events = pd.DataFrame(
            {
                "patient_id": ["PAT_1"],
                "event_date": [pd.Timestamp("2022-02-15")],
                "drug_name": ["Xolair"],
            }
        )
        target = derive_survival_target(journeys, events)
        assert len(target) == 3
        assert set(target["patient_id"].tolist()) == {"PAT_1", "PAT_2", "PAT_3"}
        # PAT_1 observed; others censored at horizon.
        assert target[target["patient_id"] == "PAT_1"]["event_observed"].iloc[0] == 1
        assert target[target["patient_id"] == "PAT_2"]["event_observed"].iloc[0] == 0
        assert target[target["patient_id"] == "PAT_3"]["event_observed"].iloc[0] == 0

    def test_custom_prediction_horizon(self) -> None:
        journeys = _journeys("PAT_1")
        events = _events_for("PAT_1", 100)
        # With horizon=90, day 100 is censored.
        target = derive_survival_target(journeys, events, prediction_horizon_days=90)
        row = target[target["patient_id"] == "PAT_1"].iloc[0]
        assert row["event_observed"] == 0
        assert row["time_to_initiation_days"] == 90.0
        # With horizon=180, day 100 is observed.
        target_180 = derive_survival_target(journeys, events, prediction_horizon_days=180)
        row_180 = target_180[target_180["patient_id"] == "PAT_1"].iloc[0]
        assert row_180["event_observed"] == 1
        assert row_180["time_to_initiation_days"] == 100.0


class TestEvaluateEndToEnd:
    """Smoke test: the evaluate() entry point runs without raising and
    returns the canonical dict shape."""

    def test_evaluate_smoke(self, tmp_path) -> None:
        """Synthetic 100-patient cohort with 10 observed events. The eval
        should at minimum return all canonical keys."""
        from scripts.evaluate_optum_survival import evaluate

        # Build minimal synthetic cohort: write parquets so _load_cohort
        # can read them. patient_id format must match the
        # `f"PAT_{int(patid)}"` reconstruction used by the loader.
        # The raw med must sit at PROJECT_ROOT/data/rwd/Optum_Parquet/medication.parquet
        # because evaluate()'s default raw_optum_dir resolves there.
        cohort_dir = tmp_path / "cohort"
        cohort_dir.mkdir()
        raw_dir = tmp_path / "data" / "rwd" / "Optum_Parquet"
        raw_dir.mkdir(parents=True)

        n = 100
        journeys = pd.DataFrame(
            {
                "patient_id": [f"PAT_{i}" for i in range(n)],
                "index_date": ["2022-01-01"] * n,
                "age_at_index": [50] * n,
                "primary_diagnosis_code": ["L50.9"] * n,
            }
        )
        journeys.to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")

        # 10 patients have a Xolair event 60 days post-index in raw med.
        med = pd.DataFrame(
            {
                "patid": list(range(10)),
                "medication_date": [pd.Timestamp("2022-03-02")] * 10,
                "Generic_Name": ["omalizumab"] * 10,
                "Brand_Name": ["XOLAIR"] * 10,
            }
        )
        med.to_parquet(raw_dir / "medication.parquet")

        # Stub the raw_optum_dir lookup. evaluate() loads via _load_cohort
        # which defaults raw_optum_dir to PROJECT_ROOT/data/rwd/Optum_Parquet.
        # Patch via monkeypatching the module-level constant.
        import scripts.evaluate_optum_survival as mod

        original_root = mod.PROJECT_ROOT
        mod.PROJECT_ROOT = tmp_path
        try:
            result = evaluate(cohort_dir)
        finally:
            mod.PROJECT_ROOT = original_root

        # Canonical keys present.
        for key in (
            "evaluation_metadata",
            "target_stats",
            "cox_concordance",
            "decision_gate",
        ):
            assert key in result
        # Target stats reflect the 10 events.
        assert result["target_stats"]["n_total"] == 100
        assert result["target_stats"]["n_event_observed"] == 10
