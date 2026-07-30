"""#1355: agent_activities synthetic substrate — generator + load-path wiring.

agent_activities was dropped in the v3 -> src/ml/synthetic DGP migration (the
legacy src/ml/data_generator.py:_generate_agent_activities produced it; the new
pipeline had zero references), leaving the chat agent-analysis tool, the
business_impact_roi_agent_activities KPI and the RAG index running on an empty
table. These tests pin the restored substrate:

* the generator covers the legacy v3 agent roster with the legacy tier map;
* curated analysis rows MIRROR the COMM-ARMS ground truth (brand-scaled CATE
  maps from the treatment_arm ARM_REGISTRY / brand_scaled_cate SSOT, and the
  content-addressed causal_paths commercial-arm display effects) — never
  invented numbers;
* every emitted column is registered in the BatchLoader TABLE_COLUMNS whitelist
  (repo lesson: unregistered columns are SILENTLY dropped at load) and the
  table is in LOADING_ORDER;
* the standard load path (scripts/load_synthetic_data.py) and the weekly
  frontier-append reseed path both emit the table.
"""

from datetime import date

import pandas as pd
import pytest

from src.ml.synthetic.config import Brand
from src.ml.synthetic.dgp.treatment_arm import (
    _BRAND_CATE_SCALE,
    ARM_REGISTRY,
    brand_scaled_cate,
)
from src.ml.synthetic.generators import GeneratorConfig
from src.ml.synthetic.generators.agent_activities_generator import (
    LEGACY_AGENT_TIERS,
    AgentActivitiesGenerator,
)
from src.ml.synthetic.loaders.batch_loader import LOADING_ORDER, TABLE_COLUMNS

# Legacy v3 roster (src/ml/data_generator.py AGENT_NAMES/AGENT_TIERS) — the
# agent set the v3 generator covered, with the agent_tier_type enum values.
_LEGACY_TIERS = {
    "orchestrator": "coordination",
    "causal_impact": "causal_analytics",
    "gap_analyzer": "causal_analytics",
    "heterogeneous_optimizer": "causal_analytics",
    "drift_monitor": "monitoring",
    "experiment_designer": "monitoring",
    "health_score": "monitoring",
    "prediction_synthesizer": "ml_predictions",
    "resource_optimizer": "ml_predictions",
    "explainer": "self_improvement",
    "feedback_learner": "self_improvement",
}

_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return AgentActivitiesGenerator(
        GeneratorConfig(
            seed=42, n_records=120, start_date=date(2026, 6, 1), end_date=date(2026, 7, 1)
        )
    ).generate()


@pytest.mark.unit
class TestRoster:
    def test_legacy_tier_map_is_the_v3_map(self):
        assert LEGACY_AGENT_TIERS == _LEGACY_TIERS

    def test_covers_full_legacy_agent_roster(self, frame):
        assert set(_LEGACY_TIERS) <= set(frame["agent_name"].unique())

    def test_tiers_match_legacy_map(self, frame):
        for agent, tier in _LEGACY_TIERS.items():
            tiers = frame.loc[frame["agent_name"] == agent, "agent_tier"].unique()
            assert list(tiers) == [tier], f"{agent}: {tiers} != {tier}"


@pytest.mark.unit
class TestCuratedCATEConsistency:
    """heterogeneous_optimizer analysis_results mirror the COMM-ARMS constants."""

    def test_treatment_arm_rows_carry_brand_scaled_cate(self, frame):
        het = frame[frame["agent_name"] == "heterogeneous_optimizer"]
        for brand in _BRANDS:
            rows = [
                r
                for r in het["analysis_results"]
                if r.get("brand") == brand and r.get("treatment_var") == "treatment_arm"
            ]
            assert rows, f"no heterogeneous_optimizer treatment_arm row for {brand}"
            expected = brand_scaled_cate(Brand(brand))
            assert rows[0]["cate_by_segment"] == expected

    def test_commercial_arm_rows_carry_brand_scaled_arm_cate(self, frame):
        het = frame[frame["agent_name"] == "heterogeneous_optimizer"]
        for brand in _BRANDS:
            scale = _BRAND_CATE_SCALE[Brand(brand)]
            for arm_name, spec in ARM_REGISTRY.items():
                if not spec.cate_by_segment:
                    continue  # treatment_arm handled above
                rows = [
                    r
                    for r in het["analysis_results"]
                    if r.get("brand") == brand and r.get("treatment_var") == arm_name
                ]
                assert rows, f"no het-opt row for ({brand}, {arm_name})"
                expected = {seg: round(v * scale, 4) for seg, v in spec.cate_by_segment.items()}
                assert rows[0]["cate_by_segment"] == expected

    def test_every_cate_row_names_a_registry_outcome(self, frame):
        het = frame[frame["agent_name"] == "heterogeneous_optimizer"]
        for r in het["analysis_results"]:
            arm = r.get("treatment_var")
            if arm in ARM_REGISTRY:
                assert r.get("outcome_var") in ARM_REGISTRY[arm].target_outcomes


@pytest.mark.unit
class TestCausalImpactMirrorsRegistry:
    """causal_impact rows carry the SAME content-addressed display effects as
    the causal_paths commercial-arm edges (scp_a*) — one story, two tables."""

    def test_ate_matches_causal_paths_comm_arm_effect(self, frame):
        from src.ml.synthetic.generators.causal_paths_generator import (
            _COMM_ARM_EDGES,
            _commercial_edge_rng,
        )

        ci = frame[frame["agent_name"] == "causal_impact"]
        by_key = {
            (r.get("brand"), r.get("treatment_var"), r.get("outcome_var")): r
            for r in ci["analysis_results"]
        }
        for brand in _BRANDS:
            for arm, outcome, confounders, lo, hi in _COMM_ARM_EDGES:
                row = by_key.get((brand, arm, outcome))
                assert row is not None, f"no causal_impact row for ({brand}, {arm}, {outcome})"
                rng = _commercial_edge_rng(f"arm|{brand}", arm, outcome)
                expected_effect = round(float(rng.uniform(lo, hi)), 4)
                assert row["ate_estimate"] == expected_effect
                assert row["confounders_controlled"] == list(confounders)


@pytest.mark.unit
class TestGapAnalyzerROISubstrate:
    """gap_analyzer rows fuel business_impact_roi_agent_activities
    (AVG(roi_estimate) over the last 30d, migration 044)."""

    def test_gap_rows_carry_bounded_roi(self, frame):
        gap = frame[frame["agent_name"] == "gap_analyzer"]
        assert not gap.empty
        roi = gap["roi_estimate"].dropna()
        assert not roi.empty
        # numeric(5,2): |value| < 1000
        assert (roi.abs() < 1000).all()
        assert (roi > 0).all()

    def test_gap_rows_are_brand_stamped(self, frame):
        gap = frame[frame["agent_name"] == "gap_analyzer"]
        brands = {r.get("brand") for r in gap["analysis_results"]}
        assert set(_BRANDS) <= brands


@pytest.mark.unit
class TestSchemaContract:
    def test_agent_activities_registered_in_loading_order(self):
        assert "agent_activities" in LOADING_ORDER

    def test_no_emitted_column_is_silently_dropped(self, frame):
        """CRITICAL (repo lesson): the BatchLoader TABLE_COLUMNS whitelist
        silently drops unregistered columns — every column the generator emits
        must be registered or it never reaches the DB."""
        registered = set(TABLE_COLUMNS.get("agent_activities", []))
        emitted = set(frame.columns)
        dropped = emitted - registered
        assert not dropped, f"emitted columns silently dropped by loader: {sorted(dropped)}"

    def test_registered_columns_are_all_emitted(self, frame):
        """The inverse: registration matches emission (no phantom required
        columns that would trip validate_datasets' critical_missing)."""
        registered = set(TABLE_COLUMNS.get("agent_activities", []))
        missing = registered - set(frame.columns)
        assert not missing, f"registered but never emitted: {sorted(missing)}"

    def test_provenance_and_ids(self, frame):
        assert frame["is_synthetic"].all()
        assert frame["activity_id"].is_unique
        # varchar(30) PK cap, including a frontier week prefix (5 chars)
        assert frame["activity_id"].str.len().max() <= 30
        assert frame["activity_timestamp"].notna().all()

    def test_id_prefix_namespacing_and_cap(self):
        df = AgentActivitiesGenerator(
            GeneratorConfig(seed=7, n_records=10, id_prefix="w2631")
        ).generate()
        assert df["activity_id"].str.startswith("w2631").all()
        assert df["activity_id"].str.len().max() <= 30

    def test_confidence_level_fits_numeric_4_3(self, frame):
        conf = frame["confidence_level"].dropna()
        assert ((conf > 0) & (conf < 1)).all()

    def test_workstream_and_split_enum_safe(self, frame):
        assert set(frame["workstream"].dropna().unique()) <= {"WS1", "WS2", "WS3"}
        assert set(frame["data_split"].unique()) <= {
            "train",
            "validation",
            "test",
            "holdout",
            "unassigned",
        }

    def test_timestamps_inside_config_window(self, frame):
        ts = pd.to_datetime(frame["activity_timestamp"], utc=True)
        assert (ts >= pd.Timestamp("2026-06-01", tz="UTC")).all()
        assert (ts < pd.Timestamp("2026-07-02", tz="UTC")).all()

    def test_deterministic_for_same_seed(self):
        cfg = {
            "seed": 99,
            "n_records": 30,
            "start_date": date(2026, 6, 1),
            "end_date": date(2026, 6, 8),
        }
        a = AgentActivitiesGenerator(GeneratorConfig(**cfg)).generate()
        b = AgentActivitiesGenerator(GeneratorConfig(**cfg)).generate()
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
class TestLoadPathWiring:
    def test_generate_datasets_emits_agent_activities(self):
        import importlib

        load_mod = importlib.import_module("scripts.load_synthetic_data")
        from src.ml.synthetic.config import DGPType

        small = {
            "hcp": 50,
            "patient": 200,
            "treatment": 200,
            "prediction": 50,
            "trigger": 400,
            "business_metrics": 30,
            "feature_values": 50,
        }
        datasets = load_mod.generate_datasets(sizes=small, dgp_type=DGPType.CONFOUNDED, seed=42)
        assert "agent_activities" in datasets
        aa = datasets["agent_activities"]
        assert not aa.empty
        assert bool(aa["is_synthetic"].all())
        assert set(_LEGACY_TIERS) <= set(aa["agent_name"].unique())

    def test_frontier_append_reseed_path_covers_agent_activities(self):
        """The weekly reseed (scripts/reseed_synthetic.sh default mode) must
        keep the NOW()-30d ROI KPI window populated: the weekly cohort emits
        agent_activities and the frontier filter knows its occurrence column."""
        from src.ml.synthetic import frontier_append as fa

        assert fa.OCCURRENCE_COLUMNS.get("agent_activities") == "activity_timestamp"

        from src.ml.synthetic.generators import HCPGenerator

        hcp_df = HCPGenerator(GeneratorConfig(id_prefix="scv", seed=42, n_records=60)).generate()
        cohort = fa.generate_week_cohort(date(2026, 7, 20), hcp_df)
        assert "agent_activities" in cohort
        aa = cohort["agent_activities"]
        assert not aa.empty
        ts = pd.to_datetime(aa["activity_timestamp"], utc=True)
        assert (ts >= pd.Timestamp("2026-07-20", tz="UTC")).all()
        assert (ts < pd.Timestamp("2026-07-27", tz="UTC")).all()
