"""TDD contract for the HCP-grain adoption-propensity converter.

``convert_optum_hcp_adoption.py`` turns the entity-stacked Optum mart's
``optum_hcp`` rows into the canonical tier-0 cohort contract, with:

* a binary ``adopted_target_brand`` target derived from ``adoption_status``;
* ONLY admissible, pre-adoption practice-profile features (claims-network,
  all-cause volume, specialty, geography) — every adoption-DERIVED column is
  excluded by positive enumeration (so no column the target is computed from
  can leak into the model frame);
* a stratified-random ``data_split`` (the HCP grain has no temporal index, so a
  chronological split is meaningless) that preserves the rare positive rate.

These tests pin the leakage and contract invariants on a synthetic fixture
(fast, no real data). The faithful tier0 run is the deploy arbiter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.convert_optum_hcp_adoption import (
    HCP_SAFE_FEATURES,
    HCP_TARGET,
    _LEAKY_HCP_COLS,
    assign_stratified_split,
    build_hcp_journey_records,
    select_hcp_cohort,
)

# Journey-contract keys every cohort record must carry (matches the mart cohort).
_CONTRACT_KEYS = {
    "patient_journey_id",
    "patient_id",
    "patient_hash",
    "index_date",
    "journey_start_date",
    "journey_status",
    "discontinuation_flag",
    "data_quality_score",
    HCP_TARGET,
}


def _fixture(n_adopt: int = 30, n_non: int = 270) -> pd.DataFrame:
    """A synthetic entity-stacked frame: optum_hcp rows + some other entities."""
    rng = np.random.RandomState(0)
    n = n_adopt + n_non
    base = {
        "entity_type": ["optum_hcp"] * n,
        "hcp_id": [f"H{i:05d}" for i in range(n)],
        "npi": [f"{1000000000 + i}" for i in range(n)],
        "adoption_status": ["ADOPTER"] * n_adopt + ["NON_ADOPTER"] * n_non,
        # leaky adoption-derived columns that MUST NOT survive
        "adopter_rank": rng.randint(0, 50000, n),
        "adoption_category": ["Innovator"] * n_adopt + ["NON_ADOPTER"] * n_non,
        "days_to_first": rng.randint(0, 900, n),
        "target_patient_count": rng.randint(0, 95, n),
        "target_event_count": rng.randint(0, 600, n),
        # constants that must be dropped
        "brand": ["XOLAIR"] * n,
        "is_csu_approved": [1.0] * n,
        # ids that must be dropped
        "hcp_name": [f"Dr {i}" for i in range(n)],
        "dea": [f"D{i}" for i in range(n)],
    }
    for f in HCP_SAFE_FEATURES:
        if f in ("specialty_group", "prov_type", "prov_state", "kol_category", "cred_type"):
            base[f] = rng.choice(["A", "B", "C"], n).tolist()
        else:
            base[f] = rng.gamma(2.0, 5.0, n)
    df = pd.DataFrame(base)
    # add non-HCP entities that the selector must filter out
    other = pd.DataFrame({"entity_type": ["patient", "market"], "hcp_id": [None, None]})
    return pd.concat([df, other], ignore_index=True)


class TestSelectHcpCohort:
    def test_filters_to_optum_hcp_and_derives_binary_target(self) -> None:
        df, attrition = select_hcp_cohort(_fixture(n_adopt=30, n_non=270))
        assert (df["entity_type"] == "optum_hcp").all()
        assert set(df[HCP_TARGET].unique()) <= {0, 1}
        assert df[HCP_TARGET].sum() == 30  # ADOPTER -> 1
        assert len(df) == 300
        # attrition records the funnel + positives
        steps = dict(attrition)
        assert steps["target_positives"] == 30

    def test_target_is_zero_for_non_adopter(self) -> None:
        df, _ = select_hcp_cohort(_fixture(n_adopt=0, n_non=50))
        assert df[HCP_TARGET].sum() == 0


class TestBuildHcpJourneyRecords:
    def test_records_carry_full_contract_and_all_safe_features(self) -> None:
        df, _ = select_hcp_cohort(_fixture())
        recs = build_hcp_journey_records(df)
        assert len(recs) == len(df)
        keys = set(recs[0])
        assert _CONTRACT_KEYS <= keys
        for f in HCP_SAFE_FEATURES:
            assert f in keys, f"missing admissible feature {f}"

    def test_no_leaky_or_id_column_survives(self) -> None:
        df, _ = select_hcp_cohort(_fixture())
        recs = build_hcp_journey_records(df)
        keys = set(recs[0])
        for leaky in _LEAKY_HCP_COLS:
            assert leaky not in keys, f"leaky column {leaky} leaked into model frame"
        for ident in ("npi", "hcp_name", "dea", "brand", "is_csu_approved"):
            assert ident not in keys

    def test_patient_id_surrogate_is_hcp(self) -> None:
        df, _ = select_hcp_cohort(_fixture())
        recs = build_hcp_journey_records(df)
        assert recs[0]["patient_id"].startswith("HCP_")
        assert recs[0]["patient_journey_id"].startswith("PJ_")
        # every surrogate id is unique (HCP-level isolation)
        assert len({r["patient_id"] for r in recs}) == len(recs)


class TestSafeFeatureAllowList:
    def test_allow_list_excludes_all_leaky_and_constants(self) -> None:
        assert _LEAKY_HCP_COLS, "leaky column ledger must be non-empty"
        assert not (set(HCP_SAFE_FEATURES) & set(_LEAKY_HCP_COLS))
        # adoption-derived + brand-specific columns are NOT model features
        for forbidden in (
            "adoption_status",
            "adopter_rank",
            "adoption_category",
            "adoption_cumulative_share",
            "days_to_first",
            "first_adoption_dt",
            "target_patient_count",
            "target_event_count",
            "distinct_target_code_count",
        ):
            assert forbidden not in HCP_SAFE_FEATURES
        assert HCP_TARGET not in HCP_SAFE_FEATURES


class TestStratifiedSplit:
    def test_split_assigns_all_records_and_preserves_prevalence(self) -> None:
        df, _ = select_hcp_cohort(_fixture(n_adopt=120, n_non=880))
        recs = build_hcp_journey_records(df)
        summary = assign_stratified_split(recs, target=HCP_TARGET, seed=42)
        splits = {r["data_split"] for r in recs}
        assert splits == {"train", "validation", "test", "holdout"}
        overall = sum(r[HCP_TARGET] for r in recs) / len(recs)
        for name in ("train", "validation", "test"):
            grp = [r for r in recs if r["data_split"] == name]
            prev = sum(r[HCP_TARGET] for r in grp) / len(grp)
            assert abs(prev - overall) < 0.03, f"{name} prevalence drift {prev} vs {overall}"
        # summary carries per-split counts
        assert sum(summary["counts"].values()) == len(recs)

    def test_split_is_deterministic(self) -> None:
        df, _ = select_hcp_cohort(_fixture(n_adopt=50, n_non=450))
        r1 = build_hcp_journey_records(df)
        r2 = build_hcp_journey_records(df)
        assign_stratified_split(r1, target=HCP_TARGET, seed=42)
        assign_stratified_split(r2, target=HCP_TARGET, seed=42)
        assert [r["data_split"] for r in r1] == [r["data_split"] for r in r2]


class TestGateExcludedFeatures:
    def test_referral_out_features_not_emitted(self) -> None:
        from scripts.convert_optum_hcp_adoption import _GATE_EXCLUDED_FEATURES

        assert "referral_out_patient_count" in _GATE_EXCLUDED_FEATURES
        assert "referral_out_degree" in _GATE_EXCLUDED_FEATURES
        # excluded from the emit allow-list...
        for f in _GATE_EXCLUDED_FEATURES:
            assert f not in HCP_SAFE_FEATURES
        df, _ = select_hcp_cohort(_fixture(n_adopt=3, n_non=7))
        recs = build_hcp_journey_records(df)
        for f in _GATE_EXCLUDED_FEATURES:
            assert f not in recs[0]

    def test_excluded_features_remain_admissible_in_manifest(self) -> None:
        # NOT leaks — they stay declared pre-index in the manifest (honesty:
        # the converter curates them out for a conservative-gate reason, it does
        # not relabel them as leakage).
        from src.data.manifests.optum_hcp_feature_manifest import (
            OPTUM_HCP_SAFE_FEATURES,
            optum_hcp_contract_for,
        )

        from scripts.convert_optum_hcp_adoption import _GATE_EXCLUDED_FEATURES

        for f in _GATE_EXCLUDED_FEATURES:
            assert f in OPTUM_HCP_SAFE_FEATURES
            assert optum_hcp_contract_for(f).knowable_at.is_pre_or_at_index()


class TestLog1pTransform:
    def test_count_features_are_log1p_transformed_scores_are_not(self) -> None:
        from scripts.convert_optum_hcp_adoption import _LOG1P_FEATURES

        # heavy-tailed counts must be log1p'd; bounded scores must stay raw
        assert "shared_patient_edge_count" in _LOG1P_FEATURES
        assert "referral_in_degree" in _LOG1P_FEATURES
        assert "kol_score" not in _LOG1P_FEATURES
        assert "shared_patient_kol_score_pct" not in _LOG1P_FEATURES

        df, _ = select_hcp_cohort(_fixture(n_adopt=3, n_non=3))
        # pin a known raw value on a logged + a non-logged feature
        df = df.copy()
        df["shared_patient_edge_count"] = 999.0  # log1p(999) ≈ 6.9078
        df["kol_score"] = 0.42  # bounded score, must pass through unchanged
        recs = build_hcp_journey_records(df)
        assert recs[0]["shared_patient_edge_count"] == pytest.approx(np.log1p(999.0))
        assert recs[0]["kol_score"] == pytest.approx(0.42)

    def test_log1p_is_monotone_preserving_order(self) -> None:
        # the transform must not reorder providers (ranking model integrity)
        df, _ = select_hcp_cohort(_fixture(n_adopt=5, n_non=15))
        df = df.copy()
        raw = np.arange(len(df), dtype=float) * 10.0
        df["medical_patient_count"] = raw
        recs = build_hcp_journey_records(df)
        out = [r["medical_patient_count"] for r in recs]
        assert out == sorted(out)  # strictly increasing preserved


def test_data_quality_score_is_populated_fraction() -> None:
    df, _ = select_hcp_cohort(_fixture(n_adopt=5, n_non=5))
    recs = build_hcp_journey_records(df)
    for r in recs:
        assert 0.0 <= r["data_quality_score"] <= 1.0
