"""Tests for the CSU RWD loader (shard 07 §C)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic_v2 import ScenarioName, generate_scenario
from src.ml.synthetic_v2.rwd_loaders.csu_rwd import (
    RWD_PROVENANCE_TAGS,
    RwdCsuCohort,
    compute_feature_distribution_ks,
    derive_csu_remib_response_outcome,
    fail_rate,
    load_rwd_csu_cohort,
)


class TestProvenanceTags:
    def test_rwd_direct_count(self) -> None:
        direct = sum(1 for v in RWD_PROVENANCE_TAGS.values() if v == "RWD-direct")
        # Per shard 07 §C.3 claims-only RWD: ~20-22 direct features
        assert 18 <= direct <= 24

    def test_rwd_derived_count(self) -> None:
        derived = sum(1 for v in RWD_PROVENANCE_TAGS.values() if v == "RWD-derived")
        assert 6 <= derived <= 12

    def test_rwd_missing_count(self) -> None:
        missing = sum(1 for v in RWD_PROVENANCE_TAGS.values() if v == "RWD-missing")
        # Cluster 2 PRO scores + Cluster 3 biomarkers + a few others = ~28-32
        assert 28 <= missing <= 36


class TestLoadRwdCsuCohort:
    def test_synthesized_fixture_has_patients(self) -> None:
        cohort = load_rwd_csu_cohort(
            "data/does/not/exist",
            allow_synthesized_fixture=True,
        )
        assert isinstance(cohort, RwdCsuCohort)
        assert cohort.n_patients > 0
        assert cohort.outcome.shape[0] == cohort.n_patients
        assert set(cohort.outcome.tolist()) <= {0, 1}

    def test_missing_data_raises_without_fixture_flag(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_rwd_csu_cohort("data/does/not/exist", allow_synthesized_fixture=False)

    def test_real_loader_paths_raise_not_implemented(self, tmp_path) -> None:
        """Real-RWD JSON / Excel loaders deferred per shard 07 §C.4."""
        from src.ml.synthetic_v2.rwd_loaders.csu_rwd import (
            _load_from_excel,
            _load_from_json_outputs,
        )

        with pytest.raises(NotImplementedError):
            _load_from_json_outputs(tmp_path, outcome_window_weeks=12, tolerance_weeks=4)
        with pytest.raises(NotImplementedError):
            _load_from_excel(tmp_path / "x.xlsx", outcome_window_weeks=12, tolerance_weeks=4)

    def test_synthesized_fixture_provenance_tags(self) -> None:
        cohort = load_rwd_csu_cohort("data/does/not/exist", allow_synthesized_fixture=True)
        # All non-missing features in PROVENANCE_TAGS should be present in the fixture
        for name, prov in RWD_PROVENANCE_TAGS.items():
            if prov == "RWD-missing":
                continue
            assert cohort.has_feature(name), f"missing fixture feature {name!r}"

    def test_synthesized_fixture_n_patients_default(self) -> None:
        cohort = load_rwd_csu_cohort("data/does/not/exist", allow_synthesized_fixture=True)
        assert cohort.n_patients == 200


class TestRwdCsuCohortHelpers:
    def _cohort(self) -> RwdCsuCohort:
        return load_rwd_csu_cohort("data/does/not/exist", allow_synthesized_fixture=True)

    def test_has_feature_true_false(self) -> None:
        cohort = self._cohort()
        assert cohort.has_feature("sex_female") is True
        assert cohort.has_feature("baseline_uas7_total") is False  # RWD-missing

    def test_rwd_direct_or_derived_features_excludes_missing(self) -> None:
        cohort = self._cohort()
        names = cohort.rwd_direct_or_derived_features()
        for n in names:
            assert RWD_PROVENANCE_TAGS[n] in {"RWD-direct", "RWD-derived"}
        assert "baseline_uas7_total" not in names


class TestOutcomeDerivation:
    def test_no_visit_in_window_returns_none(self) -> None:
        visits = [{"week_post_remib": 4, "uas7": 5}, {"week_post_remib": 30, "uas7": 0}]
        result = derive_csu_remib_response_outcome(visits)
        assert result is None

    def test_uas7_zero_at_target_week_returns_zero(self) -> None:
        visits = [{"week_post_remib": 12, "uas7": 0}]
        assert derive_csu_remib_response_outcome(visits) == 0

    def test_uas7_nonzero_returns_one(self) -> None:
        visits = [{"week_post_remib": 12, "uas7": 8}]
        assert derive_csu_remib_response_outcome(visits) == 1

    def test_picks_nearest_visit_distinct_distances(self) -> None:
        # Week 11 (distance 1) closer than Week 14 (distance 2). Pick week 11.
        visits = [
            {"week_post_remib": 11, "uas7": 0},  # closer; UAS7=0 -> outcome=0
            {"week_post_remib": 14, "uas7": 8},  # farther
        ]
        assert derive_csu_remib_response_outcome(visits) == 0

    def test_equidistant_picks_earlier_per_codex_i4(self) -> None:
        """Tie-breaking rule (Codex I-4): equidistant visits -> earlier wins."""
        visits = [
            {"week_post_remib": 10, "uas7": 8},  # distance 2; earlier
            {"week_post_remib": 14, "uas7": 0},  # distance 2; later
        ]
        # Earlier visit (week 10, UAS7=8) wins -> outcome 1
        assert derive_csu_remib_response_outcome(visits) == 1

    def test_window_tolerance_respected(self) -> None:
        # Default tolerance ±4 around week 12 -> [8, 16]
        # Week 17 is just outside
        visits = [{"week_post_remib": 17, "uas7": 0}]
        assert derive_csu_remib_response_outcome(visits) is None

    def test_custom_window(self) -> None:
        visits = [{"week_post_remib": 24, "uas7": 0}]
        assert derive_csu_remib_response_outcome(visits, target_week=24, tolerance_weeks=2) == 0


class TestKSAndFailRate:
    def test_ks_pass_path_synthetic_close_to_rwd(self) -> None:
        """Pass path: synthetic and RWD should mostly agree (low fail rate)."""
        rwd = load_rwd_csu_cohort("data/does/not/exist", allow_synthesized_fixture=True)
        # Build synthetic_X dict from generate_scenario, indexed by feature name
        ds = generate_scenario(ScenarioName.C_TREATMENT_CSU_RESPONSE, seed=42, n_total=2000)
        # ds.X_train is post-standardization; for KS we need raw distributions.
        # Use the manifest's distributions directly, sampling at scale.
        synthetic_X = {}
        rng = np.random.default_rng(0)
        for m in ds.metadata.feature_manifest:
            if m.distribution == "normal":
                synthetic_X[m.name] = rng.normal(
                    m.distribution_params["loc"],
                    m.distribution_params["scale"],
                    size=2000,
                )
            elif m.distribution == "bernoulli":
                synthetic_X[m.name] = rng.binomial(1, m.distribution_params["p"], size=2000).astype(
                    float
                )

        ks = compute_feature_distribution_ks(synthetic_X, rwd, p_value_threshold=0.001)
        assert ks  # non-empty
        # Pass-path: most features should match (we built the fixture from
        # the same manifest distributions)
        rate = fail_rate(ks)
        assert rate <= 0.30, f"Expected fail rate <= 0.30 on pass path; got {rate}"

    def test_ks_fail_path_offset_synthetic(self) -> None:
        """Fail path: shift synthetic by 5σ so KS detects mismatch."""
        rwd = load_rwd_csu_cohort("data/does/not/exist", allow_synthesized_fixture=True)
        # Build a deliberately-offset synthetic distribution
        synthetic_X = {}
        rng = np.random.default_rng(1)
        for name in rwd.rwd_direct_or_derived_features():
            rwd_col = rwd.feature_matrix[name]
            offset = 5.0 * rwd_col.std() if rwd_col.std() > 0 else 5.0
            synthetic_X[name] = rwd_col + offset + rng.normal(0, 0.001, len(rwd_col))

        ks = compute_feature_distribution_ks(synthetic_X, rwd, p_value_threshold=0.001)
        rate = fail_rate(ks)
        assert rate >= 0.50, f"Expected fail rate >= 0.50 on fail path; got {rate}"

    def test_fail_rate_empty_returns_zero(self) -> None:
        assert fail_rate({}) == 0.0
