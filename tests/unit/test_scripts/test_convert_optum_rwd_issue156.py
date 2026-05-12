"""Unit tests for issue #156 PR-0 (foundation): items 7 (GAP_THRESHOLDS) +
3 (Quan/Sundararajan Charlson + van Walraven Elixhauser).

Items deferred to follow-up sub-PRs:
- Item 1 (priority_tier ZIP3 decile) — PR-2
- Item 2 (influence_network_size + peer_influence_score) — PR-2
- Item 4 (weighted DQS + cost field loading) — PR-1
- Item 5 (soft enrollment filter) — PR-1
- Item 6 (payer_category + schema migration) — PR-1
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.convert_optum_rwd import (
    BIOLOGIC_DISCONT_GAP_DAYS,
    BIOLOGIC_PERSISTENCE_GAP_DAYS,
    COMORBIDITY_METHOD_DEFAULT,
    COMORBIDITY_METHODS_ALLOWED,
    GAP_THRESHOLDS,
    QUAN_CHARLSON,
    QUAN_ELIXHAUSER,
    VAN_WALRAVEN_WEIGHTS,
    OptumDataConverter,
    _resolve_gap_thresholds,
)

# --------------------------------------------------------------------------- #
# Item 7: GAP_THRESHOLDS dict (drug-class aware)                              #
# --------------------------------------------------------------------------- #


class TestGapThresholdsDict:
    """Item 7 — drug-class-aware gap thresholds.

    CRITICAL ACCEPTANCE CRITERION (from issue body):
    "CSU biologic behavior bit-for-bit unchanged"
    """

    def test_biologic_entry_matches_legacy_constants_bit_for_bit(self) -> None:
        """CSU regression guard — Xolair/Dupixent must use the historical 90/60."""
        assert GAP_THRESHOLDS["biologic"]["discontinuation"] == 90
        assert GAP_THRESHOLDS["biologic"]["persistence"] == 60
        # Cross-check against module-level constants kept for back-compat
        assert GAP_THRESHOLDS["biologic"]["discontinuation"] == BIOLOGIC_DISCONT_GAP_DAYS
        assert GAP_THRESHOLDS["biologic"]["persistence"] == BIOLOGIC_PERSISTENCE_GAP_DAYS

    def test_resolver_returns_biologic_tuple(self) -> None:
        assert _resolve_gap_thresholds("biologic") == (90, 60)

    def test_resolver_returns_oral_chronic_tuple(self) -> None:
        assert _resolve_gap_thresholds("oral_chronic") == (60, 30)

    def test_resolver_returns_specialty_injectable_tuple(self) -> None:
        assert _resolve_gap_thresholds("specialty_injectable") == (90, 60)

    def test_resolver_unknown_class_falls_back_to_default(self) -> None:
        assert _resolve_gap_thresholds("not_a_real_class") == (60, 30)
        assert _resolve_gap_thresholds("") == (60, 30)

    def test_default_entry_present_and_consistent(self) -> None:
        assert "default" in GAP_THRESHOLDS
        assert GAP_THRESHOLDS["default"]["discontinuation"] == 60
        assert GAP_THRESHOLDS["default"]["persistence"] == 30

    def test_all_entries_have_required_keys(self) -> None:
        for cls_name, entry in GAP_THRESHOLDS.items():
            assert "discontinuation" in entry, f"{cls_name} missing discontinuation"
            assert "persistence" in entry, f"{cls_name} missing persistence"
            assert isinstance(entry["discontinuation"], int)
            assert isinstance(entry["persistence"], int)
            assert entry["discontinuation"] > 0
            assert entry["persistence"] > 0


# --------------------------------------------------------------------------- #
# Item 3: Quan / Sundararajan Charlson + van Walraven Elixhauser              #
# --------------------------------------------------------------------------- #


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _converter(comorbidity_method: str = "quan") -> OptumDataConverter:
    return OptumDataConverter(
        parquet_dir=Path("."),
        output_dir=Path("."),
        cohorts=("initiation",),
        comorbidity_method=comorbidity_method,
    )


def _inpatient_fixture(codes: list[str]) -> dict[int, pd.DataFrame]:
    """Build a single-patient inpatient frame placing codes in diag1..5."""
    row: dict[str, object] = {"admit_date": _ts("2025-01-15")}
    for i, c in enumerate(codes[:5]):
        row[f"diag{i + 1}"] = c
    # Fill remaining slots with None so the column exists
    for i in range(len(codes), 5):
        row[f"diag{i + 1}"] = None
    return {1: pd.DataFrame([row])}


class TestQuanCharlson:
    def test_empty_inpatient_returns_zero(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = {}
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 0

    def test_single_mi_returns_weight_one(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["I21"])
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 1

    def test_metastatic_supersedes_any_malignancy(self) -> None:
        """Hierarchy test: C77 (metastatic, w=6) must NOT also score C50 (w=2)."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["C77", "C50"])
        # Only metastatic should count → 6 (not 6 + 2 = 8)
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 6

    def test_severe_liver_supersedes_mild_liver(self) -> None:
        """K704 (severe, w=3) excludes B18 (mild, w=1)."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["K704", "B18"])
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 3

    def test_diabetes_with_complications_supersedes_without(self) -> None:
        """E114 (with-complications, w=2) excludes E100 (without, w=1)."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["E114", "E100"])
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 2

    def test_compound_score_sums_distinct_weights(self) -> None:
        """I21 (MI, w=1) + N18 (renal, w=2) + C77 (metastatic, w=6) = 9."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["I21", "N18", "C77"])
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 9

    def test_aids_hiv_weight_six(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["B20"])
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 6

    def test_c97_any_malignancy_scores_two(self) -> None:
        """codex pass-1 HIGH-1: C97 (malignant neoplasm of multiple sites) must
        be present in any_malignancy per Quan 2005 Table 1."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["C97"])
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 2

    def test_code_outside_window_not_counted(self) -> None:
        """Admission BEFORE lb_start must not contribute."""
        conv = _converter()
        # Single record with admit_date pre-window
        conv._inpatient_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "admit_date": _ts("2020-01-01"),
                        "diag1": "I21",
                        "diag2": None,
                        "diag3": None,
                        "diag4": None,
                        "diag5": None,
                    }
                ]
            )
        }
        assert conv._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 0


class TestVanWalravenElixhauser:
    def test_empty_returns_zero(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = {}
        assert conv._elixhauser_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 0

    def test_metastatic_cancer_scores_twelve(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["C78"])
        assert conv._elixhauser_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 12

    def test_obesity_subtracts_four(self) -> None:
        """Protective category — E66 has weight -4 per van Walraven."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["E66"])
        assert conv._elixhauser_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == -4

    def test_compound_score(self) -> None:
        """CHF (7) + liver_disease (11) + obesity (-4) = 14."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["I50", "K70", "E66"])
        assert conv._elixhauser_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 14

    def test_i278_i279_chronic_pulmonary_scores_three(self) -> None:
        """codex pass-1 HIGH-2: I27.8 / I27.9 belong in chronic_pulmonary_disease
        per Quan 2005 Table 2. Van Walraven weight = +3.

        Regression guard: Quan reclassifies I27.8/I27.9 from pulmonary_circulation
        (weight 4) into chronic_pulmonary (weight 3). The pulmonary_circulation
        prefix list must enumerate I270..I277 explicitly to avoid double-counting.
        """
        for code in ("I278", "I279"):
            conv = _converter()
            conv._inpatient_by_pat = _inpatient_fixture([code])
            assert conv._elixhauser_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 3, code

    def test_i274_pulmonary_circulation_scores_four(self) -> None:
        """Counterpart regression: I274 (sub-code of pulmonary circulation, NOT
        moved to chronic pulmonary in Quan 2005) must still score weight 4."""
        conv = _converter()
        conv._inpatient_by_pat = _inpatient_fixture(["I274"])
        assert conv._elixhauser_quan(1, _ts("2024-01-01"), _ts("2025-12-31")) == 4

    def test_e12_e13_e14_diabetes_present_in_quan_mapping(self) -> None:
        """codex pass-1 MEDIUM: E12/E13/E14 (deprecated ICD-10 diabetes prefixes)
        must appear in QUAN_ELIXHAUSER diabetes categories per Quan 2005 Table 2.
        Numeric score impact is 0 (van Walraven weight = 0) but mapping must be
        faithful to the published list."""
        uncomp = QUAN_ELIXHAUSER["diabetes_uncomplicated"]
        comp = QUAN_ELIXHAUSER["diabetes_complicated"]
        # Uncomplicated terminal digits 0/1/9 for E12/E13/E14
        for prefix in ("E120", "E121", "E129", "E130", "E131", "E139", "E140", "E141", "E149"):
            assert prefix in uncomp, f"uncomp missing {prefix}"
        # Complicated terminal digits 2-8 for E12/E13/E14 — codex pass-2
        # LOW: previously only E12x was asserted. Tighten to also enforce
        # E13x and E14x complicated buckets so a regression dropping either
        # is caught.
        for prefix in (
            "E122",
            "E123",
            "E124",
            "E125",
            "E126",
            "E127",
            "E128",
            "E132",
            "E133",
            "E134",
            "E135",
            "E136",
            "E137",
            "E138",
            "E142",
            "E143",
            "E144",
            "E145",
            "E146",
            "E147",
            "E148",
        ):
            assert prefix in comp, f"comp missing {prefix}"


class TestComorbidityMethodFlag:
    """Backwards-compat: COMORBIDITY_METHOD = 'approx' must still work and
    must produce DIFFERENT values from 'quan' on a curated fixture (the issue
    body requires a 'CI parity test' that confirms the two paths diverge)."""

    def test_default_method_is_quan(self) -> None:
        assert COMORBIDITY_METHOD_DEFAULT == "quan"

    def test_allowed_methods(self) -> None:
        assert set(COMORBIDITY_METHODS_ALLOWED) == {"quan", "approx"}

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="comorbidity_method"):
            OptumDataConverter(
                parquet_dir=Path("."),
                output_dir=Path("."),
                cohorts=("initiation",),
                comorbidity_method="invalid_method",
            )

    def test_quan_and_approx_diverge_on_fixture(self) -> None:
        """Parity test required by issue #156 item 3.

        Fixture: I21 (MI) + E114 (diabetes-w/-complications) + C77 (metastatic).
        - quan path: 1 + 2 + 6 = 9
        - approx path: 3 distinct categories ({mi, diabetes (E11 prefix), cancer
          (C prefix)}) → 3
        Confirming the two methods produce DISTINCT values demonstrates the
        switch actually toggles behavior.
        """
        conv_quan = _converter(comorbidity_method="quan")
        conv_approx = _converter(comorbidity_method="approx")
        fixture = _inpatient_fixture(["I21", "E114", "C77"])
        conv_quan._inpatient_by_pat = fixture
        conv_approx._inpatient_by_pat = fixture

        quan_score = conv_quan._charlson_quan(1, _ts("2024-01-01"), _ts("2025-12-31"))
        approx_score = conv_approx._charlson_approx(1, _ts("2024-01-01"), _ts("2025-12-31"))
        assert quan_score == 9
        # approx counts present cats among {mi, chf, cancer, diabetes, renal}
        # I21 → mi; E114 → diabetes (E11 prefix matches); C77 → cancer (C prefix).
        assert approx_score == 3
        assert quan_score != approx_score


class TestQuanMappingIntegrity:
    """Defensive constants tests."""

    def test_charlson_has_seventeen_categories(self) -> None:
        # Quan 2005 Charlson has 17 categories (with hierarchies for cancer,
        # liver, diabetes — kept as separate dict entries here).
        assert len(QUAN_CHARLSON) == 17

    def test_elixhauser_categories_align_with_weights(self) -> None:
        # van Walraven weights must cover every Elixhauser category exactly.
        assert set(QUAN_ELIXHAUSER.keys()) == set(VAN_WALRAVEN_WEIGHTS.keys())

    def test_all_prefixes_upper_and_dedotted(self) -> None:
        """Optum stores ICD-10 UPPER and de-dotted. Constants must match."""
        for mapping in (QUAN_CHARLSON, QUAN_ELIXHAUSER):
            for cat, prefixes in mapping.items():
                for p in prefixes:
                    assert p == p.upper(), f"{cat}: {p!r} not upper-cased"
                    assert "." not in p, f"{cat}: {p!r} contains dot"
                    assert p, f"{cat}: empty prefix"
