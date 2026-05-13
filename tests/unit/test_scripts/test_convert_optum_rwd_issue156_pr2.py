"""Unit tests for issue #156 PR-2: items 1 (priority_tier rolling 12-mo TRx
ZIP3 decile) + 2 (influence_network_size + peer_influence_score via
shared-patient clique proxy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.convert_optum_rwd import (
    PEER_INFLUENCE_SCALE,
    PRIORITY_TIER_DECILE_MAP,
    PRIORITY_TIER_DEFAULT,
    PRIORITY_TIER_TRX_WINDOW_DAYS,
    OptumDataConverter,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _converter(**kwargs: Any) -> OptumDataConverter:
    return OptumDataConverter(
        parquet_dir=Path("."),
        output_dir=Path("."),
        cohorts=("initiation",),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Item 1: PRIORITY_TIER constants                                             #
# --------------------------------------------------------------------------- #


class TestPriorityTierConstants:
    def test_decile_to_tier_map_matches_issue_body(self) -> None:
        # decile 10 → tier 1; 8-9 → 2; 4-7 → 3; 2-3 → 4; 1 → 5.
        assert PRIORITY_TIER_DECILE_MAP[10] == 1
        assert PRIORITY_TIER_DECILE_MAP[9] == 2
        assert PRIORITY_TIER_DECILE_MAP[8] == 2
        for d in (4, 5, 6, 7):
            assert PRIORITY_TIER_DECILE_MAP[d] == 3
        for d in (2, 3):
            assert PRIORITY_TIER_DECILE_MAP[d] == 4
        assert PRIORITY_TIER_DECILE_MAP[1] == 5

    def test_default_tier_is_5(self) -> None:
        # HCPs with TRx=0 → tier 5 per issue body.
        assert PRIORITY_TIER_DEFAULT == 5

    def test_window_days_is_365(self) -> None:
        # "rolling 12-month" = 365 days.
        assert PRIORITY_TIER_TRX_WINDOW_DAYS == 365


# --------------------------------------------------------------------------- #
# Item 1: priority_tier computation                                           #
# --------------------------------------------------------------------------- #


class TestComputePriorityTiers:
    def test_zero_trx_npis_default_to_tier_5(self) -> None:
        conv = _converter()
        conv.demo = pd.DataFrame(
            [{"patid": 1, "zipcode_5": "10001"}, {"patid": 2, "zipcode_5": "10002"}]
        )
        # No medication.parquet biologic rows → zero TRx everywhere.
        conv.med = pd.DataFrame(
            columns=["patid", "npi", "medication_date", "code", "Brand_Name", "Generic_Name"]
        )
        npi_pat = {"NPI_A": {1}, "NPI_B": {2}}
        tiers, _z, trx = conv._compute_priority_tiers(
            kept_patids={1, 2}, npi_pat=npi_pat, idx_by_patid=None
        )
        assert tiers == {"NPI_A": 5, "NPI_B": 5}
        assert trx == {"NPI_A": 0, "NPI_B": 0}

    def test_top_decile_in_zip3_gets_tier_1(self) -> None:
        # Construct 10 NPIs in the same ZIP3, each with distinct biologic
        # TRx counts. The top-ranked HCP gets tier 1.
        conv = _converter()
        conv.demo = pd.DataFrame([{"patid": p, "zipcode_5": "10001"} for p in range(1, 11)])
        # Each NPI treats a distinct patient with N TRx (CSU biologic HCPCS J2357).
        rows = []
        for i in range(1, 11):
            npi = f"NPI_{i:02d}"
            for _ in range(i):  # NPI_10 has 10 fills, NPI_01 has 1 fill
                rows.append(
                    {
                        "patid": i,
                        "npi": npi,
                        "medication_date": _ts("2026-01-01"),
                        "code": "J2357",
                        "Brand_Name": "XOLAIR",
                        "Generic_Name": "omalizumab",
                    }
                )
        conv.med = pd.DataFrame(rows)
        npi_pat = {f"NPI_{i:02d}": {i} for i in range(1, 11)}
        kept = set(range(1, 11))
        # Anchor window endpoint at 2026-06-30 so the 2026-01-01 fills are inside.
        idx = {p: _ts("2026-06-30") for p in kept}
        tiers, zip3s, trx = conv._compute_priority_tiers(
            kept_patids=kept, npi_pat=npi_pat, idx_by_patid=idx
        )
        # All HCPs share ZIP3 "100"
        assert all(z == "100" for z in zip3s.values())
        # NPI_10 (highest TRx) is decile 10 → tier 1
        assert tiers["NPI_10"] == 1
        # NPI_01 (lowest non-zero TRx) is decile 1 → tier 5
        assert tiers["NPI_01"] == 5
        # All HCPs have non-zero TRx
        assert all(v > 0 for v in trx.values())

    def test_window_excludes_fills_older_than_365_days(self) -> None:
        # A biologic fill more than 365d before window endpoint should NOT
        # count toward TRx and the HCP should fall to tier 5.
        conv = _converter()
        conv.demo = pd.DataFrame([{"patid": 1, "zipcode_5": "20001"}])
        conv.med = pd.DataFrame(
            [
                {
                    "patid": 1,
                    "npi": "NPI_OLD",
                    "medication_date": _ts("2024-01-01"),  # > 365d before 2026-06-30
                    "code": "J2357",
                    "Brand_Name": "XOLAIR",
                    "Generic_Name": "omalizumab",
                }
            ]
        )
        npi_pat = {"NPI_OLD": {1}}
        idx = {1: _ts("2026-06-30")}
        tiers, _z, trx = conv._compute_priority_tiers(
            kept_patids={1}, npi_pat=npi_pat, idx_by_patid=idx
        )
        assert trx["NPI_OLD"] == 0
        assert tiers["NPI_OLD"] == 5

    def test_window_includes_fills_inside_365_days(self) -> None:
        conv = _converter()
        conv.demo = pd.DataFrame([{"patid": 1, "zipcode_5": "20001"}])
        conv.med = pd.DataFrame(
            [
                {
                    "patid": 1,
                    "npi": "NPI_RECENT",
                    "medication_date": _ts("2026-01-01"),  # < 365d before 2026-06-30
                    "code": "J2357",
                    "Brand_Name": "XOLAIR",
                    "Generic_Name": "omalizumab",
                }
            ]
        )
        npi_pat = {"NPI_RECENT": {1}}
        idx = {1: _ts("2026-06-30")}
        tiers, _z, trx = conv._compute_priority_tiers(
            kept_patids={1}, npi_pat=npi_pat, idx_by_patid=idx
        )
        assert trx["NPI_RECENT"] == 1
        # Single HCP in ZIP3 → decile 10 → tier 1
        assert tiers["NPI_RECENT"] == 1

    def test_non_biologic_fills_do_not_count(self) -> None:
        # A non-CSU-biologic NDC code should NOT be counted toward priority_tier TRx.
        conv = _converter()
        conv.demo = pd.DataFrame([{"patid": 1, "zipcode_5": "30001"}])
        conv.med = pd.DataFrame(
            [
                {
                    "patid": 1,
                    "npi": "NPI_OTHER",
                    "medication_date": _ts("2026-01-01"),
                    "code": "00378999999",  # not Xolair / Dupixent NDC prefix
                    "Brand_Name": "GENERIC",
                    "Generic_Name": "atorvastatin",
                }
            ]
        )
        npi_pat = {"NPI_OTHER": {1}}
        idx = {1: _ts("2026-06-30")}
        tiers, _z, trx = conv._compute_priority_tiers(
            kept_patids={1}, npi_pat=npi_pat, idx_by_patid=idx
        )
        assert trx["NPI_OTHER"] == 0
        assert tiers["NPI_OTHER"] == 5

    def test_missing_zip3_falls_back_to_tier_5(self) -> None:
        conv = _converter()
        # Demo has no zipcode for patid 1.
        conv.demo = pd.DataFrame([{"patid": 1, "zipcode_5": None}])
        conv.med = pd.DataFrame(
            [
                {
                    "patid": 1,
                    "npi": "NPI_NOZIP",
                    "medication_date": _ts("2026-01-01"),
                    "code": "J2357",
                    "Brand_Name": "XOLAIR",
                    "Generic_Name": "omalizumab",
                }
            ]
        )
        npi_pat = {"NPI_NOZIP": {1}}
        idx = {1: _ts("2026-06-30")}
        tiers, zip3s, trx = conv._compute_priority_tiers(
            kept_patids={1}, npi_pat=npi_pat, idx_by_patid=idx
        )
        # TRx is non-zero (filled) but ZIP3 is None → tier 5 default.
        assert trx["NPI_NOZIP"] == 1
        assert zip3s["NPI_NOZIP"] is None
        assert tiers["NPI_NOZIP"] == 5

    def test_modal_zip3_across_patient_set(self) -> None:
        # Patient 1 in ZIP3 100, patient 2 in ZIP3 100, patient 3 in ZIP3 200.
        # HCP treats all three → modal ZIP3 = 100.
        conv = _converter()
        conv.demo = pd.DataFrame(
            [
                {"patid": 1, "zipcode_5": "10001"},
                {"patid": 2, "zipcode_5": "10002"},
                {"patid": 3, "zipcode_5": "20001"},
            ]
        )
        z = conv._hcp_zip3_modal("NPI_X", {"NPI_X": {1, 2, 3}})
        assert z == "100"

    def test_zip3_tie_break_alphabetical(self) -> None:
        # 1 patient in ZIP3 100, 1 patient in ZIP3 200 → tied → alphabetical → 100.
        conv = _converter()
        conv.demo = pd.DataFrame(
            [
                {"patid": 1, "zipcode_5": "20001"},
                {"patid": 2, "zipcode_5": "10001"},
            ]
        )
        z = conv._hcp_zip3_modal("NPI_X", {"NPI_X": {1, 2}})
        assert z == "100"

    def test_ndc_distinct_tie_break(self) -> None:
        # Two NPIs in same ZIP3 with identical TRx — the one with more distinct
        # NDC codes ranks higher (gets the higher tier = lower number).
        conv = _converter()
        conv.demo = pd.DataFrame(
            [{"patid": 1, "zipcode_5": "40001"}, {"patid": 2, "zipcode_5": "40001"}]
        )
        rows = [
            # NPI_A has 2 fills both Xolair (1 distinct NDC)
            {
                "patid": 1,
                "npi": "NPI_A",
                "medication_date": _ts("2026-01-01"),
                "code": "J2357",
                "Brand_Name": "XOLAIR",
                "Generic_Name": "omalizumab",
            },
            {
                "patid": 1,
                "npi": "NPI_A",
                "medication_date": _ts("2026-02-01"),
                "code": "J2357",
                "Brand_Name": "XOLAIR",
                "Generic_Name": "omalizumab",
            },
            # NPI_B has 2 fills one Xolair + one Dupixent (2 distinct codes)
            {
                "patid": 2,
                "npi": "NPI_B",
                "medication_date": _ts("2026-01-01"),
                "code": "J2357",
                "Brand_Name": "XOLAIR",
                "Generic_Name": "omalizumab",
            },
            {
                "patid": 2,
                "npi": "NPI_B",
                "medication_date": _ts("2026-02-01"),
                "code": "J0517",
                "Brand_Name": "DUPIXENT",
                "Generic_Name": "dupilumab",
            },
        ]
        conv.med = pd.DataFrame(rows)
        npi_pat = {"NPI_A": {1}, "NPI_B": {2}}
        idx = {1: _ts("2026-06-30"), 2: _ts("2026-06-30")}
        tiers, _z, _t = conv._compute_priority_tiers(
            kept_patids={1, 2}, npi_pat=npi_pat, idx_by_patid=idx
        )
        # 2 NPIs in ZIP3 100: NPI_B has 2 distinct codes vs NPI_A's 1
        # → NPI_B ranks first → decile 10 (tier 1), NPI_A → decile 5 (tier 3)
        assert tiers["NPI_B"] < tiers["NPI_A"]
        assert tiers["NPI_B"] == 1


# --------------------------------------------------------------------------- #
# Item 2: influence_network computation                                       #
# --------------------------------------------------------------------------- #


class TestComputeInfluenceNetwork:
    def test_singleton_hcp_has_zero_degree_and_zero_score(self) -> None:
        conv = _converter()
        conv.med = pd.DataFrame(
            [{"patid": 1, "npi": "NPI_SOLO", "medication_date": _ts("2026-01-01")}]
        )
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        deg, score = conv._compute_influence_network({1})
        assert deg["NPI_SOLO"] == 0
        assert score["NPI_SOLO"] == 0.0

    def test_two_hcps_sharing_patient_form_edge(self) -> None:
        conv = _converter()
        conv.med = pd.DataFrame(
            [
                {"patid": 1, "npi": "NPI_A", "medication_date": _ts("2026-01-01")},
                {"patid": 1, "npi": "NPI_B", "medication_date": _ts("2026-02-01")},
            ]
        )
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        deg, score = conv._compute_influence_network({1})
        # Both NPIs treat patient 1 → connected; each has degree 1.
        assert deg["NPI_A"] == 1
        assert deg["NPI_B"] == 1
        # On a 2-node graph, centrality is ~0.707 each → scaled by 9.99 ≈ 7.07.
        assert score["NPI_A"] > 0.0
        assert score["NPI_B"] > 0.0
        assert score["NPI_A"] <= 9.99
        assert score["NPI_B"] <= 9.99

    def test_med_and_proc_sources_both_contribute(self) -> None:
        conv = _converter()
        conv.med = pd.DataFrame(
            [{"patid": 1, "npi": "NPI_RX", "medication_date": _ts("2026-01-01")}]
        )
        conv.proc = pd.DataFrame([{"patid": 1, "npi": "NPI_PROC", "proc_date": _ts("2026-02-01")}])
        deg, _s = conv._compute_influence_network({1})
        # Both NPIs are in patient 1's HCP set → edge present.
        assert deg["NPI_RX"] == 1
        assert deg["NPI_PROC"] == 1

    def test_hub_hcp_has_highest_degree(self) -> None:
        # Hub HCP treats all 3 patients; peripheral HCPs each treat 1 distinct patient.
        conv = _converter()
        rows = []
        for p in (1, 2, 3):
            rows.append({"patid": p, "npi": "HUB", "medication_date": _ts("2026-01-01")})
            rows.append(
                {
                    "patid": p,
                    "npi": f"PERIPH_{p}",
                    "medication_date": _ts("2026-02-01"),
                }
            )
        conv.med = pd.DataFrame(rows)
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        deg, score = conv._compute_influence_network({1, 2, 3})
        assert deg["HUB"] == 3
        assert deg["PERIPH_1"] == 1
        # Hub should also have the highest centrality.
        assert score["HUB"] >= score["PERIPH_1"]
        assert score["HUB"] >= score["PERIPH_2"]
        assert score["HUB"] >= score["PERIPH_3"]

    def test_score_clamped_to_decimal_3_2_range(self) -> None:
        conv = _converter()
        conv.med = pd.DataFrame(
            [
                {"patid": 1, "npi": "NPI_X", "medication_date": _ts("2026-01-01")},
                {"patid": 1, "npi": "NPI_Y", "medication_date": _ts("2026-02-01")},
            ]
        )
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        _d, score = conv._compute_influence_network({1})
        for v in score.values():
            assert 0.0 <= v <= 9.99, f"score {v} out of DECIMAL(3,2) range"

    def test_empty_kept_patids_returns_empty(self) -> None:
        conv = _converter()
        conv.med = pd.DataFrame(columns=["patid", "npi", "medication_date"])
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        deg, score = conv._compute_influence_network(set())
        assert deg == {}
        assert score == {}

    def test_disconnected_components_handled(self) -> None:
        # Two disjoint patient-cliques — eigenvector_centrality must not crash.
        conv = _converter()
        conv.med = pd.DataFrame(
            [
                # Component A: NPI_A1 ↔ NPI_A2 via patient 1
                {"patid": 1, "npi": "NPI_A1", "medication_date": _ts("2026-01-01")},
                {"patid": 1, "npi": "NPI_A2", "medication_date": _ts("2026-02-01")},
                # Component B: NPI_B1 ↔ NPI_B2 via patient 2
                {"patid": 2, "npi": "NPI_B1", "medication_date": _ts("2026-01-01")},
                {"patid": 2, "npi": "NPI_B2", "medication_date": _ts("2026-02-01")},
            ]
        )
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        deg, score = conv._compute_influence_network({1, 2})
        assert deg["NPI_A1"] == 1
        assert deg["NPI_A2"] == 1
        assert deg["NPI_B1"] == 1
        assert deg["NPI_B2"] == 1
        # All four get non-zero centrality on their respective edges.
        for n in ("NPI_A1", "NPI_A2", "NPI_B1", "NPI_B2"):
            assert score[n] > 0.0


# --------------------------------------------------------------------------- #
# Item 1+2: end-to-end _build_hcp_profiles wiring                             #
# --------------------------------------------------------------------------- #


class TestBuildHcpProfilesPopulatesPr2Fields:
    def _setup(self) -> OptumDataConverter:
        conv = _converter()
        conv.demo = pd.DataFrame(
            [
                {"patid": 1, "zipcode_5": "10001"},
                {"patid": 2, "zipcode_5": "10001"},
            ]
        )
        conv.med = pd.DataFrame(
            [
                {
                    "patid": 1,
                    "npi": "NPI_A",
                    "medication_date": _ts("2026-01-01"),
                    "code": "J2357",
                    "Brand_Name": "XOLAIR",
                    "Generic_Name": "omalizumab",
                },
                {
                    "patid": 2,
                    "npi": "NPI_B",
                    "medication_date": _ts("2026-02-01"),
                    "code": "J0517",
                    "Brand_Name": "DUPIXENT",
                    "Generic_Name": "dupilumab",
                },
                # Shared patient creates an A-B edge
                {
                    "patid": 1,
                    "npi": "NPI_B",
                    "medication_date": _ts("2026-03-01"),
                    "code": "J0517",
                    "Brand_Name": "DUPIXENT",
                    "Generic_Name": "dupilumab",
                },
            ]
        )
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        return conv

    def test_priority_tier_populated_on_all_rows(self) -> None:
        conv = self._setup()
        idx = {1: _ts("2026-06-30"), 2: _ts("2026-06-30")}
        profiles = conv._build_hcp_profiles({1, 2}, idx_by_patid=idx)
        assert len(profiles) >= 2
        for row in profiles:
            assert row["priority_tier"] is not None
            assert 1 <= row["priority_tier"] <= 5

    def test_influence_fields_populated_on_connected_rows(self) -> None:
        conv = self._setup()
        idx = {1: _ts("2026-06-30"), 2: _ts("2026-06-30")}
        profiles = conv._build_hcp_profiles({1, 2}, idx_by_patid=idx)
        for row in profiles:
            assert row["influence_network_size"] is not None
            assert row["influence_network_size"] >= 0
            assert row["peer_influence_score"] is not None
            assert 0.0 <= row["peer_influence_score"] <= 9.99

    def test_data_dictionary_documents_pr2_fields(self) -> None:
        conv = self._setup()
        entries = conv._build_data_dictionary("initiation")
        features = {e["feature"] for e in entries}
        assert "priority_tier" in features
        assert "influence_network_size" in features
        assert "peer_influence_score" in features
        # Influence fields MUST be documented as CLAIMS-DERIVED PROXY.
        for e in entries:
            if e["feature"] in ("influence_network_size", "peer_influence_score"):
                assert "CLAIMS-DERIVED PROXY" in e["notes"]


# --------------------------------------------------------------------------- #
# Regression guard: pre-existing _build_hcp_profiles signature still works    #
# --------------------------------------------------------------------------- #


class TestRegressionGuard:
    def test_build_hcp_profiles_works_without_idx_by_patid(self) -> None:
        # Default kwarg preserves callers that did NOT thread idx_by_patid.
        conv = _converter()
        conv.demo = pd.DataFrame([{"patid": 1, "zipcode_5": "10001"}])
        conv.med = pd.DataFrame(
            [
                {
                    "patid": 1,
                    "npi": "NPI_LEGACY",
                    "medication_date": _ts("2026-01-01"),
                    "code": "J2357",
                    "Brand_Name": "XOLAIR",
                    "Generic_Name": "omalizumab",
                }
            ]
        )
        conv.proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        # No idx_by_patid kwarg — should still run.
        profiles = conv._build_hcp_profiles({1})
        assert len(profiles) == 1
        assert profiles[0]["priority_tier"] is not None

    def test_peer_influence_scale_constant(self) -> None:
        # The scaling constant must match the DECIMAL(3,2) headroom.
        assert PEER_INFLUENCE_SCALE == 9.99
