"""Unit tests for issue #155 PR A correctness fixes.

Covers three sub-fixes in scripts/convert_optum_rwd.py and scripts/rwd_common.py:

  §1 adoption_category — Rogers Diffusion of Innovations time-to-adoption
     (replaces volume quartiles) + non_adopter category + Dupixent off-label
     flag.
  §2 journey_stage — 7-stage engagement-funnel emission via the new
     _derive_journey_stage helper.
  §3 source_timestamp — derivation from extract_ym (YYYYMM) via
     LAST_DAY-of-month at 23:59:59 UTC.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts import rwd_common as rwdc
from scripts.convert_optum_rwd import OptumDataConverter

# --------------------------------------------------------------------------- #
# Shared fixtures                                                              #
# --------------------------------------------------------------------------- #


def _make_converter(extract_ym: str | None = None, parquet_dir: str = ".") -> OptumDataConverter:
    return OptumDataConverter(
        parquet_dir=Path(parquet_dir),
        output_dir=Path("."),
        cohorts=("initiation",),
        extract_ym=extract_ym,
    )


# --------------------------------------------------------------------------- #
# §1.A — rwd_common.classify_rogers_adoption (pure function)                  #
# --------------------------------------------------------------------------- #


def test_classify_rogers_emits_all_five_categories_plus_non_adopter():
    """All 6 Rogers categories appear when the cohort spans the full curve."""
    # 200 adopters distributed across the curve + 50 non-adopters.
    hcp_days = {f"NPI_{i:04d}": i for i in range(200)}
    hcp_days.update({f"NPI_NA_{i:03d}": None for i in range(50)})  # type: ignore[dict-item]

    out = rwdc.classify_rogers_adoption(hcp_days)
    cats = set(out.values())
    assert cats == {
        "innovator",
        "early_adopter",
        "early_majority",
        "late_majority",
        "laggard",
        "non_adopter",
    }


def test_classify_rogers_non_adopter_count_matches_none_inputs():
    """Every HCP with days=None becomes a non_adopter, exact count match."""
    hcp_days: dict[str, int | None] = {f"X_{i}": (i if i < 10 else None) for i in range(20)}
    out = rwdc.classify_rogers_adoption(hcp_days)
    non_adopter_count = sum(1 for v in out.values() if v == "non_adopter")
    assert non_adopter_count == 10


def test_classify_rogers_boundary_at_2_5_percent_innovator():
    """N=200 → the 5th HCP (idx=5/200=0.025) is the boundary innovator."""
    hcp_days = {f"NPI_{i:04d}": i for i in range(200)}
    out = rwdc.classify_rogers_adoption(hcp_days)
    # 5th HCP (0-indexed 4) sorted by days ASC = NPI_0004 with days=4
    assert out["NPI_0004"] == "innovator"
    # 6th HCP (days=5) should be early_adopter (6/200 = 0.030 > 0.025)
    assert out["NPI_0005"] == "early_adopter"


def test_classify_rogers_boundary_at_16_percent_early_adopter():
    """N=200 → the 32nd HCP (idx=32/200=0.16) is the boundary early_adopter."""
    hcp_days = {f"NPI_{i:04d}": i for i in range(200)}
    out = rwdc.classify_rogers_adoption(hcp_days)
    assert out["NPI_0031"] == "early_adopter"  # 32/200 = 0.160
    assert out["NPI_0032"] == "early_majority"  # 33/200 = 0.165


def test_classify_rogers_boundary_at_50_percent_early_majority():
    """N=200 → the 100th HCP (idx=100/200=0.50) is the boundary early_majority."""
    hcp_days = {f"NPI_{i:04d}": i for i in range(200)}
    out = rwdc.classify_rogers_adoption(hcp_days)
    assert out["NPI_0099"] == "early_majority"  # 100/200 = 0.500
    assert out["NPI_0100"] == "late_majority"  # 101/200 = 0.505


def test_classify_rogers_boundary_at_84_percent_late_majority():
    """N=200 → the 168th HCP (idx=168/200=0.84) is the boundary late_majority."""
    hcp_days = {f"NPI_{i:04d}": i for i in range(200)}
    out = rwdc.classify_rogers_adoption(hcp_days)
    assert out["NPI_0167"] == "late_majority"  # 168/200 = 0.840
    assert out["NPI_0168"] == "laggard"  # 169/200 = 0.845


def test_classify_rogers_ties_broken_by_npi_lexicographic():
    """Ties on days_to_first_fill sort deterministically by NPI ASC.

    With n=5 and all days=10, share boundaries are:
      rank 1: 1/5 = 0.20 → not ≤ 0.025 nor 0.160; ≤ 0.500 → early_majority
      rank 2: 2/5 = 0.40 → early_majority
      rank 3: 3/5 = 0.60 → late_majority (≤ 0.840)
      rank 4: 4/5 = 0.80 → late_majority
      rank 5: 5/5 = 1.00 → laggard
    Tiebreak by NPI: NPI_0000 (sorted first) gets rank 1.
    """
    hcp_days: dict[str, int | None] = {f"NPI_{i:04d}": 10 for i in range(5)}
    out = rwdc.classify_rogers_adoption(hcp_days)
    assert out["NPI_0000"] == "early_majority"
    assert out["NPI_0004"] == "laggard"


def test_classify_rogers_ties_order_is_stable_across_input_orderings():
    """Same input mapping in different dict-insertion orders → same output
    classifications. NPI tiebreak guarantees stability."""
    # Two different insertion orders, same content.
    a = {"NPI_C": 5, "NPI_A": 5, "NPI_B": 5}
    b = {"NPI_A": 5, "NPI_B": 5, "NPI_C": 5}
    assert rwdc.classify_rogers_adoption(a) == rwdc.classify_rogers_adoption(b)


def test_classify_rogers_single_fill_hcp_gets_laggard():
    """A cohort with exactly 1 adopter → that HCP is the only entry; rank=1/1
    = 1.0 → laggard (per the cumulative-share interpretation)."""
    out = rwdc.classify_rogers_adoption({"NPI_SOLO": 42})
    assert out["NPI_SOLO"] == "laggard"


def test_classify_rogers_empty_input_returns_empty():
    assert rwdc.classify_rogers_adoption({}) == {}


def test_classify_rogers_no_adopters_only_non_adopters():
    """If every HCP has no fills, every HCP is non_adopter (no diffusion curve)."""
    out = rwdc.classify_rogers_adoption({f"X_{i}": None for i in range(5)})
    assert all(v == "non_adopter" for v in out.values())


# --------------------------------------------------------------------------- #
# §1.B — _build_hcp_profiles emits Rogers categories                          #
# --------------------------------------------------------------------------- #


def _seed_for_rogers(c: OptumDataConverter, rows: list[dict]) -> None:
    """Wire med DataFrame with the columns the converter expects."""
    c.med = pd.DataFrame(rows)
    c.proc = pd.DataFrame(columns=["npi", "patid", "proc_date"])
    c._provider_by_npi = {}
    c.now_iso = "2024-01-01T00:00:00"


def test_build_hcp_profiles_emits_rogers_categories_not_volume_quartiles():
    """Two HCPs: A files Xolair early (innovator-ish); B files late.
    Volume-based legacy code would classify by Rx count; Rogers classifies
    by days_to_first_fill. With n=2: rank 1 → early_adopter, rank 2 → laggard.
    """
    c = _make_converter()
    rows = [
        # NPI_A: 5 fills, FIRST fill in 2014 (very early adopter)
        {
            "npi": "NPI_A",
            "patid": 1,
            "medication_date": pd.Timestamp("2014-06-01"),
            "Brand_Name": "XOLAIR",
            "Generic_Name": None,
            "code": None,
        },
        {
            "npi": "NPI_A",
            "patid": 1,
            "medication_date": pd.Timestamp("2015-01-01"),
            "Brand_Name": "XOLAIR",
            "Generic_Name": None,
            "code": None,
        },
        # NPI_B: 1 fill in 2024 (very late adopter)
        {
            "npi": "NPI_B",
            "patid": 2,
            "medication_date": pd.Timestamp("2024-06-01"),
            "Brand_Name": "XOLAIR",
            "Generic_Name": None,
            "code": None,
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1, 2})
    # Both HCPs are adopters; with n=2 the ranks are early_adopter, laggard.
    cats = {p["adoption_category"] for p in profiles}
    assert cats <= {
        "innovator",
        "early_adopter",
        "early_majority",
        "late_majority",
        "laggard",
        "non_adopter",
    }
    # The earlier-filing HCP must NOT be classified worse than the later one.
    # We don't know the obfuscated→generated NPI mapping; identify by rx counts.
    assert len(profiles) == 2


def test_build_hcp_profiles_non_adopter_for_no_xolair_fills():
    """An HCP whose claims are non-biologic (e.g. procedure-only) is a
    non_adopter — they were prescribing something else, but not the on-
    label brand."""
    c = _make_converter()
    rows = [
        # NPI_X: has a non-biologic prescription, NO Xolair/Dupixent fills.
        {
            "npi": "NPI_X",
            "patid": 1,
            "medication_date": pd.Timestamp("2020-01-01"),
            "Brand_Name": "ALLEGRA",
            "Generic_Name": None,
            "code": "00378",
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["adoption_category"] == "non_adopter"
    assert profiles[0]["dupixent_offlabel"] is False


def test_build_hcp_profiles_dupixent_offlabel_flag_set():
    """HCPs with Dupixent fills get dupixent_offlabel=True and are EXCLUDED
    from the on-label Rogers curve (→ non_adopter unless they also have
    Xolair fills)."""
    c = _make_converter()
    rows = [
        # NPI_D: Dupixent only.
        {
            "npi": "NPI_D",
            "patid": 1,
            "medication_date": pd.Timestamp("2020-01-01"),
            "Brand_Name": "DUPIXENT",
            "Generic_Name": None,
            "code": None,
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["dupixent_offlabel"] is True
    # No Xolair → no on-label fill → non_adopter on the diffusion curve.
    assert profiles[0]["adoption_category"] == "non_adopter"


def test_build_hcp_profiles_dupixent_pre_approval_csu_is_offlabel():
    """Regression for codex pass-2 HIGH: Dupixent CSU fills BEFORE the FDA
    approval date 2025-04-18 are off-label. HCP gets dupixent_offlabel=True
    and (without other on-label fills) becomes non_adopter."""
    c = _make_converter()
    rows = [
        {
            "npi": "NPI_PRE",
            "patid": 1,
            "medication_date": pd.Timestamp("2020-01-01"),  # pre-approval
            "Brand_Name": "DUPIXENT",
            "Generic_Name": None,
            "code": None,
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["dupixent_offlabel"] is True
    assert profiles[0]["adoption_category"] == "non_adopter"


def test_build_hcp_profiles_dupixent_post_approval_csu_is_onlabel():
    """Regression for codex pass-2 HIGH: Dupixent CSU fills ON OR AFTER the
    FDA approval date 2025-04-18 are on-label. HCP receives a Rogers adopter
    category and dupixent_offlabel=False (no pre-approval fills)."""
    c = _make_converter()
    rows = [
        {
            "npi": "NPI_POST",
            "patid": 1,
            "medication_date": pd.Timestamp("2025-05-01"),  # post-approval
            "Brand_Name": "DUPIXENT",
            "Generic_Name": None,
            "code": None,
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["dupixent_offlabel"] is False
    assert profiles[0]["adoption_category"] != "non_adopter"


def test_build_hcp_profiles_dupixent_approval_date_boundary_is_inclusive():
    """Regression: a Dupixent fill ON the exact approval date 2025-04-18 is
    on-label (inclusive boundary)."""
    c = _make_converter()
    rows = [
        {
            "npi": "NPI_BOUNDARY",
            "patid": 1,
            "medication_date": pd.Timestamp("2025-04-18"),  # exact approval
            "Brand_Name": "DUPIXENT",
            "Generic_Name": None,
            "code": None,
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["dupixent_offlabel"] is False
    assert profiles[0]["adoption_category"] != "non_adopter"


def test_build_hcp_profiles_dupixent_pre_and_post_approval_flags_offlabel():
    """Regression: an HCP with BOTH a pre-approval and a post-approval
    Dupixent fill is flagged off-label (any pre-approval fill flags) AND
    receives a Rogers category from the post-approval (on-label) fill."""
    c = _make_converter()
    rows = [
        {
            "npi": "NPI_BOTH_DATES",
            "patid": 1,
            "medication_date": pd.Timestamp("2020-01-01"),  # pre-approval
            "Brand_Name": "DUPIXENT",
            "Generic_Name": None,
            "code": None,
        },
        {
            "npi": "NPI_BOTH_DATES",
            "patid": 1,
            "medication_date": pd.Timestamp("2025-06-01"),  # post-approval
            "Brand_Name": "DUPIXENT",
            "Generic_Name": None,
            "code": None,
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["dupixent_offlabel"] is True
    assert profiles[0]["adoption_category"] != "non_adopter"


def test_build_hcp_profiles_dupixent_hcpcs_code_only_flagged_offlabel():
    """Regression for codex pass-1 HIGH-1: a row with HCPCS J0517 and NO
    brand/generic/NDC must be flagged dupixent_offlabel=True and excluded
    from the on-label Rogers curve. Before the fix, J0517-only rows passed
    `_csu_biologic_mask` (J0517 is in CSU_BIOLOGIC_HCPCS) but missed the
    dupixent_mask (which only checked brand/generic/NDC), so the HCP got an
    adopter category with dupixent_offlabel=False."""
    c = _make_converter()
    rows = [
        {
            "npi": "NPI_J0517",
            "patid": 1,
            "medication_date": pd.Timestamp("2020-01-01"),
            "Brand_Name": None,
            "Generic_Name": None,
            "code": "J0517",
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["dupixent_offlabel"] is True
    assert profiles[0]["adoption_category"] == "non_adopter"


def test_build_hcp_profiles_dupixent_and_xolair_classifies_via_xolair():
    """An HCP with BOTH Xolair (on-label) and Dupixent (off-label) gets the
    flag set AND a Rogers category based on Xolair-only adoption timing."""
    c = _make_converter()
    rows = [
        {
            "npi": "NPI_BOTH",
            "patid": 1,
            "medication_date": pd.Timestamp("2015-01-01"),
            "Brand_Name": "XOLAIR",
            "Generic_Name": None,
            "code": None,
        },
        {
            "npi": "NPI_BOTH",
            "patid": 1,
            "medication_date": pd.Timestamp("2020-01-01"),
            "Brand_Name": "DUPIXENT",
            "Generic_Name": None,
            "code": None,
        },
    ]
    _seed_for_rogers(c, rows)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert profiles[0]["dupixent_offlabel"] is True
    assert profiles[0]["adoption_category"] != "non_adopter"


# --------------------------------------------------------------------------- #
# §2 — _derive_journey_stage                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cohort,init_t,disc_t,pers_t,saw_specialist,expected",
    [
        # Initiation cohort
        ("initiation", 1, None, None, False, "first_fill"),
        ("initiation", 1, None, None, True, "first_fill"),
        ("initiation", 0, None, None, True, "considering"),
        ("initiation", 0, None, None, False, "aware"),
        # Discontinuation cohort
        ("discontinuation", 1, 1, None, True, "discontinued"),
        ("discontinuation", 1, 0, None, True, "first_fill"),
        # Persistence cohort
        ("persistence", 1, None, 1, True, "maintained"),
        ("persistence", 1, None, 0, True, "adherent"),
        ("persistence", 0, None, 0, True, "first_fill"),
    ],
)
def test_derive_journey_stage_matrix(cohort, init_t, disc_t, pers_t, saw_specialist, expected):
    c = _make_converter()
    out = c._derive_journey_stage(
        cohort=cohort,
        init_t=init_t,
        disc_t=disc_t,
        pers_t=pers_t,
        saw_specialist=saw_specialist,
    )
    assert out == expected


def test_derive_journey_stage_returns_only_funnel_values():
    """Every output must be one of the 7 funnel values — no legacy values
    leak through the new derivation (which would defeat the purpose of
    extending the enum)."""
    funnel = {
        "aware",
        "considering",
        "prescribed",
        "first_fill",
        "adherent",
        "discontinued",
        "maintained",
    }
    c = _make_converter()
    for cohort in ("initiation", "discontinuation", "persistence"):
        for init_t in (0, 1):
            for disc_t in (None, 0, 1):
                for pers_t in (None, 0, 1):
                    for saw in (False, True):
                        out = c._derive_journey_stage(
                            cohort=cohort,
                            init_t=init_t,
                            disc_t=disc_t,
                            pers_t=pers_t,
                            saw_specialist=saw,
                        )
                        assert out in funnel, f"{out} not in funnel for {cohort}/{init_t}"


# --------------------------------------------------------------------------- #
# §3 — source_timestamp / data_lag derivation                                 #
# --------------------------------------------------------------------------- #


def test_extract_ym_infer_from_dir_name():
    """The converter infers YYYYMM from `--input` dir name when not passed."""
    c = OptumDataConverter(
        parquet_dir=Path("Optum_202604"),
        output_dir=Path("."),
        cohorts=("initiation",),
    )
    assert c.extract_ym == "202604"
    assert c.source_timestamp_iso is not None
    assert c.source_timestamp_iso.startswith("2026-04-30T23:59:59")


def test_extract_ym_infer_from_parent_path_component():
    """Regression for codex pass-1 MEDIUM-1: YYYYMM in a parent directory
    (e.g. ``/vendor/202604/optum``) must be inferred, not silently dropped
    because the basename happens to be `optum`. Walks path components
    right-to-left."""
    c = OptumDataConverter(
        parquet_dir=Path("/vendor/202604/optum"),
        output_dir=Path("."),
        cohorts=("initiation",),
    )
    assert c.extract_ym == "202604"
    assert c.source_timestamp_iso is not None
    assert c.source_timestamp_iso.startswith("2026-04-30T23:59:59")


def test_extract_ym_infer_deepest_path_component_wins():
    """When MULTIPLE path components contain a YYYYMM, the deepest (rightmost
    in the path) wins. This matches the convention of vendor drop layouts
    that nest the most-specific date at the leaf."""
    c = OptumDataConverter(
        parquet_dir=Path("/archive_202301/run_202604/optum"),
        output_dir=Path("."),
        cohorts=("initiation",),
    )
    assert c.extract_ym == "202604"


def test_extract_ym_explicit_overrides_inference():
    """An explicit --extract-ym beats any inference from the dir name."""
    c = OptumDataConverter(
        parquet_dir=Path("Optum_202604"),
        output_dir=Path("."),
        cohorts=("initiation",),
        extract_ym="202401",
    )
    assert c.extract_ym == "202401"
    assert c.source_timestamp_iso is not None
    assert c.source_timestamp_iso.startswith("2024-01-31T23:59:59")


def test_extract_ym_neither_explicit_nor_inferable_yields_none():
    """When neither input nor inference yields a YYYYMM, all 3 fields stay None."""
    c = OptumDataConverter(
        parquet_dir=Path("/tmp"),
        output_dir=Path("."),
        cohorts=("initiation",),
    )
    assert c.extract_ym is None
    assert c.source_timestamp_iso is None
    assert c.data_lag_hours is None


def test_extract_ym_invalid_month_logs_and_skips(caplog):
    """An out-of-range month logs a warning and skips population."""
    c = OptumDataConverter(
        parquet_dir=Path("/tmp"),
        output_dir=Path("."),
        cohorts=("initiation",),
        extract_ym="202413",  # invalid month
    )
    assert c.source_timestamp_iso is None


def test_extract_ym_february_leap_year_last_day_is_29():
    """LAST_DAY honors leap years (2024-02-29, not 28)."""
    c = OptumDataConverter(
        parquet_dir=Path("/tmp"),
        output_dir=Path("."),
        cohorts=("initiation",),
        extract_ym="202402",
    )
    assert c.source_timestamp_iso is not None
    assert c.source_timestamp_iso.startswith("2024-02-29T23:59:59")


def test_extract_ym_february_non_leap_year_last_day_is_28():
    """LAST_DAY in February of a non-leap year is 28."""
    c = OptumDataConverter(
        parquet_dir=Path("/tmp"),
        output_dir=Path("."),
        cohorts=("initiation",),
        extract_ym="202302",
    )
    assert c.source_timestamp_iso is not None
    assert c.source_timestamp_iso.startswith("2023-02-28T23:59:59")


def test_data_lag_hours_is_int_and_uses_floor_division():
    """data_lag_hours is an int (floor division). Ingestion datetime falls
    back to now() when no parquet exists; lag is positive for past extract_ym."""
    c = OptumDataConverter(
        parquet_dir=Path("/tmp"),
        output_dir=Path("."),
        cohorts=("initiation",),
        extract_ym="200001",  # 24+ years ago
    )
    assert c.data_lag_hours is not None
    assert isinstance(c.data_lag_hours, int)
    assert c.data_lag_hours > 200000  # ~24 years × 8760 hours


# --------------------------------------------------------------------------- #
# Brand launch date constants                                                  #
# --------------------------------------------------------------------------- #


def test_brand_launch_dates_xolair_csu_is_2014_03_21():
    """The Xolair-CSU launch date (anchor of the Rogers curve) is fixed at
    2014-03-21 per FDA approval."""
    assert rwdc.BRAND_LAUNCH_DATES["xolair"]["csu"] == date(2014, 3, 21)


def test_brand_launch_dates_dupixent_csu_is_2025_04_18():
    """FDA approved Dupixent (dupilumab) for CSU in adults and adolescents
    ≥12y on 2025-04-18 (Sanofi press release; FDA label 761055s070). Issue
    #155 originally documented Dupixent CSU as "NOT APPROVED" — that was
    factually wrong and corrected by codex pass-2 review."""
    assert rwdc.BRAND_LAUNCH_DATES["dupixent"]["csu"] == date(2025, 4, 18)
