"""Phase 2 (2026-07-13): brand-aware clinical covariate/effect-modifier selection.

After the DGP brand-gating, indication-specific clinical columns are populated only
for their own brand's rows and NULL off-brand. The causal/segment routes must select
the brand-relevant subset so a gated off-brand column never reaches EconML as NaN.
These are pure-logic unit tests (no DB, no FastAPI app state)."""

import pytest

from src.api.routes.causal import (
    _ALL_CLINICAL_COVARIATES,
    _BRAND_CLINICAL_COVARIATES,
    _COLUMN_LABELS,
    _UNIVERSAL_COVARIATES,
    _brand_scoped_covariates,
)
from src.ml.synthetic.clinical_codes import BRAND_ELIGIBILITY_FIELDS

_FULL = [
    "disease_severity",
    "age_at_diagnosis",
    "academic_hcp",
    "ecog_performance_status",
    "egfr",
    "proteinuria_g_day",
    "ldh_ratio",
    "urticaria_severity_uas7",
    "biologic_experienced",
]
_UNIVERSALS = ["disease_severity", "age_at_diagnosis", "academic_hcp"]


def test_all_brands_keeps_universals_only():
    # brand=None (all-brands cohort): every clinical column is populated for only one
    # brand, so cross-brand they are ~2/3 NULL -> excluded. Universals survive.
    assert _brand_scoped_covariates(_FULL, None) == _UNIVERSALS


@pytest.mark.parametrize(
    "brand,expected_clinical",
    [
        ("Remibrutinib", ["urticaria_severity_uas7", "biologic_experienced"]),
        ("Kisqali", ["ecog_performance_status"]),
        ("Fabhalta", ["egfr", "proteinuria_g_day", "ldh_ratio"]),
    ],
)
def test_brand_keeps_universals_plus_own_clinical(brand, expected_clinical):
    got = _brand_scoped_covariates(_FULL, brand)
    assert got[: len(_UNIVERSALS)] == _UNIVERSALS, "universals must survive, order-preserved"
    assert set(got) - set(_UNIVERSALS) == set(expected_clinical)
    # An off-brand clinical column is never kept.
    for other_brand, cols in _BRAND_CLINICAL_COVARIATES.items():
        if other_brand != brand:
            assert not (set(cols) & set(got)), f"{other_brand} clinical leaked into {brand}"


def test_order_preserved_and_nonclinical_passthrough():
    # A non-clinical column that is not a universal still passes through unchanged
    # (the helper only gates the known indication-specific clinical set).
    cols = ["geographic_region", "egfr", "engagement_score", "ecog_performance_status"]
    assert _brand_scoped_covariates(cols, "Kisqali") == [
        "geographic_region",
        "engagement_score",
        "ecog_performance_status",
    ]


def test_disease_severity_label_states_cross_indication():
    """Part C (2026-07-13): disease_severity is a UNIVERSAL confounder — one generic
    0–10 index shared identically across CSU / breast cancer / PNH, not a per-disease
    clinical instrument. Its display label MUST make the cross-indication nature
    explicit so it never reads as an indication-specific severity score next to the
    real per-brand biomarkers (UAS7 / ECOG / eGFR). Locks the honest label."""
    assert "disease_severity" in _UNIVERSAL_COVARIATES
    label = _COLUMN_LABELS["disease_severity"]
    assert "cross-indication" in label.lower(), (
        f"disease_severity label {label!r} must state its cross-indication nature"
    )
    # The per-brand indication biomarkers must NOT be relabeled as cross-indication.
    for col in _ALL_CLINICAL_COVARIATES:
        if col in _COLUMN_LABELS:
            assert "cross-indication" not in _COLUMN_LABELS[col].lower()


def test_api_map_is_subset_of_dgp_ssot():
    """Drift guard: every clinical covariate the causal route assigns to a brand
    MUST be a real eligibility field for that brand in the DGP SSOT. If the generator
    stops populating a column for a brand, this fails before the route can feed NaN
    to EconML."""
    for brand, clinical in _BRAND_CLINICAL_COVARIATES.items():
        eligible = BRAND_ELIGIBILITY_FIELDS[brand]
        assert clinical <= eligible, (
            f"{brand}: causal-route clinical covariates {set(clinical) - eligible} "
            f"are not in the DGP BRAND_ELIGIBILITY_FIELDS -> would be NULL/NaN"
        )
    # The union used for gating matches what the maps declare.
    assert _ALL_CLINICAL_COVARIATES == frozenset().union(*_BRAND_CLINICAL_COVARIATES.values())
