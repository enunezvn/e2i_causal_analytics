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
    "complement_inhibitor_status",  # #1321 Fabhalta-only modifier
    "disease_stage",  # #1321 Kisqali-only modifier
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
        ("Kisqali", ["ecog_performance_status", "disease_stage"]),
        (
            "Fabhalta",
            ["egfr", "proteinuria_g_day", "ldh_ratio", "complement_inhibitor_status"],
        ),
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


def test_complement_inhibitor_status_is_dual_role_treatment_and_fabhalta_modifier():
    """#1321 pilot: complement_inhibitor_status is wired as BOTH a treatment (the
    main-effect edge) and a Fabhalta-scoped effect-modifier (the covariate), with a
    text->0/1 derivation and a display label. Locks the full plumbing so a partial
    revert (e.g. treatment added but derivation dropped) fails loudly."""
    from src.api.routes.causal import (
        _CAUSAL_DATASET_SPECS,
        _CAUSAL_NUMERIC_COLUMNS,
        _CAUSAL_NUMERIC_DERIVATIONS,
        _COLUMN_LABELS,
        _derive_is_prior_c5,
    )

    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    assert "complement_inhibitor_status" in spec["treatment"]  # main-effect edge treatment
    assert "complement_inhibitor_status" in spec["covariate"]  # effect-modifier role
    assert "complement_inhibitor_status" in _CAUSAL_NUMERIC_COLUMNS["patient_journeys"]
    assert (
        _CAUSAL_NUMERIC_DERIVATIONS["patient_journeys"]["complement_inhibitor_status"]
        is _derive_is_prior_c5
    )
    assert "complement_inhibitor_status" in _COLUMN_LABELS
    # It is a Fabhalta covariate ONLY (dropped for the other brands / all-brands).
    assert "complement_inhibitor_status" in _BRAND_CLINICAL_COVARIATES["Fabhalta"]
    for other in ("Remibrutinib", "Kisqali"):
        assert "complement_inhibitor_status" not in _BRAND_CLINICAL_COVARIATES[other]


def test_derive_is_prior_c5_maps_switch_population_to_one():
    """'prior' (the eculizumab/ravulizumab switch population) -> 1.0; 'current' and a
    NULL off-brand cell -> 0.0. The treatment contrast is prior-vs-current."""
    from src.api.routes.causal import _derive_is_prior_c5

    assert _derive_is_prior_c5("prior") == 1.0
    assert _derive_is_prior_c5("PRIOR") == 1.0  # case-insensitive, like _derive_is_accepted
    assert _derive_is_prior_c5("current") == 0.0
    assert _derive_is_prior_c5(None) == 0.0


@pytest.mark.parametrize(
    "axis,brand,derive_name,label_fragment",
    [
        ("disease_stage", "Kisqali", "_derive_is_advanced_line", "Advanced line"),
        ("urticaria_severity_uas7", "Remibrutinib", "_derive_is_uncontrolled_csu", "Uncontrolled"),
    ],
)
def test_new_axis_is_dual_role_treatment_and_brand_modifier(
    axis, brand, derive_name, label_fragment
):
    """#1321 rollout: each new axis is wired as BOTH a treatment (the main-effect edge)
    and a brand-scoped effect-modifier (the covariate), with a derivation + display
    label — the same full plumbing as complement_inhibitor_status. A partial revert
    (treatment added but derivation dropped) fails loudly."""
    import src.api.routes.causal as causal_mod

    spec = causal_mod._CAUSAL_DATASET_SPECS["patient_journeys"]
    assert axis in spec["treatment"], f"{axis} missing from treatment (main-effect edge)"
    assert axis in spec["covariate"], f"{axis} missing from covariate (modifier role)"
    assert axis in causal_mod._CAUSAL_NUMERIC_COLUMNS["patient_journeys"]
    assert causal_mod._CAUSAL_NUMERIC_DERIVATIONS["patient_journeys"][axis] is getattr(
        causal_mod, derive_name
    )
    assert label_fragment.lower() in _COLUMN_LABELS[axis].lower()
    # It is that brand's covariate ONLY (dropped for the other brands / all-brands).
    assert axis in _BRAND_CLINICAL_COVARIATES[brand]
    for other in set(_BRAND_CLINICAL_COVARIATES) - {brand}:
        assert axis not in _BRAND_CLINICAL_COVARIATES[other], f"{axis} leaked into {other}"


def test_derive_is_advanced_line_maps_advanced_stages_to_one():
    """metastatic / stage_iv (advanced-line CDK4/6 burden) -> 1.0; earlier stages and a
    NULL off-brand cell -> 0.0."""
    from src.api.routes.causal import _derive_is_advanced_line

    assert _derive_is_advanced_line("metastatic") == 1.0
    assert _derive_is_advanced_line("STAGE_IV") == 1.0  # case-insensitive
    assert _derive_is_advanced_line("locally_advanced") == 0.0
    assert _derive_is_advanced_line("advanced") == 0.0
    assert _derive_is_advanced_line(None) == 0.0


def test_derive_is_uncontrolled_csu_maps_high_uas7_to_one():
    """UAS7 >= 28 (uncontrolled CSU) -> 1.0; controlled scores and a non-numeric / NULL
    cell -> 0.0. The threshold is the severe band of the 0-42 UAS7."""
    from src.api.routes.causal import _derive_is_uncontrolled_csu

    assert _derive_is_uncontrolled_csu(30) == 1.0
    assert _derive_is_uncontrolled_csu(28.0) == 1.0  # boundary inclusive
    assert _derive_is_uncontrolled_csu(27) == 0.0
    assert _derive_is_uncontrolled_csu(None) == 0.0
    assert _derive_is_uncontrolled_csu("n/a") == 0.0  # non-numeric -> not the axis
