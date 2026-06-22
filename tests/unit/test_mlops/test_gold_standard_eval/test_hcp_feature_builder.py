"""Tests for the HCP-grain cohort (HCP-T3): make_hcp_spec + FeatureBuilder load path.

The 4th gold-standard cohort is HCP brand adoption × 3 brands. These tests pin:
  1. make_hcp_spec returns the expected HCP-grain CohortSpec fields.
  2. FeatureBuilder.load_frame branches on grain == "hcp": it queries
     hcp_brand_adoption, embeds hcp_profiles, flattens the nested dict, and
     aliases consideration_date -> journey_start_date (so walk_forward._DATE_COL
     is reused untouched) while applying the brand + is_synthetic filters.
  3. build_from_frame on an HCP frame derives features ONLY from the 5 covariates
     (one-hot for specialty/geographic_region, numeric for the rest); the
     identity/temporal/split columns are NOT features.
  4. REGRESSION: the patient path (make_patient_spec) is byte-for-byte unchanged —
     it still uses KEEP_COLUMNS and still queries patient_journeys with the brand
     filter.

The async supabase client is mocked the same way the existing feature_builder
tests do (see test_load_frame_omits_brand_filter_when_brand_none): a tiny
chainable query stub whose ``execute`` returns a ``.data`` list, paginated so an
empty second page terminates the loop.
"""

import asyncio

import pandas as pd
import pytest

from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    goldstd_experiment_name,
    goldstd_model_name,
    make_hcp_spec,
    make_patient_spec,
)
from src.mlops.gold_standard_eval.feature_builder import (
    KEEP_COLUMNS,
    FeatureBuilder,
)

_HCP_COVARIATES = (
    "peer_influence_score",
    "influence_network_size",
    "years_experience",
    "specialty",
    "geographic_region",
)


# ---------------------------------------------------------------------------
# Mock async supabase client (mirrors test_load_frame_omits_brand_filter_*)
# ---------------------------------------------------------------------------
class _RecordingQuery:
    """Chainable query stub that records the filter calls and returns whatever
    page the owning ``_RecordingDB`` hands it.

    load_frame calls ``db.table(...)`` once PER page, so the "rows then empty"
    pagination state must live on the DB (shared across pages), not on a single
    query instance — otherwise every page re-serves the same rows and the
    cap-agnostic loop never terminates.
    """

    def __init__(self, recorder, page_supplier):
        self._rec = recorder
        self._page_supplier = page_supplier

    def select(self, expr, *a, **k):
        self._rec["select"].append(expr)
        return self

    def eq(self, col, val):
        self._rec["eq"].append((col, val))
        return self

    def in_(self, col, vals):
        self._rec["in_"].append((col, list(vals)))
        return self

    def lt(self, col, val):
        self._rec["lt"].append((col, val))
        return self

    def order(self, col, *a, **k):
        self._rec["order"].append(col)
        return self

    def range(self, start, end, *a, **k):
        self._rec["range"].append((start, end))
        return self

    async def execute(self):
        rows = self._page_supplier()

        class _R:
            data = rows

        return _R()


class _RecordingDB:
    def __init__(self, rows):
        # First page returns ``rows``; every subsequent page returns [] so the
        # cap-agnostic pagination loop in load_frame stops after one real page.
        self._pages = [rows] if rows else [[]]
        self.recorder = {
            "table": [],
            "select": [],
            "eq": [],
            "in_": [],
            "lt": [],
            "order": [],
            "range": [],
        }

    def _next_page(self):
        return self._pages.pop(0) if self._pages else []

    def table(self, name, *a, **k):
        self.recorder["table"].append(name)
        return _RecordingQuery(self.recorder, self._next_page)


# ---------------------------------------------------------------------------
# 1. make_hcp_spec
# ---------------------------------------------------------------------------
def test_make_hcp_spec_fields():
    spec = make_hcp_spec("Remibrutinib")
    assert spec.name == "hcp_adoption_remibrutinib"
    assert spec.target == "hcp_adoption_remibrutinib"
    assert spec.brand == "Remibrutinib"
    assert spec.label_column == "adopted"
    assert spec.grain == "hcp"
    assert spec.base_covariates == _HCP_COVARIATES


def test_make_hcp_spec_covers_all_brands_lowercased():
    for brand in BRANDS:
        spec = make_hcp_spec(brand)
        assert spec.name == f"hcp_adoption_{brand.lower()}"
        assert spec.target == f"hcp_adoption_{brand.lower()}"
        assert spec.grain == "hcp"
        assert spec.label_column == "adopted"


def test_make_hcp_spec_rejects_unknown_brand():
    with pytest.raises(ValueError):
        make_hcp_spec("Tasigna")


def test_goldstd_name_helpers_yield_hcp_names_unchanged():
    # The (cohort, brand) name helpers already produce the canonical HCP names;
    # this task must NOT modify them.
    assert (
        goldstd_model_name("hcp_adoption", "Remibrutinib")
        == "hcp_adoption_remibrutinib_goldstd_lr_v1"
    )
    assert (
        goldstd_experiment_name("hcp_adoption", "Kisqali") == "hcp_adoption_kisqali_goldstd_eval_v1"
    )


# ---------------------------------------------------------------------------
# 2. HCP load_frame: queries hcp_brand_adoption, flattens embed, aliases date
# ---------------------------------------------------------------------------
def test_hcp_load_frame_flattens_embed_and_aliases_date():
    rows = [
        {
            "hcp_id": "hcp_001",
            "consideration_date": "2024-01-01",
            "data_split": "train",
            "adopted": 1,
            "hcp_profiles": {
                "peer_influence_score": 3.2,
                "influence_network_size": 24,
                "years_experience": 12,
                "specialty": "Dermatology",
                "geographic_region": "west",
            },
        },
        {
            "hcp_id": "hcp_002",
            "consideration_date": "2024-02-01",
            "data_split": "holdout",
            "adopted": 0,
            "hcp_profiles": {
                "peer_influence_score": 1.1,
                "influence_network_size": 3,
                "years_experience": 5,
                "specialty": "Allergy",
                "geographic_region": "south",
            },
        },
    ]
    db = _RecordingDB(rows)
    spec = make_hcp_spec("Remibrutinib")
    fb = FeatureBuilder(spec)
    frame = asyncio.run(fb.load_frame(db, splits=["train", "holdout"]))

    # Queried the HCP table, NOT patient_journeys (load_frame calls table() once
    # per page — the loop needs one extra empty page to terminate, so >= 1 call,
    # all to hcp_brand_adoption).
    assert db.recorder["table"]
    assert set(db.recorder["table"]) == {"hcp_brand_adoption"}
    # consideration_date aliased to journey_start_date (walk_forward._DATE_COL).
    assert "journey_start_date" in frame.columns
    assert "consideration_date" not in frame.columns
    assert sorted(frame["journey_start_date"]) == ["2024-01-01", "2024-02-01"]
    # Split + label preserved.
    assert "data_split" in frame.columns
    assert "adopted" in frame.columns
    assert list(frame["adopted"]) == [1, 0]
    # The nested hcp_profiles dict was flattened to top-level columns and removed.
    assert "hcp_profiles" not in frame.columns
    for cov in _HCP_COVARIATES:
        assert cov in frame.columns
    assert frame.loc[frame["hcp_id"] == "hcp_001", "specialty"].iloc[0] == "Dermatology"
    assert frame.loc[frame["hcp_id"] == "hcp_002", "peer_influence_score"].iloc[0] == 1.1


def test_hcp_load_frame_applies_brand_and_synthetic_filters():
    db = _RecordingDB(rows=[])  # empty → returns empty frame, filters still recorded
    spec = make_hcp_spec("Fabhalta")
    fb = FeatureBuilder(spec)
    asyncio.run(fb.load_frame(db, splits=["train"]))

    eq_cols = dict(db.recorder["eq"])
    assert eq_cols.get("brand") == "Fabhalta"
    assert eq_cols.get("is_synthetic") is True
    # splits propagated via .in_(data_split, ...)
    assert ("data_split", ["train"]) in db.recorder["in_"]
    # PK-ordered pagination by hcp_id (the HCP grain primary key).
    assert "hcp_id" in db.recorder["order"]


def test_hcp_load_frame_handles_missing_embed_gracefully():
    # A row whose hcp_profiles embed is absent/None must not crash; the covariate
    # columns simply come back as NaN (the builder median-imputes downstream).
    rows = [
        {
            "hcp_id": "hcp_003",
            "consideration_date": "2024-03-01",
            "data_split": "train",
            "adopted": 1,
            "hcp_profiles": None,
        }
    ]
    db = _RecordingDB(rows)
    fb = FeatureBuilder(make_hcp_spec("Kisqali"))
    frame = asyncio.run(fb.load_frame(db))
    assert "journey_start_date" in frame.columns
    assert "hcp_profiles" not in frame.columns
    assert frame.loc[0, "adopted"] == 1


# ---------------------------------------------------------------------------
# 3. HCP build_from_frame: features derive from the 5 covariates only
# ---------------------------------------------------------------------------
def test_hcp_build_from_frame_features_from_covariates_only():
    spec = make_hcp_spec("Remibrutinib")
    fb = FeatureBuilder(spec)
    hcp_frame = pd.DataFrame(
        {
            "hcp_id": ["hcp_1", "hcp_2", "hcp_3", "hcp_4"],
            "journey_start_date": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
            ],
            "data_split": ["train", "train", "holdout", "train"],
            "adopted": [1, 0, 1, 0],
            "peer_influence_score": [3.2, 1.1, 2.7, 0.5],
            "influence_network_size": [24, 3, 14, 1],
            "years_experience": [12, 5, 20, 2],
            "specialty": ["Dermatology", "Allergy", "Dermatology", "Allergy"],
            "geographic_region": ["west", "south", "west", "midwest"],
        }
    )
    X, y = fb.build_from_frame(hcp_frame)

    assert list(y) == [1, 0, 1, 0]
    # Label + identity/temporal/split columns are NOT features.
    for non_feat in ("adopted", "hcp_id", "journey_start_date", "data_split"):
        assert non_feat not in X.columns
        assert not any(c.startswith(f"{non_feat}_") for c in X.columns)
    # Numeric covariates present (with their missingness flags).
    for num in ("peer_influence_score", "influence_network_size", "years_experience"):
        assert num in X.columns
    # Categorical covariates one-hot encoded.
    assert any(c.startswith("specialty_") for c in X.columns)
    assert any(c.startswith("geographic_region_") for c in X.columns)
    # No NaNs reach the model.
    assert not X.isnull().any().any()
    assert len(fb.feature_columns) == X.shape[1]


def test_hcp_keep_columns_default_is_covariates_not_patient_keep():
    fb = FeatureBuilder(make_hcp_spec("Remibrutinib"))
    assert fb.keep_columns == _HCP_COVARIATES
    # Sanity: the HCP default is NOT the patient KEEP_COLUMNS.
    assert fb.keep_columns != KEEP_COLUMNS


# ---------------------------------------------------------------------------
# 4. REGRESSION: the patient path honors the spec's base_covariates
# ---------------------------------------------------------------------------
def test_patient_path_uses_spec_base_covariates():
    # Post-T9 landmine fix, the patient path uses spec.base_covariates (not the
    # module KEEP_COLUMNS default). Pre-T11 this coincidentally equaled KEEP_COLUMNS
    # because initiation was the 3-covariate set; T11 enriched initiation to 7, so
    # keep_columns now follows the spec's 7-covariate set and DIFFERS from KEEP_COLUMNS.
    spec = make_patient_spec("initiation", "Remibrutinib")
    fb = FeatureBuilder(spec)
    assert fb.keep_columns == spec.base_covariates
    assert len(fb.keep_columns) == 7
    assert fb.keep_columns != KEEP_COLUMNS


def test_patient_load_frame_still_queries_patient_journeys_with_brand_filter():
    db = _RecordingDB(rows=[])
    fb = FeatureBuilder(make_patient_spec("initiation", "Remibrutinib"))
    asyncio.run(fb.load_frame(db, splits=["train"]))

    # Patient grain → patient_journeys, NOT hcp_brand_adoption.
    assert db.recorder["table"] == ["patient_journeys"]
    eq_cols = dict(db.recorder["eq"])
    assert eq_cols.get("brand") == "Remibrutinib"
    assert eq_cols.get("is_synthetic") is True
    assert ("data_split", ["train"]) in db.recorder["in_"]
    # PK-ordered by patient_id (the patient grain PK) — unchanged.
    assert "patient_id" in db.recorder["order"]


def test_patient_build_from_frame_unchanged_keep_columns_restriction():
    # Mirrors test_feature_builder's allowlist behavior to prove the patient FIT
    # path is byte-for-byte unchanged through the grain branch.
    fb = FeatureBuilder(make_patient_spec("initiation", "Remibrutinib"))
    raw = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "treatment_initiated": [1, 0],
            "disease_severity": [0.8, 0.2],
            "academic_hcp": [1, 0],
            "geographic_region": ["west", "south"],
            "age_group": ["45-54", "65-74"],  # not in KEEP_COLUMNS → dropped
        }
    )
    X, y = fb.build_from_frame(raw)
    assert list(y) == [1, 0]
    assert "age_group" not in X.columns
    assert not any(c.startswith("age_group") for c in X.columns)
    assert "disease_severity" in X.columns
    assert "academic_hcp" in X.columns
    assert any(c.startswith("geographic_region_") for c in X.columns)
