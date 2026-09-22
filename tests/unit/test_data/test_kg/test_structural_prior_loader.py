"""Lane B (real-data causal estimation, 2026-09-22) — approved review → prior.

Spec §3 Lane B item 5: an approved review's structure becomes the run's
``anchored_confounders``; unapproved machine attestations never do. The rows
here are hand-built in the ``expert_reviews`` shape the CLI writes; the repo is
an in-memory stand-in (the loader only calls ``get_by_id`` /
``get_reviews_for_estimand``). No DB, no network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.data.kg.structural_prior_loader import (
    ApprovedStructuralPrior,
    StructuralPriorError,
    apply_structural_prior_to_state,
    find_approved_structural_prior_row,
    load_approved_structural_prior,
    resolve_structural_prior_for_run,
    structural_prior_from_review_row,
)
from src.repositories.expert_review import estimand_key_for

T, Y = "treatment_dupixent", "persistent_at_180d_g28"
HASH = "a" * 64


def _feature(name, role, *, in_set=True, leak=False, leak_source=None, review=False):
    return {
        "feature": name,
        "fragment_role": role,
        "cohort_role": role,
        "role_drift": False,
        "ambiguous": False,
        "review_required": review,
        "review_reasons": [],
        "leak_verdict": leak,
        "leak_source": leak_source,
        "panel_final_role": None,
        "edge_grades": {},
        "in_adjustment_set": in_set and role == "confounder",
        "in_minimal_adjustment_set": in_set and role == "confounder",
        "adjustment_exclusion": None,
        "provenance": "machine",
        "model_id": "dummy",
        "n_edges": 3,
    }


FEATURES = [
    _feature("age_at_index", "confounder"),
    _feature("charlson_score", "confounder"),
    _feature("payer_category", "instrument"),
    _feature("post_index_visits", "confounder", leak=True, leak_source="layer_3_high"),
    _feature("family_atopy", "ancestor"),
    _feature("unauthored", None, review=True),
]


def _row(
    *,
    status="approved",
    valid_until=None,
    review_type="initial_dag",
    evidence=True,
    row_hash=HASH,
    sa_hash=HASH,
    as_string=False,
    brand=None,
):
    assessment = (
        {
            "structural_author": {
                "schema_version": "1",
                "manifest": "optum_mart",
                "treatment": T,
                "outcome": Y,
                "model_id": "dummy",
                "dag_version_hash": sa_hash,
                "adjustment_set": ["age_at_index", "charlson_score"],
                "minimal_adjustment_set": ["age_at_index", "charlson_score"],
                "features": FEATURES,
            }
        }
        if evidence
        else {"items": []}
    )
    return {
        "review_id": "11111111-1111-1111-1111-111111111111",
        "review_type": review_type,
        "approval_status": status,
        "valid_until": valid_until,
        "dag_version_hash": row_hash,
        "treatment_variable": T,
        "outcome_variable": Y,
        "brand": brand,
        "estimand_key": estimand_key_for(brand, T, Y),
        "agent_assessment_json": json.dumps(assessment) if as_string else assessment,
    }


class _FakeRepo:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def get_by_id(self, review_id):
        self.calls.append(("get_by_id", review_id))
        return next((r for r in self.rows if r["review_id"] == review_id), None)

    async def get_reviews_for_estimand(self, estimand_key, include_expired=True):
        self.calls.append(("get_reviews_for_estimand", estimand_key, include_expired))
        return [r for r in self.rows if r["estimand_key"] == estimand_key]


# ---------------------------------------------------------------------------
# Pure: row -> prior
# ---------------------------------------------------------------------------


def test_approved_row_yields_anchored_confounders_instruments_and_exclusions():
    prior = structural_prior_from_review_row(_row())
    assert isinstance(prior, ApprovedStructuralPrior)
    assert prior.review_id == "11111111-1111-1111-1111-111111111111"
    assert prior.dag_version_hash == HASH
    assert prior.anchored_confounders == ["age_at_index", "charlson_score"]
    assert prior.instruments == ["payer_category"]
    assert prior.roles == {
        "age_at_index": "confounder",
        "charlson_score": "confounder",
        "payer_category": "instrument",
        "family_atopy": "ancestor",
    }
    # A leak-verdict feature is never a prior, whatever the authored role.
    assert prior.excluded["post_index_visits"] == "leak verdict (layer_3_high)"
    assert "post_index_visits" not in prior.roles
    assert prior.excluded["unauthored"] == "no cohort role (review only)"
    assert prior.excluded["family_atopy"] == "cohort role ancestor"
    assert prior.source == "review_row"
    assert (prior.manifest, prior.treatment, prior.outcome) == ("optum_mart", T, Y)


def test_json_string_assessment_column_is_parsed():
    prior = structural_prior_from_review_row(_row(as_string=True))
    assert prior is not None and prior.anchored_confounders == ["age_at_index", "charlson_score"]


@pytest.mark.parametrize(
    "row",
    [
        _row(status="pending"),
        _row(status="rejected"),
        _row(valid_until="2020-01-01"),  # expired approval
        _row(review_type="quarterly_audit"),
        _row(evidence=False),  # an ordinary gate-minted initial_dag review
    ],
    ids=["pending", "rejected", "expired", "other_review_type", "no_structural_evidence"],
)
def test_unapproved_or_foreign_rows_are_never_a_prior(row):
    assert structural_prior_from_review_row(row, today=date(2026, 9, 22)) is None


def test_hash_mismatch_fails_closed():
    with pytest.raises(StructuralPriorError, match="structure moved after authoring"):
        structural_prior_from_review_row(_row(row_hash="b" * 64))
    with pytest.raises(StructuralPriorError, match="structure moved"):
        structural_prior_from_review_row(_row(sa_hash=None))


def test_attestations_file_is_cross_checked_when_reachable(tmp_path):
    path = tmp_path / "attestations.json"
    records = [
        {"feature_name": f["feature"], "derived_role": f["fragment_role"], "provenance": "machine"}
        for f in FEATURES
    ]
    path.write_text(json.dumps({"meta": {}, "records": records}))
    prior = structural_prior_from_review_row(_row(), attestations_path=path)
    assert prior is not None and prior.source == "review_row+attestations_file"
    # A file that disagrees on a fragment role fails closed.
    records[0]["derived_role"] = "instrument"
    path.write_text(json.dumps({"meta": {}, "records": records}))
    with pytest.raises(StructuralPriorError, match="fragment role differs"):
        structural_prior_from_review_row(_row(), attestations_path=path)
    # A file covering a different feature set fails closed.
    path.write_text(json.dumps({"meta": {}, "records": records[:2]}))
    with pytest.raises(StructuralPriorError, match="covers 2 features"):
        structural_prior_from_review_row(_row(), attestations_path=path)


def test_unreachable_attestations_file_is_a_warning_not_a_refusal(tmp_path):
    prior = structural_prior_from_review_row(_row(), attestations_path=tmp_path / "missing.json")
    assert prior is not None
    assert prior.source == "review_row"
    assert any("not reachable" in w for w in prior.warnings)


# ---------------------------------------------------------------------------
# Async lookups against the fake repo
# ---------------------------------------------------------------------------


async def test_load_by_review_id():
    repo = _FakeRepo([_row()])
    prior = await load_approved_structural_prior("11111111-1111-1111-1111-111111111111", repo)
    assert prior is not None and prior.anchored_confounders == ["age_at_index", "charlson_score"]
    assert (
        await load_approved_structural_prior("22222222-2222-2222-2222-222222222222", repo) is None
    )


async def test_find_row_prefers_brand_key_then_falls_back_to_brandless():
    brandless = _row()
    repo = _FakeRepo([_row(status="pending", brand="Dupixent"), brandless])
    row = await find_approved_structural_prior_row(repo, treatment=T, outcome=Y, brand="Dupixent")
    assert row is not None and row["brand"] is None
    assert repo.calls[0][1] == estimand_key_for("Dupixent", T, Y)
    assert repo.calls[1][1] == estimand_key_for(None, T, Y)
    assert await find_approved_structural_prior_row(repo, treatment=T, outcome="other") is None


async def test_resolve_for_run_returns_prior_and_never_raises():
    repo = _FakeRepo([_row()])

    async def factory():
        return repo

    prior, notes = await resolve_structural_prior_for_run(
        treatment=T, outcome=Y, brand=None, repo_factory=factory
    )
    assert prior is not None and notes == []

    async def boom():
        raise ConnectionError("supabase down")

    prior, notes = await resolve_structural_prior_for_run(
        treatment=T, outcome=Y, brand=None, repo_factory=boom
    )
    assert prior is None
    assert notes == ["structural prior not consulted: ConnectionError: supabase down"]

    async def mismatched():
        return _FakeRepo([_row(row_hash="b" * 64)])

    prior, notes = await resolve_structural_prior_for_run(
        treatment=T, outcome=Y, brand=None, repo_factory=mismatched
    )
    assert prior is None
    assert notes and notes[0].startswith("structural prior refused (fail closed)")


# ---------------------------------------------------------------------------
# State application
# ---------------------------------------------------------------------------


def test_apply_to_state_restricts_anchors_to_declared_covariates_and_records_provenance():
    prior = structural_prior_from_review_row(_row())
    state = {"anchored_confounders": [], "warnings": []}
    lines = apply_structural_prior_to_state(
        state, prior, covariates=["age_at_index", "payer_category=commercial", "family_atopy"]
    )
    assert state["anchored_confounders"] == ["age_at_index"]
    assert state["approved_structure_roles"] == prior.roles
    assert state["warnings"] == lines
    assert lines[0].startswith("structural prior: approved expert review 11111111")
    assert "anchors 1 confounder(s); 1 instrument(s), 3 excluded" in lines[0]
    assert lines[1] == (
        "structural prior: approved confounder(s) not among this run's covariates, "
        "not anchored: charlson_score"
    )
