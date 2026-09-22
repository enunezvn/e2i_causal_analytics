"""Lane B (real-data causal estimation, 2026-09-22) — approved review → prior.

Spec §3 Lane B item 5: an approved review's structure becomes the run's
``anchored_confounders``; unapproved machine attestations never do. The rows
are hand-built in the ``expert_reviews`` shape the CLI writes — a REAL DAG
snapshot whose hashes are computed with the production functions — so the
loader is exercised on what it actually reads: the approved edges. The repo
is an in-memory stand-in (the loader only calls ``get_by_id`` /
``get_reviews_for_estimand``). No DB, no network.
"""

from __future__ import annotations

import copy
import json
from datetime import date

import pytest

from src.causal_engine.dag_hash import compute_adjustment_set_hash, compute_dag_hash
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
REVIEW_ID = "11111111-1111-1111-1111-111111111111"


def _snapshot():
    """The assembled cohort DAG as the CLI stores it: two confounders (one
    latent-driven), an instrument, an ancestor, a leaky confounder and an
    M-structure collider; adjustment sets = minimal, then full."""
    edges = [
        ["age_at_index", T],
        ["age_at_index", Y],
        ["charlson_score", T],
        ["charlson_score", Y],
        ["U_sev", "charlson_score"],
        ["U_sev", Y],
        ["payer_category", T],
        ["family_atopy", Y],
        ["post_index_visits", T],
        ["post_index_visits", Y],
        [T, "on_therapy_90d"],
        ["U_dis", "on_therapy_90d"],
        ["U_dis", Y],
        [T, Y],
    ]
    nodes = sorted({n for e in edges for n in e})
    return {
        "nodes": nodes,
        "edges": edges,
        "treatment_nodes": [T],
        "outcome_nodes": [Y],
        "adjustment_sets": [
            ["age_at_index", "charlson_score", "post_index_visits"],
            ["age_at_index", "charlson_score", "family_atopy", "post_index_visits"],
        ],
        "latent_nodes": ["U_dis", "U_sev"],
    }


def _feature(name, role, *, leak=False, leak_source=None, review=False):
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
        "in_adjustment_set": role == "confounder" and not leak,
        "in_minimal_adjustment_set": role == "confounder" and not leak,
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
    _feature("on_therapy_90d", "collider"),
    _feature("unauthored", None, review=True),
]


def _row(
    *,
    status="approved",
    valid_until=None,
    review_type="initial_dag",
    evidence=True,
    snapshot=None,
    row_hash=None,
    sa_hash=None,
    row_adj=None,
    features=None,
    as_string=False,
    brand=None,
    treatment_variable=T,
):
    snap = snapshot if snapshot is not None else _snapshot()
    dag_hash = compute_dag_hash(causal_graph=snap)
    adj_hash = compute_adjustment_set_hash(snap.get("adjustment_sets") or [])
    snap = dict(snap, dag_version_hash=dag_hash)
    assessment = (
        {
            "structural_author": {
                "schema_version": "1",
                "manifest": "optum_mart",
                "treatment": T,
                "outcome": Y,
                "model_id": "dummy",
                "dag_version_hash": sa_hash if sa_hash is not None else dag_hash,
                "adjustment_set_hash": adj_hash,
                "adjustment_set": snap["adjustment_sets"][-1],
                "minimal_adjustment_set": snap["adjustment_sets"][0],
                "features": features if features is not None else FEATURES,
            }
        }
        if evidence
        else {"items": []}
    )
    return {
        "review_id": REVIEW_ID,
        "review_type": review_type,
        "approval_status": status,
        "valid_until": valid_until,
        "dag_version_hash": row_hash if row_hash is not None else dag_hash,
        "adjustment_set_hash": row_adj if row_adj is not None else adj_hash,
        "treatment_variable": treatment_variable,
        "outcome_variable": Y,
        "brand": brand,
        "estimand_key": estimand_key_for(brand, T, Y),
        "dag_structure_json": json.dumps(snap) if as_string else snap,
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
# Pure: approved row -> prior, derived from the snapshot
# ---------------------------------------------------------------------------


def test_approved_row_yields_prior_derived_from_the_approved_edges():
    prior = structural_prior_from_review_row(_row())
    assert isinstance(prior, ApprovedStructuralPrior)
    assert prior.review_id == REVIEW_ID
    assert prior.dag_version_hash == compute_dag_hash(causal_graph=_snapshot())
    assert prior.adjustment_set_hash == compute_adjustment_set_hash(_snapshot()["adjustment_sets"])
    assert prior.anchored_confounders == ["age_at_index", "charlson_score"]
    assert prior.instruments == ["payer_category"]
    assert prior.roles == {
        "age_at_index": "confounder",
        "charlson_score": "confounder",
        "payer_category": "instrument",
        "family_atopy": "ancestor",
        "on_therapy_90d": "collider",
    }
    # A leak-verdict feature is never a prior, even inside the approved set.
    assert prior.excluded["post_index_visits"] == "leak verdict (layer_3_high)"
    assert "post_index_visits" not in prior.roles
    assert prior.excluded["unauthored"] == "no cohort role (review only)"
    assert prior.excluded["family_atopy"] == "cohort role ancestor"
    assert prior.excluded["on_therapy_90d"] == "cohort role collider"
    assert prior.source == "review_row"
    assert (prior.manifest, prior.treatment, prior.outcome) == ("optum_mart", T, Y)
    assert prior.warnings == []


def test_json_string_columns_are_parsed():
    prior = structural_prior_from_review_row(_row(as_string=True))
    assert prior is not None and prior.anchored_confounders == ["age_at_index", "charlson_score"]


@pytest.mark.parametrize(
    "row",
    [
        _row(status="pending"),
        _row(status="rejected"),
        _row(valid_until="2020-01-01"),  # expired approval
        _row(review_type="quarterly_audit"),
        _row(review_type=None),  # untyped rows are not structural-author reviews
        _row(evidence=False),  # an ordinary gate-minted initial_dag review
    ],
    ids=[
        "pending",
        "rejected",
        "expired",
        "other_review_type",
        "untyped",
        "no_structural_evidence",
    ],
)
def test_unapproved_or_foreign_rows_are_never_a_prior(row):
    assert structural_prior_from_review_row(row, today=date(2026, 9, 22)) is None


def test_edges_that_moved_under_unchanged_hash_strings_fail_closed():
    """codex r1 HIGH 2: the hash must be RECOMPUTED from the stored snapshot,
    not compared with a copied string. Mutate an edge and keep both hash
    strings as they were."""
    row = _row()
    snap = copy.deepcopy(row["dag_structure_json"])
    snap["edges"] = [e for e in snap["edges"] if e != ["age_at_index", Y]]
    row["dag_structure_json"] = snap  # dag_version_hash strings untouched
    with pytest.raises(StructuralPriorError, match="not the hash of the stored snapshot"):
        structural_prior_from_review_row(row)


def test_adjustment_sets_that_moved_under_unchanged_hash_strings_fail_closed():
    row = _row()
    snap = copy.deepcopy(row["dag_structure_json"])
    snap["adjustment_sets"] = [["age_at_index"]]  # the DAG hash excludes adjustment sets
    row["dag_structure_json"] = snap
    with pytest.raises(StructuralPriorError, match="covariate set moved"):
        structural_prior_from_review_row(row)


def test_evidence_that_graded_another_structure_fails_closed():
    with pytest.raises(StructuralPriorError, match="structure moved after authoring"):
        structural_prior_from_review_row(_row(sa_hash="b" * 64))
    with pytest.raises(StructuralPriorError, match="not the hash of the stored snapshot"):
        structural_prior_from_review_row(_row(row_hash="b" * 64))


def test_claimed_role_the_approved_edges_do_not_derive_fails_closed():
    # The snapshot's edges make family_atopy an ancestor; the evidence claims confounder.
    feats = copy.deepcopy(FEATURES)
    next(f for f in feats if f["feature"] == "family_atopy")["cohort_role"] = "confounder"
    with pytest.raises(StructuralPriorError, match="claims 'family_atopy' is a confounder"):
        structural_prior_from_review_row(_row(features=feats))
    # A feature the approved DAG does not contain cannot be anchored either.
    feats = copy.deepcopy(FEATURES) + [_feature("ghost", "confounder")]
    with pytest.raises(StructuralPriorError, match="not in the approved DAG"):
        structural_prior_from_review_row(_row(features=feats))


def test_confounder_outside_the_approved_adjustment_sets_is_not_anchored():
    snap = _snapshot()
    snap["adjustment_sets"] = [["age_at_index"]]
    prior = structural_prior_from_review_row(_row(snapshot=snap))
    assert prior is not None
    assert prior.anchored_confounders == ["age_at_index"]
    assert prior.excluded["charlson_score"] == (
        "cohort role confounder not in the approved adjustment sets"
    )


def test_row_without_snapshot_or_with_mismatched_estimand_fails_closed():
    row = _row()
    row["dag_structure_json"] = None
    with pytest.raises(StructuralPriorError, match="no DAG snapshot"):
        structural_prior_from_review_row(row)
    with pytest.raises(StructuralPriorError, match="treatment_variable"):
        structural_prior_from_review_row(_row(treatment_variable="other_treatment"))


def test_attestations_file_is_cross_checked_when_reachable(tmp_path):
    path = tmp_path / "attestations.json"
    records = [
        {"feature_name": f["feature"], "derived_role": f["fragment_role"], "provenance": "machine"}
        for f in FEATURES
    ]
    path.write_text(json.dumps({"meta": {}, "records": records}))
    prior = structural_prior_from_review_row(_row(), attestations_path=path)
    assert prior is not None and prior.source == "review_row+attestations_file"
    records[0]["derived_role"] = "instrument"
    path.write_text(json.dumps({"meta": {}, "records": records}))
    with pytest.raises(StructuralPriorError, match="fragment role differs"):
        structural_prior_from_review_row(_row(), attestations_path=path)
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
    prior = await load_approved_structural_prior(REVIEW_ID, repo)
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
        treatment=T, outcome=Y, brand=None, manifest="optum_mart", repo_factory=factory
    )
    assert prior is not None and notes == []

    async def boom():
        raise ConnectionError("supabase down")

    prior, notes = await resolve_structural_prior_for_run(
        treatment=T, outcome=Y, brand=None, manifest="optum_mart", repo_factory=boom
    )
    assert prior is None
    assert notes == ["structural prior not consulted: ConnectionError: supabase down"]

    async def mismatched():
        return _FakeRepo([_row(row_hash="b" * 64)])

    prior, notes = await resolve_structural_prior_for_run(
        treatment=T, outcome=Y, brand=None, manifest="optum_mart", repo_factory=mismatched
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
    assert lines[0].startswith(f"structural prior: approved expert review {REVIEW_ID}")
    assert "anchors 1 confounder(s); 1 instrument(s), 4 excluded" in lines[0]
    assert lines[1] == (
        "structural prior: approved confounder(s) not among this run's covariates, "
        "not anchored: charlson_score"
    )


def test_missing_version_pair_halves_fail_closed():
    """codex r2 HIGH 1: both adjustment hashes are REQUIRED, not merely
    checked when present."""
    row = _row()
    row["adjustment_set_hash"] = None
    with pytest.raises(StructuralPriorError, match="never bound"):
        structural_prior_from_review_row(row)
    row = _row()
    del row["agent_assessment_json"]["structural_author"]["adjustment_set_hash"]
    with pytest.raises(StructuralPriorError, match="graded adjustment sets None"):
        structural_prior_from_review_row(row)
    row = _row()
    row["dag_version_hash"] = None
    with pytest.raises(StructuralPriorError, match="not the hash of the stored snapshot"):
        structural_prior_from_review_row(row)


def test_manifest_scoping_refuses_a_review_authored_for_another_contract():
    """codex r2 HIGH 3: (T, Y) names are not a dataset identity."""
    assert structural_prior_from_review_row(_row(), manifest="optum_mart") is not None
    with pytest.raises(StructuralPriorError, match="authored for manifest 'optum_mart'"):
        structural_prior_from_review_row(_row(), manifest="csu")


async def test_find_row_filters_on_manifest_and_resolver_refuses_undeclared_datasets():
    repo = _FakeRepo([_row()])
    assert (
        await find_approved_structural_prior_row(repo, treatment=T, outcome=Y, manifest="csu")
        is None
    )
    assert await find_approved_structural_prior_row(
        repo, treatment=T, outcome=Y, manifest="optum_mart"
    )

    fresh = _FakeRepo([_row()])

    async def factory():
        return fresh

    prior, notes = await resolve_structural_prior_for_run(
        treatment=T, outcome=Y, brand=None, manifest=None, repo_factory=factory
    )
    assert prior is None and notes == [
        "structural prior not applied: the dataset declares no feature_manifest_source, "
        "so no authored structure can be matched to it"
    ]
    assert fresh.calls == []  # never consulted without a manifest
    prior, notes = await resolve_structural_prior_for_run(
        treatment=T, outcome=Y, brand=None, manifest="optum_mart", repo_factory=factory
    )
    assert prior is not None and notes == []
