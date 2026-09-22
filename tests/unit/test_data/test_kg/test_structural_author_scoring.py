"""Lane B (real-data causal estimation, 2026-09-22) — scorer on the golden
fixtures (spec §6). The committed CSU validation record
(``docs/layer4/csu_golden_validation_review_record.json``: n=31, exact 28,
missed leaks 0) is the known answer the scorer must reproduce from the
committed blind edges; a planted missed leak must fail the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from src.data.kg.ensemble_voter import LEAK_ROLES
from src.data.kg.structural_author_scoring import (
    BRIEF_FIELDS,
    golden_briefs,
    load_golden_entries,
    score_roles,
)
from src.ml.causal_role_dgp.extractor import extract_role

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
GOLDEN = FIXTURES / "causal_role_golden_set.json"
BLIND_BRIEFS = FIXTURES / "causal_role_csu_blind_briefs.json"
CSU_EDGES = FIXTURES / "causal_role_golden_set_csu_edges.json"
RECORD = PROJECT_ROOT / "docs" / "layer4" / "csu_golden_validation_review_record.json"
CSU = "CSU_remibrutinib"


def _csu_predictions() -> dict[str, str | None]:
    payload = json.loads(CSU_EDGES.read_text(encoding="utf-8"))
    out: dict[str, str | None] = {}
    for e in payload["entries"]:
        g = nx.DiGraph([tuple(x) for x in e["edges"]])
        try:
            out[f"{CSU}/{e['feature_name']}"] = extract_role(
                e["feature_node"], e["treatment_node"], e["outcome_node"], g
            )
        except ValueError:
            out[f"{CSU}/{e['feature_name']}"] = None
    return out


def test_golden_set_has_91_entries_over_three_cohorts():
    entries = load_golden_entries(GOLDEN)
    assert len(entries) == 91
    assert {e["cohort"] for e in entries} == {CSU, "PNH_fabhalta", "BC_kisqali"}


def test_blind_briefs_fixture_is_the_label_free_projection_of_the_csu_golden_entries():
    entries = load_golden_entries(GOLDEN)
    projected = golden_briefs(entries, cohort=CSU)
    assert len(projected) == 31
    assert all(set(b) == set(BRIEF_FIELDS) | {"cohort"} for b in projected)
    committed = json.loads(BLIND_BRIEFS.read_text(encoding="utf-8"))["briefs"]
    as_set = lambda rows: {tuple(r[k] for k in BRIEF_FIELDS) for r in rows}  # noqa: E731
    assert as_set(projected) == as_set(committed)
    # Label-free: no role string anywhere in a brief.
    for b in projected:
        assert "ground_truth_role" not in b and "rationale" not in b


def test_scorer_reproduces_the_committed_csu_validation_record():
    entries = [e for e in load_golden_entries(GOLDEN) if e["cohort"] == CSU]
    report = score_roles(_csu_predictions(), entries)
    record = json.loads(RECORD.read_text(encoding="utf-8"))["summary"]
    assert report.n == record["n"] == 31
    assert report.n_review == 0
    assert report.exact_role_agreement == record["exact_role_agreement"] == 28
    assert report.leak_decision_accuracy == record["leak_decision_accuracy"] == 1.0
    assert len(report.missed_leaks) == record["missed_leaks"] == 0
    assert report.gate_passed is True
    assert report.per_cohort == {
        CSU: {"n": 31, "n_scored": 31, "n_review": 0, "exact_role_agreement": 28, "missed_leaks": 0}
    }
    disagreeing = sorted(
        e["feature_name"]
        for e in entries
        if _csu_predictions()[f"{CSU}/{e['feature_name']}"] != e["ground_truth_role"]
    )
    assert disagreeing == sorted(record["disagreements"])
    # The three disagreements are intra-LEAK (mediator/collider/descendant), so
    # they are neither missed leaks nor conservative errors.
    assert report.conservative_errors == []
    lines = report.summary_lines()
    assert lines[0].startswith("PASS: gate missed_leaks == 0")


def test_planted_missed_leak_fails_the_gate():
    entries = [e for e in load_golden_entries(GOLDEN) if e["cohort"] == CSU]
    preds = _csu_predictions()
    victim = next(e for e in entries if e["ground_truth_role"] in LEAK_ROLES)
    preds[f"{CSU}/{victim['feature_name']}"] = "confounder"
    report = score_roles(preds, entries)
    assert report.gate_passed is False
    assert [m["feature_name"] for m in report.missed_leaks] == [victim["feature_name"]]
    assert report.missed_leaks[0]["derived_role"] == "confounder"
    assert report.missed_leak_rate == pytest.approx(1 / 31)
    assert report.summary_lines()[0].startswith("FAIL: gate missed_leaks == 0 — missed leaks 1")


def test_review_routed_feature_is_listed_not_scored():
    entries = [e for e in load_golden_entries(GOLDEN) if e["cohort"] == CSU]
    preds = _csu_predictions()
    first = entries[0]["feature_name"]
    preds[f"{CSU}/{first}"] = None
    report = score_roles(preds, entries)
    assert report.n == 31 and report.n_scored == 30 and report.n_review == 1
    assert report.review == [
        {"cohort": CSU, "feature_name": first, "ground_truth_role": entries[0]["ground_truth_role"]}
    ]
    assert report.gate_passed is True
    assert any("routed to review" in n for n in report.notes)


def test_missing_prediction_is_an_error_not_a_silent_skip():
    entries = [e for e in load_golden_entries(GOLDEN) if e["cohort"] == CSU]
    preds = _csu_predictions()
    del preds[f"{CSU}/{entries[0]['feature_name']}"]
    with pytest.raises(KeyError, match="no prediction"):
        score_roles(preds, entries)


def test_per_role_precision_recall_on_a_hand_set():
    entries = [
        {"cohort": "c", "feature_name": "a", "ground_truth_role": "confounder"},
        {"cohort": "c", "feature_name": "b", "ground_truth_role": "confounder"},
        {"cohort": "c", "feature_name": "m", "ground_truth_role": "mediator"},
        {"cohort": "c", "feature_name": "z", "ground_truth_role": "instrument"},
    ]
    preds = {"c/a": "confounder", "c/b": "confounder", "c/m": "confounder", "c/z": "instrument"}
    report = score_roles(preds, entries)
    by_role = {m.role: m for m in report.per_role}
    assert by_role["confounder"].precision == pytest.approx(2 / 3)
    assert by_role["confounder"].recall == 1.0
    assert by_role["mediator"].recall == 0.0 and by_role["mediator"].precision is None
    assert by_role["instrument"].precision == 1.0 and by_role["instrument"].recall == 1.0
    assert report.missed_leaks == [
        {
            "cohort": "c",
            "feature_name": "m",
            "ground_truth_role": "mediator",
            "derived_role": "confounder",
        }
    ]
    assert report.confusion["mediator"]["confounder"] == 1
    assert report.gate_passed is False
