"""Unit tests for the Step 0 candidate-scoring pure functions (#1337).

Covers the scoring/aggregation module only — no LLM, no DB, no env. The
candidate adapters and runner are exercised by the committed benchmark run
itself (real results in review/step0_scores/).
"""

import math

import pytest

from scripts.benchmarks.routing.step0_scoring import (
    aggregate,
    confusion_matrix,
    contract_cards_from_registry,
    derive_legacy_pattern,
    disagreement_rows,
    parse_candidate_json,
    score_row,
    wilson_ci,
)

# =============================================================================
# score_row
# =============================================================================


def test_score_row_exact_match():
    s = score_row("SINGLE_AGENT", ["causal_impact"], "SINGLE_AGENT", ["causal_impact"])
    assert s.pattern_correct is True
    assert s.agents_exact is True
    assert s.agents_jaccard == 1.0


def test_score_row_wrong_pattern_partial_agents():
    s = score_row(
        "PARALLEL_DELEGATION",
        ["causal_impact", "gap_analyzer"],
        "SINGLE_AGENT",
        ["causal_impact"],
    )
    assert s.pattern_correct is False
    assert s.agents_exact is False
    assert s.agents_jaccard == pytest.approx(0.5)


def test_score_row_clarification_both_empty_agents():
    s = score_row("CLARIFICATION_NEEDED", [], "CLARIFICATION_NEEDED", [])
    assert s.pattern_correct is True
    assert s.agents_exact is True
    assert s.agents_jaccard == 1.0


def test_score_row_agent_order_and_dupes_ignored():
    s = score_row(
        "PARALLEL_DELEGATION",
        ["gap_analyzer", "causal_impact"],
        "PARALLEL_DELEGATION",
        ["causal_impact", "gap_analyzer", "causal_impact"],
    )
    assert s.agents_exact is True
    assert s.agents_jaccard == 1.0


# =============================================================================
# wilson_ci
# =============================================================================


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.49, abs=0.02)
    assert hi == pytest.approx(0.943, abs=0.02)


def test_wilson_ci_edges():
    lo0, hi0 = wilson_ci(0, 20)
    assert lo0 == 0.0
    assert hi0 < 0.25
    lo1, hi1 = wilson_ci(20, 20)
    assert hi1 == 1.0
    assert lo1 > 0.75
    assert wilson_ci(0, 0) == (0.0, 1.0)


# =============================================================================
# confusion matrix / aggregation
# =============================================================================


def _mk_rows():
    # gold, pred: 3 SA correct, 1 SA->TC miss, 1 TC correct, 1 CLAR->SA miss
    spec = [
        ("SINGLE_AGENT", "SINGLE_AGENT"),
        ("SINGLE_AGENT", "SINGLE_AGENT"),
        ("SINGLE_AGENT", "SINGLE_AGENT"),
        ("SINGLE_AGENT", "TOOL_COMPOSER"),
        ("TOOL_COMPOSER", "TOOL_COMPOSER"),
        ("CLARIFICATION_NEEDED", "SINGLE_AGENT"),
    ]
    rows = []
    for i, (g, p) in enumerate(spec):
        rows.append(
            {
                "query_id": f"bench-{i:04d}",
                "gold_pattern": g,
                "pred_pattern": p,
                "score": score_row(g, [], p, []),
                "source": "demo" if i % 2 == 0 else "authored",
                "is_followup": False,
            }
        )
    return rows


def test_confusion_matrix_counts():
    cm = confusion_matrix(_mk_rows())
    assert cm[("SINGLE_AGENT", "SINGLE_AGENT")] == 3
    assert cm[("SINGLE_AGENT", "TOOL_COMPOSER")] == 1
    assert cm[("CLARIFICATION_NEEDED", "SINGLE_AGENT")] == 1
    assert cm[("TOOL_COMPOSER", "TOOL_COMPOSER")] == 1


def test_aggregate_overall_and_per_pattern():
    agg = aggregate(_mk_rows())
    assert agg["n"] == 6
    assert agg["pattern_accuracy"] == pytest.approx(4 / 6)
    per = agg["per_pattern"]
    assert per["SINGLE_AGENT"]["recall"] == pytest.approx(3 / 4)
    assert per["TOOL_COMPOSER"]["precision"] == pytest.approx(1 / 2)
    assert per["CLARIFICATION_NEEDED"]["recall"] == 0.0
    # Wilson CI attached to the headline number
    lo, hi = agg["pattern_accuracy_ci95"]
    assert 0.0 < lo < 4 / 6 < hi < 1.0


def test_aggregate_slices_by_source():
    agg = aggregate(_mk_rows(), slice_key="source")
    assert set(agg["slices"]) == {"demo", "authored"}
    assert agg["slices"]["demo"]["n"] == 3


# =============================================================================
# derive_legacy_pattern
# =============================================================================


@pytest.mark.parametrize(
    "intent,agents,expected",
    [
        ("causal_effect", ["causal_impact"], "SINGLE_AGENT"),
        ("general", ["explainer"], "SINGLE_AGENT"),
        ("multi_faceted", ["tool_composer"], "TOOL_COMPOSER"),
        ("causal_effect", ["tool_composer"], "TOOL_COMPOSER"),
        ("causal_effect", ["causal_impact", "gap_analyzer"], "PARALLEL_DELEGATION"),
        ("general", [], "SINGLE_AGENT"),
    ],
)
def test_derive_legacy_pattern(intent, agents, expected):
    assert derive_legacy_pattern(intent, agents) == expected


# =============================================================================
# parse_candidate_json
# =============================================================================

KNOWN = frozenset({"causal_impact", "gap_analyzer", "explainer", "tool_composer"})


def test_parse_candidate_json_fenced():
    text = (
        "Here you go:\n```json\n"
        '{"routing_pattern": "SINGLE_AGENT", "target_agents": ["causal_impact"],'
        ' "confidence": 0.9}\n```'
    )
    out = parse_candidate_json(text, known_agents=KNOWN)
    assert out == {
        "routing_pattern": "SINGLE_AGENT",
        "target_agents": ["causal_impact"],
        "confidence": 0.9,
    }


def test_parse_candidate_json_bare_and_unknown_agent_dropped():
    text = '{"routing_pattern": "PARALLEL_DELEGATION", "target_agents": ["causal_impact", "nonexistent_agent"], "confidence": 1.7}'
    out = parse_candidate_json(text, known_agents=KNOWN)
    assert out["routing_pattern"] == "PARALLEL_DELEGATION"
    assert out["target_agents"] == ["causal_impact"]
    assert out["confidence"] == 1.0  # clamped


def test_parse_candidate_json_invalid_pattern_or_garbage_is_none():
    assert parse_candidate_json("not json at all", known_agents=KNOWN) is None
    bad = '{"routing_pattern": "MEGA_PATTERN", "target_agents": [], "confidence": 0.5}'
    assert parse_candidate_json(bad, known_agents=KNOWN) is None


# =============================================================================
# disagreement worksheet
# =============================================================================


def test_disagreement_rows_only_pattern_misses():
    rows = _mk_rows()
    dis = disagreement_rows({"legacy": rows})
    ids = {d["query_id"] for d in dis}
    assert ids == {"bench-0003", "bench-0005"}
    assert all("legacy" in d["candidates"] for d in dis)


# =============================================================================
# contract cards
# =============================================================================


def test_contract_cards_from_registry_compact_and_complete():
    registry = {
        "agents": {
            "causal_impact": {
                "covers": ["ATE estimation", "causal attribution"],
                "does_not_cover": ["forecasting"],
            },
            "gap_analyzer": {
                "covers": ["ROI opportunity sizing"],
                "does_not_cover": [],
            },
        }
    }
    cards = contract_cards_from_registry(registry)
    assert "causal_impact" in cards and "gap_analyzer" in cards
    assert "ATE estimation" in cards
    assert "NOT: forecasting" in cards
    # compact: one line per agent
    assert cards.count("\n") <= 3


def _nan_free(x):
    return not (isinstance(x, float) and math.isnan(x))


def test_aggregate_handles_empty_pattern_cells_without_nan():
    rows = [r for r in _mk_rows() if r["gold_pattern"] == "SINGLE_AGENT"]
    agg = aggregate(rows)
    for stats in agg["per_pattern"].values():
        assert _nan_free(stats["precision"]) and _nan_free(stats["recall"])
