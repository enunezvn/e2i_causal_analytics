"""Golden-set fixture validation for the DSPy-lane provider A/B.

The fixture at tests/fixtures/dspy_lane_golden_queries.json is the labeled
query set used by scripts/run_dspy_lane_ab.py to compare candidate LMs on the
two real intent-classification surfaces that ride DSPY_LM_MODEL:

- cognitive RAG ``IntentClassificationSignature`` (6 uppercase intents)
- chatbot ``ChatbotIntentClassificationSignature`` (9 lowercase intents)

Labels are acceptable-sets: a prediction is correct when it lands in the set.
A ``null`` per-taxonomy label excludes the query from that taxonomy's accuracy
(used for e.g. PREDICTION-coverage synthetics that have no chatbot analogue).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

dspy = pytest.importorskip("dspy")

FIXTURE_PATH = Path(__file__).parents[3] / "tests" / "fixtures" / "dspy_lane_golden_queries.json"

COGNITIVE_TAXONOMY = {
    "CAUSAL_ANALYSIS",
    "GAP_ANALYSIS",
    "PREDICTION",
    "EXPERIMENT_DESIGN",
    "EXPLANATION",
    "GENERAL",
}
CHATBOT_TAXONOMY = {
    "kpi_query",
    "causal_analysis",
    "agent_status",
    "recommendation",
    "search",
    "multi_faceted",
    "greeting",
    "help",
    "general",
}


@pytest.fixture(scope="module")
def golden() -> dict:
    assert FIXTURE_PATH.exists(), f"golden set fixture missing: {FIXTURE_PATH}"
    return json.loads(FIXTURE_PATH.read_text())


def test_fixture_schema(golden):
    assert golden["version"] >= 1
    assert set(golden["taxonomies"]) == {"cognitive_rag", "chatbot"}
    assert set(golden["taxonomies"]["cognitive_rag"]) == COGNITIVE_TAXONOMY
    assert set(golden["taxonomies"]["chatbot"]) == CHATBOT_TAXONOMY
    assert isinstance(golden["queries"], list)
    for item in golden["queries"]:
        assert set(item) >= {"id", "query", "source", "expected_cognitive", "expected_chatbot"}
        assert item["query"].strip()


def test_ids_unique(golden):
    ids = [q["id"] for q in golden["queries"]]
    assert len(ids) == len(set(ids))


def test_labels_within_taxonomies(golden):
    for item in golden["queries"]:
        cog = item["expected_cognitive"]
        chat = item["expected_chatbot"]
        assert cog is None or (cog and set(cog) <= COGNITIVE_TAXONOMY), item["id"]
        assert chat is None or (chat and set(chat) <= CHATBOT_TAXONOMY), item["id"]
        # A query scored on neither taxonomy is dead weight.
        assert cog is not None or chat is not None, item["id"]


def test_size_and_provenance(golden):
    queries = golden["queries"]
    assert len(queries) >= 35
    real = [q for q in queries if q["source"].startswith("chatbot_training_signals")]
    disproof = [q for q in queries if q["source"] == "disproof_2026-07-18"]
    assert len(real) >= 28, "golden set must stay grounded in real production queries"
    assert len(disproof) == 5


def test_cognitive_intent_coverage(golden):
    """Every cognitive intent must be reachable, else the A/B has blind spots."""
    covered = set()
    for item in golden["queries"]:
        if item["expected_cognitive"]:
            covered |= set(item["expected_cognitive"])
    assert covered == COGNITIVE_TAXONOMY


def test_ambiguous_labels_are_justified(golden):
    """Acceptable-sets wider than 2 must carry a note explaining the ambiguity."""
    for item in golden["queries"]:
        for labels in (item["expected_cognitive"], item["expected_chatbot"]):
            if labels is not None and len(labels) > 2:
                assert item.get("notes", "").strip(), item["id"]


def test_cognitive_taxonomy_matches_signature(golden):
    """Fixture taxonomy must track the real signature's enumerated intents."""
    from src.rag.cognitive_rag_dspy import IntentClassificationSignature

    desc = IntentClassificationSignature.output_fields["primary_intent"].json_schema_extra["desc"]
    from_signature = set(re.findall(r"[A-Z][A-Z_]+", desc)) - {"I"}
    assert from_signature == set(golden["taxonomies"]["cognitive_rag"])
