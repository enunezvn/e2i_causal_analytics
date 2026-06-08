"""Shard 06: PatternAnalyzerNode consumes the optimized DSPy module (closes loop)."""

from __future__ import annotations

import inspect

import pytest

dspy = pytest.importorskip("dspy")

from src.agents.feedback_learner.nodes.pattern_analyzer import PatternAnalyzerNode


def _state_with_feedback():
    return {
        "feedback_items": [
            {
                "feedback_id": "f1",
                "timestamp": "t",
                "feedback_type": "rating",
                "source_agent": "causal_impact",
                "user_feedback": 2,
                "agent_response": "",
                "metadata": {},
            },
        ],
        "feedback_summary": {"by_type": {}, "by_agent": {}},
        "status": "analyzing",
    }


@pytest.mark.asyncio
async def test_falls_back_to_deterministic_when_no_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty ./optimized_modules -> no artifact
    node = PatternAnalyzerNode(use_llm=False, prefer_optimized=True)
    out = await node.execute(_state_with_feedback())
    # No artifact -> deterministic fallback, never error.
    assert out["status"] in {"extracting", "failed"}
    assert out.get("model_used", "deterministic") == "deterministic"


def test_loader_finds_saved_artifact(tmp_path, monkeypatch):
    from src.agents.feedback_learner.dspy_integration import PatternDetectionSignature
    from src.optimization.gepa import save_optimized_module

    monkeypatch.chdir(tmp_path)
    module = dspy.ChainOfThought(PatternDetectionSignature)
    save_optimized_module(module, agent_name="feedback_learner_pattern")

    node = PatternAnalyzerNode(use_llm=False, prefer_optimized=True)
    loaded = node._load_optimized_pattern_module()
    assert loaded is not None
    assert hasattr(loaded, "predictors") or hasattr(loaded, "forward")


def test_graph_builder_accepts_prefer_optimized():
    from src.agents.feedback_learner.graph import build_feedback_learner_graph

    params = inspect.signature(build_feedback_learner_graph).parameters
    assert "prefer_optimized" in params
