"""Unit tests for the pure logic in src/repositories/prompt_optimization.py.

No DB, no mocks of production behavior: these pin the pure builders the
real-DB integration suite (tests/integration/test_gepa_persistence_realdb.py)
relies on — agent profile resolution, percentage-POINT improvement math, hash
parity with the artifact saver, and instruction extraction from a real dspy
module.
"""

from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Agent profile resolution (tier/type for prompt_optimization_runs NOT NULLs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent_name", "expected"),
    [
        ("feedback_learner", (5, "deep")),
        ("feedback_learner_pattern", (5, "deep")),  # phase-derived artifact name
        ("feedback_learner_recommendation", (5, "deep")),
        ("explainer", (5, "deep")),
        ("causal_impact", (2, "hybrid")),
        ("gap_analyzer", (2, "hybrid")),
        ("heterogeneous_optimizer", (2, "hybrid")),
        ("experiment_designer", (3, "hybrid")),
        ("tool_composer", (1, "hybrid")),
    ],
)
def test_resolve_agent_profile_known_agents(agent_name, expected):
    from src.repositories.prompt_optimization import resolve_agent_profile

    assert resolve_agent_profile(agent_name) == expected


def test_resolve_agent_profile_unknown_gets_default():
    from src.repositories.prompt_optimization import (
        DEFAULT_AGENT_PROFILE,
        resolve_agent_profile,
    )

    assert resolve_agent_profile("some_new_agent") == DEFAULT_AGENT_PROFILE


def test_resolve_agent_profile_longest_prefix_wins():
    """'experiment_monitor' must NOT match the 'experiment_designer' profile:
    resolution is exact-or-derived (name == key or name startswith key + '_'),
    never a loose substring match."""
    from src.repositories.prompt_optimization import (
        DEFAULT_AGENT_PROFILE,
        resolve_agent_profile,
    )

    assert resolve_agent_profile("experiment_monitor") == DEFAULT_AGENT_PROFILE


# ---------------------------------------------------------------------------
# Improvement math: percentage POINTS (Agrawal et al. correction, not relative %)
# ---------------------------------------------------------------------------


def test_improvement_percentage_points():
    from src.repositories.prompt_optimization import improvement_percentage_points

    assert improvement_percentage_points(0.41, 0.55) == pytest.approx(14.0)
    assert improvement_percentage_points(0.5, 0.5) == pytest.approx(0.0)
    assert improvement_percentage_points(0.9, 0.6) == pytest.approx(-30.0)


def test_improvement_percentage_points_none_when_unmeasured():
    from src.repositories.prompt_optimization import improvement_percentage_points

    assert improvement_percentage_points(None, 0.5) is None
    assert improvement_percentage_points(0.5, None) is None
    assert improvement_percentage_points(None, None) is None


# ---------------------------------------------------------------------------
# Hash parity with the artifact saver
# ---------------------------------------------------------------------------


def test_instruction_hash_matches_versioning_hash():
    """The DB dedup hash MUST equal versioning.compute_instruction_hash for
    the same text, or DB rows and file artifacts would dedup differently."""
    from src.optimization.gepa.versioning import compute_instruction_hash
    from src.repositories.prompt_optimization import instruction_hash

    for text in ("", "Answer concisely.", "multi\nline\ninstruction"):
        assert instruction_hash(text) == compute_instruction_hash(text)


# ---------------------------------------------------------------------------
# Instruction extraction from a real dspy module
# ---------------------------------------------------------------------------


def test_extract_module_instructions_real_dspy_predict():
    import dspy

    from src.repositories.prompt_optimization import extract_module_instructions

    module = dspy.Predict("question -> answer")
    entries = extract_module_instructions(module)

    assert len(entries) >= 1
    name, text = entries[0]
    assert isinstance(name, str) and name
    assert isinstance(text, str) and text


def test_extract_module_instructions_tolerates_non_module():
    from src.repositories.prompt_optimization import extract_module_instructions

    assert extract_module_instructions(object()) == []
    assert extract_module_instructions(None) == []


# ---------------------------------------------------------------------------
# Measured-stats extraction (dspy DspyGEPAResult duck shape: seed idx 0 is the
# baseline candidate, best_idx the winner — verified against installed dspy
# 3.1.0 teleprompt/gepa source)
# ---------------------------------------------------------------------------


def test_extract_run_stats_from_detailed_results():
    from src.repositories.prompt_optimization import extract_run_stats

    module = SimpleNamespace(
        detailed_results=SimpleNamespace(
            val_aggregate_scores=[0.41, 0.44, 0.39, 0.55],
            best_idx=3,
            total_metric_calls=38,
            candidates=[object()] * 4,
        )
    )
    stats = extract_run_stats(module)

    assert stats["baseline_score"] == pytest.approx(0.41)
    assert stats["optimized_score"] == pytest.approx(0.55)
    assert stats["best_candidate_idx"] == 3
    assert stats["total_metric_calls"] == 38
    assert stats["num_candidates_explored"] == 4


def test_extract_run_stats_absent_detailed_results():
    """No track_stats -> no fabricated numbers: every field absent/None."""
    from src.repositories.prompt_optimization import extract_run_stats

    assert extract_run_stats(object()) == {}
    assert extract_run_stats(None) == {}
