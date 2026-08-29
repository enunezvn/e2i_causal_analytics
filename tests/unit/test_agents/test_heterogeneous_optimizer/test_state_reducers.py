"""State-channel reducer semantics for the Heterogeneous Optimizer graph.

Regression (2026-08-29, /segment-analysis): a single cross-library FAILED
warning emitted by ``uplift_analysis`` was persisted FOUR times on every run.
``warnings``/``errors`` were ``Annotated[..., operator.add]`` channels, but
``learn_policy`` and ``generate_profiles`` return ``{**state, ...}`` (the full
state, warnings included) — so each downstream node re-appended the existing
list: 1 → 2 → 4. The nodes are written as if the channel were last-value
(``profile_generator`` even builds ``[*existing_warnings, new]`` itself), so the
reducer must be idempotent under a full-state spread: append only what is new,
order-preserving.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.heterogeneous_optimizer.state import (
    HeterogeneousOptimizerState,
    append_unique,
)


def test_append_unique_appends_only_new_items_in_order():
    assert append_unique(["a"], ["a", "b"]) == ["a", "b"]
    assert append_unique(["a", "b"], ["b", "a"]) == ["a", "b"]
    assert append_unique([], ["x", "x"]) == ["x"]
    assert append_unique(["a"], []) == ["a"]


def test_append_unique_handles_none_and_unhashable_items():
    # errors is a list of dicts (unhashable) — equality, not hashing, decides.
    err = {"node": "uplift_analyzer", "error": "boom"}
    assert append_unique(None, [err]) == [err]
    assert append_unique([err], [dict(err)]) == [err]
    assert append_unique([err], [{"node": "other", "error": "x"}]) == [
        err,
        {"node": "other", "error": "x"},
    ]


def test_full_state_spread_does_not_duplicate_warnings_or_errors():
    """A node returning ``{**state, ...}`` must not re-append the channel.

    Mirrors the real topology: uplift emits one warning, then two downstream
    nodes spread the full state (learn_policy, generate_profiles).
    """
    warning = "Cross-library validation FAILED: EconML and CausalML agree only 42%"
    error = {"node": "uplift_analyzer", "error": "synthetic"}

    def emit(state):
        return {"warnings": [warning], "errors": [error], "status": "analyzing"}

    def spread_a(state):
        return {**state, "status": "optimizing"}

    def spread_b(state):
        # profile_generator style: rebuilds the list from the existing one.
        return {**state, "warnings": [*state.get("warnings", [])], "status": "completed"}

    graph = StateGraph(HeterogeneousOptimizerState)
    graph.add_node("emit", emit)
    graph.add_node("spread_a", spread_a)
    graph.add_node("spread_b", spread_b)
    graph.set_entry_point("emit")
    graph.add_edge("emit", "spread_a")
    graph.add_edge("spread_a", "spread_b")
    graph.add_edge("spread_b", END)

    out = graph.compile().invoke({"warnings": [], "errors": [], "status": "pending"})

    assert out["warnings"] == [warning]
    assert out["errors"] == [error]


def test_distinct_warnings_from_successive_nodes_all_survive():
    """Dedup must not swallow DIFFERENT warnings from different nodes."""

    def first(state):
        return {"warnings": ["w1"]}

    def second(state):
        return {**state, "warnings": ["w2"]}

    graph = StateGraph(HeterogeneousOptimizerState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.set_entry_point("first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    out = graph.compile().invoke({"warnings": [], "errors": [], "status": "pending"})
    assert out["warnings"] == ["w1", "w2"]
