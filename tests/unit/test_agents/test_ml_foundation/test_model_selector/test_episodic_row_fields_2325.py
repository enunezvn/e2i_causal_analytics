"""#2325: the model_selector memory rows must record the selection the agent made.

``store_model_selection`` read ``algorithm_name`` / ``algorithm_family`` / ``selection_score``
/ ``primary_reason`` at the top level of the agent output, where they never are: ``run()``
nests them under ``model_candidate`` and ``selection_rationale``. Prod (read-only,
2026-09-29): 188/188 ``model_selection_completed`` rows have NULL ``algorithm_name`` and
``selection_score``, and every description reads
``Model Selection: unknown (unknown). Score: 0.00. Reason: N/A`` -- a plausible-looking
score that was never measured.

The semantic writer had the sibling defect: it read ``problem_type`` from
``selection_summary``, which never carries one, so every ``SUITED_FOR`` edge aimed at a
``ptype:unknown`` node that does not exist and was dropped (prod FalkorDB, read-only
2026-09-29: 0 ``SUITED_FOR`` edges, while ``ptype:binary_classification`` exists and the same
runs wrote their ``USED_IN`` edges).

These tests drive the REAL ``ModelSelectorAgent.run()`` over its real (simple) graph and
record only the storage boundaries (``insert_episodic_memory`` and the FalkorDB client), so
the row is built from the output shape the agent actually produces. An earlier pin
(test_semantic_wiring_749) handed the writer a hand-made ``selection_summary`` with a
``problem_type`` the real summary never has, which is how the dropped edge stayed green.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

_EXPERIMENT = "exp-2325"


class _GraphRecorder:
    """Stands in for the FalkorDB client: records every write the hook issues."""

    def __init__(self) -> None:
        self.entities: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

    def add_e2i_entity(self, **kwargs: Any) -> bool:
        self.entities.append(kwargs)
        return True

    def add_relationship(self, **kwargs: Any) -> bool:
        self.edges.append(kwargs)
        return True


@pytest.fixture
def boundaries(monkeypatch):
    """Record the episodic insert and the graph writes; nothing reaches a real store."""
    import src.agents.ml_foundation.model_selector.agent as agent_module
    import src.memory.episodic_memory as episodic
    import src.memory.semantic_memory as semantic

    rows: List[Dict[str, Any]] = []

    async def _insert(**kwargs: Any) -> str:
        rows.append(kwargs)
        return "mem-2325"

    graph = _GraphRecorder()
    monkeypatch.setattr(episodic, "insert_episodic_memory", _insert)
    monkeypatch.setattr(semantic, "get_semantic_memory", lambda: graph)
    monkeypatch.setattr(agent_module, "_get_opik_connector", lambda: None)
    monkeypatch.setattr(agent_module, "_get_procedural_memory", lambda: None)
    return rows, graph


async def _run_real_agent() -> Dict[str, Any]:
    from src.agents.ml_foundation.model_selector.agent import ModelSelectorAgent

    output = await ModelSelectorAgent(mode="simple").run(
        {
            "scope_spec": {
                "experiment_id": _EXPERIMENT,
                "problem_type": "binary_classification",
                "technical_constraints": [],
            },
            "qc_report": {"qc_passed": True, "row_count": 5000, "column_count": 20},
        }
    )
    assert "error" not in output, output
    return output


def _suited_for(graph: _GraphRecorder) -> List[Dict[str, Any]]:
    return [e for e in graph.edges if e["relationship_type"] == "SUITED_FOR"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_records_the_selected_algorithm_on_the_episodic_row(boundaries):
    rows, _ = boundaries
    output = await _run_real_agent()
    candidate = output["model_candidate"]
    rationale = output["selection_rationale"]
    assert candidate["algorithm_name"] and candidate["algorithm_family"]  # the real shape

    assert len(rows) == 1, rows
    row = rows[0]
    raw = row["raw_content"]
    assert raw["algorithm_name"] == candidate["algorithm_name"]
    assert raw["algorithm_family"] == candidate["algorithm_family"]
    assert raw["algorithm_class"] == candidate["algorithm_class"]
    assert raw["selection_score"] == candidate["selection_score"]
    assert raw["interpretability_score"] == candidate["interpretability_score"]
    assert raw["scalability_score"] == candidate["scalability_score"]
    assert raw["primary_reason"] == rationale["primary_reason"]
    assert raw["selection_rationale"] == rationale

    summary = row["summary"]
    assert candidate["algorithm_name"] in summary
    assert f"({candidate['algorithm_family']})" in summary
    assert f"Score: {candidate['selection_score']:.3f}" in summary
    assert rationale["primary_reason"] in summary
    for fabricated in ("unknown", "N/A", "Score: 0.00"):
        assert fabricated not in summary, summary


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_links_the_algorithm_to_the_real_problem_type(boundaries):
    _, graph = boundaries
    output = await _run_real_agent()
    candidate = output["model_candidate"]

    algo = [e for e in graph.entities if e["entity_type"] == "Algorithm"]
    assert len(algo) == 1
    assert algo[0]["properties"]["family"] == candidate["algorithm_family"]
    suited = _suited_for(graph)
    assert len(suited) == 1, graph.edges
    assert suited[0]["to_entity_id"] == "ptype:binary_classification"
    assert suited[0]["properties"]["selection_score"] == candidate["selection_score"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_hands_the_procedural_pattern_the_real_problem_type(boundaries, monkeypatch):
    """The procedural client does not exist today (the write is a no-op); when it does, the
    pattern must carry the run's problem type, not selection_summary's absent one."""
    import src.agents.ml_foundation.model_selector.agent as agent_module

    patterns: List[Dict[str, Any]] = []

    class _Procedural:
        async def store_pattern(self, **kwargs: Any) -> None:
            patterns.append(kwargs)

    monkeypatch.setattr(agent_module, "_get_procedural_memory", lambda: _Procedural())
    output = await _run_real_agent()

    assert len(patterns) == 1
    data = patterns[0]["pattern_data"]
    assert data["problem_type"] == "binary_classification"
    assert data["algorithm_name"] == output["model_candidate"]["algorithm_name"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_missing_value_reads_as_absent_never_as_a_measurement(boundaries):
    """The real output with the selection fields gone: the row must not invent them."""
    from src.agents.ml_foundation.model_selector.memory_hooks import ModelSelectorMemoryHooks

    rows, _ = boundaries
    output = copy.deepcopy(await _run_real_agent())
    rows.clear()
    for key in ("algorithm_name", "algorithm_family", "algorithm_class", "selection_score"):
        output["model_candidate"].pop(key)
    output["selection_rationale"]["primary_reason"] = ""

    await ModelSelectorMemoryHooks().store_model_selection(
        session_id=None, result=output, state={"experiment_id": _EXPERIMENT}
    )

    assert len(rows) == 1
    raw, summary = rows[0]["raw_content"], rows[0]["summary"]
    assert raw["algorithm_name"] is None
    assert raw["algorithm_family"] is None
    assert raw["selection_score"] is None
    assert raw["primary_reason"] is None
    assert "not recorded" in summary
    for fabricated in ("unknown", "N/A", "0.00", "0.000"):
        assert fabricated not in summary, summary


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_graph_writer_skips_what_it_does_not_know(boundaries):
    """No edge to a ``ptype:unknown`` node, no 0.0 score, no ``family='unknown'`` overwrite."""
    from src.agents.ml_foundation.model_selector.memory_hooks import ModelSelectorMemoryHooks

    _, graph = boundaries
    stored = await ModelSelectorMemoryHooks().store_algorithm_pattern(
        experiment_id=_EXPERIMENT,
        algorithm_name="LogisticRegression",
        algorithm_family=None,
        problem_type=None,
        selection_score=None,
        benchmark_results={},
    )

    assert stored is True
    assert "family" not in graph.entities[0]["properties"]
    assert _suited_for(graph) == []
    used_in = [e for e in graph.edges if e["relationship_type"] == "USED_IN"]
    assert len(used_in) == 1
    assert "selection_score" not in used_in[0]["properties"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_contribute_to_memory_reads_the_real_output_shape(boundaries):
    """The module's ``contribute_to_memory`` entry point, fed what ``run()`` returns."""
    from src.agents.ml_foundation.model_selector.memory_hooks import (
        ModelSelectorMemoryHooks,
        contribute_to_memory,
    )

    rows, graph = boundaries
    output = await _run_real_agent()
    rows.clear()
    graph.entities.clear()
    graph.edges.clear()
    candidate = output["model_candidate"]

    cached: List[Dict[str, Any]] = []

    class _Hooks(ModelSelectorMemoryHooks):
        async def cache_model_selection(self, session_id, selection):
            cached.append(selection)
            return True

    counts = await contribute_to_memory(
        result=output,
        state={
            "experiment_id": _EXPERIMENT,
            "session_id": "s-2325",
            "problem_type": "binary_classification",
        },
        memory_hooks=_Hooks(),
    )

    assert counts == {"episodic_stored": 1, "semantic_stored": 1, "working_cached": 1}
    assert cached == [
        {
            "algorithm_name": candidate["algorithm_name"],
            "algorithm_family": candidate["algorithm_family"],
            "selection_score": candidate["selection_score"],
        }
    ]
    assert rows[0]["raw_content"]["algorithm_name"] == candidate["algorithm_name"]
    suited = _suited_for(graph)
    assert [e["to_entity_id"] for e in suited] == ["ptype:binary_classification"]
