"""Record the model_selector suite's store writes at their boundaries (#2331).

``ModelSelectorAgent.run()`` writes two real stores after every selection: the
semantic graph (FalkorDB ``e2i_causal``: an ``Algorithm`` node plus ``SUITED_FOR`` and
``USED_IN`` edges) and an MLflow ``model_selection_<algorithm>`` run. The agent tests
built those clients from ``.env``, which on the droplet names PROD. Measured
2026-09-29: two ``SUITED_FOR`` edges landed in prod FalkorDB from test runs (18:12Z,
re-written 21:13Z), and prod MLflow experiment
``e2i_e2i_model_selection_exp_remi_us_20231215120000`` holds ``model_selection_*`` runs
from the same tests, the newest at 18:12:21Z.

This autouse fixture swaps both client factories for recorders, so every test in the
directory runs the real agent and nodes against stores that exist only in memory. A
test that is about a write asserts on the recorder; a test that installs its own
boundary (``test_episodic_row_fields_2325``, ``test_mlflow_registrar``) still wins,
because its patches apply after this autouse fixture. The unit-tree prod-store guard
(``tests/prod_store_guard.py``) remains the backstop underneath.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List

import pytest


class GraphRecorder:
    """Stands in for the FalkorDB semantic-memory client: records every write."""

    def __init__(self) -> None:
        self.entities: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

    def add_e2i_entity(self, **kwargs: Any) -> bool:
        self.entities.append(kwargs)
        return True

    def add_relationship(self, **kwargs: Any) -> bool:
        self.edges.append(kwargs)
        return True


@dataclass
class RunRecorder:
    run_id: str
    experiment_id: str
    run_name: str
    tags: Dict[str, str]
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    async def log_params(self, params: Dict[str, Any]) -> None:
        self.params.update(params)

    async def log_metrics(self, metrics: Dict[str, float], step: Any = None) -> None:
        self.metrics.update(metrics)

    async def log_artifact(self, local_path: str, artifact_path: Any = None) -> None:
        self.artifacts.append(str(artifact_path or local_path))


class MlflowRecorder:
    """Stands in for ``MLflowConnector`` (the registrar's only MLflow seam)."""

    def __init__(self) -> None:
        self.experiments: Dict[str, Dict[str, str]] = {}
        self.runs: List[RunRecorder] = []

    def __call__(self, *args: Any, **kwargs: Any) -> MlflowRecorder:
        # ``MLflowConnector()`` in the registrar -> this recorder.
        return self

    async def get_or_create_experiment(
        self, name: str, tags: Dict[str, str] | None = None, **kwargs: Any
    ) -> str:
        self.experiments.setdefault(name, dict(tags or {}))
        return f"recorded-exp-{list(self.experiments).index(name)}"

    @asynccontextmanager
    async def start_run(
        self, experiment_id: str, run_name: str, tags: Dict[str, str] | None = None, **kwargs: Any
    ) -> AsyncIterator[RunRecorder]:
        run = RunRecorder(
            f"recorded-run-{len(self.runs)}", experiment_id, run_name, dict(tags or {})
        )
        self.runs.append(run)
        yield run


@dataclass
class StoreBoundaries:
    graph: GraphRecorder
    mlflow: MlflowRecorder


@pytest.fixture(autouse=True)
def store_boundaries(monkeypatch: pytest.MonkeyPatch) -> StoreBoundaries:
    import src.memory.semantic_memory as semantic
    import src.mlops.mlflow_connector as mlflow_connector

    boundaries = StoreBoundaries(graph=GraphRecorder(), mlflow=MlflowRecorder())
    monkeypatch.setattr(semantic, "get_semantic_memory", lambda: boundaries.graph)
    monkeypatch.setattr(mlflow_connector, "MLflowConnector", boundaries.mlflow)
    return boundaries
