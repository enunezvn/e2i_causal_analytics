"""Unit tests for MLModelRegistryRepository.get_models_for_target (#840).

The prediction_synthesizer orchestrator expects a ``model_registry`` object
implementing the ``ModelRegistry`` Protocol:

    async def get_models_for_target(target: str, entity_type: str) -> List[str]

This method resolves the DEPLOYABLE champion model names for a prediction
target (joining ``ml_experiments`` for ``prediction_target``), returning only
models that are (a) champions, (b) in a serving stage (production/staging),
and (c) carry a non-null ``artifact_path`` (so a client can actually load
them). It MUST fail closed (return ``[]``) when there is no client or no
matching deployable model — never fabricate model names.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.repositories.ml_experiment import MLModelRegistryRepository


class _FakeQuery:
    """Minimal chainable PostgREST-style query over canned table rows."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def select(self, *_args: Any, **_kwargs: Any) -> "_FakeQuery":
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        return _FakeQuery([r for r in self._rows if r.get(col) == val])

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        return _FakeQuery([r for r in self._rows if r.get(col) in set(vals)])

    async def execute(self) -> Any:
        class _Res:
            def __init__(self, data: List[Dict[str, Any]]):
                self.data = data

        return _Res(self._rows)


class _FakeClient:
    """Fake async Supabase client backed by per-table canned rows."""

    def __init__(self, tables: Dict[str, List[Dict[str, Any]]]):
        self._tables = tables

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(list(self._tables.get(name, [])))


_EXPERIMENTS = [
    {"id": "exp-csu", "prediction_target": "csu_treatment_initiation"},
    {"id": "exp-other", "prediction_target": "pnh_persistence"},
]


def _registry_rows() -> List[Dict[str, Any]]:
    # Ensemble membership = serving stage + loadable artifact. is_champion is
    # NOT required (the tr_single_champion trigger caps champions at one per
    # experiment, but an ensemble draws on ALL serving models).
    return [
        # serving + artifact for csu -> SHOULD return (the single champion)
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_model_a",
            "stage": "production",
            "is_champion": True,
            "artifact_path": "/a.pkl",
        },
        # serving + artifact, NOT champion (demoted by trigger) -> SHOULD return
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_model_b",
            "stage": "production",
            "is_champion": False,
            "artifact_path": "/b.pkl",
        },
        # staging + artifact -> SHOULD return
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_model_c",
            "stage": "staging",
            "is_champion": False,
            "artifact_path": "/c.pkl",
        },
        # NO artifact (the 72 metadata-only synthetic rows) -> EXCLUDE
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_synthetic_noart",
            "stage": "production",
            "is_champion": True,
            "artifact_path": None,
        },
        # artifact but wrong (dev) stage -> EXCLUDE
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_dev",
            "stage": "development",
            "is_champion": True,
            "artifact_path": "/d.pkl",
        },
        # serving + artifact for a DIFFERENT target -> EXCLUDE
        {
            "experiment_id": "exp-other",
            "model_name": "pnh_model",
            "stage": "production",
            "is_champion": True,
            "artifact_path": "/e.pkl",
        },
    ]


@pytest.mark.asyncio
async def test_no_client_fails_closed_returns_empty():
    """No Supabase client -> [] (never fabricate)."""
    repo = MLModelRegistryRepository(supabase_client=None)
    assert await repo.get_models_for_target("csu_treatment_initiation", "hcp") == []


@pytest.mark.asyncio
async def test_returns_only_deployable_serving_models_for_target():
    """All serving-stage (production/staging) + non-null artifact, scoped to target."""
    client = _FakeClient({"ml_experiments": _EXPERIMENTS, "ml_model_registry": _registry_rows()})
    repo = MLModelRegistryRepository(supabase_client=client)
    names = await repo.get_models_for_target("csu_treatment_initiation", "hcp")
    # all 3 serving+artifact csu models (champion status irrelevant for ensemble)
    assert sorted(names) == ["csu_model_a", "csu_model_b", "csu_model_c"]
    # the metadata-only (NULL artifact) synthetic row must NOT leak
    assert "csu_synthetic_noart" not in names
    # wrong stage / different target excluded
    assert "csu_dev" not in names
    assert "pnh_model" not in names


@pytest.mark.asyncio
async def test_unknown_target_fails_closed():
    """No experiment for the target -> []."""
    client = _FakeClient({"ml_experiments": _EXPERIMENTS, "ml_model_registry": _registry_rows()})
    repo = MLModelRegistryRepository(supabase_client=client)
    assert await repo.get_models_for_target("nonexistent_target", "hcp") == []


@pytest.mark.asyncio
async def test_dedup_model_names():
    """Duplicate model_name rows are returned once."""
    rows = [
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_model_a",
            "stage": "production",
            "is_champion": True,
            "artifact_path": "/a.pkl",
        },
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_model_a",
            "stage": "staging",
            "is_champion": True,
            "artifact_path": "/a2.pkl",
        },
    ]
    client = _FakeClient({"ml_experiments": _EXPERIMENTS, "ml_model_registry": rows})
    repo = MLModelRegistryRepository(supabase_client=client)
    assert await repo.get_models_for_target("csu_treatment_initiation", "hcp") == ["csu_model_a"]
