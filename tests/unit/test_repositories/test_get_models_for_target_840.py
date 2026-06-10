"""Unit tests for MLModelRegistryRepository.get_models_for_target (#840, #857).

The prediction_synthesizer orchestrator expects a ``model_registry`` object
implementing the ``ModelRegistry`` Protocol:

    async def get_models_for_target(target: str, entity_type: str) -> List[str]

This method resolves the DEPLOYABLE model names for a prediction target,
returning only models that are (a) at ``stage='production'`` and (b) carry a
non-null ``artifact_path`` (so a client can actually load them), scoped to the
target via ``ml_experiments.prediction_target``. It MUST fail closed (return
``[]``) when there is no client or no matching deployable model.

#857: the resolution is done with a SERVER-SIDE FK-embedded inner join
(``ml_experiments!inner(prediction_target)``) rather than fetching every
experiment id and passing it to ``.in_("experiment_id", [...])`` — at prod scale
that id list overflowed the PostgREST GET URL (``414 URI too long``). The fake
client below models the embedded join so the behavioural tests exercise the real
query shape, and ``test_resolves_via_single_embedded_query`` guards the
regression (no separate ``ml_experiments`` id query).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.repositories.ml_experiment import MLModelRegistryRepository


class _FakeNot:
    """Models the ``.not_.is_(col, "null")`` PostgREST negation."""

    def __init__(self, query: "_FakeQuery"):
        self._q = query

    def is_(self, col: str, sentinel: Any) -> "_FakeQuery":
        assert sentinel == "null", "fake only models IS NOT NULL"
        self._q.rows = [r for r in self._q.rows if r.get(col) is not None]
        return self._q


class _FakeQuery:
    """Chainable PostgREST-style query that models an embedded inner join."""

    def __init__(self, table_name: str, tables: Dict[str, List[Dict[str, Any]]]):
        self.table_name = table_name
        self._tables = tables
        self.rows = list(tables.get(table_name, []))

    def select(self, cols: str, *_a: Any, **_k: Any) -> "_FakeQuery":
        # Model ``ml_experiments!inner(prediction_target)``: inner-join each
        # registry row to its experiment and attach the embedded resource. Rows
        # whose experiment_id has no match are dropped (inner join semantics).
        if "ml_experiments!inner" in cols:
            exps = {e["id"]: e for e in self._tables.get("ml_experiments", [])}
            joined: List[Dict[str, Any]] = []
            for r in self.rows:
                exp = exps.get(r.get("experiment_id"))
                if exp is None:
                    continue
                rr = dict(r)
                rr["ml_experiments"] = {"prediction_target": exp.get("prediction_target")}
                joined.append(rr)
            self.rows = joined
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        if col == "ml_experiments.prediction_target":
            self.rows = [
                r
                for r in self.rows
                if (r.get("ml_experiments") or {}).get("prediction_target") == val
            ]
        else:
            self.rows = [r for r in self.rows if r.get(col) == val]
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        allowed = set(vals)
        self.rows = [r for r in self.rows if r.get(col) in allowed]
        return self

    @property
    def not_(self) -> _FakeNot:
        return _FakeNot(self)

    async def execute(self) -> Any:
        class _Res:
            def __init__(self, data: List[Dict[str, Any]]):
                self.data = data

        return _Res(self.rows)


class _FakeClient:
    """Fake async Supabase client; records which tables were queried."""

    def __init__(self, tables: Dict[str, List[Dict[str, Any]]]):
        self._tables = tables
        self.tables_queried: List[str] = []

    def table(self, name: str) -> _FakeQuery:
        self.tables_queried.append(name)
        return _FakeQuery(name, self._tables)


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
        # STAGING + artifact -> EXCLUDE (production is the explicit operator gate)
        {
            "experiment_id": "exp-csu",
            "model_name": "csu_staging",
            "stage": "staging",
            "is_champion": False,
            "artifact_path": "/c.pkl",
        },
        # NO artifact (the metadata-only synthetic rows) -> EXCLUDE
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
async def test_returns_only_production_models_with_artifact_for_target():
    """Only stage='production' + non-null artifact, scoped to target (is_champion irrelevant)."""
    client = _FakeClient({"ml_experiments": _EXPERIMENTS, "ml_model_registry": _registry_rows()})
    repo = MLModelRegistryRepository(supabase_client=client)
    names = await repo.get_models_for_target("csu_treatment_initiation", "hcp")
    # both production+artifact csu models (champion status irrelevant for ensemble)
    assert sorted(names) == ["csu_model_a", "csu_model_b"]
    # the metadata-only (NULL artifact) synthetic row must NOT leak
    assert "csu_synthetic_noart" not in names
    # staging (not yet promoted), dev stage, and a different target excluded
    assert "csu_staging" not in names
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
    """Duplicate model_name rows (across versions) are returned once."""
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
            "stage": "production",
            "is_champion": False,
            "artifact_path": "/a2.pkl",
        },
    ]
    client = _FakeClient({"ml_experiments": _EXPERIMENTS, "ml_model_registry": rows})
    repo = MLModelRegistryRepository(supabase_client=client)
    assert await repo.get_models_for_target("csu_treatment_initiation", "hcp") == ["csu_model_a"]


@pytest.mark.asyncio
async def test_resolves_via_single_embedded_query():
    """#857 regression: resolution uses ONE embedded query on ml_model_registry,
    never a separate ml_experiments id fetch feeding ``.in_("experiment_id", …)``
    (the 253-id list that overflowed the GET URL -> 414)."""
    client = _FakeClient({"ml_experiments": _EXPERIMENTS, "ml_model_registry": _registry_rows()})
    repo = MLModelRegistryRepository(supabase_client=client)
    await repo.get_models_for_target("csu_treatment_initiation", "hcp")
    # only the registry table is queried; the target is joined server-side
    assert client.tables_queried == ["ml_model_registry"]
    assert "ml_experiments" not in client.tables_queried
