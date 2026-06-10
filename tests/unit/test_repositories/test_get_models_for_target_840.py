"""Unit tests for MLModelRegistryRepository.get_models_for_target.

#840 established the contract; #857 changed the QUERY to avoid a 414 at prod
scale. ``csu_treatment_initiation`` accumulated 253 experiments from the #850
synthetic load; the old implementation fetched every experiment id for the
target and filtered the registry with ``.in_("experiment_id", [253 uuids])`` —
a ~10KB PostgREST GET URL that returns ``414 URI too long``. The fixed query
resolves the target through a single FK-embedded inner join
(``ml_experiments!inner(prediction_target)``) and pushes the serving-stage and
loadable-artifact filters server-side, so the URL size is independent of the
experiment count.

The CONTRACT is unchanged: return the DEPLOYABLE serving model names for a
target — ``stage='production'`` AND non-null ``artifact_path`` — deduped, scoped
to the target via ``ml_experiments.prediction_target``; fail closed (``[]``) on
no client / no match. ``is_champion`` is intentionally NOT required (the
``tr_single_champion`` trigger caps champions at one per experiment; an ensemble
draws on ALL serving models).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.repositories.ml_experiment import MLModelRegistryRepository


class _Recorder:
    """Captures query shape for structural assertions (anti-414 guard)."""

    def __init__(self) -> None:
        self.select_args: List[str] = []
        self.in_cols: List[str] = []


def _embed_get(row: Dict[str, Any], dotted: str) -> Any:
    """Resolve a dotted PostgREST embed path ('ml_experiments.prediction_target')."""
    top, sub = dotted.split(".", 1)
    emb = row.get(top)
    if isinstance(emb, dict):
        return emb.get(sub)
    if isinstance(emb, list):
        for e in emb:
            if isinstance(e, dict) and sub in e:
                return e.get(sub)
    return None


class _FakeQuery:
    """Chainable PostgREST-style query modelling FK-embed + not.is_ semantics."""

    def __init__(self, rows: List[Dict[str, Any]], rec: _Recorder):
        self._rows = rows
        self._rec = rec

    def select(self, *args: Any, **_kw: Any) -> "_FakeQuery":
        if args:
            self._rec.select_args.append(str(args[0]))
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        if "." in col:
            # Embedded-resource filter + !inner => drop rows whose embed doesn't match.
            rows = [r for r in self._rows if _embed_get(r, col) == val]
        else:
            rows = [r for r in self._rows if r.get(col) == val]
        return _FakeQuery(rows, self._rec)

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        self._rec.in_cols.append(col)
        vs = set(vals)
        return _FakeQuery([r for r in self._rows if r.get(col) in vs], self._rec)

    @property
    def not_(self) -> "_FakeNot":
        return _FakeNot(self)

    async def execute(self) -> Any:
        class _Res:
            def __init__(self, data: List[Dict[str, Any]]):
                self.data = data

        return _Res(self._rows)


class _FakeNot:
    def __init__(self, q: _FakeQuery):
        self._q = q

    def is_(self, col: str, sentinel: str) -> _FakeQuery:
        # PostgREST `.not_.is_(col, "null")` => keep rows where col IS NOT NULL.
        assert sentinel == "null", f"unexpected is_ sentinel {sentinel!r}"
        rows = [r for r in self._q._rows if r.get(col) is not None]
        return _FakeQuery(rows, self._q._rec)


class _FakeClient:
    """Fake async Supabase client backed by per-table canned rows."""

    def __init__(
        self, tables: Dict[str, List[Dict[str, Any]]], rec: Optional[_Recorder] = None
    ):
        self._tables = tables
        self.rec = rec or _Recorder()

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery([dict(r) for r in self._tables.get(name, [])], self.rec)


def _csu() -> Dict[str, Any]:
    return {"prediction_target": "csu_treatment_initiation"}


# An ml_experiments table is still provided so that the OLD two-query
# implementation reaches its ``.in_("experiment_id", ...)`` call (which the
# structural guard then catches). The fixed implementation never touches it.
_EXPERIMENTS = [
    {"id": "exp-csu", "prediction_target": "csu_treatment_initiation"},
    {"id": "exp-other", "prediction_target": "pnh_persistence"},
]


def _registry_rows() -> List[Dict[str, Any]]:
    """Rows shaped like the FK-embedded select returns (nested ml_experiments).

    Ensemble membership = serving stage + loadable artifact. is_champion is NOT
    required (the tr_single_champion trigger caps champions at one per
    experiment, but an ensemble draws on ALL serving models).
    """
    pnh = {"prediction_target": "pnh_persistence"}
    return [
        # production + artifact for csu -> RETURN
        {"model_name": "csu_model_a", "stage": "production", "artifact_path": "/a.pkl", "ml_experiments": _csu()},
        # production + artifact, demoted-by-trigger (not champion) -> RETURN
        {"model_name": "csu_model_b", "stage": "production", "artifact_path": "/b.pkl", "ml_experiments": _csu()},
        # staging + artifact -> EXCLUDE (production is the operator gate)
        {"model_name": "csu_staging", "stage": "staging", "artifact_path": "/c.pkl", "ml_experiments": _csu()},
        # NULL artifact (the metadata-only synthetic rows) -> EXCLUDE
        {"model_name": "csu_noart", "stage": "production", "artifact_path": None, "ml_experiments": _csu()},
        # dev stage -> EXCLUDE
        {"model_name": "csu_dev", "stage": "development", "artifact_path": "/d.pkl", "ml_experiments": _csu()},
        # production + artifact for a DIFFERENT target -> EXCLUDE
        {"model_name": "pnh_model", "stage": "production", "artifact_path": "/e.pkl", "ml_experiments": pnh},
        # EMPTY-STRING artifact -> EXCLUDE (server `is null` does not catch ""
        # so the in-code defensive filter must)
        {"model_name": "csu_emptyart", "stage": "production", "artifact_path": "", "ml_experiments": _csu()},
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
    assert sorted(names) == ["csu_model_a", "csu_model_b"]
    # metadata-only (NULL artifact) synthetic row must NOT leak
    assert "csu_noart" not in names
    # empty-string artifact must NOT leak (defensive in-code filter)
    assert "csu_emptyart" not in names
    # staging (not promoted), dev stage, and a different target excluded
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
    """Duplicate deployable model_name rows (e.g. across experiments) returned once."""
    rows = [
        {"model_name": "csu_model_a", "stage": "production", "artifact_path": "/a.pkl", "ml_experiments": _csu()},
        {"model_name": "csu_model_a", "stage": "production", "artifact_path": "/a2.pkl", "ml_experiments": _csu()},
    ]
    client = _FakeClient({"ml_experiments": _EXPERIMENTS, "ml_model_registry": rows})
    repo = MLModelRegistryRepository(supabase_client=client)
    assert await repo.get_models_for_target("csu_treatment_initiation", "hcp") == ["csu_model_a"]


@pytest.mark.asyncio
async def test_resolves_via_fk_embed_not_experiment_id_in_list():
    """#857: the 414 came from ``.in_("experiment_id", [253 uuids])``.

    The fixed query MUST resolve the target through an FK-embedded inner join,
    never materialize experiment ids into a PostgREST ``.in_`` URL.
    """
    rec = _Recorder()
    client = _FakeClient(
        {"ml_experiments": _EXPERIMENTS, "ml_model_registry": _registry_rows()}, rec=rec
    )
    repo = MLModelRegistryRepository(supabase_client=client)
    await repo.get_models_for_target("csu_treatment_initiation", "hcp")

    assert "experiment_id" not in rec.in_cols, (
        "must NOT build a per-experiment-id IN list (the #857 414 cause); "
        f"in_ filters seen: {rec.in_cols}"
    )
    assert any("ml_experiments!inner" in s for s in rec.select_args), (
        "must resolve the target via an FK-embedded inner join "
        f"(ml_experiments!inner(...)); selects seen: {rec.select_args}"
    )
