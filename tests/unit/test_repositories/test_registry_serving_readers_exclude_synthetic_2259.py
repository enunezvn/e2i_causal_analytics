"""Every serving reader of ``ml_model_registry`` excludes ``is_synthetic`` rows (#2259).

This is what makes the synthetic generator's exemption from the #968 production gate safe. The
generator (src/ml/synthetic/generators/mlops_generator.py) stamps ``stage='production'`` on its
champion rows with ``is_synthetic=true``, no artifact, no MLflow run and no
``training_provenance`` (720 such rows on prod, 2026-09-23). They are fabricated metadata, not
trained models, so the owner exempted them from the fail-closed rule instead of relabelling them
(see ``MLModelRegistryRepository.transition_stage``). The exemption holds only while no reader
that picks production (or champion) rows can surface one.

The scan finds every function in ``src/`` that SELECTs from the registry and scopes the read to
production (``'production'`` / ``_SERVING_STAGES`` / a caller-chosen ``stage``) or to champions
(``.eq("is_champion", True)``),
and requires an ``is_synthetic`` exclusion in it: ``.eq("is_synthetic", False)``,
``apply_provenance_filter(...)``, or a client-side ``row.get("is_synthetic")`` skip. The set of
such readers is pinned, so a NEW production reader fails here until someone has looked at it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterator, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
TABLE = "ml_model_registry"
REPO_CLASS = "MLModelRegistryRepository"

#: The production/champion readers of the registry, each checked for the exclusion below.
SERVING_READERS: Set[str] = {
    "src/agents/drift_monitor/connectors/supabase_connector.py::get_available_models",
    "src/agents/orchestrator/nodes/dispatcher.py::_probe_prediction_champions",
    "src/api/routes/health_score.py::_fetch_model_registry_facts",
    "src/api/routes/predictions.py::_resolve_production_model_names",
    "src/repositories/ml_experiment.py::get_champion_model",
    "src/repositories/ml_experiment.py::get_models_for_target",
    "src/repositories/ml_experiment.py::get_model_performance_for_target",
    "src/services/hcp_segment_likelihood.py::resolve_hcp_adoption_champion",
}

#: Functions the scan matches that do not serve, with the reason.
NOT_SERVING: Dict[str, str] = {
    "src/mlops/prediction_synthesizer_deploy.py::register_model_row": (
        "a writer: its SELECT reads back the one row it just upserted, by name and version"
    ),
}


def _body_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    body = list(getattr(fn, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # the docstring describes, it does not filter
    for stmt in body:
        yield from ast.walk(stmt)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
    return ""


def _const_args(node: ast.Call) -> Tuple[object, ...]:
    return tuple(a.value if isinstance(a, ast.Constant) else None for a in node.args)


def _classify(fn: ast.AST, in_repo_class: bool) -> Tuple[bool, bool]:
    """(is a production/champion registry reader, excludes is_synthetic)."""
    names_table = selects = scoped = excludes = False
    for node in _body_nodes(fn):
        if isinstance(node, ast.Constant) and node.value == TABLE:
            names_table = True
        if in_repo_class and isinstance(node, ast.Attribute) and node.attr == "table_name":
            names_table = True
        if isinstance(node, ast.Constant) and node.value == "production":
            scoped = True
        if isinstance(node, ast.Attribute) and node.attr == "_SERVING_STAGES":
            scoped = True
        name = _call_name(node)
        if name == "select":
            selects = True
        if name == "eq" and _const_args(node) == ("is_champion", True):  # type: ignore[arg-type]
            scoped = True
        if (
            name in ("eq", "in_")
            and isinstance(node, ast.Call)
            and _const_args(node)[:1] == ("stage",)
            and len(node.args) > 1
            and not isinstance(node.args[1], ast.Constant)
        ):
            scoped = True  # a caller-chosen stage: production is one of the values it serves
        if name == "eq" and _const_args(node) == ("is_synthetic", False):  # type: ignore[arg-type]
            excludes = True
        if name == "apply_provenance_filter":
            excludes = True
        if name == "get" and _const_args(node)[:1] == ("is_synthetic",):  # type: ignore[arg-type]
            excludes = True
    return names_table and selects and scoped, excludes


def _scan() -> Dict[str, bool]:
    found: Dict[str, bool] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text()
        if TABLE not in text and REPO_CLASS not in text:
            continue
        tree = ast.parse(text)
        repo_fns = {
            id(fn)
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef) and cls.name == REPO_CLASS
            for fn in ast.walk(cls)
        }
        for fn in _outer_functions(tree):
            reader, excludes = _classify(fn, id(fn) in repo_fns)
            if reader:
                found[f"{rel}::{fn.name}"] = excludes
    return found


def _outer_functions(tree: ast.AST) -> Iterator[ast.AST]:
    """Module functions and methods; a nested helper is scanned as part of its enclosing function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield child


def test_the_production_readers_are_exactly_the_pinned_set():
    found = set(_scan())
    assert found == SERVING_READERS | set(NOT_SERVING), (
        "the set of functions reading production/champion ml_model_registry rows changed. "
        f"New: {sorted(found - SERVING_READERS - set(NOT_SERVING))}; "
        f"gone: {sorted((SERVING_READERS | set(NOT_SERVING)) - found)}. A new serving reader "
        "must exclude is_synthetic rows (the synthetic generator's production rows are exempt "
        "from the #968 gate only because no serving path can reach them)."
    )


def test_every_serving_reader_excludes_synthetic_rows():
    found = _scan()
    missing = sorted(r for r in SERVING_READERS if not found.get(r))
    assert not missing, f"serving readers without an is_synthetic exclusion: {missing}"


def test_the_scan_sees_an_unguarded_production_reader():
    """Teeth: the classifier flags a production reader that does not exclude synthetic rows."""
    fn = ast.parse(
        "async def serve(db):\n"
        "    return await db.table('ml_model_registry').select('*')"
        ".eq('stage', 'production').execute()\n"
    ).body[0]
    assert _classify(fn, in_repo_class=False) == (True, False)
    guarded = ast.parse(
        "async def serve(db):\n"
        "    return await db.table('ml_model_registry').select('*')"
        ".eq('stage', 'production').eq('is_synthetic', False).execute()\n"
    ).body[0]
    assert _classify(guarded, in_repo_class=False) == (True, True)
