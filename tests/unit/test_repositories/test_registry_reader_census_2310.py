"""Census of every ``ml_model_registry`` accessor: each one says which rows it means (#2310).

A retrain registers a ``candidate`` row under the name of the model it retrains (migration 159,
160). The owner's R3 decision (2026-09-28): the canonical row for a name is one whose stage is
not ``candidate`` / ``archived`` / ``deprecated``; ``retrain_of_id`` is lineage and never a
filter. Every reader of the table must therefore pick rows in a way that cannot surface an
unreviewed retrain by accident. Pattern of the #894 provenance guard and the #2259 serving-reader
guard: the set of accessors is pinned, each one is classified, and the classification is
checked against the code, so a NEW accessor fails here until someone has decided what it means.

Categories:

* ``CANONICAL`` — a name-style lookup restricted to canonical rows
  (``canonical_rows`` / ``.or_(CANONICAL_STAGE_FILTER)`` / ``non_candidate_rows`` /
  ``resolve_canonical_model_id``).
* ``STAGE`` — scoped to ``production`` / ``staging`` (or ``shadow``); ``candidate`` is outside.
* ``EXACT`` — one row it already identifies: by ``id`` (incl. ``get_by_id``) or by the unique
  ``(model_name, model_version)``; a candidate is returned only when it is the row asked for.
* ``OPT_IN`` — documented: returns candidates on purpose, or reads no role-bearing rows
  (inserts, schema probes). The reason is part of the pin.

The behaviour of the fixed readers against real PostgREST is proven in
``tests/unit/test_database/test_registry_candidate_readers_realdb_2310.py``; this file is the
CI ratchet that runs everywhere.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE = "ml_model_registry"
REPO_CLASS = "MLModelRegistryRepository"

CANONICAL, STAGE, EXACT, OPT_IN = "canonical", "stage", "exact", "opt_in"

#: Every function in src/ and scripts/ that queries ml_model_registry, with how it picks rows.
ACCESSORS: Dict[str, Tuple[str, str]] = {
    # --- canonical (name handles) ------------------------------------------------------------
    "src/repositories/model_registry_roles.py::resolve_canonical_model_id": (
        CANONICAL,
        "the shared name-handle resolver: canonical rows, newest first, id tie-break",
    ),
    "src/api/routes/explain.py::_resolve_model_registry_id": (
        CANONICAL,
        "SHAP rows are stored under this id; it must be the model the name serves",
    ),
    "src/api/routes/explain.py::_get_latest_versions_by_model_type": (
        CANONICAL,
        "/explain/models latest_version: a newer candidate is not the current version",
    ),
    "scripts/promote_hcp_adoption_champions.py::_fetch_registry_row": (
        CANONICAL,
        "the owner-ruled row; a retrain candidate must not make it ambiguous",
    ),
    "src/repositories/ml_experiment.py::get_champion_model": (
        CANONICAL,
        "champion flag + non-candidate unless include_candidates=True",
    ),
    # --- stage-scoped (production / staging) -------------------------------------------------
    "src/agents/drift_monitor/connectors/supabase_connector.py::get_available_models": (
        STAGE,
        "caller-chosen stages; both callers (drift sweep, retraining sweep, "
        "drift_monitoring_tasks.py) pass ['production', 'staging'] — pinned below",
    ),
    "src/agents/orchestrator/nodes/dispatcher.py::_probe_prediction_champions": (
        STAGE,
        "production champions",
    ),
    "src/api/routes/health_score.py::_fetch_model_registry_facts": (STAGE, "production/staging"),
    "src/api/routes/predictions.py::_resolve_production_model_names": (
        STAGE,
        "/models/status: production/staging",
    ),
    "src/kpi/goldstd_model_perf.py::_registry_query": (STAGE, "KPI goldstd selector: staging"),
    "src/repositories/ml_experiment.py::get_models_for_target": (STAGE, "_SERVING_STAGES"),
    "src/repositories/ml_experiment.py::get_model_performance_for_target": (
        STAGE,
        "_SERVING_STAGES",
    ),
    "src/repositories/ml_experiment.py::get_models_by_stage": (
        STAGE,
        "exactly the stage asked for: candidates only when stage='candidate' is asked",
    ),
    "src/services/hcp_segment_likelihood.py::resolve_hcp_adoption_champion": (
        STAGE,
        "production champion by name",
    ),
    # --- exact identity ----------------------------------------------------------------------
    "src/repositories/model_registry_roles.py::resolve_model_id_by_name_version": (
        EXACT,
        "UNIQUE(model_name, model_version)",
    ),
    "src/repositories/ml_experiment.py::get_by_name_version": (EXACT, "(name, version)"),
    "src/repositories/ml_experiment.py::transition_stage": (
        EXACT,
        "get_by_id + update by id; the production archive is scoped to model_name AND "
        "stage='production' (#2259), which a candidate never has",
    ),
    "src/services/cohort_contract.py::load_registry_cohort_contract": (
        EXACT,
        "reads by the id _resolve_model_id returned (canonical for a name, exact for a uuid)",
    ),
    "src/services/cohort_contract.py::load_registry_model_identity": (EXACT, "by id"),
    "src/services/cohort_contract.py::heal_registry_cohort_contract": (EXACT, "by id"),
    "src/agents/ml_foundation/model_deployer/nodes/training_provenance.py::"
    "heal_training_provenance": (EXACT, "by id"),
    "src/tasks/drift_monitoring_tasks.py::_execute_real_retraining": (
        EXACT,
        "retrain completion: the candidate row by (model_name, new_model_version) and by id "
        "(Lane A's write path)",
    ),
    "src/mlops/prediction_synthesizer_deploy.py::register_model_row": (
        EXACT,
        "writer: reads back the (name, version) row it upserted",
    ),
    "src/mlops/gold_standard_eval/cleanup_orphan_models.py::decommission": (
        EXACT,
        "archives by the id _resolve_model_id returned for its three hardcoded names",
    ),
    "scripts/backfill_goldstd_holdout_metrics.py::_update_registry_auc": (
        EXACT,
        "by the canonical id resolved once per model (was: every row of the name)",
    ),
    "scripts/promote_hcp_adoption_champions.py::_apply_update": (EXACT, "by id"),
    "scripts/promote_hcp_adoption_champions.py::_verify_written": (EXACT, "by id"),
    # --- documented opt-in -------------------------------------------------------------------
    "src/repositories/ml_experiment.py::register_model": (OPT_IN, "insert only"),
    "src/repositories/ml_experiment.py::register_model_candidate": (OPT_IN, "insert only"),
    "src/agents/drift_monitor/connectors/supabase_connector.py::health_check": (
        OPT_IN,
        "schema probe: select id limit 1, the row is never used",
    ),
}

#: Callers of the resolvers: which one each uses, and why.
RESOLVER_CALLERS: Dict[str, Set[str]] = {
    # _resolve_model_id: uuid -> that row (exact); name -> canonical row.
    # the repositories themselves; _resolve_model_id delegates names to the canonical resolver
    "src/repositories/drift_monitoring.py": {"_resolve_model_id", "resolve_canonical_model_id"},
    "src/mlops/gold_standard_eval/recorder.py": {"_resolve_model_id"},  # name -> canonical
    "src/services/cohort_contract.py": {"_resolve_model_id"},  # manual retrain by name/uuid
    "src/mlops/gold_standard_eval/cleanup_orphan_models.py": {"_resolve_model_id"},
    # the row register_model_row is about to replace: exact (name, version)
    "src/mlops/gold_standard_eval/run_initiation_eval.py": {"resolve_model_id_by_name_version"},
    "src/mlops/gold_standard_eval/run_persistence_eval.py": {"resolve_model_id_by_name_version"},
    # resolved once per model, then every read/write by that id
    "scripts/backfill_goldstd_holdout_metrics.py": {"resolve_canonical_model_id"},
}
_RESOLVERS = ("_resolve_model_id", "resolve_canonical_model_id", "resolve_model_id_by_name_version")

_CANONICAL_CALLS = {"canonical_rows", "non_candidate_rows", "resolve_canonical_model_id"}
_SERVED_STAGES = {"production", "staging", "shadow"}


# ---------------------------------------------------------------------------------------------
# Python scan
# ---------------------------------------------------------------------------------------------


def _body(fn: ast.AST) -> List[ast.AST]:
    body = list(getattr(fn, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return [n for stmt in body for n in ast.walk(stmt)]


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
    return ""


def _consts(node: ast.Call) -> Tuple[object, ...]:
    return tuple(a.value if isinstance(a, ast.Constant) else None for a in node.args)


def _table_aliases(tree: ast.Module) -> Set[str]:
    """Module-level names bound to the table name (``TABLE = "ml_model_registry"``)."""
    return {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == TABLE
        for t in node.targets
        if isinstance(t, ast.Name)
    }


def _is_accessor(nodes: List[ast.AST], aliases: Set[str], in_repo: bool) -> bool:
    for node in nodes:
        if _call_name(node) not in ("table", "from_"):
            if in_repo and _call_name(node) in ("get_many", "get_by_id"):
                return True
            continue
        arg = node.args[0] if isinstance(node, ast.Call) and node.args else None
        if isinstance(arg, ast.Constant) and arg.value == TABLE:
            return True
        if isinstance(arg, ast.Name) and arg.id in aliases:
            return True
        if in_repo and isinstance(arg, ast.Attribute) and arg.attr == "table_name":
            return True
    return False


def _outer_functions(tree: ast.AST) -> Iterator[Tuple[ast.AST, ast.AST]]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield node, child


def _python_files() -> Iterator[Path]:
    for base in ("src", "scripts"):
        yield from sorted((REPO_ROOT / base).rglob("*.py"))


def _scan() -> Dict[str, List[ast.AST]]:
    found: Dict[str, List[ast.AST]] = {}
    for path in _python_files():
        text = path.read_text()
        if TABLE not in text and REPO_CLASS not in text:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(text)
        aliases = _table_aliases(tree)
        for owner, fn in _outer_functions(tree):
            in_repo = isinstance(owner, ast.ClassDef) and owner.name == REPO_CLASS
            nodes = _body(fn)
            if _is_accessor(nodes, aliases, in_repo):
                found[f"{rel}::{fn.name}"] = nodes
    return found


def _has_canonical(nodes: List[ast.AST]) -> bool:
    for n in nodes:
        if _call_name(n) in _CANONICAL_CALLS:
            return True
        if _call_name(n) == "or_" and isinstance(n, ast.Call) and n.args:
            arg = n.args[0]
            if (
                isinstance(arg, (ast.Name, ast.Attribute))
                and (getattr(arg, "id", None) or getattr(arg, "attr", None))
                == "CANONICAL_STAGE_FILTER"
            ):
                return True
    return False


def _has_stage_scope(nodes: List[ast.AST]) -> bool:
    for n in nodes:
        name = _call_name(n)
        if isinstance(n, ast.Attribute) and n.attr == "_SERVING_STAGES":
            return True
        if not isinstance(n, ast.Call) or not n.args or _consts(n)[:1] != ("stage",):
            continue
        if name == "eq" and len(n.args) > 1:
            value = n.args[1]
            if isinstance(value, ast.Constant):
                if value.value in _SERVED_STAGES:
                    return True
            else:
                return True  # a caller-chosen single stage: exactly what was asked for
        if name == "in_" and len(n.args) > 1:
            value = n.args[1]
            if isinstance(value, (ast.List, ast.Tuple)):
                vals = {e.value for e in value.elts if isinstance(e, ast.Constant)}
                if vals and vals <= _SERVED_STAGES:
                    return True
            else:
                return True  # caller-chosen stages (pinned per caller below)
    # BaseRepository.get_many(filters={"stage": stage}) — the only stage-keyed dict form.
    for n in nodes:
        if isinstance(n, ast.Dict):
            keys = [k.value for k in n.keys if isinstance(k, ast.Constant)]
            if "stage" in keys:
                return True
    return False


def _has_exact_key(nodes: List[ast.AST]) -> bool:
    eq_cols = {_consts(n)[0] for n in nodes if _call_name(n) == "eq" and isinstance(n, ast.Call)}
    if "id" in eq_cols or {"model_name", "model_version"} <= eq_cols:
        return True
    return any(_call_name(n) == "get_by_id" for n in nodes)


_CHECKS = {
    CANONICAL: _has_canonical,
    STAGE: _has_stage_scope,
    EXACT: _has_exact_key,
    OPT_IN: lambda nodes: True,
}


def test_every_registry_accessor_is_classified():
    found = set(_scan())
    new = sorted(found - set(ACCESSORS))
    gone = sorted(set(ACCESSORS) - found)
    assert not new and not gone, (
        "ml_model_registry accessors changed. New (classify each in ACCESSORS: canonical / "
        f"stage / exact / opt_in with a reason): {new}; gone: {gone}. #2310: a name lookup "
        "without a stage predicate can return an unreviewed retrain candidate."
    )


def test_each_classification_matches_the_code():
    found = _scan()
    wrong = sorted(
        f"{key} ({category})"
        for key, (category, _reason) in ACCESSORS.items()
        if key in found and not _CHECKS[category](found[key])
    )
    assert not wrong, f"accessors whose code does not do what their class says: {wrong}"


def test_every_opt_in_has_a_reason():
    assert all(reason.strip() for _cat, reason in ACCESSORS.values())


def test_caller_chosen_sweep_stages_exclude_candidates():
    """get_available_models takes the stages from its callers: each must name served stages."""
    path = REPO_ROOT / "src/tasks/drift_monitoring_tasks.py"
    calls = [
        n for n in ast.walk(ast.parse(path.read_text())) if _call_name(n) == "get_available_models"
    ]
    assert calls, "the sweeps no longer call get_available_models: re-check the census"
    for call in calls:
        assert isinstance(call, ast.Call)
        stages = {kw.arg: kw.value for kw in call.keywords}.get("stages")
        assert isinstance(stages, ast.List), ast.unparse(call)
        values = {e.value for e in stages.elts if isinstance(e, ast.Constant)}
        assert values and values <= _SERVED_STAGES, ast.unparse(call)


def test_resolver_callers_are_pinned():
    found: Dict[str, Set[str]] = {}
    for path in _python_files():
        text = path.read_text()
        if not any(r in text for r in _RESOLVERS):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == "src/repositories/model_registry_roles.py":
            continue
        calls = {_call_name(n) for n in ast.walk(ast.parse(text)) if _call_name(n) in _RESOLVERS}
        if calls:
            found[rel] = calls
    assert set(found) == set(RESOLVER_CALLERS), (
        f"resolver callers changed: new {sorted(set(found) - set(RESOLVER_CALLERS))}, "
        f"gone {sorted(set(RESOLVER_CALLERS) - set(found))}. Decide per caller: a name that "
        "means 'the model' -> canonical; one specific row -> uuid or (name, version)."
    )
    wrong = {rel: calls for rel, calls in found.items() if calls != RESOLVER_CALLERS[rel]}
    assert not wrong, f"callers using a different resolver than pinned: {wrong}"


def test_the_checks_have_teeth():
    """A bare name lookup is not canonical, not stage-scoped and not exact."""
    fn = ast.parse(
        "async def f(c, n):\n"
        "    return await c.table('ml_model_registry').select('id').eq('model_name', n)"
        ".order('registered_at', desc=True).limit(1).execute()\n"
    ).body[0]
    nodes = _body(fn)
    assert _is_accessor(nodes, set(), in_repo=False)
    assert not _has_canonical(nodes) and not _has_stage_scope(nodes)
    assert not _has_exact_key(nodes)
    staged = _body(
        ast.parse(
            "def g(c):\n    return c.table('ml_model_registry').select('id')"
            ".in_('stage', ['production', 'candidate']).execute()\n"
        ).body[0]
    )
    assert not _has_stage_scope(staged), "a stage list naming 'candidate' is not served-only"


# ---------------------------------------------------------------------------------------------
# SQL views and functions
# ---------------------------------------------------------------------------------------------

#: Every view / function under database/ whose body reads ml_model_registry.
SQL_OBJECTS: Dict[str, Tuple[str, str]] = {
    "ml_model_health_dashboard": (STAGE, "production/staging (every definition, 017..103)"),
    "v_champion_models": (STAGE, "champion + production/staging"),
    "get_latest_model": (
        OPT_IN,
        "stage-agnostic newest-by-experiment; ZERO callers in src/, scripts/, frontend "
        "(checked 2026-09-28) — must gain a canonical predicate before anything calls it",
    ),
    "ensure_single_champion": (
        OPT_IN,
        "trigger (writer): clears is_champion in the experiment; retrain candidates are "
        "written is_champion=false",
    ),
    "ml_model_registry_retrain_of_immutable": (OPT_IN, "lineage trigger (160), reads OLD/NEW"),
}

_SQL_DEF = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?(VIEW|FUNCTION)\s+(?:public\.)?(\w+)",
    re.I,
)
_SQL_STAGE = re.compile(
    r"stage\s+IN\s*\(\s*'production'\s*,\s*'staging'\s*\)"
    r"|stage\s*=\s*ANY\s*\(\s*ARRAY\s*\[\s*'production'::model_stage_enum\s*,"
    r"\s*'staging'::model_stage_enum\s*\]\s*\)",
    re.I,
)


def _strip_sql_comments(text: str) -> str:
    return "\n".join(re.sub(r"--.*$", "", line) for line in text.splitlines())


def _sql_definitions() -> Dict[str, List[Tuple[str, str]]]:
    """name -> [(file, body)] for every view/function definition that reads the registry."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for path in sorted((REPO_ROOT / "database").rglob("*.sql")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if "/rollback_" in rel or path.name.startswith("rollback_"):
            continue
        text = _strip_sql_comments(path.read_text())
        matches = list(_SQL_DEF.finditer(text))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end() : end]
            if m.group(1).upper() == "VIEW":
                body = body.split(";", 1)[0]
            else:
                parts = re.split(r"\$(\w*)\$", body)
                body = parts[2] if len(parts) > 2 else body
            if re.search(rf"\b{TABLE}\b", body):
                out.setdefault(m.group(2).lower(), []).append((rel, body))
    return out


def test_every_sql_reader_of_the_registry_is_classified():
    found = set(_sql_definitions())
    assert found == set(SQL_OBJECTS), (
        f"SQL views/functions reading {TABLE} changed: new {sorted(found - set(SQL_OBJECTS))}, "
        f"gone {sorted(set(SQL_OBJECTS) - found)}"
    )


def test_stage_scoped_sql_readers_are_scoped_in_every_definition():
    defs = _sql_definitions()
    unscoped = [
        f"{name} in {rel}"
        for name, (category, _r) in SQL_OBJECTS.items()
        if category == STAGE
        for rel, body in defs.get(name, [])
        if not _SQL_STAGE.search(body)
    ]
    assert not unscoped, (
        f"stage-scoped SQL readers without the production/staging scope: {unscoped}"
    )
