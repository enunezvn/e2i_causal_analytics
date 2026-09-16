"""#2021 Task 9b: an ``EffectDataUnavailable`` names its cause, not only its message.

Every twin effect refusal used to reach ``counterfactual_simulator`` as one message string, so
the tool refused them all under one generic reason code. The exception now carries a closed
``EffectCause`` and numeric details, which the tool maps to a per-cause code.

The contract lives in ``src/digital_twin/effect/errors.py``, which both the twin engine and the
tool composer import. The dependency runs one way: the tool composer depends on the twin package,
never the reverse, so ``errors.py`` stays standard-library only and importing it never loads the
tool composer or the API.
"""

from __future__ import annotations

import ast
import copy
import pickle
import sys
from pathlib import Path

from src.digital_twin.effect import errors

REPO_ROOT = Path(__file__).resolve().parents[4]
ERRORS_PATH = REPO_ROOT / "src" / "digital_twin" / "effect" / "errors.py"


def test_the_cause_and_details_ride_on_the_exception():
    err = errors.EffectDataUnavailable(
        "m", cause=errors.EffectCause.NO_TREATMENT_CONTRAST, details={"n_usable_rows": 3}
    )
    assert str(err) == "m"
    assert err.cause is errors.EffectCause.NO_TREATMENT_CONTRAST
    assert err.details == {"n_usable_rows": 3}
    assert isinstance(err, RuntimeError)


def test_the_cause_and_details_are_optional():
    """Off the tool path (the synthetic provider, the uplift estimator) no cause is named."""
    err = errors.EffectDataUnavailable("m")
    assert err.cause is None
    assert err.details == {}


def test_the_cause_and_details_survive_pickle_and_deepcopy():
    """``BaseException.__reduce__`` replays args positionally, then restores ``__dict__``: a
    required keyword argument would turn a pickled refusal into a crash."""
    err = errors.EffectDataUnavailable(
        "m", cause=errors.EffectCause.TARGET_REGION_NOT_COVERED, details={"n_target_regions": 2}
    )
    for copied in (pickle.loads(pickle.dumps(err)), copy.deepcopy(err)):
        assert str(copied) == "m"
        assert copied.cause is errors.EffectCause.TARGET_REGION_NOT_COVERED
        assert copied.details == {"n_target_regions": 2}


def test_the_details_are_a_copy_of_what_was_passed():
    passed = {"n_rows": 0}
    err = errors.EffectDataUnavailable("m", cause=errors.EffectCause.EMPTY_COHORT, details=passed)
    passed["n_rows"] = 9
    assert err.details == {"n_rows": 0}


def test_a_cause_formats_as_its_value():
    """The tool maps causes by string value, so it never imports this package at load."""
    for cause in errors.EffectCause:
        assert str(cause) == cause.value == cause.name.lower()


def test_the_errors_module_imports_only_the_standard_library():
    tree = ast.parse(ERRORS_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import at line {node.lineno}"
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, sorted(imported)


def _effect_raises(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            if getattr(func, "id", getattr(func, "attr", None)) == "EffectDataUnavailable":
                yield node.lineno, node.exc


def _cause_violations(tree: ast.AST, label: str) -> list:
    """Raises whose ``cause`` is not a literal ``EffectCause.MEMBER``."""
    members = {c.name for c in errors.EffectCause}
    bad = []
    for lineno, call in _effect_raises(tree):
        value = next((kw.value for kw in call.keywords if kw.arg == "cause"), None)
        literal = (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "EffectCause"
            and value.attr in members
        )
        if not literal:
            shown = "no cause" if value is None else ast.unparse(value)
            bad.append(f"{label}:{lineno} {shown}")
    return bad


def _function(tree: ast.Module, class_name: str, method: str) -> ast.FunctionDef:
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method]
    return fn


def test_every_tool_path_effect_refusal_names_a_literal_cause():
    """The estimator and the cohort provider are the tool's path (the synthetic provider and
    the uplift estimator are not): an uncaused raise there would fall back to the generic code.
    """
    effect = REPO_ROOT / "src" / "digital_twin" / "effect"
    estimator = ast.parse((effect / "cohort_causal_estimator.py").read_text(encoding="utf-8"))
    provider = ast.parse((effect / "provider.py").read_text(encoding="utf-8"))
    get_frame = _function(provider, "CohortEffectDataProvider", "get_training_frame")
    # Floors, so the check cannot pass vacuously on a file it no longer understands.
    assert len(list(_effect_raises(estimator))) >= 10
    assert len(list(_effect_raises(get_frame))) >= 2
    bad = _cause_violations(estimator, "cohort_causal_estimator.py") + _cause_violations(
        get_frame, "provider.py"
    )
    assert bad == [], "EffectDataUnavailable raised without a literal cause:\n  " + "\n  ".join(bad)


def test_the_cause_check_flags_a_missing_or_non_literal_cause():
    source = """
def f():
    raise EffectDataUnavailable("a")
    raise EffectDataUnavailable("b", cause=cause)
    raise EffectDataUnavailable("c", cause="empty_cohort")
    raise EffectDataUnavailable("d", cause=EffectCause.NOT_A_MEMBER)
    raise EffectDataUnavailable("e", cause=EffectCause.EMPTY_COHORT)
"""
    bad = _cause_violations(ast.parse(source), "synthetic")
    assert [line.split()[0] for line in bad] == [
        "synthetic:3",
        "synthetic:4",
        "synthetic:5",
        "synthetic:6",
    ]


_NEVER_LOADED = ("src.agents.tool_composer", "src.api")


def _is_type_checking(test: ast.expr) -> bool:
    return getattr(test, "id", getattr(test, "attr", None)) == "TYPE_CHECKING"


def _module_file(root: Path, name: str) -> Path | None:
    path = root.joinpath(*name.split("."))
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else None


def _import_time_imports(module: str, path: Path):
    """``(lineno, name)`` for every ``src`` import that runs when ``module`` is imported: module
    and class bodies, including ``if``/``try`` arms, but not function bodies or ``TYPE_CHECKING``
    blocks. ``from x import y`` also yields ``x.y``, which is loaded when ``y`` is a submodule."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    stack = list(ast.parse(path.read_text(encoding="utf-8")).body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
            base = ".".join((anchor if node.level else []) + ([node.module] if node.module else []))
            names = [base] + [f"{base}.{alias.name}" for alias in node.names]
        else:
            stack.extend(ast.iter_child_nodes(node))
            continue
        yield from ((node.lineno, name) for name in names if name.split(".")[0] == "src")


def _import_closure(root: Path, start: str) -> dict:
    """Every ``src`` module importing ``start`` loads, mapped to the ``module:line`` that imported
    it (None for ``start`` and its parent packages). Static, so it costs no import."""
    reached: dict = {}
    queue = [(start, None)]
    while queue:
        name, via = queue.pop()
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            module = ".".join(parts[:depth])
            path = _module_file(root, module)
            if path is None or module in reached:
                continue
            reached[module] = via
            queue.extend(
                (imported, f"{module}:{line}")
                for line, imported in _import_time_imports(module, path)
            )
    return reached


def _forbidden_chains(reached: dict) -> list:
    chains = []
    for module in sorted(reached):
        if not any(module == p or module.startswith(p + ".") for p in _NEVER_LOADED):
            continue
        chain, via = [module], reached[module]
        while via:
            chain.append(via)
            via = reached[via.split(":")[0]]
        chains.append(" <- ".join(chain))
    return chains


def test_importing_the_errors_module_never_loads_the_tool_composer_or_the_api():
    """Replaces a subprocess import (15 s, ~560 MB, since it runs the twin package's
    ``__init__``) with the same question answered statically, transitive imports included: the
    twin package reaches ``src.agents.factory`` through ``src.mlops.opik_connector``, so only a
    closure, not a scan of ``src/digital_twin``, can see a path into the tool composer."""
    reached = _import_closure(REPO_ROOT, "src.digital_twin.effect.errors")
    # Floors, so a resolver that stopped following imports cannot pass vacuously.
    assert "src.digital_twin.simulation_engine" in reached
    assert "src.causal_engine" in reached
    assert len(reached) >= 100
    assert _forbidden_chains(reached) == []


def test_the_import_closure_names_a_transitive_path_into_the_tool_composer(tmp_path):
    files = {
        "src/__init__.py": "",
        "src/twin/__init__.py": "from . import errors\nfrom .engine import run\n",
        "src/twin/engine.py": (
            "from typing import TYPE_CHECKING\n"
            "import src.shared.helpers\n"
            "if TYPE_CHECKING:\n"
            "    import src.api.types\n"
            "def run():\n"
            "    import src.api.routes\n"
        ),
        "src/twin/errors.py": "import enum\n",
        "src/shared/__init__.py": "",
        "src/shared/helpers.py": "\nfrom src.agents.tool_composer import reason_codes\n",
        "src/agents/__init__.py": "",
        "src/agents/tool_composer/__init__.py": "",
        "src/agents/tool_composer/reason_codes.py": "",
        "src/api/__init__.py": "",
        "src/api/types.py": "",
        "src/api/routes.py": "",
    }
    for relative, text in files.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text, encoding="utf-8")

    chains = _forbidden_chains(_import_closure(tmp_path, "src.twin.errors"))

    # Found through the package __init__ and a relative import; the TYPE_CHECKING and
    # function-body imports of src.api never run at import time, so they are not reported.
    assert chains == [
        "src.agents.tool_composer <- src.shared.helpers:2 <- src.twin.engine:2 <- src.twin:2",
        "src.agents.tool_composer.reason_codes <- src.shared.helpers:2 <- src.twin.engine:2"
        " <- src.twin:2",
    ]
