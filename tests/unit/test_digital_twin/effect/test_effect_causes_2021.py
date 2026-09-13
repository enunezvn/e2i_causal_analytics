"""#2021 Task 9b: an ``EffectDataUnavailable`` names its cause, not only its message.

Every twin effect refusal used to reach ``counterfactual_simulator`` as one message string, so
the tool refused them all under one generic reason code. The exception now carries a closed
``EffectCause`` and numeric details, which the tool maps to a per-cause code.

The contract lives in ``src/digital_twin/effect/errors.py``, which both the twin engine and the
tool composer import; it must stay standard-library only so importing it never pulls in the
tool composer package (~564 MB).
"""

from __future__ import annotations

import ast
import copy
import pickle
import subprocess
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


def test_importing_the_errors_module_does_not_import_the_tool_composer_package():
    code = (
        "import sys, src.digital_twin.effect.errors; "
        "print('src.agents.tool_composer' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert out.stdout.strip() == "False"
