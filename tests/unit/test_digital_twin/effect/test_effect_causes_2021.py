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
