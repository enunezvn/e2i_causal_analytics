"""F17: cohort_constructor tier0_integration must import via the absolute package path.

The module-level `from cohort_constructor import …` (line 24) targeted a non-existent
top-level package → `ModuleNotFoundError` whenever tier0_integration was imported
(the data_source branch). Red-first: importing the module must succeed and expose
the symbols it pulls from the real `src.agents.cohort_constructor` package.
"""

import importlib


def test_tier0_integration_module_imports_cleanly():
    mod = importlib.import_module("src.agents.cohort_constructor.tier0_integration")
    assert mod is not None
    # Symbols imported at module scope must resolve from the real package.
    for sym in ("CohortConfig", "CohortConstructor", "Criterion", "Operator"):
        assert hasattr(mod, sym), f"{sym} did not resolve from src.agents.cohort_constructor"


def test_compare_cohorts_is_importable():
    # Guards the line-472 import used in the __main__ example block: compare_cohorts
    # lives in the .constructor submodule, not the package root.
    from src.agents.cohort_constructor.constructor import compare_cohorts

    assert callable(compare_cohorts)
