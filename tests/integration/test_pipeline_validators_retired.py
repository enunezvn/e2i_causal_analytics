"""Drift guard: the vestigial PipelineValidator/StageValidator surface stays retired.

Converged decision 2026-06-05 (intent research + codex): src/causal_engine/pipeline/validators.py
was VESTIGIAL — the orphaned half of the Phase B7/B8 bifurcation (commit 26ce1fff); its siblings were
retired in #463/#465; it never checked acyclicity (a cyclic graph passed is_valid=True); it had zero
non-test callers. It was DELETED. This guard prevents accidental resurrection of the import surface,
mirroring tests/integration/test_validation_package.py.

Reversibility (~10 min):
    git restore --source=2b7ffb7f src/causal_engine/pipeline/validators.py \
        tests/unit/test_causal_engine/test_pipeline/test_validators.py
    # then restore the `from .validators import (...)` block + 7 __all__ entries in pipeline/__init__.py
"""

import importlib

import pytest

RETIRED_SYMBOLS = [
    "PipelineValidator",
    "StageValidator",
    "NetworkXToDoWhyValidator",
    "DoWhyToEconMLValidator",
    "EconMLToCausalMLValidator",
    "ValidationResult",
    "validate_pipeline_state",
]


def test_validators_module_is_deleted():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.causal_engine.pipeline.validators")


@pytest.mark.parametrize("symbol", RETIRED_SYMBOLS)
def test_validator_symbol_not_reexported(symbol):
    pkg = importlib.import_module("src.causal_engine.pipeline")
    assert not hasattr(pkg, symbol), (
        f"{symbol} is still re-exported from src.causal_engine.pipeline; the vestigial validator "
        "surface must stay retired (converged DELETE, 2026-06-05)"
    )
    assert symbol not in getattr(pkg, "__all__", [])
