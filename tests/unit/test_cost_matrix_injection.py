"""Unit tests for the demo cost matrix YAML extraction (Phase 5 Task 5.1).

Locks the contract that:
1. `_default_demo_cost_matrix()` reads from `config/cost_matrix_demo.yaml`,
   not from a hardcoded dict.
2. Modifying the YAML changes the injected values.
3. `_should_inject_demo_cost_matrix()` decision rule is preserved
   (skip when caller opts out OR when scope_spec already has a matrix).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Import the runner once at module load. The module-level constant
# `_DEMO_COST_MATRIX_PATH` resolves at import time to the real config file
# so we can patch it with `unittest.mock.patch` for per-test overrides.
_RUNNER = importlib.import_module("scripts.run_tier0_test")


def test_default_demo_cost_matrix_reads_real_yaml():
    """Happy path: the runner loads the canonical config/cost_matrix_demo.yaml."""
    matrix = _RUNNER._default_demo_cost_matrix()

    assert set(matrix.keys()) == {"tp", "fn", "fp", "tn"}
    # Lock the unit-shape values that downstream `tier0_remediation_baseline_20260426.md`
    # business_utility numbers were computed against. If these change, the
    # baseline (val=-8.15, test=-9.65) would shift and require re-baselining.
    assert matrix["tp"] == 1.0
    assert matrix["fn"] == -1.0
    assert matrix["fp"] == -0.05
    assert matrix["tn"] == 0.0
    # All values must be floats (CalibratedClassifierCV downstream requires it).
    for v in matrix.values():
        assert isinstance(v, float)


def test_modifying_yaml_changes_injected_values(tmp_path: Path):
    """Editing the YAML reflects in the next call.

    Confirms the injection truly reads at call time (not module-load time).
    """
    custom = tmp_path / "custom_matrix.yaml"
    custom.write_text(yaml.safe_dump({"tp": 2.5, "fn": -3.0, "fp": -0.2, "tn": 0.1}))

    with patch.object(_RUNNER, "_DEMO_COST_MATRIX_PATH", custom):
        matrix = _RUNNER._default_demo_cost_matrix()

    assert matrix["tp"] == 2.5
    assert matrix["fn"] == -3.0
    assert matrix["fp"] == -0.2
    assert matrix["tn"] == 0.1


def test_missing_keys_raise_key_error(tmp_path: Path):
    """A malformed YAML must fail loud, not silently inject a wrong matrix."""
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(yaml.safe_dump({"tp": 1.0, "fn": -1.0}))  # missing fp + tn

    with patch.object(_RUNNER, "_DEMO_COST_MATRIX_PATH", incomplete):
        with pytest.raises(KeyError, match="missing required keys"):
            _RUNNER._default_demo_cost_matrix()


def test_missing_yaml_raises_file_not_found(tmp_path: Path):
    """A missing YAML file must raise, not silently default."""
    nonexistent = tmp_path / "does_not_exist.yaml"

    with patch.object(_RUNNER, "_DEMO_COST_MATRIX_PATH", nonexistent):
        with pytest.raises(FileNotFoundError):
            _RUNNER._default_demo_cost_matrix()


# -----------------------------------------------------------------
# Decision-rule contract on `_should_inject_demo_cost_matrix`
# (already covered by Block-5B-I-2 elsewhere; we lock the call surface here
# so any future refactor that splits the helpers stays observable.)
# -----------------------------------------------------------------


def test_should_inject_when_no_matrix_present_and_inject_true():
    """Default path: caller did not opt out AND scope_spec has no matrix → inject."""
    assert _RUNNER._should_inject_demo_cost_matrix({}, inject=True) is True
    assert _RUNNER._should_inject_demo_cost_matrix({"cost_matrix": None}, inject=True) is True


def test_should_skip_when_caller_opts_out():
    """`--no-demo-cost-matrix` (inject=False) always skips, even with no matrix."""
    assert _RUNNER._should_inject_demo_cost_matrix({}, inject=False) is False
    assert _RUNNER._should_inject_demo_cost_matrix({"cost_matrix": None}, inject=False) is False


def test_should_skip_when_caller_supplied_matrix():
    """If scope_spec already has a matrix, never overwrite — even when inject=True."""
    real_matrix = {"tp": 100.0, "fn": -50.0, "fp": -5.0, "tn": 0.0}
    assert (
        _RUNNER._should_inject_demo_cost_matrix({"cost_matrix": real_matrix}, inject=True) is False
    )
