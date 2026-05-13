"""CLI exit-gate tests for ``scripts/train_risk_score_model.py`` (issue #188).

The CLI must exit non-zero when ``honest_failures`` is non-empty so the
downstream Celery write task + CI pipelines cannot silently promote an
under-bar model. ``--allow-honest-failures`` opts out (used for synthetic-
noise plumbing tests).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_cli_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_risk_score_model_cli_gate",
        PROJECT_ROOT / "scripts" / "train_risk_score_model.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_risk_score_model_cli_gate"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
class TestCliExitGate:
    """Issue #188: CLI exit code reflects honest_failures state."""

    def test_cli_exits_zero_on_signal_cohort(self, tmp_path: Path) -> None:
        """Synthetic-smoke (high-signal) path with --allow-honest-failures
        path off but with explicit min_auc_pr=0.65 such that the bar IS met.

        The synthetic-smoke dataset is engineered to clear 0.65 AUC-PR, so
        even without --allow-honest-failures the CLI returns 0.
        """
        cli = _load_cli_module()
        json_out = tmp_path / "result.json"
        rc = cli.main(
            [
                "--synthetic-smoke",
                "--hpo-trials",
                "3",
                "--disable-mlflow",
                "--min-auc-pr",
                "0.65",
                "--json-out",
                str(json_out),
            ]
        )
        assert rc == 0, "synthetic-smoke (signal) cohort should clear 0.65 floor"
        assert json_out.exists()

    def test_cli_exits_nonzero_on_impossible_floor(self, tmp_path: Path) -> None:
        """Pin --min-auc-pr=1.0 to force honest_failures non-empty on a
        signal cohort that nonetheless cannot beat 1.0. CLI must exit
        non-zero (issue #188 enforcement).
        """
        cli = _load_cli_module()
        json_out = tmp_path / "result.json"
        rc = cli.main(
            [
                "--synthetic-smoke",
                "--hpo-trials",
                "3",
                "--disable-mlflow",
                "--min-auc-pr",
                "1.0",  # physically impossible
                "--json-out",
                str(json_out),
            ]
        )
        assert rc == 1, (
            "CLI must exit non-zero when honest_failures is non-empty "
            "(issue #188 enforcement)"
        )
        # JSON must still be emitted so the operator can inspect the failure
        # post-mortem.
        assert json_out.exists()

    def test_cli_allow_honest_failures_overrides_exit(self, tmp_path: Path) -> None:
        """--allow-honest-failures returns 0 even when honest_failures is
        non-empty. This is for synthetic-noise plumbing only.
        """
        cli = _load_cli_module()
        rc = cli.main(
            [
                "--synthetic-smoke",
                "--hpo-trials",
                "3",
                "--disable-mlflow",
                "--min-auc-pr",
                "1.0",  # forces honest_failures
                "--allow-honest-failures",
            ]
        )
        assert rc == 0


@pytest.mark.slow
class TestCliPrevalenceAwareFloor:
    """Issue #188: CLI uses prevalence-aware floor when --min-auc-pr is absent."""

    def test_prevalence_aware_floor_is_default(self, tmp_path: Path) -> None:
        """No --min-auc-pr -> trainer computes prevalence-aware floor.

        On synthetic-smoke (~30% positive cohort), the prevalence-aware
        floor is K*pi=1.5 clamped at 1.0 ceiling — physically impossible
        with this dataset, so honest_failures is populated. The CLI must
        exit non-zero unless --allow-honest-failures.
        """
        cli = _load_cli_module()
        rc = cli.main(
            [
                "--synthetic-smoke",
                "--hpo-trials",
                "3",
                "--disable-mlflow",
                # No --min-auc-pr -> prevalence-aware default.
            ]
        )
        # 5 * 0.30 = 1.50; even a perfect classifier scores < 1.50 (AUC-PR
        # is bounded by 1.0). So honest_failures IS populated.
        assert rc == 1
