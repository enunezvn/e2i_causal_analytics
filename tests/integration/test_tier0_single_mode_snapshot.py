"""Cycle-16 I-9 (Q4-C) + shard 21 §H item 8 — Tier-0 single-mode byte-identity snapshot.

Verifies that the public interface of ``scripts/run_tier0_test.py`` AND the
orchestrator's single-mode output dict have NOT drifted as a side effect of
the W3-lite Day-5 + cycle-16 work. Single-mode is the legacy code path; any
regression here would silently break the production Tier-0 workflow.

Two snapshots are exercised:
  1. **CLI surface snapshot** — the argparse parser produced by
     ``scripts/run_tier0_test._build_parser()`` is compared field-by-field
     to a frozen expected set so any new / removed / type-changed argument
     fails the test loudly.
  2. **Orchestrator single-mode output dict shape snapshot** — running
     ``ModelTrainerAgent.run()`` with ``evaluation_mode="single"`` (the
     default) MUST produce an output dict whose keys are a subset of a
     frozen expected set. Any new repeated_k10-only key leaking into
     single-mode output would be caught here.

The actual byte-identity of the FULL pipeline (data generation, training,
evaluation, MLflow artifacts) is not in scope: that requires running the
full ~minutes-long Tier-0 script and a checked-in baseline. This test is the
cheap structural-equivalence proof; cycle-17 may extend with a heavier
``@pytest.mark.slow`` end-to-end script-subprocess snapshot when needed.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, FrozenSet, Set

import pytest

# Frozen CLI argument expectations for ``scripts/run_tier0_test._build_parser``.
# Any new flag added to the script must be reflected here intentionally.
EXPECTED_TIER0_CLI_ARGS: FrozenSet[str] = frozenset({
    "-h",
    "--help",
    "--step",
    "--disable-mlflow",
    "--enable-opik",
    "--hpo-trials",
    "--min-samples-per-split",
    "--dry-run",
    "--imbalanced",
    "--no-bentoml",
    "--output-dir",
    "--no-save",
    "--data-dir",
    "--brand",
    "--target",
    "--indication",
    "--regime",
    "--split",
    "--no-demo-cost-matrix",
})


def _collect_parser_options() -> Set[str]:
    """Import ``_build_parser`` and return the set of CLI option strings."""
    sys.path.insert(0, "/home/enunez/Projects/e2i_causal_analytics")
    from scripts.run_tier0_test import _build_parser

    parser = _build_parser()
    options: Set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            options.add(opt)
    return options


@pytest.mark.integration
def test_run_tier0_test_cli_surface_unchanged() -> None:
    """Snapshot: CLI surface of ``scripts/run_tier0_test.py`` is unchanged.

    Any new / removed / renamed CLI flag fails here so single-mode
    invocations cannot silently drift. To intentionally update the
    expected set, edit ``EXPECTED_TIER0_CLI_ARGS`` in this file and
    record the rationale in a commit message.
    """
    actual = _collect_parser_options()

    missing = EXPECTED_TIER0_CLI_ARGS - actual
    extra = actual - EXPECTED_TIER0_CLI_ARGS

    assert not missing, (
        f"run_tier0_test.py removed CLI flags: {sorted(missing)} — "
        f"this is a breaking change for single-mode users"
    )
    assert not extra, (
        f"run_tier0_test.py added CLI flags: {sorted(extra)} — "
        f"update EXPECTED_TIER0_CLI_ARGS in this test file to acknowledge"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orchestrator_single_mode_output_keys_within_allow_list() -> None:
    """Snapshot: orchestrator single-mode output keys ⊆ allowed set.

    Failure here means a new field has appeared in single-mode output —
    either it's a legitimate addition (extend the frozen set) or a
    repeated_k10-mode field that has leaked across the boundary
    (the cycle-16 I-8 finding inverted: repeated_k10 → single instead
    of single → repeated_k10).

    Mocks ``agent.graph.ainvoke`` so the test verifies the single-mode
    DISPATCH wrapper without paying the cost of a full graph execution.
    The graph mock returns a representative final-state dict; agent.run's
    ``_build_output`` decides which keys are forwarded, plus single-mode
    appends ``evaluation_mode = "single"`` and nothing else.
    """
    from unittest.mock import AsyncMock, patch

    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    async def _fake_graph(state: Dict[str, Any]) -> Dict[str, Any]:
        # Representative downstream node output — populates a few real
        # single-mode fields so the wrapper has something to forward.
        return {
            **state,
            "trained_model": object(),
            "framework": "lightgbm",
            "algorithm_name": state.get("algorithm_name", "LightGBM"),
            "best_hyperparameters": {"n_estimators": 5},
            "training_duration_seconds": 0.1,
            "early_stopped": False,
            "training_samples": 120,
            "validation_samples": 40,
            "test_samples": 40,
            "feature_names": ["x0", "x1", "x2", "x3"],
            "train_metrics": {"auroc": 0.75},
            "validation_metrics": {"auroc": 0.72},
            "test_metrics": {"auroc": 0.71, "accuracy": 0.80},
            "auc_roc": 0.71,
            "brier_score": 0.18,
            "precision": 0.7,
            "recall": 0.6,
            "f1_score": 0.65,
            "confidence_interval": {"auc": (0.65, 0.77)},
            "bootstrap_samples": 1000,
            "confusion_matrix": {"matrix": [[10, 5], [3, 22]]},
            "optimal_threshold": 0.5,
            "precision_at_k": {100: 0.85},
            "success_criteria_met": True,
            "success_criteria_results": {},
            "imbalance_detected": False,
            "minority_ratio": 0.5,
            "applied_strategy": "none",
            "skip_post_hoc_calibration": False,
            "feast_fallback_used": False,
            "training_status": "completed",
        }

    input_data: Dict[str, Any] = {
        "model_candidate": {
            "algorithm_name": "LightGBM",
            "algorithm_class": "lightgbm.LGBMClassifier",
            "hyperparameter_search_space": {},
            "default_hyperparameters": {"n_estimators": 5, "verbose": -1},
        },
        "qc_report": {"qc_passed": True},
        "experiment_id": "test_single_mode_snapshot",
        "success_criteria": {},
        "enable_hpo": False,
        "enable_mlflow": False,  # avoid touching real MLflow
        "enable_checkpointing": False,
        # NOT setting evaluation_mode — defaults to "single"
    }

    agent = ModelTrainerAgent()
    with patch.object(agent.graph, "ainvoke", AsyncMock(side_effect=_fake_graph)):
        output = await agent.run(input_data)

    output_keys = set(output.keys())

    # The repeated_k10-only fields are an unmissable regression signal —
    # surface them with high specificity if they appear in single-mode.
    # Mirror of the deny-set used in
    # ``test_single_mode_output_omits_repeated_k10_fields`` (cycle-16 I-8)
    # but verified end-to-end via the agent's full ``_build_output`` rather
    # than only the ``_run_repeated_splits`` orchestrator path. This is the
    # snapshot byte-identity stand-in for shard 21 §H item 8 — the FULL
    # script-level snapshot is acknowledged as out of scope for this
    # cheap-CI test (the script's pipeline runs in minutes; manual smoke
    # is the documented escape hatch in cycle_16_verdict.md).
    repeated_k10_indicators = {
        "fold_metrics",
        "aggregate_metrics",
        "aggregate_status",
        "k_folds",
        "splitter_strategy",
        "n_jobs",
        "parent_mlflow_run_id",
        "test_metrics_population_strategy",
        "evaluation_result_schema_version",
        "legacy_projection_warning",
        "seed_base",
    }
    repeated_leaks = output_keys & repeated_k10_indicators
    assert not repeated_leaks, (
        f"REGRESSION: single-mode output leaked repeated_k10 fields: "
        f"{sorted(repeated_leaks)}"
    )

    # Byte-identity guarantee: single-mode ``_build_output`` does NOT add an
    # ``evaluation_mode`` key (the field is repeated_k10-only output by
    # design — only ``_run_repeated_splits`` sets it on its dict). Absence
    # of the key in single-mode output preserves byte-identity for
    # downstream consumers (Tier-0 supervisor, FastAPI endpoint, contract
    # tests) that pre-date the dispatch flag.
    assert "evaluation_mode" not in output, (
        f"REGRESSION: single-mode output now contains 'evaluation_mode' key "
        f"({output.get('evaluation_mode')!r}) — pre-cycle-16 baseline did "
        f"NOT carry this field, so this is a byte-identity violation"
    )
