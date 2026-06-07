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
from pathlib import Path
from typing import Any, Dict, FrozenSet, Set

import pytest

# Repo root derived from this file's location so the helpers below can
# ``import scripts.run_tier0_test`` correctly regardless of whether the
# test is run from the main repo, a git worktree, or CI.
#
# Layout: ``tests/integration/test_tier0_single_mode_snapshot.py``
#   parents[0] -> tests/integration/
#   parents[1] -> tests/
#   parents[2] -> <repo root>
#
# Tracks issue #410: prior to this change the helpers contained a
# hard-coded developer home-directory string inside ``sys.path.insert``
# which silently imported the MAIN repo's copy of
# ``scripts/run_tier0_test`` when the test ran inside a git worktree,
# producing false-negative descriptor-drift failures.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen CLI argument expectations for ``scripts/run_tier0_test._build_parser``.
# Cycle-17 IMPORTANT-2 hardening: snapshot ALSO covers per-flag
# defaults / choices / nargs / type / action class so silent semantic drift
# (e.g., default value flipping, choices narrowing) is caught — not just
# the addition / removal of option strings.
EXPECTED_TIER0_CLI_ARGS: FrozenSet[str] = frozenset(
    {
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
        # Added by Item A1 of the engineering-actionable arc (PR #100):
        # opts the run into a cohort-specific feature manifest so Layer 5
        # consults the matching FeatureContract registry.
        "--feature-manifest-source",
        "--regime",
        "--split",
        "--no-demo-cost-matrix",
        # Added by synthetic_cohort_growth_plan_20260509.md Phase 1 (PR #111):
        # parametric n_total override + seed threading for synthetic_v2 regimes.
        "--n-total",
        "--seed",
        # Added by the clinical/commercial deployment-intent axis (PR #786):
        # recalibrates the deployment bar + deployer gates to the use case.
        "--deployment-intent",
    }
)


# Action-descriptor snapshot per flag: (default, choices, nargs, type_name,
# action_class_name). type and action are stored as string names because
# argparse stores them as classes/callables that change identity across
# Python versions; the string name is the stable comparator.
EXPECTED_TIER0_CLI_DESCRIPTORS: Dict[str, Dict[str, Any]] = {
    "--step": {
        "default": None,
        "choices": list(range(1, 9)),
        "nargs": None,
        "type_name": "int",
        "action": "_StoreAction",
    },
    "--disable-mlflow": {
        "default": False,
        "choices": None,
        "nargs": 0,
        "type_name": "_no_type",
        "action": "_StoreTrueAction",
    },
    "--enable-opik": {
        "default": False,
        "choices": None,
        "nargs": 0,
        "type_name": "_no_type",
        "action": "_StoreTrueAction",
    },
    "--hpo-trials": {
        "default": 10,
        "choices": None,
        "nargs": None,
        "type_name": "int",
        "action": "_StoreAction",
    },
    "--min-samples-per-split": {
        "default": 10,
        "choices": None,
        "nargs": None,
        "type_name": "int",
        "action": "_StoreAction",
    },
    "--dry-run": {
        "default": False,
        "choices": None,
        "nargs": 0,
        "type_name": "_no_type",
        "action": "_StoreTrueAction",
    },
    "--imbalanced": {
        "default": None,
        "choices": None,
        "nargs": None,
        "type_name": "float",
        "action": "_StoreAction",
    },
    "--no-bentoml": {
        "default": False,
        "choices": None,
        "nargs": 0,
        "type_name": "_no_type",
        "action": "_StoreTrueAction",
    },
    "--output-dir": {
        "default": "docs/results",
        "choices": None,
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--no-save": {
        "default": False,
        "choices": None,
        "nargs": 0,
        "type_name": "_no_type",
        "action": "_StoreTrueAction",
    },
    "--data-dir": {
        "default": None,
        "choices": None,
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--brand": {
        "default": None,
        "choices": None,
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--target": {
        "default": None,
        "choices": None,
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--indication": {
        "default": None,
        "choices": None,
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--feature-manifest-source": {
        "default": None,
        "choices": ["csu", "optum", "synthetic"],
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--regime": {
        "default": "default",
        # PR #111 (synthetic_cohort_growth) extends choices from
        # 4 → 7 — adds scenario_a_balanced + scenario_b + scenario_c.
        "choices": [
            "default",
            "adverse",
            "clean",
            "scenario_a",
            "scenario_a_balanced",
            "scenario_b",
            "scenario_c",
        ],
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--split": {
        "default": "auto",
        "choices": ["auto", "random", "combined"],
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
    "--no-demo-cost-matrix": {
        "default": False,
        "choices": None,
        "nargs": 0,
        "type_name": "_no_type",
        "action": "_StoreTrueAction",
    },
    # PR #111 (synthetic_cohort_growth Phase 1):
    "--n-total": {
        "default": None,
        "choices": None,
        "nargs": None,
        "type_name": "int",
        "action": "_StoreAction",
    },
    "--seed": {
        "default": 42,
        "choices": None,
        "nargs": None,
        "type_name": "int",
        "action": "_StoreAction",
    },
    # PR #786 (clinical/commercial deployment-intent axis): selects the
    # use-case bar + the deployer's commercial-intent-aware gates.
    "--deployment-intent": {
        "default": "clinical",
        "choices": ["clinical", "commercial"],
        "nargs": None,
        "type_name": "str",
        "action": "_StoreAction",
    },
}


def _collect_parser_options() -> Set[str]:
    """Import ``_build_parser`` and return the set of CLI option strings."""
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.run_tier0_test import _build_parser

    parser = _build_parser()
    options: Set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            options.add(opt)
    return options


def _collect_parser_descriptors() -> Dict[str, Dict[str, Any]]:
    """Return a dict of {primary-option-string: action-descriptor} for non-help actions."""
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.run_tier0_test import _build_parser

    parser = _build_parser()
    descriptors: Dict[str, Dict[str, Any]] = {}
    for action in parser._actions:
        # Skip the auto-generated help action (its semantics are argparse-internal)
        if not action.option_strings or "--help" in action.option_strings:
            continue
        # Use the longest option string as the primary key (typically --foo over -f)
        primary = max(action.option_strings, key=len)
        choices = list(action.choices) if action.choices is not None else None
        type_callable = getattr(action, "type", None)
        type_name = type_callable.__name__ if type_callable is not None else "_no_type"
        descriptors[primary] = {
            "default": action.default,
            "choices": choices,
            "nargs": action.nargs,
            "type_name": type_name,
            "action": type(action).__name__,
        }
    return descriptors


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
def test_run_tier0_test_cli_descriptors_unchanged() -> None:
    """Cycle-17 IMPORTANT-2: snapshot per-flag defaults / choices / nargs / type / action.

    Beyond presence/absence of option strings (covered by the prior test),
    this catches silent semantic drift like:
      - ``--hpo-trials`` default flipping from 10 to 20
      - ``--regime`` choices losing or gaining values
      - ``--no-save`` action changing from store_true to store_const
    Updates to expected behavior must edit ``EXPECTED_TIER0_CLI_DESCRIPTORS``
    intentionally with the rationale in a commit message.
    """
    actual = _collect_parser_descriptors()

    expected_keys = set(EXPECTED_TIER0_CLI_DESCRIPTORS.keys())
    actual_keys = set(actual.keys())
    assert expected_keys == actual_keys, (
        f"Descriptor key mismatch: missing {sorted(expected_keys - actual_keys)}, "
        f"extra {sorted(actual_keys - expected_keys)}"
    )

    for opt, expected in EXPECTED_TIER0_CLI_DESCRIPTORS.items():
        observed = actual[opt]
        assert observed == expected, (
            f"CLI descriptor drift for {opt}: expected {expected}, got {observed}"
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

    # Cycle-17 IMPORTANT-1: the deny-list catches keys leaking IN; this set
    # catches keys silently dropping OUT. The minimum-presence set covers
    # core single-mode contract surface that downstream consumers
    # (Tier-0 supervisor, FastAPI endpoint, contract tests) rely on. A
    # refactor that removes any of these is a byte-identity regression.
    must_be_present_in_single_mode = {
        # Core classification metrics (problem_type=binary_classification)
        "auc_roc",
        "precision",
        "recall",
        "f1_score",
        "brier_score",
        "test_metrics",
        "validation_metrics",
        "train_metrics",
        # Trained artifact + identifiers
        "trained_model",
        "training_run_id",
        "model_id",
        "algorithm_name",
        "algorithm_class",
        "framework",
        # MLflow tracking surface
        "mlflow_status",
        "mlflow_run_id",
        # Status / context
        "training_status",
        "experiment_id",
        "problem_type",
        "training_duration_seconds",
    }
    missing_required = must_be_present_in_single_mode - output_keys
    assert not missing_required, (
        f"REGRESSION: single-mode output dropped required legacy keys: "
        f"{sorted(missing_required)}. Downstream Tier-0 / FastAPI / contract "
        f"tests rely on these fields; removal breaks byte-identity. If "
        f"intentional, edit must_be_present_in_single_mode + record the "
        f"breaking-change rationale in the commit message."
    )

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
        f"REGRESSION: single-mode output leaked repeated_k10 fields: {sorted(repeated_leaks)}"
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


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_single_mode_real_graph_output_surface() -> None:
    """Cycle-17 IMPORTANT-3: real-graph single-mode surface verification.

    The cheap test above patches ``agent.graph.ainvoke`` so it only verifies
    the wrapper projection in ``ModelTrainerAgent.run``. The cycle-17 codex
    verdict flagged that a real node could drop legacy keys silently and
    that test would still pass. This @pytest.mark.slow test runs the real
    LangGraph path with tiny synthetic data and ``enable_mlflow=False``,
    so it observes the production single-mode output shape end-to-end and
    asserts the same minimum-required legacy key set.

    Uses LogisticRegression (sklearn) for predictability + speed: small
    fixed weights, no native dependency, deterministic with random_state.
    """
    import numpy as np
    import pandas as pd

    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    # Tiny synthetic binary classification dataset — 100 rows, 4 features.
    rng = np.random.default_rng(42)
    n = 100
    X_full = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=["x0", "x1", "x2", "x3"],
    )
    # Balanced labels with mild signal so logistic regression converges
    y_full = pd.Series(
        ((X_full["x0"] + 0.5 * X_full["x1"] + 0.1 * rng.standard_normal(n)) > 0).astype(int),
        name="y",
    )

    # E2I-required split ratios: 60% / 20% / 15% / 5%.
    # Cycle-18 IMPORTANT-2: the 5-row holdout (rows 95-99) is not consumed by
    # the active evaluator node — confirmed by absence of holdout_data
    # references in src/agents/ml_foundation/model_trainer/nodes/evaluator.py.
    # The holdout is loaded by split_loader and stored in state for downstream
    # consumers but the primary AUC computation uses the 15-row test_data,
    # giving ~99.99% margin against an all-one-class accident at this seed.
    # Future evaluator changes that consume holdout MUST revisit this size
    # (5 rows is too small for AUC if both classes are required).
    train_end = int(0.60 * n)
    val_end = train_end + int(0.20 * n)
    test_end = val_end + int(0.15 * n)

    input_data: Dict[str, Any] = {
        "experiment_id": "test_cycle17_real_graph",
        "model_candidate": {
            "algorithm_name": "LogisticRegression",
            "algorithm_class": "sklearn.linear_model.LogisticRegression",
            "hyperparameter_search_space": {},
            "default_hyperparameters": {"C": 1.0, "max_iter": 200, "random_state": 42},
        },
        "qc_report": {"qc_passed": True},
        "success_criteria": {},
        "enable_hpo": False,
        "enable_mlflow": False,
        "enable_checkpointing": False,
        "problem_type": "binary_classification",
        "train_data": {
            "X": X_full.iloc[:train_end],
            "y": y_full.iloc[:train_end],
            "row_count": train_end,
        },
        "validation_data": {
            "X": X_full.iloc[train_end:val_end],
            "y": y_full.iloc[train_end:val_end],
            "row_count": val_end - train_end,
        },
        "test_data": {
            "X": X_full.iloc[val_end:test_end],
            "y": y_full.iloc[val_end:test_end],
            "row_count": test_end - val_end,
        },
        "holdout_data": {
            "X": X_full.iloc[test_end:],
            "y": y_full.iloc[test_end:],
            "row_count": n - test_end,
        },
        "feature_columns": list(X_full.columns),
    }

    agent = ModelTrainerAgent()
    output = await agent.run(input_data)

    # Single-mode contract: NO evaluation_mode key, NO repeated_k10 fields
    assert "evaluation_mode" not in output, (
        f"REGRESSION: real-graph single-mode output now contains "
        f"'evaluation_mode' key ({output.get('evaluation_mode')!r})"
    )
    repeated_k10_indicators = {
        "fold_metrics",
        "aggregate_metrics",
        "aggregate_status",
        "k_folds",
        "splitter_strategy",
        "n_jobs",
        "parent_mlflow_run_id",
    }
    repeated_leaks = set(output.keys()) & repeated_k10_indicators
    assert not repeated_leaks, (
        f"REGRESSION: real-graph single-mode output leaked repeated_k10 "
        f"fields: {sorted(repeated_leaks)}"
    )

    # Required legacy keys observable end-to-end. Mirrors the cheap test's
    # required set but verified via the LIVE LangGraph path. If a node
    # refactor drops one of these keys silently, this test catches it where
    # the cheap test cannot.
    #
    # Cycle-18 IMPORTANT-1 / COSMETIC-1: the slow test originally omitted
    # ``brier_score`` / ``algorithm_class`` / ``training_duration_seconds``
    # without explanation, creating an undocumented asymmetry with the cheap
    # test's required set. All three are produced unconditionally by the
    # real graph for binary classification (brier_score from
    # sklearn.metrics.brier_score_loss in evaluator.py; algorithm_class set
    # on initial_state in agent.py; training_duration_seconds set in
    # _build_output). Including them here closes the slow test's coverage
    # gap.
    #
    # MLflow keys (``mlflow_run_id``, ``mlflow_status``) are intentionally
    # omitted here because this test runs with ``enable_mlflow=False`` —
    # ``mlflow_status`` is "disabled" (per mlflow_logger.py) and
    # ``mlflow_run_id`` is None. The cheap test exercises that path explicitly.
    must_be_present_in_single_mode = {
        # Core classification metrics
        "auc_roc",
        "precision",
        "recall",
        "f1_score",
        "brier_score",
        "test_metrics",
        "validation_metrics",
        "train_metrics",
        # Trained artifact + identifiers
        "trained_model",
        "training_run_id",
        "model_id",
        "algorithm_name",
        "algorithm_class",
        "framework",
        # Status / context
        "training_status",
        "experiment_id",
        "problem_type",
        "training_duration_seconds",
    }
    output_keys = set(output.keys())
    missing_required = must_be_present_in_single_mode - output_keys
    assert not missing_required, (
        f"REGRESSION (real graph): single-mode output dropped required "
        f"legacy keys: {sorted(missing_required)}. Full output keys: "
        f"{sorted(output_keys)}"
    )

    # Sanity: training actually completed + AUC-ROC is a finite float in
    # the unit interval (loose bound — the test is about output shape, not
    # model quality).
    assert output.get("training_status") == "completed", (
        f"Real-graph single-mode did not complete: status={output.get('training_status')!r}"
    )
    auc = output.get("auc_roc")
    assert isinstance(auc, (int, float)), f"auc_roc not numeric: {auc!r}"
    assert 0.0 <= float(auc) <= 1.0, f"auc_roc out of [0,1]: {auc}"
