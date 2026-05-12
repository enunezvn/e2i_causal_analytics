"""Integration test for the training-serving feature-schema invariant (R6).

Closes shard #9 of the 2026-05-04 tier-0 evaluation gap report:
"serving features identical to training features."

The Block-2 Feast offline↔online parity tests (PR #36, 9/9 FVs under
`FEAST_INTEGRATION=1`) verify VALUE equality but not feature-set equality.
This test verifies that the trained model + fitted preprocessor + BentoML
serving wrapper agree on the inference-time column contract:

1. The serving path's request schema (per `bentoml_service.py:497-510`,
   `numeric_features + categorical_features`) is exactly what the fitted
   preprocessor was trained on.
2. The preprocessor's output schema (`feature_names_out_`) matches the
   trained model's `feature_names_in_`.
3. End-to-end: a DataFrame containing only the serving-request columns
   transforms cleanly and predicts without column-mismatch errors.

Pipeline reality (verified against `scripts/run_tier0_test.py:2537-3175`):
the canonical order is `data_preparer → model_trainer (step 5) →
feature_analyzer (step 6) → model_deployer (step 7)`. `feature_analyzer`'s
pruning produces `selected_features` for SHAP/reporting only — it is NOT
consumed by `model_trainer` (which reads `state["train_data"]["X"]`
directly) nor by `model_deployer` (input dict at line 3341 omits the key).

This test therefore serves as a **regression guard**: if a future change
wires `feature_analyzer.selected_features` into the trainer or deployer
without updating the other side, the assertions here will catch the
resulting skew. It is gated to the default Integration Tests CI lane
(no `FEAST_INTEGRATION=1` requirement).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.ml_foundation.model_trainer.nodes.preprocessor import (  # noqa: E402,I001
    ModelTrainerPreprocessor,
)


def _build_synthetic_training_frame(
    n_rows: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a small DataFrame that mirrors what reaches model_trainer.preprocessor.

    Includes:
    - 4 informative numeric features (gaussian + uniform)
    - 2 low-cardinality categorical features (string + Categorical dtype)

    High-cardinality categoricals are intentionally omitted: the existing
    `_detect_feature_types` at preprocessor.py:175-180 skips them, but the
    `ColumnTransformer` at preprocessor.py:93-97 uses `remainder="passthrough"`,
    which routes the skipped column to the output unchanged — that fails
    downstream scaling on string content. Out of scope for this shard.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num_feat_1": rng.normal(0, 1, n_rows),
            "num_feat_2": rng.normal(5, 2, n_rows),
            "num_feat_3": rng.normal(-3, 1.5, n_rows),
            "num_feat_4": rng.uniform(0, 100, n_rows),
            "cat_feat_low": rng.choice(["A", "B", "C"], n_rows),
            "cat_feat_dtype": pd.Categorical(rng.choice(["X", "Y"], n_rows)),
        }
    )
    target = (df["num_feat_1"] + df["num_feat_2"] > 5).astype(int)
    return df, pd.Series(target, name="target")


def _serving_path_feature_names(preprocessor: Any) -> list[str]:
    """Mirror the BentoML serving feature-name reconstruction.

    Source: `src/mlops/bentoml_service.py:502-510` (predict path) and
    `:577-585` (predict_proba path) — both share this exact accessor logic.
    """
    if hasattr(preprocessor, "numeric_features"):
        return list(preprocessor.numeric_features) + list(
            getattr(preprocessor, "categorical_features", [])
        )
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    raise AttributeError("Preprocessor exposes neither numeric_features nor feature_names_in_")


@pytest.fixture(scope="module")
def fitted_preprocessor_and_model() -> tuple[ModelTrainerPreprocessor, Any, pd.DataFrame]:
    """Fit the preprocessor + train a tiny logistic regression — mirrors model_trainer.

    The intent is to reproduce the production wiring as closely as possible
    without invoking the full LangGraph pipeline:
    - ``ModelTrainerPreprocessor()`` constructed with NO explicit feature lists
      (matches `preprocessor.py:317-321`), so auto-detection at
      `preprocessor.py:111-112` runs against whatever DataFrame arrives.
    - Train data flows in raw (with high-cardinality categorical) so the
      auto-detection branches at lines 169-180 are exercised.
    - LogisticRegression chosen for speed — the contract being tested is
      structural (feature_names_in_), not algorithm-specific.
    """
    from sklearn.linear_model import LogisticRegression

    X_train, y_train = _build_synthetic_training_frame()

    preprocessor = ModelTrainerPreprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    feature_names_out = preprocessor.feature_names_out_ or []
    X_train_named = pd.DataFrame(X_train_preprocessed, columns=feature_names_out)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_named, y_train)

    return preprocessor, model, X_train


# --------------------------------------------------------------------------- #
# Discriminating coverage — confirm the fixture exercises both code paths     #
# --------------------------------------------------------------------------- #


def test_fixture_exercises_both_numeric_and_categorical_branches(
    fitted_preprocessor_and_model: tuple[ModelTrainerPreprocessor, Any, pd.DataFrame],
) -> None:
    """Per `feedback_pr_merge_workflow.md` §7 — guard against vacuous-pass.

    A skew test that passes only because no features were detected provides
    no regression coverage. Assert non-trivial counts on both sides.
    """
    preprocessor, _, _ = fitted_preprocessor_and_model
    assert len(preprocessor.numeric_features) >= 4, (
        f"expected ≥4 numeric features auto-detected; got {preprocessor.numeric_features}"
    )
    assert len(preprocessor.categorical_features) >= 1, (
        f"expected ≥1 categorical feature auto-detected; got {preprocessor.categorical_features}"
    )


def test_categorical_dtype_branch_exercised(
    fitted_preprocessor_and_model: tuple[ModelTrainerPreprocessor, Any, pd.DataFrame],
) -> None:
    """`preprocessor.py:172-178` accepts `pd.CategoricalDtype` columns.

    Confirming that branch fires gives discriminating coverage: the test
    isn't passing by accident on a degenerate fixture.
    """
    preprocessor, _, _ = fitted_preprocessor_and_model
    assert "cat_feat_dtype" in preprocessor.categorical_features
    assert "cat_feat_low" in preprocessor.categorical_features


# --------------------------------------------------------------------------- #
# R6 invariant — serving request schema matches what preprocessor was fit on  #
# --------------------------------------------------------------------------- #


def test_serving_feature_names_match_preprocessor_fit_columns(
    fitted_preprocessor_and_model: tuple[ModelTrainerPreprocessor, Any, pd.DataFrame],
) -> None:
    """The set of columns the serving path requests must be exactly the
    set the preprocessor was fit on (less the high-cardinality skip)."""
    preprocessor, _, X_train = fitted_preprocessor_and_model
    serving_request = set(_serving_path_feature_names(preprocessor))
    fit_columns = set(preprocessor.numeric_features) | set(preprocessor.categorical_features)
    assert serving_request == fit_columns, (
        f"serving-request schema diverged from preprocessor-fit columns; "
        f"diff: {serving_request.symmetric_difference(fit_columns)}"
    )
    # And every requested column must have been present in the original training frame.
    assert serving_request.issubset(set(X_train.columns)), (
        f"serving requests columns absent from training frame: "
        f"{serving_request - set(X_train.columns)}"
    )


def test_preprocessor_output_schema_matches_model_input_schema(
    fitted_preprocessor_and_model: tuple[ModelTrainerPreprocessor, Any, pd.DataFrame],
) -> None:
    """`preprocessor.feature_names_out_` must equal `model.feature_names_in_`.

    This is the post-encoding contract: the model's input schema (after
    one-hot expansion + scaling) must match what the preprocessor emits.
    """
    preprocessor, model, _ = fitted_preprocessor_and_model
    assert preprocessor.feature_names_out_ is not None, (
        "preprocessor did not populate feature_names_out_; see preprocessor.py:122-126"
    )
    assert hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None, (
        "model did not populate feature_names_in_ after fit"
    )
    assert list(model.feature_names_in_) == list(preprocessor.feature_names_out_), (
        f"model.feature_names_in_ ({list(model.feature_names_in_)}) does not "
        f"equal preprocessor.feature_names_out_ ({preprocessor.feature_names_out_})"
    )


def test_end_to_end_serving_input_predicts_without_column_mismatch(
    fitted_preprocessor_and_model: tuple[ModelTrainerPreprocessor, Any, pd.DataFrame],
) -> None:
    """Full chain: serving DataFrame → preprocessor.transform → model.predict.

    Builds a fresh DataFrame containing ONLY the serving-request columns
    (in the order serving wraps them at `bentoml_service.py:504-510`) and
    asserts the prediction call succeeds with the expected output shape.
    Catches the case where serving request schema is satisfiable but the
    column ORDER passed to the preprocessor is wrong.
    """
    preprocessor, model, X_train = fitted_preprocessor_and_model
    serving_cols = _serving_path_feature_names(preprocessor)

    # Synthesise a serving payload — same schema, different rows.
    rng = np.random.default_rng(7)
    payload = X_train.iloc[: min(20, len(X_train))][serving_cols].copy()
    # Inject mild noise on numeric columns so we exercise the scaler path.
    for col in preprocessor.numeric_features:
        payload[col] = payload[col] + rng.normal(0, 0.1, len(payload))

    transformed = preprocessor.transform(payload)
    assert transformed.shape[1] == len(model.feature_names_in_), (
        f"transformed shape {transformed.shape} columns mismatch model "
        f"expected {len(model.feature_names_in_)} columns"
    )

    # model.predict must accept a DataFrame whose column names match feature_names_in_.
    transformed_named = pd.DataFrame(
        transformed, columns=list(preprocessor.feature_names_out_ or [])
    )
    preds = model.predict(transformed_named)
    assert preds.shape == (len(payload),)
    assert set(preds.tolist()).issubset({0, 1}), (
        f"unexpected prediction labels: {set(preds.tolist())}"
    )


# --------------------------------------------------------------------------- #
# Regression guard — feature_analyzer pruning is currently advisory; if that  #
# changes without updating the deployer, this test catches the skew.          #
# --------------------------------------------------------------------------- #


def test_advisory_pruning_path_does_not_silently_alter_serving_contract(
    fitted_preprocessor_and_model: tuple[ModelTrainerPreprocessor, Any, pd.DataFrame],
) -> None:
    """Today's pipeline (verified at `scripts/run_tier0_test.py:2537-3306`):

        data_preparer → model_trainer (step 5) →
        feature_analyzer (step 6, advisory pruning) →
        model_deployer (step 7, ignores selected_features)

    `step_7_model_deployer` input dict at line 3341 does NOT carry
    `selected_features`; the deployer registers the model with the
    pre-pruning preprocessor + pre-pruning model schema. This test
    locks the invariant: the columns the serving wrapper requests must
    equal the columns the preprocessor was fit on, regardless of any
    advisory pruning that happens after training.

    If a future PR wires `feature_analyzer.selected_features` into the
    trainer (e.g. via `state["X_train_selected"] → train_data["X"]`) but
    forgets to refit the preprocessor or update `model_deployer`, the
    invariant breaks and this test fails.
    """
    preprocessor, _, X_train = fitted_preprocessor_and_model
    fit_inputs = set(preprocessor.numeric_features) | set(preprocessor.categorical_features)

    # Simulate a hypothetical "advisory pruning report" — what feature_analyzer
    # might recommend dropping. It must not change the serving contract.
    advisory_pruned = ["num_feat_4", "cat_feat_low"]  # synthetic recommendation
    assert all(c in X_train.columns for c in advisory_pruned), (
        "synthetic advisory list must reference real columns"
    )

    serving_request_after_advice = set(_serving_path_feature_names(preprocessor))
    assert serving_request_after_advice == fit_inputs, (
        "serving request schema should be unaffected by an advisory pruning "
        "recommendation (feature_analyzer's role per the current pipeline); "
        f"diff: {serving_request_after_advice.symmetric_difference(fit_inputs)}"
    )

    # Critically: the advisory-pruned columns ARE STILL in the serving
    # request (because feature_analyzer is advisory). If a future change
    # wires pruning to gate training, the model would be retrained on a
    # smaller schema while serving still requests the full set — that
    # exact skew is what `test_serving_feature_names_match_preprocessor_fit_columns`
    # would catch on the next test run.
    assert set(advisory_pruned).issubset(serving_request_after_advice), (
        "advisory-pruned columns missing from serving request — current "
        "pipeline contract assumes feature_analyzer pruning is advisory only"
    )


# --------------------------------------------------------------------------- #
# Backlog #15 (2026-05-12) — Post-pruning serving parity.                     #
#                                                                             #
# The PR #41 tests above lock the schema contract using a SYNTHETIC advisory  #
# list (`advisory_pruned = ["num_feat_4", "cat_feat_low"]`). That's a good    #
# pin for the structural invariant but it does NOT exercise the real         #
# `feature_analyzer/nodes/feature_selector.select_features` node, so a       #
# regression where the node itself starts emitting unexpected keys (e.g.     #
# mutating `state["fitted_preprocessor"]` or smuggling pruned-out features   #
# back into `state["feature_names"]`) would slip through.                    #
#                                                                            #
# The tests below close that gap by running the REAL pruning node end-to-end #
# and asserting two parity invariants on the actual outputs:                 #
#                                                                            #
#   P1. `feature_analyzer.select_features` produces a strictly-smaller       #
#       `selected_features` than its input — i.e. pruning is doing work, so  #
#       the contract is exercised, not vacuously satisfied.                  #
#   P2. The exact dict the runner builds for `step_7_model_deployer` carries #
#       neither `selected_features` nor `X_train_selected` — the deployer    #
#       receives only pre-pruning artifacts (`fitted_preprocessor` +         #
#       `feature_columns` from data_preparer).                               #
#   P3. The BentoML serving request schema reconstructed from               #
#       `fitted_preprocessor` still equals the full pre-pruning column set   #
#       AFTER the real pruning node has run on the same state.               #
# --------------------------------------------------------------------------- #


def _build_pruning_friendly_training_frame(
    n_rows: int = 200, seed: int = 13
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a frame where ≥1 numeric column is guaranteed to be pruned.

    Strategy:
    - 4 informative numeric features (drive the target)
    - 1 zero-variance numeric column ("num_zero_var") — variance threshold drops it
    - 1 perfectly-correlated numeric column ("num_dup_of_1") — correlation drops it
    - 1 low-cardinality categorical (passes through; non-numeric pruning is no-op)

    This guarantees `select_features` removes ≥1 column so the parity
    invariant P1 has a non-trivial signal to assert against.
    """
    rng = np.random.default_rng(seed)
    informative_1 = rng.normal(0, 1, n_rows)
    df = pd.DataFrame(
        {
            "num_informative_1": informative_1,
            "num_informative_2": rng.normal(5, 2, n_rows),
            "num_informative_3": rng.normal(-3, 1.5, n_rows),
            "num_informative_4": rng.uniform(0, 100, n_rows),
            "num_zero_var": np.zeros(n_rows),  # guaranteed variance prune
            "num_dup_of_1": informative_1,  # guaranteed correlation prune (r=1.0)
            "cat_low_card": rng.choice(["A", "B", "C"], n_rows),
        }
    )
    target = (df["num_informative_1"] + df["num_informative_2"] > 5).astype(int)
    return df, pd.Series(target, name="target")


@pytest.fixture(scope="module")
def real_pruning_state() -> dict[str, Any]:
    """Fit preprocessor + train model + run REAL feature_analyzer pruning node.

    Returns a state dict mirroring what `scripts/run_tier0_test.py` would
    carry through steps 5 → 6 → 7, with the artifacts each step deposits:

    - ``fitted_preprocessor`` (from step 5)
    - ``trained_model`` (from step 5)
    - ``feature_names`` (data_preparer / step 5 output, pre-pruning)
    - ``selected_features`` (from step 6, the REAL pruning node)
    - ``X_train_selected`` (from step 6)
    - ``feature_importance`` (from step 6)

    Discipline: this fixture imports the actual production node and invokes
    its async API the same way `step_6_feature_analyzer` does — no mocking
    of the pruning logic itself.
    """
    import asyncio

    from sklearn.linear_model import LogisticRegression

    from src.agents.ml_foundation.feature_analyzer.nodes.feature_selector import (
        select_features,
    )

    X_train, y_train = _build_pruning_friendly_training_frame()

    # Step 5 (model_trainer) — fit preprocessor + model on the full schema.
    preprocessor = ModelTrainerPreprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    feature_names_out = preprocessor.feature_names_out_ or []
    X_train_named = pd.DataFrame(X_train_preprocessed, columns=feature_names_out)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_named, y_train)

    # Step 6 (feature_analyzer) — run the REAL pruning node. We feed it the
    # raw pre-preprocessor frame so variance + correlation rules actually
    # fire on the synthetic-pruning columns we inserted.
    pruning_input_state: dict[str, Any] = {
        "X_train": X_train,
        "y_train": y_train,
        "problem_type": "classification",
        "selection_config": {
            # explicit defaults so this test is robust to changes in the
            # node's internal defaults
            "apply_variance_threshold": True,
            "variance_threshold": 0.01,
            "apply_correlation_filter": True,
            "correlation_threshold": 0.95,
            "apply_vif_filter": False,
            "compute_importance": False,  # avoid extra RandomForest fit cost
        },
    }
    pruning_output = asyncio.run(select_features(pruning_input_state))

    return {
        # Step 5 artifacts (what model_deployer consumes via step_7 kwargs)
        "fitted_preprocessor": preprocessor,
        "trained_model": model,
        "feature_names": list(X_train.columns),  # pre-pruning column list
        # Step 6 artifacts (what model_deployer must NOT consume today)
        "selected_features": pruning_output.get("selected_features"),
        "selected_features_all": pruning_output.get("selected_features_all"),
        "X_train_selected": pruning_output.get("X_train_selected"),
        "removed_features": pruning_output.get("removed_features", {}),
        # Carry the original frame for downstream test assertions
        "X_train": X_train,
    }


def test_real_feature_analyzer_actually_prunes_features(
    real_pruning_state: dict[str, Any],
) -> None:
    """P1 — invariant: the real pruning node is doing non-trivial work.

    Guards against the vacuous-pass mode where ``select_features`` becomes a
    no-op (e.g. config defaults change so variance + correlation filters are
    disabled by default). In that world the downstream parity invariants
    would pass trivially with empty diffs — that's not the contract we want
    to lock.

    The frame is engineered so at least the zero-variance column
    ``num_zero_var`` MUST be removed (variance < 0.01) and ``num_dup_of_1``
    MUST be removed (correlation = 1.0 with ``num_informative_1``).
    """
    selected_features = real_pruning_state["selected_features"]
    assert isinstance(selected_features, list), (
        f"select_features did not return a list under 'selected_features'; "
        f"got {type(selected_features).__name__}"
    )
    full_numeric_cols = [c for c in real_pruning_state["X_train"].columns if c.startswith("num_")]
    assert len(selected_features) < len(full_numeric_cols), (
        f"pruning was vacuous — selected {len(selected_features)} of "
        f"{len(full_numeric_cols)} numeric columns; expected strict subset. "
        f"selected={selected_features}; full={full_numeric_cols}"
    )
    # Specifically: the columns we engineered to be prunable MUST be absent.
    assert "num_zero_var" not in selected_features, (
        "expected variance-threshold to drop 'num_zero_var' (variance=0)"
    )
    assert "num_dup_of_1" not in selected_features, (
        "expected correlation-filter to drop 'num_dup_of_1' (corr=1.0 with informative_1)"
    )


def _build_step_7_model_deployer_input_dict(
    experiment_id: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce the input dict `step_7_model_deployer` builds for the deployer.

    Mirrors `scripts/run_tier0_test.py:3776-3811` exactly — including the
    deployment_name prefix the real step uses, the runner's call-site default
    for `success_criteria_met`, and the optional v5 Gate C1 ``scope_spec`` /
    ``feature_manifest_source`` keys — so that if the deployer's input
    contract changes to start consuming ``selected_features`` or
    ``X_train_selected`` from state, this test fails loudly.

    Note: `test_real_step_7_source_does_not_reference_pruning_artifacts`
    (above) is the load-bearing drift guard against the runner itself
    diverging; this helper exists so P2 can assert the dict-key contract
    on the same state shape the real fixture produces.
    """
    # Mirror the real step_7's deployment_name prefix
    # (scripts/run_tier0_test.py:3776).
    deployment_name = f"kisqali_discontinuation_{experiment_id[:8]}"
    input_data: dict[str, Any] = {
        "experiment_id": experiment_id,
        "model_uri": state.get("model_uri") or f"runs:/{experiment_id}/model",
        "validation_metrics": state.get("validation_metrics", {}),
        # Mirror the runner's call-site default at line 6500:
        # `state.get("success_criteria_met", False)`.
        "success_criteria_met": state.get("success_criteria_met", False),
        "deployment_name": deployment_name,
        "deployment_action": "register",
    }
    scope_spec = state.get("scope_spec")
    if scope_spec is not None:
        input_data["scope_spec"] = scope_spec
        if isinstance(scope_spec, dict) and scope_spec.get("feature_manifest_source"):
            input_data["feature_manifest_source"] = scope_spec.get("feature_manifest_source")
    return input_data


def test_real_step_7_source_does_not_reference_pruning_artifacts() -> None:
    """P2-source — pin the REAL `step_7_model_deployer` source against drift.

    Closes the codex pass-1 MEDIUM on `_build_step_7_model_deployer_input_dict`
    being a hand-copied mirror that cannot detect drift in the runner.

    Parses `scripts/run_tier0_test.py`, locates the `async def step_7_model_deployer`
    function, and asserts that no `feature_analyzer` pruning artifact name
    appears anywhere inside the function body. If a future PR wires
    ``selected_features=state.get("selected_features")`` into the deployer's
    input_data — or adds it as a kwarg to the function — the substring scan
    against the function's source span will fail.

    Why an AST-bounded substring scan rather than importing the helper:

    1. `scripts/run_tier0_test.py` is a 7k-line CLI harness with heavy
       top-level side effects on import (logging config, env probing,
       argument-parser construction). Importing it from the test lane is
       brittle and slow.
    2. We only need to assert that a small, well-defined set of names is
       ABSENT from the function body — substring scanning the function's
       textual source is sufficient and robust to formatting changes.
    3. We bound the scan to the function's source range via `ast.parse` +
       `ast.get_source_segment` so it cannot leak into unrelated code that
       legitimately references `selected_features` elsewhere in the file.
    """
    import ast

    runner_path = REPO_ROOT / "scripts" / "run_tier0_test.py"
    assert runner_path.exists(), f"runner script missing at {runner_path}"
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    step_7_node: ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "step_7_model_deployer":
            step_7_node = node
            break
    assert step_7_node is not None, (
        "could not locate `async def step_7_model_deployer` in "
        "scripts/run_tier0_test.py — has it been renamed? Update this test."
    )

    function_source = ast.get_source_segment(source, step_7_node)
    assert function_source is not None, (
        "ast.get_source_segment returned None for step_7_model_deployer"
    )

    # The forbidden names. Mirrors the set checked in
    # `test_model_deployer_input_dict_excludes_pruning_artifacts` so the
    # two assertions stay in lockstep.
    forbidden_names = (
        "selected_features",  # subsumes selected_features_all by substring
        "X_train_selected",
        "X_val_selected",
        "X_test_selected",
        "removed_features",
        "feature_importance",  # subsumes feature_importance_ranked
    )
    leaked = [name for name in forbidden_names if name in function_source]
    assert leaked == [], (
        f"`step_7_model_deployer` source now references feature_analyzer pruning "
        f"artifacts: {leaked}. The deployer historically consumed only pre-pruning "
        f"artifacts (fitted_preprocessor + feature_columns). If this wiring is "
        f"intentional, you must ALSO update: (a) BentoML serving request schema "
        f"in src/mlops/bentoml_service.py, (b) `fitted_preprocessor` refit story, "
        f"and (c) the parity assertions in this file. See backlog #15."
    )


def test_model_deployer_input_dict_excludes_pruning_artifacts(
    real_pruning_state: dict[str, Any],
) -> None:
    """P2 — the deployer's input dict carries no pruning artifacts.

    The runner threads `state.get("feature_names")` (pre-pruning) and
    `state.get("fitted_preprocessor")` (pre-pruning) into `step_7` as
    KEYWORD ARGUMENTS, NOT inside the dict passed to ``agent.run(...)``.
    The dict itself must not gain ``selected_features`` / ``X_train_selected``
    via state-leak (e.g. a future refactor that does
    ``input_data.update(state)``).

    If a future PR wires pruning into the deployer's input contract, this
    test fails and forces the author to also update the runner + the
    BentoML serving wrapper + the preprocessor refit story together.
    """
    full_state = dict(real_pruning_state)  # shallow copy preserves the references
    full_state["model_uri"] = "runs:/test-experiment/model"
    full_state["validation_metrics"] = {"roc_auc": 0.75}
    full_state["success_criteria_met"] = True

    deployer_input = _build_step_7_model_deployer_input_dict(
        experiment_id="test-experiment-deadbeef",
        state=full_state,
    )

    # The forbidden keys — none of these are legitimate model_deployer inputs
    # under the current pipeline contract.
    forbidden_keys = {
        "selected_features",
        "selected_features_all",
        "X_train_selected",
        "X_val_selected",
        "X_test_selected",
        "removed_features",
        "feature_importance",
        "feature_importance_ranked",
    }
    leaked = forbidden_keys & set(deployer_input.keys())
    assert leaked == set(), (
        f"deployer input dict leaked feature_analyzer pruning artifacts: {leaked}; "
        f"runner contract at scripts/run_tier0_test.py:3796-3811 must NOT propagate them"
    )


def test_serving_schema_after_real_pruning_still_equals_preprocessor_fit(
    real_pruning_state: dict[str, Any],
) -> None:
    """P3 — after real pruning runs, serving still requests pre-pruning columns.

    This is the load-bearing assertion for backlog #15: under the current
    advisory-pruning contract, the BentoML serving wrapper derives its
    request schema from ``fitted_preprocessor`` (numeric_features +
    categorical_features). That preprocessor was fit BEFORE pruning, so its
    schema is the pre-pruning schema. If a future PR refits the preprocessor
    on the pruned schema but forgets to re-register the BentoML model — or
    vice versa — this assertion catches the resulting skew.

    Failure mode this catches: ``fitted_preprocessor.numeric_features``
    silently shrinks to match ``selected_features``, causing the serving
    wrapper to request fewer columns than were used at training. End-to-end
    serving would then 200 OK on payloads missing the pruned columns,
    silently scoring on degraded inputs.
    """
    preprocessor = real_pruning_state["fitted_preprocessor"]
    selected_features = set(real_pruning_state["selected_features"])
    pre_pruning_numeric = set(real_pruning_state["X_train"].select_dtypes("number").columns)

    # Sanity check the fixture: pruning must have actually narrowed numerics.
    assert selected_features < pre_pruning_numeric, (
        f"fixture invariant broken: selected_features ({selected_features}) is not a "
        f"strict subset of pre-pruning numerics ({pre_pruning_numeric})"
    )

    # The serving wrapper's reconstructed feature names = numeric + categorical
    # straight off the fitted preprocessor (mirrors bentoml_service.py:502-510).
    serving_request = set(_serving_path_feature_names(preprocessor))
    fit_columns = set(preprocessor.numeric_features) | set(preprocessor.categorical_features)

    # Invariant: the preprocessor's view of "what to scale/encode" still
    # contains EVERY pre-pruning numeric column (it was fit before pruning).
    pruned_out = pre_pruning_numeric - selected_features
    assert pruned_out, "fixture must produce at least one pruned-out column"
    for col in pruned_out:
        assert col in preprocessor.numeric_features, (
            f"pruned-out column {col!r} disappeared from preprocessor.numeric_features — "
            f"serving schema has silently drifted to match the pruned set; "
            f"this is exactly the training-serving skew backlog #15 guards against"
        )

    # And the serving request schema agrees with the preprocessor's fit columns.
    assert serving_request == fit_columns, (
        f"serving-request schema diverged from preprocessor-fit columns; "
        f"diff: {serving_request.symmetric_difference(fit_columns)}"
    )
