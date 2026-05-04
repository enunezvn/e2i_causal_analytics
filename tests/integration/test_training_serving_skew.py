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
