"""Unit tests for the ``data_transformer`` node.

Block 1B (#18): the misleading comment claiming ``LabelEncoder`` was fit on
"all unique values across splits" was deleted. These tests lock in the
correct behaviour: encoders fit on TRAIN only, with ``_safe_label_encode``
absorbing unseen categories at val/test time.

Block 6B (#17): ``scope_spec['exclude_columns']`` is now deprecated in
favour of the canonical ``scope_spec['excluded_features']``. Both keys are
honored at runtime, but populating ``exclude_columns`` emits a
``DeprecationWarning`` so callers can migrate.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.data_transformer import (
    _identify_column_types,
    _safe_label_encode,
    transform_data,
)


@pytest.fixture
def disjoint_categorical_splits() -> dict:
    """Build train/val/test with deliberately disjoint categorical sets.

    The categorical column ``brand`` carries values that exist ONLY in train,
    ONLY in val, and ONLY in test. If the encoder fits on the union, val/test
    rows would be encoded with an in-vocabulary id; if the encoder is
    train-only and ``_safe_label_encode`` is in place, val/test rows fall
    onto the sentinel id ``len(classes_)``.
    """
    train_df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    validation_df = pd.DataFrame(
        {
            "brand": ["A", "VAL_ONLY"],
            "value": [7.0, 8.0],
            "target": [1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "brand": ["B", "TEST_ONLY"],
            "value": [9.0, 10.0],
            "target": [0, 1],
        }
    )
    return {"train_df": train_df, "validation_df": validation_df, "test_df": test_df}


@pytest.fixture
def base_state(disjoint_categorical_splits) -> dict:
    """Wrap the splits in the state dict ``transform_data`` expects."""
    return {
        "experiment_id": "exp_block1b_data_transformer_test",
        "scope_spec": {
            "target_column": "target",
            "encoding_method": "label",
            # Use minmax for determinism — assertions below don't depend on it
            # but standard scaler can produce noisy near-zero values when the
            # variance is small, which complicates other test extensions.
            "scaling_method": "minmax",
            "imputation_strategy": "mean",
            "extract_datetime_features": False,
        },
        **disjoint_categorical_splits,
    }


class TestSafeLabelEncode:
    """Direct tests on the ``_safe_label_encode`` helper."""

    def test_unseen_values_get_sentinel_id(self):
        """Unseen labels must encode to ``len(classes_)``."""
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        encoder.fit(["A", "B", "C"])

        encoded = _safe_label_encode(encoder, pd.Series(["A", "B", "Z", "C"]))
        # A=0, B=1, Z=3 (sentinel), C=2 — the order depends on alphabetical fit.
        # We assert that the unseen value is mapped to len(classes_) = 3.
        assert encoded[2] == 3
        # Known values stay within [0, len(classes_)).
        assert encoded[0] < 3 and encoded[1] < 3 and encoded[3] < 3


@pytest.mark.asyncio
class TestEncoderFitTrainOnly:
    """Block 1B coverage: encoders MUST fit on train only.

    A regression where the encoder fits on the union of splits would be
    masked by the previous (now-deleted) comment that claimed exactly that
    behaviour. These tests exercise the actual branch.
    """

    async def test_label_encoder_classes_match_train_only(self, base_state):
        """``encoder.classes_`` must equal the unique train values verbatim."""
        result = await transform_data(base_state)

        encoders = result["encoders"]
        assert "brand" in encoders, "brand encoder must be fitted"
        encoder = encoders["brand"]

        train_uniques = sorted(base_state["train_df"]["brand"].astype(str).unique())
        assert sorted(encoder.classes_.tolist()) == train_uniques, (
            "Encoder classes_ must match train uniques exactly. If this "
            "fails, the encoder leaked val/test categories into its "
            "vocabulary — i.e., the deleted comment described real (and "
            "wrong) behaviour."
        )
        # Explicit check that val/test-only categories are NOT present.
        assert "VAL_ONLY" not in encoder.classes_
        assert "TEST_ONLY" not in encoder.classes_

    async def test_unseen_val_test_values_get_sentinel(self, base_state):
        """Val/test rows with train-unseen categories use the sentinel id."""
        result = await transform_data(base_state)

        encoders = result["encoders"]
        encoder = encoders["brand"]
        sentinel = len(encoder.classes_)
        # Train brand encodes to ids [0, 1, 2] (3 classes); sentinel = 3.
        # The MinMax scaler fits on train ⇒ max(train) = 2, min = 0, so the
        # scaled sentinel = (3 - 0) / (2 - 0) = 1.5. Anything < 1.5 is a
        # known category; exactly 1.5 is the sentinel.
        n_known = len(encoder.classes_)
        train_min, train_max = 0, n_known - 1
        expected_scaled_sentinel = (sentinel - train_min) / (train_max - train_min)

        validation_df = result["X_val"]
        test_df = result["X_test"]
        assert validation_df is not None and test_df is not None

        # Find the rows that were "VAL_ONLY" / "TEST_ONLY" in the input.
        original_val = base_state["validation_df"].reset_index(drop=True)
        val_only_idx = original_val.index[original_val["brand"] == "VAL_ONLY"][0]
        original_test = base_state["test_df"].reset_index(drop=True)
        test_only_idx = original_test.index[original_test["brand"] == "TEST_ONLY"][0]

        scaled_val_brand = validation_df["brand"].iloc[val_only_idx]
        scaled_test_brand = test_df["brand"].iloc[test_only_idx]
        assert scaled_val_brand == pytest.approx(expected_scaled_sentinel), (
            "Validation row with unseen category must be encoded at the "
            "sentinel value (= len(classes_) before scaling). If the encoder "
            "fitted on the union of splits, this row would have a known id "
            "instead and the scaled value would land within [0, 1]."
        )
        assert scaled_test_brand == pytest.approx(expected_scaled_sentinel), (
            "Test row with unseen category must use the sentinel encoding."
        )

    async def test_only_train_seen_values_stay_in_known_range(self, base_state):
        """Val/test rows whose category exists in train are encoded normally."""
        result = await transform_data(base_state)
        encoder = result["encoders"]["brand"]
        n_classes = len(encoder.classes_)

        # Validation row 0 (brand=A) must NOT be the sentinel post-scaling.
        # The sentinel post-MinMax = sentinel_id / (n_known - 1), which is
        # strictly greater than 1.0 when sentinel_id == n_known and
        # n_known > 1. Known categories produce values within [0, 1].
        validation_df = result["X_val"]
        assert validation_df is not None
        # Row 0 of validation has brand="A" which is known to the encoder.
        scaled_known = validation_df["brand"].iloc[0]
        assert 0.0 <= scaled_known <= 1.0, (
            f"Known category should scale within [0, 1], got {scaled_known}"
        )

        # Sanity: the encoder vocabulary is strictly smaller than the union.
        all_uniques = (
            set(base_state["train_df"]["brand"])
            | set(base_state["validation_df"]["brand"])
            | set(base_state["test_df"]["brand"])
        )
        assert n_classes < len(all_uniques)


@pytest.mark.asyncio
async def test_imputer_fits_on_train_only(disjoint_categorical_splits):
    """The imputer is also fit-on-train; verify with disjoint NaN patterns."""
    train_df = disjoint_categorical_splits["train_df"].copy()
    train_df.loc[0, "value"] = np.nan  # one NaN in train
    val_df = disjoint_categorical_splits["validation_df"].copy()
    val_df.loc[0, "value"] = np.nan  # NaN in val
    test_df = disjoint_categorical_splits["test_df"].copy()
    test_df.loc[0, "value"] = np.nan  # NaN in test

    state = {
        "experiment_id": "exp_imputer_train_only",
        "scope_spec": {
            "target_column": "target",
            "encoding_method": "label",
            "scaling_method": "minmax",
            "imputation_strategy": "mean",
            "extract_datetime_features": False,
        },
        "train_df": train_df,
        "validation_df": val_df,
        "test_df": test_df,
    }

    result = await transform_data(state)
    imputers = result["imputers"]
    assert "numeric" in imputers
    # SimpleImputer's ``statistics_`` is computed from the training data only.
    expected_train_mean = train_df["value"].dropna().mean()
    assert imputers["numeric"].statistics_[0] == pytest.approx(expected_train_mean)


# ---------------------------------------------------------------------------
# Block 6B (#17): exclude_columns / excluded_features consolidation
# ---------------------------------------------------------------------------


def _exclude_columns_state(extra_scope: dict | None = None) -> dict:
    """Build a minimal valid state for transform_data with overridable scope.

    The DataFrame includes a high-cardinality string ID column so that — if
    the exclusion is honored — the ``brand`` column is the only categorical
    candidate for label-encoding. If the ID were *not* excluded it would
    also be label-encoded (or dropped as high-cardinality), which is the
    behaviour we want to verify the deprecation path preserves.
    """
    train_df = pd.DataFrame(
        {
            "patient_id": ["p001", "p002", "p003", "p004"],
            "brand": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        }
    )
    scope_spec: dict = {
        "target_column": "target",
        "encoding_method": "label",
        "scaling_method": "minmax",
        "imputation_strategy": "mean",
        "extract_datetime_features": False,
    }
    if extra_scope:
        scope_spec.update(extra_scope)
    return {
        "experiment_id": "exp_exclude_columns_deprecation",
        "scope_spec": scope_spec,
        "train_df": train_df,
    }


def _encoded_column_names(transformations_applied: list[dict]) -> set[str]:
    """Collect every column name that participated in any transformation."""
    touched: set[str] = set()
    for entry in transformations_applied:
        for key in ("columns", "original_columns", "new_features"):
            cols = entry.get(key)
            if cols:
                touched.update(cols)
    return touched


@pytest.mark.asyncio
async def test_exclude_columns_deprecation_warning():
    """Populating ``exclude_columns`` emits a ``DeprecationWarning``."""
    state = _exclude_columns_state({"exclude_columns": ["patient_id"]})

    with pytest.warns(DeprecationWarning, match="exclude_columns"):
        result = await transform_data(state)

    # Behavioural contract: the legacy key still excludes the column from
    # transformation (encoding, scaling, imputation), so the warning is
    # advisory while the underlying behaviour is preserved.
    touched = _encoded_column_names(result["transformations_applied"])
    assert "patient_id" not in touched
    # And the canonical column (brand) was still encoded.
    assert "brand" in touched


@pytest.mark.asyncio
async def test_excluded_features_alone_no_warning():
    """Using only the canonical key must NOT trigger the warning."""
    state = _exclude_columns_state({"excluded_features": ["patient_id"]})

    with warnings.catch_warnings():
        # Treat any DeprecationWarning as an error — if the canonical path
        # somehow emits one, this turns the test red instead of silently
        # ignoring it.
        warnings.simplefilter("error", DeprecationWarning)
        result = await transform_data(state)

    # The canonical path keeps the column out of transformation.
    touched = _encoded_column_names(result["transformations_applied"])
    assert "patient_id" not in touched
    assert "brand" in touched


class TestIdentifyColumnTypesBoolRouting:
    """Bool dtype must route to categorical, not numeric.

    Mixing bool + int + float in a DataFrame slice forces ``.values``
    to fall back to object dtype because pandas cannot find a safe
    common numpy dtype. The downstream imputation step calls
    ``np.isnan(train_df[numeric_cols].values).any()`` which crashes
    with ``TypeError: ufunc 'isnan' not supported for the input types``
    on object arrays.

    Real CSU patient_journeys.json carries ``source_stacking_flag``
    (bool) alongside int/float numeric features; the existing
    ``test_csu_full_data_preparer_e2e.py`` fixture worked around this
    by manually dropping bool cols. Backlog item #12 surfaced the bug
    when the runner started routing through ``_load_from_files``.

    Treating bool as binary categorical produces 0/1 via label
    encoding — same downstream effect as numeric treatment, but with
    no mixed-dtype hazard.
    """

    def test_bool_column_routed_to_categorical(self) -> None:
        df = pd.DataFrame(
            {
                "flag": [True, False, True, False],
                "value": [1.0, 2.0, 3.0, 4.0],
                "count": [10, 20, 30, 40],
            }
        )
        numeric, categorical, datetime, _unhashable = _identify_column_types(df, exclude_columns=[])
        assert "flag" in categorical
        assert "flag" not in numeric

    def test_mixed_bool_int_float_no_object_array_after_filter(self) -> None:
        """The whole point: train_df[numeric_cols].values must NOT be object.

        Pinning this prevents future regressions of the np.isnan crash
        on real CSU."""
        df = pd.DataFrame(
            {
                "flag": [True, False, True, False],
                "value": [1.0, 2.0, 3.0, 4.0],
                "count": [10, 20, 30, 40],
            }
        )
        numeric, _, _, _ = _identify_column_types(df, exclude_columns=[])
        # With bool routed away, numeric_cols holds only int+float — common
        # dtype is float64, no object fallback.
        values = df[numeric].values
        assert values.dtype != object, (
            f"Expected non-object dtype but got {values.dtype}. "
            f"numeric_cols was {numeric}; if a bool column slipped in, the "
            f"bool-routing guard regressed."
        )
        # And np.isnan works on the slice (the actual downstream check).
        assert not np.isnan(values).any()

    def test_int_and_float_still_routed_to_numeric(self) -> None:
        """Sanity: the bool routing did not break the canonical numeric path."""
        df = pd.DataFrame(
            {
                "x_int": [1, 2, 3],
                "x_float": [1.1, 2.2, 3.3],
                "x_obj": ["a", "b", "c"],
            }
        )
        numeric, categorical, _, _ = _identify_column_types(df, exclude_columns=[])
        assert "x_int" in numeric
        assert "x_float" in numeric
        assert "x_obj" in categorical


class TestIdentifyColumnTypesUnhashableReport:
    """Object columns with list/dict/set/tuple/ndarray cells must be reported
    in the fourth return slot so ``transform_data`` drops them from all splits.

    Issue #197 (Codex pass-1 MEDIUM-1, 2026-05-14): leaving the columns
    in the frame just shifts the ``TypeError: unhashable type: 'list'``
    crash downstream to ``model_trainer/nodes/preprocessor.py::
    _detect_feature_types`` — same ``X[col].nunique()`` call — so
    ``_identify_column_types`` must REPORT the columns to its caller
    and ``transform_data`` must DROP them symmetrically across splits.
    Mirrors ``data_loader._drop_unhashable_columns`` semantics so the
    transformer's contract is consistent with the loader's.
    """

    def test_list_column_reported_as_unhashable(self) -> None:
        df = pd.DataFrame(
            {
                "comorbidities": [["E11", "I10"], [], ["J45"], []],
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": ["a", "b", "a", "c"],
            }
        )
        numeric, categorical, datetime, unhashable = _identify_column_types(df, exclude_columns=[])
        assert "comorbidities" in unhashable
        assert "comorbidities" not in numeric
        assert "comorbidities" not in categorical
        assert "comorbidities" not in datetime
        # Sanity: other columns still route correctly.
        assert "x" in numeric
        assert "y" in categorical

    def test_empty_list_only_column_reported(self) -> None:
        """Empty-list cells are still ``list`` instances. Real CSU
        ``patient_journeys.json`` has all-empty ``comorbidities`` and
        ``secondary_diagnosis_codes`` for every record sampled
        2026-05-14; the column type is object, the cells are ``[]``,
        and ``.nunique()`` still crashes with ``TypeError: unhashable
        type: 'list'``."""
        df = pd.DataFrame(
            {
                "comorbidities": [[], [], [], []],
                "secondary_diagnosis_codes": [[], [], [], []],
                "scalar": [1, 2, 3, 4],
            }
        )
        numeric, categorical, _, unhashable = _identify_column_types(df, exclude_columns=[])
        assert "comorbidities" in unhashable
        assert "secondary_diagnosis_codes" in unhashable
        assert "comorbidities" not in numeric
        assert "comorbidities" not in categorical
        assert "secondary_diagnosis_codes" not in numeric
        assert "secondary_diagnosis_codes" not in categorical
        assert "scalar" in numeric

    def test_ndarray_column_reported(self) -> None:
        """``numpy.ndarray`` cells (the Parquet→pandas roundtrip shape)
        must also surface as unhashable — same crash path as Python
        lists."""
        df = pd.DataFrame(
            {
                "vec": [
                    np.array([1, 2]),
                    np.array([3]),
                    np.array([]),
                    np.array([4, 5, 6]),
                ],
                "scalar": [1.0, 2.0, 3.0, 4.0],
            }
        )
        numeric, categorical, _, unhashable = _identify_column_types(df, exclude_columns=[])
        assert "vec" in unhashable
        assert "vec" not in numeric
        assert "vec" not in categorical
        assert "scalar" in numeric

    def test_dict_column_reported(self) -> None:
        df = pd.DataFrame(
            {
                "meta": [{"k": 1}, {"k": 2}, {}, {"k": 3}],
                "scalar": [1, 2, 3, 4],
            }
        )
        numeric, categorical, _, unhashable = _identify_column_types(df, exclude_columns=[])
        assert "meta" in unhashable
        assert "meta" not in numeric
        assert "meta" not in categorical
        assert "scalar" in numeric

    def test_object_column_with_strings_still_categorical(self) -> None:
        """Regression: hashable object cells (plain strings) still route
        to categorical — the guard only fires on unhashable cells."""
        df = pd.DataFrame(
            {
                "diag": ["L50.1", "L50.1", "L50.2", "L50.9"],
                "scalar": [1, 2, 3, 4],
            }
        )
        numeric, categorical, _, unhashable = _identify_column_types(df, exclude_columns=[])
        assert "diag" in categorical
        assert "diag" not in numeric
        assert "diag" not in unhashable


@pytest.mark.asyncio
class TestTransformDataDropsUnhashableColumns:
    """``transform_data`` must DROP list-typed columns from all split
    frames (not just skip them from encoding) so downstream consumers
    (model_trainer preprocessor) see a clean encodable feature surface.

    Codex pass-1 MEDIUM-1 (2026-05-14): leaving columns in
    ``X_train``/``feature_columns`` shifts the
    ``TypeError: unhashable type: 'list'`` crash downstream to
    ``model_trainer/nodes/preprocessor.py::_detect_feature_types``
    where it calls the same ``X[col].nunique()``. Mirror the
    ``data_loader._drop_unhashable_columns`` semantics here so the
    transformer's contract is consistent with the loader's.
    """

    async def test_transform_data_drops_list_columns_from_all_splits(self) -> None:
        train_df = pd.DataFrame(
            {
                "comorbidities": [["E11", "I10"], [], ["J45"], []],
                "secondary_diagnosis_codes": [[], ["B97.4"], [], []],
                "age": [25.0, 60.0, 45.0, 30.0],
                "gender": ["M", "F", "F", "M"],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_list_columns",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "encoding_method": "label",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],  # Deliberately NOT excluding list cols
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None, f"transform_data crashed with: {result.get('error')!r}"
        assert result.get("error_type") is None

        # The transformer MUST drop list-typed columns from the frame
        # (Codex pass-1 MEDIUM-1). Otherwise model_trainer's
        # _detect_feature_types crashes on the same nunique() call.
        X_train = result["X_train"]
        X_val = result.get("X_val")
        X_test = result.get("X_test")
        assert "comorbidities" not in X_train.columns
        assert "secondary_diagnosis_codes" not in X_train.columns
        assert X_val is not None and "comorbidities" not in X_val.columns
        assert X_val is not None and "secondary_diagnosis_codes" not in X_val.columns
        assert X_test is not None and "comorbidities" not in X_test.columns
        assert X_test is not None and "secondary_diagnosis_codes" not in X_test.columns

        # feature_columns reflects the post-drop schema.
        feature_columns = result.get("feature_columns") or []
        assert "comorbidities" not in feature_columns
        assert "secondary_diagnosis_codes" not in feature_columns

        # Scalars still encoded — sanity check.
        assert "age" in feature_columns
        assert "gender" in feature_columns

        # transformations_applied surfaces the drop for auditability.
        transformations = result.get("transformations_applied") or []
        drop_entries = [t for t in transformations if t.get("type") == "drop_unhashable_columns"]
        assert len(drop_entries) == 1, (
            f"Expected one drop_unhashable_columns entry; "
            f"got {len(drop_entries)}. transformations_applied={transformations!r}"
        )
        dropped_set = set(drop_entries[0].get("columns") or [])
        assert dropped_set == {"comorbidities", "secondary_diagnosis_codes"}

    async def test_transform_data_drops_only_unhashable_keeps_others(self) -> None:
        """Regression: the drop is precise — only list-typed columns leave
        the frame; numeric and string-object columns survive intact."""
        train_df = pd.DataFrame(
            {
                "comorbidities": [["E11", "I10"], [], ["J45"], []],
                "diag": ["L50.1", "L50.1", "L50.2", "L50.9"],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_precise_drop",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        X_train = result["X_train"]
        # comorbidities dropped; diag + age survive (diag as encoded
        # categorical, age as scaled numeric). #790: with no explicit
        # encoding_method the nominal ``diag`` one-hot expands (``diag_*``),
        # so the bare column is replaced by its expansion — the information
        # still survives the unhashable drop, which is what this test guards.
        assert "comorbidities" not in X_train.columns
        diag_cols = [c for c in X_train.columns if c == "diag" or c.startswith("diag_")]
        assert diag_cols, f"diag information lost after drop: {X_train.columns.tolist()}"
        assert "age" in X_train.columns

    async def test_transform_data_no_unhashable_no_drop_transformation(self) -> None:
        """Regression: when no list columns are present, no
        ``drop_unhashable_columns`` transformation is recorded."""
        train_df = pd.DataFrame(
            {
                "diag": ["L50.1", "L50.1", "L50.2", "L50.9"],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_no_drop",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        transformations = result.get("transformations_applied") or []
        drop_entries = [t for t in transformations if t.get("type") == "drop_unhashable_columns"]
        assert drop_entries == [], (
            f"Expected NO drop_unhashable_columns when no list cols present; got {drop_entries!r}"
        )

    async def test_transform_data_output_passes_model_trainer_detect(self) -> None:
        """Cross-agent contract: transform_data output must be safe for the
        model_trainer preprocessor's auto-detect. This pins the surgical
        Codex MEDIUM-1 fix by composing both nodes — if the transformer
        leaves a list column in X_train, the preprocessor's
        ``_detect_feature_types`` crashes on ``X[col].nunique()``.
        """
        from src.agents.ml_foundation.model_trainer.nodes.preprocessor import (
            ModelTrainerPreprocessor,
        )

        train_df = pd.DataFrame(
            {
                "comorbidities": [["E11", "I10"], [], ["J45"], []],
                "age": [25.0, 60.0, 45.0, 30.0],
                "gender": ["M", "F", "F", "M"],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_cross_agent_contract",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        X_train = result["X_train"]
        # Without the drop, _detect_feature_types crashes at
        # ``n_unique = X[col].nunique()`` on the comorbidities column.
        # With the drop, this is a noop list comprehension.
        preprocessor = ModelTrainerPreprocessor()
        preprocessor._detect_feature_types(X_train)  # must NOT raise


@pytest.mark.asyncio
class TestTransformDataThreadsCleanedFramesIntoState:
    """``transform_data`` must thread the cleaned frames back into state
    under the canonical ``train_df``/``validation_df``/``test_df``/
    ``holdout_df`` keys so downstream nodes in the data_preparer graph
    consume the post-drop schema.

    Codex pass-2 MEDIUM-2 (2026-05-14): the prior fix dropped unhashable
    columns only from the local working copies returned as
    ``X_train``/``X_val``/``X_test``/``X_holdout``. The downstream
    nodes ``feast_registrar``, ``compute_baseline_metrics``, and
    ``finalize_output`` consume ``state.get("train_df")`` — which
    points at the ORIGINAL (uncleaned) frame. Result: list cells
    crash ``baseline_computer``'s ``value_counts()``/``nunique()``
    when a list col is in ``required_features``, and would silently
    expose list cols to Feast registration.

    Contract: when ``transform_data`` drops unhashable cols, it MUST
    also surface the cleaned (pre-transformation, target-preserved)
    frames under the canonical state keys so LangGraph's state-merge
    replaces the originals.
    """

    async def test_state_train_df_is_cleaned_when_unhashable_dropped(self) -> None:
        train_df = pd.DataFrame(
            {
                "comorbidities": [["E11", "I10"], [], ["J45"], []],
                "age": [25.0, 60.0, 45.0, 30.0],
                "gender": ["M", "F", "F", "M"],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_state_thread",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        # Codex pass-2 MED-2: state's train_df MUST be replaced with the
        # cleaned frame (target preserved, list cols dropped).
        assert "train_df" in result, (
            "transform_data MUST surface cleaned train_df in state delta "
            "when unhashable cols were dropped (Codex pass-2 MED-2)"
        )
        assert "validation_df" in result
        assert "test_df" in result
        state_train_df = result["train_df"]
        assert "comorbidities" not in state_train_df.columns
        # Target column preserved in state's train_df (canonical contract).
        assert "target" in state_train_df.columns
        # Sanity: feature cols preserved.
        assert "age" in state_train_df.columns
        assert "gender" in state_train_df.columns

        # Pre-transformation schema preserved — train_df["age"] is NOT
        # scaled (only X_train is scaled).
        assert state_train_df["age"].iloc[0] == 25.0

    async def test_baseline_computer_runs_on_cleaned_state_train_df(self) -> None:
        """Cross-node contract: after transform_data drops list cols, the
        ``compute_baseline_metrics`` node MUST be able to read
        ``state.get("train_df")`` and compute stats on
        ``required_features`` (including categorical cols) without
        crashing on ``value_counts()`` / ``nunique()``.

        This is the surgical pin for Codex pass-2 MEDIUM-2: pre-fix,
        if ``comorbidities`` was in ``required_features``, baseline
        crashed on ``value_counts()``. Post-fix, baseline either
        finds it absent from the cleaned state's train_df (warns and
        continues) or it's been pruned upstream.
        """
        from src.agents.ml_foundation.data_preparer.nodes.baseline_computer import (
            compute_baseline_metrics,
        )

        train_df = pd.DataFrame(
            {
                "comorbidities": [["E11", "I10"], [], ["J45"], []],
                "age": [25.0, 60.0, 45.0, 30.0],
                "gender": ["M", "F", "F", "M"],
                "target": [0, 1, 0, 1],
            }
        )
        state: dict = {
            "experiment_id": "test_issue_197_baseline_runs",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "prediction_target": "target",
                # Crucially: list col is declared as a required feature.
                # Pre-fix this caused baseline_computer to call
                # value_counts() on it and crash.
                "required_features": ["comorbidities", "age", "gender"],
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        # Simulate LangGraph's state-merge: apply transform_data's
        # state delta to the input state.
        transform_result = await transform_data(state)
        assert transform_result.get("error") is None
        merged_state = {**state, **transform_result}
        # Now run the downstream node against the merged state.
        baseline_result = await compute_baseline_metrics(merged_state)
        assert baseline_result.get("error") is None, (
            f"baseline_computer crashed: {baseline_result.get('error')!r}. "
            f"This means transform_data's state-train_df was not cleaned, "
            f"and the value_counts()/nunique() in baseline_computer "
            f"re-tripped the same TypeError: unhashable type: 'list' "
            f"crash that the issue #197 fix was supposed to close."
        )

    async def test_no_state_train_df_update_when_no_unhashable(self) -> None:
        """Regression: when no list columns are present, transform_data
        does NOT update state's train_df/validation_df/test_df/holdout_df
        — preserves the original state-contract for the canonical path."""
        train_df = pd.DataFrame(
            {
                "diag": ["L50.1", "L50.1", "L50.2", "L50.9"],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_no_state_thread",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        # No unhashable cols → no state-train_df overwrite.
        assert "train_df" not in result
        assert "validation_df" not in result
        assert "test_df" not in result
        assert "holdout_df" not in result


@pytest.mark.asyncio
class TestTransformDataSplitSkewUnhashableCells:
    """Codex pass-3 MEDIUM-3 (2026-05-14): the unhashable-col scan must
    take the UNION across all splits, not just train. A column may be
    scalar/null in train but list-typed in val/test (JSON-decoded CSU
    cohorts can split-skew on column types), or a list-only column may
    be absent from train altogether. The drop set must cover all splits.

    Pre-fix: ``unhashable_cols`` was built from train_df alone. Under
    split skew, val/test still carried list cells → downstream
    state-frames still crashed on ``nunique()``/``value_counts()``.
    """

    async def test_validation_only_list_column_dropped(self) -> None:
        """List cells in validation but scalar in train → still must drop."""
        train_df = pd.DataFrame(
            {
                "comorbidities": ["scalar1", "scalar2", "scalar3", "scalar4"],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        validation_df = pd.DataFrame(
            {
                # Same column, but list-typed in val (split skew).
                "comorbidities": [["E11"], [], ["I10"], []],
                "age": [40.0, 55.0, 50.0, 35.0],
                "target": [1, 0, 1, 0],
            }
        )
        state = {
            "experiment_id": "test_issue_197_val_only_list",
            "train_df": train_df,
            "validation_df": validation_df,
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        # State's validation_df must have the list col dropped.
        assert "validation_df" in result, (
            "transform_data MUST surface cleaned validation_df even when "
            "the unhashable col is only present in val (Codex pass-3 MED-3)"
        )
        state_val_df = result["validation_df"]
        assert "comorbidities" not in state_val_df.columns
        # And X_val output also clean.
        X_val = result["X_val"]
        assert "comorbidities" not in X_val.columns

    async def test_test_only_list_column_dropped(self) -> None:
        """List cells in test but scalar in train+val → still must drop."""
        train_df = pd.DataFrame(
            {
                "secondary_diagnosis_codes": ["L50.1", "L50.1", "L50.2", "L50.9"],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        test_df = pd.DataFrame(
            {
                # Same column, but list-typed in test (split skew).
                "secondary_diagnosis_codes": [["B97.4"], [], ["J45.0"], []],
                "age": [40.0, 55.0, 50.0, 35.0],
                "target": [1, 0, 1, 0],
            }
        )
        state = {
            "experiment_id": "test_issue_197_test_only_list",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": test_df,
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        assert "test_df" in result
        state_test_df = result["test_df"]
        assert "secondary_diagnosis_codes" not in state_test_df.columns

    async def test_union_drop_across_all_splits(self) -> None:
        """Different splits carry list cells in different columns →
        union of unhashable cols dropped from ALL splits."""
        train_df = pd.DataFrame(
            {
                "a": [["x"], [], ["y"], []],
                "b": ["s1", "s2", "s3", "s4"],
                "c": ["t1", "t2", "t3", "t4"],
                "target": [0, 1, 0, 1],
            }
        )
        validation_df = pd.DataFrame(
            {
                "a": ["s1", "s2", "s3", "s4"],
                "b": [["E11"], [], ["I10"], []],
                "c": ["t5", "t6", "t7", "t8"],
                "target": [1, 0, 1, 0],
            }
        )
        test_df = pd.DataFrame(
            {
                "a": ["s5", "s6", "s7", "s8"],
                "b": ["s9", "s10", "s11", "s12"],
                "c": [["X1"], [], ["X2"], []],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_union_drop",
            "train_df": train_df,
            "validation_df": validation_df,
            "test_df": test_df,
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        # Union {a, b, c} dropped from ALL splits.
        for key in ("train_df", "validation_df", "test_df"):
            assert key in result, f"{key} not surfaced — pass-3 MED-3 regression"
            for col in ("a", "b", "c"):
                assert col not in result[key].columns, (
                    f"{col} survived in state[{key}] — pass-3 union-drop failed"
                )
            assert "target" in result[key].columns

    async def test_drop_unhashable_columns_metadata_lists_union(self) -> None:
        """The transformations_applied entry surfaces the FULL union of
        unhashable cols across splits, not just train's."""
        train_df = pd.DataFrame(
            {
                "train_only_list": [["x"], [], ["y"], []],
                "val_only_list": ["s1", "s2", "s3", "s4"],
                "target": [0, 1, 0, 1],
            }
        )
        validation_df = pd.DataFrame(
            {
                "train_only_list": ["s5", "s6", "s7", "s8"],
                "val_only_list": [["E11"], [], ["I10"], []],
                "target": [1, 0, 1, 0],
            }
        )
        state = {
            "experiment_id": "test_issue_197_metadata_union",
            "train_df": train_df,
            "validation_df": validation_df,
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
                "excluded_features": [],
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        transformations = result.get("transformations_applied") or []
        drop_entries = [t for t in transformations if t.get("type") == "drop_unhashable_columns"]
        assert len(drop_entries) == 1
        dropped_set = set(drop_entries[0].get("columns") or [])
        # Both train-only-list AND val-only-list must surface as dropped.
        assert dropped_set == {"train_only_list", "val_only_list"}


@pytest.mark.asyncio
class TestTransformDataSafetyDropIndependentOfExcludedFeatures:
    """Codex pass-4 MEDIUM-4 (2026-05-14): the unhashable safety drop
    must run INDEPENDENTLY of ``excluded_features`` /
    ``exclude_columns``. The transformation-exclusion path only
    suppresses encoding/scaling — it does NOT remove columns from the
    returned frames. A list-typed column placed in ``excluded_features``
    would bypass the safety drop and survive into X_* / state frames,
    re-tripping the ``nunique()`` crash in
    ``model_trainer/nodes/preprocessor.py::_detect_feature_types``.

    Safety-drop is a hazard mitigation, not a transformation policy —
    it must apply regardless of caller intent.
    """

    async def test_list_col_in_excluded_features_still_dropped(self) -> None:
        """List-typed col declared in ``excluded_features`` is STILL
        dropped from all returned frames — the safety scan ignores
        ``exclude_columns`` to prevent bypass."""
        train_df = pd.DataFrame(
            {
                "comorbidities": [["E11", "I10"], [], ["J45"], []],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_excluded_list_safety",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                # Caller explicitly excludes the list col from transformation.
                # The safety drop MUST still fire — otherwise the col
                # would survive into X_train and crash model_trainer.
                "excluded_features": ["comorbidities"],
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        # X_train must NOT contain the list col.
        X_train = result["X_train"]
        assert "comorbidities" not in X_train.columns, (
            "List col in excluded_features bypassed the safety drop — "
            "Codex pass-4 MED-4 regression. The transformation-exclusion "
            "path only suppresses encoding/scaling, NOT column removal; "
            "the safety scan must run independently."
        )
        # State's train_df also clean (Codex pass-2 MED-2 + pass-4 MED-4).
        assert "train_df" in result
        assert "comorbidities" not in result["train_df"].columns
        # Drop metadata surfaces it.
        transformations = result.get("transformations_applied") or []
        drop_entries = [t for t in transformations if t.get("type") == "drop_unhashable_columns"]
        assert len(drop_entries) == 1
        assert "comorbidities" in set(drop_entries[0].get("columns") or [])

    async def test_list_col_in_legacy_exclude_columns_still_dropped(self) -> None:
        """Same regression but via the legacy ``exclude_columns`` key
        (deprecated but honored). Safety drop must still fire."""
        train_df = pd.DataFrame(
            {
                "secondary_diagnosis_codes": [["B97.4"], [], ["J45.0"], []],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_legacy_excluded_list",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                # Legacy key — still honored, still must not bypass safety.
                "exclude_columns": ["secondary_diagnosis_codes"],
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = await transform_data(state)
        assert result.get("error") is None
        X_train = result["X_train"]
        assert "secondary_diagnosis_codes" not in X_train.columns
        assert "train_df" in result
        assert "secondary_diagnosis_codes" not in result["train_df"].columns

    async def test_non_list_col_in_excluded_features_still_kept_in_frame(
        self,
    ) -> None:
        """Sanity: when an excluded col is NOT list-typed, the safety
        drop does NOT fire — and the col remains in X_train (per
        existing excluded_features semantics: suppress from
        encoding/scaling, keep in frame).
        """
        train_df = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004"],
                "age": [25.0, 60.0, 45.0, 30.0],
                "target": [0, 1, 0, 1],
            }
        )
        state = {
            "experiment_id": "test_issue_197_excluded_scalar",
            "train_df": train_df,
            "validation_df": train_df.copy(),
            "test_df": train_df.copy(),
            "scope_spec": {
                "target_column": "target",
                "excluded_features": ["patient_id"],
                "scaling_method": "minmax",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
            },
        }
        result = await transform_data(state)
        assert result.get("error") is None
        X_train = result["X_train"]
        # Scalar excluded col stays — safety drop doesn't fire.
        assert "patient_id" in X_train.columns
        # No drop_unhashable_columns metadata entry.
        transformations = result.get("transformations_applied") or []
        drop_entries = [t for t in transformations if t.get("type") == "drop_unhashable_columns"]
        assert drop_entries == []


class TestPredictionTargetKeySeparation:
    """The target must be separated via the canonical ``prediction_target`` key.

    Regression for the data_transformer target-key bug: this node read
    ``scope_spec['target_column']`` — a key the harness, scope_builder,
    baseline_computer and sufficiency_check never set (they use the canonical
    ``prediction_target``). When only ``prediction_target`` was present, the
    target was NOT separated, so the binary target column was swept through
    StandardScaler and mean-centred to ~0; baseline_computer then read
    ``target_rate`` ≈ 0 and sufficiency_check fired a false 'zero positive
    cases' HARD_FAIL on every real ``--data-dir`` run.

    The fix reads ``prediction_target`` (canonical) with a fallback to the
    legacy ``target_column`` for backward compatibility.
    """

    def _binary_train(self) -> pd.DataFrame:
        # 8 rows, 2 positives — a rare-ish binary target plus one feature.
        return pd.DataFrame(
            {
                "feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "y": [0, 0, 0, 0, 0, 0, 1, 1],
            }
        )

    @pytest.mark.asyncio
    async def test_canonical_prediction_target_is_separated_and_not_scaled(self):
        """With only ``prediction_target`` set, the target is split out (y_train
        populated, absent from X_train) and its raw values are preserved."""
        state = {
            "experiment_id": "exp_prediction_target_key",
            "scope_spec": {
                # canonical key ONLY — NO legacy 'target_column'
                "prediction_target": "y",
                "scaling_method": "standard",
                "encoding_method": "label",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
            },
            "train_df": self._binary_train(),
        }

        result = await transform_data(state)

        assert result.get("error") is None
        # Target separated into y_train (not left None).
        assert result["y_train"] is not None, "target not separated; y_train is None"
        # Target NOT swept into the scaled feature matrix.
        assert "y" not in result["X_train"].columns, "target leaked into X_train"
        # Raw target preserved (NOT StandardScaler'd to mean ~0): still {0, 1}.
        y_train = result["y_train"]
        assert sorted(int(v) for v in pd.unique(y_train)) == [0, 1]
        assert int(y_train.sum()) == 2

    @pytest.mark.asyncio
    async def test_legacy_target_column_key_still_honored(self):
        """Backward compatibility: a scope that sets only the legacy
        ``target_column`` (no ``prediction_target``) must still separate."""
        state = {
            "experiment_id": "exp_legacy_target_column_key",
            "scope_spec": {
                "target_column": "y",  # legacy key only
                "scaling_method": "standard",
                "encoding_method": "label",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
            },
            "train_df": self._binary_train(),
        }

        result = await transform_data(state)

        assert result.get("error") is None
        assert result["y_train"] is not None
        assert "y" not in result["X_train"].columns
        assert int(result["y_train"].sum()) == 2

    @pytest.mark.asyncio
    async def test_canonical_key_wins_when_both_keys_set(self):
        """When both keys are present with different values, the canonical
        ``prediction_target`` wins (pins the ``or`` precedence so a refactor
        can't silently flip it)."""
        df = self._binary_train()
        df["legacy_y"] = df["y"]  # same values, different name
        state = {
            "experiment_id": "exp_both_keys",
            "scope_spec": {
                "prediction_target": "y",  # canonical — should win
                "target_column": "legacy_y",  # legacy — should lose
                "scaling_method": "standard",
                "encoding_method": "label",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
            },
            "train_df": df,
        }

        result = await transform_data(state)

        assert result.get("error") is None
        # The canonical target 'y' is the one separated out.
        assert "y" not in result["X_train"].columns
        # The legacy-named column was NOT treated as the target.
        assert "legacy_y" in result["X_train"].columns
        assert int(result["y_train"].sum()) == 2


class TestNominalCategoricalEncodingDefault:
    """Issue #790: nominal categoricals must reach the model ONE-HOT encoded by
    default, not integer/ordinal LabelEncoded.

    LabelEncoder imposes a false magnitude order on nominal categories
    (``HMO=0, PPO=1, EPO=2``). Once integer-coded they look numeric, so the
    downstream ``ModelTrainerPreprocessor`` (which is designed to one-hot
    object-dtype columns) skips them and the LINEAR champion trains on ordinal
    codes — degrading discrimination (faithful HCP-adoption run: AUC
    0.777 ordinal -> 0.803 one-hot, on merit, no gate gamed). The fix flips the
    ``encoding_method`` default ``"label"`` -> ``"onehot"`` while honoring an
    explicit ``"label"`` override and a new ``ordinal_features`` allow-list for
    genuinely-ordered categoricals (e.g. risk bands).
    """

    def _state(self, scope_extra: dict) -> dict:
        # ``payer`` and ``region`` are NOMINAL; ``risk_band`` is genuinely ordinal.
        train_df = pd.DataFrame(
            {
                "payer": ["HMO", "PPO", "EPO", "HMO", "PPO", "EPO"],
                "region": ["NE", "S", "MW", "W", "NE", "S"],
                "risk_band": ["low", "med", "high", "low", "med", "high"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        scope_spec = {
            "prediction_target": "target",
            "scaling_method": "minmax",
            "imputation_strategy": "mean",
            "extract_datetime_features": False,
        }
        scope_spec.update(scope_extra)
        return {
            "experiment_id": "exp_issue790_onehot_default",
            "scope_spec": scope_spec,
            "train_df": train_df,
        }

    async def test_default_one_hots_nominal_categoricals(self):
        """With NO ``encoding_method`` set, nominal categoricals one-hot expand:
        the single ``payer`` column is gone; ``payer_HMO/_PPO/_EPO`` appear."""
        result = await transform_data(self._state({}))

        assert result.get("error") is None
        cols = list(result["X_train"].columns)
        # The single ordinal-coded column must NOT survive.
        assert "payer" not in cols, f"nominal 'payer' left as single (ordinal) column: {cols}"
        # One-hot expansion present (OneHotEncoder names: <col>_<value>).
        expanded = [c for c in cols if c.startswith("payer_")]
        assert len(expanded) >= 3, f"expected one-hot payer_* columns, got {cols}"
        # The encoding metadata records one-hot for the nominal columns.
        enc_steps = [t for t in result["transformations_applied"] if t["type"] == "encoding"]
        assert any(s["method"] == "onehot" for s in enc_steps), enc_steps

    async def test_explicit_label_still_integer_encodes(self):
        """Back-compat: an explicit ``encoding_method='label'`` keeps the legacy
        single-column integer encoding (no one-hot expansion)."""
        result = await transform_data(self._state({"encoding_method": "label"}))

        assert result.get("error") is None
        cols = list(result["X_train"].columns)
        assert "payer" in cols, f"explicit label must keep single 'payer' column: {cols}"
        assert not [c for c in cols if c.startswith("payer_")], cols
        enc_steps = [t for t in result["transformations_applied"] if t["type"] == "encoding"]
        assert all(s["method"] != "onehot" for s in enc_steps), enc_steps

    async def test_ordinal_features_stay_integer_under_onehot_default(self):
        """``ordinal_features`` are integer-encoded (order preserved) even under
        the one-hot default; other nominal categoricals still one-hot expand."""
        result = await transform_data(self._state({"ordinal_features": ["risk_band"]}))

        assert result.get("error") is None
        cols = list(result["X_train"].columns)
        # The declared-ordinal column stays a SINGLE column (no one-hot).
        assert "risk_band" in cols, f"ordinal 'risk_band' should stay single column: {cols}"
        assert not [c for c in cols if c.startswith("risk_band_")], cols
        # Nominal columns still expand.
        assert "payer" not in cols
        assert [c for c in cols if c.startswith("payer_")], cols
        assert [c for c in cols if c.startswith("region_")], cols

    async def test_onehot_default_leaves_no_object_columns(self):
        """The default-encoded feature matrix is fully numeric (no object dtype
        survives), so ``ModelTrainerPreprocessor`` sees an already-encoded frame
        instead of re-encoding integer codes."""
        result = await transform_data(self._state({}))

        assert result.get("error") is None
        obj_cols = result["X_train"].select_dtypes(include=["object"]).columns.tolist()
        assert obj_cols == [], f"object columns survived default encoding: {obj_cols}"

    async def test_onehot_default_output_is_preprocessor_passthrough(self):
        """Production-config guard: the one-hot + standard-scaled default output
        is recognized as already-preprocessed by ``ModelTrainerPreprocessor``
        (``preprocessing_type == "passthrough"``), so model_trainer does NOT run
        a redundant second preprocessing pass. Pins the contract under the
        production default (scaling_method defaults to "standard"); guards the
        codex #790 double-processing concern. Faithful (no mocking): drives the
        real ``transform_data`` -> real ``fit_preprocessing``."""
        from src.agents.ml_foundation.model_trainer.nodes.preprocessor import fit_preprocessing

        # Enough rows / category variety that StandardScaler yields std~1 on the
        # one-hot columns so the passthrough heuristic fires deterministically.
        n = 60
        payers = ["HMO", "PPO", "EPO"] * (n // 3)
        regions = (["NE", "S", "MW", "W"] * (n // 4 + 1))[:n]
        frame = pd.DataFrame(
            {
                "payer": payers,
                "region": regions,
                "value": [float(i % 7) for i in range(n)],
                "target": [i % 2 for i in range(n)],
            }
        )
        # scaling_method intentionally UNSET -> production default "standard".
        state = {
            "experiment_id": "exp_issue790_passthrough",
            "scope_spec": {
                "prediction_target": "target",
                "imputation_strategy": "mean",
                "extract_datetime_features": False,
            },
            "train_df": frame.copy(),
            "validation_df": frame.copy(),
            "test_df": frame.copy(),
        }
        out = await transform_data(state)
        assert out.get("error") is None, out.get("error")

        pp = await fit_preprocessing(
            {
                "train_data": {"X": out["X_train"], "y": out["y_train"]},
                "validation_data": {"X": out["X_val"], "y": out["y_val"]},
                "test_data": {"X": out["X_test"], "y": out["y_test"]},
            }
        )
        assert pp.get("error") is None, pp.get("error")
        assert pp["preprocessing_statistics"]["preprocessing_type"] == "passthrough", (
            "one-hot + standard-scaled output should passthrough, not re-preprocess; "
            f"got {pp['preprocessing_statistics']['preprocessing_type']}"
        )
