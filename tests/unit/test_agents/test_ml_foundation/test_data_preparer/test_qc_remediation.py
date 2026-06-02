"""Unit tests for the qc_remediation node's action parser.

Backlog #13 sub-gate 3 (initial fix): ``_parse_remediation_action``
used to assign ``params`` as a raw string, which crashed
``_apply_automatic_remediation`` at ``params.get("strategy")`` with
``AttributeError: 'str' object has no attribute 'get'``.

Codex review on PR #106 surfaced two follow-ons (HIGH-C + MEDIUM-C):
multi-key JSON params got truncated by comma-splitting, and malformed
params silently fell back to default behavior (fail-open). Both
addressed via the params-is-last parser + ``params_parse_failed``
flag that the apply loop honors by skipping (fail-loud).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.qc_remediation import (
    _apply_automatic_remediation,
    _coerce_params_to_dict,
    _parse_remediation_action,
)


class TestParseRemediationAction:
    """``_parse_remediation_action`` returns a structured dict."""

    def test_params_empty_object(self) -> None:
        action = _parse_remediation_action("action: drop_column, column: x, params: {}")
        assert action["type"] == "drop_column"
        assert action["column"] == "x"
        assert action["params"] == {}
        assert action["params_parse_failed"] is False

    def test_params_with_strategy(self) -> None:
        action = _parse_remediation_action(
            'action: impute, column: foo, params: {"strategy": "mean"}'
        )
        assert action["params"] == {"strategy": "mean"}
        assert action["params_parse_failed"] is False

    def test_params_multikey_json_not_truncated_by_comma(self) -> None:
        """Codex HIGH-C regression: comma-split would truncate JSON.

        Pre-fix: ``params: {"strategy": "median", "target": "y"}`` got
        split at the inner comma, leaving params unparseable. The
        params-is-last parser treats everything after ``params:`` as
        the value, preserving multi-key JSON.
        """
        action = _parse_remediation_action(
            'action: impute, column: foo, params: {"strategy": "median", "target": "y"}'
        )
        assert action["type"] == "impute"
        assert action["column"] == "foo"
        assert action["params"] == {"strategy": "median", "target": "y"}
        assert action["params_parse_failed"] is False

    def test_params_malformed_marks_parse_failed(self) -> None:
        """Codex MEDIUM-C: malformed params must NOT silently fall back.

        Free-form text the LLM might emit (e.g., ``strategy=mean``) is
        not valid JSON; the parser flags it via ``params_parse_failed``
        so the apply loop skips the action instead of applying defaults.
        """
        action = _parse_remediation_action("action: impute, column: foo, params: strategy=mean")
        assert action["params"] == {}
        assert action["params_parse_failed"] is True
        assert "strategy=mean" in action.get("params_raw", "")

    def test_params_omitted_uses_empty_dict_default(self) -> None:
        action = _parse_remediation_action("action: drop_column, column: x")
        assert action["params"] == {}
        assert action["params_parse_failed"] is False

    def test_params_non_dict_json_marks_parse_failed(self) -> None:
        # JSON but not a dict — a list or scalar.
        action_list = _parse_remediation_action("action: impute, column: foo, params: [1, 2, 3]")
        assert action_list["params"] == {}
        assert action_list["params_parse_failed"] is True

    def test_params_explicit_empty_does_not_set_failed_flag(self) -> None:
        # ``params:`` with empty value (after the colon) is the LLM's
        # explicit "no params" signal — not a parse failure.
        action = _parse_remediation_action("action: drop_column, column: x, params: ")
        assert action["params"] == {}
        assert action["params_parse_failed"] is False


class TestCoerceParamsToDict:
    """``_coerce_params_to_dict`` returns Optional[Dict] post-codex.

    None signals "non-empty but unparseable"; ``{}`` signals "explicitly
    empty"; dict signals successful decode.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("{}", {}),
            ('{"strategy": "median"}', {"strategy": "median"}),
            ('{"k": 1, "v": "x"}', {"k": 1, "v": "x"}),
            ("", {}),
            ('  {"strategy": "median"}  ', {"strategy": "median"}),  # whitespace
        ],
    )
    def test_valid_decode(self, raw: str, expected: dict) -> None:
        assert _coerce_params_to_dict(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "not_json_at_all",
            "[1, 2]",
            "42",
            '"just_a_string"',
            "null",
            "{strategy: median}",  # missing quotes
            'strategy="median"',
            "{",  # truncated
        ],
    )
    def test_unparseable_returns_none(self, raw: str) -> None:
        assert _coerce_params_to_dict(raw) is None


class TestApplyAutomaticRemediationDefensiveCoerce:
    """The apply loop honors ``params_parse_failed`` and tolerates raw strings."""

    @pytest.mark.asyncio
    async def test_string_params_does_not_crash(self) -> None:
        # Caller bypasses the parser and passes a raw string —
        # _apply_automatic_remediation must NOT crash with AttributeError.
        train_df = pd.DataFrame({"x": [1, 2, None, 4], "y": [10, 20, 30, 40]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [
            {
                "type": "impute",
                "column": "x",
                "params": "strategy: median",  # ← non-dict; would crash pre-fix
            }
        ]
        result = await _apply_automatic_remediation(state, actions)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_dict_params_still_works(self) -> None:
        train_df = pd.DataFrame({"x": [1, 2, None, 4]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [
            {
                "type": "impute",
                "column": "x",
                "params": {"strategy": "mean"},
            }
        ]
        result = await _apply_automatic_remediation(state, actions)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_params_parse_failed_action_is_skipped(self) -> None:
        """Codex MEDIUM-C: malformed params must skip, not silently default.

        Applying default impute strategy when the LLM specified a
        different (unparseable) strategy could be destructive — e.g.,
        median imputation when the LLM intended forward-fill on a
        time-series column.
        """
        train_df = pd.DataFrame({"x": [1, 2, None, 4]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [
            {
                "type": "impute",
                "column": "x",
                "params": {},
                "params_parse_failed": True,
                "params_raw": "strategy=garbage",
            }
        ]
        result = await _apply_automatic_remediation(state, actions)
        assert result["success"] is True
        # The action was skipped — no imputation applied.
        actions_taken = result.get("actions_taken", [])
        assert any("SKIPPED" in a and "malformed params" in a for a in actions_taken)

    @pytest.mark.asyncio
    async def test_mixed_actions_skip_only_malformed(self) -> None:
        """Multiple actions: malformed ones skipped, valid ones applied."""
        train_df = pd.DataFrame({"x": [1, 2, None, 4], "y": [10, None, 30, 40]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [
            {
                "type": "impute",
                "column": "x",
                "params": {},
                "params_parse_failed": True,
                "params_raw": "garbage",
            },
            {
                "type": "impute",
                "column": "y",
                "params": {"strategy": "mean"},
            },
        ]
        result = await _apply_automatic_remediation(state, actions)
        assert result["success"] is True
        actions_taken = result.get("actions_taken", [])
        # First action skipped.
        assert any("SKIPPED" in a for a in actions_taken)
        # Second action applied (we don't assert exact message, just that
        # something non-skipped happened).
        assert any("SKIPPED" not in a for a in actions_taken)


class TestApplyAutomaticRemediationAllNull:
    """#630: an all-null column must be skipped-and-reported, not imputed.

    PR #629 made ``_impute_column`` dtype-safe (all-null numeric → ``0``,
    all-null object → ``"UNKNOWN"``) so ``transform_data`` no longer crashes.
    But filling a column that has *no data* is semantically misleading: the
    placeholder can pass the QC gate on retry and reach model training as a
    constant feature, masking that a required feature is entirely absent from
    the cohort. ``_apply_automatic_remediation`` is the seam that catches both
    LLM-emitted and rule-based ``impute`` actions, so the skip guard lives
    there — leaving the column untouched (not dropped) so the completeness
    dimension keeps blocking and forces investigation.
    """

    @pytest.mark.asyncio
    async def test_all_null_numeric_column_skipped_not_zero_filled(self) -> None:
        train_df = pd.DataFrame({"x": pd.Series([None, None, None], dtype="float64")})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [{"type": "impute", "column": "x", "params": {"strategy": "mode"}}]

        result = await _apply_automatic_remediation(state, actions)

        assert result["success"] is True
        out = result["train_df"]
        # The all-null column is left untouched — NOT 0-filled.
        assert out["x"].isnull().all(), f"all-null column was imputed: {out['x'].tolist()}"
        assert pd.api.types.is_numeric_dtype(out["x"])
        actions_taken = result.get("actions_taken", [])
        assert any("SKIPPED" in a and "all-null" in a for a in actions_taken), actions_taken

    @pytest.mark.asyncio
    async def test_all_null_object_column_skipped_not_unknown_filled(self) -> None:
        train_df = pd.DataFrame({"c": pd.Series([None, None, None], dtype="object")})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [{"type": "impute", "column": "c", "params": {"strategy": "mode"}}]

        result = await _apply_automatic_remediation(state, actions)

        assert result["success"] is True
        out = result["train_df"]
        # The all-null object column is left untouched — NOT "UNKNOWN"-filled.
        assert out["c"].isnull().all(), f"all-null column was imputed: {out['c'].tolist()}"
        assert "UNKNOWN" not in out["c"].astype(str).tolist()
        actions_taken = result.get("actions_taken", [])
        assert any("SKIPPED" in a and "all-null" in a for a in actions_taken), actions_taken

    @pytest.mark.asyncio
    async def test_partial_null_column_still_imputed(self) -> None:
        """Regression guard: a column with SOME data is still imputed."""
        train_df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [{"type": "impute", "column": "x", "params": {"strategy": "median"}}]

        result = await _apply_automatic_remediation(state, actions)

        assert result["success"] is True
        out = result["train_df"]
        assert out["x"].isnull().sum() == 0, "partial-null column should still be imputed"
        actions_taken = result.get("actions_taken", [])
        assert any("SKIPPED" not in a for a in actions_taken)
        assert not any("SKIPPED" in a and "all-null" in a for a in actions_taken)

    @pytest.mark.asyncio
    async def test_all_null_column_also_untouched_in_validation_and_test(self) -> None:
        train_df = pd.DataFrame({"x": pd.Series([None, None], dtype="float64")})
        validation_df = pd.DataFrame({"x": pd.Series([None, None], dtype="float64")})
        test_df = pd.DataFrame({"x": pd.Series([None, None], dtype="float64")})
        state = {"train_df": train_df, "validation_df": validation_df, "test_df": test_df}
        actions = [{"type": "impute", "column": "x", "params": {"strategy": "mode"}}]

        result = await _apply_automatic_remediation(state, actions)

        assert result["success"] is True
        assert result["train_df"]["x"].isnull().all()
        # Splits are not silently 0-filled either (the skip short-circuits
        # before any per-split imputation runs).
        assert result["validation_df"]["x"].isnull().all()
        assert result["test_df"]["x"].isnull().all()

    @pytest.mark.asyncio
    async def test_empty_dataframe_is_not_falsely_skipped(self) -> None:
        """Guard regression: ``isnull().all()`` is vacuously True on an empty
        Series, so the ``len(train_df) > 0`` guard must prevent a 0-row column
        from being treated as 'all-null' and skipped. An empty-df impute falls
        through to ``_impute_column`` as a harmless no-op — NOT a SKIPPED."""
        train_df = pd.DataFrame({"x": pd.Series([], dtype="float64")})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [{"type": "impute", "column": "x", "params": {"strategy": "median"}}]

        result = await _apply_automatic_remediation(state, actions)

        assert result["success"] is True
        actions_taken = result.get("actions_taken", [])
        assert not any("SKIPPED" in a and "all-null" in a for a in actions_taken), actions_taken
