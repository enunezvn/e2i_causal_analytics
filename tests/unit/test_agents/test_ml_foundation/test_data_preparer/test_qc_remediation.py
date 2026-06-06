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

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.qc_remediation import (
    _apply_automatic_remediation,
    _coerce_params_to_dict,
    _parse_remediation_action,
    review_and_remediate_qc,
)

# Module-qualified path of the LLM-analysis seam. Patching it lets the
# auto-remediation branch of ``review_and_remediate_qc`` run
# deterministically without a real Anthropic call — we inject the
# analysis dict the node would otherwise build from an LLM response.
_ANALYZE_LLM = (
    "src.agents.ml_foundation.data_preparer.nodes.qc_remediation._analyze_qc_failures_with_llm"
)


def _failing_qc_state(train_df: pd.DataFrame, **overrides: object) -> dict:
    """Build a minimal state that routes ``review_and_remediate_qc`` into
    the auto-remediation branch.

    The node bails early when ``gate_passed`` is True / status "passed",
    or when ``remediation_attempts`` has hit the max; this state avoids
    both so control reaches ``_apply_automatic_remediation``.
    """
    state: dict = {
        "experiment_id": "exp-632",
        "qc_status": "failed",
        "gate_passed": False,
        "overall_score": 0.5,
        "remediation_attempts": 0,
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
    }
    state.update(overrides)
    return state


def _auto_remediation_analysis(actions: list[dict]) -> dict:
    """Analysis payload that triggers the auto-remediation branch."""
    return {
        "can_auto_remediate": True,
        "remediation_actions": actions,
        "root_cause_summary": "test-injected analysis",
    }


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


class TestReviewAndRemediateQcStatePropagation:
    """#632: the success-path return of ``review_and_remediate_qc`` must
    carry the remediated ``train_df``/``validation_df``/``test_df`` so the
    LangGraph retry edge (``qc_remediation`` -> ``run_quality_checks``,
    which reads ``state["train_df"]``) re-validates the REMEDIATED frame.

    Pre-fix, the success-path dict omitted these keys. ``drop_column`` and
    ``deduplicate`` rebind ``train_df`` to a NEW ``.drop()``/
    ``.drop_duplicates()`` copy inside ``_apply_automatic_remediation``,
    so the state's original object is unchanged and the remediation is
    inert on retry. ``impute`` only "worked" by accident — ``_impute_column``
    mutates the shared object in place — which is fragile (a future
    copy-returning refactor would silently break it too).

    These tests drive the FULL node (LLM seam patched) and assert the
    returned state dict reflects the remediation, exercising the actual
    state hand-off to LangGraph rather than the inner helper alone.
    """

    @pytest.mark.asyncio
    async def test_drop_column_propagates_to_returned_state(self) -> None:
        """RED pre-fix: returned state omits ``train_df`` -> dropped
        column is absent from the result dict, so the retry pass sees the
        original frame with the column still present."""
        train_df = pd.DataFrame({"keep": [1, 2, 3], "drop_me": [4, 5, 6]})
        state = _failing_qc_state(train_df)
        analysis = _auto_remediation_analysis(
            [{"type": "drop_column", "column": "drop_me", "params": {}}]
        )

        with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
            result = await review_and_remediate_qc(state)

        assert result["remediation_status"] == "applied"
        assert result["requires_revalidation"] is True
        # The remediated frame must be in the returned state dict.
        assert "train_df" in result, "success-path state omits train_df (#632)"
        out = result["train_df"]
        assert "drop_me" not in out.columns, "dropped column still present in returned state"
        assert "keep" in out.columns

    @pytest.mark.asyncio
    async def test_deduplicate_propagates_to_returned_state(self) -> None:
        """RED pre-fix: row reduction from ``drop_duplicates()`` (a rebind)
        never reaches the returned state dict."""
        train_df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [9, 9, 8, 8]})
        state = _failing_qc_state(train_df)
        analysis = _auto_remediation_analysis(
            [{"type": "deduplicate", "column": None, "params": {}}]
        )

        with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
            result = await review_and_remediate_qc(state)

        assert result["remediation_status"] == "applied"
        assert "train_df" in result, "success-path state omits train_df (#632)"
        out = result["train_df"]
        assert len(out) == 2, f"deduplicate did not propagate; got {len(out)} rows"

    @pytest.mark.asyncio
    async def test_impute_propagates_via_explicit_return_not_just_inplace(self) -> None:
        """Regression: ``impute`` must propagate through the EXPLICIT
        returned state, not only by accidental in-place mutation. We assert
        on the frame in the RESULT dict (the object LangGraph receives),
        which after the fix is the remediated frame."""
        train_df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
        state = _failing_qc_state(train_df)
        analysis = _auto_remediation_analysis(
            [{"type": "impute", "column": "x", "params": {"strategy": "median"}}]
        )

        with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
            result = await review_and_remediate_qc(state)

        assert result["remediation_status"] == "applied"
        assert "train_df" in result, "success-path state omits train_df (#632)"
        out = result["train_df"]
        assert out["x"].isnull().sum() == 0, (
            "partial-null column should be imputed in returned state"
        )

    @pytest.mark.asyncio
    async def test_all_split_frames_propagate_to_returned_state(self) -> None:
        """drop_column across train/validation/test must propagate every
        non-None split into the returned state dict."""
        train_df = pd.DataFrame({"keep": [1, 2], "drop_me": [3, 4]})
        validation_df = pd.DataFrame({"keep": [5], "drop_me": [6]})
        test_df = pd.DataFrame({"keep": [7], "drop_me": [8]})
        state = _failing_qc_state(train_df, validation_df=validation_df, test_df=test_df)
        analysis = _auto_remediation_analysis(
            [{"type": "drop_column", "column": "drop_me", "params": {}}]
        )

        with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
            result = await review_and_remediate_qc(state)

        assert "train_df" in result and "validation_df" in result and "test_df" in result
        assert "drop_me" not in result["train_df"].columns
        assert "drop_me" not in result["validation_df"].columns
        assert "drop_me" not in result["test_df"].columns

    @pytest.mark.asyncio
    async def test_failed_remediation_does_not_add_df_keys(self) -> None:
        """Reason-before-rules guard: only the SUCCESS path forwards the
        dfs. A failed remediation returns its existing ``failed`` dict
        unchanged — we must not blanket-inject df keys onto every path."""
        train_df = pd.DataFrame({"x": [1, 2, 3]})
        state = _failing_qc_state(train_df)
        analysis = _auto_remediation_analysis(
            [{"type": "drop_column", "column": "drop_me", "params": {}}]
        )

        # Force _apply_automatic_remediation to report failure.
        async def _fail(_state: object, _actions: object) -> dict:
            return {"success": False, "error": "boom", "actions_taken": []}

        with (
            patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)),
            patch(
                "src.agents.ml_foundation.data_preparer.nodes.qc_remediation."
                "_apply_automatic_remediation",
                new=AsyncMock(side_effect=_fail),
            ),
        ):
            result = await review_and_remediate_qc(state)

        assert result["remediation_status"] == "failed"
        assert "train_df" not in result


class TestNoProgressStop:
    """A remediation round that applies ZERO effective actions (every action
    skipped — e.g. malformed LLM params) must NOT request revalidation.

    Pre-fix the node returned ``status="applied"`` + ``requires_revalidation=True``
    even when nothing changed, so the QC loop re-flagged the identical column,
    re-emitted the same un-appliable drop, and spun until LangGraph's
    recursion_limit (the discontinuation_mart GraphRecursionError, 2026-06-06:
    ``cci_severe_liver`` perfect-separation drop perpetually skipped).
    """

    @pytest.mark.asyncio
    async def test_all_skipped_round_reports_zero_effective(self) -> None:
        # impute READS params (strategy), so a malformed params field is still
        # skipped -> 0 effective. (drop_column no longer skips on malformed
        # params — it is param-less; see TestParamlessActions...)
        train_df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [0, 1, 0]})
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
        assert result["effective_action_count"] == 0

    @pytest.mark.asyncio
    async def test_real_drop_reports_one_effective(self) -> None:
        train_df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [{"type": "drop_column", "column": "x", "params": {}}]
        result = await _apply_automatic_remediation(state, actions)
        assert result["effective_action_count"] == 1

    @pytest.mark.asyncio
    async def test_mixed_round_counts_only_effective(self) -> None:
        train_df = pd.DataFrame({"x": [1, 2, 3], "keep": [4.0, None, 6.0]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [
            {"type": "drop_column", "column": "x", "params": {}},  # effective
            {
                "type": "impute",  # params-reading -> skipped on malformed
                "column": "keep",
                "params": {},
                "params_parse_failed": True,
                "params_raw": "garbage",
            },  # skipped
        ]
        result = await _apply_automatic_remediation(state, actions)
        assert result["effective_action_count"] == 1

    @pytest.mark.asyncio
    async def test_dedup_removing_zero_rows_is_not_effective(self) -> None:
        train_df = pd.DataFrame({"a": [1, 2, 3]})  # already unique
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [{"type": "deduplicate", "column": None, "params": {}}]
        result = await _apply_automatic_remediation(state, actions)
        assert result["effective_action_count"] == 0

    @pytest.mark.asyncio
    async def test_node_no_progress_does_not_request_revalidation(self) -> None:
        """The whole point: an all-skipped round must stop the loop."""
        train_df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [0, 1, 0]})
        state = _failing_qc_state(train_df)
        analysis = _auto_remediation_analysis(
            [
                {
                    "type": "impute",  # params-reading -> skipped on malformed
                    "column": "x",
                    "params": {},
                    "params_parse_failed": True,
                    "params_raw": "strategy=garbage",
                }
            ]
        )
        with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
            result = await review_and_remediate_qc(state)

        assert result["remediation_status"] == "manual_required"
        assert not result.get("requires_revalidation")
        # attempt counter still advances + the skipped actions are surfaced
        assert result["remediation_attempts"] == 1

    @pytest.mark.asyncio
    async def test_node_real_progress_still_requests_revalidation(self) -> None:
        """Regression guard: a round that DOES change data must still retry."""
        train_df = pd.DataFrame({"keep": [1, 2, 3], "drop_me": [4, 5, 6]})
        state = _failing_qc_state(train_df)
        analysis = _auto_remediation_analysis(
            [{"type": "drop_column", "column": "drop_me", "params": {}}]
        )
        with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
            result = await review_and_remediate_qc(state)

        assert result["remediation_status"] == "applied"
        assert result["requires_revalidation"] is True


class TestParamlessActionsAppliedDespiteMalformedParams:
    """drop_column / deduplicate are fully determined by the parsed column name /
    row identity — their behavior does NOT read ``params``. So a malformed
    ``params:`` field cannot change what they do, and the Codex MEDIUM-C skip
    (which exists to stop a destructive default-strategy ``impute``) must NOT
    block them. Pre-fix the skip applied to ALL action types, so a
    perfect-separation drop with a malformed ``reason="..."`` params string was
    perpetually skipped and the cohort stayed unmodelable (discontinuation_mart,
    2026-06-06)."""

    @pytest.mark.asyncio
    async def test_drop_column_applied_despite_malformed_params(self) -> None:
        train_df = pd.DataFrame({"cci_severe_liver": [0, 0, 0], "keep": [1, 2, 3]})
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [
            {
                "type": "drop_column",
                "column": "cci_severe_liver",
                "params": {},
                "params_parse_failed": True,
                "params_raw": 'reason="perfect_class_separation_data_leakage"',
            }
        ]
        result = await _apply_automatic_remediation(state, actions)
        assert result["success"] is True
        assert result["effective_action_count"] == 1
        assert "cci_severe_liver" not in result["train_df"].columns
        actions_taken = result.get("actions_taken", [])
        assert any("Dropped column: cci_severe_liver" in a for a in actions_taken)
        assert not any("SKIPPED" in a for a in actions_taken)

    @pytest.mark.asyncio
    async def test_deduplicate_applied_despite_malformed_params(self) -> None:
        train_df = pd.DataFrame({"a": [1, 1, 2], "b": [9, 9, 8]})  # one dup pair
        state = {"train_df": train_df, "validation_df": None, "test_df": None}
        actions = [
            {
                "type": "deduplicate",
                "column": None,
                "params": {},
                "params_parse_failed": True,
                "params_raw": "garbage",
            }
        ]
        result = await _apply_automatic_remediation(state, actions)
        assert result["success"] is True
        assert result["effective_action_count"] == 1
        assert len(result["train_df"]) == 2

    @pytest.mark.asyncio
    async def test_impute_still_skipped_on_malformed_params(self) -> None:
        """Regression guard: impute READS params (strategy), so a malformed
        params field MUST still skip — defaulting the strategy could be
        destructive (the original Codex MEDIUM-C intent)."""
        train_df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
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
        assert result["effective_action_count"] == 0
        actions_taken = result.get("actions_taken", [])
        assert any("SKIPPED" in a and "malformed params" in a for a in actions_taken)


class TestRouteAfterRemediationNoProgress:
    """``_route_after_remediation`` stops on a no-progress round, retries on real
    progress (within the attempt cap)."""

    def test_router_stops_when_no_revalidation_requested(self) -> None:
        from src.agents.ml_foundation.data_preparer.graph import (
            _route_after_remediation,
        )

        state = {
            "remediation_status": "manual_required",
            "requires_revalidation": False,
            "remediation_attempts": 1,
        }
        assert _route_after_remediation(state) == "end"

    def test_router_retries_on_real_progress(self) -> None:
        from src.agents.ml_foundation.data_preparer.graph import (
            _route_after_remediation,
        )

        state = {
            "remediation_status": "applied",
            "requires_revalidation": True,
            "remediation_attempts": 1,
        }
        assert _route_after_remediation(state) == "retry"
