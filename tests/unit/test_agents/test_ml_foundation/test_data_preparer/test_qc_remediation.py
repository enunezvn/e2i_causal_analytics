"""Unit tests for the qc_remediation node's action parser.

Backlog #13 sub-gate 3: ``_parse_remediation_action`` used to assign
``params`` as a raw string, which crashed
``_apply_automatic_remediation`` at ``params.get("strategy")`` with
``AttributeError: 'str' object has no attribute 'get'``. The parser now
JSON-decodes ``params`` (or falls back to ``{}`` on malformed input) and
the apply-loop also defensively coerces non-dict ``params`` to ``{}``.
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
    """``_parse_remediation_action`` returns a dict with ``params: dict``."""

    def test_params_empty_object(self) -> None:
        action = _parse_remediation_action("action: drop_column, column: x, params: {}")
        assert action["type"] == "drop_column"
        assert action["column"] == "x"
        assert isinstance(action["params"], dict)
        assert action["params"] == {}

    def test_params_with_strategy(self) -> None:
        action = _parse_remediation_action(
            'action: impute, column: foo, params: {"strategy": "mean"}'
        )
        assert isinstance(action["params"], dict)
        assert action["params"]["strategy"] == "mean"

    def test_params_malformed_falls_back_to_empty_dict(self) -> None:
        # Free-form text the LLM might emit instead of JSON.
        action = _parse_remediation_action("action: impute, column: foo, params: strategy=mean")
        # Per the parser the comma split treats this oddly; the important
        # invariant is that ``params`` is a dict, not a string.
        assert isinstance(action["params"], dict)

    def test_params_omitted_uses_empty_dict_default(self) -> None:
        action = _parse_remediation_action("action: drop_column, column: x")
        assert isinstance(action["params"], dict)
        assert action["params"] == {}

    def test_params_non_dict_json_falls_back(self) -> None:
        # JSON but not a dict — a list or scalar.
        action_list = _parse_remediation_action("action: impute, column: foo, params: [1, 2, 3]")
        assert isinstance(action_list["params"], dict)
        assert action_list["params"] == {}


class TestCoerceParamsToDict:
    """``_coerce_params_to_dict`` is the lowest-level coercion helper."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("{}", {}),
            ('{"strategy": "median"}', {"strategy": "median"}),
            ('{"k": 1, "v": "x"}', {"k": 1, "v": "x"}),
            ("", {}),
            ("not_json_at_all", {}),
            ("[1, 2]", {}),
            ("42", {}),
            ('"just_a_string"', {}),
            ("null", {}),
        ],
    )
    def test_coercion(self, raw: str, expected: dict) -> None:
        assert _coerce_params_to_dict(raw) == expected


class TestApplyAutomaticRemediationDefensiveCoerce:
    """The apply loop defensively coerces non-dict ``params``."""

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
        # ``x`` was imputed with median (3.0 since values are 1, 2, 4).
        out = result.get("train_df")
        assert out is not None
        assert out["x"].isna().sum() == 0

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
