"""Tests for ``ScenarioName`` enum + ``SCENARIO_REGISTRY`` (shard 01 §B.1)."""

from __future__ import annotations

import json

import pytest

from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName


class TestScenarioNameEnumValues:
    def test_a_value_is_canonical_id(self) -> None:
        assert ScenarioName.A_DIAGNOSTIC_BC_IDFS.value == "scenario_a_diagnostic_ebc_idfs_5y"

    def test_b_value_is_canonical_id(self) -> None:
        assert ScenarioName.B_SCREENING_IGAN_ESKD.value == "scenario_b_screening_igan_eskd_5y"

    def test_c_value_is_canonical_id(self) -> None:
        assert (
            ScenarioName.C_TREATMENT_CSU_RESPONSE.value
            == "scenario_c_treatment_decision_csu_remib_response"
        )

    def test_three_members_total(self) -> None:
        assert {m.name for m in ScenarioName} == {
            "A_DIAGNOSTIC_BC_IDFS",
            "B_SCREENING_IGAN_ESKD",
            "C_TREATMENT_CSU_RESPONSE",
        }


class TestScenarioNameStringInheritance:
    def test_inherits_from_str(self) -> None:
        assert isinstance(ScenarioName.A_DIAGNOSTIC_BC_IDFS, str)

    def test_json_serializable_via_str_inheritance(self) -> None:
        s = json.dumps(ScenarioName.A_DIAGNOSTIC_BC_IDFS)
        assert s == '"scenario_a_diagnostic_ebc_idfs_5y"'

    def test_string_comparison_with_yaml_value(self) -> None:
        yaml_value = "scenario_a_diagnostic_ebc_idfs_5y"
        assert ScenarioName.A_DIAGNOSTIC_BC_IDFS == yaml_value


class TestFromShort:
    @pytest.mark.parametrize(
        "short,expected",
        [
            ("A", ScenarioName.A_DIAGNOSTIC_BC_IDFS),
            ("B", ScenarioName.B_SCREENING_IGAN_ESKD),
            ("C", ScenarioName.C_TREATMENT_CSU_RESPONSE),
        ],
    )
    def test_uppercase_resolves(self, short: str, expected: ScenarioName) -> None:
        assert ScenarioName.from_short(short) == expected

    @pytest.mark.parametrize("short", ["a", "b", "c"])
    def test_lowercase_resolves(self, short: str) -> None:
        # Case-insensitive
        assert ScenarioName.from_short(short) == ScenarioName.from_short(short.upper())

    def test_unknown_short_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scenario short code"):
            ScenarioName.from_short("Z")

    def test_empty_short_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scenario short code"):
            ScenarioName.from_short("")


class TestScenarioRegistry:
    def test_registry_is_dict(self) -> None:
        assert isinstance(SCENARIO_REGISTRY, dict)

    def test_registry_only_contains_registered_scenarios(self) -> None:
        """Commit 07 registers Scenario A; commits 08/09 add B/C."""
        assert ScenarioName.A_DIAGNOSTIC_BC_IDFS in SCENARIO_REGISTRY
        # B and C land in commits 08 and 09 — until then, only A is registered.
