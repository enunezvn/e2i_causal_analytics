"""Unit tests for src/utils/sufficiency_schemas.py.

Validates pydantic schemas reject invalid input and accept valid input.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.utils.sufficiency_schemas import (
    DataSufficiencyReport,
    SufficiencyConfig,
    ThresholdResolution,
)


class TestSufficiencyConfig:
    def test_empty_config_valid(self):
        cfg = SufficiencyConfig()
        assert cfg.epv_floor is None
        assert cfg.target_mde is None

    def test_valid_overrides(self):
        cfg = SufficiencyConfig(
            epv_floor=10,
            target_mde=0.05,
            power_target=0.90,
            alpha=0.01,
        )
        assert cfg.epv_floor == 10
        assert cfg.target_mde == 0.05

    def test_negative_epv_floor_rejected(self):
        with pytest.raises(ValidationError):
            SufficiencyConfig(epv_floor=-1)

    def test_alpha_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            SufficiencyConfig(alpha=1.5)

    def test_power_zero_rejected(self):
        with pytest.raises(ValidationError):
            SufficiencyConfig(power_target=0.0)

    def test_baseline_rate_at_bounds_rejected(self):
        with pytest.raises(ValidationError):
            SufficiencyConfig(baseline_rate=0.0)
        with pytest.raises(ValidationError):
            SufficiencyConfig(baseline_rate=1.0)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            SufficiencyConfig(bogus_field=42)  # type: ignore[call-arg]

    def test_strictness_preset_valid_values(self):
        for preset in ("conservative", "moderate", "strict"):
            cfg = SufficiencyConfig(strictness_preset=preset)  # type: ignore[arg-type]
            assert cfg.strictness_preset == preset

    def test_strictness_preset_invalid_rejected(self):
        with pytest.raises(ValidationError):
            SufficiencyConfig(strictness_preset="bogus")  # type: ignore[arg-type]


class TestThresholdResolution:
    def test_valid_resolution(self):
        res = ThresholdResolution(
            name="epv_floor",
            value=5,
            source="literature_default",
            citation="Vergouwe 2007",
            inputs={"algorithm_family": "linear"},
        )
        assert res.value == 5
        assert res.source == "literature_default"

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            ThresholdResolution(
                name="epv_floor",
                value=5,
                source="something_else",  # type: ignore[arg-type]
                citation="x",
            )

    def test_default_inputs_is_empty_dict(self):
        res = ThresholdResolution(
            name="x",
            value=1,
            source="user_override",
            citation="y",
        )
        assert res.inputs == {}


class TestDataSufficiencyReport:
    def test_minimum_valid_report(self):
        report = DataSufficiencyReport(
            verdict="PASS",
            verdict_rationale="data_meets_requirements",
            n_rows=1000,
            n_features=10,
            problem_type="binary_classification",
        )
        assert report.verdict == "PASS"
        assert report.learning_curve is None

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            DataSufficiencyReport(
                verdict="MAYBE",  # type: ignore[arg-type]
                verdict_rationale="x",
                n_rows=100,
                n_features=5,
                problem_type="binary_classification",
            )

    def test_with_learning_curve(self):
        report = DataSufficiencyReport(
            verdict="SOFT_FAIL",
            verdict_rationale="curve still rising",
            n_rows=500,
            n_features=10,
            problem_type="binary_classification",
            learning_curve=[(100, 0.65, 0.02), (200, 0.72, 0.02), (500, 0.78, 0.01)],
            proxy_model="lightgbm-default",
            slope_at_max_n=0.0001,
            slope_pvalue=0.02,
            recommended_additional_samples=1500,
        )
        assert report.learning_curve is not None
        assert len(report.learning_curve) == 3

    def test_with_resolved_thresholds(self):
        report = DataSufficiencyReport(
            verdict="PASS",
            verdict_rationale="ok",
            n_rows=1000,
            n_features=10,
            problem_type="binary_classification",
            resolved_thresholds=[
                ThresholdResolution(
                    name="epv_floor",
                    value=5,
                    source="literature_default",
                    citation="Vergouwe 2007",
                )
            ],
        )
        assert len(report.resolved_thresholds) == 1
        assert report.resolved_thresholds[0].name == "epv_floor"
