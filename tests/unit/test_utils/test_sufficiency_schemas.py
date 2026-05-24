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


# ---------------------------------------------------------------------------
# PR #462 hotfix: F2 (SufficiencyConfig.force_low_power_run) +
# F5 (target_mde bounds tightening) + F7 (DataSufficiencyReport override
# audit fields) + F11 (SufficiencyVerdict adds 'SKIPPED') +
# F14 (DataSufficiencyReport.detectable_mde_at_n_capped)
# ---------------------------------------------------------------------------


class TestF2SufficiencyConfigForceLowPowerRun:
    """F2: SufficiencyConfig must declare force_low_power_run so that
    typed callers (e.g. via SufficiencyConfig(...)) don't hit
    ValidationError under extra='forbid'.
    """

    def test_force_low_power_run_accepts_true(self):
        cfg = SufficiencyConfig(force_low_power_run=True)
        assert cfg.force_low_power_run is True

    def test_force_low_power_run_default_is_false(self):
        # D5: safe-by-default for pharma regulatory contexts.
        cfg = SufficiencyConfig()
        assert cfg.force_low_power_run is False

    def test_force_low_power_run_round_trips(self):
        cfg = SufficiencyConfig(force_low_power_run=True, epv_floor=8)
        restored = SufficiencyConfig.model_validate(cfg.model_dump())
        assert restored.force_low_power_run is True
        assert restored.epv_floor == 8


class TestF5TargetMdeValidation:
    """F5 (PR #462 hotfix round 1) + R2.2 (round 2): SufficiencyConfig.target_mde
    must reject hard-invalid overrides (NaN, non-positive, absurdly large)
    at schema construction time.

    R2.2 dual-semantic contract: ``target_mde`` is now ``Field(gt=0.0, lt=1e6)``
    (loosened from round-1's ``lt=1.0``) so that continuous-outcome raw
    effect sizes — ``0.5 * sigma_outcome`` for ``sigma_outcome > 2.0`` —
    don't get schema-rejected. The binary-outcome (0, 1) constraint is
    re-asserted at the resolver layer (``resolve_target_mde``) which has
    access to ``outcome_type`` and can apply the right bound.
    """

    # Hard-invalid at schema layer: <= 0 or >= 1e6 (typo guard).
    @pytest.mark.parametrize("bad_value", [-0.5, -0.01, 0.0, 1e6, 1e9])
    def test_out_of_bounds_target_mde_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            SufficiencyConfig(target_mde=bad_value)

    def test_nan_target_mde_rejected(self):
        import math as _math

        with pytest.raises(ValidationError):
            SufficiencyConfig(target_mde=_math.nan)

    @pytest.mark.parametrize("good_value", [0.001, 0.05, 0.2, 0.5, 0.95])
    def test_in_bounds_target_mde_accepted(self, good_value):
        cfg = SufficiencyConfig(target_mde=good_value)
        assert cfg.target_mde == good_value

    # R2.2: continuous-outcome raw effect sizes — schema-accepted, but the
    # resolver's outcome-type-aware guard rejects them for binary outcomes
    # (covered in test_sufficiency_resolver.py).
    @pytest.mark.parametrize("continuous_value", [1.0, 1.5, 2.0, 5.0, 100.0])
    def test_continuous_outcome_raw_effect_size_accepted(self, continuous_value):
        """R2.2: ``0.5 * sigma_outcome`` for sigma > 2 produces values >= 1.0
        (raw effect size in outcome units, NOT Cohen's d). The schema must
        accept these — the binary-shape (0, 1) constraint is enforced at the
        resolver layer where outcome_type is known.
        """
        cfg = SufficiencyConfig(target_mde=continuous_value)
        assert cfg.target_mde == continuous_value


class TestF11SkippedVerdict:
    """F11: SufficiencyVerdict must include 'SKIPPED' so SKIPPED reports
    actually validate against the schema."""

    def test_skipped_verdict_accepted(self):
        from src.utils.sufficiency_schemas import DataSufficiencyReport

        report = DataSufficiencyReport(
            verdict="SKIPPED",
            verdict_rationale="Pre-flight skipped: use_sample_data=True",
            n_rows=0,
            n_features=0,
            problem_type="binary_classification",
        )
        assert report.verdict == "SKIPPED"


class TestF7OverrideAuditFields:
    """F7: DataSufficiencyReport must declare `override_applied` +
    `original_verdict` so the audit chain captures causal-SOFT-FAIL bypasses.
    """

    def test_override_audit_fields_default(self):
        report = DataSufficiencyReport(
            verdict="PASS",
            verdict_rationale="ok",
            n_rows=1000,
            n_features=10,
            problem_type="binary_classification",
        )
        assert report.override_applied is False
        assert report.original_verdict is None

    def test_override_audit_fields_populated(self):
        report = DataSufficiencyReport(
            verdict="SOFT_FAIL",
            verdict_rationale="n=250 below recommended [OVERRIDDEN via force_low_power_run]",
            n_rows=250,
            n_features=10,
            problem_type="causal_inference",
            override_applied=True,
            original_verdict="SOFT_FAIL",
        )
        assert report.override_applied is True
        assert report.original_verdict == "SOFT_FAIL"

    def test_override_audit_fields_round_trip(self):
        from src.utils.sufficiency_schemas import DataSufficiencyReport

        original = DataSufficiencyReport(
            verdict="SOFT_FAIL",
            verdict_rationale="x [OVERRIDDEN via force_low_power_run]",
            n_rows=250,
            n_features=10,
            problem_type="causal_inference",
            override_applied=True,
            original_verdict="SOFT_FAIL",
        )
        restored = DataSufficiencyReport.model_validate(original.model_dump())
        assert restored.override_applied is True
        assert restored.original_verdict == "SOFT_FAIL"


class TestF14MdeCappedField:
    """F14: DataSufficiencyReport must declare
    `detectable_mde_at_n_capped` so the report consumer can detect when
    the binary MDE was clamped at the boundary (asymptotic formula
    invalid at that n)."""

    def test_mde_capped_default(self):
        report = DataSufficiencyReport(
            verdict="PASS",
            verdict_rationale="ok",
            n_rows=1000,
            n_features=10,
            problem_type="binary_classification",
        )
        assert report.detectable_mde_at_n_capped is None

    def test_mde_capped_true(self):
        report = DataSufficiencyReport(
            verdict="SOFT_FAIL",
            verdict_rationale="x; clamped at boundary",
            n_rows=120,
            n_features=2,
            problem_type="binary_classification",
            detectable_mde_at_current_n=0.05,
            detectable_mde_units="absolute_risk_difference",
            detectable_mde_at_n_capped=True,
        )
        assert report.detectable_mde_at_n_capped is True


class TestF4TargetMdeSource:
    """F4 / D1: SufficiencyConfig must declare `target_mde_source` so
    the audit chain records the provenance of the target_mde value."""

    @pytest.mark.parametrize(
        "source",
        ["user_override", "computed_from_data", "literature_default"],
    )
    def test_target_mde_source_accepts_valid_sources(self, source):
        cfg = SufficiencyConfig(target_mde=0.05, target_mde_source=source)
        assert cfg.target_mde_source == source

    def test_target_mde_source_rejects_invalid(self):
        with pytest.raises(ValidationError):
            SufficiencyConfig(target_mde_source="not_a_real_source")  # type: ignore[arg-type]


class TestF2F4ScopeSpecSufficiencyFieldRoundTrip:
    """F1: ScopeSpecSchema must declare `sufficiency` so user overrides
    survive pydantic coercion (extra='ignore' was silently dropping them)."""

    def test_scope_spec_accepts_sufficiency_dict(self):
        from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema

        schema = ScopeSpecSchema(
            problem_type="binary_classification",
            sufficiency={"target_mde": 0.05, "force_low_power_run": True},
        )
        assert schema.sufficiency is not None
        assert schema.sufficiency.target_mde == 0.05
        assert schema.sufficiency.force_low_power_run is True

    def test_scope_spec_accepts_typed_sufficiency_config(self):
        from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema

        schema = ScopeSpecSchema(
            problem_type="binary_classification",
            sufficiency=SufficiencyConfig(target_mde=0.05, force_low_power_run=True, epv_floor=8),
        )
        assert schema.sufficiency is not None
        assert schema.sufficiency.target_mde == 0.05
        assert schema.sufficiency.force_low_power_run is True
        assert schema.sufficiency.epv_floor == 8

    def test_scope_spec_sufficiency_round_trips_through_json(self):
        from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema

        original = ScopeSpecSchema(
            problem_type="causal_inference",
            sufficiency={
                "target_mde": 0.05,
                "force_low_power_run": True,
                "epv_floor": 12,
                "strictness_preset": "strict",
            },
        )
        restored = ScopeSpecSchema.model_validate_json(original.model_dump_json())
        assert restored.sufficiency is not None
        assert restored.sufficiency.target_mde == 0.05
        assert restored.sufficiency.force_low_power_run is True
        assert restored.sufficiency.epv_floor == 12
        assert restored.sufficiency.strictness_preset == "strict"

    def test_scope_spec_without_sufficiency_defaults_to_none(self):
        from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema

        schema = ScopeSpecSchema(problem_type="regression")
        assert schema.sufficiency is None
