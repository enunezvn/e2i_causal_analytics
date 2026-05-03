"""Tests for YAML scenario configs + loader (shard 06 §F)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName
from src.ml.synthetic_v2.yaml_loader import (
    SUPPORTED_SCHEMA_VERSION,
    ScenarioSpec,
    discover_scenarios,
    load_scenario_from_yaml,
)

CONFIGS_DIR = Path("tests/configs/scenarios")
ALL_YAML_PATHS = [CONFIGS_DIR / f"{c}.yaml" for c in ("a", "b", "c")]


class TestSchemaVersionPinning:
    @pytest.mark.parametrize("path", ALL_YAML_PATHS)
    def test_schema_version_matches(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text())
        assert raw["schema_version"] == SUPPORTED_SCHEMA_VERSION


class TestFeatureCountAlignment:
    @pytest.mark.parametrize(
        "yaml_path,expected_features",
        [
            (CONFIGS_DIR / "a.yaml", 40),
            (CONFIGS_DIR / "b.yaml", 25),
            (CONFIGS_DIR / "c.yaml", 60),
        ],
    )
    def test_feature_count_matches_manifest(self, yaml_path: Path, expected_features: int) -> None:
        spec = load_scenario_from_yaml(yaml_path)
        assert spec.synthetic_config.feature_count == expected_features
        assert len(SCENARIO_REGISTRY[spec.name]().feature_manifest) == expected_features


class TestLoaderReturnsScenarioSpec:
    @pytest.mark.parametrize("path", ALL_YAML_PATHS)
    def test_load_returns_spec(self, path: Path) -> None:
        spec = load_scenario_from_yaml(path)
        assert isinstance(spec, ScenarioSpec)
        assert spec.schema_version == SUPPORTED_SCHEMA_VERSION
        assert isinstance(spec.name, ScenarioName)


class TestRoundtrip:
    @pytest.mark.parametrize("path", ALL_YAML_PATHS)
    def test_roundtrip_preserves_keys(self, path: Path) -> None:
        spec = load_scenario_from_yaml(path)
        original = yaml.safe_load(path.read_text())
        restored = spec.to_dict()
        for key in original:
            assert key in restored, f"Round-trip dropped key {key!r}"
        for key in ("schema_version", "name", "short_code", "outcome_field"):
            assert restored[key] == original[key]


class TestThresholdRangeWellFormed:
    @pytest.mark.parametrize("path", ALL_YAML_PATHS)
    def test_tau_low_lt_primary_lt_high(self, path: Path) -> None:
        spec = load_scenario_from_yaml(path)
        ctr = spec.clinical_threshold_range
        assert ctr.tau_low < ctr.primary_tau < ctr.tau_high


class TestAUCBandWellFormed:
    @pytest.mark.parametrize("path", ALL_YAML_PATHS)
    def test_low_lt_high(self, path: Path) -> None:
        spec = load_scenario_from_yaml(path)
        assert spec.target_auc_band.low < spec.target_auc_band.high


class TestRWDBlockAsymmetry:
    def test_a_has_no_rwd_block(self) -> None:
        spec = load_scenario_from_yaml(CONFIGS_DIR / "a.yaml")
        assert spec.rwd_concurrent_validation is None

    def test_b_has_no_rwd_block(self) -> None:
        spec = load_scenario_from_yaml(CONFIGS_DIR / "b.yaml")
        assert spec.rwd_concurrent_validation is None

    def test_c_has_full_rwd_block(self) -> None:
        spec = load_scenario_from_yaml(CONFIGS_DIR / "c.yaml")
        assert spec.rwd_concurrent_validation is not None
        rwd = spec.rwd_concurrent_validation
        assert rwd.enabled is True
        assert rwd.rwd_loader == "scripts.convert_csu_rwd"
        assert "feature_distribution_ks" in rwd.validation_metrics
        assert "auc_delta" in rwd.validation_metrics
        assert rwd.acceptance_thresholds["feature_distribution_ks_max_fail_rate"] == 0.25
        assert rwd.acceptance_thresholds["auc_delta_max_abs"] == 0.10


class TestDiscoverScenarios:
    def test_discover_returns_three(self) -> None:
        specs = discover_scenarios(CONFIGS_DIR)
        assert len(specs) == 3
        codes = {spec.short_code for spec in specs}
        assert codes == {"A", "B", "C"}

    def test_discover_missing_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            discover_scenarios("does/not/exist")


class TestValidationFailures:
    def test_unknown_schema_version_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "schema_version: synthetic_v2.scenario.v999\n"
            "name: scenario_a_diagnostic_ebc_idfs_5y\n"
            "short_code: A\n"
            "franchise: x\ndisease: y\noutcome_field: z\n"
            "synthetic_config: {n_total: 10, prevalence: 0.1, signal_strength: low,"
            " feature_count: 40, feature_correlation: low}\n"
            "clinical_threshold_range: {use_case: diagnostic, primary_tau: 0.2,"
            " tau_low: 0.1, tau_high: 0.3}\n"
            "target_auc_band: {low: 0.7, high: 0.8}\n"
        )
        with pytest.raises(ValueError, match="schema_version"):
            load_scenario_from_yaml(bad)

    def test_unknown_name_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            f"schema_version: {SUPPORTED_SCHEMA_VERSION}\n"
            "name: scenario_z_unknown\n"
            "short_code: A\n"
            "franchise: x\ndisease: y\noutcome_field: z\n"
            "synthetic_config: {n_total: 10, prevalence: 0.1, signal_strength: low,"
            " feature_count: 40, feature_correlation: low}\n"
            "clinical_threshold_range: {use_case: diagnostic, primary_tau: 0.2,"
            " tau_low: 0.1, tau_high: 0.3}\n"
            "target_auc_band: {low: 0.7, high: 0.8}\n"
        )
        with pytest.raises(ValueError, match="not a valid ScenarioName"):
            load_scenario_from_yaml(bad)

    def test_invalid_threshold_range_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            f"schema_version: {SUPPORTED_SCHEMA_VERSION}\n"
            "name: scenario_a_diagnostic_ebc_idfs_5y\n"
            "short_code: A\n"
            "franchise: x\ndisease: y\noutcome_field: z\n"
            "synthetic_config: {n_total: 10, prevalence: 0.1, signal_strength: low,"
            " feature_count: 40, feature_correlation: low}\n"
            "clinical_threshold_range: {use_case: diagnostic, primary_tau: 0.5,"
            " tau_low: 0.1, tau_high: 0.3}\n"
            "target_auc_band: {low: 0.7, high: 0.8}\n"
        )
        with pytest.raises(ValueError, match="primary_tau"):
            load_scenario_from_yaml(bad)

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_scenario_from_yaml(tmp_path / "missing.yaml")
