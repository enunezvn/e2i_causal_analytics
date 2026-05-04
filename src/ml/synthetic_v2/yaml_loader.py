"""YAML scenario-config loader (shard 06).

Loads ``tests/configs/scenarios/{a,b,c}.yaml`` (schema_version
``synthetic_v2.scenario.v1``) and resolves them to frozen ``ScenarioSpec``
records. Schema validation rules per shard 06 §B.1.

The YAML is the **non-Python introspection surface** for the v2 generator
— ``Phase1MultiDiseaseRunner`` (shard 22 §A) reads these to discover and
dispatch scenarios without importing the Python registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName

SUPPORTED_SCHEMA_VERSION = "synthetic_v2.scenario.v1"
SUPPORTED_SHORT_CODES = frozenset({"A", "B", "C"})
SUPPORTED_USE_CASES = frozenset({"diagnostic", "screening", "treatment_decision"})


@dataclass(frozen=True)
class SyntheticConfigSpec:
    n_total: int
    prevalence: float
    signal_strength: str
    feature_count: int
    feature_correlation: str


@dataclass(frozen=True)
class ClinicalThresholdRangeSpec:
    use_case: str
    primary_tau: float
    tau_low: float
    tau_high: float


@dataclass(frozen=True)
class AUCBandSpec:
    low: float
    high: float


@dataclass(frozen=True)
class RWDConcurrentValidationSpec:
    enabled: bool
    rwd_loader: str
    rwd_data_path: str
    validation_metrics: tuple[str, ...]
    acceptance_thresholds: dict[str, float]


@dataclass(frozen=True)
class ReportingSpec:
    expected_winners: tuple[str, ...] = ()
    primary_decision_context: str = ""
    citation_anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioSpec:
    schema_version: str
    name: ScenarioName
    short_code: str
    franchise: str
    disease: str
    outcome_field: str
    synthetic_config: SyntheticConfigSpec
    clinical_threshold_range: ClinicalThresholdRangeSpec
    target_auc_band: AUCBandSpec
    reporting: ReportingSpec | None = None
    rwd_concurrent_validation: RWDConcurrentValidationSpec | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Round-trip dict matching the original YAML structure (drops ``raw``)."""
        d = asdict(self)
        d["name"] = self.name.value
        d.pop("raw", None)
        if self.reporting is None:
            d.pop("reporting", None)
        if self.rwd_concurrent_validation is None:
            d.pop("rwd_concurrent_validation", None)
        return d


def _validate_required(payload: dict[str, Any], required: list[str], context: str) -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"{context}: missing required keys {missing}")


def load_scenario_from_yaml(path: str | Path) -> ScenarioSpec:
    """Load + validate a YAML scenario config (shard 06 §B.1).

    Validation rules:
    1. ``schema_version == SUPPORTED_SCHEMA_VERSION`` (exact match).
    2. All required top-level keys present.
    3. ``name`` is a valid ``ScenarioName`` value.
    4. ``short_code`` ∈ ``{"A", "B", "C"}``.
    5. ``synthetic_config.feature_count`` matches the registered builder
       (warning suppressed if scenario not yet registered to allow
       commits 07-09 to land before this loader runs).
    6. ``tau_low < primary_tau < tau_high``.
    7. ``target_auc_band.low < target_auc_band.high``.
    8. ``prevalence ∈ (0, 1)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level mapping, got {type(raw).__name__}")

    _validate_required(
        raw,
        [
            "schema_version",
            "name",
            "short_code",
            "franchise",
            "disease",
            "outcome_field",
            "synthetic_config",
            "clinical_threshold_range",
            "target_auc_band",
        ],
        f"{path}",
    )

    if raw["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {raw['schema_version']!r} not supported; "
            f"expected {SUPPORTED_SCHEMA_VERSION!r}"
        )

    try:
        name = ScenarioName(raw["name"])
    except ValueError as exc:
        raise ValueError(
            f"{path}: name {raw['name']!r} is not a valid ScenarioName; "
            f"valid values: {sorted(s.value for s in ScenarioName)}"
        ) from exc

    short_code = str(raw["short_code"])
    if short_code not in SUPPORTED_SHORT_CODES:
        raise ValueError(
            f"{path}: short_code {short_code!r} not in {sorted(SUPPORTED_SHORT_CODES)}"
        )

    sc_raw = raw["synthetic_config"]
    _validate_required(
        sc_raw,
        ["n_total", "prevalence", "signal_strength", "feature_count", "feature_correlation"],
        f"{path}.synthetic_config",
    )
    if not 0.0 < sc_raw["prevalence"] < 1.0:
        raise ValueError(
            f"{path}.synthetic_config.prevalence must be in (0, 1); got {sc_raw['prevalence']}"
        )
    if name in SCENARIO_REGISTRY:
        builder = SCENARIO_REGISTRY[name]()
        if sc_raw["feature_count"] != len(builder.feature_manifest):
            raise ValueError(
                f"{path}.synthetic_config.feature_count={sc_raw['feature_count']} "
                f"does not match registered manifest length "
                f"{len(builder.feature_manifest)} for {name.value}"
            )
    synthetic_config = SyntheticConfigSpec(
        n_total=int(sc_raw["n_total"]),
        prevalence=float(sc_raw["prevalence"]),
        signal_strength=str(sc_raw["signal_strength"]),
        feature_count=int(sc_raw["feature_count"]),
        feature_correlation=str(sc_raw["feature_correlation"]),
    )

    ctr_raw = raw["clinical_threshold_range"]
    _validate_required(
        ctr_raw,
        ["use_case", "primary_tau", "tau_low", "tau_high"],
        f"{path}.clinical_threshold_range",
    )
    if ctr_raw["use_case"] not in SUPPORTED_USE_CASES:
        raise ValueError(
            f"{path}.clinical_threshold_range.use_case {ctr_raw['use_case']!r} "
            f"not in {sorted(SUPPORTED_USE_CASES)}"
        )
    if not (ctr_raw["tau_low"] < ctr_raw["primary_tau"] < ctr_raw["tau_high"]):
        raise ValueError(
            f"{path}.clinical_threshold_range: tau_low ({ctr_raw['tau_low']}) < "
            f"primary_tau ({ctr_raw['primary_tau']}) < "
            f"tau_high ({ctr_raw['tau_high']}) is violated"
        )
    ctr = ClinicalThresholdRangeSpec(
        use_case=str(ctr_raw["use_case"]),
        primary_tau=float(ctr_raw["primary_tau"]),
        tau_low=float(ctr_raw["tau_low"]),
        tau_high=float(ctr_raw["tau_high"]),
    )

    band_raw = raw["target_auc_band"]
    _validate_required(band_raw, ["low", "high"], f"{path}.target_auc_band")
    if band_raw["low"] >= band_raw["high"]:
        raise ValueError(
            f"{path}.target_auc_band: low ({band_raw['low']}) must be < high ({band_raw['high']})"
        )
    auc_band = AUCBandSpec(low=float(band_raw["low"]), high=float(band_raw["high"]))

    reporting: ReportingSpec | None = None
    if "reporting" in raw and raw["reporting"] is not None:
        rep_raw = raw["reporting"]
        reporting = ReportingSpec(
            expected_winners=tuple(rep_raw.get("expected_winners", []) or []),
            primary_decision_context=str(rep_raw.get("primary_decision_context", "")),
            citation_anchors=tuple(rep_raw.get("citation_anchors", []) or []),
        )

    rwd: RWDConcurrentValidationSpec | None = None
    if "rwd_concurrent_validation" in raw and raw["rwd_concurrent_validation"] is not None:
        rwd_raw = raw["rwd_concurrent_validation"]
        _validate_required(
            rwd_raw,
            [
                "enabled",
                "rwd_loader",
                "rwd_data_path",
                "validation_metrics",
                "acceptance_thresholds",
            ],
            f"{path}.rwd_concurrent_validation",
        )
        rwd = RWDConcurrentValidationSpec(
            enabled=bool(rwd_raw["enabled"]),
            rwd_loader=str(rwd_raw["rwd_loader"]),
            rwd_data_path=str(rwd_raw["rwd_data_path"]),
            validation_metrics=tuple(rwd_raw["validation_metrics"]),
            acceptance_thresholds=dict(rwd_raw["acceptance_thresholds"]),
        )

    return ScenarioSpec(
        schema_version=raw["schema_version"],
        name=name,
        short_code=short_code,
        franchise=str(raw["franchise"]),
        disease=str(raw["disease"]),
        outcome_field=str(raw["outcome_field"]),
        synthetic_config=synthetic_config,
        clinical_threshold_range=ctr,
        target_auc_band=auc_band,
        reporting=reporting,
        rwd_concurrent_validation=rwd,
        raw=raw,
    )


def discover_scenarios(scenarios_dir: str | Path = "tests/configs/scenarios") -> list[ScenarioSpec]:
    """Yield ``ScenarioSpec`` for every YAML in ``scenarios_dir`` (shard 06 §B.2)."""
    p = Path(scenarios_dir)
    if not p.exists():
        raise FileNotFoundError(f"scenarios directory not found: {p}")
    return [load_scenario_from_yaml(yaml_path) for yaml_path in sorted(p.glob("*.yaml"))]
