"""
ROI Calculation Service.

Implements the full E2I ROI methodology including:
- 6 value driver calculations
- Bootstrap confidence intervals (1,000 simulations)
- Causal attribution framework (4 levels)
- Risk adjustment (multiplicative formula)
- Sensitivity analysis (tornado diagrams)
- Time value of money (NPV calculations)

Reference: docs/roi_methodology.md
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Enums
# =============================================================================


class ValueDriverType(str, Enum):
    """Value driver types as defined in ROI methodology."""

    TRX_LIFT = "trx_lift"  # $850/TRx
    PATIENT_IDENTIFICATION = "patient_identification"  # $1,200/patient
    ACTION_RATE = "action_rate"  # $45/pp/1000 triggers
    INTENT_TO_PRESCRIBE = "intent_to_prescribe"  # $320/HCP/pp
    DATA_QUALITY = "data_quality"  # $200/FP, $650/FN
    DRIFT_PREVENTION = "drift_prevention"  # 2x multiplier
    UPLIFT_TARGETING = "uplift_targeting"  # Targeting efficiency from uplift models


class AttributionLevel(str, Enum):
    """Causal attribution levels."""

    FULL = "full"  # 100% - RCT validated, sole driver
    PARTIAL = "partial"  # 50-80% - Primary driver, some confounding
    SHARED = "shared"  # 20-50% - Multiple initiatives contribute
    MINIMAL = "minimal"  # <20% - Minor contributor, correlation only


class RiskLevel(str, Enum):
    """Risk factor levels for adjustment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InitiativeType(str, Enum):
    """Initiative types for cost estimation."""

    DATA_SOURCE_INTEGRATION = "data_source_integration"
    NEW_ML_MODEL = "new_ml_model"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    DASHBOARD_ENHANCEMENT = "dashboard_enhancement"
    TRIGGER_REDESIGN = "trigger_redesign"
    AB_TEST_IMPLEMENTATION = "ab_test_implementation"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValueDriverInput:
    """Input for a single value driver calculation."""

    driver_type: ValueDriverType
    quantity: float  # Number of units (TRx, patients, pp, etc.)

    # Optional parameters for specific drivers
    hcp_count: Optional[int] = None  # For ITP calculations
    trigger_count: Optional[int] = None  # For action rate calculations
    fp_reduction: Optional[int] = None  # For data quality
    fn_reduction: Optional[int] = None  # For data quality
    auc_drop_prevented: Optional[float] = None  # For drift prevention
    baseline_model_value: Optional[float] = None  # For drift prevention

    # Uplift targeting parameters (from CausalML integration)
    auuc: Optional[float] = None  # Area Under Uplift Curve (0-1)
    qini_coefficient: Optional[float] = None  # Qini coefficient
    targeting_efficiency: Optional[float] = None  # Targeting efficiency (0-1)
    baseline_treatment_value: Optional[float] = None  # Value before targeting optimization
    targeted_population_size: Optional[int] = None  # Number of individuals targeted

    # Bootstrap distribution parameters
    uncertainty_std: Optional[float] = None  # Std dev as fraction of mean (default 0.15)


@dataclass
class CostInput:
    """Cost inputs for ROI calculation."""

    engineering_days: float = 0
    engineering_day_rate: float = 2500.0

    # Data acquisition
    data_source_costs: Dict[str, float] = field(default_factory=dict)
    incremental_data_cost: float = 0

    # Training & change management
    training_cost: float = 0
    change_management_cost: float = 0

    # Infrastructure
    monthly_infrastructure_cost: float = 0
    infrastructure_months: int = 12

    # Opportunity cost
    delayed_initiative_annual_value: float = 0
    delay_months: int = 0


@dataclass
class RiskAssessment:
    """Risk factor assessment for risk-adjusted ROI."""

    technical_complexity: RiskLevel = RiskLevel.LOW  # 0/15/30%
    organizational_change: RiskLevel = RiskLevel.LOW  # 0/20/40%
    data_dependencies: RiskLevel = RiskLevel.LOW  # 0/25/50%
    timeline_uncertainty: RiskLevel = RiskLevel.LOW  # 0/10/25%


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval results."""

    lower_bound: float  # 2.5th percentile
    median: float  # 50th percentile
    upper_bound: float  # 97.5th percentile
    probability_positive: float  # P(ROI > 1x)
    probability_target: float  # P(ROI > target)
    simulation_count: int = 1000


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""

    parameter: str
    base_value: float
    low_value: float  # -20%
    high_value: float  # +20%
    roi_at_low: float
    roi_at_base: float
    roi_at_high: float
    impact_range: float  # high - low


@dataclass
class ROIResult:
    """Complete ROI calculation result."""

    # Core metrics
    incremental_value: float
    implementation_cost: float
    base_roi: float  # Before adjustments
    risk_adjusted_roi: float  # After risk adjustment

    # Confidence interval
    confidence_interval: ConfidenceInterval

    # Attribution
    attribution_level: AttributionLevel
    attribution_rate: float  # 0.0-1.0
    attributed_value: float  # Value after attribution

    # Risk breakdown
    risk_assessment: RiskAssessment
    total_risk_adjustment: float  # 0.0-1.0

    # Breakdown by value driver
    value_by_driver: Dict[str, float]
    cost_breakdown: Dict[str, float]

    # Sensitivity analysis (optional)
    sensitivity_results: Optional[List[SensitivityResult]] = None

    # NPV (if multi-year)
    npv_value: Optional[float] = None
    npv_roi: Optional[float] = None

    # Metadata
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    methodology_version: str = "1.0"


# =============================================================================
# Value Driver Calculators
# =============================================================================


class TRxLiftCalculator:
    """Calculate value from TRx lift."""

    VALUE_PER_TRX = 850.0  # $850 per incremental TRx

    def calculate(self, incremental_trx: float) -> float:
        """
        Calculate TRx lift value.

        Args:
            incremental_trx: Number of incremental prescriptions

        Returns:
            Dollar value of TRx lift
        """
        return incremental_trx * self.VALUE_PER_TRX


class PatientIdentificationCalculator:
    """Calculate value from patient identification improvements."""

    VALUE_PER_PATIENT = 1200.0  # $1,200 per identified patient

    def calculate(self, identified_patients: float) -> float:
        """
        Calculate patient identification value.

        Includes lifetime value: diagnostic confirmation, HCP engagement,
        patient support program enrollment, 60% conversion to TRx,
        downstream adherence and refill value.

        Args:
            identified_patients: Number of patients correctly identified

        Returns:
            Dollar value of patient identification
        """
        return identified_patients * self.VALUE_PER_PATIENT


class ActionRateCalculator:
    """Calculate value from HCP trigger acceptance rate improvement."""

    VALUE_PER_PP = 45.0  # $45 per percentage point per 1,000 triggers

    def calculate(
        self,
        action_rate_improvement_pp: float,
        trigger_count: int = 1000,
    ) -> float:
        """
        Calculate action rate improvement value.

        Args:
            action_rate_improvement_pp: Improvement in percentage points
            trigger_count: Monthly trigger volume (default 1,000)

        Returns:
            Annual dollar value of action rate improvement
        """
        # Annual calculation: pp × $45 × triggers × 12 months
        return action_rate_improvement_pp * self.VALUE_PER_PP * trigger_count * 12


class IntentToPrescribeCalculator:
    """Calculate value from Intent-to-Prescribe lift."""

    VALUE_PER_HCP_PP = 320.0  # $320 per HCP per percentage point

    def calculate(
        self,
        itp_improvement_pp: float,
        hcp_count: int,
    ) -> float:
        """
        Calculate ITP lift value.

        Based on: 1pp ITP increase → +0.4 TRx/HCP/year
        0.4 TRx × $850 = $340, discounted 94% for survey noise = $320

        Args:
            itp_improvement_pp: ITP improvement in percentage points
            hcp_count: Number of HCPs affected

        Returns:
            Annual dollar value of ITP lift
        """
        return itp_improvement_pp * self.VALUE_PER_HCP_PP * hcp_count


class DataQualityCalculator:
    """Calculate value from data quality improvements."""

    VALUE_PER_FP_AVOIDED = 200.0  # $200 per false positive avoided
    VALUE_PER_FN_AVOIDED = 650.0  # $650 per false negative avoided

    def calculate(
        self,
        fp_reduction: int,
        fn_reduction: int,
    ) -> float:
        """
        Calculate data quality improvement value.

        FP avoided: Saves wasted rep time, customer annoyance, channel fatigue
        FN avoided: Captures missed opportunities, prevents competitor prescribes

        Args:
            fp_reduction: Monthly false positive reduction
            fn_reduction: Monthly false negative reduction

        Returns:
            Annual dollar value of data quality improvement
        """
        monthly_value = (
            fp_reduction * self.VALUE_PER_FP_AVOIDED + fn_reduction * self.VALUE_PER_FN_AVOIDED
        )
        return monthly_value * 12


class DriftPreventionCalculator:
    """Calculate value from drift detection and prevention."""

    MULTIPLIER = 2.0  # 2x for retraining, downtime, business disruption

    def calculate(
        self,
        auc_drop_prevented: float,
        baseline_model_value: float,
    ) -> float:
        """
        Calculate drift prevention value.

        Value = Prevented AUC drop × Baseline model value × 2x multiplier

        Args:
            auc_drop_prevented: AUC degradation prevented (e.g., 0.05)
            baseline_model_value: Annual value generated by model

        Returns:
            Dollar value of drift prevention
        """
        return auc_drop_prevented * baseline_model_value * self.MULTIPLIER


class UpliftTargetingCalculator:
    """Calculate value from uplift-based targeting optimization.

    Uses CausalML uplift models to identify high/low responders and
    optimize treatment allocation for maximum ROI.

    Value formula:
    - Base: AUUC × Baseline treatment value × Targeting efficiency
    - Additional: Top-decile lift × Population × Per-unit value

    Reference: src/causal_engine/uplift/ for CausalML integration
    """

    # Base per-capita value from optimal targeting
    VALUE_PER_TARGETED_INDIVIDUAL = 125.0  # $125 per optimally targeted individual

    # AUUC multiplier (higher AUUC = better targeting)
    AUUC_MULTIPLIER = 2.5

    def calculate(
        self,
        auuc: float,
        targeting_efficiency: float,
        baseline_treatment_value: Optional[float] = None,
        targeted_population_size: Optional[int] = None,
        qini_coefficient: Optional[float] = None,
    ) -> float:
        """
        Calculate uplift targeting optimization value.

        Value represents the incremental benefit from using uplift models
        to target treatments to individuals most likely to respond.

        Args:
            auuc: Area Under Uplift Curve (0-1), measures model's ability
                  to rank individuals by treatment effect
            targeting_efficiency: Fraction of value captured vs random (0-1)
            baseline_treatment_value: Current value from treatment (default: $50,000)
            targeted_population_size: Number of individuals in target population
            qini_coefficient: Qini coefficient (optional, for validation)

        Returns:
            Dollar value of uplift-based targeting optimization
        """
        # Default baseline value if not provided
        if baseline_treatment_value is None:
            baseline_treatment_value = 50000.0

        # Default population size if not provided
        if targeted_population_size is None:
            targeted_population_size = 1000

        # Component 1: Value from improved targeting (AUUC-based)
        # Higher AUUC means better discrimination between responders/non-responders
        auuc_value = auuc * baseline_treatment_value * self.AUUC_MULTIPLIER

        # Component 2: Per-capita value from targeting efficiency
        # More efficient targeting = more value captured per person
        efficiency_value = (
            targeting_efficiency * targeted_population_size * self.VALUE_PER_TARGETED_INDIVIDUAL
        )

        # Total uplift targeting value
        total_value = auuc_value + efficiency_value

        return total_value


# =============================================================================
# Bootstrap Simulator
# =============================================================================


class BootstrapSimulator:
    """
    Monte Carlo simulation for ROI confidence intervals.

    Runs 1,000 simulations sampling from parameter distributions.
    """

    DEFAULT_SIMULATIONS = 1000
    DEFAULT_UNCERTAINTY = 0.15  # 15% std dev as fraction of mean

    def __init__(self, n_simulations: int = DEFAULT_SIMULATIONS, seed: Optional[int] = None):
        """
        Initialize bootstrap simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate_value(
        self,
        mean: float,
        std_fraction: float = DEFAULT_UNCERTAINTY,
    ) -> np.ndarray:
        """
        Simulate value using normal distribution.

        For value parameters (TRx lift, patient count, etc.)
        Uses μ = point estimate, σ = 0.15μ by default

        Args:
            mean: Point estimate
            std_fraction: Std dev as fraction of mean

        Returns:
            Array of simulated values
        """
        std = mean * std_fraction
        samples = self.rng.normal(mean, std, self.n_simulations)
        return np.maximum(samples, 0)  # Value can't be negative

    def simulate_cost(
        self,
        mean: float,
        shape: float = 2.0,
    ) -> np.ndarray:
        """
        Simulate cost using gamma distribution.

        Gamma captures right skew (costs often exceed estimates).

        Args:
            mean: Expected cost
            shape: Gamma shape parameter (default 2.0)

        Returns:
            Array of simulated costs
        """
        if mean <= 0:
            return np.zeros(self.n_simulations)
        scale = mean / shape
        return self.rng.gamma(shape, scale, self.n_simulations)

    def simulate_acceptance_rate(
        self,
        base_rate: float,
        improvement_pp: float,
    ) -> np.ndarray:
        """
        Simulate acceptance rate using beta distribution.

        Beta distribution is appropriate for rates/proportions.

        Args:
            base_rate: Base acceptance rate (0-1)
            improvement_pp: Expected improvement in percentage points

        Returns:
            Array of simulated improvement values
        """
        # Convert to alpha/beta for Beta distribution
        # Using method of moments from historical A/B tests
        expected = improvement_pp / 100  # Convert pp to fraction
        variance = (expected * 0.3) ** 2  # 30% uncertainty

        # Beta parameters from method of moments
        if variance > 0:
            alpha = expected * ((expected * (1 - expected) / variance) - 1)
            beta = (1 - expected) * ((expected * (1 - expected) / variance) - 1)
            alpha = max(0.5, alpha)
            beta = max(0.5, beta)
            samples = self.rng.beta(alpha, beta, self.n_simulations)
        else:
            samples = np.full(self.n_simulations, expected)

        return samples * 100  # Convert back to percentage points

    def compute_confidence_interval(
        self,
        roi_samples: np.ndarray,
        target_roi: float = 1.0,
    ) -> ConfidenceInterval:
        """
        Compute 95% confidence interval from simulated ROI values.

        Args:
            roi_samples: Array of simulated ROI values
            target_roi: Target ROI threshold for probability calculation

        Returns:
            ConfidenceInterval with bounds and probabilities
        """
        return ConfidenceInterval(
            lower_bound=float(np.percentile(roi_samples, 2.5)),
            median=float(np.percentile(roi_samples, 50)),
            upper_bound=float(np.percentile(roi_samples, 97.5)),
            probability_positive=float(np.mean(roi_samples > 1.0)),
            probability_target=float(np.mean(roi_samples > target_roi)),
            simulation_count=self.n_simulations,
        )


# =============================================================================
# Attribution Calculator
# =============================================================================


class AttributionCalculator:
    """
    Calculate causal attribution for ROI estimates.

    Prevents overclaiming by adjusting value based on attribution level.
    """

    ATTRIBUTION_RATES = {
        AttributionLevel.FULL: 1.0,  # 100%
        AttributionLevel.PARTIAL: 0.65,  # 65% (midpoint of 50-80%)
        AttributionLevel.SHARED: 0.35,  # 35% (midpoint of 20-50%)
        AttributionLevel.MINIMAL: 0.10,  # 10% (midpoint of <20%)
    }

    def get_attribution_rate(self, level: AttributionLevel) -> float:
        """Get attribution rate for a given level."""
        return self.ATTRIBUTION_RATES.get(level, 0.5)

    def apply_attribution(
        self,
        value: float,
        level: AttributionLevel,
    ) -> Tuple[float, float]:
        """
        Apply attribution adjustment to value.

        Args:
            value: Raw incremental value
            level: Attribution level

        Returns:
            Tuple of (attributed_value, attribution_rate)
        """
        rate = self.get_attribution_rate(level)
        return value * rate, rate


# =============================================================================
# Risk Adjustment Calculator
# =============================================================================


class RiskAdjustmentCalculator:
    """
    Apply risk adjustments to ROI estimates.

    Uses multiplicative formula: Risk = 1 - Product(1 - individual_adjustments)
    """

    RISK_FACTORS = {
        "technical_complexity": {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.15,
            RiskLevel.HIGH: 0.30,
        },
        "organizational_change": {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.20,
            RiskLevel.HIGH: 0.40,
        },
        "data_dependencies": {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.25,
            RiskLevel.HIGH: 0.50,
        },
        "timeline_uncertainty": {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.10,
            RiskLevel.HIGH: 0.25,
        },
    }

    def calculate_total_adjustment(self, assessment: RiskAssessment) -> float:
        """
        Calculate total risk adjustment using multiplicative formula.

        Formula: Total = 1 - Product(1 - individual_adjustments)

        Args:
            assessment: Risk assessment for all factors

        Returns:
            Total risk adjustment (0.0-1.0)
        """
        factors = [
            self.RISK_FACTORS["technical_complexity"][assessment.technical_complexity],
            self.RISK_FACTORS["organizational_change"][assessment.organizational_change],
            self.RISK_FACTORS["data_dependencies"][assessment.data_dependencies],
            self.RISK_FACTORS["timeline_uncertainty"][assessment.timeline_uncertainty],
        ]

        # Multiplicative formula
        product = 1.0
        for factor in factors:
            product *= 1 - factor

        return 1 - product

    def apply_risk_adjustment(
        self,
        roi: float,
        assessment: RiskAssessment,
    ) -> Tuple[float, float]:
        """
        Apply risk adjustment to ROI.

        Args:
            roi: Base ROI
            assessment: Risk assessment

        Returns:
            Tuple of (risk_adjusted_roi, total_adjustment)
        """
        adjustment = self.calculate_total_adjustment(assessment)
        adjusted_roi = roi * (1 - adjustment)
        return adjusted_roi, adjustment


# =============================================================================
# Cost Calculator
# =============================================================================


class CostCalculator:
    """Calculate implementation costs for initiatives."""

    def calculate_total_cost(self, cost_input: CostInput) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total implementation cost with breakdown.

        Args:
            cost_input: Cost input parameters

        Returns:
            Tuple of (total_cost, cost_breakdown)
        """
        breakdown = {}

        # Engineering costs
        engineering_cost = cost_input.engineering_days * cost_input.engineering_day_rate
        breakdown["engineering"] = engineering_cost

        # Data acquisition
        data_cost = cost_input.incremental_data_cost
        for _source, cost in cost_input.data_source_costs.items():
            data_cost += cost
        breakdown["data_acquisition"] = data_cost

        # Training & change management
        change_cost = cost_input.training_cost + cost_input.change_management_cost
        breakdown["change_management"] = change_cost

        # Infrastructure
        infra_cost = cost_input.monthly_infrastructure_cost * cost_input.infrastructure_months
        breakdown["infrastructure"] = infra_cost

        # Opportunity cost
        opportunity_cost = 0
        if cost_input.delayed_initiative_annual_value > 0 and cost_input.delay_months > 0:
            monthly_value = cost_input.delayed_initiative_annual_value / 12
            opportunity_cost = monthly_value * cost_input.delay_months
        breakdown["opportunity_cost"] = opportunity_cost

        total = sum(breakdown.values())
        return total, breakdown


# =============================================================================
# NPV Calculator
# =============================================================================


class NPVCalculator:
    """Calculate Net Present Value for multi-year initiatives."""

    DISCOUNT_RATE = 0.10  # 10% annual corporate discount rate

    def calculate_npv(
        self,
        annual_values: List[float],
        discount_rate: float = DISCOUNT_RATE,
    ) -> float:
        """
        Calculate NPV of future value stream.

        Args:
            annual_values: List of annual values [Year 1, Year 2, ...]
            discount_rate: Annual discount rate

        Returns:
            Net present value
        """
        npv = 0.0
        for t, fv in enumerate(annual_values, start=1):
            discount_factor = 1 / (1 + discount_rate) ** t
            npv += fv * discount_factor
        return npv


# =============================================================================
# Main Service
# =============================================================================


class ROICalculationService:
    """
    Complete ROI Calculation Service implementing E2I methodology.

    Provides:
    - Value driver calculations (6 types)
    - Bootstrap confidence intervals (1,000 simulations)
    - Causal attribution framework (4 levels)
    - Risk adjustment (multiplicative formula)
    - Sensitivity analysis
    - NPV for multi-year initiatives
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize ROI calculation service.

        Args:
            n_simulations: Number of Monte Carlo simulations for bootstrap CI
            seed: Random seed for reproducibility in bootstrap
        """
        self.n_simulations = n_simulations

        # Value driver calculators
        self.trx_lift = TRxLiftCalculator()
        self.patient_id = PatientIdentificationCalculator()
        self.action_rate = ActionRateCalculator()
        self.itp_lift = IntentToPrescribeCalculator()
        self.data_quality = DataQualityCalculator()
        self.drift_prevention = DriftPreventionCalculator()
        self.uplift_targeting = UpliftTargetingCalculator()

        # Other calculators
        self.bootstrap = BootstrapSimulator(n_simulations=n_simulations, seed=seed)
        self.attribution = AttributionCalculator()
        self.risk = RiskAdjustmentCalculator()
        self.cost = CostCalculator()
        self.npv = NPVCalculator()

    def calculate_value_driver(self, driver: ValueDriverInput) -> float:
        """
        Calculate value for a single value driver.

        Args:
            driver: Value driver input

        Returns:
            Dollar value for this driver
        """
        if driver.driver_type == ValueDriverType.TRX_LIFT:
            return self.trx_lift.calculate(driver.quantity)

        elif driver.driver_type == ValueDriverType.PATIENT_IDENTIFICATION:
            return self.patient_id.calculate(driver.quantity)

        elif driver.driver_type == ValueDriverType.ACTION_RATE:
            trigger_count = driver.trigger_count or 1000
            return self.action_rate.calculate(driver.quantity, trigger_count)

        elif driver.driver_type == ValueDriverType.INTENT_TO_PRESCRIBE:
            if driver.hcp_count is None:
                raise ValueError("hcp_count required for ITP calculation")
            return self.itp_lift.calculate(driver.quantity, driver.hcp_count)

        elif driver.driver_type == ValueDriverType.DATA_QUALITY:
            fp = driver.fp_reduction or 0
            fn = driver.fn_reduction or 0
            return self.data_quality.calculate(fp, fn)

        elif driver.driver_type == ValueDriverType.DRIFT_PREVENTION:
            auc_drop = driver.auc_drop_prevented or 0
            baseline = driver.baseline_model_value or 0
            return self.drift_prevention.calculate(auc_drop, baseline)

        elif driver.driver_type == ValueDriverType.UPLIFT_TARGETING:
            auuc = driver.auuc or 0.5
            efficiency = driver.targeting_efficiency or 0.5
            return self.uplift_targeting.calculate(
                auuc=auuc,
                targeting_efficiency=efficiency,
                baseline_treatment_value=driver.baseline_treatment_value,
                targeted_population_size=driver.targeted_population_size,
                qini_coefficient=driver.qini_coefficient,
            )

        else:
            raise ValueError(f"Unknown driver type: {driver.driver_type}")

    def calculate_roi(
        self,
        value_drivers: List[ValueDriverInput],
        cost_input: CostInput,
        attribution_level: AttributionLevel = AttributionLevel.PARTIAL,
        risk_assessment: Optional[RiskAssessment] = None,
        target_roi: float = 5.0,
        run_sensitivity: bool = False,
    ) -> ROIResult:
        """
        Calculate complete ROI with confidence intervals.

        Args:
            value_drivers: List of value driver inputs
            cost_input: Cost input parameters
            attribution_level: Causal attribution level
            risk_assessment: Risk assessment (defaults to all LOW)
            target_roi: Target ROI for probability calculation
            run_sensitivity: Whether to run sensitivity analysis

        Returns:
            Complete ROI result with CI, attribution, risk adjustment
        """
        if risk_assessment is None:
            risk_assessment = RiskAssessment()

        # Calculate value by driver
        value_by_driver: Dict[str, float] = {}
        total_value = 0.0

        for driver in value_drivers:
            driver_value = self.calculate_value_driver(driver)
            value_by_driver[driver.driver_type.value] = driver_value
            total_value += driver_value

        # Calculate costs
        total_cost, cost_breakdown = self.cost.calculate_total_cost(cost_input)

        # Apply attribution
        attributed_value, attribution_rate = self.attribution.apply_attribution(
            total_value, attribution_level
        )

        # Calculate base ROI
        if total_cost > 0:
            base_roi = (attributed_value - total_cost) / total_cost
        else:
            base_roi = float("inf") if attributed_value > 0 else 0.0

        # Apply risk adjustment
        risk_adjusted_roi, total_risk_adj = self.risk.apply_risk_adjustment(
            base_roi, risk_assessment
        )

        # Bootstrap simulation for CI
        roi_samples = self._simulate_roi(
            value_drivers, cost_input, attribution_rate, 1 - total_risk_adj
        )
        confidence_interval = self.bootstrap.compute_confidence_interval(roi_samples, target_roi)

        # Sensitivity analysis (optional)
        sensitivity_results = None
        if run_sensitivity and total_cost > 0:
            sensitivity_results = self._run_sensitivity_analysis(
                value_drivers, cost_input, attribution_rate, 1 - total_risk_adj
            )

        return ROIResult(
            incremental_value=total_value,
            implementation_cost=total_cost,
            base_roi=base_roi,
            risk_adjusted_roi=risk_adjusted_roi,
            confidence_interval=confidence_interval,
            attribution_level=attribution_level,
            attribution_rate=attribution_rate,
            attributed_value=attributed_value,
            risk_assessment=risk_assessment,
            total_risk_adjustment=total_risk_adj,
            value_by_driver=value_by_driver,
            cost_breakdown=cost_breakdown,
            sensitivity_results=sensitivity_results,
        )

    def calculate_npv_roi(
        self,
        value_drivers: List[ValueDriverInput],
        cost_input: CostInput,
        years: int = 3,
        attribution_level: AttributionLevel = AttributionLevel.PARTIAL,
        risk_assessment: Optional[RiskAssessment] = None,
    ) -> ROIResult:
        """
        Calculate ROI with NPV for multi-year initiatives.

        Args:
            value_drivers: List of value driver inputs (annual values)
            cost_input: Cost input parameters
            years: Number of years for value stream
            attribution_level: Causal attribution level
            risk_assessment: Risk assessment

        Returns:
            ROI result with NPV calculations
        """
        # First calculate base ROI
        result = self.calculate_roi(value_drivers, cost_input, attribution_level, risk_assessment)

        # Calculate NPV assuming constant annual value
        annual_values = [result.attributed_value] * years
        npv_value = self.npv.calculate_npv(annual_values)

        # Calculate NPV ROI
        if result.implementation_cost > 0:
            npv_roi = (npv_value - result.implementation_cost) / result.implementation_cost
        else:
            npv_roi = float("inf") if npv_value > 0 else 0.0

        # Update result with NPV
        result.npv_value = npv_value
        result.npv_roi = npv_roi

        return result

    def _simulate_roi(
        self,
        value_drivers: List[ValueDriverInput],
        cost_input: CostInput,
        attribution_rate: float,
        risk_multiplier: float,
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation for ROI distribution.

        Args:
            value_drivers: Value driver inputs
            cost_input: Cost inputs
            attribution_rate: Attribution rate to apply
            risk_multiplier: Risk multiplier (1 - total_risk_adjustment)

        Returns:
            Array of simulated ROI values
        """
        n = self.bootstrap.n_simulations

        # Simulate total value
        total_value_samples = np.zeros(n)

        for driver in value_drivers:
            mean_value = self.calculate_value_driver(driver)
            std_fraction = driver.uncertainty_std or 0.15

            # Simulate this driver's value
            driver_samples = self.bootstrap.simulate_value(mean_value, std_fraction)
            total_value_samples += driver_samples

        # Apply attribution
        attributed_samples = total_value_samples * attribution_rate

        # Simulate cost
        total_cost, _ = self.cost.calculate_total_cost(cost_input)
        cost_samples = self.bootstrap.simulate_cost(total_cost)

        # Calculate ROI for each simulation
        with np.errstate(divide="ignore", invalid="ignore"):
            roi_samples = np.where(
                cost_samples > 0,
                (attributed_samples - cost_samples) / cost_samples * risk_multiplier,
                np.where(attributed_samples > 0, np.inf, 0),
            )

        # Cap infinite values for statistics
        roi_samples = np.clip(roi_samples, -100, 1000)

        return roi_samples

    def _run_sensitivity_analysis(
        self,
        value_drivers: List[ValueDriverInput],
        cost_input: CostInput,
        attribution_rate: float,
        risk_multiplier: float,
    ) -> List[SensitivityResult]:
        """
        Run sensitivity analysis varying each parameter +/- 20%.

        Args:
            value_drivers: Value driver inputs
            cost_input: Cost inputs
            attribution_rate: Attribution rate
            risk_multiplier: Risk multiplier

        Returns:
            List of sensitivity results for tornado diagram
        """
        results = []
        total_cost, _ = self.cost.calculate_total_cost(cost_input)

        # Calculate base ROI
        base_value = sum(self.calculate_value_driver(d) for d in value_drivers)
        base_attributed = base_value * attribution_rate
        base_roi = (
            (base_attributed - total_cost) / total_cost * risk_multiplier if total_cost > 0 else 0
        )

        # Sensitivity on each value driver
        for driver in value_drivers:
            driver_value = self.calculate_value_driver(driver)
            if driver_value == 0:
                continue

            # Low scenario (-20%)
            low_total = base_value - (driver_value * 0.2)
            low_attributed = low_total * attribution_rate
            low_roi = (
                (low_attributed - total_cost) / total_cost * risk_multiplier
                if total_cost > 0
                else 0
            )

            # High scenario (+20%)
            high_total = base_value + (driver_value * 0.2)
            high_attributed = high_total * attribution_rate
            high_roi = (
                (high_attributed - total_cost) / total_cost * risk_multiplier
                if total_cost > 0
                else 0
            )

            results.append(
                SensitivityResult(
                    parameter=driver.driver_type.value,
                    base_value=driver.quantity,
                    low_value=driver.quantity * 0.8,
                    high_value=driver.quantity * 1.2,
                    roi_at_low=low_roi,
                    roi_at_base=base_roi,
                    roi_at_high=high_roi,
                    impact_range=high_roi - low_roi,
                )
            )

        # Sensitivity on cost
        if total_cost > 0:
            # Low cost scenario (-20%)
            low_cost = total_cost * 0.8
            low_cost_roi = (base_attributed - low_cost) / low_cost * risk_multiplier

            # High cost scenario (+20%)
            high_cost = total_cost * 1.2
            high_cost_roi = (base_attributed - high_cost) / high_cost * risk_multiplier

            results.append(
                SensitivityResult(
                    parameter="implementation_cost",
                    base_value=total_cost,
                    low_value=low_cost,
                    high_value=high_cost,
                    roi_at_low=low_cost_roi,
                    roi_at_base=base_roi,
                    roi_at_high=high_cost_roi,
                    impact_range=low_cost_roi - high_cost_roi,  # Note: reversed for cost
                )
            )

        # Sort by impact range (descending)
        results.sort(key=lambda r: abs(r.impact_range), reverse=True)

        return results

    def format_roi_summary(self, result: ROIResult) -> Dict[str, Any]:
        """
        Format ROI result for reporting.

        Args:
            result: ROI calculation result

        Returns:
            Dict formatted for stakeholder presentation
        """
        return {
            "summary": {
                "base_roi": f"{result.base_roi:.1f}x",
                "risk_adjusted_roi": f"{result.risk_adjusted_roi:.1f}x",
                "confidence_interval": {
                    "lower": f"{result.confidence_interval.lower_bound:.1f}x",
                    "median": f"{result.confidence_interval.median:.1f}x",
                    "upper": f"{result.confidence_interval.upper_bound:.1f}x",
                },
                "probability_positive": f"{result.confidence_interval.probability_positive * 100:.1f}%",
                "probability_target": f"{result.confidence_interval.probability_target * 100:.1f}%",
            },
            "value": {
                "incremental_value": f"${result.incremental_value:,.0f}",
                "attributed_value": f"${result.attributed_value:,.0f}",
                "attribution_level": result.attribution_level.value,
                "attribution_rate": f"{result.attribution_rate * 100:.0f}%",
            },
            "cost": {
                "total": f"${result.implementation_cost:,.0f}",
                "breakdown": {k: f"${v:,.0f}" for k, v in result.cost_breakdown.items()},
            },
            "risk": {
                "total_adjustment": f"{result.total_risk_adjustment * 100:.1f}%",
                "factors": {
                    "technical_complexity": result.risk_assessment.technical_complexity.value,
                    "organizational_change": result.risk_assessment.organizational_change.value,
                    "data_dependencies": result.risk_assessment.data_dependencies.value,
                    "timeline_uncertainty": result.risk_assessment.timeline_uncertainty.value,
                },
            },
            "npv": {
                "npv_value": f"${result.npv_value:,.0f}" if result.npv_value else "N/A",
                "npv_roi": f"{result.npv_roi:.1f}x" if result.npv_roi else "N/A",
            },
            "methodology_version": result.methodology_version,
            "calculated_at": result.calculated_at.isoformat(),
        }
