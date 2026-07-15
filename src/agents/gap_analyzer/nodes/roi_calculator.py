"""ROI Calculator Node for Gap Analyzer Agent.

This node calculates ROI estimates for closing performance gaps using
pharmaceutical-specific economics from ROICalculationService.

Implements the full ROI methodology from docs/roi_methodology.md:
- 7 Value Drivers (TRx Lift, Patient ID, Action Rate, ITP, Data Quality, Drift, Uplift)
- Bootstrap Monte Carlo simulations (1,000) for confidence intervals
- Attribution Framework (Full/Partial/Shared/Minimal)
- Risk Adjustment (Technical/Organizational/Data/Timeline factors)
- CausalML Uplift Integration for targeting optimization (Phase B6)

Reference: docs/roi_methodology.md, src/services/roi_calculation.py
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.services.roi_calculation import (
    AttributionLevel,
    CostInput,
    RiskAssessment,
    RiskLevel,
    ROICalculationService,
    ROIResult,
    TRxLiftCalculator,
    ValueDriverInput,
    ValueDriverType,
)

from ..state import (
    ConfidenceIntervalDict,
    GapAnalyzerState,
    PerformanceGap,
    ROIEstimate,
)

logger = logging.getLogger(__name__)

# Fraction of a performance gap a commercial initiative can realistically CLOSE.
# A gap of N TRx does NOT translate to N incremental TRx of value — no single
# initiative captures the entire gap. This "capture rate" is the system's own
# documented assumption: the opportunity-sizing skill the gap_analyzer loads
# (.claude/skills/gap-analysis/opportunity-sizing.md) defines
#   "Addressable Value = Opportunity Size x Revenue/Unit x Capture Probability x Discount"
#   "Capture Probability: 20-40% for most interventions" (worked example: 0.30).
# That factor was guidance for the LLM but was never wired into the deterministic
# ROI math, so value was computed at an implicit 100% capture — inflating ROI
# without bound as the gap grew. 0.30 is the skill's midpoint default; it is
# overridable via config (gap_analyzer.yaml economic_assumptions.capture_rate).
# Distinct from attribution_rate: capture = fraction of the gap CLOSED (execution
# reality); attribution = fraction of the realized closure causally OWNED by the
# initiative (confounding adjustment). They are orthogonal and both apply.
DEFAULT_CAPTURE_RATE = 0.30


def _coerce_capture_rate(rate: object) -> float:
    """Coerce ``rate`` to a valid capture fraction in (0, 1], else the default.

    Shared guard so BOTH config-loaded and constructor-injected rates are
    validated by the same rule. A caller passing 0.0, a negative, >1, or a
    non-numeric must never silently corrupt value (0.0 would zero out all ROI);
    such inputs fall back to :data:`DEFAULT_CAPTURE_RATE` with a warning.
    """
    try:
        value = float(rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("capture_rate %r not numeric; using default %s", rate, DEFAULT_CAPTURE_RATE)
        return DEFAULT_CAPTURE_RATE
    # A capture rate is a fraction in (0, 1].
    if not (0.0 < value <= 1.0):
        logger.warning(
            "capture_rate %s out of (0, 1]; using default %s", value, DEFAULT_CAPTURE_RATE
        )
        return DEFAULT_CAPTURE_RATE
    return value


# Generic $/TRx anchor (Kisqali-scale oncology brand). Brands with materially
# different per-script economics override it via gap_analyzer.yaml
# economic_assumptions.value_per_trx_by_brand — a flat rate structurally zeroes
# rare-disease brands (tiny script volumes x generic $/unit can never clear the
# intervention cost floor), which is exactly how Fabhalta ended up with a
# permanent "No gap opportunities available" on the live page.
DEFAULT_VALUE_PER_TRX = TRxLiftCalculator.VALUE_PER_TRX


def _coerce_value_per_trx(value: object) -> Optional[float]:
    """Coerce a per-brand $/TRx to a positive float, else None (drop entry).

    Same fail-soft discipline as capture_rate: a 0, negative, or non-numeric
    config value must never zero out or corrupt ROI — it falls back to the
    generic default with a warning.
    """
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("value_per_trx %r not numeric; entry ignored", value)
        return None
    if result <= 0.0:
        logger.warning("value_per_trx %s not positive; entry ignored", result)
        return None
    return result


def _normalize_value_per_trx_by_brand(raw: object) -> Dict[str, float]:
    """Lower-case brand keys and drop invalid $/TRx values (fail-soft)."""
    if not isinstance(raw, dict):
        if raw is not None:
            logger.warning("value_per_trx_by_brand %r not a mapping; ignored", raw)
        return {}
    normalized: Dict[str, float] = {}
    for brand, value in raw.items():
        coerced = _coerce_value_per_trx(value)
        if coerced is not None:
            normalized[str(brand).lower()] = coerced
    return normalized


def _load_value_per_trx_by_brand(config_path: Optional[str] = None) -> Dict[str, float]:
    """Load per-brand $/TRx overrides from gap_analyzer.yaml, default {}.

    Reads ``gap_analyzer.economic_assumptions.value_per_trx_by_brand``. Any
    read/parse failure falls back to an empty map (fail-soft — every brand then
    uses :data:`DEFAULT_VALUE_PER_TRX`).
    """
    if config_path is None:
        config_path = "config/agents/gap_analyzer.yaml"
    try:
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        raw = (
            cfg.get("gap_analyzer", {})
            .get("economic_assumptions", {})
            .get("value_per_trx_by_brand", {})
        )
        return _normalize_value_per_trx_by_brand(raw)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to load value_per_trx_by_brand (%s); using defaults", e)
        return {}


def _load_capture_rate(config_path: Optional[str] = None) -> float:
    """Load the gap-closure capture rate from gap_analyzer.yaml, default 0.30.

    Reads ``gap_analyzer.economic_assumptions.capture_rate``. Any read/parse
    failure falls back to :data:`DEFAULT_CAPTURE_RATE` (fail-soft — a missing
    config must never zero out value or crash the ROI node).
    """
    if config_path is None:
        config_path = "config/agents/gap_analyzer.yaml"
    try:
        path = Path(config_path)
        if not path.exists():
            return DEFAULT_CAPTURE_RATE
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        rate = (
            cfg.get("gap_analyzer", {})
            .get("economic_assumptions", {})
            .get("capture_rate", DEFAULT_CAPTURE_RATE)
        )
        return _coerce_capture_rate(rate)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Failed to load capture_rate (%s); using default %s", e, DEFAULT_CAPTURE_RATE
        )
        return DEFAULT_CAPTURE_RATE


class ROICalculatorNode:
    """Calculate ROI estimates for performance gaps.

    Uses pharmaceutical-specific economics from ROICalculationService with:
    - 6 value drivers at pharma-specific unit rates
    - Bootstrap confidence intervals (1,000 simulations)
    - Attribution framework (full/partial/shared/minimal)
    - Risk adjustment (4 factors)
    """

    # Mapping from KPI metric to primary value driver
    METRIC_TO_DRIVER: Dict[str, ValueDriverType] = {
        "trx": ValueDriverType.TRX_LIFT,
        "nrx": ValueDriverType.TRX_LIFT,
        "patient_count": ValueDriverType.PATIENT_IDENTIFICATION,
        "patient_identification": ValueDriverType.PATIENT_IDENTIFICATION,
        "trigger_acceptance": ValueDriverType.ACTION_RATE,
        "trigger_count": ValueDriverType.ACTION_RATE,
        "hcp_engagement_score": ValueDriverType.INTENT_TO_PRESCRIBE,
        "itp": ValueDriverType.INTENT_TO_PRESCRIBE,
        "conversion_rate": ValueDriverType.INTENT_TO_PRESCRIBE,
        "data_quality": ValueDriverType.DATA_QUALITY,
        "model_accuracy": ValueDriverType.DRIFT_PREVENTION,
        "market_share": ValueDriverType.TRX_LIFT,  # Translates to TRx impact
        "targeting_efficiency": ValueDriverType.UPLIFT_TARGETING,  # From uplift models
        "uplift_score": ValueDriverType.UPLIFT_TARGETING,
    }

    # Default cost category for gap initiatives
    DEFAULT_COST_CATEGORY = "algorithm_optimization"

    # Engineering cost per day
    ENGINEERING_RATE = 2500.0  # USD per day

    def __init__(
        self,
        roi_service: Optional[ROICalculationService] = None,
        use_bootstrap: bool = True,
        n_simulations: int = 1000,
        capture_rate: Optional[float] = None,
        value_per_trx_by_brand: Optional[Dict[str, float]] = None,
    ):
        """Initialize ROI calculator with service.

        Args:
            roi_service: Injected ROICalculationService (or created if None)
            use_bootstrap: Whether to compute bootstrap confidence intervals
            n_simulations: Number of Monte Carlo simulations for bootstrap
            capture_rate: Fraction of a gap an initiative realistically closes
                (see :data:`DEFAULT_CAPTURE_RATE`). When None, loaded from
                gap_analyzer.yaml (default 0.30).
            value_per_trx_by_brand: Per-brand $ value of one incremental
                TRx-equivalent unit (case-insensitive keys). When None, loaded
                from gap_analyzer.yaml. Brands absent from the map use
                :data:`DEFAULT_VALUE_PER_TRX`.
        """
        self.roi_service = roi_service or ROICalculationService(n_simulations=n_simulations)
        self.use_bootstrap = use_bootstrap
        self.capture_rate = (
            _coerce_capture_rate(capture_rate) if capture_rate is not None else _load_capture_rate()
        )
        self.value_per_trx_by_brand = (
            _normalize_value_per_trx_by_brand(value_per_trx_by_brand)
            if value_per_trx_by_brand is not None
            else _load_value_per_trx_by_brand()
        )

    def _resolve_value_per_trx(self, brand: Optional[str]) -> float:
        """Brand-scoped $/TRx (case-insensitive), else the generic default."""
        return self.value_per_trx_by_brand.get((brand or "").lower(), DEFAULT_VALUE_PER_TRX)

    async def execute(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Execute ROI calculation workflow.

        Args:
            state: Current gap analyzer state with gaps_detected

        Returns:
            Updated state with roi_estimates, total_addressable_value, roi_latency_ms
        """
        start_time = time.time()

        try:
            gaps_detected = state.get("gaps_detected", [])

            if not gaps_detected:
                # F2 fail-closed: "no gaps" can mean either (a) the analysis ran fine
                # and genuinely found nothing, or (b) an upstream node (e.g. gap_detector)
                # already FAILED and left gaps_detected empty. Distinguish via state["errors"]:
                # if a terminal error was accumulated, propagate FAILED rather than the
                # normal 'prioritizing' hand-off, so the failure is not laundered downstream.
                if state.get("errors"):
                    return {
                        "roi_estimates": [],
                        "total_addressable_value": 0.0,
                        "roi_latency_ms": 0,
                        "status": "failed",
                    }
                return {
                    "roi_estimates": [],
                    "total_addressable_value": 0.0,
                    "roi_latency_ms": 0,
                    "warnings": ["No gaps detected for ROI calculation"],
                    "status": "prioritizing",
                }

            # Extract uplift context if available (from heterogeneous_optimizer)
            uplift_context = self._extract_uplift_context(state)

            # Brand-scoped $/TRx — resolved once per run (same idiom as the
            # competitor-density surface below).
            value_per_trx = self._resolve_value_per_trx(state.get("brand"))

            # Calculate ROI for each gap using ROICalculationService
            roi_estimates: List[ROIEstimate] = []

            for gap in gaps_detected:
                roi_estimate = self._calculate_roi(
                    gap, uplift_context=uplift_context, value_per_trx=value_per_trx
                )
                roi_estimates.append(roi_estimate)

            # Surface-only competitor density for this brand (curated, no network,
            # case-insensitive; fail-open). INFORMATIONAL — does NOT change the ROI
            # value or the prioritizer ranking (which sorts on risk_adjusted_roi).
            density = self._competitor_density(state.get("brand"))
            for est in roi_estimates:
                est["competitor_products_count"] = density["competitor_products_count"]
                est["competitor_density_label"] = density["competitor_density_label"]
                est["competitor_drug_names"] = density["competitor_drug_names"]

            # Calculate total addressable value (attributed value)
            total_addressable_value = sum(est["estimated_revenue_impact"] for est in roi_estimates)

            roi_latency_ms = int((time.time() - start_time) * 1000)

            uplift_msg = " (with uplift targeting boost)" if uplift_context else ""
            logger.info(
                f"ROI calculated for {len(roi_estimates)} gaps{uplift_msg}, "
                f"total addressable value: ${total_addressable_value:,.0f}"
            )

            return {
                "roi_estimates": roi_estimates,
                "total_addressable_value": total_addressable_value,
                "roi_latency_ms": roi_latency_ms,
                "status": "prioritizing",
            }

        except Exception as e:
            logger.error(f"ROI calculation failed: {e}")
            roi_latency_ms = int((time.time() - start_time) * 1000)
            # F2: returning only the NEW error is correct -- state["errors"] uses an
            # additive reducer (operator.add), so this is MERGED onto any upstream errors
            # rather than overwriting them.
            return {
                "errors": [
                    {
                        "node": "roi_calculator",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                ],
                "roi_latency_ms": roi_latency_ms,
                "status": "failed",
            }

    @staticmethod
    def _competitor_density(brand: Optional[str]) -> Dict[str, Any]:
        """Curated competitor count + saturation label + names for a brand (no
        network — the curated SSOT; case-insensitive brand match). Fail-open to
        count 0 / "unknown". INFORMATIONAL only: surfaced on each strategic bet, it
        NEVER alters the ROI value or the prioritizer ranking."""
        empty = {
            "competitor_products_count": 0,
            "competitor_density_label": "unknown",
            "competitor_drug_names": [],
        }
        if not brand:
            return empty
        try:
            from src.services.clinical_context.brand_map import (
                BRAND_CLINICAL_MAP,
                resolve_brand_profile,
            )
            from src.services.clinical_context.providers import CuratedCompetitorProvider

            key = next((k for k in BRAND_CLINICAL_MAP if k.lower() == brand.lower()), None)
            if key is None:
                return empty
            frag = CuratedCompetitorProvider().enrich(resolve_brand_profile(key))
            n = frag.count
            if n == 0:
                label = "unknown"
            elif n <= 2:
                label = "limited"
            elif n <= 5:
                label = "moderate"
            else:
                label = "crowded"
            return {
                "competitor_products_count": n,
                "competitor_density_label": label,
                "competitor_drug_names": list(frag.competitors),
            }
        except Exception as exc:  # noqa: BLE001 — best-effort; never alters the ROI
            logger.debug("roi_calculator: competitor density unavailable for %s: %s", brand, exc)
            return empty

    def _extract_uplift_context(self, state: GapAnalyzerState) -> Optional[Dict[str, Any]]:
        """Extract uplift context from state if available.

        Args:
            state: Gap analyzer state potentially containing uplift data

        Returns:
            Uplift context dict or None if not available
        """
        auuc = state.get("uplift_auuc")
        qini = state.get("uplift_qini")
        efficiency = state.get("uplift_targeting_efficiency")

        if auuc is not None or efficiency is not None:
            return {
                "auuc": auuc or 0.5,
                "qini_coefficient": qini,
                "targeting_efficiency": efficiency or 0.5,
                "uplift_by_segment": state.get("uplift_by_segment"),
            }
        return None

    def _calculate_roi(
        self,
        gap: PerformanceGap,
        uplift_context: Optional[Dict[str, Any]] = None,
        value_per_trx: Optional[float] = None,
    ) -> ROIEstimate:
        """Calculate ROI estimate for a single gap using ROICalculationService.

        Implements full ROI methodology:
        1. Map metric to value driver
        2. Estimate intervention costs
        3. Determine attribution level from gap type
        4. Apply risk adjustment
        5. Run bootstrap simulations for confidence interval
        6. Add uplift targeting value if context available (Phase B6)

        Args:
            gap: Performance gap to analyze
            uplift_context: Optional uplift context from heterogeneous_optimizer
            value_per_trx: Brand-scoped $ per incremental TRx-equivalent unit
                (see :data:`DEFAULT_VALUE_PER_TRX`); None uses the generic default

        Returns:
            ROI estimate with confidence interval, attribution, risk adjustment
        """
        if value_per_trx is None:
            value_per_trx = DEFAULT_VALUE_PER_TRX
        metric = gap["metric"]
        # Raw gap magnitude drives the COST and RISK of the initiative (closing a
        # gap means tackling its full size). The VALUE is only the fraction we
        # realistically close -> captured_units = gap_size * capture_rate. This
        # asymmetry is conservative by design: full initiative scope/cost, partial
        # expected realized value. (Previously value used the full gap_size, an
        # implicit 100% capture that inflated ROI without bound as gaps grew.)
        gap_size = abs(gap["gap_size"])
        # market_share is mapped to TRX_LIFT but its gap_size is in SHARE POINTS,
        # not scripts — feed it raw into $/TRx and a 0.05-point gap is "worth"
        # $42.50, below any cost floor, so every share gap was structurally
        # suppressed for every brand. Convert to TRx-equivalents first; the
        # translated units then drive value, cost, and risk consistently (the
        # size thresholds in those methods are calibrated to script volumes).
        share_note: Optional[str] = None
        if metric == "market_share":
            translated = self._market_share_to_trx_equivalents(gap)
            if translated is None:
                # Fail CLOSED: without segment TRx context a share gap cannot be
                # valued in dollars. $0 (suppressed, with an explicit note) is
                # honest; pp x $/TRx would be a plausible-looking fake value.
                gap_size = 0.0
                share_note = (
                    "market_share gap NOT valued: no segment TRx context available "
                    "to convert share points to script volume (valued at $0 rather "
                    "than mis-reading share points as scripts)"
                )
            else:
                gap_size = translated
                share_note = (
                    f"market_share gap converted to ~{translated:,.0f} TRx-equivalents "
                    "(relative share growth x segment TRx; market size assumed constant)"
                )
        captured_units = gap_size * self.capture_rate
        gap_type = gap["gap_type"]

        # Map metric to value driver
        driver_type = self._get_value_driver(metric)

        # Create value driver input (value sized on the CAPTURED fraction).
        # Pass BOTH the captured improvement and the raw gap: capture applies to
        # the value-bearing improvement, NOT to base/population fields (else the
        # haircut compounds for non-linear drivers — see _create_value_driver_input).
        value_driver = self._create_value_driver_input(
            driver_type, captured_units, gap_size, gap, value_per_trx
        )

        # Build list of value drivers (may include uplift)
        value_drivers = [value_driver]

        # Add uplift targeting value driver if context available
        if uplift_context:
            uplift_driver = self._create_uplift_value_driver(
                gap, uplift_context, captured_units, value_per_trx
            )
            if uplift_driver:
                value_drivers.append(uplift_driver)

        # Estimate costs for closing the gap (full gap scope, NOT capture-adjusted)
        cost_input = self._estimate_intervention_costs(metric, gap_size)

        # Determine attribution level from gap type
        attribution = self._determine_attribution(gap_type)

        # Assess risks based on gap characteristics (size in effective units —
        # for market_share that is the translated script volume, so the
        # script-calibrated size thresholds keep their meaning)
        risk_assessment = self._assess_risks(gap, effective_gap_size=gap_size)

        # Calculate ROI using the full service
        roi_result: ROIResult = self.roi_service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            attribution_level=attribution,
            risk_assessment=risk_assessment,
        )

        # Build confidence interval dict
        confidence_interval: Optional[ConfidenceIntervalDict] = None
        if roi_result.confidence_interval:
            ci = roi_result.confidence_interval
            confidence_interval = {
                "lower_bound": ci.lower_bound,
                "median": ci.median,
                "upper_bound": ci.upper_bound,
                "probability_positive": ci.probability_positive,
                "probability_target": ci.probability_target,
            }

        # Build assumptions list
        assumptions = self._build_assumptions(
            metric, driver_type, attribution, risk_assessment, value_per_trx
        )
        if share_note is not None:
            assumptions.append(share_note)

        # Legacy confidence (use probability_positive if available)
        legacy_confidence = (
            confidence_interval["probability_positive"]
            if confidence_interval
            else self._calculate_legacy_confidence(gap)
        )

        # Convert ROIResult to ROIEstimate TypedDict
        roi_estimate: ROIEstimate = {
            "gap_id": gap["gap_id"],
            "estimated_revenue_impact": roi_result.attributed_value,
            "estimated_cost_to_close": roi_result.implementation_cost,
            "expected_roi": roi_result.base_roi,
            "risk_adjusted_roi": roi_result.risk_adjusted_roi,
            "payback_period_months": self._calculate_payback_months(
                roi_result.attributed_value, roi_result.implementation_cost
            ),
            "confidence_interval": confidence_interval,
            "attribution_level": attribution.value,
            "attribution_rate": roi_result.attribution_rate,
            "total_risk_adjustment": roi_result.total_risk_adjustment,
            "value_by_driver": roi_result.value_by_driver,
            "confidence": legacy_confidence,
            "assumptions": assumptions,
        }

        return roi_estimate

    @staticmethod
    def _market_share_to_trx_equivalents(gap: PerformanceGap) -> Optional[float]:
        """Convert a market_share gap to incremental TRx-equivalent units.

        Closing a share gap S -> T on a constant market of M scripts yields
        (T - S)/100 * M incremental scripts, and M = segment_trx / (S/100), so
        incremental scripts = |gap_size| / current_share * segment_trx — the
        RELATIVE share growth times the brand's current segment volume. The
        ratio form is deliberately unit-scale invariant: business_metrics
        stores share as a 0-1 fraction while the tier0 fallback derives 0-100,
        and both cancel out identically.

        Returns None when the gap carries no usable context (no positive
        ``segment_trx`` from the detector, or current share <= 0 so relative
        growth is undefined) — the caller fails closed to $0.
        """
        segment_trx = gap.get("segment_trx")
        current_share = gap["current_value"]
        if segment_trx is None or segment_trx <= 0.0 or current_share <= 0.0:
            return None
        return abs(gap["gap_size"]) / current_share * segment_trx

    def _get_value_driver(self, metric: str) -> ValueDriverType:
        """Map KPI metric to primary value driver type.

        Args:
            metric: KPI metric name

        Returns:
            Corresponding value driver type
        """
        return self.METRIC_TO_DRIVER.get(metric, ValueDriverType.TRX_LIFT)

    def _create_value_driver_input(
        self,
        driver_type: ValueDriverType,
        captured_units: float,
        raw_gap_size: float,
        gap: PerformanceGap,
        value_per_trx: Optional[float] = None,
    ) -> ValueDriverInput:
        """Create value driver input for ROI calculation.

        Maps gap size to the appropriate driver quantity based on type.

        Capture-rate discipline: the haircut (``captured_units = raw_gap_size *
        capture_rate``) must be applied to each driver's value EXACTLY ONCE — on
        the value-bearing IMPROVEMENT magnitude, never on a base/population
        field. Several drivers are non-linear: ACTION_RATE multiplies the pp
        improvement (``quantity``) by ``trigger_count``, and INTENT_TO_PRESCRIBE
        multiplies the pp improvement by ``hcp_count``. Those count fields are
        the EXISTING market base (trigger volume / HCP panel), not part of the
        gap being closed — so they derive from the RAW gap, NOT the captured
        value. Feeding ``captured_units`` into both factors would square the
        haircut (0.30 -> 0.09) and over-suppress those drivers. Conversely,
        DATA_QUALITY (value flows through fp/fn, ``quantity`` ignored) and
        DRIFT_PREVENTION (value flows through ``baseline_model_value``) carry the
        capture on those derived fields so each still gets exactly one haircut.

        Args:
            driver_type: Type of value driver
            captured_units: Realized improvement = raw_gap_size * capture_rate
            raw_gap_size: Absolute gap size BEFORE the capture haircut, used for
                base/population fields that must not be capture-discounted
            gap: Full gap details for context

        Returns:
            ValueDriverInput for ROI service
        """
        return ValueDriverInput(
            driver_type=driver_type,
            # Value-bearing IMPROVEMENT (carries the single capture haircut).
            # TRX_LIFT / PATIENT_IDENTIFICATION / ACTION_RATE / INTENT_TO_PRESCRIBE
            # read value off `quantity` (the latter two as the pp improvement).
            quantity=captured_units,
            # Brand-scoped $/unit — consumed by the TRX_LIFT calculator only.
            unit_value=value_per_trx if driver_type == ValueDriverType.TRX_LIFT else None,
            # Base/population fields = existing market size, NOT the gap we close
            # -> derive from the RAW gap so capture is not applied a second time.
            hcp_count=(
                int(raw_gap_size / 10)
                if driver_type == ValueDriverType.INTENT_TO_PRESCRIBE
                else None
            ),
            trigger_count=int(raw_gap_size) if driver_type == ValueDriverType.ACTION_RATE else None,
            # DATA_QUALITY value flows through fp/fn (quantity ignored) -> these
            # carry the capture haircut so the driver still gets exactly one.
            fp_reduction=(
                int(captured_units * 0.3) if driver_type == ValueDriverType.DATA_QUALITY else None
            ),
            fn_reduction=(
                int(captured_units * 0.7) if driver_type == ValueDriverType.DATA_QUALITY else None
            ),
            auc_drop_prevented=0.02 if driver_type == ValueDriverType.DRIFT_PREVENTION else None,
            # DRIFT_PREVENTION value flows through baseline_model_value (quantity
            # ignored) -> capture applied here for the single haircut.
            baseline_model_value=(
                captured_units * 850 if driver_type == ValueDriverType.DRIFT_PREVENTION else None
            ),
        )

    def _create_uplift_value_driver(
        self,
        gap: PerformanceGap,
        uplift_context: Dict[str, Any],
        captured_units: float,
        value_per_trx: Optional[float] = None,
    ) -> Optional[ValueDriverInput]:
        """Create uplift targeting value driver from uplift context.

        This adds incremental value from using CausalML uplift models for
        optimized targeting when closing performance gaps.

        Args:
            gap: Performance gap being analyzed
            uplift_context: Uplift context with AUUC, Qini, efficiency
            captured_units: Capture-adjusted gap units (gap_size * capture_rate);
                value is sized on the fraction realistically closed, consistent
                with the base value driver.

        Returns:
            ValueDriverInput for uplift targeting, or None if not applicable
        """
        # Only add uplift value for metrics that benefit from targeting
        targeting_metrics = {
            "trx",
            "nrx",
            "patient_count",
            "patient_identification",
            "trigger_acceptance",
            "conversion_rate",
            "hcp_engagement_score",
        }

        if gap["metric"] not in targeting_metrics:
            return None

        # Size uplift value on the captured fraction, not the full gap.
        gap_size = captured_units

        # Get segment-specific uplift if available
        if uplift_context.get("uplift_by_segment"):
            segment_key = gap["segment_value"]
            segment_data = uplift_context["uplift_by_segment"].get(segment_key, [])
            if segment_data:
                # Get average uplift score for segment
                scores = [s.get("mean_uplift_score", 0) for s in segment_data]
                sum(scores) / len(scores) if scores else None

        # Calculate baseline treatment value from gap size at the brand-scoped
        # $/unit (TRx equivalent) — consistent with the base TRX_LIFT driver.
        baseline_value = gap_size * (
            value_per_trx if value_per_trx is not None else DEFAULT_VALUE_PER_TRX
        )

        return ValueDriverInput(
            driver_type=ValueDriverType.UPLIFT_TARGETING,
            quantity=gap_size,
            auuc=uplift_context.get("auuc", 0.5),
            qini_coefficient=uplift_context.get("qini_coefficient"),
            targeting_efficiency=uplift_context.get("targeting_efficiency", 0.5),
            baseline_treatment_value=baseline_value,
            targeted_population_size=int(gap_size) if gap_size > 0 else 100,
        )

    def _estimate_intervention_costs(
        self,
        metric: str,
        gap_size: float,
    ) -> CostInput:
        """Estimate intervention costs for closing a gap.

        Cost components:
        - Engineering effort (based on gap size)
        - Data acquisition (if data quality or patient ID)
        - Change management (if organizational change needed)

        Args:
            metric: KPI metric
            gap_size: Size of gap to close

        Returns:
            CostInput for ROI calculation
        """
        # Engineering cost (scaled by gap complexity)
        engineering_days = self._estimate_engineering_days(metric, gap_size)

        # Data acquisition (for patient ID and data quality metrics)
        data_source_costs: Dict[str, float] = {}
        incremental_data_cost = 0.0
        if metric in ["patient_identification", "patient_count", "data_quality"]:
            # ~$100 per patient identified, attributed to the IQVIA APLD source.
            # Record it ONCE on the named source. Previously the same value was
            # ALSO assigned to `incremental_data_cost`, and
            # CostCalculator.calculate_total_cost sums BOTH incremental_data_cost
            # and every data_source_costs entry -> the data cost was double-counted
            # for patient/data-quality metrics. `incremental_data_cost` stays 0.
            data_source_costs["IQVIA APLD"] = gap_size * 100

        # Change management (for org-level changes)
        change_management_cost = 0.0
        if gap_size > 100 or metric in ["conversion_rate", "market_share"]:
            change_management_cost = min(50000, gap_size * 200)  # Cap at $50k

        return CostInput(
            engineering_days=engineering_days,
            engineering_day_rate=self.ENGINEERING_RATE,
            data_source_costs=data_source_costs,
            incremental_data_cost=incremental_data_cost,
            change_management_cost=change_management_cost,
        )

    def _estimate_engineering_days(self, metric: str, gap_size: float) -> float:
        """Estimate engineering days required to close a gap.

        Args:
            metric: KPI metric
            gap_size: Size of gap

        Returns:
            Estimated engineering days
        """
        # Base days by metric type
        base_days = {
            "trx": 5,
            "nrx": 5,
            "patient_count": 10,
            "patient_identification": 10,
            "trigger_acceptance": 8,
            "conversion_rate": 15,
            "hcp_engagement_score": 8,
            "data_quality": 12,
            "model_accuracy": 20,
            "market_share": 10,
        }

        base = base_days.get(metric, 10)

        # Scale by gap size (larger gaps = more effort)
        if gap_size > 1000:
            scale = 2.0
        elif gap_size > 100:
            scale = 1.5
        elif gap_size > 10:
            scale = 1.0
        else:
            scale = 0.5

        return base * scale

    def _determine_attribution(self, gap_type: str) -> AttributionLevel:
        """Determine attribution level from gap type.

        Attribution reflects how much of the gap closure can be CAUSALLY
        attributed to the initiative (per docs/roi_methodology.md attribution
        framework):
        - vs_target: Partial (65%) - a vs-target gap is an OBSERVATION (current
          below a stored target) with no causal validation. The methodology
          reserves FULL (100%) for "RCT validates effect, no confounding"; a bare
          target comparison does not meet that bar, so it maps to PARTIAL
          ("observational causal inference"). FULL would only be justified with a
          validated causal signal (CATE/uplift) attached to the gap.
        - vs_benchmark: Partial (65%) - peer comparison has some noise
        - vs_potential: Shared (35%) - multiple factors for top decile
        - temporal: Minimal (10%) - many confounders over time

        Args:
            gap_type: Type of gap comparison

        Returns:
            Attribution level
        """
        attribution_mapping = {
            # vs_target downgraded FULL -> PARTIAL: an unvalidated target gap is
            # not RCT-grade evidence (see docstring). Together with the missing
            # capture rate, FULL attribution was a primary driver of inflated ROI.
            "vs_target": AttributionLevel.PARTIAL,
            "vs_benchmark": AttributionLevel.PARTIAL,
            "vs_potential": AttributionLevel.SHARED,
            "temporal": AttributionLevel.MINIMAL,
        }
        return attribution_mapping.get(gap_type, AttributionLevel.PARTIAL)

    def _assess_risks(
        self, gap: PerformanceGap, effective_gap_size: Optional[float] = None
    ) -> RiskAssessment:
        """Assess risk factors for closing a gap.

        Risk factors:
        - Technical complexity: Based on metric type
        - Organizational change: Based on gap size
        - Data dependencies: Based on metric data requirements
        - Timeline uncertainty: Based on gap percentage

        Args:
            gap: Performance gap
            effective_gap_size: Gap size in effective (script-equivalent) units
                when the raw gap_size is in a different unit — market_share gaps
                are share points, and the >100/>500 size thresholds below are
                calibrated to script volumes. None falls back to the raw size.

        Returns:
            Risk assessment for ROI adjustment
        """
        metric = gap["metric"]
        gap_size = effective_gap_size if effective_gap_size is not None else abs(gap["gap_size"])
        gap_pct = abs(gap["gap_percentage"])

        # Technical complexity
        complex_metrics = ["model_accuracy", "data_quality", "conversion_rate"]
        if metric in complex_metrics:
            technical = RiskLevel.HIGH
        elif metric in ["patient_identification", "hcp_engagement_score"]:
            technical = RiskLevel.MEDIUM
        else:
            technical = RiskLevel.LOW

        # Organizational change
        if gap_size > 500:
            organizational = RiskLevel.HIGH
        elif gap_size > 100:
            organizational = RiskLevel.MEDIUM
        else:
            organizational = RiskLevel.LOW

        # Data dependencies
        data_heavy = ["patient_identification", "patient_count", "data_quality"]
        if metric in data_heavy:
            data_deps = RiskLevel.HIGH
        elif metric in ["trigger_acceptance", "model_accuracy"]:
            data_deps = RiskLevel.MEDIUM
        else:
            data_deps = RiskLevel.LOW

        # Timeline uncertainty
        if gap_pct > 50:
            timeline = RiskLevel.HIGH
        elif gap_pct > 20:
            timeline = RiskLevel.MEDIUM
        else:
            timeline = RiskLevel.LOW

        return RiskAssessment(
            technical_complexity=technical,
            organizational_change=organizational,
            data_dependencies=data_deps,
            timeline_uncertainty=timeline,
        )

    def _calculate_payback_months(
        self,
        revenue_impact: float,
        cost: float,
    ) -> int:
        """Calculate payback period in months.

        Args:
            revenue_impact: Annual revenue impact
            cost: One-time implementation cost

        Returns:
            Payback period in months (1-24)
        """
        if revenue_impact <= 0:
            return 24

        monthly_revenue = revenue_impact / 12
        if monthly_revenue <= 0:
            return 24

        months = int(cost / monthly_revenue)
        return max(1, min(months, 24))  # Clamp to 1-24

    def _build_assumptions(
        self,
        metric: str,
        driver_type: ValueDriverType,
        attribution: AttributionLevel,
        risk: RiskAssessment,
        value_per_trx: Optional[float] = None,
    ) -> List[str]:
        """Build list of assumptions for transparency.

        Args:
            metric: KPI metric
            driver_type: Value driver used
            attribution: Attribution level
            risk: Risk assessment
            value_per_trx: Brand-resolved $/TRx actually used for TRX_LIFT —
                the displayed assumption must reflect it, not a hardcoded $850

        Returns:
            List of assumption statements
        """
        # Get unit value from service
        unit_values = {
            ValueDriverType.TRX_LIFT: (
                f"${(value_per_trx if value_per_trx is not None else DEFAULT_VALUE_PER_TRX):,.0f}/TRx"
            ),
            ValueDriverType.PATIENT_IDENTIFICATION: "$1,200/patient",
            ValueDriverType.ACTION_RATE: "$45/pp/1000 triggers",
            ValueDriverType.INTENT_TO_PRESCRIBE: "$320/HCP/pp",
            ValueDriverType.DATA_QUALITY: "$200/FP, $650/FN",
            ValueDriverType.DRIFT_PREVENTION: "2x value multiplier",
        }

        assumptions = [
            f"Value driver: {driver_type.value}",
            f"Unit value: {unit_values.get(driver_type, 'N/A')}",
            f"Attribution level: {attribution.value} ({self._get_attribution_pct(attribution)})",
            f"Risk adjustment applied based on {self._summarize_risks(risk)}",
            "Bootstrap CI from 1,000 Monte Carlo simulations",
        ]

        # Add metric-specific assumptions
        if metric in ["trx", "nrx"]:
            assumptions.append("Market conditions assumed stable")
        elif metric == "market_share":
            assumptions.append("Market size assumed constant")
        elif metric == "conversion_rate":
            assumptions.append("Patient journey optimization feasible")

        return assumptions

    def _get_attribution_pct(self, level: AttributionLevel) -> str:
        """Get attribution percentage string."""
        pcts = {
            AttributionLevel.FULL: "100%",
            AttributionLevel.PARTIAL: "65%",
            AttributionLevel.SHARED: "35%",
            AttributionLevel.MINIMAL: "10%",
        }
        return pcts.get(level, "N/A")

    def _summarize_risks(self, risk: RiskAssessment) -> str:
        """Summarize risk factors."""
        high_count = sum(
            1
            for r in [
                risk.technical_complexity,
                risk.organizational_change,
                risk.data_dependencies,
                risk.timeline_uncertainty,
            ]
            if r == RiskLevel.HIGH
        )
        if high_count >= 2:
            return "multiple high-risk factors"
        elif high_count == 1:
            return "one high-risk factor"
        else:
            return "low-to-medium risk factors"

    def _calculate_legacy_confidence(self, gap: PerformanceGap) -> float:
        """Calculate legacy confidence score for backwards compatibility.

        Args:
            gap: Performance gap

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.7  # Base confidence

        # Gap size factor
        gap_size = abs(gap["gap_size"])
        if gap_size > 100:
            confidence += 0.1
        elif gap_size < 10:
            confidence -= 0.1

        # Gap percentage factor
        gap_pct = abs(gap["gap_percentage"])
        if 10 <= gap_pct <= 50:
            confidence += 0.1
        elif gap_pct > 100:
            confidence -= 0.2

        # Gap type factor
        gap_type_confidence = {
            "vs_target": 0.1,
            "vs_benchmark": 0.05,
            "vs_potential": 0.0,
            "temporal": -0.05,
        }
        confidence += gap_type_confidence.get(gap["gap_type"], 0.0)

        return max(0.0, min(1.0, confidence))
