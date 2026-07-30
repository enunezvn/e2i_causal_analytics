"""Synthetic agent_activities substrate (#1355).

agent_activities was dropped in the v3 -> src/ml/synthetic DGP migration: the
legacy generator (src/ml/data_generator.py:_generate_agent_activities) produced
the table for every agent, but the new pipeline carried zero references, so the
chat agent-analysis tool (_query_agent_analysis), the
business_impact_roi_agent_activities KPI (migration 044: AVG(roi_estimate) over
NOW()-30d) and the RAG fulltext index all ran on an empty table.

This generator restores the substrate with is_synthetic=true provenance and —
unlike the legacy random-only rows — CURATED analysis summaries CONSISTENT with
the COMM-ARMS causal ground truth:

* heterogeneous_optimizer rows mirror the brand-scaled CATE constants
  (dgp/treatment_arm.py ARM_REGISTRY + brand_scaled_cate — the SSOT the DGP
  actually plants, never re-hardcoded here);
* causal_impact rows mirror the causal_paths commercial-arm registry edges
  (scp_a*), reproducing the SAME content-addressed display effect per
  (brand, arm, outcome) so the two tables tell one story;
* gap_analyzer rows carry bounded roi_estimate values — the substrate the
  business-impact ROI KPI reads.

The remaining legacy-roster agents (orchestrator, drift_monitor,
experiment_designer, health_score, prediction_synthesizer, resource_optimizer,
explainer, feedback_learner) get generic tiered activity rows mirroring the
legacy shape (analysis_results / confidence_level / recommendations).

Ids are content-addressed under the run's ``id_prefix``
(``{prefix}aa_{sha1[:12]}``, <= 30 chars for varchar(30)) so a re-run of the
same cohort upserts idempotently, and the weekly frontier-append cohorts
(prefix ``w<yy><ww>``) can never collide with the base ``scv`` namespace.
``created_at`` equals ``activity_timestamp`` so a frame is bit-reproducible
from (seed, config) alone.
"""

import hashlib
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..config import Brand
from ..dgp.treatment_arm import _BRAND_CATE_SCALE, ARM_REGISTRY, brand_scaled_cate
from .base import BaseGenerator
from .causal_paths_generator import _COMM_ARM_EDGES, _commercial_edge_rng

# Legacy v3 roster + tier map (src/ml/data_generator.py AGENT_NAMES/AGENT_TIERS).
# Values MUST stay inside the DB agent_tier_type enum: coordination /
# causal_analytics / monitoring / ml_predictions / self_improvement.
LEGACY_AGENT_TIERS: Dict[str, str] = {
    "orchestrator": "coordination",
    "causal_impact": "causal_analytics",
    "gap_analyzer": "causal_analytics",
    "heterogeneous_optimizer": "causal_analytics",
    "drift_monitor": "monitoring",
    "experiment_designer": "monitoring",
    "health_score": "monitoring",
    "prediction_synthesizer": "ml_predictions",
    "resource_optimizer": "ml_predictions",
    "explainer": "self_improvement",
    "feedback_learner": "self_improvement",
}

# Agents whose rows are the generic legacy-shaped block (the three curated
# causal_analytics agents get their dedicated blocks below).
_GENERIC_AGENTS: Tuple[str, ...] = tuple(
    a
    for a in LEGACY_AGENT_TIERS
    if a not in ("heterogeneous_optimizer", "causal_impact", "gap_analyzer")
)

_BRANDS: Tuple[str, ...] = ("Remibrutinib", "Kisqali", "Fabhalta")
_REGIONS: Tuple[str, ...] = ("northeast", "south", "midwest", "west")
_WORKSTREAMS: Tuple[str, ...] = ("WS1", "WS2", "WS3")
_TIME_WINDOWS: Tuple[str, ...] = ("1h", "6h", "24h", "7d")


def _activity_id(id_prefix: str, key: str) -> str:
    """Content-addressed id namespaced by the cohort prefix.

    ``{prefix}aa_{sha1[:12]}``: base prefix 'scv' -> 18 chars; weekly frontier
    prefix (5) -> 20 chars — safely inside the varchar(30) PK. sha1 is content
    addressing, not security (usedforsecurity=False, Bandit B324).
    """
    digest = hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:12]
    return f"{id_prefix}aa_{digest}"


def _brand_scaled_arm_cate(brand: Brand, arm_name: str) -> Dict[str, float]:
    """The CATE-by-segment map the DGP actually plants for (brand, arm).

    treatment_arm reads the config HETEROGENEOUS map through brand_scaled_cate;
    the commercial arms brand-scale their ArmSpec latent map with the same
    _BRAND_CATE_SCALE SSOT (patient_generator.py idiom). Never re-hardcoded.
    """
    if arm_name == "treatment_arm":
        return brand_scaled_cate(brand)
    scale = _BRAND_CATE_SCALE.get(brand, 1.0)
    return {seg: round(v * scale, 4) for seg, v in ARM_REGISTRY[arm_name].cate_by_segment.items()}


class AgentActivitiesGenerator(BaseGenerator[pd.DataFrame]):
    """agent_activities frame: curated causal blocks + generic legacy roster."""

    @property
    def entity_type(self) -> str:
        return "agent_activities"

    # ------------------------------------------------------------------
    # row assembly
    # ------------------------------------------------------------------

    def _timestamp(self) -> str:
        """One activity timestamp inside the configured window (anchor-aware)."""
        day = self._random_dates(1)[0]
        hour = int(self._rng.integers(6, 23))
        minute = int(self._rng.integers(0, 60))
        return f"{day}T{hour:02d}:{minute:02d}:00+00:00"

    def _base_row(
        self,
        key: str,
        agent_name: str,
        activity_type: str,
        analysis_results: Dict[str, Any],
        *,
        recommendations: List[Dict[str, Any]],
        confidence_level: float,
        causal_paths_analyzed: int,
        impact_estimate: float,
        roi_estimate: float,
    ) -> Dict[str, Any]:
        ts = self._timestamp()
        return {
            "activity_id": _activity_id(self.config.id_prefix, key),
            "agent_name": agent_name,
            "agent_tier": LEGACY_AGENT_TIERS[agent_name],
            "activity_timestamp": ts,
            "activity_type": activity_type,
            "workstream": str(self._rng.choice(list(_WORKSTREAMS))),
            "processing_duration_ms": int(self._rng.integers(200, 30000)),
            "input_data": {
                "patient_count": int(self._rng.integers(50, 2000)),
                "hcp_count": int(self._rng.integers(5, 200)),
            },
            "records_processed": int(self._rng.integers(100, 10000)),
            "time_window": str(self._rng.choice(list(_TIME_WINDOWS))),
            "analysis_results": analysis_results,
            "causal_paths_analyzed": causal_paths_analyzed,
            "confidence_level": confidence_level,
            "recommendations": recommendations,
            "actions_initiated": [],
            "impact_estimate": impact_estimate,
            "roi_estimate": roi_estimate,
            "status": "completed",
            "error_message": None,
            "resource_usage": {
                "cpu_seconds": round(float(self._rng.uniform(0.1, 10.0)), 2),
                "memory_mb": int(self._rng.integers(50, 500)),
            },
            # Agent activities are operational telemetry, not model-training
            # rows — like causal_paths they stay outside the ML split quota.
            "data_split": "unassigned",
            # == activity_timestamp so the frame is reproducible from the seed
            # alone (idempotent upsert; no wall-clock in the payload).
            "created_at": ts,
            "is_synthetic": True,
        }

    # ------------------------------------------------------------------
    # curated blocks
    # ------------------------------------------------------------------

    def _heterogeneous_optimizer_rows(self) -> List[Dict[str, Any]]:
        """One CATE analysis per (brand x ARM_REGISTRY arm), mirroring the
        curated brand-scaled CATE constants the DGP plants."""
        rows: List[Dict[str, Any]] = []
        for brand_str in _BRANDS:
            brand = Brand(brand_str)
            for arm_name, spec in ARM_REGISTRY.items():
                cate_map = _brand_scaled_arm_cate(brand, arm_name)
                if not cate_map:
                    continue
                outcome = spec.target_outcomes[0]
                segment_mean = round(sum(cate_map.values()) / len(cate_map), 4)
                top_segment = max(cate_map, key=lambda s: cate_map[s])
                rows.append(
                    self._base_row(
                        key=f"het|{brand_str}|{arm_name}",
                        agent_name="heterogeneous_optimizer",
                        activity_type="cate_analysis",
                        analysis_results={
                            "brand": brand_str,
                            "treatment_var": arm_name,
                            "outcome_var": outcome,
                            # The curated design map (latent scale x brand
                            # scale) — the constants the DGP plants, SSOT
                            # dgp/treatment_arm.py.
                            "cate_by_segment": cate_map,
                            # Unweighted mean of the design map's segment
                            # values (a design summary, NOT an estimate).
                            "overall_ate": segment_mean,
                            "heterogeneity_score": round(
                                max(cate_map.values()) - min(cate_map.values()), 4
                            ),
                            "top_segment": top_segment,
                            "segments": list(cate_map),
                            "source": "comm_arms_design_constants",
                        },
                        recommendations=[
                            {
                                "action": f"prioritize_{top_segment}_for_{arm_name}",
                                "priority": "high",
                            }
                        ],
                        confidence_level=round(float(self._rng.uniform(0.80, 0.95)), 3),
                        causal_paths_analyzed=int(self._rng.integers(3, 20)),
                        impact_estimate=round(segment_mean * float(self._rng.uniform(1e5, 5e5)), 2),
                        roi_estimate=round(float(self._rng.uniform(2.0, 8.0)), 2),
                    )
                )
        return rows

    def _causal_impact_rows(self) -> List[Dict[str, Any]]:
        """One causal analysis per (brand x commercial-arm edge), reproducing
        the causal_paths registry's content-addressed display effect so the
        agent-analysis view and the registry never disagree."""
        rows: List[Dict[str, Any]] = []
        for brand_str in _BRANDS:
            for arm, outcome, confounders, lo, hi in _COMM_ARM_EDGES:
                # Same content-addressed rng (and FIRST draw) as the scp_a*
                # causal_paths rows -> identical effect value by construction.
                edge_rng = _commercial_edge_rng(f"arm|{brand_str}", arm, outcome)
                effect = round(float(edge_rng.uniform(lo, hi)), 4)
                rows.append(
                    self._base_row(
                        key=f"ci|{brand_str}|{arm}|{outcome}",
                        agent_name="causal_impact",
                        activity_type="causal_analysis",
                        analysis_results={
                            "brand": brand_str,
                            "treatment_var": arm,
                            "outcome_var": outcome,
                            "ate_estimate": effect,
                            "confounders_controlled": list(confounders),
                            "method": "backdoor.linear_regression",
                            "refutation_passed": True,
                            "source": "causal_paths_registry_display_value",
                        },
                        recommendations=[{"action": f"scale_{arm}", "priority": "medium"}],
                        confidence_level=round(float(self._rng.uniform(0.80, 0.95)), 3),
                        causal_paths_analyzed=int(self._rng.integers(1, 10)),
                        impact_estimate=round(effect * float(self._rng.uniform(1e5, 5e5)), 2),
                        roi_estimate=round(float(self._rng.uniform(1.5, 6.0)), 2),
                    )
                )
        return rows

    def _gap_analyzer_rows(self) -> List[Dict[str, Any]]:
        """One gap/ROI analysis per (brand x region) — the roi_estimate
        substrate for business_impact_roi_agent_activities (migration 044)."""
        rows: List[Dict[str, Any]] = []
        for brand_str in _BRANDS:
            for region in _REGIONS:
                addressable = round(float(self._rng.uniform(5e4, 8e5)), 2)
                gaps = int(self._rng.integers(2, 9))
                rows.append(
                    self._base_row(
                        key=f"gap|{brand_str}|{region}",
                        agent_name="gap_analyzer",
                        activity_type="gap_analysis",
                        analysis_results={
                            "brand": brand_str,
                            "region": region,
                            "gaps_found": gaps,
                            "total_addressable_value": addressable,
                            "quick_wins_count": int(self._rng.integers(0, 4)),
                        },
                        recommendations=[
                            {
                                "action": f"close_top_gap_{region}",
                                "priority": str(self._rng.choice(["critical", "high", "medium"])),
                            }
                        ],
                        confidence_level=round(float(self._rng.uniform(0.75, 0.92)), 3),
                        causal_paths_analyzed=int(self._rng.integers(0, 8)),
                        impact_estimate=addressable,
                        # numeric(5,2): keep well inside +/-999.99
                        roi_estimate=round(float(self._rng.uniform(1.5, 8.0)), 2),
                    )
                )
        return rows

    def _generic_rows(self) -> List[Dict[str, Any]]:
        """Legacy-shaped tiered activity rows for the rest of the v3 roster.
        n_records sizes this block (the curated blocks are fixed additive)."""
        rows: List[Dict[str, Any]] = []
        n = max(len(_GENERIC_AGENTS), self.config.n_records)
        for i in range(n):
            agent = _GENERIC_AGENTS[i % len(_GENERIC_AGENTS)]
            brand_str = _BRANDS[i % len(_BRANDS)]
            rows.append(
                self._base_row(
                    key=f"gen|{agent}|{i}",
                    agent_name=agent,
                    activity_type=str(
                        self._rng.choice(["analysis", "recommendation", "alert", "experiment"])
                    ),
                    analysis_results={
                        "brand": brand_str,
                        "insights_found": int(self._rng.integers(1, 20)),
                        "anomalies_detected": int(self._rng.integers(0, 5)),
                    },
                    recommendations=[
                        {
                            "action": f"review_finding_{j}",
                            "priority": str(self._rng.choice(["high", "medium", "low"])),
                        }
                        for j in range(int(self._rng.integers(0, 3)))
                    ],
                    confidence_level=round(float(self._rng.uniform(0.70, 0.99)), 3),
                    causal_paths_analyzed=int(self._rng.integers(0, 50)),
                    impact_estimate=round(float(self._rng.uniform(1e3, 5e5)), 2),
                    roi_estimate=round(float(self._rng.uniform(1.5, 10.0)), 2),
                )
            )
        return rows

    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        rows = (
            self._heterogeneous_optimizer_rows()
            + self._causal_impact_rows()
            + self._gap_analyzer_rows()
            + self._generic_rows()
        )
        return pd.DataFrame(rows)
