#!/usr/bin/env python3
"""
E2I Causal Analytics - KPI Coverage Validator V3.0
Validates that all 46 KPIs are calculable from the V3 schema.

Usage:
    python validate_kpi_coverage.py                    # Validate against Supabase
    python validate_kpi_coverage.py --dry-run          # Schema check only (no DB)
    python validate_kpi_coverage.py --verbose          # Detailed output
    python validate_kpi_coverage.py --output report.md # Generate markdown report
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root (parent of e2i_ml_compliant_data)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback to current directory
        load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Optional: Supabase client for live validation
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


class CalculationType(Enum):
    DIRECT = "direct"       # Single column/table query
    DERIVED = "derived"     # Requires calculation from multiple columns
    VIEW = "view"           # Uses KPI helper view


class ValidationStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"
    SKIP = "⏭️ SKIP"


@dataclass
class KPIDefinition:
    """KPI definition with validation requirements."""
    id: str
    name: str
    workstream: str
    category: str
    calculation_type: CalculationType
    tables: List[str]
    columns: List[str]
    view: Optional[str] = None
    is_v3_new: bool = False
    note: str = ""


@dataclass
class ValidationResult:
    """Result of validating a single KPI."""
    kpi: KPIDefinition
    status: ValidationStatus
    message: str
    table_exists: Dict[str, bool] = field(default_factory=dict)
    column_exists: Dict[str, bool] = field(default_factory=dict)
    view_exists: bool = False
    sample_query: str = ""
    sample_count: int = 0


# =============================================================================
# KPI DEFINITIONS (46 Total)
# =============================================================================

KPI_DEFINITIONS: List[KPIDefinition] = [
    # -------------------------------------------------------------------------
    # WS1: Data Quality (9 KPIs)
    # -------------------------------------------------------------------------
    KPIDefinition(
        id="WS1-DQ-001", name="Source Coverage - Patients", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.DERIVED,
        tables=["patient_journeys", "reference_universe"],
        columns=["patient_journeys.patient_id", "reference_universe.total_count"],
        is_v3_new=True, note="V3: reference_universe table"
    ),
    KPIDefinition(
        id="WS1-DQ-002", name="Source Coverage - HCPs", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.DERIVED,
        tables=["hcp_profiles", "reference_universe"],
        columns=["hcp_profiles.coverage_status", "reference_universe.total_count"],
        is_v3_new=True, note="V3: reference_universe table"
    ),
    KPIDefinition(
        id="WS1-DQ-003", name="Cross-source Match Rate", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.VIEW,
        tables=["data_source_tracking"],
        columns=["data_source_tracking.match_rate_vs_claims", "data_source_tracking.match_rate_vs_ehr"],
        view="v_kpi_cross_source_match",
        is_v3_new=True, note="V3: NEW data_source_tracking table"
    ),
    KPIDefinition(
        id="WS1-DQ-004", name="Stacking Lift", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.VIEW,
        tables=["data_source_tracking"],
        columns=["data_source_tracking.stacking_lift_percentage"],
        view="v_kpi_stacking_lift",
        is_v3_new=True, note="V3: NEW data_source_tracking table"
    ),
    KPIDefinition(
        id="WS1-DQ-005", name="Completeness Pass Rate", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.DERIVED,
        tables=["patient_journeys"],
        columns=["patient_journeys.data_quality_score"]
    ),
    KPIDefinition(
        id="WS1-DQ-006", name="Geographic Consistency", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.DERIVED,
        tables=["patient_journeys"],
        columns=["patient_journeys.geographic_region", "patient_journeys.state"]
    ),
    KPIDefinition(
        id="WS1-DQ-007", name="Data Lag (Median)", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.VIEW,
        tables=["patient_journeys"],
        columns=["patient_journeys.source_timestamp", "patient_journeys.ingestion_timestamp", 
                 "patient_journeys.data_lag_hours"],
        view="v_kpi_data_lag",
        is_v3_new=True, note="V3: NEW fields in patient_journeys"
    ),
    KPIDefinition(
        id="WS1-DQ-008", name="Label Quality (IAA)", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.VIEW,
        tables=["ml_annotations"],
        columns=["ml_annotations.iaa_group_id", "ml_annotations.annotation_value", 
                 "ml_annotations.annotation_confidence"],
        view="v_kpi_label_quality",
        is_v3_new=True, note="V3: NEW ml_annotations table"
    ),
    KPIDefinition(
        id="WS1-DQ-009", name="Time-to-Release (TTR)", workstream="WS1",
        category="Data Quality", calculation_type=CalculationType.VIEW,
        tables=["etl_pipeline_metrics"],
        columns=["etl_pipeline_metrics.time_to_release_hours", 
                 "etl_pipeline_metrics.source_data_timestamp"],
        view="v_kpi_time_to_release",
        is_v3_new=True, note="V3: NEW etl_pipeline_metrics table"
    ),

    # -------------------------------------------------------------------------
    # WS1: Model Performance (9 KPIs)
    # -------------------------------------------------------------------------
    KPIDefinition(
        id="WS1-MP-001", name="ROC-AUC", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.model_auc"]
    ),
    KPIDefinition(
        id="WS1-MP-002", name="PR-AUC", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.model_pr_auc"],
        is_v3_new=True, note="V3: NEW field model_pr_auc"
    ),
    KPIDefinition(
        id="WS1-MP-003", name="F1 Score", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DERIVED,
        tables=["ml_predictions"],
        columns=["ml_predictions.model_precision", "ml_predictions.model_recall"]
    ),
    KPIDefinition(
        id="WS1-MP-004", name="Recall@Top-K", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.rank_metrics"],
        is_v3_new=True, note="V3: NEW field rank_metrics (JSONB)"
    ),
    KPIDefinition(
        id="WS1-MP-005", name="Brier Score", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.brier_score"],
        is_v3_new=True, note="V3: NEW field brier_score"
    ),
    KPIDefinition(
        id="WS1-MP-006", name="Calibration Slope", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.calibration_score"]
    ),
    KPIDefinition(
        id="WS1-MP-007", name="SHAP Coverage", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DERIVED,
        tables=["ml_predictions"],
        columns=["ml_predictions.shap_values"]
    ),
    KPIDefinition(
        id="WS1-MP-008", name="Fairness Gap (ΔRecall)", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.fairness_metrics"]
    ),
    KPIDefinition(
        id="WS1-MP-009", name="Feature Drift (PSI)", workstream="WS1",
        category="Model Performance", calculation_type=CalculationType.DERIVED,
        tables=["ml_preprocessing_metadata", "ml_predictions"],
        columns=["ml_preprocessing_metadata.feature_distributions"]
    ),

    # -------------------------------------------------------------------------
    # WS2: Trigger Performance (8 KPIs)
    # -------------------------------------------------------------------------
    KPIDefinition(
        id="WS2-TR-001", name="Trigger Precision", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.DERIVED,
        tables=["triggers"],
        columns=["triggers.outcome_tracked", "triggers.outcome_value"]
    ),
    KPIDefinition(
        id="WS2-TR-002", name="Trigger Recall", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.DERIVED,
        tables=["triggers", "treatment_events"],
        columns=["triggers.trigger_id", "treatment_events.event_type"]
    ),
    KPIDefinition(
        id="WS2-TR-003", name="Action Rate Uplift", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.DERIVED,
        tables=["triggers"],
        columns=["triggers.action_taken", "triggers.control_group_flag"]
    ),
    KPIDefinition(
        id="WS2-TR-004", name="Acceptance Rate", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.DIRECT,
        tables=["triggers"],
        columns=["triggers.acceptance_status"]
    ),
    KPIDefinition(
        id="WS2-TR-005", name="False Alert Rate", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.DIRECT,
        tables=["triggers"],
        columns=["triggers.false_positive_flag"]
    ),
    KPIDefinition(
        id="WS2-TR-006", name="Override Rate", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.DERIVED,
        tables=["triggers"],
        columns=["triggers.acceptance_status"]
    ),
    KPIDefinition(
        id="WS2-TR-007", name="Lead Time", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.DIRECT,
        tables=["triggers"],
        columns=["triggers.lead_time_days"]
    ),
    KPIDefinition(
        id="WS2-TR-008", name="Change-Fail Rate (CFR)", workstream="WS2",
        category="Trigger Performance", calculation_type=CalculationType.VIEW,
        tables=["triggers"],
        columns=["triggers.previous_trigger_id", "triggers.change_type", 
                 "triggers.change_failed", "triggers.change_outcome_delta"],
        view="v_kpi_change_fail_rate",
        is_v3_new=True, note="V3: NEW change tracking fields"
    ),

    # -------------------------------------------------------------------------
    # WS3: Business Impact (10 KPIs)
    # -------------------------------------------------------------------------
    KPIDefinition(
        id="WS3-BI-001", name="Monthly Active Users (MAU)", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.VIEW,
        tables=["user_sessions"],
        columns=["user_sessions.user_id", "user_sessions.session_start"],
        view="v_kpi_active_users",
        is_v3_new=True, note="V3: NEW user_sessions table"
    ),
    KPIDefinition(
        id="WS3-BI-002", name="Weekly Active Users (WAU)", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.VIEW,
        tables=["user_sessions"],
        columns=["user_sessions.user_id", "user_sessions.session_start"],
        view="v_kpi_active_users",
        is_v3_new=True, note="V3: NEW user_sessions table"
    ),
    KPIDefinition(
        id="WS3-BI-003", name="Patient Touch Rate", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DERIVED,
        tables=["triggers", "patient_journeys"],
        columns=["triggers.patient_id", "patient_journeys.patient_id"]
    ),
    KPIDefinition(
        id="WS3-BI-004", name="HCP Coverage", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DIRECT,
        tables=["hcp_profiles"],
        columns=["hcp_profiles.coverage_status"]
    ),
    KPIDefinition(
        id="WS3-BI-005", name="Total Prescriptions (TRx)", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DERIVED,
        tables=["treatment_events"],
        columns=["treatment_events.event_type"]
    ),
    KPIDefinition(
        id="WS3-BI-006", name="New Prescriptions (NRx)", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DERIVED,
        tables=["treatment_events"],
        columns=["treatment_events.event_type", "treatment_events.sequence_number"]
    ),
    KPIDefinition(
        id="WS3-BI-007", name="New-to-Brand (NBRx)", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DERIVED,
        tables=["treatment_events"],
        columns=["treatment_events.event_type", "treatment_events.brand"]
    ),
    KPIDefinition(
        id="WS3-BI-008", name="TRx Share", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DERIVED,
        tables=["treatment_events"],
        columns=["treatment_events.brand"]
    ),
    KPIDefinition(
        id="WS3-BI-009", name="Conversion Rate", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DERIVED,
        tables=["triggers", "treatment_events"],
        columns=["triggers.trigger_id", "treatment_events.event_type"]
    ),
    KPIDefinition(
        id="WS3-BI-010", name="ROI", workstream="WS3",
        category="Business Impact", calculation_type=CalculationType.DIRECT,
        tables=["business_metrics", "agent_activities"],
        columns=["business_metrics.roi", "agent_activities.roi_estimate"]
    ),

    # -------------------------------------------------------------------------
    # Brand-Specific (5 KPIs)
    # -------------------------------------------------------------------------
    KPIDefinition(
        id="BR-001", name="Remi - AH Uncontrolled %", workstream="Brand",
        category="Brand-Specific", calculation_type=CalculationType.DERIVED,
        tables=["patient_journeys", "treatment_events"],
        columns=["patient_journeys.diagnosis", "treatment_events.treatment_response"]
    ),
    KPIDefinition(
        id="BR-002", name="Remi - Intent-to-Prescribe Δ", workstream="Brand",
        category="Brand-Specific", calculation_type=CalculationType.VIEW,
        tables=["hcp_intent_surveys"],
        columns=["hcp_intent_surveys.intent_to_prescribe_score", 
                 "hcp_intent_surveys.intent_to_prescribe_change"],
        view="v_kpi_intent_to_prescribe",
        is_v3_new=True, note="V3: NEW hcp_intent_surveys table"
    ),
    KPIDefinition(
        id="BR-003", name="Fabhalta - % PNH Tested", workstream="Brand",
        category="Brand-Specific", calculation_type=CalculationType.DERIVED,
        tables=["treatment_events"],
        columns=["treatment_events.event_type"]
    ),
    KPIDefinition(
        id="BR-004", name="Kisqali - Dx Adoption", workstream="Brand",
        category="Brand-Specific", calculation_type=CalculationType.DERIVED,
        tables=["patient_journeys", "treatment_events"],
        columns=["patient_journeys.diagnosis_date", "treatment_events.event_date"]
    ),
    KPIDefinition(
        id="BR-005", name="Kisqali - Oncologist Reach", workstream="Brand",
        category="Brand-Specific", calculation_type=CalculationType.DERIVED,
        tables=["hcp_profiles", "triggers"],
        columns=["hcp_profiles.specialty", "triggers.hcp_id"]
    ),

    # -------------------------------------------------------------------------
    # Causal Metrics (5 KPIs)
    # -------------------------------------------------------------------------
    KPIDefinition(
        id="CM-001", name="Average Treatment Effect (ATE)", workstream="Causal",
        category="Causal Metrics", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.treatment_effect_estimate"]
    ),
    KPIDefinition(
        id="CM-002", name="Conditional ATE (CATE)", workstream="Causal",
        category="Causal Metrics", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.heterogeneous_effect", "ml_predictions.segment_assignment"]
    ),
    KPIDefinition(
        id="CM-003", name="Causal Impact", workstream="Causal",
        category="Causal Metrics", calculation_type=CalculationType.DIRECT,
        tables=["causal_paths"],
        columns=["causal_paths.causal_effect_size", "causal_paths.confidence_level"]
    ),
    KPIDefinition(
        id="CM-004", name="Counterfactual Outcome", workstream="Causal",
        category="Causal Metrics", calculation_type=CalculationType.DIRECT,
        tables=["ml_predictions"],
        columns=["ml_predictions.counterfactual_outcome"]
    ),
    KPIDefinition(
        id="CM-005", name="Mediation Effect", workstream="Causal",
        category="Causal Metrics", calculation_type=CalculationType.DERIVED,
        tables=["causal_paths"],
        columns=["causal_paths.mediators_identified", "causal_paths.pathway_details"]
    ),
]


# =============================================================================
# V3 SCHEMA DEFINITION
# =============================================================================

V3_SCHEMA = {
    "tables": {
        # Core tables
        "patient_journeys": {
            "columns": [
                "patient_id", "journey_id", "brand", "geographic_region", "state",
                "journey_stage", "journey_status", "diagnosis", "diagnosis_date",
                "data_quality_score", "data_split", "created_at", "updated_at",
                # V3 NEW fields
                "data_source", "data_sources_matched", "source_match_confidence",
                "source_stacking_flag", "source_combination_method",
                "source_timestamp", "ingestion_timestamp", "data_lag_hours"
            ],
            "is_new": False
        },
        "hcp_profiles": {
            "columns": [
                "hcp_id", "specialty", "geographic_region", "state", "priority_tier",
                "coverage_status", "engagement_score", "created_at", "updated_at"
            ],
            "is_new": False
        },
        "treatment_events": {
            "columns": [
                "event_id", "patient_id", "hcp_id", "brand", "event_type", "event_date",
                "sequence_number", "treatment_response", "data_split", "created_at"
            ],
            "is_new": False
        },
        "ml_predictions": {
            "columns": [
                "prediction_id", "patient_id", "hcp_id", "brand", "prediction_type",
                "prediction_score", "confidence_interval_lower", "confidence_interval_upper",
                "model_version", "model_auc", "model_precision", "model_recall",
                "calibration_score", "shap_values", "fairness_metrics",
                "treatment_effect_estimate", "heterogeneous_effect", "segment_assignment",
                "counterfactual_outcome", "features_available_at_prediction",
                "data_split", "created_at",
                # V3 NEW fields
                "model_pr_auc", "rank_metrics", "brier_score"
            ],
            "is_new": False
        },
        "triggers": {
            "columns": [
                "trigger_id", "patient_id", "hcp_id", "brand", "trigger_type",
                "trigger_reason", "priority", "acceptance_status", "action_taken",
                "outcome_tracked", "outcome_value", "false_positive_flag",
                "lead_time_days", "control_group_flag", "data_split", "created_at",
                # V3 NEW fields
                "previous_trigger_id", "change_type", "change_reason",
                "change_timestamp", "change_failed", "change_outcome_delta"
            ],
            "is_new": False
        },
        "agent_activities": {
            "columns": [
                "activity_id", "agent_name", "workstream", "brand", "analysis_type",
                "analysis_results", "roi_estimate", "confidence_score", "data_split",
                "created_at",
                # V3 NEW field
                "agent_tier"
            ],
            "is_new": False
        },
        "business_metrics": {
            "columns": [
                "metric_id", "brand", "geographic_region", "workstream", "metric_name",
                "metric_value", "period_start", "period_end", "roi", "data_split",
                "created_at"
            ],
            "is_new": False
        },
        "causal_paths": {
            "columns": [
                "path_id", "brand", "source_node", "target_node", "pathway_details",
                "causal_effect_size", "confidence_level", "mediators_identified",
                "data_split", "created_at"
            ],
            "is_new": False
        },
        "ml_split_registry": {
            "columns": [
                "split_id", "split_name", "train_ratio", "validation_ratio",
                "test_ratio", "holdout_ratio", "train_start", "train_end",
                "validation_start", "validation_end", "test_start", "test_end",
                "holdout_start", "holdout_end", "temporal_gap_days", "is_active",
                "created_at"
            ],
            "is_new": False
        },
        "ml_preprocessing_metadata": {
            "columns": [
                "metadata_id", "split_id", "feature_name", "computed_on_split",
                "mean_value", "std_value", "min_value", "max_value",
                "feature_distributions", "created_at"
            ],
            "is_new": False
        },
        
        # V3 NEW TABLES
        "user_sessions": {
            "columns": [
                "session_id", "user_id", "user_role", "session_start", "session_end",
                "session_duration_seconds", "page_views", "queries_executed",
                "engagement_score", "created_at"
            ],
            "is_new": True
        },
        "data_source_tracking": {
            "columns": [
                "tracking_id", "tracking_date", "source_name", "brand",
                "total_records", "records_matched", "match_rate_vs_claims",
                "match_rate_vs_ehr", "match_rate_vs_specialty", "match_rate_vs_lab",
                "stacking_eligible_records", "stacking_applied_records",
                "stacking_lift_percentage", "source_combination_flags", "created_at"
            ],
            "is_new": True
        },
        "ml_annotations": {
            "columns": [
                "annotation_id", "entity_type", "entity_id", "annotation_type",
                "annotator_id", "annotation_value", "annotation_confidence",
                "iaa_group_id", "created_at"
            ],
            "is_new": True
        },
        "etl_pipeline_metrics": {
            "columns": [
                "metric_id", "pipeline_name", "run_id", "run_started_at",
                "run_completed_at", "source_data_timestamp", "time_to_release_hours",
                "records_processed", "records_failed", "stage_timings",
                "error_details", "created_at"
            ],
            "is_new": True
        },
        "hcp_intent_surveys": {
            "columns": [
                "survey_id", "hcp_id", "brand", "survey_date",
                "intent_to_prescribe_score", "intent_to_prescribe_change",
                "previous_survey_id", "survey_source", "created_at"
            ],
            "is_new": True
        },
        "reference_universe": {
            "columns": [
                "universe_id", "universe_type", "brand", "geographic_region",
                "specialty", "total_count", "target_count", "effective_date",
                "source", "created_at"
            ],
            "is_new": True
        },
        "agent_registry": {
            "columns": [
                "agent_id", "agent_name", "agent_tier", "description",
                "capabilities", "routes_from_intents", "is_active",
                "config", "created_at", "updated_at"
            ],
            "is_new": True
        }
    },
    "views": [
        "v_kpi_cross_source_match",
        "v_kpi_stacking_lift",
        "v_kpi_data_lag",
        "v_kpi_label_quality",
        "v_kpi_time_to_release",
        "v_kpi_change_fail_rate",
        "v_kpi_active_users",
        "v_kpi_intent_to_prescribe"
    ]
}


# =============================================================================
# VALIDATOR CLASS
# =============================================================================

class KPIValidator:
    """Validates KPI calculability against V3 schema."""
    
    def __init__(self, supabase_client: Optional[Any] = None, verbose: bool = False):
        self.client = supabase_client
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        
    def validate_all(self, dry_run: bool = False) -> Tuple[int, int, int]:
        """Validate all KPIs. Returns (passed, failed, warnings)."""
        self.results = []
        
        for kpi in KPI_DEFINITIONS:
            if dry_run:
                result = self._validate_schema_only(kpi)
            else:
                result = self._validate_with_db(kpi)
            self.results.append(result)
            
            if self.verbose:
                self._print_result(result)
        
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAIL)
        warnings = sum(1 for r in self.results if r.status == ValidationStatus.WARN)
        
        return passed, failed, warnings
    
    def _validate_schema_only(self, kpi: KPIDefinition) -> ValidationResult:
        """Validate KPI against schema definition (no DB connection)."""
        result = ValidationResult(kpi=kpi, status=ValidationStatus.PASS, message="")
        
        # Check tables exist in schema
        for table in kpi.tables:
            exists = table in V3_SCHEMA["tables"]
            result.table_exists[table] = exists
            if not exists:
                result.status = ValidationStatus.FAIL
                result.message = f"Table '{table}' not found in V3 schema"
                return result
        
        # Check columns exist in schema
        for col_spec in kpi.columns:
            if "." in col_spec:
                table, column = col_spec.split(".", 1)
                if table in V3_SCHEMA["tables"]:
                    exists = column in V3_SCHEMA["tables"][table]["columns"]
                    result.column_exists[col_spec] = exists
                    if not exists:
                        result.status = ValidationStatus.FAIL
                        result.message = f"Column '{col_spec}' not found in V3 schema"
                        return result
        
        # Check view exists if required
        if kpi.view:
            result.view_exists = kpi.view in V3_SCHEMA["views"]
            if not result.view_exists:
                result.status = ValidationStatus.WARN
                result.message = f"View '{kpi.view}' not found (may need creation)"
        
        # Generate sample query
        result.sample_query = self._generate_sample_query(kpi)
        result.message = "Schema validation passed"
        
        return result
    
    def _validate_with_db(self, kpi: KPIDefinition) -> ValidationResult:
        """Validate KPI against live database."""
        if not self.client:
            return self._validate_schema_only(kpi)
        
        result = ValidationResult(kpi=kpi, status=ValidationStatus.PASS, message="")
        
        try:
            # Check tables exist
            for table in kpi.tables:
                response = self.client.table(table).select("*").limit(1).execute()
                result.table_exists[table] = True
                result.sample_count = len(response.data) if response.data else 0
        except Exception as e:
            result.status = ValidationStatus.FAIL
            result.message = f"Table check failed: {str(e)}"
            return result
        
        # Check view if required
        if kpi.view:
            try:
                response = self.client.table(kpi.view).select("*").limit(1).execute()
                result.view_exists = True
            except Exception:
                result.view_exists = False
                result.status = ValidationStatus.WARN
                result.message = f"View '{kpi.view}' not accessible"
        
        result.sample_query = self._generate_sample_query(kpi)
        if result.status == ValidationStatus.PASS:
            result.message = "Database validation passed"
        
        return result
    
    def _generate_sample_query(self, kpi: KPIDefinition) -> str:
        """Generate sample SQL query for KPI calculation."""
        if kpi.view:
            return f"SELECT * FROM {kpi.view};"
        
        if len(kpi.tables) == 1:
            table = kpi.tables[0]
            cols = [c.split(".")[-1] for c in kpi.columns if c.startswith(table)]
            return f"SELECT {', '.join(cols)} FROM {table} LIMIT 10;"
        
        # Multi-table query
        main_table = kpi.tables[0]
        return f"SELECT * FROM {main_table} LIMIT 10;  -- Join with {', '.join(kpi.tables[1:])}"
    
    def _print_result(self, result: ValidationResult) -> None:
        """Print validation result to console."""
        status_str = result.status.value
        print(f"  {status_str} [{result.kpi.id}] {result.kpi.name}")
        if result.message and result.status != ValidationStatus.PASS:
            print(f"       └─ {result.message}")
        if result.kpi.is_v3_new:
            print(f"       └─ V3 NEW: {result.kpi.note}")
    
    def generate_report(self) -> str:
        """Generate markdown report of validation results."""
        lines = [
            "# E2I KPI Coverage Validation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAIL)
        warnings = sum(1 for r in self.results if r.status == ValidationStatus.WARN)
        v3_new = sum(1 for r in self.results if r.kpi.is_v3_new)
        
        lines.extend([
            f"- **Total KPIs**: {len(self.results)}",
            f"- **Passed**: {passed} ✅",
            f"- **Failed**: {failed} ❌",
            f"- **Warnings**: {warnings} ⚠️",
            f"- **V3 New**: {v3_new}",
            "",
            "## By Workstream",
            ""
        ])
        
        # Group by workstream
        workstreams = {}
        for result in self.results:
            ws = result.kpi.workstream
            if ws not in workstreams:
                workstreams[ws] = []
            workstreams[ws].append(result)
        
        for ws, results in sorted(workstreams.items()):
            ws_passed = sum(1 for r in results if r.status == ValidationStatus.PASS)
            lines.append(f"### {ws} ({ws_passed}/{len(results)})")
            lines.append("")
            lines.append("| ID | KPI | Status | V3 New | Notes |")
            lines.append("|---|---|---|---|---|")
            
            for r in results:
                v3 = "✓" if r.kpi.is_v3_new else ""
                status = "✅" if r.status == ValidationStatus.PASS else (
                    "❌" if r.status == ValidationStatus.FAIL else "⚠️"
                )
                notes = r.kpi.note if r.kpi.is_v3_new else r.message
                lines.append(f"| {r.kpi.id} | {r.kpi.name} | {status} | {v3} | {notes} |")
            
            lines.append("")
        
        # V3 New Items
        lines.extend([
            "## V3 New Tables & Fields",
            "",
            "### New Tables",
            ""
        ])
        
        for table, info in V3_SCHEMA["tables"].items():
            if info["is_new"]:
                lines.append(f"- `{table}`")
        
        lines.extend([
            "",
            "### New KPI Helper Views",
            ""
        ])
        
        for view in V3_SCHEMA["views"]:
            lines.append(f"- `{view}`")
        
        return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate E2I KPI coverage against V3 schema"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Schema check only (no database connection)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for each KPI"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output markdown report to file"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("E2I Causal Analytics - KPI Coverage Validator V3.0")
    print("=" * 70)
    print()

    # Show .env loading status
    if DOTENV_AVAILABLE:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            print(f"✓ Loaded environment from: {env_path}")
        else:
            print("⚠ .env file not found, using system environment")
    else:
        print("⚠ python-dotenv not installed, using system environment")
        print("  Install with: pip install python-dotenv")

    # Initialize Supabase client if available and not dry-run
    client = None
    if not args.dry_run and SUPABASE_AVAILABLE:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if url and key:
            try:
                client = create_client(url, key)
                print(f"✓ Connected to Supabase: {url}")
            except Exception as e:
                print(f"⚠ Could not connect to Supabase: {e}")
                print("  Running in dry-run mode...")
                args.dry_run = True
        else:
            print("⚠ SUPABASE_URL/SUPABASE_SERVICE_KEY not set")
            print("  Running in dry-run mode...")
            args.dry_run = True
    elif args.dry_run:
        print("Running in dry-run mode (schema check only)")
    else:
        print("⚠ Supabase client not available")
        print("  Install with: pip install supabase")
        print("  Running in dry-run mode...")
        args.dry_run = True
    
    print()
    
    # Run validation
    validator = KPIValidator(supabase_client=client, verbose=args.verbose)
    
    print(f"Validating {len(KPI_DEFINITIONS)} KPIs...")
    print("-" * 70)
    
    passed, failed, warnings = validator.validate_all(dry_run=args.dry_run)
    
    print("-" * 70)
    print()
    
    # Summary
    total = len(KPI_DEFINITIONS)
    v3_new = sum(1 for kpi in KPI_DEFINITIONS if kpi.is_v3_new)
    
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total KPIs:     {total}")
    print(f"  ✅ Passed:      {passed}")
    print(f"  ❌ Failed:      {failed}")
    print(f"  ⚠️  Warnings:    {warnings}")
    print(f"  🆕 V3 New:      {v3_new}")
    print()
    
    coverage_pct = (passed / total) * 100 if total > 0 else 0
    print(f"  Coverage: {passed}/{total} ({coverage_pct:.1f}%)")
    print()
    
    if coverage_pct == 100:
        print("  🎉 100% KPI COVERAGE ACHIEVED!")
    elif coverage_pct >= 90:
        print("  ✓ Near complete coverage")
    else:
        print("  ⚠ Coverage gaps remain")
    
    print("=" * 70)
    
    # Generate report if requested
    if args.output:
        report = validator.generate_report()
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
