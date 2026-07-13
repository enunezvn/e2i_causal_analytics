"""
Batch Loader for Supabase.

Loads synthetic data to Supabase in batches with validation and error handling.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of a batch load operation."""

    table_name: str
    records_loaded: int
    records_failed: int
    total_batches: int
    failed_batches: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.records_loaded + self.records_failed
        return self.records_loaded / total if total > 0 else 0.0

    @property
    def is_success(self) -> bool:
        """Check if load was successful (>95% success rate)."""
        return self.success_rate >= 0.95


@dataclass
class LoaderConfig:
    """Configuration for batch loader."""

    batch_size: int = 1000
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    validate_before_load: bool = True
    dry_run: bool = False
    verbose: bool = False

    # Connection settings
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    def __post_init__(self):
        """Load from environment if not provided."""
        if not self.supabase_url:
            self.supabase_url = os.getenv("SUPABASE_URL")
        if not self.supabase_key:
            # ETL/seeding writes require the service-role grant: anon/authenticated
            # are INSERT-denied on this stack (post-#058 grant tightening). Prefer the
            # service-role key, mirroring src/api/dependencies/supabase_client.py:96 and
            # src/feature_store/client.py. Fall back to anon only for read-only/dry-run.
            self.supabase_key = (
                os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                or os.getenv("SUPABASE_SERVICE_KEY")
                or os.getenv("SUPABASE_KEY")
                or os.getenv("SUPABASE_ANON_KEY")
            )


# Table loading order (respects foreign key dependencies)
# Note: engagement_events and business_outcomes don't exist in current schema
LOADING_ORDER = [
    "hcp_profiles",
    "patient_journeys",
    "treatment_events",
    "ml_predictions",
    "triggers",
    "business_metrics",
    # Feature store tables (after feature_groups and features are seeded)
    "feature_groups",
    "features",
    "feature_values",
    # --- Shard 09: experiment / MLOps / observability / feedback / view-backed ---
    "ml_experiments",  # parent of ab_* and ml_model_registry/ml_training_runs
    "ml_model_registry",  # parent of ml_training_runs.model_registry_id, ml_deployments.model_registry_id
    "ml_training_runs",
    "ml_deployments",
    "ab_experiment_assignments",  # parent of ab_experiment_enrollments.assignment_id
    "ab_experiment_enrollments",
    "ab_experiment_results",
    "ml_observability_spans",
    "causal_paths",  # CM-003/CM-005 substrate (Task 5c)
    "learning_signals",
    "user_sessions",
    "hcp_intent_surveys",
    "data_source_tracking",
    "etl_pipeline_metrics",
    "ml_annotations",
]

# Column mappings for each table (aligned with actual Supabase schema)
TABLE_COLUMNS = {
    "hcp_profiles": [
        "hcp_id",
        "npi",
        "specialty",
        "practice_type",
        "geographic_region",
        "years_experience",
        "total_patient_volume",
        # Shard 06.3 adoption cohort substrate (resolver reads these from hcp_profiles).
        "peer_influence_score",
        "influence_network_size",
        "adoption_category",
        # Sales territory + rep (territory_metrics ETL groups on territory_id).
        "territory_id",
        "sales_rep_id",
        # Field-force targeting (WS3-BI-004 HCP Coverage substrate).
        "priority_tier",
        "coverage_status",
        "is_synthetic",
    ],
    "patient_journeys": [
        "patient_journey_id",
        "patient_id",
        "hcp_id",
        "brand",
        "journey_start_date",
        "insurance_type",
        "geographic_region",
        "disease_severity",
        "academic_hcp",
        "engagement_score",
        "treatment_initiated",
        "days_to_treatment",
        "age_at_diagnosis",
        # Shard 04 M5 eligibility columns (migration 068). primary_diagnosis_code
        # pre-existed; the other 10 are cohort_constructor required_fields.
        "primary_diagnosis_code",
        "urticaria_severity_uas7",
        "prior_antihistamine_therapy",
        "hr_status",
        "her2_status",
        "disease_stage",
        "ecog_performance_status",
        "ldh_ratio",
        "complement_inhibitor_status",
        "proteinuria_g_day",
        "egfr",
        # Phase 2 anti-IgE axis (migration 107) — Remibrutinib/CSU only, NULL for
        # the oncology/PNH brands. Registered here so the loader carries them (an
        # unregistered generator column is silently dropped at load, line ~885).
        "biologic_experienced",
        "ige_level",
        "data_split",
        # causal substrate (Shard 01 M2 DDL; values filled by Shard 03/06).
        # Emitted as NULL placeholders by PatientGenerator so validate_datasets
        # does not flag them as critical_missing and the loader carries them.
        "treatment_arm",
        "propensity_score",
        "segment_assignment",
        "discontinued_180d",
        "persistent_180d",
        # Commercial-arms enrichment (migration 088). Phase 0 fills the adherence
        # columns; the arm + propensity + insurance_access_score columns load as
        # NULL until their phase populates them (nullable in DB).
        "adherent_180d",
        "low_gap_180d",
        "adherence_rate",
        "gap_days",
        "copay_support",
        "psp_enrolled",
        "rep_detailing_high",
        "sample_dropped",
        "copay_support_propensity",
        "psp_enrolled_propensity",
        "rep_detailing_high_propensity",
        "sample_dropped_propensity",
        "insurance_access_score",
        # Shard 09 Task 5b WS1-DQ-007: recent ingest lag (column exists on the DB,
        # integer; stamped on the synthetic frame by data_lag.stamp_data_lag_hours).
        "data_lag_hours",
        "is_synthetic",
    ],
    "treatment_events": [
        "treatment_event_id",
        "patient_journey_id",
        "patient_id",
        "brand",
        "event_date",
        "event_type",
        "duration_days",
        # Shard 04: indication-correct coding. These DB columns already exist; the
        # loader gates anything unlisted (batch_loader.py:344), so register them or
        # they vanish. NOTE: primary_diagnosis_code is intentionally NOT listed — it
        # is a patient_journeys scalar, not a treatment_events column (DB carries the
        # dx as icd_codes text[]); registering it would 42703 the insert.
        "drug_ndc",
        "drug_name",
        "drug_class",
        "event_subtype",
        "icd_codes",
        # #1116 (BR-003): PNH flow-cytometry lab events carry a real PNH LOINC in
        # loinc_codes (text[]; the registry SQL matches with &&) and the clone-size
        # reading in lab_values (jsonb). Unregistered columns are gated out at load
        # -> the BR-003 numerator would silently stay zero.
        "loinc_codes",
        "lab_values",
        # Shard 09 WS3-BI-006 (NRx): per-(patient,brand) chronological prescription
        # index. NRx counts sequence_number=1 prescriptions; stamped by
        # sequence_number.stamp_sequence_number in the load script.
        "sequence_number",
        "data_split",
        "is_synthetic",
    ],
    "ml_predictions": [
        "prediction_id",
        "patient_id",
        "hcp_id",
        "prediction_type",
        "prediction_value",
        "confidence_score",
        "model_version",
        "prediction_timestamp",
        "data_split",
        # Causal substrate read by migration 044 (causal_metrics_ate / _cate);
        # values populated by Shard 03's DGP, columns must survive the loader.
        "treatment_effect_estimate",
        "heterogeneous_effect",
        "segment_assignment",
        # Shard 09 WS1-MP-002..008 + CM-004: model-quality metrics the
        # model-performance KPIs read. Stamped onto the synthetic frame by
        # model_metrics.stamp_model_metrics (nullable columns, faithful-DB verified).
        "model_auc",
        "model_pr_auc",
        "model_precision",
        "model_recall",
        "brier_score",
        "calibration_score",
        "rank_metrics",
        "fairness_metrics",
        "shap_values",
        "counterfactual_outcome",
        "is_synthetic",
    ],
    "triggers": [
        "trigger_id",
        "patient_id",
        "hcp_id",
        # brand_id is text NOT NULL on the triggers table; the generator already
        # emits it from the patient's brand (trigger_generator.py:201). Must be
        # registered or the loader strips it -> 23502 not-null violation.
        "brand_id",
        "trigger_timestamp",
        "trigger_type",
        "priority",
        "confidence_score",
        "lead_time_days",
        "expiration_date",
        "delivery_channel",
        "delivery_status",
        "acceptance_status",
        # #577 WS2-TR-003: persist the arm + action so a fresh synthetic load stays
        # coherent for action_rate_uplift (else the loader strips them and the
        # registry query's `WHERE control_group_flag IS NOT NULL` finds empty arms).
        "action_taken",
        "control_group_flag",
        "outcome_tracked",
        "outcome_value",
        # #1118 WS2-TR-005: false-positive marking emitted by TriggerGenerator.
        # Must be registered or the loader strips it -> every row reverts to the
        # schema default FALSE and False Alert Rate is vacuously GOOD again.
        "false_positive_flag",
        "trigger_reason",
        "causal_chain",
        "supporting_evidence",
        "recommended_action",
        # Shard 09 WS2-TR-008 (CFR): change-tracking substrate stamped by
        # change_tracking.stamp_change_tracking (nullable cols, faithful-DB verified).
        "previous_trigger_id",
        "change_type",
        "change_failed",
        "change_outcome_delta",
        "data_split",
        "is_synthetic",
    ],
    "business_metrics": [
        "metric_id",
        "metric_date",
        "metric_type",
        "metric_name",
        "brand",
        "region",
        "value",
        "target",
        "achievement_rate",
        "year_over_year_change",
        "month_over_month_change",
        "roi",
        "statistical_significance",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "sample_size",
        "data_split",
        "is_synthetic",
    ],
    "feature_groups": [
        "id",
        "name",
        "description",
        "owner",
        "tags",
        "source_table",
        "expected_update_frequency_hours",
        "max_age_hours",
        "is_synthetic",
    ],
    "features": [
        "id",
        "feature_group_id",
        "name",
        "description",
        "value_type",
        "entity_keys",
        "owner",
        "tags",
        "drift_threshold",
        "is_synthetic",
    ],
    "feature_values": [
        "id",
        "feature_id",
        "entity_values",
        "value",
        "event_timestamp",
        "freshness_status",
        "is_synthetic",
    ],
    # --- Shard 09: columns verified against information_schema.columns on the
    # faithful docker DB. is_synthetic on the 8 MLOps/AB/observability/feedback
    # tables is added by migration 069 (this shard); the 5 view-backed tables +
    # causal_paths + ab_experiment_assignments already carry it (063 / earlier).
    "ml_experiments": [
        "id",
        "experiment_name",
        "description",
        "prediction_target",
        # /experiments usefulness review (2026-07-11): meaningful portfolio
        # metadata — cohort text + the digital-twin intervention taxonomy value
        # (migration 100) the monitor badge + /insights/experiments group by.
        "target_population",
        "intervention_channel",
        # Enrollment plan (migration 101): the monitor's plan-relative health
        # checks + information fraction read these — unregistered they are
        # silently gated out here and every row loads with a NULL plan
        # (caught live 2026-07-11: refresh wrote 360 rows, 0 with a target).
        "target_enrollment",
        "planned_duration_days",
        "observation_window_days",
        "prediction_horizon_days",
        "minimum_auc",
        "minimum_precision_at_k",
        "maximum_fpr",
        "brand",
        "region",
        "created_by",
        "created_at",
        "status",
        "is_synthetic",
    ],
    "ml_model_registry": [
        "id",
        "experiment_id",
        "model_name",
        "model_version",
        "algorithm",
        "feature_count",
        "training_samples",
        "auc",
        "pr_auc",
        "brier_score",
        "calibration_slope",
        "stage",
        "is_champion",
        "trained_at",
        "registered_at",
        "is_synthetic",
    ],
    "ml_training_runs": [
        "id",
        "experiment_id",
        "model_registry_id",
        "run_name",
        "algorithm",
        "hyperparameters",
        "training_samples",
        "validation_samples",
        "test_samples",
        "feature_names",
        "train_metrics",
        "validation_metrics",
        "test_metrics",
        "status",
        "started_at",
        "completed_at",
        "duration_seconds",
        "is_best_trial",
        "is_synthetic",
    ],
    "ml_deployments": [
        "id",
        "model_registry_id",
        "deployment_name",
        "environment",
        "endpoint_name",
        "status",
        "deployed_by",
        "deployment_config",
        "production_metrics",
        "created_at",
        "deployed_at",
        "latency_p50_ms",
        "latency_p95_ms",
        "error_rate",
        "is_synthetic",
    ],
    "ab_experiment_assignments": [
        "id",
        "experiment_id",
        "unit_id",
        "unit_type",
        "variant",
        "assigned_at",
        "randomization_method",
        "stratification_key",
        "assignment_hash",
        "created_by",
        "is_synthetic",
    ],
    "ab_experiment_enrollments": [
        "id",
        "assignment_id",
        "enrolled_at",
        "enrollment_status",
        "eligibility_criteria_met",
        "eligibility_check_timestamp",
        "is_synthetic",
    ],
    "ab_experiment_results": [
        "id",
        "experiment_id",
        "analysis_type",
        "analysis_method",
        "computed_at",
        "primary_metric",
        "control_mean",
        "control_std",
        "control_n",
        "treatment_mean",
        "treatment_std",
        "treatment_n",
        "effect_estimate",
        "effect_type",
        "effect_ci_lower",
        "effect_ci_upper",
        "confidence_level",
        "p_value",
        "is_significant",
        "observed_power",
        "is_synthetic",
    ],
    "ml_observability_spans": [
        "id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "agent_name",
        "agent_tier",
        "operation_type",
        "started_at",
        "ended_at",
        "duration_ms",
        "model_name",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "status",
        "fallback_used",
        "attributes",
        "is_synthetic",
    ],
    "causal_paths": [
        "path_id",
        "discovery_date",
        "causal_chain",
        "start_node",
        "end_node",
        "intermediate_nodes",
        "path_length",
        "causal_effect_size",
        "confidence_level",
        "method_used",
        "confounders_controlled",
        "mediators_identified",
        "time_lag_days",
        "validation_status",
        "business_impact_estimate",
        "data_split",
        "direct_effect",
        "indirect_effect",
        "brand",
        "region",
        "confirmation_count",
        "created_at",
        "is_synthetic",
    ],
    "learning_signals": [
        "signal_id",
        "signal_type",
        "signal_value",
        "signal_details",
        "applies_to_type",
        "applies_to_id",
        "brand",
        "region",
        "rated_agent",
        "is_training_example",
        "dspy_metric_name",
        "dspy_metric_value",
        "training_input",
        "training_output",
        "reward",
        "created_at",
        "is_synthetic",
    ],
    "user_sessions": [
        "session_id",
        "user_id",
        "user_role",
        "user_region",
        "session_start",
        "session_end",
        "session_duration_seconds",
        "page_views",
        "queries_executed",
        "actions_taken",
        "engagement_score",
        "created_at",
        "is_synthetic",
    ],
    "hcp_intent_surveys": [
        "survey_id",
        "hcp_id",
        "survey_date",
        "survey_type",
        "brand",
        "intent_to_prescribe_score",
        "intent_to_prescribe_change",
        "awareness_score",
        "favorability_score",
        "previous_survey_id",
        "days_since_last_survey",
        "survey_source",
        "created_at",
        "is_synthetic",
    ],
    "data_source_tracking": [
        "tracking_id",
        "tracking_date",
        "source_name",
        "source_type",
        "records_received",
        "records_matched",
        "records_unique",
        "match_rate_vs_iqvia",
        "match_rate_vs_healthverity",
        "match_rate_vs_komodo",
        "match_rate_vs_veeva",
        "stacking_eligible_records",
        "stacking_applied_records",
        "stacking_lift_percentage",
        "data_quality_score",
        "created_at",
        "is_synthetic",
    ],
    "etl_pipeline_metrics": [
        "pipeline_run_id",
        "pipeline_name",
        "pipeline_version",
        "run_start",
        "run_end",
        "duration_seconds",
        "source_data_date",
        "source_data_timestamp",
        "time_to_release_hours",
        "records_processed",
        "records_failed",
        "status",
        "quality_checks_passed",
        "quality_checks_failed",
        "created_at",
        "is_synthetic",
    ],
    "ml_annotations": [
        "annotation_id",
        "entity_type",
        "entity_id",
        "annotation_type",
        "annotator_id",
        "annotator_role",
        "annotation_value",
        "annotation_confidence",
        "annotation_timestamp",
        "is_adjudicated",
        "iaa_group_id",
        "created_at",
        "is_synthetic",
    ],
}


# Registered-but-OPTIONAL columns: nullable analytical / enrichment fields that the
# load script stamps via separate, order-independent enrichment passes (e.g.
# data_lag.stamp_data_lag_hours, sequence_number.stamp_sequence_number) or that are
# only populated for a subset of rows (ml_predictions quality metrics, trigger
# change-event fields). They are column-GATED into the load when present, but their
# ABSENCE is not a critical validation error — the DB columns are nullable, so a base
# dataset that has not been enriched still loads cleanly. validate_datasets exempts
# them from critical_missing, mirroring the existing ``_split`` exemption. (The core
# causal-substrate columns — treatment_arm/propensity_score/etc. — are deliberately
# NOT here: PatientGenerator emits them as NULL placeholders, so they remain required.)
OPTIONAL_COLUMNS = frozenset(
    {
        "data_lag_hours",  # patient_journeys (Shard 09 WS1-DQ-007 enrichment stamp)
        "sequence_number",  # treatment_events (NRx ordering enrichment stamp)
        # treatment_events PNH lab columns (#1116 BR-003): populated only on
        # pnh_flow_cytometry rows; other treatment_events producers (e.g. the
        # injected conversion prescriptions) legitimately omit them.
        "loinc_codes",
        "lab_values",
        # ml_predictions model-quality metrics (nullable; only on evaluated predictions)
        "model_auc",
        "model_pr_auc",
        "model_precision",
        "model_recall",
        "brier_score",
        "calibration_score",
        "rank_metrics",
        "fairness_metrics",
        "shap_values",
        "counterfactual_outcome",
        # triggers change-event fields (nullable; only on change-type triggers)
        "previous_trigger_id",
        "change_type",
        "change_failed",
        "change_outcome_delta",
    }
)


# Per-table conflict targets for the upsert (#852). The default PostgREST upsert
# conflicts on the PRIMARY KEY (``id``). For the feature-store tables that is WRONG:
# the seeder mints fresh random ids every run, but the four canonical group names
# (and their features) already exist in the DB with DIFFERENT ids. An upsert-by-id
# therefore INSERTs and collides with the *secondary* UNIQUE constraints
# (feature_groups_name_key, unique_feature_per_group, feature_entity_timestamp_unique)
# -> 23505 -> the whole batch fails -> 0 loaded -> children orphaned. Declaring the
# NATURAL KEY as the conflict target makes the upsert idempotent (INSERT-or-UPDATE on
# the business key), mirroring src/feature_store/client.py:285. Tables absent from this
# map keep the default (conflict-on-PK) behaviour.
TABLE_ON_CONFLICT = {
    "feature_groups": "name",  # feature_groups_name_key UNIQUE(name)
    "features": "feature_group_id,name",  # unique_feature_per_group
    "feature_values": "feature_id,entity_values,event_timestamp",  # feature_entity_timestamp_unique
}


class BatchLoader:
    """
    Batch loader for synthetic data to Supabase.

    Features:
    - Respects foreign key loading order
    - Batch processing with configurable size
    - Retry logic for transient failures
    - Validation before loading
    - Dry run mode for testing
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the batch loader.

        Args:
            config: Loader configuration.
        """
        self.config = config or LoaderConfig()
        self._client = None

    @property
    def client(self):
        """Get or create Supabase client."""
        if self._client is None and not self.config.dry_run:
            try:
                from supabase import create_client

                self._client = create_client(
                    self.config.supabase_url,
                    self.config.supabase_key,
                )
            except ImportError:
                logger.warning("Supabase client not available")
            except Exception as e:
                logger.error(f"Failed to create Supabase client: {e}")
        return self._client

    def reconcile_feature_store_ids(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Remap generated feature-store ids onto the EXISTING DB ids by natural key,
        in place, so a re-load is idempotent AND FK-coherent (#852).

        The seeder mints fresh random UUIDs every run, but the four canonical feature
        groups and their 15 features already exist in the DB (registered out-of-band).
        Upserting on the natural key (see TABLE_ON_CONFLICT) is necessary but NOT
        sufficient: if the payload still carries a *fresh* ``id``, the ``ON CONFLICT
        (name) DO UPDATE`` tries to rewrite the existing row's PRIMARY KEY, which the
        ``features_feature_group_id_fkey`` constraint rejects (23503 — the old id is
        still referenced). So before loading we look up the existing ids and rewrite:

          * ``feature_groups.id`` for any group whose ``name`` already exists,
          * ``features.feature_group_id`` (to the reconciled group id) and
            ``features.id`` for any (group, name) that already exists,
          * ``feature_values.feature_id`` to the reconciled feature id.

        Genuinely-new rows keep their fresh ids. Mirrors the production feature-store
        client (src/feature_store/client.py), which looks up the parent group id before
        upserting features. No-op when the client is unavailable (dry-run/offline) or
        none of the three frames are present.
        """
        if self.config.dry_run or self.client is None:
            return
        if not any(t in datasets for t in ("feature_groups", "features", "feature_values")):
            return

        # 1) feature_groups: name -> existing id
        existing_group_id_by_name: Dict[str, str] = {}
        try:
            resp = self.client.table("feature_groups").select("id,name").execute()
            for row in resp.data or []:
                existing_group_id_by_name[row["name"]] = row["id"]
        except Exception as e:  # pragma: no cover - network/permission edge
            logger.warning(f"feature-store reconcile: could not read feature_groups: {e}")
            return

        # group_id remap: generated group id -> reconciled (existing-or-fresh) id
        group_id_remap: Dict[str, str] = {}
        if "feature_groups" in datasets:
            fg = datasets["feature_groups"]
            if "id" in fg.columns and "name" in fg.columns:
                for idx, row in fg.iterrows():
                    existing = existing_group_id_by_name.get(row["name"])
                    if existing and existing != row["id"]:
                        group_id_remap[row["id"]] = existing
                        fg.at[idx, "id"] = existing
                # Refresh the name->id map so child features resolve against the
                # reconciled (now authoritative) group ids, including new groups.
                for _, row in fg.iterrows():
                    existing_group_id_by_name.setdefault(row["name"], row["id"])
                    existing_group_id_by_name[row["name"]] = row["id"]

        # 2) features: re-point feature_group_id to reconciled group ids, then map
        #    (feature_group_id, name) -> existing feature id.
        #
        # FAIL-CLOSED: if this lookup fails AFTER the group-id remap above, we cannot
        # reconcile feature ids and would silently fall back to the buggy "fresh
        # feature id on a natural-key upsert" path (-> 23503/23505 -> 0 loaded). Raise
        # so the caller fails loudly instead of fabricating a partial, FK-incoherent
        # load. (The earlier feature_groups read returns before any mutation on error.)
        existing_feature_id_by_key: Dict[Tuple[str, str], str] = {}
        try:
            resp = self.client.table("features").select("id,feature_group_id,name").execute()
            for row in resp.data or []:
                existing_feature_id_by_key[(row["feature_group_id"], row["name"])] = row["id"]
        except Exception as e:  # pragma: no cover - network/permission edge
            logger.error(f"feature-store reconcile: could not read features: {e}")
            raise

        feature_id_remap: Dict[str, str] = {}
        if "features" in datasets:
            ft = datasets["features"]
            if {"id", "feature_group_id", "name"}.issubset(ft.columns):
                for idx, row in ft.iterrows():
                    # repoint FK to reconciled group id first
                    gid = group_id_remap.get(row["feature_group_id"], row["feature_group_id"])
                    if gid != row["feature_group_id"]:
                        ft.at[idx, "feature_group_id"] = gid
                    existing = existing_feature_id_by_key.get((gid, row["name"]))
                    if existing and existing != row["id"]:
                        feature_id_remap[row["id"]] = existing
                        ft.at[idx, "id"] = existing

        # 3) feature_values: re-point feature_id to reconciled feature ids.
        if "feature_values" in datasets and feature_id_remap:
            fv = datasets["feature_values"]
            if "feature_id" in fv.columns:
                fv["feature_id"] = fv["feature_id"].map(lambda fid: feature_id_remap.get(fid, fid))

        if group_id_remap or feature_id_remap:
            logger.info(
                "feature-store reconcile: remapped %d group ids, %d feature ids to "
                "existing DB rows (idempotent, FK-coherent)",
                len(group_id_remap),
                len(feature_id_remap),
            )

    def load_all(
        self,
        datasets: Dict[str, pd.DataFrame],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, LoadResult]:
        """
        Load all datasets in dependency order.

        Args:
            datasets: Dictionary of table_name -> DataFrame.
            progress_callback: Optional callback(table, current, total).

        Returns:
            Dictionary of table_name -> LoadResult.
        """
        results = {}

        # #852: reconcile feature-store ids against existing DB rows BEFORE loading so
        # a re-run upserts idempotently instead of colliding (23505) or rewriting a
        # referenced PK (23503). No-op in dry-run/offline.
        self.reconcile_feature_store_ids(datasets)

        # Determine tables to load
        tables_to_load = [t for t in LOADING_ORDER if t in datasets]
        total_tables = len(tables_to_load)

        for i, table_name in enumerate(tables_to_load):
            df = datasets[table_name]

            if progress_callback:
                progress_callback(table_name, i + 1, total_tables)

            if self.config.verbose:
                logger.info(f"Loading {table_name} ({len(df)} records)")

            result = self.load_table(table_name, df)
            results[table_name] = result

            if not result.is_success:
                logger.warning(
                    f"Table {table_name} load had failures: "
                    f"{result.records_failed}/{result.records_loaded + result.records_failed}"
                )

        return results

    def load_table(
        self,
        table_name: str,
        df: pd.DataFrame,
    ) -> LoadResult:
        """
        Load a single table in batches.

        Args:
            table_name: Target table name.
            df: DataFrame to load.

        Returns:
            LoadResult with statistics.
        """
        start_time = datetime.now()
        records_loaded = 0
        records_failed = 0
        failed_batches = []
        errors = []

        # Select and order columns
        available_columns = TABLE_COLUMNS.get(table_name, list(df.columns))
        columns_to_load = [c for c in available_columns if c in df.columns]
        df_to_load = df[columns_to_load].copy()

        # Handle JSON columns
        json_columns = ["causal_chain", "supporting_evidence"]
        for col in json_columns:
            if col in df_to_load.columns:
                df_to_load[col] = df_to_load[col].apply(
                    lambda x: x if isinstance(x, (dict, list)) else {}
                )

        # Handle None/NaN values - replace with None for JSON compatibility
        import numpy as np

        df_to_load = df_to_load.replace({np.nan: None, np.inf: None, -np.inf: None})
        df_to_load = df_to_load.where(pd.notnull(df_to_load), None)

        # Drop intra-frame duplicate conflict keys for natural-key-conflict tables.
        # PostgREST issues one `INSERT ... ON CONFLICT (<key>) DO UPDATE` per batch;
        # Postgres rejects the whole statement with 21000 ("ON CONFLICT DO UPDATE
        # command cannot affect row a second time") if two rows in the batch share the
        # conflict key. feature_values hits this because low-cardinality entity_values
        # (brand/region: 3-4 distinct) collide on (feature_id, entity_values,
        # event_timestamp) -> ~1 batch in 7 fails -> ~9% silent loss. Drop dups here
        # (keep="last" mirrors DO UPDATE's last-write-wins) so every batch is conflict-
        # free. No-op for tables without an on_conflict target (the bulk fact tables).
        on_conflict = TABLE_ON_CONFLICT.get(table_name)
        if on_conflict:
            conflict_cols = [c for c in on_conflict.split(",") if c in df_to_load.columns]
            if conflict_cols:
                # entity_values is jsonb (unhashable dict). Serialize for the dedup key;
                # sort_keys makes it key-order-independent, matching Postgres jsonb equality.
                dedup_key = df_to_load[conflict_cols].apply(
                    lambda col: col.map(
                        lambda v: json.dumps(v, sort_keys=True, default=str)
                        if isinstance(v, (dict, list))
                        else v
                    )
                )
                dup_mask = dedup_key.duplicated(keep="last")
                n_dup = int(dup_mask.sum())
                if n_dup:
                    df_to_load = df_to_load[~dup_mask]
                    logger.info(
                        "%s: dropped %d intra-frame duplicate (%s) conflict-key rows "
                        "before upsert (avoids PostgREST 21000)",
                        table_name,
                        n_dup,
                        on_conflict,
                    )

        # Calculate batches
        total_records = len(df_to_load)
        batch_size = self.config.batch_size
        total_batches = (total_records + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_records)
            batch_df = df_to_load.iloc[start_idx:end_idx]

            success, error = self._load_batch(table_name, batch_df, batch_idx)

            if success:
                records_loaded += len(batch_df)
            else:
                records_failed += len(batch_df)
                failed_batches.append(batch_idx)
                if error:
                    errors.append(f"Batch {batch_idx}: {error}")

        duration = (datetime.now() - start_time).total_seconds()

        return LoadResult(
            table_name=table_name,
            records_loaded=records_loaded,
            records_failed=records_failed,
            total_batches=total_batches,
            failed_batches=failed_batches,
            errors=errors,
            duration_seconds=duration,
        )

    def _load_batch(
        self,
        table_name: str,
        batch_df: pd.DataFrame,
        batch_idx: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Load a single batch with retry logic.

        Args:
            table_name: Target table name.
            batch_df: Batch DataFrame.
            batch_idx: Batch index for logging.

        Returns:
            Tuple of (success, error_message).
        """
        if self.config.dry_run:
            if self.config.verbose:
                logger.info(f"[DRY RUN] Would load {len(batch_df)} records to {table_name}")
            return True, None

        if not self.client:
            return False, "Supabase client not available"

        records = batch_df.to_dict(orient="records")

        # Coerce integral floats -> int. pandas upcasts integer columns to float
        # whenever a row carries NaN (missingness), so e.g. age_at_diagnosis renders
        # as "13.0", which a Postgres integer/smallint column rejects (22P02). int is
        # also accepted by double-precision columns, so this is safe for both.
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and v.is_integer():
                    rec[k] = int(v)

        # #852: feature-store tables must conflict on their NATURAL key (name /
        # feature_group_id,name / feature_id,entity_values,event_timestamp), not the
        # default PK, or a re-load collides with the existing canonical rows -> 23505.
        on_conflict = TABLE_ON_CONFLICT.get(table_name)

        for attempt in range(self.config.max_retries):
            try:
                table = self.client.table(table_name)
                if on_conflict is not None:
                    table.upsert(records, on_conflict=on_conflict).execute()
                else:
                    table.upsert(records).execute()
                return True, None
            except Exception as e:
                error_msg = str(e)
                if attempt < self.config.max_retries - 1:
                    import time

                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    logger.error(
                        f"Batch {batch_idx} failed after {self.config.max_retries} attempts: {error_msg}"
                    )
                    return False, error_msg

        return False, "Unknown error"

    def validate_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
    ) -> Tuple[bool, List[str]]:
        """
        Validate datasets before loading.

        Args:
            datasets: Dictionary of table_name -> DataFrame.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []

        for table_name, df in datasets.items():
            # Check if table is known
            if table_name not in TABLE_COLUMNS:
                errors.append(f"Unknown table: {table_name}")
                continue

            # Check required columns
            required_columns = TABLE_COLUMNS[table_name]
            missing_columns = [c for c in required_columns if c not in df.columns]

            # Some columns are optional: data_split for static entities, plus the
            # nullable analytical/enrichment columns in OPTIONAL_COLUMNS (stamped by
            # order-independent enrichment passes or populated for a subset of rows).
            critical_missing = [
                c for c in missing_columns if not c.endswith("_split") and c not in OPTIONAL_COLUMNS
            ]
            if critical_missing:
                errors.append(f"{table_name}: Missing columns {critical_missing}")

            # Check for empty DataFrame
            if len(df) == 0:
                errors.append(f"{table_name}: Empty DataFrame")

        return len(errors) == 0, errors

    def get_loading_summary(
        self,
        results: Dict[str, LoadResult],
    ) -> str:
        """
        Generate a summary of loading results.

        Args:
            results: Dictionary of table_name -> LoadResult.

        Returns:
            Formatted summary string.
        """
        lines = ["=" * 60, "SYNTHETIC DATA LOADING SUMMARY", "=" * 60]

        total_loaded = 0
        total_failed = 0
        total_duration = 0.0

        for table_name in LOADING_ORDER:
            if table_name not in results:
                continue

            result = results[table_name]
            total_loaded += result.records_loaded
            total_failed += result.records_failed
            total_duration += result.duration_seconds

            status = "✓" if result.is_success else "✗"
            lines.append(
                f"{status} {table_name}: "
                f"{result.records_loaded:,} loaded, "
                f"{result.records_failed:,} failed "
                f"({result.success_rate:.1%}) "
                f"[{result.duration_seconds:.1f}s]"
            )

        lines.append("-" * 60)
        overall_rate = (
            total_loaded / (total_loaded + total_failed) if (total_loaded + total_failed) > 0 else 0
        )
        lines.append(
            f"TOTAL: {total_loaded:,} loaded, {total_failed:,} failed "
            f"({overall_rate:.1%}) [{total_duration:.1f}s]"
        )
        lines.append("=" * 60)

        return "\n".join(lines)


class AsyncBatchLoader(BatchLoader):
    """
    Async version of batch loader for concurrent loading.

    Use when loading large datasets where parallelism can help.
    """

    async def load_all_async(
        self,
        datasets: Dict[str, pd.DataFrame],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, LoadResult]:
        """
        Load all datasets asynchronously (but respecting order).

        Args:
            datasets: Dictionary of table_name -> DataFrame.
            progress_callback: Optional callback(table, current, total).

        Returns:
            Dictionary of table_name -> LoadResult.
        """
        # For now, just run synchronously since Supabase sync client
        # In future, could use async client for true async loading
        return self.load_all(datasets, progress_callback)

    async def load_table_async(
        self,
        table_name: str,
        df: pd.DataFrame,
    ) -> LoadResult:
        """
        Load a single table asynchronously.

        Args:
            table_name: Target table name.
            df: DataFrame to load.

        Returns:
            LoadResult with statistics.
        """
        # Run in executor for non-blocking behavior
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.load_table(table_name, df))
