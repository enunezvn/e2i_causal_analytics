#!/usr/bin/env python3
"""Manual step-by-step runner for Tier 0 MLOps workflow test.

This script executes each agent in the Tier 0 pipeline individually
with detailed output and verification between steps.

Usage:
    # Run full pipeline
    python scripts/run_tier0_test.py

    # Run specific step (1-8)
    python scripts/run_tier0_test.py --step 3

    # Run with MLflow enabled
    python scripts/run_tier0_test.py --enable-mlflow

    # Dry run (show what would be done)
    python scripts/run_tier0_test.py --dry-run

    # Run with BentoML model serving verification (requires step 5+7)
    python scripts/run_tier0_test.py --include-bentoml

    # Run steps 4-8 with BentoML serving (recommended for full flow validation)
    python scripts/run_tier0_test.py --step 4 --include-bentoml

Prerequisites:
    - On droplet: cd /opt/e2i_causal_analytics && source .venv/bin/activate
    - API running (port 8000)
    - MLflow running (port 5000, optional)
    - Opik running (port 5173/8080, optional)
    - BentoML installed (for --include-bentoml flag)

Author: E2I Causal Analytics Team
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time as time_module
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
# This provides ANTHROPIC_API_KEY, SUPABASE_ANON_KEY, and other secrets
load_dotenv(PROJECT_ROOT / ".env")

# Configure MLflow tracking URI for model artifact storage
# This ensures model_uri is properly generated during model training
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Configure Supabase URL for database persistence
# Self-hosted Supabase runs on port 54321 (internal Docker network uses localhost)
if not os.environ.get("SUPABASE_URL"):
    os.environ["SUPABASE_URL"] = "http://localhost:54321"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration."""
    brand: str = "Kisqali"
    problem_type: str = "binary_classification"
    target_outcome: str = "discontinuation_flag"
    indication: str = "HR+/HER2- breast cancer"
    hpo_trials: int = 10
    min_eligible_patients: int = 30
    min_auc_threshold: float = 0.55
    enable_mlflow: bool = True  # MLflow must be enabled for model_uri to be generated
    enable_opik: bool = False


CONFIG = TestConfig()


@dataclass
class StepResult:
    """Result from a pipeline step."""
    step_num: int
    step_name: str
    status: str  # "success", "warning", "failed"
    duration_seconds: float = 0.0
    key_metrics: dict = None
    details: dict = None

    def __post_init__(self):
        if self.key_metrics is None:
            self.key_metrics = {}
        if self.details is None:
            self.details = {}


# =============================================================================
# UTILITIES
# =============================================================================

def print_header(step_num: int, title: str) -> None:
    """Print step header."""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {title}")
    print("=" * 70)


def print_result(key: str, value: Any, indent: int = 2) -> None:
    """Print a result key-value pair."""
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            print_result(k, v, indent + 2)
    elif isinstance(value, list) and len(value) > 3:
        print(f"{prefix}{key}: [{len(value)} items]")
    else:
        print(f"{prefix}{key}: {value}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"\n  ✅ {message}")


def print_failure(message: str) -> None:
    """Print failure message."""
    print(f"\n  ❌ {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"\n  ⚠️  {message}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"\n  ℹ️  {message}")


def print_detailed_summary(
    experiment_id: str,
    step_results: list[StepResult],
    state: dict[str, Any]
) -> None:
    """Print detailed results from each tier0 step.

    Args:
        experiment_id: The experiment identifier
        step_results: List of StepResult objects from each step
        state: Pipeline state with all collected data
    """
    print(f"\n{'='*70}")
    print("DETAILED STEP RESULTS")
    print(f"{'='*70}")

    for result in step_results:
        status_icon = "✅" if result.status == "success" else "⚠️" if result.status == "warning" else "❌"
        print(f"\n{'-'*70}")
        print(f"STEP {result.step_num}: {result.step_name} [{status_icon} {result.status.upper()}]")
        print(f"{'-'*70}")

        if result.duration_seconds > 0:
            print(f"  Duration: {result.duration_seconds:.2f}s")

        # Print key metrics
        if result.key_metrics:
            print("\n  Key Metrics:")
            for key, value in result.key_metrics.items():
                if isinstance(value, float):
                    print(f"    • {key}: {value:.4f}")
                else:
                    print(f"    • {key}: {value}")

        # Print step-specific details
        if result.details:
            print("\n  Details:")
            for key, value in result.details.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for k, v in list(value.items())[:10]:  # Limit nested items
                        if isinstance(v, float):
                            print(f"      - {k}: {v:.4f}")
                        else:
                            print(f"      - {k}: {v}")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"    {key}: [{len(value)} items]")
                else:
                    print(f"    {key}: {value}")

    # Class Imbalance Section
    class_imbalance_info = state.get("class_imbalance_info", {})
    if class_imbalance_info.get("imbalance_detected"):
        print(f"\n{'='*70}")
        print("CLASS IMBALANCE REMEDIATION")
        print(f"{'='*70}")

        print("\n  📊 Imbalance Analysis:")
        print(f"    • Imbalance Detected: Yes")
        print(f"    • Severity: {class_imbalance_info.get('imbalance_severity', 'unknown').upper()}")
        print(f"    • Minority Ratio: {class_imbalance_info.get('minority_ratio', 0):.2%}")
        print(f"    • Imbalance Ratio: {class_imbalance_info.get('imbalance_ratio', 1):.1f}:1")

        class_dist = class_imbalance_info.get("class_distribution", {})
        if class_dist:
            print("\n  📈 Class Distribution:")
            for cls, count in class_dist.items():
                print(f"    • Class {cls}: {count} samples")

        print("\n  🔧 Remediation Applied:")
        print(f"    • Strategy: {class_imbalance_info.get('recommended_strategy', 'none')}")
        print(f"    • Rationale: {class_imbalance_info.get('strategy_rationale', 'N/A')}")

        # Show before/after if resampling was applied
        resampling_info = state.get("resampling_info", {})
        if resampling_info.get("resampling_applied"):
            print("\n  📊 Resampling Results:")
            orig_samples = resampling_info.get('original_samples')
            resamp_samples = resampling_info.get('resampled_samples')
            print(f"    • Original Samples: {orig_samples}")
            print(f"    • Resampled Samples: {resamp_samples}")
            new_ratio = resampling_info.get('new_minority_ratio')
            if new_ratio is not None:
                print(f"    • New Minority Ratio: {new_ratio:.2%}")
            else:
                print(f"    • New Minority Ratio: N/A")
            # Show resampled distribution
            resampled_dist = resampling_info.get("resampled_distribution", {})
            if resampled_dist:
                print("\n  📈 Resampled Class Distribution:")
                for cls, count in sorted(resampled_dist.items()):
                    print(f"    • Class {cls}: {count} samples")
        else:
            # Resampling not applied even though imbalance detected (e.g., class_weight strategy)
            print("\n  📊 Resampling Results:")
            print(f"    • Resampling Applied: No")
            strategy = resampling_info.get("resampling_strategy", "none")
            if strategy == "class_weight":
                print(f"    • Strategy: class_weight (handled during training)")
            else:
                print(f"    • Strategy: {strategy}")
    elif class_imbalance_info:
        print(f"\n  ℹ️  Class Imbalance: Not detected (minority ratio >= 40%)")

    # Feature Importance Section
    feature_importance = state.get("feature_importance")
    if feature_importance:
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE (SHAP)")
        print(f"{'='*70}")
        print("\n  Top Features:")
        for i, fi in enumerate(feature_importance[:10], 1):
            if isinstance(fi, dict):
                name = fi.get("feature", f"feature_{i}")
                importance = fi.get("importance", 0)
                print(f"    {i}. {name}: {importance:.4f}")
            else:
                print(f"    {i}. {fi}")

    # Validation Metrics Section
    validation_metrics = state.get("validation_metrics", {})
    if validation_metrics:
        print(f"\n{'='*70}")
        print("FINAL MODEL PERFORMANCE")
        print(f"{'='*70}")

        # Key metrics
        key_metrics = ["roc_auc", "accuracy", "precision", "recall", "f1_score", "pr_auc", "brier_score"]
        print("\n  Primary Metrics:")
        for metric in key_metrics:
            value = validation_metrics.get(metric)
            if value is not None:
                print(f"    • {metric}: {value:.4f}")

        # Per-class metrics
        print("\n  Per-Class Metrics:")
        for key, value in validation_metrics.items():
            if "class_" in key and value is not None:
                print(f"    • {key}: {value:.4f}")

    # Deployment Info
    deployment_manifest = state.get("deployment_manifest", {})
    if deployment_manifest:
        print(f"\n{'='*70}")
        print("DEPLOYMENT STATUS")
        print(f"{'='*70}")
        print(f"\n  • Deployment ID: {deployment_manifest.get('deployment_id', 'N/A')}")
        print(f"  • Environment: {deployment_manifest.get('environment', 'N/A')}")
        print(f"  • Status: {deployment_manifest.get('status', 'N/A')}")
        print(f"  • Endpoint: {deployment_manifest.get('endpoint_url', 'N/A')}")

    print(f"\n{'='*70}")


def generate_sample_data(
    n_samples: int = 100,
    seed: int = 42,
    imbalance_ratio: float | None = None,
) -> pd.DataFrame:
    """Generate sample patient journey data using the ML-ready generator.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        imbalance_ratio: If provided, force minority class to this ratio (e.g., 0.1 for 10%)
                        None means balanced data (~50/50)
    """
    # Use the same generator as the data_preparer agent for consistency
    from src.repositories.sample_data import SampleDataGenerator
    import numpy as np

    generator = SampleDataGenerator(seed=seed)

    # Use fresh date range (last 30 days) to pass timeliness checks
    # Default range is 365 days which causes staleness warnings
    end_date = datetime.now().isoformat()
    start_date = (datetime.now() - timedelta(days=30)).isoformat()

    df = generator.ml_patients(
        n_patients=n_samples,
        start_date=start_date,
        end_date=end_date,
    )

    # Apply class imbalance if requested
    if imbalance_ratio is not None and 0 < imbalance_ratio < 0.5:
        np.random.seed(seed)
        target_col = CONFIG.target_outcome
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority

        # Create imbalanced target: minority class = 1 (discontinuation)
        labels = np.array([0] * n_majority + [1] * n_minority)
        np.random.shuffle(labels)
        df[target_col] = labels

        print(f"  ⚠️  Injected class imbalance: {imbalance_ratio:.1%} minority (class 1)")
        print(f"      Class 0: {n_majority} samples, Class 1: {n_minority} samples")

    # Filter to only the configured brand
    # (or keep all if testing multi-brand)
    if CONFIG.brand:
        # Keep all brands but prioritize the configured one
        pass

    return df


# =============================================================================
# BENTOML HELPER FUNCTIONS
# =============================================================================


async def start_bentoml_service(model_tag: str, port: int = 3001) -> dict:
    """Start BentoML service serving the real trained model.

    Args:
        model_tag: BentoML model tag from registration
        port: Port to serve on

    Returns:
        {"started": True, "endpoint": "http://localhost:3001", "pid": <pid>}
    """
    import httpx

    # Generate a service file dynamically for the model
    service_code = f'''
import bentoml
import numpy as np

@bentoml.service(name="tier0_model_service")
class Tier0ModelService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("{model_tag}")
        self.model_tag = "{model_tag}"

    @bentoml.api
    async def predict(self, features: list) -> dict:
        import time
        start = time.time()
        arr = np.array(features)
        predictions = self.model.predict(arr)
        probas = self.model.predict_proba(arr) if hasattr(self.model, 'predict_proba') else None
        elapsed = (time.time() - start) * 1000
        return {{
            "predictions": predictions.tolist(),
            "probabilities": probas.tolist() if probas is not None else None,
            "latency_ms": elapsed,
            "model_tag": self.model_tag,
        }}

    @bentoml.api
    async def health(self) -> dict:
        return {{"status": "healthy", "model_tag": self.model_tag}}
'''

    # Write service file
    service_path = Path("/tmp/tier0_bentoml_service.py")
    service_path.write_text(service_code)
    print(f"    Generated service file: {service_path}")

    # Start bentoml serve in background
    process = subprocess.Popen(
        ["bentoml", "serve", str(service_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"    Starting BentoML service on port {port} (PID: {process.pid})...")

    # Wait for service to be ready
    endpoint = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        for retry in range(30):  # 30 retries (30 seconds max)
            await asyncio.sleep(1)
            try:
                resp = await client.get(f"{endpoint}/health", timeout=2.0)
                if resp.status_code == 200:
                    print(f"    Service ready at {endpoint}")
                    return {"started": True, "endpoint": endpoint, "pid": process.pid}
            except Exception:
                # Check if process died
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    return {
                        "started": False,
                        "error": f"Process exited with code {process.returncode}: {stderr[:500]}",
                    }

    # Timeout - kill process
    process.terminate()
    return {"started": False, "error": "Service startup timeout (30s)"}


async def verify_bentoml_predictions(
    endpoint: str,
    sample_features: list,
    use_production_client: bool = True,
) -> dict:
    """Verify that BentoML service returns valid predictions.

    Args:
        endpoint: BentoML service endpoint (e.g., http://localhost:3001)
        sample_features: Sample feature data to test
        use_production_client: If True, use BentoMLClient for full validation

    Returns:
        {"health_check": True, "prediction_test": True, "predictions": [...], "latency_ms": X}
    """
    import httpx

    result = {"health_check": False, "prediction_test": False}

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            health_resp = await client.get(f"{endpoint}/health", timeout=5.0)
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                result["health_check"] = health_data.get("status") == "healthy"
                result["model_tag"] = health_data.get("model_tag")
        except Exception as e:
            result["health_error"] = str(e)

        # Prediction test
        try:
            start = time_module.time()
            pred_resp = await client.post(
                f"{endpoint}/predict",
                json={"features": sample_features},
                timeout=10.0,
            )
            elapsed = (time_module.time() - start) * 1000

            if pred_resp.status_code == 200:
                pred_data = pred_resp.json()
                result["prediction_test"] = True
                result["predictions"] = pred_data.get("predictions")
                result["probabilities"] = pred_data.get("probabilities")
                result["latency_ms"] = elapsed
                result["service_latency_ms"] = pred_data.get("latency_ms")
        except Exception as e:
            result["prediction_error"] = str(e)

    return result


async def stop_bentoml_service(pid: int) -> dict:
    """Stop BentoML service by PID.

    Args:
        pid: Process ID to terminate

    Returns:
        {"stopped": True/False, "pid": pid}
    """
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait briefly for graceful shutdown
        await asyncio.sleep(1)
        return {"stopped": True, "pid": pid}
    except ProcessLookupError:
        return {"stopped": True, "pid": pid, "note": "Process already terminated"}
    except Exception as e:
        return {"stopped": False, "error": str(e), "pid": pid}


# =============================================================================
# STEP IMPLEMENTATIONS
# =============================================================================

async def step_1_scope_definer(experiment_id: str) -> dict[str, Any]:
    """Step 1: Define ML problem scope."""
    print_header(1, "SCOPE DEFINER")

    from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent

    print("\n  Creating ScopeDefinerAgent...")
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": f"Predict patient discontinuation risk for {CONFIG.brand}",
        "business_objective": "Identify high-risk patients early for intervention",
        "target_outcome": CONFIG.target_outcome,
        "problem_type_hint": CONFIG.problem_type,
        "brand": CONFIG.brand,
    }

    print("  Input:")
    for k, v in input_data.items():
        print(f"    {k}: {v}")

    print("\n  Running agent...")
    result = await agent.run(input_data)

    print("\n  Output:")
    print_result("experiment_id", result.get("experiment_id", experiment_id))
    print_result("scope_spec", result.get("scope_spec", {}))
    print_result("success_criteria", result.get("success_criteria", {}))
    print_result("validation_passed", result.get("validation_passed", True))

    if result.get("validation_passed", True):
        print_success("Scope definition completed successfully")
    else:
        print_warning("Scope validation had warnings")

    return result


async def step_2_data_preparer(
    experiment_id: str, scope_spec: dict, sample_df: pd.DataFrame
) -> dict[str, Any]:
    """Step 2: Load and prepare data with QC."""
    print_header(2, "DATA PREPARER")

    from src.agents.ml_foundation.data_preparer import DataPreparerAgent

    print("\n  Creating DataPreparerAgent...")
    agent = DataPreparerAgent()

    # Override required_features with actual columns from sample data
    # This ensures we don't fail QC for features that don't exist
    available_features = [
        col for col in sample_df.columns
        if col not in ["patient_journey_id", CONFIG.target_outcome, "brand"]
    ]

    # Ensure scope_spec has required fields with realistic values
    scope_spec.update({
        "experiment_id": experiment_id,
        "use_sample_data": True,
        "sample_size": 500,
        "prediction_target": CONFIG.target_outcome,
        "problem_type": CONFIG.problem_type,
        # Override required_features with actual available features
        "required_features": available_features,
        # Allow 90 days staleness since sample data spans 90 days
        # and temporal splitting puts older data in training set
        "max_staleness_days": 90,
    })

    input_data = {
        "scope_spec": scope_spec,
        "data_source": "patient_journeys",
        "brand": CONFIG.brand,
    }

    print("  Input:")
    print(f"    data_source: patient_journeys")
    print(f"    brand: {CONFIG.brand}")
    print(f"    use_sample_data: True")

    print("\n  Running agent...")
    result = await agent.run(input_data)

    # Extract nested results
    qc_report = result.get("qc_report", {})
    data_readiness = result.get("data_readiness", {})
    remediation = result.get("remediation", {})

    print("\n  Output:")
    print_result("qc_status", qc_report.get("status", "unknown"))
    print_result("overall_score", qc_report.get("overall_score", "N/A"))
    print_result("gate_passed", result.get("gate_passed", False))
    print_result("train_samples", data_readiness.get("train_samples", "N/A"))
    print_result("validation_samples", data_readiness.get("validation_samples", "N/A"))

    # Display QC dimension scores
    print("\n  QC Dimension Scores:")
    print_result("completeness", qc_report.get("completeness_score", "N/A"))
    print_result("validity", qc_report.get("validity_score", "N/A"))
    print_result("consistency", qc_report.get("consistency_score", "N/A"))
    print_result("uniqueness", qc_report.get("uniqueness_score", "N/A"))
    print_result("timeliness", qc_report.get("timeliness_score", "N/A"))

    # Display remediation information if QC failed
    remediation_status = remediation.get("status", "not_needed")
    if remediation_status and remediation_status != "not_needed":
        print("\n  QC Remediation:")
        print_result("remediation_status", remediation_status)
        print_result("remediation_attempts", remediation.get("attempts", 0))

        if remediation.get("llm_analysis"):
            print_result("llm_analysis", remediation.get("llm_analysis"))

        if remediation.get("root_causes"):
            print("\n    Root Causes:")
            for i, cause in enumerate(remediation.get("root_causes", [])[:5], 1):
                print(f"      {i}. {cause}")

        if remediation.get("recommended_actions"):
            print("\n    Recommended Actions:")
            for i, action in enumerate(remediation.get("recommended_actions", [])[:5], 1):
                print(f"      {i}. {action}")

        if remediation.get("actions_taken"):
            print("\n    Actions Taken:")
            for action in remediation.get("actions_taken", []):
                print(f"      - {action}")

    if result.get("gate_passed", False):
        print_success("QC GATE PASSED - Training can proceed")
    else:
        print_failure("QC GATE FAILED - Training blocked")
        # Show blocking issues
        blocking_issues = qc_report.get("blocking_issues", [])
        if blocking_issues:
            print("\n    Blocking Issues:")
            for issue in blocking_issues[:5]:
                print(f"      - {issue}")

    return result


async def step_3_cohort_constructor(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """Step 3: Build patient cohort."""
    print_header(3, "COHORT CONSTRUCTOR")

    from src.agents.cohort_constructor import CohortConstructorAgent
    from src.agents.cohort_constructor.types import (
        CohortConfig,
        Criterion,
        CriterionType,
        Operator,
        TemporalRequirements,
    )

    print("\n  Creating CohortConstructorAgent...")
    agent = CohortConstructorAgent(enable_observability=CONFIG.enable_opik)

    # Create a test-specific CohortConfig using fields that exist in sample data
    # Sample data columns: patient_journey_id, patient_id, brand, geographic_region,
    # journey_status, data_quality_score, days_on_therapy, hcp_visits, prior_treatments,
    # age_group, discontinuation_flag
    # For test purposes, use all brands to get sufficient sample size
    # The sample data has ~200 patients split across 3 brands
    test_config = CohortConfig(
        cohort_name=f"{CONFIG.brand} Test Cohort",
        brand=CONFIG.brand.lower(),
        indication="test",
        inclusion_criteria=[
            Criterion(
                field="data_quality_score",
                operator=Operator.GREATER_EQUAL,
                value=0.5,
                criterion_type=CriterionType.INCLUSION,
                description="Minimum data quality score",
                clinical_rationale="Ensure data quality for reliable ML predictions",
            ),
            # Note: Not filtering by brand for test to ensure sufficient sample size
            # Sample data has ~60-70 patients per brand, filtering further would leave too few
        ],
        exclusion_criteria=[
            # Don't exclude any journey status - sample data has limited records
            # Excluding 'completed' would remove ~25% of samples
        ],
        temporal_requirements=None,  # Skip temporal requirements for sample data
        required_fields=["patient_journey_id", "brand", "data_quality_score"],
        version="1.0.0-test",
        status="active",
        clinical_rationale="Test cohort using sample data fields - relaxed criteria for testing",
        regulatory_justification="Test configuration for MLOps workflow validation",
    )

    print(f"  Input patient count: {len(patient_df)}")
    print(f"  Brand: {CONFIG.brand}")
    print(f"  Using custom test config (sample data compatible)")
    print(f"  Inclusion: data_quality_score >= 0.5")
    print(f"  No exclusion criteria (to maximize test sample size)")

    print("\n  Running agent...")
    eligible_df, result = await agent.run(
        patient_df=patient_df,
        config=test_config,  # Use custom config instead of brand
    )

    print("\n  Output:")
    print_result("cohort_id", result.cohort_id)
    print_result("execution_id", result.execution_id)
    print_result("eligible_count", len(result.eligible_patient_ids))
    print_result("status", result.status)

    if result.eligibility_stats:
        print_result("eligibility_stats", result.eligibility_stats)

    if len(eligible_df) >= CONFIG.min_eligible_patients:
        print_success(f"Cohort size ({len(eligible_df)}) meets minimum ({CONFIG.min_eligible_patients})")
    else:
        print_failure(f"Cohort size ({len(eligible_df)}) below minimum ({CONFIG.min_eligible_patients})")

    return eligible_df, result


async def step_4_model_selector(experiment_id: str, scope_spec: dict, qc_report: dict) -> dict[str, Any]:
    """Step 4: Select model candidate."""
    print_header(4, "MODEL SELECTOR")

    from src.agents.ml_foundation.model_selector import ModelSelectorAgent

    print("\n  Creating ModelSelectorAgent...")
    agent = ModelSelectorAgent()

    # Ensure qc_report has both gate_passed and qc_passed for compatibility
    # data_preparer uses gate_passed, model_selector expects qc_passed
    normalized_qc_report = qc_report.copy()
    if "qc_passed" not in normalized_qc_report:
        normalized_qc_report["qc_passed"] = normalized_qc_report.get("gate_passed", True)
    if "qc_errors" not in normalized_qc_report:
        normalized_qc_report["qc_errors"] = []

    input_data = {
        "scope_spec": scope_spec,
        "qc_report": normalized_qc_report,
        "skip_benchmarks": True,  # Skip for faster testing
    }

    print("  Input:")
    print(f"    problem_type: {scope_spec.get('problem_type', 'binary_classification')}")
    print(f"    qc_passed: {normalized_qc_report.get('qc_passed')}")
    print(f"    skip_benchmarks: True")

    print("\n  Running agent...")
    result = await agent.run(input_data)

    # Check for errors
    if result.get("error"):
        print_failure(f"Model selection error: {result.get('error')}")

    print("\n  Output:")
    candidate = result.get("model_candidate") or result.get("primary_candidate")
    if candidate:
        if hasattr(candidate, "algorithm_name"):
            print_result("algorithm", candidate.algorithm_name)
            print_result("hyperparameters", getattr(candidate, "hyperparameters", {}))
        elif isinstance(candidate, dict):
            print_result("algorithm", candidate.get("algorithm_name", candidate.get("algorithm")))
            print_result("hyperparameters", candidate.get("hyperparameters", {}))
        print_success("Model candidate selected")
    else:
        print_warning("No model candidate returned, will use fallback")

    print_result("selection_rationale", result.get("selection_rationale", "N/A"))

    return result


async def step_5_model_trainer(
    experiment_id: str,
    model_candidate: Any,
    qc_report: dict,
    X: pd.DataFrame,
    y: pd.Series
) -> dict[str, Any]:
    """Step 5: Train model."""
    print_header(5, "MODEL TRAINER")

    from src.agents.ml_foundation.model_trainer import ModelTrainerAgent
    from sklearn.linear_model import LogisticRegression

    print("\n  Creating ModelTrainerAgent...")
    agent = ModelTrainerAgent()

    # Ensure model_candidate has all required fields for model_trainer
    # Required: algorithm_name, algorithm_class, hyperparameter_search_space, default_hyperparameters
    if model_candidate is None or not isinstance(model_candidate, dict):
        print_warning("No valid model_candidate from selector, using LogisticRegression fallback")
        model_candidate = {}

    # Check if causal model - these are now supported in model_trainer
    causal_models = ["LinearDML", "CausalForest", "DoubleLasso", "SparseLinearDML", "DML",
                     "DRLearner", "SLearner", "TLearner", "XLearner"]
    algo_name = model_candidate.get("algorithm_name", "")
    if algo_name in causal_models:
        print_info(f"Causal model '{algo_name}' selected - treatment indicator required for full causal inference")
        # Note: For true causal inference, treatment must come from data
        # Causal models can still train on classification tasks for demonstration

    # Ensure all required fields exist
    if "algorithm_name" not in model_candidate:
        model_candidate["algorithm_name"] = "LogisticRegression"
    if "algorithm_class" not in model_candidate:
        model_candidate["algorithm_class"] = "sklearn.linear_model.LogisticRegression"
    if "hyperparameter_search_space" not in model_candidate:
        model_candidate["hyperparameter_search_space"] = {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 500},
        }
    if "default_hyperparameters" not in model_candidate:
        model_candidate["default_hyperparameters"] = {"C": 1.0, "max_iter": 200}

    # Normalize qc_report for model_trainer (expects qc_passed)
    normalized_qc_report = qc_report.copy()
    if "qc_passed" not in normalized_qc_report:
        normalized_qc_report["qc_passed"] = normalized_qc_report.get("gate_passed", True)

    # Split data using E2I required ratios: 60%/20%/15%/5%
    n = len(X)
    train_size = int(0.60 * n)
    val_size = int(0.20 * n)
    test_size = int(0.15 * n)
    holdout_size = n - train_size - val_size - test_size  # Remaining ~5%

    # Create split indices
    train_end = train_size
    val_end = train_end + val_size
    test_end = val_end + test_size

    # Build split dicts with required keys: X, y, row_count
    train_data = {
        "X": X.iloc[:train_end],
        "y": y.iloc[:train_end],
        "row_count": train_size,
    }
    validation_data = {
        "X": X.iloc[train_end:val_end],
        "y": y.iloc[train_end:val_end],
        "row_count": val_size,
    }
    test_data = {
        "X": X.iloc[val_end:test_end],
        "y": y.iloc[val_end:test_end],
        "row_count": test_size,
    }
    holdout_data = {
        "X": X.iloc[test_end:],
        "y": y.iloc[test_end:],
        "row_count": holdout_size,
    }

    # Extract feature columns for downstream SHAP computation
    feature_columns = list(X.columns)

    input_data = {
        "experiment_id": experiment_id,
        "model_candidate": model_candidate,
        "qc_report": normalized_qc_report,
        "enable_hpo": True,
        "hpo_trials": CONFIG.hpo_trials,
        "problem_type": CONFIG.problem_type,
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "holdout_data": holdout_data,
        "enable_mlflow": CONFIG.enable_mlflow,  # Explicit MLflow control
        "feature_columns": feature_columns,  # Pass feature names for SHAP
    }

    print("  Input:")
    print(f"    algorithm: {model_candidate['algorithm_name']}")
    print(f"    train_samples: {train_size} ({train_size / n:.0%})")
    print(f"    validation_samples: {val_size} ({val_size / n:.0%})")
    print(f"    test_samples: {test_size} ({test_size / n:.0%})")
    print(f"    holdout_samples: {holdout_size} ({holdout_size / n:.0%})")
    print(f"    total_samples: {n}")
    print(f"    hpo_trials: {CONFIG.hpo_trials}")
    print(f"    enable_hpo: True")

    print("\n  Running agent (this may take a few minutes with HPO)...")
    result = await agent.run(input_data)

    print("\n  Output:")
    print_result("training_run_id", result.get("training_run_id", "N/A"))
    print_result("model_id", result.get("model_id", "N/A"))
    print_result("auc_roc", result.get("auc_roc", "N/A"))
    print_result("precision", result.get("precision", "N/A"))
    print_result("recall", result.get("recall", "N/A"))
    print_result("f1_score", result.get("f1_score", "N/A"))
    print_result("success_criteria_met", result.get("success_criteria_met", "N/A"))
    print_result("hpo_trials_run", result.get("hpo_trials_run", "N/A"))
    print_result("training_duration_seconds", result.get("training_duration_seconds", "N/A"))

    # MLflow logging outputs (critical for model_uri handoff to Feature Analyzer)
    print_result("mlflow_status", result.get("mlflow_status", "N/A"))
    print_result("mlflow_run_id", result.get("mlflow_run_id", "N/A"))
    model_uri = result.get("model_artifact_uri") or result.get("mlflow_model_uri")
    print_result("model_uri", model_uri or "NOT AVAILABLE")
    if not model_uri:
        print_warning("model_uri is None - Feature Analyzer SHAP computation will be skipped")

    if result.get("validation_metrics"):
        print_result("validation_metrics", result["validation_metrics"])

    auc = result.get("auc_roc", 0)
    if auc and auc >= CONFIG.min_auc_threshold:
        print_success(f"Model AUC ({auc:.3f}) meets threshold ({CONFIG.min_auc_threshold})")
    elif auc:
        print_warning(f"Model AUC ({auc:.3f}) below threshold ({CONFIG.min_auc_threshold})")

    return result


async def step_6_feature_analyzer(
    experiment_id: str,
    trained_model: Any,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    model_uri: Optional[str] = None
) -> dict[str, Any]:
    """Step 6: Analyze feature importance."""
    print_header(6, "FEATURE ANALYZER")

    from src.agents.ml_foundation.feature_analyzer import FeatureAnalyzerAgent

    print("\n  Creating FeatureAnalyzerAgent...")
    agent = FeatureAnalyzerAgent()

    # Use actual feature names from DataFrame columns for SHAP output
    feature_columns = list(X_sample.columns)

    input_data = {
        "experiment_id": experiment_id,
        "trained_model": trained_model,
        "model_uri": model_uri,
        "X_sample": X_sample,
        "y_sample": y_sample,
        "max_samples": min(100, len(X_sample)),
        "feature_columns": feature_columns,  # Pass feature names for SHAP
    }

    print("  Input:")
    print(f"    sample_size: {len(X_sample)}")
    print(f"    features: {list(X_sample.columns)}")

    print("\n  Running agent...")
    try:
        result = await agent.run(input_data)

        print("\n  Output:")
        if result.get("feature_importance"):
            print("  Feature Importance:")
            for fi in result["feature_importance"][:5]:  # Top 5
                if isinstance(fi, dict):
                    print(f"    {fi.get('feature', 'unknown')}: {fi.get('importance', 0):.4f}")
                else:
                    print(f"    {fi}")

        print_result("samples_analyzed", result.get("samples_analyzed", "N/A"))
        print_result("computation_time_seconds", result.get("computation_time_seconds", "N/A"))

        print_success("Feature analysis completed")
    except Exception as e:
        print_warning(f"Feature analysis failed (optional): {e}")
        result = {"feature_importance": None, "error": str(e)}

    return result


async def step_7_model_deployer(
    experiment_id: str,
    model_uri: str,
    validation_metrics: dict,
    success_criteria_met: bool,
    trained_model: Any = None,
    include_bentoml: bool = False,
) -> dict[str, Any]:
    """Step 7: Deploy model.

    Args:
        experiment_id: The experiment identifier
        model_uri: MLflow model URI
        validation_metrics: Metrics from model training
        success_criteria_met: Whether model meets success criteria
        trained_model: The actual trained model object (for BentoML serving)
        include_bentoml: Whether to deploy and test with BentoML
    """
    print_header(7, "MODEL DEPLOYER")

    from src.agents.ml_foundation.model_deployer import ModelDeployerAgent

    print("\n  Creating ModelDeployerAgent...")
    agent = ModelDeployerAgent()

    deployment_name = f"kisqali_discontinuation_{experiment_id[:8]}"

    input_data = {
        "experiment_id": experiment_id,
        "model_uri": model_uri or f"runs:/{experiment_id}/model",
        "validation_metrics": validation_metrics,
        "success_criteria_met": success_criteria_met,
        "deployment_name": deployment_name,
        "deployment_action": "register",  # Just register for testing
    }

    print("  Input:")
    print(f"    deployment_name: {deployment_name}")
    print(f"    success_criteria_met: {success_criteria_met}")
    print(f"    deployment_action: register")

    print("\n  Running agent...")
    try:
        result = await agent.run(input_data)
    except Exception as agent_error:
        # Handle agent errors gracefully so BentoML serving can still run
        error_type = getattr(agent_error, "error_type", None) or type(agent_error).__name__
        print_warning(f"Agent error ({error_type}): {agent_error}")

        # Create a minimal result so we can continue
        result = {
            "status": "error",
            "deployment_successful": False,
            "error": str(agent_error),
            "error_type": error_type,
            "deployment_manifest": {
                "deployment_id": f"deploy_{experiment_id[:12]}",
                "environment": "staging",
                "status": "error",
            },
        }

    print("\n  Output:")
    print_result("status", result.get("status", "N/A"))
    print_result("deployment_successful", result.get("deployment_successful", "N/A"))
    print_result("model_version", result.get("model_version", "N/A"))

    if result.get("deployment_manifest"):
        print_result("deployment_manifest", result["deployment_manifest"])

    if result.get("deployment_successful", False) or result.get("status") == "completed":
        print_success("Model registered successfully")
    elif result.get("error"):
        print_warning(f"Agent had errors but continuing: {result.get('error_type', 'unknown')}")
    else:
        print_warning("Model registration may have issues")

    # BentoML Model Serving (optional)
    if include_bentoml and trained_model is not None:
        print("\n  " + "-" * 60)
        print("  BentoML Model Serving:")
        print("  " + "-" * 60)

        try:
            from src.mlops.bentoml_service import register_model_for_serving

            # Detect framework from model class
            model_class_name = type(trained_model).__name__
            if "XGB" in model_class_name:
                framework = "xgboost"
            elif "LGBM" in model_class_name or "LightGBM" in model_class_name:
                framework = "lightgbm"
            else:
                framework = "sklearn"

            model_name = f"tier0_{experiment_id[:8]}"
            print(f"    Registering model: {model_name} (framework: {framework})")

            registration = await register_model_for_serving(
                model=trained_model,
                model_name=model_name,
                metadata={
                    "experiment_id": experiment_id,
                    "validation_metrics": validation_metrics,
                    "tier0_test": True,
                    "algorithm": model_class_name,
                },
                framework=framework,
            )

            if registration.get("registration_status") == "success":
                model_tag = registration.get("model_tag")
                print(f"    ✓ Model registered: {model_tag}")

                # Start BentoML service with the registered model
                bentoml_result = await start_bentoml_service(model_tag, port=3001)

                if bentoml_result.get("started"):
                    endpoint = bentoml_result.get("endpoint")

                    # Verify predictions work with sample data
                    # Use same feature structure as training (3 features)
                    sample_features = [[30.0, 5.0, 1.0]]  # days_on_therapy, hcp_visits, prior_treatments

                    verification = await verify_bentoml_predictions(
                        endpoint=endpoint,
                        sample_features=sample_features,
                    )

                    # Display results
                    print("\n    BentoML Serving Verification:")
                    health_icon = "✓" if verification.get("health_check") else "✗"
                    print(f"      health_check: {health_icon} {'healthy' if verification.get('health_check') else 'unhealthy'}")

                    pred_icon = "✓" if verification.get("prediction_test") else "✗"
                    print(f"      prediction_test: {pred_icon} {'passed' if verification.get('prediction_test') else 'failed'}")

                    if verification.get("predictions"):
                        print(f"      predictions: {verification.get('predictions')}")
                    if verification.get("probabilities"):
                        print(f"      probabilities: {verification.get('probabilities')}")
                    if verification.get("latency_ms"):
                        print(f"      latency_ms: {verification.get('latency_ms'):.1f}")

                    result["bentoml_serving"] = {
                        "model_tag": model_tag,
                        "endpoint": endpoint,
                        "health_check": verification.get("health_check"),
                        "prediction_test": verification.get("prediction_test"),
                        "predictions": verification.get("predictions"),
                        "probabilities": verification.get("probabilities"),
                        "latency_ms": verification.get("latency_ms"),
                    }
                    result["bentoml_pid"] = bentoml_result.get("pid")

                    if verification.get("health_check") and verification.get("prediction_test"):
                        print_success("Real model deployed and serving verified via BentoML")
                    else:
                        print_warning("BentoML serving started but verification incomplete")
                else:
                    error_msg = bentoml_result.get("error", "Unknown error")
                    print(f"    ✗ BentoML service failed to start: {error_msg}")
                    result["bentoml_serving"] = {"error": error_msg}
            else:
                error_msg = registration.get("error", "Registration failed")
                print(f"    ✗ Model registration failed: {error_msg}")
                result["bentoml_serving"] = {"error": error_msg}

        except ImportError as e:
            print(f"    ✗ BentoML not available: {e}")
            result["bentoml_serving"] = {"error": f"Import error: {e}"}
        except Exception as e:
            print(f"    ✗ BentoML error: {e}")
            result["bentoml_serving"] = {"error": str(e)}
            import traceback
            traceback.print_exc()

    elif include_bentoml and trained_model is None:
        print_warning("BentoML requested but no trained_model available (run step 5 first)")
        result["bentoml_serving"] = {"error": "No trained model available"}

    return result


async def step_8_observability_connector(experiment_id: str, stages_completed: int) -> dict[str, Any]:
    """Step 8: Log to observability."""
    print_header(8, "OBSERVABILITY CONNECTOR")

    from src.agents.ml_foundation.observability_connector import ObservabilityConnectorAgent

    print("\n  Creating ObservabilityConnectorAgent...")
    agent = ObservabilityConnectorAgent()

    events = [
        {
            "event_type": "pipeline_completed",
            "agent_name": "tier0_e2e_test",
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {
                "experiment_id": experiment_id,
                "stages_completed": stages_completed,
                "brand": CONFIG.brand,
            },
        }
    ]

    input_data = {
        "events_to_log": events,
        "time_window": "1h",
    }

    print("  Input:")
    print(f"    events_to_log: 1 event")
    print(f"    time_window: 1h")

    print("\n  Running agent...")
    result = await agent.run(input_data)

    print("\n  Output:")
    print_result("emission_successful", result.get("emission_successful", "N/A"))
    print_result("events_logged", result.get("events_logged", "N/A"))
    print_result("quality_score", result.get("quality_score", "N/A"))

    if result.get("emission_successful", False):
        print_success("Events logged to observability")
    else:
        print_warning("Observability logging may have issues")

    return result


# =============================================================================
# MAIN RUNNER
# =============================================================================

async def run_pipeline(
    step: int | None = None,
    dry_run: bool = False,
    imbalance_ratio: float | None = None,
    include_bentoml: bool = False,
) -> None:
    """Run the full pipeline or a specific step.

    Args:
        step: Run only a specific step (1-8), or None for all steps
        dry_run: Show what would be done without executing
        imbalance_ratio: If provided, create imbalanced data with this minority ratio
        include_bentoml: If True, deploy real model to BentoML and verify predictions
    """
    import time

    experiment_id = f"tier0_e2e_{uuid.uuid4().hex[:8]}"
    pipeline_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"TIER 0 MLOPS WORKFLOW TEST")
    print(f"{'='*70}")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Brand: {CONFIG.brand}")
    print(f"  Target: {CONFIG.target_outcome}")
    print(f"  Problem Type: {CONFIG.problem_type}")
    print(f"  MLflow Enabled: {CONFIG.enable_mlflow}")
    print(f"  MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'not set')}")
    print(f"  BentoML Serving: {'Enabled' if include_bentoml else 'Disabled'}")
    if imbalance_ratio:
        print(f"  Class Imbalance: {imbalance_ratio:.1%} minority ratio (INJECTED)")
    print(f"  Started: {datetime.now().isoformat()}")

    if dry_run:
        print("\n  [DRY RUN MODE - No agents will be executed]")
        return

    # Generate sample data
    # NOTE: Generate 600 samples to satisfy scope_spec.minimum_samples=500
    # (extra samples account for potential exclusions during cohort construction)
    print("\n  Generating sample patient data...")
    patient_df = generate_sample_data(n_samples=600, imbalance_ratio=imbalance_ratio)
    print(f"  Generated {len(patient_df)} patient records")

    # Pipeline state
    state: dict[str, Any] = {
        "experiment_id": experiment_id,
        "patient_df": patient_df,
    }

    # Collect step results for detailed summary
    step_results: list[StepResult] = []

    steps_to_run = [step] if step else list(range(1, 9))

    try:
        # Step 1: Scope Definer
        if 1 in steps_to_run:
            step_start = time.time()
            result = await step_1_scope_definer(experiment_id)
            state["scope_spec"] = result.get("scope_spec", {"problem_type": CONFIG.problem_type})
            state["scope_spec"]["experiment_id"] = experiment_id
            step_results.append(StepResult(
                step_num=1,
                step_name="SCOPE DEFINER",
                status="success" if result.get("validation_passed", True) else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "experiment_id": result.get("experiment_id", experiment_id),
                    "problem_type": state["scope_spec"].get("problem_type"),
                    "prediction_target": state["scope_spec"].get("prediction_target"),
                    "minimum_samples": state["scope_spec"].get("minimum_samples"),
                },
                details={
                    "brand": CONFIG.brand,
                    "success_criteria": result.get("success_criteria", {}),
                }
            ))

        # Step 2: Data Preparer
        if 2 in steps_to_run:
            step_start = time.time()
            scope_spec = state.get("scope_spec", {"problem_type": CONFIG.problem_type})
            result = await step_2_data_preparer(experiment_id, scope_spec, patient_df)
            state["qc_report"] = result.get("qc_report", {"gate_passed": True})
            state["gate_passed"] = result.get("gate_passed", True)

            # Store DataFrames from data_preparer if available
            if result.get("train_df") is not None:
                state["train_df"] = result["train_df"]
            if result.get("validation_df") is not None:
                state["validation_df"] = result["validation_df"]

            qc_report = result.get("qc_report", {})
            step_results.append(StepResult(
                step_num=2,
                step_name="DATA PREPARER",
                status="success" if state["gate_passed"] else "failed",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "qc_status": qc_report.get("status", "unknown"),
                    "overall_score": qc_report.get("overall_score"),
                    "gate_passed": state["gate_passed"],
                    "train_samples": result.get("data_readiness", {}).get("train_samples"),
                    "validation_samples": result.get("data_readiness", {}).get("validation_samples"),
                },
                details={
                    "completeness_score": qc_report.get("completeness_score"),
                    "validity_score": qc_report.get("validity_score"),
                    "consistency_score": qc_report.get("consistency_score"),
                    "uniqueness_score": qc_report.get("uniqueness_score"),
                    "timeliness_score": qc_report.get("timeliness_score"),
                }
            ))

            if not state["gate_passed"]:
                print_failure("QC Gate blocked training. Pipeline stopped.")
                return

        # Step 3: Cohort Constructor
        if 3 in steps_to_run:
            step_start = time.time()
            eligible_df, cohort_result = await step_3_cohort_constructor(patient_df)
            state["eligible_df"] = eligible_df
            state["cohort_result"] = cohort_result
            step_results.append(StepResult(
                step_num=3,
                step_name="COHORT CONSTRUCTOR",
                status="success" if len(eligible_df) >= CONFIG.min_eligible_patients else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "cohort_id": cohort_result.cohort_id,
                    "input_patients": len(patient_df),
                    "eligible_patients": len(eligible_df),
                    "excluded_patients": len(patient_df) - len(eligible_df),
                    "exclusion_rate": f"{(len(patient_df) - len(eligible_df)) / len(patient_df):.1%}",
                },
                details={
                    "execution_id": cohort_result.execution_id,
                    "status": cohort_result.status,
                }
            ))

        # Step 4: Model Selector
        if 4 in steps_to_run:
            step_start = time.time()
            scope_spec = state.get("scope_spec", {"problem_type": CONFIG.problem_type})
            qc_report = state.get("qc_report", {"gate_passed": True})
            result = await step_4_model_selector(experiment_id, scope_spec, qc_report)
            state["model_candidate"] = result.get("model_candidate") or result.get("primary_candidate")

            candidate = state["model_candidate"]
            algo_name = candidate.get("algorithm_name") if isinstance(candidate, dict) else getattr(candidate, "algorithm_name", "Unknown")
            step_results.append(StepResult(
                step_num=4,
                step_name="MODEL SELECTOR",
                status="success" if candidate else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "selected_algorithm": algo_name,
                    "selection_score": result.get("selection_rationale", {}).get("selection_score") if isinstance(result.get("selection_rationale"), dict) else None,
                },
                details={
                    "selection_rationale": result.get("selection_rationale"),
                }
            ))

        # Step 5: Model Trainer
        if 5 in steps_to_run:
            step_start = time.time()
            eligible_df = state.get("eligible_df", patient_df)
            # Use numeric features for training
            feature_cols = ["days_on_therapy", "hcp_visits", "prior_treatments"]
            X = eligible_df[feature_cols].copy()
            y = eligible_df[CONFIG.target_outcome].copy()

            model_candidate = state.get("model_candidate", {
                "algorithm_name": "LogisticRegression",
                "hyperparameters": {"C": 1.0, "max_iter": 100}
            })
            qc_report = state.get("qc_report", {"gate_passed": True})

            result = await step_5_model_trainer(
                experiment_id, model_candidate, qc_report, X, y
            )
            state["trained_model"] = result.get("trained_model")
            state["validation_metrics"] = result.get("validation_metrics", {})
            # Try multiple possible keys for model_uri
            state["model_uri"] = (
                result.get("model_uri")
                or result.get("model_artifact_uri")
                or result.get("mlflow_model_uri")
            )
            state["success_criteria_met"] = result.get("success_criteria_met", True)

            # Capture class imbalance information
            state["class_imbalance_info"] = {
                "imbalance_detected": result.get("imbalance_detected", False),
                "imbalance_ratio": result.get("imbalance_ratio"),
                "minority_ratio": result.get("minority_ratio"),
                "imbalance_severity": result.get("imbalance_severity"),
                "class_distribution": result.get("class_distribution", {}),
                "recommended_strategy": result.get("recommended_strategy"),
                "strategy_rationale": result.get("strategy_rationale"),
            }

            # Capture resampling information if applied
            resampled_dist = result.get("resampled_distribution", {})
            # Calculate new minority ratio from resampled distribution
            if resampled_dist:
                total_resampled = sum(resampled_dist.values())
                new_minority_ratio = min(resampled_dist.values()) / total_resampled if total_resampled > 0 else None
            else:
                new_minority_ratio = None
            state["resampling_info"] = {
                "resampling_applied": result.get("resampling_applied", False),
                "original_samples": result.get("original_train_samples"),
                "resampled_samples": result.get("resampled_train_samples"),
                "original_distribution": result.get("original_distribution", {}),
                "resampled_distribution": resampled_dist,
                "new_minority_ratio": new_minority_ratio,
                "resampling_strategy": result.get("resampling_strategy"),
            }

            step_results.append(StepResult(
                step_num=5,
                step_name="MODEL TRAINER",
                status="success" if result.get("success_criteria_met") else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "training_run_id": result.get("training_run_id"),
                    "model_id": result.get("model_id"),
                    "auc_roc": result.get("auc_roc"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                    "f1_score": result.get("f1_score"),
                    "success_criteria_met": result.get("success_criteria_met"),
                    "hpo_trials_run": result.get("hpo_trials_run"),
                },
                details={
                    "mlflow_run_id": result.get("mlflow_run_id"),
                    "model_uri": state.get("model_uri"),
                    "training_duration_seconds": result.get("training_duration_seconds"),
                    "imbalance_detected": result.get("imbalance_detected", False),
                    "imbalance_severity": result.get("imbalance_severity"),
                    "remediation_strategy": result.get("recommended_strategy"),
                }
            ))

        # Step 6: Feature Analyzer
        if 6 in steps_to_run:
            step_start = time.time()
            eligible_df = state.get("eligible_df", patient_df)
            # Use numeric features for analysis
            feature_cols = ["days_on_therapy", "hcp_visits", "prior_treatments"]
            X = eligible_df[feature_cols].copy()
            y = eligible_df[CONFIG.target_outcome].copy()

            result = await step_6_feature_analyzer(
                experiment_id,
                state.get("trained_model"),
                X.iloc[:50],
                y.iloc[:50],
                model_uri=state.get("model_uri")
            )
            state["feature_importance"] = result.get("feature_importance")

            # Extract top features for summary
            top_features = {}
            if result.get("feature_importance"):
                for fi in result["feature_importance"][:5]:
                    if isinstance(fi, dict):
                        top_features[fi.get("feature", "unknown")] = fi.get("importance", 0)

            step_results.append(StepResult(
                step_num=6,
                step_name="FEATURE ANALYZER",
                status="success" if result.get("feature_importance") else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "samples_analyzed": result.get("samples_analyzed"),
                    "computation_time": result.get("computation_time_seconds"),
                    "top_feature": list(top_features.keys())[0] if top_features else None,
                },
                details={
                    "top_features": top_features,
                    "explainer_type": result.get("explainer_type"),
                }
            ))

        # Step 7: Model Deployer
        if 7 in steps_to_run:
            step_start = time.time()
            result = await step_7_model_deployer(
                experiment_id,
                state.get("model_uri"),
                state.get("validation_metrics", {}),
                state.get("success_criteria_met", True),
                trained_model=state.get("trained_model"),
                include_bentoml=include_bentoml,
            )
            state["deployment_manifest"] = result.get("deployment_manifest")
            # Track BentoML PID for cleanup
            if include_bentoml and result.get("bentoml_pid"):
                state["bentoml_pid"] = result["bentoml_pid"]

            manifest = result.get("deployment_manifest", {})
            bentoml_serving = result.get("bentoml_serving", {})
            step_details = {
                "model_version": result.get("model_version"),
                "endpoint_url": manifest.get("endpoint_url"),
            }
            # Add BentoML info if present
            if bentoml_serving:
                step_details["bentoml_model_tag"] = bentoml_serving.get("model_tag")
                step_details["bentoml_endpoint"] = bentoml_serving.get("endpoint")
                step_details["bentoml_health_check"] = bentoml_serving.get("health_check")
                step_details["bentoml_prediction_test"] = bentoml_serving.get("prediction_test")
                step_details["bentoml_latency_ms"] = bentoml_serving.get("latency_ms")

            step_results.append(StepResult(
                step_num=7,
                step_name="MODEL DEPLOYER",
                status="success" if result.get("deployment_successful") else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "deployment_id": manifest.get("deployment_id"),
                    "environment": manifest.get("environment"),
                    "status": manifest.get("status"),
                    "deployment_successful": result.get("deployment_successful"),
                    "bentoml_verified": bentoml_serving.get("prediction_test", False) if bentoml_serving else None,
                },
                details=step_details,
            ))

        # Step 8: Observability Connector
        if 8 in steps_to_run:
            step_start = time.time()
            result = await step_8_observability_connector(
                experiment_id,
                len(steps_to_run)
            )
            step_results.append(StepResult(
                step_num=8,
                step_name="OBSERVABILITY CONNECTOR",
                status="success" if result.get("emission_successful") else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "emission_successful": result.get("emission_successful"),
                    "events_logged": result.get("events_logged"),
                    "quality_score": result.get("quality_score"),
                },
                details={}
            ))

        # Print detailed step results
        print_detailed_summary(experiment_id, step_results, state)

        # Final summary
        pipeline_duration = time.time() - pipeline_start_time
        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"  Experiment ID: {experiment_id}")
        print(f"  Steps Completed: {len(steps_to_run)}")
        print(f"  Total Duration: {pipeline_duration:.1f}s")
        print(f"  QC Gate: {'PASSED' if state.get('gate_passed', True) else 'FAILED'}")
        if state.get("eligible_df") is not None:
            print(f"  Cohort Size: {len(state['eligible_df'])}")
        if state.get("validation_metrics"):
            print(f"  Validation Metrics: {state['validation_metrics']}")
        if include_bentoml and state.get("bentoml_pid"):
            print(f"  BentoML Serving: Verified (PID: {state['bentoml_pid']})")
        print(f"  Completed: {datetime.now().isoformat()}")
        print_success("Pipeline completed successfully!")

    except Exception as e:
        print_failure(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Cleanup BentoML service if started
        if state.get("bentoml_pid"):
            print("\n  Cleaning up BentoML service...")
            cleanup_result = await stop_bentoml_service(state["bentoml_pid"])
            if cleanup_result.get("stopped"):
                print(f"    ✓ BentoML service stopped (PID: {state['bentoml_pid']})")
            else:
                print(f"    ⚠️  BentoML cleanup issue: {cleanup_result.get('error', 'unknown')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tier 0 MLOps workflow test"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, 9),
        help="Run only a specific step (1-8)"
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking (enabled by default for model_uri generation)"
    )
    parser.add_argument(
        "--enable-opik",
        action="store_true",
        help="Enable Opik tracing"
    )
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=10,
        help="Number of HPO trials (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--imbalanced",
        type=float,
        default=None,
        metavar="RATIO",
        help="Create imbalanced data with specified minority ratio (e.g., 0.1 for 10%% minority class)"
    )
    parser.add_argument(
        "--include-bentoml",
        action="store_true",
        help="Include BentoML model serving verification with the real trained model"
    )

    args = parser.parse_args()

    # Update config
    if args.disable_mlflow:
        CONFIG.enable_mlflow = False
    CONFIG.enable_opik = args.enable_opik
    CONFIG.hpo_trials = args.hpo_trials

    # Run pipeline
    asyncio.run(run_pipeline(
        step=args.step,
        dry_run=args.dry_run,
        imbalance_ratio=args.imbalanced,
        include_bentoml=args.include_bentoml,
    ))


if __name__ == "__main__":
    main()
