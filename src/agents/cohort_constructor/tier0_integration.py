"""
E2I Tier 0 ML Foundation - Cohort Constructor Integration

Integrates CohortConstructor as a new agent in the ML Foundation tier,
positioned between scope_definer and data_preparer.

Agent Flow:
    scope_definer → cohort_constructor → data_preparer → model_selector → ...

Outputs:
    - CohortSpec: Eligibility criteria and configuration
    - EligiblePatients: Filtered patient population
    - EligibilityLog: Audit trail for regulatory compliance
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, cast

import pandas as pd

# Import CohortConstructor
from cohort_constructor import CohortConfig, CohortConstructor, Criterion, Operator
from supabase import Client, create_client

# ============================================
# E2I AGENT CONTRACT
# ============================================


class CohortConstructorAgent:
    """
    E2I Tier 0 Agent: cohort_constructor

    Responsibilities:
    1. Load cohort configuration (from scope_definer output or database)
    2. Apply inclusion/exclusion criteria
    3. Validate temporal eligibility
    4. Generate eligibility audit log
    5. Store results in Supabase
    6. Pass eligible patients to data_preparer

    Agent Type: Standard (tool-heavy, SLA-bound)
    Tier: 0 (ML Foundation)
    SLA: 120 seconds
    """

    def __init__(self):
        # Supabase connection
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        # Agent metadata
        self.agent_name = "cohort_constructor"
        self.agent_tier = 0
        self.agent_type = "standard"
        self.version = "1.0.0"

    def run(self, input_data: Dict) -> Dict:
        """
        Main agent execution method

        Input Contract:
            {
                'scope_spec': {
                    'brand': str,
                    'indication': str,
                    'target_population': str,
                    'business_objective': str
                },
                'patient_data_source': str,  # Table name or path
                'use_existing_config': bool,  # Load from DB vs create new
                'cohort_config': Optional[Dict]  # Override config if provided
            }

        Output Contract:
            {
                'cohort_spec': Dict,
                'eligible_patients': pd.DataFrame,
                'eligibility_log': List[Dict],
                'metadata': Dict,
                'success': bool,
                'error_message': Optional[str]
            }
        """

        start_time = datetime.now()

        try:
            # Step 1: Load or create cohort configuration
            config = self._load_cohort_config(input_data)

            # Step 2: Load patient data
            patient_df = self._load_patient_data(input_data["patient_data_source"])

            # Step 3: Construct cohort
            constructor = CohortConstructor(config)
            eligible_df, metadata = constructor.construct_cohort(patient_df)

            # Step 4: Store results in Supabase
            execution_id = self._store_cohort_execution(config, metadata, patient_df, eligible_df)

            # Step 5: Store eligibility log
            self._store_eligibility_log(
                metadata["cohort_id"], execution_id, metadata["eligibility_log"]
            )

            # Step 6: Store patient assignments
            self._store_patient_assignments(
                metadata["cohort_id"], execution_id, patient_df, eligible_df
            )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Return output
            return {
                "cohort_spec": config.to_dict(),
                "eligible_patients": eligible_df,
                "eligibility_log": metadata["eligibility_log"],
                "metadata": {
                    **metadata,
                    "execution_id": execution_id,
                    "execution_time_seconds": execution_time,
                    "agent_name": self.agent_name,
                    "agent_version": self.version,
                },
                "success": True,
                "error_message": None,
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "cohort_spec": None,
                "eligible_patients": None,
                "eligibility_log": None,
                "metadata": {
                    "execution_time_seconds": execution_time,
                    "agent_name": self.agent_name,
                    "agent_version": self.version,
                },
                "success": False,
                "error_message": str(e),
            }

    def _load_cohort_config(self, input_data: Dict) -> CohortConfig:
        """Load or create cohort configuration"""

        # Option 1: Use provided config
        if "cohort_config" in input_data and input_data["cohort_config"]:
            return CohortConfig(**input_data["cohort_config"])

        # Option 2: Load existing config from database
        if input_data.get("use_existing_config", False):
            brand = input_data["scope_spec"]["brand"]
            indication = input_data["scope_spec"]["indication"]

            response = (
                self.supabase.table("ml_cohort_definitions")
                .select("*")
                .eq("brand", brand)
                .eq("indication", indication)
                .eq("status", "active")
                .order("created_date", desc=True)
                .limit(1)
                .execute()
            )

            if response.data:
                db_config = cast(Dict[str, Any], response.data[0])
                # Convert DB record to CohortConfig
                return self._db_record_to_config(db_config)

        # Option 3: Create from brand/indication defaults
        brand = input_data["scope_spec"]["brand"]
        indication = input_data["scope_spec"]["indication"]

        return CohortConfig.from_brand(brand, indication)

    def _db_record_to_config(self, db_record: Dict) -> CohortConfig:
        """Convert database record to CohortConfig"""

        inclusion = [
            Criterion(
                field=c["field"],
                operator=Operator(c["operator"]),
                value=c["value"],
                description=c.get("description"),
                clinical_rationale=c.get("clinical_rationale"),
            )
            for c in db_record["inclusion_criteria"]
        ]

        exclusion = [
            Criterion(
                field=c["field"],
                operator=Operator(c["operator"]),
                value=c["value"],
                description=c.get("description"),
                clinical_rationale=c.get("clinical_rationale"),
            )
            for c in db_record["exclusion_criteria"]
        ]

        return CohortConfig(
            brand=db_record["brand"],
            indication=db_record["indication"],
            cohort_name=db_record["cohort_name"],
            inclusion_criteria=inclusion,
            exclusion_criteria=exclusion,
            lookback_days=db_record.get("lookback_days", 180),
            followup_days=db_record.get("followup_days", 90),
            index_date_field=db_record.get("index_date_field", "diagnosis_date"),
            version=db_record.get("version", "1.0.0"),
        )

    def _load_patient_data(self, data_source: str) -> pd.DataFrame:
        """Load patient data from Supabase table or file"""

        if data_source.startswith("s3://") or data_source.endswith(".parquet"):
            # Load from file
            return pd.read_parquet(data_source)
        else:
            # Load from Supabase table
            response = self.supabase.table(data_source).select("*").execute()
            return pd.DataFrame(response.data)

    def _store_cohort_execution(
        self,
        config: CohortConfig,
        metadata: Dict,
        initial_df: pd.DataFrame,
        eligible_df: pd.DataFrame,
    ) -> str:
        """Store cohort execution record"""

        execution_id = (
            f"{config.brand}_{config.indication}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        execution_record = {
            "execution_id": execution_id,
            "cohort_id": metadata["cohort_id"],
            "execution_timestamp": datetime.now().isoformat(),
            "executed_by": os.getenv("USER", "system"),
            "environment": os.getenv("ENVIRONMENT", "production"),
            "input_row_count": len(initial_df),
            "eligible_row_count": len(eligible_df),
            "excluded_row_count": len(initial_df) - len(eligible_df),
            "execution_time_seconds": metadata.get("execution_time_seconds", 0),
            "status": "success",
            "execution_metadata": metadata,
        }

        self.supabase.table("ml_cohort_executions").insert(execution_record).execute()

        return execution_id

    def _store_eligibility_log(
        self, cohort_id: str, execution_id: str, eligibility_log: List[Dict]
    ):
        """Store eligibility log entries"""

        log_records = [
            {
                "cohort_id": cohort_id,
                "execution_id": execution_id,
                "criterion_name": entry["criterion"],
                "criterion_type": entry["type"],
                "operator": entry.get("operator", ""),
                "criterion_value": json.dumps(entry.get("value")),
                "removed_count": entry["removed"],
                "remaining_count": entry["remaining"],
                "description": entry.get("description"),
                "clinical_rationale": entry.get("clinical_rationale"),
                "applied_at": entry.get("timestamp", datetime.now().isoformat()),
            }
            for entry in eligibility_log
        ]

        if log_records:
            self.supabase.table("ml_cohort_eligibility_log").insert(log_records).execute()

    def _store_patient_assignments(
        self, cohort_id: str, execution_id: str, initial_df: pd.DataFrame, eligible_df: pd.DataFrame
    ):
        """Store patient cohort assignments"""

        eligible_ids = set(eligible_df["patient_journey_id"].values)

        assignments = [
            {
                "patient_journey_id": row["patient_journey_id"],
                "cohort_id": cohort_id,
                "execution_id": execution_id,
                "is_eligible": row["patient_journey_id"] in eligible_ids,
                "assigned_at": datetime.now().isoformat(),
            }
            for _, row in initial_df.iterrows()
        ]

        # Batch insert (Supabase has limits, so chunk if needed)
        batch_size = 1000
        for i in range(0, len(assignments), batch_size):
            batch = assignments[i : i + batch_size]
            self.supabase.table("ml_patient_cohort_assignments").insert(batch).execute()


# ============================================
# E2I WORKFLOW INTEGRATION
# ============================================


class E2ITier0Workflow:
    """
    E2I Tier 0 ML Foundation Workflow with Cohort Constructor

    Agent Sequence:
        1. scope_definer
        2. cohort_constructor (NEW)
        3. data_preparer
        4. model_selector
        5. model_trainer
        6. feature_analyzer
        7. model_deployer
        8. observability_connector
    """

    def __init__(self):
        # Initialize agents
        self.cohort_constructor_agent = CohortConstructorAgent()

        # Agent outputs (state management)
        self.state = {}

    def execute_tier0_pipeline(self, initial_scope: Dict) -> Dict:
        """
        Execute Tier 0 pipeline with cohort construction

        Args:
            initial_scope: Output from scope_definer

        Returns:
            Pipeline execution results
        """

        print("=" * 60)
        print("E2I TIER 0 PIPELINE - WITH COHORT CONSTRUCTOR")
        print("=" * 60)

        # Step 1: Scope Definer (assumed complete, load from state)
        print("\n[Step 1] scope_definer")
        print("   ✓ Scope specification loaded")
        self.state["scope_spec"] = initial_scope

        # Step 2: Cohort Constructor (NEW)
        print("\n[Step 2] cohort_constructor (NEW)")
        cohort_input = {
            "scope_spec": self.state["scope_spec"],
            "patient_data_source": "patient_journeys",  # Supabase table
            "use_existing_config": True,
        }

        cohort_output = self.cohort_constructor_agent.run(cohort_input)

        if not cohort_output["success"]:
            print(f"   ✗ FAILED: {cohort_output['error_message']}")
            return {"success": False, "error": cohort_output["error_message"]}

        print(
            f"   ✓ Cohort constructed: {len(cohort_output['eligible_patients']):,} eligible patients"
        )
        print(f"   ✓ Exclusion rate: {cohort_output['metadata']['exclusion_rate']:.1%}")

        self.state["cohort_spec"] = cohort_output["cohort_spec"]
        self.state["eligible_patients"] = cohort_output["eligible_patients"]
        self.state["eligibility_log"] = cohort_output["eligibility_log"]

        # Step 3: Data Preparer (receives eligible patients only)
        print("\n[Step 3] data_preparer")
        print(f"   ✓ Processing {len(self.state['eligible_patients']):,} eligible patients")
        # data_preparer logic here...

        # Step 4-8: Continue with rest of Tier 0
        print(
            "\n[Steps 4-8] model_selector → model_trainer → feature_analyzer → model_deployer → observability"
        )
        print("   ✓ Pipeline complete")

        return {
            "success": True,
            "tier": 0,
            "eligible_population": len(self.state["eligible_patients"]),
            "cohort_metadata": cohort_output["metadata"],
        }


# ============================================
# INTEGRATION EXAMPLES
# ============================================


def example_1_standalone_cohort_construction():
    """Example 1: Use cohort_constructor as standalone agent"""

    print("\n" + "=" * 60)
    print("EXAMPLE 1: Standalone Cohort Construction")
    print("=" * 60)

    agent = CohortConstructorAgent()

    input_data = {
        "scope_spec": {
            "brand": "remibrutinib",
            "indication": "csu",
            "target_population": "Adults with moderate-to-severe CSU",
            "business_objective": "Predict treatment initiation",
        },
        "patient_data_source": "patient_journeys",
        "use_existing_config": False,  # Create new config
    }

    output = agent.run(input_data)

    if output["success"]:
        print("\n✓ Success!")
        print(f"   Eligible Patients: {len(output['eligible_patients']):,}")
        print(f"   Cohort ID: {output['metadata']['cohort_id']}")
    else:
        print(f"\n✗ Failed: {output['error_message']}")


def example_2_full_tier0_integration():
    """Example 2: Full Tier 0 pipeline with cohort construction"""

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Full Tier 0 Integration")
    print("=" * 60)

    workflow = E2ITier0Workflow()

    initial_scope = {
        "brand": "remibrutinib",
        "indication": "csu",
        "target_outcome": "treatment_initiated",
        "validation_criteria": {"min_auc": 0.75, "min_precision": 0.45},
    }

    result = workflow.execute_tier0_pipeline(initial_scope)

    if result["success"]:
        print("\n✓ Tier 0 Pipeline Complete!")
        print(f"   Eligible Population: {result['eligible_population']:,}")
    else:
        print(f"\n✗ Pipeline Failed: {result['error']}")


def example_3_compare_cohort_definitions():
    """Example 3: Compare multiple cohort definitions"""

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Compare Cohort Definitions")
    print("=" * 60)

    # Create test data
    import numpy as np
    from cohort_constructor import compare_cohorts

    np.random.seed(42)
    n = 1000

    test_df = pd.DataFrame(
        {
            "patient_journey_id": [f"patient_{i}" for i in range(n)],
            "age_at_diagnosis": np.random.randint(18, 80, n),
            "primary_diagnosis_code": np.random.choice(["L50.0", "L50.1", "L50.8"], n),
            "urticaria_severity_uas7": np.random.randint(0, 42, n),
            "antihistamine_failures_count": np.random.randint(0, 4, n),
            "pregnancy_flag": np.random.choice([True, False], n, p=[0.05, 0.95]),
            "severe_immunodeficiency": False,
            "physical_urticaria_only": False,
            "diagnosis_date": pd.date_range("2022-01-01", periods=n, freq="8H"),
            "journey_start_date": pd.date_range("2021-01-01", periods=n, freq="8H"),
            "follow_up_days": np.random.randint(30, 365, n),
        }
    )

    # Create two cohort configs with different criteria
    config_strict = CohortConfig.from_brand("remibrutinib", "csu")

    config_relaxed = CohortConfig.from_brand("remibrutinib", "csu")
    # Relax severity requirement
    config_relaxed.inclusion_criteria[2].value = 10  # UAS7 >= 10 instead of 16
    config_relaxed.cohort_name = "Remibrutinib CSU Relaxed Criteria"

    # Compare
    comparison_df = compare_cohorts(test_df, [config_strict, config_relaxed])

    print("\nCohort Comparison:")
    print(comparison_df.to_string(index=False))


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Check environment
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("⚠️  Warning: SUPABASE_URL and SUPABASE_KEY not set")
        print("   Set environment variables to run examples")

    # Run examples
    try:
        example_1_standalone_cohort_construction()
    except Exception as e:
        print(f"Example 1 error: {e}")

    try:
        example_2_full_tier0_integration()
    except Exception as e:
        print(f"Example 2 error: {e}")

    try:
        example_3_compare_cohort_definitions()
    except Exception as e:
        print(f"Example 3 error: {e}")
