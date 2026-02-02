"""Manual test for MLflow experiment tracking in CausalImpact workflow.

Run with: python -m tests.manual.test_mlflow_tracking
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.causal_impact.graph import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_EXPERIMENT_TAGS,
    create_causal_impact_graph,
    run_workflow_with_mlflow,
)
from src.agents.causal_impact.state import CausalImpactState


async def main():
    """Run MLflow tracking test."""
    print("=" * 60)
    print("MLflow Tracking Test for CausalImpact Agent")
    print("=" * 60)

    # Create workflow
    print("\n1. Creating causal impact workflow...")
    workflow = create_causal_impact_graph()
    print("   ✓ Workflow created")

    # Prepare initial state
    print("\n2. Preparing initial state...")
    initial_state: CausalImpactState = {
        "query": "What is the impact of HCP engagement on patient conversion rates?",
        "query_id": "test_mlflow_001",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region", "hcp_specialty"],
        "data_source": "synthetic",
        "interpretation_depth": "standard",
        "session_id": "test_session_001",
        "dispatch_id": "disp_test_001",
        "errors": [],
        "warnings": [],
    }
    print(f"   ✓ Query: {initial_state['query'][:50]}...")
    print(f"   ✓ Treatment: {initial_state['treatment_var']}")
    print(f"   ✓ Outcome: {initial_state['outcome_var']}")
    print(f"   ✓ Confounders: {initial_state['confounders']}")

    # Run with MLflow tracking
    print("\n3. Executing workflow with MLflow tracking...")
    print(f"   Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"   Tags: {MLFLOW_EXPERIMENT_TAGS}")

    try:
        result = await run_workflow_with_mlflow(
            workflow=workflow,
            initial_state=initial_state,
            run_name="test_mlflow_tracking",
        )

        print("\n4. Results:")
        print(f"   ✓ Status: {result.get('status')}")
        print(f"   ✓ MLflow Run ID: {result.get('mlflow_run_id')}")
        print(f"   ✓ Total Latency: {result.get('total_latency_ms', 0):.1f}ms")

        # Show estimation results
        est = result.get("estimation_result", {})
        if est:
            print("\n   Estimation Results:")
            print(f"     - ATE: {est.get('ate', 'N/A')}")
            print(f"     - P-value: {est.get('p_value', 'N/A')}")
            print(f"     - Method: {est.get('method', 'N/A')}")

        # Show refutation results
        ref = result.get("refutation_results", {})
        if ref:
            print("\n   Refutation Results:")
            print(
                f"     - Tests Passed: {ref.get('tests_passed', 'N/A')}/{ref.get('total_tests', 'N/A')}"
            )
            print(f"     - Overall Robust: {ref.get('overall_robust', 'N/A')}")

        # Show sensitivity results
        sens = result.get("sensitivity_analysis", {})
        if sens:
            print("\n   Sensitivity Analysis:")
            print(f"     - E-value: {sens.get('e_value', 'N/A')}")
            print(f"     - Robust to Confounding: {sens.get('robust_to_confounding', 'N/A')}")

        print("\n" + "=" * 60)
        print("✓ MLflow tracking test completed successfully!")
        print("=" * 60)

        if result.get("mlflow_run_id"):
            print(
                f"\nView run at: mlflow ui (then navigate to experiment '{MLFLOW_EXPERIMENT_NAME}')"
            )

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result and result.get("status") == "completed" else 1)
