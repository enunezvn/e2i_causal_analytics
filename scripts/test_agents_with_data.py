#!/usr/bin/env python
"""Test ML agents with loaded synthetic data."""

import asyncio
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def test_gap_analyzer():
    """Test Gap Analyzer Agent."""
    print("=" * 60)
    print("TESTING GAP ANALYZER AGENT")
    print("=" * 60)

    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    # Initialize agent (disable MLflow/Opik for faster testing)
    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    print(f"Agent initialized: {agent.agent_name} (Tier {agent.agent_tier})")

    # Prepare test input
    test_input = {
        "query": "Identify conversion rate gaps for Remibrutinib",
        "brand": "Remibrutinib",
        "metrics": ["conversion_rate", "prescribing_volume"],
        "segments": ["high_value", "mid_value", "low_value"],
        "gap_type": "vs_potential",
        "time_period": "current_quarter",
    }

    print(f"\nTest Input:")
    print(f"  Brand: {test_input['brand']}")
    print(f"  Metrics: {test_input['metrics']}")
    print(f"  Segments: {test_input['segments']}")

    # Run analysis
    print("\nRunning gap analysis...")
    result = await agent.run(test_input)

    print("\nResults:")
    print(f"  Status: {result.get('status', 'N/A')}")
    print(f"  Gaps Detected: {len(result.get('gaps_detected', []))}")
    print(f"  Opportunities: {len(result.get('prioritized_opportunities', []))}")
    print(f"  Quick Wins: {len(result.get('quick_wins', []))}")
    print(f"  Strategic Bets: {len(result.get('strategic_bets', []))}")

    tav = result.get("total_addressable_value", 0)
    print(f"  Total Addressable Value: ${tav:,.2f}")
    print(f"  Total Latency: {result.get('total_latency_ms', 0)}ms")

    if result.get("key_insights"):
        print("\nKey Insights:")
        for insight in result["key_insights"][:3]:
            print(f"  - {insight}")

    if result.get("error"):
        print(f"\nError: {result['error']}")

    return result


async def test_causal_impact():
    """Test Causal Impact Agent."""
    print("\n" + "=" * 60)
    print("TESTING CAUSAL IMPACT AGENT")
    print("=" * 60)

    try:
        from src.agents.causal_impact.agent import CausalImpactAgent

        # Initialize agent
        agent = CausalImpactAgent(enable_mlflow=False, enable_opik=False)
        print(f"Agent initialized: {agent.agent_name} (Tier {agent.agent_tier})")

        # Prepare test input
        test_input = {
            "query": "What is the causal effect of digital engagement on prescription volume?",
            "brand": "Remibrutinib",
            "treatment": "digital_engagement_score",
            "outcome": "prescribing_volume",
            "covariates": ["specialty", "years_experience", "practice_size"],
        }

        print(f"\nTest Input:")
        print(f"  Treatment: {test_input['treatment']}")
        print(f"  Outcome: {test_input['outcome']}")
        print(f"  Covariates: {test_input['covariates']}")

        # Run analysis
        print("\nRunning causal impact analysis...")
        result = await agent.run(test_input)

        print("\nResults:")
        print(f"  Status: {result.get('status', 'N/A')}")
        print(f"  Causal Effect: {result.get('causal_effect', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Total Latency: {result.get('total_latency_ms', 0)}ms")

        if result.get("error"):
            print(f"\nError: {result['error']}")

        return result
    except Exception as e:
        print(f"Error testing Causal Impact: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def test_drift_monitor():
    """Test Drift Monitor Agent."""
    print("\n" + "=" * 60)
    print("TESTING DRIFT MONITOR AGENT")
    print("=" * 60)

    try:
        from src.agents.drift_monitor.agent import DriftMonitorAgent

        # Initialize agent
        agent = DriftMonitorAgent(enable_mlflow=False, enable_opik=False)
        print(f"Agent initialized: {agent.agent_name} (Tier {agent.agent_tier})")

        # Prepare test input
        test_input = {
            "query": "Check for data drift in HCP engagement metrics",
            "brand": "Remibrutinib",
            "features": ["digital_engagement_score", "prescribing_volume", "conversion_rate"],
            "reference_period": "2024-Q4",
            "current_period": "2025-Q1",
        }

        print(f"\nTest Input:")
        print(f"  Features: {test_input['features']}")
        print(f"  Reference Period: {test_input['reference_period']}")
        print(f"  Current Period: {test_input['current_period']}")

        # Run analysis
        print("\nRunning drift analysis...")
        result = await agent.run(test_input)

        print("\nResults:")
        print(f"  Status: {result.get('status', 'N/A')}")
        print(f"  Drift Detected: {result.get('drift_detected', 'N/A')}")
        print(f"  Total Latency: {result.get('total_latency_ms', 0)}ms")

        if result.get("error"):
            print(f"\nError: {result['error']}")

        return result
    except Exception as e:
        print(f"Error testing Drift Monitor: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def check_data_availability():
    """Check what data is available in Supabase."""
    print("=" * 60)
    print("CHECKING DATA AVAILABILITY")
    print("=" * 60)

    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        return

    supabase = create_client(url, key)

    # Check tables
    tables = [
        "hcp_profiles",
        "patient_journeys",
        "treatment_events",
        "ml_predictions",
        "triggers",
    ]

    print("\nData Summary:")
    for table in tables:
        try:
            result = supabase.table(table).select("count", count="exact").execute()
            print(f"  {table}: {result.count} records")
        except Exception as e:
            print(f"  {table}: Error - {e}")


async def test_causal_analysis_direct():
    """Test causal analysis directly with loaded data."""
    print("\n" + "=" * 60)
    print("TESTING CAUSAL ANALYSIS WITH LOADED DATA")
    print("=" * 60)

    from supabase import create_client
    import pandas as pd
    import numpy as np

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase = create_client(url, key)

    # Fetch HCP profiles and treatment events for causal analysis
    print("\nFetching data for causal analysis...")
    hcp_result = supabase.table("hcp_profiles").select("*").limit(500).execute()
    hcp_df = pd.DataFrame(hcp_result.data)
    print(f"  HCP Profiles: {len(hcp_df)} records")

    events_result = supabase.table("treatment_events").select("*").limit(1000).execute()
    events_df = pd.DataFrame(events_result.data)
    print(f"  Treatment Events: {len(events_df)} records")

    # Analyze: Effect of digital engagement on prescribing volume
    print("\n--- Causal Analysis: Digital Engagement -> Prescribing Volume ---")

    # Simple correlation analysis as baseline
    if "digital_engagement_score" in hcp_df.columns and "prescribing_volume" in hcp_df.columns:
        correlation = hcp_df["digital_engagement_score"].corr(hcp_df["prescribing_volume"])
        print(f"  Correlation (digital_engagement -> prescribing_volume): {correlation:.4f}")

        # Split by high/low digital engagement
        median_engagement = hcp_df["digital_engagement_score"].median()
        high_engagement = hcp_df[hcp_df["digital_engagement_score"] >= median_engagement]
        low_engagement = hcp_df[hcp_df["digital_engagement_score"] < median_engagement]

        avg_high = high_engagement["prescribing_volume"].mean()
        avg_low = low_engagement["prescribing_volume"].mean()
        diff = avg_high - avg_low

        print(f"  High engagement avg prescribing: {avg_high:.2f}")
        print(f"  Low engagement avg prescribing: {avg_low:.2f}")
        print(f"  Naive ATE (difference): {diff:.2f}")

    # Try DoWhy causal inference
    print("\n--- DoWhy Causal Inference ---")
    try:
        import dowhy
        from dowhy import CausalModel

        # Check available columns
        print(f"  Available columns: {list(hcp_df.columns)[:10]}...")

        # Prepare data for DoWhy
        cols_needed = ["digital_engagement_score", "prescribing_volume",
                       "years_experience", "decile", "total_patient_volume"]
        missing_cols = [c for c in cols_needed if c not in hcp_df.columns]
        if missing_cols:
            print(f"  Missing columns: {missing_cols}")
            # Use alternative columns if available
            cols_needed = [c for c in cols_needed if c in hcp_df.columns]

        analysis_df = hcp_df[cols_needed].dropna()
        print(f"  Data after dropna: {len(analysis_df)} rows")

        print(f"  Analysis data shape: {analysis_df.shape}")
        if len(analysis_df) > 30:
            # Create binary treatment
            analysis_df["treatment"] = (
                analysis_df["digital_engagement_score"] >=
                analysis_df["digital_engagement_score"].median()
            ).astype(int)

            # Define causal model
            model = CausalModel(
                data=analysis_df,
                treatment="treatment",
                outcome="prescribing_volume",
                common_causes=["years_experience", "decile", "total_patient_volume"],
            )

            # Identify causal effect
            identified_estimand = model.identify_effect()
            print(f"  Estimand identified: {identified_estimand.estimands['backdoor']}")

            # Estimate causal effect using propensity score matching
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
            )

            print(f"  Causal Effect (ATE): {estimate.value:.4f}")
            print(f"  Method: Propensity Score Matching")

            # Refute the estimate
            refutation = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause",
            )
            print(f"  Refutation (random cause): {refutation.new_effect:.4f}")
        else:
            print("  Not enough data for DoWhy analysis")

    except ImportError:
        print("  DoWhy not available")
    except Exception as e:
        print(f"  DoWhy analysis failed: {e}")

    # Test EconML if available
    print("\n--- EconML CATE Analysis ---")
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import GradientBoostingRegressor

        cols_needed = ["digital_engagement_score", "prescribing_volume",
                       "years_experience", "decile", "total_patient_volume"]
        cols_available = [c for c in cols_needed if c in hcp_df.columns]
        analysis_df = hcp_df[cols_available].dropna()
        print(f"  EconML data shape: {analysis_df.shape}")

        if len(analysis_df) > 30 and "prescribing_volume" in analysis_df.columns:
            # Prepare data
            Y = analysis_df["prescribing_volume"].values
            T = (analysis_df["digital_engagement_score"] >=
                 analysis_df["digital_engagement_score"].median()).astype(int).values
            X = analysis_df[["decile"]].values  # Effect modifiers
            W = analysis_df[["years_experience", "total_patient_volume"]].values  # Confounders

            # Fit DML model
            dml = LinearDML(
                model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3),
                model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3),
                random_state=42,
            )
            dml.fit(Y, T, X=X, W=W)

            # Get ATE
            ate = dml.ate(X)
            print(f"  Average Treatment Effect (ATE): {ate:.4f}")

            # Get CATE for different deciles
            print("  CATE by Decile:")
            for decile in sorted(analysis_df["decile"].unique())[:5]:
                X_test = np.array([[decile]])
                cate = dml.effect(X_test)[0]
                print(f"    Decile {decile}: {cate:.4f}")
        else:
            print("  Not enough data for EconML analysis")

    except ImportError:
        print("  EconML not available")
    except Exception as e:
        print(f"  EconML analysis failed: {e}")
        import traceback
        traceback.print_exc()

    return {"status": "completed"}


async def main():
    """Run all agent tests."""
    # Check data availability first
    await check_data_availability()

    # Test causal analysis directly with loaded data
    print("\n")
    await test_causal_analysis_direct()

    # Test Gap Analyzer (requires business_metrics table)
    # print("\n")
    # await test_gap_analyzer()

    print("\n" + "=" * 60)
    print("AGENT TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
