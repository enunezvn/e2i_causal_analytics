#!/usr/bin/env python3
"""
Feature Store Example and Test Script

Demonstrates E2I Lightweight Feature Store usage with real examples.

Prerequisites:
1. Run database migration: python scripts/run_migration.py database/migrations/004_create_feature_store_schema.sql
2. Ensure Redis is running (docker compose up redis)
3. Ensure MLflow is running (docker compose up mlflow)
4. Set environment variables in .env:
   - SUPABASE_URL
   - SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)
   - REDIS_URL (default: redis://localhost:6379)

Usage:
    python scripts/feature_store_example.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from feature_store import FeatureStoreClient


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def check_environment():
    """Verify required environment variables are set."""
    print_section("Environment Check")

    load_dotenv()

    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    }

    optional_vars = {
        "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    }

    # Check required
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set these in your .env file:")
        for var in missing:
            print(f"  {var}=<your-value>")
        sys.exit(1)

    print("✅ Required environment variables set:")
    for key, value in required_vars.items():
        masked = value[:20] + "..." if value and len(value) > 20 else value
        print(f"   {key}: {masked}")

    print("\n✅ Optional environment variables:")
    for key, value in optional_vars.items():
        print(f"   {key}: {value}")

    return required_vars, optional_vars


def test_health_check(fs: FeatureStoreClient):
    """Test feature store health check."""
    print_section("Health Check")

    health = fs.health_check()

    print("Service Status:")
    for service, status in health.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {service.capitalize()}: {'Healthy' if status else 'Unhealthy'}")

    if not health["supabase"]:
        print("\n❌ Supabase connection failed. Check your SUPABASE_URL and key.")
        sys.exit(1)

    if not health["redis"]:
        print("\n⚠️  Redis not available. Caching will be disabled.")

    if not health["mlflow"]:
        print("\n⚠️  MLflow not available. Feature tracking will be limited.")


def demo_hcp_features(fs: FeatureStoreClient):
    """Demonstrate HCP demographic features."""
    print_section("Demo: HCP Demographics Features")

    # Create feature group
    print("📦 Creating feature group: hcp_demographics")
    try:
        hcp_group = fs.create_feature_group(
            name="hcp_demographics",
            description="Healthcare provider demographic features",
            owner="data-team",
            source_table="hcps",
            expected_update_frequency_hours=168,  # Weekly
            max_age_hours=720,  # 30 days
            tags=["demographics", "hcp", "core"],
            mlflow_experiment_name="hcp_features",
        )
        print(f"   ✅ Created: {hcp_group.name} (ID: {hcp_group.id})")
    except Exception as e:
        print(f"   ⚠️  Feature group may already exist: {e}")
        hcp_group = fs.get_feature_group("hcp_demographics")

    # Create features
    print("\n📊 Creating features:")

    features_to_create = [
        {
            "name": "specialty",
            "value_type": "string",
            "description": "Primary medical specialty of HCP",
            "tags": ["categorical"],
        },
        {
            "name": "years_in_practice",
            "value_type": "int64",
            "description": "Number of years HCP has been practicing",
            "tags": ["numerical"],
        },
        {
            "name": "practice_size",
            "value_type": "string",
            "description": "Size of practice (solo, small, medium, large)",
            "tags": ["categorical"],
        },
    ]

    for feature_def in features_to_create:
        try:
            feature = fs.create_feature(
                feature_group_name="hcp_demographics",
                name=feature_def["name"],
                value_type=feature_def["value_type"],
                entity_keys=["hcp_id"],
                description=feature_def["description"],
                owner="data-team",
                tags=feature_def["tags"],
            )
            print(f"   ✅ Created: {feature.name} ({feature.value_type})")
        except Exception as e:
            print(f"   ⚠️  Feature {feature_def['name']} may already exist: {e}")

    # Write feature values
    print("\n✍️  Writing feature values for sample HCPs:")

    sample_hcps = [
        {
            "hcp_id": "HCP001",
            "specialty": "Oncology",
            "years_in_practice": 12,
            "practice_size": "medium",
        },
        {
            "hcp_id": "HCP002",
            "specialty": "Cardiology",
            "years_in_practice": 8,
            "practice_size": "large",
        },
        {
            "hcp_id": "HCP003",
            "specialty": "Hematology",
            "years_in_practice": 15,
            "practice_size": "solo",
        },
    ]

    for hcp_data in sample_hcps:
        hcp_id = hcp_data["hcp_id"]
        feature_values = []

        for feature_name, value in hcp_data.items():
            if feature_name == "hcp_id":
                continue

            feature_values.append(
                {
                    "feature_name": feature_name,
                    "entity_values": {"hcp_id": hcp_id},
                    "value": value,
                    "event_timestamp": datetime.utcnow(),
                    "feature_group": "hcp_demographics",
                }
            )

        count = fs.write_batch_features(feature_values)
        print(f"   ✅ Wrote {count} features for {hcp_id}")

    # Retrieve features
    print("\n🔍 Retrieving features for HCP001 (with cache):")

    entity_features = fs.get_entity_features(
        entity_values={"hcp_id": "HCP001"},
        feature_group="hcp_demographics",
        use_cache=True,
    )

    print(f"   Entity: {entity_features.entity_values}")
    print(f"   Features: {entity_features.features}")
    print(f"   Retrieved at: {entity_features.retrieved_at}")

    # Test cache hit
    print("\n🔍 Retrieving again (should hit cache):")
    entity_features_2 = fs.get_entity_features(
        entity_values={"hcp_id": "HCP001"},
        feature_group="hcp_demographics",
        use_cache=True,
    )
    print(f"   Features: {entity_features_2.features}")
    print(f"   ✅ Cache working (instant retrieval)")

    # Convert to dict for ML model
    print("\n🤖 Convert to ML model input format:")
    feature_dict = entity_features.to_dict()
    print(f"   {feature_dict}")


def demo_brand_features(fs: FeatureStoreClient):
    """Demonstrate brand performance features."""
    print_section("Demo: Brand Performance Features")

    # Create feature group
    print("📦 Creating feature group: brand_performance")
    try:
        brand_group = fs.create_feature_group(
            name="brand_performance",
            description="Brand-level performance metrics",
            owner="analytics-team",
            expected_update_frequency_hours=24,  # Daily
            max_age_hours=168,  # 7 days
            tags=["brand", "performance", "metrics"],
        )
        print(f"   ✅ Created: {brand_group.name}")
    except Exception as e:
        print(f"   ⚠️  Feature group may already exist: {e}")

    # Create features
    print("\n📊 Creating features:")

    features_to_create = [
        {
            "name": "total_nrx_30d",
            "value_type": "int64",
            "description": "Total new prescriptions in last 30 days",
        },
        {
            "name": "market_share",
            "value_type": "float64",
            "description": "Market share percentage",
        },
        {
            "name": "growth_rate_qoq",
            "value_type": "float64",
            "description": "Quarter-over-quarter growth rate",
        },
    ]

    for feature_def in features_to_create:
        try:
            fs.create_feature(
                feature_group_name="brand_performance",
                name=feature_def["name"],
                value_type=feature_def["value_type"],
                entity_keys=["brand_id", "geography_id"],
                description=feature_def["description"],
                owner="analytics-team",
            )
            print(f"   ✅ Created: {feature_def['name']}")
        except Exception as e:
            print(f"   ⚠️  Feature {feature_def['name']} may already exist: {e}")

    # Write sample data
    print("\n✍️  Writing brand performance metrics:")

    brand_data = [
        {
            "brand_id": "remibrutinib",
            "geography_id": "US_NORTHEAST",
            "total_nrx_30d": 1250,
            "market_share": 12.5,
            "growth_rate_qoq": 8.3,
        },
        {
            "brand_id": "fabhalta",
            "geography_id": "US_WEST",
            "total_nrx_30d": 890,
            "market_share": 15.2,
            "growth_rate_qoq": 5.7,
        },
    ]

    for data in brand_data:
        entity_values = {
            "brand_id": data["brand_id"],
            "geography_id": data["geography_id"],
        }

        feature_values = []
        for key, value in data.items():
            if key in ["brand_id", "geography_id"]:
                continue

            feature_values.append(
                {
                    "feature_name": key,
                    "entity_values": entity_values,
                    "value": value,
                    "event_timestamp": datetime.utcnow(),
                    "feature_group": "brand_performance",
                }
            )

        count = fs.write_batch_features(feature_values)
        print(f"   ✅ Wrote {count} features for {data['brand_id']} in {data['geography_id']}")

    # Retrieve features
    print("\n🔍 Retrieving brand performance:")

    entity_features = fs.get_entity_features(
        entity_values={"brand_id": "remibrutinib", "geography_id": "US_NORTHEAST"},
        feature_group="brand_performance",
    )

    print(f"   Features: {entity_features.features}")


def demo_historical_features(fs: FeatureStoreClient):
    """Demonstrate historical feature retrieval."""
    print_section("Demo: Historical Feature Retrieval")

    print("📈 Retrieving HCP feature history (last 30 days):")

    historical = fs.get_historical_features(
        entity_values={"hcp_id": "HCP001"},
        feature_names=["specialty", "years_in_practice"],
        start_time=datetime.utcnow() - timedelta(days=30),
        end_time=datetime.utcnow(),
    )

    if historical:
        print(f"   Found {len(historical)} historical records:")
        for record in historical[:5]:  # Show first 5
            print(f"   - {record['feature_name']}: {record['value']} at {record['event_timestamp']}")
    else:
        print("   ℹ️  No historical data found (features just created)")


def main():
    """Run feature store examples."""
    print_section("E2I Lightweight Feature Store - Demo & Test")

    # Check environment
    required_vars, optional_vars = check_environment()

    # Initialize feature store client
    print_section("Initialize Feature Store Client")

    supabase_url = required_vars["SUPABASE_URL"]
    supabase_key = required_vars["SUPABASE_ANON_KEY"]
    redis_url = optional_vars["REDIS_URL"]
    mlflow_uri = optional_vars["MLFLOW_TRACKING_URI"]

    print(f"🔌 Connecting to:")
    print(f"   Supabase: {supabase_url}")
    print(f"   Redis: {redis_url}")
    print(f"   MLflow: {mlflow_uri}")

    try:
        fs = FeatureStoreClient(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            redis_url=redis_url,
            mlflow_tracking_uri=mlflow_uri,
            cache_ttl_seconds=3600,  # 1 hour
            enable_cache=True,
        )
        print("✅ Feature store client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize feature store: {e}")
        sys.exit(1)

    # Run health check
    test_health_check(fs)

    # Run demos
    try:
        demo_hcp_features(fs)
        demo_brand_features(fs)
        demo_historical_features(fs)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Summary
    print_section("Summary")
    print("✅ Feature store demo completed successfully!")
    print("\n📚 Next steps:")
    print("   1. Check MLflow UI (http://localhost:5000) for feature tracking")
    print("   2. Monitor feature freshness in Supabase")
    print("   3. Integrate with E2I agents (Gap Analyzer, Prediction Synthesizer, etc.)")
    print("   4. Set up scheduled freshness updates")
    print("\n📖 Documentation: docs/FEATURE_STORE.md")

    # Cleanup
    fs.close()


if __name__ == "__main__":
    main()
