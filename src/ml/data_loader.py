#!/usr/bin/env python3
"""
E2I Causal Analytics - Complete Supabase Data Loader V3.0

Loads all generated data into Supabase tables including:
- Core tables (patient_journeys, treatment_events, etc.)
- NEW KPI gap tables (user_sessions, data_source_tracking, etc.)
- ML split tracking tables

Usage:
    # Self-hosted Supabase (local)
    export SUPABASE_URL='http://localhost:8000'
    # Self-hosted Supabase (droplet)
    # export SUPABASE_URL='http://138.197.4.36:8000'
    export SUPABASE_KEY='your-service-role-key-from-self-hosted'
    python e2i_ml_complete_v3_loader.py --data-dir /path/to/json/files

Or for testing without Supabase:
    python e2i_ml_complete_v3_loader.py --dry-run --data-dir /path/to/json/files
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Look for .env in current directory and parent directories
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try current directory
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Try to import supabase client
try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # Type hint placeholder
    create_client = None
    print("Warning: supabase-py not installed. Running in dry-run mode.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Table loading order (respects foreign key constraints)
TABLE_LOAD_ORDER = [
    # ML infrastructure tables first
    ("ml_split_registry", "e2i_ml_v3_split_registry.json"),
    # Reference data
    ("reference_universe", "e2i_ml_v3_reference_universe.json"),
    ("hcp_profiles", "e2i_ml_v3_hcp_profiles.json"),
    # Core journey data
    ("patient_journeys", "e2i_ml_v3_patient_journeys.json"),
    ("treatment_events", "e2i_ml_v3_treatment_events.json"),
    # ML and prediction data
    ("ml_predictions", "e2i_ml_v3_ml_predictions.json"),
    ("ml_preprocessing_metadata", "e2i_ml_v3_preprocessing_metadata.json"),
    # Triggers and activities
    ("triggers", "e2i_ml_v3_triggers.json"),
    ("agent_activities", "e2i_ml_v3_agent_activities.json"),
    # Business metrics and causal paths
    ("business_metrics", "e2i_ml_v3_business_metrics.json"),
    ("causal_paths", "e2i_ml_v3_causal_paths.json"),
    # NEW KPI gap tables
    ("user_sessions", "e2i_ml_v3_user_sessions.json"),
    ("data_source_tracking", "e2i_ml_v3_data_source_tracking.json"),
    ("ml_annotations", "e2i_ml_v3_ml_annotations.json"),
    ("etl_pipeline_metrics", "e2i_ml_v3_etl_pipeline_metrics.json"),
    ("hcp_intent_surveys", "e2i_ml_v3_hcp_intent_surveys.json"),
]

# Batch size for inserts
BATCH_SIZE = 100


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(filepath, "r") as f:
        return json.load(f)


def transform_for_supabase(table_name: str, records: List[Dict]) -> List[Dict]:
    """Transform records for Supabase insertion."""
    transformed = []

    for record in records:
        # Create a copy to avoid modifying original
        rec = record.copy()

        # Handle array fields - convert Python lists to PostgreSQL arrays
        array_fields = [
            "secondary_diagnosis_codes",
            "affiliation_secondary",
            "comorbidities",
            "data_sources_matched",
            "icd_codes",
            "cpt_codes",
            "loinc_codes",
            "intermediate_nodes",
            "confounders_controlled",
            "mediators_identified",
            "feature_list",
        ]

        for field in array_fields:
            if field in rec and rec[field] is not None:
                # Supabase/PostgREST handles Python lists automatically
                pass

        # Handle JSONB fields - ensure they're dicts
        jsonb_fields = [
            "lab_values",
            "probability_scores",
            "feature_importance",
            "shap_values",
            "top_features",
            "rank_metrics",
            "fairness_metrics",
            "features_available_at_prediction",
            "causal_chain",
            "supporting_evidence",
            "input_data",
            "analysis_results",
            "recommendations",
            "actions_initiated",
            "resource_usage",
            "interaction_effects",
            "source_combination_flags",
            "stage_timings",
            "quality_check_details",
            "annotation_value",
            "adjudication_result",
            "interventions_since_last",
            "feature_means",
            "feature_stds",
            "feature_mins",
            "feature_maxs",
            "categorical_encodings",
            "feature_distributions",
        ]

        for field in jsonb_fields:
            if field in rec and rec[field] is not None:
                # Ensure it's serializable
                if isinstance(rec[field], str):
                    try:
                        rec[field] = json.loads(rec[field])
                    except:
                        pass

        # Handle None values for optional fields
        # Supabase handles None -> NULL automatically

        transformed.append(rec)

    return transformed


def insert_batch(client, table_name: str, records: List[Dict]) -> int:
    """Insert a batch of records into Supabase."""
    if not records:
        return 0

    try:
        response = client.table(table_name).insert(records).execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        print(f"    Error inserting into {table_name}: {e}")
        # Try inserting one by one to identify problematic records
        success_count = 0
        for rec in records:
            try:
                client.table(table_name).insert(rec).execute()
                success_count += 1
            except Exception as inner_e:
                print(
                    f"    Failed record: {rec.get('patient_id', rec.get('hcp_id', 'unknown'))}: {inner_e}"
                )
        return success_count


def validate_schema_compatibility(data_dir: str) -> Dict[str, bool]:
    """Validate that all expected files exist."""
    results = {}
    for table_name, filename in TABLE_LOAD_ORDER:
        filepath = os.path.join(data_dir, filename)
        results[table_name] = os.path.exists(filepath)
    return results


# =============================================================================
# MAIN LOADER CLASS
# =============================================================================


class E2IDataLoader:
    """Loads E2I data into Supabase."""

    def __init__(self, supabase_url: str, supabase_key: str, dry_run: bool = False):
        self.dry_run = dry_run
        self.client: Optional[Client] = None

        if not dry_run and SUPABASE_AVAILABLE:
            self.client = create_client(supabase_url, supabase_key)

    def load_all(self, data_dir: str) -> Dict[str, int]:
        """Load all data files into Supabase."""
        results = {}

        print("\n" + "=" * 60)
        print("E2I DATA LOADER V3.0")
        print("=" * 60)

        if self.dry_run:
            print("\n⚠️  DRY RUN MODE - No data will be inserted\n")

        # Validate files exist
        print("Validating data files...")
        file_check = validate_schema_compatibility(data_dir)
        missing = [t for t, exists in file_check.items() if not exists]
        if missing:
            print(f"  ⚠️  Missing files for tables: {missing}")
        else:
            print("  ✓ All data files present")

        # Load each table in order
        for table_name, filename in TABLE_LOAD_ORDER:
            filepath = os.path.join(data_dir, filename)

            if not os.path.exists(filepath):
                print(f"\n  Skipping {table_name} - file not found: {filename}")
                results[table_name] = 0
                continue

            print(f"\nLoading {table_name}...")

            # Load data
            records = load_json_file(filepath)

            # Handle nested data structure (train/validation/test files)
            if isinstance(records, dict) and "patient_journeys" in records:
                # This is a split file, skip it (we use individual table files)
                print(f"  Skipping composite file: {filename}")
                results[table_name] = 0
                continue

            print(f"  Found {len(records):,} records")

            # Transform for Supabase
            transformed = transform_for_supabase(table_name, records)

            if self.dry_run:
                # Just validate and report
                print(f"  ✓ Would insert {len(transformed):,} records")
                results[table_name] = len(transformed)
            else:
                # Actually insert
                inserted = 0
                for i in range(0, len(transformed), BATCH_SIZE):
                    batch = transformed[i : i + BATCH_SIZE]
                    count = insert_batch(self.client, table_name, batch)
                    inserted += count
                    if (i + BATCH_SIZE) % 500 == 0:
                        print(
                            f"  Progress: {min(i + BATCH_SIZE, len(transformed)):,}/{len(transformed):,}"
                        )

                print(f"  ✓ Inserted {inserted:,} records")
                results[table_name] = inserted

        # Run leakage audit
        if not self.dry_run and self.client:
            self._run_leakage_audit(data_dir)

        # Print summary
        self._print_summary(results)

        return results

    def _run_leakage_audit(self, data_dir: str):
        """Run the leakage audit function."""
        print("\nRunning leakage audit...")

        try:
            # Get the split_config_id from the loaded data
            split_registry_file = os.path.join(data_dir, "e2i_ml_v3_split_registry.json")
            if os.path.exists(split_registry_file):
                with open(split_registry_file) as f:
                    split_data = json.load(f)
                    if split_data:
                        split_config_id = split_data[0].get("split_config_id")
                        if split_config_id:
                            # Call the audit function via RPC
                            response = self.client.rpc(
                                "run_leakage_audit", {"p_split_config_id": split_config_id}
                            ).execute()
                            if response.data:
                                print("  Audit results:")
                                for check in response.data:
                                    status = "✓" if check["passed"] else "✗"
                                    print(f"    {status} {check['check_type']}: {check['details']}")
        except Exception as e:
            print(f"  ⚠️  Could not run audit: {e}")

    def _print_summary(self, results: Dict[str, int]):
        """Print loading summary."""
        print("\n" + "=" * 60)
        print("LOADING SUMMARY")
        print("=" * 60)

        total = sum(results.values())

        print(f"\n{'Table':<30} {'Records':>12}")
        print("-" * 44)

        for table_name, count in results.items():
            print(f"{table_name:<30} {count:>12,}")

        print("-" * 44)
        print(f"{'TOTAL':<30} {total:>12,}")

        if self.dry_run:
            print("\n⚠️  DRY RUN - No data was actually inserted")
        else:
            print("\n✓ All data loaded successfully")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Load E2I data into Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load data into self-hosted Supabase (local)
    export SUPABASE_URL='http://localhost:8000'
    export SUPABASE_KEY='your-service-role-key-from-self-hosted'
    python e2i_ml_complete_v3_loader.py --data-dir ./e2i_ml_complete_v3_data

    # Or for production droplet
    export SUPABASE_URL='http://138.197.4.36:8000'

    # Dry run to validate files
    python e2i_ml_complete_v3_loader.py --dry-run --data-dir ./e2i_ml_complete_v3_data
        """,
    )

    parser.add_argument("--data-dir", required=True, help="Directory containing JSON data files")

    parser.add_argument(
        "--dry-run", action="store_true", help="Validate files without inserting data"
    )

    parser.add_argument(
        "--supabase-url",
        default=os.environ.get("SUPABASE_URL"),
        help="Supabase project URL (or set SUPABASE_URL env var)",
    )

    parser.add_argument(
        "--supabase-key",
        default=os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY"),
        help="Supabase service role key (or set SUPABASE_KEY or SUPABASE_SERVICE_KEY env var)",
    )

    args = parser.parse_args()

    # Validate
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    if not args.dry_run:
        if not SUPABASE_AVAILABLE:
            print("Warning: supabase-py not installed. Install with: pip install supabase")
            args.dry_run = True
        elif not args.supabase_url or not args.supabase_key:
            print(
                "Error: Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY env vars."
            )
            print("Or use --dry-run to validate files without connecting.")
            sys.exit(1)

    # Load
    loader = E2IDataLoader(
        supabase_url=args.supabase_url or "",
        supabase_key=args.supabase_key or "",
        dry_run=args.dry_run,
    )

    results = loader.load_all(args.data_dir)

    # Exit code
    if all(v > 0 or k in ["ml_preprocessing_metadata"] for k, v in results.items()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
