#!/usr/bin/env python3
"""
Load Synthetic Data to Supabase.

Generates synthetic data for all entity types and loads to Supabase
in dependency order with validation and progress reporting.

Usage:
    python scripts/load_synthetic_data.py [--dry-run] [--small] [--verbose]

Options:
    --dry-run   Validate without loading to database
    --small     Generate smaller dataset for testing (1/10 size)
    --verbose   Enable verbose logging
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.synthetic.config import DGPType
from src.ml.synthetic.generators import (
    GeneratorConfig,
    HCPGenerator,
    PatientGenerator,
    TreatmentGenerator,
    EngagementGenerator,
    OutcomeGenerator,
    PredictionGenerator,
    TriggerGenerator,
)
from src.ml.synthetic.loaders import BatchLoader, LoaderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Dataset sizes (full vs small)
FULL_SIZES = {
    "hcp": 5000,
    "patient": 25000,
    "treatment": 75000,
    "engagement": 15000,
    "outcome": 10000,
    "prediction": 20000,
    "trigger": 12000,
}

SMALL_SIZES = {
    "hcp": 500,
    "patient": 2500,
    "treatment": 7500,
    "engagement": 1500,
    "outcome": 1000,
    "prediction": 2000,
    "trigger": 1200,
}


def generate_datasets(sizes: dict, dgp_type: DGPType, seed: int = 42, verbose: bool = False):
    """Generate synthetic datasets for tables that exist in Supabase."""
    datasets = {}

    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC DATA")
    logger.info("=" * 60)
    logger.info(f"DGP Type: {dgp_type.value}")
    logger.info(f"Seed: {seed}")
    logger.info("")

    # 1. Generate HCPs (no dependencies)
    logger.info(f"Generating {sizes['hcp']:,} HCP profiles...")
    hcp_config = GeneratorConfig(seed=seed, n_records=sizes["hcp"])
    hcp_df = HCPGenerator(hcp_config).generate()
    datasets["hcp_profiles"] = hcp_df
    logger.info(f"  Generated {len(hcp_df):,} HCPs")

    # 2. Generate Patients (depends on HCPs)
    logger.info(f"Generating {sizes['patient']:,} patient journeys...")
    patient_config = GeneratorConfig(seed=seed, n_records=sizes["patient"], dgp_type=dgp_type)
    patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()
    datasets["patient_journeys"] = patient_df
    logger.info(f"  Generated {len(patient_df):,} patients")

    # 3. Generate Treatment Events (depends on patients)
    logger.info(f"Generating {sizes['treatment']:,} treatment events...")
    treatment_config = GeneratorConfig(seed=seed, n_records=sizes["treatment"])
    treatment_df = TreatmentGenerator(treatment_config, patient_df=patient_df).generate()
    # Rename columns to match database schema
    if "treatment_date" in treatment_df.columns:
        treatment_df = treatment_df.rename(columns={"treatment_date": "event_date"})
    if "treatment_type" in treatment_df.columns:
        treatment_df = treatment_df.rename(columns={"treatment_type": "event_type"})
    if "days_supply" in treatment_df.columns:
        treatment_df = treatment_df.rename(columns={"days_supply": "duration_days"})
    datasets["treatment_events"] = treatment_df
    logger.info(f"  Generated {len(treatment_df):,} treatment events")

    # 4. Generate ML Predictions (depends on patients)
    logger.info(f"Generating {sizes['prediction']:,} ML predictions...")
    prediction_config = GeneratorConfig(seed=seed, n_records=sizes["prediction"])
    prediction_df = PredictionGenerator(prediction_config, patient_df=patient_df).generate()
    # Rename columns to match database schema
    if "prediction_date" in prediction_df.columns:
        prediction_df = prediction_df.rename(columns={"prediction_date": "prediction_timestamp"})
    datasets["ml_predictions"] = prediction_df
    logger.info(f"  Generated {len(prediction_df):,} predictions")

    # 5. Generate Triggers (depends on patients and HCPs)
    logger.info(f"Generating {sizes['trigger']:,} triggers...")
    trigger_config = GeneratorConfig(seed=seed, n_records=sizes["trigger"])
    trigger_df = TriggerGenerator(trigger_config, patient_df=patient_df, hcp_df=hcp_df).generate()
    datasets["triggers"] = trigger_df
    logger.info(f"  Generated {len(trigger_df):,} triggers")

    logger.info("")
    total_records = sum(len(df) for df in datasets.values())
    logger.info(f"Total records generated: {total_records:,}")

    return datasets


def load_to_supabase(datasets: dict, dry_run: bool = False, verbose: bool = False):
    """Load datasets to Supabase."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("LOADING TO SUPABASE" + (" (DRY RUN)" if dry_run else ""))
    logger.info("=" * 60)

    config = LoaderConfig(
        batch_size=500,
        max_retries=3,
        validate_before_load=True,
        dry_run=dry_run,
        verbose=verbose,
    )

    loader = BatchLoader(config)

    # Validate datasets first
    logger.info("Validating datasets...")
    is_valid, errors = loader.validate_datasets(datasets)
    if not is_valid:
        logger.error("Validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return None
    logger.info("  Validation passed!")

    # Load with progress callback
    def progress_callback(table: str, current: int, total: int):
        logger.info(f"  Loading {table} ({current}/{total})...")

    results = loader.load_all(datasets, progress_callback=progress_callback)

    # Print summary
    summary = loader.get_loading_summary(results)
    logger.info("")
    print(summary)

    return results


def main():
    parser = argparse.ArgumentParser(description="Load synthetic data to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Validate without loading")
    parser.add_argument("--small", action="store_true", help="Generate smaller dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dgp", type=str, default="confounded",
                       choices=["simple_linear", "confounded", "heterogeneous", "time_series", "selection_bias"],
                       help="DGP type to use")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Select sizes
    sizes = SMALL_SIZES if args.small else FULL_SIZES

    # Select DGP type
    dgp_map = {
        "simple_linear": DGPType.SIMPLE_LINEAR,
        "confounded": DGPType.CONFOUNDED,
        "heterogeneous": DGPType.HETEROGENEOUS,
        "time_series": DGPType.TIME_SERIES,
        "selection_bias": DGPType.SELECTION_BIAS,
    }
    dgp_type = dgp_map[args.dgp]

    start_time = datetime.now()

    try:
        # Generate datasets
        datasets = generate_datasets(sizes, dgp_type, verbose=args.verbose)

        # Load to Supabase
        results = load_to_supabase(datasets, dry_run=args.dry_run, verbose=args.verbose)

        duration = (datetime.now() - start_time).total_seconds()

        if results:
            # Check if all succeeded
            all_success = all(r.is_success for r in results.values())

            logger.info("")
            logger.info("=" * 60)
            if all_success:
                logger.info(f"SUCCESS! All data loaded in {duration:.1f}s")
            else:
                logger.warning(f"COMPLETED WITH WARNINGS in {duration:.1f}s")
                failed = [name for name, r in results.items() if not r.is_success]
                logger.warning(f"Tables with issues: {failed}")
            logger.info("=" * 60)

            return 0 if all_success else 1
        else:
            logger.error("Loading failed!")
            return 1

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
