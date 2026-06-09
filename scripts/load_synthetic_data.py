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
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

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
    BusinessMetricsGenerator,
    FeatureStoreSeeder,
    FeatureValueGenerator,
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
    "business_metrics": 10000,
    "feature_values": 50000,
}

SMALL_SIZES = {
    "hcp": 500,
    "patient": 2500,
    "treatment": 7500,
    "engagement": 1500,
    "outcome": 1000,
    "prediction": 2000,
    "trigger": 1200,
    "business_metrics": 1000,
    "feature_values": 5000,
}


def generate_datasets(sizes: dict, dgp_type: DGPType, seed: int = 42, verbose: bool = False,
                      id_prefix: str = ""):
    """Generate synthetic datasets for tables that exist in Supabase.

    id_prefix namespaces every generated entity id (Generator base prepends it) so a
    synthetic validation dataset's ids stay DISJOINT from the existing dev baseline —
    the loader's UPSERT then cannot clobber pre-existing rows, and cleanup by
    is_synthetic is FK-safe. Empty prefix reproduces the legacy ids.
    """
    datasets = {}

    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC DATA")
    logger.info("=" * 60)
    logger.info(f"DGP Type: {dgp_type.value}")
    logger.info(f"Seed: {seed}")
    logger.info("")

    # 1. Generate HCPs (no dependencies)
    logger.info(f"Generating {sizes['hcp']:,} HCP profiles...")
    hcp_config = GeneratorConfig(id_prefix=id_prefix, seed=seed, n_records=sizes["hcp"])
    hcp_df = HCPGenerator(hcp_config).generate()
    datasets["hcp_profiles"] = hcp_df
    logger.info(f"  Generated {len(hcp_df):,} HCPs")

    # 2. Generate Patients (depends on HCPs)
    logger.info(f"Generating {sizes['patient']:,} patient journeys...")
    patient_config = GeneratorConfig(id_prefix=id_prefix, seed=seed, n_records=sizes["patient"], dgp_type=dgp_type)
    patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()
    datasets["patient_journeys"] = patient_df
    logger.info(f"  Generated {len(patient_df):,} patients")

    # 3. Generate Treatment Events (depends on patients)
    logger.info(f"Generating {sizes['treatment']:,} treatment events...")
    treatment_config = GeneratorConfig(id_prefix=id_prefix, seed=seed, n_records=sizes["treatment"])
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
    prediction_config = GeneratorConfig(id_prefix=id_prefix, seed=seed, n_records=sizes["prediction"])
    prediction_df = PredictionGenerator(prediction_config, patient_df=patient_df).generate()
    # Rename columns to match database schema
    if "prediction_date" in prediction_df.columns:
        prediction_df = prediction_df.rename(columns={"prediction_date": "prediction_timestamp"})
    datasets["ml_predictions"] = prediction_df
    logger.info(f"  Generated {len(prediction_df):,} predictions")

    # 5. Generate Triggers (depends on patients and HCPs)
    logger.info(f"Generating {sizes['trigger']:,} triggers...")
    trigger_config = GeneratorConfig(id_prefix=id_prefix, seed=seed, n_records=sizes["trigger"])
    trigger_df = TriggerGenerator(trigger_config, patient_df=patient_df, hcp_df=hcp_df).generate()
    datasets["triggers"] = trigger_df
    logger.info(f"  Generated {len(trigger_df):,} triggers")

    # 6. Generate Business Metrics (for Gap Analyzer)
    logger.info(f"Generating {sizes['business_metrics']:,} business metrics...")
    bm_config = GeneratorConfig(id_prefix=id_prefix, seed=seed, n_records=sizes["business_metrics"])
    bm_df = BusinessMetricsGenerator(bm_config).generate()
    datasets["business_metrics"] = bm_df
    logger.info(f"  Generated {len(bm_df):,} business metrics")

    # 7. Seed Feature Store (for Drift Monitor)
    logger.info("Seeding feature store (groups and features)...")
    fs_seeder = FeatureStoreSeeder(GeneratorConfig(id_prefix=id_prefix, seed=seed))
    feature_groups_df, features_df = fs_seeder.seed()
    datasets["feature_groups"] = feature_groups_df
    datasets["features"] = features_df
    logger.info(f"  Seeded {len(feature_groups_df):,} feature groups, {len(features_df):,} features")

    # 8. Generate Feature Values (depends on feature store and patient data)
    logger.info(f"Generating {sizes['feature_values']:,} feature values...")
    fv_config = GeneratorConfig(id_prefix=id_prefix, seed=seed, n_records=sizes["feature_values"])
    fv_generator = FeatureValueGenerator(fv_config, features_df=features_df, patient_df=patient_df)
    fv_df = fv_generator.generate()
    datasets["feature_values"] = fv_df
    logger.info(f"  Generated {len(fv_df):,} feature values")

    logger.info("")
    total_records = sum(len(df) for df in datasets.values())
    logger.info(f"Total records generated: {total_records:,}")

    # Provenance: tag 100% of synthetic rows so read-path enforcement (Shard 07)
    # can default-exclude them. Central stamp guarantees coverage regardless of
    # which generator produced the frame; the column is carried through the loader
    # by TABLE_COLUMNS registration (batch_loader.py, Shard 02 Task 1).
    for table_name, df in datasets.items():
        df["is_synthetic"] = True
        datasets[table_name] = df
    logger.info("Stamped is_synthetic=True on all %d datasets", len(datasets))

    return datasets


def write_parquet_snapshots(datasets: dict, out_dir) -> str:
    """Dual-sink companion to load_to_supabase: write each dataset to
    <out_dir>/<table>.parquet + a manifest.json. Rows are is_synthetic-stamped
    upstream by generate_datasets. Requires pyarrow."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for table_name, df in datasets.items():
        path = out / f"{table_name}.parquet"
        # Some columns (e.g. feature_values.value/entity_values) hold Python
        # dict/list objects — the same shape the DB jsonb columns carry. pyarrow
        # cannot infer a uniform type for a heterogeneous object column, so
        # JSON-encode any object column that contains dict/list cells. This keeps
        # the snapshot faithful (values round-trip as JSON text) and the offline
        # path unblocked, byte-equivalent to the jsonb the loader upserts.
        df_out = df.copy()
        for col in df_out.columns:
            if df_out[col].dtype == "object" and df_out[col].map(
                lambda v: isinstance(v, (dict, list))
            ).any():
                df_out[col] = df_out[col].map(
                    lambda v: json.dumps(v, default=str)
                    if isinstance(v, (dict, list))
                    else v
                )
        df_out.to_parquet(path, index=False)
        tables.append({"table": table_name, "path": str(path), "rows": int(len(df_out))})
        logger.info("  Parquet %s (%d rows)", path, len(df_out))
    manifest = {
        "run_timestamp": datetime.now().isoformat(),
        "is_synthetic": True,
        "tables": tables,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Wrote %d parquet snapshots + manifest to %s", len(tables), out)
    return str(out)


# Canonical cohort -> outcome column map (INDEX §CANONICAL; mirrors cohort_resolution._PJ_COHORTS)
_COHORT_OUTCOME = {
    "initiation": "treatment_initiated",
    "discontinuation": "discontinued_180d",
    "persistence": "persistent_180d",
}
_CF_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]


def write_cohort_frames(out_dir) -> list:
    """Write the resolved per-(cohort,brand) causal frames estimators/agents consume,
    derived offline from the parquet snapshots. 9 patient cells + hcp_adoption."""
    import pandas as pd
    from pathlib import Path

    out = Path(out_dir)
    cf = out / "cohort_frames"
    cf.mkdir(parents=True, exist_ok=True)
    pj = pd.read_parquet(out / "patient_journeys.parquet")
    mlp = pd.read_parquet(out / "ml_predictions.parquet")[
        ["patient_id", "treatment_effect_estimate"]
    ]
    pj = pj.merge(mlp, on="patient_id", how="left")
    written = []
    for cohort, outcome in _COHORT_OUTCOME.items():
        for brand in _CF_BRANDS:
            sub = pj[pj["brand"] == brand].copy()
            if cohort in ("discontinuation", "persistence"):
                sub = sub[sub["treatment_initiated"] == 1]  # eligibility: only initiators
            cols = ["treatment_arm", outcome, "disease_severity", "age_at_diagnosis",
                    "segment_assignment", "propensity_score", "treatment_effect_estimate",
                    "brand", "is_synthetic"]
            frame = sub[[c for c in cols if c in sub.columns]].rename(columns={outcome: "outcome"})
            path = cf / f"{cohort}__{brand}.parquet"
            frame.to_parquet(path, index=False)
            written.append(str(path))
    # hcp_adoption (HCP grain): prefer Shard 06's per-HCP CATE artifact, else hcp_profiles
    for brand in _CF_BRANDS:
        cate_path = out / f"per_hcp_cate_hcp_adoption_{brand}.parquet"
        dest = cf / f"hcp_adoption__{brand}.parquet"
        if cate_path.exists():
            pd.read_parquet(cate_path).to_parquet(dest, index=False)
            written.append(str(dest))
        elif (out / "hcp_profiles.parquet").exists():
            hp = pd.read_parquet(out / "hcp_profiles.parquet")
            keep = [c for c in ("hcp_id", "adoption_category", "peer_influence_score",
                                "influence_network_size", "is_synthetic") if c in hp.columns]
            hp[keep].to_parquet(dest, index=False)
            written.append(str(dest))
    logger.info("Wrote %d cohort frames to %s", len(written), cf)
    return written


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
    parser.add_argument("--parquet-out", type=str, default=None,
                       help="Also write each dataset to <dir>/<table>.parquet + manifest.json")
    parser.add_argument("--parquet-only", action="store_true",
                       help="Write parquet only; SKIP the Supabase load (pollution-free)")
    parser.add_argument("--tag", type=str, default="scv",
                       help="Entity-id namespace prefix (default 'scv'). Keeps synthetic "
                            "ids disjoint from the dev baseline so UPSERT cannot clobber "
                            "existing rows. Pass '' to reproduce legacy un-namespaced ids.")
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
        datasets = generate_datasets(sizes, dgp_type, verbose=args.verbose, id_prefix=args.tag)

        # Parquet dual-sink (optional). --parquet-only skips the DB load entirely
        # (pollution-free: no writes to shared prod tables, no DB creds needed).
        if args.parquet_out or args.parquet_only:
            out_dir = args.parquet_out or f"data/synthetic/parquet/{datetime.now():%Y%m%dT%H%M%S}"
            write_parquet_snapshots(datasets, out_dir)
            write_cohort_frames(out_dir)  # Task 6
            if args.parquet_only:
                logger.info("parquet-only: skipping Supabase load")
                return 0

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
