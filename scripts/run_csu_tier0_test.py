#!/usr/bin/env python3
"""CSU-specific Tier 0 test runner using real-world patient journey data.

This script loads real-world CSU (Chronic Spontaneous Urticaria) patient
journey data from JSON (produced by scripts/convert_csu_rwd.py) instead of
generating synthetic data, then runs the same 8-step Tier 0 ML pipeline.

Usage:
    # Run full pipeline with CSU RWD data
    python scripts/run_csu_tier0_test.py

    # Dry run (config check only)
    python scripts/run_csu_tier0_test.py --dry-run

    # Run specific step
    python scripts/run_csu_tier0_test.py --step 1

    # Custom data directory
    python scripts/run_csu_tier0_test.py --data-dir data/rwd/csu/

    # Use discontinuation_flag as target (medicated patients only)
    python scripts/run_csu_tier0_test.py --target discontinuation_flag

    # Skip BentoML serving verification
    python scripts/run_csu_tier0_test.py --no-bentoml

Prerequisites:
    - CSU RWD data converted: python scripts/convert_csu_rwd.py
    - API running (port 8000)
    - MLflow running (port 5000, optional)

Author: E2I Causal Analytics Team
"""

import argparse
import asyncio
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project bootstrap (mirrors run_tier0_test.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

os.environ["SUPABASE_URL"] = "http://localhost:54321"

# ---------------------------------------------------------------------------
# Import the tier0 module via importlib (scripts/ is not a package)
# ---------------------------------------------------------------------------


def _load_tier0_module():
    """Load run_tier0_test as a module without requiring scripts to be a package."""
    spec = importlib.util.spec_from_file_location(
        "run_tier0_test",
        PROJECT_ROOT / "scripts" / "run_tier0_test.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tier0_module = _load_tier0_module()


# =============================================================================
# CSU CONFIGURATION
# =============================================================================

@dataclass
class CSUTestConfig:
    """Test configuration for CSU RWD pipeline."""

    brand: str = "competitor"
    problem_type: str = "binary_classification"
    target_outcome: str = "treatment_initiated"
    indication: str = "Chronic Spontaneous Urticaria (CSU)"
    hpo_trials: int = 10
    min_eligible_patients: int = 30
    min_auc_threshold: float = 0.55
    min_minority_recall: float = 0.10
    min_minority_precision: float = 0.05
    enable_mlflow: bool = True
    enable_opik: bool = False


CSU_CONFIG = CSUTestConfig()


# =============================================================================
# DATA LOADING
# =============================================================================


def load_csu_data(data_dir: str = "data/rwd/csu") -> pd.DataFrame:
    """Load CSU RWD patient journeys from JSON.

    Args:
        data_dir: Directory containing ``e2i_ml_v3_patient_journeys.json``.

    Returns:
        DataFrame with columns matching the Tier 0 pipeline schema.

    Raises:
        FileNotFoundError: If the JSON file has not been generated yet.
    """
    path = Path(data_dir) / "e2i_ml_v3_patient_journeys.json"

    if not path.exists():
        raise FileNotFoundError(
            f"CSU RWD data not found at {path}\n"
            "Run the conversion script first:\n"
            "    python scripts/convert_csu_rwd.py\n"
            "\n"
            "This converts data/rwd/csu/csu_data.xlsx into the JSON "
            "format expected by the Tier 0 pipeline."
        )

    with open(path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Map age_group values to what the pipeline expects
    #   RWD has:    '<18', '18-34', '35-49', '50-65', '65+'
    #   Pipeline:   '<50', '50-65', '>65'
    # ------------------------------------------------------------------
    age_map = {
        "<18": "<50",
        "18-34": "<50",
        "35-49": "<50",
        "50-65": "50-65",
        "65+": ">65",
    }
    df["age_group"] = df["age_group"].map(age_map).fillna("<50")

    # Ensure numeric types for ML features
    for col in ["days_on_therapy", "hcp_visits", "prior_treatments"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Ensure target columns are int
    df["treatment_initiated"] = (
        pd.to_numeric(df["treatment_initiated"], errors="coerce").fillna(0).astype(int)
    )

    # Handle discontinuation_flag (may be None for non-medicated patients)
    if "discontinuation_flag" in df.columns:
        df["discontinuation_flag"] = pd.to_numeric(
            df["discontinuation_flag"], errors="coerce"
        )

    # Ensure journey_status exists
    if "journey_status" not in df.columns:
        df["journey_status"] = df.apply(
            lambda r: "transitioning" if r.get("treatment_initiated", 0) == 1 else "active",
            axis=1,
        )

    return df


# =============================================================================
# CSU PIPELINE RUNNER
# =============================================================================


async def run_csu_pipeline(
    step: int | None = None,
    dry_run: bool = False,
    data_dir: str = "data/rwd/csu",
    include_bentoml: bool = False,
    target: str = "treatment_initiated",
) -> dict:
    """Run Tier 0 pipeline with CSU RWD data.

    This patches the tier0 module's CONFIG and generate_sample_data function
    to swap in real-world CSU data, then delegates to the original pipeline.

    Args:
        step: Run only a specific step (1-8), or None for all.
        dry_run: Show configuration without executing.
        data_dir: Path to directory containing the CSU JSON file.
        include_bentoml: Whether to include BentoML serving verification.
        target: Target outcome column ('treatment_initiated' or 'discontinuation_flag').

    Returns:
        Pipeline state dictionary.
    """
    # Update config target
    CSU_CONFIG.target_outcome = target

    # Patch the module-level CONFIG
    original_config = tier0_module.CONFIG
    tier0_module.CONFIG = CSU_CONFIG

    # Monkey-patch generate_sample_data to load RWD instead
    original_generate = tier0_module.generate_sample_data

    def csu_generate_data(n_samples: int = None, seed: int = 42, imbalance_ratio=None):
        df = load_csu_data(data_dir)

        # If targeting discontinuation_flag, filter to medicated patients only
        if target == "discontinuation_flag":
            pre_filter = len(df)
            df = df[
                (df["treatment_initiated"] == 1)
                & df["discontinuation_flag"].notna()
            ].copy()
            df["discontinuation_flag"] = df["discontinuation_flag"].astype(int)
            print(f"  Filtered to medicated patients: {pre_filter} -> {len(df)}")

        print(f"  Loaded {len(df)} CSU RWD patient records from {data_dir}")
        print(f"  Indication: {CSU_CONFIG.indication}")
        print(f"  Target: {CSU_CONFIG.target_outcome}")

        target_col = CSU_CONFIG.target_outcome
        if target_col in df.columns:
            pos = int(df[target_col].sum())
            total = len(df)
            print(f"  Class distribution: {pos}/{total} positive ({pos / total:.1%})")

        return df

    tier0_module.generate_sample_data = csu_generate_data

    try:
        return await tier0_module.run_pipeline(
            step=step,
            dry_run=dry_run,
            include_bentoml=include_bentoml,
        )
    finally:
        # Restore originals
        tier0_module.CONFIG = original_config
        tier0_module.generate_sample_data = original_generate


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tier 0 MLOps pipeline with CSU real-world data",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, 9),
        help="Run only a specific step (1-8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without executing",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/rwd/csu",
        help="Directory containing CSU RWD JSON (default: data/rwd/csu)",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["treatment_initiated", "discontinuation_flag"],
        default="treatment_initiated",
        help=(
            "Target outcome column. "
            "'treatment_initiated' (primary, all patients) or "
            "'discontinuation_flag' (secondary, medicated patients only). "
            "Default: treatment_initiated"
        ),
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--enable-opik",
        action="store_true",
        help="Enable Opik tracing",
    )
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=10,
        help="Number of HPO trials (default: 10)",
    )
    parser.add_argument(
        "--no-bentoml",
        action="store_true",
        help="Skip BentoML model serving verification",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/results",
        help="Directory to save results MD file (default: docs/results)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file (only print to console)",
    )

    args = parser.parse_args()

    # Apply CLI overrides to config
    if args.disable_mlflow:
        CSU_CONFIG.enable_mlflow = False
    CSU_CONFIG.enable_opik = args.enable_opik
    CSU_CONFIG.hpo_trials = args.hpo_trials
    CSU_CONFIG.target_outcome = args.target

    # Setup output capture (same pattern as run_tier0_test.py)
    output_buffer = None
    original_stdout = sys.stdout

    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class TeeOutput:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush()

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        output_buffer = io.StringIO()
        sys.stdout = TeeOutput(original_stdout, output_buffer)

    try:
        asyncio.run(
            run_csu_pipeline(
                step=args.step,
                dry_run=args.dry_run,
                data_dir=args.data_dir,
                include_bentoml=not args.no_bentoml,
                target=args.target,
            )
        )
    finally:
        sys.stdout = original_stdout

        if not args.no_save and output_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(args.output_dir) / f"csu_tier0_pipeline_run_{timestamp}.md"

            md_content = f"# CSU Tier 0 Pipeline Run Results\n\n"
            md_content += f"**Generated**: {datetime.now().isoformat()}\n"
            md_content += f"**Target**: {args.target}\n"
            md_content += f"**Data**: {args.data_dir}\n\n"
            md_content += "```\n"
            md_content += output_buffer.getvalue()
            md_content += "```\n"

            with open(output_file, "w") as f:
                f.write(md_content)

            print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
