#!/usr/bin/env python3
"""Feast Feature Store Migration Validation Script.

This script validates feature parity between the custom feature store
and Feast during migration. It supports shadow mode for A/B comparison.

Usage:
    # Validate feature parity
    python scripts/feast_validate_migration.py --mode validate

    # Run shadow mode comparison
    python scripts/feast_validate_migration.py --mode shadow --samples 100

    # Performance benchmark
    python scripts/feast_validate_migration.py --mode benchmark

    # Export custom features to Feast format
    python scripts/feast_validate_migration.py --mode export
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_store.client import FeatureStoreClient
from src.feature_store.feast_client import FeastClient, FeastConfig, get_feast_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MigrationValidator:
    """Validates feature parity between custom and Feast stores."""

    def __init__(
        self,
        custom_client: Optional[FeatureStoreClient] = None,
        feast_client: Optional[FeastClient] = None,
    ):
        """Initialize migration validator.

        Args:
            custom_client: Custom feature store client.
            feast_client: Feast client.
        """
        self.custom_client = custom_client
        self.feast_client = feast_client
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize both clients.

        Returns:
            True if initialization successful.
        """
        if self._initialized:
            return True

        try:
            if self.custom_client is None:
                self.custom_client = FeatureStoreClient()

            if self.feast_client is None:
                self.feast_client = await get_feast_client()

            await self.feast_client.initialize()
            self._initialized = True
            logger.info("Migration validator initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize migration validator: {e}")
            return False

    async def validate_feature_parity(
        self,
        entity_ids: List[Dict[str, str]],
        feature_refs: List[str],
        tolerance: float = 0.001,
    ) -> Dict[str, Any]:
        """Validate feature parity between custom and Feast stores.

        Args:
            entity_ids: List of entity ID dicts.
            feature_refs: Feature references to compare.
            tolerance: Numeric tolerance for comparison.

        Returns:
            Validation result dict.
        """
        if not await self.initialize():
            return {"status": "failed", "error": "Initialization failed"}

        logger.info(
            f"Validating parity for {len(entity_ids)} entities, "
            f"{len(feature_refs)} features"
        )

        results = {
            "status": "completed",
            "total_entities": len(entity_ids),
            "total_features": len(feature_refs),
            "matches": 0,
            "mismatches": [],
            "custom_only": [],
            "feast_only": [],
            "errors": [],
        }

        for entity in entity_ids:
            try:
                # Get from custom store
                custom_features = await self._get_custom_features(entity, feature_refs)

                # Get from Feast
                feast_features = await self._get_feast_features(entity, feature_refs)

                # Compare
                comparison = self._compare_features(
                    custom_features, feast_features, tolerance
                )

                if comparison["match"]:
                    results["matches"] += 1
                else:
                    results["mismatches"].append({
                        "entity": entity,
                        "differences": comparison["differences"],
                    })

            except Exception as e:
                results["errors"].append({
                    "entity": entity,
                    "error": str(e),
                })

        # Calculate summary
        total = len(entity_ids)
        results["match_rate"] = results["matches"] / total if total > 0 else 0
        results["parity_achieved"] = results["match_rate"] >= 0.99  # 99% threshold

        return results

    async def _get_custom_features(
        self,
        entity: Dict[str, str],
        feature_refs: List[str],
    ) -> Dict[str, Any]:
        """Get features from custom store."""
        # Parse feature refs to get group and feature names
        features = {}

        for ref in feature_refs:
            parts = ref.split(":")
            if len(parts) == 2:
                group_name, feature_name = parts
                # Construct entity key
                entity_key = "_".join(entity.values())

                try:
                    value = await asyncio.to_thread(
                        self.custom_client.get_feature,
                        group_name=group_name,
                        feature_name=feature_name,
                        entity_id=entity_key,
                    )
                    features[ref] = value
                except Exception:
                    features[ref] = None

        return features

    async def _get_feast_features(
        self,
        entity: Dict[str, str],
        feature_refs: List[str],
    ) -> Dict[str, Any]:
        """Get features from Feast."""
        try:
            result = await self.feast_client.get_online_features(
                entity_rows=[entity],
                feature_refs=feature_refs,
            )
            # Convert to dict format
            features = {}
            for ref in feature_refs:
                # Feast returns with __ separator
                key = ref.replace(":", "__")
                if key in result:
                    features[ref] = result[key][0] if result[key] else None
            return features
        except Exception as e:
            logger.warning(f"Feast fetch failed: {e}")
            return {}

    def _compare_features(
        self,
        custom: Dict[str, Any],
        feast: Dict[str, Any],
        tolerance: float,
    ) -> Dict[str, Any]:
        """Compare feature values from both stores."""
        differences = []

        all_keys = set(custom.keys()) | set(feast.keys())

        for key in all_keys:
            custom_val = custom.get(key)
            feast_val = feast.get(key)

            if custom_val is None and feast_val is None:
                continue
            elif custom_val is None:
                differences.append({
                    "feature": key,
                    "custom": None,
                    "feast": feast_val,
                    "reason": "missing_in_custom",
                })
            elif feast_val is None:
                differences.append({
                    "feature": key,
                    "custom": custom_val,
                    "feast": None,
                    "reason": "missing_in_feast",
                })
            elif isinstance(custom_val, (int, float)) and isinstance(feast_val, (int, float)):
                if abs(custom_val - feast_val) > tolerance:
                    differences.append({
                        "feature": key,
                        "custom": custom_val,
                        "feast": feast_val,
                        "reason": "value_mismatch",
                        "diff": abs(custom_val - feast_val),
                    })
            elif custom_val != feast_val:
                differences.append({
                    "feature": key,
                    "custom": custom_val,
                    "feast": feast_val,
                    "reason": "value_mismatch",
                })

        return {
            "match": len(differences) == 0,
            "differences": differences,
        }

    async def run_shadow_mode(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str],
        sample_size: int = 100,
    ) -> Dict[str, Any]:
        """Run shadow mode comparison on a sample of entities.

        This fetches features from both stores and compares results
        without affecting production traffic.

        Args:
            entity_df: DataFrame with entity IDs and timestamps.
            feature_refs: Feature references to compare.
            sample_size: Number of entities to sample.

        Returns:
            Shadow mode result dict.
        """
        if not await self.initialize():
            return {"status": "failed", "error": "Initialization failed"}

        # Sample entities
        if len(entity_df) > sample_size:
            sample_df = entity_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = entity_df

        logger.info(f"Running shadow mode on {len(sample_df)} entities")

        results = {
            "status": "completed",
            "mode": "shadow",
            "sample_size": len(sample_df),
            "feature_refs": feature_refs,
            "custom_latencies_ms": [],
            "feast_latencies_ms": [],
            "parity_results": [],
        }

        for _, row in sample_df.iterrows():
            entity = row.to_dict()

            # Time custom fetch
            start = time.perf_counter()
            custom_features = await self._get_custom_features(entity, feature_refs)
            custom_latency = (time.perf_counter() - start) * 1000
            results["custom_latencies_ms"].append(custom_latency)

            # Time Feast fetch
            start = time.perf_counter()
            feast_features = await self._get_feast_features(entity, feature_refs)
            feast_latency = (time.perf_counter() - start) * 1000
            results["feast_latencies_ms"].append(feast_latency)

            # Compare
            comparison = self._compare_features(custom_features, feast_features, 0.001)
            results["parity_results"].append(comparison["match"])

        # Calculate statistics
        results["statistics"] = {
            "custom": {
                "mean_latency_ms": statistics.mean(results["custom_latencies_ms"]),
                "p50_latency_ms": statistics.median(results["custom_latencies_ms"]),
                "p95_latency_ms": self._percentile(results["custom_latencies_ms"], 95),
                "p99_latency_ms": self._percentile(results["custom_latencies_ms"], 99),
            },
            "feast": {
                "mean_latency_ms": statistics.mean(results["feast_latencies_ms"]),
                "p50_latency_ms": statistics.median(results["feast_latencies_ms"]),
                "p95_latency_ms": self._percentile(results["feast_latencies_ms"], 95),
                "p99_latency_ms": self._percentile(results["feast_latencies_ms"], 99),
            },
            "parity": {
                "match_rate": sum(results["parity_results"]) / len(results["parity_results"]),
                "total_matches": sum(results["parity_results"]),
                "total_mismatches": len(results["parity_results"]) - sum(results["parity_results"]),
            },
        }

        return results

    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f < len(sorted_data) - 1 else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

    async def benchmark_performance(
        self,
        entity_count: int = 100,
        iterations: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark feature retrieval performance.

        Args:
            entity_count: Number of entities to fetch per iteration.
            iterations: Number of iterations to run.

        Returns:
            Benchmark result dict.
        """
        if not await self.initialize():
            return {"status": "failed", "error": "Initialization failed"}

        logger.info(
            f"Running benchmark: {entity_count} entities x {iterations} iterations"
        )

        # Generate test entities
        test_entities = [
            {"hcp_id": f"hcp_{i:05d}", "brand_id": "remibrutinib"}
            for i in range(entity_count)
        ]

        feature_refs = ["hcp_conversion_features:engagement_score"]

        custom_times = []
        feast_times = []

        for i in range(iterations):
            # Benchmark custom store
            start = time.perf_counter()
            for entity in test_entities:
                await self._get_custom_features(entity, feature_refs)
            custom_times.append(time.perf_counter() - start)

            # Benchmark Feast
            start = time.perf_counter()
            try:
                await self.feast_client.get_online_features(
                    entity_rows=test_entities,
                    feature_refs=feature_refs,
                )
            except Exception:
                # Feast may not be fully configured
                pass
            feast_times.append(time.perf_counter() - start)

            logger.info(f"Iteration {i + 1}/{iterations} complete")

        return {
            "status": "completed",
            "entity_count": entity_count,
            "iterations": iterations,
            "custom_store": {
                "mean_seconds": statistics.mean(custom_times),
                "std_seconds": statistics.stdev(custom_times) if len(custom_times) > 1 else 0,
                "min_seconds": min(custom_times),
                "max_seconds": max(custom_times),
                "qps": entity_count / statistics.mean(custom_times),
            },
            "feast": {
                "mean_seconds": statistics.mean(feast_times),
                "std_seconds": statistics.stdev(feast_times) if len(feast_times) > 1 else 0,
                "min_seconds": min(feast_times),
                "max_seconds": max(feast_times),
                "qps": entity_count / statistics.mean(feast_times) if statistics.mean(feast_times) > 0 else 0,
            },
        }

    async def export_to_feast_format(
        self,
        output_path: Path,
        feature_groups: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Export custom store features to Feast-compatible format.

        Args:
            output_path: Path to save exported features.
            feature_groups: Specific groups to export. None = all.

        Returns:
            Export result dict.
        """
        if not await self.initialize():
            return {"status": "failed", "error": "Initialization failed"}

        logger.info(f"Exporting features to {output_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        # Get all feature groups from custom store
        try:
            groups = self.custom_client.list_feature_groups()
            if feature_groups:
                groups = [g for g in groups if g.name in feature_groups]
        except Exception as e:
            return {"status": "failed", "error": f"Failed to list groups: {e}"}

        exported = []

        for group in groups:
            try:
                # Get all features in group
                features = self.custom_client.list_features(group_name=group.name)

                # Convert to Feast format
                feast_format = {
                    "feature_view_name": group.name,
                    "entities": group.entity_ids if hasattr(group, "entity_ids") else [],
                    "features": [],
                    "tags": {"source": "custom_store", "exported_at": datetime.now(timezone.utc).isoformat()},
                }

                for feature in features:
                    feast_format["features"].append({
                        "name": feature.name,
                        "dtype": str(feature.value_type),
                        "description": feature.description if hasattr(feature, "description") else "",
                    })

                # Save to file
                output_file = output_path / f"{group.name}.json"
                with open(output_file, "w") as f:
                    json.dump(feast_format, f, indent=2)

                exported.append({
                    "group": group.name,
                    "features": len(feast_format["features"]),
                    "file": str(output_file),
                })

            except Exception as e:
                logger.error(f"Failed to export {group.name}: {e}")

        return {
            "status": "completed",
            "exported_groups": len(exported),
            "details": exported,
            "output_path": str(output_path),
        }

    async def close(self):
        """Clean up resources."""
        if self.feast_client:
            await self.feast_client.close()
        self._initialized = False


async def main():
    """Main entry point for migration validation."""
    parser = argparse.ArgumentParser(
        description="Feast Feature Store Migration Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["validate", "shadow", "benchmark", "export"],
        default="validate",
        help="Validation mode",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples for shadow mode",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for benchmark",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="exports/feast",
        help="Output path for export mode",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validator = MigrationValidator()

    try:
        if args.mode == "validate":
            # Sample validation
            test_entities = [
                {"hcp_id": f"hcp_{i:05d}", "brand_id": "remibrutinib"}
                for i in range(10)
            ]
            result = await validator.validate_feature_parity(
                entity_ids=test_entities,
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

        elif args.mode == "shadow":
            # Create sample DataFrame
            sample_df = pd.DataFrame({
                "hcp_id": [f"hcp_{i:05d}" for i in range(args.samples)],
                "brand_id": ["remibrutinib"] * args.samples,
                "event_timestamp": [datetime.now(timezone.utc)] * args.samples,
            })
            result = await validator.run_shadow_mode(
                entity_df=sample_df,
                feature_refs=["hcp_conversion_features:engagement_score"],
                sample_size=args.samples,
            )

        elif args.mode == "benchmark":
            result = await validator.benchmark_performance(
                entity_count=100,
                iterations=args.iterations,
            )

        elif args.mode == "export":
            result = await validator.export_to_feast_format(
                output_path=Path(args.output),
            )

        # Print result
        print("\n" + "=" * 60)
        print(f"MIGRATION VALIDATION RESULT - {args.mode.upper()}")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
        print("=" * 60)

    finally:
        await validator.close()


if __name__ == "__main__":
    asyncio.run(main())
