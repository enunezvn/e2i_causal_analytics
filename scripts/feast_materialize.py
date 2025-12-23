#!/usr/bin/env python3
"""Feast Feature Materialization Script.

This script handles materialization of features from offline to online store.
It supports both full and incremental materialization modes.

Usage:
    # Full materialization for all feature views
    python scripts/feast_materialize.py --mode full

    # Incremental materialization (since last run)
    python scripts/feast_materialize.py --mode incremental

    # Specific feature views only
    python scripts/feast_materialize.py --feature-views hcp_conversion_features patient_journey_features

    # Dry run (validate only)
    python scripts/feast_materialize.py --dry-run

    # With freshness alerting
    python scripts/feast_materialize.py --check-freshness --alert-threshold 24.0
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_store.feast_client import FeastClient, FeastConfig, get_feast_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MaterializationJob:
    """Manages Feast feature materialization jobs."""

    def __init__(
        self,
        feast_client: Optional[FeastClient] = None,
        config: Optional[FeastConfig] = None,
    ):
        """Initialize materialization job.

        Args:
            feast_client: Feast client instance. If None, creates one.
            config: Feast configuration. Uses defaults if None.
        """
        self.feast_client = feast_client
        self.config = config or FeastConfig()
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the Feast client.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        try:
            if self.feast_client is None:
                self.feast_client = await get_feast_client(self.config)

            await self.feast_client.initialize()
            self._initialized = True
            logger.info("Materialization job initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize materialization job: {e}")
            return False

    async def run_full_materialization(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run full materialization from start to end date.

        Args:
            start_date: Start of materialization window
            end_date: End of materialization window
            feature_views: Specific feature views to materialize. None = all.
            dry_run: If True, validate only without materializing.

        Returns:
            Materialization result dict
        """
        if not await self.initialize():
            return {
                "status": "failed",
                "error": "Failed to initialize Feast client",
            }

        logger.info(
            f"Starting full materialization: {start_date.isoformat()} to "
            f"{end_date.isoformat()}"
        )
        if feature_views:
            logger.info(f"Feature views: {', '.join(feature_views)}")
        else:
            logger.info("Feature views: all")

        if dry_run:
            logger.info("DRY RUN - validating only")
            return await self._validate_materialization(feature_views)

        try:
            result = await self.feast_client.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=feature_views,
            )

            result["mode"] = "full"
            result["start_date"] = start_date.isoformat()
            result["end_date"] = end_date.isoformat()

            logger.info(
                f"Full materialization complete: status={result['status']}, "
                f"duration={result.get('duration_seconds', 0):.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Full materialization failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "mode": "full",
            }

    async def run_incremental_materialization(
        self,
        end_date: Optional[datetime] = None,
        feature_views: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run incremental materialization since last run.

        Args:
            end_date: End of materialization window. Defaults to now.
            feature_views: Specific feature views to materialize. None = all.
            dry_run: If True, validate only without materializing.

        Returns:
            Materialization result dict
        """
        if not await self.initialize():
            return {
                "status": "failed",
                "error": "Failed to initialize Feast client",
            }

        end_date = end_date or datetime.now(timezone.utc)
        logger.info(f"Starting incremental materialization to {end_date.isoformat()}")

        if feature_views:
            logger.info(f"Feature views: {', '.join(feature_views)}")
        else:
            logger.info("Feature views: all")

        if dry_run:
            logger.info("DRY RUN - validating only")
            return await self._validate_materialization(feature_views)

        try:
            result = await self.feast_client.materialize_incremental(
                end_date=end_date,
                feature_views=feature_views,
            )

            result["mode"] = "incremental"
            result["end_date"] = end_date.isoformat()

            logger.info(
                f"Incremental materialization complete: status={result['status']}, "
                f"duration={result.get('duration_seconds', 0):.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Incremental materialization failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "mode": "incremental",
            }

    async def check_feature_freshness(
        self,
        feature_views: Optional[List[str]] = None,
        max_staleness_hours: float = 24.0,
    ) -> Dict[str, Any]:
        """Check feature freshness and identify stale features.

        Args:
            feature_views: Feature views to check. None = all.
            max_staleness_hours: Max hours since last update.

        Returns:
            Freshness report dict
        """
        if not await self.initialize():
            return {
                "status": "failed",
                "error": "Failed to initialize Feast client",
            }

        logger.info("Checking feature freshness...")

        try:
            # Get all feature views if not specified
            if not feature_views:
                views = await self.feast_client.list_feature_views()
                feature_views = [v["name"] for v in views]

            stale_features = []
            fresh_features = []
            errors = []

            now = datetime.now(timezone.utc)

            for fv_name in feature_views:
                try:
                    # Get statistics for each feature in the view
                    stats = await self.feast_client.get_feature_statistics(
                        feature_view=fv_name,
                        feature_name="*",  # All features
                    )

                    if stats and stats.last_updated:
                        age_hours = (now - stats.last_updated).total_seconds() / 3600
                        if age_hours > max_staleness_hours:
                            stale_features.append({
                                "feature_view": fv_name,
                                "last_updated": stats.last_updated.isoformat(),
                                "age_hours": age_hours,
                            })
                        else:
                            fresh_features.append({
                                "feature_view": fv_name,
                                "last_updated": stats.last_updated.isoformat(),
                                "age_hours": age_hours,
                            })
                    else:
                        errors.append({
                            "feature_view": fv_name,
                            "error": "No statistics available",
                        })

                except Exception as e:
                    errors.append({
                        "feature_view": fv_name,
                        "error": str(e),
                    })

            result = {
                "status": "completed",
                "fresh": len(stale_features) == 0,
                "stale_features": stale_features,
                "fresh_features": fresh_features,
                "errors": errors,
                "max_staleness_hours": max_staleness_hours,
                "checked_at": now.isoformat(),
            }

            # Log summary
            if stale_features:
                logger.warning(
                    f"Found {len(stale_features)} stale feature view(s): "
                    f"{', '.join(f['feature_view'] for f in stale_features)}"
                )
            else:
                logger.info(f"All {len(fresh_features)} feature views are fresh")

            return result

        except Exception as e:
            logger.error(f"Freshness check failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    async def _validate_materialization(
        self,
        feature_views: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate materialization configuration without executing.

        Args:
            feature_views: Feature views to validate.

        Returns:
            Validation result dict
        """
        try:
            # List all feature views
            all_views = await self.feast_client.list_feature_views()
            view_names = [v["name"] for v in all_views]

            if feature_views:
                # Validate specified views exist
                missing = [fv for fv in feature_views if fv not in view_names]
                if missing:
                    return {
                        "status": "failed",
                        "error": f"Feature views not found: {', '.join(missing)}",
                        "available_views": view_names,
                    }
                validated_views = feature_views
            else:
                validated_views = view_names

            return {
                "status": "validated",
                "feature_views": validated_views,
                "total_views": len(validated_views),
                "dry_run": True,
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
            }

    async def close(self):
        """Clean up resources."""
        if self.feast_client:
            await self.feast_client.close()
            self._initialized = False


async def main():
    """Main entry point for materialization script."""
    parser = argparse.ArgumentParser(
        description="Feast Feature Materialization Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="Materialization mode (default: incremental)",
    )

    parser.add_argument(
        "--feature-views",
        nargs="+",
        help="Specific feature views to materialize (default: all)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for full materialization (ISO format, e.g., 2024-01-01)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for materialization (ISO format, default: now)",
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Days back for full materialization if no start-date (default: 7)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without materializing",
    )

    parser.add_argument(
        "--check-freshness",
        action="store_true",
        help="Check feature freshness after materialization",
    )

    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=24.0,
        help="Freshness alert threshold in hours (default: 24.0)",
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

    # Parse dates
    end_date = None
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date.replace("Z", "+00:00"))
    else:
        end_date = datetime.now(timezone.utc)

    start_date = None
    if args.mode == "full":
        if args.start_date:
            start_date = datetime.fromisoformat(args.start_date.replace("Z", "+00:00"))
        else:
            start_date = end_date - timedelta(days=args.days_back)

    # Run materialization
    job = MaterializationJob()

    try:
        if args.mode == "full":
            result = await job.run_full_materialization(
                start_date=start_date,
                end_date=end_date,
                feature_views=args.feature_views,
                dry_run=args.dry_run,
            )
        else:
            result = await job.run_incremental_materialization(
                end_date=end_date,
                feature_views=args.feature_views,
                dry_run=args.dry_run,
            )

        # Check freshness if requested
        if args.check_freshness and result.get("status") != "failed":
            freshness_result = await job.check_feature_freshness(
                feature_views=args.feature_views,
                max_staleness_hours=args.alert_threshold,
            )
            result["freshness_check"] = freshness_result

            # Exit with error if stale features found
            if not freshness_result.get("fresh", True):
                logger.warning("Stale features detected - exiting with error code")
                await job.close()
                sys.exit(1)

        # Print final result
        print("\n" + "=" * 60)
        print("MATERIALIZATION RESULT")
        print("=" * 60)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Mode: {result.get('mode', args.mode)}")

        if result.get("status") == "failed":
            print(f"Error: {result.get('error', 'unknown')}")
            await job.close()
            sys.exit(1)

        if result.get("duration_seconds"):
            print(f"Duration: {result['duration_seconds']:.2f}s")

        if "freshness_check" in result:
            fc = result["freshness_check"]
            print(f"\nFreshness Check:")
            print(f"  Fresh: {fc.get('fresh', 'unknown')}")
            print(f"  Stale features: {len(fc.get('stale_features', []))}")
            print(f"  Fresh features: {len(fc.get('fresh_features', []))}")

        print("=" * 60)

    finally:
        await job.close()


if __name__ == "__main__":
    asyncio.run(main())
