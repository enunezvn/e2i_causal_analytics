"""Feast registrar node for data_preparer agent.

This node registers validated features in Feast feature store.
It executes AFTER validation and BEFORE baseline computation to ensure:
1. Features are registered for point-in-time retrieval
2. Feature freshness is checked as part of QC
3. Features are available for model_trainer

The freshness QC gate (#556, reshaped 2026-09-22 — owner decision on PR #2223)
-----------------------------------------------------------------------------
What it measures: the freshness of the Feast feature views SOURCED FROM this run's
``data_source`` (inverse of ``FEAST_FEATURE_VIEW_SOURCE_TABLES``), through the
feast-free #559 recency probe — ``MAX(<raw timestamp column>)`` of the source table
over PostgREST. For a table source that backs no Feast view, and for file sources
(``data_loader`` accepts ``file_dir`` AND ``files``), nothing is Feast-backed and the
result says so. No feast import is needed anywhere on this path (#307).

Why it was reshaped: the previous probe asked the adapter about
``feature_analyzer_<experiment_id>`` — a view present in no Feast registry and in no
source map — so its recency was always ``None``, every run was "unverifiable" and the
gate hard-blocked training on the worker image and on any box with feast installed. It
was unpassable by construction, and a worker-run retraining stopped at data-prep with
"Feast features stale; ALLOW_STALE_FEAST not set" (measured 2026-09-22).

When it blocks: ONLY when the run actually trains on Feast-served features, i.e.
``state["features_served_by_feast"]`` is True. The 2026-06-03 investigation already
framed the gate as a feature-SERVING concern (a file-sourced run's parquet features are
not affected by Feast staleness) and carved out ``file_dir`` runs; the same reasoning
holds for a Supabase-table run — ``data_loader`` reads the table, not the online store.
Nothing in the pipeline sets the flag today: ``model_trainer.split_loader`` takes its
Feast branch only when data_preparer handed it no splits, which the pipeline never does.
The block branch is retained behind the flag so a future Feast-read training path gets
the guarantee back without re-deriving it; ``ALLOW_STALE_FEAST=1`` keeps its meaning for
that branch only. For every other run the result is advisory: recorded in
``feast_freshness_check`` / ``feast_warnings`` / ``feast_registration_status``.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.feature_store.feast_source_freshness import (
    default_source_recency,
    describe_source,
    probe_source_freshness,
)

from ..state import DataPreparerState

logger = logging.getLogger(__name__)

# The #559 recency query the gate probes with. Module attribute (not a default arg) so a
# test can swap it at the seam without reaching into the probe module.
_source_recency_query = default_source_recency


def _get_feature_analyzer_adapter():
    """Get FeatureAnalyzerAdapter (lazy import to avoid circular deps)."""
    try:
        from src.feature_store.client import FeatureStoreClient
        from src.feature_store.feature_analyzer_adapter import (
            get_feature_analyzer_adapter,
        )

        # Create client - uses default Supabase config
        fs_client = FeatureStoreClient()
        return get_feature_analyzer_adapter(
            feature_store_client=fs_client,
            enable_feast=True,
        )
    except Exception as e:
        logger.warning(f"Could not initialize FeatureAnalyzerAdapter: {e}")
        return None


async def register_features_in_feast(state: DataPreparerState) -> Dict[str, Any]:
    """Register validated features in Feast feature store.

    This node:
    1. Registers features from validated training data in Feast
    2. Checks the freshness of the Feast views sourced from the run's data source
       (QC gate — see the module docstring for when it blocks)
    3. Syncs feature metadata to Feast registry

    The node runs AFTER transform_data and BEFORE compute_baseline_metrics.
    Feature registration is non-blocking - failures generate warnings, not errors.

    Args:
        state: Current agent state

    Returns:
        Updated state with Feast registration results
    """
    logger.info(f"Registering features in Feast for experiment {state['experiment_id']}")

    updates: Dict[str, Any] = {
        "feast_registration_status": "skipped",
        "feast_features_registered": 0,
        "feast_freshness_check": None,
        "feast_warnings": [],
        "feast_registered_at": None,
        # Indicates whether the Feast historical-features fallback was used
        # during this registration step.  Propagated to model_trainer so that
        # MLflow runs can be tagged accordingly.
        "feast_fallback_used": False,
        # Set to True when a Feast-SERVED run's features are stale/unverifiable and
        # ALLOW_STALE_FEAST is unset, which hard-blocks downstream training via the QC gate.
        "feast_blocked": False,
    }

    try:
        # Get adapter
        adapter = _get_feature_analyzer_adapter()
        if adapter is None:
            updates["feast_warnings"].append(
                "Feast adapter not available - skipping feature registration"
            )
            logger.warning("Feast adapter not available")
            return updates

        # Get training data
        train_df = state.get("train_df")
        if train_df is None:
            updates["feast_warnings"].append("No training data available for registration")
            return updates

        # Get scope spec for entity and feature info
        scope_spec = state.get("scope_spec", {})
        experiment_id = state["experiment_id"]
        entity_key = scope_spec.get("entity_key", "hcp_id")
        required_features = scope_spec.get("required_features", [])

        # Build feature metadata from training data
        generated_features = []
        for feature_name in required_features:
            if feature_name in train_df.columns:
                generated_features.append(
                    {
                        "name": feature_name,
                        "type": "prepared",
                        "transformation": "data_preparer",
                        "source": state.get("data_source", "unknown"),
                    }
                )

        # Build state dict for adapter
        adapter_state = {
            "generated_features": generated_features,
            "selected_features": required_features,
            "feature_importance": {},  # Not computed yet
            "X_train_selected": train_df[[f for f in required_features if f in train_df.columns]]
            if required_features
            else train_df,
        }

        # Register features
        registration_result = await adapter.register_features_from_state(
            state=adapter_state,
            experiment_id=experiment_id,
            entity_key=entity_key,
            owner="data_preparer",
            tags=["data_preparer", "validated", f"exp_{experiment_id}"],
        )

        updates["feast_features_registered"] = registration_result.get("features_registered", 0)
        updates["feast_registration_status"] = (
            "completed" if registration_result.get("features_registered", 0) > 0 else "empty"
        )

        if registration_result.get("errors"):
            for error in registration_result["errors"]:
                updates["feast_warnings"].append(
                    f"Registration error: {error.get('error', str(error))}"
                )

        # Freshness of the Feast views sourced from THIS run's data source (#559 probe).
        data_source = state.get("data_source")
        freshness_result = await _check_feature_freshness(data_source, experiment_id)
        updates["feast_freshness_check"] = freshness_result

        # Does this run train on Feast-SERVED features? Nothing sets it today (see the
        # module docstring); when a Feast-read training path exists it must set it.
        served_by_feast = bool(state.get("features_served_by_feast"))

        # Surface every recommendation; decide the gate.
        fresh = freshness_result.get("fresh") if freshness_result else None
        if freshness_result and fresh is not True:
            for recommendation in freshness_result.get("recommendations", []):
                updates["feast_warnings"].append(f"Freshness: {recommendation}")

            if not served_by_feast:
                # Advisory: the run's features come from its data source (a Supabase
                # table, files or the synthetic sample), not from the Feast online store.
                if fresh is False:
                    updates["feast_registration_status"] = "advisory_stale_features"
                logger.info(
                    "Feast QC gate: source freshness for experiment %s is %s "
                    "(data_source kind=%s, Feast-backed=%s) but the run does not train on "
                    "Feast-served features — advisory, not blocking.",
                    experiment_id,
                    "stale/unverifiable" if fresh is False else "not applicable",
                    describe_source(data_source),
                    freshness_result.get("feast_backed"),
                )
            # Feast-served run: hard block unless the ops escape hatch is set.
            elif os.environ.get("ALLOW_STALE_FEAST") != "1":
                updates["feast_blocked"] = True
                updates["feast_registration_status"] = "blocked_stale_features"
                # Append to blocking_issues so _finalize_output forces gate_passed=False.
                # We must merge against any existing blocking_issues already in state,
                # because subsequent state updates from other nodes will overwrite this
                # key only with the value we return here.
                existing_blockers = list(state.get("blocking_issues", []) or [])
                existing_blockers.append("Feast features stale; ALLOW_STALE_FEAST not set")
                updates["blocking_issues"] = existing_blockers
                logger.warning(
                    "Feast QC gate: features are stale/unverifiable for experiment %s and "
                    "the run trains on Feast-served features. Blocking training. Set "
                    "ALLOW_STALE_FEAST=1 to bypass (ops emergency only).",
                    experiment_id,
                )
            else:
                logger.warning(
                    "Feast QC gate: features are stale for experiment %s but "
                    "ALLOW_STALE_FEAST=1 is set — proceeding with stale features.",
                    experiment_id,
                )

        # Propagate fallback flag so model_trainer can tag the MLflow run.
        # FeatureAnalyzerAdapter.__init__ always sets ``_feast_client``
        # (it stores either the injected client or None), so we read
        # directly rather than via ``getattr(..., None)`` — the
        # defensive chain hid the contract. (Block 2 polish)
        feast_client = adapter._feast_client
        if feast_client is not None:
            updates["feast_fallback_used"] = getattr(feast_client, "_fallback_used", False)

        updates["feast_registered_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Feast registration completed: {updates['feast_features_registered']} features"
        )

        return updates

    except Exception as e:
        logger.error(f"Feast registration failed: {e}", exc_info=True)
        updates["feast_registration_status"] = "error"
        updates["feast_warnings"].append(f"Registration error: {str(e)}")
        return updates


async def _check_feature_freshness(
    data_source: Any,
    experiment_id: str,
    max_staleness_hours: float = 24.0,
) -> Optional[Dict[str, Any]]:
    """Freshness of the Feast views sourced from ``data_source`` (see the module docstring).

    On exception the function returns a dict with ``fresh=False`` (unverifiable — blocks
    a Feast-served run) unless ``ALLOW_STALE_FEAST=1`` is set in the environment, which
    is an ops-only escape hatch for known Feast/Supabase outages.

    Args:
        data_source: The run's data source (table name, file dict, ...)
        experiment_id: Experiment identifier (log context)
        max_staleness_hours: Maximum allowed staleness

    Returns:
        Freshness result dict (``probe_source_freshness`` shape).
    """
    try:
        return await probe_source_freshness(
            data_source,
            max_staleness_hours=max_staleness_hours,
            recency_query=_source_recency_query,
        )
    except Exception as e:
        allow_stale = os.environ.get("ALLOW_STALE_FEAST") == "1"
        logger.warning(
            "Feast freshness probe failed for experiment %s: %s. Treating as %s.",
            experiment_id,
            e,
            "fresh (ALLOW_STALE_FEAST=1)" if allow_stale else "stale",
        )
        if allow_stale:
            return {
                "fresh": True,
                "warning": str(e),
                "recommendations": [
                    "Feast freshness probe failed; verify Supabase/Feast are reachable."
                ],
            }
        return {
            "fresh": False,
            "error": str(e),
            "recommendations": [
                "Feast freshness probe failed; verify Supabase/Feast are reachable."
            ],
        }
