"""Synthetic-data preview adapter — Phase 3 of the data-sufficiency rollout.

When the post-training learning-curve diagnostic recommends more data
(``recommended_additional_samples``), this adapter produces a *preview*
synthetic cohort sized to that recommendation so the operator can see what
additional data would look like — WITHOUT auto-mixing it into training.

Design note (deviates from the original rollout-plan spec, which was written
against APIs that don't exist):

* The plan said the predictive route calls ``E2IDataGenerator.generate_all
  (n=recommended)``. That is the wrong tool: ``E2IDataGenerator`` is a
  platform-database seeder (fixed volume, ``export_to_json`` only) — it does
  not produce a sized ``(X, y)`` training cohort. The only real sized-cohort
  generator is ``synthetic_v2.generate_scenario(scenario, seed, n_total)``,
  which returns a ``SyntheticDataset`` (X/y splits + audit metadata) and is
  already used across the codebase. This adapter uses it for all previews.
* The plan's trigger (``adaptive_verdict="INSUFFICIENT_TRAINING_DATA"``) does
  not exist; the orchestrator fires this adapter off the real post-training
  ``recommended_additional_samples`` signal instead.

The cohort is persisted to ``<output_root>/synthetic_preview_<workflow_id>/``
and a metadata dict is returned (attached to ``PipelineResult.synthetic_preview``).
The dataset is never mixed into training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Keep a preview a *preview*: floor so it's usable, ceiling so a huge
# recommendation (e.g. "+80k samples") doesn't trigger a costly full
# regeneration. The realized size + whether it was capped are recorded in
# the returned metadata so the operator knows it's a sample, not the full
# recommended cohort.
PREVIEW_MIN_N = 200
PREVIEW_MAX_N = 20_000


def build_synthetic_preview(
    *,
    scenario: str,
    recommended_n: int,
    workflow_id: str,
    output_root: str | Path = "pipeline_artifacts",
    seed: int = 42,
) -> dict[str, Any]:
    """Generate + persist a synthetic preview cohort. Never auto-mixes.

    Parameters
    ----------
    scenario
        A ``synthetic_v2`` ``ScenarioName`` value (e.g.
        ``"a_diagnostic_bc_idfs_balanced"``). Validated against the registry.
    recommended_n
        The learning-curve ``recommended_additional_samples``. Clamped to
        ``[PREVIEW_MIN_N, PREVIEW_MAX_N]`` for the preview.
    workflow_id
        Used to namespace the artifact directory.
    output_root
        Root under which ``synthetic_preview_<workflow_id>/`` is written.
    seed
        Generator seed (deterministic output).

    Returns
    -------
    A JSON-serializable metadata dict describing the preview (and recording
    ``auto_mixed_into_training=False``).

    Raises
    ------
    ValueError
        If ``scenario`` is not a registered ``ScenarioName``.
    """
    # Imported lazily so importing this module never drags in the (heavy)
    # synthetic_v2 stack unless a preview is actually requested.
    from src.ml.synthetic_v2.api import generate_scenario
    from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName

    try:
        scenario_enum = ScenarioName(scenario)
    except ValueError as exc:
        valid = sorted(s.value for s in ScenarioName)
        raise ValueError(
            f"Unknown synthetic_preview_scenario={scenario!r}; valid scenarios: {valid}"
        ) from exc
    if scenario_enum not in SCENARIO_REGISTRY:
        raise ValueError(f"scenario {scenario!r} is not registered in SCENARIO_REGISTRY")

    requested = int(recommended_n)
    n_total = max(PREVIEW_MIN_N, min(requested, PREVIEW_MAX_N))
    capped = n_total != requested

    dataset = generate_scenario(scenario_enum, seed=seed, n_total=n_total)

    out_dir = Path(output_root) / f"synthetic_preview_{workflow_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "preview_cohort.npz"
    np.savez(
        arrays_path,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
    )

    md = dataset.metadata
    meta: dict[str, Any] = {
        "scenario": md.scenario.value,
        "seed": md.seed,
        "preview_n_total": md.n_total,
        "n_train": md.n_train,
        "n_val": md.n_val,
        "n_test": md.n_test,
        "realized_prevalence": md.realized_prevalence,
        "target_prevalence": md.target_prevalence,
        "feature_names": list(md.feature_names),
        "audit_fingerprint": md.audit_fingerprint,
        "requested_recommended_n": requested,
        "preview_n_capped": capped,
        "artifacts_dir": str(out_dir),
        "arrays_file": str(arrays_path),
        # Load-bearing invariant: this preview is NOT mixed into training.
        # The operator must explicitly opt to use it downstream.
        "auto_mixed_into_training": False,
    }
    (out_dir / "preview_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    logger.info(
        "Synthetic preview generated: scenario=%s n_total=%d (requested=%d, capped=%s) "
        "→ %s  [NOT auto-mixed into training]",
        md.scenario.value,
        n_total,
        requested,
        capped,
        out_dir,
    )
    return meta
