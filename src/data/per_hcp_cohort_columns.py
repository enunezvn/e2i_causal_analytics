"""The per-HCP cohort column contract: what the Digital Twin plants on ``business_metrics``.

``per_hcp_rollup`` rows have two writers. The per-HCP ETL (``src/etl/business_metrics_per_hcp_etl.py``)
recomputes the trigger-derived value columns from the base tables on every upsert. The plant
(``scripts/backfill_segment_engagement.py --execute``) writes the synthetic-gold DGP the twin
estimates from: eight treatment channels (migration 099) and one outcome (migration 147). The
ETL never touches these columns (``tests/unit/test_digital_twin/effect/test_cohort_columns_single_writer.py``),
but its reconcile DELETES whole rows the recompute no longer produces -- and on 2026-09-21 a
full-window backfill deleted rows wholesale, the planted data went with them, and the twin went
dark for every brand. The ETL's preview therefore reports the obsolete rows that still carry
these columns, which means the ETL and the twin must share ONE list of them.

This module is that list. It lives here, not under ``src/digital_twin``, because importing any
module of that package runs ``src/digital_twin/__init__``, which pulls sklearn, dowhy and shap
(15.9 s and +507 MB, measured 2026-09-22) -- far too heavy for an ETL that runs on the light
worker. ``src.data`` modules are side-effect-free by charter. ``src/digital_twin/effect/provider.py``
re-exports these names, so every twin-side consumer keeps its import.
"""

from __future__ import annotations

from typing import Final

#: Canonical Digital Twin intervention -> the ``business_metrics`` column that carries its
#: planted treatment (see the DGP in ``scripts/backfill_segment_engagement.py``).
INTERVENTION_TREATMENT_MAP: Final[dict[str, str]] = {
    "email_campaign": "email_campaign_count",
    "call_frequency_increase": "call_frequency",
    "speaker_program_invitation": "speaker_program_count",
    "sample_distribution": "sample_volume",
    "peer_influence_activation": "peer_influence_score",
    "digital_engagement": "engagement_score",
    "patient_support_program": "patient_support_enrollment",
    "rep_training_quality": "rep_training_score",
}

#: The cohort OUTCOME has its own column (migration 147) so the ETL's ``conversion_rate``
#: recompute and the plant stop sharing one.
COHORT_OUTCOME_COLUMN: Final = "cohort_conversion_outcome"

#: Every column the plant writes, in a fixed order: the eight channels sorted, then the outcome.
PLANTED_COLUMNS: Final[tuple[str, ...]] = (
    *sorted(set(INTERVENTION_TREATMENT_MAP.values())),
    COHORT_OUTCOME_COLUMN,
)
