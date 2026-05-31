"""Anti-null-trap helpers for Feast online-feature responses (#576 / #532).

A Feast feature-server ``/get-online-features`` 200 response can carry
``PRESENT``-but-null feature values. This was verified LIVE against the prod
sidecar: a single-key ``{patient_id}`` lookup against ``patient_journey_features``
(whose entity key is the composite ``[patient_id, patient_brand_id]``) returns
``status=PRESENT`` with ``value=null`` when the composite key is absent or the
online store is empty/stale. Labeling such a response
``feature_source='feast_online'`` would feed null features to the model while
presenting them as real, audit-grade, Feast-sourced data — the exact #532 harm,
and worse than an honest 503 because no exception fires to catch it.

These helpers let the predictions and explain routes detect an incomplete Feast
response and fail loud (503) instead of mislabeling it. A real ``0`` / ``0.0`` is
a legitimate value and is NOT a violation — zero is not null. (The COALESCE-0
source-masking issue, where a NULL source column materializes as a fabricated 0,
is a data-layer concern that the route guard provably cannot distinguish from a
real 0; it is handled at data population/materialize time, not here.)

Pure stdlib only (no ``src.*`` imports), mirroring the architectural constraint
on :mod:`src.feature_store.model_feature_refs`, so the module is safe to reuse
from any layer.
"""

from __future__ import annotations

from typing import Any, List, Mapping


def required_feature_fields(feature_refs: List[str]) -> List[str]:
    """Return the feature field names a Feast response must carry non-null.

    A feature ref is ``"<feature_view>:<field>"``; the required field is the
    part after the first colon. The entity-key column the feature server echoes
    back (e.g. ``patient_id``) is not a ref field, so it is naturally excluded
    from the result. A wildcard ref (``"<view>:*"``) names no enumerable field
    and is skipped (the routes always pass explicit field refs).
    """
    fields: List[str] = []
    for ref in feature_refs:
        field = ref.split(":", 1)[1] if ":" in ref else ref
        if field and field != "*":
            fields.append(field)
    return fields


def missing_or_null_feature_fields(
    payload: Mapping[str, Any], feature_refs: List[str]
) -> List[str]:
    """Return the required ref fields that are absent from, or null in, ``payload``.

    ``payload`` is the collapsed single-row feature dict (``{field: value}``)
    produced by the route from a Feast response. A field is a violation when it
    is missing from ``payload`` or its value is ``None``. A real ``0`` / ``0.0``
    is not a violation. An empty list means every required feature is present
    and non-null (the honest ``feast_online`` success condition).
    """
    return [
        field
        for field in required_feature_fields(feature_refs)
        if field not in payload or payload[field] is None
    ]
