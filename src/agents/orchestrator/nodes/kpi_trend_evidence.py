"""Real KPI-history evidence for explainer-routed trend asks (#2191).

The orchestrator has no trend agent or frontend-action dispatch seam.  The
platform does, however, have one authoritative trend consumer: the monthly
``kpi_history`` series used by the Time-Series page and ``renderKpiTrend``.
This module lets the synchronous explainer input resolver consume that same
series without ever substituting the scalar KPI calculator.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.repositories.provenance import coerce_provenance_flag

from .kpi_clarify import KPI_LOOKUP_CONFIDENCE, brand_clarify_for_ask, region_clarify_evidence
from .structured_scope import _structured_brand

logger = logging.getLogger(__name__)

_TREND_HEADS = frozenset({"trend", "trends", "trajectory", "evolution", "history"})
_TREND_HEAD_RE = re.compile(r"\b(?:trends?|trajectory|evolution|history)\b", re.I)
_CHANGE_RE = re.compile(r"\b(?:evolv(?:e|es|ed|ing)|chang(?:e|es|ed|ing))\b", re.I)
_CUE_RE = re.compile(
    r"\b(?:what(?:'?s| is| are| was| were)|show me|tell me about|give me|how many|can you)\b",
    re.I,
)
# Scopes the monthly history endpoint cannot represent.  This is deliberately
# limited to fronted shapes known to bypass the old lookup grammar; supported
# fronted brand/region/window scopes remain eligible.
_UNSUPPORTED_FRONT_SCOPE_RE = re.compile(
    r"\b(?:by|among|amongst|at\s+(?:each|every)|for\s+(?:each|every))\b"
    r"|\bfor\s+(?:high|medium|low)[- ]severity\s+patients?\b"
    r"|\b(?:drivers?|determinants?|causes?|impacts?|effects?)\s+for\b",
    re.I,
)
_FINER_GRAIN_RE = re.compile(r"\b(?:daily|weekly|day|days|week|weeks)\b", re.I)
_FINDING_POINT_LIMIT = 12


def _has_direct_trend_relation(normalized: str, start: int, end: int) -> bool:
    """Whether the trend/change term structurally governs this KPI mention."""
    # Runtime import avoids a module cycle: dispatcher invokes this helper only
    # after its own head utilities have been defined.
    from .dispatcher import _kpi_governing_of_head, _kpi_right_head

    if _kpi_governing_of_head(normalized, start) in _TREND_HEADS:
        return True
    if _kpi_right_head(normalized, end) in _TREND_HEADS:
        return True
    tail = normalized[end:]
    return _CHANGE_RE.match(tail.lstrip()) is not None


def _as_value_shape(normalized: str) -> str:
    """Replace only the trend operator while preserving string coordinates.

    The established scalar guard already validates all representable
    brand/region/window scope around a KPI.  Recasting ``trend of KPI`` as
    ``value of KPI`` (and blanking a post-KPI change verb) lets this consumer
    reuse that safety boundary without teaching the scalar path that a series
    noun is a scalar value-head.
    """
    chars = list(normalized)
    for match in _TREND_HEAD_RE.finditer(normalized):
        token = match.group(0).lower()
        replacement = "values" if token == "trends" else "value"
        replacement = replacement + " " * (len(token) - len(replacement))
        chars[match.start() : match.end()] = replacement
    for match in _CHANGE_RE.finditer(normalized):
        chars[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(chars)


def _front_scope_is_representable(query: str) -> bool:
    cue = _CUE_RE.search(query)
    if cue is None:
        return True
    return _UNSUPPORTED_FRONT_SCOPE_RE.search(query[: cue.start()]) is None


def _series_finding(scope: str, points: List[Dict[str, Any]], kpi: Any) -> str:
    from .dispatcher import _format_kpi_value

    shown = points[-_FINDING_POINT_LIMIT:]
    values = ", ".join(
        f"{point['metric_date']}={_format_kpi_value(float(point['value']), kpi)}" for point in shown
    )
    shown_label = (
        f"latest {len(shown)} shown" if len(points) > _FINDING_POINT_LIMIT else "all points shown"
    )
    return (
        f"{scope} monthly history: {len(points)} points from "
        f"{points[0]['metric_date']} through {points[-1]['metric_date']} "
        f"({shown_label}): {values}"
    )


def resolve_kpi_trend_evidence(
    agent_input: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Resolve an explainer-routed KPI trend to materialized monthly points.

    ``None`` means this is not a KPI trend ask.  An empty list means it *is* a
    trend ask but cannot be represented or has no real series; callers must
    preserve that distinction so they do not fall through to a scalar or stale
    prior-turn evidence.
    """
    query = agent_input.get("query")
    if not isinstance(query, str) or not query.strip():
        return None

    from src.services.kpi_resolution import (
        KPI_SEMANTIC_NOTES,
        mask_spans,
        owned_mention_spans,
        recognize_distinct_metric,
        recognize_kpi_span,
    )

    from .intent_classifier import KPI_VALUE_LOOKUP_RE

    if not KPI_VALUE_LOOKUP_RE.search(query):
        return None
    match = recognize_kpi_span(query)
    if match is None:
        return None
    kpi, normalized, start, end = match
    if not _has_direct_trend_relation(normalized, start, end):
        return None

    # From here on the query is trend-shaped.  Every refusal returns [] so the
    # explainer cannot answer it with one scalar or an unrelated prior result.
    if not _front_scope_is_representable(query):
        return []
    from .kpi_value_guard import value_lookup_mentions_supported

    if not value_lookup_mentions_supported(
        _as_value_shape(normalized),
        kpi.id,
        start,
        end,
        structured_brand=_structured_brand(agent_input),
    ):
        return []
    masked = mask_spans(normalized, owned_mention_spans(normalized, kpi.id, start, end))
    if recognize_distinct_metric(masked, exclude_id=kpi.id, original_query=query) is not None:
        return []

    from .dispatcher import _extract_brand_region, _window_from_query

    decided_brand = _structured_brand(agent_input)
    brand, region = _extract_brand_region(agent_input)
    if region is None:
        from src.services.query_entities import region_scan

        ambiguous = region_scan(query).ambiguous_phrase
        if ambiguous is not None:
            return [region_clarify_evidence(kpi, ambiguous)]
    brand_clarify = brand_clarify_for_ask(kpi, query, decided_brand)
    if brand_clarify is not None:
        return [brand_clarify]

    window = _window_from_query(query)
    try:
        from src.repositories.kpi_history import get_kpi_history_sync

        rows = get_kpi_history_sync(
            kpi.id,
            brand=brand,
            region=region,
            start_date=window.get("start") if window else None,
            end_date=window.get("end") if window else None,
        )
    except Exception as exc:  # noqa: BLE001 - any read failure must fail closed
        logger.warning(
            "explainer trend resolver: kpi_history read for %s raised (%s) -> failing closed",
            kpi.id,
            exc,
        )
        return []

    points = [
        {
            "metric_date": str(row["metric_date"]),
            "value": float(row["value"]),
            "status": row.get("status"),
        }
        for row in rows
        if row.get("metric_date") and row.get("value") is not None
    ]
    if not points:
        logger.info(
            "explainer trend resolver: no kpi_history points for %s brand=%r region=%r",
            kpi.id,
            brand,
            region,
        )
        return []

    scope = " ".join(part for part in (brand, kpi.name) if part)
    if region:
        scope = f"{scope} in {region}"
    warnings: List[str] = []
    if _FINER_GRAIN_RE.search(query):
        warnings.append(
            "The request names daily or weekly granularity, but the authoritative KPI history "
            "is monthly; these points do not support within-month variation."
        )
    findings = [_series_finding(scope, points, kpi)]
    semantic_note = KPI_SEMANTIC_NOTES.get(kpi.id)
    if semantic_note:
        findings.append(semantic_note)
        warnings.append(semantic_note)

    from src.kpi.measure_basis import materialized_history_basis

    synthetic = any(coerce_provenance_flag(row.get("is_synthetic")) for row in rows)
    payload: Dict[str, Any] = {
        "agent": "kpi_history",
        "analysis_type": "kpi_trend",
        "key_findings": findings,
        "warnings": warnings,
        "confidence": KPI_LOOKUP_CONFIDENCE,
        "kpi_id": kpi.id,
        "kpi_name": kpi.name,
        "brand": brand,
        "region": region,
        "grain": "monthly",
        "point_count": len(points),
        "points": points,
        "window_requested": window,
        "data_through": points[-1]["metric_date"],
        "measure_basis": materialized_history_basis(kpi, rows=rows),
        "data_source": "synthetic" if synthetic else "database",
    }
    return [payload]
