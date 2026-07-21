"""Data-constraint context for the home-KPI strategic insight (constraint-aware
insights plan, 2026-07-20).

Renders ONE deterministic, server-derived block per request: the brand's
measurement-constraint profile (disease, prevalence class, claims-lag band,
CRM source — authored in ``config/domain_vocabulary.yaml`` under
``data_constraints``) plus per-KPI classification lines (actionability /
data_plane / measurement_caveat from the KPI registry) for the KPIs actually
present in the grounding. The LM CITES this classification instead of
re-deciding actionability per generation.

Contracts (locked by tests/insights/test_data_constraint_context.py):

* **Lag SSOT precedence (reconciled 2026-07-21, plan C0)** — the per-brand
  ``claims_lag_band`` is the ONLY LM-facing claims-ADJUDICATION lag figure,
  stated exactly once, with the under-count claim explicitly SCOPED to
  real-world claims: this synthetic substrate does not simulate adjudication
  lag, so recent windows do not under-count for that reason (the narrative
  must never invite a discount of displayed synthetic figures). The DISTINCT
  7-14 day per-source ingest/feed lag class is named once as an aggregate
  band only; the vocabulary's per-source scalar lags and vendor names
  (edge_case_taxonomy.data_source_lag) stay server-side. Two contradictory
  lag figures for the SAME class in one prompt invite LM confusion.
* **Prevalence direction guard** — verbatim: prevalence explains small samples
  and volatility, NOT low engagement/testing/coverage rates.
* **Loud degradation** — ANY failure returns ``""``: availability is preserved
  (the narrative still generates), and the CALLER is contract-bound to add the
  "Constraint context: unavailable" grounding chip and cache the degraded
  generation at the short TTL so a builder hiccup cannot pin the reverted
  (constraint-blind) narrative for an hour.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Verbatim in every profile: the direction-of-use guard that keeps prevalence
# an EXPLANATION for small samples, never an excuse for low engagement/testing
# rates (the Fabhalta diagnostic-activation recommendation must survive).
_PREVALENCE_GUARD = (
    "Disease prevalence explains small sample sizes and volatile rates, "
    "NOT low engagement/testing/coverage rates — those remain actionable."
)


def _constraints() -> dict[str, Any]:
    from src.ontology.vocabulary_registry import VocabularyRegistry

    return VocabularyRegistry.load().get_data_constraints()


def brands_with_profiles() -> tuple[str, ...]:
    """Brands carrying an authored constraint profile. The completeness test
    asserts every gold-standard brand is present — a 4th brand launch fails
    loudly there instead of silently rendering constraint-blind narratives."""
    try:
        return tuple((_constraints().get("brands") or {}).keys())
    except Exception:  # noqa: BLE001 — introspection helper, never raises
        return ()


def _kpi_lines(metas: list[Any]) -> list[str]:
    """Per-KPI classification lines for the KPIs PRESENT in the grounding.

    Plane wording: claims-plane rows point at the single lag sentence above
    (no second lag figure); crm/platform rows are current as shown."""
    lines: list[str] = []
    for m in metas:
        plane = m.data_plane
        if plane == "claims":
            plane_txt = "claims plane — subject to the claims lag above"
        elif plane in ("crm", "platform"):
            plane_txt = f"{plane} plane — no source lag, current as shown"
        else:
            plane_txt = "mixed plane — outcome side subject to the claims lag above"
        parts = [f"- {m.name}: {plane_txt}"]
        if m.actionability:
            owner = f", owner: {m.actionability_owner}" if m.actionability_owner else ""
            parts.append(f"; actionability: {m.actionability}{owner}")
        if m.levers:
            parts.append(f"; levers: {', '.join(m.levers)}")
        if m.measurement_caveat:
            caveat = " ".join(str(m.measurement_caveat).split())
            parts.append(f". Caveat: {caveat}")
        lines.append("".join(parts))
    return lines


def build_constraint_context(brand: str, metas: list[Any]) -> str:
    """The rendered constraint block for ``brand`` (or ``"All"`` portfolio
    scope), covering exactly the KPIs in ``metas``. Returns ``""`` on any
    failure (loud degradation contract — see module docstring)."""
    try:
        data = _constraints()
        profiles = data.get("brands") or {}
        lag_band = data.get("claims_lag_band")
        crm = data.get("crm_source", "CRM")
        if not profiles or not lag_band:
            raise ValueError("data_constraints section missing or incomplete")

        header: list[str]
        if brand == "All":
            brand_lines = "; ".join(
                f"{b} — {p.get('disease')} ({' '.join(str(p.get('prevalence_class', '')).split())})"
                for b, p in profiles.items()
            )
            header = [
                "Measurement constraints (portfolio scope):",
                f"- Brand profiles: {brand_lines}.",
            ]
        else:
            profile = profiles.get(brand)
            if not profile:
                raise KeyError(f"no data-constraint profile authored for brand {brand!r}")
            prevalence = " ".join(str(profile.get("prevalence_class", "")).split())
            header = [
                f"Measurement constraints for {brand} ({profile.get('disease')}; {prevalence}):",
            ]
        body = [
            f"- In real-world claims, a {lag_band} adjudication/runout lag means "
            "the most recent windows under-count true outcomes (trend is more "
            "reliable than level there); this band is distinct from the shorter "
            "7-14 day per-source ingest/feed lags, which are handled server-side "
            "and are not a second lag on these figures. In this synthetic "
            "substrate, adjudication lag is not simulated, so recent windows do "
            "not under-count for that reason. This lag and vendor claims coverage "
            "are DATA-STRATEGY constraints — the reader cannot fix them; "
            "attribute, do not recommend.",
            f"- CRM-derived figures ({crm}) have no source lag and are current as shown.",
            f"- {_PREVALENCE_GUARD}",
        ]
        kpi_lines = _kpi_lines(metas)
        if kpi_lines:
            body.append("KPI classification for rows on this dashboard:")
            body.extend(kpi_lines)
        return "\n".join(header + body)
    except Exception as e:  # noqa: BLE001 — loud degradation, never a 500
        logger.warning("data-constraint context unavailable for brand %r: %s", brand, e)
        return ""
