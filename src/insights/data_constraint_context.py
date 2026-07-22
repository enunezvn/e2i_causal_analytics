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

* **Lag SSOT precedence (reconciled 2026-07-21, plan C0; reworded for the
  backlog #45 arrival plane)** — the per-brand ``claims_lag_band`` is the
  ONLY LM-facing claims-ADJUDICATION lag figure, stated exactly once, with
  the under-count claim explicitly SCOPED to real-world claims. The claims
  ARRIVAL plane IS simulated in this substrate (treatment_events carries
  claim_available_date/adjudication_lag_days, drawn from the band —
  data_constraints.adjudication_lag_dgp), but the DISPLAYED figures are the
  MATURE values: no base KPI filters on the arrival columns, so they do not
  under-count (the narrative must never invite a discount of displayed
  synthetic figures; the provisional/nowcast overlay is a separate view).
  The DISTINCT 7-14 day per-source ingest/feed lag class is named once as an
  aggregate band only; the vocabulary's per-source scalar lags and vendor
  names (edge_case_taxonomy.data_source_lag) stay server-side. Two
  contradictory lag figures for the SAME class in one prompt invite LM
  confusion.
* **Mitigation playbook (2026-07-22, frontend-review item 2b — the ONE
  deliberate exception to the no-vendor rule, product-owner approved)** — the
  authored ``data_constraints.mitigation_playbook`` renders after the lag
  bullet: proxy source classes that deliver faster SIGNAL than adjudicated
  claims (faster CLOSED claims are not purchasable), each with a class-level
  latency band, coverage caveat, and ILLUSTRATIVE vendors, plus the
  vendor-validation criteria the LM must apply before naming any vendor.
  The playbook must not restate the claims-lag band (single-lag-figure
  contract above). ``KNOWN_DATA_VENDORS`` below is the guard lexicon:
  src.insights.home_kpi rejects (fail-closed) any lexicon vendor the LM
  mentions that the rendered context does not itself name.
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

# Guard lexicon for the home-KPI vendor-allowlist check: the playbook's
# illustrative vendors PLUS well-known claims/Rx/EHR data vendors the LM might
# import from memory. Single-word stems ("Komodo", "Prognos") deliberately
# match their full names too. Case-insensitive, word-boundary matching in
# home_kpi — so no entry may be an ordinary English word (e.g. "ICON" stays
# out; "Symphony Health" is the entry, not "Symphony").
KNOWN_DATA_VENDORS: tuple[str, ...] = (
    # mitigation_playbook illustrative vendors (allowlisted via the rendered context)
    "AssistRx",
    "CareMetx",
    "CoverMyMeds",
    "IQVIA",
    "Symphony Health",
    "Komodo",
    "HealthVerity",
    "PurpleLab",
    "Prognos",
    "Truveta",
    "Datavant",
    # NOT in the playbook — a mention of any of these is an out-of-allowlist
    # vendor pairing and is rejected fail-closed by the guard
    "Optum",
    "Merative",
    "MarketScan",
    "Clarify Health",
    "Veradigm",
    "Inovalon",
    "Definitive Healthcare",
    "Milliman",
    "ConnectiveRx",
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


def _squash(text: Any) -> str:
    """Collapse folded-scalar newlines/indentation to single spaces."""
    return " ".join(str(text or "").split())


def _playbook_lines(data: dict[str, Any]) -> list[str]:
    """Render the authored mitigation playbook for the LM context. Raises on a
    missing/empty playbook — an authoring regression must degrade LOUDLY (the
    route's "Constraint context: unavailable" chip), never silently revert the
    narrative to an unactionable lag statement."""
    pb = data.get("mitigation_playbook") or {}
    classes = pb.get("source_classes") or []
    criteria = pb.get("vendor_validation_criteria") or []
    if not classes or not criteria:
        raise ValueError("data_constraints.mitigation_playbook missing or incomplete")
    lines = [
        "Claims-lag mitigation playbook (authored — cite it, never invent sources or vendors):",
        f"{_squash(pb.get('preamble'))}",
    ]
    for sc in classes:
        entry = f"- {_squash(sc.get('name'))} ({_squash(sc.get('latency'))}): {_squash(sc.get('coverage'))}"
        vendors = sc.get("illustrative_vendors") or []
        if vendors:
            entry += f". Illustrative vendors: {', '.join(vendors)}"
        if sc.get("status"):
            entry += f" — {_squash(sc.get('status'))}"
        lines.append(f"{entry}.")
    numbered = "; ".join(f"({i}) {_squash(c)}" for i, c in enumerate(criteria, 1))
    lines.append(
        f"Vendor validation criteria — before naming ANY vendor, verify every one of: {numbered}. "
        "If any check fails, name the source class only."
    )
    lines.append(_squash(pb.get("vendor_note")))
    return [line for line in lines if line]


def build_mitigation_playbook() -> dict[str, Any] | None:
    """The authored playbook as a structured payload for the UI (rendered
    verbatim in the structural-constraints block — deterministic, never
    LM-generated). Returns None on any failure: the playbook block simply
    doesn't render, while the narrative (whose own loud-degradation contract
    is build_constraint_context's) is unaffected."""
    try:
        pb = _constraints().get("mitigation_playbook") or {}
        classes = [
            {
                "name": _squash(sc.get("name")),
                "latency": _squash(sc.get("latency")),
                "coverage": _squash(sc.get("coverage")),
                "illustrative_vendors": list(sc.get("illustrative_vendors") or []),
                "status": _squash(sc["status"]) if sc.get("status") else None,
            }
            for sc in (pb.get("source_classes") or [])
        ]
        if not classes:
            raise ValueError("data_constraints.mitigation_playbook missing or incomplete")
        return {
            "preamble": _squash(pb.get("preamble")),
            "vendor_note": _squash(pb.get("vendor_note")),
            "source_classes": classes,
        }
    except Exception as e:  # noqa: BLE001 — additive block, never a 500
        logger.warning("mitigation playbook unavailable: %s", e)
        return None


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
            "substrate the claims arrival plane is simulated (each claims-derived "
            "event carries an adjudication lag drawn from the band above), but "
            "the figures shown here are the mature values — computed over all "
            "events regardless of arrival — so they do not under-count; a "
            "separate provisional/nowcast view models the as-of-today "
            "under-count for recent windows. This lag and vendor claims coverage "
            "are DATA-STRATEGY constraints — the reader cannot fix them; "
            "attribute, do not recommend.",
            *_playbook_lines(data),
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
