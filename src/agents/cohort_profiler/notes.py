"""Standing notes on a per-HCP cohort result (canonical TRx lane, codex r8-r10).

One rule for every consumer of cohort evidence. The orchestrator synthesizer reads
AgentResult-wrapped results (``result["cohort_profile"]``); the explainer reads the
dispatcher's flattened ``analysis_results`` entries (``entry["cohort_profile"]``,
dispatcher.py ``_successful_results``). A result is cohort evidence when its profile
carries a ``basis_note``.
"""

from typing import Any, Dict, Iterable, List, Optional


def _cohort_profile(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    nested = item.get("result")
    for candidate in (
        item.get("cohort_profile"),
        nested.get("cohort_profile") if isinstance(nested, dict) else None,
    ):
        if isinstance(candidate, dict) and candidate.get("basis_note"):
            return candidate
    return None


def standing_notes(items: Iterable[Any]) -> List[str]:
    """Each cohort result's notes, distinct, in order: the canonical-request sentence (when
    the ask mentioned the canonical measure), then the grain disclosure."""
    notes: List[str] = []
    for item in items:
        profile = _cohort_profile(item)
        if profile is None:
            continue
        for key in ("canonical_request_note", "basis_note"):
            note = profile.get(key)
            if isinstance(note, str) and note and note not in notes:
                notes.append(note)
    return notes


def has_cohort_evidence(items: Iterable[Any]) -> bool:
    """True when any item is a per-HCP cohort result carrying a ``basis_note``."""
    return bool(standing_notes(items))


# Codex r11: deterministic explanation findings for a cohort result.


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.1f}"


def _window_text(window: Any) -> str:
    if not isinstance(window, dict):
        return "no time window bound"
    label = window.get("label") or "window"
    start, end = window.get("start"), window.get("end_exclusive")
    return f"{label}, {start} to {end} (end exclusive)" if start and end else str(label)


def _counts(values: Any) -> str:
    if not isinstance(values, dict) or not values:
        return "none"
    ordered = sorted(values.items(), key=lambda kv: str(kv[0]))
    return ", ".join(f"{key} {_fmt(n)}" for key, n in ordered)


def _hcp_findings(profile: Dict[str, Any]) -> List[str]:
    scope = profile.get("brand") or "all brands"
    if profile.get("region_applied") and profile.get("region"):
        scope = f"{scope}, {profile['region']} region"
    headline = (
        f"TRx Panel cohort for {scope} ({_window_text(profile.get('window'))}): "
        f"{_fmt(profile.get('cohort_size'))} HCPs, "
        f"{_fmt(profile.get('trx_total'))} events combined"
    )
    if profile.get("trx_max") is not None:
        headline += f", top prescriber {_fmt(profile.get('trx_max'))} events"
    if profile.get("volume_kpi_id"):
        headline += f" (patient-panel KPI {profile['volume_kpi_id']})"
    findings = [headline]
    threshold = profile.get("threshold")
    if isinstance(threshold, dict) and threshold.get("min_exclusive"):
        findings.append(
            f"TRx Panel threshold: more than {_fmt(threshold['min_exclusive'])} events per HCP"
        )
    tiers = profile.get("volume_tiers")
    if isinstance(tiers, dict):
        for key in ("high", "medium", "low"):
            bucket = tiers.get(key)
            if not isinstance(bucket, dict):
                continue
            if bucket.get("trx_min") is not None:
                low, high = _fmt(bucket.get("trx_min")), _fmt(bucket.get("trx_max"))
                span = f"{low}-{high} events per HCP"
            else:
                span = "no HCPs in range"
            findings.append(
                f"TRx Panel {key} volume tier: {_fmt(bucket.get('n_hcps'))} HCPs, {span}, "
                f"{_fmt(bucket.get('trx_total'))} events combined"
            )
    bounds = profile.get("tier_boundaries")
    if isinstance(bounds, dict):
        cut = (
            "TRx Panel tier cut points (events per HCP): "
            f"low up to {_fmt(bounds.get('low_max_trx'))}, "
            f"medium up to {_fmt(bounds.get('medium_max_trx'))}"
        )
        if bounds.get("collapsed_cuts"):
            cut += "; the cuts coincide, so fewer than three distinct tiers exist"
        findings.append(cut)
    if profile.get("specialty"):
        findings.append(f"HCPs by specialty: {_counts(profile.get('specialty'))}")
    if profile.get("priority_tier"):
        findings.append(f"HCPs by priority tier: {_counts(profile.get('priority_tier'))}")
    return findings


def _patient_findings(profile: Dict[str, Any]) -> List[str]:
    window = _window_text(profile.get("window"))
    findings: List[str] = []
    for brand_profile in profile.get("brands") or []:
        if not isinstance(brand_profile, dict):
            continue
        brand = brand_profile.get("brand") or "all brands"
        findings.append(
            f"NRx Panel patient cohort for {brand} ({window}): "
            f"{_fmt(brand_profile.get('headline_nrx'))} new-Rx patients"
        )
        findings.append(
            f"NRx Panel new-Rx patients for {brand} by severity: "
            f"{_counts(brand_profile.get('severity'))}"
        )
        findings.append(
            f"NRx Panel new-Rx patients for {brand} by line of therapy: "
            f"{_counts(brand_profile.get('line'))}"
        )
    applied = profile.get("criteria_applied")
    if applied:
        findings.append("Criteria applied: " + "; ".join(str(c) for c in applied))
    return findings


def cohort_key_findings(result: Dict[str, Any]) -> List[str]:
    """Explanation findings built ONLY from a cohort result's own ``cohort_profile`` fields.

    The explainer's deterministic reasoner reads nothing but ``key_findings``
    (deep_reasoner.py:117) and a cohort result carries none, so without this a forced
    deterministic explanation is a "0 key finding(s)" husk. Like #1475's KPI evidence
    (dispatcher ``_kpi_lookup_evidence``), the real figures ride in key_findings: HCP
    cohorts as TRx Panel events, patient cohorts as NRx Panel new-Rx patients. A pure,
    deterministic function of the result (codex r11).
    """
    profile = result.get("cohort_profile") if isinstance(result, dict) else None
    if not isinstance(profile, dict):
        return []
    if profile.get("entity") == "hcp":
        findings = _hcp_findings(profile)
    else:
        findings = _patient_findings(profile)
    not_applied = [
        str(c.get("label"))
        for c in profile.get("criteria_not_applied") or []
        if isinstance(c, dict) and c.get("label")
    ]
    if not_applied:
        findings.append("Criteria not applied: " + "; ".join(not_applied))
    return findings
