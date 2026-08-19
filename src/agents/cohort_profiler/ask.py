"""Parse a free-text cohort ask into bound parameters (#1356, parts 1 + 2).

The 2026-07-29 benchmark (q11/q15, both surfaces) confirmed the profiler was
query-INSENSITIVE: the brand and every inclusion criterion in the ask were
ignored, so two completely different asks collapsed to the same parameterless
KPI-call set and the (context-keyed) KPI cache legitimately served
byte-identical payloads. This module is the fix's front half: it extracts the
ask's binding parameters — entity type (patient vs HCP), brand (named directly
or via indication), inclusion criteria, KPI threshold, and time window — so the
agent can bind them into the profile query and account honestly for anything
the data model cannot serve.

Design constraints (REASON-BEFORE-RULES):
* Extraction is deliberately conservative: a criterion is only "recognized"
  when a specific pattern matches, and every recognized criterion is tagged
  ``servable`` or not. The agent then names EXACTLY which criteria were applied
  and which could not be — it never silently drops one it recognized.
* Schema facts grounding servability (verified READ-ONLY 2026-07-30):
  - ``patient_journeys.age_at_diagnosis`` is populated for all 25,499 rows
    (range 18-84) → age bounds ARE servable.
  - ``treatment_events`` has ZERO ``diagnosis`` events and there is no
    diagnosis-date column anywhere (``journey_start_date`` is only a documented
    proxy — migration 044's kisqali_dx_adoption note) → a diagnosis-year filter
    is NOT servable and must fail closed honestly.
  - Per-HCP TRx = COUNT of ``treatment_events`` prescription rows per
    ``hcp_id`` — the same substrate as the platform's TRx KPI
    (``business_impact_trx``), so HCP cohort numbers stay in lock-step with
    the KPI dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple

from src.services import query_entities

# Canonical brand casing (the KPI brand predicate is case-SENSITIVE) and the
# indication→brand grounding now live in the SHARED extraction service
# (src/services/query_entities.py, #1351): this module's proven #1356 semantics
# were lifted verbatim so every dispatcher resolver grounds the same way.
_SUPPORTED_BRANDS: Tuple[str, ...] = query_entities.SUPPORTED_BRANDS

_PATIENT_RE = re.compile(r"\bpatients?\b", re.I)
_HCP_RE = re.compile(
    r"\b(?:hcps?|prescribers?|physicians?|doctors?|health\s*care\s+professionals?)\b",
    re.I,
)

# KPI threshold: requires an explicit metric unit so it can never cross-match
# an age phrase ("adults over 18" has no unit; "more than 50 TRx" has one).
_THRESHOLD_RE = re.compile(
    r"(?:(?P<incl>at\s+least|no\s+fewer\s+than|>=|≥)|(?P<excl>more\s+than|over|above|exceeding|>))"
    r"\s*(?P<n>\d+)\s*(?P<metric>trx|nrx|rx|prescriptions?|scripts?)\b",
    re.I,
)

# Age criteria: anchored to an age word so they can never cross-match a KPI
# threshold. Each entry: (compiled regex, exclusive-bound adjustment).
_AGE_MIN_RES: Tuple[Tuple[re.Pattern[str], int], ...] = (
    # "adults over 18", "patients older than 64" → age > N (exclusive as said)
    (
        re.compile(
            r"\b(?:adults?|patients?|aged|age)\s+(?:over|above|older\s+than)\s+(\d{1,3})\b", re.I
        ),
        0,
    ),
    # "over 65 years old" → age > N
    (
        re.compile(
            r"\b(?:over|above|older\s+than)\s+(\d{1,3})\s*(?:years?(?:\s+old)?|y\.?o\.?)\b", re.I
        ),
        0,
    ),
    # "aged 18+", "aged 18 or older" → age >= N ⇒ exclusive bound N-1
    (
        re.compile(
            r"\baged?\s+(\d{1,3})\s*(?:\+|or\s+(?:older|over|above)|and\s+(?:older|over|above|up))\b",
            re.I,
        ),
        -1,
    ),
)
_AGE_MAX_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:adults?|patients?|aged|age)\s+(?:under|below|younger\s+than)\s+(\d{1,3})\b", re.I
    ),
    re.compile(
        r"\b(?:under|below|younger\s+than)\s+(\d{1,3})\s*(?:years?(?:\s+old)?|y\.?o\.?)\b", re.I
    ),
)

_DIAG_YEAR_RE = re.compile(r"\bdiagnos\w*\s+(?:in|during|since)\s+((?:19|20)\d{2})\b", re.I)

# Geographic region (#1693): hcp_profiles.geographic_region is enum region_type
# — northeast / midwest / south / west. Recognition is conservative per this
# module's design: bare "south"/"west" only count with a locating preposition
# ("in the south") or a region noun ("western territories"); the unambiguous
# compass compounds northeast/midwest also match as "the Northeast".
_REGION_NAMES = r"(?:north\s?east|mid\s?west|south|west)"
_REGION_RE = re.compile(
    rf"\b(?:in|from|across|within)\s+the\s+(?P<prep>{_REGION_NAMES})(?:ern)?"
    r"(?:\s+(?:region|territor(?:y|ies)|market|states|u\.?s\.?))?\b"
    rf"|\b(?P<noun>{_REGION_NAMES})(?:ern)?\s+(?:region|territor(?:y|ies)|market|states)\b"
    r"|\bthe\s+(?P<comp>north\s?east|mid\s?west)\b",
    re.I,
)

# Volume-tier segmentation (#1736): "Segment HCPs by prescription volume into
# high, medium, and low tiers" (eval 4.3, verbatim) / "high/medium/low
# prescription-volume tiers" (the 4.3 clarification's promised follow-up).
# Recognition is conservative per this module's design: it requires an explicit
# VOLUME word ("prescription volume", "prescribing volume", "TRx volume",
# "volume tiers") in tier context — a bare "tier" (e.g. "high priority tier")
# can never trigger it, because hcp_profiles.priority_tier is a DISTINCT
# targeting attribute (volume + brand affinity + accessibility per the
# ontology), not a prescription-volume axis. Measured 2026-08-19: the data
# model stores NO volume tier (prescribing_tier / prescribing_volume are NULL
# on all 5,000 hcp_profiles rows), so tiers are COMPUTED from windowed per-HCP
# TRx by the mig-130 statements.
_VOLUME_WORD = r"(?:prescri(?:ption|bing)|rx|trx)[\s-]*volumes?"
_VOLUME_TIER_RE = re.compile(
    rf"\b{_VOLUME_WORD}\b[^.?!\n]*?\btier(?:s|ed|ing)?\b"
    rf"|\btier(?:s|ed|ing)?\b[^.?!\n]*?\b{_VOLUME_WORD}\b"
    r"|\bvolume[\s-]*tier(?:s|ed|ing)?\b",
    re.I,
)

_DIAG_GUIDANCE = (
    "the data model carries no true diagnosis dates (treatment_events has zero "
    "'diagnosis' events; patient_journeys.journey_start_date is only a documented "
    "proxy), so a diagnosis-year filter cannot be served honestly. For audited "
    "inclusion/exclusion criteria, materialize the cohort via the ML pipeline "
    "(scope_definer → cohort_constructor)."
)

_LAST_QUARTER_RE = re.compile(r"\b(?:last|previous|past)\s+quarter\b", re.I)
_THIS_QUARTER_RE = re.compile(r"\bthis\s+quarter\b", re.I)
_LAST_MONTH_RE = re.compile(r"\b(?:last|previous|past)\s+month\b", re.I)
_LAST_N_DAYS_RE = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", re.I)


@dataclass(frozen=True)
class Criterion:
    """One recognized inclusion criterion (servable or honestly not)."""

    kind: str  # "age_min" | "age_max" | "diagnosis_year" | "region" | "volume_tiers"
    label: str  # human-readable echo of the ask, e.g. 'diagnosed in 2024'
    servable: bool
    value: Optional[int] = None  # bound value (exclusive for age bounds)
    guidance: Optional[str] = None  # why it cannot be served + what to do
    text_value: Optional[str] = None  # bound text value (region enum value, #1693)


@dataclass(frozen=True)
class Threshold:
    """A quantitative KPI threshold for an HCP-entity cohort."""

    metric: str  # normalized: "trx" (servable) | "nrx" (not yet)
    min_exclusive: int  # cohort = HCPs with metric strictly greater than this
    label: str
    servable: bool = True
    guidance: Optional[str] = None


@dataclass(frozen=True)
class Window:
    """An explicit half-open [start, end) time window."""

    label: str
    start: date
    end: date  # EXCLUSIVE
    explicit: bool = True  # named in the ask (vs a disclosed default)


@dataclass(frozen=True)
class CohortAsk:
    """The parsed, bindable parameters of one cohort ask."""

    entity_type: str  # "patient" | "hcp"
    brand: Optional[str]  # canonical casing, else None (all brands)
    criteria: Tuple[Criterion, ...] = field(default_factory=tuple)
    threshold: Optional[Threshold] = None
    window: Optional[Window] = None
    # "high/medium/low prescription-volume tiers" asked for (#1736) — served on
    # the HCP path via the mig-130 tercile statements, accounted honestly on
    # the patient path.
    volume_tiers: bool = False


# Delegations to the shared service (#1351). The names stay module-local so
# every existing call site and docstring reference in this proven template
# remains valid; behaviour is pinned identical by the shared service's tests.
_canonical_brand = query_entities.canonical_brand
_brand_from_text = query_entities.brand_from_text


def _entity_type(query: str) -> str:
    # "patient" wins when both appear ("patients treated by physicians" is a
    # patient cohort); default is patient (the pre-#1356 contract).
    if _PATIENT_RE.search(query):
        return "patient"
    if _HCP_RE.search(query):
        return "hcp"
    return "patient"


def _parse_threshold(query: str) -> Optional[Threshold]:
    m = _THRESHOLD_RE.search(query)
    if not m:
        return None
    n = int(m.group("n"))
    min_exclusive = n - 1 if m.group("incl") else n
    metric_raw = m.group("metric").lower()
    if metric_raw == "nrx":
        return Threshold(
            metric="nrx",
            min_exclusive=min_exclusive,
            label=m.group(0).strip(),
            servable=False,
            guidance=(
                "NRx thresholds are not yet servable via the allowlisted HCP "
                "cohort query — TRx thresholds are. Re-ask with a TRx threshold, "
                "or request the NRx variant."
            ),
        )
    return Threshold(metric="trx", min_exclusive=min_exclusive, label=m.group(0).strip())


def _parse_age_criteria(query: str) -> List[Criterion]:
    criteria: List[Criterion] = []
    for pattern, adjust in _AGE_MIN_RES:
        m = pattern.search(query)
        if m:
            criteria.append(
                Criterion(
                    kind="age_min",
                    label=m.group(0).strip(),
                    servable=True,
                    value=int(m.group(1)) + adjust,
                )
            )
            break
    for pattern in _AGE_MAX_RES:
        m = pattern.search(query)
        if m:
            criteria.append(
                Criterion(
                    kind="age_max",
                    label=m.group(0).strip(),
                    servable=True,
                    value=int(m.group(1)),
                )
            )
            break
    return criteria


def _parse_region(query: str) -> Optional[Criterion]:
    """Recognize a geographic-region criterion (#1693).

    Normalizes the matched compass name to the ``region_type`` enum value
    (northeast/midwest/south/west). Servability is entity-dependent and
    decided by the agent per path (HCP cohorts bind it via the mig-129
    ``_region`` statement; the patient paths disclose it as not applied) —
    here it is recognized as servable so it can never silently vanish.
    """
    m = _REGION_RE.search(query)
    if not m:
        return None
    raw = m.group("prep") or m.group("noun") or m.group("comp") or ""
    region = re.sub(r"\s+", "", raw).lower()
    return Criterion(kind="region", label=m.group(0).strip(), servable=True, text_value=region)


def _quarter_start(d: date) -> date:
    return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)


def _parse_window(query: str, today: date) -> Optional[Window]:
    if _LAST_QUARTER_RE.search(query):
        this_q = _quarter_start(today)
        start = (
            date(this_q.year - 1, 10, 1)
            if this_q.month == 1
            else date(this_q.year, this_q.month - 3, 1)
        )
        return Window(label="last quarter", start=start, end=this_q)
    if _THIS_QUARTER_RE.search(query):
        return Window(
            label="this quarter to date", start=_quarter_start(today), end=today + timedelta(days=1)
        )
    if _LAST_MONTH_RE.search(query):
        this_m = date(today.year, today.month, 1)
        start = (
            date(this_m.year - 1, 12, 1)
            if this_m.month == 1
            else date(this_m.year, this_m.month - 1, 1)
        )
        return Window(label="last month", start=start, end=this_m)
    m = _LAST_N_DAYS_RE.search(query)
    if m:
        n = int(m.group(1))
        # Inclusive-today semantics: exactly n dates in [today-(n-1), today+1).
        return Window(
            label=f"last {n} days",
            start=today - timedelta(days=n - 1),
            end=today + timedelta(days=1),
        )
    return None


def parse_cohort_ask(
    query: str,
    brand_hint: Optional[str] = None,
    today: Optional[date] = None,
) -> CohortAsk:
    """Parse ``query`` into a :class:`CohortAsk`.

    ``brand_hint`` is the resolver-grounded brand (``parsed_query.entities`` /
    ``user_context``) and takes precedence; the query-text scan is the fallback
    that fixes q11 (the forced/live surfaces carried no grounded entities, so
    "Remibrutinib" in the ask text itself was ignored). ``today`` is injectable
    for deterministic window math in tests.
    """
    query = query or ""
    today = today or date.today()

    criteria: List[Criterion] = list(_parse_age_criteria(query))
    region = _parse_region(query)
    if region is not None:
        criteria.append(region)
    m_diag = _DIAG_YEAR_RE.search(query)
    if m_diag:
        criteria.append(
            Criterion(
                kind="diagnosis_year",
                label=m_diag.group(0).strip(),
                servable=False,
                value=int(m_diag.group(1)),
                guidance=_DIAG_GUIDANCE,
            )
        )

    volume_tiers = bool(_VOLUME_TIER_RE.search(query))
    entity_type = _entity_type(query)
    if volume_tiers and not _PATIENT_RE.search(query):
        # Prescription-volume tiers bucket PRESCRIBERS by per-HCP TRx: with no
        # explicit patient word, a tier ask lands on the HCP path (the eval-4.3
        # follow-up often names only the brand plus the tier phrasing). An
        # explicit patient ask keeps the patient path and is accounted honestly
        # there.
        entity_type = "hcp"

    return CohortAsk(
        entity_type=entity_type,
        brand=_canonical_brand(brand_hint) or _brand_from_text(query),
        criteria=tuple(criteria),
        threshold=_parse_threshold(query),
        window=_parse_window(query, today),
        volume_tiers=volume_tiers,
    )


def merge_cohort_asks(primary: CohortAsk, supplement: CohortAsk) -> CohortAsk:
    """Union ``supplement``'s criteria/threshold/window into ``primary`` (#1698).

    ``primary`` is parsed from the query the chat model dispatched;
    ``supplement`` from the user's original ask. The measured 2.1 defect: the
    model's rewrite dropped "adults over 18" / "diagnosed in 2024", so the
    accounting never saw either criterion. Supplement criteria whose ``kind``
    the primary lacks are appended (primary's first); on kind collision the
    primary wins — the rewrite may have resolved anaphora the raw text leaves
    dangling. ``threshold``/``window`` fill in only when the primary has none;
    ``volume_tiers`` survives when either side asked for it (#1736);
    ``entity_type`` and ``brand`` stay the primary's.
    """
    have = {c.kind for c in primary.criteria}
    criteria = tuple(primary.criteria) + tuple(c for c in supplement.criteria if c.kind not in have)
    return CohortAsk(
        entity_type=primary.entity_type,
        brand=primary.brand,
        criteria=criteria,
        threshold=primary.threshold or supplement.threshold,
        window=primary.window or supplement.window,
        volume_tiers=primary.volume_tiers or supplement.volume_tiers,
    )
