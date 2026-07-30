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

# Canonical brand casing (the KPI brand predicate is case-SENSITIVE).
_SUPPORTED_BRANDS: Tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali")

# Indication → brand: the three commercial brands map 1:1 to indications on
# this substrate, so an indication mention grounds the brand (q11's "CSU").
_INDICATION_TO_BRAND: Tuple[Tuple[str, str], ...] = (
    (r"\bcsu\b|\bchronic\s+spontaneous\s+urticaria\b|\burticaria\b", "Remibrutinib"),
    (r"\bpnh\b|\bparoxysmal\s+nocturnal\b", "Fabhalta"),
    (r"\bbreast\s+cancer\b|\bhr\+\b|\bher2\b", "Kisqali"),
)

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

    kind: str  # "age_min" | "age_max" | "diagnosis_year"
    label: str  # human-readable echo of the ask, e.g. 'diagnosed in 2024'
    servable: bool
    value: Optional[int] = None  # bound value (exclusive for age bounds)
    guidance: Optional[str] = None  # why it cannot be served + what to do


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


def _canonical_brand(raw: Optional[str]) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    low = raw.strip().lower()
    for b in _SUPPORTED_BRANDS:
        if b.lower() == low:
            return b
    return None


def _brand_from_text(query: str) -> Optional[str]:
    """Ground the brand from the query text (name first, then indication).

    Returns a brand only when the text pins down EXACTLY ONE — two different
    brands named means the ask is ambiguous and the profiler keeps the honest
    all-brands scope rather than guess.
    """
    found: List[str] = []
    for b in _SUPPORTED_BRANDS:
        if re.search(rf"\b{b}\b", query, re.I) and b not in found:
            found.append(b)
    if not found:
        for pattern, b in _INDICATION_TO_BRAND:
            if re.search(pattern, query, re.I) and b not in found:
                found.append(b)
    return found[0] if len(found) == 1 else None


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
        return Window(
            label=f"last {n} days", start=today - timedelta(days=n), end=today + timedelta(days=1)
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

    return CohortAsk(
        entity_type=_entity_type(query),
        brand=_canonical_brand(brand_hint) or _brand_from_text(query),
        criteria=tuple(criteria),
        threshold=_parse_threshold(query),
        window=_parse_window(query, today),
    )
