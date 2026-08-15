"""KPI-aware data resolution for the Tool Composer (issue #810).

Background
----------
The Tool Composer answers multi-faceted analytical queries. Its cohort loader
(:func:`src.services.cohort_resolution.resolve_cohort_frame`) always resolves the
patient-clinical ``patient_journeys`` grain. But many flagship queries are about a
**defined KPI** — e.g. *"what drove <brand> conversion in <region>?"* — whose
outcome does NOT live in ``patient_journeys``.

"Conversion" is the defined ``Conversion Rate`` KPI (``WS3-BI-009``):
*percentage of triggers resulting in a prescription within 30 days*
(``triggers ⋈ treatment_events``; see ``config/kpi_definitions.yaml`` and the
allowlist SQL ``business_impact_conversion_rate`` in migration 044). Resolving the
patient grain can never bind that outcome, so the causal core fails.

This service makes the pipeline KPI-aware:

1. :func:`recognize_kpi` — map a query to a defined KPI via the KPI registry
   (``src/kpi/registry.py``) + a small KPI-vocabulary alias map.
2. :func:`resolve_kpi_frame` — materialize the **analyzable** frame for that KPI
   from its REAL substrate, returning the outcome column + candidate driver
   columns so the planner can bind the causal outcome to the KPI.

Dynamic, not hardcoded
----------------------
Brand and region are **parameters**, matched case-insensitively against the
ACTUAL distinct values present in the data (``treatment_events.brand`` /
``hcp_profiles.geographic_region``) — there is no hardcoded brand or region list,
and nothing is special-cased to a particular brand/region. An input value not
present in the data fails closed (``None``), never a wrong-population or
fabricated frame.

Anti-mocking discipline
-----------------------
Never fabricates a frame. Returns ``None`` (fail closed) on unrecognized
brand/region, missing substrate, or empty results — callers then proceed without
``estimation_data`` and the composable tools fail closed in turn.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.kpi.models import KPIMetadata
from src.kpi.registry import get_registry

logger = logging.getLogger(__name__)

# Conversion Rate KPI id (config/kpi_definitions.yaml -> WS3-BI-009).
CONVERSION_KPI_ID = "WS3-BI-009"

# Conversion window: a delivered trigger is "converted" if a prescription occurs
# within this many days after it (authoritative SQL business_impact_conversion_rate).
_CONVERSION_WINDOW_DAYS = 30

# Candidate causal driver columns on the trigger grain (those that exist are
# exposed; the planner picks treatment/segments from them). These are trigger
# FEATURES, not the outcome.
_DRIVER_COLUMNS = [
    "trigger_type",
    "delivery_channel",
    "priority",
    "confidence_score",
    "lead_time_days",
    "acceptance_status",
]

# Columns fetched from the triggers table for the conversion substrate.
_TRIGGER_SELECT = [
    "trigger_id",
    "patient_id",
    "hcp_id",
    "trigger_timestamp",
    *_DRIVER_COLUMNS,
]

# Generous per-request row cap; the substrate tables are small (~4k triggers,
# ~10k events). A WARNING fires if a fetch hits this (possible truncation).
_MAX_ROWS = 100_000

# KPI vocabulary aliases: common user terms -> KPI id. This maps the FIXED,
# defined KPI vocabulary to ids; it is NOT brand/region hardcoding.
_ALIASES: Dict[str, str] = {
    "conversion": CONVERSION_KPI_ID,
    "conversion rate": CONVERSION_KPI_ID,
    "nbrx": "WS3-BI-007",
    "new-to-brand": "WS3-BI-007",
    "new to brand": "WS3-BI-007",
    "nrx": "WS3-BI-006",
    "new prescription": "WS3-BI-006",
    "trx share": "WS3-BI-008",
    "market share": "WS3-BI-008",
    # Reverse share phrasing (#1475 codex iter-2): "the share of TRx" is
    # natural WS3-BI-008 language — without these it falls to the bare "trx"
    # alias and reads as a WS3-BI-005 mention inside a "share of" chain.
    "share of trx": "WS3-BI-008",
    "share of total prescriptions": "WS3-BI-008",
    "trx": "WS3-BI-005",
    "total prescription": "WS3-BI-005",
    "return on investment": "WS3-BI-010",
    "roi": "WS3-BI-010",
    "hcp coverage": "WS3-BI-004",
    "patient touch": "WS3-BI-003",
    # Trigger-effectiveness KPIs (#1360 ruling: chat-KPI-path). Explicit aliases
    # because the name-token fallback mis-resolved compound asks: any query
    # containing "trigger" matched Trigger Precision first ("trigger acceptance
    # rate" -> WS2-TR-001), and funnel phrasings fell through to the shorter
    # "conversion" alias. Longest-alias-wins keeps "funnel conversion" ahead of
    # "conversion" and "acceptance rate" ahead of the "trigger" token fallback.
    "trigger precision": "WS2-TR-001",
    "acceptance rate": "WS2-TR-004",
    "override rate": "WS2-TR-006",
    "trigger funnel conversion": "WS2-TR-009",
    "funnel conversion": "WS2-TR-009",
    "conversion funnel": "WS2-TR-009",
    "trigger funnel": "WS2-TR-009",
    # Model-performance KPIs whose names are ENTIRELY sub-4-char abbreviations
    # ("ROC-AUC", "PR-AUC", "F1 Score" — "score" is a stop token). The name-token
    # fallback has nothing it can see, so before #1637 these resolved to None and
    # the chat tool answered "did not resolve to a defined KPI" for three fully
    # implemented KPIs. Aliased rather than lowering the token floor, which would
    # admit "ttr"/"cfr"/"ate" and collide with ordinary prose (_ABBREV_BLOCKLIST).
    # Multi-character forms only — a bare "f1" would match inside unrelated words.
    "roc-auc": "WS1-MP-001",
    "roc auc": "WS1-MP-001",
    "pr-auc": "WS1-MP-002",
    "pr auc": "WS1-MP-002",
    "f1 score": "WS1-MP-003",
}

# Registry abbreviations that are ordinary English words: admitting them to
# the strict metric vocabulary would turn everyday chat prose into phantom
# metric mentions ("access issues ATE into field time" is not a CM-001 ask).
_ABBREV_BLOCKLIST = frozenset({"ate"})

# Reverse-share phrasing that tolerates a brand/modifier gap between "of" and
# the metric ("share of Kisqali TRx" -> WS3-BI-008). Mirrors the of-chain
# tolerance in the dispatcher's governing-head guard; punctuation breaks the
# chain the same way.
_REVERSE_SHARE_RE = re.compile(
    r"\bshare\s+of\s+(?:(?:the|this|our)\s+)?(?:[\w'-]+\s+){0,2}?(?:trx|total\s+prescriptions?)\b"
)


@dataclass
class KpiFrame:
    """A materialized, analyzable frame for a KPI causal question.

    Attributes:
        frame: the real per-unit DataFrame (e.g. per-trigger) carrying the
            outcome column + driver columns.
        outcome_column: the causal outcome column name (e.g. ``"converted"``).
        driver_columns: candidate causal driver/segment columns present in the
            frame.
        treatment_column: the KPI's DEFINED causal treatment/intervention column
            when the KPI defines one (e.g. ``"accepted"`` for Conversion Rate,
            derived from real ``acceptance_status``). ``None`` when the KPI has no
            single defined treatment — consumers that require an explicit
            treatment (e.g. heterogeneous_optimizer) then fail closed rather than
            guess. Always a real column present in ``frame`` when set.
        treatment_source_column: the RAW column the derived ``treatment_column``
            was computed from (e.g. ``"acceptance_status"`` for ``"accepted"``).
            ``None`` when the treatment is itself a raw column. Consumers must
            EXCLUDE this from effect-modifier/driver sets — it is a deterministic
            function of the treatment, so using it as a modifier leaks the
            treatment into itself.
        kpi_id: the resolved KPI id (e.g. ``"WS3-BI-009"``).
        kpi_name: the human-readable KPI name (e.g. ``"Conversion Rate"``).
        is_truncated: ``True`` when a source fetch hit the ``_MAX_ROWS`` cap, so the
            substrate may be a truncated sample (no silent caps — surfaced to the
            caller for logging / response provenance).
    """

    frame: pd.DataFrame
    outcome_column: str
    driver_columns: List[str]
    kpi_id: str
    kpi_name: str
    treatment_column: Optional[str] = None
    treatment_source_column: Optional[str] = None
    is_truncated: bool = False


# ---------------------------------------------------------------------------
# KPI semantics (SSOT — moved here from chatbot_tools, #1475)
# ---------------------------------------------------------------------------
# Definition clarifications every answering surface MUST carry into the answer.
# This dict lived in src/api/routes/chatbot_tools.py (which re-exports it); it
# moved here because the orchestrator's explainer resolver needs the same notes
# and importing chatbot_tools costs ~30s (it pulls the orchestrator /
# tool_composer / RAG stacks) — unaffordable inside a sync input resolver.
#
# WS3-BI-008 (2026-07-18 session review): the share denominator is every
# prescription in treatment_events, and ONLY the tracked portfolio brands
# (Fabhalta / Kisqali / Remibrutinib) exist there — the chatbot presented the
# figure as "share of the CSU market" and attributed the complement to Xolair/
# Dupixent, which are not in the data model at all (a fabricated narrative on
# top of a real number). Attaching the honest basis to every response kills
# that at the source instead of relying on prompt memory.
KPI_SEMANTIC_NOTES = {
    "WS3-BI-008": (
        "TRx Share is the brand's share of the tracked portfolio's "
        "prescriptions (Fabhalta + Kisqali + Remibrutinib, cross-indication) "
        "— NOT market share against external competitors. Competitor brands "
        "(e.g. Xolair, Dupixent) are not in the data model; never attribute "
        "the share complement to them."
    ),
    # #1360: 'trigger precision' reads like ML-model telemetry — the exact
    # confusion that routed the bench-0024 ask to health_score. Pin the real
    # meaning to every answer instead of relying on prompt memory.
    "WS2-TR-001": (
        "Trigger Precision is a BUSINESS-program metric over the NBA triggers "
        "funnel — of accepted triggers with tracked outcomes, the share whose "
        "patient converted within the 30-day window (definition v2, "
        "truth-aligned). It is NOT a deployed-ML-model precision metric; "
        "model telemetry lives with health_score."
    ),
    "WS2-TR-009": (
        "Trigger Funnel Conversion's headline is the ACTIONED share of "
        "DELIVERED triggers (delivered -> accepted -> actioned); the full "
        "stage counts ride along as funnel_stages (delivered, viewed, "
        "accepted, actioned, outcome). The headline stops at actioned by "
        "design — the outcome stage reflects outcome-TRACKING coverage, not "
        "effectiveness — and 'viewed' is a delivery-status progression state, "
        "not a funnel prerequisite for acceptance."
    ),
    # #1532 (supersedes #1527): monthly data gives every (metric, brand,
    # region) slice exactly n=1 in the 30-day headline window, so no interval
    # on the headline is possible; the band is a DIFFERENT estimand and must
    # never be presented as inferential uncertainty.
    "WS3-BI-010": (
        "ROI's headline is a pooled point estimate with no interval — the "
        "monthly substrate cannot support one. When temporal_variability_band "
        "rides along, present each slice's band as the range of its monthly "
        "ROI values over the past 12 months (recent temporal variability, "
        "with its n) — it is NOT a confidence interval and NOT uncertainty "
        "about the current value; slices with band_suppressed=true report "
        "only n, never an invented range."
    ),
}


# ---------------------------------------------------------------------------
# KPI recognition (registry-driven, dynamic across all defined KPIs)
# ---------------------------------------------------------------------------
#: Name tokens too generic to identify a KPI on their own — they appear in many
#: registry names and in ordinary analytics prose.
_NAME_TOKEN_STOP = frozenset(
    {"rate", "score", "total", "new", "of", "the", "and", "to", "per", "median"}
)

#: Shortest name token allowed to drive a match. Below this, tokens are
#: abbreviations ("ttr", "cfr", "auc") that collide with ordinary words; the
#: KPIs whose names are ENTIRELY such tokens carry explicit aliases instead.
_MIN_NAME_TOKEN = 4

#: Characters that JOIN words in real input but separate concepts for matching:
#: "conversion_rate", "TRx-share". Normalized to spaces so boundary matching sees
#: the words (#1637 codex iter-8/9). Replacement is one character for one, so the
#: normalized string keeps the LENGTH its spans are measured against.
_SEPARATOR_CHARS = "_-"

#: How a metric phrase may be inflected in prose — "override rate" also appears
#: as "override rates", "patient touch" as "patient touches", "ROI" as "ROI's".
#:
#: SHARED DELIBERATELY (#1637 codex iter-2). Two matchers decide whether a phrase
#: names a metric — :func:`_alias_pattern` (which metric is being asked about)
#: and :func:`recognize_distinct_metric` (is a SECOND metric also named). When
#: only the first learned plurals, "acceptance rates and override rates" resolved
#: the first KPI, failed to see the second, and answered one metric as complete —
#: the exact fail-silent the multi-metric guard exists to prevent. One constant
#: so the two cannot drift apart again.
#: Possessives are included because the boundary rule otherwise rejects them
#: outright: "ROI's trend" and "TRx's drivers" resolved on main and stopped
#: resolving here — and "TRx drivers" is one of the very shapes the #1475
#: governing-head guards exist to read. The curly apostrophe resolved while the
#: ASCII one did not, which is the tell that this was an accident of the
#: character class rather than a decision.
_PLURAL_SUFFIX = r"(?:'s|’s|e?s)?"


@lru_cache(maxsize=1024)
def _name_tokens(name: str) -> Tuple[str, ...]:
    """Distinctive lowercase tokens of a KPI name, in order of appearance."""
    normalized = name.lower().replace("-", " ").replace("(", " ").replace(")", " ")
    seen: Dict[str, None] = {}
    for tok in normalized.split():
        tok = tok.strip()
        if len(tok) >= _MIN_NAME_TOKEN and tok not in _NAME_TOKEN_STOP:
            seen.setdefault(tok, None)
    return tuple(seen)


@lru_cache(maxsize=1024)
def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Alias matcher: bounded on both sides, tolerating a plural suffix.

    Aliases were matched by bare substring, which let a short alias fire from
    inside an unrelated word — "roc auc" matched "p|roc auc|tion" and "f1 score"
    matched "f1 score|card" (#1637 codex iter-1). Bounding both sides fixes that.

    The plural suffix is not decoration: a plain ``\\b...\\b`` rule silently LOST
    "override rates", "acceptance rates" and "patient touches", none of which the
    187-test regression corpus covered. :data:`_PLURAL_SUFFIX` keeps those while
    still rejecting "f1 score|card", since "card" matches neither the optional
    suffix nor the closing boundary.
    """
    return re.compile(rf"(?<![\w'-]){re.escape(alias)}{_PLURAL_SUFFIX}(?![\w'-])")


@lru_cache(maxsize=1024)
def _token_pattern(token: str) -> re.Pattern[str]:
    """Word-boundary matcher for a name token.

    Boundaries matter (#1637): plain substring search let WS1-DQ-004 "Stacking
    Lift" claim every query containing "up|lift|", so "action rate uplift"
    resolved to Stacking Lift instead of WS2-TR-003 Action Rate Uplift.
    """
    return re.compile(r"\b" + re.escape(token) + r"\b")


def _best_name_match(
    q: str, kpis: List[KPIMetadata]
) -> Optional[Tuple[KPIMetadata, str, int, int]]:
    """Resolve ``q`` to the KPI whose NAME the query covers best.

    Before #1637 this was "return the first registry KPI holding any name token
    found as a substring of the query". Both halves of that were defects:

    * **substring, not word boundary** — "lift" matched inside "uplift".
    * **registry order standing in for relevance** — a one-token brush beat an
      exact full-name match, so ``WS2-TR-001 Trigger Precision`` owned every
      query containing the word "trigger" (the reported 4.6 symptom: "what is
      the false alert rate for triggers" resolved to Trigger Precision), and
      ``WS1-DQ-009 Time-to-Release`` owned "lead time" via its "time" token.

    Measured over the 45-KPI registry, ten KPIs resolved to a DIFFERENT KPI when
    asked by their own name. Scoring by coverage takes that to zero.

    Ranking, best first: most distinct name tokens matched, then the most query
    characters matched, then registry order — which keeps the previous winner on
    a genuine tie, so this narrows resolution rather than reshuffling it.

    The returned span is the EARLIEST matched token, so it points at the start
    of the KPI mention for the #1475 governing-head guards.
    """
    best: Optional[Tuple[Tuple[int, int, int], KPIMetadata, int, int]] = None
    for order, kpi in enumerate(kpis):
        starts: List[int] = []
        ends: List[int] = []
        for tok in _name_tokens(str(kpi.name)):
            m = _token_pattern(tok).search(q)
            if m is not None:
                starts.append(m.start())
                ends.append(m.end())
        if not starts:
            continue
        # Negated so that "more is better" sorts first under plain tuple order.
        key = (-len(starts), -sum(e - s for s, e in zip(starts, ends, strict=True)), order)
        if best is None or key < best[0]:
            first = min(range(len(starts)), key=lambda i: starts[i])
            best = (key, kpi, starts[first], ends[first])
    if best is None:
        return None
    return best[1], q, best[2], best[3]


def recognize_kpi_span(query: Optional[str]) -> Optional[Tuple[KPIMetadata, str, int, int]]:
    """Like :func:`recognize_kpi`, but also expose WHERE the vocabulary hit.

    Returns ``(kpi, normalized_query, match_start, match_end)`` —
    ``normalized_query`` is the whitespace-collapsed lowercase form the matcher
    actually ran on, and ``[match_start, match_end)`` is the span of the matched
    alias / name token in it. Callers that must reason about the KPI mention's
    grammatical position (the #1475 governing-head guards: "cost of TRx" names
    TRx as a modifier; "TRx drivers" names TRx as the outcome of a causal ask)
    use this instead of re-deriving the match.
    """
    if not query:
        return None
    # Separators are separators, not word characters (#1637 codex iter-8/9). The
    # model really does pass snake_case ids -- "conversion_rate" (15 calls) and
    # "market_share" (6) appear in the 51-turn eval -- and once matching moved to
    # word boundaries, "_" being a \w char made both resolve to NOTHING. Hyphens
    # were worse than nothing: "conversion-rate" resolved to Trigger Funnel
    # Conversion instead of Conversion Rate.
    #
    # Normalized AFTER the whitespace collapse and one character for one, so the
    # result is the same LENGTH as the string the spans are measured against; the
    # #1475 governing-head guards slice this string and would silently misread a
    # shorter one.
    q = " ".join(str(query).lower().split())
    for _sep in _SEPARATOR_CHARS:
        q = q.replace(_sep, " ")
    registry = get_registry()

    # 0) reverse-share phrasing with a brand/modifier gap (#1475 codex iter-4):
    # "the share of Kisqali TRx" is WS3-BI-008 language, but the contiguous
    # "share of trx" alias cannot see through the brand token, so the bare
    # "trx" alias would read it as a WS3-BI-005 mention inside a "share of"
    # chain and die on the head guard.
    m = _REVERSE_SHARE_RE.search(q)
    if m is not None:
        share_kpi = registry.get("WS3-BI-008")
        if share_kpi is not None:
            return share_kpi, q, m.start(), m.end()

    # 1) alias match — longest alias first so "conversion rate" beats "rate".
    for alias in sorted(_ALIASES, key=len, reverse=True):
        m = _alias_pattern(alias).search(q)
        if m is not None:
            kpi = registry.get(_ALIASES[alias])
            if kpi is not None:
                return kpi, q, m.start(), m.end()

    # 2) dynamic fallback: score every KPI by how much of its NAME the query
    # covers, and take the best (#1637). See _best_name_match for why coverage
    # replaced "first registry KPI holding any matching token".
    return _best_name_match(q, registry.get_all())


@lru_cache(maxsize=1)
def _strict_metric_vocabulary() -> Tuple[Tuple[str, str], ...]:
    """(phrase, kpi_id) pairs of HIGH-PRECISION metric vocabulary: the alias
    map, full registry names (parentheticals stripped, punctuation
    normalized), and parenthetical abbreviations ("MAU"). Single name TOKENS
    are deliberately absent — registry names carry brand and scope tokens
    ("kisqali", "patients") that mark an ask's SCOPE, not a metric mention.
    Longest phrase first so "conversion rate" beats "conversion"."""
    vocab: Dict[str, str] = {}
    for alias, kpi_id in _ALIASES.items():
        vocab.setdefault(alias, kpi_id)
    for kpi in get_registry().get_all():
        raw_name = str(kpi.name)
        base = " ".join(
            re.sub(r"[^a-z0-9']+", " ", re.sub(r"\([^)]*\)", " ", raw_name.lower())).split()
        )
        if len(base) >= 4:
            vocab.setdefault(base, kpi.id)
        # Parenthetical abbreviations, harvested with CASE intact: only real
        # initialisms qualify (>=2 uppercase letters — keeps MAU/TTR/NRx,
        # structurally drops "(Median)"), and common English words are
        # blocklisted even when upper-cased — "(ATE)" must not make every
        # "ate" in chat prose a metric mention (codex iter-6).
        for abbr_raw in re.findall(r"\(([^)]+)\)", raw_name):
            abbr = " ".join(re.sub(r"[^a-z0-9']+", " ", abbr_raw.lower()).split())
            if (
                len(abbr) >= 3
                and sum(1 for c in abbr_raw if c.isupper()) >= 2
                and abbr not in _ABBREV_BLOCKLIST
            ):
                vocab.setdefault(abbr, kpi.id)
    return tuple(sorted(vocab.items(), key=lambda kv: len(kv[0]), reverse=True))


@lru_cache(maxsize=1)
def _case_sensitive_metric_abbrevs() -> Tuple[Tuple[str, str], ...]:
    """Blocklisted common-word initialisms in their ORIGINAL uppercase form —
    "ATE" in a query is the CM-001 metric even though prose "ate" is not
    (codex iter-7). Matched case-sensitively against the un-normalized query."""
    out: List[Tuple[str, str]] = []
    for kpi in get_registry().get_all():
        for abbr_raw in re.findall(r"\(([^)]+)\)", str(kpi.name)):
            token = abbr_raw.strip()
            lowered = " ".join(re.sub(r"[^a-z0-9']+", " ", token.lower()).split())
            if lowered in _ABBREV_BLOCKLIST and sum(1 for c in token if c.isupper()) >= 2:
                out.append((token, kpi.id))
    return tuple(out)


def recognize_distinct_metric(
    normalized_query: str, *, exclude_id: str, original_query: Optional[str] = None
) -> Optional[Tuple[KPIMetadata, int, int]]:
    """A metric mention OTHER than ``exclude_id`` in an (already lowercase,
    possibly span-masked) query — the dispatcher's multi-KPI vetoes probe
    with this after masking the first recognized span. Word-boundary matched
    against the strict vocabulary only; returns ``(kpi, start, end)`` in the
    given string's coordinates.

    ``original_query`` (case intact) additionally admits the blocklisted
    common-word initialisms in their uppercase form: "TRx and ATE" names two
    metrics; "access issues ate into field time" does not. The span is then
    located via the lowercase occurrence in ``normalized_query`` (a query
    containing BOTH forms resolves to the first occurrence — the veto errs
    fail-closed)."""
    registry = get_registry()
    for phrase, kpi_id in _strict_metric_vocabulary():
        if kpi_id == exclude_id:
            continue
        # Plural-tolerant, matching _alias_pattern (#1637 codex iter-2): a probe
        # that only saw singulars missed the second metric in "acceptance rates
        # and override rates" and let it be answered as one. The case-sensitive
        # abbreviation branch below stays exact — "TRx"/"ATE" are not pluralized,
        # and loosening initialisms is how they start matching ordinary prose.
        m = re.search(
            rf"(?<![\w'-]){re.escape(phrase)}{_PLURAL_SUFFIX}(?![\w'-])", normalized_query
        )
        if m is not None:
            kpi = registry.get(kpi_id)
            if kpi is not None:
                return kpi, m.start(), m.end()
    if original_query:
        for abbr_raw, kpi_id in _case_sensitive_metric_abbrevs():
            if kpi_id == exclude_id:
                continue
            if re.search(rf"(?<![\w'-]){re.escape(abbr_raw)}(?![\w'-])", original_query):
                kpi = registry.get(kpi_id)
                if kpi is not None:
                    lowered = abbr_raw.lower()
                    m = re.search(rf"(?<![\w'-]){re.escape(lowered)}(?![\w'-])", normalized_query)
                    start, end = (m.start(), m.end()) if m else (0, 0)
                    return kpi, start, end
    return None


def recognize_kpi(query: Optional[str]) -> Optional[KPIMetadata]:
    """Recognize a defined KPI referenced by ``query``, else ``None``.

    Matches the query against KPI-vocabulary aliases first (longest alias wins
    for specificity), then falls back to a conservative match on the registry's
    KPI names. Brand/region in the query are ignored here (they are resolved
    separately and dynamically). Delegates to :func:`recognize_kpi_span` — one
    matcher, two views.
    """
    match = recognize_kpi_span(query)
    return match[0] if match is not None else None


# ---------------------------------------------------------------------------
# Pure outcome construction (real logic; unit-tested without a DB)
# ---------------------------------------------------------------------------
def _compute_conversion_outcome(
    triggers: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int = _CONVERSION_WINDOW_DAYS,
) -> pd.Series:
    """Per-trigger binary ``converted``: a prescription for the trigger's patient
    within ``[trigger_date, trigger_date + window_days]`` (inclusive, date-level).

    Mirrors the authoritative ``business_impact_conversion_rate`` SQL. Pure: takes
    real-shaped frames, computes the real outcome — no DB, no fabrication.
    """
    if triggers is None or len(triggers) == 0:
        return pd.Series([], dtype=int)

    trig_ts = pd.to_datetime(triggers["trigger_timestamp"], errors="coerce", utc=True)

    by_patient: Dict[Any, List[Any]] = {}
    if events is not None and len(events) and "patient_id" in events.columns:
        ev_dt = pd.to_datetime(events["event_date"], errors="coerce", utc=True)
        for pid, d in zip(events["patient_id"], ev_dt, strict=False):
            if pd.notna(d):
                by_patient.setdefault(pid, []).append(d.date())

    out: List[int] = []
    for pid, ts in zip(triggers["patient_id"], trig_ts, strict=False):
        if pd.isna(ts):
            out.append(0)
            continue
        lo = ts.date()
        hi = (ts + pd.Timedelta(days=window_days)).date()
        dates = by_patient.get(pid, [])
        out.append(int(any(lo <= d <= hi for d in dates)))
    return pd.Series(out, index=triggers.index, dtype=int)


def _assemble_conversion_frame(
    triggers: pd.DataFrame,
    hcp_regions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    region_canonical: Optional[str],
    window_days: int = _CONVERSION_WINDOW_DAYS,
) -> Optional[KpiFrame]:
    """Build the conversion ``KpiFrame`` from already-fetched frames (pure).

    ``events`` must already be brand-filtered prescriptions (the brand filter is
    applied on the prescription side, since ``triggers`` carries no usable brand).
    Region is applied here via the ``triggers.hcp_id ⋈ hcp_regions`` join. Fails
    closed (``None``) on empty triggers or an unrecognized/empty region.
    """
    if triggers is None or len(triggers) == 0:
        return None

    df = triggers.copy()

    if region_canonical:
        if hcp_regions is None or len(hcp_regions) == 0:
            return None
        reg = hcp_regions.copy()
        reg_norm = reg["geographic_region"].astype(str).str.strip().str.lower()
        in_region = set(reg.loc[reg_norm == region_canonical.strip().lower(), "hcp_id"])
        if not in_region:
            return None
        df = df[df["hcp_id"].isin(in_region)].copy()
        if len(df) == 0:
            return None

    df["converted"] = _compute_conversion_outcome(df, events, window_days).to_numpy()

    drivers = [c for c in _DRIVER_COLUMNS if c in df.columns]
    # Derived clean binary treatment from acceptance_status (real, not fabricated).
    treatment_column: Optional[str] = None
    treatment_source_column: Optional[str] = None
    if "acceptance_status" in df.columns:
        df["accepted"] = (
            df["acceptance_status"].astype(str).str.strip().str.lower() == "accepted"
        ).astype(int)
        drivers.append("accepted")
        # The Conversion Rate KPI's DEFINED treatment is trigger acceptance — a
        # real, derived binary intervention. Exposing it lets the heterogeneous
        # optimizer bind a real treatment_var without guessing.
        treatment_column = "accepted"
        # ``acceptance_status`` is the raw source of the derived treatment — it is
        # a deterministic function of it, so consumers must NOT use it as a driver/
        # effect-modifier (that would leak the treatment into itself).
        treatment_source_column = "acceptance_status"

    return KpiFrame(
        frame=df.reset_index(drop=True),
        outcome_column="converted",
        driver_columns=drivers,
        treatment_column=treatment_column,
        treatment_source_column=treatment_source_column,
        kpi_id=CONVERSION_KPI_ID,
        kpi_name="Conversion Rate",
    )


# ---------------------------------------------------------------------------
# Dynamic brand/region resolution against the REAL data values
# ---------------------------------------------------------------------------
def _match_against_distinct(value: Optional[str], distinct: set[str]) -> Optional[str]:
    """Case-insensitively match ``value`` to a member of ``distinct`` (the actual
    data values). Returns the canonical (data) spelling, or ``None`` if absent."""
    if not value or not str(value).strip():
        return None
    norm = str(value).strip().lower()
    for d in distinct:
        if str(d).strip().lower() == norm:
            return str(d)
    return None


def _default_client() -> Any:
    from src.repositories import get_supabase_client

    return get_supabase_client()


def _resolve_brand_canonical(
    client: Any, brand: str, *, include_synthetic: bool = False
) -> tuple[Optional[str], bool]:
    """Resolve ``brand`` to its canonical data spelling, case-insensitively, against
    the real ``treatment_events.brand`` values.

    ``treatment_events.brand`` is a PostgreSQL ENUM (``brand_type``) so ``ILIKE`` is
    unavailable (``operator does not exist: brand_type ~~* unknown``); we scan the
    distinct values and match in Python. To avoid a SILENT truncation, the scan's
    cap is detected and returned: a no-match while the scan was capped is reported
    as ``(None, True)`` so the caller can distinguish "brand truly absent" from
    "brand may exist beyond the row cap" — it is never a silent fail-closed.

    Returns ``(canonical_or_None, scan_truncated)``.
    """
    value = str(brand).strip()
    if not value:
        return None, False
    _bq = (
        client.table("treatment_events")
        .select("brand")
        .eq("event_type", "prescription")
        .not_.is_("brand", "null")
    )
    # Shard 07 R10: the brand distinct-scan default-excludes synthetic so a real-mode
    # resolution never canonicalizes against a synthetic-only brand value.
    from src.repositories.provenance import apply_provenance_filter

    _bq = apply_provenance_filter(_bq, include_synthetic)
    rows = getattr(_bq.limit(_MAX_ROWS).execute(), "data", None) or []
    scan_truncated = len(rows) >= _MAX_ROWS
    if scan_truncated:
        logger.warning(
            "kpi_resolution: brand distinct-scan hit the %d-row cap; a brand beyond "
            "the cap could be missed.",
            _MAX_ROWS,
        )
    distinct_brands = {str(r["brand"]) for r in rows if r.get("brand")}
    return _match_against_distinct(value, distinct_brands), scan_truncated


# Tables this module reads that carry the is_synthetic provenance column (Shard 01).
# A read on one of these default-excludes synthetic rows unless the caller opts in.
_PROVENANCE_TAGGABLE = frozenset(
    {
        "triggers",
        "treatment_events",
        "hcp_profiles",
        "patient_journeys",
        "business_metrics",
        "ml_predictions",
        "episodic_memories",
    }
)


def _fetch_df(
    client: Any,
    table: str,
    columns: str,
    *,
    brand: Optional[str] = None,
    include_synthetic: bool = False,
) -> pd.DataFrame:
    q = client.table(table).select(columns)
    if table == "treatment_events":
        q = q.eq("event_type", "prescription")
        if brand:
            q = q.eq("brand", brand)
    # Shard 07 R10: default-exclude is_synthetic on taggable tables (gated so a table
    # without the column never 42703s).
    if table in _PROVENANCE_TAGGABLE:
        from src.repositories.provenance import apply_provenance_filter

        q = apply_provenance_filter(q, include_synthetic)
    rows = getattr(q.limit(_MAX_ROWS).execute(), "data", None) or []
    if len(rows) >= _MAX_ROWS:
        logger.warning(
            "kpi_resolution: %s fetch hit the %d-row cap; results may be truncated.",
            table,
            _MAX_ROWS,
        )
    return pd.DataFrame(rows)


def _build_conversion_frame(
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    window_days: int = _CONVERSION_WINDOW_DAYS,
    include_synthetic: bool = False,
) -> Optional[KpiFrame]:
    """Materialize the conversion substrate (triggers ⋈ treatment_events) for a
    dynamic ``(brand, region)`` from the REAL tables. Fails closed on
    unrecognized brand/region or empty data; never fabricates.

    Shard 07 R10: every source read default-excludes is_synthetic; a validation run
    opts in with ``include_synthetic=True`` so it can measure the synthetic substrate.
    """
    client = supabase_client if supabase_client is not None else _default_client()

    triggers = _fetch_df(
        client, "triggers", ",".join(_TRIGGER_SELECT), include_synthetic=include_synthetic
    )
    if triggers is None or len(triggers) == 0:
        logger.info("kpi_resolution: no triggers available -> fail closed")
        return None

    hcp_regions = _fetch_df(
        client,
        "hcp_profiles",
        "hcp_id,geographic_region",
        include_synthetic=include_synthetic,
    )

    # Region resolved DYNAMICALLY against the real geographic_region values.
    region_canonical: Optional[str] = None
    if region and str(region).strip():
        distinct_regions = {
            str(r) for r in (hcp_regions.get("geographic_region", pd.Series(dtype=str)).dropna())
        }
        region_canonical = _match_against_distinct(region, distinct_regions)
        if region_canonical is None:
            logger.info("kpi_resolution: unrecognized region %r -> fail closed", region)
            return None

    # Brand resolved DYNAMICALLY against the real treatment_events.brand values.
    # brand is a PG enum -> distinct scan; the scan's cap is tracked so a no-match
    # under a truncated scan is never a silent fail-closed.
    brand_canonical: Optional[str] = None
    brand_scan_truncated = False
    if brand and str(brand).strip():
        brand_canonical, brand_scan_truncated = _resolve_brand_canonical(
            client, brand, include_synthetic=include_synthetic
        )
        if brand_canonical is None:
            logger.info(
                "kpi_resolution: unrecognized brand %r (brand_scan_truncated=%s) -> fail closed",
                brand,
                brand_scan_truncated,
            )
            return None

    events = _fetch_df(
        client,
        "treatment_events",
        "patient_id,event_date,event_type,brand",
        brand=brand_canonical,
        include_synthetic=include_synthetic,
    )

    # No silent caps: if any source fetch (incl. the brand distinct-scan) hit the
    # row cap, the substrate may be a truncated sample — flag it so the caller /
    # response can surface it.
    truncated = (
        len(triggers) >= _MAX_ROWS
        or len(events) >= _MAX_ROWS
        or len(hcp_regions) >= _MAX_ROWS
        or brand_scan_truncated
    )

    kf = _assemble_conversion_frame(
        triggers, hcp_regions, events, region_canonical=region_canonical, window_days=window_days
    )
    if kf is None:
        logger.info(
            "kpi_resolution: conversion frame empty for brand=%r region=%r -> fail closed",
            brand,
            region,
        )
        return None
    kf.is_truncated = truncated
    return kf


# ---------------------------------------------------------------------------
# Dispatch — per-KPI substrate builders (extension point)
# ---------------------------------------------------------------------------
# Map KPI id -> substrate builder. Conversion (WS3-BI-009) is implemented; other
# KPIs return None (honest "no builder yet") until their substrate is added.
_BUILDERS: Dict[str, Callable[..., Optional[KpiFrame]]] = {
    CONVERSION_KPI_ID: _build_conversion_frame,
}


def resolve_kpi_frame(
    kpi: Optional[KPIMetadata],
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    window_days: int = _CONVERSION_WINDOW_DAYS,
    include_synthetic: bool = False,
) -> Optional[KpiFrame]:
    """Resolve the analyzable :class:`KpiFrame` for ``kpi`` at ``(brand, region)``.

    Returns ``None`` (fail closed, never fabricated) when ``kpi`` is ``None``, has
    no substrate builder yet, or no real data resolves.

    Raises:
        Genuine infrastructure errors (e.g. a Supabase connection / auth failure
        from the client) propagate — the caller logs and proceeds WITHOUT data
        (both wired callers, ``chatbot_tools`` and the orchestrator dispatcher,
        wrap this in a best-effort guard). This mirrors the
        :func:`src.services.cohort_resolution.resolve_cohort_frame` contract: the
        service never silently swallows infra failures into a fabricated/empty
        frame; the composable tools then fail closed honestly.
    """
    if kpi is None:
        return None
    builder = _BUILDERS.get(kpi.id)
    if builder is None:
        logger.info(
            "kpi_resolution: no substrate builder for KPI %s (%s) yet -> None",
            kpi.id,
            kpi.name,
        )
        return None
    return builder(
        brand,
        region,
        supabase_client=supabase_client,
        window_days=window_days,
        include_synthetic=include_synthetic,
    )
