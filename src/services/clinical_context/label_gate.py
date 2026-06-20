"""Label-gate: deterministic evaluation of whether a SEGMENT falls inside a
drug's FDA-indicated population.

This is a COMMERCIAL-STRATEGY guardrail (not clinical decision support): it lets
the heterogeneous_optimizer / gap_analyzer flag and de-prioritize a segment whose
defining feature(s) place it outside the indicated population — so a recommendation
does not chase causal uplift into an off-label population.

Criteria provenance (codex HIGH#1): each criterion is tagged `label_evidenced`
(the live OpenFDA label text supports it) vs `config_unconfirmed` (it comes from
the reviewed cohort_constructor candidate but the live label does not state it).
The gate may return a HARD ``off_label`` ONLY on a label-evidenced violation; a
config-unconfirmed violation surfaces as ``indeterminate`` (review), never a silent
hardcoded flag. Pure logic — no network/DB. Criterion shape is reused from
cohort_constructor (the reviewed, column-bound SSOT candidate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

from src.agents.cohort_constructor.types import Criterion, CriterionType, Operator

Verdict = Literal["on_label", "off_label", "indeterminate", "mixed"]
_Relation = Literal["all", "none", "partial", "na"]


@dataclass(frozen=True)
class GateCriterion:
    """A reviewed candidate criterion + whether the live label evidences it."""

    criterion: Criterion
    label_evidenced: bool
    label_evidence: Optional[str] = None  # the matching live-label snippet


@dataclass(frozen=True)
class IndicatedPopulation:
    """The (indication-scoped) indicated-population criteria for a brand.

    ``source``: ``openfda_evidenced`` (>=1 criterion label-evidenced),
    ``config_unconfirmed`` (criteria present but none evidenced by the live label),
    ``unavailable`` (no criteria — fail-open: the gate returns indeterminate).
    """

    brand: str
    indication: str
    criteria: List[GateCriterion] = field(default_factory=list)
    source: str = "unavailable"


@dataclass(frozen=True)
class SegmentDescriptor:
    """How a segment is defined on ONE feature: a scalar/categorical ``value`` OR a
    continuous band [``low``, ``high``] (open-ended if one bound is None)."""

    field: str
    value: object = None
    low: Optional[float] = None
    high: Optional[float] = None
    source: str = ""

    @property
    def is_band(self) -> bool:
        return self.value is None and (self.low is not None or self.high is not None)


@dataclass(frozen=True)
class GateVerdict:
    verdict: Verdict
    failed_criteria: List[str] = field(default_factory=list)
    reason: str = ""
    confirmed_by_label: bool = False


def parse_segment_value(raw: object) -> object:
    """Parse a CATE/gap ``segment_value`` (often a string) to a comparable value:
    booleans, numerics, else the original string. cate_estimator emits one segment
    per unique column value as ``str(value)``, so this recovers the typed value."""
    if isinstance(raw, (bool, int, float)):
        return raw
    s = str(raw).strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s


def descriptor_from_segment(field: str, segment_value: object, source: str = "") -> "SegmentDescriptor":
    """Build a scalar SegmentDescriptor from a (column, value) segment. (CATE/gap
    segments are single-value, not bands; banding callers construct SegmentDescriptor
    directly with low/high.)"""
    return SegmentDescriptor(field=field, value=parse_segment_value(segment_value), source=source)


def _as_float(x: object) -> Optional[float]:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _predicate_relation(op: Operator, value: object, d: SegmentDescriptor) -> _Relation:
    """How much of the segment SATISFIES the raw criterion predicate (op, value):
    all | none | partial (band straddles) | na (not evaluable)."""
    # Set / equality operators — scalar only.
    if op in (Operator.IN, Operator.NOT_IN):
        if d.is_band or d.value is None:
            return "na"
        members = value if isinstance(value, (list, tuple, set)) else [value]
        inside = d.value in members
        if op is Operator.IN:
            return "all" if inside else "none"
        return "all" if not inside else "none"
    if op in (Operator.EQUAL, Operator.NOT_EQUAL):
        if d.is_band or d.value is None:
            return "na"
        eq = d.value == value
        if op is Operator.EQUAL:
            return "all" if eq else "none"
        return "all" if not eq else "none"
    if op is Operator.CONTAINS:
        if d.value is None or not isinstance(d.value, str) or not isinstance(value, str):
            return "na"
        return "all" if value in d.value else "none"

    # Numeric ordering operators — scalar or band.
    thr = _as_float(value)
    if thr is None:
        return "na"

    def _sat(x: float) -> bool:
        if op is Operator.GREATER_EQUAL:
            return x >= thr
        if op is Operator.GREATER:
            return x > thr
        if op is Operator.LESS_EQUAL:
            return x <= thr
        if op is Operator.LESS:
            return x < thr
        return False

    if op is Operator.BETWEEN:
        # value expected [lo, hi]; treat as inclusive range membership (scalar only).
        if d.is_band or d.value is None or not isinstance(value, (list, tuple)) or len(value) != 2:
            return "na"
        lo, hi = _as_float(value[0]), _as_float(value[1])
        xv = _as_float(d.value)
        if lo is None or hi is None or xv is None:
            return "na"
        return "all" if lo <= xv <= hi else "none"

    if not d.is_band:
        xv = _as_float(d.value)
        if xv is None:
            return "na"
        return "all" if _sat(xv) else "none"

    # Band: evaluate the extremes.
    low = d.low if d.low is not None else float("-inf")
    high = d.high if d.high is not None else float("inf")
    lo_sat, hi_sat = _sat(low), _sat(high)
    if lo_sat and hi_sat:
        return "all"
    if not lo_sat and not hi_sat:
        return "none"
    return "partial"


def _evaluate_one(d: SegmentDescriptor, gc: GateCriterion) -> _Relation:
    """satisfy | violate | straddle | na for ONE criterion against the segment.

    Inclusion: satisfy=meets predicate, violate=fails it. Exclusion inverts:
    matching the excluded condition is a violation (off-label)."""
    rel = _predicate_relation(gc.criterion.operator, gc.criterion.value, d)
    if rel == "na":
        return "na"
    if rel == "partial":
        return "straddle"  # type: ignore[return-value]
    is_inclusion = gc.criterion.criterion_type is CriterionType.INCLUSION
    matches = rel == "all"
    if is_inclusion:
        return "satisfy" if matches else "violate"  # type: ignore[return-value]
    # Exclusion: matching the excluded condition => violate (off-label).
    return "violate" if matches else "satisfy"  # type: ignore[return-value]


def _reason(gc: GateCriterion, d: SegmentDescriptor, kind: str) -> str:
    seg = d.value if not d.is_band else f"[{d.low}, {d.high}]"
    rationale = gc.criterion.clinical_rationale or gc.criterion.field
    tag = "" if gc.label_evidenced else " (label-unconfirmed)"
    return f"{d.field}={seg} {kind} '{rationale}'{tag}"


def evaluate_segment(
    descriptors: Sequence[SegmentDescriptor], population: IndicatedPopulation
) -> GateVerdict:
    """Verdict for a (possibly multi-feature) segment vs the indicated population.

    Priority: unavailable/empty -> indeterminate; any label-evidenced violation
    -> off_label; any label-evidenced straddle -> mixed; any config-unconfirmed
    violation/straddle -> indeterminate (review); else evidenced satisfy ->
    on_label; nothing bears on a criterion -> indeterminate."""
    if population.source == "unavailable" or not population.criteria:
        return GateVerdict("indeterminate", [], "no indicated-population criteria available", False)

    by_field: dict[str, list[GateCriterion]] = {}
    for gc in population.criteria:
        by_field.setdefault(gc.criterion.field, []).append(gc)

    evidenced_violation: list[str] = []
    evidenced_straddle: list[str] = []
    unconfirmed_concern: list[str] = []
    evidenced_satisfy = False
    reasons: list[str] = []

    for d in descriptors:
        for gc in by_field.get(d.field, []):
            outcome = _evaluate_one(d, gc)
            if outcome == "na":
                continue
            if outcome == "satisfy":
                if gc.label_evidenced:
                    evidenced_satisfy = True
                continue
            if outcome == "violate":
                if gc.label_evidenced:
                    evidenced_violation.append(gc.criterion.field)
                    reasons.append(_reason(gc, d, "violates"))
                else:
                    unconfirmed_concern.append(gc.criterion.field)
                    reasons.append(_reason(gc, d, "violates"))
            elif outcome == "straddle":
                if gc.label_evidenced:
                    evidenced_straddle.append(gc.criterion.field)
                    reasons.append(_reason(gc, d, "straddles"))
                else:
                    unconfirmed_concern.append(gc.criterion.field)
                    reasons.append(_reason(gc, d, "straddles"))

    if evidenced_violation:
        return GateVerdict("off_label", evidenced_violation, "; ".join(reasons), True)
    if evidenced_straddle:
        return GateVerdict("mixed", evidenced_straddle, "; ".join(reasons), True)
    if unconfirmed_concern:
        return GateVerdict(
            "indeterminate",
            [],
            "; ".join(reasons) + " — not confirmable from the live label; flagged for review",
            False,
        )
    if evidenced_satisfy:
        return GateVerdict("on_label", [], "all intersecting label criteria satisfied", True)
    return GateVerdict("indeterminate", [], "segment bears on no indicated-population criterion", False)
