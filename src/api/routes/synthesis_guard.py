"""Deterministic post-synthesis cross-check of superlative prose vs the answer's own tables.

#1691: across the 2026-08-18 eval runs the synthesis layer repeatedly printed an
exact, correct markdown table and then asserted a min/max in prose contradicted
by that same table ("largest 0.198" under a row showing 0.267; "lowest 0.231"
beside a 0.224; "fastest-materializing (30-day lag)" over an 18-day row). The
build_synthesis_prompt #1550 rule already forbids exactly this and demonstrably
does not stop it (6+ instances in 6 different turns across two runs on one
image), so this check is deterministic code, not more prompt text.

Synthesis STREAMS to the client, so a detected contradiction cannot be rewritten
away — the caller appends a factual correction note instead (see
``build_superlative_correction``). A false correction is user-visible, so
findings are two-tier: only high-confidence ones (``Finding.visible``) go into
the note; the rest are for logging/monitoring. Every suppression rule below was
forced by one measured false positive — a sweep over the 102 real responses of
the two 2026-08-18 runs, plus the certification rerun's turn 3.3 (#1701):

- paren annotations pair backward ("0.730 (Kisqali, lowest)" describes 0.730,
  not the next number in the sentence);
- inside one parenthetical, a superlative that names its own quantity never
  pairs with a number introduced by a DIFFERENT axis label ("(highest
  propensity, n=1,016)" — rerun 3.3, #1701: "n=" annotates sample size while
  "highest" binds to propensity, whose 58.4% IS the column max);
- "largest NEGATIVE driver: … -0.073" — a negative column-min satisfies a
  max-superlative (and symmetrically for min-words);
- Total/Sum rows are excluded from columns ("largest bucket (1,238)" is true
  among buckets; the 2,341 total must not defeat it);
- negated ("isn't the strongest"), ordinal ("second-largest", "next-strongest")
  and plural ("the two strongest") claims are skipped;
- restrictively-scoped claims ("largest … among rep-attributable drivers",
  "highest engagement ROI … among the three regions shown") demote to
  log-only — a column check cannot see row subsets. Deictic full-table scopes
  ("among these four") do NOT demote;
- clauses split at depth-0 commas and coordinating conjunctions, so parallel
  claims never cross-pair ("X (+0.41) is the largest, Y (+0.27) second" must
  not read +0.27 as the largest; "at 8.69 and lowest … at 1.62" must not read
  8.69 as the lowest). Commas inside parentheses ("(0.198, via trx_volume)")
  do not split;
- a backward-paired claim is visible only when the claim phrase names the
  contradicting column ("carries the largest estimated *effect size*" vs the
  "Effect Size" column — the canonical 5.7 instance);
- #1717 (found by a grader's positive control on the 2026-08-19 full eval's
  turn 4.1, "**oncology specialists number 256 HCPs** — the largest specialty
  segment … (out of 751 total HCPs, …)"): a dash immediately after a closing
  bold no longer splits the sentence — the bolded subject is appositive to
  the superlative that follows, not a separate clause; when forward pairing
  binds a superlative to a number appearing in NO table column (751 is a
  prose-only cohort total), the claim falls back to the nearest preceding
  **bolded** number instead of being silently dropped; and "out of <number>
  …" is a whole-universe denominator, not a restrictive row-subset scope;
- the #1717 sweep then measured two new FPs from exactly that unsplitting
  ("largest engagement shortfall — 63.0% achievement", "the largest target
  miss" beside 73.9%, both TRUE claims whose number is the column MINIMUM):
  a superlative naming an INVERSE quantity (shortfall, miss, gap, deficit,
  …) of the paired column is direction-ambiguous, so — like NEUTRAL_WORDS —
  it is contradicted only by a strictly interior value.

Known non-goals (measured, accepted): wrong-row attribution where the cited
number IS the column extremum (morning 5.1: "highest … at 3 each" — 3 is the
column max, just not that row's value), and contradictions between table CELLS
(rerun 2.5's two "Largest n" annotations). Catching those needs entity linking,
not arithmetic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

MAX_WORDS = {"largest", "biggest", "greatest", "highest", "strongest", "longest"}
MIN_WORDS = {"smallest", "lowest", "weakest", "shortest"}
# Direction-ambiguous ("fastest growth" is a max, "fastest lag" a min): flag
# only when the paired value is strictly interior, which contradicts EVERY
# reading of the superlative.
NEUTRAL_WORDS = {"fastest", "slowest", "best", "worst"}

_ALL_WORDS = MAX_WORDS | MIN_WORDS | NEUTRAL_WORDS
_KEYWORD_RE = re.compile(r"\b(" + "|".join(sorted(_ALL_WORDS)) + r")\b", re.IGNORECASE)
_SKIP_PREFIX_RE = re.compile(
    r"(?:second|third|fourth|fifth|sixth|2nd|3rd|4th|5th|6th|\d+th|next|two|three|four|five)"
    r"[-\s]$",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|isn['’]t|aren['’]t|wasn['’]t|weren['’]t|never|nor)\b", re.IGNORECASE
)
# Number token in prose: sign (ascii or U+2212) must not be glued to a word
# ("trailing-30d" yields 30, never -30); $/%, thousands commas allowed.
_NUMBER_RE = re.compile(r"(?<![\w.])[+\-−]?\$?\d[\d,]*(?:\.\d+)?%?")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?(?:\s+\d{2}:\d{2})?\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# A dash splits like sentence punctuation UNLESS it directly follows a closing
# bold — "**… 256 HCPs** — the largest …" is one claim whose bolded subject the
# split would sever from its superlative (#1717, eval 4.1).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|(?<!\*\*)\s+[—–]\s+")
_CONJUNCTION_RE = re.compile(r"\s(?:and|but|while|whereas|though|although)\s")
_SCOPE_RE = re.compile(r"\b(among|within|across|out of|excluding)\b\s+(.{0,40})", re.IGNORECASE)
#: Words that keep a scope phrase deictic (referring to the whole table just
#: shown) rather than restrictive (a row subset the column check cannot see).
_SCOPE_ALLOWED = {
    "the",
    "these",
    "those",
    "all",
    "both",
    "shown",
    "listed",
    "here",
    "above",
    "table",
    "results",
    "them",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}
# A table cell is numeric if it is one number plus at most a short unit suffix.
_CELL_RE = re.compile(
    r"^[+\-−]?\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:%|x|pp|pts?|days?|weeks?|months?)?$",
    re.IGNORECASE,
)
_TOTAL_ROW_RE = re.compile(r"^(?:total|sum|overall|all|combined)\b", re.IGNORECASE)
#: "highest propensity" — the quantity word the superlative itself names.
_NAMED_QUANTITY_RE = re.compile(r"\s+([A-Za-z][A-Za-z_-]*)")
#: "n=1,016" / "SE = 0.007" — an axis label introducing the number that follows.
_LABEL_INTRO_RE = re.compile(r"([A-Za-z][A-Za-z_]*)\s*=\s*$")
_MARKUP_RE = re.compile(r"[*_`]")
_WORD_RE = re.compile(r"[a-z]{3,}")
_DIGIT_RE = re.compile(r"\d")
#: One **bold** span — the answer's own emphasis marking a claim's subject.
_BOLD_SPAN_RE = re.compile(r"\*\*[^*\n]+?\*\*")
#: Nouns measuring a DEFICIT of the column quantity ("largest engagement
#: shortfall" ≈ lowest achievement). Scanned between the superlative and its
#: first following digit; a hit makes the claim direction-neutral (#1717).
_INVERSE_NOUN_RE = re.compile(
    r"\b(?:shortfalls?|miss(?:es)?|gaps?|deficits?|declines?|drops?|drawdowns?"
    r"|shortages?|loss(?:es)?|drags?|underperformance|undershoots?)\b",
    re.IGNORECASE,
)

#: Forward pairing: the first number after the keyword, this close.
_FORWARD_WINDOW = 60
#: Backward pairing: the nearest number before, this close.
_BACKWARD_WINDOW = 150
#: How far past the keyword to look for restrictive scope / column-name words.
_SCOPE_SCAN = 80
_HEADER_SCAN = 40
#: Cap on findings surfaced per answer — one is a correction, five is noise.
_MAX_FINDINGS = 2
_EPS = 1e-9


@dataclass(frozen=True)
class Finding:
    """One superlative claim contradicted by every table column carrying its number."""

    keyword: str
    number_text: str
    value: float
    column_header: str
    column_min: float
    column_max: float
    visible: bool
    #: #1717: the superlative names an inverse quantity ("largest shortfall")
    #: — direction-ambiguous, fired only on an interior value; the correction
    #: note states the column span rather than a direction.
    inverse_axis: bool = False


def _parse_cell(cell: str) -> Optional[float]:
    cleaned = _MARKUP_RE.sub("", cell).strip()
    m = _CELL_RE.match(cleaned)
    if not m:
        return None
    sign = -1.0 if cleaned.startswith(("-", "−")) else 1.0
    return sign * float(m.group(1).replace(",", ""))


def _extract_columns(text: str) -> List[Tuple[str, List[float]]]:
    """(header, numeric values) per column, over every markdown table in the text.

    Rows whose first cell is Total/Sum/Overall are excluded — an aggregate row
    would falsely defeat every true "largest bucket" claim.
    """
    columns: List[Tuple[str, List[float]]] = []
    block: List[str] = []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 3:  # header, separator, >=1 data row
            headers = [
                _MARKUP_RE.sub("", c).strip() for c in block[0].strip().strip("|").split("|")
            ]
            data_rows = []
            for data_line in block[2:]:
                cells = [c.strip() for c in data_line.strip().strip("|").split("|")]
                first = next((_MARKUP_RE.sub("", c).strip() for c in cells if c.strip()), "")
                if not _TOTAL_ROW_RE.match(first):
                    data_rows.append(cells)
            for col_idx, header in enumerate(headers):
                values = []
                for cells in data_rows:
                    if col_idx < len(cells):
                        parsed = _parse_cell(cells[col_idx])
                        if parsed is not None:
                            values.append(parsed)
                if len(values) >= 2:
                    columns.append((header or f"column {col_idx + 1}", values))
        block = []
    return columns


def _parse_prose_number(token: str) -> Optional[float]:
    sign = -1.0 if token.startswith(("-", "−")) else 1.0
    stripped = token.lstrip("+-−").lstrip("$").rstrip("%").replace(",", "")
    try:
        return sign * float(stripped)
    except ValueError:
        return None


def _split_clauses(segment: str) -> List[str]:
    """Split a sentence segment at depth-0 commas and coordinating conjunctions.

    Parallel claims must not cross-pair, but a comma inside parentheses
    ("(0.198, via trx_volume) carries the largest …") is part of one claim.
    """
    clauses: List[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            if ch == "," and i + 1 < len(segment) and segment[i + 1] == " ":
                clauses.append(segment[start:i])
                start = i + 1
            else:
                m = _CONJUNCTION_RE.match(segment, i)
                if m and i > start:
                    clauses.append(segment[start:i])
                    start = m.end() - 1
                    i = m.end() - 1
                    continue
        i += 1
    clauses.append(segment[start:])
    return [c for c in clauses if c.strip()]


def _own_axis_annotation(clause: str, num_pos: int, quantity: str) -> bool:
    """True when the number at num_pos is introduced by its own axis label
    ("n=1,016", "SE = 0.007") that differs from the quantity the superlative
    names — a same-row annotation, never the superlative's referent (#1701)."""
    m = _LABEL_INTRO_RE.search(clause[:num_pos])
    return m is not None and m.group(1).lower() != quantity


def _paren_span(clause: str, pos: int) -> Optional[Tuple[int, int]]:
    """The (start, end) of the innermost paren group containing pos, if any."""
    depth = 0
    start = None
    for i, ch in enumerate(clause):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
            if depth == 0 and start is not None:
                if start < pos < i:
                    return (start, i)
                start = None
    return None


def _preceding_bold_number(
    clause: str, numbers: List[Tuple[int, str, float]], kw_start: int
) -> Optional[Tuple[str, float]]:
    """The nearest number BEFORE kw_start that sits inside a **bold** span.

    #1717 fallback referent: in "**oncology specialists number 256 HCPs** — the
    largest … (out of 751 total …)" the bolded subject holds the claim's real
    number; used only when the forward pair matches no table column."""
    spans = [(m.start(), m.end()) for m in _BOLD_SPAN_RE.finditer(clause)]
    candidates = [
        (p, t, v)
        for p, t, v in numbers
        if p < kw_start and kw_start - p <= _BACKWARD_WINDOW and any(s < p < e for s, e in spans)
    ]
    if not candidates:
        return None
    _, t, v = max(candidates, key=lambda n: n[0])
    return (t, v)


@dataclass(frozen=True)
class _Claim:
    keyword: str
    number_text: str
    value: float
    forward: bool
    tail: str  # clause text following the keyword, for scope/header scans
    #: #1717: preceding bolded number, tried only when ``value`` (a forward
    #: pair) appears in no table column — e.g. a prose-only cohort total.
    fallback_text: Optional[str] = None
    fallback_value: Optional[float] = None
    #: #1717: superlative names an inverse quantity — direction-neutral.
    inverse_axis: bool = False


def _iter_claims(text: str) -> List[_Claim]:
    claims: List[_Claim] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            continue
        # Blank (don't delete — positions must hold) dates/years so their
        # digits can never be read as the number a superlative describes.
        clean = _DATE_RE.sub(lambda m: " " * len(m.group()), line)
        clean = _YEAR_RE.sub(lambda m: " " * len(m.group()), clean)
        clauses = (
            clause
            for segment in _SENTENCE_SPLIT_RE.split(clean)
            for clause in _split_clauses(segment)
        )
        for clause in clauses:
            numbers: List[Tuple[int, str, float]] = []
            for m in _NUMBER_RE.finditer(clause):
                parsed = _parse_prose_number(m.group())
                if parsed is not None:
                    numbers.append((m.start(), m.group(), parsed))
            if not numbers:
                continue
            for kw in _KEYWORD_RE.finditer(clause):
                prefix = clause[max(0, kw.start() - 10) : kw.start()]
                if _SKIP_PREFIX_RE.search(prefix):
                    continue
                neg_ctx = clause[max(0, kw.start() - 25) : kw.start()]
                if _NEGATION_RE.search(neg_ctx):
                    continue
                # #1717: "largest engagement shortfall — 63.0% achievement" —
                # the quantity the superlative names (scanned up to its first
                # following digit) measures a deficit of the paired column,
                # so the claim carries no usable direction.
                qspan = clause[kw.end() : kw.end() + _SCOPE_SCAN]
                digit = _DIGIT_RE.search(qspan)
                if digit:
                    qspan = qspan[: digit.start()]
                inverse_axis = bool(_INVERSE_NOUN_RE.search(qspan))
                paren = _paren_span(clause, kw.start())
                pair: Optional[Tuple[str, float, bool]] = None
                fallback: Optional[Tuple[str, float]] = None
                if paren is not None:
                    inside = [(p, t, v) for p, t, v in numbers if paren[0] < p < paren[1]]
                    # "(highest propensity, n=1,016)" — rerun 3.3 (#1701): the
                    # superlative names its own quantity while the number is a
                    # differently-labelled annotation of the same row, so they
                    # must not pair. With no other in-paren number left, fall
                    # through to the backward rule like any other annotating
                    # parenthetical.
                    named = _NAMED_QUANTITY_RE.match(clause, kw.end())
                    if named:
                        quantity = named.group(1).lower()
                        inside = [
                            (p, t, v)
                            for p, t, v in inside
                            if not _own_axis_annotation(clause, p, quantity)
                        ]
                    if inside:
                        p, t, v = min(inside, key=lambda n: abs(n[0] - kw.start()))
                        pair = (t, v, True)
                    else:
                        # "(Kisqali, 2026-07-28, lowest)" annotates the number
                        # BEFORE the parenthetical, never the one after it.
                        before = [
                            (p, t, v)
                            for p, t, v in numbers
                            if p < paren[0] and paren[0] - p <= _BACKWARD_WINDOW
                        ]
                        if before:
                            p, t, v = max(before, key=lambda n: n[0])
                            pair = (t, v, False)
                else:
                    ahead = [
                        (p, t, v)
                        for p, t, v in numbers
                        if p > kw.start() and p - kw.start() <= _FORWARD_WINDOW
                    ]
                    if ahead:
                        p, t, v = min(ahead, key=lambda n: n[0])
                        pair = (t, v, True)
                        # #1717: should this forward number turn out to live in
                        # no table column, the preceding bolded subject number
                        # is the claim's real referent.
                        fallback = _preceding_bold_number(clause, numbers, kw.start())
                    else:
                        behind = [
                            (p, t, v)
                            for p, t, v in numbers
                            if p < kw.start() and kw.start() - p <= _BACKWARD_WINDOW
                        ]
                        if behind:
                            p, t, v = max(behind, key=lambda n: n[0])
                            pair = (t, v, False)
                if pair is None:
                    continue
                claims.append(
                    _Claim(
                        keyword=kw.group().lower(),
                        number_text=pair[0],
                        value=pair[1],
                        forward=pair[2],
                        tail=clause[kw.start() : kw.start() + _SCOPE_SCAN],
                        fallback_text=fallback[0] if fallback else None,
                        fallback_value=fallback[1] if fallback else None,
                        inverse_axis=inverse_axis,
                    )
                )
    return claims


def _restrictively_scoped(tail: str) -> bool:
    """True when the claim narrows to a row subset the column check cannot see."""
    for m in _SCOPE_RE.finditer(tail):
        # "out of 751 total HCPs" states the whole-universe denominator, not a
        # row subset — the column check CAN adjudicate such a claim (#1717).
        if m.group(1).lower() == "out of" and m.group(2).lstrip("$")[:1].isdigit():
            continue
        words = _WORD_RE.findall(m.group(2).lower())
        return any(w not in _SCOPE_ALLOWED for w in words)
    return False


def _satisfied(keyword: str, value: float, values: List[float], inverse_axis: bool = False) -> bool:
    is_max = abs(value - max(values)) < _EPS
    is_min = abs(value - min(values)) < _EPS
    if inverse_axis:
        # "largest shortfall/miss/gap": the named quantity may be the column's
        # deficit ("shortfall" vs an Achievement column, where the MIN is the
        # largest shortfall) or the column itself — either extremum can be the
        # true referent, so only an interior value contradicts every reading.
        return is_max or is_min
    if keyword in MAX_WORDS:
        # A negative column-min satisfies "largest (negative driver)".
        return is_max or (value < 0 and is_min)
    if keyword in MIN_WORDS:
        return is_min or (value < 0 and is_max)
    return is_max or is_min


def find_superlative_contradictions(text: str) -> List[Finding]:
    """Superlative claims whose number fails its direction in EVERY table column carrying it."""
    columns = _extract_columns(text)
    if not columns:
        return []
    findings: List[Finding] = []
    seen: set = set()
    for claim in _iter_claims(text):
        number_text, value = claim.number_text, claim.value
        carrying = [
            (header, values)
            for header, values in columns
            if any(abs(v - value) < _EPS for v in values)
        ]
        if not carrying and claim.fallback_value is not None and claim.fallback_text is not None:
            # #1717: the forward pair matched no table column (a prose-only
            # total like "out of 751") — the claim's referent is the preceding
            # bolded subject number, so check that instead of dropping.
            fb_value = claim.fallback_value
            fb_carrying = [
                (header, values)
                for header, values in columns
                if any(abs(v - fb_value) < _EPS for v in values)
            ]
            if fb_carrying:
                number_text, value = claim.fallback_text, fb_value
                carrying = fb_carrying
        if not carrying:
            continue
        if any(
            _satisfied(claim.keyword, value, values, claim.inverse_axis) for _, values in carrying
        ):
            continue
        if (claim.keyword, value) in seen:
            continue
        seen.add((claim.keyword, value))
        header, values = carrying[0]
        if claim.forward:
            visible = not _restrictively_scoped(claim.tail)
        else:
            # Backward pairs are looser: visible only when the claim phrase
            # itself names the contradicting column and is not scope-narrowed.
            header_words = _WORD_RE.findall(header.lower())
            near = claim.tail[:_HEADER_SCAN].lower()
            visible = (
                bool(header_words)
                and all(w in near for w in header_words)
                and not _restrictively_scoped(claim.tail)
            )
        findings.append(
            Finding(
                keyword=claim.keyword,
                number_text=number_text,
                value=value,
                column_header=header,
                column_min=min(values),
                column_max=max(values),
                visible=visible,
                inverse_axis=claim.inverse_axis,
            )
        )
    return findings


def _fmt(v: float) -> str:
    return f"{v:g}"


def build_superlative_correction(text: str) -> str:
    """A factual correction note to append after streaming, or '' when clean.

    Only ``visible``-tier findings are surfaced; the note states values read
    from the answer's own table, so even a residual false positive asserts
    nothing untrue — it points at the table.
    """
    findings = [f for f in find_superlative_contradictions(text) if f.visible]
    if not findings:
        return ""
    parts = []
    for f in findings[:_MAX_FINDINGS]:
        if f.keyword in MAX_WORDS and not f.inverse_axis:
            conflict = f"its largest value is actually **{_fmt(f.column_max)}**"
        elif f.keyword in MIN_WORDS and not f.inverse_axis:
            conflict = f"its smallest value is actually **{_fmt(f.column_min)}**"
        else:
            # NEUTRAL_WORDS and inverse-axis claims (#1717) carry no usable
            # direction — state the span, which is factual under any reading.
            conflict = f"it spans **{_fmt(f.column_min)}–{_fmt(f.column_max)}**"
        parts.append(
            f'the prose describes **{f.number_text}** as "{f.keyword}", but in the '
            f'"{f.column_header}" column above, {conflict}'
        )
    return (
        "\n\n> ⚠️ Automated table cross-check: "
        + "; ".join(parts)
        + ". Where prose and table disagree, the table values are authoritative."
    )
