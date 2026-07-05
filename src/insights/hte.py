"""HTE strategic insight: interpret ONE completed segment-level CATE analysis.

Grounding contract (mirrors ``executive_brief``): the route derives EVERY figure
SERVER-SIDE from the persisted segment-analysis record (``analysis_id`` is the
only caller input), and the LM output passes a fail-closed numeric guard before
it is served — any numeric claim the grounding cannot vouch for (including a
flipped or word-spelled sign, a pp/% unit swap, a unit-bearing figure re-used
bare, a name digit re-used out of context, or a vouched number misattributed
as a segment/subgroup count, to the wrong segment, or to the wrong metric)
downgrades the response to the deterministic factual fallback. Effects on the
binary clinical outcomes are probability deltas presented in PERCENTAGE POINTS
(pp), matching the /ai-insights HTE card's display unit.

The insight must keep two questions separate (the card's core honesty issue):

* "Is the treatment working in a segment?"  -> per-segment CI vs zero.
* "Should we target segments differentially?" -> segments must beat the
  OVERALL ATE (the above-ATE lift gate); per-segment significance vs zero is
  NOT a targeting license.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class HTEInsightSignature(dspy.Signature):
        """Interpret ONE completed segment-level CATE (heterogeneous treatment
        effect) analysis for a pharma brand analyst, STRICTLY grounded in the
        figures provided. Use ONLY the numbers given — never invent, re-derive,
        round differently, or extrapolate. Effects are probability deltas in
        percentage points (pp); never call them "percent growth".

        Answer BOTH questions, keeping them clearly separate:
        (1) Is the treatment effect real? Read the overall ATE and how many
        segments' CIs exclude zero.
        (2) Is differential targeting warranted? Use ONLY the provided
        targeting verdict (expected lift / allocation summary) — a segment
        being significant vs zero does NOT by itself justify targeting it;
        that requires the segment to beat the overall ATE.

        Name the strongest and weakest segments with their pp effects, state
        the single most decision-relevant implication, and ALWAYS close with
        the caveat that these are model-based estimates from one causal-forest
        analysis on an observational cohort."""

        scope: str = dspy.InputField(desc="Treatment -> outcome, brand filter, cohort size")
        effect_summary: str = dspy.InputField(
            desc="Overall ATE (pp), significant-segment count, heterogeneity score"
        )
        segments: str = dspy.InputField(
            desc="Per-segment CATE lines: value, pp effect, CI, n, significance"
        )
        targeting: str = dspy.InputField(
            desc="Differential-targeting verdict (expected lift, allocation summary)"
        )

        interpretation: str = dspy.OutputField(
            desc="Grounded strategic read: effect reality, heterogeneity, targeting verdict, caveat"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, decision-oriented takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    HTEInsightSignature = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pp(value: Any) -> str | None:
    """Format a probability delta as signed percentage points at 1 decimal."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    v = f * 100.0
    return f"{v:+.1f}pp"


def _fmt_int(value: Any) -> str | None:
    try:
        i = int(value)
    except (TypeError, ValueError):
        return None
    return f"{i:,}"


def _num_of(rendering: str | None) -> str | None:
    """The comma-stripped digits of a rendered figure ("+11.1pp" -> "11.1")."""
    if not rendering:
        return None
    m = re.search(r"\d[\d,]*(?:\.\d+)?", rendering)
    return m.group(0).replace(",", "") if m else None


# ---------------------------------------------------------------------------
# Vouched phrases — name digits are grounded only IN CONTEXT
# ---------------------------------------------------------------------------

# Natural-word expansions for unit letters embedded in variable names, so
# "persistent_180d" keeps "180-day persistence" grounded.
_UNIT_WORDS = {"d": "day", "w": "week", "m": "month", "y": "year"}


def _phrase_variants(raw: Any) -> tuple[set[str], set[str]]:
    """A digit-bearing name plus its natural paraphrases, split by ambiguity.

    Returns ``(safe, ambiguous)``. Safe phrases are stripped from text BEFORE
    numeric-claim extraction, so their internal digits pass the guard only in
    context: "persistent_180d" and "180-day persistence" are fine, a bare
    re-use like "Treat 180 patients" still trips it. EVERY rendering of a
    pure numeric range (a band value like "50-65", any dash or "to" form) is
    AMBIGUOUS — identical to a quantity range in running text ("50-65
    significant segments") — and is stripped only when anchored by band
    context (see ``_strip_phrases``).
    """
    s = str(raw or "").strip()
    if not s or not re.search(r"\d", s):
        return set(), set()
    band = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", s)  # age bands like "50-65"
    if band:
        a, b = band.group(1), band.group(2)
        return set(), {
            f"{a}-{b}",
            f"{a}–{b}",
            f"{a}—{b}",
            f"{a} - {b}",
            f"{a} -{b}",
            f"{a}- {b}",
            f"{a} – {b}",
            f"{a} to {b}",
        }
    safe = {s}
    for sep in (" ", "-", ""):
        safe.add(s.replace("_", sep))
    for num, unit in re.findall(r"(\d+)([A-Za-z]+)", s):
        safe.update({f"{num}{unit}", f"{num} {unit}"})
        word = _UNIT_WORDS.get(unit.lower())
        if word:
            safe.update({f"{num}-{word}", f"{num} {word}", f"{num}-{word}s", f"{num} {word}s"})
    return safe, set()


# ---------------------------------------------------------------------------
# Typography normalization — every guard rule sees ONE canonical spelling
# ---------------------------------------------------------------------------

# Line-leading list markers ("- 11.1pp", "• +2.8pp"): removed so a leading
# hyphen stays a bullet, while any in-sentence "- 11.1pp" that survives reads
# as a minus sign.
_BULLET_RE = re.compile(r"(?m)^[ \t]*[-–—•*]\s+")
# Digit-adjacent unicode dashes/minus signs ("50—65", "−2.8") become ASCII
# "-"; spaced en/em dashes in prose ("the effect — strong") are punctuation
# and stay untouched. Word-adjacent dashes ("out–of") survive normalization,
# so _FRACTION_RE keeps the class in its separators.
_DASH_CLASS = "‐‑‒–—―−﹣－"
_DASH_RE = re.compile(rf"[{_DASH_CLASS}](?=\d)|(?<=\d)[{_DASH_CLASS}]")
# Fraction/division/fullwidth slashes ("3⁄3", "3／3") become ASCII "/".
_SLASH_RE = re.compile(r"[⁄∕／]")
# Unambiguous plus glyphs ("＋2.8", "➕2.8") become ASCII "+".
_PLUS_RE = re.compile(r"[＋﹢➕]")
# ASCII plus-minus ("+/-2.8pp") becomes "±" so it reads as the never-vouched
# ± sign instead of a signed claim built from its "-" half.
_PLUSMINUS_RE = re.compile(r"[+]\s*/\s*[-−]|[-−]\s*/\s*[+]")
# Whitespace runs collapse to one space (newlines kept for the bullet rule).
_WS_RE = re.compile(r"[^\S\n]+")


def _normalize(text: str) -> str:
    """Canonicalize LM typography before any guard rule runs."""
    text = _BULLET_RE.sub("", text)
    text = _PLUSMINUS_RE.sub("±", text)
    text = _DASH_RE.sub("-", text)
    text = _SLASH_RE.sub("/", text)
    text = _PLUS_RE.sub("+", text)
    return _WS_RE.sub(" ", text)


# A vouched phrase immediately followed by a unit word is a numeric claim in
# disguise ("50 to 65 percent" from the 50-65 age band), not a name mention —
# leave it in place so its digits face the guard.
_PHRASE_UNIT_LOOKAHEAD = r"(?!\s*(?:%|pp\b|percent\b|percentage\b|points?\b))"

# Nouns a number can be (mis)attributed to as a segment count. Plural-only
# forms (bands/groups) stay out of the singular set because their singulars
# name ONE band in ordinary prose ("the 50-65 band").
_SEG_NOUNS = (
    r"(?:segments?|sub-?segments?|sub-?groups?|sub-?populations?|cohorts?"
    r"|bands|groups|strata|dimensions?|categor(?:y|ies)|clusters?|buckets?|tiers?)"
)
_SEG_NOUNS_PLURAL = (
    r"(?:segments|sub-?segments|sub-?groups|sub-?populations|cohorts"
    r"|bands|groups|strata|dimensions|categories|clusters|buckets|tiers)"
)
# A run of modifier words between a number and a segment noun: any number of
# non-digit tokens (optionally comma/semicolon-tailed), never crossing a
# sentence boundary because "." is not in the token alphabet.
_SEG_MODIFIER_TOKENS = r"(?:(?!\d)[\w-]+[,;]?\s+)*"

# Ambiguous range phrases are name mentions only in band context: preceded by
# a band-ish noun ("patients 50 to 65", "aged 50-65", parenthesized forms),
# followed by a SINGULAR band noun ("the 50-65 band", "the 50-65 segment"),
# or directly attached to a dimension name ("age_band=50-65"). Unanchored,
# "50 to 65 significant segments" is a quantity claim and must face the
# guard. NO anchor may LAUNDER a count claim: whenever the range heads a
# plural segment-count phrase ("Patients (50-65) significant segments clear
# zero", "age_band=50-65 significant segments ..."), it stays in place and
# its digits face the guard.
_AMBIG_BACK_ANCHOR = r"\b(aged?|ages|patients?|bands?|groups?|cohorts?)\s+\(?"
_AMBIG_FWD_ANCHOR = (
    r"(?=\)?\s+(?:age[\s-])?(?:band|group|cohort|segment|sub-?group|range|bracket)\b"
    rf"(?!\s+{_SEG_NOUNS_PLURAL}\b))"
)
_AMBIG_COUNT_LOOKAHEAD = rf"(?!\)?[,;]?\s+{_SEG_MODIFIER_TOKENS}{_SEG_NOUNS_PLURAL}\b)"


def _strip_phrases(text: str, phrases: Sequence[str], ambiguous: Sequence[str] = ()) -> str:
    """Normalize typography, then remove vouched phrases (longest first) so
    only bare numbers remain. A phrase matches only as a whole lexical token
    — "50-65+" and "arm10" are DIFFERENT labels, not mentions of a grounded
    "50-65" or "arm1", so their digits face the guard."""
    text = _normalize(text)
    for p in ambiguous:
        if not p:
            continue
        esc = r"(?<![\w+])" + re.escape(p) + r"(?![\w+])" + _PHRASE_UNIT_LOOKAHEAD
        # eq/back anchors precede the range, so the count reading survives to
        # its right — refuse to strip when a plural segment-count phrase
        # follows (fail-closed: a faithful anchored range followed later in
        # the clause by a plural segment noun falls back rather than risk
        # laundering). The fwd anchor's own singular noun sits immediately
        # after the range, which defeats the count reading by itself.
        guarded = esc + _AMBIG_COUNT_LOOKAHEAD
        text = re.sub(r"=\s*" + guarded, "= ", text, flags=re.IGNORECASE)
        text = re.sub(_AMBIG_BACK_ANCHOR + guarded, r"\1 ", text, flags=re.IGNORECASE)
        text = re.sub(esc + _AMBIG_FWD_ANCHOR, " ", text, flags=re.IGNORECASE)
    for p in phrases:
        if p:
            text = re.sub(
                r"(?<![\w+])" + re.escape(p) + r"(?![\w+])" + _PHRASE_UNIT_LOOKAHEAD,
                " ",
                text,
                flags=re.IGNORECASE,
            )
    return text


# ---------------------------------------------------------------------------
# Fail-closed output guard
# ---------------------------------------------------------------------------

# A numeric claim is (sign, number, unit): "+11.1pp" / "-2.8" / "95%". After
# normalization (bullets removed, whitespace collapsed, digit-adjacent
# dashes ASCII, plus glyphs ASCII), a sign counts when attached or one space
# away from the number ("is - 11.1pp", "ATE: + 2.8pp"), spelled out
# ("negative 11.1pp", "minus-11.1pp"), or spelled out with a short bridge
# before a UNIT-BEARING figure ("negative net 11.1pp", "positive, 2.8pp") —
# while "top-2" keeps its hyphen inside the word and "Plus, 2 of 3" stays a
# discourse marker. "±" is a sign no grounding ever renders, so it never
# vouches. Direction VERBS ("declined by 11.1pp") are semantic paraphrase a
# lexical guard cannot adjudicate ("reduced non-persistence by 11.1pp" would
# be the same lexical shape as a true claim) — accepted boundary, as are
# spelled-out word numbers ("three of three").
_UNIT_PATTERN = r"(?:pp\b|%|percentage[\s-]points?\b|percent\b)"
_CLAIM_RE = re.compile(
    r"(?:"
    r"(?<![\w.\-])(?P<sym>[+\-−±∓]) ?"
    r"|\b(?P<word>negative|minus|positive|plus)[\s-]+(?=\d)"
    r"|\b(?P<word2>negative|minus|positive)[\s,]+(?:[a-z][\w-]*[\s,]+){0,3}?"
    rf"(?=\d[\d,]*(?:\.\d+)?\s*{_UNIT_PATTERN})"
    r")?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    rf"(?:\s*(?P<unit>{_UNIT_PATTERN}))?",
    re.IGNORECASE,
)
# Count-fraction claims: "13/14", "13 of 14", "13 out of 14", "13-of-14",
# "13 over 14", "13 in 14", any dash flavour ("3‑out‑of‑3"): digit-adjacent
# dashes and slashes are normalized to "-" and "/", word-adjacent dashes are
# matched here directly.
_FRACTION_RE = re.compile(
    rf"\b(?<![.,])(\d+)(?:\s*/\s*|[-{_DASH_CLASS}\s]+"
    rf"(?:out[-{_DASH_CLASS}\s]+of|of|over|in|per|among|amongst|from)"
    rf"[-{_DASH_CLASS}\s]+)(\d+)\b",
    re.IGNORECASE,
)
# A number attributed to segments ("81 significant segments", "1,385
# significant, clinically relevant subgroups") is a segment-count claim
# regardless of whether the number is vouched elsewhere — it must be the true
# significant or total count. The modifier run is unbounded but cannot cross
# a sentence boundary or another number. Plural population/time words keep
# their own attribution ("1,385 patients in the strongest segment" counts
# patients, not segments) and end the match — plural-only, because singular
# forms before a noun are adjectives ("1,385 patient segments" IS a
# segment-count claim).
_SEG_COUNT_EXEMPT = (
    r"(?:patients|hcps|physicians|prescribers|people|persons|individuals"
    r"|respondents|records|rows|days|weeks|months|years)"
)
# Token alphabet: anything but whitespace and sentence enders; a token may
# not START with a digit (another number ends the claim) or ")" (a closing
# paren directly after the number means it was parenthetical — "(n=1,385),"
# is not counting what follows). The tail glued to the number itself
# ("1,385-significant") tolerates punctuation but not ")". The number may
# not start mid-decimal or mid-thousands ("1pp" inside "+11.1pp" is not a
# count of anything).
_SEG_COUNT_RE = re.compile(
    rf"\b(?<![.,])(\d(?:[\d,]*\d)?)[^\s.!?)]*\s+"
    rf"(?:(?!{_SEG_COUNT_EXEMPT}[^\w\s]*\s)(?![\d)])[^\s.!?]+\s+)*"
    rf"{_SEG_NOUNS}\b",
    re.IGNORECASE,
)
# A segment count inside a clause asserting significance ("All 3 segments
# have 95% CIs excluding zero") is a SIGNIFICANT-segment count — the true
# total cannot vouch it. Negated significance ("not significant") does not
# make the clause a significance claim.
_SIG_PREDICATE_RE = re.compile(
    r"(?<!not )\bsignificant\b|CIs?\s+exclud\w*\s+zero|excludes?\s+zero|clears?\s+zero",
    re.IGNORECASE,
)
# ... but "2 of 3 segments have CIs excluding zero" is exempt: the fraction
# rule validates the numerator, so the denominator stays a total.
_FRACTION_PRECEDER_RE = re.compile(r"\d\s*(?:of|out[-\s]+of|in|over|/)\s*$", re.IGNORECASE)
# Chip-style reversed form: "Significant segments: 3".
_SIG_COUNT_LABEL_RE = re.compile(rf"significant\s+{_SEG_NOUNS}\s*[:=]\s*(\d[\d,]*)", re.IGNORECASE)


# A postpositive sign adjective after a unit-bearing figure ("an 11.1pp
# negative effect", "11.1pp, a negative result", "11.1pp, indicating a
# negative effect") signs the claim. Allows one appositive comma, a
# discourse linker, an article, and one bridging word — never a sentence
# boundary.
_POSTPOSITIVE_SIGN_RE = re.compile(
    r",? (?:(?:indicating|suggesting|implying|meaning|showing|reflecting"
    r"|signaling|representing|therefore|thus|hence|i\.e\.) )?"
    r"(?:(?:a|an|the) )?(?:[a-z][\w-]* )?(negative|positive)\b",
    re.IGNORECASE,
)


def _extract_claims(text: str) -> list[tuple[str, str, str]]:
    """(sign, comma-stripped number, normalized unit) for every number."""
    claims: list[tuple[str, str, str]] = []
    for m in _CLAIM_RE.finditer(text):
        raw_sign = (m.group("sym") or m.group("word") or m.group("word2") or "").strip().lower()
        unit_raw = (m.group("unit") or "").lower()
        if not raw_sign and unit_raw:
            post = _POSTPOSITIVE_SIGN_RE.match(text[m.end() :])
            if post:
                raw_sign = post.group(1).lower()
        if raw_sign in ("-", "−", "negative", "minus"):
            sign = "-"
        elif raw_sign in ("+", "positive", "plus"):
            sign = "+"
        elif raw_sign in ("±", "∓"):
            sign = "±"  # never rendered by any grounding -> never vouches
        else:
            sign = ""
        if unit_raw == "pp" or "point" in unit_raw:
            unit = "pp"
        elif unit_raw:
            unit = "%"
        else:
            unit = ""
        claims.append((sign, m.group("num").replace(",", ""), unit))
    return claims


def _claim_vouched(
    sign: str, num: str, unit: str, vouched: dict[str, set[tuple[str, str]]]
) -> bool:
    """True iff a grounded rendering of ``num`` exists with the SAME unit that
    the claim's sign does not contradict. Omitting the sign is fine; omitting
    or swapping the unit is not — "95" rendered only as "95%" does not vouch
    a bare "95", so unit-bearing figures cannot be re-used as counts."""
    for v_sign, v_unit in vouched.get(num, ()):
        if (not sign or sign == v_sign) and unit == v_unit:
            return True
    return False


# ---------------------------------------------------------------------------
# Attribution checks — a vouched number must belong to what it is claimed for
# ---------------------------------------------------------------------------

# Windows never cross sentence/semicolon boundaries; the sentence splitters
# leave decimals intact ("." splits only before whitespace/end).
_WINDOW_BREAK_RE = re.compile(r"[;\n]|[.!?](?=\s|$)")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?=\s|$)|\n+")
# The lift anchor also covers targeting-benefit paraphrase families ("the
# incremental gain from differential targeting", "targeting offers ...",
# "differential-targeting opportunity", "targeting improves ... by") — any
# figure claimed as the value of targeting IS a lift claim.
_LIFT_ANCHOR_RE = re.compile(
    r"\b(?:expected\s+)?(?:lift|uplift)\b"
    r"|\b(?:gains?|improvements?|benefits?|value|advantage|upside|impact)\s+"
    r"(?:from|of)\s+(?:[\w-]+\s+){0,2}?targeting\b"
    r"|\btargeting\s+(?:offers?|yields?|delivers?|provides?|adds?|generates?|produces?"
    r"|improves?|boosts?|raises?|increases?|lifts?)\b"
    r"|\b(?:differential[-\s])?targeting\s+opportunity\b",
    re.IGNORECASE,
)
_ATE_ANCHOR_RE = re.compile(
    r"\bATE\b|\baverage\s+treatment\s+effect\b"
    r"|\b(?:overall|population-level|aggregate|average)\W{0,3}"
    r"(?:the\s+)?(?:treatment\s+)?effect\b"
    r"|\btreatment\s+effect\s+overall\b",
    re.IGNORECASE,
)
_METRIC_ANCHORS = (_LIFT_ANCHOR_RE, _ATE_ANCHOR_RE)


def _metric_value_claims(window: str) -> list[str]:
    """pp/decimal figures in a window — the shapes a lift/ATE value takes.
    %-unit figures are CI-level annotations ("95% CIs"), not effect values."""
    return [
        num
        for _s, num, unit in _extract_claims(window)
        if unit == "pp" or (not unit and "." in num)
    ]


def _metric_misattributed(
    text: str, anchor_re: re.Pattern[str], value_num: str | None, allowed: set[str]
) -> bool:
    """True iff a metric-shaped figure is tied to this metric's wording but
    is not the metric's rendered value ("expected lift is +17.7pp" when the
    true lift is +0.0pp). Fail-closed: if the metric was never rendered, any
    figure attributed to it rejects. Binding follows the copula: the FIRST
    figure after the anchor in the same clause is the metric's claimed value
    ("expected lift ... , ... , is +17.7pp" rejects while "lift is +0.0pp
    because high severity (+17.7pp) leads" stays legal); with no following
    figure, the NEAREST preceding one binds ("a +17.7pp overall effect")."""
    for m in anchor_re.finditer(text):
        after = text[m.end() :]
        cut = _WINDOW_BREAK_RE.search(after)
        if cut:
            after = after[: cut.start()]
        for other in _METRIC_ANCHORS:
            if other is not anchor_re:
                o = other.search(after)
                if o:
                    after = after[: o.start()]
        after_claims = _metric_value_claims(after)
        if after_claims:
            # An explicit comparison ("overall ATE: +17.7pp versus +11.1pp")
            # names both sides — legal when the metric's value is one of them.
            if value_num in after_claims and re.search(
                r"\b(?:versus|vs\.?|compared|against)\b", after, re.IGNORECASE
            ):
                continue
            bound = after_claims[0]
        else:
            start = max(0, m.start() - 40)
            before = text[start : m.start()]
            if start > 0 and not text[start - 1].isspace():
                before = before.split(" ", 1)[-1] if " " in before else ""
            breaks = list(_WINDOW_BREAK_RE.finditer(before))
            if breaks:
                before = before[breaks[-1].end() :]
            before_claims = _metric_value_claims(before)
            if not before_claims:
                continue
            bound = before_claims[-1]
        if bound != value_num and bound not in allowed:
            return True
    return False


def _segment_attribution_ok(norm_text: str, g: dict[str, Any]) -> bool:
    """A sentence naming exactly ONE segment may only carry that row's
    figures plus the global metrics — "the age_band=50-65 segment responds
    at +17.7pp" with another row's effect rejects. Sentences naming several
    segments (comparisons) or none fall back to global vouching."""
    rows: list[dict[str, Any]] = g.get("segment_rows") or []
    if not rows:
        return True
    for sentence in _SENTENCE_SPLIT_RE.split(norm_text):
        mentioned = [r for r in rows if re.search(r["mention"], sentence, flags=re.IGNORECASE)]
        if len(mentioned) != 1:
            continue
        allowed: set[str] = mentioned[0]["numbers"] | g["global_numbers"]
        stripped = _strip_phrases(sentence, g["phrases"], g["ambiguous_phrases"])
        for _s, num, unit in _extract_claims(stripped):
            # Governed: unit-bearing figures, decimals, n-sized integers, and
            # any KNOWN sample size regardless of digit count (a 3-digit
            # segment n can cross-attribute just as well as a 4-digit one).
            governed = bool(unit) or "." in num or len(num) >= 4 or num in g["sample_numbers"]
            if governed and num not in allowed:
                return False
    return True


def _is_grounded(candidate: str, g: dict[str, Any]) -> bool:
    """True iff every numeric claim in ``candidate`` is vouched by the grounding.

    Fail-closed: any digit sequence the grounding did not render (different
    rounding, re-derived deltas, invented figures, flipped signs, swapped
    units, name digits re-used out of context) rejects the whole output.
    Count fractions whose denominator is the segment total must state the
    true significant count; a number attributed to segments must be the true
    significant or total count; a figure tied to the lift/ATE wording must be
    that metric's value; and a sentence naming one segment may only carry
    that segment's figures (plus globals).
    """
    norm = _normalize(candidate)
    metric_nums: dict[str, str | None] = g["metric_nums"]
    # The heterogeneity score is a legal annotation next to either metric
    # ("+11.1pp overall, heterogeneity 0.26"); unit-strict vouching still
    # rejects it re-used as a pp value.
    het_ok = {n for n in (metric_nums.get("het"),) if n}
    if _metric_misattributed(norm, _LIFT_ANCHOR_RE, metric_nums["lift"], het_ok):
        return False
    if _metric_misattributed(norm, _ATE_ANCHOR_RE, metric_nums["ate"], het_ok):
        return False
    if not _segment_attribution_ok(norm, g):
        return False
    text = _strip_phrases(norm, g["phrases"], g["ambiguous_phrases"])
    vouched: dict[str, set[tuple[str, str]]] = g["vouched"]
    seg_counts = {str(g["sig_count"]), str(g["total_count"])}
    for seg in _SEG_COUNT_RE.finditer(text):
        num = seg.group(1).replace(",", "")
        if num not in seg_counts:
            return False
        clause_start = 0
        for b in _WINDOW_BREAK_RE.finditer(text, 0, seg.start()):
            clause_start = b.end()
        clause_end_m = _WINDOW_BREAK_RE.search(text, seg.end())
        clause = text[clause_start : clause_end_m.start() if clause_end_m else len(text)]
        if (
            num != str(g["sig_count"])
            and _SIG_PREDICATE_RE.search(clause)
            and not _FRACTION_PRECEDER_RE.search(text[: seg.start(1)])
        ):
            return False
    for label in _SIG_COUNT_LABEL_RE.finditer(text):
        if label.group(1).replace(",", "") != str(g["sig_count"]):
            return False
    for frac in _FRACTION_RE.finditer(text):
        m_str, k_str = frac.group(1), frac.group(2)
        if k_str == str(g["total_count"]):
            # Any "m of <segment total>" claim is a significance-count claim:
            # the numerator must be the actual significant count. Both digits
            # being individually vouched is NOT enough ("3 of 3" from a true
            # 2-of-3 would otherwise pass).
            if m_str != str(g["sig_count"]):
                return False
            continue
        if not (_claim_vouched("", m_str, "", vouched) and _claim_vouched("", k_str, "", vouched)):
            return False
    return all(_claim_vouched(s, n, u, vouched) for s, n, u in _extract_claims(text))


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def build_grounding(record: dict[str, Any]) -> dict[str, Any]:
    """Build the grounded prompt inputs + the guard's vouched vocabulary.

    ``record`` is a plain-dict projection of the persisted
    ``SegmentAnalysisResponse`` (see the /insights/hte route). The guard
    vocabulary is EXTRACTED from the exact strings rendered into the prompt —
    sign and unit included — so the LM can only echo figures as given.
    """
    treatment = record.get("treatment_var") or "the treatment"
    outcome = record.get("outcome_var") or "the outcome"
    brand = record.get("brand")
    ci_level = record.get("confidence_level")
    ci_pct = f"{round(float(ci_level) * 100):d}" if ci_level else "95"

    rows: list[dict[str, Any]] = []
    for dimension, results in (record.get("cate_by_segment") or {}).items():
        for r in results or []:
            rows.append({**r, "dimension": dimension})

    overall_ate = record.get("overall_ate")
    has_signal = bool(rows) and overall_ate is not None

    n_total = sum(int(r.get("sample_size") or 0) for r in rows)
    # Segment dimensions partition the same cohort, so the cohort size is the
    # per-dimension sum, not the all-rows sum.
    dims = {r["dimension"] for r in rows}
    if dims:
        n_total = max(
            sum(int(r.get("sample_size") or 0) for r in rows if r["dimension"] == d) for d in dims
        )

    sig_count = sum(1 for r in rows if r.get("statistical_significance"))
    total_count = len(rows)

    ate_pp = _pp(overall_ate)
    het = record.get("heterogeneity_score")
    het_str = f"{float(het):.2f}" if het is not None and math.isfinite(float(het)) else None

    scope = (
        f"{treatment} -> {outcome}"
        + (f", brand filter {brand}" if brand else ", all brands")
        + (f", cohort n={_fmt_int(n_total)}" if n_total else "")
        + f", {ci_pct}% CIs"
    )

    effect_summary = (
        f"Overall ATE {ate_pp or '—'}; "
        f"{sig_count} of {total_count} segments have {ci_pct}% CIs excluding zero; "
        + (
            f"heterogeneity score {het_str} (0-1 scale)"
            if het_str
            else "heterogeneity score unavailable"
        )
    )

    seg_lines: list[str] = []
    ordered = sorted(rows, key=lambda r: float(r.get("cate_estimate") or 0.0), reverse=True)
    for r in ordered:
        name = f"{r['dimension']}={r.get('segment_value')}"
        cate_s = _pp(r.get("cate_estimate")) or "—"
        lo_s = _pp(r.get("cate_ci_lower")) or "—"
        hi_s = _pp(r.get("cate_ci_upper")) or "—"
        n_s = _fmt_int(r.get("sample_size")) or "—"
        sig = "significant" if r.get("statistical_significance") else "not significant"
        seg_lines.append(f"{name}: {cate_s} [CI {lo_s} to {hi_s}], n={n_s}, {sig}")

    # expected_lift_pp is stored as a probability FRACTION despite its name:
    # policy_learner validates it in [0, 1] and multiplies by 100 only at
    # display. Render through _pp like every other effect — a 0.10 lift is
    # +10.0pp, not +0.1pp.
    lift_s = _pp(record.get("expected_lift_pp"))
    allocation = str(record.get("optimal_allocation_summary") or "").strip()
    if len(allocation) > 300:
        allocation = allocation[:297] + "..."
    targeting = (f"Expected lift from differential targeting: {lift_s}. " if lift_s else "") + (
        allocation if allocation else "No allocation summary available."
    )

    grounding_chips = [
        {"label": "Overall ATE", "value": ate_pp or "—"},
        {"label": "Significant segments", "value": f"{sig_count}/{total_count}"},
        {"label": "Heterogeneity", "value": het_str or "—"},
        {"label": "n", "value": _fmt_int(n_total) or "—"},
    ]

    # Digit-bearing NAMES (variables, brand, dimensions, segment values) are
    # grounded as PHRASES, not free-floating numbers: their digits pass the
    # guard only in context.
    phrases: set[str] = set()
    ambiguous: set[str] = set()
    for name_part in (treatment, outcome, brand, *dims):
        safe_v, ambig_v = _phrase_variants(name_part)
        phrases |= safe_v
        ambiguous |= ambig_v
    for r in rows:
        safe_v, ambig_v = _phrase_variants(r.get("segment_value"))
        phrases |= safe_v
        ambiguous |= ambig_v
    phrase_list = sorted(phrases, key=len, reverse=True)
    ambiguous_list = sorted(ambiguous, key=len, reverse=True)

    vouched: dict[str, set[tuple[str, str]]] = {}
    # The grounding text is OURS, so ambiguous range forms in it are known
    # name mentions ("age_band=50-65: ...") — strip them unconditionally here;
    # only untrusted candidate text gets the anchored treatment.
    grounded_text = _strip_phrases(
        "\n".join([scope, effect_summary, *seg_lines, targeting]),
        [*phrase_list, *ambiguous_list],
    )
    for sign, num, unit in _extract_claims(grounded_text):
        vouched.setdefault(num, set()).add((sign, unit))

    # Attribution vocabulary: which number belongs to which segment/metric.
    # Global metrics are legal in any sentence; a sentence naming exactly one
    # segment is additionally restricted to that row's figures.
    global_numbers = {"0", "1"}  # the "(0-1 scale)" rendering
    for rendering in (ate_pp, lift_s, het_str, ci_pct, str(sig_count), str(total_count)):
        g_num = _num_of(rendering)
        if g_num:
            global_numbers.add(g_num)
    if n_total:
        global_numbers.add(str(n_total))
    sample_numbers = {str(int(r["sample_size"])) for r in rows if r.get("sample_size")}
    if n_total:
        sample_numbers.add(str(n_total))

    # Context nouns mirror the _SEG_NOUNS family: any noun the guard treats
    # as segment-language ("the high category", "the low bucket") must also
    # count as segment context here, or the attribution check silently skips.
    seg_ctx_words = {
        "severity",
        "band",
        "bands",
        "segment",
        "segments",
        "subsegment",
        "group",
        "groups",
        "subgroup",
        "subgroups",
        "cohort",
        "cohorts",
        "tier",
        "tiers",
        "bucket",
        "buckets",
        "category",
        "categories",
        "cluster",
        "clusters",
        "subpopulation",
        "subpopulations",
        "population",
        "populations",
        "stratum",
        "strata",
        "dimension",
        "dimensions",
        "responders",
        "patients",
    }
    for d in dims:
        seg_ctx_words.update(w for w in re.split(r"[_\W]+", str(d).lower()) if len(w) >= 3)
    ctx_pat = "(?:" + "|".join(sorted(seg_ctx_words)) + ")"
    segment_rows: list[dict[str, Any]] = []
    for r in ordered:
        value = str(r.get("segment_value") or "").strip()
        if not value:
            continue
        row_numbers: set[str] = set()
        for rendering in (
            _pp(r.get("cate_estimate")),
            _pp(r.get("cate_ci_lower")),
            _pp(r.get("cate_ci_upper")),
            _fmt_int(r.get("sample_size")),
        ):
            r_num = _num_of(rendering)
            if r_num:
                row_numbers.add(r_num)
        variants = {value, value.replace("_", " "), value.replace("_", "-")}
        safe_v, ambig_v = _phrase_variants(value)
        variants |= safe_v | ambig_v
        if re.search(r"\d", value) or len(value) >= 5 or " " in value or "_" in value:
            mention = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True) if v)
        else:
            # Short common-word values ("high", "low") are segment mentions
            # next to segment context ("high severity", "band=high",
            # "disease_severity_band, high") or when they head a response
            # verb ("High responds at ...") — table-like prose after an
            # explicit dimension mention.
            v_esc = re.escape(value)
            mention = (
                rf"\b{v_esc}[=,:\s-]+(?:[\w-]+[\s-]+)?{ctx_pat}\b"
                rf"|\b{ctx_pat}[\w-]*[=,:\s-]+(?:[\w-]+[\s-]+)?{v_esc}\b"
                rf"|\b{v_esc}\s+(?:responds?|shows?|leads?|gains?|performs?|lags?"
                rf"|trails?|outperforms?|underperforms?|ranks?|sits?|clears?"
                rf"|has|have|had|delivers?|posts?|records?|achieves?|reaches?"
                rf"|stands?|remains?)\b"
            )
        segment_rows.append({"mention": mention, "numbers": row_numbers})

    return {
        "scope": scope,
        "effect_summary": effect_summary,
        "segments": "\n".join(seg_lines),
        "targeting": targeting,
        "grounding": grounding_chips,
        "phrases": phrase_list,
        "ambiguous_phrases": ambiguous_list,
        "vouched": vouched,
        "has_signal": has_signal,
        "sig_count": sig_count,
        "total_count": total_count,
        "metric_nums": {"ate": _num_of(ate_pp), "lift": _num_of(lift_s), "het": _num_of(het_str)},
        "global_numbers": global_numbers,
        "sample_numbers": sample_numbers,
        "segment_rows": segment_rows,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    if not g["has_signal"]:
        return {
            "insight": (
                "The persisted analysis contains no per-segment CATE results, so no "
                "grounded heterogeneity interpretation can be produced. Re-run the "
                "segment-level CATE analysis to generate one."
            ),
            "key_takeaways": [],
            "grounding": [],
            "is_fallback": True,
        }
    insight = (
        f"For {g['scope']}: {g['effect_summary']}. {g['targeting']} "
        "These are model-based estimates from one causal-forest analysis on an "
        "observational cohort. (Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["effect_summary"], g["targeting"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    if not g["has_signal"]:
        return _fallback(g)
    pred = run_signature(
        HTEInsightSignature,
        scope=g["scope"],
        effect_summary=g["effect_summary"],
        segments=g["segments"],
        targeting=g["targeting"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    takeaways = normalize_list(getattr(pred, "key_takeaways", []))
    rejected = [t for t in [interpretation, *takeaways] if not _is_grounded(t, g)]
    if rejected:
        logger.warning(
            "HTE insight LM output carried %d ungrounded numeric claim(s); using factual fallback",
            len(rejected),
        )
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": takeaways,
        "grounding": g["grounding"],
        "is_fallback": False,
    }
