"""Per-HCP cohort volumes are the patient-panel TRx and say so (canonical TRx lane, codex r1-r6).

Intent (#1356 ruling, #1736): per-HCP prescription-event cohorts are requested
functionality, built "in lock-step with the platform TRx KPI". After the lane the
platform TRx is the canonical business_metrics series, which has no per-HCP grain,
so the lock-step partner is the patient-panel KPI WS3-BI-011. Every per-HCP volume
answer is served as that measure and discloses the canonical grain (codex r6).
"""

import ast
import inspect
import random
import re
from datetime import date
from pathlib import Path

import pytest

from src.agents.cohort_profiler import agent, ask, notes
from src.agents.cohort_profiler.ask import mentions_canonical, merge_cohort_asks, parse_cohort_ask
from src.agents.orchestrator.nodes.synthesizer import SynthesizerNode
from tests.unit.test_agents.test_orchestrator.test_cohort_profiler_volume_tiers import (
    _ALL_BRANDS_90D,
    _Q15,
    _Q43,
    _agent,
)

REPO = Path(__file__).resolve().parents[4]
# A bare "TRx" that is neither "TRx Panel" nor "canonical TRx".
_BARE_TRX = re.compile(r"(?<![Cc]anonical )\bTRx\b(?! Panel)")
_CANONICAL_THRESHOLD = "HCPs who prescribed more than 50 canonical TRx last quarter"
_CANONICAL_TIERS = "Segment HCPs by market-level TRx volume into high, medium and low tiers"
# Codex r5/r6 spellings, built without escape sequences.
_FULL_WIDTH = str.maketrans({c: chr(ord(c) + 0xFEE0) for c in "abcdefghijklmnopqrstuvwxyz"})
_THRESHOLD_ROWS = [
    {"specialty": "oncology", "priority_tier": 1, "n_hcps": 12, "total_trx": 900, "max_trx": 120}
]


def _source() -> str:
    return inspect.getsource(agent)


def test_no_narration_claims_the_platform_trx_substrate():
    source = _source()
    assert "same prescription substrate as the platform TRx KPI" not in source
    # Both HCP renderers emit the ONE footer constant.
    assert source.count("parts.append(PANEL_COHORT_FOOTER)") == 2
    assert "the patient-panel KPI WS3-BI-011" in agent.PANEL_COHORT_FOOTER


def test_both_hcp_cohort_profiles_declare_the_panel_kpi():
    hits = []
    for node in ast.walk(ast.parse(_source())):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            k.value: v
            for k, v in zip(node.keys, node.values, strict=True)
            if isinstance(k, ast.Constant)
        }
        entity = keys.get("entity")
        if isinstance(entity, ast.Constant) and entity.value == "hcp" and "cohort_size" in keys:
            kpi = keys.get("volume_kpi_id")
            hits.append(isinstance(kpi, ast.Constant) and kpi.value == "WS3-BI-011")
    assert len(hits) == 2 and all(hits), hits


def test_the_nrx_breakdown_uses_the_panel_nrx():
    assert agent._NRX_KPI_ID == "WS3-BI-012"


def test_the_routing_contract_names_the_panel():
    text = (
        REPO / "scripts" / "benchmarks" / "routing" / "data" / "agent_contracts.json"
    ).read_text()
    assert "threshold-filtered TRx Panel volume" in text
    assert "threshold-filtered TRx volume only" not in text


#: Every document that DECLARES what this agent measures. A contract doc is read by
#: people and by routing benchmarks, so a false substrate claim here outlives the code.
_CONTRACT_DOCS = (
    "scripts/benchmarks/routing/data/agent_contracts.json",
    "src/agents/cohort_profiler/CONTRACT_VALIDATION.md",
    ".claude/contracts/tier0/cohort_profiler.md",
)
#: Matches the claim however it is hyphenated or possessive. codex iter1 HIGH: the
#: earlier check grepped the single literal "platform TRx KPI", which MISSED
#: "platform TRx-KPI substrate" in CONTRACT_VALIDATION.md -- a proxy for the question
#: rather than the question. Ask what the sentence CLAIMS, not how it is spelled.
_PLATFORM_TRX_CLAIM = re.compile(r"platform(?:'s)?[ \-‑]TRx", re.IGNORECASE)


@pytest.mark.parametrize("rel", _CONTRACT_DOCS)
def test_no_contract_document_claims_the_platform_trx_substrate(rel):
    """After the lane the platform TRx is the canonical business_metrics series, which
    has NO per-HCP grain. Any document still tying this agent's per-HCP counts to it
    asserts an equivalence that is false by ~1,300x."""
    path = REPO / rel
    assert path.exists(), rel
    hits = [
        (i, line.strip()[:140])
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if _PLATFORM_TRX_CLAIM.search(line)
    ]
    assert hits == [], f"{rel} still claims the platform-TRx substrate: {hits}"


@pytest.mark.parametrize("rel", _CONTRACT_DOCS)
def test_no_contract_document_still_names_the_canonical_nrx_for_this_agent(rel):
    """``_NRX_KPI_ID`` moved to the patient-panel WS3-BI-012; a contract that still
    says WS3-BI-006 describes a path the code no longer takes."""
    text = (REPO / rel).read_text()
    assert agent._NRX_KPI_ID == "WS3-BI-012", "premise changed — re-derive this guard"
    assert "WS3-BI-006" not in text, f"{rel} still names the canonical NRx WS3-BI-006"


def test_no_contract_document_says_per_hcp_trx_without_naming_the_panel(rel=None):
    """``per-HCP TRx`` with no ``Panel`` is the two-scales phrasing this lane retires."""
    offenders = {}
    for rel in _CONTRACT_DOCS:
        bad = [
            (i, line.strip()[:140])
            for i, line in enumerate((REPO / rel).read_text().splitlines(), 1)
            if re.search(r"per-HCP TRx(?! Panel)", line)
        ]
        if bad:
            offenders[rel] = bad
    assert offenders == {}, offenders


def test_no_basis_refusal_remains():
    """A STANDING PROHIBITION on a rejected design — not a lane-regression guard.

    Codex r6 rejected a basis-refusal path: detection never changes what executes, so
    no refusal branch or ``volume_basis`` field may exist. codex iter1 LOW correctly
    observed that this passes on ``origin/main`` too, because the rejected design was
    never shipped anywhere. That is the POINT and not a defect: its job is to stop the
    rejected option being introduced later, the way a lint rule does. Recorded
    explicitly so nobody "fixes" it by deleting it after finding it has no lane teeth.
    """
    assert not hasattr(agent.CohortProfilerAgent, "_basis_unavailable")
    assert not hasattr(ask, "AMBIGUOUS_BASIS_CLARIFICATION")
    assert "volume_basis" not in inspect.getsource(ask) + _source()


# ------------------------------------------------------------ narration census


def _user_visible_strings(module):
    """Every string constant in ``module`` except docstrings (f-string pieces included)."""
    tree = ast.parse(inspect.getsource(module))
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def test_census_no_bare_trx_in_any_user_visible_string():
    hits = [
        (m.__name__, line, text)
        for m in (agent, ask, notes)  # codex r11: the adapter module obeys the census too
        for line, text in _user_visible_strings(m)
        if _BARE_TRX.search(text)
    ]
    assert hits == [], hits


def test_no_module_claims_the_platform_trx_substrate():
    for m in (agent, ask):
        source = inspect.getsource(m)
        assert "platform TRx KPI" not in source, m.__name__
        assert "platform's TRx KPI" not in source, m.__name__


# ------------------------------------------------- disclosure through synthesis


async def _synthesize(result):
    """The SINGLE-result synthesis branch. The multi-agent LLM and fallback branches are
    covered in test_synthesizer_basis_notes.py (codex r7)."""
    state = await SynthesizerNode().execute({"agent_results": [result]})
    return state["synthesized_response"]


def _assert_never_presented_as_canonical(text: str) -> None:
    """No figure is ever presented as canonical TRx (codex r6/r8). Outside the two standing
    constants no line mentions the canonical measure at all (the footer no longer does,
    and the detector sees through formatting), and every line that pairs "TRx" with a
    number names TRx Panel."""
    standing = {agent.CANONICAL_GRAIN_DISCLOSURE, agent.CANONICAL_REQUEST_SENTENCE}
    for line in text.splitlines():
        if line.strip().strip("_") in standing:
            continue
        low = line.lower()
        assert not mentions_canonical(line), line
        if "trx" in low and re.search(r"\d", line):
            assert "trx panel" in low, line


async def _served(payload, rows):
    agent_, db = _agent(db_rows=[rows], today=date(2026, 8, 19))
    out = await agent_.analyze(payload)
    assert out["status"] == "completed", out.get("errors")
    assert db.calls, payload  # the panel cohort ran
    profile = out["cohort_profile"]
    assert profile["entity"] == "hcp"
    assert profile["volume_kpi_id"] == "WS3-BI-011"
    assert profile["basis_note"] == agent.CANONICAL_GRAIN_DISCLOSURE
    assert "basis_unavailable" not in out
    assert agent.CANONICAL_GRAIN_DISCLOSURE in out["narrative"]
    text = await _synthesize({"agent_name": "cohort_profiler", "success": True, "result": out})
    assert agent.CANONICAL_GRAIN_DISCLOSURE in text
    _assert_never_presented_as_canonical(text)
    return out, text


_ORDINARY = [
    ({"query": _Q15}, _THRESHOLD_ROWS),
    ({"query": _Q43}, _ALL_BRANDS_90D),
    # Codex r6: unrelated negation must never block an ordinary #1356 ask.
    ({"query": "HCPs with no more than 50 TRx Panel events last quarter"}, _THRESHOLD_ROWS),
    (
        {"query": "HCPs who prescribed more than 50 TRx last quarter without diabetes"},
        _THRESHOLD_ROWS,
    ),
    ({"query": "Show HCP TRx Panel volume tiers"}, _ALL_BRANDS_90D),
    ({"query": "profile the Remibrutinib cohort", "raw_user_query": _Q43}, _ALL_BRANDS_90D),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,rows", _ORDINARY)
async def test_ordinary_1356_asks_are_served_with_the_standing_disclosure(payload, rows):
    out, text = await _served(payload, rows)
    assert out["cohort_profile"]["canonical_requested"] is False
    assert "canonical_request_note" not in out["cohort_profile"]
    assert agent.CANONICAL_REQUEST_SENTENCE not in out["narrative"]
    assert agent.CANONICAL_REQUEST_SENTENCE not in text


_CANONICAL_EXAMPLES = [
    ({"query": _CANONICAL_THRESHOLD}, _THRESHOLD_ROWS),
    ({"query": _CANONICAL_TIERS}, _ALL_BRANDS_90D),
    # Codex r3-r5 examples: negation, markup, link, full-width, entity, zero-width.
    ({"query": "Rank HCPs by canonical TRx, not TRx Panel"}, _THRESHOLD_ROWS),
    ({"query": "Rank HCPs by _canonical_ TRx"}, _THRESHOLD_ROWS),
    ({"query": "Rank HCPs by **canonical** TRx"}, _THRESHOLD_ROWS),
    ({"query": "Rank HCPs by `business_metrics` TRx"}, _THRESHOLD_ROWS),
    ({"query": "Rank HCPs by canonical\nTRx"}, _THRESHOLD_ROWS),
    (
        {"query": "HCPs with more than 50 [canonical](https://example.com) TRx last quarter"},
        _THRESHOLD_ROWS,
    ),
    (
        {
            "query": "HCPs with more than 50 "
            + "canonical".translate(_FULL_WIDTH)
            + " TRx last quarter"
        },
        _THRESHOLD_ROWS,
    ),
    ({"query": "HCPs with more than 50 &#99;anonical TRx last quarter"}, _THRESHOLD_ROWS),
    (
        {"query": "HCPs with more than 50 cano" + chr(0x200B) + "nical TRx last quarter"},
        _THRESHOLD_ROWS,
    ),
    ({"query": "No, canonical TRx: HCPs with more than 50 TRx last quarter"}, _THRESHOLD_ROWS),
    ({"query": "HCPs with more than 50 not non-canonical TRx last quarter"}, _THRESHOLD_ROWS),
    ({"query": "HCPs with more than 50 TRx, not canonical TRx and not TRx Panel"}, _THRESHOLD_ROWS),
    # Codex r6: formatting that splits the token.
    (
        {"query": "HCPs with more than 50 not canonical TRx; use can<b>on</b>ical TRx instead"},
        _THRESHOLD_ROWS,
    ),
    ({"query": "Rank HCPs by can**on**ical TRx"}, _THRESHOLD_ROWS),
    ({"query": "Rank HCPs by can<b>on</b>ical TRx"}, _THRESHOLD_ROWS),
    ({"query": "Rank HCPs by c-a-n-o-n-i-c-a-l TRx"}, _THRESHOLD_ROWS),
    # Merge: the mention is the OR of both texts, whichever one carries it.
    (
        {
            "query": "HCPs with more than 50 TRx Panel events last quarter",
            "raw_user_query": "Rank HCPs by can**on**ical TRx",
        },
        _THRESHOLD_ROWS,
    ),
    ({"query": _CANONICAL_TIERS, "raw_user_query": _Q43}, _ALL_BRANDS_90D),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,rows", _CANONICAL_EXAMPLES)
async def test_canonical_asks_are_served_as_the_panel_with_a_leading_canonical_sentence(
    payload, rows
):
    out, text = await _served(payload, rows)
    assert out["cohort_profile"]["canonical_requested"] is True
    assert out["cohort_profile"]["canonical_request_note"] == agent.CANONICAL_REQUEST_SENTENCE
    assert out["narrative"].startswith(agent.CANONICAL_REQUEST_SENTENCE)
    assert text.lstrip().startswith(agent.CANONICAL_REQUEST_SENTENCE)
    assert text.count(agent.CANONICAL_REQUEST_SENTENCE) == 1


@pytest.mark.asyncio
async def test_a_failed_hcp_result_carries_no_figures_and_is_returned_unchanged():
    agent_, _db = _agent(db_rows=[[], []], today=date(2026, 8, 19))
    out = await agent_.analyze({"query": _CANONICAL_THRESHOLD.replace("canonical ", "")})
    assert out["status"] == "failed"
    assert "cohort_profile" not in out and "basis_note" not in out


# ------------------------------------------------------ canonical mention detector

_CANONICAL_TOKENS = [
    "canonical",
    "market-level",
    "market level",
    "business metrics",
    "business_metrics",
    "WS3-BI-005",
    "WS3-BI-006",
    "WS3-BI-007",
    "WS3-BI-008",
]
_FORMATTING = [
    "**",
    "*",
    "_",
    "__",
    "~~",
    "`",
    "-",
    " ",
    ".",
    ",",
    "<b>",
    "</b>",
    "<span class='x'>",
    "</span>",
    chr(0x200B),
    chr(0xAD),
    chr(0xA0),
    "&shy;",
    "&#8203;",
]


def _formatted(token: str, rng: random.Random) -> str:
    """``token`` with 1-3 random formatting pieces after about half its characters,
    random upper case, and optionally full-width letters or a markdown link."""
    pieces = []
    for ch in token:
        pieces.append(ch.upper() if rng.random() < 0.3 else ch)
        if rng.random() < 0.5:
            pieces.append("".join(rng.choice(_FORMATTING) for _ in range(rng.randint(1, 3))))
    text = "".join(pieces)
    if rng.random() < 0.25:
        text = text.translate(_FULL_WIDTH)
    if rng.random() < 0.25:
        text = f"[{text}](https://example.com/a.b?c=d)"
    return text


def test_mentions_canonical_survives_formatting_inside_every_canonical_token():
    """Codex r6 property: formatting inserted ANYWHERE inside a canonical token never hides it."""
    rng = random.Random(20260915)
    for _ in range(2000):
        query = f"HCPs with more than 50 {_formatted(rng.choice(_CANONICAL_TOKENS), rng)} TRx last quarter"
        assert mentions_canonical(query), repr(query)


@pytest.mark.parametrize(
    "query",
    [
        _Q15,
        _Q43,
        "HCPs with no more than 50 TRx Panel events last quarter",
        "HCPs who prescribed more than 50 TRx last quarter without diabetes",
        "HCPs with more than 50 canon TRx",
        "anonymous HCPs above 30 TRx",
        "technical HCP profile",
        "clinical HCPs with more than 50 TRx",
        "market leaders by TRx Panel",
        "business metric for HCPs",
        "HCPs by WS3-BI-011",
        "HCPs by WS3-BI-009",
    ],
)
def test_mentions_canonical_is_false_without_a_canonical_token(query):
    assert mentions_canonical(query) is False


@pytest.mark.parametrize(
    "rewrite,raw,expected",
    [
        (_CANONICAL_THRESHOLD, _Q15, True),
        (_Q15, "Rank HCPs by can**on**ical TRx", True),
        (_CANONICAL_TIERS, _CANONICAL_THRESHOLD, True),
        (_Q15, _Q43, False),
    ],
)
def test_merge_canonical_requested_is_the_or_of_both_texts(rewrite, raw, expected):
    merged = merge_cohort_asks(parse_cohort_ask(rewrite), parse_cohort_ask(raw))
    assert merged.canonical_requested is expected
