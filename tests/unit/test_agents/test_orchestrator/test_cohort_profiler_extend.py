"""Tests for the #1356 cohort_profiler extension (parts 1 + 2 of the ratified
2026-07-29 ``extend:cohort_profiler`` ruling).

Empirical defects being pinned (benchmark q11/q15, confirmed on both surfaces
2026-07-29):

* q11 ("Build a patient cohort for Remibrutinib CSU with inclusion criteria for
  adults over 18 diagnosed in 2024") returned a canned ALL-BRANDS profile — the
  brand and every inclusion criterion were ignored.
* q15 (HCP-entity cohort ask) returned the BYTE-IDENTICAL payload in 26.4ms:
  both asks collapsed to the same parameterless KPI-call set, so the
  (context-keyed) KPI cache legitimately served identical values. The fix is
  parameter binding — two different asks must produce different parameter sets
  and therefore different payloads.

Covers: brand binding from query text, criteria binding (age applied,
diagnosis-year honestly not-applied), fail-closed when nothing asked can be
served, HCP-entity aggregation with threshold + explicit window, and the
cache-keying identity (two different asks never share a payload).

The KPI calculator and the allowlist-RPC client are faked with plain classes
(the existing suite's idiom — no MagicMock); the real DB path is exercised by
container replay / live verification.
"""

from datetime import date

import pytest

from src.agents.cohort_profiler import CohortProfilerAgent
from src.agents.cohort_profiler.ask import parse_cohort_ask

# Real Remibrutinib NRx numbers (mig-105 breakdown; verified live in PR #1208).
_REMI_NRX = {
    None: 3256.0,
    "low_severity": 855.0,
    "medium_severity": 1752.0,
    "high_severity": 649.0,
    "0": 822.0,
    "1": 825.0,
    "2": 831.0,
    "3": 778.0,
}

_Q11 = (
    "Build a patient cohort for Remibrutinib CSU with inclusion criteria "
    "for adults over 18 diagnosed in 2024"
)
_Q15 = "Build a cohort of high-value HCPs who prescribed more than 50 TRx last quarter"


class _RecordingCalc:
    """Fake KPICalculator that records every context it is called with."""

    def __init__(self, table, brands=("Remibrutinib",)):
        self._table = table
        self._brands = brands
        self.contexts = []

    def calculate(self, kpi_id, context=None):
        context = dict(context or {})
        self.contexts.append(context)
        if context.get("brand") not in self._brands:
            return {"value": None}
        key = context.get("segment") or context.get("therapy_line") or None
        return {"value": self._table.get(key)}


class _FakeRpcResponse:
    def __init__(self, data):
        self.data = data


class _FakeRpcCall:
    def __init__(self, client, data):
        self._client = client
        self._data = data

    def execute(self):
        return _FakeRpcResponse(self._data)


class _FakeDbClient:
    """Fake supabase client for the kpi_query allowlist RPC: returns the queued
    row-lists in order and records every (fn, args) call."""

    def __init__(self, row_batches):
        self._batches = list(row_batches)
        self.calls = []

    def rpc(self, fn, args):
        self.calls.append((fn, dict(args)))
        data = self._batches.pop(0) if self._batches else []
        return _FakeRpcCall(self, data)


def _agent(calc=None, db_rows=None, today=date(2026, 7, 30)):
    agent = CohortProfilerAgent()
    if calc is not None:
        agent._get_calculator = lambda: calc  # type: ignore[method-assign]
    db = _FakeDbClient(db_rows or [])
    agent._get_db_client = lambda: db  # type: ignore[method-assign]
    agent._today = lambda: today  # type: ignore[method-assign]
    return agent, db


# --------------------------------------------------------------------- parsing


def test_parse_binds_brand_from_query_text_and_indication():
    ask = parse_cohort_ask(_Q11, brand_hint=None)
    assert ask.entity_type == "patient"
    assert ask.brand == "Remibrutinib"
    # "CSU" alone must also imply Remibrutinib (indication -> brand).
    ask_csu = parse_cohort_ask("build a cohort of CSU patients", brand_hint=None)
    assert ask_csu.brand == "Remibrutinib"


def test_parse_distinguishes_age_from_trx_threshold():
    # "adults over 18" is an AGE criterion; "more than 50 TRx" is a THRESHOLD —
    # the two "over/more than N" shapes must never cross-match.
    ask11 = parse_cohort_ask(_Q11, brand_hint=None)
    kinds = {c.kind for c in ask11.criteria}
    assert "age_min" in kinds
    assert ask11.threshold is None

    ask15 = parse_cohort_ask(_Q15, brand_hint=None, today=date(2026, 7, 30))
    assert ask15.entity_type == "hcp"
    assert ask15.threshold is not None
    assert ask15.threshold.metric == "trx"
    assert ask15.threshold.min_exclusive == 50
    assert not any(c.kind == "age_min" for c in ask15.criteria)


def test_parse_marks_diagnosis_year_unservable():
    ask = parse_cohort_ask(_Q11, brand_hint=None)
    diag = [c for c in ask.criteria if c.kind == "diagnosis_year"]
    assert diag, "diagnosed-in-2024 must be recognized as a criterion"
    assert not diag[0].servable
    assert diag[0].guidance  # names WHY it cannot be served


def test_parse_last_quarter_window_is_explicit_dates():
    ask = parse_cohort_ask(_Q15, brand_hint=None, today=date(2026, 7, 30))
    assert ask.window is not None
    assert ask.window.start == date(2026, 4, 1)
    assert ask.window.end == date(2026, 7, 1)  # exclusive


# ------------------------------------------------- patient path: q11 behaviors


@pytest.mark.asyncio
async def test_q11_binds_brand_and_criteria_not_all_brands():
    """q11 must never again return the canned all-brands profile: the brand
    binds from the query text and the servable criteria flow into the
    criteria-bound profile query."""
    rows = [
        # (severity, therapy_line, nrx) grouped rows from the criteria query
        [
            {"severity": "low_severity", "therapy_line": 0, "nrx": 200},
            {"severity": "medium_severity", "therapy_line": 1, "nrx": 500},
            {"severity": "high_severity", "therapy_line": 3, "nrx": 100},
        ]
    ]
    agent, db = _agent(calc=_RecordingCalc(_REMI_NRX), db_rows=rows)
    out = await agent.analyze({"query": _Q11})

    assert out["status"] == "completed"
    narrative = out["narrative"]
    assert "all brands" not in narrative.lower()
    assert "Remibrutinib" in narrative

    # The criteria-bound allowlist query ran with the bound parameters:
    # brand=Remibrutinib, min-age-exclusive=18, no max age.
    assert db.calls, "criteria-bound profile must go through the kpi_query RPC"
    fn, args = db.calls[0]
    assert fn == "kpi_query"
    assert args["params"][0] == "Remibrutinib"
    assert args["params"][1] == 18

    # Honest criteria accounting: exactly which applied, which could not be.
    assert "diagnosed in 2024" in narrative
    assert "not applied" in narrative.lower() or "could not" in narrative.lower()
    assert "18" in narrative


@pytest.mark.asyncio
async def test_q11_result_reports_applied_and_unserved_criteria_structured():
    rows = [[{"severity": "low_severity", "therapy_line": 0, "nrx": 42}]]
    agent, _db = _agent(calc=_RecordingCalc(_REMI_NRX), db_rows=rows)
    out = await agent.analyze({"query": _Q11})
    profile = out["cohort_profile"]
    assert profile["criteria_applied"], "applied criteria must be enumerated"
    assert profile["criteria_not_applied"], "unserved criteria must be enumerated"


@pytest.mark.asyncio
async def test_unservable_only_criteria_fail_closed_with_guidance():
    """If the ONLY thing the ask pinned down cannot be served (no brand, no
    servable criterion), the agent must NOT answer a different question with a
    canned all-brands profile — it fails closed with guidance."""
    agent, db = _agent(calc=_RecordingCalc(_REMI_NRX))
    out = await agent.analyze({"query": "Build a patient cohort of patients diagnosed in 2023"})
    assert out["status"] == "failed"
    assert out["errors"]
    joined = " ".join(e.get("error", "") for e in out["errors"])
    assert "diagnos" in joined.lower()
    assert not db.calls


@pytest.mark.asyncio
async def test_plain_ask_keeps_existing_kpi_path_untouched():
    """A criteria-less ask (the pre-#1356 contract) must keep the exact
    KPI-calculator path so numbers stay in lock-step with the live chat UI."""
    calc = _RecordingCalc(_REMI_NRX)
    agent, db = _agent(calc=calc)
    out = await agent.analyze({"brand": "Remibrutinib", "query": "build a cohort"})
    assert out["status"] == "completed"
    assert "855" in out["narrative"]
    assert not db.calls  # no RPC — the mig-105 calculator path served it
    assert all(c.get("brand") == "Remibrutinib" for c in calc.contexts)


# ------------------------------------------------------ HCP path: q15 behaviors


_HCP_ROWS = [
    {"specialty": "oncology", "priority_tier": 1, "n_hcps": 3, "total_trx": 190, "max_trx": 70},
    {"specialty": "dermatology", "priority_tier": 2, "n_hcps": 2, "total_trx": 110, "max_trx": 56},
    {"specialty": "oncology", "priority_tier": 2, "n_hcps": 1, "total_trx": 51, "max_trx": 51},
]


@pytest.mark.asyncio
async def test_q15_hcp_threshold_cohort_with_explicit_window():
    agent, db = _agent(db_rows=[_HCP_ROWS], today=date(2026, 7, 30))
    out = await agent.analyze({"query": _Q15})

    assert out["status"] == "completed"
    profile = out["cohort_profile"]
    assert profile["entity"] == "hcp"
    assert profile["cohort_size"] == 6
    assert profile["specialty"]["oncology"] == 4
    assert profile["priority_tier"]["2"] == 3

    # The allowlist RPC carried ALL the ask's parameters.
    fn, args = db.calls[0]
    assert fn == "kpi_query"
    assert args["params"] == [None, "2026-04-01", "2026-07-01", 50]

    narrative = out["narrative"]
    assert "HCP" in narrative
    assert "50" in narrative
    assert "2026-04-01" in narrative and "2026-06-30" in narrative
    # Mirrors the patient-profile shape: headline + segment breakdown tables.
    assert "oncology" in narrative


@pytest.mark.asyncio
async def test_hcp_zero_match_is_honest_answer_not_failure():
    """0 HCPs over the threshold with a NONZERO prescribing base is a real
    answer (the threshold filtered everyone out), not a data failure."""
    base_rows = [
        {
            "specialty": "oncology",
            "priority_tier": 1,
            "n_hcps": 545,
            "total_trx": 2318,
            "max_trx": 16,
        }
    ]
    agent, db = _agent(db_rows=[[], base_rows], today=date(2026, 7, 30))
    out = await agent.analyze({"query": _Q15})
    assert out["status"] == "completed"
    assert out["cohort_profile"]["cohort_size"] == 0
    narrative = out["narrative"]
    assert "0" in narrative and "50" in narrative
    # The empty-vs-zero distinction required a threshold-free base probe.
    assert len(db.calls) == 2
    assert db.calls[1][1]["params"][3] == 0


@pytest.mark.asyncio
async def test_hcp_genuine_empty_fails_closed():
    agent, _db = _agent(db_rows=[[], []], today=date(2026, 7, 30))
    out = await agent.analyze({"query": _Q15})
    assert out["status"] == "failed"
    assert out["errors"]


@pytest.mark.asyncio
async def test_hcp_without_window_disclosed_default():
    agent, db = _agent(db_rows=[_HCP_ROWS], today=date(2026, 7, 30))
    out = await agent.analyze({"query": "cohort of HCPs with more than 10 TRx"})
    assert out["status"] == "completed"
    _fn, args = db.calls[0]
    # Trailing 90 days INCLUDING today (exactly 90 dates in [start, end)).
    assert args["params"][1] == "2026-05-02"
    assert args["params"][2] == "2026-07-31"
    assert args["params"][3] == 10
    assert "90" in out["narrative"] or "2026-05-02" in out["narrative"]


# -------------------------------------------------------- cache-keying identity


@pytest.mark.asyncio
async def test_two_different_asks_never_share_a_payload():
    """q11 and q15 returned byte-identical payloads pre-#1356 because neither
    ask's parameters reached the data layer. Post-fix the parameter sets — and
    therefore the payloads — must differ."""
    rows_q11 = [[{"severity": "low_severity", "therapy_line": 0, "nrx": 42}]]
    agent1, db1 = _agent(calc=_RecordingCalc(_REMI_NRX), db_rows=rows_q11)
    out11 = await agent1.analyze({"query": _Q11})

    agent2, db2 = _agent(db_rows=[_HCP_ROWS], today=date(2026, 7, 30))
    out15 = await agent2.analyze({"query": _Q15})

    assert out11 != out15
    assert out11["narrative"] != out15["narrative"]
    # And the underlying data-layer parameter sets differed too.
    assert db1.calls[0][1] != db2.calls[0][1]


def test_kpi_cache_key_distinguishes_bound_parameters():
    """The Redis KPI cache that served the 26.4ms byte-identical repeat keys on
    the calculation context. Pin that every parameter the agent now binds
    (brand / window / thresholds via distinct query params) yields a distinct
    key, so two different asks can never collide."""
    from src.kpi.cache import KPICache

    cache = KPICache.__new__(KPICache)  # no Redis connection needed for keying
    k_all = cache._make_key("WS3-BI-006")
    k_remi = cache._make_key("WS3-BI-006", brand="Remibrutinib")
    k_remi_seg = cache._make_key("WS3-BI-006", brand="Remibrutinib", segment="low_severity")
    assert len({k_all, k_remi, k_remi_seg}) == 3


# ------------------------------------------- codex iter-1 findings (red-first)


@pytest.mark.asyncio
async def test_patient_threshold_is_disclosed_not_silently_dropped():
    """Finding 1 (HIGH): a KPI threshold on a PATIENT-entity ask must never be
    silently ignored — 'size the Remibrutinib cohort' and the same ask + '>50
    TRx' are materially different questions and must not share a payload. The
    threshold is unservable on the patient path today, so it must appear in the
    NOT-applied accounting (narrative + structured)."""
    calc = _RecordingCalc(_REMI_NRX)
    agent, _db = _agent(calc=calc)
    out = await agent.analyze(
        {"query": "Size the Remibrutinib cohort of patients with more than 50 TRx"}
    )
    assert out["status"] == "completed"  # brand still binds and is servable
    labels = " ".join(c["label"] for c in out["cohort_profile"]["criteria_not_applied"])
    assert "50" in labels and "trx" in labels.lower()
    narrative = out["narrative"].lower()
    assert "not applied" in narrative or "could not" in narrative
    assert "hcp" in narrative  # guidance points at the HCP-cohort form


@pytest.mark.asyncio
async def test_patient_threshold_only_ask_fails_closed():
    """Finding 1 (HIGH), fail-closed leg: when the threshold is the ONLY thing
    the patient ask pinned down, answering the unthresholded question would be
    answering a different question — fail closed with guidance."""
    agent, db = _agent(calc=_RecordingCalc(_REMI_NRX))
    out = await agent.analyze({"query": "Build a cohort of patients with more than 50 TRx"})
    assert out["status"] == "failed"
    joined = " ".join(e.get("error", "") for e in out["errors"]).lower()
    assert "trx" in joined and "hcp" in joined
    assert not db.calls


@pytest.mark.asyncio
async def test_hcp_recognized_criteria_disclosed_not_dropped():
    """Finding 2 (HIGH): recognized criteria on an HCP ask (age /
    diagnosis-year — patient-journey attributes, unservable on the HCP path)
    must surface in the NOT-applied accounting, not vanish."""
    agent, _db = _agent(db_rows=[_HCP_ROWS], today=date(2026, 7, 30))
    out = await agent.analyze(
        {
            "query": (
                "Build a cohort of HCPs who prescribed more than 50 TRx last "
                "quarter treating adults over 18 diagnosed in 2024"
            )
        }
    )
    assert out["status"] == "completed"  # threshold + window still bind
    not_applied = out["cohort_profile"]["criteria_not_applied"]
    labels = " ".join(c["label"] for c in not_applied).lower()
    assert "over 18" in labels
    assert "diagnosed in 2024" in labels
    narrative = out["narrative"].lower()
    assert "not applied" in narrative or "could not" in narrative
    assert "diagnosed in 2024" in narrative


@pytest.mark.asyncio
async def test_hcp_unservable_only_criteria_fail_closed():
    """Finding 2 (HIGH), fail-closed leg: an HCP ask whose ONLY specifics are
    unservable criteria (no threshold, no explicit window, no brand) must fail
    closed with guidance, not profile all prescribing HCPs."""
    agent, db = _agent(db_rows=[_HCP_ROWS], today=date(2026, 7, 30))
    out = await agent.analyze({"query": "Build a cohort of HCPs diagnosed in 2024"})
    assert out["status"] == "failed"
    joined = " ".join(e.get("error", "") for e in out["errors"]).lower()
    assert "diagnos" in joined
    assert not db.calls


def test_last_n_days_window_spans_exactly_n_days():
    """Finding 3 (MEDIUM): 'last N days' must cover exactly N inclusive dates
    (inclusive-today semantics: [today-(N-1), today+1))."""
    ask = parse_cohort_ask(
        "cohort of HCPs with more than 10 TRx in the last 90 days",
        brand_hint=None,
        today=date(2026, 7, 30),
    )
    assert ask.window is not None
    assert (ask.window.end - ask.window.start).days == 90
    assert ask.window.start == date(2026, 5, 2)
    assert ask.window.end == date(2026, 7, 31)  # exclusive; today included
