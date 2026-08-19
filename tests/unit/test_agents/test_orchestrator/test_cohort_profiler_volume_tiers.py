"""Tests for the #1736 cohort_profiler volume-tier extension.

Eval turn 4.3 (post1708 AND post1730 runs) promised, after brand selection,
"counts per tier" for an HCP segmentation "into high/medium/low
prescription-volume tiers" — but the profiler supported only a single
``min_exclusive`` TRx threshold, so the promised per-tier breakdown was
undeliverable on the offered path (the post-#1696 undeliverable-promise shape,
recurred in place across two runs). The user ruling is (b): make the deliverable
REAL — extend cohort_profiler to bucket HCPs into volume tiers and return real
counts per tier.

Tier definition (data-grounded, measured READ-ONLY against the local prod
Supabase 2026-08-19): the data model carries NO stored volume tier —
``hcp_profiles.prescribing_tier`` and ``prescribing_volume`` are NULL for all
5,000 rows, and ``priority_tier`` (populated 1-5) is a DISTINCT targeting
concept (ontology: volume + brand affinity + accessibility) that must not be
conflated with a volume axis. Volume tiers are therefore COMPUTED from the same
per-HCP TRx substrate as the existing #1356 HCP cohort path (treatment_events
prescription rows, half-open window, lock-step with the platform TRx KPI):
value-based terciles — cut points ``percentile_disc(1/3)`` / ``(2/3)`` of the
per-HCP TRx distribution WITHIN the queried scope (brand/window/threshold/
region), ties assigned by value so equal TRx always shares a tier, and the
measured cut points disclosed in narrative and payload (never presented as
fixed global constants).

Every pinned row/count below is a REAL measurement (the exact SQL lives in the
mig-130 statement; measurements taken 2026-08-19 against the live substrate,
which is append-only via the Monday frontier cron — the stub returns those
measured rows and the tests ALSO assert the allowlist RPC was invoked with the
exact expected arguments, so nothing here is a plausible-fake value and the
query wiring cannot silently drift).

Fakes mirror tests/unit/test_agents/test_orchestrator/test_cohort_profiler_extend.py
(plain classes, no MagicMock); the real DB path is exercised by container
replay / live verification.
"""

from datetime import date

import pytest

from src.agents.cohort_profiler import CohortProfilerAgent
from src.agents.cohort_profiler.ask import merge_cohort_asks, parse_cohort_ask

# The eval 4.3 question, verbatim (scripts/demos/copilot_demo_questions.json).
_Q43 = "Segment HCPs by prescription volume into high, medium, and low tiers"
# The promised follow-up shape once a brand is picked (grades_n3n4.json 4.3).
_Q43_PROMISE = "an all-brands segmentation into high/medium/low prescription-volume tiers"
_Q43_BRANDED = (
    "For Remibrutinib, segment HCPs by prescription volume into high, "
    "medium and low tiers last quarter"
)
_Q15 = "Build a cohort of high-value HCPs who prescribed more than 50 TRx last quarter"


def _rows(raw):
    """(tier, specialty, c1, c2, n_hcps, total_trx, min_trx, max_trx) -> RPC rows."""
    return [
        {
            "volume_tier": t,
            "specialty": s,
            "cut_low_max": c1,
            "cut_medium_max": c2,
            "n_hcps": n,
            "total_trx": tot,
            "min_trx": lo,
            "max_trx": hi,
        }
        for (t, s, c1, c2, n, tot, lo, hi) in raw
    ]


# All brands, default 90-day window [2026-05-22, 2026-08-20) (today=2026-08-19),
# TRx floor 0. Measured 2026-08-19: 3,427 HCPs; tercile cuts 7 / 12;
# low(<=7)=1,266  medium(8-12)=1,075  high(>=13)=1,086.
_ALL_BRANDS_90D = _rows(
    [
        ("high", "oncology", 7, 12, 343, 6580, 13, 44),
        ("high", "hematology", 7, 12, 218, 4089, 13, 38),
        ("high", "dermatology", 7, 12, 195, 3630, 13, 35),
        ("high", "allergy_immunology", 7, 12, 133, 2529, 13, 45),
        ("high", "internal_medicine", 7, 12, 111, 2053, 13, 63),
        ("high", "rheumatology", 7, 12, 55, 1032, 13, 34),
        ("high", "neurology", 7, 12, 31, 535, 13, 28),
        ("low", "oncology", 7, 12, 411, 1923, 1, 7),
        ("low", "hematology", 7, 12, 248, 1207, 1, 7),
        ("low", "dermatology", 7, 12, 214, 1059, 1, 7),
        ("low", "allergy_immunology", 7, 12, 166, 777, 1, 7),
        ("low", "internal_medicine", 7, 12, 124, 547, 1, 7),
        ("low", "rheumatology", 7, 12, 62, 311, 1, 7),
        ("low", "neurology", 7, 12, 41, 203, 2, 7),
        ("medium", "oncology", 7, 12, 380, 3744, 8, 12),
        ("medium", "hematology", 7, 12, 217, 2095, 8, 12),
        ("medium", "dermatology", 7, 12, 172, 1696, 8, 12),
        ("medium", "internal_medicine", 7, 12, 108, 1076, 8, 12),
        ("medium", "allergy_immunology", 7, 12, 101, 970, 8, 12),
        ("medium", "rheumatology", 7, 12, 49, 482, 8, 12),
        ("medium", "neurology", 7, 12, 48, 453, 8, 12),
    ]
)

# Remibrutinib, last quarter [2026-04-01, 2026-07-01), TRx floor 0. Measured
# 2026-08-19: 192 HCPs; cuts 2 / 5; low=74 medium=61 high=57;
# per-tier TRx 94 / 250 / 462.
_REMI_LQ = _rows(
    [
        ("high", "oncology", 2, 5, 23, 182, 6, 14),
        ("high", "dermatology", 2, 5, 10, 89, 6, 12),
        ("high", "internal_medicine", 2, 5, 7, 64, 7, 12),
        ("high", "rheumatology", 2, 5, 6, 45, 6, 9),
        ("high", "hematology", 2, 5, 6, 46, 6, 11),
        ("high", "allergy_immunology", 2, 5, 5, 36, 6, 9),
        ("low", "oncology", 2, 5, 25, 32, 1, 2),
        ("low", "hematology", 2, 5, 17, 22, 1, 2),
        ("low", "dermatology", 2, 5, 10, 13, 1, 2),
        ("low", "internal_medicine", 2, 5, 8, 11, 1, 2),
        ("low", "allergy_immunology", 2, 5, 5, 6, 1, 2),
        ("low", "rheumatology", 2, 5, 5, 6, 1, 2),
        ("low", "neurology", 2, 5, 4, 4, 1, 1),
        ("medium", "oncology", 2, 5, 17, 70, 3, 5),
        ("medium", "hematology", 2, 5, 13, 56, 3, 5),
        ("medium", "dermatology", 2, 5, 10, 40, 3, 5),
        ("medium", "allergy_immunology", 2, 5, 10, 38, 3, 5),
        ("medium", "internal_medicine", 2, 5, 9, 37, 3, 5),
        ("medium", "neurology", 2, 5, 1, 5, 5, 5),
        ("medium", "rheumatology", 2, 5, 1, 4, 4, 4),
    ]
)

# All brands, last quarter, TRx floor 2 (threshold composes with tiers).
# Measured 2026-08-19: 326 HCPs; cuts 5 / 7; low=155 medium=76 high=95.
_THR2_LQ = _rows(
    [
        ("high", "oncology", 5, 7, 32, 310, 8, 16),
        ("high", "hematology", 5, 7, 17, 167, 8, 16),
        ("high", "dermatology", 5, 7, 17, 170, 8, 12),
        ("high", "internal_medicine", 5, 7, 11, 114, 8, 12),
        ("high", "allergy_immunology", 5, 7, 9, 85, 8, 11),
        ("high", "rheumatology", 5, 7, 7, 64, 8, 13),
        ("high", "neurology", 5, 7, 2, 16, 8, 8),
        ("low", "oncology", 5, 7, 53, 209, 3, 5),
        ("low", "hematology", 5, 7, 31, 128, 3, 5),
        ("low", "dermatology", 5, 7, 25, 95, 3, 5),
        ("low", "allergy_immunology", 5, 7, 21, 78, 3, 5),
        ("low", "internal_medicine", 5, 7, 17, 69, 3, 5),
        ("low", "rheumatology", 5, 7, 4, 14, 3, 4),
        ("low", "neurology", 5, 7, 4, 17, 3, 5),
        ("medium", "oncology", 5, 7, 25, 161, 6, 7),
        ("medium", "hematology", 5, 7, 16, 105, 6, 7),
        ("medium", "allergy_immunology", 5, 7, 11, 72, 6, 7),
        ("medium", "dermatology", 5, 7, 11, 70, 6, 7),
        ("medium", "internal_medicine", 5, 7, 7, 47, 6, 7),
        ("medium", "rheumatology", 5, 7, 4, 26, 6, 7),
        ("medium", "neurology", 5, 7, 2, 13, 6, 7),
    ]
)

# All brands, last quarter, northeast region, TRx floor 0. Measured 2026-08-19:
# 116 HCPs; WITHIN-SCOPE cuts 1 / 5 (vs 2 / 5 unscoped — the cuts MUST be
# terciles of the region-scoped cohort, not the global one); low=44 medium=37
# high=35.
_NE_LQ = _rows(
    [
        ("high", "oncology", 1, 5, 13, 110, 6, 14),
        ("high", "hematology", 1, 5, 8, 75, 7, 11),
        ("high", "dermatology", 1, 5, 5, 38, 6, 10),
        ("high", "internal_medicine", 1, 5, 4, 36, 7, 11),
        ("high", "allergy_immunology", 1, 5, 4, 30, 6, 9),
        ("high", "neurology", 1, 5, 1, 8, 8, 8),
        ("low", "oncology", 1, 5, 19, 19, 1, 1),
        ("low", "hematology", 1, 5, 9, 9, 1, 1),
        ("low", "internal_medicine", 1, 5, 5, 5, 1, 1),
        ("low", "allergy_immunology", 1, 5, 4, 4, 1, 1),
        ("low", "dermatology", 1, 5, 3, 3, 1, 1),
        ("low", "rheumatology", 1, 5, 2, 2, 1, 1),
        ("low", "neurology", 1, 5, 2, 2, 1, 1),
        ("medium", "oncology", 1, 5, 11, 35, 2, 5),
        ("medium", "dermatology", 1, 5, 8, 26, 2, 5),
        ("medium", "internal_medicine", 1, 5, 7, 19, 2, 5),
        ("medium", "hematology", 1, 5, 5, 21, 3, 5),
        ("medium", "allergy_immunology", 1, 5, 4, 13, 2, 5),
        ("medium", "rheumatology", 1, 5, 1, 3, 3, 3),
        ("medium", "neurology", 1, 5, 1, 4, 4, 4),
    ]
)

# Real Remibrutinib NRx numbers (mig-105 breakdown; verified live in PR #1208) —
# used only by the patient-path accounting test.
_KISQALI_NRX = {None: 3256.0}


class _RecordingCalc:
    def __init__(self, table, brands=("Kisqali",)):
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
    def __init__(self, data):
        self._data = data

    def execute(self):
        return _FakeRpcResponse(self._data)


class _FakeDbClient:
    """Returns queued row-lists in order and records every (fn, args) call."""

    def __init__(self, row_batches):
        self._batches = list(row_batches)
        self.calls = []

    def rpc(self, fn, args):
        self.calls.append((fn, dict(args)))
        data = self._batches.pop(0) if self._batches else []
        return _FakeRpcCall(data)


def _agent(calc=None, db_rows=None, today=date(2026, 8, 19)):
    agent = CohortProfilerAgent()
    if calc is not None:
        agent._get_calculator = lambda: calc  # type: ignore[method-assign]
    db = _FakeDbClient(db_rows or [])
    agent._get_db_client = lambda: db  # type: ignore[method-assign]
    agent._today = lambda: today  # type: ignore[method-assign]
    return agent, db


# --------------------------------------------------------------------- parsing


def test_parse_recognizes_the_eval_43_ask_verbatim():
    ask = parse_cohort_ask(_Q43)
    assert ask.entity_type == "hcp"
    assert ask.volume_tiers is True
    assert ask.brand is None
    assert ask.threshold is None


def test_parse_recognizes_promise_phrasing_and_branded_followup():
    assert parse_cohort_ask(_Q43_PROMISE).volume_tiers is True
    ask = parse_cohort_ask(_Q43_BRANDED)
    assert ask.volume_tiers is True
    assert ask.brand == "Remibrutinib"
    assert ask.window is not None and ask.window.label == "last quarter"


def test_parse_volume_tiers_defaults_entity_to_hcp():
    # Prescription-volume tiers are a per-PRESCRIBER axis: with no entity word
    # in the ask, the tier ask must land on the HCP path (the promised 4.3
    # follow-up often names only the brand + the tier phrasing).
    ask = parse_cohort_ask("a Kisqali segmentation into high/medium/low prescription-volume tiers")
    assert ask.entity_type == "hcp"
    assert ask.volume_tiers is True
    # ... but an explicit patient ask stays a patient ask (served honestly there).
    ask_p = parse_cohort_ask(
        "Segment Kisqali patients by prescription volume into high, medium and low tiers"
    )
    assert ask_p.entity_type == "patient"
    assert ask_p.volume_tiers is True


def test_parse_negatives_do_not_overtrigger():
    # The existing single-threshold ask is NOT a tier ask.
    assert parse_cohort_ask(_Q15).volume_tiers is False
    # priority_tier is a DIFFERENT concept (targeting priority, hcp_profiles
    # attribute) — "priority tier" phrasing must never trigger volume tiers.
    assert parse_cohort_ask("cohort of high priority tier HCPs").volume_tiers is False
    assert parse_cohort_ask("build a cohort of CSU patients").volume_tiers is False


def test_merge_carries_volume_tiers_from_raw_user_query():
    primary = parse_cohort_ask("profile HCPs for Remibrutinib")
    supplement = parse_cohort_ask(_Q43)
    merged = merge_cohort_asks(primary, supplement)
    assert merged.volume_tiers is True
    assert merged.brand == "Remibrutinib"


# ------------------------------------------------------------- HCP tier path


@pytest.mark.asyncio
async def test_43_ask_serves_counts_per_tier_all_brands():
    """The eval 4.3 ask, verbatim, must be servable in ONE call: real counts
    per high/medium/low tier with the measured cut points disclosed."""
    agent, db = _agent(db_rows=[_ALL_BRANDS_90D], today=date(2026, 8, 19))
    out = await agent.analyze({"query": _Q43})

    assert out["status"] == "completed"
    profile = out["cohort_profile"]
    assert profile["entity"] == "hcp"
    assert profile["segment_axis"] == "volume_tier+specialty"
    assert profile["cohort_size"] == 3427
    tiers = profile["volume_tiers"]
    assert tiers["low"]["n_hcps"] == 1266
    assert tiers["medium"]["n_hcps"] == 1075
    assert tiers["high"]["n_hcps"] == 1086
    assert tiers["high"]["trx_min"] == 13 and tiers["high"]["trx_max"] == 63
    bounds = profile["tier_boundaries"]
    assert bounds["low_max_trx"] == 7
    assert bounds["medium_max_trx"] == 12
    assert "tercile" in bounds["method"]
    # Specialty rides along ("plus specialty where available").
    assert profile["specialty"]["oncology"] == 343 + 411 + 380

    # The allowlist RPC executed with the exact expected arguments (default
    # 90-day window incl. today, no brand, TRx floor 0, tier statement id).
    assert len(db.calls) == 1
    fn, args = db.calls[0]
    assert fn == "kpi_query"
    assert args["query_id"] == "cohort_profiler_hcp_volume_tiers"
    assert args["params"] == [None, "2026-05-22", "2026-08-20", 0]

    narrative = out["narrative"]
    assert "1,266" in narrative and "1,075" in narrative and "1,086" in narrative
    # Cut points are disclosed as measured, scope-relative values.
    assert "tercile" in narrative.lower()
    assert "7" in narrative and "12" in narrative


@pytest.mark.asyncio
async def test_branded_last_quarter_counts_per_tier():
    """The promised post-brand-pick follow-up: brand-scoped tier counts."""
    agent, db = _agent(db_rows=[_REMI_LQ], today=date(2026, 7, 30))
    out = await agent.analyze({"query": _Q43_BRANDED})

    assert out["status"] == "completed"
    profile = out["cohort_profile"]
    assert profile["brand"] == "Remibrutinib"
    assert profile["cohort_size"] == 192
    tiers = profile["volume_tiers"]
    assert tiers["low"]["n_hcps"] == 74
    assert tiers["medium"]["n_hcps"] == 61
    assert tiers["high"]["n_hcps"] == 57
    assert tiers["low"]["trx_total"] == 94
    assert tiers["medium"]["trx_total"] == 250
    assert tiers["high"]["trx_total"] == 462
    assert profile["tier_boundaries"]["low_max_trx"] == 2
    assert profile["tier_boundaries"]["medium_max_trx"] == 5
    assert profile["specialty"]["oncology"] == 65

    fn, args = db.calls[0]
    assert fn == "kpi_query"
    assert args["query_id"] == "cohort_profiler_hcp_volume_tiers"
    assert args["params"] == ["Remibrutinib", "2026-04-01", "2026-07-01", 0]


@pytest.mark.asyncio
async def test_threshold_composes_with_tiers():
    agent, db = _agent(db_rows=[_THR2_LQ], today=date(2026, 7, 30))
    out = await agent.analyze(
        {
            "query": (
                "Segment HCPs with more than 2 TRx last quarter by "
                "prescription volume into high, medium and low tiers"
            )
        }
    )
    assert out["status"] == "completed"
    profile = out["cohort_profile"]
    assert profile["cohort_size"] == 326
    assert profile["volume_tiers"]["low"]["n_hcps"] == 155
    assert profile["volume_tiers"]["medium"]["n_hcps"] == 76
    assert profile["volume_tiers"]["high"]["n_hcps"] == 95
    assert profile["threshold"]["min_exclusive"] == 2
    # Cuts are terciles of the THRESHOLDED cohort (5/7, not the unfiltered 2/5).
    assert profile["tier_boundaries"]["low_max_trx"] == 5
    assert profile["tier_boundaries"]["medium_max_trx"] == 7
    _fn, args = db.calls[0]
    assert args["params"] == [None, "2026-04-01", "2026-07-01", 2]


@pytest.mark.asyncio
async def test_region_ask_binds_region_tier_statement():
    agent, db = _agent(db_rows=[_NE_LQ], today=date(2026, 7, 30))
    out = await agent.analyze(
        {
            "query": (
                "Segment HCPs in the northeast by prescription volume "
                "into high, medium and low tiers last quarter"
            )
        }
    )
    assert out["status"] == "completed"
    profile = out["cohort_profile"]
    assert profile["region"] == "northeast"
    assert profile["region_applied"] is True
    assert profile["cohort_size"] == 116
    assert profile["volume_tiers"]["low"]["n_hcps"] == 44
    assert profile["volume_tiers"]["medium"]["n_hcps"] == 37
    assert profile["volume_tiers"]["high"]["n_hcps"] == 35
    # Within-scope terciles: the northeast cohort's own cuts (1/5), not the
    # global cohort's (2/5).
    assert profile["tier_boundaries"]["low_max_trx"] == 1
    assert profile["tier_boundaries"]["medium_max_trx"] == 5
    _fn, args = db.calls[0]
    assert args["query_id"] == "cohort_profiler_hcp_volume_tiers_region"
    assert args["params"] == [None, "2026-04-01", "2026-07-01", 0, "northeast"]


@pytest.mark.asyncio
async def test_tier_genuine_empty_fails_closed():
    agent, db = _agent(db_rows=[[]], today=date(2026, 7, 30))
    out = await agent.analyze({"query": _Q43})
    assert out["status"] == "failed"
    assert out["errors"]
    # No threshold in the ask -> no zero-vs-empty probe needed: one call only,
    # and it must be the TIER statement (not the single-threshold base).
    assert len(db.calls) == 1
    assert db.calls[0][1]["query_id"] == "cohort_profiler_hcp_volume_tiers"


@pytest.mark.asyncio
async def test_tier_zero_match_with_threshold_probes_base():
    """Threshold + tiers with zero matches over a NONZERO base is an honest
    zero (the threshold filtered everyone out), not a data failure."""
    base = _rows([("low", "oncology", 1, 3, 545, 2318, 1, 16)])
    agent, db = _agent(db_rows=[[], base], today=date(2026, 7, 30))
    out = await agent.analyze(
        {
            "query": (
                "Segment HCPs with more than 100 TRx last quarter by prescription volume into tiers"
            )
        }
    )
    assert out["status"] == "completed"
    assert out["cohort_profile"]["cohort_size"] == 0
    assert len(db.calls) == 2
    assert all(a["query_id"] == "cohort_profiler_hcp_volume_tiers" for _f, a in db.calls)
    assert db.calls[0][1]["params"][3] == 100
    assert db.calls[1][1]["params"][3] == 0
    assert "545" in out["narrative"]


# ----------------------------------------------------------- patient honesty


@pytest.mark.asyncio
async def test_patient_volume_tier_ask_is_accounted_not_fabricated():
    """An explicit PATIENT ask for volume tiers cannot be served (per-HCP TRx
    is a prescriber axis): the profile still serves, with the tier request in
    the honest criteria_not_applied accounting — never silently dropped, never
    fabricated."""
    calc = _RecordingCalc(_KISQALI_NRX)
    agent, db = _agent(calc=calc)
    out = await agent.analyze(
        {
            "query": (
                "Segment Kisqali patients by prescription volume into high, medium and low tiers"
            )
        }
    )
    assert out["status"] == "completed"
    assert "volume_tiers" not in out["cohort_profile"]
    not_applied = out["cohort_profile"]["criteria_not_applied"]
    assert any("volume" in (c["label"] or "").lower() for c in not_applied)
    assert any("HCP" in (c["guidance"] or "") for c in not_applied)
    assert not db.calls


@pytest.mark.asyncio
async def test_patient_volume_tier_only_ask_fails_closed_with_guidance():
    agent, db = _agent(calc=_RecordingCalc(_KISQALI_NRX))
    out = await agent.analyze(
        {"query": "Segment patients by prescription volume into high, medium and low tiers"}
    )
    assert out["status"] == "failed"
    joined = " ".join(e.get("error", "") for e in out["errors"])
    assert "HCP" in joined
    assert not db.calls


# ------------------------------------------- existing single-threshold intact


@pytest.mark.asyncio
async def test_plain_threshold_ask_keeps_base_statement():
    """The pre-#1736 single-threshold contract must be untouched: a q15-style
    ask still runs the mig-117 base statement, not the tier statement."""
    rows = [
        {
            "specialty": "oncology",
            "priority_tier": 1,
            "n_hcps": 3,
            "total_trx": 190,
            "max_trx": 70,
        }
    ]
    agent, db = _agent(db_rows=[rows], today=date(2026, 7, 30))
    out = await agent.analyze({"query": _Q15})
    assert out["status"] == "completed"
    _fn, args = db.calls[0]
    assert args["query_id"] == "cohort_profiler_hcp_trx_cohort"
    assert args["params"] == [None, "2026-04-01", "2026-07-01", 50]
    assert "volume_tiers" not in out["cohort_profile"]
