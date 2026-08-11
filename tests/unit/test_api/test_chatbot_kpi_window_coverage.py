"""Window-coverage honesty for ``kpi_calculate_tool`` (2026-07-07 session review).

Failure mode: the synthetic-showcase substrate is dense only in the most recent
~30 days (Apr 397 / May 394 / Jun 8,185 / Jul-partial 14,577 events), so a
"90-day baseline" (15,767) was 96% composed of the same 30 days it was compared
against (15,239) — the chatbot concluded "3.4% softening" from a meaningless
overlap. The tool honestly applied the window but nothing disclosed that data
coverage inside it was wildly asymmetric.

Contract: for CUMULATIVE volume KPIs (TRx/NRx/NBRx — counts, not shares/rates)
with an applied window longer than 45 days, the tool also computes the KPI over
the trailing 30 days of that window and attaches ``window_coverage``:

    {"window_days", "trailing_30d_value", "trailing_30d_share",
     "uniform_expected_share", "coverage_warning"?}

``coverage_warning`` appears only when the trailing-30d share exceeds 2× the
uniform expectation — i.e., the window figure is dominated by its most recent
30 days and must not be treated as a baseline.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.api.routes.chatbot_tools import _window_coverage_probe, kpi_calculate_tool
from src.kpi.models import KPIResult, KPIStatus
from src.services.time_window import Window

NINETY_DAY_WINDOW = "2026-04-08 to 2026-07-07"


def _result(kpi_id: str, value: float) -> KPIResult:
    # Faithful to the real engine: KPICalculator._stamp_window marks a
    # requested window on a windowable KPI as "applied".
    return KPIResult(
        kpi_id=kpi_id,
        value=value,
        status=KPIStatus.UNKNOWN,
        metadata={"include_synthetic": True},
        window_status="applied",
    )


def _calculator(window_value: float, trailing_value: float, kpi_id: str = "WS3-BI-005"):
    """Calculator double: full-window value on the first window, trailing-30d on the sub-window."""
    calc = MagicMock()
    calls: list[dict] = []

    def calculate(kpi_id_arg, context=None, **_):
        window = (context or {}).get("window") or {}
        calls.append(dict(window))
        start = str(window.get("start", ""))
        # The trailing sub-window starts 30 days before the requested end.
        value = trailing_value if start.startswith("2026-06-07") else window_value
        return _result(kpi_id_arg, value)

    calc.calculate = MagicMock(side_effect=calculate)
    calc._calls = calls
    return calc


@pytest.mark.unit
@pytest.mark.asyncio
async def test_asymmetric_window_gets_coverage_warning():
    calc = _calculator(window_value=15767.0, trailing_value=15239.0)
    with patch("src.api.routes.kpi.get_kpi_calculator", return_value=calc):
        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Fabhalta", "window": NINETY_DAY_WINDOW}
        )
    assert resp["success"] is True
    cov = resp["window_coverage"]
    assert cov["window_days"] == 90
    assert cov["trailing_30d_value"] == 15239.0
    assert cov["trailing_30d_share"] == pytest.approx(0.9665, abs=0.001)
    assert cov["uniform_expected_share"] == pytest.approx(30 / 90, abs=0.001)
    assert "coverage_warning" in cov
    # The warning must be self-explanatory for the synthesizer LLM.
    assert "baseline" in cov["coverage_warning"].lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uniform_window_no_warning_but_shares_reported():
    # Uniform density: trailing 30d holds ~1/3 of a 90-day total.
    calc = _calculator(window_value=45000.0, trailing_value=15000.0)
    with patch("src.api.routes.kpi.get_kpi_calculator", return_value=calc):
        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Fabhalta", "window": NINETY_DAY_WINDOW}
        )
    cov = resp["window_coverage"]
    assert cov["trailing_30d_share"] == pytest.approx(1 / 3, abs=0.01)
    assert "coverage_warning" not in cov


@pytest.mark.unit
@pytest.mark.asyncio
async def test_short_window_skips_coverage_probe():
    """A ≤45-day window IS its own trailing period — no second calculation."""
    calc = _calculator(window_value=15239.0, trailing_value=15239.0)
    with patch("src.api.routes.kpi.get_kpi_calculator", return_value=calc):
        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Fabhalta", "window": "2026-06-07 to 2026-07-07"}
        )
    assert "window_coverage" not in resp
    assert calc.calculate.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_volume_kpi_skips_coverage_probe():
    """TRx Share (WS3-BI-008) is a ratio — a trailing-30d share comparison is
    meaningless there and would fire a false warning."""
    calc = _calculator(window_value=0.42, trailing_value=0.41, kpi_id="WS3-BI-008")
    with patch("src.api.routes.kpi.get_kpi_calculator", return_value=calc):
        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx Share", "window": NINETY_DAY_WINDOW}
        )
    assert "window_coverage" not in resp
    assert calc.calculate.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trailing_probe_failure_never_blocks_main_result():
    """If the trailing probe errors, the main figure still returns — with no
    fabricated coverage block."""
    calc = MagicMock()

    def calculate(kpi_id_arg, context=None, **_):
        window = (context or {}).get("window") or {}
        if str(window.get("start", "")).startswith("2026-06-07"):
            raise RuntimeError("probe failed")
        return _result(kpi_id_arg, 15767.0)

    calc.calculate = MagicMock(side_effect=calculate)
    with patch("src.api.routes.kpi.get_kpi_calculator", return_value=calc):
        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Fabhalta", "window": NINETY_DAY_WINDOW}
        )
    assert resp["success"] is True
    assert resp["value"] == 15767.0
    assert "window_coverage" not in resp


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zero_window_value_skips_shares():
    """No division by zero; an all-zero window carries no coverage math."""
    calc = _calculator(window_value=0.0, trailing_value=0.0)
    with patch("src.api.routes.kpi.get_kpi_calculator", return_value=calc):
        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Fabhalta", "window": NINETY_DAY_WINDOW}
        )
    assert resp["success"] is True
    assert "window_coverage" not in resp


# --- Future-end windows (PR #1554 codex iter-1): an in-progress calendar
# --- period ("this quarter"/"this year" from #1546, but equally explicit
# --- "Q3 2026", bare "2026", or a dict spec on main) carries a future
# --- ``window.end``. Unclamped, the trailing-30d sub-window lies entirely in
# --- the future -> trailing_value 0 -> share 0 -> the coverage warning is
# --- silently SUPPRESSED for exactly the windows users most ask about.


def _probe_kpi(kpi_id: str = "WS3-BI-005") -> SimpleNamespace:
    return SimpleNamespace(id=kpi_id)


def _probe_calculator(trailing_value: float):
    calc = MagicMock()
    calc.calculate = MagicMock(
        side_effect=lambda kpi_id_arg, context=None, **_: _result(kpi_id_arg, trailing_value)
    )
    return calc


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_future_end_window_probes_elapsed_tail():
    """'this year' asked on 2026-08-11: end is 2027-01-01, but the trailing
    probe must cover the last 30 ELAPSED days (2026-07-12 .. 2026-08-11) and
    window_days must be the 222 elapsed days, not the 365 nominal ones."""
    window = Window(
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2027, 1, 1, tzinfo=timezone.utc),
        kind="absolute",
        label="2026",
    )
    calc = _probe_calculator(trailing_value=15239.0)
    cov = await _window_coverage_probe(
        _probe_kpi(), _result("WS3-BI-005", 15767.0), window, calc, {}, now=NOW
    )
    assert cov is not None
    assert cov["window_days"] == 222
    assert cov["trailing_30d_value"] == 15239.0
    assert cov["uniform_expected_share"] == pytest.approx(30 / 222, abs=0.001)
    # 96.7% of the total in the elapsed tail >> 2x uniform -> warning restored.
    assert "coverage_warning" in cov
    probed = calc.calculate.call_args.kwargs["context"]["window"]
    assert probed["start"] == "2026-07-12T00:00:00+00:00"
    assert probed["end"] == "2026-08-11T00:00:00+00:00"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_future_end_short_elapsed_skips_probe():
    """'this quarter' 41 elapsed days in: the 45-day gate must apply to the
    ELAPSED span (41), not the 92-day nominal span — no probe at all."""
    window = Window(
        start=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end=datetime(2026, 10, 1, tzinfo=timezone.utc),
        kind="absolute",
        label="Q3 2026",
    )
    calc = _probe_calculator(trailing_value=0.0)
    cov = await _window_coverage_probe(
        _probe_kpi(), _result("WS3-BI-005", 9401.0), window, calc, {}, now=NOW
    )
    assert cov is None
    assert calc.calculate.call_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_past_window_probe_unchanged_by_clamp():
    """A window that already ended is untouched: clamping to a later ``now``
    must leave days and trailing dates byte-identical to the pre-clamp math."""
    window = Window(
        start=datetime(2026, 4, 8, tzinfo=timezone.utc),
        end=datetime(2026, 7, 7, tzinfo=timezone.utc),
        kind="absolute",
        label="2026-04-08 to 2026-07-07",
    )
    calc = _probe_calculator(trailing_value=15239.0)
    cov = await _window_coverage_probe(
        _probe_kpi(), _result("WS3-BI-005", 15767.0), window, calc, {}, now=NOW
    )
    assert cov is not None
    assert cov["window_days"] == 90
    assert "coverage_warning" in cov
    probed = calc.calculate.call_args.kwargs["context"]["window"]
    assert probed["start"] == "2026-06-07T00:00:00+00:00"
    assert probed["end"] == "2026-07-07T00:00:00+00:00"
