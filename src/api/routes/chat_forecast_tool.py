"""``forecast_kpi_tool``: the chat surface for canonical KPI forecasting (#2115, Lane B).

Demo 6.5 — "Forecast Kisqali TRx volume for the next two quarters and tell me the
biggest risk to that forecast" — has refused on every run since 2026-07-29, because the
platform had no forecaster: ``e2i_data_query_tool(query_type='predictions')`` is a
memory search and ``prediction_synthesizer`` ensembles entity-level classifiers. This
tool is the forecast half. It reads Lane A's canonical monthly series, backtests every
available model on the same rolling origins, serves the lowest-error one, and attaches
that model's own measured miss distribution as the band.

IT LIVES IN ITS OWN MODULE BY CONSTRAINT: ``chatbot_tools.py`` and ``copilotkit.py`` are
size-ratchet pinned, so the tool body and its prompt text belong outside them.

THE HALF THIS TOOL CANNOT ANSWER IS NAMED IN ITS OWN PAYLOAD. Every model here is
univariate: it sees the brand's past volume and nothing else. A competitor entry, a
payer change or a regional step inside the horizon is invisible to all of them — the
planted Kisqali midwest -15% step from 2026-10 lands inside exactly this window and
moves nothing in the fit. So the payload carries ``limitations`` and
``risk_analysis_requires``, and the answer is expected to reach for the gap analyzer and
the causal tools for the risk half rather than narrating a risk the forecast cannot see.
A number that silently implied otherwise would be confidently wrong about the one thing
the user actually asked for.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.kpi.canonical_volume_series import SUPPORTED_METRICS, CanonicalSeriesError
from src.kpi.forecast import backtest as bt
from src.kpi.forecast import service as svc
from src.kpi.measure_basis import BUSINESS_METRICS_BASIS
from src.services.enum_labels import resolve_brand_label, resolve_region_label

logger = logging.getLogger(__name__)

QUERY_TYPE = "kpi_forecast"

#: What the answer must not let the reader forget. These are properties of the MODEL
#: CLASS, not caveats about this particular fit: no model the tool runs is given any
#: series other than the one it is extrapolating.
LIMITATIONS: tuple[str, ...] = (
    "This is a univariate extrapolation: every model is fitted on the brand's own past "
    "monthly volume and nothing else.",
    "It therefore cannot see anything that has not already happened to this series — a "
    "competitor launch, a payer or formulary change, a label change, a sales-force "
    "reallocation or a regional step change inside the forecast window are all invisible "
    "to it, however large.",
    "The band is the model's own measured error on this series at each horizon step, so "
    "it describes how wrong this model has BEEN — not the chance of an event it has "
    "never seen.",
    "A month flagged `floored_at_zero` is NOT a prediction of exactly zero: the model "
    "extrapolated this non-negative volume below zero, which means its trend has run "
    "past the range the fit is valid in. Report such a month as 'at or near zero, and "
    "beyond what this model can project' — never as a precise figure.",
)

#: 6.5 asks for a forecast AND its biggest risk. The risk half is a different question
#: about a different substrate, and these are the tools that hold it.
RISK_ANALYSIS_REQUIRES: tuple[str, ...] = (
    "For the risk half of a forecast question, call the gap analyzer (via "
    "`orchestrator_tool` or `tool_composer_tool`) for the brand x region gaps that are "
    "opening now, and `causal_analysis_tool` for what is driving them.",
    "A regional step change already underway shows up in the gap and causal views "
    "before it shows up in a national monthly total, so those tools can see risks this "
    "forecast structurally cannot.",
)

NOTE = (
    "A forecast, not an observation: the monthly values below are projected from the "
    "canonical business_metrics series and no month among them has happened yet. The "
    "model was chosen by rolling-origin backtest on this same series and the band is "
    "its own measured error."
)


class ForecastKpiInput(BaseModel):
    """Arguments for forecast_kpi_tool."""

    kpi_name: str = Field(
        ...,
        description=(
            "The volume KPI to forecast: TRx, NRx or NBRx (a registry code such as "
            "WS3-BI-005, or a full name such as 'Total Prescriptions', also works). "
            "Only these three have a canonical monthly series to forecast."
        ),
    )
    brand: Optional[str] = Field(
        None, description="Kisqali, Fabhalta or Remibrutinib. Omit to forecast all brands summed."
    )
    region: Optional[str] = Field(
        None, description="Optional US census region (northeast, south, midwest, west)."
    )
    horizon_months: int = Field(
        bt.DEFAULT_HORIZON,
        description=(
            "How many months ahead to forecast. 6 (two quarters) by default; "
            f"{svc.MAX_HORIZON} is the maximum the backtest can still grade."
        ),
    )


def _resolve_metric(kpi_name: str) -> str:
    """Map the user's phrasing onto a canonical volume metric, or refuse by name.

    The vocabulary is the SHARED one (``canonical_business_metric_name``), never a second
    copy: a private alias table here would drift out of step with the one the rest of the
    KPI surface uses, and the two would disagree about what 'TRx' means without ever
    failing a test.
    """
    from src.kpi.business_metric_vocabulary import canonical_business_metric_name
    from src.kpi.canonical_volume_stored import _ROW_NAME

    raw = (kpi_name or "").strip()
    if not raw:
        raise CanonicalSeriesError("a KPI name is required to forecast")
    by_registry_code = _ROW_NAME.get(raw.upper())
    candidate = by_registry_code or canonical_business_metric_name(raw) or raw.lower()
    if candidate not in SUPPORTED_METRICS:
        raise CanonicalSeriesError(
            f"{raw!r} has no canonical monthly series to forecast. Forecasting is "
            f"available for {', '.join(m.upper() for m in SUPPORTED_METRICS)} — the "
            "volume KPIs that are stored as a brand x region monthly business_metrics "
            "series. For any other KPI, call kpi_calculate_tool for its current value."
        )
    return candidate


def _run_forecast(
    metric: str,
    brand: Optional[str],
    region: Optional[str],
    horizon: int,
    client: Any,
    as_of: Optional[date],
) -> Dict[str, Any]:
    result = svc.forecast_kpi(
        metric,
        brand,
        region,
        horizon=horizon,
        client=client,
        as_of=as_of,
    )
    payload = result.to_payload()
    payload.update(
        {
            "success": True,
            "query_type": QUERY_TYPE,
            "is_forecast": True,
            "note": NOTE,
            "measure_basis": BUSINESS_METRICS_BASIS,
            "limitations": list(LIMITATIONS),
            "risk_analysis_requires": list(RISK_ANALYSIS_REQUIRES),
        }
    )
    return payload


def _refusal(error: str, **extra: Any) -> Dict[str, Any]:
    """A refusal never carries a forecast — an empty list would read as 'flat'."""
    out: Dict[str, Any] = {
        "success": False,
        "query_type": QUERY_TYPE,
        "is_forecast": False,
        "error": error,
        "forecast": None,
    }
    out.update(extra)
    return out


async def run_forecast(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    horizon_months: int = bt.DEFAULT_HORIZON,
    *,
    client: Any = None,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """The tool's whole behaviour, callable with an explicit reader.

    Kept separate from the ``@tool`` wrapper below because a LangChain tool validates
    its arguments against ``ForecastKpiInput`` and DROPS anything not declared there —
    so a reader cannot be passed through the tool call, and a tool that accepted one
    would have to advertise it to the model. Splitting them lets the tests drive the
    real logic against real rows without a test-only hook in the production path, and
    lets the composable-tool registration reuse it instead of copying it.
    """
    try:
        metric = _resolve_metric(kpi_name)
    except CanonicalSeriesError as exc:
        return _refusal(str(exc), kpi_name=kpi_name)

    resolved_brand: Optional[str] = None
    if brand and str(brand).strip():
        resolved_brand = resolve_brand_label(brand)
        if resolved_brand is None:
            return _refusal(
                f"{brand!r} is not a brand this platform tracks. The tracked brands are "
                "Kisqali, Fabhalta and Remibrutinib.",
                kpi_name=kpi_name,
            )

    resolved_region: Optional[str] = None
    if region and str(region).strip():
        resolved_region = resolve_region_label(region, allow_synonyms=True)
        if resolved_region is None:
            return _refusal(
                f"{region!r} does not resolve to a US census region. Ask the user "
                "whether they mean northeast, south, midwest or west.",
                kpi_name=kpi_name,
                brand=resolved_brand,
            )

    try:
        horizon = int(horizon_months)
    except (TypeError, ValueError):
        return _refusal(f"horizon_months must be a whole number of months, got {horizon_months!r}")

    try:
        return await asyncio.to_thread(
            _run_forecast, metric, resolved_brand, resolved_region, horizon, client, as_of
        )
    except svc.ForecastRefused as exc:
        return _refusal(str(exc), metric=metric, brand=resolved_brand, region=resolved_region)
    except CanonicalSeriesError as exc:
        return _refusal(str(exc), metric=metric, brand=resolved_brand, region=resolved_region)
    except Exception as exc:  # noqa: BLE001 — surface as a tool error, never fabricate
        logger.error("forecast_kpi_tool failed for %s/%s: %s", metric, resolved_brand, exc)
        return _refusal(str(exc), metric=metric, brand=resolved_brand, region=resolved_region)


@tool(args_schema=ForecastKpiInput)
async def forecast_kpi_tool(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    horizon_months: int = bt.DEFAULT_HORIZON,
) -> Dict[str, Any]:
    """Forecast a canonical monthly volume KPI (TRx / NRx / NBRx) with a measured band.

    Use this for ANY forward-looking volume question — "forecast Kisqali TRx for the
    next two quarters", "where will NRx be by year end", "project NBRx through Q2".
    It is the only tool that projects a KPI forward: `kpi_calculate_tool` reports what a
    month WAS, and `e2i_data_query_tool(query_type='predictions')` searches stored
    prediction memories, neither of which forecasts anything.

    It reads the canonical brand x region monthly business_metrics series, backtests
    Holt-Winters (and TimesFM 2.5 when the forecast worker is running) over the same
    rolling origins, serves the lower-error model, and returns a per-month prediction
    band built from that model's own measured error.

    HOW TO USE THE RESULT. Report the monthly path AND the horizon total, cite
    `data_through` (the last month that actually happened) and the champion's backtest
    error, and present the band as measured error rather than as a confidence interval.
    If any month carries `floored_at_zero: true` (or appears in
    `floored_at_zero_months`), say plainly that the model ran past zero there and the
    month is at or near zero rather than a precise prediction — do NOT present it as a
    forecast of 0.
    The forecast is UNIVARIATE — it cannot see a competitor entry, a payer change or a
    regional step inside the window, so when the user also asks about RISK (demo 6.5
    does), say plainly that the forecast cannot see such events and get the risk half
    from the gap analyzer and `causal_analysis_tool`. Do not narrate a risk as if the
    forecast had accounted for it.

    Args:
        kpi_name: TRx, NRx or NBRx (registry codes and full names also resolve).
        brand: Kisqali | Fabhalta | Remibrutinib. Omit for all brands summed.
        region: optional US census region (northeast, south, midwest, west).
        horizon_months: months ahead; 6 (two quarters) by default.
    """
    return await run_forecast(kpi_name, brand, region, horizon_months)


__all__ = [
    "LIMITATIONS",
    "NOTE",
    "QUERY_TYPE",
    "RISK_ANALYSIS_REQUIRES",
    "ForecastKpiInput",
    "forecast_kpi_tool",
    "run_forecast",
]
