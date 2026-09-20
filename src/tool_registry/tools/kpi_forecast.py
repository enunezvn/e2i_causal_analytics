"""``kpi_forecaster``: the composable face of forecast_kpi_tool (#2115, Lane B).

Demo 6.5 asks two questions at once — "forecast Kisqali TRx volume for the next two
quarters AND tell me the biggest risk to that forecast" — which is a tool-composer ask,
not a single-tool one. The composer plans over the tool REGISTRY, so the forecaster has
to be in it before a plan can contain a forecast step.

The registered callable delegates to ``src.api.routes.chat_forecast_tool.run_forecast``:
the same series read, the same backtest, the same champion, the same payload. A second
implementation here would eventually disagree with the chat tool about the same brand's
forecast inside the same answer.

The description is written FOR THE PLANNER, and it is where the risk half of 6.5 gets
its cue: it says in as many words that the forecast is univariate and cannot see a
competitor entry or payer change, so the planner has a reason to add the gap and causal
steps rather than treating the forecast as the whole answer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.kpi.forecast import backtest as bt
from src.tool_registry.registry import ToolParameter, ToolSchema, get_registry

logger = logging.getLogger(__name__)

TOOL_NAME = "kpi_forecaster"

#: The composer's planner maps the PREDICTIVE intent to (tool, source_agent); this is
#: the agent half. prediction_synthesizer is the platform's forward-looking agent, so a
#: forecast step is attributed where a reader of the plan would expect to find it.
SOURCE_AGENT = "prediction_synthesizer"

DESCRIPTION = (
    "Forecast a canonical monthly volume KPI (TRx, NRx or NBRx) forward for a brand "
    "and optional region, with a per-month prediction band and the rolling-origin "
    "backtest error of the model that was chosen. Use it for any forward-looking "
    "volume question ('forecast TRx for the next two quarters', 'project NRx through "
    "Q2'). The forecast is UNIVARIATE — it is fitted on the brand's own past volume "
    "alone and cannot see a competitor launch, a payer or formulary change, or a "
    "regional step change inside the window — so when the question also asks about "
    "RISK, plan gap-analysis and causal steps alongside this one: the forecast cannot "
    "answer the risk half and must not be presented as though it had."
)


class KpiForecastInput(BaseModel):
    """Input schema for the kpi_forecaster tool."""

    kpi_name: str = Field(..., description="TRx, NRx or NBRx (registry codes also resolve)")
    brand: Optional[str] = Field(None, description="Kisqali, Fabhalta or Remibrutinib")
    region: Optional[str] = Field(
        None, description="US census region: northeast/south/midwest/west"
    )
    horizon_months: int = Field(
        bt.DEFAULT_HORIZON, description="Months ahead to forecast (6 = two quarters)"
    )


class ForecastMonth(BaseModel):
    """One forecast month as the payload carries it."""

    month: str = Field(..., description="Forecast month as YYYY-MM")
    value: float = Field(..., description="Point forecast for that month")
    lower: float = Field(..., description="Lower edge of the measured-error band")
    upper: float = Field(..., description="Upper edge of the measured-error band")


class KpiForecastOutput(BaseModel):
    """The composer needs a declared output schema to bind a step's results.

    It mirrors ``KpiForecast.to_payload`` plus the chat-tool envelope; the fields a
    downstream step or a synthesiser would actually read are named, and the rest stays
    permissive because the backtest block grows as models are added.
    """

    model_config = {"extra": "allow"}

    success: bool = Field(..., description="False when the forecast was refused")
    query_type: str = Field("kpi_forecast", description="Payload discriminator")
    metric: Optional[str] = Field(None, description="trx, nrx or nbrx")
    brand: Optional[str] = Field(None, description="Brand the forecast covers")
    region: Optional[str] = Field(None, description="Region, or null for all regions summed")
    horizon_months: Optional[int] = Field(None, description="Months forecast ahead")
    data_through: Optional[str] = Field(None, description="Last month that actually happened")
    forecast: Optional[list[ForecastMonth]] = Field(
        None, description="Per-month point forecast with its band; null on a refusal"
    )
    horizon_total: Optional[float] = Field(None, description="Sum of the forecast months")
    champion: Optional[str] = Field(None, description="Model chosen by the backtest")
    backtest: Optional[Dict[str, Any]] = Field(
        None, description="Rolling-origin scores for every model that ran"
    )
    limitations: Optional[list[str]] = Field(
        None, description="What this univariate forecast structurally cannot see"
    )
    risk_analysis_requires: Optional[list[str]] = Field(
        None, description="Where the risk half of a forecast question must come from"
    )
    error: Optional[str] = Field(None, description="Why the forecast was refused")


async def kpi_forecaster(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    horizon_months: int = bt.DEFAULT_HORIZON,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Forecast a canonical volume KPI. Delegates to the chat tool's implementation."""
    from src.api.routes.chat_forecast_tool import run_forecast

    return await run_forecast(kpi_name, brand, region, horizon_months)


def register_kpi_forecast_tool() -> None:
    """Register ``kpi_forecaster`` in the global tool registry (idempotent)."""
    schema = ToolSchema(
        name=TOOL_NAME,
        description=DESCRIPTION,
        source_agent=SOURCE_AGENT,
        tier=2,
        input_parameters=[
            ToolParameter(
                name="kpi_name",
                type="str",
                description="TRx, NRx or NBRx (registry codes such as WS3-BI-005 also resolve)",
                required=True,
            ),
            ToolParameter(
                name="brand",
                type="str",
                description="Kisqali, Fabhalta or Remibrutinib; omit for all brands summed",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="region",
                type="str",
                description="US census region (northeast, south, midwest, west)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="horizon_months",
                type="int",
                description="Months ahead to forecast; 6 (two quarters) by default",
                required=False,
                default=bt.DEFAULT_HORIZON,
            ),
        ],
        output_schema="KpiForecastOutput",
        # Measured 2026-09-20: three Holt-Winters fits over 24 rolling origins on the
        # 164-month live series. The planner uses this to estimate plan duration.
        avg_execution_ms=12000,
        is_async=True,
    )
    registry = get_registry()
    registry.register(
        schema=schema,
        callable=kpi_forecaster,
        input_model=KpiForecastInput,
        # Without an output model the composer's schema builder raises LookupError
        # and the tool is invisible to the planner (src/agents/tool_composer/
        # tool_registry.py:build_tool_schemas).
        output_model=KpiForecastOutput,
    )


__all__ = [
    "DESCRIPTION",
    "ForecastMonth",
    "KpiForecastOutput",
    "SOURCE_AGENT",
    "TOOL_NAME",
    "KpiForecastInput",
    "kpi_forecaster",
    "register_kpi_forecast_tool",
]
