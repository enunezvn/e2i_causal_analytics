"""POST-DEPLOY live verification for #2115, run INSIDE the deployed e2i_api container.

Deliberately verifies the path PROD actually takes: worker_forecast ships at
replicas: 0 (dark), so TimesFM must be REFUSED and the forecast must still
succeed on a Holt-Winters champion. A cert that only proved the happy path
would be certifying a configuration prod is not running.
"""
import asyncio, json, os, datetime as dt
import src
assert src.__file__.startswith("/app/"), src.__file__

out = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "src_file": src.__file__}
fail = []

def check(name, ok, detail):
    out.setdefault("checks", []).append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail)[:300], flush=True)
    if not ok:
        fail.append(name)

# 1. the tool is registered in the deployed chat tool set
from src.api.routes.chatbot_tools import E2I_CHATBOT_TOOLS, E2I_TOOL_MAP
names = sorted(getattr(t, "name", str(t)) for t in E2I_CHATBOT_TOOLS)
check("forecast_kpi_tool registered in E2I_CHATBOT_TOOLS", "forecast_kpi_tool" in names, names)
check("forecast_kpi_tool in E2I_TOOL_MAP", "forecast_kpi_tool" in E2I_TOOL_MAP, sorted(E2I_TOOL_MAP)[:20])

# 2. the registry tool
from src.tool_registry.tools.kpi_forecast import TOOL_NAME
check("kpi_forecaster registry tool importable", TOOL_NAME == "kpi_forecaster", TOOL_NAME)

# 3. the forecast guidance is in the REAL deployed system prompts, and no slot is
#    left unsubstituted (a literal "{capability_guidance}" would ship the brace to the model)
from src.kpi.capability_policy import forecast_guidance_block
from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT
from src.api.routes.chatbot_graph import E2I_CHATBOT_SYSTEM_PROMPT
fg = forecast_guidance_block()
for label, prompt in (("copilotkit", E2I_COPILOT_SYSTEM_PROMPT), ("chatbot_graph", E2I_CHATBOT_SYSTEM_PROMPT)):
    check(f"{label} prompt carries the forecast guidance", fg and fg in prompt, fg[:160])
    check(f"{label} prompt has no unsubstituted slot",
          "{capability_guidance}" not in prompt and "{breakdown_guidance}" not in prompt,
          [t for t in ("{capability_guidance}", "{breakdown_guidance}") if t in prompt] or "none")
    check(f"{label} prompt names forecast_kpi_tool", "forecast_kpi_tool" in prompt,
          "forecast_kpi_tool" in prompt)

# 4. the worker is DARK in prod -> TimesFM refused, HW still serves
from src.kpi.forecast import timesfm as tf
ready, why = tf.worker_available()
out["worker_available"] = {"ready": ready, "why": why}
print(f"worker_available={ready} :: {why}", flush=True)

# 5. a REAL forecast off the REAL prod series
from src.api.routes.chat_forecast_tool import run_forecast
payload = asyncio.run(run_forecast("TRx", "Kisqali", None, 6))
out["kisqali"] = payload
check("real forecast succeeded", payload.get("success"), payload.get("error"))
if payload.get("success"):
    bt = payload["backtest"]
    champ = payload["champion"]
    check("champion is a Holt-Winters model (worker dark)", champ.startswith("holt_winters"), champ)
    ran = [m["model"] for m in bt["models"]]
    notrun = {m["model"]: m["reason"] for m in bt["models_not_run"]}
    if ready:
        check("worker UP: timesfm contested", "timesfm_2_5" in ran, ran)
    else:
        check("worker DARK: timesfm refused with a reason, not silently dropped",
              "timesfm_2_5" in notrun, notrun or ran)
    check("every served month has a band containing the point",
          all(p["lower"] <= p["value"] <= p["upper"] for p in payload["forecast"]),
          [(p["month"], p["lower"], p["value"], p["upper"]) for p in payload["forecast"]])
    check("payload is strict-JSON serialisable (no NaN/Infinity)",
          bool(json.dumps(payload, allow_nan=False, default=str)), "ok")
    print(f"  champion={champ}  n={payload['n_observations']}  through={payload['data_through']}", flush=True)
    for m in bt["models"]:
        print(f"    {m['model']:26s} MAPE {m['monthly_mape_pct']:6.2f}%  origins={m['origins_scored']}", flush=True)
    for k, v in notrun.items():
        print(f"    NOT RUN {k}: {v[:100]}", flush=True)
    for p in payload["forecast"]:
        print(f"    {p['month']}  {p['value']:>12,.0f}   [{p['lower']:>12,.0f} .. {p['upper']:>12,.0f}]", flush=True)

# 6. routing. The two predicates are deliberately EXCLUSIVE (planner.py:1259 returns
#    False from is_forecast_question when the risk predicate fires), because each one
#    grades a DECOMPOSED sub-question, not the compound ask. So the thing worth
#    verifying is the capability: does each half of demo 6.5 reach the right tool?
from src.agents.tool_composer.planner import ToolPlanner


class _Reg:
    def __init__(self, known):
        self.known = set(known)

    def validate_tool_exists(self, name):
        return name in self.known

    def get_schema(self, name):
        return None


class _SQ:
    def __init__(self, q, intent="PREDICTIVE"):
        self.question, self.intent, self.id = q, intent, "sq1"


def _map(q, known=("kpi_forecaster", "risk_scorer", "causal_effect_estimator", "causal_impact")):
    pl = ToolPlanner.__new__(ToolPlanner)
    pl.registry = _Reg(known)
    m = pl._get_fallback_mapping(_SQ(q))
    return getattr(m, "tool_name", None)

forecast_half = "What's the Kisqali TRx forecast for the next two quarters?"
risk_half = "What are the risks to that forecast?"
scoring_ask = "Which HCP segments are highest risk of churn?"

out["routing"] = {
    "forecast_half": {"q": forecast_half, "tool": _map(forecast_half)},
    "risk_half": {"q": risk_half, "tool": _map(risk_half)},
    "entity_scoring_control": {"q": scoring_ask, "tool": _map(scoring_ask)},
}
check("demo 6.5 FORECAST half -> kpi_forecaster",
      _map(forecast_half) == "kpi_forecaster", out["routing"]["forecast_half"])
check("demo 6.5 RISK half -> a causal tool, NOT the entity risk_scorer",
      _map(risk_half) in ("causal_effect_estimator", "causal_impact"), out["routing"]["risk_half"])
check("CONTROL: an entity-scoring ask still reaches risk_scorer (the fix did not swallow it)",
      _map(scoring_ask) == "risk_scorer", out["routing"]["entity_scoring_control"])
check("degraded: forecast ask with NO forecaster registered still maps somewhere",
      _map(forecast_half, known=("risk_scorer",)) == "risk_scorer",
      _map(forecast_half, known=("risk_scorer",)))

out["failures"] = fail
out["verdict"] = "PASS" if not fail else "FAIL"
with open("/tmp/postdeploy_verify.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nVERDICT:", out["verdict"], "failures=", fail, flush=True)
