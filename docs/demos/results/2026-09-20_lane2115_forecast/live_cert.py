"""LIVE cert for #2115: the real chat tool, the real PROD series, the real worker."""
import asyncio, json, os, time, datetime as dt
import src
assert src.__file__.startswith("/work/"), src.__file__
from src.kpi.forecast import timesfm as tf
from src.kpi.forecast import service as svc
from src.api.routes.chat_forecast_tool import run_forecast

out = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "src_file": src.__file__}

ready, why = tf.worker_available()
print("worker_available:", ready, "|", why, flush=True)
out["worker_available"] = {"ready": ready, "why": why}
assert ready, "the forecast worker must be reachable for this cert"

# 1. a real dispatch round trip through the broker
t0 = time.time()
got = tf.dispatch_batch([[float(i) for i in range(1, 41)]], 3)
out["dispatch_round_trip_s"] = round(time.time() - t0, 2)
print("dispatch round trip:", out["dispatch_round_trip_s"], "s ->", got, flush=True)
assert got and got[0] and len(got[0]) == 3

# 2. the real Supabase-backed chat tool, TimesFM in the contest
for brand in ("Kisqali", "Fabhalta", "Remibrutinib"):
    t0 = time.time()
    payload = asyncio.run(run_forecast("TRx", brand, None, 6))
    elapsed = round(time.time() - t0, 1)
    print(f"\n=== {brand}  ({elapsed}s)  success={payload['success']}", flush=True)
    if not payload["success"]:
        print("  ERROR:", payload["error"], flush=True)
        out.setdefault("brands", {})[brand] = payload
        continue
    bt = payload["backtest"]
    print(f"  data_through={payload['data_through']}  n={payload['n_observations']}  "
          f"champion={payload['champion']}  origins={bt['origins']}", flush=True)
    for m in bt["models"]:
        print(f"    {m['model']:26s} MAPE {m['monthly_mape_pct']:6.2f}%  "
              f"total {m['horizon_total_error_pct']:6.2f}%  origins={m['origins_scored']}", flush=True)
    for m in bt["models_not_run"]:
        print(f"    NOT RUN {m['model']}: {m['reason'][:80]}", flush=True)
    for p in payload["forecast"]:
        print(f"    {p['month']}  {p['value']:>12,.0f}   [{p['lower']:>12,.0f} .. {p['upper']:>12,.0f}]", flush=True)
    print(f"  two-quarter total: {payload['horizon_total']:,.0f}", flush=True)
    payload["_elapsed_s"] = elapsed
    out.setdefault("brands", {})[brand] = payload

with open("/out/live_cert.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nWROTE /out/live_cert.json", flush=True)
