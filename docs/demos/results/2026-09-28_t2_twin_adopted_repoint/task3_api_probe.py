#!/usr/bin/env python3
"""Lane T2 Task 3: the DEPLOYED twin estimates on `adopted` with the DR interval.

Run from a checkout, AFTER the deploy that ships #2305 is live:
    .venv/bin/dotenv -f .env run -- .venv/bin/python \
        docs/demos/results/2026-09-28_t2_twin_adopted_repoint/task3_api_probe.py
Checks, each printed `[ok ]` / `[BAD]` (exit 1 on any BAD):
  1. /digital-twin/health healthy, 3 brands simulable.
  2. /intervention-types: 8/8 available_for_effect per brand on the cohort_causal basis
     (the availability gate now runs the same rule /simulate does).
  3. POST /simulate digital_engagement per brand: completed, cohort provenance,
     is_significant, CI excludes 0, CI half-width reported (plan: ~ +-0.04).
  4. POST /simulate rep_training_quality (the planted null) per brand: reported, not gated
     here (the family-level null clause lives in verify_adoption_channel_recovery.py).
  5. /proposed-experiments: envelope outcome_column == "adopted"; every stored row resolves
     to a known outcome or null per the date rule (census 24/3/3 + the new runs).
Each POST /simulate persists a twin_simulations row through the app, as a user run does.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://eznomics.site"
BRANDS = ("Kisqali", "Fabhalta", "Remibrutinib")
ok = bad = 0
out: dict = {}


def check(name, cond, detail=""):
    global ok, bad
    ok += bool(cond)
    bad += not cond
    print(f"[{'ok ' if cond else 'BAD'}] {name} — {detail}", flush=True)


def http(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


E = os.environ
st, tok = http(
    "POST",
    f"{E['SUPABASE_URL']}/auth/v1/token?grant_type=password",
    {"email": E.get("E2I_ADMIN_EMAIL", "admin@e2i.local"), "password": E["E2I_ADMIN_PASSWORD"]},
    {"apikey": E["SUPABASE_ANON_KEY"]},
)
token = (tok or {}).get("access_token")
check("operator login", st == 200 and bool(token), st)
H = {"Authorization": f"Bearer {token}"}

st, h = http("GET", f"{API}/api/digital-twin/health", headers=H)
out["health"] = h
check("health: healthy, brands_simulable 3", st == 200 and h.get("status") == "healthy"
      and h.get("brands_simulable") == 3, f"{st} {h.get('status')}/{h.get('brands_simulable')}")

for b in BRANDS:
    st, it = http("GET", f"{API}/api/digital-twin/intervention-types?brand={b}", headers=H)
    out[f"intervention_types_{b}"] = it
    types = it.get("interventions", []) if isinstance(it, dict) else []
    n_eff = sum(1 for t in types if t.get("available_for_effect"))
    bases = sorted({t.get("effect_basis") for t in types})
    check(f"{b}: 8/8 available_for_effect, cohort_causal basis",
          st == 200 and len(types) == 8 and n_eff == 8 and bases == ["cohort_causal"],
          f"{n_eff}/{len(types)} {bases}")


def simulate(brand, intervention):
    st, sim = http("POST", f"{API}/api/digital-twin/simulate",
                   {"intervention": {"intervention_type": intervention, "duration_weeks": 4},
                    "brand": brand, "twin_count": 200}, headers=H)
    out[f"simulate_{intervention}_{brand}"] = sim
    if not isinstance(sim, dict):
        return st, {}, None
    lo, hi = sim.get("simulated_ci_lower"), sim.get("simulated_ci_upper")
    row = {k: sim.get(k) for k in ("status", "data_provenance", "estimate_scope", "simulated_ate",
                                   "simulated_ci_lower", "simulated_ci_upper", "is_significant",
                                   "recommendation", "error_message")}
    half = (hi - lo) / 2 if lo is not None and hi is not None else None
    return st, row, half


for b in BRANDS:
    st, row, half = simulate(b, "digital_engagement")
    excl = row.get("simulated_ci_lower") is not None and not (
        row["simulated_ci_lower"] <= 0.0 <= row["simulated_ci_upper"])
    check(f"{b}: digital_engagement completes on the cohort, is_significant, CI excludes 0",
          st == 200 and row.get("status") == "completed"
          and str(row.get("data_provenance", "")).startswith("cohort_estimated")
          and row.get("is_significant") is True and excl and not row.get("error_message"),
          f"{st} ate={row.get('simulated_ate')} ci=({row.get('simulated_ci_lower')}, "
          f"{row.get('simulated_ci_upper')}) half-width={half if half is None else round(half, 4)} "
          f"sig={row.get('is_significant')} rec={row.get('recommendation')}")

for b in BRANDS:
    st, row, half = simulate(b, "rep_training_quality")
    print(f"[info] {b}: rep_training_quality (planted null) ate={row.get('simulated_ate')} "
          f"ci=({row.get('simulated_ci_lower')}, {row.get('simulated_ci_upper')}) "
          f"sig={row.get('is_significant')} status={row.get('status')} http={st}", flush=True)

st, pe = http("GET", f"{API}/api/digital-twin/proposed-experiments", headers=H)
out["proposed_experiments"] = pe
props = pe.get("proposals", []) if isinstance(pe, dict) else []
cols: dict = {}
for p in props:
    cols[str(p.get("outcome_column"))] = cols.get(str(p.get("outcome_column")), 0) + 1
check("proposed-experiments: envelope outcome_column is adopted; rows resolve per the date rule",
      st == 200 and isinstance(pe, dict) and pe.get("outcome_column") == "adopted"
      and set(cols) <= {"adopted", "conversion_rate", "cohort_conversion_outcome", "None"},
      f"{st} envelope={pe.get('outcome_column') if isinstance(pe, dict) else None} "
      f"rows_by_outcome={cols} n={len(props)}")

(HERE / "task3_api_probe.json").write_text(json.dumps(out, indent=1, default=str))
print(f"\n# ok={ok} bad={bad}")
sys.exit(0 if bad == 0 else 1)
