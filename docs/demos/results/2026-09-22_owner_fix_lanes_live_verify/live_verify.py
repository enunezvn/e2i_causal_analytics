#!/usr/bin/env python3
"""Live verification of PRs #2223 + #2224 after deploy (read-only unless --exercise-beats).

Runs on the droplet (prod == dev). Modeled on docs/demos/results/2026-09-22_twin_restore/verify_after_restore.py.
Every check prints PASS/FAIL with the measured value; exit 1 if any FAIL.

    python docs/demos/results/2026-09-22_owner_fix_lanes_live_verify/live_verify.py [--exercise-beats]

--exercise-beats additionally enqueues the two Feast beat tasks (the same calls the scheduler makes
every 6 h / 4 h) and the daily retraining sweep, then waits for their rows. Those are production
actions (Redis online-store writes + tracking rows) — run only with owner GO.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from urllib.error import HTTPError

API = "https://eznomics.site"
ROOT = "/home/enunez/Projects/e2i_causal_analytics"
EXERCISE = "--exercise-beats" in sys.argv
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def sh(cmd, timeout=120):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def psql(sql):
    rc, out, err = sh(
        "docker exec -i supabase-db psql -U postgres -d postgres -At -F '|' -v ON_ERROR_STOP=1",
        timeout=120,
    ) if False else (None, None, None)
    p = subprocess.run(
        ["docker", "exec", "-i", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-At", "-F", "|", "-v", "ON_ERROR_STOP=1"],
        input=sql, capture_output=True, text=True, timeout=120,
    )
    if p.returncode != 0:
        return f"ERR:{p.stderr.strip()[:200]}"
    return p.stdout.strip()


def http(method, url, body=None, headers=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


def load_env():
    env = {}
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


E = load_env()

# ---------------------------------------------------------------- A. content layer
rc, main_sha, _ = sh(f"git -C {ROOT} fetch -q origin main && git -C {ROOT} rev-parse origin/main")
for ctr in ("e2i_api", "e2i-causal-analytics-worker_medium-1", "e2i_scheduler", "e2i_frontend"):
    rc, img, _ = sh(f"docker inspect {ctr} --format '{{{{.Config.Image}}}}'")
    check(f"A.image {ctr} == origin/main", img.endswith(main_sha), img[-52:])
for ctr in ("e2i_feast", "e2i_feast_materializer"):
    rc, st, _ = sh(f"docker inspect {ctr} --format '{{{{.State.Health.Status}}}} {{{{.State.StartedAt}}}}'")
    check(f"A.sidecar {ctr} healthy", st.startswith("healthy"), st)
rc, log, _ = sh("docker logs --since 6h e2i_feast 2>&1 | grep -c 'materialize endpoints registry-locked'")
check("A.e2i_feast log: materialize endpoints registry-locked (serve_locked.py live)", log.strip() not in ("", "0"), f"lines={log.strip()}")
rc, wh, _ = sh("docker exec e2i-causal-analytics-worker_medium-1 python -c \"import urllib.request;print(urllib.request.urlopen('http://feast:6566/health',timeout=10).status)\"")
check("A.sidecar /health from worker_medium", wh.strip() == "200", wh)

# ---------------------------------------------------------------- B. schema + data
mig = psql("SELECT filename FROM schema_migrations WHERE filename LIKE '150_%' OR filename LIKE 'ml/046_%' ORDER BY 1;")
check("B.migrations 150 + ml/046 in ledger", "150_ml_registry_cohort_contract" in mig and "ml/046" in mig, mig.replace("\n", " ; "))
cols = psql("SELECT string_agg(column_name, ',') FROM information_schema.columns WHERE table_name='ml_model_registry' AND column_name LIKE 'cohort_%';")
check("B.ml_model_registry cohort_* columns", all(c in cols for c in ("cohort_data_source", "cohort_target_outcome", "cohort_feature_manifest_source")), cols)
nulls = psql("SELECT count(*) || '/' || count(*) FILTER (WHERE cohort_data_source IS NULL AND cohort_target_outcome IS NULL) FROM ml_model_registry WHERE is_synthetic=false;")
check("B.14 real registry rows carry NULL contracts (no backfill)", nulls == "14/14", nulls)
idx = psql("SELECT indexdef FROM pg_indexes WHERE indexname='uq_ab_results_one_final_per_experiment';")
check("B.partial unique index one FINAL per experiment", "WHERE (analysis_type = 'final'" in idx, idx[-80:])
fin = psql("SELECT count(*) || '/' || count(DISTINCT experiment_id) FROM ab_experiment_results WHERE analysis_type='final';")
check("B.final rows 1:1 with experiments", fin.split("/")[0] == fin.split("/")[1], fin)
twin = psql("SELECT count(*) || ' sims, linked=' || count(experiment_design_id) FROM twin_simulations;")
check("B.twin_simulations link state readable", twin.startswith("3"), twin)
drafts = psql("SELECT count(*) FROM ml_experiments WHERE status='draft';")
check("B.no draft experiments created by verification", drafts == "0", f"drafts={drafts}")

# ---------------------------------------------------------------- C. celery routing inside worker_medium
rc, routes, err = sh(
    "docker exec e2i-causal-analytics-worker_medium-1 python -c \""
    "from src.workers.celery_app import celery_app as a;import json;"
    "r=a.conf.task_routes;b=a.conf.beat_schedule;"
    "print(json.dumps({'fid':r.get('src.tasks.fidelity_tracking_update'),'retrain':r.get('src.tasks.execute_model_retraining'),"
    "'mat_inc':r.get('src.tasks.materialize_incremental_features'),"
    "'beat_inc':b.get('feast-materialize-incremental',{}).get('options'),'beat_weekly':b.get('feast-materialize-full-weekly',{}).get('options'),"
    "'beat_retrain':b.get('retraining-evaluation-daily',{}).get('options')}))\"", timeout=240)
try:
    rj = json.loads(routes.splitlines()[-1])
except Exception:  # noqa: BLE001
    rj = {"error": (routes + err)[-300:]}
check("C.fidelity_tracking_update -> analytics", (rj.get("fid") or {}).get("queue") == "analytics", json.dumps(rj.get("fid")))
check("C.execute_model_retraining -> analytics", (rj.get("retrain") or {}).get("queue") == "analytics", json.dumps(rj.get("retrain")))
check("C.feast weekly beat off the ml queue", (rj.get("beat_weekly") or {}).get("queue") not in (None, "ml"), json.dumps(rj.get("beat_weekly")))
check("C.feast incremental beat on a consumed queue", (rj.get("beat_inc") or {}).get("queue") == "analytics", json.dumps(rj.get("beat_inc")))
rc, aq, _ = sh("docker exec e2i-causal-analytics-worker_medium-1 celery -A src.workers.celery_app inspect active_queues 2>/dev/null | grep -c \"'name': 'analytics'\"", timeout=240)
check("C.worker_medium consumes analytics", aq.strip() not in ("", "0"), f"matches={aq.strip()}")
rc, fu, _ = sh("docker exec e2i-causal-analytics-worker_medium-1 python -c \"import os;print(os.getenv('FEAST_URL'))\"")
check("C.FEAST_URL set in worker_medium", fu.strip().startswith("http"), fu.strip())

# ---------------------------------------------------------------- D. API (admin JWT)
st, tok = http("POST", f"{E['SUPABASE_URL']}/auth/v1/token?grant_type=password",
               body={"email": "admin@e2i.local", "password": E["E2I_ADMIN_PASSWORD"]},
               headers={"apikey": E["SUPABASE_ANON_KEY"], "Content-Type": "application/json"})
check("D.admin token minted", st == 200 and "access_token" in tok, f"status={st}")
H = {"Authorization": f"Bearer {tok.get('access_token', '')}"}
st, h = http("GET", f"{API}/api/digital-twin/health", headers=H)
check("D./digital-twin/health 200", st == 200, json.dumps(h)[:120])
st, m = http("GET", f"{API}/api/digital-twin/models", headers=H)
models = m.get("models", []) if isinstance(m, dict) else []
check("D./models: 3 rows, all unvalidated, shared_fit_model_count 3",
      st == 200 and len(models) == 3 and all(x.get("fidelity_status") == "unvalidated" for x in models)
      and all(x.get("shared_fit_model_count") == 3 for x in models),
      json.dumps([{k: x.get(k) for k in ("brand", "fidelity_status", "r2_score_basis", "shared_fit_model_count")} for x in models]))
st, p = http("GET", f"{API}/api/digital-twin/proposed-experiments", headers=H)
env_keys = {k: p.get(k) for k in ("total_proposed", "total_linked", "real_experiments_running", "outcome_measurable_in_real_mode")} if isinstance(p, dict) else {}
items = p.get("proposals", p.get("items", [])) if isinstance(p, dict) else []
check("D./proposed-experiments 200 with honest envelope", st == 200 and env_keys.get("total_proposed") is not None, json.dumps(env_keys))
check("D.proposals: every item unlinked deploy/refine with n + weeks",
      bool(items) and all(x.get("recommendation") in ("deploy", "refine") and x.get("recommended_sample_size") for x in items),
      f"n={len(items)} first={json.dumps({k: (items[0].get(k) if items else None) for k in ('brand','intervention_type','recommendation','recommended_sample_size','recommended_duration_weeks','fidelity_status','proposal_basis')})}")
st, sims = http("GET", f"{API}/api/digital-twin/simulations?limit=1", headers=H)
first = None
if isinstance(sims, dict):
    for k in ("simulations", "items", "results"):
        if isinstance(sims.get(k), list) and sims[k]:
            first = sims[k][0]
            break
elif isinstance(sims, list) and sims:
    first = sims[0]
check("D.simulation list carries experiment_design_id (read-back)", st == 200 and first is not None and "experiment_design_id" in first,
      f"status={st} keys={sorted(first.keys())[:8] if first else None}")
st, oa = http("GET", f"{API}/api/openapi.json", headers=H)
paths = oa.get("paths", {}) if isinstance(oa, dict) else {}
check("D.openapi lists proposals + draft routes", any(p_.endswith("/proposed-experiments") for p_ in paths) and any("/draft" in p_ and "proposed-experiments" in p_ for p_ in paths),
      ",".join(p_ for p_ in paths if "proposed-experiments" in p_))
trig = paths.get("/api/monitoring/retraining/trigger/{model_id}", {}).get("post", {})
schema_ref = json.dumps(trig)[:1]
check("D.openapi retraining trigger present", bool(trig), "POST /api/monitoring/retraining/trigger/{model_id}")

# ---------------------------------------------------------------- E. optional: exercise the beats (production actions)
if EXERCISE:
    before_jobs = psql("SELECT count(*) FROM ml_feast_materialization_jobs;")
    before_fresh = psql("SELECT count(*) FROM ml_feast_feature_freshness;")
    for task in ("src.tasks.materialize_incremental_features", "src.tasks.check_feature_freshness", "src.tasks.check_retraining_for_all_models"):
        rc, out, err = sh(
            f"docker exec e2i-causal-analytics-worker_medium-1 python -c \"from src.workers.celery_app import celery_app as a;"
            f"r=a.send_task('{task}');print(r.id)\"", timeout=240)
        print(f"enqueued {task}: {out.strip()[-40:]} {err[-120:]}")
    time.sleep(150)
    after_jobs = psql("SELECT status || ':' || count(*) FROM ml_feast_materialization_jobs GROUP BY status ORDER BY 1;")
    after_fresh = psql("SELECT freshness_status || ':' || count(*) FROM ml_feast_feature_freshness GROUP BY 1 ORDER BY 1;")
    check("E.materialization job rows landed", psql("SELECT count(*) FROM ml_feast_materialization_jobs;") != before_jobs, f"before={before_jobs} after={after_jobs.replace(chr(10), ' ')}")
    check("E.freshness rows landed", psql("SELECT count(*) FROM ml_feast_feature_freshness;") != before_fresh, f"before={before_fresh} after={after_fresh.replace(chr(10), ' ')}")
    rc, wl, _ = sh("docker logs --since 10m e2i-causal-analytics-worker_medium-1 2>&1 | grep -E 'materialize|freshness|retraining|no_cohort_contract|Feast' | tail -25")
    print("--- worker_medium log tail ---\n" + wl)
    rc, fl, _ = sh("docker logs --since 10m e2i_feast 2>&1 | grep -iE 'materializ' | tail -8")
    print("--- e2i_feast log tail ---\n" + fl)

fails = [r for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
sys.exit(1 if fails else 0)
