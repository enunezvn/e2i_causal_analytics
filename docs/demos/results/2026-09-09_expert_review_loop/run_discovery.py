#!/usr/bin/env python3
"""Run the Remibrutinib patient-grain discovery job and record every question's
band and evidence. Usage: run_discovery.py <label> <out_dir>  (reads .env)."""
import base64, json, os, subprocess, sys, time, urllib.parse, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
API = os.environ.get("E2I_API_BASE", "https://eznomics.site/api")
DATASET, BRAND = "patient_journeys", "Remibrutinib"

def mint_token() -> str:
    body = json.dumps({"email": os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local"),
                       "password": os.environ["E2I_ADMIN_PASSWORD"]}).encode()
    req = urllib.request.Request(f"{os.environ['SUPABASE_URL']}/auth/v1/token?grant_type=password",
                                 data=body, headers={"apikey": os.environ["SUPABASE_ANON_KEY"],
                                                     "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["access_token"]

def call(token, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _psql_rows(where: str) -> dict:
    # The agent path writes json.dumps(details) INTO the jsonb column, so 480 live
    # rows are JSON *strings* (measured 2026-09-08: 480 string / 545 object);
    # decode both shapes or every key read below is NULL.
    sql = ("with d as (select test_type, status, created_at, "
           "case when jsonb_typeof(details_json) = 'string' then (details_json #>> '{}')::jsonb else details_json end as dj "
           f"from public.causal_validations where {where}) "
           "select test_type, status, coalesce(dj->>'stopped_for_budget','') as budget, "
           "coalesce(jsonb_array_length(dj->'subset_effects'), jsonb_array_length(dj->'bootstrap_effects'), 0) as n "
           "from d order by created_at desc")
    proc = subprocess.run(["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tA", "-F", "|", "-c", sql],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:  # never let a failed read masquerade as "no evidence"
        raise RuntimeError(f"causal_validations read failed: {proc.stderr.strip()}")
    found = {}
    for ln in proc.stdout.splitlines():
        p = ln.split("|")
        if len(p) == 4 and p[0] not in found:      # newest row per test
            found[p[0]] = {"status": p[1], "stopped_for_budget": p[2], "n_effects": int(p[3])}
    return found

def db_tests(analysis_id, treatment: str, outcome: str, since_iso: str) -> dict:
    """Per-test status and evidence size straight from causal_validations.
    The API omits SKIPPED tests from refutation.tests and keeps only the message
    text of details, so the baseline's 'skipped' and the new loops' resample
    counts are only visible here. Unlinked runs are keyed by the query-derived
    uuid5 (src/repositories/causal_validation.py); runs linked to a causal_paths
    row are keyed by the PATH-derived uuid5, which the API does not expose, so
    fall back to the pair's newest rows written since this job started."""
    if analysis_id:
        qid = uuid.uuid5(uuid.NAMESPACE_URL, f"e2i:causal_query:{analysis_id}")
        found = _psql_rows(f"estimate_id = '{qid}'")
        if found:
            return found
    # Linked suites are written with estimate_source='causal_paths' (545 live rows),
    # so no source filter. Pin ONE suite -- the newest estimate_id for this pair,
    # brand and job window -- never the newest row per test across suites.
    pick = ("select estimate_id from public.causal_validations "
            f"where treatment_variable = '{treatment}' and outcome_variable = '{outcome}' "
            f"and brand = '{BRAND}' and created_at >= '{since_iso}' order by created_at desc limit 1")
    proc = subprocess.run(["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tA", "-c", pick],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"causal_validations lookup failed: {proc.stderr.strip()}")
    suite_id = proc.stdout.strip()
    # The path-derived id is shared by EVERY run of that path (derive_causal_path_estimate_id),
    # so keep the job window on the row read too: this run's suite, nothing older.
    return _psql_rows(f"estimate_id = '{suite_id}' and created_at >= '{since_iso}'") if suite_id else {}

def main(label: str, out_dir: str) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    token = mint_token()
    since_iso = datetime.now(timezone.utc).isoformat()
    q = urllib.parse.urlencode({"dataset": DATASET, "brand": BRAND})
    job = call(token, "POST", f"/causal/discover-effects?{q}", body={})
    job_id = job["job_id"]; print("job", job_id, "total", job["total"], flush=True)
    t0 = time.time()
    while True:
        time.sleep(20)
        job = call(token, "GET", f"/causal/discover-effects/{job_id}")
        done = sum(1 for e in job["effects"] if e["status"] not in ("pending", "running"))
        print(f"  {done}/{job['total']} after {int(time.time()-t0)}s", flush=True)
        if done >= job["total"] or job.get("error"):
            break
        if time.time() - t0 > 3 * 3600:
            print("giving up after 3h", flush=True); break
    rows = []
    for e in job["effects"]:
        detail = call(token, "GET", f"/causal/agent-analyze/{e['analysis_id']}") if e.get("analysis_id") else {}
        ref = detail.get("refutation") or {}
        rows.append({
            "treatment": e["treatment"], "outcome": e["outcome"], "row_status": e["status"],
            "db_tests": db_tests(e.get("analysis_id"), e["treatment"], e["outcome"], since_iso),
            "gate_decision": ref.get("gate_decision"), "run_status": detail.get("status"),
            "expert_review_decision": ref.get("expert_review_decision"), "expert_review_id": ref.get("expert_review_id"),
            "discovered_dag_id": detail.get("discovered_dag_id"), "analysis_id": e.get("analysis_id"),
            "tests": {t["test_name"]: t.get("status") or ("passed" if t.get("passed") else "failed") for t in ref.get("tests", [])},
            "ate": detail.get("ate"), "ci": [detail.get("ate_ci_lower"), detail.get("ate_ci_upper")],
            "warnings": detail.get("warnings", []),
        })
    (out / f"{label}.json").write_text(json.dumps({"job_id": job_id, "image_marker": None, "rows": rows}, indent=2))
    bands = {}
    for r in rows: bands[r["gate_decision"]] = bands.get(r["gate_decision"], 0) + 1
    print("bands", bands)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
