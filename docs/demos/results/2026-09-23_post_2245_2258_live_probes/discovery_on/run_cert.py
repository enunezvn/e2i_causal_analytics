#!/usr/bin/env python3
"""Lane A live API cert (plan Task 12 Step 4): submit the causal_impact agent run for
the real `optum_biologic_persistence` cohort through the DEPLOYED API, one outcome at a
time, and record every raw response.

Usage (from the evidence dir; reads .env for the admin credentials):
  run_cert.py                       # the four outcomes, sequentially, auto_discover omitted
  run_cert.py --outcomes discontinued_180d --estimator LinearDML
  run_cert.py --probe-discovery     # ONE run, discontinued_180d, auto_discover=true -> live_probes/

Token + call helpers copied from docs/demos/results/2026-09-09_expert_review_loop/run_discovery.py
lines 13-27. Outcomes never run in parallel (one heavy-compute slot per worker).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
API = os.environ.get("E2I_API_BASE", "https://eznomics.site/api")
HERE = Path(__file__).resolve().parent

OUTCOMES = [
    "persistent_at_180d_g28",
    "discontinued_180d",
    "biologic_switch_180d_flag",
    "persistent_at_180d",
]
POLL_CAP_S = 1000
POLL_EVERY_S = 15


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


def container_started_at() -> str:
    return subprocess.run(["docker", "inspect", "-f", "{{.State.StartedAt}}", "e2i_api"],
                          capture_output=True, text=True).stdout.strip()


def run_one(token: str, outcome: str, out_dir: Path, estimator: str | None = None,
            auto_discover: bool | None = None) -> dict:
    body = {"treatment_var": "treatment_dupixent", "outcome_var": outcome,
            "dataset": "optum_biologic_persistence", "limit": 20000}
    if estimator:
        body["estimator"] = estimator
    if auto_discover is not None:
        body["auto_discover"] = auto_discover
    # auto_discover OMITTED by default: the per-dataset default (off) is under test
    submitted_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        pending = call(token, "POST", "/causal/agent-analyze", body=body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        job = {"status": "submit_failed", "http_status": exc.code, "detail": detail,
               "_cert": {"request": body, "submitted_at": submitted_at, "wall_s": round(time.time() - t0, 1),
                         "container_started_at": container_started_at()}}
        (out_dir / f"raw_{outcome}.json").write_text(json.dumps(job, indent=2))
        print(f"[{outcome}] SUBMIT FAILED http={exc.code} detail={detail[:300]}", flush=True)
        return job
    aid = pending["analysis_id"]
    print(f"[{outcome}] submitted analysis_id={aid} status={pending.get('status')} "
          f"warnings={pending.get('warnings')}", flush=True)
    polls = 0
    while True:
        time.sleep(POLL_EVERY_S)
        polls += 1
        job = call(token, "GET", f"/causal/agent-analyze/{aid}")
        elapsed = time.time() - t0
        print(f"[{outcome}] poll {polls} t={elapsed:.0f}s status={job['status']}", flush=True)
        if job["status"] in ("completed", "needs_review", "failed") or elapsed > POLL_CAP_S:
            break
    job["_cert"] = {"request": body, "submitted_at": submitted_at, "submitted_warnings": pending.get("warnings"),
                    "wall_s": round(time.time() - t0, 1), "polls": polls, "poll_cap_s": POLL_CAP_S,
                    "poll_cap_hit": job["status"] not in ("completed", "needs_review", "failed"),
                    "container_started_at": container_started_at(), "api_base": API}
    (out_dir / f"raw_{outcome}.json").write_text(json.dumps(job, indent=2))
    r = job.get("refutation") or {}
    print(f"[{outcome}] DONE status={job['status']} n_rows={job.get('n_rows')} est={job.get('selected_estimator')} "
          f"ate={job.get('ate')} ci=[{job.get('ate_ci_lower')}, {job.get('ate_ci_upper')}] p={job.get('p_value')} "
          f"refut={r.get('tests_passed')}/{r.get('tests_total')} gate={r.get('gate_decision')} "
          f"evalue={r.get('sensitivity_e_value')} dag_source={job.get('dag_source')} wall={job['_cert']['wall_s']}s",
          flush=True)
    return job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", nargs="*", default=OUTCOMES)
    ap.add_argument("--estimator", default=None, help="force an estimator (only if Task 9 showed Auto exceeds the cap)")
    ap.add_argument("--probe-discovery", action="store_true",
                    help="ONE run of discontinued_180d with auto_discover=true, saved under live_probes/")
    args = ap.parse_args()
    token = mint_token()
    if args.probe_discovery:
        out_dir = HERE / "live_probes"
        out_dir.mkdir(exist_ok=True)
        job = run_one(token, "discontinued_180d", out_dir, auto_discover=True)
        return 0 if job["status"] in ("completed", "needs_review", "failed") else 2
    rc = 0
    for outcome in args.outcomes:
        job = run_one(token, outcome, HERE, estimator=args.estimator)
        if job["status"] not in ("completed", "needs_review"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
