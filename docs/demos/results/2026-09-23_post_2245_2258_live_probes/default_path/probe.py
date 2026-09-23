"""Post-Lane-B live probe (deployed main 814eaa959): ONE default-path agent-analyze on the real
optum_biologic_persistence cohort (discontinued_180d), same request shape as the Lane A cert's
raw_discontinued_180d.json but AFTER #2230 (Lane B) and #2232 (real-backed label) deployed.
Read-only against the API (the run itself is the platform's normal job). Records:
  - data_source (expect "database" now, was "synthetic" under the showcase flag before #2232)
  - Lane E / Lane B channel fields present in the response (feature_role_panel, anchored_confounders,
    approved_structure_roles, structural prior warnings)
  - ATE/CI vs the pre-B cert value for the same outcome (should be unchanged: Lane B is dark on this
    dataset — no feature_manifest_source declared — and Lane E's panel abstains on machine attestations)
"""
from __future__ import annotations

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
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
PRE = Path(sys.argv[2]) if len(sys.argv) > 2 else None  # pre-B raw_discontinued_180d.json


def mint_token() -> str:
    body = json.dumps({"email": os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local"),
                       "password": os.environ["E2I_ADMIN_PASSWORD"]}).encode()
    req = urllib.request.Request(f"{os.environ['SUPABASE_URL']}/auth/v1/token?grant_type=password",
                                 data=body, headers={"apikey": os.environ["SUPABASE_ANON_KEY"],
                                                     "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["access_token"]


def call(token, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def started_at() -> str:
    return subprocess.run(["docker", "inspect", "-f", "{{.Config.Image}} {{.State.StartedAt}}", "e2i_api"],
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    token = mint_token()
    c0 = started_at()
    body = {"treatment_var": "treatment_dupixent", "outcome_var": "discontinued_180d",
            "dataset": "optum_biologic_persistence", "limit": 20000}
    pending = call(token, "POST", "/causal/agent-analyze", body=body)
    aid = pending["analysis_id"]
    t0 = time.time()
    polls = 0
    while True:
        time.sleep(15)
        polls += 1
        job = call(token, "GET", f"/causal/agent-analyze/{aid}")
        print(f"poll {polls} t={time.time()-t0:.0f}s status={job['status']}", flush=True)
        if job["status"] in ("completed", "needs_review", "failed") or time.time() - t0 > 1000:
            break
    job["_probe"] = {"request": body, "submitted_warnings": pending.get("warnings"),
                     "wall_s": round(time.time() - t0, 1), "polls": polls,
                     "container_before": c0, "container_after": started_at(),
                     "ran_at": datetime.now(timezone.utc).isoformat()}
    (OUT / "raw_post_b_discontinued_180d.json").write_text(json.dumps(job, indent=2, default=str))
    keys = sorted(job.keys())
    wanted = ["data_source", "feature_role_panel", "anchored_confounders", "approved_structure_roles",
              "approved_leak_exclusions", "structural_prior", "dag_source", "discovered_confounders"]
    print("status:", job["status"], "| data_source:", job.get("data_source"))
    for k in wanted:
        print(f"  {k:26s} present={k in job} value={str(job.get(k))[:160]}")
    warn = [w for w in (job.get("warnings") or []) if "structural" in w.lower() or "prior" in w.lower()
            or "attest" in w.lower() or "panel" in w.lower()]
    print("structural/prior warnings:", json.dumps(warn)[:600])
    print("ate:", job.get("ate"), "ci:", [job.get("ate_ci_lower"), job.get("ate_ci_upper")], "p:", job.get("p_value"))
    if PRE and PRE.exists():
        pre = json.loads(PRE.read_text())
        print("PRE-B  ate:", pre.get("ate"), "ci:", [pre.get("ate_ci_lower"), pre.get("ate_ci_upper")],
              "data_source:", pre.get("data_source"))
        print("keys only in POST:", sorted(set(keys) - set(pre.keys())))
        print("keys only in PRE :", sorted(set(pre.keys()) - set(keys)))
    return 0 if job["status"] in ("completed", "needs_review") else 2


if __name__ == "__main__":
    sys.exit(main())
