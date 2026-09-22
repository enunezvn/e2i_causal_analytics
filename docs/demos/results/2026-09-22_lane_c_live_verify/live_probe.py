"""Lane C live verify (post-deploy c0860bbf4): the csu_escalation_causal dataset on the
DEPLOYED API. Read-only. Expected: every reader answers 503 (no usable rows) in real mode
with the prod flag E2I_INCLUDE_SYNTHETIC=true set on the container; the catalog labels the
dataset by the dataset rule (synthetic-backed), not by the deployment flag.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
API = os.environ.get("E2I_API_BASE", "https://eznomics.site/api")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def mint_token() -> str:
    body = json.dumps(
        {
            "email": os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local"),
            "password": os.environ["E2I_ADMIN_PASSWORD"],
        }
    ).encode()
    req = urllib.request.Request(
        f"{os.environ['SUPABASE_URL']}/auth/v1/token?grant_type=password",
        data=body,
        headers={"apikey": os.environ["SUPABASE_ANON_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["access_token"]


def call(token: str, method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")[:500]


def main() -> int:
    token = mint_token()
    container = subprocess.run(
        ["docker", "inspect", "e2i_api", "--format", "{{.Config.Image}} {{.State.StartedAt}}"],
        capture_output=True, text=True,
    ).stdout.strip()
    flag = subprocess.run(
        ["docker", "inspect", "e2i_api", "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
        capture_output=True, text=True,
    ).stdout
    flag_line = next((l for l in flag.splitlines() if l.startswith("E2I_INCLUDE_SYNTHETIC=")), "absent")
    results = {"ran_at": datetime.now(timezone.utc).isoformat(), "container": container,
               "E2I_INCLUDE_SYNTHETIC": flag_line, "probes": {}}
    for ds in ("csu_escalation_causal", "optum_biologic_persistence"):
        p = {}
        p["brands"] = call(token, "GET", f"/causal/brands?dataset={ds}")
        p["variables"] = call(token, "GET", f"/causal/variables?dataset={ds}")
        outcome = "discontinued_180d"
        treatment = "treatment_remibrutinib" if ds == "csu_escalation_causal" else "treatment_dupixent"
        p["agent_analyze"] = call(
            token, "POST", "/causal/agent-analyze",
            {"treatment_var": treatment, "outcome_var": outcome, "dataset": ds, "limit": 20000},
        )
        results["probes"][ds] = p
    (OUT / "lane_c_live_probe.json").write_text(json.dumps(results, indent=2, default=str))
    for ds, p in results["probes"].items():
        for k, (code, body) in p.items():
            detail = body.get("detail") if isinstance(body, dict) else body
            print(f"{ds:28s} {k:14s} HTTP {code}  {str(detail)[:200]}")
    print("container:", container)
    print("flag:", flag_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
