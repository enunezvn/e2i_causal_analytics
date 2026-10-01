"""#2343 item 3 live proof: batch vs single /predict parity on the DEPLOYED API.

Read-only apart from routine request telemetry (user_activity_log counter, rate-limit
counters, a Supabase auth token mint). For each target goldstd model: score N real
cohort rows via N single POST /api/models/predict/{m} and ONE POST .../{m}/batch, compare.
Teeth: (a) same rows through a DIFFERENT goldstd model give different scores (so a
wrong-model batch would be caught), (b) the sidecar's unrouted default (no model_name)
cannot score these rows at all (model_id no_model).
"""
from __future__ import annotations

import json, os, subprocess, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
API = os.environ.get("E2I_API_BASE_LOCAL", "http://127.0.0.1:8000/api")
SIDECAR = "http://127.0.0.1:3000"
N = int(os.environ.get("N_ROWS", "8"))
TOL = 1e-9
KEEP = ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region"]
PAIRS = [("Kisqali", "hcp_adoption_kisqali_goldstd_lr_v1", "hcp_adoption_fabhalta_goldstd_lr_v1"),
         ("Fabhalta", "hcp_adoption_fabhalta_goldstd_lr_v1", "hcp_adoption_kisqali_goldstd_lr_v1")]


def http(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")[:500]


def mint_token():
    st, body = http("POST", f"{os.environ['SUPABASE_URL']}/auth/v1/token?grant_type=password",
                    {"email": os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local"),
                     "password": os.environ["E2I_ADMIN_PASSWORD"]},
                    {"apikey": os.environ["SUPABASE_ANON_KEY"]})
    assert st == 200, (st, body)
    return body["access_token"]


def psql_json(sql):
    out = subprocess.run(["docker", "exec", "-i", "supabase-db", "psql", "-U", "postgres", "-At", "-c", sql],
                         capture_output=True, text=True, check=True).stdout.strip()
    return json.loads(out) if out else []


def rows_for(brand):
    cols = ", ".join(KEEP)
    sql = (f"select coalesce(json_agg(t), '[]') from (select hcp_id, {cols} from public.hcp_adoption_goldstd_v "
           f"where brand = '{brand}' and data_split = 'test' and "
           + " and ".join(f"{c} is not null" for c in KEEP)
           + f" order by hcp_id limit {N}) t")
    rows = psql_json(sql)
    for r in rows:
        r["peer_influence_score"] = float(r["peer_influence_score"])
    return rows


def main():
    print("run_at", datetime.now(timezone.utc).isoformat())
    print("git_head", subprocess.run(["git", "-C", "/home/enunez/Projects/e2i_causal_analytics", "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip())
    H = {"Authorization": f"Bearer {mint_token()}"}
    verdict = True
    for brand, model, other in PAIRS:
        # predict routes are rate-limited to 10 req/60s per client; each block is exactly 10
        time.sleep(65)
        rows = rows_for(brand)
        feats = [{k: r[k] for k in KEEP} for r in rows]
        print(f"\n=== model={model} brand={brand} n={len(rows)} (test split, ordered by hcp_id)")
        single = []
        for f in feats:
            st, b = http("POST", f"{API}/models/predict/{model}", {"features": f, "return_probabilities": True}, H)
            assert st == 200, (st, b)
            single.append((b["prediction"], b["probabilities"]["positive_class"], b["model_version"]))
        st, bb = http("POST", f"{API}/models/predict/{model}/batch",
                      {"instances": [{"features": f} for f in feats]}, H)
        assert st == 200, (st, bb)
        batch = [(p["prediction"], p["probabilities"]["positive_class"], p["model_version"]) for p in bb["predictions"]]
        assert len(batch) == len(single), (len(batch), len(single))
        # teeth (a): same rows, a different goldstd model, via the batch route
        st, ob = http("POST", f"{API}/models/predict/{other}/batch", {"instances": [{"features": f} for f in feats]}, H)
        assert st == 200, (st, ob)
        other_p = [p["probabilities"]["positive_class"] for p in ob["predictions"]]
        print(f"{'hcp_id':<12}{'single_p':>12}{'batch_p':>12}{'absdiff':>11}{'s_pred':>7}{'b_pred':>7}{'other_p':>12}")
        maxd = 0.0
        maxd_other = 0.0
        for r, s, b, o in zip(rows, single, batch, other_p):
            d = abs(s[1] - b[1]); maxd = max(maxd, d); maxd_other = max(maxd_other, abs(b[1] - o))
            print(f"{r['hcp_id']:<12}{s[1]:>12.8f}{b[1]:>12.8f}{d:>11.2e}{s[0]!s:>7}{b[0]!s:>7}{o:>12.8f}")
        pred_eq = all(s[0] == b[0] for s, b in zip(single, batch))
        ver = {s[2] for s in single}
        bver = {str(b[2]) for b in batch}
        print(f"max|single-batch|={maxd:.3e}  predictions_equal={pred_eq}  single model_version={sorted(map(str, ver))}  batch model_version={sorted(bver)} (sidecar BatchPredictionOutput has no model_id)")
        print(f"max|batch({model}) - batch({other})|={maxd_other:.4f}  other model_version={ob['predictions'][0]['model_version']}")
        ok = maxd <= TOL and pred_eq and ver == {model} and maxd_other > 1e-3
        print("model_verdict", "PASS" if ok else "FAIL")
        verdict &= ok
        # teeth (b): unrouted default on the sidecar (what the pre-#2346 route effectively asked for)
        st, d = http("POST", f"{SIDECAR}/predict_batch", {"input_data": {"batch_id": "parity-2343-default", "raw_features": feats}})
        print(f"sidecar default (no model_name) raw batch: status={st} error={d.get('error') if isinstance(d, dict) else d!r} "
              f"n_pred={len(d.get('predictions') or []) if isinstance(d, dict) else 'n/a'}")
    print("\nVERDICT", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
