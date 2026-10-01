"""#2343 live proof, post-#2339: batch vs single /predict parity on the DEPLOYED API.

Re-run of the pre-#2339 parity script (baseline_pre2339_*.py/.out in this folder) against
the rebuilt sidecar (#2339: sklearn 1.6.1, per-model bundle_sha256, routed named numeric
batches). Adds a patient-grain model pair and a pre->post #2339 comparison.

Read-only apart from routine request telemetry (user_activity_log counter, rate-limit
counters, a Supabase auth token mint). For each target goldstd model: score N real cohort
rows via N single POST /api/models/predict/{m} and ONE POST .../{m}/batch, compare.
Teeth: (a) the same rows through a DIFFERENT goldstd model give clearly different scores
(so a wrong-model batch would be caught), (b) the sidecar's unrouted default (no
model_name) cannot score these rows (n_pred=0).
Pre/post: the single-predict probabilities are compared with the pre-#2339 baseline .out
for the same rows (baseline printed 8 decimals, so the resolution is ~5e-9).
"""
from __future__ import annotations

import json, os, re, subprocess, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
API = os.environ.get("E2I_API_BASE_LOCAL", "http://127.0.0.1:8000/api")
SIDECAR = "http://127.0.0.1:3000"
N = int(os.environ.get("N_ROWS", "8"))
TOL = 1e-9
TEETH_MIN = 1e-3
BASELINE = Path(__file__).with_name("baseline_pre2339_live_batch_single_parity_2343.out")

HCP_KEEP = ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region"]
PAT_KEEP = ["disease_severity", "academic_hcp", "geographic_region", "insurance_type", "age_at_diagnosis",
            "comorbidity_burden", "prior_therapy_lines", "rep_detailing_high", "sample_dropped", "trigger_accepted"]


def hcp_sql(brand):
    return (f"select coalesce(json_agg(t), '[]') from (select hcp_id as row_id, {', '.join(HCP_KEEP)} "
            f"from public.hcp_adoption_goldstd_v where brand = '{brand}' and data_split = 'test' and "
            + " and ".join(f"{c} is not null" for c in HCP_KEEP) + f" order by hcp_id limit {N}) t")


def pat_sql(brand):
    # FeatureBuilder patient path: patient_journeys, brand partition, is_synthetic = true
    return (f"select coalesce(json_agg(t), '[]') from (select patient_journey_id as row_id, {', '.join(PAT_KEEP)} "
            f"from public.patient_journeys where brand = '{brand}' and is_synthetic and data_split = 'test' and "
            + " and ".join(f"{c} is not null" for c in PAT_KEEP) + f" order by patient_journey_id limit {N}) t")


# (model, brand, other_model_for_teeth, keep_columns, rows_sql)
CASES = [
    ("hcp_adoption_kisqali_goldstd_lr_v1", "Kisqali", "hcp_adoption_fabhalta_goldstd_lr_v1", HCP_KEEP, hcp_sql),
    ("hcp_adoption_fabhalta_goldstd_lr_v1", "Fabhalta", "hcp_adoption_kisqali_goldstd_lr_v1", HCP_KEEP, hcp_sql),
    ("initiation_kisqali_goldstd_lr_v1", "Kisqali", "initiation_fabhalta_goldstd_lr_v1", PAT_KEEP, pat_sql),
]


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


def sh(*cmd):
    return subprocess.run(list(cmd), capture_output=True, text=True).stdout.strip()


def psql_json(sql):
    out = subprocess.run(["docker", "exec", "-i", "supabase-db", "psql", "-U", "postgres", "-At", "-c", sql],
                         capture_output=True, text=True, check=True).stdout.strip()
    return json.loads(out) if out else []


def model_info(model):
    st, d = http("POST", f"{SIDECAR}/model_info", {"input_data": {"model_name": model}})
    assert st == 200 and isinstance(d, dict), (st, d)
    return d


def load_baseline():
    """{(model, row_id): single_p} parsed from the pre-#2339 .out."""
    base, model = {}, None
    if not BASELINE.exists():
        return base
    for line in BASELINE.read_text().splitlines():
        m = re.match(r"=== model=(\S+)", line)
        if m:
            model = m.group(1)
            continue
        m = re.match(r"(scv\S+)\s+([0-9.]+)\s", line)
        if m and model:
            base[(model, m.group(1))] = float(m.group(2))
    return base


def main():
    print("run_at", datetime.now(timezone.utc).isoformat())
    print("git_head", sh("git", "-C", "/home/enunez/Projects/e2i_causal_analytics", "rev-parse", "--short", "HEAD"))
    api_image = sh("docker", "inspect", "e2i_api", "--format", "{{.Config.Image}}")
    print("api image_sha:", api_image.rsplit(":", 1)[-1])
    print("sidecar image_id:", sh("docker", "inspect", "e2i_bentoml", "--format", "{{.Image}}"),
          "started_at:", sh("docker", "inspect", "e2i_bentoml", "--format", "{{.State.StartedAt}}"))
    print("sidecar sklearn:", sh("docker", "exec", "e2i_bentoml", "python", "-c", "import sklearn;print(sklearn.__version__)"))
    for m in sorted({c[0] for c in CASES} | {c[2] for c in CASES}):
        mi = model_info(m)
        print(f"model_info {m}: model_id={mi.get('model_id')} bundle_sha256={mi.get('bundle_sha256')} "
              f"keep_columns={mi.get('keep_columns')}")
    baseline = load_baseline()
    H = {"Authorization": f"Bearer {mint_token()}"}
    verdict = True
    max_prepost = 0.0
    n_prepost = 0
    for model, brand, other, keep, sqlf in CASES:
        mi = model_info(model)
        assert mi.get("keep_columns") == keep, (model, mi.get("keep_columns"), keep)
        # predict routes are rate-limited to 10 req/60s per client; each block is exactly N+2 = 10
        time.sleep(65)
        rows = psql_json(sqlf(brand))
        feats = [{k: r[k] for k in keep} for r in rows]
        print(f"\n=== model={model} brand={brand} n={len(rows)} (test split, ordered by row id)")
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
        st, ob = http("POST", f"{API}/models/predict/{other}/batch", {"instances": [{"features": f} for f in feats]}, H)
        assert st == 200, (st, ob)
        other_p = [p["probabilities"]["positive_class"] for p in ob["predictions"]]
        print(f"{'row_id':<18}{'single_p':>12}{'batch_p':>12}{'absdiff':>11}{'s_pred':>7}{'b_pred':>7}{'other_p':>12}{'pre2339_p':>12}{'pre_post_d':>11}")
        maxd = maxd_other = 0.0
        for r, s, b, o in zip(rows, single, batch, other_p):
            d = abs(s[1] - b[1]); maxd = max(maxd, d); maxd_other = max(maxd_other, abs(b[1] - o))
            pre = baseline.get((model, r["row_id"]))
            pp = "" if pre is None else f"{abs(s[1] - pre):.2e}"
            if pre is not None:
                max_prepost = max(max_prepost, abs(s[1] - pre)); n_prepost += 1
            pre_s = "n/a" if pre is None else f"{pre:.8f}"
            print(f"{r['row_id']:<18}{s[1]:>12.8f}{b[1]:>12.8f}{d:>11.2e}{s[0]!s:>7}{b[0]!s:>7}{o:>12.8f}{pre_s:>12}{pp:>11}")
        pred_eq = all(s[0] == b[0] for s, b in zip(single, batch))
        ver = {s[2] for s in single}
        bver = sorted({str(b[2]) for b in batch})
        print(f"max|single-batch|={maxd:.3e}  predictions_equal={pred_eq}  single model_version={sorted(map(str, ver))}  "
              f"batch model_version={bver} (#2351: batch rows carry no model_version)")
        print(f"teeth: max|batch({model}) - batch({other})|={maxd_other:.4f}  (must be > {TEETH_MIN})")
        ok = len(rows) == N and maxd <= TOL and pred_eq and ver == {model} and maxd_other > TEETH_MIN
        print("model_verdict", "PASS" if ok else "FAIL")
        verdict &= ok
        st, d = http("POST", f"{SIDECAR}/predict_batch", {"input_data": {"batch_id": "parity-2343-default", "raw_features": feats}})
        print(f"teeth: sidecar default (no model_name) raw batch: status={st} "
              f"error={d.get('error') if isinstance(d, dict) else d!r} "
              f"n_pred={len(d.get('predictions') or []) if isinstance(d, dict) else 'n/a'}")
    print(f"\npre->post #2339 single-predict: rows_compared={n_prepost} max|post-pre|={max_prepost:.3e} "
          f"(baseline printed 8 dp, resolution ~5e-9)")
    prepost_ok = n_prepost == 2 * N and max_prepost <= 1e-8
    print("pre_post_verdict", "PASS" if prepost_ok else "FAIL")
    verdict &= prepost_ok
    print("\nVERDICT", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
