"""Why does the twin report a significant effect for the planted-NULL channel on Remibrutinib?

READ-ONLY against prod: every DB call is a ``.select()`` (the twin's own ``load_cohort_frame``
through an async client, plus the backfill script's paged readers). No Redis, no writes.

Stages (each caches to the scratch dir so a re-run resumes):
  fetch      live frames: load_cohort_frame per brand, hcp_profiles centrality/specialty,
             hcp_brand_adoption (adopted + treatment_arm), planted rollups.
  decompose  re-derive the live labels with derive(seed=427) (the executed re-plant), check
             they equal the live table, and split the adoption logit into its DGP terms
             (centrality, specialty affinity, channel shift, treatment-arm term, logit noise,
             Bernoulli residual).  Report how each term differs between rep_training tbin=1/0,
             raw and adjusted for the estimator's X+W (OLS), plus correlations (measurement 1).
  refit      estimate_cohort_effect on every channel x brand with W = default, +7 other
             channel tbins, +DGP drivers (centrality_z, affinity, treatment_arm), +both
             (measurement 2 and 4).
  placebo    permute rep_training_score within region, refit, count DR CIs excluding 0
             (measurement 3).
  seeds      the CHANCE test conditional on the design: re-draw adopted with fresh derive
             seeds (same rollups, centrality, specialty; new arm + noise) and refit the null on
             Remibrutinib; where does seed 427's +0.047 fall?
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import time
import warnings
from datetime import date

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WORKTREE = "/home/enunez/Projects/e2i_causal_analytics/.worktrees/t2-evidence"
sys.path.insert(0, WORKTREE)
import src  # noqa: E402

assert src.__file__.startswith(WORKTREE), f"WRONG TREE: {src.__file__}"

import logging  # noqa: E402

logging.disable(logging.WARNING)

from scripts.backfill_hcp_treatment_arm import (  # noqa: E402
    derive,
    fetch_centrality,
    fetch_channel_rollups,
    fetch_live_adoption,
)
from src.data.per_hcp_cohort_collapse import CHANNEL_COLUMNS  # noqa: E402
from src.digital_twin.effect import cohort_causal_estimator as cce  # noqa: E402
from src.digital_twin.effect.cohort_loader import load_cohort_frame  # noqa: E402
from src.ml.synthetic.generators import hcp_adoption_artifact as gen  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get(
    "REMI_CACHE",
    "/tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/"
    "b1d41f57-1169-44e4-af12-96c28babe7a7/scratchpad/remi_cache",
)
os.makedirs(CACHE, exist_ok=True)
BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
NULL = "rep_training_score"
LIVE_SEED = 427
RUN_DATE = date(2026, 9, 23)  # the executed re-plant's run date (owner log line 2)
Z = 1.959963984540054
DRIVERS = ["centrality_z", "affinity", "treatment_arm"]


def say(*a):
    print(*a, flush=True)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


# --------------------------------------------------------------------------------- fetch
async def _load_frames():
    from src.memory.services.factories import loop_scoped_async_supabase_client

    frames = {}
    async with loop_scoped_async_supabase_client() as aclient:
        for b in BRANDS:
            frames[b] = await load_cohort_frame(aclient, b)
    return frames


def fetch():
    p = f"{CACHE}/loader_Remibrutinib.parquet"
    if os.path.exists(p):
        return
    from src.repositories import get_supabase_client

    client = get_supabase_client()
    frames = asyncio.run(_load_frames())
    for b, f in frames.items():
        f.to_parquet(f"{CACHE}/loader_{b}.parquet", index=False)
        say(f"loader {b}: {len(f)} pairs, adopted non-null {int(f['adopted'].notna().sum())}")
    fetch_centrality(client).to_parquet(f"{CACHE}/centrality.parquet", index=False)
    fetch_live_adoption(client).to_parquet(f"{CACHE}/live_adoption.parquet", index=False)
    fetch_channel_rollups(client).to_parquet(f"{CACHE}/rollups.parquet", index=False)


def planted_rollups():
    r = pd.read_parquet(f"{CACHE}/rollups.parquet")
    # The ETL rows added after the plant carry every channel NULL (the loader drops them too);
    # derive() refuses nulls, and the executed re-plant ran before they existed.
    keep = r[list(CHANNEL_COLUMNS)].notna().any(axis=1)
    return r[keep].reset_index(drop=True), int((~keep).sum())


def derived(seed: int) -> pd.DataFrame:
    cen = pd.read_parquet(f"{CACHE}/centrality.parquet")
    roll, _ = planted_rollups()
    return derive(cen, seed=seed, channel_rollups=roll, run_date=RUN_DATE)


# ----------------------------------------------------------------------------- analysis frame
def analysis_frame(brand: str, d: pd.DataFrame) -> pd.DataFrame:
    """The twin's own loader frame (usable rows) + the DGP terms of the derive() row."""
    cen = pd.read_parquet(f"{CACHE}/centrality.parquet")
    pis = cen["peer_influence_score"].to_numpy(float)
    cz = pd.Series((pis - pis.mean()) / pis.std(), index=cen["hcp_id"])
    f = pd.read_parquet(f"{CACHE}/loader_{brand}.parquet")
    f = f[f["adopted"].notna()].copy()
    db = d[d["brand"] == brand].set_index("hcp_id")
    f["centrality_z"] = cz.reindex(f["hcp_id"]).to_numpy()
    f["treatment_arm"] = db["treatment_arm"].reindex(f["hcp_id"]).to_numpy(float)
    f["derived_adopted"] = db["adopted"].reindex(f["hcp_id"]).to_numpy(float)
    f["channel_shift"] = db["channel_shift"].reindex(f["hcp_id"]).to_numpy(float)
    f["logit"] = db["adoption_logit"].reindex(f["hcp_id"]).to_numpy(float)
    spec = cen.set_index("hcp_id")["specialty"].reindex(f["hcp_id"]).tolist()
    f["affinity"] = gen._specialty_affinity(brand, spec, len(f))
    seg = np.where(
        f["centrality_z"] > 0.5, "high_influence",
        np.where(f["centrality_z"] > -0.5, "medium_influence", "low_influence"),
    )
    seg_t = np.array([gen._ADOPT_TREATMENT_LOGIT[s] for s in seg])
    f["arm_term"] = gen._BRAND_ADOPT_SCALE[brand] * seg_t * f["treatment_arm"]
    f["cz_term"] = gen._ADOPT_CENTRALITY_SLOPE * f["centrality_z"]
    f["logit_noise"] = f["logit"] - (
        gen._ADOPT_INTERCEPT + f["cz_term"] + f["affinity"] + f["channel_shift"] + f["arm_term"]
    )
    f["p_true"] = 1 / (1 + np.exp(-f["logit"]))
    f["bern_resid"] = f["adopted"] - f["p_true"]
    for c in CHANNEL_COLUMNS:  # the estimator's contrast: median over the usable rows
        v = f[c].astype(float)
        f[f"tb_{c}"] = (v > v.median()).astype(float)
    return f.reset_index(drop=True)


def xw_matrix(f: pd.DataFrame) -> np.ndarray:
    x = pd.get_dummies(f[["region", "specialty"]].fillna("unknown"), dtype=float)
    w = np.column_stack([f["market_share"], np.log1p(f["triggers_total_count"].clip(lower=0))])
    return np.column_stack([np.ones(len(f)), x.to_numpy()[:, 1:], w])


def ols_coef(y: np.ndarray, t: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    m = np.column_stack([t, base])
    beta, *_ = np.linalg.lstsq(m, y, rcond=None)
    resid = y - m @ beta
    cov = np.linalg.pinv(m.T @ m) * (resid @ resid) / (len(y) - m.shape[1])
    return float(beta[0]), float(np.sqrt(cov[0, 0]))


def decompose():
    d = derived(LIVE_SEED)
    live = pd.read_parquet(f"{CACHE}/live_adoption.parquet")
    rows, corr_rows = [], []
    for b in BRANDS:
        f = analysis_frame(b, d)
        lv = live[live["brand"] == b].set_index("hcp_id")
        match_adopted = float((lv["adopted"].reindex(f["hcp_id"]).astype(float).to_numpy()
                               == f["adopted"].to_numpy()).mean())
        match_derived = float((f["derived_adopted"] == f["adopted"]).mean())
        match_arm = float((lv["treatment_arm"].reindex(f["hcp_id"]).astype(float).to_numpy()
                           == f["treatment_arm"].to_numpy()).mean())
        say(f"{b}: n={len(f)} loader.adopted==live {match_adopted:.4f}  "
            f"derive(427).adopted==loader {match_derived:.4f}  arm==live {match_arm:.4f}")
        t = f[f"tb_{NULL}"].to_numpy()
        base = xw_matrix(f)
        # How much of the adopted contrast each DGP term carries (terms on the PROBABILITY
        # scale are not additive; the logit terms are reported on the logit scale, plus the
        # Bernoulli residual and p_true on the probability scale).
        for term in ["adopted", "p_true", "bern_resid", "logit", "cz_term", "affinity",
                     "channel_shift", "arm_term", "logit_noise", "treatment_arm",
                     "n_metric_rows"]:
            y = f[term].to_numpy(float)
            raw = float(y[t == 1].mean() - y[t == 0].mean())
            adj, se = ols_coef(y, t, base)
            rows.append({"brand": b, "term": term, "raw_diff": raw, "adj_diff": adj,
                         "adj_se": se, "adj_z": adj / se if se else np.nan,
                         "n": len(f), "match_derived": match_derived,
                         "match_arm_live": match_arm})
        # Correlations of the null tbin with everything (measurement 1).
        others = {f"tb_{c}": f[f"tb_{c}"] for c in CHANNEL_COLUMNS if c != NULL}
        others.update({k: f[k] for k in ["treatment_arm", "centrality_z", "affinity",
                                         "channel_shift", "logit_noise", "bern_resid",
                                         "n_metric_rows", "market_share"]})
        others["log_triggers"] = np.log1p(f["triggers_total_count"])
        for reg in sorted(f["region"].unique()):
            others[f"region={reg}"] = (f["region"] == reg).astype(float)
        for sp in sorted(f["specialty"].fillna("unknown").unique()):
            others[f"specialty={sp}"] = (f["specialty"].fillna("unknown") == sp).astype(float)
        for k, v in others.items():
            v = pd.Series(np.asarray(v, float))
            raw_r = float(np.corrcoef(t, v)[0, 1])
            # partial r given the estimator's X+W (residualise both on base)
            rt = t - base @ np.linalg.lstsq(base, t, rcond=None)[0]
            rv = v.to_numpy() - base @ np.linalg.lstsq(base, v.to_numpy(), rcond=None)[0]
            part = float(np.corrcoef(rt, rv)[0, 1]) if rv.std() > 1e-12 else np.nan
            corr_rows.append({"brand": b, "var": k, "r": raw_r, "partial_r_given_XW": part,
                              "n": len(f)})
        f.to_parquet(f"{CACHE}/analysis_{b}.parquet", index=False)
    pd.DataFrame(rows).to_csv(f"{OUT}/decompose.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(f"{OUT}/null_tbin_correlations.csv", index=False)


# ---------------------------------------------------------------------------------- refits
def fit(f: pd.DataFrame, channel: str, extra=()):
    t0 = time.time()
    r = cce.estimate_cohort_effect(
        f, channel, confounders=(*cce.DEFAULT_CONFOUNDERS, *extra), seed=42
    )
    return {"ate": r.ate, "lo": r.ate_ci_lower, "hi": r.ate_ci_upper, "se": r.ate_stderr,
            "excl0": bool(r.ate_ci_lower > 0 or r.ate_ci_upper < 0),
            "sec": round(time.time() - t0, 1)}


def refit():
    from src.data.per_hcp_cohort_columns import (
        ADOPTION_CHANNEL_PLANTED_RD,
        INTERVENTION_TREATMENT_MAP,
    )

    planted = {INTERVENTION_TREATMENT_MAP[k]: float(v) for k, v in ADOPTION_CHANNEL_PLANTED_RD.items()}
    path = f"{OUT}/refit_augmented_w.csv"
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["brand"], r["channel"], r["spec"]) for r in rows}
    for b in BRANDS:
        f = pd.read_parquet(f"{CACHE}/analysis_{b}.parquet")
        for ch in CHANNEL_COLUMNS:
            other_tb = [f"tb_{c}" for c in CHANNEL_COLUMNS if c != ch]
            specs = {"default": (), "+other7_tbins": other_tb, "+dgp_drivers": DRIVERS,
                     "+both": [*other_tb, *DRIVERS]}
            for name, extra in specs.items():
                if (b, ch, name) in done:
                    continue
                rec = {"brand": b, "channel": ch, "spec": name, "planted": planted[ch]}
                rec.update(fit(f, ch, extra))
                rec["err"] = rec["ate"] - planted[ch]
                rec["covers_planted"] = bool(rec["lo"] <= planted[ch] <= rec["hi"])
                rows.append(rec)
                pd.DataFrame(rows).to_csv(path, index=False)
                say(f"{b:13s} {ch:28s} {name:14s} ate={rec['ate']:+.4f} "
                    f"({rec['lo']:+.4f},{rec['hi']:+.4f}) err={rec['err']:+.4f} "
                    f"{rec['sec']}s rss={rss():.2f}G")


def placebo(k_by_brand=(("Remibrutinib", 100), ("Fabhalta", 40), ("Kisqali", 40))):
    path = f"{OUT}/placebo_within_region.csv"
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["brand"], r["k"]) for r in rows}
    for b, K in k_by_brand:
        f = pd.read_parquet(f"{CACHE}/analysis_{b}.parquet")
        for k in range(K):
            if (b, k) in done:
                continue
            rng = np.random.default_rng([2026, 928, k])
            g = f.copy()
            perm = g[NULL].to_numpy().copy()
            for reg in g["region"].unique():
                m = (g["region"] == reg).to_numpy()
                perm[m] = rng.permutation(perm[m])
            g[NULL] = perm
            rec = {"brand": b, "k": k}
            rec.update(fit(g, NULL))
            rows.append(rec)
            pd.DataFrame(rows).to_csv(path, index=False)
            say(f"placebo {b} k={k} ate={rec['ate']:+.4f} excl0={rec['excl0']} rss={rss():.2f}G")


def seeds(seed_list=tuple(range(2001, 2101))):
    """Conditional-on-design null distribution for Remibrutinib: fresh arm + noise per seed."""
    path = f"{OUT}/seed_null_remibrutinib.csv"
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {r["seed"] for r in rows}
    base = pd.read_parquet(f"{CACHE}/analysis_Remibrutinib.parquet")
    for s in (LIVE_SEED, *seed_list):
        if s in done:
            continue
        d = derived(s)
        db = d[d["brand"] == "Remibrutinib"].set_index("hcp_id")
        g = base.copy()
        g["adopted"] = db["adopted"].reindex(g["hcp_id"]).to_numpy(float)
        rec = {"seed": s}
        rec.update(fit(g, NULL))
        rows.append(rec)
        pd.DataFrame(rows).to_csv(path, index=False)
        say(f"seed {s} ate={rec['ate']:+.4f} excl0={rec['excl0']} rss={rss():.2f}G")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["fetch", "decompose", "refit", "placebo", "seeds"]
    for st in stages:
        say(f"===== {st} =====")
        globals()[st]()
    say("DONE")
