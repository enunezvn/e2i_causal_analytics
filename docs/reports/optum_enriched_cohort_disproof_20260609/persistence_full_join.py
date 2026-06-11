"""The true mart x enriched join for persistence: patient features (enriched) +
FULL old-mart optum_hcp referral-network features (joined via primary_hcp_npi -> npi).
Does the richer HCP context clear the 0.65 commercial AUC floor?

Memory-safe: build cohort first, then read the old optum_hcp grain filtered to just the
cohort's NPIs. Temporal split, GBM, bootstrap CI, commercial bar, leak sniff.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.parquet as pq, pyarrow.dataset as ds, pyarrow.compute as pc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support
from src.data.manifests import MART_SAFE_FEATURES
import scripts.convert_optum_mart as cm

ENR = "data/rwd/Optum_Parquet/Optum_enriched.parquet"
OLD = "data/rwd/Optum_Parquet/Optum.parquet"
SAFE = [c for c in MART_SAFE_FEATURES if c not in ("geographic_region", "enrollment_duration_days")]
NET = ["influence_network_size", "shared_patient_edge_count", "shared_patient_weight",
       "max_shared_patient_edge_weight", "shared_patient_kol_score_pct", "referral_in_degree",
       "referral_in_patient_count", "max_referral_in_edge_weight", "referral_out_degree",
       "referral_out_patient_count", "referral_kol_score_pct", "kol_score_100pt", "kol_score",
       "specialty_group", "medical_claim_count", "medical_patient_count", "treated_patient_count",
       "specialty_primary"]
LOG1P = ["influence_network_size", "shared_patient_edge_count", "shared_patient_weight",
         "max_shared_patient_edge_weight", "referral_in_degree", "referral_in_patient_count",
         "max_referral_in_edge_weight", "referral_out_degree", "referral_out_patient_count",
         "medical_claim_count", "medical_patient_count", "treated_patient_count"]
COHORT_COLS = ["patid", "entity_type", "index_biologic_brand", "treatment_start_date", "index_date",
               "claim_record_count", "elig_start_date", "zipcode_5", "last_observed_date",
               "last_coverage_end", "max_internal_gap_days", "terminal_gap_days", "primary_hcp_npi"]
GBM = dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=80,
           l2_regularization=2.0, class_weight="balanced", early_stopping=True, random_state=0)
RNG = np.random.RandomState(0)


def enc(frame):
    X = frame.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
        X[c] = X[c].astype("float32")
    return X


def patient_feats(frame):
    X = frame[[c for c in SAFE if c in frame.columns]].copy()
    X["enrollment_duration_days"] = (pd.to_datetime(frame["index_date"]) - pd.to_datetime(frame["elig_start_date"])).dt.days
    X["zip3_region"] = frame["zipcode_5"].astype("string").str.slice(0, 3)
    return enc(X)


def auc_of(Xtr, ytr, Xte, yte):
    m = HistGradientBoostingClassifier(**GBM); m.fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), m.predict_proba(Xte)[:, 1]


def boot_ci(y, p, n=2000):
    a = []; idx = np.arange(len(y))
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) > 1: a.append(roc_auc_score(y[s], p[s]))
    return np.percentile(a, [2.5, 97.5])


def main():
    sch = set(pq.ParquetFile(ENR).schema_arrow.names)
    need = [c for c in sorted(set(SAFE) | set(COHORT_COLS)) if c in sch]
    df = pq.read_table(ENR, columns=need).to_pandas()
    coh, _ = cm.select_persistence_cohort(df); del df
    coh = coh.reset_index(drop=True)
    coh["npi"] = coh["primary_hcp_npi"].astype("string")
    npi_set = [x for x in coh["npi"].dropna().unique().tolist() if x not in ("na", "")]
    print(f"persistence n={len(coh):,}  distinct primary HCP NPI={len(npi_set):,}")

    # Read old optum_hcp network features for just these NPIs
    dset = ds.dataset(OLD, format="parquet")
    flt = (pc.field("entity_type") == "optum_hcp") & pc.field("npi").isin(npi_set)
    hcp = dset.to_table(columns=["npi"] + NET, filter=flt).to_pandas()
    hcp["npi"] = hcp["npi"].astype("string")
    hcp = hcp.drop_duplicates("npi", keep="first")
    print(f"matched HCP network rows: {len(hcp):,}")
    for c in LOG1P:
        if c in hcp: hcp[c] = np.log1p(hcp[c].clip(lower=0))

    merged = coh.merge(hcp, on="npi", how="left")
    cov = merged["kol_score"].notna().mean()
    print(f"persistence patients with joined network features: {cov:.1%}")

    y = merged[cm.TARGET_PERSISTENT].astype(int).values
    Xp = patient_feats(merged)
    Xn = enc(merged[[c for c in NET if c in merged.columns]])
    Xc = pd.concat([Xp, Xn], axis=1)
    ts = pd.to_datetime(merged["treatment_start_date"]).values
    order = np.argsort(ts); n = len(order); tr, te = order[:int(.8*n)], order[int(.8*n):]
    yt, ye = y[tr], y[te]
    ap, _ = auc_of(Xp.iloc[tr], yt, Xp.iloc[te], ye)
    an, _ = auc_of(Xn.iloc[tr], yt, Xn.iloc[te], ye)
    ac, pc_ = auc_of(Xc.iloc[tr], yt, Xc.iloc[te], ye)
    lo, hi = boot_ci(ye, pc_)

    sf = {}
    for c in Xn.columns:
        col = Xn[c].fillna(Xn[c].median())
        if col.nunique() > 1: sf[c] = round(max(roc_auc_score(y, col), 1 - roc_auc_score(y, col)), 3)
    top = sorted(sf.items(), key=lambda kv: -kv[1])[:5]

    # commercial bar on combined
    def nb(yy, pp, pt):
        yp = (pp >= pt).astype(int); nn = len(yy)
        tp = int(((yp == 1) & (yy == 1)).sum()); fp = int(((yp == 1) & (yy == 0)).sum())
        return tp/nn - (fp/nn)*(pt/(1-pt))
    grid = np.unique(np.quantile(pc_, np.linspace(0.5, 0.999, 400)))
    bmcc = -1; rec_ok = None
    for t in grid:
        yp = (pc_ >= t).astype(int)
        if yp.sum() == 0 or len(np.unique(yp)) < 2: continue
        _, rc, _, _ = precision_recall_fscore_support(ye, yp, average="binary", zero_division=0)
        bmcc = max(bmcc, matthews_corrcoef(ye, yp))
        if rc >= 0.50: rec_ok = True
    g = [ac >= 0.65, rec_ok is not None, bmcc >= 0.10, nb(ye, pc_, 0.10) > 0]

    print("="*74)
    print(f"PERSISTENCE + FULL old-mart HCP network (temporal split, test n={len(te):,}, pos={int(ye.sum())})")
    print(f"  patient-only AUC      = {ap:.4f}")
    print(f"  full-HCP-network only = {an:.4f}")
    print(f"  COMBINED AUC          = {ac:.4f}  (95% CI {lo:.3f}-{hi:.3f})   floor 0.65 -> "
          f"{'CLEARS' if ac >= 0.65 else 'MISSES'}")
    print(f"  prior (enriched subset HCP) combined was 0.631")
    print(f"  top network single-feat AUC (leak sniff, want <0.80): {top}")
    print(f"  commercial bar: AUC {'P' if g[0] else 'F'} recall {'P' if g[1] else 'F'} "
          f"MCC {bmcc:.3f} {'P' if g[2] else 'F'} NB@.10 {'P' if g[3] else 'F'} -> "
          f"{sum(g)}/4 {'DEPLOYABLE' if all(g) else 'BLOCKED'}")
    print("="*74)


if __name__ == "__main__":
    main()
