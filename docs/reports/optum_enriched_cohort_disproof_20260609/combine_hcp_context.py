"""Does combining patient features + HCP context (the enriched primary_hcp_* block,
which a full mart+enriched join would supply more richly) yield a deployable model?

- disc/persistence (initiators -> HCP block populated): legitimate test. Compare
  patient-only vs patient+HCP vs HCP-only AUC on the temporal split + leak sniff.
- initiation: demonstrate the present-iff-treated leak (has_primary_hcp_flag predicts
  the outcome by missingness alone).
Run: PYTHONPATH=. python combine_hcp_context.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from src.data.manifests import MART_SAFE_FEATURES
import scripts.convert_optum_mart as cm

PATH = "data/rwd/Optum_Parquet/Optum_enriched.parquet"
SAFE = [c for c in MART_SAFE_FEATURES if c not in ("geographic_region", "enrollment_duration_days")]
# Leakage-safe HCP context (exclude adoption-derived: adoption_status/category/is_*/treated_count/rx_count)
HCP = ["primary_hcp_prov_type", "primary_hcp_provcat", "primary_hcp_taxonomy1", "primary_hcp_taxonomy2",
       "primary_hcp_prov_state", "primary_hcp_bed_sz_range", "primary_hcp_cred_type",
       "primary_hcp_grp_practice", "primary_hcp_hosp_affil", "primary_hcp_kol_score",
       "primary_hcp_influence_network_size"]
COHORT_COLS = ["patid", "entity_type", "index_biologic_brand", "treatment_start_date", "index_date",
               "claim_record_count", "elig_start_date", "zipcode_5", "last_observed_date",
               "last_coverage_end", "max_internal_gap_days", "terminal_gap_days", "has_primary_hcp_flag"]
GBM = dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=100,
           l2_regularization=2.0, class_weight="balanced", early_stopping=True, random_state=0)


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
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def run_cohort(name, selector, target):
    schema = set(pq.ParquetFile(PATH).schema_arrow.names)
    need = [c for c in sorted(set(SAFE) | set(COHORT_COLS) | set(HCP)) if c in schema]
    df = pq.read_table(PATH, columns=need).to_pandas()
    coh, _ = selector(df); del df
    coh = coh.reset_index(drop=True)
    y = coh[target].astype(int).values
    cov = (coh["has_primary_hcp_flag"] == 1).mean()
    Xp = patient_feats(coh)
    Xh = enc(coh[[c for c in HCP if c in coh.columns]])
    Xc = pd.concat([Xp, Xh], axis=1)
    ts = pd.to_datetime(coh["treatment_start_date"]).values
    order = np.argsort(ts); n = len(order)
    tr, te = order[:int(.8*n)], order[int(.8*n):]
    yt, ye = y[tr], y[te]
    a_p = auc_of(Xp.iloc[tr], yt, Xp.iloc[te], ye)
    a_h = auc_of(Xh.iloc[tr], yt, Xh.iloc[te], ye)
    a_c = auc_of(Xc.iloc[tr], yt, Xc.iloc[te], ye)
    # single-feature leak sniff on HCP block
    sf = {}
    for c in Xh.columns:
        col = Xh[c].fillna(Xh[c].median())
        if col.nunique() > 1:
            sf[c] = round(max(roc_auc_score(y, col), 1 - roc_auc_score(y, col)), 3)
    top = sorted(sf.items(), key=lambda kv: -kv[1])[:3]
    print(f"\n### {name.upper()}  (n={len(y):,}, pos={int(y.sum())}, prev={y.mean():.2%}, HCP-block coverage={cov:.1%})")
    print(f"  test AUC  patient-only = {a_p:.4f}")
    print(f"  test AUC  HCP-only     = {a_h:.4f}")
    print(f"  test AUC  COMBINED     = {a_c:.4f}   (Δ vs patient-only = {a_c - a_p:+.4f})  floor 0.65 -> "
          f"{'CLEARS' if a_c >= 0.65 else 'MISSES'}")
    print(f"  top HCP single-feat AUC (leak sniff): {top}")


def initiation_leak():
    df = pq.read_table(PATH, columns=["patid", "entity_type", "index_biologic_brand",
                                      "treatment_start_date", "index_date", "claim_record_count",
                                      "has_primary_hcp_flag"]).to_pandas()
    coh, _ = cm.select_initiation_cohort(df); del df
    y = coh[cm.TARGET].astype(int).values
    flag = (coh["has_primary_hcp_flag"] == 1).astype(int).values
    auc = roc_auc_score(y, flag)
    ct = pd.crosstab(flag, y, normalize="index")
    print("\n### INITIATION — why combining LEAKS (present-iff-treated)")
    print(f"  has_primary_hcp_flag ALONE predicts initiation: AUC = {auc:.4f}")
    print(f"  P(initiate | has_primary_hcp=0) = {ct.loc[0,1]:.4f}   P(initiate | has_primary_hcp=1) = {ct.loc[1,1]:.4f}")
    print("  -> the HCP block is populated almost exclusively for patients who initiate, so adding ANY")
    print("     primary_hcp_* feature leaks the label through missingness. Combining is invalid for initiation.")


if __name__ == "__main__":
    run_cohort("discontinuation", cm.select_discontinuation_cohort, cm.TARGET_DISCONTINUED)
    run_cohort("persistence", cm.select_persistence_cohort, cm.TARGET_PERSISTENT)
    initiation_leak()
