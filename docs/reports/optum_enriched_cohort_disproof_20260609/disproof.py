"""Cheapest-disproof: leakage-safe AUC ceiling per cohort on Optum_enriched.parquet.

Memory-frugal (one cohort per process invocation, n_jobs=1, stratified sample for
the large initiation cohort). Uses the EXACT converter selector logic so the cohort
definitions + leakage governance are faithful.

Usage: python disproof.py <initiation|discontinuation|persistence>
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.data.manifests import MART_SAFE_FEATURES
import scripts.convert_optum_mart as cm

PATH = "data/rwd/Optum_Parquet/Optum_enriched.parquet"
CAP = 200_000  # max rows for the AUC-ceiling fit (all positives kept; negatives capped)
SEED = 0

SAFE = [c for c in MART_SAFE_FEATURES if c not in ("geographic_region", "enrollment_duration_days")]
COHORT_COLS = ["patid", "entity_type", "index_biologic_brand", "treatment_start_date",
               "index_date", "claim_record_count", "elig_start_date", "zipcode_5",
               "last_observed_date", "last_coverage_end", "max_internal_gap_days",
               "terminal_gap_days"]

SELECTORS = {
    "initiation": (cm.select_initiation_cohort, cm.TARGET, "index_date"),
    "discontinuation": (cm.select_discontinuation_cohort, cm.TARGET_DISCONTINUED, "treatment_start_date"),
    "persistence": (cm.select_persistence_cohort, cm.TARGET_PERSISTENT, "treatment_start_date"),
}


def featurize(frame):
    X = frame[[c for c in SAFE if c in frame.columns]].copy()
    X["enrollment_duration_days"] = (
        pd.to_datetime(frame["index_date"]) - pd.to_datetime(frame["elig_start_date"])
    ).dt.days
    X["zip3_region"] = frame["zipcode_5"].astype("string").str.slice(0, 3)
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
        X[c] = X[c].astype("float32")
    return X


def main(name):
    selector, target, anchor = SELECTORS[name]
    schema = set(pq.ParquetFile(PATH).schema_arrow.names)
    need = [c for c in sorted(set(SAFE) | set(COHORT_COLS)) if c in schema]
    df = pq.read_table(PATH, columns=need).to_pandas()
    coh, attr = selector(df)
    del df
    y_full = coh[target].astype(int).values
    n_full, pos_full = len(y_full), int(y_full.sum())

    # Memory cap: keep all positives + a random sample of negatives up to CAP.
    if n_full > CAP:
        rng = np.random.RandomState(SEED)
        pos_idx = np.where(y_full == 1)[0]
        neg_idx = np.where(y_full == 0)[0]
        n_neg = min(len(neg_idx), CAP - len(pos_idx))
        neg_keep = rng.choice(neg_idx, size=n_neg, replace=False)
        keep = np.concatenate([pos_idx, neg_keep])
        coh = coh.iloc[keep]
        sampled = True
    else:
        sampled = False

    y = coh[target].astype(int).values
    X = featurize(coh)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, l2_regularization=1.0,
        random_state=SEED, class_weight="balanced", early_stopping=True,
    )
    auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=1)

    # Single-feature AUC leakage sanity (top 3 by |0.5-auc|)
    from sklearn.metrics import roc_auc_score
    sf = {}
    for c in X.columns:
        col = X[c].fillna(X[c].median())
        if col.nunique() > 1:
            a = roc_auc_score(y, col)
            sf[c] = max(a, 1 - a)
    top = sorted(sf.items(), key=lambda kv: -kv[1])[:4]

    print(f"### {name.upper()}  (anchor={anchor})")
    print(f"  full: n={n_full:,}  positives={pos_full:,}  base_rate={pos_full/n_full:.3%}")
    if sampled:
        print(f"  fit on sample: n={len(y):,}  positives={int(y.sum()):,}  base_rate={y.mean():.3%}  (all pos + capped neg)")
    print(f"  features used: {X.shape[1]}  (leakage-safe MART_SAFE_FEATURES + 2 derived)")
    print(f"  5-fold CV AUC = {auc.mean():.4f} +/- {auc.std():.4f}   folds={np.round(auc,3).tolist()}")
    print(f"  top single-feature AUC (leak check, want <0.80): {[(c, round(v,3)) for c,v in top]}")
    print(f"  attrition: {attr}")


if __name__ == "__main__":
    main(sys.argv[1])
