"""Temporal-split + commercial-bar evaluation for discontinuation & persistence
(enriched file), matching the initiation deep-dive. Treatment-anchored cohorts:
temporal split by treatment_start_date (features are knowable at dx-index <= ts).

Commercial bar (run_tier0_test --deployment-intent commercial):
  AUC>=0.65, recall>=0.50, MCC>=0.10, net-benefit @ p_t=0.10 > 0.
Run: PYTHONPATH=. python other_cohorts_eval.py <discontinuation|persistence>
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support
from src.data.manifests import MART_SAFE_FEATURES
import scripts.convert_optum_mart as cm

PATH = "data/rwd/Optum_Parquet/Optum_enriched.parquet"
SAFE = [c for c in MART_SAFE_FEATURES if c not in ("geographic_region", "enrollment_duration_days")]
COHORT_COLS = ["patid", "entity_type", "index_biologic_brand", "treatment_start_date", "index_date",
               "claim_record_count", "elig_start_date", "zipcode_5", "last_observed_date",
               "last_coverage_end", "max_internal_gap_days", "terminal_gap_days"]
SEL = {"discontinuation": (cm.select_discontinuation_cohort, cm.TARGET_DISCONTINUED),
       "persistence": (cm.select_persistence_cohort, cm.TARGET_PERSISTENT)}


def featurize(frame):
    X = frame[[c for c in SAFE if c in frame.columns]].copy()
    X["enrollment_duration_days"] = (pd.to_datetime(frame["index_date"]) - pd.to_datetime(frame["elig_start_date"])).dt.days
    X["zip3_region"] = frame["zipcode_5"].astype("string").str.slice(0, 3)
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
        X[c] = X[c].astype("float32")
    return X


def net_benefit(y, p, pt):
    yp = (p >= pt).astype(int); n = len(y)
    tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    return tp / n - (fp / n) * (pt / (1 - pt))


def main(name):
    selector, target = SEL[name]
    schema = set(pq.ParquetFile(PATH).schema_arrow.names)
    need = [c for c in sorted(set(SAFE) | set(COHORT_COLS)) if c in schema]
    df = pq.read_table(PATH, columns=need).to_pandas()
    coh, _ = selector(df); del df
    coh = coh.reset_index(drop=True)
    y = coh[target].astype(int).values
    X = featurize(coh)
    anchor = pd.to_datetime(coh["treatment_start_date"]).values
    order = np.argsort(anchor); n = len(order)
    tr, ca, te = order[:int(.70*n)], order[int(.70*n):int(.85*n)], order[int(.85*n):]
    base = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=1))])
    base.fit(X.iloc[tr], y[tr])
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit"); cal.fit(X.iloc[ca], y[ca])
    yte = y[te]; p = cal.predict_proba(X.iloc[te])[:, 1]
    prev = yte.mean(); auc = roc_auc_score(yte, p)

    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.999, 600)))
    best_mcc, peak_prec = -1.0, 0.0
    rec_ok = None  # highest threshold s.t. recall >= 0.50
    for t in grid:
        yp = (p >= t).astype(int)
        if yp.sum() == 0 or len(np.unique(yp)) < 2:
            continue
        pr, rc, _, _ = precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)
        m = matthews_corrcoef(yte, yp)
        best_mcc = max(best_mcc, m); peak_prec = max(peak_prec, pr)
        if rc >= 0.50:
            rec_ok = (t, pr, rc)  # grid ascends → last qualifying = highest threshold

    print("="*78)
    print(f"{name.upper()}  (enriched, temporal split by treatment_start_date)")
    print(f"  train {len(tr):,} / calib {len(ca):,} / test {len(te):,}  | test positives={int(yte.sum()):,}  prevalence={prev:.2%}")
    print(f"  ts range: train<={pd.Timestamp(anchor[tr].max()).date()}  test>={pd.Timestamp(anchor[te].min()).date()}")
    print("="*78)
    print(f"Gate 1  AUC >= 0.65 ......... {auc:.4f}   {'PASS' if auc>=0.65 else 'FAIL'}")
    if rec_ok:
        print(f"Gate 2  recall >= 0.50 ..... recall={rec_ok[2]:.3f} @ thr={rec_ok[0]:.4f} (precision={rec_ok[1]:.3%})   PASS")
    else:
        print(f"Gate 2  recall >= 0.50 ..... not achievable   FAIL")
    print(f"Gate 3  MCC >= 0.10 ........ best={best_mcc:.4f}   {'PASS' if best_mcc>=0.10 else 'FAIL'}")
    print("Gate 4  net-benefit vs treat-none:")
    for pt in [0.05, 0.10, 0.20, 0.30]:
        nb = net_benefit(yte, p, pt)
        print(f"        p_t={pt:.2f}: NB={nb:+.5f} {'>0 PASS' if nb>0 else '<=0 FAIL'}"
              f"{'  <-- gate' if pt==0.10 else ''}")
    print(f"\n  peak precision anywhere = {peak_prec:.3%} (lift {peak_prec/prev:.2f}x)")
    gates = [auc>=0.65, rec_ok is not None, best_mcc>=0.10, net_benefit(yte,p,0.10)>0]
    print(f"  COMMERCIAL gates passed: {sum(gates)}/4  -> {'DEPLOYABLE' if all(gates) else 'BLOCKED'}")
    print("="*78)


if __name__ == "__main__":
    main(sys.argv[1])
