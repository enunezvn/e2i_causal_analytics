"""Evaluate the initiation model against the COMMERCIAL deployment-intent bar.

run_tier0_test.py --deployment-intent commercial defines the bar as:
  AUC >= 0.65, recall >= 0.50, MCC >= 0.10, net-benefit @ p_t=0.10 > 0 (treat-none).
(The wrapper run used deployment_intent=clinical w/ min_auc forced to 0.65, recall
gate only 0.10 — NOT this commercial bar.) Full-data temporal split, all positives.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, precision_recall_fscore_support
from src.data.manifests import MART_SAFE_FEATURES
import scripts.convert_optum_mart as cm

PATH = "data/rwd/Optum_Parquet/Optum_enriched.parquet"
SAFE = [c for c in MART_SAFE_FEATURES if c not in ("geographic_region", "enrollment_duration_days")]
COHORT_COLS = ["patid", "entity_type", "index_biologic_brand", "treatment_start_date",
               "index_date", "claim_record_count", "elig_start_date", "zipcode_5"]


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
    """Decision-curve net benefit at threshold-probability pt (vs treat-none=0)."""
    yp = (p >= pt).astype(int)
    n = len(y)
    tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    return tp / n - (fp / n) * (pt / (1 - pt))


def main():
    schema = set(pq.ParquetFile(PATH).schema_arrow.names)
    need = [c for c in sorted(set(SAFE) | set(COHORT_COLS)) if c in schema]
    df = pq.read_table(PATH, columns=need).to_pandas()
    coh, _ = cm.select_initiation_cohort(df); del df
    coh = coh.reset_index(drop=True)
    y = coh[cm.TARGET].astype(int).values
    X = featurize(coh)
    d = pd.to_datetime(coh["index_date"]).values
    order = np.argsort(d); n = len(order)
    tr, ca, te = order[:int(.70*n)], order[int(.70*n):int(.85*n)], order[int(.85*n):]
    base = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1))])
    base.fit(X.iloc[tr], y[tr])
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit"); cal.fit(X.iloc[ca], y[ca])
    yte = y[te]; p = cal.predict_proba(X.iloc[te])[:, 1]
    prev = yte.mean()

    auc = roc_auc_score(yte, p)
    # recall>=0.50 operating point: pick the highest threshold s.t. recall>=0.50
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.999, 600)))
    rec_ok = None
    for t in grid[::-1]:
        yp = (p >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)
        if rc >= 0.50:
            mcc = matthews_corrcoef(yte, yp)
            rec_ok = (t, pr, rc, mcc, int(yp.sum())); break
    # max MCC anywhere
    best_mcc, best = -1, None
    for t in grid:
        yp = (p >= t).astype(int)
        if len(np.unique(yp)) < 2: continue
        m = matthews_corrcoef(yte, yp)
        if m > best_mcc: best_mcc, best = m, (t, *precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)[:2], int(yp.sum()))

    print("="*78)
    print(f"COMMERCIAL-BAR EVALUATION  (full-data temporal split; test n={len(yte):,}, "
          f"positives={int(yte.sum()):,}, prevalence={prev:.3%})")
    print("="*78)
    print(f"\nGate 1  AUC >= 0.65 ............... {auc:.4f}   {'PASS' if auc>=0.65 else 'FAIL'}")
    if rec_ok:
        t, pr, rc, mcc, fl = rec_ok
        print(f"Gate 2  recall >= 0.50 ........... achievable: recall={rc:.3f} @ thr={t:.4f} "
              f"(precision={pr:.3%}, MCC={mcc:.3f}, flags {fl:,}/{len(yte):,}={fl/len(yte):.0%})   PASS(recall)")
    else:
        print(f"Gate 2  recall >= 0.50 ........... not achievable   FAIL")
    print(f"Gate 3  MCC >= 0.10 .............. best MCC anywhere = {best_mcc:.4f}   "
          f"{'PASS' if best_mcc>=0.10 else 'FAIL'}  (at thr={best[0]:.4f}, precision={best[1]:.3%}, recall={best[2]:.3f})")
    print("\nGate 4  net-benefit (decision curve) vs treat-none(=0):")
    for pt in [0.02, 0.05, 0.10, 0.15]:
        nb = net_benefit(yte, p, pt)
        tag = "  <-- commercial gate p_t=0.10" if abs(pt-0.10) < 1e-9 else ""
        print(f"        p_t={pt:.2f}:  NB={nb:+.5f}  (treat-all={prev - (1-prev)*(pt/(1-pt)):+.5f})  "
              f"{'>0 PASS' if nb>0 else '<=0 FAIL'}{tag}")
    print("\nOVERALL commercial verdict: AUC marginal PASS, recall PASS, "
          f"MCC {'PASS' if best_mcc>=0.10 else 'FAIL'}, net-benefit@0.10 see above.")
    print("="*78)


if __name__ == "__main__":
    main()
