"""Task 1: precision-constrained threshold sweep + business_utility on FULL data.

Faithful to the tier-0 harness: same champion class (LogisticRegression + sigmoid
calibration, class_weight=balanced), same leakage-safe MART_SAFE_FEATURES, same
cost matrix {tp:+1, fp:-0.05, fn:-1, tn:0}, and a realistic TEMPORAL split
(train on earlier index_date, test on later) — the harness's
combined_temporal_entity_with_holdout collapses to a temporal split here because
each patient is one row. Full cohort (all 11,079 positives) → reliable precision.

Single-threaded, memory-frugal. Run: PYTHONPATH=. python threshold_analysis.py
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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from src.data.manifests import MART_SAFE_FEATURES
import scripts.convert_optum_mart as cm

PATH = "data/rwd/Optum_Parquet/Optum_enriched.parquet"
# Harness cost matrix (run_tier0_test.py Block 5B default).
COST = {"tp": 1.0, "fp": -0.05, "fn": -1.0, "tn": 0.0}
SAFE = [c for c in MART_SAFE_FEATURES if c not in ("geographic_region", "enrollment_duration_days")]
COHORT_COLS = ["patid", "entity_type", "index_biologic_brand", "treatment_start_date",
               "index_date", "claim_record_count", "elig_start_date", "zipcode_5"]


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


def business_utility(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return COST["tp"]*tp + COST["fp"]*fp + COST["fn"]*fn + COST["tn"]*tn, (tp, fp, fn, tn)


def main():
    schema = set(pq.ParquetFile(PATH).schema_arrow.names)
    need = [c for c in sorted(set(SAFE) | set(COHORT_COLS)) if c in schema]
    df = pq.read_table(PATH, columns=need).to_pandas()
    coh, attr = cm.select_initiation_cohort(df)
    del df
    coh = coh.reset_index(drop=True)
    y = coh[cm.TARGET].astype(int).values
    X = featurize(coh)
    idx_date = pd.to_datetime(coh["index_date"]).values

    # TEMPORAL split by index_date: train earliest 70% / calib next 15% / test latest 15%
    order = np.argsort(idx_date)
    n = len(order)
    tr = order[: int(0.70*n)]; ca = order[int(0.70*n):int(0.85*n)]; te = order[int(0.85*n):]
    Xtr, ytr = X.iloc[tr], y[tr]
    Xca, yca = X.iloc[ca], y[ca]
    Xte, yte = X.iloc[te], y[te]

    base = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, n_jobs=1)),
    ])
    base.fit(Xtr, ytr)
    # Sigmoid calibration on the (temporally-held-out) calibration slice.
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(Xca, yca)
    p = cal.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(yte, p)
    prevalence = yte.mean()
    print("="*78)
    print(f"FULL-DATA TEMPORAL SPLIT  (train {len(tr):,} / calib {len(ca):,} / test {len(te):,})")
    print(f"  test positives = {int(yte.sum()):,}  prevalence = {prevalence:.3%}")
    print(f"  index_date ranges: train≤{pd.Timestamp(idx_date[tr].max()).date()}  "
          f"test≥{pd.Timestamp(idx_date[te].min()).date()}")
    print(f"  TEST AUC = {auc:.4f}")
    print("="*78)

    # Threshold sweep
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9995, 400)))
    rows = []
    for t in grid:
        yp = (p >= t).astype(int)
        if yp.sum() == 0:
            continue
        pr, rc, f1, _ = precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)
        mcc = matthews_corrcoef(yte, yp) if len(np.unique(yp)) > 1 else 0.0
        util, (tp, fp, fn, tn) = business_utility(yte, yp)
        rows.append(dict(thr=t, prec=pr, rec=rc, f1=f1, mcc=mcc, util=util,
                         flagged=int(yp.sum()), tp=tp, fp=fp, fn=fn, lift=pr/prevalence if prevalence else 0))
    R = pd.DataFrame(rows)

    def show(label, r):
        if r is None:
            print(f"\n{label}: NOT ACHIEVABLE at any threshold")
            return
        print(f"\n{label}")
        print(f"  threshold={r['thr']:.4f}  precision={r['prec']:.3%} (lift {r['lift']:.2f}x)  "
              f"recall={r['rec']:.3%}  F1={r['f1']:.3f}  MCC={r['mcc']:.3f}")
        print(f"  flagged={r['flagged']:,}/{len(yte):,} ({r['flagged']/len(yte):.1%})  "
              f"tp={r['tp']} fp={r['fp']} fn={r['fn']}  business_utility={r['util']:.1f}")

    # Operating points of interest
    show("[max business_utility]", R.loc[R["util"].idxmax()])
    show("[max F1]", R.loc[R["f1"].idxmax()])
    show("[max MCC]", R.loc[R["mcc"].idxmax()])
    p5 = R[R["prec"] >= 0.05]
    show("[min thr s.t. precision >= 5% (the harness gate)]", p5.iloc[0] if len(p5) else None)
    p10 = R[R["prec"] >= 0.10]
    show("[min thr s.t. precision >= 10% (stronger commercial target)]", p10.iloc[0] if len(p10) else None)
    p20 = R[R["prec"] >= 0.20]
    show("[min thr s.t. precision >= 20%]", p20.iloc[0] if len(p20) else None)

    print("\n--- precision/recall/utility along the curve (deciles of flagged fraction) ---")
    R2 = R.sort_values("flagged")
    for q in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
        sub = R2.iloc[(R2["flagged"]/len(yte) - q).abs().argsort()[:1]]
        r = sub.iloc[0]
        print(f"  flag top {q*100:4.1f}%: precision={r['prec']:.3%} (lift {r['lift']:.2f}x)  "
              f"recall={r['rec']:.3%}  utility={r['util']:.1f}")
    print("="*78)


if __name__ == "__main__":
    main()
