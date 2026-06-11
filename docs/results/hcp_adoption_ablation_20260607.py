"""Leakage ablation for the HCP-grain adoption-propensity model.

Validates that the AUC~0.85 signal (optum_hcp, admissible network/volume/geo
features) is NOT a tautology/leak. Splits features into groups and measures
each group's standalone AUC + leave-one-group-out, plus LightGBM importances.
Real data, no mocks.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pyarrow.dataset as ds
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

np.random.seed(42)
P = "data/rwd/Optum_Parquet/Optum.parquet"
dset = ds.dataset(P)

GROUPS = {
    "network": ["shared_patient_edge_count", "max_shared_patient_edge_weight",
                "shared_patient_kol_score_pct", "referral_in_patient_count",
                "referral_out_patient_count"],
    "volume": ["medical_patient_count"],
    "geo": ["prov_state", "prov_type"],
}
ALL = [c for g in GROUPS.values() for c in g]
cols = ["entity_type", "adoption_status"] + ALL
t = dset.to_table(columns=cols, filter=ds.field("entity_type") == "optum_hcp").to_pandas()
t["y"] = (t["adoption_status"] == "ADOPTER").astype(int)
t = t.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), 120000), random_state=42))
y = t["y"].values
print(f"sample n={len(t)} prev={y.mean():.4f}")

def fit_auc(feats):
    cat = [c for c in feats if t[c].dtype == object]
    num = [c for c in feats if c not in cat]
    Xtr, Xte, ytr, yte = train_test_split(t[feats], y, test_size=0.3, stratify=y, random_state=42)
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30))]), cat),
    ]) if cat else ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num)])
    Xtr_ = pre.fit_transform(Xtr); Xte_ = pre.transform(Xte)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                           min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, random_state=42, verbose=-1, class_weight="balanced").fit(Xtr_, ytr)
    pt = m.predict_proba(Xte_)[:, 1]
    return roc_auc_score(yte, pt), average_precision_score(yte, pt) / yte.mean()

print("\n--- standalone group AUC (LightGBM) ---")
for g, feats in GROUPS.items():
    a, l = fit_auc(feats); print(f"  {g:10s} ({len(feats)} feat): AUC={a:.4f} lift={l:.2f}x")
print("\n--- leave-one-group-out ---")
for g in GROUPS:
    feats = [c for gg, fs in GROUPS.items() if gg != g for c in fs]
    a, l = fit_auc(feats); print(f"  drop {g:10s}: AUC={a:.4f} lift={l:.2f}x")
a, l = fit_auc(ALL); print(f"\n  ALL: AUC={a:.4f} lift={l:.2f}x")

# importances on full model
cat = [c for c in ALL if t[c].dtype == object]; num = [c for c in ALL if c not in cat]
Xtr, Xte, ytr, yte = train_test_split(t[ALL], y, test_size=0.3, stratify=y, random_state=42)
pre = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
    ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30))]), cat)])
Xtr_ = pre.fit_transform(Xtr)
feat_names = num + list(pre.named_transformers_["cat"].named_steps["o"].get_feature_names_out(cat))
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
                       verbose=-1, class_weight="balanced").fit(Xtr_, ytr)
imp = sorted(zip(feat_names, m.feature_importances_), key=lambda x: -x[1])[:12]
print("\n--- top importances ---")
for n, i in imp: print(f"  {n:40s} {i}")
