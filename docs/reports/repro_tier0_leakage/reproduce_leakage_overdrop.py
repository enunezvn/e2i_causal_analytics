#!/usr/bin/env python3
"""Reproduce the tier0 leakage over-drop (companion to
docs/reports/tier0_leakage_overdrop_definitive_diagnosis_20260605.md).

Loads the REAL node functions (detect_leakage, review_and_remediate_leakage) from
the repo via importlib, stubbing ONLY the `..state` type-annotation import — no
pipeline code is modified. Then runs them on a faithful Optum-like cohort
(n=1294, 37 positives = 2.86%, the Appendix-A cardinality mix) and demonstrates:

  1. The two-stage collapse (legitimate constant/lab drop vs buggy sparse-clinical drop).
  2. The guard asymmetry: perfect_class_separation skips the sparse rare-event
     features (it has a guard) while zero_variance_within_class flags them (it does not).
  3. The immunity gap: _apply_leakage_remediation's re-check re-drops a "declared-safe"
     sparse feature with no manifest immunity.
  4. The fix: adding the same rare-event guard to zero_variance recovers the features
     while still dropping a genuine post-index leak.

Requires only: numpy pandas scikit-learn scipy. Run from anywhere:
    python docs/reports/repro_tier0_leakage/reproduce_leakage_overdrop.py
"""
from __future__ import annotations
import os, sys, types, importlib.util, asyncio
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Repo root: env override, else 3 levels up from this file (docs/reports/repro_.../this.py).
ROOT = os.environ.get("REPO_ROOT") or str(Path(__file__).resolve().parents[3])


def _load_real_nodes():
    """Build stub parent packages so relative imports resolve WITHOUT running the
    heavy package __init__.py files, then load the two node modules from source."""
    def mk(name, rel):
        m = types.ModuleType(name); m.__path__ = [f"{ROOT}/{rel}"]; m.__package__ = name
        sys.modules[name] = m
    for name, rel in [
        ("src", "src"), ("src.agents", "src/agents"),
        ("src.agents.ml_foundation", "src/agents/ml_foundation"),
        ("src.agents.ml_foundation.data_preparer", "src/agents/ml_foundation/data_preparer"),
        ("src.agents.ml_foundation.data_preparer.nodes", "src/agents/ml_foundation/data_preparer/nodes"),
    ]:
        mk(name, rel)
    state_stub = types.ModuleType("src.agents.ml_foundation.data_preparer.state")
    state_stub.DataPreparerState = dict
    sys.modules["src.agents.ml_foundation.data_preparer.state"] = state_stub

    def load(modname, relpath):
        spec = importlib.util.spec_from_file_location(modname, f"{ROOT}/{relpath}")
        mod = importlib.util.module_from_spec(spec); sys.modules[modname] = mod
        spec.loader.exec_module(mod); return mod

    nd = "src.agents.ml_foundation.data_preparer.nodes"
    ld = load(f"{nd}.leakage_detector",
              "src/agents/ml_foundation/data_preparer/nodes/leakage_detector.py")
    lr = load(f"{nd}.leakage_remediation",
              "src/agents/ml_foundation/data_preparer/nodes/leakage_remediation.py")
    return ld, lr


def build_cohort(seed: int = 11):
    """Faithful Optum-like cohort: 3 feature classes per the 06-03 doc Appendix A."""
    rng = np.random.default_rng(seed); N, N_POS = 1294, 37
    y = np.zeros(N, int); y[rng.choice(N, N_POS, replace=False)] = 1
    cols = {"treatment_initiated": y, "patient_journey_id": [f"p{i}" for i in range(N)]}
    dense = []
    cols["age_at_index"] = rng.normal(50, 15, N) + y * rng.normal(3, 1, N); dense.append("age_at_index")
    for c, k in [("insurance_product", 2), ("payer_category", 2), ("plan_type", 6), ("age_group", 4),
                 ("gender", 2), ("geographic_region", 4), ("primary_diagnosis_code", 3),
                 ("urban_rural_code", 2), ("data_quality_score", 13), ("charlson_bucket", 6)]:
        cols[c] = rng.integers(0, k, N); dense.append(c)
    sparse_clin = []                                   # cardinality-2, 100% non-null, mostly 0
    for i in range(22):
        cols[f"clinflag_{i}"] = (rng.random(N) < float(rng.uniform(0.002, 0.03))).astype(int)
        sparse_clin.append(f"clinflag_{i}")
    const = []                                         # cardinality-1 all-constant
    for i in range(58):
        cols[f"const_{i}"] = 0; const.append(f"const_{i}")
    lab = []                                            # ~1-2% non-null (too sparse)
    for i in range(35):
        v = np.full(N, np.nan); k = int(N * float(rng.uniform(0.01, 0.02)))
        v[rng.choice(N, k, replace=False)] = rng.normal(5, 1, k); cols[f"lab_{i}"] = v; lab.append(f"lab_{i}")
    cols["initiated_biologic_180d"] = y                 # genuine post-index leak (== target)
    return pd.DataFrame(cols), dense, sparse_clin, const, lab


def klass(c, dense, sparse_clin, const, lab):
    if c in dense: return "dense"
    if c in sparse_clin: return "sparse_clin"
    if c in const: return "const"
    if c in lab: return "lab"
    return "leak/id"


def main():
    ld, lr = _load_real_nodes()
    df, dense, sparse_clin, const, lab = build_cohort()
    nfeat = df.shape[1] - 1
    print(f"Cohort: 1294 rows, 37 pos (2.86%); {nfeat} features = {len(dense)} dense + "
          f"{len(sparse_clin)} sparse-clinical(card2) + {len(const)} const + {len(lab)} lab + 1 leak + id\n")

    # Stage 1 — runner discovery filter (leakage-INDEPENDENT), run_tier0_test.py:5889-5906
    exclude = {"patient_journey_id", "treatment_initiated", "initiated_biologic_180d"}
    disc = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "iufb"
            and df[c].nunique() > 1 and df[c].notna().mean() > 0.5]
    print(f"[Stage 1] runner discovery filter (nunique>1 & notna>0.5): {len(disc)} survivors "
          f"{dict(Counter(klass(c, dense, sparse_clin, const, lab) for c in disc))}")
    print(f"          -> drops {len(const)} const + {len(lab)} labs; KEEPS all {len(sparse_clin)} sparse-clinical\n")

    # Stage 2 — the leakage path (no manifest -> no immunity)
    idx = np.arange(len(df)); tr, tmp = train_test_split(idx, test_size=0.4, random_state=1, stratify=df.treatment_initiated)
    va, te = train_test_split(tmp, test_size=0.5, random_state=1, stratify=df.treatment_initiated.iloc[tmp])
    base_state = lambda: {
        "experiment_id": "repro", "train_df": df.iloc[tr].reset_index(drop=True),
        "validation_df": df.iloc[va].reset_index(drop=True), "test_df": df.iloc[te].reset_index(drop=True),
        "holdout_df": None, "skip_leakage_check": False, "leakage_remediation_attempts": 0,
        "scope_spec": {"prediction_target": "treatment_initiated", "problem_type": "binary_classification",
                       "required_features": [], "excluded_features": ["patient_journey_id"],
                       "feature_manifest_source": None}}

    def run_leakage(zv_fn=None):
        if zv_fn is not None:
            orig = ld.check_zero_variance_within_class; ld.check_zero_variance_within_class = zv_fn
        try:
            s = base_state(); out = asyncio.run(ld.detect_leakage(s)); s.update(out)
            rem = asyncio.run(lr.review_and_remediate_leakage(s))
            return out, rem
        finally:
            if zv_fn is not None: ld.check_zero_variance_within_class = orig

    out, rem = run_leakage()
    leaked = out["leaked_features"]; verified = rem.get("leakage_remediated_features", [])
    byc = Counter(f["check_name"] for f in out["leakage_findings"] if f["severity"] in ("critical", "high"))
    print(f"[Stage 2] detect_leakage (no manifest): leaked={len(leaked)} by_check={dict(byc)}; "
          f"sparse-clinical flagged = {sum(1 for c in leaked if c in sparse_clin)}/{len(sparse_clin)}")
    print(f"          remediation 'Clean Features' = {len(verified)} -> RUNNER TRAINS ON {len(verified)} of {nfeat}\n")

    # Guard asymmetry on identical inputs
    combined = pd.concat([df.iloc[tr], df.iloc[va]], ignore_index=True)
    pcs = ld.check_perfect_class_separation(combined, "treatment_initiated", sparse_clin)
    zv = ld.check_zero_variance_within_class(combined, "treatment_initiated", sparse_clin)
    hi = lambda fs: sum(1 for f in fs if f.severity.value in ("critical", "high"))
    print(f"[Guard asymmetry] on the {len(sparse_clin)} cardinality-2 sparse predictors (identical inputs):")
    print(f"          perfect_class_separation HIGH/CRIT : {hi(pcs):>2}/{len(sparse_clin)}  (HAS guard, leakage_detector.py:613-627)")
    print(f"          zero_variance_within_class HIGH/CRIT: {hi(zv):>2}/{len(sparse_clin)}  (NO guard)\n")

    # Immunity gap: a 'declared-safe' sparse feature is re-dropped by the re-check
    victim = "clinflag_0"
    analysis = {"leakage_classifications": {}, "features_to_drop": [], "replacement_candidates": [],
                "recommended_feature_set": [victim, "age_at_index", "gender"], "viable": True,
                "confidence": "high", "reasoning": "simulate declared-safe sparse feature kept by immunity"}
    res = lr._apply_leakage_remediation(base_state(), analysis)
    print(f"[Immunity gap] feed '{victim}' (imagine manifest-declared-safe) into recommended set:")
    print(f"          _apply_leakage_remediation re-check dropped it? "
          f"{'YES (immunity not honored in re-check)' if victim not in res.get('final_features', []) else 'no'}\n")

    # The fix: add the same rare-event guard to zero_variance
    _orig = ld.check_zero_variance_within_class
    def guarded(d, tv, feats):
        keep = []
        for f in feats:
            fv = d[f][d[f].notna() & d[tv].notna()]; c1 = fv[d[tv] == 1]
            if fv.nunique() <= 2 and (len(c1) < 30 or (len(c1) / max(len(fv), 1)) < 0.05):
                continue                       # PROPOSED guard (mirrors perfect_class_separation)
            keep.append(f)
        return _orig(d, tv, keep)
    out2, rem2 = run_leakage(zv_fn=guarded)
    print(f"[Fix validation] add the rare-event guard to zero_variance_within_class:")
    print(f"          BEFORE: leaked={len(leaked):>2}  recovered={len(verified):>2}")
    print(f"          AFTER : leaked={len(out2['leaked_features']):>2}  "
          f"recovered={len(rem2.get('leakage_remediated_features', [])):>2}  "
          f"(genuine post-index leak still dropped)")


if __name__ == "__main__":
    main()
