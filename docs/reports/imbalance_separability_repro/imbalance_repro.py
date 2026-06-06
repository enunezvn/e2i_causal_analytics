"""
Reproduction of the Tier-0 Optum-mart INITIATION disproof, to test whether
class-imbalance handling is the right lever or the ceiling is separability/
feature-bound.

Grounded in three sources:
  1. The disproof md: 1.41% prevalence, ~64 features, AUC ceiling ~0.68,
     PR-AUC ~0.029 (~2x the prevalence baseline); sampling/scaling/
     regularization all measured null -> "feature-bound".
  2. ULB fraud-detection-handbook Ch.6: resampling/cost-sensitive trade AUC/
     balanced-accuracy for Average Precision; full rebalancing (IR=1) is worst
     for AP; tree models robust without resampling.
  3. Daily-Dose-of-Data-Science "Separability-in-Class-Imbalance": imbalance
     only hurts when classes OVERLAP; separable classes train fine at any ratio.

Generator: a LOGISTIC-LINK DGP (y ~ Bernoulli(sigmoid(b0 + X.beta))) — a weak,
DIFFUSE, *linear* signal with no separable blob. This is the faithful analog of
the disproof regime: AUC can look ~0.68 while PR-AUC sits at ~2x baseline because
at 1.4% prevalence the top of the ranking is still flooded by the majority. It is
linear, so XGBoost gets no nonlinear bonus -> the ceiling binds BOTH model
classes (a true feature ceiling, not an LR artifact). "Separability" = signal
strength: low signal = inseparable/overlapping; high signal = separable.

Methodology guard: split FIRST, resample TRAIN ONLY, evaluate on untouched test
(mirrors the codebase apply_resampling contract; no SMOTE-before-split leakage).
LR uses lbfgs — the disproof proved it matches the thrashing saga's AUC in ~20
iters vs 1000.
"""
from __future__ import annotations
import warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

RNG = 42
PREV = 0.0141      # 1.41% prevalence (disproof)
NFEAT = 64         # ~64 pre-index features (disproof)
KINF = 12          # only 12 weakly-informative features; the rest are noise
N = 120_000        # ~1,700 minority events; EPV ~26 (disproof: "well clear of the >=10 floor")

def make_logit(n, signal, seed=RNG):
    """Weak diffuse LINEAR signal; intercept solved to hit PREV exactly."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, NFEAT))
    beta = np.zeros(NFEAT); beta[:KINF] = signal
    latent = X @ beta
    b0 = brentq(lambda b: expit(b + latent).mean() - PREV, -50, 50)
    y = rng.binomial(1, expit(b0 + latent))
    return X, y

def split(X, y):
    return train_test_split(X, y, test_size=0.4, stratify=y, random_state=RNG)

def evaluate(clf, Xte_s, yte):
    p = clf.predict_proba(Xte_s)[:, 1]
    yhat = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
                bal_acc=balanced_accuracy_score(yte, yhat))

def lr(class_weight=None):
    return LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, class_weight=class_weight)

def xgbc(scale_pos_weight=1.0):
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                             eval_metric="aucpr", scale_pos_weight=scale_pos_weight,
                             random_state=RNG, n_jobs=4, verbosity=0)

def run_treatments(X, y, tag):
    Xtr, Xte, ytr, yte = split(X, y)
    sc = StandardScaler().fit(Xtr)                      # fit on TRAIN only
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    base_prev = yte.mean(); n_min = int(ytr.sum()); rows = []
    def add(name, clf, Xt, yt):
        clf.fit(Xt, yt); m = evaluate(clf, Xte_s, yte); m["lift"] = m["ap"]/base_prev
        rows.append((name, m))
    k = min(5, n_min - 1)
    add("LR baseline",               lr(),               Xtr_s, ytr)
    add("LR class_weight=balanced",  lr("balanced"),     Xtr_s, ytr)
    Xr, yr = SMOTE(random_state=RNG, k_neighbors=k).fit_resample(Xtr_s, ytr)
    add("LR + SMOTE 1:1",            lr(),               Xr, yr)
    Xr, yr = SMOTE(random_state=RNG, sampling_strategy=0.5, k_neighbors=k).fit_resample(Xtr_s, ytr)
    add("LR + SMOTE 0.5 + cw (combined)", lr("balanced"), Xr, yr)
    Xr, yr = RandomUnderSampler(random_state=RNG).fit_resample(Xtr_s, ytr)
    add("LR + RandomUnderSample",    lr(),               Xr, yr)
    spw = (len(ytr) - n_min) / max(n_min, 1)
    add("XGB baseline",              xgbc(1.0),          Xtr_s, ytr)
    add("XGB scale_pos_weight=IR",   xgbc(spw),          Xtr_s, ytr)
    Xr, yr = SMOTE(random_state=RNG, k_neighbors=k).fit_resample(Xtr_s, ytr)
    add("XGB + SMOTE 1:1",           xgbc(1.0),          Xr, yr)
    print(f"\n=== {tag}  (test prevalence={base_prev:.4f}, train minority n={n_min}) ===")
    print(f"{'treatment':<32}{'AUC-ROC':>9}{'PR-AUC':>9}{'lift':>7}{'bal_acc':>9}")
    for name, m in rows:
        print(f"{name:<32}{m['auc']:>9.4f}{m['ap']:>9.4f}{m['lift']:>7.2f}{m['bal_acc']:>9.4f}")
    return rows

print("#" * 80)
print("# PART A — calibrate signal to the disproof's feature-bound ceiling")
print("#  target: baseline AUC ~= 0.68, PR-AUC lift ~= 2x the 0.0141 baseline")
print("#" * 80)
TARGET_AUC = 0.68; cand = {}
for sig in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    X, y = make_logit(N, sig)
    Xtr, Xte, ytr, yte = split(X, y)
    sc = StandardScaler().fit(Xtr)
    m = evaluate(lr().fit(sc.transform(Xtr), ytr), sc.transform(Xte), yte)
    cand[sig] = m
    print(f"signal={sig:<5}  baseline LR  AUC={m['auc']:.4f}  PR-AUC={m['ap']:.4f}  lift={m['ap']/yte.mean():.2f}")
LOW = min(cand, key=lambda s: abs(cand[s]["auc"] - TARGET_AUC))
print(f"\n-> locked LOW signal={LOW} (baseline AUC={cand[LOW]['auc']:.4f}, closest to {TARGET_AUC})")

print("\n" + "#" * 80)
print(f"# PART B — INSEPARABLE (signal={LOW}): does imbalance handling move PR-AUC?")
print("#  hypothesis (handbook+disproof): treatments shift bal_acc, NOT PR-AUC")
print("#" * 80)
Xlo, ylo = make_logit(N, LOW)
run_treatments(Xlo, ylo, f"INSEPARABLE signal={LOW}")

print("\n" + "#" * 80)
print("# PART C — codebase's OWN nodes on the inseparable data (faithful, not mocked)")
print("#" * 80)
import sys, asyncio, importlib.util, types
from pathlib import Path
# Repo root derived from this file's location (docs/reports/imbalance_separability_repro/).
REPO = str(Path(__file__).resolve().parents[3]); sys.path.insert(0, REPO)
# Minimal shim: stub ONLY the path-finder utility so detect_class_imbalance's
# `from src.utils.project_root import find_project_root` resolves WITHOUT
# triggering src/utils/__init__.py (which imports supabase). The strategy-matrix
# logic and imblearn resampling under test still run for real, against the real
# config/imbalance_strategy.yaml. This stubs a locator, not the logic.
for nm in ("src", "src.utils"):
    if nm not in sys.modules:
        pkg = types.ModuleType(nm); pkg.__path__ = []; sys.modules[nm] = pkg
_pr = types.ModuleType("src.utils.project_root")
_pr.find_project_root = lambda: Path(REPO)
sys.modules["src.utils.project_root"] = _pr
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, f"{REPO}/{rel}")
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m
try:
    _det = _load("e2i_detect", "src/agents/ml_foundation/model_trainer/nodes/detect_class_imbalance.py")
    _res = _load("e2i_resample", "src/agents/ml_foundation/model_trainer/nodes/apply_resampling.py")
    Xtr, Xte, ytr, yte = split(Xlo, ylo)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    for algo in ["LogisticRegression", "XGBoost"]:
        det = asyncio.run(_det.detect_class_imbalance(
            {"train_data": {"y": ytr}, "algorithm_name": algo, "problem_type": "binary_classification"}))
        res = asyncio.run(_res.apply_resampling({
            "recommended_strategy": det["recommended_strategy"], "imbalance_detected": det["imbalance_detected"],
            "X_train_preprocessed": Xtr_s, "train_data": {"y": ytr}, "class_distribution": det["class_distribution"]}))
        Xrs, yrs = res["X_train_resampled"], res["y_train_resampled"]
        cw = "balanced" if det["recommended_strategy"] in ("class_weight", "combined") else None
        if algo == "LogisticRegression":
            clf = lr(cw)
        else:
            spw = (len(ytr)-int(ytr.sum()))/max(int(ytr.sum()),1) if det["recommended_strategy"]=="class_weight" else 1.0
            clf = xgbc(spw)
        clf.fit(Xrs, yrs); m = evaluate(clf, Xte_s, yte)
        print(f"{algo:<20} severity={det['imbalance_severity']:<8} strategy={det['recommended_strategy']:<10} "
              f"resampled={str(res['resampling_applied']):<5} -> AUC={m['auc']:.4f}  PR-AUC={m['ap']:.4f}  lift={m['ap']/yte.mean():.2f}")
except Exception as e:
    import traceback; print("Could not drive codebase nodes:", e); traceback.print_exc()

print("\n" + "#" * 80)
print("# PART D — SEPARABLE contrast (high signal=0.90, SAME 1.41% prevalence)")
print("#  separability thesis: same imbalance, separable -> strong PR-AUC, NO handling")
print("#" * 80)
Xhi, yhi = make_logit(N, 0.90)
Xtr, Xte, ytr, yte = split(Xhi, yhi)
sc = StandardScaler().fit(Xtr)
m = evaluate(lr().fit(sc.transform(Xtr), ytr), sc.transform(Xte), yte)
print(f"SEPARABLE baseline LR (NO imbalance handling): AUC={m['auc']:.4f}  PR-AUC={m['ap']:.4f}  "
      f"lift={m['ap']/yte.mean():.1f}  bal_acc={m['bal_acc']:.4f}")

print("\n" + "#" * 80)
print("# PART E — 'more events doesn't raise the ceiling' (disproof's 50k vs 300k)")
print("#  faithful: ONE population, stratified-subsampled, only event COUNT varies")
print("#" * 80)
Xpop, ypop = make_logit(300_000, LOW)
for n in [50_000, 300_000]:
    if n < len(ypop):
        idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=RNG).split(Xpop, ypop))
        X, y = Xpop[idx], ypop[idx]
    else:
        X, y = Xpop, ypop
    Xtr, Xte, ytr, yte = split(X, y)
    sc = StandardScaler().fit(Xtr)
    m = evaluate(lr().fit(sc.transform(Xtr), ytr), sc.transform(Xte), yte)
    print(f"n={n:>7}  events={int(y.sum()):>6}  EPV~{int(y.sum()*0.6/NFEAT):>3}  "
          f"baseline LR  AUC={m['auc']:.4f}  PR-AUC={m['ap']:.4f}  lift={m['ap']/yte.mean():.2f}")
print("\nDONE.")
