# Imbalance vs separability — reproduction

Companion to `docs/results/tier0_imbalance_methodology_review_20260606.md`.

`imbalance_repro.py` reproduces the **mechanism** behind the Optum-mart
INITIATION disproof (the real 787K mart isn't committed, so the literal data
can't run here) and tests whether class-imbalance handling is the right lever or
the ceiling is **separability / feature-bound**.

It uses a logistic-link DGP (`y ~ Bernoulli(sigmoid(b0 + X·β))`) — a weak,
diffuse, *linear* signal with no separable blob — which is the faithful analog
of the disproof regime (AUC can look ~0.68 while PR-AUC sits at ~2× baseline,
because at 1.4% prevalence the top of the ranking is flooded by the majority).
Being linear, XGBoost gets no nonlinear bonus, so the ceiling binds **both**
model classes — a true feature ceiling, not an LR artifact.

What it shows (see `imbalance_repro_output.txt`):
- **Part A** calibrates signal to the disproof's AUC≈0.68 ceiling.
- **Part B** every imbalance treatment (class_weight, SMOTE 1:1, SMOTE 0.5,
  RUS, XGB scale_pos_weight, XGB+SMOTE) leaves PR-AUC flat or **degrades** it
  while inflating balanced accuracy.
- **Part C** drives the codebase's *own* `detect_class_imbalance` +
  `apply_resampling` nodes (faithful, not mocked) — same conclusion.
- **Part D** the SEPARABLE contrast: same 1.41% prevalence, separable features
  → PR-AUC 0.49 (34× lift) with **no** imbalance handling.
- **Part E** more events do not raise the ceiling (50k vs 300k subsample).

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install "scikit-learn>=1.6.1,<1.7" imbalanced-learn xgboost numpy scipy pyyaml
python docs/reports/imbalance_separability_repro/imbalance_repro.py
```

Methodology guard: split first, resample **train only**, evaluate on untouched
test (mirrors the `apply_resampling` contract; no SMOTE-before-split leakage).
LR uses `lbfgs` — the disproof proved it matches the thrashing `saga`'s AUC in
~20 iters vs 1000.
