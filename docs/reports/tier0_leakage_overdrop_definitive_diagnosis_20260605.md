# Tier-0 Leakage Detection Over-Drop — Definitive Diagnosis & Solution

**Date**: 2026-06-05
**Author**: research mission (read-only; no pipeline code modified)
**Scope**: `step_2_data_preparer` leakage detection/remediation in the tier-0 pipeline
**Context reviewed**: `docs/results/tier0_cohort_comparison_optum_vs_synthetic_20260603.md`
**Method**: REASON-BEFORE-RULES + CHEAPEST-DISPROOF-FIRST. Read the code as the source of
truth; **reproduced the symptom by running the real node functions** on a faithful
Optum-like cohort; **two independent adversarial reviewers tried to overturn the
diagnosis**; the surviving claims were re-tested with a controlled cardinality-mix
experiment; the proposed fix was validated with data.

> **Faithfulness caveat (cheapest-disproof discipline).** The real cohort parquet is *not
> committed* to this repo (`data/rwd/**` is empty here), so I could not replay the exact
> Optum bytes. I reproduced with the in-repo node functions on synthetically-constructed
> cohorts whose *distributional profile is matched to the 06-03 doc* (n=1294, 37 positives
> = 2.86%, the Appendix-A cardinality mix). The **code is fully present and is the
> authority**; the reproductions confirm the code's *logic* and land on the user's reported
> "~8". Treat exact counts as faithful-to-the-logic, suggestive-of-the-exact-cohort.

---

## TL;DR — Verdict

**The user is right: the leakage logic is genuinely too aggressive — it deletes legitimate
sparse pre-index predictors as if they were leaks.** But the full "hundreds → 8" is
**over-determined**: it is the sum of a *legitimate* reduction and a *buggy* one, and only
the second is a leakage defect.

| Reduction | ~Count | Mechanism | Bug? |
|---|---|---|---|
| **Legitimate** | ~128 → ~33 | All-constant columns (cardinality ≤1) dropped by `nunique()>1`; too-sparse labs (<50% populated) dropped by `notna()>0.5`. Genuinely zero-information columns. | **No** — correct. This is the "data sparsity" the 06-03 doc describes. |
| **Buggy** | ~33 → ~8 | The leakage layer wrongly flags the **cardinality-2 sparse clinical predictors** (`*_tested`, `has_*`, `dx_*_count`) — which pass every other filter — via `zero_variance_within_class`, and the runner then trains on the survivor set. | **Yes** — three compounding defects below. |

The buggy reduction has **three compounding root causes**:

| # | Root cause | What it is | Evidence |
|---|---|---|---|
| **RC1** | **`check_zero_variance_within_class` is missing the rare-event guard** its twin `check_perfect_class_separation` has | On a rare-event cohort, any *cardinality-2* sparse feature that is all-zero in the tiny positive class fires HIGH "leakage" — a false positive | Identical inputs: `perfect_class_separation` flags **0**, `zero_variance` flags **18/22** of the legit sparse-clinical predictors |
| **RC2** | **The runner trains on the remediation survivor set, not `train_df`** | `run_tier0_test.py` sets `X = eligible_df[leakage_remediated_features]` (the small `verified_features`), **not** `train_df.columns` | `run_tier0_test.py:5841-5843, 5911, 6112-6115` |
| **RC3** | **Declared-safe immunity (#648) has two holes** | (a) the remediation re-check re-runs the unguarded check with **no immunity**; (b) immunity covers only manifest columns (Optum: 63 of ~136), and **nothing** when no manifest resolves | `leakage_remediation.py:985-1042` (no manifest param); `adaptive_validity_check.py:3120-3148` |

**Validated fix (data, not theory):** adding the same rare-event guard to
`check_zero_variance_within_class` raised the recovered set **9 → 53** on the
all-sparse cohort, **while still dropping the genuine post-index leak**.

---

## 1. The question and the user's hypothesis

> "Leakage detection in `step_2_data_preparer` is too aggressive — from hundreds of features
> I only recover 8. I think something is wrong with the leakage detection logic."

The 06-03 doc had concluded the opposite (*"Not a leakage-handling defect … PR #648 fixed
the over-drop"*). Per REASON-BEFORE-RULES the prior verdict was not assumed; per
CHEAPEST-DISPROOF the disproving data was obtained first. **The user's hypothesis survives
for the cardinality-2 sparse-clinical features; the doc's blanket "not a defect" does not.**

---

## 2. Pipeline intent (what leakage detection is *supposed* to do)

From `docs/data/08-LEAKAGE-DETECTION-CONTRACT.md`:

- **Goal**: drop features that leak the outcome (post-index / target-derived); **retain
  legitimate pre-index predictors, even sparse ones.**
- **Core principle (verbatim):** *"A correlation, structural, or FDR detector … cannot
  distinguish a strong, legitimate PRE-INDEX predictor from a genuine leak … Only the
  temporal contract can certify a feature."*
- **It explicitly names this failure mode:** *"the deterministic structural checks
  (`zero_variance_within_class`, `perfect_class_separation`) false-fire on sparse,
  rare-event columns — a column that is constant within the tiny minority class looks
  'tautological' even when it is a legitimate pre-index clinical predictor."*

So deleting sparse legitimate features is **explicitly against intent**. The intended
safety net is declared-safe immunity. The defect is that the net has holes **and** one of
the two structural checks never got the small-N guard the other one has.

---

## 3. The two-stage collapse (where hundreds becomes eight)

There are three feature classes in a real Optum-like cohort (06-03 doc, Appendix A):

| Class | Example columns | Cardinality / population | Fate | Correct? |
|---|---|---|---|---|
| **Dense demographics** | `age_at_index`, `insurance_product`, `payer_category`, `plan_type`, `age_group`, `gender`, `geographic_region` | ≥2 distinct, 100% populated | **survive** | ✅ |
| **Sparse clinical flags** | `*_tested`, `has_asthma`, `dx_l50_8_count`, `charlson_score` | cardinality 2, **100% non-null**, ~0–5% density | **wrongly dropped** | ❌ **the bug** |
| **Constant / empty** | `*_fill_count`, `office_visits_*`, `ed_visits_*` (all-zero); labs at 1–2% non-null | cardinality ≤1, or <50% populated | dropped | ✅ (useless) |

**Stage 1 — legitimate reduction (leakage-independent).** The runner's Step-5 feature
discovery (`run_tier0_test.py:5889-5906`) keeps only `nunique()>1 and notna().mean()>0.5`
columns; the remediation candidate filter (`leakage_remediation.py:864-887`) keeps only
`nunique>1 and null_pct<=50`. These correctly delete the ~58 all-constant families and the
~35 too-sparse labs. **Measured: 128 → 33 survivors (11 dense + 22 sparse-clinical).**
Crucially, the sparse-clinical flags are 100%-non-null and cardinality-2, so they **pass
this filter** — the discovery filter does *not* explain the collapse to 8.

**Stage 2 — buggy reduction (the leakage layer).** Of the 33 survivors, the leakage layer
flags **18 of the 22 sparse-clinical predictors** as HIGH "leakage" (all via
`zero_variance_within_class`), leaving ~11–14. **Measured: 33 → 14.** On the real cohort
(even sparser positives, LLM curation) this lands at ~8 — the dense demographics only,
matching Appendix A's dense block (`insurance_product`, `payer_category`, `age_at_index`,
`plan_type`, `age_group`, `gender`, `geographic_region`, …).

**The runner then trains on Stage-2's survivor list, not `train_df`** (RC2, §4.2).

---

## 4. Root-cause analysis

### RC1 — `check_zero_variance_within_class` is missing the rare-event guard (the false-positive engine)

`check_perfect_class_separation` **has** a small-N guard (`leakage_detector.py:613-627`):

```python
n_unique = feat_valid.nunique()
pos_rate = len(class_1) / max(len(feat_valid), 1)
if n_unique <= 2 and (len(class_1) < 30 or pos_rate < 0.05):
    continue   # rare-event binary: skip — degeneracy is small-sample, not leakage
```

Its structurally-identical twin `check_zero_variance_within_class`
(`leakage_detector.py:729-816`) has **no such guard** — only `len(class_0)>=5 and
len(class_1)>=5`, then fires HIGH when one class has zero variance and the means differ
(`leakage_detector.py:792-811`).

**Why it's a false positive.** With ~30 positives in train+val and a feature at <5%
density, every positive-class row is 0 → `std_1==0, mean_1==0`; the negative class has a
few non-zeros → `std_0>0, mean_0>0`. The condition fires. The feature is constant in the
positive class **because the positive class is tiny and the feature is rare**, not because
it is target-derived — exactly the failure mode the contract warns about.

**This fires only on cardinality-2 sparse features, not all-constant ones.** For an
all-zero column `mean_0==mean_1==0` → `abs(mean_0-mean_1)>1e-10` is False → it does **not**
fire (those are handled by the `nunique>1` filter in Stage 1). This is why the realistic
count is **18/22** of the cardinality-2 clinical flags, not "40/46 of all sparse" — a
correction surfaced by the adversarial review (§6).

**Provenance — oversight, not design.** `git blame` puts both checks in the same change
(`7711b94`): `perfect_class_separation` landed *with* the guard, `zero_variance` landed
*without* it. The reasoning applies equally to both.

**Isolation (real check functions, identical 22 cardinality-2 sparse predictors):**

| Structural check | HIGH/CRIT fired | Guard? |
|---|---|---|
| `perfect_class_separation` | **0 / 22** | yes (L613-627) |
| `zero_variance_within_class` | **18 / 22** | **none** |
| `logical_dependency` | 0 / 22 | n/a (tracks real target-coupling) |

The genuine post-index leak (`initiated_biologic_180d` == target) was still caught — by
`logical_dependency` and `single_feature_auc` — so adding the guard to `zero_variance`
**does not weaken genuine-leak detection.**

### RC2 — The runner trains on the remediation survivor set (the amplifier)

On the runner path the model's feature matrix is the remediation survivor list:

```python
# run_tier0_test.py:5841-5843
remediated = state.get("leakage_remediated_features")
if remediated:
    feature_cols = [f for f in remediated if f in eligible_df.columns]
# :5911  (and again at :6112-6115 after the in-runner re-check)
X = eligible_df[feature_cols].copy()
```

`leakage_remediated_features` = `verified_features` (the survivors of recommended ∩
re-check). So Stage-2's over-flag does not merely drop a few columns — it **defines the
training matrix**.

**This resolves the apparent contradiction with the 06-03 doc.** The doc says Optum
"retained 125 / fits ~136 parameters." That number is `train_df`/`available_features`
(`finalize_output` sets `available_features = list(train_df.columns)`, drops only
leaked+rejected) and the doc's *manual* "restore all features → CV-AUC" probe — **not** the
runner's training matrix. The runner trains on `leakage_remediated_features` (~8). The doc
labeled that count "cosmetic" and never measured it as the training set. **It is not
cosmetic on the runner path** — and `docs/data/08-LEAKAGE-DETECTION-CONTRACT.md`'s "Clean
Features is cosmetic" caveat is wrong here (it is true only for the separate
`MLFoundationPipeline` API path, `src/agents/tier_0/pipeline.py`, which forwards
caller-supplied `train_data`).

### RC3 — Declared-safe immunity (#648) has two holes

**3a — the remediation re-check has no immunity.** Even when `adaptive_validity_check`
strips a declared-safe feature from `leaked_features` (`adaptive_validity_check.py:3885-3901`),
`_apply_leakage_remediation` **re-runs the same structural checks on every recommended
feature** with no manifest parameter (`leakage_remediation.py:995-1042`):

```python
check_findings.extend(check_zero_variance_within_class(combined, target_variable, numeric_feats))
…
if _aggregate_severity(check_findings) in ("critical", "high"):
    rejected_features.append(feat)   # ← a declared-safe sparse feature is re-dropped here
```

Reproduced: a sparse feature placed into `recommended_feature_set` (simulating immunity
having kept it) is **rejected again** by this re-check. #648's immunity was validated only
on *dense* synthetic predictors (`days_on_therapy`, `hcp_visits`, `prior_treatments`,
which never trip `zero_variance`), so this hole was never exercised. **This is the
cross-path survivor: it defeats the over-drop fix even when a manifest is present.**

**3b — coverage gap.** `_declared_safe_immune_features` (`adaptive_validity_check.py:3120-3148`)
rescues only features with a `FeatureContract` in the active manifest. Optum declares **63**
contracts vs ~136 columns; and it returns the empty set when `manifest_source` is falsy —
so a run with **no resolved manifest** (no `--feature-manifest-source`, unrecognized
`--data-dir`) grants **no immunity to anything**, and the full RC1 over-fire is live (the
path my reproductions exercise).

---

## 5. Empirical reproduction (the cheap disproof)

Real node functions (`detect_leakage`, `review_and_remediate_leakage`) loaded from the repo
via `importlib`; only the `DataPreparerState` *type annotation* was stubbed (no pipeline
code changed). Realistic cohort: 1294 rows, 37 positives (2.86%), 128 columns = 11 dense +
22 cardinality-2 sparse-clinical + 58 all-constant + 35 sparse labs + 1 leak; **no manifest.**

| Mechanism | Measured |
|---|---|
| **Stage 1** — runner discovery filter (`nunique>1 & notna>0.5`), leakage-independent | 128 → **33** survivors (11 dense + **all 22** sparse-clinical kept; 58 const + 35 labs dropped) |
| **Stage 2** — `detect_leakage` (no manifest) | leaked = 19; `zero_variance_within_class` flags **18/22** sparse-clinical; genuine leak caught by `logical_dependency`/`single_feature_auc` |
| **Stage 2** — `leakage_remediation` → `leakage_remediated_features` | **14** (11 dense + 3 lucky sparse) |
| **Features the runner trains on** (`run_tier0_test.py:5911`) | **14 of 128** (→ ~8 on the real cohort with LLM curation + sparser positives) |

The collapse reproduces with the **deterministic rule-based** remediation path — no LLM key
needed; the over-drop does not depend on the LLM.

### Fix validation (validate the solution, per cheapest-disproof)

Same machinery, rare-event guard monkey-patched onto `check_zero_variance_within_class`
(repo file untouched), on the all-cardinality-2 sparse cohort:

| | `leaked_features` | recovered / "Clean Features" | runner trains on |
|---|---|---|---|
| **Before (current code)** | 41 | 9 | **9** |
| **After (guard added)** | **1** (only the genuine post-index leak) | **53** | **53** |

The guard recovers the sparse pre-index predictors and **still drops the genuine leak**.
All-constant families remain correctly excluded (as `nunique<=1`).

---

## 6. Adversarial review (two reviewers tried to overturn this)

Per the anti-mocking / verification discipline, the diagnosis was attacked, not defended.
The challenges **materially sharpened** it; the core survived.

**Challenge A — "the over-drop is correct / `zero_variance` barely fires."** *Partly upheld,
and incorporated.* The all-constant majority (`mean_0==mean_1`) does **not** trip
`zero_variance`; it is dropped by `nunique>1` (Stage 1, legitimate). So the original "40/46
of all sparse" over-counted — the true target is the **cardinality-2** sparse-clinical
class, where the realistic figure is **18/22** (re-measured in §5). The *narrow code fact*
(missing guard) stands, and dropping those features is against the contract's stated intent.

**Challenge A — "manifest immunity rescues them, so it's moot; the doc shows 125/136
retained."** *Rejected.* (i) Immunity strips the first flag but the remediation re-check
re-drops them with no immunity (RC3a, reproduced). (ii) No-manifest runs get no immunity at
all (RC3b). (iii) The doc's "125/136" is `train_df`/a manual probe, **not** the runner's
training matrix (`leakage_remediated_features`); conflating the two is the same error the
doc made (RC2).

**Challenge B — "a leakage-independent runner filter is the real, dominant cause."**
*Adjudicated and rejected as the cause of the bug.* The discovery filter (`nunique>1 &
notna>0.5`) **keeps all 22 sparse-clinical flags** (they are 100%-non-null, cardinality-2)
— it leaves **33**, not 8. Only the leakage layer drops the sparse-clinical predictors
(§5). Challenge B is *correct* that the bulk reduction (constants/labs, 128→33) is
legitimate and leakage-independent — that is now Stage 1 — but it does **not** explain the
collapse to ~8; the leakage over-fire does.

**Challenges that fully failed:** "`std()` returns NaN so it never fires" (for an all-zero
class of ~30, `std()` is `0.0`, not NaN; the `<1e-10` branch fires); "the 8 is just the LLM
curating down" (reproduced with the deterministic rule-based path, no LLM).

Net: the **only** load-bearing claim that changed is *magnitude/targeting* — the bug hits
the cardinality-2 sparse-clinical class (~18–22 features), not "all sparse." RC1, RC2, RC3
stand.

---

## 7. Honest scope — what the fix does and does not do

- **It fixes the user's symptom**: the cardinality-2 sparse pre-index predictors stop being
  deleted as "leakage"; recovered count rises from ~8 to the full legitimate set.
- **It does NOT make the 37-event Optum cohort modelable.** EPV ≈ 0.13 stands; the 06-03
  doc's data-volume verdict is *independent* of this bug and remains valid. (On *this*
  cohort the wrongly-dropped flags are also low-signal — AUC ≈ 0.50 — so modelability is
  unchanged either way; but they are dropped for the **wrong reason**, and the leakage
  audit trail is lying about why.)
- **It matters most for OTHER cohorts.** Any rare-event cohort with *informative* sparse
  pre-index predictors is currently having them silently deleted as "leakage." That is the
  real ongoing harm, and the reason "the features were weak anyway" does not excuse it.

---

## 8. Definitive solution

Ordered by leverage. (Per the mission, code is **not** modified here; these are prescribed.)

**Fix 1 (primary, ~6 lines) — give `check_zero_variance_within_class` the same rare-event
guard as `check_perfect_class_separation`.** In `leakage_detector.py:check_zero_variance_within_class`,
after the class split:

```python
n_unique = feat_valid.nunique()
pos_rate = len(class_1) / max(len(feat_valid), 1)
if n_unique <= 2 and (len(class_1) < 30 or pos_rate < 0.05):
    continue   # rare-event binary: zero within-class variance is small-sample, not leakage
```

Apply the same guard to `check_feature_target_logical_dependency` for consistency (latent
twin). **Validated**: 9 → 53 recovered, genuine leak still dropped (§5).

**Fix 2 (close the immunity hole, RC3a) — thread `manifest_source` into
`_apply_leakage_remediation` and skip the structural re-check for declared-safe features**
(reuse `_declared_safe_immune_features`). Without this, Fix 1 still leaves *declared-safe
sparse* features re-dropped by the re-check on manifest runs.

**Fix 3 (decouple "is a leak" from "is a training feature", RC2) — on the runner path,
train on the retained post-remediation feature columns, not the curated `verified_features`
subset.** Make `leakage_remediated_features` equal "all retained features (minus genuine
leaks)", or change `run_tier0_test.py:5841-5843/6112-6115` to select from the
post-remediation `train_df` columns. Also correct
`docs/data/08-LEAKAGE-DETECTION-CONTRACT.md`: the "Clean Features count is cosmetic" caveat
is false on the runner path.

**Fix 4 (defense in depth) — when `pos_rate < ~5%`, demote structural
`zero_variance`/`perfect_separation` from auto-drop (HIGH) to review (MODERATE)** and let
the temporal manifest + FDR (which already carries HBLP small-N inflation) be the auto-fire
authority. On rare-event cohorts the within-class-variance premise is statistically
unreliable.

**Regression coverage to add**: a rare-event fixture (n≈1000, prevalence≈3%, several
*cardinality-2* sparse-but-legitimate clinical flags) asserting `detect_leakage` flags
**none** of the sparse non-leaks while still catching an injected post-index leak — the
test #648 lacked (its cohort was dense, so the bug passed).

---

## Appendix A — Key citations

- `src/agents/ml_foundation/data_preparer/nodes/leakage_detector.py`
  - `check_perfect_class_separation` rare-event guard: **613-627**
  - `check_zero_variance_within_class` (unguarded twin): **729-816** (fire at **792-811**)
- `src/agents/ml_foundation/data_preparer/nodes/leakage_remediation.py`
  - immunity-less structural re-check: **985-1042**; drop set: **1156-1159**
  - rule-based `recommended_feature_set` filter (`null_pct<=50 and nunique>1`): **864-887**
- `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py`
  - `_declared_safe_immune_features` (immunity, manifest-gated): **3120-3148**
  - immunity application + severity downgrade: **3885-3901**
- `scripts/run_tier0_test.py`
  - `scope_spec`: `required_features = available_features`, `excluded_features = non-manifest`: **2261-2281**
  - runner selects X from `leakage_remediated_features`: **5841-5843, 5911**; second override **6112-6115**
  - leakage-independent discovery filter (Stage 1): **5889-5906**; Step-5a remediation gate: **6002**
  - "Clean Features" console line: **7221**
- `src/data/manifests/optum_feature_manifest.py`: 63 declared `FeatureContract`s vs ~136 parquet columns
- Provenance of the guard asymmetry: `git blame` → `7711b94` (guard on `perfect_class_separation` only)

## Appendix B — Reproduction scripts (load the real node functions; need only numpy/pandas/scikit-learn/scipy)

- `/tmp/repro_leakage.py` — end-to-end over-drop (5% sparse density → 22 recovered)
- `/tmp/repro_leakage_v2.py` — faithful all-card-2 sparse (<1% density → **9**) + guard-asymmetry + immunity-gap demo
- `/tmp/repro_fix.py` — fix validation (guard added → **53**, leak still dropped)
- `/tmp/repro_v4.py` — **adjudication** with the realistic Appendix-A cardinality mix: discovery filter → 33 (keeps all sparse-clinical), leakage → drops 18/22 → 14

Run in a venv with `numpy pandas scikit-learn scipy`, e.g. `python /tmp/repro_v4.py`.
