# G1 Optum baseline — held-out non-inferiority anchor

**Status:** ENGINEERING_COMPLETE / SIGNATURE_PENDING (pending N3 reviewer infra)

**Subject:** Plan v4 §2 Gate G1 acceptance criterion (Optum half) — empirical
baseline for the held-out non-inferiority gate.

**Date:** 2026-05-10

**Branch / commit:** `v4-g1-phase-b` (codex pass-1 MED-9 closure)

**Related:** [`docs/calibration/g1_completion_signoff_20260510.md`](g1_completion_signoff_20260510.md)

---

## Baseline metrics — held-out test split

| Field | Value | Source |
|-------|-------|--------|
| `auc_value` | 0.4347 | `docs/results/optum_initiation_revalidation_20260510.md` table line 24 |
| `split_name` | held_out_test | `scripts/run_tier0_test.py:5977` (`test_metrics`) |
| `cohort_n` | 1294 | PR #116 closure record (default-window cohort) |
| `target` | initiated_biologic_180d | `COHORT_TARGETS["initiation"]` in `scripts/run_optum_tier0_test.py` |
| `window_regime` | default | PRE=360 / POST=180 |
| `pr_reference` | PR #116 | merge commit `0dc85a4` (backlog #19 closure) |
| `n_train_pos` | ≈22 | PR #116 closure (n_test=195, ~6 positives in test split) |

## Non-inferiority slack

| Field | Value | Rationale |
|-------|-------|-----------|
| `epsilon` | 0.02 | Plan v4 §2 G1 explicit pin |
| `floor` | 0.4147 | `auc_value − epsilon` |

The 0.02 slack is the seed-to-seed noise floor on a held-out test set with
~6 positives. On so few positives a single misclassified case shifts AUC by
~0.05+; the 0.02 slack is the best-case noise floor v4 G1 accepts as
non-inferiority.

## CSU companion (negative-control)

For symmetry with the Optum non-inferiority gate, the CSU
negative-control gate's numerical tolerances are documented here:

| Field | Value | Rationale |
|-------|-------|-----------|
| CSU `auc_min` | 0.62 | PR #106 closure honest band lower edge |
| CSU `auc_max` | 0.68 | PR #106 closure honest band upper edge (CSU literature anchors: psoriasis 0.67, AD 0.63, severe asthma 0.66) |
| CSU `auc_anchor` | 0.6592 | val_AUC at PR #106 merge |
| CSU `permutation_p_max` | 0.01 | Effective floor of 100-perm test (1/200 = 0.005); ≤ 0.01 = "indistinguishable from <0.001" |
| CSU `expected_deployer_verdict` | ACCEPTABLE | `_compute_verdict(auc_roc≥0.65, recall≥0.3, precision≥0.05)` → ACCEPTABLE |
| CSU `cohort_n` | 9607 | PR #116 closure baseline |

## What the test enforces (codex pass-1 MED-9)

`tests/integration/test_optum_held_out_noninferiority_20260510.py`
reads this baseline file's metric values directly via the constants
`OPTUM_BASELINE_HELDOUT_AUC`, `OPTUM_NONINFERIORITY_EPSILON`,
`OPTUM_HELDOUT_AUC_FLOOR`, and `OPTUM_EXPECTED_DEFAULT_COHORT_SIZE`.
A new `test_optum_baseline_artifact_present_and_complete` check in
that file validates the baseline document exists and contains all
required fields (artifact-present + non-empty assertions).

If any field above is updated, the test must be updated in the same
commit; the baseline document is the single source of truth for the
G1 Optum non-inferiority anchor. Test constants must NOT drift from
this document.

## N3 sign-off note

This baseline encodes the empirical anchor at PR #116 closure. The
decision to use the default-window cohort (n=1294) instead of the
relaxed-window cohort (n=1697) is itself an N3 methodology decision
documented at:

- `docs/results/optum_initiation_revalidation_20260510.md` — empirical comparison
- `.claude/plans/disease_agnostic_quality_uplift_v4.md` §5 — "data_snooped" risk on n=1697
- `docs/governance/n3_known_limitations_20260510.md` — N3 reviewer infra status

The relaxed-window cohort is forbidden as a regression baseline until
the N3 reviewer registry + CoI infrastructure lands. Once it does,
this document will be updated with a SIGNED status field referencing
the signed N3 decision.

## Cross-references

- **Plan:** `.claude/plans/disease_agnostic_quality_uplift_v4.md` §2 Gate G1 (Optum half)
- **Baseline empirical anchor:** `docs/results/optum_initiation_revalidation_20260510.md`
- **Companion test:** `tests/integration/test_optum_held_out_noninferiority_20260510.py`
- **CSU companion test:** `tests/integration/test_csu_negative_control_20260510.py`
- **G1 signoff:** `docs/calibration/g1_completion_signoff_20260510.md`

## Sign-off block (deferred)

```
status: ENGINEERING_COMPLETE
signature_pending_until: N3_reviewer_registry_landed
verifier_pgp: <reviewer fingerprint — to be filled by signing reviewer when infra lands>
verifier_github_handle: <@github_handle>
verification_date: <YYYY-MM-DD>
```
