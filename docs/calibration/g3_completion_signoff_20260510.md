# Plan v4 §2 Gate G3 — Completion Sign-off

**Status:** TEMPLATE / PENDING N3 REVIEWER SIGNATURE

**Subject:** Tier 1B HBLP default-path wiring per Plan v4 §2 G3.

**Date:** 2026-05-10

**Branch / commit:** *(to be filled at sign-off — the wiring PR's
merged-commit SHA)*

> **WARNING:** This signoff is INTERIM. The cryptographic signature
> requirement is gated on the N3 reviewer registry + CoI infrastructure
> deferred to backlog `v4-N3-signature-infra` (per Plan v4 §A status,
> 4 N3 partials require infrastructure decisions documented at
> `docs/governance/n3_known_limitations_20260510.md`). Until that infra
> lands, this signoff is recorded with placeholders so the G3 wiring
> guard (`.github/workflows/g3_wiring_guard.yml`) can detect the file's
> existence and required-fields shape, but the reviewer-cryptographic-
> signature gate is documented-not-enforced.

## Plan v4 §2 G3 acceptance criteria — closure status

| # | Criterion | Closure artifact | Status |
|---|---|---|---|
| 1 | `_build_verdict` and `_compose_legacy_verdict` route through `hblp_classify` | `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py` (commit `<commit>`) | **CLOSED** |
| 2 | Legacy `if z > HIGH_Z` branch removed (no parallel paths) | same file — `_adversarial_input` body | **CLOSED** (legacy 3-arm `if/elif/else` ladder replaced by single `hblp_classify` call) |
| 3 | Caller threading: `n_train_pos` and `layer_1_declared_safe` propagated from orchestrator | `adaptive_validity_check` orchestrator (n_train_pos computed once at entry; layer_1_declared_safe per-feature from manifest contract) | **CLOSED** |
| 4 | Three-cohort regression sweep clean (no MARGINAL→GENUINE flips on synthetic; CSU verdict unchanged; Optum verdict unchanged or improved) | `tests/integration/test_g3_three_cohort_regression_sweep.py` (33 tests) | **CLOSED** |
| 5 | Mechanical CI enforcement: `.github/workflows/g3_wiring_guard.yml` AST-scans the gated file and FAILS the build unless G1+G2 signoff files exist with valid ancestor SHAs | `scripts/check_g3_wiring_guard.py` + `.github/workflows/g3_wiring_guard.yml` (36 unit tests) | **CLOSED** |

## Pre-condition (Plan v4 §3 sequencing)

Plan v4 §3 sequencing rule:

> G3 must NOT land until G1 + G2 close.

**G1 status:** *(to be filled at sign-off — verify
`docs/calibration/g1_completion_signoff_20260510.md` exists at HEAD with
valid ancestor SHA)*

**G2 status:** *(to be filled at sign-off — verify
`docs/calibration/g2_completion_signoff_20260510.md` exists at HEAD with
valid ancestor SHA)*

The mechanical CI guard at `.github/workflows/g3_wiring_guard.yml`
enforces this pre-condition automatically on every PR. Both signoff
files MUST exist on `main` (with their `commit:` SHAs as ancestors of
the wiring PR's HEAD) before the guard transitions from FAIL to PASS.

## Test counts

```
tests/integration/test_g3_three_cohort_regression_sweep.py    33 tests (31 pass + 2 real_data skip)
tests/scripts/test_check_g3_wiring_guard.py                   36 tests (all pass; synthetic git fixtures)
                                                              -------
Total NEW tests under v4-g3-phase-c                          69 tests
```

Cross-suite regression: 290 existing data-preparer tests pass with
zero regressions after the wiring refactor.

## Refactor summary

### Files modified

- `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py`:
  - `_adversarial_input` now accepts `n_train_pos` + `layer_1_declared_safe`
    keyword args; routes severity classification through `hblp_classify`;
    legacy 3-arm `if z > HIGH_Z / elif z > MODERATE_Z / else` ladder removed.
  - `_compose_legacy_verdict` accepts and forwards both args.
  - `_build_verdict` accepts and forwards both args (defaults reproduce
    legacy fixed-threshold behaviour for ad-hoc tests).
  - Orchestrator `adaptive_validity_check` computes `n_train_pos` once at
    entry from the binary-label-mask-filtered target, derives
    `layer_1_declared_safe` per-feature from the manifest contract
    (`contract.knowable_at.is_pre_or_at_index()` when contract present;
    False otherwise), and threads both into the Layer 3 pass.

### Files added

- `scripts/check_g3_wiring_guard.py` — AST scanner + signoff-file
  existence/ancestor checks.
- `.github/workflows/g3_wiring_guard.yml` — CI workflow invoking the
  scanner with H3 base-ref-pinned-validator mitigation.
- `tests/scripts/__init__.py` + `tests/scripts/test_check_g3_wiring_guard.py`
  — 36 unit tests for the guard.
- `tests/integration/test_g3_three_cohort_regression_sweep.py` — 33
  tests pinning the post-G3 verdict matrix.

### Lines deleted from legacy branch

- 3-arm `if z > HIGH_Z / elif z > MODERATE_Z / else` ladder: ~25 lines
  of severity/remediation/evidence dispatch replaced by a single
  `hblp_classify` call + per-severity narrative builder.

## Tier 1 invariants honored

- `hblp_classify` signature is unchanged — G3 wires existing helper
  into production codepath; does not modify the helper itself.
- Advisory paths (Layer 1 manifest verdicts, short-circuit too-few-rows /
  scoring-error verdicts, degenerate-score verdicts) untouched.
- `n_train_pos=None` + `layer_1_declared_safe=False` (default kwargs)
  reproduce legacy fixed 5σ/3σ thresholds — backward-compatible for any
  caller that hasn't been updated to thread cohort metadata.

## Reviewer (deferred to N3 infra)

- **Name:** `<full legal name>` — to be filled when N3 reviewer
  registry lands.
- **GitHub handle:** `@<github_handle>`
- **Registry row:** `docs/governance/methodology_reviewer_registry.md`
  (template currently; CI guard `--require-signature-registry-match`
  flag not yet enforced — promotes from advisory-warn to fail-closed
  once registry has active rows).
- **CoI declaration commit SHA:** `<sha>` (deferred to N3 infra).

## Cryptographic signature (deferred to N3 infra)

The Plan v4 §N3 spec requires PGP or sigstore signature on this file.
That requirement is currently **documented but not enforced** because:

1. The reviewer registry (`docs/governance/methodology_reviewer_registry.md`)
   exists as a template only — no `active` rows yet.
2. The G3 wiring-guard workflow's `--require-signature-registry-match`
   flag is opt-in; the default policy is advisory-warn.

The signature block below is therefore left as a **placeholder** with
the schema the future enforcement will validate:

```
-----BEGIN PGP SIGNATURE-----

<signature blob — to be filled by signing reviewer when infra lands>

-----END PGP SIGNATURE-----
```

## Acceptance checklist

- [x] Plan v4 §2 G3 criterion 1 (`_build_verdict` + `_compose_legacy_verdict`
      route through `hblp_classify`) — see refactor summary above
- [x] Plan v4 §2 G3 criterion 2 (legacy `if z > HIGH_Z` branch removed) —
      see lines-deleted summary above
- [x] Plan v4 §2 G3 criterion 3 (caller threading from orchestrator) —
      see refactor summary above
- [x] Plan v4 §2 G3 criterion 4 (three-cohort regression sweep clean) —
      `tests/integration/test_g3_three_cohort_regression_sweep.py`
- [x] Plan v4 §2 G3 criterion 5 (mechanical CI enforcement) —
      `.github/workflows/g3_wiring_guard.yml` + `scripts/check_g3_wiring_guard.py`
- [x] All new tests pass locally (`pytest -v` on the two new files)
- [x] Existing data-preparer test suite continues to pass (290 tests, 5 skipped)
- [x] `ruff format --check` clean on touched files
- [x] `ruff check` clean on touched files
- [ ] Cryptographic signature — DEFERRED (gated on N3 infrastructure)
- [ ] G1 signoff exists at HEAD with valid ancestor SHA — DEFERRED
      (gated on PR #137 merging to main)
- [ ] G2 signoff exists at HEAD with valid ancestor SHA — DEFERRED
      (gated on PR #136 merging to main)

## Cross-references

- **Plan:** `.claude/plans/disease_agnostic_quality_uplift_v4.md` §2 Gate G3
- **Plan v3 baseline:** `.claude/plans/disease_agnostic_quality_uplift_v3.md`
  §3 Tier 1B step 2 (HBLP rationale + variance-inflation derivation)
- **Empirical anchor:** `docs/results/optum_initiation_revalidation_20260510.md`
  (the Optum n=22 small-N permutation-null variance scaling that motivated
  the HBLP design)
- **G1 signoff:** `docs/calibration/g1_completion_signoff_20260510.md` (PR #137)
- **G2 signoff:** `docs/calibration/g2_completion_signoff_20260510.md` (PR #136)
- **HBLP helpers (PR #127):** `hblp_classify` + `hblp_effective_z_threshold` +
  `lineage_audit_declared_path`
- **N3 known limitations:** `docs/governance/n3_known_limitations_20260510.md`
- **Companion gates:** G1 (`v4-g1-phase-b`), G2 (`v4-g2-phase-b`)
