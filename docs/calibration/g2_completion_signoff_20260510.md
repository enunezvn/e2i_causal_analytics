# Gate G2 — Completion Sign-off (HUMAN signature deferred)

**Status:** Sign-off TEMPLATE. Until the first green G2 run from
`S_prespec` lands and a human reviewer signs this artifact per the
plan-v4 N3 reviewer-registry policy, this document is a SCAFFOLD only;
its presence does NOT promote the G2 lifecycle state from `ADVISORY` to
`ENFORCED`.

**Plan reference:** `.claude/plans/disease_agnostic_quality_uplift_v4.md`
§2 Gate G2 + §3 sequencing + §8 acceptance summary.

**Branch / commit:** `e75c99170d0e9fb31710a47303f90e11647e40cc`
(G2 merge commit on `origin/main`, merged 2026-05-11 11:07:17Z via
`--rebase`; originally from branch `v4-g2-phase-b`).
The G3 wiring-guard CI workflow verifies this SHA is an ancestor of
the wiring-PR HEAD per the v4 §3 sequencing rule.

---

## 1. Pre-spec memo

- **Pre-spec memo path:** `docs/specs/tier1b_b2_prespec_20260510.md`
- **`S_prespec` commit SHA:** *(to be filled in at sign-off — the
  introducing commit of the pre-spec memo)*
- **Allowed-update procedure:** edits to the memo's load-bearing
  values require a NEW memo at a fresh date per v3 §8. The
  threshold-shopping audit reads the introducing commit of the memo
  in this fresh-date filename to determine `S_prespec`.

## 2. Experiment commit + run

- **Experiment commit SHA:** *(to be filled in at sign-off)*
- **Tag name:** *(`tier1b-b2-experiment-<n>`)*
- **CI workflow run URL:** *(to be filled in at sign-off)*
- **Manifest artifact attachment:** *(g2_run_manifest.json from the
  uploaded artifact)*

## 3. Pre-spec parent-check evidence

- **`scripts/check_g2_commit_graph.py` output:**

```
*(paste the [OK] line from the CI run's "Verify commit-graph parent
constraint" step here)*
```

- **Parent-graph check verified:** *(YES/NO)* (renamed from "Commit-graph parent verified" to avoid collision with the G3 wiring-guard's commit-field regex)
- **Verifier evidence:**
  - `scripts/check_g2_commit_graph.py` exit code: *(0 = pass)*
  - `git merge-base --is-ancestor S_prespec experiment_sha` exit code:
    *(0 = pass)*

## 4. Dataset content-hash evidence

- **`scripts/verify_g2_prespec_dataset_hashes.py` output:**

```
*(paste the [OK] lines from the CI run's "Verify cohort dataset
hashes vs pre-spec memo" step here)*
```

- **Pinned hashes (from memo §5):**
  - `optum_initiation_patient_journeys_parquet`:
    *(paste pinned sha256)*
  - `optum_initiation_treatment_events_parquet`:
    *(paste pinned sha256)*
- **Live hashes at experiment SHA:**
  - `optum_initiation_patient_journeys_parquet`:
    *(paste live sha256, or MISSING)*
  - `optum_initiation_treatment_events_parquet`:
    *(paste live sha256, or MISSING)*

## 5. Threshold evaluation (pre-spec §1)

| # | Threshold | Pre value | Post value | Δ / ratio | Pass? |
|---|-----------|-----------|------------|-----------|-------|
| **T1** | dAUC >= 0.03 | *(baseline_auc_mean)* | *(hblp_auc_mean)* | *(post - pre)* | *(YES/NO)* |
| **T2** | ECE_post <= 0.5 × ECE_pre | *(baseline_ece_mean)* | *(hblp_ece_mean)* | *(post / pre)* | *(YES/NO)* |
| **T3** | (std/mean)_post <= 0.7 × (std/mean)_pre | *(baseline_cv_mean)* | *(hblp_cv_mean)* | *(post / pre)* | *(YES/NO)* |

- **Combined `g2_passes_pre_spec`:** *(YES iff T1 AND T2 AND T3)*

## 6. Cohort identifiers

- **Cohort label:** `optum_initiation_default`
- **Cohort path:** `data/rwd/optum/initiation/`
- **Cohort N:** *(observed patient count)*
- **Target column:** `treatment_initiated`
- **Data snooped flag:** `false` (per pre-spec §2.1)

## 7. Reviewer

- **Reviewer name:**
- **GitHub handle:**
- **Email:**
- **Role:**
- **Reviewer registry entry SHA:** *(cross-reference the entry in
  `docs/governance/methodology_reviewer_registry.md` — N3 policy)*
- **CoI declaration commit SHA:** *(cross-reference)*
- **CoI selection-rule evidence:** *(`git log --author=<email>` output
  showing 0 touches to `scripts/run_tier1b_b2_experiment.py` /
  `scripts/check_g2_commit_graph.py` /
  `scripts/verify_g2_prespec_dataset_hashes.py` /
  `docs/specs/tier1b_b2_prespec_20260510.md` between the period of
  pre-spec authoring and the sign-off date)*

## 8. Cryptographic signature

- **Signing method:** *(PGP / sigstore)*
- **Signature block:**

```
-----BEGIN PGP SIGNED MESSAGE-----
*(or sigstore equivalent — to be filled in at sign-off)*
-----END PGP SIGNED MESSAGE-----
```

- **Public key fingerprint:** *(cross-reference to the reviewer
  registry's listed key)*

## 9. Lifecycle-state transition

If this sign-off lands cleanly:

- The G2 experiment workflow's `LIFECYCLE_STATE_G2` MAY be promoted
  from `ADVISORY` to `ENFORCED` via a separate diff that updates
  `scripts/run_tier1b_b2_experiment.py` (the lifecycle constant
  declaration) AND adds a corresponding lifecycle-change doc per the
  N2 policy at
  `docs/calibration/G2_lifecycle_change_advisory_to_enforced_<date>.md`.
- The lifecycle-change doc must include start_date, end_date,
  drift_summary, signing_reviewer (cross-referenced to the registry).
- The CI lifecycle-state-guard scan
  (`.github/workflows/lifecycle_state_guard.yml`) enforces the doc's
  presence on PRs that change the lifecycle constant.

## 10. Plan-sequencing implication

Per v4 §3 sequencing:

- G3 (HBLP default-path wiring) is BLOCKED until G1 + G2 close.
- G2 closure = this sign-off lands AND the corresponding
  manifest passes T1/T2/T3 AND the lifecycle constant is promoted to
  `ENFORCED` (or remains in `ADVISORY` per operator preference if the
  manifest fails).
- A G2 manifest that fails T1/T2/T3 is itself a v3 §6 acceptance
  failure: G3 cannot proceed.

## 11. Acceptance checklist (HUMAN to verify before signing)

- [ ] Pre-spec memo's introducing commit (`S_prespec`) is the same
      SHA the experiment commit's parent-check resolved against.
- [ ] Dataset content hashes at the memo were pinned via a SEPARATE
      diff (no threshold edits in the same commit).
- [ ] Experiment commit is a strict descendant of `S_prespec` (not an
      alias).
- [ ] Tag annotation references `S_prespec` (full or short SHA).
- [ ] Manifest artifact captured all 5 seeds and aggregated at
      seed-mean per pre-spec §3.
- [ ] T1/T2/T3 evaluations match the pre-spec memo §1.
- [ ] Reviewer's CoI declaration shows 0 touches to G2 verifier
      scripts in the named period (per N3 selection-rule).
- [ ] Reviewer is listed in `docs/governance/methodology_reviewer_registry.md`.
- [ ] Cryptographic signature verifies against the registered public
      key.
- [ ] Lifecycle-change doc (if promoting to ENFORCED) is co-landed in
      a follow-up diff per the N2 policy.

---

**Until the HUMAN signature lands, this artifact is a SCAFFOLD ONLY
and does NOT close the G2 acceptance criterion.**
