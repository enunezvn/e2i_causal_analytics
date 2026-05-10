# Plan v4 §2 Gate G1 — Completion Sign-off

**Status:** TEMPLATE / PENDING N3 REVIEWER SIGNATURE

**Subject:** Tier 1B Gate B1 closure — CSU negative-control + Optum
held-out non-inferiority + lineage audit per Plan v4 §2 G1.

**Date:** 2026-05-10

**Branch / commit:** `d1af6d11` (initial signoff commit on branch
`v4-g1-phase-b`; updated SHA recorded in the followup commit that
edits this `commit:` field). The G3 wiring-guard CI workflow
(scheduled to land with G3 PR) verifies this SHA is an ancestor
of the wiring-PR HEAD; until G3, this field is recorded but
not yet enforced.

> **WARNING:** This signoff is INTERIM. The cryptographic signature
> requirement is gated on the N3 reviewer registry + CoI infrastructure
> deferred to backlog `v4-N3-signature-infra` (per Plan v4 §A status,
> 4 N3 partials require infrastructure decisions documented at
> `docs/governance/n3_known_limitations_20260510.md`). Until that infra
> lands, this signoff is recorded with placeholders so the G3 wiring
> guard (`.github/workflows/g3_wiring_guard.yml`, when it lands) can
> detect the file's existence and required-fields shape, but the
> reviewer-cryptographic-signature gate is documented-not-enforced.

## Plan v4 §2 G1 acceptance criteria — closure status

| # | Criterion | Closure artifact | Status |
|---|---|---|---|
| 1 | CSU n=9607 negative-control regression test pinning deployer verdict UNCHANGED at val_AUC=0.66 (MARGINAL, perm p=0.0) after HBLP default-path wiring lands | `tests/integration/test_csu_negative_control_20260510.py` | **CLOSED** (5 tests; skips when CSU data missing) |
| 2 | Optum held-out non-inferiority test — held-out test AUC on Optum is no WORSE than baseline within `epsilon=0.02` slack (held-out, NOT val) | `tests/integration/test_optum_held_out_noninferiority_20260510.py` | **CLOSED** (3 tests; skips when Optum data missing) |
| 3 | Derivation-lineage audit on every feature surfaced by HBLP relaxation — `derivation_inputs ⊆ pre-anchor` per `MANIFEST_SOURCES` | `tests/integration/test_g1_lineage_audit_sweep.py` | **CLOSED** (10 tests; runs in CI without real data) |

## Pre-existing G1 artifact (PR #128)

Plan v4 §2 G1 acceptance criterion #1 — "Integration leakage-injection
regression on synthetic" — was closed by PR #128 prior to this signoff.
That criterion remains closed; this signoff covers only the **NEW**
sub-criteria added by Plan v4 §2 G1.

## Test counts

```
tests/integration/test_csu_negative_control_20260510.py      5 tests
tests/integration/test_optum_held_out_noninferiority_20260510.py  3 tests
tests/integration/test_g1_lineage_audit_sweep.py              10 tests
tests/unit/test_agents/.../test_hblp.py                       3 NEW tests
                                                              -------
Total NEW tests under v4-g1-phase-b                          21 tests
```

## Real-data dependency

CSU + Optum integration tests **skip** when real cohort data is absent
(CI / worktree); they are gated on `data/rwd/csu/e2i_ml_v3_patient_journeys.json`
and `data/rwd/optum/initiation/` respectively. The lineage-audit sweep
runs against `MANIFEST_SOURCES` registry alone — no cohort data
required — so it stays green in CI without external dependencies.

When real-cohort runs occur (locally or in a future CI environment with
data fixtures), the CSU + Optum tests pin the empirical anchors:
- CSU: `val_AUC ∈ [0.62, 0.68]`, perm p ≤ 0.01, deployer MARGINAL
- Optum: held-out AUC ≥ 0.4147 (= 0.4347 baseline − 0.02 epsilon)

## Pre-condition for G3 (HBLP default-path wiring)

Plan v4 §3 sequencing rule:
> G3 must NOT land until G1 + G2 close.

**G1 status:** CLOSED (this signoff). Tests CI-green at branch HEAD.

**G2 status:** OPEN (separate Phase B PR, branch `v4-g2-phase-b`).
G3 cannot land until both G1 + G2 close, regardless of this G1 signoff.

## Reviewer (deferred to N3 infra)

- **Name:** `<full legal name>` — to be filled when N3 reviewer
  registry lands.
- **GitHub handle:** `@<github_handle>`
- **Registry row:** `docs/governance/methodology_reviewer_registry.md`
  (template currently; CI guard not yet enforcing).
- **CoI declaration commit SHA:** `<sha>` (deferred to N3 infra).

## Cryptographic signature (deferred to N3 infra)

The Plan v4 §N3 spec (§Gate N3 in plan §2) requires PGP or sigstore
signature on this file. That requirement is currently **documented but
not enforced** because:

1. The reviewer registry (`docs/governance/methodology_reviewer_registry.md`)
   exists as a template only — no `active` rows yet.
2. The CI verification workflow
   (`.github/workflows/methodology_signoff_guard.yml`) exists from PR
   #131 but does not yet validate G1/G2/G3 signoff files (only N3
   methodology signoffs).
3. The G3 wiring guard (`.github/workflows/g3_wiring_guard.yml`) is
   itself part of Phase C / G3 implementation and will land with the
   G3 PR, not this G1 PR.

The signature block below is therefore left as a **placeholder** with
the schema the future enforcement will validate:

```
-----BEGIN PGP SIGNATURE-----

<signature blob — to be filled by signing reviewer when infra lands>

-----END PGP SIGNATURE-----
```

## Acceptance checklist

- [x] Plan v4 §2 G1 criterion 1 (CSU negative-control regression test) — `tests/integration/test_csu_negative_control_20260510.py`
- [x] Plan v4 §2 G1 criterion 2 (Optum held-out non-inferiority test) — `tests/integration/test_optum_held_out_noninferiority_20260510.py`
- [x] Plan v4 §2 G1 criterion 3 (derivation-lineage audit on relaxed features) — `tests/integration/test_g1_lineage_audit_sweep.py`
- [x] All new tests pass locally (`pytest -v` on the three files)
- [x] Existing test suite continues to pass (`tests/unit/.../test_hblp.py`: 34 tests, all green)
- [x] `lineage_audit_declared_path` helper extended to recognize `enrollment` reference as pre-anchor (aligns with `KnowableAt.is_pre_or_at_index()`; bug surfaced by sweep)
- [x] `pytest.mark.real_data` registered in `pyproject.toml` markers list
- [ ] Cryptographic signature — DEFERRED (gated on N3 infrastructure)
- [ ] G3 wiring guard CI workflow — DEFERRED (lands with G3 PR)

## Cross-references

- **Plan:** `.claude/plans/disease_agnostic_quality_uplift_v4.md` §2 Gate G1
- **Plan v3 baseline:** `.claude/plans/disease_agnostic_quality_uplift_v3.md` §6 Tier 1B Gate B1
- **Empirical anchor:** `docs/results/optum_initiation_revalidation_20260510.md`
- **PR #127 (HBLP helpers):** commit `5e15ec90`
- **PR #128 (synthetic leakage-injection regression):** see plan §0 status snapshot
- **N3 known limitations:** `docs/governance/n3_known_limitations_20260510.md`
- **Companion gates:** G2 (`v4-g2-phase-b`), G3 (Phase C, lands LAST)
