# Optum Methodology Rejection — TEMPLATE (NOT a real rejection)

> **WARNING:** This file is a *template*. The filename ends in `_template`
> rather than a date. The CI guard
> (`.github/workflows/methodology_signoff_guard.yml`) does NOT validate this
> file. Real rejections are published as
> `docs/results/optum_methodology_rejection_<YYYYMMDD>.md` and validated on
> PR landing.

**Decision:** REJECT.

**Subject:** Tier 1A methodology decision for Optum CSU initiation cohort —
relaxed enrollment window `(PRE_DAYS=180, POST_DAYS=90)` ("research" regime).

**Empirical anchor:**
`docs/results/optum_initiation_revalidation_20260510.md` — n=1697 GENUINE
outcome at permutation `p=0.02`. The default-window n=1294 result halts at
MARGINAL.

**Date:** `YYYY-MM-DD`

## Reviewer

- **Name:** `<full legal name>`
- **GitHub handle:** `@<github_handle>`
- **Registry row:** `docs/governance/methodology_reviewer_registry.md` —
  `<github_handle>` row at commit `<sha>` (must be `active`).

## Conflict-of-interest declaration

- **CoI document:**
  `docs/governance/coi_declarations/<github_handle>_<YYYYMMDD>.md`
- **CoI declaration commit SHA:** `<sha>`

### Selection-rule evidence (`scripts/convert_optum_rwd.py`)

Required `git log` output for the named period `2026-04-15 → 2026-05-10`:

```
$ git log --author=<email> \
        --since=2026-04-15 --until=2026-05-10 \
        -- scripts/convert_optum_rwd.py
<paste empty output here — empty == eligible reviewer>
```

Required `gh pr list` output (PRs authored *and* PRs reviewed) filtered to
PRs touching `scripts/convert_optum_rwd.py`:

```
[]
```

## Reasons for rejection

Provide explicit, falsifiable rejections grouped by category:

### Methodological

`<e.g. "The 90-day post-anchor window is incompatible with the 180-day
biologic-discontinuation gap defined upstream — patients can be classified
as 'persistent' before they are observed to discontinue, biasing the
target.">`

### Statistical

`<e.g. "The n=1697 result was obtained at permutation p=0.02 on a single
ingest. Codex-rescue CLAIM-D notes the parameters were chosen because they
crossed p<0.05; the reviewer concurs that without a held-out validation
the result is not robustly genuine.">`

### Clinical

`<e.g. "180-day pre-window is insufficient to capture comorbidity onset for
CSU patients with diagnostic latency >180 days; risks misclassifying
prevalent comorbidities as new-onset.">`

### Operational

`<e.g. "The relaxed window weakens enrollment-feasibility for downstream
real-world data partners; a strict-window re-validation plan should be
delivered before promotion.">`

## Recommended remediation

Concrete next steps the methodology team should take before a future
sign-off attempt:

1. `<...>`
2. `<...>`

## Cryptographic signature

Same format as approval — see
`docs/results/optum_methodology_signoff_template.md` §Cryptographic
signature.

```
-----BEGIN PGP SIGNATURE-----

<signature blob>

-----END PGP SIGNATURE-----
```

## Acceptance checklist (CI-enforced — same as approval)

- [ ] Reviewer's GitHub handle appears in the registry as an `active` row.
- [ ] CoI declaration filename, commit SHA, and signature all reference the
      same reviewer.
- [ ] Selection-rule `git log` evidence is empty (zero commits in named
      period for each subject file).
- [ ] Selection-rule `gh pr list` evidence is empty for each subject file.
- [ ] Cryptographic signature verifies against the reviewer's published key.
- [ ] All required sections (`## Reviewer`, `## Conflict-of-interest
      declaration`, `## Reasons for rejection`, `## Cryptographic
      signature`) are present.

## Cross-references

- Plan: `.claude/plans/adaptive_disease_agnostic_quality_uplift.md` §Gate N3.
- Empirical anchor:
  `docs/results/optum_initiation_revalidation_20260510.md`.
- Codex-rescue concern: CLAIM-D in the v4 plan review pass-2.
