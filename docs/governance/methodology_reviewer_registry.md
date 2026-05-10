# Methodology Reviewer Registry

**Version:** 1.0
**Last updated:** 2026-05-10
**Owner:** E2I Causal Analytics governance
**Source of truth for:** Tier 1A methodology sign-off (Gate N3 of v4 plan
`adaptive_disease_agnostic_quality_uplift.md`).

## Purpose

This registry is the canonical, version-controlled list of reviewers eligible
to approve methodology decisions for the E2I Causal Analytics platform. It is
the analogue of GitHub's `CODEOWNERS` file but specialised for **methodology**
sign-offs (cohort-window selection, target definition, leakage policy, etc.)
rather than file ownership.

The most-load-bearing use-case is **Tier 1A enrollment-window selection** for
the Optum CSU cohort: the n=1697 GENUINE outcome reported in
`docs/results/optum_initiation_revalidation_20260510.md` was obtained at
`(PRE_DAYS=180, POST_DAYS=90)` — values chosen *after* observing that they
crossed `permutation p<0.05`. Any reviewer who selects, signs off, or is the
PR author of those parameters is not eligible to perform the subsequent
methodology sign-off, because the parameters were data-snooped.

See `docs/governance/coi_declaration_template.md` for the conflict-of-interest
declaration each reviewer must sign before performing a sign-off.

## Selection rule (enforced by `scripts/check_methodology_signoff.py`)

For any sign-off whose subject file is `scripts/convert_optum_rwd.py`:

1. Reviewer GitHub handle MUST appear in the `## Active reviewers` table
   below.
2. Reviewer's published CoI declaration MUST be referenced in the sign-off
   document by commit SHA.
3. CoI declaration MUST report ZERO commits or PRs touching
   `scripts/convert_optum_rwd.py` in the calendar window
   **2026-04-15 → 2026-05-10** (the period during which the empirical anchor
   for the n=1697 outcome was generated).

The CI workflow `.github/workflows/methodology_signoff_guard.yml` runs the
script on every PR that adds or modifies a file matching
`docs/results/optum_methodology_signoff_*.md` or
`docs/results/optum_methodology_rejection_*.md`.

## Row schema

| Field | Type | Notes |
|---|---|---|
| `name` | string | Full legal name as on contract / employment record. |
| `email` | string | Single canonical email; used for `git log --author=<email>` evidence. |
| `github_handle` | string | Without leading `@`. Cross-referenced from sign-off docs. |
| `role` | string | E.g. `principal_biostatistician`, `external_clinical_advisor`. |
| `date_added` | ISO-8601 date | Date the row was added to this registry. |
| `areas_of_expertise` | string | Free-text, comma-separated. Used by humans to choose a reviewer for a given subject. |
| `status` | string | One of `active`, `inactive`, `recused`. Only `active` rows are eligible. |

## Active reviewers

<!-- Append rows below; do not edit historical rows. To deactivate a reviewer
     set their `status` to `inactive` and add a new row with the new state. -->

| name | email | github_handle | role | date_added | areas_of_expertise | status |
|---|---|---|---|---|---|---|
| _PLACEHOLDER_ | placeholder@example.com | placeholder | placeholder_role | 2026-05-10 | methodology, biostatistics | inactive |

> The placeholder row is `inactive` so the registry parses correctly but does
> not satisfy any selection rule. Replace it with one or more `active` rows
> before requesting a sign-off.

## Inactive / historical reviewers

(Empty — no historical rows yet.)

## Adding a reviewer

1. Open a PR adding a new row to the `## Active reviewers` table with all
   required columns populated.
2. The new reviewer publishes their CoI declaration at
   `docs/governance/coi_declarations/<github_handle>_<YYYYMMDD>.md` using the
   template at `docs/governance/coi_declaration_template.md`.
3. Both files MUST land in the same PR.
4. PR is reviewed by an existing `active` reviewer (or by repo owner for the
   bootstrap case).

## Removing / recusing a reviewer

Set `status` to `inactive` (left the team) or `recused` (still on team but
self-recused from a particular sign-off domain). Do not delete the row.
