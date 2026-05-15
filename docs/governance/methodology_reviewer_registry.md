# Methodology Reviewer Registry

**Version:** 1.1 (issue #226: added `fingerprint` column 2026-05-14)
**Last updated:** 2026-05-14
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
| `fingerprint` | string | **Issue #226 H1:** GPG public-key fingerprint, 40-char hex (no spaces, no `0x` prefix). The validator parses this cell, normalizes it, and at signature-verification time confirms the signing key matches a registered fingerprint. Use the placeholder `<TBD — populated by operator>` for rows pending operator key-collection (the validator treats placeholders as "not pinned yet" → WARN in default mode, FAIL under `STRICT_GPG=1`). |

**Fingerprint format & population workflow (issue #226 H1).** The fingerprint is derived from the reviewer's public key, e.g.

```
$ gpg --list-keys --with-fingerprint --keyid-format=long etn3724@gmail.com
pub   rsa4096/AAAAAAAAAAAAAAAA 2026-05-14 [SC]
      Key fingerprint = ABCD EF01 2345 6789 ABCD  EF01 2345 6789 ABCD EF01
```

Strip the spaces and paste the 40 hex characters into the `fingerprint` cell. The
operator handoff doc at `docs/governance/operator_gpg_keyring_setup.md` walks
through pubkey collection, the `GPG_REVIEWER_KEYS_ARMOR_BASE64` secret, and
fingerprint pinning end-to-end.

## Active reviewers

<!-- Append rows below; do not edit historical rows. To deactivate a reviewer
     set their `status` to `inactive` and add a new row with the new state. -->

| name | email | github_handle | role | date_added | areas_of_expertise | status | fingerprint |
|---|---|---|---|---|---|---|---|
| _PLACEHOLDER_ | placeholder@example.com | placeholder | placeholder_role | 2026-05-10 | methodology, biostatistics | inactive | `<TBD — populated by operator>` |
| E. Nunez | etn3724@gmail.com | enunezvn | engineering_owner | 2026-05-11 | methodology, causal-inference, biostatistics, mlops | active | `C4AFB94B4FBB05B40E894CBABDE9BB957AD7B060` |
| E. Nunez (GitHub no-reply) | 64854959+enunezvn@users.noreply.github.com | enunezvn | engineering_owner | 2026-05-11 | methodology, causal-inference, biostatistics, mlops | active | `C4AFB94B4FBB05B40E894CBABDE9BB957AD7B060` |

> The placeholder row remains `inactive` for back-compat with the registry's
> parse-test fixtures. Both `enunezvn` rows are the load-bearing `active`
> reviewers of record for the v4 Phase B/C G1+G2+G3 engineering signoffs —
> two rows because the GitHub rebase-merge writes the committer field as
> the `<user-id>+<handle>@users.noreply.github.com` no-reply address even
> when the author field stays the human-readable address. The G3 wiring
> guard reads committer (not author) email, so both must appear here for
> signoff-introducing commits to satisfy the email-parity check.
>
> **N3 INTERIM status:** This row attests engineering completeness only.
> The Plan v4 §N3 cryptographic-signature requirement (PGP / sigstore +
> CoI declaration) is gated on backlog item `v4-N3-signature-infra` and
> remains deferred per `docs/governance/n3_known_limitations_20260510.md`.
> Until that infra lands, this row satisfies the G3 wiring guard's
> `--require-signature-registry-match` precondition for the G1/G2
> engineering signoffs by establishing committer-email parity; it does
> NOT promote any gate's lifecycle state from `ADVISORY` to `ENFORCED`.

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
