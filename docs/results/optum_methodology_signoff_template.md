# Optum Methodology Sign-off — TEMPLATE (NOT a real sign-off)

> **WARNING:** This file is a *template*. The filename ends in `_template`
> rather than a date. The CI guard
> (`.github/workflows/methodology_signoff_guard.yml`) does NOT validate this
> file. Real sign-offs are published as
> `docs/results/optum_methodology_signoff_<YYYYMMDD>.md` and validated on PR
> landing.

**Decision:** APPROVE / REJECT (delete the inappropriate option in the real
sign-off).

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
- **CoI declaration commit SHA:** `<sha>` (must be the SHA at which the
  declaration was *first* committed; CI tooling cross-references this).

### Selection-rule evidence (`scripts/convert_optum_rwd.py`)

Required `git log` output, named period `2026-04-15 → 2026-05-10`, must be
empty:

```
$ git log --author=<email> \
        --since=2026-04-15 --until=2026-05-10 \
        -- scripts/convert_optum_rwd.py
<paste empty output here — i.e. nothing between the fences IS the evidence>
```

Required `gh pr list` output (PRs authored *and* PRs reviewed) for the same
period and the same subject file. Filter to PRs that touched
`scripts/convert_optum_rwd.py`:

```
[]
```

If either output is non-empty, the reviewer is ineligible — see
`docs/governance/methodology_reviewer_registry.md` §Selection rule. Submit a
rejection rather than an approval.

## Methodology decision

### Approved parameters

- `ENROLLMENT_PRE_DAYS = 180`
- `ENROLLMENT_POST_DAYS = 90`
- `WASHOUT_DAYS = 30` (unchanged)
- Other constants in `scripts/convert_optum_rwd.py` left at production
  defaults.

### Rationale (free-form)

`<reviewer's clinical / methodological argument for why the relaxed window
preserves enrollment-feasibility validity. Must explicitly address: (a) why
the post-anchor 90-day post-window does not introduce immortal-time bias; (b)
how the 180-day pre-window captures sufficient comorbidity history given CSU
diagnostic latency; (c) plan to validate the n=1697 GENUINE outcome on an
unblinded held-out cohort to mitigate the data-snooping concern flagged by
codex-rescue CLAIM-D.>`

### Held-out validation plan

`<concrete next steps — e.g. "split the 2027-Q1 PROD ingest into a
60/20/20 with the 20% holdout reserved for post-decision regression
testing of the n=1697 outcome".>`

## Cryptographic signature

The sign-off MUST carry one of:

1. **PGP** signature over the entire body of this document up to (but not
   including) this `## Cryptographic signature` heading. Paste the armored
   signature inline:

   ```
   -----BEGIN PGP SIGNATURE-----

   <signature blob>

   -----END PGP SIGNATURE-----
   ```

2. **Sigstore** bundle (`.sigstore` JSON) — paste the bundle inline:

   ```json
   { "...": "..." }
   ```

The CI guard accepts either format. If the signature block is missing,
malformed, or fails verification, the guard fails the PR and the sign-off is
not landed.

## Acceptance checklist (CI-enforced)

- [ ] Reviewer's GitHub handle appears in the registry as an `active` row.
- [ ] CoI declaration filename, commit SHA, and signature all reference the
      same reviewer.
- [ ] Selection-rule `git log` evidence is empty (zero commits in named
      period for each subject file).
- [ ] Selection-rule `gh pr list` evidence is empty for each subject file.
- [ ] Cryptographic signature verifies against the reviewer's published key.
- [ ] All required sections (`## Reviewer`, `## Conflict-of-interest
      declaration`, `## Methodology decision`, `## Cryptographic signature`)
      are present.

## Cross-references

- Plan: `.claude/plans/adaptive_disease_agnostic_quality_uplift.md` §Gate N3.
- Empirical anchor:
  `docs/results/optum_initiation_revalidation_20260510.md`.
- Codex-rescue concern: CLAIM-D in the v4 plan review pass-2.
- Future regression test:
  `tests/regression/test_optum_n1697_genuine.py` will pin the n=1697 outcome
  with `data_snooped: true` in its docstring until the held-out validation
  plan completes.

## Future regression test docstring template

The future test author should use the following docstring scaffold to comply
with the v4 plan §Gate N3 data-snooping mitigation:

```python
def test_optum_n1697_perm_pvalue_genuine() -> None:
    """Pin the Optum n=1697 GENUINE permutation outcome (relaxed window).

    Subject:
        Optum CSU initiation cohort under the "research" enrollment regime
        (PRE_DAYS=180, POST_DAYS=90). Asserts perm p<=0.05.

    data_snooped: true
        Per Gate N3 sign-off (this file's real, dated counterpart), the
        n=1697 outcome is data-snooped: the relaxed-window parameters were
        chosen *because* they crossed permutation p<0.05 on a single Optum
        ingest. Pinning the outcome encodes the snoop into the regression
        test, which is acceptable ONLY because the methodology sign-off has
        scheduled a held-out validation pass that will re-test the outcome
        on a cohort whose split was decided before any AUC was observed.

    held_out_validation:
        See `## Held-out validation plan` in the sign-off doc. Until that
        pass lands and confirms p<=0.05 on the held-out cohort, this test
        MUST stay marked `data_snooped: true` in its docstring and any
        broken-test failure mode MUST be triaged manually rather than
        auto-pinned.
    """
    ...
```
