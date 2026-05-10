# Conflict-of-Interest Declaration Template

**Filename convention:**
`docs/governance/coi_declarations/<github_handle>_<YYYYMMDD>.md`

**Reviewer:** `<full name>`
**GitHub handle:** `@<github_handle>` (cross-referenced to
`docs/governance/methodology_reviewer_registry.md`)
**Email:** `<email>`
**Declaration date:** `YYYY-MM-DD`
**Sign-off subject:** `<short description>` — e.g. `Optum CSU enrollment
window (PRE=180/POST=90) Tier 1A methodology decision`.
**Subject files:** newline-separated list — e.g. `scripts/convert_optum_rwd.py`.

## Purpose

This declaration enumerates every commit, pull request, and code review the
reviewer has authored or performed against the **subject files** during the
**named period**. The declaration is the evidentiary basis for the
selection-rule check enforced by
`scripts/check_methodology_signoff.py` and
`.github/workflows/methodology_signoff_guard.yml`.

A reviewer is INELIGIBLE for a methodology sign-off if their declaration
shows ANY commit or merged PR that touched the subject files inside the
named period.

## Named period

`<YYYY-MM-DD>` (inclusive) → `<YYYY-MM-DD>` (inclusive)

For the Optum n=1697 sign-off the named period is **2026-04-15 →
2026-05-10**, the calendar window during which the empirical anchor that
produced the n=1697 GENUINE outcome was generated.

## Evidence

### `git log` against subject files in named period

Run for each subject file:

```bash
git log --author=<email> \
        --since=<period_start> --until=<period_end> \
        -- <subject_file>
```

Paste the literal output below. If no commits are found, paste the empty
output verbatim (i.e. nothing between the fences) — the empty output IS the
evidence.

```
<paste git log output here>
```

### `gh pr list` (PRs authored / reviewed) in named period

```bash
# PRs authored
gh pr list --author <github_handle> --state all --search \
   "created:<period_start>..<period_end>" \
   --json number,title,files

# PRs reviewed
gh pr list --reviewer <github_handle> --state all --search \
   "updated:<period_start>..<period_end>" \
   --json number,title,files
```

Paste the JSON output below. Filter (manually or programmatically) to only
PRs whose `files` list intersects the subject files. If no qualifying PRs
exist, paste `[]`.

```
<paste gh pr list output here>
```

### Other potentially-conflicting work

List any non-code involvement that might constitute a conflict:

- internal slack threads where the reviewer recommended specific parameter
  values for the subject files
- mentor / mentee relationships with the original authors of the subject
  files
- co-authorship on internal memos that reference the subject files
- consulting engagements

If none, write `None.` here.

## Declaration

I, `<full name>`, declare under penalty of professional misconduct that the
above evidence is complete and accurate to the best of my knowledge as of
the declaration date. I commit to amend this declaration before performing
any subsequent sign-off if my involvement with the subject files changes.

**Signature:** `<PGP-signed or sigstore-signed payload>`

> The signature MUST cover the entire body of this document above the
> `## Declaration` heading. See
> `docs/results/optum_methodology_signoff_template.md` for the cryptographic
> signature format.

## Cross-references

- Registry row: `docs/governance/methodology_reviewer_registry.md` —
  `<github_handle>` row.
- Sign-off document this CoI supports:
  `docs/results/optum_methodology_signoff_<YYYYMMDD>.md` (or
  `_rejection_<YYYYMMDD>.md`).
