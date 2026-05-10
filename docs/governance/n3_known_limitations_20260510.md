# N3 Methodology Sign-off Guard — Known Limitations (2026-05-10)

This document enumerates the four infrastructure-dependent partials surfaced
by the v4 N3 phase A pass-2 codex review that are deferred from PR #131
because they require security/infra decisions outside the engineering scope
of the v4 Phase A landing. Each item lists: (1) the limitation, (2) the
attacker capability it leaves open, (3) the mitigation status today, and
(4) the backlog reference.

The four in-scope deltas (NEW MED future-date rejection, M1 PARTIAL gh-skip
CRITICAL surfacing, NEW HIGH sigstore verify-blob payload arg, and this
document) were landed in this PR. The four items below are tracked under a
single backlog entry `v4-N3-signature-infra` and require security/infra team
ownership before they can be closed.

## 1. Registry-pinned GPG keyring on CI runners (H1 PARTIAL)

**Limitation.** The `--require-signature` path of `check_methodology_signoff.py`
invokes `gpg --verify` against the system keyring (or the `--keyring-dir`
override) but the CI runner does not yet provision a registry-pinned
keyring containing the trusted public keys of the active reviewers listed
in `docs/governance/methodology_reviewer_registry.md`. As a result:

- A reviewer can sign with any key whose fingerprint is not bound to their
  identity, and gpg will report `Good signature` so long as that key is
  importable.
- An attacker with access to the runner's keyring can add an arbitrary key
  and produce verifying signatures.

**Attacker capability.** Substitute a malicious signature signed by an
attacker-controlled key for a legitimate reviewer's signature; the validator
will pass `signature_verifies` if the malicious key is in the runner's
keyring.

**Mitigation today.** None at the validator level. The
`methodology_reviewer_registry.md` row for each reviewer is intended to
include the gpg fingerprint as a future column; the validator can then
require a fingerprint match. Until that column exists AND the runner ships
with the pinned keyring (e.g. `gpg --import` from a SHA-pinned manifest
during job setup), `--require-signature` is best-effort.

**Mitigation status.** DEFERRED.

**Backlog reference.** `v4-N3-signature-infra` (registry-pinned keyring).

---

## 2. `gh` CLI authentication for PR/review provenance (H2 PARTIAL, M1 PARTIAL)

**Limitation.** The selection-rule check shells out to `gh pr list
--author <handle>` and `gh pr list --reviewer <handle>` to confirm a
reviewer has not authored or reviewed PRs touching the subject files in
the named period. On runners without `gh` on PATH OR without an
authenticated `GH_TOKEN` with `repo:read`, those queries return None and
the validator emits a CRITICAL warning (iter-3 M1 fix) — but the check
still PASSES on the canonical git-log signal alone.

**Attacker capability.** A reviewer who has authored or reviewed a PR
touching the subject file via the GitHub web UI (no commit attributable to
them via `git log`) can self-declare clean and the validator cannot detect
it. They would still be caught by:

- Their own CoI declaration if they list the PR honestly (the
  `coi-self-declared` signal IS authoritative).
- Any reviewer who manually scans the PR history for the named period.

**Mitigation today.** iter-3 M1 fix surfaces a CRITICAL warning AND sets
`CheckResult.provenance_check_skipped=True` so the CI workflow can scan
validator output for the marker (currently posts a workflow `::warning::`
annotation; can be flipped to `exit 1` once a `GH_TOKEN` is provisioned).

**Mitigation status.** DEFERRED for fail-closed enforcement.

**Backlog reference.** `v4-N3-signature-infra` (gh enforcement).

---

## 3. Validator pinned to protected base ref via reusable workflow (H3 PARTIAL)

**Limitation.** The current `methodology_signoff_guard.yml` already fetches
the validator from the PR's base SHA via `git show
<base_sha>:scripts/check_methodology_signoff.py` (the existing H3 fix), so
a PR that weakens the validator and the workflow simultaneously cannot use
its own weakened copy to validate itself. HOWEVER, a PR that lands the
weakening on a protected base ref (e.g. by getting an upstream merge to
`main`) before opening the sign-off PR can still bypass the protection.

**Attacker capability.** A two-step attack:
1. Land a PR that weakens `check_methodology_signoff.py` on `main`.
2. Open a separate PR with the malicious sign-off; the now-weakened base-ref
   validator validates it.

**Mitigation today.** The validator is CODEOWNERS-gated (per the H3
SECURITY ADVISORY in the workflow YAML) so step 1 requires security-team
review. The fix is to move the validator to a separate, protected
repository (e.g. `org/governance-actions`) and reference it via
`uses: org/governance-actions/methodology-signoff@<sha>` pinned by SHA on
this repo's protected refs — a reusable-workflow pattern. This requires
provisioning a separate repo, configuring its protected branches, and
agreeing on the cross-repo trust boundary.

**Mitigation status.** DEFERRED pending decision on the
`org/governance-actions` repository.

**Backlog reference.** `v4-N3-signature-infra` (protected-base-ref reusable
workflow).

---

## 4. CoI declaration signature verification (H4 PARTIAL)

**Limitation.** The CoI declaration referenced from the sign-off
artifact's `## Conflict-of-interest declaration` section is checked for:
(a) a parseable `<handle>_<YYYYMMDD>.md` filename, (b) a SHA + path that
resolve in git, and (c) the SHA being the first-add commit for the path
(per the existing H4 fix). It is NOT cryptographically signed. A reviewer
can edit a CoI document, force-push a rewritten history that "first-added"
the modified version, and the validator cannot distinguish.

**Attacker capability.** The reviewer can modify their CoI declaration
content (e.g. remove a previously-disclosed PR from the declared-PRs JSON
array) and re-write the SHA, evading the `coi-self-declared` signal.

**Mitigation today.** None at the validator level. Branch protection on
`main` (no force-push) prevents rewriting committed history but does not
prevent a reviewer from preparing a fresh CoI on a feature branch with
selectively-omitted content; that branch's SHA will still be first-add.
The fix is to require the CoI file to itself carry a PGP signature
verifiable against the registry-pinned keyring (item 1) — at which point
the `coi-self-declared` JSON array is bound to a specific reviewer
identity at a specific commit time.

**Mitigation status.** DEFERRED pending registry-pinned keyring (item 1).

**Backlog reference.** `v4-N3-signature-infra` (CoI signature verify).

---

## Summary — what's mitigated today vs deferred

| Item | Mitigation today | Mitigation status |
| --- | --- | --- |
| Registry-pinned keyring (H1) | `--require-signature` invokes gpg | DEFERRED |
| gh PR/review enforcement (H2 / M1) | CRITICAL warn + skip flag | DEFERRED for fail-closed |
| Protected-base-ref reusable workflow (H3) | Base-SHA pinned + CODEOWNERS | DEFERRED for separate repo |
| CoI signature verify (H4) | First-add SHA + filename match | DEFERRED pending H1 |

All four items are tracked under a single backlog entry
`v4-N3-signature-infra` owned by the security/infra team.
