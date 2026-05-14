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

---

## 2026-05-14 update — issue #192 closure

PR closing issue #192 (the `v4-N3-signature-infra` backlog item) split
the four PARTIAL findings into:

* **CODE-DOABLE** (addressed in this PR): H2 / M1 (gh enforcement —
  RESOLVED) and H3 (validator-from-protected-ref via reusable workflow
  split — **PARTIALLY RESOLVED**, see codex pass-1 HIGH-1 honesty
  correction below).
* **OPERATOR-REQUIRED** (risk-accepted in this PR; tracked as a follow-up
  issue): H1 (registry-pinned GPG keyring on CI runners) and H4 (CoI body
  signature verification, blocked on H1's keyring infra).

**H3 honesty correction (codex pass-1 HIGH-1).** Initial framing claimed
H3 was fully RESOLVED by the reusable-workflow split. Codex review
correctly pointed out that GitHub same-repo path-pinned reusable
workflow calls (`uses: ./.github/workflows/...`) load the called
workflow YAML *from the same commit as the caller*. So a malicious PR
can edit BOTH workflow YAMLs in the same commit, and its own modified
reusable-workflow can swap out / weaken the
`actions/checkout@v4 ref: main` step that pins the validator script.

The reusable-workflow split is therefore a NARROWING of the trusted
computing base from {validator script + caller YAML, all from PR head}
to {validator script (from main) + both workflow YAMLs (from PR head)}.
The validator script being pinned to main IS load-bearing (a PR
weakening only the script no longer affects the run); the workflow
YAML pinning is NOT load-bearing without one of:

* (a) Cross-repo SHA-pinned invocation
  (`uses: org/governance-actions/...@<sha>`) — requires separate-repo
  decision, deferred.
* (b) `pull_request_target` workflow from main with checkout-by-
  immutable-head-SHA — has its own threat model, deferred.

Until (a) or (b), the operator complement (branch protection + CODEOWNERS
gate on both workflow YAMLs AND the validator script) remains REQUIRED
— same requirement as pre-#192 but with narrower attack surface.

Per-item status is updated in the table at the end of this document AND
inline in each section header below.

---

## 1. Registry-pinned GPG keyring on CI runners (H1 PARTIAL → ACCEPTED-RISK 2026-05-14)

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

**Mitigation status (2026-05-14).** ACCEPTED-RISK pending operator action.
Issue #192 closure landed the code-doable mitigations (H2/M1 + H3) but
GPG keyring provisioning on CI runners requires per-runner state outside
declarative CI scope. Tracked as a follow-up sub-issue (linked from
#192) for the infra team.

**Risk owner.** OWNER: TBD — infra/security team (the follow-up
sub-issue spawned from #192's merge is the forcing function for owner
assignment; the sub-issue's first action is to assign a named GitHub
user/team and a target review date). Codex pass-1 MED-3 (2026-05-14)
correctly flagged that placeholder ownership is weak for a security
risk-acceptance; the mitigation chosen here is to make sub-issue
creation a hard precondition for #192 closure (PR comment on #192 will
link the sub-issue URL) AND to require the sub-issue to name an owner
within 14 days of creation. If no owner is named within 14 days, the
sub-issue is escalated to the security team's incident channel.

**Revisit trigger.** When the standard CI runner image standardizes on a
GPG keyring strategy (Actions cache + `gpg --homedir` binding OR pre-baked
runner image), reopen this item and wire the validator's
`--keyring-dir` flag to that location.

**Backlog reference.** `v4-N3-signature-infra` (registry-pinned keyring) →
follow-up sub-issue spawned by issue #192 closure.

---

## 2. `gh` CLI authentication for PR/review provenance (H2 PARTIAL, M1 PARTIAL → RESOLVED 2026-05-14)

**Limitation (historical).** The selection-rule check shells out to
`gh pr list --author <handle>` and `gh pr list --reviewer <handle>` to
confirm a reviewer has not authored or reviewed PRs touching the subject
files in the named period. On runners without `gh` on PATH OR without an
authenticated `GH_TOKEN` with `repo:read`, those queries return None and
the validator emits a CRITICAL warning (iter-3 M1 fix) — but the check
still PASSED on the canonical git-log signal alone.

**Attacker capability (historical).** A reviewer who has authored or
reviewed a PR touching the subject file via the GitHub web UI (no commit
attributable to them via `git log`) can self-declare clean and the
validator cannot detect it. They would still be caught by:

- Their own CoI declaration if they list the PR honestly (the
  `coi-self-declared` signal IS authoritative).
- Any reviewer who manually scans the PR history for the named period.

**Mitigation today (2026-05-14).** RESOLVED via two changes in the issue
#192 PR:

1. **Validator** (`scripts/check_methodology_signoff.py`): added a
   `--strict-gh` CLI flag (and matching `STRICT_GH=1` env var) that
   promotes `provenance_check_skipped=True` from a logged warning to a
   hard exit (code 3). The exit-code contract is now:
   * 0 = all checks passed AND (strict-gh disabled OR no provenance skips).
   * 1 = generic validation failure (selection-rule violation, missing
     section, unverifiable signature, etc.).
   * 2 = script invocation error.
   * 3 = `--strict-gh` is set AND at least one check has
     `provenance_check_skipped=True`.

   Local devs running the script ad-hoc retain the warn-only back-compat
   path (no `--strict-gh`, no `STRICT_GH` env var → exit 0).

2. **Workflow** (`.github/workflows/methodology-signoff-validator.yml`,
   the new reusable workflow): provisions
   `GH_TOKEN: ${{ github.token }}` AND exports `STRICT_GH: '1'` on the
   validator step. The auto-provisioned `GITHUB_TOKEN` carries
   `pull-requests:read` + `contents:read` (set in the workflow's
   `permissions:` block — least privilege for the gh-CLI provenance
   queries). When the runner provisions a keyring in the future
   (item 1 above), the validator step will satisfy gh PR/review queries
   AND signature verification in a single CI run.

**Mitigation status.** RESOLVED 2026-05-14.

**Backlog reference.** `v4-N3-signature-infra` (gh enforcement) — closed
by issue #192.

---

## 3. Validator pinned to protected base ref via reusable workflow (H3 PARTIAL → PARTIALLY RESOLVED 2026-05-14)

**Limitation (historical).** The original `methodology_signoff_guard.yml`
fetched the validator from the PR's base SHA via `git show
<base_sha>:scripts/check_methodology_signoff.py` (the H3 fix from PR
#131), so a PR that weakens the validator and the workflow simultaneously
could not use its own weakened copy to validate itself. HOWEVER, a PR
that lands the weakening on a protected base ref (e.g. by getting an
upstream merge to `main`) before opening the sign-off PR could still
bypass the protection.

**Attacker capability (historical).** A two-step attack:
1. Land a PR that weakens `check_methodology_signoff.py` on `main`.
2. Open a separate PR with the malicious sign-off; the now-weakened base-ref
   validator validates it.

**Mitigation today (2026-05-14).** PARTIALLY RESOLVED via
reusable-workflow split. The N3 guard is now two workflow files:

* `.github/workflows/methodology_signoff_guard.yml` — the THIN CALLER.
  Identifies touched artifacts on the PR diff, then delegates to the
  reusable workflow via
  `uses: ./.github/workflows/methodology-signoff-validator.yml`. Passes
  `validator_ref: 'main'` to pin the validator-script source to the
  protected branch.

* `.github/workflows/methodology-signoff-validator.yml` — the REUSABLE
  WORKFLOW (`on: workflow_call:`). Performs TWO `actions/checkout@v4`
  invocations:
  1. PR head into `pr-checkout/` (the artifacts being validated).
  2. The protected ref (default `main`, sparse-checkout of
     `scripts/check_methodology_signoff.py` only) into `validator-source/`.

  Runs `python3 ${{ github.workspace }}/validator-source/scripts/check_methodology_signoff.py`
  against the PR-checkout artifacts. The validator code is sourced from
  the protected ref; the artifacts are sourced from the PR. This closes
  the "PR weakens validator and validates itself in same commit" attack
  class.

**Architecture decision (with codex pass-1 HIGH-1 honesty
correction).** Same-repo same-org reusable-workflow invocation
(`uses: ./...`) was chosen over cross-repo SHA-pinned invocation
(`uses: org/governance-actions/...@<sha>`) because:

* Same-repo invocation requires no separate-repo provisioning,
  CODEOWNERS migration, or cross-repo trust-boundary decision.
* Migration to cross-repo SHA-pinned invocation is a future-state
  optimization once the `org/governance-actions` repo question is
  answered. The reusable-workflow split is the prerequisite for that
  migration regardless.

INITIAL framing claimed the load-bearing defense lives INSIDE the
reusable workflow (the explicit `actions/checkout@v4 ref: main` for
the validator script), and that path-pinning is sufficient. **Codex
pass-1 HIGH-1 corrected this**: GitHub same-repo `uses: ./...` calls
load the called workflow YAML from the same commit as the caller, so a
malicious PR that edits BOTH workflow files in one commit can swap out
the `ref: main` step. The validator SCRIPT pin is still load-bearing
(a PR weakening only the script no longer matters); the workflow YAML
is NOT load-bearing without a real cross-repo SHA pin or
`pull_request_target` from main.

**Residual threats (post-#192).**

1. **Same-commit dual-edit**: A PR that edits the caller workflow YAML
   AND the reusable workflow YAML AND the validator script in the same
   commit can disable the `ref: main` checkout and run a weakened
   validator. Mitigated only by CODEOWNERS gating on all three files
   (operator config, documented in workflow header).
2. **Cross-PR weakening**: Land a weak validator on `main` first; then
   open the sign-off PR. The reusable workflow loads main's now-weak
   validator and validates the sign-off cleanly. Mitigated only by
   CODEOWNERS gating on `scripts/check_methodology_signoff.py`.
3. **Cross-repo migration deferred**: The full close requires
   `uses: org/governance-actions/.github/workflows/...@<sha>` with the
   reusable workflow + validator both in a separately-protected repo
   that THIS repo cannot edit. That decision is open.

The required operator complement is **branch protection on `main`
(no force-push, requires PR) + CODEOWNERS gate on
`scripts/check_methodology_signoff.py`,
`.github/workflows/methodology_signoff_guard.yml`, AND
`.github/workflows/methodology-signoff-validator.yml`** so any
weakening attempt requires security-team review. This is the SAME
operator requirement as pre-#192 (the workflow's H3 SECURITY ADVISORY
called for it then), but the attack surface is narrower because the
validator script is now sourced from main.

**Mitigation status.** PARTIALLY RESOLVED 2026-05-14 (validator script
pinned to main; workflow YAML pinning deferred to cross-repo migration).

**Backlog reference.** `v4-N3-signature-infra` (protected-base-ref reusable
workflow) — partially addressed by issue #192. Cross-repo SHA-pinned
migration tracked separately under the same backlog entry, blocked on
`org/governance-actions` repo decision.

---

## 4. CoI declaration signature verification (H4 PARTIAL → ACCEPTED-RISK 2026-05-14)

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

**Mitigation status (2026-05-14).** ACCEPTED-RISK pending operator
action. Same blocker as item 1: requires the registry-pinned keyring
infra. Closing item 1 (the follow-up sub-issue) unblocks this item, at
which point the validator can be extended to invoke `gpg --verify` on the
CoI body using the same keyring.

**Risk owner.** OWNER: TBD — same forcing-function as item 1 (the
follow-up sub-issue spawned from #192's merge is the assignment vehicle;
14-day owner-name deadline + escalation to security team's incident
channel applies symmetrically here).

**Revisit trigger.** When item 1 (registry-pinned keyring) closes —
verify CoI body signature using the same keyring; update the validator's
`check_coi_referenced` (or add a sibling `check_coi_signature_verifies`)
to invoke `gpg --verify` on the CoI markdown body.

**Backlog reference.** `v4-N3-signature-infra` (CoI signature verify) →
same follow-up sub-issue as item 1.

---

## Summary — what's mitigated today vs deferred

| Item | Mitigation today | Mitigation status |
| --- | --- | --- |
| Registry-pinned keyring (H1) | `--require-signature` invokes gpg | ACCEPTED-RISK 2026-05-14 (operator follow-up) |
| gh PR/review enforcement (H2 / M1) | `--strict-gh` + GH_TOKEN provisioned | RESOLVED 2026-05-14 |
| Protected-base-ref reusable workflow (H3) | Reusable workflow checks out `main` for validator script (workflow YAML pinning still PR-controlled) | PARTIALLY RESOLVED 2026-05-14 (validator script pinned; workflow YAML pinning deferred to cross-repo migration) |
| CoI signature verify (H4) | First-add SHA + filename match | ACCEPTED-RISK 2026-05-14 (blocked on H1) |

H2/M1 closed by issue #192 PR; H3 partially closed (validator script
sourced from main; full close requires cross-repo SHA-pinned migration).
H1 + H4 deferred to operator action via the follow-up sub-issue spawned
from that PR's merge. CODEOWNERS gating on both workflow YAMLs +
validator script remains REQUIRED operator complement until cross-repo
migration lands.
