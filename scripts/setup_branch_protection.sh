#!/usr/bin/env bash
# =============================================================================
# Configure GitHub branch protection for main branch
# =============================================================================
# Requirements:
#   - gh CLI authenticated (gh auth login)
#   - Admin access to the repository
#
# Usage:
#   bash scripts/setup_branch_protection.sh
# =============================================================================

set -euo pipefail

REPO="enunezvn/e2i_causal_analytics"
BRANCH="main"

# =============================================================================
# WHY THIS CONFIG (matches the APPLIED live state on main):
#
#  - required_status_checks.contexts = ["Backend CI Success", "Tier 1-5 agent
#    harness"]. ONLY checks that report on EVERY PR may be required here. Both
#    of these checks are PATH-FILTERED in their workflows (they only run when
#    matching paths change). A required check that does not report on a given
#    PR leaves that PR "Expected - Waiting for status to be reported" forever,
#    which DEADLOCKS docs-only / scripts-only PRs. We accept the two checks
#    above as the agreed must-pass set, and rely on admin override (below) to
#    unblock PRs the path filter skips.
#
#  - enforce_admins = false. This is REQUIRED so an admin can override-merge a
#    PR that a path-filtered required check never reported on (the footgun
#    above). Do not flip this to true without first removing path-filtered
#    required checks.
#
#  - required_approving_review_count = 0 and require_code_owner_reviews = false.
#    This is a solo-dev repo; a self-approval gate would just block every PR.
#    dismiss_stale_reviews is likewise false.
#
#  - required_linear_history = false. The repo policy is ALWAYS preserve history
#    via --merge merge-commits and NEVER squash. Linear history would forbid
#    merge commits, so it MUST stay false to keep the --merge policy legal.
#
#  - allow_force_pushes = false, allow_deletions = false. Protect main from
#    history rewrites and accidental deletion.
#
#  - strict = false. Do not force a branch to be up to date with main before
#    merging (avoids a re-run treadmill on a solo-dev repo).
# =============================================================================

echo "Configuring branch protection for ${REPO}:${BRANCH}..."

gh api \
  --method PUT \
  "repos/${REPO}/branches/${BRANCH}/protection" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["Backend CI Success", "Tier 1-5 agent harness"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "require_code_owner_reviews": false,
    "dismiss_stale_reviews": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "Branch protection configured successfully."
echo ""
echo "Rules applied to ${BRANCH}:"
echo "  - Require status checks: Backend CI Success, Tier 1-5 agent harness (strict=false)"
echo "  - No approvals required (solo-dev repo): 0 approvals, no CODEOWNERS gate"
echo "  - Do not dismiss stale reviews on new pushes"
echo "  - enforce_admins=false (admin can override-merge a PR a path-filtered check skipped)"
echo "  - required_linear_history=false (keeps the --merge merge-commit / never-squash policy legal)"
echo "  - Block force push"
echo "  - Block branch deletion"
