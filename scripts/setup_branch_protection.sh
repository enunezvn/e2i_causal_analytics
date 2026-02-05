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

REPO="enunez/e2i_causal_analytics"
BRANCH="main"

echo "Configuring branch protection for ${REPO}:${BRANCH}..."

gh api \
  --method PUT \
  "repos/${REPO}/branches/${BRANCH}/protection" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["ci-success"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "require_code_owner_reviews": true,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "Branch protection configured successfully."
echo ""
echo "Rules applied to ${BRANCH}:"
echo "  - Require 1 PR approval"
echo "  - Require CODEOWNERS review"
echo "  - Require status check: ci-success"
echo "  - Dismiss stale reviews on new pushes"
echo "  - Block force push"
echo "  - Block branch deletion"
