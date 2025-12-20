---
description: Comprehensive PR code review - checks diff, patterns, runs validation, comments on PR
argument-hint: <pr-number or pr-url> [--approve|--request-changes]
---

# PR Code Review

**Input**: $ARGUMENTS

---

## Input Validation

Before proceeding, validate the input arguments.

```bash
# Check if argument is provided
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: PR number or URL is required"
  echo ""
  echo "Usage: /review-pr <pr-number-or-url> [--approve|--request-changes]"
  echo ""
  echo "Examples:"
  echo "  /review-pr 123"
  echo "  /review-pr https://github.com/owner/repo/pull/123"
  echo "  /review-pr 123 --approve"
  echo "  /review-pr 123 --request-changes"
  exit 1
fi

# Check for optional flags
APPROVE_FLAG=false
REQUEST_CHANGES_FLAG=false

if [[ "$ARGUMENTS" == *"--approve"* ]]; then
  APPROVE_FLAG=true
  CLEAN_ARGS=$(echo "$ARGUMENTS" | sed 's/--approve//g' | xargs)
  echo "ℹ️  Will approve if no critical/high issues found"
elif [[ "$ARGUMENTS" == *"--request-changes"* ]]; then
  REQUEST_CHANGES_FLAG=true
  CLEAN_ARGS=$(echo "$ARGUMENTS" | sed 's/--request-changes//g' | xargs)
  echo "ℹ️  Will request changes if issues found"
else
  CLEAN_ARGS="$ARGUMENTS"
fi

# Extract PR number from URL or use number directly
if [[ "$CLEAN_ARGS" =~ github\.com/.*/pull/([0-9]+) ]]; then
  PR_NUMBER="${BASH_REMATCH[1]}"
  echo "✓ Extracted PR #$PR_NUMBER from URL"
elif [[ "$CLEAN_ARGS" =~ ^[0-9]+$ ]]; then
  PR_NUMBER="$CLEAN_ARGS"
  echo "✓ Using PR #$PR_NUMBER"
else
  echo "❌ Error: Invalid PR number or URL"
  echo "Expected: number (123) or GitHub URL"
  echo "Received: $CLEAN_ARGS"
  exit 1
fi
```

---

## Your Mission

Perform a thorough, senior-engineer-level code review:

1. **Understand** what the PR is trying to accomplish
2. **Check** the code against project patterns and constraints
3. **Run** all validation (type-check, lint, tests, build)
4. **Identify** issues by severity
5. **Report** findings as PR comment AND local file

**Golden Rule**: Be constructive and actionable. Every issue should have a clear recommendation. Acknowledge good work too.

---

## Phase 1: Fetch PR Context

### Make sure you are on the correct branch (PR Branch!)

### 1.1 Parse Input

Input could be:
- PR number: `123`
- PR URL: `https://github.com/owner/repo/pull/123`
- Branch name: `feature-branch` (find associated PR)

### 1.2 Get PR Metadata

```bash
# Get PR details
gh pr view [NUMBER] --json number,title,body,author,headRefName,baseRefName,state,additions,deletions,changedFiles,files,reviews,comments

# Get the diff
gh pr diff [NUMBER]

# List changed files with stats
gh pr diff [NUMBER] --name-only
```

Extract:
- PR number, title, description
- Author
- Base and head branches
- Files changed with line counts
- Existing review comments

### 1.3 Checkout PR Branch

```bash
# Fetch and checkout the PR branch
gh pr checkout [NUMBER]
```

---

## Phase 2: Understand Context

### 2.1 Read Project Rules

Read and internalize:
- `CLAUDE.md` - Project conventions and constraints
- `.claude/reference/` - Any relevant reference docs
- Type definitions in `src/types/`

**Extract key constraints:**
- Type safety requirements
- Code style rules
- Testing requirements
- Architecture patterns

### 2.2 Understand PR Intent

From the PR title and description:
- What problem does this solve?
- What approach was taken?
- Are there any notes from the author?

### 2.3 Analyze Changed Files

For each changed file, determine:
- What type of file is it? (adapter, handler, util, test, config)
- What existing patterns should it follow?
- What's the scope of change? (new file, modification, deletion)

---

## Phase 3: Code Review

### 3.1 Read Each Changed File

For each file in the diff:

1. **Read the full file** (not just the diff) to understand context
2. **Read similar files** to understand expected patterns
3. **Check the specific changes** against patterns

### 3.2 Review Checklist

**For EVERY changed file, check:**

#### Correctness
- [ ] Does the code do what the PR claims?
- [ ] Are there logic errors?
- [ ] Are edge cases handled?
- [ ] Is error handling appropriate?

#### Type Safety
- [ ] Are all types explicit (no implicit `any`)?
- [ ] Are return types declared?
- [ ] Are interfaces used appropriately?
- [ ] Are type guards used where needed?

#### Pattern Compliance
- [ ] Does it follow existing patterns in the codebase?
- [ ] Is naming consistent with project conventions?
- [ ] Is file organization correct?
- [ ] Are imports from the right places?

#### Security
- [ ] Any user input without validation?
- [ ] Any secrets that could be exposed?
- [ ] Any injection vulnerabilities (SQL, command, etc.)?
- [ ] Any unsafe operations?

#### Performance
- [ ] Any obvious N+1 queries or loops?
- [ ] Any unnecessary async/await?
- [ ] Any memory leaks (unclosed resources, growing arrays)?
- [ ] Any blocking operations in hot paths?

#### Completeness
- [ ] Are there tests for new code?
- [ ] Is documentation updated if needed?
- [ ] Are all TODOs addressed?
- [ ] Is error handling complete?

#### Maintainability
- [ ] Is the code readable?
- [ ] Is it over-engineered?
- [ ] Is it under-engineered (missing necessary abstractions)?
- [ ] Are there magic numbers/strings that should be constants?

### 3.3 Categorize Issues

**🔴 Critical (Blocking)**
- Security vulnerabilities
- Data loss/corruption potential
- Breaking changes without migration
- Crashes or undefined behavior

**🟠 High (Should Fix)**
- Type safety violations
- Missing error handling for likely failures
- Missing tests for new functionality
- Logic errors that affect functionality

**🟡 Medium (Should Consider)**
- Pattern inconsistencies
- Missing edge case handling
- Complex code that could be simplified
- Performance concerns in non-critical paths

**🔵 Low (Suggestions)**
- Style preferences
- Minor optimizations
- Documentation improvements
- Future considerations

---

## Phase 4: Run Validation

### 4.1 Automated Checks

```bash
# Type checking
npm run type-check
# Capture: pass/fail, any errors

# Linting
npm run lint
# Capture: pass/fail, warning count, error count

# Tests
npm test
# Capture: pass/fail, test count, coverage if available

# Build
npm run build
# Capture: pass/fail, any warnings
```

### 4.2 Specific Validation

Based on what changed:
- If new API endpoint → test with curl
- If new adapter → check interface compliance
- If database changes → check migration exists
- If config changes → verify .env.example updated

### 4.3 Regression Check

```bash
# Run the full test suite
npm test

# If there's a test for the changed functionality, run it specifically
npm test -- [relevant-test-file]
```

---

## Phase 5: Generate Report

### 5.1 Report Structure

```markdown
# PR Review: #[NUMBER] - [TITLE]

**Author**: @[author]
**Branch**: [head] → [base]
**Files Changed**: [count] (+[additions]/-[deletions])

---

## Summary

[2-3 sentences: What this PR does and your overall assessment]

---

## Changes Overview

| File | Changes | Assessment |
|------|---------|------------|
| `path/to/file.ts` | +50/-20 | ✅ / ⚠️ / ❌ |

---

## Issues Found

### 🔴 Critical
[If none: "No critical issues found."]

- **[file.ts:123]** - [Issue description]
  - **Why**: [Explanation of the problem]
  - **Fix**: [Specific recommendation]

### 🟠 High Priority
[Issues that should be fixed before merge]

### 🟡 Medium Priority
[Issues worth addressing but not blocking]

### 🔵 Suggestions
[Nice-to-haves and future improvements]

---

## Validation Results

| Check | Status | Details |
|-------|--------|---------|
| Type Check | ✅ Pass / ❌ Fail | [any notes] |
| Lint | ✅ Pass / ⚠️ Warnings | [count] |
| Tests | ✅ Pass ([count]) | [coverage if available] |
| Build | ✅ Pass / ❌ Fail | [any notes] |

---

## Pattern Compliance

- [x] Follows existing code structure
- [x] Type safety maintained
- [x] Naming conventions followed
- [ ] Tests added for new code
- [ ] Documentation updated

---

## What's Good

[Acknowledge positive aspects of the PR - good patterns, clean code, etc.]

---

## Recommendation

**✅ APPROVE** / **🔄 REQUEST CHANGES** / **❌ BLOCK**

[Clear explanation of recommendation and what needs to happen next]

---

*Reviewed by Claude Code*
*Report: `.claude/PRPs/pr-reviews/pr-[NUMBER]-review.md`*
```

### 5.2 Decision Logic

**APPROVE** if:
- No critical or high issues
- All validation passes
- Code follows patterns

**REQUEST CHANGES** if:
- High priority issues exist
- Validation fails but is fixable
- Pattern violations that need addressing

**BLOCK** if:
- Critical security or data issues
- Fundamental approach is wrong
- Major architectural concerns

---

## Phase 6: Publish Report

### 6.1 Save Local Report

```bash
mkdir -p .claude/PRPs/pr-reviews
```

Save to: `.claude/PRPs/pr-reviews/pr-[NUMBER]-review.md`

Include metadata header:
```markdown
---
pr: [NUMBER]
title: [TITLE]
author: [AUTHOR]
reviewed: [TIMESTAMP]
recommendation: [approve/request-changes/block]
---
```

### 6.2 Comment on PR

```bash
# If approve flag was passed and no critical/high issues:
gh pr review [NUMBER] --approve --body "$(cat .claude/PRPs/pr-reviews/pr-[NUMBER]-review.md)"

# If request-changes flag was passed or high issues found:
gh pr review [NUMBER] --request-changes --body "$(cat .claude/PRPs/pr-reviews/pr-[NUMBER]-review.md)"

# Otherwise just comment:
gh pr comment [NUMBER] --body "$(cat .claude/PRPs/pr-reviews/pr-[NUMBER]-review.md)"
```

### 6.3 Output Summary

After publishing:
```
## PR Review Complete

**PR**: #[NUMBER] - [TITLE]
**Recommendation**: ✅ APPROVE / 🔄 REQUEST CHANGES / ❌ BLOCK

**Issues Found**:
- 🔴 Critical: [count]
- 🟠 High: [count]
- 🟡 Medium: [count]
- 🔵 Suggestions: [count]

**Validation**: [pass/fail summary]

**Report saved**: .claude/PRPs/pr-reviews/pr-[NUMBER]-review.md
**PR comment**: [link to comment]
```

---

## Special Cases

### If PR is a Draft
- Still review but note it's a draft
- Focus on direction rather than polish
- Don't approve/request-changes, just comment

### If PR is Large (>500 lines)
- Note that thorough review may miss things
- Suggest breaking into smaller PRs
- Focus on architecture over details

### If PR Touches Sensitive Areas
- Security-related code: Extra scrutiny
- Database migrations: Check reversibility
- Configuration: Check all environments

### If Tests Are Missing
- Strong recommendation to add tests
- Suggest specific test cases
- Don't necessarily block, but note the risk

---

## Critical Reminders

1. **Understand before judging.** Read the full context, not just the diff.

2. **Be specific.** "This could be better" is useless. "Use `execFile` instead of `exec` to prevent command injection at line 45" is helpful.

3. **Prioritize.** Not everything is critical. Use severity levels honestly.

4. **Be constructive.** Offer solutions, not just problems.

5. **Acknowledge good work.** If something is done well, say so.

6. **Run validation.** Don't skip the automated checks.

7. **Check patterns.** Read existing similar code to understand what's expected.

8. **Think about edge cases.** What happens with null, empty, very large, concurrent?

Now fetch the PR, review the code, and generate the report.
