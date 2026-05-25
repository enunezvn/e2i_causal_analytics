# E2I Causal Analytics - Claude Code Instructions

## REASON BEFORE RULES (HIGHEST PRIORITY — 2026-05-21, user-pinned)

> **ABOVE ALL ELSE: REASON. DO NOT FOLLOW ARBITRARY RULES.**

Every rule in this file (and in memory) was written in response to a specific incident. The rules capture lessons but cannot anticipate every situation. Your obligation is to **understand WHY a rule exists** before applying it — and to set the rule aside (with explicit reasoning shown to the user) when the case in front of you doesn't match.

### The reasoning step BEFORE any classification, recommendation, or action

Answer in order:

1. **What is this code trying to do?** (intent — `git log`, PR body, linked issues, surrounding comments, conversations)
2. **Why does it exist in its current shape?** (scaffolded placeholder, copy-paste error, deprecated, feature-flag-gated, vestigial)
3. **Is it causing harm right now?** (user-visible? plausible-wrong values that look real? security gap?)
4. **What does the user actually want?** (their stated goal, not your inferred goal from rule-matching)
5. **Only then**: classify, recommend, act — and explain your reasoning, not just your conclusion.

Skipping steps 1–4 and jumping to step 5 is "lazy programming." A grep is a snapshot of *now*. Product intent lives in PRs, issues, conversations, and the user's head — not in `grep -rn`.

### When a rule conflicts with reasoning

If a rule says "DELETE on zero consumers" but you discover the code is a scaffolded placeholder for functionality the user requested, **the rule does not fit this case**. Stop. Tell the user what you found and why the rule doesn't apply. Propose a path that solves the actual problem.

### Incident that forced this directive

After the plan-354 LABEL-vs-DELETE incident (2026-05-21), I wrote an anti-mocking rule saying "no consumer == DELETE candidate, LABEL forbidden" and dispatched a census agent under this rule. The user stopped me: *"if the mock exists it is because I requested the functionality, why would we delete it? I think we are making decisions without thinking — why can you not add a reasoning step to your work?"*

I had been pattern-matching on the previous incident (which was correctly diagnosed as LABEL-when-DELETE-warranted), not reasoning about each case. The anti-mocking rule captured a real lesson but I generalized it into a blanket policy that conflated four very different reasons a mock can legitimately exist:

- **Scaffolded placeholder** — user requested the functionality, real implementation is in flight or blocked
- **Roadmap stake** — endpoint is documented in product roadmap; integration is planned
- **Feature flag / dev path** — intentionally non-production, never reachable in prod traffic
- **Vestigial / copy-paste error** — actually dead

Only the last is unambiguously DELETE. The others require different responses, and rule-following without intent investigation conflates them.

Memory: [[feedback-reason-before-rules-20260521]].

---

## CHEAPEST-DISPROOF FIRST (HIGHEST PRIORITY — user-pinned 2026-05-25)

> **Before building or recommending any expensive solution, run the cheapest experiment that could DISPROVE it — in an environment FAITHFUL to the target. Do not theorize or pattern-match a solution into existence — get the disproving data FIRST, then propose the solution.**

Mechanism-validation is not premise-validation. A green test suite, a clean codex audit, and tidy commits prove your *code* is correct; they say NOTHING about whether the *solution's core assumption* holds. The assumption is only validated by measuring the real outcome **in the real environment**. Elaborate rigor built on an unverified premise is guessing in a lab coat.

### Required BEFORE writing code or proposing a fix

1. Name the **single assumption** the solution depends on ("batching will parallelize this", "the bottleneck is X", "this consumer needs Y").
2. Identify the **cheapest experiment that would prove that assumption FALSE**.
3. **Run it and show the result.** Proceed only if the assumption survives.

- Prefer free/local/instant over expensive: READ the dependency's source, run a one-line repro, inspect the real runtime/telemetry/config — BEFORE a full build, a long CI run, a multi-agent dispatch, or a PR.
- A projection or model ("should be ~6× faster") is a **hypothesis**. Label it unverified; never present it as a measured result; falsify it cheaply before acting.
- **The experiment must be FAITHFUL to the target environment** — same key/tier/config/scale. A passing run in a non-faithful environment is a false green. If you can't run it faithfully (e.g. you lack the prod/CI key), say so and treat the local result as suggestive, not decisive — the faithful experiment IS the target environment.

### Incident that forced this directive (#504, 2026-05-25)

Asked to speed up the ~96-min RAGAS CI gate, I took the investigation's "~12–18 min via batching" projection as fact and built the whole thing — batched refactor, 9 red-first tests, two codex rounds to ACCEPT, clean commits, a PR — on the **unverified premise that batching parallelises the gpt-4o judge calls**. A `to_thread` wrapper serialised it in CI (1/120 jobs in 75 min). The cheap disproof I'd skipped — a ~5-min read of the cached ragas source — then found the real cause: our **sync** `openai.OpenAI()` client forces ragas off its async path. Fixed with `openai.AsyncOpenAI()` → a local benchmark showed **6× faster (n=30 in ~64 s, gates matching main)** — GREEN. But that local key was a **higher tier than CI's**; on CI's key the real concurrency tripped ragas's per-job timeouts (faithfulness→0.000, a 49-min failing run). The binding constraint was never the code — it's **CI's OpenAI key throughput**. A cheap experiment in an *unfaithful* environment (wrong key tier) gave a false green and cost another failed run. Resolved by making the eval manual-only. Memory: [[feedback-cheapest-disproof-first-20260525]].

---

## Anti-Mocking & Verification Discipline (SUBORDINATE to REASON-BEFORE-RULES)

This section captures specific lessons from plan-354. It does NOT override the requirement to investigate intent first.

### The actual rule

> **DATA-DRIVEN EVIDENCE-BASED CODING, NOT LAZY PROGRAMMING.**
> NO silent mocks in production code paths returning plausible-but-fake values.
> NO patching around stubs without first understanding why the stub exists.
> NO theoretical assumptions about consumer needs — investigate what the user actually requested.

A mock is **not bad** per se. A mock that returns plausible-wrong values silently in a user-facing or production-reachable path IS bad. A mock clearly marked as a scaffolded placeholder, intent-documented, not reachable in prod traffic, awaiting real implementation — is a normal product-development pattern.

### Required investigation BEFORE classifying any mock

**Step 1 — Intent investigation:**

```bash
# When was it added, in what PR, by whom?
git log --diff-filter=A --follow <file_path> | head -20

# What did the PR body say about intent?
gh pr view <pr_number> --json title,body

# Are there open issues / roadmap items referencing it?
gh issue list --search "<symbol_name>"

# Any inline comments / docstrings about future work?
grep -B2 -A5 "<placeholder_pattern>" <file>
```

**Step 2 — Harm assessment:**

- Is the mock currently reachable in production traffic? (check router wiring, auth gates, feature flags)
- Are returned values plausible enough to be mistaken for real? (e.g., `ate=0.12` looks like pharma uplift)
- Is it user-visible? (chat UI, exported reports, dashboards)

**Step 3 — Consumer check:**

```bash
grep -rn "<symbol_or_endpoint>" src/ --include="*.py" | grep -v "<file_under_review>"
grep -rn "<endpoint>" frontend/src/ --include="*.ts" --include="*.tsx" | grep -v "/generated/"
grep -rn "<endpoint>" --include="*.py" --include="*.yaml" --include="*.yml" --include="*.sh"
```

**Step 4 — THEN classify, with intent + harm in hand:**

- **HARMFUL-NOW** → user-facing or plausible-wrong values → immediate REWIRE, or gate behind flag with intent-documented stub
- **REWIRE** → functionality requested + real implementation feasible now
- **KEEP-AS-INTENTIONAL-PLACEHOLDER** → functionality requested + real implementation blocked + mock is non-harmful (clearly-fake values, behind flag, internal-only) + intent documented inline
- **DELETE** → no recoverable intent (vestigial, copy-paste error, no requested functionality)

"LABEL" (adding `mock_data: True` schema field or disclaimer to keep silently-fake values reachable) is rejected — but that rejection follows from the harm-assessment + intent-investigation, not from a blanket prohibition. The plan-354 case WAS LABEL-when-DELETE-warranted. Another mock might be KEEP-AS-INTENTIONAL-PLACEHOLDER.

### Codex audit briefs MUST invite design pushback

Every codex iter-N brief MUST include this paragraph verbatim:

> If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given.

### Detection signals — patterns to INVESTIGATE (not auto-act on)

When you encounter these in production code, STOP and investigate intent — do not classify on pattern-match alone:

- `# Placeholder` / `# Mock` / `# Stub` / `# TODO: real` comments
- `random.uniform(...)` / `np.random.seed(42)` in production paths
- Hardcoded numeric returns matching plausible production values (`ate=0.12`, `confidence=0.85`, `p_value=0.001`)
- Functions returning structured results with all-default or all-zero fields
- "Placeholder implementation - actual X would go here" docstrings

**Action**: investigate intent + harm, classify per the 4-way framework above, recommend with reasoning shown.

### Recommendation form

Every recommendation must answer:

> "What is this code trying to do, why does it exist in this shape, is it causing harm, and what should we do about it?"

Skipping any of those is rule-following without reasoning.

### Verify before reporting plan results

When summarizing a plan agent's output, the dispatcher MUST verify the plan's central claims (intent investigation, harm assessment, consumer counts) BEFORE passing the summary up. A codex-ACCEPT plan is not authoritative until claims are independently verified.

---

## Git & GitHub

### Corporate Proxy Bypass (REQUIRED)
Always bypass the corporate proxy for GitHub operations. Before any git push/pull/fetch:

```bash
git config --global http.https://github.com.proxy ""
```

This prevents 403 errors from the Novartis corporate proxy intercepting GitHub traffic.

### Authentication
- GitHub PAT is stored in `.env` as `GITHUB_PAT`
- Use HTTPS with credential helper, not SSH

### Merge policy: always preserve, never squash
- Always preserve commit history when merging PRs. Use `--merge` (merge commit) or `--rebase` (linear); never `--squash`.
- Applies to all branches regardless of size — single-commit chore/security branches included.
- If GitHub auto-merge is configured for this repo, set the squash option off before merging.

## Project Overview

- **Type**: Pharmaceutical analytics platform with causal inference
- **Stack**: Python 3.12, FastAPI, LangGraph, DSPy, Supabase, Redis, FalkorDB
- **Frontend**: React/TypeScript in `frontend/`

## Code Quality

- **Type checking**: `mypy --config-file pyproject.toml src/`
- **Linting**: `ruff check src/`
- **Tests**: `pytest tests/`

## Known Issues

- Large codebase (~5GB with dependencies) - see `OOM_FIX_README.md` for memory optimization
- Use `.claudeignore` patterns to prevent indexing heavy directories
