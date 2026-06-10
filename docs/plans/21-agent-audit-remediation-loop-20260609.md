# 21-Agent Audit — Remediation Loop Brief

Autonomous loop brief for finishing the audit's prioritized remediation. Execute
**one finding at a time** until every remaining finding is PR'd (or skipped as
already-handled). This is the worklist + method the loop must follow.

## Sources of truth (read these per finding before coding)

- `docs/reports/21-agent-audit-20260609.md` — per-finding analysis + §7 priorities
- `docs/reports/21-agent-audit-20260609-remediation-status.md` — living status tracker
- `docs/reports/21-agent-audit-20260609-repro/reverify_results.json` — precise current
  fix locations (re-verified vs main)

## Status

- **DONE — do not redo:** F1 health_score → PR #823; F2 gap_analyzer → PR #824;
  F5 orchestrator + F7 experiment_monitor → already merged.
- **REMAINING (process in this P0→P3 order):**
  1. F6 — tool_composer: `success=True` on 0/N tools
  2. F8 — feature_analyzer: unlabeled `np.random` SHAP background
  3. F3 — observability_connector: `client=` kwarg bug → serves mock while 5313 real spans exist
  4. F4 — model_deployer: simulated registration → `success=True`, writes 0 rows
  5. F9 — model_selector: `MLDataLoader.execute_query` missing → 40% frozen score
  6. F11 — drift_monitor: `structural_drift` dropped
  7. F10 — experiment_designer: `MockKnowledgeStore` unmarked
  8. F15 — feedback_learner: empty stores → 0.0 effectiveness
  9. F16 — data_preparer: `DataFrame.append` removed in pandas 2.x
  10. F17 — cohort_constructor: bad top-level import
  11. F12 — heterogeneous_optimizer / F13 — resource_optimizer / F14 — prediction_synthesizer:
      no #260 `input_model` bridge → dead via chat, currently fail-closed
  12. F19 — model_trainer: no in-agent thread cap → 5.9 GB CV/perm/bootstrap
- **F18** (causal_impact) is fail-closed-CORRECT / optional — skip unless time remains.

## Method per finding (non-negotiable)

1. **Check first / no duplicates.** Run `gh pr list --state open` and
   `git ls-remote --heads origin`. A concurrent session is active (issue #821). If a
   branch/PR already exists for this finding, SKIP it and move on.
2. **Worktree isolation.** `git fetch origin`, then create a worktree off current
   `origin/main` with a descriptive branch `fix/<agent>-<short>-fNN`.
3. **TDD red-first.** Write the failing test that encodes the audited defect before the fix.
4. **REAL fix, no mocks. REASON-BEFORE-RULES.** Investigate intent (git log, PR, linked
   issue, surrounding code) BEFORE classifying any mock. Use the 4-way framework:
   HARMFUL-NOW / REWIRE / KEEP-AS-INTENTIONAL-PLACEHOLDER / DELETE. The systemic defect
   class here is "graceful degradation that fails OPEN" — fix = fail CLOSED (honest
   error/empty/unknown), never fabricate success.
5. **Faithful environment.** Prod DB is the LOCAL docker Supabase, queried via
   `docker exec supabase-db psql -U postgres -d postgres`. The cloud Supabase MCP project
   is a stale, NON-FAITHFUL mirror — never use it for data/row-count checks.
6. **Memory-guarded tests ONLY (OOM-constrained shared box).** Pre-flight `free -m`,
   require >= 2.5 GB available. Run scoped tests under a cgroup ceiling with single-thread
   caps and the worktree on PYTHONPATH (so the worktree's src shadows the editable
   install). Template:
   `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=<worktree> <repo>/.venv/bin/python -m pytest <scoped test paths> -p no:cacheprovider -q`
   NEVER run whole-tree pytest or whole-tree mypy on this box. CI is the authoritative
   full-suite + MyPy gate. Some health/RAG tests hit real services and flake/hang under
   xdist — that is CI-neutral; trust CI.
7. **Codex convergence.** Use `codex:codex-rescue` with model `-m gpt-5.5` (the default
   gpt-5.3-codex is rejected by this account). Iterate to fixed point. VERIFY codex
   findings against source before accepting (codex can over-route / pattern-match).
8. **Push + PR.** Bypass the corporate proxy first:
   `git config --global http.https://github.com.proxy ""`. Push the branch, open a PR
   referencing the finding number + the audit report. NO-squash policy.
9. **Do NOT merge, do NOT deploy.** Merges + a single deploy are BATCHED at the very end
   (deploy is held). Update `21-agent-audit-20260609-remediation-status.md` (mark the
   finding's Status + PR#) as each PR opens.

## OOM discipline

Serialize heavy findings (F8 SHAP, F9/F19 sklearn, F4 model-ops). Light findings
(F16/F17 import fixes, F10/F15 wiring) may run with limited parallelism. One heavy
test-runner active at a time.

## Stop condition

Stop the loop once every REMAINING finding above is either PR'd or skipped as
already-handled by a concurrent session.
