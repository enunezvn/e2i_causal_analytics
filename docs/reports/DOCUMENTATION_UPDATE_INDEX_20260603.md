# Documentation Update Index — 2026-06-03

**Purpose:** authoritative backlog of project docs that must be **created** or **updated**
because recent merged work (PRs #600–#648 + the uncommitted Optum LOINC fix) has drifted
from them.

**Method:** 10 parallel doc-area auditors (compare real code vs. real docs, concrete drift
evidence) → completeness critic (missed areas + over-reach) → manual P0 re-verification.
Findings are the **union of workflow + critic + hand-verification**, each load-bearing claim
checked against the tree/`gh api`. This index lists the *gaps*, not the fixes — each entry is
self-contained enough to dispatch.

**Confidence:** HIGH. Every P0 claim was independently re-verified (see "Verification" at end).

---

## Summary

| | Count |
|---|---|
| Docs to **CREATE** (after consolidation) | 7 |
| Docs to **UPDATE** | 18 |
| P0 (blocks understanding / documents a shipped user/operator-facing change) | 9 items |
| P1 (important drift) | ~16 items |
| P2 (minor / nice-to-have) | ~9 items |
| Flagged as **code change, not a doc gap** | 1 (audit.py 503 OpenAPI) |

Consolidations applied (per critic, to avoid doc sprawl): the proposed `docs/qc_gate_reference.md`
is **folded into `docs/model_success_criteria.md`**; the 3 leakage findings collapse into one
`docs/data/08-LEAKAGE-DETECTION-CONTRACT.md`; the `audit.py` 503 finding is folded into the
`audit_chain.md` create as a small code change.

---

## 🔴 P0 hot-list (do first)

1. **CREATE `docs/model_success_criteria.md`** — the v3 adaptive success-criteria engine is the **production default** (#641) and governs every model pass/fail today, yet has *zero* doc home and its only cited spec path (`.claude/plans/adaptive_success_criteria/01-design.md`) does not exist.
2. **CREATE `docs/runbooks/migrations.md`** — prod migrations are **manual** (`deploy.yml` skips them when `SUPABASE_DB_URL` unset; the droplet has none). The working apply path (`docker exec -i supabase-db psql …`) is documented nowhere; `run_migrations.sh` documents the wrong path.
3. **CREATE `docs/data/08-LEAKAGE-DETECTION-CONTRACT.md`** — the Layer 1-5 adaptive temporal-validity leakage contract (manifest auto-detect #648, declared-safe FULL immunity, FDR/σ-band, the *cosmetic* Clean/Dropped console caveat) lives only in code docstrings pointing at a deleted plan.
4. **UPDATE `scripts/setup_branch_protection.sh` + `docs/ONBOARDING.md`** — both describe a policy (wrong repo slug `enunez/`, `ci-success` check, 1 approval + CODEOWNERS) that **contradicts the applied live state** (required: `Backend CI Success` + `Tier 1-5 agent harness`; 0 approvals; no CODEOWNERS). Running the script today would clobber the working solo-dev `--merge` policy.
5. **UPDATE `docs/OPTUM_CONVERSION.md`** — no entry for the #644 HCP gap-enrichment script / `optum_gap_enriched/*` cohorts (the cohort the entire 2026-06-03 analysis runs on).
6. **UPDATE `docs/SYNTHETIC_DATA.md`** — the `default`/`adverse`/`clean` `--regime` cohorts (backbone of #640/#645/#646) are entirely undocumented; Layer-4 still shows fixed bars while adaptive criteria are the default.
7. **UPDATE `INFRASTRUCTURE.md`** — CI/CD section describes the pre-#528 deploy (`git pull` + restart); real flow is `reset --hard` + GHCR image pull + Feast-gated rollout + conditional migrations.

---

## CREATE — new docs (7)

### C1 · `docs/model_success_criteria.md` — **P0** — effort M
**Why:** `criteria_validator.py:78` defaults `ADAPTIVE_CRITERIA="true"` since #641 (flipped 2026-06-02). The live v3 contract drops `minimum_precision`/`minimum_f1` and adds `net_benefit_at_p_t` (regime p_t adverse 0.05 / default 0.20 / clean 0.30), `minimum_mcc`, calibration slope/intercept/ECE gates, and `maximum_train_val_delta` (`evaluator.py:3460-3498`). No `docs/` file mentions `adaptive_success_criteria`, `net_benefit`, MCC, or the calibration gates.
**Add:** active v3 gates + meanings; the regime→p_t table; the dropped precision/F1 gates + Van Calster 2025 rationale; the `ADAPTIVE_CRITERIA=false` rollback to the fixed Apr-26 baseline (auc0.75/prec0.70/rec0.65/f1 0.70).
**Fold in (was proposed as separate `qc_gate_reference.md`):** the dynamic QC `overall_score` bar — `resolve_qc_min_overall_score` (#642, default 0.80, override precedence state > scope_spec > `QC_MIN_OVERALL_SCORE` env > default, can't be lowered unsafely) — and the all-null-column skip behavior (#630/#631, `qc_remediation.py:603-626` skips imputing all-null cols and leaves completeness blocking).

### C2 · `docs/data/08-LEAKAGE-DETECTION-CONTRACT.md` — **P0** — effort L
**Why:** the entire adaptive leakage stack (`adversarial_leakage.py`, `manifests/resolution.py:autodetect_manifest_source` #648, `adaptive_validity_check.py`, `structural_empirical_crosscheck.py`) is documented only in code docstrings that cite a now-deleted `.claude/plans/adaptive_temporal_validity_redesign.md`. `docs/data/` has zero coverage.
**Add (merges 3 findings):** (1) the 5-layer ensemble precedence ladder + σ-band thresholds (z>5σ high/drop, 3–5σ moderate/review, ≤3σ keep) and the #194 |ΔAUC| clamp; (2) the **declared-safe FULL manifest immunity** (#648, `_declared_safe_immune_features`, the only sanctioned severity downgrade; enforced in both the validity-check node and `leakage_remediation.py`; opt-in safety = `None` manifest_source ⇒ no immunity) vs. the synthetic-only FDR carve-out (#604); (3) the **cosmetic Clean/Dropped console caveat** — the report is the LLM `recommended_feature_set`, *not* the actual `train_df`; verify retained count via the sufficiency floor `2*n_features/minority_prevalence` or Feast-reg count; (4) the core principle: a correlation/structural/FDR detector cannot tell a strong pre-index predictor from a leak — only the temporal contract can; deterministic zero_variance/perfect_separation checks false-fire on sparse rare-event cols.
**Also:** register in `docs/data/00-INDEX.md` (see U17).

### C3 · `docs/runbooks/migrations.md` — **P0** — effort M
**Why:** `deploy.yml:244-247` runs migrations only if `SUPABASE_DB_URL` is set; the droplet env has none → migrations **never** auto-apply. `run_migrations.sh:46-50` hard-requires a `SUPABASE_DB_URL` connection string, but the real droplet path is `docker exec -i supabase-db psql …`. Migs 029/055/056/057 (#607) were applied by hand. Zero docs mention `supabase-db` or "migrations are manual."
**Add:** state plainly that prod migrations are manual; the authoritative apply command (`docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < database/migrations/<file>.sql`); the no-inner-transaction constraint (no `BEGIN/COMMIT`; `DO` blocks exempt); a cast/select disproof to verify; that `schema_migrations` tracking is bypassed by docker-exec. Cross-link from `DEPLOYMENT_ADMIN.md`.

### C4 · `docs/runbooks/ci-and-branch-protection.md` — **P1** — effort M
**Why:** decisive CI semantics live only in an agent-private plan + YAML comments. `docs/runbooks/` has only `sentinels.md`; no CONTRIBUTING, no CI runbook.
**Add:** the Tier-1-5 **honest-gate** (`tier1-5-test.yml:262-287`, empty `TIER1_5_EXPECTED_FAIL_AGENTS` ⇒ a green "Tier 1-5 agent harness" check provably means 13/13; how to allow-list an expected-fail agent); the dormant alarm-only `exit 0` and how to read the *real* signal (`tier1_5_pipeline_latest.json`, not check color); refreshing the committed fixture (`generate_tier0_fixture.py`); the slow-tests job map (A tracked / **B blocking heavy e2e** #636 / C mem+perf / D synthetic regimes) + the nightly issue-alarm & label self-heal (#643); branch-protection state + the docs-only-PR admin-merge footgun.

### C5 · `docs/api/audit_chain.md` — **P1** — effort M
**Why:** #609 wired `set_audit_chain_service(AuditChainService(supabase))` into the FastAPI lifespan (`main.py:264-271`); before it, every `/api/audit/*` returned 503 and `audit_chain_entries` was never written. The capability is now real and compliance-facing but has no API doc (only one ARCHITECTURE table row); the sibling executive-insights surface has a full `crystal_digests.md`.
**Add (mirror `crystal_digests.md`):** what the chain is (SHA-256 hash-linked genesis→per-node); the 4 endpoints (`GET /api/audit/workflow/{id}`, `/verify`, `/summary`, `/recent`) + response models + the 503 degraded-mode contract; how rows get written (each agent's `audit_init` genesis node, active since #609); auth = `require_auth`. State the pre-#609 no-op explicitly.
**Fold in (was a separate audit.py finding, reclassified as code):** the audit routes raise 503 but the router only declares `responses={401,422,500}` → the auto-generated `openapi.json` omits 503. **This is a code change, not a doc edit** (the spec is gitignored/auto-built): add `503: {model: ErrorResponse}` to `src/api/routes/audit.py` and regenerate.

### C6 · `docs/README.md` — **P1** — effort M
**Why:** no docs landing index exists (only `docs/data/00-INDEX.md` covers the data subtree). 50+ docs across 12 subtrees have no map; README links omit OPTUM_CONVERSION, RWD_PIPELINE, synthetic_v3_design, runbooks, reports, governance/calibration/lineage/layer4/specs/rca/results and top-level INFRASTRUCTURE/DEPLOYMENT_ADMIN/OOM_FIX_README.
**Add:** a thin one-line-per-doc landing index grouped by purpose (Architecture & Onboarding; Data & Schema; Pipelines; Ops & Runbooks; Governance/Calibration/Lineage/Layer4; Reports & Results). Link from README "Documentation."

### C7 · `docs/CHANGELOG.md` — **P1** — effort M — *(critic add)*
**Why:** README still stamps **Version 4.2.1 / Last Updated February 2026**; the newest "What's New" is v4.2.1. The entire tier-0 lifecycle arc (#640/#641/#642), leakage/class-imbalance (#648/#645/#646), 21-agent reconciliation (#607/#601), audit-chain wiring (#609), keyless harness (#600/#606) are all post-Feb-2026 with no chronological home. No CHANGELOG exists anywhere.
**Add:** a Keep-a-Changelog-style file seeded from the merged PRs since Feb 2026; fix the README version/date stamp (the first staleness signal a contributor sees).

---

## UPDATE — existing docs (18)

### U1 · `docs/OPTUM_CONVERSION.md` — **P0/P1** — effort M (bundle)
- **P0** — no entry for `scripts/enrich_cohort_with_hcp_features.py` (#644) or the `optum_gap_enriched/<cohort>/` it produces (8 leakage-safe `treating_hcp_*` cols, joined off `medication.npi` with `medication_date ≤ index`; Gap clinical tables deliberately excluded as post-index; rolling-window scores ⇒ harness-only, not deployable). Add to Outputs + Related files.
- **P1** — surface the documented **unmodelability** of the gap-enriched cohort (37 events, EPV 0.13, CV-AUC ~chance) as a genuine raw-data limit, NOT a conversion bug; link `docs/results/tier0_cohort_comparison_optum_vs_synthetic_20260603.md`.
- **P1** — ✅ DONE (`fix/loinc-corrections-doctrail`): recorded the 2026-06-03 `CSU_LABS_LOINC` corrections (eosinophil/tpo_ab/ana/cbc were mislabeled; now extract-verified codes; guarded by `TestCsuLabsLoincMapping`). The converter LOINC change + the cited forensics doc (`docs/results/tier0_cohort_comparison_*` now allow-listed in `.gitignore`) ship in the same change.
- **P2** — add 3 missing converter CLI flags to the flags table: `--enrollment-regime`, `--extract-ym`, `--comorbidity-method`.
- **P2** — document the #648 Feast advisory for `file_dir` runs + the `ALLOW_STALE_FEAST=1` escape hatch (no `feature_store.yaml`).

### U2 · `docs/RWD_PIPELINE.md` — **P1** — effort M
Dated 2026-04-12; "New CLI flags" table lists only 4 flags. `run_tier0_test.py` now also has `--regime`, `--feature-manifest-source {csu,optum,synthetic}` (auto-detected from `--data-dir`; load-bearing for leakage immunity; **correction (verified at runtime 2026-06-03):** `optum_gap_enriched` **does** auto-detect to `optum` via the #648 `<source>_`-prefix match in `resolution.py` — the earlier "needs explicit override" claim was wrong), `--split`, `--min-samples-per-split`, `--n-total`, `--seed`, etc. The non-flag claims (wrapper removed, JSON format) are *verified still accurate* — scope the edit to the flags table + date.

### U3 · `docs/SYNTHETIC_DATA.md` — **P0×2 / P1×2** — effort M (bundle)
- **P0** — document the legacy `--regime default/adverse/clean` cohorts (`_VALID_REGIMES`, `run_tier0_test.py:4374`): default (balanced baseline), adverse (extreme imbalance ~0.02 for #645/#646), clean (signal 1.35 / 4000 rows, honest-green deploy-calibrated #640). Only the v2 `scenario_*` family is currently documented.
- **P0** — Layer-4 shows fixed bars (AUC≥0.55, recall≥10%, ATE±0.05); note `ADAPTIVE_CRITERIA` defaults true so adaptive regime/N-keyed thresholds govern by default (fixed table = rollback).
- **P1** — "Generates 1500 patients" is regime-conditional: `clean` uses 4000 (`_REGIME_N_SAMPLES`) to close the overfit gate (#633/#640).
- **P1** — the Tier-0 fixture is **committed** (`scripts/generate_tier0_fixture.py`, #600); doc still says Tier 1-5 "requires a prior Tier-0 run."

### U4 · `docs/ARCHITECTURE.md` — **P1/P2** — effort M
- **P1** — add the Tier-0 gate stack: data_preparer QC `overall_score` dynamic bar (#642); model_trainer adaptive v3 criteria gate (default-on, #641); the deploy-calibrated artifact contract (#640 — the calibrated model is the checkpointed/deployed artifact so calibration gates judge deployed probs). *(Agent count 21 / 6-tier is already correct — verified; do NOT "fix" it.)*
- **P2** — note the AuditChainService is bound in the FastAPI lifespan (#609) and reset on shutdown; refresh the "Last Updated: February 2026" stamp.

### U5 · `docs/data/02-CORE-DATA-DICTIONARY.md` — **P1** — effort S
Lines 64/67/1581/1585/1882 list dropped enum `agent_name_type_v2` as live and cite 11/18/20-agent counts. Migration 056 drops `agent_name_type_v2/v3` + `agent_tier_type_v2`; migs 055/057 land 21 agents + `cohort_constructor`/`experiment_monitor`. Siblings 04 & 07 already say 21. Fix counts → 21, mark v2/v3 enums RETIRED (cite 056), add migs 055/056/057 to the migrations table.

### U6 · `docs/data/03-ML-PIPELINE-SCHEMA.md` — **P1** — effort S
`minimum_auc` (lines 59/69/73) presented as THE success threshold; silent on the adaptive engine (#641) that overrides it by default. Add a note: fixed-mode (`ADAPTIVE_CRITERIA=false`) bar vs. default adaptive gates (NB/MCC/calibration, precision/F1 not enforced). Cross-link C1.

### U7 · `docs/reports/tier1_5_pipeline_status.md` — **P1** — effort M
Header says "no live harness run … no measured numbers" (2026-06-01); harness is now faithfully 13/13 on main. Rewrite Finding 1 as ✅ with the honest-gate (empty allow-list ⇒ green==13/13) + the dormant alarm-only `exit 0` caveat; mark Finding 2 resolved (ONBOARDING/ARCHITECTURE already at 13); fix §8 line anchors; consider linking C4 rather than re-deriving CI detail.

### U8 · `README.md` — **P1×2** — effort M (bundle)
- Repoint/remove the 4 broken "What's New in v4.2.1" doc links (all 404'd; files now in `docs/Archive/`).
- Add a Tier-0 ML pipeline + Tier-1-5 keyless harness subsection; add `tier1-5-test.yml` (a **required** check) + `slow-tests.yml` to the CI/CD table; list the new entry-point scripts (`run_tier0_test.py`, `run_optum_tier0_test.py`, `run_tier1_5_test.py`, `convert_optum_rwd.py`, `generate_tier0_fixture.py`); refresh the version/date footer (overlaps C7).

### U9 · `scripts/setup_branch_protection.sh` — **P0** — effort S
`REPO="enunez/e2i_causal_analytics"` (wrong; should be `enunezvn/…`), `contexts=["ci-success"]` (wrong), 1 approval + CODEOWNERS (wrong). Live state: required `Backend CI Success` + `Tier 1-5 agent harness`, 0 approvals, no CODEOWNERS, `enforce_admins=false`, `required_linear_history=false` (keeps `--merge`). Rewrite to emit the applied config + a comment on why only every-PR checks are required and approvals are 0.

### U10 · `docs/ONBOARDING.md` — **P0** — effort S
Lines 380-382 ("≥1 approval / CODEOWNERS required / stale reviews auto-dismissed") + :353/:881 contradict the applied protection. Update to 0 approvals, no CODEOWNERS gate, the two required checks, and the path-filtered-checks→admin-merge footgun. (Same P0 as U9.)

### U11 · `DEPLOYMENT_ADMIN.md` — **P1** — effort S
No mention of the applied branch protection or the admin-merge override for path-filtered required checks (the #644 deadlock: scripts/tests-only PR never triggers the path-filtered harness check → blocked forever without `gh pr merge --admin`). Add a "Branch protection & merging" section (required checks, 0 approvals, no force-push/delete, always-merge/never-squash, the override + that it needs explicit authorization).

### U12 · `DEPLOYMENT.md` — **P1** — effort S
Rollback section (122-128) says the workflow "runs migrations"; the step is conditional and **skipped on the droplet**. Correct to "runs migrations only if `SUPABASE_DB_URL` is set — on the droplet it is not, applied manually (see `docs/runbooks/migrations.md`)"; note the Feast-materializer gate + `rollback_to_prev`.

### U13 · `INFRASTRUCTURE.md` — **P0 / P2×2** — effort M (bundle)
- **P0** — CI/CD section (595-624, 455) describes the pre-#528 deploy (`git pull` + restart, uvicorn `--reload`). Real `deploy.yml`: `git reset --hard origin/main`, GHCR build-and-push of api+frontend, droplet `docker login` + pull, Feast-materializer freshness gate, ordered force-recreate to the prod target, conditional migrations, 30m timeout, `rollback_to_prev` + health check.
- **P2** — add a Dependency-security note (pip-audit in CI; PyJWT 2.13.0 #638; **requirements.txt change MUST sync requirements.lock** or lock-drift test reds the PR; prod installs from the lock under `--require-hashes`).
- **P2 (downgraded — pre-existing, not recent drift, per critic)** — reconcile the "Configure automatic backups" checklist box with the existing `backup_*.sh` scripts; bundle only if a runbooks pass happens.

### U14 · `docs/superpowers/plans/2026-05-22-data-sufficiency-diagnostics-rollout.md` — **P1** — effort S
Add the #646 split-time **class-presence guard** (`split_enforcer._check_class_presence`): trigger (a classification split lands 0 of a class on a low-prevalence cohort), effect (blocks via `ratios_valid=False`), scope (classification only; whole-dataset single-class deferred to the sufficiency HARD_FAIL). *(Alternatively fold into C2.)*

### U15 · `docs/governance/n3_known_limitations_20260510.md` — **P2** — effort S
Record the all-null-column skip-and-report behavior (#630/#631) as a deliberate limitation. *(Overlaps C1's folded QC content — pick one home; prefer a one-line pointer here → C1.)*

### U16 · `docs/api_connectivity_review.md` — **P2** — effort S
Dated 2026-05-16: "161 routes" (now 157), stale `main.py` mount line refs, and documents `AuditChain.tsx → /api/audit/recent` as live when audit was unwired until #609. Add a dated addendum (audit now returns real data via #609) + refresh counts/line-refs, OR prepend a "point-in-time, see C5" banner.

### U17 · `docs/data/00-INDEX.md` — **P2** — effort S — *(critic add)*
Register the new `08-LEAKAGE-DETECTION-CONTRACT.md` (C2) in the document map, and verify the index itself doesn't carry the same stale 11/18/20-agent drift confirmed in its sibling 02 (U5).

### U18 · `closure_memo_T1.2.md` — **P2** — effort S
Stale in-flight memo at repo root; PR #229 merged 2026-05-15 and the ablation hook shipped. `git mv closure_memo_T1.2.md docs/Archive/` (no content change) to declutter the root and signal it's a historical decision record.

---

## Critic adjustments applied

**Over-reach dropped/downgraded:**
- `INFRASTRUCTURE.md` backups-checklist → kept only as a **P2 bundle-if-convenient** note (pre-existing, not recent-PR drift).
- `src/api/routes/audit.py` 503-in-OpenAPI → reclassified as a **code change** (the spec is gitignored/auto-generated, so there's no doc to edit); folded into C5.
- proposed standalone `docs/qc_gate_reference.md` → **folded into C1** to avoid a third loose tier-0 doc.

**Cross-cutting adds:** C7 (CHANGELOG + README version stamp), U17 (docs-index registration), and an explicit note that several **memory-only invariants** (cosmetic Clean/Dropped report; "only the temporal contract distinguishes a pre-index predictor from a leak"; gap-enriched Optum unmodelability; the docker-exec migration path) should be **promoted out of agent memory into C1/C2/C3** so a human operator can find them.

---

## Verification (P0 claims, hand-checked 2026-06-03)

- `grep -rliE "adaptive_success_criteria|ADAPTIVE_CRITERIA|net_benefit" docs/ README.md` → **0 hits**; `.claude/plans/adaptive_success_criteria/01-design.md` **absent** → C1 gap real.
- `scripts/enrich_cohort_with_hcp_features.py` **present** (10.7 KB, 2026-06-03) → U1 P0 real.
- `setup_branch_protection.sh`: `REPO="enunez/…"`, `contexts=["ci-success"]`, `required_approving_review_count:1`, `require_code_owner_reviews:true`; `ONBOARDING.md:380-382` asserts the same → contradicts applied state → U9/U10 P0 real.
- `deploy.yml:244-247` gates migrations on `SUPABASE_DB_URL`, else "Skipping migrations" → C3 P0 real.
- README's 4 "What's New" links → **all 4 missing** from `docs/`, present in `docs/Archive/` → U8 real.
- `02-CORE-DATA-DICTIONARY.md:64/67/1581/1585/1882` stale enums + 11/18/20 counts; `056_*.sql:40-42` drops all three v2/v3 enums; `04`/`07` already say 21 → U5 real.
- `ARCHITECTURE.md:33/307/1051` already say **21 agents / 6 tiers** → confirmed; agent-count is NOT a drift item (would be over-reach).

**Source:** workflow `wf_2af142f9-490` (10 auditors + critic, 39 raw findings).
