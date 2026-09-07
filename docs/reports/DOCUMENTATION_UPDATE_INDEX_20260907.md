# Documentation Update Index — 2026-09-07

**Purpose:** authoritative backlog of project docs that must be **created** or **updated**
because merged work since the July doc-audit (DOC-AUDIT-202607, PRs #1279–#1284) has drifted
from them. This is the phase-1 ledger; phase 2 executes it as docs PRs. Nothing outside this
file was edited while building it.

**Window audited:** merged PRs 2026-07-20 → 2026-09-07 (first-parent `Merge pull request`
commits on `main`: 78 in 07-20..07-31, 232 in August, 37 in September so far; 344 by a single
`--since=2026-07-20` count — the difference is day-boundary rounding, use the measure command
below when a number is needed).

**Method:** 8 parallel doc-area auditors (doc-driven: every concrete claim in the in-scope
docs re-derived from the tree, live droplet state read-only where the claim is about the live
box) + 4 merged-PR sweeps (change-driven: every PR in the window classified DOCUMENTED /
UNDOCUMENTED / STALE / NOT_DOC_WORTHY with the doc it should have touched) → completeness
critic (missed areas, over-reach, contradictions between auditors, consolidation) → hand
re-verification of every P0 and of the P1 claims quoted below. Workflow `wf_3bca4bc3-0f6`
(13 agents, 359 doc findings + 151 PR-coverage rows; 171 STALE, 76 MISSING, 89 CONFIRMED_OK,
14 UNCERTAIN, 9 NOT_A_DOC_GAP). Findings are the **union of workflow + critic + hand
verification**; where an auditor was wrong the ledger carries the corrected fact and says so.

**Confidence:** HIGH on P0 (all 8 re-run by hand, see *Verification*), HIGH on the P1 rows
that carry a measure command (each was re-run by hand or by the critic), MEDIUM on P2 rows
(auditor evidence accepted when the measure command is deterministic; not all re-run).

**Scope (in):** README.md, CLAUDE.md, DEPLOYMENT.md, docker/README.md, docker/env.example,
.env.example, docs/ARCHITECTURE.md, docs/ONBOARDING.md, frontend/README.md,
docs/LLM_CONFIGURATION.md, docs/decisions/, CHANGELOG.md, docs/runbooks/*, docs/api/*,
docs/data/* (12 md + csv + templates), docs/SYNTHETIC_DATA.md, docs/RWD_PIPELINE.md,
docs/roi_methodology.md, docs/OPTUM_CONVERSION.md, docs/OPTUM_MART_CONVERSION.md,
docs/model_success_criteria.md, docs/synthetic_v3_design.md, docs/api_connectivity_review.md.

**Scope (out, deliberately):** docs/Archive/, docs/reports/, docs/results/, docs/demos/,
docs/superpowers/, docs/calibration/, docs/lineage/, docs/specs/, per-agent
CONTRACT_VALIDATION.md files, the in-app "How E2I Works" page content
(`frontend/src/components/documentation/content.ts` — checked only as evidence, not audited),
generated artifacts (openapi.json, api.ts).

**Conventions:** priorities P0 (an operator or newcomer following the doc does the wrong thing
or is blocked) / P1 (wrong fact, procedure still works; or a shipped user/operator-facing change
with no doc home) / P2 (minor, cosmetic, age-well rewording). Effort S/M/L. Every row that
states a number carries the command that produced it; phase 2 must re-run it at write time
(July lesson: numbers drift between audit and PR).

---

## Summary

| | Count |
|---|---|
| Docs to **CREATE** (after consolidation) | 10 (3 runbooks, 5 ADRs, docs/api README, error-envelope note) + 1 decision item C9 (docs/api/causal.md & segments.md) |
| Docs to **UPDATE** | 34 files (bundled below as U1–U34) |
| **Cross-cutting facts** (one truth, many docs) | 17 (X1–X17) |
| P0 (verified, do first) | 8 findings → **5 phase-2 work items** (X1, X2, X3, U5-Step 7, U11) |
| P1 | ~95 findings → ~60 ledger rows |
| P2 | ~185 findings → ~110 ledger rows (many folded into age-well rewrites) |
| Flagged **code/config change, not a doc gap** | 10 (N1–N10) |
| Docs verified **CONFIRMED_OK, no action** | CLAUDE.md (all factual claims), docs/decisions/adr-009..012 bodies, docs/roi_methodology.md constants/formulas, docs/runbooks/kg_cache.md build/verify sections, docs/runbooks/frontend-env-and-csp.md §#24, most of docs/api/chat.md §2/§6/§7 |

Consolidations applied (per critic): the four SUPABASE_POSTGRES_PASSWORD findings → **X1**;
the four provider-truth findings → **X2**; nine "which compose set is production / make deploy"
findings → **X3**; five monitoring-profile findings → **X4**; twelve "compose-forwarded knob
nobody listed" rows + the 23-missing-vars finding → **X5** (one derived env-var reference);
three CHANGELOG umbrellas + README L127/L129 + 14 sweep rows whose only doc home is the
CHANGELOG → **X6**; twelve per-migration rows → **X7** (dictionary catch-up 112–132 with a
checklist); eleven sweep-4 rows proposing docs/api/causal.md → one **decision item (C9)**.

---

## 🔴 P0 hot-list (do first — all hand-verified 2026-09-07)

1. **X1 · `SUPABASE_POSTGRES_PASSWORD` is compose-enforced but absent from `.env.example`
   and every Required-env table.** `docker compose --env-file .env.example -f
   docker/docker-compose.yml config -q` fails: *"required variable SUPABASE_POSTGRES_PASSWORD
   is missing a value"*. Quick Start step 3 in DEPLOYMENT.md, docker/README.md and ONBOARDING
   §3 cannot succeed as written. Targets: `.env.example` §2, DEPLOYMENT.md L43-52,
   docker/README.md L59-68, docs/ONBOARDING.md L96-108.
2. **X2 · Provider truth is inverted in docker/README.md and ONBOARDING.** Both say
   `ANTHROPIC_API_KEY` required / `OPENAI_API_KEY` optional; the code default is
   `LLM_PROVIDER=openai` (`src/utils/llm_factory.py:112`) and RAG embeddings are OpenAI
   regardless of provider. A newcomer with only an Anthropic key gets a provider with no key.
   DEPLOYMENT.md / .env.example / LLM_CONFIGURATION.md were already corrected in July (#1279);
   mirror that wording. Targets: docker/README.md L59, docs/ONBOARDING.md L73-77,
   docker/env.example (decision: pointer or delete, see U4).
3. **X3 · Production runs the BASE compose only, deployed only by `deploy.yml`; `make deploy`
   / `scripts/deploy.sh` is a legacy dev-overlay path that must never run on the droplet.**
   `deploy.yml` `pick_overlay()` returns no overlay because `docker/frontend/Dockerfile:106` is
   `FROM nginx:alpine AS production`; live `docker ps` shows `e2i_api` and `e2i_frontend` on
   GHCR images (3002:80). `scripts/deploy.sh:25` uses `docker-compose.dev.yml`, `:77` does
   `git reset --hard origin/main`, `:136` `git checkout "$PREV_SHA"` in the shared checkout.
   docker/README.md L94-98 and ONBOARDING L419-420 / L534-538 / L950-951 still present
   `make deploy` as *the* deploy command; ARCHITECTURE §2.2 / ADR-002 / ADR-006 / §9.3 still
   describe `e2i_api_dev`, bind-mount auto-reload and `git pull`. **The architecture auditor's
   proposed replacement ("base + docker-compose.frontend-dev.yml") was itself wrong** — use the
   wording in X3 below.
4. **U5 · ONBOARDING §3 Step 7 "Initialize Database" produces a partial schema.** It applies 1
   of 117 files in `database/migrations/` plus a hand-picked subset of 4 of the 8 schema dirs;
   `scripts/run_migrations.sh` (MIGRATION_DIRS = migrations, memory, core, ml, causal, chat,
   rag, audit) is the real path and is what every deploy runs. `make db-init` is an echo stub.
5. **U11 · docs/runbooks/reviewer-provisioning.md L26 reads a variable that does not exist in
   the container.** `docker exec e2i_api printenv SUPABASE_SERVICE_ROLE_KEY` → rc 1;
   `SUPABASE_SERVICE_KEY` → rc 0 (`docker-compose.yml:31`). Every curl in the runbook sends an
   empty bearer.

Downgraded from P0 by the critic (kept as P1, reasoning recorded under *Critic adjustments*):
README "all services incl. observability run via compose" (X4); #1307 nowcast endpoint and
#1898 discover-effects endpoints "no doc home" (both are OpenAPI-served; see C9).

---

## Cross-cutting facts (X) — one truth, many docs

Each X row is one phase-2 edit applied to several files. Rows in the UPDATE section refer back
to these instead of repeating them.

### X1 · `SUPABASE_POSTGRES_PASSWORD` — **P0** — effort S
- **Truth:** compose `:?`-enforced vars are `REDIS_PASSWORD`, `FALKORDB_PASSWORD`,
  `SUPABASE_POSTGRES_PASSWORD` (base) and `GRAFANA_ADMIN_PASSWORD` (evaluated even without the
  profile — `config` fails on `postgres-exporter` too). The password mirrors
  `/opt/supabase/docker/.env` `POSTGRES_PASSWORD`; compose builds the container-internal
  `SUPABASE_DB_URL` from it (`docker-compose.yml:40`). Required since cd4317fa5 (2026-03-26).
- **Measure:** `grep -o '\${[A-Z_]*:?' docker/docker-compose.yml | sort -u`;
  `grep -c SUPABASE_POSTGRES_PASSWORD .env.example DEPLOYMENT.md docker/README.md docs/ONBOARDING.md` → 0 0 0 0;
  `docker compose --env-file .env.example -f docker/docker-compose.yml config -q` → rc 1.
- **Fix:** add to `.env.example` §2 (`SUPABASE_POSTGRES_PASSWORD=… # [REQUIRED] password of the
  self-hosted supabase-db container (= POSTGRES_PASSWORD in /opt/supabase/docker/.env); compose
  derives the in-network SUPABASE_DB_URL from it and refuses to start without it`); add the row
  to the Required tables in DEPLOYMENT.md, docker/README.md and ONBOARDING §3 Step 2.
- **Targets:** `.env.example`, `DEPLOYMENT.md`, `docker/README.md`, `docs/ONBOARDING.md`.

### X2 · LLM provider truth — **P0** — effort S
- **Truth:** default provider **OpenAI** (`llm_factory.py:112`; tiers `gpt-5.6-luna` fast,
  `gpt-5.6-terra` standard/reasoning); Anthropic is the alternative (`claude-haiku-4-5-20251001`
  / `claude-sonnet-5`). RAG embeddings are OpenAI `text-embedding-3-small` regardless of
  `LLM_PROVIDER` (`src/rag/config.py`), so `OPENAI_API_KEY` is always required.
  `ANTHROPIC_API_KEY` is required with `LLM_PROVIDER=anthropic` and also gates (fail-open) the
  nightly routing-label judge (`src/tasks/routing_label_tasks.py`) and the Layer-4
  adaptive-validity evaluator (`src/data/causal_role_evaluator.py`). **As deployed** (droplet
  `.env`, ADR-010): `LLM_PROVIDER=anthropic` **and** `DSPY_LM_MODEL=openai/gpt-5.6-terra`.
- **Measure:** `grep -n 'LLM_PROVIDER", "openai"' src/utils/llm_factory.py`;
  `grep -n EMBEDDING_MODEL src/rag/config.py`; `grep -n ANTHROPIC_API_KEY src/tasks/routing_label_tasks.py src/data/causal_role_evaluator.py`.
- **Fix:** copy the DEPLOYMENT.md L45 / `.env.example` §3 wording into docker/README.md Required
  table (add `OPENAI_API_KEY`, demote `ANTHROPIC_API_KEY` to optional with the two extra
  consumers named) and ONBOARDING §2 API-keys table; add an "as deployed" callout to
  LLM_CONFIGURATION.md TL;DR and DEPLOYMENT.md (see U2, U8).
- **Targets:** `docker/README.md`, `docs/ONBOARDING.md`, `docker/env.example` (→ U4 decision),
  `DEPLOYMENT.md` L58, `docs/LLM_CONFIGURATION.md`.

### X3 · What production actually runs and how it is deployed — **P0** — effort M
- **Truth (verified live):** production = **base `docker/docker-compose.yml` alone**, no
  overlay. `deploy.yml` `pick_overlay()` (L419-427): `AS production` in the frontend Dockerfile
  → `""`; `docker-compose.frontend-dev.yml` → the #528-A rollback era; `docker-compose.dev.yml`
  → the pre-flip #527 dev-in-prod era. Containers: `e2i_api` (gunicorn ×2 UvicornWorker,
  `read_only: true`, GHCR image tagged by sha, 8000:8000) and `e2i_frontend` (nginx serving the
  built bundle, 3002:80). Local dev = base + `docker-compose.dev.yml` (`e2i_*_dev` names,
  `uvicorn --reload`, Vite HMR on 3002:5173). Deploys happen only by merging to `main`
  (`deploy.yml`); `scripts/deploy.sh` / `make deploy` / `make deploy-build` is the legacy
  local-dev path (dev overlay, no feast/bentoml gates, `git checkout <sha>` rollback) — never run
  it on the droplet; to redeploy, re-run `deploy.yml` (`gh workflow run deploy.yml`).
- **Measure:** `sed -n '419,427p' .github/workflows/deploy.yml`; `grep -n 'AS production'
  docker/frontend/Dockerfile` → 106; `docker ps --format '{{.Names}} {{.Image}} {{.Ports}}' |
  grep -E '^e2i_(api|frontend) '`; `grep -n 'deploy.sh\|make deploy' docs/ONBOARDING.md
  README.md DEPLOYMENT.md docker/README.md`.
- **Fix:** one paragraph (above) placed in DEPLOYMENT.md §Production Deploy, docker/README.md
  L5 intro note, ONBOARDING §8 Deploy Process (+ delete/annotate the three `make deploy` blocks
  L419-420, L534-538, L950-951 and add a "dev overlay names vs deployed names" note to the §5
  container table), ARCHITECTURE §2.2 Container Inventory (replace the two `_dev` rows),
  ADR-002, ADR-006 (list all 7 compose files and which one deploys), §9.3 diagram. Remove
  `make deploy` from docker/README.md Common Commands or annotate it as above. README.md:266
  already frames deploy.sh correctly — keep.
- **Source PRs:** #1317, #1428, #1436, #1480, #1782, #1789, #1792, #1794 (deploy.yml
  evolution), #528 (frontend prod stage).

### X4 · Observability stack is opt-in via the `monitoring` compose profile — **P1** — S
- **Truth:** PR #1806 (2026-08-24) put prometheus, alertmanager, node-exporter,
  postgres-exporter, loki, promtail, grafana behind `profiles: [monitoring]`; a plain `up -d`
  does not start them; `COMPOSE_PROFILES=monitoring docker compose -f docker/docker-compose.yml
  up -d` does; `scripts/health_check.sh` derives its probe set from `docker compose config
  --services` and reports profile-gated services as SKIPPED. `falkordb-browser` is behind
  `debug`; Flower / Redis Commander are `dev-tools` in the dev overlay (not `debug`).
  docker/README.md already documents this (L45-48, L106-115); nothing else does.
- **Measure:** `grep -c -- '- monitoring' docker/docker-compose.yml` → 7;
  `grep -c 'monitoring profile\|COMPOSE_PROFILES' README.md DEPLOYMENT.md docs/ONBOARDING.md .env.example` → 0 0 0 0.
- **Fix:** README L33/L184 (drop "observability" from the always-on list, add the profile
  sentence), DEPLOYMENT.md Service Map (asterisk + "monitoring profile; 127.0.0.1 only" on the
  four rows, Flower → `dev-tools`), ONBOARDING §3 Step 3 / §5 / §8 / §11 URLs, `.env.example`
  §5/§6 (`GRAFANA_ADMIN_PASSWORD` "[REQUIRED only with the monitoring profile]", Flower vars
  "[REQUIRED only with --profile dev-tools]"), ADR-008 August amendment (U9).

### X5 · Operator env-var reference derived from the `x-common-env` whitelist — **P1** — M
- **Truth:** compose forwards **no** `.env` file wholesale; `x-common-env` (docker-compose.yml
  L27-228) is a whitelist of **39** host variables (plus hardcoded `ENVIRONMENT=production`,
  `LOG_LEVEL=INFO`, in-network URLs). A host `.env` value reaches api/workers/scheduler only if
  listed there. **23** of the 39 have no entry in `.env.example`, and DEPLOYMENT.md L64-75 /
  docker/README.md L70-80 still show a 5-row "auto-configured" table. Knobs added in the window
  with no doc home: `CHATBOT_STARTUP_WARM_ENABLED`, `CHATBOT_STARTUP_WARM_LLM_ENABLED` (#1474/
  #1483), `CHATBOT_RAG_LLM_TIMEOUT_S`, `CHATBOT_RAG_DRY_HOP_LIMIT` (#1490), `CHATBOT_RAG_REWRITE_COT`,
  `CHATBOT_RAG_SKIP_EMPTY_DECIDER` (#1522), `CHATBOT_OPT_*` ×5 (#1521), `DSPY_RAG_*` ×4 (#1513),
  `E2I_REQUIRE_FULL_AGENT_REGISTRY` (#1453), `GUNICORN_PRELOAD` (#1589), `NCBI_API_KEY` (#1631),
  `UMLS_UTS_API_KEY`, `OPENFDA_API_KEY` (#1624), `AGENT_COMPUTE_EXECUTOR_WORKERS` + compute-pool
  vars (#1606), `SEGMENT_ANALYSIS_BUDGET_SECONDS` (#1845), `ROUTING_LABEL_*` ×4 (#1342; present
  in .env.example under the wrong heading).
- **Measure:** `sed -n 27,228p docker/docker-compose.yml | grep -o '\${[A-Z_0-9]*' | tr -d '${' | sort -u | wc -l` → 39;
  `comm -23 <(that list) <(grep -oE '^#? ?[A-Z_0-9]+=' .env.example | tr -d '# =' | sort -u) | wc -l` → 23.
- **Fix:** replace both 5-row tables with the whitelist rule + one derived table (var, compose
  default, purpose, owning PR) — generate it from the compose file so it ages well; add a
  "# 12. RUNTIME KNOBS FORWARDED BY COMPOSE" section to `.env.example` (commented, with defaults
  and one-line purpose lifted from the compose comments); put the LLM-lane subset (`DSPY_RAG_*`,
  `CHATBOT_RAG_*`, startup warm, `ROUTING_LABEL_*`) in LLM_CONFIGURATION.md §3/§5 as well; add an
  "Optional external biomedical API credentials (soft-degrade)" table (NCBI / UMLS / OPENFDA:
  absent = lower rate limit or KG `--live` unavailable, never a hard failure).
- **Targets:** `DEPLOYMENT.md`, `docker/README.md`, `.env.example`, `docs/LLM_CONFIGURATION.md`,
  `docs/api/chat.md` §7 (ROUTING_LABEL_*).

### X6 · CHANGELOG stopped at 2026-07 — **P1** — M
- **Truth:** `CHANGELOG.md` last touched 2026-07-19 (ee522194e); sections are `## Unreleased /
  2026-07` and `## Pre-2026-07`; highest PR referenced #1281; the file's own policy is
  month-grouped, PR-train granularity. 232 PRs merged in August and 37 in September have no
  entry. README L129 promises "ongoing change tracking lives in CHANGELOG.md" and L127 is titled
  "Recent Highlights (June–July 2026)", footer L495 "Last Updated: July 2026".
- **Measure:** `git log -1 --format=%cs -- CHANGELOG.md`; `grep -n '^## ' CHANGELOG.md`;
  `git log --merges --first-parent --since=2026-08-01T00:00:00 --until=2026-08-31T23:59:59 --format=%s main | grep -c 'Merge pull request'` → 232.
- **Fix:** add `## 2026-09 (in progress)` and `## 2026-08` sections (proposed trains in
  *CHANGELOG proposed trains* below), append the late-July trains (#1299–#1490 up to 07-31) to
  the 2026-07 section; retitle README L127 to "Recent Highlights (June–September 2026)" with
  5 new bullets (X4, maintenance freshness, causal-discovery UX #1898/#1899, model-performance
  trend #1916, copilot pills #1900–#1919) and set the footer to September 2026 — or retitle the
  section as an archive pointer and let CHANGELOG carry it. Also tighten CHANGELOG L5 (there are
  no platform release tags; the only tags are two `vocabulary-v5.0.0` snapshots and one
  `archive/…` marker) and drop the unverifiable "62 swaps" on L10 (#1276 diff = 52 files).

### X7 · Data dictionary lags migrations 112–132 — **P1** — L
- **Truth:** `docs/data/*` last moved 2026-08-11; highest migration mentioned anywhere in
  docs/data is 118; `database/migrations/` runs to 132. Per-migration checklist (object →
  doc): 112 patient_journeys substrate cols → 02; 113 WS2 `_brand`/`_brand_region` registry
  rows + truth-metric redefinition → 06 (WS2-TR-001/002 formulas are the v1 ones); 114 split
  registry v3.1.0 (documented ✓); 115 `treatment_events.claim_available_date`,
  `adjudication_lag_days` → 02/03; 116 nowcast triangles → 06; 117 cohort_profiler HCP
  KPI-threshold cohorts → 06 note; 118 trigger_effectiveness family (06 ✓); 119
  `validation_status` domain + `enforce_*` trigger → 04 L188, 03 §2; 120 `kpi_query` param cap
  4→6 (06 L132 and L960 still say 4); 121 `validation_outcomes` realign → 03 §2 (table absent);
  122 cohort_profiler windowed → 06 note; 123 `chatbot_messages.computed_user_id` owner-inherit
  trigger → 07; 124/125 ROI temporal band + scoped headline → 06 WS3-BI-010; 126
  `v_kpi_history_coverage` regrained by region → 02 Views; 127 brand_specific `_region` ×6 (+
  twins) → 06 BR-001..005; 128 `business_impact_conversion_rate_brand_region` → 06 WS3-BI-009
  (L1190 says brand×region is refused); 129 `cohort_profiler_hcp_trx_cohort_region` → 06; 130
  `cohort_profiler_hcp_volume_tiers` → 06; 131 `drift_qualifying_features()` RPC → 03 §7; 132
  `persistent_180d` / `discontinued_180d` COMMENTs → 02 patient_journeys rows.
- **Measure:** `ls database/migrations | sort | tail -1` → 132; `grep -rhoE 'migration 1[0-9]{2}'
  docs/data | sort -u | tail -1`; `for t in kpi_history drift_qualifying_features
  hcp_brand_adoption kpi_query_registry validation_outcomes; do grep -c $t docs/data/02-*.md
  docs/data/03-*.md docs/data/06-*.md; done` (all 0 today except one 06 row each for kpi_history
  and kpi_query_registry).
- **Fix:** one dictionary pass over docs/data 02/03/04/06/07 using the checklist; also add the
  tables that predate the window but were never documented (`kpi_query_registry`, `kpi_history`,
  `hcp_brand_adoption`, `territory_metrics`, `npi_taxonomy`, `validation_outcomes`,
  `twin_retraining_jobs`) — see U20/U21/U24/U25.
- **Source PRs:** #1300, #1306, #1307, #1380, #1385, #1396, #1433, #1535, #1537, #1539, #1570,
  #1579, #1696, #1741, #1748, #1894, #1896.

### X8 · Agent roster is 22, Tier 0 = 9 (incl. `cohort_profiler`) — **P1** — S
- **Truth:** `config/agent_config.yaml` has 22 agents; `ml_foundation` tier lists 9; PR #1790
  (2026-08-21) added `cohort_profiler` to the config rosters. README (L17 "22-agent"),
  ARCHITECTURE text (L31/L327), docs/data/00-INDEX are right; stale: ONBOARDING L4 title, L37,
  L274-276 ("TIER 0: ML Foundation (8 agents)"); ARCHITECTURE §3.1 mermaid (21 nodes);
  docs/data/01 L16, 03 L16/L46/L199, 04 L18/L227 ("21-agent"); 07 L96 ("All 21 agent names").
- **Measure:** `python3 -c "import yaml;d=yaml.safe_load(open('config/agent_config.yaml'))['agents'];print(len(d), sum(1 for v in d.values() if v['tier']=='ml_foundation'))"` → `22 9`;
  `grep -rn '21-agent\|21 AI agents\|21 agent' docs/ONBOARDING.md docs/data docs/ARCHITECTURE.md`.
- **Fix:** replace every count with 22 / Tier 0 = 9 and add `cohort_profiler` to the Tier-0
  lists and the §3.1 diagram (dispatched by the orchestrator, not on the sequential SD→OC chain);
  prefer "the roster in `config/agent_config.yaml`" wording where a number is not needed.

### X9 · Feast has 11 feature views / 65 fields, Feast 0.43.0 — **P1** — S
- **Truth:** `feature_repo/features/*.py` define 11 `FeatureView(` (9 domain + `goldstd_cohort_features`
  + `goldstd_hcp_cohort_features`, June 2026) and 65 `Field(`; 7 `PostgreSQLSource(`; the
  sidecar runs `feastdev/feature-server:0.43.0` (`docker/Dockerfile.feast:10`), the SDK is not
  in the API venv. Stale: README L446 (10/48), ONBOARDING L347 (9/48), ARCHITECTURE §4.5 table
  (5 views), docs/data/00-INDEX L21/L146 (9/48), **05-FEATURE-STORE-REFERENCE L3 "Feast 0.58.0",
  L99, L113 "PostgreSQL Sources (5)", L382 "9 feature views, 48 features"** (the canonical doc,
  last touched 2026-02-07 — critic add).
- **Measure:** `grep -hE '^[a-z_]+ = FeatureView\(' feature_repo/features/*.py | wc -l` → 11;
  `grep -ho 'Field(' feature_repo/features/*.py | wc -l` → 65; `grep -n '^FROM' docker/Dockerfile.feast`.
- **Fix:** update 05 first (add a "Gold-standard serving views" section, the two sources, the
  totals, the version, and replace the L440-451 `feature_store.yaml` snippet with the real
  `feature_repo/feature_store.yaml.tmpl` + rendering note), then make README/ONBOARDING/
  ARCHITECTURE/00-INDEX point at it instead of carrying their own counts.

### X10 · Droplet is 8 vCPU / 16 GB, not 32 GB — **P1** — S
- **Truth:** `nproc` → 8; `free -g` total 15 (MemTotal ≈ 15.6 GiB). Stale: ARCHITECTURE L86
  (C4 diagram), L1133 (ADR-002); ONBOARDING L529. The doc's own memory-pressure notes (CLAUDE.md
  mypy 1.6 GiB, worker limits 1.5G/4G/3G) only make sense at 16 GB.
- **Fix:** "8 vCPU / 16 GB RAM" in all three places; ONBOARDING L529 add "runs under memory
  pressure — do not run whole-tree mypy or the full pytest suite on it (CI is the arbiter)".

### X11 · Celery beat schedule has 28 entries, several docs show a 2025-era table — **P1** — M
- **Truth:** `src/workers/celery_app.py` `beat_schedule` has 28 entries (SSOT, guarded by
  `tests/unit/test_workers/test_beat_schedule_registration.py`); the scaffolded `health-check` /
  `cache-cleanup` entries were removed (#897); no DLQ entry; daily entries moved from 86400-s
  intervals to wall-clock crontabs (#1653; e.g. `ab-interim-analysis-check` = 01:15, not "2 AM");
  beat state persists on the `e2i_celerybeat_state` volume; `graph-emptiness-sentinel` every 30
  min (#1761); `routing-label-nightly` 04:30; `chatbot-optimization-drain` 05:30 opt-in;
  `sync-chunk-corpus` nightly (#1374). Stale: ARCHITECTURE §3.5 L390-411 ("15+ periodic tasks"
  + wrong rows); DEPLOYMENT L141/L165-174 (FalkorDB seeding "manual" — now also self-healing).
- **Measure:** `python3 -c "import re;s=open('src/workers/celery_app.py').read();i=s.index('beat_schedule = {');print(len(re.findall(r'^\s{4}\"([a-z0-9_\-]+)\":\s*\{',s[i:],re.M))-1)"` → 28.
- **Fix:** regenerate the ARCHITECTURE table from the dict (name, schedule as written, queue);
  DEPLOYMENT FalkorDB note: "an EMPTY graph self-heals via the sentinel; `seed_falkordb_all.sh`
  for a full manual reseed".

### X12 · Required CI checks are no longer path-filtered — **P1** — S
- **Truth:** `backend-tests.yml` and `tier1-5-test.yml` both carry "DELIBERATELY NOT
  path-filtered — the filtering moved into the `changes` job" (#1444/#1445); the required
  contexts always report. Stale: README L235 ("PR (path-filtered)"), ONBOARDING L387 footgun.
- **Measure:** `grep -c 'DELIBERATELY NOT path-filtered' .github/workflows/tier1-5-test.yml .github/workflows/backend-tests.yml` → 1 1.
- **Fix:** README row → "Every PR (path gating inside a `changes` job so the required check
  never sits Pending) + weekly Monday cron"; ONBOARDING footgun → note that a docs-only /
  frontend-only PR merges normally; `enforce_admins=false` remains as an escape hatch.

### X13 · `make lint` runs whole-tree mypy; `make format` runs black + ruff --fix; the CI formatter gate is `ruff format --check` — **P2** — S
- **Measure:** `sed -n '/^lint:/,/^$/p;/^format:/,/^$/p' Makefile`; `grep -n 'ruff format --check' .github/workflows/backend-tests.yml`.
- **Targets:** README L295, DEPLOYMENT L205-206, ONBOARDING §6 Code Style (Black row). Add the
  CLAUDE.md caveat ("do not run whole-tree mypy on the droplet") and `make generate-types`.

### X14 · 45 KPIs (`config/kpi_definitions.yaml` `summary.total_kpis`) — **P2** — S
- **Targets:** ARCHITECTURE L373 / L1345, ONBOARDING L232 / L734 (all say 44). README and
  docs/data/06 are right. Prefer "the registry in `config/kpi_definitions.yaml`" wording.
- **Measure:** `python3 -c "import yaml;print(yaml.safe_load(open('config/kpi_definitions.yaml'))['summary']['total_kpis'])"` → 45.

### X15 · `health_check.sh` probes are derived, not "24 services" — **P2** — S
- **Truth:** 21 static `check_*` calls today; TOTAL is computed at runtime from `docker compose
  config --services`; profile-gated services SKIPPED; a maintenance-cron freshness probe was
  added (#1805); Opik reports UNHEALTHY by design. **The onboarding auditor's CONFIRMED_OK on
  "24" was wrong** (critic ruling).
- **Measure:** `grep -cE '^\s*check_[a-z_]+ ' scripts/health_check.sh` → 21.
- **Targets:** README L267, ONBOARDING L148 / L720 → age-well wording ("probes every service a
  default `up` starts; profile-gated ones report SKIPPED; Opik UNHEALTHY by design").

### X16 · Hard-coded source line numbers → symbol anchors — **P2** — M (mechanical)
- Docs citing `file.py:NNN` that have drifted: ARCHITECTURE §4.6/§5.1-5.4 (crystallizer.py:102
  → class at 105, types.py:407 → 450, …), docs/runbooks/sentinels.md (~15 citations, registry.py
  grew to 819 lines), docs/api/crystal_digests.md (§1-§3, §7, §8), docs/model_success_criteria.md
  (L33/46/56/74/90/112/113/119/159), docs/roi_methodology.md (L27/86/188/259/262),
  docs/synthetic_v3_design.md (cross-file refs; the regime file itself is unchanged and its refs
  hold), docs/data/SYNTHETIC-CAUSAL-DATA-GUIDE.md (`_PJ_COHORTS` :259 → 328).
- **Fix:** cite symbols (`` `Crystallizer` in src/memory/crystallization/crystallizer.py ``) — the
  convention docs/api/chat.md already uses. Where a number must stay, regenerate with `grep -n`.

### X17 · Pointers to `.claude/plans/*` files that are archived or absent — **P2** — S
- ARCHITECTURE L639-641/L692-693 (layer4 evaluator plans, memory subsystems plan → all under
  `.claude/plans/archive/…`), docs/runbooks/sentinels.md L7/L825 and docs/api/crystal_digests.md
  L7/L606 (memory subsystems plan → archived 2026-05-20), docs/OPTUM_CONVERSION.md L5-6/L564-567
  (`csu-rwd-analyst-spec.md`, `optum-rwd-ingestion.md` absent), docs/OPTUM_MART_CONVERSION.md
  L657, docs/synthetic_v3_design.md L6-8 (`adaptive_temporal_validity_redesign.md` absent).
- **Fix:** point at the archived path and label "(local, untracked planning note; historical)" or
  replace with the in-repo artifact (calibration reports, `data_dictionary.csv`).

---

## CREATE — new docs (C1–C11)

### C1 · `docs/runbooks/maintenance-cron.md` — **P1** — effort M
**Why:** the whole `scripts/maintenance/` layer (setup_cron.sh, cleanup_orphans.sh,
memory_monitor.sh, docker_cleanup.sh, check_maintenance_freshness.sh — #1798/#1799/#1802/#1803)
plus the live `/etc/cron.d/e2i-maintenance` and the `Maintenance Freshness` workflow
(`maintenance-freshness.yml`, daily 07:30 UTC + dispatch, files/updates a tracking issue on
staleness, #1807/#1810) are documented only in script and workflow comments.
`grep -rln 'check_maintenance_freshness\|setup_cron\|cron.d/e2i' docs/ README.md DEPLOYMENT.md docker/README.md` → 0.
**Add:** what runs and when (derive from `setup_cron.sh:48-70`: orphans */15, memory monitor */5
with `--auto-cleanup`, log truncation 02:00, docker cleanup Sun 03:00), logs under `/var/log/e2i`,
success stamps `.<script>.success`; install/repair (`sudo scripts/maintenance/setup_cron.sh`);
how freshness is judged (interval × tolerance from the crontab, `health_check.sh` section);
what the workflow does on red and the live-cert recipe in its header; cross-link from README
Operational Scripts and CI table (U1). **Source PRs:** #1798, #1799, #1800, #1802, #1803, #1805,
#1807, #1810.

### C2 · `docs/runbooks/deploy-operations.md` (or a "Deploy operations" section in DEPLOYMENT.md) — **P1** — effort M
**Why:** operator procedures that exist only in `deploy.yml` comments and session memory: (a)
how the deploy picks its sha — newest `origin/main` ancestor (30-commit walk) that has BOTH GHCR
images, downgrade floor, so prod can lag `origin/main` by a commit (#1431/#1436); (b)
`ensure-main-image` job and the published-image assertion — a missing image FAILS before
anything is migrated or flipped, "Recover: gh workflow run deploy.yml" (#1780/#1782/#1785); (c)
the converged-but-FAILED case — verify with container content markers, not the job conclusion
(#1780); (d) the image-drift gate and `scripts/deploy/image_drift_allowlist.json` (#1479/#1480);
(e) abort conditions: dirty tracked files, `main` held by another worktree (`git worktree
prune`, #1787), and the shared checkout being re-pointed to `main`; (f) rollback anchors on the
RUNNING `e2i_api` image sha, not the checkout (#1780); (g) prune is a separate best-effort step
(#1784); (h) restart side-effects: an in-flight discover-effects run is read-repaired to `failed`
on the next poll (heartbeat 15 s, TTL 120 s, #1899); (i) concurrency group
`deploy-production`, never cancel-in-progress (#1412/#1428).
**Measure:** `grep -nE 'newest BUILT ancestor|Deploy FAILED before any change|check_image_drift.py|reattach_to_main|concurrency:' .github/workflows/deploy.yml`.

### C3 · `docs/runbooks/synthetic_reseed.md` — **P1** — effort M
**Why:** the Monday 03:00 reseed chain — `scripts/reseed_synthetic.sh` (`--append-frontier`
default, `--full` destructive recovery, `--skip-retrain`, stage-resilient staging via
`scripts/lib/reseed_stages.sh`, #1578) → `scripts/retrain_goldstd.sh` → champion re-promotion
(`scripts/promote_hcp_adoption_champions.py --execute`; on failure chat propensity fails closed,
#1692) → plus the one-off/manual data jobs deploy.yml never runs (`scripts/rag/ingest_chunk_corpus.py`
#1374, causal-path seed/backfill scripts #1325, `seed_falkordb_all.sh`, `load_synthetic_data.py
--only-tables` #1387) — has no doc home. docs/SYNTHETIC_DATA.md L621-626 shows one bare
invocation. **Anchor** from DEPLOYMENT.md L141-142 "Not part of the CI deploy".
**Measure:** `grep -rln 'reseed_synthetic\|retrain_goldstd\|promote_hcp_adoption' docs/ README.md DEPLOYMENT.md` → 0.

### C4–C8 · ADR-013 … ADR-017 in `docs/decisions/` — **P1** — effort M each (C8 L)
Standalone records for decisions shipped since ADR-012 (07-19). Bodies drafted by the auditor
from PR bodies; verify each against the code at write time.
- **C4 · ADR-013 model-performance trend is classified against sampling noise, never a fixed ±5%
  fold rule** — #1916; `src/services/performance_trend_stats.py` (analytic SE for AUC/accuracy/
  recall, t-adjusted empirical spread k≥6, z 2.5 + slope t, 5% materiality, open-month skip,
  `alert_threshold` is a floor not a boundary).
- **C5 · ADR-014 copilot suggestion pills come from a code-derived capability catalog and a
  deterministic validator** — #1900 + #1903/#1908–#1910/#1912/#1913/#1915/#1919;
  `src/services/chat_capability_catalog.py`; live kept-NO rate 42% → ~2%.
- **C6 · ADR-015 an agent run has failed only when a node left a `<node>_error` audit row;
  `validation_passed` is a scientific verdict** — #1902; `src/api/utils/audit_outcomes.py`.
- **C7 · ADR-016 long causal-discovery jobs are cancelled cooperatively (job-store marker) and
  orphan-detected by heartbeat + poll-time read-repair, not signals or startup sweeps** — #1898,
  #1899; `DurableJobStore.set_marker/has_marker`, `_repair_if_orphaned` in `src/api/routes/causal.py`.
- **C8 · ADR-017 discovery corroboration is gated on bootstrap edge stability for
  single-algorithm runs; latent-confounding (FCI) and refutation warnings annotate, never gate;
  a prior-asserted DAG is never reported as discovered** — #1869, #1871, #1879, #1883, #1886,
  #1887, #1888, #1891 (one record, multi-PR arc 08-31 → 09-04).
- Lower-grade candidates (P2, record so the omission is deliberate): ADR for the RAGAS posture
  (fixture eval = manual-only judge-drift sentinel, real-pipeline gate fails loud — #1491–#1494,
  origin #504) and an "Operational decisions (no ADR)" list in decisions/README.md (#1445 CI
  never path-filters a required context; #1589 GUNICORN_PRELOAD on; #1606 bounded compute pools;
  #1852 mutations never retry; #1759 FalkorDB `/data`; #1653 beat crontabs; #1731 allowlist).
**Measure:** `ls docs/decisions/adr-*.md | tail -1` → adr-012; `ls src/services/performance_trend_stats.py src/services/chat_capability_catalog.py src/api/utils/audit_outcomes.py` all exist.

### C9 · DECISION: `docs/api/causal.md` + `docs/api/segments.md` — yes or no — **P1** — effort L if yes, S if no
**Question:** eleven sweep-4 rows (#1879, #1883, #1886, #1887, #1889, #1890, #1891, #1892,
#1893, #1898, #1899) plus #1873 (propose-questions fails closed) and #1307 (nowcast) each propose
a hand-written causal/segments API reference. Unlike the CopilotKit routes (`include_in_schema=False`,
which is why `docs/api/chat.md` exists), every causal/segments route carries a summary and is
served at `/api/docs`, and the response fields (`dag_source`, `edge_provenance`, `discovery_guided`,
`library_agreement_score`, `definitions`, bootstrap stability) are regenerated into
`frontend/src/types/generated/api.ts` under the verify-types gate.
**Recommendation (critic + this ledger): (b) no new per-field reference.** Instead: (1) widen
the ARCHITECTURE §3.4 causal / segments rows with the job lifecycle (submit → poll → cancel,
heartbeat read-repair) and the 400 conditions (#1890 covariate-role, #1873 patient-joined columns);
(2) put the semantics that are policy, not schema, on the in-app "How E2I Works" page (already
carries the FCI/latent policy, #1888) and in ADR-017 (C8); (3) record the endpoints in the
CHANGELOG (X6). Choose (a) only if the team wants prose contracts for `POST /api/causal/discover-effects`,
`GET …/questions`, `POST …/{job_id}/cancel`, `POST /api/causal/agent-analyze`,
`GET /api/segments/datasets`, `POST /api/segments/analyze` beyond OpenAPI — then one file each,
verified-against-code like chat.md. **Fact that would reverse the recommendation:** any of these
routes becoming `include_in_schema=False`, or the OpenAPI summaries being removed.
**Measure:** `grep -n 'summary=' src/api/routes/causal.py | grep -c discover-effects`;
`grep -n include_in_schema src/api/routes/causal.py src/api/routes/segments.py | wc -l` → 0.

### C10 · `docs/api/README.md` (5 lines) — **P2** — S
State that `chat.md` and `crystal_digests.md` are hand-written references for surfaces OpenAPI
does not describe, and that `openapi.json` / `index.html` in the same directory are `make api-docs`
outputs (openapi.json gitignored at .gitignore:223; index.html should be added to .gitignore).
Also fixes README L119 ("docs/api — OpenAPI spec, not tracked": two files ARE tracked).

### C11 · Error-envelope reference (paragraph in ARCHITECTURE §3.4 or `docs/api/errors.md`) — **P2** — S
`src/api/main.py` `_e2i_http_error` / `_e2i_404_error` (#1831/#1832): shape
`{error: {error_id, category, severity, message, details}}`, `ErrorCategory` values, status →
category mapping (400/422 validation, 409 conflict, 404 not_found keeping the client-facing detail).

---

## UPDATE — existing docs (U1–U34)

Format per bullet: **priority · section (lines)** — claim vs truth. *Fix.* `measure` (source PRs).
Rows marked → Xn are executed by that cross-cutting item.

### U1 · `README.md` — **P1 bundle** — effort M (last touch 2026-08-05, 4de24afda; 234 PRs merged since)
- **P1 · L184/L198/L33** observability always-on → X4.
- **P1 · L235** tier1-5 "PR (path-filtered)" → X12.
- **P1 · L229-242 CI table** names 8 of 23 workflows; absent entirely: `maintenance-freshness.yml`
  (#1807), `feast-apply.yml`, `benchmarks.yml`, `ragas-smoke.yml`, `g3_wiring_guard.yml`,
  `tier1b_b2_diagnostic.yml`, `tier1b_b2_experiment.yml`, `methodology-signoff-validator.yml`;
  `slow-tests.yml` now nightly 05:00 with the upstream-transient rolling issue (#1816/#1823).
  *Fix:* add rows for Maintenance Freshness and Slow Tests (nightly), extend the L242 sentence.
  `ls .github/workflows/*.yml | xargs -n1 basename | sort | comm -23 - <(grep -oE '[a-z0-9_-]+\.yml' README.md | sort -u)`.
- **P1 · L127/L129/L495** Recent Highlights (June–July), CHANGELOG pointer, footer → X6.
- **P1 · L331** integration snippet `app.include_router(explain_router, prefix="/api/v1")` — the
  app mounts under `/api` (`src/api/main.py:1166`); README L397 itself uses `/api/admin/...`.
  *Fix:* `prefix="/api"` + comment "public API base is /api, not /api/v1". `grep -n 'include_router(explain_router' src/api/main.py`.
- **P1 · L92-110 src/ tree** omits `services`, `kpi`, `insights`, `tasks`, `repositories`,
  `etl`, `lifecycle`, `ontology`, `security`, `skills`, `testing`, `causal`, `data`, `optimization`
  (27 packages exist; README L396 itself cites `src/services/llm_pricing.py`). *Fix:* add the
  missing lines with one-line purposes. `comm -23 <(ls -d src/*/ | sed 's|src/||;s|/||' | grep -v __pycache__ | sort) <(grep -oE '│ ├── [a-z_]+/' README.md | sed 's/.*── //;s|/||' | sort)`.
- **P1 · L261-286 Operational Scripts** — all 15 named scripts exist; missing operator-facing
  ones: the `scripts/maintenance/` group (→ C1), `promote_hcp_adoption_champions.py` (#1384),
  `deploy/check_image_drift.py` (#1480), `rag/ingest_chunk_corpus.py` (#1374),
  `run_real_pipeline_ragas.py` (#1492), `gen_kpi_catalog.py` (#1383), `reseed_synthetic.sh`
  modes (→ C3). `git log --diff-filter=A --since=2026-07-20 --name-only --format= -- scripts/ | sort -u`.
- **P2 · L267** health_check "24 services" → X15.
- **P2 · L89** `kg_cache/ # Knowledge-graph cache (git-ignored)` — the Layer-2 cache
  `1cdaa038__96bfd2e0.json` is committed (.gitignore:250 un-ignores it) and COPYed into the API
  image; `data/kg_cache/**` is a deploy trigger (#1619/#1621/#1783). *Fix:* "committed Layer-2 KG
  cache (packaged into the API image; rebuild via docs/runbooks/kg_cache.md)". `git ls-files data/kg_cache`.
- **P2 · L112** "1,400+ test files (unit, integration, tier0-5)" — 1,809 `test_*.py` today and
  no tier dirs under tests/. *Fix:* age-well list of the real top-level dirs. `find tests -name 'test_*.py' | wc -l`.
- **P2 · L99** "11 more agents" → 12 (2 Tier-0 cohort agents + 10). `ls -d src/agents/*/ | grep -vE 'base|orchestrator|tool_composer|experiment_designer|ml_foundation|tier_0|__pycache__' | wc -l`.
- **P2 · L311** "5 REST endpoints (/predict, /batch, /history, /models, /health)" → 7 under
  `/api/explain` (`/predict`, `/predict/batch`, `/history/{patient_id}`, `/models`,
  `/sample-entities`, `/global`, `/health`). `grep -A1 -E '@router\.(get|post)\(' src/api/routes/explain.py | grep -oE '"/[^"]*"'`.
- **P2 · L446** Feast 10/48 → X9. **P2 · L441/L444** table names `semantic_cache` →
  `semantic_memory_cache`, `verification_log` → `audit_chain_verification_log`. **P2 · L439/L457**
  core-table count (19 vs 20 CREATE TABLE incl. `agent_tier_mapping`) — UNCERTAIN; prefer
  age-well wording and reconcile with docs/data/02's owner.
- **P2 · L119** `docs/api/ # OpenAPI spec (auto-generated, not tracked)` → see C10.
- **P2 · L13** badge alt text "Type Check" points at `verify-types.yml` (OpenAPI types); the
  mypy gate is the `Type Check (MyPy)` job in backend-tests.yml. *Fix:* relabel "OpenAPI Types".
- **P2 · L295** `make lint` note + `make generate-types` → X13.
- **P2 · L466** only the migrations runbook is linked; six others exist. *Fix:* link `docs/runbooks/`
  with a one-line list (+ the three new runbooks C1–C3).
- **P2 · L172** "Domain Vocabulary: Enhanced with Tool Composer ENUMs and routing patterns" — the
  routing-pattern enum lives in `src/agents/orchestrator/classifier/schemas.py`, not the vocab yaml.
- **P2 · L420-427 Causal Validation** — add the three-state per-test verdict
  (passed/warning/failed + skipped_tests with reason) shipped in #1869; gate semantics unchanged.
- **P2 · L237 Security row** add "npm audit" to the tool list; **P2 ·** add a one-line pointer to
  the in-app "How E2I Works" page at `/documentation` (#1824) under the dashboard feature bullet.
- **CONFIRMED_OK (no action):** agent count 22 / Tier-0 7 in ml_foundation + 2, 27 config
  files, 45 KPIs, 140+ tables (146 CREATE TABLE), 31 pages, all 46 tree paths + 35 links resolve,
  Quick Start migration commands, other CI rows, 43 test batches, import snippets, tech stack
  versions, LLM provider paragraph L390 (optionally add the as-deployed note, X2).

### U2 · `DEPLOYMENT.md` — **P0/P1 bundle** — effort M (last touch 2026-07-18; deploy.yml changed 8× since, compose 20×)
- **P0 · L43-52 Required table** → X1. **P1 · L8-33 Prerequisites/Quick Start** — a running
  self-hosted Supabase stack is required first: compose joins the external `supabase-network`
  and the `supabase-db` container (`docker-compose.yml:364-366`); `up` fails on a fresh machine
  without `docker/supabase/start.sh`. *Fix:* add as step 0. `grep -n -A2 'supabase-network:' docker/docker-compose.yml | grep external`.
- **P1 · L64-75 Auto-configured table** → X5. **P1 · L78-98 Service Map** → X4 (+ Flower row
  says `debug`, is `dev-tools`; Redis 6382 / FalkorDB 6381 rows lack "127.0.0.1 only" —
  `grep -n '127.0.0.1:638[12]' docker/docker-compose.yml`).
- **P1 · L107-110 Trigger** — deploy.yml `on.push.paths` now also lists `docker/Dockerfile`,
  `docker/frontend/**`, `scripts/deploy/**` (#1479), **all of `scripts/**`** (#1783 — a
  benchmark/demo script change deploys, by design) and `data/kg_cache/**`; `database/**` is
  deliberately NOT a trigger. `sed -n '/^ *paths:/,/workflow_dispatch/p' .github/workflows/deploy.yml | grep -E "^\s+- '"`.
- **P1 · L111 Pipeline** — add `ensure-main-image` between the builds and `deploy` (#1780/#1782);
  deploys are serialized under `concurrency: deploy-production`, never cancel-in-progress, all
  jobs timeout-bounded (#1412/#1428). `grep -nE '^ [a-z-]+:$|concurrency:' .github/workflows/deploy.yml`.
- **P1 · L114-116 item 1** "a local build happens only as a fallback when the GHCR pull fails" —
  now the deploy REFUSES the on-box build and fails before touching anything ("Refusing the
  droplet local-build path (#1785)… Recover: gh workflow run deploy.yml"). `grep -n 'Refusing the droplet local-build path' .github/workflows/deploy.yml`.
- **P1 · L117-120 item 2 Hard sync** — target is the newest `origin/main` ancestor with both
  GHCR images (30-commit walk, downgrade floor vs the RUNNING image sha), HEAD is re-attached to
  `main` if a branch was left checked out, and the deploy aborts if `main` is held by another
  worktree (#1431/#1780/#1787). `grep -n 'select_built_sha\|reattach_to_main\|running_image_sha()' .github/workflows/deploy.yml | head`.
- **P1 · L102-142 MISSING item 7 image-drift gate** (#1479): `scripts/deploy/check_image_drift.py`
  compares every compose-pinned service's running image to the pin; mismatch not in
  `image_drift_allowlist.json` FAILS the run with NO rollback (fix = recreate the sidecar deliberately).
- **P1 · L168-170 FalkorDB seeding / deploy.sh sentence** → X3. **P2 · L141/L170** add the
  emptiness sentinel self-heal (X11) and anchor C3 here.
- **P2 · L135-137 item 5 rollback** — target is the sha the running `e2i_api` image reports, not
  the checkout HEAD (#1780); falls back to `--build` only if the old image cannot be pulled.
- **P2 · L138-139 item 6 prune** — separate best-effort step, 15-min budget; image prune only on
  a successful rollout (#1784).
- **P2 · L58** `ANTHROPIC_API_KEY` "only needed with LLM_PROVIDER=anthropic" → X2 wording.
  **P2 · after L45/L59** add the as-deployed note (X2). **P2 · L205-206** → X13.
- **P2 · L308-318 File Reference** — `docker/nginx/nginx.conf` has no consumer; the frontend image
  bakes `docker/frontend/nginx.conf`; the host site is `docker/nginx/host-nginx.conf`; add rows
  for the other compose files (7 exist: base, dev, frontend-dev [legacy], monitoring [superseded,
  N3], opik, rxnav [stub], secure [excluded]). `grep -rn 'nginx.conf' docker/frontend/Dockerfile docker/docker-compose*.yml`.
- **P2 · new "Persistent volumes" table** (#1759): `e2i_falkordb_data` mounted at `/data` with
  `FALKORDB_DATA_PATH=/data` (without it RDB persistence was a no-op), redis, mlflow, feast
  registry, `e2i_celerybeat_state`.
- **P2 · new "Optional tuning" rows** → X5 (`SEGMENT_ANALYSIS_BUDGET_SECONDS` default 900 —
  #1845; compute pools — #1606; `GUNICORN_PRELOAD` kill switch + troubleshooting note "worker
  SIGABRT mid-stream / stream tears on cold workers" — #1589; `E2I_REQUIRE_FULL_AGENT_REGISTRY`
  as a one-boot post-deploy verification, cross-link `/api/cognitive/status` 503 `gate_failed` — #1453/#1466).
- **P2 · L239-247 Opik** — optionally note `OPIK_ENABLED=false` must stay in the droplet `.env`
  (compose default is true) so tracers skip client construction.
- **CONFIRMED_OK:** L121-134 migrations-unconditional + feast/health/bentoml gates (note the
  migration step now runs after the image assertion), L176-184 symlink, L189-190 pytest addopts,
  Opik overlay 10 services, port/tunnel notes.

### U3 · `docker/README.md` — **P0/P1 bundle** — effort S (last touch 2026-08-24)
- **P1 · L5** "both local development and the production droplet use the same setup (base + dev
  overlay)" → X3. **P0 · L57-68 Required table** → X1 + X2. **P0 · L94-98 `make deploy`** → X3.
- **P1 · L13-31 Quick Start** — add step 0 (self-hosted Supabase up, external network) and the
  password (X1). **P1 · L70-80 Auto-configured** → X5.
- **P2 · L49** Flower "debug" → `dev-tools`; Redis/FalkorDB rows add "127.0.0.1 only".
  **P2 · L101** `docker exec -it e2i_api_dev bash` — dev-overlay name; on the droplet `e2i_api`.
  **P2 · L161** `nginx/nginx.conf` row → the two real files (see U2).
- **CONFIRMED_OK:** L35-53 services + L106-115 monitoring profile section (this doc is the model
  for X4).

### U4 · `.env.example` (+ decision on `docker/env.example`) — **P0/P1 bundle** — effort S
- **P0 · §2** missing `SUPABASE_POSTGRES_PASSWORD` → X1.
- **P1 · L25** `SUPABASE_DB_URL=postgresql://postgres:password@localhost:54322/postgres` — this
  stack publishes Postgres on host **5433** (`docker port supabase-db` → 5432/tcp → 0.0.0.0:5433;
  5432 is the Supavisor pooler); the var is NOT forwarded into containers (compose derives the
  in-network URL). *Fix:* port 5433 + "host-side tools only (run_migrations.sh psql mode)".
- **P1 · L38** `# ANTHROPIC_MODEL=claude-sonnet-5 # Optional: Anthropic model for DSPy/chat paths`
  — deliberately NOT forwarded into containers (compose comment); host-side runs only; inside
  Docker the in-code default governs. `grep -c 'ANTHROPIC_MODEL:' docker/docker-compose.yml` → 0.
- **P1 · whole file** 23 compose-forwarded knobs missing → X5 (new §12).
- **P2 · L7 header** "[REQUIRED] … Compose will refuse to start without them" — compose hard-fails
  only on REDIS/FALKORDB/SUPABASE_POSTGRES (base) + GRAFANA (monitoring) + FLOWER_* (dev-tools);
  the rest fail at runtime. **P2 · L24** `SUPABASE_JWT_SECRET [REQUIRED]` — optional/unused
  (`src/api/dependencies/auth.py:86-108` verifies via `get_user()`), not forwarded. **P2 · L60**
  `# OPIK_ENABLED=false` — in-code default is ON; production relies on `false`; uncomment and say
  "opt-OUT switch". **P2 · §5/§6** profile-gated [REQUIRED] markers → X4. **P2 · L82-84**
  `ENVIRONMENT`/`LOG_LEVEL` are hardcoded `production`/`INFO` inside api/workers (host value only
  feeds `config-check`); `CORS_ORIGINS` has no reader. **P2 · L61-65** `ORCHESTRATOR_CLASSIFIER_MODE`
  / `ROUTING_LABEL_*` sit under the OPIK heading — move to a "4b. ORCHESTRATOR ROUTING" section.
- **N1 (code) + doc caveat · §10/§11** `ADAPTIVE_CRITERIA`, `ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`,
  `ADAPTIVE_VALIDITY_EVALUATOR_MODEL` are read by code but NOT forwarded by compose (0 hits), so
  the documented "ROLLBACK switch" is inert in containers. Until N1 lands, add the caveat.
- **DECISION · `docker/env.example`** (P1): a second, older template (last substantive touch
  2026-05-15) with `ANTHROPIC_API_KEY` as the only LLM key, `OPIK_VERSION=latest` (overrides the
  compose 1.10.8 pin), dead `BENTOML_URL`/`FEAST_URL` (should be `BENTOML_SERVICE_URL`, port 6567)
  and `DEBUGPY_*` nobody reads. Only reference: `Makefile:47` names it in a hint. *Recommend:*
  move the `SUPABASE_POSTGRES_PASSWORD` + 5433 notes into the root file, then reduce
  `docker/env.example` to a 3-line pointer ("Superseded — use ../.env.example; docker/.env is a
  symlink") and fix `Makefile:47`; deleting it is equally acceptable. Not four per-line edits.
  `grep -rn 'docker/env.example' --include='*.md' --include='Makefile' --include='*.sh' --include='*.yml' . | grep -v docs/Archive`.
- **CONFIRMED_OK:** §3 LLM block (optionally add the as-deployed note), §7 HSTS/rate-limit lines
  (add "host-side only"), §9 admin creds consumers, VITE_* note, routing knob values.

### U5 · `docs/ONBOARDING.md` — **P0/P1 bundle** — effort L (last touch 2026-08-05; footer says 2026-02-07)
- **P0 · §2 L73-77 API keys** → X2. **P0 · §3 Step 7 L169-190** → replace the psql list with
  `./scripts/run_migrations.sh` (applies every pending file under the 8 schema dirs, ledger
  `public.schema_migrations`, idempotent; `--dry-run`; docker-exec mode on the droplet) and note
  `make db-init` is a stub. `ls database/migrations | wc -l` → 117; `sed -n '50,58p' scripts/run_migrations.sh`.
- **P1 · L4 / L37 / L274-276** 21 agents / Tier 0 = 8 → X8. **P1 · §3 Step 2 L96-108** → X1.
  **P1 · L387 footgun** → X12. **P1 · §3 Step 3 / §5 / §8 / §11 observability URLs** → X4.
  **P1 · L529** 32 GB → X10. **P1 · §8 L531-555 Deploy Process + L419-420 / L534-538 / L950-951
  `make deploy`** → X3 (the critic found the auditor cited only one of the three copies).
- **P1 · L466 / L518-521 coverage** "fail_under=70%" and a 70/70/70/70 backend row —
  `pyproject.toml` `fail_under = 20` (re-baselined 2026-04-28; 70 is aspirational; single
  line-coverage gate). Frontend row 62/55/54/62 matches vitest.config.ts. `grep -n '^fail_under' pyproject.toml`.
- **P1 · L692 / L907** router example `APIRouter(prefix="/api/v1/<domain>")` — routers are mounted
  under `/api` by `main.py` (28 include_router calls; only rag declares `/api/v1/rag`). *Fix:*
  `prefix="/<domain>"` + "mounted under /api". `grep -rhoE 'prefix="/api[^"]*"' src/api/routes/ src/api/main.py | sort | uniq -c`.
- **P1 · L395 / L913** "MyPy non-blocking" — CI enforces `MYPY_CEILING=61` inside the required
  `Backend CI Success` gate. `grep -n 'MYPY_CEILING=' .github/workflows/backend-tests.yml`.
- **P1 · §5 L321-326 worker table** — worker_light concurrency 2 (not 4), queues
  default/quick/api; worker_medium concurrency 2, queues analytics/reports/aggregations;
  worker_heavy replicas 0 (on-demand), concurrency 1, queues shap/causal/ml/twins, time-limit
  3600; limits 1.5G / 4G / 3G. `grep -nE 'queues=|concurrency=' docker/docker-compose.yml`.
- **P1 · L585** "Tokens validated against `SUPABASE_JWT_SECRET`" — verification is
  `client.auth.get_user()` (URL + anon key); the secret is unused on this path; testing-mode
  bypass only when `ENVIRONMENT != production`. `sed -n '85,91p' src/api/dependencies/auth.py`.
- **P1 · L154-158 Step 5** `pip install -e ".[dev]"` alone resolves unpinned tool versions; CI
  and `make dev-install` install `requirements.txt` first. **P1 · L924-925 FAQ generate types**
  — `make generate-types` (static OpenAPI export, what CI's verify-types gate runs); commit the
  regenerated `api.ts` in the same PR.
- **P2:** L372-379 add `detect-secrets` hook; L393-394 Ruff/Black rows → X13; L498 markers list
  is 9 of 14 (`real_data`, `benchmark`, `live_llm`, `live_lm`, `real_supabase`); L608-616 rate
  table add `calculate` 30/60 s and `copilotkit_status` 100/60 s; L232/L734 44 KPIs → X14; L347
  Feast → X9; L346 FalkorDB "8 node types, 15 edge types" — the memory schema cypher declares 12
  labels / 18 relationship types (ontology yaml is 8/15 — say which is meant); tree L236
  (pages live in `pages/`, 31), L250 ("37+ tables" contradicts L339 "140+"), L259 ("36+ scripts"
  → 140 files); L195-196 `# Or: python src/ml/data_generator.py` omits the loader step (README
  L221-222 same); L279 classifier wording (staged pipeline under `orchestrator/classifier/`, Haiku
  or gpt-4o-mini fast fallback); L784 "Supabase — Project member" (self-hosted; Studio via
  tunnel 3001); L754 label is "good first issue"; L963 footer date; L148/L720 → X15; §7 add
  "tests/unit runs with dead Supabase creds (127.0.0.1:1) so unit tests can never touch prod"
  (#1421) and the stall watchdog / xdist first-crash fast-fail (`E2I_PYTEST_STALL_TIMEOUT`, #1660).
- **CONFIRMED_OK:** PR requirements (two required checks, 0 approvals), §5 dev-overlay container
  table (add the X3 dev-vs-deployed note), RBAC/security headers, read_only set, MLflow auth_basic
  (live host has it — see Verification), testing statics, Tri-memory TTLs.

### U6 · `docs/ARCHITECTURE.md` — **P0/P1 bundle** — effort L (last touch 2026-08-04 — an mlflow pin bump; most drift predates it)
- **P0 · §2.2 L131-137 Container Inventory** → X3 (use the deploy-docker wording; the auditor's
  own replacement was wrong). **P1 · ADR-006 L1209-1212, ADR-002 L1131-1139, §9.3 L1290-1318** → X3.
- **P1 · §3.1 mermaid L216-273** 21 nodes → X8. **P1 · §2.1 L88 / ADR-002 L1133** 32 GB → X10.
  **P1 · §3.5 L362-411 beat table + diagram** → X11 (+ heavy worker "16 CPU, 32 GB" box → 2 CPU /
  3 GB, replicas 0). **P1 · §2.2 L135-136** worker_light 2 GB → 1.5 GB, worker_medium 8 GB → 4 GB.
  `python3 -c "import yaml;d=yaml.safe_load(open('docker/docker-compose.yml'))['services'];[print(k,d[k]['deploy'].get('replicas'),d[k]['deploy']['resources']['limits']) for k in ('worker_light','worker_medium','worker_heavy')]"`.
- **P1 · §3.4 L350-358 + §6.1 L942-950 middleware** list has 7; `main.py` adds 8
  (`InsightVerifierMiddleware`, `ActivityTrackingMiddleware` missing) + OpenTelemetry ASGI.
  `grep -c 'app.add_middleware(' src/api/main.py` → 8. L91 label "6 middleware layers" → 8.
- **P1 · §3.4 L332 copilotkit row** — three distinct chat surfaces: orchestrator brain
  (`POST /api/copilotkit/chat/stream`, `/chat`) with the #1336 bridge fallback to the AG-UI brain
  (`chat_bridge.py`), the AG-UI runtime (`/api/copilotkit/{path}`), and `/api/chat/suggestions`.
  *Fix:* split the row; cite docs/api/chat.md.
- **P1 · §3.2 L304-319 timeouts** causal_effect 120 s → 300 s, experiment_design 150 → 240,
  system_health 5 → 20 (#1419/#1353); also §3.1 labels and §9.5 L1361. `grep -nE 'timeout_ms=' src/agents/orchestrator/nodes/router.py | head -14`.
- **P1 · §4.3 L505-527 KG schema** "8 node types … 11 relationship types (PRACTICES_IN, ANALYZES)"
  — `E2IEntityType` has 10 (adds EPISODE, COMMUNITY); `E2IRelationshipType` 11 but MENTIONS /
  MEMBER_OF / RELATES_TO replace PRACTICES_IN / ANALYZES; also §4.1 L444, §5.4 L855.
- **P1 · §6.1 L937** "Rate limiting (100 req/s API)" — zones in `/etc/nginx/nginx.conf:48-50`:
  `api_limit` **10 r/s** (burst 20, `/api/`), `copilot_limit` 30 r/s (burst 10), `general_limit`
  100 r/s (burst 50); applied in `docker/nginx/host-nginx.conf:104/128/292`. (Auditor cited the
  zone lines in the wrong file; values verified live.)
- **P1 · MISSING §3.0 source package map** — no table of the 27 `src/` packages; several are
  load-bearing to claims elsewhere (`services`, `kpi`, `insights`, `tasks`, `repositories`,
  `causal` vs `causal_engine` — say which is canonical). *Fix:* one table `src/<pkg>` →
  responsibility → entry points.
- **P1 · ADR-008 amendment (August 2026)** → X4 (+ maintenance-freshness alarm #1807, "an
  unmanaged service is not an outage" #1806).
- **P1 · new sections for shipped surfaces with no doc home:** clinical-context service
  (`src/services/clinical_context/`, `GET /api/causal/clinical-context`, brand_map, Europe PMC /
  Open Targets providers — #1764); strategic insights row (`/api/insights/…` incl.
  `POST /insights/clinical-narrative` — #1812); feedback-learner optimizer gate
  (`/api/feedback/health` `optimizer`, supply-based, `GET /api/feedback/patterns` 30-day default
  `max_age_days` — #1677, #1902); RAG chunk corpus (`scripts/rag/ingest_chunk_corpus.py` one-off
  + nightly `sync-chunk-corpus` — #1374).
- **P2:** L3 header date; §2.2 data-store rows (image `redis:7-alpine`, loopback-bound, names
  `e2i_redis`/`e2i_falkordb`); feast row (local build from `Dockerfile.feast` FROM 0.43.0; add
  `feast-materializer`); omitted services table (`falkordb-browser` 3030, `config-check`,
  `worker_heavy`, dev-only flower/redis-commander/falkordb-seeder/test); Opik table `:latest` →
  `${OPIK_VERSION:-1.10.8}` + minio pins; L323 "33 route files" → age-well (35 files, 27 routers);
  missing route rows (admin, executive-insights, insights, sentinels, alerts SSE, expert-reviews,
  chat); L340/L1339 44 KPIs → X14; L277 orchestrator workflow sentence (node ids `classify` →
  [`rag_context`] → `route` → `dispatch` → `synthesize`, RAG conditional); §4.1/4.2 box counts
  (label families with pointers, not counts); L502 `dspy_training_signals` → `dspy_agent_training_signals`;
  §4.5 feast views → X9; §4.6/§5 plan paths → X17; line numbers → X16; §6.4 add
  `copilotkit_other` 60/60 s row + health bypass; §6.5 tmpfs "mode 1770" → 0700 uid/gid 1000
  (256 MB light, 2 GB medium/heavy), api/scheduler 1777 `/tmp`; §6.6 add nightly 02:00 cron; §7.1
  Opik `@track` hooks — say they no-op with `OPIK_ENABLED=false` (UNCERTAIN at runtime; measure:
  `grep -rlE '@track\b|from opik|import opik' src/agents src/mlops | wc -l` → 8); §7.3 webhook
  URL → N2 caveat; §9.4 ontology "14 files" → 17, compose "4" → 7; §9.6 FalkorDB sync (add the
  sentinel); §1.3 mention ADR-011/012; ADR-004 add `GET /causal/pipeline/{pipeline_id}`; §2.3
  add `/copilotkit/` and `/mlflow/` host locations; synthesis guard paragraph (#1695); chat
  latency spans note (#1471/#1474); `GET /api/graph/health` `degraded` on an empty curated graph
  (#1762); one-hot confounders / subsample energy-score tournament one-liner (#1413).
- **CONFIRMED_OK:** agent count, LLM split L59-62, observability image table, RRF/graph-boost
  constants, memory-subsystem constants and beat names, ADR index pointer, security pipeline
  tools, rate-limit table values, circuit-breaker states, testing table, rxnav offline mode,
  MLflow auth_basic (live).

### U7 · `frontend/README.md` — **P1 bundle** — effort S (last verified 2026-07-18; 67 frontend merges since)
- **P1 · L74-76** "add a page by adding one entry" — three places in `src/router/routes.tsx`:
  the `lazy()` import (31 today), a `routeConfigs` entry (sidebar), and the hand-written
  `routes: RouteObject[]` element list (L285) wrapping each page in `ProtectedRoute`.
  `grep -nE '^export const (routeConfigs|routes)' frontend/src/router/routes.tsx`.
- **P1 · Gotchas L216-227 MISSING** — CopilotKit hooks (`useCopilotReadable`, `useCopilotAction`)
  throw without a `<CopilotKit>` provider and chat is off in dev; gate behind `useCopilotEnabled()`
  (`E2ICopilotProvider.tsx:436-450`; 8 files do). (#1818)
- **P1 · Chat L131-139 + L41 Charts** — add Flint-compiled charts for every registry KPI
  (`FlintChart`, `flint-chart`, generated `src/lib/kpi-catalog.generated.ts` via
  `scripts/gen_kpi_catalog.py` — #1383), chat→page filter sync `AgentFiltersBridge` over the
  AG-UI CoAgent `filters` state channel (#1724/#1750/#1755), page-context readables
  (`copilotReadableConverters.ts`, #1818), catalog-grounded suggestion pills (#1900).
- **P2:** L66 `config/env.ts` reads only 3 of the VITE_* vars (Supabase vars in `lib/supabase.ts`,
  `VITE_DEBUG` in logger/api-client, `VITE_MSW_ENABLED` in mocks); layout omits `hooks/api/`
  (~20 per-domain query hooks), `mocks/`, `test/`, `assets/`; L186-190 four `live-*.spec.ts`
  exist (+ env needed); L153 `VITE_SUPABASE_URL` is INTENTIONALLY EMPTY in `.env.production`
  (same-origin via nginx) — add as a gotcha; L202-206 add `ensure-main-image` + health gate /
  GHCR rollback (#1780); scripts table add `preview`, `test:ui`; L124-127 query-client: mutations
  never retry (`mutations.retry: 0`, #1846/#1852) and long-running analyses are started once then
  polled with measured ceilings + "Keep waiting" (#1836/#1841/#1844); L84-88 api-client refreshes
  the session once and replays a 401 (#1891); API layer bullet: column labels/definitions come
  from the backend SSOT `src/insights/column_labels.py` mirrored in `lib/column-labels` (#1895);
  L3 date.
- **CONFIRMED_OK:** 30+ pages / six sections, ports, stack versions, auth flow basics, env
  precedence, CI job names.

### U8 · `docs/LLM_CONFIGURATION.md` — **P1 bundle** — effort S (substantive last verification 2026-07-18)
- **P1 · §1 L60-67 Reasoning effort** — since #1299 the knob also drives Claude 5-family models
  (`_ADAPTIVE_THINKING_PREFIXES`: claude-sonnet-5, claude-opus-4-8, claude-fable-5): `"none"` →
  thinking disabled, otherwise adaptive thinking bounded via `output_config.effort`; synthesis
  lanes use 8192 max_tokens. `sed -n 94,102p src/utils/llm_factory.py; sed -n 198,205p src/utils/llm_factory.py`.
- **P1 · §2 L79 `ANTHROPIC_MODEL` row** — consumed for model choice only by `dspy_lm.py` (DSPy
  lane, when `LLM_PROVIDER=anthropic` and `DSPY_LM_MODEL` unset); `chatbot_graph.py:1970` reads it
  only as a telemetry label with a different default (→ N4); not forwarded into containers.
- **P1 · TL;DR L17-19 / §2 L75-77** add the **as-deployed** callout → X2 (both keys needed on the
  droplet; DSPy lane pinned to OpenAI).
- **P2:** L3 date (pin to `llm_factory.py @ <sha>`); L144-145 embeddings → concrete
  (`text-embedding-3-small`, `src/rag/config.py`, OpenAI regardless of provider); §5 add
  direct-SDK consumers that bypass the factory and metering (`src/rag/insight_enricher.py`,
  `src/rag/query_optimizer.py`, Graphiti client — all on `claude-sonnet-4-6`) and the fast-tier
  `POST /api/chat/suggestions` consumer (#1900); L98-99 nine modules now call
  `ensure_dspy_configured`; L101-102 `num_retries=3` (dspy 3.1.0 default); §3/§5 LLM-lane knobs →
  X5 (`DSPY_RAG_*`, `CHATBOT_RAG_*`, startup warm, `ROUTING_LABEL_*` fixed-model list); optional
  note that `src/mlops/agent_cost_tracker.MODEL_PRICING` is an unwired scaffold with stale ids.
- **CONFIRMED_OK:** tier table, entry points/temperature, DSPy precedence, pricing/metering
  (`PRICING_VERSION = "2026-07-18"`, all four factory ids price), RAGAS posture, cross-references.

### U9 · `docs/decisions/README.md` + ADR-008 body — **P1** — effort S
- **P1 · L25 ADR-008 row** add "amended Aug 2026 (#1806 monitoring profile; #1807 freshness
  alarm)" + matching paragraph in ARCHITECTURE §8 → X4.
- **P1 · index rows for ADR-013..017** → C4–C8. **P2 ·** "Next free number: ADR-013" line in
  Conventions; ADR-010 status note ("golden-set DSPy A/B not yet executed; DSPy lane remains on
  openai/gpt-5.6-terra" — UNCERTAIN whether it ran; no PR title indicates it); operational
  decisions list (C4–C8 tail).
- **CONFIRMED_OK:** ADR-009/010/011/012 bodies verified against llm_factory, compose, explain.py,
  causal.py, migrations 104/106/109.

### U10 · `CHANGELOG.md` — **P1** — effort M → X6 (proposed trains below).

### U11 · `docs/runbooks/reviewer-provisioning.md` — **P0** — effort S
- **P0 · L26** `SUPABASE_SERVICE_ROLE_KEY` → `SUPABASE_SERVICE_KEY` + `test -n "$SVC"` sanity line.
  `docker exec e2i_api printenv SUPABASE_SERVICE_ROLE_KEY >/dev/null 2>&1; echo $?` → 1; `… SUPABASE_SERVICE_KEY` → 0.
- **CONFIRMED_OK:** override file location/line, `GOTRUE_DISABLE_SIGNUP` live, settings probe.

### U12 · `docs/runbooks/migrations.md` — **P1 bundle** — effort S
- **P1 · L29-31 / L62-63** "the droplet's self-hosted Supabase stack exposes only REST creds, no
  DB URL" — the droplet `.env` DOES define `SUPABASE_DB_URL` (127.0.0.1:5432 = Supavisor pooler;
  supabase-db publishes 5433); deploys use docker mode only because the SSH step does not export
  `.env`. *Fix:* say that, and warn that a sourced `.env` flips the runner into url mode.
  `grep -c '^SUPABASE_DB_URL=' .env` → 1.
- **P1 · L200-218 §5 verify** `SELECT 'experiment_monitor'::e2i_agent_name;` — migration 055
  altered `agent_name_enum` (observability), not `e2i_agent_name` (memory schema, mig memory/029).
  *Fix:* cast against the enum the migration touched; note there are two.
- **P2:** L66-68 "two 099, two 101" → five shared numbers (063/064/065/099/101) — age-well
  wording + command; L47 deploy wording (reset to the resolved target sha after the image
  assertion — #1780/#1785); L248-252 trigger list (`scripts/**`, `data/kg_cache/**`; `database/**`
  deliberately not) — #1783; header date.
- **N5 (code):** `scripts/run_migrations.sh:5-6` header says "migrations/ AND memory/" — scans 8 dirs.

### U13 · `docs/runbooks/sentinels.md` — **P1 bundle** — effort S
- **P1 · L645 / L551** `pytest tests/unit/test_memory/test_sentinels/` — directory does not exist
  (pytest exit 4); tests are `tests/unit/test_memory/test_sentinel*.py` (8 files) +
  `tests/unit/test_security/test_sentinel_external_unreachable.py`.
- **P1 · L660 / L674 / L694** bare `psql -c` fails on the droplet (no local socket); use
  `docker exec -i supabase-db psql -U postgres -d postgres -c "…"` (the migrations-runbook convention).
- **P1 · L710** `docker compose stop scheduler` — no compose file at repo root; `docker compose -f
  docker/docker-compose.yml stop scheduler` or `docker stop e2i_scheduler`.
- **P2:** L44-46 task name (`src.tasks.sentinel_dispatcher` in `insight_lifecycle_tasks.py`, beat
  key `insight-lifecycle-sentinels` 300 s); L101-111 `data_drop`/freshness `table` must be in
  `THRESHOLD_WATCHABLE_TABLES` (5 tables) and `ts_column` a plain identifier; L507-529 REST
  requires OPERATOR, `brand="all"` ADMIN, other brands must be granted; PATCH/DELETE exist;
  L760-765 `celery inspect scheduled` shows ETA tasks, not beat — use `docker logs e2i_scheduler
  --since 10m | grep sentinel_dispatcher` (UNCERTAIN, reasoned not executed); L7/L825 plan pointer
  → X17; ~15 line citations → X16.
- **N6 (code):** `src/memory/sentinels/config_loader.py:20` docstring maps `staleness_threshold →
  threshold_breach`; the code map says `invalidation_count`.
- **CONFIRMED_OK:** §2-§5 yaml ids/cooldowns, lifecycle transitions, guard substrings, channel,
  queue cap.

### U14 · `docs/runbooks/gotrue-smtp.md` — **P1** — effort S
- **P1 · L68-69 / L96-97 Status** claims the Gmail SMTP creds are loaded — live `supabase-auth`
  env is back on the template placeholders (`GOTRUE_SMTP_HOST=supabase-mail`, `GOTRUE_SITE_URL=http://138.197.4.36`).
  *Fix:* rewrite Status with a dated re-check (do not put the mailbox address in the doc).
  `docker exec supabase-auth printenv | grep -E '^GOTRUE_(SMTP_HOST|SITE_URL)='`.
- **P1 · L46-49 Activation** `MAILER_URLPATHS_RECOVERY=/reset-password` — GoTrue's default is its
  own `/auth/v1/verify` endpoint; the SPA route is reached via `redirectTo` (`AuthProvider.tsx:363`).
  UNCERTAIN end-to-end (no SMTP path on the box); recommend leaving the default + `SITE_URL` +
  `ADDITIONAL_REDIRECT_URLS`.
- **P2:** L13-14 says the tracked override interpolates `GOTRUE_SMTP_*` — it deliberately defines
  none (comment block L82-93; the base compose maps them); L73-79 DO outbound-SMTP block is a
  2026-06-13 measurement — add the re-probe one-liner and a date.

### U15 · `docs/runbooks/kg_cache.md` — **P2** — effort S
- L100 "~72 KB" → ~80 KB (or drop); MISSING: the cache is COPYed into the API image and
  `data/kg_cache/**` is a deploy trigger (#1783) — a rebuilt cache reaches prod only through a new
  image; add a "Citation channel / audit sidecar 1.8" pointer only if the team wants it (critic:
  internal plumbing, NOT_DOC_WORTHY). **CONFIRMED_OK:** build flags, 7 of 74 `treats` edges, pass
  ENABLED log line.

### U16 · `docs/runbooks/frontend-env-and-csp.md` + `frontend-serving-flip.md` — **P2** — effort S
- frontend-env-and-csp L22-23: drop `VITE_DEFAULT_MODEL_ID` (no reader, not in `.env.example`).
- frontend-serving-flip: a completed one-time procedure with no done/archived banner (live nginx
  proxies `/` to 127.0.0.1:3002; `e2i_frontend` serves prod) — stamp "Completed 2026-06 — point-
  in-time; do not re-run" at the top and fix L117-119 (add `ensure-main-image`, newest-built-
  ancestor selection, on-box build refused). `grep -ciE 'completed|archived' docs/runbooks/frontend-serving-flip.md` → 0.
- **CONFIRMED_OK:** CSP `connect-src https:`, `/functions/v1/` proxy, zero `functions.invoke`.

### U17 · `docs/api/chat.md` — **P1 bundle** — effort M (last moved 2026-07-31)
- **P1 · §5 L158-183 `ChatResponse`** add `routing_authority` (#1596; values `pipeline` |
  `legacy` | `explicit_target` #1723). `grep -c routing_authority docs/api/chat.md` → 0.
- **P1 · §3 L126-147 `dispatch_info`** — key set grew (routing_authority, node-span timing keys
  `node_wall_ms`, `graph_total_ms`, `untimed_overhead_ms`, `first_request_in_worker`, `worker_pid`,
  `orchestrator_*_ms`, `rag_stage_ms`, `rag_meta`, `empty_response_fallback`) — #1471/#1474/#1596.
- **P1 · §2.2 L58-92 AG-UI body** add the `state.filters` channel (shape, 'All'/'All US'
  sentinels, precedence over asking "which brand?" — #1724) and `context` readables (every
  `useCopilotReadable` on the page arrives as ON-SCREEN APP CONTEXT — #1818); the empty-delta
  invariant (TEXT_MESSAGE_CONTENT deltas are never empty — #1724).
- **P1 · §6 L206-207** "CLARIFICATION_NEEDED never reaches the user" — `/chat/stream` now asks
  back via `clarify_node` for the clarify-eligible intents, pending ask stored in conversation
  state (#1441). **P1 · §6 L214-216** quoted bridge preamble is "the old wording"; two current
  variants selected by tool-grounded evidence (#1465).
- **P2:** §2.2 keepalive (`: keepalive` SSE comment every 15 s, `X-Accel-Buffering: no`,
  `PROXY_READ_TIMEOUT_SECONDS = 300` mirrors nginx — #1664/#1674); §8 add `analytics/errors`,
  `analytics/hourly`; §8 L244 suggestions row → catalog + validator (#1900); §2 add a "Bound
  tools" table (10 `@tool` functions incl. `predict_hcp_segment_likelihood` — #1399); region
  resolution / clarify note (#1565/#1572); §7 `ROUTING_LABEL_*` knobs → X5.
- **CONFIRMED_OK:** §2.2 auth gating (note #1434 double-gate), frame types, classifier modes,
  bridge mechanism, telemetry, labeler cron.

### U18 · `docs/api/crystal_digests.md` — **P1** — effort S
- **P1 · §3 L148-160** "audit struct is not persisted" — `crystal_narrative_audits` exists
  (`database/memory/028_provenance_append_only.sql`), one row per crystal incl. `input_prompt` for
  the PHI scanner. **P2:** L133/L135-146 add `input_prompt` (9 fields, class at types.py:450);
  L318-319 `src/api/middleware/jwt_auth.py` does not exist → `JWTAuthMiddleware` in
  `auth_middleware.py`; L7/L606 plan pointer → X17; §1 add provenance filtering
  (`apply_provenance_filter`, synthetic episodics excluded — #850/#977); line numbers → X16.
- **CONFIRMED_OK:** endpoint reference, auth deps, 202, SSE details (`: ping` 15 s, cap 100).

### U19 · `docs/data/00-INDEX.md` — **P1** — effort S
- **P1 · L18/L131/L139** "19 tables, 12 enums, 28 views, 6 functions" — 19 / **11** enums
  (`agent_name_type_v2` retired by mig 056) / **37** views in the schema file (+2 by migrations)
  / 6 functions + migration RPCs. Prefer age-well wording.
  `S=database/core/e2i_ml_complete_v3_schema.sql; grep -ciE '^CREATE TYPE .* AS ENUM' $S; grep -ciE '^CREATE (OR REPLACE )?VIEW' $S`.
- **P1 · L21/L68/L146** Feast → X9. **P1 · L15-25 / L161-181** four files in docs/data are
  absent from the map: `SYNTHETIC-CAUSAL-DATA-GUIDE.md`, `OPTUM-MART-DATA-DICTIONARY.md`,
  `kpi_coverage_map_synthetic.md`, `optum_mart_column_schema.csv`. **P2 · L23/L141/L144**
  memory/audit table counts → families with pointers.
- **CONFIRMED_OK:** header 22 agents, 45 KPIs.

### U20 · `docs/data/02-CORE-DATA-DICTIONARY.md` — **P1** — effort L (→ X7)
- **P1 · L52-67 Enum reference** remove `agent_name_type_v2` (retired, mig 056 #607); rewrite
  the L67 note (both orphan enums gone; roster is code-defined, 22).
- **P1 · L1596-1639 agent_registry** "11-agent … (11 agents)" — mig 057 adds `tool_composer`,
  `experiment_monitor` (13 rows); Tier-0 agents and `cohort_profiler` are not registered here.
- **P1 · patient_journeys L831-874** missing columns from migs 033 (adherence_rate, refill_count,
  gap_days), 036 (payer_*), 063 (`is_synthetic`), 064/087/088/112 (synthetic causal substrate),
  068/107 (brand-gated clinical eligibility), 132 (`treatment_arm`, `persistent_180d`,
  `discontinued_180d` semantics). **P1 ·** hcp_profiles tiers (033), triggers `brand_id`/
  `roi_estimate`/`control_group_flag` (033/051), business_metrics/causal_paths/treatment_events
  additions (049/063/067/115) + one shared `is_synthetic` callout. **P1 ·** "KPI registry &
  history tables" section: `kpi_query_registry` + `kpi_query` RPC (cap 6 since 120), `kpi_history`
  (079; writers `history_capture`/`history_backfill`, brand axis #1896, open-month skip #1916),
  `hcp_brand_adoption`, `territory_metrics`, `npi_taxonomy`, `v_kpi_history_coverage` (126).
  `for c in $(grep -hoE 'ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS [a-z_0-9]+' database/migrations/*.sql | awk '{print $NF}' | sort -u); do printf '%s %s\n' $c $(grep -c "\`$c\`" docs/data/02-CORE-DATA-DICTIONARY.md); done`.
- **P2:** L772/L829 hard column counts (32 / 40 in the schema file + migrations) → drop;
  `path_id` wording if it says UUID (content-addressed since #1725 — borderline).

### U21 · `docs/data/03-ML-PIPELINE-SCHEMA.md` — **P1** — effort M
- **P1 · L16-18** "through `028_cohort_constructor_tables.sql`" → 035; "21-agent" ×3 → X8.
- **P1 · §7** add `drift_qualifying_features(p_window_days, p_min_samples=30, p_include_synthetic=false)`
  (mig 131, #1748 — the drift-monitor dispatcher's substrate). **P1 · §2** add
  `validation_outcomes` (007 + 121) and the mig-119 `validation_status` domain / `enforce_*` trigger
  (#1352/#1385). **P2:** §3 `twin_retraining_jobs` (029) + `twin_simulations.data_provenance`
  (030); §10 GEPA uniqueness indexes + `v_active_instructions` (035); §9.1 `answer_correctness`
  (033); discovery section: FCI latent diagnostic + bootstrap edge stability (#1883/#1886) if C9 =
  (b); L369 "13 default tools" UNCERTAIN — say where the count comes from.

### U22 · `docs/data/04-KNOWLEDGE-GRAPH-ONTOLOGY.md` — **P1** — effort S
- **P1 · L18 / L227** 21 agents → X8 (add `cohort_profiler` to Tier 0). **P1 · L517-547** the
  runtime graph is **`e2i_causal`** (`FALKORDB_GRAPH_NAME`, `config/005_memory_config.yaml`);
  `falkordb_config.yaml`'s `e2i_semantic` is a legacy seed-only name (#749) — the doc never says
  which graph to `GRAPH.QUERY`. **P1 · L188** `validation_status` enum → mig-119 domain
  (`pending`, `validated`, `needs_review`, `refuted`, `overturned`, …) + semantics (#1385).
- **P2:** L498 "11 Agent nodes" — `scripts/seed_falkordb.py` AGENTS has 6; add a "Graphiti
  temporal layer" subsection (Episode/Community nodes, MENTIONS/MEMBER_OF/RELATES_TO).
- **CONFIRMED_OK:** 8 node types / 15 edge types of the ontology yaml, 5 inference rules.

### U23 · `docs/data/05-FEATURE-STORE-REFERENCE.md` — **P1** — effort M → X9 (+ P2 registry
lifecycle / `feast-apply.yml` CI gate / `clear_goldstd_ts_markers.py` subsection, #1298).

### U24 · `docs/data/06-KPI-REFERENCE.md` — **P1 bundle** — effort L (last moved 2026-08-11)
- **P1 · L132 / L960** "RPC caps at 4 parameters / region and window mutually exclusive" —
  cap is 6 since mig 120; `_windowed_region` variants exist (#1396).
- **P1 · L1188-1195 WS3-BI-009** "brand × region refused" — served by mig-128
  `business_impact_conversion_rate_brand_region` (#1579). **P1 · BR-001..005 L1226-1356** add the
  region axis (mig 127 `brand_specific_*_region`, BR-002 primary/fallback twins — #1570).
  **P1 · WS3-BI-010 L1201-1222** add scoping (125) + temporal-variability band (124, suppressed
  below n=6) — see roi_methodology §8.2. **P1 · L152-157 Registry migrations** extend with 113,
  116, 118, 120, 124/125, 127, 128, 129, 130. **P1 · WS2-TR-001/002 L41-42, L745-790** still the v1
  `TP/(TP+FP)` formulas — mig 113 redefined the truth metrics (2026-07-20 definition break;
  #1300). **P1 · L93-108 window grammar** "Not supported: this month…" — calendar-aligned phrases
  are accepted since #1554 (clamped to elapsed span, #1546). **P1 · new "Region provenance"
  subsection** (`region_requested` / `region_applied` / `region_status` default|applied|
  not_applicable; a not_applicable value must not be labelled with the region — #1540).
  **P1 · new "Measure basis"** (`measure_basis` on four response models, `src/kpi/measure_basis.py`;
  figures with different bases must not share an axis — #1647). **P1 · L128-133 / L152-158**
  patient axis on a KPI whose calculator does not bind it is refused, not dropped (#1911/#1913).
  **P1 · new "Claims-lag nowcast" subsection** (`GET /api/kpis/{kpi_id}/history/nowcast`, Rx-volume
  family gate, mature/provisional/nowcast series, mig 116 triangles, Time-Series overlay — #1307/#1308).
  **P1 · L1601 `v_kpi_history_coverage` / new "KPI history" section** (table 079, writers, brand
  axis for six present-state KPIs #1896, weekly capture, open-month skip #1916). **P1 · §WS1
  Model Performance L495** add the sampling-aware trend status note (#1916, → ADR-013).
- **P2:** L1534 Brier target 0.15 → 0.185; L1589 helper views "eight… defined" vs 9 rows (7
  referenced + `v_kpi_label_quality` retained + `v_kpi_history_coverage`); L172-173
  `KPI_SEMANTIC_NOTES` lives in `src/services/kpi_resolution.py`; L520 ROC-AUC calculator order
  (goldstd holdout → `model_performance_roc_auc` registry query → MLflow fail-closed); L272/L1244/
  L1296/L1322 non-existent source columns (`match_rate_vs_claims`, `diagnosis`, `test_type`,
  `diagnosis_date`); L3 header version relationship; WS2-TR-005 `data_through` + polarity
  (#1713/#1720); MLflow-unavailable fail-closed status (#1663); cohort_profiler statements note
  (117/129/130); Model Performance page trend default `auc_roc` (#1319).
- **CONFIRMED_OK:** 45 KPIs header, WS1 rows retuned (#1316), WS1-DQ-006 / WS1-MP-006 renames,
  OOS-union holdout mention, `v_kpi_history_coverage` row text (#1539).

### U25 · `docs/data/07-SUPPORTING-SCHEMAS.md` — **P1 bundle** — effort M
- **P1 · L92 / L96** `memory_event_type` (base 8 values + 21 `*_completed` extensions = 29) and
  `e2i_agent_name` "All 21 agent names" (23 without `cohort_profiler` — see N7). **P1 · L245**
  `learning_signals.cycle_id` "soft reference, FK not re-added" — mig 072 added the FK with ON
  DELETE CASCADE (#884). **P1 · L748-756 / L465-471 permissions** "Memory SELECT/INSERT … Full
  access" — mig 058 (#703) REVOKEd all grants from `anon`/`authenticated`; access is via the
  service-role backend. **P1 · Memory schema** add sections for `agent_knowledge_store` (065),
  `dspy_agent_training_signals` (014), `ml_hpo_patterns` (009), `procedural_templates`, `sentinels`,
  insight lifecycle tables, `crystal_narrative_audits` (028), feedback-loop tables, `gap_analyses`
  + a "Memory RPCs" table. **P1 · L662-690 audit_chain_entries** `action_type` now includes
  `<node>` / `<node>_error` and `validation_passed` is a verdict, not a run outcome (#1902).
- **P2:** `security_audit_log` section; RAG `is_synthetic` chunks + OR full-text (004/005/006);
  chat `computed_user_id` owner-inherit trigger (123, #1433); `routing_classifier_metrics` extra
  columns (#1342); L461-462 `chatbot_optimization_requests` "dormant" → consumed when
  `CHATBOT_OPT_DRAIN_ENABLED=true` (05:30, #1521); L84/L276/L658 line counts → drop.

### U26 · `docs/data/08-LEAKAGE-DETECTION-CONTRACT.md` — **P1** — effort S
- **P1 · L52-76 ladder / L307-318 quick reference** no Layer-2 (knowledge-graph causal-role
  signal) step although the voter names a "KG-contradictory abstain": `classify_kg_signal` over
  the UMLS drug–disease cache, `kg_mode` ∈ {off, shadow, active} (`_resolve_kg_mode`), activated
  with a real signal in #1619. `grep -ciE 'kg_mode' docs/data/08-LEAKAGE-DETECTION-CONTRACT.md` → 0.

### U27 · `docs/data/01-DATA-CONVERSION-GUIDE.md` + `docs/data/templates/README.md` — **P1/P2** — S
- **P1 · 01 L395/L398** `E2I_JOURNEY_STAGES` lists 4 and claims `treatment_switch` is absent —
  pandera has all 12 `journey_stage_type` values. **P2 · 01 L16** 21-agent → X8; L554 "13
  agents" → "the 13 Tier 1–5 agents the harness exercises"; add a "columns added after v3.0" note.
  **P2 · templates L66/L71** journey stages (12) and `insurance_type` vocabulary conflict between
  docs 01/02/templates (`uninsured` vs `cash`; VARCHAR(20), no CHECK) — pick one.

### U28 · `docs/data/SYNTHETIC-CAUSAL-DATA-GUIDE.md`, `kpi_coverage_map_synthetic.md`, `OPTUM-MART-DATA-DICTIONARY.md` — **P1/P2** — S
- **P1 · SYNTHETIC-CAUSAL §3.1 L141 / docs/SYNTHETIC_DATA L173** business_metrics now carry a
  deterministic brand × region execution matrix + anchored step events (#1833/#1849).
  **P1 ·** "Brand-distinct causal axes & commercial-arm paths" subsection for the seed/backfill
  scripts (#1321/#1325). **P2 ·** §2.2 heading "(initiators only)" contradicts mig 132 (outcomes
  drawn for every row as a function of `treatment_arm` — #1894); §2.4 `--anchor-to-now` note vs
  mig-089 frontier anchoring (UNCERTAIN which reseed mode is operative — verify against
  `scripts/reseed_synthetic.sh` defaults); line numbers → X16.
- **P2 · kpi_coverage_map** dated banner: measured on the pre-095 registry; DQ-003/004/007/009
  read source tables directly since 095; re-run `scripts/check_kpi_coverage.py`.
- **P2 · OPTUM-MART dict L6/L107** link to `docs/reports/optum-mart-data-treatment-findings-20260608.md`
  does not resolve — locate or drop. Everything else in that file checks out.

### U29 · `docs/SYNTHETIC_DATA.md` — **P1 bundle** — effort M (last touch 2026-07-21)
- **P1 · L30 / L667** "v3 design exists as a proposal only, no `src/ml/synthetic_v3/`" — the v3
  `rwd_realistic` regime IS implemented (`src/repositories/synthetic_rwd_realistic.py`, 20
  referencing files, drives the leakage-defense suite and T2.2/T2.3 calibration); still not a
  tier-0 `--regime`. **P1 · L160-191 generator inventory** "11 generators" (table lists 10) — 18
  concrete classes; add agent_activities, causal_paths, coverage_tables, experiment/ab_experiment,
  feedback, mlops, observability, hcp_brand_adoption (#1355/#1379/#1551/#1555/#1725/#1833) and the
  load order. **P1 · L577 / L583** `--enable-mlflow`, `--include-bentoml` do not exist (argparse
  rejects); MLflow/BentoML are ON by default, `--disable-mlflow` / `--no-bentoml` turn them off.
  `grep -c '"--enable-mlflow"\|"--include-bentoml"' scripts/run_tier0_test.py` → 0.
  **P1 · L508-561 Digital Twin** effects now come from an `EffectDataProvider`
  (`src/digital_twin/effect/`: Synthetic default, Cohort) — the fixed base-effect × multiplier
  pipeline is gone (2026-07-08). **P1 · L621-626** `load_synthetic_data.py` flag table
  (`--anchor-to-now/--anchor-ref`, `--append-frontier/--frontier-ref`, `--only-tables` #1387,
  `--dgp`, `--parquet-out/--parquet-only`, `--small`, `--tag`, `--refresh-ab`) → also C3.
- **P2:** L368-375 step order (2b/2c Feast steps, cohort_constructor is step 3, feature_analyzer
  step 6); L422-435 Layer-2 table missing `experiment_monitor` (13 validators); Key Source Files
  add `src/ml/synthetic/claims/` (#1306) and `frontier_append.py`; L70 "~line 4374" → symbol.
- **CONFIRMED_OK:** regimes/kwargs/N/bands, DGP ATEs, splits 60/20/10/10, volumes, thresholds,
  fixtures, golden sets, `_adaptive_criteria_enabled` reference.

### U30 · `docs/RWD_PIPELINE.md` — **P1** — effort S (dated 2026-04-12, oldest method doc)
- **P1 · L36** loader prefers `e2i_ml_v3_patient_journeys.parquet` (Optum converters) and falls
  back to the JSON (CSU). **P1 · L23-32** flags table misses `--feature-manifest-source`
  (auto-detected from `--data-dir`), `--deployment-intent`, `--min-samples-per-split`, `--regime`,
  `--split`, `--n-total`, `--seed`. **P1 · status line** Optum/mart cohorts run through
  `scripts/run_optum_tier0_test.py --cohort <name>`, not bare `--data-dir`. **P2:** collapse the
  L59-68 migration table (wrapper removed 2026-04-12) to one sentence; L38-40 use the converter's
  documented `--input/--output` form (UNCERTAIN whether bare invocation works).

### U31 · `docs/model_success_criteria.md` — **P1** — effort M
- **P1 · L18-19** `is_adaptive_criteria_on()` does not exist → `_adaptive_criteria_enabled()`
  (`criteria_validator.py:185`). **P1 · §2 L43-101** missing the clinical|commercial
  `deployment_intent` axis (2026-06-07): commercial `minimum_auc = max(0.60, baseline+0.05)`
  (adverse 0.58/+0.03), recall 0.50, MCC 0.10, lift 0.08, p_t 0.05. **P1 · L56-68 / L87-97**
  calibration slope/intercept/ECE caps and the overfit table are FLOORS widened for small
  evaluation splits (√(1000/n_test), #866). **P2:** L177 index path →
  `docs/reports/DOCUMENTATION_UPDATE_INDEX_20260603.md`; line refs → X16.
- **CONFIRMED_OK:** fixed defaults, clinical gate values, p_t table, QC gate (3 enforcement
  points, 0.80), all-null-skip log line.

### U32 · `docs/OPTUM_CONVERSION.md` + `docs/OPTUM_MART_CONVERSION.md` — **P1/P2** — S
- **P1 · MART L555 / L578** "commercial AUC bar (0.65)" — the adaptive commercial floor is
  `max(0.60, baseline+0.05)`; 0.65 is the Optum runner's own `--min-auc` step gate (the doc's own
  L599 says ≥ 0.60). **P2:** MART L441-446 `_LEAKY_HCP_COLS` omits `adoption_category_method`
  (13 entries); L623-626 cosmetic follow-up appears done (`kisqali_discontinuation_tier0_e2` 0
  hits) — delete or mark historical; L6-9 parquet shapes UNCERTAIN (not re-read on the memory-
  capped box; command given). OPTUM_CONVERSION L519-531 persist script flags (`--lookback-days`,
  `--batch-size`, `--log-level`) + `scripts/cleanup_falkordb_shells.py` (#890); L489-499 runner
  flags (`--feature-manifest-source optum`, `--single-model`, `--min-auc`, `--deployment-intent`);
  plan references → X17. **CONFIRMED_OK:** converter flags, GAP_THRESHOLDS, allow/forbid list
  counts (64/16/21/19), `_LOG1P_FEATURES` 10.

### U33 · `docs/roi_methodology.md` + `docs/synthetic_v3_design.md` — **P2** — S → X16 (all
constants, formulas, migrations 124/125 and the regime-file refs verified; only cross-file line
numbers and the absent plan header drift).

### U34 · `docs/api_connectivity_review.md` — **P1 disposition** — effort S
- Point-in-time audit (2026-05-16). Verified 2026-09-07: headline issues 1-7 and §8 actions 1-6
  are resolved (`RAG_BASE = '/v1/rag'`, explain `/predict/batch`, cognitive `/session*`, pages
  wired to `hooks/api`; `CausalDiscovery.tsx` no longer exists); only §8 action 7 (AgentOrchestration
  inline `useQuery` → `api/agents.ts` + `use-agents.ts`) remains open (→ N10, file an issue).
  *Fix:* `git mv` to `docs/Archive/api_connectivity_review_20260516.md` with a banner; do not
  refresh its counts (they used a different inventory method). Point readers at
  `docs/decisions/m5-backend-orphans-triage-20260608.md` for current wiring status.

---

## NOT-A-DOC-GAP — code/config items surfaced by the audit (N1–N10)

Flagged for the owner, not executed (REASON-BEFORE-RULES: each needs an intent check first).

| # | Item | Priority | Evidence / measure |
|---|---|---|---|
| N1 | `x-common-env` does not forward `ADAPTIVE_CRITERIA`, `ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`, `ADAPTIVE_VALIDITY_EVALUATOR_MODEL` — the documented rollback switch is inert inside containers | P1 | `for v in ADAPTIVE_CRITERIA ADAPTIVE_VALIDITY_EVALUATOR_ENABLED ADAPTIVE_VALIDITY_EVALUATOR_MODEL; do grep -c "\${$v" docker/docker-compose.yml; done` → 0 0 0 |
| N2 | Alertmanager routes to `http://api:8000/api/v1/webhooks/alertmanager`; no router serves it (404). Latent: the container is behind the `monitoring` profile and not running | P2 | `grep -rn webhooks src/api/ --include='*.py' \| wc -l` → 0 |
| N3 | `docker/docker-compose.monitoring.yml` is a superseded standalone overlay (2026-01 pins, different network) not referenced by deploy.yml — delete or add a SUPERSEDED header | P2 | `grep -c monitoring.yml .github/workflows/deploy.yml` → 0 |
| N4 | `chatbot_graph.py:2023` labels the generate span with `os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")` while the LLM is built by the factory — telemetry misattribution on the droplet (host pins opus-4-8) | P2 | `sed -n 1965,1971p;2021,2024p src/api/routes/chatbot_graph.py` |
| N5 | `scripts/run_migrations.sh:5-6` header comment says 2 dirs; it scans 8 | P2 | `sed -n '5,6p' scripts/run_migrations.sh` |
| N6 | `src/memory/sentinels/config_loader.py:20` docstring maps `staleness_threshold → threshold_breach`; code says `invalidation_count` | P2 | `grep -n staleness_threshold src/memory/sentinels/config_loader.py` |
| N7 | `e2i_agent_name` enum (memory schema) has no `cohort_profiler` value although the agent is in the 22-roster — add a memory migration if it writes memory rows, else document as intentional | P1 | `grep -rn "'cohort_profiler'" database/memory/ database/migrations/ \| grep -c 'ADD VALUE'` → 0 |
| N8 | `scripts/run_tier0_test.py` `--regime` help/docstring lags `_VALID_REGIMES` | P2 | see SYNTHETIC_DATA finding [23] |
| N9 | `docs/api/index.html` (a `make api-docs` output) is not gitignored | P2 | `grep -n 'docs/api' .gitignore` → only openapi.json |
| N10 | `frontend/src/pages/AgentOrchestration.tsx:268` inline `useQuery` — the last open action of the May connectivity review; file a GitHub issue so archiving U34 does not lose it | P2 | `ls frontend/src/api/agents.ts frontend/src/hooks/api/use-agents.ts` → absent |

---

## CHANGELOG proposed trains (input for X6 — verify PR numbers at write time)

**## 2026-09 (in progress)** — Added: guided causal-discovery hardening (#1878, #1879, #1883,
#1885–#1890: prior-asserted DAGs never reported as discovered; bootstrap edge-stability gate;
FCI latent diagnostic annotate-only; `edge_provenance`; covariate-role 400); discover-effects
question subset + cooperative cancel + restart-orphan read-repair (#1898, #1899); copilot
suggestion pills from a capability catalog + validator (#1900, #1903, #1908–#1910, #1912, #1913,
#1915, #1919); segment-analysis definitions + CI-aware cross-library agreement + estimand fix
(#1891–#1893, #1895); model-performance sampling-aware trend (#1916); agent-health error-row
definition + 30-day patterns (#1902); kpi_history brand axis (#1896); migration 132 outcome
comments (#1894). Fixed: investigator reward floor (#1905), one-decimal health score (#1897),
insights enumeration-marker guard (#1881). Changed: transformers 5.x / hf-hub 1.x (#1884),
tornado + allowlist (#1882).

**## 2026-08** — Added: real-pipeline RAGAS gate + RAGAS-fed GEPA (#1490–#1494, #1499,
#1508–#1513, #1519, #1521, #1522, #1529); KPI region provenance / `measure_basis` / calendar
windows / region variants (#1539, #1540, #1554, #1570, #1571, #1579, #1647; migs 126–128); ROI
temporal band + scoped headline (#1525, #1535, #1537; migs 124/125); chat cold-start warm, RAG hop
economics, GUNICORN_PRELOAD (#1474, #1483, #1490, #1589); KG Layer-2 activation + committed cache
(#1607, #1619, #1621); clinical context + clinical narrative (#1764, #1812); AG-UI empty-delta P0
fix + filters channel + readables (#1724, #1750, #1755, #1818); SSE keepalive (#1664, #1674);
routing_authority (#1596); feedback optimizer gate (#1677); monitoring opt-in profile + maintenance
freshness cron/workflow (#1798–#1807, #1810); graph emptiness sentinel + FalkorDB `/data` volume
(#1759, #1761, #1771); deploy: newest-built-sha, ensure-main-image, published-image assertion,
image-drift gate, worktree abort, separate prune (#1436, #1480, #1782, #1785, #1789, #1792, #1794);
segments run budget, unmodeled-pair 400, keep-waiting, no mutation retry (#1829, #1836, #1844,
#1845, #1852); gap time_period grammar (#1837); three-state refutation verdicts (#1869, #1871);
nav rename Executive Insights + How E2I Works (#1824, #1866); synthetic brand×region matrix
(#1849); Celery beat crontabs + persisted state (#1653, #1777). Late July (append to 2026-07):
nowcast + claims arrival plane (#1306–#1308), WS2 truth metrics (#1300), kpi_query 6 params
(#1396), clarify ask-back (#1441), conditional required checks (#1445), Flint charts (#1383),
causal SLA 300 s (#1422), deploy concurrency (#1428).

---

## Critic adjustments applied

- **Downgraded P0 → P1:** README monitoring sentence (a plain `up -d` still yields a working
  app; docker/README already documents the profile); #1307 nowcast and #1898 discover-effects
  "no doc home" (OpenAPI-served with summaries; sibling rows with identical rubric were P1; the
  proposed hand-written docs/api/causal.md duplicates `/api/docs`) → C9 decision instead.
- **Held but corrected:** ARCHITECTURE §2.2 — verdict STALE holds, the auditor's replacement
  wording ("base + frontend-dev overlay") was wrong; X3 carries the live-verified truth.
- **Auditor contradictions resolved:** `health_check.sh` "24 services" — README auditor
  (STALE, derived total) is right, ONBOARDING auditor's CONFIRMED_OK is wrong (X15). Compose set
  in production — deploy-docker auditor right, architecture auditor wrong (X3). Nginx rate-limit
  zones — values right, file wrong (zones live in `/etc/nginx/nginx.conf`, applied in
  `host-nginx.conf`). `docker/env.example` "zero consumers" — one hint string in `Makefile:47`
  names it (decision unchanged).
- **Over-reach dropped:** #1618 audit sidecar 1.8, #1471 span instrumentation, #1759 FalkorDB
  path → NOT_DOC_WORTHY (internal plumbing; kept only as optional one-liners). README L127
  "Recent Highlights" is not a separate defect — folded into X6. Four `docker/env.example`
  per-line findings → one decision (U4). Six per-field causal-schema rows → C9. Alertmanager
  webhook P1 → P2 (latent behind the profile).
- **Critic misses added:** 05-FEATURE-STORE-REFERENCE L382 (canonical Feast doc, X9);
  frontend-serving-flip completion banner (U16); ONBOARDING `make deploy` appears three times
  (X3); migrations 130/131 (#1741/#1748) and #1873 were in no coverage row (X7, C9); 34 PRs named
  in no coverage row — inspected: #1873 (doc-relevant, → C9), #1493 (comment-only migration —
  check 03 §9.1 semantics), #1455/#1468 (health_score chat narration, borderline, no action),
  rest test/internal.
- **Uncertainties resolved by hand this session:** MLflow `auth_basic` IS on the live host
  `/mlflow/` block (`/etc/nginx/sites-enabled/e2i-analytics:158`) → ARCHITECTURE L998 / ONBOARDING
  L638 CONFIRMED_OK. Remaining UNCERTAIN rows keep their measure command in place.

---

## Verification (hand-checked 2026-09-07 on the droplet checkout, read-only)

- `grep -o '\${[A-Z_]*:?' docker/docker-compose.yml | sort -u` → FALKORDB_PASSWORD, GRAFANA_ADMIN_PASSWORD, REDIS_PASSWORD, SUPABASE_POSTGRES_PASSWORD; `grep -c SUPABASE_POSTGRES_PASSWORD DEPLOYMENT.md docker/README.md docs/ONBOARDING.md .env.example` → 0 0 0 0 (docker/env.example → 1); `docker compose --env-file .env.example -f docker/docker-compose.yml config -q` → **rc 1**, "required variable SUPABASE_POSTGRES_PASSWORD is missing a value" → **X1 real**.
- `sed -n '55,70p' docker/README.md` → Required table lists `ANTHROPIC_API_KEY`, no `OPENAI_API_KEY`; `sed -n '70,80p' docs/ONBOARDING.md` → Anthropic required / OpenAI "(optional)"; `src/utils/llm_factory.py:112` → `os.environ.get("LLM_PROVIDER", "openai")` → **X2 real**.
- `sed -n '419,427p' .github/workflows/deploy.yml` → `pick_overlay()` returns "" when `AS production` present; `grep -n 'AS production' docker/frontend/Dockerfile` → 106; `docker ps` → `e2i_frontend ghcr.io/enunezvn/e2i-frontend:3fbd5cd5… 3002->80`, `e2i_api ghcr.io/enunezvn/e2i-api:3fbd5cd5… 8000->8000`; `docker-compose.yml:803 container_name: e2i_api`, `:1156 e2i_frontend`, `:1160 "3002:80"`; `Makefile:142-146 deploy: ./scripts/deploy.sh`; `scripts/deploy.sh:25` dev overlay, `:77 reset --hard`, `:136 git checkout "$PREV_SHA"`; `make deploy` present at docker/README.md:95/98, ONBOARDING:419/420/535/538/950/951; ARCHITECTURE:133-134 `e2i_api_dev` / `e2i_frontend_dev` → **X3 real, corrected wording**.
- `sed -n '169,190p' docs/ONBOARDING.md` → psql list; `ls database/migrations | wc -l` → 117; `scripts/run_migrations.sh:50-58` → 8 dirs; `Makefile:148-150 db-init` echo stub → **U5 Step 7 real**.
- `docker exec e2i_api printenv SUPABASE_SERVICE_ROLE_KEY` → rc 1; `SUPABASE_SERVICE_KEY` → rc 0; `docker-compose.yml:31`; runbook L26 → **U11 real**.
- P1 spot checks: agents `22 9`; ONBOARDING L4/L37 "21"; `grep -cE '^\s*check_[a-z_]+ ' scripts/health_check.sh` → 21 vs "24" at README:267, ONBOARDING:148/720; FeatureViews 11, goldstd mentions in 05 → 0, 05:382 "9 feature views, 48 features", ONBOARDING:347 9/48, README:446 10/48; `free -g` → 15, `nproc` → 8 vs "32 GB" at ARCHITECTURE:86/1133, ONBOARDING:529; beat entries 28; route files 35 vs "33" at ARCHITECTURE:356; x-common-env 39 forwarded / 23 missing from .env.example; `- monitoring` ×7, mentions outside docker/README → 0; "44 KPI" at ONBOARDING:232/734, ARCHITECTURE:373/1345; `DELIBERATELY NOT path-filtered` in both required workflows vs README:235 / ONBOARDING:387; `supabase-network external: true`; `.env.example:25` port 54322 vs `docker port supabase-db` → 5433; `docker-compose.monitoring.yml` not in deploy.yml; alertmanager webhook route 0 hits; `app.add_middleware(` → 8; nginx zones `/etc/nginx/nginx.conf:48-50` api 10r/s, general 100r/s, copilot 30r/s; `tests/unit/test_memory/test_sentinels/` absent (8 `test_sentinel*.py` files); `src/api/middleware/jwt_auth.py` absent; `routing_authority` 0 in chat.md / 6 in copilotkit.py; routes.tsx `routeConfigs` L69 + `routes` L285 + 31 lazy imports; `Dockerfile.feast` FROM 0.43.0 vs "Feast 0.58.0" at 05:3/99; mig 058 `REVOKE ALL` ×2 vs 07:752-753; `RAG_BASE = '/v1/rag'`; `is_adaptive_criteria_on` 0 hits, `_adaptive_criteria_enabled` at :185; `--enable-mlflow`/`--include-bentoml` 0 hits vs SYNTHETIC_DATA:577/583; ADRs end at 012, the three ADR-013/014/015 source modules exist; `scripts/maintenance/` 8 scripts + live `/etc/cron.d/e2i-maintenance`, 0 doc mentions; 00-INDEX names none of the four extra docs/data files; kpi_history / drift_qualifying_features / hcp_brand_adoption / kpi_query_registry / validation_outcomes → 0 hits in docs/data 02 and 03; 06:132 "caps at 4 parameters", 06:960 "at most 4 positional params" vs mig 120 "cap 4 -> 6"; frontend-serving-flip has no completion marker; worker limits 1.5G/4G/3G, heavy replicas 0 vs ARCHITECTURE:135 "2 GB"; `_ADAPTIVE_THINKING_PREFIXES` at llm_factory.py:94 vs one "thinking" mention in the LLM doc; `.env` defines `SUPABASE_DB_URL` (count 1) vs migrations.md:31; live `GOTRUE_SMTP_HOST=supabase-mail` vs gotrue-smtp.md Status; live `/mlflow/` block has `auth_basic` (:158).
- PR denominators: `git log --merges --first-parent --since=<from>T00:00:00 --until=<to>T23:59:59 --format=%s main | grep -c 'Merge pull request'` → 07-20..07-31 = 78, 08-01..08-31 = 232, 09-01..09-07 = 37.

**Source:** workflow `wf_3bca4bc3-0f6` (8 auditors + 4 sweeps + critic; 359 findings, 151
PR-coverage rows, 13 agents, 630 tool calls). Per-agent raw returns:
`~/.claude/projects/-home-enunez-Projects-e2i-causal-analytics/4ebd5c32-e943-490a-9686-4f704e1a9352/subagents/workflows/wf_3bca4bc3-0f6/journal.jsonl`.

---

## Coverage, exclusions and known gaps

- **Not run (by policy on the droplet):** whole-tree mypy / pytest; any compose `up`, deploy, or
  DB write. Live checks were read-only (`docker ps`, `docker exec … printenv`, `docker port`,
  nginx files, `/etc/cron.d`, `free`, `nproc`).
- **Not verified:** runtime figures quoted in docs (SHAP latency SLAs, "~20 min" batched suite,
  "~1.6 GiB" mypy, Feast "<1 ms/<50 ms"); live database state (all schema claims were checked
  against DDL/migrations, not `\d`); the DO outbound-SMTP block (2026-06 measurement, re-probe
  one-liner in U14); whether the ADR-010 DSPy A/B ever ran; Optum parquet shapes; whether Opik
  `@track` hooks no-op at runtime with `OPIK_ENABLED=false`.
- **Coverage note from the critic:** the 359-finding list it received was truncated mid-
  onboarding, so its "misses" were de-duplicated against the llm/runbooks/data/method auditor
  outputs by hand while assembling this ledger (the 05-FEATURE-STORE and frontend-serving-flip
  misses were genuine; the migration 130/131 miss was already partly in the data-dictionary
  auditor's [54] and is now in X7).
- **PRs in the window named in no coverage row (34):** #1303 #1315 #1328 #1329 #1455 #1467
  #1468 #1491 #1492 #1493 #1494 #1498 #1499 #1503 #1504 #1506 #1508 #1739 #1740 #1741 #1742
  #1745 #1746 #1748 #1750 #1754 #1755 #1756 #1823 #1873 #1875 #1876 #1877 #1878 — inspected by
  the critic: doc-relevant ones are folded in (#1741/#1748 → X7, #1873 → C9, #1491–#1494 → CHANGELOG
  + ADR candidate, #1750/#1754–#1756 → U7, #1823 → U1 CI table); the rest are test-only or
  internal.
- **Out of scope by design** (see header) — notably per-agent CONTRACT_VALIDATION.md files were
  updated in-PR by #1380, #1429, #1646 and are not audited here.

---

## Suggested phase-2 batching (not started — awaiting go)

Docs-only PRs skip Backend CI → sanctioned `gh pr merge --admin --merge` (never squash).
Anything touching `frontend/**`, `scripts/**`, `docker/Dockerfile`, compose files or `src/**`
fires a production deploy — keep those in the code-item PRs, not the doc PRs. `.env.example`,
`docker/README.md`, `Makefile`, `docs/**`, `CHANGELOG.md`, `README.md` are not deploy triggers.
Work from a worktree, never on the shared droplet checkout; re-run every measure command at
write time.

| Bundle | Contents | Status |
|---|---|---|
| **A · P0 operator safety** | X1, X2, X3 (DEPLOYMENT, docker/README, .env.example, ONBOARDING §2/§3 Step 2 + Step 7/§5 note/§8 + 3× make deploy, ARCHITECTURE §2.2/ADR-002/ADR-006/§9.3), U11, U4 `docker/env.example` decision | **DONE** — #1920, #1921, #1922 |
| **B · README + CI truth** | U1 remainder, X4, X12, X15, X13, X14 (README/ONBOARDING/ARCHITECTURE count rows) | **DONE** — #1920, #1924 |
| **C · Change log + decisions** | X6 (CHANGELOG Aug/Sep + README highlights), U9 (ADR-008 amendment), C4–C8 ADRs | **DONE** — #1924, #1925 |
| **D · Env-var reference** | X5 across DEPLOYMENT, docker/README, .env.example, LLM_CONFIGURATION (+ U8 remainder) | **DONE** — #1921, #1928 |
| **E · Deploy & ops runbooks** | U2 pipeline items, C1 maintenance-cron, C2 deploy-operations, C3 reseed, U12–U16 runbook fixes, U18 | **DONE** — #1921, #1922, #1927 |
| **F · Data dictionary catch-up** | X7 (U19–U25), X9 (U23 + pointers), X8 in docs/data, U26, U27, U28 | **DONE** — #1926 |
| **G · Architecture & API docs** | U6 remainder (X10, X11, middleware, routes, KG types, package map, new service sections), U17 chat.md, C9 decision outcome, C10, C11 | **DONE** — #1923, #1927, #1929 |
| **H · Method docs & frontend** | U29–U33, U34 archive, U7 | **DONE** — #1928 (frontend/README.md deferred — deploy trigger) |
| **I · Code items** | N1–N10 as small separate PRs (each needs an intent check; N1/N7 first) | **NOT STARTED** — deliberately held: each needs an intent check and several fire a production deploy |

*Last updated: 2026-09-07 (phase 1 complete; **phase 2 doc bundles A–H executed** — see Phase-2 execution record).*

---

## Phase-2 execution record (2026-09-07)

Ten lanes with globally disjoint file scopes, each in its own git worktree
(never the shared droplet checkout), each re-running this ledger's measure
commands at write time. **Where a measure disagreed with the ledger, the measure
won** — that rule fired often enough to be the main finding of phase 2.

| PR | Lane | Scope |
|---|---|---|
| #1920 | onboarding | `docs/ONBOARDING.md` |
| #1921 | deploy-env | `DEPLOYMENT.md`, `docker/README.md`, `.env.example`, `docker/env.example`, `Makefile` |
| #1922 | runbooks | `docs/runbooks/**` (+3 new) |
| #1923 | architecture | `docs/ARCHITECTURE.md` |
| #1924 | readme-changelog | `README.md`, `CHANGELOG.md` |
| #1925 | decisions | `docs/decisions/**` (ADR-013…017) |
| #1926 | data-dict | `docs/data/**` |
| #1927 | api-docs | `docs/api/**` |
| #1928 | method-docs | LLM config + method docs, `api_connectivity_review` archived |
| #1929 | wrap-up | C9 follow-through, dead links, this record |

### All five P0s closed, with proof
- **X1** — `docker compose --env-file .env.example -f docker/docker-compose.yml config -q` went **rc 1 → rc 0**. The missing `SUPABASE_POSTGRES_PASSWORD` was the sole blocker.
- **X2** — provider truth corrected in `docker/README.md` and ONBOARDING.
- **X3** — all three ONBOARDING `make deploy` blocks plus the docker/README and ARCHITECTURE copies annotated or removed.
- **U5 Step 7** — replaced with `scripts/run_migrations.sh`.
- **U11** — `SUPABASE_SERVICE_ROLE_KEY` → `SUPABASE_SERVICE_KEY`.

### Ledger claims that were WRONG and were corrected in flight
These are recorded so a future audit does not re-derive them from this file.

| Ledger row | What the ledger said | Measured truth |
|---|---|---|
| X11 | beat schedule has 28 entries, "no DLQ entry" | **29**; `monitor-dead-letter-queue` is assigned at `celery_app.py:751`, *after* the dict literal, so the ledger's regex could not see it |
| C11 | helper `_e2i_http_error`, envelope `{error: {…}}` | `_e2i_http_error` has **zero hits**; the helper is `_generic_http_error`; the envelope is **flat**, with `severity` only under debug |
| X5 | `SEGMENT_ANALYSIS_BUDGET_SECONDS`, `AGENT_COMPUTE_EXECUTOR_WORKERS`, compute-pool vars are compose-forwarded knobs | **Not in `x-common-env`** (0 hits). They are read by code but never forwarded — same class as N1 |
| U12 | the §5 verify query casts against the wrong enum | Worse: `experiment_monitor` is in **both** enums, so the check passed whether or not migration 055 applied — a **false green** |
| U24 | calendar-aligned windows are "clamped to elapsed span" | The opposite — the **full calendar period**, deliberately, so the window is valid at the period's first instant |
| U26 | `kg_mode` ∈ {off, shadow, **active**} | {off, shadow, **promoted**} |
| U21 | migration 119 adds a `validation_status` **domain** | A **CHECK constraint** — the migration's own comment explains why an enum `ADD VALUE` cannot run in its transaction |
| U32 | the cosmetic follow-up "appears done (0 hits)" | **False negative** — the string is an f-string result; `run_tier0_test.py:4240` still builds it. The note was kept, not deleted |
| U20 | the #1916 open-month skip is on `kpi_history` | The backfill's partial-month drop is `_complete_months()` and predates #1916; #1916's guard is in `walk_forward.py` |
| U25 | 2 enum rows are wrong | **4 of 6** were wrong |
| C8 | #1891 belongs to the discovery-corroboration arc | It is the wave-53 segment-HTE work. Dropped from ADR-017 |
| C9 | `discovery_guided` is a regenerated response field | **0 hits** in the generated types — it is an internal agent-state key |
| X6 | 37 September merges | **33**; four of the 37 are dated 2026-08-31 (the day-boundary artifact this ledger's own header warns about) |
| X6 | #1607, #1761, #1785, #1798, #1801 are PRs | They are **issue** numbers appearing inside PR titles. **#1804 has no merge at all** |
| X6 | #1428, #1441, #1445 are "late July" | They merged 2026-08-01, 08-01 and 08-03 |
| U8 | seven knob attributions cited as PRs | They are **issues** (incl. #1342 → **#1341**) |
| U29 | 18 concrete generator classes "add hcp_brand_adoption" | `hcp_brand_adoption_generator.py` defines no class; 18 is the count *without* it |
| U9 | ADR-010 DSPy A/B "not yet executed" (UNCERTAIN) | **It ran** 2026-07-18/19; both candidates failed pre-registered gates → a data-driven no-flip |
| U28 | `--anchor-to-now` vs frontier anchoring (UNCERTAIN) | `reseed_synthetic.sh` defaults to `--append-frontier`; `--anchor-to-now` only under `--full` |
| U30 | bare converter invocation (UNCERTAIN) | `--input`/`--output` both carry defaults — it works |

### Decisions taken
- **C9 → (b) adopted**, premise re-verified first (0 `include_in_schema` in causal/segments, 25 OpenAPI summaries, fields present in the generated types). Recorded in `docs/api/README.md` with the reversal condition. Follow-throughs: ADR-017 (#1925), CHANGELOG (#1924), ARCHITECTURE §3.4 rows (#1929).
- **U4 `docker/env.example` → 3-line pointer**, `Makefile` hint fixed.

### Deliberately NOT done
- **U7 `frontend/README.md`** — correct and ready, but `frontend/**` is a deploy trigger, so merging it rebuilds and redeploys production. Held for an explicit go.
- **Bundle I (N1–N10)** — code/config items. Each needs an intent check under REASON-BEFORE-RULES, and several fire a deploy. Two were confirmed and widened during phase 2: **N1** (`ADAPTIVE_CRITERIA` and friends are documented as rollback switches but are inert in containers — now also `SEGMENT_ANALYSIS_BUDGET_SECONDS` and the compute-pool vars) and **N7** (`e2i_agent_name` has no `cohort_profiler` value; latent only because that agent writes no memory rows — and the same enum carries `fairness_guardian` and `corpus_ingestion`, which are not in the 22-roster).

### New code defects surfaced by phase 2 (not in N1–N10)
- `.github/workflows/deploy.yml` repeats the same false "the droplet exposes no `SUPABASE_DB_URL`" claim that U12 fixed in the runbook.
- `scripts/reseed_synthetic.sh` `--dry-run` is forwarded to the loader only — the kpi backfill, weekly capture and retrain stages still write. Documented as a hazard in the new reseed runbook.
- `scripts/run_tier0_test.py:4240` still hard-codes a Kisqali deployment name and problem description (see U32 above).
- `src/agents/orchestrator/nodes/intent_classifier.py` comment names `gpt-4o-mini` as the fast tier; `llm_factory.py` says `gpt-5.6-luna`.
- `docs/api/index.html` is a `make api-docs` output but is not git-ignored (N9 confirmed).
