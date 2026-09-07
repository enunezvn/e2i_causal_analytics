# Changelog

Notable changes to the E2I Causal Analytics platform.

**Scope and honesty note.** This file starts in July 2026 (DOC-AUDIT-202607). The repository carries **no platform release tags** — `git tag` lists exactly three, none of them a release: `vocabulary-v5.0.0` and `vocabulary-pre-consolidation-v5.0.0` (domain-vocabulary snapshots) and one `archive/…` marker. The `pyproject.toml` version (4.2.1) predates this file and does not correspond to a tagged release, so every section below is effectively unreleased. Until a tagged-release process exists, entries are grouped by month (newest first) at PR-train granularity — the authoritative fine-grained history is `git log` and the merged-PR record. Earlier months are deliberately **not back-filled**.

Every PR number below was verified against `git log --merges --first-parent main` at write time. Numbers that appear inside a PR title (`fix(x): … (#1798)`) are *issues*, not PRs, and are named as such.

## 2026-09 (in progress)

### Added
- **Guided causal-discovery hardening** (#1878, #1879, #1883, #1886–#1890): a structural-recovery benchmark for guided DAG generation (#1878); a prior-asserted DAG is no longer reported as discovered (#1879); the gate scores bootstrap **edge stability** rather than self-agreement on single-algorithm runs (#1883); an FCI latent-confounding diagnostic that annotates and never gates (#1886); confounder wiring split into structural-prior and adjustment-guarantee channels (#1887); a corroboration-gated latent warning with base-rate observability (#1888); agent-analyze wiring so guided discovery reaches the graph and runs reach MLflow (#1889); role-aware covariate validation at the API boundary (#1890). #1885 is the negative result: the chisq-selection premise was measured, rejected, and the verdict recorded.
- **Causal-discovery UX** (#1898, #1899): discovery-question multiselect and a cooperative Cancel on `/causal-analysis`; runs orphaned by an API restart now end `failed` instead of spinning to the 8 h TTL.
- **Copilot suggestion pills from a capability catalog** (#1900, #1903, #1908–#1910, #1912, #1913, #1915, #1919): pills are generated from a declared capability catalog and filtered by a validator, with a page-summary readable; the follow-up train removed pills the platform cannot serve (live experiment status, outcome-as-KPI trends, patient axes on KPIs whose calculator does not bind them).
- **`kpi_history` brand axis** (#1896) for trigger family, conversion rate and weekly capture (Time-Series brand selector / Compare Brands).
- **Segment Analysis option definitions** under the dropdowns, with unambiguous commercial-arm labels (#1893), and the curated treatment/outcome labels applied globally across causal analysis and the discovery summary (#1895).

### Fixed
- **Segment HTE estimand** (#1891, #1892): the treatment is kept out of `X`, clinical axes run as their 0/1 contrast, 401s replay after refresh; cross-library ordering agreement became CI-aware with a chance floor.
- **Model-performance trend** (#1916): sampling-aware trend classification (analytic SE for accuracy/recall/auc_roc, t-adjusted empirical spread) and the open walk-forward month is skipped — a partial-month fold no longer reads as "degrading".
- **Agent health metrics** (#1902): the success rate counts runs that raised, every fail-closed path writes the error row the readers count, and detected patterns age out after 30 days.
- **Investigator reward** (#1905): a run graded `sufficient` is never scored negative.
- **`/system-health` Overall Health score** renders to one decimal (#1897).
- **Insight enumeration markers** are treated as prose structure, not numeric claims (#1881).
- **Migration 132** — persistence outcome column comments now describe the shipped data (#1894).

### Changed
- **Security bumps** (#1882, #1884): tornado 6.5.8 with a temporary transformers CVE allowlist, then transformers 5.10.1 + huggingface-hub 1.5.0 clearing 12 pip-audit allowlist rows.
- **Documentation truthing (DOC-AUDIT-202609 phase 2)** (#1920–#1923): ONBOARDING, DEPLOYMENT + `docker/README.md` + `.env.example`, ARCHITECTURE, and the runbooks set — including three new operator runbooks (maintenance cron, deploy operations, synthetic reseed).

## 2026-08

### Added
- **Real-pipeline RAGAS gate + RAGAS-fed GEPA** (#1491–#1494, #1499, #1508, #1510–#1513, #1519, #1521, #1529): the fixture evaluation was renamed a judge-drift sentinel and a real-pipeline gate stood up beside it (#1492); RAGAS now fails loud, and unmeasured is never 0.0 (#1491); per-turn scores persist into the self-improvement schema (#1493); a RAGAS-backed GEPA leg optimizes the RAG synthesis prompts, with container wiring and DB-sourced feedstock (#1494, #1512, #1513); GEPA artifact versioning became numeric, collision-safe and atomic (#1499, #1508, #1510); the migration-023 GEPA persistence tables were wired (#1529); the chatbot DSPy optimization queue gained a drainer (#1521); the combined-score COALESCE trap and a decomposed ratchet landed together (#1511); `_get_phase_weights` was retired once the wired path had no phase axis (#1519).
- **KPI region axis, calendar windows and substrate provenance** (#1539, #1540, #1554, #1570, #1571, #1579, #1647; migrations 126–128): `kpi_history` region backfill and coverage lattice with Time-Series and chat region scope; chat region vocabulary with provenance-gated region charts; `parse_window` accepting natural calendar phrases; region variants for the brand-specific family; a brand+region joint leg for conversion rate; and a declared measurement basis so cross-substrate scales cannot read as comparable.
- **ROI temporal band and scoped headline** (#1525, #1535, #1537; migrations 124/125): the Monte Carlo band surfaced in chat with `roi_estimator` uncertainty measured, a per-slice trailing-12-month variability band, and a headline that honors brand/region scope.
- **Chat cold-start warming and worker preload** (#1474, #1483, #1589): a fail-open startup warm task, synthetic-LLM warm legs for classify and the RAG rewriter, and the worker import-wedge fix (preload enabled, master heavy-leaf warm, SIGABRT stack dumps, fork safety).
- **Knowledge-graph Layer 2** (#1619, #1621; issue #1607): activated with a real drug–disease signal, and the committed cache packaged into the API image so it cannot exist only on one laptop.
- **Clinical context on the causal drill-down** (#1764, #1812): the clinical panel is tied to the analysis being interrogated, with narrative distillation.
- **Copilot AG-UI hardening** (#1724, #1750, #1755, #1818): the empty-delta stream kill (P0) plus a brand-filter state channel; the dashboard and Causal Analysis brand filters single-sourced through the copilot provider; AG-UI readables reaching the chat graph.
- **SSE survives the proxy** (#1664, #1674): the silent window is bounded so nginx cannot sever a completing turn, and nginx **gzip** — not `proxy_buffering` — was identified as what ate the AG-UI turn.
- **Per-turn routing authority named in `dispatch_info`** (#1596).
- **Observability made opt-in, and the maintenance cron given an alarm** (#1799, #1800, #1802, #1803, #1805–#1807, #1810; issue #1798): prometheus, alertmanager, node-exporter, postgres-exporter, loki, promtail and grafana moved behind the `monitoring` compose profile (#1806); after the droplet's `/etc/cron.d/e2i-maintenance` stopped silently for eight weeks, a freshness check keyed on completed-run stamps (#1799, #1802) was wired into `health_check.sh` (#1803, #1805) and given an unattended GitHub Actions caller that files a tracking issue (#1807, #1810).
- **Graph emptiness sentinel and FalkorDB persistence** (#1759, #1771; issue #1761): the data dir points at the `/data` volume, and a beat-scheduled sentinel self-heals a wiped knowledge graph.
- **Segment-analysis budgeting and refusals** (#1829, #1836, #1844, #1845): unmodeled treatment→outcome pairs are refused at POST time unless `allow_unmodeled`; a heavy-compute slot budget with in-flight dedup; poll ceilings that keep the `analysis_id` and offer "Keep waiting" instead of re-submitting.
- **Three-state refutation verdicts** (#1869, #1871): per-test `passed` / `warning` / `failed` surfaced through API and UI, with warning-aware gate phrasing. Gate semantics unchanged.
- **Navigation** (#1824, #1866): the documentation page became "How E2I Works" and was promoted under Home (naming the 4 predictive cohorts and 8 intervention channels); "AI Insights" became "Executive Insights".
- **Synthetic brand×region execution matrix** with anchored events, so gap analysis carries brand-specific geography (#1849).
- **Celery beat SSOT and persisted state** (#1653, #1777): beat state persists on a volume, daily entries moved to wall clock, and the boot-time overwrite of the schedule SSOT was removed.
- **Stateful multi-turn clarification ask-back on `/chat/stream`** (#1441).

### Changed
- **Deploy pipeline** (#1428, #1436, #1480, #1782, #1789, #1792, #1794; issues #1784/#1785): production deploys serialized by a concurrency group; the newest *built* ancestor is deployed with a downgrade floor rather than blindly `origin/main`; a fail-loud image-drift check with an allowlist; an image guaranteed for the sha the droplet resolves, with `PREV_SHA` anchored on what is actually running; a deploy will not rewind the branch a human left checked out; triggers cover every production image input; gates run before cleanup, with the prune as a separate step carrying its own verdict.
- **Required backend contexts are conditional, not path-filtered** (#1445) — a docs-only or frontend-only PR no longer leaves a required check Pending.
- **Nightly slow tests moved 07:00 → 05:00 UTC** off EBI's measured morning instability, with upstream-transient reds routed to a rolling issue instead of the red alarm (#1816, #1823).
- **Region/brand enum normalization consolidated** into one shared module (#1509).
- **`gap_analyzer` time_period grammar** fails closed on unknown labels and surfaces the resolved window (#1837).
- **RAG chain performance** (#1490, #1522): an async `retrieve_rag` chain with hop economics and chain-internal spans; a predict-only rewriter and an empty-board decider skip (two levers adopted, two rejected with data).
- **Feedback optimizer gated on trainable supply, not defect yield** (#1677).
- **App-wide mutation retry default 1 → 0** so non-idempotent mutations are never replayed (#1852).

### Fixed
- **Markdown tables render with real columns** in the copilot sidebar (#1444).

## 2026-07

### Added
- **LLM factory tier system + July model refresh** (#1274–#1276): central `MODEL_MAPPINGS` with fast/standard/reasoning tiers per provider (claude-haiku-4-5 / claude-sonnet-5; gpt-5.6-luna / gpt-5.6-terra), `LLM_MODEL` override, temperature-compatibility handling, versioned pricing; plus a dead model-ID sweep. See ADR-009.
- **Feature-importance stability sampling** (#1268, #1270, #1272, #1277; migration 109): adaptive random SHAP sampling with a statistical stopping rule; the gate certifies the *displayed* covariate-group ranking (user-approved estimand change), with provenance persisted under `__sampling__` and a "covariate ranking stable" badge. See ADR-011.
- **RCT baseline-ANCOVA efficiency adjustment** (#1217, #1219; migration 106): opt-in `adjust_baselines` for randomized questions, `adjustment_type="efficiency"`, E-value gate skipped for design-declared randomized treatments. See ADR-012.
- **Admin LLM observability** (#1206/#1207, #1213–#1215; migration 104): `llm_usage_events` dual-hook usage tracking (model, tokens, cost, latency, user) surfaced at `/admin` → Observability.
- **Copilot learning signals** (#1240, #1241): the copilot chat path now also writes learner-visible `learning_signals` (`dspy_signal`) rows; feedback-learner demo pipeline (backfill → signals → pattern → gated proposal).
- **KPI trend/segment charting** (#1269; migration 110) and **biologic/IgE KPI axes** (#1223; migration 108, Remibrutinib-only, fail-closed for other brands).
- **`frontend/README.md`** (#1280) and **`docs/LLM_CONFIGURATION.md`** (#1279) — net-new documentation.
- **Claims-lag nowcast plane** (#1306–#1308): arrival-plane stamps on treatment events, a chain-ladder `/history/nowcast` endpoint, and a provisional tail with an opt-in nowcast overlay on Time Series.
- **WS2 trigger-KPI truth redefinition** with brand variants and the COMM-ARMS phase-4 `trigger_accepted` arm (#1300); goldstd holdout enlargement 5%→10% (#1305).
- **Flint-compiled charts for the whole KPI registry** in chat (#1383).

### Changed
- **LLM provider split** (#1278): factory lanes flipped to Anthropic (`LLM_PROVIDER=anthropic`; sonnet-5/haiku-4-5), DSPy pinned to `openai/gpt-5.6-terra`; compose now forwards `LLM_PROVIDER`/`LLM_MODEL`/`DSPY_LM_MODEL`. See ADR-010.
- **DSPy default model** gpt-4o → `openai/gpt-5.6-terra` (#1275) — the highest-volume LLM surface leaves 2024-era defaults.
- **KPI conversion-rate routing** (#1271, #1273; migration 111): brand/segment/line/window now route correctly (was region-only, silently flat); TRx share windowed; bare "last year" parses. **`kpi_query` positional-param cap 4→6** so region and an explicit window can co-bind (#1396; migration 120).
- **Causal clinical context de-verbosed + job TTL 1h→8h** (#1235; intentionally reverses #1227's params).
- **Refutation cost profile** (#1422): subsample-for-refutation, critical gates first, and a measured SLA raise.
- **Documentation truthing waves 1+2** (#1279, #1281): README/DEPLOYMENT/.env.example provider truth, migrations runbook (PROD=AUTO), data dictionary caught up through migration 111.

### Fixed
- **July page-review fix series** (selection): gap-analysis market-share unit bug (#1238) and Fabhalta zero-opps floor (#1237); scenario-comparison degenerate scatter (#1239); frontier-append FK + NaN crashes (#1234, #1236); audit-chain fabricated "0% confidence" (#1205); ml_experiments status lifecycle (#1197/#1199); waterfall negative-width blank chart (#1268).
- **Ultra-review findings #1256–#1263** fixed via train #1264/#1265/#1267.
- **Honest-CI defect in DML wrappers** (via the #1188 arc): `ate_inference`-based intervals replace a fallback ~50× too narrow; observational CIs widened intentionally.
- **Chat-RAG chunk corpus** populated in the `text-embedding-3-small` space (#1374).

## Pre-2026-07 (not back-filled)

History before July 2026 — including the June remediation arcs (21-agent audit, T-goal series, gold-standard eval, deploy/OOM mechanics) and everything earlier — lives in `git log`, the merged-PR record, and `docs/ARCHITECTURE.md` §8 (ADR-001–008).
