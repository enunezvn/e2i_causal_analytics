# H1 + H3b Backend Bug-Fix Design (2026-06-09)

Branch `fix/h1-h3b-kpi-realdata-monitor-async` off `main`@3253790e. No mocking, real results.
Design converged with Codex (gpt-5.5) + live-DB cheapest-disproof. PROD DB = docker `supabase-db`.

## H1 — Home "Total TRx (MTD)" / "HCPs Reached" tiles serve fabricated values

### Root cause (3 compounding, all verified)
1. `copilotkit.py:908` `BusinessMetricRepository(client=client)` — wrong kwarg vs `BaseRepository.__init__(supabase_client=)` (base.py:28) → TypeError → repo None → `data_source='fallback'`. (Sibling :921 agents path same bug — OUT OF SCOPE, filed.)
2. Even fixed, `get_supabase()` is the SYNC client but `BaseRepository` does `await execute()` → TypeError.
3. **PIVOT (cheapest-disproof):** the data source `business_metrics` is SYNTHETIC (`scripts/load_synthetic_data.py`, uniform-random 0–1M). Wiring tiles to it = synthetic-as-real, unbadged → WORSE. The audit's premise ("business_metrics holds real TRx") was wrong.

### Live-DB reality (NOW=2026-06-09)
- Real TRx (WS3-BI-005 = `COUNT(*) treatment_events prescription 30d brand=$1`) = **0** (treatment_events ends 2025-12-22; 30d window empty). All brands 0.
- `triggers.brand_id` = **'UNKNOWN' for all 4356 rows** → brand-specific HCPs-reached NOT computable. Total delivered/viewed reach = 543 HCPs (brand-agnostic).
- HCP coverage WS3-BI-004 = 27.3 (a ratio, NOT a count → wrong for the tile).
- The KPI **grid** already uses the real `BusinessImpactCalculator` (sync RPC `kpi_query(query_id text, params jsonb)`); the **tiles** wrongly use synthetic business_metrics + hardcoded `_FALLBACK_KPIS`.

### Fix (honest, no-mocking)
- Re-point `get_kpi_summary(brand)` OFF synthetic business_metrics and ONTO the real `BusinessImpactCalculator` (sync call inside the async handler — mirrors kpi.py:436/498/560). Map: trx_volume→WS3-BI-005, nrx_volume→006, nbrx→007, conversion_rate→business_impact_conversion_rate, market_share→WS3-BI-008 trx_share, hcp_coverage→004.
- `hcp_reach`: add a real allowlisted query `business_impact_hcp_reach` = `COUNT(DISTINCT hcp_id) FROM triggers WHERE delivery_status IN ('delivered','viewed') AND trigger_timestamp >= NOW()-30d` (brand_id is UNKNOWN → brand-agnostic reach; documented). New migration.
- DELETE `_FALLBACK_KPIS`, `_fetch_kpis_from_db`, `_get_business_metric_repository` (now dead). Intent: `_FALLBACK_KPIS` was an intentional sample fallback (commits 96e2ca24e/2662a2c04) — superseded, documented.
- `data_source`: `'database'` for real values (incl real 0), `'unavailable'` (fail-closed) on query error. NEVER fabricated.
- Harden invalid-brand path (currently returns non-conforming `{"error":...}` dict).
- Honest caveat documented: KPIs are 0 because source data is stale to 2025-12-22 (DATA-pipeline issue, not a code bug); triggers lacks brand attribution.

### FE contract (do not break): `metrics.hcp_reach?: number` (home-stats.ts), count tile (Home.tsx:725-750). Shape unchanged.

## H3b — ExperimentRecommendations panel: honest-looking empty masks a crash

### Root cause (verified)
1. `health_checker.py:45` `await get_supabase_client()` awaits the SYNC factory (factories.py:674) → TypeError; async is `get_async_supabase_client()` (:748). Same await-sync bug in ALL 4 DB nodes: health_checker, fidelity_checker, interim_analyzer, srm_detector.
2. `trigger_experiment_monitoring` (experiments.py:1217) never inspects `result.errors` → HTTP 200 empty masks the crash. `MonitorResponse` has no `errors` field. 621 REAL running ml_experiments exist (2026-04-25..06-09).
3. `_get_mock_experiments` path (health_checker.py:63-70/150-181) fabricates experiments when client is None — no-mocking concern (test scaffold, commit fc9fae4b; test_integration.py:607-619 depends on it).

### Fix
- `_get_client` → `await get_async_supabase_client()` in all 4 monitor nodes.
- Add `errors: list[str]` to backend `MonitorResponse` + populate from `result.errors`; harden the 4 nodes' catch-all-returns-[] to RECORD errors, not swallow. FE `MonitorResponse` type gains `errors`; ExperimentRecommendations renders a non-empty error state.
- Fail-close the mock path behind explicit `allow_mock=False` (established #816 pattern); update test_integration.py:607-619 to opt in explicitly.

## Out of scope → FILED as follow-ups (not silently left)
- 7 other `await get_supabase_client()` sync-client bugs: feedback_loop_tasks.py:148/418, ab_testing_tasks.py:356/534/946/1033, drift_monitoring_tasks.py:404, optuna_optimizer.py:762.
- Agents path async bug: `_get_agent_registry_repository`/`_fetch_agents_from_db` (copilotkit.py:921/1328).
- FE dead monitor request fields check_srm/check_enrollment/check_fidelity (experiments.ts) ignored by backend.

## TDD (faithful, real-DB, no mocks) — the mocks that HID these bugs
- H3b `test_get_client_lazy_loads` mocked `get_supabase_client` as AsyncMock → hid the await-sync bug. Replace with real-factory/real-DB tests.
- H1 real-DB: get_kpi_summary never returns `_FALLBACK_KPIS`; data_source∈{database,unavailable}; trx from real RPC; hcp_reach is an int count ≠ coverage fraction.
- H3b real-DB: monitor count == `SELECT count(*) ml_experiments status=running`; result.errors surface to API (not 200-empty); no mock exp IDs in prod.
