# Synthetic-Causal 11-Gate Certification — 2026-06-13

**Issue:** #898 — re-run the 11-gate certification; the standing 11/11 (2026-06-11) predated the #883 remediation family (#884/#885/#886) that landed 2026-06-12.

**Result: 11/11 PASS, 0 FAIL.**

- **Code:** `695b6cc5a3f9a2eea886f0e049da59c1b1d2975b` (current main; post-#884/#885/#886 enum remaps + resolvers + memory wiring; includes #895 ETL provenance inheritance via #910). Delta to repo HEAD `8a01923f` is only #912 (#891 NaN-sanitize) + #917 (test-hygiene) — neither gate-relevant on clean seed=42 synthetic-gold data.
- **Substrate:** live docker-Supabase synthetic-gold (the 2026-06-11 load; per-brand synthetic TRx all > 0 under NOW()−30d).
- **Harness:** `scripts/validate_synthetic_causal.py --all` (E2I_DB_INTEGRATION=1, LOKY_MAX_CPU_COUNT=1). Real agents, real RPCs, real CausalForestDML/CausalML — no mocks.
- **Corroboration:** gates 1,2,4,5,6,8,9 independently re-passed inside the deployed `e2i_api` container against the same live DB (gates 3/7/10/11 require gitignored build artifacts the read-only container image can't host).

## Gate ladder

| # | Gate | Key measure | Verdict |
|---|------|-------------|---------|
| 1 | DATE-FRESHNESS | Remi 14292 / Kisqali 13392 / Fabhalta 14986 TRx (NOW−30d); conv 0.554 | ✅ |
| 2 | KPI→DASHBOARD | data_source=database; 5 non-zero synthetic KPIs | ✅ |
| 3 | ATE/CATE RECOVERY | recovered 0.1938 vs TRUE_ATE 0.1737 (≤0.10 tol) | ✅ |
| 4 | TRIGGER EFFECTIVENESS | uplift +0.236 (treat 0.376 > control 0.304) | ✅ |
| 5 | gap_analyzer | 10 opportunities, TAV $520,809 | ✅ |
| 6 | heterogeneous_optimizer | designed CATE ordering recovered; ATE 0.098; het 0.310 | ✅ |
| 7 | prediction_synthesizer | model_agreement 0.973, 2 real models | ✅ |
| 8 | resource_optimizer | POST /optimize → solver=optimal (200) | ✅ |
| 9 | PROVENANCE | 0 untagged on every taggable table; real-mode TRx excludes synthetic | ✅ |
| 10 | 4-COHORT × 3-BRAND × AGENT | initiation/persistence +ATE, discontinuation −ATE across all 12 cells | ✅ |
| 11 | CHAT-PATH e2e | routed_to_het=True, het_success=True, 5 CATE segments, overall_ate 0.097 | ✅ |

## #898 lineage observations (both confirmed green)

- **tier0 episodic writes landing** — gates 5/6/10/11 exercise the agent paths whose episodic enum values come from `database/memory/039`; all green.
- **business_metrics 0 untagged after ETL** — gate 9 PROVENANCE: `failures: []` across all taggable tables, `trx_real_mode: 0` (real KPI excludes synthetic by default). Confirms #895 ETL provenance inheritance holds.

Run: local, 2026-06-13T11:40Z. Authoritative full-ladder source = this harness's --all output (per the script's runbook footer).
