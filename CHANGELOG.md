# Changelog

Notable changes to the E2I Causal Analytics platform.

**Scope and honesty note.** This file starts in July 2026 (DOC-AUDIT-202607). There are no release tags before this point; the `pyproject.toml` version (4.2.1) predates this file and does not correspond to a tagged release. Until a tagged-release process exists, entries are grouped by month (newest first) at PR-train granularity — the authoritative fine-grained history is `git log` and the merged-PR record. Earlier months are deliberately **not back-filled**.

## Unreleased / 2026-07

### Added
- **LLM factory tier system + July model refresh** (#1274–#1276): central `MODEL_MAPPINGS` with fast/standard/reasoning tiers per provider (claude-haiku-4-5 / claude-sonnet-5; gpt-5.6-luna / gpt-5.6-terra), `LLM_MODEL` override, temperature-compatibility handling, versioned pricing; dead model-ID sweep (62 swaps). See ADR-009.
- **Feature-importance stability sampling** (#1268, #1270, #1272, #1277; migration 109): adaptive random SHAP sampling with a statistical stopping rule; the gate certifies the *displayed* covariate-group ranking (user-approved estimand change), with provenance persisted under `__sampling__` and a "covariate ranking stable" badge. See ADR-011.
- **RCT baseline-ANCOVA efficiency adjustment** (#1217, #1219; migration 106): opt-in `adjust_baselines` for randomized questions, `adjustment_type="efficiency"`, E-value gate skipped for design-declared randomized treatments. See ADR-012.
- **Admin LLM observability** (#1206/#1207, #1213–#1215; migration 104): `llm_usage_events` dual-hook usage tracking (model, tokens, cost, latency, user) surfaced at `/admin` → Observability.
- **Copilot learning signals** (#1240, #1241): the copilot chat path now also writes learner-visible `learning_signals` (`dspy_signal`) rows; feedback-learner demo pipeline (backfill → signals → pattern → gated proposal).
- **KPI trend/segment charting** (#1269; migration 110) and **biologic/IgE KPI axes** (#1223; migration 108, Remibrutinib-only, fail-closed for other brands).
- **`frontend/README.md`** (#1280) and **`docs/LLM_CONFIGURATION.md`** (#1279) — net-new documentation.

### Changed
- **LLM provider split** (#1278): factory lanes flipped to Anthropic (`LLM_PROVIDER=anthropic`; sonnet-5/haiku-4-5), DSPy pinned to `openai/gpt-5.6-terra`; compose now forwards `LLM_PROVIDER`/`LLM_MODEL`/`DSPY_LM_MODEL`. See ADR-010.
- **DSPy default model** gpt-4o → `openai/gpt-5.6-terra` (#1275) — the highest-volume LLM surface leaves 2024-era defaults.
- **KPI conversion-rate routing** (#1271, #1273; migration 111): brand/segment/line/window now route correctly (was region-only, silently flat); TRx share windowed; bare "last year" parses.
- **Causal clinical context de-verbosed + job TTL 1h→8h** (#1235; intentionally reverses #1227's params).
- **Documentation truthing waves 1+2** (#1279, #1281): README/DEPLOYMENT/.env.example provider truth, migrations runbook (PROD=AUTO), data dictionary caught up through migration 111.

### Fixed
- **July page-review fix series** (selection): gap-analysis market-share unit bug (#1238) and Fabhalta zero-opps floor (#1237); scenario-comparison degenerate scatter (#1239); frontier-append FK + NaN crashes (#1234, #1236); audit-chain fabricated "0% confidence" (#1205); ml_experiments status lifecycle (#1197/#1199); waterfall negative-width blank chart (#1268).
- **Ultra-review findings #1256–#1263** fixed via train #1264/#1265/#1267.
- **Honest-CI defect in DML wrappers** (via the #1188 arc): `ate_inference`-based intervals replace a fallback ~50× too narrow; observational CIs widened intentionally.

## Pre-2026-07 (not back-filled)

History before July 2026 — including the June remediation arcs (21-agent audit, T-goal series, gold-standard eval, deploy/OOM mechanics) and everything earlier — lives in `git log`, the merged-PR record, and `docs/ARCHITECTURE.md` §8 (ADR-001–008).
