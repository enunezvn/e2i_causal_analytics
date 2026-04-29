# PR #2 (`feat/tier0-feast-infra`) — Pre-Migration Snapshot

Captured: **2026-04-28 01:48 UTC** (host droplet, single-node dev=prod stack).

This document is the pre-flight snapshot required by `.claude/plans/tier0_close_out_3pr.md` § "Pre-flight (must complete before touching any file)" before applying migration `033_feast_canonical_schema.sql`. It captures DB state immediately before the verification phase so any post-migration regression is unambiguously attributable to PR #2.

## Branch state

- Branch: `feat/tier0-feast-infra` @ `067a494` (HEAD).
- Cut from `origin/main` @ `f4fef17` (PR #1 merge commit).
- 14 commits ahead of `main`. All sub-blocks 6B-infra-1 → 6B-infra-5 landed with reviewer-approved fix-ups.

```
067a494 test(feast): add entity proto walker for schema-deep idempotency (6B-infra-5 fix-up)
0100916 test(feast): schema-deep idempotency check via proto-byte diff (6B-infra-5)
19c2c42 chore(infra): apply 6B-infra-4 code review fixes (I-1, I-2, I-3, I-4, C-1 deferral docs)
b728fbe chore(infra): worker network + shared Feast registry + version-pin enforcement (6B-infra-4)
2c95cf8 chore(feast): boy-scout ruff cleanup + parametrise composite-key test (6B-infra-3 fix-up)
ec4a1ca refactor(feast): point data_sources at canonical tables; drop bridging views (6B-infra-3)
3cec0a3 fix(db,etl): drop NOT NULL on territory aggregate cols; ETL writes explicit NULL (6B-infra-2c fix-up)
8209e75 feat(etl): real territory rollup; null over random (6B-infra-2c)
ff04287 refactor(etl): extract shared helpers to _common.py + tighten gap-CTE predicate (6B-infra-2b fix-up)
ba7bdd4 feat(etl): per-patient adherence/refill/gap derivation (6B-infra-2b)
04ba8b6 fix(etl): retry psycopg2.OperationalError; md5-hash metric_id to fit VARCHAR(50) (6B-infra-2a fix-up)
efb2d0b feat(etl): per-HCP business metrics rollup with idempotent upsert (6B-infra-2a)
fb90b90 fix(db): set ON DELETE SET NULL on business_metrics.hcp_id FK (6B-infra-1 fix-up)
3c8fda4 feat(db): canonical schema migration 033; drops Feast bridging views (6B-infra-1)
```

## Container health

```
e2i_feast       Up 36 hours (healthy)
e2i_api_dev     Up 36 hours (healthy)
supabase-db     Up 4 weeks (healthy)
```

The compose stack has NOT yet been restarted with the 6B-infra-4 changes (worker network attachment, shared `feast_registry` volume, entrypoint version-pin). Restart is part of the verification step.

## Row counts (canonical tables)

| Table | Row count |
|-------|-----------|
| `hcp_profiles` | 550 |
| `triggers` | 4 356 |
| `patient_journeys` | 2 700 |
| `business_metrics` | 4 128 |
| `territory_metrics` | 18 |
| `feast_business_metrics_seed` (legacy seed table; will be dropped by migration 033) | 1 650 |

Of `hcp_profiles`, **500 rows have `territory_id IS NULL`** — migration 033 will backfill these with the `'UNASSIGNED'` sentinel as part of section 033.1 row backfill.

## Feast bridging views (will be dropped by migration 033 §033.6)

```
public | feast_business_metrics_source | view | postgres
public | feast_hcp_profile_source      | view | postgres
public | feast_patient_journey_source  | view | postgres
public | feast_trigger_response_source | view | postgres
```

(`feast_business_metrics_seed` is a TABLE, not a view; dropped via `DROP TABLE IF EXISTS` in §033.6.)

## Schema migrations applied (head)

```
032_feast_offline_views_extended.sql | 2026-04-26 11:21:36+00
031_feast_offline_views.sql          | 2026-04-26 10:09:52+00
023_extend_agent_name_enum.sql       | 2026-04-26 11:21:36+00
022_shap_nullable_model_registry.sql | 2026-04-26 11:21:36+00
021_add_agent_context_column.sql     | 2026-04-26 11:21:36+00
020_add_patient_causal_columns.sql   | 2026-04-26 11:21:36+00
009_hpo_pattern_memory.sql           | 2026-04-26 11:21:36+00
008_agentic_memory_schema.sql        | 2026-04-26 11:21:36+00
```

`033_feast_canonical_schema.sql` is NOT yet in `schema_migrations`. Verification will assert post-apply that `033_feast_canonical_schema.sql` row exists with a fresh `applied_at` timestamp.

## Backup path (taken before any migration runs)

`pg_dump` of the public schema:

```
/home/enunez/Projects/e2i_causal_analytics/backups/postgres/pre_033_20260428_014853.sql
```

- Size: **16 MiB** (72 899 lines).
- Format: plain SQL.
- Captured via `docker exec supabase-db pg_dump -U postgres --schema=public --format=plain postgres`.
- Retained for at least the duration of the PR #2 review cycle.

Restore command (if rollback needed):

```bash
docker exec -i supabase-db psql -U postgres < backups/postgres/pre_033_20260428_014853.sql
```

NOTE: `scripts/backup_data_stores.sh` covers Redis + FalkorDB only (not Supabase Postgres) — used `pg_dump` directly per the plan's intent.

## Pre-existing carve-outs (NOT in scope for PR #2)

- 13 mypy errors in unrelated modules.
- 76 ruff errors in `scripts/run_tier0_test.py`.
- 8 pre-existing pytest failures around Redis auth in `_check_redis_service`.
- ~22 pre-existing ruff errors in `feature_repo/features/*.py` (touched by 6B-infra-3's boy-scout cleanup of `data_sources.py` only; the others are out of scope).
- 6 pre-existing format issues across `tests/integration/*.py` files (touched by 6B-infra-5 only for the 3 modified files; the rest are out of scope).

## Outstanding verification steps (require user approval before proceeding)

The remaining PR #2 verification steps are **destructive** in the dev=prod sense (single droplet, shared production data) and per auto-mode require explicit user confirmation:

1. **Apply migration 033** (`docker exec supabase-db psql -U postgres -f /path/to/033_feast_canonical_schema.sql` or via existing migration runner).
2. **Restart docker compose stack** (so worker_light / worker_medium / scheduler pick up `supabase-network`; so feast container picks up the version-pin assertion at startup).
3. **Run all 3 ETLs** at least once: `business-metrics-per-hcp-rollup`, `patient-adherence-rollup`, `territory-metrics-rollup`.
4. **Run live Feast test suites** with `FEAST_INTEGRATION=1` from inside `e2i_api_dev`.
5. **Run tier-0 e2e** smoke tests (default + adverse).
6. **Run `bash scripts/run_tests_batched.sh`** for the full regression sweep.

Each of these must succeed before opening PR #2.
