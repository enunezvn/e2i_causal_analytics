-- Migration 159: enum values for retrain candidates and register-only deployments
-- (#2310, #2308; owner decision 2026-09-28: apply the #2310 recommendation).
--
-- WHY: a retrain registers its new model row at stage 'staging'. The gold-standard
-- *_goldstd_lr_v1 v1.0 reference rows sit at 'staging' too (collision guard, 7a8bc9e0e,
-- src/mlops/gold_standard_eval/cohort_deployer.py), so every stage-filtered reader
-- (stage IN ('production','staging')) sees an unreviewed retrain as a peer of the model it
-- retrains. A retrain now registers at its own stage, 'candidate': not reviewed, not served.
-- Moving a candidate to staging/production is a later, separate decision.
--
-- A register-only deploy (no endpoint) records its ml_deployments row as 'active' today
-- (#2308), so the table claims a live deployment that does not exist. The row is kept (the
-- table exists to record deployments, 214890aa3) under a new non-active status,
-- 'registered': the version was registered, nothing serves it.
--
-- SHAPE: only the two enum additions live here. Postgres forbids using an enum value in the
-- transaction that added it, and the runner applies this file un-wrapped (it detects the
-- enum-extension statement), so each statement commits on its own. Migration 160 (the
-- columns and the backfill) then runs wrapped and atomic, after these values are committed.
-- Values are appended (no BEFORE/AFTER): nothing orders by these enums (checked 2026-09-28).
--
-- IDEMPOTENT: IF NOT EXISTS. A value cannot be removed once added (see the rollback file).

ALTER TYPE model_stage_enum ADD VALUE IF NOT EXISTS 'candidate';

ALTER TYPE deployment_status_enum ADD VALUE IF NOT EXISTS 'registered';
