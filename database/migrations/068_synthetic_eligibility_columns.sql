-- 068_synthetic_eligibility_columns.sql  (Shard 04 M5)
-- Add the brand-specific eligibility columns that cohort_constructor configs read
-- (src/agents/cohort_constructor/configs.py required_fields). Nullable so pre-existing
-- real rows are unaffected; synthetic rows populate them per brand. Idempotent:
-- ADD COLUMN IF NOT EXISTS. primary_diagnosis_code already exists on the table.
-- No script-level BEGIN/COMMIT — scripts/run_migrations.sh owns the outer txn via
-- psql --single-transaction (test_migrations_no_inner_txn).
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS urticaria_severity_uas7     integer,
    ADD COLUMN IF NOT EXISTS prior_antihistamine_therapy boolean,
    ADD COLUMN IF NOT EXISTS hr_status                   varchar(10),
    ADD COLUMN IF NOT EXISTS her2_status                 varchar(10),
    ADD COLUMN IF NOT EXISTS disease_stage               varchar(20),
    ADD COLUMN IF NOT EXISTS ecog_performance_status     integer,
    ADD COLUMN IF NOT EXISTS ldh_ratio                   numeric(5,2),
    ADD COLUMN IF NOT EXISTS complement_inhibitor_status varchar(10),
    ADD COLUMN IF NOT EXISTS proteinuria_g_day           numeric(5,2),
    ADD COLUMN IF NOT EXISTS egfr                        numeric(6,2);
