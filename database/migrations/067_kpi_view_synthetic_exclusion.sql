-- ============================================================================
-- Migration 067: close the view-backed kpi_query synthetic leak (codex HIGH).
-- (1) is_synthetic on the 3 view-backed tables M1 missed; (2) CREATE OR REPLACE
-- the KPI views that read a taggable table to default-exclude synthetic rows.
-- Wrap is alias-preserving so view output columns are unchanged (CREATE OR
-- REPLACE safe). Idempotent. Depends on: 063 (M1 is_synthetic columns).
-- ----------------------------------------------------------------------------

ALTER TABLE data_source_tracking ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE etl_pipeline_metrics ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ml_annotations ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;

CREATE OR REPLACE VIEW public.v_patient_eligibility AS
SELECT DISTINCT pj.patient_id,
    pj.brand,
    (EXISTS ( SELECT 1
           FROM (SELECT * FROM triggers WHERE is_synthetic = false) t
          WHERE t.patient_id::text = pj.patient_id::text AND (t.delivery_status::text = ANY (ARRAY['delivered'::character varying, 'viewed'::character varying]::text[])))) AS has_delivered_touch
   FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj
  WHERE pj.primary_diagnosis_code::text = ANY (ARRAY['C50.1'::character varying, 'C50.2'::character varying, 'C50.9'::character varying, 'D59.5'::character varying, 'L50.1'::character varying, 'L50.8'::character varying, 'L50.9'::character varying]::text[]);

CREATE OR REPLACE VIEW public.v_kpi_active_users AS
SELECT date_trunc('month'::text, user_sessions.session_start) AS month,
    count(DISTINCT user_sessions.user_id) AS monthly_active_users,
    count(DISTINCT
        CASE
            WHEN user_sessions.session_start >= date_trunc('week'::text, now()) THEN user_sessions.user_id
            ELSE NULL::character varying
        END) AS weekly_active_users,
    count(DISTINCT
        CASE
            WHEN user_sessions.session_start >= date_trunc('day'::text, now()) THEN user_sessions.user_id
            ELSE NULL::character varying
        END) AS daily_active_users
   FROM (SELECT * FROM user_sessions WHERE is_synthetic = false) user_sessions
  GROUP BY (date_trunc('month'::text, user_sessions.session_start));

CREATE OR REPLACE VIEW public.v_kpi_intent_to_prescribe AS
SELECT hcp_intent_surveys.brand,
    date_trunc('month'::text, hcp_intent_surveys.survey_date::timestamp with time zone) AS survey_month,
    avg(hcp_intent_surveys.intent_to_prescribe_score) AS avg_intent_score,
    avg(hcp_intent_surveys.intent_to_prescribe_change) AS avg_intent_change,
    count(*) AS survey_count
   FROM (SELECT * FROM hcp_intent_surveys WHERE is_synthetic = false) hcp_intent_surveys
  WHERE hcp_intent_surveys.response_quality_flag = true
  GROUP BY hcp_intent_surveys.brand, (date_trunc('month'::text, hcp_intent_surveys.survey_date::timestamp with time zone));

CREATE OR REPLACE VIEW public.v_kpi_data_lag AS
SELECT date(patient_journeys.created_at) AS report_date,
    patient_journeys.data_source,
    avg(patient_journeys.data_lag_hours) AS avg_lag_hours,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (patient_journeys.data_lag_hours::double precision)) AS median_lag_hours,
    percentile_cont(0.95::double precision) WITHIN GROUP (ORDER BY (patient_journeys.data_lag_hours::double precision)) AS p95_lag_hours
   FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys
  WHERE patient_journeys.data_lag_hours IS NOT NULL
  GROUP BY (date(patient_journeys.created_at)), patient_journeys.data_source;

CREATE OR REPLACE VIEW public.v_kpi_cross_source_match AS
SELECT data_source_tracking.tracking_date,
    data_source_tracking.source_name,
    data_source_tracking.records_received,
    data_source_tracking.records_matched,
        CASE
            WHEN data_source_tracking.records_received > 0 THEN data_source_tracking.records_matched::numeric / data_source_tracking.records_received::numeric
            ELSE 0::numeric
        END AS match_rate
   FROM (SELECT * FROM data_source_tracking WHERE is_synthetic = false) data_source_tracking;

CREATE OR REPLACE VIEW public.v_kpi_stacking_lift AS
SELECT data_source_tracking.tracking_date,
    sum(data_source_tracking.stacking_eligible_records) AS total_eligible,
    sum(data_source_tracking.stacking_applied_records) AS total_stacked,
    avg(data_source_tracking.stacking_lift_percentage) AS avg_lift_pct
   FROM (SELECT * FROM data_source_tracking WHERE is_synthetic = false) data_source_tracking
  GROUP BY data_source_tracking.tracking_date;

CREATE OR REPLACE VIEW public.v_kpi_time_to_release AS
SELECT date(etl_pipeline_metrics.run_start) AS run_date,
    etl_pipeline_metrics.pipeline_name,
    avg(etl_pipeline_metrics.time_to_release_hours) AS avg_ttr_hours,
    min(etl_pipeline_metrics.time_to_release_hours) AS min_ttr_hours,
    max(etl_pipeline_metrics.time_to_release_hours) AS max_ttr_hours
   FROM (SELECT * FROM etl_pipeline_metrics WHERE is_synthetic = false) etl_pipeline_metrics
  WHERE etl_pipeline_metrics.status::text = 'success'::text
  GROUP BY (date(etl_pipeline_metrics.run_start)), etl_pipeline_metrics.pipeline_name;

CREATE OR REPLACE VIEW public.v_kpi_change_fail_rate AS
SELECT date(triggers.change_timestamp) AS change_date,
    triggers.change_type,
    count(*) AS total_changes,
    sum(
        CASE
            WHEN triggers.change_failed THEN 1
            ELSE 0
        END) AS failed_changes,
    sum(
        CASE
            WHEN triggers.change_failed THEN 1
            ELSE 0
        END)::numeric / NULLIF(count(*), 0)::numeric AS fail_rate
   FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers
  WHERE triggers.change_type IS NOT NULL
  GROUP BY (date(triggers.change_timestamp)), triggers.change_type;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
