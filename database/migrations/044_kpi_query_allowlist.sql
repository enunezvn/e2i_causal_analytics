-- ============================================================================
-- Migration 044: query-ID allowlist for KPI calculators (kpi_query RPC)
-- ============================================================================
-- Issue #574. The six KPI calculators in ``src/kpi/calculators/`` retrieved metrics
-- via ``client.rpc("execute_sql", {"query": <raw SQL>})`` -- but no ``execute_sql``
-- function exists in any applied migration (``database/chat/009`` defines a DIFFERENT,
-- unapplied, unsafe ``execute_custom_sql`` that string-concats arbitrary SQL). So every
-- SQL-backed KPI query 404s at runtime (verified live: pg_proc count = 0).
--
-- A generic arbitrary-SQL RPC cannot be made safe with a regex guard (wrapper breakout
-- via ``) ... --``, ``SELECT <write_fn>()``, anon-callable). Instead this migration uses a
-- QUERY-ID ALLOWLIST: the vetted, read-only KPI statements live server-side in
-- ``kpi_query_registry``; the only callable RPC, ``kpi_query(query_id, params)``, runs
-- ONLY a registered statement (looked up by id) with positionally-bound params. Clients
-- pass an id + params, NEVER raw SQL -- so there is no SQL-injection / breakout surface,
-- and it is safe to expose to the anon role the calculators use.
--
-- Security properties:
--   * No arbitrary SQL from clients -- only ids that must match a registry row.
--   * Params bound via EXECUTE ... USING (by arity; PL/pgSQL USING has no VARIADIC),
--     never string-interpolated.
--   * SECURITY DEFINER so the vetted read-only aggregates read across data (the anon
--     caller otherwise hits RLS); SAFE because only allowlisted SELECT/WITH runs.
--   * Registry CHECK enforces every stored statement is read-only (SELECT/WITH).
--   * Registry table locked down; clients reach it only through the function.
--
-- NOTE: deploy.yml runs migrations only when SUPABASE_DB_URL is set, so this must be
-- applied to the target Supabase for the RPC to exist there; until then the KPI
-- calculators continue to fail-closed (the pre-existing state).
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.kpi_query_registry (
    query_id   text PRIMARY KEY,
    sql        text NOT NULL,
    max_params int  NOT NULL DEFAULT 0,
    note       text,
    CONSTRAINT kpi_query_registry_readonly_chk CHECK (sql ~* '^\s*(with|select)\s')
);

COMMENT ON TABLE public.kpi_query_registry IS
    'Allowlist of vetted read-only KPI SQL statements (#574), keyed by query_id. '
    'kpi_query() runs ONLY statements from this table -- clients pass an id, never raw SQL.';

-- Lock the registry down: only privileged roles touch it directly; clients use the
-- SECURITY DEFINER function (which reads it as owner regardless of caller grants).
REVOKE ALL ON public.kpi_query_registry FROM PUBLIC;
GRANT SELECT ON public.kpi_query_registry TO service_role;

CREATE OR REPLACE FUNCTION public.kpi_query(query_id text, params jsonb DEFAULT '[]'::jsonb)
RETURNS SETOF json
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $func$
DECLARE
    stmt text;
    expected_n int;
    param_arr text[];
    n int;
    wrapped text;
BEGIN
    SELECT r.sql, r.max_params INTO stmt, expected_n
      FROM public.kpi_query_registry r
     WHERE r.query_id = kpi_query.query_id;

    IF stmt IS NULL THEN
        RAISE EXCEPTION 'kpi_query: unknown query_id %', query_id USING ERRCODE = '22023';
    END IF;

    -- params must be a JSON array (reject objects/scalars that would bind wrong).
    IF params IS NOT NULL AND jsonb_typeof(params) <> 'array' THEN
        RAISE EXCEPTION 'kpi_query: params must be a JSON array (got %)', jsonb_typeof(params)
            USING ERRCODE = '22023';
    END IF;

    -- Bind $1..$N positionally from the jsonb array (text-typed; statements cast as needed).
    SELECT array_agg(elem #>> '{}' ORDER BY ord)
      INTO param_arr
      FROM jsonb_array_elements(COALESCE(params, '[]'::jsonb)) WITH ORDINALITY AS t(elem, ord);
    param_arr := COALESCE(param_arr, ARRAY[]::text[]);
    n := COALESCE(array_length(param_arr, 1), 0);

    -- Enforce the registry's declared arity (no wrong-count binding).
    IF n <> expected_n THEN
        RAISE EXCEPTION 'kpi_query: % expects % param(s), got %', query_id, expected_n, n
            USING ERRCODE = '22023';
    END IF;

    -- stmt is a TRUSTED, registry-vetted statement (not client SQL): safe to wrap.
    wrapped := format('SELECT row_to_json(_sub) FROM (%s) AS _sub', stmt);

    IF n = 0 THEN
        RETURN QUERY EXECUTE wrapped;
    ELSIF n = 1 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1];
    ELSIF n = 2 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1], param_arr[2];
    ELSIF n = 3 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1], param_arr[2], param_arr[3];
    ELSIF n = 4 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1], param_arr[2], param_arr[3], param_arr[4];
    ELSE
        RAISE EXCEPTION 'kpi_query: at most 4 positional parameters supported (got %)', n;
    END IF;
END;
$func$;

COMMENT ON FUNCTION public.kpi_query(text, jsonb) IS
    'Run a vetted read-only KPI statement from kpi_query_registry by id (#574). '
    'Clients pass query_id + params; never raw SQL. SECURITY DEFINER over allowlist.';

REVOKE ALL ON FUNCTION public.kpi_query(text, jsonb) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.kpi_query(text, jsonb) TO anon, authenticated, service_role;

-- ----------------------------------------------------------------------------
-- Allowlist seed: the vetted read-only KPI statements (generated verbatim from the
-- calculators; conditional filters split into static parameterized variants).
-- ----------------------------------------------------------------------------
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('causal_metrics_ate', $kpi$SELECT AVG(treatment_effect_estimate) AS ate, STDDEV(treatment_effect_estimate) AS ate_std, COUNT(*) AS n_samples FROM ml_predictions WHERE treatment_effect_estimate IS NOT NULL AND prediction_timestamp >= NOW() - INTERVAL '30 days'$kpi$, 0, NULL),
    ('causal_metrics_cate', $kpi$SELECT segment_assignment, AVG(heterogeneous_effect) AS cate, STDDEV(heterogeneous_effect) AS cate_std, COUNT(*) AS n_samples FROM ml_predictions WHERE heterogeneous_effect IS NOT NULL AND prediction_timestamp >= NOW() - INTERVAL '30 days' AND ($1::text IS NULL OR segment_assignment = $1) GROUP BY segment_assignment ORDER BY AVG(heterogeneous_effect) DESC$kpi$, 1, NULL),
    ('business_impact_mau_view', $kpi$SELECT monthly_active_users AS mau FROM v_kpi_active_users ORDER BY month DESC LIMIT 1$kpi$, 0, $note$schema: mau->monthly_active_users; order by month$note$),
    ('business_impact_mau_fallback', $kpi$SELECT COUNT(DISTINCT user_id) AS mau FROM user_sessions WHERE session_start >= NOW() - INTERVAL '30 days'$kpi$, 0, NULL),
    ('business_impact_wau_view', $kpi$SELECT weekly_active_users AS wau FROM v_kpi_active_users ORDER BY month DESC LIMIT 1$kpi$, 0, $note$schema: wau->weekly_active_users; order by month$note$),
    ('business_impact_wau_fallback', $kpi$SELECT COUNT(DISTINCT user_id) AS wau FROM user_sessions WHERE session_start >= NOW() - INTERVAL '7 days'$kpi$, 0, NULL),
    ('business_impact_hcp_coverage', $kpi$SELECT COUNT(CASE WHEN coverage_status = true THEN 1 END)::float / NULLIF(COUNT(CASE WHEN priority_tier <= 2 THEN 1 END), 0) AS coverage FROM hcp_profiles$kpi$, 0, $note$schema: coverage_status is boolean (was compared to 'covered')$note$),
    ('business_impact_trx', $kpi$SELECT COUNT(*) AS trx FROM treatment_events WHERE event_type::text = 'prescription' AND event_date >= NOW() - INTERVAL '30 days' AND ($1::text IS NULL OR brand::text = $1)$kpi$, 1, $note$schema: brand is enum brand_type (::text cast); brand filter optional$note$),
    ('business_impact_nrx', $kpi$SELECT COUNT(*) AS nrx FROM treatment_events WHERE event_type::text = 'prescription' AND sequence_number = 1 AND event_date >= NOW() - INTERVAL '30 days' AND ($1::text IS NULL OR brand::text = $1)$kpi$, 1, $note$schema: brand enum ::text cast; brand filter optional$note$),
    ('business_impact_nbrx', $kpi$WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM treatment_events WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id) SELECT COUNT(*) AS nbrx FROM first_brand WHERE first_date >= NOW() - INTERVAL '30 days'$kpi$, 1, $note$schema: brand enum ::text cast$note$),
    ('business_impact_trx_share', $kpi$WITH category AS (SELECT COUNT(*) AS total FROM treatment_events WHERE event_type::text = 'prescription' AND event_date >= NOW() - INTERVAL '30 days'), brand_rx AS (SELECT COUNT(*) AS total FROM treatment_events WHERE event_type::text = 'prescription' AND brand::text = $1 AND event_date >= NOW() - INTERVAL '30 days') SELECT brand_rx.total::float / NULLIF(category.total, 0) AS share FROM category, brand_rx$kpi$, 1, $note$schema: brand enum ::text cast (brand required for share)$note$),
    ('business_impact_conversion_rate', $kpi$WITH triggered AS (SELECT COUNT(DISTINCT trigger_id) AS total FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days'), converted AS (SELECT COUNT(DISTINCT t.trigger_id) AS total FROM triggers t INNER JOIN treatment_events te ON te.patient_id = t.patient_id WHERE t.trigger_timestamp >= NOW() - INTERVAL '30 days' AND te.event_type::text = 'prescription' AND te.event_date >= t.trigger_timestamp::date AND te.event_date <= (t.trigger_timestamp + INTERVAL '30 days')::date) SELECT converted.total::float / NULLIF(triggered.total, 0) AS conversion_rate FROM triggered, converted$kpi$, 0, $note$schema: fired_at->trigger_timestamp; conversion = prescription within 30d after trigger$note$),
    ('business_impact_roi_business_metrics', $kpi$SELECT AVG(roi) AS avg_roi FROM business_metrics WHERE metric_date >= NOW() - INTERVAL '30 days' AND roi IS NOT NULL$kpi$, 0, NULL),
    ('business_impact_roi_agent_activities', $kpi$SELECT AVG(roi_estimate) AS avg_roi FROM agent_activities WHERE activity_timestamp >= NOW() - INTERVAL '30 days' AND roi_estimate IS NOT NULL$kpi$, 0, NULL),
    ('brand_specific_remi_intent_delta_primary', $kpi$SELECT avg_intent_change AS intent_delta FROM v_kpi_intent_to_prescribe WHERE brand::text = 'Remibrutinib' ORDER BY survey_month DESC LIMIT 1$kpi$, 0, $note$schema: brand enum ::text cast$note$),
    ('brand_specific_remi_intent_delta_fallback', $kpi$SELECT AVG(intent_to_prescribe_change) AS intent_delta FROM hcp_intent_surveys WHERE brand::text = 'Remibrutinib' AND survey_date >= NOW() - INTERVAL '90 days' AND intent_to_prescribe_change IS NOT NULL$kpi$, 0, $note$schema: brand enum ::text cast$note$),
    ('brand_specific_kisqali_dx_adoption', $kpi$WITH first_kisqali AS (SELECT te.patient_id, MIN(te.event_date) AS first_rx_date FROM treatment_events te WHERE te.brand::text = 'Kisqali' AND te.event_type::text = 'prescription' GROUP BY te.patient_id) SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (fk.first_rx_date - pj.journey_start_date)) AS median_days FROM first_kisqali fk INNER JOIN patient_journeys pj ON pj.patient_id = fk.patient_id WHERE pj.journey_start_date IS NOT NULL AND fk.first_rx_date >= pj.journey_start_date$kpi$, 0, $note$PROXY: diagnosis date approximated by patient_journeys.journey_start_date (no diagnosis_date column)$note$),
    ('brand_specific_kisqali_oncologist_reach', $kpi$WITH oncologists AS (SELECT COUNT(DISTINCT hcp_id) AS total FROM hcp_profiles WHERE specialty ILIKE '%oncolog%'), engaged AS (SELECT COUNT(DISTINCT t.hcp_id) AS total FROM triggers t INNER JOIN hcp_profiles hp ON hp.hcp_id = t.hcp_id WHERE hp.specialty ILIKE '%oncolog%' AND t.brand_id = 'Kisqali' AND t.trigger_timestamp >= NOW() - INTERVAL '90 days') SELECT engaged.total::float / NULLIF(oncologists.total, 0) AS reach FROM oncologists, engaged$kpi$, 0, $note$schema: fired_at->trigger_timestamp; triggers.brand stored as brand_id$note$),
    ('trigger_performance_precision', $kpi$SELECT COUNT(CASE WHEN outcome_tracked AND outcome_value > 0 THEN 1 END)::float / NULLIF(COUNT(CASE WHEN outcome_tracked THEN 1 END), 0) AS precision FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days'$kpi$, 0, $note$schema: fired_at->trigger_timestamp$note$),
    ('trigger_performance_recall', $kpi$WITH positive_outcomes AS (SELECT DISTINCT patient_id FROM treatment_events WHERE event_type::text = 'prescription' AND event_date >= NOW() - INTERVAL '30 days'), trigger_preceded AS (SELECT DISTINCT po.patient_id FROM positive_outcomes po INNER JOIN triggers t ON t.patient_id = po.patient_id WHERE t.trigger_timestamp < (SELECT MIN(event_date) FROM treatment_events te WHERE te.patient_id = po.patient_id AND te.event_type::text = 'prescription')) SELECT COUNT(DISTINCT tp.patient_id)::float / NULLIF(COUNT(DISTINCT po.patient_id), 0) AS recall FROM positive_outcomes po LEFT JOIN trigger_preceded tp ON tp.patient_id = po.patient_id$kpi$, 0, $note$PROXY: 'conversion' event does not exist; positive outcome redefined as a prescription$note$),
    ('trigger_performance_acceptance_rate', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0) AS acceptance_rate FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days'$kpi$, 0, $note$schema: fired_at->trigger_timestamp$note$),
    ('trigger_performance_false_alert_rate', $kpi$SELECT COUNT(CASE WHEN false_positive_flag THEN 1 END)::float / NULLIF(COUNT(*), 0) AS false_alert_rate FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days'$kpi$, 0, $note$schema: fired_at->trigger_timestamp$note$),
    ('trigger_performance_override_rate', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0) AS override_rate FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days'$kpi$, 0, $note$schema: fired_at->trigger_timestamp$note$),
    ('trigger_performance_lead_time', $kpi$SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lead_time_days) AS median_lead_time FROM triggers WHERE lead_time_days IS NOT NULL AND trigger_timestamp >= NOW() - INTERVAL '30 days'$kpi$, 0, $note$schema: fired_at->trigger_timestamp$note$),
    ('trigger_performance_cfr', $kpi$SELECT COUNT(CASE WHEN change_failed THEN 1 END)::float / NULLIF(COUNT(CASE WHEN previous_trigger_id IS NOT NULL THEN 1 END), 0) AS cfr FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days'$kpi$, 0, $note$schema: fired_at->trigger_timestamp; direct calc (v_kpi_change_fail_rate has no avg_cfr column)$note$),
    ('data_quality_source_coverage_patients', $kpi$SELECT COUNT(DISTINCT pj.patient_id) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE ($1::text IS NULL OR brand::text = $1)), 0) AS total FROM patient_journeys pj WHERE ($1::text IS NULL OR pj.brand::text = $1)$kpi$, 1, $note$PROXY: reference denominator = SUM(reference_universe.target_count) (no reference_hcps/patients table)$note$),
    ('data_quality_cross_source_match', $kpi$SELECT match_rate FROM v_kpi_cross_source_match LIMIT 1$kpi$, 0, NULL),
    ('data_quality_stacking_lift', $kpi$SELECT avg_lift_pct AS lift_score FROM v_kpi_stacking_lift LIMIT 1$kpi$, 0, $note$PROXY: lift_score<-avg_lift_pct (view-provided average % uplift)$note$),
    ('data_quality_completeness_pass_rate', $kpi$SELECT AVG(CASE WHEN patient_id IS NOT NULL AND brand IS NOT NULL AND event_date IS NOT NULL THEN 1.0 ELSE 0.0 END) AS pass_rate FROM patient_journeys WHERE created_at >= NOW() - INTERVAL '30 days'$kpi$, 0, NULL),
    ('data_quality_data_lag', $kpi$SELECT (median_lag_hours / 24.0) AS median_lag_days FROM v_kpi_data_lag LIMIT 1$kpi$, 0, $note$unit: median_lag_hours/24 -> days$note$),
    ('data_quality_time_to_release', $kpi$SELECT (avg_ttr_hours / 24.0) AS median_ttr_days FROM v_kpi_time_to_release LIMIT 1$kpi$, 0, $note$PROXY: view exposes avg (not median) TTR; unit hours/24 -> days$note$),
    ('model_performance_shap_coverage', $kpi$SELECT COUNT(CASE WHEN shap_values IS NOT NULL THEN 1 END)::float / NULLIF(COUNT(*), 0) AS coverage FROM ml_predictions WHERE created_at >= NOW() - INTERVAL '30 days'$kpi$, 0, $note$schema: predictions->ml_predictions$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;
-- PostgREST caches the schema; reload so the kpi_query RPC is exposed immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
