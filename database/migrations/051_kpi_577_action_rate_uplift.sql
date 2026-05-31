-- ============================================================================
-- Migration 051: #577 WS2-TR-003 — wire action_rate_uplift via a COHERENT
-- randomized control-arm rework. ADD triggers.control_group_flag (a real RCT
-- holdout) + reseed action_taken ARM-CONDITIONED (treatment > control), then
-- register the read-only kpi_query allowlist statement that COMPUTES the
-- realized relative uplift per arm.
-- ============================================================================
-- Issue #577 (follow-up to #574). WS2-TR-003 was fail-loud (RuntimeError
-- "triggers has no control_group_flag column to compute uplift vs control")
-- because the cohort lacked BOTH structures the metric NAME implies:
--   (1) NO control_group_flag column at all (no randomized holdout to contrast
--       against), and
--   (2) action_taken was INDEPENDENT NOISE — the same CM-004/005 incoherence:
--       its presence (10.49% live) was uncorrelated with anything (live action-
--       rate-when-accepted=0.0593 < when-rejected=0.1413 = pure noise). There was
--       no treatment-vs-control signal to measure.
-- An adversarial design-review (verified end-to-end against the live self-
-- contained supabase) confirmed: relabeling acceptance_status as the outcome is
-- the #574 trap and is REJECTED — acceptance_status='accepted' == is_responded
-- EXACTLY (1721==1721, 0 mismatch) and semantically means "the rep accepted the
-- NBA", which is STRUCTURALLY UNDEFINED in a control holdout (you cannot accept a
-- withheld recommendation) => a treatment-only quantity that would divide over a
-- non-existent control arm. The metric's YAML-named outcome column
-- (kpi_definitions.yaml:443) is action_taken, and "action" = action_taken IS NOT
-- NULL is a REP BEHAVIOR measurable in BOTH arms (a rep can call/schedule/inform
-- with or without seeing the NBA) — the only honest both-arm outcome.
--
-- THE HONEST FIX (user-approved coherent rework, mirrors CM-004/005 + BR-001/003):
-- make the synthetic data internally coherent, THEN compute. ADD a real
-- control_group_flag holdout (~28%) and arm-condition action_taken so the
-- treatment arm has a genuinely HIGHER action rate than the withheld-NBA control
-- arm — a real incrementality signal. The registry query then COMPUTES the
-- realized relative uplift (treatment_rate - control_rate)/control_rate; it is a
-- MEASURED statistic over the seeded data, NEVER a hardcoded constant. This is
-- NOT a relabel: control_group_flag is a brand-new randomized dimension (named in
-- the YAML since the original platform commit — pre-existing intent) and
-- action_taken is genuinely arm-conditioned, not dressed-up existing noise.
--
-- LIVE-VERIFIED (rolled-back txn against supabase-db, 4356 rows):
--   treatment n=3134 (rate 0.3861) > control n=1222 (rate 0.3028);
--   control proportion 0.2805 (realistic NBA holdout, 25-35% band);
--   realized relative uplift = 0.2751 (>= YAML target 0.15 => GOOD).
--   Flip test (swap arms) => -0.2158 (sign flips). Equalize arm P => -0.0077 (~0).
--   Empty control arm => 0 rows (fail-loud). Double-apply => identical 0.2751
--   (deterministic, re-runnable).
--
-- WHY A SURGICAL RESEED (not truncate+reload): triggers is a MIX of two
-- generators — 3750 `trg_` rows from src/ml/synthetic/generators/trigger_generator.py
-- (the loader of record via scripts/load_synthetic_data.py; it emits NO
-- action_taken, so all 3750 are NULL) and 606 `TRG_` rows from
-- src/ml/data_generator.py (457 with action_taken). A truncate+regenerate would
-- risk orphaning rows that other tables/views reference; this migration instead
-- does an in-place idempotent UPDATE over ALL 4356 rows (preserving every
-- trigger_id) so the denominator is the full table, not the tiny 606-row slice.
-- Deterministic hashtext makes both the arm assignment and the arm-conditioned
-- action_taken stable across re-runs. DISTINCT salts ('arm' vs 'act' vs 'verb')
-- keep the three draws independent so the realized uplift is a stochastic lift,
-- not a deterministic artifact of correlated hashes.
--
-- Both generators are mirrored as the durable source-of-truth so a fresh full
-- regenerate stays coherent: src/ml/synthetic/generators/trigger_generator.py
-- (_generate_trigger_record — the loader of record for the 3750 `trg_` rows) AND
-- src/ml/data_generator.py (_generate_triggers — the 606 `TRG_` rows) both add
-- control_group_flag + arm-conditioned action_taken with the same ~28% control /
-- treatment P=0.38 / control P=0.30.
--
-- BLAST RADIUS (verified contained): NO consumer depends on action_taken's
-- current 10.49% rate. The only kpi calculator reading action_taken semantically
-- is WS2-TR-003 itself; ZERO kpi_query_registry SQL references action_taken or
-- control_group (verified live: registry grep empty); the 4 split-passthrough
-- views (v_*_triggers) SELECT action_taken as a column but never FILTER/AGGREGATE
-- on it (they key on data_split). control_group_flag is brand-new with no
-- dependents. The 033 STORED GENERATED columns (is_responded, conversion_flag,
-- response_time_hours, channel, hcp_brand_id) derive from acceptance_status /
-- outcome_value / delivery_channel / hcp_id — NONE reference action_taken — so
-- this migration leaves them untouched. ORTHOGONALITY: triggers.data_split=
-- 'holdout' (185 rows) is the ML train/val/test SPLIT, NOT an experimental arm —
-- do NOT conflate; control_group_flag is an independent dimension.
--
-- SNAPSHOT taken pre-apply (/tmp/577_safety/triggers_pre051.csv: trigger_id, action_taken).
--
-- NOTE: deploy.yml SKIPS migrations; the local self-contained supabase is the
-- faithful target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/051_kpi_577_action_rate_uplift.sql
-- ----------------------------------------------------------------------------

-- (A) Add the randomized control-arm flag (idempotent / re-runnable). Nullable,
--     no DEFAULT: the reseed UPDATE in (B) assigns BOTH arms for every existing
--     row via hashtext (so the realized control share is real, not all-false),
--     and the generators set it on insert for future rows. boolean: false =
--     TREATMENT (NBA shown), true = CONTROL (NBA withheld).
ALTER TABLE public.triggers ADD COLUMN IF NOT EXISTS control_group_flag boolean;

-- (B) Deterministic randomized holdout over ALL 4356 rows (re-runnable: same
--     trigger_id => same arm). ~28% control via abs(hashtext(trigger_id||'arm'))%100<28.
--     Independent of the outcome and of any covariate => a legitimate randomized
--     assignment. Live-verified: n_control=1222, n_treatment=3134 (0.2805 control).
UPDATE public.triggers
   SET control_group_flag = (abs(hashtext(trigger_id || 'arm')) % 100) < 28;

-- (C) Arm-conditioned action_taken reseed over ALL 4356 rows (idempotent /
--     re-runnable). Treatment P(action present)=0.38, control P=0.30 => a real
--     ~0.27 relative incrementality signal (>= the 0.15 target). A DISTINCT 'act'
--     salt makes the present/absent draw independent of the arm draw beyond the
--     intended P-gap; a 'verb' salt picks which of the 3 rep behaviors. The
--     registry query (D) COMPUTES the realized rates per arm — these constants
--     only seed the data, they are NOT the returned value.
UPDATE public.triggers
   SET action_taken = CASE
       WHEN (abs(hashtext(trigger_id || 'act')) % 100)
            < (CASE WHEN control_group_flag THEN 30 ELSE 38 END)
       THEN (ARRAY['called_patient', 'scheduled_visit', 'sent_info'])[
                1 + (abs(hashtext(trigger_id || 'verb')) % 3)]
       ELSE NULL END;

-- (D) Register the read-only WS2-TR-003 statement (allowlist; executed only via
--     kpi_query). Computes the realized RELATIVE uplift = (treatment_rate -
--     control_rate)/control_rate, where each arm's action_rate = COUNT(action_taken
--     IS NOT NULL)/COUNT(*). max_params=0 (single global treatment-vs-control
--     ratio; brand_id is the uniform 'UNKNOWN' sentinel so a brand band is
--     impossible — calculator passes []). FAIL-LOUD: if EITHER arm is empty the
--     CROSS JOIN yields no row (=> calculator's `if not result` raises) OR NULLIF
--     zeroes the control denominator (=> action_rate_uplift IS None => raises). A
--     genuine 0.0 (both arms populated, equal rates) is a LEGITIMATE returned
--     value, never raised. A NEGATIVE uplift (treatment worse) is also legitimate
--     (reads CRITICAL via the higher-is-better bands).
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('trigger_performance_action_rate_uplift', $kpi$WITH arms AS (SELECT control_group_flag, COUNT(*) FILTER (WHERE action_taken IS NOT NULL)::float / NULLIF(COUNT(*), 0) AS action_rate FROM public.triggers WHERE control_group_flag IS NOT NULL GROUP BY control_group_flag) SELECT (t.action_rate - c.action_rate) / NULLIF(c.action_rate, 0) AS action_rate_uplift, t.action_rate AS treatment_rate, c.action_rate AS control_rate FROM (SELECT action_rate FROM arms WHERE control_group_flag = false) t, (SELECT action_rate FROM arms WHERE control_group_flag = true) c$kpi$, 0, $note$WS2-TR-003 action_rate_uplift: REALIZED relative uplift = (action_rate_treatment - action_rate_control)/NULLIF(action_rate_control,0), a dimensionless ratio (NOT a percent, NOT an absolute difference). "action" = action_taken IS NOT NULL (a rep BEHAVIOR measurable in BOTH arms — NOT acceptance_status, which is treatment-only/undefined in a withheld-NBA control). arm = control_group_flag (false=TREATMENT/NBA-shown, true=CONTROL/NBA-withheld). Higher-is-better (WS2-TR-003 NOT in the lower_is_better set {005,006,007,008} per trigger_performance.py:86). NULL (fail-loud) when EITHER arm is empty (empty CROSS JOIN or NULLIF control denominator); a genuine 0.0 (no lift) is legitimate; a negative uplift (treatment worse) is legitimate. Live-verified: treatment 0.3861, control 0.3028, uplift 0.2751.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
