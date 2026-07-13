-- ============================================================================
-- Migration 106: #1188 — make triggers.action_taken PROGNOSTIC on
-- pre-treatment patient baselines (disease_severity, age_at_diagnosis) while
-- keeping the randomized arm assignment (control_group_flag) UNTOUCHED.
-- ============================================================================
-- Issue #1188 (follow-up to #577 / migration 051). The nba_triggers RCT gains
-- an OPT-IN ANCOVA-style baseline adjustment (variance reduction). For that
-- adjustment to be able to tighten the ATE confidence interval, the outcome
-- must actually be prognostic on the baselines. It is NOT today: migration 051
-- seeded action_taken with an ARM-ONLY probability (control 0.30 / treatment
-- 0.38), so patient baselines carry ZERO outcome signal.
--
-- MEASURED on the live table before this migration (37,541 triggers, 100%
-- patient_journeys join coverage): within-arm baseline R^2 = 0.0003-0.0005;
-- ANCOVA delta-SE = -0.00%. A baseline adjustment cannot narrow an interval
-- when the baselines are pure noise — the #1188 acceptance criterion
-- ("adjusted CI narrower, verified in a faithful run") is unsatisfiable
-- against the current data.
--
-- THE REWORK (same coherence philosophy as 051): re-seed ONLY action_taken,
-- arm-conditioned AND baseline-prognostic:
--
--   p_action = clip( base(arm)                       -- 0.30 control / 0.38 treatment
--                  + 0.12  * (disease_severity - 5.0) -- prognostic, sd(sev)~2
--                  - 0.002 * (age_at_diagnosis - 50)  -- mild secondary signal
--                  , 0.02, 0.95)
--
--   action_taken = verb  when u_act < p_action  else NULL
--
-- clinical story: high-severity (and younger) patients get acted on more,
-- regardless of arm. The severity/age terms are BALANCED across arms by
-- randomization, so the ~8pp arm contrast is preserved in expectation
-- (clipping erodes it slightly; measured post-apply below). #1188 NON-GOAL
-- honored: control_group_flag is NOT re-drawn — no covariate influences
-- treatment assignment; the RCT's empty backdoor stays valid.
--
-- DETERMINISM / IDEMPOTENCY (051 pattern): both draws are pure functions of
-- trigger_id via salted md5 -> uniform [0,1), so re-running this migration is
-- a no-op (double-apply identical). DISTINCT salts ('act1188' vs 'verb1188')
-- keep presence and verb draws independent, and both independent of 051's
-- 'arm'/'act'/'verb' hashtext draws.
--
-- SOURCE-OF-TRUTH MIRRORS (a fresh full regenerate stays coherent):
--   src/ml/synthetic/generators/trigger_generator.py
--     (_prognostic_action_probability + _generate_trigger_record — the loader
--      of record; the standalone path stays arm-only because its fabricated
--      patient_ids join to no patient_journeys row)
--   src/ml/data_generator.py keeps arm-only Ps: its legacy patient dicts
--     carry no baseline columns and zero of its TRG_ rows exist live (live
--     prefix census: 37,541/37,541 'scvt').
--
-- BLAST RADIUS (re-verified for #1188, mirrors 051's census):
--   * WS2-TR-003 action_rate_uplift (kpi calculator + registry SQL variants,
--     incl. 066/078 region/synthetic variants) COMPUTE realized arm rates —
--     no dependence on within-arm homogeneity. Expected realized relative
--     uplift ~0.17-0.19 (>= 0.15 YAML target; was ~0.26).
--   * conversion_flag / outcome_value / acceptance_status / injected
--     prescriptions: UNTOUCHED (conversion substrate does not read
--     action_taken).
--   * causal route nba_triggers RCT (control_group_flag -> action_taken):
--     the POINT estimate stays ~8pp (unbiased); its CI becomes honestly
--     narrower under the new opt-in baseline adjustment — that is the point.
--   * split-passthrough views select action_taken but never filter on it.
--
-- Rows whose patient has no patient_journeys row (0 live today) or NULL
-- baselines (0 live today) fall back to the arm base via COALESCE — the
-- mechanism degrades to exactly the 051 behavior.
-- ============================================================================

UPDATE triggers t
SET action_taken = CASE
    WHEN (
        ('x' || substr(md5(t.trigger_id::text || ':act1188'), 1, 8))::bit(32)::bigint::float8
        / 4294967296.0
    ) < LEAST(
        GREATEST(
            (CASE WHEN t.control_group_flag THEN 0.30 ELSE 0.38 END)
            + 0.12 * (COALESCE(pj.disease_severity, 5.0) - 5.0)
            - 0.002 * (COALESCE(pj.age_at_diagnosis, 50)::float8 - 50.0),
            0.02
        ),
        0.95
    )
    THEN (ARRAY['called_patient', 'scheduled_visit', 'sent_info'])[
        1 + (
            ('x' || substr(md5(t.trigger_id::text || ':verb1188'), 1, 8))::bit(32)::bigint % 3
        )::int
    ]
    ELSE NULL
END
FROM patient_journeys pj
WHERE pj.patient_id = t.patient_id
  AND t.control_group_flag IS NOT NULL;

-- Orphan triggers (no patient_journeys row; 0 live today, guard for future
-- loads): arm-only probability, same deterministic draws.
UPDATE triggers t
SET action_taken = CASE
    WHEN (
        ('x' || substr(md5(t.trigger_id::text || ':act1188'), 1, 8))::bit(32)::bigint::float8
        / 4294967296.0
    ) < (CASE WHEN t.control_group_flag THEN 0.30 ELSE 0.38 END)
    THEN (ARRAY['called_patient', 'scheduled_visit', 'sent_info'])[
        1 + (
            ('x' || substr(md5(t.trigger_id::text || ':verb1188'), 1, 8))::bit(32)::bigint % 3
        )::int
    ]
    ELSE NULL
END
WHERE t.control_group_flag IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM patient_journeys pj WHERE pj.patient_id = t.patient_id);

-- ============================================================================
-- Post-apply verification (run manually; MEASURED in the pre-apply
-- rolled-back dry run on the 37,541-row live set, 2026-07-13):
--   1) arm rates & contrast:
--        SELECT control_group_flag,
--               COUNT(*) FILTER (WHERE action_taken IS NOT NULL)::float / COUNT(*)
--        FROM triggers GROUP BY 1;
--      -> treatment 0.3868, control 0.3049 (contrast 0.0819 — clipping at the
--         0.02 floor is nearly arm-symmetric, so the 8pp design survives)
--   2) realized WS2-TR-003 relative uplift = 0.2686 (>= 0.15 target; 0.275 pre)
--   3) prognostic signal exists (the point of #1188): within-arm action rate,
--      severity>=7 vs <=3: treatment 0.7405 vs 0.0664; control 0.6588 vs 0.0230
--   4) double-apply: re-run this file; row-identical (deterministic draws).
-- ============================================================================
