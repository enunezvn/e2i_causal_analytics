-- ============================================================================
-- Migration 119: validation_status semantics pin + synthetic evidence backfill
-- (#1352 — refutation provenance was dormant end-to-end)
--
-- THE RULING (owner decision): STRONG semantics + dual evidence.
--   causal_paths.validation_status = 'validated' is PINNED to mean
--   "RefutationSuite evidence exists and passed", enforced in the schema.
--   The 2,729 currently-'validated' DGP-authored synthetic paths KEEP their
--   status, with consistent content-addressed synthetic refutation evidence
--   seeded behind them in causal_validations (0 rows before this migration).
--   Real paths enter as 'pending' going forward; ONLY the causal_impact
--   RefutationNode promotes them (#1352 item 3, separate lane).
--
-- ATOMICITY DESIGN (hard invariant): src/api/routes/causal.py
-- (get_causal_value_chains, the Home dashboard) filters
-- .eq("validation_status", "validated"). There must NEVER be a moment where
-- validated rows lack evidence or the dashboard's row count blinks. Two
-- mechanisms guarantee that:
--   (a) scripts/run_migrations.sh wraps this file in ONE --single-transaction
--       (this file deliberately avoids the runner's un-wrappable patterns:
--       enum ADD VALUE — which is also why the domain is a CHECK constraint
--       on the existing varchar, NOT an enum conversion — and self-managed
--       transaction control);
--   (b) within the transaction, ORDER: seed evidence for existing validated
--       rows FIRST (section 3), hard-abort if any validated row is still
--       unbacked (section 4), and only THEN install enforcement (5, 6).
--       A failure at any point rolls the whole file back — the live table is
--       either fully migrated or untouched.
--
-- CONTENT-ADDRESSED EVIDENCE (no invented random numbers): every seeded
-- metric is a pure function of the path row itself —
-- md5(path_id:test_type:causal_effect_size) -> deterministic unit fraction u
-- -> pseudo-metrics placed INSIDE RefutationRunner.PASS_THRESHOLDS' pass
-- bands (src/causal_engine/refutation_runner.py: placebo p > 0.05, effect
-- deltas < 20%, e-value >= 2.0), so the evidence is consistent with its own
-- 'passed'/'proceed' verdict. Every row is explicitly labeled synthetic
-- (details_json.is_synthetic, provenance marker, analysis_context) — mirrors
-- the no-invented-numbers discipline of the wave-1 scp_a* activity mirroring.
--
-- LINKAGE: causal_validations.estimate_id is uuid; causal_paths.path_id is
-- varchar(20). Canonical mapping (Python mirror:
-- src.repositories.causal_validation.derive_causal_path_estimate_id):
--   uuid_generate_v5(uuid_ns_url(), 'e2i:causal_paths:' || path_id)
-- Cross-language pin verified byte-identical (uuid-ossp vs Python uuid5).
-- ----------------------------------------------------------------------------


-- ----------------------------------------------------------------------------
-- 1) CANONICAL DERIVATION — path_id -> estimate_id
--    Schema-qualified: uuid-ossp lives in the `extensions` schema on this
--    Supabase, and the trigger below fires under arbitrary caller
--    search_paths (PostgREST roles), so nothing here relies on search_path.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.causal_path_estimate_id(p_path_id text)
RETURNS uuid
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT extensions.uuid_generate_v5(extensions.uuid_ns_url(), 'e2i:causal_paths:' || p_path_id);
$$;

COMMENT ON FUNCTION public.causal_path_estimate_id(text) IS
    'Canonical causal_paths.path_id -> causal_validations.estimate_id mapping (#1352, migration 119). Python mirror: src.repositories.causal_validation.derive_causal_path_estimate_id (uuid5(NAMESPACE_URL, "e2i:causal_paths:" || path_id)). Any producer or consumer of path-linked refutation evidence MUST use this derivation.';


-- ----------------------------------------------------------------------------
-- 2) SEED FUNCTION — content-addressed synthetic refutation evidence
--    One row per refutation_test_type (the 5-test RefutationSuite shape) for
--    a given synthetic path. Idempotent: NOT EXISTS per (estimate, test).
--    Reused by BOTH the section-3 backfill and the section-6 trigger's
--    auto-seed branch, so backfilled and future DGP-inserted synthetic paths
--    carry byte-identical evidence derivations.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.seed_synthetic_refutation_evidence(p public.causal_paths)
RETURNS integer
LANGUAGE plpgsql
AS $fn$
DECLARE
    v_estimate uuid    := public.causal_path_estimate_id(p.path_id);
    v_orig     numeric := COALESCE(p.causal_effect_size, 0);
    v_inserted integer := 0;
BEGIN
    INSERT INTO public.causal_validations (
        estimate_id, estimate_source, test_type, status,
        original_effect, refuted_effect, p_value, delta_percent,
        confidence_score, gate_decision, test_config, details_json,
        agent_activity_id, brand, analysis_context,
        treatment_variable, outcome_variable, data_split
    )
    SELECT
        v_estimate,
        'causal_paths',
        d.test_type::refutation_test_type,
        'passed'::validation_status,
        round(v_orig, 6),
        round(d.refuted, 6),
        d.p_value,
        CASE WHEN v_orig = 0 THEN 0
             ELSE round((d.refuted - v_orig) / v_orig * 100.0, 4)
        END,
        p.confidence_level,
        'proceed'::gate_decision,
        jsonb_build_object('synthetic_seed', true, 'source_migration', '119'),
        jsonb_build_object(
            'is_synthetic', true,
            'provenance', 'dgp_backfill_migration_119',
            'derivation',
                'content_addressed: md5(path_id || test_type || causal_effect_size) -> unit fraction -> pseudo-metrics inside RefutationRunner PASS bands',
            'content_hash', d.h,
            'issue', '#1352'
        )
        || CASE WHEN d.test_type = 'sensitivity_e_value'
                -- pass band: e-value >= 2.0 (PASS_THRESHOLDS.e_value_min)
                THEN jsonb_build_object('e_value', round(2.0 + 2.0 * d.u, 3))
                ELSE '{}'::jsonb
           END,
        NULL,  -- agent_activity_id: no fake FK — no agent ran this
        p.brand,
        'synthetic evidence backfill (#1352, migration 119): content-addressed pseudo-metrics derived from the causal_paths row; NOT output of a real RefutationSuite run',
        p.start_node,
        p.end_node,
        p.data_split::text
    FROM (
        SELECT
            t0.test_type,
            t0.h,
            -- deterministic unit fraction u in [0,1): first 8 hex chars of the
            -- content hash, zero-extended (bit(32)::bigint is unsigned here).
            (('x' || substr(t0.h, 1, 8))::bit(32)::bigint)::numeric / 4294967296.0 AS u
        FROM (
            SELECT
                tt AS test_type,
                md5(p.path_id || ':' || tt || ':' || COALESCE(p.causal_effect_size::text, '')) AS h
            FROM unnest(ARRAY[
                'placebo_treatment',
                'random_common_cause',
                'data_subset',
                'bootstrap',
                'sensitivity_e_value'
            ]) AS tt
        ) t0
    ) t
    CROSS JOIN LATERAL (
        SELECT
            t.test_type,
            t.h,
            t.u,
            CASE t.test_type
                -- placebo: effect must vanish (near-zero refuted effect) and
                -- the placebo p-value must clear the pass band (> 0.05).
                WHEN 'placebo_treatment'   THEN v_orig * 0.05 * t.u
                -- delta-based tests: refuted effect within the < 20% pass
                -- band of PASS_THRESHOLDS (common_cause_delta et al).
                WHEN 'random_common_cause' THEN v_orig * (1.0 + (t.u - 0.5) * 0.20)
                WHEN 'data_subset'         THEN v_orig * (1.0 + (t.u - 0.5) * 0.16)
                WHEN 'bootstrap'           THEN v_orig * (1.0 + (t.u - 0.5) * 0.10)
                -- sensitivity: effect unchanged; the derived e-value (above)
                -- carries the pass signal.
                ELSE v_orig
            END AS refuted,
            CASE t.test_type
                WHEN 'placebo_treatment' THEN round(0.50 + 0.45 * t.u, 5)
                ELSE NULL
            END AS p_value
    ) d
    WHERE NOT EXISTS (
        SELECT 1 FROM public.causal_validations cv
        WHERE cv.estimate_id     = v_estimate
          AND cv.estimate_source = 'causal_paths'
          AND cv.test_type       = d.test_type::refutation_test_type
    )
    -- Concurrency belt on top of the NOT EXISTS: two overlapping reseeds
    -- cannot see each other's uncommitted rows, so the partial unique index
    -- (section 2b) arbitrates and the loser no-ops instead of duplicating.
    ON CONFLICT (estimate_id, estimate_source, test_type)
        WHERE details_json->>'provenance' = 'dgp_backfill_migration_119'
        DO NOTHING;

    GET DIAGNOSTICS v_inserted = ROW_COUNT;
    RETURN v_inserted;
END;
$fn$;

-- ----------------------------------------------------------------------------
-- 2b) SEED-UNIQUENESS GUARD — one synthetic-seed row per (estimate, test).
--    Deliberately a PARTIAL unique index scoped to the migration-119 seed
--    provenance: causal_validations is a HISTORY log for the real writer
--    (RefutationNode persists one row per test per RUN — repeated runs are
--    legitimate history, see CausalValidationRepository docstrings), so a
--    table-wide unique constraint would break that data model. Synthetic
--    seeds, by contrast, are content-addressed and must be singletons.
--    Live table measured at 0 rows pre-migration; created before the
--    backfill, inside the same transaction.
-- ----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_cv_synthetic_seed_estimate_test
    ON public.causal_validations (estimate_id, estimate_source, test_type)
    WHERE details_json->>'provenance' = 'dgp_backfill_migration_119';

COMMENT ON FUNCTION public.seed_synthetic_refutation_evidence(public.causal_paths) IS
    'Seeds the 5-test content-addressed SYNTHETIC refutation-evidence suite behind a synthetic causal path (#1352, migration 119). Fully deterministic from the path row — hash-derived, never sampled; every row labeled synthetic in details_json/analysis_context. Used by the migration-119 backfill AND the validated-requires-evidence trigger auto-seed branch.';


-- ----------------------------------------------------------------------------
-- 3) BACKFILL — evidence for every currently-validated synthetic path.
--    Measured pre-migration state (2026-07-30): 2,729 causal_paths rows, ALL
--    validation_status='validated' AND is_synthetic=true; causal_validations
--    empty. Expected: 2,729 x 5 = 13,645 evidence rows. Idempotent re-run
--    seeds 0.
-- ----------------------------------------------------------------------------
DO $$
DECLARE
    v_rows bigint;
BEGIN
    SELECT COALESCE(sum(public.seed_synthetic_refutation_evidence(cp)), 0)
    INTO v_rows
    FROM public.causal_paths cp
    WHERE cp.validation_status = 'validated'
      AND cp.is_synthetic;
    RAISE NOTICE 'migration 119: seeded % synthetic refutation-evidence rows', v_rows;
END $$;


-- ----------------------------------------------------------------------------
-- 4) ATOMICITY GATE — refuse to pin semantics over an unbacked status.
--    If ANY validated row still lacks passed evidence here (e.g. a divergent
--    environment holding real validated paths, which synthetic seeding
--    rightly does not bless), the WHOLE transaction rolls back: no evidence,
--    no constraint, no trigger — dashboard state untouched. Silent demotion
--    to 'pending' is deliberately NOT done: it would change the dashboard's
--    validated-row count (the no-blink invariant) without an operator
--    decision.
-- ----------------------------------------------------------------------------
DO $$
DECLARE
    v_missing integer;
BEGIN
    SELECT count(*) INTO v_missing
    FROM public.causal_paths cp
    WHERE cp.validation_status = 'validated'
      AND NOT EXISTS (
          SELECT 1 FROM public.causal_validations cv
          WHERE cv.estimate_id     = public.causal_path_estimate_id(cp.path_id)
            AND cv.estimate_source = 'causal_paths'
            AND cv.status          = 'passed'
      );
    IF v_missing > 0 THEN
        RAISE EXCEPTION 'migration 119 aborted: % causal_paths rows are validated WITHOUT passed refutation evidence (non-synthetic rows are never auto-blessed). Whole transaction rolls back; live state unchanged. Operator must adjudicate those rows before re-running.', v_missing;
    END IF;
END $$;


-- ----------------------------------------------------------------------------
-- 5) SEMANTICS PIN — value domain, default, nullability.
--    CHECK constraint (not an enum): every listed value is referenced by live
--    code — 'validated' (DGP + causal.py dashboard gate), 'pending' (new
--    real-path default), 'needs_review'/'pending' (src/ml/data_generator),
--    'overturned' (memory consolidator skip-rule), 'refuted' (the demotion
--    value the RefutationNode promoter lane needs). An enum conversion would
--    force non-transactional ADD VALUE statements on future extension,
--    breaking exactly the single-transaction atomicity this file depends on.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname  = 'causal_paths_validation_status_domain_chk'
          AND conrelid = 'public.causal_paths'::regclass
    ) THEN
        ALTER TABLE public.causal_paths
            ADD CONSTRAINT causal_paths_validation_status_domain_chk
            CHECK (validation_status IN ('pending', 'validated', 'needs_review', 'overturned', 'refuted'));
    END IF;
END $$;

-- New paths enter 'pending' unless a writer explicitly claims otherwise (and
-- a 'validated' claim must then survive the section-6 trigger). Measured: 0
-- NULL statuses pre-migration, so SET NOT NULL is a no-op scan.
ALTER TABLE public.causal_paths ALTER COLUMN validation_status SET DEFAULT 'pending';
ALTER TABLE public.causal_paths ALTER COLUMN validation_status SET NOT NULL;

COMMENT ON COLUMN public.causal_paths.validation_status IS
    'PINNED SEMANTICS (#1352, migration 119): ''validated'' asserts "RefutationSuite evidence exists and passed" — enforced by trg_causal_paths_validated_evidence (passed causal_validations rows under causal_path_estimate_id(path_id)). Real paths enter ''pending''; only the causal_impact RefutationNode promotes them. ''refuted''/''overturned'' mark demoted paths; ''needs_review'' awaits adjudication. Domain: causal_paths_validation_status_domain_chk.';


-- ----------------------------------------------------------------------------
-- 6) ENFORCEMENT TRIGGER — 'validated' requires passed evidence.
--    * real rows (is_synthetic = false): claiming 'validated' without passed
--      evidence is REJECTED (check_violation) — only the RefutationNode
--      promoter path, which persists a passed suite first, can flip them;
--    * synthetic rows: the DGP reseeds legitimately author 'validated'
--      (operational pattern: batch_loader upserts), so the trigger AUTO-SEEDS
--      the same content-addressed evidence instead of breaking every reseed —
--      the invariant "validated => passed evidence exists" holds universally
--      either way.
--    Scope note: evidence DELETION is unguarded by design — nothing deletes
--    from causal_validations today; a delete-side guard is the RefutationNode
--    lane's call if demotion mechanics ever need it.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.enforce_validated_requires_refutation_evidence()
RETURNS trigger
LANGUAGE plpgsql
AS $fn$
DECLARE
    v_estimate uuid;
BEGIN
    IF NEW.validation_status = 'validated' THEN
        v_estimate := public.causal_path_estimate_id(NEW.path_id);
        IF NOT EXISTS (
            SELECT 1 FROM public.causal_validations cv
            WHERE cv.estimate_id     = v_estimate
              AND cv.estimate_source = 'causal_paths'
              AND cv.status          = 'passed'
        ) THEN
            IF NEW.is_synthetic THEN
                PERFORM public.seed_synthetic_refutation_evidence(NEW);
                -- Re-verify: seeding skips (estimate, test) pairs that already
                -- hold evidence, so pre-existing NON-passed rows (e.g. a real
                -- refutation run that FAILED this path) can leave the invariant
                -- unmet. Never mark validated over contradicting evidence, and
                -- never overwrite recorded refutation output to force a pass.
                IF NOT EXISTS (
                    SELECT 1 FROM public.causal_validations cv
                    WHERE cv.estimate_id     = v_estimate
                      AND cv.estimate_source = 'causal_paths'
                      AND cv.status          = 'passed'
                ) THEN
                    RAISE EXCEPTION USING
                        ERRCODE = 'check_violation',
                        MESSAGE = format(
                            'causal_paths.%s: synthetic path claims validation_status=''validated'' but existing non-passed refutation evidence under estimate_id %s blocks it (#1352, migration 119). Refusing to validate over contradicting evidence.',
                            NEW.path_id, v_estimate
                        ),
                        HINT = 'Recorded evidence for this path contains no ''passed'' rows and auto-seeding will not overwrite it. Resolve the conflict explicitly: set the path to ''refuted''/''needs_review'', or adjudicate and remove the stale evidence.';
                END IF;
            ELSE
                RAISE EXCEPTION USING
                    ERRCODE = 'check_violation',
                    MESSAGE = format(
                        'causal_paths.%s: validation_status=''validated'' asserts "RefutationSuite evidence exists and passed" (#1352, migration 119), but no passed causal_validations rows exist under estimate_id %s.',
                        NEW.path_id, v_estimate
                    ),
                    HINT = 'Real paths enter as ''pending''. Only the causal_impact RefutationNode may promote them: persist a passed RefutationSuite under causal_path_estimate_id(path_id) FIRST, then set validation_status=''validated''.';
            END IF;
        END IF;
    END IF;
    RETURN NEW;
END;
$fn$;

DROP TRIGGER IF EXISTS trg_causal_paths_validated_evidence ON public.causal_paths;
CREATE TRIGGER trg_causal_paths_validated_evidence
    BEFORE INSERT OR UPDATE OF validation_status ON public.causal_paths
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_validated_requires_refutation_evidence();

COMMENT ON TRIGGER trg_causal_paths_validated_evidence ON public.causal_paths IS
    'Schema enforcement of the #1352 semantics pin: validated => passed refutation evidence exists in causal_validations. Synthetic rows auto-seed content-addressed evidence (DGP reseed compatibility); real rows are rejected until the RefutationNode persists a passed suite.';

NOTIFY pgrst, 'reload schema';
-- (No transaction control here; run_migrations.sh owns the outer --single-transaction.)
