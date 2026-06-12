-- ============================================================================
-- Migration 072: learning_signals deferred items (#883 §5 / PR #884 "flagged,
-- not fixed here") — cycle_id FK ON DELETE CASCADE + JSONB double-encode repair
-- ============================================================================
-- Section 1 — missing FK. The SSOT
-- (database/memory/001_agentic_memory_schema_v1.3.sql:502) declares
--     cycle_id UUID REFERENCES cognitive_cycles(cycle_id) ON DELETE CASCADE
-- but the live table carried NO cycle_id constraint at all (pg_constraint
-- showed only the hcp/patient/trigger FKs), so deleting a cognitive cycle
-- stranded its learning signals as orphans. Verified live 2026-06-12 before
-- this migration: 0 orphan rows (all 300 rows had cycle_id NULL), so the
-- defensive orphan NULL-out below is a no-op on this DB — it exists so the
-- ADD CONSTRAINT can never fail on an environment that DOES hold orphans
-- (NULL FKs are not enforced, and SSOT keeps the column nullable).
-- Idempotency semantics (codex R1 MEDIUM): the existence check requires
-- CASCADE delete behavior (confdeltype = 'c'), not just any cycle_id FK — a
-- drifted non-cascade variant is dropped and recreated as cascade. A fresh DB
-- built from SSOT 001 (auto-named learning_signals_cycle_id_fkey, cascade) is
-- recognized and left alone. Constraint name follows the table's existing
-- fk_signals_* family.
--
-- Section 2 — JSONB string-scalar repair. record_learning_signal /
-- insert_procedural_memory / hpo_pattern_memory (store_hpo_pattern) ran
-- json.dumps() on structured payloads before the supabase insert; postgrest
-- JSON-encodes the payload itself, so those JSONB columns stored JSON
-- *string scalars* instead of objects/arrays. Live shapes verified
-- 2026-06-12 pre-repair:
--   procedural_memories.tool_sequence   1566/1566 'string'
--   ml_hpo_patterns.best_hyperparameters 887/887  'string'
--   ml_hpo_patterns.search_space / feature_types   same writer, same bug
--   learning_signals.signal_details        0/300  (defensive no-op here)
-- The writers are fixed in the same change; this repairs historical rows so
-- readers see one shape. Per-row exception guard: any row whose inner text is
-- not valid JSON is left untouched (readers remain shape-tolerant).
--
-- Idempotent: re-running is a clean no-op. Transactional-safe (no ALTER TYPE
-- ... ADD VALUE), so run_migrations.sh applies it wrapped as usual.
-- ----------------------------------------------------------------------------

-- Section 1: orphan guard + FK ON DELETE CASCADE (SSOT parity)
DO $$
DECLARE
    orphans INTEGER;
    drifted RECORD;
BEGIN
    UPDATE learning_signals ls
    SET cycle_id = NULL
    WHERE ls.cycle_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM cognitive_cycles cc WHERE cc.cycle_id = ls.cycle_id
      );
    GET DIAGNOSTICS orphans = ROW_COUNT;
    IF orphans > 0 THEN
        RAISE NOTICE 'learning_signals: NULLed % orphan cycle_id row(s) before adding FK', orphans;
    END IF;

    -- Drop any drifted cycle_id FK that lacks CASCADE delete semantics so the
    -- recreate below restores the SSOT-declared behavior.
    FOR drifted IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'learning_signals'::regclass
          AND contype = 'f'
          AND confrelid = 'cognitive_cycles'::regclass
          AND confdeltype <> 'c'
          AND pg_get_constraintdef(oid) ILIKE '%(cycle_id)%'
    LOOP
        EXECUTE format('ALTER TABLE learning_signals DROP CONSTRAINT %I', drifted.conname);
        RAISE NOTICE 'learning_signals: dropped non-cascade cycle_id FK %', drifted.conname;
    END LOOP;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'learning_signals'::regclass
          AND contype = 'f'
          AND confrelid = 'cognitive_cycles'::regclass
          AND confdeltype = 'c'
          AND pg_get_constraintdef(oid) ILIKE '%(cycle_id)%'
    ) THEN
        ALTER TABLE learning_signals
            ADD CONSTRAINT fk_signals_cycle
            FOREIGN KEY (cycle_id)
            REFERENCES cognitive_cycles(cycle_id)
            ON DELETE CASCADE;
        RAISE NOTICE 'learning_signals: added fk_signals_cycle (ON DELETE CASCADE)';
    END IF;
END $$;

-- Section 2: repair double-encoded JSONB string scalars -> real objects/arrays
DO $$
DECLARE
    spec RECORD;
    row_rec RECORD;
    repaired INTEGER;
    skipped INTEGER;
BEGIN
    FOR spec IN
        SELECT *
        FROM (VALUES
            ('procedural_memories', 'procedure_id', 'tool_sequence'),
            ('learning_signals', 'signal_id', 'signal_details'),
            ('ml_hpo_patterns', 'pattern_id', 'search_space'),
            ('ml_hpo_patterns', 'pattern_id', 'best_hyperparameters'),
            ('ml_hpo_patterns', 'pattern_id', 'feature_types')
        ) AS t(tbl, pk, col)
    LOOP
        repaired := 0;
        skipped := 0;
        FOR row_rec IN EXECUTE format(
            'SELECT %I AS id FROM %I WHERE jsonb_typeof(%I) = ''string''',
            spec.pk, spec.tbl, spec.col
        )
        LOOP
            BEGIN
                EXECUTE format(
                    'UPDATE %I SET %I = (%I #>> ''{}'')::jsonb WHERE %I = $1',
                    spec.tbl, spec.col, spec.col, spec.pk
                ) USING row_rec.id;
                repaired := repaired + 1;
            EXCEPTION WHEN others THEN
                skipped := skipped + 1; -- inner text not valid JSON; readers tolerate
            END;
        END LOOP;
        RAISE NOTICE '%.% repair: % repaired, % skipped', spec.tbl, spec.col, repaired, skipped;
    END LOOP;
END $$;
