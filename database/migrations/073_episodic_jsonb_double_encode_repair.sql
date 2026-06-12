-- ============================================================================
-- Migration 073: episodic_memories JSONB double-encode repair (#883 read-side)
-- ============================================================================
-- The episodic writers (src/memory/episodic_memory.py insert_episodic_memory
-- + bulk_insert_episodic_memories) ran json.dumps() on raw_content / entities
-- / outcome_details before the supabase insert; postgrest JSON-encodes the
-- payload itself, so those JSONB columns stored JSON *string scalars* instead
-- of objects — the root cause of the gap_analyzer raw_content reader gap
-- fixed on this branch, and the same writer-bug class migration 072 repaired
-- for procedural_memories / ml_hpo_patterns / learning_signals (sibling
-- verification, #883 §5 follow-up).
--
-- Live shapes verified 2026-06-12 pre-repair (docker supabase-db):
--   episodic_memories.raw_content      628/628 'string'
--   episodic_memories.entities         628/628 'string'
--   episodic_memories.outcome_details  628/628 'string'
--
-- The writers are fixed in the same change (dicts pass through); this repairs
-- the historical rows so readers see one shape. Blast-radius census
-- (2026-06-12): the search_episodic_memory RPC returns none of these columns;
-- search_episodic_by_e2i_entity / get_enriched_episodic_memory (which select
-- them) have zero non-module consumers; hydrate_raw_content — the only
-- production reader of raw_content content — stays tolerant of both shapes.
--
-- Pattern: migration 072 §2 (per-row exception guard: any row whose inner
-- text is not valid JSON is left untouched and counted; NOTICE totals) plus a
-- second-chance pass for Python-emitted non-standard tokens: json.dumps
-- writes bare NaN/Infinity (accepted by Python json.loads, REJECTED by
-- Postgres ::jsonb). Live: 137/628 raw_content rows (all model_trainer
-- metric payloads) failed the plain cast for exactly this; value-position
-- ': NaN' / ': Infinity' / ': -Infinity' are rewritten to ': null' before
-- one retry — semantically honest (an unknown metric), and a row that STILL
-- fails is skipped and counted, never corrupted.
-- Idempotent: re-running is a clean no-op (jsonb_typeof no longer 'string').
-- Transactional-safe (no ALTER TYPE ... ADD VALUE), so run_migrations.sh
-- applies it wrapped as usual.
--
-- APPLICATION STATE (2026-06-12): the plain-cast pass ran live in-session
-- (raw_content 491 repaired / 137 NaN-skipped; entities 628; outcome_details
-- 628) BEFORE the NaN second-chance pass below was added. The file is left
-- untracked in schema_migrations on purpose so the normal migration pipeline
-- applies this final version (no-op for repaired rows, NaN->null for the
-- remaining 137). Readers (hydrate_raw_content) parse the NaN rows fine in
-- the meantime — Python json.loads accepts the tokens Postgres rejects.
-- ----------------------------------------------------------------------------

DO $$
DECLARE
    col TEXT;
    row_rec RECORD;
    fixed_text TEXT;
    repaired INTEGER;
    nan_repaired INTEGER;
    skipped INTEGER;
BEGIN
    FOREACH col IN ARRAY ARRAY['raw_content', 'entities', 'outcome_details']
    LOOP
        repaired := 0;
        nan_repaired := 0;
        skipped := 0;
        FOR row_rec IN EXECUTE format(
            'SELECT memory_id, (%I #>> ''{}'') AS txt FROM episodic_memories '
            'WHERE jsonb_typeof(%I) = ''string''',
            col, col
        )
        LOOP
            BEGIN
                EXECUTE format(
                    'UPDATE episodic_memories SET %I = (%I #>> ''{}'')::jsonb WHERE memory_id = $1',
                    col, col
                ) USING row_rec.memory_id;
                repaired := repaired + 1;
            EXCEPTION WHEN others THEN
                BEGIN
                    -- Second chance: Python json.dumps NaN/Infinity tokens in
                    -- VALUE position (after ':' or in arrays after ',' / '[')
                    -- become null; retry the cast on the rewritten text.
                    fixed_text := regexp_replace(
                        row_rec.txt,
                        '([:\[,]\s*)-?(NaN|Infinity)',
                        '\1null',
                        'g'
                    );
                    EXECUTE format(
                        'UPDATE episodic_memories SET %I = $2::jsonb WHERE memory_id = $1',
                        col
                    ) USING row_rec.memory_id, fixed_text;
                    nan_repaired := nan_repaired + 1;
                EXCEPTION WHEN others THEN
                    skipped := skipped + 1; -- still not valid JSON; readers tolerate
                END;
            END;
        END LOOP;
        RAISE NOTICE 'episodic_memories.% repair: % repaired, % nan-repaired, % skipped',
            col, repaired, nan_repaired, skipped;
    END LOOP;
END $$;
