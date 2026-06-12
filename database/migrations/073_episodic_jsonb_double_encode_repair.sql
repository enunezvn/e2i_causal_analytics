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
-- Pattern: migration 072 §2 exactly (per-row exception guard: any row whose
-- inner text Postgres cannot cast to jsonb is left untouched and counted;
-- NOTICE totals). Live: 137/628 raw_content rows (all model_trainer metric
-- payloads) fail the cast because Python's json.dumps emitted bare
-- NaN/Infinity tokens, which ::jsonb rejects. Those rows are DELIBERATELY
-- LEFT as string scalars: an in-SQL text rewrite cannot be made quote-aware
-- (codex R2: a global regex would also rewrite ': NaN' inside legitimate
-- string values, and the corrupted text still casts, so the per-row guard
-- cannot catch it). The production reader (hydrate_raw_content) parses them
-- fine — Python json.loads accepts the tokens Postgres rejects (pinned by
-- tests/integration/test_episodic_jsonb_shape_883c.py). If full shape
-- convergence is ever wanted, the safe path is a Python repair
-- (json.loads(txt, parse_constant=lambda _: None) -> strict json.dumps ->
-- update), NOT SQL text surgery.
--
-- Idempotent: re-running is a clean no-op (plain-cast rows are no longer
-- 'string'; NaN rows re-skip). Transactional-safe (no ALTER TYPE ... ADD
-- VALUE), so run_migrations.sh applies it wrapped as usual.
--
-- APPLICATION STATE (2026-06-12): this plain-cast pass already ran live
-- in-session (raw_content 491 repaired / 137 NaN-skipped; entities 628;
-- outcome_details 628). The file is left untracked in schema_migrations on
-- purpose so the normal user-authorized migration pipeline records it
-- through the standard path (re-run is a no-op).
-- ----------------------------------------------------------------------------

DO $$
DECLARE
    col TEXT;
    row_rec RECORD;
    repaired INTEGER;
    skipped INTEGER;
BEGIN
    FOREACH col IN ARRAY ARRAY['raw_content', 'entities', 'outcome_details']
    LOOP
        repaired := 0;
        skipped := 0;
        FOR row_rec IN EXECUTE format(
            'SELECT memory_id FROM episodic_memories WHERE jsonb_typeof(%I) = ''string''',
            col
        )
        LOOP
            BEGIN
                EXECUTE format(
                    'UPDATE episodic_memories SET %I = (%I #>> ''{}'')::jsonb WHERE memory_id = $1',
                    col, col
                ) USING row_rec.memory_id;
                repaired := repaired + 1;
            EXCEPTION WHEN others THEN
                -- Inner text not valid JSON for Postgres (the live class is
                -- Python-emitted bare NaN tokens); left untouched BY DESIGN —
                -- see header. Readers tolerate.
                skipped := skipped + 1;
            END;
        END LOOP;
        RAISE NOTICE 'episodic_memories.% repair: % repaired, % skipped', col, repaired, skipped;
    END LOOP;
END $$;
