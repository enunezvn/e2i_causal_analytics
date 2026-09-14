-- ============================================================================
-- Migration: 044_composer_episodes_feedback_id.sql
-- Purpose: Persist WHICH rating labelled a composition (#2035)
-- Dependencies: 013_tool_composer_tables.sql (composer_episodes),
--               chat/031_chatbot_message_feedback.sql (chatbot_message_feedback)
-- ============================================================================
--
-- NUMBERING: 044, not 043. On main, `ls database/ml/` ends at 042, so 043 reads as free -- but
-- 043 was already written by the refusal-reason-codes lane (043_composer_refusal_reason_codes.sql
-- + rollback_043.sql, commit 2e06d19f7 on claude/2021-2050-2020-refusal-reason-codes), which is
-- 20+ commits ahead of main with no open PR. That branch's earlier PR #2083 DID merge, without
-- carrying its 043 files, which is why main looks clear. The live ledger agrees that neither is
-- applied yet: public.schema_migrations holds ml/040, ml/041 and ml/042 only. 043 stays with the
-- lane that wrote it first.
--
-- The nightly linker (src/tasks/composition_feedback_tasks.py) recomputed the rating-to-
-- composition assignment on every run instead of recording it. The only durable trace was on
-- the composition side: success and feedback_at. Attribution that is reconstructed can change.
--
-- The failure that forces this column (codex's sequence, reproduced against the pure matcher):
--   compositions A@10:00 and B@10:10, ratings R1@10:11 and R2@10:12.
--   Run 1 assigns R1->B and R2->A. B's write succeeds; A's write fails and increments `failed`.
--   R1 is then deleted -- structurally possible and not rare-by-design:
--   chatbot_message_feedback.message_id and session_id both cascade from chatbot_messages /
--   chatbot_conversations, so deleting a message or a conversation deletes its ratings.
--   Run 2 rebuilds the claims from the SURVIVING ratings, so R2 shifts onto B. B is skipped
--   (already labelled) and A receives nothing -- the failed write is starved of its retry, and
--   B now carries a different rating than the one it consumed.
--
-- With the claim recorded, run 2 seeds the matcher's `taken` set from the persisted ids: B stays
-- claimed even though R1 is gone, R2 is free to reach A, and A gets the retry it lost.
--
-- Why a COLUMN and not a link table: the client is PostgREST (get_supabase_client), so every
-- .execute() is its own implicit transaction and no multi-statement transaction is available.
-- As a column the claim rides in the SAME single-row UPDATE as success/feedback_at (the table
-- is unique on composition_id), so the label and its attribution are written atomically. A
-- separate table needs two unspanned writes, and their partial states reproduce this very bug.
--
-- Why NO foreign key -- deliberate, and the point of the column:
--   CASCADE       deletes the EPISODE when its rating goes. The episode is the record we keep.
--   SET NULL      re-creates the starvation this migration exists to fix.
--   RESTRICT /
--   NO ACTION     makes a learning table veto deleting a user's own chat message.
-- A dangling id is the INTENDED durable trace: it says "this composition was labelled by rating
-- N", and it stays true after N is gone. The linker logs at WARNING when a labelled episode's
-- feedback_id matches no surviving rating in the window, so the divergence is detectable
-- instead of silent. The key is stable and never reused: add_feedback INSERTs and falls back to
-- an UPDATE on the unique violation, so a re-rating preserves id and created_at.
--
-- Nullable with no default and no backfill: NULL means "claim not recorded", which covers every
-- row written before this migration and every episode no rating has labelled. It is never a
-- claim by rating 0.
--
-- Additive + idempotent. scripts/run_migrations.sh applies it inside --single-transaction with
-- its ledger row, BEFORE the deploy flips the app services, so the code that writes the column
-- never runs against a table without it. Rollback: rollback_044.sql, by hand, after the code
-- revert.
-- ============================================================================

ALTER TABLE public.composer_episodes
    ADD COLUMN IF NOT EXISTS feedback_id BIGINT;

COMMENT ON COLUMN public.composer_episodes.feedback_id IS
    'chatbot_message_feedback.id of the rating that labelled this composition, alongside '
    'success and feedback_at. NULL = no claim recorded (pre-044, or never labelled). '
    'Intentionally NOT a foreign key: ratings cascade away with their message or conversation, '
    'and every referential action is worse than a dangling id here -- CASCADE would delete the '
    'episode, SET NULL would let a surviving rating re-claim the composition and starve a '
    'failed write of its retry (#2035), RESTRICT would block deleting a chat message. A '
    'dangling id is the intended durable trace; the linker warns when it matches no surviving '
    'rating. Added by migration 044 (#2035).';

-- The pgrst_ddl_watch event trigger already reloads PostgREST's schema cache on DDL; this makes
-- the reload explicit so the API can write the new column as soon as the transaction commits.
NOTIFY pgrst, 'reload schema';
