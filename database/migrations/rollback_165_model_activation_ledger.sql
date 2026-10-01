-- Rollback for migration 165 (NOT auto-applied: the runner skips rollback_* files).
-- Run it as the runbook runs rollbacks: psql as postgres, one transaction.
--
-- Refuses while any activation is live (prepared / serving_switched / active / aborting /
-- rolling_back): roll it back or abort it with scripts/model_activation.py first, otherwise
-- the registry would be left with the activated roles and no ledger to undo them.
--
-- Keeps the audit history: the ledger and the allowlist are RENAMED to *_retired_165 (their
-- indexes too, so a re-applied 165 can create its own), readable by postgres only. Drops the
-- registry role guard, the ledger triggers, every 165 function and the (always empty between
-- transactions) RPC authority table, then removes 165's ledger
-- row so a later deploy re-applies it (the 164 rollback does the same).
--
-- A second run after a clean first run is a no-op apart from the ledger-row DELETE. A second
-- apply-then-rollback cycle fails loudly on the rename (the *_retired_165 names are taken):
-- rename the first retired tables by hand before it.

-- The live-row check runs under ACCESS EXCLUSIVE on the ledger (held to the end of this
-- transaction): an insert still in flight either commits first and is seen here, or waits
-- until the rollback is done and then fails on the renamed table (codex r1). A registry writer
-- whose role guard needs the ledger meanwhile fails at once (the guard takes it NOWAIT) instead
-- of deadlocking with the DROP TRIGGER below (codex r2): retry it after the rollback.
DO $$
BEGIN
    IF to_regclass('public.ml_model_activations') IS NOT NULL THEN
        LOCK TABLE public.ml_model_activations IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM public.ml_model_activations
                    WHERE phase IN ('prepared', 'serving_switched', 'active', 'aborting', 'rolling_back')) THEN
            RAISE EXCEPTION 'live activation(s) in ml_model_activations: roll back or abort them first';
        END IF;
    END IF;
END $$;

DROP TRIGGER IF EXISTS tr_ml_model_registry_activation_role_guard ON public.ml_model_registry;
DROP TRIGGER IF EXISTS tr_ml_model_activations_guard ON public.ml_model_activations;
DROP TRIGGER IF EXISTS tr_ml_model_activations_insert_guard ON public.ml_model_activations;

ALTER TABLE IF EXISTS public.ml_model_activations RENAME TO ml_model_activations_retired_165;
ALTER INDEX IF EXISTS public.ml_model_activations_pkey RENAME TO ml_model_activations_retired_165_pkey;
ALTER INDEX IF EXISTS public.uq_ml_model_activations_one_live RENAME TO uq_ml_model_activations_retired_165_one_live;
ALTER INDEX IF EXISTS public.uq_ml_model_activations_candidate_once RENAME TO uq_ml_model_activations_retired_165_candidate_once;
ALTER TABLE IF EXISTS public.ml_activation_production_allowlist RENAME TO ml_activation_production_allowlist_retired_165;
ALTER INDEX IF EXISTS public.ml_activation_production_allowlist_pkey RENAME TO ml_activation_production_allowlist_retired_165_pkey;

DO $$
BEGIN
    IF to_regclass('public.ml_model_activations_retired_165') IS NOT NULL THEN
        REVOKE ALL ON TABLE public.ml_model_activations_retired_165 FROM PUBLIC, anon, authenticated, service_role;
    END IF;
    IF to_regclass('public.ml_activation_production_allowlist_retired_165') IS NOT NULL THEN
        REVOKE ALL ON TABLE public.ml_activation_production_allowlist_retired_165 FROM PUBLIC, anon, authenticated, service_role;
    END IF;
END $$;

DROP FUNCTION IF EXISTS public.ml_model_registry_activation_role_guard();
DROP FUNCTION IF EXISTS public._activation_ledger_share_nowait();
DROP FUNCTION IF EXISTS public.rollback_model_activation(uuid);
DROP FUNCTION IF EXISTS public.activate_model_candidate(uuid);
DROP FUNCTION IF EXISTS public._activation_lock_and_count_served(text, uuid);
DROP FUNCTION IF EXISTS public.ml_model_activations_insert_guard();
DROP FUNCTION IF EXISTS public.ml_model_activations_guard();
DROP FUNCTION IF EXISTS public.activation_gate_passes(jsonb, text, text, text, text);
DROP FUNCTION IF EXISTS public._activation_num(jsonb, text[]);
DROP FUNCTION IF EXISTS public.activation_gate_config();
DROP TABLE IF EXISTS public.ml_activation_rpc_authority;   -- transient RPC tokens: nothing to keep

DELETE FROM public.schema_migrations
 WHERE filename = '165_model_activation_ledger.sql';

NOTIFY pgrst, 'reload schema';
