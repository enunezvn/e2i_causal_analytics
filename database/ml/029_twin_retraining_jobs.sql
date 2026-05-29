-- ============================================================================
-- Migration: 029_twin_retraining_jobs.sql
-- Purpose: Durable, shared persistence for digital-twin retraining jobs (#549)
-- Dependencies: 012_digital_twin_tables.sql (digital_twin_models)
-- ============================================================================
--
-- Before #549 a twin retraining job lived ONLY in the in-process
-- TwinRetrainingService._pending_jobs dict. A Celery worker runs in a SEPARATE
-- process from the API that queued the job, so the worker's store was empty,
-- complete_retraining() returned None, and the completion (with its real
-- validation metric) was never recorded. This table is the twin analogue of
-- ml_retraining_history (017_model_monitoring_tables.sql): a durable store so a
-- job created in the API process is found + updated by the worker and re-read
-- by the API. Backed by TwinRetrainingJobRepository
-- (src/digital_twin/twin_repository.py).
-- ============================================================================

CREATE TABLE IF NOT EXISTS twin_retraining_jobs (
    -- Job identity (the service's TwinRetrainingJob.job_id).
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The twin model being retrained, and the new model produced on success.
    -- SET NULL on delete so an archived/removed model never orphans job history.
    model_id UUID REFERENCES digital_twin_models(model_id) ON DELETE SET NULL,
    new_model_id UUID REFERENCES digital_twin_models(model_id) ON DELETE SET NULL,

    -- Why retraining was triggered (TwinTriggerReason).
    trigger_reason VARCHAR(50) NOT NULL DEFAULT 'manual',

    -- Lifecycle status (TwinRetrainingStatus).
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Fidelity before retraining (the service's fidelity_before).
    fidelity_before FLOAT NOT NULL DEFAULT 0,

    -- The REAL held-out validation R² of the retrained model. NULL until a
    -- CERTIFIED completion; left NULL on failure (the #548 fail-closed
    -- invariant — never a fabricated 0.0 that could be misread as a poor score).
    -- Intentionally UNCONSTRAINED: a finite R² can be negative, so a [0,1]
    -- range check would reject honest-but-poor metrics.
    fidelity_after FLOAT,

    -- The retraining contract (data_source / target_column / tuning knobs).
    training_config JSONB NOT NULL DEFAULT '{}',

    -- Failure reason on a failed/cancelled job (NULL on success).
    error_message TEXT,

    -- Timing.
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Constrain to the known enum values (VARCHAR + CHECK, mirroring the brand
    -- check on digital_twin_models — keeps the schema enum-free for portability).
    CONSTRAINT valid_twin_trigger_reason CHECK (
        trigger_reason IN (
            'fidelity_degradation', 'prediction_error', 'ci_coverage_drop',
            'scheduled', 'manual', 'new_data_available'
        )
    ),
    CONSTRAINT valid_twin_retraining_status CHECK (
        status IN (
            'pending', 'approved', 'training', 'validating',
            'completed', 'failed', 'cancelled'
        )
    )
);

-- Indexes mirror ml_retraining_history's access patterns.
CREATE INDEX IF NOT EXISTS idx_twin_retraining_model ON twin_retraining_jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_twin_retraining_status ON twin_retraining_jobs(status);
CREATE INDEX IF NOT EXISTS idx_twin_retraining_reason ON twin_retraining_jobs(trigger_reason);
CREATE INDEX IF NOT EXISTS idx_twin_retraining_created ON twin_retraining_jobs(created_at DESC);

COMMENT ON TABLE twin_retraining_jobs IS
    'Durable twin-retraining job state shared across the API and Celery worker '
    'processes (#549). fidelity_after holds the real validation R² only on a '
    'certified completion; NULL on failure (no fabricated metric).';
