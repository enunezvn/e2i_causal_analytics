"""#891 writer-side fix: model_trainer's episodic contribution must never emit
bare NaN/Infinity.

RED (pre-fix), probe-verified live 2026-06-12: a non-finite float anywhere in
the training-result payload makes supabase-py's strict JSON encoder raise
``ValueError: Out of range float values are not JSON compliant: nan`` BEFORE
the request is sent; ``store_training_result`` swallows the exception and
returns ``None`` — the whole episodic write is SILENTLY DROPPED. (This is the
modern variant of the bug class that produced the 137 NaN-bearing
string-scalar rows migration 073 had to skip: the pre-#888 writer json.dumps'd
the payload itself, which let the bare tokens reach the column.)

GREEN: the central episodic writers sanitize JSONB-bound payloads
(non-finite floats -> JSON null) via ``sanitize_jsonb_payload``, so the
NaN-bearing result persists as a proper JSONB object with nulls, and string
values containing the literal text ``NaN`` survive byte-identical (the
codex-R2 corruption payload, also pinned by
tests/integration/test_episodic_jsonb_shape_883c.py).

model_trainer produces these non-finite metrics for real: stacking
``float("nan")`` fold means, advanced_validation ``brier_*`` NaN defaults,
learning_curve ``score_mean``/``score_std`` NaN — so this is live data loss,
not a hypothetical.

Run with the shared-DB lock::

    flock /tmp/e2i_dbtest.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_model_trainer_nan_sanitize_891.py'
"""

import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB writer-sanitize test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]

# The exact string value codex R2 flagged as the corruption canary: it must
# survive VERBATIM through the writer (quote-aware by construction).
CODEX_R2_NOTE = "threshold: NaN means missing, Infinity capped"


def _cleanup(memory_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("memory_id", memory_id).execute()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_store_training_result_with_nan_metrics_persists_sanitized():
    """RED: returns None (insert raised ValueError on the NaN float, hook
    swallowed it — silent episodic data loss). GREEN: returns a memory_id and
    the stored raw_content is a proper JSONB object with non-finite floats as
    nulls and string content byte-identical."""
    from src.agents.ml_foundation.model_trainer.memory_hooks import ModelTrainerMemoryHooks
    from src.memory.episodic_memory import get_supabase_client

    marker = f"891-writer-{uuid.uuid4().hex[:12]}"
    hooks = ModelTrainerMemoryHooks()

    result = {
        "training_run_id": marker,
        "model_id": f"model-{marker}",
        "test_metrics": {
            "auc_roc": 0.81,
            "rmse": float("nan"),
            "brier_reliability": float("inf"),
        },
        "train_metrics": {"log_loss": float("-inf")},
        "validation_metrics": {},
        "success_criteria_met": True,
        "mlflow_run_id": None,
        "model_artifact_uri": None,
        "total_training_duration_seconds": 12.5,
    }
    state = {
        "experiment_id": f"exp-{marker}",
        "algorithm_name": "xgboost",
        "best_hyperparameters": {"max_depth": 6, "note": CODEX_R2_NOTE},
        "session_id": str(uuid.uuid4()),
    }

    memory_id = await hooks.store_training_result(
        session_id=state["session_id"],
        result=result,
        state=state,
    )
    assert memory_id, (
        "NaN-bearing training result was SILENTLY DROPPED: store_training_result "
        "returned None because supabase-py's strict JSON encoder raised on the "
        "non-finite float and the hook swallowed it (#891 writer bug)"
    )

    try:
        row = (
            get_supabase_client()
            .table("episodic_memories")
            .select("raw_content")
            .eq("memory_id", memory_id)
            .execute()
        ).data[0]
        rc = row["raw_content"]
        # Proper JSONB object (dict through PostgREST), not a string scalar.
        assert isinstance(rc, dict), f"raw_content landed as {type(rc).__name__}, not object"
        # Non-finite floats mapped to JSON null...
        assert rc["test_metrics"]["rmse"] is None
        assert rc["test_metrics"]["brier_reliability"] is None
        assert rc["train_metrics"]["log_loss"] is None
        # ...finite values untouched...
        assert rc["test_metrics"]["auc_roc"] == 0.81
        assert rc["algorithm_name"] == "xgboost"
        # ...and the codex-R2 string canary survives byte-identical.
        assert rc["best_hyperparameters"]["note"] == CODEX_R2_NOTE
        assert rc["best_hyperparameters"]["note"].encode() == CODEX_R2_NOTE.encode()
    finally:
        _cleanup(memory_id)
