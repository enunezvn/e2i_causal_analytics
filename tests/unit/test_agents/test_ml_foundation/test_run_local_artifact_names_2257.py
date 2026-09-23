"""#2257 codex r1: run-local artifacts must not collide once runs share an experiment id.

Before #2242/#2257 every pipeline run minted its own experiment id, so the model
checkpoint (``<algo>_<experiment_id>_<second>``) and the adaptive-verdicts sidecar
(``<dir>/<experiment_id>/adaptive_verdicts_<second>.json``) were unique per run. Runs of
one scope now share the scope row's id; two runs of the same scope and algorithm in the
same second would overwrite each other's files. Each name now carries a per-write suffix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_two_checkpoints_of_one_experiment_in_one_second_do_not_collide(
    tmp_path, monkeypatch
):
    from src.agents.ml_foundation.model_trainer.nodes import checkpointer

    class _FrozenDatetime(checkpointer.datetime):  # type: ignore[misc,name-defined]
        @classmethod
        def now(cls, tz=None):
            return checkpointer.datetime(2026, 9, 23, 7, 0, 0, tzinfo=tz)

    monkeypatch.setattr(checkpointer, "datetime", _FrozenDatetime)
    state = {
        "trained_model": LogisticRegression(),
        "experiment_id": "exp_remi_al_20260610180110_119813",
        "algorithm_name": "LogisticRegression",
        "checkpoint_dir": str(tmp_path),
    }
    first = await checkpointer.save_checkpoint(dict(state))
    second = await checkpointer.save_checkpoint(dict(state))
    assert first["checkpoint_status"] == second["checkpoint_status"] == "success"
    assert first["checkpoint_path"] != second["checkpoint_path"]
    assert Path(first["checkpoint_path"]).exists() and Path(second["checkpoint_path"]).exists()
    assert "exp_remi_al_20260610180110_119813" in first["checkpoint_name"]


def test_two_sidecars_of_one_experiment_in_one_second_do_not_collide(tmp_path, monkeypatch):
    from src.agents.ml_foundation.data_preparer import graph

    class _FrozenDatetime(graph.datetime):  # type: ignore[misc,name-defined]
        @classmethod
        def now(cls, tz=None):
            return graph.datetime(2026, 9, 23, 7, 0, 0, tzinfo=tz)

    monkeypatch.setattr(graph, "datetime", _FrozenDatetime)
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    state = {
        "experiment_id": "exp_remi_al_20260610180110_119813",
        "adaptive_verdicts": [{"feature": "disease_severity", "verdict": "keep"}],
    }
    first = graph.write_adaptive_verdicts_sidecar(dict(state))
    second = graph.write_adaptive_verdicts_sidecar(dict(state))
    assert first is not None and second is not None
    assert first != second
    assert first.name.startswith("adaptive_verdicts_20260923T070000Z")  # reader glob + ts kept
    assert len(list(tmp_path.rglob("adaptive_verdicts_*.json"))) == 2
