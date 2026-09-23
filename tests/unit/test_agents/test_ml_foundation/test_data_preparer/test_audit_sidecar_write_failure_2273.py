"""#2273: a failed write of the adaptive-validity audit sidecar must be loud and reported.

#238 names the sidecar "the canonical record" (the DB table is only a mirror). On prod
the ``audit_artifacts`` volume was root-owned, so every write raised PermissionError,
and the writer's ``except`` downgraded it to a WARN: the audit trail of every real run
was lost with nothing in the run's output saying so.

Intent, kept: the sidecar is audit-only and must never block the QC gate (the
``finalize_output`` comments say a producer-side bug must not block it). So a failure
is not raised. Instead it is logged at ERROR, recorded on the state as
``adaptive_audit_sidecar = {"status": "write_failed", ...}``, carried in the agent
output, and appended to the tier-0 pipeline's ``warnings``.

The unwritable directory is REAL (``chmod 0o555`` on a tmp dir), so the PermissionError
comes from the filesystem, not from a mock.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.graph import (
    finalize_output,
    write_adaptive_verdicts_sidecar,
)
from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)

pytestmark = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions, so the unwritable dir would be writable",
)

_VERDICT = {"feature": "f1", "layer": "3", "severity": "info", "z_score": 1.0}


@pytest.fixture
def unwritable_dir(tmp_path: Path):
    """Mirror prod: the artifacts dir exists but uid != owner cannot create in it."""
    d = tmp_path / "audit_artifacts"
    d.mkdir()
    d.chmod(0o555)
    yield d
    d.chmod(0o755)


def _state() -> dict:
    return {
        "experiment_id": "exp_2273",
        "data_source": "synthetic",
        "qc_status": "passed",
        "overall_score": 0.95,
        "blocking_issues": [],
        "scope_spec": {"required_features": ["f1"]},
        "train_df": pd.DataFrame({"f1": [1.0, 2.0]}),
        "adaptive_verdicts": [dict(_VERDICT)],
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
    }


def test_failed_sidecar_write_is_logged_at_error(unwritable_dir, monkeypatch, caplog):
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(unwritable_dir))
    with caplog.at_level(logging.WARNING):
        assert write_adaptive_verdicts_sidecar(_state()) is None
    failures = [r for r in caplog.records if "sidecar" in r.getMessage().lower()]
    assert failures, "the failed write left no log line at all"
    assert all(r.levelno >= logging.ERROR for r in failures), (
        "a failed write of the canonical audit record was logged below ERROR: "
        f"{[(r.levelname, r.getMessage()) for r in failures]}"
    )
    assert "Permission denied" in failures[0].getMessage()


@pytest.mark.asyncio
async def test_finalize_output_reports_write_failed_without_blocking_the_gate(
    unwritable_dir, monkeypatch
):
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(unwritable_dir))
    updates = await finalize_output(_state())
    assert updates["gate_passed"] is True, "an audit-only write failure must not block QC"
    outcome = updates.get("adaptive_audit_sidecar")
    assert outcome is not None, "finalize_output does not report the sidecar outcome"
    assert outcome["status"] == "write_failed"
    assert outcome["path"] is None
    assert "PermissionError" in outcome["error"]


@pytest.mark.asyncio
async def test_finalize_output_reports_written_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    updates = await finalize_output(_state())
    outcome = updates["adaptive_audit_sidecar"]
    assert outcome["status"] == "written"
    assert outcome["error"] is None
    assert Path(outcome["path"]).is_file()


@pytest.mark.asyncio
async def test_finalize_output_reports_skipped_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", raising=False)
    updates = await finalize_output(_state())
    assert updates["adaptive_audit_sidecar"] == {"status": "skipped", "path": None, "error": None}


@pytest.mark.asyncio
async def test_pipeline_surfaces_a_failed_sidecar_write_as_a_run_warning():
    """The data_preparer agent is replaced by a stub here: the behaviour under test is
    the PIPELINE's reaction to the output field, and the real agent needs a live data
    source to produce it (the field itself is covered by the finalize_output tests)."""
    pipeline = MLFoundationPipeline(config=PipelineConfig(skip_mlflow=True, enable_hpo=False))
    result = PipelineResult(
        pipeline_run_id="run", status="running", current_stage=PipelineStage.DATA_PREPARATION
    )
    result.experiment_id = "exp_2273"
    result.scope_spec = {"problem_type": "binary_classification"}
    fake_dp = MagicMock()
    fake_dp.run = AsyncMock(
        return_value={
            "qc_report": {"overall_score": 0.9},
            "baseline_metrics": {},
            "gate_passed": True,
            "adaptive_audit_sidecar": {
                "status": "write_failed",
                "path": None,
                "error": "PermissionError: [Errno 13] Permission denied: '/app/data/x'",
            },
        }
    )
    with (
        patch.object(pipeline, "_get_agent", return_value=fake_dp),
        patch.object(pipeline.config, "enable_feast", False),
    ):
        await pipeline._run_data_preparation(
            input_data={"data_source": "patient_journeys"}, result=result, obs_context=None
        )
    matching = [w for w in result.warnings if "audit sidecar" in w.lower()]
    assert len(matching) == 1, f"no run warning for the lost audit record: {result.warnings}"
    assert "Permission denied" in matching[0]
