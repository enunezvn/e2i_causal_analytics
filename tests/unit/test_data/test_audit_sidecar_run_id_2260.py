"""#2260: the adaptive-verdicts sidecar names the run that wrote it.

Runs of one scope share an experiment id (#2257) and ``written_at`` has second
resolution, so ``(experiment_id, written_at)`` no longer identifies a run. The
producer now writes the pipeline's ``audit_workflow_id`` (minted once per
``MLFoundationPipeline.run`` and threaded into the data-preparer state) and the
reader surfaces it on every ``VerdictRecord`` for the mirror's key.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit


def _state(**extra):
    from src.agents.ml_foundation.data_preparer.state import DataPreparerState

    # The graph hands finalize_output a DataPreparerState, not a dict.
    return DataPreparerState(
        experiment_id="exp_remi_al_20260610180110_119813",
        scope_spec={},
        data_source="patient_journeys",
        adaptive_verdicts=[{"feature": "disease_severity", "severity": "info"}],
        **extra,
    )


def test_the_sidecar_carries_the_graph_states_run_id(tmp_path, monkeypatch):
    from src.agents.ml_foundation.data_preparer.graph import write_adaptive_verdicts_sidecar

    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    run_id = uuid4()
    path = write_adaptive_verdicts_sidecar(_state(audit_workflow_id=run_id))
    assert path is not None
    assert json.loads(path.read_text())["audit_workflow_id"] == str(run_id)


def test_the_reader_surfaces_the_run_id_on_every_record(tmp_path, monkeypatch):
    from src.agents.ml_foundation.data_preparer.graph import write_adaptive_verdicts_sidecar
    from src.data.audit_sidecar_reader import SidecarReader

    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    run_id = uuid4()
    write_adaptive_verdicts_sidecar(_state(audit_workflow_id=run_id))
    records = list(SidecarReader(artifacts_dir=tmp_path).iter_verdict_records())
    assert [r.audit_workflow_id for r in records] == [str(run_id)]


def _write_raw(tmp_path, **payload_extra):
    sub = tmp_path / "exp"
    sub.mkdir()
    payload = {
        "schema_version": "1.9",
        "experiment_id": "exp",
        "written_at": "20260923T070000Z",
        "adaptive_verdicts": [{"feature": "f"}],
        **payload_extra,
    }
    (sub / "adaptive_verdicts_20260923T070000Z_abcdef.json").write_text(json.dumps(payload))


def test_a_legacy_sidecar_without_a_run_id_reads_as_none(tmp_path):
    from src.data.audit_sidecar_reader import SidecarReader

    _write_raw(tmp_path)
    [record] = SidecarReader(artifacts_dir=tmp_path).iter_verdict_records()
    assert record.audit_workflow_id is None


def test_a_malformed_run_id_reads_as_none_with_a_warning(tmp_path, caplog):
    """The mirror column is a UUID; a non-UUID value must not abort the whole batch."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_raw(tmp_path, audit_workflow_id="not-a-uuid")
    with caplog.at_level(logging.WARNING, logger="src.data.audit_sidecar_reader"):
        [record] = SidecarReader(artifacts_dir=tmp_path).iter_verdict_records()
    assert record.audit_workflow_id is None
    assert any("audit_workflow_id" in r.getMessage() for r in caplog.records)
