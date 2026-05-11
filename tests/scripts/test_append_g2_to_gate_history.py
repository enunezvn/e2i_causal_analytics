"""Unit tests for ``scripts/append_g2_to_gate_history.py``.

Pins the MED-10 contract: G2 manifest is decomposed into per-threshold
``GateEvaluationEntry``-shaped records that the N1 audit trail can
ingest. The fixture below is the canonical happy-path manifest shape
emitted by ``run_tier1b_b2_experiment.run_experiment``.

MED-10 (pass-2) N1 integration: ``TestN1AuditIntegration`` verifies
that calling ``append_to_n1_audit`` routes entries through the canonical
``RegulatoryEligibilityAudit.append_gate_evaluation`` API and that the
records land in ``gate_history`` where N1 expects them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import append_g2_to_gate_history as A  # noqa: E402

from src.agents.ml_foundation.model_deployer.regulatory_audit import (  # noqa: E402
    RegulatoryEligibilityAudit,
)


@pytest.fixture
def passing_manifest() -> dict:
    """Canonical happy-path manifest where all three thresholds pass."""
    return {
        "experiment_commit_sha": "abc123def456",
        "cohort_label": "optum_initiation_default",
        "cohort_data_dir": "data/rwd/optum/initiation",
        "cohort_target": "treatment_initiated",
        "cohort_data_snooped": False,
        "dataset_hashes": {
            "patient_journeys_parquet": "0" * 64,
            "treatment_events_parquet": "1" * 64,
        },
        "lifecycle_state": "advisory",
        "thresholds": [
            {
                "name": "T1",
                "description": "dAUC >= 0.03",
                "threshold": 0.03,
                "delta": 0.05,
                "passes": True,
                "rationale": "dAUC = 0.05 >= 0.03; pre=0.60, post=0.65",
                "pre_value": 0.60,
                "post_value": 0.65,
            },
            {
                "name": "T2",
                "description": "ECE_post <= 0.5 * ECE_pre",
                "threshold": 0.5,
                "delta": 0.40,
                "passes": True,
                "rationale": "ECE ratio = 0.40 <= 0.5; pre=0.10, post=0.04",
                "pre_value": 0.10,
                "post_value": 0.04,
            },
            {
                "name": "T3",
                "description": "(std/mean)_post <= 0.7 * (std/mean)_pre",
                "threshold": 0.7,
                "delta": 0.60,
                "passes": True,
                "rationale": "CV-stability ratio = 0.60 <= 0.7",
                "pre_value": 0.10,
                "post_value": 0.06,
            },
        ],
        "g2_passes_pre_spec": True,
    }


@pytest.fixture
def failing_t1_manifest(passing_manifest: dict) -> dict:
    """Variant where T1 fails — combined verdict must reflect."""
    passing_manifest["thresholds"][0]["passes"] = False
    passing_manifest["thresholds"][0]["delta"] = 0.01
    passing_manifest["g2_passes_pre_spec"] = False
    return passing_manifest


class TestBuildAuditEntries:
    def test_returns_four_entries_one_per_threshold_plus_combined(
        self, passing_manifest: dict
    ) -> None:
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="refs/tags/tier1b-b2-experiment-1",
            tag_sha="abc123def456",
            s_prespec_sha="7f616f6f",
            workflow_run_id="12345",
        )
        assert len(entries) == 4
        gate_names = [e["gate_name"] for e in entries]
        assert gate_names == ["G2_T1", "G2_T2", "G2_T3", "G2"]

    def test_passing_manifest_all_outcomes_pass(self, passing_manifest: dict) -> None:
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="refs/tags/x",
            tag_sha="abc",
            s_prespec_sha="def",
            workflow_run_id="1",
        )
        for entry in entries:
            assert entry["outcome"] == "pass", f"{entry['gate_name']} should pass: {entry}"

    def test_failing_t1_makes_combined_fail(self, failing_t1_manifest: dict) -> None:
        entries = A.build_audit_entries(
            manifest=failing_t1_manifest,
            tag_ref="refs/tags/x",
            tag_sha="abc",
            s_prespec_sha="def",
            workflow_run_id="1",
        )
        # T1 fails, T2/T3 still pass, combined fails.
        outcomes = {e["gate_name"]: e["outcome"] for e in entries}
        assert outcomes["G2_T1"] == "fail"
        assert outcomes["G2_T2"] == "pass"
        assert outcomes["G2_T3"] == "pass"
        assert outcomes["G2"] == "fail"

    def test_provenance_block_present_on_every_entry(self, passing_manifest: dict) -> None:
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="refs/tags/tier1b-b2-experiment-1",
            tag_sha="abc123def456",
            s_prespec_sha="7f616f6f",
            workflow_run_id="12345",
        )
        for entry in entries:
            prov = entry["g2_provenance"]
            assert prov["tag_ref"] == "refs/tags/tier1b-b2-experiment-1"
            assert prov["tag_sha"] == "abc123def456"
            assert prov["s_prespec_sha"] == "7f616f6f"
            assert prov["workflow_run_id"] == "12345"
            assert prov["cohort_label"] == "optum_initiation_default"
            assert prov["dataset_hashes"]["patient_journeys_parquet"] == "0" * 64

    def test_threshold_provenance_is_literature_anchored(self, passing_manifest: dict) -> None:
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="x",
            tag_sha="y",
            s_prespec_sha="z",
            workflow_run_id="1",
        )
        for entry in entries:
            assert entry["threshold_provenance"] == "literature_anchored"

    def test_timestamp_format(self, passing_manifest: dict) -> None:
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="x",
            tag_sha="y",
            s_prespec_sha="z",
            workflow_run_id="1",
        )
        # All entries share the same timestamp (single audit pass).
        ts_set = {e["timestamp"] for e in entries}
        assert len(ts_set) == 1
        ts = ts_set.pop()
        # ISO-8601-like UTC: YYYY-MM-DDTHH:MM:SSZ
        assert "T" in ts and ts.endswith("Z")


class TestMain:
    def test_main_writes_entries_to_output(self, passing_manifest: dict, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(passing_manifest), encoding="utf-8")
        output_path = tmp_path / "audit.json"
        rc = A.main(
            [
                "--manifest",
                str(manifest_path),
                "--tag-ref",
                "refs/tags/tier1b-b2-experiment-1",
                "--tag-sha",
                "abc",
                "--s-prespec-sha",
                "def",
                "--workflow-run-id",
                "1",
                "--output",
                str(output_path),
                "--audit-output",
                str(tmp_path / "audit_state.json"),
            ]
        )
        assert rc == 0
        assert output_path.exists()
        entries = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(entries) == 4
        gate_names = [e["gate_name"] for e in entries]
        assert "G2" in gate_names
        assert "G2_T1" in gate_names

    def test_main_returns_nonzero_on_missing_manifest(self, tmp_path: Path) -> None:
        rc = A.main(
            [
                "--manifest",
                str(tmp_path / "nonexistent.json"),
                "--tag-ref",
                "x",
                "--tag-sha",
                "y",
                "--s-prespec-sha",
                "z",
                "--workflow-run-id",
                "1",
                "--output",
                str(tmp_path / "audit.json"),
                "--audit-output",
                str(tmp_path / "audit_state.json"),
            ]
        )
        assert rc == 1

    def test_main_returns_nonzero_on_invalid_json(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("not valid json {", encoding="utf-8")
        rc = A.main(
            [
                "--manifest",
                str(manifest_path),
                "--tag-ref",
                "x",
                "--tag-sha",
                "y",
                "--s-prespec-sha",
                "z",
                "--workflow-run-id",
                "1",
                "--output",
                str(tmp_path / "audit.json"),
                "--audit-output",
                str(tmp_path / "audit_state.json"),
            ]
        )
        assert rc == 1

    def test_main_writes_audit_output_via_n1_api(
        self, passing_manifest: dict, tmp_path: Path
    ) -> None:
        """MED-10 (pass-2): --audit-output must contain the G2 entries
        readable by RegulatoryEligibilityAudit.from_dict.  This verifies
        the N1 API is the load-bearing path, not just a side call."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(passing_manifest), encoding="utf-8")
        output_path = tmp_path / "shim.json"
        audit_output_path = tmp_path / "audit_state.json"

        rc = A.main(
            [
                "--manifest",
                str(manifest_path),
                "--tag-ref",
                "refs/tags/tier1b-b2-experiment-1",
                "--tag-sha",
                "abc",
                "--s-prespec-sha",
                "def",
                "--workflow-run-id",
                "1",
                "--output",
                str(output_path),
                "--audit-output",
                str(audit_output_path),
            ]
        )
        assert rc == 0
        assert audit_output_path.exists()

        # Read the audit snapshot back through RegulatoryEligibilityAudit.from_dict
        # so we assert on the canonical N1 API surface, not raw JSON.
        audit_payload = json.loads(audit_output_path.read_text(encoding="utf-8"))
        audit = RegulatoryEligibilityAudit.from_dict(audit_payload)

        gate_history = audit.gate_history
        assert len(gate_history) == 4, (
            f"Expected 4 gate entries (T1, T2, T3, G2 combined); got {len(gate_history)}"
        )
        gate_names = [e["gate_name"] for e in gate_history]
        assert "G2_T1" in gate_names
        assert "G2_T2" in gate_names
        assert "G2_T3" in gate_names
        assert "G2" in gate_names

        # Verify all entries have the correct provenance and outcome.
        for entry in gate_history:
            assert entry["threshold_provenance"] == "literature_anchored", (
                f"{entry['gate_name']}: threshold_provenance must be literature_anchored"
            )
            assert entry["outcome"] == "pass", (
                f"{entry['gate_name']}: passing manifest should produce outcome=pass"
            )

        # Verify g2_provenance is embedded in the reason field (N1 schema extension).
        g2_t1 = next(e for e in gate_history if e["gate_name"] == "G2_T1")
        assert g2_t1["reason"] is not None
        assert "g2_provenance=" in g2_t1["reason"]
        assert "optum_initiation_default" in g2_t1["reason"]

    def test_main_audit_state_roundtrip(self, passing_manifest: dict, tmp_path: Path) -> None:
        """MED-10 (pass-2): when --audit-state is provided with pre-existing
        entries, G2 entries are appended to the loaded audit — not to a fresh one.
        The round-trip verifies that prior entries survive the load+append cycle."""
        # Create a pre-existing audit with one synthetic entry.
        prior_audit = RegulatoryEligibilityAudit()
        prior_audit.append_gate_evaluation(
            timestamp="2026-05-01T00:00:00Z",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
            threshold_provenance="literature_anchored",
        )
        audit_state_path = tmp_path / "prior_audit.json"
        audit_state_path.write_text(json.dumps(prior_audit.to_dict(), indent=2), encoding="utf-8")

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(passing_manifest), encoding="utf-8")
        output_path = tmp_path / "shim.json"
        audit_output_path = tmp_path / "updated_audit.json"

        rc = A.main(
            [
                "--manifest",
                str(manifest_path),
                "--tag-ref",
                "refs/tags/x",
                "--tag-sha",
                "abc",
                "--s-prespec-sha",
                "def",
                "--workflow-run-id",
                "1",
                "--output",
                str(output_path),
                "--audit-state",
                str(audit_state_path),
                "--audit-output",
                str(audit_output_path),
            ]
        )
        assert rc == 0

        audit_payload = json.loads(audit_output_path.read_text(encoding="utf-8"))
        audit = RegulatoryEligibilityAudit.from_dict(audit_payload)

        gate_history = audit.gate_history
        # 1 prior + 4 G2 entries = 5 total.
        assert len(gate_history) == 5, (
            f"Expected 5 gate entries (1 prior + 4 G2); got {len(gate_history)}"
        )
        prior_entry = gate_history[0]
        assert prior_entry["gate_name"] == "minimum_auc"
        g2_names = [e["gate_name"] for e in gate_history[1:]]
        assert "G2" in g2_names

    def test_main_returns_nonzero_on_missing_audit_state(
        self, passing_manifest: dict, tmp_path: Path
    ) -> None:
        """MED-10 (pass-2): --audit-state pointing at a nonexistent file must
        produce rc=1 to prevent silent creation of orphaned audit snapshots."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(passing_manifest), encoding="utf-8")
        rc = A.main(
            [
                "--manifest",
                str(manifest_path),
                "--tag-ref",
                "x",
                "--tag-sha",
                "y",
                "--s-prespec-sha",
                "z",
                "--workflow-run-id",
                "1",
                "--output",
                str(tmp_path / "shim.json"),
                "--audit-output",
                str(tmp_path / "audit_state.json"),
                "--audit-state",
                str(tmp_path / "nonexistent_audit.json"),
            ]
        )
        assert rc == 1


class TestN1AuditIntegration:
    """Verify the ``append_to_n1_audit`` function uses the N1 API correctly.

    These tests operate directly on the public ``RegulatoryEligibilityAudit``
    surface — no file I/O — so they are fast and isolated.
    """

    def test_append_to_n1_audit_populates_gate_history(self, passing_manifest: dict) -> None:
        """MED-10 (pass-2) core invariant: after ``append_to_n1_audit`` the
        audit's ``gate_history`` must contain the G2 fields where N1 expects them."""
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="refs/tags/tier1b-b2-experiment-1",
            tag_sha="abc123def456",
            s_prespec_sha="7f616f6f",
            workflow_run_id="99999",
        )
        audit = RegulatoryEligibilityAudit()
        A.append_to_n1_audit(audit, entries)

        gate_history = audit.gate_history
        assert len(gate_history) == 4

        # Verify the G2 combined entry lands correctly.
        g2 = next(e for e in gate_history if e["gate_name"] == "G2")
        assert g2["outcome"] == "pass"
        assert g2["threshold_provenance"] == "literature_anchored"
        # g2_provenance embedded in reason.
        assert g2["reason"] is not None
        assert "g2_provenance=" in g2["reason"]

    def test_append_to_n1_audit_gate_history_is_immutable_after_append(
        self, passing_manifest: dict
    ) -> None:
        """The N1 API returns a tuple snapshot — external mutation must not
        corrupt the audit's internal state."""
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="x",
            tag_sha="y",
            s_prespec_sha="z",
            workflow_run_id="1",
        )
        audit = RegulatoryEligibilityAudit()
        A.append_to_n1_audit(audit, entries)

        # Attempt to mutate the returned snapshot — must not affect audit.
        snapshot = audit.gate_history
        assert isinstance(snapshot, tuple)
        # list() conversion and pop do not propagate back.
        as_list = list(snapshot)
        as_list.clear()
        # Original audit is unaffected.
        assert len(audit.gate_history) == 4

    def test_append_to_n1_audit_fails_on_setitem(self, passing_manifest: dict) -> None:
        """The N1 API's __setitem__ guard must raise RegulatoryAuditMutationError
        — verifies the append-only invariant is enforced end-to-end."""
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            RegulatoryAuditMutationError,
        )

        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="x",
            tag_sha="y",
            s_prespec_sha="z",
            workflow_run_id="1",
        )
        audit = RegulatoryEligibilityAudit()
        A.append_to_n1_audit(audit, entries)

        with pytest.raises(RegulatoryAuditMutationError):
            audit["gate_history"] = []  # type: ignore[index]

    def test_append_to_n1_audit_from_dict_roundtrip_preserves_g2_fields(
        self, passing_manifest: dict
    ) -> None:
        """MED-10 (pass-2) serialisation check: audit.to_dict() → from_dict()
        round-trip must preserve all G2 gate fields so a checkpoint restart
        re-reads the same audit state."""
        entries = A.build_audit_entries(
            manifest=passing_manifest,
            tag_ref="refs/tags/tier1b-b2-experiment-1",
            tag_sha="abc123def456",
            s_prespec_sha="7f616f6f",
            workflow_run_id="12345",
        )
        audit = RegulatoryEligibilityAudit()
        A.append_to_n1_audit(audit, entries)

        # Serialise and deserialise.
        snapshot = audit.to_dict()
        restored = RegulatoryEligibilityAudit.from_dict(snapshot)

        assert len(restored.gate_history) == len(audit.gate_history) == 4
        for orig, restored_entry in zip(audit.gate_history, restored.gate_history, strict=True):
            assert orig["gate_name"] == restored_entry["gate_name"]
            assert orig["outcome"] == restored_entry["outcome"]
            assert orig["threshold_provenance"] == restored_entry["threshold_provenance"]
            assert orig["reason"] == restored_entry["reason"]
