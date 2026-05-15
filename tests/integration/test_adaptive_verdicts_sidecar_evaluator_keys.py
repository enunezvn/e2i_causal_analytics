"""Verify the 5 new evaluator audit keys reach the sidecar JSON.

Plan: .claude/plans/layer4_evaluator_audit_signal.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.ml_foundation.data_preparer.graph import (
    write_adaptive_verdicts_sidecar,
)


def _make_verdict_with_evaluator_keys(*, satisfied: bool | None) -> dict:
    """Build a verdict dict shaped exactly as `_ensemble_to_legacy_dict`
    emits post-Task-5: includes the 5 evaluator keys with the same Python
    types it produces. Critically, `evaluator_missed_considerations` is
    a tuple here (matching `LLMEvaluatorAudit.missed_considerations:
    tuple[str, ...]`) — `json.dumps` will serialize it as a JSON array,
    deserialized as a Python list. The test asserts BOTH the serialized
    list shape AND the round-trip semantics."""
    return {
        "feature": "f",
        "layer": "4",
        "severity": "moderate",
        "remediation": "keep_with_caveat",
        "evidence": "layer-4 llm",
        "decided_by": "llm",
        "disagreements": [],
        "kg_signal": "no_signal",
        "z_score": 4.2,
        "actual_auc": 0.66,
        "null_mean": 0.50,
        "null_std": 0.02,
        "p_value": 0.0001,
        "n_permutations": 200,
        "delta_auc": 0.12,
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": False,
        "severity_pre_joint_check": "moderate",
        "ablation_z_score": None,
        "ablation_delta_auc": None,
        "ablation_null_mean": None,
        "ablation_null_std": None,
        "ablation_severity": None,
        "contract_source": None,
        "contract_window_days": None,
        "llm_role": "confounder",
        "llm_remediation": "keep_with_caveat",
        "evaluator_satisfied": satisfied,
        "evaluator_rationale_complete": True if satisfied else None,
        # Tuple here matches the producer at `_ensemble_to_legacy_dict`.
        "evaluator_missed_considerations": (("pearl_arrows",) if satisfied else None),
        "evaluator_notes": "ok" if satisfied else None,
        "evaluator_model": ("anthropic/claude-haiku-4-5-20251001" if satisfied else None),
    }


def test_sidecar_serialises_evaluator_keys_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    state = {
        "experiment_id": "test-experiment",
        "data_source": "synthetic",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": [_make_verdict_with_evaluator_keys(satisfied=True)],
    }
    path = write_adaptive_verdicts_sidecar(state)
    assert path is not None and path.exists()
    payload = json.loads(Path(path).read_text())
    verdict = payload["adaptive_verdicts"][0]
    for key in (
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
    ):
        assert key in verdict, f"sidecar JSON missing {key}"
    assert verdict["evaluator_satisfied"] is True
    assert verdict["evaluator_model"] == "anthropic/claude-haiku-4-5-20251001"
    # tuple -> JSON array -> Python list round-trip (codex Gate-2 MED-2).
    assert verdict["evaluator_missed_considerations"] == ["pearl_arrows"]
    assert isinstance(verdict["evaluator_missed_considerations"], list)


def test_sidecar_emits_none_for_evaluator_keys_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    state = {
        "experiment_id": "test-experiment",
        "data_source": "synthetic",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": [
            _make_verdict_with_evaluator_keys(satisfied=None),
        ],
    }
    path = write_adaptive_verdicts_sidecar(state)
    payload = json.loads(Path(path).read_text())
    verdict = payload["adaptive_verdicts"][0]
    for key in (
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
    ):
        assert verdict[key] is None


def test_sidecar_info_log_includes_verdict_count(tmp_path, monkeypatch, caplog):
    """Plan layer4_evaluator_audit_consumer.md Task 3: operators need
    verdict-count visibility in the INFO log, not just the path."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    state = {
        "experiment_id": "test-experiment",
        "data_source": "synthetic",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": ["f1", "f2"],
        "adaptive_verdicts": [
            _make_verdict_with_evaluator_keys(satisfied=True),
            _make_verdict_with_evaluator_keys(satisfied=False),
        ],
    }
    with caplog.at_level("INFO"):
        path = write_adaptive_verdicts_sidecar(state)
    assert path is not None
    matching = [
        r for r in caplog.records
        if "Wrote adaptive-validity audit trail" in r.message
    ]
    assert len(matching) == 1, (
        f"expected exactly one matching INFO line, got: "
        f"{[r.message for r in matching]}"
    )
    assert str(path) in matching[0].message
    assert "verdicts=2" in matching[0].message
