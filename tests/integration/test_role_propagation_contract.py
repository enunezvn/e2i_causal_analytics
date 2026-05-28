"""Phase 1 contract tests for causal-role propagation (Issue #237 reframe).

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §1.9.

Seven cases enforce the producer-side contract that makes Layer-4 LLM
``causal_role`` (today persisted only into ``adaptive_verdicts``) ALSO
flow into a typed ``RoleAttribution`` list, with trust-source labels so
Phase 2 (collider/mediator exclusion policy) can act on verified
attributions only.

Falsifiability anchor: revert the ``role_attributions`` write inside
``finalize_output`` → case 1 trips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.audit_sidecar_reader import SidecarReader
from src.data.feature_contract import FeatureContract, KnowableAt
from src.data.role_attribution import RoleAttribution, derive_role_attributions

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_verdict(
    *,
    feature: str,
    llm_role: str | None,
    evaluator_satisfied: bool | None,
    evaluator_model: str | None = "anthropic/claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    """Build a verdict dict matching the producer at
    ``adaptive_validity_check.py::_ensemble_to_legacy_dict``.

    ``llm_role=None`` means Layer-4 did not fire (LLM verdict absent for
    this feature); the helper omits ``evaluator_model`` text in that case
    to keep the shape honest. Case 7 exercises this path.
    """
    return {
        "feature": feature,
        "layer": "4" if llm_role is not None else "3",
        "severity": "moderate",
        "remediation": "keep_with_caveat",
        "evidence": "test fixture",
        "decided_by": "llm" if llm_role is not None else "adversarial",
        "disagreements": [],
        "kg_signal": "no_signal",
        "llm_role": llm_role,
        "llm_remediation": "keep_with_caveat" if llm_role is not None else None,
        "evaluator_satisfied": evaluator_satisfied,
        "evaluator_rationale_complete": True if evaluator_satisfied else None,
        "evaluator_missed_considerations": None,
        "evaluator_notes": "ok" if evaluator_satisfied else None,
        "evaluator_model": evaluator_model if evaluator_satisfied is not None else None,
    }


def _make_contract(name: str, *, causal_role: str | None = None) -> FeatureContract:
    """Build a minimal FeatureContract for manifest_role_map construction."""
    return FeatureContract(
        name=name,
        knowable_at=KnowableAt(reference="index_date"),
        source="demo",
        derivation_inputs=(),
        aggregation=None,
        window_days=None,
        causal_role=causal_role,
    )


# ---------------------------------------------------------------------------
# Case 1 — LLM-only path (no manifest)
# ---------------------------------------------------------------------------


def test_case_1_llm_collider_satisfied_no_manifest() -> None:
    """LLM verdict ``causal_role="collider"``, ``evaluator_audit.satisfied=True``,
    empty manifest → ``source="llm"``, ``evaluator_satisfied=True``,
    ``causal_role="collider"``."""
    verdicts = [
        _make_verdict(feature="f1", llm_role="collider", evaluator_satisfied=True),
    ]
    attributions = derive_role_attributions(verdicts, {})
    assert len(attributions) == 1
    attr = attributions[0]
    assert attr["feature"] == "f1"
    assert attr["source"] == "llm"
    assert attr["evaluator_satisfied"] is True
    assert attr["causal_role"] == "collider"
    assert attr["evaluator_model"] == "anthropic/claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Case 2 — Manifest takes precedence over conflicting LLM verdict
# ---------------------------------------------------------------------------


def test_case_2_manifest_overrides_conflicting_llm() -> None:
    """Manifest ``f1: causal_role="confounder"``; LLM verdict for f1 says
    ``collider`` → output is ``source="manifest"``, ``evaluator_satisfied=True``,
    ``causal_role="confounder"``."""
    verdicts = [
        _make_verdict(feature="f1", llm_role="collider", evaluator_satisfied=True),
    ]
    contracts = {"f1": _make_contract("f1", causal_role="confounder")}
    attributions = derive_role_attributions(verdicts, contracts)
    assert len(attributions) == 1
    attr = attributions[0]
    assert attr["feature"] == "f1"
    assert attr["source"] == "manifest"
    assert attr["evaluator_satisfied"] is True
    assert attr["causal_role"] == "confounder"
    assert attr["evaluator_model"] == "n/a"


# ---------------------------------------------------------------------------
# Case 3 — LLM satisfied=False is preserved (no manifest override)
# ---------------------------------------------------------------------------


def test_case_3_llm_unsatisfied_no_manifest() -> None:
    """LLM verdict ``satisfied=False``, empty manifest → emitted with
    ``evaluator_satisfied=False``. (Phase 2 policy will then gate it.)"""
    verdicts = [
        _make_verdict(feature="f1", llm_role="mediator", evaluator_satisfied=False),
    ]
    attributions = derive_role_attributions(verdicts, {})
    assert len(attributions) == 1
    attr = attributions[0]
    assert attr["source"] == "llm"
    assert attr["evaluator_satisfied"] is False
    assert attr["causal_role"] == "mediator"


# ---------------------------------------------------------------------------
# Case 4 — Manifest overrides LLM-unsatisfied (codex-2 B2 reframe)
# ---------------------------------------------------------------------------


def test_case_4_manifest_overrides_llm_unsatisfied() -> None:
    """Construct verdict where Layer-4 fired AND ``evaluator_audit.satisfied=False``,
    AND manifest declares ``f2: causal_role="confounder"``.

    The LLM's ``satisfied=False`` is *ignored* because manifest is
    verification-grade per C1 (trust-boundary constraint).
    """
    verdicts = [
        _make_verdict(feature="f2", llm_role="collider", evaluator_satisfied=False),
    ]
    contracts = {"f2": _make_contract("f2", causal_role="confounder")}
    attributions = derive_role_attributions(verdicts, contracts)
    assert len(attributions) == 1
    attr = attributions[0]
    assert attr["source"] == "manifest"
    assert attr["evaluator_satisfied"] is True
    assert attr["causal_role"] == "confounder"


# ---------------------------------------------------------------------------
# Case 5 — Sidecar round-trip with schema_version "1.4"
# ---------------------------------------------------------------------------


def test_case_5_sidecar_round_trip(tmp_path: Path, monkeypatch, caplog) -> None:
    """Write payload via ``write_adaptive_verdicts_sidecar`` → read via
    ``SidecarReader`` → assert each ``VerdictRecord.role_attribution``
    matches the produced attribution. Reader does NOT emit a
    ``_check_schema_version`` WARN (since the "1.x" minor matches MAJOR=1).
    """
    from src.agents.ml_foundation.data_preparer.graph import (
        write_adaptive_verdicts_sidecar,
    )

    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    verdicts = [
        _make_verdict(feature="f1", llm_role="collider", evaluator_satisfied=True),
        _make_verdict(feature="f2", llm_role="mediator", evaluator_satisfied=True),
    ]
    contracts = {"f2": _make_contract("f2", causal_role="confounder")}
    attributions = derive_role_attributions(verdicts, contracts)
    state = {
        "experiment_id": "exp-round-trip",
        "data_source": "synthetic",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": verdicts,
        "role_attributions": attributions,
    }
    path = write_adaptive_verdicts_sidecar(state)
    assert path is not None and path.exists()
    payload = json.loads(Path(path).read_text())
    # Schema bump assertion (current producer minor; 1.7 since Layer-4 Phase 2
    # added the additive per-verdict ``structural_unclassifiable`` key — still
    # MAJOR=1; Phase 1 bumped 1.5 → 1.6 for leakage_fdr, #501 1.4 → 1.5 for the
    # M-structure shadow keys, #508 1.3 → 1.4 for the leak-crosscheck key, #240
    # Stage 3 1.2 → 1.3 for its soft-gate keys).
    assert payload["schema_version"] == "1.7"
    assert "role_attributions" in payload
    assert len(payload["role_attributions"]) == 2

    # Reader produces VerdictRecords with role_attribution attached.
    with caplog.at_level("WARNING"):
        reader = SidecarReader(artifacts_dir=tmp_path)
        records = list(reader.iter_verdict_records())

    # No schema-version WARN on "1.7" (exact match with the reader; MAJOR=1).
    schema_warns = [
        r for r in caplog.records if "schema_version" in r.message and r.levelname == "WARNING"
    ]
    assert schema_warns == [], (
        f"reader emitted unexpected schema_version warns for 1.7: {[w.message for w in schema_warns]}"
    )

    by_feature = {r.feature: r for r in records}
    # f1: LLM source.
    assert by_feature["f1"].role_attribution is not None
    assert by_feature["f1"].role_attribution["source"] == "llm"
    assert by_feature["f1"].role_attribution["causal_role"] == "collider"
    # f2: manifest source (overrode LLM's "mediator" -> "confounder").
    assert by_feature["f2"].role_attribution is not None
    assert by_feature["f2"].role_attribution["source"] == "manifest"
    assert by_feature["f2"].role_attribution["causal_role"] == "confounder"


# ---------------------------------------------------------------------------
# Case 6 — Empty manifest does not raise; all attributions are llm
# ---------------------------------------------------------------------------


def test_case_6_empty_manifest() -> None:
    """``derive_role_attributions(verdicts, {})`` does not raise; all
    attributions are ``source="llm"``."""
    verdicts = [
        _make_verdict(feature="f1", llm_role="collider", evaluator_satisfied=True),
        _make_verdict(feature="f2", llm_role="confounder", evaluator_satisfied=False),
    ]
    attributions = derive_role_attributions(verdicts, {})
    assert len(attributions) == 2
    assert all(a["source"] == "llm" for a in attributions)


# ---------------------------------------------------------------------------
# Case 7 — Manifest branch works when Layer-4 did NOT fire (common path)
# ---------------------------------------------------------------------------


def test_case_7_manifest_only_no_llm_verdict() -> None:
    """Synthetic verdict where Layer-4 did NOT fire (no LLM ``causal_role``
    in verdict dict). Manifest declares ``f3: causal_role="collider"``.

    Producer still emits attribution ``source="manifest", causal_role="collider"``
    — verifying the manifest-first branch works even when LLM verdict is
    absent (the common path per ``adaptive_validity_check.py:2944-2952``
    Layer-4 trigger conditions)."""
    verdicts = [
        _make_verdict(feature="f3", llm_role=None, evaluator_satisfied=None),
    ]
    contracts = {"f3": _make_contract("f3", causal_role="collider")}
    attributions = derive_role_attributions(verdicts, contracts)
    assert len(attributions) == 1
    attr = attributions[0]
    assert attr["feature"] == "f3"
    assert attr["source"] == "manifest"
    assert attr["evaluator_satisfied"] is True
    assert attr["causal_role"] == "collider"


# ---------------------------------------------------------------------------
# Additional: TypedDict shape pins (forward-compat for Phase 2/6)
# ---------------------------------------------------------------------------


def test_role_attribution_has_required_keys() -> None:
    """RoleAttribution declared keys: feature, causal_role, source,
    evaluator_satisfied, evaluator_model. Phase 2's
    ``_should_act(attr)`` predicate keys on ``source`` + ``evaluator_satisfied``;
    pin these so a future field rename trips the test."""
    required = {"feature", "causal_role", "source", "evaluator_satisfied", "evaluator_model"}
    declared = set(RoleAttribution.__annotations__.keys())
    assert required.issubset(declared), (
        f"RoleAttribution missing required keys: {required - declared}"
    )
