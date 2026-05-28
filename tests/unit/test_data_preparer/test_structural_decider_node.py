"""Node wiring for the structural decider (Plan v4 Layer B / Phase 2, Task 3).

Exercises the REAL ``adaptive_validity_check`` node path: an attested feature's
deterministic structural role is derived at the top of the Layer-3 loop and
threaded through ``_compose_legacy_verdict`` → ``EnsembleVoter`` so the verdict
is tagged ``decided_by="structural"``, the LLM is skipped, an empirical
adversarial-high veto still wins, a malformed attestation routes to review
without crashing, and a too-few-rows short-circuit STILL gets its (data-
independent) structural decision.

The contract is resolved via the node's real ``lookup_feature_contract`` seam
(monkeypatched on the node module), not a fabricated ``scope_spec["contracts"]``.
"""

import asyncio
import sys

import numpy as np
import pandas as pd

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    adaptive_validity_check,
)
from src.data.feature_contract import CausalStructureAttestation, FeatureContract, KnowableAt

# ``nodes/__init__.py`` re-exports the ``adaptive_validity_check`` FUNCTION, which
# shadows the submodule of the same name — so ``import ...adaptive_validity_check as
# node_mod`` binds the function, not the module. Resolve the real module (whose
# globals the function's ``lookup_feature_contract`` call binds against) so the
# monkeypatch below patches the name the node actually reads.
node_mod = sys.modules[adaptive_validity_check.__module__]


def _contract(edges, *, name="v", role=None):
    return FeatureContract(
        name=name,
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        causal_role=role,
        causal_structure=CausalStructureAttestation(
            treatment_node="T", outcome_node="y", feature_node=name, edges=edges
        ),
    )


def _state(target="y"):
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"v": rng.normal(size=n), target: rng.integers(0, 2, size=n)})
    return {
        "experiment_id": "struct-decider-test",
        "train_df": df,
        "scope_spec": {
            "prediction_target": target,
            "required_features": ["v"],
            "feature_manifest_source": "synthetic",
        },
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_layer4_enabled": False,
        "adaptive_structural_decider_enabled": True,
    }


def _verdict_for(state, feat="v"):
    out = asyncio.run(adaptive_validity_check(state))
    return next(v for v in out["adaptive_verdicts"] if v["feature"] == feat)


def test_node_structural_decides_collider_and_skips_llm(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "lookup_feature_contract",
        lambda feat, data_source=None: _contract((("T", "v"), ("y", "v"))),  # collider
    )
    v = _verdict_for(_state())
    assert v["decided_by"] == "structural"
    assert v["structural_role"] == "collider"
    assert v["severity"] == "high" and v["remediation"] == "drop"
    assert v.get("llm_role") is None  # LLM never ran


def test_node_unclassifiable_routes_to_review_not_crash(monkeypatch):
    # Edges omit the declared T/y nodes → extract_role cannot classify → the
    # voter routes to moderate/review under decided_by="structural".
    monkeypatch.setattr(
        node_mod,
        "lookup_feature_contract",
        lambda feat, data_source=None: _contract((("v", "unrelated"),)),
    )
    v = _verdict_for(_state())
    assert v.get("structural_unclassifiable") is True
    assert v["remediation"] == "review" and v["decided_by"] == "structural"


def test_fdr_high_beats_benign_attestation(monkeypatch):
    # Empirical leak (v == y) attested benign 'confounder' → the empirical
    # adversarial-high veto MUST win over the structural rule.
    monkeypatch.setattr(
        node_mod,
        "lookup_feature_contract",
        lambda feat, data_source=None: _contract((("v", "T"), ("v", "y"))),  # confounder
    )
    n = 400
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    state = _state()
    state["train_df"] = pd.DataFrame({"v": y.copy(), "y": y})  # perfect leak
    v = _verdict_for(state)
    assert v["severity"] == "high"
    assert v["decided_by"] in {"adversarial", "adversarial_ablation"}


def test_short_circuit_with_attestation_still_decides_structural(monkeypatch):
    """A too-few-rows attested feature must STILL get its structural decision —
    the role is data-independent. Without the bypass-gate extension it would hit
    ``_legacy_short_circuit_verdict`` and skip the voter entirely.
    """
    monkeypatch.setattr(
        node_mod,
        "lookup_feature_contract",
        lambda feat, data_source=None: _contract((("T", "v"), ("y", "v"))),  # collider
    )
    state = _state()
    # < MIN_LAYER3_SAMPLES (30) rows → too-few-rows short-circuit at the setter.
    state["train_df"] = pd.DataFrame({"v": [0.1, 0.2, 0.3, 0.4, 0.5], "y": [0, 1, 0, 1, 0]})
    v = _verdict_for(state)
    assert v["decided_by"] == "structural"
    assert v["structural_role"] == "collider"
    assert v["severity"] == "high" and v["remediation"] == "drop"
