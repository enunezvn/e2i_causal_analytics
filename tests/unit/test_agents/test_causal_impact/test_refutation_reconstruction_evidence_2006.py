"""#2006: every refutation evidence row's ``details`` (persisted as
``details_json``) carries the reconstruction evidence -- ``reported_ate``,
``reconstructed_ate``, ``reconstruction_shift`` (reconstructed - reported,
signed) and ``reconstruction_method`` -- so the decomposition
(reconstructed - reported) vs (refit - reconstructed) is readable per row.

Owner decision 2026-09-11 "F1 SHIP". Disproof 2026-09-11: the reconstruction
is faithful (shift +0.0004 on the live pair); the +0.02 seen live is
nuisance-fit instability in the refits (#2031). This is EVIDENCE, not a gate:
no verdict changes, and when the rebuilt estimate exposes no usable ``.value``
``reconstructed_ate`` and ``reconstruction_shift`` are ``None`` -- never a
fabricated number -- while ``reported_ate`` (known from the estimation node)
and ``reconstruction_method`` are still stamped.

The only patched seams are the ones the existing node tests already patch
(``_reconstruct_dowhy_artifacts`` / ``_fit_negative_control`` at module level
and ``node.runner.run_all_tests``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.causal_impact.nodes import refutation as _ref_mod
from src.agents.causal_impact.nodes.refutation import (
    RefutationNode,
    _resolve_dowhy_method,
)
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from tests.unit.test_agents.test_causal_impact.test_refutation_negative_control_2007 import (
    NC,
    _frames,
    _node_state,
)

RECONSTRUCTION_EVIDENCE_KEYS = (
    "reported_ate",
    "reconstructed_ate",
    "reconstruction_shift",
    "reconstruction_method",
)
DISCLOSURE_KEYS_1419 = (
    "refutation_subsampled",
    "refutation_n_rows",
    "refutation_n_rows_total",
)


@pytest.fixture(scope="module")
def frames():
    return _frames()


def _two_test_suite() -> RefutationSuite:
    """At least two suite rows so 'every row' is more than a single stamp."""
    return RefutationSuite(
        passed=True,
        confidence_score=1.0,
        tests=[
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.036,
                refuted_effect=0.001,
                p_value=0.9,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.PASSED,
                original_effect=0.036,
                refuted_effect=0.0362,
                p_value=0.9,
            ),
        ],
        gate_decision=GateDecision.PROCEED,
    )


def _wire(node: RefutationNode, monkeypatch, estimate_stub, nc_return):
    def fake_recon(**kwargs):
        return (SimpleNamespace(), object(), estimate_stub)

    async def fake_fit(**kwargs):
        return nc_return

    def fake_run_all_tests(**kwargs):
        return _two_test_suite()

    async def _no_signal(outcome):
        return None

    monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
    monkeypatch.setattr(_ref_mod, "_fit_negative_control", fake_fit)
    monkeypatch.setattr(node.runner, "run_all_tests", fake_run_all_tests)
    monkeypatch.setattr(node, "_log_validation_outcome_signal", _no_signal)


def _rows(result: dict) -> list[dict]:
    rows = result["refutation_suite"]["tests"]
    names = [t["test_name"] for t in rows]
    assert RefutationTestType.NEGATIVE_CONTROL_OUTCOME.value in names, names
    assert len(rows) >= 3, names
    return rows


@pytest.mark.asyncio
async def test_every_evidence_row_carries_the_reconstruction_evidence(frames, monkeypatch):
    frame, nc_df = frames
    node = RefutationNode()
    _wire(
        node,
        monkeypatch,
        estimate_stub=SimpleNamespace(value=0.0554),
        nc_return=((NC, 0.01, (-0.02, 0.04), 290), None),
    )
    state = _node_state(
        frame,
        ate=0.036,
        negative_control_outcome=NC,
        data_cache={"negative_control_data": nc_df},
    )
    expected_method = _resolve_dowhy_method(state["estimation_result"])

    result = await node.execute(state)

    assert result["gate_decision"] == "proceed"  # evidence only, no gate change
    for row in _rows(result):
        details = row["details"]
        assert details["reported_ate"] == pytest.approx(0.036), row["test_name"]
        assert details["reconstructed_ate"] == pytest.approx(0.0554), row["test_name"]
        assert details["reconstruction_shift"] == pytest.approx(0.0194), row["test_name"]
        assert details["reconstruction_method"] == expected_method, row["test_name"]
        # The #1419 subsample disclosure is untouched.
        for key in DISCLOSURE_KEYS_1419:
            assert key in details, (row["test_name"], key)
        assert details["refutation_subsampled"] is False
        assert details["refutation_n_rows"] == len(frame)


@pytest.mark.asyncio
async def test_unavailable_reconstructed_value_is_none_not_fabricated(frames, monkeypatch):
    frame, nc_df = frames
    node = RefutationNode()
    _wire(
        node,
        monkeypatch,
        estimate_stub=object(),  # no ``.value`` (stubs, exotic estimators)
        nc_return=(None, "negative_control_too_few_rows"),
    )
    state = _node_state(
        frame,
        ate=0.036,
        negative_control_outcome=NC,
        data_cache={"negative_control_data": nc_df},
    )
    expected_method = _resolve_dowhy_method(state["estimation_result"])

    result = await node.execute(state)

    for row in _rows(result):
        details = row["details"]
        for key in RECONSTRUCTION_EVIDENCE_KEYS:
            assert key in details, (row["test_name"], key)
        # The reported ATE is known from the estimation node regardless of the
        # reconstruction; only the reconstructed value and the shift are unknown.
        assert details["reported_ate"] == pytest.approx(0.036)
        assert details["reconstructed_ate"] is None
        assert details["reconstruction_shift"] is None
        assert isinstance(details["reconstruction_method"], str)
        assert details["reconstruction_method"] == expected_method
        assert details["reconstruction_method"]
