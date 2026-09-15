"""Regression tests for #2123: the causal_impact ``interpretation`` audit-chain
entry was never written.

The ``traced_node`` wrapper (``src/agents/causal_impact/graph.py``,
interpretation branch) forwarded ``interp["causal_confidence"]`` — a
categorical LABEL (``"high"`` / ``"medium"`` / ``"low"`` / ``"N/A"``) — as
``AuditChainService.add_entry(confidence_score=...)``, which is declared
``Optional[float]`` and lands in ``audit_chain_entries.confidence_score
numeric(5,4)``. Postgres refused every insert that carried a label
(``22P02 invalid input syntax for type numeric: "high"``); the wrapper's
``except Exception`` turned that into ``WARNING Failed to record audit entry``
and continued. Measured 2026-09-15: 0 of 177 causal_impact workflows (7 d)
carried an ``interpretation`` entry while every other node had one.

The node already owns the label→number mapping
(``InterpretationNode._confidence_to_score``: low 0.33 / medium 0.66 /
high 1.0) for the DSPy signal. The fix promotes that mapping to a module-level
``confidence_label_to_score`` and has the wrapper use it — same numbers, no
guessing: an unknown label (``"N/A"``) becomes NULL in the audit row, while
the DSPy signal keeps its historical ``0.5`` default for every label the node
writes. The one difference is unreachable from the node: a non-string label
used to raise ``AttributeError`` inside the signal's swallowed ``try`` and now
yields ``0.5``.

Same call-shape style as ``test_audit_chain_kwargs.py`` (#355).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.agents.base.audit_chain_mixin import set_audit_chain_service
from src.agents.causal_impact.graph import _extract_mlflow_metrics, traced_node
from src.agents.causal_impact.nodes.interpretation import InterpretationNode
from src.utils.audit_chain import AuditChainService


@pytest.fixture(autouse=True)
def _reset_audit_service():
    set_audit_chain_service(None)
    yield
    set_audit_chain_service(None)


@pytest.fixture
def mock_audit_service():
    svc = MagicMock(spec=AuditChainService)
    set_audit_chain_service(svc)
    return svc


@pytest.fixture
def strict_audit_service():
    """A double that enforces the ``numeric(5,4)`` column contract at the
    ``add_entry`` boundary: ``confidence_score`` must be ``None`` or a float in
    [0, 1]. It rejects a superset of what the column rejects — anything but
    ``None`` or a float in [0, 1] (an int or bool that ``numeric(5,4)`` would
    accept is refused here too) — so the categorical label pre-fix raises just
    as the real insert did; that is what the wrapper's ``except Exception``
    swallowed into a WARNING."""
    svc = MagicMock(spec=AuditChainService)

    def _add_entry(**kwargs):
        cs = kwargs.get("confidence_score")
        if cs is None:
            return MagicMock()
        if isinstance(cs, bool) or not isinstance(cs, float) or not (0.0 <= cs <= 1.0):
            raise TypeError(
                f"invalid input syntax for type numeric: {cs!r} (confidence_score must be "
                "None or a float in [0, 1])"
            )
        return MagicMock()

    svc.add_entry.side_effect = _add_entry
    set_audit_chain_service(svc)
    return svc


@pytest.fixture
def mock_opik():
    """Patch get_opik_connector to a no-op async context manager."""
    mock = MagicMock()
    mock.is_enabled = True
    span = MagicMock()
    span.span_id = "span_test"
    span.set_output = MagicMock()
    span.set_attribute = MagicMock()
    mock.trace_agent = MagicMock()
    mock.trace_agent.return_value.__aenter__ = AsyncMock(return_value=span)
    mock.trace_agent.return_value.__aexit__ = AsyncMock(return_value=None)
    with patch("src.agents.causal_impact.graph.get_opik_connector", return_value=mock):
        yield mock


@pytest.fixture
def base_state():
    return {
        "audit_workflow_id": uuid4(),
        "query": "Did dispatch X cause uplift?",
        "treatment_var": "spend",
        "outcome_var": "uplift",
        "current_phase": "interpretation",
        "session_id": "sess-1",
        "user_id": "user-1",
        "brand": "BrandX",
        "query_id": "qid-1",
        "span_id": None,
        "dispatch_id": "dispatch-1",
    }


def _interpretation_node(causal_confidence: str):
    """A fake interpretation node result carrying the given confidence LABEL —
    the exact shape ``InterpretationNode.execute`` returns (``interpretation``
    sub-dict with ``causal_confidence`` and ``depth_level``)."""

    @traced_node("interpretation")
    async def _node(state):
        return {
            "status": "completed",
            "current_phase": "completed",
            "interpretation": {
                "narrative": "x",
                "causal_confidence": causal_confidence,
                "depth_level": "standard",
            },
        }

    return _node


# ---------------------------------------------------------------------------
# T1 — a real label maps to the node's existing number; the LABEL stays in the
# hashed output payload.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpretation_high_label_reaches_add_entry_as_float_1_0(
    mock_audit_service, mock_opik, base_state
):
    await _interpretation_node("high")(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "interpretation"
    cs = kwargs["confidence_score"]
    assert isinstance(cs, float) and cs == 1.0, (
        "confidence_score must be the node's numeric mapping of the label "
        f"(high → 1.0, a float); got {cs!r} — the categorical label is refused "
        "by audit_chain_entries.confidence_score numeric(5,4) (#2123)"
    )
    # The output payload (hashed into the chain) keeps the label unchanged.
    assert kwargs["output_data"]["causal_confidence"] == "high"
    assert kwargs["output_data"]["depth_level"] == "standard"


# ---------------------------------------------------------------------------
# T2 — an unknown / not-applicable label is NULL in the audit row, never a
# guessed number.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpretation_na_label_reaches_add_entry_as_none(
    mock_audit_service, mock_opik, base_state
):
    await _interpretation_node("N/A")(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["confidence_score"] is None, (
        "'N/A' has no numeric meaning — the audit row must carry NULL, not the "
        f"label and not a default; got {kwargs['confidence_score']!r}"
    )
    assert kwargs["output_data"]["causal_confidence"] == "N/A"


# ---------------------------------------------------------------------------
# T3 (teeth) — with a double that enforces the column type, pre-fix the
# wrapper swallowed the TypeError into ``Failed to record audit entry`` and
# recorded nothing; post-fix the entry is recorded and nothing is warned.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpretation_entry_is_recorded_without_swallowed_warning(
    strict_audit_service, mock_opik, base_state, caplog
):
    with caplog.at_level(logging.WARNING, logger="src.agents.causal_impact.graph"):
        result = await _interpretation_node("medium")(base_state)

    # The node result itself is untouched either way (the wrapper never masks it).
    assert result["interpretation"]["causal_confidence"] == "medium"

    failure_warnings = [
        r
        for r in caplog.records
        if r.name == "src.agents.causal_impact.graph"
        and r.levelno == logging.WARNING
        and "Failed to record audit entry" in r.getMessage()
    ]
    assert not failure_warnings, (
        "add_entry refused the interpretation entry and the wrapper swallowed it "
        "into a WARNING — the interpretation row is never written (#2123). "
        f"Warnings: {[r.getMessage() for r in failure_warnings]}"
    )
    assert strict_audit_service.add_entry.call_count == 1
    assert strict_audit_service.add_entry.call_args.kwargs["confidence_score"] == 0.66


# ---------------------------------------------------------------------------
# T4 — the promoted mapping: same numbers as the node, case-insensitive,
# unknown → ``default`` (None for the audit chain, 0.5 for the DSPy signal).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("low", 0.33),
        ("Medium", 0.66),
        ("HIGH", 1.0),
        ("N/A", None),
        (None, None),
        ("bogus", None),
    ],
)
def test_confidence_label_to_score_table(label, expected):
    # Imported here on purpose: on base (502f95028) the symbol does not exist, and a
    # module-level import would fail collection, hiding T1-T3's own red reasons.
    from src.agents.causal_impact.nodes.interpretation import confidence_label_to_score

    assert confidence_label_to_score(label) == expected


def test_confidence_label_to_score_default_is_used_for_unknown_only():
    # Lazy import for the same red-first reason as the table test above.
    from src.agents.causal_impact.nodes.interpretation import confidence_label_to_score

    assert confidence_label_to_score("bogus", default=0.5) == 0.5
    assert confidence_label_to_score(None, default=0.5) == 0.5
    assert confidence_label_to_score("N/A", default=0.5) == 0.5
    # A real label ignores the default.
    assert confidence_label_to_score("low", default=0.5) == 0.33


@pytest.mark.parametrize(
    ("label", "expected"),
    [("high", 1.0), ("Medium", 0.66), ("low", 0.33)],
)
def test_mlflow_metrics_block_uses_the_same_mapping(label, expected):
    """``_extract_mlflow_metrics`` carried its own copy of the three numbers;
    it now calls the shared mapping. Known labels → the same values as before,
    an unknown label leaves the metric unset (never a default), and an absent
    label leaves it unset too."""
    metrics = _extract_mlflow_metrics(
        {"interpretation": {"causal_confidence": label}}, total_latency_ms=1.0
    )
    assert metrics["causal_confidence"] == expected


@pytest.mark.parametrize("label", ["N/A", "bogus"])
def test_mlflow_metrics_block_leaves_unknown_label_unset(label):
    metrics = _extract_mlflow_metrics(
        {"interpretation": {"causal_confidence": label}}, total_latency_ms=1.0
    )
    assert "causal_confidence" not in metrics
    assert "causal_confidence" not in _extract_mlflow_metrics({}, total_latency_ms=1.0)


def test_dspy_signal_mapping_is_byte_identical():
    """``InterpretationNode._confidence_to_score`` (the DSPy signal path) keeps
    its historical behaviour: unknown → 0.5, known → the same three numbers."""
    node = InterpretationNode()
    assert node._confidence_to_score("bogus") == 0.5
    assert node._confidence_to_score("low") == 0.33
    assert node._confidence_to_score("medium") == 0.66
    assert node._confidence_to_score("HIGH") == 1.0
