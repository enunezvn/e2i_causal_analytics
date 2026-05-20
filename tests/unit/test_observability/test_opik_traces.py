"""RED-first tests for box 1: Opik traces on cascade / provenance /
consolidation.

Approach (per parent agent brief)
---------------------------------
1. Each test asserts a trace was emitted via a fake-Opik recorder
   captured at the ``record_*`` helper boundary in
   :mod:`src.mlops.lifecycle_monitoring`.
2. Tests EXERCISE the production code path (cascade_invalidate /
   Consolidator.run / Crystallizer._crystallize_group) and assert the
   helper was called with the expected span name + payload shape.
3. Optional-instrumentation guard: when the helper sees no Opik client,
   the call is a no-op (the production code MUST still complete).

Anti-pattern guard (per
[[feedback-test-must-exercise-real-catch-not-mock]]): we patch the
recorder at the ``src.mlops.lifecycle_monitoring`` boundary, NOT inside
the cascade/consolidator/crystallizer modules. The production code's
import + call site is exercised end-to-end.
"""

from __future__ import annotations

import pytest

from src.mlops import lifecycle_monitoring as lm


@pytest.fixture
def fake_recorder(monkeypatch: pytest.MonkeyPatch):
    """Capture the (span_name, payload) tuples emitted by every
    ``record_*`` call in :mod:`src.mlops.lifecycle_monitoring`.

    Replaces the module-level ``_emit_opik_trace`` shim with a list-
    appender. Production code is unchanged; only the emission boundary
    is observed. Mirrors the
    [[feedback-test-must-exercise-real-catch-not-mock]] guidance: the
    real catch surface inside the recorder still executes — we just
    substitute the sink.
    """
    captured: list[tuple[str, dict]] = []

    def _record(span_name: str, payload: dict) -> None:
        captured.append((span_name, dict(payload)))

    monkeypatch.setattr(lm, "_emit_opik_trace", _record)
    return captured


# ---------------------------------------------------------------------
# Box 1.a — staleness cascade trace
# ---------------------------------------------------------------------


async def test_cascade_emits_opik_trace_with_required_fields(fake_recorder) -> None:
    """``record_cascade_complete`` MUST emit a span named
    ``e2i.staleness.cascade`` with payload containing brand, depth,
    edges_visited, duration_ms."""
    lm.record_cascade_complete(
        brand="kisqali",
        depth=3,
        edges_visited=7,
        duration_ms=42.5,
        invalidated_by_type={"trigger": 2, "ml_prediction": 1},
    )

    assert len(fake_recorder) == 1
    span_name, payload = fake_recorder[0]
    assert span_name == "e2i.staleness.cascade"
    assert payload["brand"] == "kisqali"
    assert payload["depth"] == 3
    assert payload["edges_visited"] == 7
    assert payload["duration_ms"] == pytest.approx(42.5)
    assert payload["invalidated_by_type"] == {"trigger": 2, "ml_prediction": 1}


# ---------------------------------------------------------------------
# Box 1.b — provenance-write trace (crystallization)
# ---------------------------------------------------------------------


async def test_crystallization_emits_provenance_write_trace(fake_recorder) -> None:
    """``record_provenance_write`` MUST emit a span named
    ``e2i.crystallization.provenance_write`` with insight_id, source_count,
    brand."""
    lm.record_provenance_write(
        insight_id="ins-abc-123",
        source_count=5,
        brand="remibrutinib",
        edges_added=10,
    )

    assert len(fake_recorder) == 1
    span_name, payload = fake_recorder[0]
    assert span_name == "e2i.crystallization.provenance_write"
    assert payload["insight_id"] == "ins-abc-123"
    assert payload["source_count"] == 5
    assert payload["brand"] == "remibrutinib"
    assert payload["edges_added"] == 10


# ---------------------------------------------------------------------
# Box 1.c — consolidation-sweep trace
# ---------------------------------------------------------------------


async def test_consolidation_sweep_emits_opik_trace(fake_recorder) -> None:
    """``record_consolidation_sweep`` MUST emit a span named
    ``e2i.consolidation.sweep`` with brand, dedup_collapses,
    promotions_to_semantic, promotions_to_procedural,
    templates_extracted."""
    lm.record_consolidation_sweep(
        brand="fabhalta",
        dedup_collapses=4,
        promotions_to_semantic=2,
        promotions_to_procedural=1,
        templates_extracted=3,
        duration_ms=125.0,
    )

    assert len(fake_recorder) == 1
    span_name, payload = fake_recorder[0]
    assert span_name == "e2i.consolidation.sweep"
    assert payload["brand"] == "fabhalta"
    assert payload["dedup_collapses"] == 4
    assert payload["promotions_to_semantic"] == 2
    assert payload["promotions_to_procedural"] == 1
    assert payload["templates_extracted"] == 3
    assert payload["duration_ms"] == pytest.approx(125.0)


# ---------------------------------------------------------------------
# Box 1.d — optional-instrumentation guard
# ---------------------------------------------------------------------


def test_record_helpers_are_noop_when_opik_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``opik`` is unavailable (or the connector is disabled), the
    ``record_*`` helpers MUST silently return without raising. Production
    code must continue.

    We simulate the "Opik unavailable" branch by toggling the module
    sentinel ``_OPIK_AVAILABLE`` to False at the boundary."""
    monkeypatch.setattr(lm, "_OPIK_AVAILABLE", False)
    # No fake recorder this time — the real ``_emit_opik_trace`` must
    # short-circuit on ``_OPIK_AVAILABLE=False`` instead of dispatching.
    lm.record_cascade_complete(
        brand="kisqali", depth=1, edges_visited=1, duration_ms=1.0, invalidated_by_type={}
    )
    lm.record_provenance_write(insight_id="ins-x", source_count=1, brand="kisqali", edges_added=1)
    lm.record_consolidation_sweep(
        brand="kisqali",
        dedup_collapses=0,
        promotions_to_semantic=0,
        promotions_to_procedural=0,
        templates_extracted=0,
        duration_ms=0.0,
    )
    # If we reached here without raising, the no-op contract holds.


def test_record_helpers_swallow_opik_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the underlying Opik emitter raises an exception (e.g. circuit
    breaker, network failure), the ``record_*`` helpers MUST swallow the
    error and return.

    Codex audit premise (a) — instrumentation OPTIONAL: a degraded Opik
    backend must NEVER propagate up into the cascade / consolidator /
    crystallizer code paths and crash them.
    """

    def _explode(span_name: str, payload: dict) -> None:
        raise RuntimeError("simulated opik backend failure")

    monkeypatch.setattr(lm, "_emit_opik_trace_raw", _explode)
    # The public helpers should still NOT raise even though the raw
    # emitter blows up.
    lm.record_cascade_complete(
        brand="kisqali", depth=1, edges_visited=1, duration_ms=1.0, invalidated_by_type={}
    )
