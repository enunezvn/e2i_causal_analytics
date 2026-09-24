"""Graph-level regression for #2283: a FAILED Pandera schema validation must
reach the QC gate.

``blocking_issues`` (``state.py:258``) is a plain ``Optional[List[str]]`` with
no reducer, so LangGraph gives it ``LastValue`` semantics: last writer wins.
``run_schema_validation`` appends its blocking entry; two downstream nodes then
destroy it before ``finalize_output`` reads it —

* ``quality_checker.py`` starts from a fresh local ``[]`` and returns it, and
* ``ge_validator.py`` returns ``None`` on the happy path (no GE blockers).

Both writes land: a dict returned by a node becomes a ``ChannelWriteTupleEntry``
(``langgraph/graph/state.py``), which has no ``skip_none`` filter, so ``[]`` and
``None`` overwrite just like any other value.

Why these tests must drive a COMPILED graph and not the node functions: the
failure is a *channel* effect. ``run_schema_validation`` returns the correct
dict — asserting on that return (as ``test_schema_validator.py`` does) can never
see a downstream node overwrite the channel. Only an invoked graph can.

The graph below wires the REAL production chain from ``graph.py:320-324``
(``load_data -> audit_sampling_frame -> run_schema_validation ->
run_quality_checks -> run_ge_validation``) onto the REAL ``finalize_output``
gate, over the REAL ``DataPreparerState`` schema. The nodes between
``run_ge_validation`` and ``finalize_output`` in production (feature
engineering, leakage, transform, Feast, baseline, sufficiency, KG enrichment)
are omitted because several need external services. Several of them ARE
producers of ``blocking_issues`` — ``detect_leakage``, ``transform_data``,
``register_features_in_feast`` and ``sufficiency_check`` all write the channel
— so this file does NOT claim to cover the whole channel. What it covers is
the seam #2283 is about: the two nodes that DESTROYED other producers' entries
between the schema validator and the gate. ``leakage_remediation``'s prune,
the one other node that could evict a foreign entry, is covered in
``test_blocking_issues_merge_2283.py`` against the real node.

Data is ingested from a real local CSV through ``load_data``'s file path, so no
part of the chain is mocked and nothing touches Supabase, Redis or MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.graph import finalize_output
from src.agents.ml_foundation.data_preparer.nodes.data_loader import load_data
from src.agents.ml_foundation.data_preparer.nodes.ge_validator import run_ge_validation
from src.agents.ml_foundation.data_preparer.nodes.quality_checker import run_quality_checks
from src.agents.ml_foundation.data_preparer.nodes.sampling_frame_audit import (
    audit_sampling_frame,
)
from src.agents.ml_foundation.data_preparer.nodes.schema_validator import (
    run_schema_validation,
)
from src.agents.ml_foundation.data_preparer.state import DataPreparerState

_AUDIT_WORKFLOW_ID = "00000000-0000-0000-0000-000000002283"


def _build_validation_chain():
    """Compile the real ``load_data`` -> ... -> ``finalize_output`` seam."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("load_data", load_data)  # type: ignore[arg-type]
    graph.add_node("audit_sampling_frame", audit_sampling_frame)  # type: ignore[arg-type]
    graph.add_node("run_schema_validation", run_schema_validation)  # type: ignore[arg-type]
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[arg-type]
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]

    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "audit_sampling_frame")
    graph.add_edge("audit_sampling_frame", "run_schema_validation")
    graph.add_edge("run_schema_validation", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    graph.add_edge("run_ge_validation", "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _write_patient_journeys_csv(tmp_path: Path, *, with_nulls: bool = False) -> Path:
    """A frame that FAILS Pandera and PASSES Great Expectations.

    ``PatientJourneysSchema`` (``src/mlops/pandera_schemas.py:191``) requires a
    non-null, unique ``patient_journey_id``; it is omitted here, which raises a
    ``column_in_dataframe`` error — the same shape as the failure captured in
    ``docs/demos/results/2026-09-23_split_contract_cert/``.

    The GE ``patient_journeys`` suite (``src/mlops/data_quality.py:652``) only
    requires ``patient_id`` / ``event_type`` / ``event_date`` to exist and the
    first two to be non-null, all of which hold. GE therefore PASSES and takes
    ``ge_validator``'s happy path — the one that writes ``None`` over the
    channel. Keeping ``event_type`` present also stops ``ge_validator``'s
    ml_patients auto-detect from switching suites.
    """
    n = 60
    dates = pd.date_range("2099-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "patient_id": [f"pat-{i:04d}" for i in range(n)],
            "event_type": ["prescription"] * n,
            "event_date": dates.strftime("%Y-%m-%d"),
            "created_at": dates.strftime("%Y-%m-%d"),
            "days_on_therapy": [30 + (i % 15) for i in range(n)],
            # Precomputed split so ``_split_from_column`` is used verbatim and
            # the fixture does not depend on splitter heuristics.
            "data_split": (["train"] * 36 + ["validation"] * 12 + ["test"] * 6 + ["holdout"] * 6),
        }
    )
    if with_nulls:
        # Nulls in a NON-required column: they drag ``completeness_score``
        # (and so ``overall_score``) down without taking the required-column
        # branch, which lets a caller pin ``qc_min_overall_score`` high enough
        # that ``run_quality_checks`` emits a blocking entry OF ITS OWN. The
        # re-entry tests need that to avoid being vacuous.
        frame["sparse_flag"] = [None if i % 3 == 0 else 1 for i in range(n)]

    path = tmp_path / "patient_journeys.csv"
    frame.to_csv(path, index=False)
    return path


def _base_state(csv_path: Path) -> Dict[str, Any]:
    return {
        "audit_workflow_id": _AUDIT_WORKFLOW_ID,
        "experiment_id": "exp-2283-channel",
        "data_source": {"type": "files", "paths": {"patient_journeys": str(csv_path)}},
        "scope_spec": {
            "date_column": "event_date",
            # ``schema_validator`` cannot key the Pandera registry off a dict
            # ``data_source``; it falls back to ``scope_spec['data_source']``
            # (schema_validator.py:70-77), exactly as a real file-ingestion run
            # does. Without it, schema validation reports "skipped".
            "data_source": "patient_journeys",
            # No deployment_reference -> audit_sampling_frame is an advisory
            # pass-through and contributes no blocking entry of its own, so the
            # schema entry is the only thing under test.
        },
    }


@pytest.mark.asyncio
async def test_schema_failure_survives_to_the_qc_gate(tmp_path: Path) -> None:
    """RED pre-fix: ``run_schema_validation`` writes its blocking entry,
    ``run_quality_checks`` overwrites the channel with a fresh ``[]`` and
    ``run_ge_validation`` then writes ``None`` over it. ``finalize_output``
    sees no blocking issues and passes the gate on data that failed its
    schema. Post-fix the entry survives both hops and the gate blocks."""
    csv_path = _write_patient_journeys_csv(tmp_path)
    app = _build_validation_chain()

    final_state = await app.ainvoke(_base_state(csv_path))

    # Precondition: the producer really did fail. That GE takes its
    # ``None``-writing happy path on this fixture is pinned separately by
    # ``test_fixture_passes_ge_validation`` (the ``ge_validation_status`` key
    # is not readable here — see that test's docstring).
    assert final_state["schema_validation_status"] == "failed"

    blocking_issues = final_state.get("blocking_issues") or []
    assert any("Schema validation failed" in issue for issue in blocking_issues), (
        f"schema blocking entry was destroyed before the gate; blocking_issues={blocking_issues!r}"
    )

    # The observable consequence: training must not be cleared to proceed.
    assert final_state["gate_passed"] is False
    assert final_state["is_ready"] is False
    assert any("Schema validation failed" in blocker for blocker in final_state["blockers"])


@pytest.mark.asyncio
async def test_quality_checker_preserves_upstream_blocking_issues(tmp_path: Path) -> None:
    """The first clobber site in isolation: after ``run_quality_checks`` the
    channel must still carry the schema entry."""
    csv_path = _write_patient_journeys_csv(tmp_path)

    graph = StateGraph(DataPreparerState)
    graph.add_node("load_data", load_data)  # type: ignore[arg-type]
    graph.add_node("run_schema_validation", run_schema_validation)  # type: ignore[arg-type]
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[arg-type]
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_schema_validation")
    graph.add_edge("run_schema_validation", "run_quality_checks")
    graph.add_edge("run_quality_checks", END)

    final_state = await graph.compile().ainvoke(_base_state(csv_path))

    assert final_state["schema_validation_status"] == "failed"
    blocking_issues = final_state.get("blocking_issues") or []
    assert any("Schema validation failed" in issue for issue in blocking_issues), (
        f"run_quality_checks overwrote the channel; blocking_issues={blocking_issues!r}"
    )


@pytest.mark.asyncio
async def test_ge_validator_does_not_wipe_a_populated_channel(tmp_path: Path) -> None:
    """The second clobber site in isolation: ``run_ge_validation``'s happy path
    must not write ``None`` over a populated channel."""
    csv_path = _write_patient_journeys_csv(tmp_path)

    graph = StateGraph(DataPreparerState)
    graph.add_node("load_data", load_data)  # type: ignore[arg-type]
    graph.add_node("run_schema_validation", run_schema_validation)  # type: ignore[arg-type]
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[arg-type]
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_schema_validation")
    graph.add_edge("run_schema_validation", "run_ge_validation")
    graph.add_edge("run_ge_validation", END)

    final_state = await graph.compile().ainvoke(_base_state(csv_path))

    assert final_state["schema_validation_status"] == "failed"
    blocking_issues = final_state.get("blocking_issues")
    assert blocking_issues is not None, "ge_validator wrote None over a populated channel"
    assert any("Schema validation failed" in issue for issue in blocking_issues)


@pytest.mark.asyncio
async def test_sampling_frame_entry_reaches_the_gate_without_re_promotion(
    tmp_path: Path,
) -> None:
    """``graph.py`` used to re-derive ``audit_sampling_frame``'s blocking entry
    at the gate because ``run_quality_checks`` destroyed it. That workaround is
    deleted by #2283; this pins the property it was protecting on THIS chain —
    the audit's own entry survives quality_checker and ge_validator unaided,
    because it is a FOREIGN entry to both and ``merge_blocking_issues`` carries
    it through untouched.

    Scope, stated plainly: this does not by itself prove the re-promotion is
    unnecessary across the whole production graph. The other way the entry
    could be lost is ``leakage_remediation``'s prune, which is on a branch this
    chain omits; that is covered directly against the real node by
    ``test_blocking_issues_merge_2283.py::
    test_leakage_remediation_cannot_evict_a_foreign_entry``. Together the two
    cover both loss paths.
    """
    csv_path = _write_patient_journeys_csv(tmp_path)
    state = _base_state(csv_path)
    # A reference whose ``days_on_therapy`` mean is nowhere near the training
    # frame's (~37) drives max_drift_score far past the 0.3 blocking threshold.
    state["scope_spec"] = {
        **state["scope_spec"],
        "deployment_reference": {
            "distributions": {
                "days_on_therapy": {"mean": 1000.0, "std": 4.0},
            }
        },
    }

    app = _build_validation_chain()
    final_state = await app.ainvoke(state)

    report = final_state["sampling_frame_audit_report"]
    assert report.get("blocking_detail"), f"audit did not block; report={report!r}"

    blocking_issues = final_state["blocking_issues"]
    sampling_entries = [i for i in blocking_issues if i.startswith("sampling_frame_drift: ")]
    assert len(sampling_entries) == 1, (
        f"sampling-frame entry lost or duplicated: {blocking_issues!r}"
    )
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_re_entry_replaces_a_nodes_own_blocker_instead_of_appending(
    tmp_path: Path,
) -> None:
    """``graph.py`` routes ``finalize_output -> qc_remediation --retry-->
        run_quality_checks -> run_ge_validation ...``, so a QC retry re-runs the
        whole downstream chain. This pins the property that rules out both an
        ``operator.add`` reducer (#2238 / PR #2251) and a plain ``incoming + own``
        merge: a second pass must REPLACE each node's own entries, not append.

    The fixture's ``sparse_flag`` column carries nulls and is NOT declared
    required, so it drags ``overall_score`` to ~0.988 without taking the
    required-column branch; with ``qc_min_overall_score`` pinned at 0.99 on the
    STATE (not ``scope_spec`` — that is a typed contract and drops the unknown
    key at the channel boundary, measured), ``run_quality_checks`` genuinely
    emits an entry OF ITS OWN on every pass. Without that this test would be
    vacuous: with only the upstream schema entry in the channel there is
    nothing a naive concatenation could duplicate, so it would pass against the
    very implementation it is meant to reject.

    The required-column branch is avoided on purpose — it appends a dict to
    ``failed_expectations``, which ``DataPreparerState`` declares
    ``Optional[List[str]]``, so the graph dies on a pydantic ValidationError at
    the next channel boundary. That is a real pre-existing defect, unrelated to
    this fix and reported separately.
    """
    csv_path = _write_patient_journeys_csv(tmp_path, with_nulls=True)
    state = _base_state(csv_path)
    # Pinned on the STATE, not on scope_spec: ``scope_spec`` is a typed
    # contract (ScopeSpecSchema) and an undeclared key is dropped at the
    # channel boundary, so a scope_spec override silently does nothing here.
    state["qc_min_overall_score"] = 0.99

    graph = StateGraph(DataPreparerState)
    graph.add_node("load_data", load_data)  # type: ignore[arg-type]
    graph.add_node("run_schema_validation", run_schema_validation)  # type: ignore[arg-type]
    # The same node functions wired twice, standing in for the retry edge
    # without dragging qc_remediation's LLM call into a unit test.
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[arg-type]
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[arg-type]
    graph.add_node("run_quality_checks_retry", run_quality_checks)  # type: ignore[arg-type]
    graph.add_node("run_ge_validation_retry", run_ge_validation)  # type: ignore[arg-type]
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_schema_validation")
    graph.add_edge("run_schema_validation", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    graph.add_edge("run_ge_validation", "run_quality_checks_retry")
    graph.add_edge("run_quality_checks_retry", "run_ge_validation_retry")
    graph.add_edge("run_ge_validation_retry", END)

    final_state = await graph.compile().ainvoke(state)
    blocking_issues = final_state["blocking_issues"]

    # The precondition this test turns on: QC really did contribute its own
    # entry, so a naive concatenation would have something to duplicate.
    qc_entries = [i for i in blocking_issues if i.startswith("quality_check: ")]
    assert len(qc_entries) == 1, (
        f"expected exactly one quality_check entry after two passes: {blocking_issues!r}"
    )
    assert sum("Schema validation failed" in i for i in blocking_issues) == 1
    assert len(set(blocking_issues)) == len(blocking_issues)

    # And a single pass over the same fixture yields the identical channel.
    single = await _build_single_pass_reference(state)
    assert blocking_issues == single, f"two_pass={blocking_issues!r} one_pass={single!r}"


@pytest.mark.asyncio
async def test_a_nodes_stale_own_entry_is_retracted_when_it_passes(
    tmp_path: Path,
) -> None:
    """The other half of the ownership contract, on real nodes: an entry this
    node wrote on a previous pass must disappear once the condition clears.

    Without this, a QC retry could never clear the gate — the blocker that
    remediation actually fixed would sit in the channel forever. A plain
    ``incoming + own`` merge fails exactly here.
    """
    csv_path = _write_patient_journeys_csv(tmp_path)
    state = _base_state(csv_path)
    # Stand in for a previous pass's contributions, alongside a foreign entry.
    state["blocking_issues"] = [
        "quality_check: Overall QC score (0.42) below minimum threshold (0.80)",
        "ge_validation: failed for train: 3 expectations failed",
        "sampling_frame_drift: some earlier, still-unresolved drift",
    ]

    graph = StateGraph(DataPreparerState)
    graph.add_node("load_data", load_data)  # type: ignore[arg-type]
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[arg-type]
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[arg-type]
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    graph.add_edge("run_ge_validation", END)

    final_state = await graph.compile().ainvoke(state)
    blocking_issues = final_state["blocking_issues"]

    assert not [i for i in blocking_issues if i.startswith("quality_check: ")], (
        f"stale quality_check entry survived a clean pass: {blocking_issues!r}"
    )
    assert not [i for i in blocking_issues if i.startswith("ge_validation: ")], (
        f"stale ge_validation entry survived a clean pass: {blocking_issues!r}"
    )
    # The foreign entry is untouched — retraction is per-owner, not a wipe.
    assert "sampling_frame_drift: some earlier, still-unresolved drift" in blocking_issues


async def _build_single_pass_reference(state: Dict[str, Any]) -> list:
    """The single-pass channel the two-pass run above must reproduce exactly."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("load_data", load_data)  # type: ignore[arg-type]
    graph.add_node("run_schema_validation", run_schema_validation)  # type: ignore[arg-type]
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[arg-type]
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[arg-type]
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_schema_validation")
    graph.add_edge("run_schema_validation", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    graph.add_edge("run_ge_validation", END)
    final_state = await graph.compile().ainvoke(dict(state))
    return list(final_state["blocking_issues"])


@pytest.mark.asyncio
async def test_fixture_passes_ge_validation(tmp_path: Path) -> None:
    """Fixture invariant, not a bug assertion.

    The graph tests above are only exercising ``ge_validator``'s
    ``None``-writing happy path while GE PASSES on this frame. That cannot be
    asserted from the graph's final state: ``ge_validation_status`` (and every
    other ``ge_*`` key the node returns) is **not declared on
    ``DataPreparerState``**, and LangGraph drops undeclared keys at the channel
    boundary, so the node's verdict never reaches state. Only
    ``blocking_issues`` — which is declared — survives. No production code
    reads ``ge_validation_status`` from state today (grepped repo-wide), so
    that is a separate, currently-harmless defect; it is recorded here because
    it is the reason this precondition is checked on the node's return value
    instead.
    """
    csv_path = _write_patient_journeys_csv(tmp_path)
    state = _base_state(csv_path)
    state.update(await load_data(state))  # type: ignore[arg-type]

    result = await run_ge_validation(state)  # type: ignore[arg-type]

    assert result["ge_validation_status"] in ("passed", "warning"), (
        "fixture no longer takes ge_validator's happy path; the graph tests "
        f"above would stop covering the None write. result={result!r}"
    )
