"""Graph-level regressions for #2294: blocking conditions that never reached the
``blocking_issues`` channel, so ``finalize_output`` passed the QC gate.

#2283 / PR #2285 fixed the DESTRUCTION of entries already in the channel. These
tests pin the complementary producer-side gaps: a condition that should block
and was never written. Each one drives the REAL node(s) through a compiled
``StateGraph(DataPreparerState)`` that ends at the REAL ``finalize_output`` and
asserts ``gate_passed`` — asserting on a node's return dict alone cannot see
the gate (see ``test_blocking_issues_channel_2283.py``'s docstring).

Upstream QC is seeded on the initial state (``qc_status="passed"``,
``overall_score=0.95``) rather than by running ``run_quality_checks``, so the
only thing that can fail the gate in these graphs is the channel under test.
Where a crash path has to be triggered, it is either a real data shape that
crashes the check (preferred) or a ``monkeypatch`` that raises inside the
check's own dependency. Neither the node under test nor the gate is stubbed.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.graph import (
    _route_after_leakage_detection,
    _route_after_leakage_remediation,
    finalize_output,
)
from src.agents.ml_foundation.data_preparer.nodes import leakage_detector as ld
from src.agents.ml_foundation.data_preparer.nodes.feast_registrar import (
    register_features_in_feast,
)
from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import detect_leakage
from src.agents.ml_foundation.data_preparer.nodes.leakage_remediation import (
    review_and_remediate_leakage,
)
from src.agents.ml_foundation.data_preparer.nodes.schema_validator import (
    run_schema_validation,
)
from src.agents.ml_foundation.data_preparer.nodes.sufficiency_check import (
    run_sufficiency_check,
)
from src.agents.ml_foundation.data_preparer.state import DataPreparerState

# Upstream QC verdict every graph below starts from: a clean pass, so a
# ``gate_passed=False`` can only come from ``blocking_issues``.
_CLEAN_UPSTREAM_QC: Dict[str, Any] = {
    "audit_workflow_id": "00000000-0000-0000-0000-000000002294",
    "qc_status": "passed",
    "overall_score": 0.95,
}


def _leakage_gate_graph():
    """``detect_leakage -> finalize_output`` over the real state schema."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("detect_leakage", detect_leakage)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point("detect_leakage")
    graph.add_edge("detect_leakage", "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _moderate_only_frame(n: int = 400) -> pd.DataFrame:
    """A frame whose ONLY structured finding is a MODERATE target correlation.

    MODERATE findings are hard to produce alone: a feature correlated
    0.70-0.85 with a binary target normally also has single-feature AUC > 0.80
    (HIGH). Here 75% of positives carry a large spike and the rest sit slightly
    BELOW the negatives, which keeps Pearson r ~0.76 (MODERATE, p < 0.001) while
    rank AUC stays under 0.80 and the class ranges still overlap. Measured
    against every structural check: the only finding is
    ``target_correlation/moderate``.
    """
    rng = np.random.default_rng(2294)
    target = np.array([0, 1] * (n // 2))
    spend_spike = rng.standard_normal(n)
    positives = np.where(target == 1)[0]
    k = int(0.75 * len(positives))
    spend_spike[positives[:k]] += 60.0
    spend_spike[positives[k:]] += -1.5
    return pd.DataFrame({"spend_spike": spend_spike, "target": target})


@pytest.mark.asyncio
async def test_legacy_temporal_leak_blocks_alongside_a_moderate_finding() -> None:
    """Item 1. A legacy temporal leak must block even when a MODERATE
    structured finding coexists with it.

    ``f953304ea`` replaced "every leakage issue blocks" with a severity filter
    for STRUCTURED findings and kept legacy issues blocking through the
    predicate ``blocking_findings or (leakage_detected and not findings)`` —
    "leakage detected but no findings means it came from a legacy check". One
    MODERATE finding breaks that approximation and the temporal leak blocks
    nothing.
    """
    frame = _moderate_only_frame()
    n = len(frame)
    event = pd.date_range("2024-01-01", periods=n, freq="D")
    target_date = event + pd.Timedelta(days=30)
    # 40 rows whose event happens AFTER the label date: a real temporal leak.
    target_date = target_date.where(np.arange(n) >= 40, event - pd.Timedelta(days=1))
    frame["event_date"] = event.strftime("%Y-%m-%d")
    frame["target_date"] = target_date.strftime("%Y-%m-%d")

    state: Dict[str, Any] = {
        "experiment_id": "exp-2294-legacy-moderate",
        "train_df": frame,
        "scope_spec": {
            "prediction_target": "target",
            # ``check_target_leakage`` only scans ``required_features``.
            "required_features": ["spend_spike"],
            "event_date_column": "event_date",
            "target_date_column": "target_date",
        },
        **_CLEAN_UPSTREAM_QC,
    }

    final_state = await _leakage_gate_graph().ainvoke(state)

    # Preconditions: the temporal leak was found, and the structured findings
    # are exactly one MODERATE target correlation — the shape that failed open.
    assert any(i.startswith("Temporal leakage:") for i in final_state["leakage_issues"])
    findings = final_state["leakage_findings"]
    assert [(f["check_name"], f["severity"]) for f in findings] == [
        ("target_correlation", "moderate")
    ], findings

    blocking = final_state["blocking_issues"] or []
    assert any(i.startswith("leakage: Temporal leakage:") for i in blocking), (
        f"legacy temporal leak never reached the channel: {blocking!r}"
    )
    # The MODERATE finding itself is review-only and must NOT block.
    assert not any("target_correlation" in i for i in blocking), blocking
    assert final_state["gate_passed"] is False


# =============================================================================
# Item 2 — schema_validator: crashed audits and non-passed-without-errors
# =============================================================================


def _schema_gate_graph(*, passes: int = 1):
    """``run_schema_validation`` (``passes`` times) ``-> finalize_output``.

    ``passes=2`` wires the same node twice to prove its entry is replaced, not
    duplicated, if the node ever runs again on a populated channel.
    """
    graph = StateGraph(DataPreparerState)
    names = [f"run_schema_validation_{i}" for i in range(passes)]
    for name in names:
        graph.add_node(name, run_schema_validation)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point(names[0])
    for a, b in zip(names, names[1:], strict=False):
        graph.add_edge(a, b)
    graph.add_edge(names[-1], "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _schema_state() -> Dict[str, Any]:
    """A frame the REAL ``patient_journeys`` Pandera schema is resolved for."""
    n = 40
    frame = pd.DataFrame(
        {
            "patient_journey_id": [f"pj-{i:04d}" for i in range(n)],
            "patient_id": [f"pat-{i:04d}" for i in range(n)],
        }
    )
    return {
        "experiment_id": "exp-2294-schema",
        "train_df": frame,
        # A string ``data_source`` keys the Pandera registry directly.
        "data_source": "patient_journeys",
        "scope_spec": {"data_source": "patient_journeys"},
        # A foreign entry that every schema path must carry through untouched.
        "blocking_issues": ["sampling_frame_drift: an unrelated upstream blocker"],
        **_CLEAN_UPSTREAM_QC,
    }


def _raise_runtime(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("forced crash inside a dependency")


@pytest.mark.asyncio
async def test_crashed_schema_validation_blocks_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 2, generic ``except``: the node recorded status ``error`` and an
    ``error`` key but no blocker, and ``finalize_output`` reads neither.
    The crash is forced inside the validator's own dependency."""
    import src.mlops.pandera_schemas as pandera_schemas

    monkeypatch.setattr(pandera_schemas, "validate_dataframe", _raise_runtime)

    final_state = await _schema_gate_graph().ainvoke(_schema_state())

    assert final_state["schema_validation_status"] == "error"
    blocking = final_state["blocking_issues"]
    schema_entries = [i for i in blocking if i.startswith("schema: validation error:")]
    assert len(schema_entries) == 1, f"crashed schema audit left no blocker: {blocking!r}"
    assert "forced crash inside a dependency" in schema_entries[0]
    assert "sampling_frame_drift: an unrelated upstream blocker" in blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_schema_import_failure_blocks_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 2, ``ImportError`` branch: a validator that could not even load
    audited nothing, and must not read as a pass."""
    import sys

    # ``None`` in ``sys.modules`` makes ``from src.mlops.pandera_schemas
    # import ...`` raise ImportError inside the node's own try.
    monkeypatch.setitem(sys.modules, "src.mlops.pandera_schemas", None)

    final_state = await _schema_gate_graph().ainvoke(_schema_state())

    assert final_state["schema_validation_status"] == "error"
    blocking = final_state["blocking_issues"]
    assert any(i.startswith("schema: validation error:") for i in blocking), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_non_passed_split_without_error_detail_still_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 2, the status branch: the overall verdict keyed on ``all_errors``,
    so a split whose status was NOT ``passed`` but which carried an empty
    ``errors`` list counted as passing. No real Pandera path produces that
    shape today (``validate_dataframe``'s ``error`` branch always attaches one
    error), so the empty-errors result is injected at the dependency; the
    point is that the node's verdict must key on the split's STATUS."""
    import src.mlops.pandera_schemas as pandera_schemas

    monkeypatch.setattr(
        pandera_schemas,
        "validate_dataframe",
        lambda *_a, **_k: {"status": "error", "errors": []},
    )

    final_state = await _schema_gate_graph().ainvoke(_schema_state())

    assert final_state["schema_validation_status"] == "failed"
    blocking = final_state["blocking_issues"]
    assert any(i.startswith("schema: Schema validation failed:") for i in blocking), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_schema_entry_is_replaced_not_duplicated_on_re_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 2, idempotence: the schema entry goes through
    ``merge_blocking_issues`` under its own kind, so a second pass replaces it."""
    import src.mlops.pandera_schemas as pandera_schemas

    monkeypatch.setattr(pandera_schemas, "validate_dataframe", _raise_runtime)

    final_state = await _schema_gate_graph(passes=2).ainvoke(_schema_state())

    blocking = final_state["blocking_issues"]
    assert sum(i.startswith("schema: ") for i in blocking) == 1, blocking
    assert blocking.count("sampling_frame_drift: an unrelated upstream blocker") == 1


# =============================================================================
# Item 3 — inner leakage checks that swallowed their own crash
# =============================================================================


def _leakage_state(train_df: pd.DataFrame, **extra: Any) -> Dict[str, Any]:
    scope_spec = {"prediction_target": "target", **extra.pop("scope_spec", {})}
    return {
        "experiment_id": "exp-2294-leakage-inner",
        "train_df": train_df,
        "scope_spec": scope_spec,
        **extra,
        **_CLEAN_UPSTREAM_QC,
    }


def _noise_frame(n: int = 200) -> pd.DataFrame:
    """No leakage anywhere: a clean run must pass, so a block means the crash."""
    rng = np.random.default_rng(22940)
    return pd.DataFrame(
        {
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "target": np.array([0, 1] * (n // 2)),
        }
    )


def _incomplete_entries(final_state: Dict[str, Any], check: str) -> list:
    prefix = f"leakage: Leakage audit incomplete: {check}"
    return [i for i in (final_state["blocking_issues"] or []) if i.startswith(prefix)]


@pytest.mark.asyncio
async def test_clean_frame_passes_the_leakage_gate() -> None:
    """Control for the tests below: the same frame with nothing crashing
    passes the gate, so each block below is caused by the crash alone."""
    state = _leakage_state(_noise_frame(), scope_spec={"required_features": ["noise_a"]})

    final_state = await _leakage_gate_graph().ainvoke(state)

    assert final_state["blocking_issues"] == [], final_state["blocking_issues"]
    assert final_state["gate_passed"] is True


@pytest.mark.asyncio
async def test_crashed_target_leakage_check_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """``check_target_leakage`` wraps its whole loop in one ``except`` that only
    logged, so a crash mid-loop left every remaining feature unaudited and
    reported nothing.

    Data that crashes this check for real (e.g. a nullable ``Int64`` column,
    which ``np.issubdtype`` rejects) also crashes ``_get_numeric_features`` a
    few lines later, which the node's OUTER ``except`` already blocks on — so
    it cannot isolate this path. The crash is forced instead inside the check's
    own dependency: on a continuous target the check correlates with
    ``Series.corr``, which no other leakage check calls.
    """
    frame = _noise_frame()
    frame["target"] = frame["noise_b"] * 3.0 + 1.0  # continuous target
    monkeypatch.setattr(pd.Series, "corr", _raise_runtime)

    state = _leakage_state(frame, scope_spec={"required_features": ["noise_a"]})
    final_state = await _leakage_gate_graph().ainvoke(state)

    entries = _incomplete_entries(final_state, "target_correlation")
    assert len(entries) == 1, final_state["blocking_issues"]
    assert "RuntimeError" in entries[0]
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_contamination_is_audited_despite_a_list_column() -> None:
    """A list-valued column (a JSON array from the source) made the row-hash
    fallback raise ``unhashable type: 'list'`` when there is no id column, and
    the check swallowed it: a real 40-row train/validation overlap went
    unreported and the gate passed. No monkeypatching — the data crashes it.

    The fix does not merely record the crash: such a column never reaches a
    model (``data_transformer`` drops it), so the rows are compared on the
    remaining columns and the overlap is actually FOUND.
    """
    train = _noise_frame()
    train["codes"] = [[i % 7] for i in range(len(train))]
    validation = train.iloc[:40].copy()

    state = _leakage_state(train, validation_df=validation)
    final_state = await _leakage_gate_graph().ainvoke(state)

    blocking = final_state["blocking_issues"] or []
    assert any(
        i.startswith("leakage: Train-validation contamination: 40 samples") for i in blocking
    ), blocking
    # ...and the list column is "not applicable" to the categorical check,
    # not an incomplete audit.
    assert not [i for i in blocking if "Leakage audit incomplete" in i], blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_crashed_contamination_check_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contamination check's own ``except``: forced inside the row-hash
    fallback (``DataFrame.apply`` — used by no other leakage check)."""
    train = _noise_frame()
    validation = train.iloc[:40].copy()
    monkeypatch.setattr(pd.DataFrame, "apply", _raise_runtime)

    final_state = await _leakage_gate_graph().ainvoke(
        _leakage_state(train, validation_df=validation)
    )

    entries = _incomplete_entries(final_state, "train_test_contamination")
    assert len(entries) == 1, final_state["blocking_issues"]
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_path", "attr", "check"),
    [
        ("sklearn.feature_selection", "mutual_info_classif", "mutual_information"),
        ("sklearn.metrics", "roc_auc_score", "single_feature_auc"),
    ],
)
async def test_crashed_structural_check_blocks(
    monkeypatch: pytest.MonkeyPatch, module_path: str, attr: str, check: str
) -> None:
    """The structural checks catch per check (MI) or per feature (AUC and the
    rest). The crash is forced inside the check's own sklearn dependency."""
    import importlib

    monkeypatch.setattr(importlib.import_module(module_path), attr, _raise_runtime)

    final_state = await _leakage_gate_graph().ainvoke(_leakage_state(_noise_frame()))

    assert _incomplete_entries(final_state, check), final_state["blocking_issues"]
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_tz_aware_event_dates_are_compared_with_naive_label_dates() -> None:
    """Comparing a tz-aware event date with a naive label date raised
    ``TypeError`` and ``_check_date_ordering`` returned ``(0, 0.0)`` — "no
    temporal leakage" — for every row, including rows that do leak. Both sides
    are now normalised alike (aware -> UTC, naive stays naive) and compared, so
    the leak on every row is FOUND."""
    frame = _noise_frame()
    n = len(frame)
    event = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    frame["event_date"] = event
    # Every label date PRECEDES its event: a leak on every row.
    frame["target_date"] = (event - pd.Timedelta(days=1)).tz_localize(None)

    state = _leakage_state(
        frame,
        scope_spec={
            "required_features": ["noise_a"],
            "event_date_column": "event_date",
            "target_date_column": "target_date",
        },
    )
    final_state = await _leakage_gate_graph().ainvoke(state)

    blocking = final_state["blocking_issues"] or []
    assert any(
        i.startswith(f"leakage: Temporal leakage: {n} rows") and "event_date > target_date" in i
        for i in blocking
    ), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_unparseable_split_date_is_an_incomplete_temporal_audit() -> None:
    """Temporal strategies 2 and 3 both need ``split_date``; an unparseable one
    skipped them without a word, which read as "no temporal leakage"."""
    state = _leakage_state(
        _noise_frame(),
        scope_spec={
            "required_features": ["noise_a"],
            "split_date": "not-a-date",
            "date_column": "event_date",
        },
    )

    final_state = await _leakage_gate_graph().ainvoke(state)

    blocking = final_state["blocking_issues"] or []
    assert any(
        i.startswith("leakage: Temporal leakage check incomplete: split_date 'not-a-date'")
        for i in blocking
    ), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_scope",
    [
        {"split_date": "2024-06-01", "feature_date_columns": ["event_ts"]},
        {"event_date_column": "event_ts", "target_date_column": "label_ts"},
    ],
)
async def test_configured_date_column_that_never_parses_is_an_incomplete_audit(
    extra_scope: Dict[str, Any],
) -> None:
    """Codex r2: ``pd.to_datetime(errors="coerce")`` turns a configured date
    column whose values are ALL unparseable into all-``NaT``, so the check
    counted zero future rows / zero bad pairs and read as "no temporal
    leakage" without examining a single date."""
    frame = _noise_frame()
    frame["event_ts"] = ["not a date"] * len(frame)
    frame["label_ts"] = pd.date_range("2024-01-01", periods=len(frame), freq="D").strftime(
        "%Y-%m-%d"
    )
    state = _leakage_state(frame, scope_spec={"required_features": ["noise_a"], **extra_scope})

    final_state = await _leakage_gate_graph().ainvoke(state)

    blocking = final_state["blocking_issues"] or []
    assert any(
        i.startswith("leakage: Temporal leakage check incomplete") and "event_ts" in i
        for i in blocking
    ), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_all_null_configured_date_column_is_not_an_incomplete_audit() -> None:
    """Control: a configured date column with no values has nothing that could
    leak; only values that exist and fail to parse make the audit incomplete."""
    frame = _noise_frame()
    frame["event_ts"] = [None] * len(frame)
    state = _leakage_state(
        frame,
        scope_spec={
            "required_features": ["noise_a"],
            "split_date": "2024-06-01",
            "feature_date_columns": ["event_ts"],
        },
    )

    final_state = await _leakage_gate_graph().ainvoke(state)

    assert final_state["blocking_issues"] == [], final_state["blocking_issues"]


@pytest.mark.asyncio
async def test_string_labelled_binary_target_is_audited_not_blocked() -> None:
    """Codex r3: ``check_single_feature_auc`` cast the target with
    ``astype(float)``, which raises on ``"yes"``/``"no"`` labels. On main that
    was swallowed (the AUC audit silently did not run); recording it as an
    incomplete audit blocked EVERY string-labelled run. The labels are a valid
    binary target: the check must run, find nothing here, and not block."""
    frame = _noise_frame()
    frame["target"] = np.where(frame["target"] == 1, "yes", "no")

    final_state = await _leakage_gate_graph().ainvoke(_leakage_state(frame))

    assert final_state["blocking_issues"] == [], final_state["blocking_issues"]
    assert final_state["gate_passed"] is True


@pytest.mark.asyncio
async def test_string_labelled_binary_target_leak_is_found() -> None:
    """...and the same check now actually audits string labels: a feature that
    separates ``"yes"`` from ``"no"`` is flagged instead of skipped."""
    rng = np.random.default_rng(22947)
    n = 200
    labels = np.array(["no", "yes"] * (n // 2))
    frame = pd.DataFrame(
        {
            "leaky": (labels == "yes") * 2.5 + rng.standard_normal(n),
            "noise": rng.standard_normal(n),
            "target": labels,
        }
    )

    final_state = await _leakage_gate_graph().ainvoke(_leakage_state(frame))

    blocking = final_state["blocking_issues"] or []
    assert any("] single_feature_auc: Feature 'leaky'" in i for i in blocking), blocking
    assert not any("incomplete" in i for i in blocking), blocking
    assert final_state["gate_passed"] is False


def _epoch_state(values: list, split_date: str, **scope: Any) -> Dict[str, Any]:
    frame = _noise_frame()
    frame["event_ts"] = (values * len(frame))[: len(frame)]
    return _leakage_state(
        frame,
        scope_spec={
            "required_features": ["noise_a"],
            "split_date": split_date,
            "feature_date_columns": ["event_ts"],
            **scope,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "unit", "split_date"),
    [
        # 2025-02-01T00:00:00Z in epoch seconds and millis (codex r3).
        ([1_738_368_000], "s", "2025-01-01"),
        ([1_738_368_000_000], "ms", "2025-01-01"),
        # 1969-12-30 in epoch millis, after a 1969-01-01 split: magnitude-based
        # inference read it as SECONDS (1964) and missed it (codex r4).
        ([-172_800_000], "ms", "1969-01-01"),
    ],
    ids=["epoch_seconds", "epoch_millis", "pre_1970_millis"],
)
async def test_numeric_date_column_with_declared_unit_is_audited(
    values: list, unit: str, split_date: str
) -> None:
    """``pd.to_datetime`` reads a bare integer as NANOseconds, so an epoch date
    became 1970 and a real temporal leak read as clean. With the unit DECLARED
    (``scope_spec.epoch_units``, which the typed ScopeSpecSchema carries) the
    leak is found."""
    state = _epoch_state(values, split_date, epoch_units={"event_ts": unit})

    final_state = await _leakage_gate_graph().ainvoke(state)

    blocking = final_state["blocking_issues"] or []
    assert any(i.startswith("leakage: Temporal leakage:") and "event_ts" in i for i in blocking), (
        blocking
    )
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [[1_738_368_000], [2025]], ids=["epoch_like", "year_like"])
async def test_numeric_date_column_without_declared_unit_is_unverifiable(values: list) -> None:
    """No unit is ever guessed: a numeric configured date column without a
    declared unit is an unverifiable temporal audit, which blocks loudly."""
    final_state = await _leakage_gate_graph().ainvoke(_epoch_state(values, "2025-01-01"))

    blocking = final_state["blocking_issues"] or []
    assert any(
        i.startswith("leakage: Temporal leakage check incomplete")
        and "'event_ts' is numeric with no declared epoch unit" in i
        for i in blocking
    ), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_numeric_duration_named_like_a_date_is_not_auto_detected() -> None:
    """Strategy 3 auto-detects date columns by NAME; ``time_on_therapy``
    matches ``time_`` but is a duration. A numeric column is a date only with a
    declared unit, so this neither hides as 1970 nor blocks the run."""
    frame = _noise_frame()
    frame["time_on_therapy"] = np.arange(len(frame)) % 90
    state = _leakage_state(
        frame,
        scope_spec={
            "required_features": ["noise_a"],
            "split_date": "2025-01-01",
            "date_column": "event_date",
        },
    )

    final_state = await _leakage_gate_graph().ainvoke(state)

    assert final_state["blocking_issues"] == [], final_state["blocking_issues"]


@pytest.mark.asyncio
async def test_offset_aware_split_date_is_compared_in_utc() -> None:
    """Codex r4: the event dates were converted to UTC but the split date only
    had its offset STRIPPED. ``2025-01-01T00:00+05:00`` is 2024-12-31T19:00Z,
    so events at 22:00Z / 23:00Z are after it — and were all missed against a
    naive 2025-01-01T00:00."""
    frame = _noise_frame()
    frame["event_ts"] = ["2024-12-31T22:00:00Z", "2024-12-31T23:00:00Z"] * (len(frame) // 2)
    state = _leakage_state(
        frame,
        scope_spec={
            "required_features": ["noise_a"],
            "split_date": "2025-01-01T00:00:00+05:00",
            "feature_date_columns": ["event_ts"],
        },
    )

    final_state = await _leakage_gate_graph().ainvoke(state)

    blocking = final_state["blocking_issues"] or []
    assert any(
        i.startswith(f"leakage: Temporal leakage: {len(frame)} rows") and "event_ts" in i
        for i in blocking
    ), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second", "leaks"),
    [("2024-07-15T12:00:00-04:00", False), ("2025-07-15T12:00:00-04:00", True)],
    ids=["before_split", "after_split"],
)
async def test_dst_spanning_offset_strings_are_audited(second: str, leaks: bool) -> None:
    """Codex r3: offset-aware strings spanning a DST change parse to an OBJECT
    series (mixed UTC offsets), and ``.dt.tz`` raised — a false "incomplete"
    block on perfectly valid timestamps. They are compared in UTC now."""
    frame = _noise_frame()
    frame["event_ts"] = ["2024-01-15T12:00:00-05:00", second] * (len(frame) // 2)
    state = _leakage_state(
        frame,
        scope_spec={
            "required_features": ["noise_a"],
            "split_date": "2025-01-01",
            "feature_date_columns": ["event_ts"],
        },
    )

    final_state = await _leakage_gate_graph().ainvoke(state)

    blocking = final_state["blocking_issues"] or []
    assert not any("incomplete" in i for i in blocking), blocking
    assert any(i.startswith("leakage: Temporal leakage:") for i in blocking) is leaks, blocking


def _string_dates_state(values: list, **scope: Any) -> Dict[str, Any]:
    frame = _noise_frame()
    frame["event_ts"] = (values * len(frame))[: len(frame)]
    return _leakage_state(
        frame,
        scope_spec={
            "required_features": ["noise_a"],
            "split_date": "2025-01-01",
            "feature_date_columns": ["event_ts"],
            **scope,
        },
    )


@pytest.mark.asyncio
async def test_mixed_format_iso_dates_are_all_parsed() -> None:
    """Codex r5: pandas inferred ONE format from the first value
    (``2024-12-31``) and coerced ``2025-02-01T00:00:00`` — the leaking row —
    to ``NaT``, which dropped out of the comparison. Every ISO 8601 shape is
    parsed now, so the future row is found."""
    final_state = await _leakage_gate_graph().ainvoke(
        _string_dates_state(["2024-12-31", "2025-02-01T00:00:00"])
    )

    blocking = final_state["blocking_issues"] or []
    assert any(i.startswith("leakage: Temporal leakage:") and "event_ts" in i for i in blocking), (
        blocking
    )
    assert not any("unverifiable" in i for i in blocking), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_partly_unparseable_date_column_is_unverifiable() -> None:
    """A value that does not parse could be the leaking row; a partly
    unparseable column can no longer read clean."""
    final_state = await _leakage_gate_graph().ainvoke(
        _string_dates_state(["2024-06-01", "not a date"])
    )

    blocking = final_state["blocking_issues"] or []
    assert any(
        "temporal check unverifiable: 'event_ts' has 100 unparseable values" in i for i in blocking
    ), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_clean_iso_dates_before_the_split_pass() -> None:
    """Control: valid ISO dates (with nulls) that are all before the split do
    not block."""
    final_state = await _leakage_gate_graph().ainvoke(
        _string_dates_state(["2024-06-01", None, "2024-11-30T08:15:00"])
    )

    assert final_state["blocking_issues"] == [], final_state["blocking_issues"]


@pytest.mark.asyncio
async def test_incomplete_audit_entry_is_retracted_by_a_clean_rerun() -> None:
    """The incomplete-audit entry is a ``leakage:`` entry, so the next
    ``detect_leakage`` pass (the recheck, or a QC retry) replaces it: once the
    check completes, the gate can pass again."""
    state = _leakage_state(_noise_frame(), scope_spec={"required_features": ["noise_a"]})
    state["blocking_issues"] = [
        "leakage: Leakage audit incomplete: target_correlation: TypeError: earlier pass"
    ]

    final_state = await _leakage_gate_graph().ainvoke(state)

    assert final_state["blocking_issues"] == []
    assert final_state["gate_passed"] is True


@pytest.mark.parametrize(
    "check",
    [
        ld.check_perfect_class_separation,
        ld.check_zero_variance_within_class,
        ld.check_feature_target_logical_dependency,
        ld.check_single_feature_auc,
        ld.check_categorical_class_separation,
    ],
)
def test_every_per_feature_check_records_its_crash(check: Any) -> None:
    """Function-level sweep over the per-feature ``except`` branches: a feature
    name that is not a column raises ``KeyError`` inside each check's loop.
    Every one must report it through ``audit_errors``."""
    frame = _noise_frame()
    audit_errors: list = []

    check(frame, "target", ["not_a_column"], audit_errors=audit_errors)

    assert len(audit_errors) == 1, audit_errors
    assert "'not_a_column'" in audit_errors[0] and "KeyError" in audit_errors[0]


@pytest.mark.asyncio
async def test_regression_target_is_not_an_incomplete_audit() -> None:
    """Measured while fixing item 3: ``mutual_info_classif`` raised "Unknown
    label type: continuous" on EVERY regression-target run. Recording every
    swallowed crash would have blocked all of them. The MI check does not
    APPLY to a continuous target; that is not an incomplete audit."""
    frame = _noise_frame()
    frame["target"] = frame["noise_b"] * 3.0 + 1.0

    final_state = await _leakage_gate_graph().ainvoke(_leakage_state(frame))

    assert final_state["blocking_issues"] == [], final_state["blocking_issues"]
    assert final_state["gate_passed"] is True


# =============================================================================
# Item 4 — leakage remediation's free-text retraction + the unrechecked 5th pass
# =============================================================================

_ANALYZE_LEAKAGE_LLM = (
    "src.agents.ml_foundation.data_preparer.nodes.leakage_remediation._analyze_leakage_with_llm"
)


def _remediation_loop_graph(adaptive_node: Any):
    """The REAL leakage loop from ``graph.py``, closed onto the gate.

    ``detect_leakage -> adaptive_validity_check --_route_after_leakage_detection-->
    leakage_remediation --_route_after_leakage_remediation--> {detect_leakage |
    finalize_output}``, with the production routing functions and the real
    adaptive node (only its Layer-4 LLM loader stubbed — see
    ``_no_layer_4_llm``). The transform..sufficiency chain is omitted: none of
    it reads or writes ``leakage:`` entries, and several nodes need external
    services.
    """
    graph = StateGraph(DataPreparerState)
    graph.add_node("detect_leakage", detect_leakage)  # type: ignore[arg-type]
    graph.add_node("adaptive_validity_check", adaptive_node)  # type: ignore[arg-type]
    graph.add_node("leakage_remediation", review_and_remediate_leakage)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point("detect_leakage")
    graph.add_edge("detect_leakage", "adaptive_validity_check")
    graph.add_conditional_edges(
        "adaptive_validity_check",
        _route_after_leakage_detection,
        {"remediate": "leakage_remediation", "continue": "finalize_output"},
    )
    graph.add_conditional_edges(
        "leakage_remediation",
        _route_after_leakage_remediation,
        {"recheck": "detect_leakage", "continue": "finalize_output", "end": END},
    )
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _kept_leak_frame(n: int = 240) -> pd.DataFrame:
    """``age`` leaks at HIGH (single-feature AUC ~0.85, not auto-droppable);
    ``leak_b`` is a near copy of the target; three clean noise features."""
    rng = np.random.default_rng(22944)
    target = np.array([0, 1] * (n // 2))
    return pd.DataFrame(
        {
            "age": target * 1.5 + rng.standard_normal(n),
            "leak_b": target * 3.0 + rng.normal(0.0, 0.05, n),
            "clean_a": rng.standard_normal(n),
            "clean_b": rng.standard_normal(n),
            "clean_c": rng.standard_normal(n),
            "target": target,
        }
    )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_final_remediation_pass_is_rechecked_before_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The remediation node retracted the ``leakage:`` entry of EVERY leaked
    feature — including one it did not drop — and relied on the recheck to
    rebuild what was still true. ``_route_after_leakage_remediation`` skipped
    that recheck once ``attempts`` reached the max, and the node sets
    ``attempts`` to 5 before routing, so the 5th successful pass went straight
    on with the over-retracted channel: a still-present HIGH leak passed.

    The LLM (an external service) is the only thing patched: its analysis
    drops ``leak_b`` and keeps ``age`` as a "legitimate predictor".
    """
    from unittest.mock import AsyncMock, patch

    # Isolate the on-disk analysis cache so a cached analysis cannot stand in
    # for the patched one.
    monkeypatch.setenv("E2I_CACHE_DIR", str(tmp_path))
    analysis = {
        "leakage_classifications": {"leak_b": "tautological", "age": "legitimate"},
        "features_to_drop": ["leak_b"],
        "replacement_candidates": [],
        "recommended_feature_set": ["clean_a", "clean_b", "clean_c"],
        "viable": True,
        "reasoning": "leak_b encodes the label; age is a legitimate predictor",
    }
    state = {
        "experiment_id": "exp-2294-final-pass",
        "train_df": _kept_leak_frame(),
        "scope_spec": {"prediction_target": "target"},
        # Four passes already spent: the next successful one is the last.
        "leakage_remediation_attempts": 4,
        **_CLEAN_UPSTREAM_QC,
    }

    with patch(_ANALYZE_LEAKAGE_LLM, new=AsyncMock(return_value=analysis)):
        final_state = await _remediation_loop_graph(_no_layer_4_llm(monkeypatch)).ainvoke(state)

    assert final_state["leakage_remediation_status"] == "applied"
    assert final_state["leakage_remediation_attempts"] == 5
    assert "leak_b" not in final_state["train_df"].columns
    assert "age" in final_state["train_df"].columns  # kept: still leaking

    blocking = final_state["blocking_issues"] or []
    assert any(i.startswith("leakage: ") and "'age'" in i for i in blocking), (
        f"the kept HIGH leak on 'age' was retracted and never rebuilt: {blocking!r}"
    )
    # What remediation genuinely fixed is gone — rebuilt from the recheck.
    assert not any("leak_b" in i for i in blocking), blocking
    assert final_state["gate_passed"] is False


# =============================================================================
# Item 5 — sufficiency_check and feast_registrar onto merge_blocking_issues
# =============================================================================


def _twice_then_gate(node: Any, name: str):
    """``node -> node -> finalize_output``: the same function wired twice,
    standing in for the QC retry edge (``qc_remediation --retry-->
    run_quality_checks`` re-runs the whole downstream chain, these nodes
    included) without dragging qc_remediation's LLM call into a unit test."""
    graph = StateGraph(DataPreparerState)
    graph.add_node(f"{name}_1", node)  # type: ignore[arg-type]
    graph.add_node(f"{name}_2", node)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point(f"{name}_1")
    graph.add_edge(f"{name}_1", f"{name}_2")
    graph.add_edge(f"{name}_2", "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _single_then_gate(node: Any, name: str):
    graph = StateGraph(DataPreparerState)
    graph.add_node(name, node)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point(name)
    graph.add_edge(name, "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _sufficiency_state(n: int) -> Dict[str, Any]:
    rng = np.random.default_rng(22945)
    y = np.zeros(n, dtype=int)
    y[: int(round(n * 0.3))] = 1
    rng.shuffle(y)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(10)})
    frame["y"] = y
    return {
        "experiment_id": "exp-2294-sufficiency",
        "train_df": frame,
        "target_rate": 0.30,
        "scope_spec": {"problem_type": "binary_classification", "prediction_target": "y"},
        "blocking_issues": ["sampling_frame_drift: an unrelated upstream blocker"],
        **_CLEAN_UPSTREAM_QC,
    }


@pytest.mark.asyncio
async def test_sufficiency_entry_is_not_duplicated_on_a_qc_retry() -> None:
    """HARD_FAIL (n=30, below the absolute floor) on both passes: one entry."""
    final_state = await _twice_then_gate(run_sufficiency_check, "sufficiency_check").ainvoke(
        _sufficiency_state(30)
    )

    blocking = final_state["blocking_issues"]
    assert final_state["sufficiency_report"]["verdict"] == "HARD_FAIL"
    assert sum(i.startswith("data_sufficiency: ") for i in blocking) == 1, blocking
    assert blocking.count("sampling_frame_drift: an unrelated upstream blocker") == 1
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_resolved_sufficiency_entry_is_retracted() -> None:
    """A previous pass's HARD_FAIL entry must go once the verdict is PASS —
    otherwise a QC retry that fixed the data could never clear the gate."""
    state = _sufficiency_state(5000)
    state["blocking_issues"] = [
        "data_sufficiency: HARD_FAIL (n=30 below absolute floor). earlier pass",
        "sampling_frame_drift: an unrelated upstream blocker",
    ]

    final_state = await _single_then_gate(run_sufficiency_check, "sufficiency_check").ainvoke(state)

    assert final_state["sufficiency_report"]["verdict"] == "PASS"
    assert final_state["blocking_issues"] == ["sampling_frame_drift: an unrelated upstream blocker"]


def _feast_state() -> Dict[str, Any]:
    return {
        "experiment_id": "exp-2294-feast",
        "train_df": pd.DataFrame(
            {"hcp_id": ["h1", "h2", "h3"], "feature1": np.arange(3.0), "target": [0, 1, 0]}
        ),
        # A table that backs Feast views, on a run that trains on Feast-served
        # features: the one configuration in which the freshness gate blocks.
        "data_source": "triggers",
        "features_served_by_feast": True,
        "scope_spec": {
            "required_features": ["feature1"],
            "entity_key": "hcp_id",
            "prediction_target": "target",
        },
        "blocking_issues": ["sampling_frame_drift: an unrelated upstream blocker"],
        **_CLEAN_UPSTREAM_QC,
    }


def _feast_seams(monkeypatch: pytest.MonkeyPatch, *, recency_age: Any) -> None:
    """The two external seams ``test_feast_registrar_source_gate_2207.py``
    already patches: the Feast adapter (registration talks to Feast) and the
    #559 recency query (reads Supabase). The node and its gate logic are real.
    """
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    from src.agents.ml_foundation.data_preparer.nodes import feast_registrar

    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={"features_registered": 1, "errors": []}
    )
    adapter._feast_client = None
    monkeypatch.setattr(feast_registrar, "_get_feature_analyzer_adapter", lambda: adapter)

    async def _recency(_table: str) -> Any:
        return datetime.now(timezone.utc) - recency_age

    monkeypatch.setattr(feast_registrar, "_source_recency_query", _recency)
    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)


@pytest.mark.asyncio
async def test_feast_entry_is_not_duplicated_on_a_qc_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import timedelta

    _feast_seams(monkeypatch, recency_age=timedelta(days=40))

    final_state = await _twice_then_gate(
        register_features_in_feast, "register_features_in_feast"
    ).ainvoke(_feast_state())

    blocking = final_state["blocking_issues"]
    assert final_state["feast_blocked"] is True
    assert sum("Feast features stale" in i for i in blocking) == 1, blocking
    assert all(
        i.startswith("feast_freshness: ") for i in blocking if "Feast features stale" in i
    ), blocking
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_resolved_feast_entry_is_retracted(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    _feast_seams(monkeypatch, recency_age=timedelta(hours=2))
    state = _feast_state()
    state["blocking_issues"] = [
        "feast_freshness: Feast features stale; ALLOW_STALE_FEAST not set",
        "sampling_frame_drift: an unrelated upstream blocker",
    ]

    final_state = await _single_then_gate(
        register_features_in_feast, "register_features_in_feast"
    ).ainvoke(state)

    assert final_state["feast_blocked"] is False
    assert final_state["blocking_issues"] == ["sampling_frame_drift: an unrelated upstream blocker"]


# =============================================================================
# Codex r1 (HIGH) — declared-safe manifest immunity vs. the detector's blocker
# =============================================================================

_ADAPTIVE_MODULE = "src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check"


def _no_layer_4_llm(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return the REAL adaptive node with only its Layer-4 LLM loader stubbed.

    The loader configures a paid LLM when an API key is in the environment;
    ``None`` is its own documented "no LM configured" result, so Layer 3, the
    manifest immunity and the severity merge all run for real.
    """
    import importlib

    module = importlib.import_module(_ADAPTIVE_MODULE)
    monkeypatch.setattr(module, "_try_load_layer_4_classifier", lambda: None)
    return module.adaptive_validity_check


def _detect_adaptive_gate_graph(adaptive_node: Any):
    """``detect_leakage -> adaptive_validity_check -> finalize_output`` with the
    production routing; the remediation branch ends the graph (not under test)."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("detect_leakage", detect_leakage)  # type: ignore[arg-type]
    graph.add_node("adaptive_validity_check", adaptive_node)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point("detect_leakage")
    graph.add_edge("detect_leakage", "adaptive_validity_check")
    graph.add_conditional_edges(
        "adaptive_validity_check",
        _route_after_leakage_detection,
        {"remediate": END, "continue": "finalize_output"},
    )
    graph.add_edge("finalize_output", END)
    return graph.compile()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_declared_safe_feature_does_not_block_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``lab_claim_count`` is declared pre-index in the CSU manifest. Full
    manifest immunity (owner decision 2026-06-03, ``8f0eb68be``) exempts it
    from leakage: ``adaptive_validity_check`` strips its finding and downgrades
    the severity. But ``detect_leakage`` had already written a ``leakage:``
    blocker for the same finding, and nothing retracted it — so a
    contract-certified feature failed the gate anyway. Pre-existing and
    fail-CLOSED; found by codex r1 because the loop tests had omitted the
    adaptive node. The detector owns its entries, so it applies the contract.
    """
    adaptive = _no_layer_4_llm(monkeypatch)
    rng = np.random.default_rng(22946)
    n = 400
    target = np.array([0, 1] * (n // 2))
    frame = pd.DataFrame(
        {
            # single-feature AUC ~0.86 -> a HIGH structured finding.
            "lab_claim_count": target * 1.5 + rng.standard_normal(n),
            "noise": rng.standard_normal(n),
            "target": target,
        }
    )
    state = _leakage_state(frame, scope_spec={"feature_manifest_source": "csu"})

    final_state = await _detect_adaptive_gate_graph(adaptive).ainvoke(state)

    # Precondition: the immunity really fired downstream.
    assert final_state["leakage_severity"] not in ("critical", "high")
    assert final_state["blocking_issues"] == [], final_state["blocking_issues"]
    assert final_state["gate_passed"] is True


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_the_same_finding_still_blocks_without_a_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: immunity is opt-in per cohort. Without a manifest the same
    HIGH finding keeps blocking (and routes to remediation)."""
    adaptive = _no_layer_4_llm(monkeypatch)
    rng = np.random.default_rng(22946)
    n = 400
    target = np.array([0, 1] * (n // 2))
    frame = pd.DataFrame(
        {
            "lab_claim_count": target * 1.5 + rng.standard_normal(n),
            "noise": rng.standard_normal(n),
            "target": target,
        }
    )

    final_state = await _detect_adaptive_gate_graph(adaptive).ainvoke(_leakage_state(frame))

    assert any("lab_claim_count" in i for i in final_state["blocking_issues"])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["adapter_unavailable", "registration_raises"])
async def test_feast_served_run_blocks_when_freshness_is_never_checked(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Codex r1 (MEDIUM): the freshness probe runs only after the adapter
    initialises and registration succeeds. On a Feast-SERVED run, an
    unavailable adapter or a registration exception returned before the probe,
    so freshness was never verified and — on a first pass, with no earlier
    entry to preserve — nothing blocked. Unverifiable is not fresh (#556)."""
    from datetime import timedelta
    from unittest.mock import AsyncMock

    from src.agents.ml_foundation.data_preparer.nodes import feast_registrar

    _feast_seams(monkeypatch, recency_age=timedelta(hours=2))
    if failure == "adapter_unavailable":
        monkeypatch.setattr(feast_registrar, "_get_feature_analyzer_adapter", lambda: None)
    else:
        adapter = feast_registrar._get_feature_analyzer_adapter()
        adapter.register_features_from_state = AsyncMock(side_effect=RuntimeError("feast down"))

    final_state = await _single_then_gate(
        register_features_in_feast, "register_features_in_feast"
    ).ainvoke(_feast_state())

    blocking = final_state["blocking_issues"]
    assert any(i.startswith("feast_freshness: Feast features unverifiable") for i in blocking), (
        blocking
    )
    assert final_state["feast_blocked"] is True
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_non_feast_served_run_is_not_blocked_by_an_unavailable_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: every run today (nothing sets ``features_served_by_feast``)
    keeps the documented non-blocking behaviour."""
    from datetime import timedelta

    from src.agents.ml_foundation.data_preparer.nodes import feast_registrar

    _feast_seams(monkeypatch, recency_age=timedelta(hours=2))
    monkeypatch.setattr(feast_registrar, "_get_feature_analyzer_adapter", lambda: None)
    state = _feast_state()
    state["features_served_by_feast"] = False

    final_state = await _single_then_gate(
        register_features_in_feast, "register_features_in_feast"
    ).ainvoke(state)

    assert final_state["blocking_issues"] == ["sampling_frame_drift: an unrelated upstream blocker"]
