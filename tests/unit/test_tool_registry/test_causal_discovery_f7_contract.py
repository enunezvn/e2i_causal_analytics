"""F7 — unified data contract for ``discover_dag`` / ``rank_drivers``.

Before F7 the Tool Composer had THREE incompatible "data" contracts: the 7
causal tools read a real ``pandas.DataFrame`` from kwargs (canonical keys
``_DATAFRAME_KWARGS_KEYS``) and the executor auto-injected
``context["estimation_data"]`` for them, but ``discover_dag`` / ``rank_drivers``
wanted a ``Dict[str, List]`` and the auto-inject did NOT serve that contract.
A plan that chained ``discover_dag`` failed with a pydantic ValidationError.

F7 makes ``discover_dag`` / ``rank_drivers`` ALSO accept the real DataFrame via
the standard ``estimation_data`` kwarg, converting internally — while preserving
back-compat for an explicit ``data: Dict``. These tests exercise the REAL PC /
SHAP engines (NO mocks of the tool logic) on small real DataFrames, and pin the
fail-closed honesty (descriptive ``RuntimeError`` when neither a frame nor a
valid dict is supplied — never fabricate).

Falsifiability: each test enumerates an exact regression path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tool_registry.tools.causal_discovery import (
    _frame_to_numeric_dict,
    _is_valid_data_dict,
    _normalize_edge_list,
    _numeric_frame,
    discover_dag,
    rank_drivers,
)


def _linear_chain_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """A small REAL frame with a planted dependency a -> b -> c (+ noise d).

    Not a mock: real numeric data the PC / SHAP engines run over.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = 2.0 * a + rng.normal(0, 0.3, n)
    c = 1.5 * b + rng.normal(0, 0.3, n)
    d = rng.normal(0, 1, n)  # independent noise
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


# ---------------------------------------------------------------------------
# Helper-level pins
# ---------------------------------------------------------------------------
def test_is_valid_data_dict() -> None:
    assert _is_valid_data_dict({"x": [1, 2], "y": [3, 4]}) is True
    assert _is_valid_data_dict({"x": (1, 2)}) is True
    # planner's column->reference-string dict is NOT a valid discovery dict
    assert _is_valid_data_dict({"x": "$step.x"}) is False
    assert _is_valid_data_dict({}) is False
    assert _is_valid_data_dict(None) is False
    assert _is_valid_data_dict(pd.DataFrame({"x": [1]})) is False


def test_frame_to_numeric_dict_drops_nonnumeric_and_nan() -> None:
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["x", "y", "z"], "withnan": [1.0, None, 3.0]})
    out = _frame_to_numeric_dict(df)
    assert "cat" not in out, "non-numeric column must be dropped"
    assert "num" in out
    # rows with any NaN dropped -> 'num' keeps rows 0 and 2 (where 'withnan' is non-null)
    assert len(out["num"]) == 2


def test_frame_to_numeric_dict_fails_closed_on_no_numeric() -> None:
    df = pd.DataFrame({"cat": ["x", "y"], "cat2": ["a", "b"]})
    with pytest.raises(RuntimeError, match="no numeric columns"):
        _frame_to_numeric_dict(df)


def test_numeric_frame_drops_sparse_columns_before_complete_case() -> None:
    """Regression for the real-cohort integration bug: a few mostly-null columns
    must NOT annihilate the cohort.

    On the real 52-col Kisqali frame, requiring complete rows across ALL numeric
    columns (incl. ~94%-null adherence_rate / refill_count) dropped EVERY row ->
    "cannot run on an empty frame". The dense signal columns must survive; the
    sparse ones are dropped (with a WARNING) before the complete-case filter.
    """
    n = 100
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "dense_a": rng.normal(0, 1, n),
            "dense_b": rng.normal(0, 1, n),
            "dense_c": rng.normal(0, 1, n),
            # 10% populated -> below the 0.5 non-null threshold -> dropped.
            "sparse": [1.0 if i < 10 else None for i in range(n)],
        }
    )
    out = _numeric_frame(df)
    assert "sparse" not in out.columns
    assert set(out.columns) == {"dense_a", "dense_b", "dense_c"}
    # NOT reduced to ~10 rows by the sparse column's nulls.
    assert len(out) == n


def test_numeric_frame_fails_closed_when_too_few_dense_columns() -> None:
    """If dropping sparse columns leaves <2 dense numeric columns, fail closed
    with a descriptive error rather than discovering structure on one variable."""
    n = 100
    df = pd.DataFrame(
        {
            "dense": list(range(n)),
            "sparse1": [1.0 if i < 5 else None for i in range(n)],
            "sparse2": [2.0 if i < 5 else None for i in range(n)],
        }
    )
    with pytest.raises(RuntimeError, match="fewer than 2"):
        _numeric_frame(df)


def test_normalize_edge_list_strips_extra_keys() -> None:
    edges = [
        {"source": "a", "target": "b", "confidence": 1.0, "type": "directed", "algorithms": ["pc"]},
        {"source": "b", "target": "c", "confidence": 0.8},
        {"bad": "edge"},  # missing source/target -> skipped
    ]
    out = _normalize_edge_list(edges)
    assert out == [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
    assert all(set(e.keys()) == {"source", "target"} for e in out)


# ---------------------------------------------------------------------------
# discover_dag — unified contract (real PC engine, no mocks)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_discover_dag_accepts_estimation_data_dataframe() -> None:
    """F7: discover_dag consumes a real DataFrame via the 'estimation_data'
    kwarg (the key the executor auto-injects), converting internally to the
    Dict shape. Before F7 the only accepted shape was a Dict and the planner's
    chain failed.
    """
    df = _linear_chain_frame()
    res = await discover_dag(estimation_data=df, algorithms=["pc"], alpha=0.1)
    assert res["success"] is True
    assert res["n_nodes"] >= 2
    assert res["n_edges"] >= 1, "PC must find at least one edge in the planted chain"


@pytest.mark.asyncio
async def test_discover_dag_back_compat_explicit_dict() -> None:
    """F7: the legacy explicit ``data: Dict[str, List]`` contract still works."""
    df = _linear_chain_frame()
    data_dict = df.to_dict("list")
    res = await discover_dag(data=data_dict, algorithms=["pc"], alpha=0.1)
    assert res["success"] is True
    assert res["n_edges"] >= 1


@pytest.mark.asyncio
async def test_discover_dag_dataframe_in_data_kwarg() -> None:
    """F7: an explicit DataFrame supplied directly as ``data`` is converted too
    (caller-explicit DataFrame path).
    """
    df = _linear_chain_frame()
    res = await discover_dag(data=df, algorithms=["pc"], alpha=0.1)
    assert res["success"] is True


@pytest.mark.asyncio
async def test_discover_dag_fails_closed_without_data() -> None:
    """F7 anti-mocking: no frame and no valid dict -> descriptive RuntimeError,
    NOT a fabricated frame.
    """
    with pytest.raises(RuntimeError, match="requires a real DataFrame"):
        await discover_dag(algorithms=["pc"])


@pytest.mark.asyncio
async def test_discover_dag_ignores_broken_planner_data_dict() -> None:
    """F7: when ``data`` is the planner's column->ref dict (invalid) BUT a real
    frame is threaded under ``estimation_data`` (executor auto-inject), the real
    frame is used and discovery succeeds — no ValidationError.
    """
    df = _linear_chain_frame()
    broken = {"a": "a", "b": "$step_0.b"}  # column->reference strings
    res = await discover_dag(data=broken, estimation_data=df, algorithms=["pc"], alpha=0.1)
    assert res["success"] is True
    assert res["n_edges"] >= 1


# ---------------------------------------------------------------------------
# rank_drivers — unified contract (real SHAP engine, no mocks)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rank_drivers_derives_shap_from_estimation_data() -> None:
    """F7: rank_drivers consumes the SAME real frame as discover_dag. With NO
    explicit shap_values/feature_names, it derives REAL SHAP from
    ``(features, target)`` on the injected frame.
    """
    df = _linear_chain_frame()
    # DAG over a,b,c (target=c); features a,b are connected to c
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
    res = await rank_drivers(
        dag_edge_list=edges,
        target="c",
        feature_names=["a", "b"],
        estimation_data=df,
    )
    assert res["success"] is True, res.get("errors")
    assert res["n_features"] == 2
    names = {r["feature_name"] for r in res["rankings"]}
    assert names == {"a", "b"}


@pytest.mark.asyncio
async def test_rank_drivers_normalizes_discover_dag_edge_list() -> None:
    """F7: the chain ``discover_dag.edge_list -> rank_drivers.dag_edge_list``
    works even though discover_dag edges carry extra confidence/type/algorithms
    keys (Dict[str, str] schema would otherwise reject the float/list values).
    """
    df = _linear_chain_frame()
    discovery = await discover_dag(estimation_data=df, algorithms=["pc"], alpha=0.1)
    edge_list = discovery["edge_list"]
    assert edge_list, "need at least one discovered edge for the chain"
    # extra keys present on the raw edges
    assert "confidence" in edge_list[0]
    # pick a connected node as target and its neighbours as features
    nodes = sorted({e["source"] for e in edge_list} | {e["target"] for e in edge_list})
    target = nodes[-1]
    feats = [n for n in nodes if n != target]
    res = await rank_drivers(
        dag_edge_list=edge_list,  # raw discover_dag output, extra keys and all
        target=target,
        feature_names=feats,
        estimation_data=df,
    )
    # No pydantic ValidationError on dag_edge_list; the chain flows.
    assert "validation error" not in (str(res.get("errors")) or "").lower()


@pytest.mark.asyncio
async def test_rank_drivers_explicit_shap_wins() -> None:
    """F7: explicit shap_values + feature_names are caller-explicit and used
    as-is (no frame needed).
    """
    edges = [{"source": "a", "target": "b"}]
    shap = [[0.5, 0.1], [0.4, 0.2], [0.6, 0.05]]
    res = await rank_drivers(
        dag_edge_list=edges,
        target="b",
        shap_values=shap,
        feature_names=["a", "x"],
    )
    # 'a' is in the DAG; 'x' is isolated -> ranker may exclude it, but the call
    # must not require a frame and must not raise.
    assert res["target_variable"] == "b"


@pytest.mark.asyncio
async def test_rank_drivers_fails_closed_without_shap_or_frame() -> None:
    """F7 anti-mocking: no explicit SHAP and no frame -> descriptive
    RuntimeError, NOT fabricated SHAP values.
    """
    edges = [{"source": "a", "target": "b"}]
    with pytest.raises(RuntimeError, match="requires either explicit"):
        await rank_drivers(dag_edge_list=edges, target="b")


@pytest.mark.asyncio
async def test_rank_drivers_fails_closed_when_target_absent_from_frame() -> None:
    """F7 anti-mocking: target not a numeric column of the frame -> RuntimeError
    (cannot honestly compute predictive importance).
    """
    df = _linear_chain_frame()
    edges = [{"source": "a", "target": "b"}]
    with pytest.raises(RuntimeError, match="is not a numeric column"):
        await rank_drivers(
            dag_edge_list=edges,
            target="nonexistent_target",
            estimation_data=df,
        )
