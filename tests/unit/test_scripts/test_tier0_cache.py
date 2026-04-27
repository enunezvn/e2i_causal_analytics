"""Tests for the Tier-0 cache contract — Block 4 / Finding #12.

The tier0 cache (``scripts/tier0_output_cache/latest.pkl``) must persist
the entity → split label mapping so that downstream consumers (tier1-5
agent tests) cannot accidentally re-derive splits and break train/val/test
isolation. These tests cover three strands of the contract:

1. ``run_tier0_test.step_5_model_trainer`` returns ``split_assignments``
   alongside the trained model (assignments mapping is dict-typed and
   contains at minimum train/val/test labels when entity_ids are passed).

2. ``run_tier0_test.step_5_model_trainer`` honours ``pre_assigned_splits``
   when provided — the function reuses the cached mapping verbatim
   instead of re-running the splitter.

3. ``run_tier1_5_test.load_tier0_state`` round-trips the
   ``split_assignments`` field through pickle and surfaces a notice on
   reload so downstream callers can react.
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIER0_SCRIPT = PROJECT_ROOT / "scripts" / "run_tier0_test.py"
TIER1_5_SCRIPT = PROJECT_ROOT / "scripts" / "run_tier1_5_test.py"


def _load_script_module(name: str, path: Path):
    """Helper: load a script-as-module, registering it in ``sys.modules``
    BEFORE executing the module body.

    The pre-registration is required so that any ``@dataclass`` declared
    in the script can resolve ``cls.__module__`` to a real module entry
    in ``sys.modules`` (Python 3.12 dataclass machinery walks
    ``sys.modules`` to inspect class type hints — fails otherwise).
    """
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Tear down the registration on failure so subsequent attempts
        # see a clean slate.
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def tier0_module():
    """Import ``run_tier0_test`` as a module without relying on the CLI.

    Skips if the heavy ML deps cannot be imported in the current env (so
    this test file remains usable in stripped-down CI shards).
    """
    try:
        module = _load_script_module("run_tier0_test", TIER0_SCRIPT)
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"Could not import run_tier0_test: {exc}")
    if module is None:
        pytest.skip("Could not build import spec for run_tier0_test")
    return module


@pytest.fixture(scope="module")
def tier1_5_module():
    """Import ``run_tier1_5_test`` for the load_tier0_state round-trip
    tests."""
    try:
        module = _load_script_module("run_tier1_5_test", TIER1_5_SCRIPT)
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"Could not import run_tier1_5_test: {exc}")
    if module is None:
        pytest.skip("Could not build import spec for run_tier1_5_test")
    return module


def _toy_dataset(n_entities: int = 80) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build a small entity+date+feature dataset for the splitter tests."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2026-01-01")
    rows = []
    for i in range(n_entities):
        rows.append(
            {
                "patient_journey_id": f"patient-{i:03d}",
                "journey_start_date": start + pd.Timedelta(days=i),
                "feature_a": rng.normal(0, 1),
                "feature_b": rng.normal(0, 1),
                "y": int(rng.random() < 0.5),
            }
        )
    df = pd.DataFrame(rows)
    X = df[["feature_a", "feature_b"]]
    y = df["y"]
    entity_ids = df["patient_journey_id"]
    dates = df["journey_start_date"]
    return X, y, entity_ids, dates


def test_step_5_model_trainer_signature_includes_block4_kwargs(tier0_module):
    """The function must accept ``entity_ids``, ``dates``, ``split_mode``,
    and ``pre_assigned_splits`` keyword arguments."""
    import inspect

    sig = inspect.signature(tier0_module.step_5_model_trainer)
    params = sig.parameters
    assert "entity_ids" in params
    assert "dates" in params
    assert "split_mode" in params
    assert "pre_assigned_splits" in params
    assert params["split_mode"].default == "auto"
    assert params["pre_assigned_splits"].default is None


def test_run_pipeline_signature_includes_regime_and_split_mode(tier0_module):
    """The plan added ``regime`` and ``split_mode`` parameters to
    ``run_pipeline``; cache reload also threads ``pre_assigned_splits``."""
    import inspect

    sig = inspect.signature(tier0_module.run_pipeline)
    params = sig.parameters
    assert "regime" in params
    assert params["regime"].default == "default"
    assert "split_mode" in params
    assert params["split_mode"].default == "auto"
    assert "pre_assigned_splits" in params
    assert params["pre_assigned_splits"].default is None


def test_split_assignments_round_trip_through_pickle(tmp_path):
    """The cache pickle preserves ``split_assignments`` and
    ``split_strategy`` verbatim across save/load."""
    state = {
        "experiment_id": "tier0_e2e_test",
        "split_assignments": {
            "patient-001": "train",
            "patient-002": "val",
            "patient-003": "test",
            "patient-004": "holdout",
        },
        "split_strategy": "combined_temporal_entity_with_holdout",
        "trained_model": None,
    }
    cache_file = tmp_path / "tier0_state.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(state, f)
    with open(cache_file, "rb") as f:
        loaded = pickle.load(f)
    assert loaded["split_assignments"] == state["split_assignments"]
    assert loaded["split_strategy"] == state["split_strategy"]


def test_load_tier0_state_surfaces_assignments(
    tier1_5_module, tmp_path, capsys
):
    """``load_tier0_state`` prints a notice when the cache contains
    ``split_assignments``, signalling that downstream consumers MUST NOT
    re-derive splits."""
    state = {
        "experiment_id": "tier0_e2e_test",
        "split_assignments": {f"patient-{i:03d}": "train" for i in range(10)},
        "split_strategy": "combined_temporal_entity_with_holdout",
    }
    cache_file = tmp_path / "tier0_state.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(state, f)
    loaded = tier1_5_module.load_tier0_state(str(cache_file))
    captured = capsys.readouterr()
    assert loaded["split_assignments"] == state["split_assignments"]
    # The function should print a notice — exact wording is allowed to
    # drift, but it must signal that re-derivation is forbidden.
    assert "split_assignments" in captured.out or "REUSE" in captured.out
    assert "Block 4" in captured.out or "forbidden" in captured.out.lower()


def test_load_tier0_state_warns_when_assignments_absent(
    tier1_5_module, tmp_path, capsys
):
    """Older caches without ``split_assignments`` should still load but
    print a warning telling the operator to re-run tier0."""
    state = {"experiment_id": "tier0_e2e_test"}  # no split_assignments
    cache_file = tmp_path / "tier0_state.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(state, f)
    loaded = tier1_5_module.load_tier0_state(str(cache_file))
    captured = capsys.readouterr()
    assert loaded == state
    assert (
        "no split_assignments" in captured.out
        or "older cache" in captured.out.lower()
    )


@pytest.mark.asyncio
async def test_step_5_pre_assigned_splits_refuses_to_re_split(tier0_module):
    """When the caller supplies ``pre_assigned_splits``, the function must
    reuse the mapping and refuse to invoke the splitter — even when entity
    + date columns are available."""
    X, y, entity_ids, dates = _toy_dataset(n_entities=40)
    # Build a valid pre-assignment covering every entity.
    pre_assignments: dict[str, str] = {}
    for idx, eid in enumerate(entity_ids):
        if idx % 5 == 0:
            pre_assignments[eid] = "test"
        elif idx % 5 == 1:
            pre_assignments[eid] = "val"
        elif idx % 5 == 2:
            pre_assignments[eid] = "holdout"
        else:
            pre_assignments[eid] = "train"

    # We can't easily call the full ``step_5_model_trainer`` (it spins up
    # the agent and trains a model). Instead we exercise the split-resolution
    # logic in isolation by patching the agent run with an inert stub.
    class _StubAgent:
        async def run(self, payload):
            return {
                "trained_model": None,
                "validation_metrics": {"auc_roc": 0.5},
                "train_metrics": {},
                "test_metrics": {},
            }

    # Use a manual swap (pytest's monkeypatch fixture isn't available
    # in plain async tests outside a fixture scope; manual swap suffices).
    import src.agents.ml_foundation.model_trainer as mt_pkg

    original_agent = getattr(mt_pkg, "ModelTrainerAgent", None)
    mt_pkg.ModelTrainerAgent = _StubAgent  # type: ignore[attr-defined,misc]
    try:
        result = await tier0_module.step_5_model_trainer(
            "test_exp",
            {"algorithm_name": "LogisticRegression"},
            {"qc_passed": True},
            X,
            y,
            success_criteria={},
            entity_ids=entity_ids,
            dates=dates,
            split_mode="auto",
            pre_assigned_splits=pre_assignments,
        )
    finally:
        if original_agent is not None:
            mt_pkg.ModelTrainerAgent = original_agent  # type: ignore[attr-defined,misc]

    # The returned mapping must equal the input mapping verbatim — the
    # function must NOT re-derive splits when assignments are supplied.
    assert result["split_assignments"] == pre_assignments
    assert result["split_strategy"] == "cached_replay"


@pytest.mark.asyncio
async def test_step_5_pre_assigned_splits_errors_on_missing_entity(tier0_module):
    """If the supplied mapping is missing an entity present in the data,
    the function must error out rather than silently re-splitting."""
    X, y, entity_ids, dates = _toy_dataset(n_entities=20)
    # Drop the last entity from the assignment map.
    pre_assignments: dict[str, str] = {}
    for idx, eid in enumerate(entity_ids[:-1]):
        pre_assignments[eid] = "train" if idx < 10 else "val"

    class _StubAgent:
        async def run(self, payload):  # pragma: no cover - never called
            return {}

    import src.agents.ml_foundation.model_trainer as mt_pkg

    original_agent = getattr(mt_pkg, "ModelTrainerAgent", None)
    mt_pkg.ModelTrainerAgent = _StubAgent  # type: ignore[attr-defined,misc]
    try:
        with pytest.raises(ValueError, match="missing"):
            await tier0_module.step_5_model_trainer(
                "test_exp",
                {"algorithm_name": "LogisticRegression"},
                {"qc_passed": True},
                X,
                y,
                success_criteria={},
                entity_ids=entity_ids,
                dates=dates,
                split_mode="auto",
                pre_assigned_splits=pre_assignments,
            )
    finally:
        if original_agent is not None:
            mt_pkg.ModelTrainerAgent = original_agent  # type: ignore[attr-defined,misc]


def test_step_5_split_mode_validation(tier0_module):
    """``split_mode`` must reject unknown values via a clear error before
    any heavy work happens."""
    import asyncio

    X, y, entity_ids, dates = _toy_dataset(n_entities=10)

    async def _run():
        return await tier0_module.step_5_model_trainer(
            "test_exp",
            {"algorithm_name": "LogisticRegression"},
            {"qc_passed": True},
            X,
            y,
            success_criteria={},
            entity_ids=entity_ids,
            dates=dates,
            split_mode="not_a_real_mode",
        )

    with pytest.raises(ValueError, match="split_mode"):
        asyncio.run(_run())


@pytest.mark.asyncio
async def test_step_5_pre_assigned_splits_rejects_unknown_labels(tier0_module):
    """Block 4 4-IMP-1: ``pre_assigned_splits`` must reject unknown
    label values (e.g. typo ``"trian"``) loudly, BEFORE training, so a
    cache-corruption silently producing empty splits is impossible."""
    X, y, entity_ids, dates = _toy_dataset(n_entities=20)
    # Inject a typo into one entity's label.
    pre_assignments: dict[str, str] = {}
    for idx, eid in enumerate(entity_ids):
        if idx == 0:
            pre_assignments[eid] = "trian"  # deliberate typo
        elif idx % 4 == 0:
            pre_assignments[eid] = "test"
        elif idx % 4 == 1:
            pre_assignments[eid] = "val"
        elif idx % 4 == 2:
            pre_assignments[eid] = "holdout"
        else:
            pre_assignments[eid] = "train"

    class _StubAgent:
        async def run(self, payload):  # pragma: no cover - never called
            return {}

    import src.agents.ml_foundation.model_trainer as mt_pkg

    original_agent = getattr(mt_pkg, "ModelTrainerAgent", None)
    mt_pkg.ModelTrainerAgent = _StubAgent  # type: ignore[attr-defined,misc]
    try:
        with pytest.raises(ValueError, match="unknown split labels"):
            await tier0_module.step_5_model_trainer(
                "test_exp",
                {"algorithm_name": "LogisticRegression"},
                {"qc_passed": True},
                X,
                y,
                success_criteria={},
                entity_ids=entity_ids,
                dates=dates,
                split_mode="auto",
                pre_assigned_splits=pre_assignments,
            )
    finally:
        if original_agent is not None:
            mt_pkg.ModelTrainerAgent = original_agent  # type: ignore[attr-defined,misc]


def test_step_5_combined_mode_requires_entity_and_date(tier0_module):
    """Block 4 4-MIN-2: ``split_mode='combined'`` is strict-mode (extra
    spec) and must surface a clear ValueError when ``entity_ids`` /
    ``dates`` are absent — no silent fallback to legacy random."""
    import asyncio

    X, y, _entity_ids, _dates = _toy_dataset(n_entities=10)

    async def _run_no_entity():
        return await tier0_module.step_5_model_trainer(
            "test_exp",
            {"algorithm_name": "LogisticRegression"},
            {"qc_passed": True},
            X,
            y,
            success_criteria={},
            entity_ids=None,
            dates=None,
            split_mode="combined",
        )

    with pytest.raises(
        ValueError,
        match="combined.*requires both entity_ids and dates",
    ):
        asyncio.run(_run_no_entity())
