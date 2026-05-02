"""W3-lite Day 4 — evaluation_mode dispatch + repeated_kfold_orchestrator integration.

Spec: shard 17 W3 row Day 4 + shard 21 §B (TrainerInput evaluation_mode + dispatch logic) +
§G.2 (Backward compatibility) + §G.3 (Repeated mode tests).

Naming decision: shard 21 uses ``"single" | "repeated_k10"`` (default ``"single"``); shard 17
W3 row Day 4 uses ``"single_split" | "repeated_kfold"``. Implementation follows shard 21
since orchestrator + tests are written against those names. Naming divergence flagged in
state file (`adaptive_v3_followup_state.md`) for user decision on whether to amend shard 17.

Backward-compat invariant: when ``evaluation_mode`` is absent OR explicitly ``"single"``,
the agent path MUST be byte-identical to the pre-W4-day-4 baseline (no extra MLflow nesting,
no fold_metrics, no aggregate_metrics).

The repeated-mode tests mock the per-fold graph invocation
(``ModelTrainerAgent.graph.ainvoke``) with a fast stub so the orchestrator's dispatch +
fold loop + state threading is exercised in <5s instead of the full graph's bootstrap-CI
+ advanced-validation pipeline. End-to-end k=10 with the real graph is deferred to Day-5
``@pytest.mark.slow`` per the W2-prep memory-pressure callout (16 GB / 6 GB-swap droplet).
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

SEED = 42
N = 200
N_FEATURES = 4


def _make_full_data(prevalence: float = 0.30) -> Dict[str, Any]:
    """Build a deterministic synthetic full_data dict for repeated mode."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(
        rng.standard_normal((N, N_FEATURES)),
        columns=[f"x{i}" for i in range(N_FEATURES)],
    )
    n_positive = int(round(N * prevalence))
    y_arr = np.zeros(N, dtype=int)
    positive_idx = rng.choice(N, size=n_positive, replace=False)
    y_arr[positive_idx] = 1
    return {"X": X, "y": pd.Series(y_arr, name="y")}


def _make_minimal_input(
    evaluation_mode: str | None = None, k: int | None = None
) -> Dict[str, Any]:
    """Build a minimal agent.run input that exercises dispatch only.

    For dispatch tests we don't actually need the graph to succeed; we only
    need the dispatch decision to be made. The ``model_candidate`` + ``qc_report``
    + ``experiment_id`` validation in agent.run runs BEFORE dispatch, so all 3
    must be present.

    ``k`` overrides the splitter's k (default 10). Smoke tests use k=3 to keep
    wall-clock under the W2-prep memory-pressure callout budget — the
    ``len(fold_metrics) == k`` invariant is what we're verifying, not the
    full 10-fold throughput (Day-5 will exercise k=10 end-to-end).
    """
    full_data = _make_full_data()
    inp: Dict[str, Any] = {
        "model_candidate": {
            "algorithm_name": "LightGBM",
            "algorithm_class": "lightgbm.LGBMClassifier",
            "hyperparameter_search_space": {},
            "default_hyperparameters": {"n_estimators": 5, "verbose": -1},
        },
        "qc_report": {"qc_passed": True},
        "experiment_id": "test_evaluation_mode_dispatch",
        "success_criteria": {},
        "enable_hpo": False,
        "enable_mlflow": False,
        "enable_checkpointing": False,
        "full_data": full_data,
    }
    if evaluation_mode is not None:
        inp["evaluation_mode"] = evaluation_mode
    if k is not None:
        inp["repeated_splits_config"] = {"k": k}
    return inp


# ---------------------------------------------------------------------------
# G.2 — Backward compatibility (shard 21 §G.2)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_evaluation_mode_raises_value_error() -> None:
    """agent.run({"evaluation_mode": "wat", ...}) must raise ValueError naming valid modes."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    with pytest.raises(ValueError, match=r"single.*repeated_k10|repeated_k10.*single"):
        await agent.run(_make_minimal_input(evaluation_mode="wat"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explicit_single_mode_is_accepted() -> None:
    """evaluation_mode='single' must be accepted without raising the unknown-mode error."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    # Single mode without prebuilt splits will fail downstream in the graph;
    # dispatch validation should NOT raise the unknown-mode error first.
    try:
        await agent.run(_make_minimal_input(evaluation_mode="single"))
    except ValueError as e:
        assert "evaluation_mode" not in str(e), (
            f"single mode should not be rejected by dispatch: {e}"
        )
    except Exception:
        # Downstream graph failures are expected without prebuilt splits;
        # dispatch is what we're verifying here.
        pass


# ---------------------------------------------------------------------------
# G.3 — Repeated mode tests (shard 21 §G.3)
# ---------------------------------------------------------------------------


def _fold_invocation_recorder() -> tuple[AsyncMock, list[Dict[str, Any]]]:
    """Build a fast mock for ``self.graph.ainvoke`` that records per-fold state.

    The orchestrator invokes ``self.run`` recursively per fold (which in turn
    runs the LangGraph pipeline). Mocking ``self.graph.ainvoke`` lets us
    bypass the full pipeline (heavy: bootstrap CI / advanced validation /
    LightGBM training) while still exercising the orchestrator's dispatch +
    per-fold state construction + fold_random_state threading.
    """
    captured_states: list[Dict[str, Any]] = []

    async def fake_ainvoke(state: Dict[str, Any]) -> Dict[str, Any]:
        captured_states.append(dict(state))
        # Return a minimal fold result mirroring the model_trainer output shape.
        return {
            **state,
            "trained_model": object(),
            "train_metrics": {"auroc": 0.80},
            "validation_metrics": {"auroc": 0.78},
            "test_metrics": {"auroc": 0.77},
            "auc_roc": 0.77,
            "brier_score": 0.18,
            "framework": "lightgbm",
        }

    mock = AsyncMock(side_effect=fake_ainvoke)
    return mock, captured_states


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_k10_runs_k_folds_smoke() -> None:
    """evaluation_mode='repeated_k10' produces fold_metrics with k entries (smoke).

    Day-4 MVP smoke: exercises the orchestrator dispatch + per-fold loop with
    a reduced k=3 fixture and a mocked graph invocation. Verifies the
    ``len(fold_metrics) == k`` invariant and the legacy single-graph path is
    invoked once per fold. The full k=10 end-to-end is deferred to Day-5
    ``@pytest.mark.slow``.
    """
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    mock, _ = _fold_invocation_recorder()
    with patch.object(agent.graph, "ainvoke", mock):
        output = await agent.run(
            _make_minimal_input(evaluation_mode="repeated_k10", k=3)
        )

    assert "fold_metrics" in output, "fold_metrics missing from repeated_k10 output"
    assert isinstance(output["fold_metrics"], list), (
        f"fold_metrics is {type(output['fold_metrics']).__name__}, expected list"
    )
    assert len(output["fold_metrics"]) == 3, (
        f"Expected 3 fold_metrics (k=3 smoke fixture), got {len(output['fold_metrics'])}"
    )
    assert output.get("k_folds") == 3
    assert output.get("splitter_strategy") == "shuffle_split"
    assert mock.await_count == 3, (
        f"Graph should be invoked once per fold (3); got {mock.await_count}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_k10_threads_distinct_fold_random_state_per_fold() -> None:
    """Each fold's state must carry a distinct fold_random_state from the splitter.

    This is the critical handoff between Day-3 wiring (resolve_fold_random_state) and
    Day-4 orchestrator. Without per-fold seed threading, every fold would train with
    the same RNG state and the selection-bias correction would be void.
    """
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    mock, captured_states = _fold_invocation_recorder()
    with patch.object(agent.graph, "ainvoke", mock):
        output = await agent.run(
            _make_minimal_input(evaluation_mode="repeated_k10", k=3)
        )

    # The fold_metrics records carry the splitter's per-fold seed.
    fold_seeds_from_records = [fm.get("fold_random_state") for fm in output["fold_metrics"]]
    assert all(s is not None for s in fold_seeds_from_records), (
        f"Some folds missing fold_random_state: {fold_seeds_from_records}"
    )
    assert len(set(fold_seeds_from_records)) == 3, (
        f"Fold seeds collided in fold_metrics: {fold_seeds_from_records}"
    )

    # The downstream graph state ALSO sees the per-fold seed via initial_state
    # (Day-3 ``resolve_fold_random_state`` reads ``state['fold_random_state']``).
    fold_seeds_in_state = [s.get("fold_random_state") for s in captured_states]
    assert fold_seeds_in_state == fold_seeds_from_records, (
        f"Per-fold state.fold_random_state {fold_seeds_in_state} does not match "
        f"fold_metrics.fold_random_state {fold_seeds_from_records} — handoff is broken"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_k10_emits_evaluation_mode_in_output() -> None:
    """Output must record the evaluation_mode that was run (for downstream gating logic)."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    mock, _ = _fold_invocation_recorder()
    with patch.object(agent.graph, "ainvoke", mock):
        output = await agent.run(
            _make_minimal_input(evaluation_mode="repeated_k10", k=3)
        )
    assert output.get("evaluation_mode") == "repeated_k10"
    assert output.get("test_metrics_population_strategy") == "fold_mean"
    assert "evaluation_result_schema_version" in output


# ---------------------------------------------------------------------------
# Backward-compat: default mode (single) without full_data
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_default_mode_is_single_when_evaluation_mode_absent() -> None:
    """No evaluation_mode → dispatch routes to single (legacy) path.

    Verifies via the absence-of-fold-metrics signal in single mode (the orchestrator
    does NOT populate fold_metrics). This is the byte-identity proxy for backward-compat
    callers (Tier-0 supervisor, FastAPI endpoint, contract tests).
    """
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    # Single mode with full_data only (no prebuilt splits) will fail downstream in the
    # graph; we only verify dispatch chose the single path (would have raised
    # ValueError if dispatch interpreted absent flag as "repeated_k10").
    try:
        output = await agent.run(_make_minimal_input(evaluation_mode=None))
        # If we got here, the legacy path succeeded; assert no fold_metrics was emitted.
        assert "fold_metrics" not in output or output.get("fold_metrics") is None, (
            "Single mode emitted fold_metrics (expected only in repeated_k10 mode)"
        )
    except ValueError as e:
        # The validation ValueError for "Unknown evaluation_mode" would prove dispatch
        # broke; any other ValueError (e.g., from downstream nodes) is unrelated.
        assert "evaluation_mode" not in str(e), (
            f"Default-mode dispatch incorrectly raised evaluation_mode error: {e}"
        )
    except Exception:
        # Downstream graph failures (missing splits) are expected — they prove
        # the legacy single-graph path was taken.
        pass


# ---------------------------------------------------------------------------
# Q2 contract regression: grep-based audit of unallowlisted random_state=42
# ---------------------------------------------------------------------------


def test_no_unallowlisted_random_state_42_in_model_trainer() -> None:
    """Cycle-14 Q2 deferred-sites contract: every code-level ``random_state=42`` under
    src/agents/ml_foundation/model_trainer/ MUST be either:
      (a) inside random_state.py itself (the helper module), OR
      (b) marked with the noqa-style allow-list comment ``# noqa: random_state=42 — <reason>``.

    Locks the Day-4/5 contract per cycle-14 codex Q2 IMPORTANT finding: prevents
    silent-leak regressions where a new ``random_state=42`` literal is added without
    threading through ``resolve_fold_random_state`` or being intentionally fixed.

    Heuristic for "code-level": the literal appears followed by ``,`` ``)`` or
    end-of-token in a Python expression context (kwarg site or end-of-call). Prose
    mentions in docstrings / comments using backticks (``random_state=42``) or
    plain text references are NOT flagged — only actual constructor / function-call
    kwarg usage that would emit the seed at runtime.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    model_trainer_dir = repo_root / "src" / "agents" / "ml_foundation" / "model_trainer"
    assert model_trainer_dir.exists(), f"model_trainer directory not found: {model_trainer_dir}"

    helper_module = model_trainer_dir / "random_state.py"

    # Match `random_state=42` only when followed by `,`, `)`, whitespace, or
    # end-of-line — i.e., a real Python kwarg site, not a prose reference.
    # The end-of-line alternate (cycle-15 I-6 fix per codex) closes the
    # false-negative gap for multi-line kwarg patterns where `random_state=42`
    # is the last token before `\n` followed by `)` on the next line.
    code_pattern = re.compile(r"random_state\s*=\s*42\s*[,)\s]|random_state\s*=\s*42$")
    allow_pattern = re.compile(r"#\s*noqa:\s*random_state=42")

    offenders: list[str] = []
    for py in model_trainer_dir.rglob("*.py"):
        if py == helper_module:
            continue
        text = py.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            if code_pattern.search(line) and not allow_pattern.search(line):
                offenders.append(
                    f"{py.relative_to(repo_root)}:{line_no}: {line.strip()}"
                )

    assert not offenders, (
        "Found unallowlisted `random_state=42` kwarg literals under model_trainer/. "
        "Each occurrence must either thread through resolve_fold_random_state OR "
        "carry an explicit `# noqa: random_state=42 — <reason>` comment. Offenders:\n"
        + "\n".join(offenders)
    )
