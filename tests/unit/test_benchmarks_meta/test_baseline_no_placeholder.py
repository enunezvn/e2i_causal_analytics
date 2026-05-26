"""Meta-test: drift guard against placeholder baselines (issue #403).

Issue #403 re-blessed 4 of 6 baselines in
``tests/benchmarks/baselines/performance.json`` from CI-measured values;
issue #414 re-blessed the last 2 (the hybrid-retriever p50/p95 baselines)
from a local pgvector substrate — no Supabase/OpenAI secrets needed. All 6
are now CI-blessed-median. This meta-test pins the post-re-bless state so a
future revert to placeholder mode (or accidental zeroing-out of a baseline)
fails loudly here, in the default unit-test sweep, rather than only when
someone happens to run the benchmark harness.

Drift-guard contract:
  * All 6 re-blessed boxes (cascade_5hop_bfs, bm25_build_1k, bm25_build_5k,
    bm25_build_10k, hybrid_retriever_search_p50, hybrid_retriever_search_p95)
    MUST have ``mean_ms > 0.0`` and MUST NOT carry placeholder breadcrumbs
    (``_observed_on_dev_box_*`` or ``_seeded_*`` keys) — those breadcrumbs
    belong to the "placeholder-first-run-blesses" era, which closed in #403.
  * The top-level ``_baseline_strategy`` MUST be the post-re-bless value
    (``ci-blessed-median``), not the pre-re-bless value
    (``placeholder-first-run-blesses``).
  * The top-level metadata MUST identify which CI runs blessed the
    baseline (``_blessed_from_ci_runs``).

This test deliberately lives under ``tests/unit/`` so it runs in every
``pytest tests/`` invocation — making the drift visible without needing
to run the slow benchmark harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASELINE_FILE = _REPO_ROOT / "tests" / "benchmarks" / "baselines" / "performance.json"

# Boxes whose baselines were re-blessed from CI. cascade + bm25 from issue
# #403; the 2 hybrid boxes from issue #414 (CI-blessed-median against a local
# pgvector substrate — see their _ci_observation in performance.json). Their
# ``mean_ms`` MUST be strictly positive.
_REBLESSED_BOXES = (
    "cascade_5hop_bfs",
    "bm25_build_1k",
    "bm25_build_5k",
    "bm25_build_10k",
    "hybrid_retriever_search_p50",
    "hybrid_retriever_search_p95",
)

# Keys that belong to the pre-re-bless era and MUST be absent from the
# re-blessed boxes' specs. Their presence indicates a partial-revert.
_PLACEHOLDER_BREADCRUMB_PREFIXES = ("_observed_on_dev_box_", "_seeded_")


def _load_baseline() -> Dict[str, Any]:
    assert _BASELINE_FILE.exists(), f"baseline file missing: {_BASELINE_FILE}"
    with _BASELINE_FILE.open("r", encoding="utf-8") as fh:
        loaded: Dict[str, Any] = json.load(fh)
    return loaded


def test_top_level_strategy_is_ci_blessed() -> None:
    """Top-level ``_baseline_strategy`` must read ``ci-blessed-median``.

    Drift-guard premise (honest-rename pattern, propagated cross-PR per
    [[feat-376-phase4-schema-close-20260520]] L1): when a baseline's
    blessing-mode changes, the metadata field that names the mode MUST
    rename too — otherwise the documentation drifts silently from the
    actual state.
    """
    baseline = _load_baseline()
    strategy = baseline.get("_baseline_strategy")
    assert strategy == "ci-blessed-median", (
        f"_baseline_strategy must be 'ci-blessed-median' after the #403 "
        f"re-bless; got {strategy!r}. If you legitimately need to revert "
        "to placeholder mode, file a follow-up issue first and update this "
        "test to match — do NOT silently downgrade the field."
    )


def test_top_level_carries_ci_provenance() -> None:
    """Re-blessed baselines must identify their source CI runs.

    Drift-guard premise: without a recorded provenance, a future
    re-bless PR cannot tell whether the current numbers came from 1 run
    (one-shot luck) or ≥3 runs (variance-aware median). Issue #403's
    acceptance criterion says ≥3 — pin it here.
    """
    baseline = _load_baseline()
    runs = baseline.get("_blessed_from_ci_runs")
    assert isinstance(runs, list), (
        f"_blessed_from_ci_runs must be a list of CI run IDs; got {type(runs)}"
    )
    assert len(runs) >= 3, (
        f"_blessed_from_ci_runs must include ≥3 CI run IDs per issue "
        f"#403 acceptance criterion; got {len(runs)}: {runs!r}"
    )
    for run_id in runs:
        assert isinstance(run_id, (str, int)), (
            f"each entry in _blessed_from_ci_runs must be a run ID "
            f"(str or int); got {run_id!r} of type {type(run_id)}"
        )


@pytest.mark.parametrize("box", _REBLESSED_BOXES)
def test_reblessed_box_has_positive_mean_ms(box: str) -> None:
    """Re-blessed boxes must carry a non-placeholder ``mean_ms``.

    Drift-guard premise: the placeholder-first-run-blesses policy used
    ``mean_ms == 0.0`` as a sentinel that the benchmark would
    unconditionally pass. Post-re-bless, every CI-blessable box MUST
    have a strictly positive ``mean_ms`` so the assertion path actually
    fires on regressions.
    """
    baseline = _load_baseline()
    spec = baseline[box]
    mean_ms = spec.get("mean_ms")
    assert isinstance(mean_ms, (int, float)) and not isinstance(mean_ms, bool), (
        f"{box}.mean_ms must be numeric (not bool); got {mean_ms!r}"
    )
    assert mean_ms > 0.0, (
        f"{box}.mean_ms must be > 0.0 after the #403 re-bless; got "
        f"{mean_ms!r}. A zero mean_ms reverts the benchmark to the "
        "placeholder-first-run-blesses sentinel and the assertion is "
        "skipped — defeating the purpose of the harness."
    )


@pytest.mark.parametrize("box", _REBLESSED_BOXES)
def test_reblessed_box_has_no_placeholder_breadcrumbs(box: str) -> None:
    """Re-blessed boxes must shed the pre-re-bless breadcrumbs.

    Drift-guard premise: leaving ``_observed_on_dev_box_*`` or
    ``_seeded_*`` keys on a re-blessed spec is documentation drift —
    the spec is no longer dev-box-observed but the breadcrumb says it
    is. Honest-rename: the breadcrumb must reflect the CURRENT state of
    the spec, not its history.
    """
    baseline = _load_baseline()
    spec = baseline[box]
    placeholder_keys = sorted(
        k for k in spec if any(k.startswith(prefix) for prefix in _PLACEHOLDER_BREADCRUMB_PREFIXES)
    )
    assert not placeholder_keys, (
        f"{box} still carries placeholder breadcrumbs from the pre-"
        f"re-bless era: {placeholder_keys!r}. These keys belong to "
        "the placeholder-first-run-blesses policy and must be removed "
        "(or renamed to reflect the post-re-bless reality) when a box "
        "is re-blessed."
    )


@pytest.mark.parametrize("box", _REBLESSED_BOXES)
def test_reblessed_box_provenance(box: str) -> None:
    """Each re-blessed box should carry per-box CI provenance metadata.

    Drift-guard premise: per-box ``_blessed_from_ci_runs`` lets a future
    bisect identify which specific CI run a value came from when the
    top-level value covers multiple boxes; we accept either the
    per-box field OR inheritance from the top-level list — pin that
    AT LEAST one of the two surfaces names the runs that blessed this
    box.
    """
    baseline = _load_baseline()
    spec = baseline[box]
    has_per_box = isinstance(spec.get("_blessed_from_ci_runs"), list) and bool(
        spec.get("_blessed_from_ci_runs")
    )
    has_top_level = isinstance(baseline.get("_blessed_from_ci_runs"), list) and bool(
        baseline.get("_blessed_from_ci_runs")
    )
    assert has_per_box or has_top_level, (
        f"{box} is missing CI-run provenance — neither the spec nor "
        "the top-level baseline lists _blessed_from_ci_runs. Document "
        "which CI run(s) produced this baseline so future bisects can "
        "trace a regression back to its source."
    )
