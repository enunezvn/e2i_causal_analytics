"""Meta-test: drift guard against placeholder baselines (issue #403).

Issue #403 re-blesses 4 of 6 baselines in
``tests/benchmarks/baselines/performance.json`` from CI-measured values
(the 2 hybrid-retriever baselines remain placeholder because their
benchmark skips in CI without Supabase + OpenAI secrets). This meta-test
pins the post-re-bless state so a future revert to placeholder mode (or
accidental zeroing-out of a baseline) fails loudly here, in the default
unit-test sweep, rather than only when someone happens to run the
benchmark harness.

Drift-guard contract:
  * The 4 re-blessable boxes (cascade_5hop_bfs, bm25_build_1k,
    bm25_build_5k, bm25_build_10k) MUST have ``mean_ms > 0.0`` and MUST
    NOT carry placeholder breadcrumbs (``_observed_on_dev_box_*`` or
    ``_seeded_*`` keys) — those breadcrumbs belong to the
    "placeholder-first-run-blesses" era, which closed in issue #403.
  * The 2 hybrid baselines (p50/p95) MAY remain at 0.0 with a refreshed
    breadcrumb pointing at the follow-up tracker; we don't pin their
    state because the legitimate re-bless path lands in a separate PR.
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

# Boxes whose baselines were re-blessed from CI in issue #403. Their
# ``mean_ms`` MUST be strictly positive after the re-bless lands.
_REBLESSED_BOXES = (
    "cascade_5hop_bfs",
    "bm25_build_1k",
    "bm25_build_5k",
    "bm25_build_10k",
)

# Boxes whose benchmarks skip in CI (require SUPABASE_URL + SUPABASE_KEY
# + OPENAI_API_KEY) and therefore CANNOT be re-blessed from a CI run.
# These intentionally remain at 0.0 with a refreshed breadcrumb; the
# follow-up tracker is filed when the secrets become available.
_PLACEHOLDER_BOXES_HYBRID = (
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


def test_hybrid_boxes_are_still_placeholder_with_refreshed_breadcrumb() -> None:
    """The 2 hybrid boxes stay placeholder until Supabase + OpenAI CI secrets land.

    Drift-guard premise: silently re-blessing the hybrid boxes from a
    dev-box measurement is exactly the failure mode that prompted issue
    #403 in the first place. The 2 hybrid boxes MUST stay at
    ``mean_ms == 0.0`` until a CI run with both secrets present produces
    a measurement; the placeholder breadcrumb must point at the follow-up
    tracker so a reviewer knows the gap is tracked, not forgotten.
    """
    baseline = _load_baseline()
    for box in _PLACEHOLDER_BOXES_HYBRID:
        spec = baseline[box]
        mean_ms = spec.get("mean_ms")
        assert mean_ms == 0.0, (
            f"{box}.mean_ms must remain 0.0 (placeholder) until a CI "
            f"run with SUPABASE_URL + SUPABASE_KEY + OPENAI_API_KEY "
            f"produces a measurement; got {mean_ms!r}. If those secrets "
            "are now available in CI, file a follow-up PR to re-bless "
            "these boxes from CI data — do NOT bless from dev-box "
            "measurements."
        )
        # The breadcrumb must explicitly explain why the placeholder
        # stays — naming the missing secrets and pointing at a follow-up
        # tracker. We don't pin the exact key name (the breadcrumb may
        # rename to a more honest shape post-re-bless) but we DO pin that
        # some text-valued metadata key explains the placeholder.
        rationale_values = [
            v for k, v in spec.items() if k.startswith("_") and isinstance(v, str) and v.strip()
        ]
        assert rationale_values, (
            f"{box} is a placeholder but carries no explanatory "
            "metadata. Add a string-valued underscore-key (e.g. "
            "_placeholder_rationale) explaining why the value is 0.0 "
            "and naming the follow-up tracker."
        )
