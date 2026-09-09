# Expert-Review Loop Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the expert-review loop visible and usable end to end: real evidence from the two non-critical refutation tests, a guarded promote that can never land over a rejection, a review lookup route, review state and the discovered-DAG id on the drill-down, a workable queue page, a refreshed lineage document, and a live verification of approve, reject and the enforcement switch.

**Architecture:** Spec `docs/superpowers/specs/2026-09-08-expert-review-loop-closure-design.md` (read §2 and §4 first). Backend: the refutation runner replaces two discard-after-compute DoWhy calls with its own resample loops; a new SQL RPC (migration 134) evaluates the review chronology rule inside the promote statement; one new read route returns a review in any status plus its same-structure history. Frontend: hand-written types learn the fields the API already returns; a small review-status panel on the drill-down deep-links to the queue page, which gains a linked-review card, a brand filter, a summary error state and assessment prefetch. Docs: the lineage page is rewritten to the shipped state and its anchors re-resolved.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, supabase-py (async), DoWhy 0.14 / EconML 0.16, PostgreSQL (plpgsql), pytest + pytest-asyncio; React 18 + TypeScript, TanStack Query, react-router-dom 7, vitest + Testing Library.

**Conventions for every task**

- Work in the lane worktree `/home/enunez/Projects/e2i_causal_analytics/.worktrees/lane1-review-loop` on branch `claude/lane1-expert-review-loop`. Run `git branch --show-current` before every commit.
- Python: `PY=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python`. Tests: `$PY -m pytest <paths> -q -p no:cacheprovider`. Lint: `$PY -m ruff check <files>` and `$PY -m ruff format --check <files>` (ruff is pinned 0.14.10). Type-check only the changed files: `$PY -m mypy --config-file pyproject.toml <files>` (never the whole tree on this box).
- Frontend: run from `frontend/`: `npx vitest run <paths>`, `npm run typecheck`, `npx eslint <files>`.
- Commit message footer on every commit:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv
```

- Never `git stash` (shared stash). Never merge with squash. Nothing is pushed until Task 14.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `src/causal_engine/refutation_runner.py` | thresholds; `_refit_effect_on`, `_refutation_frame`, `_resample_effects`, `_significance_p_value`, `_budget_skip_result`, `_resample_seed_for`; rewritten `_run_data_subset_test` / `_run_bootstrap_test`; `run_all_tests` passes deadline + seed | 1, 2 |
| `tests/unit/test_causal_engine/test_refutation_runner.py` | shared stubs gain `_data`, `_stub_estimate`, `_sequence_estimate`; two tests rewritten | 1 |
| `tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py` | new: evidence, thresholds, deadline, seeding, band arithmetic, wiring | 1, 2 |
| `tests/unit/test_causal_engine/test_refutation_runner_1419.py`, `..._randomized.py` | `estimate=object()` → `_stub_estimate()` | 1 |
| `database/migrations/134_guarded_causal_path_promote.sql` | `dag_structure_rejected`, `promote_causal_path_guarded`, grants, assertion | 3 |
| `tests/unit/test_database/test_migration_134_guarded_promote.py` | pins the migration's contract | 3 |
| `src/repositories/causal_path.py` | `set_validation_status` routes through the RPC when a hash is given | 4 |
| `tests/unit/test_repositories/test_causal_path_promoter_1352.py` | new class `TestGuardedPromoteRpc` | 4 |
| `src/agents/causal_impact/nodes/refutation.py` | promote call passes hash + brand | 5 |
| `tests/unit/test_agents/test_causal_impact/test_refutation_promoter_1352.py` | fake accepts kwargs; new test | 5 |
| `src/api/schemas/expert_review.py` | `ReviewRecord`, `ExpertReviewDetailResponse` | 6 |
| `src/api/routes/expert_review.py` | `GET /expert-reviews/{review_id}` | 6 |
| `tests/unit/test_api/test_expert_review_detail_route.py` | new route tests | 6 |
| `frontend/src/types/generated/api.ts` | regenerated | 7 |
| `frontend/src/types/expert-review.ts`, `api/expert-review.ts`, `lib/query-client.ts`, `hooks/api/use-expert-review.ts` (+ test) | detail type, client, key, hook | 8 |
| `frontend/src/types/causal.ts` | review fields + `discovered_dag_id` | 9 |
| `frontend/src/components/causal/ReviewStatusPanel.tsx` (+ test), `CausalAnalysisDetail.tsx` (+ test) | drill-down review state | 9 |
| `frontend/src/components/expert-review/{checklist.ts,DagPanel.tsx,ResolveForm.tsx,LinkedReviewCard.tsx,PrepareAssessmentsButton.tsx}` | extracted + new queue components | 10 |
| `frontend/src/pages/ExpertReviews.tsx` (+ test) | linked card, brand filter, summary error, prefetch | 10 |
| `docs/lineage/causal_dag_lineage.html` | sections + anchors | 11 |
| `docs/demos/results/<date>_expert_review_loop/` | live evidence | 13, 15 |
| `src/repositories/json_utils.py` (new), `src/repositories/causal_validation.py`, `src/repositories/discovered_dag.py` | evidence rows written as JSON OBJECTS (non-finite → null); shared sanitiser | 2b |
| `tests/unit/test_repositories/test_causal_validation.py` | new class `TestEvidenceRowsAreJsonObjects` | 2b |
| `database/migrations/135_causal_validations_json_objects.sql`, `tests/unit/test_database/test_migration_135_json_objects.py` | backfill of the 480 string-shaped rows + contract test | 2b |
| `src/causal_engine/expert_review_gate.py`, `tests/unit/test_causal_engine/test_expert_review_gate.py` | tie-only rule; durable-rejection block moved above the pending check | 3b |

---

## Execution protocol (owner decisions, 2026-09-09)

- **Subagent-driven**: a fresh subagent per task, working IN this worktree (`.worktrees/lane1-review-loop`, branch `claude/lane1-expert-review-loop`), ONE task at a time — never two subagents at once (this box is prod and has been OOM-killed before). The dispatcher reviews each task's result against the plan and the spec before the next task starts (`superpowers:subagent-driven-development`).
- **TDD red-first** for every task (the steps are written that way). **No mocks in production paths, no stubbed "results"**: real evidence, real DB rehearsals (BEGIN … ROLLBACK), real container content.
- **Fixed point per task**: once a task's tests are green, run `ralph-wiggum:ralph-loop` around a codex read-only audit (`codex:codex-rescue`, subscription channel only; if the plugin answers "CLI not installed", run `codex exec -s read-only --ephemeral --skip-git-repo-check -C "$PWD" "$(cat brief.md)" < /dev/null` directly) until `VERDICT: ACCEPT`, every HIGH/MED fixed with a test first; the brief carries the mandated pushback paragraph. Task 12 repeats this on the whole diff before the PR.
- **Questions that come up**: ascertain the codebase's intent first (`git log`, PR bodies, linked issues, comments), use web research when the tree cannot answer, converge with ralph-loop + codex-rescue, and answer with data — run the cheap disproof (a test, a one-line repro, a live read) rather than theorising or pattern-matching.
- **Memory**: `free -m` before every heavy step (a whole test directory, vitest, `tsc -b`, codex); never whole-tree mypy (CI is the arbiter); targeted pytest only; if `MemAvailable` < 1.5 GiB, stop and report instead of starting the step.
- **CI batched at the end**: no push per task. One push + one PR at Task 14 after Task 12's fixed point; one deploy; then Task 15's live verification. Migrations 134 and 135 are applied by that deploy (`scripts/run_migrations.sh` runs on every deploy).
- **Permissions**: migrations, bash, `docker exec` on the live stack, the live BEGIN/ROLLBACK rehearsals and the two synthetic adjudications are authorised (2026-09-08/09). The merge still waits for the owner's explicit go.
- **Completeness**: nothing in this plan is optional; no listed feature is deferred to a follow-up without the owner's word.

---

### Task 1: Runner — real evidence for data_subset and bootstrap, intended bootstrap thresholds

**Files:**
- Modify: `src/causal_engine/refutation_runner.py` (PASS_THRESHOLDS ~line 470; new helpers after `_require_p_value` ~line 75; `_run_data_subset_test` ~line 1196; `_run_bootstrap_test` ~line 1320)
- Modify: `tests/unit/test_causal_engine/test_refutation_runner.py` (helpers lines 42–61; tests at lines 751–780 and 798–832)
- Modify: `tests/unit/test_causal_engine/test_refutation_runner_1419.py`, `tests/unit/test_causal_engine/test_refutation_runner_randomized.py`
- Create: `tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py`

- [ ] **Step 1: Extend the shared test stubs**

In `tests/unit/test_causal_engine/test_refutation_runner.py`, add to the imports at the top (keep existing ones):

```python
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
```

Replace the existing `_make_stub_causal_model` (lines 49–61) with:

```python
class _StubRefitEstimator:
    """What ``estimate.estimator.get_new_estimator_object`` returns (spec §4.1).

    ``fit`` records the resample and ``estimate_effect`` reports an effect
    computed FROM it, so subset and bootstrap draws vary the way a real re-fit
    would (a constant series would make DoWhy's normal test divide by zero).
    """

    def __init__(self, effect_fn: Callable[[pd.DataFrame], float]) -> None:
        self._effect_fn = effect_fn
        self.fitted_on: Optional[pd.DataFrame] = None

    def fit(self, data: pd.DataFrame, effect_modifier_names=None, **_fit_params) -> None:  # noqa: ANN001
        self.fitted_on = data

    def estimate_effect(  # noqa: ANN001
        self, data: pd.DataFrame, control_value=0, treatment_value=1, target_units="ate"
    ) -> SimpleNamespace:
        return SimpleNamespace(value=float(self._effect_fn(data)))


def _stub_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """The frame a stub CausalModel was 'built on' (DoWhy keeps it as ``_data``)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"t": rng.integers(0, 2, n), "y": rng.random(n), "c": rng.random(n)})


def _stub_estimate(
    value: float = 0.15,
    effect_fn: Optional[Callable[[pd.DataFrame], float]] = None,
) -> SimpleNamespace:
    """A DoWhy-shaped CausalEstimate: ``.value`` plus the four estimator
    attributes ``refutation_runner._refit_effect_on`` reads."""
    fn = effect_fn or (lambda df: value + 0.01 * (float(df["y"].mean()) - 0.5))
    estimator = SimpleNamespace(
        get_new_estimator_object=lambda _estimand: _StubRefitEstimator(fn),
        _effect_modifier_names=[],
        _target_units="ate",
    )
    return SimpleNamespace(value=value, estimator=estimator, control_value=0, treatment_value=1)


def _sequence_estimate(values: List[float], value: float = 0.15) -> SimpleNamespace:
    """An estimate whose successive re-fits report ``values`` in order."""
    it = iter(values)
    return _stub_estimate(value=value, effect_fn=lambda _df: next(it))


def _make_stub_causal_model(
    refutation_results_by_method: dict, data: Optional[pd.DataFrame] = None
) -> SimpleNamespace:
    """Construct a stub object shaped like DoWhy's CausalModel.

    ``refute_estimate(estimand, estimate, method_name=..., **kwargs)`` returns
    the canned result for ``method_name`` (placebo / random_common_cause still
    go through it); ``_data`` is the frame the two resample loops draw from.
    """

    def refute_estimate(*_args, method_name: str, **_kwargs):  # noqa: ANN001
        if method_name not in refutation_results_by_method:
            raise KeyError(f"stub did not register method_name={method_name!r}")
        return refutation_results_by_method[method_name]

    return SimpleNamespace(
        refute_estimate=refute_estimate,
        _data=data if data is not None else _stub_frame(),
    )
```

- [ ] **Step 2: Point every stub-driven `estimate=object()` at the new estimate stub**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane1-review-loop
sed -i 's/estimate=object()/estimate=_stub_estimate()/g' tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_randomized.py
sed -i 's/"estimate": object(),/"estimate": _stub_estimate(),/' tests/unit/test_causal_engine/test_refutation_runner_1419.py
grep -c '_stub_estimate()' tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_randomized.py tests/unit/test_causal_engine/test_refutation_runner_1419.py
```

Expected counts: 15, 2, 1. Then add `_stub_estimate` to the existing import lines in the two importing files:

```python
# test_refutation_runner_randomized.py line 27 and test_refutation_runner_1419.py line 48
from tests.unit.test_causal_engine.test_refutation_runner import _full_stub_causal_model, _stub_estimate
```

- [ ] **Step 3: Rewrite the two "passed" tests in `test_refutation_runner.py`**

Replace `test_run_data_subset_test_passed` (lines 751–778) with:

```python
    def test_run_data_subset_test_passed(self, runner):
        """Real evidence (spec §4.1): the test re-fits on subsets of the model's
        frame and scores how many subset effects fall inside original_ci."""
        result = runner._run_data_subset_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.PASSED
        assert len(result.details["subset_effects"]) == 5
        assert result.details["ci_coverage"] == 1.0
```

Replace `test_run_bootstrap_test_passed` (lines 798–832) with:

```python
    def test_run_bootstrap_test_passed(self, runner):
        """Real evidence (spec §4.1): bootstrap re-fits on row resamples; the
        2.5–97.5 percentile width is compared with original_ci under the
        INTENDED thresholds (pass ≤ 1.5× the original width)."""
        result = runner._run_bootstrap_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.PASSED
        assert (
            len(result.details["bootstrap_effects"]) == runner.config["bootstrap"]["num_bootstraps"]
        )
        assert result.details["ci_ratio"] <= 1.5
```

- [ ] **Step 4: Write the new failing test file**

Create `tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py`:

```python
"""Lane 1 (spec §4.1): data_subset and bootstrap produce REAL distributional evidence.

Before this change both tests called ``causal_model.refute_estimate`` and then
discarded the result because DoWhy 0.14 does not expose per-sample effects on
the refutation object: 96 of 96 live runs recorded them SKIPPED (measured
2026-09-08) at the same compute cost these loops have. The methods now run
their own resample loops with the public estimator calls DoWhy's refuters use,
keep DoWhy's significance test for the p-value, stop at the cooperative
deadline, and reproduce their draws from a seed.
"""

from __future__ import annotations

import time as _t
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from src.causal_engine.errors import RefutationError
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)
from tests.unit.test_causal_engine.test_refutation_runner import (
    _make_stub_causal_model,
    _sequence_estimate,
    _stub_estimate,
)

CI = (0.10, 0.20)


def _subset(runner: RefutationRunner, estimate, **kw):
    return runner._run_data_subset_test(
        original_effect=0.15,
        original_ci=CI,
        causal_model=_make_stub_causal_model({}),
        identified_estimand=object(),
        estimate=estimate,
        use_dowhy=True,
        **kw,
    )


def _bootstrap(runner: RefutationRunner, estimate, **kw):
    return runner._run_bootstrap_test(
        original_effect=0.15,
        original_ci=CI,
        causal_model=_make_stub_causal_model({}),
        identified_estimand=object(),
        estimate=estimate,
        use_dowhy=True,
        **kw,
    )


def _bootstrap_values(width: float, center: float = 0.15, n: int = 40) -> List[float]:
    """40 evenly spaced values whose 2.5th–97.5th percentile span is ``width``
    (np.percentile's linear interpolation gives span = 0.95 × range)."""
    span = width / 0.95
    return [float(v) for v in np.linspace(center - span / 2, center + span / 2, n)]


class TestDataSubsetRealEvidence:
    def test_records_per_subset_effects_and_scores_coverage(self):
        runner = RefutationRunner()
        result = _subset(runner, _stub_estimate())
        assert result.status == RefutationStatus.PASSED
        assert len(result.details["subset_effects"]) == runner.config["data_subset"]["num_subsets"]
        assert result.details["ci_coverage"] == 1.0
        assert result.details["resamples_completed"] == 5
        assert result.details["resamples_requested"] == 5
        assert result.details["stopped_for_budget"] is False
        assert result.p_value is not None and 0.0 <= result.p_value <= 1.0

    def test_constant_refits_are_an_honest_skip(self):
        """Owner decision 2026-09-09: a zero-variance distribution is SKIPPED with
        a reason -- never a fabricated p-value, never a fail-closed halt from a
        NON-critical test (the critical placebo gate catches an estimator that
        ignores its data)."""
        runner = RefutationRunner()
        result = _subset(runner, _sequence_estimate([0.15] * 5))
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("degenerate_resample_distribution")
        assert result.details["resamples_completed"] == 5
        assert result.details["resample_effects"] == [0.15] * 5
        assert result.p_value is None

    @pytest.mark.parametrize(
        "inside,expected",
        [(8, RefutationStatus.PASSED), (7, RefutationStatus.WARNING), (6, RefutationStatus.FAILED)],
    )
    def test_coverage_thresholds_at_the_boundaries(self, inside, expected):
        runner = RefutationRunner(config={"data_subset": {"num_subsets": 10}})
        values = [0.15 + 0.001 * i for i in range(inside)] + [
            0.50 + 0.01 * i for i in range(10 - inside)
        ]
        result = _subset(runner, _sequence_estimate(values))
        assert result.details["ci_coverage"] == pytest.approx(inside / 10)
        assert result.status == expected

    def test_p_value_is_dowhys_significance_test(self):
        from dowhy.causal_refuter import test_significance

        runner = RefutationRunner()
        est = _stub_estimate()
        result = _subset(runner, est)
        expected = test_significance(est, np.asarray(result.details["subset_effects"]))["p_value"]
        assert result.p_value == pytest.approx(float(expected))

    def test_seeded_runs_reproduce_their_evidence(self):
        runner = RefutationRunner()
        a = _subset(runner, _stub_estimate(), resample_seed=7)
        b = _subset(runner, _stub_estimate(), resample_seed=7)
        c = _subset(runner, _stub_estimate(), resample_seed=8)
        assert a.details["subset_effects"] == b.details["subset_effects"]
        assert a.details["subset_effects"] != c.details["subset_effects"]

    def test_model_without_frame_fails_closed(self):
        runner = RefutationRunner()
        with pytest.raises(RefutationError) as ei:
            runner._run_data_subset_test(
                original_effect=0.15,
                original_ci=CI,
                causal_model=SimpleNamespace(),
                identified_estimand=object(),
                estimate=_stub_estimate(),
                use_dowhy=True,
            )
        assert ei.value.details["reason"] == "refutation_frame_missing"

    def test_refit_failure_fails_closed(self):
        runner = RefutationRunner()

        def boom(_df):
            raise ValueError("estimator exploded")

        with pytest.raises(RefutationError) as ei:
            _subset(runner, _stub_estimate(effect_fn=boom))
        assert ei.value.details["test_name"] == "data_subset"


class TestBootstrapRealEvidence:
    def test_thresholds_are_the_intended_values(self):
        assert RefutationRunner.PASS_THRESHOLDS["bootstrap_ci_ratio"] == {
            "pass": 1.50,
            "warning": 1.75,
        }

    @pytest.mark.parametrize(
        "width,expected",
        [
            (0.149, RefutationStatus.PASSED),
            (0.160, RefutationStatus.WARNING),
            (0.180, RefutationStatus.FAILED),
        ],
    )
    def test_width_ratio_thresholds_mean_what_the_comment_says(self, width, expected):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 40}})
        result = _bootstrap(runner, _sequence_estimate(_bootstrap_values(width)))
        assert result.details["ci_ratio"] == pytest.approx(width / 0.10, rel=1e-6)
        assert result.status == expected

    def test_constant_refits_are_an_honest_skip(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 12}})
        result = _bootstrap(runner, _sequence_estimate([0.15] * 12))
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("degenerate_resample_distribution")
        assert "bootstrap_ci" not in result.details
        assert result.p_value is None

    def test_records_per_bootstrap_effects(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 12}})
        result = _bootstrap(runner, _stub_estimate())
        assert len(result.details["bootstrap_effects"]) == 12
        assert result.details["bootstrap_ci_available"] is True
        assert result.details["resamples_requested"] == 12
        assert result.details["resamples_completed"] == 12
        assert len(result.details["bootstrap_ci"]) == 2


class TestDeadlineInsideTheLoop:
    def _clocked(self, monkeypatch, cost_s: float):
        clock = {"now": 0.0}
        monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])

        def effect(_df):
            clock["now"] += cost_s
            return 0.15 + 0.001 * clock["now"]

        return effect

    def test_stops_early_and_scores_on_the_completed_resamples(self, monkeypatch):
        effect = self._clocked(monkeypatch, 10.0)
        result = _subset(RefutationRunner(), _stub_estimate(effect_fn=effect), deadline=25.0)
        assert result.status == RefutationStatus.PASSED
        assert result.details["resamples_completed"] == 3
        assert result.details["resamples_requested"] == 5
        assert result.details["stopped_for_budget"] is True

    def test_below_the_minimum_is_an_honest_budget_skip(self, monkeypatch):
        effect = self._clocked(monkeypatch, 10.0)
        result = _subset(RefutationRunner(), _stub_estimate(effect_fn=effect), deadline=15.0)
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["resamples_completed"] == 2
        assert "time_budget" in result.details["reason"]
        assert "message" in result.details

    @pytest.mark.parametrize("deadline,skipped", [(9.5, False), (8.5, True)])
    def test_bootstrap_minimum_is_ten(self, monkeypatch, deadline, skipped):
        effect = self._clocked(monkeypatch, 1.0)
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 20}})
        result = _bootstrap(runner, _stub_estimate(effect_fn=effect), deadline=deadline)
        assert (result.status == RefutationStatus.SKIPPED) is skipped
        assert result.details["resamples_completed"] == (9 if skipped else 10)


class TestReviewBandArithmetic:
    """Why REVIEW was unreachable, pinned as arithmetic (spec §2)."""

    @staticmethod
    def _r(name: RefutationTestType, status: RefutationStatus) -> RefutationResult:
        return RefutationResult(
            test_name=name, status=status, original_effect=0.15, refuted_effect=0.15
        )

    def test_sensitivity_warning_with_both_noncritical_failed_is_review(self):
        runner = RefutationRunner()
        tests = [
            self._r(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED),
            self._r(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED),
            self._r(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING),
            self._r(RefutationTestType.DATA_SUBSET, RefutationStatus.FAILED),
            self._r(RefutationTestType.BOOTSTRAP, RefutationStatus.FAILED),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.65)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.REVIEW

    def test_sensitivity_warning_with_both_passed_is_proceed(self):
        runner = RefutationRunner()
        tests = [
            self._r(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED),
            self._r(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED),
            self._r(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING),
            self._r(RefutationTestType.DATA_SUBSET, RefutationStatus.PASSED),
            self._r(RefutationTestType.BOOTSTRAP, RefutationStatus.PASSED),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.90)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.PROCEED

    def test_only_criticals_scoring_could_never_reach_review(self):
        """What production did until this lane: both non-critical tests SKIPPED."""
        runner = RefutationRunner()
        tests = [
            self._r(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED),
            self._r(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED),
            self._r(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING),
            self._r(RefutationTestType.DATA_SUBSET, RefutationStatus.SKIPPED),
            self._r(RefutationTestType.BOOTSTRAP, RefutationStatus.SKIPPED),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.8667, abs=1e-3)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.PROCEED


class TestNonFiniteRefitFailsClosed:
    """Quality review A (spec §5): a NaN/inf re-fit is an anomaly inside the loop
    and must be fail-closed like an exception -- never scored (a NaN counts as
    "below the estimate" in DoWhy's percentile test and poisons np.percentile)."""

    def test_subset_nan_refit_raises(self):
        runner = RefutationRunner()
        with pytest.raises(RefutationError) as ei:
            _subset(runner, _sequence_estimate([0.15, 0.16, float("nan"), 0.14, 0.15]))
        assert ei.value.details["reason"] == "non_finite_resample_effect"
        assert ei.value.details["test_name"] == "data_subset"
        assert ei.value.details["first_non_finite_index"] == 2
        assert ei.value.details["resamples_completed"] == 5

    def test_bootstrap_inf_refit_raises(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 12}})
        values = [0.15 + 0.001 * i for i in range(12)]
        values[7] = float("inf")
        with pytest.raises(RefutationError) as ei:
            _bootstrap(runner, _sequence_estimate(values))
        assert ei.value.details["reason"] == "non_finite_resample_effect"
        assert ei.value.details["test_name"] == "bootstrap"
        assert ei.value.details["first_non_finite_index"] == 7


class TestBelowMinimumConfigIsNotABudgetSkip:
    """Quality review B: with no deadline and a requested count below the
    minimum the loop COMPLETES; the skip must say so, not blame the budget."""

    def test_subset_config_below_minimum(self):
        runner = RefutationRunner(config={"data_subset": {"num_subsets": 2}})
        result = _subset(runner, _stub_estimate())
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("config_below_minimum")
        assert result.details["stopped_for_budget"] is False
        assert result.details["resamples_completed"] == 2
        assert result.details["resamples_requested"] == 2
        assert "message" in result.details
        assert result.execution_time_ms >= 0.0

    def test_bootstrap_config_below_minimum(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 4}})
        result = _bootstrap(runner, _stub_estimate())
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("config_below_minimum")
        assert result.details["stopped_for_budget"] is False
        assert result.details["resamples_completed"] == 4


class TestResampleSeedIs31Bit:
    """Quality review C: the docstring promises a 31-bit seed."""

    def test_seed_is_31_bit_stable_and_distinct(self):
        from src.causal_engine.refutation_runner import _resample_seed_for

        for est_id in ("est-1", "est-2", "3f1c2a9e-0000-4000-8000-000000000000", "x" * 64):
            seed = _resample_seed_for(est_id)
            assert seed is not None and 0 <= seed < 2**31
            assert _resample_seed_for(est_id) == seed
        assert _resample_seed_for("est-1") != _resample_seed_for("est-2")
        assert _resample_seed_for(None) is None
        assert _resample_seed_for("") is None

    def test_a_known_id_would_exceed_31_bits_unmasked(self):
        """Positive control: the mask matters (an unmasked 8-hex-digit prefix is
        32-bit; the reviewer measured a max of 4294943764)."""
        import hashlib

        from src.causal_engine.refutation_runner import _resample_seed_for

        hits = 0
        for i in range(64):
            est_id = f"est-{i}"
            raw = int(hashlib.sha256(est_id.encode("utf-8")).hexdigest()[:8], 16)
            if raw >= 2**31:
                hits += 1
                assert _resample_seed_for(est_id) == raw & 0x7FFFFFFF
        assert hits > 0, "no id in the sample exercised the mask"


class TestDegenerateOriginalCiSkipsBeforeCompute:
    """Quality review D: a widthless reported interval cannot score coverage or a
    width ratio; skip honestly BEFORE any re-fit, never blame the estimate."""

    @staticmethod
    def _never_called(_df):
        raise AssertionError("re-fit must not run for a widthless original_ci")

    def test_subset_widthless_ci(self):
        runner = RefutationRunner()
        result = runner._run_data_subset_test(
            original_effect=0.15,
            original_ci=(0.15, 0.15),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(effect_fn=self._never_called),
            use_dowhy=True,
        )
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("original_ci_degenerate")
        assert result.details["original_ci"] == (0.15, 0.15)
        assert "message" in result.details
        assert result.details["num_subsets"] == runner.config["data_subset"]["num_subsets"]
        assert result.p_value is None
        assert result.execution_time_ms >= 0.0

    def test_bootstrap_inverted_ci(self):
        runner = RefutationRunner()
        result = runner._run_bootstrap_test(
            original_effect=0.15,
            original_ci=(0.20, 0.10),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(effect_fn=self._never_called),
            use_dowhy=True,
        )
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("original_ci_degenerate")
        assert result.details["num_bootstraps"] == runner.config["bootstrap"]["num_bootstraps"]
        assert result.p_value is None


class TestSkipResultsCarryExecutionTime:
    """Quality review E: every skip result records execution_time_ms."""

    def test_budget_skip_has_execution_time(self, monkeypatch):
        clock = {"now": 0.0}
        monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])

        def effect(_df):
            clock["now"] += 10.0
            return 0.15 + 0.001 * clock["now"]

        result = _subset(RefutationRunner(), _stub_estimate(effect_fn=effect), deadline=15.0)
        assert result.status == RefutationStatus.SKIPPED
        assert "time_budget" in result.details["reason"]
        assert result.execution_time_ms > 0.0

    def test_degenerate_skip_has_execution_time(self):
        result = _subset(RefutationRunner(), _sequence_estimate([0.15] * 5))
        assert result.status == RefutationStatus.SKIPPED
        assert result.execution_time_ms > 0.0


class TestRatioHasNoFloor:
    """Codex iter-1 F1: ``max(width, 1e-10)`` understated the ratio for a tiny
    but valid interval (width 1e-12, bootstrap width 2e-11 -> 0.2 PASSED where
    the true ratio is 20). Once the width is validated finite and positive the
    divisor is the ACTUAL width, so the verdict is invariant under rescaling."""

    def test_failed_case_stays_failed_when_rescaled_by_1e_minus_12(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 40}})
        reference = _bootstrap(runner, _sequence_estimate(_bootstrap_values(0.180)))
        assert reference.status == RefutationStatus.FAILED

        scaled_values = [v * 1e-12 for v in _bootstrap_values(0.180)]
        result = runner._run_bootstrap_test(
            original_effect=0.15e-12,
            original_ci=(1e-13, 2e-13),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_sequence_estimate(scaled_values, value=0.15e-12),
            use_dowhy=True,
        )
        assert result.details["ci_ratio"] == pytest.approx(1.8, rel=1e-6)
        assert result.status == RefutationStatus.FAILED


class TestNonFiniteReferenceIntervalFailsClosed:
    """Codex iter-1 F2: a non-finite endpoint is not a value (same class as a
    NaN re-fit -> fail-closed, spec §5; defense-in-depth behind the node's own
    refusal), unlike a finite zero-width interval which is an honest SKIPPED."""

    @staticmethod
    def _never_called(_df):
        raise AssertionError("re-fit must not run for a non-finite original_ci")

    _CIS = [
        (float("-inf"), float("inf")),
        (float("nan"), 0.2),
        (0.1, float("inf")),
    ]

    @pytest.mark.parametrize("ci", _CIS)
    def test_subset_non_finite_ci_raises_before_any_refit(self, ci):
        with pytest.raises(RefutationError) as ei:
            RefutationRunner()._run_data_subset_test(
                original_effect=0.15,
                original_ci=ci,
                causal_model=_make_stub_causal_model({}),
                identified_estimand=object(),
                estimate=_stub_estimate(effect_fn=self._never_called),
                use_dowhy=True,
            )
        assert ei.value.details["reason"] == "original_ci_non_finite"
        assert ei.value.details["test_name"] == "data_subset"
        assert "original_ci" in ei.value.details

    @pytest.mark.parametrize("ci", _CIS)
    def test_bootstrap_non_finite_ci_raises_before_any_refit(self, ci):
        with pytest.raises(RefutationError) as ei:
            RefutationRunner()._run_bootstrap_test(
                original_effect=0.15,
                original_ci=ci,
                causal_model=_make_stub_causal_model({}),
                identified_estimand=object(),
                estimate=_stub_estimate(effect_fn=self._never_called),
                use_dowhy=True,
            )
        assert ei.value.details["reason"] == "original_ci_non_finite"
        assert ei.value.details["test_name"] == "bootstrap"
        assert "original_ci" in ei.value.details


def _exact_percentile_values(upper: float) -> List[float]:
    """41 sorted values whose 2.5th / 97.5th percentiles are EXACTLY the second
    and fortieth (positions 0.025*40 = 1.0 and 0.975*40 = 39.0, no
    interpolation): lower = 0.0, upper = ``upper``."""
    x = [-0.01, 0.0] + [0.01 * k for k in range(2, 39)] + [upper, upper + 0.01]
    assert len(x) == 41 and x == sorted(x)
    return x


class TestExactBoundaries:
    """Spec §6: the thresholds at their exact boundaries (0.79 / 0.80 coverage,
    1.50 / 1.51 and 1.75 / 1.76 width ratio) and bootstrap seed reproducibility."""

    @pytest.mark.parametrize(
        "inside,expected_cov,expected",
        [(80, 0.80, RefutationStatus.PASSED), (79, 0.79, RefutationStatus.WARNING)],
    )
    def test_coverage_boundary_exact(self, inside, expected_cov, expected):
        runner = RefutationRunner(config={"data_subset": {"num_subsets": 100}})
        values = [0.15 + 0.0001 * i for i in range(inside)] + [
            0.5 + 0.001 * i for i in range(100 - inside)
        ]
        result = _subset(runner, _sequence_estimate(values))
        assert result.details["ci_coverage"] == pytest.approx(expected_cov)
        assert result.details["ci_coverage"] == inside / 100
        assert result.status == expected

    @pytest.mark.parametrize(
        "upper,expected_ratio,expected",
        [
            (0.75, 1.50, RefutationStatus.PASSED),
            (0.755, 1.51, RefutationStatus.WARNING),
            (0.875, 1.75, RefutationStatus.WARNING),
            (0.88, 1.76, RefutationStatus.FAILED),
        ],
    )
    def test_ratio_boundary_exact(self, upper, expected_ratio, expected):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 41}})
        result = runner._run_bootstrap_test(
            original_effect=0.25,
            original_ci=(0.0, 0.5),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_sequence_estimate(_exact_percentile_values(upper), value=0.25),
            use_dowhy=True,
        )
        assert result.details["bootstrap_ci"] == (0.0, upper)
        assert result.details["ci_ratio"] == pytest.approx(expected_ratio)
        assert result.status == expected

    def test_bootstrap_seeded_runs_reproduce_their_evidence(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 12}})
        a = _bootstrap(runner, _stub_estimate(), resample_seed=7)
        b = _bootstrap(runner, _stub_estimate(), resample_seed=7)
        c = _bootstrap(runner, _stub_estimate(), resample_seed=8)
        assert a.details["bootstrap_effects"] == b.details["bootstrap_effects"]
        assert a.details["bootstrap_effects"] != c.details["bootstrap_effects"]


_ONLY_NONCRITICAL = {
    "placebo_treatment": {"enabled": False},
    "random_common_cause": {"enabled": False},
    "sensitivity_e_value": {"enabled": False},
}


class TestRunAllTestsWiring:
    def _kw(self):
        return {
            "original_effect": 0.15,
            "original_ci": CI,
            "causal_model": _make_stub_causal_model({}),
            "identified_estimand": object(),
        }

    def test_estimate_id_seeds_the_resamples(self):
        runner = RefutationRunner(config=_ONLY_NONCRITICAL)
        a = runner.run_all_tests(estimate=_stub_estimate(), estimate_id="est-1", **self._kw())
        b = runner.run_all_tests(estimate=_stub_estimate(), estimate_id="est-1", **self._kw())
        c = runner.run_all_tests(estimate=_stub_estimate(), estimate_id="est-2", **self._kw())

        def effects(suite):
            return {
                t.test_name.value: t.details.get("subset_effects")
                or t.details.get("bootstrap_effects")
                for t in suite.tests
            }

        assert effects(a)["data_subset"] == effects(b)["data_subset"]
        assert effects(a)["data_subset"] != effects(c)["data_subset"]
        assert effects(a)["bootstrap"] == effects(b)["bootstrap"]
        assert effects(a)["bootstrap"] != effects(c)["bootstrap"]

    def test_deadline_and_seed_reach_the_loops(self, monkeypatch):
        runner = RefutationRunner(config=_ONLY_NONCRITICAL)
        seen: dict = {"_run_data_subset_test": {}, "_run_bootstrap_test": {}}

        for method in seen:
            real = getattr(runner, method)

            def spy(*args, _real=real, _method=method, **kwargs):
                seen[_method].update(kwargs)
                return _real(*args, **kwargs)

            monkeypatch.setattr(runner, method, spy)

        far = _t.monotonic() + 3600.0
        runner.run_all_tests(estimate=_stub_estimate(), deadline=far, **self._kw())
        for method in ("_run_data_subset_test", "_run_bootstrap_test"):
            assert seen[method]["deadline"] == far, method
            assert seen[method]["resample_seed"] is None, method
```

- [ ] **Step 5: Run the new file and confirm it fails for the right reason**

```bash
PY=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python
$PY -m pytest tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py -q -p no:cacheprovider -x 2>&1 | tail -15
```

Expected: FAIL. The first failure is `test_records_per_subset_effects_and_scores_coverage` with `RefutationError` (the old code looks for `refute_estimate` on the stub and finds no `data_subset_refuter` result) or `KeyError: 'subset_effects'`. `TestReviewBandArithmetic` passes already (the arithmetic is unchanged); `test_thresholds_are_the_intended_values` fails with `{'pass': 0.5, 'warning': 0.75}`.

- [ ] **Step 6: Change the bootstrap thresholds**

In `src/causal_engine/refutation_runner.py`, inside `PASS_THRESHOLDS`, replace

```python
        "bootstrap_ci_ratio": {
            "pass": 0.50,  # Bootstrap CI must not be > 50% wider than original
            "warning": 0.75,
        },
```

with

```python
        "bootstrap_ci_ratio": {
            # ratio = bootstrap_width / original_width. A stable estimate's
            # bootstrap interval is about as wide as its analytic one (ratio
            # ≈ 1; measured 1.01 and 0.81 on two live pairs, 2026-09-08). The
            # pre-lane-1 values 0.50 / 0.75 contradicted the comment beside them
            # ("must not be > 50% wider") and would have failed nearly every
            # real run; they never scored because the test was always SKIPPED.
            "pass": 1.50,  # bootstrap CI at most 50% wider than the original
            "warning": 1.75,
        },
```

- [ ] **Step 7: Add the resample helpers after `_require_p_value`**

Insert after the `_require_p_value` function (before the `ENUMS` banner):

```python
# ============================================================================
# REAL NON-CRITICAL EVIDENCE (lane 1, spec §4.1)
# ============================================================================
# DoWhy 0.14's data_subset / bootstrap refuters compute per-sample effects and
# keep only their mean and a p-value on the CausalRefutation object, so the two
# distributional tests below could never score and were recorded SKIPPED on
# every live run (96/96 measured 2026-09-08). The loops below re-fit the SAME
# reported estimator with the SAME public calls DoWhy's ``_refute_once`` uses
# and keep every effect. Reference interval: ``original_ci`` from the
# estimation node (the reported interval) -- never the reconstruction's own.

_MIN_SUBSET_RESAMPLES = 3
_MIN_BOOTSTRAP_RESAMPLES = 10


def _refit_effect_on(new_data: Any, identified_estimand: Any, estimate: Any) -> float:
    """Re-fit the reported estimator on ``new_data`` and return its effect.

    The four calls are the public estimator API DoWhy 0.14's own
    ``data_subset_refuter._refute_once`` / ``bootstrap_refuter._refute_once``
    use; nothing here substitutes a different model.
    """
    new_estimator = estimate.estimator.get_new_estimator_object(identified_estimand)
    fit_params = getattr(new_estimator, "_fit_params", None) or {}
    new_estimator.fit(
        new_data,
        effect_modifier_names=estimate.estimator._effect_modifier_names,
        **fit_params,
    )
    new_effect = new_estimator.estimate_effect(
        new_data,
        control_value=estimate.control_value,
        treatment_value=estimate.treatment_value,
        target_units=estimate.estimator._target_units,
    )
    return float(new_effect.value)


def _refutation_frame(causal_model: Any, test_name: str, original_effect: float) -> Any:
    """The frame the CausalModel was built on (DoWhy stores it as ``_data``)."""
    frame = getattr(causal_model, "_data", None)
    if frame is None or not hasattr(frame, "sample") or not hasattr(frame, "columns"):
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"{test_name} needs the CausalModel's DataFrame (``_data``) to resample; "
            "the model exposes none.",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "refutation_frame_missing",
            },
        )
    return frame


def _resample_effects(
    *,
    kind: str,
    frame: Any,
    identified_estimand: Any,
    estimate: Any,
    requested: int,
    rng: np.random.Generator,
    deadline: Optional[float],
    subset_fraction: float = 0.8,
) -> Tuple[List[float], bool]:
    """Run up to ``requested`` re-fits, stopping at the cooperative deadline.

    ``kind`` is ``"subset"`` (``frame.sample(frac=subset_fraction)``) or
    ``"bootstrap"`` (row resample WITH replacement, same size; no confounder
    noise -- DoWhy's default bootstrap refuter also perturbs the chosen
    covariates, which answers a measurement-error question, not the variance
    question this test scores). Each draw is seeded from ``rng`` so a seeded
    caller reproduces its evidence. Returns ``(effects, stopped_for_budget)``.
    """
    effects: List[float] = []
    for _ in range(max(1, int(requested))):
        if deadline is not None and time.monotonic() >= deadline:
            return effects, True
        seed = int(rng.integers(0, 2**31 - 1))
        if kind == "subset":
            new_data = frame.sample(frac=subset_fraction, random_state=seed)
        else:
            new_data = frame.sample(n=len(frame), replace=True, random_state=seed)
        effects.append(_refit_effect_on(new_data, identified_estimand, estimate))
    return effects, False


def _significance_p_value(
    estimate: Any, effects: List[float], test_name: str, original_effect: float
) -> float:
    """p-value of the reported estimate under the resample distribution --
    DoWhy's own ``test_significance`` (the refuters' p-value), kept real."""
    try:
        from dowhy.causal_refuter import test_significance
    except ImportError as ie:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "DoWhy import failed while scoring resample evidence.",
            details={"test_name": test_name, "reason": "dowhy_import_failed"},
            original_error=ie,
        ) from ie
    result = test_significance(estimate, np.asarray(effects, dtype=float))
    pv = result.get("p_value") if isinstance(result, dict) else None
    if pv is None or not np.isfinite(float(pv)):
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"{test_name} significance test returned no finite p_value; refusing to "
            "substitute a placeholder.",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "missing_p_value",
            },
        )
    return float(pv)


def _require_finite_effects(effects: List[float], test_name: str, original_effect: float) -> None:
    """Fail closed on a NaN / inf re-fit (spec §5: an anomaly inside the loop is
    treated like an exception). A non-finite effect would otherwise be SCORED:
    DoWhy's percentile test counts NaN as "below the estimate", np.percentile
    poisons the bootstrap interval, and coverage silently drops one sample."""
    for i, e in enumerate(effects):
        if not np.isfinite(e):
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                f"{test_name} re-fit #{i} returned a non-finite effect ({e!r}); refusing "
                "to score a distribution that contains it.",
                details={
                    "test_name": test_name,
                    "original_effect": original_effect,
                    "reason": "non_finite_resample_effect",
                    "resamples_completed": len(effects),
                    "first_non_finite_index": i,
                },
            )


def _budget_skip_result(
    test_name: RefutationTestType,
    original_effect: float,
    completed: int,
    requested: int,
    minimum: int,
    stopped: bool,
    config_details: Dict[str, Any],
    execution_time_ms: float = 0.0,
) -> RefutationResult:
    """Honest SKIPPED when fewer than ``minimum`` re-fits completed.

    ``stopped`` says WHY: the deadline stopped the loop (``time_budget``, same
    ``reason`` / ``message`` contract as the #1419 pre-start skip) or the loop
    ran to completion because the configured count is below the minimum
    (``config_below_minimum``) -- a skip must not blame the budget when the
    budget was never hit.
    """
    name = test_name.value
    if stopped:
        reason = (
            "time_budget — non-critical test stopped before its minimum resample "
            "count; the critical gates decide the suite"
        )
        message = (
            f"{name} skipped: {completed}/{requested} resamples completed before the "
            f"compute deadline (minimum {minimum}); non-critical, degraded honestly"
        )
    else:
        reason = (
            "config_below_minimum — requested resample count is below the test's "
            "minimum; the critical gates decide the suite"
        )
        message = (
            f"{name} skipped: {completed}/{requested} resamples requested, below the "
            f"minimum {minimum}; non-critical, degraded honestly"
        )
    return RefutationResult(
        test_name=test_name,
        status=RefutationStatus.SKIPPED,
        original_effect=original_effect,
        refuted_effect=original_effect,
        details={
            "reason": reason,
            "message": message,
            "resamples_completed": completed,
            "resamples_requested": requested,
            "stopped_for_budget": stopped,
            **config_details,
        },
        execution_time_ms=execution_time_ms,
    )


def _degenerate_skip_result(
    test_name: RefutationTestType,
    original_effect: float,
    effects: List[float],
    requested: int,
    stopped: bool,
    config_details: Dict[str, Any],
    execution_time_ms: float = 0.0,
) -> RefutationResult:
    """Honest SKIPPED when every re-fit returned the SAME effect (owner decision
    2026-09-09). A zero-variance distribution cannot be scored (DoWhy's normal
    test divides by its standard deviation) and a constant re-fit is not
    evidence of instability: an estimator that ignores its data fails the
    CRITICAL placebo test, which decides the suite. Never a placeholder p-value,
    never a fail-closed halt from a non-critical test."""
    name = test_name.value
    return RefutationResult(
        test_name=test_name,
        status=RefutationStatus.SKIPPED,
        original_effect=original_effect,
        refuted_effect=float(effects[0]),
        details={
            "reason": (
                "degenerate_resample_distribution — every re-fit returned the same "
                "effect; a zero-variance distribution cannot be scored; the critical "
                "gates decide the suite"
            ),
            "message": (
                f"{name} skipped: {len(effects)} re-fits all returned "
                f"{float(effects[0]):.6g}; non-critical, degraded honestly"
            ),
            "resample_effects": [float(e) for e in effects],
            "resamples_completed": len(effects),
            "resamples_requested": requested,
            "stopped_for_budget": stopped,
            **config_details,
        },
        execution_time_ms=execution_time_ms,
    )


def _require_finite_ci(
    original_ci: Tuple[float, float], test_name: str, original_effect: float
) -> None:
    """Fail closed when either endpoint of the reported interval is not finite.

    A non-finite endpoint is not a value: it is the same class as a NaN re-fit
    (spec §5, fail-closed) and would otherwise be SCORED -- ``(-inf, inf)``
    covers every subset effect (coverage 1.0) and makes the width ratio 0, a
    PASSED verdict without a usable interval; a NaN endpoint fails from NaN
    arithmetic and blames the estimate. The node already refuses such an
    interval upstream (nodes/refutation.py), so this is defense-in-depth. A
    FINITE zero-width interval is different: it is a real value that merely
    cannot score a coverage / width test, the same class as a degenerate
    resample distribution, and stays an honest SKIPPED
    (``_degenerate_ci_skip_result``).
    """
    lo, hi = original_ci[0], original_ci[1]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"{test_name} received a non-finite reference interval {tuple(original_ci)!r}; "
            "refusing to score against it.",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "original_ci_non_finite",
                "original_ci": (float(lo), float(hi)),
            },
        )


def _degenerate_ci_skip_result(
    test_name: RefutationTestType,
    original_effect: float,
    original_ci: Tuple[float, float],
    config_details: Dict[str, Any],
    execution_time_ms: float = 0.0,
) -> RefutationResult:
    """Honest SKIPPED, decided BEFORE any re-fit, when the reported interval has
    no width: coverage of a point and a width ratio against ~0 cannot be scored
    and would blame the estimate for an upstream degenerate interval."""
    name = test_name.value
    return RefutationResult(
        test_name=test_name,
        status=RefutationStatus.SKIPPED,
        original_effect=original_effect,
        refuted_effect=original_effect,
        details={
            "reason": (
                "original_ci_degenerate — the reported interval has no width, so "
                "coverage / width ratio cannot be scored; the critical gates decide "
                "the suite"
            ),
            "message": (
                f"{name} skipped: original_ci={tuple(original_ci)!r} has width "
                f"{float(original_ci[1] - original_ci[0]):.6g}; no re-fit was run; "
                "non-critical, degraded honestly"
            ),
            "original_ci": (float(original_ci[0]), float(original_ci[1])),
            "resamples_completed": 0,
            "stopped_for_budget": False,
            **config_details,
        },
        execution_time_ms=execution_time_ms,
    )


def _resample_seed_for(estimate_id: Optional[str]) -> Optional[int]:
    """Stable 31-bit seed from the estimate id (``None`` → unseeded, as before).

    The first 8 hex digits of the digest are 32 bits (measured max 4294943764
    over the live ids, 2026-09-09); the mask keeps the promise in this docstring.
    """
    if not estimate_id:
        return None
    import hashlib

    return int(hashlib.sha256(str(estimate_id).encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
```

- [ ] **Step 8: Replace `_run_data_subset_test`**

Replace the whole method (from `def _run_data_subset_test(` through its `return RefutationResult(...)`) with:

```python
    def _run_data_subset_test(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        causal_model: Optional[Any],
        identified_estimand: Optional[Any],
        estimate: Optional[Any],
        use_dowhy: bool,
        *,
        deadline: Optional[float] = None,
        resample_seed: Optional[int] = None,
    ) -> RefutationResult:
        """Data-subset consistency test on REAL per-subset evidence (spec §4.1).

        Re-fits the reported estimator on ``num_subsets`` random subsets of
        ``subset_fraction`` of the model's frame and scores the SHARE of subset
        effects that fall inside ``original_ci`` (the estimation node's reported
        interval). Stops at ``deadline`` between re-fits; below
        ``_MIN_SUBSET_RESAMPLES`` completed it returns an honest SKIPPED.
        """
        import time

        start_time = time.time()
        test_name = RefutationTestType.DATA_SUBSET

        if not (use_dowhy and causal_model is not None):
            # F-014 fail-closed: defense-in-depth for legacy non-agent callers.
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                "data_subset test requires a real DoWhy CausalModel; "
                "caller passed causal_model=None.",
                details={
                    "test_name": "data_subset",
                    "dowhy_available": DOWHY_AVAILABLE,
                    "original_effect": original_effect,
                },
            )

        cfg = self.config["data_subset"]
        requested = int(cfg["num_subsets"])
        subset_fraction = float(cfg["subset_fraction"])
        config_details = {"subset_fraction": subset_fraction, "num_subsets": requested}
        frame = _refutation_frame(causal_model, "data_subset", original_effect)
        _require_finite_ci(original_ci, "data_subset", original_effect)
        if original_ci[1] - original_ci[0] <= 0:
            return _degenerate_ci_skip_result(
                test_name,
                original_effect,
                original_ci,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        rng = np.random.default_rng(resample_seed)
        try:
            subset_effects, stopped = _resample_effects(
                kind="subset",
                frame=frame,
                identified_estimand=identified_estimand,
                estimate=estimate,
                requested=requested,
                rng=rng,
                deadline=deadline,
                subset_fraction=subset_fraction,
            )
        except RefutationError:
            raise
        except Exception as e:
            # F-014 fail-closed: no silent mock fallback.
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                f"data_subset re-fit failed: {e}",
                details={"test_name": "data_subset", "original_effect": original_effect},
                original_error=e,
            ) from e

        _require_finite_effects(subset_effects, "data_subset", original_effect)
        if len(subset_effects) < _MIN_SUBSET_RESAMPLES:
            return _budget_skip_result(
                test_name,
                original_effect,
                len(subset_effects),
                requested,
                _MIN_SUBSET_RESAMPLES,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # "Every re-fit returned the same effect" is tested EXACTLY (max == min):
        # np.std of n identical floats is not 0.0 for most n (twelve 0.15s give
        # 2.8e-17, measured 2026-09-09), which would let DoWhy's normal test
        # score a constant series with a meaningless p-value.
        if float(np.ptp(subset_effects)) == 0.0:
            return _degenerate_skip_result(
                test_name,
                original_effect,
                subset_effects,
                requested,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        refuted_effect = float(np.mean(subset_effects))
        p_value = _significance_p_value(estimate, subset_effects, "data_subset", original_effect)
        ci_coverage = self._calculate_ci_coverage(subset_effects, original_ci)
        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )

        if ci_coverage >= self.thresholds["subset_ci_coverage"]["pass"]:
            status = RefutationStatus.PASSED
            message = f"Effect consistent across {int(ci_coverage * 100)}% of data subsets"
        elif ci_coverage >= self.thresholds["subset_ci_coverage"]["warning"]:
            status = RefutationStatus.WARNING
            message = f"Effect varies in {int((1 - ci_coverage) * 100)}% of subsets"
        else:
            status = RefutationStatus.FAILED
            message = f"WARNING: Effect inconsistent across data subsets ({int(ci_coverage * 100)}% coverage)"

        execution_time = (time.time() - start_time) * 1000
        return RefutationResult(
            test_name=test_name,
            status=status,
            original_effect=original_effect,
            refuted_effect=refuted_effect,
            p_value=p_value,
            delta_percent=delta_percent,
            details={
                "message": message,
                "ci_coverage": ci_coverage,
                "subset_effects": [float(e) for e in subset_effects],
                "resamples_completed": len(subset_effects),
                "resamples_requested": requested,
                "stopped_for_budget": stopped,
                **config_details,
            },
            execution_time_ms=execution_time,
        )
```

- [ ] **Step 9: Replace `_run_bootstrap_test`**

Replace the whole method with:

```python
    def _run_bootstrap_test(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        causal_model: Optional[Any],
        identified_estimand: Optional[Any],
        estimate: Optional[Any],
        use_dowhy: bool,
        *,
        deadline: Optional[float] = None,
        resample_seed: Optional[int] = None,
    ) -> RefutationResult:
        """Bootstrap stability test on REAL per-resample evidence (spec §4.1).

        Re-fits the reported estimator on ``num_bootstraps`` row resamples (with
        replacement, same size) of the model's frame; the 2.5th–97.5th
        percentile width of the resample effects is compared with the width of
        ``original_ci``. Thresholds: pass ≤ 1.5×, warning ≤ 1.75×, else failed
        (``PASS_THRESHOLDS["bootstrap_ci_ratio"]``). Stops at ``deadline``
        between re-fits; below ``_MIN_BOOTSTRAP_RESAMPLES`` it returns SKIPPED.
        """
        import time

        start_time = time.time()
        test_name = RefutationTestType.BOOTSTRAP

        if not (use_dowhy and causal_model is not None):
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                "bootstrap test requires a real DoWhy CausalModel; "
                "caller passed causal_model=None.",
                details={
                    "test_name": "bootstrap",
                    "dowhy_available": DOWHY_AVAILABLE,
                    "original_effect": original_effect,
                },
            )

        requested = int(self.config["bootstrap"]["num_bootstraps"])
        config_details = {"num_bootstraps": requested}
        frame = _refutation_frame(causal_model, "bootstrap", original_effect)
        _require_finite_ci(original_ci, "bootstrap", original_effect)
        if original_ci[1] - original_ci[0] <= 0:
            return _degenerate_ci_skip_result(
                test_name,
                original_effect,
                original_ci,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        rng = np.random.default_rng(resample_seed)
        try:
            bootstrap_effects, stopped = _resample_effects(
                kind="bootstrap",
                frame=frame,
                identified_estimand=identified_estimand,
                estimate=estimate,
                requested=requested,
                rng=rng,
                deadline=deadline,
            )
        except RefutationError:
            raise
        except Exception as e:
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                f"bootstrap re-fit failed: {e}",
                details={"test_name": "bootstrap", "original_effect": original_effect},
                original_error=e,
            ) from e

        _require_finite_effects(bootstrap_effects, "bootstrap", original_effect)
        if len(bootstrap_effects) < _MIN_BOOTSTRAP_RESAMPLES:
            return _budget_skip_result(
                test_name,
                original_effect,
                len(bootstrap_effects),
                requested,
                _MIN_BOOTSTRAP_RESAMPLES,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Exact degeneracy check (max == min); see _run_data_subset_test.
        if float(np.ptp(bootstrap_effects)) == 0.0:
            return _degenerate_skip_result(
                test_name,
                original_effect,
                bootstrap_effects,
                requested,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        refuted_effect = float(np.mean(bootstrap_effects))
        p_value = _significance_p_value(estimate, bootstrap_effects, "bootstrap", original_effect)
        bootstrap_ci = (
            float(np.percentile(bootstrap_effects, 2.5)),
            float(np.percentile(bootstrap_effects, 97.5)),
        )
        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )
        original_ci_width = original_ci[1] - original_ci[0]
        bootstrap_ci_width = bootstrap_ci[1] - bootstrap_ci[0]
        # The width is finite and > 0 here (_require_finite_ci + the widthless
        # guard above), so divide by the ACTUAL width: a floor (formerly 1e-10)
        # understated the ratio for a tiny but valid interval (codex iter-1 F1:
        # width 1e-12, bootstrap width 2e-11 read 0.2 PASSED; true ratio 20).
        ci_ratio = bootstrap_ci_width / original_ci_width

        if ci_ratio <= self.thresholds["bootstrap_ci_ratio"]["pass"]:
            status = RefutationStatus.PASSED
            message = f"Effect stable across {len(bootstrap_effects)} bootstrap samples"
        elif ci_ratio <= self.thresholds["bootstrap_ci_ratio"]["warning"]:
            status = RefutationStatus.WARNING
            message = "Bootstrap CI moderately wider than original"
        else:
            status = RefutationStatus.FAILED
            message = "WARNING: High variance in bootstrap estimates"

        execution_time = (time.time() - start_time) * 1000
        return RefutationResult(
            test_name=test_name,
            status=status,
            original_effect=original_effect,
            refuted_effect=refuted_effect,
            p_value=p_value,
            delta_percent=delta_percent,
            details={
                "message": message,
                "bootstrap_ci": bootstrap_ci,
                "ci_ratio": ci_ratio,
                "bootstrap_ci_available": True,
                "bootstrap_effects": [float(e) for e in bootstrap_effects],
                "resamples_completed": len(bootstrap_effects),
                "resamples_requested": requested,
                "stopped_for_budget": stopped,
                **config_details,
            },
            execution_time_ms=execution_time,
        )
```

- [ ] **Step 10: Run the runner test files**

```bash
$PY -m pytest tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_1419.py tests/unit/test_causal_engine/test_refutation_runner_randomized.py -q -p no:cacheprovider 2>&1 | tail -8
```

Expected: all PASS except the two `TestRunAllTestsWiring` cases, which do not exist yet (Task 2). If `test_bootstrap_default_bootstraps_bounded` or `test_data_subset_default_subsets_bounded` fail, read them: they pin DEFAULT_CONFIG counts, which this task did not change.

- [ ] **Step 11: Lint and type-check the changed file, then commit**

```bash
$PY -m ruff check src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/ && $PY -m ruff format --check src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py
$PY -m mypy --config-file pyproject.toml src/causal_engine/refutation_runner.py
git add src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/
git commit -m "feat(refutation): real data_subset/bootstrap evidence; intended bootstrap thresholds (spec §4.1)

Both tests ran DoWhy's refuter and discarded the result (per-sample effects
are not on the refutation object): 96/96 live runs recorded them SKIPPED.
They now re-fit the reported estimator in their own resample loops (same
public calls as DoWhy's _refute_once), keep every effect, score coverage /
width ratio against the reported interval, use DoWhy's test_significance
for the p-value, stop at the cooperative deadline, and seed their draws.
bootstrap_ci_ratio thresholds 0.50/0.75 -> 1.50/1.75 (the code contradicted
its own comment; measured ratios 1.01 and 0.81 on live pairs).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

---

### Task 2: Runner — `run_all_tests` passes the deadline and a stable seed

**Files:**
- Modify: `src/causal_engine/refutation_runner.py` (`run_all_tests`, the two `_run_test_with_tracing` calls for data_subset and bootstrap ~lines 748–790)
- Modify: `tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py`

- [ ] **Step 1: Add the failing wiring tests**

Append to `test_refutation_runner_real_evidence.py`:

```python
_ONLY_NONCRITICAL = {
    "placebo_treatment": {"enabled": False},
    "random_common_cause": {"enabled": False},
    "sensitivity_e_value": {"enabled": False},
}


class TestRunAllTestsWiring:
    def _kw(self):
        return dict(
            original_effect=0.15,
            original_ci=CI,
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
        )

    def test_estimate_id_seeds_the_resamples(self):
        runner = RefutationRunner(config=_ONLY_NONCRITICAL)
        a = runner.run_all_tests(estimate=_stub_estimate(), estimate_id="est-1", **self._kw())
        b = runner.run_all_tests(estimate=_stub_estimate(), estimate_id="est-1", **self._kw())
        c = runner.run_all_tests(estimate=_stub_estimate(), estimate_id="est-2", **self._kw())

        def effects(suite):
            return {t.test_name.value: t.details.get("subset_effects") for t in suite.tests}

        assert effects(a)["data_subset"] == effects(b)["data_subset"]
        assert effects(a)["data_subset"] != effects(c)["data_subset"]

    def test_deadline_and_seed_reach_the_loops(self, monkeypatch):
        runner = RefutationRunner(config=_ONLY_NONCRITICAL)
        seen: dict = {}
        real = runner._run_data_subset_test

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(runner, "_run_data_subset_test", spy)
        far = _t.monotonic() + 3600.0
        runner.run_all_tests(estimate=_stub_estimate(), deadline=far, **self._kw())
        assert seen["deadline"] == far
        assert seen["resample_seed"] is None
```

- [ ] **Step 2: Run them to see the failure**

```bash
$PY -m pytest tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py::TestRunAllTestsWiring -q -p no:cacheprovider 2>&1 | tail -6
```

Expected: FAIL — `KeyError: 'deadline'` in the spy test; the seed test fails because two unseeded runs differ.

- [ ] **Step 3: Wire the two calls**

In `run_all_tests`, directly after the `_record` inner function definition, add:

```python
        # Spec §4.1: the two resample loops check the deadline between re-fits
        # and seed their draws from the estimate id so a re-run reproduces its
        # evidence (None → unseeded, the pre-lane-1 behaviour).
        resample_seed = _resample_seed_for(estimate_id)
```

In the `data_subset` block, change the `_run_test_with_tracing(` call to add two kwargs after `use_dowhy=use_dowhy,`:

```python
                    use_dowhy=use_dowhy,
                    deadline=deadline,
                    resample_seed=resample_seed,
                )
```

Do the same in the `bootstrap` block.

- [ ] **Step 4: Run all runner tests, lint, commit**

```bash
$PY -m pytest tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_1419.py tests/unit/test_causal_engine/test_refutation_runner_randomized.py -q -p no:cacheprovider 2>&1 | tail -4
$PY -m ruff check src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py
git add -A src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py
git commit -m "feat(refutation): thread the cooperative deadline and an estimate-id seed into the resample loops

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

Expected: all green.

---

### Task 2b: Evidence writer — JSON objects in `details_json` / `test_config`, backfill migration 135

**Files:**
- Create: `src/repositories/json_utils.py` (`json_default`, `to_plain_json` — moved from `src/repositories/discovered_dag.py`, which imports them back)
- Modify: `src/repositories/causal_validation.py` (`save_single_test` ~lines 199–200; `_test_to_row` ~lines 497–502)
- Modify: `tests/unit/test_repositories/test_causal_validation.py` (new class)
- Create: `database/migrations/135_causal_validations_json_objects.sql`, `tests/unit/test_database/test_migration_135_json_objects.py`

Why (measured 2026-09-08/09): both writer sites `json.dumps(...)` into the jsonb columns, a pattern that dates from the original RefutationRunner commit `0742b81f6`; the client already serialises a dict as a JSON object (the DGP seed of migration 119 wrote objects, which is why 545 rows are objects). Every agent-path row — 480 live rows, all of `estimate_source = causal_impact_query` — is therefore a JSON *string*: `details_json->>'message'` is NULL on them and the lane's new per-resample arrays would be unqueryable (`jsonb_array_length(details_json->'subset_effects')`). Readers today decode both shapes (`src/api/routes/chatbot_tools.py:_details`, the expert-review schema validators). Owner decision 2026-09-09 (decision 6): fix the writer in this lane so evidence is written, backfilled and tested in ONE shape. The same pattern on `expert_reviews.agent_assessment_json` (4 string rows) and `checklist_json` (1) is out of this lane — Task 16 (e) files it.

- [ ] **Step 1: Failing tests** — append to `tests/unit/test_repositories/test_causal_validation.py`:

```python
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)


class TestEvidenceRowsAreJsonObjects:
    """Lane 1 (owner decision 2026-09-09): evidence is written as JSON OBJECTS,
    not JSON strings inside the jsonb column, so it can be queried and tested in
    one shape; non-finite floats become null (the transport encodes with
    allow_nan=False, so a NaN would otherwise fail the whole write)."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def repo(self, mock_client):
        repo = CausalValidationRepository()
        repo.client = mock_client
        return repo

    @staticmethod
    def _suite() -> RefutationSuite:
        test = RefutationResult(
            test_name=RefutationTestType.BOOTSTRAP,
            status=RefutationStatus.PASSED,
            original_effect=0.15,
            refuted_effect=0.151,
            p_value=0.4,
            details={
                "message": "ok",
                "bootstrap_effects": [0.14, 0.16],
                "ci_ratio": float("nan"),
                "config": {"n": 2},
            },
            execution_time_ms=12.5,
        )
        return RefutationSuite(
            passed=True,
            confidence_score=0.9,
            tests=[test],
            gate_decision=GateDecision.PROCEED,
            estimate_id="est-1",
            brand="Kisqali",
        )

    @pytest.mark.asyncio
    async def test_save_suite_writes_objects_not_strings(self, repo, mock_client):
        insert = mock_client.table.return_value.insert
        insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"validation_id": "v1"}]))
        ids = await repo.save_suite(self._suite(), estimate_id="e1")
        assert ids == ["v1"]
        row = insert.call_args[0][0][0]
        assert isinstance(row["details_json"], dict)
        assert isinstance(row["test_config"], dict)
        assert row["details_json"]["bootstrap_effects"] == [0.14, 0.16]
        assert row["details_json"]["ci_ratio"] is None  # NaN -> null, never a transport failure
        assert row["test_config"] == {"execution_time_ms": 12.5}

    @pytest.mark.asyncio
    async def test_save_single_test_writes_objects_not_strings(self, repo, mock_client):
        insert = mock_client.table.return_value.insert
        insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"validation_id": "v2"}]))
        suite = self._suite()
        vid = await repo.save_single_test(
            suite.tests[0], estimate_id="e1", gate_decision=GateDecision.PROCEED, confidence_score=0.9
        )
        assert vid == "v2"
        row = insert.call_args[0][0]
        assert isinstance(row["details_json"], dict)
        assert row["details_json"]["ci_ratio"] is None
        assert row["test_config"] == {"n": 2}
```

And create `tests/unit/test_database/test_migration_135_json_objects.py`:

```python
"""Migration 135 decodes string-shaped evidence rows into JSON objects (lane 1, owner decision 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "135_causal_validations_json_objects.sql"


@pytest.mark.unit
def test_backfills_both_columns_and_asserts_none_remain():
    sql = MIGRATION.read_text(encoding="utf-8")
    for col in ("details_json", "test_config"):
        assert f"SET {col} = ({col} #>> '{{}}')::jsonb WHERE jsonb_typeof({col}) = 'string'" in sql
    assert "RAISE EXCEPTION 'migration 135: string-shaped evidence rows remain'" in sql


@pytest.mark.unit
def test_no_constraint_that_would_break_the_old_image_during_the_deploy_swap():
    """A CHECK (jsonb_typeof = 'object') would make the OLD writer fail between the
    migration run and the container flip; the writer fix + this backfill are the
    guarantee, and Task 14 certifies zero string rows after the deploy."""
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "ALTER TABLE" not in sql
    assert "CHECK (" not in sql
```

Run: `$PY -m pytest tests/unit/test_repositories/test_causal_validation.py tests/unit/test_database/test_migration_135_json_objects.py -q -p no:cacheprovider` → Expected: the two writer tests FAIL on `isinstance(row["details_json"], dict)` (today it is a `str`), the migration tests FAIL with `FileNotFoundError`.

- [ ] **Step 2: Shared sanitiser** — create `src/repositories/json_utils.py` by MOVING `_json_default` and `_to_plain_json` out of `src/repositories/discovered_dag.py` (public names `json_default`, `to_plain_json`, docstrings unchanged, `import json` / `numpy as np` as needed), and in `discovered_dag.py` replace the two definitions with

```python
from src.repositories.json_utils import json_default as _json_default  # noqa: F401  (kept name)
from src.repositories.json_utils import to_plain_json as _to_plain_json
```

so its call sites and tests are untouched. Run `$PY -m pytest tests/unit/test_repositories/test_discovered_dag*.py -q -p no:cacheprovider` → Expected: unchanged green.

- [ ] **Step 3: The writer** — in `src/repositories/causal_validation.py` add `from src.repositories.json_utils import to_plain_json` and replace, at BOTH sites,

```python
            "test_config": json.dumps(test.details.get("config", {})),
            "details_json": json.dumps(test.details),
```
with
```python
            # Lane 1 (owner decision 2026-09-09): JSON OBJECTS, not JSON strings,
            # so evidence is queryable (jsonb_array_length(details_json->'subset_effects'))
            # and testable in one shape; non-finite floats -> null (allow_nan=False transport).
            "test_config": to_plain_json(test.details.get("config", {})),
            "details_json": to_plain_json(test.details),
```
and in `_test_to_row`
```python
            "test_config": json.dumps(
                {
                    "execution_time_ms": test.execution_time_ms,
                }
            ),
            "details_json": json.dumps(test.details),
```
with
```python
            "test_config": to_plain_json({"execution_time_ms": test.execution_time_ms}),
            "details_json": to_plain_json(test.details),
```
Drop the now-unused `import json` if ruff reports it.

- [ ] **Step 4: Migration 135** — create `database/migrations/135_causal_validations_json_objects.sql`:

```sql
-- ============================================================================
-- Migration 135: causal_validations evidence columns hold JSON OBJECTS (lane 1)
-- ============================================================================
-- WHAT: decode every row whose details_json / test_config is a JSON *string*
--   into the object it encodes. The Python writer json.dumps'ed into the jsonb
--   column since 0742b81f6, so every agent-path row was a string (480 live rows
--   on 2026-09-08, all estimate_source = causal_impact_query); the 545 rows the
--   migration-119 DGP seed wrote were already objects. Idempotent: a re-run
--   finds nothing to decode.
-- WHY: the lane's per-resample evidence must be queryable
--   (jsonb_array_length(details_json->'subset_effects')) and testable in ONE
--   shape (owner decision 2026-09-09).
-- SAFETY: pure data fix, no DDL, deliberately NO CHECK constraint -- migrations
--   run before the container flips, and a constraint would make the OLD image's
--   writer fail during that window. The writer fix ships in the same deploy;
--   Task 14 certifies zero string rows afterwards. Rehearsed BEGIN/ROLLBACK
--   2026-09-09: 480 -> 0 string rows in both columns, every decoded row readable.
-- ============================================================================

UPDATE public.causal_validations
   SET details_json = (details_json #>> '{}')::jsonb
 WHERE jsonb_typeof(details_json) = 'string';

UPDATE public.causal_validations
   SET test_config = (test_config #>> '{}')::jsonb
 WHERE jsonb_typeof(test_config) = 'string';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.causal_validations
         WHERE jsonb_typeof(details_json) = 'string' OR jsonb_typeof(test_config) = 'string'
    ) THEN
        RAISE EXCEPTION 'migration 135: string-shaped evidence rows remain';
    END IF;
END $$;
```

- [ ] **Step 5: Rehearse on the live database (BEGIN … ROLLBACK, applied twice)**

```bash
{ echo 'BEGIN;'; cat database/migrations/135_causal_validations_json_objects.sql; cat database/migrations/135_causal_validations_json_objects.sql
  echo "SELECT 'after', jsonb_typeof(details_json), jsonb_typeof(test_config), count(*) FROM public.causal_validations GROUP BY 2,3;"
  echo "SELECT 'readable', count(*) FILTER (WHERE details_json ? 'message'), count(*) FROM public.causal_validations WHERE estimate_source='causal_impact_query';"
  echo 'ROLLBACK;'; } | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 -tA -F' | ' 2>&1 | tail -5
docker exec supabase-db psql -U postgres -d postgres -tA -c "select jsonb_typeof(details_json), count(*) from public.causal_validations group by 1"
```

Expected (measured 2026-09-09 with the same statements): `UPDATE 480`, `UPDATE 480`, then `UPDATE 0`, `UPDATE 0` (second apply), `after | object | object | 1025`, `readable | 480 | 480`, `ROLLBACK`; and after the rollback still `string|480`, `object|545`.

- [ ] **Step 6: Run, lint, commit**

```bash
$PY -m pytest tests/unit/test_repositories/test_causal_validation.py tests/unit/test_repositories/test_discovered_dag*.py tests/unit/test_database/test_migration_135_json_objects.py tests/unit/test_api/test_chatbot_causal_validation_provenance.py -q -p no:cacheprovider 2>&1 | tail -3
$PY -m ruff check src/repositories/causal_validation.py src/repositories/json_utils.py src/repositories/discovered_dag.py tests/unit/test_repositories/test_causal_validation.py tests/unit/test_database/test_migration_135_json_objects.py && $PY -m ruff format --check src/repositories/causal_validation.py src/repositories/json_utils.py src/repositories/discovered_dag.py
$PY -m mypy --config-file pyproject.toml src/repositories/causal_validation.py src/repositories/json_utils.py src/repositories/discovered_dag.py
git add src/repositories/json_utils.py src/repositories/causal_validation.py src/repositories/discovered_dag.py tests/unit/test_repositories/test_causal_validation.py database/migrations/135_causal_validations_json_objects.sql tests/unit/test_database/test_migration_135_json_objects.py
git commit -m "fix(evidence): write causal_validations details_json/test_config as JSON objects; migration 135 backfills the 480 string-shaped rows

The writer json.dumps'ed into the jsonb columns since 0742b81f6, so every
agent-path evidence row was a JSON string and its keys unqueryable. Shared
sanitiser (non-finite -> null) moved to json_utils; readers already accept
both shapes. Owner decision 2026-09-09: fix in-lane so evidence is written,
backfilled and tested in one shape.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

The `db_tests` reader in Task 13 keeps its decode of both shapes: the baseline runs on the OLD image, whose rows are still strings; after the deploy every row is an object (Task 14 certifies it).

---

### Task 3: Migration 134 — guarded promote in SQL

**Files:**
- Create: `database/migrations/134_guarded_causal_path_promote.sql`
- Create: `tests/unit/test_database/test_migration_134_guarded_promote.py`

- [ ] **Step 1: Write the failing contract test**

```python
"""Migration 134 ships the guarded causal_paths promote (lane 1, spec §4.3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "134_guarded_causal_path_promote.sql"

FUNCTIONS = (
    "public.dag_structure_rejected(text, text)",
    "public.promote_causal_path_guarded(text, text, text[], text, text)",
)


@pytest.mark.unit
def test_migration_defines_both_functions():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "CREATE OR REPLACE FUNCTION public.dag_structure_rejected(" in sql
    assert "CREATE OR REPLACE FUNCTION public.promote_causal_path_guarded(" in sql


@pytest.mark.unit
def test_rejection_is_evaluated_inside_the_update_statement():
    """The whole point: no window between reading the verdict and writing the status."""
    sql = MIGRATION.read_text(encoding="utf-8")
    update = re.search(r"UPDATE public\.causal_paths.*?;", sql, re.S)
    assert update is not None
    assert "NOT public.dag_structure_rejected(" in update.group(0)
    assert "validation_status = ANY (p_allowed_current)" in update.group(0)


@pytest.mark.unit
def test_promote_takes_a_table_share_lock_before_the_update():
    """A rejection racing the promote must either be seen by the UPDATE or wait
    for it; the STABLE predicate alone leaves a statement-sized window and a
    row lock cannot cover a review row inserted meanwhile (pre-execution review
    iter-2 + iter-3, codex HIGH)."""
    sql = MIGRATION.read_text(encoding="utf-8")
    fn = sql[sql.index("CREATE OR REPLACE FUNCTION public.promote_causal_path_guarded(") :]
    lock = "LOCK TABLE public.expert_reviews IN SHARE MODE;"
    assert lock in fn
    assert fn.index(lock) < fn.index("UPDATE public.causal_paths")


@pytest.mark.unit
def test_empty_string_brand_means_no_brand():
    """``get_reviews_for_dag`` filters with ``if brand:`` -- '' is unfiltered. The
    SQL must read '' the same way or a same-hash rejection under another brand
    is missed (pre-execution review 2026-09-08, codex HIGH)."""
    sql = MIGRATION.read_text(encoding="utf-8")
    assert sql.count("NULLIF(p_brand, '') IS NULL OR") == 2
    assert "(p_brand IS NULL OR" not in sql


@pytest.mark.unit
def test_service_role_only():
    sql = MIGRATION.read_text(encoding="utf-8")
    for fn in FUNCTIONS:
        assert f"REVOKE ALL ON FUNCTION {fn} FROM PUBLIC, anon, authenticated;" in sql
        assert f"GRANT EXECUTE ON FUNCTION {fn} TO service_role;" in sql
    assert "has_function_privilege" in sql  # the migration asserts its own grants


@pytest.mark.unit
def test_created_at_is_made_not_null_so_the_chronology_is_total():
    """A NULL created_at sorts FIRST under ``ORDER BY created_at DESC`` (the Python
    probe reads it as newest) while ``NULL > ts`` is UNKNOWN in the SQL predicate
    (read as "not newer"): the two readers could disagree on such a row. No live
    row has one and both writers rely on DEFAULT now(), so the migration closes
    the class (lane 1 codex iter-1, MED)."""
    sql = MIGRATION.read_text(encoding="utf-8")
    alter = "ALTER TABLE public.expert_reviews ALTER COLUMN created_at SET NOT NULL;"
    assert alter in sql
    assert sql.index(alter) < sql.index("CREATE OR REPLACE FUNCTION public.dag_structure_rejected(")
```

Run: `$PY -m pytest tests/unit/test_database/test_migration_134_guarded_promote.py -q -p no:cacheprovider` → Expected: FAIL with `FileNotFoundError` (6 tests).

- [ ] **Step 2: Write the migration**

Create `database/migrations/134_guarded_causal_path_promote.sql`:

```sql
-- ============================================================================
-- Migration 134: guarded causal_paths promotion (lane 1, spec §4.3)
-- ============================================================================
-- WHAT: two functions.
--   public.dag_structure_rejected(p_dag_version_hash text, p_brand text)
--     → boolean. The expert-review chronology rule
--     (src/causal_engine/expert_review_gate.py, ExpertReviewGate
--     ._latest_adjudication / check_rejection) in SQL: a structure is rejected
--     when the NEWEST non-pending review row for the hash (and brand, when a
--     brand is given) is 'rejected' and no pending row is newer than it. A
--     NULL hash means "no structure to check" and reads false. An EMPTY-STRING
--     brand means "no brand" (NULLIF) -- the Python reader filters with
--     ``if brand:`` (src/repositories/expert_review.py get_reviews_for_dag), so
--     '' must be unfiltered here too or a same-hash rejection is missed
--     (pre-execution review 2026-09-08, codex HIGH). A pending row with the SAME
--     created_at as the rejection does NOT reopen it (strict >); Task 3b gives
--     ExpertReviewGate._latest_adjudication the same tie rule, so the probe
--     and the promote read a tie identically.
--   CONCURRENCY: promote_causal_path_guarded takes LOCK TABLE expert_reviews
--     IN SHARE MODE before its UPDATE (released with the RPC's transaction, a
--     few ms). SHARE conflicts with ROW EXCLUSIVE, so a resolve (UPDATE of the
--     pending row) or a renew (INSERT of a new pending row that could then be
--     rejected) racing the promote either committed first -- READ COMMITTED
--     gives the UPDATE below a fresh snapshot that sees it -- or waits and lands
--     strictly after the promote; plain reads are not blocked. A row-level FOR
--     SHARE was not enough: it cannot cover a row that does not exist yet
--     (pre-execution review iter-2 + iter-3, codex HIGH x2; row lock and table
--     lock both measured live 2026-09-08, Step 4c).
--   public.promote_causal_path_guarded(p_path_id, p_new_status,
--     p_allowed_current text[], p_dag_version_hash, p_brand) → jsonb
--     One UPDATE that moves causal_paths.validation_status only when the
--     current status is allowed AND the structure is not rejected -- the
--     rejection predicate is evaluated INSIDE the UPDATE statement, so there
--     is no window between reading the verdict and writing the status.
--     Returns {"moved": 0|1, "rejected": bool}.
-- WHY: CausalPathRepository.set_validation_status conditioned the promote on
--   the current status only; a rejection committed between the RefutationNode's
--   read-only probe and its status write was not seen (#1985 residue).
-- SAFETY: SECURITY INVOKER; service_role EXECUTE only (precedent
--   database/ml/036 record_discovered_dag); idempotent (CREATE OR REPLACE);
--   the asserting DO block below RAISEs on a grant regression. Migration 119's
--   trigger on 'validated' stays the second line of defence.
--   Also: expert_reviews.created_at SET NOT NULL (below) so the chronology is total.
-- ============================================================================

-- The chronology rule below orders review rows by created_at and compares a
-- pending row's created_at with the adjudication's using strict ``>``. A NULL
-- created_at would sort FIRST under ORDER BY … DESC (read as "newest" by the
-- Python probe) while ``NULL > ts`` is UNKNOWN here (read as "not newer"), so the
-- two readers could disagree on such a row (lane 1 codex iter-1, MED). No live
-- row has one (0/40 on 2026-09-09), both writers rely on DEFAULT now(), and the
-- column has carried that default since ml/010, so the honest fix is to make the
-- anomaly impossible: NOT NULL. Idempotent; fails loudly if a NULL row exists.
ALTER TABLE public.expert_reviews ALTER COLUMN created_at SET NOT NULL;

CREATE OR REPLACE FUNCTION public.dag_structure_rejected(
    p_dag_version_hash text,
    p_brand text DEFAULT NULL
) RETURNS boolean
LANGUAGE sql
STABLE
SET search_path = public
AS $$
    WITH latest_np AS (
        SELECT r.approval_status, r.created_at
        FROM public.expert_reviews r
        WHERE r.dag_version_hash = p_dag_version_hash
          AND (NULLIF(p_brand, '') IS NULL OR r.brand = p_brand)
          AND r.approval_status <> 'pending'
        ORDER BY r.created_at DESC
        LIMIT 1
    )
    SELECT COALESCE(
        (SELECT l.approval_status = 'rejected'
                AND NOT EXISTS (
                    SELECT 1
                    FROM public.expert_reviews p
                    WHERE p.dag_version_hash = p_dag_version_hash
                      AND (NULLIF(p_brand, '') IS NULL OR p.brand = p_brand)
                      AND p.approval_status = 'pending'
                      AND p.created_at > l.created_at  -- a tie is NOT a reopen
                )
         FROM latest_np l),
        false
    );
$$;

COMMENT ON FUNCTION public.dag_structure_rejected(text, text) IS
    'Lane 1 (migration 134): the expert-review chronology rule in SQL -- true when '
    'the newest non-pending review of this DAG hash (and brand, when given) is '
    'rejected and no pending review is newer. Python mirror: ExpertReviewGate'
    '._latest_adjudication. NULL hash reads false ("unchecked").';

CREATE OR REPLACE FUNCTION public.promote_causal_path_guarded(
    p_path_id text,
    p_new_status text,
    p_allowed_current text[],
    p_dag_version_hash text DEFAULT NULL,
    p_brand text DEFAULT NULL
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
DECLARE
    v_moved integer := 0;
    v_rejected boolean := false;
BEGIN
    IF p_path_id IS NULL OR p_new_status IS NULL OR p_allowed_current IS NULL THEN
        RAISE EXCEPTION 'promote_causal_path_guarded: p_path_id, p_new_status and p_allowed_current are required';
    END IF;

    IF p_dag_version_hash IS NOT NULL THEN
        -- Pin the review chronology for the rest of this transaction (see
        -- header): blocks concurrent review INSERT/UPDATE/DELETE, never reads.
        LOCK TABLE public.expert_reviews IN SHARE MODE;
    END IF;

    UPDATE public.causal_paths
       SET validation_status = p_new_status
     WHERE path_id = p_path_id
       AND validation_status = ANY (p_allowed_current)
       AND NOT public.dag_structure_rejected(p_dag_version_hash, p_brand);
    GET DIAGNOSTICS v_moved = ROW_COUNT;

    IF v_moved = 0 THEN
        v_rejected := public.dag_structure_rejected(p_dag_version_hash, p_brand);
    END IF;

    RETURN jsonb_build_object('moved', v_moved, 'rejected', v_rejected);
END;
$$;

COMMENT ON FUNCTION public.promote_causal_path_guarded(text, text, text[], text, text) IS
    'Lane 1 (migration 134): the RefutationNode''s SOLE promoter write. Moves '
    'causal_paths.validation_status only when the current status is in '
    'p_allowed_current AND dag_structure_rejected(hash, brand) is false, in one '
    'statement. Returns {"moved": 0|1, "rejected": bool}.';

REVOKE ALL ON FUNCTION public.dag_structure_rejected(text, text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.dag_structure_rejected(text, text) TO service_role;
REVOKE ALL ON FUNCTION public.promote_causal_path_guarded(text, text, text[], text, text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.promote_causal_path_guarded(text, text, text[], text, text) TO service_role;

DO $$
DECLARE
    v_fn text;
BEGIN
    FOREACH v_fn IN ARRAY ARRAY[
        'public.dag_structure_rejected(text, text)',
        'public.promote_causal_path_guarded(text, text, text[], text, text)'
    ] LOOP
        IF NOT has_function_privilege('service_role', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 134: service_role cannot EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('anon', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 134: anon can still EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('authenticated', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 134: authenticated can still EXECUTE %', v_fn;
        END IF;
    END LOOP;
    -- Behavioural smoke: an unknown hash is never "rejected"; an absent path never moves.
    IF public.dag_structure_rejected('migration-134-no-such-hash', NULL) THEN
        RAISE EXCEPTION 'migration 134: unknown hash reads as rejected';
    END IF;
    IF (public.promote_causal_path_guarded('migration-134-no-such-path', 'validated',
            ARRAY['pending'], NULL, NULL) ->> 'moved')::int <> 0 THEN
        RAISE EXCEPTION 'migration 134: a non-existent path moved';
    END IF;
END $$;
```

- [ ] **Step 3: Run the contract test**

`$PY -m pytest tests/unit/test_database/test_migration_134_guarded_promote.py -q -p no:cacheprovider` → Expected: 6 passed.

- [ ] **Step 4: Rehearse on the live database (BEGIN … ROLLBACK, applied twice, with a positive control)**

The live table holds exactly one rejected review (2026-07-13, brand NULL); it must read as rejected. Run from the worktree:

```bash
{
  echo 'BEGIN;'
  cat database/migrations/134_guarded_causal_path_promote.sql
  cat database/migrations/134_guarded_causal_path_promote.sql
  cat <<'SQL'
DO $$
DECLARE v_hash text;
BEGIN
  SELECT dag_version_hash INTO v_hash FROM public.expert_reviews
   WHERE approval_status = 'rejected' ORDER BY created_at DESC LIMIT 1;
  IF v_hash IS NULL THEN RAISE EXCEPTION 'rehearsal: expected one live rejected row'; END IF;
  IF NOT public.dag_structure_rejected(v_hash, NULL) THEN
    RAISE EXCEPTION 'rehearsal: the live rejected structure must read as rejected';
  END IF;
  -- A pending row newer than the rejection reopens it: simulate inside the txn.
  INSERT INTO public.expert_reviews (review_type, dag_version_hash, approval_status, reviewer_id, created_at)
  VALUES ('dag_approval', v_hash, 'pending', 'rehearsal', now());
  IF public.dag_structure_rejected(v_hash, NULL) THEN
    RAISE EXCEPTION 'rehearsal: a newer pending row must reopen the structure';
  END IF;
  RAISE NOTICE 'rehearsal OK for hash %', left(v_hash, 12);
END $$;
SQL
  echo 'ROLLBACK;'
} | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 2>&1 | tail -6
```

Expected: `NOTICE: rehearsal OK for hash …` then `ROLLBACK`, no ERROR; the output carries `ALTER TABLE` twice
(the second apply of `created_at SET NOT NULL` is a no-op on an already NOT NULL column) — measured 2026-09-09:
`ALTER TABLE` lines 2, `ERROR` lines 0, `NOTICE:  rehearsal OK for hash 4f56d9278960`, `ROLLBACK`. If the INSERT fails on a NOT NULL column, add the column to the INSERT (read `\d public.expert_reviews`); the rehearsal must end in ROLLBACK either way. Confirm nothing persisted:

```bash
docker exec supabase-db psql -U postgres -d postgres -tA -c "select count(*) from pg_proc where proname in ('dag_structure_rejected','promote_causal_path_guarded')"
```

Expected: `0`.

- [ ] **Step 4b: Equivalence scenarios against the Python rule (BEGIN … ROLLBACK)**

The pre-execution review (2026-09-08) ran these 13 scenarios through the function and through
`ExpertReviewGate._latest_adjudication` on the same rows (with `get_reviews_for_dag`'s
`if brand:` filter). Re-run them after any change to the function; the expected column is the
MEASURED output of the NULLIF version. Scenario rows use unique hashes, so live rows never interfere. S10 is
not a rule scenario: it proves the NULL-`created_at` class is CLOSED (migration 134 makes the column NOT NULL)
rather than defined, inside a SAVEPOINT so the transaction recovers and the 13 SELECTs still run; psql's
`ON_ERROR_STOP` is switched off around it and back on after.

```bash
{
  echo 'BEGIN;'
  cat database/migrations/134_guarded_causal_path_promote.sql
  cat <<'SQL'
CREATE TEMP TABLE sc(hash text, brand text, status text, ts timestamptz, vu date);
INSERT INTO sc VALUES
 ('h1', NULL, 'pending',  '2026-01-01', NULL),
 ('h2', NULL, 'rejected', '2026-01-01', NULL),
 ('h3', NULL, 'rejected', '2026-01-01', NULL), ('h3', NULL, 'pending',  '2026-01-02', NULL),
 ('h4', NULL, 'rejected', '2026-01-01', NULL), ('h4', NULL, 'approved', '2026-01-02', '2026-02-01'),
 ('h5', NULL, 'approved', '2026-01-01', '2099-01-01'), ('h5', NULL, 'rejected', '2026-01-02', NULL),
 ('h6', NULL, 'rejected', '2026-01-01', NULL),
 ('h7', NULL, 'rejected', '2026-01-01', NULL), ('h7', NULL, 'pending',  '2026-01-01', NULL),
 ('h8', 'X',  'rejected', '2026-01-01', NULL),
 ('h9', 'X',  'rejected', '2026-01-01', NULL), ('h9', 'Y',  'pending',  '2026-01-02', NULL);
INSERT INTO public.expert_reviews (review_type, dag_version_hash, brand, approval_status, reviewer_id, created_at, valid_until)
SELECT 'dag_approval', 'lane1-eq-'||hash, brand, status, 'equiv', ts, vu FROM sc;
-- S10: a NULL created_at can no longer be written (the class is closed, not defined).
\set ON_ERROR_STOP off
SAVEPOINT s10;
INSERT INTO public.expert_reviews (review_type, dag_version_hash, approval_status, reviewer_id, created_at)
VALUES ('dag_approval', 'lane1-eq-h10', 'pending', 'equiv', NULL);
ROLLBACK TO SAVEPOINT s10;
\set ON_ERROR_STOP on
SELECT 'S1 only pending' s, public.dag_structure_rejected('lane1-eq-h1', NULL) rejected
UNION ALL SELECT 'S2 rejected only', public.dag_structure_rejected('lane1-eq-h2', NULL)
UNION ALL SELECT 'S3 rejected, newer pending (reopened)', public.dag_structure_rejected('lane1-eq-h3', NULL)
UNION ALL SELECT 'S4 rejected, newer EXPIRED approval', public.dag_structure_rejected('lane1-eq-h4', NULL)
UNION ALL SELECT 'S5 approval, newer rejection', public.dag_structure_rejected('lane1-eq-h5', NULL)
UNION ALL SELECT 'S6 NULL-brand rejection, query brand X', public.dag_structure_rejected('lane1-eq-h6', 'X')
UNION ALL SELECT 'S7 pending with the SAME timestamp (tie)', public.dag_structure_rejected('lane1-eq-h7', NULL)
UNION ALL SELECT 'S8 brand-X rejection, query NULL', public.dag_structure_rejected('lane1-eq-h8', NULL)
UNION ALL SELECT 'S8 brand-X rejection, query X', public.dag_structure_rejected('lane1-eq-h8', 'X')
UNION ALL SELECT 'S8 brand-X rejection, query EMPTY STRING', public.dag_structure_rejected('lane1-eq-h8', '')
UNION ALL SELECT 'S9 X rejected, Y pending newer, query NULL', public.dag_structure_rejected('lane1-eq-h9', NULL)
UNION ALL SELECT 'S9 X rejected, Y pending newer, query X', public.dag_structure_rejected('lane1-eq-h9', 'X')
UNION ALL SELECT 'NULL hash', public.dag_structure_rejected(NULL, NULL);
SQL
  echo 'ROLLBACK;'
} | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 2>&1 | grep -E '^ S|NULL hash|ROLLBACK|ERROR|SAVEPOINT'
```

Expected (measured 2026-09-08; the Python rule agrees on every row except the tie, where Python is row-order-dependent and the SQL reads the conservative side):

| scenario | rejected |
|---|---|
| S1 only pending | f |
| S2 rejected only | t |
| S3 rejected, newer pending (reopened) | f |
| S4 rejected, newer EXPIRED approval | f (expiry is not an adjudication; both readers ignore `valid_until`) |
| S5 approval, newer rejection | t |
| S6 NULL-brand rejection, query brand X | f (a brand query excludes NULL-brand rows, as `.eq("brand", X)` does) |
| S7 pending with the SAME timestamp | t (tie ≠ reopen) |
| S8 query NULL / X / '' | t / t / t ('' is "no brand"; before NULLIF the '' row read **f** — the codex HIGH) |
| S9 query NULL / X | f / t |
| NULL hash | f |
| S10 pending with NULL created_at (SAVEPOINT) | `ERROR:  null value in column "created_at" of relation "expert_reviews" violates not-null constraint` — the class is closed, not defined (measured 2026-09-09); `ROLLBACK TO SAVEPOINT s10` recovers and the 13 rows above still all match |

Then `ROLLBACK`; confirm `select count(*) from public.expert_reviews where reviewer_id='equiv'` is `0`, the
`pg_proc` count is `0`, `information_schema.columns.is_nullable` for `expert_reviews.created_at` is still `YES`
(the constraint was rolled back with everything else) and the tallies are unchanged (pending 39 / rejected 1).

- [ ] **Step 4c: Concurrency rehearsal — the table SHARE lock blocks a racing resolve AND a racing renew, not reads (no writes)**

Session A takes the lock the function takes and holds it 6 s inside a transaction it rolls back (owner decision
2026-09-09, decision 5: 6 s, not the 20 s of the pre-execution measurement). Once the lock row is visible, session B
(a resolve-shaped UPDATE on the live probe row `4eab7033-…`), session B' (a renew-shaped INSERT of a scratch
pending row) and session R (a plain read) are launched **concurrently**, B and B' each with `lock_timeout='1s'`
and `statement_timeout='3s'`, each wrapped in `date -u +%T.%N` timestamps and rolled back. Nothing is written by
any of them. Why concurrent + `lock_timeout`: at a 6 s hold the earlier sequential recipe (B, then B', each under a
3 s `statement_timeout`) cannot discriminate — B burns its full 3 s, B' starts ~3.5 s into the hold and its 3 s
window straddles A's release, so B' completes either way (measured 2026-09-09, first run: B timed out, B' returned
`INSERT 0 1` after `A-release`). A `lock timeout` error is unambiguous: the statement was waiting on the table lock.
The monitor loop must exclude its own backend (`pid <> pg_backend_pid()`) — its query text also contains
`pg_sleep`, and the first attempt waited on itself until A had finished (a false "no block").

```bash
RID=4eab7033-7422-422d-83f6-659c9c3b9987
PSQL="docker exec supabase-db psql -U postgres -d postgres -tA"
$PSQL -v ON_ERROR_STOP=1 -c "BEGIN; LOCK TABLE public.expert_reviews IN SHARE MODE; SELECT 'A-locked '||clock_timestamp()::time; SELECT pg_sleep(6); SELECT 'A-release '||clock_timestamp()::time; ROLLBACK;" > /tmp/sessA.log 2>&1 &
until $PSQL -c "select count(*) from pg_stat_activity where pid <> pg_backend_pid() and query ilike '%pg_sleep(6)%' and state='active'" | grep -q '^1'; do sleep 0.2; done
$PSQL -F' ' -c "select l.locktype, l.mode, l.granted from pg_locks l join pg_stat_activity a on a.pid=l.pid where l.relation='public.expert_reviews'::regclass and a.pid <> pg_backend_pid()"
{ echo "B-start $(date -u +%T.%N)"; $PSQL -c "SET lock_timeout='1s'; SET statement_timeout='3s'; BEGIN; UPDATE public.expert_reviews SET updated_at = updated_at WHERE review_id='$RID'; SELECT 'B-updated'; ROLLBACK;" 2>&1 | tr '\n' ' '; echo; echo "B-end $(date -u +%T.%N)"; } > /tmp/sessB.log 2>&1 &
{ echo "B2-start $(date -u +%T.%N)"; $PSQL -c "SET lock_timeout='1s'; SET statement_timeout='3s'; BEGIN; INSERT INTO public.expert_reviews (review_type, dag_version_hash, approval_status, reviewer_id) VALUES ('dag_approval','lane1-lock-probe','pending','lockprobe'); SELECT 'B2-inserted'; ROLLBACK;" 2>&1 | tr '\n' ' '; echo; echo "B2-end $(date -u +%T.%N)"; } > /tmp/sessB2.log 2>&1 &
{ echo "R-start $(date -u +%T.%N)"; $PSQL -c "SET statement_timeout='3s'; SELECT 'R-read '||count(*) FROM public.expert_reviews;" 2>&1 | tr '\n' ' '; echo; echo "R-end $(date -u +%T.%N)"; } > /tmp/sessR.log 2>&1 &
wait
for f in A B B2 R; do tr '\n' ' ' < /tmp/sess$f.log; echo; done
$PSQL -c "SET statement_timeout='3s'; BEGIN; UPDATE public.expert_reviews SET updated_at = updated_at WHERE review_id='$RID'; SELECT 'B-updated'; ROLLBACK;" 2>&1 | tr '\n' ' '; echo
$PSQL -c "SET statement_timeout='3s'; BEGIN; INSERT INTO public.expert_reviews (review_type, dag_version_hash, approval_status, reviewer_id) VALUES ('dag_approval','lane1-lock-probe','pending','lockprobe'); SELECT 'B2-inserted'; ROLLBACK;" 2>&1 | tr '\n' ' '; echo
$PSQL -c "select count(*) from public.expert_reviews where reviewer_id='lockprobe'; select approval_status, count(*) from public.expert_reviews group by 1; select count(*) from pg_proc where proname in ('dag_structure_rejected','promote_causal_path_guarded')"
```

Measured 2026-09-09 (Task 3 execution, 6 s hold): lock row `relation ShareLock t`; A → `A-locked 02:25:07.809747 …
A-release 02:25:13.811422 ROLLBACK`; B → `B-start 02:25:08.122 … ERROR:  canceling statement due to lock timeout …
B-end 02:25:09.316`; B' → `B2-start 02:25:08.122 … ERROR:  canceling statement due to lock timeout … B2-end
02:25:09.300` — both ended ~4.5 s BEFORE `A-release`, i.e. inside the hold, each after waiting the full 1 s
`lock_timeout` on the table lock; R → `R-read 40` at 02:25:08.30 (0.18 s, not blocked); positive controls after A
ended → `UPDATE 1 B-updated ROLLBACK` and `INSERT 0 1 B2-inserted ROLLBACK`; `lockprobe` count `0`, tallies
unchanged (`rejected 1`, `pending 39`), `pg_proc` count `0`. History: the 2026-09-08 pre-execution measurement held
the lock 20 s and ran B then B' sequentially under 3 s `statement_timeout`s; both timed out (`canceling statement
due to statement timeout`), R read 40 immediately, and the row-level FOR SHARE of an earlier draft could not block
the INSERT. A B or B' that COMPLETES inside A's hold window (its end timestamp before `A-release`, no lock-timeout
error) means the function's lock does not cover that write path — stop and investigate before committing.

The amended migration itself was also rehearsed on the live DB in BEGIN … ROLLBACK (applied twice, its DO-block
grant/smoke assertions passing): `promote_causal_path_guarded('no-such-path','validated',ARRAY['pending'],<hash>,<brand>)`
returned `{"moved": 0, "rejected": false}` for the probe's pending structure, `{"moved": 0, "rejected": true}` for the
live rejected structure with brand NULL and with brand `''`, `{"moved": 0, "rejected": false}` for a NULL hash, and
`pg_locks` showed the lock held by the calling backend until ROLLBACK.

- [ ] **Step 5: Commit**

```bash
git add database/migrations/134_guarded_causal_path_promote.sql tests/unit/test_database/test_migration_134_guarded_promote.py
git commit -m "feat(db): migration 134 -- guarded causal_paths promote evaluates the review chronology inside the UPDATE

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

---

### Task 3b: Gate — both readers treat a timestamp tie the way migration 134 does

**Files:**
- Modify: `src/causal_engine/expert_review_gate.py` (`_latest_adjudication` ~line 428; in `check_approval` the existing "REJECTED verdict is durable" block ~line 365 MOVES above the pending check ~line 332)
- Modify: `tests/unit/test_causal_engine/test_expert_review_gate.py` (`TestCheckRejection`, after `test_pending_row_means_reopened_not_rejected` ~line 939)

Why: the SQL rule treats a pending row with the SAME `created_at` as the rejection as NOT newer (strict `>`);
the Python probe was row-order-dependent on that tie (measured 2026-09-08: `false/true` depending on the order
the repository returned). Codex iter-2 (HIGH) was right that documenting the difference does not make the two
readers one rule. A global re-sort by `created_at` breaks two existing tests whose rows carry no timestamps, so
the change is tie-only: the repository's order is kept except for an exact tie. With `expert_reviews.created_at`
NOT NULL since migration 134, only mock rows in tests can lack a timestamp — the tie-only rule never has to
define an order for a real NULL-timestamp row.

- [ ] **Step 1: Failing test** — add to `TestCheckRejection`:

```python
    @pytest.mark.asyncio
    async def test_pending_row_tied_with_the_rejection_is_not_a_reopen(self, mock_repo):
        """Migration 134 reads a pending row with the SAME created_at as the
        rejection as NOT newer (strict >). The probe must read the tie the same
        way whatever order the repository returns it in (lane 1, Task 3b); a
        genuinely newer pending row still reopens."""
        ts = "2026-09-08T12:00:00+00:00"
        rejected = self._rejected(created_at=ts)
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        for rows in (
            [{"review_id": "rev-tie", "approval_status": "pending", "created_at": ts}, rejected],
            [rejected, {"review_id": "rev-tie", "approval_status": "pending", "created_at": ts}],
        ):
            mock_repo.get_reviews_for_dag = AsyncMock(return_value=rows)
            result = await ExpertReviewGate(repository=mock_repo).check_rejection("abc123")
            assert result is not None and result.decision == ReviewGateDecision.REJECTED
            # Both readers, one rule: check_approval must not read the tied
            # pending row as PENDING_REVIEW while check_rejection says REJECTED.
            approval = await ExpertReviewGate(repository=mock_repo).check_approval("abc123")
            assert approval.decision == ReviewGateDecision.REJECTED

        newer = [
            {"review_id": "rev-new", "approval_status": "pending", "created_at": "2026-09-09T00:00:00+00:00"},
            rejected,
        ]
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=newer)
        assert await ExpertReviewGate(repository=mock_repo).check_rejection("abc123") is None
```

Run: `$PY -m pytest tests/unit/test_causal_engine/test_expert_review_gate.py -q -p no:cacheprovider -k tied` → Expected: FAIL on the pending-first ordering (`result is None`).

- [ ] **Step 2: The tie-only rule** — in `_latest_adjudication` replace the loop

```python
        reopened = False
        for row in history:
            if row.get("approval_status") == "pending":
                reopened = True
                continue
            return row, reopened
        return None, reopened
```

with

```python
        reopened = False
        for idx, row in enumerate(history):
            if row.get("approval_status") == "pending":
                # Tie-break (lane 1): a pending row that shares its created_at
                # with the adjudication that follows it is NOT newer than it --
                # the reading migration 134's strict ``>`` gives -- so the probe
                # and the promote can never disagree on a tie. Rows without a
                # timestamp keep the repository's order (unchanged behaviour).
                nxt = next(
                    (r for r in history[idx + 1 :] if r.get("approval_status") != "pending"),
                    None,
                )
                if (
                    nxt is not None
                    and row.get("created_at")
                    and row.get("created_at") == nxt.get("created_at")
                ):
                    continue
                reopened = True
                continue
            return row, reopened
        return None, reopened
```

and extend the docstring's ordering sentence with: "An exact `created_at` tie between a pending row and the adjudication after it is not a reopen (migration 134 reads it the same way)."

Then, in `check_approval`, MOVE (do not copy) the existing block that starts with the comment
`# A REJECTED verdict is durable (#1970).` and ends with `return self._rejection_result(latest_verdict, dag_hash)`
so that it sits immediately BEFORE the comment `# No usable approval - check for pending review (a pending row NEWER`.
Today that block runs AFTER the pending check, so a pending row that is tied with (or older than) the newest
rejection makes `check_approval` answer PENDING_REVIEW while `check_rejection` answers REJECTED (codex iter-3,
MED). With the block first, a rejection nobody re-opened wins in both readers; a genuinely newer pending row still
sets `reopened` and reaches the pending branch unchanged.

- [ ] **Step 3: Run, lint, commit**

```bash
$PY -m pytest tests/unit/test_causal_engine/test_expert_review_gate.py tests/unit/test_agents/test_causal_impact/test_refutation_expert_review_enforcement_1971.py -q -p no:cacheprovider 2>&1 | tail -2
$PY -m ruff format src/causal_engine/expert_review_gate.py && $PY -m ruff check src/causal_engine/expert_review_gate.py tests/unit/test_causal_engine/test_expert_review_gate.py
git add src/causal_engine/expert_review_gate.py tests/unit/test_causal_engine/test_expert_review_gate.py
git commit -m "fix(expert-review): a created_at tie between a pending row and the adjudication after it is not a reopen (matches migration 134)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

Expected: 93 passed. Measured 2026-09-08 with exactly these two changes applied in scratch (before the new test): 109 passed across these two files plus `test_refutation_promoter_1352.py`; both readers answered REJECTED on the tie in either row order, PENDING_REVIEW / None on a genuinely newer pending row and on rows without timestamps. An earlier global re-sort by `created_at` broke `test_reopened_after_rejection_is_not_cleared_by_an_older_approval` and `test_pending_row_means_reopened_not_rejected` (their rows carry no timestamps), which is why the rule is tie-only. The moved block needs `ruff format` (blank-line placement), hence the format step above.

---

### Task 4: Repository — `set_validation_status` routes through the guarded RPC

**Files:**
- Modify: `src/repositories/causal_path.py` (`set_validation_status`, ~line 377)
- Modify: `tests/unit/test_repositories/test_causal_path_promoter_1352.py` (append a class)

- [ ] **Step 1: Add the failing tests**

Append to `tests/unit/test_repositories/test_causal_path_promoter_1352.py`:

```python
@pytest.mark.unit
class TestGuardedPromoteRpc:
    """Lane 1 (spec §4.3): with a DAG hash the transition runs through
    ``promote_causal_path_guarded`` (migration 134); without one the plain
    conditional update is kept."""

    def _install_rpc(self, mock_client, payload):
        call = MagicMock()
        call.execute = AsyncMock(return_value=MagicMock(data=payload))
        mock_client.rpc.return_value = call
        return call

    @pytest.mark.asyncio
    async def test_hash_routes_through_the_guarded_rpc(self, repo, mock_client):
        self._install_rpc(mock_client, {"moved": 1, "rejected": False})
        moved = await repo.set_validation_status(
            "cp_1", "validated", ("pending", "needs_review"),
            dag_version_hash="h" * 64, brand="Kisqali",
        )
        assert moved is True
        name, params = mock_client.rpc.call_args.args
        assert name == "promote_causal_path_guarded"
        assert params == {
            "p_path_id": "cp_1",
            "p_new_status": "validated",
            "p_allowed_current": ["pending", "needs_review"],
            "p_dag_version_hash": "h" * 64,
            "p_brand": "Kisqali",
        }
        mock_client.table.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejected_structure_moves_nothing(self, repo, mock_client):
        self._install_rpc(mock_client, {"moved": 0, "rejected": True})
        moved = await repo.set_validation_status(
            "cp_1", "validated", ("pending",), dag_version_hash="h" * 64
        )
        assert moved is False

    @pytest.mark.asyncio
    async def test_list_wrapped_payload_is_read(self, repo, mock_client):
        self._install_rpc(mock_client, [{"moved": 1, "rejected": False}])
        assert (
            await repo.set_validation_status("cp_1", "validated", ("pending",), dag_version_hash="h" * 64)
            is True
        )

    @pytest.mark.asyncio
    async def test_no_hash_keeps_the_plain_conditional_update(self, repo, mock_client):
        query = MagicMock()
        query.eq.return_value = query
        query.in_.return_value = query
        query.execute = AsyncMock(return_value=MagicMock(data=[{"path_id": "cp_1"}]))
        mock_client.table.return_value.update.return_value = query
        assert await repo.set_validation_status("cp_1", "validated", ("pending",)) is True
        mock_client.rpc.assert_not_called()

    @pytest.mark.asyncio
    async def test_rpc_error_propagates(self, repo, mock_client):
        call = MagicMock()
        call.execute = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_client.rpc.return_value = call
        with pytest.raises(RuntimeError):
            await repo.set_validation_status("cp_1", "validated", ("pending",), dag_version_hash="h" * 64)
```

Run: `$PY -m pytest tests/unit/test_repositories/test_causal_path_promoter_1352.py -q -p no:cacheprovider -k Guarded` → Expected: FAIL with `TypeError: ... unexpected keyword argument 'dag_version_hash'`.

- [ ] **Step 2: Implement**

Replace `set_validation_status` in `src/repositories/causal_path.py` with:

```python
    async def set_validation_status(
        self,
        path_id: str,
        new_status: str,
        allowed_current: tuple,
        *,
        dag_version_hash: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> bool:
        """Conditionally move a path's ``validation_status`` (SOLE-promoter write).

        The transition is guarded server-side: the UPDATE matches only when the
        row's CURRENT status is in ``allowed_current``, so a concurrent writer
        (or an operator adjudication) is never silently overwritten. Returns
        True iff a row was actually updated. Raises on query errors — the
        caller (RefutationNode) degrades with a logged warning; a silent False
        on infra failure would be indistinguishable from a legitimate
        no-transition.

        Lane 1 (spec §4.3): when ``dag_version_hash`` is given the transition
        runs through the ``promote_causal_path_guarded`` RPC (migration 134),
        which evaluates the expert-review chronology rule INSIDE the same
        UPDATE statement, so a rejection committed after the node's read-only
        probe can never be promoted over. Without a hash there is no structure
        to check and the plain conditional update is kept.
        """
        if not self.client:
            return False
        if dag_version_hash:
            result = await self.client.rpc(
                "promote_causal_path_guarded",
                {
                    "p_path_id": path_id,
                    "p_new_status": new_status,
                    "p_allowed_current": list(allowed_current),
                    "p_dag_version_hash": dag_version_hash,
                    "p_brand": brand,
                },
            ).execute()
            payload: Any = result.data
            if isinstance(payload, list):
                payload = payload[0] if payload else {}
            if not isinstance(payload, dict):
                payload = {}
            if payload.get("rejected"):
                logger.info(
                    "guarded promote: structure %s… is rejected by expert review; "
                    "causal_paths.%s not moved to %s",
                    dag_version_hash[:12],
                    path_id,
                    new_status,
                )
            return int(payload.get("moved") or 0) > 0
        result = await (
            self.client.table(self.table_name)
            .update({"validation_status": new_status})
            .eq(self.id_column, path_id)
            .in_("validation_status", list(allowed_current))
            .execute()
        )
        return bool(result.data)
```

`Any` and `Optional` are already imported in that module (check the top; add `Any` to the `typing` import if missing).

- [ ] **Step 3: Run, lint, commit**

```bash
$PY -m pytest tests/unit/test_repositories/test_causal_path_promoter_1352.py -q -p no:cacheprovider
$PY -m ruff check src/repositories/causal_path.py tests/unit/test_repositories/test_causal_path_promoter_1352.py
$PY -m mypy --config-file pyproject.toml src/repositories/causal_path.py
git add src/repositories/causal_path.py tests/unit/test_repositories/test_causal_path_promoter_1352.py
git commit -m "feat(repo): set_validation_status goes through promote_causal_path_guarded when a DAG hash is known

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

Expected: all green (existing `TestSetValidationStatus` cases unchanged).

---

### Task 5: Node — the promote carries the run's hash and brand

**Files:**
- Modify: `src/agents/causal_impact/nodes/refutation.py` (`_persist_suite_and_promote`, the `set_validation_status` call ~line 981)
- Modify: `tests/unit/test_agents/test_causal_impact/test_refutation_promoter_1352.py` (`_FakePathRepo.set_validation_status` ~line 90; new test in `TestLinkedPromotion`)

- [ ] **Step 1: Let the fake record keyword arguments and add the failing test**

Replace `_FakePathRepo.set_validation_status` with:

```python
    async def set_validation_status(
        self, path_id: str, new_status: str, allowed_current: tuple, **kwargs: Any
    ) -> bool:
        if self.fail_update:
            raise RuntimeError("db write failed")
        self.status_calls.append(
            {
                "path_id": path_id,
                "new_status": new_status,
                "allowed_current": allowed_current,
                "dag_version_hash": kwargs.get("dag_version_hash"),
                "brand": kwargs.get("brand"),
            }
        )
        return True
```

Append to `class TestLinkedPromotion`:

```python
    @pytest.mark.asyncio
    async def test_promote_carries_the_structure_hash_and_brand(self) -> None:
        """Lane 1 (spec §4.3): the guarded RPC needs the run's DAG hash and brand
        to evaluate the rejection rule inside the UPDATE."""
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001", dag_version_hash="h" * 64, brand="Kisqali"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        call = path_repo.status_calls[0]
        assert call["dag_version_hash"] == "h" * 64
        assert call["brand"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_promote_without_a_hash_passes_none(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        state = _state(causal_path_id="cp_real_000000001")
        state.pop("dag_version_hash", None)
        await node._persist_suite_and_promote(
            state, _suite(GateDecision.PROCEED), structure_verdict="clear"
        )
        assert path_repo.status_calls[0]["dag_version_hash"] is None
```

(`_state(**overrides)` in that file merges overrides into the base state dict; if it does not accept `dag_version_hash`/`brand`, set them on the returned dict instead: `s = _state(...); s["dag_version_hash"] = "h" * 64; s["brand"] = "Kisqali"`.)

Run: `$PY -m pytest tests/unit/test_agents/test_causal_impact/test_refutation_promoter_1352.py -q -p no:cacheprovider -k hash_and_brand` → Expected: FAIL, `call["dag_version_hash"] is None`.

- [ ] **Step 2: Pass the hash and brand at the call site**

In `_persist_suite_and_promote`, change

```python
            moved = await self.causal_path_repo.set_validation_status(
                path_id, new_status, allowed_current
            )
```

to

```python
            # Lane 1 (spec §4.3): the guarded RPC re-evaluates the rejection rule
            # inside the UPDATE, so the probe→write window can no longer be won
            # by a rejection committed in between.
            moved = await self.causal_path_repo.set_validation_status(
                path_id,
                new_status,
                allowed_current,
                dag_version_hash=(str(state.get("dag_version_hash") or "") or None),
                # '' is "no brand" for the Python probe (``if brand:``) and, via
                # NULLIF, for the SQL rule; pass None so the two can never
                # disagree (pre-execution review 2026-09-08).
                brand=(cast(Optional[str], state.get("brand")) or None),
            )
```

- [ ] **Step 3: Run the node's refutation tests, lint, commit**

```bash
$PY -m pytest tests/unit/test_agents/test_causal_impact/test_refutation_promoter_1352.py tests/unit/test_agents/test_causal_impact/test_refutation_expert_review_enforcement_1971.py tests/unit/test_agents/test_causal_impact/test_refutation.py -q -p no:cacheprovider -m "not slow" 2>&1 | tail -4
$PY -m ruff check src/agents/causal_impact/nodes/refutation.py tests/unit/test_agents/test_causal_impact/test_refutation_promoter_1352.py
git add src/agents/causal_impact/nodes/refutation.py tests/unit/test_agents/test_causal_impact/test_refutation_promoter_1352.py
git commit -m "feat(causal-impact): the promoter passes the run's DAG hash and brand to the guarded transition

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

---

### Task 6: Backend — `GET /expert-reviews/{review_id}` with same-structure history

**Files:**
- Modify: `src/api/schemas/expert_review.py` (add two models after `PendingReviewItem`)
- Modify: `src/api/routes/expert_review.py` (import the models; add the route at the END of the file)
- Create: `tests/unit/test_api/test_expert_review_detail_route.py`
- Modify (Task 12 fold, Step 5): `src/repositories/expert_review.py`, `tests/unit/test_repositories/test_expert_review.py`, `tests/api/test_expert_review_routes.py`
- Create (Task 12 fold, Step 5): `database/migrations/136_expert_reviews_resolved_at.sql`, `tests/unit/test_database/test_migration_136_resolved_at.py`

- [ ] **Step 1: Write the failing route tests**

```python
"""Lane 1 (spec §4.2): GET /expert-reviews/{review_id} returns a review in ANY
status plus the same-structure history, so the drill-down's deep link resolves
for pending, approved and rejected structures alike."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import src.api.routes.expert_review as route_mod
from src.api.errors import SAFE_503_DETAIL_PREFIX

# Real UUID literals: ``expert_reviews.review_id`` is a uuid column, and the
# route canonicalises the path id before the store read (review F1).
RID = "1c8f3d6a-5b7e-4c21-9f0a-2d4e6b8a0c13"
RID_OLDER = "7a2b9c40-3d1e-4f65-8a7b-0c9d2e1f3a58"
RID_UNKNOWN = "9e8d7c6b-5a49-4382-b1c0-d9e8f7a6b5c4"
DAG_HASH = "h" * 64

ROW: Dict[str, Any] = {
    "review_id": RID,
    "review_type": "dag_approval",
    "dag_version_hash": DAG_HASH,
    "brand": "Kisqali",
    "treatment_variable": "treatment_arm",
    "outcome_variable": "persistent_180d",
    "approval_status": "rejected",
    "reviewer_name": "Dr. No",
    "concerns_raised": ["collider"],
    "created_at": "2026-07-13T10:00:00+00:00",
    "valid_from": "2026-07-13",
    "approved_at": "2026-07-14T09:30:00+00:00",
    "dag_structure_json": json.dumps({"nodes": ["t", "y"], "edges": [["t", "y"]]}),
    "comments_json": json.dumps({"note": "engagement is post-treatment"}),
}


class _Repo:
    def __init__(
        self,
        row: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fail: Optional[str] = None,
    ):
        self.row, self.history, self.fail = row, history or [], fail
        self.get_calls: List[str] = []
        self.history_calls: List[tuple] = []

    async def get_by_id(self, review_id: str):
        self.get_calls.append(review_id)
        if self.fail == "row":
            raise RuntimeError("connection refused")
        try:
            uuid.UUID(review_id)
        except ValueError:  # what the live uuid column does: PostgREST APIError 22P02
            raise RuntimeError(f'invalid input syntax for type uuid: "{review_id}"') from None
        # Matches on the CANONICAL id only, like the uuid column would.
        return self.row if self.row and self.row["review_id"] == review_id else None

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ):
        self.history_calls.append((dag_hash, include_expired, brand))
        if self.fail == "history":
            raise RuntimeError("connection refused")
        return self.history

    async def get_pending_reviews(self, brand=None, reviewer_id=None, limit=50):
        return []


def _install(monkeypatch, repo: _Repo) -> None:
    async def _factory():
        return repo

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _factory)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_the_row_and_its_same_structure_history(monkeypatch):
    repo = _Repo(ROW, history=[ROW, {**ROW, "review_id": RID_OLDER, "approval_status": "pending"}])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID, user={})
    assert resp.review.review_id == RID
    assert resp.review.approval_status == "rejected"
    assert resp.review.reviewer_name == "Dr. No"
    assert resp.review.dag_structure_json == {"nodes": ["t", "y"], "edges": [["t", "y"]]}
    assert resp.review.comments_json == {"note": "engagement is post-treatment"}
    assert [r.review_id for r in resp.history] == [RID, RID_OLDER]
    # the same read the gate's rejection probe performs: expired included, brand-scoped
    assert repo.history_calls == [(DAG_HASH, True, "Kisqali")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolution_provenance_columns_are_surfaced(monkeypatch):
    """Codex whole-diff HIGH F1: ``reviewer_email`` and ``resolved_at`` (migration
    136) reach the client so the card can render RECORDED provenance. Both are
    None on a row resolved before the migration (positive/negative pair)."""
    resolved = {
        **ROW,
        "reviewer_email": "no@example.com",
        "resolved_at": "2026-07-14T09:31:00+00:00",
    }
    repo = _Repo(resolved, history=[resolved, {**ROW, "review_id": RID_OLDER}])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID, user={})
    assert resp.review.reviewer_email == "no@example.com"
    assert resp.review.resolved_at == datetime(2026, 7, 14, 9, 31, tzinfo=timezone.utc)
    assert resp.history[0].resolved_at == resp.review.resolved_at
    # A pre-136 row carries neither; the reader reports None, not a stand-in.
    assert resp.history[1].reviewer_email is None
    assert resp.history[1].resolved_at is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_review_is_404(monkeypatch):
    """A VALID but unknown uuid: the store answers None -> 404."""
    repo = _Repo(None)
    _install(monkeypatch, repo)
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review(RID_UNKNOWN, user={})
    assert ei.value.status_code == 404
    assert repo.get_calls == [RID_UNKNOWN]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_id_is_404_without_touching_the_store(monkeypatch):
    """A non-uuid id would make PostgREST raise 22P02 (a 503 through the
    store-failure guard); the pre-check answers 404 and never reads the store."""

    class _NeverRead(_Repo):
        async def get_by_id(self, review_id: str):
            raise AssertionError("store must not be read")

    _install(monkeypatch, _NeverRead(ROW))
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review("nope", user={})
    assert ei.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_canonical_id_reaches_the_store_canonical(monkeypatch):
    repo = _Repo(ROW, history=[ROW])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID.upper(), user={})
    assert resp.review.review_id == RID
    assert repo.get_calls == [RID]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("fail", ["row", "history"])
async def test_store_failure_is_503_never_an_empty_200(monkeypatch, fail):
    _install(monkeypatch, _Repo(ROW, history=[ROW], fail=fail))
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review(RID, user={})
    assert ei.value.status_code == 503


@pytest.mark.unit
@pytest.mark.asyncio
async def test_client_factory_failure_is_503(monkeypatch):
    """``_get_expert_review_repo`` raises when Supabase is unset/unreachable; that
    must be the same honest 503, not an unhandled 500."""

    async def _boom():
        raise RuntimeError("supabase unavailable")

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _boom)
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review(RID, user={})
    assert ei.value.status_code == 503


@pytest.mark.unit
@pytest.mark.asyncio
async def test_row_without_a_hash_has_empty_history(monkeypatch):
    repo = _Repo({**ROW, "dag_version_hash": None})
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID, user={})
    assert resp.history == []
    assert repo.history_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_row_without_a_brand_reads_an_unfiltered_history(monkeypatch):
    """A brand-less row's history is cross-brand -- exactly the gate's read."""
    repo = _Repo({**ROW, "brand": None}, history=[{**ROW, "brand": None}])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID, user={})
    assert [r.review_id for r in resp.history] == [RID]
    assert repo.history_calls == [(DAG_HASH, True, None)]


@pytest.mark.unit
def test_lookup_route_is_declared_after_pending_and_summary():
    """FastAPI matches in declaration order: /{review_id} must not shadow the
    two literal GET routes."""
    paths = [r.path for r in route_mod.router.routes]
    assert paths.index("/expert-reviews/pending") < paths.index("/expert-reviews/{review_id}")
    assert paths.index("/expert-reviews/summary") < paths.index("/expert-reviews/{review_id}")


# --- in-process client: Depends, HTTP serialisation, error bodies, routing ---
# Minimal app (no ``src.api.main`` import, so no lifespan); precedent
# tests/unit/test_api/test_executive_insights.py. E2I_TESTING_MODE=1 (conftest)
# makes ``require_operator`` yield the mock user.


def _client(monkeypatch, repo: _Repo) -> TestClient:
    _install(monkeypatch, repo)
    app = FastAPI()
    app.include_router(route_mod.router, prefix="/api")
    return TestClient(app)


@pytest.mark.unit
def test_http_200_serialises_dates_and_parsed_json(monkeypatch):
    client = _client(monkeypatch, _Repo(ROW, history=[ROW, {**ROW, "review_id": RID_OLDER}]))
    r = client.get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["review"]["review_id"] == RID
    assert body["review"]["valid_from"] == "2026-07-13"
    assert datetime.fromisoformat(body["review"]["approved_at"]) == datetime(
        2026, 7, 14, 9, 30, tzinfo=timezone.utc
    )
    assert body["review"]["comments_json"] == {"note": "engagement is post-treatment"}
    assert len(body["history"]) == 2


@pytest.mark.unit
def test_http_unknown_valid_uuid_is_404_with_detail(monkeypatch):
    r = _client(monkeypatch, _Repo(None)).get(f"/api/expert-reviews/{RID_UNKNOWN}")
    assert r.status_code == 404
    assert isinstance(r.json()["detail"], str) and RID_UNKNOWN in r.json()["detail"]


@pytest.mark.unit
def test_http_malformed_id_is_404(monkeypatch):
    r = _client(monkeypatch, _Repo(ROW)).get("/api/expert-reviews/nope")
    assert r.status_code == 404, r.text


@pytest.mark.unit
def test_http_store_failure_is_a_safe_503(monkeypatch):
    r = _client(monkeypatch, _Repo(ROW, fail="row")).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 503
    detail = r.json()["detail"]
    # The marker the app's global handler surfaces verbatim -- pin it.
    assert detail.startswith(SAFE_503_DETAIL_PREFIX)
    assert "Expert-review store unavailable" in detail


@pytest.mark.unit
def test_http_pending_is_not_shadowed_by_the_lookup(monkeypatch):
    r = _client(monkeypatch, _Repo(ROW)).get("/api/expert-reviews/pending")
    # The 200 + the pending body are the proof. A store-call count could not
    # discriminate: were /{review_id} declared first, "pending" would fail the
    # uuid pre-check and answer 404 BEFORE any store read (measured), which the
    # status assertion catches.
    assert r.status_code == 200, r.text
    assert r.json() == {"reviews": [], "total": 0}
```

Run: `$PY -m pytest tests/unit/test_api/test_expert_review_detail_route.py -q -p no:cacheprovider` → Expected: FAIL, `AttributeError: module ... has no attribute 'get_expert_review'`.

- [ ] **Step 2: Add the schemas**

In `src/api/schemas/expert_review.py`, change the datetime import to `from datetime import date, datetime` and add after `PendingReviewItem`:

```python
class ReviewRecord(PendingReviewItem):
    """One ``expert_reviews`` row in ANY status (lane 1, spec §4.2).

    Extends ``PendingReviewItem`` with the resolution columns so the queue
    page's linked-review card can show who decided what, and until when.

    Provenance (codex whole-diff HIGH F1): ``reviewer_id`` holds the REQUESTER
    (the originating query id the gate wrote), never the resolver. The
    resolver is ``reviewer_name`` / ``reviewer_email`` and the decision time is
    ``resolved_at`` -- the resolution time for BOTH statuses since migration
    136, NULL for rows resolved before it (``approved_at`` is approval-only).
    """

    approval_status: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewer_name: Optional[str] = None
    reviewer_email: Optional[str] = None
    approved_at: Optional[datetime] = None
    #: Resolution time for both statuses (migration 136); None before it.
    resolved_at: Optional[datetime] = None
    valid_from: Optional[date] = None
    valid_until: Optional[date] = None
    concerns_raised: Optional[List[str]] = None
    conditions: Optional[str] = None
    checklist_json: Optional[Dict[str, Any]] = None
    comments_json: Optional[Dict[str, Any]] = None
    supersedes_review_id: Optional[str] = None

    @field_validator("checklist_json", "comments_json", mode="before")
    @classmethod
    def _parse_resolution_json(cls, value: Any) -> Any:
        return _json_string_to_dict(value)


class ExpertReviewDetailResponse(BaseModel):
    """``GET /expert-reviews/{review_id}``: the row plus its same-structure history.

    ``history`` is every review sharing the DAG hash (and brand), newest first,
    expired rows included -- the same read the gate's rejection probe performs.
    """

    review: ReviewRecord
    history: List[ReviewRecord]
```

- [ ] **Step 3: Add the route (at the end of `src/api/routes/expert_review.py`)**

Extend the schema import:

```python
from src.api.schemas.expert_review import (
    AgentAssessmentResponse,
    ExpertReviewDetailResponse,
    PendingReviewItem,
    PendingReviewsResponse,
    ResolveReviewRequest,
    ResolveReviewResponse,
    ReviewRecord,
    ReviewSummaryResponse,
)
```

Append after `get_summary`:

```python
@router.get(
    "/{review_id}",
    response_model=ExpertReviewDetailResponse,
    summary="One expert review (any status) with its same-structure history",
    operation_id="get_expert_review",
    responses={
        404: {"model": ErrorResponse, "description": "Review not found"},
        503: {"model": ErrorResponse, "description": "Expert-review store unavailable"},
    },
)
async def get_expert_review(
    review_id: str,
    user: Dict[str, Any] = Depends(require_operator),
) -> ExpertReviewDetailResponse:
    """Return one review row in any status plus every review of the same DAG structure.

    Powers the linked-review card the causal drill-down deep-links to
    (``/expert-reviews?review=<id>``), so a run whose structure is pending,
    approved or rejected always resolves to its record. ``history`` is the full
    same-hash (and same-brand) list, newest first, expired included -- the read
    ``ExpertReviewGate.check_rejection`` performs. Declared LAST in this module
    so it cannot shadow ``/pending`` and ``/summary``.
    """
    # A malformed id is 404, not 503 (review 2026-09-09, measured live):
    # ``expert_reviews.review_id`` is a uuid column, so a non-UUID string makes
    # PostgREST raise APIError 22P02 ("invalid input syntax for type uuid"),
    # which the store-failure guard below would report as an outage with an
    # ERROR traceback -- while the sibling ``POST /{review_id}/resolve``
    # answers 404 for the same input. Pre-check for parity, and hand the store
    # the CANONICAL form so any form Python accepts (uppercase, braces) can
    # never trip the cast. The raw path value stays in the 404 messages.
    try:
        canonical_id = str(uuid.UUID(review_id))
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Review {review_id} was not found (not a valid review id)."
        ) from None
    try:
        # The client factory raises ServiceConnectionError when Supabase is
        # unset/unreachable; inside the try so that is a 503 as well, not a
        # 500 (pre-execution review 2026-09-08, codex MED).
        repo = await _get_expert_review_repo()
        row = await repo.get_by_id(canonical_id)
    except Exception as e:  # store failure (R3): honest 503
        raise _store_unavailable("review read", e) from e
    if not row:
        raise HTTPException(status_code=404, detail=f"Review {review_id} was not found.")
    dag_hash = row.get("dag_version_hash")
    history_rows: List[Dict[str, Any]] = []
    if dag_hash:
        try:
            history_rows = await repo.get_reviews_for_dag(
                dag_hash, include_expired=True, brand=row.get("brand")
            )
        except Exception as e:
            raise _store_unavailable("review history read", e) from e
    return ExpertReviewDetailResponse(
        review=ReviewRecord.model_validate(row),
        history=[ReviewRecord.model_validate(r) for r in history_rows],
    )
```

- [ ] **Step 4: Run, lint, type-check, commit**

```bash
$PY -m pytest tests/unit/test_api/test_expert_review_detail_route.py tests/unit/test_api/test_causal_agent_analyze_expert_review_1971.py -q -p no:cacheprovider
$PY -m ruff check src/api/routes/expert_review.py src/api/schemas/expert_review.py tests/unit/test_api/test_expert_review_detail_route.py
$PY -m mypy --config-file pyproject.toml src/api/routes/expert_review.py src/api/schemas/expert_review.py
git add src/api/routes/expert_review.py src/api/schemas/expert_review.py tests/unit/test_api/test_expert_review_detail_route.py
git commit -m "feat(api): GET /expert-reviews/{review_id} -- a review in any status with its same-structure history

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

- [ ] **Step 5: Resolution provenance — migration 136; the writer and the route record the resolving operator (Task 12 whole-diff fold, codex HIGH F1)**

The linked-review card rendered `reviewer_name ?? reviewer_id` and `approved_at ?? created_at`. `reviewer_id` holds the REQUESTER (`src/causal_engine/expert_review_gate.py` `create_review(reviewer_id=requester_id)`, the originating query id), no live row carried `reviewer_name` / `reviewer_email` (0 of 40 on 2026-09-09), and a rejection wrote no timestamp at all (`approved_at` is approval-only; `updated_at` is trigger-maintained and not a decision time). Red-first: the migration contract test, the repository payload pins for BOTH statuses, the route call-kwargs pin with an explicit operator override, and the detail-route field round-trip (Step 1's file above, regenerated). `reviewer_id` is deliberately never overwritten. Nothing is backfilled: unknown stays unknown.

Create `database/migrations/136_expert_reviews_resolved_at.sql` (rehearse it `BEGIN … ROLLBACK` on the live database: the column is present inside the transaction and `select count(*) from information_schema.columns where table_name='expert_reviews' and column_name='resolved_at'` is 0 afterwards):

```sql
-- ============================================================================
-- Migration 136: expert_reviews.resolved_at (lane 1, codex whole-diff HIGH F1)
-- ============================================================================
-- WHAT: one nullable column, public.expert_reviews.resolved_at TIMESTAMPTZ --
--   the time an operator resolved the review, for BOTH statuses. Written by
--   ExpertReviewRepository.submit_review (src/repositories/expert_review.py)
--   as now() together with the resolver's reviewer_name / reviewer_email,
--   which the resolve route derives from the authenticated operator
--   (src/api/routes/expert_review.py resolve_review).
-- WHY: the linked-review card showed Reviewer = reviewer_name ?? reviewer_id
--   and Decided = approved_at ?? created_at. reviewer_id holds the REQUESTER
--   (the originating query id: src/agents/causal_impact/nodes/refutation.py
--   -> expert_review_gate.py create_review(reviewer_id=requester_id)), no
--   live row carries reviewer_name / reviewer_email (0 of 40 on 2026-09-09),
--   and a rejection wrote no timestamp at all (approved_at is set only on
--   approval) -- so every resolved row would have named a query id as the
--   reviewer and its creation date as the decision date. This column plus the
--   writer change make the decision time recordable; the card renders only
--   what was recorded.
-- NO BACKFILL: rows resolved before this migration stay NULL. updated_at is
--   trigger-maintained and is NOT a decision time (the one live rejected row
--   had its cached agent assessment written after its rejection). Unknown
--   stays unknown.
-- SAFETY: additive, idempotent (ADD COLUMN IF NOT EXISTS), no default, no
--   constraint; the old image's writer keeps working during the deploy window.
--   No BEGIN/COMMIT of its own -- scripts/run_migrations.sh wraps the file in
--   --single-transaction.
-- ============================================================================

ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

COMMENT ON COLUMN public.expert_reviews.resolved_at IS
    'Lane 1 (migration 136): when an operator resolved this review, for BOTH '
    'statuses (approved and rejected). Written as now() by '
    'ExpertReviewRepository.submit_review together with the resolver''s '
    'reviewer_name / reviewer_email. NULL for rows resolved before this '
    'migration -- deliberately not backfilled, the trigger-maintained updated_at '
    'is not a decision time.';
```

Create `tests/unit/test_database/test_migration_136_resolved_at.py`:

```python
"""Migration 136 adds ``expert_reviews.resolved_at`` (lane 1, codex whole-diff HIGH F1).

The resolution time for BOTH statuses, written by ``submit_review`` together
with the resolver's ``reviewer_name`` / ``reviewer_email``. Rows resolved before
the migration keep NULL: ``updated_at`` is trigger-maintained and is NOT a
decision time (the one live rejected row had its cached assessment written
after its rejection), so there is nothing honest to backfill from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "136_expert_reviews_resolved_at.sql"

ADD_COLUMN = "ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;"


@pytest.mark.unit
def test_adds_the_column_idempotently():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert ADD_COLUMN in sql


@pytest.mark.unit
def test_column_comment_states_the_contract():
    """The comment is the column's contract: both statuses, written by
    ``submit_review`` with the resolver identity, NULL before the migration."""
    sql = MIGRATION.read_text(encoding="utf-8")
    comment = re.search(
        r"COMMENT ON COLUMN public\.expert_reviews\.resolved_at IS\s*(.*?);", sql, re.S
    )
    assert comment is not None
    text = comment.group(1)
    assert "submit_review" in text
    assert "reviewer_name" in text and "reviewer_email" in text
    assert "approved" in text and "rejected" in text
    assert "NULL" in text


@pytest.mark.unit
def test_no_backfill_unknown_stays_unknown():
    """``updated_at`` is not a decision time; a backfill would fabricate one."""
    sql = MIGRATION.read_text(encoding="utf-8")
    statements = [s for s in sql.split("\n") if not s.lstrip().startswith("--")]
    body = "\n".join(statements)
    assert not re.search(r"\bUPDATE\b", body, re.I)
    assert not re.search(r"\bSET\s+resolved_at\b", body, re.I)
    # No DEFAULT either: a default would stamp future rows at INSERT time.
    assert not re.search(r"\bDEFAULT\b", body, re.I)


@pytest.mark.unit
def test_runner_wraps_it_no_own_transaction():
    """scripts/run_migrations.sh wraps a file in --single-transaction unless it
    manages its own; a COMMIT here would end the wrapper's transaction."""
    sql = MIGRATION.read_text(encoding="utf-8")
    body = "\n".join(s for s in sql.split("\n") if not s.lstrip().startswith("--"))
    assert not re.search(r"^\s*(BEGIN|COMMIT)\s*;", body, re.I | re.M)
```

In `src/repositories/expert_review.py`, replace `submit_review` with:

```python
    async def submit_review(
        self,
        review_id: str,
        approval_status: str,
        checklist: Dict[str, Any],
        comments: Optional[Dict[str, Any]] = None,
        concerns_raised: Optional[List[str]] = None,
        conditions: Optional[str] = None,
        validity_days: int = DEFAULT_VALIDITY_DAYS,
        reviewer_name: Optional[str] = None,
        reviewer_email: Optional[str] = None,
    ) -> bool:
        """
        Submit a completed expert review.

        Only a PENDING row can be resolved (R2, lane-1971 audit): the UPDATE
        itself carries ``approval_status = 'pending'``, so an already-resolved
        row -- including an OLDER approval while a NEWER one exists -- matches
        zero rows and returns False (the route surfaces that as 404). Before
        this the filter was ``review_id`` alone and the pending-only claim was
        documentation, not enforcement.

        Resolution provenance (lane 1, codex whole-diff HIGH F1): BOTH statuses
        stamp ``resolved_at = now()`` (migration 136; ``approved_at`` stays
        approval-only) and record the RESOLVER's ``reviewer_name`` /
        ``reviewer_email`` when the caller knows them. ``reviewer_id`` is
        deliberately NOT written here: the gate stores the REQUESTER in it --
        the originating query id (src/causal_engine/expert_review_gate.py
        create_review(reviewer_id=requester_id), ~:392) -- and that breadcrumb
        must survive the resolution. An absent identity is left absent (the
        None-strip below drops it): unknown stays unknown, never a placeholder.

        Args:
            review_id: UUID of the review to complete
            approval_status: 'approved' or 'rejected'
            checklist: Completed checklist with responses
            comments: Reviewer notes and feedback
            concerns_raised: List of specific concerns
            conditions: Any conditions on approval
            validity_days: Days until review expires (default 90)
            reviewer_name: Display name of the resolving operator, if known
            reviewer_email: Email of the resolving operator, if known

        Returns:
            True if exactly this pending row was resolved, False otherwise
            (nonexistent, already resolved, or persistence error)
        """
        if not self.client:
            return False

        if approval_status not in ("approved", "rejected"):
            logger.error(f"Invalid approval_status: {approval_status}")
            return False

        update_data = {
            "approval_status": approval_status,
            "checklist_json": json.dumps(checklist),
            "comments_json": json.dumps(comments) if comments else None,
            "concerns_raised": concerns_raised,
            "conditions": conditions,
            # Decision time for BOTH statuses (migration 136).
            "resolved_at": "now()",
            "reviewer_name": reviewer_name,
            "reviewer_email": reviewer_email,
        }

        if approval_status == "approved":
            valid_until = date.today() + timedelta(days=validity_days)
            update_data["valid_from"] = date.today().isoformat()
            update_data["valid_until"] = valid_until.isoformat()
            update_data["approved_at"] = "now()"

        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}

        try:
            result = await (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("review_id", review_id)
                .eq("approval_status", "pending")
                .execute()
            )
            # FIX B (codex HIGH): a zero-row update (nonexistent or already-resolved
            # review_id) matches nothing — supabase-py returns the updated rows in
            # ``result.data`` (same convention as base.py:131), so empty data means
            # nothing was touched. Returning True there is a fabricated success that
            # would make the route 200 a record it never changed.
            if not result.data:
                logger.warning(
                    f"submit_review matched no rows for {review_id} "
                    "(nonexistent or already-resolved); returning False"
                )
                return False
            logger.info(f"Submitted review {review_id} with status {approval_status}")
            return True
        except Exception as e:
            logger.error(f"Failed to submit review {review_id}: {e}")
            return False
```

In `src/api/routes/expert_review.py`, replace `resolve_review` with (its docstring is OpenAPI text -- `api.ts` is regenerated once at the end, Task 7):

```python
@router.post(
    "/{review_id}/resolve",
    response_model=ResolveReviewResponse,
    summary="Resolve (approve/reject) an expert review",
    operation_id="resolve_expert_review",
)
async def resolve_review(
    review_id: str,
    request: ResolveReviewRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> ResolveReviewResponse:
    """Approve or reject a pending review; the resolution persists.

    The authenticated operator is recorded as the resolver: ``reviewer_name``
    (their profile name, else their email, else their id) and
    ``reviewer_email`` are written with the resolution, and ``resolved_at`` is
    stamped for BOTH statuses (migration 136). ``reviewer_id`` is left as the
    requester breadcrumb the gate wrote. An identity the token does not carry
    stays unrecorded.

    An ``approved`` resolution sets ``valid_from``/``valid_until``/``approved_at``
    inside ``submit_review`` (repo :169-173). A repo ``False`` is fail-closed —
    never a fabricated success. FIX B (codex HIGH): ``submit_review`` now returns
    False on a ZERO-ROW update (nonexistent / already-resolved review_id), so we
    surface that as 404 (the honest 'not found / not resolvable' code), not a
    fabricated 200. A genuine persistence error also returns False -> 404, which
    is still a correct non-200 (never a fake success); the repo logs the
    distinction (zero-row WARNING vs exception ERROR).
    """
    # The resolver's identity, from the verified token (dependencies/auth.py
    # builds ``id`` / ``email`` / ``user_metadata`` from the Supabase user).
    # Unknown stays unknown: when the token carries none of them, pass None.
    reviewer_name = (
        (user.get("user_metadata") or {}).get("name") or user.get("email") or user.get("id")
    )
    reviewer_email = user.get("email")
    repo = await _get_expert_review_repo()
    success = await repo.submit_review(
        review_id=review_id,
        approval_status=request.approval_status,
        checklist=request.checklist,
        comments=request.comments,
        concerns_raised=request.concerns_raised,
        conditions=request.conditions,
        validity_days=request.validity_days,
        reviewer_name=reviewer_name or None,
        reviewer_email=reviewer_email or None,
    )
    if not success:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Review {review_id} was not found or is not resolvable "
                "(it may not exist or has already been resolved)."
            ),
        )
    return ResolveReviewResponse(
        review_id=review_id,
        approval_status=request.approval_status,
        success=True,
    )
```

Append to `TestExpertReviewRepository` in `tests/unit/test_repositories/test_expert_review.py`:

```python
    @pytest.mark.asyncio
    @pytest.mark.parametrize("approval_status", ["rejected", "approved"])
    async def test_submit_review_records_resolution_provenance(
        self, repo, mock_client, approval_status
    ):
        """Codex whole-diff HIGH F1: BOTH statuses stamp ``resolved_at`` and carry
        the resolver's identity; ``reviewer_id`` is never overwritten (it holds
        the REQUESTER -- the originating query id, gate :392)."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-123"}]))
        mock_client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = mock_execute

        result = await repo.submit_review(
            review_id="rev-123",
            approval_status=approval_status,
            checklist={"confounder_check": True},
            reviewer_name="Dr. Operator",
            reviewer_email="operator@example.com",
        )

        assert result is True
        payload = mock_client.table.return_value.update.call_args[0][0]
        assert payload["resolved_at"] == "now()"
        assert payload["reviewer_name"] == "Dr. Operator"
        assert payload["reviewer_email"] == "operator@example.com"
        assert "reviewer_id" not in payload
        # approved_at stays approval-only; a rejection records resolved_at alone.
        assert ("approved_at" in payload) is (approval_status == "approved")

    @pytest.mark.asyncio
    async def test_submit_review_without_identity_leaves_the_columns_untouched(
        self, repo, mock_client
    ):
        """Unknown stays unknown: no identity -> no identity keys in the UPDATE
        (the None-strip drops them), but ``resolved_at`` is still stamped."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-123"}]))
        mock_client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = mock_execute

        result = await repo.submit_review(
            review_id="rev-123", approval_status="rejected", checklist={}
        )

        assert result is True
        payload = mock_client.table.return_value.update.call_args[0][0]
        assert payload["resolved_at"] == "now()"
        assert "reviewer_name" not in payload
        assert "reviewer_email" not in payload
        assert "reviewer_id" not in payload
```

Append to `TestResolveReview` in `tests/api/test_expert_review_routes.py` (its fake `submit_review` gains the `reviewer_name` / `reviewer_email` kwargs; the file is not in `backend-tests.yml`'s allowlist, so run it locally):

```python
    def test_resolve_records_the_authenticated_operator(self, client, fake_repo):
        """Codex whole-diff HIGH F1: the operator the route authenticated is the
        resolver, so their identity must reach ``submit_review`` (call-kwargs
        pin). The override carries BOTH a metadata name and an email so the
        assertion cannot pass vacuously on the testing-mode default user."""
        app.dependency_overrides[require_operator] = lambda: {
            "id": "op-1",
            "email": "operator@example.com",
            "app_metadata": {"role": "admin"},
            "user_metadata": {"name": "Dr. Operator"},
        }
        resp = client.post(
            "/api/expert-reviews/66666666-6666-6666-6666-666666666666/resolve",
            json={"approval_status": "rejected", "checklist": {}},
        )
        assert resp.status_code == 200, resp.text
        call = fake_repo.submit_calls[0]
        assert call["reviewer_name"] == "Dr. Operator"
        assert call["reviewer_email"] == "operator@example.com"

    @pytest.mark.parametrize(
        ("user", "expected_name", "expected_email"),
        [
            # No metadata name -> the email stands in as the display name.
            (
                {"id": "op-2", "email": "op2@example.com", "user_metadata": {}},
                "op2@example.com",
                "op2@example.com",
            ),
            # No name, no email -> the id; email stays unknown.
            ({"id": "op-3", "user_metadata": {}}, "op-3", None),
            # Nothing usable -> unknown stays unknown (None, never a placeholder).
            ({"user_metadata": {}}, None, None),
        ],
    )
    def test_resolve_operator_identity_fallbacks(
        self, client, fake_repo, user, expected_name, expected_email
    ):
        app.dependency_overrides[require_operator] = lambda: {
            "app_metadata": {"role": "admin"},
            **user,
        }
        resp = client.post(
            "/api/expert-reviews/77777777-7777-7777-7777-777777777777/resolve",
            json={"approval_status": "approved", "checklist": {}},
        )
        assert resp.status_code == 200, resp.text
        call = fake_repo.submit_calls[0]
        assert call["reviewer_name"] == expected_name
        assert call["reviewer_email"] == expected_email
```

Run: `$PY -m pytest tests/unit/test_database/test_migration_136_resolved_at.py tests/unit/test_repositories/test_expert_review.py tests/api/test_expert_review_routes.py tests/unit/test_api/test_expert_review_detail_route.py tests/unit/test_database/test_migration_134_guarded_promote.py -q -p no:cacheprovider -n 0` → 82 passed; `$PY -m ruff check` + `ruff format --check` on the touched files; `$PY -m mypy --config-file pyproject.toml src/repositories/expert_review.py src/api/routes/expert_review.py src/api/schemas/expert_review.py` → 0 errors in those three files.

---

### Task 7: Regenerate the OpenAPI TypeScript types

**Files:**
- Modify: `frontend/src/types/generated/api.ts` (generated)

- [ ] **Step 1: Regenerate from the lane's app**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane1-review-loop
$PY -m scripts.export_openapi --output openapi.json
cd frontend && npx openapi-typescript ../openapi.json -o src/types/generated/api.ts && cd ..
rm -f openapi.json
git diff --stat frontend/src/types/generated/api.ts
grep -c 'get_expert_review\b' frontend/src/types/generated/api.ts
grep -c 'ExpertReviewDetailResponse' frontend/src/types/generated/api.ts
```

Expected: the diff adds the operation and the two schemas; both greps ≥ 1. If the export fails to import the app on this box for memory reasons, run it inside the container instead: `docker exec -w /app e2i_api python -m scripts.export_openapi --output /tmp/openapi.json && docker cp e2i_api:/tmp/openapi.json openapi.json` — but that exports the DEPLOYED app, not the lane; only use it to compare shapes, never to commit.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/types/generated/api.ts
git commit -m "chore(types): regenerate api.ts for the expert-review lookup route

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

(Re-run this task at the end of Task 11 if any later task touches a route or schema docstring; CI's `verify-types` gate compares the committed file with a fresh export.)

---

### Task 8: Frontend data layer — detail type, client, query key, hook

**Files:**
- Modify: `frontend/src/types/expert-review.ts` (append)
- Modify: `frontend/src/api/expert-review.ts` (append)
- Modify: `frontend/src/lib/query-client.ts` (`expertReviews` block ~line 401)
- Modify: `frontend/src/hooks/api/use-expert-review.ts`
- Modify: `frontend/src/hooks/api/use-expert-review.test.ts`

- [ ] **Step 1: Update the hook test file's mocks and add failing tests**

In `frontend/src/hooks/api/use-expert-review.test.ts`:

1. Add `getExpertReview: vi.fn(),` to the `vi.mock('@/api/expert-review', ...)` factory.
2. Add to the mocked `queryKeys.expertReviews`:

```ts
      detail: (reviewId: string) => ['e2i', 'expert-reviews', 'detail', reviewId] as const,
```

3. Add `useExpertReview` to the import from `'./use-expert-review'` and `ExpertReviewDetailResponse` to the type import.
4. In `useResolveReview` › `'posts a resolution and invalidates pending + summary queries'`, change the last assertion to `expect(invalidateSpy).toHaveBeenCalledTimes(3);` and its comment to `// invalidate the pending queue, the summary AND any open linked-review detail`.
5. Append:

```ts
const mockDetailResponse: ExpertReviewDetailResponse = {
  review: {
    review_id: 'rev-1',
    review_type: 'dag_approval',
    dag_version_hash: 'deadbeefcafebabe0123',
    brand: 'Kisqali',
    treatment_variable: 'treatment_arm',
    outcome_variable: 'persistent_180d',
    approval_status: 'rejected',
    reviewer_name: 'Dr. No',
    created_at: '2026-07-13T10:00:00Z',
  },
  history: [],
};

describe('useExpertReview', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches one review (any status) by id', async () => {
    vi.mocked(expertReviewApi.getExpertReview).mockResolvedValueOnce(mockDetailResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExpertReview('rev-1'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(mockDetailResponse);
    expect(expertReviewApi.getExpertReview).toHaveBeenCalledWith('rev-1');
  });

  it('stays idle without an id', () => {
    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useExpertReview(null), { wrapper });
    expect(result.current.fetchStatus).toBe('idle');
    expect(expertReviewApi.getExpertReview).not.toHaveBeenCalled();
  });
});
```

Run: `cd frontend && npx vitest run src/hooks/api/use-expert-review.test.ts` → Expected: FAIL (`useExpertReview` is not exported; the resolve test expects 3 invalidations).

- [ ] **Step 2: Types**

Append to `frontend/src/types/expert-review.ts`:

```ts
/**
 * One expert_reviews row in ANY status (GET /expert-reviews/{review_id}).
 * Extends the pending shape with the resolution columns.
 */
export interface ReviewRecord extends PendingReviewItem {
  /** pending / approved / rejected (expired is derived at read time from valid_until) */
  approval_status?: string | null;
  /**
   * The REQUESTER, not the resolver: the gate stores the originating query id
   * here (expert_review_gate.py create_review(reviewer_id=requester_id)).
   * Never render it as the reviewer.
   */
  reviewer_id?: string | null;
  /** The resolving operator's name, written on resolve; null when not recorded. */
  reviewer_name?: string | null;
  /** The resolving operator's email, written on resolve; null when not recorded. */
  reviewer_email?: string | null;
  /** Approval-only timestamp; a rejection never sets it. */
  approved_at?: string | null;
  /**
   * When the review was resolved, for BOTH statuses (migration 136). Null for
   * rows resolved before the migration — unknown stays unknown.
   */
  resolved_at?: string | null;
  valid_from?: string | null;
  valid_until?: string | null;
  concerns_raised?: string[] | null;
  conditions?: string | null;
  checklist_json?: Record<string, unknown> | null;
  comments_json?: Record<string, unknown> | null;
  supersedes_review_id?: string | null;
}

/**
 * Response for GET /expert-reviews/{review_id}: the row plus every review of
 * the same DAG structure (newest first, expired included).
 */
export interface ExpertReviewDetailResponse {
  review: ReviewRecord;
  history: ReviewRecord[];
}
```

- [ ] **Step 3: API client**

Append to `frontend/src/api/expert-review.ts` (and add `ExpertReviewDetailResponse` to its type import):

```ts
/**
 * One review in any status plus its same-structure history.
 *
 * @param reviewId - The review identifier
 */
export async function getExpertReview(reviewId: string): Promise<ExpertReviewDetailResponse> {
  return get<ExpertReviewDetailResponse>(`${EXPERT_REVIEW_BASE}/${encodeURIComponent(reviewId)}`);
}
```

Also add `- GET  /expert-reviews/{review_id}          : One review (any status) + same-DAG history` to the endpoint list in the file's header comment.

- [ ] **Step 4: Query key**

In `frontend/src/lib/query-client.ts`, inside `expertReviews`, add after `summary`:

```ts
    detail: (reviewId: string) =>
      [...queryKeys.expertReviews.all(), 'detail', reviewId] as const,
```

- [ ] **Step 5: Hook**

In `frontend/src/hooks/api/use-expert-review.ts`:

1. Add `getExpertReview` to the API import and `ExpertReviewDetailResponse` to the type import.
2. Append after `useReviewSummary`:

```ts
/**
 * Hook to fetch one review (any status) with its same-structure history —
 * the linked-review card behind the drill-down's deep link. Idle until an id
 * is present.
 */
export function useExpertReview(
  reviewId: string | null | undefined,
  options?: Omit<
    UseQueryOptions<ExpertReviewDetailResponse, ApiError>,
    'queryKey' | 'queryFn' | 'enabled'
  >
) {
  return useQuery<ExpertReviewDetailResponse, ApiError>({
    queryKey: queryKeys.expertReviews.detail(reviewId ?? ''),
    queryFn: () => getExpertReview(reviewId as string),
    enabled: !!reviewId,
    ...options,
  });
}
```

3. In `useResolveReview`'s `onSuccess`, add after the summary invalidation:

```ts
      // A resolution also changes any open linked-review card.
      queryClient.invalidateQueries({
        queryKey: [...queryKeys.expertReviews.all(), 'detail'],
      });
```

4. In `useReviewAssessment`'s `onSuccess`, add the same `detail` invalidation after the pending one.
5. Add `- useExpertReview:   read one review (any status) + same-DAG history` to the module header list.

- [ ] **Step 6: Run, typecheck, commit**

```bash
cd frontend && npx vitest run src/hooks/api/use-expert-review.test.ts && npm run typecheck && npx eslint src/hooks/api/use-expert-review.ts src/api/expert-review.ts src/types/expert-review.ts src/lib/query-client.ts && cd ..
git add frontend/src/types/expert-review.ts frontend/src/api/expert-review.ts frontend/src/lib/query-client.ts frontend/src/hooks/api/use-expert-review.ts frontend/src/hooks/api/use-expert-review.test.ts
git commit -m "feat(frontend): expert-review detail type, client, query key and useExpertReview hook

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

---

### Task 9: Drill-down — review state and discovered-DAG id

**Files:**
- Modify: `frontend/src/types/causal.ts` (`RefutationSummary` ~line 239; `AgentCausalAnalysisResponse` ~line 281)
- Create: `frontend/src/components/causal/ReviewStatusPanel.tsx`
- Create: `frontend/src/components/causal/ReviewStatusPanel.test.tsx`
- Modify: `frontend/src/components/causal/CausalAnalysisDetail.tsx` (after the discovered-confounders paragraph, ~line 409)
- Modify: `frontend/src/components/causal/CausalAnalysisDetail.test.tsx` (append two tests)

- [ ] **Step 1: Types**

In `RefutationSummary` add after `needs_review: boolean;`:

```ts
  /**
   * The expert-review row this run touched: the queue row on a REVIEW/BLOCK
   * gate, the approval row when one is active, or the rejection row when a
   * reviewer rejected the structure (any gate). Absent when none was involved.
   */
  expert_review_id?: string | null;
  /**
   * proceed / renewal_required / pending_review / rejected / blocked /
   * unavailable — the structural verdict (#1971). Approval never promotes a
   * borderline estimate; a rejection halts the run on every band.
   */
  expert_review_decision?: string | null;
```

In `AgentCausalAnalysisResponse` add after `dag_source?: string;`:

```ts
  /**
   * Row id of this run's durable discovery record in public.discovered_dags
   * (#1974). Absent when discovery did not run or persistence failed (then
   * `warnings` carries the reason).
   */
  discovered_dag_id?: string | null;
```

- [ ] **Step 2: Failing component tests**

Create `frontend/src/components/causal/ReviewStatusPanel.test.tsx`. Every case renders through `renderWithAllProviders` (frontend/src/test/utils.tsx): the panel's "Open review" deep link is a react-router `<Link>`, and `renderWithProviders` wraps only a `QueryClientProvider` — measured, the two link-rendering cases threw `TypeError: Cannot destructure property 'basename' of 'React10.useContext(...)' as it is null.` under it. The router wrapper renders no DOM of its own, so `toBeEmptyDOMElement` still holds for the null render:

```tsx
import { describe, it, expect, vi } from 'vitest';
// The "Open review" deep link is a router <Link>: render under the router-wrapped helper.
import { fireEvent, renderWithAllProviders, screen, waitFor } from '@/test/utils';
import { ReviewStatusPanel } from './ReviewStatusPanel';

const SWITCH_HALT =
  'Estimate withheld: CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=true requires an active expert approval of the DAG structure for a REVIEW-band estimate, and this structure holds none (gate decision: pending_review). Re-run once the review is resolved.';
const GATE_BLOCKED = 'Refutation gate BLOCKED — the estimate did not survive robustness checks.';

describe('ReviewStatusPanel', () => {
  it('renders nothing when the run carried no review state and no DAG record', () => {
    const { container } = renderWithAllProviders(<ReviewStatusPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows a pending structure with a deep link into the queue', () => {
    renderWithAllProviders(
      <ReviewStatusPanel decision="pending_review" reviewId="rev-9" discoveredDagId={null} />
    );
    expect(screen.getByText('Pending expert review')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /open review/i });
    expect(link).toHaveAttribute('href', '/expert-reviews?review=rev-9');
  });

  it('shows the rejection and the halt message from the run warnings', () => {
    renderWithAllProviders(
      <ReviewStatusPanel
        decision="rejected"
        reviewId="rev-rejected"
        warnings={[
          'Estimate withheld: a domain expert REJECTED this DAG structure by Dr. No (review rev-rejected): collider.',
          'Refutation gate BLOCKED — the estimate did not survive robustness checks.',
        ]}
      />
    );
    expect(screen.getByText('Structure rejected')).toBeInTheDocument();
    expect(screen.getByText(/by Dr\. No/)).toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
  });

  it('shows the durable discovery record id', () => {
    renderWithAllProviders(<ReviewStatusPanel discoveredDagId="8a61b3db-6aad-4b01-96e4-bbea0af861b4" />);
    expect(screen.getByText(/Durable discovery record/)).toBeInTheDocument();
    expect(screen.getByText('8a61b3db-6aad-4b01-96e4-bbea0af861b4')).toBeInTheDocument();
  });

  it('never invents a label for an unknown decision', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="something_new" />);
    expect(screen.getByText('something_new')).toBeInTheDocument();
  });

  // The run's warnings are rendered nowhere else in the drill-down, and the
  // approval-enforcement switch withholds the estimate on pending / blocked /
  // unavailable structures too (refutation.py:1215) — the halt must show for
  // any decision that carries one, not only a rejection.
  it.each(['pending_review', 'blocked', 'unavailable'])(
    'shows the approval-enforcement halt from the run warnings for %s',
    (decision) => {
      renderWithAllProviders(
        <ReviewStatusPanel decision={decision} warnings={[SWITCH_HALT, GATE_BLOCKED]} />
      );
      expect(screen.getByText(SWITCH_HALT)).toBeInTheDocument();
      expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
    }
  );

  it('shows no halt line for an approved structure whose warnings carry none', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="proceed" warnings={[GATE_BLOCKED]} />);
    expect(screen.getByText('Structure approved')).toBeInTheDocument();
    expect(screen.queryByText(/Estimate withheld/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
  });

  // A plain-object lookup resolves inherited members: "toString" must render
  // verbatim like any other unknown decision, never as an empty badge.
  it('renders an inherited-property decision verbatim, never an empty badge', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="toString" />);
    expect(screen.getByText('toString')).toBeInTheDocument();
  });

  it('copies the full discovery record id to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    renderWithAllProviders(
      <ReviewStatusPanel discoveredDagId="8a61b3db-6aad-4b01-96e4-bbea0af861b4" />
    );
    const button = screen.getByRole('button', { name: /copy discovery record id/i });
    expect(button).toHaveTextContent('Copy id');
    fireEvent.click(button);
    expect(writeText).toHaveBeenCalledWith('8a61b3db-6aad-4b01-96e4-bbea0af861b4');
    await waitFor(() => expect(button).toHaveTextContent('Copied'));
    // The full id stays visible beside the affordance.
    expect(screen.getByText('8a61b3db-6aad-4b01-96e4-bbea0af861b4')).toBeInTheDocument();
  });
});
```

Run: `cd frontend && npx vitest run src/components/causal/ReviewStatusPanel.test.tsx` → Expected: FAIL (module not found).

- [ ] **Step 3: Component**

Create `frontend/src/components/causal/ReviewStatusPanel.tsx`:

```tsx
/**
 * ReviewStatusPanel — the expert-review state of one agent run's DAG structure.
 * =============================================================================
 *
 * Renders ONLY what the API returned (spec §4.4): the structural verdict from
 * `refutation.expert_review_decision`, a link to the review row when the run
 * touched one, the expert-review halt message from `warnings` (a rejection,
 * the approval-enforcement switch, or the route's fallback — the run's
 * warnings are rendered nowhere else in the drill-down, so the halt shows for
 * any decision that carries one), and the durable discovered-DAG record id in
 * full with a copy affordance. Absent fields render nothing; an unknown
 * decision renders verbatim rather than a guessed label.
 *
 * @module components/causal/ReviewStatusPanel
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';

type Variant = 'default' | 'secondary' | 'destructive' | 'outline';

const DECISION_COPY: Record<string, { label: string; meaning: string; variant: Variant }> = {
  proceed: {
    label: 'Structure approved',
    meaning:
      'A reviewer approved this DAG structure. Approval covers the structure only; the estimate still stands or falls on its own robustness checks.',
    variant: 'default',
  },
  renewal_required: {
    label: 'Approval expiring',
    meaning:
      'The structural approval is inside its renewal window. A reviewer should renew it before it lapses.',
    variant: 'secondary',
  },
  pending_review: {
    label: 'Pending expert review',
    meaning:
      'This DAG structure is queued for a reviewer. Re-running the analysis does not change that; resolving the review does.',
    variant: 'secondary',
  },
  rejected: {
    label: 'Structure rejected',
    meaning:
      'A reviewer rejected this DAG structure. The estimate is withheld on every band until a reviewer reopens the structure.',
    variant: 'destructive',
  },
  blocked: {
    label: 'No review possible',
    meaning: 'The structure holds no approval and no review could be queued for it.',
    variant: 'outline',
  },
  unavailable: {
    label: 'Review gate unavailable',
    meaning:
      'The review store could not be consulted for this run; nothing was checked or queued.',
    variant: 'outline',
  },
};

export interface ReviewStatusPanelProps {
  decision?: string | null;
  reviewId?: string | null;
  discoveredDagId?: string | null;
  /**
   * The run's warnings; the expert-review halt message — a rejection, the
   * approval-enforcement switch, or the route's fallback — lives there.
   */
  warnings?: string[];
}

export function ReviewStatusPanel({
  decision,
  reviewId,
  discoveredDagId,
  warnings,
}: ReviewStatusPanelProps) {
  const [copied, setCopied] = useState(false);
  if (!decision && !discoveredDagId) return null;
  // Own-property lookup: a plain object resolves inherited members ("toString",
  // "constructor"), which would render an empty badge instead of the verbatim string.
  const copy =
    decision && Object.prototype.hasOwnProperty.call(DECISION_COPY, decision)
      ? DECISION_COPY[decision]
      : undefined;
  // At most one halt per run, and every producer of it starts with this prefix.
  const halt = (warnings ?? []).find((w) => w.startsWith('Estimate withheld'));

  return (
    <div
      className="space-y-1 rounded-md border border-[var(--color-border)] p-3 text-sm"
      data-testid="review-status"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">Review status</span>
        {copy ? (
          <Badge variant={copy.variant}>{copy.label}</Badge>
        ) : decision ? (
          <Badge variant="outline">{decision}</Badge>
        ) : null}
        {reviewId && (
          <Link
            to={`/expert-reviews?review=${encodeURIComponent(reviewId)}`}
            className="text-xs underline"
          >
            Open review
          </Link>
        )}
      </div>
      {copy && <p className="text-xs text-muted-foreground">{copy.meaning}</p>}
      {halt && <p className="text-xs text-muted-foreground">{halt}</p>}
      {discoveredDagId && (
        <p className="text-xs text-muted-foreground">
          Durable discovery record:{' '}
          <code className="font-mono text-[11px]">{discoveredDagId}</code>{' '}
          <button
            type="button"
            aria-label="Copy discovery record id"
            className="text-xs underline"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(discoveredDagId);
                setCopied(true);
              } catch {
                setCopied(false);
              }
            }}
          >
            {copied ? 'Copied' : 'Copy id'}
          </button>
        </p>
      )}
    </div>
  );
}
```

Why (Task 9 review fold, codex HIGH/MED/LOW): the halt line is found for ANY decision because the run's `warnings` are rendered nowhere else in the drill-down and the approval-enforcement switch withholds the estimate on pending/blocked/unavailable structures too (all producers start with `Estimate withheld`, at most one per run); the decision lookup is own-property because a plain-object index resolves inherited members (`"toString"` would render an empty badge instead of the verbatim string); the DAG id stays visible in full (the exact lineage handle) with a clipboard copy affordance per spec §4.4.

- [ ] **Step 4: Wire into the detail view and add its test**

In `CausalAnalysisDetail.tsx`, add the import `import { ReviewStatusPanel } from './ReviewStatusPanel';` next to the `ClinicalContextPanel` import, and insert directly before `<ConfoundingAdjustmentPanel result={result} />`:

```tsx
      <ReviewStatusPanel
        decision={result.refutation.expert_review_decision}
        reviewId={result.refutation.expert_review_id}
        discoveredDagId={result.discovered_dag_id}
        warnings={result.warnings}
      />
```

Append to `CausalAnalysisDetail.test.tsx` (inside `describe('CausalAnalysisDetail', …)`). The positive test renders through `renderWithAllProviders` (imported beside the file's existing `renderWithProviders`) because it renders the panel's router `<Link>`; the negative test stays on `renderWithProviders`, which also proves the detail view still renders without a router when the run carries no review id:

```tsx
  it('surfaces the review state and the discovered-DAG record when the run carries them', () => {
    // The panel's "Open review" deep link is a router <Link>; render under the router.
    renderWithAllProviders(
      <CausalAnalysisDetail
        result={{
          ...RESULT,
          discovered_dag_id: 'dag-123',
          refutation: {
            ...RESULT.refutation,
            expert_review_decision: 'pending_review',
            expert_review_id: 'rev-9',
          },
        }}
      />
    );
    // Positive control for the negative test below: the block carries this testid.
    expect(screen.getByTestId('review-status')).toBeInTheDocument();
    expect(screen.getByText('Pending expert review')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /open review/i })).toHaveAttribute(
      'href',
      '/expert-reviews?review=rev-9'
    );
    expect(screen.getByText('dag-123')).toBeInTheDocument();
  });

  it('renders no review block for a run that touched no review and persisted no DAG', () => {
    renderWithProviders(<CausalAnalysisDetail result={RESULT} />);
    expect(screen.queryByTestId('review-status')).not.toBeInTheDocument();
  });
```

- [ ] **Step 5: Run, typecheck, lint, commit**

```bash
cd frontend && npx vitest run src/components/causal && npm run typecheck && npx eslint src/components/causal/ReviewStatusPanel.tsx src/components/causal/CausalAnalysisDetail.tsx src/types/causal.ts && cd ..
git add frontend/src/types/causal.ts frontend/src/components/causal/ReviewStatusPanel.tsx frontend/src/components/causal/ReviewStatusPanel.test.tsx frontend/src/components/causal/CausalAnalysisDetail.tsx frontend/src/components/causal/CausalAnalysisDetail.test.tsx
git commit -m "feat(frontend): review status and discovered-DAG id on the causal drill-down

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

---

### Task 10: Queue page — linked-review card, brand filter, summary error, assessment prefetch

The page currently holds `CHECKLIST_ITEMS`, `shortHash`, `VERDICT_VARIANT`, `DagPanel` and `ResolveForm` inline. This task moves them into `frontend/src/components/expert-review/` (one responsibility per file), adds two new components, and rewrites the page around them. Behaviour of the moved code is unchanged except the auto-assessment in `ResolveForm`.

**Files:**
- Create: `frontend/src/components/expert-review/checklist.ts`
- Create: `frontend/src/components/expert-review/DagPanel.tsx`
- Create: `frontend/src/components/expert-review/ResolveForm.tsx`
- Create: `frontend/src/components/expert-review/LinkedReviewCard.tsx`
- Create: `frontend/src/components/expert-review/PrepareAssessmentsButton.tsx`
- Modify: `frontend/src/pages/ExpertReviews.tsx` (full rewrite)
- Modify: `frontend/src/pages/ExpertReviews.test.tsx` (full rewrite)

- [ ] **Step 1: Shared checklist constants**

Create `frontend/src/components/expert-review/checklist.ts`:

```ts
/**
 * The minimal reviewer checklist (the migration-010 SYSTEM_TEMPLATE required
 * items) and the advisory-verdict chip styling shared by the queue page and
 * the linked-review card. Ids MUST stay in sync with
 * src/insights/expert_review_assessment.py CHECKLIST_QUESTIONS.
 */
import type { AssessmentVerdict } from '@/types/expert-review';

export const CHECKLIST_ITEMS: { id: string; question: string }[] = [
  { id: 'conf_complete', question: 'Are all known confounders included?' },
  { id: 'edge_plausible', question: 'Do causal arrows reflect domain knowledge?' },
  { id: 'no_forbidden', question: 'Are there no forbidden edges (future→past)?' },
  { id: 'mediators_correct', question: 'Are intermediate variables correctly positioned?' },
  { id: 'sutva_plausible', question: 'Is the no-interference assumption reasonable?' },
  { id: 'positivity', question: 'Is there sufficient overlap in treatment groups?' },
];

/** Concern is the only destructive signal; the other verdicts stay visually calm. */
export const VERDICT_VARIANT: Record<AssessmentVerdict, 'secondary' | 'destructive' | 'outline'> = {
  supports: 'secondary',
  concern: 'destructive',
  unclear: 'outline',
  no_evidence: 'outline',
};

export function shortHash(hash?: string | null): string {
  if (!hash) return '—';
  return hash.length > 12 ? `${hash.slice(0, 12)}…` : hash;
}

/** approved / rejected / pending / anything else → badge variant. */
export function statusVariant(status?: string | null): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (status === 'approved') return 'default';
  if (status === 'rejected') return 'destructive';
  if (status === 'pending') return 'secondary';
  return 'outline';
}
```

- [ ] **Step 2: `DagPanel` (moved verbatim)**

Create `frontend/src/components/expert-review/DagPanel.tsx`:

```tsx
/** Render a review's stored DAG snapshot, or an honest fallback for pre-097 rows. */
import { CausalDAG } from '@/components/visualizations/causal/CausalDAG';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import type { DagStructure } from '@/types/expert-review';

export function DagPanel({ structure }: { structure?: DagStructure | null }) {
  if (!structure?.nodes?.length) {
    return (
      <div className="rounded-md border border-dashed border-[var(--color-border)] p-4 text-sm text-[var(--color-muted-foreground)]">
        DAG structure not captured for this review (created before snapshot capture was
        added). The DAG hash identifies the structure but cannot be rendered from it.
      </div>
    );
  }

  const treatments = new Set(structure.treatment_nodes ?? []);
  const outcomes = new Set(structure.outcome_nodes ?? []);
  const augmented = new Set((structure.augmented_edges ?? []).map(([s, t]) => `${s}->${t}`));

  const nodes: CausalNode[] = structure.nodes.map((id) => ({
    id,
    label: id,
    type: treatments.has(id) ? 'treatment' : outcomes.has(id) ? 'outcome' : 'variable',
  }));
  const edges: CausalEdge[] = (structure.edges ?? []).map(([source, target]) => ({
    id: `${source}->${target}`,
    source,
    target,
    // Discovery-augmented edges are visually distinct: the discovery gate added them.
    type: augmented.has(`${source}->${target}`) ? 'association' : 'causal',
  }));

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">DAG under review</h4>
      <CausalDAG nodes={nodes} edges={edges} minHeight={320} ariaLabel="Causal DAG under review" />
      {structure.augmented_edges && structure.augmented_edges.length > 0 && (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Dashed/association edges were discovery-augmented (gate=
          {structure.discovery_gate_decision ?? 'unknown'}).
        </p>
      )}
    </div>
  );
}
```

- [ ] **Step 3: `ResolveForm` (moved, plus auto-assessment on mount)**

Create `frontend/src/components/expert-review/ResolveForm.tsx`:

```tsx
/**
 * Approve / reject one pending review with the 010 checklist and an advisory
 * agent assessment. The assessment is generated automatically the first time
 * the form mounts for a row without a cached one (spec §4.5) — once per
 * review id, StrictMode-safe — and can be regenerated on demand. It never
 * pre-fills the human checklist.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import { CheckCircle2, RefreshCw, Sparkles, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { useResolveReview, useReviewAssessment } from '@/hooks/api/use-expert-review';
import type { AgentAssessment, PendingReviewItem, ReviewApprovalStatus } from '@/types/expert-review';
import { CHECKLIST_ITEMS, VERDICT_VARIANT } from './checklist';

export interface ResolveFormProps {
  review: PendingReviewItem;
  onClose: () => void;
  /**
   * Page-level once-per-review-id guard for the auto-generated assessment. The
   * linked-review card and the queue row can both mount a form for the SAME
   * review; sharing one Set keeps the backend from building the assessment
   * twice (pre-execution review 2026-09-08, codex MED). Optional so the form
   * still guards itself when rendered alone.
   */
  autoAssessGuard?: MutableRefObject<Set<string>>;
}

export function ResolveForm({ review, onClose, autoAssessGuard }: ResolveFormProps) {
  // Unique per FORM INSTANCE: the linked card and the queue row can render the
  // same review, and duplicate element ids would let a label operate the other
  // form (pre-execution review iter-2, codex MED).
  const uid = useId();
  const [checklist, setChecklist] = useState<Record<string, boolean>>({});
  const [comments, setComments] = useState('');
  const resolve = useResolveReview();

  // Once-per-review-id guard for the auto-generated assessment (spec §4.5): a
  // Set shared by every form on the page when the page provides one, so a second
  // form for the same id (linked card + queue row) and StrictMode's double-run
  // effect both hit it. Defined BEFORE the mutation hook so onError can use it.
  const localGuard = useRef<Set<string>>(new Set());
  const guard = autoAssessGuard ?? localGuard;
  const assessmentMutation = useReviewAssessment({
    // Hook-level, not per-call: the Mutation itself runs this, so it fires even
    // after the form has unmounted (a collapsed row), whereas TanStack skips the
    // per-call mutate callbacks once the observer is gone. Releasing the id lets
    // a later expand, or the page's Prepare button, retry the FAILED generation
    // once (the effect's deps do not change on error, so there is no loop).
    // OWNERSHIP: only the AUTOMATIC request's failure releases the id. A manual
    // Generate from a second form for the same review (linked card + queue row)
    // fails independently while the automatic request may still be in flight;
    // releasing then would let a remount or Prepare start a duplicate request.
    onError: (_error, variables) => {
      if (variables.auto) guard.current.delete(variables.reviewId);
    },
  });
  const { mutate: generateAssessment } = assessmentMutation;

  // Prefer the freshly generated assessment; fall back to the row's cache.
  const assessment: AgentAssessment | null =
    assessmentMutation.data?.assessment ?? review.agent_assessment_json ?? null;
  const assessmentById = new Map((assessment?.items ?? []).map((item) => [item.id, item]));

  useEffect(() => {
    if (assessment) return;
    if (guard.current.has(review.review_id)) return;
    // Deferred past StrictMode's synchronous effect cleanup + re-run. A mutate
    // issued in the FIRST pass is orphaned: query-core's MutationObserver
    // detaches from the in-flight mutation on unsubscribe and never re-attaches
    // (mutationObserver.js onUnsubscribe), so the form stayed pending after the
    // request had completed (measured under <StrictMode>, which main.tsx uses).
    // The guard is marked when the timer fires, so a cancelled pass marks nothing.
    const timer = setTimeout(() => {
      if (guard.current.has(review.review_id)) return;
      guard.current.add(review.review_id);
      generateAssessment({ reviewId: review.review_id, auto: true });
    }, 0);
    return () => clearTimeout(timer);
  }, [assessment, generateAssessment, guard, review.review_id]);

  const submit = useCallback(
    (approval_status: ReviewApprovalStatus) => {
      resolve.mutate(
        {
          reviewId: review.review_id,
          body: {
            approval_status,
            checklist,
            comments: comments ? { note: comments } : undefined,
          },
        },
        { onSuccess: onClose }
      );
    },
    [resolve, review.review_id, checklist, comments, onClose]
  );

  return (
    <div className="space-y-4 rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]/20 p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
          <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
          Agent assessment (advisory — the checklist answers are yours)
          {assessment?.is_fallback && ' · deterministic, no LLM'}
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => generateAssessment({ reviewId: review.review_id, force: !!assessment })}
          disabled={assessmentMutation.isPending}
        >
          <RefreshCw
            className={`mr-1 h-3.5 w-3.5 ${assessmentMutation.isPending ? 'animate-spin' : ''}`}
          />
          {assessment ? 'Regenerate agent assessment' : 'Generate agent assessment'}
        </Button>
      </div>

      {assessmentMutation.isError && (
        <WarningBanner
          title="Failed to generate agent assessment"
          messages={[assessmentMutation.error?.message ?? 'An unexpected error occurred.']}
        />
      )}

      <div className="space-y-2">
        {CHECKLIST_ITEMS.map((item) => {
          const graded = assessmentById.get(item.id);
          return (
            <div key={item.id} className="space-y-0.5">
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`${uid}-${item.id}`}
                  checked={!!checklist[item.id]}
                  onCheckedChange={(v) =>
                    setChecklist((prev) => ({ ...prev, [item.id]: v === true }))
                  }
                />
                <Label htmlFor={`${uid}-${item.id}`} className="text-sm">
                  {item.question}
                </Label>
                {graded && (
                  <Badge variant={VERDICT_VARIANT[graded.verdict] ?? 'outline'}>{graded.verdict}</Badge>
                )}
              </div>
              {graded && (
                <p className="pl-6 text-xs text-[var(--color-muted-foreground)]">{graded.rationale}</p>
              )}
            </div>
          );
        })}
      </div>

      <div className="space-y-1">
        <Label htmlFor={`${uid}-comments`} className="text-sm">
          Comments
        </Label>
        <textarea
          id={`${uid}-comments`}
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          rows={3}
          className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 text-sm"
          placeholder="Reviewer notes (optional)"
        />
      </div>

      {resolve.isError && (
        <WarningBanner
          title="Failed to submit review"
          messages={[resolve.error?.message ?? 'An unexpected error occurred.']}
        />
      )}

      <div className="flex items-center gap-2">
        <Button size="sm" onClick={() => submit('approved')} disabled={resolve.isPending}>
          <CheckCircle2 className="mr-1 h-4 w-4" />
          Approve
        </Button>
        <Button size="sm" variant="destructive" onClick={() => submit('rejected')} disabled={resolve.isPending}>
          <XCircle className="mr-1 h-4 w-4" />
          Reject
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose} disabled={resolve.isPending}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
```

Why (Task 10 review fold, codex 3×MED + review): the auto-assessment guard is released on a FAILED generation through the hook-level `onError` (the guard is defined before the hook so the closure can reach it) because the Mutation itself runs hook-level callbacks even after the form has unmounted, whereas TanStack skips per-call `mutate(vars, { onError })` once the observer is gone — a collapsed row could otherwise never retry and Prepare silently dropped the row (`missing` counted it, `todo` excluded it); the auto-assessment is issued from a zero-delay timer because query-core 5.90 `MutationObserver.onUnsubscribe` detaches from the in-flight mutation and nothing re-attaches on re-subscribe, so under `<StrictMode>` (main.tsx) a `mutate` in the first effect pass left the form pending forever after the request had completed (measured with real hooks: the guard held at one POST, but the Generate button stayed disabled with its spinner and the result never reached the form); the summary banner REPLACES the badges (`summary.data && !summary.isError`) because TanStack keeps the last data on a refetch error and spec §4.5 says the banner replaces the counts; the Prepare button lost its `aria-label`, which masked the visible "(N missing)" / "Preparing k / n" for screen readers; the resolved-copy wording no longer reads as if the row itself reopens; the component tests (Step 7b) use the REAL hooks and deferred promises so sequencing, the first-error stop, the guard release and Stop are observed mid-flight, which the page test (hooks mocked) cannot do; and the backend checklist comment (`src/insights/expert_review_assessment.py`) points at the constants' new home. Second fold (codex iter-2 MED+LOW, review): the guard is released only by the failing AUTOMATIC request (`auto: true` on `ReviewAssessmentVariables`, set by the effect's timer and never by the manual button) because a second form for the same review (linked card + queue row) can fail a MANUAL Generate while the automatic request is still in flight, and an unconditional release would let a remount or Prepare start a duplicate LM call (`PrepareAssessmentsButton` needs no change: it skips guarded ids and deletes only the ids it added itself); the resolved-review copy is status-specific because `check_approval` (src/causal_engine/expert_review_gate.py ~:272-350) consults the ACTIVE approval unless a NEWER rejection supersedes it, and a newer pending row reopens a REJECTED structure without displacing an approval, so a single sentence overstated the gate; and the test files carry a real-timers note above their timer helpers (Node orders the coerced zero-delay timer before the 10 ms wait; `vi.useFakeTimers` would break it).

- [ ] **Step 4: `LinkedReviewCard`**

Create `frontend/src/components/expert-review/LinkedReviewCard.tsx`:

```tsx
/**
 * The review the causal drill-down linked to (`/expert-reviews?review=<id>`):
 * one row in ANY status plus every review of the same DAG structure. A pending
 * linked review resolves in place; a resolved one shows who decided what.
 */
import type { MutableRefObject } from 'react';
import { RefreshCw } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { useExpertReview } from '@/hooks/api/use-expert-review';
import { DagPanel } from './DagPanel';
import { ResolveForm } from './ResolveForm';
import { shortHash, statusVariant } from './checklist';

function fmtDate(value?: string | null): string {
  if (!value) return '—';
  return value.slice(0, 10);
}

/**
 * Provenance rule (codex whole-diff HIGH F1): render only RECORDED provenance.
 * `reviewer_id` holds the REQUESTER (the originating query id the gate wrote,
 * expert_review_gate.py create_review(reviewer_id=requester_id)) and
 * `created_at` is not a decision time, so neither may stand in for the
 * reviewer or the decision. Unknown stays "not recorded".
 */
const NOT_RECORDED = 'not recorded';

function reviewerLabel(row: { reviewer_name?: string | null; reviewer_email?: string | null }): string | null {
  return row.reviewer_name ?? row.reviewer_email ?? null;
}

function decidedLabel(row: { resolved_at?: string | null; approved_at?: string | null }): string {
  const when = row.resolved_at ?? row.approved_at;
  return when ? fmtDate(when) : NOT_RECORDED;
}

/**
 * The reviewer's reason, faithfully: the resolve form sends `{ note }`, so a
 * string note is shown as written; any other non-empty object is shown as its
 * JSON, never paraphrased (codex F2).
 */
function commentsLabel(comments?: Record<string, unknown> | null): string {
  if (!comments) return '—';
  if (typeof comments.note === 'string') return comments.note;
  return Object.keys(comments).length > 0 ? JSON.stringify(comments) : '—';
}

/**
 * Status-specific resolved copy matching the gate's precedence
 * (src/causal_engine/expert_review_gate.py check_approval, ~:272-350): the
 * ACTIVE approval governs unless a NEWER rejection supersedes it; a newer
 * pending row reopens a REJECTED structure but does not displace an approval.
 */
function resolvedCopy(status?: string | null): string {
  if (status === 'approved') {
    return 'This review is resolved. Its approval applies until it expires or a newer review rejects the structure.';
  }
  if (status === 'rejected') {
    return 'This review is resolved. The rejection holds until a newer pending review of the same structure reopens it.';
  }
  return 'This review is resolved.';
}

export function LinkedReviewCard({
  reviewId,
  autoAssessGuard,
}: {
  reviewId: string;
  autoAssessGuard?: MutableRefObject<Set<string>>;
}) {
  const q = useExpertReview(reviewId);
  // The history INCLUDES the linked review itself (GET /expert-reviews/{id});
  // hoisted so the row marker survives TS narrowing inside the map callback.
  const currentId = q.data?.review.review_id;
  // Branch on the HTTP status (ApiError.status); the message text is only a
  // fallback for an error that carries no status at all.
  const notFound =
    q.error?.status === 404 ||
    (q.error?.status === undefined && /not found/i.test(q.error?.message ?? ''));

  return (
    <Card data-testid="linked-review">
      <CardHeader>
        <CardTitle>Linked review</CardTitle>
        <CardDescription>
          Opened from a causal analysis. Review <span className="font-mono">{shortHash(reviewId)}</span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {q.isLoading && (
          <div className="flex items-center justify-center py-6">
            <RefreshCw className="h-5 w-5 animate-spin text-[var(--color-muted-foreground)]" />
          </div>
        )}
        {q.isError && (
          <WarningBanner
            title={notFound ? 'This review no longer exists' : 'Failed to load the linked review'}
            messages={[q.error?.message ?? 'An unexpected error occurred.']}
          />
        )}
        {q.data && (
          <>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge variant={statusVariant(q.data.review.approval_status)}>
                {q.data.review.approval_status ?? 'unknown'}
              </Badge>
              <span>{q.data.review.brand ?? 'no brand'}</span>
              <span>·</span>
              <span>
                {q.data.review.treatment_variable ?? '—'} → {q.data.review.outcome_variable ?? '—'}
              </span>
              <span>·</span>
              <span>created {fmtDate(q.data.review.created_at)}</span>
            </div>
            {q.data.review.approval_status !== 'pending' && (
              <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
                <dt className="text-[var(--color-muted-foreground)]">Reviewer</dt>
                <dd>{reviewerLabel(q.data.review) ?? NOT_RECORDED}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Decided</dt>
                <dd>{decidedLabel(q.data.review)}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Valid until</dt>
                <dd>{q.data.review.valid_until ? fmtDate(q.data.review.valid_until) : 'no expiry recorded'}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Concerns</dt>
                <dd>{q.data.review.concerns_raised?.length ? q.data.review.concerns_raised.join('; ') : '—'}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Conditions</dt>
                <dd>{q.data.review.conditions ?? '—'}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Comments</dt>
                <dd className="whitespace-pre-wrap">{commentsLabel(q.data.review.comments_json)}</dd>
              </dl>
            )}
            <div className="grid gap-4 xl:grid-cols-2">
              <DagPanel structure={q.data.review.dag_structure_json} />
              {q.data.review.approval_status === 'pending' ? (
                <ResolveForm
                  review={q.data.review}
                  onClose={() => undefined}
                  autoAssessGuard={autoAssessGuard}
                />
              ) : (
                <div className="text-sm text-[var(--color-muted-foreground)]">
                  {resolvedCopy(q.data.review.approval_status)}
                </div>
              )}
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Same structure, all reviews</h4>
              {q.data.history.length === 0 ? (
                <p className="text-xs text-[var(--color-muted-foreground)]">No other reviews share this DAG hash.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Review</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Reviewer</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {q.data.history.map((h) => {
                      const isCurrent = h.review_id === currentId;
                      return (
                        <TableRow key={h.review_id} data-current={isCurrent ? 'true' : undefined}>
                          <TableCell className="font-mono text-xs">
                            {shortHash(h.review_id)}
                            {isCurrent && (
                              <span className="ml-1 text-[var(--color-muted-foreground)]">(this review)</span>
                            )}
                          </TableCell>
                          <TableCell>
                            <Badge variant={statusVariant(h.approval_status)}>{h.approval_status ?? '—'}</Badge>
                          </TableCell>
                          <TableCell>{fmtDate(h.created_at)}</TableCell>
                          <TableCell>{reviewerLabel(h) ?? '—'}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
```

Why (Task 12 whole-diff fold, codex 1 HIGH + 3 MED + 1 LOW): the linked-review card renders ONLY RECORDED provenance -- Reviewer is `reviewer_name`, else `reviewer_email`, else `not recorded`; Decided is `resolved_at`, else `approved_at`, else `not recorded`; the history table's Reviewer cell follows the same rule -- because `reviewer_id` holds the REQUESTER (the originating query id the gate writes, `expert_review_gate.py` ~:392) and `created_at` is not a decision time, so the previous fallbacks (`reviewer_name ?? reviewer_id`, `approved_at ?? created_at`) would have named a query id as the reviewer and the creation date as the decision on every resolved row (0 of 40 live rows carry `reviewer_name`; a rejection wrote no timestamp). Unknown stays `not recorded`: the backend now records the resolver and `resolved_at` (Task 6 Step 5, migration 136) and the card shows what was recorded, never a stand-in. The card also shows the reviewer's `comments_json.note` (what `ResolveForm` sends) or a non-empty object's JSON verbatim (F2). Bulk Prepare reads the response: on `persisted: false` (HTTP 200, the store rejected the cache write) it releases the guard for that id, stops the walk and names the review that was generated but not saved -- before, the id stayed guarded and every later Prepare skipped the uncached row (F3); and it invalidates the detail prefix as well as the pending prefix on both the success and the error path, mirroring `useReviewAssessment`, so the linked card refreshes (F4; the test pins `toHaveBeenCalledWith` the prefix, not a count). Task 1's Step 4/6/7/8/9 blocks are regenerated from the shipped runner (`np.ptp` zero-spread guards and finite-CI guards; no `np.std` / `1e-10` floor) so the plan describes the code that shipped (F5).

- [ ] **Step 5: `PrepareAssessmentsButton`**

Create `frontend/src/components/expert-review/PrepareAssessmentsButton.tsx`:

```tsx
/**
 * Walk the visible pending rows that have no cached assessment and generate
 * one each, ONE AT A TIME (each call is cached server-side). Shows k / n,
 * is cancellable, stops on the first error and shows it. No new endpoint.
 */
import { useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Sparkles, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { generateReviewAssessment } from '@/api/expert-review';
import { queryKeys } from '@/lib/query-client';
import type { PendingReviewItem } from '@/types/expert-review';
import { shortHash } from './checklist';

interface RunState {
  running: boolean;
  done: number;
  total: number;
  error: string | null;
}

export function PrepareAssessmentsButton({
  reviews,
  autoAssessGuard,
}: {
  reviews: PendingReviewItem[];
  /** The page's once-per-review-id guard; marked before each request so a form
   *  expanded meanwhile does not start a second generation for the same review. */
  autoAssessGuard?: MutableRefObject<Set<string>>;
}) {
  const queryClient = useQueryClient();
  const cancelRef = useRef(false);
  const [state, setState] = useState<RunState>({ running: false, done: 0, total: 0, error: null });
  const missing = reviews.filter((r) => !r.agent_assessment_json);

  // Mirrors useReviewAssessment (use-expert-review.ts): a fresh assessment
  // changes the pending queue AND the linked card's detail query (F4).
  const invalidate = async () => {
    await queryClient.invalidateQueries({ queryKey: [...queryKeys.expertReviews.all(), 'pending'] });
    await queryClient.invalidateQueries({ queryKey: [...queryKeys.expertReviews.all(), 'detail'] });
  };

  const run = async () => {
    cancelRef.current = false;
    // Skip reviews whose generation a form already started (shared guard); a
    // second request for the same id would only race the first.
    const todo = missing.filter((r) => !autoAssessGuard?.current.has(r.review_id));
    setState({ running: true, done: 0, total: todo.length, error: null });
    for (const review of todo) {
      if (cancelRef.current) break;
      // Re-check per iteration: a form expanded while an earlier request was
      // in flight may have started this one meanwhile.
      if (autoAssessGuard?.current.has(review.review_id)) {
        setState((s) => ({ ...s, done: s.done + 1 }));
        continue;
      }
      try {
        autoAssessGuard?.current.add(review.review_id);
        const result = await generateReviewAssessment(review.review_id);
        if (result.persisted === false) {
          // HTTP 200 with a valid assessment the store did not keep (F3). The
          // row still lacks a cache, so treat it like an error: release the id
          // (a guarded id would be skipped by every later Prepare) and stop.
          autoAssessGuard?.current.delete(review.review_id);
          setState((s) => ({
            ...s,
            running: false,
            error: `Assessment for review ${shortHash(review.review_id)} was generated but not saved (the store rejected the write). Retry from the row's Generate button or run Prepare again.`,
          }));
          await invalidate();
          return;
        }
        setState((s) => ({ ...s, done: s.done + 1 }));
      } catch (e) {
        // Release the id so a later "Prepare" (or the form's button) can retry it.
        autoAssessGuard?.current.delete(review.review_id);
        setState((s) => ({
          ...s,
          running: false,
          error: e instanceof Error ? e.message : 'Assessment generation failed.',
        }));
        await invalidate();
        return;
      }
    }
    setState((s) => ({ ...s, running: false }));
    await invalidate();
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        size="sm"
        variant="outline"
        onClick={run}
        disabled={state.running || missing.length === 0}
      >
        <Sparkles className="mr-1 h-4 w-4" />
        {state.running
          ? `Preparing ${state.done} / ${state.total}…`
          : `Prepare assessments (${missing.length} missing)`}
      </Button>
      {state.running && (
        <Button size="sm" variant="ghost" onClick={() => (cancelRef.current = true)}>
          <Square className="mr-1 h-3.5 w-3.5" />
          Stop
        </Button>
      )}
      {state.error && (
        <WarningBanner
          title={`Stopped after ${state.done} of ${state.total}`}
          messages={[state.error]}
          className="basis-full"
        />
      )}
    </div>
  );
}
```

- [ ] **Step 6: Rewrite the page**

Replace the full contents of `frontend/src/pages/ExpertReviews.tsx` with:

```tsx
/**
 * Expert Reviews Page (R6-F2 Phase B4; DAG snapshot + advisory assessment 097;
 * lane 1: linked review, brand filter, honest summary, assessment prefetch)
 * ============================================================================
 *
 * Admin review-queue UI for the causal-DAG human-in-the-loop loop.
 *
 * A REVIEW- or BLOCK-band causal estimate creates a `pending` expert_reviews
 * row; an operator sees it here and resolves it (approve/reject) with the 010
 * checklist items + comments. The expanded row renders the DAG under review
 * from its stored snapshot and an ADVISORY agent assessment that never
 * pre-fills the human checklist.
 *
 * Lane 1 (spec §4.5):
 * - `?review=<id>` opens a linked-review card (any status + same-DAG history),
 *   the destination of the causal drill-down's "Open review" link.
 * - The queue and the counts follow the GLOBAL brand filter (the same SSOT the
 *   Causal Analysis page reads, #1752). "All" is the only way to see the rows
 *   that carry no brand.
 * - A summary read failure renders a banner instead of silently dropping the
 *   counts.
 * - "Prepare assessments" generates the missing advisory assessments one row
 *   at a time; expanding a row generates its own if none is cached.
 *
 * Honest states: loading spinner, error banner, and an EmptyState (no hardcoded
 * SAMPLE_ data) when the live queue is empty.
 *
 * @module pages/ExpertReviews
 */

import { Fragment, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ClipboardCheck, Inbox, RefreshCw } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/EmptyState';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { DagPanel } from '@/components/expert-review/DagPanel';
import { LinkedReviewCard } from '@/components/expert-review/LinkedReviewCard';
import { PrepareAssessmentsButton } from '@/components/expert-review/PrepareAssessmentsButton';
import { ResolveForm } from '@/components/expert-review/ResolveForm';
import { shortHash } from '@/components/expert-review/checklist';
import { usePendingReviews, useReviewSummary } from '@/hooks/api/use-expert-review';
import { useE2IFilters } from '@/hooks/use-e2i-filters';

export default function ExpertReviews() {
  const [searchParams] = useSearchParams();
  const linkedReviewId = searchParams.get('review')?.trim() || null;

  const { filters } = useE2IFilters();
  const brand = filters.brand === 'All' ? undefined : (filters.brand as string);
  const params = brand ? { brand } : undefined;

  const { data, isLoading, isError, error, refetch, isFetching } = usePendingReviews(params);
  const summary = useReviewSummary(params);
  const [openRow, setOpenRow] = useState<string | null>(null);
  // One auto-generated assessment per review id across the page (the linked
  // card and a queue row can show the same review).
  const autoAssessGuard = useRef<Set<string>>(new Set());

  const reviews = data?.reviews ?? [];

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <ClipboardCheck className="h-6 w-6" />
            Expert Reviews
          </h1>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Human-in-the-loop validation queue for causal DAGs awaiting expert sign-off.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw className={`mr-1 h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {summary.isError && (
        <WarningBanner
          title="Review counts unavailable"
          messages={[summary.error?.message ?? 'An unexpected error occurred.']}
        />
      )}
      {/* TanStack keeps the last data on a refetch error; the banner REPLACES the
          counts (spec §4.5) rather than sitting above stale ones. */}
      {summary.data && !summary.isError && (
        <div className="flex flex-wrap gap-2">
          {/* pending/approved/rejected/expired partition the rows; expiring_soon
              is a SUBSET of approved (#1972), so it is labelled and styled as a
              qualifier rather than a fourth peer count that could be added in. */}
          <Badge variant="secondary">Pending: {summary.data.pending}</Badge>
          <Badge variant="secondary">Approved: {summary.data.approved}</Badge>
          <Badge variant="secondary">Rejected: {summary.data.rejected}</Badge>
          <Badge variant="secondary">Expired: {summary.data.expired}</Badge>
          <Badge variant="outline">of which expiring soon: {summary.data.expiring_soon}</Badge>
        </div>
      )}

      {linkedReviewId && (
        <LinkedReviewCard reviewId={linkedReviewId} autoAssessGuard={autoAssessGuard} />
      )}

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div>
              <CardTitle>Pending Queue</CardTitle>
              <CardDescription>
                {brand
                  ? `Oldest reviews first · brand: ${brand}. Reviews with no brand are listed under All.`
                  : 'Oldest reviews first · all brands, including reviews with no brand.'}
              </CardDescription>
            </div>
            {reviews.length > 0 && (
              <PrepareAssessmentsButton reviews={reviews} autoAssessGuard={autoAssessGuard} />
            )}
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="h-6 w-6 animate-spin text-[var(--color-muted-foreground)]" />
            </div>
          ) : isError ? (
            <WarningBanner
              title="Failed to load pending reviews"
              messages={[error?.message ?? 'An unexpected error occurred.']}
            />
          ) : reviews.length === 0 ? (
            <EmptyState
              icon={<Inbox className="h-8 w-8" aria-hidden="true" />}
              title="No pending reviews"
              description="REVIEW-band causal estimates will appear here for expert sign-off."
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Brand</TableHead>
                  <TableHead>Treatment</TableHead>
                  <TableHead>Outcome</TableHead>
                  <TableHead>DAG hash</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Age (days)</TableHead>
                  <TableHead className="text-right">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {reviews.map((review) => (
                  <Fragment key={review.review_id}>
                    <TableRow>
                      <TableCell>{review.brand ?? '—'}</TableCell>
                      <TableCell>{review.treatment_variable ?? '—'}</TableCell>
                      <TableCell>{review.outcome_variable ?? '—'}</TableCell>
                      <TableCell className="font-mono text-xs">{shortHash(review.dag_version_hash)}</TableCell>
                      <TableCell>{review.review_type ?? '—'}</TableCell>
                      <TableCell>
                        {review.days_pending != null ? Math.round(review.days_pending) : '—'}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            setOpenRow((prev) => (prev === review.review_id ? null : review.review_id))
                          }
                        >
                          {openRow === review.review_id ? 'Close' : 'Review'}
                        </Button>
                      </TableCell>
                    </TableRow>
                    {openRow === review.review_id && (
                      <TableRow>
                        <TableCell colSpan={7}>
                          <div className="grid gap-4 xl:grid-cols-2">
                            <DagPanel structure={review.dag_structure_json} />
                            <ResolveForm
                              review={review}
                              onClose={() => setOpenRow(null)}
                              autoAssessGuard={autoAssessGuard}
                            />
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
```

- [ ] **Step 7: Rewrite the page tests**

Replace the full contents of `frontend/src/pages/ExpertReviews.test.tsx` with:

```tsx
/**
 * ExpertReviews Page Tests (R6-F2 Phase B4 + lane 1)
 * ===================================================
 *
 * The page renders ONLY the live pending queue (no hardcoded SAMPLE_ rows) with
 * honest loading / error / empty states, resolves a review, follows the global
 * brand filter, shows a summary error instead of dropping the counts, opens a
 * linked review from `?review=`, auto-generates a missing assessment when a row
 * is expanded, and prefetches assessments one row at a time.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import ExpertReviews from './ExpertReviews';
import type {
  AgentAssessment,
  ExpertReviewDetailResponse,
  PendingReviewsResponse,
} from '@/types/expert-review';

vi.mock('@/hooks/api/use-expert-review', () => ({
  usePendingReviews: vi.fn(),
  useReviewSummary: vi.fn(),
  useResolveReview: vi.fn(),
  useReviewAssessment: vi.fn(),
  useExpertReview: vi.fn(),
}));

const mockSetBrand = vi.fn();
const filtersState = { brand: 'All' as string };
vi.mock('@/hooks/use-e2i-filters', () => ({
  useE2IFilters: () => ({ filters: { brand: filtersState.brand }, setBrand: mockSetBrand }),
}));

vi.mock('@/api/expert-review', () => ({
  generateReviewAssessment: vi.fn(),
}));

// The DAG renderer is D3-heavy; the page test only asserts it is MOUNTED with
// the right graph (its own rendering is covered by causal.test.tsx).
vi.mock('@/components/visualizations/causal/CausalDAG', () => {
  const FakeDag = ({ nodes, edges }: { nodes: unknown[]; edges: unknown[] }) => (
    <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
  );
  return { CausalDAG: FakeDag, default: FakeDag };
});

import {
  useExpertReview,
  usePendingReviews,
  useReviewAssessment,
  useReviewSummary,
  useResolveReview,
} from '@/hooks/api/use-expert-review';
import { generateReviewAssessment } from '@/api/expert-review';

function createWrapper(initialPath = '/expert-reviews') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialPath]}>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

// Relies on REAL timers: Node orders the form's coerced zero-delay timer before this
// 10 ms wait. Do not add vi.useFakeTimers to this file.
/** Let a form's deferred auto-assessment timer fire so a "still N calls" assertion is not vacuous. */
const flushTimers = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 10)));

const mockPending: PendingReviewsResponse = {
  reviews: [
    {
      review_id: 'rev-1',
      review_type: 'dag_approval',
      dag_version_hash: 'deadbeefcafebabe0123',
      brand: 'Remibrutinib',
      treatment_variable: 'email_frequency',
      outcome_variable: 'trx',
      analysis_context: 'confidence=0.60',
      created_at: '2026-06-01T00:00:00Z',
      days_pending: 5,
    },
  ],
  total: 1,
};

function mockResolveReturn(overrides = {}) {
  return { mutate: vi.fn(), isPending: false, isError: false, error: null, ...overrides };
}

function mockAssessmentReturn(overrides = {}) {
  return { mutate: vi.fn(), isPending: false, isError: false, error: null, data: undefined, ...overrides };
}

function mockQueue(response: PendingReviewsResponse | undefined, extra = {}) {
  vi.mocked(usePendingReviews).mockReturnValue({
    data: response,
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    ...extra,
  } as never);
}

beforeEach(() => {
  vi.clearAllMocks();
  filtersState.brand = 'All';
  vi.mocked(useReviewSummary).mockReturnValue({ data: undefined, isError: false } as never);
  vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn() as never);
  vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn() as never);
  vi.mocked(useExpertReview).mockReturnValue({ data: undefined, isLoading: false, isError: false } as never);
});

describe('ExpertReviews page', () => {
  it('shows a loading state while fetching', () => {
    mockQueue(undefined, { isLoading: true, isFetching: true });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Expert Reviews')).toBeInTheDocument();
  });

  it('shows an honest empty state (no SAMPLE rows) when the queue is empty', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('No pending reviews')).toBeInTheDocument();
  });

  it('shows an error banner on failure', () => {
    mockQueue(undefined, { isError: true, error: { message: 'boom' } });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Failed to load pending reviews')).toBeInTheDocument();
  });

  it('renders the live pending queue and resolves a review', async () => {
    const mutate = vi.fn();
    vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn({ mutate }) as never);
    mockQueue(mockPending);

    render(<ExpertReviews />, { wrapper: createWrapper() });

    expect(screen.getByText('email_frequency')).toBeInTheDocument();
    expect(screen.getByText('Remibrutinib')).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^review$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /approve/i })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /approve/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const [vars] = mutate.mock.calls[0];
    expect(vars.reviewId).toBe('rev-1');
    expect(vars.body.approval_status).toBe('approved');
  });
});

describe('ExpertReviews brand filter and summary (lane 1)', () => {
  it('passes the global brand to BOTH the queue and the summary; All sends no brand', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(vi.mocked(usePendingReviews)).toHaveBeenLastCalledWith(undefined);
    expect(vi.mocked(useReviewSummary)).toHaveBeenLastCalledWith(undefined);
    expect(screen.getByText(/including reviews with no brand/i)).toBeInTheDocument();
  });

  it('scopes to the selected brand', () => {
    filtersState.brand = 'Kisqali';
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(vi.mocked(usePendingReviews)).toHaveBeenLastCalledWith({ brand: 'Kisqali' });
    expect(vi.mocked(useReviewSummary)).toHaveBeenLastCalledWith({ brand: 'Kisqali' });
    expect(screen.getByText(/brand: Kisqali/)).toBeInTheDocument();
  });

  it('shows a banner instead of silently dropping the counts when the summary fails', () => {
    vi.mocked(useReviewSummary).mockReturnValue({
      data: undefined,
      isError: true,
      error: { message: 'Expert-review store unavailable. Retry shortly.' },
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Review counts unavailable')).toBeInTheDocument();
    expect(screen.queryByText(/Pending:/)).not.toBeInTheDocument();
  });

  it('renders the counts when the summary succeeds (positive control for the two banner cases)', () => {
    vi.mocked(useReviewSummary).mockReturnValue({
      data: { pending: 3, approved: 1, rejected: 0, expired: 0, expiring_soon: 1 },
      isError: false,
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Pending: 3')).toBeInTheDocument();
    expect(screen.getByText('of which expiring soon: 1')).toBeInTheDocument();
    expect(screen.queryByText('Review counts unavailable')).not.toBeInTheDocument();
  });

  it('replaces STALE counts with the banner when a refetch fails (TanStack keeps data on error)', () => {
    vi.mocked(useReviewSummary).mockReturnValue({
      data: { pending: 3, approved: 1, rejected: 0, expired: 0, expiring_soon: 1 },
      isError: true,
      error: { message: 'Expert-review store unavailable. Retry shortly.' },
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Review counts unavailable')).toBeInTheDocument();
    expect(screen.getByText('Expert-review store unavailable. Retry shortly.')).toBeInTheDocument();
    expect(screen.queryByText(/Pending:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/expiring soon/)).not.toBeInTheDocument();
  });
});

const STRUCTURE = {
  nodes: ['t', 'y', 'c'],
  edges: [
    ['t', 'y'],
    ['c', 't'],
    ['c', 'y'],
  ],
  treatment_nodes: ['t'],
  outcome_nodes: ['y'],
};

// Typed so the literal verdicts stay `AssessmentVerdict` when the fixture is
// handed to the typed queue mock (untyped, they widen to `string`).
const ASSESSMENT: AgentAssessment = {
  items: [
    { id: 'conf_complete', question: 'Are all known confounders included?', verdict: 'supports', rationale: 'confounder refuters passed' },
    { id: 'positivity', question: 'Is there sufficient overlap in treatment groups?', verdict: 'concern', rationale: 'data_subset failed' },
  ],
  is_fallback: true,
  evidence: { refutation_tests: 2, has_dag_structure: true },
};

function renderWithRow(row: Record<string, unknown>, path?: string) {
  mockQueue({ reviews: [{ ...mockPending.reviews[0], ...row }], total: 1 });
  return render(<ExpertReviews />, { wrapper: createWrapper(path) });
}

describe('ExpertReviews DAG snapshot (mig 097)', () => {
  it('renders the stored DAG in the expanded row', async () => {
    renderWithRow({ dag_structure_json: STRUCTURE });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    const dag = await screen.findByTestId('causal-dag');
    expect(dag).toHaveAttribute('data-nodes', '3');
    expect(dag).toHaveAttribute('data-edges', '3');
  });

  it('shows an honest fallback when the structure was never captured', async () => {
    renderWithRow({ dag_structure_json: null });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect(await screen.findByText(/DAG structure not captured for this review/i)).toBeInTheDocument();
    expect(screen.queryByTestId('causal-dag')).not.toBeInTheDocument();
  });
});

describe('ExpertReviews agent assessment (advisory)', () => {
  it('generates the assessment once when a row without a cache is expanded, then regenerates on click', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    renderWithRow({ dag_structure_json: STRUCTURE });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^review$/i }));

    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    // `auto: true` marks the once-per-review automatic request: only ITS failure releases the guard.
    expect(mutate.mock.calls[0][0]).toEqual({ reviewId: 'rev-1', auto: true });

    await user.click(await screen.findByRole('button', { name: /agent assessment/i }));
    expect(mutate).toHaveBeenCalledTimes(2);
  });

  it('does not auto-generate when a cached assessment exists', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    renderWithRow({ dag_structure_json: STRUCTURE, agent_assessment_json: ASSESSMENT });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect(await screen.findByText('supports')).toBeInTheDocument();
    await flushTimers();
    expect(mutate).not.toHaveBeenCalled();
  });

  it('generates ONCE when the linked card and the queue row show the same pending review', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    // A ReviewRecord always carries approval_status (GET /expert-reviews/{id});
    // the card only mounts a form for a PENDING record.
    vi.mocked(useExpertReview).mockReturnValue({
      data: {
        review: { ...mockPending.reviews[0], approval_status: 'pending', dag_structure_json: STRUCTURE },
        history: [],
      },
      isLoading: false,
      isError: false,
    } as never);
    mockQueue({ reviews: [{ ...mockPending.reviews[0], dag_structure_json: STRUCTURE }], total: 1 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-1') });
    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect((await screen.findAllByRole('button', { name: /approve/i })).length).toBe(2);
    await flushTimers();
    expect(mutate).toHaveBeenCalledTimes(1);
    // Two forms for one review must not share element ids (labels would target the other form).
    const ids = Array.from(document.querySelectorAll('[id]')).map((el) => el.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('renders cached verdict chips beside the checklist, labeled advisory, never pre-checked', async () => {
    renderWithRow({ dag_structure_json: STRUCTURE, agent_assessment_json: ASSESSMENT });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect(await screen.findByText('supports')).toBeInTheDocument();
    expect(screen.getByText('concern')).toBeInTheDocument();
    expect(screen.getAllByText(/advisory/i).length).toBeGreaterThan(0);
    screen.getAllByRole('checkbox').forEach((cb) => expect(cb).not.toBeChecked());
  });

  it('prepares the missing assessments one row at a time', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    vi.mocked(generateReviewAssessment).mockResolvedValue({
      review_id: 'x', assessment: ASSESSMENT, cached: false, persisted: true,
    } as never);
    mockQueue({
      reviews: [
        { ...mockPending.reviews[0], review_id: 'rev-1' },
        { ...mockPending.reviews[0], review_id: 'rev-2', agent_assessment_json: ASSESSMENT },
        { ...mockPending.reviews[0], review_id: 'rev-3' },
      ],
      total: 3,
    });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    const button = screen.getByRole('button', { name: /prepare assessments/i });
    expect(button).toHaveTextContent('2 missing');
    await userEvent.setup().click(button);
    await waitFor(() => expect(generateReviewAssessment).toHaveBeenCalledTimes(2));
    expect(vi.mocked(generateReviewAssessment).mock.calls.map((c) => c[0])).toEqual(['rev-1', 'rev-3']);
    // The bulk run marked rev-1 in the shared guard: expanding it must not start a second generation.
    await userEvent.setup().click(screen.getAllByRole('button', { name: /^review$/i })[0]);
    await screen.findByRole('button', { name: /approve/i });
    await flushTimers();
    expect(mutate).not.toHaveBeenCalled();
  });

  it('skips a review whose generation an expanded form already started', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    vi.mocked(generateReviewAssessment).mockResolvedValue({
      review_id: 'x', assessment: ASSESSMENT, cached: false, persisted: true,
    } as never);
    mockQueue({
      reviews: [
        { ...mockPending.reviews[0], review_id: 'rev-1' },
        { ...mockPending.reviews[0], review_id: 'rev-3' },
      ],
      total: 2,
    });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    const user = userEvent.setup();
    await user.click(screen.getAllByRole('button', { name: /^review$/i })[0]); // rev-1's form fires once
    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: /prepare assessments/i }));
    await waitFor(() => expect(generateReviewAssessment).toHaveBeenCalledTimes(1));
    expect(vi.mocked(generateReviewAssessment).mock.calls[0][0]).toBe('rev-3');
  });

  it('stops the prefetch on the first error and says how far it got', async () => {
    vi.mocked(generateReviewAssessment)
      .mockResolvedValueOnce({ review_id: 'rev-1', assessment: ASSESSMENT, cached: false, persisted: true } as never)
      .mockRejectedValueOnce(new Error('LM unavailable'));
    mockQueue({
      reviews: [
        { ...mockPending.reviews[0], review_id: 'rev-1' },
        { ...mockPending.reviews[0], review_id: 'rev-2' },
        { ...mockPending.reviews[0], review_id: 'rev-3' },
      ],
      total: 3,
    });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    await userEvent.setup().click(screen.getByRole('button', { name: /prepare assessments/i }));
    expect(await screen.findByText('Stopped after 1 of 3')).toBeInTheDocument();
    expect(screen.getByText('LM unavailable')).toBeInTheDocument();
    expect(generateReviewAssessment).toHaveBeenCalledTimes(2);
  });

  it('lets a later Prepare retry a review whose earlier attempt failed', async () => {
    vi.mocked(generateReviewAssessment)
      .mockRejectedValueOnce(new Error('LM unavailable'))
      .mockResolvedValue({ review_id: 'rev-1', assessment: ASSESSMENT, cached: false, persisted: true } as never);
    mockQueue({ reviews: [{ ...mockPending.reviews[0], review_id: 'rev-1' }], total: 1 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /prepare assessments/i }));
    expect(await screen.findByText('LM unavailable')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /prepare assessments/i }));
    await waitFor(() => expect(generateReviewAssessment).toHaveBeenCalledTimes(2));
  });
});

const DETAIL: ExpertReviewDetailResponse = {
  review: {
    review_id: 'rev-rejected',
    review_type: 'dag_approval',
    dag_version_hash: 'deadbeefcafebabe0123',
    brand: null,
    treatment_variable: 'treatment_arm',
    outcome_variable: 'persistent_180d',
    approval_status: 'rejected',
    reviewer_name: 'Dr. No',
    concerns_raised: ['collider'],
    created_at: '2026-07-13T10:00:00Z',
    dag_structure_json: STRUCTURE,
  },
  history: [
    { review_id: 'rev-rejected', approval_status: 'rejected', created_at: '2026-07-13T10:00:00Z', reviewer_name: 'Dr. No' },
    { review_id: 'rev-older', approval_status: 'pending', created_at: '2026-07-01T10:00:00Z' },
  ],
};

describe('ExpertReviews linked review (lane 1)', () => {
  it('renders nothing extra without the review param', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.queryByTestId('linked-review')).not.toBeInTheDocument();
    // The hook lives in LinkedReviewCard, which is not mounted without the param
    // (pre-execution review 2026-09-08, codex MED: the old assertion could not pass).
    expect(vi.mocked(useExpertReview)).not.toHaveBeenCalled();
  });

  it('shows a resolved linked review with its decision and same-structure history', () => {
    vi.mocked(useExpertReview).mockReturnValue({ data: DETAIL, isLoading: false, isError: false } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-rejected') });
    expect(vi.mocked(useExpertReview)).toHaveBeenLastCalledWith('rev-rejected');
    const card = screen.getByTestId('linked-review');
    expect(card).toHaveTextContent('rejected');
    expect(card).toHaveTextContent('Dr. No');
    expect(card).toHaveTextContent('collider');
    expect(card).toHaveTextContent('This review is resolved');
    // Status-specific copy matching the gate's precedence (check_approval): a newer
    // pending row reopens a REJECTED structure; it does not displace an approval.
    expect(card).toHaveTextContent(
      'The rejection holds until a newer pending review of the same structure reopens it.'
    );
    expect(card).not.toHaveTextContent('Its approval applies');
    expect(screen.getAllByTestId('causal-dag').length).toBe(1);
    expect(card).toHaveTextContent('rev-older');
    // The backend history INCLUDES the linked review itself: it is marked exactly
    // once, on its own row, and never on the sibling rows (dispatcher deviation 1).
    expect(screen.getAllByText('(this review)')).toHaveLength(1);
    const current = card.querySelectorAll('[data-current="true"]');
    expect(current).toHaveLength(1);
    expect(current[0]).toHaveTextContent('rev-rejected');
    expect(current[0]).toHaveTextContent('(this review)');
    expect(current[0]).not.toHaveTextContent('rev-older');
  });

  it('shows an approved linked review with the approval-precedence copy', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: {
        ...DETAIL,
        review: { ...DETAIL.review, review_id: 'rev-ok', approval_status: 'approved', valid_until: '2027-01-01T00:00:00Z' },
        history: [],
      },
      isLoading: false,
      isError: false,
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-ok') });
    const card = screen.getByTestId('linked-review');
    expect(card).toHaveTextContent('This review is resolved');
    expect(card).toHaveTextContent(
      'Its approval applies until it expires or a newer review rejects the structure.'
    );
    expect(card).not.toHaveTextContent('The rejection holds');
    expect(card).toHaveTextContent('2027-01-01');
  });

  // Provenance rule (codex whole-diff HIGH F1): render only RECORDED provenance.
  // `reviewer_id` holds the REQUESTER (the originating query id the gate wrote)
  // and `created_at` is not a decision time; both are "not recorded" on the card.
  function renderLinked(review: Record<string, unknown>, history: Record<string, unknown>[] = []) {
    vi.mocked(useExpertReview).mockReturnValue({
      data: { review: { ...DETAIL.review, ...review }, history },
      isLoading: false,
      isError: false,
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-rejected') });
    const card = screen.getByTestId('linked-review');
    const facts = within(card.querySelector('dl') as HTMLElement);
    const cell = (label: string) => facts.getByText(label).nextElementSibling as HTMLElement;
    return { card, cell };
  }

  it('says "not recorded" instead of the requester id and the creation date on a row with no recorded resolver', () => {
    const { card, cell } = renderLinked({
      reviewer_id: 'q-7d3f9a2c-1b4e-4c8a-9f0e-2a6b8c4d1e3f',
      reviewer_name: null,
      reviewer_email: null,
      resolved_at: null,
      approved_at: null,
      created_at: '2026-07-13T10:00:00Z',
    });
    expect(cell('Reviewer')).toHaveTextContent('not recorded');
    expect(cell('Decided')).toHaveTextContent('not recorded');
    expect(cell('Decided')).not.toHaveTextContent('2026-07-13');
    expect(card).not.toHaveTextContent('q-7d3f9a2c');
    // The honestly labelled creation chip stays.
    expect(card).toHaveTextContent('created 2026-07-13');
    expect(cell('Comments')).toHaveTextContent('—');
  });

  it('renders the recorded reviewer and decision time when they exist (positive control)', () => {
    const { cell } = renderLinked({
      reviewer_name: 'Dr. No',
      reviewer_email: 'no@example.com',
      resolved_at: '2026-09-09T10:00:00Z',
      approved_at: null,
      created_at: '2026-07-13T10:00:00Z',
    });
    expect(cell('Reviewer')).toHaveTextContent('Dr. No');
    expect(cell('Decided')).toHaveTextContent('2026-09-09');
    expect(cell('Decided')).not.toHaveTextContent('2026-07-13');
  });

  it('falls back to the recorded email when no name was recorded, in the card and in the history', () => {
    const { card, cell } = renderLinked(
      { reviewer_name: null, reviewer_email: 'no@example.com' },
      [
        { review_id: 'rev-rejected', approval_status: 'rejected', created_at: '2026-07-13T10:00:00Z', reviewer_email: 'no@example.com' },
        { review_id: 'rev-older', approval_status: 'pending', created_at: '2026-07-01T10:00:00Z', reviewer_id: 'q-older-query' },
      ]
    );
    expect(cell('Reviewer')).toHaveTextContent('no@example.com');
    const rows = card.querySelectorAll('tbody tr');
    expect(rows[0]).toHaveTextContent('no@example.com');
    expect(rows[1]).toHaveTextContent('—');
    expect(card).not.toHaveTextContent('q-older-query');
  });

  it("shows the reviewer's comment note on a resolved review (codex F2)", () => {
    const { cell } = renderLinked({ comments_json: { note: 'DAG omits the payer confounder' } });
    expect(cell('Comments')).toHaveTextContent('DAG omits the payer confounder');
  });

  it('shows a structured comments object faithfully, never a paraphrase', () => {
    const { cell } = renderLinked({ comments_json: { reason: 'collider', severity: 2 } });
    expect(cell('Comments')).toHaveTextContent('{"reason":"collider","severity":2}');
  });

  it('resolves a pending linked review in place', async () => {
    const mutate = vi.fn();
    vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn({ mutate }) as never);
    vi.mocked(useExpertReview).mockReturnValue({
      data: { ...DETAIL, review: { ...DETAIL.review, review_id: 'rev-p', approval_status: 'pending' }, history: [] },
      isLoading: false,
      isError: false,
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-p') });
    await userEvent.setup().click(screen.getByRole('button', { name: /reject/i }));
    expect(mutate.mock.calls[0][0].reviewId).toBe('rev-p');
    expect(mutate.mock.calls[0][0].body.approval_status).toBe('rejected');
  });

  it('says so when the linked review no longer exists (404), and keeps the queue usable', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { status: 404, message: 'Review nope was not found.' },
    } as never);
    mockQueue(mockPending);
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=nope') });
    expect(screen.getByText('This review no longer exists')).toBeInTheDocument();
    expect(screen.getByText('Review nope was not found.')).toBeInTheDocument();
    expect(screen.queryByText('Failed to load the linked review')).not.toBeInTheDocument();
    expect(screen.getByText('email_frequency')).toBeInTheDocument();
  });

  it('shows the generic failure title with the backend message when the store is unavailable (503)', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { status: 503, message: 'Expert-review store unavailable. Retry shortly.' },
    } as never);
    mockQueue(mockPending);
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-1') });
    expect(screen.getByText('Failed to load the linked review')).toBeInTheDocument();
    expect(screen.getByText('Expert-review store unavailable. Retry shortly.')).toBeInTheDocument();
    expect(screen.queryByText('This review no longer exists')).not.toBeInTheDocument();
    expect(screen.getByText('email_frequency')).toBeInTheDocument();
  });

  it('falls back to the message text only when the error carries no status', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { message: 'Review nope was not found.' },
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=nope') });
    expect(screen.getByText('This review no longer exists')).toBeInTheDocument();
  });

  it('ignores a blank review param', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=%20%20') });
    expect(screen.queryByTestId('linked-review')).not.toBeInTheDocument();
    expect(vi.mocked(useExpertReview)).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 7b: Component tests with real hooks and deferred promises**

Create `frontend/src/components/expert-review/ResolveForm.test.tsx` (real `useReviewAssessment`/`useResolveReview` under `<StrictMode>`; only `@/api/expert-review` is mocked):

```tsx
/**
 * ResolveForm tests — REAL TanStack hooks, only the API module mocked.
 *
 * Pins the once-per-review-id auto-assessment under StrictMode (double-invoked
 * effects) INCLUDING delivery of the result to the form, across an
 * unmount/remount, the guard RELEASE on an automatic failure (the hook-level
 * onError is run by the Mutation itself, so it fires even after the form has
 * unmounted — a collapsed queue row — unlike per-call mutate callbacks), and
 * guard OWNERSHIP: a failed MANUAL request never releases an id another form's
 * automatic request still holds.
 */
import { StrictMode } from 'react';
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ResolveForm } from './ResolveForm';
import type { ResolveFormProps } from './ResolveForm';
import type { AgentAssessment, AgentAssessmentResponse, PendingReviewItem } from '@/types/expert-review';

vi.mock('@/api/expert-review', () => ({
  generateReviewAssessment: vi.fn(),
  resolveReview: vi.fn(),
  getExpertReview: vi.fn(),
  getPendingReviews: vi.fn(),
  getReviewSummary: vi.fn(),
}));
import { generateReviewAssessment } from '@/api/expert-review';

const api = vi.mocked(generateReviewAssessment);

const REVIEW: PendingReviewItem = { review_id: 'rev-1', brand: 'Kisqali', treatment_variable: 't', outcome_variable: 'y' };
const ASSESSMENT: AgentAssessment = { items: [], is_fallback: true };
const RESPONSE: AgentAssessmentResponse = { review_id: 'rev-1', assessment: ASSESSMENT, cached: false, persisted: true };

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function newGuard() {
  return { current: new Set<string>() };
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
}

// Relies on REAL timers: Node orders the form's coerced zero-delay timer before this
// 10 ms wait. Do not add vi.useFakeTimers to this file.
/** Let any deferred auto-assessment timer fire so a "still N calls" assertion is not vacuous. */
const flushTimers = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 10)));

function renderForm(props: Partial<ResolveFormProps> = {}, queryClient = newClient()) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <StrictMode>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </StrictMode>
  );
  return render(<ResolveForm review={REVIEW} onClose={() => undefined} {...props} />, { wrapper });
}

beforeEach(() => {
  // mockReset (not clear): a test that aborts early must not leak its `Once` queue.
  api.mockReset();
});

describe('ResolveForm auto-assessment (real hooks, StrictMode)', () => {
  it('generates exactly once on mount with a shared guard, and the result reaches the form', async () => {
    api.mockResolvedValue(RESPONSE);
    const guard = newGuard();
    renderForm({ autoAssessGuard: guard });
    // "Regenerate" + enabled = the mutation result was delivered (not orphaned by StrictMode's resubscribe).
    const button = await screen.findByRole('button', { name: /regenerate agent assessment/i });
    expect(button).toBeEnabled();
    expect(api).toHaveBeenCalledTimes(1);
    expect(api.mock.calls[0][0]).toBe('rev-1');
    expect(api.mock.calls[0][1]).toBeFalsy();
    expect(guard.current.has('rev-1')).toBe(true);
  });

  it('generates exactly once with its own local guard (no page guard)', async () => {
    api.mockResolvedValue(RESPONSE);
    renderForm();
    await screen.findByRole('button', { name: /regenerate agent assessment/i });
    expect(api).toHaveBeenCalledTimes(1);
  });

  it('does not generate again after an unmount and remount with the same guard (success path)', async () => {
    api.mockResolvedValue(RESPONSE);
    const guard = newGuard();
    const { unmount } = renderForm({ autoAssessGuard: guard });
    await screen.findByRole('button', { name: /regenerate agent assessment/i });
    unmount();
    renderForm({ autoAssessGuard: guard });
    // The remounted form has no cache yet (the page refetches it) and offers a manual Generate.
    await screen.findByRole('button', { name: /^generate agent assessment$/i });
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(1);
  });

  it('releases the guard when the AUTOMATIC generation fails after the form unmounted, so a remount retries once', async () => {
    const first = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(first.promise).mockResolvedValueOnce(RESPONSE);
    const guard = newGuard();
    const { unmount } = renderForm({ autoAssessGuard: guard });
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(guard.current.has('rev-1')).toBe(true); // positive control for the release below
    unmount(); // the row collapsed while the request was in flight
    first.reject(new Error('LM unavailable'));
    await waitFor(() => expect(guard.current.has('rev-1')).toBe(false));
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(1); // no retry loop: the effect deps do not change on error
    renderForm({ autoAssessGuard: guard });
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(api.mock.calls[1][0]).toBe('rev-1');
  });

  it('a failed MANUAL request from a second form never releases the id the automatic request still holds', async () => {
    const requestA = deferred<AgentAssessmentResponse>();
    const requestB = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(requestA.promise).mockReturnValueOnce(requestB.promise);
    const guard = newGuard();
    const queryClient = newClient();
    const formA = renderForm({ autoAssessGuard: guard }, queryClient); // automatic → request A in flight
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(guard.current.has('rev-1')).toBe(true);
    const formB = renderForm({ autoAssessGuard: guard }, queryClient); // linked card + queue row: same review
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(1); // B's automatic generation is skipped by the shared guard
    await userEvent
      .setup()
      .click(within(formB.container).getByRole('button', { name: /^generate agent assessment$/i }));
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2)); // B's MANUAL request B
    requestB.reject(new Error('LM unavailable'));
    await within(formB.container).findByText('Failed to generate agent assessment');
    expect(guard.current.has('rev-1')).toBe(true); // B's failure is not the automatic request's failure
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(2); // no request C
    requestA.resolve(RESPONSE);
    await within(formA.container).findByRole('button', { name: /regenerate agent assessment/i });
    expect(guard.current.has('rev-1')).toBe(true);
  });

  it('does not auto-generate for a cached assessment; Regenerate forces a fresh one', async () => {
    api.mockResolvedValue(RESPONSE);
    renderForm({ review: { ...REVIEW, agent_assessment_json: ASSESSMENT } });
    await flushTimers();
    expect(api).not.toHaveBeenCalled();
    await userEvent.setup().click(screen.getByRole('button', { name: /regenerate agent assessment/i }));
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(api).toHaveBeenCalledWith('rev-1', true);
  });
});
```

Create `frontend/src/components/expert-review/PrepareAssessmentsButton.test.tsx` (real `QueryClient` with `invalidateQueries` spied; deferred promises observe the walk mid-flight):

```tsx
/**
 * PrepareAssessmentsButton tests — real QueryClient, only the API module mocked,
 * DEFERRED promises so the strictly sequential walk, the first-error stop, the
 * guard release and the Stop button are observed mid-flight, not inferred.
 */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PrepareAssessmentsButton } from './PrepareAssessmentsButton';
import { queryKeys } from '@/lib/query-client';
import type { AgentAssessmentResponse, PendingReviewItem } from '@/types/expert-review';

vi.mock('@/api/expert-review', () => ({
  generateReviewAssessment: vi.fn(),
  resolveReview: vi.fn(),
  getExpertReview: vi.fn(),
  getPendingReviews: vi.fn(),
  getReviewSummary: vi.fn(),
}));
import { generateReviewAssessment } from '@/api/expert-review';

const api = vi.mocked(generateReviewAssessment);

const ROWS: PendingReviewItem[] = [{ review_id: 'rev-1' }, { review_id: 'rev-2' }];
const THREE_ROWS: PendingReviewItem[] = [...ROWS, { review_id: 'rev-3' }];
const PENDING_PREFIX = [...queryKeys.expertReviews.all(), 'pending'];
// The linked card reads the detail query; the form's hook invalidates it too (F4).
const DETAIL_PREFIX = [...queryKeys.expertReviews.all(), 'detail'];

function response(id: string): AgentAssessmentResponse {
  return { review_id: id, assessment: { items: [], is_fallback: true }, cached: false, persisted: true };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

// Deferred promises and `waitFor` rely on REAL timers here; do not add vi.useFakeTimers to this file.
function renderButton(rows: PendingReviewItem[] = ROWS) {
  const guard = { current: new Set<string>() };
  const queryClient = new QueryClient();
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  render(<PrepareAssessmentsButton reviews={rows} autoAssessGuard={guard} />, { wrapper });
  return { guard, invalidate, button: screen.getByRole('button', { name: /prepare assessments/i }) };
}

beforeEach(() => {
  // mockReset (not clear): a test that aborts early must not leak its `Once` queue
  // of deferred promises into the next test (measured: a never-resolving leftover
  // left the button stuck at "Preparing 0 / 2…").
  api.mockReset();
});

describe('PrepareAssessmentsButton', () => {
  it('walks the missing rows strictly one at a time, then invalidates the pending queue once', async () => {
    const first = deferred<AgentAssessmentResponse>();
    const second = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { guard, invalidate, button } = renderButton();
    // The visible label IS the accessible name (no aria-label masking the count / progress).
    expect(button).toHaveAccessibleName('Prepare assessments (2 missing)');

    await userEvent.setup().click(button);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(api).toHaveBeenLastCalledWith('rev-1');
    expect(guard.current.has('rev-1')).toBe(true); // positive control for the release assertion below
    expect(button).toHaveTextContent('Preparing 0 / 2');
    expect(button).toBeDisabled();

    first.resolve(response('rev-1'));
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(api).toHaveBeenLastCalledWith('rev-2');
    expect(button).toHaveTextContent('Preparing 1 / 2');
    expect(invalidate).not.toHaveBeenCalled();

    second.resolve(response('rev-2'));
    await waitFor(() => expect(button).toHaveTextContent('Prepare assessments (2 missing)'));
    expect(button).toBeEnabled();
    // Once per prefix, after the walk: the queue AND the linked card's detail
    // query (F4 -- the form's hook invalidates both; a count alone is vacuous).
    expect(invalidate).toHaveBeenCalledTimes(2);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: PENDING_PREFIX });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: DETAIL_PREFIX });
  });

  it('stops on the first error, says how far it got, and releases the failed id from the guard', async () => {
    api.mockRejectedValueOnce(new Error('LM unavailable'));
    const { guard, invalidate, button } = renderButton();
    await userEvent.setup().click(button);
    expect(await screen.findByText('Stopped after 0 of 2')).toBeInTheDocument();
    expect(screen.getByText('LM unavailable')).toBeInTheDocument();
    expect(api).toHaveBeenCalledTimes(1);
    expect(api).not.toHaveBeenCalledWith('rev-2');
    expect(guard.current.has('rev-1')).toBe(false);
    expect(guard.current.has('rev-2')).toBe(false);
    await waitFor(() => expect(invalidate).toHaveBeenCalledTimes(2));
    expect(invalidate).toHaveBeenCalledWith({ queryKey: PENDING_PREFIX });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: DETAIL_PREFIX });
    expect(button).toBeEnabled();
  });

  it('stops on persisted:false (HTTP 200, the store rejected the write), releases that id so it can be retried, and never requests the next row', async () => {
    // F3: the endpoint returns the (valid) assessment with persisted:false when the
    // cache write failed. Ignoring it left the id guarded and the row uncached, so
    // every later Prepare skipped it forever.
    api
      .mockResolvedValueOnce(response('rev-1'))
      .mockResolvedValueOnce({ ...response('rev-2'), persisted: false })
      .mockResolvedValueOnce(response('rev-3'));
    const { guard, invalidate, button } = renderButton(THREE_ROWS);
    await userEvent.setup().click(button);
    expect(await screen.findByText('Stopped after 1 of 3')).toBeInTheDocument();
    expect(
      screen.getByText(
        "Assessment for review rev-2 was generated but not saved (the store rejected the write). Retry from the row's Generate button or run Prepare again."
      )
    ).toBeInTheDocument();
    expect(api).toHaveBeenCalledTimes(2);
    expect(api).not.toHaveBeenCalledWith('rev-3');
    expect(guard.current.has('rev-1')).toBe(true); // the saved one stays guarded (positive control)
    expect(guard.current.has('rev-2')).toBe(false);
    expect(guard.current.has('rev-3')).toBe(false);
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: PENDING_PREFIX }));
    expect(invalidate).toHaveBeenCalledWith({ queryKey: DETAIL_PREFIX });
    expect(button).toBeEnabled();
  });

  it('completes 3 of 3 when every write persisted (control for the persisted:false stop)', async () => {
    api.mockImplementation(async (id: string) => response(id));
    const { guard, button } = renderButton(THREE_ROWS);
    await userEvent.setup().click(button);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(button).toBeEnabled());
    expect(screen.queryByText(/Stopped after/)).not.toBeInTheDocument();
    expect(['rev-1', 'rev-2', 'rev-3'].every((id) => guard.current.has(id))).toBe(true);
  });

  it('Stop ends the walk after the in-flight request; the next row is never requested', async () => {
    const first = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(first.promise).mockResolvedValue(response('rev-2'));
    const { button } = renderButton();
    const user = userEvent.setup();
    await user.click(button);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: /stop/i }));
    first.resolve(response('rev-1'));
    await waitFor(() => expect(button).toBeEnabled());
    expect(api).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: /stop/i })).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 8: Run, typecheck, lint, commit**

```bash
cd frontend && npx vitest run src/pages/ExpertReviews.test.tsx src/components/expert-review src/hooks/api/use-expert-review.test.ts && npm run typecheck && npx eslint src/pages/ExpertReviews.tsx src/components/expert-review && cd ..
git add frontend/src/pages/ExpertReviews.tsx frontend/src/pages/ExpertReviews.test.tsx frontend/src/components/expert-review
git commit -m "feat(frontend): expert-review queue -- linked review card, global brand filter, honest summary, assessment prefetch

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

If `npm run typecheck` complains that `filters.brand` is not assignable to `string`, the cast in the page (`filters.brand as string`) is the intended narrowing of the `E2IFilters['brand']` union.

---

### Task 11: Lineage document — sections, map labels, anchors, pinned commit

**Files:**
- Modify: `docs/lineage/causal_dag_lineage.html`
- Scratch (not committed): `<scratchpad>/lineage_edit.py`, `<scratchpad>/refresh_anchors.py`

Do this task LAST among the code tasks (after Task 12's gates pass) so the anchors resolve against final line numbers. Every replacement below asserts its `old` string occurs exactly once; the script aborts otherwise, so nothing is edited by guesswork.

- [ ] **Step 1: Section edits (scripted, exact-match replacements)**

Save as `<scratchpad>/lineage_edit.py` and run with `python3 <scratchpad>/lineage_edit.py` from the worktree root:

```python
#!/usr/bin/env python3
"""Lane 1: rewrite the lineage page's stale sections to the shipped state."""
from pathlib import Path

DOC = Path("docs/lineage/causal_dag_lineage.html")
s = DOC.read_text(encoding="utf-8")
EDITS = []

def rep(old: str, new: str) -> None:
    EDITS.append((old, new))

# §3.3 — discovery tables now live in public and have a writer (#1974, #1984).
rep(
    '<tr><td><code>ml.discovered_dags</code>, <code>ml.discovered_edges</code></td><td>Schema exists (migration 026). <strong>No writer in <code>src/</code>.</strong></td><td>—</td><td>—</td><td>database/ml/026_causal_discovery_tables.sql</td></tr>',
    '<tr><td><code>public.discovered_dags</code>, <code>discovered_edges</code>, <code>discovery_algorithm_runs</code></td><td>Ensemble edges, per-algorithm runs, gate evaluation and the shipped DAG with per-edge provenance, written atomically by <code>record_discovered_dag</code> whenever discovery ran (issue #1974; the tables moved from <code>ml</code> to <code>public</code> in ml/036). Persistence failures are visible in <code>warnings</code>, never a crash; the response carries <code>discovered_dag_id</code>, which the drill-down shows as the durable discovery record.</td><td><code>query_id</code> = analysis id; <code>dag_version_hash</code></td><td>durable</td><td>src/repositories/discovered_dag.py:344</td></tr>',
)

# §4.2 — the two non-critical tests score real evidence; bootstrap thresholds corrected.
rep(
    '<tr><td>data_subset</td><td>5 subsets of 80 %</td><td>no</td><td>≥ 80 % of subsets contain the original effect</td><td>70–80 %</td><td class="num">0.125</td></tr>',
    '<tr><td>data_subset</td><td>5 subsets of 80 %</td><td>no</td><td>≥ 80 % of the per-subset re-fit effects fall inside the reported CI (real evidence since lane 1; recorded SKIPPED on 96/96 live runs before it)</td><td>70–80 %</td><td class="num">0.125</td></tr>',
)
rep(
    '<tr><td>bootstrap</td><td>50 (20)</td><td>no</td><td>bootstrap CI ≤ 50 % wider than original</td><td>50–75 %</td><td class="num">0.125</td></tr>',
    '<tr><td>bootstrap</td><td>50 (20)</td><td>no</td><td>2.5–97.5 percentile width of the re-fit effects ≤ 1.5 × the reported CI width (lane 1 corrected the thresholds 0.50/0.75, which contradicted their own comment and never scored)</td><td>1.5–1.75 ×</td><td class="num">0.125</td></tr>',
)
rep(
    '(a critical test in WARNING still permits PROCEED, by design)</code></pre>',
    '(a critical test in WARNING still permits PROCEED, by design)</code></pre>\n  <div class="callout finding"><div class="label">Measured</div><p>96 live runs on record (2026-09-08): 49 PROCEED, 47 BLOCK, <b>0 REVIEW</b>. With only the three critical tests scoring, the reachable confidence values without a critical failure are 1.0 and 0.867, both PROCEED; REVIEW needs a sensitivity WARNING <em>and</em> both non-critical tests FAILED (0.65). Lane 1 made the non-critical evidence real, so REVIEW is now reachable for genuinely unstable estimates; on the two live pairs replicated offline both tests PASS and the band is unchanged. Whether a sensitivity WARNING alone should read as REVIEW is an open owner decision, not a code defect.</p></div>',
)

# §4.3 — the gate is consulted on every band; the switch; approval structural.
rep(
    'Nothing reads those fields to halt execution. The caveat names an expert approval only when a real approval row exists; the no-repository bypass used to claim one (fixed, issue #1969).',
    'Since issue #1971 the node also runs a read-only rejection probe on <em>every</em> band before persisting anything: the newest adjudication of the structure wins, a pending row newer than a rejection reopens it, and a rejected structure halts the run (status <code>failed</code>, reviewer and review id in the message) on PROCEED as well. An enforcement switch, <code>CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL</code> (env, default off), additionally withholds a REVIEW-band estimate whose structure holds no active approval. Approval is structural only: it never promotes a borderline estimate. The caveat names an expert approval only when a real approval row exists; the no-repository bypass used to claim one (fixed, issue #1969).',
)
rep(
    '<p>On the live agent path a DAG can be estimated against before any human has approved it. The gate queues and annotates; it does not block. The only true precondition is the SQL function <code>can_use_estimate()</code> in migration 010, which no Python code calls. Whether new structures should hard-block is an open product decision, tracked in issue #1971.</p>',
    '<p>A human REJECTION now halts on every band and is never promoted over (issue #1971; lane 1 evaluates the rule inside the promote statement). Approval is structural and changes no estimate. The enforcement switch is inert on today\'s traffic because the live gate has never produced a REVIEW band (see 4.2, Measured); both <code>can_use_estimate</code> definitions were retired (migration 133). The queue held 39 pending / 1 rejected / 0 approved on 2026-09-08 — every pending row a BLOCK-band structure.</p>',
)
rep(
    'cached in <code>agent_assessment_json</code>. <span class="anchor">src/api/routes/expert_review.py:76</span></li>',
    'cached in <code>agent_assessment_json</code>; <code>GET /expert-reviews/{id}</code> returns one review in any status with its same-structure history — the destination of the drill-down\'s "Open review" link (lane 1). <span class="anchor">src/api/routes/expert_review.py:76</span></li>\n      <li>The causal drill-down shows the structure\'s review state (<code>refutation.expert_review_decision</code>, the review id, the rejection reason) and the durable <code>discovered_dag_id</code>; the queue page follows the global brand filter, shows an honest error when the counts are unavailable, and generates the advisory assessment when a row is expanded or on demand for every row lacking one. <span class="anchor">frontend/src/components/causal/ReviewStatusPanel.tsx:1</span></li>',
)

# §4.4 — the guarded promote.
rep(
    'Evidence first, status second. <span class="anchor">nodes/refutation.py:57</span> <span class="anchor">:769</span></p>',
    'Evidence first, status second. Since lane 1 the status write is the SQL function <code>promote_causal_path_guarded</code> (migration 134): one UPDATE conditioned on the allowed current status <em>and</em> on <code>dag_structure_rejected(hash, brand)</code> being false, so a rejection committed after the node\'s read-only probe can never be promoted over. <span class="anchor">nodes/refutation.py:57</span> <span class="anchor">:769</span> <span class="anchor">database/migrations/134_guarded_causal_path_promote.sql:1</span></p>',
)

# §4.8 — gaps register.
rep(
    '<li><b>No hard block on the live path.</b> Expert-review BLOCKED and PENDING_REVIEW are recorded and now surfaced in the API, not enforced. Whether a new structure should block is an owner decision (issue #1971). Three infrastructure-absent paths still resolve to PROCEED; since issue #1969 none of them can claim an approval to the user.</li>',
    '<li><b>REVIEW is rare by design.</b> The band needs a sensitivity WARNING plus both non-critical tests FAILED; 0 of 96 live runs reached it. The enforcement switch (<code>CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL</code>, default off) therefore has no live traffic to act on until an unstable estimate appears. A human rejection, by contrast, halts every band (issue #1971). Infrastructure-absent paths resolve to <code>unavailable</code>, never to a claimed approval (issue #1969).</li>',
)
rep(
    '<li><b>The discovery tables are not applied on the live database.</b> Migration 026 defines <code>ml.discovered_dags</code> and friends; the live <code>ml</code> schema has zero tables, so no writer can be wired until the migration is applied (issue #1974).</li>\n',
    '',
)

# Map labels (text only; no geometry changes).
rep(
    '<text class="s" x="910" y="263">expert_reviews keyed by dag_version_hash</text><text class="s" x="910" y="275">90-day approval · 14-day renewal warning</text>',
    '<text class="s" x="910" y="263">expert_reviews keyed by dag_version_hash · probe on every band</text><text class="s" x="910" y="275">rejection halts · approval structural · switch default off</text>',
)

for old, new in EDITS:
    n = s.count(old)
    assert n == 1, f"expected exactly one occurrence, found {n}: {old[:80]!r}"
    s = s.replace(old, new)
DOC.write_text(s, encoding="utf-8")
print(f"applied {len(EDITS)} edits")
```

Expected output: `applied 11 edits`. If an assertion fires, the fragment drifted: open the file at that section, adjust `old` to the exact current text, re-run.

- [ ] **Step 2: Add the map's discovery box label**

The API-response box on the map already exists; append a discovery line to it. Find the `<text class="t"` element whose text is `API response` and add, after its last sibling `<text class="s" …>` inside the same `<g>`, a third small line (copy the sibling's `x`, use `y` + 12):

```html
<text class="s" x="X" y="Y+12">+ public.discovered_dags record (discovered_dag_id)</text>
```

Verify visually by opening the file in a browser (`python3 -m http.server` from `docs/lineage/` on a spare port) that the new line sits inside its box.

- [ ] **Step 3: Re-resolve anchors and the pinned commit**

Save as `<scratchpad>/refresh_anchors.py`:

```python
#!/usr/bin/env python3
"""Re-resolve every file:line anchor in the lineage page against the current tree.

For each anchor, take the referenced line's text at the OLD pinned commit and
find the same text in the CURRENT file (the occurrence nearest the old line
number wins). Anchors that cannot be resolved are printed, not guessed.
Usage: python3 refresh_anchors.py <old_commit> <new_commit>
"""
import re, subprocess, sys
from pathlib import Path

DOC = Path("docs/lineage/causal_dag_lineage.html")
OLD, NEW = sys.argv[1], sys.argv[2]
ALIASES = {
    "state.py": "src/agents/causal_impact/state.py",  # 10 state.py files; the page means this one
    "causal.py": "src/api/routes/causal.py",
    "nodes/refutation.py": "src/agents/causal_impact/nodes/refutation.py",
    "refutation.py": "src/agents/causal_impact/nodes/refutation.py",
    "estimation.py": "src/agents/causal_impact/nodes/estimation.py",
    "graph_builder.py": "src/agents/causal_impact/nodes/graph_builder.py",
    "graph.py": "src/agents/causal_impact/graph.py",
    "mlflow_tracker.py": "src/agents/causal_impact/mlflow_tracker.py",
    "memory_hooks.py": "src/agents/causal_impact/memory_hooks.py",
    "dispatcher.py": "src/agents/orchestrator/nodes/dispatcher.py",
    "router.py": "src/agents/orchestrator/nodes/router.py",
    "expert_review_gate.py": "src/causal_engine/expert_review_gate.py",
    "runner.py": "src/causal_engine/discovery/runner.py",
    "gate.py": "src/causal_engine/discovery/gate.py",
    "base.py": "src/causal_engine/discovery/base.py",
}
TRACKED = subprocess.check_output(["git", "ls-files"], text=True).split()

def repo_path(short: str):
    if short in ALIASES:
        return ALIASES[short]
    hits = [p for p in TRACKED if p == short or p.endswith("/" + short)]
    return hits[0] if len(hits) == 1 else None

_cache = {}
def lines(commit: str, path: str):
    key = (commit, path)
    if key not in _cache:
        try:
            _cache[key] = subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True).splitlines()
        except subprocess.CalledProcessError:
            _cache[key] = None
    return _cache[key]

def resolve(path: str, old_line: int):
    old = lines(OLD, path)
    new = lines(NEW, path)
    if not old or not new or old_line > len(old):
        return None
    needle = old[old_line - 1].strip()
    if not needle:
        return None
    cands = [i + 1 for i, t in enumerate(new) if t.strip() == needle]
    if not cands:
        return None
    return min(cands, key=lambda n: abs(n - old_line))

s = DOC.read_text(encoding="utf-8")
unresolved, changed, last_path = [], 0, None
ANCHOR = re.compile(r'(<span class="anchor">)(?:([^<:]*):)?(\d+)(</span>)')  # `*`: ":769" shorthand inherits the previous path
def fix_anchor(m):
    global last_path, changed
    pre, short, line, post = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    if short:
        last_path = repo_path(short)
    path = last_path
    if not path:
        unresolved.append(m.group(0)); return m.group(0)
    new_line = resolve(path, line)
    if new_line is None:
        unresolved.append(f"{path}:{line}"); return m.group(0)
    if new_line != line:
        changed += 1
    return f"{pre}{(short + ':') if short else ''}{new_line}{post}"
s = ANCHOR.sub(fix_anchor, s)

IDX = re.compile(r'(<td>)([A-Za-z0-9_./-]+\.(?:py|tsx?|sql))\:(\d+)(</td>)')
def fix_idx(m):
    global changed
    short, line = m.group(2), int(m.group(3))
    path = repo_path(short)  # the index also uses short names (base.py, graph_builder.py)
    if not path:
        unresolved.append(m.group(0)); return m.group(0)
    new_line = resolve(path, line)
    if new_line is None:
        unresolved.append(f"{path}:{line}"); return m.group(0)
    if new_line != line:
        changed += 1
    return f"{m.group(1)}{short}:{new_line}{m.group(4)}"
s = IDX.sub(fix_idx, s)

s = s.replace(f'at commit <span class="mono">{OLD}</span>', f'at commit <span class="mono">{NEW}</span>')
DOC.write_text(s, encoding="utf-8")
print(f"changed {changed} anchors; unresolved {len(unresolved)}")
for u in unresolved:
    print("  UNRESOLVED", u)
```

Run it against the lane's last code commit:

```bash
NEW=$(git rev-parse --short HEAD)
python3 <scratchpad>/refresh_anchors.py 28dbafb "$NEW"
```

Measured 2026-09-08 on the unchanged tree (28dbafb → f30e9e9df): `changed 42 anchors; unresolved 5`. Three of the page's anchors cannot be resolved by text: two already pointed at BLANK lines at 28dbafb (each present once as an anchor and once as an index row) and one shorthand anchor (`:1387`, the `_consult_review_gate` call) whose line was rewritten by #1985. Fix them by hand with the current line numbers (earlier versions of this script also silently skipped the 18 `:NNN` shorthand anchors, failed on 10 index rows with short names and on `state.py`; all fixed above):

```bash
grep -n '^async def _discover_candidate_questions' src/api/routes/causal.py        # 1679 at f30e9e9df; re-read
grep -n '^export default function ExpertReviews' frontend/src/pages/ExpertReviews.tsx  # after Task 10's rewrite
grep -n 'return await self._consult_review_gate(' src/agents/causal_impact/nodes/refutation.py   # 1086 at f30e9e9df; re-read
sed -i "s#src/api/routes/causal.py:1677#src/api/routes/causal.py:<n1>#g; s#frontend/src/pages/ExpertReviews.tsx:69#frontend/src/pages/ExpertReviews.tsx:<n2>#g; s#<span class=\"anchor\">:1387</span>#<span class=\"anchor\">:<n3></span>#" docs/lineage/causal_dag_lineage.html
```

Anchors that reference files this lane created (`ReviewStatusPanel.tsx:1`, `134_…sql:1`) resolve trivially (line 1 exists in both). Any OTHER unresolved anchor means a lane commit moved text: open both versions (`git show 28dbafb:<path> | sed -n '<line>p'`) and fix the number by hand; do not leave a stale anchor.

- [ ] **Step 4: Add two rows to the code anchor index**

In the `<tbody>` of `<table id="idx-table">`, append (keep the row style of the existing rows):

```html
      <tr><td><span class="pill gov">gov</span></td><td>promote_causal_path_guarded</td><td>database/migrations/134_guarded_causal_path_promote.sql:1</td><td>status write conditioned on the review chronology inside the UPDATE</td></tr>
      <tr><td><span class="pill gov">gov</span></td><td>get_expert_review</td><td>src/api/routes/expert_review.py:1</td><td>one review in any status with its same-structure history (the drill-down's deep link)</td></tr>
      <tr><td><span class="pill gov">gov</span></td><td>_resample_effects</td><td>src/causal_engine/refutation_runner.py:1</td><td>real per-resample evidence for data_subset and bootstrap, deadline-aware, seeded</td></tr>
```

Then replace the three `:1` line numbers with the real ones:

```bash
grep -n 'CREATE OR REPLACE FUNCTION public.promote_causal_path_guarded' database/migrations/134_guarded_causal_path_promote.sql
grep -n 'async def get_expert_review' src/api/routes/expert_review.py
grep -n '^def _resample_effects' src/causal_engine/refutation_runner.py
```

- [ ] **Step 5: Verify and commit**

```bash
python3 - <<'EOF'
import re
s=open('docs/lineage/causal_dag_lineage.html',encoding='utf-8').read()
assert 'ml.discovered_dags' not in s, 'stale ml.discovered_dags mention'
assert s.count('promote_causal_path_guarded') >= 2
assert '28dbafb' not in s, 'pinned commit not updated'
print('lineage checks OK')
EOF
git add docs/lineage/causal_dag_lineage.html
git commit -m "docs(lineage): rewrite the expert-review, promotion and gaps sections to the shipped state; refresh anchors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

---

### Task 12: Quality gates and the codex read-only audit

**Files:** none new (fixes land in the files above)

- [ ] **Step 1: Backend gates on the changed files**

```bash
$PY -m pytest tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_1419.py tests/unit/test_causal_engine/test_refutation_runner_randomized.py tests/unit/test_repositories/test_causal_path_promoter_1352.py tests/unit/test_agents/test_causal_impact/test_refutation_promoter_1352.py tests/unit/test_agents/test_causal_impact/test_refutation_expert_review_enforcement_1971.py tests/unit/test_api/test_expert_review_detail_route.py tests/unit/test_database/test_migration_134_guarded_promote.py tests/unit/test_repositories/test_causal_validation.py tests/unit/test_repositories/test_discovered_dag*.py tests/unit/test_database/test_migration_135_json_objects.py tests/unit/test_causal_engine/test_expert_review_gate.py -q -p no:cacheprovider -m "not slow" 2>&1 | tail -3
$PY -m ruff check src/causal_engine/refutation_runner.py src/causal_engine/expert_review_gate.py src/repositories/causal_path.py src/repositories/causal_validation.py src/repositories/json_utils.py src/repositories/discovered_dag.py src/agents/causal_impact/nodes/refutation.py src/api/routes/expert_review.py src/api/schemas/expert_review.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py tests/unit/test_api/test_expert_review_detail_route.py tests/unit/test_database/test_migration_134_guarded_promote.py
$PY -m ruff format --check src/causal_engine/refutation_runner.py src/causal_engine/expert_review_gate.py src/repositories/causal_path.py src/repositories/causal_validation.py src/repositories/json_utils.py src/api/routes/expert_review.py src/api/schemas/expert_review.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py tests/unit/test_api/test_expert_review_detail_route.py tests/unit/test_database/test_migration_134_guarded_promote.py
free -m | awk '/Mem:/ {print "MemAvailable MiB:", $7}'   # stop and report if < 1500
$PY -m mypy --config-file pyproject.toml src/causal_engine/refutation_runner.py src/causal_engine/expert_review_gate.py src/repositories/causal_path.py src/repositories/causal_validation.py src/repositories/json_utils.py src/repositories/discovered_dag.py src/api/routes/expert_review.py src/api/schemas/expert_review.py
```

Expected: all green. Fix and amend into the owning task's commit style (a new `fix(...)` commit is fine).

- [ ] **Step 2: Frontend gates**

```bash
cd frontend && npx vitest run src/components/causal src/components/expert-review src/pages/ExpertReviews.test.tsx src/hooks/api/use-expert-review.test.ts && npm run typecheck && npx eslint src/components/causal src/components/expert-review src/pages/ExpertReviews.tsx src/hooks/api/use-expert-review.ts src/api/expert-review.ts src/types && cd ..
```

- [ ] **Step 3: Codex read-only audit to a fixed point (subscription channel only)**

Run from the worktree; the brief MUST contain the pushback paragraph verbatim:

```bash
SPEC=docs/superpowers/specs/2026-09-08-expert-review-loop-closure-design.md
git diff origin/main...HEAD --stat > /tmp/lane1_diffstat.txt
codex exec --sandbox read-only -C "$PWD" < /dev/null "You are auditing branch claude/lane1-expert-review-loop against the spec at $SPEC. Read the spec §2 and §4 first, then 'git diff origin/main...HEAD'. Report findings as HIGH/MED/LOW with file:line and a one-line repro or reasoning, then end with exactly one line 'VERDICT: ACCEPT' or 'VERDICT: REJECT'. Focus: (1) refutation_runner.py resample loops — any path that fabricates evidence, any way SKIPPED/partial results could read as passed, deadline handling, seeding, p-value provenance; (2) migration 134 — is the rejection rule inside the UPDATE equivalent to ExpertReviewGate._latest_adjudication (brand handling, NULL hash, reopened-by-newer-pending), grants; (3) the route — declaration order, 404/503 honesty, brand-scoped history; (4) frontend — anything rendered that the API did not return; (5) tests — vacuous assertions. If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given." 2>&1 | tee /tmp/lane1_codex_iter1.txt | tail -40
```

Read only the FINAL codex block for the verdict. Fix every HIGH and MED with a test first, commit, re-run with `iter2`, `iter3` … until `VERDICT: ACCEPT`. Codex's sandbox has no network and can hang asyncio teardown; a finding about test timing under the sandbox is the sandbox's, not ours (control: an unchanged CI-green test file). On an auth error, ask the owner to run `codex login`; never use an API key.

---

### Task 13: Baseline live run on the CURRENT image (before merge)

The before-half of the impact measurement. Runs against production as an operator would; writes are ordinary product writes (job store, `causal_validations`, `discovered_dags`, existing pending rows are re-used by hash). Authorised 2026-09-08.

**Files:**
- Create (scratch, then copied into the results dir): `<scratchpad>/lane1_live/run_discovery.py`
- Create: `docs/demos/results/<YYYY-MM-DD>_expert_review_loop/baseline.json` (+ `baseline.md`)

- [ ] **Step 1: The discovery runner script**

```python
#!/usr/bin/env python3
"""Run the Remibrutinib patient-grain discovery job and record every question's
band and evidence. Usage: run_discovery.py <label> <out_dir>  (reads .env)."""
import base64, json, os, subprocess, sys, time, urllib.parse, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
API = os.environ.get("E2I_API_BASE", "https://eznomics.site/api")
DATASET, BRAND = "patient_journeys", "Remibrutinib"

def mint_token() -> str:
    body = json.dumps({"email": os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local"),
                       "password": os.environ["E2I_ADMIN_PASSWORD"]}).encode()
    req = urllib.request.Request(f"{os.environ['SUPABASE_URL']}/auth/v1/token?grant_type=password",
                                 data=body, headers={"apikey": os.environ["SUPABASE_ANON_KEY"],
                                                     "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["access_token"]

def call(token, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _psql_rows(where: str) -> dict:
    # The agent path writes json.dumps(details) INTO the jsonb column, so 480 live
    # rows are JSON *strings* (measured 2026-09-08: 480 string / 545 object);
    # decode both shapes or every key read below is NULL.
    sql = ("with d as (select test_type, status, created_at, "
           "case when jsonb_typeof(details_json) = 'string' then (details_json #>> '{}')::jsonb else details_json end as dj "
           f"from public.causal_validations where {where}) "
           "select test_type, status, coalesce(dj->>'stopped_for_budget','') as budget, "
           "coalesce(jsonb_array_length(dj->'subset_effects'), jsonb_array_length(dj->'bootstrap_effects'), 0) as n "
           "from d order by created_at desc")
    proc = subprocess.run(["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tA", "-F", "|", "-c", sql],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:  # never let a failed read masquerade as "no evidence"
        raise RuntimeError(f"causal_validations read failed: {proc.stderr.strip()}")
    found = {}
    for ln in proc.stdout.splitlines():
        p = ln.split("|")
        if len(p) == 4 and p[0] not in found:      # newest row per test
            found[p[0]] = {"status": p[1], "stopped_for_budget": p[2], "n_effects": int(p[3])}
    return found

def db_tests(analysis_id, treatment: str, outcome: str, since_iso: str) -> dict:
    """Per-test status and evidence size straight from causal_validations.
    The API omits SKIPPED tests from refutation.tests and keeps only the message
    text of details, so the baseline's 'skipped' and the new loops' resample
    counts are only visible here. Unlinked runs are keyed by the query-derived
    uuid5 (src/repositories/causal_validation.py); runs linked to a causal_paths
    row are keyed by the PATH-derived uuid5, which the API does not expose, so
    fall back to the pair's newest rows written since this job started."""
    if analysis_id:
        qid = uuid.uuid5(uuid.NAMESPACE_URL, f"e2i:causal_query:{analysis_id}")
        found = _psql_rows(f"estimate_id = '{qid}'")
        if found:
            return found
    # Linked suites are written with estimate_source='causal_paths' (545 live rows),
    # so no source filter. Pin ONE suite -- the newest estimate_id for this pair,
    # brand and job window -- never the newest row per test across suites.
    pick = ("select estimate_id from public.causal_validations "
            f"where treatment_variable = '{treatment}' and outcome_variable = '{outcome}' "
            f"and brand = '{BRAND}' and created_at >= '{since_iso}' order by created_at desc limit 1")
    proc = subprocess.run(["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tA", "-c", pick],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"causal_validations lookup failed: {proc.stderr.strip()}")
    suite_id = proc.stdout.strip()
    # The path-derived id is shared by EVERY run of that path (derive_causal_path_estimate_id),
    # so keep the job window on the row read too: this run's suite, nothing older.
    return _psql_rows(f"estimate_id = '{suite_id}' and created_at >= '{since_iso}'") if suite_id else {}

def main(label: str, out_dir: str) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    token = mint_token()
    since_iso = datetime.now(timezone.utc).isoformat()
    q = urllib.parse.urlencode({"dataset": DATASET, "brand": BRAND})
    job = call(token, "POST", f"/causal/discover-effects?{q}", body={})
    job_id = job["job_id"]; print("job", job_id, "total", job["total"], flush=True)
    t0 = time.time()
    while True:
        time.sleep(20)
        job = call(token, "GET", f"/causal/discover-effects/{job_id}")
        done = sum(1 for e in job["effects"] if e["status"] not in ("pending", "running"))
        print(f"  {done}/{job['total']} after {int(time.time()-t0)}s", flush=True)
        if done >= job["total"] or job.get("error"):
            break
        if time.time() - t0 > 3 * 3600:
            print("giving up after 3h", flush=True); break
    rows = []
    for e in job["effects"]:
        detail = call(token, "GET", f"/causal/agent-analyze/{e['analysis_id']}") if e.get("analysis_id") else {}
        ref = detail.get("refutation") or {}
        rows.append({
            "treatment": e["treatment"], "outcome": e["outcome"], "row_status": e["status"],
            "db_tests": db_tests(e.get("analysis_id"), e["treatment"], e["outcome"], since_iso),
            "gate_decision": ref.get("gate_decision"), "run_status": detail.get("status"),
            "expert_review_decision": ref.get("expert_review_decision"), "expert_review_id": ref.get("expert_review_id"),
            "discovered_dag_id": detail.get("discovered_dag_id"), "analysis_id": e.get("analysis_id"),
            "tests": {t["test_name"]: t.get("status") or ("passed" if t.get("passed") else "failed") for t in ref.get("tests", [])},
            "ate": detail.get("ate"), "ci": [detail.get("ate_ci_lower"), detail.get("ate_ci_upper")],
            "warnings": detail.get("warnings", []),
        })
    (out / f"{label}.json").write_text(json.dumps({"job_id": job_id, "image_marker": None, "rows": rows}, indent=2))
    bands = {}
    for r in rows: bands[r["gate_decision"]] = bands.get(r["gate_decision"], 0) + 1
    print("bands", bands)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Record the image, run, and summarise**

```bash
OUT=docs/demos/results/$(date +%F)_expert_review_loop
mkdir -p $OUT
docker inspect e2i_api --format '{{.Config.Image}}' | tee $OUT/baseline_image.txt
$PY <scratchpad>/lane1_live/run_discovery.py baseline $OUT 2>&1 | tee $OUT/baseline.log
```

Expected: 12 rows; bands consistent with the historical per-pair table (spec §7 step 1). Write `$OUT/baseline.md`: a table of question, band, sensitivity status, review decision, DAG id, plus the image tag. Copy the script into `$OUT/run_discovery.py`. Commit the results directory on the lane branch (docs only):

```bash
git add $OUT && git commit -m "docs(demos): lane 1 baseline discovery run on the pre-lane image

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv"
```

---

### Task 14: PR, merge, deploy, certify content

Merging deploys to production (the PR touches `src/**`, `frontend/**`, `database/**`). Pushing the branch and opening the PR are within the lane; **merging waits for the owner's explicit go** after CI is green and Task 13's baseline is committed.

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin claude/lane1-expert-review-loop
cat > /tmp/lane1_pr_body.md <<'EOF'
Closes the expert-review loop (spec docs/superpowers/specs/2026-09-08-expert-review-loop-closure-design.md).

## What ships
- refutation_runner: data_subset and bootstrap score REAL per-resample evidence (previously computed and discarded; 96/96 live runs SKIPPED). Bootstrap thresholds 0.50/0.75 → 1.50/1.75 (code contradicted its comment; measured live ratios 1.01 / 0.81). Loops are deadline-aware and seeded from the estimate id.
- migration 134: promote_causal_path_guarded evaluates the review chronology INSIDE the UPDATE; node passes hash + brand. Rehearsed BEGIN/ROLLBACK ×2 on the live DB.
- GET /expert-reviews/{review_id}: one review in any status + same-structure history.
- Drill-down: review state, rejection reason, discovered_dag_id, deep link. Queue page: linked-review card, global brand filter, honest summary error, assessment auto-generate + prefetch.
- Lineage page rewritten to the shipped state; anchors re-resolved.

## Measured before building
0 REVIEW in 96 live runs by construction (arithmetic in the spec §2); the two non-critical tests were computed and discarded at the same cost the new loops have. The plan's own assumptions were attacked before Task 1 (section "Adversarial review before Task 1" at the end of this file): three Task-1 stub assumptions measured true, the SQL rule measured against the Python rule on 13 scenarios (one real divergence, fixed), the lineage edit fragments and anchor script measured, the live scripts' field names and the operator's role checked. Codex iter-1 returned REJECT with 1 HIGH + 5 MED, iter-2 REJECT with 2 HIGH + 4 MED + 1 LOW, iter-3 REJECT with 1 HIGH + 4 MED, iter-4 (the last pre-execution round) REJECT with 3 MED and no HIGH; all twenty-one were verified and folded in, four of them by live measurement (row FOR SHARE blocks a racing resolve but not a racing INSERT; LOCK TABLE IN SHARE MODE blocks both and not reads; the tie-only Python rule plus the moved rejection block keep every gate test green with both readers agreeing; the amended migration applies twice and returns the expected verdicts).

## Verification
Baseline discovery run on the pre-lane image: docs/demos/results/<date>_expert_review_loop/baseline.md. Post-deploy impact run, approve/reject re-runs and the switch step follow the spec §7 and are recorded in the same directory.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv
EOF
gh pr create --title "Expert-review loop closure: real non-critical evidence, guarded promote, review lookup, drill-down + queue UI, lineage refresh" --body-file /tmp/lane1_pr_body.md --base main --head claude/lane1-expert-review-loop
gh pr view --json number,url -q '"\(.number) \(.url)"'
```

- [ ] **Step 2: Wait for CI; fix reds on the branch**

```bash
gh pr checks --watch --interval 60
```

All required checks green (Backend Tests incl. heavy + agents lanes, Verify OpenAPI Types, Frontend Tests, Security Scanning, guards). A `verify-types` red means Task 7 must be re-run after a later docstring change.

- [ ] **Step 3: Owner go, then merge (never squash)**

```bash
gh pr merge <number> --merge
```

- [ ] **Step 4: Certify the deployed content (not the job status)**

Wait for the LAST `deploy.yml` run on main to be terminal, then:

```bash
gh run list --workflow=deploy.yml --branch main --limit 3
MERGE_SHA=$(git rev-parse --short origin/main)
docker inspect e2i_api --format '{{.Config.Image}}' | grep -c "$MERGE_SHA"       # 1
docker exec e2i_api grep -c '_resample_effects' /app/src/causal_engine/refutation_runner.py   # >= 3
docker exec e2i_api grep -c 'promote_causal_path_guarded' /app/src/repositories/causal_path.py  # >= 1
docker exec supabase-db psql -U postgres -d postgres -tA -c "select count(*) from pg_proc where proname in ('dag_structure_rejected','promote_causal_path_guarded')"   # 2
docker exec supabase-db psql -U postgres -d postgres -tA -c "select has_function_privilege('anon','public.promote_causal_path_guarded(text,text,text[],text,text)','EXECUTE')"   # f
docker exec supabase-db psql -U postgres -d postgres -tA -c "select count(*) from public.schema_migrations where version::text like '%134%' or version::text like '%135%'"   # 2 (read the table's columns first: \d public.schema_migrations)
docker exec supabase-db psql -U postgres -d postgres -tA -c "select count(*) from public.causal_validations where jsonb_typeof(details_json)='string' or jsonb_typeof(test_config)='string'"   # 0 (was 480 before the deploy)
docker exec e2i_api grep -c 'to_plain_json' /app/src/repositories/causal_validation.py   # >= 2
curl -s https://eznomics.site/api/openapi.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('/api/expert-reviews/{review_id}' in d['paths'] or '/expert-reviews/{review_id}' in d['paths'])"   # True
curl -s -o /dev/null -w '%{http_code}\n' https://eznomics.site/health   # 200
```

Negative control for the frontend bundle: before merging, record `docker exec e2i_frontend grep -rl 'Pending expert review' /usr/share/nginx/html/assets | wc -l` (expect 0); after deploy expect ≥ 1.

---

### Task 15: Live verification — impact run, approve, reject, switch

**Files:** `docs/demos/results/<date>_expert_review_loop/{impact.json,impact.md,adjudications.md,switch.md}`

- [ ] **Step 1: Impact run on the new image**

```bash
OUT=docs/demos/results/<date>_expert_review_loop
docker inspect e2i_api --format '{{.Config.Image}}' | tee $OUT/impact_image.txt
$PY $OUT/run_discovery.py impact $OUT 2>&1 | tee $OUT/impact.log
python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
b = {(r["treatment"], r["outcome"]): r for r in json.loads((out/"baseline.json").read_text())["rows"]}
i = {(r["treatment"], r["outcome"]): r for r in json.loads((out/"impact.json").read_text())["rows"]}
lines = ["| question | baseline band | impact band | data_subset | bootstrap | sensitivity | review decision |", "|---|---|---|---|---|---|---|"]
review_rows = []
for k in sorted(i):
    r = i[k]; t = r["tests"]; db = r.get("db_tests", {}); bb = b.get(k, {}); bdb = bb.get("db_tests", {})
    def st(name): return (db.get(name) or {}).get("status") or t.get(name) or "absent"
    def bst(name): return (bdb.get(name) or {}).get("status") or bb.get("tests", {}).get(name) or "absent"
    sens = st("sensitivity_e_value") if st("sensitivity_e_value") != "absent" else t.get("unobserved_common_cause")
    lines.append(f"| {k[0]} → {k[1]} | {bb.get('gate_decision')} | {r['gate_decision']} | {bst('data_subset')} → {st('data_subset')} ({(db.get('data_subset') or {}).get('n_effects', 0)} effects) | {bst('bootstrap')} → {st('bootstrap')} ({(db.get('bootstrap') or {}).get('n_effects', 0)} effects) | {sens} | {r['expert_review_decision']} |")
    if r["gate_decision"] == "review": review_rows.append(k)
(out/"impact.md").write_text("\n".join(lines) + f"\n\nREVIEW rows: {review_rows}\n")
print("\n".join(lines)); print("REVIEW rows:", review_rows)
EOF
```

Expected on stable pairs: data_subset and bootstrap now `passed` (baseline `skipped` → `passed`, with 5 and 50 effects recorded), bands unchanged from baseline. The API omits SKIPPED tests from `refutation.tests` and strips `details` to its message, so both columns read from `causal_validations` (the `db_tests` field) — never from the API's absence of a key. Record any REVIEW row; it drives Step 4.

- [ ] **Step 2: Approve the probe's row through the UI, then re-run its pair**

1. Open `https://eznomics.site/expert-reviews?review=4eab7033-7422-422d-83f6-659c9c3b9987` as the operator. The linked-review card shows `pending`, `treatment_arm → persistent_180d`, no brand. Tick the checklist items you can vouch for, add the comment `lane-1 live verification: structural approval`, click **Approve**. The card re-reads as `approved`.
2. Re-run the exact probe request and read the review fields:

```bash
$PY - <<'EOF'
import json, os, sys, time, urllib.request
sys.path.insert(0, "docs/demos/results/<date>_expert_review_loop"); from run_discovery import mint_token, call
tok = mint_token()
body = {"treatment_var": "treatment_arm", "outcome_var": "persistent_180d", "dataset": "patient_journeys", "limit": 1500}
r = call(tok, "POST", "/causal/agent-analyze", body=body)
aid = r["analysis_id"]
while r["status"] not in ("completed", "needs_review", "failed"):
    time.sleep(10); r = call(tok, "GET", f"/causal/agent-analyze/{aid}")
ref = r["refutation"]
print(json.dumps({"analysis_id": aid, "status": r["status"], "gate": ref["gate_decision"], "decision": ref.get("expert_review_decision"), "review_id": ref.get("expert_review_id"), "dag_id": r.get("discovered_dag_id"), "warnings": r.get("warnings")}, indent=2))
EOF
```

Expected: `decision: proceed`, `review_id: 4eab7033-…`, `gate: block` (approval is structural; the estimate still fails the sensitivity test), `status: failed`, a caveat naming the approval. Open the run in the Causal Analysis page: the Review status block reads **Structure approved** with the link. Record in `adjudications.md`.

- [ ] **Step 3: Reject one other synthetic row, re-run its pair**

Pick a brand-less pending row other than the probe's (e.g. `treatment_initiated → persistent_180d`; list them with `docker exec supabase-db psql -U postgres -d postgres -c "select review_id, treatment_variable, outcome_variable from public.expert_reviews where approval_status='pending' and brand is null order by created_at desc"`). Open `…/expert-reviews?review=<id>`, comment `lane-1 live verification: rejected to exercise the halt`, click **Reject**. Re-run that pair with the script above (change `treatment_var`/`outcome_var`). Expected: `status: failed`, `decision: rejected`, warnings contain `Estimate withheld: a domain expert REJECTED this DAG structure`, no new pending row for that hash (`select count(*) from expert_reviews where dag_version_hash='<hash>' and approval_status='pending'` = 0), the drill-down shows **Structure rejected** with the reason and link. Record.

- [ ] **Step 4: The switch (only if Step 1 produced a REVIEW row)**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
grep -q '^CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=' .env || echo 'CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=true' >> .env
docker compose -f docker/docker-compose.yml up -d --no-deps api
docker exec e2i_api sh -c 'printenv CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL'   # true
curl -s -o /dev/null -w '%{http_code}\n' https://eznomics.site/health       # 200
```

Re-run the REVIEW-band question with the re-run script. Expected: `status: failed`, `current_phase awaiting_expert_review` in the record, warnings naming the review id and `POST /expert-reviews/{id}/resolve`. Record in `switch.md`. **Owner decides**: leave on, or revert with `sed -i '/^CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=/d' .env && docker compose -f docker/docker-compose.yml up -d --no-deps api`.

If Step 1 produced no REVIEW row: write `switch.md` stating so with the impact table, cite `tests/unit/test_agents/test_causal_impact/test_refutation_expert_review_enforcement_1971.py` as the switch's executable demonstration, and put the band-semantics question to the owner with the numbers (spec §2: a sensitivity-WARNING→REVIEW rule would have moved 34 of 49 historical PROCEED runs).

- [ ] **Step 5: Commit the evidence**

```bash
git -C /home/enunez/Projects/e2i_causal_analytics checkout -b docs/lane1-live-verification origin/main
# copy docs/demos/results/<date>_expert_review_loop/ into this branch, commit, push, PR (docs only), merge with --merge
```

---

### Task 16: Record and close out

- [ ] **Step 1: PR certification comment** — image tag, marker counts, migration + privilege checks, impact table, the two adjudications with review ids and analysis ids, the switch outcome.
- [ ] **Step 2: Issues** (owner already asked for these to be filed with evidence): (a) "REVIEW band unreachable by construction; band-semantics decision" with the §2 arithmetic and the impact table; (b) "Reconstruction's own interval is unusable (SE 4.9 vs 0.034 reported); never use it as a reference" as a documented caveat; (c) the CausalPFN trial (spec §11) as the next lane; (d) the four simplification candidates (spec §10) as one tracking issue; (e) the same json.dumps-into-jsonb pattern remains on `expert_reviews.agent_assessment_json` (4 string rows live) and `checklist_json` (1) — readers decode both today; fix those writers (`src/repositories/expert_review.py` ~lines 350, 393, 431) and backfill the way Task 2b did for `causal_validations`; (f) `POST /expert-reviews/{id}/assessment` has no in-flight lock, so two concurrent uncached requests both build (the UI now guards client-side).
- [ ] **Step 3: Memory** — one project memory file for this lane (what the live DB said, the threshold inversion, the arithmetic, the disproof numbers, what the switch step showed), plus a MEMORY.md index line under 200 chars.
- [ ] **Step 4: Handoff** — `.claude/handoffs/current.md` with `status: complete` (or `in_progress` with the exact next step), and `git worktree remove .worktrees/lane1-review-loop` once merged.

---

## Self-review against the spec

- §4.1 engine → Tasks 1, 2 (loops, thresholds, deadline, seed, p-value, details, minimums). ✔
- §4.2 route → Task 6 (+ Task 7 types). ✔
- §4.3 guarded promote → Tasks 3, 4, 5. ✔
- §4.4 drill-down → Task 9. §4.5 queue page → Tasks 8, 10 (linked card, brand filter, summary error, auto-assessment, prefetch, `detail` key + invalidations). ✔
- §4.6 lineage → Task 11. §6 testing → each task's red-first steps; CI coverage confirmed (heavy lane runs `test_causal_engine`, agents shards run `test_agents`, main shard runs `test_api`, `test_repositories`, `test_database`). ✔
- §7 live verification → Tasks 13, 14, 15 in the spec's order (baseline BEFORE merge). ✔
- §8 rollout → Task 14. §9 decisions → Task 16. ✔
- Names used consistently: `_resample_effects`, `_refit_effect_on`, `_refutation_frame`, `_significance_p_value`, `_budget_skip_result`, `_resample_seed_for`, `resample_seed`, `deadline`; `promote_causal_path_guarded(p_path_id, p_new_status, p_allowed_current, p_dag_version_hash, p_brand)`, `dag_structure_rejected(hash, brand)`; `ReviewRecord`, `ExpertReviewDetailResponse`, `get_expert_review`, `getExpertReview`, `useExpertReview`, `queryKeys.expertReviews.detail`; `ReviewStatusPanel`, `LinkedReviewCard`, `PrepareAssessmentsButton`, `ResolveForm`, `DagPanel`, `checklist.ts`. ✔

## Adversarial review before Task 1 (2026-09-08)

Run before any plan code was written: targeted local disproofs (about a minute each), a 13-scenario
SQL-versus-Python equivalence check on the live database (BEGIN … ROLLBACK, nothing persisted), and
one codex read-only audit of spec + plan (subscription channel, pushback paragraph included) which
returned `VERDICT: REJECT` with 1 HIGH and 5 MED. Everything below was measured on `f30e9e9df` plus
the three lane doc commits; the edits are folded into the tasks above.

### Local disproofs

| # | Assumption attacked | Result |
|---|---|---|
| 1 | `RefutationRunner(config={"data_subset": {"num_subsets": 10}})` merges onto the defaults | TRUE — `__init__` deep-copies `DEFAULT_CONFIG` and `.update()`s per key → `{'enabled': True, 'subset_fraction': 0.8, 'num_subsets': 10, 'critical': False}` |
| 2 | DoWhy's `test_significance` accepts an estimate with only `.value` | TRUE — dowhy 0.14 `causal_refuter.py`: `perform_normal_distribution_test` (< 100 sims) and `perform_bootstrap_test` (≥ 100) read `estimate.value` only. A zero-variance resample set gives a non-finite z → the plan raises `RefutationError`, the same F-014 contract as today's `_require_p_value` |
| 3 | `causal_model._data` is the frame the reconstructed estimator was fitted on | TRUE — the refutation node builds `CausalModel(data=…)` AFTER binarising the treatment and encoding covariates (`nodes/refutation.py:607`), DoWhy fits `estimate_effect` on `self._data` (`causal_model.py:64, 416`), and the runner receives that model (`:1606`). Both DoWhy refuters' `_refute_once` make the four calls `_refit_effect_on` makes, verbatim |
| 4 | sed counts 15 / 2 / 1; helper at 49–61; tests at 751 / 798; import lines 27 / 48 | all TRUE |
| 5 | `TestReviewBandArithmetic` passes on the CURRENT code | 3 passed |
| 6 | `_run_test_with_tracing` forwards `**kwargs` (Task 2 wiring) | TRUE (`refutation_runner.py:934`) |
| 7 | Task 10: `/^review$/i` collides with nothing; the auto-assessment fires once under StrictMode | TRUE — the row button's name toggles `Review` / `Close`; "Prepare assessments" is an aria-label and "(Re)generate agent assessment" do not match; fixtures render one row; `main.tsx:36` wraps in StrictMode and refs survive its simulated remount. A SECOND form for the same review (linked card + queue row) was the real gap — fixed |
| 8 | Task 11: each `old` fragment occurs exactly once; pinned `28dbafb` present | 11 / 11 count = 1; TRUE |
| 9 | Task 11: the anchor script yields "zero unresolved" | FALSE — 15 unresolved: 10 index rows use SHORT paths the script never alias-resolved, `state.py` is ambiguous (10 files), and two anchors already pointed at BLANK lines at 28dbafb. With the fixed script: changed 35 / unresolved 4 (the two blank ones, each as anchor + index row) — hand-fix recipe added |
| 10 | Tasks 13 / 15: field names, parameter style, auth, reachability | TRUE — `DiscoverEffectsResponse` (job_id / total / completed / error / effects[]), `DiscoveredEffect` (treatment / outcome / status / analysis_id), `AgentCausalAnalysisResponse` (refutation.gate_decision / expert_review_id / expert_review_decision, discovered_dag_id, ate_ci_lower / upper), statuses completed / needs_review / failed; discover-effects = query params + optional body; the admin account's `app_metadata.role` is `admin` (≥ operator); `SUPABASE_URL` is the docker bridge (host-reachable); `/health` and `/api/health` 200. BUT the API omits SKIPPED tests and strips `details` to a message → the scripts now read `causal_validations` |
| 11 | Task 3 / 4: column types, roles, client key, rehearsal safety | `causal_paths.path_id` and `validation_status` are varchar(20) (text params fine); `service_role` / `anon` / `authenticated` exist; the node's client is service-role (`memory/services/factories.py:665–691`, `SUPABASE_SERVICE_KEY` is set) so service_role-only EXECUTE cannot break the promote; the live rejected hash has no open pending row, so the rehearsal INSERT cannot hit `uq_er_pending_dag_brand` |

### SQL ↔ Python equivalence (13 scenarios, live DB, BEGIN … ROLLBACK)

Equal on 12 scenarios. Divergent on S8-'' — Python's `if brand:` treats `''` as unfiltered (rejected), the
SQL filtered `brand = ''` (not rejected) — fixed with `NULLIF(p_brand, '')` and re-measured equal. S7
(a pending row with the SAME timestamp as the rejection): SQL `t`, Python row-order-dependent; kept strict
and documented as the conservative side, LOW follow-up in Task 16 (e). The scenario script and its
expected table are Task 3 Step 4b.

### Codex iter-1 findings → dispositions

1. HIGH — empty-string brand lets a promote pass over a rejection. CONFIRMED (independently measured) → Task 3 `NULLIF` ×2 + contract test; Task 5 passes `brand or None`.
2. MED — equal-timestamp pending / rejected rows read differently. CONFIRMED → documented as conservative; scenario S7 pinned; Task 16 (e).
3. MED — the repository factory sits outside the new route's `try`, so a client failure is a 500 not the promised 503. CONFIRMED (the existing routes behave the same; only the new route changes) → Task 6 + `test_client_factory_failure_is_503`.
4. MED — the live script cannot see SKIPPED tests or resample sizes. CONFIRMED (`refutation_runner.py:308` omits SKIPPED from `individual_tests`; `routes/causal.py:3621` keeps only message text) → Tasks 13 / 15 read `causal_validations` through the uuid5 the node derives.
5. MED — `toHaveBeenLastCalledWith(null)` on a hook that lives in an unmounted component. CONFIRMED → assertion inverted to `not.toHaveBeenCalled()`.
6. MED — two mounted forms for one review fire two auto-assessments. CONFIRMED structurally → page-level `Set` guard threaded to the card and the row, plus a test.

Owner decision 2026-09-09 (decision 2): a degenerate (zero-variance) resample distribution in a NON-critical
test is an honest SKIPPED with reason `degenerate_resample_distribution` (Task 1, `_degenerate_skip_result`),
not a fail-closed halt and never a placeholder p-value; the critical placebo gate catches an estimator that
ignores its data. Exceptions inside the loops still raise `RefutationError` (spec §5).

### Codex iter-2 findings → dispositions (after the iter-1 fold)

1. HIGH — the tie disposition "documented as conservative" is not equivalence. CONFIRMED → **Task 3b** (new): tie-only rule in `_latest_adjudication`; measured in scratch: 92 existing gate tests stay green (a global re-sort by `created_at` broke two tests whose rows carry no timestamps, hence tie-only), new test red-first.
2. HIGH — the STABLE predicate inside the UPDATE still leaves a statement-sized window for a rejection committed after the snapshot. CONFIRMED → `FOR SHARE` on the structure's review rows before the UPDATE (rejections are in-place UPDATEs of the pending row, `repository.resolve`). **Measured live, no writes**: with the lock held the racing UPDATE blocked (`while locking tuple … expert_reviews`) until its 3 s `statement_timeout`; after release it went through. First attempt was a false "no block" because the monitor loop matched its own `pg_sleep` query text — recipe fixed (Step 4c).
3. MED — the evidence reader would see NULL keys because the agent path stores `json.dumps(details)` in the JSONB column. CONFIRMED (480 string rows / 545 object rows live) → decode both shapes; writer fix filed as Task 16 (e).
4. MED — the fallback filtered `estimate_source='causal_impact_query'` and so excluded linked suites (`'causal_paths'`, 545 rows) and could pick another run. CONFIRMED → no source filter; pair + brand + job window.
5. MED — duplicate element ids when the linked card and the queue row render the same review. CONFIRMED (`${review.review_id}-${item.id}`) → `useId()` per form instance + an id-uniqueness assertion.
6. MED — "Prepare assessments" bypassed the shared guard. CONFIRMED → the button marks each id in the guard before its request; test expands a prepared row and asserts no second generation.
7. LOW — the anchor regex skipped the 18 `:NNN` shorthand anchors. CONFIRMED → `[^<:]*`; re-measured 42 changed / 5 unresolved, third hand-fix (`:1387` → the `_consult_review_gate` call, 1086 at f30e9e9df).

### Codex iter-3 findings → dispositions (after the iter-2 fold)

1. HIGH — a row-level FOR SHARE cannot cover a review row that does not exist yet (renew INSERTs a new pending row, which can then be rejected). CONFIRMED by reasoning (row locks cover retrieved rows only — the INSERT-under-FOR-SHARE case was NOT measured) and the replacement **measured**: with `LOCK TABLE public.expert_reviews IN SHARE MODE` held, the resolve-shaped UPDATE and the renew-shaped INSERT both waited (cancelled at their 3 s `statement_timeout`), a plain read went through, positive controls succeeded after release, nothing persisted → the function takes the table SHARE lock (Step 4c rewritten).
2. MED — "Prepare assessments" could re-request an id whose generation an expanded form already started. CONFIRMED → the button filters `missing` through the shared guard before running; new test.
3. MED — the evidence fallback could assemble rows from different suites or a brandless concurrent run. CONFIRMED → it now pins the newest `estimate_id` for the pair + brand + job window and reads that one suite.
4. MED — a failed `psql` read silently became `{}`. CONFIRMED → non-zero exit raises with stderr.
5. MED — `check_approval` still read a tied pending row as PENDING_REVIEW while `check_rejection` said REJECTED. CONFIRMED (the existing "REJECTED verdict is durable" block runs after the pending check) → Task 3b MOVES that block above the pending check; **measured in scratch**: both readers agree on the tie in either row order, newer pending rows and timestamp-less rows behave as before, 109 tests green across the gate, enforcement and promoter files.

Codex could not run the gate tests itself (its sandbox lacks a writable cache directory); the 109-green figure is this session's measurement, recorded in Task 3b.

### Codex iter-4 findings → dispositions (last pre-execution round)

1. MED — the linked fallback pins a path-derived `estimate_id` that every run of that path shares, so newest-per-test could still mix runs. CONFIRMED (`derive_causal_path_estimate_id` is per path) → the row read keeps the job window too.
2. MED — the bulk loop computed its skip list once; a form expanded while an earlier request was in flight could start an id the loop then requests again. CONFIRMED → the guard is re-checked per iteration.
3. MED — a failed bulk attempt left the id in the guard, so a later "Prepare" skipped it forever (for the page mount). CONFIRMED → the id is released on error; new test.

Stopping rule: four pre-execution rounds; the last returned no HIGH and three MED, all folded. Everything from
here is code, and Task 12's read-only audit runs on the real diff, where each of these dispositions is
re-checked against executed tests rather than plan text.

### Owner decisions (2026-09-09) folded

1. Subagent-driven execution (protocol section near the top). 2. Degenerate resample distribution → honest SKIPPED with reason (Task 1). 3. Table SHARE lock now; advisory lock only if reviewer contention ever appears (Task 3, unchanged). 4. Tie rule and the moved durable-rejection block kept (Task 3b). 5. Lock rehearsal hold 20 s → 6 s (Task 3 Step 4c). 6. Evidence writer fixed in-lane with backfill migration 135 so evidence is written, backfilled and tested in one shape (new Task 2b; Task 12 gates, Task 14 certification and Task 16 (e) updated).
