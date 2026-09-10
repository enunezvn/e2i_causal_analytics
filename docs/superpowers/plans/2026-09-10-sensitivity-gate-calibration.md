# Sensitivity Gate Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uncalibrated E-value cutoff in the refutation gate with a benchmarked reading so every correctly recovered planted effect is served, a CI that includes zero is served as an explicit null finding, and the runner, both agent nodes, the chat tool and the documentation page state the same verdict in the same words.

**Architecture:** Spec `docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md` (read §2, §3, §4 first). One new pure module `src/causal_engine/evalue.py` owns the E-value math, the measured-confounding benchmark and the classifier that returns a `SensitivityReading`. The runner's sensitivity test becomes non-critical and delegates to it; the refutation node computes the benchmark inputs on the full frame and threads them into the runner the way `outcome_std` is threaded today; the sensitivity node, the interpretation node and the chat tool import the same module. A heavy-lane calibration test pins the readings on the DGP's planted truth; a committed script re-bands the live runs before merge.

**Tech Stack:** Python 3.12, numpy, pandas, EconML LinearDML (tests only), pytest + pytest-asyncio; React 18 + TypeScript + vitest for the documentation page; PostgreSQL via `docker exec supabase-db psql` for live reads.

**Conventions for every task**

- Work in the lane worktree `W=/home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-d-1991` on branch `claude/1991-sensitivity-calibration`. Every Bash call uses absolute paths or a subshell `(cd $W && …)`; never a bare `cd` (the shell cwd is shared with the dispatcher). Run `git -C $W branch --show-current` before every commit. NEW COMMITS ONLY: never amend, reword, reset or rebase a commit already reported.
- Python: `PY=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python`. Tests: `(cd $W && $PY -m pytest <paths> -q -p no:cacheprovider -n 0)`. Lint: `$PY -m ruff check <files>` and `$PY -m ruff format --check <files>` (ruff 0.14.10). Type-check only the changed files: `(cd $W && $PY -m mypy --config-file pyproject.toml <files>)`; never the whole tree on this box.
- Memory: `free -m` before Task 8 (LinearDML) and Task 9 (frame pulls); stop and report if available memory is under 1.5 GiB.
- Frontend: `(cd $W/frontend && npx vitest run <paths>)`, `(cd $W/frontend && npm run typecheck)`. Never `prettier --write`.
- After each task is green: `ralph-wiggum:ralph-loop` around `codex:codex-rescue` (read-only) until `VERDICT: ACCEPT`; if the plugin says "CLI not installed", run `codex exec … < /dev/null` directly with `-C $W`. Cap briefs at the task's files. Every codex brief includes the CLAUDE.md pushback paragraph verbatim.
- Commit footer on every commit:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv
```

- Nothing is pushed until Task 12. Never squash.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `src/causal_engine/evalue.py` | new: E-value math, benchmark, `BenchmarkInputs`, `SensitivityReading`, `classify`, `benchmark_inputs_from_frame` | 1 |
| `tests/unit/test_causal_engine/test_evalue.py` | new: hand-value tests for every function and all five readings | 1 |
| `src/causal_engine/refutation_runner.py` | `DEFAULT_CONFIG` sensitivity non-critical, thresholds key deleted, critical set from config, `_run_sensitivity_test` delegates, new kwargs, #1989 pin, #1994 comment | 2 |
| `tests/unit/test_causal_engine/test_refutation_runner.py`, `test_refutation_runner_1419.py` | sensitivity tests rewritten to the readings; config dict cleaned | 2 |
| `tests/unit/test_causal_engine/test_refutation_bands_enumeration.py` | new: every reachable confidence value and band | 2 |
| `config/agent_config.yaml`, `tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py` | YAML describes the benchmark rule; residue test pins the absence of the cutoff | 3 |
| `src/agents/causal_impact/nodes/refutation.py` | benchmark inputs on the full frame → runner kwargs; null-finding caveat into `warnings` | 4 |
| `tests/unit/test_agents/test_causal_impact/test_refutation.py`, new `test_refutation_null_caveat.py` | threshold tests re-based on placebo; caveat test | 4 |
| `src/agents/causal_impact/nodes/sensitivity.py`, `src/agents/causal_impact/state.py` | node delegates to `evalue`; `SensitivityAnalysis` gains the reading keys | 5 |
| `tests/unit/test_agents/test_causal_impact/test_sensitivity.py`, `test_sensitivity_randomized.py`, `tests/unit/test_causal_engine/test_evalue_standardization_p4.py` | rewritten to the readings | 5 |
| `src/agents/causal_impact/nodes/interpretation.py` | robustness sentence and key finding from the reading | 6 |
| `tests/unit/test_agents/test_causal_impact/test_interpretation.py` | narrative assertions | 6 |
| `src/agents/tool_composer/tool_registrations.py`, `src/agents/tool_composer/tool_registry.py` | chat tool on the shared module; `reading` replaces `robustness` | 7 |
| `tests/unit/test_agents/test_tool_composer/test_tools_fail_closed.py` | tool tests | 7 |
| `tests/unit/test_causal_engine/test_sensitivity_calibration.py` | new, `heavy_ml`: planted truths read beyond; null pairs read null; omitted-confounder limit pinned | 8 |
| `scripts/calibration/reband_sensitivity_readings.py` | new: re-band the live `causal_impact_query` runs; writes `reband.md` | 9 |
| `docs/demos/results/<run-date>_sensitivity_calibration/reband.md` | the re-band table the owner reads before merge | 9 |
| `docs/lineage/causal_dag_lineage.html` | scoring table, callouts, E-value details, calculator, scenario rows, anchors | 10 |
| `frontend/src/components/documentation/content.ts`, `RefutationGate.tsx`, new `content.test.ts` | sensitivity entry, intro, illustration | 11 |

---

### Task 1: The E-value module

**Files:**
- Create: `src/causal_engine/evalue.py`
- Test: `tests/unit/test_causal_engine/test_evalue.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Hand-value tests for src/causal_engine/evalue.py (spec §4.1, §4.4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.causal_engine import evalue as ev


class TestMath:
    def test_e_value_from_rr_matches_vanderweele_ding(self):
        # RR 2 -> 2 + sqrt(2) = 3.414...
        assert ev.e_value_from_rr(2.0) == pytest.approx(2.0 + math.sqrt(2.0))
        assert ev.e_value_from_rr(1.0) == 1.0

    def test_protective_rr_is_inverted(self):
        assert ev.e_value_from_rr(0.5) == pytest.approx(ev.e_value_from_rr(2.0))

    def test_rr_from_smd_uses_the_chinn_factor(self):
        assert ev.rr_from_smd(1.0) == pytest.approx(math.exp(0.91))
        assert ev.rr_from_smd(-1.0) == pytest.approx(math.exp(0.91))

    def test_rr_from_risk_difference_orients_and_guards_domain(self):
        assert ev.rr_from_risk_difference(0.10, 0.30) == pytest.approx(0.40 / 0.30)
        # a negative RD reverses the exposure coding: RR = p0 / p1
        assert ev.rr_from_risk_difference(-0.10, 0.30) == pytest.approx(0.30 / 0.20)
        assert ev.rr_from_risk_difference(0.10, 0.0) is None
        assert ev.rr_from_risk_difference(0.80, 0.30) is None  # p1 >= 1

    def test_bias_factor(self):
        assert ev.bias_factor(2.0, 2.0) == pytest.approx(4.0 / 3.0)
        assert ev.bias_factor(0.5, 2.0) == pytest.approx(4.0 / 3.0)  # oriented

    def test_joint_benchmark_is_oriented_and_none_without_naive(self):
        # naive RD 0.29 vs adjusted 0.16 at p0 0.30: (0.59/0.30)/(0.46/0.30)
        b = ev.joint_confounding_benchmark(0.29, 0.16, baseline_risk=0.30, outcome_std=None)
        assert b == pytest.approx((0.59 / 0.30) / (0.46 / 0.30))
        assert ev.joint_confounding_benchmark(0.10, 0.20, baseline_risk=0.30, outcome_std=None) == pytest.approx(
            (0.50 / 0.30) / (0.40 / 0.30)
        )
        assert ev.joint_confounding_benchmark(None, 0.16, baseline_risk=0.30, outcome_std=None) is None

    def test_joint_benchmark_falls_back_to_smd_without_baseline_risk(self):
        b = ev.joint_confounding_benchmark(0.4, 0.2, baseline_risk=None, outcome_std=1.0)
        assert b == pytest.approx(math.exp(0.91 * 0.4) / math.exp(0.91 * 0.2))

    def test_measured_confounding_benchmark_prefers_joint(self):
        assert ev.measured_confounding_benchmark(1.2, {"a": 1.5}) == (1.2, "joint_naive_vs_adjusted")
        assert ev.measured_confounding_benchmark(None, {"a": 1.5, "b": 1.1}) == (1.5, "strongest_covariate")
        assert ev.measured_confounding_benchmark(None, {}) == (None, "none_measured")


class TestCovariateBiasFactors:
    def _frame(self, seed: int = 0, n: int = 4000) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        c = rng.normal(size=n)  # continuous confounder
        b = rng.integers(0, 2, size=n)  # binary confounder
        t = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * c + 0.6 * b - 0.5)))).astype(int)
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * c + 0.4 * b + 0.3 * t - 0.4)))).astype(int)
        return pd.DataFrame({"t": t, "y": y, "c": c, "b": b, "noise": rng.normal(size=n)})

    def test_factors_exceed_one_for_real_confounders_and_hug_one_for_noise(self):
        f = ev.covariate_bias_factors(self._frame(), "t", "y", ["c", "b", "noise"])
        assert set(f) == {"c", "b", "noise"}
        assert f["c"] > 1.05 and f["b"] > 1.02
        assert f["noise"] < 1.05

    def test_empty_covariates_gives_empty_dict(self):
        assert ev.covariate_bias_factors(self._frame(), "t", "y", []) == {}

    def test_benchmark_inputs_from_frame(self):
        frame = self._frame()
        inp = ev.benchmark_inputs_from_frame(frame, "t", "y", ["c", "b"])
        p0 = frame.loc[frame.t == 0, "y"].mean()
        assert inp.baseline_risk == pytest.approx(p0)
        assert inp.naive_effect == pytest.approx(frame.loc[frame.t == 1, "y"].mean() - p0)
        assert inp.treatment_is_binary and inp.outcome_is_binary
        assert set(inp.covariate_bias_factors) == {"c", "b"}

    def test_continuous_treatment_has_no_naive_contrast(self):
        frame = self._frame().assign(t=lambda d: d.c)  # continuous treatment
        inp = ev.benchmark_inputs_from_frame(frame, "t", "y", ["b"])
        assert inp.naive_effect is None and inp.baseline_risk is None
        assert not inp.treatment_is_binary
        assert "b" in inp.covariate_bias_factors  # median split on T still works


class TestClassify:
    def _c(self, effect, ci, **kw):
        base = dict(randomized=False, baseline_risk=0.30, outcome_std=0.46, naive_effect=None,
                    covariate_factors={}, n_rows=1500)
        base.update(kw)
        return ev.classify(effect, ci, **base)

    def test_randomized_is_skipped_and_still_carries_numbers(self):
        r = self._c(0.15, (0.08, 0.22), randomized=True, naive_effect=0.20)
        assert r.reading == "not_applicable_randomized" and r.status == "skipped"
        assert r.e_value_point > 1.0 and r.headline.startswith("Not applicable")

    def test_null_finding_when_ci_includes_zero(self):
        r = self._c(0.05, (-0.02, 0.12), naive_effect=0.10)
        assert r.reading == "null_finding" and r.status == "warning"
        assert r.e_value_ci == 1.0 and r.ci_includes_null
        assert "includes zero" in r.message and "n = 1500" in r.message

    def test_beyond_when_point_rr_exceeds_benchmark(self):
        # adjusted 0.15 at p0 0.30 -> RR 1.5; naive 0.20 -> RR 1.667; B_obs 1.11
        r = self._c(0.15, (0.08, 0.22), naive_effect=0.20)
        assert r.reading == "beyond_measured_confounding" and r.status == "passed"
        assert r.rr_point == pytest.approx(1.5) and r.benchmark == pytest.approx((0.50 / 0.30) / 1.5)
        assert r.benchmark_basis == "joint_naive_vs_adjusted" and r.conversion == "risk_ratio"
        assert r.headline == "Robust to confounding at measured strength"

    def test_within_when_benchmark_is_at_least_the_point_rr(self):
        # adjusted 0.03 at p0 0.30 -> RR 1.10; naive 0.10 -> RR 1.333; B_obs 1.21 >= 1.10
        r = self._c(0.03, (0.01, 0.05), naive_effect=0.10)
        assert r.reading == "within_measured_confounding" and r.status == "warning"
        assert r.headline == "Sensitive to confounding"
        assert "could account for the whole effect" in r.message

    def test_tie_reads_within(self):
        # naive == adjusted -> B_obs == 1.0; force rr_point == 1.0 is impossible with CI>0,
        # so pin the rule on an exact equality via covariate factor instead.
        r = self._c(0.15, (0.08, 0.22), naive_effect=None, covariate_factors={"c": 1.5})
        assert r.benchmark == pytest.approx(1.5) and r.rr_point == pytest.approx(1.5)
        assert r.reading == "within_measured_confounding"

    def test_unbenchmarked_without_any_measured_confounding(self):
        r = self._c(0.15, (0.08, 0.22), naive_effect=None, covariate_factors={})
        assert r.reading == "unbenchmarked" and r.status == "warning"
        assert r.benchmark is None and r.benchmark_basis == "none_measured"

    def test_smd_path_without_baseline_risk(self):
        r = self._c(0.15, (0.08, 0.22), baseline_risk=None, naive_effect=0.20)
        assert r.conversion == "standardized_difference"
        assert r.rr_point == pytest.approx(math.exp(0.91 * 0.15 / 0.46))

    def test_negative_effect_is_oriented(self):
        r = self._c(-0.15, (-0.22, -0.08), naive_effect=-0.20)
        assert r.reading == "beyond_measured_confounding"
        assert r.rr_point == pytest.approx(0.30 / 0.15)

    def test_details_dict_is_json_plain(self):
        r = self._c(0.15, (0.08, 0.22), naive_effect=0.20, covariate_factors={"c": 1.1})
        d = r.as_details()
        assert d["reading"] == "beyond_measured_confounding"
        assert isinstance(d["covariate_bias_factors"], dict)
        assert all(isinstance(v, (float, int, str, bool, dict, type(None))) for v in d.values())

    def test_non_finite_inputs_raise(self):
        with pytest.raises(ValueError):
            self._c(float("nan"), (0.08, 0.22))
        with pytest.raises(ValueError):
            self._c(0.15, (0.08, float("inf")))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_evalue.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: `ModuleNotFoundError: No module named 'src.causal_engine.evalue'`

- [ ] **Step 3: Write the module**

```python
"""E-value math, the measured-confounding benchmark, and the sensitivity reading.

One home for what three engines used to copy (spec
``docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md`` §4.1):
the refutation runner, the agent's sensitivity node and the chat tool
``sensitivity_analyzer`` all import from here, so the platform reports ONE
E-value and ONE verdict for a run.

The E-value (VanderWeele & Ding 2017) is the minimum risk ratio an unmeasured
confounder would need with both treatment and outcome to explain an association
away. It is monotone in effect size and CANNOT detect confounding (measured
2026-09-10: omitted-confounder refits score as high as correct fits), so it is a
READING benchmarked against the confounding this run actually measured — never a
pass/fail gate. Precision is a separate statement: a CI that includes zero is a
null finding.

Pure functions, numpy/pandas only, no I/O. Out-of-domain inputs raise ValueError.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# Chinn (2000) / VanderWeele (2017): a standardized mean difference d maps to an
# odds ratio of exp(pi*d/sqrt(3)) ~ exp(1.81 d); the square-root transformation for
# a common outcome gives RR ~ exp(0.91 d). Continuous outcomes only.
SMD_TO_LOG_RR = 0.91

READING_BEYOND = "beyond_measured_confounding"
READING_WITHIN = "within_measured_confounding"
READING_NULL = "null_finding"
READING_UNBENCHMARKED = "unbenchmarked"
READING_RANDOMIZED = "not_applicable_randomized"

HEADLINES: Dict[str, str] = {
    READING_BEYOND: "Robust to confounding at measured strength",
    READING_WITHIN: "Sensitive to confounding",
    READING_NULL: "No detectable effect at this sample size",
    READING_UNBENCHMARKED: "Robustness not benchmarked: no measured confounders",
    READING_RANDOMIZED: "Not applicable: randomized design",
}

STATUS_BY_READING: Dict[str, str] = {
    READING_BEYOND: "passed",
    READING_WITHIN: "warning",
    READING_NULL: "warning",
    READING_UNBENCHMARKED: "warning",
    READING_RANDOMIZED: "skipped",
}

BASIS_IN_WORDS: Dict[str, str] = {
    "joint_naive_vs_adjusted": "the confounding the adjustment removed, naive vs adjusted",
    "strongest_covariate": "the strongest measured covariate's bias factor",
    "none_measured": "no measured confounders",
}


def _finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _orient(rr: float) -> float:
    """A protective ratio is reported on the harmful side (RR >= 1)."""
    if rr <= 0 or not math.isfinite(rr):
        raise ValueError(f"risk ratio must be positive and finite, got {rr!r}")
    return rr if rr >= 1.0 else 1.0 / rr


def e_value_from_rr(rr: float) -> float:
    """``E = RR + sqrt(RR*(RR-1))`` for RR >= 1; a protective RR is inverted first."""
    r = _orient(_finite("rr", rr))
    if r <= 1.0:
        return 1.0
    return float(r + math.sqrt(r * (r - 1.0)))


def rr_from_smd(d: float) -> float:
    """Approximate risk ratio for a standardized mean difference (continuous outcomes)."""
    return float(math.exp(SMD_TO_LOG_RR * abs(_finite("d", d))))


def rr_from_risk_difference(rd: float, baseline_risk: float) -> Optional[float]:
    """Risk ratio of a risk difference at control-arm risk ``baseline_risk``, oriented >= 1.

    ``p1 = p0 + rd``; ``RR = p1/p0``; a negative RD reverses the exposure coding
    (EValue package convention), i.e. ``RR = p0/p1``. ``None`` outside
    ``0 < p0 < 1`` and ``0 < p1 < 1`` (no risk-ratio path exists there).
    """
    p0 = _finite("baseline_risk", baseline_risk)
    p1 = p0 + _finite("rd", rd)
    if not (0.0 < p0 < 1.0) or not (0.0 < p1 < 1.0):
        return None
    return _orient(p1 / p0)


def bias_factor(rr_eu: float, rr_ud: float) -> float:
    """Ding & VanderWeele (2016) joint bounding factor ``B = RR_EU*RR_UD/(RR_EU+RR_UD-1)``."""
    a = _orient(_finite("rr_eu", rr_eu))
    b = _orient(_finite("rr_ud", rr_ud))
    return float(a * b / (a + b - 1.0))


def _rr_of_effect(effect: float, baseline_risk: Optional[float], outcome_std: Optional[float]) -> Tuple[float, str]:
    """Risk ratio of an effect: risk-difference path when a baseline risk exists, else SMD."""
    if baseline_risk is not None:
        rr = rr_from_risk_difference(effect, baseline_risk)
        if rr is not None:
            return rr, "risk_ratio"
    d = abs(effect)
    if outcome_std is not None and math.isfinite(outcome_std) and outcome_std > 0:
        d = d / outcome_std
    return rr_from_smd(d), "standardized_difference"


def joint_confounding_benchmark(
    naive_effect: Optional[float],
    adjusted_effect: float,
    *,
    baseline_risk: Optional[float],
    outcome_std: Optional[float],
) -> Optional[float]:
    """``B_obs = RR(naive)/RR(adjusted)`` oriented >= 1: the confounding the adjustment removed."""
    if naive_effect is None:
        return None
    rr_naive, _ = _rr_of_effect(_finite("naive_effect", naive_effect), baseline_risk, outcome_std)
    rr_adj, _ = _rr_of_effect(_finite("adjusted_effect", adjusted_effect), baseline_risk, outcome_std)
    return _orient(rr_naive / rr_adj)


def measured_confounding_benchmark(
    joint: Optional[float], covariate_factors: Mapping[str, float]
) -> Tuple[Optional[float], str]:
    """The benchmark and its basis: joint when available, else the strongest covariate, else none."""
    if joint is not None:
        return float(joint), "joint_naive_vs_adjusted"
    finite = {k: float(v) for k, v in covariate_factors.items() if v is not None and math.isfinite(float(v))}
    if finite:
        return max(finite.values()), "strongest_covariate"
    return None, "none_measured"


def _is_binary(values: np.ndarray) -> bool:
    u = np.unique(values[~np.isnan(values)])
    return len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}


def _high_mask(values: np.ndarray) -> np.ndarray:
    """Binary as-is (== 1); continuous split at the median (strictly above)."""
    if _is_binary(values):
        return values == 1.0
    return values > np.nanmedian(values)


def covariate_bias_factors(
    frame: Any, treatment: str, outcome: str, covariates: Sequence[str]
) -> Dict[str, float]:
    """Per-covariate bias factor from the frame (spec §4.1).

    RR_EU: share of high-covariate units among treated / among controls (treated is
    ``T == 1`` for a binary treatment, ``T > median`` otherwise). RR_UD: outcome rate
    among high-covariate CONTROLS / low-covariate controls (binary outcome), or the
    SMD path on the control-arm mean difference (continuous outcome). Covariates
    absent from the frame or without variation are skipped.
    """
    out: Dict[str, float] = {}
    if not covariates or frame is None:
        return out
    t = np.asarray(frame[treatment], dtype=float)
    y = np.asarray(frame[outcome], dtype=float)
    treated = _high_mask(t)
    control = ~treated
    y_binary = _is_binary(y)
    y_sd = float(np.nanstd(y)) if not y_binary else None
    for cov in covariates:
        if cov not in getattr(frame, "columns", []):
            continue
        x = np.asarray(frame[cov], dtype=float)
        ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(t)
        if ok.sum() < 20:
            continue
        hi = _high_mask(x[ok])
        tr, co = treated[ok], control[ok]
        p_hi_t, p_hi_c = hi[tr].mean() if tr.any() else np.nan, hi[co].mean() if co.any() else np.nan
        if not (p_hi_t > 0 and p_hi_c > 0):
            continue
        rr_eu = p_hi_t / p_hi_c
        yy = y[ok]
        hi_c, lo_c = hi & co, (~hi) & co
        if not (hi_c.any() and lo_c.any()):
            continue
        if y_binary:
            m_hi, m_lo = yy[hi_c].mean(), yy[lo_c].mean()
            if not (m_hi > 0 and m_lo > 0):
                continue
            rr_ud = m_hi / m_lo
        else:
            if not y_sd or y_sd <= 0:
                continue
            rr_ud = rr_from_smd((yy[hi_c].mean() - yy[lo_c].mean()) / y_sd)
        out[cov] = bias_factor(rr_eu, rr_ud)
    return out


@dataclass(frozen=True)
class BenchmarkInputs:
    """What the classifier needs from the FULL estimation frame (spec §4.3)."""

    baseline_risk: Optional[float]
    naive_effect: Optional[float]
    covariate_bias_factors: Dict[str, float] = field(default_factory=dict)
    treatment_is_binary: bool = False
    outcome_is_binary: bool = False
    n_rows: int = 0


def benchmark_inputs_from_frame(
    frame: Any,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    *,
    naive_effect: Optional[float] = None,
) -> BenchmarkInputs:
    """Baseline risk, naive contrast and covariate factors from one frame.

    ``naive_effect`` from the estimation node wins when given (it is the same
    contrast); it is recomputed only when missing and the treatment is binary.
    """
    t = np.asarray(frame[treatment], dtype=float)
    y = np.asarray(frame[outcome], dtype=float)
    ok = ~np.isnan(t) & ~np.isnan(y)
    t, y = t[ok], y[ok]
    t_bin, y_bin = _is_binary(t), _is_binary(y)
    baseline_risk: Optional[float] = None
    naive: Optional[float] = naive_effect
    if t_bin and (t == 0).any() and (t == 1).any():
        p0 = float(y[t == 0].mean())
        baseline_risk = p0 if y_bin else None
        if naive is None:
            naive = float(y[t == 1].mean() - p0)
    else:
        naive = None
    return BenchmarkInputs(
        baseline_risk=baseline_risk,
        naive_effect=naive,
        covariate_bias_factors=covariate_bias_factors(frame, treatment, outcome, covariates),
        treatment_is_binary=t_bin,
        outcome_is_binary=y_bin,
        n_rows=int(ok.sum()),
    )


@dataclass(frozen=True)
class SensitivityReading:
    """The verdict a leader reads, with every number behind it (spec §4.4)."""

    reading: str
    status: str
    headline: str
    message: str
    e_value_point: float
    e_value_ci: float
    rr_point: float
    rr_ci: float
    ci_includes_null: bool
    conversion: str
    baseline_risk: Optional[float]
    naive_effect: Optional[float]
    benchmark: Optional[float]
    benchmark_basis: str
    covariate_bias_factors: Dict[str, float]
    n_rows: Optional[int]

    def as_details(self) -> Dict[str, Any]:
        d = asdict(self)
        d["covariate_bias_factors"] = {k: float(v) for k, v in self.covariate_bias_factors.items()}
        return d


def classify(
    effect: float,
    ci: Tuple[float, float],
    *,
    randomized: bool,
    baseline_risk: Optional[float],
    outcome_std: Optional[float],
    naive_effect: Optional[float],
    covariate_factors: Mapping[str, float],
    n_rows: Optional[int],
) -> SensitivityReading:
    """Evaluate the reading rules of spec §4.4 in order."""
    eff = _finite("effect", effect)
    lo, hi = _finite("ci_lower", ci[0]), _finite("ci_upper", ci[1])
    if lo > hi:
        lo, hi = hi, lo
    includes_null = lo <= 0.0 <= hi
    bound = min(abs(lo), abs(hi))
    rr_point, conversion = _rr_of_effect(eff, baseline_risk, outcome_std)
    if includes_null:
        rr_ci = 1.0
    else:
        rr_ci, _ = _rr_of_effect(math.copysign(bound, eff), baseline_risk, outcome_std)
    e_point, e_ci = e_value_from_rr(rr_point), e_value_from_rr(rr_ci)
    joint = joint_confounding_benchmark(naive_effect, eff, baseline_risk=baseline_risk, outcome_std=outcome_std)
    benchmark, basis = measured_confounding_benchmark(joint, covariate_factors)

    if randomized:
        reading = READING_RANDOMIZED
    elif includes_null:
        reading = READING_NULL
    elif benchmark is None:
        reading = READING_UNBENCHMARKED
    elif rr_point > benchmark:
        reading = READING_BEYOND
    else:
        reading = READING_WITHIN

    n_txt = f"n = {n_rows}" if n_rows else "this sample size"
    basis_words = BASIS_IN_WORDS[basis]
    if reading == READING_RANDOMIZED:
        message = (
            "not applicable: randomized design — treatment assignment is exogenous by "
            "construction, so the unmeasured-confounding gate does not apply; E-value "
            f"(CI bound) {e_ci:.2f} reported for information only"
        )
    elif reading == READING_NULL:
        message = (
            f"The 95 % CI [{lo:.3f}, {hi:.3f}] includes zero at {n_txt}. The estimate is "
            "reported as a null finding; no unmeasured confounder is needed to explain it."
        )
    elif reading == READING_UNBENCHMARKED:
        message = (
            f"The interval excludes zero (E-value {e_point:.2f}), but no measured confounders "
            "exist for this design, so robustness to confounding cannot be benchmarked."
        )
    elif reading == READING_BEYOND:
        message = (
            "Explaining this effect away would need an unmeasured confounder with a risk ratio "
            f"of at least {e_point:.2f} with both treatment and outcome ({e_ci:.2f} at the CI "
            f"bound), stronger than all measured confounding combined ({benchmark:.2f}, {basis_words})."
        )
    else:
        message = (
            f"A confounder no stronger than the measured set ({benchmark:.2f}, {basis_words}) "
            f"could account for the whole effect (risk ratio at the estimate {rr_point:.2f}; "
            f"E-value {e_point:.2f}). Treat the direction as more reliable than the size."
        )
    return SensitivityReading(
        reading=reading,
        status=STATUS_BY_READING[reading],
        headline=HEADLINES[reading],
        message=message,
        e_value_point=e_point,
        e_value_ci=e_ci,
        rr_point=float(rr_point),
        rr_ci=float(rr_ci),
        ci_includes_null=includes_null,
        conversion=conversion,
        baseline_risk=None if baseline_risk is None else float(baseline_risk),
        naive_effect=None if naive_effect is None else float(naive_effect),
        benchmark=benchmark,
        benchmark_basis=basis,
        covariate_bias_factors={k: float(v) for k, v in covariate_factors.items()},
        n_rows=n_rows,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_evalue.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: `22 passed`. If `test_tie_reads_within` fails on the benchmark value, the SMD-path point RR for 0.15/0.46 is `exp(0.91·0.326) = 1.345`, not 1.5 — in that test pass `baseline_risk=0.30` (the default) so `rr_point = 0.45/0.30 = 1.5` exactly; keep the covariate factor at 1.5.

- [ ] **Step 5: Lint, type-check, commit**

```bash
(cd $W && $PY -m ruff check src/causal_engine/evalue.py tests/unit/test_causal_engine/test_evalue.py && $PY -m ruff format --check src/causal_engine/evalue.py tests/unit/test_causal_engine/test_evalue.py && $PY -m mypy --config-file pyproject.toml src/causal_engine/evalue.py)
git -C $W branch --show-current
git -C $W add src/causal_engine/evalue.py tests/unit/test_causal_engine/test_evalue.py
git -C $W commit -m "feat(causal): one E-value module with the measured-confounding benchmark and reading (#1991 debt 2)"
```

---

### Task 2: The runner reads, never gates

**Files:**
- Modify: `src/causal_engine/refutation_runner.py` (`DEFAULT_CONFIG` ~753, `PASS_THRESHOLDS` ~784, `run_all_tests` signature ~825 and its sensitivity dispatch ~1011, `_run_sensitivity_test` 1802–1935, `_determine_gate_decision` ~2040)
- Modify: `tests/unit/test_causal_engine/test_refutation_runner.py` (class `TestSensitivityTest`, ~874–956)
- Modify: `tests/unit/test_causal_engine/test_refutation_runner_1419.py:62`
- Create: `tests/unit/test_causal_engine/test_refutation_bands_enumeration.py`

- [ ] **Step 1: Rewrite the sensitivity tests to the readings (red)**

Replace the whole `class TestSensitivityTest` in `tests/unit/test_causal_engine/test_refutation_runner.py` with:

```python
class TestSensitivityTest:
    """The sensitivity test is a READING, never a gate (spec §4.4, §4.5)."""

    def test_sensitivity_is_not_critical_and_has_no_threshold(self, runner):
        assert runner.config["sensitivity_e_value"]["critical"] is False
        assert "e_value_threshold" not in runner.config["sensitivity_e_value"]
        assert "e_value_min" not in runner.thresholds

    def test_beyond_reads_passed_with_the_benchmark_in_details(self, runner):
        result = runner._run_sensitivity_test(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            baseline_risk=0.30,
            naive_effect=0.20,
        )
        assert result.test_name == RefutationTestType.SENSITIVITY_E_VALUE
        assert result.status == RefutationStatus.PASSED
        d = result.details
        assert d["reading"] == "beyond_measured_confounding"
        assert d["benchmark_basis"] == "joint_naive_vs_adjusted"
        assert d["rr_point"] == pytest.approx(1.5)
        assert d["headline"] == "Robust to confounding at measured strength"
        assert "stronger than all measured confounding" in d["message"]
        assert d["e_value"] == pytest.approx(d["e_value_point"])  # legacy key kept

    def test_within_reads_warning(self, runner):
        result = runner._run_sensitivity_test(
            original_effect=0.03,
            original_ci=(0.01, 0.05),
            baseline_risk=0.30,
            naive_effect=0.10,
        )
        assert result.status == RefutationStatus.WARNING
        assert result.details["reading"] == "within_measured_confounding"

    def test_null_crossing_ci_is_a_null_finding_not_a_failure(self, runner):
        """Replaces M-stat2: a strong point effect with a null-crossing CI is served
        as a null finding (WARNING), never FAILED, and e_value_ci collapses to 1.0."""
        result = runner._run_sensitivity_test(
            original_effect=0.5,
            original_ci=(-0.3, 0.5),
            baseline_risk=0.30,
            naive_effect=0.6,
        )
        assert result.status == RefutationStatus.WARNING
        assert result.details["reading"] == "null_finding"
        assert result.details["e_value_ci"] == 1.0
        assert result.details["e_value"] > 1.0  # point value still surfaced
        assert "includes zero" in result.details["message"]

    def test_smd_path_when_no_baseline_risk(self, runner):
        import numpy as np

        result = runner._run_sensitivity_test(
            original_effect=0.5, original_ci=(0.4, 0.6), outcome_std=1.0, naive_effect=0.55
        )
        rr_ci = np.exp(0.91 * 0.4)
        assert result.details["conversion"] == "standardized_difference"
        assert result.details["e_value_ci"] == pytest.approx(rr_ci + np.sqrt(rr_ci * (rr_ci - 1)), rel=1e-6)

    def test_unbenchmarked_without_naive_or_covariates(self, runner):
        result = runner._run_sensitivity_test(original_effect=0.15, original_ci=(0.08, 0.22))
        assert result.status == RefutationStatus.WARNING
        assert result.details["reading"] == "unbenchmarked"

    def test_sensitivity_never_fails(self, runner):
        for effect, ci in [(0.001, (0.0005, 0.0015)), (0.5, (-0.3, 0.5)), (0.02, (0.01, 0.03))]:
            result = runner._run_sensitivity_test(original_effect=effect, original_ci=ci, baseline_risk=0.3)
            assert result.status != RefutationStatus.FAILED

    def test_sensitivity_never_blocks_the_gate(self, runner):
        tests = [
            RefutationResult(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED, 0.1, 0.1),
            RefutationResult(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED, 0.1, 0.1),
            RefutationResult(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING, 0.1, 0.1),
            RefutationResult(RefutationTestType.DATA_SUBSET, RefutationStatus.PASSED, 0.1, 0.1),
            RefutationResult(RefutationTestType.BOOTSTRAP, RefutationStatus.PASSED, 0.1, 0.1),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.90)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.PROCEED

    def test_critical_set_comes_from_config(self):
        r = RefutationRunner(config={"random_common_cause": {"critical": False}})
        tests = [
            RefutationResult(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED, 0.1, 0.1),
            RefutationResult(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.FAILED, 0.1, 0.1),
            RefutationResult(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.PASSED, 0.1, 0.1),
        ]
        conf = r._calculate_confidence_score(tests)
        assert r._determine_gate_decision(tests, conf) == GateDecision.REVIEW  # 0.667, no critical failure
```

Ensure `RefutationResult` and `GateDecision` are imported at the top of the file (they already are for other classes; add if missing). In `tests/unit/test_causal_engine/test_refutation_runner_1419.py:62` change
`"sensitivity_e_value": {"enabled": sensitivity_enabled, "e_value_threshold": 2.0},` to
`"sensitivity_e_value": {"enabled": sensitivity_enabled},`.

- [ ] **Step 2: Run to verify red**

Run: `(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_refutation_runner.py -q -p no:cacheprovider -n 0 -k Sensitivity 2>&1 | tail -3)`
Expected: failures on `critical is False`, `unexpected keyword argument 'baseline_risk'`, `reading` KeyError.

- [ ] **Step 3: Change the runner**

(a) `DEFAULT_CONFIG` entry:

```python
        "sensitivity_e_value": {
            "enabled": True,
            # A READING, never a gate (spec 2026-09-10 §4.4): measured 2026-09-10, the
            # old 2.0/1.5 cutoffs BLOCKed 7 of 11 correctly recovered planted truths at
            # the live row cap and could not distinguish an omitted-confounder fit from
            # a correct one. The benchmark is the confounding this run measured.
            "critical": False,
        },
```

(b) Delete the `"e_value_min": {...}` entry from `PASS_THRESHOLDS`. Fix the placebo comment (#1994 option 1):

```python
        "placebo_p_value": {
            # PASSED at p >= 0.05, FAILED below. The 0.05–0.10 "warning band" this
            # comment used to name is unreachable as coded (pass is tested first,
            # #1994 option 1: documented, behaviour unchanged).
            "pass": 0.05,
            "warning": 0.10,
        },
```

(c) `run_all_tests` signature: after `outcome_std: Optional[float] = None,` add

```python
        baseline_risk: Optional[float] = None,
        naive_effect: Optional[float] = None,
        covariate_bias_factors: Optional[Dict[str, float]] = None,
```

and in its docstring, immediately after the `original_ci:` line, replace the one-line description with:

```
            original_ci: Confidence interval (lower, upper). This is the REPORTED
                interval from the estimation node (``ate_inference(X).conf_int_mean()``)
                and is the reference interval for every width comparison
                (``bootstrap_ci_ratio``) and for the sensitivity reading. Never derive
                a reference interval from the refutation node's RECONSTRUCTION of the
                estimator: its own interval is unusable (SE 4.9 measured against 0.034
                reported on the same pair, spec 2026-09-08 §2; issue #1989).
```

and after the `outcome_std:` paragraph add:

```
            baseline_risk: Control-arm outcome rate on the FULL estimation frame
                (binary treatment and outcome), the risk-ratio path's anchor.
            naive_effect: Unadjusted difference in means on the full frame; with
                the adjusted effect it gives the joint measured-confounding
                benchmark. ``None`` for a continuous treatment.
            covariate_bias_factors: Per-covariate bias factors of the backdoor set
                (``evalue.covariate_bias_factors``), the fallback benchmark.
                When all three are ``None`` and ``data``/``treatment``/``outcome``
                are present, the runner computes them from ``data`` (which may be
                the refutation subsample); caller-supplied values win.
```

(d) The sensitivity dispatch inside `run_all_tests` (the block starting `if self.config["sensitivity_e_value"]["enabled"]:`): after the `evalue_outcome_std` computation and before `test_result = self._run_test_with_tracing(...)`, add the fallback and pass the kwargs:

```python
                # Benchmark inputs: caller-supplied (full frame) win; otherwise derive
                # from the passthrough frame with the model's common causes.
                _baseline_risk, _naive, _factors = baseline_risk, naive_effect, covariate_bias_factors
                if (
                    _baseline_risk is None
                    and _naive is None
                    and _factors is None
                    and data is not None
                    and treatment is not None
                    and outcome is not None
                ):
                    try:
                        _covs: List[str] = []
                        getter = getattr(causal_model, "get_common_causes", None)
                        if callable(getter):
                            _covs = [str(c) for c in (getter() or [])]
                        _inputs = evalue.benchmark_inputs_from_frame(data, treatment, outcome, _covs)
                        _baseline_risk, _naive, _factors = (
                            _inputs.baseline_risk,
                            _inputs.naive_effect,
                            _inputs.covariate_bias_factors,
                        )
                    except Exception:  # noqa: BLE001 - no benchmark → unbenchmarked reading
                        _baseline_risk, _naive, _factors = None, None, None
                test_result = self._run_test_with_tracing(
                    test_name="sensitivity_e_value",
                    test_func=self._run_sensitivity_test,
                    opik=opik,
                    trace_id=trace_id,
                    estimate_id=estimate_id,
                    original_effect=original_effect,
                    original_ci=original_ci,
                    outcome_std=evalue_outcome_std,
                    randomized_design=randomized_design,
                    baseline_risk=_baseline_risk,
                    naive_effect=_naive,
                    covariate_bias_factors=_factors,
                    n_rows=(len(data) if data is not None else None),
                )
```

Add `from src.causal_engine import evalue` to the imports.

(e) Replace `_run_sensitivity_test` entirely:

```python
    def _run_sensitivity_test(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        outcome_std: Optional[float] = None,
        randomized_design: bool = False,
        baseline_risk: Optional[float] = None,
        naive_effect: Optional[float] = None,
        covariate_bias_factors: Optional[Dict[str, float]] = None,
        n_rows: Optional[int] = None,
    ) -> RefutationResult:
        """E-value sensitivity READING (spec 2026-09-10 §4.4).

        The E-value (VanderWeele & Ding 2017) is reported against the confounding
        this run measured — ``evalue.classify`` — and is never a gate: the test is
        non-critical and has no FAILED outcome. A CI that includes zero is a null
        finding (WARNING). ``randomized_design=True`` keeps today's SKIPPED /
        not-applicable behaviour with the numbers kept for information.
        """
        import time

        start_time = time.time()
        reading = evalue.classify(
            original_effect,
            original_ci,
            randomized=randomized_design,
            baseline_risk=baseline_risk,
            outcome_std=(
                outcome_std
                if outcome_std is not None and np.isfinite(outcome_std) and outcome_std > 0
                else None
            ),
            naive_effect=naive_effect,
            covariate_factors=covariate_bias_factors or {},
            n_rows=n_rows,
        )
        status = RefutationStatus(reading.status)
        details = reading.as_details()
        details.update(
            {
                # legacy keys consumers already read
                "e_value": reading.e_value_point,
                "standardized": reading.conversion == "standardized_difference"
                and outcome_std is not None,
                "outcome_std": outcome_std,
                "gate_applicable": not randomized_design,
            }
        )
        return RefutationResult(
            test_name=RefutationTestType.SENSITIVITY_E_VALUE,
            status=status,
            original_effect=original_effect,
            refuted_effect=original_effect,
            p_value=None,
            delta_percent=0.0,
            details=details,
            execution_time_ms=(time.time() - start_time) * 1000,
        )
```

(f) `_determine_gate_decision`: replace the hardcoded `critical_tests` set with

```python
        critical_tests = {
            RefutationTestType(name)
            for name, cfg in self.config.items()
            if isinstance(cfg, dict) and cfg.get("critical") and name in RefutationTestType._value2member_map_
        }
```

- [ ] **Step 4: Run the runner tests**

Run: `(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_1419.py tests/unit/test_causal_engine/test_refutation_runner_randomized.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py -q -p no:cacheprovider -n 0 2>&1 | tail -5)`
Expected: all pass. `test_refutation_runner_randomized.py` needs no change (SKIPPED path and equal numbers for randomized vs observational still hold).

- [ ] **Step 5: Write the band-enumeration test (red-first is moot here: it documents; write and run)**

```python
"""Every reachable refutation band under the 2026-09-10 rule (spec §6).

Sensitivity is non-critical with statuses {PASSED, WARNING, SKIPPED}; the other four
keep {PASSED, WARNING, FAILED, SKIPPED}. The enumerated table below is copied into
the lineage page's REVIEW-band callout (Task 10); if the arithmetic changes, both
this test and that callout must change together.
"""

from __future__ import annotations

import itertools

import pytest

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)

P, W, F, S = (
    RefutationStatus.PASSED,
    RefutationStatus.WARNING,
    RefutationStatus.FAILED,
    RefutationStatus.SKIPPED,
)
CRITICAL = (RefutationTestType.PLACEBO_TREATMENT, RefutationTestType.RANDOM_COMMON_CAUSE)
NONCRIT = (RefutationTestType.DATA_SUBSET, RefutationTestType.BOOTSTRAP)
SENS = RefutationTestType.SENSITIVITY_E_VALUE


def _suite(pl, rcc, sens, sub, boot):
    mk = lambda n, s: RefutationResult(n, s, 0.1, 0.1)  # noqa: E731
    return [mk(CRITICAL[0], pl), mk(CRITICAL[1], rcc), mk(SENS, sens), mk(NONCRIT[0], sub), mk(NONCRIT[1], boot)]


def _all_combos():
    for pl, rcc, sens, sub, boot in itertools.product([P, W, F, S], [P, W, F, S], [P, W, S], [P, W, F, S], [P, W, F, S]):
        yield pl, rcc, sens, sub, boot


def test_sensitivity_status_never_changes_a_proceed_when_the_other_tests_pass():
    r = RefutationRunner()
    for sens in (P, W, S):
        tests = _suite(P, P, sens, P, P)
        conf = r._calculate_confidence_score(tests)
        assert r._determine_gate_decision(tests, conf) == GateDecision.PROCEED, sens


def test_only_placebo_or_random_common_cause_can_block_on_their_own():
    r = RefutationRunner()
    for pl, rcc, sens, sub, boot in _all_combos():
        tests = _suite(pl, rcc, sens, sub, boot)
        conf = r._calculate_confidence_score(tests)
        gate = r._determine_gate_decision(tests, conf)
        if pl == F or rcc == F:
            assert gate == GateDecision.BLOCK
        else:
            assert gate == (GateDecision.PROCEED if conf >= 0.70 else GateDecision.REVIEW if conf >= 0.50 else GateDecision.BLOCK)


def test_reachable_confidence_values_without_a_critical_failure():
    """The exact set of confidence values a run can carry without a critical FAILED.
    REVIEW (0.50 <= c < 0.70) is reachable; BLOCK by confidence alone needs c < 0.50."""
    r = RefutationRunner()
    reachable = set()
    for pl, rcc, sens, sub, boot in _all_combos():
        if pl == F or rcc == F:
            continue
        tests = _suite(pl, rcc, sens, sub, boot)
        if all(t.status == S for t in tests):
            continue
        reachable.add(round(r._calculate_confidence_score(tests), 3))
    review = sorted(v for v in reachable if 0.50 <= v < 0.70)
    block = sorted(v for v in reachable if v < 0.50)
    assert review, "REVIEW must be reachable without a critical failure"
    assert 0.65 in review and 0.6 in review
    # Confidence-only BLOCK needs BOTH critical tests in WARNING and BOTH non-critical
    # tests FAILED (sensitivity PASSED/WARNING/SKIPPED): 0.40, 0.45 and 0.48. Pinned so
    # the lineage callout and this test move together.
    assert block == [0.4, 0.45, 0.48], block
    assert max(reachable) == 1.0


def test_null_finding_run_proceeds_when_everything_else_passes():
    """Owner decision 2026-09-10: a CI including zero is served, not blocked."""
    r = RefutationRunner()
    tests = _suite(P, P, W, P, P)  # sensitivity WARNING = null_finding or within
    conf = r._calculate_confidence_score(tests)
    assert conf == pytest.approx(0.90)
    assert r._determine_gate_decision(tests, conf) == GateDecision.PROCEED
```

Run: `(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_refutation_bands_enumeration.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: `4 passed`. The pinned BLOCK set is derived by hand: two critical WARNINGs (0.15 + 0.15) with both non-critical FAILED (0) give 0.30/0.75 = 0.40 when sensitivity is SKIPPED, 0.45/1.0 = 0.45 when it is WARNING, and (0.30 + 0.25)/1.0 = 0.55 (REVIEW) when it is PASSED; two critical WARNINGs + sensitivity SKIPPED + one non-critical FAILED and the other SKIPPED give 0.30/0.625 = 0.48. If the run prints a different set, the arithmetic in the runner changed: stop and report, never edit the pin to match.

- [ ] **Step 6: Lint, type-check, commit**

```bash
(cd $W && $PY -m ruff check src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_bands_enumeration.py tests/unit/test_causal_engine/test_refutation_runner_1419.py && $PY -m ruff format --check src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/test_refutation_bands_enumeration.py && $PY -m mypy --config-file pyproject.toml src/causal_engine/refutation_runner.py)
git -C $W branch --show-current
git -C $W add src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/
git -C $W commit -m "feat(refutation): sensitivity is a benchmarked reading, non-critical, never FAILED; bands enumerated (#1988 #1989 #1994)"
```

---

### Task 3: The YAML and its pin move with the runner

**Files:**
- Modify: `config/agent_config.yaml:655-668`
- Modify: `tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py` (`test_e_value_threshold_matches_the_pass_bar_not_the_warning_band`, `test_documentation_only_block_says_so`, `test_the_yaml_anchor_names_a_real_constant`)

- [ ] **Step 1: Rewrite the three residue assertions (red)**

Replace `test_e_value_threshold_matches_the_pass_bar_not_the_warning_band` with:

```python
    def test_the_e_value_cutoff_is_gone_from_runner_and_yaml(self):
        """2026-09-10 (spec sensitivity-gate-calibration §4.5): the sensitivity test
        is a benchmarked reading with no cutoff. Neither the runner nor the YAML may
        carry an ``e_value_threshold`` / ``e_value_min`` again."""
        pass_thresholds = self._runner_constant("PASS_THRESHOLDS")
        assert "e_value_min" not in pass_thresholds
        default_config = self._runner_constant("DEFAULT_CONFIG")
        assert default_config["sensitivity_e_value"]["critical"] is False
        assert "e_value_threshold" not in default_config["sensitivity_e_value"]
        block = self._validation_block(self._cfg())
        assert "e_value_threshold" not in block
        assert block["sensitivity_rule"] == "measured_confounding_benchmark"
```

In `test_documentation_only_block_says_so` change the line-finding expression to look for `sensitivity_rule:` instead of `e_value_threshold:`:

```python
        idx = next(i for i, line in enumerate(text) if line.strip().startswith("sensitivity_rule:"))
```

Replace `test_the_yaml_anchor_names_a_real_constant` with:

```python
    def test_the_yaml_anchor_names_a_real_constant(self):
        cfg = AGENT_CONFIG.read_text(encoding="utf-8")
        assert 'RefutationRunner.DEFAULT_CONFIG["sensitivity_e_value"]' in cfg
        assert "src/causal_engine/evalue.py" in cfg
        runner = (REPO_ROOT / "src" / "causal_engine" / "refutation_runner.py").read_text(encoding="utf-8")
        assert "DEFAULT_CONFIG" in runner and '"sensitivity_e_value"' in runner
        assert '"e_value_threshold"' not in runner, "the cutoff was removed on 2026-09-10; do not re-add it"
        assert (REPO_ROOT / "src" / "causal_engine" / "evalue.py").is_file()
```

- [ ] **Step 2: Run to verify red**

Run: `(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: 3 failures (`KeyError: 'sensitivity_rule'`, missing `evalue.py` mention).

- [ ] **Step 3: Rewrite the YAML block**

Replace lines 655–668 of `config/agent_config.yaml` with:

```yaml
    # DOCUMENTATION-ONLY (#1975). Nothing reads this block: the live values are
    # class constants on RefutationRunner (src/causal_engine/refutation_runner.py)
    # and the reading rules live in src/causal_engine/evalue.py. Since 2026-09-10
    # the sensitivity test carries NO cutoff: RefutationRunner.DEFAULT_CONFIG
    # ["sensitivity_e_value"] is {enabled, critical: false}; the E-value is read
    # against the confounding the run measured (spec
    # docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md §4.4).
    # min_refutation_pass_rate's only src/ hits are in src/ml/synthetic/, an
    # unrelated validator with its own Python default -- a name collision, not a
    # consumer of this key. The gate is confidence-weighted, not pass-rate based.
    validation:  # V4.1: Validation settings
      min_refutation_pass_rate: 0.6   # not consumed; the gate weights confidence
      require_sensitivity_analysis: true  # not consumed
      # not consumed; names the rule so a reader does not look for a number:
      # PASSED = point risk ratio above the measured-confounding benchmark,
      # WARNING = within it / null finding / unbenchmarked, SKIPPED = randomized.
      sensitivity_rule: measured_confounding_benchmark
      persist_to_causal_validations: true  # not consumed
```

- [ ] **Step 4: Run to verify green, commit**

Run: `(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: all pass.

```bash
git -C $W branch --show-current
git -C $W add config/agent_config.yaml tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py
git -C $W commit -m "docs(config): the sensitivity test has no cutoff; residue test pins its absence (#1975 follow-on)"
```

---

### Task 4: The refutation node computes the benchmark on the full frame and records the null caveat

**Files:**
- Modify: `src/agents/causal_impact/nodes/refutation.py` (imports ~37; after `outcome_std_full` ~1508; the `run_all_tests` call ~1608–1629; the result dict ~1747)
- Modify: `tests/unit/test_agents/test_causal_impact/test_refutation.py:343-382`
- Create: `tests/unit/test_agents/test_causal_impact/test_refutation_null_caveat.py`

- [ ] **Step 1: Re-base the two threshold tests on placebo and write the caveat test (red)**

In `test_refutation.py`, `test_blocked_estimate_fails_workflow`: replace the thresholds argument with
`thresholds={"placebo_p_value": {"pass": 1.01, "warning": 1.01}},  # p >= 1.01 impossible → FAILED → BLOCK`
and change the docstring to `"""A critical FAILED (placebo, forced by an impossible threshold) sets status failed."""`. In `test_custom_thresholds_passed_to_runner` replace `"e_value_min": {"pass": 3.0}` with `"placebo_p_value": {"pass": 0.10}` and the assertion with `assert node.runner.thresholds["placebo_p_value"]["pass"] == 0.10`.

New file `tests/unit/test_agents/test_causal_impact/test_refutation_null_caveat.py`:

```python
"""Spec 2026-09-10 §4.3/§4.6: the refutation node hands the runner FULL-frame
benchmark inputs and records a null finding in ``warnings`` exactly once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes import refutation as node_mod
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)


def _frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sev = rng.normal(5, 1, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * (sev - 5) - 0.3)))).astype(int)
    y = (rng.random(n) < 0.30 + 0.10 * t + 0.02 * (sev - 5)).astype(int)
    return pd.DataFrame({"treatment_arm": t, "treatment_initiated": y, "disease_severity": sev})


class TestBenchmarkInputs:
    def test_inputs_are_computed_on_the_full_frame_with_the_backdoor_set(self):
        frame = _frame()
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=frame,
            treatment="treatment_arm",
            outcome="treatment_initiated",
            estimation_result={"naive_ate": 0.123, "covariates_adjusted": ["disease_severity"],
                               "baseline_covariates_adjusted": ["noise"]},
        )
        p0 = frame.loc[frame.treatment_arm == 0, "treatment_initiated"].mean()
        assert inputs.baseline_risk == pytest.approx(p0)
        assert inputs.naive_effect == 0.123  # the estimation node's contrast wins
        assert set(inputs.covariate_bias_factors) == {"disease_severity"}  # efficiency controls excluded
        assert inputs.n_rows == len(frame)

    def test_missing_frame_yields_empty_inputs(self):
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=None, treatment="t", outcome="y", estimation_result={}
        )
        assert inputs.baseline_risk is None and inputs.naive_effect is None
        assert inputs.covariate_bias_factors == {}


class TestNullCaveat:
    def _suite(self, reading: str) -> RefutationSuite:
        sens = RefutationResult(
            RefutationTestType.SENSITIVITY_E_VALUE,
            RefutationStatus.WARNING,
            0.01,
            0.01,
            details={"reading": reading, "message": "The 95 % CI [-0.020, 0.040] includes zero at n = 600. "
                     "The estimate is reported as a null finding; no unmeasured confounder is needed to explain it."},
        )
        ok = [RefutationResult(n, RefutationStatus.PASSED, 0.01, 0.01) for n in
              (RefutationTestType.PLACEBO_TREATMENT, RefutationTestType.RANDOM_COMMON_CAUSE,
               RefutationTestType.DATA_SUBSET, RefutationTestType.BOOTSTRAP)]
        return RefutationSuite(passed=True, confidence_score=0.9, tests=[*ok, sens], gate_decision=GateDecision.PROCEED)

    def test_null_finding_message_is_returned_once_as_a_new_warning(self):
        warnings = node_mod._sensitivity_caveat_warnings(self._suite("null_finding"))
        assert len(warnings) == 1
        assert warnings[0].startswith("No detectable effect at this sample size: ")
        assert "includes zero" in warnings[0]

    def test_other_readings_add_no_warning(self):
        for reading in ("beyond_measured_confounding", "within_measured_confounding", "not_applicable_randomized"):
            assert node_mod._sensitivity_caveat_warnings(self._suite(reading)) == []
```

- [ ] **Step 2: Run to verify red**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_causal_impact/test_refutation_null_caveat.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: `AttributeError: module ... has no attribute '_sensitivity_benchmark_inputs'`.

- [ ] **Step 3: Implement in the node**

Add to the imports of `src/agents/causal_impact/nodes/refutation.py`:

```python
from src.causal_engine import evalue
```

Add two module-level helpers (place them after `_effective_reconstruction_common_causes`):

```python
def _sensitivity_benchmark_inputs(
    *,
    estimation_data: Any,
    treatment: str,
    outcome: str,
    estimation_result: Dict[str, Any],
) -> evalue.BenchmarkInputs:
    """FULL-frame inputs for the sensitivity reading (spec 2026-09-10 §4.3).

    Baseline risk, naive contrast and the backdoor set's covariate bias factors are
    computed on ``estimation_data`` (the full frame), never on the refutation
    subsample, so the benchmark describes the frame the reported effect came from.
    ``naive_ate`` from the estimation node wins; ``baseline_covariates_adjusted``
    (efficiency controls, #1188) are excluded from the factors. Fail-open to empty
    inputs: the runner then reads ``unbenchmarked`` rather than crashing the suite.
    """
    empty = evalue.BenchmarkInputs(baseline_risk=None, naive_effect=None)
    if estimation_data is None or not hasattr(estimation_data, "columns"):
        return empty
    if treatment not in estimation_data.columns or outcome not in estimation_data.columns:
        return empty
    covariates = [
        str(c)
        for c in (estimation_result.get("covariates_adjusted") or [])
        if c in estimation_data.columns
    ]
    try:
        return evalue.benchmark_inputs_from_frame(
            estimation_data,
            treatment,
            outcome,
            covariates,
            naive_effect=estimation_result.get("naive_ate"),
        )
    except Exception:  # noqa: BLE001 - benchmark is a reading aid, never a blocker
        logger.debug("sensitivity benchmark inputs unavailable", exc_info=True)
        return empty


_NULL_CAVEAT_PREFIX = "No detectable effect at this sample size: "


def _sensitivity_caveat_warnings(suite: RefutationSuite) -> List[str]:
    """The null-finding caveat, once, for the ``warnings`` accumulator (spec §4.6)."""
    for test in suite.tests:
        if test.test_name != RefutationTestType.SENSITIVITY_E_VALUE:
            continue
        if test.details.get("reading") == evalue.READING_NULL:
            return [_NULL_CAVEAT_PREFIX + str(test.details.get("message", "")).strip()]
    return []
```

Right after the `outcome_std_full` block (the `try/except` that ends `outcome_std_full = None`), add:

```python
            benchmark_inputs = _sensitivity_benchmark_inputs(
                estimation_data=estimation_data,
                treatment=treatment,
                outcome=outcome,
                estimation_result=estimation_result,
            )
```

In the `run_all_tests` call, after `outcome_std=outcome_std_full,` add:

```python
                    baseline_risk=benchmark_inputs.baseline_risk,
                    naive_effect=benchmark_inputs.naive_effect,
                    covariate_bias_factors=benchmark_inputs.covariate_bias_factors,
```

In the result dict, after `"needs_review": suite.needs_review,` add:

```python
                # 2026-09-10: a CI including zero is served as a null finding; the
                # caveat rides the warnings accumulator so the API record and the
                # drill-down show it (new entries only — the channel is additive).
                "warnings": _sensitivity_caveat_warnings(suite),
```

Check the `estimation_result` variable name at the call site (it is read near line 1380 as `estimation_result = state.get("estimation_result")`); use that name. Also confirm `Any`, `Dict`, `List` are imported from `typing` in the node (they are).

- [ ] **Step 4: Run node tests**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_causal_impact/test_refutation_null_caveat.py tests/unit/test_agents/test_causal_impact/test_refutation.py tests/unit/test_agents/test_causal_impact/test_refutation_failopen_p3.py tests/unit/test_agents/test_causal_impact/test_latent_warning_policy.py -q -p no:cacheprovider -n 0 2>&1 | tail -5)`
Expected: all pass. `test_latent_warning_policy.py::test_every_state_spread_in_nodes_uses_spread_safe` must stay green (the node keeps `**spread_safe(state)`; `warnings` is returned as NEW entries only).

- [ ] **Step 5: Lint, type-check, commit**

```bash
(cd $W && $PY -m ruff check src/agents/causal_impact/nodes/refutation.py tests/unit/test_agents/test_causal_impact/test_refutation_null_caveat.py tests/unit/test_agents/test_causal_impact/test_refutation.py && $PY -m ruff format --check src/agents/causal_impact/nodes/refutation.py tests/unit/test_agents/test_causal_impact/test_refutation_null_caveat.py && $PY -m mypy --config-file pyproject.toml src/agents/causal_impact/nodes/refutation.py)
git -C $W branch --show-current
git -C $W add src/agents/causal_impact/nodes/refutation.py tests/unit/test_agents/test_causal_impact/test_refutation_null_caveat.py tests/unit/test_agents/test_causal_impact/test_refutation.py
git -C $W commit -m "feat(refutation-node): full-frame benchmark inputs for the sensitivity reading; null finding recorded in warnings"
```

---

### Task 5: The sensitivity node speaks the same numbers

**Files:**
- Modify: `src/agents/causal_impact/nodes/sensitivity.py` (whole `execute`; delete `_calculate_e_value`, `_interpret_e_value`)
- Modify: `src/agents/causal_impact/state.py:165-173` (`SensitivityAnalysis`)
- Modify: `tests/unit/test_agents/test_causal_impact/test_sensitivity.py`, `tests/unit/test_agents/test_causal_impact/test_sensitivity_randomized.py`, `tests/unit/test_causal_engine/test_evalue_standardization_p4.py`

- [ ] **Step 1: Rewrite the node tests (red)**

In `test_sensitivity.py`: keep `test_calculate_e_value`, `test_e_value_for_ci`, `test_interpretation_text`, `test_latency_measurement`, `test_error_handling_missing_estimation`. Replace `test_robustness_classification`, `test_confounder_strength_classification`, `test_e_value_ci_collapses_to_one_when_ci_straddles_null`, `test_e_value_ci_unchanged_when_ci_does_not_straddle_null`, and delete the classes `TestEValueCalculation` and `TestEValueInterpretation` (their subject moved to `test_evalue.py`). New tests inside `TestSensitivityNode`:

```python
    def _frame(self, n: int = 800, seed: int = 5):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        sev = rng.normal(5, 1, n)
        t = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * (sev - 5) - 0.3)))).astype(int)
        y = (rng.random(n) < 0.30 + 0.15 * t + 0.02 * (sev - 5)).astype(int)
        return pd.DataFrame({"treatment_arm": t, "treatment_initiated": y, "disease_severity": sev})

    def _state_with_frame(self, ate, lo, hi, naive=None, **extra) -> CausalImpactState:
        est = self._create_test_estimation(ate=ate)
        est.update({"ate_ci_lower": lo, "ate_ci_upper": hi, "covariates_adjusted": ["disease_severity"]})
        if naive is not None:
            est["naive_ate"] = naive
        return {
            "query": "q", "query_id": "q-1", "status": "pending",
            "estimation_result": est, "estimation_data": self._frame(),
            "treatment_var": "treatment_arm", "outcome_var": "treatment_initiated", **extra,
        }

    @pytest.mark.asyncio
    async def test_beyond_reading_is_robust(self):
        result = await SensitivityNode().execute(self._state_with_frame(0.15, 0.08, 0.22, naive=0.20))
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "beyond_measured_confounding"
        assert sens["robust_to_confounding"] is True
        assert sens["headline"] == "Robust to confounding at measured strength"
        assert sens["unmeasured_confounder_strength"] == "beyond_measured_confounding"
        assert sens["benchmark_basis"] == "joint_naive_vs_adjusted"
        assert sens["conversion"] == "risk_ratio"
        assert "stronger than all measured confounding" in sens["interpretation"]

    @pytest.mark.asyncio
    async def test_within_reading_is_not_robust(self):
        result = await SensitivityNode().execute(self._state_with_frame(0.03, 0.01, 0.05, naive=0.12))
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "within_measured_confounding"
        assert sens["robust_to_confounding"] is False

    @pytest.mark.asyncio
    async def test_null_crossing_ci_is_a_null_finding(self):
        result = await SensitivityNode().execute(self._state_with_frame(0.05, -0.02, 0.12, naive=0.08))
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "null_finding"
        assert sens["e_value_ci"] == 1.0
        assert sens["robust_to_confounding"] is False
        assert "includes zero" in sens["interpretation"]

    @pytest.mark.asyncio
    async def test_point_e_value_matches_the_shared_module_on_the_smd_path(self):
        """No frame → SMD path with the 0.91 factor: the node and the runner agree."""
        from src.causal_engine import evalue

        result = await SensitivityNode().execute({
            "query": "q", "query_id": "q-2", "status": "pending",
            "estimation_result": self._create_test_estimation(ate=0.5),
        })
        sens = result["sensitivity_analysis"]
        assert sens["e_value"] == pytest.approx(evalue.e_value_from_rr(evalue.rr_from_smd(0.5)))
        assert sens["reading"] == "unbenchmarked"
        assert sens["robust_to_confounding"] is False
```

In `test_sensitivity_randomized.py` the randomized assertions stay (`unmeasured_confounder_strength == "not_applicable_randomized"`, `robust_to_confounding is True`); the observational control at line ~69 asserts `robust_to_confounding is False` — keep it (an unbenchmarked or within reading is not robust). Read the file first; change only assertions that name `weak`/`moderate`/`strong`.

In `tests/unit/test_causal_engine/test_evalue_standardization_p4.py`, class `TestAgentEngineStandardization`: replace the two tests that call `node._calculate_e_value` with

```python
class TestAgentEngineStandardization:
    def test_node_delegates_to_the_shared_module(self):
        """Since 2026-09-10 the node has no private E-value; the runner and the node
        share ``evalue`` so the same run reports the same number (0.91 factor)."""
        from src.causal_engine import evalue

        assert not hasattr(SensitivityNode, "_calculate_e_value")
        e1 = evalue.e_value_from_rr(evalue.rr_from_smd(2.0 / 1.0))
        e2 = evalue.e_value_from_rr(evalue.rr_from_smd(2000.0 / 1000.0))
        assert e1 == pytest.approx(e2, rel=1e-6)
```

- [ ] **Step 2: Run to verify red**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_causal_impact/test_sensitivity.py tests/unit/test_causal_engine/test_evalue_standardization_p4.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: failures on missing `reading` key and on `_calculate_e_value` still existing.

- [ ] **Step 3: Rewrite the node and extend the state**

`src/agents/causal_impact/state.py` — replace the `SensitivityAnalysis` TypedDict:

```python
class SensitivityAnalysis(TypedDict, total=False):
    """Sensitivity to unmeasured confounding as a benchmarked READING (spec 2026-09-10 §4.4)."""

    e_value: float  # E-value for the point estimate
    e_value_ci: float  # E-value at the CI bound (1.0 when the CI includes zero)
    interpretation: str  # the leader-facing sentence (= reading message)
    robust_to_confounding: bool  # True only for reading == beyond_measured_confounding
    # Kept for consumers; since 2026-09-10 its value IS the reading name below.
    unmeasured_confounder_strength: str
    reading: str  # beyond_measured_confounding / within_measured_confounding / null_finding / unbenchmarked / not_applicable_randomized
    headline: str
    rr_point: float
    rr_ci: float
    benchmark: Optional[float]
    benchmark_basis: str  # joint_naive_vs_adjusted / strongest_covariate / none_measured
    conversion: str  # risk_ratio / standardized_difference
```

(`Optional` is already imported in `state.py`; verify.)

`src/agents/causal_impact/nodes/sensitivity.py` — full new body:

```python
"""Sensitivity Analysis Node - E-value READING for unmeasured confounding.

Spec docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md §4.6:
the node delegates every number and word to ``src.causal_engine.evalue`` so the
agent narrative, the refutation runner and the chat tool cannot disagree on a run.
"""

import time
from typing import Dict, Optional

import numpy as np

from src.agents.causal_impact.state import CausalImpactState, SensitivityAnalysis, spread_safe
from src.causal_engine import evalue


class SensitivityNode:
    """Performs sensitivity analysis for unmeasured confounding.

    Performance target: <5s
    Type: Standard (computation-light)
    """

    def __init__(self):
        """Initialize sensitivity node."""
        pass

    async def execute(self, state: CausalImpactState) -> Dict:
        """Compute the sensitivity reading from the estimation result and the full frame."""
        start_time = time.time()

        try:
            estimation_result = state.get("estimation_result")
            if not estimation_result:
                raise ValueError("Estimation result not found in state")

            ate = float(estimation_result["ate"])
            ci = (float(estimation_result["ate_ci_lower"]), float(estimation_result["ate_ci_upper"]))
            outcome_std = self._resolve_outcome_std(state)
            inputs = self._benchmark_inputs(state, estimation_result)

            reading = evalue.classify(
                ate,
                ci,
                randomized=bool(state.get("randomized_design")),
                baseline_risk=inputs.baseline_risk,
                outcome_std=outcome_std,
                naive_effect=inputs.naive_effect,
                covariate_factors=inputs.covariate_bias_factors,
                n_rows=inputs.n_rows or estimation_result.get("sample_size"),
            )
            randomized = reading.reading == evalue.READING_RANDOMIZED

            sensitivity_analysis: SensitivityAnalysis = {
                "e_value": reading.e_value_point,
                "e_value_ci": reading.e_value_ci,
                "interpretation": (
                    "Randomized design: treatment assignment is exogenous by construction, so "
                    "unmeasured confounding of assignment is excluded by design. E-value "
                    f"{reading.e_value_point:.2f} (CI bound {reading.e_value_ci:.2f}) is reported "
                    "for information only and does not indicate a validity risk."
                    if randomized
                    else reading.message
                ),
                # Randomized designs are robust to confounding of assignment by design.
                "robust_to_confounding": randomized or reading.reading == evalue.READING_BEYOND,
                "unmeasured_confounder_strength": reading.reading,
                "reading": reading.reading,
                "headline": reading.headline,
                "rr_point": reading.rr_point,
                "rr_ci": reading.rr_ci,
                "benchmark": reading.benchmark,
                "benchmark_basis": reading.benchmark_basis,
                "conversion": reading.conversion,
            }

            return {
                **spread_safe(state),
                "sensitivity_analysis": sensitivity_analysis,
                "sensitivity_latency_ms": (time.time() - start_time) * 1000,
                "current_phase": "interpreting",
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                **spread_safe(state),
                "sensitivity_error": str(e),
                "sensitivity_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Sensitivity analysis failed: {e}",
            }

    def _resolve_outcome_std(self, state: CausalImpactState) -> Optional[float]:
        """Outcome SD (σ_Y) from the estimation-data passthrough; None when unavailable."""
        data = state.get("estimation_data")
        outcome_var = state.get("outcome_var")
        if data is None or not outcome_var:
            return None
        try:
            if hasattr(data, "columns") and outcome_var in data.columns:
                sd = float(np.std(np.asarray(data[outcome_var], dtype=float)))
                return sd if np.isfinite(sd) and sd > 0 else None
        except Exception:  # noqa: BLE001 - non-numeric / missing → no standardization
            return None
        return None

    def _benchmark_inputs(self, state: CausalImpactState, estimation_result: Dict) -> evalue.BenchmarkInputs:
        """Same full-frame inputs the refutation node computes (spec §4.3)."""
        empty = evalue.BenchmarkInputs(
            baseline_risk=None, naive_effect=estimation_result.get("naive_ate")
        )
        data = state.get("estimation_data")
        treatment, outcome = state.get("treatment_var"), state.get("outcome_var")
        if data is None or not hasattr(data, "columns") or not treatment or not outcome:
            return empty
        if treatment not in data.columns or outcome not in data.columns:
            return empty
        covariates = [
            str(c) for c in (estimation_result.get("covariates_adjusted") or []) if c in data.columns
        ]
        try:
            return evalue.benchmark_inputs_from_frame(
                data, treatment, outcome, covariates, naive_effect=estimation_result.get("naive_ate")
            )
        except Exception:  # noqa: BLE001 - reading aid, never a blocker
            return empty


async def analyze_sensitivity(state: CausalImpactState) -> Dict:
    """LangGraph node entrypoint."""
    node = SensitivityNode()
    return await node.execute(state)
```

Check the existing module tail (line ~227 `async def analyze_sensitivity`) and keep its exact signature/docstring if it differs.

- [ ] **Step 4: Run the node tests**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_causal_impact/test_sensitivity.py tests/unit/test_agents/test_causal_impact/test_sensitivity_randomized.py tests/unit/test_agents/test_causal_impact/test_sensitivity_failopen_p_mfo3.py tests/unit/test_causal_engine/test_evalue_standardization_p4.py -q -p no:cacheprovider -n 0 2>&1 | tail -5)`
Expected: all pass. If `test_e_value_for_ci` fails because `e_value_ci > e_value` on the SMD path, the CI bound (ate − 0.1) is always nearer the null than ate, so `e_value_ci <= e_value` holds; investigate before changing the test.

- [ ] **Step 5: Lint, type-check, commit**

```bash
(cd $W && $PY -m ruff check src/agents/causal_impact/nodes/sensitivity.py src/agents/causal_impact/state.py tests/unit/test_agents/test_causal_impact/test_sensitivity.py tests/unit/test_causal_engine/test_evalue_standardization_p4.py && $PY -m ruff format --check src/agents/causal_impact/nodes/sensitivity.py && $PY -m mypy --config-file pyproject.toml src/agents/causal_impact/nodes/sensitivity.py src/agents/causal_impact/state.py)
git -C $W branch --show-current
git -C $W add src/agents/causal_impact/nodes/sensitivity.py src/agents/causal_impact/state.py tests/unit/test_agents/test_causal_impact/test_sensitivity.py tests/unit/test_agents/test_causal_impact/test_sensitivity_randomized.py tests/unit/test_causal_engine/test_evalue_standardization_p4.py
git -C $W commit -m "feat(sensitivity-node): delegate to the shared E-value module; state carries the reading"
```

---

### Task 6: The interpretation node narrates the reading

**Files:**
- Modify: `src/agents/causal_impact/nodes/interpretation.py:378-435` (robustness line) and `:474-482` (key finding)
- Modify: `tests/unit/test_agents/test_causal_impact/test_interpretation.py`

- [ ] **Step 1: Add narrative tests (red)**

Append to `tests/unit/test_agents/test_causal_impact/test_interpretation.py` (inside the module, using the file's existing state-builder helper — read the file's `_make_state`/fixture name first and use it):

```python
class TestReadingNarrative:
    """Spec 2026-09-10 §4.6: the robustness sentence comes from the reading."""

    @pytest.mark.asyncio
    async def test_beyond_reading_leads_with_the_headline_and_benchmark(self, base_state):
        base_state["sensitivity_analysis"] = {
            "e_value": 2.4, "e_value_ci": 1.9, "robust_to_confounding": True,
            "reading": "beyond_measured_confounding",
            "headline": "Robust to confounding at measured strength",
            "interpretation": "Explaining this effect away would need an unmeasured confounder with a risk ratio of at least 2.40 with both treatment and outcome (1.90 at the CI bound), stronger than all measured confounding combined (1.21, the confounding the adjustment removed, naive vs adjusted).",
            "benchmark": 1.21, "benchmark_basis": "joint_naive_vs_adjusted",
        }
        result = await InterpretationNode().execute(base_state)
        narrative = result["interpretation"]["narrative"]
        assert "Robust to confounding at measured strength" in narrative
        assert "1.21" in narrative
        assert "moderate robustness" not in narrative and "weak robustness" not in narrative

    @pytest.mark.asyncio
    async def test_null_finding_is_not_called_robust(self, base_state):
        base_state["estimation_result"]["statistical_significance"] = False
        base_state["sensitivity_analysis"] = {
            "e_value": 1.3, "e_value_ci": 1.0, "robust_to_confounding": False,
            "reading": "null_finding", "headline": "No detectable effect at this sample size",
            "interpretation": "The 95 % CI [-0.020, 0.120] includes zero at n = 1500. The estimate is reported as a null finding; no unmeasured confounder is needed to explain it.",
            "benchmark": 1.1, "benchmark_basis": "joint_naive_vs_adjusted",
        }
        result = await InterpretationNode().execute(base_state)
        narrative = result["interpretation"]["narrative"]
        assert "No detectable effect at this sample size" in narrative
        assert "robust to unmeasured confounding" not in narrative.lower()
        assert result["interpretation"]["confidence"] == "low"
```

If the file has no `base_state` fixture, build the state inline from the file's existing `_create_full_state()`-style helper (read lines 60–110 first and mirror it).

- [ ] **Step 2: Run to verify red**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_causal_impact/test_interpretation.py -q -p no:cacheprovider -n 0 -k ReadingNarrative 2>&1 | tail -3)`
Expected: 2 failures (headline absent from narrative).

- [ ] **Step 3: Replace the robustness sentence and the key finding**

Replace the block from `if sensitivity_failed:` through the second `robustness_line += (...)` (lines ~416–435) with:

```python
        if sensitivity_failed:
            # M-fo3: do NOT cite the defaulted E-value of 1.00 as a real result.
            robustness_line += (
                "The sensitivity analysis (E-value) could not be completed, so "
                "robustness to unmeasured confounding is UNVERIFIED; do not rely on "
                "any reported E-value."
            )
        else:
            # 2026-09-10: the reading's headline first, then its one-sentence message
            # with the numbers (spec §4.4). No fixed E-value bands.
            headline = sensitivity_analysis.get("headline") or evalue.HEADLINES.get(
                str(sensitivity_analysis.get("reading", "")), ""
            )
            message = sensitivity_analysis.get("interpretation") or ""
            robustness_line += f"{headline}. {message}"  # every reading message ends with a period
```

Add `from src.causal_engine import evalue` to the imports. Replace the key-finding entry:

```python
            (
                "E-value: unavailable (sensitivity analysis failed)"
                if sensitivity_failed
                else f"E-value: {e_value:.2f} — {sensitivity_analysis.get('headline', '')}".rstrip(" —")
            ),
```

- [ ] **Step 4: Run the interpretation tests**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_causal_impact/test_interpretation.py tests/unit/test_agents/test_causal_impact/test_interpretation_randomized.py tests/unit/test_agents/test_causal_impact/test_interpretation_failopen_p11.py tests/unit/test_agents/test_causal_impact/test_interpretation_clinical_context.py tests/unit/test_agents/test_causal_impact/test_latent_warning_policy.py -q -p no:cacheprovider -n 0 2>&1 | tail -5)`
Expected: all pass. Existing tests that assert `"E-value" in narrative` still hold through the key findings and the interpretation sentence; a test asserting the words "moderate robustness" must be updated to the headline (read the failure, change the assertion to the new sentence, never the code).

- [ ] **Step 5: Lint, type-check, commit**

```bash
(cd $W && $PY -m ruff check src/agents/causal_impact/nodes/interpretation.py tests/unit/test_agents/test_causal_impact/test_interpretation.py && $PY -m ruff format --check src/agents/causal_impact/nodes/interpretation.py && $PY -m mypy --config-file pyproject.toml src/agents/causal_impact/nodes/interpretation.py)
git -C $W branch --show-current
git -C $W add src/agents/causal_impact/nodes/interpretation.py tests/unit/test_agents/test_causal_impact/test_interpretation.py
git -C $W commit -m "feat(interpretation): narrate the sensitivity reading, no fixed E-value bands"
```

---

### Task 7: The chat tool uses the shared module

**Files:**
- Modify: `src/agents/tool_composer/tool_registrations.py:1280-1360` (`_e_value_from_rr`, `sensitivity_analyzer`)
- Modify: `src/agents/tool_composer/tool_registry.py:392-394` (description only)
- Modify: `tests/unit/test_agents/test_tool_composer/test_tools_fail_closed.py:26-53`

- [ ] **Step 1: Rewrite the tool tests (red)**

Replace the two `sensitivity_analyzer` tests with:

```python
# ---------------------------------------------------------------------------
# Task 1 — sensitivity_analyzer: shared E-value module, benchmarked reading (2026-09-10)
# ---------------------------------------------------------------------------
def test_sensitivity_analyzer_matches_the_shared_module_and_reads_unbenchmarked_without_a_naive():
    from src.causal_engine import evalue

    out = tr.sensitivity_analyzer(ate=0.5, ci_lower=0.1)
    assert out["e_value_point"] == pytest.approx(evalue.e_value_from_rr(evalue.rr_from_smd(0.5)), rel=1e-9)
    assert out["e_value_ci"] == pytest.approx(evalue.e_value_from_rr(evalue.rr_from_smd(0.1)), rel=1e-9)
    assert out["reading"] == "unbenchmarked"
    assert "no universal" in out["interpretation"].lower()
    assert "robustness" not in out  # the weak/moderate/strong verdict is gone

    out_big = tr.sensitivity_analyzer(ate=1.0, ci_lower=0.2)
    assert out_big["e_value_point"] > out["e_value_point"]

    out_cross = tr.sensitivity_analyzer(ate=0.5, ci_lower=0.0)
    assert out_cross["e_value_ci"] == pytest.approx(1.0, abs=1e-12)


def test_sensitivity_analyzer_reads_beyond_or_within_with_a_naive_contrast_and_baseline_risk():
    beyond = tr.sensitivity_analyzer(ate=0.15, ci_lower=0.08, ci_upper=0.22, baseline_risk=0.30, naive_ate=0.20)
    assert beyond["reading"] == "beyond_measured_confounding"
    assert beyond["headline"] == "Robust to confounding at measured strength"
    assert beyond["benchmark"] == pytest.approx((0.50 / 0.30) / (0.45 / 0.30))
    within = tr.sensitivity_analyzer(ate=0.03, ci_lower=0.01, ci_upper=0.05, baseline_risk=0.30, naive_ate=0.10)
    assert within["reading"] == "within_measured_confounding"
    null = tr.sensitivity_analyzer(ate=0.05, ci_lower=-0.02, ci_upper=0.12, baseline_risk=0.30, naive_ate=0.10)
    assert null["reading"] == "null_finding" and null["e_value_ci"] == 1.0


def test_sensitivity_analyzer_fail_closes_on_non_finite():
    with pytest.raises(RuntimeError):
        tr.sensitivity_analyzer(ate=float("nan"), ci_lower=0.1)
    with pytest.raises(RuntimeError):
        tr.sensitivity_analyzer(ate=0.5, ci_lower=float("inf"))
```

- [ ] **Step 2: Run to verify red**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_tool_composer/test_tools_fail_closed.py -q -p no:cacheprovider -n 0 -k sensitivity 2>&1 | tail -3)`
Expected: `KeyError: 'reading'` / unexpected keyword `ci_upper`.

- [ ] **Step 3: Rewrite the tool**

Delete `_e_value_from_rr` (lines 1280–1293) and replace the `sensitivity_analyzer` registration + function with:

```python
@composable_tool(
    name="sensitivity_analyzer",
    description=(
        "Compute VanderWeele-Ding E-values and, when a naive contrast is given, the "
        "measured-confounding reading (beyond / within / null finding) the refutation "
        "gate uses"
    ),
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {"name": "ate", "type": "float", "description": "Estimated average treatment effect"},
        {"name": "ci_lower", "type": "float", "description": "Lower confidence bound"},
        {"name": "ci_upper", "type": "float", "description": "Upper confidence bound (optional; defaults to ate + (ate - ci_lower))"},
        {"name": "baseline_risk", "type": "float", "description": "Control-arm outcome rate for a binary outcome (optional; enables the risk-ratio path)"},
        {"name": "naive_ate", "type": "float", "description": "Unadjusted difference in means (optional; enables the measured-confounding benchmark)"},
    ],
    output_schema="SensitivityReport",
    avg_execution_ms=1500,
)
def sensitivity_analyzer(
    ate: float,
    ci_lower: float,
    ci_upper: Optional[float] = None,
    baseline_risk: Optional[float] = None,
    naive_ate: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """E-values and the sensitivity READING from the shared ``evalue`` module.

    Spec docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md §4.7.
    Without ``baseline_risk`` the inputs are taken on the standardized-mean-difference
    scale (``RR = exp(0.91*d)``). Without ``naive_ate`` no benchmark exists and the
    reading is ``unbenchmarked``: the E-value is reported with the statement that no
    universal threshold exists. Refuses non-finite inputs (anti-mocking: never a
    fabricated E-value).
    """
    for name, value in (("ate", ate), ("ci_lower", ci_lower), ("ci_upper", ci_upper),
                        ("baseline_risk", baseline_risk), ("naive_ate", naive_ate)):
        if value is not None and not math.isfinite(float(value)):
            raise ToolRefusalError(
                f"sensitivity_analyzer requires finite inputs; got {name}={value!r}. Refusing to "
                "fabricate an E-value — per anti-mocking discipline non-finite inputs surface as "
                "a structured error."
            )
    hi = float(ci_upper) if ci_upper is not None else float(ate) + (float(ate) - float(ci_lower))
    reading = evalue.classify(
        float(ate),
        (float(ci_lower), hi),
        randomized=False,
        baseline_risk=baseline_risk,
        outcome_std=None,
        naive_effect=naive_ate,
        covariate_factors={},
        n_rows=None,
    )
    interpretation = reading.message
    if reading.reading == evalue.READING_UNBENCHMARKED:
        interpretation = (
            f"An unobserved confounder would need to be associated with both treatment and "
            f"outcome by a risk ratio of at least {reading.e_value_point:.2f} (and the CI bound "
            f"by {reading.e_value_ci:.2f}) to explain away the observed effect. There is no "
            "universal E-value threshold: benchmark it against the confounding the measured "
            "covariates carried (pass naive_ate and baseline_risk to get that reading)."
        )
    return {
        "e_value_point": reading.e_value_point,
        "e_value_ci": reading.e_value_ci,
        "reading": reading.reading,
        "headline": reading.headline,
        "benchmark": reading.benchmark,
        "benchmark_basis": reading.benchmark_basis,
        "interpretation": interpretation,
    }
```

Add `from src.causal_engine import evalue` to the module imports and make sure `Optional` is imported from `typing` (check the header). In `tool_registry.py:394` change the description to `"Computes E-values and the measured-confounding sensitivity reading for causal estimates"`. Leave its `input_schema`/`output_schema` untouched (the mismatch with the function is filed as an issue in Task 12).

- [ ] **Step 4: Run, lint, type-check, commit**

Run: `(cd $W && $PY -m pytest tests/unit/test_agents/test_tool_composer/test_tools_fail_closed.py -q -p no:cacheprovider -n 0 2>&1 | tail -3)`
Expected: all pass.

```bash
(cd $W && $PY -m ruff check src/agents/tool_composer/tool_registrations.py src/agents/tool_composer/tool_registry.py tests/unit/test_agents/test_tool_composer/test_tools_fail_closed.py && $PY -m ruff format --check src/agents/tool_composer/tool_registrations.py && $PY -m mypy --config-file pyproject.toml src/agents/tool_composer/tool_registrations.py)
git -C $W branch --show-current
git -C $W add src/agents/tool_composer/tool_registrations.py src/agents/tool_composer/tool_registry.py tests/unit/test_agents/test_tool_composer/test_tools_fail_closed.py
git -C $W commit -m "feat(tool-composer): sensitivity_analyzer on the shared E-value module; reading replaces the weak/moderate/strong verdict"
```

---

### Task 8: Calibration test on planted truth (heavy lane)

**Files:**
- Create: `tests/unit/test_causal_engine/test_sensitivity_calibration.py`

- [ ] **Step 1: Check memory**

Run: `free -m | sed -n 2p`. Continue only if "available" is above 1500 MiB (the test peaks at ~400 MiB).

- [ ] **Step 2: Write the test**

```python
"""Calibration of the sensitivity READING against the DGP's planted truth (spec §6).

Measured 2026-09-10 (LinearDML, production RF nuisances, seed-21 Remibrutinib frame,
n=1500 = the live row cap): every one of the 11 planted true effects reads
``beyond_measured_confounding``; the null pairs read ``null_finding``; and the
omitted-confounder refits are NOT distinguishable from the correct fits — the
E-value cannot detect confounding, which is why it is a reading and not a gate.
These pins stop anyone re-adding a cutoff believing it would catch a confounder.
~21 s, ~400 MiB.
"""

from __future__ import annotations

import numpy as np
import pytest
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.causal_engine import evalue
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

pytestmark = pytest.mark.heavy_ml

N_ROWS = 1500  # the live estimation row cap (routes/causal.py ``limit=1500``)
# Arm -> outcome pairs with a ZERO planted effect at n=1500 (spec §2.3): the arm
# does not target the outcome and the outcome is not downstream of it.
NULL_PAIRS = [
    ("treatment_arm", "persistent_180d"),
    ("copay_support", "treatment_initiated"),
    ("psp_enrolled", "treatment_initiated"),
    ("rep_detailing_high", "adherent_180d"),
    ("trigger_accepted", "adherent_180d"),
]


def _fit(df, treatment, outcome, covariates):
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = df[covariates].to_numpy(dtype=float)
    m = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42),
        discrete_treatment=True,
        random_state=42,
    )
    m.fit(Y, T, X=X, W=None)
    inf = m.ate_inference(X)
    lo, hi = (float(v) for v in inf.conf_int_mean())
    return float(inf.mean_point), (lo, hi)


def _reading(df, treatment, outcome, covariates):
    ate, ci = _fit(df, treatment, outcome, covariates)
    inputs = evalue.benchmark_inputs_from_frame(df, treatment, outcome, covariates)
    return evalue.classify(
        ate,
        ci,
        randomized=False,
        baseline_risk=inputs.baseline_risk,
        outcome_std=float(df[outcome].std()),
        naive_effect=inputs.naive_effect,
        covariate_factors=inputs.covariate_bias_factors,
        n_rows=len(df),
    )


@pytest.fixture(scope="module")
def frame():
    cfg = GeneratorConfig(seed=21, n_records=N_ROWS, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS)
    return PatientGenerator(cfg).generate()


def _planted_pairs(frame):
    truth = frame.attrs["true_ate_by_arm"]
    return [
        (arm, outcome, list(ARM_REGISTRY[arm].confounders))
        for arm, outs in truth.items()
        if arm in ARM_REGISTRY and arm in frame.columns
        for outcome in outs
        if outcome in frame.columns
    ]


def test_every_planted_truth_reads_beyond_measured_confounding(frame):
    pairs = _planted_pairs(frame)
    assert len(pairs) == 11, [p[:2] for p in pairs]
    readings = {(a, o): _reading(frame, a, o, covs) for a, o, covs in pairs}
    not_beyond = {k: r.reading for k, r in readings.items() if r.reading != evalue.READING_BEYOND}
    assert not not_beyond, f"planted truths not served as robust: {not_beyond}"
    assert all(r.status == "passed" for r in readings.values())


def test_null_pairs_read_null_finding(frame):
    readings = {(a, o): _reading(frame, a, o, list(ARM_REGISTRY[a].confounders)).reading for a, o in NULL_PAIRS}
    assert all(r == evalue.READING_NULL for r in readings.values()), readings


def test_known_limit_omitted_confounder_is_not_distinguishable_by_e_value(frame):
    """Pinned LIMIT, not a goal: dropping the strongest declared confounder leaves the
    reading unchanged on at least 9 of 11 pairs. A cutoff gate could never catch it."""
    same = 0
    pairs = _planted_pairs(frame)
    for arm, outcome, covs in pairs:
        strongest = max(ARM_REGISTRY[arm].confounders, key=lambda c: abs(ARM_REGISTRY[arm].confounders[c]))
        reduced = [c for c in covs if c != strongest] or covs
        if _reading(frame, arm, outcome, covs).reading == _reading(frame, arm, outcome, reduced).reading:
            same += 1
    assert same >= 9, f"only {same}/11 pairs read the same with a confounder omitted"


def test_sensitivity_never_carries_a_failed_status_on_the_dgp(frame):
    for arm, outcome, covs in _planted_pairs(frame):
        assert _reading(frame, arm, outcome, covs).status in {"passed", "warning"}
```

- [ ] **Step 3: Run it**

Run: `(cd $W && /usr/bin/time -v $PY -m pytest tests/unit/test_causal_engine/test_sensitivity_calibration.py -q -p no:cacheprovider -n 0 2>&1 | grep -E "passed|failed|Maximum resident|Elapsed")`
Expected: `4 passed`, elapsed ~25–60 s, maximum resident set size under 600 MB. If `test_null_pairs_read_null_finding` fails on one pair, print that pair's `ate`/`ci`: a chance positive at n=1500 must NOT be papered over by removing it silently — record the measured CI in the test's docstring and use `n_records=3000` for that pair only if the spec's §2.3 numbers are reproduced there.

- [ ] **Step 4: Lint and commit**

```bash
(cd $W && $PY -m ruff check tests/unit/test_causal_engine/test_sensitivity_calibration.py && $PY -m ruff format --check tests/unit/test_causal_engine/test_sensitivity_calibration.py)
git -C $W branch --show-current
git -C $W add tests/unit/test_causal_engine/test_sensitivity_calibration.py
git -C $W commit -m "test(refutation): calibrate the sensitivity reading on planted truth; pin the omitted-confounder limit"
```

---

### Task 9: Re-band the live runs before merge

**Files:**
- Create: `scripts/calibration/reband_sensitivity_readings.py`
- Create: `docs/demos/results/<run-date>_sensitivity_calibration/reband.md` (output; `<run-date>` = today's `YYYY-MM-DD` when the script runs)

- [ ] **Step 1: Check memory**

Run: `free -m | sed -n 2p`. Continue only above 1500 MiB available.

- [ ] **Step 2: Write the script**

```python
#!/usr/bin/env python3
"""Re-band every live agent run under the 2026-09-10 sensitivity reading (spec §7).

For each ``causal_validations`` estimate with ``estimate_source = causal_impact_query``:
read the five stored test rows (statuses, effect, e_value_ci, n_rows), pull the
frame the way the route pulls it (``_load_agent_estimation_frame`` /
``_load_hcp_adoption_join_frame``, brand-scoped default covariates, the stored row
cap), compute the benchmark inputs on it, classify, and recompute the band with the
runner's own ``_calculate_confidence_score`` / ``_determine_gate_decision``.
Writes a markdown table of today's band vs the new band per pair.

Caveats printed into the output: a ``limit N`` pull has no guaranteed row order, so
population quantities can differ slightly from the original run's frame; pairs
without a current dataset mapping are listed as unmapped, never guessed.

Run (from the repo root, venv active, the live stack reachable):

    .venv/bin/python scripts/calibration/reband_sensitivity_readings.py \
        --out docs/demos/results/$(date +%F)_sensitivity_calibration/reband.md
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.causal_engine import evalue  # noqa: E402
from src.causal_engine.refutation_runner import (  # noqa: E402
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)

# (treatment, outcome) -> dataset, for the pairs the live table holds. Anything
# absent is reported as unmapped.
DATASET_BY_PAIR: Dict[Tuple[str, str], str] = {
    **{
        (t, o): "patient_journeys"
        for t in ("treatment_arm", "copay_support", "psp_enrolled", "rep_detailing_high", "sample_dropped",
                  "trigger_accepted", "treatment_initiated", "urticaria_severity_uas7", "disease_stage")
        for o in ("treatment_initiated", "adherent_180d", "low_gap_180d", "persistent_180d")
    },
    ("peer_influence_score", "adopted"): "hcp_adoption",
    ("treatment_arm", "adopted"): "hcp_adoption",
    ("control_group_flag", "action_taken"): "nba_triggers",
    ("acceptance_status", "conversion_flag"): "nba_triggers",
}


async def _rows(client) -> List[Dict[str, Any]]:
    res = await (
        client.table("causal_validations")
        .select("estimate_id,test_type,status,original_effect,brand,treatment_variable,outcome_variable,details_json,gate_decision,confidence_score")
        .eq("estimate_source", "causal_impact_query")
        .limit(5000)
        .execute()
    )
    return res.data or []


async def _frame(dataset: str, treatment: str, outcome: str, brand: Optional[str], limit: int):
    from src.api.routes.causal import (
        _CAUSAL_DATASET_SPECS,
        _brand_scoped_covariates,
        _load_agent_estimation_frame,
    )

    spec = _CAUSAL_DATASET_SPECS[dataset]
    covariates = [c for c in spec["covariate"] if c not in (treatment, outcome)]
    if brand:
        covariates = _brand_scoped_covariates(covariates, brand)
    df, select_cols = await _load_agent_estimation_frame(
        dataset=dataset, treatment_var=treatment, outcome_var=outcome,
        covariates=covariates, limit=limit, brand=brand or None,
    )
    covs = [c for c in select_cols if c not in (treatment, outcome) and c in df.columns]
    return df, covs


def _recover_ci(ate: float, e_value_ci: float, outcome_sd: float) -> Tuple[float, float]:
    """The stored row keeps ``e_value_ci`` (old SMD formula on the bound nearest the
    null), not the CI. Invert it exactly: ``RR = (E^2 + 1) / (2E)``, ``d = ln(RR) / 0.91``,
    ``bound = d * sd``. Only two facts matter to ``classify``: whether the CI includes
    zero (E <= 1.0) and the bound nearest the null; the far bound is set symmetric."""
    import math

    if e_value_ci <= 1.0:
        return (min(ate, 0.0) - 1e-9, max(ate, 0.0) + 1e-9)  # includes zero
    rr = (e_value_ci * e_value_ci + 1.0) / (2.0 * e_value_ci)
    bound = (math.log(rr) / 0.91) * outcome_sd
    near, far = bound, max(bound, 2.0 * abs(ate) - bound)
    return (near, far) if ate >= 0 else (-far, -near)


def _band(runner: RefutationRunner, statuses: Dict[str, str]) -> Tuple[str, float]:
    tests = [
        RefutationResult(RefutationTestType(name), RefutationStatus(st), 0.0, 0.0)
        for name, st in statuses.items()
        if name in RefutationTestType._value2member_map_
    ]
    conf = runner._calculate_confidence_score(tests)
    return runner._determine_gate_decision(tests, conf).value, conf


async def main(out: Path) -> int:
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        print("no supabase client", file=sys.stderr)
        return 2
    runner = RefutationRunner()
    by_estimate: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tests": {}})
    for r in await _rows(client):
        e = by_estimate[r["estimate_id"]]
        e["tests"][r["test_type"]] = r["status"]
        e.update(brand=r.get("brand") or "", t=r["treatment_variable"], o=r["outcome_variable"], old_gate=r["gate_decision"])
        if r["test_type"] == "sensitivity_e_value":
            d = r.get("details_json") or {}
            e.update(ate=float(r["original_effect"]), e_ci=float(d.get("e_value_ci") or 1.0),
                     randomized=(r["status"] == "skipped"), n=int(d.get("refutation_n_rows_total") or 1500))

    readings: Counter = Counter()
    moves: Counter = Counter()
    per_pair: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    frame_cache: Dict[Tuple[str, str, str, str, int], Any] = {}
    for eid, e in by_estimate.items():
        key = (e["brand"] or "<all>", e["t"], e["o"])
        dataset = DATASET_BY_PAIR.get((e["t"], e["o"]))
        if e.get("randomized"):
            reading = evalue.READING_RANDOMIZED
            new_sens = "skipped"
        elif dataset is None:
            reading, new_sens = "unmapped", None
        else:
            ck = (dataset, e["brand"], e["t"], e["o"], e["n"])
            if ck not in frame_cache:
                try:
                    frame_cache[ck] = await _frame(dataset, e["t"], e["o"], e["brand"], e["n"])
                except Exception as exc:  # noqa: BLE001 - report, never guess
                    frame_cache[ck] = exc
            got = frame_cache[ck]
            if isinstance(got, Exception):
                reading, new_sens = f"frame_error: {type(got).__name__}", None
            else:
                df, covs = got
                inputs = evalue.benchmark_inputs_from_frame(df, e["t"], e["o"], covs)
                # The stored CI is not persisted; recover the bound from e_value_ci on the
                # SMD path: RR = (E^2 + 1) / (2E), d = ln(RR)/0.91, bound = d * sd.
                sd = float(df[e["o"]].std())
                ate = e["ate"]
                ci = _recover_ci(ate, e["e_ci"], sd)
                rd = evalue.classify(ate, ci, randomized=False, baseline_risk=inputs.baseline_risk,
                                     outcome_std=sd, naive_effect=inputs.naive_effect,
                                     covariate_factors=inputs.covariate_bias_factors, n_rows=len(df))
                reading, new_sens = rd.reading, rd.status
                e.update(rr_point=rd.rr_point, benchmark=rd.benchmark, basis=rd.benchmark_basis)
        readings[reading] += 1
        e["reading"] = reading
        if new_sens is None:
            e["new_gate"] = "?"
        else:
            statuses = dict(e["tests"]); statuses["sensitivity_e_value"] = new_sens
            e["new_gate"], e["new_conf"] = _band(runner, statuses)
            moves[(e["old_gate"], e["new_gate"])] += 1
        per_pair[key].append(e)

    lines = [f"# Live re-band under the 2026-09-10 sensitivity reading ({date.today().isoformat()})", "",
             f"Estimates: {len(by_estimate)} (`estimate_source = causal_impact_query`).", "",
             "## Readings", "", "| reading | runs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in readings.most_common()]
    lines += ["", "## Gate moves (today → new)", "", "| move | runs |", "|---|---|"]
    lines += [f"| {a} → {b} | {n} |" for (a, b), n in sorted(moves.items())]
    lines += ["", "## Per pair", "", "| brand | treatment → outcome | runs | today | new | readings | median rr_point | median benchmark |", "|---|---|---|---|---|---|---|---|"]
    for key in sorted(per_pair):
        es = per_pair[key]
        med = lambda k: (f"{statistics.median([x[k] for x in es if k in x and x[k] is not None]):.2f}" if any(k in x and x[k] is not None for x in es) else "-")  # noqa: E731
        lines.append(f"| {key[0]} | {key[1]} → {key[2]} | {len(es)} | {dict(Counter(x['old_gate'] for x in es))} | "
                     f"{dict(Counter(x['new_gate'] for x in es))} | {dict(Counter(x['reading'] for x in es))} | {med('rr_point')} | {med('benchmark')} |")
    lines += ["", "## Caveats", "",
              "- A `limit N` frame pull has no guaranteed row order; population quantities may differ slightly from the original run's frame.",
              "- The stored row carries `e_value_ci`, not the CI; the CI bound is recovered from it on the SMD path (exact inverse of the runner's old formula).",
              "- Pairs listed as `unmapped` have no current dataset mapping and were not guessed.",
              "- Runs whose sensitivity row was SKIPPED (randomized design) keep SKIPPED."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    sys.exit(asyncio.run(main(ap.parse_args().out)))
```

Before running: read `src/memory/services/factories.py` for the exact name of the async client factory (`get_async_supabase_client` is what `routes/causal.py` imports) and load `.env` the way `run_discovery.py` does if the client needs env vars (`from dotenv import load_dotenv; load_dotenv(REPO / ".env")` at the top of `main`).

- [ ] **Step 3: Run it against the live stack**

Run: `(cd $W && $PY scripts/calibration/reband_sensitivity_readings.py --out docs/demos/results/$(date +%F)_sensitivity_calibration/reband.md 2>&1 | tail -30)`
Expected: the readings table (beyond ≈ 91, within ≈ 6, null_finding ≈ 5, not_applicable_randomized 6, the 13 continuous-treatment runs now benchmarked by covariate factors, unmapped 3) and the moves table (BLOCK → PROCEED ≈ 56, BLOCK → BLOCK 6 on random-common-cause failures). If any pair reads `frame_error`, print the exception and fix the loader call, never the mapping by guess.

- [ ] **Step 4: Lint, commit the script and the table**

```bash
(cd $W && $PY -m ruff check scripts/calibration/reband_sensitivity_readings.py && $PY -m ruff format --check scripts/calibration/reband_sensitivity_readings.py)
git -C $W branch --show-current
git -C $W add scripts/calibration/reband_sensitivity_readings.py docs/demos/results/*_sensitivity_calibration/reband.md
git -C $W commit -m "chore(calibration): re-band the live agent runs under the sensitivity reading; table for the owner (#1988 #1991)"
```

Then STOP and show the owner the table (this is the pre-merge review point of spec §7). Post it as a comment on #1988 and #1991 only after the owner has seen it.

---

### Task 10: The lineage page tells the truth about the gate

**Files:**
- Modify: `docs/lineage/causal_dag_lineage.html` (rows ~949/951, callouts ~960/1072/1073, E-value details ~962, calculator ~1250–1290, anchors 945/962/768)

- [ ] **Step 1: Apply exact-match replacements with a throwaway script**

Save as `/tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/83bb892d-e500-4741-85b3-2b40673a2f7f/scratchpad/lineage_edit.py` (scratch, not committed) and run it with `$PY … $W/docs/lineage/causal_dag_lineage.html`:

```python
import re, sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)

def rep_between(start, end, new):
    """Replace the unique span from `start` through `end` (inclusive) with `new`."""
    global s
    i = s.index(start); j = s.index(end, i) + len(end)
    assert s.count(start) == 1 and s.count(end) == 1
    s = s[:i] + new + s[j:]

# scoring table rows
rep('<tr><td>placebo_treatment</td><td>30 (10)</td><td>yes</td><td>placebo p ≥ 0.05</td><td>none reachable — the code tests pass (p ≥ 0.05) before warning (p ≥ 0.10), so the 0.05–0.10 band its comment names never scores</td><td class="num">0.25</td></tr>',
    '<tr><td>placebo_treatment</td><td>30 (10)</td><td>yes</td><td>placebo p ≥ 0.05; FAILED below</td><td>none — the code has no WARNING band; its comment now says so (#1994 option 1, 2026-09-10)</td><td class="num">0.25</td></tr>')
rep('<tr><td>sensitivity_e_value</td><td>—</td><td>yes</td><td>E-value(CI) ≥ 2.0</td><td>1.5–2.0</td><td class="num">0.25</td></tr>',
    '<tr><td>sensitivity_e_value</td><td>—</td><td><b>no</b> (2026-09-10)</td><td>reading <em>beyond measured confounding</em>: the CI excludes zero and the point estimate\'s risk ratio exceeds the confounding the adjustment removed (naive vs adjusted; strongest covariate bias factor for a continuous treatment)</td><td>WARNING readings: <em>within measured confounding</em>, <em>null finding</em> (CI includes zero, served with a caveat), <em>unbenchmarked</em> (no measured confounders); randomized designs SKIPPED. No FAILED outcome exists.</td><td class="num">0.25</td></tr>')

# decision pre block: the parenthetical about critical WARNING
rep('           (a critical test in WARNING still permits PROCEED, by design)</code></pre>',
    '           (a critical test in WARNING still permits PROCEED, by design;\n            since 2026-09-10 only placebo and random_common_cause are critical)</code></pre>')

# Measured callout (REVIEW-band arithmetic) — rewrite the paragraph
rep_between('<p>96 live runs on record (2026-09-08)', 'not a code defect.</p></div>',
  '<p>124 live agent runs on record (2026-09-10, before this change): 59 PROCEED, 65 BLOCK, <b>0 REVIEW</b>. Every BLOCK came from the sensitivity test alone (placebo passed 231/231; random common cause 225/231), and 42 of 59 PROCEED runs carried its WARNING — the 2.0 / 1.5 E-value cutoffs, unchanged since <code>0742b81f6</code>, were effect-size bands on a binary outcome and BLOCKed 7 of 11 correctly recovered planted truths at the live row cap (LinearDML, n = 1500). The E-value cannot detect confounding (omitted-confounder refits score as high as correct fits), so since 2026-09-10 the sensitivity test is a non-critical READING with no FAILED outcome (spec <code>2026-09-10-sensitivity-gate-calibration-design.md</code>). Band arithmetic is otherwise unchanged: confidence is the weighted mean over the NON-SKIPPED tests (critical 0.25 each, non-critical 0.125 each; PASSED 1.0, WARNING 0.6, FAILED 0.0); any critical FAILED → BLOCK, else ≥ 0.70 PROCEED, ≥ 0.50 REVIEW, else BLOCK. Reachable without a critical failure (enumerated by <code>tests/unit/test_causal_engine/test_refutation_bands_enumeration.py</code>): a sensitivity WARNING alone scores 0.90 (PROCEED); one critical WARNING plus both non-critical FAILED = 0.65 (REVIEW); the only confidence-only BLOCKs need BOTH critical tests in WARNING with BOTH non-critical tests FAILED (0.40, 0.45 or 0.48). Pre-merge re-band of the 124 runs under the new reading: 56 BLOCK → PROCEED, 6 stay BLOCK (all on a random-common-cause FAILED), 6 WARNING readings (5 %), 5 null findings served with a caveat.</p></div>')

# E-value details paragraph
rep_between('<p><code>rr = exp(0.91 · |effect| / outcome_std)</code>', 'refutation_runner.py:1802</span></p>',
  '<p>One module, <code>src/causal_engine/evalue.py</code>, serves the runner, the agent\'s sensitivity and interpretation nodes and the chat tool. Binary treatment and outcome: <code>RR = (p0 + |effect|) / p0</code> with the control-arm baseline risk on the FULL estimation frame (the VanderWeele–Ding risk-difference path); otherwise <code>RR = exp(0.91 · |effect| / outcome_std)</code>. <code>E = RR + sqrt(RR·(RR − 1))</code> on the point estimate and on the CI bound nearest the null (1.0 when the CI includes zero). The <b>benchmark</b> is the confounding the adjustment removed, <code>RR(naive) / RR(adjusted)</code>, or the strongest measured covariate\'s bias factor <code>B = RR_EU·RR_UD / (RR_EU + RR_UD − 1)</code> for a continuous treatment. Reading: <em>beyond</em> when the point risk ratio exceeds the benchmark (PASSED); <em>within</em> otherwise (WARNING); <em>null finding</em> when the CI includes zero (WARNING, served with the caveat "No detectable effect at this sample size"); <em>unbenchmarked</em> with no measured confounders (WARNING); randomized designs SKIPPED as before. Every message prints the numbers and the benchmark\'s basis. <span class="anchor">evalue.py:classify</span> <span class="anchor">refutation_runner.py:_run_sensitivity_test</span></p>')

# REVIEW-band list item + placebo list item
rep_between('<li><b>REVIEW is rare by design.</b>', '(issue #1969).</li>',
  '<li><b>REVIEW is rare by design; BLOCK is now rare too.</b> REVIEW is the weighted band (confidence ≥ 0.50 and &lt; 0.70 with no critical FAILED). Since 2026-09-10 only placebo and random_common_cause are critical; the sensitivity E-value is a reading (beyond / within / null finding / unbenchmarked) that can lower confidence but never blocks. Reachable values are enumerated in <code>test_refutation_bands_enumeration.py</code>: 0.65 (one critical WARNING + both non-critical FAILED) is REVIEW; a confidence-only BLOCK needs both critical tests in WARNING with both non-critical tests FAILED (0.40–0.48). A human rejection still halts every band (issue #1971). Infrastructure-absent paths resolve to <code>unavailable</code>, never to a claimed approval (issue #1969).</li>')
rep_between('<li><b>The placebo WARNING band is unreachable.</b>', "raised at the lane's close-out.</li>",
  '<li><b>The placebo test has no WARNING band.</b> <code>refutation_runner.py</code> tests the pass threshold (p ≥ 0.05) first, so a placebo result is PASSED or FAILED; the comment and this table say so since 2026-09-10 (#1994 option 1, behaviour unchanged — 231 of 231 live placebo p-values were ≥ 0.10).</li>')

# calculator: sensitivity non-critical, statuses per test
rep("    { id: 'sensitivity_e_value', label: 'Sensitivity (E-value)', w: 0.25, critical: true },",
    "    { id: 'sensitivity_e_value', label: 'Sensitivity (E-value)', w: 0.25, critical: false, statuses: ['PASSED', 'WARNING', 'SKIPPED'] },")
rep("    ['PASSED', 'WARNING', 'FAILED', 'SKIPPED'].forEach(function (s) {",
    "    (t.statuses || ['PASSED', 'WARNING', 'FAILED', 'SKIPPED']).forEach(function (s) {")
rep("    lab.innerHTML = '<b>' + t.label + (t.critical ? ' · critical' : '') + '</b>';",
    "    lab.innerHTML = '<b>' + t.label + (t.critical ? ' · critical' : t.id === 'sensitivity_e_value' ? ' · reading' : '') + '</b>';")

open(p, "w", encoding="utf-8").write(s)
print("lineage edits applied")
```

If any assertion fails, the page text differs from what this plan quotes: print the current line (`grep -n` on a distinctive fragment) and adjust the `old` string to the page — never skip the edit.

- [ ] **Step 2: Re-measure the two remaining line-number anchors**

```bash
grep -n "DEFAULT_CONFIG: Dict" $W/src/causal_engine/refutation_runner.py | head -1
grep -n "_latent_confounding_warning\|def _latent_warning" $W/src/agents/causal_impact/nodes/interpretation.py | head -2
```

Update `<span class="anchor">src/causal_engine/refutation_runner.py:730</span>` (line ~945) to the measured `DEFAULT_CONFIG` line and `<span class="anchor">interpretation.py:253</span>` (line ~768) to the measured line of the latent-warning method with `sed -i` on the exact old anchor strings. No map label changes in this task, so no rendered `getBBox` check is needed; if a later edit touches a map label, measure it in a real Chrome per memory.

- [ ] **Step 3: Check the page still parses and the residue tests pass**

```bash
$PY - <<'EOF'
from html.parser import HTMLParser
class P(HTMLParser):
    depth = 0
    def handle_starttag(self, t, a):
        if t not in ("br", "img", "input", "meta", "link", "hr"): self.depth += 1
    def handle_endtag(self, t): self.depth -= 1
p = P(); p.feed(open("/home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-d-1991/docs/lineage/causal_dag_lineage.html", encoding="utf-8").read()); print("tag depth", p.depth)
EOF
(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py -q -p no:cacheprovider -n 0 2>&1 | tail -2)
```

Expected: `tag depth 0` (same as before the edit — measure before AND after; the number must not change) and the residue tests pass.

- [ ] **Step 4: Commit**

```bash
git -C $W branch --show-current
git -C $W add docs/lineage/causal_dag_lineage.html
git -C $W commit -m "docs(lineage): refutation gate rows, callouts, E-value details and calculator reflect the sensitivity reading (#1988 #1994)"
```

---

### Task 11: The documentation page stops drawing a threshold

**Files:**
- Modify: `frontend/src/components/documentation/content.ts:748-749` (intro), `:797-806` (sensitivity entry)
- Modify: `frontend/src/components/documentation/RefutationGate.tsx:189-222` (`EValueIllustration`), `:341` (comment)
- Create: `frontend/src/components/documentation/content.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest';
import { REFUTATION_INTRO, REFUTATION_TESTS } from './content';

describe('refutation documentation content (2026-09-10 sensitivity reading)', () => {
  it('lists sensitivity as a non-critical reading with the benchmark rule', () => {
    const sens = REFUTATION_TESTS.find((t) => t.id === 'sensitivity_e_value');
    expect(sens).toBeDefined();
    expect(sens!.critical).toBe(false);
    expect(sens!.passRule).toMatch(/measured confounding/i);
    expect(sens!.passRule).not.toMatch(/2\.0/);
    expect(sens!.failSign).toMatch(/null finding|within measured confounding/i);
  });

  it('says two tests are critical', () => {
    expect(REFUTATION_INTRO).toMatch(/Two are critical/);
    expect(REFUTATION_TESTS.filter((t) => t.critical).map((t) => t.id).sort()).toEqual([
      'placebo_treatment',
      'random_common_cause',
    ]);
  });
});
```

- [ ] **Step 2: Run to verify red**

Run: `(cd $W/frontend && npx vitest run src/components/documentation/content.test.ts 2>&1 | tail -6)`
Expected: 2 failures (`critical` true, intro says "Three").

- [ ] **Step 3: Edit the content and the illustration**

`content.ts` intro:

```ts
export const REFUTATION_INTRO =
  'No causal estimate is reported until it survives five refutation tests — adversarial attacks that try to break it. Two are critical: a single failure blocks the estimate outright. The E-value sensitivity test is a reading, not a gate: it says how strong a hidden confounder would have to be, benchmarked against the confounding the adjustment actually removed. All five feed a weighted confidence score that decides the gate.';
```

Replace the sensitivity entry:

```ts
  {
    id: 'sensitivity_e_value',
    name: 'Sensitivity (E-value)',
    action: 'How strong would a hidden confounder have to be to explain the effect away?',
    mustHold: 'the effect outgrows the confounding we could measure',
    defaults: 'E-value on the point estimate and on the CI bound; benchmark = naive vs adjusted risk ratio (strongest covariate bias factor for a continuous treatment)',
    passRule: 'point-estimate risk ratio above the measured confounding benchmark → "Robust to confounding at measured strength"',
    critical: false,
    failSign:
      'A caveat, never a block: "Sensitive to confounding" when a confounder no stronger than the measured set could account for the whole effect; "No detectable effect at this sample size" (a null finding) when the CI includes zero.',
  },
```

`RefutationGate.tsx` — replace `EValueIllustration`:

```tsx
/* --------------------------------------------------------------- E-value */
function EValueIllustration({ outcome }: { outcome: Outcome }) {
  const ok = outcome === 'pass';
  const x0 = 24, x1 = 216, y = 56; // risk-ratio scale 1.0 → 2.0
  const px = (rr: number) => x0 + ((rr - 1) / 1) * (x1 - x0);
  const benchmark = 1.25; // illustrative: the confounding the adjustment removed
  const marker = ok ? 1.6 : 1.12;
  return (
    <>
      <text x="120" y="14" fontSize="9" textAnchor="middle" className={MUTED}>risk ratio needed to explain the effect away →</text>
      <rect x={px(1)} y={y - 9} width={px(benchmark) - px(1)} height="18" rx="3" fill={FAIL} fillOpacity="0.15" />
      <rect x={px(benchmark)} y={y - 9} width={px(2) - px(benchmark)} height="18" rx="3" fill={PASS} fillOpacity="0.15" />
      <line x1={x0} y1={y} x2={x1} y2={y} className={AXIS} strokeWidth="1" />
      {[1, 1.25, 1.5, 1.75, 2].map((rr) => (
        <g key={rr}>
          <line x1={px(rr)} y1={y - 12} x2={px(rr)} y2={y + 12} className={AXIS} strokeWidth={rr === benchmark ? 2 : 1} />
          <text x={px(rr)} y={y + 24} fontSize="9" textAnchor="middle" className={MUTED}>{rr.toFixed(2)}</text>
        </g>
      ))}
      <text x={px(1.12)} y={y - 16} fontSize="9" textAnchor="middle" className={MUTED}>within measured confounding</text>
      <text x={px(1.65)} y={y - 16} fontSize="9" textAnchor="middle" className={MUTED}>beyond it</text>
      <text x={px(benchmark)} y={y + 34} fontSize="9" fontWeight="600" textAnchor="middle" className={TXT}>measured confounding (per run)</text>
      <path
        d={`M${px(marker)},${y - 2} l-6,-10 l12,0 z`}
        fill={ok ? PASS : FAIL}
        className={ANIM}
        style={{ transitionProperty: 'd, fill' }}
      />
      <Verdict x={Math.min(176, Math.max(64, px(marker)))} y={110} outcome={outcome} pass="robust at measured strength" fail="sensitive — a caveat, not a block" />
    </>
  );
}
```

Change the comment at line ~341 to `// Every test failing at once is a BLOCK (two of them are critical); every`. Read `ILLUSTRATION_ALT` (grep it) and update the `sensitivity_e_value` alt strings to the new wording ("beyond" / "within measured confounding").

- [ ] **Step 4: Run the docs tests, typecheck, lint**

```bash
(cd $W/frontend && npx vitest run src/components/documentation/ 2>&1 | tail -6)
(cd $W/frontend && npm run typecheck 2>&1 | tail -3)
(cd $W/frontend && npx eslint src/components/documentation/content.ts src/components/documentation/RefutationGate.tsx src/components/documentation/content.test.ts 2>&1 | tail -3)
```

Expected: tests pass, typecheck clean, eslint clean. `free -m` before vitest (the heaviest local step); run only this directory.

- [ ] **Step 5: Commit**

```bash
git -C $W branch --show-current
git -C $W add frontend/src/components/documentation/content.ts frontend/src/components/documentation/RefutationGate.tsx frontend/src/components/documentation/content.test.ts
git -C $W commit -m "docs(frontend): refutation page describes the sensitivity reading; two critical tests"
```

---

### Task 12: Whole-diff review, PR, deploy, live certification, close-out

**Files:** none new except the cert record `docs/demos/results/<run-date>_sensitivity_calibration/cert.md` and the handoff.

- [ ] **Step 1: Final gate list on the branch tip**

```bash
(cd $W && $PY -m pytest tests/unit/test_causal_engine/test_evalue.py tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_1419.py tests/unit/test_causal_engine/test_refutation_runner_randomized.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py tests/unit/test_causal_engine/test_refutation_bands_enumeration.py tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py tests/unit/test_causal_engine/test_evalue_standardization_p4.py tests/unit/test_causal_engine/test_sensitivity_calibration.py tests/unit/test_agents/test_causal_impact/ tests/unit/test_agents/test_tool_composer/test_tools_fail_closed.py -q -p no:cacheprovider -n 0 2>&1 | tail -5)
(cd $W && git diff --name-only main -- '*.py' | xargs $PY -m ruff check && git diff --name-only main -- '*.py' | xargs $PY -m ruff format --check)
(cd $W && git diff --name-only main -- 'src/*.py' | xargs $PY -m mypy --config-file pyproject.toml)
```

Expected: all green. Any red is fixed with a NEW commit.

- [ ] **Step 2: Whole-diff codex fixed point**

`ralph-wiggum:ralph-loop` around `codex:codex-rescue` (read-only, `-C $W`) on `git diff main...HEAD` until `VERDICT: ACCEPT`. Brief: the spec path, the plan path, the file list, ≤25 commands, ≤40 lines per command, the CLAUDE.md pushback paragraph verbatim, and the question "does any change make a served estimate's narrative contradict its gate?". Fold findings as new commits.

- [ ] **Step 3: Push and open the PR (owner has approved the spec and the plan; the merge itself still waits for a go)**

```bash
git -C $W push -u origin claude/1991-sensitivity-calibration
gh pr create --repo enunezvn/e2i-causal-analytics --base main --head claude/1991-sensitivity-calibration \
  --title "feat(refutation): sensitivity E-value becomes a benchmarked reading; sensitivity non-critical (#1988 #1994 #1989, #1991 debt 2)" \
  --body-file /tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/83bb892d-e500-4741-85b3-2b40673a2f7f/scratchpad/pr_body.md
```

`pr_body.md` carries: the one-paragraph why (7 of 11 truths blocked; E-value cannot detect confounding), the spec and plan paths, the re-band table from Task 9 verbatim, the list of folded items (#1994 option 1, #1989 pin, YAML, lineage, docs page), "no migration, no OpenAPI change", `Closes #1988`, `Closes #1994`, `Closes #1989`, and the attribution footer:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01XBPxeAJJVgMnskP6jw6cPv
```

Watch CI: `gh pr checks <n> --watch`. Before merging run `gh pr update-branch <n>` (PR CI reuses the original merge SHA). Ask the owner for the merge go; merge with `--merge`, never squash.

- [ ] **Step 4: Deploy watch and container-content certification**

Hold until the LAST deploy run on main is terminal (`gh run list --branch main --workflow deploy.yml -L 3`). Then certify CONTENT, never the job conclusion:

```bash
docker exec e2i_api sh -c 'test -f /app/src/causal_engine/evalue.py && echo evalue:present'
docker exec e2i_api sh -c 'grep -c "\"critical\": False" /app/src/causal_engine/refutation_runner.py'
docker exec e2i_api sh -c 'grep -c "e_value_threshold" /app/src/causal_engine/refutation_runner.py'   # expect 0
docker ps --format '{{.Names}} {{.Image}}' | grep e2i_api
```

Positive control for the negative check: the same `grep -c "e_value_threshold"` on `git show main~1:src/causal_engine/refutation_runner.py` must return ≥ 1.

- [ ] **Step 5: Live verification (spec §8)**

1. Re-run the Remibrutinib 11-question discovery job with `docs/demos/results/2026-09-09_expert_review_loop/run_discovery.py <label> <out_dir>` (copy it into the new results dir; it reads `.env`). Expected: the six pairs BLOCKed on 2026-09-09 (`copay_support → persistent_180d`, `psp_enrolled → adherent_180d`, `psp_enrolled → persistent_180d`, `rep_detailing_high → treatment_initiated`, `sample_dropped → treatment_initiated`, `treatment_arm → persistent_180d`) now PROCEED unless a critical test fails; every sensitivity row in `causal_validations.details_json` carries `reading` and `headline`:

```bash
docker exec supabase-db psql -U postgres -d postgres -At -F' | ' -c "select treatment_variable, outcome_variable, status, gate_decision, details_json->>'reading', left(details_json->>'headline', 48) from causal_validations where test_type='sensitivity_e_value' and estimate_source='causal_impact_query' and created_at > now() - interval '2 hours' order by created_at;"
```

2. Null finding: run the brand-less `treatment_arm → persistent_180d` probe pair through the causal analyze endpoint (same method as lane 1's `rerun_*.json`); expected `gate_decision = proceed` when placebo and random common cause pass, `warnings` containing "No detectable effect at this sample size", the drill-down showing the sensitivity row as WARNING with that message, and no new `expert_reviews` row for the run (`select count(*) from expert_reviews where created_at > <job start>`).
3. Narrative: the API record's interpretation for a `beyond` run contains "Robust to confounding at measured strength" and the benchmark number; for the null run it does not contain "robust to unmeasured confounding".
4. Record the counts of `expert_reviews` pending rows before and after the job (expected unchanged for PROCEED runs).

Write `cert.md` in the results dir with every command and its output; commit it on a docs branch or as an untracked results dir per the repo's habit (results dirs are untracked today; the re-band table from Task 9 is the one committed artefact).

- [ ] **Step 6: Issues, memory, handoff**

- Comment the re-band table and the cert summary on #1988 and #1991; #1988, #1994 and #1989 close via the PR's `Closes` lines (verify with `gh issue view`); post the debt-2 status on #1991 (first slice shipped; second slice — continuous scores — waits on a continuous bootstrap term).
- File the new issue: "tool registry `ToolSchema` for `sensitivity_analyzer` declares `causal_result`/`gamma_range` → `e_value`/`robustness_value`/`sensitivity_plot_data`, unlike the registered function (`ate`, `ci_lower`, … → `e_value_point`, `e_value_ci`, `reading`)"; cite `tool_registry.py:392` and `tool_registrations.py`.
- Update memory `evalue_cannot_be_confounding_gate_calibration_20260910.md` with the PR number, merge SHA, cert result and the measured re-band table; add the MEMORY.md pointer line.
- Remove the worktree only after the merge and cert: `git -C /home/enunez/Projects/e2i_causal_analytics worktree remove .worktrees/lane-d-1991`.
- Write `.claude/handoffs/current.md` with the final state and "next: #1990 CausalPFN scratch trial, run ALONE by memory".

---

## Self-review against the spec

- §4.1 module and every function → Task 1. §4.2 conversion table → Task 1 (`_rr_of_effect`, `benchmark_inputs_from_frame`) and Task 8 exercises the binary path. §4.3 full-frame inputs threaded like `outcome_std` → Task 4 (node) and Task 2 (runner kwargs + fallback). §4.4 five readings, statuses, headlines, messages → Task 1; the tie rule → `test_tie_reads_within`. §4.5 config, thresholds deletion, critical-from-config, #1989 pin, #1994 comment, YAML + residue → Tasks 2 and 3. §4.6 refutation node caveat, sensitivity node, interpretation node, state TypedDict → Tasks 4, 5, 6. §4.7 chat tool + registry description + filed mismatch → Tasks 7 and 12. §4.8 lineage page and documentation page → Tasks 10 and 11. §4.9 no migration / no OpenAPI change → no task adds either; Task 12 step 1 lint/mypy only. §5 error handling → `evalue` raises `ValueError` (Task 1 test), nodes fail open to empty inputs (Task 4/5 tests). §6 tests → Tasks 1, 2, 4, 5, 6, 7, 8, 11. §7 re-band before merge → Task 9 with the explicit STOP for the owner. §8 live verification → Task 12 step 5. §9 rollout → conventions block + Task 12. §10 known limits → Task 8's pinned limit test and Task 10's callout. §11 issues → Task 12 step 6.
- Placeholders: `<run-date>` is a naming convention, not a gap; the executor substitutes today's date. No "TBD"/"add validation" anywhere.
- Type consistency: `evalue.classify(effect, ci, *, randomized, baseline_risk, outcome_std, naive_effect, covariate_factors, n_rows)` is called with those exact keyword names in Tasks 2, 5, 7, 8, 9. `BenchmarkInputs(baseline_risk, naive_effect, covariate_bias_factors, treatment_is_binary, outcome_is_binary, n_rows)` is used identically in Tasks 1, 4, 5, 9. Reading constants `READING_BEYOND` / `READING_WITHIN` / `READING_NULL` / `READING_UNBENCHMARKED` / `READING_RANDOMIZED` and `HEADLINES` are referenced in Tasks 4, 5, 6, 7. `_run_sensitivity_test(original_effect, original_ci, outcome_std, randomized_design, baseline_risk, naive_effect, covariate_bias_factors, n_rows)` matches its callers in Task 2. `RefutationResult(test_name, status, original_effect, refuted_effect, ...)` positional order matches the dataclass.
