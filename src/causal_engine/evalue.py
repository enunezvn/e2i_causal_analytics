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


def _validate_outcome_std(outcome_std: Optional[float]) -> None:
    """``outcome_std``, when given, must be finite and positive (spec §5).

    ``None`` alone means "no SD available" and falls back to the raw effect as the
    standardized difference. Silently accepting a NaN or non-positive SD would take
    that SAME fallback and report a plausible-wrong risk ratio (a binary outcome at
    effect 0.15 would read 1.146 instead of the correctly standardized 1.345), so
    this validates eagerly at every entry point that accepts ``outcome_std`` —
    regardless of which conversion path a given call ends up using.
    """
    if outcome_std is None:
        return
    s = _finite("outcome_std", outcome_std)
    if s <= 0:
        raise ValueError(f"outcome_std must be positive, got {outcome_std!r}")


def _rr_smd_path(effect: float, outcome_std: Optional[float]) -> float:
    """SMD-derived risk ratio for one effect. Caller must have validated ``outcome_std``."""
    d = abs(effect) if outcome_std is None else abs(effect) / outcome_std
    return rr_from_smd(d)


def _use_risk_ratio_path(
    effect: float, naive_effect: Optional[float], baseline_risk: Optional[float]
) -> bool:
    """Whether the risk-difference/risk-ratio conversion applies to the WHOLE reading.

    Spec §5 (conversion defect): never mix a risk ratio (from a risk difference) with
    an SMD-derived ratio inside one comparison. The RD path applies only when BOTH the
    point/adjusted effect and the naive effect (when given) land inside the
    risk-difference domain (``0 < baseline_risk + effect < 1``); otherwise
    EVERYTHING — point, CI bound, naive and adjusted alike — uses the SMD path.
    """
    if baseline_risk is None:
        return False
    if rr_from_risk_difference(effect, baseline_risk) is None:
        return False
    if naive_effect is not None:
        naive = _finite("naive_effect", naive_effect)
        if rr_from_risk_difference(naive, baseline_risk) is None:
            return False
    return True


def joint_confounding_benchmark(
    naive_effect: Optional[float],
    adjusted_effect: float,
    *,
    baseline_risk: Optional[float],
    outcome_std: Optional[float],
) -> Optional[float]:
    """``B_obs = RR(naive)/RR(adjusted)`` oriented >= 1: the confounding the adjustment removed.

    Naive and adjusted share ONE conversion for the pair (spec §2.7/§5): the
    risk-ratio path only when BOTH land inside the risk-difference domain, otherwise
    BOTH use the SMD path — never a risk ratio divided by an SMD-derived ratio.
    """
    _validate_outcome_std(outcome_std)
    if naive_effect is None:
        return None
    naive = _finite("naive_effect", naive_effect)
    adjusted = _finite("adjusted_effect", adjusted_effect)
    if _use_risk_ratio_path(adjusted, naive, baseline_risk):
        assert baseline_risk is not None  # guaranteed by _use_risk_ratio_path
        rr_naive = rr_from_risk_difference(naive, baseline_risk)
        rr_adj = rr_from_risk_difference(adjusted, baseline_risk)
        assert rr_naive is not None and rr_adj is not None  # guaranteed by the check above
    else:
        rr_naive = _rr_smd_path(naive, outcome_std)
        rr_adj = _rr_smd_path(adjusted, outcome_std)
    return _orient(rr_naive / rr_adj)


def measured_confounding_benchmark(
    joint: Optional[float], covariate_factors: Mapping[str, float]
) -> Tuple[Optional[float], str]:
    """The benchmark and its basis: joint when available, else the strongest covariate, else none.

    A non-finite ``joint``, or any ``None``/non-finite covariate factor, raises
    ``ValueError`` rather than being silently dropped — a drop can turn a benchmarked
    run into an ``unbenchmarked`` one and understate what confounding was measured.
    """
    if joint is not None:
        return _finite("joint", joint), "joint_naive_vs_adjusted"
    values: Dict[str, float] = {}
    for name, factor in covariate_factors.items():
        if factor is None:
            raise ValueError(f"covariate factor {name!r} is None; omit it instead of passing None")
        values[name] = _finite(f"covariate factor {name!r}", factor)
    if values:
        return max(values.values()), "strongest_covariate"
    return None, "none_measured"


def _is_binary(values: np.ndarray) -> bool:
    """Binary means BOTH 0 and 1 are present (spec §4.2): exact equality with
    ``{0.0, 1.0}`` after dropping NaNs, mirroring the estimation node's
    ``_compute_naive_contrast``. A constant all-0 or all-1 column is NOT binary —
    there is no contrast to form from a single observed level.
    """
    u = np.unique(values[~np.isnan(values)])
    return set(u.tolist()) == {0.0, 1.0}


def _high_mask(values: np.ndarray) -> np.ndarray:
    """Binary as-is (== 1); continuous split at the median (strictly above)."""
    if _is_binary(values):
        return np.asarray(values == 1.0)
    return np.asarray(values > np.nanmedian(values))


def _ratio_or_limit(numerator: float, denominator: float) -> Tuple[Optional[float], bool]:
    """A finite ratio, or a Ding-VanderWeele one-sided-zero limit flag.

    ``(ratio, False)`` when both inputs are strictly positive (a normal, finite
    ratio); ``(None, True)`` when exactly one is zero — the ratio's limit is
    infinity, and the caller falls back to the OTHER factor alone, per
    ``bias_factor``'s own limit (as RR -> infinity, B -> the other factor);
    ``(None, False)`` when both are zero (genuinely undefined, 0/0).
    """
    if numerator > 0 and denominator > 0:
        return numerator / denominator, False
    if (numerator == 0.0) != (denominator == 0.0):
        return None, True
    return None, False


def covariate_bias_factors(
    frame: Any, treatment: str, outcome: str, covariates: Sequence[str]
) -> Dict[str, float]:
    """Per-covariate bias factor from the frame (spec §4.1).

    RR_EU: share of high-covariate units among treated / among controls (treated is
    ``T == 1`` for a binary treatment, ``T > median`` otherwise). RR_UD: outcome rate
    among high-covariate CONTROLS / low-covariate controls (binary outcome), or the
    SMD path on the control-arm mean difference (continuous outcome).

    A one-sided zero share or zero event count is a REAL Ding-VanderWeele limit, not
    an undefined value: as RR_UD -> infinity, B -> RR_EU, and as RR_EU -> infinity,
    B -> RR_UD (``bias_factor``'s own formula, taken to its limit). Those limits are
    applied explicitly rather than dropped, because dropping them silently discards
    the STRONGEST possible confounders and understates the fallback benchmark.

    A covariate is skipped only when genuinely undefined and carrying no
    information: covariate absent from the frame; no complete-case rows; no treated
    or no control units; no high- or low-covariate stratum among controls (RR_UD has
    nothing to compare); zero events in BOTH control strata (0/0, binary outcome);
    zero outcome variance (continuous outcome, no SMD denominator); or RR_EU and
    RR_UD are simultaneously at their one-sided-zero limit (an indeterminate double
    limit).
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
        if not ok.any():
            continue
        hi = _high_mask(x[ok])
        tr, co = treated[ok], control[ok]
        if not tr.any() or not co.any():
            continue
        hi_c, lo_c = hi & co, (~hi) & co
        if not (hi_c.any() and lo_c.any()):
            continue
        p_hi_t, p_hi_c = float(hi[tr].mean()), float(hi[co].mean())
        rr_eu, eu_limit = _ratio_or_limit(p_hi_t, p_hi_c)
        yy = y[ok]
        if y_binary:
            m_hi, m_lo = float(yy[hi_c].mean()), float(yy[lo_c].mean())
            rr_ud, ud_limit = _ratio_or_limit(m_hi, m_lo)
            if rr_ud is None and not ud_limit:
                continue  # 0/0: no events in either control stratum
        else:
            if not y_sd or y_sd <= 0:
                continue  # no outcome variance to standardize against
            rr_ud = rr_from_smd((yy[hi_c].mean() - yy[lo_c].mean()) / y_sd)
            ud_limit = False  # the SMD path is always finite for a positive y_sd

        if eu_limit and ud_limit:
            continue  # both ratios at their one-sided-zero limit: indeterminate
        if eu_limit:
            assert rr_ud is not None  # eu_limit True implies ud_limit False (checked above)
            out[cov] = _orient(rr_ud)  # RR_EU -> infinity: B -> RR_UD
        elif ud_limit:
            # p_hi_c > 0 always (hi_c.any() was checked above), so rr_eu is never
            # None here — eu_limit False (this branch) means it is a real ratio.
            assert rr_eu is not None
            out[cov] = _orient(rr_eu)  # RR_UD -> infinity: B -> RR_EU
        else:
            assert rr_eu is not None and rr_ud is not None
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
    _validate_outcome_std(outcome_std)
    eff = _finite("effect", effect)
    lo, hi = _finite("ci_lower", ci[0]), _finite("ci_upper", ci[1])
    if lo > hi:
        lo, hi = hi, lo
    includes_null = lo <= 0.0 <= hi
    bound = min(abs(lo), abs(hi))

    # One conversion governs point, CI bound, naive and adjusted alike (spec §5): the
    # risk-ratio path applies only when the point effect AND the naive effect (if
    # any) both land inside the risk-difference domain — never RD for one and SMD
    # for the other within a single reading. The CI bound nearest the null sits
    # strictly between 0 and the point effect in magnitude, so if the point effect's
    # baseline_risk + effect is inside (0, 1) the bound's baseline_risk + bound is
    # too (a convex combination of two in-domain points); no separate check needed.
    use_rr = _use_risk_ratio_path(eff, naive_effect, baseline_risk)
    if use_rr:
        assert baseline_risk is not None  # guaranteed by _use_risk_ratio_path
        conversion = "risk_ratio"
        rr_point = rr_from_risk_difference(eff, baseline_risk)
        assert rr_point is not None  # guaranteed by _use_risk_ratio_path
        if includes_null:
            rr_ci = 1.0
        else:
            rr_ci_rd = rr_from_risk_difference(math.copysign(bound, eff), baseline_risk)
            assert rr_ci_rd is not None  # guaranteed: the CI bound is closer to null than effect
            rr_ci = rr_ci_rd
    else:
        conversion = "standardized_difference"
        rr_point = _rr_smd_path(eff, outcome_std)
        rr_ci = 1.0 if includes_null else _rr_smd_path(math.copysign(bound, eff), outcome_std)

    e_point, e_ci = e_value_from_rr(rr_point), e_value_from_rr(rr_ci)
    joint = joint_confounding_benchmark(
        naive_effect, eff, baseline_risk=baseline_risk, outcome_std=outcome_std
    )
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

    n_rows_i: Optional[int] = None if n_rows is None else int(n_rows)
    n_txt = f"n = {n_rows_i}" if n_rows_i else "this sample size"
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
        n_rows=n_rows_i,
    )
