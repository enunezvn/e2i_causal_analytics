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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


class _LevelNotScoreable(ValueError):
    """A covariate column cannot be scored on THIS frame: RR_EU and RR_UD are at
    their one-sided-zero limits at once, so the bias factor diverges.

    A ``ValueError`` subclass, so the public contract of ``covariate_bias_factors``
    is unchanged for numeric covariates. It exists so the categorical route can tell
    this one refusal apart from a genuine domain error and skip just that level.
    """


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
    """``E = RR + sqrt(RR*(RR-1))`` for RR >= 1; a protective RR is inverted first.

    Computed as ``RR + sqrt(RR)*sqrt(RR-1)`` rather than ``RR + sqrt(RR*(RR-1))``:
    the product ``RR*(RR-1)`` overflows to ``inf`` for a very large but still finite
    RR (~1e158) before the sqrt ever runs, silently reporting an infinite E-value —
    and Postgres JSONB rejects ``Infinity`` outright. Any residual non-finite result
    still raises ``ValueError`` rather than escaping as ``inf``.
    """
    r = _orient(_finite("rr", rr))
    if r <= 1.0:
        return 1.0
    return _finite("e_value", r + math.sqrt(r) * math.sqrt(r - 1.0))


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
    effect: float,
    naive_effect: Optional[float],
    baseline_risk: Optional[float],
    *,
    bound: Optional[float] = None,
) -> bool:
    """Whether the risk-difference/risk-ratio conversion applies to the WHOLE reading.

    Spec §5 (conversion defect): never mix a risk ratio (from a risk difference) with
    an SMD-derived ratio inside one comparison. The RD path applies only when
    ``baseline_risk`` is given AND ``effect``, ``naive_effect`` (when given) and
    ``bound`` (when given) ALL land inside the risk-difference domain
    (``0 < baseline_risk + x < 1``); otherwise EVERYTHING — point, CI bound, naive
    and adjusted alike — uses the SMD path.

    ``bound`` is the already-signed CI bound nearest the null. Effect alone landing
    in-domain is NOT sufficient to guarantee the bound does too — the domain's upper
    edge can sit strictly between them (``baseline_risk + effect`` a hair below 1,
    ``baseline_risk + bound`` exactly 1) — so the caller must check the bound
    explicitly rather than relying on a "bound is closer to null" argument. Pass
    ``None`` when there is no bound to check: a null-including CI (``rr_ci`` is 1.0
    regardless of conversion) or a caller with no CI at all (``joint_confounding_benchmark``).
    """
    if baseline_risk is None:
        return False
    if rr_from_risk_difference(effect, baseline_risk) is None:
        return False
    if naive_effect is not None:
        naive = _finite("naive_effect", naive_effect)
        if rr_from_risk_difference(naive, baseline_risk) is None:
            return False
    if bound is not None and rr_from_risk_difference(bound, baseline_risk) is None:
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
    Every supplied factor is validated FIRST, regardless of whether ``joint`` is
    given — a bad factor must not slip through just because the joint benchmark
    happened to make it unused.
    """
    values: Dict[str, float] = {}
    for name, factor in covariate_factors.items():
        if factor is None:
            raise ValueError(f"covariate factor {name!r} is None; omit it instead of passing None")
        values[name] = _finite(f"covariate factor {name!r}", factor)
    if joint is not None:
        return _finite("joint", joint), "joint_naive_vs_adjusted"
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


def _is_categorical_column(column: Any) -> bool:
    """Object / string / categorical dtype: no median to split the column at.

    ``bool`` is NOT here. A bool column IS binary, and spec §4.1 takes a binary
    covariate as-is; ``np.asarray(col, dtype=float)`` maps True/False to 1/0, so it
    belongs on the numeric path and scores identically to its 0/1 integer twin.
    Routing it through the max-over-levels rule instead made the two disagree.
    """
    kind = getattr(getattr(column, "dtype", None), "kind", None)
    return kind in ("O", "U", "S")


def _missing_mask(column: Any) -> np.ndarray:
    """Null mask for a categorical column, via pandas when the object offers it."""
    isna = getattr(column, "isna", None)
    if callable(isna):
        return np.asarray(isna(), dtype=bool)
    values = np.asarray(column, dtype=object)
    return np.array(
        [v is None or (isinstance(v, float) and math.isnan(v)) for v in values], dtype=bool
    )


def _distinct_levels(column: Any) -> List[Any]:
    """Non-null levels in first-seen order — deterministic for a given frame, so the
    refusal below always names the same level for the same data. Nulls are dropped
    before anything is compared, so a nullable dtype's ``pd.NA`` never becomes a
    level and never reaches an ambiguous truth test."""
    values = np.asarray(column, dtype=object)
    missing = _missing_mask(column)
    return list(dict.fromkeys(v for v, m in zip(values, missing, strict=True) if not m))


def _level_indicator(column: Any, level: Any) -> np.ndarray:
    """``column == level`` as float 0/1, NaN where the source value is null.

    The comparison runs over the NON-NULL positions only. pandas' nullable dtypes
    (``string``, ``Int64``, …) hold ``pd.NA``, and ``pd.NA == level`` is itself
    ``pd.NA``, so comparing the whole raw array first raises "boolean value of NA is
    ambiguous" — a ``TypeError`` that would take a valid categorical column, and the
    whole refutation node with it, down. Masking first also keeps the result exactly
    the column a caller would hand-build and pass as a numeric covariate, so both
    routes score identically.
    """
    values = np.asarray(column, dtype=object)
    missing = _missing_mask(column)
    indicator = np.full(values.shape, np.nan, dtype=float)
    present = ~missing
    if present.any():
        indicator[present] = np.asarray(values[present] == level, dtype=float)
    return indicator


def _as_numeric_or_none(column: Any) -> Optional[np.ndarray]:
    """The column as floats, or ``None`` when it is categorical rather than numeric."""
    if _is_categorical_column(column):
        return None
    try:
        return np.asarray(column, dtype=float)
    except (TypeError, ValueError):
        return None


def _factor_from_numeric_covariate(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    *,
    y_binary: bool,
    y_sd: Optional[float],
    cov: str,
    level_note: str = "",
) -> Optional[float]:
    """One bias factor from an ALREADY-NUMERIC covariate column (spec §4.1).

    The single home for the per-covariate math: the median/binary split, RR_EU from
    the treated-vs-control high shares, RR_UD among controls, the Ding-VanderWeele
    one-sided limits and the positivity refusal. A numeric column reaches it once; a
    categorical column reaches it once per level indicator, so the two paths cannot
    drift apart. ``None`` means skip (see ``covariate_bias_factors``' skip list).
    """
    ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(t)
    if not ok.any():
        return None
    hi = _high_mask(x[ok])
    tr, co = treated[ok], control[ok]
    if not tr.any() or not co.any():
        return None
    hi_c, lo_c = hi & co, (~hi) & co
    if not (hi_c.any() and lo_c.any()):
        return None
    p_hi_t, p_hi_c = float(hi[tr].mean()), float(hi[co].mean())
    rr_eu, eu_limit = _ratio_or_limit(p_hi_t, p_hi_c)
    yy = y[ok]
    if y_binary:
        m_hi, m_lo = float(yy[hi_c].mean()), float(yy[lo_c].mean())
        rr_ud, ud_limit = _ratio_or_limit(m_hi, m_lo)
        if rr_ud is None and not ud_limit:
            return None  # 0/0: no events in either control stratum
    else:
        if not y_sd or y_sd <= 0:
            return None  # no outcome variance to standardize against
        rr_ud = rr_from_smd((yy[hi_c].mean() - yy[lo_c].mean()) / y_sd)
        ud_limit = False  # the SMD path is always finite for a positive y_sd

    if eu_limit and ud_limit:
        raise _LevelNotScoreable(
            f"covariate {cov!r} perfectly separates treatment and outcome in this "
            f"frame{level_note} (positivity violation); its bias factor is unbounded"
        )
    if eu_limit:
        assert rr_ud is not None  # eu_limit True implies ud_limit False (checked above)
        return _orient(rr_ud)  # RR_EU -> infinity: B -> RR_UD
    if ud_limit:
        # p_hi_c > 0 always (hi_c.any() was checked above), so rr_eu is never None
        # here — eu_limit False (this branch) means it is a real ratio.
        assert rr_eu is not None
        return _orient(rr_eu)  # RR_UD -> infinity: B -> RR_EU
    assert rr_eu is not None and rr_ud is not None
    return bias_factor(rr_eu, rr_ud)


def _factor_from_categorical_covariate(
    column: Any,
    t: np.ndarray,
    y: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    *,
    y_binary: bool,
    y_sd: Optional[float],
    cov: str,
) -> Optional[float]:
    """The strongest level's bias factor for a categorical covariate.

    A categorical confounder IS measured confounding: the live #1351 resolver binds
    string driver columns into the adjustment set and the estimation node fits their
    one-hot encoding (#1417). Dropping them would understate the fallback benchmark
    and let a run read ``beyond_measured_confounding`` on confounding that was in
    fact measured — the false robustness this reading exists to prevent. Each level
    becomes a 0/1 indicator scored by the SAME per-covariate path as a numeric
    column, and the covariate keeps ONE entry under its own name (consumers key by
    covariate, not by level) holding the MAX: the strongest confounding this column
    could carry.

    A LEVEL that hits the both-limits case is SKIPPED, not fatal — the asymmetry with
    the numeric route is deliberate. A numeric covariate that separates both is a
    whole DECLARED confounder the estimator cannot have adjusted for: a real data
    problem, and it still refuses. A categorical LEVEL is a single sparse cell of a
    declared variable, and at small n a rare level routinely has a few non-event
    controls and no treated rows: measured on synthetic long-tail frames (30–80
    levels, n = 200, 10 % events), the refusal fired in 289–300 of 300 seeded runs on
    SPARSITY alone, which would have failed the whole refutation node. Live driver
    columns are single-digit cardinality (0 of 200 raises at 4 and 12 levels,
    n = 1500) and ``_MAX_CATEGORICAL_CARDINALITY = 50`` in
    ``src/agents/causal_impact/nodes/estimation.py`` (enforced by
    ``_encode_categorical_covariates``) bounds what can reach here at all, so this
    was latent — but a sparse cell must never fail a run.

    When NO level is scorable the covariate simply yields no factor, exactly like the
    other no-variation skips. A covariate perfectly aligned with treatment is already
    a skip on the numeric route, so refusing here would open a new fail-closed path
    for a contrived case rather than reporting a real one.
    """
    levels = _distinct_levels(column)
    if len(levels) < 2:
        return None  # a single level has no variation to benchmark
    factors: List[float] = []
    for level in levels:
        try:
            factor = _factor_from_numeric_covariate(
                _level_indicator(column, level),
                t,
                y,
                treated,
                control,
                y_binary=y_binary,
                y_sd=y_sd,
                cov=cov,
                level_note=f" at level {level!r}",
            )
        except _LevelNotScoreable:
            continue  # a sparse cell, not a declared variable — see the docstring
        if factor is not None:
            factors.append(factor)
    return max(factors) if factors else None


def covariate_bias_factors(
    frame: Any, treatment: str, outcome: str, covariates: Sequence[str]
) -> Dict[str, float]:
    """Per-covariate bias factor from the frame (spec §4.1).

    RR_EU: share of high-covariate units among treated / among controls (treated is
    ``T == 1`` for a binary treatment, ``T > median`` otherwise). RR_UD: outcome rate
    among high-covariate CONTROLS / low-covariate controls (binary outcome), or the
    SMD path on the control-arm mean difference (continuous outcome).

    A NUMERIC covariate is split at its median (binary — ``bool`` included — as-is).
    A CATEGORICAL one (object / string / categorical dtype, or a column no float
    conversion accepts) has no median: every level becomes a 0/1 indicator scored by
    the same path, and the covariate's factor is the strongest of them, under the
    covariate's own name. See ``_factor_from_categorical_covariate`` for why they are
    scored rather than dropped.

    A one-sided zero share or zero event count is a REAL Ding-VanderWeele limit, not
    an undefined value: as RR_UD -> infinity, B -> RR_EU, and as RR_EU -> infinity,
    B -> RR_UD (``bias_factor``'s own formula, taken to its limit). Those limits are
    applied explicitly rather than dropped, because dropping them silently discards
    the STRONGEST possible confounders and understates the fallback benchmark.

    A covariate is skipped only when genuinely undefined and carrying no
    information: covariate absent from the frame; no complete-case rows; no treated
    or no control units; no high- or low-covariate stratum among controls (RR_UD has
    nothing to compare); zero events in BOTH control strata (0/0, binary outcome);
    zero outcome variance (continuous outcome, no SMD denominator); a categorical
    covariate with fewer than two non-null levels (no variation to benchmark); a
    categorical LEVEL that is unscoreable at this sample size (see
    ``_factor_from_categorical_covariate``); or a categorical covariate whose EVERY
    level was skipped for one of those reasons — including the both-unscoreable case
    of a covariate perfectly aligned with treatment, which the numeric route also
    skips.

    When RR_EU and RR_UD are SIMULTANEOUSLY at their one-sided-zero limit, the bias
    factor B = RR_EU*RR_UD/(RR_EU+RR_UD-1) DIVERGES rather than settling on either
    limit. For a NUMERIC covariate that is a positivity violation the estimator
    cannot have adjusted for — a whole declared confounder separating both — and it
    is neither skipped nor fabricated as a finite number (an infinite bias factor
    must never reach ``details_json``; Postgres JSONB rejects ``Infinity``):
    ``ValueError`` is raised naming the covariate. For a CATEGORICAL covariate the
    same arithmetic on ONE level means only that this sparse cell cannot be scored at
    this sample size, so the level is skipped and the covariate keeps the strongest
    of the rest; the categorical route never raises it.
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
        column = frame[cov]
        numeric = _as_numeric_or_none(column)
        if numeric is not None:
            factor = _factor_from_numeric_covariate(
                numeric, t, y, treated, control, y_binary=y_binary, y_sd=y_sd, cov=cov
            )
        else:
            factor = _factor_from_categorical_covariate(
                column, t, y, treated, control, y_binary=y_binary, y_sd=y_sd, cov=cov
            )
        if factor is not None:
            out[cov] = factor
    return out


@dataclass(frozen=True)
class BenchmarkInputs:
    """What the classifier needs from the FULL estimation frame (spec §4.3)."""

    baseline_risk: Optional[float]
    naive_effect: Optional[float]
    covariate_bias_factors: Dict[str, float] = field(default_factory=dict)
    treatment_is_binary: bool = False
    outcome_is_binary: bool = False
    # ``None`` = no frame was looked at, so a consumer may fall back to its own count.
    # ``benchmark_inputs_from_frame`` always sets an int, ZERO included: a frame with
    # no usable rows has a computed count of zero, which is a measurement, not an
    # absence. Collapsing the two lets the refutation runner substitute the
    # SUBSAMPLE's length into a reading whose frame yielded nothing.
    n_rows: Optional[int] = None


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
    if eff < lo - 1e-12 or eff > hi + 1e-12:
        raise ValueError(
            f"ci must contain the point estimate (effect={eff!r}, ci=({lo!r}, {hi!r}))"
        )
    includes_null = lo <= 0.0 <= hi
    bound = min(abs(lo), abs(hi))
    # always a definite float; only meaningful (as the CI-bound conversion input)
    # when the CI excludes zero — guarded explicitly at each use below
    signed_bound = math.copysign(bound, eff)

    # One conversion governs point, CI bound, naive and adjusted alike (spec §5):
    # never RD for some of them and SMD for others within one reading. The point
    # effect landing in the risk-difference domain does NOT guarantee the bound does
    # too — the domain's edge can sit strictly between them (baseline_risk + effect a
    # hair below 1, baseline_risk + bound exactly 1) — so the bound is part of the
    # single decision, not assumed safe by a "closer to null" argument. None is
    # passed for the bound check when the CI includes zero, since rr_ci is 1.0
    # regardless of conversion and there is nothing to check.
    use_rr = _use_risk_ratio_path(
        eff, naive_effect, baseline_risk, bound=None if includes_null else signed_bound
    )
    if use_rr:
        assert baseline_risk is not None  # guaranteed by _use_risk_ratio_path
        conversion = "risk_ratio"
        rr_point = rr_from_risk_difference(eff, baseline_risk)
        assert rr_point is not None  # guaranteed by _use_risk_ratio_path
        if includes_null:
            rr_ci = 1.0
        else:
            rr_ci_rd = rr_from_risk_difference(signed_bound, baseline_risk)
            # guaranteed by _use_risk_ratio_path's own bound check just above: it
            # already confirmed signed_bound converts before returning True
            assert rr_ci_rd is not None
            rr_ci = rr_ci_rd
    else:
        conversion = "standardized_difference"
        rr_point = _rr_smd_path(eff, outcome_std)
        rr_ci = 1.0 if includes_null else _rr_smd_path(signed_bound, outcome_std)

    e_point, e_ci = e_value_from_rr(rr_point), e_value_from_rr(rr_ci)
    # the joint benchmark must use the SAME conversion classify just chose (spec §5):
    # passing baseline_risk unconditionally let the joint's own (bound-unaware) check
    # pick RD when effect and naive both happened to be RD-valid even though classify
    # fell back to SMD over the CI bound, mixing an RD-derived rr_point against an
    # RD-derived benchmark that used a different domain test than the CI bound did.
    joint = joint_confounding_benchmark(
        naive_effect,
        eff,
        baseline_risk=(baseline_risk if use_rr else None),
        outcome_std=outcome_std,
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
            f"E-value {e_point:.2f}). Do not act on the size of this effect, and treat "
            "its direction as unconfirmed against confounding of that strength."
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
