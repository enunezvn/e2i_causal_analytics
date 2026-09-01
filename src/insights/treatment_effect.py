"""Treatment-effect strategic insight: interpret ONE de-confounded ATE cell."""

from __future__ import annotations

import logging
import math
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class TreatmentEffectInsightSignature(dspy.Signature):
        """Interpret ONE de-confounded average treatment effect for a brand analyst,
        STRICTLY grounded in the provided numbers. Use ONLY the ATE, CI, p-value, n,
        estimator, treatment/outcome, and confounders given; NEVER invent numbers or
        claim any refutation/robustness test was run. State the effect's magnitude and
        direction; judge ACTIONABILITY from whether the 95% CI excludes 0 (robust) or
        straddles 0 (not distinguishable from no effect), using p-value and n as
        supporting evidence; name the confounders adjusted for; and ALWAYS close with
        the caveat that this is a single model-based estimate whose robustness was NOT
        validated (refutation tests were not run).

        Translate the finding into CONCRETE, EXECUTABLE commercial actions grounded in
        the specific treatment and outcome — say HOW, not just THAT it is actionable.
        Name the lever to pull (the treatment), the segment or audience to prioritize,
        and the metric to monitor and how to validate it in-market. Match the action to
        the CI-vs-0 verdict carried in the estimate — read that verdict from the input
        and let it decide the action's nature; do not restate a template of branches.
        Vague filler such as "make data-driven decisions", "leverage the positive
        effect", or "monitor outcomes closely" — without saying WHICH action, for WHOM,
        measured HOW — is NOT acceptable.

        The registry context lists curated directional chains related to this pair —
        SEPARATE knowledge, not evidence for or against this estimate: you may use it
        qualitatively to situate the effect, but NEVER present it as corroboration and
        NEVER attribute numbers to it.

        Write every output as PLAIN PROSE — no markdown syntax: no asterisks,
        no underscore emphasis, no backticks, no # heading markers, no
        bullet-list markers; plain numbered enumeration like "1." is fine."""

        scope: str = dspy.InputField(desc="Cohort + brand for this estimate")
        estimate: str = dspy.InputField(
            desc="ATE [95% CI], p-value, n, estimator, and CI-vs-0 verdict"
        )
        design: str = dspy.InputField(desc="Treatment -> outcome and the confounders adjusted for")
        registry_context: str = dspy.InputField(
            desc="Curated registry chains related to this pair (directional, no figures)"
        )

        interpretation: str = dspy.OutputField(
            desc=(
                "Grounded read of the effect, its actionability, the concrete commercial "
                "action(s) it implies (lever, target segment, metric to watch), and the "
                "robustness caveat"
            )
        )
        key_takeaways: list = dspy.OutputField(
            desc=(
                "3-5 grounded takeaways, each a SPECIFIC action a brand team could execute — "
                "name the lever, the target segment, and the metric to monitor; NOT vague "
                "guidance like 'make data-driven decisions' or 'leverage the effect'"
            )
        )

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    TreatmentEffectInsightSignature = None  # type: ignore[assignment,misc]


def _fmt_num(v: Any, places: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:+.{places}f}"


def _ci_str(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{_fmt_num(lo)}, {_fmt_num(hi)}]"


def _p_str(p: Any) -> str:
    if p is None:
        return "—"
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(pv):
        return "—"
    return "< 0.001" if pv < 0.001 else f"{pv:.3f}"


def _ci_excludes_zero(lo: Any, hi: Any) -> bool | None:
    if lo is None or hi is None:
        return None
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
        return None
    return lo_f > 0.0 or hi_f < 0.0


def build_grounding(
    cohort: str,
    brand: str,
    treatment_var: str,
    outcome_var: str,
    confounders: list[str],
    ate: float,
    ci_lower: float | None,
    ci_upper: float | None,
    p_value: float | None,
    n: int,
    estimator: str | None,
    causal_drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    excludes = _ci_excludes_zero(ci_lower, ci_upper)
    if excludes is None:
        verdict = "no CI available (single-estimator fallback)"
    elif excludes:
        verdict = "95% CI excludes 0 (distinguishable from no effect)"
    else:
        verdict = "95% CI straddles 0 (not distinguishable from no effect)"
    estimate = (
        f"ATE {_fmt_num(ate)} {_ci_str(ci_lower, ci_upper)}, "
        f"p={_p_str(p_value)}, n={n}, estimator={estimator or '—'}; {verdict}"
    )
    design = f"{treatment_var} -> {outcome_var}; adjusted for {', '.join(confounders) or 'none'}"
    # Registry context is digit-free BY DESIGN (format_qualitative_context): a
    # curated effect size rendered next to the fitted ATE would read as an
    # estimate. Kept a separate grounding key so the cache key and fallback
    # treat it as its own dimension.
    from src.insights.causal_context import format_driver_names, format_qualitative_context

    drivers = causal_drivers or []
    registry_context = format_qualitative_context(drivers)
    chips = [
        {"label": "ATE", "value": _fmt_num(ate)},
        {"label": "95% CI", "value": _ci_str(ci_lower, ci_upper)},
        {"label": "p", "value": _p_str(p_value)},
        {"label": "n", "value": str(n)},
    ]
    named = format_driver_names(drivers)
    if named:
        chips.append({"label": "Registry chains", "value": str(len(named))})
    return {
        "scope": f"{cohort} / {brand}",
        "estimate": estimate,
        "design": design,
        "verdict": verdict,
        "registry_context": registry_context,
        "has_registry_context": bool(named),
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}: {g['estimate']}. Design: {g['design']}. "
        "This is a single model-based estimate; its robustness was NOT validated "
        "(refutation tests were not run). "
        + (f"{g['registry_context']} " if g.get("has_registry_context") else "")
        + "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["verdict"], g["design"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        TreatmentEffectInsightSignature,
        scope=g["scope"],
        estimate=g["estimate"],
        design=g["design"],
        registry_context=g["registry_context"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": normalize_list(getattr(pred, "key_takeaways", [])),
        "grounding": g["grounding"],
        "is_fallback": False,
    }
