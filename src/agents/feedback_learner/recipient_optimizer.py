"""Per-recipient prompt optimizer (audit F2 producer-half — follow-on, Shard 09).

Optimizes each recipient's DSPy signatures and materializes improved .format()
templates into a PromptBundle (Shard 07). Materialization is placeholder-safe:
the recipient code calls .format(**kwargs), so every original {placeholder} must
survive or the recipient will raise KeyError at runtime.

OPEN DESIGN DECISION (see 09-followon-per-recipient-optimizer.md): recipients
consume prompts but most do not emit training signals, so the optimizer needs a
supervised example source. This module is parameterized by an `example_provider`
so the data-source decision (self-emission / shared pool / golden seed) is
pluggable; it DEFAULTS to a small golden seed set (recipient_seeds.py) so the
path is runnable + testable today. Swap in real self-emission later.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]*)?\}")

# template field -> DSPy signature name on the recipient's dspy_integration module,
# used by the live optimizer to know which signature backs which .format() template.
RECIPIENT_SIGNATURE_FIELDS: Dict[str, Dict[str, str]] = {
    "experiment_monitor": {
        "srm_template": "SRMDescriptionSignature",
        "summary_template": "MonitorSummarySignature",
        "alert_template": "AlertGenerationSignature",
    },
}


def extract_placeholders(template: str) -> Set[str]:
    """Return the set of field names referenced by a .format() template."""
    return set(_PLACEHOLDER_RE.findall(template))


def validate_materialized(original: str, candidate: str) -> bool:
    """A materialized template is valid only if it keeps ALL original placeholders."""
    return extract_placeholders(original).issubset(extract_placeholders(candidate))


def materialize_template(current_template: str, improved_instruction: str) -> str:
    """Produce an improved template that preserves every placeholder.

    Conservative default: prepend the optimized guidance as a leading clause
    while keeping the original body verbatim (so all placeholders survive). A
    future version may use the LM to rewrite the body, gated by
    validate_materialized().
    """
    guidance = improved_instruction.strip().rstrip(".")
    candidate = f"{guidance}. {current_template}" if guidance else current_template
    if not validate_materialized(current_template, candidate):
        # Never ship a template that dropped a placeholder.
        return current_template
    return candidate


def produce_bundle_from_instructions(
    agent_name: str,
    current_templates: Dict[str, Any],
    instructions: Dict[str, str],
    score: float,
) -> str:
    """Materialize improved templates for fields we have instructions for; save a bundle."""
    from .prompt_bundles import save_prompt_bundle

    new_templates: Dict[str, str] = {}
    for field, current in current_templates.items():
        if not isinstance(current, str) or not field.endswith("_template"):
            continue
        instr = instructions.get(field)
        new_templates[field] = materialize_template(current, instr) if instr else current
    return save_prompt_bundle(agent_name, templates=new_templates, score=score)


def _current_templates(agent_name: str) -> Dict[str, str]:
    """Pull the recipient's current default templates via its prompts dataclass."""
    from .prompt_bundles import _resolve_factory

    factory = _resolve_factory(agent_name)
    if factory is None:
        return {}
    integration = factory()
    prompts = getattr(integration, "prompts", None)
    if prompts is None or not hasattr(prompts, "to_dict"):
        return {}
    return {k: v for k, v in prompts.to_dict().items() if isinstance(v, str)}


async def optimize_recipient(
    agent_name: str,
    example_provider: Optional[Callable[[str], List[Any]]] = None,
    budget: str = "light",
    fields: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Optimize a recipient's signatures and return {template_field: instruction}.

    For each (template_field, signature) the recipient defines, compile the
    signature with GEPA over example_provider(field), then read the optimized
    instruction from the compiled module's predictor signature. Best-effort per
    field. Returns the instructions dict (possibly empty); the caller materializes
    + saves the bundle via produce_bundle_from_instructions.
    """
    import importlib

    from src.optimization.dspy_lm import ensure_dspy_configured

    field_map = RECIPIENT_SIGNATURE_FIELDS.get(agent_name, {})
    if not field_map:
        logger.info("No optimizable signatures registered for recipient %s", agent_name)
        return {}
    if fields:
        field_map = {k: v for k, v in field_map.items() if k in fields}

    if not ensure_dspy_configured():
        logger.warning("No DSPy LM; cannot optimize recipient %s", agent_name)
        return {}

    if example_provider is None:
        from .recipient_seeds import default_example_provider

        example_provider = default_example_provider(agent_name)

    import dspy

    from src.optimization.gepa import create_gepa_optimizer, get_metric_for_agent

    recipient_mod = importlib.import_module(f"src.agents.{agent_name}.dspy_integration")
    # Normalize the recipient's metric so GEPA's valset Evaluate never hits the
    # plain-dict `int + dict` crash (the StandardAgentGEPAMetric still returns a
    # plain dict; wrapping here is surgical — no blast radius on other agents).
    metric = _wrap_metric(get_metric_for_agent(agent_name))
    lm = getattr(dspy.settings, "lm", None)

    instructions: Dict[str, str] = {}
    for field, sig_name in field_map.items():
        try:
            signature = getattr(recipient_mod, sig_name, None)
            if signature is None:
                continue
            examples = list(example_provider(field) or [])
            if len(examples) < 2:
                logger.info("Too few seed examples for %s.%s; skipping", agent_name, field)
                continue
            split = max(1, int(len(examples) * 0.8))
            trainset, valset = examples[:split], examples[split:] or examples[:1]
            module = dspy.ChainOfThought(signature)
            if lm is not None and hasattr(module, "set_lm"):
                module.set_lm(lm)
            optimizer = create_gepa_optimizer(
                metric=metric, trainset=trainset, valset=valset, auto=budget, seed=42
            )
            optimized = optimizer.compile(module, trainset=trainset, valset=valset)
            instr = _read_instruction(optimized)
            if instr:
                instructions[field] = instr
                logger.info("Optimized recipient %s.%s", agent_name, field)
        except Exception as e:  # noqa: BLE001 - one field failing must not abort the rest
            logger.error("Failed to optimize %s.%s: %s", agent_name, field, e)
    return instructions


def _wrap_metric(metric: Any) -> Callable:
    """Normalize a GEPA metric's return to dspy.Prediction(score, feedback).

    GEPA's valset evaluation sums metric returns via dspy.Evaluate; a plain-dict
    return crashes it (int + dict). Some E2I metrics (e.g. StandardAgentGEPAMetric)
    still return plain dicts. This adapter coerces dict / scalar / Prediction into
    the dspy 3.1 ScoreWithFeedback Prediction contract.
    """
    import dspy

    def wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
        try:
            r = metric(gold, pred, trace, pred_name, pred_trace)
        except TypeError:
            r = metric(gold, pred, trace)
        if isinstance(r, dict):
            return dspy.Prediction(
                score=float(r.get("score", 0.0)), feedback=str(r.get("feedback", ""))
            )
        if isinstance(r, (int, float, bool)):
            return dspy.Prediction(score=float(r), feedback="")
        return r  # already a Prediction / ScoreWithFeedback

    return wrapped


def _read_instruction(module: Any) -> str:
    """Read the optimized instruction text from a compiled DSPy module."""
    try:
        predictors = module.predictors() if hasattr(module, "predictors") else []
        for predictor in predictors:
            sig = getattr(predictor, "signature", None) or getattr(
                predictor, "extended_signature", None
            )
            if sig is not None and getattr(sig, "instructions", None):
                return str(sig.instructions)
    except Exception:  # noqa: BLE001
        pass
    return ""


async def optimize_and_save_recipient(
    agent_name: str,
    example_provider: Optional[Callable[[str], List[Any]]] = None,
    budget: str = "light",
) -> Optional[str]:
    """Optimize a recipient and save a PromptBundle. Returns the bundle path or None."""
    instructions = await optimize_recipient(agent_name, example_provider, budget=budget)
    if not instructions:
        return None
    current = _current_templates(agent_name)
    if not current:
        return None
    # Score is a coarse signal that an optimization ran; the install path uses it
    # for last-write provenance. We keep it modest (0.7) since the golden-seed
    # supervision is weak until real self-emission lands.
    return produce_bundle_from_instructions(
        agent_name, current_templates=current, instructions=instructions, score=0.7
    )
