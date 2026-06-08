"""Per-recipient prompt optimizer (audit F2 producer-half — follow-on, Shard 09).

Optimizes each recipient's DSPy signatures and materializes improved .format()
templates into a PromptBundle (Shard 07). Materialization is placeholder-safe:
the recipient code calls .format(**kwargs), so every original {placeholder} must
survive or the recipient will raise KeyError at runtime.

DATA SOURCE (Gap B §5.4/§5.5): each recipient optimizes on its OWN real emitted
training signals (``source_agent=<recipient>`` in ``dspy_agent_training_signals``,
written by ``recipient_emit.emit_recipient_signal``). The default
``example_provider`` is :func:`signal_example_provider`, which reads those rows
and builds ``dspy.Example``s. If a recipient has fewer than two real examples for
a field, that field is SKIPPED (cold-start) and the recipient keeps its current
default template. Production NEVER falls back to a golden seed set — synthetic
seeds live only as a test fixture (``tests/.../_recipient_seed_fixtures.py``).
A caller may still inject a custom ``example_provider`` (e.g. tests pass the seed
fixture's provider) to exercise the compile/materialize path offline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]*)?\}")

# template field -> DSPy signature name on the recipient's dspy_integration module,
# used by the live optimizer to know which signature backs which .format() template.
# Only fields with a backing signature are listed; fields with no signature
# (e.g. experiment_monitor.enrollment_template) are intentionally omitted.
RECIPIENT_SIGNATURE_FIELDS: Dict[str, Dict[str, str]] = {
    "experiment_monitor": {
        "srm_template": "SRMDescriptionSignature",
        "summary_template": "MonitorSummarySignature",
        "alert_template": "AlertGenerationSignature",
    },
    "explainer": {
        "executive_summary_template": "ExplanationSynthesisSignature",
        "detailed_explanation_template": "ExplanationSynthesisSignature",
        "insight_extraction_template": "InsightExtractionSignature",
        "narrative_section_template": "NarrativeStructureSignature",
    },
    "health_score": {
        "summary_template": "HealthSummarySignature",
        "recommendation_template": "HealthRecommendationSignature",
    },
    "resource_optimizer": {
        "summary_template": "OptimizationSummarySignature",
        "recommendation_template": "AllocationRecommendationSignature",
        "scenario_comparison_template": "ScenarioNarrativeSignature",
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


def _signature_for(agent_name: str, field: str) -> Optional[Any]:
    """Resolve the DSPy signature class backing a recipient's template field."""
    import importlib

    sig_name = RECIPIENT_SIGNATURE_FIELDS.get(agent_name, {}).get(field)
    if not sig_name:
        return None
    try:
        mod = importlib.import_module(f"src.agents.{agent_name}.dspy_integration")
    except Exception as e:  # noqa: BLE001
        logger.warning("Cannot import dspy_integration for %s: %s", agent_name, e)
        return None
    return getattr(mod, sig_name, None)


def _fetch_recipient_signals(agent_name: str, client: Any, min_reward: float, limit: int):
    """Read this recipient's emitted signals via SignalCollectorAdapter (sync wrapper)."""
    from src.rag.memory_adapters import SignalCollectorAdapter

    adapter = SignalCollectorAdapter(supabase_client=client)
    # get_signals_for_optimization is async; run it synchronously so the provider
    # callable matches the existing (field) -> list signature GEPA expects.
    import asyncio

    coro = adapter.get_signals_for_optimization(
        source_agent=agent_name, min_reward=min_reward, limit=limit
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Already inside an event loop (e.g. optimize_recipient is async): run the
        # adapter's blocking client call on a worker thread to avoid re-entrancy.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


def signal_example_provider(
    agent_name: str,
    client: Optional[Any] = None,
    min_reward: float = 0.5,
    limit: int = 1000,
) -> Callable[[str], List[Any]]:
    """Build a provider that reads REAL emitted signals into dspy.Examples.

    Returns ``(template_field) -> list[dspy.Example]``. For each emitted signal
    row (``source_agent=agent_name``), constructs
    ``dspy.Example(**signature_inputs, <gold_field>=generated).with_inputs(
    *signature_input_fields)`` where ``<gold_field>`` is the signature's first
    output field. Rows missing the inputs/generated payload are skipped. If a
    field has fewer than two usable examples, the caller (optimize_recipient)
    skips it (cold-start) — production never falls back to golden seeds.
    """
    try:
        import dspy
    except ImportError:  # pragma: no cover - dspy is a hard dep in prod

        def _empty(_field: str) -> List[Any]:
            return []

        return _empty

    resolved_client = client
    if resolved_client is None:
        try:
            from src.memory.services.factories import get_supabase_client

            maybe = get_supabase_client()
            # get_supabase_client may be sync or async depending on wiring.
            import inspect

            if inspect.isawaitable(maybe):
                import asyncio

                resolved_client = asyncio.run(maybe)
            else:
                resolved_client = maybe
        except Exception as e:  # noqa: BLE001
            logger.warning("No Supabase client for %s signal provider: %s", agent_name, e)
            resolved_client = None

    if resolved_client is None:

        def _empty(_field: str) -> List[Any]:
            logger.info("No client; recipient %s has no real signals (cold-start)", agent_name)
            return []

        return _empty

    try:
        signals = _fetch_recipient_signals(agent_name, resolved_client, min_reward, limit)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to read signals for %s: %s", agent_name, e)
        signals = []

    def provider(field: str) -> List[Any]:
        signature = _signature_for(agent_name, field)
        if signature is None:
            return []
        input_fields = list(getattr(signature, "input_fields", {}).keys())
        output_fields = list(getattr(signature, "output_fields", {}).keys())
        if not input_fields or not output_fields:
            return []
        gold_field = output_fields[0]

        examples: List[Any] = []
        for row in signals:
            ctx = row.get("input_context") or {}
            sig_inputs = ctx.get("signature_inputs") or {}
            generated = (row.get("output") or {}).get("generated")
            if not sig_inputs or not generated:
                continue
            # Only keep examples that carry every input the signature needs.
            kwargs = {k: sig_inputs[k] for k in input_fields if k in sig_inputs}
            if len(kwargs) != len(input_fields):
                continue
            kwargs[gold_field] = generated
            try:
                examples.append(dspy.Example(**kwargs).with_inputs(*input_fields))
            except Exception as e:  # noqa: BLE001
                logger.debug("Skipping malformed signal for %s.%s: %s", agent_name, field, e)
        return examples

    return provider


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
        # PRODUCTION default: each recipient optimizes on its OWN real emitted
        # signals. No golden-seed fallback — a field with <2 real examples is
        # skipped below (cold-start), the recipient keeps its default template.
        example_provider = signal_example_provider(agent_name)

    import dspy

    from src.optimization.gepa import create_gepa_optimizer

    from .recipient_metrics import get_recipient_metric

    recipient_mod = importlib.import_module(f"src.agents.{agent_name}.dspy_integration")
    # Per-recipient deterministic heuristic metric (Gap B §5.1), normalized through
    # _wrap_metric so GEPA's valset Evaluate never hits the plain-dict `int + dict`
    # crash. get_recipient_metric already returns a Prediction, but wrapping is a
    # harmless idempotent safety net (and keeps the contract uniform).
    metric = _wrap_metric(get_recipient_metric(agent_name))
    lm = getattr(dspy.settings, "lm", None)

    instructions: Dict[str, str] = {}
    for field, sig_name in field_map.items():
        try:
            signature = getattr(recipient_mod, sig_name, None)
            if signature is None:
                continue
            examples = list(example_provider(field) or [])
            if len(examples) < 2:
                logger.info(
                    "Too few real examples for %s.%s (cold-start); skipping", agent_name, field
                )
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
    # for last-write provenance. We keep it modest (0.7) since the heuristic
    # supervision over a small real-signal corpus is weak early in the loop.
    return produce_bundle_from_instructions(
        agent_name, current_templates=current, instructions=instructions, score=0.7
    )
