"""Shared self-emission util for recipient training signals (Gap B §5.2).

Each recipient (experiment_monitor, explainer, health_score, resource_optimizer)
calls :func:`emit_recipient_signal` from its generating node, right after it
produces an LLM/template output, to log ONE training row to
``dspy_agent_training_signals`` with ``source_agent=<recipient>``. The
per-recipient optimizer (:mod:`recipient_optimizer`) then reads those rows back
to build its supervised ``dspy.Example`` set — so each recipient optimizes on
its OWN real emitted data, not a hand-authored golden seed.

The row captures the REAL signature inputs, the generated output, and a
deterministic heuristic reward, mirroring ``signal_store.persist_training_signal``
(same table, same best-effort contract). ``source_agent`` is a free-text
VARCHAR(50) (migration 014), so recipient names write without any DDL/enum.

Best-effort: a DB outage or serialization error must NEVER fail an agent run, so
this never raises — it logs and returns ``False`` on any failure.

B1-B4 call this; this module owns the write path so the four recipient packages
do not each reimplement it.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TABLE = "dspy_agent_training_signals"


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def build_recipient_signal_record(
    agent_name: str,
    signature_inputs: Dict[str, Any],
    generated_output: str,
    reward: float,
    template_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Map a recipient emission to a migration-014 row dict.

    Pure: no I/O. Column names match
    ``database/memory/014_dspy_training_signals.sql``. The optimizer's
    signal->example conversion reads ``input_context.signature_inputs`` and
    ``output.generated`` back out, so the shape here is the contract those two
    halves share.
    """
    input_context: Dict[str, Any] = {"signature_inputs": dict(signature_inputs)}
    if template_field is not None:
        input_context["template_field"] = template_field
    return {
        "source_agent": agent_name,
        "input_context": input_context,
        "output": {"generated": generated_output},
        "reward": float(reward),
        "is_training_example": True,
    }


async def emit_recipient_signal(
    agent_name: str,
    signature_inputs: Dict[str, Any],
    generated_output: str,
    reward: float,
    client: Optional[Any] = None,
    template_field: Optional[str] = None,
) -> bool:
    """Emit one recipient training signal. Returns True on success, else False.

    Never raises: a persistence failure must not break the recipient's run.

    Args:
        agent_name: The emitting recipient (``experiment_monitor``, ``explainer``,
            ``health_score``, ``resource_optimizer``). Written to ``source_agent``.
        signature_inputs: The REAL inputs that backed the generated output, keyed
            by the recipient signature's input field names.
        generated_output: The text the recipient produced (the supervised target).
        reward: A deterministic heuristic reward for the output (see
            :mod:`recipient_metrics`).
        client: Optional Supabase client; resolved from the factory if None.
        template_field: Optional originating template field (e.g. ``srm_template``)
            so a multi-signature recipient can disambiguate which field emitted it.
    """
    record = build_recipient_signal_record(
        agent_name=agent_name,
        signature_inputs=signature_inputs,
        generated_output=generated_output,
        reward=reward,
        template_field=template_field,
    )
    try:
        if client is None:
            from src.memory.services.factories import get_supabase_client

            client = await _maybe_await(get_supabase_client())
        if client is None:
            logger.warning("No Supabase client; %s recipient signal not persisted", agent_name)
            return False
        await _maybe_await(client.table(TABLE).insert(record).execute())
        logger.info("Emitted recipient training signal source_agent=%s", agent_name)
        return True
    except Exception as e:  # noqa: BLE001 - emission is best-effort, never fail the run
        logger.error("Failed to emit recipient training signal for %s: %s", agent_name, e)
        return False
