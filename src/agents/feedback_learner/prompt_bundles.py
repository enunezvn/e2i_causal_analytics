"""Durable PromptBundle install path for recipient agents (audit F2 install-half).

A PromptBundle holds optimized `.format()` templates for one recipient agent.
It is produced by the per-recipient optimizer (Shard 09) and installed into the
live recipient via its update_optimized_prompts() method. Persisted to
./optimized_prompts/<agent>/latest.json so all processes on the droplet share it.

Until F2 is wired, recipients always served constructor defaults because
update_optimized_prompts() had zero production callers; this module is the
production caller (invoked at app startup and after each optimization cycle).
"""

from __future__ import annotations

import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BUNDLE_ROOT = "optimized_prompts"

# agent_name -> "module_path:factory_callable" for the recipient's singleton.
RECIPIENT_FACTORIES: Dict[str, str] = {
    "experiment_monitor": "src.agents.experiment_monitor.dspy_integration:get_experiment_monitor_dspy_integration",
    "resource_optimizer": "src.agents.resource_optimizer.dspy_integration:get_resource_optimizer_dspy_integration",
    "explainer": "src.agents.explainer.dspy_integration:get_explainer_dspy_integration",
    "health_score": "src.agents.health_score.dspy_integration:get_health_score_dspy_integration",
}


def _bundle_path(agent_name: str, root: str = BUNDLE_ROOT) -> Path:
    return Path(root) / agent_name / "latest.json"


def save_prompt_bundle(
    agent_name: str,
    templates: Dict[str, str],
    score: float,
    version: Optional[str] = None,
    root: str = BUNDLE_ROOT,
) -> str:
    """Persist a PromptBundle for an agent; returns the file path."""
    path = _bundle_path(agent_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "agent_name": agent_name,
        "templates": templates,
        "score": score,
        "version": version or datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)
    logger.info("Saved prompt bundle for %s -> %s", agent_name, path)
    return str(path)


def load_prompt_bundle(agent_name: str, root: str = BUNDLE_ROOT) -> Optional[Dict[str, Any]]:
    """Load the latest PromptBundle for an agent, or None if absent/invalid."""
    path = _bundle_path(agent_name, root)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read prompt bundle for %s: %s", agent_name, e)
        return None


def _resolve_factory(agent_name: str):
    ref = RECIPIENT_FACTORIES.get(agent_name)
    if not ref:
        return None
    module_path, func_name = ref.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def install_prompt_bundle(agent_name: str, root: str = BUNDLE_ROOT) -> bool:
    """Install the latest bundle into the recipient's live singleton. Best-effort."""
    bundle = load_prompt_bundle(agent_name, root)
    if bundle is None:
        return False
    factory = _resolve_factory(agent_name)
    if factory is None:
        logger.warning("No recipient factory registered for %s", agent_name)
        return False
    try:
        integration = factory()
        integration.update_optimized_prompts(
            prompts=bundle.get("templates", {}),
            optimization_score=float(bundle.get("score", 0.0)),
        )
        logger.info("Installed prompt bundle %s into %s", bundle.get("version"), agent_name)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to install prompt bundle for %s: %s", agent_name, e)
        return False


def install_all_prompt_bundles(root: str = BUNDLE_ROOT) -> Dict[str, bool]:
    """Install latest bundles for every registered recipient. Never raises."""
    return {agent: install_prompt_bundle(agent, root) for agent in RECIPIENT_FACTORIES}
