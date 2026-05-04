"""Per-fold random_state resolution for the model_trainer agent.

Phase 1 W3-lite Day 3 (shard 17 W3 row Day 3, shard 21 §A audit-table row
"Hardcoded ``random_state=42`` sites"). The W3-lite repeated-splits protocol
(shard 21) requires every node that consumes ``random_state`` to be able to
take a per-fold seed driven by the orchestrator instead of the historical
hard-coded ``42``.

This module ships a single helper used by the three split-touching nodes
named in shard 17 Day 3 (split_loader, hyperparameter_tuner,
model_trainer_node). The Day-4/5 ``RepeatedStratifiedSplitter`` orchestrator
will populate ``state['fold_random_state']`` per fold; until that lands the
helper is a no-op for legacy callers (default fallback preserves the prior
``random_state=42`` behavior).

Resolution precedence::

    state['fold_random_state']  >  state['random_state']  >  fallback (42)

Treating ``0`` as a valid seed is intentional — shard 21 §A.3's seed-derivation
helper may produce zero, and reinterpreting that as "unset" would corrupt
fold 0 of any deterministic chain.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["resolve_fold_random_state"]


def resolve_fold_random_state(state: Mapping[str, Any], *, fallback: int = 42) -> int:
    """Resolve the random_state to use for a single fold-iteration step.

    Args:
        state: LangGraph state dict for the model_trainer agent. May contain
            ``fold_random_state`` (per-fold orchestrator seed) and/or
            ``random_state`` (legacy single-split seed).
        fallback: Value returned when neither key is present. Defaults to 42
            to match the historical hard-coded value.

    Returns:
        The integer seed for the current fold-iteration step.
    """
    fold_seed = state.get("fold_random_state")
    if fold_seed is not None:
        return int(fold_seed)
    legacy_seed = state.get("random_state")
    if legacy_seed is not None:
        return int(legacy_seed)
    return int(fallback)
