"""Canonical DataFrame resolver for pipeline executors.

Wave-1 (#354 C-2..C-5) merged with each executor inventing its own
``state[...]`` key for DataFrame conveyance because the locked
``PipelineState`` TypedDict had no in-state DataFrame slot:

| Executor | Wave-1 key |
|---|---|
| DoWhy (C-2) | ``state["filters"]["estimation_data"]`` |
| EconML (C-3) | ``state["data_cache"]["estimation_data"]`` |
| CausalML (C-4) | ``state["filters"]["dataframe"]`` |
| NetworkX (C-5) | (none — symbolic from variable names) |

C-6 introduces this helper (Option B from `.claude/plans/354_dispatch_plan_v1.md`
§2.3 R5) so:

- **New code** (C-7 Surface B, C-8 Surface C, future orchestration) can call
  ``resolve_estimation_dataframe(state)`` and read whichever key the caller
  populated. Callers SHOULD prefer ``data_cache["estimation_data"]`` — the
  canonical path mirrors ``agents/causal_impact/nodes/estimation.py::_get_data``
  so any caller that already supports that flow needs zero changes.
- **Wave-1 executors** keep their existing per-executor reads for
  back-compat. The helper is non-breaking: each executor's failure path
  still raises its own ``ExecutorDataUnavailable``/equivalent when its key
  is missing. Migration from per-executor keys to the canonical key is a
  future, separate workstream (see §3 R5 of the dispatch plan).

Priority order (highest to lowest):

1. ``state["data_cache"]["estimation_data"]`` — canonical
2. ``state["filters"]["estimation_data"]``
3. ``state["filters"]["dataframe"]``
4. ``state["filters"]["data"]``

The helper returns ``None`` when no DataFrame is found at any of those
sites; it does NOT raise. The caller is responsible for fail-closing
(per CLAUDE.md anti-mocking discipline — never silently substitute).

This module is intentionally tiny: it has zero runtime dependencies
beyond ``pandas`` for the type check, and zero behavioral side effects.
"""

from __future__ import annotations

from typing import Any, Optional, cast

from .state import PipelineState

__all__ = ["resolve_estimation_dataframe"]


# Priority-ordered list of (top-level state key, sub-key) tuples to inspect.
# `data_cache.estimation_data` is the canonical path (matches the
# `agents/causal_impact/nodes/estimation.py::_get_data` convention).
_DATA_LOOKUP_PATHS: tuple[tuple[str, str], ...] = (
    ("data_cache", "estimation_data"),
    ("filters", "estimation_data"),
    ("filters", "dataframe"),
    ("filters", "data"),
)


def resolve_estimation_dataframe(state: PipelineState) -> Optional[Any]:
    """Resolve the estimation DataFrame from any of the supported state slots.

    Returns the first DataFrame found following ``_DATA_LOOKUP_PATHS`` priority
    order, or ``None`` if no DataFrame is present at any site.

    The helper validates that the value at the resolved key is a
    ``pandas.DataFrame`` (duck-typed via ``isinstance``). Non-DataFrame
    values (e.g., the legitimate ``state["filters"]["dowhy_method"]`` string
    set by callers passing DoWhy method overrides) are skipped silently
    rather than returned — this is the priority-fallthrough mechanism, not
    a silent substitution: the helper falls through to the next path. If
    no path yields a DataFrame, the helper returns ``None``.

    Args:
        state: PipelineState that may carry an estimation DataFrame.

    Returns:
        The first ``pandas.DataFrame`` found at one of the priority paths,
        or ``None`` if no DataFrame is present anywhere.

    Notes:
        - The helper does NOT raise; the caller is responsible for fail-
          closing on ``None`` (per CLAUDE.md anti-mocking discipline).
        - Wave-1 executors continue to read their per-executor keys
          directly; this helper exists for new code (C-7+) to consume
          state via a single canonical contract.
    """
    # Lazy import: keeps module-level imports cheap and avoids a hard pandas
    # dependency at type-check time.
    import pandas as pd

    state_dict = cast(dict[str, Any], state)

    for top_key, sub_key in _DATA_LOOKUP_PATHS:
        container = state_dict.get(top_key)
        if not isinstance(container, dict):
            continue
        candidate = container.get(sub_key)
        if isinstance(candidate, pd.DataFrame):
            return candidate

    return None
