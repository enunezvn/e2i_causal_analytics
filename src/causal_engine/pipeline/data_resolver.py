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

C-6 introduced this helper (Option B from `.claude/plans/354_dispatch_plan_v1.md`
§2.3 R5). #458 promotes ``estimation_data`` to a first-class field on
``PipelineState`` / ``PipelineInput`` and re-orders the resolver so the
first-class slot wins; legacy keys are still honored for one release with
a ``DeprecationWarning`` describing the path that matched.

Priority order (highest to lowest):

1. ``state["estimation_data"]`` — first-class field (#458, no warning)
2. ``state["data_cache"]["estimation_data"]`` — legacy, warns about ``data_cache``
3. ``state["filters"]["estimation_data"]`` — legacy, warns about ``filters``
4. ``state["filters"]["dataframe"]`` — legacy, warns about ``filters``
5. ``state["filters"]["data"]`` — legacy, warns about ``filters``

The helper returns ``None`` when no DataFrame is found at any of those
sites; it does NOT raise. The caller is responsible for fail-closing
(per CLAUDE.md anti-mocking discipline — never silently substitute).

This module is intentionally tiny: it has zero runtime dependencies
beyond ``pandas`` for the type check, and zero behavioral side effects
beyond the ``DeprecationWarning`` on legacy reads.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, cast

from .state import PipelineState

__all__ = ["resolve_estimation_dataframe"]


# First-class top-level slot (highest priority, no deprecation warning).
_FIRST_CLASS_KEY = "estimation_data"


# Legacy (top_key, sub_key) lookups, in priority order. Each emits a
# ``DeprecationWarning`` mentioning the top-level key that matched so
# callers can grep for the migration site.
_LEGACY_LOOKUP_PATHS: tuple[tuple[str, str], ...] = (
    ("data_cache", "estimation_data"),
    ("filters", "estimation_data"),
    ("filters", "dataframe"),
    ("filters", "data"),
)


def resolve_estimation_dataframe(state: PipelineState) -> Optional[Any]:
    """Resolve the estimation DataFrame from the canonical or legacy slots.

    Returns the first DataFrame found following the priority order
    documented in the module docstring, or ``None`` if no DataFrame is
    present at any site.

    The helper validates that the value at the resolved key is a
    ``pandas.DataFrame`` (duck-typed via ``isinstance``). Non-DataFrame
    values (e.g., the legitimate ``state["filters"]["dowhy_method"]``
    string set by callers passing DoWhy method overrides) are skipped
    silently rather than returned — this is the priority-fallthrough
    mechanism, not a silent substitution: the helper falls through to
    the next path. If no path yields a DataFrame, the helper returns
    ``None``.

    When the resolved DataFrame comes from a legacy ``state[<top>]
    [<sub>]`` path (not the first-class ``state["estimation_data"]``
    field), a ``DeprecationWarning`` is emitted whose message contains
    the matching top-level key name (``data_cache`` or ``filters``).
    The warning is emitted via ``warnings.warn(..., stacklevel=2)`` so
    it points at the caller of this function, not at this line.

    Args:
        state: PipelineState that may carry an estimation DataFrame.

    Returns:
        The first ``pandas.DataFrame`` found at one of the priority paths,
        or ``None`` if no DataFrame is present anywhere.

    Notes:
        - The helper does NOT raise; the caller is responsible for fail-
          closing on ``None`` (per CLAUDE.md anti-mocking discipline).
        - Per #458, callers should now write the DataFrame to the
          first-class ``state["estimation_data"]`` slot. The legacy paths
          remain readable for one release to give Wave-1 wiring time to
          migrate, but each read emits a ``DeprecationWarning``.
    """
    # Lazy import: keeps module-level imports cheap and avoids a hard pandas
    # dependency at type-check time.
    import pandas as pd

    state_dict = cast(dict[str, Any], state)

    # 1. First-class top-level field (#458). Never deprecation-warns.
    first_class = state_dict.get(_FIRST_CLASS_KEY)
    if isinstance(first_class, pd.DataFrame):
        return first_class

    # 2. Legacy nested-dict paths. Each emits a DeprecationWarning whose
    #    message contains the matching top-level key so the test suite and
    #    downstream callers can grep for migration sites.
    for top_key, sub_key in _LEGACY_LOOKUP_PATHS:
        container = state_dict.get(top_key)
        if not isinstance(container, dict):
            continue
        candidate = container.get(sub_key)
        if isinstance(candidate, pd.DataFrame):
            warnings.warn(
                (
                    f"Legacy DataFrame conveyance via state[{top_key!r}]"
                    f"[{sub_key!r}] is deprecated (#458). Write the "
                    f"DataFrame to state['estimation_data'] instead. "
                    f"The legacy path will be removed in a future release."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            return candidate

    return None
