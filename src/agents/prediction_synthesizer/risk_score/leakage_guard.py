"""Anti-leakage contract for the risk_score model (issue #171).

The CSU initiation target is ``initiated_biologic_180d`` (any Xolair/Dupixent
fill in the 180d post-index window). Any feature that encodes biologic exposure
(brand name, generic name, NDC prefix, HCPCS code) would leak the target. The
converter at ``scripts/convert_optum_rwd.py`` §7.5 already excludes biologic
rows from ``<drug_class>_ever_filled`` features via ``_csu_biologic_mask``; this
module pins that contract by failing loud if a forbidden substring appears in
the training feature matrix column names.

Forbidden substrings (case-insensitive):
    - ``xolair``           — brand
    - ``dupixent``         — brand
    - ``omalizumab``       — Xolair generic
    - ``dupilumab``        — Dupixent generic
    - ``50242``            — Xolair NDC prefix
    - ``00024``            — Dupixent NDC prefix

Reference:
    - Issue #171 §"Anti-leakage integration test (required)"
    - ``scripts/convert_optum_rwd.py:2218-2238`` (``_csu_biologic_mask`` filter)
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Case-insensitive substrings. Any feature whose column name contains any of
# these is considered a target leak for the ``initiated_biologic_180d`` target.
FORBIDDEN_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "xolair",
    "dupixent",
    "omalizumab",
    "dupilumab",
    "50242",
    "00024",
)


class LeakageError(ValueError):
    """Raised when a training feature matrix contains target-leaking columns.

    Carries the offending feature names so callers can log them.
    """

    def __init__(self, leaked: Sequence[str]):
        self.leaked: list[str] = list(leaked)
        super().__init__(
            "Risk-score training feature matrix contains target-leaking columns: "
            + ", ".join(self.leaked)
            + ". Forbidden substrings (case-insensitive): "
            + ", ".join(FORBIDDEN_FEATURE_SUBSTRINGS)
            + ". See src/agents/prediction_synthesizer/risk_score/leakage_guard.py."
        )


def find_leaked_features(
    feature_names: Iterable[str],
    forbidden: Sequence[str] = FORBIDDEN_FEATURE_SUBSTRINGS,
) -> list[str]:
    """Return the subset of ``feature_names`` that match a forbidden substring.

    Matching is case-insensitive on the feature name; forbidden substrings are
    compared against the lower-cased feature name. Order of the returned list
    follows the input iteration order.
    """
    if not forbidden:
        return []
    lowered = tuple(s.lower() for s in forbidden)
    leaked: list[str] = []
    for name in feature_names:
        low = str(name).lower()
        if any(token in low for token in lowered):
            leaked.append(str(name))
    return leaked


def assert_no_leakage_in_features(
    feature_names: Iterable[str],
    forbidden: Sequence[str] = FORBIDDEN_FEATURE_SUBSTRINGS,
) -> None:
    """Raise :class:`LeakageError` if any forbidden substring appears.

    Args:
        feature_names: column names of the training feature matrix.
        forbidden: substrings to match (case-insensitive). Defaults to
            :data:`FORBIDDEN_FEATURE_SUBSTRINGS`.
    """
    leaked = find_leaked_features(feature_names, forbidden=forbidden)
    if leaked:
        raise LeakageError(leaked)
