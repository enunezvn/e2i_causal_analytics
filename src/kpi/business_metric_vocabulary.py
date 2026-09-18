"""Canonical vocabulary for chat lookups backed by ``business_metrics``.

Recognition, intent routing, and repository filtering all need to agree on
these phrases.  Keeping the KPI id and stored ``metric_name`` beside each
phrase prevents a display name from resolving successfully and then being
queried under a key the table never stores (#2130).
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# (phrase, registry KPI id, business_metrics.metric_name, direct-value lookup)
#
# ``direct-value lookup`` is deliberately false for shorter conversational
# aliases that are safe for recognition in context but too broad to route by
# themselves (for example, "new to brand" could describe a strategy).
_VOCABULARY: Tuple[Tuple[str, str, str, bool], ...] = (
    ("trx", "WS3-BI-005", "trx", True),
    ("total prescription", "WS3-BI-005", "trx", False),
    ("total prescriptions", "WS3-BI-005", "trx", True),
    ("nrx", "WS3-BI-006", "nrx", True),
    ("new prescription", "WS3-BI-006", "nrx", False),
    ("new prescriptions", "WS3-BI-006", "nrx", True),
    ("nbrx", "WS3-BI-007", "nbrx", True),
    ("new to brand", "WS3-BI-007", "nbrx", False),
    ("new to brand prescription", "WS3-BI-007", "nbrx", False),
    ("new to brand prescriptions", "WS3-BI-007", "nbrx", True),
    ("trx share", "WS3-BI-008", "market_share", True),
    ("market share", "WS3-BI-008", "market_share", True),
    ("share of trx", "WS3-BI-008", "market_share", True),
    ("share of total prescription", "WS3-BI-008", "market_share", False),
    ("share of total prescriptions", "WS3-BI-008", "market_share", True),
    ("conversion", "WS3-BI-009", "conversion_rate", False),
    ("conversion rate", "WS3-BI-009", "conversion_rate", True),
)

BUSINESS_METRIC_KPI_ALIASES: Dict[str, str] = {
    phrase: kpi_id for phrase, kpi_id, _stored_name, _routes in _VOCABULARY
}

_STORED_NAMES: Dict[str, str] = {
    phrase: stored_name for phrase, _kpi_id, stored_name, _routes in _VOCABULARY
}


def _normalize_phrase(value: str) -> str:
    """Normalize the same separators KPI recognition treats as word joins."""
    return " ".join(re.sub(r"[_\-/.\u2013\u2014]+", " ", value.strip().lower()).split())


def canonical_business_metric_name(value: str) -> Optional[str]:
    """Return the stored metric key for a supported KPI phrase, if any."""
    canonical = _STORED_NAMES.get(_normalize_phrase(value))
    if canonical is not None:
        return canonical
    # Registry display names append their abbreviation, e.g. "Total
    # Prescriptions (TRx)". The base name is the metric phrase; a parenthetical
    # suffix is presentation metadata, not part of the stored key.
    without_parenthetical = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    return _STORED_NAMES.get(_normalize_phrase(without_parenthetical))


def _route_phrase_source(phrase: str) -> str:
    # Natural and model-produced forms use spaces, hyphens, and underscores
    # interchangeably. Recognition normalizes all three; routing must too.
    return r"[\s_-]+".join(re.escape(part) for part in phrase.split())


KPI_VALUE_LOOKUP_METRIC_PATTERN = (
    "(?:"
    + "|".join(
        _route_phrase_source(phrase)
        for phrase, _kpi_id, _stored_name, routes in sorted(
            _VOCABULARY, key=lambda item: len(item[0]), reverse=True
        )
        if routes
    )
    + ")"
)
