"""Canonical vocabulary for chat lookups backed by ``business_metrics``.

Recognition, intent routing, and repository filtering all need to agree on
these phrases. Keeping the KPI id and safe metric filter key beside each phrase
prevents a display name from resolving successfully and then being queried
under an unrelated key (#2130). A safe key may intentionally select no current
rows; fail-loud is preferable to conflating two quantities.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# (phrase, registry KPI id, safe metric filter key, direct-value lookup)
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
    # TRx Share and the modeled ``market_share`` rows are different quantities.
    # Current main has no ``trx_share`` rows, so the key safely returns zero
    # rather than a wrong market-share number. After #2159, its canonical-volume
    # handler owns these raw phrases and returns a row with this exact key; keep
    # both behaviors when resolving that rebase.
    ("trx share", "WS3-BI-008", "trx_share", True),
    ("market share", "WS3-BI-008", "market_share", True),
    ("share of trx", "WS3-BI-008", "trx_share", True),
    ("share of total prescription", "WS3-BI-008", "trx_share", False),
    ("share of total prescriptions", "WS3-BI-008", "trx_share", True),
    ("conversion", "WS3-BI-009", "conversion_rate", False),
    ("conversion rate", "WS3-BI-009", "conversion_rate", True),
)

BUSINESS_METRIC_KPI_ALIASES: Dict[str, str] = {
    phrase: kpi_id for phrase, kpi_id, _stored_name, _routes in _VOCABULARY
}

_STORED_NAMES: Dict[str, str] = {
    phrase: stored_name for phrase, _kpi_id, stored_name, _routes in _VOCABULARY
}

# Recognition and filtering must accept the same grammatical inflections.
# Imported by kpi_resolution so this rule cannot drift between the two seams.
KPI_ALIAS_SUFFIX_PATTERN = r"(?:'s|’s|e?s)?"


def _normalize_phrase(value: str) -> str:
    """Normalize the same separators KPI recognition treats as word joins."""
    return " ".join(re.sub(r"[_\-/.\u2013\u2014]+", " ", value.strip().lower()).split())


def canonical_business_metric_name(value: str) -> Optional[str]:
    """Return the stored metric key for a supported KPI phrase, if any."""
    normalized = _normalize_phrase(value)
    canonical = _STORED_NAMES.get(normalized)
    if canonical is not None:
        return canonical
    # Registry display names append their abbreviation, e.g. "Total
    # Prescriptions (TRx)". The base name is the metric phrase; a parenthetical
    # suffix is presentation metadata, not part of the stored key.
    without_parenthetical = _normalize_phrase(re.sub(r"\s*\([^)]*\)\s*$", "", value))
    canonical = _STORED_NAMES.get(without_parenthetical)
    if canonical is not None:
        return canonical
    for phrase, stored_name in sorted(
        _STORED_NAMES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if re.fullmatch(rf"{re.escape(phrase)}{KPI_ALIAS_SUFFIX_PATTERN}", without_parenthetical):
            return stored_name
    return None


def _route_phrase_source(phrase: str) -> str:
    # Natural and model-produced forms use the same separator set accepted by
    # filtering and recognition. Keep the regex source here beside that rule.
    return r"[\s_\-/.–—]+".join(re.escape(part) for part in phrase.split())


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
