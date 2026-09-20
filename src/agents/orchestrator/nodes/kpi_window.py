"""KPI-query time-window extraction for deterministic orchestrator evidence."""

from __future__ import annotations

import re
from typing import Dict, Optional

# Longest window phrase (in tokens) offered to the KPI engine's parser.
_WINDOW_MAX_TOKENS = 4

# Explicit range shapes accepted by ``parse_window``. Detect these before the
# permissive n-gram search so an invalid/reversed range cannot be confused with
# an absent window and answered using the engine default.
_WINDOW_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_EXPLICIT_WINDOW_RANGE_RE = re.compile(
    r"(?:\b\d{4}-\d{2}-\d{2}\s*(?:to|\u2013|-)\s*\d{4}-\d{2}-\d{2}\b"
    rf"|\b{_WINDOW_MONTH_NAME}\s*(?:-|to|\u2013)\s*{_WINDOW_MONTH_NAME}\s+\d{{4}}\b)",
    re.IGNORECASE,
)


def window_from_query(query: str) -> Optional[Dict[str, str]]:
    """Return a parsed query window, or raise for an explicit invalid range.

    Candidate token n-grams are delegated to the KPI engine's parser, which
    remains the sole authority on supported window grammar. Single tokens are
    deliberately skipped: otherwise a phrase such as ``top 2000 HCPs`` would
    silently become calendar year 2000.
    """
    from src.services.time_window import WindowParseError, parse_window

    for explicit in _EXPLICIT_WINDOW_RANGE_RE.finditer(query):
        try:
            window = parse_window(explicit.group(0))
        except (WindowParseError, ValueError) as exc:
            raise WindowParseError(f"invalid explicit window: {explicit.group(0)!r}") from exc
        if window is not None:
            return window.as_dict()

    tokens = re.findall(r"[\w'-]+", query.lower())
    for size in range(min(_WINDOW_MAX_TOKENS, len(tokens)), 1, -1):
        for start in range(len(tokens) - size + 1):
            try:
                window = parse_window(" ".join(tokens[start : start + size]))
            except (WindowParseError, ValueError):
                continue
            if window is not None:
                return window.as_dict()
    return None
