"""Redact user query text before it lands in logs or persisted telemetry.

Roughly twenty call sites hand-rolled ``query[:N]`` slices to keep free-text
pharma-KPI queries out of logs; one site (``causal_impact/agent.py``) logged
the full query untruncated at DEBUG and another (``copilotkit.py``) logged it
untruncated at INFO. Routing every such site through :func:`redact_query`
makes the truncation length — and any future PII scrubbing of query text — a
single knob instead of a scattered convention (#1367).

The default cap is 50 characters, the prevailing slice length. Sites that
deliberately keep more context — the ``_llm_classify`` observability log
(80 chars, PR #1366), the tool-composer INFO logs and the persisted
``query_preview``/``query_analyzed`` metadata (100 chars) — pass ``max_len``
explicitly. The mechanism is centralized here; per-site lengths are not
flattened.
"""

from __future__ import annotations

__all__ = ["redact_query"]

_TRUNCATION_MARKER = "..."


def redact_query(query: str | None, max_len: int = 50) -> str:
    """Return ``query`` shortened for safe logging/telemetry.

    ``None`` (and empty string) become ``""``. When the text exceeds
    ``max_len`` it is cut to ``max_len`` characters and a ``...`` marker is
    appended — byte-for-byte the ``query[:N] + "..."`` idiom these call sites
    replaced — so a truncated log line is unchanged from before. A query at or
    under the cap is returned unchanged, with no marker.

    This is the single hook for any future PII scrubbing of query text. Keep it
    dependency-free and cheap: it runs on hot request paths.
    """
    if not query:
        return ""
    if len(query) <= max_len:
        return query
    return query[:max_len] + _TRUNCATION_MARKER
