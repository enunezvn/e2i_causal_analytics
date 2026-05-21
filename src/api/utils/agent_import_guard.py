"""Agent ImportError guard utility for API route handlers.

Provides a single decision function that determines whether route handlers
that catch ``ImportError`` from agent module imports may silently fall through
to mock-data responses, or must instead raise ``HTTPException(503)``.

Context (issue #429 / F-010-backend)
------------------------------------
Five FastAPI route handlers (``gaps``, ``segments``, ``resource_optimizer``,
``feedback``, ``health_score``) wrap their primary agent execution in
``try: ... except ImportError`` and silently call ``_generate_mock_response``
on import failure. In a production environment this masks a deployment bug
(missing/broken agent module) by returning plausible-looking fabricated data
with a ``warnings: ["Using mock data - X not available"]`` field that the
frontend does not always render.

The guard makes the behavior environment-aware and explicit:

* ``E2I_REQUIRE_AGENT_IMPORT=1`` (truthy) → ALWAYS raise on ImportError
  (production-grade fail-closed).
* ``E2I_REQUIRE_AGENT_IMPORT=0`` (falsy) → ALWAYS allow mock-fallback
  (explicit dev/test opt-in).
* Unset → default depends on ``ENVIRONMENT`` env var: production raises,
  dev/test allows mock-fallback (offline development preserved).

This module deliberately does NOT delete the ``_generate_mock_response``
helpers in the route files — they remain reachable in dev environments where
agent modules may be intentionally absent (offline development without
heavy ML deps). The intent change is: a *consumer* of a production
deployment must NOT receive fabricated data without their explicit opt-in.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_truthy(value: Optional[str]) -> Optional[bool]:
    """Parse an environment variable value as a tristate bool.

    Returns
    -------
    True if value is truthy ("1", "true", ...)
    False if value is falsy ("0", "false", ...)
    None if value is unset or unrecognized
    """
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return None


_KNOWN_DEV_ENVIRONMENTS = {"development", "dev", "test", "testing", "local"}


def should_fail_closed_on_import_error() -> bool:
    """Decide whether a route should fail-closed on agent ImportError.

    Decision order (closed-by-default — codex iter-1 H1 fix):

    1. ``E2I_REQUIRE_AGENT_IMPORT`` (explicit override): truthy → True,
       falsy → False.
    2. ``ENVIRONMENT`` (deployment context): only an EXPLICIT
       ``development``/``dev``/``test``/``testing``/``local`` value permits
       mock-fallback. Any other value — including unset, misspelled,
       ``production``, ``staging``, or empty — fails closed.

    Rationale: missing deployment metadata MUST NOT silently enable
    fabricated data. The old default (unset → development → allow-mock)
    let a misconfigured production deploy serve mock responses.

    Returns
    -------
    bool
        ``True`` if the route MUST raise ``HTTPException(503)`` instead of
        returning fabricated mock data; ``False`` if mock-fallback is
        permitted (dev/test only).
    """
    explicit = _env_truthy(os.environ.get("E2I_REQUIRE_AGENT_IMPORT"))
    if explicit is not None:
        return explicit
    env = os.environ.get("ENVIRONMENT", "").strip().lower()
    return env not in _KNOWN_DEV_ENVIRONMENTS


def raise_503_for_import_error(
    error: ImportError,
    *,
    agent_name: str,
) -> HTTPException:
    """Build the canonical 503 response for an agent ImportError.

    Parameters
    ----------
    error : ImportError
        The original ImportError (its message is included for diagnostics).
    agent_name : str
        Human-readable agent identifier, e.g. ``"Gap Analyzer"``.

    Returns
    -------
    HTTPException
        Suitable for ``raise`` in the route handler.
    """
    logger.error(
        "Agent import failed and mock-fallback disabled: agent=%s error=%s",
        agent_name,
        error,
    )
    return HTTPException(
        status_code=503,
        detail={
            "error": "agent_unavailable",
            "agent": agent_name,
            "message": (
                f"{agent_name} agent module failed to import. "
                "Mock-fallback is disabled in this environment "
                "(E2I_REQUIRE_AGENT_IMPORT=1 or ENVIRONMENT=production). "
                "Set E2I_REQUIRE_AGENT_IMPORT=0 for explicit dev opt-in."
            ),
            "import_error": str(error),
        },
    )


def guard_or_raise(error: ImportError, *, agent_name: str) -> None:
    """Raise HTTPException(503) if fail-closed is required.

    Convenience helper to be called inside ``except ImportError`` blocks.
    Returns silently when mock-fallback is allowed (caller may then proceed
    to its existing ``_generate_mock_response`` path).

    Parameters
    ----------
    error : ImportError
        The original ImportError caught by the handler.
    agent_name : str
        Human-readable agent identifier for the error message / log.

    Raises
    ------
    HTTPException
        With status 503 when ``should_fail_closed_on_import_error()`` is True.
    """
    if should_fail_closed_on_import_error():
        raise raise_503_for_import_error(error, agent_name=agent_name)
    logger.warning(
        "%s agent not available: %s, falling back to mock data "
        "(set E2I_REQUIRE_AGENT_IMPORT=1 to disable in production)",
        agent_name,
        error,
    )
