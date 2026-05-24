"""Environment-variable diagnostic helpers.

The dominant misleading-log anti-pattern across this codebase is:

    if not os.environ.get("FOO"):
        logger.warning("FOO not configured, using fallback")

That message is wrong whenever:

  (a) ``.env`` *exists* with ``FOO`` but the entry-point never called
      ``load_dotenv()`` — the var IS configured by the operator, but the
      process can't see it.
  (b) ``FOO=""`` was set explicitly (common in docker-compose ``.env``
      passthrough) — distinct from unset.
  (c) ``FOO`` is set in the parent shell but stripped by a process
      manager (systemd ``EnvironmentFile``, k8s pod-spec override).

Issue #471 (anti-mocking audit) introduces this helper so every
fallback-path log line distinguishes ``<unset>`` from ``<empty-string>``
from ``<set,len=N>`` and surfaces the .env-load-chain ambiguity
explicitly to the operator.

This module has no third-party dependencies (importable from any
``src/`` location without circular-import risk).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["env_state"]


def env_state(var: str, dotenv_path: Path | None = None) -> str:
    """Render the diagnostic state of a single env var.

    Distinguishes three states that ``os.environ.get(var)`` collapses:

      * ``<unset>``        — key absent from ``os.environ``
      * ``<empty-string>`` — key present, value is ``""``
      * ``<set,len=N>``    — key present with non-empty value of length N

    When ``dotenv_path`` is supplied, also reports whether that file
    exists on disk (the *load-chain ambiguity* this helper exists to
    surface: a present .env with an unloaded key looks identical to an
    absent env var to ``os.environ.get``).

    The key VALUE is never returned — only its length — so this is safe
    to use in any log stream.

    Args:
        var: Environment variable name to inspect.
        dotenv_path: Optional path to a ``.env`` file to also report on.

    Returns:
        A short diagnostic string suitable for inclusion in log
        messages. Examples:

          * ``ANTHROPIC_API_KEY=<unset>``
          * ``ANTHROPIC_API_KEY=<empty-string>, dotenv_path=/app/.env(exists)``
          * ``ANTHROPIC_API_KEY=<set,len=108>, dotenv_path=/app/.env(missing)``
    """
    raw = os.environ.get(var)
    if raw is None:
        state = f"{var}=<unset>"
    elif raw == "":
        state = f"{var}=<empty-string>"
    else:
        state = f"{var}=<set,len={len(raw)}>"
    if dotenv_path is not None:
        state += f", dotenv_path={dotenv_path}({'exists' if dotenv_path.exists() else 'missing'})"
    return state
