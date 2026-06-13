"""Shared resolution of the Supabase key for service/internal operations.

Single source of truth for the env-var precedence that internal (non-user-facing)
Supabase clients must use. Several agent connectors historically resolved their
key as ``SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY`` and therefore skipped
``SUPABASE_SERVICE_KEY`` — the name the deployment (docker-compose / .env)
actually sets. The result was a silent downgrade to the **anon** role, which has
no GRANTs on the service_role-only ``ml_*`` tables, so reads failed with
``42501 permission denied`` and the drift/monitoring sweeps became no-ops.

Precedence (matches the canonical resolvers in ``src/memory/services/factories``
and ``src/feature_store/client``):

1. an explicit key passed by the caller (non-empty),
2. ``SUPABASE_SERVICE_ROLE_KEY``,
3. ``SUPABASE_SERVICE_KEY``,
4. ``SUPABASE_ANON_KEY`` — only when ``allow_anon`` (dev/test convenience).

When no service key is configured and ``allow_anon`` is False, this returns
``None`` so the caller can fail closed (raise) rather than silently use anon.
"""

from __future__ import annotations

import os
from typing import Optional


def resolve_supabase_service_key(
    explicit: Optional[str] = None,
    *,
    allow_anon: bool = True,
) -> Optional[str]:
    """Resolve the Supabase key for internal/service-role operations.

    Args:
        explicit: A key supplied directly by the caller. Honored first when
            non-empty; an empty string falls through to env resolution.
        allow_anon: When True, fall back to ``SUPABASE_ANON_KEY`` if no
            service-role key is set (dev/test). When False, return ``None``
            instead so the caller fails closed rather than downgrading to anon.

    Returns:
        The resolved key, or ``None`` if nothing applicable is configured.
    """
    return (
        (explicit or None)
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or (os.getenv("SUPABASE_ANON_KEY") if allow_anon else None)
    )
