"""
JIT (just-in-time) provenance verification middleware.

Wraps reads of causal_paths, executive_insights, and explain responses
so that callers get a 410 Gone instead of stale data when an ancestor
artifact has been overturned or invalidated.

How it works
------------
- The middleware inspects the response body only on configured route
  prefixes (configurable via env), and only when the JSON body has a
  recognizable insight identifier and type.
- It calls the SQL function ``verify_insight_chain(p_insight_id, p_insight_type)``
  which walks ``insight_edges`` upward and reports the first invalidated
  ancestor (if any).
- If the chain is broken, the middleware replaces the response with a
  410 Gone containing ``{stale_reason, broken_at_type, broken_at_id}``.
- Every verification is logged to ``audit_chain_verification_log``
  with ``verification_method='jit_provenance'``.

Opt-in / caching
----------------
- Header ``X-Verify-Provenance: strict`` forces a fresh check on every
  request.
- Default (header missing or ``lazy``) uses a redis-cached verdict with
  a 1-hour TTL keyed by ``(insight_type, insight_id)``. The cache is
  invalidated by ``invalidator.cascade_invalidate`` via the
  ``invalidation:e2i:{brand}`` pub/sub channel — see consumers in
  per-agent ``memory_hooks.invalidate_cache``.

Performance
-----------
- Only fires on configured routes (see ``INSIGHT_VERIFIER_PATHS`` env).
- For each route, the JSON parse happens only once; if the body has no
  insight identifier, no DB call is made.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Default path prefixes the verifier inspects. Override via env.
_DEFAULT_PATHS = "/api/causal,/api/explain,/api/executive-insights"
INSIGHT_VERIFIER_PATHS = [
    p.strip()
    for p in os.environ.get("INSIGHT_VERIFIER_PATHS", _DEFAULT_PATHS).split(",")
    if p.strip()
]

CACHE_TTL_SECONDS = int(os.environ.get("INSIGHT_VERIFIER_CACHE_TTL", "3600"))


def _extract_insight_id(body: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Best-effort lookup of (insight_type, insight_id) from a JSON response.

    Recognized shapes:
      {"insight_id": "...", "insight_type": "executive_insight", ...}
      {"path_id": "...", ...}            -> causal_path
      {"trigger_id": "...", ...}         -> trigger
      {"data": {...}} unwrap one level
    """
    if not isinstance(body, dict):
        return None
    if "data" in body and isinstance(body["data"], dict):
        body = body["data"]

    if "insight_id" in body and "insight_type" in body:
        return body["insight_type"], body["insight_id"]
    if "path_id" in body:
        return "causal_path", body["path_id"]
    if "trigger_id" in body and "patient_id" in body:
        return "trigger", body["trigger_id"]
    if "insight_id" in body and "narrative" in body:
        return "executive_insight", body["insight_id"]
    return None


class InsightVerifierMiddleware(BaseHTTPMiddleware):
    """JIT provenance gate on configured GET routes."""

    def __init__(self, app, paths: Optional[list] = None):
        super().__init__(app)
        self.paths = paths or INSIGHT_VERIFIER_PATHS

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if request.method != "GET" or not any(path.startswith(p) for p in self.paths):
            return cast(Response, await call_next(request))

        response = cast(Response, await call_next(request))

        # Only inspect successful JSON responses.
        ct = response.headers.get("content-type", "")
        if response.status_code != 200 or "application/json" not in ct:
            return response

        # call_next() returns a _StreamingResponse internally; body_iterator
        # exists on it but Starlette's public type is Response which doesn't
        # declare it. Standard middleware pattern; safe to access.
        body_bytes = b""
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            body_bytes += chunk

        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        target = _extract_insight_id(body)
        if not target:
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        insight_type, insight_id = target
        mode = request.headers.get("x-verify-provenance", "lazy").lower()
        try:
            verdict = await verify_insight(insight_type, insight_id, strict=(mode == "strict"))
        except Exception:
            logger.exception("insight verifier failed; passing through original response")
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        if not verdict.get("is_valid", True):
            return JSONResponse(
                status_code=410,
                content={
                    "error": "insight_stale",
                    "stale_reason": verdict.get("reason"),
                    "broken_at_type": verdict.get("broken_at_type"),
                    "broken_at_id": verdict.get("broken_at_id"),
                    "insight_type": insight_type,
                    "insight_id": insight_id,
                },
            )

        return Response(
            content=body_bytes,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


# ----------------------------------------------------------------------------
# Verification core — exposed as a function so other code paths can call it
# directly without going through HTTP middleware.
# ----------------------------------------------------------------------------


async def verify_insight(
    insight_type: str,
    insight_id: str,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Verify an insight's provenance chain. Returns a dict with at least
    ``is_valid``. On failure, includes ``reason``, ``broken_at_type``,
    ``broken_at_id``.

    Caches verdicts for ``CACHE_TTL_SECONDS`` unless strict=True.
    """
    cache_key = f"jit_provenance:{insight_type}:{insight_id}"

    if not strict:
        cached = await _get_cached_verdict(cache_key)
        if cached is not None:
            return cached

    verdict = await _call_verify_insight_chain(insight_type, insight_id)
    await _set_cached_verdict(cache_key, verdict)
    await _log_verification(insight_type, insight_id, verdict, strict)
    return verdict


async def _call_verify_insight_chain(insight_type: str, insight_id: str) -> Dict[str, Any]:
    """Call the SQL RPC and shape its single-row result into a dict."""
    from src.memory.services.factories import get_supabase_client

    client = get_supabase_client()
    try:
        rpc = client.rpc(
            "verify_insight_chain",
            {"p_insight_id": insight_id, "p_insight_type": insight_type},
        )
        result = rpc.execute()
        rows = result.data or []
        if not rows:
            return {"is_valid": True, "depth_walked": 0}
        row = rows[0]
        return {
            "is_valid": bool(row.get("is_valid")),
            "broken_at_type": row.get("broken_at_type"),
            "broken_at_id": row.get("broken_at_id"),
            "reason": row.get("reason"),
            "depth_walked": row.get("depth_walked"),
        }
    except Exception as exc:
        logger.exception(f"verify_insight_chain RPC failed for {insight_type}:{insight_id}")
        # Conservative: refuse to vouch when we can't verify.
        return {
            "is_valid": False,
            "reason": f"verification RPC error: {exc}",
            "broken_at_type": insight_type,
            "broken_at_id": insight_id,
        }


async def _get_cached_verdict(key: str) -> Optional[Dict[str, Any]]:
    try:
        from src.memory.services.factories import get_redis_client

        redis = get_redis_client()
        raw = await redis.get(key)
        if raw:
            verdict: Dict[str, Any] = json.loads(raw)
            return verdict
    except Exception:
        return None
    return None


async def _set_cached_verdict(key: str, verdict: Dict[str, Any]) -> None:
    try:
        from src.memory.services.factories import get_redis_client

        redis = get_redis_client()
        await redis.set(key, json.dumps(verdict), ex=CACHE_TTL_SECONDS)
    except Exception:
        # Cache failures are silent — the next call just re-runs the RPC.
        pass


async def _log_verification(
    insight_type: str,
    insight_id: str,
    verdict: Dict[str, Any],
    strict: bool,
) -> None:
    """Append to audit_chain_verification_log for regulatory traceability."""
    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
        client.table("audit_chain_verification_log").insert(
            {
                "entries_verified": int(verdict.get("depth_walked") or 0),
                "chain_valid": bool(verdict.get("is_valid", True)),
                "verification_method": "jit_provenance",
                "verification_notes": json.dumps(
                    {
                        "insight_type": insight_type,
                        "insight_id": insight_id,
                        "strict": strict,
                        "reason": verdict.get("reason"),
                    }
                ),
            }
        ).execute()
    except Exception:
        # Log failures are non-fatal — verifier behavior must not depend on it.
        logger.debug("jit verifier: audit log insert failed", exc_info=True)
