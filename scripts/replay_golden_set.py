#!/usr/bin/env python3
"""Replay the RAGAS golden QA set through the real chat pipeline.

Purpose (spec: docs/superpowers/specs/2026-07-15-feedback-learning-honest-demo-design.md)
------------------------------------------------------------------------------------------
Sends each of the 30 curated golden questions
(``src.rag.evaluation.get_default_evaluation_dataset``) through the REAL
non-streaming chat endpoint (``POST /api/copilotkit/chat`` -> ``run_chatbot``
-> cognitive pipeline). Every completed turn makes the cognitive workflow
write ~3 genuine reward signals (agent / investigator / summarizer) to
``learning_signals`` with ``is_synthetic=false`` — the real system grading
its real answers. A feedback-learning cycle over the replay window then has
honest material for the /feedback-learning page.

This does NOT run RAGAS metric evaluation (manual-only per incident #504);
it only reuses the dataset's questions.

Provenance: session ids are ``goldset-replay-<YYYYMMDD>-q<NN>`` so replay
turns stay identifiable in chat history. Verify signals landed after a run:

    docker exec supabase-db psql -U postgres -d postgres -c \\
      "SELECT count(*) FROM learning_signals \\
       WHERE created_at > now() - interval '2 hours' AND is_synthetic = false;"

Usage:
    .venv/bin/python scripts/replay_golden_set.py --dry-run
    .venv/bin/python scripts/replay_golden_set.py --limit 2   # smoke, then verify
    .venv/bin/python scripts/replay_golden_set.py             # full 30
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replay_golden_set")


def mint_token() -> str:
    """Mint a JWT via the GoTrue password grant (mirrors sync_goldstd_serving)."""
    su = os.environ["SUPABASE_URL"]
    anon = os.environ["SUPABASE_ANON_KEY"]
    email = os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local")
    pw = os.environ["E2I_ADMIN_PASSWORD"]
    body = json.dumps({"email": email, "password": pw}).encode()
    req = urllib.request.Request(
        f"{su}/auth/v1/token?grant_type=password",
        data=body,
        headers={"apikey": anon, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return str(json.loads(resp.read().decode())["access_token"])


def jwt_sub(token: str) -> str:
    """Decode the JWT payload's ``sub`` claim (the authoritative user id).

    The chat endpoint rejects (403) any body ``user_id`` that disagrees with
    the token identity (``_resolve_chat_identity``), so we send exactly the
    token's subject.
    """
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
    return str(payload["sub"])


def build_chat_payload(query: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Body for POST /api/copilotkit/chat (``ChatRequest``)."""
    return {"query": query, "user_id": user_id, "session_id": session_id}


def send_chat(api_base: str, token: str, payload: Dict[str, Any], timeout: int) -> Tuple[bool, str]:
    """POST one chat turn; fail-soft — returns (ok, detail), never raises."""
    req = urllib.request.Request(
        f"{api_base}/copilotkit/chat",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            ok = bool(body.get("success")) and bool(body.get("response"))
            return ok, f"agent={body.get('agent_name')} len={len(body.get('response') or '')}"
    except HTTPError as exc:
        try:
            detail = exc.read().decode(errors="replace")[:200]
        except Exception:  # noqa: BLE001 - reading an error body can itself fail
            detail = "<unreadable error body>"
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001 - fail-soft wrapper; the loop must survive any turn
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay golden QA questions through the real chat pipeline."
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("E2I_API_BASE", "https://eznomics.site/api"),
        help="API base URL (default: E2I_API_BASE or https://eznomics.site/api)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Send only the first N questions (smoke run)"
    )
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds between turns")
    parser.add_argument("--timeout", type=int, default=300, help="Per-turn HTTP timeout (s)")
    parser.add_argument("--dry-run", action="store_true", help="Print questions; send nothing")
    args = parser.parse_args(argv)

    from src.rag.evaluation import get_default_evaluation_dataset

    samples = get_default_evaluation_dataset()
    if args.limit is not None:
        samples = samples[: args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    if args.dry_run:
        for i, sample in enumerate(samples, 1):
            print(f"[dry-run] goldset-replay-{stamp}-q{i:02d}: {sample.query}")
        print(f"[dry-run] {len(samples)} questions -> {args.api_base}/copilotkit/chat")
        return 0

    token = mint_token()
    user_id = jwt_sub(token)
    sent, failed = 0, 0
    for i, sample in enumerate(samples, 1):
        session_id = f"goldset-replay-{stamp}-q{i:02d}"
        ok, detail = send_chat(
            args.api_base,
            token,
            build_chat_payload(sample.query, user_id, session_id),
            args.timeout,
        )
        if not ok and detail.startswith("HTTP 401"):
            # GoTrue JWTs expire after 3600s (docker/supabase/.env.template)
            # and a full 30-question run can outlive one token — re-mint and
            # retry this question once.
            logger.info("[%d/%d] %s got 401 — re-minting token", i, len(samples), session_id)
            token = mint_token()
            user_id = jwt_sub(token)
            ok, detail = send_chat(
                args.api_base,
                token,
                build_chat_payload(sample.query, user_id, session_id),
                args.timeout,
            )
        if ok:
            sent += 1
            logger.info("[%d/%d] %s OK %s", i, len(samples), session_id, detail)
        else:
            failed += 1
            logger.warning("[%d/%d] %s FAILED %s", i, len(samples), session_id, detail)
        if i < len(samples):
            time.sleep(args.sleep)

    print(f"replay complete: {sent} ok, {failed} failed of {len(samples)}")
    print(
        "verify: docker exec supabase-db psql -U postgres -d postgres -c "
        '"SELECT count(*) FROM learning_signals WHERE created_at > now() - '
        "interval '2 hours' AND is_synthetic = false;\""
    )
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
