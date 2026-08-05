#!/usr/bin/env python3
"""Replay the RAGAS golden QA set through the real pipeline (#1242).

Purpose (spec: docs/superpowers/specs/2026-07-15-feedback-learning-honest-demo-design.md)
------------------------------------------------------------------------------------------
Sends each of the 30 curated golden questions
(``src.rag.evaluation.get_default_evaluation_dataset``) through one of two
REAL endpoints, selected by ``--target``:

``--target cognitive`` (DEFAULT — feeds the Tier-5 feedback learner)
    ``POST /api/cognitive/rag`` runs the full 4-phase cognitive workflow;
    its Phase-4 Reflector writes ~3 genuine reward signals per turn
    (agent / investigator / summarizer) to ``learning_signals`` with
    ``is_synthetic=false`` and ``signal_details->>'domain_signal' =
    'dspy_signal'`` — the ONLY substrate the feedback learner reads
    (``LearningSignalsFeedbackStore``). ~15-18 s per question.

``--target chat`` (chatbot optimization ONLY — the learner never sees it)
    ``POST /api/copilotkit/chat`` -> ``run_chatbot`` persists signals via
    ``ChatbotSignalCollector`` into ``chatbot_training_signals`` — a table
    the Tier-5 feedback learner NEVER reads. Empirically verified 2026-07-15:
    a replay through this path produced 0 ``learning_signals`` rows; the
    same 30 questions through ``/api/cognitive/rag`` produced exactly 90.

Recording replays for the real-pipeline RAGAS gate (#1485)
---------------------------------------------------------
``--record-out PATH`` (``--target cognitive`` only) additionally captures each
turn's REAL generated answer and REALLY-retrieved contexts to a JSON file that
``scripts/run_real_pipeline_ragas.py`` feeds to the frozen gpt-4o judge. Until
#1485 this script checked each response for truthiness and discarded the body,
so nothing downstream could see what the pipeline actually produced.

``--target chat`` cannot be recorded: ``POST /api/copilotkit/chat`` returns no
evidence list, so there are no retrieved contexts to judge and a recorded run
would silently score answers against nothing.

This script still does NOT run RAGAS itself (metric evaluation is manual-only
per incident #504) — judging is a separate, explicit step.

Provenance: conversation/session ids are ``goldset-replay-<YYYYMMDD>-q<NN>``
so replay turns stay identifiable. (Non-UUID conversation ids are fine on
the cognitive path — session_id stays NULL and the conversation id lives in
``signal_details.metadata`` jsonb.) Verify signals landed after a cognitive
run:

    docker exec supabase-db psql -U postgres -d postgres -c \\
      "SELECT count(*) FROM learning_signals \\
       WHERE created_at > now() - interval '2 hours' AND is_synthetic = false;"

After a ``--target chat`` run, check ``chatbot_training_signals`` instead.

Usage:
    .venv/bin/python scripts/replay_golden_set.py --dry-run
    .venv/bin/python scripts/replay_golden_set.py --limit 2   # smoke, then verify
    .venv/bin/python scripts/replay_golden_set.py             # full 30 -> learner
    .venv/bin/python scripts/replay_golden_set.py --target chat  # chatbot training
    # record 12 real replays for the #1485 RAGAS gate:
    .venv/bin/python scripts/replay_golden_set.py --limit 12 \\
        --record-out /tmp/goldset_records.json
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
from pathlib import Path
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


def build_cognitive_payload(query: str, conversation_id: str) -> Dict[str, Any]:
    """Body for POST /api/cognitive/rag (``CognitiveRAGRequest``)."""
    return {"query": query, "conversation_id": conversation_id}


def build_cognitive_record(
    query_id: str,
    query: str,
    conversation_id: str,
    body: Dict[str, Any],
    latency_s: float,
) -> Dict[str, Any]:
    """Capture one cognitive-RAG turn as a judgeable replay record (#1485).

    The record carries what the RAGAS judge needs and nothing invented: the
    generated answer (``response``) and the contexts the run actually
    retrieved (``evidence``). A turn that retrieved no evidence records an
    EMPTY ``contexts`` list — that zero is the measurement (only 3 of 10
    replays retrieved anything on 2026-07-18), and substituting a reference
    stand-in would let ``RAGASEvaluator.evaluate_sample`` score the answer
    against curated evidence it never saw (src/rag/evaluation.py:982).

    Key names mirror the DSPy-lane e2e records so ``build_ragas_samples``
    consumes both unchanged.
    """
    from src.rag.real_pipeline_eval import contexts_from_evidence

    error = body.get("error")
    response_text = body.get("response") or ""
    contexts = contexts_from_evidence(body.get("evidence"))
    return {
        "query_id": query_id,
        "query": query,
        "conversation_id": conversation_id,
        "target": "cognitive",
        "model": "real-pipeline:cognitive",
        "response_text": response_text,
        "answer_chars": len(response_text),
        "contexts": contexts,
        "evidence_count": len(contexts),
        "hop_count": body.get("hop_count", 0),
        "detected_intent": body.get("intent"),
        "routed_agents": body.get("routed_agents") or [],
        "latency_s": latency_s,
        "error": f"CognitiveSearchError: {error}" if error else None,
    }


def build_failed_record(
    query_id: str,
    query: str,
    conversation_id: str,
    error: str,
    latency_s: float,
) -> Dict[str, Any]:
    """Record a turn that never produced a body (HTTP/transport failure).

    Failures are recorded rather than dropped so the run's denominator stays
    honest: the adapter excludes them from judging, and ``summarize_retrieval``
    still counts them against the retrieval-hit rate.
    """
    return {
        "query_id": query_id,
        "query": query,
        "conversation_id": conversation_id,
        "target": "cognitive",
        "model": "real-pipeline:cognitive",
        "response_text": "",
        "answer_chars": 0,
        "contexts": [],
        "evidence_count": 0,
        "hop_count": 0,
        "detected_intent": None,
        "routed_agents": [],
        "latency_s": latency_s,
        "error": error,
    }


def send_chat(
    api_base: str, token: str, payload: Dict[str, Any], timeout: int
) -> Tuple[bool, str, Dict[str, Any]]:
    """POST one chat turn; fail-soft — returns (ok, detail, body), never raises."""
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
            detail = f"agent={body.get('agent_name')} len={len(body.get('response') or '')}"
            return ok, detail, body
    except HTTPError as exc:
        try:
            detail = exc.read().decode(errors="replace")[:200]
        except Exception:  # noqa: BLE001 - reading an error body can itself fail
            detail = "<unreadable error body>"
        return False, f"HTTP {exc.code}: {detail}", {}
    except Exception as exc:  # noqa: BLE001 - fail-soft wrapper; the loop must survive any turn
        return False, f"{type(exc).__name__}: {exc}", {}


def send_cognitive(
    api_base: str, token: str, payload: Dict[str, Any], timeout: int
) -> Tuple[bool, str, Dict[str, Any]]:
    """POST one cognitive-RAG turn; fail-soft — returns (ok, detail, body), never raises.

    Success requires a non-empty ``response`` AND a null ``error`` field
    (``CognitiveRAGResponse`` reports workflow failures in-band via ``error``
    rather than an HTTP status).
    """
    req = urllib.request.Request(
        f"{api_base}/cognitive/rag",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            error = body.get("error")
            ok = bool(body.get("response")) and not error
            if error:
                return False, f"in-band error: {str(error)[:200]}", body
            detail = (
                f"agents={body.get('routed_agents')} hops={body.get('hop_count')} "
                f"evidence={len(body.get('evidence') or [])} "
                f"latency_ms={body.get('latency_ms')}"
            )
            return ok, detail, body
    except HTTPError as exc:
        try:
            detail = exc.read().decode(errors="replace")[:200]
        except Exception:  # noqa: BLE001 - reading an error body can itself fail
            detail = "<unreadable error body>"
        return False, f"HTTP {exc.code}: {detail}", {}
    except Exception as exc:  # noqa: BLE001 - fail-soft wrapper; the loop must survive any turn
        return False, f"{type(exc).__name__}: {exc}", {}


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
    parser.add_argument(
        "--target",
        choices=("cognitive", "chat"),
        default="cognitive",
        help=(
            "cognitive (default): POST /api/cognitive/rag — writes the learning_signals "
            "dspy_signal rows the Tier-5 feedback learner consumes (~15-18s/question). "
            "chat: POST /api/copilotkit/chat — writes chatbot_training_signals only "
            "(chatbot optimization; the feedback learner never reads that table)."
        ),
    )
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds between turns")
    parser.add_argument("--timeout", type=int, default=300, help="Per-turn HTTP timeout (s)")
    parser.add_argument("--dry-run", action="store_true", help="Print questions; send nothing")
    parser.add_argument(
        "--record-out",
        default=None,
        help=(
            "Write each turn's real answer + really-retrieved contexts to this JSON "
            "path for the #1485 real-pipeline RAGAS gate (--target cognitive only). "
            "Feed it to scripts/run_real_pipeline_ragas.py."
        ),
    )
    args = parser.parse_args(argv)

    if args.record_out and args.target != "cognitive":
        # /api/copilotkit/chat returns no evidence list, so a recorded chat run
        # would carry answers with zero contexts and be judged against nothing.
        parser.error("--record-out requires --target cognitive (chat returns no evidence)")

    from src.rag.evaluation import get_default_evaluation_dataset

    samples = get_default_evaluation_dataset()
    if args.limit is not None:
        samples = samples[: args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    endpoint = "/cognitive/rag" if args.target == "cognitive" else "/copilotkit/chat"

    if args.dry_run:
        for i, sample in enumerate(samples, 1):
            print(f"[dry-run] goldset-replay-{stamp}-q{i:02d}: {sample.query}")
        print(f"[dry-run] {len(samples)} questions -> {args.api_base}{endpoint}")
        return 0

    token = mint_token()
    # The chat endpoint rejects (403) a body user_id that disagrees with the
    # token identity; the cognitive endpoint takes no user_id (token-only auth).
    user_id = jwt_sub(token) if args.target == "chat" else ""

    def _send_turn(
        tok: str, uid: str, query: str, conv_id: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if args.target == "cognitive":
            return send_cognitive(
                args.api_base, tok, build_cognitive_payload(query, conv_id), args.timeout
            )
        return send_chat(args.api_base, tok, build_chat_payload(query, uid, conv_id), args.timeout)

    records: List[Dict[str, Any]] = []
    sent, failed = 0, 0
    for i, sample in enumerate(samples, 1):
        session_id = f"goldset-replay-{stamp}-q{i:02d}"
        query_id = f"q{i:02d}"
        turn_start = time.perf_counter()
        ok, detail, body = _send_turn(token, user_id, sample.query, session_id)
        if not ok and detail.startswith("HTTP 401"):
            # GoTrue JWTs expire after 3600s (docker/supabase/.env.template)
            # and a full 30-question run can outlive one token — re-mint and
            # retry this question once.
            logger.info("[%d/%d] %s got 401 — re-minting token", i, len(samples), session_id)
            token = mint_token()
            user_id = jwt_sub(token) if args.target == "chat" else ""
            turn_start = time.perf_counter()
            ok, detail, body = _send_turn(token, user_id, sample.query, session_id)
        if args.record_out:
            latency_s = time.perf_counter() - turn_start
            records.append(
                build_cognitive_record(query_id, sample.query, session_id, body, latency_s)
                if body
                else build_failed_record(query_id, sample.query, session_id, detail, latency_s)
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

    if args.record_out:
        from src.rag.real_pipeline_eval import summarize_retrieval

        retrieval = summarize_retrieval(records)
        out_path = Path(args.record_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(
                {
                    "target": args.target,
                    "api_base": args.api_base,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "retrieval": retrieval,
                    "records": records,
                },
                fh,
                indent=2,
            )
        print(
            f"records: {len(records)} -> {out_path} "
            f"(retrieval hit {retrieval['n_with_contexts']}/{retrieval['n_records']}, "
            f"{retrieval['n_errors']} errored)"
        )
        print(
            f"judge them: .venv/bin/python scripts/run_real_pipeline_ragas.py --records {out_path}"
        )
    if args.target == "cognitive":
        print(
            "verify: docker exec supabase-db psql -U postgres -d postgres -c "
            '"SELECT count(*) FROM learning_signals WHERE created_at > now() - '
            "interval '2 hours' AND is_synthetic = false;\""
        )
    else:
        print(
            "NOTE: --target chat feeds chatbot_training_signals (chatbot optimization) — "
            "the Tier-5 feedback learner never reads that table. Verify with: "
            "docker exec supabase-db psql -U postgres -d postgres -c "
            '"SELECT count(*) FROM chatbot_training_signals WHERE created_at > now() - '
            "interval '2 hours';\""
        )
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
