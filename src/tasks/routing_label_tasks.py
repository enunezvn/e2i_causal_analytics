"""Routing-label Celery task — closes the classification_logs feedback gap (#1341 Phase 1).

``classification_logs`` (PR #1330) records every ClassificationPipeline
decision but was write-only: ``was_correct`` stayed NULL forever, so
``v_classification_accuracy`` had nothing to aggregate and no component
consumed routing telemetry. This nightly task labels those rows from
live-traffic outcome signals, strongest first:

1. **Explicit feedback** — ``chatbot_message_feedback`` thumbs matched on
   session + query text. ``thumbs_up`` confirms the dispatch; ``thumbs_down``
   is only judge CONTEXT — a bad answer does not by itself prove bad routing,
   so it must never auto-write ``was_correct=False``.
2. **Implicit outcome** — ``chatbot_analytics`` on the same session
   (nearest turn in time). ``user_satisfied=true`` confirms; errors and
   failed tools become judge context + priority, for the same reason.
3. **LLM judge** — capped per run (token spend, droplet heavy-compute
   capacity), abstains below a confidence floor, and never revisits a row
   it already judged (``feedback_notes`` doubles as the visited marker).

Routing behavior is never mutated here: the task produces labels and
metrics inputs only (#1341 keeps authority changes human-gated).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

VALID_PATTERNS = {
    "SINGLE_AGENT",
    "PARALLEL_DELEGATION",
    "TOOL_COMPOSER",
    "CLARIFICATION_NEEDED",
}

# Judge verdicts below this confidence are recorded as abstentions
# (was_correct stays NULL) rather than written as noisy labels.
JUDGE_CONFIDENCE_FLOOR = 0.6

# A chatbot_analytics turn must sit within this window of the classification
# row's created_at to count as the same turn.
ANALYTICS_MATCH_WINDOW_S = 180.0

_FETCH_LIMIT = 500
_SESSION_CHUNK = 100

_FEEDBACK_COLUMNS = "session_id,query_text,rating,comment,created_at"
_ANALYTICS_COLUMNS = (
    "session_id,query_received_at,primary_agent,agents_consulted,"
    "error_occurred,error_type,tools_failed,user_satisfied"
)


def run_async(coro):
    """Run an async coroutine from sync Celery context (mirrors dspy_optimization_tasks)."""
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


def _min_new_rows() -> int:
    return int(os.getenv("ROUTING_LABEL_MIN_NEW_ROWS", "10"))


def _judge_cap_default() -> int:
    return int(os.getenv("ROUTING_LABEL_JUDGE_CAP", "50"))


def _judge_model() -> str:
    return os.getenv("ROUTING_LABEL_JUDGE_MODEL", "claude-haiku-4-5-20251001")


def _lookback_days() -> int:
    return int(os.getenv("ROUTING_LABEL_LOOKBACK_DAYS", "30"))


def _normalize_query(text: Optional[str]) -> str:
    return " ".join((text or "").lower().split())


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def match_feedback(
    row: Dict[str, Any], feedback_rows: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Latest same-session feedback whose query_text matches this classification."""
    query = _normalize_query(row.get("query_text"))
    if not query:
        return None
    matches = [fb for fb in feedback_rows if _normalize_query(fb.get("query_text")) == query]
    if not matches:
        return None
    return max(matches, key=lambda fb: fb.get("created_at") or "")


def match_analytics(
    row: Dict[str, Any], analytics_rows: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Same-session analytics turn nearest in time, within the match window."""
    row_ts = _parse_ts(row.get("created_at"))
    if row_ts is None:
        return None
    best: Optional[Dict[str, Any]] = None
    best_delta = ANALYTICS_MATCH_WINDOW_S
    for entry in analytics_rows:
        entry_ts = _parse_ts(entry.get("query_received_at"))
        if entry_ts is None:
            continue
        delta = abs((entry_ts - row_ts).total_seconds())
        if delta <= best_delta:
            best = entry
            best_delta = delta
    return best


def decide_label(
    row: Dict[str, Any],
    feedback: Optional[Dict[str, Any]],
    analytics: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Pure label decision for one row.

    Returns ``(auto_update, judge_priority)``. ``auto_update`` is the
    apply_label kwargs when an outcome signal confirms the routing; otherwise
    None and the row is a judge candidate at ``judge_priority`` (0 = negative
    feedback, 1 = agent error, 2 = no signal). Negative signals NEVER
    auto-label False — outcome failure does not prove routing failure.
    """
    rating = (feedback or {}).get("rating")
    if rating == "thumbs_up":
        notes = {
            "source": "explicit_feedback",
            "rating": rating,
            "labeled_at": _now_iso(),
        }
        return {"was_correct": True, "feedback_notes": json.dumps(notes)}, 2

    satisfied = (analytics or {}).get("user_satisfied")
    if rating is None and satisfied is True:
        notes = {"source": "implicit_outcome", "labeled_at": _now_iso()}
        return {"was_correct": True, "feedback_notes": json.dumps(notes)}, 2

    if rating == "thumbs_down" or satisfied is False:
        return None, 0
    if analytics and (analytics.get("error_occurred") or analytics.get("tools_failed")):
        return None, 1
    return None, 2


def parse_judge_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse the judge's JSON verdict; None on any structural failure."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("was_correct"), bool):
        return None
    pattern = payload.get("correct_pattern")
    if pattern not in VALID_PATTERNS:
        pattern = None
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "was_correct": payload["was_correct"],
        "correct_pattern": pattern,
        "confidence": min(1.0, max(0.0, confidence)),
        "rationale": str(payload.get("rationale") or "")[:300],
    }


def _judge_prompt(
    row: Dict[str, Any],
    feedback: Optional[Dict[str, Any]],
    analytics: Optional[Dict[str, Any]],
) -> str:
    outcome_lines: List[str] = []
    if analytics:
        if analytics.get("primary_agent"):
            outcome_lines.append(f"- primary_agent: {analytics['primary_agent']}")
        if analytics.get("agents_consulted"):
            outcome_lines.append(f"- agents_consulted: {analytics['agents_consulted']}")
        if analytics.get("error_occurred"):
            outcome_lines.append(
                f"- agent error occurred: {analytics.get('error_type') or 'unknown'}"
            )
        if analytics.get("tools_failed"):
            outcome_lines.append(f"- tools_failed: {analytics['tools_failed']}")
    if feedback:
        line = f"- user feedback: {feedback.get('rating')}"
        if feedback.get("comment"):
            line += f' — "{str(feedback["comment"])[:200]}"'
        outcome_lines.append(line)
    outcome = "\n".join(outcome_lines) or "- (no outcome data recorded)"

    return f"""You are auditing the query-routing layer of a pharmaceutical analytics chatbot.

Routing patterns:
- SINGLE_AGENT: one specialist agent can answer the query.
- PARALLEL_DELEGATION: several independent agents answer, results merged.
- TOOL_COMPOSER: a multi-step plan with dependent tool calls across domains.
- CLARIFICATION_NEEDED: correct ONLY when a reasonable analyst could not infer the intent — genuinely ambiguous queries.

User query: {row.get("query_text", "")}

Classifier decision under audit:
- routing_pattern: {row.get("routing_pattern")}
- target_agents: {row.get("target_agents") or []}
- confidence: {row.get("confidence")}

What actually happened on this turn:
{outcome}

Judge whether the classifier's routing_pattern (and targets, if any) was the correct routing for this query. A poor final answer does not by itself prove wrong routing; judge the routing choice.

Respond with ONLY a JSON object:
{{"was_correct": true|false, "correct_pattern": "SINGLE_AGENT"|"PARALLEL_DELEGATION"|"TOOL_COMPOSER"|"CLARIFICATION_NEEDED"|null, "confidence": 0.0-1.0, "rationale": "<= 40 words"}}
Set correct_pattern only when was_correct is false. If unsure, lower confidence."""


class RoutingJudge:
    """AI-as-judge for routing correctness (mirrors RubricEvaluator's client
    handling: ANTHROPIC_API_KEY-gated, fail-open to unavailable)."""

    def __init__(self, model: Optional[str] = None, max_tokens: int = 300):
        self.model = model or _judge_model()
        self.max_tokens = max_tokens
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client: Optional[Any] = None
        if api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("RoutingJudge: anthropic package not installed; judge disabled")
        else:
            logger.info("RoutingJudge: no ANTHROPIC_API_KEY; judge disabled for this run")

    @property
    def available(self) -> bool:
        return self.client is not None

    def judge(
        self,
        row: Dict[str, Any],
        feedback: Optional[Dict[str, Any]],
        analytics: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": _judge_prompt(row, feedback, analytics)}],
            )
            return parse_judge_response(response.content[0].text)
        except Exception as e:  # noqa: BLE001 — one bad call must not abort the run
            logger.warning("RoutingJudge call failed (fail-open): %s", e)
            return None


async def _fetch_context(
    client: Any, sessions: List[str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Fetch feedback + analytics rows for the sessions, grouped by session_id."""
    feedback: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    analytics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for start in range(0, len(sessions), _SESSION_CHUNK):
        chunk = sessions[start : start + _SESSION_CHUNK]
        for table, columns, sink in (
            ("chatbot_message_feedback", _FEEDBACK_COLUMNS, feedback),
            ("chatbot_analytics", _ANALYTICS_COLUMNS, analytics),
        ):
            try:
                result = (
                    await client.table(table).select(columns).in_("session_id", chunk).execute()
                )
                for entry in result.data or []:
                    sink[entry["session_id"]].append(entry)
            except Exception as e:  # noqa: BLE001 — context is best-effort
                logger.warning("Context fetch from %s failed (fail-open): %s", table, e)
    return feedback, analytics


async def _run_label_cycle(
    task_id: str,
    force: bool = False,
    judge_cap: Optional[int] = None,
    *,
    repo: Optional[Any] = None,
    judge: Optional[Any] = None,
    fetch_context: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute one labeling cycle. repo/judge/fetch_context are injectable for tests."""
    if repo is None:
        from src.memory.services.factories import get_async_supabase_client
        from src.repositories.classification_log import ClassificationLogRepository

        client = await get_async_supabase_client()
        if client is None:
            return {"status": "skipped", "reason": "no supabase client", "task_id": task_id}
        repo = ClassificationLogRepository(client)

    rows = await repo.fetch_unlabeled(lookback_days=_lookback_days(), limit=_FETCH_LIMIT)
    if not force and len(rows) < _min_new_rows():
        return {
            "status": "skipped",
            "reason": f"only {len(rows)} unlabeled rows (< {_min_new_rows()})",
            "unlabeled": len(rows),
            "task_id": task_id,
        }

    sessions = sorted({r["session_id"] for r in rows if r.get("session_id")})
    context_fetcher = fetch_context or _fetch_context
    feedback_by_session, analytics_by_session = await context_fetcher(repo.client, sessions)

    auto_labeled = 0
    judge_queue: List[Tuple[int, Dict[str, Any], Optional[Dict], Optional[Dict]]] = []
    for row in rows:
        session_id = row.get("session_id") or ""
        fb = match_feedback(row, feedback_by_session.get(session_id, []))
        an = match_analytics(row, analytics_by_session.get(session_id, []))
        auto, priority = decide_label(row, fb, an)
        if auto is not None:
            if await repo.apply_label(row["classification_id"], **auto):
                auto_labeled += 1
        elif row.get("feedback_notes") is None:
            # feedback_notes doubles as the visited marker: judged/abstained
            # rows are never re-judged (idempotent nightly runs).
            judge_queue.append((priority, row, fb, an))

    def _queue_key(item: Tuple[int, Dict[str, Any], Any, Any]) -> Tuple[int, float]:
        ts = _parse_ts(item[1].get("created_at"))
        return (item[0], -(ts.timestamp() if ts else 0.0))

    judge_queue.sort(key=_queue_key)

    judged = abstained = judge_errors = 0
    judge = judge if judge is not None else RoutingJudge()
    cap = judge_cap if judge_cap is not None else _judge_cap_default()
    if judge.available:
        for _priority, row, fb, an in judge_queue[:cap]:
            verdict = judge.judge(row, fb, an)
            if verdict is None:
                judge_errors += 1
                continue
            notes = {
                "judge_model": getattr(judge, "model", None),
                "confidence": verdict["confidence"],
                "rationale": verdict["rationale"],
                "labeled_at": _now_iso(),
            }
            if verdict["confidence"] < JUDGE_CONFIDENCE_FLOOR:
                notes["source"] = "llm_judge_abstain"
                if await repo.apply_label(
                    row["classification_id"], feedback_notes=json.dumps(notes)
                ):
                    abstained += 1
                continue
            notes["source"] = "llm_judge"
            update: Dict[str, Any] = {
                "was_correct": verdict["was_correct"],
                "feedback_notes": json.dumps(notes),
            }
            if verdict["was_correct"] is False and verdict["correct_pattern"]:
                update["correct_pattern"] = verdict["correct_pattern"]
            if await repo.apply_label(row["classification_id"], **update):
                judged += 1

    # Phase 2 (#1341): standing safety telemetry over the whole labeled window,
    # AFTER this run's labels land. Emitted into the run summary (immediately
    # visible) and persisted as a per-run snapshot for a queryable time series
    # (fail-open — the labeler degrades to log-only if migration 032 is absent).
    metrics = await _emit_metrics(repo, task_id)

    summary = {
        "status": "completed",
        "unlabeled_fetched": len(rows),
        "auto_labeled": auto_labeled,
        "judge_candidates": len(judge_queue),
        "judged": judged,
        "abstained": abstained,
        "judge_errors": judge_errors,
        "judge_available": judge.available,
        "metrics": metrics,
        "task_id": task_id,
    }
    logger.info("Routing label cycle complete: %s", summary)
    return summary


async def _emit_metrics(repo: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """Compute + persist Phase-2 telemetry; fail-open, returns the metrics dict.

    Reads the labeled window via ``fetch_for_metrics``, aggregates with the pure
    ``compute_run_metrics``, and snapshots it to routing_classifier_metrics. Any
    failure logs a warning and returns None — telemetry must never abort the
    labeler or mutate routing.
    """
    from src.tasks.routing_metrics import compute_run_metrics

    try:
        window_days = _lookback_days()
        rows = await repo.fetch_for_metrics(lookback_days=window_days)
        metrics = compute_run_metrics(rows)
        await repo.record_metrics_snapshot(metrics, task_id=task_id, window_days=window_days)
        return metrics
    except Exception as e:  # noqa: BLE001 — telemetry is best-effort
        logger.warning("Phase-2 metrics emission failed (fail-open): %s", e)
        return None


@celery_app.task(bind=True, name="src.tasks.run_routing_label_cycle")
def run_routing_label_cycle(
    self, force: bool = False, judge_cap: Optional[int] = None
) -> Dict[str, Any]:
    """Nightly labeler for classification_logs (#1341 Phase 1).

    Populates was_correct / correct_pattern / feedback_notes from live-traffic
    outcome signals so v_classification_accuracy aggregates real data. Gated
    on ROUTING_LABEL_MIN_NEW_ROWS unlabeled rows (force=True overrides) with
    an LLM-judge cap per run (judge_cap arg > ROUTING_LABEL_JUDGE_CAP env).
    """
    logger.info(
        "Starting routing label cycle: task %s (force=%s, judge_cap=%s)",
        self.request.id,
        force,
        judge_cap,
    )
    try:
        return cast(Dict[str, Any], run_async(_run_label_cycle(self.request.id, force, judge_cap)))
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise out of task
        logger.error("Routing label cycle failed: task %s — %s", self.request.id, exc)
        return {"status": "failed", "error": str(exc), "task_id": self.request.id}
