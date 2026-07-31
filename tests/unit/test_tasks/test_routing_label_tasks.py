"""Tests for the routing-label Celery task (#1341 Phase 1).

The labeler consumes classification_logs live-traffic rows and populates
was_correct / correct_pattern / feedback_notes from outcome signals
(explicit feedback > implicit outcome > capped LLM judge). Routing behavior
is never mutated — the task produces labels only.
"""

import json
from typing import Any, Dict, List, Optional

from src.tasks.routing_label_tasks import (
    JUDGE_CONFIDENCE_FLOOR,
    _run_label_cycle,
    decide_label,
    match_analytics,
    match_feedback,
    parse_judge_response,
)


def _row(**overrides) -> Dict[str, Any]:
    base = {
        "classification_id": "11111111-1111-1111-1111-111111111111",
        "query_text": "What is Kisqali TRx this month?",
        "routing_pattern": "SINGLE_AGENT",
        "target_agents": ["explainer"],
        "confidence": 0.51,
        "session_id": "u1~s1",
        "created_at": "2026-07-29T10:00:00+00:00",
        "feedback_notes": None,
    }
    base.update(overrides)
    return base


def _feedback(**overrides) -> Dict[str, Any]:
    base = {
        "session_id": "u1~s1",
        "query_text": "What is Kisqali TRx this month?",
        "rating": "thumbs_up",
        "comment": None,
        "created_at": "2026-07-29T10:01:00+00:00",
    }
    base.update(overrides)
    return base


def _analytics(**overrides) -> Dict[str, Any]:
    base = {
        "session_id": "u1~s1",
        "query_received_at": "2026-07-29T10:00:05+00:00",
        "primary_agent": "explainer",
        "agents_consulted": ["explainer"],
        "error_occurred": False,
        "error_type": None,
        "tools_failed": None,
        "user_satisfied": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Signal matching
# ---------------------------------------------------------------------------


class TestMatchFeedback:
    def test_matches_on_normalized_query_text(self):
        fb = _feedback(query_text="  what is kisqali trx this month?  ")
        assert match_feedback(_row(), [fb]) is fb

    def test_no_match_on_different_query(self):
        fb = _feedback(query_text="Why did Fabhalta persistence drop?")
        assert match_feedback(_row(), [fb]) is None

    def test_latest_feedback_wins(self):
        older = _feedback(rating="thumbs_up", created_at="2026-07-29T10:01:00+00:00")
        newer = _feedback(rating="thumbs_down", created_at="2026-07-29T11:00:00+00:00")
        assert match_feedback(_row(), [older, newer]) is newer


class TestMatchAnalytics:
    def test_nearest_within_window(self):
        near = _analytics(query_received_at="2026-07-29T10:00:05+00:00")
        far = _analytics(query_received_at="2026-07-29T10:02:00+00:00")
        assert match_analytics(_row(), [far, near]) is near

    def test_outside_window_returns_none(self):
        stale = _analytics(query_received_at="2026-07-29T09:00:00+00:00")
        assert match_analytics(_row(), [stale]) is None


# ---------------------------------------------------------------------------
# Label decision (pure)
# ---------------------------------------------------------------------------


class TestDecideLabel:
    def test_thumbs_up_auto_labels_true(self):
        auto, _ = decide_label(_row(), _feedback(rating="thumbs_up"), None)
        assert auto is not None
        assert auto["was_correct"] is True
        notes = json.loads(auto["feedback_notes"])
        assert notes["source"] == "explicit_feedback"

    def test_user_satisfied_auto_labels_true(self):
        auto, _ = decide_label(_row(), None, _analytics(user_satisfied=True))
        assert auto is not None
        assert auto["was_correct"] is True
        assert json.loads(auto["feedback_notes"])["source"] == "implicit_outcome"

    def test_thumbs_down_is_judge_context_not_auto_false(self):
        # A bad answer does not prove bad routing — thumbs_down must never
        # auto-write was_correct=False; it prioritizes the row for the judge.
        auto, priority = decide_label(_row(), _feedback(rating="thumbs_down"), None)
        assert auto is None
        assert priority == 0

    def test_agent_error_is_judge_context_not_auto_false(self):
        auto, priority = decide_label(
            _row(), None, _analytics(error_occurred=True, error_type="agent_error")
        )
        assert auto is None
        assert priority == 1

    def test_no_signal_is_lowest_judge_priority(self):
        auto, priority = decide_label(_row(), None, None)
        assert auto is None
        assert priority == 2


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_valid_json(self):
        verdict = parse_judge_response(
            '{"was_correct": false, "correct_pattern": "TOOL_COMPOSER",'
            ' "confidence": 0.8, "rationale": "multi-step"}'
        )
        assert verdict == {
            "was_correct": False,
            "correct_pattern": "TOOL_COMPOSER",
            "confidence": 0.8,
            "rationale": "multi-step",
        }

    def test_fenced_json(self):
        verdict = parse_judge_response(
            '```json\n{"was_correct": true, "correct_pattern": null,'
            ' "confidence": 0.9, "rationale": "ok"}\n```'
        )
        assert verdict is not None
        assert verdict["was_correct"] is True

    def test_invalid_pattern_dropped(self):
        verdict = parse_judge_response(
            '{"was_correct": false, "correct_pattern": "NOT_A_PATTERN",'
            ' "confidence": 0.8, "rationale": "x"}'
        )
        assert verdict is not None
        assert verdict["correct_pattern"] is None

    def test_garbage_returns_none(self):
        assert parse_judge_response("I think the routing was fine.") is None

    def test_confidence_clamped(self):
        verdict = parse_judge_response(
            '{"was_correct": true, "correct_pattern": null, "confidence": 7, "rationale": "x"}'
        )
        assert verdict is not None
        assert verdict["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Cycle orchestration (injected fakes; no network, no DB)
# ---------------------------------------------------------------------------


class FakeRepo:
    def __init__(
        self, rows: List[Dict[str, Any]], metrics_rows: Optional[List[Dict[str, Any]]] = None
    ):
        self.client = object()
        self.rows = rows
        self.metrics_rows = metrics_rows if metrics_rows is not None else rows
        self.labels: List[Dict[str, Any]] = []
        self.snapshots: List[Dict[str, Any]] = []

    async def fetch_unlabeled(self, **_kwargs) -> List[Dict[str, Any]]:
        return self.rows

    async def apply_label(self, classification_id: str, **kwargs) -> bool:
        self.labels.append({"classification_id": classification_id, **kwargs})
        return True

    async def fetch_for_metrics(self, **_kwargs) -> List[Dict[str, Any]]:
        return self.metrics_rows

    async def record_metrics_snapshot(self, metrics, *, task_id, window_days) -> bool:
        self.snapshots.append({"metrics": metrics, "task_id": task_id, "window_days": window_days})
        return True


class FakeJudge:
    def __init__(self, verdicts: Optional[Dict[str, Optional[Dict[str, Any]]]] = None):
        self.available = True
        self.verdicts = verdicts or {}
        self.judged: List[str] = []

    def judge(self, row, feedback, analytics) -> Optional[Dict[str, Any]]:
        self.judged.append(row["classification_id"])
        return self.verdicts.get(row["classification_id"])


async def _no_context(_client, _sessions):
    return {}, {}


class TestRunLabelCycle:
    async def test_skips_below_min_rows(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "5")
        repo = FakeRepo([_row()])
        result = await _run_label_cycle(
            "t1",
            force=False,
            judge_cap=None,
            repo=repo,
            judge=FakeJudge(),
            fetch_context=_no_context,
        )
        assert result["status"] == "skipped"
        assert repo.labels == []

    async def test_force_overrides_min_rows(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "5")
        repo = FakeRepo([_row()])
        judge = FakeJudge({_row()["classification_id"]: None})
        result = await _run_label_cycle(
            "t1", force=True, judge_cap=None, repo=repo, judge=judge, fetch_context=_no_context
        )
        assert result["status"] == "completed"

    async def test_auto_label_and_judge_flow(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")
        auto_row = _row(classification_id="a" * 32, session_id="u1~auto")
        judged_row = _row(classification_id="b" * 32, session_id="u1~judge")
        repo = FakeRepo([auto_row, judged_row])
        judge = FakeJudge(
            {
                "b" * 32: {
                    "was_correct": False,
                    "correct_pattern": "TOOL_COMPOSER",
                    "confidence": 0.9,
                    "rationale": "needs composition",
                }
            }
        )

        async def fetch_context(_client, _sessions):
            return {"u1~auto": [_feedback(session_id="u1~auto")]}, {}

        result = await _run_label_cycle(
            "t1", force=False, judge_cap=None, repo=repo, judge=judge, fetch_context=fetch_context
        )
        assert result["status"] == "completed"
        assert result["auto_labeled"] == 1
        assert result["judged"] == 1
        by_id = {label["classification_id"]: label for label in repo.labels}
        assert by_id["a" * 32]["was_correct"] is True
        assert by_id["b" * 32]["was_correct"] is False
        assert by_id["b" * 32]["correct_pattern"] == "TOOL_COMPOSER"

    async def test_judge_abstains_below_floor(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")
        row = _row()
        repo = FakeRepo([row])
        judge = FakeJudge(
            {
                row["classification_id"]: {
                    "was_correct": True,
                    "correct_pattern": None,
                    "confidence": JUDGE_CONFIDENCE_FLOOR - 0.1,
                    "rationale": "unsure",
                }
            }
        )
        result = await _run_label_cycle(
            "t1", force=False, judge_cap=None, repo=repo, judge=judge, fetch_context=_no_context
        )
        assert result["abstained"] == 1
        assert result["judged"] == 0
        (label,) = repo.labels
        assert "was_correct" not in label or label["was_correct"] is None
        assert json.loads(label["feedback_notes"])["source"] == "llm_judge_abstain"

    async def test_visited_rows_never_rejudged(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")
        visited = _row(feedback_notes=json.dumps({"source": "llm_judge_abstain"}))
        repo = FakeRepo([visited])
        judge = FakeJudge()
        result = await _run_label_cycle(
            "t1", force=False, judge_cap=None, repo=repo, judge=judge, fetch_context=_no_context
        )
        assert result["status"] == "completed"
        assert judge.judged == []

    async def test_judge_cap_enforced(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")
        rows = [_row(classification_id=f"{i:032d}") for i in range(5)]
        repo = FakeRepo(rows)
        judge = FakeJudge()
        result = await _run_label_cycle(
            "t1", force=False, judge_cap=2, repo=repo, judge=judge, fetch_context=_no_context
        )
        assert result["status"] == "completed"
        assert len(judge.judged) == 2

    async def test_thumbs_down_judged_before_plain_rows(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")
        plain = _row(classification_id="p" * 32, session_id="u1~plain")
        negative = _row(classification_id="n" * 32, session_id="u1~neg")
        repo = FakeRepo([plain, negative])
        judge = FakeJudge()

        async def fetch_context(_client, _sessions):
            return {"u1~neg": [_feedback(session_id="u1~neg", rating="thumbs_down")]}, {}

        await _run_label_cycle(
            "t1", force=False, judge_cap=1, repo=repo, judge=judge, fetch_context=fetch_context
        )
        assert judge.judged == ["n" * 32]

    async def test_unavailable_judge_leaves_rows_unlabeled(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")
        repo = FakeRepo([_row()])
        judge = FakeJudge()
        judge.available = False
        result = await _run_label_cycle(
            "t1", force=False, judge_cap=None, repo=repo, judge=judge, fetch_context=_no_context
        )
        assert result["status"] == "completed"
        assert result["judged"] == 0
        assert repo.labels == []


# ---------------------------------------------------------------------------
# Phase 2 — metrics emission (#1341)
# ---------------------------------------------------------------------------


class TestPhase2MetricsEmission:
    async def test_summary_carries_metrics_and_snapshot_persisted(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")
        # Labeled window feeding the aggregation (independent of the unlabeled queue).
        metrics_rows = [
            _row(routing_pattern="SINGLE_AGENT", confidence=0.9, was_correct=True),
            _row(routing_pattern="CLARIFICATION_NEEDED", confidence=0.0, was_correct=False),
            _row(routing_pattern="TOOL_COMPOSER", confidence=0.8, was_correct=True),
        ]
        repo = FakeRepo([_row()], metrics_rows=metrics_rows)
        result = await _run_label_cycle(
            "t-metrics",
            force=True,
            judge_cap=None,
            repo=repo,
            judge=FakeJudge(),
            fetch_context=_no_context,
        )
        assert result["status"] == "completed"
        metrics = result["metrics"]
        assert metrics is not None
        assert metrics["total"] == 3
        assert metrics["labeled"] == 3
        assert metrics["overall_accuracy_pct"] == 66.67
        assert metrics["abstention"]["total"] == 1
        # A snapshot row was persisted for the time series.
        assert len(repo.snapshots) == 1
        assert repo.snapshots[0]["task_id"] == "t-metrics"

    async def test_metrics_failure_does_not_abort_cycle(self, monkeypatch):
        monkeypatch.setenv("ROUTING_LABEL_MIN_NEW_ROWS", "1")

        class BadMetricsRepo(FakeRepo):
            async def fetch_for_metrics(self, **_kwargs):
                raise RuntimeError("view missing")

        repo = BadMetricsRepo([_row()])
        result = await _run_label_cycle(
            "t-bad",
            force=True,
            judge_cap=None,
            repo=repo,
            judge=FakeJudge(),
            fetch_context=_no_context,
        )
        # Labeler still completes; metrics degrade to None (fail-open).
        assert result["status"] == "completed"
        assert result["metrics"] is None


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestWiring:
    def test_task_registered_with_celery(self):
        import src.tasks.routing_label_tasks  # noqa: F401
        from src.workers.celery_app import celery_app

        assert "src.tasks.run_routing_label_cycle" in celery_app.tasks

    def test_beat_schedule_entry(self):
        from src.workers.celery_app import celery_app

        entry = celery_app.conf.beat_schedule["routing-label-nightly"]
        assert entry["task"] == "src.tasks.run_routing_label_cycle"
        assert entry["options"]["queue"] == "analytics"
        # Fixed off-peak wall-clock slot (04:30 UTC), not a relative interval:
        # must avoid the droplet's 02:00 backup and Mon-03:00 reseed windows.
        schedule = entry["schedule"]
        assert {4} == getattr(schedule, "hour", None)
        assert {30} == getattr(schedule, "minute", None)
