"""#1727: the triggers query must APPLY the requested time window.

``e2i_data_query_tool`` accepts ``time_range`` and computes ``since`` via
``_get_time_filter``, but ``_query_triggers`` ignored it — an
accepted-but-unapplied argument (the same no-op shape as #1714's
``target_agent``). The triggers table DOES carry a usable timestamp
(``trigger_timestamp TIMESTAMPTZ``) and ``TriggerRepository`` already owns the
windowed query shape (``get_recent_triggers``), so the honest fix is to apply
the window, not to keep disclosing that it isn't applied.

Contract after the fix:
- the repository is asked for ``trigger_timestamp >= since`` (via the new
  ``get_triggers_since``);
- ``time_period_applied: True`` and ``window_start: <since ISO>`` on every
  success payload;
- brand/region honesty is UNCHANGED (still not applied, still disclosed —
  the #1700/#1718 guarantees carry over, re-asserted in
  test_chatbot_triggers_scope_1700.py);
- the scope_note no longer claims "this query applies no time filter".

Measured BEFORE state (deployed container, 2026-08-19): last_7_days and
all_time returned identical rows with ``time_period_applied: false``
(docs/demos/results/2026-08-19_wave17_cert/before_triggers_seam_1727.txt).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_tools import _query_triggers, e2i_data_query_tool

SAMPLE_TRIGGER = {
    "trigger_id": "TR-0001",
    "trigger_type": "hcp_engagement_drop",
    "priority_score": 0.91,
    "action_taken": False,
}

SINCE = datetime(2026, 8, 12, tzinfo=timezone.utc)


def _mock_repo(triggers):
    repo = MagicMock()
    repo.get_triggers_since = AsyncMock(return_value=triggers)
    return repo


@pytest.mark.unit
@pytest.mark.asyncio
async def test_time_window_is_applied_1727():
    """The repo must be asked for rows since the window start, and the payload
    must say so."""
    repo = _mock_repo([SAMPLE_TRIGGER])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand=None, region=None, since=SINCE, limit=10)
    assert result["success"] is True
    assert result["time_period_applied"] is True
    assert result["window_start"] == SINCE.isoformat()
    call = repo.get_triggers_since.await_args
    assert call.args[0] == SINCE or call.kwargs.get("since") == SINCE
    assert call.kwargs.get("limit") == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scope_note_drops_the_no_time_filter_claim_1727():
    """Brand/region stay honestly-not-applied, but the note must stop claiming
    the query applies no time filter — it does now."""
    repo = _mock_repo([SAMPLE_TRIGGER])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand="Kisqali", region="Northeast", since=SINCE, limit=10)
    note = result["scope_note"]
    assert "applies no time filter" not in note
    # The #1700/#1718 guarantees survive the rewording.
    assert "NOT applied" in note
    assert "Kisqali" in note
    assert "Northeast" in note
    assert "brand_id" in note
    assert "region does not exist in this table" in note
    assert "Do not present them as specific to any brand or region" in note
    # And the note states the window IS applied, so synthesis can scope time
    # honestly without inferring it from the booleans alone.
    assert "time window" in note.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_wrapper_threads_the_window_end_to_end_1727():
    """Through the @tool wrapper: time_range -> _get_time_filter -> repo."""
    repo = _mock_repo([SAMPLE_TRIGGER])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await e2i_data_query_tool.ainvoke(
            {"query_type": "triggers", "time_range": "last_7_days", "limit": 5}
        )
    assert result["success"] is True
    assert result["time_period_applied"] is True
    since = repo.get_triggers_since.await_args.args[0]
    # last_7_days: the cutoff is ~7 days back from now (UTC).
    age_days = (datetime.now(timezone.utc) - since).days
    assert 6 <= age_days <= 8
    assert result["window_start"] == since.isoformat()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_path_unchanged_1727():
    repo = MagicMock()
    repo.get_triggers_since = AsyncMock(side_effect=RuntimeError("boom"))
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand="Kisqali", region=None, since=SINCE, limit=10)
    assert result["success"] is False
    assert result["query_type"] == "triggers"
