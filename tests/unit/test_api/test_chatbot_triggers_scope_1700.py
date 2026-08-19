"""#1700: triggers query payloads must disclose their true (unfiltered) scope.

Sibling of #1694 (fixed by PR #1695 for ``causal_analysis_tool``): the
triggers path through ``e2i_data_query_tool`` accepts ``brand``, ``region``
and ``time_range``, but the triggers table has no brand/region columns and
``_query_triggers`` applies no filters at all (empty ``filters={}``, ``since``
unused). In the 2026-08-18 certification run, turn A.9-seed passed
``region='Northeast'`` (and ``brand='Kisqali'``, ``time_range='last_90_days'``)
and the synthesis layer answered "two unactioned ... triggers in the region"
over rows that carry no region field — unearned regional scope.

Contract (mirrors #1695's fields exactly):
- ``region_applied: False``, ``time_period_applied: False`` and
  ``brand_applied: False`` on every triggers payload (the booleans are
  unconditional, like #1695's — certified 12/12 payloads carry them);
- a ``scope_note`` synthesis can quote whenever a brand and/or region was
  requested, saying the results are NOT filtered;
- ADDITIVE only: the pre-existing keys (success/query_type/count/data) stay.
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

SINCE = datetime(2026, 5, 20, tzinfo=timezone.utc)


def _mock_repo(triggers):
    repo = MagicMock()
    repo.get_many = AsyncMock(return_value=triggers)
    return repo


@pytest.mark.unit
@pytest.mark.asyncio
async def test_region_requested_discloses_it_was_not_applied_1700():
    """The A.9-seed shape: brand+region requested, neither is a column."""
    repo = _mock_repo([SAMPLE_TRIGGER, {**SAMPLE_TRIGGER, "trigger_id": "TR-0002"}])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand="Kisqali", region="Northeast", since=SINCE, limit=10)
    assert result["success"] is True
    # Pre-existing keys are untouched (ADDITIVE-only contract).
    assert result["query_type"] == "triggers"
    assert result["count"] == 2
    assert result["data"][0]["trigger_id"] == "TR-0001"
    # #1695-mirrored honesty fields.
    assert result["region_applied"] is False
    assert result["brand_applied"] is False
    assert result["time_period_applied"] is False
    assert "Northeast" in result["scope_note"]
    assert "Kisqali" in result["scope_note"]
    assert "NOT" in result["scope_note"]
    # The repo call must stay unfiltered — the fix is payload honesty, not a
    # phantom filter the table cannot honor.
    assert repo.get_many.await_args.kwargs["filters"] == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_filters_requested_keeps_flags_but_no_scope_note_1700():
    repo = _mock_repo([SAMPLE_TRIGGER])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand=None, region=None, since=SINCE, limit=10)
    assert result["success"] is True
    assert result["region_applied"] is False
    assert result["brand_applied"] is False
    assert result["time_period_applied"] is False
    assert "scope_note" not in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_region_only_scope_note_names_the_region_1700():
    repo = _mock_repo([SAMPLE_TRIGGER])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand=None, region="West", since=SINCE, limit=10)
    assert result["region_applied"] is False
    assert "West" in result["scope_note"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scope_note_does_not_overstate_schema_1718():
    """#1718: the note must distinguish column EXISTENCE from filter
    APPLICATION. The certified #1700 wording prevented fabricated scoping but
    planted a false schema claim -- "the triggers table has no brand or region
    columns" -- while returned rows DO carry a ``brand_id`` column (mixed
    Kisqali/Fabhalta/Remibrutinib values in the 2026-08-19 A.9-seed run); only
    region is genuinely absent from the row schema. An answer quoting the note
    verbatim would inherit the false claim, and the wording suppresses
    legitimate per-brand tallies the rows support. Keep the do-not-scope
    instruction; drop the false "no brand columns" claim."""
    repo = _mock_repo([SAMPLE_TRIGGER])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand="Kisqali", region="Northeast", since=SINCE, limit=10)
    note = result["scope_note"]
    # The false schema claim is gone...
    assert "has no brand or region columns" not in note
    # ...replaced by the true claim: the scopes were not APPLIED as filters...
    assert "NOT applied as filters" in note
    # ...while the note stays honest about what the rows DO and DON'T carry.
    assert "brand_id" in note
    assert "region does not exist in this table" in note
    # The do-not-scope instruction survives the rewording.
    assert "Do not present them as specific to any brand or region" in note


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_path_unchanged_1700():
    repo = MagicMock()
    repo.get_many = AsyncMock(side_effect=RuntimeError("boom"))
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await _query_triggers(brand="Kisqali", region="Northeast", since=SINCE, limit=10)
    assert result["success"] is False
    assert result["query_type"] == "triggers"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_e2i_data_query_tool_triggers_branch_carries_honesty_fields_1700():
    """End-to-end through the @tool wrapper, as the A.9-seed call arrived."""
    repo = _mock_repo([SAMPLE_TRIGGER])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.TriggerRepository", return_value=repo),
    ):
        result = await e2i_data_query_tool.ainvoke(
            {
                "query_type": "triggers",
                "brand": "Kisqali",
                "region": "Northeast",
                "time_range": "last_90_days",
            }
        )
    assert result["success"] is True
    assert result["region_applied"] is False
    assert result["brand_applied"] is False
    assert result["time_period_applied"] is False
    assert "Northeast" in result["scope_note"]
