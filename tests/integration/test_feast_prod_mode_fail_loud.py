"""Integration tests: FeastClient production-mode boundary behaviours.

These three scenarios exercise pure-Python boundary behaviours on the
FeastClient instance.  No live Feast registry is needed; FEAST_INTEGRATION=1
is not required.

Scenario A — FeastFallbackError raises in production mode
    When ENVIRONMENT=production and the fallback path is taken (because
    _store=None and _custom_store is not None), the client must raise
    FeastFallbackError rather than silently returning degraded data.

Scenario B — get_feature_freshness returns is_fresh=False on exception path
    When FeastClient is not initialised (_initialized=False, _store=None,
    _custom_store=None), _ensure_initialized raises RuntimeError, which is
    caught by the broad except block inside get_feature_freshness.  The method
    must return is_fresh=False, freshness_status=UNKNOWN (block by default).

Scenario C — ALLOW_STALE_FEAST=1 opt-out flips is_fresh=True
    Same exception path as Scenario B but with ALLOW_STALE_FEAST=1.  The ops
    emergency escape hatch must flip the result to is_fresh=True,
    freshness_status=UNKNOWN.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.feature_store.feast_client import (
    FeastClient,
    FeastFallbackError,
    FeatureFreshness,
    FreshnessStatus,
)


def _make_entity_df() -> pd.DataFrame:
    """Minimal entity DataFrame satisfying get_historical_features validation."""
    return pd.DataFrame(
        {
            "hcp_id": ["abc123"],
            "event_timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        }
    )


@pytest.mark.asyncio
async def test_feast_fallback_error_raises_in_production_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario A: FeastFallbackError must propagate when ENVIRONMENT=production."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    client = FeastClient()

    # Skip the real Feast import / _store assignment in initialize().
    client.initialize = AsyncMock()  # type: ignore[method-assign]

    # Force the fallback dispatch: _store=None means Feast offline store is
    # unavailable; _custom_store=any-non-None triggers the fallback branch in
    # get_historical_features.
    client._initialized = True
    client._store = None
    client._custom_store = MagicMock()

    entity_df = _make_entity_df()

    with pytest.raises(FeastFallbackError) as exc_info:
        await client.get_historical_features(
            entity_df=entity_df,
            feature_refs=["dummy_view:feature_a"],
        )

    error_message = str(exc_info.value).lower()
    assert "production" in error_message, (
        f"Error message must mention 'production' for ops grep-ability; got: {exc_info.value}"
    )
    assert "fallback" in error_message, (
        f"Error message must mention 'fallback' for ops grep-ability; got: {exc_info.value}"
    )


@pytest.mark.asyncio
async def test_get_feature_freshness_returns_false_on_exception_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario B: get_feature_freshness must return is_fresh=False on exception path."""
    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)

    client = FeastClient()

    # Skip real initialize() so the fields stay at their __init__ defaults:
    # _initialized=False, _store=None, _custom_store=None.
    # This causes _ensure_initialized() to raise RuntimeError, which is
    # caught by the broad except in get_feature_freshness.
    client.initialize = AsyncMock()  # type: ignore[method-assign]

    # Confirm defaults: all three must be falsy to trigger the RuntimeError path.
    assert not client._initialized
    assert client._store is None
    assert client._custom_store is None

    result: FeatureFreshness = await client.get_feature_freshness("any_view_name")

    assert result.is_fresh is False, (
        "Expected is_fresh=False when freshness check raises (block by default); "
        f"got is_fresh={result.is_fresh}"
    )
    assert result.freshness_status == FreshnessStatus.UNKNOWN, (
        f"Expected UNKNOWN freshness status on exception path; got {result.freshness_status}"
    )


@pytest.mark.asyncio
async def test_allow_stale_feast_env_flips_freshness_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario C: ALLOW_STALE_FEAST=1 must flip is_fresh=True (ops escape hatch)."""
    monkeypatch.setenv("ALLOW_STALE_FEAST", "1")

    client = FeastClient()

    # Same exception path as Scenario B.
    client.initialize = AsyncMock()  # type: ignore[method-assign]

    assert not client._initialized
    assert client._store is None
    assert client._custom_store is None

    result: FeatureFreshness = await client.get_feature_freshness("any_view_name")

    assert result.is_fresh is True, (
        "Expected is_fresh=True when ALLOW_STALE_FEAST=1 overrides the exception path; "
        f"got is_fresh={result.is_fresh}"
    )
    assert result.freshness_status == FreshnessStatus.UNKNOWN, (
        f"Expected UNKNOWN freshness status (exception path); got {result.freshness_status}"
    )
