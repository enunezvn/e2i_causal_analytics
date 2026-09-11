"""The API lifespan starts and drains the tool-composer learning loop (spec §5.4).

- With ``TOOL_COMPOSER_LEARNING_LOOP_ENABLED`` set, startup schedules the registry sync and the
  column-allowlist fetch in the background (never awaited on the startup path), and shutdown
  waits up to 5 s for in-flight recording writes, stopping heartbeats first.
- Without it, neither runs: composer runs outside the API (tests, scripts) record nothing.
- The flag is forwarded into the containers: ``x-common-env`` is an explicit whitelist.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.agents.tool_composer.learning_recorder import LEARNING_LOOP_ENV, learning_loop_enabled
from tests.unit.test_api.test_audit_chain_lifespan_wiring import _fake_app, _hermetic_lifespan_io


async def _run_lifespan() -> None:
    from src.api import main

    with _hermetic_lifespan_io(), patch("src.api.main.init_supabase", return_value=None):
        async with main.lifespan(_fake_app()):
            await asyncio.sleep(0)  # let the scheduled startup task run


async def test_enabled_lifespan_starts_the_sync_and_drains_on_shutdown(monkeypatch):
    monkeypatch.setenv(LEARNING_LOOP_ENV, "true")
    startup = AsyncMock()
    drain = AsyncMock(return_value=0)
    with (
        patch("src.agents.tool_composer.registry_sync.learning_loop_startup", new=startup),
        patch("src.agents.tool_composer.learning_recorder.drain", new=drain),
    ):
        await _run_lifespan()
    startup.assert_awaited_once_with()
    drain.assert_awaited_once_with(timeout=5.0, cancel_heartbeats=True)


async def test_disabled_lifespan_neither_syncs_nor_drains(monkeypatch):
    monkeypatch.delenv(LEARNING_LOOP_ENV, raising=False)
    startup = AsyncMock()
    drain = AsyncMock(return_value=0)
    with (
        patch("src.agents.tool_composer.registry_sync.learning_loop_startup", new=startup),
        patch("src.agents.tool_composer.learning_recorder.drain", new=drain),
    ):
        await _run_lifespan()
    startup.assert_not_awaited()
    drain.assert_not_awaited()


async def test_a_failing_drain_does_not_break_shutdown(monkeypatch):
    monkeypatch.setenv(LEARNING_LOOP_ENV, "true")
    with (
        patch("src.agents.tool_composer.registry_sync.learning_loop_startup", new=AsyncMock()),
        patch(
            "src.agents.tool_composer.learning_recorder.drain",
            new=AsyncMock(side_effect=RuntimeError("loop gone")),
        ),
    ):
        await _run_lifespan()


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        ("true", True),
        ("1", True),
        (" YES ", True),
    ],
)
def test_the_flag_is_an_explicit_opt_in(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(LEARNING_LOOP_ENV, raising=False)
    else:
        monkeypatch.setenv(LEARNING_LOOP_ENV, value)
    assert learning_loop_enabled() is expected


def test_the_flag_is_forwarded_into_the_containers_and_on_by_default():
    compose = Path(__file__).resolve().parents[3] / "docker" / "docker-compose.yml"
    common_env = yaml.safe_load(compose.read_text())["x-common-env"]
    assert common_env[LEARNING_LOOP_ENV] == "${TOOL_COMPOSER_LEARNING_LOOP_ENABLED:-true}"
