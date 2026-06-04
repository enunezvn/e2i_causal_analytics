"""Tests for graph-stream WebSocket publisher-grant brand authorization (authZ).

PR #681 closed the *authentication* gap on ``@router.websocket("/stream")`` —
an anonymous handshake is rejected when auth is enabled. But any *authenticated*
recipient still received EVERY ``episode_added`` broadcast regardless of which
brand grant the publisher (the user who POSTed the episode) holds. The broadcast
fans out to ALL connected clients, filtered only by the client's self-chosen
subscription. That is fail-open *authorization*.

This module covers the publisher-grant filter (Option A): each broadcast carries
the publisher's ``visible_brands`` and is delivered to a recipient only if the
recipient's cached brand grants intersect it (with ``"all"`` matching anything).

The distinction the tests pin down:

* fail-open authN (PR #681): when auth is DISABLED there is no user, so the
  message is *unscoped* (``visible_brands=None``) and delivered to everyone — a
  deliberate dev/test convenience.
* fail-closed authZ: when a message IS scoped (auth on), a recipient with no
  matching brand — including a recipient whose own grants are unknown
  (``None``) — receives NOTHING. The authN fail-open posture must NOT leak into
  authZ.

``visible_brands`` is a ``broadcast()`` PARAMETER only; it is never serialized
to the client, so no request/response model and no frontend Zod schema change.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import auth as auth_dep
from src.api.models.graph import GraphStreamMessage, GraphSubscription
from src.api.routes import graph as graph_route
from src.api.routes.graph import ConnectionManager, get_current_user, router


def _make_message(payload=None, session_id=None) -> GraphStreamMessage:
    """An ``episode_added``-shaped broadcast message (metadata-only payload)."""
    return GraphStreamMessage(
        event_type="episode_added",
        payload=payload
        if payload is not None
        else {
            "episode_id": "ep_1",
            "source": "test",
            "entities_count": 0,
            "relationships_count": 0,
        },
        session_id=session_id,
    )


async def _register(
    manager: ConnectionManager,
    client_id: str,
    *,
    user_brands,
    subscription: GraphSubscription | None = None,
) -> AsyncMock:
    """Register a fake client on the manager and return its ``send_json`` mock."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    await manager.connect(ws, client_id, user_brands=user_brands)
    if subscription is not None:
        manager.set_subscription(client_id, subscription)
    return ws.send_json


# ---------------------------------------------------------------------------
# Core property tests directly on ConnectionManager
# ---------------------------------------------------------------------------


class TestPublisherGrantFilter:
    @pytest.mark.asyncio
    async def test_shared_brand_is_delivered(self):
        manager = ConnectionManager()
        send = await _register(manager, "c1", user_brands=["Kisqali"])
        await manager.broadcast(_make_message(), visible_brands=["Kisqali"])
        send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disjoint_brand_is_not_delivered(self):
        manager = ConnectionManager()
        send = await _register(manager, "c1", user_brands=["Kisqali"])
        await manager.broadcast(_make_message(), visible_brands=["Fabhalta"])
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_admin_recipient_receives_any_scoped_message(self):
        manager = ConnectionManager()
        send = await _register(manager, "c1", user_brands=["all"])
        await manager.broadcast(_make_message(), visible_brands=["Fabhalta"])
        send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_admin_publisher_reaches_any_recipient(self):
        manager = ConnectionManager()
        send = await _register(manager, "c1", user_brands=["Kisqali"])
        await manager.broadcast(_make_message(), visible_brands=["all"])
        send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_recipient_brands_fail_closed_on_scoped_message(self):
        manager = ConnectionManager()
        send = await _register(manager, "c1", user_brands=None)
        await manager.broadcast(_make_message(), visible_brands=["Kisqali"])
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unscoped_message_delivers_to_all_recipients(self):
        manager = ConnectionManager()
        send_kisqali = await _register(manager, "c1", user_brands=["Kisqali"])
        send_unknown = await _register(manager, "c2", user_brands=None)
        send_admin = await _register(manager, "c3", user_brands=["all"])
        send_empty = await _register(manager, "c4", user_brands=[])
        await manager.broadcast(_make_message(), visible_brands=None)
        send_kisqali.assert_awaited_once()
        send_unknown.assert_awaited_once()
        send_admin.assert_awaited_once()
        send_empty.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subscription_filter_composes_with_brand_filter(self):
        """Brand allows, but a subscription entity_types mismatch still blocks."""
        manager = ConnectionManager()
        # Subscribe ONLY to Brand-typed events.
        sub = GraphSubscription(entity_types=["Brand"])
        send = await _register(manager, "c1", user_brands=["Kisqali"], subscription=sub)
        # Payload carries a "type" the subscription does NOT include -> filtered.
        msg = _make_message(payload={"type": "HCP", "episode_id": "ep_1"})
        await manager.broadcast(msg, visible_brands=["Kisqali"])
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_grant_recipient_fails_closed_on_scoped_message(self):
        manager = ConnectionManager()
        send = await _register(manager, "c1", user_brands=[])
        await manager.broadcast(_make_message(), visible_brands=["Kisqali"])
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_caches_brands_and_disconnect_removes_them(self):
        manager = ConnectionManager()
        await _register(manager, "c1", user_brands=["Kisqali"])
        assert manager.connection_brands["c1"] == ["Kisqali"]
        manager.disconnect("c1")
        assert "c1" not in manager.connection_brands


# ---------------------------------------------------------------------------
# Wiring test: add_episode stamps visible_brands from the publisher's grants
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _graphiti_returning_episode():
    """A mock graphiti service whose add_episode returns a minimal result."""
    from unittest.mock import MagicMock

    service = AsyncMock()
    result = MagicMock()
    result.episode_id = "ep_1"
    result.entities_extracted = []
    result.relationships_extracted = []
    service.add_episode = AsyncMock(return_value=result)
    return service


class TestAddEpisodeStampsVisibleBrands:
    def test_authenticated_publisher_brands_are_stamped(self, app, client):
        """A publisher with brands ['Kisqali'] -> broadcast(visible_brands=['Kisqali'])."""
        publisher = {"id": "u1", "app_metadata": {"brands": ["Kisqali"]}}
        app.dependency_overrides[get_current_user] = lambda: publisher
        broadcast = AsyncMock()
        try:
            with (
                patch.object(
                    graph_route,
                    "_get_graphiti_service",
                    new_callable=AsyncMock,
                    return_value=_graphiti_returning_episode(),
                ),
                patch.object(graph_route.manager, "broadcast", broadcast),
            ):
                resp = client.post(
                    "/graph/episodes",
                    json={"content": "hello", "source": "test"},
                )
            assert resp.status_code == 200
        finally:
            app.dependency_overrides.pop(get_current_user, None)
        broadcast.assert_awaited_once()
        assert broadcast.await_args.kwargs.get("visible_brands") == ["Kisqali"]

    def test_no_user_yields_unscoped_broadcast(self, app, client):
        """Auth disabled / no user -> visible_brands=None (unscoped, deliver-all)."""
        app.dependency_overrides[get_current_user] = lambda: None
        broadcast = AsyncMock()
        try:
            with (
                patch.object(
                    graph_route,
                    "_get_graphiti_service",
                    new_callable=AsyncMock,
                    return_value=_graphiti_returning_episode(),
                ),
                patch.object(graph_route.manager, "broadcast", broadcast),
            ):
                resp = client.post(
                    "/graph/episodes",
                    json={"content": "hello", "source": "test"},
                )
            assert resp.status_code == 200
        finally:
            app.dependency_overrides.pop(get_current_user, None)
        broadcast.assert_awaited_once()
        assert broadcast.await_args.kwargs.get("visible_brands") is None


# Keep an explicit reference so linters don't flag the auth_dep import as unused;
# it documents the module under test for readers and mirrors test_graph_ws_auth.
_ = auth_dep
