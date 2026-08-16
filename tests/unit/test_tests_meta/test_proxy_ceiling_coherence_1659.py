"""#1659 — keep the nginx proxy ceiling and the code that must live under it coupled.

The defect this guards against is a *drift* defect. ``router.py`` budgets
``heterogeneous_optimizer`` at 420 000 ms; the nginx location that fronts the
chat SSE stream tolerates 300 s of silence. Neither file mentions the other's
number, so the two were tuned independently until they contradicted.

``proxy_read_timeout`` bounds the SILENT window, not the request duration — it
resets on every byte nginx reads from upstream. Measured 2026-08-16 through the
live host nginx (see ``tests/unit/test_api/test_sse_keepalive_1659.py`` for the
full byte timeline): with no keepalive the silent window equals the ENTIRE turn
(34 395.7 ms client-side vs 34 389.4 ms of summed server-side node wall time),
so the binding constraint was ``total turn wall time < proxy_read_timeout`` —
a constraint no single dispatch budget can honour on its own.

The fix decouples them: ``src/api/utils/sse_keepalive`` bounds the silent window
to a constant. These tests make sure the constant stays anchored to the nginx
value it mirrors, so editing one side without the other fails CI.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.api.utils.sse_keepalive import (
    PROXY_READ_TIMEOUT_SECONDS,
    SSE_KEEPALIVE_INTERVAL_SECONDS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
HOST_NGINX_CONF = REPO_ROOT / "docker" / "nginx" / "host-nginx.conf"

# Locations that serve the chat SSE surfaces. ``/api/`` fronts
# ``POST /api/copilotkit/chat/stream`` and ``POST /api/copilotkit`` (AG-UI
# ``agent/run``); ``/copilotkit/`` fronts the same handler on the bare prefix.
SSE_LOCATIONS = ("/api/", "/copilotkit/")


def _proxy_read_timeouts_by_location(conf_text: str) -> dict[str, int]:
    """Map ``location <path> {`` -> the ``proxy_read_timeout`` seconds in ITS block.

    Brace-tracked, so a location that declares no ``proxy_read_timeout`` cannot
    silently absorb a later directive from outside its block. Deliberately NOT a
    general "effective timeout" resolver: it does not follow ``include``, and it
    does not fall back to a server-level default. Both limitations are safe
    *because* the tests below assert the location is present in the result — a
    location whose timeout moved into an include, or that lost its own
    directive, drops out of the map and fails loudly rather than silently
    reporting an inherited number.
    """
    timeouts: dict[str, int] = {}
    current: str | None = None
    depth = 0
    for raw in conf_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Braces inside a quoted value (e.g. add_header X-Debug "{";) are data,
        # not block structure — blank them before counting, or a stray quoted
        # brace lets one location absorb the NEXT location's timeout.
        depth_line = re.sub(r"'[^']*'|\"[^\"]*\"", "", line)

        loc = re.match(r"^location\s+(?:[=~^*]+\s+)?(\S+)\s*\{", depth_line)
        if loc and current is None:
            current = loc.group(1)
            depth = 1
            continue

        if current is None:
            continue

        prt = re.match(r"^proxy_read_timeout\s+(\d+)s?\s*;", line)
        if prt:
            timeouts[current] = int(prt.group(1))

        depth += depth_line.count("{") - depth_line.count("}")
        if depth <= 0:
            current = None
            depth = 0
    return timeouts


def _streaming_response_bodies_are_wrapped(module_path: Path, function_name: str) -> bool:
    """True iff every ``StreamingResponse`` body in ``function_name`` is keepalive-wrapped.

    Deliberately AST-based rather than a substring search. A ``"with_sse_keepalive"
    in source`` check would still pass with the import and this module's comments
    left in place while the call site itself was reverted — which is exactly the
    regression this guard exists to catch.

    Local aliases of ``StreamingResponse`` (``from fastapi.responses import
    StreamingResponse as SR``) are resolved from the module's own imports, so a
    second, aliased, unwrapped return cannot slip past.
    """
    tree = ast.parse(module_path.read_text())

    # Every name that currently refers to StreamingResponse in this module.
    response_names = {"StreamingResponse"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "StreamingResponse" and alias.asname:
                    response_names.add(alias.asname)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith(".StreamingResponse") and alias.asname:
                    response_names.add(alias.asname)

    target: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == function_name:
            target = node
            break
    assert target is not None, f"{function_name} not found in {module_path} — did it move?"

    found_any = False
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", None)
        if name not in response_names or not node.args:
            continue
        found_any = True
        body = node.args[0]
        body_name = (
            body.func.attr
            if isinstance(body, ast.Call) and isinstance(body.func, ast.Attribute)
            else getattr(getattr(body, "func", None), "id", None)
        )
        if body_name != "with_sse_keepalive":
            return False

    assert found_any, f"no StreamingResponse call found in {function_name} — did it move?"
    return True


@pytest.mark.parametrize("location", SSE_LOCATIONS)
def test_constant_mirrors_host_nginx(location: str) -> None:
    """``PROXY_READ_TIMEOUT_SECONDS`` must equal the deployed nginx value.

    ``docker/nginx/host-nginx.conf`` is the config actually deployed — verified
    2026-08-16 by diffing it against ``/etc/nginx/sites-enabled/e2i-analytics``
    on the production droplet (identical apart from one comment). The other file
    in the tree, ``docker/nginx/nginx.conf`` (60 s), is referenced by no compose
    file and fronts nothing.
    """
    assert HOST_NGINX_CONF.is_file(), f"missing {HOST_NGINX_CONF}"
    timeouts = _proxy_read_timeouts_by_location(HOST_NGINX_CONF.read_text())

    assert location in timeouts, (
        f"no proxy_read_timeout parsed for location {location} in {HOST_NGINX_CONF}; "
        "if the location was renamed, update SSE_LOCATIONS here too"
    )
    assert timeouts[location] == PROXY_READ_TIMEOUT_SECONDS, (
        f"{HOST_NGINX_CONF.name} location {location} declares "
        f"proxy_read_timeout {timeouts[location]}s but "
        f"src/api/utils/sse_keepalive.PROXY_READ_TIMEOUT_SECONDS is "
        f"{PROXY_READ_TIMEOUT_SECONDS}s. These are the same physical ceiling — "
        "change both, or the SSE keepalive cadence is sized against a number "
        "nginx no longer enforces (#1659)."
    )


def test_keepalive_interval_is_an_order_of_magnitude_under_the_ceiling() -> None:
    """One dropped keepalive must not be enough to trip the ceiling."""
    assert SSE_KEEPALIVE_INTERVAL_SECONDS > 0
    assert SSE_KEEPALIVE_INTERVAL_SECONDS * 10 <= PROXY_READ_TIMEOUT_SECONDS, (
        f"keepalive every {SSE_KEEPALIVE_INTERVAL_SECONDS}s leaves too little "
        f"margin under a {PROXY_READ_TIMEOUT_SECONDS}s ceiling"
    )


def test_longest_dispatch_budget_is_covered_by_the_keepalive_not_the_ceiling() -> None:
    """The budgets are allowed to exceed the ceiling — but only because of the keepalive.

    This test exists to catch the reverse regression: if the keepalive is ever
    removed from the SSE path, the longest dispatch budget silently becomes a
    silent window again. It asserts the two facts that make the budgets safe:
    a budget longer than the ceiling exists, and the SSE route wraps its body in
    the keepalive.
    """
    from src.agents.orchestrator.nodes.router import RouterNode

    budgets_ms = [
        dispatch["timeout_ms"]
        for dispatches in RouterNode.INTENT_TO_AGENTS.values()
        for dispatch in dispatches
    ]
    assert budgets_ms, "no dispatch budgets found — did INTENT_TO_AGENTS move?"
    longest_s = max(budgets_ms) / 1000

    if longest_s < PROXY_READ_TIMEOUT_SECONDS:
        pytest.skip("no dispatch budget currently exceeds the proxy ceiling; nothing to guard")

    # #1669: ``stream_chat`` is not the only route under this ceiling.
    # ``copilotkit_custom_handler`` serves the AG-UI brain from the SAME nginx
    # locations, and its ``agent/run`` branch builds its own StreamingResponse.
    #
    # Its OTHER branch delegates to the third-party ``sdk_handler``, which
    # constructs the response inside the installed ``copilotkit`` package — no
    # literal ``StreamingResponse(...)`` call exists here for this AST walk to
    # find, so that branch is guarded BEHAVIOURALLY instead, by draining the
    # real response body in
    # ``tests/unit/test_api/test_agui_stream_health_1667_1669.py``. Keep both:
    # this one catches a reverted call site, that one catches a lost wrapper.
    for function_name in ("stream_chat", "copilotkit_custom_handler"):
        wrapped = _streaming_response_bodies_are_wrapped(
            REPO_ROOT / "src" / "api" / "routes" / "copilotkit.py",
            function_name=function_name,
        )
        assert wrapped, (
            f"the longest dispatch budget is {longest_s:.0f}s against a "
            f"{PROXY_READ_TIMEOUT_SECONDS}s nginx ceiling, and {function_name}'s "
            "StreamingResponse body is no longer wrapped in with_sse_keepalive — the "
            "silent window is back and completed work will be severed before it "
            "reaches the user (#1659, #1669)."
        )
