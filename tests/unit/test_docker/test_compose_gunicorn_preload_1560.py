"""#1560: the gunicorn preload activation must actually reach the container.

``config/gunicorn.conf.py`` reads ``GUNICORN_PRELOAD`` from ``os.environ``
with an in-code default of False (dark). The measured #1560 root cause fix is
flipping preload ON in production, and the ONLY way the flag reaches the api
container is compose's ``x-common-env`` anchor — an env var absent from the
anchor silently never arrives and the in-code default governs (the
OPIK_ENABLED lesson documented in the compose file). These tests pin:

- the anchor carries ``GUNICORN_PRELOAD`` with a host-overridable default of
  ``true`` (the kill switch is ``GUNICORN_PRELOAD=false`` in the host .env);
- the api service (the one that runs gunicorn) merges the anchor;
- the production image bakes ``src/`` + ``config/`` bytecode (``compileall``)
  so the read-only rootfs + PYTHONDONTWRITEBYTECODE=1 combination stops
  recompiling every module from source on every boot (#1560 phase A tax).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest

yaml = pytest.importorskip("yaml")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE_PATH = _REPO_ROOT / "docker" / "docker-compose.yml"
_DOCKERFILE_PATH = _REPO_ROOT / "docker" / "Dockerfile"


def _compose() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_COMPOSE_PATH.read_text()))


def test_common_env_carries_gunicorn_preload_defaulting_true() -> None:
    doc = _compose()
    common_env = doc["x-common-env"]
    assert "GUNICORN_PRELOAD" in common_env, (
        "GUNICORN_PRELOAD missing from x-common-env: the host .env value would "
        "never reach the container and config/gunicorn.conf.py's dark default "
        "would silently govern"
    )
    assert common_env["GUNICORN_PRELOAD"] == "${GUNICORN_PRELOAD:-true}", (
        "compose default must be true (this line IS the #1560 activation) "
        "while staying host-.env overridable as the rollback kill switch"
    )


def test_api_service_merges_the_common_env_anchor() -> None:
    """The api service runs gunicorn; it must actually receive the flag."""
    doc = _compose()
    api_env = doc["services"]["api"]["environment"]
    assert api_env.get("GUNICORN_PRELOAD") == "${GUNICORN_PRELOAD:-true}", (
        "api service environment does not include GUNICORN_PRELOAD — the "
        "x-common-env anchor merge is broken or the entry moved"
    )


def test_compose_comment_names_the_kill_switch() -> None:
    """The rollback path must be discoverable at the flag's definition site."""
    text = _COMPOSE_PATH.read_text()
    assert re.search(r"#1560[\s\S]{0,900}GUNICORN_PRELOAD=false", text), (
        "the #1560 compose comment must document the GUNICORN_PRELOAD=false host-.env kill switch"
    )


def test_production_image_bakes_src_and_config_bytecode() -> None:
    text = _DOCKERFILE_PATH.read_text()
    m = re.search(r"^RUN .*compileall.*$", text, flags=re.MULTILINE)
    assert m, "production stage must bake bytecode via compileall (#1560)"
    line = m.group(0)
    assert "/app/src" in line and "/app/config" in line, (
        "compileall must cover /app/src and /app/config — the raw-COPY'd trees "
        "that recompile from source every boot on the read-only rootfs"
    )
    assert "unchecked-hash" in line, (
        "immutable image fs: use unchecked-hash invalidation so pycs stay "
        "valid regardless of COPY mtime behavior"
    )


def test_compileall_runs_before_chown() -> None:
    """The pycs are written as root; the blanket chown must still cover them
    so the non-root runtime user can read them."""
    text = _DOCKERFILE_PATH.read_text()
    compile_idx = text.index("compileall")
    chown_idx = text.rindex("chown -R e2i:e2i /app")
    assert compile_idx < chown_idx, "compileall must run before the final chown -R e2i:e2i /app"
