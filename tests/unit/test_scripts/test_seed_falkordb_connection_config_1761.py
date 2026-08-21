"""The seeder's connection resolution must work in-container AND not regress its
existing host-side callers (#1761).

Before #1761 the seeder built its connection from ``src.rag.config.FalkorDBConfig``
with argparse defaults of ``FALKORDB_HOST``/``FALKORDB_PORT`` (fallback 6381) and a
password from ``FALKORDB_PASSWORD``. It knew nothing about ``FALKORDB_URL`` — the
one connection var docker compose actually sets — so running it inside a container
died with::

    ERROR - Failed to connect to FalkorDB: Error 111 connecting to localhost:6381.
    Connection refused.

(measured in the deployed image on 2026-08-21). The beat-scheduled emptiness
sentinel subprocesses this script from inside ``worker_light``, so that had to work.

The two callers that already exist — ``scripts/seed_falkordb_all.sh`` and
``scripts/seed_falkordb_init.py`` — both pass ``--host``/``--port`` explicitly and
set ``FALKORDB_PASSWORD`` in the child environment. Their behaviour must be
unchanged, which is why ``FALKORDB_PASSWORD`` outranks a password inside
``FALKORDB_URL`` here (a deliberate deviation from the app client, documented on
the function).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed_falkordb.py"

FALKORDB_ENV_VARS = (
    "FALKORDB_URL",
    "FALKORDB_HOST",
    "FALKORDB_PORT",
    "FALKORDB_PASSWORD",
    "FALKORDB_GRAPH_NAME",
)


@pytest.fixture(scope="module")
def seed_module() -> ModuleType:
    """Load the seeder by path (it is a script, not an importable package member)."""
    spec = importlib.util.spec_from_file_location("_seed_falkordb_cfg", SEED_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch):
    for name in FALKORDB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_falkordb_url_supplies_host_port_and_password(seed_module, clean_env) -> None:
    """The container path: FALKORDB_URL is the only var worker_light sets."""
    clean_env.setenv("FALKORDB_URL", "redis://:urlpw@falkordb:6379/0")

    assert seed_module._parse_falkordb_config() == ("falkordb", 6379, "urlpw")


def test_explicit_password_outranks_the_url_password(seed_module, clean_env) -> None:
    """scripts/seed_falkordb_init.py sets FALKORDB_PASSWORD in the child env and
    must keep winning — pre-#1761 that was the ONLY password source."""
    clean_env.setenv("FALKORDB_URL", "redis://:urlpw@falkordb:6379/0")
    clean_env.setenv("FALKORDB_PASSWORD", "explicitpw")

    assert seed_module._parse_falkordb_config() == ("falkordb", 6379, "explicitpw")


def test_discrete_vars_are_used_when_no_url_is_set(seed_module, clean_env) -> None:
    """The host path, e.g. `set -a; source .env` (FALKORDB_PORT=6381 there)."""
    clean_env.setenv("FALKORDB_HOST", "localhost")
    clean_env.setenv("FALKORDB_PORT", "6381")
    clean_env.setenv("FALKORDB_PASSWORD", "hostpw")

    assert seed_module._parse_falkordb_config() == ("localhost", 6381, "hostpw")


def test_bare_fallback_keeps_the_scripts_historical_host_port(seed_module, clean_env) -> None:
    """No env at all -> localhost:6381, this script's pre-#1761 default.

    NOT the app client's 6379: that client only runs inside the docker network,
    while a human running this seeder with no env is on the host, where FalkorDB
    is published on 6381. Changing it would be a silent regression for them.
    """
    assert seed_module._parse_falkordb_config() == ("localhost", 6381, None)


def test_config_dataclass_carries_the_four_fields_the_seeder_uses(seed_module, clean_env) -> None:
    """Positive control: the replacement for src.rag.config.FalkorDBConfig must
    still expose exactly what FalkorDBSeeder.connect() reads."""
    config = seed_module.SeedFalkorDBConfig(host="h", port=1234, graph_name="g", password="p")

    assert (config.host, config.port, config.graph_name, config.password) == (
        "h",
        1234,
        "g",
        "p",
    )


def test_config_falls_back_to_falkordb_password_like_the_class_it_replaced(
    seed_module, clean_env
) -> None:
    """src.rag.config.FalkorDBConfig.__post_init__ did exactly this; every existing
    caller relies on it, since none of them passes a password positionally."""
    clean_env.setenv("FALKORDB_PASSWORD", "frompw")

    assert seed_module.SeedFalkorDBConfig(host="h", port=1).password == "frompw"
