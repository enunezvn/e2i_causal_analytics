"""CI guard: ``scripts/seed_falkordb.py`` must not drag the RAG/dspy stack in (#1761).

Why this exists
---------------
#1761 makes the beat-scheduled emptiness sentinel (``src/tasks/graph_reseed_tasks.py``)
subprocess this seeder from inside ``worker_light``. The seeder used to do::

    from src.rag.config import FalkorDBConfig

purely to obtain a 4-field connection dataclass (``.host/.port/.password/.graph_name``).
``src/rag/__init__`` eagerly imports the dspy stack, so that one line cost, measured
2026-08-21 in the deployed image ``ghcr.io/enunezvn/e2i-api:0ff77a084``::

    falkordb : base=14.1MB peak=29.5MB   delta=15.4MB
    src.rag  : base=14.0MB peak=719.0MB  delta=705.0MB
    python scripts/seed_falkordb.py --dry-run  ->  child peak RSS 721.7 MB

``worker_light`` runs under a 1.5 GiB cgroup limit and was measured at 1.034 GiB
resident (~476 MiB headroom), so a 721 MB child is an OOM kill, not a slow task.
This guard pins the regression at both ends: the module must neither import
``src.rag`` (transitively — we inspect the child's real ``sys.modules``) nor
mention it in source (which would catch a function-local re-introduction that
only explodes at runtime), and its import footprint must stay small.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed_falkordb.py"

# Measured post-fix footprint is ~20 MB resident against a ~12 MB bare-interpreter
# baseline (in-container: 721.7 MB -> 21.7 MB child peak for the same --dry-run).
# 200 MB leaves generous room for interpreter/CI variance while still failing
# loudly on any re-introduction of the dspy pull.
MAX_IMPORT_RSS_MB = 200.0

# NOTE on the memory metric: this probe reports the child's CURRENT resident set
# from /proc/self/statm, deliberately NOT resource.getrusage(...).ru_maxrss.
# Measured on this kernel, a fork+exec child inherits the PARENT's high-water
# mark: a child whose true RSS was 11.6 MB reported ru_maxrss=912.5 MB when its
# parent held a 900 MB ballast. Under pytest (a ~1 GB process once conftest has
# loaded) ru_maxrss would therefore "fail" this guard no matter what the seeder
# imports — a metric that measures the harness instead of the subject. Current
# RSS cannot be inherited, and for a pure-import probe it IS the peak.
_PROBE = """
import importlib.util, json, os, resource, sys

spec = importlib.util.spec_from_file_location("_seed_falkordb_probe", {path!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

pages = int(open("/proc/self/statm").read().split()[1])
heavy = sorted(
    name
    for name in sys.modules
    if name == "src.rag" or name.startswith("src.rag.") or name in ("dspy", "torch", "transformers")
)
print(
    json.dumps(
        {{
            "heavy": heavy,
            "rss_mb": pages * os.sysconf("SC_PAGE_SIZE") / 1048576.0,
            "inherited_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "has_config": hasattr(module, "FalkorDBSeeder"),
        }}
    )
)
"""


@pytest.fixture(scope="module")
def seed_import_probe() -> dict:
    """Import the seeder in a clean child and report what it dragged in.

    Module-scoped: the child is the expensive part of this file, and both
    assertions read the same measurement.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(path=str(SEED_SCRIPT))],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"importing {SEED_SCRIPT} failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_seed_script_does_not_import_src_rag(seed_import_probe: dict) -> None:
    """Importing the seeder must not pull ``src.rag`` (the 721 MB dspy stack)."""
    result = seed_import_probe

    # Positive control: a null result here would otherwise be vacuously green
    # (e.g. if the probe silently imported nothing at all).
    assert result["has_config"] is True, (
        "probe did not actually execute the seed module — FalkorDBSeeder missing"
    )

    assert result["heavy"] == [], (
        "scripts/seed_falkordb.py pulled the RAG/dspy stack into its import graph: "
        f"{result['heavy']}. Measured cost of that import in the deployed image is "
        "705 MB delta / 721.7 MB child peak, against ~476 MiB of worker_light "
        "headroom — the sentinel's subprocess would be OOM-killed. Derive the "
        "connection config locally (FALKORDB_URL / FALKORDB_HOST / FALKORDB_PORT / "
        "FALKORDB_PASSWORD / FALKORDB_GRAPH_NAME) instead — see #1761."
    )


def test_seed_script_import_footprint_stays_small(seed_import_probe: dict) -> None:
    """Pin the actual RSS, not just the module names (catches a new heavy dep)."""
    result = seed_import_probe
    assert result["rss_mb"] < MAX_IMPORT_RSS_MB, (
        f"importing scripts/seed_falkordb.py left {result['rss_mb']:.1f} MB resident "
        f"(limit {MAX_IMPORT_RSS_MB} MB). worker_light has ~476 MiB of headroom "
        "under its 1.5 GiB cgroup limit; a heavy import here OOM-kills the "
        "self-heal reseed subprocess (#1761)."
    )


def test_seed_script_has_no_src_rag_import_statement_anywhere() -> None:
    """Static backstop over every import node, executed or not.

    The sys.modules probe above only sees imports that actually RAN. A
    ``src.rag`` import nested inside a function (e.g. back in ``connect()``)
    would sail past it and still cost 705 MB the first time the reseed
    subprocess touched that code path. ``ast`` walks the whole tree, and unlike a
    substring scan it does not fire on the comments that document this decision.
    """
    tree = ast.parse(SEED_SCRIPT.read_text(encoding="utf-8"), filename=str(SEED_SCRIPT))

    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name == "src.rag" or name.startswith("src.rag."):
                offending.append(f"line {node.lineno}: imports {name!r}")

    assert not offending, (
        "scripts/seed_falkordb.py imports src.rag — the seeder must stay runnable "
        f"inside a memory-capped worker (#1761). Offending nodes: {offending}"
    )


def test_seed_script_exposes_its_own_connection_config() -> None:
    """Positive control for the guard above: the seeder must still HAVE a config.

    Deleting the import and nothing else would make every test in this file pass
    while leaving the script unable to connect at all.
    """
    tree = ast.parse(SEED_SCRIPT.read_text(encoding="utf-8"), filename=str(SEED_SCRIPT))
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert "SeedFalkorDBConfig" in defined, (
        "the local connection dataclass is gone — the seeder has no config to connect with (#1761)"
    )
    assert "_parse_falkordb_config" in defined, (
        "the FALKORDB_URL-preferring resolver is gone — the seeder falls back to "
        "localhost:6381 and refuses the connection inside the container (#1761)"
    )
