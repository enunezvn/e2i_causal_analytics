"""Unit tests for the FalkorDB emptiness sentinel + self-heal reseed (#1761).

Context
-------
#1758: a container recreation wiped the ``e2i_causal`` graph and ``/knowledge-graph``
stayed empty for four days. #1759 removed that wipe vector; this task is the recovery
half — a 30-minute beat tick that probes the curated core and, when it is gone, runs
the two seed scripts that rebuild it.

What is doubled here and what is not
------------------------------------
The graph handle, the Redis lock and the ``subprocess`` boundary are doubled — those
are the three edges this task owns and the only way to exercise the empty branch
without wiping a real graph. Everything the task actually decides (the emptiness
predicate, the ordering of the two scripts, the argv it builds, the lock protocol,
the post-verify gate) runs for real. The live half of the certification is a forced
trigger against a scratch graph on the real FalkorDB — never ``e2i_causal``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import pytest

from src.tasks import graph_reseed_tasks as grt
from src.tasks.graph_reseed_tasks import GraphReseedError, graph_emptiness_sentinel

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeGraph:
    """Graph handle returning a scripted sequence of curated-node counts.

    An element that is an Exception instance is raised instead of returned, so a
    probe failure can be distinguished from a count of zero.
    """

    def __init__(self, counts: Sequence[Any]) -> None:
        self._counts = list(counts)
        self.queries: List[str] = []

    def query(self, cypher: str, params: Optional[dict] = None) -> Any:
        self.queries.append(cypher)
        nxt = self._counts.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return SimpleNamespace(result_set=[[nxt]])


class FakeRedis:
    def __init__(self, acquired: bool = True) -> None:
        self._acquired = acquired
        self.set_calls: List[tuple] = []
        self.eval_calls: List[tuple] = []

    def set(self, key, value, nx=False, ex=None):  # noqa: ANN001
        self.set_calls.append((key, value, nx, ex))
        return True if self._acquired else None

    def eval(self, script, numkeys, *args):  # noqa: ANN001
        self.eval_calls.append((script, numkeys, args))
        return 1


class SubprocessRecorder:
    """Records argv/env for each subprocess and replays scripted return codes."""

    def __init__(self, returncodes: Optional[Sequence[int]] = None) -> None:
        self._returncodes = list(returncodes) if returncodes is not None else None
        self.calls: List[dict] = []

    def __call__(self, cmd, **kwargs):  # noqa: ANN001
        self.calls.append({"cmd": list(cmd), **kwargs})
        rc = self._returncodes.pop(0) if self._returncodes else 0
        return subprocess.CompletedProcess(args=list(cmd), returncode=rc, stdout="", stderr="")

    @property
    def scripts(self) -> List[str]:
        return [Path(call["cmd"][1]).name for call in self.calls]

    @property
    def flat_argv(self) -> List[str]:
        return [token for call in self.calls for token in call["cmd"]]


@pytest.fixture
def wired(monkeypatch):
    """Wire the three doubled edges; each test supplies the graph/redis it needs."""

    def _wire(counts, acquired: bool = True, returncodes=None):
        graph = FakeGraph(counts)
        redis_client = FakeRedis(acquired=acquired)
        runner = SubprocessRecorder(returncodes)
        monkeypatch.setattr(grt, "_open_graph", lambda graph_name: graph)
        monkeypatch.setattr(grt, "_redis_client", lambda: redis_client)
        monkeypatch.setattr(grt.subprocess, "run", runner)
        return SimpleNamespace(graph=graph, redis=redis_client, runner=runner)

    return _wire


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------


def test_emptiness_predicate_is_the_pages_curated_predicate() -> None:
    """The sentinel must measure what /knowledge-graph renders, not total nodes.

    ``KnowledgeGraph.tsx`` always sends ``curated_only: true``; the API turns that
    into ``n.agent IS NULL`` (``semantic_memory.list_nodes``). After the #1758 wipe
    agents repopulated runtime nodes within hours, so a TOTAL count reads non-empty
    while the entire curated layer is gone (live prod 2026-08-21: 100 total nodes,
    89 curated, 11 agent-written).
    """
    assert "n.agent IS NULL" in grt.CURATED_COUNT_QUERY
    assert "count(n)" in grt.CURATED_COUNT_QUERY

    from src.memory import semantic_memory

    source = Path(semantic_memory.__file__).read_text(encoding="utf-8")
    assert 'where_parts.append("n.agent IS NULL")' in source, (
        "semantic_memory's curated_only clause changed — the sentinel's predicate "
        "must track it or the sentinel measures a different graph than the page."
    )


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


def test_non_empty_graph_is_a_noop(wired) -> None:
    ctx = wired(counts=[89])

    result = graph_emptiness_sentinel.run()

    assert result["status"] == "ok"
    assert result["curated_node_count"] == 89
    assert ctx.redis.set_calls == [], "no lock should be taken on a healthy graph"
    assert ctx.runner.calls == [], "no reseed should run on a healthy graph"


def test_empty_graph_runs_both_seed_scripts_in_order(wired) -> None:
    # probe=0, recheck-inside-lock=0, post-verify=64
    ctx = wired(counts=[0, 0, 64])

    result = graph_emptiness_sentinel.run()

    assert result["status"] == "reseeded"
    assert result["curated_node_count"] == 64
    assert ctx.runner.scripts == [
        "seed_falkordb.py",
        "sync_causal_paths_to_falkordb.py",
    ], "structural seed must run before the causal_paths MERGE sync"
    assert "--execute" in ctx.runner.calls[1]["cmd"], (
        "sync_causal_paths_to_falkordb.py defaults to dry-run; without --execute the "
        "self-heal writes nothing"
    )


def test_seed_is_never_invoked_with_clear_first(wired) -> None:
    """``--clear-first`` is the #1758 wipe vector — the self-heal must never pass it."""
    ctx = wired(counts=[0, 0, 64])

    graph_emptiness_sentinel.run()

    assert "--clear-first" not in ctx.runner.flat_argv, (
        "the sentinel passed --clear-first: a self-heal that wipes the graph it is "
        "healing is the #1758 outage with extra steps"
    )


def test_reseed_targets_the_graph_that_was_probed(wired, monkeypatch) -> None:
    """Probe, seed and sync must all address one graph, or the heal lands elsewhere."""
    monkeypatch.setenv("FALKORDB_GRAPH_NAME", "e2i_cert_scratch")
    ctx = wired(counts=[0, 0, 12])

    result = graph_emptiness_sentinel.run()

    assert result["graph"] == "e2i_cert_scratch"
    for call in ctx.runner.calls:
        assert call["env"]["FALKORDB_GRAPH_NAME"] == "e2i_cert_scratch"
    seed_cmd = ctx.runner.calls[0]["cmd"]
    assert "--graph-name" in seed_cmd
    assert seed_cmd[seed_cmd.index("--graph-name") + 1] == "e2i_cert_scratch"


def test_lock_held_by_another_worker_skips_the_reseed(wired) -> None:
    """The structural seed is CREATE-based: two concurrent runs duplicate the graph."""
    ctx = wired(counts=[0], acquired=False)

    result = graph_emptiness_sentinel.run()

    assert result["status"] == "lock_unavailable"
    assert ctx.runner.calls == []
    key, _token, nx, ex = ctx.redis.set_calls[0]
    assert key == grt.RESEED_LOCK_KEY
    assert nx is True, "lock must be SET NX or it is not a lock"
    assert ex == grt.RESEED_LOCK_TTL_SECONDS, "lock must expire or a crash wedges it forever"


def test_recheck_inside_the_lock_finds_the_graph_already_recovered(wired) -> None:
    """Another worker may have reseeded between the probe and the lock."""
    ctx = wired(counts=[0, 77])

    result = graph_emptiness_sentinel.run()

    assert result["status"] == "recovered_before_reseed"
    assert result["curated_node_count"] == 77
    assert ctx.runner.calls == []
    assert ctx.redis.eval_calls, "lock must still be released on the early return"


def test_post_verify_still_empty_is_an_error(wired) -> None:
    ctx = wired(counts=[0, 0, 0])

    with pytest.raises(GraphReseedError) as excinfo:
        graph_emptiness_sentinel.run()

    assert "still empty" in str(excinfo.value).lower()
    assert ctx.runner.calls, "the reseed must have been attempted before erroring"
    assert ctx.redis.eval_calls, "lock must be released even when the reseed fails"


def test_probe_failure_never_reads_as_empty(wired) -> None:
    """A failed scan is UNKNOWN, never zero (the #1760 lesson).

    A silent zero here would trigger a full reseed on every tick during a FalkorDB
    outage — CREATE-based, so it would duplicate the graph the moment FalkorDB came
    back.
    """
    ctx = wired(counts=[ConnectionError("falkordb unreachable")])

    result = graph_emptiness_sentinel.run()

    assert result["status"] == "probe_failed"
    assert "unreachable" in result["error"]
    assert ctx.redis.set_calls == []
    assert ctx.runner.calls == []


def test_successful_reseed_releases_the_lock_with_a_token_compare(wired) -> None:
    """Release must be compare-and-delete: a plain DEL can drop another holder's lock."""
    ctx = wired(counts=[0, 0, 64])

    graph_emptiness_sentinel.run()

    _key, token, _nx, _ex = ctx.redis.set_calls[0]
    assert ctx.redis.eval_calls, "lock was never released"
    script, numkeys, args = ctx.redis.eval_calls[0]
    assert "redis.call('del'" in script.replace('"', "'")
    assert numkeys == 1
    assert args == (grt.RESEED_LOCK_KEY, token)


def test_a_failing_seed_step_is_reported_not_swallowed(wired) -> None:
    ctx = wired(counts=[0, 0, 41], returncodes=[1, 0])

    result = graph_emptiness_sentinel.run()

    assert result["status"] == "reseeded_with_errors"
    assert result["curated_node_count"] == 41
    failed = [step for step in result["steps"] if step["returncode"] != 0]
    assert [step["script"] for step in failed] == ["seed_falkordb.py"]
    assert len(ctx.runner.calls) == 2, "a failed structural seed must not abort the sync"


# ---------------------------------------------------------------------------
# Wiring / boot-safety
# ---------------------------------------------------------------------------


def test_task_is_registered_under_the_beat_entry_name() -> None:
    from src.workers.celery_app import celery_app

    assert "src.tasks.graph_emptiness_sentinel" in celery_app.tasks


def test_module_never_imports_scripts_at_top_level() -> None:
    """Prod 2026-05-26: a top-level ``import scripts.*`` in a task module crash-looped
    every worker and the beat scheduler. The reseed runs scripts as SUBPROCESSES for
    exactly this reason."""
    source = Path(grt.__file__).read_text(encoding="utf-8")
    offending = [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(source.splitlines(), start=1)
        if line.startswith(("import scripts", "from scripts"))
    ]
    assert not offending, f"top-level scripts import in a task module: {offending}"
