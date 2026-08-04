"""Root conftest.py - Global pytest fixtures and service availability management.

This module provides:
1. Service availability detection at session start
2. Auto-skip fixtures for external services (Redis, FalkorDB, Supabase)
3. Safe async fixtures with built-in timeouts
4. pytest hooks for service-based test filtering
5. Memory-aware test grouping for heavy ML imports

Usage:
    # In test files, use the fixtures:
    @pytest.mark.requires_redis
    async def test_redis_operation(redis_client):
        await redis_client.ping()

    # Or use skip markers directly:
    @pytest.mark.skipif(not SERVICES_AVAILABLE["redis"], reason="Redis not available")
    def test_something():
        ...

    # For memory-heavy tests (dspy, econml, etc.):
    @pytest.mark.xdist_group(name="dspy_integration")
    def test_heavy_ml_operation():
        import dspy  # Heavy import grouped on single worker
        ...

Memory Management:
    - Default: 4 parallel workers (safe for 7.5GB RAM systems)
    - xdist_group markers ensure heavy imports share workers
    - Use `pytest -n 0` for sequential runs on low-memory systems
"""

from __future__ import annotations

import asyncio
import os
import time
import warnings
from typing import Dict, Optional

import pytest
import pytest_asyncio
from dotenv import load_dotenv

# =============================================================================
# LOAD ENVIRONMENT VARIABLES from .env file IMMEDIATELY
# =============================================================================
# Load .env at module import time to ensure API keys are available before
# any test files are collected. Use override=True so real .env values win
# over any placeholder test keys that may have been set earlier.
load_dotenv(override=True)

# =============================================================================
# ASYNCIO POLLUTION PROBE (issue #218 — follow-up to #215)
# =============================================================================
# ``nest_asyncio.apply()`` is a PROCESS-WIDE monkey-patch of ``asyncio.run``.
# Once any test (directly or via a dependency such as DSPy's ``syncify`` or
# mlflow.genai's ``_make_sync_wrapper``) triggers ``apply()`` on an xdist
# worker, every subsequent ``asyncio.run(coro)`` on that worker routes through
# nest_asyncio's patched runner — which holds a reference to the *original*
# event loop. If that loop has since been closed (e.g., by pytest-asyncio
# tearing down a per-test loop), the next sync ``asyncio.run`` call raises
# ``RuntimeError: Event loop is closed``.
#
# PR #217 closed one polluter (``experiment_designer.graph``'s eager singleton)
# but left the broader bug class open: any of the 10 gated callsites in ``src/``
# or the 3 known third-party callers can pollute if invoked from inside a
# running loop:
#   1. ``dspy/utils/syncify.py:20`` — guarded by ``loop.is_running()`` check
#   2. ``mlflow/genai/utils/trace_utils.py:395`` — guarded by running-loop check
#   3. ``ragas/async_utils.py:49`` — **UNCONDITIONAL** ``apply_nest_asyncio()``
#      helper (called at import-time inside ragas evaluation code; identified by
#      this probe in PR #219 CI run 25879019929 — invoked via
#      ``test_gepa_integration.py::TestRAGASGEPAOpikIntegration``)
#
# The RAGAS polluter is the actual root cause of issue #218 — the test that
# was failing in #215 (Layer-3 ablation) ran on the same xdist worker as the
# RAGAS evaluation test. PR #217's victim-site fix (`loop.run_until_complete`
# instead of `asyncio.run`) keeps the suite green today; the lint + probe
# here prevent a regression if a future victim site reintroduces bare
# ``asyncio.run`` while RAGAS pollution is in effect.
#
# This probe instruments ``nest_asyncio.apply`` at session start so we can:
#   1. Record the FIRST ``apply()`` call's full stack trace (the polluter).
#   2. Detect the test boundary AT WHICH ``asyncio.run`` first becomes
#      monkey-patched, and surface BOTH the test nodeid and the call stack
#      in a session-summary line + a pytest warning.
#
# Default behaviour is *observational* — we do not fail the session, because
# the existing victim-site mitigations (PR #217 commit ``a321b64f``,
# ``test_layer_5_pipeline_integration.py``, etc.) keep the suite green. Set
# ``E2I_ASSERT_NO_ASYNCIO_POLLUTION=1`` in CI to promote a detected polluter
# into a hard ``pytest.exit`` so the offending test is named on a red CI run.
#
# Issue #221: strict mode is intended to be enabled on the Integration
# Tests CI lane (see ``.github/workflows/backend-tests.yml``
# ``integration-tests`` job env block). When ``E2I_ASSERT_NO_ASYNCIO_POLLUTION``
# is set, the probe promotes a detected polluter into a hard
# ``pytest.exit(returncode=2)`` — UNLESS the polluter is in
# ``_STRICT_MODE_ALLOWLISTED_POLLUTERS``, in which case the warning +
# terminal-summary diagnostic still render but the session is not killed.
# Two trip paths:
#   1. *Runtime-observed pollution* (``pytest_runtest_logfinish``):
#      ``nest_asyncio.apply()`` fired DURING the session after the
#      trace installed. The probe records the offending test
#      nodeid (``apply_first_nodeid``) + full stack trace, so the
#      offender is named on a red CI run. The allowlist check
#      happens at apply() capture time — strict mode skips exit if
#      the first apply came from a known-benign caller (e.g., RAGAS).
#   2. *Preexisting-baseline pollution* (``pytest_configure``,
#      codex pass-1 HIGH-1): ``asyncio.run`` was already patched
#      BEFORE this conftest loaded (sitecustomize, earlier conftest,
#      pytest plugin loaded out-of-order). The probe cannot trace
#      the polluter on this path — there is no captured stack — but
#      strict mode still fails loud so a polluted baseline never
#      silently passes. Allowlist DOES NOT APPLY to this path because
#      we can't identify the caller. Audit surface: sitecustomize.py
#      + earlier pytest plugins + parent conftest files.
# Sequenced after issue #220 / PR #222 (all bare ``asyncio.run``
# callsites in ``tests/integration/`` migrated to the shared ``run_sync``
# helper), so on a green branch the only path-1 trip should come from
# allowlisted callers — and those are warning-only under strict mode.
# A non-allowlisted trip means a NEW polluter has appeared. Rollback:
# unset the env var in the workflow; this comment stays accurate either
# way because it only describes the strict-mode contract, not the live
# workflow state.
_ORIG_ASYNCIO_RUN = asyncio.run

# Issue #221 strict-mode allowlist: known-benign third-party callers of
# ``nest_asyncio.apply()`` whose pollution is mitigated by the issue-#220
# explicit-loop migration in tests/integration/. Strict mode only fires on
# polluters NOT in this list — the goal is "catch NEW polluters introduced
# by a future dependency upgrade", not "no apply() ever". Match is a
# substring check against each frame's ``filename`` in the first apply()
# stack trace; both ``/`` and ``\\`` work because traceback.format_stack
# yields the raw path as captured by the interpreter.
#
# RAGAS (``ragas/async_utils.py:49 apply_nest_asyncio``) calls apply()
# unconditionally at import-time inside its evaluation helper. Identified
# by PR #219's runtime probe in CI run 25879019929. See [[issue-218-close-20260514]]
# memory for why we explicitly did NOT patch RAGAS upstream: stable
# transitive dep, victim-side mitigation (PR #217 + issue #220) keeps the
# suite green, runtime probe + AST lint prevent future victim-site regressions.
_STRICT_MODE_ALLOWLISTED_POLLUTERS = (
    "ragas/async_utils.py",
    "ragas\\async_utils.py",
)

_ASYNCIO_POLLUTION_STATE: Dict[str, object] = {
    "apply_first_stack": None,  # type: Optional[str]
    "apply_first_nodeid": None,  # type: Optional[str]
    "polluter_nodeid": None,  # type: Optional[str]
    "current_nodeid": None,  # type: Optional[str]
    "apply_count": 0,
    "installed_trace": False,
    # Codex pass-1 LOW: surface pre-existing pollution (a sitecustomize,
    # earlier conftest, or pytest plugin that imported nest_asyncio +
    # called ``apply()`` before this module loaded). When True, the
    # baseline ``_ORIG_ASYNCIO_RUN`` is already the patched function and
    # the runtime probe cannot detect new pollution by identity check.
    "preexisting_pollution_detected": False,
    # Issue #221: whether the FIRST observed apply() call came from a
    # known-benign caller (allowlist match in the stack trace). When True,
    # strict mode treats this as a known-mitigated polluter and does NOT
    # fire ``pytest.exit``. The terminal-summary block still notes which
    # caller polluted so a future audit has the evidence.
    "first_apply_allowlisted": False,
    # Issue #221: which allowlist entry matched, if any. Used for the
    # terminal-summary diagnostic so the operator can see "RAGAS polluted
    # as expected" vs "unknown polluter — investigate."
    "first_apply_allowlist_match": None,  # type: Optional[str]
    # Issue #221 codex pass-4 MEDIUM: track whether ANY apply() call —
    # not just the first — came from a non-allowlisted caller. Without
    # this, a non-allowlisted polluter that fires AFTER an allowlisted
    # one (e.g., RAGAS polluted first, then a NEW dependency polluted
    # later on the same worker) would be silently suppressed by the
    # first-apply allowlist match. Strict mode now re-evaluates this
    # flag on every test boundary instead of only at first-apply time.
    "non_allowlisted_apply_seen": False,
    # Issue #221 codex pass-5 MEDIUM-1: when strict mode fires on a
    # non-allowlisted apply that came AFTER an allowlisted first
    # apply, the fatal message must surface the non-allowlisted stack
    # (the novel culprit) instead of the first-apply stack (which
    # would be RAGAS — the known-safe one). Capture the FIRST
    # non-allowlisted apply's stack + nodeid; subsequent ones are
    # ignored.
    "non_allowlisted_apply_stack": None,  # type: Optional[str]
    "non_allowlisted_apply_nodeid": None,  # type: Optional[str]
}


def _check_stack_against_allowlist(stack: str) -> Optional[str]:
    """Return the first allowlist entry whose substring appears in ``stack``,
    or ``None`` if nothing matched.

    Substring match against the formatted traceback string. Each
    ``traceback.format_stack`` frame includes ``File "<path>"`` so the
    path-substring approach reliably identifies the caller without
    parsing frame objects. Cheap (single ``in`` per allowlist entry,
    no regex).
    """

    for needle in _STRICT_MODE_ALLOWLISTED_POLLUTERS:
        if needle in stack:
            return needle
    return None


def _check_preexisting_pollution() -> None:
    """Detect pollution that happened BEFORE this conftest imported.

    Heuristic: nest_asyncio's ``apply`` rewrites ``asyncio.run`` to point
    at its own ``run`` helper. If ``asyncio.run.__module__`` is no longer
    the stdlib ``asyncio.runners`` (or ``asyncio``), something has
    already patched it. We record this for the terminal-summary block so
    a future debugger run knows the baseline is unreliable.

    No-op if nothing is detectable; this is purely informational and
    never raises (avoids breaking unrelated test sessions where another
    legitimate plugin has wrapped asyncio.run).
    """

    module_name = getattr(asyncio.run, "__module__", "") or ""
    # Stdlib spellings across Python versions.
    if module_name in {"asyncio", "asyncio.runners"}:
        return
    qual = getattr(asyncio.run, "__qualname__", "") or ""
    # If a nest_asyncio fingerprint is visible on the callable, mark it.
    if "nest_asyncio" in module_name or "nest_asyncio" in qual:
        _ASYNCIO_POLLUTION_STATE["preexisting_pollution_detected"] = True


def _install_nest_asyncio_apply_trace() -> None:
    """Wrap ``nest_asyncio.apply`` so we record the first-call stack trace.

    Idempotent: re-installing on top of itself is a no-op. The wrapper still
    delegates to the real ``apply`` (we instrument, not block) so PR #217's
    lazy-apply pattern continues to work for legitimate nested-loop cases.
    """
    if _ASYNCIO_POLLUTION_STATE["installed_trace"]:
        return
    try:
        import nest_asyncio  # type: ignore[import-not-found]
    except ImportError:
        # nest_asyncio is a project dep; absence means the suite cannot be
        # polluted via it. Nothing to instrument.
        _ASYNCIO_POLLUTION_STATE["installed_trace"] = True
        return

    _real_apply = nest_asyncio.apply

    def _traced_apply(*args, **kwargs):  # type: ignore[no-untyped-def]
        import traceback

        # Increment for every apply() call so a follow-up regression test
        # can pin "apply_count==0" on clean sessions.
        _ASYNCIO_POLLUTION_STATE["apply_count"] = int(_ASYNCIO_POLLUTION_STATE["apply_count"]) + 1

        # Capture the stack ONCE per apply() call so both the first-apply
        # bookkeeping and the codex-pass-4 MEDIUM "any later non-allowlisted
        # apply" check see the same evidence.
        stack = "".join(traceback.format_stack())
        allowlist_hit = _check_stack_against_allowlist(stack)

        if _ASYNCIO_POLLUTION_STATE["apply_first_stack"] is None:
            _ASYNCIO_POLLUTION_STATE["apply_first_stack"] = stack
            _ASYNCIO_POLLUTION_STATE["apply_first_nodeid"] = _ASYNCIO_POLLUTION_STATE[
                "current_nodeid"
            ]
            # Issue #221: stamp the allowlist match (or None) onto state at
            # capture time so the strict-mode check in
            # ``pytest_runtest_logfinish`` doesn't have to re-walk the stack.
            if allowlist_hit is not None:
                _ASYNCIO_POLLUTION_STATE["first_apply_allowlisted"] = True
                _ASYNCIO_POLLUTION_STATE["first_apply_allowlist_match"] = allowlist_hit

        # Issue #221 codex pass-4 MEDIUM: even if the FIRST apply was
        # allowlisted, a LATER apply from a non-allowlisted caller should
        # still trip strict mode. Flag this on every non-allowlisted
        # call so ``pytest_runtest_logfinish`` can fire late.
        if allowlist_hit is None:
            _ASYNCIO_POLLUTION_STATE["non_allowlisted_apply_seen"] = True
            # Issue #221 codex pass-5 MEDIUM-1: capture the FIRST
            # non-allowlisted stack so the strict-mode fatal message
            # names the actual culprit instead of the (allowlisted)
            # first apply. Subsequent non-allowlisted applies are
            # ignored — first one wins for diagnostic purposes.
            if _ASYNCIO_POLLUTION_STATE["non_allowlisted_apply_stack"] is None:
                _ASYNCIO_POLLUTION_STATE["non_allowlisted_apply_stack"] = stack
                _ASYNCIO_POLLUTION_STATE["non_allowlisted_apply_nodeid"] = _ASYNCIO_POLLUTION_STATE[
                    "current_nodeid"
                ]

        return _real_apply(*args, **kwargs)

    nest_asyncio.apply = _traced_apply  # type: ignore[assignment]
    _ASYNCIO_POLLUTION_STATE["installed_trace"] = True


# Run the pre-existing pollution check BEFORE installing the trace so
# ``_ORIG_ASYNCIO_RUN`` is captured at the earliest moment we have access
# to. If the baseline is already patched we surface it in the terminal
# summary rather than silently treating the patched function as authentic.
_check_preexisting_pollution()

# Install ASAP so we catch apply() calls fired during fixture setup, not just
# inside the test body itself.
_install_nest_asyncio_apply_trace()


# =============================================================================
# ASYNCIO POLLUTION CONTAINMENT (issue #218 follow-up — autouse finalizer)
# =============================================================================
# The probe above only *observes* pollution; it does not undo it. Issues
# #218/#220 mitigated KNOWN victim sites in ``tests/integration/`` by
# migrating their bare ``asyncio.run(coro)`` calls to the explicit-loop
# ``run_sync`` helper. But that migration is scoped to one tree: victims
# OUTSIDE ``tests/integration/`` (e.g.
# ``tests/ml/synthetic_v2/test_integration_diagnostic_runner.py``) still
# call bare ``asyncio.run`` and crash with ``RuntimeError: Event loop is
# closed`` whenever they land on the SAME xdist worker, AFTER, the RAGAS
# polluter (``ragas/async_utils.py:49`` -> ``nest_asyncio.apply()``).
#
# Under the slow lane's ``--dist=loadscope`` mode (see
# ``.github/workflows/slow-tests.yml`` job A: ``-n 2 --dist=loadscope``),
# ``xdist_group`` markers are NOT honoured, so we cannot reliably pin the
# polluter and the victims onto separate workers — co-location is left to
# loadscope's scheduling, which is why the failure is INTERMITTENT (green
# only when the victims happen to land on a different worker than the
# polluter).
#
# This autouse, function-scoped finalizer is the robust, co-scheduling-
# INDEPENDENT containment: before every test it snapshots ``asyncio.run``
# and the event-loop policy; after every test it RESTORES the pristine
# ``_ORIG_ASYNCIO_RUN`` (captured at conftest import, before any apply())
# and the original loop policy if either was monkey-patched during the
# test. Any test that triggers ``nest_asyncio.apply()`` is therefore
# cleaned up at its own boundary, so the NEXT test on that worker always
# gets a working ``asyncio.run`` — regardless of which tree it lives in or
# which worker it landed on.
#
# Why this does NOT weaken the existing guards:
#   - The probe's ``_traced_apply`` still increments ``apply_count`` and
#     captures the polluter stack at apply()-time; restoring ``asyncio.run``
#     AFTERWARDS does not erase that observation. The ``[issue-218]``
#     warning + ``E2I_ASSERT_NO_ASYNCIO_POLLUTION`` strict-mode exit on a
#     NON-allowlisted polluter still fire exactly as before.
#   - The AST lints (``test_no_unconditional_nest_asyncio_apply.py`` over
#     src/, ``test_no_bare_asyncio_run_in_integration_tests.py`` over
#     tests/integration/) are untouched — this is a runtime safety net, not
#     a replacement for keeping src/ apply() calls gated.
#   - We restore to ``_ORIG_ASYNCIO_RUN`` (the genuine stdlib runner), not
#     to a possibly-already-polluted snapshot, so a polluter that fires
#     during fixture setup of the SAME test is also healed for the next one.
@pytest.fixture(autouse=True)
def _restore_asyncio_run_after_pollution():
    """Restore ``asyncio.run`` + the event-loop policy after each test.

    Function-scoped autouse finalizer. Snapshots the live ``asyncio.run``
    and the current event-loop policy before the test runs, then on
    teardown restores the pristine ``_ORIG_ASYNCIO_RUN`` (and the original
    policy) whenever a ``nest_asyncio.apply()`` (or any other monkey-patch)
    swapped them out during the test. This neutralises the cross-test
    ``RuntimeError: Event loop is closed`` pollution (issue #218) without
    relying on xdist worker co-scheduling.
    """

    policy_before = asyncio.get_event_loop_policy()
    try:
        yield
    finally:
        # Heal ``asyncio.run`` if a polluter (RAGAS / DSPy / mlflow.genai)
        # swapped it for nest_asyncio's runner during this test. Comparing
        # against ``_ORIG_ASYNCIO_RUN`` keeps the no-op fast path cheap for
        # the overwhelming majority of tests that never pollute.
        if asyncio.run is not _ORIG_ASYNCIO_RUN:
            asyncio.run = _ORIG_ASYNCIO_RUN  # type: ignore[assignment]
        # nest_asyncio.apply() can also patch the loop policy / loop class;
        # restore the policy object if it changed so the next test starts
        # from a clean loop factory.
        if asyncio.get_event_loop_policy() is not policy_before:
            asyncio.set_event_loop_policy(policy_before)


# =============================================================================
# TOOL REGISTRY POLLUTION CONTAINMENT (issue #782 — autouse finalizer)
# =============================================================================
# ``ToolRegistry`` (src/tool_registry/registry.py) is a process-wide singleton:
# ``ToolRegistry()`` and ``get_registry()`` return the SAME instance, and the
# real composable tools (``causal_effect_estimator`` etc.) are registered into it
# at import time via the ``@composable_tool`` decorator. Several unit tests and
# fixtures (``tests/unit/test_agents/test_tool_composer/`` +
# ``tests/unit/test_tool_registry/``) call ``registry.clear()`` on the singleton
# for their own isolation — which WIPES those real tools for every later test in
# the same process. The integration functional gate
# (``test_canonical_query_produces_real_tool_successes_stub_planner``) then fails
# with ``Unknown tool in plan: causal_effect_estimator`` and zero tools
# succeeding (#782). CI runs unit and integration in SEPARATE jobs so it does not
# trigger there, but a combined local run does — and it is a latent fragility if
# the lanes are ever merged.
#
# Mirrors ``_restore_asyncio_run_after_pollution`` (#218): snapshot the global
# registry before each test, then MERGE-restore afterwards via the registry's own
# ``snapshot()`` / ``restore_snapshot()`` API. Merge (re-add / un-shadow, never
# remove) resets the singleton to EXACTLY its pre-test state, healing any
# ``clear()``, same-name mock, or stray registration a test performed.
@pytest.fixture(autouse=True)
def _restore_tool_registry_after_pollution():
    """Restore the global ToolRegistry singleton's tools after each test (#782)."""
    try:
        from src.tool_registry.registry import get_registry
    except ImportError:
        # Minimal test environments (e.g. the synthetic-benchmarks job, which
        # installs only requirements-synthetic.txt and so lacks pydantic — a
        # transitive dep of the registry module) never touch the ToolRegistry, so
        # there is nothing to restore. No-op instead of erroring every collected
        # test. (The registry import was added to this autouse fixture in #782;
        # before that the benchmark env never imported it — hence the regression.)
        yield
        return

    registry = get_registry()
    snapshot = registry.snapshot()
    try:
        yield
    finally:
        # Fast no-op path: skip the restore unless the registry changed — a tool
        # was removed/replaced (identity differs) OR added (count differs).
        saved_tools = snapshot["tools"]
        if registry.tool_count != len(saved_tools) or any(
            registry.get(name) is not tool for name, tool in saved_tools.items()
        ):
            registry.restore_snapshot(snapshot)


# =============================================================================
# TESTING MODE - Set before any src imports to bypass JWT auth
# =============================================================================
os.environ["E2I_TESTING_MODE"] = "1"

# =============================================================================
# MLFLOW 3.15 TEST-HARNESS COMPAT (#1476 — mlflow 3.11.1 -> 3.15.1)
# =============================================================================
# Two mlflow behavior changes land between 3.11 and 3.15 that affect ONLY the
# test harness (prod uses an HTTP tracking server + sqlite-backed mlflow
# service container; neither var is set in any compose environment):
#
# 1. mlflow 3.13.0 (upstream #22773): pointing the tracking or model-registry
#    store at a local filesystem path raises MlflowException by default.
#    Several unit tests deliberately use throwaway ``file://{tmp_path}`` stores
#    (test_twin_persistence, test_training_job, test_execute_twin_retraining_real,
#    test_mlflow_tracker_artifact_dest_1452, ...). The upstream opt-out restores
#    them without changing what they exercise.
#
# 2. mlflow ~3.14 uv auto-detection (MLFLOW_UV_AUTO_DETECT, default true):
#    ``log_model`` detects a uv project (uv.lock + pyproject.toml in cwd
#    ancestry) and infers the MODEL's pip requirements from ``uv export`` of
#    the repo's uv.lock instead of capture-based inference. Our uv.lock is NOT
#    the installed-env source of truth (requirements.lock is) and drifts from
#    it (measured 2026-08-04: uv.lock xgboost==3.2.0 vs installed 3.1.2), which
#    makes ``mlflow.xgboost.log_model`` raise "requirements versions are
#    incompatible" from inside the repo checkout. Disable auto-detection so
#    model-env inference reflects the ACTUAL installed packages. Prod docker
#    images never contain uv.lock (docker/Dockerfile copies requirements.txt /
#    requirements.lock / pyproject.toml only), so detection cannot fire there.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")


def _load_dotenv_at_configure():
    """Run load_dotenv again at configure time for safety."""
    load_dotenv(override=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Service URLs from environment with defaults.
#
# Port mapping (per docker/docker-compose.yml):
#   - e2i_redis (Working Memory): 6382
#   - e2i_falkordb (Semantic Memory): 6381
#   - opik-redis (Opik, separate): 6390
#   - auto-claude-falkordb (external): 6380
#
# CI note (issue #183): GitHub Actions integration-tests/unit-tests/slow-tests
# jobs do NOT stand up a FalkorDB service container. Historically those
# workflows still exported ``FALKORDB_URL='redis://localhost:6380'``, which
# caused ``_check_redis_service`` to log a noisy ECONNREFUSED on every CI run
# and confuse the failure picture when an xdist worker died for unrelated
# reasons (see PR #181's run 25809588036 where a transient gw0 message was
# misread as a FalkorDB outage). The workflow YAML has been updated to drop
# FALKORDB_URL when no FalkorDB service is declared; the conftest now opts
# the probe out cleanly when ``FALKORDB_REQUIRED`` is not set, so the session
# header reads ``FALKORDB: UNCONFIGURED (expected in this lane)`` instead of
# producing an ECONNREFUSED traceback.
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # No password in docker/docker-compose.yml
_redis_url_env = os.getenv("REDIS_URL")
if _redis_url_env:
    REDIS_URL = _redis_url_env
elif REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@localhost:6382"
else:
    REDIS_URL = "redis://localhost:6382"

# FALKORDB_URL is optional. When it is unset AND the lane has not opted into
# FALKORDB_REQUIRED=1, the service-check probe is skipped (returns False)
# rather than attempting a TCP connect that we know cannot succeed. This
# preserves the existing skip semantics for ``requires_falkordb`` tests while
# stopping the misleading ECONNREFUSED log line.
FALKORDB_URL = os.getenv("FALKORDB_URL", "")
FALKORDB_REQUIRED = os.getenv("FALKORDB_REQUIRED", "").lower() in {"1", "true", "yes"}
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "") or os.getenv("SUPABASE_SERVICE_KEY", "")

# Connection timeout for service checks (seconds)
SERVICE_CHECK_TIMEOUT = 3.0

# Global service availability cache (populated at session start)
SERVICES_AVAILABLE: Dict[str, bool] = {
    "redis": False,
    "falkordb": False,
    "supabase": False,
}


# =============================================================================
# SERVICE AVAILABILITY CHECKING
# =============================================================================


async def _check_redis_service(url: str, timeout: float = SERVICE_CHECK_TIMEOUT) -> bool:
    """Check if a Redis-compatible service is available.

    Args:
        url: Redis connection URL
        timeout: Connection timeout in seconds

    Returns:
        True if service is reachable and responds to PING
    """
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            url,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
        )
        try:
            await asyncio.wait_for(client.ping(), timeout=timeout)
            return True
        finally:
            await client.aclose()
    except Exception as e:
        # Debug: print exception to help diagnose connectivity issues
        import sys

        print(f"  DEBUG _check_redis_service({url}): {type(e).__name__}: {e}", file=sys.stderr)
        return False


def _check_supabase_service() -> bool:
    """Check if Supabase credentials are configured.

    Note: We only check for credentials, not actual connectivity,
    because Supabase is a remote service and connectivity checks
    would add latency to every test run.

    Returns:
        True if SUPABASE_URL and key are configured
    """
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _run_service_checks() -> Dict[str, bool]:
    """Run all service availability checks.

    Returns:
        Dictionary mapping service names to availability status
    """
    results = {}

    # Use a fresh event loop to avoid conflicts with pytest-asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Check Redis
        import sys as _debug_sys

        print(f"  DEBUG _run_service_checks: REDIS_URL={REDIS_URL}", file=_debug_sys.stderr)
        results["redis"] = loop.run_until_complete(_check_redis_service(REDIS_URL))
    except Exception as e:
        import sys

        print(f"  DEBUG: Redis check failed: {e}", file=sys.stderr)
        results["redis"] = False

    # FalkorDB check is skipped cleanly when no URL is configured (issue #183).
    # The probe ONLY fires when a non-empty ``FALKORDB_URL`` is exported
    # (local-droplet runs, where docker-compose auto-injects the env var
    # pointing at :6381). The independent ``FALKORDB_REQUIRED`` env var does
    # NOT itself trigger a probe — it only surfaces a warning when the URL is
    # empty so future FalkorDB-backed CI lanes can't silently mis-configure
    # the env without log evidence (codex pass-1 LOW-1 wording clarification).
    if FALKORDB_URL:
        try:
            results["falkordb"] = loop.run_until_complete(_check_redis_service(FALKORDB_URL))
        except Exception as e:
            import sys

            print(f"  DEBUG: FalkorDB check failed: {e}", file=sys.stderr)
            results["falkordb"] = False
        finally:
            loop.close()
    else:
        # No URL configured. Treat as unconfigured (skip-equivalent) instead of
        # attempting a TCP connect that we know will fail with ECONNREFUSED.
        results["falkordb"] = False
        loop.close()
        if FALKORDB_REQUIRED:
            import sys

            print(
                "  WARN: FALKORDB_REQUIRED=1 set but FALKORDB_URL is empty; "
                "FalkorDB-marked tests will skip.",
                file=sys.stderr,
            )

    # Check Supabase (credentials only)
    results["supabase"] = _check_supabase_service()

    return results


# =============================================================================
# PYTEST HOOKS
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with service availability information.

    This runs once at the start of the test session to:
    1. Re-load .env for safety (mirrors top-of-file ``load_dotenv`` call).
    2. Disable rate limiting for tests.
    3. Apply the issue-#221 strict-mode trip-switch for preexisting
       asyncio pollution (must run before collection / service checks
       so a polluted baseline doesn't silently pass).
    4. Check which services are available.
    5. Store results for skip decision making.
    6. Print service status to console.
    """
    global SERVICES_AVAILABLE

    # Re-load .env defensively (the top-of-file ``load_dotenv`` already
    # ran at import time; this catches the case where a parent conftest
    # mutated os.environ between import and configure).
    _load_dotenv_at_configure()

    # Issue #221 codex pass-1 HIGH-1 (re-located here by codex pass-2
    # HIGH so the hook is not shadowed by the duplicate-definition bug):
    # when strict mode is requested AND ``_check_preexisting_pollution``
    # flagged the baseline ``asyncio.run`` as already-patched, the
    # identity-based probe in ``pytest_runtest_logfinish`` cannot detect
    # further pollution — strict mode would silently pass. Fail-loud
    # here so the session never enters collection with an unreliable
    # baseline. This path has no ``apply_first_nodeid`` and no captured
    # first-apply stack (pollution happened before this conftest
    # loaded), so the exit message names sitecustomize / earlier-loaded
    # plugins / parent conftests as the audit surface rather than
    # promising a stack trace that doesn't exist. Configure-time
    # ``pytest.exit`` fires before ``pytest_sessionfinish``, so the
    # terminal-summary diagnostic block is NOT guaranteed on this path
    # (codex pass-2 LOW); the message is self-contained.
    if _ASYNCIO_POLLUTION_STATE["preexisting_pollution_detected"] and os.getenv(
        "E2I_ASSERT_NO_ASYNCIO_POLLUTION", ""
    ).lower() in {"1", "true", "yes"}:
        pytest.exit(
            "[issue-218] FATAL: asyncio.run was already patched before "
            "tests/conftest.py loaded — the runtime probe cannot detect "
            "further nest_asyncio.apply() calls by identity check, so "
            "strict mode cannot give a meaningful guarantee. Audit "
            "sitecustomize.py, earlier-loaded pytest plugins, and parent "
            "conftest files for the import-time polluter. "
            "(E2I_ASSERT_NO_ASYNCIO_POLLUTION is set; preexisting "
            "pollution detected; no apply_first_nodeid or stack trace "
            "available on this path.)",
            returncode=2,
        )

    # Disable rate limiting for tests to prevent 429 errors from state accumulation
    os.environ["DISABLE_RATE_LIMITING"] = "1"

    # Run service checks
    start_time = time.time()
    SERVICES_AVAILABLE = _run_service_checks()
    check_duration = time.time() - start_time

    # Store in config for access by other hooks
    config._service_availability = SERVICES_AVAILABLE

    # Print service status (only if not in quiet mode)
    quiet = getattr(config.option, "quiet", 0) or getattr(config.option, "q", 0)
    if not quiet:
        print("\n" + "=" * 60)
        print("SERVICE AVAILABILITY CHECK")
        print("=" * 60)
        for service, available in SERVICES_AVAILABLE.items():
            if service == "falkordb" and not FALKORDB_URL and not available:
                # Distinguish "intentionally unconfigured" from "configured but
                # unreachable" so a session header in CI reads
                # ``FALKORDB: UNCONFIGURED`` rather than implying a service
                # outage (issue #183).
                status = "UNCONFIGURED (skipping FalkorDB-marked tests)"
                icon = "\u2013"  # en dash
            elif available:
                status = "AVAILABLE"
                icon = "\u2713"
            else:
                status = "UNAVAILABLE"
                icon = "\u2717"
            print(f"  {icon} {service.upper()}: {status}")
        print(f"  (checked in {check_duration:.2f}s)")
        print("=" * 60 + "\n")


# =============================================================================
# ASYNCIO POLLUTION HOOKS (issue #218)
# =============================================================================


def pytest_collectstart(collector):  # type: ignore[no-untyped-def]
    """Attribute import-phase ``nest_asyncio.apply()`` calls to the
    collection item being processed (codex pass-1 MEDIUM-3).

    Without this hook, a polluter fired during a test module's import
    would record ``apply_first_nodeid=None`` and the diagnostic would
    surface only at the next test boundary. By setting
    ``current_nodeid`` here, the trace correctly names the module that
    triggered the import-time pollution.
    """

    nodeid = getattr(collector, "nodeid", None)
    if nodeid:
        _ASYNCIO_POLLUTION_STATE["current_nodeid"] = f"<collect> {nodeid}"


def pytest_runtest_protocol(item, nextitem):  # type: ignore[no-untyped-def]
    """Record which test is currently executing so the apply() trace can
    attribute the first ``nest_asyncio.apply()`` call to a specific nodeid."""

    _ASYNCIO_POLLUTION_STATE["current_nodeid"] = item.nodeid
    # Don't return a value — we only observe; pytest continues the protocol.
    return None


def _format_pollution_warning(nodeid: str) -> str:
    """Build the warning/exit message body. Inlined here so the message
    surfaces in BOTH the controller terminal summary AND the per-worker
    captured output (codex pass-1 MEDIUM-2 — under xdist
    ``_ASYNCIO_POLLUTION_STATE`` is process-local, so the controller has
    only what comes back through the worker's stdout/warnings stream).
    """

    apply_nodeid = _ASYNCIO_POLLUTION_STATE.get("apply_first_nodeid")
    stack = _ASYNCIO_POLLUTION_STATE.get("apply_first_stack") or "<not captured>"
    apply_count = _ASYNCIO_POLLUTION_STATE.get("apply_count", 0)
    return (
        f"[issue-218] asyncio.run was monkey-patched by nest_asyncio.apply() "
        f"during test: {nodeid}. First apply() called from: "
        f"{apply_nodeid!r} (total apply count this worker: {apply_count}). "
        f"Set E2I_ASSERT_NO_ASYNCIO_POLLUTION=1 to promote this to a hard "
        f"failure.\n"
        f"--- first apply() call stack ---\n{stack}\n--- end stack ---"
    )


def _format_non_allowlisted_strict_exit(test_nodeid: str) -> str:
    """Build the strict-mode fatal message for the non-allowlisted path.

    Issue #221 codex pass-5 MEDIUM-1: when strict mode fires on a
    non-allowlisted apply that fired AFTER an allowlisted first apply
    (e.g., RAGAS polluted first then a NEW dep polluted), the operator
    needs the NEW culprit's stack — not RAGAS's. This formatter
    surfaces ``non_allowlisted_apply_stack`` + ``non_allowlisted_apply_nodeid``
    while still acknowledging the allowlisted first apply for full
    context (so the operator can verify the allowlist isn't masking
    something subtle).
    """

    novel_nodeid = _ASYNCIO_POLLUTION_STATE.get("non_allowlisted_apply_nodeid")
    novel_stack = _ASYNCIO_POLLUTION_STATE.get("non_allowlisted_apply_stack") or "<not captured>"
    first_match = _ASYNCIO_POLLUTION_STATE.get("first_apply_allowlist_match")
    apply_count = _ASYNCIO_POLLUTION_STATE.get("apply_count", 0)
    allowlist_context = (
        f"First apply was allowlisted ({first_match!r}); "
        if first_match
        else "First apply was NOT allowlisted; "
    )
    return (
        f"[issue-218] asyncio.run was monkey-patched by a non-allowlisted "
        f"nest_asyncio.apply() during test: {test_nodeid}. "
        f"{allowlist_context}"
        f"first non-allowlisted apply() recorded while running: "
        f"{novel_nodeid!r} (total apply count this worker: {apply_count}). "
        f"Audit this stack — the polluter is NOT in "
        f"_STRICT_MODE_ALLOWLISTED_POLLUTERS:\n"
        f"--- non-allowlisted apply() call stack ---\n"
        f"{novel_stack}\n--- end stack ---"
    )


def pytest_runtest_logfinish(nodeid, location):  # type: ignore[no-untyped-def]
    """Detect the first test boundary at which ``asyncio.run`` is polluted.

    Fires after every test (passed/failed/skipped/xfailed). Once we observe
    ``asyncio.run is not _ORIG_ASYNCIO_RUN``, we record the polluter nodeid
    and emit a pytest warning so the session summary shows the culprit.

    Under xdist, this hook fires on each worker; the warning message embeds
    the full apply() call stack so it survives the worker→controller
    boundary (codex pass-1 MEDIUM-2). The controller's terminal summary
    still re-prints whatever local state exists, which is empty on the
    controller but populated on each worker.
    """

    # The warning emission below is one-shot per worker (gated on
    # ``polluter_nodeid is None``). The strict-mode check, however, is
    # re-evaluated on every test boundary so a non-allowlisted apply()
    # that fires AFTER an allowlisted one (codex pass-4 MEDIUM) still
    # trips strict mode. We split the two responsibilities here.
    if _ASYNCIO_POLLUTION_STATE["polluter_nodeid"] is None and asyncio.run is not _ORIG_ASYNCIO_RUN:
        _ASYNCIO_POLLUTION_STATE["polluter_nodeid"] = nodeid
        msg = _format_pollution_warning(str(nodeid))
        warnings.warn(msg, category=RuntimeWarning, stacklevel=1)
        # Also write to stderr so the message survives even when warnings
        # are captured/suppressed by the user's pytest filter config.
        import sys as _sys

        print(msg, file=_sys.stderr, flush=True)

    if os.getenv("E2I_ASSERT_NO_ASYNCIO_POLLUTION", "").lower() in {"1", "true", "yes"}:
        # Issue #221 codex pass-4 MEDIUM: strict mode fires on ANY apply()
        # from a non-allowlisted caller, regardless of whether an earlier
        # allowlisted apply already set the first-apply state. This
        # preserves issue-#220 victim-site mitigation for RAGAS while
        # still catching a NEW polluter that fires later on the same
        # worker. Allowlisted-only sessions skip the exit; the warning
        # and terminal-summary diagnostic already rendered above.
        if _ASYNCIO_POLLUTION_STATE.get("non_allowlisted_apply_seen"):
            # Hard-fail. We use pytest.exit (not raise) so the surrounding
            # test isn't blamed for the pollution — the session summary
            # names the actual culprit via ``polluter_nodeid``. Issue #221
            # codex pass-5 MEDIUM-1: surface the NON-ALLOWLISTED stack
            # (the novel culprit) instead of ``apply_first_stack`` (which
            # would be RAGAS — the known-safe one if it polluted first).
            current_msg = _format_non_allowlisted_strict_exit(str(nodeid))
            pytest.exit(
                f"[issue-218] FATAL: nest_asyncio.apply() polluted "
                f"asyncio.run from a non-allowlisted caller.\n{current_msg}",
                returncode=2,
            )


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # type: ignore[no-untyped-def]
    """Append an issue-218 diagnostic block to the terminal summary when
    pollution was detected during the session.

    Under xdist this fires on the controller, where
    ``_ASYNCIO_POLLUTION_STATE`` is per-process and therefore empty when
    the actual polluter ran on a worker. The per-worker warning emitted
    by ``pytest_runtest_logfinish`` carries the full stack, so missing
    state on the controller is a documented xdist limitation (codex
    pass-1 MEDIUM-2). When polluted ON the controller (single-process
    runs, ``-p no:xdist``), the block below renders normally.
    """

    if (
        _ASYNCIO_POLLUTION_STATE["polluter_nodeid"] is None
        and not _ASYNCIO_POLLUTION_STATE["preexisting_pollution_detected"]
    ):
        return
    terminalreporter.write_sep("=", "issue-218 asyncio pollution diagnostic")
    if _ASYNCIO_POLLUTION_STATE["preexisting_pollution_detected"]:
        terminalreporter.write_line(
            "WARNING: asyncio.run was ALREADY patched before this conftest "
            "loaded — runtime detection by identity check may miss further "
            "polluters. Audit sitecustomize.py / earlier-loaded pytest "
            "plugins / parent conftest files."
        )
    if _ASYNCIO_POLLUTION_STATE["polluter_nodeid"] is None:
        return
    terminalreporter.write_line(
        f"polluter_nodeid (first test after pollution detected): "
        f"{_ASYNCIO_POLLUTION_STATE['polluter_nodeid']!r}"
    )
    terminalreporter.write_line(
        f"apply_first_nodeid (running when first apply() fired): "
        f"{_ASYNCIO_POLLUTION_STATE['apply_first_nodeid']!r}"
    )
    terminalreporter.write_line(
        f"nest_asyncio.apply() call count: {_ASYNCIO_POLLUTION_STATE['apply_count']}"
    )
    # Issue #221 codex pass-4 LOW: surface the allowlist decision so the
    # operator can distinguish "RAGAS polluted as expected" from "unknown
    # polluter — investigate."
    allowlist_match = _ASYNCIO_POLLUTION_STATE.get("first_apply_allowlist_match")
    if allowlist_match:
        terminalreporter.write_line(
            f"first_apply_allowlist_match: {allowlist_match!r} "
            f"(strict-mode pytest.exit suppressed for first apply; "
            f"see _STRICT_MODE_ALLOWLISTED_POLLUTERS)"
        )
    if _ASYNCIO_POLLUTION_STATE.get("non_allowlisted_apply_seen"):
        terminalreporter.write_line(
            "non_allowlisted_apply_seen: True (at least one apply() came "
            "from a caller NOT in _STRICT_MODE_ALLOWLISTED_POLLUTERS; "
            "strict mode would fail this session if "
            "E2I_ASSERT_NO_ASYNCIO_POLLUTION=1)"
        )
        novel_nodeid = _ASYNCIO_POLLUTION_STATE.get("non_allowlisted_apply_nodeid")
        terminalreporter.write_line(
            f"non_allowlisted_apply_nodeid (first NEW-polluter apply attributed to): "
            f"{novel_nodeid!r}"
        )
        novel_stack = _ASYNCIO_POLLUTION_STATE.get("non_allowlisted_apply_stack")
        if novel_stack:
            terminalreporter.write_line("--- non-allowlisted apply() call stack ---")
            for line in str(novel_stack).splitlines():
                terminalreporter.write_line(line)
            terminalreporter.write_line("--- end non-allowlisted stack ---")
    stack = _ASYNCIO_POLLUTION_STATE.get("apply_first_stack")
    if stack:
        terminalreporter.write_line("--- first apply() call stack ---")
        for line in str(stack).splitlines():
            terminalreporter.write_line(line)
        terminalreporter.write_line("--- end stack ---")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection to skip tests based on service availability.

    This automatically skips tests marked with requires_* markers
    when the corresponding service is not available.
    """
    services = getattr(config, "_service_availability", SERVICES_AVAILABLE)

    skip_markers = {
        "requires_redis": ("redis", "Redis not available"),
        "requires_falkordb": ("falkordb", "FalkorDB not available"),
        "requires_supabase": ("supabase", "Supabase not configured"),
    }

    for item in items:
        for marker_name, (service, reason) in skip_markers.items():
            if marker_name in [m.name for m in item.iter_markers()]:
                if not services.get(service, False):
                    item.add_marker(pytest.mark.skip(reason=reason))


# =============================================================================
# SERVICE AVAILABILITY FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def service_availability() -> Dict[str, bool]:
    """Get service availability status.

    Returns:
        Dictionary mapping service names to availability boolean
    """
    return SERVICES_AVAILABLE.copy()


@pytest.fixture
def skip_without_redis():
    """Skip test if Redis is not available."""
    if not SERVICES_AVAILABLE["redis"]:
        pytest.skip("Redis not available")


@pytest.fixture
def skip_without_falkordb():
    """Skip test if FalkorDB is not available."""
    if not SERVICES_AVAILABLE["falkordb"]:
        pytest.skip("FalkorDB not available")


@pytest.fixture
def skip_without_supabase():
    """Skip test if Supabase is not configured."""
    if not SERVICES_AVAILABLE["supabase"]:
        pytest.skip("Supabase not configured")


# =============================================================================
# SAFE ASYNC CLIENT FIXTURES
# =============================================================================


@pytest_asyncio.fixture
async def redis_client():
    """Create a Redis client with automatic skip if unavailable.

    This fixture:
    1. Skips if Redis is not available
    2. Creates client with connection timeout
    3. Verifies connection with PING
    4. Cleans up after test

    Yields:
        redis.asyncio.Redis: Connected Redis client
    """
    if not SERVICES_AVAILABLE["redis"]:
        pytest.skip("Redis not available")

    import redis.asyncio as aioredis

    client = aioredis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=SERVICE_CHECK_TIMEOUT,
        socket_connect_timeout=SERVICE_CHECK_TIMEOUT,
    )

    try:
        # Verify connection
        await asyncio.wait_for(client.ping(), timeout=SERVICE_CHECK_TIMEOUT)
        yield client
    except asyncio.TimeoutError:
        pytest.skip(f"Redis connection timeout ({REDIS_URL})")
    except Exception as e:
        pytest.skip(f"Redis connection failed: {e}")
    finally:
        await client.aclose()


def _enforce_falkordb_preconditions() -> None:
    """Shared skip-or-proceed guard for the ``falkordb_client`` fixture.

    Extracted as a top-level helper so unit tests can exercise it without
    pulling the full pytest_asyncio fixture machinery (codex pass-2 LOW-2).

    Raises:
        pytest.skip.Exception: when either the global availability flag is
            False (no probe success) or ``FALKORDB_URL`` is empty (issue
            #183 defensive guard against ``aioredis.from_url("")``).
    """
    if not SERVICES_AVAILABLE["falkordb"]:
        pytest.skip("FalkorDB not available")
    if not FALKORDB_URL:
        pytest.skip("FalkorDB URL not configured")


@pytest_asyncio.fixture
async def falkordb_client():
    """Create a FalkorDB client with automatic skip if unavailable.

    This fixture:
    1. Skips if FalkorDB is not available
    2. Creates client with connection timeout
    3. Verifies connection with PING
    4. Cleans up after test

    Yields:
        redis.asyncio.Redis: Connected FalkorDB client (Redis protocol)
    """
    _enforce_falkordb_preconditions()

    import redis.asyncio as aioredis

    client = aioredis.from_url(
        FALKORDB_URL,
        decode_responses=True,
        socket_timeout=SERVICE_CHECK_TIMEOUT,
        socket_connect_timeout=SERVICE_CHECK_TIMEOUT,
    )

    try:
        # Verify connection
        await asyncio.wait_for(client.ping(), timeout=SERVICE_CHECK_TIMEOUT)
        yield client
    except asyncio.TimeoutError:
        pytest.skip(f"FalkorDB connection timeout ({FALKORDB_URL})")
    except Exception as e:
        pytest.skip(f"FalkorDB connection failed: {e}")
    finally:
        await client.aclose()


@pytest.fixture
def supabase_client():
    """Create a Supabase client with automatic skip if not configured.

    This fixture:
    1. Skips if Supabase credentials are not set
    2. Creates sync Supabase client

    Returns:
        supabase.Client: Connected Supabase client
    """
    if not SERVICES_AVAILABLE["supabase"]:
        pytest.skip("Supabase not configured")

    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# ASYNC UTILITIES
# =============================================================================


@pytest.fixture
def async_timeout():
    """Provide an async timeout wrapper for use in tests.

    Usage:
        async def test_something(async_timeout):
            result = await async_timeout(some_async_func(), timeout=5.0)

    Returns:
        Callable that wraps coroutines with asyncio.wait_for
    """

    async def _timeout_wrapper(coro, timeout: float = 5.0):
        return await asyncio.wait_for(coro, timeout=timeout)

    return _timeout_wrapper


# =============================================================================
# TEST ISOLATION FIXTURES
# =============================================================================

# AGENT-ACTIVITY WRITER KILL SWITCH (#1355, incident 2026-07-30). The runtime
# agent_activities writer (src/agents/activity_writer.py) lazily creates the
# REAL service-role Supabase client, and this conftest's load_dotenv
# (override=True) hands it real creds — so pre-existing agent unit suites that
# invoke contribute_to_memory wrote 16 real rows into the LIVE table (same bug
# family as the #1371 CHATBOT_MLFLOW_METRICS unit-test HTTP calls). Arm the
# kill switch at IMPORT time (before any test, before any fixture ordering
# question) so no test can implicitly write the live table; tests that WANT the
# writer inject an explicit fake client, which the switch does not block.
# Locked by tests/unit/test_agents/test_agent_activity_writer.py::
# TestPersistAgentActivity::test_pytest_session_arms_kill_switch.
os.environ["E2I_DISABLE_AGENT_ACTIVITY_WRITER"] = "1"


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment state before each test.

    This ensures tests don't leak state through environment variables.
    """
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================


@pytest.fixture
def timer():
    """Provide a simple timer for performance measurement.

    Usage:
        def test_performance(timer):
            timer.start()
            do_something()
            elapsed = timer.stop()
            assert elapsed < 1.0, "Too slow!"
    """

    class Timer:
        def __init__(self):
            self._start: Optional[float] = None
            self._elapsed: Optional[float] = None

        def start(self) -> None:
            self._start = time.perf_counter()
            self._elapsed = None

        def stop(self) -> float:
            if self._start is None:
                raise RuntimeError("Timer not started")
            self._elapsed = time.perf_counter() - self._start
            return self._elapsed

        @property
        def elapsed(self) -> Optional[float]:
            return self._elapsed

    return Timer()


# ---------------------------------------------------------------------------
# D1.3 — audit_workflow_id fixture for ml_foundation agent tests
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_workflow_id():
    """Workflow-scoped UUID for threading through ml_foundation agent
    ``input_data`` dicts in tests.

    Tests that call ``agent.run({...})`` should include
    ``"audit_workflow_id": audit_workflow_id`` in input_data so the State
    receives a caller-provided UUID rather than relying on the
    ``Field(default_factory=uuid4)`` transition shim. After sub-shard
    D1.4 drops the default_factory, this fixture becomes load-bearing
    for any test that constructs initial agent state.

    Example::

        @pytest.mark.asyncio
        async def test_thing(audit_workflow_id):
            input_data = {
                "audit_workflow_id": audit_workflow_id,
                ...
            }
            result = await agent.run(input_data)
    """
    from uuid import uuid4

    return uuid4()


# =============================================================================
# CAUSAL ROLE GOLDEN SET (issue #358)
# =============================================================================
# Literature-derived golden evaluation set for Layer-4 causal-role classifier.
# Per #358 acceptance: ≥30 entries per cohort × 3 cohorts (CSU/PNH/BC) =
# ≥90 entries, each carrying feature_name + ground_truth_role + rationale +
# provenance. Used by scripts/measure_layer4_precision.py and any unit test
# that needs a real (vs. synthetic A1-A4) gold standard.


@pytest.fixture(scope="session")
def causal_role_golden_set() -> Dict:
    """Session-scoped loader for the literature-derived golden set.

    Returns the full JSON document (``{"entries": [...], "cohorts": {...}}``).
    Loaded once per pytest session; tests that need entry-level access should
    depend on ``causal_role_golden_set_entry`` (parametrized fixture below).

    Source of truth: ``tests/fixtures/causal_role_golden_set.json``. Built
    via the 3-cohort literature extraction under #358 (commit 64aeffd8+ in
    PR for #358). If the file is missing the fixture raises so tests fail
    loudly rather than silently passing.
    """
    import json
    from pathlib import Path

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "causal_role_golden_set.json"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Causal role golden set fixture missing at {fixture_path}. "
            "Build via the #358 PR workflow; do not run these tests against "
            "an empty fixture."
        )
    return json.loads(fixture_path.read_text())


def _golden_set_entries() -> list:
    """Read entries at collection time for pytest parametrization.

    Called during test collection (before session fixtures are evaluated),
    so this can't use the session-scoped fixture above. Reads the JSON
    directly. Returns empty list if the fixture file is missing (collection
    proceeds, individual tests skip via pytest.skip in the test body).
    """
    import json
    from pathlib import Path

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "causal_role_golden_set.json"
    if not fixture_path.exists():
        return []
    try:
        data = json.loads(fixture_path.read_text())
        return data.get("entries", [])
    except (json.JSONDecodeError, OSError):
        return []


@pytest.fixture(
    params=_golden_set_entries(),
    ids=lambda entry: (
        f"{entry.get('cohort', 'unknown')}::{entry.get('feature_name', '?')}"
        if isinstance(entry, dict)
        else str(entry)
    ),
)
def causal_role_golden_set_entry(request) -> Dict:
    """Parametrized fixture: one test invocation per golden-set entry.

    Use to write per-entry assertions (e.g., "every entry has a non-empty
    rationale" or "every instrument-labeled entry has a verifiable PMID").
    Skips with a clear message if the fixture file is missing.
    """
    if not request.param:
        pytest.skip(
            "Causal role golden set fixture is empty or missing. Build via the #358 PR workflow."
        )
    return request.param
