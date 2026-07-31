"""#1420: the unit tree must never see live Supabase credentials.

``tests/conftest.py``'s ``load_dotenv(override=True)`` walks up from nested
worktrees into the repo-root ``.env``; on the droplet (PROD == DEV) that hands
unit tests the REAL local-Supabase URL and key. ``refute_causal_estimate``
(and any other lazy live-writer) then builds a real repository and unit-test
fixture runs write real rows — 1,760 ``causal_impact_query`` rows landed in
prod ``causal_validations`` on 2026-07-31 alone, unmasked by the #1352
uuid-cast fix. Third recurrence of the live-writer family (#1371 MLflow HTTP,
#1355 agent_activities → ``E2I_DISABLE_AGENT_ACTIVITY_WRITER``).

``tests/unit/conftest.py`` now pins the Supabase env per-test via an autouse
fixture (dead endpoint + fake keys; CI unit jobs run exactly this way, with
no Supabase service, and are green). It must be a fixture, not an import-time
write: the root conftest's ``pytest_configure`` re-runs
``load_dotenv(override=True)`` AFTER conftest imports and clobbers module-level
values. These lock tests mirror ``test_pytest_session_arms_kill_switch`` so
the pin cannot silently regress.
"""

import os

import pytest

from tests.unit.conftest import _DEAD_SUPABASE_ENV


class TestUnitTreeDeadSupabase:
    """Every unit test runs with the dead-Supabase pin from conftest import."""

    def test_supabase_url_is_the_dead_sentinel(self) -> None:
        assert os.environ.get("SUPABASE_URL") == _DEAD_SUPABASE_ENV["SUPABASE_URL"], (
            "unit tests must never see a live SUPABASE_URL — the repo-root .env "
            "walk-up points at the REAL droplet Supabase (#1420)"
        )
        # The sentinel must never be CI's literal 'http://localhost:54321':
        # that address is DEAD on GitHub runners but is the LIVE prod Supabase
        # on the droplet — the exact trap this pin exists to disarm.
        assert "54321" not in _DEAD_SUPABASE_ENV["SUPABASE_URL"]

    def test_all_supabase_key_variants_are_fake(self) -> None:
        for var in (
            "SUPABASE_KEY",
            "SUPABASE_ANON_KEY",
            "SUPABASE_SERVICE_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        ):
            assert os.environ.get(var) == "test-key", (
                f"{var} must be the fake CI value in the unit tree; a real key "
                "here means the .env walk-up reached this process (#1420)"
            )

    def test_factory_builds_only_dead_clients(self) -> None:
        """The client factory must resolve to the dead endpoint, so any
        best-effort write path fails fast instead of landing in prod."""
        url = os.environ.get("SUPABASE_URL", "")
        assert url.startswith("http://127.0.0.1:1"), url

    @pytest.mark.real_supabase
    def test_real_supabase_marker_opts_out_of_the_pin(self) -> None:
        """The documented escape hatch for reachability-gated READ-ONLY
        faithful checks: a ``real_supabase``-marked test keeps the ambient
        env (whatever load_dotenv resolved — real on the droplet, CI's fake
        localhost values on runners) instead of the dead sentinel."""
        # The only invariant that holds in BOTH environments: the pin's
        # sentinel was NOT applied.
        assert os.environ.get("SUPABASE_URL") != _DEAD_SUPABASE_ENV["SUPABASE_URL"]
