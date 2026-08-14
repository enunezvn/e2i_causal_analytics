"""#1602: the unit tree must never inherit the deployment's synthetic switches.

``tests/conftest.py``'s ``load_dotenv(override=True)`` walks up from nested
worktrees into the repo-root ``.env``; this droplet is a synthetic-gold
showcase instance, so that ``.env`` sets ``E2I_INCLUDE_SYNTHETIC=true`` and
``E2I_KPI_INCLUDE_SYNTHETIC=true``. Both are CORRECT for the running services
— the tests are what needs isolating, never the environment.

With the flags inherited, ``apply_provenance_filter`` legitimately skips its
``.eq('is_synthetic', False)`` link and ``resolve_kpi_query_id`` swaps in the
``_include_synthetic`` RPC twin, so 56 real-mode pins failed on the droplet
while passing in CI (no ``.env`` up the runner's checkout path): 47 in
tests/unit/test_repositories/ + 9 in tests/unit/test_kpi/. Same contamination
class as #1420 (live Supabase creds) and #1414/#1495/#1497, which fixed it
per-module — so every new provenance-asserting file re-introduced it.

``tests/unit/conftest.py`` now pins real mode per-test via an autouse fixture.
It must be a fixture, not an import-time write: the root conftest's
``pytest_configure`` re-runs ``load_dotenv(override=True)`` AFTER conftest
imports and clobbers module-level values. These lock tests mirror
``test_unit_tree_dead_supabase_1420`` so the pin cannot silently regress.

Unlike that predecessor, this module SIMULATES the droplet contamination at
module scope (see ``_simulate_droplet_contamination``) rather than relying on
the host's ``.env``. The #1420 pin substitutes sentinel values, so its absence
is directly observable; a delenv pin's effect is not — on CI the flags are
absent either way, and the lock would pass vacuously with the fixture removed.
Forcing them ON first makes every assertion below prove the pin ACTED.
"""

import os

import pytest

from src.kpi.synthetic_mode import kpi_include_synthetic, resolve_kpi_query_id
from src.repositories.provenance import deployment_includes_synthetic
from tests.unit.conftest import _REAL_MODE_SYNTHETIC_FLAGS


@pytest.fixture(scope="module", autouse=True)
def _simulate_droplet_contamination():
    """Force both flags ON for this module, so the pin is proved to ACT.

    Without this the module would pass vacuously on CI, where no ``.env``
    carries the flags: absent-because-nothing-set-them is indistinguishable
    from absent-because-pinned, so deleting the fixture under test would not
    turn the lock red. MODULE scope is what makes it work — a broader-scope
    fixture runs BEFORE the function-scope pin in tests/unit/conftest.py, so
    every test below starts from the contaminated droplet state and observes
    what the pin did to it, in any environment.
    """
    mp = pytest.MonkeyPatch()
    for var in _REAL_MODE_SYNTHETIC_FLAGS:
        mp.setenv(var, "true")
    yield
    mp.undo()


class TestUnitTreeRealModeSyntheticPin:
    """Every unit test runs with the deployment synthetic switches OFF."""

    def test_both_deployment_flags_are_absent(self) -> None:
        for var in _REAL_MODE_SYNTHETIC_FLAGS:
            assert os.environ.get(var) is None, (
                f"{var} must be absent in the unit tree — the repo-root .env "
                "walk-up carries this showcase deployment's synthetic switch "
                "into local pytest runs (#1602)"
            )

    def test_pin_covers_both_gates_not_just_the_generalized_one(self) -> None:
        """``kpi_include_synthetic`` ORs BOTH flags (synthetic_mode.py), so
        deleting only ``E2I_INCLUDE_SYNTHETIC`` would leave the KPI read path
        contaminated by ``E2I_KPI_INCLUDE_SYNTHETIC`` — the 9 tests/unit/test_kpi
        failures. Pin the OR, not one side of it."""
        assert set(_REAL_MODE_SYNTHETIC_FLAGS) == {
            "E2I_INCLUDE_SYNTHETIC",
            "E2I_KPI_INCLUDE_SYNTHETIC",
        }

    def test_provenance_gate_reads_real_mode(self) -> None:
        assert deployment_includes_synthetic() is False

    def test_kpi_gate_reads_real_mode(self) -> None:
        assert kpi_include_synthetic() is False

    def test_kpi_query_ids_do_not_resolve_to_synthetic_twins(self) -> None:
        """The observable consequence the 9 forwarding tests assert on."""
        assert (
            resolve_kpi_query_id("data_quality_source_coverage_hcps")
            == "data_quality_source_coverage_hcps"
        )

    def test_test_local_setenv_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Showcase-mode tests stay expressible: the autouse delenv runs at
        setup, a test's own ``setenv`` runs after it, so both compose."""
        monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
        assert deployment_includes_synthetic() is True
        assert kpi_include_synthetic() is True

    @pytest.mark.real_supabase
    def test_real_supabase_marker_opts_out_of_the_pin(self) -> None:
        """The documented escape hatch (#1420, widened here): a
        reachability-gated READ-ONLY faithful check runs against the LIVE
        deployment, whose substrate is entirely ``is_synthetic=true``, so it
        must keep the ambient flags. Without this opt-out
        ``test_kpi_resolution.py::test_resolve_conversion_frame_real_supabase``
        fails 3/3 with "no KPI frame resolved" — an environment artifact, not
        a defect.

        The simulated contamination is what makes this assertable off the
        droplet: the marked test must still see the module fixture's values,
        because the pin skipped it."""
        for var in _REAL_MODE_SYNTHETIC_FLAGS:
            assert os.environ.get(var) == "true", (
                f"{var} was deleted despite @pytest.mark.real_supabase — the "
                "faithful live checks need the whole ambient deployment env"
            )
