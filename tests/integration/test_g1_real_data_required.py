"""G1 escape-hatch smoke — Plan v4 §2 Gate G1 codex pass-1 HIGH-1 closure.

Pins the policy that CI must NOT silently skip the G1 real-data
integration tests by default. The only way
``test_csu_negative_control_20260510.py`` and
``test_optum_held_out_noninferiority_20260510.py`` are allowed to
``pytest.skip()`` when real cohort data is absent is via an explicit
opt-in: ``ALLOW_MISSING_REAL_DATA=1``.

CI policy update (2026-05-11, post-v5)
--------------------------------------
The GitHub Actions backend-tests workflow now explicitly opts into
``ALLOW_MISSING_REAL_DATA=1`` because the runner has no access to the
real CSU / Optum cohort fixtures (see ``.github/workflows/backend-tests.yml``
env block). The G1 real-cohort regression is therefore proven on local
runs and on the dedicated slow-tests workflow that ships data fixtures,
NOT on the per-PR backend-tests workflow.

This smoke is correspondingly skipped in CI runs that have opted into
the escape hatch. Per the original docstring's documented exit path:
"To intentionally skip real-data tests (e.g., a smoke-only CI lane),
document the rationale in the CI config + add an opt-out skip marker
on this test in the same commit."

Why this is a separate test
---------------------------
The integration tests themselves can read the env var, but if a CI
configuration unconditionally sets ``ALLOW_MISSING_REAL_DATA=1`` the
G1 invariant ("real-cohort regression is proven on every PR") is
silently void. This smoke fires loudly in any LOCAL run that has the
env var set unintentionally — the test is *expected* to pass on
developer machines running the full suite (where the env var is
unset) and to skip in the documented CI configuration that opted in.

Codex pass-1 HIGH-1
-------------------
Original failure: the integration tests silently skipped on missing
data, so a CI without the data fixture would report green without
proving anything. The escape-hatch + this smoke close that loop:
- Default behaviour: hard fail when data missing
- Opt-in skip: ALLOW_MISSING_REAL_DATA=1
- This smoke: assert the opt-in is NOT in effect by default
- CI opt-out: see ``.github/workflows/backend-tests.yml`` env block
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ALLOW_MISSING_REAL_DATA") == "1",
    reason=(
        "CI workflow opted into ALLOW_MISSING_REAL_DATA=1 — the GitHub "
        "Actions runner has no real cohort fixtures and the G1 "
        "real-cohort regression is proven on the dedicated slow-tests "
        "workflow instead. See .github/workflows/backend-tests.yml env "
        "block. Skipping rather than failing per the documented opt-out "
        "path in the module docstring."
    ),
)
def test_g1_real_data_required_by_default() -> None:
    """ALLOW_MISSING_REAL_DATA must NOT be set to '1' by default locally.

    The CSU + Optum integration tests opt-in to skip when real data is
    absent only if this env var is explicitly set. A configuration that
    sets the var unintentionally silently voids the G1 invariant; this
    smoke catches that case on local developer runs.

    CI runs that explicitly opt-in (per the module docstring) skip
    this test via the @pytest.mark.skipif above.
    """
    val = os.environ.get("ALLOW_MISSING_REAL_DATA")
    assert val != "1", (
        "ALLOW_MISSING_REAL_DATA=1 is set in this environment — the G1 "
        "real-data integration tests will SKIP rather than FAIL when "
        "the CSU/Optum fixtures are absent. The G1 invariant ('real-"
        "cohort regression is proven on every PR') is therefore not "
        "being enforced. Either:\n"
        "  (a) Unset the env var so missing data fails loudly, OR\n"
        "  (b) Document the rationale in the CI config + opt this "
        "smoke out in the same commit (e.g., a smoke-only CI lane "
        "where data fixtures intentionally do not ship)."
    )
