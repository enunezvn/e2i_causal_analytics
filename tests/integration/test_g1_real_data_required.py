"""G1 escape-hatch smoke — Plan v4 §2 Gate G1 codex pass-1 HIGH-1 closure.

Pins the policy that CI must NOT silently skip the G1 real-data
integration tests. The only way ``test_csu_negative_control_20260510.py``
and ``test_optum_held_out_noninferiority_20260510.py`` are allowed to
``pytest.skip()`` when real cohort data is absent is via an explicit
opt-in: ``ALLOW_MISSING_REAL_DATA=1``.

Why this is a separate test
---------------------------
The integration tests themselves can read the env var, but if a CI
configuration unconditionally sets ``ALLOW_MISSING_REAL_DATA=1`` the
G1 invariant ("real-cohort regression is proven on every PR") is
silently void. This smoke fires loudly in any CI run that has the
env var set — the test is *expected* to pass on developer machines
running the full suite (where the env var is unset) and to fail on a
CI configuration that mass-skips real-data tests.

Codex pass-1 HIGH-1
-------------------
Original failure: the integration tests silently skipped on missing
data, so a CI without the data fixture would report green without
proving anything. The escape-hatch + this smoke close that loop:
- Default behaviour: hard fail when data missing
- Opt-in skip: ALLOW_MISSING_REAL_DATA=1
- This smoke: assert the opt-in is NOT in effect by default
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_g1_real_data_required_by_default() -> None:
    """ALLOW_MISSING_REAL_DATA must NOT be set to '1' by default in CI.

    The CSU + Optum integration tests opt-in to skip when real data is
    absent only if this env var is explicitly set. A CI configuration
    that unconditionally sets the var silently voids the G1 invariant;
    this smoke catches that case.

    To intentionally skip real-data tests (e.g., a smoke-only CI lane),
    document the rationale in the CI config + add an opt-out skip
    marker on this test in the same commit.
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
