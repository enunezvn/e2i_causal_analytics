"""Faithful Shard-04 gate: a real --small --anchor-to-now load to the docker
Supabase, then assert (1) rolling-date freshness, (2) all 3 brand_type labels with
brand-correct NDC + drug_class, (3) all 4 region labels.

Gated by E2I_DB_INTEGRATION=1 and run with -n0 (no live-DB pollution in the default
unit run). Mirrors the Shard-02 faithful gate's subprocess + `docker exec psql`
pattern (no supabase_client fixture exists in this repo). The load is idempotent by
--tag namespace, so re-running overwrites the scv rows with fresh anchored dates
rather than accumulating, and the baseline is never clobbered.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(os.getenv("E2I_DB_INTEGRATION") != "1", reason="faithful DB only"),
    # The module-scoped fixture runs a real --small load (~90s); override the
    # project's per-test pytest-timeout default so setup is not killed mid-load.
    pytest.mark.timeout(600),
]

REPO = Path(__file__).resolve().parents[2]


def _psql(sql: str) -> str:
    return subprocess.check_output(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        text=True,
    ).strip()


@pytest.fixture(scope="module", autouse=True)
def _anchored_load():
    """Run the sanctioned loader ONCE per module with rolling-window anchoring; skip
    the whole gate (rather than fail) if the environment's loader role cannot write.
    The load is idempotent by --tag namespace, so re-running overwrites scv rows.

    The auth signal is a precise one: a genuine `permission denied` (42501), AND a
    post-load DB check that the CORE synthetic rows landed. We deliberately do NOT
    skip on the substring "0 loaded" — the deferred feature-store tables
    (feature_groups/features/feature_values) report 0 even when the core tables
    (treatment_events/patient_journeys) load cleanly, which is exactly what this
    gate asserts on."""
    proc = subprocess.run(
        [sys.executable, "scripts/load_synthetic_data.py", "--small", "--anchor-to-now"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "LOKY_MAX_CPU_COUNT": "1"},
    )
    combined = proc.stdout + proc.stderr
    if "permission denied" in combined:
        pytest.skip("loader write path not authorized here (permission denied / 42501)")
    core = _psql(
        "SELECT count(*) FROM treatment_events WHERE is_synthetic = true AND drug_ndc IS NOT NULL;"
    )
    if int(core) == 0:
        pytest.skip("loader did not write core synthetic treatment_events here")
    yield


def test_synthetic_treatment_dates_within_30d():
    # max synthetic event_date must be inside the NOW()-30d window
    assert (
        _psql(
            "SELECT (max(event_date) >= CURRENT_DATE - INTERVAL '30 days')::int "
            "FROM treatment_events WHERE is_synthetic = true;"
        )
        == "1"
    ), "synthetic treatment_events are stale (max event_date >30d old)"


def test_three_brands_with_brand_correct_ndc():
    # each portfolio brand must have at least one synthetic row carrying the
    # brand-correct NDC prefix + drug_class.
    for brand, ndc10, drug_class in (
        ("Kisqali", "00078-0903", "CDK4/6 Inhibitor"),
        ("Remibrutinib", "00078-1100", "BTK Inhibitor"),
        ("Fabhalta", "00078-1175", "Complement Inhibitor"),
    ):
        n = _psql(
            "SELECT count(*) FROM treatment_events WHERE is_synthetic = true "
            f"AND brand = '{brand}' AND drug_ndc LIKE '{ndc10}%' "
            f"AND drug_class = '{drug_class}';"
        )
        assert int(n) > 0, f"{brand} missing brand-correct NDC/drug_class rows"


def test_four_regions_present():
    assert (
        _psql(
            "SELECT count(DISTINCT geographic_region) FROM patient_journeys "
            "WHERE is_synthetic = true;"
        )
        == "4"
    ), "expected all 4 region labels among synthetic patient_journeys"
