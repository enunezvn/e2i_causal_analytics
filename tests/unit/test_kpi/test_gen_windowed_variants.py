"""Golden tests for the windowed-allowlist codegen (arbitrary KPI time window).

These pin the SECURITY-SENSITIVE transform that turns a vetted base KPI
statement into its ``*_windowed*`` variant: the ``<col> >= NOW() - INTERVAL
'<n> days'`` predicate is replaced with a positional ``<col> >= $K::timestamptz
AND <col> < $(K+1)::timestamptz`` window, with everything else (synthetic
wrapper, region join, brand filter) preserved byte-for-byte.

The NRx ``_windowed_include_synthetic`` form below was validated against the live
kpi_query RPC (returned 3394 for 90-day Kisqali).
"""

from scripts.gen_kpi_windowed_variants import generate_variant


def test_nrx_windowed_synthetic_matches_validated():
    v = generate_variant("business_impact_nrx", region=False, include_synthetic=True)
    assert v.query_id == "business_impact_nrx_windowed_include_synthetic"
    assert v.max_params == 3
    assert "event_date >= $2::timestamptz" in v.sql
    assert "event_date < $3::timestamptz" in v.sql
    assert "($1::text IS NULL OR brand::text = $1)" in v.sql
    assert "INTERVAL '30 days'" not in v.sql
    assert "is_synthetic" not in v.sql  # include_synthetic = no wrapper


def test_nrx_windowed_region_params():
    v = generate_variant("business_impact_nrx", region=True, include_synthetic=True)
    assert v.max_params == 4
    assert "event_date >= $3::timestamptz" in v.sql  # brand=$1, region=$2
    assert "LOWER" in v.sql  # region join present


def test_nrx_windowed_excludes_synthetic_by_default():
    v = generate_variant("business_impact_nrx", region=False, include_synthetic=False)
    assert "is_synthetic = false" in v.sql
    assert v.query_id == "business_impact_nrx_windowed"
