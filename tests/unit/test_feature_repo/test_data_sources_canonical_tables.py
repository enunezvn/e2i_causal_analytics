"""Pin Feast PostgreSQL data sources at the canonical tables (no bridging views).

Sub-block 6B-infra-3 repointed every PostgreSQLSource in
``feature_repo/data_sources.py`` away from the migration 031/032 bridging
views (``feast_business_metrics_source``, ``feast_patient_journey_source``,
``feast_trigger_response_source``, ``feast_hcp_profile_source``) and onto the
canonical tables that migration 033 promoted into.  These tests pin those
choices so a later edit cannot silently regress to a view-based read.

The live offline/online parity check still lives in
``tests/integration/test_feast_offline_online_parity.py`` and runs only with
``FEAST_INTEGRATION=1``.  This module is the unit-level guard: it imports the
real ``data_sources`` module (no SQL copies) and asserts the SQL string
shape that PR #2 verification requires before it touches Feast.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# Skip if feast is unavailable (matches sibling test_feast_entities.py).
try:
    import feast  # noqa: F401

    HAS_FEAST = True
except ImportError:
    HAS_FEAST = False

pytestmark = pytest.mark.skipif(
    not HAS_FEAST,
    reason="feast package not installed - install with: pip install feast",
)

# feature_repo/ is not a package (no __init__.py); follow the existing test
# pattern from tests/unit/test_feature_store/test_feast_entities.py and
# insert the directory on sys.path so ``import data_sources`` resolves.
_FEATURE_REPO = Path(__file__).resolve().parents[3] / "feature_repo"
if str(_FEATURE_REPO) not in sys.path:
    sys.path.insert(0, str(_FEATURE_REPO))


@pytest.fixture(scope="module")
def sources():
    """Load the five PostgreSQLSource instances under test.

    Each entry exposes ``source`` (the live PostgreSQLSource instance) and
    ``query`` (the SQL extracted via ``get_table_query_string()``).  The
    PostgreSQLSource API doesn't expose the raw query as a top-level
    attribute, so we call its public method here once per module instead
    of reaching into the internal ``_postgres_options._query`` field.
    The method wraps the SQL in parentheses (it's used for inline
    sub-selects); that wrapper is irrelevant to the substring checks.
    """
    from data_sources import (  # type: ignore[import-not-found]
        business_metrics_source,
        hcp_profiles_source,
        patient_journey_source,
        territory_metrics_source,
        triggers_source,
    )

    raw = {
        "business_metrics": business_metrics_source,
        "patient_journey": patient_journey_source,
        "triggers": triggers_source,
        "hcp_profiles": hcp_profiles_source,
        "territory_metrics": territory_metrics_source,
    }
    return {
        name: type("Probe", (), {"source": src, "query": src.get_table_query_string()})()
        for name, src in raw.items()
    }


class TestCanonicalTableTargets:
    """Each PG source must read from its canonical table, not a bridging view."""

    def test_business_metrics_source_reads_canonical_table(self, sources):
        """business_metrics_source: FROM business_metrics, hcp_id filter retained."""
        query = sources["business_metrics"].query
        assert "FROM business_metrics" in query, (
            "business_metrics_source must read the canonical business_metrics "
            "table after migration 033 (sub-block 6B-infra-3)."
        )
        assert "feast_business_metrics_source" not in query, (
            "Bridging view feast_business_metrics_source was dropped in "
            "migration 033; data_sources must not reference it."
        )
        # The hcp_id IS NOT NULL filter excludes the legacy per-(brand, region)
        # aggregate rows that share business_metrics with the per-HCP rollup
        # rows produced by the 6B-infra-2a ETL.
        assert "hcp_id IS NOT NULL" in query, (
            "Filter must keep per-(brand, region) aggregate rows out of Feast."
        )

    def test_patient_journey_source_reads_canonical_table(self, sources):
        """patient_journey_source: FROM patient_journeys."""
        query = sources["patient_journey"].query
        assert "FROM patient_journeys" in query, (
            "patient_journey_source must read the canonical patient_journeys "
            "table after migration 033."
        )
        assert "feast_patient_journey_source" not in query, (
            "Bridging view feast_patient_journey_source was dropped."
        )

    def test_triggers_source_reads_canonical_table(self, sources):
        """triggers_source: FROM triggers, no COALESCE fallback on brand_id."""
        query = sources["triggers"].query
        assert "FROM triggers" in query, "triggers_source must read the canonical triggers table."
        assert "feast_trigger_response_source" not in query, (
            "Bridging view feast_trigger_response_source was dropped."
        )
        # Migration 033 backfilled triggers.brand_id NOT NULL with the
        # 'UNKNOWN' sentinel; the COALESCE-on-brand_id fallback the bridging
        # view needed must be gone.
        assert "COALESCE" not in query, (
            "brand_id is NOT NULL after migration 033 backfill; the COALESCE "
            "fallback the bridging view used should be gone."
        )

    def test_hcp_profiles_source_reads_canonical_table(self, sources):
        """hcp_profiles_source: FROM hcp_profiles, no 1h-backdate alias."""
        query = sources["hcp_profiles"].query
        assert "FROM hcp_profiles" in query, (
            "hcp_profiles_source must read the canonical hcp_profiles table."
        )
        assert "feast_hcp_profile_source" not in query, (
            "Bridging view feast_hcp_profile_source was dropped."
        )
        # The bridging view exposed `(NOW() - INTERVAL '1 hour') AS
        # last_updated`; the canonical column is updated_at and we use it
        # directly per plan ("Use the real updated_at for event_timestamp,
        # NOT the 1h-backdate hack").
        assert "updated_at AS event_timestamp" in query, (
            "Must alias the canonical updated_at column, not the bridging "
            "view's synthetic last_updated 1h-backdate."
        )
        assert "last_updated" not in query, (
            "last_updated was the bridging view's synthetic alias; canonical column is updated_at."
        )


class TestCompositeKeysAreGeneratedColumns:
    """The synthetic ``a || '_' || b`` composite-key SQL must be gone.

    Migration 033 promoted hcp_brand_id (business_metrics, triggers) and
    patient_brand_id (patient_journeys) to STORED generated columns on the
    canonical tables.  Re-emitting the string-concat in the SELECT is dead
    code and would diverge from the table's authoritative composite-key
    expression.
    """

    @pytest.mark.parametrize("name", ["business_metrics", "patient_journey", "triggers"])
    def test_no_synthetic_composite_keys_in_query(self, sources, name):
        """Parametrised so a failure in one source doesn't mask failures in
        the others (a single ``for``-loop with ``assert`` would short-circuit
        at the first failing source)."""
        query = sources[name].query
        assert "||" not in query, (
            f"{name}_source must not synthesise a composite key with "
            f"'||'; the canonical table now exposes the generated column "
            f"directly. Found in:\n{query}"
        )


class TestEventTimestampWiring:
    """Every PG source must declare ``timestamp_field='event_timestamp'``.

    The Feast offline store joins on ``timestamp_field`` (point-in-time
    correctness), so a typo or accidental rename would break parity tests
    only at integration time.  Pin the contract here.
    """

    @pytest.mark.parametrize(
        "name",
        ["business_metrics", "patient_journey", "triggers", "hcp_profiles", "territory_metrics"],
    )
    def test_all_sources_use_canonical_event_timestamp_column(self, sources, name):
        src = sources[name].source
        assert src.timestamp_field == "event_timestamp", (
            f"{name}_source.timestamp_field must be 'event_timestamp' "
            f"(got {src.timestamp_field!r}); Feast point-in-time joins "
            f"depend on this exact name."
        )


class TestSchemaDriftFix556:
    """#556: pin the source-query drift fixes so they cannot silently regress."""

    def test_business_metrics_source_drops_territory_and_brand_id(self, sources):
        """territory_id / brand_id were never added to canonical business_metrics
        (migration 033 made it per-HCP). Selecting them broke materialize()."""
        query = sources["business_metrics"].query
        # word-boundary so hcp_brand_id (still selected) doesn't match brand_id
        assert not re.search(r"\bterritory_id\b", query), (
            "business_metrics_source must not reference territory_id; it does not "
            "exist on the per-HCP canonical business_metrics (#556)."
        )
        assert not re.search(r"(?<!hcp_)\bbrand_id\b", query), (
            "business_metrics_source must not reference a standalone brand_id; "
            "only the generated hcp_brand_id exists on canonical business_metrics (#556)."
        )

    def test_patient_journey_source_maps_to_canonical_columns(self, sources):
        """The FV fields therapy_start_date / days_on_therapy / churn_risk_score
        must alias the real canonical columns, not select nonexistent ones."""
        query = sources["patient_journey"].query
        # journey_start_date is DATE on the canonical table; the FV field is
        # UnixTimestamp, so it must be cast to TIMESTAMPTZ (as the old bridge did).
        assert "journey_start_date::TIMESTAMPTZ AS therapy_start_date" in query
        assert "COALESCE(journey_duration_days, 0) AS days_on_therapy" in query
        assert "COALESCE(risk_score, 0) AS churn_risk_score" in query
        # therapy_start_date must appear only as the alias target, never selected
        # as a bare (nonexistent) canonical column.
        assert "therapy_start_date" in query and not re.search(
            r"^\s*therapy_start_date\s*,?\s*$", query, re.MULTILINE
        ), "therapy_start_date must be an alias of journey_start_date, not a bare column."

    def test_market_dynamics_online_disabled_pending_real_source(self):
        """market_dynamics_features is (territory, brand)-keyed but sourced from
        the per-HCP business_metrics, which cannot supply those join keys after
        #556. Online serving must stay off until a real source exists."""
        from features.market_features import (  # type: ignore[import-not-found]
            market_dynamics_fv,
        )

        assert market_dynamics_fv.online is False, (
            "market_dynamics_features must keep online=False until a per-"
            "(territory, brand) source exists; the per-HCP business_metrics_source "
            "lacks territory_id/brand_id join keys (#556)."
        )
