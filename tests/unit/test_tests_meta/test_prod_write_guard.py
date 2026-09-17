"""The prod-write guard's DECISION function (owner-approved addition, 2026-09-17).

Hermetic and pure: every test here hands ``assess`` counts and reads its verdict.
No database, no environment, no clock. That boundary is deliberate and it is the
whole reason the decision is a separate function from the census -- the live half
(do the real counts discriminate file by file?) is measured by the dispatcher
against the real database, because nothing at this layer can produce it.

What these tests CANNOT show, stated so nobody mistakes green here for safety:
whether the five files' real windows actually contain unplanted rows. A guard that
decides correctly on every count is still a no-op if the counts come back zero
everywhere, and still refuses everything if they come back non-zero everywhere.
That is a property of the DATA, and it is measured elsewhere.
"""

from __future__ import annotations

import re

import pytest

from tests.integration._prod_write_guard import (
    PER_HCP_METRIC_TYPE,
    CensusQuery,
    WindowCensus,
    WriteWindowSpec,
    adherence_spec,
    assess,
    census_sql,
    per_hcp_rollup_spec,
    territory_rollup_spec,
)

ISOLATED = WindowCensus(
    source_rows_total=12,
    source_rows_planted=12,
    target_entities_total=2,
    target_entities_planted=2,
    teardown_reach_preexisting=0,
)


# =============================================================================
# The permitting branch
# =============================================================================


def test_a_fully_planted_window_is_permitted() -> None:
    """The positive control. A guard that refuses everything is useless, so the
    permit branch is exercised first and with non-zero totals -- an all-zero census
    would also pass and would prove nothing about discrimination."""
    verdict = assess(ISOLATED)
    assert verdict.refused is False
    assert verdict.reasons == ()
    assert bool(verdict) is True


def test_an_empty_window_is_permitted() -> None:
    """Zero rows anywhere is isolated too: there is nothing to derive from and
    nothing to overwrite. This is the shape the 2019 and 2003 windows are expected
    to have, and it must not be confused with the planted case above."""
    assert assess(WindowCensus(0, 0, 0, 0, 0)).refused is False


# =============================================================================
# The refusing branch -- each leg alone, because each catches a different hazard
# =============================================================================


def test_the_derivation_leg_alone_refuses() -> None:
    """444 triggers in the window, none planted: the measured January-2024 case."""
    verdict = assess(
        WindowCensus(
            source_rows_total=444,
            source_rows_planted=0,
            target_entities_total=0,
            target_entities_planted=0,
            teardown_reach_preexisting=0,
        )
    )
    assert verdict.refused is True
    assert len(verdict.reasons) == 1
    reason = verdict.reasons[0]
    assert reason.startswith("derivation:")
    # The counts must be IN the message: a refusal that does not say how much it
    # found cannot be triaged, and the operator's next move depends on the number.
    assert "444 source row(s)" in reason and "444 total, 0 planted" in reason


def test_the_key_space_leg_alone_refuses_when_every_source_row_is_planted() -> None:
    """The territory cross-join hazard, and the reason leg 1 is not sufficient.

    Source rows fully planted -- a derivation-only guard would PERMIT this -- but
    ``territories`` is every ``territory_id`` in ``hcp_profiles``, so the rollup
    still writes a row for every real territory on every windowed date.
    """
    verdict = assess(
        WindowCensus(
            source_rows_total=9,
            source_rows_planted=9,
            target_entities_total=41,
            target_entities_planted=3,
            teardown_reach_preexisting=0,
        )
    )
    assert verdict.refused is True
    assert len(verdict.reasons) == 1
    assert verdict.reasons[0].startswith("key space:")
    assert "38 target entity key(s)" in verdict.reasons[0]


def test_the_teardown_leg_alone_refuses_when_nothing_else_trips() -> None:
    """The window-scoped ``DELETE FROM territory_metrics`` hazard. Neither the
    derivation nor the key space is implicated: the cleanup is what destroys rows."""
    verdict = assess(
        WindowCensus(
            source_rows_total=0,
            source_rows_planted=0,
            target_entities_total=1,
            target_entities_planted=1,
            teardown_reach_preexisting=7,
        )
    )
    assert verdict.refused is True
    assert len(verdict.reasons) == 1
    assert verdict.reasons[0].startswith("teardown reach:")
    assert "7 pre-existing row(s)" in verdict.reasons[0]


def test_every_tripped_leg_is_reported_not_just_the_first() -> None:
    """A refusal that stops at the first hazard sends the reader round the loop
    once per leg. All three must be named in one failure."""
    verdict = assess(
        WindowCensus(
            source_rows_total=444,
            source_rows_planted=4,
            target_entities_total=41,
            target_entities_planted=2,
            teardown_reach_preexisting=7,
        )
    )
    assert verdict.refused is True
    assert [r.split(":")[0] for r in verdict.reasons] == [
        "derivation",
        "key space",
        "teardown reach",
    ]


def test_one_unplanted_row_is_enough_to_refuse() -> None:
    """The threshold is one, not a fraction. A single real row in the window means
    the written values are derived from live data."""
    assert assess(WindowCensus(13, 12, 2, 2, 0)).refused is True
    assert assess(WindowCensus(12, 12, 3, 2, 0)).refused is True
    assert assess(WindowCensus(12, 12, 2, 2, 1)).refused is True


def test_a_spec_whose_planted_count_exceeds_its_total_refuses_as_broken() -> None:
    """Fails CLOSED on an incoherent census. If the planted query does not select a
    subset of the leg's rows, the two counts describe different row sets and neither
    means what the caller thinks -- so the guard must not read the negative
    difference as "extra isolation"."""
    verdict = assess(WindowCensus(5, 9, 2, 2, 0))
    assert verdict.refused is True
    assert any(r.startswith("broken spec:") for r in verdict.reasons)


def test_a_missing_planted_predicate_fails_closed() -> None:
    """The failure mode that matters most in practice: a spec whose planted query
    matches nothing (wrong prefix, renamed column) reads as planted=0, which
    refuses. A guard that failed open here would be worse than no guard, because it
    would look green."""
    assert assess(WindowCensus(12, 0, 2, 0, 0)).refused is True


# =============================================================================
# Spec shape -- the census cannot silently widen or start writing
# =============================================================================

_SPECS = (
    per_hcp_rollup_spec(
        test_file="f.py",
        start="2024-01-01",
        end="2024-01-31",
        hcp_like="hcp_abc_%",
        trigger_like="tr_abc_%",
    ),
    territory_rollup_spec(
        test_file="g.py",
        start="2024-06-01",
        end="2024-06-04",
        territory_like="%_abc",
        teardown_deletes_window=True,
    ),
    territory_rollup_spec(
        test_file="g2.py",
        start="2024-06-01",
        end="2024-06-04",
        territory_like="%_abc",
        teardown_deletes_window=False,
    ),
    adherence_spec(
        test_file="h.py",
        start="2024-01-01",
        end="2024-01-31",
        journey_like="pj_abc_%",
    ),
)


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.test_file)
def test_every_census_query_is_read_only(spec: WriteWindowSpec) -> None:
    """The guard must not be able to write. It also runs inside a READ ONLY
    transaction, so this is the second of two independent barriers."""
    for q in spec.queries:
        assert re.match(r"^\s*SELECT\b", q.sql), (q.leg, q.sql[:60])
        assert not re.search(
            r"\b(INSERT|UPDATE|DELETE|TRUNCATE|ALTER|DROP|CREATE|GRANT)\b", q.sql, re.I
        ), (q.leg, q.sql)


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.test_file)
def test_every_census_query_binds_its_params_instead_of_interpolating(
    spec: WriteWindowSpec,
) -> None:
    """Every value reaches Postgres as a bound param. The planted prefixes come
    from a per-run uuid, so an interpolated spec would also be an injection seam in
    a file that writes to prod."""
    for q in spec.queries:
        for name in q.params:
            assert f"%({name})s" in q.sql, (q.leg, name)
        for placeholder in re.findall(r"%\((\w+)\)s", q.sql):
            assert placeholder in q.params, (q.leg, placeholder)


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.test_file)
def test_every_counting_query_is_bounded_by_the_window_or_is_deliberately_total(
    spec: WriteWindowSpec,
) -> None:
    """A census query with no window bound would count the whole table and refuse
    everything for ever -- the no-op-by-over-refusal failure.

    Two legs are total ON PURPOSE and are listed by name rather than waved through
    by a pattern: the territory key space is every ``territory_id`` in
    ``hcp_profiles`` (that IS the cross join's reach, so bounding it by the window
    would understate the hazard), and the ``WHERE false`` legs are the
    by-construction-empty ones.
    """
    for q in spec.queries:
        bounded = "%(start)s" in q.sql and "%(end)s" in q.sql
        deliberately_total = "FROM hcp_profiles" in q.sql or "WHERE false" in q.sql
        assert bounded or deliberately_total, (spec.test_file, q.leg, q.sql)


def test_a_spec_must_declare_its_five_legs_in_order() -> None:
    """The census is unpacked positionally into WindowCensus, so a reordered or
    short spec would assign counts to the wrong legs -- silently, and in the
    permissive direction. The constructor refuses instead."""
    good = _SPECS[0].queries
    with pytest.raises(ValueError, match="queries must be"):
        WriteWindowSpec(test_file="x.py", window_description="w", queries=tuple(reversed(good)))
    with pytest.raises(ValueError, match="queries must be"):
        WriteWindowSpec(test_file="x.py", window_description="w", queries=good[:4])
    with pytest.raises(ValueError, match="queries must be"):
        WriteWindowSpec(
            test_file="x.py",
            window_description="w",
            queries=(*good[:4], CensusQuery("typo_leg", "l", "SELECT 1", {})),
        )


def test_the_metric_type_constant_matches_both_etls() -> None:
    """Redeclared, not imported, so the guard does not depend on what it guards --
    which makes a drift check mandatory. A rename that reached only the ETLs would
    otherwise make the teardown-reach leg count zero and permit everything."""
    from src.etl.business_metrics_per_hcp_etl import METRIC_TYPE as per_hcp_metric_type
    from src.etl.territory_metrics_etl import PER_HCP_METRIC_TYPE as territory_metric_type

    assert PER_HCP_METRIC_TYPE == per_hcp_metric_type == territory_metric_type


def test_census_sql_renders_a_read_only_transaction_for_hand_off() -> None:
    """The live half is run by someone else from this rendering, so the rendering
    itself must carry the READ ONLY boundary rather than rely on them adding it."""
    rendered = census_sql(_SPECS[0])
    assert rendered.startswith("-- f.py")
    assert "BEGIN TRANSACTION READ ONLY;" in rendered
    assert rendered.rstrip().endswith("ROLLBACK;")
    for leg in (
        "source_rows_total",
        "source_rows_planted",
        "target_entities_total",
        "target_entities_planted",
        "teardown_reach_preexisting",
    ):
        assert f"[{leg}]" in rendered
