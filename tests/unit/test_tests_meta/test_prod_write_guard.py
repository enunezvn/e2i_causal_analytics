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
    TERRITORY_ACTIVE_HCP_LOOKBACK_DAYS,
    CensusQuery,
    WindowCensus,
    WriteWindowSpec,
    adherence_spec,
    assess,
    census_sql,
    per_hcp_rollup_spec,
    selected_metric_dates_sql,
    territory_arrival_spec,
    territory_rollup_spec,
)

#: A stand-in for the ETL's ``metric_dates`` CTE, so the shape tests stay hermetic. The
#: pin tests below use the ETL's REAL constant.
_FAKE_METRIC_DATES_CTE = """
metric_dates AS (
    -- a comment, like the real one carries
    SELECT DISTINCT bm.metric_date FROM business_metrics bm
     WHERE bm.metric_type = %(per_hcp_metric_type)s AND bm.hcp_id IS NOT NULL
       AND bm.created_at >= %(start_date)s AND bm.created_at < %(end_date)s
)"""


def _arrival_spec(cte: str = _FAKE_METRIC_DATES_CTE, **overrides: object) -> WriteWindowSpec:
    kwargs: dict = {
        "test_file": "arr.py",
        "metric_dates_cte": cte,
        "start": "2019-01-06 21:45:00+00",
        "end": "2019-01-14 03:45:00+00",
        "hcp_like": "hl_abc_%",
        "trigger_like": "trlate_abc_%",
        "territory_like": "T_LATE_abc%",
    }
    kwargs.update(overrides)
    return territory_arrival_spec(**kwargs)


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
    _arrival_spec(),
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

    NARROWED after the dbd1063e7 defect. The old version allowed any query naming
    ``hcp_profiles`` through as "deliberately total", and that blanket is exactly the
    hole the defect slipped past: an unconditional key-space leg refused every
    territory suite alive. The exemption is now tied to the LEG, so only the two
    key-space legs may be unbounded, and only when they are live at all -- every other
    leg must carry the window or be explicitly waived.
    """
    may_be_total = {"target_entities_total", "target_entities_planted"}
    for q in spec.queries:
        # The arrival census carries the run's OWN bounds, spelled the ETL's way.
        bounded = ("%(start)s" in q.sql and "%(end)s" in q.sql) or (
            "%(start_date)s" in q.sql and "%(end_date)s" in q.sql
        )
        waived = q.sql.rstrip().endswith("WHERE false")
        assert bounded or waived or q.leg in may_be_total, (spec.test_file, q.leg, q.sql)


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


# =============================================================================
# The leg-2 / leg-3 pairing (fix for the dbd1063e7 defect)
# =============================================================================


def _leg(spec: WriteWindowSpec, leg: str) -> CensusQuery:
    return next(q for q in spec.queries if q.leg == leg)


def _is_waived(q: CensusQuery) -> bool:
    """A waived leg is a query that cannot count anything -- `WHERE false`."""
    return q.sql.rstrip().endswith("WHERE false")


@pytest.mark.parametrize("sweeps", [True, False])
def test_exactly_one_of_the_key_space_and_teardown_legs_is_live(sweeps: bool) -> None:
    """BOTH directions matter, which is why this is parametrized rather than two
    assertions about one spec.

    The hazard is not "foreign rows are written", it is "foreign rows SURVIVE". A
    window-scoped teardown sweeps the cross join's foreign rows, so leg 2 is waived
    and leg 3 carries the safety; a prefix-scoped teardown leaves them, so leg 2
    applies and leg 3 has nothing to say. Asking both at once was the shipped defect:
    leg 2 unconditional refused every territory suite on a live DB.
    """
    spec = territory_rollup_spec(
        test_file="t.py",
        start="2019-01-01",
        end="2019-01-21",
        territory_like="T_x%",
        teardown_deletes_window=sweeps,
    )
    keyspace_waived = _is_waived(_leg(spec, "target_entities_total"))
    teardown_waived = _is_waived(_leg(spec, "teardown_reach_preexisting"))
    assert keyspace_waived is sweeps, "leg 2 must be waived exactly when the teardown sweeps"
    assert teardown_waived is not sweeps, "leg 3 must be live exactly when the teardown sweeps"
    # Complementary, not merely different: never both live, never both waived.
    assert keyspace_waived != teardown_waived


def test_the_derivation_leg_is_never_waived_by_the_teardown_setting() -> None:
    """The pairing is about where WRITES land, not about where values come from.
    A teardown that sweeps the window does not make it acceptable to derive those
    values from real rows, so leg 1 must stay live in both configurations."""
    for sweeps in (True, False):
        spec = territory_rollup_spec(
            test_file="t.py",
            start="2019-01-01",
            end="2019-01-21",
            territory_like="T_x%",
            teardown_deletes_window=sweeps,
        )
        assert not _is_waived(_leg(spec, "source_rows_total"))
        assert "%(start)s" in _leg(spec, "source_rows_total").sql


def test_the_measured_live_verdicts_are_reproduced_from_their_counts() -> None:
    """Regression guard over the dispatcher's read-only measurement of 2026-09-17.

    These are real counts from the real database, pinned here so a future change to
    `assess` cannot silently flip a verdict that was checked against production data.
    The census QUERIES are what changed in this commit; the decision must still read
    these numbers the same way.
    """
    # test_business_metrics_per_hcp_etl_integration.py, [2024-01-01, +30d)
    assert assess(WindowCensus(438, 0, 386, 0, 0)).refused is True
    # test_patient_adherence_etl_integration.py, [2024-01-01, 2024-01-31)
    assert assess(WindowCensus(285, 0, 285, 0, 0)).refused is True
    # test_territory_metrics_etl_integration.py, [2024-06-01, +3d): leg 2 ALONE.
    territory_integration = assess(WindowCensus(0, 0, 40, 0, 0))
    assert territory_integration.refused is True
    assert len(territory_integration.reasons) == 1
    assert territory_integration.reasons[0].startswith("key space:")
    # The counterfactual the dispatcher ran: waiving leg 2 unconditionally -- the
    # simpler, WRONG design -- permits that same file. Measured, not assumed.
    assert assess(WindowCensus(0, 0, 0, 0, 0)).refused is False
    # test_per_hcp_rollup_late_arrival.py: per-HCP and adherence phases were already 0.
    assert assess(WindowCensus(0, 0, 0, 0, 0)).refused is False


def test_the_late_arrival_territory_census_now_permits() -> None:
    """The defect, end to end, in the numbers that exposed it.

    Live counts for that file's territory phase were leg1 0/0, leg2 40/0, leg3 0.
    Shipped, leg 2 was asked unconditionally and the file REFUSED. With the pairing,
    leg 2 is waived (its teardown sweeps the window) and leg 3 -- measured 0, against
    a positive control returning 1840 over territory_metrics' real span -- permits.
    """
    assert assess(WindowCensus(0, 0, 40, 0, 0)).refused is True  # as shipped
    assert assess(WindowCensus(0, 0, 0, 0, 0)).refused is False  # with leg 2 waived
    spec = territory_rollup_spec(
        test_file="test_per_hcp_rollup_late_arrival.py",
        start="2019-01-01",
        end="2019-01-21",
        territory_like="T_LATE_x%",
        teardown_deletes_window=True,
    )
    assert _is_waived(_leg(spec, "target_entities_total"))
    assert not _is_waived(_leg(spec, "teardown_reach_preexisting"))


# =============================================================================
# The per-HCP reconcile scope (fix 2): window vs arrival disagree
# =============================================================================


def test_the_reconcile_scope_follows_the_variant_and_the_two_disagree() -> None:
    """The arrival reconcile scopes by affected_dates, NOT by the window
    (_PER_HCP_RECONCILE_SCOPE_BY_ARRIVAL). Censusing an arrival run with the window
    scope understates it, and an understating leg is worse than a missing one because
    it reads as a measurement.

    Both directions are asserted, so an implementation that ignored `window_column`
    would fail one of them whichever scope it hardcoded. What this CANNOT show is
    that Postgres agrees -- it is a check on the predicate we emit, at the SQL-text
    layer; the live half is what showed the counts.
    """
    common = {
        "test_file": "f.py",
        "start": "2019-01-02 03:00:55+00",
        "end": "2019-01-21 00:00:00+00",
        "hcp_like": "hcplate_x_%",
        "trigger_like": "trlate_x_%",
    }
    arrival = _leg(
        per_hcp_rollup_spec(**common, window_column="created_at"),
        "teardown_reach_preexisting",
    )
    windowed = _leg(
        per_hcp_rollup_spec(**common, window_column="trigger_timestamp"),
        "teardown_reach_preexisting",
    )

    # The arrival scope is a set membership with no lower bound tied to `start` --
    # which is exactly why it reaches 2019-01-01 while the window scope starts at
    # 2019-01-02 and misses it.
    assert "b.metric_date IN (" in arrival.sql
    assert "t.created_at >= %(start)s" in arrival.sql
    assert "b.metric_date >= %(start)s::DATE" not in arrival.sql

    # The explicit variant keeps the window bounds and grows no subquery.
    assert "b.metric_date >= %(start)s::DATE" in windowed.sql
    assert "b.metric_date <  %(end)s::DATE" in windowed.sql or (
        "b.metric_date < %(end)s::DATE" in windowed.sql
    )
    assert "b.metric_date IN (" not in windowed.sql

    # And they really are different statements, not two spellings of one.
    assert re.sub(r"\s+", " ", arrival.sql) != re.sub(r"\s+", " ", windowed.sql)


# =============================================================================
# The territory ARRIVAL census (#2213): the run's own selection, not a date range
# =============================================================================


def _normalised(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def test_the_arrival_census_composes_the_runs_own_selection_verbatim() -> None:
    """The gap #2213 closed: the ARRIVAL run selects dates with
    ``_TERRITORY_METRIC_DATES_BY_ARRIVAL`` (a per-HCP date written in the window, or within
    30 days of a trigger that ARRIVED in it), which a ``metric_date`` range cannot mirror.
    The census therefore embeds the CTE the caller hands it -- the ETL's own constant, so
    the two cannot drift -- in every live leg, with the run's own bounds and its
    ``hcp_id IS NOT NULL``."""
    from src.etl.territory_metrics_etl import _TERRITORY_METRIC_DATES_BY_ARRIVAL

    spec = _arrival_spec(_TERRITORY_METRIC_DATES_BY_ARRIVAL)
    cte = _normalised(_TERRITORY_METRIC_DATES_BY_ARRIVAL)
    for leg in ("source_rows_total", "source_rows_planted", "teardown_reach_preexisting"):
        sql = _normalised(_leg(spec, leg).sql)
        assert cte in sql, leg
        assert "hcp_id IS NOT NULL" in sql, leg
        for name in ("start_date", "end_date", "per_hcp_metric_type"):
            assert name in _leg(spec, leg).params, (leg, name)
    # The selection the test records for its teardown is the SAME subquery the legs use.
    selection = _normalised(selected_metric_dates_sql(_TERRITORY_METRIC_DATES_BY_ARRIVAL))
    assert cte in selection
    inner = selection[selection.index("(WITH") : selection.rindex(")") + 1]
    for leg in ("source_rows_total", "source_rows_planted", "teardown_reach_preexisting"):
        assert inner in _normalised(_leg(spec, leg).sql), leg


def test_the_arrival_census_lookback_is_the_etls_active_hcp_window() -> None:
    """A trigger contributes to a date's ``active_hcp_count`` when it falls in the ETL's
    30-day window ending on the date (inclusive). The census reads triggers by the same
    window; the number is pinned to the ETL's SQL by extraction, not by assertion of a
    literal, so a widened ETL window turns this red instead of narrowing the census."""
    from src.etl.territory_metrics_etl import (
        _TERRITORY_METRIC_DATES_BY_ARRIVAL,
        _TERRITORY_ROLLUP_CTES_TEMPLATE,
    )

    etl_days = {
        int(m)
        for m in re.findall(
            r"t\.trigger_timestamp >= \w+\.metric_date - INTERVAL '(\d+) days'",
            _TERRITORY_ROLLUP_CTES_TEMPLATE + _TERRITORY_METRIC_DATES_BY_ARRIVAL,
        )
    }
    assert etl_days == {TERRITORY_ACTIVE_HCP_LOOKBACK_DAYS}
    spec = _arrival_spec()
    for leg in ("source_rows_total", "source_rows_planted"):
        sql = _normalised(_leg(spec, leg).sql)
        assert (
            f"t.trigger_timestamp >= sd.metric_date - INTERVAL '{TERRITORY_ACTIVE_HCP_LOOKBACK_DAYS} days'"
            in sql
        ), leg
        assert "t.trigger_timestamp < sd.metric_date + INTERVAL '1 day'" in sql, leg


def test_the_arrival_census_counts_per_hcp_rows_and_triggers_and_owns_them_by_prefix() -> None:
    """Leg 1 counts what the aggregate READS for a selected date: the per-HCP rows on it
    (total_trx / total_nrx) and the triggers in its lookback (active_hcp_count). The planted
    leg is the same two counts under the file's prefixes, so it is a subset by construction
    and a foreign row on either side reads as unplanted."""
    spec = _arrival_spec()
    total = _normalised(_leg(spec, "source_rows_total").sql)
    planted = _normalised(_leg(spec, "source_rows_planted").sql)
    for sql in (total, planted):
        assert "FROM business_metrics bm" in sql and "FROM triggers t" in sql
    assert "LIKE" not in total
    assert "bm.hcp_id LIKE %(hcp_like)s" in planted
    assert "t.trigger_id LIKE %(trigger_like)s" in planted
    assert _leg(spec, "source_rows_planted").params["hcp_like"] == "hl_abc_%"
    assert _leg(spec, "source_rows_planted").params["trigger_like"] == "trlate_abc_%"


def test_the_arrival_census_waives_the_key_space_leg_and_keeps_the_teardown_leg() -> None:
    """Same pairing as ``territory_rollup_spec(teardown_deletes_window=True)``: the file
    deletes ``territory_metrics`` for EVERY selected date, so the cross join's foreign rows
    do not survive (leg 2 waived) and what matters is what pre-exists on those dates
    (leg 3 live, on the selected set, not on a range)."""
    spec = _arrival_spec()
    assert _is_waived(_leg(spec, "target_entities_total"))
    assert _is_waived(_leg(spec, "target_entities_planted"))
    teardown = _normalised(_leg(spec, "teardown_reach_preexisting").sql)
    assert not _is_waived(_leg(spec, "teardown_reach_preexisting"))
    assert "FROM territory_metrics m" in teardown
    assert "m.metric_date IN (WITH" in teardown
    assert "m.territory_id NOT LIKE %(territory_like)s" in teardown
    assert "m.metric_date >=" not in teardown


@pytest.mark.parametrize(
    ("cte", "why"),
    [
        ("affected_dates AS (SELECT 1)", "metric_dates AS"),
        (_FAKE_METRIC_DATES_CTE + "; DELETE FROM territory_metrics", "semicolon"),
        (_FAKE_METRIC_DATES_CTE + ", extra AS (SELECT 1)", "single CTE"),
        (_FAKE_METRIC_DATES_CTE.replace("%(end_date)s", "now()"), "end_date"),
        (_FAKE_METRIC_DATES_CTE.replace("%(per_hcp_metric_type)s", "'x'"), "per_hcp_metric_type"),
    ],
    ids=["other-name", "second-statement", "second-cte", "missing-bound", "missing-type"],
)
def test_the_arrival_census_refuses_a_cte_that_is_not_one_metric_dates_selection(
    cte: str, why: str
) -> None:
    """The legs read ``FROM metric_dates`` and bind the run's three params (codex r1-5:
    the parameter is raw SQL, so its shape is checked at construction rather than left
    for Postgres inside the live half). One CTE named metric_dates, no second statement
    or clause, every run param present."""
    with pytest.raises(ValueError, match=why):
        _arrival_spec(cte)


def test_the_arrival_census_shape_check_reads_the_sql_not_its_comments() -> None:
    """The first version of the check read the raw text and refused the ETL's REAL
    constant, whose comments carry a ';' and parentheses. The check must look at the SQL
    proper; the real constant is the positive control."""
    from src.etl.territory_metrics_etl import _TERRITORY_METRIC_DATES_BY_ARRIVAL

    assert "; and" in _TERRITORY_METRIC_DATES_BY_ARRIVAL  # the comment that bit
    _arrival_spec(_TERRITORY_METRIC_DATES_BY_ARRIVAL)
    _arrival_spec(_FAKE_METRIC_DATES_CTE.replace("-- a comment", "-- a (comment); with '('"))


def test_the_disproof_counts_are_read_as_measured() -> None:
    """Measured live 2026-09-22 with an UNCOMMITTED plant (rolled back): a foreign per-HCP
    row dated 2019-01-30 reached by the file's own Tuesday trigger arriving 2019-01-14, and
    one dated 2019-03-15 written inside the run's lookback. The range census read
    0/0/0/0/0 and PERMITTED; the arrival census read 3 source rows (two foreign per-HCP
    rows, one owned trigger), 1 planted, and refused on derivation alone."""
    assert assess(WindowCensus(0, 0, 0, 0, 0)).refused is False  # the range census
    verdict = assess(WindowCensus(3, 1, 0, 0, 0))  # the arrival census
    assert verdict.refused is True
    assert [r.split(":")[0] for r in verdict.reasons] == ["derivation"]


def test_census_sql_strips_line_comments_before_collapsing_to_one_line() -> None:
    """The ETL's CTE carries ``--`` comments. Collapsed onto one line, the first comment
    would swallow the rest of the statement, and the hand-off script would run nothing
    (or worse, something else). Comments go; the statement stays whole."""
    rendered = census_sql(_arrival_spec())
    statements = [line for line in rendered.splitlines() if line.startswith("SELECT ")]
    assert len(statements) == 5
    for statement in statements:
        assert "--" not in statement, statement[:80]
        assert "FROM metric_dates" in statement or statement.endswith("WHERE false;")


def test_census_sql_keeps_a_double_dash_inside_a_string_literal() -> None:
    """codex r1-4: the comment stripper must not treat ``--`` inside a quoted literal as
    a comment, or the rest of that statement (placeholders included) would go with it."""
    spec = WriteWindowSpec(
        test_file="q.py",
        window_description="w",
        queries=(
            CensusQuery(
                "source_rows_total",
                "l",
                "SELECT count(*) FROM triggers t -- a real comment\n"
                " WHERE t.trigger_id LIKE '--%' AND t.created_at >= %(start)s",
                {"start": "2019-01-01"},
            ),
            *(
                CensusQuery(leg, "l", "SELECT count(*) FROM triggers WHERE false", {})
                for leg in (
                    "source_rows_planted",
                    "target_entities_total",
                    "target_entities_planted",
                    "teardown_reach_preexisting",
                )
            ),
        ),
    )
    rendered = census_sql(spec)
    statement = next(line for line in rendered.splitlines() if line.startswith("SELECT "))
    assert statement == (
        "SELECT count(*) FROM triggers t WHERE t.trigger_id LIKE '--%' "
        "AND t.created_at >= %(start)s;"
    )


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
