"""Data-derived refusal for integration tests that write to the live database.

Owner-approved addition outside the frozen canonical-TRx plan (2026-09-17).

Why this exists
---------------
Five integration files run the real ETLs against the real database when
``E2I_DB_INTEGRATION=1``. That env var is the *only* thing standing between them
and a live write:

* ``tests/conftest.py`` calls ``load_dotenv(override=True)``, so ``SUPABASE_DB_URL``
  is always set inside pytest on this box — the skip guard's left conjunct is
  permanently satisfied, and ``override=True`` means a ``.env`` entry would beat an
  explicit ``E2I_DB_INTEGRATION=0`` in the shell rather than lose to it.
* Nothing in ``deploy.yml`` or ``backend-tests.yml`` sets the var. The one automated
  setter is ``scripts/deploy/realdb_suite_gate.sh``, scoped to
  ``tests/unit/test_database/learning_loop/`` only.

So the residual risk is a human or an agent typing
``E2I_DB_INTEGRATION=1 pytest tests/integration/``. Measured read-only on the live
DB: the January-2024 window of the per-HCP suite holds 444 triggers, **all 444
unplanted**, with zero per-HCP rollup rows on those dates — so such a run does not
overwrite anything, it *fabricates* a month of plausible rollup rows, and the
teardown (which deletes by planted ``hcp_id`` prefix) removes none of them.

The principle
-------------
**Derive the refusal from the DATA, not from the environment.** A test whose writes
would be derived from rows it did not plant, or would land on entity keys it does
not own, must refuse before it writes — whatever the environment says. This module
is the generalisation of the pre-flight check
``test_per_hcp_rollup_late_arrival.py`` already carried inline, which is the one
file in the family that was written with this hazard in mind.

What this guard proves, and what it does not
--------------------------------------------
It proves **input isolation**: every row the writes derive values from, and every
entity key they target, belongs to this test run. It does **not** prove
*restoration* — that the database is byte-identical after teardown. Those are
different claims, and conflating them is how a check ends up unable to fail for
the reason it exists. Restoration is what a live certification proves; this guard
is what makes running one safe to attempt.

Three independent legs, because the family has three distinct hazards
---------------------------------------------------------------------
1. **Derivation.** The ETL computes output values from source rows inside the
   window (``triggers`` for the per-HCP rollup, per-HCP ``business_metrics`` rows
   for the territory rollup, ``patient_journeys`` for the adherence ETL). An
   unplanted source row in the window means the written values are derived from
   real data.
2. **Key space.** The write's target keys are not always confined to planted
   entities. ``territory_metrics_etl``'s ``territories`` CTE is
   ``SELECT DISTINCT territory_id FROM hcp_profiles WHERE territory_id IS NOT NULL``
   **cross joined** with every date in the window, so it emits a row for every
   *real* territory on every windowed date even when no planted source row exists.
   Leg 1 cannot see that: the counts can be zero while the hazard is present.
3. **Teardown reach.** Two files delete by window rather than by planted prefix
   (``DELETE FROM territory_metrics WHERE metric_date >= … AND < …`` in
   ``test_etl_provenance_inheritance_895.py``, and ``… WHERE metric_date IN (…)``
   in ``test_per_hcp_rollup_late_arrival.py``). Any pre-existing row inside that
   reach is destroyed by the cleanup, not by the ETL.

A guard with only leg 1 would permit the territory suite; a guard with only leg 2
would permit the per-HCP suite. All three are required for the refusal to
discriminate file by file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "CensusQuery",
    "TERRITORY_ACTIVE_HCP_LOOKBACK_DAYS",
    "WindowCensus",
    "Verdict",
    "WriteWindowSpec",
    "adherence_spec",
    "assess",
    "census_sql",
    "per_hcp_rollup_spec",
    "require_isolated_windows",
    "selected_metric_dates_sql",
    "territory_arrival_spec",
    "territory_rollup_spec",
]

#: ``business_metrics.metric_type`` for the per-HCP rollup's own rows. Redeclared here
#: rather than imported so the guard does not depend on the ETL it guards; a unit test
#: pins it equal to both ETLs' constants, so a rename cannot silently widen the census.
PER_HCP_METRIC_TYPE: str = "per_hcp_rollup"

#: The territory rollup's "active HCP" window: a trigger counts towards a date when its
#: ``trigger_timestamp`` falls in the 30 days ending on that date, inclusive
#: (``territory_metrics_etl`` active_hcp_per_territory_date). Redeclared like the metric
#: type; a unit test extracts the number from the ETL's SQL and pins this equal to it.
TERRITORY_ACTIVE_HCP_LOOKBACK_DAYS: int = 30


@dataclass(frozen=True)
class CensusQuery:
    """One read-only count the guard needs, with its own SQL and params.

    ``leg`` names which hazard the count feeds, so a refusal message can say which
    of the three tripped rather than just that something did.
    """

    leg: str
    label: str
    sql: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class WindowCensus:
    """Measured counts for one file's write window. Every field is a row count.

    ``*_total`` counts every row the leg reaches; ``*_planted`` counts the subset
    this test run created. The difference is what the guard refuses on, so a spec
    whose planted predicate is wrong fails CLOSED (planted reads as 0 → refuse)
    rather than open.
    """

    source_rows_total: int
    source_rows_planted: int
    target_entities_total: int
    target_entities_planted: int
    teardown_reach_preexisting: int

    @property
    def unplanted_source_rows(self) -> int:
        return self.source_rows_total - self.source_rows_planted

    @property
    def unplanted_target_entities(self) -> int:
        return self.target_entities_total - self.target_entities_planted


@dataclass(frozen=True)
class Verdict:
    refused: bool
    reasons: tuple[str, ...]

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return not self.refused


def assess(census: WindowCensus) -> Verdict:
    """Decide, from counts alone, whether the window is isolated enough to write.

    Pure: no DB, no environment, no clock. Every leg is reported, not just the
    first, so a refusal names every hazard present instead of sending the reader
    round the loop once per leg.

    A negative difference (more planted than total) means the spec's planted
    predicate does not select a subset of the leg's rows -- the two queries are
    describing different row sets, so neither count means what the caller thinks.
    That is a broken spec, and a broken spec refuses.
    """
    reasons: list[str] = []

    if census.unplanted_source_rows > 0:
        reasons.append(
            f"derivation: {census.unplanted_source_rows} source row(s) in the window were "
            f"not planted by this run ({census.source_rows_total} total, "
            f"{census.source_rows_planted} planted) -- the written values would be "
            f"derived from live data"
        )
    if census.unplanted_target_entities > 0:
        reasons.append(
            f"key space: {census.unplanted_target_entities} target entity key(s) are not "
            f"owned by this run ({census.target_entities_total} total, "
            f"{census.target_entities_planted} planted) -- rows would be written against "
            f"real entities that the prefix-scoped teardown cannot remove"
        )
    if census.teardown_reach_preexisting > 0:
        reasons.append(
            f"teardown reach: {census.teardown_reach_preexisting} pre-existing row(s) sit "
            f"inside the window this file's cleanup deletes wholesale -- the teardown, not "
            f"the ETL, would destroy them"
        )
    if census.unplanted_source_rows < 0 or census.unplanted_target_entities < 0:
        reasons.append(
            "broken spec: a planted count exceeds its total "
            f"(source {census.source_rows_planted}/{census.source_rows_total}, "
            f"entities {census.target_entities_planted}/{census.target_entities_total}), "
            "so the planted predicate does not select a subset of the leg's rows"
        )

    return Verdict(refused=bool(reasons), reasons=tuple(reasons))


@dataclass(frozen=True)
class WriteWindowSpec:
    """Everything the guard needs to census one file, declared by that file.

    The five queries map one-to-one onto :class:`WindowCensus`'s five fields, in
    order. Keeping them declarative is what lets the live half be run and reviewed
    without reading any test's control flow.
    """

    test_file: str
    window_description: str
    queries: Sequence[CensusQuery]

    def __post_init__(self) -> None:
        legs = [q.leg for q in self.queries]
        expected = [
            "source_rows_total",
            "source_rows_planted",
            "target_entities_total",
            "target_entities_planted",
            "teardown_reach_preexisting",
        ]
        if legs != expected:
            raise ValueError(f"{self.test_file}: queries must be {expected}, got {legs}")


def _strip_line_comments(sql: str) -> str:
    """Drop ``--`` line comments. A ``--`` inside a quoted literal is not a comment
    (codex r1-4): the alternation consumes literals first, so the comment branch never
    sees their contents."""
    return re.sub(
        r"'[^']*'|--[^\n]*", lambda m: m.group(0) if m.group(0).startswith("'") else "", sql
    )


def census_sql(spec: WriteWindowSpec) -> str:
    """Render a spec as a copy-pasteable read-only script, for review and hand-off.

    The guard runs the same statements through psycopg2 with bound params; this
    renderer exists so the queries can be read and run by a human (or by the
    dispatcher who owns the live half) without executing any test.
    """
    lines = [
        f"-- {spec.test_file}",
        f"-- window: {spec.window_description}",
        "BEGIN TRANSACTION READ ONLY;",
    ]
    for q in spec.queries:
        lines.append(f"-- [{q.leg}] {q.label}")
        lines.append(f"--   params: {dict(q.params)!r}")
        # The arrival census embeds the ETL's CTE, which carries ``--`` comments; on one
        # line the first of them would comment out the rest of the statement.
        lines.append(" ".join(_strip_line_comments(q.sql).split()) + ";")
    lines.append("ROLLBACK;")
    return "\n".join(lines)


def per_hcp_rollup_spec(
    *,
    test_file: str,
    start: Any,
    end: Any,
    hcp_like: str,
    trigger_like: str,
    window_column: str = "trigger_timestamp",
) -> WriteWindowSpec:
    """Census for a file that runs ``business_metrics_per_hcp_etl`` on explicit dates.

    Derivation is windowed on ``trigger_timestamp``, but the rollup then re-reads
    EVERY trigger whose ``DATE(trigger_timestamp)`` falls on an affected date
    (``business_metrics_per_hcp_etl.py`` affected_dates -> line 189), so the census
    uses the affected-date formulation rather than the raw window. For whole-day
    windows the two coincide; for a partial-day window they do not, and the wider
    one is the honest count.

    Teardown-reach leg: the reconcile DELETEs any per-HCP row in scope that the
    recomputation does not reproduce, including rows this run never planted -- and
    **the two variants scope it differently**
    (``business_metrics_per_hcp_etl._PER_HCP_RECONCILE_SCOPE_BY_WINDOW`` vs
    ``_BY_ARRIVAL``). The explicit variant scopes by ``metric_date >= start AND < end``;
    an ARRIVAL run scopes by ``metric_date IN (SELECT metric_date FROM affected_dates)``,
    which has no lower bound tied to ``start`` at all. Censusing an arrival run with the
    window scope UNDERSTATES it: on the late-arrival window it would start at 2019-01-02
    and miss ``TUESDAY = 2019-01-01``, a date that run really does touch. An
    understating leg is worse than a missing one, because it reads as a measurement.
    """
    if window_column not in ("trigger_timestamp", "created_at"):
        raise ValueError(f"unknown per-HCP window column {window_column!r}")
    affected = (
        "SELECT DISTINCT DATE(t.trigger_timestamp) AS d FROM triggers t "
        f" WHERE t.{window_column} >= %(start)s AND t.{window_column} < %(end)s"
    )
    window = {"start": start, "end": end}
    # Mirror the reconcile's own scope, which differs per variant (see the docstring).
    reconcile_scope = (
        f"b.metric_date IN ({affected})"
        if window_column == "created_at"
        else "b.metric_date >= %(start)s::DATE AND b.metric_date < %(end)s::DATE"
    )
    return WriteWindowSpec(
        test_file=test_file,
        window_description=f"per-HCP rollup, triggers by {window_column} in [{start}, {end})",
        queries=(
            CensusQuery(
                "source_rows_total",
                "triggers on every affected metric_date",
                f"SELECT count(*) FROM triggers t WHERE DATE(t.trigger_timestamp) IN ({affected})",
                window,
            ),
            CensusQuery(
                "source_rows_planted",
                "of those, triggers this run planted",
                f"SELECT count(*) FROM triggers t WHERE DATE(t.trigger_timestamp) "
                f"IN ({affected}) AND t.trigger_id LIKE %(trigger_like)s",
                {**window, "trigger_like": trigger_like},
            ),
            CensusQuery(
                "target_entities_total",
                "distinct hcp_id the rollup would write rows for",
                f"SELECT count(DISTINCT t.hcp_id) FROM triggers t "
                f"WHERE DATE(t.trigger_timestamp) IN ({affected})",
                window,
            ),
            CensusQuery(
                "target_entities_planted",
                "of those, hcp_id this run owns",
                f"SELECT count(DISTINCT t.hcp_id) FROM triggers t "
                f"WHERE DATE(t.trigger_timestamp) IN ({affected}) "
                f"AND t.hcp_id LIKE %(hcp_like)s",
                {**window, "hcp_like": hcp_like},
            ),
            CensusQuery(
                "teardown_reach_preexisting",
                "per-HCP rollup rows in the reconcile's own scope that are not ours",
                "SELECT count(*) FROM business_metrics b "
                " WHERE b.metric_type = %(metric_type)s AND b.hcp_id IS NOT NULL "
                f"   AND {reconcile_scope} "
                "   AND b.hcp_id NOT LIKE %(hcp_like)s",
                {**window, "hcp_like": hcp_like, "metric_type": PER_HCP_METRIC_TYPE},
            ),
        ),
    )


def territory_rollup_spec(
    *,
    test_file: str,
    start: Any,
    end: Any,
    territory_like: str,
    teardown_deletes_window: bool,
) -> WriteWindowSpec:
    """Census for a file that runs ``territory_metrics_etl`` on explicit dates.

    The key-space leg is the one a derivation-only guard cannot see: ``territories AS
    (SELECT DISTINCT territory_id FROM hcp_profiles WHERE territory_id IS NOT NULL)``
    is CROSS JOINed with every date in the window, so the rollup emits a
    ``territory_metrics`` row for every real territory on every windowed date even when
    no planted source row exists.

    **Legs 2 and 3 are complementary, and ``teardown_deletes_window`` selects between
    them.** The hazard is not "foreign rows are written" -- it is "foreign rows
    SURVIVE". So:

    * teardown **prefix-scoped** (``False``): the cross join's foreign rows are never
      cleaned up, so leg 2 applies and leg 3 is empty by construction.
    * teardown **window-scoped** (``True``): those same foreign rows are swept by the
      cleanup, so leg 2 is waived -- and leg 3 carries the safety instead, because a
      window-wide DELETE destroys anything that pre-existed there.

    Asking leg 2 unconditionally was the shipped defect (``dbd1063e7``): it counts every
    territory in ``hcp_profiles`` with no window bound, so it refused EVERY territory
    suite on a live database, including correctly-isolated ones. That is a proxy for
    the real condition -- it asks "are there foreign keys in the key space?" when what
    matters is "will foreign rows survive teardown?" -- and a leg that always refuses
    is as useless as one that never does.

    Measured live 2026-09-17 (read-only, by the dispatcher), which is what makes the
    conditional defensible rather than merely plausible:

    * ``test_territory_metrics_etl_integration.py`` over ``[2024-06-01, 2024-06-04)``:
      leg 1 = **0/0**, leg 3 = **0**, leg 2 = **40/0**. It is refused by leg 2 ALONE.
    * The counterfactual was run: with leg 2 waived unconditionally -- the simpler,
      wrong design -- that file **PERMITS**. So leg 2 is the only thing between it and
      a live write, and this conditional discriminates by measurement, not by
      construction.
    * ``test_per_hcp_rollup_late_arrival.py`` over ``[2019-01-01, 2019-01-21)``: the
      same ``40/0`` on leg 2, but its teardown sweeps the window, and leg 3 measures
      **0** there -- against a positive control showing leg 3's shape returns **1840**
      over ``[2026-01-01, 2027-01-01)``, the real span of ``territory_metrics``. Leg 3's
      zero is a measurement, not a dead predicate, so it can carry the safety.

    A waived leg is expressed as a false-predicate query rather than a hardcoded 0, so
    every leg is measured the same way and the census stays five statements.
    """
    window = {"start": start, "end": end}
    keyspace_total_sql = (
        "SELECT count(DISTINCT territory_id) FROM hcp_profiles WHERE false"
        if teardown_deletes_window
        else "SELECT count(DISTINCT territory_id) FROM hcp_profiles  WHERE territory_id IS NOT NULL"
    )
    keyspace_planted_sql = (
        "SELECT count(DISTINCT territory_id) FROM hcp_profiles WHERE false"
        if teardown_deletes_window
        else "SELECT count(DISTINCT territory_id) FROM hcp_profiles "
        " WHERE territory_id IS NOT NULL AND territory_id LIKE %(territory_like)s"
    )
    teardown_sql = (
        "SELECT count(*) FROM territory_metrics m "
        " WHERE m.metric_date >= %(start)s::DATE AND m.metric_date < %(end)s::DATE "
        "   AND m.territory_id NOT LIKE %(territory_like)s"
        if teardown_deletes_window
        else "SELECT count(*) FROM territory_metrics m WHERE false"
    )
    return WriteWindowSpec(
        test_file=test_file,
        window_description=f"territory rollup, metric_dates in [{start}, {end})",
        queries=(
            CensusQuery(
                "source_rows_total",
                "per-HCP rollup rows the territory aggregate reads in the window",
                "SELECT count(*) FROM business_metrics bm "
                " WHERE bm.metric_type = %(metric_type)s AND bm.hcp_id IS NOT NULL "
                "   AND bm.metric_date >= %(start)s::DATE AND bm.metric_date < %(end)s::DATE",
                {**window, "metric_type": PER_HCP_METRIC_TYPE},
            ),
            CensusQuery(
                "source_rows_planted",
                "of those, rows in a territory this run owns",
                "SELECT count(*) FROM business_metrics bm "
                " JOIN hcp_profiles hp ON hp.hcp_id = bm.hcp_id "
                " WHERE bm.metric_type = %(metric_type)s AND bm.hcp_id IS NOT NULL "
                "   AND bm.metric_date >= %(start)s::DATE AND bm.metric_date < %(end)s::DATE "
                "   AND hp.territory_id LIKE %(territory_like)s",
                {**window, "territory_like": territory_like, "metric_type": PER_HCP_METRIC_TYPE},
            ),
            CensusQuery(
                "target_entities_total",
                "territories the CROSS JOIN writes a row for (waived when the teardown"
                " sweeps the window, which removes them again)",
                keyspace_total_sql,
                {},
            ),
            CensusQuery(
                "target_entities_planted",
                "of those, territories this run owns",
                keyspace_planted_sql,
                {} if teardown_deletes_window else {"territory_like": territory_like},
            ),
            CensusQuery(
                "teardown_reach_preexisting",
                "territory_metrics rows a window-scoped teardown would delete, not ours",
                teardown_sql,
                {**window, "territory_like": territory_like} if teardown_deletes_window else {},
            ),
        ),
    )


def _selected_metric_dates_subquery(metric_dates_cte: str) -> str:
    """The dates an ARRIVAL run would rebuild, as a parenthesised subquery over the run's
    own ``metric_dates`` CTE (``WITH`` is legal inside a subquery in PostgreSQL)."""
    # The parameter is raw SQL (codex r1-5), so its shape is checked here rather than left
    # to Postgres inside the live half: exactly one CTE named metric_dates, whose body is
    # one parenthesised block, no second statement, and every run param it must bind.
    if not re.search(r"^\s*metric_dates\s+AS\s*\(", metric_dates_cte):
        raise ValueError(
            "territory arrival census: the CTE must define 'metric_dates AS (...)' -- pass "
            "territory_metrics_etl._TERRITORY_METRIC_DATES_BY_ARRIVAL"
        )
    # Checked on the SQL proper: the ETL's constant carries a ';' and parentheses inside
    # its comments, and a check that read them refused the real constant.
    body = _strip_line_comments(metric_dates_cte).strip()
    if ";" in body:
        raise ValueError("territory arrival census: the CTE must not contain a semicolon")
    depth = 0
    close_at = -1
    for i, ch in enumerate(body):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and ch == ")":
            close_at = i
            break
    if close_at != len(body) - 1:
        raise ValueError(
            "territory arrival census: the CTE must be a single CTE 'metric_dates AS ( ... )' "
            "with nothing after its closing parenthesis"
        )
    for name in ("start_date", "end_date", "per_hcp_metric_type"):
        if f"%({name})s" not in metric_dates_cte:
            raise ValueError(f"territory arrival census: the CTE must bind %({name})s")
    return f"(WITH {metric_dates_cte}\nSELECT md.metric_date FROM metric_dates md)"


def selected_metric_dates_sql(metric_dates_cte: str) -> str:
    """A read-only statement listing the dates the arrival run would select, ordered.

    The test records this set right before the run and its teardown deletes
    ``territory_metrics`` for every date in it: the run writes a row for EVERY territory
    on each selected date, and the two planted dates are not the whole selection when a
    foreign per-HCP date sits within reach. Binds the run's own params
    (``start_date``, ``end_date``, ``per_hcp_metric_type``).
    """
    return f"SELECT sel.metric_date FROM {_selected_metric_dates_subquery(metric_dates_cte)} sel ORDER BY 1"


def territory_arrival_spec(
    *,
    test_file: str,
    metric_dates_cte: str,
    start: Any,
    end: Any,
    hcp_like: str,
    trigger_like: str,
    territory_like: str,
) -> WriteWindowSpec:
    """Census for a file that runs ``territory_metrics_etl`` by ARRIVAL (#2213).

    A scheduled territory run selects its dates with
    ``territory_metrics_etl._TERRITORY_METRIC_DATES_BY_ARRIVAL``: every per-HCP date
    whose row was WRITTEN in the run window, or that lies within the 30-day active-HCP
    lookback of a trigger that ARRIVED in the window. A ``metric_date`` range cannot
    mirror that: the file's own Tuesday trigger arriving on 2019-01-14 reaches any
    foreign per-HCP date up to 2019-01-31, and a foreign per-HCP row written inside the
    lookback is selected whatever its date. Measured 2026-09-22 with an uncommitted
    plant: ``territory_rollup_spec`` over ``[2019-01-01, 2019-01-21)`` read 0/0/0/0/0 and
    permitted while the run would have selected 2019-01-30 and 2019-03-15.

    So this census does not restate the selection -- it embeds the CTE the caller hands
    it, which is the ETL's own constant, so the census cannot drift from the run. The
    guard itself still imports nothing from the ETL (the module's standing rule); the
    caller passes the constant, exactly as the late-arrival fixture derives its window
    from the ETL's ``ARRIVAL_WINDOW_HOURS``. ``start``/``end`` are the RUN's window
    (``arrived_before`` less ``ARRIVAL_WINDOW_HOURS``, and ``arrived_before``), not a
    date range, and the selection depends on the rows the file has already planted, so
    the census must run right before the territory run, not at fixture time.

    Legs:

    1. **Derivation** counts what the aggregate READS for the selected dates: the
       per-HCP rows on them (``total_trx`` / ``total_nrx``) plus the triggers inside
       each date's lookback (``active_hcp_count``). Planted is the same two counts under
       the file's prefixes -- a subset by construction -- so one foreign row on either
       side refuses. A date selected by a foreign trigger or a foreign per-HCP row is
       always caught here, because whatever selected it is also read for it.
    2. **Key space** is waived, as for a window-sweeping teardown: the file deletes
       ``territory_metrics`` for every selected date (recorded via
       :func:`selected_metric_dates_sql`), so the cross join's foreign rows do not
       survive. The run's reads of ``hcp_profiles`` (the ``territories`` CTE, and
       ``covered_lives`` from every real territory's ``total_patient_volume``) are
       this hazard, not leg 1's: they produce a row per real territory whose values are
       the profile table's, and that row is what the per-date teardown removes. This is
       the same pairing, and the same measured decision, as
       ``territory_rollup_spec(teardown_deletes_window=True)`` (codex r1-1 on #2213
       asked for leg 1 to count them; the guard's rule is "foreign rows must not
       SURVIVE", and scoping the ETL itself to planted territories would change
       production SQL).
    3. **Teardown reach** counts the foreign ``territory_metrics`` rows already on the
       selected dates -- the rows that per-date DELETE would destroy.

    The census, the recording of the selected set and the run are separate
    transactions (codex r1-2). What closes the gap between them is the caller's
    protocol, not this spec: pin the recorded set BEFORE recording it (a foreign date
    cannot be recorded and swept), and re-read the selection AFTER the run (a date that
    appeared in between is reported by a failing assertion, not swept). A writer landing
    2019-dated rows inside that gap is the file's standing premise, measured: the
    earliest live ``triggers.created_at`` is 2026-06-10.
    """
    selected = _selected_metric_dates_subquery(metric_dates_cte)
    in_lookback = (
        f"EXISTS (SELECT 1 FROM {selected} sd "
        f" WHERE t.trigger_timestamp >= sd.metric_date"
        f" - INTERVAL '{TERRITORY_ACTIVE_HCP_LOOKBACK_DAYS} days' "
        "   AND t.trigger_timestamp <  sd.metric_date + INTERVAL '1 day')"
    )
    run_params = {"start_date": start, "end_date": end, "per_hcp_metric_type": PER_HCP_METRIC_TYPE}
    return WriteWindowSpec(
        test_file=test_file,
        window_description=(
            f"territory rollup by ARRIVAL, run window [{start}, {end}): per-HCP dates written "
            f"in it or within {TERRITORY_ACTIVE_HCP_LOOKBACK_DAYS} days of a trigger arriving in it"
        ),
        queries=(
            CensusQuery(
                "source_rows_total",
                "per-HCP rows on the selected dates + triggers in their active-HCP lookback",
                "SELECT (SELECT count(*) FROM business_metrics bm "
                "         WHERE bm.metric_type = %(per_hcp_metric_type)s AND bm.hcp_id IS NOT NULL "
                f"          AND bm.metric_date IN {selected}) "
                "     + (SELECT count(*) FROM triggers t "
                f"         WHERE t.hcp_id IS NOT NULL AND {in_lookback})",
                run_params,
            ),
            CensusQuery(
                "source_rows_planted",
                "of those, per-HCP rows on hcp_ids this run owns + triggers it planted",
                "SELECT (SELECT count(*) FROM business_metrics bm "
                "         WHERE bm.metric_type = %(per_hcp_metric_type)s AND bm.hcp_id IS NOT NULL "
                f"          AND bm.metric_date IN {selected} AND bm.hcp_id LIKE %(hcp_like)s) "
                "     + (SELECT count(*) FROM triggers t "
                f"         WHERE t.hcp_id IS NOT NULL AND {in_lookback} "
                "           AND t.trigger_id LIKE %(trigger_like)s)",
                {**run_params, "hcp_like": hcp_like, "trigger_like": trigger_like},
            ),
            CensusQuery(
                "target_entities_total",
                "territories the CROSS JOIN writes a row for (waived: the teardown deletes"
                " every selected date, which removes them again)",
                "SELECT count(DISTINCT territory_id) FROM hcp_profiles WHERE false",
                {},
            ),
            CensusQuery(
                "target_entities_planted",
                "of those, territories this run owns (waived with the leg above)",
                "SELECT count(DISTINCT territory_id) FROM hcp_profiles WHERE false",
                {},
            ),
            CensusQuery(
                "teardown_reach_preexisting",
                "territory_metrics rows already on a selected date that are not ours",
                "SELECT count(*) FROM territory_metrics m "
                f" WHERE m.metric_date IN {selected} "
                "   AND m.territory_id NOT LIKE %(territory_like)s",
                {**run_params, "territory_like": territory_like},
            ),
        ),
    )


def adherence_spec(
    *,
    test_file: str,
    start: Any,
    end: Any,
    journey_like: str,
    window_column: str = "journey_start_date",
) -> WriteWindowSpec:
    """Census for a file that runs ``patient_adherence_etl`` on explicit dates.

    The explicit variant selects on ``journey_start_date`` and UPDATEs every selected
    journey whose value would change, so an unplanted journey in the window is both a
    derivation source and a write target -- the same row on both legs, which is why
    they are counted separately rather than assumed to coincide.

    After 22C the ETL writes only ``adherence_rate``, skips rows whose value would not
    change, and reads no ``triggers``. That narrows the blast radius; it does not make
    it zero, so the census is unchanged in shape.
    """
    if window_column not in ("journey_start_date", "created_at"):
        raise ValueError(f"unknown adherence window column {window_column!r}")
    # journey_start_date is a DATE and created_at a timestamptz, so the cast differs;
    # casting a timestamptz bound to ::DATE would silently widen the window by a day.
    cast = "::DATE" if window_column == "journey_start_date" else ""
    window = {"start": start, "end": end}
    sel = f" WHERE pj.{window_column} >= %(start)s{cast}    AND pj.{window_column} <  %(end)s{cast}"
    return WriteWindowSpec(
        test_file=test_file,
        window_description=f"adherence rollup, {window_column} in [{start}, {end})",
        queries=(
            CensusQuery(
                "source_rows_total",
                "journeys selected by the window",
                f"SELECT count(*) FROM patient_journeys pj{sel}",
                window,
            ),
            CensusQuery(
                "source_rows_planted",
                "of those, journeys this run planted",
                f"SELECT count(*) FROM patient_journeys pj{sel} "
                f"AND pj.patient_journey_id LIKE %(journey_like)s",
                {**window, "journey_like": journey_like},
            ),
            CensusQuery(
                "target_entities_total",
                "journeys the UPDATE could touch (same row set; the write is in place)",
                f"SELECT count(DISTINCT pj.patient_journey_id) FROM patient_journeys pj{sel}",
                window,
            ),
            CensusQuery(
                "target_entities_planted",
                "of those, journeys this run owns",
                f"SELECT count(DISTINCT pj.patient_journey_id) FROM patient_journeys pj{sel} "
                f"AND pj.patient_journey_id LIKE %(journey_like)s",
                {**window, "journey_like": journey_like},
            ),
            CensusQuery(
                "teardown_reach_preexisting",
                "this file deletes by planted prefix only, so the leg is empty by construction",
                "SELECT count(*) FROM patient_journeys WHERE false",
                {},
            ),
        ),
    )


def require_isolated_windows(conn: Any, *specs: WriteWindowSpec) -> tuple[WindowCensus, ...]:
    """Census every spec and FAIL on the first refusal. One call per test file."""
    return tuple(_require_isolated_window(conn, spec) for spec in specs)


def _require_isolated_window(conn: Any, spec: WriteWindowSpec) -> WindowCensus:
    """Census the live DB and FAIL -- never skip -- when the window is not isolated.

    Fail rather than skip, deliberately: all five files already skip without the
    opt-in, so another skip would change nothing. The hazard only materialises when
    somebody has opted in, and at that moment the useful outcome is a red test
    naming the file, the window and the counts.

    Read-only by construction: the census runs inside a READ ONLY transaction, so a
    mistake in a spec's SQL cannot itself write.
    """
    import pytest

    counts: list[int] = []
    with conn.cursor() as cur:
        cur.execute("BEGIN TRANSACTION READ ONLY")
        try:
            for q in spec.queries:
                cur.execute(q.sql, dict(q.params))
                row = cur.fetchone()
                counts.append(int(row[0]) if row and row[0] is not None else 0)
        finally:
            cur.execute("ROLLBACK")

    census = WindowCensus(*counts)  # type: ignore[arg-type]
    verdict = assess(census)
    if verdict.refused:
        detail = "\n".join(f"  - {r}" for r in verdict.reasons)
        pytest.fail(
            f"REFUSED: {spec.test_file} would write to the live database from data it "
            f"does not own.\n"
            f"  window: {spec.window_description}\n"
            f"{detail}\n"
            f"  This refusal is derived from the data, not from E2I_DB_INTEGRATION. "
            f"Re-point the test's window at a range it plants entirely, or plant the "
            f"rows it derives from. See tests/integration/_prod_write_guard.py.",
            pytrace=False,
        )
    return census
