"""#1640: an event-ledger count and a business_metrics level are not the same TRx.

The 2026-08-15 eval reported Kisqali TRx at scales that cannot be reconciled,
and two of them come from genuinely different substrates:

    11,298      kpi_calculate_tool  -> COUNT(*) over treatment_events
                                       where event_type='prescription', 30d
    207,270.27  e2i_data_query_tool -> business_metrics.value, northeast, 2026-08

Measured against the live DB (2026-08-15): the national business_metrics TRx
total for 2026-08 is 825,242 against 11,298 trailing-30-day prescription events
— **73.0x**, and stable month over month (2026-07: 830,103; 2026-06: 812,266).
That is not a window or grain artifact. ``business_metrics.value`` is a MODELED
market-scale level: BusinessMetricsGenerator draws Kisqali from a base of
50,000 per region per month with REGION_FACTORS northeast 1.15 and 2% monthly
trend, so national 50,000 x 4.00 x 4.26 = 852,000 against a measured 825,242.
An event count cannot be fractional, and these values are (min 5,923.95).

So the scales cannot be reconciled by arithmetic, and the issue's own
acceptance criterion resolves to its second branch: cross-substrate figures
must not be presentable as comparable.

The fence is DERIVED, never hardcoded. ``KPIMetadata.tables`` already declares
each KPI's substrate (measured: all 45 registry KPIs populate it), so the rule
is "two figures are comparable only if their substrates match". That derivation
gets the one genuine exception right for free: WS3-BI-010 (ROI) really does
read ``business_metrics``, so it IS comparable with the data-query tool's ROI —
where a hardcoded "KPI tool = events, data tool = business_metrics" map would
wrongly fence it off.
"""

import pytest

pytestmark = pytest.mark.unit


class TestTheRegistryDeclaresEverySubstrate:
    """The fence is only derivable if the declaration is complete."""

    def test_every_kpi_declares_its_tables(self):
        from src.kpi.registry import get_registry

        undeclared = [k.id for k in get_registry().get_all() if not k.tables]
        assert not undeclared, f"KPIs with no substrate declared: {undeclared}"

    def test_trx_reads_the_event_ledger_not_business_metrics(self):
        from src.services.kpi_resolution import recognize_kpi

        trx = recognize_kpi("TRx")
        assert trx is not None and trx.id == "WS3-BI-005"
        assert "treatment_events" in trx.tables
        assert "business_metrics" not in trx.tables

    def test_roi_genuinely_reads_business_metrics(self):
        """The exception that makes a derived rule better than a map."""
        from src.kpi.registry import get_registry

        roi = get_registry().get("WS3-BI-010")
        assert roi is not None
        assert "business_metrics" in roi.tables


class TestBothToolsDeclareTheirBasis:
    def test_kpi_tool_basis_is_derived_from_the_registry(self):
        from src.api.routes.chatbot_tools import _measure_basis_for_kpi
        from src.services.kpi_resolution import recognize_kpi

        basis = _measure_basis_for_kpi(recognize_kpi("TRx"))
        assert basis["substrate"] == ["treatment_events"]
        assert basis["computed"] is True

    def test_roi_basis_reports_business_metrics(self):
        from src.api.routes.chatbot_tools import _measure_basis_for_kpi
        from src.kpi.registry import get_registry

        basis = _measure_basis_for_kpi(get_registry().get("WS3-BI-010"))
        assert "business_metrics" in basis["substrate"]

    def test_stored_rows_declare_business_metrics(self):
        from src.api.routes.chatbot_tools import _BUSINESS_METRICS_BASIS

        assert _BUSINESS_METRICS_BASIS["substrate"] == ["business_metrics"]
        assert _BUSINESS_METRICS_BASIS["computed"] is False

    def test_the_two_bases_are_not_comparable(self):
        from src.api.routes.chatbot_tools import (
            _BUSINESS_METRICS_BASIS,
            _measure_basis_for_kpi,
            bases_are_comparable,
        )
        from src.services.kpi_resolution import recognize_kpi

        assert not bases_are_comparable(
            _measure_basis_for_kpi(recognize_kpi("TRx")), _BUSINESS_METRICS_BASIS
        )

    def test_roi_is_not_certified_comparable_either(self):
        """ROI declares ['agent_activities', 'business_metrics'] — a UNION of
        possible sources, not the one a given call used: the calculator can fall
        back to agent_activities. An earlier version of this fence used set
        INTERSECTION and certified ROI comparable with stored business_metrics
        rows on that overlap. That was over-confident — the declaration cannot
        tell us which leg actually ran — so comparability now requires the
        substrate sets to be EQUAL, and ROI fails closed."""
        from src.api.routes.chatbot_tools import (
            _BUSINESS_METRICS_BASIS,
            _measure_basis_for_kpi,
            bases_are_comparable,
        )
        from src.kpi.registry import get_registry

        assert not bases_are_comparable(
            _measure_basis_for_kpi(get_registry().get("WS3-BI-010")), _BUSINESS_METRICS_BASIS
        )

    def test_a_shared_leg_is_not_comparability(self):
        """codex: conversion_rate declares ['triggers', 'treatment_events'] and
        TRx declares ['treatment_events']. A ratio and a count are not
        comparable just because one SQL leg overlaps."""
        from src.api.routes.chatbot_tools import _measure_basis_for_kpi, bases_are_comparable
        from src.services.kpi_resolution import recognize_kpi

        conv = _measure_basis_for_kpi(recognize_kpi("conversion rate"))
        trx = _measure_basis_for_kpi(recognize_kpi("TRx"))
        assert set(conv["substrate"]) & set(trx["substrate"]), "precondition: they DO overlap"
        assert not bases_are_comparable(conv, trx)

    def test_a_basis_is_comparable_with_itself(self):
        from src.api.routes.chatbot_tools import _measure_basis_for_kpi, bases_are_comparable
        from src.services.kpi_resolution import recognize_kpi

        trx = _measure_basis_for_kpi(recognize_kpi("TRx"))
        assert bases_are_comparable(trx, trx)

    def test_the_note_names_the_measured_ratio(self):
        """A bare label is a labeling fix. The reader needs the magnitude."""
        from src.api.routes.chatbot_tools import _BUSINESS_METRICS_BASIS

        note = _BUSINESS_METRICS_BASIS["note"]
        assert "73" in note, note
        assert "treatment_events" in note


class TestBothChatBrainsCarryTheScaleGuard:
    """This repo has TWO answering surfaces, and fixing one is the mistake the
    #1638 round already paid for: AG-UI (``E2I_COPILOT_SYSTEM_PROMPT``) and
    ``/chat/stream`` (``chatbot_graph.E2I_CHATBOT_SYSTEM_PROMPT``)."""

    def _prompts(self):
        from src.api.routes.chatbot_graph import E2I_CHATBOT_SYSTEM_PROMPT
        from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT

        return {
            "copilot": E2I_COPILOT_SYSTEM_PROMPT,
            "chatbot": E2I_CHATBOT_SYSTEM_PROMPT,
        }

    def test_every_surface_states_the_rule(self):
        missing = [n for n, p in self._prompts().items() if "measure_basis" not in p]
        assert not missing, f"surfaces with no scale guard: {missing}"

    def test_every_surface_names_the_comparability_rule(self):
        for name, prompt in self._prompts().items():
            assert "SCALE GUARD" in prompt, name
            assert "73x" in prompt or "73×" in prompt, name
            assert "treatment_events" in prompt and "business_metrics" in prompt, name

    def test_the_rule_fails_closed_on_a_missing_basis(self):
        """An undeclared basis is not evidence of agreement."""
        for name, prompt in self._prompts().items():
            assert "NOT comparable" in prompt, name


class TestTheBasisIsNotSelfReferential:
    """The payload fixture in test_chatbot_kpi_tool.py builds its expectation by
    CALLING ``_measure_basis_for_kpi``, which would pass even if the helper
    returned nonsense. This pins the actual content independently."""

    def test_nbrx_basis_is_the_event_ledger(self):
        from src.api.routes.chatbot_tools import _measure_basis_for_kpi
        from src.kpi.registry import get_registry

        basis = _measure_basis_for_kpi(get_registry().get("WS3-BI-007"))
        assert basis["substrate"] == ["treatment_events"]
        assert basis["computed"] is True

    def test_an_undeclared_kpi_fails_closed(self):
        """A KPI with no declared substrate must not read as comparable with
        anything — an undeclared basis is not evidence of agreement."""
        from src.api.routes.chatbot_tools import (
            _BUSINESS_METRICS_BASIS,
            _measure_basis_for_kpi,
            bases_are_comparable,
        )

        class _Undeclared:
            tables: list = []

        basis = _measure_basis_for_kpi(_Undeclared())
        assert basis["substrate"] == []
        assert not bases_are_comparable(basis, _BUSINESS_METRICS_BASIS)
        assert not bases_are_comparable(basis, basis)


class TestTheFenceIsFunctionalNotJustLabelled:
    """codex HIGH, and the finding I explicitly invited: the first version added
    a `measure_basis` label, a prompt rule, and a `bases_are_comparable` helper
    that NOTHING CALLED outside its own tests. A helper with no caller does not
    fence anything — it makes the fix look functional while leaving compliance
    entirely to the model.

    The seam that needs no cross-tool state: when ``e2i_data_query_tool`` is
    asked for a ``kpi_name`` the registry can COMPUTE from a different
    substrate, that single call already knows both bases. It says so in the
    payload, in code, at that moment.
    """

    def test_asking_for_trx_declares_the_conflict(self):
        from src.api.routes.chatbot_tools import _cross_substrate_conflict

        conflict = _cross_substrate_conflict("TRx")
        assert conflict, "TRx is computable from a different substrate"
        assert conflict["other_tool"] == "kpi_calculate_tool"
        assert conflict["other_substrate"] == ["treatment_events"]
        assert conflict["this_substrate"] == ["business_metrics"]
        assert "not comparable" in conflict["note"].lower()

    def test_the_metric_name_is_normalized_like_the_query_is(self):
        """``trx`` is what reaches the column filter; the notice must fire on
        the same spellings the tool actually serves."""
        from src.api.routes.chatbot_tools import _cross_substrate_conflict

        for spelling in ("TRx", "trx", "total prescriptions", "NRx"):
            assert _cross_substrate_conflict(spelling), spelling

    def test_an_unrecognized_metric_declares_nothing(self):
        from src.api.routes.chatbot_tools import _cross_substrate_conflict

        assert _cross_substrate_conflict("not a metric at all") is None
        assert _cross_substrate_conflict(None) is None

    def test_the_conflict_is_decided_by_the_helper_not_a_literal(self):
        """If the comparability rule ever says these DO agree, the notice must
        vanish — it must be a consequence of the rule, not a hardcoded string."""
        import src.api.routes.chatbot_tools as ct
        import src.kpi.measure_basis as mb

        # Patch where the name is RESOLVED (the service), not where it is
        # re-exported — the re-export is a binding, not an indirection.
        original = mb.substrates_agree
        try:
            mb.substrates_agree = lambda a, b: True
            assert ct._cross_substrate_conflict("TRx") is None
        finally:
            mb.substrates_agree = original

    def test_the_zero_row_path_carries_the_basis_but_not_a_notice(self):
        """This asserted the notice fired here too, on the reasoning that an
        empty result must not read as agreement. codex iter-3 showed that was
        backwards: with no rows there is no stored figure to confuse, and the
        notice could name a KPI the query never actually filtered for (see
        ``TestTheNoticeDescribesTheRowsItSitsBeside``).

        The BASIS still rides along — the substrate is a property of the query,
        not of whether it matched."""
        import asyncio
        import datetime

        from src.api.routes.chatbot_tools import _query_kpis

        out = asyncio.run(
            _query_kpis(
                brand="NotARealBrand",
                region=None,
                kpi_name="TRx",
                since=datetime.datetime(2026, 1, 1),
                limit=5,
            )
        )
        assert out["count"] == 0
        assert out["measure_basis"]["substrate"] == ["business_metrics"]
        assert out["cross_substrate_conflict"] is None


class TestTheBasisSsotIsCheapToImport:
    """codex HIGH: other surfaces emit KPI figures with no basis — the
    orchestrator's ``kpi_lookup`` payload and the Home KPI summary. They cannot
    import ``chatbot_tools`` to get it: that module costs ~30s to import
    (orchestrator/tool_composer/RAG stacks), which is exactly why #1475 moved
    ``KPI_SEMANTIC_NOTES`` to ``src.services.kpi_resolution``. Same precedent.
    """

    def test_the_ssot_module_carries_the_rule(self):
        from src.kpi.measure_basis import (
            BUSINESS_METRICS_BASIS,
            bases_are_comparable,
            cross_substrate_conflict,
            measure_basis_for_kpi,
        )

        assert BUSINESS_METRICS_BASIS["substrate"] == ["business_metrics"]
        assert callable(bases_are_comparable)
        assert callable(cross_substrate_conflict)
        assert callable(measure_basis_for_kpi)

    def test_importing_it_does_not_drag_in_the_heavy_stacks(self):
        """A basis lookup must not cost what importing chatbot_tools costs.

        The first version of this test listed three module names I had picked
        and asserted their absence — which passed while the module was under
        ``src.services``, whose ``__init__`` eagerly imports ``alert_routing``
        and pulls in ``aiohttp`` (measured: 0.54s / 394 modules vs 0.43s / 344).
        A test that checks the names you thought of is not a cheapness test.

        ``src.services`` itself is now the assertion: any transitive import of
        that package means the whole service layer came along.
        """
        import subprocess
        import sys

        code = (
            "import sys; import src.kpi.measure_basis as m; "
            "print([n for n in sorted(sys.modules) if n in "
            "('aiohttp','dspy','langgraph','src.services','src.api.routes.chatbot_tools') "
            "or n.startswith('src.services.')])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip() == "[]", out.stdout

    def test_chatbot_tools_reexports_the_same_objects(self):
        import src.api.routes.chatbot_tools as ct
        import src.kpi.measure_basis as mb

        assert ct.bases_are_comparable is mb.bases_are_comparable
        assert ct._BUSINESS_METRICS_BASIS is mb.BUSINESS_METRICS_BASIS


class TestEveryKpiEmitterDeclaresItsBasis:
    """Asserted against the dict LITERAL, not the source text.

    A grep for "measure_basis" passes if the payload key is deleted and only the
    comment above it survives — which is exactly the shape of an accidental
    revert. Parsing the AST and looking for the key inside a dict literal cannot
    be satisfied by a comment.
    """

    @staticmethod
    def _dict_literals_with_key(module, key):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(module))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            and any(
                isinstance(k, ast.Constant) and k.value == key for k in node.keys if k is not None
            )
        ]

    def test_orchestrator_kpi_lookup_payload(self):
        from src.agents.orchestrator.nodes import dispatcher

        assert self._dict_literals_with_key(dispatcher, "measure_basis"), (
            "no dict literal in the dispatcher carries a measure_basis key"
        )

    def test_home_kpi_summary_payload(self):
        from src.api.routes import copilotkit

        assert self._dict_literals_with_key(copilotkit, "measure_basis"), (
            "no dict literal in copilotkit carries a measure_basis key"
        )

    def test_a_comment_alone_would_not_satisfy_this(self):
        """Guard the guard: prove the assertion is structural, not textual."""
        import ast

        tree = ast.parse('x = 1  # measure_basis\ny = {"other": 1}\n')
        dicts = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Dict)
            and any(isinstance(k, ast.Constant) and k.value == "measure_basis" for k in n.keys)
        ]
        assert not dicts


class TestTheBasisPrefersTheSourceThatActuallyAnswered:
    """codex MED: the registry's ``tables`` is a UNION of POSSIBLE sources.

    ROI declares ``['agent_activities', 'business_metrics']`` because its
    calculator tries `business_impact_roi_business_metrics_scoped` first and
    only falls back to `agent_activities` when unscoped and empty
    (business_impact.py). The calculator's own comment already says "provenance
    reflects whichever source actually answered" — it just never recorded which.

    So a scoped ROI that genuinely came from business_metrics WAS being fenced
    off from stored business_metrics rows. Over-fencing is the mirror of
    over-claiming, and both are wrong for the same reason: the payload asserting
    something it does not know.
    """

    def test_runtime_metadata_overrides_the_static_union(self):
        from src.kpi.measure_basis import measure_basis_for_kpi
        from src.kpi.registry import get_registry

        roi = get_registry().get("WS3-BI-010")
        static = measure_basis_for_kpi(roi)
        assert set(static["substrate"]) == {"agent_activities", "business_metrics"}

        actual = measure_basis_for_kpi(
            roi, {"context": {"measure_basis_substrate": ["business_metrics"]}}
        )
        assert actual["substrate"] == ["business_metrics"]
        assert actual["runtime_confirmed"] is True

    def test_the_static_union_is_marked_as_unconfirmed(self):
        from src.kpi.measure_basis import measure_basis_for_kpi
        from src.kpi.registry import get_registry

        static = measure_basis_for_kpi(get_registry().get("WS3-BI-010"))
        assert static["runtime_confirmed"] is False

    def test_a_single_source_kpi_needs_no_runtime_record(self):
        """TRx declares one table, so the declared set is already exact —
        ``runtime_confirmed`` is False because no calculator recorded anything,
        NOT because the substrate is in doubt. The earlier ``source_known`` name
        conflated those two and reported False for the 11 registry KPIs whose
        two tables are JOINED in one query, where the union is exact."""
        from src.kpi.measure_basis import measure_basis_for_kpi
        from src.services.kpi_resolution import recognize_kpi

        basis = measure_basis_for_kpi(recognize_kpi("TRx"))
        assert basis["substrate"] == ["treatment_events"]
        assert basis["declared_sources"] == ["treatment_events"]
        assert basis["runtime_confirmed"] is False

    def test_a_scoped_roi_becomes_comparable_with_stored_rows(self):
        from src.kpi.measure_basis import (
            BUSINESS_METRICS_BASIS,
            bases_are_comparable,
            measure_basis_for_kpi,
        )
        from src.kpi.registry import get_registry

        roi = get_registry().get("WS3-BI-010")
        assert not bases_are_comparable(measure_basis_for_kpi(roi), BUSINESS_METRICS_BASIS)
        assert bases_are_comparable(
            measure_basis_for_kpi(
                roi, {"context": {"measure_basis_substrate": ["business_metrics"]}}
            ),
            BUSINESS_METRICS_BASIS,
        )

    def test_the_calculator_records_which_branch_answered(self):
        import ast
        import inspect

        from src.kpi.calculators import business_impact

        tree = ast.parse(inspect.getsource(business_impact))
        assigns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant)
            and n.slice.value == "measure_basis_substrate"
        ]
        assert len(assigns) >= 2, "both ROI branches must record their own source"

    def test_a_cached_pre_change_result_fails_closed(self):
        """Observed live: a KPIResult cached BEFORE this change carries no
        ``measure_basis_substrate``, so the basis falls back to the declared
        union and `source_known` is False.

        That is the safe direction — an unconfirmed multi-source basis is not
        comparable with anything — and it self-heals as the cache expires. Pinned
        so nobody "fixes" it by assuming the first declared table."""
        from src.kpi.measure_basis import (
            BUSINESS_METRICS_BASIS,
            bases_are_comparable,
            measure_basis_for_kpi,
        )
        from src.kpi.registry import get_registry

        roi = get_registry().get("WS3-BI-010")
        stale = measure_basis_for_kpi(roi, {"context": {"data_through": "2026-08-01"}})
        assert stale["runtime_confirmed"] is False
        assert stale["substrate"] == ["agent_activities", "business_metrics"]
        assert not bases_are_comparable(stale, BUSINESS_METRICS_BASIS)


class TestTheHelperAnswersSubstrateNotMeasure:
    """Measured limit, pinned so it is not mistaken for a bug — or for a
    guarantee the helper does not make.

    Trigger Recall (WS2-TR-002) and Conversion Rate (WS3-BI-009) both declare
    ``['treatment_events', 'triggers']``. ``substrates_agree`` returns True for
    them, correctly: they DO share a substrate. But a recall and a conversion
    rate are not interchangeable figures, so this is not a general "may I
    compare these numbers" oracle, and the one production caller only ever asks
    it about ONE metric surfaced two ways.
    """

    def test_two_different_metrics_can_share_a_substrate(self):
        from src.kpi.measure_basis import measure_basis_for_kpi, substrates_agree
        from src.kpi.registry import get_registry

        reg = get_registry()
        recall = measure_basis_for_kpi(reg.get("WS2-TR-002"))
        conversion = measure_basis_for_kpi(reg.get("WS3-BI-009"))
        assert recall["substrate"] == conversion["substrate"]
        assert substrates_agree(recall, conversion)

    def test_the_production_caller_only_asks_about_one_metric(self):
        """`cross_substrate_conflict` resolves ONE kpi_name and compares its
        computed basis against the stored-row basis — never two metrics."""
        import ast
        import inspect

        from src.kpi import measure_basis

        fn = next(
            n
            for n in ast.walk(ast.parse(inspect.getsource(measure_basis)))
            if isinstance(n, ast.FunctionDef) and n.name == "cross_substrate_conflict"
        )
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "measure_basis_for_kpi"
        ]
        assert len(calls) == 1, "more than one KPI resolved — the helper cannot judge that"

    def test_the_old_name_still_resolves(self):
        from src.kpi.measure_basis import bases_are_comparable, substrates_agree

        assert bases_are_comparable is substrates_agree


class TestTheNoticeDescribesTheRowsItSitsBeside:
    """codex iter-3 HIGH: the notice and the query disagreed on what was asked.

    ``_query_kpis`` filters ``business_metrics.metric_name`` with
    ``_normalize_metric_name(kpi_name)``, while the notice resolved the RAW name
    through ``recognize_kpi``. Measured, they diverge:

        "total prescriptions" -> filter key 'total_prescriptions'  (never stored)
                              -> notice claimed WS3-BI-005 TRx
        "hcp coverage"        -> filter key 'hcp_coverage'         (never stored)
                              -> notice claimed WS3-BI-004

    So a TRx cross-substrate warning could be attached to zero rows for a key
    that was never queried as TRx. The notice is a caveat ON the rows above it,
    so it now fires only when there ARE rows: with none, there is no stored
    figure to be confused with anything, and naming one is worse than silence.
    """

    def test_no_rows_means_no_notice(self):
        import asyncio
        import datetime

        from src.api.routes.chatbot_tools import _query_kpis

        out = asyncio.run(
            _query_kpis(
                brand="NotARealBrand",
                region=None,
                kpi_name="TRx",
                since=datetime.datetime(2026, 1, 1),
                limit=5,
            )
        )
        assert out["count"] == 0
        assert out["cross_substrate_conflict"] is None
        # The BASIS still rides along — the rows' substrate is a property of the
        # query, not of whether it matched anything.
        assert out["measure_basis"]["substrate"] == ["business_metrics"]

    def test_the_helper_still_answers_for_a_real_metric(self):
        from src.api.routes.chatbot_tools import _cross_substrate_conflict

        assert _cross_substrate_conflict("TRx")["kpi_id"] == "WS3-BI-005"

    def test_a_name_that_is_not_a_stored_key_cannot_produce_rows(self):
        """Why gating on rows is sufficient rather than a second name check:
        if the filter key is not a stored metric_name, the query returns
        nothing, so the mismatch can never reach a reader."""
        from src.api.routes.chatbot_tools import _normalize_metric_name

        assert _normalize_metric_name("total prescriptions") == "total_prescriptions"
        assert _normalize_metric_name("TRx") == "trx"


class TestTheSummaryDerivesSubstrateFromTheQueryItRuns:
    """codex iter-4 HIGH, and the strongest argument against my own hand map.

    I mapped the ``hcp_reach`` tile to WS3-BI-004 (HCP Coverage,
    ``['hcp_profiles']``). Measured, the tile runs::

        business_impact_hcp_reach
          SELECT COUNT(DISTINCT hcp_id) FROM treatment_events WHERE ...

    — an event-ledger count. **No registry KPI corresponds to it**: WS3-BI-004
    is a coverage FRACTION over a different table
    (``tests/integration/test_kpi_summary_realdb.py`` pins hcp_reach as a whole
    number, not that fraction). So the tile was labelled with a confident WRONG
    substrate, which is worse than none — and my previous test blessed it.

    The tiles run QUERIES, not KPIs. The substrate now comes from the registry
    SQL those queries are made of, so no correspondence has to be invented.
    """

    def test_substrates_come_from_the_registry_sql(self):
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        assert _tables_in_sql(
            "SELECT COUNT(DISTINCT hcp_id) AS hcp_reach FROM treatment_events WHERE x"
        ) == ["treatment_events"]

    def test_cte_aliases_are_not_mistaken_for_tables(self):
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        sql = (
            "WITH first_brand AS (SELECT patient_id, MIN(event_date) FROM treatment_events "
            "GROUP BY 1) SELECT COUNT(*) FROM first_brand JOIN treatment_events USING (patient_id)"
        )
        assert _tables_in_sql(sql) == ["treatment_events"]

    def test_a_join_reports_both_tables(self):
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        sql = "SELECT 1 FROM triggers t JOIN treatment_events e ON e.id = t.id"
        assert _tables_in_sql(sql) == ["treatment_events", "triggers"]

    def test_an_unreadable_registry_yields_no_basis_rather_than_a_wrong_one(self):
        from src.api.routes.copilotkit import _kpi_summary_measure_bases

        assert _kpi_summary_measure_bases(client=None) == {}

    def test_the_hand_written_kpi_id_map_is_gone(self):
        import src.api.routes.copilotkit as ck

        assert not hasattr(ck, "_KPI_SUMMARY_KPI_IDS"), (
            "the map that invented a KPI correspondence for hcp_reach is back"
        )


class TestTheRestKpiApiDeclaresItsBasis:
    """codex iter-4 HIGH: `GET /api/kpis/{id}` and `POST /api/kpis/calculate`
    return a numeric KPI value with no substrate, and the schema had no field to
    carry one. Frontend and chart code read these, so a TRx value from the REST
    API could sit beside a business_metrics figure with none of the fence
    metadata this branch added to the chat payloads."""

    def test_the_response_schema_has_the_field(self):
        from src.api.schemas.kpi import KPIResultResponse

        assert "measure_basis" in KPIResultResponse.model_fields, sorted(
            KPIResultResponse.model_fields
        )

    def test_it_is_populated_from_the_registry(self):
        from src.api.schemas.kpi import KPIResultResponse

        field = KPIResultResponse.model_fields["measure_basis"]
        assert field.default is None or field.default_factory is not None


@pytest.mark.integration
class TestTheTileSubstratesMatchTheLiveRegistry:
    """Runs against the real ``kpi_query_registry`` when one is reachable.

    This is the assertion the hand-written map could not make, and the reason it
    shipped a wrong ``hcp_reach`` label: nothing tied the declared substrate to
    the SQL the tile actually runs. Skips rather than fails without a DB — a
    unit lane with no database must not turn into a false green OR a false red.
    """

    def _bases(self):
        from src.api.dependencies.supabase_client import get_supabase, init_supabase

        if get_supabase() is None:
            init_supabase()
        if get_supabase() is None:
            pytest.skip("no Supabase client available")
        from src.api.routes.copilotkit import _kpi_summary_measure_bases

        bases = _kpi_summary_measure_bases(get_supabase())
        if not bases:
            pytest.skip("kpi_query_registry unreadable")
        return bases

    def test_hcp_reach_is_the_event_ledger_not_hcp_profiles(self):
        """The exact label that was wrong: the tile runs
        ``COUNT(DISTINCT hcp_id) FROM treatment_events``, not a coverage
        fraction over ``hcp_profiles``."""
        bases = self._bases()
        assert bases["hcp_reach"]["substrate"] == ["treatment_events"], bases["hcp_reach"]

    def test_every_tile_declares_a_substrate(self):
        bases = self._bases()
        assert set(bases) == {
            "trx_volume",
            "nrx_volume",
            "market_share",
            "conversion_rate",
            "hcp_reach",
            "patient_starts",
        }, sorted(bases)

    def test_no_tile_claims_business_metrics(self):
        for field, basis in self._bases().items():
            assert "business_metrics" not in basis["substrate"], (field, basis["substrate"])

    def test_a_tile_is_not_comparable_with_stored_rows(self):
        from src.kpi.measure_basis import BUSINESS_METRICS_BASIS, substrates_agree

        assert not substrates_agree(self._bases()["trx_volume"], BUSINESS_METRICS_BASIS)


@pytest.mark.integration
class TestTheBasisFollowsTheQueryThatActuallyRan:
    """Found by checking codex's own question before it reported: a region
    filter routes each tile to a ``*_region`` variant, and those read MORE
    tables than the base.

    Measured on the live registry::

        business_impact_hcp_reach         -> {treatment_events}
        business_impact_hcp_reach_region  -> {patient_journeys, treatment_events}
        business_impact_conversion_rate_region
                                          -> adds patient_journeys

    Deriving from the BASE id would understate the substrate under a region
    filter — the same shape as the hcp_reach defect, one level down.
    """

    def _bases(self, region):
        from src.api.dependencies.supabase_client import get_supabase, init_supabase

        if get_supabase() is None:
            init_supabase()
        if get_supabase() is None:
            pytest.skip("no Supabase client available")
        from src.api.routes.copilotkit import _kpi_summary_measure_bases

        bases = _kpi_summary_measure_bases(get_supabase(), brand="Kisqali", region=region)
        if not bases:
            pytest.skip("kpi_query_registry unreadable")
        return bases

    def test_unscoped_hcp_reach_is_the_event_ledger_alone(self):
        assert self._bases(None)["hcp_reach"]["substrate"] == ["treatment_events"]

    def test_region_scoped_hcp_reach_reports_the_variant_it_ran(self):
        assert self._bases("northeast")["hcp_reach"]["substrate"] == [
            "patient_journeys",
            "treatment_events",
        ]

    def test_the_declared_query_id_is_the_resolved_one(self):
        scoped = self._bases("northeast")["hcp_reach"]
        assert scoped["query_id"] == "business_impact_hcp_reach_region", scoped


class TestTheSqlExtractorDoesNotInventTables:
    """codex iter-5 HIGH: a PHANTOM table is a wrong basis, not a narrow one.

    Measured on the live registry: **12 of 306** queries are schema-qualified
    (``public.data_source_tracking``, ``public.v_patient_eligibility``, ...).
    None is a summary tile today, so no shipped label was wrong — but the
    helper is general, and the first schema-qualified tile query would have
    declared a substrate called ``public``.
    """

    def test_schema_qualified_names_report_the_table(self):
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        assert _tables_in_sql("SELECT 1 FROM public.treatment_events") == ["treatment_events"]
        assert _tables_in_sql("SELECT 1 FROM public.a JOIN public.b ON 1=1") == ["a", "b"]

    def test_lateral_is_not_a_table(self):
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        sql = "SELECT 1 FROM triggers t JOIN LATERAL (SELECT 1 FROM treatment_events) x ON true"
        tables = _tables_in_sql(sql)
        assert "lateral" not in tables, tables
        assert tables == ["treatment_events", "triggers"]

    def test_a_subquery_is_not_a_table(self):
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        assert _tables_in_sql("SELECT 1 FROM (SELECT 1 FROM triggers) x") == ["triggers"]

    def test_quoted_identifiers_are_read(self):
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        assert _tables_in_sql('SELECT 1 FROM "treatment_events"') == ["treatment_events"]

    def test_the_live_registry_produces_no_phantom_schema_names(self):
        """Every real query, checked against the actual table list."""
        import pytest as _pytest

        from src.api.dependencies.supabase_client import get_supabase, init_supabase

        if get_supabase() is None:
            init_supabase()
        client = get_supabase()
        if client is None:
            _pytest.skip("no Supabase client available")
        from src.kpi.measure_basis import tables_in_sql as _tables_in_sql

        try:
            rows = (client.table("kpi_query_registry").select("query_id,sql").execute()).data or []
        except Exception:
            _pytest.skip("kpi_query_registry unreadable")
        offenders = {
            r["query_id"]: t
            for r in rows
            for t in _tables_in_sql(r.get("sql") or "")
            if t in {"public", "lateral", "select"}
        }
        assert not offenders, offenders


class TestTheRegistryCacheCannotWedge:
    """codex iter-5 HIGH: an `lru_cache` for process lifetime meant a transient
    registry read failure cached ``{}`` forever, so Home would emit numeric KPI
    tiles with no basis until restart — and a registry migration could not be
    picked up either."""

    def test_a_failed_read_is_not_cached(self):
        import src.kpi.measure_basis as mb

        mb.reset_substrate_cache()
        calls = {"n": 0}

        def _boom(_ids):
            calls["n"] += 1
            return {}

        original = mb.read_query_substrates
        try:
            mb.read_query_substrates = _boom
            assert mb.query_substrates_cached(("a",)) == {}
            assert mb.query_substrates_cached(("a",)) == {}
            assert calls["n"] == 2, "an empty read was cached and never retried"
        finally:
            mb.read_query_substrates = original
            mb.reset_substrate_cache()

    def test_a_successful_read_is_cached(self):
        import src.kpi.measure_basis as mb

        mb.reset_substrate_cache()
        calls = {"n": 0}

        def _ok(_ids):
            calls["n"] += 1
            return {"a": ["treatment_events"]}

        original = mb.read_query_substrates
        try:
            mb.read_query_substrates = _ok
            assert mb.query_substrates_cached(("a",)) == {"a": ["treatment_events"]}
            assert mb.query_substrates_cached(("a",)) == {"a": ["treatment_events"]}
            assert calls["n"] == 1
        finally:
            mb.read_query_substrates = original
            mb.reset_substrate_cache()

    def test_the_cache_expires_so_a_migration_is_picked_up(self):
        import src.kpi.measure_basis as mb

        assert mb.SUBSTRATE_CACHE_TTL_SECONDS > 0
        assert mb.SUBSTRATE_CACHE_TTL_SECONDS <= 3600, "a day-long TTL is a restart in disguise"


class TestCteFormsThatWouldOtherwiseBecomePhantomTables:
    """codex iter-6 HIGH: `WITH RECURSIVE` / `AS MATERIALIZED` escape the CTE regex.

    Not reachable today — measured against the live registry 2026-08-15,
    0 of 306 rows use either form — but the migration-044 CHECK admits them
    (it only requires the text to start with `with` or `select`), and the
    failure mode is a PHANTOM substrate, which is a wrong basis rather than a
    narrow one.
    """

    @pytest.mark.parametrize(
        "sql, expected",
        [
            (
                "WITH RECURSIVE months AS (SELECT 1) "
                "SELECT * FROM months JOIN treatment_events ON true",
                ["treatment_events"],
            ),
            (
                "WITH m AS MATERIALIZED (SELECT 1) SELECT * FROM m, patient_journeys",
                ["patient_journeys"],
            ),
            (
                "WITH m AS NOT MATERIALIZED (SELECT 1) SELECT * FROM m JOIN triggers ON true",
                ["triggers"],
            ),
            (
                "WITH RECURSIVE a AS (SELECT 1), b AS MATERIALIZED (SELECT 2) "
                "SELECT * FROM a JOIN b ON true JOIN hcp_profiles ON true",
                ["hcp_profiles"],
            ),
        ],
    )
    def test_the_cte_name_is_never_reported_as_a_table(self, sql, expected):
        from src.kpi.measure_basis import tables_in_sql

        assert tables_in_sql(sql) == expected


class TestEveryChartableSeriesCarriesItsBasis:
    """codex iter-6 HIGH: the trend chart is the surface the issue is ABOUT.

    `renderKpiTrend` fetches real history through `getKPIHistory` ->
    GET /api/kpis/{id}/history (E2ICopilotProvider.tsx:1008), so a TRx trend
    can be charted beside a business_metrics TRx figure from
    e2i_data_query_tool. Point values got a basis in the first pass; the
    SERIES did not.
    """

    def test_the_materialized_history_basis_names_kpi_history_not_the_source(self):
        from src.kpi.measure_basis import materialized_history_basis
        from src.services.kpi_resolution import recognize_kpi

        kpi = recognize_kpi("TRx")
        assert kpi is not None
        basis = materialized_history_basis(kpi)
        # The series is READ from kpi_history. Saying "treatment_events" would
        # claim it was computed live, which is what the segmented endpoint
        # does and this one explicitly does not.
        assert basis["substrate"] == ["kpi_history"]
        assert basis["computed"] is False
        # NOT `sorted(kpi.tables)`, which is what this asserted first: that is
        # the registry's union of the calculator's POSSIBLE sources, and for
        # ROI it names agent_activities, which never backfilled a single row.
        # Provenance comes from the backfill tag (codex iter-7).
        assert basis["materialized_from"] == ["treatment_events.event_date"]
        assert "business_metrics" in basis["note"]

    @pytest.mark.parametrize(
        "model_name",
        ["KPIHistoryResponse", "KPISegmentedHistoryResponse", "KPINowcastHistoryResponse"],
    )
    def test_the_response_model_can_carry_a_basis(self, model_name):
        import src.api.schemas.kpi as kpi_schemas

        model = getattr(kpi_schemas, model_name)
        assert "measure_basis" in model.model_fields, (
            f"{model_name} emits numeric KPI figures with no substrate field"
        )

    def test_the_history_route_populates_it_from_the_registry(self):
        """The field existing is not the fix; the route filling it is."""
        import ast
        import inspect

        from src.api.routes import kpi as kpi_routes

        for fn_name in ("get_kpi_history", "get_kpi_history_segmented", "get_kpi_history_nowcast"):
            fn = getattr(kpi_routes, fn_name)
            tree = ast.parse(inspect.getsource(fn).lstrip())
            keywords = {
                kw.arg
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                for kw in node.keywords
            }
            assert "measure_basis" in keywords, f"{fn_name} never passes measure_basis"

    def test_a_set_returning_function_is_not_reported_as_a_table(self):
        """`FROM generate_series(...)` is a function call, not a substrate.

        The pre-existing scan added it. Latent, not live — measured
        2026-08-15, no registry query uses the form — but a phantom is a wrong
        basis, which is the thing this helper exists to avoid.
        """
        from src.kpi.measure_basis import tables_in_sql

        assert tables_in_sql("SELECT * FROM generate_series(1, 3)") == []
        assert tables_in_sql(
            "SELECT * FROM treatment_events JOIN generate_series(1,3) ON true"
        ) == ["treatment_events"]


class TestHistoryProvenanceComesFromTheRowsNotTheRegistry:
    """codex iter-7 HIGH: `materialized_from` named the wrong source for ROI.

    Deriving it from `KPIMetadata.tables` gave ROI the UNION of its calculator's
    possible sources, but the stored history is backfilled from
    `business_metrics.roi` specifically (`HANDLER_SOURCES`, and measured live
    2026-08-15: all 3,280 WS3-BI-010 rows carry `source='business_metrics.roi'`,
    all 720 WS3-BI-005 rows carry `source='treatment_events.event_date'`).

    That over-claim is not cosmetic: it would fence a history-vs-current ROI
    comparison that is business-metrics-backed on BOTH sides — the exact kind
    of false fence `measure_basis_for_kpi` was written to avoid.
    """

    def _basis(self, kpi_name, rows=None):
        from src.kpi.measure_basis import materialized_history_basis
        from src.services.kpi_resolution import recognize_kpi

        return materialized_history_basis(recognize_kpi(kpi_name), rows=rows)

    def test_roi_history_names_business_metrics_not_the_calculators_union(self):
        basis = self._basis("ROI", rows=[{"source": "business_metrics.roi", "value": 1.8}])
        assert basis["materialized_from"] == ["business_metrics.roi"], basis
        assert basis["runtime_confirmed"] is True

    def test_trx_history_names_the_event_ledger(self):
        basis = self._basis("TRx", rows=[{"source": "treatment_events.event_date", "value": 11298}])
        assert basis["materialized_from"] == ["treatment_events.event_date"]

    def test_an_empty_series_falls_back_to_the_registered_handler(self):
        """No rows is not "no provenance" — the backfill's tag is still known."""
        basis = self._basis("ROI", rows=[])
        assert basis["materialized_from"] == ["business_metrics.roi"], basis
        assert basis["runtime_confirmed"] is False

    def test_the_read_path_is_still_kpi_history(self):
        for name in ("ROI", "TRx"):
            assert self._basis(name)["substrate"] == ["kpi_history"]

    def test_the_route_passes_the_rows_it_returned(self):
        import ast
        import inspect

        from src.api.routes import kpi as kpi_routes

        tree = ast.parse(inspect.getsource(kpi_routes.get_kpi_history).lstrip())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "materialized_history_basis"
            ):
                assert any(kw.arg == "rows" for kw in node.keywords), (
                    "the route computes the basis without the rows it just read"
                )
                return
        raise AssertionError("get_kpi_history never calls materialized_history_basis")


class TestTheDispatcherUsesTheRuntimeSubstrate:
    """codex iter-8 HIGH: the orchestrator payload still read the registry.

    Its LOW is the sharper half: my existing dispatcher test only asserts the
    payload HAS a measure_basis key, so `measure_basis_for_kpi(kpi)` satisfied
    it while dropping the branch the calculator actually ran. For ROI that
    means reporting the static union ['agent_activities', 'business_metrics']
    with runtime_confirmed=false — a FALSE fence against stored ROI, which is
    business_metrics on both sides.
    """

    def test_the_call_passes_the_result_metadata(self):
        import ast
        import inspect

        from src.agents.orchestrator.nodes import dispatcher

        src = inspect.getsource(dispatcher)
        tree = ast.parse(src)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "measure_basis_for_kpi"
        ]
        assert calls, "the dispatcher no longer declares a measure basis at all"
        for call in calls:
            assert len(call.args) + len(call.keywords) >= 2, (
                "measure_basis_for_kpi(kpi) drops the runtime substrate the calculator recorded"
            )

    def test_runtime_metadata_beats_the_declared_union(self):
        from src.kpi.measure_basis import measure_basis_for_kpi
        from src.services.kpi_resolution import recognize_kpi

        roi = recognize_kpi("ROI")
        assert roi is not None
        metadata = {"context": {"measure_basis_substrate": ["business_metrics"]}}
        basis = measure_basis_for_kpi(roi, metadata)
        assert basis["substrate"] == ["business_metrics"]
        assert basis["runtime_confirmed"] is True
        # The union is still reported, as the declaration it is.
        assert "agent_activities" in basis["declared_sources"]


class TestAMixedProvenanceSeriesFailsClosed:
    """codex iter-8 HIGH: two source tags in one series is not confirmation.

    A reseed or handler change could leave ROI history with older
    `agent_activities` rows beside newer `business_metrics.roi` ones. Returning
    both with runtime_confirmed=True, under a note that says "compare on
    materialized_from", lets one mixed line read as comparable with stored ROI
    because ONE of its sources matches.

    Latent, not live: measured 2026-08-15, no kpi_id in kpi_history carries
    more than one distinct source tag (32 KPIs).
    """

    def _basis(self, rows):
        from src.kpi.measure_basis import materialized_history_basis
        from src.services.kpi_resolution import recognize_kpi

        return materialized_history_basis(recognize_kpi("ROI"), rows=rows)

    def test_a_single_source_series_is_confirmed(self):
        basis = self._basis([{"source": "business_metrics.roi"}])
        assert basis["runtime_confirmed"] is True
        assert basis.get("mixed_sources") is not True

    def test_two_sources_are_not_confirmation(self):
        basis = self._basis([{"source": "business_metrics.roi"}, {"source": "agent_activities"}])
        assert basis["mixed_sources"] is True
        assert basis["runtime_confirmed"] is False, (
            "a series spanning two substrates confirms nothing about either"
        )

    def test_the_note_says_the_line_cannot_be_compared_as_a_unit(self):
        basis = self._basis([{"source": "business_metrics.roi"}, {"source": "agent_activities"}])
        assert "not comparable" in basis["note"].lower(), basis["note"]
        # Both are still named — hiding one would be a different lie.
        assert basis["materialized_from"] == ["agent_activities", "business_metrics.roi"]


class TestComparabilityRestsOnOneKey:
    """codex iter-9 HIGH x2: `substrate` had come to mean two different things.

    Making materialized history `substrate: ["kpi_history"]` — right, because
    that IS where the series is read from — broke the stated rule in BOTH
    directions:

    * TRx history and ROI history both say `["kpi_history"]`, so a rule of
      "compare when substrate matches" calls a prescription count and a return
      ratio comparable;
    * stored ROI says `["business_metrics"]` while ROI history is materialized
      FROM `business_metrics.roi`, and the note says they are comparable — the
      substrate rule says they are not.

    `comparison_key` is the one thing a comparison may rest on: the underlying
    TABLES, whatever surface the figure came from.
    """

    def _keys(self):
        from src.kpi.measure_basis import (
            BUSINESS_METRICS_BASIS,
            materialized_history_basis,
            measure_basis_for_kpi,
        )
        from src.services.kpi_resolution import recognize_kpi

        return {
            "roi_history": materialized_history_basis(
                recognize_kpi("ROI"), rows=[{"source": "business_metrics.roi"}]
            ),
            "trx_history": materialized_history_basis(
                recognize_kpi("TRx"), rows=[{"source": "treatment_events.event_date"}]
            ),
            "stored_roi": BUSINESS_METRICS_BASIS,
            "computed_trx": measure_basis_for_kpi(recognize_kpi("TRx")),
        }

    def test_two_histories_of_different_measures_are_not_comparable(self):
        from src.kpi.measure_basis import substrates_agree

        b = self._keys()
        assert not substrates_agree(b["roi_history"], b["trx_history"]), (
            "both read from kpi_history, but one is a return ratio and one a prescription count"
        )

    def test_roi_history_and_stored_roi_are_comparable(self):
        from src.kpi.measure_basis import substrates_agree

        b = self._keys()
        assert substrates_agree(b["roi_history"], b["stored_roi"]), (
            "both rest on business_metrics; fencing them is a FALSE fence"
        )

    def test_trx_history_and_computed_trx_are_comparable(self):
        from src.kpi.measure_basis import substrates_agree

        b = self._keys()
        assert substrates_agree(b["trx_history"], b["computed_trx"])

    def test_trx_history_and_stored_business_metrics_are_not(self):
        """The original defect: ~73x apart, and the whole point of the fence."""
        from src.kpi.measure_basis import substrates_agree

        b = self._keys()
        assert not substrates_agree(b["trx_history"], b["stored_roi"])

    def test_a_mixed_series_is_comparable_with_nothing(self):
        """codex iter-9 HIGH: the flag has to reach the COMPARATOR.

        My iter-8 test only asserted the flag and the note existed, so
        `substrates_agree` happily matched a mixed basis against a clean one
        on the shared `kpi_history` substrate.
        """
        from src.kpi.measure_basis import materialized_history_basis, substrates_agree
        from src.services.kpi_resolution import recognize_kpi

        mixed = materialized_history_basis(
            recognize_kpi("ROI"),
            rows=[{"source": "business_metrics.roi"}, {"source": "agent_activities"}],
        )
        clean = materialized_history_basis(
            recognize_kpi("ROI"), rows=[{"source": "business_metrics.roi"}]
        )
        assert not substrates_agree(mixed, clean)
        assert not substrates_agree(mixed, mixed)

    def test_the_prompt_states_the_rule_that_is_actually_enforced(self):
        """A prompt that names the wrong field is a labeling defect."""
        from pathlib import Path

        for path in (
            "src/api/routes/copilotkit.py",
            "src/api/routes/chatbot_graph.py",
        ):
            text = Path(path).read_text()
            marker = "SCALE GUARD"
            assert marker in text, path
            guard = text[text.index(marker) : text.index(marker) + 1400]
            assert "comparison_key" in guard, (
                f"{path} still tells the model to compare on a field the comparator does not use"
            )


class TestSourceTagsAreParsedNotSplitOnADot:
    """codex iter-10 HIGH: `_tables_of` reintroduced the phantom class.

    Splitting a backfill tag on "." assumes every tag is `table.column`. It is
    not. Measured against the live DB 2026-08-15, the ten distinct tags in
    kpi_history include `triggers+treatment_events`,
    `patient_journeys+treatment_events.pnh`, `treatment_events+patient_journeys`
    and `weekly_capture` — so four of ten produced a comparison key naming a
    table that does not exist, which is the exact defect `tables_in_sql` spent
    several rounds eliminating.

    The consequence is a FALSE fence: conversion-rate history
    (`triggers+treatment_events`) never matched computed conversion rate
    (`['treatment_events', 'triggers']`), though they are the same measure.
    """

    @pytest.mark.parametrize(
        "tag, expected",
        [
            ("business_metrics.roi", ["business_metrics"]),
            ("treatment_events.event_date", ["treatment_events"]),
            ("triggers+treatment_events", ["treatment_events", "triggers"]),
            ("treatment_events+patient_journeys", ["patient_journeys", "treatment_events"]),
            ("patient_journeys+treatment_events.pnh", ["patient_journeys", "treatment_events"]),
            ("hcp_intent_surveys.survey_date", ["hcp_intent_surveys"]),
            # Not a table at all — a capture MECHANISM. Fails closed rather
            # than claiming a substrate named after the mechanism.
            ("weekly_capture", []),
        ],
    )
    def test_every_live_tag_resolves_to_real_tables(self, tag, expected):
        from src.kpi.measure_basis import _tables_of

        assert _tables_of([tag]) == expected

    def test_one_unrecognized_component_voids_the_whole_key(self):
        """Narrow beats wrong: a partially-understood tag confirms nothing."""
        from src.kpi.measure_basis import _tables_of

        assert _tables_of(["treatment_events+weekly_capture"]) == []

    def test_conversion_rate_history_matches_its_computed_form(self):
        """codex iter-10's fifth comparability case, and it was failing."""
        from src.kpi.measure_basis import (
            materialized_history_basis,
            measure_basis_for_kpi,
            substrates_agree,
        )
        from src.services.kpi_resolution import recognize_kpi

        kpi = recognize_kpi("Conversion Rate")
        assert kpi is not None
        history = materialized_history_basis(kpi, rows=[{"source": "triggers+treatment_events"}])
        computed = measure_basis_for_kpi(kpi)
        assert substrates_agree(history, computed), (
            f"same KPI, falsely fenced: {history['comparison_key']} vs {computed['comparison_key']}"
        )


class TestEveryBasisDeclaresTheKeyThePromptNames:
    """codex iter-10 HIGH: the Home tiles never set `comparison_key`.

    Both prompts now say `comparison_key` is the only field a comparison may
    rest on. A basis that omits it forces a consumer following that instruction
    to treat a live computed tile as incomparable with anything.
    """

    def test_the_home_tile_basis_carries_it(self):
        import ast
        import inspect

        from src.api.routes import copilotkit

        src = inspect.getsource(copilotkit._kpi_summary_measure_bases)
        tree = ast.parse(src.lstrip())
        keys = {
            key.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert "comparison_key" in keys, (
            "Home tiles declare a basis without the field the prompt names"
        )


class TestAConfirmedRoiIsNotFalselyFenced:
    """codex iter-10 HIGH: the conflict notice used the static union.

    `cross_substrate_conflict` recognized the KPI and compared the registry's
    UNION of ROI's possible sources against business_metrics — so a scoped ROI
    that actually ran on business_metrics still got a "NOT comparable" notice
    attached to the stored rows it genuinely matches.
    """

    def test_a_business_metrics_confirmed_roi_raises_no_conflict(self):
        from src.kpi.measure_basis import cross_substrate_conflict

        confirmed = {"context": {"measure_basis_substrate": ["business_metrics"]}}
        assert cross_substrate_conflict("ROI", result_metadata=confirmed) is None

    def test_an_unconfirmed_roi_still_warns(self):
        """Fail closed when the branch that answered was not recorded."""
        from src.kpi.measure_basis import cross_substrate_conflict

        assert cross_substrate_conflict("ROI") is not None

    def test_a_volume_kpi_always_warns(self):
        from src.kpi.measure_basis import cross_substrate_conflict

        conflict = cross_substrate_conflict("TRx")
        assert conflict is not None
        assert "treatment_events" in conflict["other_substrate"]

    def test_a_union_containing_business_metrics_is_stated_as_conditional(self):
        """ROI's declared substrate is ['agent_activities', 'business_metrics'].

        The stored rows ARE business_metrics, so whether a computed ROI matches
        them depends on which branch answered — something the tool cannot know
        at this point, because nothing has been computed yet. Asserting "NOT
        comparable" there overstates in the fencing direction, which is the
        same defect as understating, pointed the other way.
        """
        from src.kpi.measure_basis import cross_substrate_conflict

        conflict = cross_substrate_conflict("ROI")
        assert conflict is not None
        note = conflict["note"].lower()
        assert conflict.get("conditional") is True, conflict
        assert "depend" in note or "may" in note, conflict["note"]
        # And it must not flatly assert the two are different quantities.
        assert "~73x" not in note

    def test_a_disjoint_substrate_is_still_stated_flatly(self):
        """TRx has no business_metrics leg — nothing conditional about it."""
        from src.kpi.measure_basis import cross_substrate_conflict

        conflict = cross_substrate_conflict("TRx")
        assert conflict is not None
        assert conflict.get("conditional") is not True
        assert "73x" in conflict["note"]


class TestTheSuggestionsPromptWillNotCompareUnlabelledNumbers:
    """codex iter-10 HIGH: the Home prose path was the fifth emitter.

    Home publishes tile numbers as PROSE into `pageChatSummary`, which
    `/chat/suggestions` feeds to an LLM whose prompt asks for a trend or
    comparison pill whenever numeric KPIs are on screen. A bare
    `Total TRx (MTD): 11298` could therefore seed a comparison against a
    business_metrics figure ~73x larger.

    Prose has nowhere to attach a structured basis, so the substrate is printed
    inline and the prompt is told what the marks mean.
    """

    def test_the_prompt_forbids_cross_source_comparison_pills(self):
        from pathlib import Path

        # Normalized the way the MODEL receives it: the source uses backslash
        # line-continuations, so raw-text asserts break across them and prove
        # nothing about the assembled prompt. (My first version of this did
        # exactly that and failed on its own wrapping.)
        raw = Path("src/api/routes/chat.py").read_text()
        prompt = " ".join(raw.replace("\\\n", " ").split()).lower()
        assert "source unstated" in prompt, "the prompt never explains the marks"
        assert "never propose comparing" in prompt
        # A single-figure trend pill stays allowed — this must not become a
        # blanket ban on the feature.
        assert "trend pill for one figure is always safe" in prompt

    def test_the_marks_the_prompt_relies_on_are_the_ones_home_emits(self):
        """A prompt that describes a mark the page never prints is a lie.

        Both halves are checked together because they are one contract across
        two languages, and nothing else would catch them drifting apart.
        """
        from pathlib import Path

        prompt = Path("src/api/routes/chat.py").read_text()
        home = Path("frontend/src/pages/Home.tsx").read_text()
        for mark in ("[from ", "source unstated"):
            assert mark in home, f"Home never emits {mark!r}"
            assert mark.strip("[") in prompt, f"the prompt never mentions {mark!r}"
