"""The tool-composer observability aggregates (spec §8).

The page and the planning prompt must describe the SAME population: if the tool rows exclude
synthetic-substrate runs, the composition counts beside them cannot include those runs, and a
percentile shown here cannot be computed differently from the one the database computes for
tools. These tests pin the parts a real database would otherwise only reveal later.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.services.tool_composer_observability_service import (
    ToolComposerObservabilityService,
    _percentile,
)


class FakeQuery:
    """A query builder that actually filters, orders and pages.

    A double that ignored its own filters would let a pagination bug pass: the tests below
    depend on it returning what the query asked for, not everything it holds.
    """

    def __init__(self, client: "FakeClient", table: str) -> None:
        self.client = client
        self.table = table
        self.filters: List[tuple] = []
        self.orders: List[tuple] = []
        self.columns: Optional[set] = None
        # Not ``self.range``: that would shadow the range() method this double must expose.
        self.window: Optional[tuple] = None

    def select(self, *columns: str) -> "FakeQuery":
        # Honour the projection: a double that returned columns the query never asked for would
        # hide a missing one, which is exactly how the created_at ordering defect escaped.
        self.columns = {c.strip() for spec in columns for c in spec.split(",") if c.strip()}
        return self

    def gte(self, column: str, value: Any) -> "FakeQuery":
        self.filters.append(("gte", column, value))
        return self

    def gt(self, column: str, value: Any) -> "FakeQuery":
        self.filters.append(("gt", column, value))
        return self

    def eq(self, column: str, value: Any) -> "FakeQuery":
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column: str, values: List[Any]) -> "FakeQuery":
        self.filters.append(("in", column, list(values)))
        return self

    def order(self, column: str, desc: bool = False) -> "FakeQuery":
        self.orders.append((column, desc))
        return self

    def range(self, start: int, end: int) -> "FakeQuery":  # noqa: A003 - supabase-py's name
        self.window = (start, end)
        return self

    def limit(self, count: int) -> "FakeQuery":
        self.window = (0, count - 1)
        return self

    def _matches(self, row: Dict[str, Any]) -> bool:
        for kind, column, value in self.filters:
            cell = row.get(column)
            if kind == "eq" and cell != value:
                return False
            if kind == "gte" and not (cell is not None and str(cell) >= str(value)):
                return False
            if kind == "gt" and not (cell is not None and str(cell) > str(value)):
                return False
            if kind == "in" and str(cell) not in {str(v) for v in value}:
                return False
        return True

    def execute(self) -> Any:
        self.client.queries.append(self)
        rows = [row for row in self.client.rows.get(self.table, []) if self._matches(row)]
        for column, desc in reversed(self.orders):
            rows.sort(key=lambda r: str(r.get(column) or ""), reverse=desc)
        if self.columns is not None:
            rows = [{k: v for k, v in row.items() if k in self.columns} for row in rows]
        if self.window is not None:
            start, end = self.window
            rows = rows[start : end + 1]
        self.client.after_page(self.table)
        return type("Result", (), {"data": rows})()


class FakeClient:
    def __init__(
        self,
        rows: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        after_page: Optional[Any] = None,
    ) -> None:
        self.rows = rows or {}
        self.queries: List[FakeQuery] = []
        self._after_page = after_page

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)

    def queries_for(self, table: str) -> List[FakeQuery]:
        return [q for q in self.queries if q.table == table]

    def after_page(self, table: str) -> None:
        """Hook: lets a test insert a row between page reads, as a live database would."""
        if self._after_page is not None:
            self._after_page(self, table)


def _episode(**over: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    episode = {
        "episode_id": "11111111-1111-1111-1111-111111111111",
        "composition_id": "comp_a",
        # Window membership is created_at; the double filters on it, so a row without one
        # would silently drop out of every query.
        "created_at": now.isoformat(),
        "query_text": "q",
        "status": "COMPLETED",
        "outcome": "success",
        "failed_phase": None,
        "error_type": None,
        "plan_source": "llm",
        "entry_point": "chat_tool",
        "total_latency_ms": 100.0,
        "last_activity_at": now.isoformat(),
        "tools_executed": 2,
        "tools_succeeded": 2,
        "is_synthetic": False,
    }
    episode.update(over)
    return episode


def _service(rows: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Any:
    return ToolComposerObservabilityService(client=FakeClient(rows or {}))


# ---------------------------------------------------------------------------
# Percentiles agree with the database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, fraction, expected",
    [
        ([100.0, 1000.0], 0.50, 550.0),  # percentile_cont interpolates
        ([100.0, 1000.0], 0.95, 955.0),
        ([5.0], 0.95, 5.0),
        ([], 0.50, None),
        ([1.0, 2.0, 3.0], 0.50, 2.0),
    ],
)
def test_percentiles_interpolate_like_percentile_cont(values, fraction, expected):
    assert _percentile(values, fraction) == expected


# ---------------------------------------------------------------------------
# Terminal statuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "TIMEOUT"])
def test_a_terminal_episode_is_never_unfinished_or_abandoned(status):
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    rows = {"composer_episodes": [_episode(status=status, last_activity_at=stale)]}

    compositions = _service(rows).overview(30)["compositions"]

    assert compositions["unfinished"] == 0 and compositions["abandoned"] == 0


def test_an_in_flight_episode_gone_quiet_is_abandoned():
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    rows = {
        "composer_episodes": [_episode(status="EXECUTING", outcome=None, last_activity_at=stale)]
    }

    compositions = _service(rows).overview(30)["compositions"]

    assert compositions["unfinished"] == 1 and compositions["abandoned"] == 1


# ---------------------------------------------------------------------------
# One population: the window, and the provenance flag
# ---------------------------------------------------------------------------


def test_window_membership_is_when_the_composition_started():
    service = _service({"composer_episodes": []})

    service.overview(30)

    episodes = service.client.queries_for("composer_episodes")[0]
    assert [f for f in episodes.filters if f[0] == "gte"][0][1] == "created_at"
    assert all(f[1] != "last_activity_at" for f in episodes.filters)


def test_synthetic_runs_are_filtered_out_unless_the_deployment_includes_them(monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    service = _service({"composer_episodes": []})
    service.overview(30)
    filters = service.client.queries_for("composer_episodes")[0].filters
    assert ("eq", "is_synthetic", False) in filters

    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "1")
    included = _service({"composer_episodes": []})
    included.overview(30)
    assert all(
        f[1] != "is_synthetic" for f in included.client.queries_for("composer_episodes")[0].filters
    )


# ---------------------------------------------------------------------------
# Paging cannot drop or double-count a row
# ---------------------------------------------------------------------------


def test_episode_paging_orders_by_a_unique_key_as_well():
    """last_activity_at moves under heartbeats and repeats; a page boundary needs a tiebreaker."""
    service = _service({"composer_episodes": []})

    service.overview(30)

    orders = service.client.queries_for("composer_episodes")[0].orders
    assert any(column == "episode_id" for column, _ in orders), orders


def _many_episodes(count: int, *, prefix: str = "a") -> List[Dict[str, Any]]:
    return [
        _episode(
            episode_id=f"{prefix}{n:07d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{prefix}{n}",
        )
        for n in range(count)
    ]


def test_a_composition_started_mid_read_is_neither_counted_twice_nor_lost(monkeypatch):
    """Offset paging double-counts under concurrent inserts: row N-1 slides to offset N.

    Compositions are being recorded while an admin reads the page, so this is the ordinary
    case, not a rare one.
    """
    monkeypatch.setattr("src.services.tool_composer_observability_service._PAGE", 10, raising=False)
    rows = {"composer_episodes": _many_episodes(25)}
    inserted: List[int] = []

    def insert_one(client: FakeClient, table: str) -> None:
        # One new episode arrives between pages, as a live recorder would write it. It is NEWER,
        # so under `created_at DESC` it takes first place and pushes every read row down by one.
        if table == "composer_episodes" and not inserted:
            inserted.append(1)
            client.rows["composer_episodes"].append(
                _episode(
                    episode_id="00000000-1111-1111-1111-111111111111",
                    composition_id="comp_arrived",
                    created_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
                )
            )

    service = ToolComposerObservabilityService(client=FakeClient(rows, after_page=insert_one))
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    fetched = service._fetch_episodes(since, True)

    ids = [row["episode_id"] for row in fetched]
    assert len(ids) == len(set(ids)), "an episode was read twice across a page boundary"
    assert len(ids) in (25, 26), f"an episode was lost across a page boundary: {len(ids)}"


def test_the_recent_failures_are_the_newest_ones_not_the_first_by_id():
    """The ten shown are the ten most recent, so the id order must not decide the ordering."""
    now = datetime.now(timezone.utc)
    failures = [
        _episode(
            # UUID order runs OPPOSITE to time order: the lowest id is the OLDEST episode, so a
            # sort that lost created_at would show exactly the wrong ten.
            episode_id=f"{n:08d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{n}",
            created_at=(now - timedelta(minutes=14 - n)).isoformat(),
            status="FAILED",
            outcome="failed",
        )
        for n in range(14)
    ]
    service = _service({"composer_episodes": failures, "composition_steps": []})

    shown = [row["composition_id"] for row in service.overview(30)["recent_failures"]]

    assert shown == [f"comp_{n}" for n in range(13, 3, -1)]


def test_steps_are_fetched_across_chunks_and_pages_exactly_once(monkeypatch):
    monkeypatch.setattr("src.services.tool_composer_observability_service._PAGE", 3, raising=False)
    monkeypatch.setattr(
        "src.services.tool_composer_observability_service._IN_CHUNK", 2, raising=False
    )
    failures = [
        _episode(
            episode_id=f"{n:08d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{n}",
            status="FAILED",
            outcome="failed",
        )
        for n in range(5)
    ]
    steps = [
        {
            "episode_id": episode["episode_id"],
            "step_id": f"{episode['episode_id']}-{n}",
            "step_number": n,
            "tool_name": "gap_calculator",
            "outcome_class": "error",
        }
        for episode in failures
        for n in range(4)  # more than one page of steps per episode
    ]
    service = _service({"composer_episodes": failures, "composition_steps": steps})

    failures_out = service.overview(30)["recent_failures"]

    by_composition = {row["composition_id"]: row["step_classes"] for row in failures_out}
    assert len(by_composition) == 5
    for composition_id, classes in by_composition.items():
        numbers = [s["step_number"] for s in classes]
        assert numbers == [0, 1, 2, 3], f"{composition_id} got {numbers}"


def test_steps_are_fetched_for_every_failed_episode_in_pages():
    failures = [
        _episode(
            episode_id=f"1111111{n:04d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{n}",
            status="FAILED",
            outcome="failed",
        )
        for n in range(12)
    ]
    rows = {"composer_episodes": failures, "composition_steps": []}
    service = _service(rows)

    result = service.overview(30)

    # At most ten recent failures are shown, and every one of them had its steps looked up.
    assert len(result["recent_failures"]) == 10
    asked = [f for q in service.client.queries_for("composition_steps") for f in q.filters]
    looked_up = {cid for kind, _, values in asked if kind == "in" for cid in values}
    assert len(looked_up) == 10


# ---------------------------------------------------------------------------
# #2021 D3 / D1′: codes come from the database; sentences are rendered here
# ---------------------------------------------------------------------------


def _verdict(**over: Any) -> Any:
    from src.agents.tool_composer.reliability import ToolReliability

    row = {
        "tool_name": "causal_effect_estimator",
        "n_invoked": 12,
        "n_succeeded": 4,
        "n_refused": 8,
        "n_health_failures": 0,
        "n_health": 4,
        "n_retried": 0,
        "n_synthetic": 0,
    }
    row.update(over)
    return ToolReliability.from_row(row)


def test_a_tool_row_carries_the_refusal_code_and_its_rendered_sentence():
    verdicts = {
        "causal_effect_estimator": _verdict(most_common_refusal_reason="non_binary_treatment")
    }
    service = _service({"composer_episodes": [], "composition_steps": []})
    (tool,) = service.overview(30, verdicts)["tools"]
    assert tool["most_common_refusal_reason"] == "non_binary_treatment"
    assert tool["most_common_refusal_sentence"] == (
        "the treatment column is not a binary 0/1 indicator"
    )


def test_a_tool_row_carries_how_many_refusals_the_code_was_drawn_from():
    """Most refusals recorded before ml/043 are uncoded, so the most common code can name a
    minority; the page gets both numbers rather than a code presented as representative."""
    verdicts = {
        "causal_effect_estimator": _verdict(
            n_invoked=54,
            n_refused=50,
            n_refused_coded=3,
            most_common_refusal_reason="coverage_gap",
        )
    }
    service = _service({"composer_episodes": [], "composition_steps": []})
    (tool,) = service.overview(30, verdicts)["tools"]
    assert tool["n_refused"] == 50
    assert tool["n_refused_coded"] == 3


def test_a_tool_row_carries_how_many_refusals_carry_the_most_common_code():
    """A 1-of-3 tie winner and a 3-of-3 majority name the same code; only this count differs."""
    verdicts = {
        "causal_effect_estimator": _verdict(
            n_invoked=54,
            n_refused=50,
            n_refused_coded=3,
            n_most_common_refusal_reason=1,
            most_common_refusal_reason="coverage_gap",
        )
    }
    service = _service({"composer_episodes": [], "composition_steps": []})
    (tool,) = service.overview(30, verdicts)["tools"]
    assert (tool["n_refused"], tool["n_refused_coded"], tool["n_most_common_refusal_reason"]) == (
        50,
        3,
        1,
    )


def test_an_unknown_coded_refusal_count_stays_null_not_zero():
    """A database without ml/043 cannot say how many refusals were coded; 0 would claim none were."""
    verdicts = {"causal_effect_estimator": _verdict(n_refused_coded=None)}
    service = _service({"composer_episodes": [], "composition_steps": []})
    (tool,) = service.overview(30, verdicts)["tools"]
    assert tool["n_refused_coded"] is None


def test_an_unknown_code_shows_the_code_and_no_invented_sentence():
    verdicts = {
        "causal_effect_estimator": _verdict(
            most_common_refusal_reason="a_code_this_build_does_not_know"
        )
    }
    service = _service({"composer_episodes": [], "composition_steps": []})
    (tool,) = service.overview(30, verdicts)["tools"]
    assert tool["most_common_refusal_reason"] == "a_code_this_build_does_not_know"
    assert tool["most_common_refusal_sentence"] is None


def test_a_failed_step_class_carries_its_code_and_rendered_reason():
    failed = _episode(status="FAILED", outcome="failed")
    steps = [
        {
            "episode_id": failed["episode_id"],
            "step_number": 0,
            "tool_name": "gap_calculator",
            "outcome_class": "refused",
            "reason_code": "coverage_gap",
        }
    ]
    service = _service({"composer_episodes": [failed], "composition_steps": steps})
    (row,) = service.overview(30)["recent_failures"]
    assert row["step_classes"] == [
        {
            "step_number": 0,
            "tool_name": "gap_calculator",
            "outcome_class": "refused",
            "reason_code": "coverage_gap",
            "reason": "the data does not cover everything the question asked about",
            "reason_details": {},
        }
    ]


def _step_classes_for(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    failed = _episode(status="FAILED", outcome="failed")
    steps = [{"episode_id": failed["episode_id"], "step_number": 0, **step}]
    service = _service({"composer_episodes": [failed], "composition_steps": steps})
    (row,) = service.overview(30)["recent_failures"]
    return row["step_classes"]


def test_an_uncoded_step_has_no_code_and_no_reason():
    """Every step recorded before ml/043 is uncoded: not recorded, never the tool-failure sentence."""
    (step,) = _step_classes_for({"tool_name": "gap_calculator", "outcome_class": "refused"})
    assert step["reason_code"] is None and step["reason"] is None


def test_a_step_with_an_unknown_code_keeps_the_code_and_invents_no_reason():
    (step,) = _step_classes_for(
        {"tool_name": "gap_calculator", "outcome_class": "refused", "reason_code": "not_a_code"}
    )
    assert step["reason_code"] == "not_a_code" and step["reason"] is None


# ---------------------------------------------------------------------------
# #2050: the numeric details recorded with a refusal reach the page
# ---------------------------------------------------------------------------

_LOGGER = "src.services.tool_composer_observability_service"


def test_a_failed_step_class_carries_its_recorded_reason_details(caplog):
    """ml/043 persists them; without them two diagnostics under one code read as one sentence."""
    caplog.set_level("WARNING", logger=_LOGGER)
    details = {"n_segments_named": 4, "n_no_contrast": 3, "n_non_finite": 0}

    (step,) = _step_classes_for(
        {
            "tool_name": "cate_analyzer",
            "outcome_class": "refused",
            "reason_code": "insufficient_groups",
            "reason_details": details,
        }
    )

    assert step["reason_details"] == details
    assert not [r for r in caplog.records if r.name == _LOGGER]


@pytest.mark.parametrize(
    "over",
    [pytest.param({}, id="key-absent"), pytest.param({"reason_details": None}, id="null")],
)
def test_a_step_without_recorded_details_carries_an_empty_mapping(over, caplog):
    caplog.set_level("WARNING", logger=_LOGGER)

    (step,) = _step_classes_for({"tool_name": "gap_calculator", "outcome_class": "refused", **over})

    assert step["reason_details"] == {}
    assert not [r for r in caplog.records if r.name == _LOGGER]


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param({"n_segments_named": "Brand_Secret"}, id="string-value"),
        pytest.param({"Treatment_Column": 3}, id="bad-key"),
        pytest.param({f"n_key_{i}": i for i in range(9)}, id="more-than-8-keys"),
        pytest.param({"n_no_contrast": 3, "n_brand": "Brand_Secret"}, id="one-bad-value"),
        pytest.param({"n_rows": 2**53}, id="unsafe-integer"),
        pytest.param(["n_no_contrast", 3], id="not-an-object"),
        pytest.param("Brand_Secret", id="bare-string"),
    ],
)
def test_an_invalid_stored_details_value_is_dropped_on_read_and_logged_without_values(
    stored, caplog
):
    """The table only requires a JSON object; only the RPC reducer enforces numbers. A direct write
    can store text, and text is what these details exist to keep off the page."""
    caplog.set_level("WARNING", logger=_LOGGER)

    (step,) = _step_classes_for(
        {
            "tool_name": "cate_analyzer",
            "outcome_class": "refused",
            "reason_code": "insufficient_groups",
            "reason_details": stored,
        }
    )

    assert step["reason_details"] == {}
    warnings = [r for r in caplog.records if r.name == _LOGGER and r.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "Brand_Secret" not in message and "Treatment_Column" not in message


def test_two_refusals_under_one_code_stay_distinguishable_through_the_response_model():
    """#2050: cate_analyzer's 'no treatment contrast' and 'no usable outcome' share one code and one
    sentence; only the details tell them apart. The pair must survive the service and the response
    model unchanged, booleans still booleans (bool is an int subclass, so == alone cannot tell)."""
    from src.api.schemas.admin_tool_composer import ToolComposerObservability

    failed = _episode(status="FAILED", outcome="failed")
    no_contrast = {"n_segments_named": 4, "n_no_contrast": 3, "n_non_finite": 0, "is_scoped": True}
    no_outcome = {"n_segments_named": 4, "n_no_contrast": 0, "n_non_finite": 3, "share_kept": 0.25}
    steps = [
        {"step_number": n, "tool_name": "cate_analyzer", "outcome_class": "refused",
         "reason_code": "insufficient_groups", "reason_details": details}
        for n, details in enumerate([no_contrast, no_outcome])
    ]  # fmt: skip
    rows = {
        "composer_episodes": [failed],
        "composition_steps": [{"episode_id": failed["episode_id"], **s} for s in steps],
    }

    out = _service(rows).overview(30)

    (classes,) = [f["step_classes"] for f in out["recent_failures"]]
    assert classes[0]["reason_code"] == classes[1]["reason_code"] == "insufficient_groups"
    assert classes[0]["reason"] == classes[1]["reason"]
    assert [s["reason_details"] for s in classes] == [no_contrast, no_outcome]

    dumped = ToolComposerObservability.model_validate(out).model_dump(mode="json")

    (dumped_classes,) = [f["step_classes"] for f in dumped["recent_failures"]]
    assert dumped_classes == classes
    assert dumped_classes[0]["reason_details"] != dumped_classes[1]["reason_details"]
    first, second = (s["reason_details"] for s in dumped_classes)
    assert type(first["is_scoped"]) is bool
    assert type(first["n_no_contrast"]) is int and type(second["n_non_finite"]) is int
    assert type(second["share_kept"]) is float


def test_the_service_output_validates_against_the_response_model_unchanged():
    """ToolReliabilityRow and StepClass forbid extra keys, so a key the service adds and the model
    lacks would 500 the route. The real-DB route test skips on the droplet; this one does not."""
    from src.api.schemas.admin_tool_composer import ToolComposerObservability

    failed = _episode(status="FAILED", outcome="failed")
    steps = [
        {"step_number": 0, "tool_name": "gap_calculator", "outcome_class": "refused",
         "reason_code": "coverage_gap"},
        {"step_number": 1, "tool_name": "cate_analyzer", "outcome_class": "error"},
        {"step_number": 2, "tool_name": "roi_estimator", "outcome_class": "refused",
         "reason_code": "not_a_code"},
    ]  # fmt: skip
    rows = {
        "composer_episodes": [failed],
        "composition_steps": [{"episode_id": failed["episode_id"], **s} for s in steps],
    }
    verdicts = {
        "gap_calculator": _verdict(tool_name="gap_calculator", n_refused_coded=None),
        "cate_analyzer": _verdict(tool_name="cate_analyzer", n_refused_coded=0),
        "causal_effect_estimator": _verdict(
            n_refused_coded=4,
            n_most_common_refusal_reason=2,
            most_common_refusal_reason="non_binary_treatment",
        ),
    }
    out = _service(rows).overview(30, verdicts)

    dumped = ToolComposerObservability.model_validate(out).model_dump(mode="json")

    assert dumped["tools"] == out["tools"]
    assert [f["step_classes"] for f in dumped["recent_failures"]] == [
        f["step_classes"] for f in out["recent_failures"]
    ]
    (step_classes,) = [f["step_classes"] for f in out["recent_failures"]]
    assert [s["reason_code"] for s in step_classes] == ["coverage_gap", None, "not_a_code"]
    assert sorted((t["n_refused_coded"] is None, t["n_refused_coded"]) for t in out["tools"]) == [
        (False, 0),
        (False, 4),
        (True, None),
    ]


def test_the_service_module_does_not_import_the_tool_composer_package():
    """Measured 2026-09-12: the package costs ~564 MB and ~17 s; this module alone ~47 MB."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, src.services.tool_composer_observability_service; "
        "print('src.agents.tool_composer' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert out.stdout.strip() == "False"
