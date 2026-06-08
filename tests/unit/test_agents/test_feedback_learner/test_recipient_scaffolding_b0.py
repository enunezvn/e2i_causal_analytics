"""B0 scaffolding for real per-recipient optimization (Gap B substrate).

Offline-only: fake clients, no real LM / DB. Asserts the substrate that the
later B1-B4 recipient tasks plug into:

- RECIPIENT_SIGNATURE_FIELDS covers all four recipients with importable signatures.
- emit_recipient_signal writes exactly one correctly-shaped row keyed by source_agent.
- signal_example_provider builds >=2 dspy.Example(...).with_inputs(...) from >=2
  emitted signals and <2 from one (so optimize_recipient skips that field).
- The generic heuristic metric returns a dspy.Prediction(score in [0,1], feedback=str),
  scoring a populated output high-ish and an empty output low.
- Guardrail: the golden seeds are gone from src/ and optimize_recipient no longer
  imports them — production must run on real emitted data or skip.
"""

from __future__ import annotations

import importlib.util
import inspect
from typing import Any, Dict, List

import pytest


# --------------------------------------------------------------------------- #
# Fake Supabase client recording inserts (no DB).
# --------------------------------------------------------------------------- #
class _FakeTable:
    def __init__(self, recorder: List[Dict[str, Any]]):
        self._recorder = recorder

    def insert(self, record):
        # supabase accepts a dict or a list of dicts
        if isinstance(record, list):
            self._recorder.extend(record)
        else:
            self._recorder.append(record)
        return self

    def execute(self):
        return {"data": list(self._recorder)}


class _RecordingClient:
    """Counts/records every row inserted into any table."""

    def __init__(self):
        self.inserted: List[Dict[str, Any]] = []
        self.tables_touched: List[str] = []

    def table(self, name: str):
        self.tables_touched.append(name)
        return _FakeTable(self.inserted)


class _SignalReturningClient:
    """A client whose select() returns a fixed list of emitted signal rows.

    Mirrors the surface SignalCollectorAdapter.get_signals_for_optimization uses:
    client.table(...).select(...).eq(...).gte(...).limit(...).execute().data
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def table(self, name: str):
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def gte(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        class _Resp:
            data = self._rows

        return _Resp()


def _signal_row(
    agent_name: str, signature_inputs: Dict[str, Any], generated: str
) -> Dict[str, Any]:
    return {
        "source_agent": agent_name,
        "input_context": {"signature_inputs": signature_inputs},
        "output": {"generated": generated},
        "reward": 0.9,
        "is_training_example": True,
    }


# --------------------------------------------------------------------------- #
# 1. RECIPIENT_SIGNATURE_FIELDS covers all four recipients with real signatures.
# --------------------------------------------------------------------------- #
def test_recipient_signature_fields_cover_all_four_with_importable_signatures():
    pytest.importorskip("dspy")
    import importlib

    from src.agents.feedback_learner.recipient_optimizer import RECIPIENT_SIGNATURE_FIELDS

    expected = {"experiment_monitor", "explainer", "health_score", "resource_optimizer"}
    assert expected.issubset(set(RECIPIENT_SIGNATURE_FIELDS)), (
        f"missing recipients: {expected - set(RECIPIENT_SIGNATURE_FIELDS)}"
    )

    for agent, field_map in RECIPIENT_SIGNATURE_FIELDS.items():
        assert field_map, f"{agent} has no template->signature entries"
        mod = importlib.import_module(f"src.agents.{agent}.dspy_integration")
        for field, sig_name in field_map.items():
            assert field.endswith("_template"), f"{agent}.{field} is not a *_template field"
            sig = getattr(mod, sig_name, None)
            assert sig is not None, f"{agent}: signature {sig_name} not importable"
            # Real DSPy signatures expose input_fields/output_fields.
            assert getattr(sig, "input_fields", None), f"{sig_name} has no input_fields"
            assert getattr(sig, "output_fields", None), f"{sig_name} has no output_fields"

    # experiment_monitor's original entries are preserved.
    em = RECIPIENT_SIGNATURE_FIELDS["experiment_monitor"]
    assert em["srm_template"] == "SRMDescriptionSignature"
    assert em["summary_template"] == "MonitorSummarySignature"
    assert em["alert_template"] == "AlertGenerationSignature"


# --------------------------------------------------------------------------- #
# 2. emit_recipient_signal writes one correctly-shaped row keyed by source_agent.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_emit_recipient_signal_writes_one_row():
    from src.agents.feedback_learner.recipient_emit import emit_recipient_signal

    client = _RecordingClient()
    ok = await emit_recipient_signal(
        agent_name="health_score",
        signature_inputs={"overall_score": 72.0, "grade": "C", "component_scores": "x"},
        generated_output="Health is fair (72/100, grade C); data freshness is the weak component.",
        reward=0.81,
        client=client,
    )

    assert ok is True
    assert len(client.inserted) == 1, "must write exactly one row"
    row = client.inserted[0]
    assert row["source_agent"] == "health_score"
    assert row["input_context"]["signature_inputs"]["grade"] == "C"
    assert row["output"]["generated"].startswith("Health is fair")
    assert row["reward"] == 0.81
    assert row["is_training_example"] is True
    assert "dspy_agent_training_signals" in client.tables_touched


@pytest.mark.asyncio
async def test_emit_recipient_signal_is_best_effort_never_raises():
    from src.agents.feedback_learner.recipient_emit import emit_recipient_signal

    class _Boom:
        def table(self, *_a, **_k):
            raise RuntimeError("db down")

    ok = await emit_recipient_signal(
        agent_name="explainer",
        signature_inputs={"analysis_results": "r"},
        generated_output="out",
        reward=0.5,
        client=_Boom(),
    )
    assert ok is False  # swallowed, returns False, does not raise


# --------------------------------------------------------------------------- #
# 3. signal_example_provider: >=2 signals -> >=2 Examples with inputs; 1 -> <2.
# --------------------------------------------------------------------------- #
def test_signal_example_provider_builds_examples_from_emitted_signals():
    pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_optimizer import signal_example_provider

    inputs = {
        "experiment_name": "Kisqali-NE",
        "chi_squared": 12.4,
        "p_value": 0.0004,
        "expected_ratio": "50/50",
        "actual_counts": "640/360",
    }
    rows = [
        _signal_row("experiment_monitor", inputs, "Significant SRM detected (p=0.0004)."),
        _signal_row("experiment_monitor", inputs, "Arm imbalance threatens validity."),
        _signal_row("experiment_monitor", inputs, "Freeze enrollment; audit assignment."),
    ]
    provider = signal_example_provider("experiment_monitor", client=_SignalReturningClient(rows))
    examples = provider("srm_template")
    assert len(examples) >= 2
    first = examples[0]
    # Inputs declared via with_inputs so GEPA treats them as inputs, not labels.
    assert "experiment_name" in first.inputs()
    # The gold output field (first output field of the signature) carries the text.
    assert getattr(first, "explanation", None)


def test_signal_example_provider_skips_below_two():
    pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_optimizer import signal_example_provider

    inputs = {
        "experiment_name": "E",
        "chi_squared": 1.0,
        "p_value": 0.5,
        "expected_ratio": "50/50",
        "actual_counts": "1/1",
    }
    rows = [_signal_row("experiment_monitor", inputs, "only one signal")]
    provider = signal_example_provider("experiment_monitor", client=_SignalReturningClient(rows))
    examples = provider("srm_template")
    assert len(examples) < 2  # optimize_recipient will skip this field


# --------------------------------------------------------------------------- #
# 3b. Emit<->provider round-trip for explainer / health_score / resource_optimizer
#     (the 3 recipients whose template placeholders DIFFER from signature inputs).
#     These double as the B1-B4 contract spec: emit signature_inputs keyed by the
#     signature's input_fields (discovered via recipient_required_input_keys).
# --------------------------------------------------------------------------- #
def _dummy_value(key: str) -> Any:
    """A plausible value per input field name; numeric for *score*/*value*."""
    if "score" in key or "value" in key or key.endswith("_count"):
        return 72.0
    return f"<{key}>"


@pytest.mark.parametrize(
    "agent,field",
    [
        ("explainer", "executive_summary_template"),
        ("explainer", "insight_extraction_template"),
        ("explainer", "narrative_section_template"),
        ("health_score", "summary_template"),
        ("health_score", "recommendation_template"),
        ("resource_optimizer", "summary_template"),
        ("resource_optimizer", "recommendation_template"),
        ("resource_optimizer", "scenario_comparison_template"),
    ],
)
def test_emit_provider_roundtrip_for_each_recipient(agent, field):
    pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_optimizer import (
        recipient_required_input_keys,
        signal_example_provider,
    )

    required = recipient_required_input_keys(agent)[field]
    assert required, f"{agent}.{field} must declare required input keys"

    # B1-B4 contract: key signature_inputs by the SIGNATURE's input_fields.
    sig_inputs = {k: _dummy_value(k) for k in required}
    rows = [_signal_row(agent, sig_inputs, f"grounded output #{i} for {field}") for i in range(2)]

    provider = signal_example_provider(agent, client=_SignalReturningClient(rows))
    examples = provider(field)
    assert len(examples) >= 2, f"{agent}.{field} round-trip produced <2 examples"
    # Every required input is declared as an input on the built Example.
    declared = set(examples[0].inputs().keys())
    assert set(required).issubset(declared)


def test_recipient_required_input_keys_contract():
    pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_optimizer import (
        RECIPIENT_SIGNATURE_FIELDS,
        recipient_required_input_keys,
    )

    for agent, field_map in RECIPIENT_SIGNATURE_FIELDS.items():
        keys = recipient_required_input_keys(agent)
        assert set(keys) == set(field_map), f"{agent}: helper must cover every template field"
        for field, required in keys.items():
            assert required, f"{agent}.{field} has no required input keys"


def test_provider_warns_on_key_mismatch(caplog):
    """When rows exist but use the WRONG keys (template placeholders, say), the
    provider must WARN loudly and return nothing — never silently 0 examples."""
    import logging

    pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_optimizer import signal_example_provider

    # health_score.summary_template needs overall_score/grade/component_scores/
    # critical_issues; emit with wrong (template-placeholder-ish) keys instead.
    wrong = {"health_status": "fair", "score_value": 72}
    rows = [_signal_row("health_score", wrong, f"out {i}") for i in range(2)]

    provider = signal_example_provider("health_score", client=_SignalReturningClient(rows))
    with caplog.at_level(logging.WARNING):
        examples = provider("summary_template")
    assert examples == []
    assert any(
        "NONE matched" in r.message and "health_score" in r.message for r in caplog.records
    ), "key mismatch must produce a WARNING naming the agent"


# --------------------------------------------------------------------------- #
# 4. Generic heuristic metric: Prediction(score in [0,1], feedback:str).
# --------------------------------------------------------------------------- #
def test_generic_metric_scores_populated_high_empty_low():
    dspy = pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_metrics import get_recipient_metric

    metric = get_recipient_metric("explainer")
    gold = dspy.Example(
        analysis_results="TRx for Kisqali rose 12% in the Northeast after the new HCP program.",
        user_expertise="executive",
        focus_areas="adoption",
        output_format="brief",
    ).with_inputs("analysis_results", "user_expertise", "focus_areas", "output_format")

    populated = dspy.Prediction(
        executive_summary=(
            "Kisqali TRx grew 12% in the Northeast following the HCP adoption program; "
            "recommend sustaining the program and expanding to adjacent regions."
        )
    )
    empty = dspy.Prediction(executive_summary="")

    good = metric(gold, populated)
    bad = metric(gold, empty)

    for r in (good, bad):
        assert isinstance(r, dspy.Prediction)
        assert 0.0 <= float(r.score) <= 1.0
        assert isinstance(r.feedback, str)

    assert float(good.score) > float(bad.score)
    assert float(bad.score) < 0.3  # empty output scores low


def test_get_recipient_metric_returns_callable_for_all_recipients():
    pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_metrics import get_recipient_metric

    for agent in ("experiment_monitor", "explainer", "health_score", "resource_optimizer"):
        assert callable(get_recipient_metric(agent))


# --------------------------------------------------------------------------- #
# 5. Guardrail: golden seeds gone from src; optimize_recipient never imports them.
# --------------------------------------------------------------------------- #
def test_recipient_seeds_not_importable_from_src():
    assert importlib.util.find_spec("src.agents.feedback_learner.recipient_seeds") is None, (
        "recipient_seeds must be relocated out of src/ (test-only fixture)"
    )


def test_optimize_recipient_source_has_no_seed_import():
    from src.agents.feedback_learner import recipient_optimizer

    src = inspect.getsource(recipient_optimizer)
    assert "recipient_seeds" not in src, (
        "production optimizer must not import the golden seeds; "
        "real emitted signals or skip (cold-start) only"
    )
