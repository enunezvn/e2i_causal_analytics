"""#1791: the SLO monitor's unknown-agent policy.

``AGENT_TIER_MAP`` is now a projection of ``AGENT_REGISTRY_CONFIG``, so
``get_agent_tier`` raises for a name it does not know instead of quietly
returning ``TIER_2_CAUSAL`` and letting ``get_slo_target`` hand back real
TIER_2 targets for a typo.

That raise creates a hazard of its own, which is what this file pins.
``SLOMonitor.record()`` accepts ANY name, so without care a single unknown one
would sit in ``_records`` and turn every aggregate read into an exception --
trading a silently-wrong answer for a broken dashboard. The policy is
deliberately different per path, because each has a different job:

===========================  ==================================================
path                         unknown agent
===========================  ==================================================
``get_agent_tier``           raises -- a precise question gets a precise answer
``get_slo_target``           raises -- same, it is a thin wrapper
``record()``                 records + warns once -- instrumentation must never
                             break the system it instruments
``get_all_compliance()``     omits it -- an aggregate must degrade, not explode
``get_tier_compliance()``    unreachable -- it iterates AGENT_TIER_MAP itself
===========================  ==================================================

The through-line is omit-or-report, never fabricate. Omitting an agent from a
summary is honest about having no SLO for it; the old default invented one.
"""

import logging

import pytest

from src.agents.factory import AGENT_REGISTRY_CONFIG
from src.mlops.slo_monitor import (
    AGENT_TIER_MAP,
    SLOMonitor,
    UnknownAgentError,
    get_agent_tier,
    get_slo_target,
    reset_slo_monitor,
)

pytestmark = pytest.mark.unit

UNKNOWN = "definitely_not_a_registered_agent_1791"
#: A real agent, taken from the registry rather than hardcoded, so this file
#: cannot rot the way the roster it guards did.
KNOWN = sorted(AGENT_REGISTRY_CONFIG)[0]


@pytest.fixture(autouse=True)
def _reset():
    reset_slo_monitor()
    yield
    reset_slo_monitor()


@pytest.fixture()
def monitor() -> SLOMonitor:
    return SLOMonitor()


class TestPreconditions:
    """Positive controls -- without these the tests below prove nothing."""

    def test_the_known_agent_really_is_known(self) -> None:
        assert KNOWN in AGENT_TIER_MAP

    def test_the_unknown_agent_really_is_unknown(self) -> None:
        """If UNKNOWN ever became a real agent, every test here would go vacuous."""
        assert UNKNOWN not in AGENT_TIER_MAP
        assert UNKNOWN not in AGENT_REGISTRY_CONFIG


class TestPreciseLookupRaises:
    """A direct question about one agent must not be answered with a guess."""

    def test_get_agent_tier_raises(self) -> None:
        with pytest.raises(UnknownAgentError):
            get_agent_tier(UNKNOWN)

    def test_get_slo_target_raises(self) -> None:
        with pytest.raises(UnknownAgentError):
            get_slo_target(UNKNOWN)

    def test_the_error_names_the_agent(self) -> None:
        """A guard that raises without saying WHAT it rejected is hard to act on."""
        with pytest.raises(UnknownAgentError) as excinfo:
            get_agent_tier(UNKNOWN)
        assert UNKNOWN in str(excinfo.value)

    def test_it_is_catchable_as_keyerror(self) -> None:
        """Subclassing KeyError keeps `except KeyError` callers working."""
        assert issubclass(UnknownAgentError, KeyError)
        with pytest.raises(KeyError):
            get_agent_tier(UNKNOWN)


class TestInstrumentationNeverBreaks:
    """``record()`` is called from the code being measured. It must not raise."""

    def test_recording_an_unknown_agent_does_not_raise(self, monitor: SLOMonitor) -> None:
        record = monitor.record(agent_name=UNKNOWN, latency_ms=12.0, success=True)
        assert record.agent_name == UNKNOWN

    def test_the_record_is_kept_not_dropped(self, monitor: SLOMonitor) -> None:
        """Warn, don't discard -- silently losing metrics is its own defect."""
        monitor.record(agent_name=UNKNOWN, latency_ms=12.0, success=True)
        assert UNKNOWN in monitor._records
        assert len(monitor._records[UNKNOWN]) == 1

    def test_it_warns_once_naming_the_agent(
        self, monitor: SLOMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="src.mlops.slo_monitor"):
            for _ in range(3):
                monitor.record(agent_name=UNKNOWN, latency_ms=1.0, success=True)
        hits = [r for r in caplog.records if UNKNOWN in r.getMessage()]
        assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"

    def test_a_known_agent_does_not_warn(
        self, monitor: SLOMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Positive control for the test above: the warning is not unconditional."""
        with caplog.at_level(logging.WARNING, logger="src.mlops.slo_monitor"):
            monitor.record(agent_name=KNOWN, latency_ms=1.0, success=True)
        assert [r for r in caplog.records if KNOWN in r.getMessage()] == []


class TestAggregatesDegradeRatherThanExplode:
    """One bad name must not take out every dashboard read.

    This is the regression the raise would otherwise have introduced: before
    the omit-filter, a single unknown name in ``_records`` made
    ``get_all_compliance`` -- and therefore ``get_summary``,
    ``get_violated_slos`` and ``get_metrics_for_prometheus`` -- raise for ALL
    agents, including the healthy ones.
    """

    @pytest.fixture()
    def poisoned(self, monitor: SLOMonitor) -> SLOMonitor:
        """A monitor holding one good agent and one unregistered name."""
        monitor.record(agent_name=KNOWN, latency_ms=10.0, success=True)
        monitor.record(agent_name=UNKNOWN, latency_ms=10.0, success=True)
        return monitor

    def test_the_poison_is_really_there(self, poisoned: SLOMonitor) -> None:
        """Positive control: without this the tests below could pass vacuously."""
        assert UNKNOWN in poisoned._records
        assert KNOWN in poisoned._records

    def test_get_all_compliance_does_not_raise(self, poisoned: SLOMonitor) -> None:
        result = poisoned.get_all_compliance()
        assert KNOWN in result, "the healthy agent must still be reported"
        assert UNKNOWN not in result, "the unknown agent must be omitted, not invented"

    def test_get_summary_does_not_raise(self, poisoned: SLOMonitor) -> None:
        assert poisoned.get_summary() is not None

    def test_get_violated_slos_does_not_raise(self, poisoned: SLOMonitor) -> None:
        assert isinstance(poisoned.get_violated_slos(), list)

    def test_a_direct_query_still_raises(self, poisoned: SLOMonitor) -> None:
        """Degrading the AGGREGATE must not soften the PRECISE read."""
        with pytest.raises(UnknownAgentError):
            poisoned.get_compliance(UNKNOWN)

    def test_tier_compliance_is_unaffected(self, poisoned: SLOMonitor) -> None:
        """It iterates AGENT_TIER_MAP, so an unknown recorded name cannot reach it."""
        tier = AGENT_TIER_MAP[KNOWN]
        result = poisoned.get_tier_compliance(tier)
        assert UNKNOWN not in result
