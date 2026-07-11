"""Test fixtures for Experiment Monitor Agent tests.

Provides mock database clients, sample experiments, and shared utilities
for testing the 4-node monitoring workflow:
    health_checker -> srm_detector -> interim_analyzer -> alert_generator

Fixtures are designed to cover:
- Healthy experiments (no issues)
- Warning experiments (moderate issues)
- Critical experiments (severe issues)
- Edge cases (empty, single, large scale)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.agents.experiment_monitor.state import (
    AlertSeverity,
    EnrollmentIssue,
    ErrorDetails,
    ExperimentMonitorState,
    ExperimentSummary,
    FidelityIssue,
    HealthStatus,
    InterimTrigger,
    MonitorAlert,
    MonitorStatus,
    SRMIssue,
)

# ============================================================================
# MOCK SUPABASE CLIENT
# ============================================================================


class MockSupabaseResult:
    """Mock Supabase query result."""

    def __init__(self, data: Optional[List[Dict]] = None, count: Optional[int] = None):
        self.data = data or []
        self.count = count


class MockSupabaseQuery:
    """Mock Supabase query builder with chainable methods."""

    def __init__(self, table_name: str, client: "MockSupabaseClient"):
        self.table_name = table_name
        self.client = client
        self._filters: List[Dict] = []
        self._select_cols: str = "*"
        self._count_type: Optional[str] = None
        self._order: Optional[tuple] = None
        self._limit: Optional[int] = None

    def select(self, cols: str = "*", count: Optional[str] = None) -> "MockSupabaseQuery":
        """Mock select method."""
        self._select_cols = cols
        self._count_type = count
        return self

    def eq(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock eq filter."""
        self._filters.append({"type": "eq", "column": column, "value": value})
        return self

    def in_(self, column: str, values: List[Any]) -> "MockSupabaseQuery":
        """Mock in_ filter."""
        self._filters.append({"type": "in", "column": column, "values": values})
        return self

    @property
    def not_(self) -> "_MockNotProxy":
        """Mock PostgREST negation, e.g. ``.not_.is_('col', 'null')``.

        HealthCheckerNode._get_experiments filters lineage rows out of the
        active sweep with ``.not_.is_('intervention_channel', 'null')``
        (migration 102); without this property the real chain raised
        AttributeError, which the node's ``except`` swallowed into an empty
        roster."""
        return _MockNotProxy(self)

    def order(self, column: str, desc: bool = False, **kwargs: Any) -> "MockSupabaseQuery":
        """Mock order: record the sort key so execute() returns deterministic rows.

        HealthCheckerNode._get_experiments bounds the active sweep with
        ``.order('created_at', desc=True).limit(25)``; without these chainable
        methods the real query chain raised AttributeError, which the node's
        ``except`` swallowed into an empty roster (the synthetic-plumb tests then
        saw set())."""
        self._order = (column, desc)
        return self

    def limit(self, count: int) -> "MockSupabaseQuery":
        """Mock limit: cap the returned rows (applied after order in execute)."""
        self._limit = count
        return self

    async def execute(self) -> MockSupabaseResult:
        """Execute the mock query."""
        self.client.call_count += 1
        self.client.call_history.append(
            {
                "table": self.table_name,
                "select": self._select_cols,
                "filters": self._filters,
                "count": self._count_type,
                "order": self._order,
                "limit": self._limit,
            }
        )

        # Return mock data based on table and filters
        data = self.client.get_mock_data(self.table_name, self._filters)
        # Apply order then limit to mirror the real PostgREST chain (ISO-8601
        # timestamps sort lexicographically == chronologically).
        if self._order is not None:
            col, desc = self._order
            data = sorted(data, key=lambda d: (d.get(col) is None, d.get(col)), reverse=desc)
        if self._limit is not None:
            data = data[: self._limit]
        count = len(data) if self._count_type == "exact" else None

        return MockSupabaseResult(data=data, count=count)


class _MockNotProxy:
    """Negation proxy so ``query.not_.is_(col, 'null')`` records a not_is filter."""

    def __init__(self, query: "MockSupabaseQuery"):
        self._query = query

    def is_(self, column: str, value: Any) -> "MockSupabaseQuery":
        self._query._filters.append({"type": "not_is", "column": column, "value": value})
        return self._query


class MockSupabaseClient:
    """Mock Supabase client for testing without database connection."""

    def __init__(self):
        self.call_count = 0
        self.call_history: List[Dict] = []
        self._mock_data: Dict[str, List[Dict]] = {
            "ml_experiments": [],
            "ab_experiment_assignments": [],
            "ab_interim_analyses": [],
        }

    def table(self, name: str) -> MockSupabaseQuery:
        """Get a table query builder."""
        return MockSupabaseQuery(name, self)

    def set_mock_data(self, table: str, data: List[Dict]) -> None:
        """Set mock data for a table."""
        self._mock_data[table] = data

    def get_mock_data(self, table: str, filters: List[Dict]) -> List[Dict]:
        """Get filtered mock data."""
        data = self._mock_data.get(table, [])

        for f in filters:
            if f["type"] == "eq":
                data = [d for d in data if d.get(f["column"]) == f["value"]]
            elif f["type"] == "in":
                data = [d for d in data if d.get(f["column"]) in f["values"]]
            elif f["type"] == "not_is" and f["value"] == "null":
                # Faithful IS NOT NULL: a missing key is NULL, so rows without
                # the column are excluded (mirrors PostgREST, keeps the
                # genuine-A/B predicate honest in tests).
                data = [d for d in data if d.get(f["column"]) is not None]

        return data

    def reset(self) -> None:
        """Reset call history and counts."""
        self.call_count = 0
        self.call_history = []


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    return MockSupabaseClient()


@pytest.fixture
def mock_supabase_with_experiments(mock_supabase_client):
    """Mock client with sample experiment data."""
    now = datetime.now(timezone.utc)

    # Add experiments
    mock_supabase_client.set_mock_data(
        "ml_experiments",
        [
            {
                "id": "exp-001",
                "name": "Healthy Experiment",
                "status": "running",
                "config": {
                    "target_sample_size": 1000,
                    "allocation_ratio": {"control": 0.5, "treatment": 0.5},
                },
                "created_at": (now - timedelta(days=7)).isoformat(),
            },
            {
                "id": "exp-002",
                "name": "Warning Experiment",
                "status": "running",
                "config": {
                    "target_sample_size": 500,
                    "allocation_ratio": {"control": 0.5, "treatment": 0.5},
                },
                "created_at": (now - timedelta(days=10)).isoformat(),
            },
            {
                "id": "exp-003",
                "name": "Critical Experiment",
                "status": "running",
                "config": {
                    "target_sample_size": 2000,
                    "allocation_ratio": {"control": 0.5, "treatment": 0.5},
                },
                "created_at": (now - timedelta(days=21)).isoformat(),
            },
        ],
    )

    # Add assignments (balanced for exp-001, imbalanced for exp-002)
    assignments = []
    # exp-001: 500 assignments, balanced (250/250)
    for _i in range(250):
        assignments.append({"experiment_id": "exp-001", "variant": "control"})
        assignments.append({"experiment_id": "exp-001", "variant": "treatment"})
    # exp-002: 200 assignments, imbalanced (140/60 - SRM)
    for _i in range(140):
        assignments.append({"experiment_id": "exp-002", "variant": "control"})
    for _i in range(60):
        assignments.append({"experiment_id": "exp-002", "variant": "treatment"})
    # exp-003: 50 assignments only (low enrollment)
    for _i in range(25):
        assignments.append({"experiment_id": "exp-003", "variant": "control"})
        assignments.append({"experiment_id": "exp-003", "variant": "treatment"})

    mock_supabase_client.set_mock_data("ab_experiment_assignments", assignments)

    return mock_supabase_client


# ============================================================================
# SAMPLE EXPERIMENT FIXTURES
# ============================================================================


@pytest.fixture
def sample_experiment_healthy() -> Dict[str, Any]:
    """A single healthy experiment."""
    now = datetime.now(timezone.utc)
    return {
        "id": "exp-healthy",
        "name": "Healthy Test Experiment",
        "status": "running",
        "config": {
            "target_sample_size": 1000,
            "allocation_ratio": {"control": 0.5, "treatment": 0.5},
        },
        "created_at": (now - timedelta(days=7)).isoformat(),
    }


@pytest.fixture
def sample_experiment_warning() -> Dict[str, Any]:
    """Experiment with warning-level issues."""
    now = datetime.now(timezone.utc)
    return {
        "id": "exp-warning",
        "name": "Warning Test Experiment",
        "status": "running",
        "config": {
            "target_sample_size": 500,
            "allocation_ratio": {"control": 0.5, "treatment": 0.5},
        },
        "created_at": (now - timedelta(days=10)).isoformat(),
    }


@pytest.fixture
def sample_experiment_critical() -> Dict[str, Any]:
    """Experiment with critical issues."""
    now = datetime.now(timezone.utc)
    return {
        "id": "exp-critical",
        "name": "Critical Test Experiment",
        "status": "running",
        "config": {
            "target_sample_size": 2000,
            "allocation_ratio": {"control": 0.5, "treatment": 0.5},
        },
        "created_at": (now - timedelta(days=21)).isoformat(),
    }


@pytest.fixture
def sample_experiments(
    sample_experiment_healthy, sample_experiment_warning, sample_experiment_critical
) -> List[Dict[str, Any]]:
    """List of experiments covering healthy, warning, and critical states."""
    return [sample_experiment_healthy, sample_experiment_warning, sample_experiment_critical]


# ============================================================================
# EXPERIMENT SUMMARY FIXTURES
# ============================================================================


@pytest.fixture
def sample_summary_healthy() -> ExperimentSummary:
    """Summary of a healthy experiment."""
    return ExperimentSummary(
        experiment_id="exp-healthy",
        name="Healthy Test Experiment",
        status="running",
        health_status="healthy",
        days_running=7,
        total_enrolled=500,
        enrollment_rate=71.43,
        current_information_fraction=0.50,
    )


@pytest.fixture
def sample_summary_warning() -> ExperimentSummary:
    """Summary of an experiment with warning status."""
    return ExperimentSummary(
        experiment_id="exp-warning",
        name="Warning Test Experiment",
        status="running",
        health_status="warning",
        days_running=10,
        total_enrolled=200,
        enrollment_rate=20.0,
        current_information_fraction=0.40,
    )


@pytest.fixture
def sample_summary_critical() -> ExperimentSummary:
    """Summary of an experiment with critical status."""
    return ExperimentSummary(
        experiment_id="exp-critical",
        name="Critical Test Experiment",
        status="running",
        health_status="critical",
        days_running=21,
        total_enrolled=50,
        enrollment_rate=2.38,
        current_information_fraction=0.025,
    )


@pytest.fixture
def sample_experiment_summaries(
    sample_summary_healthy, sample_summary_warning, sample_summary_critical
) -> List[ExperimentSummary]:
    """List of experiment summaries."""
    return [sample_summary_healthy, sample_summary_warning, sample_summary_critical]


# ============================================================================
# ISSUE FIXTURES
# ============================================================================


@pytest.fixture
def sample_srm_issue() -> SRMIssue:
    """Sample SRM issue (Sample Ratio Mismatch)."""
    return SRMIssue(
        experiment_id="exp-warning",
        detected=True,
        p_value=0.00012,
        chi_squared=14.56,
        expected_ratio={"control": 0.5, "treatment": 0.5},
        actual_counts={"control": 140, "treatment": 60},
        severity="critical",
    )


@pytest.fixture
def sample_srm_issue_warning() -> SRMIssue:
    """SRM issue with warning severity."""
    return SRMIssue(
        experiment_id="exp-other",
        detected=True,
        p_value=0.00085,
        chi_squared=9.2,
        expected_ratio={"control": 0.5, "treatment": 0.5},
        actual_counts={"control": 120, "treatment": 80},
        severity="warning",
    )


@pytest.fixture
def sample_enrollment_issue() -> EnrollmentIssue:
    """Sample enrollment issue."""
    return EnrollmentIssue(
        experiment_id="exp-critical",
        current_rate=2.38,
        expected_rate=5.0,
        days_below_threshold=21,
        severity="critical",
    )


@pytest.fixture
def sample_enrollment_issue_warning() -> EnrollmentIssue:
    """Enrollment issue with warning severity."""
    return EnrollmentIssue(
        experiment_id="exp-warning",
        current_rate=3.5,
        expected_rate=5.0,
        days_below_threshold=10,
        severity="warning",
    )


@pytest.fixture
def sample_fidelity_issue() -> FidelityIssue:
    """Sample Digital Twin fidelity issue."""
    return FidelityIssue(
        experiment_id="exp-fidelity",
        twin_simulation_id="sim-001",
        predicted_effect=0.15,
        actual_effect=0.08,
        prediction_error=0.467,
        calibration_needed=True,
        severity="warning",
    )


@pytest.fixture
def sample_interim_trigger() -> InterimTrigger:
    """Sample interim analysis trigger."""
    return InterimTrigger(
        experiment_id="exp-healthy",
        analysis_number=2,
        information_fraction=0.52,
        milestone_reached="50%",
        triggered=True,
    )


# ============================================================================
# ALERT FIXTURES
# ============================================================================


@pytest.fixture
def sample_alert_srm() -> MonitorAlert:
    """Sample SRM alert."""
    return MonitorAlert(
        alert_id="alert-001",
        alert_type="srm",
        severity="critical",
        experiment_id="exp-warning",
        experiment_name="Warning Test Experiment",
        message="Sample Ratio Mismatch detected (p=0.000120)",
        details={
            "p_value": 0.00012,
            "chi_squared": 14.56,
            "expected_ratio": {"control": 0.5, "treatment": 0.5},
            "actual_counts": {"control": 140, "treatment": 60},
        },
        recommended_action="Investigate randomization process and data collection.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_alert_enrollment() -> MonitorAlert:
    """Sample enrollment alert."""
    return MonitorAlert(
        alert_id="alert-002",
        alert_type="enrollment",
        severity="critical",
        experiment_id="exp-critical",
        experiment_name="Critical Test Experiment",
        message="Low enrollment rate: 2.38/day (expected: 5.0/day)",
        details={
            "current_rate": 2.38,
            "expected_rate": 5.0,
            "days_below_threshold": 21,
        },
        recommended_action="Review experiment eligibility criteria and targeting.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_alert_interim() -> MonitorAlert:
    """Sample interim trigger alert."""
    return MonitorAlert(
        alert_id="alert-003",
        alert_type="interim_trigger",
        severity="info",
        experiment_id="exp-healthy",
        experiment_name="Healthy Test Experiment",
        message="Interim analysis #2 triggered at 50% enrollment",
        details={
            "analysis_number": 2,
            "information_fraction": 0.52,
            "milestone": "50%",
        },
        recommended_action="Review interim analysis results.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================================
# STATE FIXTURES
# ============================================================================


@pytest.fixture
def base_monitor_state() -> ExperimentMonitorState:
    """Base/empty ExperimentMonitorState."""
    return ExperimentMonitorState(
        query="Check experiment health",
        experiment_ids=[],
        check_all_active=True,
        srm_threshold=0.001,
        enrollment_threshold=5.0,
        fidelity_threshold=0.2,
        check_interim=True,
        experiments=[],
        srm_issues=[],
        enrollment_issues=[],
        fidelity_issues=[],
        interim_triggers=[],
        alerts=[],
        monitor_summary="",
        recommended_actions=[],
        check_latency_ms=0,
        experiments_checked=0,
        errors=[],
        warnings=[],
        status="pending",
    )


@pytest.fixture
def state_after_health_check(
    base_monitor_state, sample_experiment_summaries
) -> ExperimentMonitorState:
    """State after health_checker node execution."""
    state = base_monitor_state.copy()
    state["status"] = "checking"
    state["experiments"] = sample_experiment_summaries
    state["experiments_checked"] = len(sample_experiment_summaries)
    state["check_latency_ms"] = 150
    return state


@pytest.fixture
def state_after_srm_detection(state_after_health_check, sample_srm_issue) -> ExperimentMonitorState:
    """State after srm_detector node execution."""
    state = state_after_health_check.copy()
    state["srm_issues"] = [sample_srm_issue]
    state["check_latency_ms"] = 250
    return state


@pytest.fixture
def state_after_interim_analysis(
    state_after_srm_detection, sample_interim_trigger
) -> ExperimentMonitorState:
    """State after interim_analyzer node execution."""
    state = state_after_srm_detection.copy()
    state["interim_triggers"] = [sample_interim_trigger]
    state["check_latency_ms"] = 320
    return state


@pytest.fixture
def state_with_all_issues(
    sample_experiment_summaries,
    sample_srm_issue,
    sample_enrollment_issue,
    sample_fidelity_issue,
    sample_interim_trigger,
) -> ExperimentMonitorState:
    """State with all types of issues for comprehensive testing."""
    return ExperimentMonitorState(
        query="Full health check",
        experiment_ids=[],
        check_all_active=True,
        srm_threshold=0.001,
        enrollment_threshold=5.0,
        fidelity_threshold=0.2,
        check_interim=True,
        experiments=sample_experiment_summaries,
        srm_issues=[sample_srm_issue],
        enrollment_issues=[sample_enrollment_issue],
        fidelity_issues=[sample_fidelity_issue],
        interim_triggers=[sample_interim_trigger],
        alerts=[],
        monitor_summary="",
        recommended_actions=[],
        check_latency_ms=0,
        experiments_checked=3,
        errors=[],
        warnings=[],
        status="analyzing",
    )


@pytest.fixture
def completed_state_healthy() -> ExperimentMonitorState:
    """Completed state with all healthy experiments."""
    return ExperimentMonitorState(
        query="Check all active experiments",
        experiment_ids=[],
        check_all_active=True,
        srm_threshold=0.001,
        enrollment_threshold=5.0,
        fidelity_threshold=0.2,
        check_interim=True,
        experiments=[
            ExperimentSummary(
                experiment_id="exp-001",
                name="Experiment 1",
                status="running",
                health_status="healthy",
                days_running=7,
                total_enrolled=350,
                enrollment_rate=50.0,
                current_information_fraction=0.35,
            )
        ],
        srm_issues=[],
        enrollment_issues=[],
        fidelity_issues=[],
        interim_triggers=[],
        alerts=[],
        monitor_summary="All experiments healthy",
        recommended_actions=["All experiments are running healthily - no action required"],
        check_latency_ms=450,
        experiments_checked=1,
        errors=[],
        warnings=[],
        status="completed",
    )


@pytest.fixture
def state_with_errors() -> ExperimentMonitorState:
    """State with error details for error handling tests."""
    return ExperimentMonitorState(
        query="Check experiments",
        experiment_ids=[],
        check_all_active=True,
        srm_threshold=0.001,
        enrollment_threshold=5.0,
        fidelity_threshold=0.2,
        check_interim=True,
        experiments=[],
        srm_issues=[],
        enrollment_issues=[],
        fidelity_issues=[],
        interim_triggers=[],
        alerts=[],
        monitor_summary="Monitoring failed",
        recommended_actions=[],
        check_latency_ms=50,
        experiments_checked=0,
        errors=[
            ErrorDetails(
                node="health_checker",
                error="Database connection failed",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ],
        warnings=["Using fallback data"],
        status="failed",
    )


# ============================================================================
# VARIANT COUNT FIXTURES (for SRM testing)
# ============================================================================


@pytest.fixture
def balanced_variant_counts() -> Dict[str, int]:
    """Balanced variant counts (no SRM)."""
    return {"control": 250, "treatment": 250}


@pytest.fixture
def imbalanced_variant_counts() -> Dict[str, int]:
    """Imbalanced variant counts (SRM detected)."""
    return {"control": 300, "treatment": 200}


@pytest.fixture
def severely_imbalanced_counts() -> Dict[str, int]:
    """Severely imbalanced counts (critical SRM)."""
    return {"control": 400, "treatment": 100}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_experiment_summary(
    experiment_id: str,
    name: str = "Test Experiment",
    health_status: HealthStatus = "healthy",
    days_running: int = 7,
    total_enrolled: int = 500,
    enrollment_rate: float = 71.43,
    information_fraction: float = 0.5,
) -> ExperimentSummary:
    """Helper to create experiment summaries with customizable fields."""
    return ExperimentSummary(
        experiment_id=experiment_id,
        name=name,
        status="running",
        health_status=health_status,
        days_running=days_running,
        total_enrolled=total_enrolled,
        enrollment_rate=enrollment_rate,
        current_information_fraction=information_fraction,
    )


def create_srm_issue(
    experiment_id: str,
    p_value: float = 0.00012,
    severity: AlertSeverity = "critical",
    control_count: int = 140,
    treatment_count: int = 60,
) -> SRMIssue:
    """Helper to create SRM issues with customizable fields."""
    return SRMIssue(
        experiment_id=experiment_id,
        detected=True,
        p_value=p_value,
        chi_squared=14.56,
        expected_ratio={"control": 0.5, "treatment": 0.5},
        actual_counts={"control": control_count, "treatment": treatment_count},
        severity=severity,
    )


def create_enrollment_issue(
    experiment_id: str,
    current_rate: float = 2.5,
    expected_rate: float = 5.0,
    days_below: int = 14,
    severity: AlertSeverity = "critical",
) -> EnrollmentIssue:
    """Helper to create enrollment issues with customizable fields."""
    return EnrollmentIssue(
        experiment_id=experiment_id,
        current_rate=current_rate,
        expected_rate=expected_rate,
        days_below_threshold=days_below,
        severity=severity,
    )


def create_monitor_state(
    experiments: Optional[List[ExperimentSummary]] = None,
    srm_issues: Optional[List[SRMIssue]] = None,
    enrollment_issues: Optional[List[EnrollmentIssue]] = None,
    status: MonitorStatus = "pending",
    **kwargs,
) -> ExperimentMonitorState:
    """Helper to create monitor state with defaults and overrides."""
    return ExperimentMonitorState(
        query=kwargs.get("query", "Test query"),
        experiment_ids=kwargs.get("experiment_ids", []),
        check_all_active=kwargs.get("check_all_active", True),
        srm_threshold=kwargs.get("srm_threshold", 0.001),
        enrollment_threshold=kwargs.get("enrollment_threshold", 5.0),
        fidelity_threshold=kwargs.get("fidelity_threshold", 0.2),
        check_interim=kwargs.get("check_interim", True),
        experiments=experiments or [],
        srm_issues=srm_issues or [],
        enrollment_issues=enrollment_issues or [],
        fidelity_issues=kwargs.get("fidelity_issues", []),
        interim_triggers=kwargs.get("interim_triggers", []),
        alerts=kwargs.get("alerts", []),
        monitor_summary=kwargs.get("monitor_summary", ""),
        recommended_actions=kwargs.get("recommended_actions", []),
        check_latency_ms=kwargs.get("check_latency_ms", 0),
        experiments_checked=kwargs.get("experiments_checked", 0),
        errors=kwargs.get("errors", []),
        warnings=kwargs.get("warnings", []),
        status=status,
    )


# ============================================================================
# OFFLINE GUARD — recipient self-emission isolation
# ============================================================================


@pytest.fixture(autouse=True)
def _no_live_supabase_in_recipient_tests(monkeypatch):
    """Isolate recipient self-emission tests from the live Supabase signals table.

    ``emit_recipient_signal`` (src/agents/feedback_learner/recipient_emit.py)
    resolves a Supabase client from the default factory when none is passed.
    Without this guard, unit tests that call the recipient's generating nodes
    would write synthetic rows to ``dspy_agent_training_signals`` in the live
    DB, polluting the optimizer's real training corpus.

    Force the factory-level client to ``None`` so the best-effort persist path
    no-ops.  The import of ``get_supabase_client`` happens INSIDE the async
    function (not at module level), so patching the factory module attribute is
    sufficient — no second patch on ``recipient_emit`` is needed.

    Tests that supply an explicit ``client=`` kwarg bypass the factory entirely
    and are unaffected by this patch.
    """
    monkeypatch.setattr(
        "src.memory.services.factories.get_supabase_client",
        lambda: None,
    )

    # The experiment-monitor nodes resolve their client via the ASYNC factory
    # (get_async_supabase_client). Neutralize it too so monitor unit tests that
    # do not inject/patch a client stay OFFLINE -> the nodes fail closed (empty +
    # recorded error) rather than touching the live DB or fabricating mock data.
    async def _async_none():
        return None

    monkeypatch.setattr(
        "src.memory.services.factories.get_async_supabase_client",
        _async_none,
    )
