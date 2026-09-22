"""Lane C planted-truth end to end (spec 2026-09-22 §3C.3): the synthetic CSU
escalation cohort -- remibrutinib vs competitor, planted confounders -- run
through the WHOLE production path the real cohort will take, in CI, with a
fixed seed and no data files:

    generator (src.ml.synthetic.generators.csu_escalation_causal)
      -> the table's rows through the production loader
         (registry allowlist, is_synthetic provenance filter, numeric coercion,
          one-hot with the __missing__ level)
      -> the submit endpoint (dataset default: discovery OFF, the curated
         common-cause DAG)
      -> the REAL causal_impact graph (graph_builder -> Auto estimator
         selection -> estimation -> refutation -> interpretation)
      -> the response the client polls.

The estimate must recover the planted RD-scale ATE within the generator's
tolerance. The planted confounding is real: the naive diff-in-means sits
outside the tolerance (asserted), so a run that loses the adjustment set --
or the one-hot payer dummies the planted backdoor runs through -- fails here.

Run shape in CI: the planted confounders plus a handful of the other baseline
features (k ~ 13 resolved columns). Measured on main (evidence dir
docs/demos/results/2026-09-22_lane_c_remibrutinib_prewiring/, D2): the whole
graph completes in 83-85 s at k = 5 (n = 2,000 / 3,000; tournament ~37 s,
refutation ~38 s) and takes 671 s at the full width (k = 73) because the
refutation node's DoWhy reconstruction burns identify_effect's
100,000-iteration cap once the adjustment set has ~17+ members (Lane A's fix
is on its unmerged branch) -- the run then ends ``failed`` with the refutation
unrun. So the FULL 64-feature default is exercised by the same test under
``E2I_CSU_PLANTED_FULL_WIDTH=1`` (manual / after Lane A merges), not on every
CI run. The marker is 300 s: the unit lane's stall window is 600 s and
tests/unit/test_tests_meta/test_session_stall_watchdog_1655.py requires
window >= 2x the largest literal marker it collects.

What is asserted: the POINT estimate recovers the planted RD-scale ATE as a
CAPABILITY, not a column-presence proxy (verifier MED-A, 2026-09-22): the
error must be inside ``RECOVERY_TOLERANCE`` (0.06) AND below half the naive
contrast's error (``PlantedTruth.is_recovery_convincing``). The spec's
sidecar tolerance (0.10) is still asserted but is NOT the discriminating
check -- measured through this graph (evidence README, D2b) the full
adjustment lands at 0.406 (error 0.042) while dropping ``payer_category``
from the adjustment set lands at 0.436 (error 0.072 -- inside 0.10, outside
0.06), and a plain OLS without payer errs 0.070; the naive contrast errs
0.134. 0.06 is the midpoint of the two measured outcomes (0.057 rounded up),
so both sides keep a >= 0.012 margin against a measured run-to-run spread of
~0.001. The 95% CI is NOT asserted to cover the truth: measured, the
tournament's winner (LinearDML) sits +0.04 above the truth with a CI that
excludes it (D2), while a correctly specified g-computation on the same
frame lands within 0.011 -- an estimator-side property of a linear final
stage on a step-function CATE, not a generator defect, recorded as a
follow-up rather than loosened here.

The synthetic backing is read ONLY under the planted-truth opt-in
(``E2I_CSU_PLANTED_TRUTH_RUN``); the deployment-wide
``E2I_INCLUDE_SYNTHETIC`` (set on the deployed e2i_api) does not unlock it
(verifier MED-B) -- the registry test pins that, this test sets the opt-in
alone.

Write-free: unit tests run with dead Supabase credentials (the refutation
persistence fails closed with a warning), the MLflow tracker and the job store
are replaced by in-memory recorders, and nothing touches Redis.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from src.api.routes.causal import agent as causal_routes
from src.api.routes.causal.datasets import _CAUSAL_DATASET_SPECS, PLANTED_TRUTH_RUN_ENV
from src.api.schemas.causal import AgentCausalAnalysisRequest
from src.ml.synthetic.generators.csu_escalation_causal import (
    DATASET,
    PLANTED_CONFOUNDERS,
    PRIMARY_OUTCOME,
    TREATMENT,
    generate_csu_escalation_cohort,
)

pytestmark = pytest.mark.unit

_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"
FULL_WIDTH = os.environ.get("E2I_CSU_PLANTED_FULL_WIDTH", "").lower() in ("1", "true", "yes")
# The CI run shape: every planted confounder (age, charlson, payer) plus a few
# other baseline features the DGP draws independently (so the estimator has
# nuisance columns to ignore, including a NULL-bearing categorical).
CI_COVARIATES = [
    *PLANTED_CONFOUNDERS,
    "gdr_cd",
    "geographic_region",
    "elx_depression",
    "lis_dual_flag",
    "enrollment_duration_days",
]


class _FakeQuery:
    def __init__(self, rows, log):
        self._rows, self._log = rows, log

    def select(self, cols, *_a, **_k):
        self._log.append(("select", cols))
        return self

    def eq(self, col, value, *_a, **_k):
        self._log.append(("eq", col, value))
        if col == "is_synthetic":
            self._rows = [r for r in self._rows if bool(r.get("is_synthetic")) is bool(value)]
        else:
            self._rows = [r for r in self._rows if r.get(col) == value]
        return self

    def limit(self, n, *_a, **_k):
        self._rows = self._rows[:n]
        return self

    async def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows):
        self._rows, self.log, self.tables = rows, [], []

    def table(self, name, *_a, **_k):
        self.tables.append(name)
        return _FakeQuery(list(self._rows), self.log)


class _MemStore:
    def __init__(self) -> None:
        self.d: dict = {}

    async def get(self, key):
        return self.d.get(key)

    async def set(self, key, value):
        self.d[key] = value


class _RecordingTracker:
    """Stands in for CausalImpactMLflowTracker: records, never writes."""

    calls: dict = {}

    def __init__(self, *args, **kwargs):
        type(self).calls["created"] = True

    def start_analysis_run(self, **kwargs):
        cls = type(self)

        @asynccontextmanager
        async def _cm():
            cls.calls["start_kwargs"] = kwargs
            yield object()

        return _cm()

    async def log_analysis_result(self, output, state=None, **kwargs):
        type(self).calls["logged_output"] = output


class _BG:
    def __init__(self):
        self.scheduled: list = []

    def add_task(self, fn, *args):
        self.scheduled.append((fn, args))


def _rows(frame):
    frame = frame.astype(object).where(frame.notna(), None)
    return frame.to_dict(orient="records")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_planted_ate_is_recovered_through_the_production_path(monkeypatch, tmp_path):
    # The backing IS synthetic: the planted-truth opt-in is the ONLY switch
    # that reads it; the deployment-wide showcase flag is left unset so the
    # run proves the opt-in alone carries the whole path.
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.setenv(PLANTED_TRUTH_RUN_ENV, "1")
    # Write-free belt and braces on top of the unit tree's dead-Supabase pin:
    # the tracker seam below is replaced, but any stray mlflow call lands in a
    # throwaway file store, and Redis (prod on this box) is a dead port.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    frame, truths = generate_csu_escalation_cohort()
    truth = truths[PRIMARY_OUTCOME]
    # Teeth: adjustment is required to land inside the tolerance.
    assert not truth.is_estimate_valid(truth.naive_diff)

    client = _FakeClient(_rows(frame))
    store = _MemStore()
    _RecordingTracker.calls = {}
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", store)
    monkeypatch.setattr(
        "src.agents.causal_impact.mlflow_tracker.CausalImpactMLflowTracker", _RecordingTracker
    )

    request_kwargs: dict = {
        "treatment_var": TREATMENT,
        "outcome_var": PRIMARY_OUTCOME,
        "dataset": DATASET,
        "limit": 20000,
    }
    if not FULL_WIDTH:
        request_kwargs["covariates"] = CI_COVARIATES
    req = AgentCausalAnalysisRequest(**request_kwargs)  # auto_discover NOT set
    bg = _BG()
    pending = await causal_routes.run_causal_agent_analysis(req, bg, user={"role": "analyst"})

    # Submit: the production loader read the table's rows and resolved the
    # covariates (the payer one-hot dummies and the __missing__ region level).
    assert pending.status == "pending" and pending.n_rows == len(frame)
    assert client.tables == [DATASET]
    assert not any(e[0] == "eq" and e[1] == "is_synthetic" for e in client.log)
    (fn, args) = bg.scheduled[0]
    task_request, task_frame, task_covariates = args[1], args[2], args[3]
    assert task_request.auto_discover is False  # the dataset default
    assert any("discovery" in w.lower() and DATASET in w for w in pending.warnings)
    assert "payer_category=medicare" in task_covariates
    assert "geographic_region=__missing__" in task_covariates
    assert "payer_category" not in task_covariates
    assert TREATMENT not in task_covariates and PRIMARY_OUTCOME not in task_covariates
    if FULL_WIDTH:
        assert len(task_covariates) >= len(_CAUSAL_DATASET_SPECS[DATASET]["covariate"])
    assert task_frame[TREATMENT].nunique() == 2

    # The REAL graph on the loaded frame (the client's poll target).
    await fn(*args)
    result = store.d[pending.analysis_id]

    assert result.status in ("completed", "needs_review"), result.warnings
    assert result.dag_source == "domain_knowledge"  # discovery off -> curated DAG
    assert result.ate is not None
    report = {**truth.recovery_report(result.ate), "estimator": result.selected_estimator}
    assert truth.is_estimate_valid(result.ate), report  # the spec's sidecar tolerance
    # The capability: inside RECOVERY_TOLERANCE AND well below the naive error
    # -- losing the payer backdoor (error 0.072 measured) fails here.
    assert truth.is_recovery_convincing(result.ate), report
    # The agent's own naive-vs-adjusted surfacing agrees the backdoor was real.
    if result.naive_ate is not None:
        assert abs(result.naive_ate - truth.naive_diff) < 0.03
        assert abs(result.naive_ate - truth.true_ate) > abs(result.ate - truth.true_ate)
    assert result.n_rows == len(frame)
    assert result.data_source == "synthetic"
    # Observability went through the tracker seam (recorded, never written).
    assert _RecordingTracker.calls.get("created") is True
    assert _RecordingTracker.calls["start_kwargs"]["treatment_var"] == TREATMENT
