#!/usr/bin/env python3
"""Lane C planted-truth verification against the LIVE ``csu_escalation_causal``
table (2026-09-23, after the owner-GO load): the CI E2E
``tests/unit/test_api/test_causal_csu_escalation_planted_truth.py`` re-run
IN-PROCESS with the real ``.env`` -- the production loader reads the deployed
table through the dataset-level provenance guard, the REAL causal_impact graph
estimates, and the estimate must recover the planted ATE under the PR's
tolerance rule (``PlantedTruth.is_recovery_convincing``: error < 0.06 AND < half
the naive contrast's error), plus the spec's 0.10 sidecar tolerance.

Three probes, in this order, all against the live table:

1. REAL MODE (the production condition, ``datasets.PLANTED_TRUTH_RUN`` False,
   the deployment flag exactly as ``.env`` carries it): the loader must answer
   503 "no usable rows" -- the planted rows are NOT served as real.
2. PLANTED-TRUTH MODE (the module seam flipped in THIS process only, the way
   the E2E's monkeypatch does; no environment variable can): the loader reads
   ONLY ``is_synthetic = true`` rows, n must equal the live synthetic count, the
   graph runs, recovery is asserted.
3. REAL MODE again after the seam is closed: 503 once more (the seam left no
   residue).

Optionally (``--api-probe``) the deployed API itself is asked the READ-ONLY
``GET /causal/brands?dataset=csu_escalation_causal`` with an admin token: the
brand dropdown reads the table through the same dataset-level predicate, so
the container must answer ``[]`` although the table now holds three brand
labels (the faithful environment for probe 1; a POST would enqueue a real job
if the guard were absent -- codex r1 HIGH).

WRITE-FREE by construction (this is a certification read, the owner's load is
the only prod write): the refutation node's persistence + expert-review
repositories are replaced with the "unavailable" degrade (no
causal_validations / causal_paths / expert_reviews rows), the node's
INDEPENDENT Feedback-Learner writer ``log_validation_outcome_with_status``
(validation_outcomes; codex r1 HIGH) is a recorder that reports a non-durable
write, the MLflow tracker is a recorder and MLFLOW_TRACKING_URI points at a
scratch file store, the job store is in-memory and REDIS_URL is a dead port,
OPIK_ENABLED=false, discovery is OFF (the dataset default -> no
discovered-DAG persist), the DSPy signal router entry point is a recorder
(codex r1 MED), and the audit-chain initializer holds no store. The only
network reads are the PostgREST reads of the table (and the structural-prior
lookup of expert_reviews, read-only). Every recorder's call count is written
to the raw record so "nothing was written" is a measurement, not a claim.

Usage (from the lane worktree; the main checkout's .env is found up the tree):
  $PY docs/demos/results/2026-09-23_lane_c_planted_truth_live/run_e2e_live.py [--full-width] [--api-probe]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import resource
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
MAIN_CHECKOUT = Path("/home/enunez/Projects/e2i_causal_analytics")
SCRATCH = Path(
    "/tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/"
    "d16767d4-243f-4a69-a1c1-36c8599d7ba9/scratchpad"
)

# --- process environment BEFORE any ``src`` import -------------------------
# The real .env (prod Supabase URL + service-role key, the deployment's
# E2I_INCLUDE_SYNTHETIC) -- then the write-free overrides on top.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(MAIN_CHECKOUT / ".env")
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
os.environ["OPIK_ENABLED"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = f"file://{SCRATCH / 'cload_live_mlruns'}"
for _paid in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ[_paid] = ""  # no paid call can be made from this process

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpcore", "urllib3", "opik", "sentry_sdk"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(REPO_ROOT))
import src  # noqa: E402

assert Path(src.__file__).resolve().is_relative_to(REPO_ROOT), (src.__file__, REPO_ROOT)

from fastapi import HTTPException  # noqa: E402

from scripts.load_csu_escalation_cohort import (  # noqa: E402
    SPEC,
    TABLE,
    _client,
    fetch_live_provenance_counts,
    fetch_live_split,
    guard_problem,
)
from src.agents import tier2_signal_router as signal_router_mod  # noqa: E402
from src.agents.causal_impact.nodes import refutation as refutation_mod  # noqa: E402
from src.api.routes.causal import agent as causal_routes  # noqa: E402
from src.api.routes.causal import datasets as datasets_mod  # noqa: E402
from src.api.routes.causal.datasets import (  # noqa: E402
    _CAUSAL_SYNTHETIC_BACKED,
    deployment_includes_synthetic,
    serves_synthetic_rows,
)
from src.api.routes.causal.loaders import _load_agent_estimation_frame  # noqa: E402
from src.api.schemas.causal import AgentCausalAnalysisRequest  # noqa: E402
from src.causal_engine.validation_outcome_store import StoreResult  # noqa: E402
from src.ml.synthetic.generators.csu_escalation_causal import (  # noqa: E402
    DATASET,
    DEFAULT_N,
    DEFAULT_SEED,
    PLANTED_CONFOUNDERS,
    PRIMARY_OUTCOME,
    TREATMENT,
    generate_csu_escalation_cohort,
)

assert DATASET == TABLE == SPEC.dataset

# The CI run shape (mirrors the E2E): the planted confounders plus a few
# independent baselines, k ~ 13 resolved columns.
CI_COVARIATES = [
    *PLANTED_CONFOUNDERS,
    "gdr_cd",
    "geographic_region",
    "elx_depression",
    "lis_dual_flag",
    "enrollment_duration_days",
]
GROUND_TRUTH_SIDECAR = (
    MAIN_CHECKOUT / "data/rwd/synthetic_CSU/csu_escalation_causal/ground_truth.json"
)


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


def _sh(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def container_facts() -> dict:
    env = _sh(
        ["docker", "inspect", "e2i_api", "--format", "{{range .Config.Env}}{{println .}}{{end}}"]
    )
    flag = next((l for l in env.splitlines() if l.startswith("E2I_INCLUDE_SYNTHETIC=")), "absent")
    image = _sh(["docker", "inspect", "e2i_api", "--format", "{{.Config.Image}}"])
    # The image commit alone (the ``<repo>:<40-hex>`` tag form trips the
    # repo's secret scanner in evidence files).
    repo, _, image_commit = image.rpartition(":")
    return {
        "image_repo": repo,
        "image_commit": image_commit,
        "image_id": _sh(["docker", "inspect", "e2i_api", "--format", "{{.Image}}"]),
        "started_at": _sh(["docker", "inspect", "e2i_api", "--format", "{{.State.StartedAt}}"]),
        "E2I_INCLUDE_SYNTHETIC": flag,
        "host_git_head": _sh(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "host_git_branch": _sh(["git", "-C", str(REPO_ROOT), "branch", "--show-current"]),
        "main_checkout_head": _sh(["git", "-C", str(MAIN_CHECKOUT), "rev-parse", "HEAD"]),
    }


RECORDED: dict = {"validation_outcomes": 0, "signals": 0, "persistence_repos": 0, "gate": 0}


def install_write_free_seams() -> _MemStore:
    async def _no_repos():
        RECORDED["persistence_repos"] += 1
        return None, None

    async def _no_gate():
        RECORDED["gate"] += 1
        return None

    async def _record_outcome(outcome):
        RECORDED["validation_outcomes"] += 1
        return StoreResult(
            outcome_id="recorded-not-written", persisted=False, degraded=True, backend="recorder"
        )

    async def _record_signal(signal):
        RECORDED["signals"] += 1
        return False

    refutation_mod._build_persistence_repos = _no_repos
    refutation_mod._build_expert_review_gate = _no_gate
    refutation_mod.log_validation_outcome_with_status = _record_outcome
    signal_router_mod.route_causal_impact_signal = _record_signal
    import src.agents.causal_impact.mlflow_tracker as tracker_mod

    tracker_mod.CausalImpactMLflowTracker = _RecordingTracker
    store = _MemStore()
    causal_routes._agent_analysis_store = store
    return store


async def probe_real_mode(label: str) -> dict:
    """The production condition: the loader must find NO usable rows."""
    datasets_mod.PLANTED_TRUTH_RUN = False
    facts = {
        "label": label,
        "PLANTED_TRUTH_RUN": datasets_mod.PLANTED_TRUTH_RUN,
        "deployment_includes_synthetic": deployment_includes_synthetic(),
        "serves_synthetic_rows": serves_synthetic_rows(DATASET),
        "in_synthetic_backed": DATASET in _CAUSAL_SYNTHETIC_BACKED,
    }
    try:
        df, _cols = await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var=PRIMARY_OUTCOME,
            covariates=list(CI_COVARIATES),
            limit=20000,
            brand=None,
        )
        facts["outcome"] = f"SERVED {int(df.shape[0])} rows"
        facts["ok"] = False
    except HTTPException as exc:
        facts["outcome"] = f"HTTPException {exc.status_code}: {exc.detail}"
        facts["ok"] = exc.status_code == 503 and "No usable estimation rows" in str(exc.detail)
    print(f"[{label}] {facts}")
    return facts


async def run_planted_truth(full_width: bool, store: _MemStore, live_n_synthetic: int) -> dict:
    datasets_mod.PLANTED_TRUTH_RUN = True
    assert serves_synthetic_rows(DATASET) is True
    frame, truths = generate_csu_escalation_cohort(n=DEFAULT_N, seed=DEFAULT_SEED)
    truth = truths[PRIMARY_OUTCOME]
    assert not truth.is_estimate_valid(truth.naive_diff)  # teeth: adjustment is required
    sidecar = None
    if GROUND_TRUTH_SIDECAR.exists():
        sidecar = json.loads(GROUND_TRUTH_SIDECAR.read_text())[PRIMARY_OUTCOME]
        assert sidecar["true_ate"] == truth.true_ate, (sidecar["true_ate"], truth.true_ate)
        assert sidecar["n_samples"] == len(frame) == DEFAULT_N

    _RecordingTracker.calls = {}
    request_kwargs: dict = {
        "treatment_var": TREATMENT,
        "outcome_var": PRIMARY_OUTCOME,
        "dataset": DATASET,
        "limit": 20000,
    }
    if not full_width:
        request_kwargs["covariates"] = list(CI_COVARIATES)
    req = AgentCausalAnalysisRequest(**request_kwargs)  # auto_discover NOT set -> dataset default
    bg = _BG()
    t0 = time.monotonic()
    pending = await causal_routes.run_causal_agent_analysis(req, bg, user={"role": "analyst"})
    submit_s = time.monotonic() - t0
    (fn, args) = bg.scheduled[0]
    task_request, task_frame, task_covariates = args[1], args[2], args[3]
    submit = {
        "status": pending.status,
        "n_rows": pending.n_rows,
        "data_source": pending.data_source,
        "auto_discover": task_request.auto_discover,
        "k_resolved": len(task_covariates),
        "resolved_covariates": list(task_covariates),
        "treatment_nunique": int(task_frame[TREATMENT].nunique()),
        "submit_s": round(submit_s, 2),
        "warnings": list(pending.warnings),
    }
    print(f"[planted] submit: {submit}")
    checks = {
        "n_rows_equals_live_synthetic_count": pending.n_rows == live_n_synthetic == DEFAULT_N,
        "data_source_synthetic": pending.data_source == "synthetic",
        "discovery_off_by_default": task_request.auto_discover is False,
        "payer_one_hot_present": "payer_category=medicare" in task_covariates,
        "region_missing_level_present": "geographic_region=__missing__" in task_covariates,
        "two_arms": int(task_frame[TREATMENT].nunique()) == 2,
    }

    t1 = time.monotonic()
    await fn(*args)
    graph_s = time.monotonic() - t1
    result = store.d[pending.analysis_id]
    report = {**truth.recovery_report(result.ate), "estimator": result.selected_estimator}
    checks.update(
        {
            "status_completed_or_needs_review": result.status in ("completed", "needs_review"),
            "dag_source_domain_knowledge": result.dag_source == "domain_knowledge",
            "ate_present": result.ate is not None,
            "sidecar_tolerance_0_10": bool(
                result.ate is not None and truth.is_estimate_valid(result.ate)
            ),
            "recovery_convincing": bool(
                result.ate is not None and truth.is_recovery_convincing(result.ate)
            ),
            "result_n_rows": result.n_rows == DEFAULT_N,
            "result_data_source_synthetic": result.data_source == "synthetic",
            "tracker_recorded_not_written": _RecordingTracker.calls.get("created") is True,
        }
    )
    if result.naive_ate is not None:
        checks["naive_ate_matches_planted_naive"] = abs(result.naive_ate - truth.naive_diff) < 0.03
        checks["adjusted_closer_than_naive"] = abs(result.naive_ate - truth.true_ate) > abs(
            result.ate - truth.true_ate
        )
    datasets_mod.PLANTED_TRUTH_RUN = False
    payload = result.model_dump(mode="json")
    return {
        "submit": submit,
        "recovery": report,
        "planted_truth": truth.to_dict(),
        "sidecar": sidecar,
        "checks": checks,
        "graph_s": round(graph_s, 1),
        "full_width": full_width,
        "result": payload,
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def api_probe() -> dict:
    """The deployed container answering for the dataset, READ-ONLY: the brand
    dropdown applies the dataset-level predicate, so real mode must list no
    brand although the loaded table carries RHAPSIDO / XOLAIR / DUPIXENT."""
    import urllib.error
    import urllib.request

    api = os.environ.get("E2I_API_BASE", "https://eznomics.site/api")
    body = json.dumps(
        {
            "email": os.environ.get("E2I_ADMIN_EMAIL", "admin@e2i.local"),
            "password": os.environ["E2I_ADMIN_PASSWORD"],
        }
    ).encode()
    req = urllib.request.Request(
        f"{os.environ['SUPABASE_URL']}/auth/v1/token?grant_type=password",
        data=body,
        headers={"apikey": os.environ["SUPABASE_ANON_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        token = json.loads(r.read())["access_token"]
    req = urllib.request.Request(
        f"{api}/causal/brands?dataset={DATASET}",
        method="GET",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status, detail = r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            status, detail = e.code, json.loads(raw)
        except Exception:
            status, detail = e.code, raw.decode(errors="replace")[:500]
    brands = detail.get("brands") if isinstance(detail, dict) else None
    out = {"status": status, "detail": detail, "ok": status == 200 and brands == []}
    print(f"[api-probe] GET /causal/brands?dataset={DATASET}: HTTP {status} {str(detail)[:200]}")
    return out


async def main_async(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc).isoformat()
    facts = container_facts()
    print(f"container: {facts}")
    client = _client()
    live_split = fetch_live_split(client)
    provenance = fetch_live_provenance_counts(client)
    print(f"live split: {live_split}")
    print(f"live provenance: {provenance}")
    if not live_split or not provenance:
        print("ABORT: the live table is unreachable")
        return 2
    live_n_synthetic = provenance["true"]
    if live_n_synthetic == 0:
        print("ABORT: the live table holds 0 synthetic rows -- the owner load has not run")
        return 2

    guard = guard_problem()
    print(f"loader guard_problem(): {guard!r}")
    store = install_write_free_seams()
    real_before = await probe_real_mode("real-mode-before")
    planted = await run_planted_truth(args.full_width, store, live_n_synthetic)
    real_after = await probe_real_mode("real-mode-after")
    api = api_probe() if args.api_probe else None

    verdict_checks = dict(planted["checks"])
    verdict_checks["real_mode_before_503"] = bool(real_before["ok"])
    verdict_checks["real_mode_after_503"] = bool(real_after["ok"])
    verdict_checks["loader_guard_clear"] = guard is None
    verdict_checks["outcome_writer_recorded_not_written"] = RECORDED["validation_outcomes"] >= 1
    verdict_checks["persistence_repos_degraded"] = RECORDED["persistence_repos"] >= 1
    if api is not None:
        verdict_checks["deployed_api_brands_empty"] = bool(api["ok"])
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"
    record = {
        "verdict": verdict,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "container": facts,
        "live_split": live_split,
        "live_provenance": provenance,
        "real_mode_before": real_before,
        "planted": planted,
        "real_mode_after": real_after,
        "api_probe": api,
        "loader_guard_problem": guard,
        "recorded_write_seams": dict(RECORDED),
        "checks": verdict_checks,
    }
    tag = "full_width" if args.full_width else "ci_width"
    path = OUT / f"raw_live_{tag}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    failed = sorted(k for k, v in verdict_checks.items() if not v)
    print(f"VERDICT: {verdict}  failed={failed}")
    print(
        f"recovery: ate={planted['recovery']['ate']:.4f} true={planted['recovery']['true_ate']} "
        f"error={planted['recovery']['error']:.4f} (< 0.06 and < {planted['recovery']['naive_error_ceiling']:.4f}) "
        f"naive={planted['recovery']['naive_diff']} estimator={planted['recovery']['estimator']} "
        f"graph={planted['graph_s']}s n={planted['submit']['n_rows']} k={planted['submit']['k_resolved']}"
    )
    print(f"wrote {path}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--full-width", action="store_true", help="the full 64-feature covariate width"
    )
    parser.add_argument(
        "--api-probe", action="store_true", help="also POST the deployed API (expects 503)"
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
