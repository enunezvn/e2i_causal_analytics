"""Migration 165 (#2318 Lane 4): the model-activation ledger and its self-checking RPCs.

An activation makes a retrained ``candidate`` the served row for its name and archives the row
it retrains (the predecessor). ``ml_model_activations`` records every prior value a rollback
needs; ``activate_model_candidate`` / ``rollback_model_activation`` switch the registry roles
and the deployment in one transaction, re-checking the gate report, every identity and the
single-canonical-row invariant themselves (the CLI is not trusted). A registry trigger keeps the
weekly re-registration (``register_model_row``'s upsert) from changing an activated row's
stage, champion flag or ``registered_at`` (OD-5), and refuses any OTHER row of the name that
would become canonical or champion while the activation is live.

The pure tests run in CI. The rest is real Postgres, opt-in (``E2I_DB_INTEGRATION=1``, docker):
clones of prod's schema as it is now plus the pending migrations (``clone_db``), a throwaway
PostgREST of prod's own image for the transports production uses, and the upgrade path on a
copy of prod before 165 (``base_clone_db``). It lives in the learning_loop suite so the deploy's
real-DB gate (``scripts/deploy/realdb_suite_gate.sh``) runs it before the migration ships.
Nothing here writes to ``supabase-db``.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest

from tests.unit.test_database.learning_loop import _pg

pytestmark = pytest.mark.timeout(300)

KEY = "165_model_activation_ledger.sql"
M165 = _pg.REPO_ROOT / "database" / "migrations" / KEY
R165 = _pg.REPO_ROOT / "database" / "migrations" / f"rollback_{KEY}"

#: OD-2, frozen. The SQL literal, Lane 5's ``GateConfig`` and this dict must agree.
FROZEN_CONFIG: Dict[str, Any] = {
    "auc_margin": 0.010,
    "alpha": 0.05,
    "brier_margin": 0.005,
    "slope_band": [0.8, 1.25],
    "bootstrap_b": 2000,
    "seed": 0,
    "min_class_n": 100,
    "min_usable_bootstrap_frac": 0.90,
}

LIVE = ("prepared", "serving_switched", "active", "aborting", "rolling_back")
SHA_C = "c" * 64
SHA_P = "a" * 64
ROWS_SHA = "ab" * 32


def _code(path) -> str:
    return "\n".join(re.sub(r"--.*$", "", line) for line in path.read_text().splitlines())


def _sql_config_literal() -> Dict[str, Any]:
    body = _code(M165)
    m = re.search(
        r"FUNCTION\s+public\.activation_gate_config\(\).*?\$\$\s*SELECT\s*'(\{.*?\})'::jsonb",
        body,
        re.S,
    )
    assert m, "activation_gate_config() literal not found in 165"
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# Pure (CI)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_165_is_wrapped_and_its_number_is_unique():
    assert _pg.runner_unwraps(M165.read_text()) is False
    assert not re.search(r"^\s*(BEGIN|COMMIT|ROLLBACK)\s*;", _code(M165).upper(), re.M)
    migrations = _pg.REPO_ROOT / "database" / "migrations"
    assert [p.name for p in migrations.glob("165_*.sql")] == [KEY]
    assert R165.name.startswith("rollback_")
    assert f"filename = '{KEY}'" in _code(R165)


@pytest.mark.unit
def test_the_sql_gate_constants_are_the_frozen_ones():
    assert _sql_config_literal() == FROZEN_CONFIG


@pytest.mark.unit
def test_the_sql_gate_constants_equal_lane5_gateconfig():
    """OD-2: frozen in code AND SQL. Runs once Lane 5 (holdout_gate) is on the branch."""
    from dataclasses import asdict

    holdout_gate = pytest.importorskip(
        "src.mlops.activation.holdout_gate", reason="#2318 Lane 5 (holdout_gate) not merged yet"
    )
    python = json.loads(json.dumps(asdict(holdout_gate.GateConfig())))
    assert _sql_config_literal() == python
    assert holdout_gate.GATE_KIND == "deterministic_acceptance_rule"
    assert list(holdout_gate.OOS_EVAL_SPLITS) == ["test", "holdout"]
    assert tuple(holdout_gate.HCP_PATHOLOGY_SLOPE_RANGE) == (0.5, 2.0)


def _sql_z_literal() -> float:
    m = re.search(r"([0-9]+\.[0-9]+)::numeric AS z\b", _code(M165))
    assert m, "the one-sided z literal (... ::numeric AS z) not found in 165"
    return float(m.group(1))


@pytest.mark.unit
def test_the_sql_lower_bound_z_is_lane5s_norm_ppf():
    """The SQL re-derives auc_lower_bound = auc_delta - z * se_delta with z hard-coded (the
    config jsonb must stay exactly GateConfig); z must be Lane 5's norm.ppf(1 - alpha)."""
    from scipy.stats import norm

    holdout_gate = pytest.importorskip(
        "src.mlops.activation.holdout_gate", reason="#2318 Lane 5 (holdout_gate) not merged yet"
    )
    assert _sql_z_literal() == float(norm.ppf(1.0 - holdout_gate.GateConfig().alpha))
    assert _sql_z_literal() == float(norm.ppf(1.0 - FROZEN_CONFIG["alpha"]))


@pytest.mark.unit
def test_the_rpc_authority_is_not_a_caller_settable_guc():
    """codex r1 MEDIUM: any role can set_config() a custom GUC, so none may authorise."""
    assert "e2i.activation_rpc" not in _code(M165)
    assert "current_setting(" not in _code(M165)


@pytest.mark.unit
def test_the_role_guard_fires_before_the_single_champion_trigger():
    """BEFORE row triggers fire in name order. The guard must restore an activated row's flag
    before ``tr_single_champion`` acts on it, so its name must sort first."""
    assert "tr_ml_model_registry_activation_role_guard" < "tr_single_champion"
    assert "CREATE TRIGGER tr_ml_model_registry_activation_role_guard" in _code(M165)


# ---------------------------------------------------------------------------
# Real Postgres helpers
# ---------------------------------------------------------------------------


def _sql(
    db: _pg.PgConn, sql: str, params: Any = None, *, role: Optional[str] = None
) -> List[tuple]:
    """One autocommit statement, as ``postgres`` (the table owner) or ``SET ROLE role``."""
    with db.connect(autocommit=True) as c:
        if role:
            c.execute(f'set role "{role}"')
        cur = c.execute(sql, params)
        return cur.fetchall() if cur.description else []


def _one(db: _pg.PgConn, sql: str, params: Any = None, **kw: Any) -> Any:
    (row,) = _sql(db, sql, params, **kw)
    return row[0] if len(row) == 1 else row


def report(
    name: str,
    cand_sha: str = SHA_C,
    served_sha: str = SHA_P,
    *,
    stage: Optional[str] = None,
    **over: Any,
) -> Dict[str, Any]:
    """A passing gate report in Lane 5's ``evaluate_gate`` shape (probe numbers, F22).

    ``stage`` is the target stage (default: production for hcp_adoption names, else
    staging); the pathology fields ride along for an hcp name or a production target."""
    if stage is None:
        stage = "production" if name.startswith("hcp_adoption_") else "staging"
    n, n_pos = 1766, 609
    r: Dict[str, Any] = {
        "kind": "deterministic_acceptance_rule",
        "n": n,
        "n_pos": n_pos,
        "prevalence": n_pos / n,
        "auc_served": 0.8509,
        "auc_candidate": 0.8517,
        "auc_delta": 0.0008,
        "se_delta": 0.0012,
        "auc_lower_bound": 0.0008 - 1.6448536269514722 * 0.0012,
        "delong_corr": 0.991,
        "brier_served": 0.1601,
        "brier_candidate": 0.1598,
        "brier_delta_upper": 0.0011,
        "bootstrap_auc_delta_p05": -0.0011,
        "bootstrap_usable": 2000,
        "bootstrap_skipped_single_class": 0,
        "pr_auc_served": 0.71,
        "pr_auc_candidate": 0.712,
        "calibration_slope": 1.02,
        "calibration_intercept": 0.01,
        "served_bundle_sha256": served_sha,
        "candidate_bundle_sha256": cand_sha,
        "snapshot": {
            "splits": ["test", "holdout"],
            "n": n,
            "n_pos": n_pos,
            "rows_sha256": ROWS_SHA,
            "loaded_at": "2026-09-30T12:00:00+00:00",
            "key": "patient_id",
            "label_column": "initiated",
            "keep_columns": ["age"],
            "cohort": "initiation",
            "brand": "Kisqali",
        },
        "model_name": name,
        "served_stage": stage,
    }
    if name.startswith("hcp_adoption_") or stage == "production":
        r.update(
            hcp_pathology_passed=True,
            hcp_pathology_slope_ok=True,
            hcp_pathology_brier_ok=True,
            hcp_pathology_reasons=[],
        )
    r["failed_checks"] = []
    r["passed"] = True
    r["config"] = copy.deepcopy(FROZEN_CONFIG)
    r.update(over)
    return r


class Slot:
    """One served name: predecessor P, candidate C (a retrain of P), C's deployment D."""

    def __init__(
        self,
        db: _pg.PgConn,
        *,
        hcp: bool = False,
        name: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        self.db = db
        tag = uuid.uuid4().hex[:8]
        self.name = name or (
            f"hcp_adoption_l4_{tag}_goldstd_lr_v1" if hcp else f"initiation_l4_{tag}_goldstd_lr_v1"
        )
        self.stage = stage or ("production" if hcp else "staging")
        self.champion = hcp
        self.exp = str(uuid.uuid4())
        self.p, self.c, self.d = (str(uuid.uuid4()) for _ in range(3))
        self.p_artifact = f"/app/data/ml_artifacts/l4/{self.name}.pkl"
        self.c_bundle = (
            f"/app/data/ml_artifacts/serving_versions/{self.name}/{self.c}.cccccccccccc.bundle.pkl"
        )
        self.p_bundle = (
            f"/app/data/ml_artifacts/serving_versions/{self.name}/{self.p}.aaaaaaaaaaaa.bundle.pkl"
        )
        _sql(
            db,
            "insert into ml_experiments (id, experiment_name, prediction_target) values (%s, %s, %s)",
            (self.exp, f"{self.name}_exp", "l4_target"),
        )
        self.add_row(
            self.p, "1.0", self.stage, champion=self.champion, artifact=self.p_artifact,
            registered_at="2026-09-28 03:03:46+00",
        )  # fmt: skip
        self.c_version = "1.0_retrained_20260930_0100_l4"
        self.add_row(
            self.c, self.c_version, "candidate", retrain_of=self.p,
            mlflow_version=8, registered_at="2026-09-30 01:00:00+00",
        )  # fmt: skip
        self.add_deployment(self.d, self.c)

    def add_row(
        self,
        rid: str,
        version: str,
        stage: Optional[str],
        *,
        champion: bool = False,
        artifact: Optional[str] = None,
        retrain_of: Optional[str] = None,
        mlflow_version: Optional[int] = None,
        registered_at: str = "2026-09-29 00:00:00+00",
        synthetic: bool = False,
        name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        _sql(
            self.db,
            "insert into ml_model_registry (id, experiment_id, model_name, model_version, algorithm,"
            " stage, is_champion, is_synthetic, artifact_path, retrain_of_id, mlflow_model_version,"
            " registered_at, training_provenance) values (%s, %s, %s, %s,"
            " 'logistic_regression_calibrated', %s, %s, %s, %s, %s, %s, %s, 'synthetic_gold')",
            (rid, self.exp, name or self.name, version, stage, champion, synthetic, artifact,
             retrain_of, mlflow_version, registered_at),
            role=role,
        )  # fmt: skip

    def add_deployment(self, did: str, registry_id: str) -> None:
        _sql(
            self.db,
            "insert into ml_deployments (id, model_registry_id, deployment_name, environment, status)"
            " values (%s, %s, %s, 'candidate', 'registered')",
            (did, registry_id, f"{self.name}_candidate"),
        )

    def ledger_row(self, **over: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "model_name": self.name,
            "candidate_registry_id": self.c,
            "predecessor_registry_id": self.p,
            "served_stage": self.stage,
            "predecessor_prior_stage": self.stage,
            "predecessor_prior_is_champion": self.champion,
            "predecessor_prior_artifact_path": self.p_artifact,
            "candidate_deployment_id": self.d,
            "candidate_mlflow_model_version": 8,
            "prior_mlflow_served_version": None,
            "candidate_bundle_sha256": SHA_C,
            "candidate_bundle_path": self.c_bundle,
            "predecessor_bundle_sha256": SHA_P,
            "predecessor_bundle_path": self.p_bundle,
            "gate_report": report(self.name, stage=self.stage),
            "approved_by": "owner",
        }
        row.update(over)
        return row

    def insert_sql(self, **over: Any) -> tuple:
        row = self.ledger_row(**over)
        cols = list(row)
        vals = [json.dumps(v) if k == "gate_report" else v for k, v in row.items()]
        placeholders = ", ".join("%s::jsonb" if k == "gate_report" else "%s" for k in cols)
        sql = (
            f"insert into ml_model_activations ({', '.join(cols)}) values ({placeholders}) "
            "returning id"
        )
        return sql, vals

    def insert(self, role: Optional[str] = "service_role", **over: Any) -> str:
        sql, vals = self.insert_sql(**over)
        return str(_one(self.db, sql, vals, role=role))

    def set_phase(
        self, aid: str, phase: str, role: Optional[str] = "service_role", **cols: Any
    ) -> None:
        sets = ", ".join(["phase = %s"] + [f"{k} = %s" for k in cols])
        _sql(
            self.db,
            f"update ml_model_activations set {sets} where id = %s",
            [phase, *cols.values(), aid],
            role=role,
        )

    def switched(self, **over: Any) -> str:
        """A ledger row the CLI has driven to ``serving_switched`` (candidate file live)."""
        aid = self.insert(**over)
        self.set_phase(aid, "serving_switched", serving_switched_at="now")
        return aid

    def activate(self, aid: str, role: str = "service_role") -> None:
        _sql(self.db, "select public.activate_model_candidate(%s)", (aid,), role=role)

    def rollback_db(self, aid: str, role: str = "service_role") -> None:
        _sql(self.db, "select public.rollback_model_activation(%s)", (aid,), role=role)

    def begin_rollback(self, aid: str) -> None:
        self.set_phase(aid, "rolling_back", rolled_back_by="owner", rollback_reason="l4 test")
        _sql(
            self.db,
            "update ml_model_activations set rollback_serving_at = now() where id = %s",
            (aid,),
            role="service_role",
        )

    def reg(self, rid: str) -> Dict[str, Any]:
        stage, champ, artifact, registered, pre, auc = _one(
            self.db,
            "select stage::text, is_champion, artifact_path, registered_at, "
            "preprocessing_pipeline_path, auc from ml_model_registry where id = %s",
            (rid,),
        )
        return {"stage": stage, "champion": champ, "artifact": artifact,
                "registered_at": registered, "pre": pre, "auc": auc}  # fmt: skip

    def dep(self) -> Dict[str, Any]:
        status, env, reason = _one(
            self.db,
            "select status::text, environment, rollback_reason from ml_deployments where id = %s",
            (self.d,),
        )
        return {"status": status, "env": env, "reason": reason}

    def ledger(self, aid: str) -> Dict[str, Any]:
        (row,) = _sql(
            self.db,
            "select to_jsonb(a) from ml_model_activations a where id = %s",
            (aid,),
        )
        return row[0]

    def canonical(self) -> List[str]:
        return [
            str(r[0])
            for r in _sql(
                self.db,
                "select id from ml_model_registry where model_name = %s and "
                "(stage is null or stage not in ('candidate', 'archived', 'deprecated')) order by id",
                (self.name,),
            )
        ]

    def state(self) -> Dict[str, Any]:
        return {"p": self.reg(self.p), "c": self.reg(self.c), "d": self.dep()}


@pytest.fixture
def db(clone_db) -> _pg.PgConn:
    return clone_db("m165")


def _raises(match: str) -> Any:
    import psycopg

    return pytest.raises(psycopg.Error, match=match)


def _owner_write(db: _pg.PgConn, sql: str, params: Any = None) -> None:
    """Drift the registry behind the role guard's back, as the table OWNER: the owner writes
    this transaction's token into ``ml_activation_rpc_authority`` (the table only the RPCs --
    and the owner -- can write) and removes it again before committing. Test setup only."""
    with db.connect() as c:
        c.execute("insert into ml_activation_rpc_authority (xact) values (pg_current_xact_id())")
        c.execute(sql, params)
        c.execute("delete from ml_activation_rpc_authority where xact = pg_current_xact_id()")
        c.commit()


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def test_activate_switches_roles_in_one_transaction(db):
    s = Slot(db)
    aid = s.switched()
    s.activate(aid)

    p, c, d, a = s.reg(s.p), s.reg(s.c), s.dep(), s.ledger(aid)
    assert (p["stage"], p["champion"], p["artifact"]) == ("archived", False, s.p_artifact)
    assert (c["stage"], c["champion"], c["artifact"], c["pre"]) == (
        "staging", False, s.c_bundle, s.c_bundle,
    )  # fmt: skip
    assert (d["status"], d["env"]) == ("active", "staging")
    assert a["phase"] == "active" and a["activated_at"] is not None
    assert s.canonical() == [s.c]


def test_activate_rerun_verifies_the_postcondition(db):
    s = Slot(db)
    aid = s.switched()
    s.activate(aid)
    before = s.state()
    s.activate(aid)  # a lost response re-run: no-op
    assert s.state() == before

    # Drift a role behind the guard's back (the owner, with a token), then re-run.
    _owner_write(db, "update ml_model_registry set stage = 'candidate' where id = %s", (s.c,))
    with _raises("drifted|exactly one canonical"):
        s.activate(aid)


@pytest.mark.parametrize("phase", ["prepared", "aborting", "aborted"])
def test_activate_refuses_any_phase_but_serving_switched(db, phase):
    s = Slot(db)
    aid = s.insert()
    if phase in ("aborting", "aborted"):
        s.set_phase(aid, "aborting")
    if phase == "aborted":
        s.set_phase(aid, "aborted")
    before = s.state()
    with _raises("expected serving_switched"):
        s.activate(aid)
    assert s.state() == before


def _mutate(path: str, value: Any) -> Callable[[Dict[str, Any]], None]:
    def f(r: Dict[str, Any]) -> None:
        *head, last = path.split(".")
        node = r
        for k in head:
            node = node[k]
        if value is _DROP:
            node.pop(last)
        else:
            node[last] = value

    return f


_DROP = object()

#: Each makes the report fail the SQL acceptance rule, whatever its ``passed`` says.
BAD_REPORTS = {
    "failed_gate": _mutate("passed", False),
    "passed_as_string": _mutate("passed", "true"),
    "field_omitted": _mutate("brier_delta_upper", _DROP),
    "forged_pass_auc": _mutate("auc_lower_bound", -0.02),
    "auc_bound_at_margin": _mutate("auc_lower_bound", -0.010),
    "nan_string": _mutate("auc_lower_bound", "NaN"),
    "infinity_string": _mutate("brier_delta_upper", "-Infinity"),
    "numeric_as_string": _mutate("calibration_slope", "1.0"),
    "brier_at_margin": _mutate("brier_delta_upper", 0.005),
    "slope_low": _mutate("calibration_slope", 0.79),
    "slope_high": _mutate("calibration_slope", 1.26),
    "slope_null": _mutate("calibration_slope", None),
    "margin_changed": _mutate("config.auc_margin", 0.05),
    "bootstrap_b_changed": _mutate("config.bootstrap_b", 200),
    "seed_changed": _mutate("config.seed", 1),
    "slope_band_changed": _mutate("config.slope_band", [0.5, 2.0]),
    "config_extra_key": _mutate("config.extra", 1),
    "config_key_omitted": _mutate("config.min_usable_bootstrap_frac", _DROP),
    "failed_checks_nonempty": _mutate("failed_checks", ["calibration_slope"]),
    "failed_checks_missing": _mutate("failed_checks", _DROP),
    "bootstrap_unusable": _mutate("bootstrap_usable", 1799),
    "thin_positives": _mutate("n_pos", 99),
    "wrong_kind": _mutate("kind", "confirmatory_trial"),
    "other_model": _mutate("model_name", "persistence_kisqali_goldstd_lr_v1"),
    "candidate_sha_differs": _mutate("candidate_bundle_sha256", "d" * 64),
    "served_sha_differs": _mutate("served_bundle_sha256", "e" * 64),
    # Lane 3: a store-loaded bundle reports bundle_sha256=None -- not comparable, never a match.
    "served_sha_none": _mutate("served_bundle_sha256", None),
    "snapshot_n_differs": _mutate("snapshot.n", 1765),
    "snapshot_n_pos_differs": _mutate("snapshot.n_pos", 610),
    "snapshot_splits": _mutate("snapshot.splits", ["holdout"]),
    "snapshot_hash_bad": _mutate("snapshot.rows_sha256", "not-a-hash"),
    "snapshot_missing": _mutate("snapshot", _DROP),
    "prevalence_forged": _mutate("prevalence", 0.5),
    "not_an_object": None,
    # codex r1 HIGH 1: the verdict must follow from the report's own numbers.
    "lower_bound_raised": _mutate("auc_lower_bound", 0.0),  # auc_delta / se_delta unchanged
    # auc_candidate - auc_served = 0.0008; the bound is kept consistent with the forged delta,
    # so only the auc_delta identity can refuse it.
    "auc_delta_forged": lambda r: r.update(auc_delta=0.0108,
                                           auc_lower_bound=0.0108 - 1.6448536269514722 * 0.0012),
    "usable_plus_skipped_not_b": _mutate("bootstrap_skipped_single_class", 5),
    "usable_non_integral": lambda r: r.update(bootstrap_usable=1999.5,
                                              bootstrap_skipped_single_class=0.5),
    "skipped_negative": lambda r: r.update(bootstrap_usable=2001,
                                           bootstrap_skipped_single_class=-1),
    "auc_out_of_range": lambda r: r.update(auc_served=1.5, auc_candidate=1.5008),
    "pr_auc_out_of_range": _mutate("pr_auc_candidate", 1.2),
    "brier_out_of_range": _mutate("brier_served", 1.2),
    "se_delta_negative": lambda r: r.update(se_delta=-0.0012,
                                            auc_lower_bound=0.0008 + 1.6448536269514722 * 0.0012),
    "auc_served_as_string": _mutate("auc_served", "0.8509"),
    "delong_corr_as_string": _mutate("delong_corr", "0.991"),
    "intercept_as_string": _mutate("calibration_intercept", "0.01"),
    "p05_null_with_usable_resamples": _mutate("bootstrap_auc_delta_p05", None),
    # codex r1 MEDIUM 6: the report must be for the stage the activation targets.
    "served_stage_missing": _mutate("served_stage", _DROP),
    "served_stage_other": _mutate("served_stage", "production"),
}  # fmt: skip

#: Every field Lane 5's evaluate_gate always emits; a report without any of them was not
#: produced by it. (The nullable ones must be present too: Lane 5 writes an explicit null.)
LANE5_REQUIRED_FIELDS = (
    "kind", "n", "n_pos", "prevalence", "auc_served", "auc_candidate", "auc_delta", "se_delta",
    "auc_lower_bound", "delong_corr", "brier_served", "brier_candidate", "brier_delta_upper",
    "bootstrap_auc_delta_p05", "bootstrap_usable", "bootstrap_skipped_single_class",
    "pr_auc_served", "pr_auc_candidate", "calibration_slope", "calibration_intercept",
    "served_bundle_sha256", "candidate_bundle_sha256", "snapshot", "model_name", "served_stage",
    "failed_checks", "passed", "config",
)  # fmt: skip


@pytest.mark.parametrize("field", LANE5_REQUIRED_FIELDS)
def test_ledger_rejects_a_report_missing_a_lane5_field(db, field):
    s = Slot(db)
    bad = report(s.name)
    bad.pop(field)
    with _raises("acceptance rule"):
        s.insert(gate_report=bad)
    assert _one(db, "select count(*) from ml_model_activations") == 0


def test_the_nullable_lane5_fields_may_be_null(db):
    """delong_corr and calibration_intercept are None in a real report when undefined."""
    s = Slot(db)
    s.insert(gate_report=report(s.name, delong_corr=None, calibration_intercept=None))


@pytest.mark.parametrize("case", sorted(BAD_REPORTS))
def test_ledger_rejects_a_report_that_fails_the_rule(db, case):
    s = Slot(db)
    if case == "not_an_object":
        bad: Any = [report(s.name)]
    else:
        bad = report(s.name)
        BAD_REPORTS[case](bad)
    with _raises("acceptance rule"):
        s.insert(gate_report=bad)
    assert _one(db, "select count(*) from ml_model_activations") == 0


def test_ledger_accepts_the_passing_report_and_the_rule_is_null_safe(db):
    s = Slot(db)
    s.insert()
    assert _one(db, "select public.activation_gate_passes(NULL, NULL, NULL, NULL, NULL)") is False
    assert _one(db, "select public.activation_gate_config()") == FROZEN_CONFIG


def test_a_candidate_identical_to_the_served_bundle_is_refused(db):
    s = Slot(db)
    with _raises("acceptance rule"):
        s.insert(predecessor_bundle_sha256=SHA_C, gate_report=report(s.name, SHA_C, SHA_C))


@pytest.mark.parametrize(
    "over, match",
    [
        ({"candidate_bundle_sha256": "C" * 64}, "check constraint|acceptance rule"),
        ({"predecessor_bundle_sha256": "a" * 63}, "check constraint|acceptance rule"),
        ({"approved_by": "  "}, "check constraint"),
        ({"candidate_bundle_path": "/app/data/ml_artifacts/shap_serving/x/y.bundle.pkl"},
         "check constraint"),
        ({"served_stage": "shadow", "predecessor_prior_stage": "shadow"}, "check constraint"),
    ],
)  # fmt: skip
def test_ledger_column_checks(db, over, match):
    s = Slot(db)
    if over.get("predecessor_prior_stage") == "shadow":  # a predecessor served at shadow
        _sql(db, "update ml_model_registry set stage = 'shadow' where id = %s", (s.p,))
        # a report for that stage, so the CHECK (not the gate's served_stage match) refuses it
        over = dict(over, gate_report=report(s.name, stage="shadow"))
    with _raises(match):
        s.insert(**over)


@pytest.mark.parametrize(
    "cols",
    [
        {"phase": "active", "activated_at": "now"},
        {"serving_switched_at": "now"},
        {"mlflow_synced_at": "now"},
        {"rollback_serving_at": "now"},
        {"rolled_back_by": "x", "rollback_reason": "y"},
        {"abort_serving_restored_at": "now"},
    ],
)
def test_a_new_ledger_row_starts_prepared_with_no_markers(db, cols):
    s = Slot(db)
    with _raises("starts in phase prepared"):
        s.insert(role=None, **cols)


def test_immutable_columns(db):
    import psycopg

    s = Slot(db)
    aid = s.insert()
    for col, value in [
        ("gate_report", json.dumps(report(s.name, cand_sha="f" * 64))),
        ("candidate_bundle_sha256", "f" * 64),
        ("approved_by", "someone else"),
        ("predecessor_prior_stage", "production"),
        ("activated_at", "2026-09-30"),
        ("rollback_db_at", "2026-09-30"),
    ]:
        cast = "::jsonb" if col == "gate_report" else ""
        sql = f"update ml_model_activations set {col} = %s{cast} where id = %s"
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _sql(db, sql, (value, aid), role="service_role")
        if col in ("activated_at", "rollback_db_at"):
            continue  # markers: the owner path is covered by the marker tests
        with _raises("immutable"):
            _sql(db, sql, (value, aid))  # the table owner: the trigger refuses
    assert s.ledger(aid)["gate_report"] == report(s.name)


def test_markers_are_write_once_and_the_phase_machine_holds(db):
    s = Slot(db)
    aid = s.switched()
    with _raises("write-once"):
        _sql(db, "update ml_model_activations set serving_switched_at = NULL where id = %s", (aid,))
    with _raises("write-once"):
        _sql(db, "update ml_model_activations set serving_switched_at = now() - interval '1 h' "
                 "where id = %s", (aid,))  # fmt: skip
    with _raises("illegal phase"):
        s.set_phase(aid, "prepared")
    with _raises("illegal phase"):
        s.set_phase(aid, "rolling_back", rolled_back_by="o", rollback_reason="r")

    t = Slot(db)
    bid = t.insert()
    with _raises("illegal phase"):
        t.set_phase(bid, "active", role=None, activated_at="now")

    s.activate(aid)
    with _raises("illegal phase"):
        s.set_phase(aid, "aborting")
    s.begin_rollback(aid)
    s.rollback_db(aid)
    with _raises("requires its progress markers"):
        s.set_phase(aid, "rolled_back", rolled_back_at="now")


@pytest.mark.parametrize(
    "setup, col",
    [
        ("prepared", "mlflow_synced_at"),
        ("serving_switched", "shap_refreshed_at"),
        ("active", "rollback_serving_at"),
        ("active", "rolled_back_at"),
        ("rolling_back_no_db", "rollback_mlflow_synced_at"),
        ("rolling_back_no_db", "rollback_shap_refreshed_at"),
    ],
)
def test_a_marker_is_only_set_in_its_own_phase(db, setup, col):
    s = Slot(db)
    if setup == "prepared":
        aid = s.insert()
    else:
        aid = s.switched()
    if setup in ("active", "rolling_back_no_db"):
        s.activate(aid)
    if setup == "rolling_back_no_db":
        s.begin_rollback(aid)
    with _raises("outside its phase"):
        _sql(db, f"update ml_model_activations set {col} = now() where id = %s", (aid,),
             role="service_role")  # fmt: skip


def _set_restored(s: Slot, aid: str, role: Optional[str] = "service_role", value: str = "now()"):
    _sql(s.db, f"update ml_model_activations set abort_serving_restored_at = {value} where id = %s",
         (aid,), role=role)  # fmt: skip


def test_an_abort_after_the_serving_switch_requires_the_served_bundle_restored(db):
    """codex r1 HIGH 3: once the candidate bundle was live, 'aborted' (which releases the
    live-name index and lets the rollback file retire the ledger) needs the predecessor bundle
    live again and verified -- abort_serving_restored_at, write-once."""
    s = Slot(db)
    aid = s.switched()
    s.set_phase(aid, "aborting")
    with _raises("requires its progress markers"):
        s.set_phase(aid, "aborted")
    _set_restored(s, aid)
    with _raises("write-once"):
        _set_restored(s, aid, role=None, value="now() - interval '1 h'")
    with _raises("write-once"):
        _set_restored(s, aid, role=None, value="NULL")
    s.set_phase(aid, "aborted")
    a = s.ledger(aid)
    assert a["phase"] == "aborted" and a["abort_serving_restored_at"] is not None


def test_an_abort_before_the_serving_switch_needs_no_restore_marker(db):
    s = Slot(db)
    aid = s.insert()
    s.set_phase(aid, "aborting")
    with _raises("outside its phase"):  # nothing was switched, so nothing to restore
        _set_restored(s, aid)
    s.set_phase(aid, "aborted")
    assert s.ledger(aid)["abort_serving_restored_at"] is None


@pytest.mark.parametrize("setup", ["prepared", "serving_switched", "active", "rolling_back"])
def test_the_abort_restore_marker_is_only_set_while_aborting(db, setup):
    s = Slot(db)
    aid = s.insert() if setup == "prepared" else s.switched()
    if setup in ("active", "rolling_back"):
        s.activate(aid)
    if setup == "rolling_back":
        s.begin_rollback(aid)
    with _raises("outside its phase"):
        _set_restored(s, aid)
    with _raises("outside its phase"):  # the owner is held to it as well
        _set_restored(s, aid, role=None)


def test_activate_refuses_a_second_served_row(db):
    for stage in ("staging", "production", "development", "shadow", None):
        s = Slot(db)
        aid = s.switched()
        # Past the ledger's own checks: the owner, with a token, adds a canonical row.
        _owner_write(
            db,
            "insert into ml_model_registry (id, experiment_id, model_name, model_version, "
            "algorithm, stage, is_synthetic) values (gen_random_uuid(), %s, %s, '0.9', 'lr', "
            "%s, %s)",
            (s.exp, s.name, stage, stage == "staging"),  # a synthetic row counts too
        )
        before = s.state()
        with _raises("exactly one canonical"):
            s.activate(aid)
        assert s.state() == before, stage
        assert s.ledger(aid)["phase"] == "serving_switched"


def test_an_unrelated_champion_in_the_experiment_leaves_one_champion(db):
    s = Slot(db)
    old = str(uuid.uuid4())
    s.add_row(old, "0.5", "archived", champion=True)  # an archived row that kept its flag
    aid = s.switched()
    s.activate(aid)
    champions = _sql(
        db, "select id from ml_model_registry where is_champion and experiment_id = %s", (s.exp,)
    )
    assert [str(r[0]) for r in champions] == [old]
    assert s.reg(s.c)["champion"] is False and s.reg(s.p)["champion"] is False


def test_the_ledger_refuses_a_predecessor_that_is_not_the_canonical_row(db):
    s = Slot(db)
    s.add_row(str(uuid.uuid4()), "0.9", "development")
    with _raises("exactly one canonical"):
        s.insert()


def test_activate_refuses_when_the_predecessor_moved(db):
    s = Slot(db)
    aid = s.switched()
    # The guard preserves the predecessor's role against an ordinary writer ...
    _sql(db, "update ml_model_registry set stage = 'production' where id = %s", (s.p,),
         role="service_role")  # fmt: skip
    assert s.reg(s.p)["stage"] == "staging"
    # ... and the RPC re-checks it against the ledger when the guard is bypassed.
    _owner_write(db, "update ml_model_registry set artifact_path = '/moved.pkl' where id = %s",
                 (s.p,))  # fmt: skip
    before = s.state()
    with _raises("changed since the gate ran"):
        s.activate(aid)
    assert s.state() == before


def test_activate_refuses_a_foreign_deployment(db):
    s = Slot(db)
    other = str(uuid.uuid4())
    s.add_deployment(other, s.p)
    with _raises("does not belong to the candidate"):
        s.insert(candidate_deployment_id=other)
    aid = s.switched()
    _sql(db, "update ml_deployments set model_registry_id = %s where id = %s", (s.p, s.d))
    before = s.state()
    with _raises("does not belong to the candidate"):
        s.activate(aid)
    assert s.state() == before


def test_activate_refuses_a_candidate_of_another_parent_or_name(db):
    s = Slot(db)
    stranger = str(uuid.uuid4())
    s.add_row(stranger, "2.0_retrained_x", "candidate", retrain_of=None, mlflow_version=8)
    with _raises("not a candidate retrain"):
        s.insert(candidate_registry_id=stranger)

    t = Slot(db)
    with _raises("not a candidate retrain|exactly one canonical"):
        t.insert(model_name=s.name, gate_report=report(s.name))  # t's rows under s's name


def test_activate_refuses_an_mlflow_version_mismatch(db):
    s = Slot(db)
    with _raises("not a candidate retrain"):
        s.insert(candidate_mlflow_model_version=9)
    aid = s.switched()
    _sql(db, "update ml_model_registry set mlflow_model_version = 9 where id = %s", (s.c,))
    before = s.state()
    with _raises("not a candidate retrain"):
        s.activate(aid)
    assert s.state() == before


def test_one_live_activation_per_name_and_a_rolled_back_candidate_is_never_reactivated(db):
    import psycopg

    s = Slot(db)
    aid = s.insert()
    with pytest.raises(psycopg.errors.UniqueViolation):
        s.insert()
    # An aborted activation may be retried with the same candidate ...
    s.set_phase(aid, "aborting")
    s.set_phase(aid, "aborted")
    bid = s.switched()
    s.activate(bid)
    _finish_rollback(s, bid)
    # ... a rolled-back one never (OD-7), even if someone puts it back to 'candidate'.
    _owner_write(db, "update ml_model_registry set stage = 'candidate' where id = %s", (s.c,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        s.insert()


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def _finish_rollback(s: Slot, aid: str) -> None:
    s.begin_rollback(aid)
    s.rollback_db(aid)
    _sql(
        s.db,
        "update ml_model_activations set rollback_mlflow_synced_at = now(), "
        "rollback_shap_refreshed_at = now() where id = %s",
        (aid,),
        role="service_role",
    )
    s.set_phase(aid, "rolled_back", rolled_back_at="now")


def test_rollback_requires_its_audit(db):
    import psycopg

    s = Slot(db)
    aid = s.switched()
    s.activate(aid)
    with pytest.raises(psycopg.errors.CheckViolation):
        s.set_phase(aid, "rolling_back")


def test_rollback_db_requires_the_serving_restored(db):
    s = Slot(db)
    aid = s.switched()
    s.activate(aid)
    with _raises("outside its phase"):
        _sql(db, "update ml_model_activations set rollback_serving_at = now() where id = %s",
             (aid,), role="service_role")  # fmt: skip
    s.set_phase(aid, "rolling_back", rolled_back_by="owner", rollback_reason="r")
    with _raises("restore and verify the predecessor bundle"):
        s.rollback_db(aid)
    with _raises("expected rolling_back"):
        t = Slot(db)
        tid = t.switched()
        t.activate(tid)
        t.rollback_db(tid)


def test_rollback_restores_the_prior_state(db):
    s = Slot(db)
    before = s.state()
    aid = s.switched()
    s.activate(aid)
    s.begin_rollback(aid)
    s.rollback_db(aid)

    p, c, d, a = s.reg(s.p), s.reg(s.c), s.dep(), s.ledger(aid)
    assert (p["stage"], p["champion"], p["artifact"]) == ("staging", False, s.p_artifact)
    assert p["registered_at"] == before["p"]["registered_at"]
    assert (c["stage"], c["champion"]) == ("archived", False)  # OD-7
    assert (d["status"], d["reason"]) == ("rolled_back", "l4 test")
    assert a["phase"] == "rolling_back" and a["rollback_db_at"] is not None
    assert s.canonical() == [s.p]

    after = s.state()
    s.rollback_db(aid)  # re-run: verifies, changes nothing
    assert s.state() == after
    _sql(db, "update ml_model_activations set rollback_mlflow_synced_at = now(), "
             "rollback_shap_refreshed_at = now() where id = %s", (aid,), role="service_role")  # fmt: skip
    s.set_phase(aid, "rolled_back", rolled_back_at="now")
    s.rollback_db(aid)  # and once rolled_back
    assert s.state() == after


def test_rollback_restores_a_production_champion(db):
    s = Slot(db, hcp=True)
    _sql(db, "insert into ml_activation_production_allowlist (model_name, ruling) values (%s, %s)",
         (s.name, "l4 test ruling"))  # fmt: skip
    aid = s.switched()
    s.activate(aid)
    assert (s.reg(s.c)["stage"], s.reg(s.c)["champion"]) == ("production", True)
    s.begin_rollback(aid)
    s.rollback_db(aid)
    assert (s.reg(s.p)["stage"], s.reg(s.p)["champion"]) == ("production", True)
    assert _one(db, "select count(*) from ml_model_registry where is_champion and experiment_id = %s",
                (s.exp,)) == 1  # fmt: skip


# ---------------------------------------------------------------------------
# OD-6: production needs the allowlist and the pathology gate; roles keep their stage
# ---------------------------------------------------------------------------


def test_production_needs_the_allowlist_and_the_pathology_gate(db):
    s = Slot(db, hcp=True)
    with _raises("acceptance rule"):
        s.insert()  # allowlist empty
    _sql(db, "insert into ml_activation_production_allowlist (model_name, ruling) values (%s, %s)",
         (s.name, "l4 test ruling"))  # fmt: skip
    prevalence = 609 / 1766
    for bad in (
        # No skill over the base rate. (Exactly AT the float baseline the exact numeric product
        # is ~1e-17 larger, but Lane 5's float pathology_gate then reports passed=false, which
        # the SQL requires to be true.)
        {"brier_candidate": prevalence * (1 - prevalence) + 1e-6},
        {"hcp_pathology_passed": False},
        {"hcp_pathology_passed": _DROP},
        {"hcp_pathology_slope_ok": False},
        {"hcp_pathology_brier_ok": "true"},
        {"hcp_pathology_brier_ok": _DROP},
        {"hcp_pathology_reasons": ["brier_score >= prevalence baseline"]},
        {"served_stage": "staging"},
    ):
        r = report(s.name)
        for k, v in bad.items():
            _mutate(k, v)(r)
        with _raises("acceptance rule"):
            s.insert(gate_report=r)
    s.insert()


def test_an_allowlisted_non_hcp_production_name_takes_a_real_lane5_report(db):
    """codex r1 MEDIUM 6: Lane 5 runs the pathology gate for every production target, so the
    report of an allowlisted non-hcp production name satisfies the SQL rule."""
    hg = pytest.importorskip("src.mlops.activation.holdout_gate")
    s = Slot(db, stage="production")
    _sql(db, "insert into ml_activation_production_allowlist (model_name, ruling) values (%s, %s)",
         (s.name, "l4 test ruling"))  # fmt: skip
    y, served, cand = _lane5_scores()
    r = hg.evaluate_gate(
        y, served, cand, served_bundle_sha256=SHA_P, candidate_bundle_sha256=SHA_C,
        snapshot=_lane5_snapshot(y), model_name=s.name, served_stage="production",
    )  # fmt: skip
    assert r["passed"] and r["hcp_pathology_passed"] is True, r["failed_checks"]
    aid = s.switched(gate_report=json.loads(json.dumps(r)))
    s.activate(aid)
    assert s.canonical() == [s.c] and s.reg(s.c)["stage"] == "production"


def test_an_hcp_name_is_never_activated_below_its_served_stage(db):
    s = Slot(db, hcp=True)
    staging_report = report(s.name, stage="staging")  # so the CHECK, not the gate, refuses
    with _raises("check constraint|changed since the gate ran|acceptance rule"):
        s.insert(served_stage="staging", predecessor_prior_stage="staging",
                 gate_report=staging_report)  # fmt: skip
    with _raises("check constraint"):
        s.insert(served_stage="staging", gate_report=staging_report)


# ---------------------------------------------------------------------------
# OD-5: the weekly re-registration never changes an activated row's role
# ---------------------------------------------------------------------------


UPSERT = (
    # register_model_row's PostgREST upsert, as SQL: ON CONFLICT (model_name, model_version)
    # DO UPDATE SET every column in the payload = EXCLUDED.
    "insert into ml_model_registry (experiment_id, model_name, model_version, algorithm, stage, "
    "is_champion, is_synthetic, artifact_path, auc, feature_count, trained_at, registered_at, "
    "training_provenance) values (%s, %s, '1.0', 'logistic_regression_calibrated', 'staging', "
    "%s, false, %s, 0.9, 25, now(), now(), 'synthetic_gold') "
    "on conflict (model_name, model_version) do update set experiment_id = excluded.experiment_id, "
    "algorithm = excluded.algorithm, stage = excluded.stage, is_champion = excluded.is_champion, "
    "is_synthetic = excluded.is_synthetic, artifact_path = excluded.artifact_path, "
    "auc = excluded.auc, feature_count = excluded.feature_count, trained_at = excluded.trained_at, "
    "registered_at = excluded.registered_at, training_provenance = excluded.training_provenance"
)


@pytest.mark.parametrize("phase", LIVE)
def test_the_weekly_upsert_keeps_the_roles_in_every_live_phase(db, phase):
    s = Slot(db)
    aid = s.insert()
    if phase in ("serving_switched", "active", "rolling_back"):
        s.set_phase(aid, "serving_switched", serving_switched_at="now")
    if phase in ("active", "rolling_back"):
        s.activate(aid)
    if phase == "rolling_back":
        s.set_phase(aid, "rolling_back", rolled_back_by="o", rollback_reason="r")
    if phase == "aborting":
        s.set_phase(aid, "aborting")
    before = s.reg(s.p)
    _sql(db, UPSERT, (s.exp, s.name, False, s.p_artifact), role="service_role")
    after = s.reg(s.p)
    assert (after["stage"], after["champion"], after["registered_at"]) == (
        before["stage"], before["champion"], before["registered_at"],
    )  # fmt: skip
    assert float(after["auc"]) == 0.9  # the refit's metrics still land (the 717dd7158 intent)
    assert len(s.canonical()) == 1


#: The weekly writer's upsert aimed at the CANDIDATE's key (a refit registering under the
#: candidate's version), also claiming a champion flag and a new preprocessing path.
CANDIDATE_UPSERT = (
    "insert into ml_model_registry (experiment_id, model_name, model_version, algorithm, stage, "
    "is_champion, is_synthetic, artifact_path, preprocessing_pipeline_path, auc, feature_count, "
    "trained_at, registered_at, training_provenance) values (%s, %s, %s, "
    "'logistic_regression_calibrated', 'staging', true, false, '/app/data/ml_artifacts/l4/r.pkl', "
    "'/app/data/ml_artifacts/l4/r_pre.pkl', 0.9, 25, now(), now(), 'synthetic_gold') "
    "on conflict (model_name, model_version) do update set experiment_id = excluded.experiment_id, "
    "algorithm = excluded.algorithm, stage = excluded.stage, is_champion = excluded.is_champion, "
    "is_synthetic = excluded.is_synthetic, artifact_path = excluded.artifact_path, "
    "preprocessing_pipeline_path = excluded.preprocessing_pipeline_path, auc = excluded.auc, "
    "feature_count = excluded.feature_count, trained_at = excluded.trained_at, "
    "registered_at = excluded.registered_at, training_provenance = excluded.training_provenance"
)


@pytest.mark.parametrize("phase", LIVE)
def test_the_weekly_upsert_keeps_the_live_candidates_role_and_bundle(db, phase):
    """codex r1 LOW 7: the candidate row of a live activation keeps stage, champion flag,
    registered_at and both bundle paths; the refit's metrics still land."""
    s = Slot(db)
    aid = s.insert()
    if phase in ("serving_switched", "active", "rolling_back"):
        s.set_phase(aid, "serving_switched", serving_switched_at="now")
    if phase in ("active", "rolling_back"):
        s.activate(aid)
    if phase == "rolling_back":
        s.set_phase(aid, "rolling_back", rolled_back_by="o", rollback_reason="r")
    if phase == "aborting":
        s.set_phase(aid, "aborting")
    before = s.reg(s.c)
    _sql(db, CANDIDATE_UPSERT, (s.exp, s.name, s.c_version), role="service_role")
    after = s.reg(s.c)
    keep = ("stage", "champion", "registered_at", "artifact", "pre")
    assert {k: after[k] for k in keep} == {k: before[k] for k in keep}
    assert float(after["auc"]) == 0.9
    assert len(s.canonical()) == 1


def test_the_weekly_upsert_moves_roles_again_once_nothing_is_live(db):
    s = Slot(db)
    aid = s.insert()
    s.set_phase(aid, "aborting")
    s.set_phase(aid, "aborted")
    _sql(db, "update ml_model_registry set stage = 'archived' where id = %s", (s.p,),
         role="service_role")  # fmt: skip
    _sql(db, UPSERT, (s.exp, s.name, False, s.p_artifact), role="service_role")
    assert s.reg(s.p)["stage"] == "staging"


def test_an_hcp_weekly_upsert_claiming_the_champion_flag_leaves_one_champion(db):
    s = Slot(db, hcp=True)
    _sql(db, "insert into ml_activation_production_allowlist (model_name, ruling) values (%s, %s)",
         (s.name, "l4 test ruling"))  # fmt: skip
    aid = s.switched()
    s.activate(aid)
    _sql(db, UPSERT.replace("'staging'", "'production'"), (s.exp, s.name, True, s.p_artifact),
         role="service_role")  # fmt: skip
    assert (s.reg(s.c)["stage"], s.reg(s.c)["champion"]) == ("production", True)
    assert (s.reg(s.p)["stage"], s.reg(s.p)["champion"]) == ("archived", False)
    assert _one(db, "select count(*) from ml_model_registry where is_champion and experiment_id = %s",
                (s.exp,)) == 1  # fmt: skip


def test_no_other_row_of_an_activated_name_becomes_canonical_or_champion(db):
    s = Slot(db)
    aid = s.switched()
    s.activate(aid)
    other = str(uuid.uuid4())
    s.add_row(other, "1.0_retrained_20261001_x", "candidate", retrain_of=s.c, mlflow_version=9,
              role="service_role")  # a new retrain candidate still registers  # fmt: skip
    with _raises("live activation"):
        _sql(db, "update ml_model_registry set stage = 'staging' where id = %s", (other,),
             role="service_role")  # fmt: skip
    with _raises("live activation"):
        _sql(db, "update ml_model_registry set is_champion = true where id = %s", (other,),
             role="service_role")  # fmt: skip
    with _raises("live activation"):
        s.add_row(str(uuid.uuid4()), "1.1", "staging", role="service_role")
    assert s.canonical() == [s.c]


def test_the_rpc_token_does_not_outlive_the_rpc(db):
    s = Slot(db)
    aid = s.switched()
    with db.connect() as c:
        c.execute('set role "service_role"')
        c.execute("select public.activate_model_candidate(%s)", (aid,))
        # Same transaction, after the RPC returned: the guard is back in force.
        c.execute("update ml_model_registry set stage = 'staging' where id = %s", (s.p,))
        (stage,) = c.execute("select stage::text from ml_model_registry where id = %s",
                             (s.p,)).fetchone()  # fmt: skip
        assert stage == "archived"
        c.commit()


def test_setting_the_old_guc_is_no_bypass(db):
    """codex r1 MEDIUM 5: any role can set_config() a custom GUC. The retired
    ``e2i.activation_rpc`` flag must not let service_role past the role guard."""
    s = Slot(db)
    aid = s.switched()
    s.activate(aid)
    with db.connect() as c:
        c.execute('set role "service_role"')
        c.execute("select set_config('e2i.activation_rpc', 'on', true)")
        c.execute(UPSERT, (s.exp, s.name, False, s.p_artifact))
        c.execute("update ml_model_registry set stage = 'staging', is_champion = true "
                  "where id = %s", (s.p,))  # fmt: skip
        c.commit()
    p = s.reg(s.p)
    assert (p["stage"], p["champion"]) == ("archived", False)
    assert float(p["auc"]) == 0.9  # the refit still lands
    assert s.canonical() == [s.c]
    with db.connect() as c:
        c.execute('set role "service_role"')
        c.execute("select set_config('e2i.activation_rpc', 'on', true)")
        with _raises("live activation"):
            c.execute(
                "insert into ml_model_registry (experiment_id, model_name, model_version, "
                "algorithm, stage) values (%s, %s, '1.1', 'lr', 'staging')",
                (s.exp, s.name),
            )
    assert s.canonical() == [s.c]


def test_the_rpc_token_table_is_owner_only_and_empty_after_each_rpc(db):
    import psycopg

    s = Slot(db)
    aid = s.switched()
    for role in ("service_role", "authenticated", "anon"):
        for sql in (
            "insert into ml_activation_rpc_authority (xact) values (pg_current_xact_id())",
            "select count(*) from ml_activation_rpc_authority",
            "delete from ml_activation_rpc_authority",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                _sql(db, sql, role=role)
    for rpc in ("activate_model_candidate", "rollback_model_activation"):
        if rpc == "rollback_model_activation":
            s.begin_rollback(aid)
        with db.connect() as c:
            c.execute('set role "service_role"')
            c.execute(f"select public.{rpc}(%s)", (aid,))
            c.execute("reset role")  # the owner looks, in the RPC's own transaction
            assert c.execute("select count(*) from ml_activation_rpc_authority").fetchone() == (0,)
            c.commit()
    assert _one(db, "select count(*) from ml_activation_rpc_authority") == 0
    assert s.canonical() == [s.p] and s.ledger(aid)["rollback_db_at"] is not None


def _wait_until_blocked(db: _pg.PgConn, pid: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = _sql(db, "select wait_event_type = 'Lock' from pg_stat_activity where pid = %s",
                    (pid,))  # fmt: skip
        if not rows:
            raise AssertionError(f"backend {pid} finished without ever blocking on a lock")
        if rows[0][0]:
            return
        time.sleep(0.05)
    raise AssertionError(f"backend {pid} never blocked on a lock")


class _Background:
    """Run one statement on its own connection in a thread; record its pid and outcome."""

    def __init__(self, db: _pg.PgConn, sql: str, params: Any, role: str = "service_role"):
        self.pid: Optional[int] = None
        self.error: Optional[BaseException] = None
        self._ready = threading.Event()

        def run() -> None:
            try:
                with db.connect(autocommit=True) as c:
                    c.execute(f'set role "{role}"')
                    self.pid = c.info.backend_pid
                    self._ready.set()
                    c.execute(sql, params)
            except BaseException as e:  # noqa: BLE001 -- asserted by the test
                self.error = e
            finally:
                self._ready.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self._ready.wait(20)

    def join(self) -> None:
        self.thread.join(30)
        assert not self.thread.is_alive()


def test_the_weekly_upsert_racing_the_rpc_ends_with_one_canonical_row(db):
    # Order 1: the RPC's transaction is open; the upsert blocks behind it, then runs against
    # the committed switch and keeps the predecessor archived.
    s = Slot(db)
    aid = s.switched()
    with db.connect() as rpc:
        rpc.execute('set role "service_role"')
        rpc.execute("select public.activate_model_candidate(%s)", (aid,))
        upsert = _Background(db, UPSERT, (s.exp, s.name, False, s.p_artifact))
        _wait_until_blocked(db, upsert.pid)
        rpc.commit()
    upsert.join()
    assert upsert.error is None, upsert.error
    assert s.canonical() == [s.c]
    assert s.reg(s.p)["stage"] == "archived" and float(s.reg(s.p)["auc"]) == 0.9

    # Order 2: the upsert's transaction is open; the RPC blocks behind it, then activates.
    t = Slot(db)
    bid = t.switched()
    with db.connect() as up:
        up.execute('set role "service_role"')
        up.execute(UPSERT, (t.exp, t.name, False, t.p_artifact))
        rpc2 = _Background(db, "select public.activate_model_candidate(%s)", (bid,))
        _wait_until_blocked(db, rpc2.pid)
        up.commit()
    rpc2.join()
    assert rpc2.error is None, rpc2.error
    assert t.canonical() == [t.c]
    assert t.ledger(bid)["phase"] == "active"


def test_a_new_canonical_row_racing_the_rpc_is_refused(db):
    s = Slot(db)
    aid = s.switched()
    insert = (
        "insert into ml_model_registry (experiment_id, model_name, model_version, algorithm, stage)"
        " values (%s, %s, '1.1', 'lr', 'staging')"
    )
    with db.connect() as rpc:
        rpc.execute('set role "service_role"')
        rpc.execute("select public.activate_model_candidate(%s)", (aid,))
        writer = _Background(db, insert, (s.exp, s.name))
        _wait_until_blocked(db, writer.pid)
        rpc.commit()
    writer.join()
    assert writer.error is not None and "live activation" in str(writer.error)
    assert s.canonical() == [s.c]


NEW_CANONICAL = (
    "insert into ml_model_registry (experiment_id, model_name, model_version, algorithm, stage)"
    " values (%s, %s, '1.1', 'lr', 'staging')"
)


def test_a_registry_writer_racing_an_open_ledger_insert_blocks_then_is_refused(db):
    """codex r1 HIGH 2, order (a): the ledger insert's transaction is open. A new canonical
    row of the name blocks behind it and, once the ledger row is committed, is refused."""
    s = Slot(db)
    sql, vals = s.insert_sql()
    with db.connect() as a:
        a.execute('set role "service_role"')
        a.execute(sql, vals)
        writer = _Background(db, NEW_CANONICAL, (s.exp, s.name))
        try:
            _wait_until_blocked(db, writer.pid)
        finally:
            a.commit()
    writer.join()
    assert writer.error is not None and "live activation" in str(writer.error), writer.error
    assert s.canonical() == [s.p]


def test_a_ledger_insert_racing_an_open_registry_writer_blocks_then_is_refused(db):
    """codex r1 HIGH 2, order (b): a new canonical row is uncommitted. The ledger insert
    blocks behind it and, once it is committed, counts two canonical rows and refuses."""
    s = Slot(db)
    sql, vals = s.insert_sql()
    with db.connect() as b:
        b.execute('set role "service_role"')
        b.execute(NEW_CANONICAL, (s.exp, s.name))
        ledger = _Background(db, sql, vals)
        try:
            _wait_until_blocked(db, ledger.pid)
        finally:
            b.commit()
    ledger.join()
    assert ledger.error is not None and "exactly one canonical" in str(ledger.error), ledger.error
    assert _one(db, "select count(*) from ml_model_activations where model_name = %s",
                (s.name,)) == 0  # fmt: skip


# ---------------------------------------------------------------------------
# Privileges and the production transports
# ---------------------------------------------------------------------------


def test_role_privileges(db):
    import psycopg

    s = Slot(db)
    aid = s.switched()
    for role in ("anon", "authenticated"):
        for sql, params in [
            ("select count(*) from ml_model_activations", None),
            ("select count(*) from ml_activation_production_allowlist", None),
            ("select public.activate_model_candidate(%s)", (aid,)),
            ("select public.rollback_model_activation(%s)", (aid,)),
            ("select public.activation_gate_passes('{}'::jsonb, '', '', '', '')", None),
        ]:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                _sql(db, sql, params, role=role)
    for sql, params in [
        ("delete from ml_model_activations where id = %s", (aid,)),
        ("truncate ml_model_activations", None),
        ("insert into ml_activation_production_allowlist (model_name, ruling) values ('x', 'y')",
         None),
        ("select public._activation_lock_and_count_served('x', gen_random_uuid())", None),
    ]:  # fmt: skip
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _sql(db, sql, params, role="service_role")
    assert _one(db, "select count(*) from ml_model_activations", role="service_role") == 1
    s.activate(aid)  # service_role executes the RPC


@pytest.fixture
def rest(db) -> Iterator[Any]:
    from tests.unit.test_database.learning_loop.test_ml_registry_promotion_gate_realdb import (
        ThrowawayRest,
    )

    server = ThrowawayRest(db)
    try:
        server.start()
        yield server
    finally:
        server.stop()


async def test_the_rpcs_and_the_real_weekly_writer_through_postgrest(db, rest, tmp_path):
    """What Lanes 6 and 7 will call, over the transport they use: PostgREST as service_role."""
    from src.mlops.prediction_synthesizer_deploy import register_model_row

    s = Slot(db)
    client = rest.service_role_client()
    ins = await client.table("ml_model_activations").insert(s.ledger_row()).execute()
    aid = ins.data[0]["id"]
    await (
        client.table("ml_model_activations")
        .update({"phase": "serving_switched", "serving_switched_at": "now"})
        .eq("id", aid)
        .eq("phase", "prepared")
        .execute()
    )
    await client.rpc("activate_model_candidate", {"p_activation_id": aid}).execute()
    assert s.ledger(aid)["phase"] == "active" and s.canonical() == [s.c]

    # The REAL weekly writer against the predecessor: the refit lands, the role does not,
    # and its read-back reports that the row did not land at 'staging' (Lane 7 teaches it
    # that this is the preserved role of an activation's predecessor).
    artifact = tmp_path / "v1.pkl"
    artifact.write_bytes(b"artifact")
    registered_before = s.reg(s.p)["registered_at"]
    with pytest.raises(RuntimeError, match="did not persist as a staging row"):
        await register_model_row(
            client, experiment_id=s.exp, model_name=s.name, model_version="1.0",
            algorithm="logistic_regression_calibrated", artifact_path=str(artifact), auc=0.8612,
            feature_count=25, stage="staging", training_provenance="synthetic_gold",
        )  # fmt: skip
    p = s.reg(s.p)
    assert (p["stage"], p["champion"], p["registered_at"]) == ("archived", False, registered_before)
    assert float(p["auc"]) == 0.8612 and p["artifact"] == str(artifact)
    assert s.canonical() == [s.c]

    # A direct write of an RPC-only marker is refused by PostgREST as well.
    from postgrest.exceptions import APIError

    with pytest.raises(APIError):
        await (
            client.table("ml_model_activations")
            .update({"activated_at": "2026-01-01"})
            .eq("id", aid)
            .execute()
        )
    await (
        client.table("ml_model_activations")
        .update({"phase": "rolling_back", "rolled_back_by": "owner", "rollback_reason": "l4"})
        .eq("id", aid)
        .eq("phase", "active")
        .execute()
    )
    await (
        client.table("ml_model_activations")
        .update({"rollback_serving_at": "now"})
        .eq("id", aid)
        .execute()
    )
    await client.rpc("rollback_model_activation", {"p_activation_id": aid}).execute()
    assert s.canonical() == [s.p] and s.reg(s.c)["stage"] == "archived"


def _lane5_scores(worse: bool = False) -> tuple:
    import numpy as np

    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 1200)
    y = (rng.uniform(size=1200) < p).astype(int)
    served = np.clip(p + rng.normal(0, 0.02, 1200), 0.01, 0.99)
    if not worse:
        return y, served, p
    # Served plus U(-0.075, 0.075) jitter: fails ONLY auc_noninferiority (lower bound -0.0105;
    # Brier upper 0.0049 and slope 0.93 still pass), so a forged verdict needs just the bound.
    jitter = np.random.default_rng(3).uniform(-0.075, 0.075, 1200)
    return y, served, np.clip(served + jitter, 0.0, 1.0)


def _lane5_snapshot(y: Any) -> Dict[str, Any]:
    return {"splits": ["test", "holdout"], "n": len(y), "n_pos": int(y.sum()),
            "rows_sha256": ROWS_SHA}  # fmt: skip


def test_a_real_lane5_report_passes_the_sql_rule(db):
    """Cross-lane contract: Lane 5's evaluate_gate output is accepted by the SQL predicate."""
    hg = pytest.importorskip(
        "src.mlops.activation.holdout_gate", reason="#2318 Lane 5 (holdout_gate) not merged yet"
    )
    s = Slot(db)
    y, served, p = _lane5_scores()
    r = hg.evaluate_gate(
        y, served, p, served_bundle_sha256=SHA_P, candidate_bundle_sha256=SHA_C,
        snapshot=_lane5_snapshot(y), model_name=s.name, served_stage="staging",
    )  # fmt: skip
    assert r["passed"], r["failed_checks"]
    s.insert(gate_report=json.loads(json.dumps(r)))
    t = Slot(db)
    failed = dict(r, model_name=t.name, passed=False, failed_checks=["auc_noninferiority"])
    with _raises("acceptance rule"):
        t.insert(gate_report=failed)


def test_a_failed_lane5_report_with_a_forged_verdict_is_refused(db):
    """codex r1 HIGH 1: a real FAILED report whose verdict and bound were edited to pass --
    auc_lower_bound raised, auc_delta / se_delta left as Lane 5 computed them -- is refused."""
    hg = pytest.importorskip("src.mlops.activation.holdout_gate")
    s = Slot(db)
    y, served, worse = _lane5_scores(worse=True)
    r = hg.evaluate_gate(
        y, served, worse, served_bundle_sha256=SHA_P, candidate_bundle_sha256=SHA_C,
        snapshot=_lane5_snapshot(y), model_name=s.name, served_stage="staging",
    )  # fmt: skip
    assert r["failed_checks"] == ["auc_noninferiority"], r["failed_checks"]
    forged = dict(json.loads(json.dumps(r)), passed=True, failed_checks=[], auc_lower_bound=0.0)
    with _raises("acceptance rule"):
        s.insert(gate_report=forged)
    assert _one(db, "select count(*) from ml_model_activations") == 0


# ---------------------------------------------------------------------------
# The upgrade path and the rollback file
# ---------------------------------------------------------------------------


@pytest.mark.realdb_upgrade(KEY)
def test_the_runner_applies_165_to_prod_as_it_is_now(base_clone_db, tmp_path):
    conn = base_clone_db("m165_runner")
    dry = _pg.run_runner(conn, _pg.REPO_ROOT, tmp_path / "shims", "--dry-run")
    assert dry.returncode == 0, dry.stderr.decode()
    out = re.sub(r"\x1b\[[0-9;]*m", "", dry.stdout.decode())  # the runner's colour codes
    assert f"[PENDING] {KEY}" in out.splitlines()
    real = _pg.run_runner(conn, _pg.REPO_ROOT, tmp_path / "shims")
    assert real.returncode == 0, real.stdout.decode() + real.stderr.decode()
    assert conn.rows(f"select count(*) from schema_migrations where filename = '{KEY}'") == ["1"]
    assert conn.rows("select to_regclass('public.ml_model_activations') is not null") == ["t"]


def _rollback(conn: _pg.PgConn):
    return conn.pg.run_script(conn.db, R165.read_bytes(), single_transaction=True, user="postgres")


def _wait_until_psql_blocked(db: _pg.PgConn, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _one(db, "select count(*) from pg_stat_activity where datname = current_database() "
                    "and application_name = 'psql' and wait_event_type = 'Lock'"):  # fmt: skip
            return
        time.sleep(0.05)
    raise AssertionError("the rollback file never blocked on a lock")


def test_the_rollback_file_waits_for_an_uncommitted_activation_then_refuses(db):
    """codex r1 HIGH 4: the live-row check runs under ACCESS EXCLUSIVE on the ledger, so a
    'prepared' row committed while the rollback runs is seen, and nothing is dropped."""
    s = Slot(db)
    sql, vals = s.insert_sql()
    out: Dict[str, Any] = {}
    with db.connect() as ins:
        ins.execute('set role "service_role"')
        ins.execute(sql, vals)
        worker = threading.Thread(target=lambda: out.update(proc=_rollback(db)), daemon=True)
        worker.start()
        try:
            _wait_until_psql_blocked(db)
        finally:
            ins.commit()
    worker.join(120)
    assert not worker.is_alive()
    proc = out["proc"]
    assert proc.returncode != 0 and "live activation" in proc.stderr.decode(), proc.stderr
    assert db.rows(
        "select to_regclass('public.ml_model_activations') is not null, "
        "to_regprocedure('public.activate_model_candidate(uuid)') is not null, "
        "to_regclass('public.ml_activation_rpc_authority') is not null, "
        "(select count(*) from pg_trigger where tgname = 'tr_ml_model_registry_activation_role_guard'), "
        f"(select count(*) from schema_migrations where filename = '{KEY}')"
    ) == ["t|t|t|1|1"]


def test_the_rollback_refuses_while_an_activation_is_live_then_retires_and_reapplies(db):
    s = Slot(db)
    aid = s.switched()
    s.activate(aid)
    proc = _rollback(db)
    assert proc.returncode != 0 and "live activation" in proc.stderr.decode()
    assert db.rows("select to_regclass('public.ml_model_activations') is not null") == ["t"]

    _finish_rollback(s, aid)
    proc = _rollback(db)
    assert proc.returncode == 0, proc.stderr.decode()
    assert db.rows(
        "select to_regclass('public.ml_model_activations') is null, "
        "to_regclass('public.ml_model_activations_retired_165') is not null, "
        "to_regprocedure('public.activate_model_candidate(uuid)') is null, "
        "to_regclass('public.ml_activation_rpc_authority') is null, "
        "(select count(*) from ml_model_activations_retired_165), "
        f"(select count(*) from schema_migrations where filename = '{KEY}')"
    ) == ["t|t|t|t|1|0"]
    assert db.rows(
        "select count(*) from pg_trigger where tgrelid = 'ml_model_registry'::regclass "
        "and tgname = 'tr_ml_model_registry_activation_role_guard'"
    ) == ["0"]
    # Without the guard the weekly writer moves roles again, as before 165.
    _sql(db, UPSERT, (s.exp, s.name, False, s.p_artifact), role="service_role")
    assert s.reg(s.p)["stage"] == "staging"

    # Re-applying 165 after a rollback builds a complete, working ledger again.
    assert _pg.apply_migration(db, M165, record=KEY) == "wrapped"
    assert db.rows("select count(*) from pg_indexes where tablename = 'ml_model_activations'") == [
        "3"
    ]
    t = Slot(db)
    tid = t.switched()
    t.activate(tid)
    assert t.canonical() == [t.c]
