"""scripts/load_csu_escalation_cohort.py -- the SYNTHETIC backing parquet
(scripts/build_csu_escalation_synthetic_cohort.py) -> csu_escalation_causal.

Lane C-load: the Optum loader generalised through ``scripts/causal_cohort_loader``
(one ``CohortSpec`` per table). No DB here: the client is a recording fake. The
frames are the REAL generator's output (the contract's SSOT), never a hand-built
imitation, so a contract drift shows up here before the owner runs the load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import causal_cohort_loader as _engine  # noqa: E402
from scripts.causal_cohort_loader import CSU_SPEC, OPTUM_SPEC  # noqa: E402
from scripts.causal_cohort_loader import load_frame as engine_load_frame  # noqa: E402
from scripts.load_csu_escalation_cohort import (  # noqa: E402
    ARMS,
    BRAND,
    DEFAULT_INPUT,
    ON_CONFLICT,
    OUTCOME_COLUMNS,
    REQUIRED_COLUMNS,
    SPEC,
    TABLE,
    TREATMENT,
    arm_split,
    fetch_live_rows,
    fetch_live_split,
    load_frame,
    main,
    to_records,
    upsert,
    verify,
    verify_rows,
)
from scripts.load_optum_causal_cohort import REQUIRED_COLUMNS as OPTUM_REQUIRED  # noqa: E402
from src.ml.synthetic.generators.csu_escalation_causal import (  # noqa: E402
    COMPETITOR_ARMS,
    CONTRACT_COLUMNS,
    TREATED_ARM,
    generate_csu_escalation_cohort,
)
from tests.unit.test_scripts.causal_cohort_fakes import FakeClient  # noqa: E402

_N = 400  # enough rows for both arms, all three brand labels and every categorical level
_DEPLOYED = "e0468f0de" + "0" * 31  # a stand-in for the deployed image's commit
_REAL_DEPLOYED_IMAGE_COMMIT = _engine.deployed_image_commit  # bound before the autouse stand-in


@pytest.fixture(autouse=True)
def _deployed_guard_attested(monkeypatch):
    """The deployed-image attestation is a docker + git read on the prod box;
    unit tests stand it in (a readable image whose commit descends from the
    guard commit) and override it per test where the refusal is the subject."""
    monkeypatch.setattr(_engine, "deployed_image_commit", lambda container="e2i_api": _DEPLOYED)
    monkeypatch.setattr(_engine, "is_ancestor", lambda ancestor, descendant: True)


def _frame(n: int = _N, seed: int = 7) -> pd.DataFrame:
    frame, _truths = generate_csu_escalation_cohort(n=n, seed=seed)
    return frame


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "csu.parquet"
    df.to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# spec + contract
# ---------------------------------------------------------------------------


def test_constants():
    assert SPEC is CSU_SPEC
    assert TABLE == "csu_escalation_causal"
    assert ON_CONFLICT == "patient_id"
    assert TREATMENT == "treatment_remibrutinib"
    assert BRAND == "index_biologic_brand"
    assert SPEC.treated_arms == (TREATED_ARM,) == ("RHAPSIDO",)
    assert SPEC.control_arms == COMPETITOR_ARMS == ("XOLAIR", "DUPIXENT")
    assert set(ARMS) == {"RHAPSIDO", "XOLAIR", "DUPIXENT"}
    assert SPEC.synthetic is True  # every backing row MUST be is_synthetic=true
    assert SPEC.migration == 149
    assert DEFAULT_INPUT == (
        "data/rwd/synthetic_CSU/csu_escalation_causal/csu_escalation_causal_synthetic.parquet"
    )
    assert OUTCOME_COLUMNS == (
        "persistent_at_180d_g28",
        "discontinued_180d",
        "biologic_switch_180d_flag",
        "persistent_at_180d",
    )


def test_required_columns_equal_the_generator_contract():
    """The loader's exact-contract check must be the generator's CONTRACT_COLUMNS
    (81 columns): a drift on either side is refused before any write."""
    assert set(REQUIRED_COLUMNS) == set(CONTRACT_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(set(REQUIRED_COLUMNS)) == 81
    assert TREATMENT in REQUIRED_COLUMNS and "treatment_dupixent" not in REQUIRED_COLUMNS


def test_optum_contract_is_unchanged_by_the_generalisation():
    """The Optum wrapper still declares Lane A's 81-column contract with ITS
    treatment; the two specs differ only in the treatment column name."""
    assert len(OPTUM_REQUIRED) == 81
    assert "treatment_dupixent" in OPTUM_REQUIRED and TREATMENT not in OPTUM_REQUIRED
    assert set(OPTUM_REQUIRED) - set(REQUIRED_COLUMNS) == {"treatment_dupixent"}
    assert set(REQUIRED_COLUMNS) - set(OPTUM_REQUIRED) == {TREATMENT}
    assert OPTUM_SPEC.synthetic is False and CSU_SPEC.synthetic is True


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_load_frame_accepts_the_builder_output(tmp_path):
    df = load_frame(_write(tmp_path, _frame()))
    assert len(df) == _N
    assert set(df[BRAND].unique()) == {"RHAPSIDO", "XOLAIR", "DUPIXENT"}
    assert bool(df["is_synthetic"].all())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        # provenance: the backing is synthetic-only -- a real-looking row is refused
        (lambda d: d.assign(is_synthetic=False), "is_synthetic"),
        (lambda d: d.assign(is_synthetic=[False] + [True] * (len(d) - 1)), "is_synthetic"),
        (lambda d: d.assign(is_synthetic=None), "is_synthetic"),
        # a STRING "true"/"false" or an int 0/1 column is not a boolean: truth
        # coercion would read "false" as True and admit a real-looking row
        (lambda d: d.assign(is_synthetic="true"), "is_synthetic"),
        (lambda d: d.assign(is_synthetic="false"), "is_synthetic"),
        (lambda d: d.assign(is_synthetic=1), "is_synthetic"),
        (lambda d: d.assign(is_synthetic=[True] * (len(d) - 1) + [None]), "is_synthetic"),
        (lambda d: pd.concat([d, d.iloc[[0]]]), "patient_id"),
        (lambda d: d.drop(columns=["persistent_at_180d_g28"]), "persistent_at_180d_g28"),
        (lambda d: d.drop(columns=["cci_mi"]), "cci_mi"),
        (lambda d: d.assign(unexpected_extra_column="oops"), "unexpected_extra_column"),
        # Lane A's column name is NOT this table's treatment
        (
            lambda d: d.rename(columns={"treatment_remibrutinib": "treatment_dupixent"}),
            "treatment_remibrutinib",
        ),
        (
            lambda d: d.assign(treatment_remibrutinib=1 - d["treatment_remibrutinib"]),
            "index_biologic_brand",
        ),
        (lambda d: d.assign(index_biologic_brand="TEZSPIRE"), "index_biologic_brand"),
        (lambda d: d.assign(persistent_at_180d_g28=2), "persistent_at_180d_g28"),
        (lambda d: d.iloc[0:0], "empty"),
        # one arm only: no remibrutinib rows, or no competitor rows
        (lambda d: d[d["index_biologic_brand"] != "RHAPSIDO"], "RHAPSIDO"),
        (lambda d: d[d["index_biologic_brand"] == "RHAPSIDO"], "XOLAIR"),
    ],
)
def test_load_frame_fails_loud(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        load_frame(_write(tmp_path, mutate(_frame())))


def test_load_frame_accepts_a_nullable_boolean_provenance_column(tmp_path):
    """pandas' nullable ``boolean`` dtype round-trips through parquet as real
    booleans (no NA): accepted like the plain bool column."""
    df = _frame().assign(is_synthetic=pd.array([True] * _N, dtype="boolean"))
    out = load_frame(_write(tmp_path, df))
    assert len(out) == _N and bool(out["is_synthetic"].all())


def test_load_frame_accepts_a_competitor_pool_missing_one_brand(tmp_path):
    """The contrast is remibrutinib vs the competitor POOL: a frame whose only
    competitor is XOLAIR still carries both treatment arms."""
    df = _frame()
    df = df[df[BRAND] != "DUPIXENT"]
    out = load_frame(_write(tmp_path, df))
    assert set(out[TREATMENT].unique()) == {0, 1}


def test_specs_refuse_each_others_rows(tmp_path):
    """Provenance is per TABLE: the CSU spec refuses real rows and the Optum spec
    refuses synthetic rows -- the same engine, opposite rule."""
    # In the Optum CONTRACT (its treatment column name) but synthetic rows: the
    # provenance rule, not the column check, is what refuses it.
    synthetic = _write(tmp_path, _frame().rename(columns={TREATMENT: "treatment_dupixent"}))
    with pytest.raises(ValueError, match="is_synthetic"):
        engine_load_frame(synthetic, spec=OPTUM_SPEC)
    real_looking = _frame().assign(is_synthetic=False)
    with pytest.raises(ValueError, match="is_synthetic"):
        engine_load_frame(_write(tmp_path, real_looking), spec=CSU_SPEC)


def test_arm_split_counts_by_brand_and_treatment():
    df = _frame()
    split = arm_split(df)
    assert split["n"] == _N
    assert sum(split["arms"].values()) == _N
    assert set(split["arms"]) == {"RHAPSIDO", "XOLAIR", "DUPIXENT"}
    assert split["treatment"]["1"] == split["arms"]["RHAPSIDO"] == int(df[TREATMENT].sum())
    assert split["treatment"]["0"] == split["arms"]["XOLAIR"] + split["arms"]["DUPIXENT"]
    assert set(split["outcome_positives"]) == set(OUTCOME_COLUMNS)
    for col in OUTCOME_COLUMNS:
        assert split["outcome_positives"][col]["RHAPSIDO"] == int(
            df.loc[df[BRAND] == "RHAPSIDO", col].sum()
        )


def test_to_records_is_json_safe():
    rec = to_records(_frame(n=210))[0]
    assert rec["is_synthetic"] is True
    assert isinstance(rec["index_date"], str) and len(rec["index_date"]) == 10
    assert isinstance(rec[TREATMENT], int)
    assert rec["geographic_region"] is None or isinstance(rec["geographic_region"], str)


# ---------------------------------------------------------------------------
# DB path with the recording fake
# ---------------------------------------------------------------------------


def test_upsert_targets_the_csu_table_on_patient_id():
    client = FakeClient()
    n = upsert(client, to_records(_frame(n=700)), batch_size=500)
    assert n == 700
    assert [len(b) for b, _ in client.t.upserts] == [500, 200]
    assert {oc for _, oc in client.t.upserts} == {ON_CONFLICT}
    assert set(client.tables) == {TABLE}
    upsert(client, to_records(_frame(n=700)), batch_size=500)
    assert len(client.t.rows) == 700  # idempotent


def test_upsert_is_the_write_boundary_for_provenance():
    """codex r1 HIGH: a caller that bypasses main/load_frame still cannot write
    a real-looking row -- upsert refuses BEFORE the first batch."""
    client = FakeClient()
    records = to_records(_frame(n=210))
    records[3]["is_synthetic"] = False
    with pytest.raises(ValueError, match="is_synthetic"):
        upsert(client, records)
    assert client.t.upserts == [] and client.t.rows == {}
    records[3]["is_synthetic"] = "true"
    with pytest.raises(ValueError, match="non-boolean"):
        upsert(client, records)
    assert client.t.upserts == []


def test_upsert_is_the_write_boundary_for_the_guard(monkeypatch):
    from src.api.routes.causal import datasets as datasets_mod

    client = FakeClient()
    records = to_records(_frame(n=210))
    monkeypatch.setattr(datasets_mod, "_CAUSAL_SYNTHETIC_BACKED", frozenset())
    with pytest.raises(RuntimeError, match="GUARD"):
        upsert(client, records)
    assert client.t.upserts == []


def test_upsert_refuses_when_the_deployed_image_predates_the_guard(monkeypatch):
    client = FakeClient()
    records = to_records(_frame(n=210))
    monkeypatch.setattr(_engine, "is_ancestor", lambda a, d: False)
    with pytest.raises(RuntimeError, match="does NOT descend"):
        upsert(client, records)
    assert client.t.upserts == []
    monkeypatch.setattr(_engine, "deployed_image_commit", lambda container="e2i_api": None)
    with pytest.raises(RuntimeError, match="cannot read the deployed"):
        upsert(client, records)
    assert client.t.upserts == []


def test_deployed_guard_uses_the_attested_commit_when_docker_is_unreadable(monkeypatch):
    seen = {}
    monkeypatch.setattr(_engine, "deployed_image_commit", lambda container="e2i_api": None)
    monkeypatch.setattr(
        _engine, "is_ancestor", lambda a, d: seen.setdefault("pair", (a, d)) and True
    )
    assert _engine.guard_problem(SPEC, deployed_commit=_DEPLOYED) is None
    assert seen["pair"] == (SPEC.deployed_guard_commit, _DEPLOYED)
    assert SPEC.deployed_guard_commit == "c0860bbf42296160396a8355ea5d225fc8daa1ad"
    assert OPTUM_SPEC.deployed_guard_commit is None and OPTUM_SPEC.provenance_guard is None


def test_deployed_image_commit_parses_the_image_tag(monkeypatch):
    import subprocess

    class _Out:
        def __init__(self, rc, stdout):
            self.returncode, self.stdout = rc, stdout

    tag = "ghcr.io/enunezvn/e2i-api:" + "a" * 40
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Out(0, tag + "\n"))
    assert _REAL_DEPLOYED_IMAGE_COMMIT() == "a" * 40
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Out(0, "ghcr.io/enunezvn/e2i-api:latest\n")
    )
    assert _REAL_DEPLOYED_IMAGE_COMMIT() is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Out(1, ""))
    assert _REAL_DEPLOYED_IMAGE_COMMIT() is None


def test_fetch_live_rows_orders_and_pages_the_whole_table():
    """Two pages (ROW_PAGE_SIZE = 1000) must come back in a stable order with no
    duplicate or missing patient_id (codex r1 MED)."""
    client = FakeClient()
    df = _frame(n=1500)
    upsert(client, to_records(df))
    rows = fetch_live_rows(client)
    assert rows is not None and len(rows) == 1500
    ids = [r["patient_id"] for r in rows]
    assert ids == sorted(ids) and len(set(ids)) == 1500
    assert verify_rows(to_records(df), rows) == []


def test_fetch_live_split_and_rows_agree_with_the_parquet():
    client = FakeClient()
    df = _frame()
    records = to_records(df)
    upsert(client, records)
    live = fetch_live_split(client)
    assert live is not None and verify(arm_split(df), live) == []
    rows = fetch_live_rows(client)
    assert rows is not None and verify_rows(records, rows) == []


def test_fetch_live_reports_a_missing_table_as_none():
    assert fetch_live_split(FakeClient(missing=True)) is None
    assert fetch_live_rows(FakeClient(missing=True)) is None


def test_main_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    import scripts.load_csu_escalation_cohort as mod

    client = FakeClient()
    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(_write(tmp_path, _frame()))]) == 0
    assert client.t.upserts == []
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "RHAPSIDO" in out and TABLE in out
    assert "WOULD WRITE" in out and f"n={_N}" in out


def test_main_execute_loads_then_verifies(tmp_path, monkeypatch, capsys):
    import scripts.load_csu_escalation_cohort as mod

    client = FakeClient()
    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(_write(tmp_path, _frame())), "--execute"]) == 0
    assert len(client.t.rows) == _N
    assert all(r["is_synthetic"] is True for r in client.t.rows.values())
    out = capsys.readouterr().out
    assert "VERIFIED" in out and f"compared {_N} exported rows" in out


def test_main_execute_returns_nonzero_when_a_live_row_is_corrupted(tmp_path, monkeypatch, capsys):
    import scripts.load_csu_escalation_cohort as mod

    client = FakeClient()
    monkeypatch.setattr(mod, "_client", lambda: client)
    real_fetch_live_rows = mod.fetch_live_rows

    def _corrupting(c):
        rows = real_fetch_live_rows(c)
        if rows:
            rows[0] = {**rows[0], "age_at_index": (rows[0]["age_at_index"] or 0) + 999}
        return rows

    monkeypatch.setattr(mod, "fetch_live_rows", _corrupting)
    assert main(["--input", str(_write(tmp_path, _frame())), "--execute"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_main_execute_refuses_when_the_dataset_guard_is_absent(tmp_path, monkeypatch, capsys):
    """PR #2228 owner decision 2: the load must not run unless the dataset-level
    provenance guard (``_CAUSAL_SYNTHETIC_BACKED``) covers this dataset on the
    tree the loader runs from -- otherwise the deployed showcase flag would
    serve the planted rows as real. Refused BEFORE any upsert."""
    import scripts.load_csu_escalation_cohort as mod
    from src.api.routes.causal import datasets as datasets_mod

    client = FakeClient()
    monkeypatch.setattr(mod, "_client", lambda: client)
    monkeypatch.setattr(datasets_mod, "_CAUSAL_SYNTHETIC_BACKED", frozenset())
    assert main(["--input", str(_write(tmp_path, _frame())), "--execute"]) == 1
    assert client.t.upserts == []
    assert "GUARD" in capsys.readouterr().out


def test_main_execute_refuses_when_the_deployed_image_predates_the_guard(
    tmp_path, monkeypatch, capsys
):
    import scripts.load_csu_escalation_cohort as mod

    client = FakeClient()
    monkeypatch.setattr(mod, "_client", lambda: client)
    monkeypatch.setattr(_engine, "is_ancestor", lambda a, d: False)
    assert main(["--input", str(_write(tmp_path, _frame())), "--execute"]) == 1
    assert client.t.upserts == []
    assert "GUARD: REFUSED" in capsys.readouterr().out


def test_main_accepts_an_attested_deployed_commit(tmp_path, monkeypatch, capsys):
    import scripts.load_csu_escalation_cohort as mod

    client = FakeClient()
    monkeypatch.setattr(mod, "_client", lambda: client)
    monkeypatch.setattr(_engine, "deployed_image_commit", lambda container="e2i_api": None)
    argv = ["--input", str(_write(tmp_path, _frame())), "--execute", "--deployed-commit", _DEPLOYED]
    assert main(argv) == 0
    assert "VERIFIED" in capsys.readouterr().out


def test_load_env_is_not_run_at_import(monkeypatch):
    """codex r1 HIGH: importing the loader must not hand a test process the
    production credentials; .env is read only when a client is built."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(_engine))
    top_level_calls = {
        getattr(node.value.func, "id", getattr(node.value.func, "attr", None))
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    }
    assert not top_level_calls & {"load_dotenv", "load_env", "find_dotenv"}, top_level_calls
    calls = []
    monkeypatch.setattr(_engine, "load_env", lambda: calls.append("env"))
    monkeypatch.setattr(
        "src.memory.services.factories.get_supabase_client", lambda: "client", raising=False
    )
    assert _engine.get_client() == "client" and calls == ["env"]


def test_main_execute_refuses_when_the_planted_truth_seam_is_open(tmp_path, monkeypatch, capsys):
    import scripts.load_csu_escalation_cohort as mod
    from src.api.routes.causal import datasets as datasets_mod

    client = FakeClient()
    monkeypatch.setattr(mod, "_client", lambda: client)
    monkeypatch.setattr(datasets_mod, "PLANTED_TRUTH_RUN", True)
    assert main(["--input", str(_write(tmp_path, _frame())), "--execute"]) == 1
    assert client.t.upserts == []
    assert "GUARD" in capsys.readouterr().out
