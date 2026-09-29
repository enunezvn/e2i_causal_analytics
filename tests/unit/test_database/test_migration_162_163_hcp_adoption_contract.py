"""Migrations 162 + 163 (#2286 / #2287): the HCP-adoption champions get a retrainable contract.

162 creates ``hcp_adoption_goldstd_v`` (hcp_brand_adoption LEFT JOIN hcp_profiles, the
frame ``FeatureBuilder._load_hcp_frame`` builds); 163 names it in ``cohort_data_source``
on the 3 ``hcp_adoption_<brand>_goldstd_lr_v1`` rows, with the ``synthetic_csu`` Layer-5
manifest (declared-safe HCP covariates; without it Layer 3 flags the DGP's designed
drivers and routes every retrain to LLM leakage remediation — measured, see the PR).

Hermetic tests read the FILES and drive the real consumers of the contract (registry
decode, ``has_cohort_contract``, the sweep's pipeline-input builder, the loader's target
guard, the manifest resolver). The opt-in real-DB tests (``E2I_DB_INTEGRATION=1``)
rehearse both migrations and both rollbacks on a throwaway container of prod's own
Postgres image, fronted by PostgREST at prod's own image tag, and prove the contract load
reproduces the goldstd builder's frame through the real loader and the real transport.
Nothing here touches ``supabase-db``.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import pytest

from src.mlops.gold_standard_eval.cohort_spec import BRANDS, make_hcp_spec
from src.repositories.ml_data_loader import ML_TABLES, PROVENANCE_TAGGED_TABLES
from src.services.cohort_contract import (
    contract_from_registry_row,
    decode_data_source,
    encode_data_source,
    training_provenance_from_contract,
)
from src.services.retraining_trigger import has_cohort_contract

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO / "database" / "migrations"
VIEW = "hcp_adoption_goldstd_v"
MANIFEST = "synthetic_csu"
MODELS = {f"hcp_adoption_{b.lower()}_goldstd_lr_v1": b for b in BRANDS}


def _one(pattern: str) -> Path:
    found = sorted(MIGRATIONS.glob(pattern))
    assert len(found) == 1, found
    return found[0]


M162 = _one("162_*.sql")
M163 = _one("163_*.sql")
R162 = MIGRATIONS / "rollback_162_hcp_adoption_goldstd_view.sql"
R163 = MIGRATIONS / "rollback_163_registry_cohort_contract_hcp_adoption.sql"


def _sql(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.strip().startswith("--")
    )


def expected_contract(brand: str) -> dict:
    """The contract the goldstd training code proves (make_hcp_spec + _load_hcp_frame)."""
    spec = make_hcp_spec(brand)
    return {
        "type": "table",
        "table": VIEW,
        "filters": {"brand": brand, "is_synthetic": True},
        "columns": list(spec.base_covariates) + [spec.label_column],
    }


_UPDATE_RE = re.compile(
    r"UPDATE ml_model_registry\s+SET (?P<set>.*?)\s+"
    r"WHERE model_name = '(?P<model>[^']+)'(?P<where>.*?);",
    re.S,
)


def _updates(path: Path) -> dict:
    """{model: {"set": {column: literal-or-None}, "where": text}} for every UPDATE."""
    out = {}
    for m in _UPDATE_RE.finditer(_sql(path)):
        sets = {}
        for col, lit, null in re.findall(r"(cohort_\w+) = (?:'([^']*)'|(NULL))", m["set"]):
            sets[col] = None if null else lit
        assert m["model"] not in out, m["model"]
        out[m["model"]] = {"set": sets, "where": m["where"]}
    return out


def _row_after_163(model: str) -> dict:
    """The registry row as 163 leaves it: its SET over 151's label."""
    return {"cohort_target_outcome": "adopted", **_updates(M163)[model]["set"]}


# --------------------------------------------------------------------------
# 162: the view
# --------------------------------------------------------------------------


def test_rollbacks_exist_and_are_skipped_by_the_runner() -> None:
    assert R162.name.startswith("rollback_") and R163.name.startswith("rollback_")
    assert R162.exists() and R163.exists()


def test_162_view_selects_the_goldstd_frame_and_nothing_wider() -> None:
    s = _sql(M162)
    body = s[s.index("SELECT") : s.index("FROM public.hcp_brand_adoption")]
    cols = [c.strip().split(".")[-1] for c in body[len("SELECT") :].split(",")]
    spec = make_hcp_spec("Kisqali")
    assert set(cols) == {
        "hcp_id",
        "brand",
        "consideration_date",
        "adopted",
        "data_split",
        "is_synthetic",
    } | set(spec.base_covariates)
    # provenance, split and label are the ADOPTION row's, covariates the profile's.
    for col in ("adopted", "data_split", "is_synthetic", "brand"):
        assert f"a.{col}" in body
    for col in spec.base_covariates:
        assert f"p.{col}" in body
    assert "treatment_arm" not in body
    # A PostgREST FK embed is a LEFT join; so is the view.
    assert re.search(r"LEFT JOIN public\.hcp_profiles p ON p\.hcp_id = a\.hcp_id", s)


def test_162_view_is_security_invoker_and_service_role_only() -> None:
    s = _sql(M162)
    assert "WITH (security_invoker = true)" in s
    assert re.search(
        r"REVOKE ALL ON public\.hcp_adoption_goldstd_v FROM PUBLIC, anon, authenticated;", s
    )
    assert re.search(r"GRANT SELECT ON public\.hcp_adoption_goldstd_v TO service_role;", s)
    assert "NOTIFY pgrst, 'reload schema';" in s
    upper = s.upper()
    assert "\nBEGIN;" not in upper and "\nCOMMIT;" not in upper


def test_rollback_162_refuses_while_a_contract_names_the_view() -> None:
    s = _sql(R162)
    assert s.index("RAISE EXCEPTION") < s.index("DROP VIEW IF EXISTS public.hcp_adoption_goldstd_v")
    assert f"DELETE FROM public.schema_migrations WHERE filename = '{M162.name}'" in s


# --------------------------------------------------------------------------
# 163: the contract, driven through its real consumers
# --------------------------------------------------------------------------


def test_163_sets_the_pair_on_exactly_the_three_hcp_rows_compare_and_set() -> None:
    ups = _updates(M163)
    assert set(ups) == set(MODELS)
    for model, up in ups.items():
        # data_source + manifest as ONE unit; the label is 151's and never rewritten.
        assert set(up["set"]) == {"cohort_data_source", "cohort_feature_manifest_source"}
        where = up["where"]
        assert "is_synthetic = false" in where
        assert "cohort_data_source IS NULL" in where
        assert "cohort_feature_manifest_source IS NULL" in where
        assert "cohort_target_outcome = 'adopted'" in where


@pytest.mark.parametrize("model", sorted(MODELS))
def test_163_literal_is_the_encoded_goldstd_contract(model: str) -> None:
    lit = _updates(M163)[model]["set"]["cohort_data_source"]
    contract = expected_contract(MODELS[model])
    assert lit == encode_data_source(contract)
    assert decode_data_source(lit) == contract
    assert contract["table"] in ML_TABLES
    assert contract["table"] in PROVENANCE_TAGGED_TABLES


@pytest.mark.parametrize("model", sorted(MODELS))
def test_the_backfilled_row_passes_has_cohort_contract_and_a_half_filled_one_does_not(
    model: str,
) -> None:
    row = _row_after_163(model)
    assert has_cohort_contract(contract_from_registry_row(row)) is True
    # Today's row (migration 151: label only) and the mirror half are both refused.
    today = {"cohort_target_outcome": "adopted"}
    assert has_cohort_contract(contract_from_registry_row(today)) is False
    half = {"cohort_data_source": row["cohort_data_source"]}
    assert has_cohort_contract(contract_from_registry_row(half)) is False


@pytest.mark.parametrize("model", sorted(MODELS))
def test_the_sweeps_retrain_input_pins_label_brand_and_manifest(model: str) -> None:
    from src.agents.ml_foundation.data_preparer.nodes.data_loader import (
        _require_target_in_columns,
    )
    from src.agents.ml_foundation.scope_definer.nodes.problem_classifier import (
        resolve_target_variable_hint,
    )
    from src.data.manifests.resolution import resolve_manifest_source
    from src.tasks.drift_monitoring_tasks import _cohort_input_from_training_config

    contract = contract_from_registry_row(_row_after_163(model))
    input_data = _cohort_input_from_training_config(dict(contract))
    assert input_data["data_source"] == expected_contract(MODELS[model])
    assert input_data["brand"] == MODELS[model]
    # the pipeline's scope stage resolves it exactly like this (tier_0/pipeline.py)
    assert (
        resolve_manifest_source(input_data["data_source"], input_data["feature_manifest_source"])
        == MANIFEST
    )
    target = resolve_target_variable_hint(input_data)
    assert target == "adopted"
    _require_target_in_columns(input_data["data_source"]["columns"], target)
    # The rewrite #2284 removed would still be caught loud by the guard.
    with pytest.raises(ValueError, match="omit the target"):
        _require_target_in_columns(input_data["data_source"]["columns"], "will_adopt")


def test_every_contract_covariate_is_declared_pre_index_by_the_manifest() -> None:
    """The manifest is only a correct declaration if it covers every covariate the
    contract selects; an undeclared one would still reach Layer 3 unprotected."""
    from src.data.manifests import lookup_feature_contract

    for up in _updates(M163).values():
        assert up["set"]["cohort_feature_manifest_source"] == MANIFEST
        cols = decode_data_source(up["set"]["cohort_data_source"])["columns"]
        for col in cols[:-1]:
            fc = lookup_feature_contract(col, MANIFEST)
            assert fc is not None, col
            assert fc.knowable_at.reference in ("index_date", "enrollment"), (col, fc)
        assert lookup_feature_contract(cols[-1], MANIFEST) is None  # the label is no feature


def test_the_contract_records_synthetic_gold_provenance() -> None:
    for up in _updates(M163).values():
        contract = decode_data_source(up["set"]["cohort_data_source"])
        assert training_provenance_from_contract(contract) == "synthetic_gold"


def test_the_contract_selects_no_post_outcome_column() -> None:
    from src.mlops.gold_standard_eval.feature_builder import LEAKAGE_DENYLIST

    for up in _updates(M163).values():
        cols = decode_data_source(up["set"]["cohort_data_source"])["columns"]
        assert [c for c in cols if c in LEAKAGE_DENYLIST] == []
        assert "consideration_date" not in cols and "hcp_id" not in cols


def test_rollback_163_restores_null_only_on_163s_own_pair() -> None:
    ups, rolls = _updates(M163), _updates(R163)
    assert set(rolls) == set(MODELS)
    for model, roll in rolls.items():
        assert roll["set"] == {"cohort_data_source": None, "cohort_feature_manifest_source": None}
        written = ups[model]["set"]
        assert f"cohort_data_source = '{written['cohort_data_source']}'" in roll["where"]
        assert f"cohort_feature_manifest_source = '{MANIFEST}'" in roll["where"]
    assert f"DELETE FROM public.schema_migrations WHERE filename = '{M163.name}'" in _sql(R163)
    assert "cohort_target_outcome" not in _sql(R163)


# --------------------------------------------------------------------------
# Real Postgres + real PostgREST (opt-in)
# --------------------------------------------------------------------------

pg_only = pytest.mark.timeout(600)


@pytest.fixture(scope="module")
def throwaway_pg() -> Iterator[object]:
    from tests.unit.test_database._hcp_adoption_pg import OPT_IN, prod_image

    if os.environ.get(OPT_IN) != "1":
        pytest.skip(
            f"real-DB rehearsal: opt-in with {OPT_IN}=1 (docker, prod's Postgres image); "
            "a skipped real-DB test is not coverage"
        )
    image = prod_image("supabase-db")
    if image is None:
        pytest.skip("docker or the supabase-db container is not reachable")
    from tests.unit.test_database.learning_loop._pg import ThrowawayPg

    pg = ThrowawayPg(image=image)
    pg.start()
    try:
        yield pg
    finally:
        pg.stop()


def _fresh(pg, name: str):
    from tests.unit.test_database._hcp_adoption_pg import build_base
    from tests.unit.test_database.learning_loop._pg import PgConn

    pg.rows("postgres", f"CREATE DATABASE {name} OWNER postgres")
    conn = PgConn(pg, name)
    build_base(conn)
    return conn


def _registry_rows(conn) -> dict:
    """{model_name: (source, target, manifest)}; synthetic rows keyed ``<name>#synthetic``."""
    out = {}
    for line in conn.rows(
        "select model_name || case when is_synthetic then '#synthetic' else '' end || '|' || "
        "coalesce(cohort_data_source, '<NULL>') || '|' || "
        "coalesce(cohort_target_outcome, '<NULL>') || '|' || "
        "coalesce(cohort_feature_manifest_source, '<NULL>') from ml_model_registry order by 1"
    ):
        name, source, target, manifest = line.split("|", 3)
        assert name not in out, name
        out[name] = (source, target, manifest)
    return out


def _seed_registry(conn) -> None:
    values = ",".join(f"('{m}', 'production', false, 'adopted')" for m in MODELS)
    conn.execute(
        "INSERT INTO ml_model_registry (model_name, stage, is_synthetic, cohort_target_outcome) "
        f"VALUES {values}, "
        # controls: a synthetic twin of a goldstd row, and an unrelated model
        "('hcp_adoption_kisqali_goldstd_lr_v1', 'archived', true, 'adopted'), "
        "('other_model', 'production', false, 'adopted')",
        user="postgres",
    )


def _run(pg, conn, path: Path):
    return pg.run_script(conn.db, path.read_bytes(), single_transaction=True, user="postgres")


@pg_only
def test_real_db_162_then_163_apply_twice_and_both_roll_back(throwaway_pg) -> None:
    from tests.unit.test_database._hcp_adoption_pg import apply_file

    conn = _fresh(throwaway_pg, "t162_apply")
    _seed_registry(conn)
    before = _registry_rows(conn)

    for _ in range(2):  # idempotent: the second pass changes nothing
        apply_file(conn, M162)
        apply_file(conn, M163)
    after = _registry_rows(conn)
    for model, brand in MODELS.items():
        assert after[model] == (
            encode_data_source(expected_contract(brand)),
            "adopted",
            MANIFEST,
        )
    assert after["other_model"] == before["other_model"]
    synth = "hcp_adoption_kisqali_goldstd_lr_v1#synthetic"
    assert after[synth] == before[synth] == ("<NULL>", "adopted", "<NULL>")
    assert conn.rows(
        "select reloptions::text from pg_class where relname = 'hcp_adoption_goldstd_v'"
    ) == ["{security_invoker=true}"]
    grants = conn.rows(
        "select grantee || ':' || privilege_type from information_schema.role_table_grants "
        "where table_name = 'hcp_adoption_goldstd_v' and grantee <> 'postgres' order by 1"
    )
    assert grants == ["service_role:SELECT"]

    # rollback order: 162 refuses while a contract names the view
    proc = _run(throwaway_pg, conn, R162)
    assert proc.returncode != 0 and b"apply rollback_163 first" in proc.stderr
    assert conn.rows("select count(*) from pg_views where viewname = 'hcp_adoption_goldstd_v'") == [
        "1"
    ]

    for path in (R163, R162):
        proc = _run(throwaway_pg, conn, path)
        assert proc.returncode == 0, proc.stderr.decode()
    assert _registry_rows(conn) == before  # NULL restored, label untouched
    assert conn.rows("select count(*) from pg_views where viewname = 'hcp_adoption_goldstd_v'") == [
        "0"
    ]
    assert conn.rows(
        f"select count(*) from schema_migrations where filename in ('{M162.name}', '{M163.name}')"
    ) == ["0"]


@pg_only
def test_real_db_163_never_overwrites_a_healed_contract_or_composes_a_mixed_pair(
    throwaway_pg,
) -> None:
    from tests.unit.test_database._hcp_adoption_pg import apply_file

    conn = _fresh(throwaway_pg, "t163_cas")
    healed = json.dumps({"type": "table", "table": "somewhere"}, sort_keys=True)
    conn.execute(
        "INSERT INTO ml_model_registry (model_name, is_synthetic, cohort_data_source, "
        "cohort_target_outcome, cohort_feature_manifest_source) VALUES "
        # a contract healed by a completed retrain
        f"('hcp_adoption_kisqali_goldstd_lr_v1', false, '{healed}', 'adopted', NULL), "
        # a label that no longer says 'adopted'
        "('hcp_adoption_fabhalta_goldstd_lr_v1', false, NULL, 'will_adopt', NULL), "
        # a manifest set by hand: writing only data_source would compose a mixed pair
        "('hcp_adoption_remibrutinib_goldstd_lr_v1', false, NULL, 'adopted', 'optum_hcp')",
        user="postgres",
    )
    before = _registry_rows(conn)
    apply_file(conn, M162)
    apply_file(conn, M163)
    assert _registry_rows(conn) == before
    # and rollback_163 leaves all of them alone
    assert _run(throwaway_pg, conn, R163).returncode == 0
    assert _registry_rows(conn) == before


@pytest.fixture(scope="module")
def served_cohort(throwaway_pg) -> Iterator[object]:
    """The seeded cohort behind real PostgREST: (server, conn, seed counts)."""
    from tests.unit.test_database._hcp_adoption_pg import (
        PostgrestServer,
        apply_file,
        prod_image,
        seed,
    )

    conn = _fresh(throwaway_pg, "t162_frame")
    counts = seed(conn)
    apply_file(conn, M162)
    image = prod_image("supabase-rest")
    if image is None:
        pytest.skip("the supabase-rest container is not reachable (image tag unknown)")
    server = PostgrestServer(image=image, pg=throwaway_pg, db=conn.db)
    server.start()
    try:
        yield server, conn, counts
    finally:
        server.stop()


@pg_only
def test_real_postgrest_serves_the_view_to_service_role_only(served_cohort) -> None:
    from postgrest.exceptions import APIError

    server, _, _ = served_cohort
    page = (
        server.client("service_role")
        .from_(VIEW)
        .select("adopted,data_split")
        .eq("brand", "Kisqali")
        .eq("is_synthetic", True)
        .order("hcp_id")
        .range(4000, 5999)
        .execute()
    )
    assert len(page.data) == 1000  # the tail page of a 5,000-row brand
    for role in (None, "authenticated"):
        with pytest.raises(APIError) as err:
            server.client(role).from_(VIEW).select("adopted").limit(1).execute()
        assert err.value.code == "42501"


@pg_only
@pytest.mark.asyncio
@pytest.mark.parametrize("brand", BRANDS)
async def test_real_contract_load_reproduces_the_goldstd_builder_frame(
    served_cohort, brand: str, monkeypatch
) -> None:
    """The retrain's frame == the frame the champion was trained on, through the real
    loader node, the real MLDataLoader and real PostgREST, per brand."""
    import pandas as pd

    from src.agents.ml_foundation.data_preparer.nodes import data_loader
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
    from src.repositories.ml_data_loader import MLDataLoader
    from tests.unit.test_database._hcp_adoption_pg import SupabaseShim

    server, _, _ = served_cohort
    spec = make_hcp_spec(brand)
    goldstd = await FeatureBuilder(spec)._load_hcp_frame(
        SupabaseShim(server.client("service_role", is_async=True))
    )

    loader = MLDataLoader(supabase_client=SupabaseShim(server.client("service_role")))
    monkeypatch.setattr(data_loader, "get_ml_data_loader", lambda: loader)
    contract = contract_from_registry_row(
        _row_after_163(f"hcp_adoption_{brand.lower()}_goldstd_lr_v1")
    )
    out = await data_loader.load_data(
        {
            "experiment_id": "t2287",
            "data_source": contract["data_source"],
            "scope_spec": {"prediction_target": "adopted"},
        }
    )
    assert "error" not in out, out.get("error")
    # data_split resolved: the precomputed-split path ran (the temporal path has no holdout)
    assert out["holdout_df"] is not None and len(out["holdout_df"]) > 0
    splits = {
        "train": out["train_df"],
        "validation": out["validation_df"],
        "test": out["test_df"],
        "holdout": out["holdout_df"],
    }
    for name, frame in splits.items():
        assert set(frame["data_split"]) == {name}

    loaded = pd.concat(list(splits.values()), ignore_index=True)
    cols = list(spec.base_covariates) + ["adopted", "data_split"]
    assert set(loaded.columns) == set(cols)
    assert len(loaded) == len(goldstd) == 5000

    def _rows(df):
        return Counter(
            tuple(None if pd.isna(v) else v for v in r)
            for r in df[cols].itertuples(index=False, name=None)
        )

    assert _rows(loaded) == _rows(goldstd)
    assert loaded["adopted"].mean() == pytest.approx(goldstd["adopted"].mean())
    assert Counter(loaded["data_split"]) == Counter(goldstd["data_split"])
