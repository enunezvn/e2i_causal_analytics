"""Shard 09 Task 5c (CM-003 / CM-005): minimal is_synthetic-tagged causal_paths rows
so causal_effect_size (CM-003) and mediators_identified (CM-005) are non-NULL.
data_split enum-exact; brand from brand_type; we do NOT touch the 50 stale real rows."""

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.causal_paths_generator import (
    N_BRAND_CLINICAL_ROWS,
    N_COMM_ARM_ROWS,
    N_COMMERCIAL_ROWS,
    CausalPathsGenerator,
)

# Grains whose edges are DIRECT (1-hop, no mediator): the trigger RCT/effect-
# modifier edges, the patient-grain commercial-arm edges, and the brand-distinct
# clinical-axis edges (#1321 — one per brand).
_DIRECT_EDGE_GRAINS = ("trigger", "patient_arm", "patient_clinical")


def test_causal_paths_nonnull_effect_and_mediators_and_tagged():
    n = 12
    df = CausalPathsGenerator(GeneratorConfig(seed=8, n_records=n)).generate()
    # HCP/trigger/commercial/patient-arm rows are ADDITIVE fixed blocks (6 HCP +
    # 6 trigger + N_COMMERCIAL_ROWS commercial + N_COMM_ARM_ROWS patient-arm),
    # independent of the n_records knob.
    assert len(df) == n + 12 + N_COMMERCIAL_ROWS + N_COMM_ARM_ROWS + N_BRAND_CLINICAL_ROWS
    assert df["causal_effect_size"].notna().all()  # CM-003 non-NULL
    assert (
        df[~df["grain"].isin(_DIRECT_EDGE_GRAINS)]["mediators_identified"]
        .apply(lambda m: len(m) >= 1)
        .all()
    )  # CM-005 (trigger + patient-arm edges are direct, no mediator)
    assert df["is_synthetic"].all()
    # data_split enum-exact (faithful: train/validation/test/holdout/unassigned)
    assert set(df["data_split"]).issubset({"holdout", "test", "train", "unassigned", "validation"})
    assert set(df["brand"]).issubset({"Remibrutinib", "Kisqali", "Fabhalta"})
    # NOT-NULL columns on the faithful DB must be present + populated
    for col in ("path_id", "discovery_date", "causal_chain", "created_at", "confirmation_count"):
        assert df[col].notna().all()


def test_causal_paths_cover_all_three_gold_standard_cohort_outcomes():
    """The KG must carry persistence + discontinuation chains, not just
    initiation. The gold standard validates treatment_arm -> persistent_180d and
    -> discontinued_180d (src/mlops/gold_standard_eval/cohort_spec.py;
    scripts/validate_synthetic_causal.py), but causal_paths only ever emitted
    treatment_initiated, so "Discover chains in KG" returned nothing for the
    persistent_180d default outcome."""
    df = CausalPathsGenerator(GeneratorConfig(seed=3, n_records=30)).generate()
    end_nodes = set(df["end_node"])
    assert {"treatment_initiated", "persistent_180d", "discontinued_180d"} <= end_nodes
    # Every chain starts at the treatment arm and terminates at its end_node,
    # and causal_chain.nodes agrees with start/end (so the FalkorDB sync builds a
    # correct (:Variable treatment_arm)-[:CAUSES]->(:Variable <outcome>) path).
    # Patient chains all start at treatment_arm; HCP chains (end_node 'adopted')
    # start at peer_influence_score / treatment_arm and are asserted separately.
    patient = df[df["grain"] == "patient"]
    assert (patient["start_node"] == "treatment_arm").all()
    for _, row in patient.iterrows():
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == "treatment_arm"
        assert nodes[-1] == row["end_node"]
        # No repeated nodes (would be dropped by _clean_causal_chains).
        assert len(nodes) == len(set(nodes))


def test_all_brand_outcome_cells_emitted():
    """Every (brand x outcome) cell must appear — not just the i%3 diagonal.
    HCP rows (end_node='adopted') are asserted separately; exclude them here."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=27)).generate()
    patient = df[df["grain"] == "patient"]
    cells = set(zip(patient["brand"], patient["end_node"], strict=True))
    brands = {"Remibrutinib", "Kisqali", "Fabhalta"}
    outcomes = {"treatment_initiated", "persistent_180d", "discontinued_180d"}
    assert cells == {(b, o) for b in brands for o in outcomes}


def test_confounders_match_modeled_set_per_outcome():
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=27)).generate()
    row = df[(df.brand == "Kisqali") & (df.end_node == "persistent_180d")].iloc[0]
    assert set(row["confounders_controlled"]) == {
        "disease_severity",
        "academic_hcp",
        "geographic_region",
    }
    row2 = df[df.end_node == "treatment_initiated"].iloc[0]
    assert set(row2["confounders_controlled"]) == {"disease_severity", "age_at_diagnosis"}


def test_all_rows_tagged_patient_grain():
    """Patient rows carry grain='patient' (shared convention; HCP/trigger phases add their own).
    The column is stripped by batch_loader before DB insert (not in causal_paths TABLE_COLUMNS)."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=27)).generate()
    patient = df[df["grain"] == "patient"]
    assert set(patient["grain"]) == {"patient"}


def test_hcp_adoption_edges_emitted_per_brand():
    """The SSOT must carry BOTH HCP questions for EVERY brand so the HCP-grain
    leaderboard enumerates them the same way as patient edges:
      peer_influence_score -> adopted (EMPTY backdoor, exogenous root)
      treatment_arm        -> adopted (adjust {centrality_z})."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=12)).generate()
    hcp = df[df["end_node"] == "adopted"]
    cells = set(zip(hcp["start_node"], hcp["brand"], strict=False))
    brands = {"Remibrutinib", "Kisqali", "Fabhalta"}
    assert cells == {(s, b) for s in ("peer_influence_score", "treatment_arm") for b in brands}


def test_hcp_adoption_confounder_sets_are_modeled():
    """peer_influence_score is exogenous (EMPTY backdoor); treatment_arm adjusts
    for centrality_z. These are the SSOT adjustment sets the loader will honor."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=12)).generate()
    exo = df[(df["start_node"] == "peer_influence_score") & (df["end_node"] == "adopted")].iloc[0]
    assert list(exo["confounders_controlled"]) == []
    rep = df[(df["start_node"] == "treatment_arm") & (df["end_node"] == "adopted")].iloc[0]
    assert list(rep["confounders_controlled"]) == ["centrality_z"]


def test_hcp_adoption_chain_is_clean_two_hop():
    """HCP chains terminate at adopted with a non-empty mediator list (existing
    invariant) and causal_chain.nodes starts at the treatment, ends at adopted."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=12)).generate()
    for _, row in df[df["end_node"] == "adopted"].iterrows():
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == row["start_node"]
        assert nodes[-1] == "adopted"
        assert len(row["mediators_identified"]) >= 1
        assert "adopted" not in row["mediators_identified"]
        assert row["start_node"] not in row["mediators_identified"]


def test_trigger_grain_edges_emitted_with_empty_and_modeled_backdoor():
    """The trigger grain seeds two SSOT edges: the randomized RCT
    (control_group_flag -> action_taken, EMPTY backdoor) and the acceptance
    question (acceptance_status -> conversion_flag). Since COMM-ARMS Phase 4
    planted trigger_accepted with {disease_severity, engagement_score}
    confounders, acceptance is CONFOUNDED and its SSOT edge must model that
    backdoor (an empty set would tell the estimator adjustment is unnecessary).
    Both per brand."""
    df = CausalPathsGenerator(GeneratorConfig(seed=11, n_records=30)).generate()
    edges = set(zip(df["start_node"], df["end_node"], strict=False))
    assert ("control_group_flag", "action_taken") in edges
    assert ("acceptance_status", "conversion_flag") in edges

    rct = df[(df.start_node == "control_group_flag") & (df.end_node == "action_taken")]
    assert len(rct) >= 1
    # The RCT is randomized -> its modeled backdoor set is empty.
    for _, row in rct.iterrows():
        assert list(row["confounders_controlled"]) == []
        assert row["grain"] == "trigger"
        # causal_chain stays a direct edge (no mediators injected for the RCT).
        assert row["causal_chain"]["nodes"] == ["control_group_flag", "action_taken"]
    # Trigger edges exist for every brand (so a brand-scoped leaderboard sees them).
    rct_brands = set(rct["brand"])
    assert {"Remibrutinib", "Kisqali", "Fabhalta"} <= rct_brands

    mod = df[(df.start_node == "acceptance_status") & (df.end_node == "conversion_flag")]
    assert len(mod) >= 1
    for _, row in mod.iterrows():
        # Phase 4's trigger_accepted arm confounders — the backdoor the DGP plants.
        assert list(row["confounders_controlled"]) == ["disease_severity", "engagement_score"]
        assert row["grain"] == "trigger"


def test_patient_edges_retain_treatment_arm_start_and_grain():
    """Patient-grain rows are unchanged: they still start at treatment_arm and are
    tagged grain='patient' (the trigger edges must not perturb the patient cells)."""
    df = CausalPathsGenerator(GeneratorConfig(seed=11, n_records=30)).generate()
    patient = df[df["grain"] == "patient"]
    assert (patient["start_node"] == "treatment_arm").all()
    assert set(patient["end_node"]) <= {
        "treatment_initiated",
        "persistent_180d",
        "discontinued_180d",
    }


# ---------------------------------------------------------------------------
# Patient-grain commercial-arm edges (2026-07-23): surface the five planted
# commercial levers on the discovery leaderboard (they were estimable but had no
# causal_paths edges, so only treatment_arm ever appeared). Direct 1-hop edges,
# content-addressed for idempotent targeted upsert (no full reseed).
# ---------------------------------------------------------------------------

_EXPECTED_ARM_OUTCOMES = {
    "copay_support": {"adherent_180d", "low_gap_180d", "persistent_180d"},
    "psp_enrolled": {"adherent_180d", "persistent_180d"},
    "rep_detailing_high": {"treatment_initiated"},
    "sample_dropped": {"treatment_initiated"},
    "trigger_accepted": {"treatment_initiated"},
}


def test_comm_arm_edges_surface_all_five_levers_per_brand():
    df = CausalPathsGenerator(GeneratorConfig(seed=7, n_records=20)).generate()
    arm_rows = df[df["grain"] == "patient_arm"]
    # Every (arm -> outcome) pair present, for every brand, as a direct 1-hop edge.
    for arm, outcomes in _EXPECTED_ARM_OUTCOMES.items():
        got = set(arm_rows[arm_rows["start_node"] == arm]["end_node"])
        assert got == outcomes, f"{arm}: expected {outcomes}, got {got}"
    for _, row in arm_rows.iterrows():
        assert row["path_length"] == 1
        assert list(row["mediators_identified"]) == []
        assert row["causal_chain"]["nodes"] == [row["start_node"], row["end_node"]]
    per_brand = {
        b: set(zip(g["start_node"], g["end_node"], strict=False))
        for b, g in arm_rows.groupby("brand")
    }
    assert set(per_brand) == {"Remibrutinib", "Kisqali", "Fabhalta"}
    assert all(len(edges) == 8 for edges in per_brand.values())  # 8 arm->outcome edges each


def test_comm_arm_edge_confounders_match_arm_registry_ssot():
    """Each edge must carry the arm's EXACT DGP backdoor set AND point only at the
    arm's planted target_outcomes (ARM_REGISTRY is the SSOT). A hardcoded set that
    drifts from the DGP would tell the estimator to adjust for the wrong columns
    → a confounded estimate, or surface a lever→outcome the DGP never planted.
    This locks BOTH invariants to the live registry (not a local mirror)."""
    from collections import defaultdict

    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY
    from src.ml.synthetic.generators.causal_paths_generator import _COMM_ARM_EDGES

    edge_outcomes = defaultdict(set)
    for arm, outcome, confounders, _lo, _hi in _COMM_ARM_EDGES:
        assert set(confounders) == set(ARM_REGISTRY[arm].confounders), (
            f"{arm} edge confounders {sorted(confounders)} != ARM_REGISTRY "
            f"{sorted(ARM_REGISTRY[arm].confounders)}"
        )
        assert outcome in ARM_REGISTRY[arm].target_outcomes, (
            f"{arm} edge outcome {outcome!r} not in ARM_REGISTRY target_outcomes "
            f"{tuple(ARM_REGISTRY[arm].target_outcomes)}"
        )
        edge_outcomes[arm].add(outcome)
    # Completeness: edges surface EXACTLY the planted target_outcomes per arm —
    # none spurious (subset above), none missing (equality here).
    for arm, outcomes in edge_outcomes.items():
        assert outcomes == set(ARM_REGISTRY[arm].target_outcomes), (
            f"{arm} edges cover {sorted(outcomes)} != target_outcomes "
            f"{sorted(ARM_REGISTRY[arm].target_outcomes)}"
        )


def test_comm_arm_rows_for_upsert_are_content_addressed_and_idempotent():
    from src.ml.synthetic.generators.causal_paths_generator import comm_arm_rows_for_upsert

    a = comm_arm_rows_for_upsert()
    b = comm_arm_rows_for_upsert()
    ids_a = [r["path_id"] for r in a]
    ids_b = [r["path_id"] for r in b]
    assert ids_a == ids_b  # stable across calls (content-addressed)
    assert len(ids_a) == len(set(ids_a))  # no collisions
    assert all(pid.startswith("scp_a") for pid in ids_a)  # arm namespace
    assert len(a) == N_COMM_ARM_ROWS
    # The generator-only 'grain' column is projected out for the DB insert.
    assert all("grain" not in r for r in a)


# ---------------------------------------------------------------------------
# Brand-distinct clinical-axis edges (#1321): ONE per brand — the acceptance criterion
# for a divergent variable set. Each axis column is 100% NULL off-brand, so no other
# brand's KG gains the node.
# ---------------------------------------------------------------------------

# (brand, axis treatment node) — each edge is emitted for its brand ONLY.
_EXPECTED_AXES = {
    "Fabhalta": "complement_inhibitor_status",
    "Kisqali": "disease_stage",
    "Remibrutinib": "urticaria_severity_uas7",
}


# Per-brand persistence-effect sign: Fabhalta/Kisqali axis=1 REDUCES persistence
# (negative); Remibrutinib is INVERTED (2026-07-28) — uncontrolled CSU is stickier, so
# axis=1 INCREASES persistence (positive).
_EXPECTED_AXIS_SIGN = {"Fabhalta": -1, "Kisqali": -1, "Remibrutinib": +1}


def test_brand_clinical_axes_are_distinct_and_signed():
    """Each brand carries its OWN axis -> persistent_180d edge (a direct 1-hop edge with
    the disease_severity precision covariate, signed per _EXPECTED_AXIS_SIGN) and NO OTHER
    brand's axis. That mutual brand-distinctness IS the #1321 acceptance criterion."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=20)).generate()
    clin = df[df["grain"] == "patient_clinical"]
    assert len(clin) == N_BRAND_CLINICAL_ROWS == 3
    assert dict(zip(clin["brand"], clin["start_node"], strict=True)) == _EXPECTED_AXES
    for _, row in clin.iterrows():
        assert row["end_node"] == "persistent_180d"
        assert row["path_length"] == 1
        assert list(row["mediators_identified"]) == []
        assert row["causal_chain"]["nodes"] == [row["start_node"], "persistent_180d"]
        assert list(row["confounders_controlled"]) == ["disease_severity"]
        sign = _EXPECTED_AXIS_SIGN[row["brand"]]
        assert sign * row["causal_effect_size"] > 0, (
            f"{row['brand']} axis effect {row['causal_effect_size']:+.4f} not in dir {sign}"
        )
    # Mutual brand-distinctness: each axis node belongs to EXACTLY its own brand.
    for brand, axis in _EXPECTED_AXES.items():
        carriers = set(df[df["start_node"] == axis]["brand"])
        assert carriers == {brand}, f"{axis} leaked to {carriers - {brand}}"


def test_clinical_axis_rows_for_upsert_are_content_addressed_and_brand_filterable():
    from src.ml.synthetic.generators.causal_paths_generator import (
        clinical_axis_rows_for_upsert,
        fabhalta_clinical_rows_for_upsert,
    )

    a = clinical_axis_rows_for_upsert()
    b = clinical_axis_rows_for_upsert()
    ids_a = [r["path_id"] for r in a]
    assert ids_a == [r["path_id"] for r in b]  # stable across calls (content-addressed)
    assert len(ids_a) == len(set(ids_a)) == N_BRAND_CLINICAL_ROWS == 3
    assert all(pid.startswith("scp_f") for pid in ids_a)  # clinical-axis family namespace
    assert {r["brand"] for r in a} == set(_EXPECTED_AXES)
    assert all("grain" not in r for r in a)  # generator-only column projected out
    # Brand filtering: seed only the new brands (Fabhalta already live in prod).
    new = clinical_axis_rows_for_upsert(["Kisqali", "Remibrutinib"])
    assert {r["brand"] for r in new} == {"Kisqali", "Remibrutinib"}
    # Back-compat wrapper still returns the single Fabhalta pilot row.
    fab = fabhalta_clinical_rows_for_upsert()
    assert len(fab) == 1 and fab[0]["brand"] == "Fabhalta"


def test_cohort_hcp_trigger_path_ids_are_content_addressed_and_idempotent_1725():
    """#1725: the patient/hcp/trigger grains minted RANDOM uuid4 path_ids, so
    the loader's conflict-on-PK upsert inserted a fresh copy of every path on
    each reseed — measured live 2026-08-19: 2,657 rows over just 21 distinct
    (start, end, brand) identities (~126x). The commercial/arm/clinical
    families were already content-addressed; these three are now too.

    Namespaces: scp_p (patient), scp_h (hcp), scp_t (trigger) — 'p'/'h'/'t'
    are NON-hex, so the id space is disjoint from both the legacy uuid family
    (scp_ + 13 hex) and the scp_c/scp_a/scp_f content families (hex letters),
    which is what lets the one-off cleanup target the legacy rows by pattern.
    """
    import re

    df_a = CausalPathsGenerator(GeneratorConfig(n_records=25)).generate()
    df_b = CausalPathsGenerator(GeneratorConfig(n_records=25)).generate()

    for grain, prefix in (("patient", "scp_p"), ("hcp", "scp_h"), ("trigger", "scp_t")):
        ids_a = df_a[df_a["grain"] == grain]["path_id"].tolist()
        ids_b = df_b[df_b["grain"] == grain]["path_id"].tolist()
        assert ids_a, f"no {grain} rows generated"
        # Deterministic across runs (the upsert updates in place on reseed).
        assert ids_a == ids_b, f"{grain} path_ids differ across identical runs"
        # No collisions within a run.
        assert len(ids_a) == len(set(ids_a)), f"{grain} path_id collision"
        # Namespaced, 16 chars, fits varchar(20).
        assert all(re.fullmatch(rf"{prefix}[0-9a-f]{{11}}", pid) for pid in ids_a), (
            f"{grain} ids not in the {prefix} namespace: {ids_a[:3]}"
        )

    # The legacy uuid4 shape must be gone entirely.
    legacy = [pid for pid in df_a["path_id"] if re.fullmatch(r"scp_[0-9a-f]{13}", pid)]
    assert legacy == [], f"legacy uuid-family ids still minted: {legacy[:3]}"


def test_patient_grain_ids_distinct_per_occurrence_1725():
    """n_records cycles the 9 (brand x outcome) cells, so the same cell recurs
    (occurrence k = i // 9). Each occurrence must keep a DISTINCT deterministic
    id — content-addressing must not collapse the generation contract."""
    n = 25  # 9 cells -> occurrences 0..2 for the first 7 cells
    df = CausalPathsGenerator(GeneratorConfig(n_records=n)).generate()
    patient_ids = df[df["grain"] == "patient"]["path_id"].tolist()
    assert len(patient_ids) == n
    assert len(set(patient_ids)) == n


def test_cohort_hcp_trigger_rows_for_upsert_shape_1725():
    """Targeted apply helper for the #1725 cleanup script: DB-shaped records for
    the patient/hcp/trigger grains ONLY, projected to the loader's causal_paths
    columns (no generator-only 'grain' key), ids in the new content-addressed
    namespaces, and identity-stable across calls so the on-PK upsert is
    idempotent (row count can never grow on re-run)."""
    import re

    from src.ml.synthetic.generators.causal_paths_generator import (
        cohort_hcp_trigger_rows_for_upsert,
    )

    records = cohort_hcp_trigger_rows_for_upsert()
    # 25 patient (canonical load_synthetic_data count) + 6 hcp + 6 trigger.
    assert len(records) == 37
    ids = [r["path_id"] for r in records]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"scp_[pht][0-9a-f]{11}", pid) for pid in ids), ids[:3]
    assert all("grain" not in r for r in records)
    assert all(r["is_synthetic"] is True for r in records)
    # Identity-stable: a second call emits the same PK set (values may restamp).
    assert set(ids) == {r["path_id"] for r in cohort_hcp_trigger_rows_for_upsert()}
