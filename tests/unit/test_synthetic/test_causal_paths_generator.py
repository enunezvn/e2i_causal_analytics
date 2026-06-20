"""Shard 09 Task 5c (CM-003 / CM-005): minimal is_synthetic-tagged causal_paths rows
so causal_effect_size (CM-003) and mediators_identified (CM-005) are non-NULL.
data_split enum-exact; brand from brand_type; we do NOT touch the 50 stale real rows."""

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.causal_paths_generator import CausalPathsGenerator


def test_causal_paths_nonnull_effect_and_mediators_and_tagged():
    n = 12
    df = CausalPathsGenerator(GeneratorConfig(seed=8, n_records=n)).generate()
    # HCP rows are ADDITIVE (fixed 6-row block: 2 questions x 3 brands), so total
    # is n_records + 6 HCP + 6 trigger = n_records + 12 regardless of the n_records knob.
    assert len(df) == n + 12
    assert df["causal_effect_size"].notna().all()  # CM-003 non-NULL
    assert (
        df[df["grain"] != "trigger"]["mediators_identified"].apply(lambda m: len(m) >= 1).all()
    )  # CM-005 (trigger RCT edges are direct, no mediator)
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
    (control_group_flag -> action_taken, EMPTY backdoor) and the effect-modifier
    question (acceptance_status -> conversion_flag, EMPTY backdoor; priority is an
    effect modifier, not a confounder). Both per brand."""
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
        assert list(row["confounders_controlled"]) == []
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
