"""P1b — HCP shared-patient network + exogenous centrality (Shard 10).

DGP item 6 — the converter's ``build_hcp_influence_graph`` (:1129) reconstructs
an HCP co-treatment graph from ``med.npi ∪ proc.npi`` shared-patient cliques.
To recover a forward-causal signal WITHOUT circularity, centrality is drawn
EXOGENOUSLY (before any adoption) and adoption lag is a decreasing function of
it. These tests pin the no-circularity contract.
"""

import numpy as np

from src.ml.synthetic.claims.config import ClaimsDGPConfig
from src.ml.synthetic.claims.hcp_network import (
    assign_patients_to_hcps,
    build_hcp_pool,
    make_npi_assigner,
)
from src.ml.synthetic.claims.patient_state import generate_patients


def test_centrality_is_exogenous_and_drives_adoption_timing():
    pool = build_hcp_pool(np.random.default_rng(5), n_hcps=80)
    # Item 6: degree skew (a few hubs), centrality assigned BEFORE any adoption.
    assert pool["centrality"].std() > 0.05
    assert {"npi", "degree", "centrality", "expected_adoption_lag"} <= set(pool.columns)
    # Higher-centrality HCPs must have EARLIER expected first-adoption (sign check).
    hi = pool.sort_values("centrality").tail(10)["expected_adoption_lag"].mean()
    lo = pool.sort_values("centrality").head(10)["expected_adoption_lag"].mean()
    assert hi < lo  # higher centrality -> earlier adoption


def test_degree_distribution_is_heavy_tailed():
    pool = build_hcp_pool(np.random.default_rng(11), n_hcps=200)
    # A power-law-ish degree distribution: a few hubs above the mean+2sd.
    d = pool["degree"]
    assert (d > d.mean() + 2 * d.std()).sum() >= 1


def test_patient_assignment_forms_shared_patient_cliques():
    cfg = ClaimsDGPConfig(n_patients=300, seed=3, n_hcps=40)
    rng = np.random.default_rng(3)
    pats = generate_patients(rng, cfg)
    pool = build_hcp_pool(rng, n_hcps=40)
    mapping = assign_patients_to_hcps(rng, pats, pool)
    # Every patient maps to a pool NPI; multiple patients share an NPI (clique).
    assert set(mapping.keys()) == {int(p) for p in pats["patid"]}
    assert set(mapping.values()) <= set(pool["npi"])
    counts = {}
    for npi in mapping.values():
        counts[npi] = counts.get(npi, 0) + 1
    # At least one HCP treats >1 patient (otherwise no shared-patient edge).
    assert max(counts.values()) > 1


def test_high_degree_hcps_see_more_patients():
    cfg = ClaimsDGPConfig(n_patients=600, seed=7, n_hcps=40)
    rng = np.random.default_rng(7)
    pats = generate_patients(rng, cfg)
    pool = build_hcp_pool(rng, n_hcps=40)
    mapping = assign_patients_to_hcps(rng, pats, pool)
    counts = dict.fromkeys(pool["npi"], 0)
    for npi in mapping.values():
        counts[npi] += 1
    pool = pool.copy()
    pool["n_patients"] = pool["npi"].map(counts)
    hi = pool.nlargest(10, "degree")["n_patients"].mean()
    lo = pool.nsmallest(10, "degree")["n_patients"].mean()
    assert hi > lo  # hubs see more patients


def test_npi_assigner_is_deterministic_and_total():
    cfg = ClaimsDGPConfig(n_patients=100, seed=9)
    rng = np.random.default_rng(9)
    pats = generate_patients(rng, cfg)
    npi_for = make_npi_assigner(np.random.default_rng(9), pats, cfg)
    npi_for2 = make_npi_assigner(np.random.default_rng(9), pats, cfg)
    for pid in pats["patid"].head(20):
        assert npi_for(int(pid)) == npi_for2(int(pid))
        assert isinstance(npi_for(int(pid)), str)
