"""HCP shared-patient network + exogenous centrality (Shard 10 P1b).

DGP item 6 — the converter's ``build_hcp_influence_graph`` (:1129) reconstructs
an HCP co-treatment graph from ``med.npi ∪ proc.npi`` shared-patient cliques and
emits ``peer_influence_score`` (eigenvector centrality) + ``influence_network_size``
(degree). To recover a *forward-causal* signal WITHOUT circularity, this module:

  1. draws an HCP pool with a skewed degree distribution -> ``centrality``
     assigned EXOGENOUSLY (before any adoption is decided);
  2. sets ``expected_adoption_lag`` as a decreasing function of centrality
     (higher-centrality HCPs adopt earlier);
  3. assigns each patient a treating HCP so that med/proc NPIs reconstruct the
     same shared-patient cliques the converter reads.

The minimal P1.3 entrypoint is ``make_npi_assigner``; ``build_hcp_pool`` is the
exogenous-centrality core exercised by the P1b tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import ClaimsDGPConfig

_BASE_NPI = 1_000_000_000
_LAG0_DAYS = 90.0  # baseline expected adoption lag for a zero-centrality HCP


def build_hcp_pool(rng: np.random.Generator, n_hcps: int) -> pd.DataFrame:
    """Build an HCP pool with EXOGENOUS centrality + adoption lag.

    Degree is drawn from a heavy-tailed (Pareto) distribution so a few hubs
    dominate. ``centrality`` is a normalised function of degree, fixed BEFORE
    any adoption decision (breaking the circularity Fact #4 warns of).
    ``expected_adoption_lag`` decreases in centrality: hubs adopt earlier.
    """
    n_hcps = max(int(n_hcps), 2)
    # Heavy-tailed degree -> a few high-degree hubs, many low-degree peripherals.
    raw_degree = rng.pareto(2.5, size=n_hcps) + 1.0
    degree = raw_degree / raw_degree.max()  # in (0, 1]
    # Centrality monotonically increasing in degree, normalised to ~[0, 1].
    centrality = (degree - degree.min()) / (degree.max() - degree.min() + 1e-9)
    # Higher centrality -> shorter expected adoption lag (earlier adoption).
    expected_adoption_lag = _LAG0_DAYS * (1.0 - 0.7 * centrality)
    npi = np.array([str(_BASE_NPI + i) for i in range(n_hcps)], dtype=object)
    return pd.DataFrame(
        {
            "npi": npi,
            "degree": degree,
            "centrality": centrality,
            "expected_adoption_lag": expected_adoption_lag,
        }
    )


def assign_patients_to_hcps(
    rng: np.random.Generator, pats: pd.DataFrame, pool: pd.DataFrame
) -> dict[int, str]:
    """Assign each patient a primary treating HCP NPI, weighted by degree.

    High-degree HCPs see more patients (so shared-patient cliques form around
    hubs). The mapping is patid -> npi; med + proc both draw from it so the
    converter reconstructs the same co-treatment graph.
    """
    weights = pool["degree"].to_numpy()
    probs = weights / weights.sum()
    chosen = rng.choice(pool["npi"].to_numpy(), size=len(pats), p=probs)
    return {int(pid): str(npi) for pid, npi in zip(pats["patid"].to_numpy(), chosen, strict=True)}


def make_npi_assigner(
    rng: np.random.Generator, pats: pd.DataFrame, cfg: ClaimsDGPConfig
) -> Callable[[int], str]:
    """Return a ``patid -> npi`` callable backed by the exogenous HCP pool.

    Used by the CLI so ``emit_medication`` / ``emit_procedure`` share one
    coherent shared-patient HCP graph.
    """
    n_hcps = cfg.n_hcps if cfg.n_hcps else max(8, len(pats) // 10)
    pool = build_hcp_pool(rng, n_hcps)
    pat_to_npi = assign_patients_to_hcps(rng, pats, pool)

    def npi_for(pid: int) -> str:
        return pat_to_npi[int(pid)]

    return npi_for
