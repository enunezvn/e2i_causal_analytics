# ADR-011: Feature-importance stability gate certifies the displayed covariate-group ranking

**Date**: 2026-07-18 | **Status**: Accepted (user-approved estimand change) | **Implemented by**: PRs #1268, #1270, #1272, #1277 (migration 109)

## Context

`/api/explain/global` moved from fixed prefix sampling to adaptive random sampling (RPC `sample_entity_ids`, migration 109) with a statistical stopping rule: sampling stops early once every adjacent pair in the top-5 mean-|SHAP| ranking is separated by more than 1.96·SE of the gap, with two exemptions — jointly-negligible pairs (< 2% of the top mean) and *confirmed practical ties* (`gap + 1.96·SE(gap) < floor`, an equivalence-test bound; a merely-small observed gap with large SE does NOT qualify).

The mismatch that forced this decision: the gate certified the top-5 **encoded features**, but the page's headline ranking displays **covariate groups** (one-hot children summed into one row). The gate was burning samples separating siblings of the same displayed row — e.g. persistence/Remibrutinib spent n=152 separating `geo_midwest` from `geo_west`, two children of one "geographic_region" row. The badge claimed stability of a ranking the user never sees.

## Decision

The stability criterion operates on the **displayed covariate-group ranking** — this changes the estimand, and was explicitly signed off by the owner (codex adversarial review flagged the estimand change as requiring sign-off; approved 2026-07-18):

1. `_parent_covariate` mirrors the frontend's grouping exactly (bare name / `__isna` twin / one-hot prefix, longest match wins).
2. Group |SHAP| lists are zero-backfilled to full length n, making them index-aligned by entity — which enables **paired SE** over per-entity differences (one-hot siblings correlate at r≈−0.37, so the independent-SE formula is anti-conservative for exactly the pairs that matter).
3. Sampling caps: 60 default, 200 for persistence/discontinuation model types.
4. Provenance is persisted in the reserved `__sampling__` key of `ml_shap_analyses.global_importance`, including `stability_criterion` (`covariate_group` | `encoded_feature`); legacy rows read NULL. The frontend badge reads "covariate ranking stable" only under the new criterion, and the bar chart sorts by the producer's `contribution_rank` (fixing a divergent |signed-value| sort).

## Consequences

- (+) The gate certifies what the page displays. Post-change: 10/12 brand×model combos stable, most far below cap (5–39 s refreshes) — the encoded gate really was wasting n on within-row siblings.
- (−) An honest regression by construction: persistence/Remibrutinib was "stable n=152" under the old (wrong) estimand and is now a truthful cap-hit (projected n≈433). Do not "fix" this by trusting the old badge.
- (−) Combos near the stability boundary flip between refreshes (fresh random draws) — expected, not a regression.

## References

- `docs/data/03-ML-PIPELINE-SCHEMA.md` §1.6 — `__sampling__` provenance schema
- `src/api/routes/explain.py` — `_ranking_stable`, `_parent_covariate`
