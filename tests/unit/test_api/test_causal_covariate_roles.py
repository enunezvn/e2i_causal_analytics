"""Role-aware covariate validation on the agent estimation-frame loaders.

Recovery-benchmark gap 3 (``tests/unit/test_causal_engine/test_discovery/
test_structural_recovery.py`` docstring item 3): a post-treatment variable
declared as a confounder is tier-forced into the DAG with its true edge
REVERSED, gate-ACCEPTed, and shipped in the adjustment set — conditioning on a
mediator attenuated the measured ATE by 60% on the mediator DGP. Engine-side
detection is measured-impossible there (the mediator DGP's true graph is
complete, so every orientation is Markov-equivalent; FCI returns an all-circle
PAG), which makes the API's DECLARATION BOUNDARY the only seam that holds the
temporal knowledge: the dataset specs' role lists. The curated ``covariate``
lists are pre-treatment by curation (the 2026-06-29 overcontrol review), but
the loaders validated requests against the role-INSENSITIVE union
``treatment | outcome | covariate`` — so an analyst's explicit picks could put
an outcome-role column (``adherent_180d``, ``discontinued_180d``,
``treatment_initiated``) into the covariate slot and reach the engine as a
declared confounder.

These tests pin the role-AWARE guard: a requested covariate must hold the
covariate role in the dataset spec. Dual-role columns (treatment+covariate,
e.g. ``disease_stage``) keep working — the rule is membership in
``spec["covariate"]``, not absence from the other lists. Question slots are
deliberately NOT tightened here (no measured harm; the reversed-estimand gate
already rejects outcome-as-treatment runs).

Guard-only test idiom (same as ``test_segment_hte_route.py``): a 400 carrying
"not permitted" is the assertion surface; any OTHER exception means the guard
passed and the loader proceeded toward the (absent) unit-test DB.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.routes import causal as causal_routes

pytestmark = pytest.mark.unit


async def _guard_verdict(**kwargs) -> HTTPException | None:
    """Run the loader and return the 400 'not permitted' guard rejection, or
    None when the request passed the guard (whatever happened downstream)."""
    try:
        await causal_routes._load_agent_estimation_frame(**kwargs)
    except HTTPException as exc:
        if exc.status_code == 400 and "not permitted" in str(exc.detail):
            return exc
    except Exception:
        pass  # no DB in unit context — the guard passed
    return None


class TestPatientJourneysCovariateRoleGuard:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_covariate",
        [
            "adherent_180d",  # outcome-role: post-treatment descendant
            "discontinued_180d",  # outcome-role: post-treatment descendant
            "treatment_initiated",  # treatment+outcome roles: a mediator here
            "low_gap_180d",  # outcome-role: post-treatment descendant
            "copay_support",  # treatment-role only
        ],
    )
    async def test_non_covariate_role_column_is_rejected_as_covariate(
        self, bad_covariate: str
    ) -> None:
        exc = await _guard_verdict(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=[bad_covariate],
            limit=1500,
        )
        assert exc is not None, (
            f"{bad_covariate!r} holds no covariate role in the patient_journeys "
            "spec but was accepted into the covariate slot — the role-insensitive "
            "union allowlist admits post-treatment columns as declared "
            "confounders (benchmark gap 3: measured -60% ATE on the mediator DGP)"
        )
        assert bad_covariate in str(exc.detail)
        assert "covariate" in str(exc.detail)

    @pytest.mark.asyncio
    async def test_rejection_names_every_offending_column(self) -> None:
        exc = await _guard_verdict(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=["adherent_180d", "disease_severity", "treatment_initiated"],
            limit=1500,
        )
        assert exc is not None
        assert "adherent_180d" in str(exc.detail)
        assert "treatment_initiated" in str(exc.detail)
        assert "disease_severity" not in str(exc.detail.split("Allowed")[0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dual_role_covariate",
        [
            "disease_stage",  # treatment + covariate roles (Kisqali modifier)
            "urticaria_severity_uas7",  # treatment + covariate roles (Remibrutinib)
            "complement_inhibitor_status",  # treatment + covariate roles (Fabhalta)
        ],
    )
    async def test_dual_role_covariate_still_accepted(self, dual_role_covariate: str) -> None:
        """Dual-role columns hold the covariate role too — membership in
        spec['covariate'] is the rule, not absence from the other lists."""
        exc = await _guard_verdict(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=[dual_role_covariate],
            limit=1500,
        )
        assert exc is None, f"curated dual-role covariate rejected: {exc}"

    @pytest.mark.asyncio
    async def test_full_curated_covariate_list_still_accepted(self) -> None:
        spec = causal_routes._CAUSAL_DATASET_SPECS["patient_journeys"]
        exc = await _guard_verdict(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=list(spec["covariate"]),
            limit=1500,
        )
        assert exc is None, f"curated covariate list rejected: {exc}"

    @pytest.mark.asyncio
    async def test_unknown_column_still_rejected(self) -> None:
        """The pre-existing union gate (unknown column anywhere) is unchanged."""
        exc = await _guard_verdict(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=["made_up_column"],
            limit=1500,
        )
        assert exc is not None
        assert "made_up_column" in str(exc.detail)

    @pytest.mark.asyncio
    async def test_question_slots_keep_union_validation(self) -> None:
        """Deliberate scope limit: treatment/outcome slots are NOT tightened.
        treatment_initiated is valid in either question slot (dual-listed)."""
        exc = await _guard_verdict(
            dataset="patient_journeys",
            treatment_var="copay_support",
            outcome_var="treatment_initiated",
            covariates=[],
            limit=1500,
        )
        assert exc is None


class TestHcpAdoptionCovariateRoleGuard:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_covariate",
        [
            "adopted",  # outcome-role
            "peer_influence_score",  # treatment-role only
            "treatment_arm",  # treatment-role only
        ],
    )
    async def test_non_covariate_role_column_is_rejected_as_covariate(
        self, bad_covariate: str
    ) -> None:
        exc = await _guard_verdict(
            dataset="hcp_adoption",
            treatment_var="treatment_arm"
            if bad_covariate != "treatment_arm"
            else "peer_influence_score",
            outcome_var="adopted",
            covariates=[bad_covariate],
            limit=1500,
            brand="Kisqali",
        )
        assert exc is not None, (
            f"{bad_covariate!r} holds no covariate role in the hcp_adoption spec "
            "but was accepted into the covariate slot"
        )

    @pytest.mark.asyncio
    async def test_curated_covariate_still_accepted(self) -> None:
        exc = await _guard_verdict(
            dataset="hcp_adoption",
            treatment_var="treatment_arm",
            outcome_var="adopted",
            covariates=["centrality_z"],
            limit=1500,
            brand="Kisqali",
        )
        assert exc is None, f"curated hcp covariate rejected: {exc}"
