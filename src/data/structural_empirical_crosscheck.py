"""Offline structural-vs-empirical crosscheck (Track-2B-v3 D4).

The no-label production gate for activating the deterministic structural
causal-role decider. Production features have no ground-truth labels, so we
validate authored attestations against the *existing empirical signal* (the
Phase-1 FDR confident set / Layer-1 verdicts) on the same cohort, pairing by
feature name.

Safety invariant: the structural role must NEVER place a feature the empirical
signal flags as a leak into ACCEPT. ``compare_structural_vs_empirical`` surfaces
exactly those cases as ``missed_leaks`` (the hard gate: must be 0).

Why structural roles are passed in, NOT read from the flag-on sidecar: in
production the EnsembleVoter returns ``decided_by="adversarial"`` on an
empirical-high veto BEFORE the structural rule fires, and the sidecar records
``structural_role`` only when ``decided_by=="structural"`` — so in exactly the
critical case (an attested feature with high empirical severity) the flag-on
sidecar's ``structural_role`` is ``None`` and the crosscheck would be blind to
it. Compute the structural role independently/offline (via
``src.ml.causal_role_dgp.extractor.derive_structural_role``) and hand it in.
This helper is pure: no graph work, no I/O.

Safety framing (do NOT mis-sell this gate): the voter precedence already
guarantees structural can never override an empirical-high leak in production
(empirical-high wins). So a ``missed_leak`` here is an AUTHORING-quality signal
(the authored edges are wrong), production-safe-by-precedence — but it must
still be 0. The genuinely dangerous case — structural ACCEPT where the empirical
signal ALSO misses a real leak — is NOT detectable by this crosscheck; it is
mitigated by per-feature domain review and by leaving ambiguous features
un-attested (they fall through to the empirical gate).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.kg.ensemble_voter import ACCEPT_ROLES, LEAK_ROLES

# Empirical severities that mean "the data-driven signal calls this a leak".
# Mirrors the empirical-high veto threshold in the EnsembleVoter (which fires on
# "high", see ensemble_voter.py:776/873); "critical" is strictly more severe
# than "high" so it is included. "moderate"/"info"/"none"/"abstain" are NOT
# confident-leak severities (the voter does not veto on a standalone moderate).
LEAK_SEVERITIES: frozenset[str] = frozenset({"critical", "high"})

# Roles the extractor can emit (LEAK_ROLES ∪ ACCEPT_ROLES). A non-None role
# outside this set is unclassifiable and routed to review rather than guessed.
_KNOWN_ROLES: frozenset[str] = LEAK_ROLES | ACCEPT_ROLES


@dataclass(frozen=True)
class CrosscheckResult:
    """A 4-way, mutually-exclusive, exhaustive partition of the attested
    features — every attested feature lands in exactly one tuple.

    Attributes:
        agree: structural leak/accept bucket matches the empirical bucket.
        missed_leaks: structural role ∈ ACCEPT_ROLES but the empirical severity
            is a leak severity (the dangerous direction — the hard gate).
        disagreements: structural role ∈ LEAK_ROLES but empirical is not a leak
            (structural over-flags — the safe direction; still review-worthy).
        reviewed: structural role is None or unclassifiable (the decider
            abstains; the empirical gate governs).
    Feature names within each tuple are sorted for deterministic reporting.
    """

    agree: tuple[str, ...]
    missed_leaks: tuple[str, ...]
    disagreements: tuple[str, ...]
    reviewed: tuple[str, ...]

    @property
    def gate_passed(self) -> bool:
        """The activation gate: no attested feature where structural says ACCEPT
        and the empirical signal says leak."""
        return len(self.missed_leaks) == 0


def compare_structural_vs_empirical(
    structural_roles: dict[str, str | None],
    empirical_severity: dict[str, str],
) -> CrosscheckResult:
    """Pair each attested feature's structural role against its empirical
    leakage severity and partition into agree / missed_leaks / disagreements /
    reviewed.

    Args:
        structural_roles: feature name -> deterministic structural role (one of
            the six extractor roles) or ``None`` when the authored DAG is absent
            or unclassifiable. Compute these OFFLINE via ``derive_structural_role``
            — do NOT read them from the flag-on sidecar (empirical-high overrides
            mask the ``structural_role`` there; see module docstring).
        empirical_severity: feature name -> empirical leakage severity string
            (e.g. "critical"/"high"/"moderate"/"info"). Missing entries are
            treated as no empirical signal (not a leak).

    Returns:
        CrosscheckResult. The activation gate is ``result.gate_passed``
        (equivalently ``len(result.missed_leaks) == 0``).
    """
    agree: list[str] = []
    missed_leaks: list[str] = []
    disagreements: list[str] = []
    reviewed: list[str] = []

    for feature, role in structural_roles.items():
        # Abstain: no role, or a role outside the known taxonomy → review.
        if role is None or role not in _KNOWN_ROLES:
            reviewed.append(feature)
            continue

        structural_is_leak = role in LEAK_ROLES
        empirical_is_leak = empirical_severity.get(feature, "") in LEAK_SEVERITIES

        if structural_is_leak == empirical_is_leak:
            agree.append(feature)
        elif empirical_is_leak:
            # structural ACCEPT but empirical flags a leak — the dangerous case.
            missed_leaks.append(feature)
        else:
            # structural LEAK but empirical quiet — conservative/safe over-flag.
            disagreements.append(feature)

    return CrosscheckResult(
        agree=tuple(sorted(agree)),
        missed_leaks=tuple(sorted(missed_leaks)),
        disagreements=tuple(sorted(disagreements)),
        reviewed=tuple(sorted(reviewed)),
    )
