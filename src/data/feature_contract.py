"""Layer 1 — Declarative Temporal Contracts for features.

Every feature in the agentic ML pipeline carries a `FeatureContract` declaring:
- WHEN the feature is knowable (`knowable_at`)
- WHERE its input data comes from (`source`, `derivation_inputs`)
- HOW it's aggregated, if at all (`aggregation`, `window_days`)

The framework validates contracts at AUTHOR TIME (when the contract is
constructed) and at PIPELINE TIME (when the contract chain is checked for
knowable_at propagation). This catches leaks BEFORE the LLM agents see them
and BEFORE the data is materialized — the highest-leverage, lowest-cost
defense layer.

Why this is disease-agnostic: the contract vocabulary (knowable_at,
aggregation, window_days, source) is universal. Every feature in every
indication carries the same metadata shape; the framework rejects ad-hoc
unwindowed event aggregations regardless of whether the data is CSU, Optum,
synthetic, or a future indication.

Reference: .claude/plans/adaptive_temporal_validity_redesign.md (Layer 1).
Compile set evidence: .claude/state/leakage_compile_set_20260507.md (18
documented incidents; the contract layer would catch most of them at author
time IF derivations are written honestly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# Tables that are inherently event-typed (one row per timestamped event,
# extending across the patient panel). Aggregation over these tables MUST
# specify a `window_days` to bound the temporal scope.
EVENT_SOURCES = frozenset(
    {
        "medication_events",
        "procedure_events",
        "lab_events",
        "diagnosis_events",
        "encounter_events",
        "claim_events",
    }
)

# Tables that are NOT event-typed (e.g., one row per patient, or per
# enrollment record). Aggregation over these is unusual and not supported by
# the contract.
NON_EVENT_SOURCES = frozenset(
    {
        "demo",
        "cohort",
        "derived",
        "enrollment",
    }
)

AggregationFunc = Literal["sum", "mean", "max", "min", "count", "nunique"]


class ContractViolation(Exception):
    """Raised when a feature contract is internally inconsistent OR
    when a contract chain has knowable_at propagation violations.

    Attributes:
        feature: name of the feature whose contract is violated
        reason: human-readable explanation
    """

    def __init__(self, message: str, feature: Optional[str] = None, reason: Optional[str] = None):
        super().__init__(message)
        self.feature = feature or ""
        self.reason = reason or message


@dataclass(frozen=True)
class KnowableAt:
    """Declarative timestamp at which a feature's value is knowable.

    `reference` is one of:
    - "index_date" — knowable AT the prediction time (acceptable for features)
    - "enrollment" — knowable BEFORE prediction time (acceptable; e.g., demographics)
    - "post_index" — knowable AFTER prediction time (FORBIDDEN for features)
    - a column-reference string like "diagnosis_date" — the timestamp of that column
    `offset_days` allows expressing relative timestamps (e.g., index_date - 180d).
    """

    reference: str
    offset_days: int = 0

    def is_pre_or_at_index(self) -> bool:
        """True if this timestamp is at or before the prediction time."""
        if self.reference == "post_index":
            return False
        if self.reference in ("index_date", "enrollment"):
            return self.offset_days <= 0
        # For column references, we assume conservatively that they could
        # be post-index unless the offset says otherwise.
        return self.offset_days <= 0

    def __str__(self) -> str:
        if self.offset_days == 0:
            return self.reference
        sign = "+" if self.offset_days > 0 else ""
        return f"{self.reference}{sign}{self.offset_days}d"


@dataclass(frozen=True)
class FeatureContract:
    """Declarative spec for a single feature.

    Examples:
        # Legitimate pre-index demographic feature
        age = FeatureContract(
            name="age_at_index",
            knowable_at=KnowableAt(reference="index_date"),
            source="demo",
            derivation_inputs=["birth_date", "index_date"],
            aggregation=None,
        )

        # Legitimate windowed event aggregation
        recent_fills = FeatureContract(
            name="med_fill_count_180d",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["medication_date"],
            aggregation="count",
            window_days=180,
        )

        # FORBIDDEN — unwindowed event aggregation (raises ContractViolation)
        bad = FeatureContract(
            name="all_med_fills",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["medication_date"],
            aggregation="count",
            window_days=None,  # ← contract violation
        )
    """

    name: str
    knowable_at: KnowableAt
    source: str
    derivation_inputs: tuple[str, ...] = ()
    aggregation: Optional[AggregationFunc] = None
    window_days: Optional[int] = None
    # Test-only escape hatch: allows constructing a contract that DECLARES
    # itself unwindowed (so its honest knowable_at is post_index). Used in
    # validate_contract_chain tests to verify propagation. Never set in
    # production code.
    _allow_unwindowed_for_test: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # Normalize derivation_inputs to tuple (frozen dataclass mutability)
        if not isinstance(self.derivation_inputs, tuple):
            object.__setattr__(self, "derivation_inputs", tuple(self.derivation_inputs))

        self._validate()

    def _validate(self) -> None:
        if not self.name:
            raise ContractViolation("name is required", feature=self.name)

        if self.aggregation is not None:
            if self.source not in EVENT_SOURCES:
                raise ContractViolation(
                    f"aggregation requires event-typed source; got {self.source!r}. "
                    f"Allowed event sources: {sorted(EVENT_SOURCES)}",
                    feature=self.name,
                    reason="aggregation requires event-typed source",
                )
            if self.window_days is None and not self._allow_unwindowed_for_test:
                raise ContractViolation(
                    f"feature {self.name!r}: aggregation without window_days is forbidden. "
                    f"Set window_days to a positive integer to bound the temporal scope.",
                    feature=self.name,
                    reason="aggregation without window_days",
                )
            if self.window_days is not None and self.window_days < 1:
                raise ContractViolation(
                    f"feature {self.name!r}: window_days must be >= 1; got {self.window_days}",
                    feature=self.name,
                    reason="window_days must be >= 1",
                )

        # The ``_allow_unwindowed_for_test`` escape hatch only makes sense for
        # contracts that declare ``knowable_at=post_index``. The hatch's intent
        # is to let chain-validation tests construct contracts whose honest
        # knowable_at is post_index because the unwindowed aggregation makes
        # them so. Allowing the hatch on a pre-or-at-index claim would let an
        # author silently bypass the windowing requirement while pretending the
        # feature is computable at index time — a footgun the contract layer
        # is meant to prevent.
        if self._allow_unwindowed_for_test and self.knowable_at.is_pre_or_at_index():
            raise ContractViolation(
                f"feature {self.name!r}: _allow_unwindowed_for_test=True is only valid "
                f"with knowable_at=post_index (the hatch's purpose is to declare an "
                f"honestly-post-index unwindowed aggregation for chain-validation tests); "
                f"got knowable_at={self.knowable_at}.",
                feature=self.name,
                reason="_allow_unwindowed_for_test requires post_index knowable_at",
            )

    @property
    def is_aggregation(self) -> bool:
        return self.aggregation is not None


@dataclass(frozen=True)
class ContractChainViolation:
    """A violation found during chain validation (not at single-contract construction)."""

    feature: str
    reason: str
    inputs_implicated: tuple[str, ...] = ()


def validate_contract_chain(
    contracts: dict[str, FeatureContract],
) -> list[ContractChainViolation]:
    """Verify that each contract's knowable_at is consistent with its inputs.

    A feature's claimed knowable_at MUST be greater-than-or-equal-to the
    knowable_at of every one of its derivation_inputs. If a feature claims
    knowable_at=index_date but one of its inputs has knowable_at=post_index,
    that's a violation: the feature can't actually be computed at index time.

    Args:
        contracts: mapping from feature name to FeatureContract.

    Returns:
        List of violations. Empty list means the chain is valid.
    """
    violations: list[ContractChainViolation] = []

    # Helper: rank knowable_at on a comparable scale.
    # enrollment < index_date < post_index
    # Within the same reference, larger offset_days is later: an input claimed
    # at index_date+30 is later than its parent at index_date+0, even though
    # both share the index_date reference. Returning a (base, offset) tuple
    # makes lexicographic comparison enforce that ordering. Without this,
    # post-index inputs that nominally share a reference with the parent are
    # silently accepted.
    rank = {"enrollment": 0, "index_date": 1, "post_index": 2}

    def rank_of(k: KnowableAt) -> tuple[int, int]:
        base = rank.get(k.reference, rank["post_index"])
        return (base, k.offset_days)

    for feat_name, contract in contracts.items():
        # Skip if no inputs declared in the chain
        feat_rank = rank_of(contract.knowable_at)
        for input_name in contract.derivation_inputs:
            if input_name not in contracts:
                # Input is a raw column or external source; we don't check
                continue
            input_contract = contracts[input_name]
            input_rank = rank_of(input_contract.knowable_at)
            if input_rank > feat_rank:
                violations.append(
                    ContractChainViolation(
                        feature=feat_name,
                        reason=(
                            f"feature {feat_name!r} claims knowable_at={contract.knowable_at}, "
                            f"but its input {input_name!r} has knowable_at={input_contract.knowable_at} "
                            f"(later in time). The feature cannot actually be computed at "
                            f"its claimed time."
                        ),
                        inputs_implicated=(input_name,),
                    )
                )

    return violations
