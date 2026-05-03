"""``FeatureManifest`` dataclass + helpers (shard 01 §C.5).

A frozen per-feature audit record carrying name, distribution, signed
coefficient (logit scale), monotone direction, and clinical justification.

Consumed by:
- ``ScenarioBuilder.compute_logits`` (shard 02 §C.1) — multiplies
  ``coefficient`` by the per-scenario ``slope_multiplier``.
- ``ScenarioMetadata.monotone_vector`` (shard 01 §B.2) — feeds W2 day-4
  monotone-LightGBM as a sign-aligned tuple ``(-1, 0, +1, ...)``.
- ``_fingerprint(scenario, seed, n_total, manifest)`` (shard 01 §C.6) —
  serialized via ``to_dict()``/``manifest_to_jsonable`` for SHA-256 input.

``__post_init__`` validates the ``Literal`` invariants at runtime since
``typing.Literal`` is purely a static-checker hint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

DistributionName = Literal["normal", "uniform", "bernoulli", "categorical"]
MonotoneDirection = Literal[-1, 0, 1]
CitationStrength = Literal["strong", "moderate", "weak"]

_VALID_DISTRIBUTIONS: Final[frozenset[str]] = frozenset(
    {"normal", "uniform", "bernoulli", "categorical"}
)
_VALID_MONOTONE: Final[frozenset[int]] = frozenset({-1, 0, 1})
_VALID_CITATION_STRENGTH: Final[frozenset[str]] = frozenset(
    {"strong", "moderate", "weak"}
)


def _coerce_to_jsonable(value: Any) -> Any:
    """Coerce numpy / non-builtin values to JSON-serializable Python builtins.

    Why: shard 03/04/05 per-scenario manifests may compute coefficients or
    distribution params via numpy and end up with ``np.float64`` /
    ``np.ndarray`` values inside ``distribution_params``. ``json.dumps``
    raises ``TypeError`` on those without help. ``asdict`` does not coerce.
    Run output of ``to_dict()`` through this helper so the audit fingerprint
    (shard 01 §C.6) can ``json.dumps(..., sort_keys=True)`` straight away.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_to_jsonable(x) for x in value]
    # numpy.ndarray + numpy.matrix: tolist() returns a Python list / nested list
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _coerce_to_jsonable(tolist())
    # numpy scalars: .item() returns a Python builtin
    item = getattr(value, "item", None)
    if callable(item):
        return _coerce_to_jsonable(item())
    raise TypeError(
        "FeatureManifest serialization received a non-JSON-serializable value "
        f"of type {type(value).__name__}: {value!r}"
    )


@dataclass(frozen=True)
class FeatureManifest:
    """Per-feature audit record (shard 01 §C.5).

    Frozen so consumers (W2 day-4 monotone-LightGBM, audit fingerprint) can
    rely on immutability. Note: instances are intentionally unhashable because
    ``distribution_params`` is a dict; use ``to_dict()`` for serialization,
    not ``hash()``. ``slots=True`` is omitted because the dict field already
    blocks hashability and instance counts (~125 across A/B/C) make the
    memory savings negligible.
    """

    name: str
    distribution: DistributionName
    distribution_params: dict[str, Any]
    coefficient: float
    monotone_direction: MonotoneDirection
    is_noise: bool
    clinical_justification: str
    citation_strength: CitationStrength

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureManifest.name must be non-empty")
        if self.distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                "FeatureManifest.distribution must be one of "
                f"{sorted(_VALID_DISTRIBUTIONS)}; got {self.distribution!r}"
            )
        if self.monotone_direction not in _VALID_MONOTONE:
            raise ValueError(
                "FeatureManifest.monotone_direction must be -1/0/+1; "
                f"got {self.monotone_direction!r}"
            )
        if self.citation_strength not in _VALID_CITATION_STRENGTH:
            raise ValueError(
                "FeatureManifest.citation_strength must be one of "
                f"{sorted(_VALID_CITATION_STRENGTH)}; got {self.citation_strength!r}"
            )
        if self.is_noise and self.coefficient != 0.0:
            raise ValueError(
                "FeatureManifest.is_noise=True requires coefficient=0.0; "
                f"got coefficient={self.coefficient!r} for feature {self.name!r}"
            )
        if not self.is_noise and self.coefficient == 0.0:
            raise ValueError(
                "FeatureManifest.is_noise=False requires coefficient!=0.0; "
                f"feature {self.name!r} has coefficient=0.0 but is_noise=False — "
                "either flip is_noise=True or assign a non-zero coefficient. "
                "(Audit invariant: is_noise is the canonical signal/noise label, "
                "and a zero-coefficient signal feature is contradictory.)"
            )
        if not self.clinical_justification.strip():
            raise ValueError(
                "FeatureManifest.clinical_justification must be non-empty "
                f"for feature {self.name!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (used by audit fingerprint, shard 01 §C.6).

        Numpy scalars / arrays inside ``distribution_params`` are coerced to
        Python builtins so the consumer can ``json.dumps(..., sort_keys=True)``
        without a ``TypeError``.
        """
        raw = asdict(self)
        return {k: _coerce_to_jsonable(v) for k, v in raw.items()}


def manifest_to_jsonable(
    manifest: tuple[FeatureManifest, ...] | list[FeatureManifest],
) -> list[dict[str, Any]]:
    """Convert a manifest sequence to JSON-serializable list of dicts.

    Consumed by ``_fingerprint`` (shard 01 §C.6); the caller wraps this in
    ``json.dumps(..., sort_keys=True)`` for SHA-256 stability across runs.
    """
    return [m.to_dict() for m in manifest]
