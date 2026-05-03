"""Tests for ``FeatureManifest`` (shard 01 §C.5)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from src.ml.synthetic_v2.manifest import FeatureManifest, manifest_to_jsonable


def _valid_manifest() -> FeatureManifest:
    return FeatureManifest(
        name="lace_score",
        distribution="normal",
        distribution_params={"loc": 5.5, "scale": 2.0},
        coefficient=0.42,
        monotone_direction=1,
        is_noise=False,
        clinical_justification="LACE-HF score predicts iDFS at 5y (Smith 2024).",
        citation_strength="strong",
    )


class TestFeatureManifestConstruction:
    def test_construct_with_valid_fields(self) -> None:
        m = _valid_manifest()
        assert m.name == "lace_score"
        assert m.distribution == "normal"
        assert m.distribution_params == {"loc": 5.5, "scale": 2.0}
        assert m.coefficient == pytest.approx(0.42)
        assert m.monotone_direction == 1
        assert m.is_noise is False
        assert m.citation_strength == "strong"

    def test_frozen_blocks_attribute_reassignment(self) -> None:
        m = _valid_manifest()
        with pytest.raises(FrozenInstanceError):
            m.coefficient = 0.99  # type: ignore[misc]

    def test_unknown_attribute_assignment_blocked_by_frozen(self) -> None:
        """Frozen blocks new-attribute assignment as well as existing-field reassignment."""
        m = _valid_manifest()
        with pytest.raises(FrozenInstanceError):
            m.unknown_field = "x"  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "distribution,params",
        [
            ("normal", {"loc": 0.0, "scale": 1.0}),
            ("uniform", {"low": 0.0, "high": 1.0}),
            ("bernoulli", {"p": 0.3}),
            ("categorical", {"categories": ["a", "b"], "probabilities": [0.5, 0.5]}),
        ],
    )
    def test_supported_distributions(self, distribution: str, params: dict[str, object]) -> None:
        m = FeatureManifest(
            name="x",
            distribution=distribution,  # type: ignore[arg-type]
            distribution_params=params,
            coefficient=0.0,
            monotone_direction=0,
            is_noise=True,
            clinical_justification="placeholder rationale",
            citation_strength="weak",
        )
        assert m.distribution == distribution
        assert m.distribution_params == params


class TestFeatureManifestValidation:
    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            FeatureManifest(
                name="",
                distribution="normal",
                distribution_params={"loc": 0.0, "scale": 1.0},
                coefficient=0.0,
                monotone_direction=0,
                is_noise=True,
                clinical_justification="placeholder",
                citation_strength="weak",
            )

    def test_invalid_distribution_raises(self) -> None:
        with pytest.raises(ValueError, match="distribution must be one of"):
            FeatureManifest(
                name="x",
                distribution="poisson",  # type: ignore[arg-type]
                distribution_params={"lam": 3.0},
                coefficient=0.0,
                monotone_direction=0,
                is_noise=True,
                clinical_justification="placeholder",
                citation_strength="weak",
            )

    @pytest.mark.parametrize("bad", [-2, 2, 100, -100])
    def test_invalid_monotone_direction_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match=r"monotone_direction must be -1/0/\+1"):
            FeatureManifest(
                name="x",
                distribution="normal",
                distribution_params={"loc": 0.0, "scale": 1.0},
                coefficient=0.1,
                monotone_direction=bad,  # type: ignore[arg-type]
                is_noise=False,
                clinical_justification="placeholder rationale",
                citation_strength="weak",
            )

    def test_invalid_citation_strength_raises(self) -> None:
        with pytest.raises(ValueError, match="citation_strength must be one of"):
            FeatureManifest(
                name="x",
                distribution="normal",
                distribution_params={"loc": 0.0, "scale": 1.0},
                coefficient=0.0,
                monotone_direction=0,
                is_noise=True,
                clinical_justification="placeholder",
                citation_strength="medium",  # type: ignore[arg-type]
            )

    def test_is_noise_true_with_nonzero_coefficient_raises(self) -> None:
        with pytest.raises(ValueError, match="is_noise=True requires coefficient=0"):
            FeatureManifest(
                name="x",
                distribution="normal",
                distribution_params={"loc": 0.0, "scale": 1.0},
                coefficient=0.3,
                monotone_direction=0,
                is_noise=True,
                clinical_justification="placeholder",
                citation_strength="weak",
            )

    def test_is_noise_true_with_zero_coefficient_ok(self) -> None:
        m = FeatureManifest(
            name="x",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.0,
            monotone_direction=0,
            is_noise=True,
            clinical_justification="noise feature for AUC band targeting",
            citation_strength="weak",
        )
        assert m.is_noise is True
        assert m.coefficient == 0.0

    def test_is_noise_false_allows_nonzero_coefficient(self) -> None:
        m = FeatureManifest(
            name="x",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.5,
            monotone_direction=1,
            is_noise=False,
            clinical_justification="signal feature",
            citation_strength="moderate",
        )
        assert m.is_noise is False
        assert m.coefficient == pytest.approx(0.5)

    def test_is_noise_false_with_zero_coefficient_raises(self) -> None:
        """Symmetric invariant: signal feature must have non-zero coefficient.

        A zero-coefficient signal feature is silently inert (logit dot product
        ignores it) but the manifest claims it is signal — contradicts the
        clinical_justification audit story.
        """
        with pytest.raises(ValueError, match="is_noise=False requires coefficient!=0"):
            FeatureManifest(
                name="x",
                distribution="normal",
                distribution_params={"loc": 0.0, "scale": 1.0},
                coefficient=0.0,
                monotone_direction=1,
                is_noise=False,
                clinical_justification="signal feature mistakenly zeroed",
                citation_strength="moderate",
            )

    def test_blank_justification_raises(self) -> None:
        with pytest.raises(ValueError, match="clinical_justification must be non-empty"):
            FeatureManifest(
                name="x",
                distribution="normal",
                distribution_params={"loc": 0.0, "scale": 1.0},
                coefficient=0.0,
                monotone_direction=0,
                is_noise=True,
                clinical_justification="   ",
                citation_strength="weak",
            )


class TestFeatureManifestSerialization:
    def test_to_dict_returns_asdict(self) -> None:
        m = _valid_manifest()
        assert m.to_dict() == asdict(m)

    def test_to_dict_is_json_serializable(self) -> None:
        m = _valid_manifest()
        s = json.dumps(m.to_dict(), sort_keys=True)
        parsed = json.loads(s)
        assert parsed["name"] == "lace_score"
        assert parsed["coefficient"] == pytest.approx(0.42)
        assert parsed["monotone_direction"] == 1

    def test_to_dict_preserves_distribution_params(self) -> None:
        m = _valid_manifest()
        d = m.to_dict()
        assert d["distribution_params"] == {"loc": 5.5, "scale": 2.0}

    def test_manifest_to_jsonable_tuple(self) -> None:
        m1 = _valid_manifest()
        m2 = FeatureManifest(
            name="age",
            distribution="uniform",
            distribution_params={"low": 18.0, "high": 90.0},
            coefficient=0.0,
            monotone_direction=0,
            is_noise=True,
            clinical_justification="age prior in EBC cohort",
            citation_strength="weak",
        )
        out = manifest_to_jsonable((m1, m2))
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0]["name"] == "lace_score"
        assert out[1]["name"] == "age"

    def test_manifest_to_jsonable_list_input(self) -> None:
        m = _valid_manifest()
        out = manifest_to_jsonable([m])
        assert out == [m.to_dict()]

    def test_manifest_to_jsonable_empty_input(self) -> None:
        assert manifest_to_jsonable(()) == []
        assert manifest_to_jsonable([]) == []

    def test_manifest_to_jsonable_round_trips_through_json(self) -> None:
        m = _valid_manifest()
        out = manifest_to_jsonable([m])
        s = json.dumps(out, sort_keys=True)
        parsed = json.loads(s)
        assert parsed[0]["name"] == "lace_score"
        assert parsed[0]["distribution_params"] == {"loc": 5.5, "scale": 2.0}


class TestFeatureManifestNumpyCoercion:
    """``to_dict()`` must coerce numpy types so consumers can ``json.dumps``."""

    def test_to_dict_coerces_numpy_scalar_in_params(self) -> None:
        np = pytest.importorskip("numpy")
        m = FeatureManifest(
            name="x",
            distribution="normal",
            distribution_params={"loc": np.float64(0.5), "scale": np.float32(1.5)},
            coefficient=0.1,
            monotone_direction=0,
            is_noise=False,
            clinical_justification="numpy-derived params",
            citation_strength="weak",
        )
        d = m.to_dict()
        # After coercion the values must be JSON-native
        s = json.dumps(d, sort_keys=True)
        parsed = json.loads(s)
        assert parsed["distribution_params"]["loc"] == pytest.approx(0.5)
        assert parsed["distribution_params"]["scale"] == pytest.approx(1.5)

    def test_to_dict_coerces_numpy_array_in_params(self) -> None:
        np = pytest.importorskip("numpy")
        m = FeatureManifest(
            name="x",
            distribution="categorical",
            distribution_params={
                "categories": ["a", "b", "c"],
                "probabilities": np.array([0.5, 0.3, 0.2]),
            },
            coefficient=0.1,
            monotone_direction=0,
            is_noise=False,
            clinical_justification="numpy-derived probabilities",
            citation_strength="weak",
        )
        d = m.to_dict()
        s = json.dumps(d, sort_keys=True)
        parsed = json.loads(s)
        assert parsed["distribution_params"]["probabilities"] == pytest.approx([0.5, 0.3, 0.2])

    def test_to_dict_coerces_numpy_int(self) -> None:
        np = pytest.importorskip("numpy")
        m = FeatureManifest(
            name="x",
            distribution="bernoulli",
            distribution_params={"p": 0.3, "extra_count": np.int64(5)},
            coefficient=0.1,
            monotone_direction=0,
            is_noise=False,
            clinical_justification="numpy-derived count",
            citation_strength="weak",
        )
        d = m.to_dict()
        s = json.dumps(d, sort_keys=True)
        parsed = json.loads(s)
        assert parsed["distribution_params"]["extra_count"] == 5

    def test_to_dict_coerces_nested_numpy(self) -> None:
        np = pytest.importorskip("numpy")
        m = FeatureManifest(
            name="x",
            distribution="categorical",
            distribution_params={
                "matrix": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            },
            coefficient=0.1,
            monotone_direction=0,
            is_noise=False,
            clinical_justification="nested numpy",
            citation_strength="weak",
        )
        d = m.to_dict()
        s = json.dumps(d, sort_keys=True)
        parsed = json.loads(s)
        assert parsed["distribution_params"]["matrix"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_to_dict_raises_on_unsupported_type(self) -> None:
        class _Unrepresentable:
            pass

        m = FeatureManifest(
            name="x",
            distribution="normal",
            distribution_params={"weird": _Unrepresentable()},
            coefficient=0.1,
            monotone_direction=0,
            is_noise=False,
            clinical_justification="bad params",
            citation_strength="weak",
        )
        with pytest.raises(TypeError, match="non-JSON-serializable"):
            m.to_dict()


class TestFeatureManifestEqualityAndHash:
    def test_two_identical_manifests_are_equal(self) -> None:
        m1 = _valid_manifest()
        m2 = _valid_manifest()
        assert m1 == m2

    def test_distinct_coefficients_are_unequal(self) -> None:
        m1 = _valid_manifest()
        m2 = FeatureManifest(
            name="lace_score",
            distribution="normal",
            distribution_params={"loc": 5.5, "scale": 2.0},
            coefficient=0.43,
            monotone_direction=1,
            is_noise=False,
            clinical_justification="LACE-HF score predicts iDFS at 5y (Smith 2024).",
            citation_strength="strong",
        )
        assert m1 != m2

    def test_unhashable_due_to_dict_field(self) -> None:
        """``distribution_params: dict`` makes the dataclass unhashable.

        The audit fingerprint (shard 01 §C.6) uses JSON serialization, not
        ``hash()``, so this is intentional. Documents the contract.
        """
        m = _valid_manifest()
        with pytest.raises(TypeError):
            hash(m)
