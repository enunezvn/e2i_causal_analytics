"""Shared heavy-compute helper for digital-twin population simulation (P2).

Single source of truth for the blocking twin-generation + simulation compute so
the synchronous API path (``src/api/routes/digital_twin.py``) and the Celery
``simulate_population`` task run *identical* logic and produce an *identical*
``SimulationResult``.

Contract preservation
---------------------
* :func:`run_simulation_compute` builds the population + runs the engine and
  returns the in-memory ``SimulationResult`` (used by the inline API path's
  bounded executor and by the Celery task).
* :func:`simulation_result_to_dict` / :func:`simulation_result_from_dict` round-
  trip that result through a JSON-safe dict for the Celery result backend
  (serializer = ``json``). The API route rebuilds the same ``SimulationResult``
  and runs its EXISTING response extraction, so both paths yield the same
  ``SimulationResponse`` shape.

Heavy imports stay function-local so importing this module from the API process
(only to enqueue ``.apply_async()``) does not pull the twin libraries into the
API worker's memory budget.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID


def run_simulation_compute(
    *,
    twin_type_value: str,
    brand_value: str,
    twin_count: int,
    intervention_dict: Dict[str, Any],
    population_filter_dict: Optional[Dict[str, Any]],
    calculate_heterogeneity: bool,
    model_id_value: Optional[str],
    model_uri: Optional[str] = None,
    model_run_id: Optional[str] = None,
) -> "Any":
    """Run twin generation + simulation, returning the ``SimulationResult``.

    Takes only JSON-serializable primitives so the Celery task can pass the same
    arguments over the wire.

    Args:
        twin_type_value: ``TwinType`` enum value (e.g. ``"hcp"``).
        brand_value: ``Brand`` enum value (e.g. ``"Remibrutinib"``).
        twin_count: Number of twins to generate/simulate.
        intervention_dict: Plain dict of ``InterventionConfig`` fields.
        population_filter_dict: Plain dict of ``PopulationFilter`` fields, or
            ``None`` when no filter was supplied.
        calculate_heterogeneity: Whether to compute heterogeneous effects.
        model_id_value: Optional explicit model UUID string.
        model_uri: MLflow model URI of the persisted trained model to load
            before generating (#705 H4).
        model_run_id: MLflow run id holding the preprocessor bundle for that model.

    Returns:
        A ``SimulationResult`` (pydantic model).

    Raises:
        RuntimeError: if no trained model can be hydrated — the worker fails
            loudly rather than generate from an untrained model.
    """
    from src.digital_twin import twin_persistence
    from src.digital_twin.models.simulation_models import (
        InterventionConfig,
        PopulationFilter,
    )
    from src.digital_twin.models.twin_models import Brand, TwinType
    from src.digital_twin.simulation_engine import SimulationEngine
    from src.digital_twin.twin_generator import TwinGenerator

    intervention = InterventionConfig(**intervention_dict)
    pop_filter = (
        PopulationFilter(**population_filter_dict) if population_filter_dict is not None else None
    )

    twin_type = TwinType(twin_type_value)
    brand = Brand(brand_value)

    generator = TwinGenerator(twin_type=twin_type, brand=brand)
    # Load the persisted trained model before generating. A fresh untrained
    # generator would raise RuntimeError in generate(); fail loudly here so the
    # task surfaces an honest error instead of fabricating output (#705 H4).
    if not twin_persistence.hydrate_generator(generator, model_uri, model_run_id):
        raise RuntimeError(
            f"No loadable trained twin model (uri={model_uri!r}, run_id="
            f"{model_run_id!r}) for {brand_value}/{twin_type_value} — cannot simulate."
        )
    model_id = UUID(model_id_value) if model_id_value else (generator.model_id or UUID(int=0))

    population = generator.generate(n=twin_count)
    engine = SimulationEngine(
        population=population,
        model_id=model_id,  # type: ignore[call-arg]
    )
    return engine.simulate(
        intervention_config=intervention,
        population_filter=pop_filter,
        calculate_heterogeneity=calculate_heterogeneity,
    )


def simulation_result_to_dict(result: "Any") -> Dict[str, Any]:
    """Serialize a ``SimulationResult`` to a JSON-safe dict for Celery transport.

    Uses pydantic ``model_dump(mode="json")`` so UUIDs/datetimes/enums become
    JSON primitives (the Celery result backend serializer is ``json``).
    """
    out: Dict[str, Any] = result.model_dump(mode="json")
    return out


def simulation_result_from_dict(data: Dict[str, Any]) -> "Any":
    """Rebuild a ``SimulationResult`` from the JSON dict returned by the task.

    The API route calls this on the offload path so its existing response
    extraction (attribute access + ``.is_significant()`` etc.) is byte-identical
    to the inline path.
    """
    from src.digital_twin.models.simulation_models import SimulationResult

    return SimulationResult.model_validate(data)


def run_population_simulation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Celery-task entrypoint: run a population simulation from a JSON payload.

    Args:
        payload: JSON-serializable dict mirroring :func:`run_simulation_compute`
            kwargs.

    Returns:
        JSON-serializable ``SimulationResult`` dict (see
        :func:`simulation_result_to_dict`).
    """
    result = run_simulation_compute(
        twin_type_value=str(payload["twin_type_value"]),
        brand_value=str(payload["brand_value"]),
        twin_count=int(payload["twin_count"]),
        intervention_dict=dict(payload["intervention_dict"]),
        population_filter_dict=payload.get("population_filter_dict"),
        calculate_heterogeneity=bool(payload.get("calculate_heterogeneity", True)),
        model_id_value=payload.get("model_id_value"),
        model_uri=payload.get("model_uri"),
        model_run_id=payload.get("model_run_id"),
    )
    return simulation_result_to_dict(result)


__all__ = [
    "run_population_simulation",
    "run_simulation_compute",
    "simulation_result_from_dict",
    "simulation_result_to_dict",
]
