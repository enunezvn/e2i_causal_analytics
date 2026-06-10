"""Dispatcher node for orchestrator agent.

Parallel agent dispatch with timeout handling.
"""

import asyncio
import dataclasses
import functools
import importlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

from .._agent_method_map import get_method_spec
from ..state import AgentDispatch, AgentResult, OrchestratorState

logger = logging.getLogger(__name__)


def _declared_field_names(input_cls: Type[Any]) -> Set[str]:
    """Return the set of field names declared on a dataclass or pydantic model.

    Supports the two wrapping shapes the dispatcher must serve: strict
    ``@dataclass`` (e.g. ``ExperimentMonitorInput``) and pydantic v2
    ``BaseModel`` (e.g. ``DriftMonitorInput``, ``ExperimentDesignerInput``).
    Returns an empty set for anything else; the caller treats that as "do not
    project — splat as-is and let the constructor decide".
    """
    if dataclasses.is_dataclass(input_cls):
        return {f.name for f in dataclasses.fields(input_cls)}
    # Pydantic v2: BaseModel exposes ``model_fields`` on the class.
    model_fields = getattr(input_cls, "model_fields", None)
    if isinstance(model_fields, dict):
        return set(model_fields.keys())
    return set()


# Per-agent AC-3 defaults sourced from the orchestrator dispatch context.
# Each entry maps an agent_name → a callable returning a dict of (field_name →
# default_value) computed from ``payload`` (the generic orchestrator payload)
# and ``dispatch`` (the AgentDispatch entry). Defaults are applied AFTER the
# generic payload + dispatch.parameters merge but BEFORE final field-projection,
# so they only land for fields the input_model actually declares and only when
# the merge did not already supply a value.
#
# Issue #260 AC-3: required fields the wrapped input model declares but the
# orchestrator payload does not naturally carry must get a reasonable default.
def _wrapped_input_defaults(
    agent_name: str, payload: Dict[str, Any], dispatch: AgentDispatch
) -> Dict[str, Any]:
    """Return per-agent ACs-#3 defaults for wrapped-input fields."""
    query = payload.get("query") or ""
    defaults: Dict[str, Any] = {}
    if agent_name == "experiment_monitor":
        # ExperimentMonitorInput.experiment_ids defaults to None on the
        # dataclass itself; explicitly normalising to [] keeps the agent's
        # contract clearer (None vs [] both mean "no specific IDs", but
        # pinning [] avoids ambiguity downstream).
        defaults["experiment_ids"] = []
    elif agent_name == "drift_monitor":
        # DriftMonitorInput.features_to_monitor is required with min_length=1.
        # The router should normally populate
        # ``dispatch.parameters['features_to_monitor']`` — when it doesn't,
        # derive a sensible default from ``parsed_query.entities`` (KPI/feature
        # mentions in the user's query) so the agent runs against the entities
        # the user actually named. When neither source produces ≥1 entry,
        # leave it absent so the pydantic min_length=1 validator surfaces a
        # clear structured AgentResult.error — better than fabricating phantom
        # feature names. (Codex MED-required on PR #275 issue #260.)
        parsed_query = payload.get("parsed_query") or {}
        entities = (parsed_query.get("entities") if isinstance(parsed_query, dict) else None) or []
        kpi_features = [
            ent["value"]
            for ent in entities
            if isinstance(ent, dict)
            and ent.get("type") in {"kpi", "feature_name"}
            and isinstance(ent.get("value"), str)
            and ent["value"]
        ]
        if kpi_features:
            defaults["features_to_monitor"] = kpi_features
    elif agent_name == "experiment_designer":
        # ExperimentDesignerInput.business_question is required with
        # min_length=10. Default it from the orchestrator-level ``query`` —
        # that's how the harness's Tier0OutputMapper bridges the two contracts.
        defaults["business_question"] = query
    return defaults


def _to_kwargs_dict(model_instance: Any) -> Dict[str, Any]:
    """Re-flatten a wrapped-input model instance back into a kwargs dict.

    Used only when an ``AGENT_METHOD_MAP`` entry declares BOTH ``input_model``
    AND ``uses_kwargs`` — the dispatcher first wraps the payload into the
    model (validating it), then splats it into the agent method. The current
    registry has no such entry; this is defense-in-depth so future additions
    are robust. Prefers pydantic v2 ``model_dump()`` then falls back to
    ``dataclasses.asdict()``. Returns ``{}`` for shapes it cannot flatten.
    """
    dump = getattr(model_instance, "model_dump", None)
    if callable(dump):
        result = dump()
        if isinstance(result, dict):
            return cast(Dict[str, Any], result)
    if dataclasses.is_dataclass(model_instance) and not isinstance(model_instance, type):
        return dataclasses.asdict(model_instance)
    return {}


def _coerce_to_input_model(
    input_cls: Type[Any],
    payload: Dict[str, Any],
    dispatch: AgentDispatch,
    agent_name: str,
) -> Any:
    """Project ``payload`` to the fields declared by ``input_cls`` and build it.

    Resolves issue #260: instantiating the per-agent ``input_model``
    (``ExperimentMonitorInput``, ``DriftMonitorInput``,
    ``ExperimentDesignerInput``) directly from the generic orchestrator
    payload fails because the generic payload carries
    ``user_context``/``parsed_query``/``span_id``/``dispatch_id``/
    ``execution_mode`` — none of which any wrapped model declares — and
    because some wrapped models require fields the generic payload does
    not carry (``features_to_monitor``, ``business_question``).

    Strategy:

    1. Merge ``dispatch.parameters`` (router-supplied per-agent kwargs)
       over the generic payload. ``parameters`` wins because the router put
       them there specifically for this agent.
    2. Apply per-agent ACs-#3 defaults via ``_wrapped_input_defaults``;
       defaults only fill values that are not already present in the merge.
    3. Project to the union of the model's declared field names. Models
       declaring no detectable fields (neither dataclass nor pydantic v2
       BaseModel) are constructed with the raw merged dict — the original
       splat path — to preserve backward compatibility.
    """
    parameters = dispatch.get("parameters") or {}
    merged: Dict[str, Any] = {**payload, **parameters}

    declared = _declared_field_names(input_cls)

    # AC-3: fill in per-agent defaults for fields the model declares but the
    # merge did not supply.
    if declared:
        defaults = _wrapped_input_defaults(agent_name, payload, dispatch)
        for field_name, default_value in defaults.items():
            if field_name in declared and field_name not in merged:
                merged[field_name] = default_value

    # If we can introspect declared fields, project the merge onto them. This
    # is the load-bearing fix: strict @dataclass models would otherwise raise
    # TypeError on user_context / parsed_query / span_id / etc.
    if declared:
        projected = {k: v for k, v in merged.items() if k in declared}
        return input_cls(**projected)

    # Backward-compat path: model we cannot introspect — splat the merge as-is
    # (the original behaviour before this fix). The caller wraps construction
    # exceptions into a structured AgentResult.error.
    return input_cls(**merged)


def _entity_value(payload: Dict[str, Any], entity_type: str) -> Optional[str]:
    """Return the first ``parsed_query.entities`` value of ``entity_type``.

    Mirrors the ``parsed_query.entities`` derivation used for ``drift_monitor``'s
    ``features_to_monitor`` default (KPI/feature mentions): walk the structured
    NLP entities the orchestrator already carries and return the first non-empty
    string ``value`` whose ``type`` matches. Returns ``None`` when no such entity
    exists (the caller then falls back to ``user_context`` or proceeds without).
    """
    parsed_query = payload.get("parsed_query") or {}
    entities = (parsed_query.get("entities") if isinstance(parsed_query, dict) else None) or []
    for ent in entities:
        if (
            isinstance(ent, dict)
            and ent.get("type") == entity_type
            and isinstance(ent.get("value"), str)
            and ent["value"].strip()
        ):
            return cast(str, ent["value"])
    return None


def _extract_brand_region(payload: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Derive ``(brand, region)`` for cohort resolution from the dispatch context.

    Order: structured ``parsed_query.entities`` (the NLP layer's typed
    extractions) first, then a ``user_context`` fallback (chat callers may stash
    ``brand``/``region`` there). Returns ``(None, None)`` when neither source
    names them — :func:`resolve_cohort_frame` then fails closed honestly.
    """
    brand = _entity_value(payload, "brand")
    region = _entity_value(payload, "region")

    user_context = payload.get("user_context") or {}
    if isinstance(user_context, dict):
        if brand is None and isinstance(user_context.get("brand"), str):
            brand = user_context["brand"] or None
        if region is None and isinstance(user_context.get("region"), str):
            region = user_context["region"] or None

    return brand, region


def _resolve_tool_composer_data(
    payload: Dict[str, Any],
) -> tuple[Optional[Any], Optional[str], bool]:
    """Best-effort resolve real estimation data for the ``tool_composer`` agent.

    Returns ``(frame, kpi_outcome, is_truncated)`` — ``is_truncated`` mirrors
    :attr:`KpiFrame.is_truncated` so this (orchestrator) path surfaces capped-data
    provenance exactly like the chatbot-tools path, instead of dropping it:

    * Issue #810 — if the query targets a defined KPI (e.g. *"what drove <brand>
      conversion ..."*), resolve that KPI's REAL substrate (e.g. triggers ⋈
      treatment_events for Conversion Rate) via
      :func:`src.services.kpi_resolution.resolve_kpi_frame`; ``kpi_outcome`` is the
      KPI's outcome column so the planner binds the causal outcome to the KPI.
    * Otherwise fall back to the patient-clinical cohort via
      :func:`src.services.cohort_resolution.resolve_cohort_frame` (``kpi_outcome``
      is ``None``).

    Both resolvers fail closed (``None``) and NEVER fabricate. Any exception is
    logged and swallowed so dispatch proceeds WITHOUT data; the composable tools
    then fail closed honestly. ``(None, None, False)`` means the dispatcher adds no
    ``data`` key, so the executor's duck-typed gate is not tripped by a ``None``.
    """
    brand, region = _extract_brand_region(payload)
    query = payload.get("query") if isinstance(payload.get("query"), str) else None

    # Issue #810: KPI-aware resolution takes precedence for KPI-outcome queries.
    if query:
        try:
            from src.services import kpi_resolution

            kpi = kpi_resolution.recognize_kpi(query)
            if kpi is not None:
                kpi_frame = kpi_resolution.resolve_kpi_frame(kpi, brand, region)
                if kpi_frame is not None:
                    logger.info(
                        "tool_composer dispatch: resolved KPI '%s' substrate (%d rows, "
                        "outcome=%s) for brand=%r region=%r.",
                        kpi_frame.kpi_name,
                        len(kpi_frame.frame),
                        kpi_frame.outcome_column,
                        brand,
                        region,
                    )
                    if kpi_frame.is_truncated:
                        logger.warning(
                            "tool_composer dispatch: KPI '%s' substrate is a TRUNCATED "
                            "sample (row cap hit) for brand=%r region=%r; results are "
                            "based on a partial slice.",
                            kpi_frame.kpi_name,
                            brand,
                            region,
                        )
                    return kpi_frame.frame, kpi_frame.outcome_column, kpi_frame.is_truncated
        except Exception as exc:  # noqa: BLE001 - best-effort, never block dispatch
            logger.warning(
                "tool_composer dispatch: KPI resolution failed, falling back to cohort: %s",
                exc,
            )

    if brand is None and region is None:
        # Nothing to resolve against; the cohort resolver would return None anyway.
        # Skip the import/call entirely so we don't take a DB round-trip for an
        # unscoped query.
        logger.info(
            "tool_composer dispatch: no brand/region in parsed_query/user_context; "
            "proceeding without estimation_data (tools fail closed)."
        )
        return None, None, False
    try:
        from src.services.cohort_resolution import resolve_cohort_frame

        frame = resolve_cohort_frame(brand, region)
        if frame is None:
            logger.info(
                "tool_composer dispatch: cohort_resolution returned no frame for "
                "brand=%r region=%r; proceeding without estimation_data.",
                brand,
                region,
            )
            return None, None, False
        logger.info(
            "tool_composer dispatch: resolved cohort frame (%d rows) for brand=%r region=%r.",
            len(frame),
            brand,
            region,
        )
        return frame, None, False
    except Exception as exc:  # noqa: BLE001 - best-effort, never block dispatch
        logger.warning(
            "tool_composer dispatch: cohort resolution failed for brand=%r "
            "region=%r, proceeding without estimation_data: %s",
            brand,
            region,
            exc,
        )
        return None, None, False


# ---------------------------------------------------------------------------
# Generic, data-driven input-resolver registry (audit F12/F13/F14)
# ---------------------------------------------------------------------------
#
# Some agents require structured analytical inputs that the generic orchestrator
# payload does not carry. Rather than a per-agent ``if`` chain in the dispatcher,
# a single ``INPUT_RESOLVERS`` registry maps each such agent to a resolver. Each
# resolver — given the prepared payload + dispatch — returns EITHER:
#
#   * a ``Dict`` of REAL, data-grounded inputs to apply, OR
#   * a ``NeedsStructuredInput`` fail-closed signal when the required inputs
#     cannot be honestly grounded in real data.
#
# Nothing is fabricated. Where the real data substrate exists (e.g. the KPI
# ``KpiFrame`` for heterogeneous_optimizer) the resolver BUILDS the inputs from
# real columns; where it does not (resource_optimizer's allocation problem,
# prediction_synthesizer's trained model) the resolver fails closed with a clear,
# actionable reason — and self-activates automatically once the data lands.


@dataclass(frozen=True)
class NeedsStructuredInput:
    """Tagged-union return from an input resolver: the agent's required inputs
    could NOT be grounded in real data, so dispatch must FAIL CLOSED with a clear
    message instead of fabricating inputs or raising a raw TypeError/ValueError.
    """

    agent_name: str
    missing: Tuple[str, ...]
    reason: str
    rest_endpoint: Optional[str] = None

    def to_error(self) -> str:
        endpoint = f" Supply them via {self.rest_endpoint}." if self.rest_endpoint else ""
        return (
            f"{self.agent_name} needs structured inputs that could not be grounded in "
            f"real data ({self.reason}); missing: {', '.join(self.missing)}.{endpoint} "
            "Failing closed — no values were fabricated."
        )


# A resolver maps (prepared agent_input, dispatch) -> real inputs OR a fail-closed signal.
InputResolver = Callable[
    [Dict[str, Any], AgentDispatch], Union[Dict[str, Any], NeedsStructuredInput]
]


def _resolve_tool_composer_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Dict[str, Any]:
    """Resolve the real cohort/KPI substrate for ``tool_composer`` (F2(a)/#810).

    Returns a dict carrying ``data`` (a real cohort/KPI frame) plus optional
    ``kpi_outcome``/``kpi_truncated`` provenance. tool_composer NEVER fails closed
    at dispatch — when no frame resolves it proceeds WITHOUT ``data`` and its
    composable tools fail closed honestly — so this resolver always returns a
    (possibly empty) dict, never ``NeedsStructuredInput``.
    """
    out: Dict[str, Any] = {}
    frame, kpi_outcome, kpi_truncated = _resolve_tool_composer_data(agent_input)
    if frame is not None:
        out["data"] = frame
        if kpi_outcome is not None:
            out["kpi_outcome"] = kpi_outcome
        if kpi_truncated:
            out["kpi_truncated"] = True
    return out


# heterogeneous_optimizer's tier0 passthrough (cate_estimator.py) only engages a
# supplied frame at >= 100 rows; below that it would silently fall through to the
# Supabase/mock path, so the resolver fails closed rather than feed it.
_HET_MIN_ROWS = 100
_HET_REQUIRED = ("treatment_var", "outcome_var", "effect_modifiers")


def _resolve_heterogeneous_optimizer_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Build heterogeneous_optimizer's causal spec from REAL data, or fail closed.

    (1) An explicit analyst-supplied spec in ``dispatch.parameters`` wins — that
        is a deliberate, honest choice. (2) Otherwise BUILD from the real KPI
        substrate: recognize the KPI, materialize the real ``KpiFrame``, and bind
        ``treatment_var``/``outcome_var``/``effect_modifiers`` to the frame's REAL
        columns (the KPI's defined treatment, its outcome, and the remaining real
        driver columns), threading the frame via ``tier0_data``. (3) When no KPI
        substrate (with a defined treatment and enough real rows) can be resolved,
        fail closed — never a fabricated treatment/outcome/modifier.
    """
    params = dispatch.get("parameters") or {}

    # (1) explicit analyst-supplied causal spec passes through verbatim.
    if all(params.get(k) for k in _HET_REQUIRED):
        passthrough = (
            "treatment_var",
            "outcome_var",
            "effect_modifiers",
            "segment_vars",
            "data_source",
            "filters",
            "tier0_data",
            "confounders",
            "role_attributions",
        )
        out: Dict[str, Any] = {k: params[k] for k in passthrough if params.get(k) is not None}
        out.setdefault("segment_vars", [])
        out.setdefault("data_source", "router_parameters")
        return out

    # (2) build the substrate from the real KPI frame.
    query = agent_input.get("query")
    brand, region = _extract_brand_region(agent_input)
    try:
        from src.services import kpi_resolution

        kpi = kpi_resolution.recognize_kpi(query)
        if kpi is not None:
            kpi_frame = kpi_resolution.resolve_kpi_frame(kpi, brand, region)
            treatment = getattr(kpi_frame, "treatment_column", None)
            if kpi_frame is not None and treatment and len(kpi_frame.frame) >= _HET_MIN_ROWS:
                # Exclude the treatment AND its raw source column from the effect
                # modifiers — the source is a deterministic function of the
                # treatment, so using it as a modifier would leak the treatment
                # into itself (invalid heterogeneity).
                excluded = {treatment, getattr(kpi_frame, "treatment_source_column", None)}
                modifiers = [c for c in kpi_frame.driver_columns if c not in excluded]
                if modifiers:
                    logger.info(
                        "heterogeneous_optimizer dispatch: built causal spec from KPI '%s' "
                        "substrate (treatment=%s, outcome=%s, modifiers=%s, %d real rows).",
                        kpi_frame.kpi_name,
                        treatment,
                        kpi_frame.outcome_column,
                        modifiers,
                        len(kpi_frame.frame),
                    )
                    return {
                        "treatment_var": treatment,
                        "outcome_var": kpi_frame.outcome_column,
                        "effect_modifiers": modifiers,
                        "segment_vars": modifiers,
                        "data_source": f"kpi_substrate:{kpi_frame.kpi_id}",
                        "tier0_data": kpi_frame.frame,
                    }
    except Exception as exc:  # noqa: BLE001 - best-effort; fail closed below
        logger.warning(
            "heterogeneous_optimizer dispatch: KPI substrate build failed (%s); failing closed.",
            exc,
        )

    # (3) cannot ground in real data → fail closed (no fabricated causal spec).
    return NeedsStructuredInput(
        agent_name="heterogeneous_optimizer",
        missing=_HET_REQUIRED,
        reason=(
            "no recognized KPI substrate with a defined treatment and "
            f">={_HET_MIN_ROWS} real rows to bind the causal spec; a chat query "
            "alone cannot name the treatment/outcome/effect-modifier columns"
        ),
        rest_endpoint="POST /segments/analyze",
    )


def _resolve_resource_optimizer_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Pass through a REAL allocation problem from ``dispatch.parameters``, else
    fail closed.

    ``optimize`` requires ``allocation_targets`` (entities + response
    coefficients) and ``constraints`` (budgets). These constitute a fully
    specified optimization problem. There is NO data substrate today that
    materializes per-entity response coefficients (the rep-activity/allocation
    rows are absent), so fabricating them is forbidden. When an API/router caller
    supplies a real problem in ``parameters`` it passes through (as a CLEAN kwarg
    set — eliminating the generic-payload leak); otherwise fail closed.
    """
    params = dispatch.get("parameters") or {}
    targets = params.get("allocation_targets")
    constraints = params.get("constraints") or []
    # The optimizer's problem_formulator REQUIRES a budget constraint; passing
    # targets with no budget would fail internally (and would otherwise be counted
    # as a successful dispatch). Require BOTH a real target set and a real budget
    # constraint, else fail closed naming exactly what is missing.
    has_budget = any(
        isinstance(c, dict) and str(c.get("constraint_type")) == "budget" for c in constraints
    )
    if targets and has_budget:
        out: Dict[str, Any] = {
            "allocation_targets": targets,
            "constraints": constraints,
            "query": agent_input.get("query") or "",
        }
        session_id = agent_input.get("session_id")
        if session_id is not None:
            out["session_id"] = session_id
        for opt in ("resource_type", "objective", "solver_type", "run_scenarios", "scenario_count"):
            if params.get(opt) is not None:
                out[opt] = params[opt]
        return out

    missing: List[str] = []
    if not targets:
        missing.append("allocation_targets")
    if not has_budget:
        missing.append("constraints (with a budget constraint)")
    return NeedsStructuredInput(
        agent_name="resource_optimizer",
        missing=tuple(missing),
        reason=(
            "a real allocation problem (entities with response coefficients AND a budget "
            "constraint) must be supplied; no per-entity allocation/response substrate exists "
            "in the data (rep-activity allocation rows are absent) to build one without "
            "inventing entities, response coefficients and budgets"
        ),
        rest_endpoint="POST /resources/optimize",
    )


def _resolve_prediction_synthesizer_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Pass through a REAL prediction request from ``dispatch.parameters``, else
    fail closed.

    ``synthesize`` requires a specific ``entity_id`` and ``prediction_target``.
    There is no registered champion model today and a chat query names no specific
    entity, so picking "the first entity the user happened to mention" would
    fabricate a prediction for an arbitrary unit. When an API/router caller
    supplies a real (entity_id, prediction_target) it passes through (clean kwarg
    set); otherwise fail closed.
    """
    params = dispatch.get("parameters") or {}
    entity_id = params.get("entity_id")
    target = params.get("prediction_target")
    if entity_id and target:
        out: Dict[str, Any] = {
            "entity_id": entity_id,
            "prediction_target": target,
            "query": agent_input.get("query") or "",
        }
        session_id = agent_input.get("session_id")
        if session_id is not None:
            out["session_id"] = session_id
        for opt in (
            "features",
            "entity_type",
            "time_horizon",
            "models_to_use",
            "ensemble_method",
            "include_context",
        ):
            if params.get(opt) is not None:
                out[opt] = params[opt]
        return out

    return NeedsStructuredInput(
        agent_name="prediction_synthesizer",
        missing=("entity_id", "prediction_target"),
        reason=(
            "no registered champion model and no specific real entity to predict for; "
            "a chat query names neither, so a prediction cannot be synthesized without "
            "inventing an entity — supply a specific entity_id and prediction_target as "
            "structured dispatch parameters"
        ),
        rest_endpoint=None,
    )


# Single source of truth: agent_name -> input resolver. Add a resolver here, not
# an ``if`` branch in ``_dispatch_agent`` (#F12/F13/F14).
INPUT_RESOLVERS: Dict[str, InputResolver] = {
    "tool_composer": _resolve_tool_composer_input,
    "heterogeneous_optimizer": _resolve_heterogeneous_optimizer_input,
    "resource_optimizer": _resolve_resource_optimizer_input,
    "prediction_synthesizer": _resolve_prediction_synthesizer_input,
}


# Resolver-backed agents that must FAIL CLOSED when the agent's OWN output reports
# an internal failure (``status == "failed"``). A domain failure (e.g. no models
# registered, infeasible optimization) must never be reported as a successful
# dispatch — otherwise the dispatcher's transport-level ``success=True`` would
# launder an empty/failed analysis into a "success" (#F12/F13/F14). tool_composer
# is intentionally EXCLUDED: its success semantics are governed by F6 (#827)
# tool-level fail-closed + the synthesizer's filtering, not a status field.
_FAIL_CLOSED_ON_FAILED_STATUS = frozenset(
    {"heterogeneous_optimizer", "resource_optimizer", "prediction_synthesizer"}
)


def _agent_failed(agent_name: str, result: Dict[str, Any]) -> Optional[str]:
    """Return a failure detail string if ``result`` from ``agent_name`` reports an
    internal domain failure that must fail the dispatch closed, else ``None``.

    Only applies to the resolver-backed fail-closed agents and only on an explicit
    ``status == "failed"`` (the contract all three set on their failure paths,
    e.g. heterogeneous_optimizer agent.py: ``"failed" if errors else "completed"``).
    """
    if agent_name not in _FAIL_CLOSED_ON_FAILED_STATUS:
        return None
    if str(result.get("status")) != "failed":
        return None
    errors = result.get("errors") or []
    detail = "; ".join(str(e.get("error", e)) if isinstance(e, dict) else str(e) for e in errors)
    return detail or "agent reported status=failed"


def _generate_dispatch_id() -> str:
    """Generate unique dispatch identifier."""
    return f"disp_{uuid.uuid4().hex[:16]}"


def _generate_span_id() -> str:
    """Generate unique span identifier for observability."""
    return f"span_{uuid.uuid4().hex[:16]}"


class DispatcherNode:
    """Parallel agent dispatch with timeout handling."""

    def __init__(
        self,
        agent_registry: Optional[Dict[str, Any]] = None,
        *,
        allow_mock: bool = False,
    ):
        """Initialize dispatcher with agent registry.

        Args:
            agent_registry: Dict mapping agent_name to agent instance.
            allow_mock: TEST-ONLY. When ``True``, a dispatch to an agent that is
                absent from ``agent_registry`` returns a canned
                :meth:`_mock_agent_execution` response (used by unit tests that
                exercise routing/timeout/fallback mechanics without instantiating
                real agents). MUST stay ``False`` in production (the default): a
                missing agent then FAILS CLOSED with a structured error instead of
                fabricating plausible-but-fake analytics values (#814).
        """
        self.agents = agent_registry or {}
        self.allow_mock = allow_mock

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute agent dispatch.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with agent results
        """
        start_time = time.time()

        dispatch_plan = state.get("dispatch_plan") or []
        parallel_groups = state.get("parallel_groups") or []
        all_results: List[AgentResult] = []

        # Execute each parallel group sequentially
        for group in parallel_groups:
            group_dispatches = [d for d in dispatch_plan if d["agent_name"] in group]

            # Run agents in parallel within group
            tasks = [self._dispatch_agent(d, state) for d in group_dispatches]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for dispatch, result in zip(group_dispatches, group_results, strict=False):
                if isinstance(result, Exception):
                    # Handle unexpected exceptions from asyncio.gather
                    failed_result = AgentResult(
                        agent_name=dispatch["agent_name"],
                        success=False,
                        result=None,
                        error=str(result),
                        latency_ms=0,
                    )
                    all_results.append(failed_result)

                    # Try fallback if available
                    fallback_agent = dispatch.get("fallback_agent")
                    if fallback_agent:
                        fallback_result = await self._dispatch_fallback(str(fallback_agent), state)
                        all_results.append(fallback_result)
                elif isinstance(result, dict) and not result.get("success", True):
                    # AgentResult returned with success=False
                    all_results.append(result)  # type: ignore[arg-type]

                    # Try fallback if available
                    fallback_agent2 = dispatch.get("fallback_agent")
                    if fallback_agent2:
                        fallback_result = await self._dispatch_fallback(str(fallback_agent2), state)
                        all_results.append(fallback_result)
                else:
                    # Result is AgentResult (TypedDict cannot use isinstance, check dict)
                    if isinstance(result, dict) and "agent_name" in result:
                        all_results.append(result)

        dispatch_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "agent_results": all_results,
            "dispatch_latency_ms": dispatch_time,
            "current_phase": "synthesizing",
        }

    async def _dispatch_agent(
        self, dispatch: AgentDispatch, state: OrchestratorState
    ) -> AgentResult:
        """Dispatch to a single agent with timeout.

        Real agents are reached via the per-agent dispatch spec in
        ``AGENT_METHOD_MAP`` (method name, async vs sync, kwargs splat, optional
        Pydantic input model).

        When the agent name is absent from the registry the dispatcher FAILS
        CLOSED (#814): it returns a structured ``success=False`` error rather
        than a fabricated result, so a partial/empty registry (e.g. an agent that
        failed to instantiate) can never surface plausible-but-fake analytics to
        the user. The canned :meth:`_mock_agent_execution` scaffold is reachable
        ONLY when ``allow_mock=True`` (test-only; unit tests that exercise
        routing without instantiating real agents).
        """
        agent_name = dispatch["agent_name"]
        start_time = time.time()

        if agent_name not in self.agents:
            if self.allow_mock:
                return await self._mock_agent_execution(dispatch, state)
            latency = int((time.time() - start_time) * 1000)
            logger.warning(
                "dispatcher: no registry entry for agent %r and allow_mock is off; "
                "failing closed (no fabricated fallback).",
                agent_name,
            )
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=(
                    f"Agent '{agent_name}' is not available in the dispatcher registry; "
                    "dispatch fails closed (no fabricated result)."
                ),
                latency_ms=latency,
            )

        agent = self.agents[agent_name]
        timeout_ms = dispatch["timeout_ms"]
        spec = get_method_spec(agent_name)

        try:
            agent_input = self._prepare_agent_input(state, dispatch)

            # Generic, data-driven input resolution (#F12/F13/F14). A per-agent
            # resolver either returns REAL inputs to apply or a structured
            # ``NeedsStructuredInput`` fail-closed signal — replacing the old
            # ``tool_composer`` special-case. Resolvers NEVER fabricate inputs.
            resolver = INPUT_RESOLVERS.get(agent_name)
            if resolver is not None:
                resolved = resolver(agent_input, dispatch)
                if isinstance(resolved, NeedsStructuredInput):
                    latency = int((time.time() - start_time) * 1000)
                    logger.info(
                        "dispatcher: %r failing closed — %s",
                        agent_name,
                        resolved.reason,
                    )
                    return AgentResult(
                        agent_name=agent_name,
                        success=False,
                        result=None,
                        error=resolved.to_error(),
                        latency_ms=latency,
                    )
                if spec.uses_kwargs:
                    # kwargs agents (optimize/synthesize): the resolver returns the
                    # COMPLETE clean kwarg set, eliminating the generic-payload leak
                    # (user_context/parsed_query/span_id/...) that previously raised
                    # TypeError on the ``method(**agent_input)`` splat.
                    agent_input = resolved
                else:
                    # single-dict agents (run): merge the resolved real inputs;
                    # the agent reads the keys it declares and ignores extras.
                    agent_input.update(resolved)

            # Wrap input in a Pydantic / dataclass model when the agent expects
            # one (e.g. DriftMonitorInput, ExperimentMonitorInput).
            #
            # Issue #260: the generic orchestrator payload carries fields the
            # wrapped models don't declare (user_context, parsed_query, span_id,
            # ...) and is missing required fields some wrapped models declare
            # (features_to_monitor, business_question). ``_coerce_to_input_model``
            # projects the merged payload onto the model's declared fields and
            # supplies per-agent defaults from the dispatch context.
            if spec.input_model and spec.input_module:
                try:
                    input_module = importlib.import_module(spec.input_module)
                    input_cls = getattr(input_module, spec.input_model)
                    agent_input = _coerce_to_input_model(
                        input_cls,
                        cast(Dict[str, Any], agent_input),
                        dispatch,
                        agent_name,
                    )
                except (ImportError, AttributeError, TypeError, ValueError) as e:
                    # ValueError covers pydantic.ValidationError (subclass) so a
                    # bad input dict produces a structured AgentResult.error
                    # instead of propagating as an unhandled exception.
                    latency = int((time.time() - start_time) * 1000)
                    return AgentResult(
                        agent_name=agent_name,
                        success=False,
                        result=None,
                        error=f"Failed to build {spec.input_model}: {e}",
                        latency_ms=latency,
                    )

            method = getattr(agent, spec.method, None)
            if method is None:
                latency = int((time.time() - start_time) * 1000)
                return AgentResult(
                    agent_name=agent_name,
                    success=False,
                    result=None,
                    error=(
                        f"Agent '{agent_name}' is registered but has no "
                        f"method '{spec.method}'. Check AGENT_METHOD_MAP."
                    ),
                    latency_ms=latency,
                )

            timeout_seconds = timeout_ms / 1000

            # When the spec declares BOTH ``input_model`` AND ``uses_kwargs``,
            # the validated model must be re-flattened back into a kwargs dict
            # before splatting — splatting a dataclass/pydantic instance with
            # ``**`` raises TypeError. No production AGENT_METHOD_MAP entry
            # combines both today; this guard makes the dispatcher robust
            # against future additions (codex MED-tracker on PR #275 / #260).
            if (
                spec.input_model
                and spec.input_module
                and spec.uses_kwargs
                and not isinstance(agent_input, dict)
            ):
                agent_input = _to_kwargs_dict(agent_input)

            if spec.is_async:
                if spec.uses_kwargs:
                    coro = method(**agent_input)
                else:
                    coro = method(agent_input)
                raw_result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            else:
                # asyncio.get_event_loop() is deprecated in Python 3.12+ when
                # called outside a running loop; this dispatch path is always
                # inside an active loop (we're in an async method), so
                # get_running_loop() is the correct API.
                loop = asyncio.get_running_loop()
                if spec.uses_kwargs:
                    call = functools.partial(method, **agent_input)
                else:
                    call = functools.partial(method, agent_input)
                raw_result = await asyncio.wait_for(
                    loop.run_in_executor(None, call), timeout=timeout_seconds
                )

            latency = int((time.time() - start_time) * 1000)
            normalized = _normalize_agent_result(raw_result)

            # Domain-failure guard: a resolver-backed agent that ran but reported
            # an internal failure (e.g. prediction_synthesizer with no registered
            # models, resource_optimizer with an infeasible/under-specified problem)
            # must FAIL CLOSED — never be laundered into a successful dispatch.
            failure_detail = _agent_failed(agent_name, normalized)
            if failure_detail is not None:
                logger.info(
                    "dispatcher: %r ran but reported status=failed; failing closed (%s).",
                    agent_name,
                    failure_detail,
                )
                return AgentResult(
                    agent_name=agent_name,
                    success=False,
                    result=normalized,
                    error=(
                        f"{agent_name} could not produce a real result "
                        f"({failure_detail}); failing closed — no values were fabricated."
                    ),
                    latency_ms=latency,
                )

            return AgentResult(
                agent_name=agent_name,
                success=True,
                result=normalized,
                error=None,
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=f"Agent timed out after {timeout_ms}ms",
                latency_ms=timeout_ms,
            )
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=str(e),
                latency_ms=latency,
            )

    async def _mock_agent_execution(
        self, dispatch: AgentDispatch, state: OrchestratorState
    ) -> AgentResult:
        """Mock agent execution — TEST-ONLY.

        Returns canned narratives with fabricated illustrative values for unit
        tests that exercise dispatch mechanics (routing/parallel/timeout/
        fallback) without instantiating real agents. This is reachable ONLY when
        the dispatcher is constructed with ``allow_mock=True``; in production
        (``allow_mock=False``, the default) a missing agent fails closed instead
        (see :meth:`_dispatch_agent`). It must never run on production traffic.

        Args:
            dispatch: Dispatch configuration
            state: Current state

        Returns:
            Mock agent result
        """
        agent_name = dispatch["agent_name"]

        # Simulate processing time
        await asyncio.sleep(0.05)  # 50ms

        # Mock responses by agent type
        mock_responses = {
            "causal_impact": {
                "narrative": "Analysis shows that HCP engagement has a significant positive effect on patient conversion (ATE=0.12, p<0.01).",
                "recommendations": [
                    "Increase HCP engagement in oncology segment",
                    "Focus on high-potential HCPs",
                ],
                "confidence": 0.87,
            },
            "gap_analyzer": {
                "narrative": "Identified 3 key gaps with combined ROI potential of $2.5M: underperforming regions, undertreated patients, and suboptimal messaging.",
                "recommendations": [
                    "Expand coverage in Northeast region",
                    "Increase patient identification initiatives",
                ],
                "confidence": 0.82,
            },
            "heterogeneous_optimizer": {
                "narrative": "Segment-level analysis reveals heterogeneous treatment effects. Oncology specialists show 2x higher response rate compared to general practitioners.",
                "recommendations": [
                    "Differentiate strategies by HCP specialty",
                    "Allocate more resources to oncology segment",
                ],
                "confidence": 0.79,
            },
            "prediction_synthesizer": {
                "narrative": "Forecast indicates 15% increase in conversions over next quarter, driven by recent HCP engagement initiatives.",
                "recommendations": [
                    "Maintain current engagement levels",
                    "Monitor conversion trends weekly",
                ],
                "confidence": 0.75,
            },
            "explainer": {
                "narrative": f"Based on the query '{state.get('query', '')}', here's a detailed explanation of the analysis approach and findings.",
                "recommendations": ["Review additional metrics", "Compare with benchmarks"],
                "confidence": 0.70,
            },
            "resource_optimizer": {
                "narrative": "Optimal resource allocation suggests reallocating 20% of budget from low-ROI channels to high-performing HCP engagement.",
                "recommendations": [
                    "Reallocate budget to top-performing channels",
                    "Monitor ROI weekly",
                ],
                "confidence": 0.81,
            },
            "health_score": {
                "narrative": "System health is nominal. All models performing within expected ranges. No critical issues detected.",
                "recommendations": ["Continue monitoring", "Schedule quarterly review"],
                "confidence": 0.95,
            },
            "drift_monitor": {
                "narrative": "Slight data drift detected in HCP engagement patterns (0.05 Jensen-Shannon divergence). Within acceptable thresholds.",
                "recommendations": [
                    "Monitor drift trends",
                    "Consider retraining in 2 months",
                ],
                "confidence": 0.88,
            },
            "experiment_designer": {
                "narrative": "Designed A/B test for HCP engagement strategy. Required sample size: 500 HCPs per arm. Expected runtime: 8 weeks.",
                "recommendations": [
                    "Preregister experiment",
                    "Set up monitoring dashboard",
                ],
                "confidence": 0.83,
            },
            "feedback_learner": {
                "narrative": "Analyzed feedback from previous campaigns. Key learning: personalized messaging increases engagement by 25%.",
                "recommendations": [
                    "Implement personalization in next campaign",
                    "Track engagement metrics",
                ],
                "confidence": 0.76,
            },
            "cohort_constructor": {
                "narrative": "Cohort construction complete. Applied inclusion/exclusion criteria to patient population.",
                "recommendations": [
                    "Review eligibility log for detailed filtering breakdown",
                    "Monitor cohort size against SLA thresholds",
                ],
                "confidence": 0.92,
                "eligible_count": 150,
                "total_input": 500,
                "eligibility_rate": 0.30,
            },
        }

        # Get mock response or default
        mock_result = mock_responses.get(
            agent_name,
            {
                "narrative": f"Mock response from {agent_name} agent.",
                "recommendations": ["Follow up with additional analysis"],
                "confidence": 0.70,
            },
        )

        return AgentResult(
            agent_name=agent_name,
            success=True,
            result=mock_result,
            error=None,
            latency_ms=50,
        )

    def _prepare_agent_input(
        self, state: OrchestratorState, dispatch: AgentDispatch
    ) -> Dict[str, Any]:
        """Prepare input for specific agent.

        Args:
            state: Current state
            dispatch: Dispatch configuration

        Returns:
            Agent input data with contract-required pass-through fields
        """
        # Generate dispatch_id if not already set
        dispatch_id = dispatch.get("dispatch_id") or _generate_dispatch_id()

        # Generate span_id for observability
        span_id = _generate_span_id()

        agent_input: Dict[str, Any] = {
            "query": state.get("query"),
            "user_context": state.get("user_context", {}),
            "parameters": dispatch.get("parameters", {}),
            # Contract: BaseAgentState pass-through fields
            "session_id": state.get("session_id"),
            "parsed_query": state.get("parsed_query"),
            # Contract: Orchestrator dispatch fields
            "dispatch_id": dispatch_id,
            "span_id": span_id,
            "execution_mode": dispatch.get("execution_mode", "sequential"),
        }

        # Per-agent input resolution (real cohort/KPI data for tool_composer, the
        # causal spec for heterogeneous_optimizer, etc.) now lives in the generic
        # ``INPUT_RESOLVERS`` registry, applied in ``_dispatch_agent`` after this
        # builds the generic payload (#F12/F13/F14). This method only assembles
        # the contract pass-through fields.
        return agent_input

    async def _dispatch_fallback(self, agent_name: str, state: OrchestratorState) -> AgentResult:
        """Dispatch to fallback agent.

        Args:
            agent_name: Fallback agent name
            state: Current state

        Returns:
            Fallback agent result
        """
        fallback_dispatch = AgentDispatch(
            agent_name=agent_name,
            priority="low",  # Contract: Literal priority type
            parameters={},
            timeout_ms=30000,
            fallback_agent=None,
        )
        return await self._dispatch_agent(fallback_dispatch, state)


def _normalize_agent_result(raw: Any) -> Dict[str, Any]:
    """Coerce an agent's return value to the dict shape AgentResult expects.

    Agents return one of: a TypedDict (already a dict), a dataclass output
    object (e.g. ExperimentMonitorOutput, DriftMonitorOutput), or a plain
    string. ``isinstance(raw, dict)`` short-circuits the TypedDict case;
    dataclasses are flattened via ``__dict__``; anything else is wrapped.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return cast(Dict[str, Any], raw)
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        try:
            result = raw.to_dict()
            if isinstance(result, dict):
                return cast(Dict[str, Any], result)
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(raw, "__dict__"):
        return {k: v for k, v in vars(raw).items() if not k.startswith("_")}
    return {"raw_output": str(raw)}


# Export for use in graph
async def dispatch_to_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for agent dispatch (the graph's registry-less else-branch).

    Constructs a dispatcher with NO registry and ``allow_mock=False``, so every
    dispatch FAILS CLOSED with a structured error rather than fabricating a
    result (#814). This branch runs only when the orchestrator graph was built
    without an ``agent_registry``; the production graph always passes a populated
    registry and uses :meth:`DispatcherNode.execute` directly.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    dispatcher = DispatcherNode()
    result = await dispatcher.execute(cast(OrchestratorState, state))
    return cast(Dict[str, Any], result)
