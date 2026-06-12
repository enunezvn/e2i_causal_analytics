"""Dispatcher node for orchestrator agent.

Parallel agent dispatch with timeout handling.
"""

import asyncio
import dataclasses
import functools
import importlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

from src.repositories.provenance import coerce_provenance_flag

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
    # Validation runs opt into the synthetic substrate explicitly (the #851
    # include_synthetic plumb, default False): every kpi_resolution source read
    # default-excludes is_synthetic rows, so on a clean substrate a real-mode
    # build correctly fails closed unless the caller opted in. Channels +
    # strict parsing are shared with the gap resolver (#880 — was a loose
    # ``bool()`` over filters/user_context, so ``"false"`` opted IN and the
    # chat-path ``parameters`` channels were silently ignored). The flag stays
    # dispatch-local: branch (1) explicit specs do no provenance read here and
    # the het agent takes no include_synthetic input, so nothing is forwarded
    # into the agent (unlike gap_analyzer's per-run connector opt-in).
    include_synthetic = _resolve_include_synthetic_opt_in(agent_input, params)
    try:
        from src.services import kpi_resolution

        kpi = kpi_resolution.recognize_kpi(query)
        if kpi is not None:
            kpi_frame = kpi_resolution.resolve_kpi_frame(
                kpi, brand, region, include_synthetic=include_synthetic
            )
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


# gap_analyzer's substrate probe scans single-column reads off business_metrics —
# the table the agent's production connector reads post-#856. Cap mirrors
# kpi_resolution._MAX_ROWS-style truncation safety on the distinct scans.
_GAP_PROBE_ROW_CAP = 5000
# Distinct-value paging bound: up to N pages of ROW_CAP rows (PK-ordered, paged to
# slice exhaustion). Bounds dispatch latency while staying correct as the brand's
# substrate grows past one page; hitting the bound warns (never fails closed).
_GAP_PROBE_MAX_PAGES = 4
_GAP_REQUIRED = ("metrics", "segments", "brand")


# Strict provenance opt-in parser (codex #874 R2). Lifted to the shared SSOT in
# src/repositories/provenance.py (#883 §4) so agents and celery tasks parse the
# flag identically without importing the orchestrator; this thin alias keeps the
# dispatcher-local name every existing caller/test references.
_coerce_provenance_flag = coerce_provenance_flag


def _resolve_include_synthetic_opt_in(agent_input: Dict[str, Any], params: Dict[str, Any]) -> bool:
    """Resolve the per-dispatch synthetic-provenance opt-in (#872/#877/#880).

    Shared by the gap_analyzer and heterogeneous_optimizer resolvers so the
    opt-in contract cannot drift between them again (gap_analyzer is het's
    fallback agent — the same dispatch must never run in two different
    provenance modes depending on which agent fires). Channels:

    * ``agent_input.filters`` — direct resolver invocations (validation gates);
    * ``agent_input.user_context`` — the only caller-stash field
      ``_prepare_agent_input`` threads through the live chat path;
    * ``parameters.filters`` and the explicit ``parameters.include_synthetic``,
      which WINS when present and non-None (an analyst's explicit choice beats
      the ambient opt-in channels). ``parameters`` is router-supplied and DOES
      flow on the chat path, unlike ``filters``.

    Every value is STRICTLY parsed via :func:`_coerce_provenance_flag`: an
    ambiguous value (``"false"``, ``"0"``, non-bool/non-str types) fails CLOSED
    to the real-mode default-exclude predicate.
    """
    channel_opt_in = (
        _coerce_provenance_flag((params.get("filters") or {}).get("include_synthetic"))
        or _coerce_provenance_flag((agent_input.get("filters") or {}).get("include_synthetic"))
        or _coerce_provenance_flag((agent_input.get("user_context") or {}).get("include_synthetic"))
    )
    explicit_flag = params.get("include_synthetic")
    return _coerce_provenance_flag(explicit_flag) if explicit_flag is not None else channel_opt_in


# The single segment dimension business_metrics actually carries (the benchmark
# store discovers its VALUES from the data; see BenchmarkStore._resolve_segment).
_GAP_SEGMENT_COLUMN = "region"


def _probe_gap_substrate(
    brand: str, include_synthetic: bool
) -> Tuple[List[str], Optional[str], List[str]]:
    """Probe the REAL ``business_metrics`` substrate for ``brand``.

    Returns ``(metric_names, canonical_brand, region_values)`` — every element
    discovered from real rows under the provenance predicate (default-exclude
    synthetic; ``include_synthetic=True`` opts in, the #851/#872 plumb).

    ``brand`` is an enum column, so canonicalization uses bounded per-candidate
    presence probes (``.eq(brand, <case-variant>).limit(1)``) rather than an
    unfiltered distinct scan: the table's physical order groups rows by brand, so a
    capped scan can miss a brand entirely, and a cased ``.eq`` miss on an enum
    raises 22P02 (caught per-candidate — an invalid label means "not this
    spelling", never an operational failure). ``([], None, [])`` means the
    substrate genuinely has no rows for the brand.
    """
    from src.repositories import get_supabase_client
    from src.repositories.provenance import apply_provenance_filter

    client = get_supabase_client()

    def _brand_present(candidate: str) -> bool:
        q = client.table("business_metrics").select("brand").eq("brand", candidate)
        q = apply_provenance_filter(q, include_synthetic)
        try:
            rows = getattr(q.limit(1).execute(), "data", None) or []
        except Exception as exc:
            # 22P02 = candidate is not a valid enum label (a casing miss), NOT an
            # operational error — try the next case variant. Anything else raises.
            if getattr(exc, "code", None) == "22P02":
                return False
            raise
        return bool(rows)

    raw = str(brand).strip()
    candidates = list(dict.fromkeys([raw, raw.capitalize(), raw.title(), raw.lower(), raw.upper()]))
    canonical = next((c for c in candidates if c and _brand_present(c)), None)
    if canonical is None:
        return [], None, []

    # Distinct metric/region discovery: page through the brand slice until
    # EXHAUSTED, bounded by _GAP_PROBE_MAX_PAGES. No early "saturation" stop — a
    # full page adding no new distinct values proves nothing about later pages
    # (PostgREST gives no distinct/ordering guarantee per page; codex #874 R2).
    # Ordered by the PK so OFFSET pagination is deterministic under concurrent
    # writes. A single capped prefix could silently omit metric names as the
    # substrate grows; failing closed on the bound would be worse (MORE data must
    # never disable a bindable substrate) — so page to the bound and warn.
    metrics_set: set[str] = set()
    regions_set: set[str] = set()
    for page in range(_GAP_PROBE_MAX_PAGES):
        q = client.table("business_metrics").select("metric_name,region").eq("brand", canonical)
        q = apply_provenance_filter(q, include_synthetic)
        offset = page * _GAP_PROBE_ROW_CAP
        rows = (
            getattr(
                q.order("metric_id").range(offset, offset + _GAP_PROBE_ROW_CAP - 1).execute(),
                "data",
                None,
            )
            or []
        )
        for r in rows:
            if isinstance(r, dict):
                if r.get("metric_name"):
                    metrics_set.add(str(r["metric_name"]))
                if r.get("region"):
                    regions_set.add(str(r["region"]))
        if len(rows) < _GAP_PROBE_ROW_CAP:
            break  # slice exhausted
    else:
        logger.warning(
            "gap_analyzer dispatch: business_metrics metric/region scan for brand=%s "
            "hit the %d-page x %d-row probe bound before exhausting the slice; "
            "distinct values beyond it could be missed.",
            canonical,
            _GAP_PROBE_MAX_PAGES,
            _GAP_PROBE_ROW_CAP,
        )
    return sorted(metrics_set), canonical, sorted(regions_set)


def _resolve_gap_analyzer_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Build gap_analyzer's required structured inputs from REAL data, or fail closed.

    gap_analyzer requires ``metrics``/``segments``/``brand`` (agent.py
    ``_validate_input``); the generic chat payload carries none of them, so every
    chat dispatch died in ~7ms with a raw ``Missing required field: metrics``
    ValueError (#874). Mirroring ``_resolve_heterogeneous_optimizer_input``:

    (1) An explicit analyst-supplied input set in ``dispatch.parameters`` wins.
    (2) Otherwise DERIVE the inputs from the real ``business_metrics`` substrate —
        the exact table the agent's production connector reads post-#856: the
        brand's real ``metric_name`` values become ``metrics`` and the real
        segment dimension (``region``, whose values exist in the data) becomes
        ``segments``. ``include_synthetic`` is read from the opt-in channels
        shared with the het resolver (#872/#880) and ALSO forwarded into the agent
        input: the registry's agent instance is constructed real-mode, so the
        per-run flag is what lets an opted-in validation dispatch actually read
        the synthetic substrate (gap_detector resolves a per-run connector pair).
    (3) When the (provenance-filtered) substrate genuinely has no rows — or the
        chat query names no brand to scope the read — fail closed with an
        actionable ``NeedsStructuredInput``; nothing fabricated.
    """
    params = dispatch.get("parameters") or {}

    # Provenance opt-in, resolved ONCE for both branches via the shared helper
    # (channels + precedence documented on _resolve_include_synthetic_opt_in;
    # now shared with the het resolver, #880). Dropping the channel opt-in on
    # the explicit-params path would silently run the analysis against the
    # wrong provenance mode (codex #874 R1 HIGH). All values are STRICTLY
    # parsed (codex R2): an ambiguous value stays real-mode.
    include_synthetic = _resolve_include_synthetic_opt_in(agent_input, params)

    # (1) explicit analyst-supplied inputs pass through verbatim.
    if params.get("metrics") and params.get("segments") and params.get("brand"):
        passthrough = (
            "metrics",
            "segments",
            "brand",
            "time_period",
            "gap_type",
            "filters",
            "tier0_data",
            "min_gap_threshold",
            "max_opportunities",
            "instrument_specs",
        )
        out_explicit: Dict[str, Any] = {
            k: params[k] for k in passthrough if params.get(k) is not None
        }
        out_explicit["include_synthetic"] = include_synthetic
        return out_explicit

    # (2) derive from the real business_metrics substrate. A partial structured
    # ``parameters.brand`` wins over the chat-derived brand (the router put it
    # there specifically for this agent — same precedence as _coerce_to_input_model).
    params_brand = params.get("brand")
    if isinstance(params_brand, str) and params_brand.strip():
        brand: Optional[str] = params_brand
    else:
        brand, _ = _extract_brand_region(agent_input)
    if brand:
        try:
            metrics, canonical_brand, regions = _probe_gap_substrate(brand, include_synthetic)
            if canonical_brand and metrics and regions:
                logger.info(
                    "gap_analyzer dispatch: bound real business_metrics substrate for "
                    "brand=%s (metrics=%s, segment=%s with %d values, "
                    "include_synthetic=%s).",
                    canonical_brand,
                    metrics,
                    _GAP_SEGMENT_COLUMN,
                    len(regions),
                    include_synthetic,
                )
                out: Dict[str, Any] = {
                    "metrics": metrics,
                    "segments": [_GAP_SEGMENT_COLUMN],
                    "brand": canonical_brand,
                    "include_synthetic": include_synthetic,
                }
                # Config-only router overrides apply on top of the derived inputs
                # (they don't constitute the required trio, so they reach here).
                for opt in ("gap_type", "time_period", "min_gap_threshold", "max_opportunities"):
                    if params.get(opt) is not None:
                        out[opt] = params[opt]
                return out
        except Exception as exc:  # noqa: BLE001 - best-effort; fail closed below
            logger.warning(
                "gap_analyzer dispatch: business_metrics substrate probe failed (%s); "
                "failing closed.",
                exc,
            )

    # (3) cannot ground in real data → fail closed (no fabricated metrics/segments).
    if brand:
        reason = (
            f"the business_metrics substrate has no rows for brand {brand!r} under the "
            "active provenance mode "
            f"({'synthetic opted in' if include_synthetic else 'real-mode default-exclude'}), "
            "so the metrics/segments to analyze cannot be derived from real data"
        )
    else:
        reason = (
            "the dispatch names no brand (parameters / parsed_query entities / "
            "user_context), so there is no real business_metrics substrate to derive "
            "metrics/segments from"
        )
    return NeedsStructuredInput(
        agent_name="gap_analyzer",
        missing=_GAP_REQUIRED,
        reason=reason,
        rest_endpoint="POST /api/gaps/analyze",
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


def _successful_upstream_results(agent_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the REAL result payloads from the upstream ``AgentResult`` list.

    ``_prepare_agent_input`` threads ``state["agent_results"]`` into the payload
    (#883 §3): on the failure-fallback path ``execute()`` enriches it with the
    results accumulated this turn, and on a checkpointer-resumed multi-turn state
    it carries prior turns' results (the ``operator.add`` reducer). Only
    SUCCESSFUL results with a non-empty result dict qualify — a failed upstream
    dispatch has nothing real to explain. The explainer's own prior outputs are
    excluded: re-explaining an explanation is circular, not analysis.

    Each qualifying result dict is passed through as-is, with the dispatching
    agent's REAL name attached (``setdefault`` — never overwriting a result's own
    field) so the narrative can attribute findings to their source agent.
    """
    raw = agent_input.get("agent_results") or []
    upstream: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("success"):
            continue
        if item.get("agent_name") == "explainer":
            continue
        result = item.get("result")
        if isinstance(result, dict) and result:
            enriched = dict(result)
            if isinstance(item.get("agent_name"), str):
                enriched.setdefault("agent_name", item["agent_name"])
            upstream.append(enriched)
    return upstream


def _resolve_explainer_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Bind ``explain``'s required ``analysis_results`` to REAL upstream agent
    results, or fail closed (#883 §3).

    explainer is BOTH the 'explanation'-intent primary AND the universal
    fallback agent (router.py), so before this resolver EVERY direct dispatch
    and every agent's failure-fallback crashed identically on the generic
    payload splat (``explain() got an unexpected keyword argument
    'user_context'``).

    (1) An explicit analyst-supplied ``parameters.analysis_results`` wins.
    (2) Otherwise bind the successful upstream results carried in the dispatch
        state — this turn's earlier/sibling agent outputs on the fallback path,
        prior turns' outputs on a resumed conversation state.
    (3) With neither, fail closed: an explanation of nothing would have to be
        fabricated.
    """
    params = dispatch.get("parameters") or {}

    # (1) explicit analyst-supplied results pass through verbatim.
    explicit = params.get("analysis_results")
    if isinstance(explicit, list) and explicit and all(isinstance(r, dict) for r in explicit):
        analysis_results: List[Dict[str, Any]] = explicit
    else:
        # (2) real upstream results from the orchestrator state.
        analysis_results = _successful_upstream_results(agent_input)

    if not analysis_results:
        return NeedsStructuredInput(
            agent_name="explainer",
            missing=("analysis_results",),
            reason=(
                "no successful upstream agent results exist in this conversation "
                "state to explain, and dispatch.parameters supplies none — run an "
                "analysis first (e.g. a causal, gap or segmentation query), then "
                "ask for the explanation"
            ),
            rest_endpoint=None,
        )

    out: Dict[str, Any] = {
        "analysis_results": analysis_results,
        "query": agent_input.get("query") or "",
    }
    session_id = agent_input.get("session_id")
    if session_id is not None:
        out["session_id"] = session_id
    for opt in ("user_expertise", "output_format", "focus_areas", "memory_config"):
        if params.get(opt) is not None:
            out[opt] = params[opt]
    # Real brand/region from the parsed query feed the skill loader (NOT a
    # fabricated default: absent entities simply omit the config).
    if "memory_config" not in out:
        brand, region = _extract_brand_region(agent_input)
        memory_config = {k: v for k, v in (("brand", brand), ("region", region)) if v}
        if memory_config:
            out["memory_config"] = memory_config
    return out


# check_health's scope Literal (agent.py). Every kwarg has a default, so this
# resolver NEVER fails closed — it only maps the generic payload onto the clean
# kwarg set (verified against the #881 signature: scope/query/experiment_name/
# session_id).
_HEALTH_SCOPES = ("full", "quick", "models", "pipelines", "agents")
_HEALTH_SCOPE_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    (r"\bmodels?\b", "models"),
    (r"\bpipelines?\b", "pipelines"),
    (r"\bagents?\b", "agents"),
    (r"\bquick\b", "quick"),
)


def _derive_health_scope(query: str) -> str:
    """Derive ``check_health``'s scope from the user's query, default ``full``.

    A single unambiguous subsystem mention scopes the check to that subsystem;
    zero or multiple mentions run the FULL check (which covers every subsystem
    — the conservative, never-narrower default).
    """
    lowered = (query or "").lower()
    matched = {scope for pattern, scope in _HEALTH_SCOPE_KEYWORDS if re.search(pattern, lowered)}
    if len(matched) == 1:
        return matched.pop()
    return "full"


def _resolve_health_score_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Dict[str, Any]:
    """Map the generic payload onto ``check_health``'s clean kwarg set (#883 §3).

    'system_health' is a sole-agent intent with no fallback, so the generic
    payload splat (``check_health() got an unexpected keyword argument
    'user_context'``) made the intent 100% dead via chat. Every ``check_health``
    kwarg has a default, so nothing here can be ungroundable — no fail-closed
    branch. ``session_id`` is passed through for the #881 memory wiring.
    """
    params = dispatch.get("parameters") or {}
    query = agent_input.get("query") or ""

    scope = params.get("scope")
    if not (isinstance(scope, str) and scope in _HEALTH_SCOPES):
        scope = _derive_health_scope(query)

    out: Dict[str, Any] = {"scope": scope, "query": query}
    experiment_name = params.get("experiment_name")
    if isinstance(experiment_name, str) and experiment_name.strip():
        out["experiment_name"] = experiment_name
    session_id = agent_input.get("session_id")
    if session_id is not None:
        out["session_id"] = session_id
    return out


# feedback_learner window grounding (#883 §3). The default trailing window
# MIRRORS the 6h Celery beat path (src/tasks/dspy_optimization_tasks.py
# ``run_feedback_learning_cycle``): end = now UTC, start = end - DSPY_LEARN_WINDOW_HOURS
# (default 24) — the chat path and the beat path must read the same window
# contract or the two learning surfaces drift.
_FL_WINDOW_ENV = "DSPY_LEARN_WINDOW_HOURS"
_FL_DEFAULT_WINDOW_HOURS = 24.0
_FL_REQUIRED = ("time_range_start", "time_range_end")

_FL_RELATIVE_RE = re.compile(
    r"\b(?:last|past|previous)\s+(?:(\d+)\s+)?(day|week|month|quarter|year)s?\b"
)
_FL_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_FL_YEAR_RE = re.compile(r"\b(20[0-9]{2})\b")
_FL_RELATIVE_DAYS = {"day": 1, "week": 7, "month": 30, "quarter": 91, "year": 365}


def _parse_time_period_text(text: str, now: datetime) -> Optional[Tuple[datetime, datetime]]:
    """Parse a ``time_period`` entity text into a concrete UTC window, or ``None``.

    Grounded in the same shapes the NLP layer emits (classifier
    ``feature_extractor`` patterns ``Q[1-4]`` / ``20xx`` plus common relative
    phrases). Returns ``None`` when the text matches none of them — the caller
    FAILS CLOSED rather than silently substituting a different window than the
    one the user named.
    """
    relative = _FL_RELATIVE_RE.search(text.lower())
    if relative:
        count = int(relative.group(1) or 1)
        days = count * _FL_RELATIVE_DAYS[relative.group(2)]
        return now - timedelta(days=days), now

    quarter = _FL_QUARTER_RE.search(text)
    year_match = _FL_YEAR_RE.search(text)
    if quarter:
        year = int(year_match.group(1)) if year_match else now.year
        q = int(quarter.group(1))
        start = datetime(year, 3 * (q - 1) + 1, 1, tzinfo=timezone.utc)
        end = (
            datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            if q == 4
            else datetime(year, 3 * q + 1, 1, tzinfo=timezone.utc)
        )
        return start, end
    if year_match:
        year = int(year_match.group(1))
        return (
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year + 1, 1, 1, tzinfo=timezone.utc),
        )
    return None


def _resolve_feedback_learner_input(
    agent_input: Dict[str, Any], dispatch: AgentDispatch
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Ground ``learn``'s required ``(time_range_start, time_range_end)`` window
    in real dispatch context, or fail closed (#883 §3).

    'feedback' is a sole-agent intent with no fallback; the generic payload splat
    (``learn() got an unexpected keyword argument 'query'``) made it 100% dead
    via chat (#839 leftover — the 6h Celery beat path calls ``learn()`` correctly
    and is untouched here).

    (1) An explicit analyst-supplied window in ``dispatch.parameters`` wins, but
        must actually BE a window (both bounds, ISO-parseable, start < end) —
        a half/garbled window is ungroundable, so it fails closed rather than
        being silently replaced.
    (2) Otherwise derive the window from the parsed_query ``time_period``
        entities. An entity that names a period we cannot parse fails CLOSED:
        learning over a different window than the user asked for would
        misrepresent the result.
    (3) With no temporal entity at all, use the default trailing window the 6h
        Celery beat already learns on (``DSPY_LEARN_WINDOW_HOURS``, default 24h,
        ending now UTC) — the settled production window contract.
    """
    params = dispatch.get("parameters") or {}
    now = datetime.now(timezone.utc)

    explicit_start = params.get("time_range_start")
    explicit_end = params.get("time_range_end")
    if explicit_start is not None or explicit_end is not None:
        # (1) explicit analyst-supplied window — validate, never repair.
        try:
            start_dt = datetime.fromisoformat(str(explicit_start))
            end_dt = datetime.fromisoformat(str(explicit_end))
            valid = start_dt < end_dt
        except (TypeError, ValueError):
            valid = False
        if not valid:
            return NeedsStructuredInput(
                agent_name="feedback_learner",
                missing=_FL_REQUIRED,
                reason=(
                    "dispatch.parameters supplied an explicit feedback window that is "
                    f"not a valid ISO interval (time_range_start={explicit_start!r}, "
                    f"time_range_end={explicit_end!r}; both bounds required, start < end)"
                ),
                rest_endpoint="POST /feedback/learn",
            )
        window = (str(explicit_start), str(explicit_end))
        window_source = "parameters"
    else:
        parsed_query = agent_input.get("parsed_query") or {}
        entities = (parsed_query.get("entities") if isinstance(parsed_query, dict) else None) or []
        period_texts = [
            str(ent["value"])
            for ent in entities
            if isinstance(ent, dict) and ent.get("type") == "time_period" and ent.get("value")
        ]
        if period_texts:
            # (2) the user NAMED a period — bind it or fail closed. Values are
            # joined so split entities ("Q3" + "2025") resolve as one period.
            parsed = _parse_time_period_text(" ".join(period_texts), now)
            if parsed is None:
                return NeedsStructuredInput(
                    agent_name="feedback_learner",
                    missing=_FL_REQUIRED,
                    reason=(
                        f"the query names a time period ({', '.join(period_texts)!s}) "
                        "that could not be parsed into a concrete window; learning "
                        "over a substituted window would misrepresent the result"
                    ),
                    rest_endpoint="POST /feedback/learn",
                )
            window = (parsed[0].isoformat(), parsed[1].isoformat())
            window_source = "parsed_query.time_period"
        else:
            # (3) default trailing window — mirrors the 6h Celery beat path.
            try:
                hours = float(os.getenv(_FL_WINDOW_ENV, "") or _FL_DEFAULT_WINDOW_HOURS)
            except ValueError:
                hours = _FL_DEFAULT_WINDOW_HOURS
            window = ((now - timedelta(hours=hours)).isoformat(), now.isoformat())
            window_source = f"trailing_{hours:g}h_beat_default"

    logger.info(
        "feedback_learner dispatch: bound learning window %s → %s (source=%s).",
        window[0],
        window[1],
        window_source,
    )
    out: Dict[str, Any] = {"time_range_start": window[0], "time_range_end": window[1]}
    for opt in ("batch_id", "focus_agents"):
        if params.get(opt) is not None:
            out[opt] = params[opt]
    return out


# Single source of truth: agent_name -> input resolver. Add a resolver here, not
# an ``if`` branch in ``_dispatch_agent`` (#F12/F13/F14).
INPUT_RESOLVERS: Dict[str, InputResolver] = {
    "tool_composer": _resolve_tool_composer_input,
    "gap_analyzer": _resolve_gap_analyzer_input,
    "heterogeneous_optimizer": _resolve_heterogeneous_optimizer_input,
    "resource_optimizer": _resolve_resource_optimizer_input,
    "prediction_synthesizer": _resolve_prediction_synthesizer_input,
    # #883 §3 — the last three ``uses_kwargs`` agents with neither a resolver
    # nor an input_model (the generic-payload splat raised TypeError on every
    # chat dispatch): explainer (the 'explanation' primary AND universal
    # fallback agent), health_score ('system_health', sole agent), and
    # feedback_learner ('feedback', sole agent).
    "explainer": _resolve_explainer_input,
    "health_score": _resolve_health_score_input,
    "feedback_learner": _resolve_feedback_learner_input,
}


# Resolver-backed agents that must FAIL CLOSED when the agent's OWN output reports
# an internal failure (``status == "failed"``). A domain failure (e.g. no models
# registered, infeasible optimization) must never be reported as a successful
# dispatch — otherwise the dispatcher's transport-level ``success=True`` would
# launder an empty/failed analysis into a "success" (#F12/F13/F14). tool_composer
# is intentionally EXCLUDED: its success semantics are governed by F6 (#827)
# tool-level fail-closed + the synthesizer's filtering, not a status field.
#
# #883 §3 — the three newly resolver-backed agents are included after reading
# their output status semantics:
#   * explainer: ``ExplainerOutput.status = result.get("status", "failed")`` —
#     "failed" iff the explanation graph itself errored; an empty-but-honest
#     explanation completes. Laundering a failed explanation as dispatch
#     success would surface a blank narrative as a real one.
#   * health_score: status is "failed" ONLY on the agent's exception path,
#     where the output carries fabricated-looking placeholders (score 0.0,
#     grade "F") — a degraded-but-measured system still reports "completed",
#     so failing closed never hides a real unhealthy reading.
#   * feedback_learner: graph nodes set "failed" only on internal errors; a
#     window with ZERO feedback rows completes honestly ("analyzing" →
#     "completed", warnings=['No feedback items collected']) and is NOT failed
#     closed — the honest no-data outcome must reach the user.
_FAIL_CLOSED_ON_FAILED_STATUS = frozenset(
    {
        "gap_analyzer",
        "heterogeneous_optimizer",
        "resource_optimizer",
        "prediction_synthesizer",
        "explainer",
        "health_score",
        "feedback_learner",
    }
)


def _agent_failed(agent_name: str, result: Dict[str, Any]) -> Optional[str]:
    """Return a failure detail string if ``result`` from ``agent_name`` reports an
    internal domain failure that must fail the dispatch closed, else ``None``.

    Only applies to the resolver-backed fail-closed agents and only on an explicit
    ``status == "failed"`` (the contract all four set on their failure paths,
    e.g. heterogeneous_optimizer agent.py: ``"failed" if errors else "completed"``;
    gap_analyzer agent.py ``_build_error_output`` / gap_detector's error path, #874).
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

        # #883 §3: dispatches must see the results accumulated SO FAR — the
        # incoming state's prior-turn results (operator.add reducer on a
        # checkpointer-resumed conversation) plus this turn's earlier groups —
        # so the explainer resolver (fallback agent + later-group dispatches)
        # can bind REAL upstream analysis_results instead of crashing or
        # fabricating. The enriched dict is dispatch-local: the node still
        # returns only ``all_results`` for the reducer to append.
        prior_results: List[AgentResult] = list(state.get("agent_results") or [])

        def _state_so_far() -> OrchestratorState:
            return cast(
                OrchestratorState,
                {**state, "agent_results": prior_results + all_results},
            )

        # Execute each parallel group sequentially
        for group in parallel_groups:
            group_dispatches = [d for d in dispatch_plan if d["agent_name"] in group]

            # Run agents in parallel within group
            group_state = _state_so_far()
            tasks = [self._dispatch_agent(d, group_state) for d in group_dispatches]

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
                        fallback_result = await self._dispatch_fallback(
                            str(fallback_agent), _state_so_far()
                        )
                        all_results.append(fallback_result)
                elif isinstance(result, dict) and not result.get("success", True):
                    # AgentResult returned with success=False
                    all_results.append(result)  # type: ignore[arg-type]

                    # Try fallback if available
                    fallback_agent2 = dispatch.get("fallback_agent")
                    if fallback_agent2:
                        fallback_result = await self._dispatch_fallback(
                            str(fallback_agent2), _state_so_far()
                        )
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
            # #883 §3: upstream agent results — the REAL substrate the explainer
            # resolver binds ``analysis_results`` from. ``execute()`` threads in
            # the results accumulated this turn for later groups and fallback
            # dispatches; a checkpointer-resumed state carries prior turns'.
            # Like the other generic keys, resolver-backed kwargs agents never
            # see it (the resolver output REPLACES the payload) and run(dict)/
            # input-model agents ignore undeclared keys.
            "agent_results": list(state.get("agent_results") or []),
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
