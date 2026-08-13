"""
E2I Tool Composer - Main Orchestrator
Version: 4.2
Purpose: Orchestrate the 4-phase tool composition pipeline

The Tool Composer enables dynamic composition of analytical tools to answer
complex, multi-faceted queries that span multiple agent capabilities.

Pipeline:
    Phase 1: DECOMPOSE - Break query into atomic sub-questions
    Phase 2: PLAN     - Map sub-questions to tools, create execution plan
    Phase 3: EXECUTE  - Run tools in dependency order
    Phase 4: SYNTHESIZE - Combine results into coherent response

Observability:
- Audit chain recording for tamper-evident logging
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

# Import tool modules to ensure tools are registered in the global registry
# These imports trigger the auto-registration decorators
import src.tool_registry.tools  # noqa: F401
from src.agents.base.audit_chain_mixin import get_audit_chain_service
from src.tool_registry.registry import ToolRegistry, get_registry
from src.tool_registry.tools.causal_discovery import register_all_discovery_tools
from src.tool_registry.tools.model_inference import register_model_inference_tool
from src.tool_registry.tools.structural_drift import register_structural_drift_tool
from src.utils.audit_chain import AgentTier
from src.utils.redaction import redact_query

from .decomposer import DecompositionError, QueryDecomposer
from .executor import ExecutionError, PlanExecutor
from .memory_hooks import (
    ToolComposerMemoryHooks,
    contribute_to_memory,
    get_tool_composer_memory_hooks,
)
from .models.composition_models import (
    CompositionPhase,
    CompositionResult,
    CompositionStatus,
    SynthesisInput,
)
from .planner import PlanningError, ToolPlanner
from .synthesizer import ResponseSynthesizer

logger = logging.getLogger(__name__)

# Cap for tool-authored failure reasons carried into the F6 total-failure
# envelope (#1574). The reasons are what let a fail-closed tool state the scope
# it actually covered, but they are of unknown length, and this envelope is
# returned to the caller verbatim (no synthesis pass to compress it).
_MAX_FAILURE_REASON_CHARS = 2000
_TRUNCATION_MARKER = "…(truncated)"


def _truncate(text: str, limit: int = _MAX_FAILURE_REASON_CHARS) -> str:
    """Bound ``text`` to ``limit`` characters, marking any elision."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


# Per-phase LLM client sizing (#1365). Prod runs claude-sonnet-5 (adaptive
# thinking), whose thinking tokens count against max_tokens — at the old shared
# 2048 default an eager thinking pass truncated the planning JSON mid-first-value
# and the whole composition failed (0 tools). Measured against the real API
# (2026-07-30): the natural (thinking + JSON) planning generation reaches ~2058
# output tokens, and with thinking DISABLED it collapses to a deterministic
# ~900-1000. So:
#   - plan: disable thinking ("none") — the planning call emits structured
#     tool-mapping JSON (its reasoning is a JSON FIELD); hidden thinking is
#     wasted AND is what overran the budget. This makes truncation impossible.
#   - decompose / synthesize: keep adaptive thinking (they benefit from
#     reasoning) but raise the budget to 4096 for comfortable headroom over the
#     observed ~2058 — a strict improvement over the old 2048/2000 caps.
# These are DEFAULTS: an explicit per-phase config (config["phases"][<phase>])
# overrides them, honoring the per-phase knobs the LangChain migration had left
# severed from the actual call.
_PHASE_LLM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "decompose": {"max_tokens": 4096, "reasoning_effort": None},
    "plan": {"max_tokens": 4096, "reasoning_effort": "none"},
    "synthesize": {"max_tokens": 4096, "reasoning_effort": None},
}


def _ensure_tools_registered():
    """Ensure composable tools are registered in the registry."""
    try:
        register_all_discovery_tools()
        register_model_inference_tool()
        register_structural_drift_tool()
        logger.info("Tool registry initialized with composable tools")
    except Exception as e:
        logger.warning(f"Tool registration failed: {e}")


# Register tools when module loads
_ensure_tools_registered()


# ============================================================================
# TOOL COMPOSER CLASS
# ============================================================================


class ToolComposer:
    """
    Orchestrates the 4-phase tool composition pipeline.

    The Tool Composer is invoked by the Orchestrator when a query is
    classified as MULTI_FACETED - requiring capabilities from multiple
    agents combined in novel ways.

    Usage:
        composer = ToolComposer(llm_client=anthropic_client)
        result = await composer.compose(
            query="Compare causal impact of X vs Y and predict outcome Z"
        )
        print(result.response.answer)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
        memory_hooks: Optional[ToolComposerMemoryHooks] = None,
        enable_memory_contribution: bool = True,
    ):
        """
        Initialize the Tool Composer.

        Args:
            llm_client: Anthropic/OpenAI-compatible LLM client shared across
                phases (dependency-injection mode — used by tests). When
                ``None`` the composer builds a correctly-SIZED client PER PHASE
                from the factory (#1365): the planning phase gets a real token
                budget with thinking disabled, so the planning JSON can no
                longer truncate. Both production entry points (the orchestrator
                agent and the chatbot_tools chat path, #1557) pass ``None``.
            tool_registry: Optional custom tool registry (uses global if not provided)
            config: Optional configuration overrides
            memory_hooks: Optional memory hooks instance (G1, G2 integration)
            enable_memory_contribution: Whether to store compositions in memory
        """
        self.llm_client = llm_client
        self.registry = tool_registry or get_registry()
        self.config = config or {}
        self.memory_hooks = memory_hooks or get_tool_composer_memory_hooks()
        self.enable_memory_contribution = enable_memory_contribution

        # Initialize phase handlers
        self._init_phase_handlers()

        logger.info(f"ToolComposer initialized with {self.registry.tool_count} tools")

    def _init_phase_handlers(self) -> None:
        """Initialize handlers for each phase"""
        decompose_config = self.config.get("phases", {}).get("decompose", {})
        plan_config = self.config.get("phases", {}).get("plan", {})
        execute_config = self.config.get("phases", {}).get("execute", {})
        synthesize_config = self.config.get("phases", {}).get("synthesize", {})

        self.decomposer = QueryDecomposer(
            llm_client=self._resolve_phase_client("decompose", decompose_config),
            model=decompose_config.get("model", "claude-sonnet-4-6"),
            temperature=decompose_config.get("temperature", 0.3),
            max_sub_questions=decompose_config.get("max_sub_questions", 6),
            min_sub_questions=decompose_config.get("min_sub_questions", 2),
        )

        self.planner = ToolPlanner(
            llm_client=self._resolve_phase_client("plan", plan_config),
            tool_registry=self.registry,
            model=plan_config.get("model", "claude-sonnet-4-6"),
            temperature=plan_config.get("temperature", 0.2),
            max_tools_per_plan=plan_config.get("max_tools_per_plan", 8),
            memory_hooks=self.memory_hooks,
            use_episodic_memory=plan_config.get("use_episodic_memory", True),
        )

        self.executor = PlanExecutor(
            tool_registry=self.registry,
            max_parallel=execute_config.get("parallel_execution_limit", 3),
            max_retries=execute_config.get("max_retries", 2),
            timeout_seconds=execute_config.get("max_execution_time_seconds", 120),
        )

        self.synthesizer = ResponseSynthesizer(
            llm_client=self._resolve_phase_client("synthesize", synthesize_config),
            model=synthesize_config.get("model", "claude-sonnet-4-6"),
            temperature=synthesize_config.get("temperature", 0.4),
            max_tokens=synthesize_config.get("max_tokens", 2000),
        )

    def _resolve_phase_client(self, phase: str, phase_config: Dict[str, Any]) -> Any:
        """Return the LLM client for a phase (#1365).

        Two modes:

        - **Dependency-injection.** When a client was injected
          (``self.llm_client`` is not ``None``) — tests — every phase SHARES it
          unchanged. This preserves the mock-injection contract the whole test
          suite relies on. Production entry points must NOT inject (#1557): a
          shared client puts the plan phase back on adaptive thinking, whose
          tokens eat the budget and truncate the planner JSON.
        - **Factory.** When no client was injected, build a client sized for
          THIS phase from the factory (``get_chat_llm``), which encapsulates
          provider switching (anthropic vs openai) and the sonnet-5 thinking
          semantics. Defaults come from ``_PHASE_LLM_DEFAULTS``; an explicit
          per-phase ``config["phases"][phase]`` (``max_tokens`` /
          ``reasoning_effort``) overrides them — honoring the per-phase knobs
          the LangChain migration (55a7f749) had severed from the actual call.

        Note: the per-phase ``model`` config string is DELIBERATELY not passed
        through here. Model selection is the factory's tier authority
        (``model_tier`` -> ``MODEL_MAPPINGS``, all ids verified callable). The
        phase-config ``model`` still defaults to the pre-migration
        ``claude-sonnet-4-6``; forwarding that literal would bypass the curated
        mapping and resurrect the retired dead-model-id 404 class that commit
        b144b6a5 fixed. So the budget/effort knobs are honored; model is not.
        """
        if self.llm_client is not None:
            return self.llm_client

        from src.utils.llm_factory import get_chat_llm

        defaults = _PHASE_LLM_DEFAULTS[phase]
        return get_chat_llm(
            model_tier="standard",
            max_tokens=phase_config.get("max_tokens", defaults["max_tokens"]),
            reasoning_effort=phase_config.get("reasoning_effort", defaults["reasoning_effort"]),
        )

    async def compose(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CompositionResult:
        """
        Execute the full composition pipeline.

        Args:
            query: The user's multi-faceted query
            context: Optional context (filters, data references, etc.)

        Returns:
            CompositionResult with the synthesized response and full trace
        """
        started_at = datetime.now(timezone.utc)
        phase_durations: Dict[str, int] = {}
        context = context or {}

        logger.info(f"Starting composition for query: {redact_query(query, max_len=100)}")

        # S14 (Phase 7 prerequisite): extract experiment_id from the
        # existing context-dict carrier. Phase 7.2's auto-population
        # hook in executor.py reads ``context["experiment_id"]`` to
        # query active role attributions. We log it here for audit
        # provenance (downstream queries can join compositions to
        # experiments) while keeping ``start_workflow``'s kwarg surface
        # unchanged.
        experiment_id = context.get("experiment_id")

        # Initialize audit chain workflow
        audit_workflow_id: Optional[UUID] = None
        audit_service = get_audit_chain_service()
        if audit_service:
            try:
                audit_input_data: Dict[str, Any] = {"query": query[:500]}
                if isinstance(experiment_id, str) and experiment_id:
                    audit_input_data["experiment_id"] = experiment_id
                entry = audit_service.start_workflow(
                    agent_name="tool_composer",
                    agent_tier=AgentTier.COORDINATION,
                    action_type="workflow_start",
                    input_data=audit_input_data,  # Truncate for storage
                    user_id=context.get("user_id"),
                    session_id=context.get("session_id"),
                    query_text=query,
                    brand=context.get("brand"),
                )
                audit_workflow_id = entry.workflow_id
                context["audit_workflow_id"] = audit_workflow_id
                logger.debug(f"Started audit workflow {audit_workflow_id} for tool_composer")
            except Exception as e:
                logger.warning(f"Failed to start audit workflow: {e}")

        try:
            # ================================================================
            # PHASE 1: DECOMPOSE
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 1: Decomposing query...")

            decomposition = await self.decomposer.decompose(query)

            phase_durations["decompose"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 1 complete: {decomposition.question_count} sub-questions "
                f"({phase_durations['decompose']}ms)"
            )

            # Record audit entry for decompose phase
            self._record_audit_entry(
                audit_service,
                audit_workflow_id,
                "decompose",
                phase_durations["decompose"],
                {"sub_questions_count": decomposition.question_count},
            )

            # ================================================================
            # PHASE 2: PLAN
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 2: Creating execution plan...")

            # F6(b): hand the planner a RICH column profile (dtype family,
            # cardinality, low-card value lists) — not just names — so the LLM
            # binds column-typed args to real columns AND avoids degenerate /
            # near-constant targets. available_columns is kept for back-compat.
            column_profiles = self._extract_column_profiles(context)
            available_columns = self._extract_available_columns(context)
            # #810: when the caller resolved a defined-KPI substrate, the causal
            # outcome column is carried in context["kpi_outcome"]; bind it so the
            # analysis measures the KPI rather than an LLM-guessed column.
            outcome_hint = (context or {}).get("kpi_outcome")
            # #810: for a defined-KPI causal question, build a DETERMINISTIC causal
            # analysis plan (causal_effect_estimator + cate_analyzer + driver
            # ranking) over the KPI substrate. Free-form LLM planning is unreliable
            # for this (it picks descriptive tools / categorical treatments / passes
            # string cohort args that cascade-fail), so we bind the proven causal
            # plan directly. Falls back to LLM planning when no usable treatment.
            plan = None
            if outcome_hint:
                plan = self._build_kpi_causal_plan(decomposition, context, outcome_hint)
            if plan is None:
                plan = await self.planner.plan(
                    decomposition,
                    available_columns=available_columns,
                    column_profiles=column_profiles,
                    outcome_hint=outcome_hint,
                )

            phase_durations["plan"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 2 complete: {plan.step_count} steps planned ({phase_durations['plan']}ms)"
            )

            # Record audit entry for plan phase
            self._record_audit_entry(
                audit_service,
                audit_workflow_id,
                "plan",
                phase_durations["plan"],
                {"steps_planned": plan.step_count},
            )

            # ================================================================
            # PHASE 3: EXECUTE
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 3: Executing tool chain...")

            execution_trace = await self.executor.execute(plan, context)

            phase_durations["execute"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 3 complete: {execution_trace.tools_succeeded}/"
                f"{execution_trace.tools_executed} tools succeeded "
                f"({phase_durations['execute']}ms)"
            )

            # Record audit entry for execute phase
            self._record_audit_entry(
                audit_service,
                audit_workflow_id,
                "execute",
                phase_durations["execute"],
                {
                    "tools_executed": execution_trace.tools_executed,
                    "tools_succeeded": execution_trace.tools_succeeded,
                },
                validation_passed=execution_trace.tools_succeeded == execution_trace.tools_executed,
            )

            # ================================================================
            # F6 FAIL-CLOSED GATE: total tool failure (0/N succeeded)
            # ================================================================
            # If EVERY executed tool failed, the composition has no successful
            # tool output to synthesize. Do NOT invoke the LLM to fabricate a
            # confident answer over nothing — fail CLOSED with an honest FAILED
            # result. (A PARTIAL success — at least one tool — still synthesizes
            # below over the results that DID succeed.)
            if execution_trace.tools_executed > 0 and execution_trace.tools_succeeded == 0:
                logger.warning(
                    "All %d executed tool(s) failed; failing closed without synthesis "
                    "(no fabricated response).",
                    execution_trace.tools_executed,
                )
                return self._create_total_failure_result(
                    query,
                    decomposition,
                    plan,
                    execution_trace,
                    started_at,
                    phase_durations,
                )

            # ================================================================
            # PHASE 4: SYNTHESIZE
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 4: Synthesizing response...")

            synthesis_input = SynthesisInput(
                original_query=query, decomposition=decomposition, execution_trace=execution_trace
            )

            response = await self.synthesizer.synthesize(synthesis_input)

            phase_durations["synthesize"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 4 complete: confidence={response.confidence} "
                f"({phase_durations['synthesize']}ms)"
            )

            # Record audit entry for synthesize phase
            self._record_audit_entry(
                audit_service,
                audit_workflow_id,
                "synthesize",
                phase_durations["synthesize"],
                {"confidence": response.confidence},
                confidence_score=response.confidence,
            )

            # ================================================================
            # BUILD RESULT
            # ================================================================
            completed_at = datetime.now(timezone.utc)
            total_duration = int((completed_at - started_at).total_seconds() * 1000)

            # Determine contract-compliant status based on execution results
            # Note: If we reach synthesis, the composition completed, so minimum is PARTIAL
            if execution_trace.tools_succeeded == execution_trace.tools_executed:
                status = CompositionStatus.SUCCESS
            else:
                # PARTIAL means composition completed but not all tools succeeded
                # This includes the case where no tools succeeded but synthesis ran
                status = CompositionStatus.PARTIAL

            result = CompositionResult(
                query=query,
                decomposition=decomposition,
                plan=plan,
                execution=execution_trace,
                response=response,
                total_duration_ms=total_duration,
                phase_durations=phase_durations,
                # Contract-compliant status
                status=status,
                # SUCCESS or PARTIAL both count as "success" for quality gate
                success=status in (CompositionStatus.SUCCESS, CompositionStatus.PARTIAL),
                errors=[],  # No errors on successful path
                started_at=started_at,
                completed_at=completed_at,
            )

            logger.info(f"Composition complete in {total_duration}ms")

            # ================================================================
            # MEMORY CONTRIBUTION (G1, G2)
            # ================================================================
            if self.enable_memory_contribution:
                await self._contribute_to_memory(result, context)

            return result

        except DecompositionError as e:
            return self._create_error_result(
                query,
                started_at,
                phase_durations,
                f"Decomposition failed: {e}",
                CompositionPhase.DECOMPOSE,
            )

        except PlanningError as e:
            return self._create_error_result(
                query, started_at, phase_durations, f"Planning failed: {e}", CompositionPhase.PLAN
            )

        except ExecutionError as e:
            return self._create_error_result(
                query,
                started_at,
                phase_durations,
                f"Execution failed: {e}",
                CompositionPhase.EXECUTE,
            )

        except Exception as e:
            logger.exception(f"Unexpected error during composition: {e}")
            return self._create_error_result(
                query, started_at, phase_durations, f"Unexpected error: {e}", None
            )

    def _elapsed_ms(self, start: datetime) -> int:
        """Calculate elapsed milliseconds since start"""
        return int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

    def _extract_available_columns(self, context: Dict[str, Any]) -> Optional[List[str]]:
        """Derive real dataset column names from the in-context DataFrame (F2).

        The canonical carrier is ``context["estimation_data"]``. We only return
        column names when it is an actual pandas DataFrame, so a non-frame value
        under the key (or a missing key) cleanly yields ``None`` and the planner
        falls back to schema-free planning.
        """
        import pandas as pd

        frame = context.get("estimation_data")
        if isinstance(frame, pd.DataFrame):
            return [str(col) for col in frame.columns]
        return None

    def _extract_column_profiles(self, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Build a per-column PROFILE from the in-context DataFrame (F6(b)).

        For each column the profile carries:
        - ``name``: the column name
        - ``dtype_family``: one of ``binary`` / ``numeric-continuous`` /
          ``categorical`` / ``other`` — the semantic family the planner uses to
          pick a treatment (binary/low-card), an outcome (numeric), or a
          segment (low-card categorical)
        - ``n_unique``: distinct non-null value count (cardinality)
        - ``n_nonnull``: non-null cell count
        - ``values``: the actual distinct values for LOW-cardinality columns
          (<= ``_LOW_CARD_MAX``), else ``None`` — so the LLM avoids
          near-constant / degenerate targets

        Robustness: real cohort frames contain list-valued (unhashable) columns
        (e.g. ``comorbidities``). ``Series.nunique()`` raises
        ``TypeError: unhashable type: 'list'`` on those, so cardinality and the
        value list fall back to a string-cast path rather than crashing.

        Returns ``None`` when no DataFrame is present (schema-free planning).
        """
        import pandas as pd

        frame = context.get("estimation_data")
        if not isinstance(frame, pd.DataFrame):
            return None

        _LOW_CARD_MAX = 12
        profiles: List[Dict[str, Any]] = []

        for col in frame.columns:
            series = frame[col]
            n_nonnull = int(series.notna().sum())

            # Cardinality + distinct values, robust to unhashable cells.
            unhashable = False
            try:
                n_unique = int(series.nunique(dropna=True))
                distinct = series.dropna().unique().tolist()
            except TypeError:
                # list/dict-valued cells -> count distinct string reprs.
                unhashable = True
                as_str = series.dropna().astype(str)
                n_unique = int(as_str.nunique())
                distinct = as_str.unique().tolist()

            dtype_family = self._classify_dtype_family(series, n_unique, unhashable)

            values: Optional[List[Any]] = None
            if not unhashable and 0 < n_unique <= _LOW_CARD_MAX:
                # JSON-safe scalars for the prompt (numpy -> python).
                values = [v.item() if hasattr(v, "item") else v for v in distinct]
            elif unhashable and 0 < n_unique <= _LOW_CARD_MAX:
                values = [str(v) for v in distinct]

            profiles.append(
                {
                    "name": str(col),
                    "dtype_family": dtype_family,
                    "n_unique": n_unique,
                    "n_nonnull": n_nonnull,
                    "values": values,
                }
            )

        return profiles

    @staticmethod
    def _classify_dtype_family(series: Any, n_unique: int, unhashable: bool) -> str:
        """Classify a column into a planner-facing dtype family.

        - ``binary``: numeric/bool with exactly 2 distinct non-null values
        - ``numeric-continuous``: numeric dtype with > 2 distinct values
        - ``categorical``: object/category/bool (or string-like) low-ish card
        - ``other``: unhashable (list/dict-valued) or otherwise unclassifiable
        """
        import pandas as pd

        if unhashable:
            return "other"

        if pd.api.types.is_bool_dtype(series):
            return "binary" if n_unique <= 2 else "categorical"

        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= 2:
                return "binary"
            return "numeric-continuous"

        # object / category / string -> categorical
        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        ):
            return "categorical"

        return "other"

    def _build_kpi_causal_plan(
        self,
        decomposition: Any,
        context: Optional[Dict[str, Any]],
        outcome: str,
    ) -> Optional[Any]:
        """Build a DETERMINISTIC causal-analysis plan over a KPI substrate (#810).

        For a defined-KPI causal question (``context["kpi_outcome"]`` set) the
        analysis is well-posed: estimate the causal effect of a driver on the KPI
        outcome (``causal_effect_estimator``), rank drivers by importance
        (``discover_dag`` -> ``rank_drivers``), and measure segment heterogeneity
        (``cate_analyzer``) — exactly "what drove <KPI>, and which segments respond
        best". The treatment is the best BINARY (then numeric) driver; confounders
        are the other numeric drivers; segments are the categorical drivers. All
        bind to REAL columns of the KPI frame (no fabrication).

        Returns ``None`` (caller falls back to LLM planning) when there is no KPI
        frame or no usable treatment driver.
        """
        from .models.composition_models import ExecutionPlan, ExecutionStep, ToolMapping

        profiles = self._extract_column_profiles(context or {})
        if not profiles:
            return None
        prof = {str(p["name"]): p for p in profiles if p.get("name")}

        def _family(name: str) -> Any:
            return prof[name].get("dtype_family")

        def _card(name: str) -> int:
            try:
                return int(prof[name].get("n_unique") or 0)
            except (TypeError, ValueError):
                return 0

        binary = [n for n in prof if _family(n) == "binary" and n != outcome]
        numeric = [n for n in prof if _family(n) == "numeric-continuous" and n != outcome]
        # Segments: LOW-cardinality categoricals only (2..12 distinct). Excludes
        # high-cardinality ID columns (trigger_id/patient_id/hcp_id) that would
        # produce one-row segments with NaN CATE.
        seg_candidates = [
            n for n in prof if _family(n) == "categorical" and n != outcome and 2 <= _card(n) <= 12
        ]

        treatment_candidates = binary or numeric
        treatment: Optional[str] = treatment_candidates[0] if treatment_candidates else None
        if treatment is None:
            logger.info(
                "KPI causal plan: no usable (binary/numeric) treatment driver for "
                "outcome %r; falling back to LLM planning.",
                outcome,
            )
            return None
        confounders = [c for c in numeric if c != treatment][:5]
        segments = seg_candidates[:2]

        sub_questions = list(getattr(decomposition, "sub_questions", []) or [])
        first_sq = str(sub_questions[0].id) if sub_questions else "sq_1"

        def _sq_for(intents: set[str]) -> str:
            for sq in sub_questions:
                if str(getattr(sq, "intent", "")).upper() in intents:
                    return str(sq.id)
            return first_sq

        causal_sq = _sq_for({"CAUSAL", "DESCRIPTIVE"})
        comparative_sq = _sq_for({"COMPARATIVE"})
        src = "composable"

        steps: List[Any] = []
        mappings: List[Any] = []

        steps.append(
            ExecutionStep(
                step_id="kpi_ate",
                sub_question_id=causal_sq,
                tool_name="causal_effect_estimator",
                source_agent=src,
                input_mapping={
                    "treatment": treatment,
                    "outcome": outcome,
                    "confounders": confounders,
                },
            )
        )
        mappings.append(
            ToolMapping(
                sub_question_id=causal_sq,
                tool_name="causal_effect_estimator",
                source_agent=src,
                confidence=0.9,
                reasoning=f"Estimate the causal effect of {treatment} on the KPI outcome {outcome}.",
            )
        )
        steps.append(
            ExecutionStep(
                step_id="kpi_dag",
                sub_question_id=causal_sq,
                tool_name="discover_dag",
                source_agent=src,
                input_mapping={},
            )
        )
        mappings.append(
            ToolMapping(
                sub_question_id=causal_sq,
                tool_name="discover_dag",
                source_agent=src,
                confidence=0.7,
                reasoning="Discover structure among the KPI drivers.",
            )
        )
        steps.append(
            ExecutionStep(
                step_id="kpi_rank",
                sub_question_id=causal_sq,
                tool_name="rank_drivers",
                source_agent=src,
                input_mapping={"target": outcome, "dag_edge_list": "$kpi_dag.edge_list"},
                depends_on_steps=["kpi_dag"],
            )
        )
        mappings.append(
            ToolMapping(
                sub_question_id=causal_sq,
                tool_name="rank_drivers",
                source_agent=src,
                confidence=0.7,
                reasoning=f"Rank the drivers of the KPI outcome {outcome} by importance.",
            )
        )
        parallel_first = ["kpi_ate", "kpi_dag"]
        if segments:
            steps.append(
                ExecutionStep(
                    step_id="kpi_cate",
                    sub_question_id=comparative_sq,
                    tool_name="cate_analyzer",
                    source_agent=src,
                    input_mapping={
                        "treatment": treatment,
                        "outcome": outcome,
                        "segments": segments,
                    },
                )
            )
            mappings.append(
                ToolMapping(
                    sub_question_id=comparative_sq,
                    tool_name="cate_analyzer",
                    source_agent=src,
                    confidence=0.85,
                    reasoning="Segment-level treatment effects to find the best-responding segments.",
                )
            )
            parallel_first.append("kpi_cate")

        logger.info(
            "KPI causal plan: outcome=%r treatment=%r confounders=%s segments=%s (%d steps).",
            outcome,
            treatment,
            confounders,
            segments,
            len(steps),
        )
        return ExecutionPlan(
            decomposition=decomposition,
            steps=steps,
            tool_mappings=mappings,
            parallel_groups=[parallel_first, ["kpi_rank"]],
            planning_reasoning=(
                f"Deterministic KPI causal analysis for outcome '{outcome}': effect of "
                f"'{treatment}' (confounders={confounders}), driver ranking, and segment "
                f"heterogeneity (segments={segments})."
            ),
        )

    def _record_audit_entry(
        self,
        audit_service: Any,
        workflow_id: Optional[UUID],
        action_type: str,
        duration_ms: int,
        output_data: Dict[str, Any],
        validation_passed: Optional[bool] = None,
        confidence_score: Optional[float] = None,
    ) -> None:
        """Record an audit chain entry for a composition phase.

        Args:
            audit_service: The audit chain service instance
            workflow_id: UUID of the current workflow
            action_type: Type of action (decompose, plan, execute, synthesize)
            duration_ms: Duration of the phase in milliseconds
            output_data: Phase output data to record
            validation_passed: Optional validation status
            confidence_score: Optional confidence score
        """
        if workflow_id and audit_service:
            try:
                audit_service.add_entry(
                    workflow_id=workflow_id,
                    agent_name="tool_composer",
                    agent_tier=AgentTier.COORDINATION,
                    action_type=action_type,
                    duration_ms=duration_ms,
                    output_data=output_data,
                    validation_passed=validation_passed,
                    confidence_score=confidence_score,
                )
            except Exception as e:
                logger.warning(f"Failed to record audit entry for {action_type}: {e}")

    async def _contribute_to_memory(
        self,
        result: CompositionResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store composition results in memory systems (G1, G2).

        Stores in:
        - Working memory (Redis): Cache for quick retrieval
        - Episodic memory (Supabase): Historical record for future lookups
        - Procedural memory: Successful patterns for optimization

        Args:
            result: The composition result to store
            context: Context containing session_id, brand, region, etc.
        """
        try:
            context = context or {}
            session_id = context.get("session_id")
            brand = context.get("brand")
            region = context.get("region")

            # Convert result to dict for memory storage
            result_dict = result.model_dump(mode="json")

            counts = await contribute_to_memory(
                result=result_dict,
                session_id=session_id,
                memory_hooks=self.memory_hooks,
                brand=brand,
                region=region,
            )

            logger.debug(
                f"Memory contribution complete: "
                f"episodic={counts['episodic_stored']}, "
                f"procedural={counts['procedural_stored']}, "
                f"cached={counts['working_cached']}"
            )
        except Exception as e:
            # Don't fail composition if memory contribution fails
            logger.warning(f"Failed to contribute to memory: {e}")

    def _create_error_result(
        self,
        query: str,
        started_at: datetime,
        phase_durations: Dict[str, int],
        error: str,
        failed_phase: Optional[CompositionPhase],
    ) -> CompositionResult:
        """Create an error result when composition fails"""
        from .models.composition_models import (
            ComposedResponse,
            DecompositionResult,
            ExecutionPlan,
            ExecutionTrace,
        )

        # Create minimal placeholder objects
        decomposition = DecompositionResult(
            original_query=query, sub_questions=[], decomposition_reasoning=error
        )

        plan = ExecutionPlan(
            decomposition=decomposition, steps=[], tool_mappings=[], planning_reasoning=error
        )

        execution = ExecutionTrace(plan_id=plan.plan_id)

        response = ComposedResponse(
            answer=f"Unable to complete analysis: {error}",
            confidence=0.0,
            caveats=[error],
            failed_components=[failed_phase.value if failed_phase else "unknown"],
        )

        completed_at = datetime.now(timezone.utc)

        return CompositionResult(
            query=query,
            decomposition=decomposition,
            plan=plan,
            execution=execution,
            response=response,
            total_duration_ms=int((completed_at - started_at).total_seconds() * 1000),
            phase_durations=phase_durations,
            # Contract-compliant status
            status=CompositionStatus.FAILED,
            success=False,
            errors=[error],
            error=error,
            started_at=started_at,
            completed_at=completed_at,
        )

    def _create_total_failure_result(
        self,
        query: str,
        decomposition: Any,
        plan: Any,
        execution_trace: Any,
        started_at: datetime,
        phase_durations: Dict[str, int],
    ) -> CompositionResult:
        """Build a FAILED result when every executed tool failed (F6 fail-closed).

        Unlike ``_create_error_result`` (which uses placeholder phase objects for
        an exception), this preserves the REAL decomposition / plan / execution
        trace and returns an honest, zero-confidence response instead of a
        synthesized answer fabricated over zero successful tool outputs.
        """
        from .models.composition_models import ComposedResponse

        failed_tools: List[str] = []
        reasons: List[str] = []
        for step in getattr(execution_trace, "step_results", None) or []:
            output = getattr(step, "output", None)
            # ``is_success`` (success AND a result present) is the model's own
            # definition of a successful step — it is what ``ExecutionTrace``
            # counts and what ``get_all_outputs`` returns, so the failed-step
            # collector must agree with it or a ``success=True, result=None``
            # step would be counted failed and then listed nowhere.
            if getattr(output, "is_success", False):
                continue
            tool_name = str(
                getattr(step, "tool_name", None) or getattr(output, "tool_name", None) or "unknown"
            )
            failed_tools.append(tool_name)
            reason = str(getattr(output, "error", None) or "").strip()
            if reason:
                reasons.append(f"{tool_name}: {reason}")
        msg = (
            f"All {execution_trace.tools_executed} tool(s) failed; no analysis could "
            "be completed. Returning a failed result rather than a fabricated answer."
        )
        # This answer is NOT synthesized — it is returned verbatim to the caller
        # (``agent.py`` maps it to ``ToolComposerOutput.response``). Carrying the
        # per-step reasons is what keeps a fail-closed tool's honest envelope
        # reaching the user: #1574's ``gap_calculator`` guard states which entity
        # groups the estimation data actually covered, and a one-step plan
        # fail-closes here, before synthesis, so dropping the reason would leave
        # the answer LESS informative than the fabricated comparison it replaced.
        # Bounded because the reason is tool-authored text of unknown length.
        answer = f"Unable to complete analysis: {msg}"
        if reasons:
            answer = _truncate(f"{answer} Reason(s): {' | '.join(reasons)}")
        response = ComposedResponse(
            answer=answer,
            confidence=0.0,
            caveats=[msg, *(_truncate(r) for r in reasons)],
            failed_components=failed_tools or ["execute"],
        )
        completed_at = datetime.now(timezone.utc)

        return CompositionResult(
            query=query,
            decomposition=decomposition,
            plan=plan,
            execution=execution_trace,
            response=response,
            total_duration_ms=int((completed_at - started_at).total_seconds() * 1000),
            phase_durations=phase_durations,
            status=CompositionStatus.FAILED,
            success=False,
            errors=[msg, *(_truncate(r) for r in reasons)],
            error=msg,
            started_at=started_at,
            completed_at=completed_at,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def compose_query(
    query: str,
    llm_client: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> CompositionResult:
    """
    Convenience function to compose a query.

    Args:
        query: The user's multi-faceted query
        llm_client: LLM client shared by every composition phase (DI mode —
            tests). Leave ``None`` (#1557) for factory mode: a correctly-SIZED
            client per phase (#1365), notably thinking-off planning so the
            planner JSON cannot truncate. Production entry points want ``None``.
        context: Optional context dictionary
        **kwargs: Additional arguments for ToolComposer

    Returns:
        CompositionResult with the synthesized response
    """
    composer = ToolComposer(llm_client=llm_client, **kwargs)
    return await composer.compose(query, context)


def compose_query_sync(
    query: str,
    llm_client: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> CompositionResult:
    """
    Synchronous wrapper for query composition.

    Handles event loop conflicts when called from async contexts.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(compose_query(query, llm_client, context, **kwargs))
    else:
        return asyncio.run(compose_query(query, llm_client, context, **kwargs))


# ============================================================================
# INTEGRATION WITH ORCHESTRATOR
# ============================================================================


class ToolComposerIntegration:
    """
    Integration helper for the Orchestrator to use Tool Composer.

    This class provides the interface that the Orchestrator uses
    to invoke the Tool Composer for MULTI_FACETED queries.
    """

    def __init__(self, composer: ToolComposer):
        self.composer = composer

    async def handle_multi_faceted_query(
        self, query: str, extracted_entities: Dict[str, Any], user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a multi-faceted query from the Orchestrator.

        Args:
            query: The classified MULTI_FACETED query
            extracted_entities: Entities extracted by the NLP layer
            user_context: User context (filters, permissions, etc.)

        Returns:
            Response dictionary in the format expected by Orchestrator
        """
        # Merge context
        context = {**extracted_entities, **user_context}

        # Run composition
        result = await self.composer.compose(query, context)

        # Format for Orchestrator
        return {
            "success": result.success,
            "response": result.response.answer,
            "confidence": result.response.confidence,
            "supporting_data": result.response.supporting_data,
            "citations": result.response.citations,
            "caveats": result.response.caveats,
            "tools_executed": result.execution.tools_executed,
            "tools_succeeded": result.execution.tools_succeeded,
            "metadata": {
                "composition_id": result.composition_id,
                "sub_questions": result.decomposition.question_count,
                "tools_executed": result.execution.tools_executed,
                "tools_succeeded": result.execution.tools_succeeded,
                "total_duration_ms": result.total_duration_ms,
                "phase_durations": result.phase_durations,
            },
        }
