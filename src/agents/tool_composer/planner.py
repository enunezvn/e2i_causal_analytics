"""
E2I Tool Composer - Phase 2: Planner
Version: 4.4
Purpose: Map sub-questions to tools and create execution plan

V4.4 Updates:
- Added causal discovery tool hints (discover_dag, rank_drivers, detect_structural_drift)
- Added tool chaining guidance for discover_dag → rank_drivers pipeline
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.tool_registry.registry import ToolRegistry
from src.utils.llm_content import normalize_llm_content, parse_llm_json

from .cache import get_cache_manager
from .memory_hooks import ToolComposerMemoryHooks, get_tool_composer_memory_hooks
from .models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStep,
    ToolMapping,
)

logger = logging.getLogger(__name__)

# Input-mapping keys whose VALUES are expected to be dataset column names.
# Used by the F6 schema-binding warning to spot invented columns.
_COLUMN_ARG_KEYS = frozenset(
    {
        "treatment",
        "outcome",
        "segment",
        "segments",
        "metric",
        "covariate",
        "covariates",
        "dimension",
    }
)

# #810: argument keys that name the causal OUTCOME/target. When a query targets a
# defined KPI, these are bound deterministically to the KPI's outcome column.
_OUTCOME_ARG_KEYS = ("outcome", "target")


# ============================================================================
# PLANNING PROMPT
# ============================================================================

PLANNING_SYSTEM_PROMPT = """You are a tool planning specialist for a pharmaceutical analytics platform.

Your task is to map sub-questions to available tools and create an execution plan.

## Available Tools:
{tools_description}

## Causal Discovery Tool Hints:
Use these mappings for causal discovery queries:
- "discover causal structure" / "learn DAG" / "causal graph" / "causal relationships" → discover_dag
- "causal vs predictive" / "driver ranking" / "feature importance" / "which features cause" → rank_drivers
- "structural drift" / "causal structure changed" / "DAG stability" → detect_structural_drift

Tool Chaining for Causal Discovery:
- discover_dag produces `edge_list` → rank_drivers can consume it as `dag_edge_list`
- Example chain: discover_dag (step_1) → rank_drivers (step_2) with input_mapping: {{"dag_edge_list": "$step_1.edge_list"}}

## Guidelines:
1. Match each sub-question to the most appropriate tool based on:
   - The question's intent (CAUSAL, COMPARATIVE, etc.)
   - Required inputs and outputs
   - Tool capabilities
2. Identify which tool outputs feed into which tool inputs
3. Determine execution order based on dependencies
4. Group independent tools for parallel execution

## Output Format:
Return a JSON object with:
{{
  "reasoning": "Your step-by-step reasoning for the plan",
  "tool_mappings": [
    {{
      "sub_question_id": "sq_1",
      "tool_name": "tool_name_here",
      "confidence": 0.95,
      "reasoning": "Why this tool fits"
    }}
  ],
  "execution_steps": [
    {{
      "step_id": "step_1",
      "sub_question_id": "sq_1",
      "tool_name": "tool_name_here",
      "input_mapping": {{
        "param_name": "value or $step_X.field for prior outputs"
      }},
      "depends_on_steps": []
    }}
  ],
  "parallel_groups": [
    ["step_1", "step_2"],  // Steps that can run in parallel
    ["step_3"]             // Must wait for group 1
  ]
}}

## CRITICAL REQUIREMENTS:
- You MUST provide a tool_mapping for EVERY sub-question - no exceptions
- You MUST provide an execution_step for EVERY sub-question - no exceptions
- If no tool fits perfectly, choose the closest match (prefer causal_effect_estimator for analysis, risk_scorer for predictions)
- Use $step_X.field syntax to reference prior step outputs
- Parallel groups should be ordered by execution wave
- Verify your response includes ALL sub-question IDs before returning"""


PLANNING_USER_TEMPLATE = """Create an execution plan for these sub-questions:

SUB-QUESTIONS:
{sub_questions}

Map each to the best tool and create an execution plan.
Return valid JSON only."""


# ============================================================================
# PLANNER CLASS
# ============================================================================


class ToolPlanner:
    """
    Maps sub-questions to tools and creates execution plans.

    This is Phase 2 of the Tool Composer pipeline.

    Memory Integration (G1, G2):
    - Uses episodic memory to find similar past compositions
    - Leverages successful patterns to optimize tool selection
    """

    def __init__(
        self,
        llm_client: Any,
        tool_registry: Optional[ToolRegistry] = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.2,
        max_tools_per_plan: int = 8,
        memory_hooks: Optional[ToolComposerMemoryHooks] = None,
        use_episodic_memory: bool = True,
        enable_caching: bool = True,
    ):
        self.llm_client = llm_client
        self.registry = tool_registry or ToolRegistry()
        self.model = model
        self.temperature = temperature
        self.max_tools_per_plan = max_tools_per_plan
        self.memory_hooks = memory_hooks or get_tool_composer_memory_hooks()
        self.use_episodic_memory = use_episodic_memory
        self.enable_caching = enable_caching

        # G6: Initialize cache manager for plan similarity matching
        self._cache_manager = get_cache_manager() if enable_caching else None

    async def plan(
        self,
        decomposition: DecompositionResult,
        available_columns: Optional[List[str]] = None,
        column_profiles: Optional[List[Dict[str, Any]]] = None,
        outcome_hint: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan from decomposed sub-questions.

        Args:
            decomposition: Result from Phase 1 (Decomposer)
            available_columns: Optional list of REAL dataset column names. When
                provided, the planner instructs the LLM to bind tool arguments
                (treatment/outcome/segment/metric/covariates) to these exact
                names (F6) instead of inventing column names.
            column_profiles: Optional richer per-column profile (F6(b)): each
                item is a dict with ``name`` / ``dtype_family`` / ``n_unique`` /
                ``n_nonnull`` / ``values``. When supplied, the planner gives the
                LLM semantic binding guidance (binary->treatment, numeric->
                outcome, low-card categorical->segments; map business terms to
                real columns; never use a brand/region VALUE as a column) AND
                enforces the bindings against the real schema (best-effort
                resolution first, then fail-fast). ``available_columns`` is
                derived from the profile when not explicitly passed (back-compat).

        Returns:
            ExecutionPlan with tool mappings and execution steps
        """
        logger.info(f"Planning execution for {decomposition.question_count} sub-questions")

        # F6(b): keep available_columns working / derivable from the profile.
        if available_columns is None and column_profiles:
            available_columns = [str(p["name"]) for p in column_profiles if p.get("name")]

        try:
            # G6: Check for similar cached plan
            if self._cache_manager:
                cached_result = self._cache_manager.get_similar_plan(decomposition)
                if cached_result:
                    cached_plan, similarity = cached_result
                    logger.info(f"Found similar cached plan (similarity: {similarity:.2f})")
                    # Adapt cached plan to current decomposition
                    adapted_plan = self._adapt_cached_plan(cached_plan, decomposition)
                    if adapted_plan:
                        # KPI outcome hint + treatment guard apply to cached plans
                        # too (#810).
                        self._apply_outcome_hint(
                            adapted_plan.steps, outcome_hint, available_columns
                        )
                        self._apply_treatment_guard(
                            adapted_plan.steps, column_profiles, outcome_hint
                        )
                        return adapted_plan

            # Get available tools for planning
            tools_description = self._format_tools_for_prompt()

            # G1/G2: Check episodic memory for similar past compositions
            similar_compositions = await self._check_episodic_memory(decomposition.original_query)

            # Call LLM for planning (with episodic context if available)
            response = await self._call_llm(
                decomposition,
                tools_description,
                similar_compositions,
                available_columns,
                column_profiles,
                outcome_hint,
            )

            # Parse response
            parsed = self._parse_response(response)

            # Build plan components
            tool_mappings = self._build_tool_mappings(parsed)
            execution_steps = self._build_execution_steps(parsed, decomposition)

            # #810 (KPI-aware): when the query targets a defined KPI, the causal
            # outcome is DEFINITIONALLY the KPI's outcome column — bind it
            # deterministically (the LLM is also prompt-hinted) so the analysis
            # measures the KPI, not an LLM-guessed column. Then guard the
            # treatment to a binary/numeric driver so the estimator can run.
            self._apply_outcome_hint(execution_steps, outcome_hint, available_columns)
            self._apply_treatment_guard(execution_steps, column_profiles, outcome_hint)

            # F6(b) enforcement (defense-in-depth): when the real schema is
            # available, best-effort RESOLVE column-typed args to real columns
            # (case-insensitive / alias / substring), then FAIL FAST on any that
            # still cannot be bound — so an invented column never reaches the
            # fail-closed tool. When NO schema is available we keep the original
            # F6 warn-only fallback (we cannot validate without a schema).
            if available_columns:
                self._enforce_column_bindings(execution_steps, available_columns)
            else:
                self._warn_unbound_columns(execution_steps, available_columns)

            parallel_groups = parsed.get("parallel_groups", [])

            # Validate plan
            self._validate_plan(tool_mappings, execution_steps, decomposition)

            # Calculate estimated duration
            estimated_duration = self._estimate_duration(execution_steps)

            plan = ExecutionPlan(
                decomposition=decomposition,
                steps=execution_steps,
                tool_mappings=tool_mappings,
                estimated_duration_ms=estimated_duration,
                parallel_groups=parallel_groups,
                planning_reasoning=parsed.get("reasoning", ""),
                timestamp=datetime.now(timezone.utc),
            )

            # G6: Cache the plan for future similarity matching
            if self._cache_manager:
                self._cache_manager.cache_plan(decomposition, plan)
                logger.debug("Cached plan for future similarity matching")

            logger.info(f"Created plan with {len(execution_steps)} steps")
            return plan

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            raise PlanningError(f"Failed to create execution plan: {e}") from e

    def _adapt_cached_plan(
        self, cached_plan: ExecutionPlan, decomposition: DecompositionResult
    ) -> Optional[ExecutionPlan]:
        """
        Adapt a cached plan to a new decomposition if possible.

        Returns None if adaptation is not feasible.
        """
        try:
            # Check if sub-question counts match
            if len(cached_plan.steps) != decomposition.question_count:
                logger.debug("Cached plan step count doesn't match, skipping adaptation")
                return None

            # Create new plan with same structure but updated decomposition
            # Only adapt if tool sequences match the intent patterns
            new_plan = ExecutionPlan(
                decomposition=decomposition,
                steps=cached_plan.steps,  # Reuse steps structure
                tool_mappings=cached_plan.tool_mappings,
                estimated_duration_ms=cached_plan.estimated_duration_ms,
                parallel_groups=cached_plan.parallel_groups,
                planning_reasoning=f"Adapted from cached plan: {cached_plan.planning_reasoning}",
                timestamp=datetime.now(timezone.utc),
            )
            return new_plan
        except Exception as e:
            logger.debug(f"Failed to adapt cached plan: {e}")
            return None

    async def _check_episodic_memory(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Check episodic memory for similar past compositions (G1, G2).

        Uses vector search to find successful compositions that can inform
        tool selection and execution planning.

        Args:
            query: The original query to find similar compositions for
            limit: Maximum number of similar compositions to retrieve

        Returns:
            List of similar compositions with their tool sequences
        """
        if not self.use_episodic_memory or not self.memory_hooks:
            return []

        try:
            # find_similar_compositions hydrates raw_content by memory_id
            # (#889): the search RPC's TABLE shape carries no raw_content, so
            # before the hook-side fix every row here read {} and the
            # tool_sequence/confidence below were always the defaults.
            similar = await self.memory_hooks.find_similar_compositions(query=query, limit=limit)

            if similar:
                logger.info(f"Found {len(similar)} similar compositions in episodic memory")
                # Log the tool sequences for debugging
                for comp in similar:
                    raw = comp.get("raw_content", {})
                    logger.debug(
                        f"  Similar: tools={raw.get('tool_sequence', [])}, "
                        f"confidence={raw.get('confidence', 0):.2f}"
                    )
            return similar
        except Exception as e:
            logger.warning(f"Failed to check episodic memory: {e}")
            return []

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the planning prompt"""
        schemas = self.registry.get_schemas_for_planning()

        if not schemas:
            raise PlanningError("No tools available in registry")

        lines = []
        for tool in schemas:
            lines.append(f"### {tool['name']} ({tool['source']})")
            lines.append(f"Description: {tool['description']}")
            lines.append(f"Inputs: {', '.join(tool['inputs'])}")
            lines.append(f"Output: {tool['output']}")
            lines.append(f"Avg execution: {tool['avg_ms']}ms")
            lines.append("")

        return "\n".join(lines)

    async def _call_llm(
        self,
        decomposition: DecompositionResult,
        tools_description: str,
        similar_compositions: Optional[List[Dict[str, Any]]] = None,
        available_columns: Optional[List[str]] = None,
        column_profiles: Optional[List[Dict[str, Any]]] = None,
        outcome_hint: Optional[str] = None,
    ) -> str:
        """Call the LLM for planning with optional episodic + schema context."""
        # Format sub-questions
        sq_text = "\n".join(
            [
                f"- {sq.id}: {sq.question} [Intent: {sq.intent}] [Depends on: {sq.depends_on}]"
                for sq in decomposition.sub_questions
            ]
        )

        system_prompt = PLANNING_SYSTEM_PROMPT.format(tools_description=tools_description)

        # G1/G2: Include similar compositions as context
        episodic_context = ""
        if similar_compositions:
            episodic_context = self._format_episodic_context(similar_compositions)

        user_message = PLANNING_USER_TEMPLATE.format(sub_questions=sq_text)
        if episodic_context:
            user_message = f"{episodic_context}\n\n{user_message}"

        # F6 / F6(b): schema-binding -- when a real dataset is present, list its
        # columns (with dtype family + value distributions when a profile is
        # supplied) and instruct the LLM to bind argument values to EXACT names
        # so the downstream tools do not fail-closed on invented columns AND so
        # it avoids degenerate / near-constant targets.
        columns_block = self._format_columns_block(available_columns, column_profiles)
        if columns_block:
            user_message = f"{user_message}\n\n{columns_block}"

        # #810: when the query targets a defined KPI, tell the LLM the causal
        # outcome column explicitly so it selects sensible drivers/segments.
        if outcome_hint:
            user_message = (
                f"{user_message}\n\n## KPI outcome\n"
                f"This query targets a defined KPI whose causal OUTCOME is the column "
                f"`{outcome_hint}`. Bind every causal `outcome` (and any `target`) argument "
                f"to `{outcome_hint}`, and choose `treatment`/`segments` from the OTHER "
                f"available columns (the drivers)."
            )

        # Using LangChain's message format (works with ChatAnthropic/ChatOpenAI)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = await self.llm_client.ainvoke(messages)

        # AIMessage.content is str | list of content blocks (#1350)
        return normalize_llm_content(response.content)

    def _format_columns_block(
        self,
        available_columns: Optional[List[str]],
        column_profiles: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the schema-binding prompt section for real dataset columns.

        When ``column_profiles`` is supplied (F6(b)) the block carries each
        column's dtype family, cardinality, non-null count and (for low-card
        columns) its value list, plus explicit SEMANTIC binding guidance so the
        LLM picks a binary column for ``treatment``, a numeric column for
        ``outcome`` and low-card categoricals for ``segments`` -- and never
        binds a brand/region VALUE as a column name.

        Falls back to the legacy F6 name-only block when only
        ``available_columns`` is supplied. Returns an empty string when neither
        is available.
        """
        if column_profiles:
            return self._format_profile_block(column_profiles)

        if not available_columns:
            return ""
        column_list = ", ".join(str(c) for c in available_columns)
        return (
            f"## Available dataset columns: {column_list}\n"
            "A real dataset is loaded. Bind treatment/outcome/segments/metric/"
            "covariates argument VALUES to EXACT names from this list; "
            "do NOT invent column names. Use $step_X.field syntax only to "
            "reference prior step outputs, not for dataset columns."
        )

    def _format_profile_block(self, column_profiles: List[Dict[str, Any]]) -> str:
        """Render the rich per-column profile + semantic binding guidance (F6(b))."""
        # Keep the legacy "## Available dataset columns:" header so existing
        # name-only assertions and the comma-list contract still hold.
        names = [str(p.get("name", "")) for p in column_profiles if p.get("name")]
        column_list = ", ".join(names)

        lines = [
            f"## Available dataset columns: {column_list}",
            "",
            "## Column profile (dtype family | cardinality | values):",
        ]
        for p in column_profiles:
            name = str(p.get("name", ""))
            fam = str(p.get("dtype_family", "other"))
            n_unique = p.get("n_unique", "?")
            n_nonnull = p.get("n_nonnull", "?")
            values = p.get("values")
            detail = f"- {name}: {fam}, n_unique={n_unique}, n_nonnull={n_nonnull}"
            if values is not None:
                detail += f", values={values}"
            lines.append(detail)

        lines.extend(
            [
                "",
                "## Column-binding rules (CRITICAL — a real dataset is loaded):",
                "- Bind every column-typed argument (treatment, outcome, "
                "confounders, covariates, segments, metric, dimension) to an "
                "EXACT column name from the list above. Do NOT invent column "
                "names.",
                "- Prefer a BINARY (or low-cardinality) column for `treatment`.",
                "- Prefer a NUMERIC-CONTINUOUS column for `outcome`. NEVER pick a "
                "near-constant / degenerate column (one whose values are almost "
                "all the same) as the outcome or treatment — check the value "
                "distribution above.",
                "- Prefer LOW-CARDINALITY CATEGORICAL columns for `segments`.",
                "- Map business terms to the closest REAL column: e.g. "
                '"conversion"/"adoption"/"response" -> the relevant '
                'numeric/binary column; "driver"/"factor" -> a candidate '
                'treatment/confounder column; "segment" -> a categorical '
                "column. If no real column matches a business term, pick the "
                "nearest admissible column rather than inventing one.",
                '- NEVER use a brand or region VALUE (e.g. "Kisqali", '
                '"Northeast") as a column name — those are filter VALUES, not '
                "columns.",
                "- Use $step_X.field syntax ONLY to reference prior step outputs, "
                "never for dataset columns.",
            ]
        )
        return "\n".join(lines)

    def _format_episodic_context(self, similar_compositions: List[Dict[str, Any]]) -> str:
        """Format similar compositions as context for the LLM.

        The rows arrive from ``_check_episodic_memory`` →
        ``find_similar_compositions``, which hydrates ``raw_content`` by
        memory_id (#889) — the search RPC itself returns no ``raw_content``,
        so before the hook-side fix this formatter rendered all-default
        zeros into the prompt. The ``.get(..., {})`` stays as tolerance for
        rows whose stored content is missing/unparseable (hydration yields
        ``{}`` for those — never fabricated values).
        """
        if not similar_compositions:
            return ""

        lines = [
            "## Similar Past Compositions (Use as Reference)",
            "The following successful compositions may inform your planning:",
            "",
        ]

        for i, comp in enumerate(similar_compositions, 1):
            raw = comp.get("raw_content", {})
            tool_seq = raw.get("tool_sequence", [])
            confidence = raw.get("confidence", 0)
            duration = raw.get("total_duration_ms", 0)

            lines.append(f"### Reference {i}")
            lines.append(f"- Tools used: {', '.join(tool_seq)}")
            lines.append(f"- Success confidence: {confidence:.2f}")
            lines.append(f"- Execution time: {duration}ms")
            lines.append("")

        lines.append("Consider similar tool sequences if they match the current query's intent.")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the planning JSON from the LLM response.

        Uses ``parse_llm_json`` (#1364): try the whole payload as bare JSON
        first — so a value legitimately containing ``` is never mangled by
        fence logic — then fall back to each markdown fence, tolerating an
        UNTERMINATED closing fence (models truncate it when they run low on
        tokens). A genuinely truncated payload (the #1365 defect: cut
        mid-first-value) is still unrecoverable and surfaces as a clear
        ``PlanningError`` — the real fix for that is generation-side (a real
        planning token budget with thinking disabled), not parsing.
        """
        try:
            parsed = parse_llm_json(response)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse planning JSON: {str(response)[:200]}...")
            raise PlanningError(f"Invalid JSON in LLM response: {e}") from e

        if not isinstance(parsed, dict):
            raise PlanningError(
                f"Invalid JSON in LLM response: expected a JSON object, got {type(parsed).__name__}"
            )
        return cast(Dict[str, Any], parsed)

    def _build_tool_mappings(self, parsed: Dict[str, Any]) -> List[ToolMapping]:
        """Build ToolMapping objects from parsed response"""
        raw_mappings = parsed.get("tool_mappings", [])

        mappings = []
        for m in raw_mappings:
            # Validate tool exists
            tool_name = m["tool_name"]
            if not self.registry.validate_tool_exists(tool_name):
                logger.warning(f"Tool '{tool_name}' not in registry, skipping")
                continue

            schema = self.registry.get_schema(tool_name)

            mappings.append(
                ToolMapping(
                    sub_question_id=m["sub_question_id"],
                    tool_name=tool_name,
                    source_agent=schema.source_agent if schema else "unknown",
                    confidence=m.get("confidence", 0.8),
                    reasoning=m.get("reasoning", ""),
                )
            )

        return mappings

    def _build_execution_steps(
        self, parsed: Dict[str, Any], decomposition: DecompositionResult
    ) -> List[ExecutionStep]:
        """Build ExecutionStep objects from parsed response"""
        raw_steps = parsed.get("execution_steps", [])

        # Build mapping from sub_question_id to dependencies
        {sq.id: sq.depends_on for sq in decomposition.sub_questions}

        steps: List[ExecutionStep] = []
        for s in raw_steps:
            tool_name = s["tool_name"]

            # Validate tool exists
            if not self.registry.validate_tool_exists(tool_name):
                raise PlanningError(f"Unknown tool in plan: {tool_name}")

            schema = self.registry.get_schema(tool_name)

            # Determine dependency type
            dep_type = DependencyType.SEQUENTIAL
            if not s.get("depends_on_steps"):
                dep_type = DependencyType.PARALLEL

            steps.append(
                ExecutionStep(
                    step_id=s.get("step_id", f"step_{len(steps) + 1}"),
                    sub_question_id=s["sub_question_id"],
                    tool_name=tool_name,
                    source_agent=schema.source_agent if schema else "unknown",
                    input_mapping=s.get("input_mapping", {}),
                    dependency_type=dep_type,
                    depends_on_steps=s.get("depends_on_steps", []),
                )
            )

        return steps

    def _validate_plan(
        self,
        mappings: List[ToolMapping],
        steps: List[ExecutionStep],
        decomposition: DecompositionResult,
    ) -> None:
        """Validate the execution plan"""
        # Check all sub-questions are mapped
        sq_ids = {sq.id for sq in decomposition.sub_questions}
        mapped_ids = {m.sub_question_id for m in mappings}

        missing = sq_ids - mapped_ids
        if missing:
            # Try to auto-map missing sub-questions to appropriate tools
            logger.warning(f"Auto-mapping {len(missing)} unmapped sub-questions: {missing}")
            for sq_id in missing:
                sq = next((sq for sq in decomposition.sub_questions if sq.id == sq_id), None)
                if sq:
                    fallback = self._get_fallback_mapping(sq)
                    if fallback:
                        mappings.append(fallback)
                        # Also add a step for this mapping
                        step = ExecutionStep(
                            step_id=f"step_{len(steps) + 1}",
                            sub_question_id=sq_id,
                            tool_name=fallback.tool_name,
                            source_agent=fallback.source_agent,
                            input_mapping={},
                            dependency_type=DependencyType.PARALLEL,
                            depends_on_steps=[],
                        )
                        steps.append(step)
                        logger.info(f"Auto-mapped {sq_id} to {fallback.tool_name}")
                    else:
                        raise PlanningError(f"Cannot map sub-question {sq_id} to any tool")

        # Check all steps reference valid tools
        for step in steps:
            if not self.registry.validate_tool_exists(step.tool_name):
                raise PlanningError(f"Step references unknown tool: {step.tool_name}")

        # Check step dependencies are valid
        step_ids = {s.step_id for s in steps}
        for step in steps:
            for dep in step.depends_on_steps:
                if dep not in step_ids:
                    raise PlanningError(f"Step {step.step_id} depends on unknown step {dep}")

        # Check for dependency cycles
        self._check_cycles(steps)

    def _warn_unbound_columns(
        self,
        steps: List[ExecutionStep],
        available_columns: Optional[List[str]],
    ) -> None:
        """Warn when a column-arg literal is not a real dataset column (F6).

        This is intentionally non-fatal: the LLM may legitimately use
        ``$step_X.field`` references, list-valued covariates, or dict params,
        so we only flag plain-string literals under known column-arg keys
        (treatment/outcome/segment(s)/metric/covariate(s)/dimension) that are
        absent from ``available_columns``.
        """
        if not available_columns:
            return

        column_set = {str(c) for c in available_columns}
        for step in steps:
            for arg_name, value in step.input_mapping.items():
                if arg_name not in _COLUMN_ARG_KEYS:
                    continue
                # Only literal strings can be a column reference; skip step
                # refs ($step_X.field) and non-string (dict/list) values.
                if not isinstance(value, str) or value.startswith("$"):
                    continue
                if value not in column_set:
                    logger.warning(
                        "Plan step %s binds '%s'='%s' which is not in available "
                        "dataset columns %s — tool may fail-closed on this column.",
                        step.step_id,
                        arg_name,
                        value,
                        sorted(column_set),
                    )

    def _enforce_column_bindings(
        self,
        steps: List[ExecutionStep],
        available_columns: List[str],
    ) -> None:
        """Resolve, then enforce, column-typed args against the real schema (F6(b)).

        Defense-in-depth on top of the LLM's prompt guidance. For each step,
        every column-typed argument (``_COLUMN_ARG_KEYS``: treatment / outcome /
        segment(s) / metric / covariate(s) / dimension) is checked:

        1. **Best-effort resolution.** If the literal value is not an exact
           column but resolves unambiguously to one via a case-insensitive,
           simple-alias (strip non-alphanumerics) or substring match, the step's
           ``input_mapping`` is rewritten in place to the REAL column name.
        2. **Fail-fast.** If a column-typed literal still cannot be resolved to a
           real column, raise ``PlanningError`` with a clear ``unbound column``
           reason BEFORE the plan reaches the executor / fail-closed tool.

        This is deliberately scoped to plain-string literals under known
        column-arg keys: ``$step_X.field`` references, list-valued args and dict
        params are left untouched (they are not single-column references).
        """
        column_list = [str(c) for c in available_columns]
        column_set = set(column_list)
        # case-folded + alias index for resolution (built once)
        norm_index: Dict[str, List[str]] = {}
        for col in column_list:
            norm_index.setdefault(self._normalize_name(col), []).append(col)

        for step in steps:
            for arg_name, value in list(step.input_mapping.items()):
                if arg_name not in _COLUMN_ARG_KEYS:
                    continue
                # Only single-string literals are single-column references.
                if not isinstance(value, str) or value.startswith("$"):
                    continue
                if value in column_set:
                    continue  # already a real column

                resolved = self._resolve_column(value, column_list, norm_index)
                if resolved is not None:
                    logger.info(
                        "Plan step %s: resolved column-arg '%s'='%s' -> real column '%s'.",
                        step.step_id,
                        arg_name,
                        value,
                        resolved,
                    )
                    step.input_mapping[arg_name] = resolved
                    continue

                # Unresolvable -> fail fast with a clear reason.
                raise PlanningError(
                    f"unbound column: {value!r} (not in schema) — plan step "
                    f"{step.step_id} binds '{arg_name}'='{value}', which is not a "
                    f"real dataset column and could not be resolved to one. "
                    f"Available columns: {sorted(column_set)}."
                )

    def _apply_outcome_hint(
        self,
        steps: List[ExecutionStep],
        outcome_hint: Optional[str],
        available_columns: Optional[List[str]],
    ) -> None:
        """Bind every causal ``outcome``/``target`` arg to ``outcome_hint`` (#810).

        When a query targets a defined KPI, the causal outcome is definitionally
        the KPI's outcome column — so override any LLM-guessed ``outcome``/``target``
        literal to it. No-op when there is no hint, or the hint is not a real
        column (we never inject a non-existent column), or the existing value is a
        ``$step`` reference (a real dependency, never clobbered).
        """
        if not outcome_hint:
            return
        if available_columns is None:
            # No schema to validate against -> do NOT inject an unvalidated column.
            logger.info(
                "outcome-hint %r skipped: no available_columns to validate against.",
                outcome_hint,
            )
            return
        if outcome_hint not in available_columns:
            return
        for step in steps:
            for key in _OUTCOME_ARG_KEYS:
                value = step.input_mapping.get(key)
                if isinstance(value, str) and not value.startswith("$") and value != outcome_hint:
                    logger.info(
                        "Plan step %s: binding '%s' -> KPI outcome %r (was %r)",
                        step.step_id,
                        key,
                        outcome_hint,
                        value,
                    )
                    step.input_mapping[key] = outcome_hint

    def _apply_treatment_guard(
        self,
        steps: List[ExecutionStep],
        column_profiles: Optional[List[Dict[str, Any]]],
        outcome_hint: Optional[str],
    ) -> None:
        """Ensure KPI causal steps use a binary/numeric ``treatment`` (#810).

        DoWhy/CATE estimators need a binary (or numeric) treatment; when the LLM
        binds a categorical driver (e.g. ``trigger_type`` with 6 levels) the
        estimate is NaN/empty. Override such a ``treatment`` to the best binary
        (then numeric) driver from the column profile. Categorical columns remain
        valid for ``segments``. Gated on a KPI query (``outcome_hint`` set) so
        non-KPI plans keep the LLM's treatment choice. ``$step`` references are
        never clobbered.
        """
        if not outcome_hint or not column_profiles:
            return
        family = {str(p["name"]): p.get("dtype_family") for p in column_profiles if p.get("name")}
        binary = [n for n, f in family.items() if f == "binary" and n != outcome_hint]
        numeric = [n for n, f in family.items() if f == "numeric-continuous" and n != outcome_hint]
        candidates = binary + numeric
        if not candidates:
            return
        usable = set(candidates)
        for step in steps:
            value = step.input_mapping.get("treatment")
            if isinstance(value, str) and not value.startswith("$") and value not in usable:
                logger.info(
                    "Plan step %s: treatment %r is not binary/numeric; binding to %r "
                    "(KPI causal analysis needs a usable treatment).",
                    step.step_id,
                    value,
                    candidates[0],
                )
                step.input_mapping["treatment"] = candidates[0]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Lowercase and strip non-alphanumerics for alias matching."""
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _resolve_column(
        self,
        value: str,
        column_list: List[str],
        norm_index: Dict[str, List[str]],
    ) -> Optional[str]:
        """Best-effort resolve ``value`` to a single real column, else None.

        Resolution is intentionally conservative: it only returns a column when
        the match is UNAMBIGUOUS (exactly one candidate). Order of attempts:
        exact case-insensitive / alias-normalized, then substring
        (value-in-column direction only). Ambiguous matches return None so
        enforcement fails fast rather than silently picking the wrong column.
        """
        norm_value = self._normalize_name(value)
        if not norm_value:
            return None

        # 1. exact normalized (case-insensitive / alias) match
        exact = norm_index.get(norm_value)
        if exact and len(exact) == 1:
            return exact[0]
        if exact and len(exact) > 1:
            return None  # ambiguous

        # 2. substring match, must be unambiguous. ONLY the value-in-column
        # direction is allowed: an abbreviated / partial LLM value matching a
        # real column (e.g. "engagement" -> "engagement_score"). The reverse
        # (a real column name appearing INSIDE the value) is deliberately NOT
        # used — it let a short real column (e.g. "age") match an unrelated
        # value (e.g. "dosage") and silently substitute the wrong column, which
        # is worse than failing fast on an unbound column.
        candidates = [col for col in column_list if norm_value in self._normalize_name(col)]
        # de-dup while preserving order
        uniq = list(dict.fromkeys(candidates))
        if len(uniq) == 1:
            return uniq[0]
        return None

    def _check_cycles(self, steps: List[ExecutionStep]) -> None:
        """Check for cycles in step dependencies"""
        visited = set()
        rec_stack = set()

        step_map = {s.step_id: s for s in steps}

        def has_cycle(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            step = step_map.get(step_id)
            if step:
                for dep in step.depends_on_steps:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(step_id)
            return False

        for step in steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id):
                    raise PlanningError("Cycle detected in execution plan")

    def _estimate_duration(self, steps: List[ExecutionStep]) -> int:
        """Estimate total execution duration"""
        total_ms = 0
        for step in steps:
            schema = self.registry.get_schema(step.tool_name)
            if schema:
                total_ms += schema.avg_execution_ms
            else:
                total_ms += 1000  # Default estimate

        return total_ms

    def _get_fallback_mapping(self, sq) -> Optional[ToolMapping]:
        """Get a fallback tool mapping for an unmapped sub-question.

        Uses intent-based heuristics to select appropriate tools when
        the LLM fails to provide a mapping.

        Args:
            sq: SubQuestion object with intent and question text

        Returns:
            ToolMapping if a suitable fallback found, None otherwise
        """
        intent = sq.intent.upper() if hasattr(sq, "intent") and sq.intent else "UNKNOWN"
        question_lower = sq.question.lower() if hasattr(sq, "question") else ""

        # Intent-based tool mapping
        intent_to_tool = {
            "CAUSAL": ("causal_effect_estimator", "causal_impact"),
            "COMPARATIVE": ("gap_calculator", "gap_analyzer"),
            "PREDICTIVE": ("risk_scorer", "prediction_synthesizer"),
            "EXPERIMENTAL": ("power_calculator", "experiment_designer"),
            "DESCRIPTIVE": ("cohort_statistics", "cohort_constructor"),
        }

        # Keyword-based fallbacks (when intent doesn't match well)
        keyword_mappings = [
            (["segment", "high-risk", "risk", "score"], ("risk_scorer", "prediction_synthesizer")),
            (["causal", "effect", "impact", "cause"], ("causal_effect_estimator", "causal_impact")),
            (["compare", "difference", "gap", "vs"], ("gap_calculator", "gap_analyzer")),
            (["treatment", "cate", "heterogen"], ("cate_analyzer", "heterogeneous_optimizer")),
            (["sample", "power", "experiment"], ("power_calculator", "experiment_designer")),
            (["drift", "distribution", "change"], ("psi_calculator", "drift_monitor")),
            (["cohort", "patient", "eligible"], ("cohort_builder", "cohort_constructor")),
        ]

        # Try intent-based mapping first
        if intent in intent_to_tool:
            tool_name, source_agent = intent_to_tool[intent]
            if self.registry.validate_tool_exists(tool_name):
                return ToolMapping(
                    sub_question_id=sq.id,
                    tool_name=tool_name,
                    source_agent=source_agent,
                    confidence=0.6,  # Lower confidence for fallback
                    reasoning=f"Auto-mapped based on {intent} intent",
                )

        # Try keyword-based mapping
        for keywords, (tool_name, source_agent) in keyword_mappings:
            if any(kw in question_lower for kw in keywords):
                if self.registry.validate_tool_exists(tool_name):
                    return ToolMapping(
                        sub_question_id=sq.id,
                        tool_name=tool_name,
                        source_agent=source_agent,
                        confidence=0.5,  # Even lower for keyword match
                        reasoning="Auto-mapped based on keyword match",
                    )

        # Last resort: use causal_effect_estimator as generic fallback
        if self.registry.validate_tool_exists("causal_effect_estimator"):
            return ToolMapping(
                sub_question_id=sq.id,
                tool_name="causal_effect_estimator",
                source_agent="causal_impact",
                confidence=0.4,
                reasoning="Fallback mapping - no suitable tool found",
            )

        return None


# ============================================================================
# EXCEPTIONS
# ============================================================================


class PlanningError(Exception):
    """Error during execution planning"""

    pass


# ============================================================================
# SYNC WRAPPER
# ============================================================================


def plan_sync(decomposition: DecompositionResult, llm_client: Any, **kwargs) -> ExecutionPlan:
    """
    Synchronous wrapper for planning.

    Handles event loop conflicts when called from async contexts.
    """
    import asyncio

    planner = ToolPlanner(llm_client=llm_client, **kwargs)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(planner.plan(decomposition))
    else:
        return asyncio.run(planner.plan(decomposition))
