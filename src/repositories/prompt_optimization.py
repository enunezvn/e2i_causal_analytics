"""Persistence layer for the migration-023 GEPA prompt-optimization tables.

Wires the five tables that shipped in database/ml/023_gepa_optimization_tables.sql
(and stayed unwired until now — see database/memory/033_drop_orphan_dspy_tables.sql):

- ``prompt_optimization_runs``       — one row per optimizer ``compile()`` invocation
- ``optimized_instructions``         — versioned per-predictor instruction history
- ``optimized_tool_descriptions``    — versioned tool descriptions (persistence is
  complete; the runtime *producer* waits on dspy GEPA tool-optimization support,
  which dspy 3.1.0 does not ship — see optimizer_setup.create_gepa_optimizer)
- ``prompt_ab_tests`` / ``prompt_ab_test_observations`` — GEPAABTest lifecycle

Requires migration database/ml/035_gepa_persistence_constraints.sql (history
and hash constraints; version width).

Design constraints, in order:

1. NEVER fail an optimization run. The ``record_run_*`` seam functions are
   best-effort: any failure is logged and swallowed (mirrors
   src/agents/feedback_learner/signal_store.py). Compiles cost real LLM budget;
   losing their artifact over a DB hiccup is never acceptable.
2. NEVER fabricate a measurement. Scores come only from dspy GEPA's
   ``detailed_results`` (attached when ``track_stats=True``, which
   create_gepa_optimizer always sets); absent stats persist as NULL.
   ``improvement_percent`` is percentage POINTS ((optimized - baseline) x 100
   on the 0-1 metric scale), not a relative percentage.
3. Work with BOTH sync and async supabase clients (``_exec``), because the
   celery beat path resolves the sync client (signal_store parity) while API
   and test contexts hold the async one. This applies to the module-level
   ``record_run_*`` seams and the methods defined in this module; methods
   inherited from BaseRepository (``get_by_id``, ``get_many``, ...) keep the
   codebase-wide async-client-only contract.
4. Import light. This module must NOT import dspy or anything under
   src.optimization.gepa at module scope: that package's __init__ eagerly
   imports dspy (~714MB RSS). Instruction/stat extraction is duck-typed;
   GEPAABTest is imported function-locally.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent profiles (prompt_optimization_runs.agent_tier / agent_type NOT NULLs)
# ---------------------------------------------------------------------------

#: (tier, type) per the 22-agent 6-tier architecture, mirroring the tier
#: comments on src/optimization/gepa/optimizer_setup.AGENT_BUDGETS. Kept as a
#: local table rather than imported: importing src.optimization.gepa eagerly
#: loads dspy (see module docstring), and the runs table needs only this pair.
AGENT_OPTIMIZATION_PROFILES: Dict[str, Tuple[int, str]] = {
    "tool_composer": (1, "hybrid"),
    "causal_impact": (2, "hybrid"),
    "gap_analyzer": (2, "hybrid"),
    "heterogeneous_optimizer": (2, "hybrid"),
    "experiment_designer": (3, "hybrid"),
    "explainer": (5, "deep"),
    "feedback_learner": (5, "deep"),
}

#: Agents not in the registry above (recipients, the cognitive-RAG synthesis
#: module, future agents) — Tier 4 standard, the architecture's largest band.
DEFAULT_AGENT_PROFILE: Tuple[int, str] = (4, "standard")

_OPTIMIZER_TYPES = {"miprov2", "gepa", "bootstrap_fewshot", "copro", "simba", "manual"}
_BUDGET_PRESETS = {"light", "medium", "heavy", "custom"}


def resolve_agent_profile(agent_name: str) -> Tuple[int, str]:
    """(tier, type) for an agent or an agent-derived artifact name.

    Derived names extend the base agent with an underscore-separated suffix
    (``feedback_learner_pattern``, ``feedback_learner_recommendation`` — see
    optimization_runner). Matching is exact-or-derived (``name == key`` or
    ``name.startswith(key + "_")``), longest key first, NEVER a loose substring
    match — ``experiment_monitor`` must not inherit ``experiment_designer``'s
    profile.
    """
    for key in sorted(AGENT_OPTIMIZATION_PROFILES, key=len, reverse=True):
        if agent_name == key or agent_name.startswith(key + "_"):
            return AGENT_OPTIMIZATION_PROFILES[key]
    return DEFAULT_AGENT_PROFILE


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def instruction_hash(text: str) -> str:
    """SHA256 hex of instruction text.

    MUST stay identical to src/optimization/gepa/versioning.py::
    compute_instruction_hash (pinned by a unit test) — DB rows and file
    artifacts must dedup identically. Not imported from there because that
    package's __init__ eagerly imports dspy.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def improvement_percentage_points(
    baseline: Optional[float], optimized: Optional[float]
) -> Optional[float]:
    """Score delta in percentage POINTS on the 0-1 metric scale, or None.

    The GEPA paper's gains are percentage points (Agrawal et al., 2025,
    arXiv:2507.19457); this column follows the same convention. None when
    either side is unmeasured — never a fabricated number.
    """
    if baseline is None or optimized is None:
        return None
    return round((optimized - baseline) * 100.0, 2)


def extract_module_instructions(module: Any) -> List[Tuple[str, str]]:
    """[(predictor_name, instruction_text)] from a compiled DSPy module.

    Duck-typed (``named_predictors`` + ``signature.instructions``, with the
    pre-3.x ``extended_signature`` fallback — same resolution order as
    versioning.save_optimized_module) so it needs no dspy import and returns
    [] for anything that is not a module.
    """
    named = getattr(module, "named_predictors", None)
    if not callable(named):
        return []
    entries: List[Tuple[str, str]] = []
    try:
        for name, predictor in named():
            sig = getattr(predictor, "extended_signature", None) or getattr(
                predictor, "signature", None
            )
            instructions = getattr(sig, "instructions", None) if sig is not None else None
            if instructions:
                entries.append((str(name), str(instructions)))
    except Exception as e:  # noqa: BLE001 - extraction is best-effort
        logger.warning("Could not extract instructions from module: %s", e)
        return []
    return entries


def extract_run_stats(module: Any) -> Dict[str, Any]:
    """Measured run stats from ``module.detailed_results`` (dspy GEPA attaches
    a DspyGEPAResult when track_stats=True; candidate 0 is the seed program,
    so its aggregate val score IS the baseline). {} when stats are absent —
    the caller persists NULLs, never invented numbers.
    """
    detailed = getattr(module, "detailed_results", None)
    if detailed is None:
        return {}
    stats: Dict[str, Any] = {}
    try:
        scores = getattr(detailed, "val_aggregate_scores", None)
        if scores:
            stats["baseline_score"] = float(scores[0])
            best_idx = getattr(detailed, "best_idx", None)
            if best_idx is not None and 0 <= int(best_idx) < len(scores):
                stats["optimized_score"] = float(scores[int(best_idx)])
                stats["best_candidate_idx"] = int(best_idx)
        total_calls = getattr(detailed, "total_metric_calls", None)
        if total_calls is not None:
            stats["total_metric_calls"] = int(total_calls)
        candidates = getattr(detailed, "candidates", None)
        if candidates is not None:
            stats["num_candidates_explored"] = len(candidates)
        per_instance = getattr(detailed, "per_val_instance_best_candidates", None)
        if per_instance:
            frontier: set = set()
            for winners in per_instance:
                frontier.update(winners)
            stats["pareto_frontier_size"] = len(frontier)
    except Exception as e:  # noqa: BLE001 - stats are best-effort
        logger.warning("Could not extract GEPA run stats: %s", e)
    return stats


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _clean_float(value: Any) -> Optional[float]:
    """float() that maps None/NaN (scipy can emit NaN p-values) to None."""
    if value is None:
        return None
    f = float(value)
    return None if math.isnan(f) else f


async def _exec(query: Any) -> Any:
    """Execute a postgrest query on a sync OR async supabase client."""
    result = query.execute()
    return await result if inspect.isawaitable(result) else result


# ---------------------------------------------------------------------------
# prompt_optimization_runs
# ---------------------------------------------------------------------------


class PromptOptimizationRunRepository(BaseRepository):
    """One row per optimizer compile() invocation."""

    table_name = "prompt_optimization_runs"
    id_column = "run_id"

    async def start_run(
        self,
        *,
        agent_name: str,
        trainset_size: int,
        optimizer_type: str = "gepa",
        budget_preset: str = "light",
        valset_size: Optional[int] = None,
        agent_tier: Optional[int] = None,
        agent_type: Optional[str] = None,
        run_name: Optional[str] = None,
        max_metric_calls: Optional[int] = None,
        reflection_model: Optional[str] = None,
        enable_tool_optimization: bool = False,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        created_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Insert a 'running' row and return it (run_id included)."""
        if not self.client:
            return None
        tier, agent_kind = resolve_agent_profile(agent_name)
        now = _utcnow_iso()
        record: Dict[str, Any] = {
            "run_name": run_name
            or f"{agent_name}_{optimizer_type}_{now[:19].replace(':', '').replace('-', '')}",
            "agent_name": agent_name,
            "agent_tier": agent_tier if agent_tier is not None else tier,
            "agent_type": agent_type or agent_kind,
            # Unknown values are NOT coerced to a wrong enum member: postgres
            # rejects them, the recorder logs and skips — no row over a wrong row.
            "optimizer_type": optimizer_type,
            "budget_preset": budget_preset if budget_preset in _BUDGET_PRESETS else "custom",
            "trainset_size": trainset_size,
            "status": "running",
            "started_at": now,
        }
        if valset_size is not None:
            record["valset_size"] = valset_size
        if max_metric_calls is not None:
            record["max_metric_calls"] = max_metric_calls
        if reflection_model:
            record["reflection_model"] = reflection_model
        if enable_tool_optimization:
            record["enable_tool_optimization"] = True
        if config is not None:
            record["config_json"] = config
        if seed is not None:
            record["seed"] = seed
        if created_by:
            record["created_by"] = created_by

        result = await _exec(self.client.table(self.table_name).insert(record))
        return dict(result.data[0]) if result.data else None

    async def _finish_run(self, run_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply terminal-status updates, deriving duration from started_at."""
        if not self.client:
            return None
        existing = await _exec(
            self.client.table(self.table_name).select("started_at").eq(self.id_column, run_id)
        )
        started = _parse_ts(existing.data[0]["started_at"]) if existing.data else None
        completed = datetime.now(timezone.utc)
        updates = {"completed_at": completed.isoformat(), **updates}
        if started is not None:
            updates["duration_seconds"] = max(0, int((completed - started).total_seconds()))
        result = await _exec(
            self.client.table(self.table_name).update(updates).eq(self.id_column, run_id)
        )
        return dict(result.data[0]) if result.data else None

    async def complete_run(
        self,
        run_id: str,
        *,
        baseline_score: Optional[float] = None,
        optimized_score: Optional[float] = None,
        total_metric_calls: Optional[int] = None,
        num_candidates_explored: Optional[int] = None,
        pareto_frontier_size: Optional[int] = None,
        best_candidate_idx: Optional[int] = None,
        log_dir: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark completed, persisting only what was measured (rest stays NULL)."""
        updates: Dict[str, Any] = {"status": "completed"}
        if baseline_score is not None:
            updates["baseline_score"] = baseline_score
        if optimized_score is not None:
            updates["optimized_score"] = optimized_score
        improvement = improvement_percentage_points(baseline_score, optimized_score)
        if improvement is not None:
            updates["improvement_percent"] = improvement
        if total_metric_calls is not None:
            updates["total_metric_calls"] = total_metric_calls
        if num_candidates_explored is not None:
            updates["num_candidates_explored"] = num_candidates_explored
        if pareto_frontier_size is not None:
            updates["pareto_frontier_size"] = pareto_frontier_size
        if best_candidate_idx is not None:
            updates["best_candidate_idx"] = best_candidate_idx
        if log_dir:
            updates["log_dir"] = log_dir
        if mlflow_run_id:
            updates["mlflow_run_id"] = mlflow_run_id
        return await self._finish_run(run_id, updates)

    async def fail_run(
        self,
        run_id: str,
        error_message: str,
        error_traceback: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {"status": "failed", "error_message": error_message}
        if error_traceback:
            updates["error_traceback"] = error_traceback
        return await self._finish_run(run_id, updates)

    async def list_runs(self, agent_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        result = await _exec(
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_name", agent_name)
            .order("created_at", desc=True)
            .limit(limit)
        )
        return [dict(row) for row in result.data or []]


# ---------------------------------------------------------------------------
# optimized_instructions
# ---------------------------------------------------------------------------


class OptimizedInstructionRepository(BaseRepository):
    """Versioned per-predictor instruction history."""

    table_name = "optimized_instructions"
    id_column = "instruction_id"

    async def record_instructions(
        self,
        *,
        run_id: str,
        agent_name: str,
        version: str,
        entries: Sequence[Tuple[str, str]],
        val_score: Optional[float] = None,
        val_score_components: Optional[Dict[str, Any]] = None,
        candidate_idx: Optional[int] = None,
        parent_indices: Optional[List[int]] = None,
        discovery_eval_count: Optional[int] = None,
        is_baseline: bool = False,
    ) -> List[Dict[str, Any]]:
        """Insert one history row per (predictor_name, instruction_text).

        Dedup: identical text for the same (agent, predictor) is one row, ever
        (upsert on the 035 predictor-scoped hash index, duplicates ignored).
        Returns the CANONICAL rows for every entry — pre-existing rows
        included, so a dedup'd re-record still hands back usable ids.
        """
        if not self.client or not entries:
            return []
        records = []
        for predictor_name, text in entries:
            record: Dict[str, Any] = {
                "run_id": run_id,
                "agent_name": agent_name,
                "predictor_name": predictor_name,
                "version": version,
                "is_active": False,
                "is_baseline": is_baseline,
                "instruction_text": text,
                "instruction_hash": instruction_hash(text),
            }
            if val_score is not None:
                record["val_score"] = val_score
            if val_score_components is not None:
                record["val_score_components"] = val_score_components
            if candidate_idx is not None:
                record["candidate_idx"] = candidate_idx
            if parent_indices is not None:
                record["parent_indices"] = parent_indices
            if discovery_eval_count is not None:
                record["discovery_eval_count"] = discovery_eval_count
            records.append(record)

        await _exec(
            self.client.table(self.table_name).upsert(
                records,
                on_conflict="agent_name,predictor_name,instruction_hash",
                ignore_duplicates=True,
            )
        )
        wanted = {(r["predictor_name"], r["instruction_hash"]) for r in records}
        result = await _exec(
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_name", agent_name)
            .in_("instruction_hash", [r["instruction_hash"] for r in records])
        )
        return [
            dict(row)
            for row in result.data or []
            if (row["predictor_name"], row["instruction_hash"]) in wanted
        ]

    async def activate(self, instruction_id: str) -> Optional[Dict[str, Any]]:
        """Make this version the single active one for its (agent, predictor).

        Two statements, not a transaction (postgrest has none). If the second
        statement RAISES, the rows the first one deactivated are restored
        best-effort, so an exception does not leave the pair with no active
        version. A hard process crash between the statements still can — that
        residual window fails safe: the artifact loader
        (versioning.load_optimized_module) never reads this table and keeps
        serving the newest saved artifact.
        """
        if not self.client:
            return None
        target = await _exec(
            self.client.table(self.table_name).select("*").eq(self.id_column, instruction_id)
        )
        if not target.data:
            return None
        row = target.data[0]
        now = _utcnow_iso()
        deactivated = await _exec(
            self.client.table(self.table_name)
            .update({"is_active": False, "deactivated_at": now})
            .eq("agent_name", row["agent_name"])
            .eq("predictor_name", row["predictor_name"])
            .eq("is_active", True)
        )
        try:
            result = await _exec(
                self.client.table(self.table_name)
                .update({"is_active": True, "activated_at": now, "deactivated_at": None})
                .eq(self.id_column, instruction_id)
            )
        except Exception:
            for prev in deactivated.data or []:
                if prev[self.id_column] == instruction_id:
                    continue  # re-activating the failed target would fail again
                try:
                    await _exec(
                        self.client.table(self.table_name)
                        .update({"is_active": True, "deactivated_at": None})
                        .eq(self.id_column, prev[self.id_column])
                    )
                except Exception as restore_error:  # noqa: BLE001 - best-effort restore
                    logger.warning(
                        "activate() restore of %s failed: %s",
                        prev[self.id_column],
                        restore_error,
                    )
            raise
        return dict(result.data[0]) if result.data else None

    async def get_active(self, agent_name: str) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        result = await _exec(
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_name", agent_name)
            .eq("is_active", True)
        )
        return [dict(row) for row in result.data or []]

    async def get_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        result = await _exec(self.client.table(self.table_name).select("*").eq("run_id", run_id))
        return [dict(row) for row in result.data or []]


# ---------------------------------------------------------------------------
# optimized_tool_descriptions
# ---------------------------------------------------------------------------


class OptimizedToolDescriptionRepository(BaseRepository):
    """Versioned tool-description history.

    The persistence side is complete; the runtime producer arrives when dspy
    GEPA supports tool-description optimization (create_gepa_optimizer's
    enable_tool_optimization is reserved for exactly that).
    """

    table_name = "optimized_tool_descriptions"
    id_column = "tool_description_id"

    async def record_tool_description(
        self,
        *,
        run_id: str,
        agent_name: str,
        tool_name: str,
        version: str,
        description_text: str,
        argument_descriptions: Optional[Dict[str, Any]] = None,
        original_description: Optional[str] = None,
        original_arguments: Optional[Dict[str, Any]] = None,
        tool_selection_accuracy: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        record: Dict[str, Any] = {
            "run_id": run_id,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "version": version,
            "is_active": False,
            "description_text": description_text,
            "description_hash": instruction_hash(description_text),
        }
        if argument_descriptions is not None:
            record["argument_descriptions"] = argument_descriptions
        if original_description is not None:
            record["original_description"] = original_description
        if original_arguments is not None:
            record["original_arguments"] = original_arguments
        if tool_selection_accuracy is not None:
            record["tool_selection_accuracy"] = tool_selection_accuracy
        result = await _exec(self.client.table(self.table_name).insert(record))
        return dict(result.data[0]) if result.data else None

    async def activate(self, tool_description_id: str) -> Optional[Dict[str, Any]]:
        """Single active description per (agent, tool); same two-step,
        exception-restore, and crash-fail-safe direction as
        OptimizedInstructionRepository.activate."""
        if not self.client:
            return None
        target = await _exec(
            self.client.table(self.table_name).select("*").eq(self.id_column, tool_description_id)
        )
        if not target.data:
            return None
        row = target.data[0]
        deactivated = await _exec(
            self.client.table(self.table_name)
            .update({"is_active": False})
            .eq("agent_name", row["agent_name"])
            .eq("tool_name", row["tool_name"])
            .eq("is_active", True)
        )
        try:
            result = await _exec(
                self.client.table(self.table_name)
                .update({"is_active": True})
                .eq(self.id_column, tool_description_id)
            )
        except Exception:
            for prev in deactivated.data or []:
                if prev[self.id_column] == tool_description_id:
                    continue  # re-activating the failed target would fail again
                try:
                    await _exec(
                        self.client.table(self.table_name)
                        .update({"is_active": True})
                        .eq(self.id_column, prev[self.id_column])
                    )
                except Exception as restore_error:  # noqa: BLE001 - best-effort restore
                    logger.warning(
                        "activate() restore of %s failed: %s",
                        prev[self.id_column],
                        restore_error,
                    )
            raise
        return dict(result.data[0]) if result.data else None

    async def get_active(self, agent_name: str) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        result = await _exec(
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_name", agent_name)
            .eq("is_active", True)
        )
        return [dict(row) for row in result.data or []]


# ---------------------------------------------------------------------------
# prompt_ab_tests + prompt_ab_test_observations
# ---------------------------------------------------------------------------


class PromptABTestRepository(BaseRepository):
    """Persistence for GEPAABTest (src/optimization/gepa/ab_test.py), which is
    otherwise memory-only and loses every observation on process exit."""

    table_name = "prompt_ab_tests"
    id_column = "test_id"
    observations_table = "prompt_ab_test_observations"

    _OBSERVATION_CHUNK = 500

    async def save_test(
        self, ab_test: Any, created_by: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Upsert the test's configuration + lifecycle status (keyed by the
        GEPAABTest's own test_id, so in-memory and DB identity stay one)."""
        if not self.client:
            return None
        record: Dict[str, Any] = {
            "test_id": ab_test.test_id,
            "test_name": ab_test.test_name,
            "agent_name": ab_test.agent_name,
            "traffic_split": ab_test.traffic_split,
            "target_sample_size": ab_test.target_sample_size,
            "status": ab_test.status,
            "baseline_instruction_id": ab_test.baseline_instruction_id,
            "treatment_instruction_id": ab_test.treatment_instruction_id,
            "started_at": ab_test.started_at.isoformat() if ab_test.started_at else None,
            "ended_at": ab_test.ended_at.isoformat() if ab_test.ended_at else None,
        }
        if created_by:
            record["created_by"] = created_by
        result = await _exec(
            self.client.table(self.table_name).upsert(record, on_conflict="test_id")
        )
        return dict(result.data[0]) if result.data else None

    async def record_observations(self, observations: Iterable[Any]) -> int:
        """Bulk-insert observations; idempotent on observation_id (a re-sync of
        an already-persisted window inserts nothing twice). Returns the number
        of NEW rows persisted."""
        if not self.client:
            return 0
        rows = [
            {
                "observation_id": o.observation_id,
                "test_id": o.test_id,
                "request_id": o.request_id,
                "variant": o.variant,
                "score": o.score,
                "latency_ms": o.latency_ms,
                "success": o.success,
                "error_type": o.error_type,
                "user_id": o.user_id,
                "session_id": o.session_id,
            }
            for o in observations
        ]
        inserted = 0
        for start in range(0, len(rows), self._OBSERVATION_CHUNK):
            chunk = rows[start : start + self._OBSERVATION_CHUNK]
            result = await _exec(
                self.client.table(self.observations_table).upsert(
                    chunk, on_conflict="observation_id", ignore_duplicates=True
                )
            )
            inserted += len(result.data or [])
        return inserted

    async def finalize_test(
        self, ab_test: Any, results: Any, status: str = "completed"
    ) -> Optional[Dict[str, Any]]:
        """Persist an ABTestResults (the REAL analyze() output) onto the test row."""
        if not self.client:
            return None
        ci = results.confidence_interval
        updates: Dict[str, Any] = {
            "status": status,
            "ended_at": (ab_test.ended_at.isoformat() if ab_test.ended_at else _utcnow_iso()),
            "baseline_requests": results.baseline_requests,
            "treatment_requests": results.treatment_requests,
            "baseline_score_avg": _clean_float(results.baseline_score_avg),
            "treatment_score_avg": _clean_float(results.treatment_score_avg),
            "baseline_latency_p50": results.baseline_latency_p50,
            "treatment_latency_p50": results.treatment_latency_p50,
            "p_value": _clean_float(results.p_value),
            "confidence_interval_lower": _clean_float(ci[0]) if ci else None,
            "confidence_interval_upper": _clean_float(ci[1]) if ci else None,
            # scipy comparisons yield numpy bools, which json.dumps rejects.
            "is_significant": (
                bool(results.is_significant) if results.is_significant is not None else None
            ),
            "winner": results.winner,
            "decision_reason": results.recommendation,
        }
        result = await _exec(
            self.client.table(self.table_name).update(updates).eq(self.id_column, ab_test.test_id)
        )
        return dict(result.data[0]) if result.data else None

    async def load_test(self, test_id: str) -> Optional[Any]:
        """Reconstruct a GEPAABTest (config + status + observations) from the
        DB, so a test survives process restart. Returns None when unknown."""
        if not self.client:
            return None
        found = await _exec(
            self.client.table(self.table_name).select("*").eq(self.id_column, test_id)
        )
        if not found.data:
            return None
        row = found.data[0]

        # Function-local: importing src.optimization.gepa.* runs that package's
        # dspy-eager __init__ (~714MB) — pay it only when a caller loads a test.
        from src.optimization.gepa.ab_test import ABTestObservation, GEPAABTest

        ab_test = GEPAABTest(
            test_name=row["test_name"],
            agent_name=row["agent_name"],
            traffic_split=float(row["traffic_split"]),
            baseline_instruction_id=row.get("baseline_instruction_id"),
            treatment_instruction_id=row.get("treatment_instruction_id"),
            target_sample_size=row.get("target_sample_size") or 1000,
        )
        ab_test.test_id = row["test_id"]
        ab_test.status = row.get("status") or "draft"
        ab_test.started_at = _parse_ts(row.get("started_at"))
        ab_test.ended_at = _parse_ts(row.get("ended_at"))

        observations = await _exec(
            self.client.table(self.observations_table)
            .select("*")
            .eq("test_id", test_id)
            .order("created_at", desc=False)
        )
        for obs in observations.data or []:
            ab_test.observations.append(
                ABTestObservation(
                    observation_id=obs["observation_id"],
                    test_id=obs["test_id"],
                    variant=obs["variant"],
                    request_id=obs["request_id"],
                    score=_clean_float(obs.get("score")),
                    latency_ms=obs.get("latency_ms"),
                    success=bool(obs.get("success", True)),
                    error_type=obs.get("error_type"),
                    user_id=obs.get("user_id"),
                    session_id=obs.get("session_id"),
                    created_at=_parse_ts(obs.get("created_at")) or datetime.now(timezone.utc),
                )
            )
        return ab_test


# ---------------------------------------------------------------------------
# Recorder seam — what the optimization runner / celery legs call.
# Best-effort: NEVER raises (a DB outage must not fail an optimization run).
# ---------------------------------------------------------------------------


async def _resolve_client(client: Optional[Any]) -> Optional[Any]:
    """Caller's client, or the process-wide one (sync client + _exec, exactly
    like signal_store — proven under the celery beat's loop management)."""
    if client is not None:
        return client
    try:
        from src.memory.services.factories import get_supabase_client

        maybe = get_supabase_client()
        return await maybe if inspect.isawaitable(maybe) else maybe
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        logger.warning("No Supabase client for prompt-optimization recording: %s", e)
        return None


async def record_run_started(
    *,
    agent_name: str,
    trainset_size: int,
    optimizer_type: str = "gepa",
    budget_preset: str = "light",
    valset_size: Optional[int] = None,
    max_metric_calls: Optional[int] = None,
    reflection_model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    created_by: Optional[str] = None,
    client: Optional[Any] = None,
) -> Optional[str]:
    """Insert a 'running' prompt_optimization_runs row; run_id or None."""
    try:
        resolved = await _resolve_client(client)
        if resolved is None:
            return None
        row = await PromptOptimizationRunRepository(resolved).start_run(
            agent_name=agent_name,
            trainset_size=trainset_size,
            optimizer_type=optimizer_type,
            budget_preset=budget_preset,
            valset_size=valset_size,
            max_metric_calls=max_metric_calls,
            reflection_model=reflection_model,
            config=config,
            seed=seed,
            created_by=created_by,
        )
        return row["run_id"] if row else None
    except Exception as e:  # noqa: BLE001 - persistence must never fail the run
        logger.warning("prompt_optimization_runs start not recorded for %s: %s", agent_name, e)
        return None


async def record_run_completed(
    run_id: Optional[str],
    *,
    module: Any = None,
    artifact_info: Optional[Dict[str, Any]] = None,
    instruction_entries: Optional[Sequence[Tuple[str, str]]] = None,
    val_score: Optional[float] = None,
    mlflow_run_id: Optional[str] = None,
    client: Optional[Any] = None,
) -> bool:
    """Mark a run completed with its measured stats, and persist its
    instruction history.

    - Stats come from ``module.detailed_results`` when present ({} otherwise).
    - Instruction rows come from ``instruction_entries`` when given (callers
      whose predictor naming is domain-specific, e.g. recipient template
      fields), else from the module's own named predictors. Pass ``[]``
      explicitly to suppress instruction rows (e.g. a no-improvement run whose
      winning candidate is the unchanged seed).
    - ``version`` is the saved artifact's version_id when one exists, else a
      run-scoped tag — instruction rows are always traceable to their run.
    """
    if not run_id:
        return False
    try:
        resolved = await _resolve_client(client)
        if resolved is None:
            return False
        stats = extract_run_stats(module) if module is not None else {}
        artifact = artifact_info or {}
        updated = await PromptOptimizationRunRepository(resolved).complete_run(
            run_id,
            baseline_score=stats.get("baseline_score"),
            optimized_score=stats.get("optimized_score"),
            total_metric_calls=stats.get("total_metric_calls"),
            num_candidates_explored=stats.get("num_candidates_explored"),
            pareto_frontier_size=stats.get("pareto_frontier_size"),
            best_candidate_idx=stats.get("best_candidate_idx"),
            log_dir=artifact.get("path"),
            mlflow_run_id=mlflow_run_id,
        )
        if updated is None:
            return False

        entries = (
            list(instruction_entries)
            if instruction_entries is not None
            else extract_module_instructions(module)
        )
        if entries:
            await OptimizedInstructionRepository(resolved).record_instructions(
                run_id=run_id,
                agent_name=updated["agent_name"],
                version=artifact.get("version_id") or f"run_{run_id}",
                entries=entries,
                val_score=(val_score if val_score is not None else stats.get("optimized_score")),
                candidate_idx=stats.get("best_candidate_idx"),
            )
        return True
    except Exception as e:  # noqa: BLE001 - persistence must never fail the run
        logger.warning("prompt_optimization_runs completion not recorded (%s): %s", run_id, e)
        return False


async def record_run_failed(
    run_id: Optional[str],
    error_message: str,
    error_traceback: Optional[str] = None,
    client: Optional[Any] = None,
) -> bool:
    """Mark a run failed with its real error. Best-effort."""
    if not run_id:
        return False
    try:
        resolved = await _resolve_client(client)
        if resolved is None:
            return False
        updated = await PromptOptimizationRunRepository(resolved).fail_run(
            run_id, error_message, error_traceback
        )
        return updated is not None
    except Exception as e:  # noqa: BLE001 - persistence must never fail the run
        logger.warning("prompt_optimization_runs failure not recorded (%s): %s", run_id, e)
        return False


async def record_run_discarded(
    run_id: Optional[str],
    client: Optional[Any] = None,
) -> bool:
    """Delete a provisional 'running' row whose optimizer skipped before
    spending any budget.

    Both FeedbackLearnerOptimizer paths return None ONLY from pre-compile
    guards (dspy/GEPA unavailable, too few examples, unavailable phase);
    compile failures raise and land in record_run_failed instead. Discarding
    the provisional row keeps the table's contract: a row exists exactly when
    real metric/LLM calls were made (or died trying). Best-effort.
    """
    if not run_id:
        return False
    try:
        resolved = await _resolve_client(client)
        if resolved is None:
            return False
        result = await _exec(
            resolved.table("prompt_optimization_runs").delete().eq("run_id", run_id)
        )
        return bool(result.data)
    except Exception as e:  # noqa: BLE001 - persistence must never fail the run
        logger.warning("prompt_optimization_runs discard not recorded (%s): %s", run_id, e)
        return False


__all__ = [
    "AGENT_OPTIMIZATION_PROFILES",
    "DEFAULT_AGENT_PROFILE",
    "resolve_agent_profile",
    "instruction_hash",
    "improvement_percentage_points",
    "extract_module_instructions",
    "extract_run_stats",
    "PromptOptimizationRunRepository",
    "OptimizedInstructionRepository",
    "OptimizedToolDescriptionRepository",
    "PromptABTestRepository",
    "record_run_started",
    "record_run_completed",
    "record_run_failed",
    "record_run_discarded",
]
