"""Agent memory hooks cache through RedisWorkingMemory's get/set/delete.

Nine agents' memory hooks called ``working_memory.set(...)`` (and drift_monitor
also ``get`` / ``delete``) on a ``RedisWorkingMemory`` that had none of those
methods. Every call raised AttributeError, the hooks swallowed it as a warning,
and no working-memory cache was ever written. A live drift_monitor run on
2026-09-19 logged ``'RedisWorkingMemory' object has no attribute 'set'``. The
hooks' existing tests replaced ``working_memory`` with a mock, so they could not
see it.

These tests drive the REAL hook code through the REAL ``RedisWorkingMemory``;
only the Redis client underneath is faked. The fake behaves like redis-py with
``decode_responses=True`` (the production client, see ``get_redis_client``):
it rejects a dict value, returns ``str``, and records the TTL.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Optional

import pytest

from src.memory.working_memory import RedisWorkingMemory


class _RedisDataError(Exception):
    """Mirrors redis.exceptions.DataError for a non-scalar value."""


class _FakeRedis:
    def __init__(self) -> None:
        self.values: Dict[str, str] = {}
        self.ttls: Dict[str, Optional[int]] = {}

    async def set(self, key: str, value: Any, ex: Optional[int] = None, **_: Any) -> bool:
        if not isinstance(value, (str, bytes, int, float)):
            raise _RedisDataError(f"Invalid input of type: '{type(value).__name__}'")
        self.values[key] = value.decode() if isinstance(value, bytes) else str(value)
        self.ttls[key] = ex
        return True

    async def get(self, key: str) -> Optional[str]:
        return self.values.get(key)

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += self.values.pop(key, None) is not None
            self.ttls.pop(key, None)
        return removed


@pytest.fixture
def fake_redis() -> _FakeRedis:
    return _FakeRedis()


@pytest.fixture
def working_memory(fake_redis: _FakeRedis) -> RedisWorkingMemory:
    wm = RedisWorkingMemory()
    wm._client = fake_redis
    return wm


def _hooks(module_path: str, class_name: str, working_memory: RedisWorkingMemory) -> Any:
    hooks = getattr(importlib.import_module(module_path), class_name)()
    hooks._working_memory = working_memory
    return hooks


ML_HOOKS = [
    (
        "src.agents.ml_foundation.data_preparer.memory_hooks",
        "DataPreparerMemoryHooks",
        "cache_qc_report",
        "data_preparer:qc_report:",
    ),
    (
        "src.agents.ml_foundation.observability_connector.memory_hooks",
        "ObservabilityConnectorMemoryHooks",
        "cache_metrics",
        "observability_connector:metrics:",
    ),
    (
        "src.agents.ml_foundation.model_selector.memory_hooks",
        "ModelSelectorMemoryHooks",
        "cache_model_selection",
        "model_selector:selection:",
    ),
    (
        "src.agents.ml_foundation.feature_analyzer.memory_hooks",
        "FeatureAnalyzerMemoryHooks",
        "cache_feature_analysis",
        "feature_analyzer:analysis:",
    ),
    (
        "src.agents.ml_foundation.model_deployer.memory_hooks",
        "ModelDeployerMemoryHooks",
        "cache_deployment_manifest",
        "model_deployer:manifest:",
    ),
    (
        "src.agents.ml_foundation.model_trainer.memory_hooks",
        "ModelTrainerMemoryHooks",
        "cache_training_result",
        "model_trainer:result:",
    ),
    (
        "src.agents.ml_foundation.scope_definer.memory_hooks",
        "ScopeDefinerMemoryHooks",
        "cache_scope_definition",
        "scope_definer:result:",
    ),
    (
        "src.agents.cohort_constructor.memory_hooks",
        "CohortConstructorMemoryHooks",
        "cache_cohort_config",
        "cohort_constructor:config:",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("module_path,class_name,method,key_prefix", ML_HOOKS)
async def test_hook_cache_is_written_with_its_ttl(
    module_path, class_name, method, key_prefix, working_memory, fake_redis
) -> None:
    hooks = _hooks(module_path, class_name, working_memory)
    payload = {"metric": 0.87, "status": "ok"}

    assert await getattr(hooks, method)("sess-1", payload) is True

    [key] = [k for k in fake_redis.values if k.startswith(key_prefix)]
    assert key.endswith("sess-1")
    assert json.loads(fake_redis.values[key]) == payload  # stored once, not double-encoded
    assert fake_redis.ttls[key] == hooks.CACHE_TTL_SECONDS


@pytest.mark.asyncio
async def test_drift_monitor_cache_round_trips_and_invalidates(working_memory, fake_redis) -> None:
    hooks = _hooks(
        "src.agents.drift_monitor.memory_hooks", "DriftMonitorMemoryHooks", working_memory
    )
    result = {"overall_drift_score": 0.4, "features_with_drift": ["f1"]}

    assert await hooks.cache_drift_result("sess-2", result, features=["f1", "f2"], model_id="m1")

    assert await hooks.get_cached_drift_result("sess-2") == result
    feature = json.loads(fake_redis.values["drift_monitor:feature:f1:model:m1"])
    assert feature["has_drift"] is True
    assert fake_redis.ttls["drift_monitor:session:sess-2"] == hooks.CACHE_TTL_SECONDS

    assert await hooks.invalidate_cache(session_id="sess-2", feature="f1", model_id="m1")
    assert "drift_monitor:session:sess-2" not in fake_redis.values
    assert "drift_monitor:feature:f1:model:m1" not in fake_redis.values
    assert await hooks.get_cached_drift_result("sess-2") is None


@pytest.mark.asyncio
async def test_kv_set_serialises_non_strings_and_keeps_strings_verbatim(
    working_memory, fake_redis
) -> None:
    await working_memory.set("k:dict", {"a": 1}, ttl=30)
    await working_memory.set("k:str", '{"b": 2}', ex=60)
    await working_memory.set("k:default", "plain")

    assert json.loads(fake_redis.values["k:dict"]) == {"a": 1}
    assert fake_redis.values["k:str"] == '{"b": 2}'
    assert (fake_redis.ttls["k:dict"], fake_redis.ttls["k:str"]) == (30, 60)
    assert fake_redis.ttls["k:default"] == working_memory.ttl_seconds

    assert await working_memory.get("k:dict") == {"a": 1}
    assert await working_memory.get("k:default") == "plain"  # non-JSON string comes back as-is
    assert await working_memory.get("k:missing") is None
    assert await working_memory.delete("k:dict") == 1
