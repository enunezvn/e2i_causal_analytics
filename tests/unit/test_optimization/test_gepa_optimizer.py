"""Unit tests for GEPA optimizer setup and utilities.

Tests the GEPA optimizer factory, versioning, and A/B testing infrastructure:
- create_gepa_optimizer factory function
- save_optimized_module / load_optimized_module versioning
- GEPAABTest class

Version: 4.3
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Mark all tests in this module to run on the same worker (DSPy import safety)
pytestmark = pytest.mark.xdist_group(name="gepa_optimizer")


class TestGEPAOptimizerImports:
    """Test that GEPA optimizer imports work correctly."""

    def test_optimizer_setup_import(self):
        """Test optimizer setup module import."""
        from src.optimization.gepa.optimizer_setup import (
            create_gepa_optimizer,
            create_optimizer_for_agent,
        )

        assert create_gepa_optimizer is not None
        assert create_optimizer_for_agent is not None

    def test_versioning_import(self):
        """Test versioning module import."""
        from src.optimization.gepa.versioning import (
            load_optimized_module,
            rollback_to_version,
            save_optimized_module,
        )

        assert save_optimized_module is not None
        assert load_optimized_module is not None
        assert rollback_to_version is not None

    def test_ab_test_import(self):
        """Test A/B test module import."""
        from src.optimization.gepa.ab_test import GEPAABTest

        assert GEPAABTest is not None

    def test_package_init_exports(self):
        """Test main package exports."""
        from src.optimization.gepa import (
            create_gepa_optimizer,
            create_optimizer_for_agent,
            get_metric_for_agent,
            load_optimized_module,
            save_optimized_module,
        )

        assert create_gepa_optimizer is not None
        assert create_optimizer_for_agent is not None
        assert get_metric_for_agent is not None
        assert save_optimized_module is not None
        assert load_optimized_module is not None


class TestCreateGEPAOptimizer:
    """Test create_gepa_optimizer factory function."""

    @pytest.fixture
    def mock_metric(self):
        """Create a mock metric function."""
        from src.optimization.gepa.metrics.standard_agent_metric import (
            StandardAgentGEPAMetric,
        )

        return StandardAgentGEPAMetric()

    @pytest.fixture
    def sample_trainset(self) -> List[Dict[str, Any]]:
        """Create sample training data."""
        return [
            {"question": f"Query {i}", "context": {}, "ground_truth": {"score": 0.8}}
            for i in range(10)
        ]

    @pytest.fixture
    def sample_valset(self) -> List[Dict[str, Any]]:
        """Create sample validation data."""
        return [
            {"question": f"Val Query {i}", "context": {}, "ground_truth": {"score": 0.9}}
            for i in range(5)
        ]

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_with_light_budget(
        self, mock_lm_class, mock_gepa_class, mock_metric, sample_trainset, sample_valset
    ):
        """Test creating optimizer with light budget preset."""
        from src.optimization.gepa.optimizer_setup import create_gepa_optimizer

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_gepa_optimizer(
            metric=mock_metric,
            trainset=sample_trainset,
            valset=sample_valset,
            auto="light",
        )

        assert optimizer is not None
        mock_gepa_class.assert_called_once()
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "light"

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_with_medium_budget(
        self, mock_lm_class, mock_gepa_class, mock_metric, sample_trainset, sample_valset
    ):
        """Test creating optimizer with medium budget preset."""
        from src.optimization.gepa.optimizer_setup import create_gepa_optimizer

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_gepa_optimizer(
            metric=mock_metric,
            trainset=sample_trainset,
            valset=sample_valset,
            auto="medium",
        )

        assert optimizer is not None
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "medium"

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_with_heavy_budget(
        self, mock_lm_class, mock_gepa_class, mock_metric, sample_trainset, sample_valset
    ):
        """Test creating optimizer with heavy budget preset."""
        from src.optimization.gepa.optimizer_setup import create_gepa_optimizer

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_gepa_optimizer(
            metric=mock_metric,
            trainset=sample_trainset,
            valset=sample_valset,
            auto="heavy",
        )

        assert optimizer is not None
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "heavy"

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_with_tool_optimization(
        self, mock_lm_class, mock_gepa_class, mock_metric, sample_trainset, sample_valset
    ):
        """Test creating optimizer with tool optimization enabled.

        Note: enable_tool_optimization is accepted by create_gepa_optimizer
        but not passed to GEPA as GEPA doesn't support it yet (reserved for future).
        """
        from src.optimization.gepa.optimizer_setup import create_gepa_optimizer

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_gepa_optimizer(
            metric=mock_metric,
            trainset=sample_trainset,
            valset=sample_valset,
            auto="medium",
            enable_tool_optimization=True,
        )

        assert optimizer is not None
        # Verify optimizer was created (enable_tool_optimization is reserved for future)
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "medium"

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_with_seed(
        self, mock_lm_class, mock_gepa_class, mock_metric, sample_trainset, sample_valset
    ):
        """Test creating optimizer with specific seed."""
        from src.optimization.gepa.optimizer_setup import create_gepa_optimizer

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_gepa_optimizer(
            metric=mock_metric,
            trainset=sample_trainset,
            valset=sample_valset,
            auto="light",
            seed=123,
        )

        assert optimizer is not None
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["seed"] == 123


class TestCreateOptimizerForAgent:
    """Test create_optimizer_for_agent convenience function."""

    @pytest.fixture
    def sample_trainset(self) -> List[Dict[str, Any]]:
        """Create sample training data."""
        return [{"question": f"Query {i}", "context": {}, "ground_truth": {}} for i in range(10)]

    @pytest.fixture
    def sample_valset(self) -> List[Dict[str, Any]]:
        """Create sample validation data."""
        return [{"question": f"Val Query {i}", "context": {}, "ground_truth": {}} for i in range(5)]

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_for_causal_impact(
        self, mock_lm_class, mock_gepa_class, sample_trainset, sample_valset
    ):
        """Test creating optimizer for causal_impact agent."""
        from src.optimization.gepa.optimizer_setup import create_optimizer_for_agent

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_optimizer_for_agent(
            agent_name="causal_impact",
            trainset=sample_trainset,
            valset=sample_valset,
        )

        assert optimizer is not None
        # causal_impact should get medium budget
        # Note: enable_tool_optimization is reserved for future (not passed to GEPA)
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "medium"

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_for_experiment_designer(
        self, mock_lm_class, mock_gepa_class, sample_trainset, sample_valset
    ):
        """Test creating optimizer for experiment_designer agent."""
        from src.optimization.gepa.optimizer_setup import create_optimizer_for_agent

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_optimizer_for_agent(
            agent_name="experiment_designer",
            trainset=sample_trainset,
            valset=sample_valset,
        )

        assert optimizer is not None
        # experiment_designer should get medium budget
        # Note: enable_tool_optimization is reserved for future (not passed to GEPA)
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "medium"

    @patch("dspy.GEPA")
    @patch("dspy.LM")
    def test_create_optimizer_for_feedback_learner(
        self, mock_lm_class, mock_gepa_class, sample_trainset, sample_valset
    ):
        """Test creating optimizer for feedback_learner agent."""
        from src.optimization.gepa.optimizer_setup import create_optimizer_for_agent

        mock_gepa_instance = MagicMock()
        mock_gepa_class.return_value = mock_gepa_instance

        optimizer = create_optimizer_for_agent(
            agent_name="feedback_learner",
            trainset=sample_trainset,
            valset=sample_valset,
        )

        assert optimizer is not None
        # feedback_learner should get heavy budget (Tier 5 Deep)
        # Note: enable_tool_optimization is reserved for future (not passed to GEPA)
        call_kwargs = mock_gepa_class.call_args[1]
        assert call_kwargs["auto"] == "heavy"


class TestVersioning:
    """Test GEPA module versioning utilities."""

    @pytest.fixture
    def mock_module(self):
        """Create a mock optimized module."""
        module = MagicMock()
        module.dump_state = MagicMock(return_value={"prompts": ["test prompt"]})
        module.predictors = MagicMock(return_value=[])
        return module

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_optimized_module(self, mock_module, temp_output_dir):
        """Test saving an optimized module."""
        from src.optimization.gepa.versioning import save_optimized_module

        result = save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            output_dir=temp_output_dir,
            metadata={"score": 0.85},
        )

        assert "path" in result
        assert "version_id" in result
        assert "instruction_hash" in result
        assert Path(result["path"]).exists()

    def test_save_optimized_module_with_custom_version(self, mock_module, temp_output_dir):
        """Test saving with custom version ID."""
        from src.optimization.gepa.versioning import save_optimized_module

        result = save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            version_id="gepa_v1_test_custom",
            output_dir=temp_output_dir,
        )

        assert result["version_id"] == "gepa_v1_test_custom"

    def test_load_optimized_module(self, mock_module, temp_output_dir):
        """Test loading an optimized module."""
        from src.optimization.gepa.versioning import (
            load_optimized_module,
            save_optimized_module,
        )

        # First save a module
        save_result = save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            output_dir=temp_output_dir,
        )

        # Create a mock module class
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        # Load it back
        loaded_module, metadata = load_optimized_module(
            module_cls=mock_cls,
            agent_name="test_agent",
            version_id=save_result["version_id"],
            input_dir=temp_output_dir,
        )

        assert loaded_module is not None
        assert metadata["version_id"] == save_result["version_id"]

    def test_load_latest_version(self, mock_module, temp_output_dir):
        """Test loading the latest version when version_id is None."""
        from src.optimization.gepa.versioning import (
            load_optimized_module,
            save_optimized_module,
        )

        # Save two modules
        save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            version_id="gepa_v1_test_agent_20251201_100000",
            output_dir=temp_output_dir,
        )
        save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            version_id="gepa_v1_test_agent_20251202_100000",
            output_dir=temp_output_dir,
        )

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()

        # Load latest (no version_id)
        loaded_module, metadata = load_optimized_module(
            module_cls=mock_cls,
            agent_name="test_agent",
            input_dir=temp_output_dir,
        )

        # Should load the later version (sorted by filename)
        assert loaded_module is not None

    def test_load_latest_resolves_v10_over_v9(self, mock_module, temp_output_dir):
        """#1496: newest-version resolution must sort the gepa_v<n> suffix numerically.

        Lexicographic name order inverts at v10 ("gepa_v10..." < "gepa_v2...", and
        "gepa_v9..." greatest of all), so after the 10th save a name sort resolves
        a stale artifact as "newest" forever.
        """
        from src.optimization.gepa.versioning import (
            load_optimized_module,
            save_optimized_module,
        )

        for n in range(1, 11):
            save_optimized_module(
                module=mock_module,
                agent_name="test_agent",
                version_id=f"gepa_v{n}_test_agent_202512{n:02d}_100000",
                output_dir=temp_output_dir,
            )

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        _, metadata = load_optimized_module(
            module_cls=mock_cls,
            agent_name="test_agent",
            input_dir=temp_output_dir,
        )

        assert metadata["version_id"] == "gepa_v10_test_agent_20251210_100000"

    def test_load_latest_equal_versions_fall_back_to_save_time_order(
        self, mock_module, temp_output_dir
    ):
        """generate_version_id emits v1 for every save today; the tie-break must
        keep resolving equal version numbers by the zero-padded timestamp in the
        name, exactly as the pre-#1496 name sort did."""
        from src.optimization.gepa.versioning import (
            load_optimized_module,
            save_optimized_module,
        )

        for ts in ("20251201_100000", "20251202_100000"):
            save_optimized_module(
                module=mock_module,
                agent_name="test_agent",
                version_id=f"gepa_v1_test_agent_{ts}",
                output_dir=temp_output_dir,
            )

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        _, metadata = load_optimized_module(
            module_cls=mock_cls,
            agent_name="test_agent",
            input_dir=temp_output_dir,
        )

        assert metadata["version_id"] == "gepa_v1_test_agent_20251202_100000"

    def test_load_latest_never_resolves_a_name_without_a_version(
        self, mock_module, temp_output_dir
    ):
        """A gepa_*.json name the saver could not have produced (no parseable
        gepa_v<n> prefix — e.g. a hand-made copy) must not win "newest", even
        when it lexicographically outranks every real version."""
        from src.optimization.gepa.versioning import (
            load_optimized_module,
            save_optimized_module,
        )

        save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            version_id="gepa_v2_test_agent_20251202_100000",
            output_dir=temp_output_dir,
        )
        # Lexicographically greater than any gepa_v* name; valid JSON so that a
        # wrong SELECTION fails the assert below rather than erroring earlier.
        straggler = Path(temp_output_dir) / "test_agent" / "gepa_zzz_manual_copy.json"
        straggler.write_text(
            json.dumps(
                {
                    "version_id": "gepa_zzz_manual_copy",
                    "created_at": "2025-12-03T10:00:00",
                    "instruction_hash": "0" * 64,
                    "module_state": {},
                    "metadata": {},
                }
            )
        )

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        _, metadata = load_optimized_module(
            module_cls=mock_cls,
            agent_name="test_agent",
            input_dir=temp_output_dir,
        )

        assert metadata["version_id"] == "gepa_v2_test_agent_20251202_100000"

    def test_load_latest_all_versionless_names_is_deterministic_not_a_crash(self, temp_output_dir):
        """With ONLY unversioned names on disk the loader must still resolve
        deterministically (name order among themselves) instead of raising."""
        from src.optimization.gepa.versioning import load_optimized_module

        agent_dir = Path(temp_output_dir) / "test_agent"
        agent_dir.mkdir(parents=True)
        for name in ("gepa_alpha.json", "gepa_beta.json"):
            (agent_dir / name).write_text(
                json.dumps(
                    {
                        "version_id": name.removesuffix(".json"),
                        "created_at": "2025-12-01T10:00:00",
                        "instruction_hash": "0" * 64,
                        "module_state": {},
                        "metadata": {},
                    }
                )
            )

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        _, metadata = load_optimized_module(
            module_cls=mock_cls,
            agent_name="test_agent",
            input_dir=temp_output_dir,
        )

        assert metadata["version_id"] == "gepa_beta"

    def test_artifact_sort_key_orders_versions_numerically(self):
        """The shared ordering key both resolution sites use (#1496)."""
        from src.optimization.gepa.versioning import gepa_artifact_sort_key

        names = [Path(f"gepa_v{n}_agent_20251201_100000.json") for n in (10, 2, 1, 11, 9)]
        ordered = [p.name.split("_")[1] for p in sorted(names, key=gepa_artifact_sort_key)]
        assert ordered == ["v1", "v2", "v9", "v10", "v11"]

    def test_rollback_to_version(self, mock_module, temp_output_dir):
        """Test rolling back to a previous version."""
        from src.optimization.gepa.versioning import (
            rollback_to_version,
            save_optimized_module,
        )

        # Save a module
        save_result = save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            output_dir=temp_output_dir,
        )

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()

        # Rollback is just load with specific version
        module, metadata = rollback_to_version(
            module_cls=mock_cls,
            agent_name="test_agent",
            version_id=save_result["version_id"],
            input_dir=temp_output_dir,
        )

        assert module is not None
        assert metadata["version_id"] == save_result["version_id"]

    def test_list_versions(self, mock_module, temp_output_dir):
        """Test listing all versions."""
        from src.optimization.gepa.versioning import list_versions, save_optimized_module

        # Save two versions
        save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            version_id="gepa_v1_test_agent_20251201_100000",
            output_dir=temp_output_dir,
        )
        save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            version_id="gepa_v1_test_agent_20251202_100000",
            output_dir=temp_output_dir,
        )

        versions = list_versions(agent_name="test_agent", input_dir=temp_output_dir)

        assert len(versions) == 2
        assert all("version_id" in v for v in versions)

    # -- #1500: auto-generated version ids must increment per save -----------

    def test_generate_version_id_takes_explicit_version(self):
        """#1500: the v<n> component is a parameter, defaulting to 1.

        The default pins the pure-function contract that
        scripts/gepa_integration_test.py asserts (``gepa_v1_`` with no
        directory in sight): with no prior artifacts the first version IS 1.
        """
        from src.optimization.gepa.versioning import generate_version_id

        assert generate_version_id("test_agent").startswith("gepa_v1_test_agent_")
        assert generate_version_id("test_agent", version=7).startswith("gepa_v7_test_agent_")

    def test_next_artifact_version_scans_directory(self, tmp_path):
        """#1500: next version = max parsed gepa_v<n> on disk + 1.

        A missing directory, an empty one, or one holding only names without a
        parseable version (which gepa_artifact_sort_key ranks as -1) all mean
        the next save is the first real version: 1.
        """
        from src.optimization.gepa.versioning import next_artifact_version

        assert next_artifact_version(tmp_path / "never_saved") == 1

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        assert next_artifact_version(agent_dir) == 1

        (agent_dir / "gepa_zzz_manual_copy.json").write_text("{}")
        assert next_artifact_version(agent_dir) == 1

        (agent_dir / "gepa_v3_agent_20251201_100000.json").write_text("{}")
        assert next_artifact_version(agent_dir) == 4

    def test_auto_version_increments_across_saves(self, mock_module, temp_output_dir):
        """#1500: generate_version_id hardcoded ``gepa_v1_`` for every save.

        Auto-named saves must mint v1 then v2 — under the hardcoded prefix two
        saves in the same second even collided on one filename, silently
        overwriting the first artifact. Latest-resolution must pick the v2.
        """
        from src.optimization.gepa.versioning import (
            load_optimized_module,
            save_optimized_module,
        )

        first = save_optimized_module(
            module=mock_module, agent_name="test_agent", output_dir=temp_output_dir
        )
        second = save_optimized_module(
            module=mock_module, agent_name="test_agent", output_dir=temp_output_dir
        )

        assert first["version_id"].startswith("gepa_v1_test_agent_")
        assert second["version_id"].startswith("gepa_v2_test_agent_")
        agent_dir = Path(temp_output_dir) / "test_agent"
        assert len(list(agent_dir.glob("gepa_*.json"))) == 2

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        _, metadata = load_optimized_module(
            module_cls=mock_cls, agent_name="test_agent", input_dir=temp_output_dir
        )
        assert metadata["version_id"] == second["version_id"]

    def test_auto_version_outranks_legacy_v1_with_later_timestamp(
        self, mock_module, temp_output_dir
    ):
        """#1500 backward compat with hardcoded-era artifacts already on disk.

        Every pre-#1500 artifact is named ``gepa_v1_...``. A new auto save must
        scan them, mint v2, and win "newest" even against a legacy v1 whose
        embedded timestamp is LATER — the numeric version outranks the
        timestamp tie-break in gepa_artifact_sort_key (#1496).
        """
        from src.optimization.gepa.versioning import (
            newest_saved_artifact,
            save_optimized_module,
        )

        # A legacy artifact exactly as the hardcoded-era saver named it, with a
        # far-future timestamp in the name.
        save_optimized_module(
            module=mock_module,
            agent_name="test_agent",
            version_id="gepa_v1_test_agent_21001231_235959",
            output_dir=temp_output_dir,
        )

        info = save_optimized_module(
            module=mock_module, agent_name="test_agent", output_dir=temp_output_dir
        )

        assert info["version_id"].startswith("gepa_v2_test_agent_")
        newest = newest_saved_artifact(Path(temp_output_dir) / "test_agent")
        assert newest is not None
        assert newest.name == f"{info['version_id']}.json"

    def test_auto_save_never_overwrites_a_colliding_artifact(
        self, mock_module, temp_output_dir, monkeypatch
    ):
        """#1500 iter-2: scan-then-write is a TOCTOU window, not a lock.

        Two concurrent saves for one agent can both observe the same on-disk
        max version, mint the identical ``gepa_v{n}_{agent}_{ts}`` name in the
        same second, and the second plain ``open(..., "w")`` clobbers the
        first — the original silent-overwrite mode, just narrowed.

        Deterministic reproduction of that race (no flaky threads): freeze the
        timestamp and pin the directory scan to a stale version, so this save
        mints exactly the name a "concurrent" saver already wrote. The save
        must leave that artifact untouched and land on the next version.
        """
        from src.optimization.gepa import versioning

        frozen = datetime(2025, 12, 1, 10, 0, 0)

        class _FrozenDatetime:
            @staticmethod
            def now() -> datetime:
                return frozen

        monkeypatch.setattr(versioning, "datetime", _FrozenDatetime)
        # The stale scan: the concurrent saver's v1 already exists on disk,
        # but this saver's scan predates it and still reports 1 as next.
        monkeypatch.setattr(versioning, "next_artifact_version", lambda directory: 1)

        agent_dir = Path(temp_output_dir) / "test_agent"
        agent_dir.mkdir(parents=True)
        collider = agent_dir / "gepa_v1_test_agent_20251201_100000.json"
        sentinel = json.dumps(
            {
                "version_id": "gepa_v1_test_agent_20251201_100000",
                "created_at": "2025-12-01T10:00:00",
                "instruction_hash": "0" * 64,
                "module_state": {},
                "metadata": {"writer": "concurrent-saver"},
            }
        )
        collider.write_text(sentinel)

        info = versioning.save_optimized_module(
            module=mock_module, agent_name="test_agent", output_dir=temp_output_dir
        )

        # The concurrent saver's artifact survives byte-for-byte...
        assert collider.read_text() == sentinel
        # ...and this save advanced past the occupied name instead.
        assert info["version_id"] == "gepa_v2_test_agent_20251201_100000"
        assert (agent_dir / "gepa_v2_test_agent_20251201_100000.json").exists()
        assert len(list(agent_dir.glob("gepa_*.json"))) == 2

    def test_failed_auto_save_leaves_no_partial_artifact(self, mock_module, temp_output_dir):
        """#1500 iter-3: a dump failure must not leave the reserved file behind.

        json.dump streams into the exclusively-created file, so an exception
        mid-serialization (a value str() cannot render past ``default=str``,
        disk full) used to flush-and-close a PARTIAL file — and because
        newest_saved_artifact resolves by filename alone, that corrupt
        artifact became the permanent "newest": a direct
        load_optimized_module(version_id=None) raises JSONDecodeError, and the
        cognitive-RAG fail-soft wrapper silently pins the base prompt on every
        retry. The reservation must be unlinked and the original exception
        re-raised.
        """
        from src.optimization.gepa.versioning import (
            newest_saved_artifact,
            save_optimized_module,
        )

        class _ExplodesOnStr:
            def __str__(self) -> str:
                raise RuntimeError("mid-dump serialization failure")

        # A good artifact that must still be "newest" after the failed save.
        good = save_optimized_module(
            module=mock_module, agent_name="test_agent", output_dir=temp_output_dir
        )
        agent_dir = Path(temp_output_dir) / "test_agent"

        with pytest.raises(RuntimeError, match="mid-dump serialization failure"):
            save_optimized_module(
                module=mock_module,
                agent_name="test_agent",
                output_dir=temp_output_dir,
                metadata={"bad": _ExplodesOnStr()},
            )

        # The failed save's reservation is gone: only the good artifact
        # remains, and it is still what "newest" resolves to.
        assert [p.name for p in agent_dir.glob("gepa_*.json")] == [f"{good['version_id']}.json"]
        newest = newest_saved_artifact(agent_dir)
        assert newest is not None
        assert newest.name == f"{good['version_id']}.json"


class TestGEPAABTest:
    """Test GEPAABTest class for A/B testing optimized modules."""

    @pytest.fixture
    def ab_test_instance(self):
        """Create GEPAABTest instance."""
        from src.optimization.gepa.ab_test import GEPAABTest

        ab_test = GEPAABTest(
            test_name="test_experiment",
            agent_name="causal_impact",
            traffic_split=0.5,
        )
        ab_test.start()  # Start the test to allow variant assignment
        return ab_test

    def test_ab_test_initialization(self):
        """Test A/B test initializes correctly."""
        from src.optimization.gepa.ab_test import GEPAABTest

        ab_test = GEPAABTest(
            test_name="test_experiment",
            agent_name="causal_impact",
            traffic_split=0.5,
        )

        assert ab_test.test_name == "test_experiment"
        assert ab_test.agent_name == "causal_impact"
        assert ab_test.traffic_split == 0.5
        assert ab_test.status == "draft"

    def test_ab_test_start_stop(self):
        """Test A/B test start and stop."""
        from src.optimization.gepa.ab_test import GEPAABTest

        ab_test = GEPAABTest(
            test_name="test_experiment",
            agent_name="causal_impact",
        )

        assert ab_test.status == "draft"

        ab_test.start()
        assert ab_test.status == "running"
        assert ab_test.started_at is not None

        ab_test.stop()
        assert ab_test.status == "stopped"
        assert ab_test.ended_at is not None

    def test_ab_test_assign_variant(self, ab_test_instance):
        """Test A/B test variant assignment."""
        # Should return either baseline or gepa
        variant = ab_test_instance.assign_variant(user_id="user_123")

        assert variant in ["baseline", "gepa"]

    def test_ab_test_consistent_variant_for_user(self, ab_test_instance):
        """Test that same user gets same variant consistently."""
        user_id = "consistent_user_456"

        variant1 = ab_test_instance.assign_variant(user_id=user_id)
        variant2 = ab_test_instance.assign_variant(user_id=user_id)

        assert variant1 == variant2

    def test_ab_test_returns_baseline_when_not_running(self):
        """Test that non-running test returns baseline."""
        from src.optimization.gepa.ab_test import GEPAABTest

        ab_test = GEPAABTest(
            test_name="test_experiment",
            agent_name="causal_impact",
            traffic_split=0.9,  # High split, should get gepa if running
        )

        # Not started, should always return baseline
        variant = ab_test.assign_variant(user_id="any_user")
        assert variant == "baseline"

    def test_ab_test_record_observation(self, ab_test_instance):
        """Test recording experiment observations."""
        observation = ab_test_instance.record_observation(
            request_id="req_123",
            variant="baseline",
            score=0.9,
            latency_ms=150,
            user_id="user_456",
        )

        assert observation is not None
        assert observation.request_id == "req_123"
        assert observation.variant == "baseline"
        assert observation.score == 0.9
        assert observation.latency_ms == 150

    def test_ab_test_analyze(self, ab_test_instance):
        """Test A/B test analysis."""
        # Record enough observations for both variants
        for i in range(35):
            ab_test_instance.record_observation(
                request_id=f"req_baseline_{i}",
                variant="baseline",
                score=0.75 + (i % 5) * 0.01,
                latency_ms=100 + i,
            )
            ab_test_instance.record_observation(
                request_id=f"req_gepa_{i}",
                variant="gepa",
                score=0.80 + (i % 5) * 0.01,
                latency_ms=95 + i,
            )

        results = ab_test_instance.analyze()

        assert results is not None
        assert results.baseline_requests == 35
        assert results.treatment_requests == 35
        assert results.baseline_score_avg is not None
        assert results.treatment_score_avg is not None

    def test_ab_test_to_dict(self, ab_test_instance):
        """Test A/B test serialization."""
        data = ab_test_instance.to_dict()

        assert data["test_name"] == "test_experiment"
        assert data["agent_name"] == "causal_impact"
        assert data["traffic_split"] == 0.5
        assert "test_id" in data


class TestGEPABudgetPresets:
    """Test GEPA budget preset configurations."""

    def test_budget_presets_exist(self):
        """Test that budget presets are defined."""
        from src.optimization.gepa.optimizer_setup import BUDGET_PRESETS

        assert "light" in BUDGET_PRESETS
        assert "medium" in BUDGET_PRESETS
        assert "heavy" in BUDGET_PRESETS

    def test_light_budget_configuration(self):
        """Test light budget has appropriate settings."""
        from src.optimization.gepa.optimizer_setup import BUDGET_PRESETS

        light = BUDGET_PRESETS.get("light", {})

        # Light should have max_metric_calls defined
        assert "max_metric_calls" in light
        assert light["max_metric_calls"] <= 1000  # Light budget should be limited
        assert "description" in light

    def test_medium_budget_configuration(self):
        """Test medium budget has appropriate settings."""
        from src.optimization.gepa.optimizer_setup import BUDGET_PRESETS

        medium = BUDGET_PRESETS.get("medium", {})

        # Medium should be balanced
        assert "max_metric_calls" in medium
        assert medium["max_metric_calls"] > BUDGET_PRESETS["light"]["max_metric_calls"]
        assert "description" in medium

    def test_heavy_budget_configuration(self):
        """Test heavy budget has appropriate settings."""
        from src.optimization.gepa.optimizer_setup import BUDGET_PRESETS

        heavy = BUDGET_PRESETS.get("heavy", {})

        # Heavy should have more iterations
        assert "max_metric_calls" in heavy
        assert heavy["max_metric_calls"] > BUDGET_PRESETS["medium"]["max_metric_calls"]
        assert "description" in heavy

    def test_agent_budget_mapping(self):
        """Test agent to budget mapping."""
        from src.optimization.gepa.optimizer_setup import AGENT_BUDGETS

        # Hybrid agents should get medium budget
        assert AGENT_BUDGETS.get("causal_impact") == "medium"
        assert AGENT_BUDGETS.get("experiment_designer") == "medium"

        # Deep agents should get heavy budget
        assert AGENT_BUDGETS.get("feedback_learner") == "heavy"
        assert AGENT_BUDGETS.get("explainer") == "heavy"


class TestToolOptimization:
    """Test GEPA tool optimization for DoWhy/EconML tools."""

    def test_causal_tools_import(self):
        """Test causal tools can be imported."""
        from src.optimization.gepa.tools.causal_tools import CAUSAL_TOOLS

        assert CAUSAL_TOOLS is not None
        assert isinstance(CAUSAL_TOOLS, list)

    def test_causal_tools_structure(self):
        """Test causal tools have required fields."""
        from src.optimization.gepa.tools.causal_tools import CAUSAL_TOOLS

        for tool in CAUSAL_TOOLS:
            assert hasattr(tool, "name"), f"Tool missing name: {tool}"
            assert hasattr(tool, "description"), f"Tool missing description: {tool}"

    def test_expected_causal_tools_present(self):
        """Test expected DoWhy/EconML tools are defined."""
        from src.optimization.gepa.tools.causal_tools import CAUSAL_TOOLS

        tool_names = [t.name for t in CAUSAL_TOOLS]

        # Key causal inference tools should be present
        expected_tools = [
            "linear_dml",
            "causal_forest",
            "double_robust_learner",
        ]

        for expected in expected_tools:
            # Check if tool exists (case-insensitive)
            assert any(expected.lower() in name.lower() for name in tool_names), (
                f"Missing tool: {expected}"
            )
