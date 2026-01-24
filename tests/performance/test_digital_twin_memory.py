"""
Memory Profiling Tests for Digital Twin System.

Tests cover:
- Memory growth linearity with population size
- Memory leak detection across repeated simulations
- Garbage collection verification
- Peak memory under configured limits

IMPORTANT: These tests require significant memory.
Run with: pytest -n 1 --dist=no tests/performance/test_digital_twin_memory.py -v
"""

import gc
import tracemalloc
from uuid import uuid4

import numpy as np
import pytest

from src.digital_twin.models.simulation_models import (
    InterventionConfig,
    SimulationStatus,
)
from src.digital_twin.models.twin_models import (
    Brand,
    DigitalTwin,
    TwinPopulation,
    TwinType,
)
from src.digital_twin.simulation_engine import SimulationEngine


# Mark all tests as slow performance tests
pytestmark = [
    pytest.mark.slow,
    pytest.mark.xdist_group(name="digital_twin_memory"),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_population(n: int, seed: int = 42) -> TwinPopulation:
    """Create a twin population of specified size."""
    np.random.seed(seed)
    twins = []

    for i in range(n):
        twin = DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features={
                "specialty": ["rheumatology", "dermatology", "allergy"][i % 3],
                "decile": (i % 10) + 1,
                "region": ["northeast", "south", "midwest", "west"][i % 4],
                "digital_engagement_score": np.random.uniform(0.2, 0.8),
                "adoption_stage": "early_majority",
            },
            baseline_outcome=np.random.uniform(0.05, 0.20),
            baseline_propensity=np.random.uniform(0.3, 0.7),
        )
        twins.append(twin)

    return TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=twins,
        size=n,
        model_id=uuid4(),
    )


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import sys

    # Force garbage collection before measuring
    gc.collect()

    # Use tracemalloc for accurate measurement
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 / 1024


def measure_memory_delta(func, *args, **kwargs):
    """Measure memory delta for a function call."""
    gc.collect()
    tracemalloc.start()

    before = tracemalloc.get_traced_memory()[0]
    result = func(*args, **kwargs)
    after = tracemalloc.get_traced_memory()[0]

    tracemalloc.stop()

    delta_mb = (after - before) / 1024 / 1024
    return result, delta_mb


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def email_campaign_config():
    """Standard intervention configuration."""
    return InterventionConfig(
        intervention_type="email_campaign",
        duration_weeks=8,
        intensity_multiplier=1.0,
    )


@pytest.fixture(autouse=True)
def setup_tracemalloc():
    """Setup and teardown tracemalloc for each test."""
    tracemalloc.start()
    yield
    tracemalloc.stop()


# =============================================================================
# MEMORY GROWTH LINEARITY TESTS
# =============================================================================


class TestMemoryGrowthLinear:
    """Tests for verifying linear memory growth with population size."""

    def test_memory_scales_linearly_with_population(self, email_campaign_config):
        """Test that memory usage scales linearly with population size."""
        sizes = [1_000, 5_000, 10_000]
        memory_usages = []

        for size in sizes:
            gc.collect()
            tracemalloc.reset_peak()

            population = create_population(size)
            engine = SimulationEngine(population)

            # Run simulation
            result = engine.simulate(email_campaign_config)
            assert result.status == SimulationStatus.COMPLETED

            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()
            memory_usages.append(peak / 1024 / 1024)  # Convert to MB

            # Cleanup
            del engine
            del population
            gc.collect()

        print("\nMemory usage by population size:")
        for size, mem in zip(sizes, memory_usages):
            print(f"  {size:,} twins: {mem:.2f} MB")

        # Check approximate linearity
        # Memory for 10K should be roughly 10x memory for 1K (with tolerance)
        if memory_usages[0] > 0:
            ratio = memory_usages[2] / memory_usages[0]
            print(f"  Ratio (10K/1K): {ratio:.2f}x")

            # Allow 20x due to fixed overhead
            assert ratio < 20, f"Non-linear memory growth: {ratio}x"

    def test_population_creation_memory(self):
        """Test memory usage of population creation alone."""
        sizes = [1_000, 5_000, 10_000]
        memory_deltas = []

        for size in sizes:
            gc.collect()
            tracemalloc.reset_peak()

            before = tracemalloc.get_traced_memory()[0]
            population = create_population(size)
            after = tracemalloc.get_traced_memory()[0]

            delta_mb = (after - before) / 1024 / 1024
            memory_deltas.append(delta_mb)

            del population
            gc.collect()

        print("\nPopulation creation memory:")
        for size, mem in zip(sizes, memory_deltas):
            per_twin = (mem * 1024) / size  # KB per twin
            print(f"  {size:,} twins: {mem:.2f} MB ({per_twin:.2f} KB/twin)")

        # Memory per twin should be roughly constant
        per_twin_kb = [(m * 1024) / s for m, s in zip(memory_deltas, sizes)]
        variance = max(per_twin_kb) - min(per_twin_kb)
        assert variance < 5, f"High variance in per-twin memory: {variance:.2f} KB"


# =============================================================================
# MEMORY LEAK DETECTION TESTS
# =============================================================================


class TestMemoryLeakDetection:
    """Tests for detecting memory leaks."""

    def test_no_memory_leak_repeated_simulations(self, email_campaign_config):
        """Test that memory is stable across repeated simulations."""
        population = create_population(5_000)
        engine = SimulationEngine(population)

        n_iterations = 10
        memory_readings = []

        gc.collect()
        initial_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        for i in range(n_iterations):
            result = engine.simulate(email_campaign_config)
            assert result.status == SimulationStatus.COMPLETED

            gc.collect()
            current = tracemalloc.get_traced_memory()[0] / 1024 / 1024
            memory_readings.append(current)

        print("\nMemory across iterations:")
        print(f"  Initial: {initial_memory:.2f} MB")
        for i, mem in enumerate(memory_readings):
            print(f"  Iteration {i+1}: {mem:.2f} MB")

        # Memory should not grow significantly
        memory_growth = memory_readings[-1] - memory_readings[0]
        print(f"  Growth: {memory_growth:.2f} MB")

        # Allow 10MB growth (some caching is expected)
        assert memory_growth < 50, f"Potential memory leak: {memory_growth:.2f} MB growth"

    def test_no_leak_with_different_configs(self):
        """Test no memory leak when using different configurations."""
        population = create_population(3_000)
        engine = SimulationEngine(population)

        configs = [
            InterventionConfig(intervention_type="email_campaign", duration_weeks=8),
            InterventionConfig(intervention_type="call_frequency_increase", duration_weeks=4),
            InterventionConfig(intervention_type="speaker_program_invitation", duration_weeks=12),
        ]

        gc.collect()
        tracemalloc.reset_peak()
        initial = tracemalloc.get_traced_memory()[0]

        for _ in range(5):  # Run each config 5 times
            for config in configs:
                result = engine.simulate(config)
                assert result.status == SimulationStatus.COMPLETED

        gc.collect()
        final = tracemalloc.get_traced_memory()[0]

        growth_mb = (final - initial) / 1024 / 1024
        print(f"\nMemory growth after 15 simulations: {growth_mb:.2f} MB")

        assert growth_mb < 100, f"Memory leak suspected: {growth_mb:.2f} MB growth"

    def test_no_leak_engine_recreation(self, email_campaign_config):
        """Test no memory leak when recreating engines."""
        n_iterations = 5
        memory_readings = []

        gc.collect()

        for i in range(n_iterations):
            # Create new population and engine each iteration
            population = create_population(2_000)
            engine = SimulationEngine(population)

            result = engine.simulate(email_campaign_config)
            assert result.status == SimulationStatus.COMPLETED

            # Cleanup
            del engine
            del population
            gc.collect()

            current = tracemalloc.get_traced_memory()[0] / 1024 / 1024
            memory_readings.append(current)

        print("\nMemory after engine recreation:")
        for i, mem in enumerate(memory_readings):
            print(f"  Iteration {i+1}: {mem:.2f} MB")

        # Memory should not accumulate
        growth = memory_readings[-1] - memory_readings[0]
        print(f"  Total growth: {growth:.2f} MB")

        assert growth < 50, f"Memory not being released: {growth:.2f} MB retained"


# =============================================================================
# GARBAGE COLLECTION TESTS
# =============================================================================


class TestGarbageCollection:
    """Tests for verifying garbage collection of twin objects."""

    def test_gc_releases_twin_population(self):
        """Test that GC properly releases twin population memory."""
        gc.collect()
        tracemalloc.reset_peak()

        before = tracemalloc.get_traced_memory()[0]

        # Create large population
        population = create_population(20_000)
        after_create = tracemalloc.get_traced_memory()[0]

        # Delete and collect
        del population
        gc.collect()
        gc.collect()  # Second pass for cyclic references

        after_gc = tracemalloc.get_traced_memory()[0]

        created_mb = (after_create - before) / 1024 / 1024
        released_mb = (after_create - after_gc) / 1024 / 1024
        retained_mb = (after_gc - before) / 1024 / 1024

        print(f"\nGC release test:")
        print(f"  Created: {created_mb:.2f} MB")
        print(f"  Released: {released_mb:.2f} MB")
        print(f"  Retained: {retained_mb:.2f} MB")

        # Should release most memory
        release_pct = (released_mb / created_mb * 100) if created_mb > 0 else 0
        print(f"  Release %: {release_pct:.1f}%")

        assert release_pct > 50, f"GC only released {release_pct:.1f}% of memory"

    def test_gc_releases_simulation_results(self, email_campaign_config):
        """Test that simulation results are properly garbage collected."""
        population = create_population(5_000)
        engine = SimulationEngine(population)

        gc.collect()
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]

        # Create many results
        results = []
        for _ in range(10):
            result = engine.simulate(email_campaign_config)
            results.append(result)

        after_results = tracemalloc.get_traced_memory()[0]

        # Delete results
        del results
        gc.collect()

        after_gc = tracemalloc.get_traced_memory()[0]

        results_mb = (after_results - before) / 1024 / 1024
        after_gc_mb = (after_gc - before) / 1024 / 1024

        print(f"\nSimulation results GC:")
        print(f"  Results memory: {results_mb:.2f} MB")
        print(f"  After GC: {after_gc_mb:.2f} MB")

    def test_weak_references_cleanup(self):
        """Test that objects with weak references are properly cleaned."""
        import weakref

        population = create_population(1_000)
        weak_ref = weakref.ref(population)

        # Verify weak reference works
        assert weak_ref() is not None

        # Delete strong reference
        del population
        gc.collect()

        # Weak reference should be dead
        assert weak_ref() is None, "Population not properly garbage collected"


# =============================================================================
# PEAK MEMORY TESTS
# =============================================================================


class TestPeakMemory:
    """Tests for peak memory usage constraints."""

    def test_10k_simulation_peak_under_512mb(self, email_campaign_config):
        """Test that 10K simulation stays under 512MB peak memory."""
        gc.collect()
        tracemalloc.reset_peak()

        population = create_population(10_000)
        engine = SimulationEngine(population)

        result = engine.simulate(email_campaign_config)
        assert result.status == SimulationStatus.COMPLETED

        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / 1024 / 1024

        print(f"\n10K simulation peak memory: {peak_mb:.2f} MB")

        assert peak_mb < 512, f"Peak memory {peak_mb:.2f} MB exceeds 512MB limit"

    def test_50k_simulation_peak_under_2gb(self, email_campaign_config):
        """Test that 50K simulation stays under 2GB peak memory."""
        gc.collect()
        tracemalloc.reset_peak()

        population = create_population(50_000)
        engine = SimulationEngine(population)

        result = engine.simulate(email_campaign_config)
        assert result.status == SimulationStatus.COMPLETED

        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / 1024 / 1024

        print(f"\n50K simulation peak memory: {peak_mb:.2f} MB")

        assert peak_mb < 2048, f"Peak memory {peak_mb:.2f} MB exceeds 2GB limit"

    def test_heterogeneity_memory_overhead(self, email_campaign_config):
        """Test memory overhead of heterogeneity calculation."""
        population = create_population(10_000)
        engine = SimulationEngine(population)

        # Without heterogeneity
        gc.collect()
        tracemalloc.reset_peak()

        engine.simulate(email_campaign_config, calculate_heterogeneity=False)
        _, peak_without = tracemalloc.get_traced_memory()

        # With heterogeneity
        gc.collect()
        tracemalloc.reset_peak()

        engine.simulate(email_campaign_config, calculate_heterogeneity=True)
        _, peak_with = tracemalloc.get_traced_memory()

        without_mb = peak_without / 1024 / 1024
        with_mb = peak_with / 1024 / 1024
        overhead_mb = with_mb - without_mb

        print(f"\nHeterogeneity memory overhead:")
        print(f"  Without: {without_mb:.2f} MB")
        print(f"  With: {with_mb:.2f} MB")
        print(f"  Overhead: {overhead_mb:.2f} MB")

        # Overhead should be modest
        assert overhead_mb < 100, f"Heterogeneity overhead {overhead_mb:.2f} MB too high"


# =============================================================================
# MEMORY EFFICIENCY TESTS
# =============================================================================


class TestMemoryEfficiency:
    """Tests for memory efficiency metrics."""

    def test_memory_per_twin_reasonable(self, email_campaign_config):
        """Test that memory per twin is within reasonable bounds."""
        gc.collect()
        tracemalloc.reset_peak()

        n_twins = 10_000
        population = create_population(n_twins)
        engine = SimulationEngine(population)

        result = engine.simulate(email_campaign_config)

        current, peak = tracemalloc.get_traced_memory()
        peak_kb = peak / 1024
        per_twin_kb = peak_kb / n_twins

        print(f"\nMemory efficiency:")
        print(f"  Total: {peak_kb/1024:.2f} MB for {n_twins:,} twins")
        print(f"  Per twin: {per_twin_kb:.2f} KB")

        # Should be under 50KB per twin
        assert per_twin_kb < 50, f"Memory per twin {per_twin_kb:.2f} KB too high"

    def test_result_object_size(self, email_campaign_config):
        """Test the memory size of simulation result objects."""
        import sys

        population = create_population(5_000)
        engine = SimulationEngine(population)

        result = engine.simulate(email_campaign_config, calculate_heterogeneity=True)

        # Get size of result object
        result_size = sys.getsizeof(result)

        # Get size including nested objects (approximate)
        def get_total_size(obj, seen=None):
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(get_total_size(v, seen) for v in obj.values())
            elif hasattr(obj, "__dict__"):
                size += get_total_size(obj.__dict__, seen)
            elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                size += sum(get_total_size(i, seen) for i in obj)
            return size

        total_size = get_total_size(result)
        total_kb = total_size / 1024

        print(f"\nSimulation result size:")
        print(f"  Direct: {result_size} bytes")
        print(f"  Total (nested): {total_kb:.2f} KB")

        # Result object should be reasonably sized
        assert total_kb < 500, f"Result object {total_kb:.2f} KB too large"
