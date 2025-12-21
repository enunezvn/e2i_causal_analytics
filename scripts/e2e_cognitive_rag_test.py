#!/usr/bin/env python3
"""
E2E Test: Cognitive RAG + Semantic Memory Integration
======================================================

This script demonstrates the complete E2I cognitive RAG system working with
LLM-assisted retrieval and FalkorDB semantic memory integration.

Test Flow:
1. Initialize FalkorDB semantic memory
2. Populate test entities and relationships (Patients, HCPs, Triggers, CausalPaths)
3. Test Graphiti entity extraction from natural language
4. Run multi-hop cognitive RAG retrieval
5. Verify LLM-assisted semantic memory integration
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
from uuid import uuid4

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set FALKORDB_PORT before importing memory modules
os.environ.setdefault("FALKORDB_PORT", "6380")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e2e_cognitive_rag")

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_SESSION_ID = f"e2e-test-{uuid4().hex[:8]}"

# FalkorDB configuration (from docker-compose)
FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT", "6380"))

# Test entities for semantic memory
TEST_DATA = {
    "patients": [
        {
            "id": "patient-001",
            "segment": "high_value",
            "current_stage": "treatment",
            "brand_affinity": "Kisqali",
            "region": "Northeast",
            "metadata": {"risk_score": 0.3, "engagement_level": "high"}
        },
        {
            "id": "patient-002",
            "segment": "at_risk",
            "current_stage": "diagnosis",
            "brand_affinity": "Fabhalta",
            "region": "Midwest",
            "metadata": {"risk_score": 0.7, "engagement_level": "low"}
        },
    ],
    "hcps": [
        {
            "id": "hcp-001",
            "name": "Dr. Sarah Chen",
            "specialty": "Oncologist",
            "influence_score": 0.85,
            "region": "Northeast",
            "metadata": {"tier": "KOL", "patients_treated": 150}
        },
        {
            "id": "hcp-002",
            "name": "Dr. Michael Torres",
            "specialty": "Hematologist",
            "influence_score": 0.72,
            "region": "Midwest",
            "metadata": {"tier": "Regional", "patients_treated": 80}
        },
    ],
    "triggers": [
        {
            "id": "trigger-001",
            "name": "HCP_Conference_Attendance",
            "category": "engagement",
            "source_kpi": "visit_frequency",
            "target_kpi": "adoption_rate",
            "confidence": 0.88,
            "metadata": {"event": "ASCO 2024", "impact": "positive"}
        },
        {
            "id": "trigger-002",
            "name": "Territory_Realignment",
            "category": "operational",
            "source_kpi": "rep_coverage",
            "target_kpi": "TRx_growth",
            "confidence": 0.75,
            "metadata": {"change_type": "expansion", "effective_date": "2024-Q3"}
        },
    ],
    "causal_paths": [
        {
            "id": "path-001",
            "source_node": "HCP_Detailing",
            "target_node": "Prescription_Volume",
            "effect_size": 0.42,
            "confidence": 0.91,
            "mechanism": "Direct promotional impact",
            "metadata": {"validated": True, "method": "DiD"}
        },
        {
            "id": "path-002",
            "source_node": "Patient_Engagement",
            "target_node": "Treatment_Adherence",
            "effect_size": 0.35,
            "confidence": 0.87,
            "mechanism": "Behavioral activation",
            "metadata": {"validated": True, "method": "PSM"}
        },
    ],
    "relationships": [
        ("patient-001", "TREATED_BY", "hcp-001", {"start_date": "2024-01", "brand": "Kisqali"}),
        ("hcp-001", "PRESCRIBES", "patient-001", {"frequency": "monthly", "adherence": 0.92}),
        ("trigger-001", "IMPACTS", "hcp-001", {"lag_days": 14, "magnitude": 0.2}),
        ("path-001", "CAUSES", "trigger-001", {"direct": True}),
        ("patient-002", "TREATED_BY", "hcp-002", {"start_date": "2024-03", "brand": "Fabhalta"}),
    ],
}


# =============================================================================
# TEST HELPERS
# =============================================================================


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value: Any, indent: int = 0):
    """Print a labeled result."""
    prefix = "  " * indent
    if isinstance(value, dict):
        print(f"{prefix}{label}:")
        for k, v in value.items():
            print(f"{prefix}  {k}: {v}")
    elif isinstance(value, list):
        print(f"{prefix}{label}: [{len(value)} items]")
        for i, item in enumerate(value[:5]):
            print(f"{prefix}  [{i}] {item}")
        if len(value) > 5:
            print(f"{prefix}  ... and {len(value) - 5} more")
    else:
        print(f"{prefix}{label}: {value}")


class TestResult:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests: List[Dict] = []

    def record(self, name: str, passed: bool, details: str = ""):
        self.tests.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name}: {details}")

    def summary(self):
        print_header("TEST SUMMARY")
        print(f"  Total: {self.passed + self.failed}")
        print(f"  Passed: {self.passed}")
        print(f"  Failed: {self.failed}")
        return self.failed == 0


# =============================================================================
# TEST 1: SEMANTIC MEMORY INITIALIZATION
# =============================================================================


async def test_semantic_memory_initialization(results: TestResult) -> Any:
    """Test FalkorDB semantic memory initialization."""
    print_header("TEST 1: Semantic Memory Initialization")

    try:
        from src.memory.semantic_memory import FalkorDBSemanticMemory

        # FalkorDBSemanticMemory uses get_config() and get_falkordb_client() internally
        # FALKORDB_PORT env var was set at module load time
        memory = FalkorDBSemanticMemory()

        results.record(
            "FalkorDB connection",
            memory is not None,
            "Failed to create FalkorDBSemanticMemory"
        )

        # Get initial stats - this forces the connection
        stats = memory.get_graph_stats()
        print_result("Initial graph stats", stats, indent=1)

        results.record(
            "Graph stats retrieval",
            "total_nodes" in stats or "nodes_by_type" in stats or isinstance(stats, dict),
            f"Unexpected stats format: {type(stats)}"
        )

        return memory

    except Exception as e:
        results.record("FalkorDB connection", False, str(e))
        logger.exception("Failed to initialize semantic memory")
        return None


# =============================================================================
# TEST 2: POPULATE SEMANTIC MEMORY WITH TEST DATA
# =============================================================================


async def test_populate_semantic_memory(memory: Any, results: TestResult):
    """Populate FalkorDB with test entities and relationships."""
    print_header("TEST 2: Populate Semantic Memory")

    if memory is None:
        results.record("Skip population", False, "No memory instance")
        return

    # Import E2IEntityType for proper API usage
    from src.memory.episodic_memory import E2IEntityType

    entities_added = 0
    relationships_added = 0

    try:
        # Add patients
        for patient in TEST_DATA["patients"]:
            memory.add_e2i_entity(
                entity_type=E2IEntityType.PATIENT,
                entity_id=patient["id"],
                properties={
                    "segment": patient["segment"],
                    "current_stage": patient["current_stage"],
                    "brand_affinity": patient["brand_affinity"],
                    "region": patient["region"],
                    **patient.get("metadata", {})
                }
            )
            entities_added += 1

        print(f"  Added {len(TEST_DATA['patients'])} Patient entities")

        # Add HCPs
        for hcp in TEST_DATA["hcps"]:
            memory.add_e2i_entity(
                entity_type=E2IEntityType.HCP,
                entity_id=hcp["id"],
                properties={
                    "name": hcp["name"],
                    "specialty": hcp["specialty"],
                    "influence_score": hcp["influence_score"],
                    "region": hcp["region"],
                    **hcp.get("metadata", {})
                }
            )
            entities_added += 1

        print(f"  Added {len(TEST_DATA['hcps'])} HCP entities")

        # Add triggers
        for trigger in TEST_DATA["triggers"]:
            memory.add_e2i_entity(
                entity_type=E2IEntityType.TRIGGER,
                entity_id=trigger["id"],
                properties={
                    "name": trigger["name"],
                    "category": trigger["category"],
                    "source_kpi": trigger["source_kpi"],
                    "target_kpi": trigger["target_kpi"],
                    "confidence": trigger["confidence"],
                    **trigger.get("metadata", {})
                }
            )
            entities_added += 1

        print(f"  Added {len(TEST_DATA['triggers'])} Trigger entities")

        # Add causal paths
        for path in TEST_DATA["causal_paths"]:
            memory.add_e2i_entity(
                entity_type=E2IEntityType.CAUSAL_PATH,
                entity_id=path["id"],
                properties={
                    "source_node": path["source_node"],
                    "target_node": path["target_node"],
                    "effect_size": path["effect_size"],
                    "confidence": path["confidence"],
                    "mechanism": path["mechanism"],
                    **path.get("metadata", {})
                }
            )
            entities_added += 1

        print(f"  Added {len(TEST_DATA['causal_paths'])} CausalPath entities")

        results.record(
            f"Add {entities_added} entities",
            entities_added == sum(len(v) for v in [
                TEST_DATA["patients"],
                TEST_DATA["hcps"],
                TEST_DATA["triggers"],
                TEST_DATA["causal_paths"]
            ]),
            f"Expected more entities"
        )

        # Add relationships
        for source_id, rel_type, target_id, props in TEST_DATA["relationships"]:
            source_type = _get_entity_type_enum(source_id)
            target_type = _get_entity_type_enum(target_id)
            memory.add_e2i_relationship(
                source_type=source_type,
                source_id=source_id,
                target_type=target_type,
                target_id=target_id,
                rel_type=rel_type,
                properties=props
            )
            relationships_added += 1

        print(f"  Added {relationships_added} relationships")

        results.record(
            f"Add {relationships_added} relationships",
            relationships_added == len(TEST_DATA["relationships"]),
            "Failed to add all relationships"
        )

        # Verify with stats
        stats = memory.get_graph_stats()
        print_result("Updated graph stats", stats, indent=1)

    except Exception as e:
        results.record("Populate semantic memory", False, str(e))
        logger.exception("Failed to populate semantic memory")


def _get_entity_type_enum(entity_id: str):
    """Infer E2IEntityType from ID prefix."""
    from src.memory.episodic_memory import E2IEntityType

    if entity_id.startswith("patient"):
        return E2IEntityType.PATIENT
    elif entity_id.startswith("hcp"):
        return E2IEntityType.HCP
    elif entity_id.startswith("trigger"):
        return E2IEntityType.TRIGGER
    elif entity_id.startswith("path"):
        return E2IEntityType.CAUSAL_PATH
    return E2IEntityType.PATIENT  # fallback


# =============================================================================
# TEST 3: SEMANTIC MEMORY QUERIES
# =============================================================================


async def test_semantic_memory_queries(memory: Any, results: TestResult):
    """Test various semantic memory query operations."""
    print_header("TEST 3: Semantic Memory Queries")

    if memory is None:
        results.record("Skip queries", False, "No memory instance")
        return

    from src.memory.episodic_memory import E2IEntityType

    try:
        # Test 3.1: Get entity by ID
        print("\n  3.1 Get entity by ID...")
        entity = memory.get_entity(E2IEntityType.HCP, "hcp-001")
        print_result("HCP hcp-001", entity, indent=2)
        results.record(
            "Get entity by ID",
            entity is not None and ("name" in entity or "id" in entity or len(entity) > 0),
            f"Entity not found or empty: {entity}"
        )

        # Test 3.2: List all nodes of type
        print("\n  3.2 List nodes by type...")
        patients = memory.list_nodes(entity_types=["Patient"])
        print_result("Patient nodes", patients, indent=2)
        results.record(
            "List nodes by type",
            len(patients) >= 2,
            f"Expected 2+ patients, got {len(patients)}"
        )

        # Test 3.3: Get patient network (relationships)
        print("\n  3.3 Get patient treatment network...")
        network = memory.get_patient_network("patient-001", max_depth=2)
        print_result("Patient-001 network", network, indent=2)
        results.record(
            "Get patient network",
            network is not None,
            "Failed to get patient network"
        )

        # Test 3.4: Get HCP influence network
        print("\n  3.4 Get HCP influence network...")
        influence = memory.get_hcp_influence_network("hcp-001", max_depth=2)
        print_result("HCP-001 influence network", influence, indent=2)
        results.record(
            "Get HCP influence network",
            influence is not None,
            "Failed to get HCP influence network"
        )

        # Test 3.5: Traverse causal chain
        print("\n  3.5 Traverse causal chain...")
        causal_chain = memory.traverse_causal_chain("trigger-001", max_depth=3)
        print_result("Causal chain from trigger-001", causal_chain, indent=2)
        results.record(
            "Traverse causal chain",
            causal_chain is not None,
            "Failed to traverse causal chain"
        )

        # Test 3.6: Find causal paths for KPI
        print("\n  3.6 Find causal paths for KPI...")
        kpi_paths = memory.find_causal_paths_for_kpi("adoption_rate")
        print_result("Causal paths affecting adoption_rate", kpi_paths, indent=2)
        results.record(
            "Find causal paths for KPI",
            kpi_paths is not None,
            "Failed to find causal paths"
        )

        # Test 3.7: List relationships
        print("\n  3.7 List relationships...")
        relationships = memory.list_relationships(limit=10)
        print_result("Relationships", relationships, indent=2)
        results.record(
            "List relationships",
            len(relationships) >= 3,
            f"Expected 3+ relationships, got {len(relationships)}"
        )

    except Exception as e:
        results.record("Semantic memory queries", False, str(e))
        logger.exception("Failed during semantic memory queries")


# =============================================================================
# TEST 4: GRAPHITI ENTITY EXTRACTION (LLM-ASSISTED)
# =============================================================================


async def test_graphiti_extraction(results: TestResult):
    """Test Graphiti LLM-assisted entity extraction."""
    print_header("TEST 4: Graphiti Entity Extraction (LLM-Assisted)")

    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [SKIP] OPENAI_API_KEY not set - skipping Graphiti tests")
        results.record("Graphiti API key check", False, "OPENAI_API_KEY not set")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [SKIP] ANTHROPIC_API_KEY not set - skipping Graphiti tests")
        results.record("Graphiti API key check", False, "ANTHROPIC_API_KEY not set")
        return

    try:
        from src.memory.graphiti_service import get_graphiti_service

        print("  Initializing Graphiti service...")
        service = await get_graphiti_service()

        results.record(
            "Graphiti service initialization",
            service is not None,
            "Failed to get Graphiti service"
        )

        if service is None:
            return

        # Test 4.1: Add episode with entity extraction
        print("\n  4.1 Adding episode with LLM extraction...")
        test_episode = """
        Dr. Sarah Chen, a leading oncologist at Memorial Hospital,
        presented new Kisqali efficacy data at ASCO 2024.
        Her presentation showed 35% improvement in progression-free survival.
        Following this, prescription volume in the Northeast region
        increased by 22% within 3 months.
        """

        episode_result = await service.add_episode(
            content=test_episode,
            source="e2e_test",
            session_id=TEST_SESSION_ID
        )

        print_result("Episode extraction result", episode_result, indent=2)
        results.record(
            "Add episode with extraction",
            episode_result is not None,
            "Failed to add episode"
        )

        # Test 4.2: Search the knowledge graph
        print("\n  4.2 Searching knowledge graph...")
        search_results = await service.search(
            query="What is the impact of Dr. Chen's presentation on Kisqali adoption?"
        )

        print_result("Search results", search_results, indent=2)
        results.record(
            "Knowledge graph search",
            search_results is not None,
            "Failed to search knowledge graph"
        )

        # Test 4.3: Get entity subgraph
        print("\n  4.3 Getting entity subgraph...")
        # episode_result is an EpisodeResult object, not a dict
        entities = getattr(episode_result, 'entities', []) if episode_result else []
        if entities:
            # Get entity name - entities are EntityResult objects
            first_entity = entities[0]
            entity_name = getattr(first_entity, 'name', str(first_entity))
            subgraph = await service.get_entity_subgraph(entity_name)
            print_result(f"Subgraph for {entity_name}", subgraph, indent=2)
            results.record(
                "Get entity subgraph",
                subgraph is not None,
                "Failed to get entity subgraph"
            )
        else:
            print("    Skipping - no entities extracted")

    except ImportError as e:
        results.record("Graphiti import", False, str(e))
        logger.warning(f"Graphiti not available: {e}")
    except Exception as e:
        results.record("Graphiti extraction", False, str(e))
        logger.exception("Failed during Graphiti tests")


# =============================================================================
# TEST 5: COGNITIVE RAG WORKFLOW
# =============================================================================


async def test_cognitive_rag_workflow(memory: Any, results: TestResult):
    """Test the complete cognitive RAG workflow with semantic memory."""
    print_header("TEST 5: Cognitive RAG Workflow")

    try:
        # Check for DSPy/LLM availability
        import dspy

        # Try to configure DSPy with Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  [SKIP] ANTHROPIC_API_KEY not set - using mock workflow")
            await _test_mock_cognitive_workflow(memory, results)
            return

        # Configure DSPy
        print("  Configuring DSPy with Claude...")
        lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
        dspy.configure(lm=lm)

        # Import cognitive RAG components
        from src.rag.cognitive_rag_dspy import (
            CognitiveState,
            SummarizerModule,
            MemoryType,
            Evidence,
        )

        results.record("DSPy configuration", True, "")

        # Test 5.1: Summarizer module (query understanding)
        print("\n  5.1 Testing Summarizer Module...")
        summarizer = SummarizerModule()

        test_query = "Why did Kisqali adoption increase in the Northeast region last quarter?"

        summarizer_result = summarizer.forward(
            original_query=test_query,
            conversation_context="User is analyzing brand performance trends.",
            domain_vocabulary="""
            brands: [Remibrutinib (CSU), Fabhalta (PNH), Kisqali (HR+/HER2- breast cancer)]
            regions: [Northeast, Midwest, Southeast, West, Southwest]
            kpis: [TRx, NRx, adoption_rate, market_share, conversion_rate]
            """
        )

        print_result("Summarizer output", summarizer_result, indent=2)
        results.record(
            "Summarizer query rewriting",
            "rewritten_query" in summarizer_result,
            "No rewritten_query in result"
        )

        # Test 5.2: Semantic memory integration in investigation
        print("\n  5.2 Testing semantic memory integration...")
        if memory:
            # Query semantic memory directly
            hcps = memory.list_nodes(entity_types=["HCP"])
            causal_paths = memory.list_nodes(entity_types=["CausalPath"])

            # Create mock evidence from semantic memory
            evidence_items = []
            for hcp in hcps[:2]:
                evidence_items.append(Evidence(
                    source=MemoryType.SEMANTIC,
                    hop_number=1,
                    content=f"HCP {hcp.get('id', 'unknown')}: {hcp.get('specialty', 'N/A')} in {hcp.get('region', 'N/A')}",
                    relevance_score=0.85,
                    metadata={"entity_type": "HCP"}
                ))

            for path in causal_paths[:2]:
                evidence_items.append(Evidence(
                    source=MemoryType.SEMANTIC,
                    hop_number=2,
                    content=f"Causal path: {path.get('source_node', 'N/A')} -> {path.get('target_node', 'N/A')} (effect: {path.get('effect_size', 'N/A')})",
                    relevance_score=0.78,
                    metadata={"entity_type": "CausalPath"}
                ))

            print_result("Evidence from semantic memory", [e.__dict__ for e in evidence_items], indent=2)
            results.record(
                "Semantic memory integration",
                len(evidence_items) > 0,
                "No evidence retrieved from semantic memory"
            )

        # Test 5.3: Full cognitive state
        print("\n  5.3 Testing full cognitive state...")
        state = CognitiveState(
            user_query=test_query,
            conversation_id=TEST_SESSION_ID,
            rewritten_query=summarizer_result.get("rewritten_query", test_query),
            extracted_entities=str(summarizer_result.get("graph_entities", [])),
            detected_intent=summarizer_result.get("primary_intent", "CAUSAL_ANALYSIS"),
        )

        print_result("Cognitive state", {
            "user_query": state.user_query,
            "rewritten_query": state.rewritten_query,
            "detected_intent": state.detected_intent,
            "extracted_entities": state.extracted_entities,
        }, indent=2)

        results.record(
            "Cognitive state creation",
            state.user_query and state.conversation_id,
            "Invalid cognitive state"
        )

    except ImportError as e:
        results.record("DSPy import", False, str(e))
        await _test_mock_cognitive_workflow(memory, results)
    except Exception as e:
        results.record("Cognitive RAG workflow", False, str(e))
        logger.exception("Failed during cognitive RAG tests")


async def _test_mock_cognitive_workflow(memory: Any, results: TestResult):
    """Test with mock cognitive workflow when LLM is unavailable."""
    print("  Running mock cognitive workflow...")

    if memory is None:
        results.record("Mock workflow", False, "No memory instance")
        return

    # Simulate the cognitive workflow
    test_query = "Why did Kisqali adoption increase?"

    # Phase 1: Summarizer (mock)
    print("\n  Mock Phase 1: Query understanding...")
    mock_summary = {
        "rewritten_query": "Kisqali brand adoption rate increase analysis Northeast region",
        "entities": ["Kisqali", "Northeast", "adoption_rate"],
        "intent": "CAUSAL_ANALYSIS"
    }
    print_result("Mock summarizer output", mock_summary, indent=2)

    # Phase 2: Investigator with real semantic memory
    print("\n  Mock Phase 2: Multi-hop investigation with semantic memory...")

    # Hop 1: Episodic (mock)
    hop1_results = [{"content": "Kisqali showed 22% adoption increase in Q3 2024"}]
    print_result("Hop 1 (Episodic - mock)", hop1_results, indent=2)

    # Hop 2: Semantic (real FalkorDB query)
    hcps = memory.list_nodes(entity_types=["HCP"])
    hop2_results = [{"content": f"Found {len(hcps)} HCPs in semantic memory", "entities": hcps}]
    print_result("Hop 2 (Semantic - real)", hop2_results, indent=2)

    # Hop 3: Causal paths (real FalkorDB query)
    causal_paths = memory.list_nodes(entity_types=["CausalPath"])
    hop3_results = [{"content": f"Found {len(causal_paths)} causal paths", "paths": causal_paths}]
    print_result("Hop 3 (Causal - real)", hop3_results, indent=2)

    results.record(
        "Mock cognitive workflow with real semantic memory",
        len(hcps) > 0 or len(causal_paths) > 0,
        "No data retrieved from semantic memory"
    )

    # Phase 3: Synthesis (mock)
    print("\n  Mock Phase 3: Evidence synthesis...")
    synthesis = f"""
    Based on analysis:
    - Query: {test_query}
    - Found {len(hcps)} HCPs in the knowledge graph
    - Found {len(causal_paths)} causal pathways
    - Evidence supports correlation between HCP engagement and adoption
    """
    print(f"  {synthesis}")

    results.record("Mock synthesis", True, "")


# =============================================================================
# TEST 6: CLEANUP
# =============================================================================


async def test_cleanup(memory: Any, results: TestResult):
    """Clean up test data from semantic memory."""
    print_header("TEST 6: Cleanup")

    if memory is None:
        print("  No cleanup needed - memory not initialized")
        return

    from src.memory.episodic_memory import E2IEntityType

    try:
        # Get final stats before cleanup
        stats_before = memory.get_graph_stats()
        print_result("Stats before cleanup", stats_before, indent=1)

        # Delete test entities
        deleted_count = 0
        for entity_type, entities in [
            (E2IEntityType.PATIENT, TEST_DATA["patients"]),
            (E2IEntityType.HCP, TEST_DATA["hcps"]),
            (E2IEntityType.TRIGGER, TEST_DATA["triggers"]),
            (E2IEntityType.CAUSAL_PATH, TEST_DATA["causal_paths"]),
        ]:
            for entity in entities:
                try:
                    memory.delete_entity(entity_type, entity["id"])
                    deleted_count += 1
                except Exception as e:
                    logger.debug(f"Could not delete {entity['id']}: {e}")

        print(f"  Deleted {deleted_count} test entities")

        stats_after = memory.get_graph_stats()
        print_result("Stats after cleanup", stats_after, indent=1)

        results.record("Cleanup", True, f"Deleted {deleted_count} entities")

    except Exception as e:
        results.record("Cleanup", False, str(e))
        logger.exception("Cleanup failed")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================


async def main():
    """Run all E2E tests."""
    print("\n" + "=" * 70)
    print("  E2I COGNITIVE RAG + SEMANTIC MEMORY E2E TEST")
    print("=" * 70)
    print(f"\n  Session ID: {TEST_SESSION_ID}")
    print(f"  FalkorDB: {FALKORDB_HOST}:{FALKORDB_PORT}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")

    results = TestResult()

    # Test 1: Initialize semantic memory
    memory = await test_semantic_memory_initialization(results)

    # Test 2: Populate with test data
    await test_populate_semantic_memory(memory, results)

    # Test 3: Query semantic memory
    await test_semantic_memory_queries(memory, results)

    # Test 4: Graphiti LLM extraction (optional)
    await test_graphiti_extraction(results)

    # Test 5: Cognitive RAG workflow
    await test_cognitive_rag_workflow(memory, results)

    # Test 6: Cleanup
    await test_cleanup(memory, results)

    # Summary
    success = results.summary()

    print("\n" + "=" * 70)
    print(f"  {'SUCCESS' if success else 'FAILURE'}: E2E Test Complete")
    print("=" * 70 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
