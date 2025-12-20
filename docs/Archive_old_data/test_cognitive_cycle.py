"""
Test script for the E2I Cognitive Cycle
Requires: Redis (6379), FalkorDB (6380), Supabase configured in .env
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the agentic memory system to path
sys.path.insert(0, str(Path(__file__).parent / "e2i_agentic_memory_system"))

async def initialize_falkordb_graph():
    """Initialize FalkorDB with seed data."""
    from falkordb import FalkorDB

    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6380"))

    print(f"Connecting to FalkorDB at {host}:{port}...")

    try:
        client = FalkorDB(host=host, port=port)
        graph = client.select_graph("e2i_semantic")

        # Create indexes
        print("Creating indexes...")
        try:
            graph.query("CREATE INDEX FOR (p:Patient) ON (p.id)")
        except: pass
        try:
            graph.query("CREATE INDEX FOR (h:HCP) ON (h.id)")
        except: pass
        try:
            graph.query("CREATE INDEX FOR (b:Brand) ON (b.name)")
        except: pass

        # Seed some data
        print("Seeding graph data...")

        # Create brands
        graph.query("""
            MERGE (k:Brand {name: 'Kisqali', id: 'brand_kisqali'})
            SET k.therapeutic_area = 'Oncology', k.drug_class = 'CDK4/6 inhibitor'
        """)
        graph.query("""
            MERGE (f:Brand {name: 'Fabhalta', id: 'brand_fabhalta'})
            SET f.therapeutic_area = 'Nephrology', f.drug_class = 'Factor B inhibitor'
        """)
        graph.query("""
            MERGE (r:Brand {name: 'Remibrutinib', id: 'brand_remibrutinib'})
            SET r.therapeutic_area = 'Immunology', r.drug_class = 'BTK inhibitor'
        """)

        # Create regions
        for region in ['northeast', 'south', 'midwest', 'west']:
            graph.query(f"""
                MERGE (r:Region {{id: '{region}', name: '{region.title()}'}})
            """)

        # Create sample HCPs
        graph.query("""
            MERGE (h:HCP {id: 'HCP-001', name: 'Dr. Smith'})
            SET h.specialty = 'Oncology', h.region = 'northeast', h.tier = 1
        """)
        graph.query("""
            MERGE (h:HCP {id: 'HCP-002', name: 'Dr. Johnson'})
            SET h.specialty = 'Oncology', h.region = 'northeast', h.tier = 2
        """)

        # Create relationships
        graph.query("""
            MATCH (h:HCP {id: 'HCP-001'}), (b:Brand {name: 'Kisqali'})
            MERGE (h)-[r:PRESCRIBES]->(b)
            SET r.volume_monthly = 25, r.market_share = 0.35
        """)

        graph.query("""
            MATCH (h:HCP {id: 'HCP-002'}), (b:Brand {name: 'Kisqali'})
            MERGE (h)-[r:PRESCRIBES]->(b)
            SET r.volume_monthly = 18, r.market_share = 0.28
        """)

        # HCP influence relationship
        graph.query("""
            MATCH (h1:HCP {id: 'HCP-001'}), (h2:HCP {id: 'HCP-002'})
            MERGE (h1)-[r:INFLUENCES]->(h2)
            SET r.influence_strength = 0.75, r.network_type = 'academic'
        """)

        print("✓ FalkorDB initialized with seed data")

        # Show what was created
        result = graph.query("MATCH (n) RETURN labels(n)[0] as type, count(*) as count")
        print("\nGraph contents:")
        for record in result.result_set:
            print(f"  {record[0]}: {record[1]} nodes")

        result = graph.query("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
        print("\nRelationships:")
        for record in result.result_set:
            print(f"  {record[0]}: {record[1]} edges")

        return True
    except Exception as e:
        print(f"✗ FalkorDB initialization failed: {e}")
        return False


async def test_cognitive_cycle():
    """Run a test cognitive cycle."""

    # First initialize FalkorDB
    await initialize_falkordb_graph()

    print("\n" + "="*60)
    print("TESTING COGNITIVE CYCLE")
    print("="*60)

    # Import the cognitive workflow module
    try:
        # We need to set up the module structure for imports
        from importlib import import_module
        import importlib.util

        workflow_path = Path(__file__).parent / "e2i_agentic_memory_system" / "004_cognitive_workflow.py"
        backends_path = Path(__file__).parent / "e2i_agentic_memory_system" / "006_memory_backends_v1_3.py"

        # Load memory backends first
        spec = importlib.util.spec_from_file_location("memory_backends", backends_path)
        memory_backends = importlib.util.module_from_spec(spec)
        sys.modules["memory_backends"] = memory_backends
        spec.loader.exec_module(memory_backends)

        # Create a simple mock for agent_registry since it doesn't exist
        class MockAgentRegistry:
            @staticmethod
            async def invoke_agent(agent_name, context):
                """Mock agent invocation."""
                return {
                    "agent": agent_name,
                    "analysis": f"Mock analysis from {agent_name}",
                    "confidence": 0.85,
                    "findings": ["Finding 1", "Finding 2"]
                }

        sys.modules["agent_registry"] = MockAgentRegistry

        # Now load cognitive workflow
        spec = importlib.util.spec_from_file_location("cognitive_workflow", workflow_path)
        cognitive_workflow = importlib.util.module_from_spec(spec)

        # Patch the relative imports
        cognitive_workflow.memory_backends = memory_backends

        spec.loader.exec_module(cognitive_workflow)

        print("\n✓ Modules loaded successfully")

    except Exception as e:
        print(f"\n✗ Failed to load modules: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test query
    test_query = "Why did Kisqali adoption increase in the Northeast last quarter?"

    print(f"\nTest Query: {test_query}")
    print("-" * 60)

    try:
        # Use in-memory checkpointer for testing (Redis checkpointer requires async context manager)
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        print("✓ Using in-memory checkpointer for test")

        # Run cognitive cycle
        print("\nRunning cognitive cycle...")

        result = await cognitive_workflow.run_cognitive_cycle(
            user_query=test_query,
            user_id="test_analyst_001",
            checkpointer=checkpointer
        )

        print("\n" + "="*60)
        print("COGNITIVE CYCLE COMPLETE")
        print("="*60)
        print(f"\nDetected Intent: {result.get('detected_intent', 'N/A')}")
        print(f"Entities: {result.get('detected_entities', {})}")
        print(f"Investigation Hops: {result.get('current_hop', 0)}")
        print(f"Evidence Items: {len(result.get('evidence_trail', []))}")
        print(f"Agents Used: {result.get('agents_to_invoke', [])}")
        print(f"Confidence: {result.get('confidence_score', 0):.0%}" if result.get('confidence_score') else "Confidence: N/A")
        print(f"\nResponse:\n{result.get('synthesized_response', 'No response generated')[:500]}...")
        print(f"\nWorth Remembering: {result.get('worth_remembering', False)}")
        print(f"New Facts Learned: {len(result.get('new_facts', []))}")
        print(f"New Procedures Learned: {len(result.get('new_procedures', []))}")

        if result.get('error'):
            print(f"\nError: {result['error']}")

    except Exception as e:
        print(f"\n✗ Cognitive cycle failed: {e}")
        import traceback
        traceback.print_exc()


async def test_memory_backends():
    """Test individual memory backends."""
    print("\n" + "="*60)
    print("TESTING MEMORY BACKENDS")
    print("="*60)

    # Test Redis
    print("\n1. Testing Redis Working Memory...")
    try:
        import redis.asyncio as redis_async
        r = redis_async.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        await r.ping()
        print("   ✓ Redis connection OK")
    except Exception as e:
        print(f"   ✗ Redis failed: {e}")

    # Test FalkorDB
    print("\n2. Testing FalkorDB Semantic Memory...")
    try:
        from falkordb import FalkorDB
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6380"))
        client = FalkorDB(host=host, port=port)
        graph = client.select_graph("e2i_semantic")
        result = graph.query("RETURN 1 as test")
        print("   ✓ FalkorDB connection OK")
    except Exception as e:
        print(f"   ✗ FalkorDB failed: {e}")

    # Test Supabase
    print("\n3. Testing Supabase Episodic Memory...")
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if url and key:
            client = create_client(url, key)
            result = client.table("episodic_memories").select("count", count="exact").limit(1).execute()
            print(f"   ✓ Supabase connection OK (episodic_memories table exists)")
        else:
            print("   ✗ Supabase credentials not found")
    except Exception as e:
        print(f"   ✗ Supabase failed: {e}")


async def main():
    """Main test runner."""
    print("E2I Agentic Memory System - Test Suite")
    print("="*60)

    # Test backends first
    await test_memory_backends()

    # Then run the full cognitive cycle
    await test_cognitive_cycle()


if __name__ == "__main__":
    asyncio.run(main())
