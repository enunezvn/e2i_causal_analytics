#!/usr/bin/env python3
"""
Test Graphiti Entity Extraction
Tests the full Graphiti pipeline with real entity extraction from text.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.graphiti_service import E2IGraphitiService, get_graphiti_service


async def test_graphiti_extraction():
    """Test Graphiti service with sample pharmaceutical content."""

    print("=" * 60)
    print("E2I Graphiti Entity Extraction Test")
    print("=" * 60)

    # Check API keys
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("\n⚠️  ANTHROPIC_API_KEY not set - will use fallback mode")
    else:
        print(f"\n✓ ANTHROPIC_API_KEY found: {anthropic_key[:10]}...")

    # Set FalkorDB port
    os.environ.setdefault("FALKORDB_PORT", "6380")

    print(f"\n📊 FalkorDB: localhost:{os.environ.get('FALKORDB_PORT')}")

    # Initialize service
    print("\n🔧 Initializing Graphiti service...")
    try:
        service = E2IGraphitiService()
        await service.initialize()
        print("✓ Graphiti service initialized")

        if service._graphiti is not None:
            print("✓ Full Graphiti mode enabled (LLM extraction)")
        else:
            print("⚠️  Fallback mode (no LLM extraction)")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Test episodes with pharmaceutical content
    test_episodes = [
        {
            "content": "Dr. Sarah Chen, an oncologist in the Northeast region, prescribed Kisqali to patient P12345 for HR+/HER2- breast cancer treatment. The TRx volume increased by 15% following this prescription.",
            "source": "orchestrator",
            "session_id": "test-session-001",
        },
        {
            "content": "Gap analysis revealed that Dr. James Wilson in the Midwest region has low awareness of Remibrutinib for CSU patients. Recommended action: schedule educational meeting. Expected ROI: $45,000.",
            "source": "gap_analyzer",
            "session_id": "test-session-001",
        },
        {
            "content": "Causal impact analysis shows that HCP engagement activities CAUSE a 0.32 effect size increase in NRx volume with 95% confidence. The trigger 'New Patient Diagnosis' INFLUENCES prescription behavior.",
            "source": "causal_impact",
            "session_id": "test-session-001",
        },
    ]

    print(f"\n📝 Testing {len(test_episodes)} episodes...\n")

    for i, episode in enumerate(test_episodes, 1):
        print(f"--- Episode {i}: {episode['source']} ---")
        print(f"Content: {episode['content'][:100]}...")

        result = await service.add_episode(
            content=episode["content"],
            source=episode["source"],
            session_id=episode["session_id"],
        )

        if result.success:
            print(f"✓ Episode {result.episode_id[:8]} added successfully")
            print(f"  Entities extracted: {len(result.entities_extracted)}")
            for entity in result.entities_extracted[:5]:
                print(f"    - {entity.entity_type.value}: {entity.name} (conf: {entity.confidence:.2f})")
            print(f"  Relationships extracted: {len(result.relationships_extracted)}")
            for rel in result.relationships_extracted[:5]:
                print(f"    - {rel.source_id} -{rel.relationship_type.value}-> {rel.target_id}")
        else:
            print(f"❌ Failed: {result.error}")
        print()

    # Test search
    print("\n🔍 Testing graph search...")
    search_results = await service.search("What treatments does Dr. Chen prescribe?", limit=5)
    print(f"Found {len(search_results)} results")
    for result in search_results[:3]:
        print(f"  - {result.entity_type}: {result.name} (score: {result.score:.2f})")

    # Get stats
    print("\n📊 Graph statistics...")
    stats = await service.get_graph_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print("Nodes by type:")
    for node_type, count in stats.get('nodes_by_type', {}).items():
        if count > 0:
            print(f"  - {node_type}: {count}")

    # Close service
    await service.close()
    print("\n✓ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_graphiti_extraction())
