#!/usr/bin/env python3
"""
E2I Causal Analytics - FalkorDB Knowledge Graph Seed Script.

Seeds the FalkorDB knowledge graph with E2I domain entities and relationships:
- Brands (Remibrutinib, Fabhalta, Kisqali)
- Regions (northeast, south, midwest, west)
- KPIs (TRx, NRx, market share, etc.)
- Agents (18 agents in 6 tiers)
- Causal relationships between entities

Part of Phase 2, Checkpoint 2.3.

Usage:
    python scripts/seed_falkordb.py [--dry-run] [--clear-first]

Options:
    --dry-run       Print Cypher queries without executing
    --clear-first   Clear existing data before seeding
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.config import FalkorDBConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN DATA - From domain_vocabulary_v4.2.0.yaml and kpi_definitions.yaml
# =============================================================================

BRANDS = [
    {"name": "Remibrutinib", "type": "BTK_inhibitor", "indication": "CSU"},
    {"name": "Fabhalta", "type": "Factor_B_inhibitor", "indication": "PNH"},
    {"name": "Kisqali", "type": "CDK4_6_inhibitor", "indication": "HR_HER2_breast_cancer"},
]

REGIONS = [
    {"name": "northeast", "states": "NY, NJ, PA, CT, MA, RI, NH, VT, ME"},
    {"name": "south", "states": "TX, FL, GA, NC, SC, VA, TN, AL, LA, MS, AR, OK"},
    {"name": "midwest", "states": "IL, OH, MI, IN, WI, MN, MO, IA, KS, NE, ND, SD"},
    {"name": "west", "states": "CA, WA, OR, AZ, NV, CO, UT, NM, ID, MT, WY, AK, HI"},
]

# Core KPIs from kpi_definitions.yaml
KPIS = [
    # Volume KPIs
    {"name": "trx", "display_name": "TRx", "category": "volume", "description": "Total prescriptions"},
    {"name": "nrx", "display_name": "NRx", "category": "volume", "description": "New prescriptions"},
    {"name": "nbr", "display_name": "NBRx", "category": "volume", "description": "New-to-brand prescriptions"},
    {"name": "refills", "display_name": "Refills", "category": "volume", "description": "Prescription refills"},

    # Market Share KPIs
    {"name": "market_share", "display_name": "Market Share", "category": "market", "description": "Brand share of market"},
    {"name": "share_of_voice", "display_name": "SOV", "category": "market", "description": "Share of voice in market"},

    # Conversion KPIs
    {"name": "conversion_rate", "display_name": "Conversion Rate", "category": "conversion", "description": "Diagnosis to prescription rate"},
    {"name": "switch_rate", "display_name": "Switch Rate", "category": "conversion", "description": "Rate of switching from competitor"},
    {"name": "retention_rate", "display_name": "Retention Rate", "category": "conversion", "description": "Patient retention rate"},

    # HCP KPIs
    {"name": "hcp_reach", "display_name": "HCP Reach", "category": "hcp", "description": "Percentage of target HCPs reached"},
    {"name": "call_frequency", "display_name": "Call Frequency", "category": "hcp", "description": "Average calls per HCP"},
    {"name": "hcp_adoption", "display_name": "HCP Adoption", "category": "hcp", "description": "HCP adoption rate"},

    # Quality KPIs
    {"name": "data_coverage", "display_name": "Data Coverage", "category": "quality", "description": "Data source coverage"},
    {"name": "model_accuracy", "display_name": "Model Accuracy", "category": "quality", "description": "Prediction model accuracy"},
]

AGENTS = [
    # Tier 0: ML Foundation
    {"name": "scope_definer", "tier": 0, "category": "ml_foundation", "description": "Define ML problem scope"},
    {"name": "data_preparer", "tier": 0, "category": "ml_foundation", "description": "Data preparation & validation"},
    {"name": "model_selector", "tier": 0, "category": "ml_foundation", "description": "Model selection & benchmarking"},
    {"name": "model_trainer", "tier": 0, "category": "ml_foundation", "description": "Model training & tuning"},
    {"name": "model_evaluator", "tier": 0, "category": "ml_foundation", "description": "Model evaluation & metrics"},
    {"name": "model_deployer", "tier": 0, "category": "ml_foundation", "description": "Model deployment & versioning"},
    {"name": "model_monitor", "tier": 0, "category": "ml_foundation", "description": "Model monitoring & drift"},

    # Tier 1: Coordination
    {"name": "orchestrator", "tier": 1, "category": "coordination", "description": "Coordinates all agents, routes queries"},
    {"name": "tool_composer", "tier": 1, "category": "coordination", "description": "Multi-faceted query decomposition"},

    # Tier 2: Causal Analytics
    {"name": "causal_impact", "tier": 2, "category": "causal", "description": "Causal chain tracing & effect estimation"},
    {"name": "heterogeneous_optimizer", "tier": 2, "category": "causal", "description": "Segment-level CATE analysis"},
    {"name": "gap_analyzer", "tier": 2, "category": "causal", "description": "ROI opportunity detection"},
    {"name": "experiment_designer", "tier": 2, "category": "causal", "description": "A/B test design with Digital Twin"},

    # Tier 3: Monitoring
    {"name": "drift_monitor", "tier": 3, "category": "monitoring", "description": "Data/model drift detection"},
    {"name": "data_quality_monitor", "tier": 3, "category": "monitoring", "description": "Data quality monitoring"},

    # Tier 4: Prediction
    {"name": "prediction_synthesizer", "tier": 4, "category": "prediction", "description": "ML prediction aggregation"},
    {"name": "risk_assessor", "tier": 4, "category": "prediction", "description": "Risk assessment & scoring"},

    # Tier 5: Self-Improvement
    {"name": "explainer", "tier": 5, "category": "self_improvement", "description": "Natural language explanations"},
    {"name": "feedback_learner", "tier": 5, "category": "self_improvement", "description": "Self-improvement from feedback"},
]

HCP_SPECIALTIES = [
    {"name": "oncologist", "brand_relevance": ["Kisqali"]},
    {"name": "rheumatologist", "brand_relevance": []},
    {"name": "dermatologist", "brand_relevance": ["Remibrutinib"]},
    {"name": "nephrologist", "brand_relevance": ["Fabhalta"]},
    {"name": "hematologist", "brand_relevance": ["Fabhalta", "Kisqali"]},
    {"name": "primary_care", "brand_relevance": []},
]

JOURNEY_STAGES = [
    {"name": "diagnosis", "order": 1},
    {"name": "treatment_naive", "order": 2},
    {"name": "first_line", "order": 3},
    {"name": "second_line", "order": 4},
    {"name": "maintenance", "order": 5},
    {"name": "discontinuation", "order": 6},
    {"name": "switch", "order": 7},
]

# Causal relationships (domain-specific)
CAUSAL_RELATIONSHIPS = [
    # Brand -> KPI relationships
    {"from_type": "Brand", "from_name": "Kisqali", "to_type": "KPI", "to_name": "trx", "rel_type": "TRACKS", "weight": 0.9},
    {"from_type": "Brand", "from_name": "Kisqali", "to_type": "KPI", "to_name": "market_share", "rel_type": "TRACKS", "weight": 0.85},
    {"from_type": "Brand", "from_name": "Remibrutinib", "to_type": "KPI", "to_name": "trx", "rel_type": "TRACKS", "weight": 0.9},
    {"from_type": "Brand", "from_name": "Fabhalta", "to_type": "KPI", "to_name": "trx", "rel_type": "TRACKS", "weight": 0.9},

    # KPI -> KPI causal relationships
    {"from_type": "KPI", "from_name": "hcp_reach", "to_type": "KPI", "to_name": "hcp_adoption", "rel_type": "CAUSES", "weight": 0.75},
    {"from_type": "KPI", "from_name": "hcp_adoption", "to_type": "KPI", "to_name": "nrx", "rel_type": "CAUSES", "weight": 0.8},
    {"from_type": "KPI", "from_name": "nrx", "to_type": "KPI", "to_name": "trx", "rel_type": "CAUSES", "weight": 0.95},
    {"from_type": "KPI", "from_name": "trx", "to_type": "KPI", "to_name": "market_share", "rel_type": "AFFECTS", "weight": 0.9},
    {"from_type": "KPI", "from_name": "call_frequency", "to_type": "KPI", "to_name": "hcp_reach", "rel_type": "CAUSES", "weight": 0.7},
    {"from_type": "KPI", "from_name": "retention_rate", "to_type": "KPI", "to_name": "trx", "rel_type": "AFFECTS", "weight": 0.65},
    {"from_type": "KPI", "from_name": "switch_rate", "to_type": "KPI", "to_name": "nrx", "rel_type": "AFFECTS", "weight": 0.6},
    {"from_type": "KPI", "from_name": "conversion_rate", "to_type": "KPI", "to_name": "nrx", "rel_type": "CAUSES", "weight": 0.85},

    # Region -> KPI relationships
    {"from_type": "Region", "from_name": "northeast", "to_type": "KPI", "to_name": "market_share", "rel_type": "AFFECTS", "weight": 0.5},
    {"from_type": "Region", "from_name": "south", "to_type": "KPI", "to_name": "market_share", "rel_type": "AFFECTS", "weight": 0.45},
    {"from_type": "Region", "from_name": "midwest", "to_type": "KPI", "to_name": "market_share", "rel_type": "AFFECTS", "weight": 0.4},
    {"from_type": "Region", "from_name": "west", "to_type": "KPI", "to_name": "market_share", "rel_type": "AFFECTS", "weight": 0.55},

    # Agent -> KPI monitoring relationships
    {"from_type": "Agent", "from_name": "drift_monitor", "to_type": "KPI", "to_name": "model_accuracy", "rel_type": "MONITORS", "weight": 1.0},
    {"from_type": "Agent", "from_name": "data_quality_monitor", "to_type": "KPI", "to_name": "data_coverage", "rel_type": "MONITORS", "weight": 1.0},
    {"from_type": "Agent", "from_name": "gap_analyzer", "to_type": "KPI", "to_name": "conversion_rate", "rel_type": "ANALYZES", "weight": 0.9},
    {"from_type": "Agent", "from_name": "causal_impact", "to_type": "KPI", "to_name": "trx", "rel_type": "ANALYZES", "weight": 0.95},

    # Specialty -> Brand relationships
    {"from_type": "HCPSpecialty", "from_name": "oncologist", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "PRESCRIBES", "weight": 0.9},
    {"from_type": "HCPSpecialty", "from_name": "dermatologist", "to_type": "Brand", "to_name": "Remibrutinib", "rel_type": "PRESCRIBES", "weight": 0.85},
    {"from_type": "HCPSpecialty", "from_name": "nephrologist", "to_type": "Brand", "to_name": "Fabhalta", "rel_type": "PRESCRIBES", "weight": 0.8},
    {"from_type": "HCPSpecialty", "from_name": "hematologist", "to_type": "Brand", "to_name": "Fabhalta", "rel_type": "PRESCRIBES", "weight": 0.75},
    {"from_type": "HCPSpecialty", "from_name": "hematologist", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "PRESCRIBES", "weight": 0.7},

    # Journey stage relationships
    {"from_type": "JourneyStage", "from_name": "diagnosis", "to_type": "JourneyStage", "to_name": "treatment_naive", "rel_type": "LEADS_TO", "weight": 1.0},
    {"from_type": "JourneyStage", "from_name": "treatment_naive", "to_type": "JourneyStage", "to_name": "first_line", "rel_type": "LEADS_TO", "weight": 0.9},
    {"from_type": "JourneyStage", "from_name": "first_line", "to_type": "JourneyStage", "to_name": "second_line", "rel_type": "LEADS_TO", "weight": 0.4},
    {"from_type": "JourneyStage", "from_name": "first_line", "to_type": "JourneyStage", "to_name": "maintenance", "rel_type": "LEADS_TO", "weight": 0.5},
    {"from_type": "JourneyStage", "from_name": "second_line", "to_type": "JourneyStage", "to_name": "maintenance", "rel_type": "LEADS_TO", "weight": 0.6},
    {"from_type": "JourneyStage", "from_name": "maintenance", "to_type": "JourneyStage", "to_name": "discontinuation", "rel_type": "LEADS_TO", "weight": 0.15},
    {"from_type": "JourneyStage", "from_name": "first_line", "to_type": "JourneyStage", "to_name": "switch", "rel_type": "LEADS_TO", "weight": 0.1},
]


# =============================================================================
# CYPHER QUERY GENERATORS
# =============================================================================

def generate_clear_query() -> str:
    """Generate query to clear all nodes and relationships."""
    return "MATCH (n) DETACH DELETE n"


def generate_index_queries() -> List[str]:
    """Generate index creation queries."""
    return [
        "CREATE INDEX ON :Brand(name)",
        "CREATE INDEX ON :Region(name)",
        "CREATE INDEX ON :KPI(name)",
        "CREATE INDEX ON :Agent(name)",
        "CREATE INDEX ON :HCPSpecialty(name)",
        "CREATE INDEX ON :JourneyStage(name)",
    ]


def generate_brand_queries() -> List[str]:
    """Generate brand node creation queries."""
    queries = []
    for brand in BRANDS:
        queries.append(f"""
            CREATE (:Brand {{
                name: '{brand["name"]}',
                type: '{brand["type"]}',
                indication: '{brand["indication"]}',
                created_at: datetime()
            }})
        """)
    return queries


def generate_region_queries() -> List[str]:
    """Generate region node creation queries."""
    queries = []
    for region in REGIONS:
        queries.append(f"""
            CREATE (:Region {{
                name: '{region["name"]}',
                states: '{region["states"]}',
                created_at: datetime()
            }})
        """)
    return queries


def generate_kpi_queries() -> List[str]:
    """Generate KPI node creation queries."""
    queries = []
    for kpi in KPIS:
        queries.append(f"""
            CREATE (:KPI {{
                name: '{kpi["name"]}',
                display_name: '{kpi["display_name"]}',
                category: '{kpi["category"]}',
                description: '{kpi["description"]}',
                created_at: datetime()
            }})
        """)
    return queries


def generate_agent_queries() -> List[str]:
    """Generate agent node creation queries."""
    queries = []
    for agent in AGENTS:
        queries.append(f"""
            CREATE (:Agent {{
                name: '{agent["name"]}',
                tier: {agent["tier"]},
                category: '{agent["category"]}',
                description: '{agent["description"]}',
                created_at: datetime()
            }})
        """)
    return queries


def generate_specialty_queries() -> List[str]:
    """Generate HCP specialty node creation queries."""
    queries = []
    for specialty in HCP_SPECIALTIES:
        brand_relevance = str(specialty["brand_relevance"]).replace("'", '"')
        queries.append(f"""
            CREATE (:HCPSpecialty {{
                name: '{specialty["name"]}',
                brand_relevance: {brand_relevance},
                created_at: datetime()
            }})
        """)
    return queries


def generate_journey_queries() -> List[str]:
    """Generate journey stage node creation queries."""
    queries = []
    for stage in JOURNEY_STAGES:
        queries.append(f"""
            CREATE (:JourneyStage {{
                name: '{stage["name"]}',
                stage_order: {stage["order"]},
                created_at: datetime()
            }})
        """)
    return queries


def generate_relationship_queries() -> List[str]:
    """Generate relationship creation queries."""
    queries = []
    for rel in CAUSAL_RELATIONSHIPS:
        queries.append(f"""
            MATCH (a:{rel["from_type"]} {{name: '{rel["from_name"]}'}}),
                  (b:{rel["to_type"]} {{name: '{rel["to_name"]}'}})
            CREATE (a)-[:{rel["rel_type"]} {{
                weight: {rel["weight"]},
                created_at: datetime()
            }}]->(b)
        """)
    return queries


def generate_verification_queries() -> Dict[str, str]:
    """Generate verification queries."""
    return {
        "node_counts": """
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
        """,
        "relationship_counts": """
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(*) as count
            ORDER BY count DESC
        """,
        "causal_paths": """
            MATCH path = (a:KPI)-[:CAUSES|AFFECTS*1..3]->(b:KPI)
            RETURN a.name as from_kpi, b.name as to_kpi, length(path) as path_length
            LIMIT 10
        """,
        "brand_kpi_connections": """
            MATCH (b:Brand)-[:TRACKS]->(k:KPI)
            RETURN b.name as brand, collect(k.name) as kpis
        """,
    }


# =============================================================================
# EXECUTION
# =============================================================================

class FalkorDBSeeder:
    """Seeds the FalkorDB knowledge graph with E2I domain data."""

    def __init__(self, config: FalkorDBConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.client = None
        self.graph = None

    def connect(self) -> bool:
        """Connect to FalkorDB."""
        if self.dry_run:
            logger.info("[DRY RUN] Would connect to FalkorDB at %s:%d",
                       self.config.host, self.config.port)
            return True

        try:
            from falkordb import FalkorDB
            self.client = FalkorDB(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password
            )
            self.graph = self.client.select_graph(self.config.graph_name)
            logger.info("Connected to FalkorDB graph: %s", self.config.graph_name)
            return True
        except Exception as e:
            logger.error("Failed to connect to FalkorDB: %s", e)
            return False

    def execute_query(self, query: str, description: str = "") -> bool:
        """Execute a single Cypher query."""
        query = query.strip()
        if not query:
            return True

        if self.dry_run:
            logger.info("[DRY RUN] %s:\n%s", description, query[:200])
            return True

        try:
            self.graph.query(query)
            if description:
                logger.debug("Executed: %s", description)
            return True
        except Exception as e:
            logger.error("Query failed: %s\nError: %s", description, e)
            return False

    def clear_graph(self) -> bool:
        """Clear all data from the graph."""
        logger.info("Clearing existing graph data...")
        return self.execute_query(generate_clear_query(), "Clear graph")

    def create_indexes(self) -> bool:
        """Create indexes on node properties."""
        logger.info("Creating indexes...")
        for query in generate_index_queries():
            if not self.execute_query(query, "Create index"):
                # Indexes may already exist, continue
                pass
        return True

    def seed_nodes(self) -> bool:
        """Seed all node types."""
        success = True

        # Brands
        logger.info("Seeding %d brands...", len(BRANDS))
        for query in generate_brand_queries():
            success = self.execute_query(query, "Create brand") and success

        # Regions
        logger.info("Seeding %d regions...", len(REGIONS))
        for query in generate_region_queries():
            success = self.execute_query(query, "Create region") and success

        # KPIs
        logger.info("Seeding %d KPIs...", len(KPIS))
        for query in generate_kpi_queries():
            success = self.execute_query(query, "Create KPI") and success

        # Agents
        logger.info("Seeding %d agents...", len(AGENTS))
        for query in generate_agent_queries():
            success = self.execute_query(query, "Create agent") and success

        # HCP Specialties
        logger.info("Seeding %d HCP specialties...", len(HCP_SPECIALTIES))
        for query in generate_specialty_queries():
            success = self.execute_query(query, "Create specialty") and success

        # Journey Stages
        logger.info("Seeding %d journey stages...", len(JOURNEY_STAGES))
        for query in generate_journey_queries():
            success = self.execute_query(query, "Create journey stage") and success

        return success

    def seed_relationships(self) -> bool:
        """Seed all relationships."""
        logger.info("Seeding %d relationships...", len(CAUSAL_RELATIONSHIPS))
        success = True
        for query in generate_relationship_queries():
            success = self.execute_query(query, "Create relationship") and success
        return success

    def verify(self) -> Dict[str, Any]:
        """Verify the seeded data."""
        if self.dry_run:
            logger.info("[DRY RUN] Would verify seeded data")
            return {"dry_run": True}

        results = {}
        verification_queries = generate_verification_queries()

        for name, query in verification_queries.items():
            try:
                result = self.graph.query(query)
                results[name] = result.result_set
                logger.info("Verification - %s: %d results", name, len(result.result_set))
            except Exception as e:
                logger.error("Verification query failed - %s: %s", name, e)
                results[name] = {"error": str(e)}

        return results

    def seed(self, clear_first: bool = False) -> bool:
        """Run the complete seeding process."""
        logger.info("=" * 60)
        logger.info("E2I FalkorDB Knowledge Graph Seeding")
        logger.info("=" * 60)

        if not self.connect():
            return False

        if clear_first:
            if not self.clear_graph():
                return False

        if not self.create_indexes():
            logger.warning("Some indexes may not have been created")

        if not self.seed_nodes():
            logger.error("Node seeding had failures")
            return False

        if not self.seed_relationships():
            logger.error("Relationship seeding had failures")
            return False

        # Verify
        verification = self.verify()

        logger.info("=" * 60)
        logger.info("Seeding complete!")
        logger.info("=" * 60)

        if not self.dry_run and verification:
            for name, data in verification.items():
                if isinstance(data, list):
                    logger.info("  %s: %d results", name, len(data))

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed FalkorDB knowledge graph with E2I domain data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print queries without executing"
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear existing data before seeding"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("FALKORDB_HOST", "localhost"),
        help="FalkorDB host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FALKORDB_PORT", "6381")),
        help="FalkorDB port (default: 6381)"
    )
    parser.add_argument(
        "--graph-name",
        default=os.getenv("FALKORDB_GRAPH", "e2i_knowledge"),
        help="Graph name (default: e2i_knowledge)"
    )

    args = parser.parse_args()

    config = FalkorDBConfig(
        host=args.host,
        port=args.port,
        graph_name=args.graph_name
    )

    seeder = FalkorDBSeeder(config, dry_run=args.dry_run)

    if seeder.seed(clear_first=args.clear_first):
        logger.info("✓ Seeding completed successfully")
        sys.exit(0)
    else:
        logger.error("✗ Seeding failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
