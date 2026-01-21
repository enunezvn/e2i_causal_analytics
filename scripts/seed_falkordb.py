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
from datetime import datetime, timezone
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
# DOMAIN DATA - Pharmaceutical Commercial Operations Knowledge Graph
# =============================================================================

BRANDS = [
    {"name": "Remibrutinib", "type": "BTK_inhibitor", "indication": "CSU", "launch_year": 2024},
    {"name": "Fabhalta", "type": "Factor_B_inhibitor", "indication": "PNH", "launch_year": 2023},
    {"name": "Kisqali", "type": "CDK4_6_inhibitor", "indication": "HR_HER2_breast_cancer", "launch_year": 2017},
]

REGIONS = [
    {"name": "Northeast", "states": "NY, NJ, PA, CT, MA", "market_potential": "high"},
    {"name": "South", "states": "TX, FL, GA, NC, SC", "market_potential": "high"},
    {"name": "Midwest", "states": "IL, OH, MI, IN, WI", "market_potential": "medium"},
    {"name": "West", "states": "CA, WA, OR, AZ, CO", "market_potential": "high"},
]

# Core KPIs - streamlined for meaningful connections
KPIS = [
    {"name": "TRx", "display_name": "Total Prescriptions", "category": "volume", "description": "Total prescription volume"},
    {"name": "NRx", "display_name": "New Prescriptions", "category": "volume", "description": "New patient prescriptions"},
    {"name": "Market_Share", "display_name": "Market Share", "category": "market", "description": "Brand share of therapeutic market"},
    {"name": "HCP_Reach", "display_name": "HCP Reach", "category": "engagement", "description": "Percentage of target HCPs engaged"},
    {"name": "Conversion_Rate", "display_name": "Conversion Rate", "category": "funnel", "description": "Rate of diagnosis to prescription"},
    {"name": "Patient_Retention", "display_name": "Patient Retention", "category": "loyalty", "description": "Patients continuing therapy"},
]

# HCPs - Using supported node type "HCP"
HCPS = [
    {"name": "Dr. Sarah Chen", "specialty": "Oncologist", "region": "Northeast", "tier": "KOL", "npi": "1234567890"},
    {"name": "Dr. Michael Roberts", "specialty": "Oncologist", "region": "South", "tier": "High", "npi": "1234567891"},
    {"name": "Dr. Emily Watson", "specialty": "Hematologist", "region": "West", "tier": "KOL", "npi": "1234567892"},
    {"name": "Dr. James Miller", "specialty": "Dermatologist", "region": "Northeast", "tier": "High", "npi": "1234567893"},
    {"name": "Dr. Lisa Park", "specialty": "Nephrologist", "region": "Midwest", "tier": "Medium", "npi": "1234567894"},
    {"name": "Dr. David Kim", "specialty": "Hematologist", "region": "South", "tier": "High", "npi": "1234567895"},
]

# Patients - Using supported node type "Patient"
PATIENTS = [
    {"name": "Patient_A001", "condition": "HR_HER2_breast_cancer", "stage": "first_line", "region": "Northeast"},
    {"name": "Patient_A002", "condition": "HR_HER2_breast_cancer", "stage": "second_line", "region": "South"},
    {"name": "Patient_B001", "condition": "PNH", "stage": "maintenance", "region": "West"},
    {"name": "Patient_B002", "condition": "PNH", "stage": "first_line", "region": "Midwest"},
    {"name": "Patient_C001", "condition": "CSU", "stage": "first_line", "region": "Northeast"},
    {"name": "Patient_C002", "condition": "CSU", "stage": "maintenance", "region": "South"},
]

# Treatments - Using supported node type "Treatment"
TREATMENTS = [
    {"name": "Kisqali_Regimen", "brand": "Kisqali", "line": "first_line", "duration_weeks": 24},
    {"name": "Kisqali_Maintenance", "brand": "Kisqali", "line": "maintenance", "duration_weeks": 52},
    {"name": "Fabhalta_Therapy", "brand": "Fabhalta", "line": "first_line", "duration_weeks": 26},
    {"name": "Remibrutinib_Treatment", "brand": "Remibrutinib", "line": "first_line", "duration_weeks": 12},
]

# CausalPaths - Discovered causal relationships
CAUSAL_PATHS = [
    {"name": "HCP_Engagement_to_NRx", "description": "HCP engagement drives new prescriptions", "effect_size": 0.35, "confidence": 0.92},
    {"name": "NRx_to_Market_Share", "description": "New prescriptions increase market share", "effect_size": 0.45, "confidence": 0.88},
    {"name": "Retention_to_TRx", "description": "Patient retention sustains total prescriptions", "effect_size": 0.52, "confidence": 0.95},
    {"name": "Regional_Access_to_Conversion", "description": "Regional formulary access improves conversion", "effect_size": 0.28, "confidence": 0.85},
]

# Agents - Streamlined to key agents
AGENTS = [
    {"name": "orchestrator", "tier": 1, "category": "coordination", "description": "Routes queries to specialist agents"},
    {"name": "causal_impact", "tier": 2, "category": "causal", "description": "Estimates causal effects of interventions"},
    {"name": "gap_analyzer", "tier": 2, "category": "causal", "description": "Identifies ROI opportunities in data"},
    {"name": "experiment_designer", "tier": 3, "category": "monitoring", "description": "Designs A/B tests and experiments"},
    {"name": "prediction_synthesizer", "tier": 4, "category": "prediction", "description": "Generates ML-based forecasts"},
    {"name": "explainer", "tier": 5, "category": "self_improvement", "description": "Provides natural language explanations"},
]

# =============================================================================
# RELATIONSHIPS - Meaningful connections between entities
# =============================================================================

RELATIONSHIPS = [
    # === HCP -> Brand (PRESCRIBES) ===
    {"from_type": "HCP", "from_name": "Dr. Sarah Chen", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "PRESCRIBES", "weight": 0.9},
    {"from_type": "HCP", "from_name": "Dr. Michael Roberts", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "PRESCRIBES", "weight": 0.85},
    {"from_type": "HCP", "from_name": "Dr. Emily Watson", "to_type": "Brand", "to_name": "Fabhalta", "rel_type": "PRESCRIBES", "weight": 0.88},
    {"from_type": "HCP", "from_name": "Dr. Emily Watson", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "PRESCRIBES", "weight": 0.7},
    {"from_type": "HCP", "from_name": "Dr. James Miller", "to_type": "Brand", "to_name": "Remibrutinib", "rel_type": "PRESCRIBES", "weight": 0.92},
    {"from_type": "HCP", "from_name": "Dr. Lisa Park", "to_type": "Brand", "to_name": "Fabhalta", "rel_type": "PRESCRIBES", "weight": 0.8},
    {"from_type": "HCP", "from_name": "Dr. David Kim", "to_type": "Brand", "to_name": "Fabhalta", "rel_type": "PRESCRIBES", "weight": 0.85},
    {"from_type": "HCP", "from_name": "Dr. David Kim", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "PRESCRIBES", "weight": 0.75},

    # === HCP -> Region (PRACTICES_IN) ===
    {"from_type": "HCP", "from_name": "Dr. Sarah Chen", "to_type": "Region", "to_name": "Northeast", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_name": "Dr. Michael Roberts", "to_type": "Region", "to_name": "South", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_name": "Dr. Emily Watson", "to_type": "Region", "to_name": "West", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_name": "Dr. James Miller", "to_type": "Region", "to_name": "Northeast", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_name": "Dr. Lisa Park", "to_type": "Region", "to_name": "Midwest", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_name": "Dr. David Kim", "to_type": "Region", "to_name": "South", "rel_type": "PRACTICES_IN", "weight": 1.0},

    # === Patient -> HCP (TREATED_BY) ===
    {"from_type": "Patient", "from_name": "Patient_A001", "to_type": "HCP", "to_name": "Dr. Sarah Chen", "rel_type": "TREATED_BY", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_A002", "to_type": "HCP", "to_name": "Dr. Michael Roberts", "rel_type": "TREATED_BY", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_B001", "to_type": "HCP", "to_name": "Dr. Emily Watson", "rel_type": "TREATED_BY", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_B002", "to_type": "HCP", "to_name": "Dr. Lisa Park", "rel_type": "TREATED_BY", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_C001", "to_type": "HCP", "to_name": "Dr. James Miller", "rel_type": "TREATED_BY", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_C002", "to_type": "HCP", "to_name": "Dr. David Kim", "rel_type": "TREATED_BY", "weight": 1.0},

    # === Patient -> Treatment (RECEIVES) ===
    {"from_type": "Patient", "from_name": "Patient_A001", "to_type": "Treatment", "to_name": "Kisqali_Regimen", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_A002", "to_type": "Treatment", "to_name": "Kisqali_Maintenance", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_B001", "to_type": "Treatment", "to_name": "Fabhalta_Therapy", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_B002", "to_type": "Treatment", "to_name": "Fabhalta_Therapy", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_C001", "to_type": "Treatment", "to_name": "Remibrutinib_Treatment", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_C002", "to_type": "Treatment", "to_name": "Remibrutinib_Treatment", "rel_type": "RECEIVES", "weight": 1.0},

    # === Patient -> Region (LOCATED_IN) ===
    {"from_type": "Patient", "from_name": "Patient_A001", "to_type": "Region", "to_name": "Northeast", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_A002", "to_type": "Region", "to_name": "South", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_B001", "to_type": "Region", "to_name": "West", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_B002", "to_type": "Region", "to_name": "Midwest", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_C001", "to_type": "Region", "to_name": "Northeast", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_name": "Patient_C002", "to_type": "Region", "to_name": "South", "rel_type": "LOCATED_IN", "weight": 1.0},

    # === Treatment -> Brand (USES) ===
    {"from_type": "Treatment", "from_name": "Kisqali_Regimen", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "USES", "weight": 1.0},
    {"from_type": "Treatment", "from_name": "Kisqali_Maintenance", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "USES", "weight": 1.0},
    {"from_type": "Treatment", "from_name": "Fabhalta_Therapy", "to_type": "Brand", "to_name": "Fabhalta", "rel_type": "USES", "weight": 1.0},
    {"from_type": "Treatment", "from_name": "Remibrutinib_Treatment", "to_type": "Brand", "to_name": "Remibrutinib", "rel_type": "USES", "weight": 1.0},

    # === Brand -> KPI (TRACKS) ===
    {"from_type": "Brand", "from_name": "Kisqali", "to_type": "KPI", "to_name": "TRx", "rel_type": "TRACKS", "weight": 0.95},
    {"from_type": "Brand", "from_name": "Kisqali", "to_type": "KPI", "to_name": "Market_Share", "rel_type": "TRACKS", "weight": 0.9},
    {"from_type": "Brand", "from_name": "Fabhalta", "to_type": "KPI", "to_name": "TRx", "rel_type": "TRACKS", "weight": 0.9},
    {"from_type": "Brand", "from_name": "Fabhalta", "to_type": "KPI", "to_name": "NRx", "rel_type": "TRACKS", "weight": 0.85},
    {"from_type": "Brand", "from_name": "Remibrutinib", "to_type": "KPI", "to_name": "NRx", "rel_type": "TRACKS", "weight": 0.92},
    {"from_type": "Brand", "from_name": "Remibrutinib", "to_type": "KPI", "to_name": "Conversion_Rate", "rel_type": "TRACKS", "weight": 0.88},

    # === KPI -> KPI (CAUSES / AFFECTS) - Causal Chain ===
    {"from_type": "KPI", "from_name": "HCP_Reach", "to_type": "KPI", "to_name": "NRx", "rel_type": "CAUSES", "weight": 0.75},
    {"from_type": "KPI", "from_name": "NRx", "to_type": "KPI", "to_name": "TRx", "rel_type": "CAUSES", "weight": 0.9},
    {"from_type": "KPI", "from_name": "TRx", "to_type": "KPI", "to_name": "Market_Share", "rel_type": "CAUSES", "weight": 0.85},
    {"from_type": "KPI", "from_name": "Conversion_Rate", "to_type": "KPI", "to_name": "NRx", "rel_type": "CAUSES", "weight": 0.8},
    {"from_type": "KPI", "from_name": "Patient_Retention", "to_type": "KPI", "to_name": "TRx", "rel_type": "AFFECTS", "weight": 0.7},

    # === Region -> KPI (INFLUENCES) ===
    {"from_type": "Region", "from_name": "Northeast", "to_type": "KPI", "to_name": "Market_Share", "rel_type": "INFLUENCES", "weight": 0.6},
    {"from_type": "Region", "from_name": "South", "to_type": "KPI", "to_name": "HCP_Reach", "rel_type": "INFLUENCES", "weight": 0.55},
    {"from_type": "Region", "from_name": "West", "to_type": "KPI", "to_name": "Conversion_Rate", "rel_type": "INFLUENCES", "weight": 0.5},
    {"from_type": "Region", "from_name": "Midwest", "to_type": "KPI", "to_name": "Patient_Retention", "rel_type": "INFLUENCES", "weight": 0.45},

    # === CausalPath -> KPI (EXPLAINS) ===
    {"from_type": "CausalPath", "from_name": "HCP_Engagement_to_NRx", "to_type": "KPI", "to_name": "NRx", "rel_type": "EXPLAINS", "weight": 0.92},
    {"from_type": "CausalPath", "from_name": "NRx_to_Market_Share", "to_type": "KPI", "to_name": "Market_Share", "rel_type": "EXPLAINS", "weight": 0.88},
    {"from_type": "CausalPath", "from_name": "Retention_to_TRx", "to_type": "KPI", "to_name": "TRx", "rel_type": "EXPLAINS", "weight": 0.95},
    {"from_type": "CausalPath", "from_name": "Regional_Access_to_Conversion", "to_type": "KPI", "to_name": "Conversion_Rate", "rel_type": "EXPLAINS", "weight": 0.85},

    # === Agent -> KPI (ANALYZES / MONITORS) ===
    {"from_type": "Agent", "from_name": "causal_impact", "to_type": "KPI", "to_name": "TRx", "rel_type": "ANALYZES", "weight": 0.95},
    {"from_type": "Agent", "from_name": "causal_impact", "to_type": "KPI", "to_name": "Market_Share", "rel_type": "ANALYZES", "weight": 0.9},
    {"from_type": "Agent", "from_name": "gap_analyzer", "to_type": "KPI", "to_name": "Conversion_Rate", "rel_type": "ANALYZES", "weight": 0.88},
    {"from_type": "Agent", "from_name": "gap_analyzer", "to_type": "KPI", "to_name": "HCP_Reach", "rel_type": "ANALYZES", "weight": 0.85},
    {"from_type": "Agent", "from_name": "prediction_synthesizer", "to_type": "KPI", "to_name": "NRx", "rel_type": "PREDICTS", "weight": 0.92},
    {"from_type": "Agent", "from_name": "prediction_synthesizer", "to_type": "KPI", "to_name": "Patient_Retention", "rel_type": "PREDICTS", "weight": 0.87},

    # === Agent -> CausalPath (DISCOVERED) ===
    {"from_type": "Agent", "from_name": "causal_impact", "to_type": "CausalPath", "to_name": "HCP_Engagement_to_NRx", "rel_type": "DISCOVERED", "weight": 1.0},
    {"from_type": "Agent", "from_name": "causal_impact", "to_type": "CausalPath", "to_name": "NRx_to_Market_Share", "rel_type": "DISCOVERED", "weight": 1.0},
    {"from_type": "Agent", "from_name": "causal_impact", "to_type": "CausalPath", "to_name": "Retention_to_TRx", "rel_type": "DISCOVERED", "weight": 1.0},
    {"from_type": "Agent", "from_name": "gap_analyzer", "to_type": "CausalPath", "to_name": "Regional_Access_to_Conversion", "rel_type": "DISCOVERED", "weight": 1.0},

    # === Agent -> Brand (MONITORS) ===
    {"from_type": "Agent", "from_name": "orchestrator", "to_type": "Brand", "to_name": "Kisqali", "rel_type": "MONITORS", "weight": 1.0},
    {"from_type": "Agent", "from_name": "orchestrator", "to_type": "Brand", "to_name": "Fabhalta", "rel_type": "MONITORS", "weight": 1.0},
    {"from_type": "Agent", "from_name": "orchestrator", "to_type": "Brand", "to_name": "Remibrutinib", "rel_type": "MONITORS", "weight": 1.0},
]

# Legacy variables for backward compatibility (not used in new seeding)
HCP_SPECIALTIES = []
JOURNEY_STAGES = []
CAUSAL_RELATIONSHIPS = RELATIONSHIPS  # Alias for backward compatibility


# =============================================================================
# HELPERS
# =============================================================================

def get_timestamp() -> str:
    """Get current UTC timestamp as ISO string (FalkorDB-compatible)."""
    return datetime.now(timezone.utc).isoformat()


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
        "CREATE INDEX ON :HCP(name)",
        "CREATE INDEX ON :Patient(name)",
        "CREATE INDEX ON :Treatment(name)",
        "CREATE INDEX ON :CausalPath(name)",
    ]


def generate_brand_queries() -> List[str]:
    """Generate brand node creation queries."""
    queries = []
    ts = get_timestamp()
    for brand in BRANDS:
        queries.append(f"""
            CREATE (:Brand {{
                name: '{brand["name"]}',
                type: '{brand["type"]}',
                indication: '{brand["indication"]}',
                launch_year: {brand.get("launch_year", 2020)},
                created_at: '{ts}'
            }})
        """)
    return queries


def generate_region_queries() -> List[str]:
    """Generate region node creation queries."""
    queries = []
    ts = get_timestamp()
    for region in REGIONS:
        queries.append(f"""
            CREATE (:Region {{
                name: '{region["name"]}',
                states: '{region["states"]}',
                market_potential: '{region.get("market_potential", "medium")}',
                created_at: '{ts}'
            }})
        """)
    return queries


def generate_hcp_queries() -> List[str]:
    """Generate HCP node creation queries."""
    queries = []
    ts = get_timestamp()
    for hcp in HCPS:
        queries.append(f"""
            CREATE (:HCP {{
                name: '{hcp["name"]}',
                specialty: '{hcp["specialty"]}',
                region: '{hcp["region"]}',
                tier: '{hcp["tier"]}',
                npi: '{hcp["npi"]}',
                created_at: '{ts}'
            }})
        """)
    return queries


def generate_patient_queries() -> List[str]:
    """Generate Patient node creation queries."""
    queries = []
    ts = get_timestamp()
    for patient in PATIENTS:
        queries.append(f"""
            CREATE (:Patient {{
                name: '{patient["name"]}',
                condition: '{patient["condition"]}',
                stage: '{patient["stage"]}',
                region: '{patient["region"]}',
                created_at: '{ts}'
            }})
        """)
    return queries


def generate_treatment_queries() -> List[str]:
    """Generate Treatment node creation queries."""
    queries = []
    ts = get_timestamp()
    for treatment in TREATMENTS:
        queries.append(f"""
            CREATE (:Treatment {{
                name: '{treatment["name"]}',
                brand: '{treatment["brand"]}',
                line: '{treatment["line"]}',
                duration_weeks: {treatment["duration_weeks"]},
                created_at: '{ts}'
            }})
        """)
    return queries


def generate_causal_path_queries() -> List[str]:
    """Generate CausalPath node creation queries."""
    queries = []
    ts = get_timestamp()
    for path in CAUSAL_PATHS:
        queries.append(f"""
            CREATE (:CausalPath {{
                name: '{path["name"]}',
                description: '{path["description"]}',
                effect_size: {path["effect_size"]},
                confidence: {path["confidence"]},
                created_at: '{ts}'
            }})
        """)
    return queries


def generate_kpi_queries() -> List[str]:
    """Generate KPI node creation queries."""
    queries = []
    ts = get_timestamp()
    for kpi in KPIS:
        queries.append(f"""
            CREATE (:KPI {{
                name: '{kpi["name"]}',
                display_name: '{kpi["display_name"]}',
                category: '{kpi["category"]}',
                description: '{kpi["description"]}',
                created_at: '{ts}'
            }})
        """)
    return queries


def generate_agent_queries() -> List[str]:
    """Generate agent node creation queries."""
    queries = []
    ts = get_timestamp()
    for agent in AGENTS:
        queries.append(f"""
            CREATE (:Agent {{
                name: '{agent["name"]}',
                tier: {agent["tier"]},
                category: '{agent["category"]}',
                description: '{agent["description"]}',
                created_at: '{ts}'
            }})
        """)
    return queries


# Legacy functions removed - HCPSpecialty and JourneyStage replaced with HCP, Patient, Treatment, CausalPath


def generate_relationship_queries() -> List[str]:
    """Generate relationship creation queries."""
    queries = []
    ts = get_timestamp()
    for rel in CAUSAL_RELATIONSHIPS:
        queries.append(f"""
            MATCH (a:{rel["from_type"]} {{name: '{rel["from_name"]}'}}),
                  (b:{rel["to_type"]} {{name: '{rel["to_name"]}'}})
            CREATE (a)-[:{rel["rel_type"]} {{
                weight: {rel["weight"]},
                created_at: '{ts}'
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

        # HCPs
        logger.info("Seeding %d HCPs...", len(HCPS))
        for query in generate_hcp_queries():
            success = self.execute_query(query, "Create HCP") and success

        # Patients
        logger.info("Seeding %d patients...", len(PATIENTS))
        for query in generate_patient_queries():
            success = self.execute_query(query, "Create patient") and success

        # Treatments
        logger.info("Seeding %d treatments...", len(TREATMENTS))
        for query in generate_treatment_queries():
            success = self.execute_query(query, "Create treatment") and success

        # CausalPaths
        logger.info("Seeding %d causal paths...", len(CAUSAL_PATHS))
        for query in generate_causal_path_queries():
            success = self.execute_query(query, "Create causal path") and success

        # Agents
        logger.info("Seeding %d agents...", len(AGENTS))
        for query in generate_agent_queries():
            success = self.execute_query(query, "Create agent") and success

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
        default=os.getenv("FALKORDB_GRAPH_NAME", "e2i_causal"),
        help="Graph name (default: e2i_causal)"
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
