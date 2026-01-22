#!/usr/bin/env python3
"""
E2I Causal Analytics - Semantic Graph Seed Script.

Seeds the FalkorDB `e2i_semantic` graph with E2I domain entities and relationships.

This script complements `seed_falkordb.py` which seeds `e2i_causal`. Both graphs
contain similar data but serve different purposes:
- e2i_causal: Causal inference analysis
- e2i_semantic: Semantic memory for agent reasoning

Features:
- Applies schema constraints and indexes from falkordb_config.yaml
- Populates entities: Brands, Regions, KPIs, Agents, HCPs, Patients, etc.
- Creates relationships: PRESCRIBES, TREATS, CAUSES, IMPACTS, etc.
- Idempotent using MERGE statements (safe to re-run)

Usage:
    python scripts/seed_semantic_graph.py [--dry-run] [--clear-first] [--skip-schema]

Options:
    --dry-run       Print Cypher queries without executing
    --clear-first   Clear existing data before seeding
    --skip-schema   Skip constraint and index creation
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Explicit graph name - NOT from env var (e2i_causal uses env var)
SEMANTIC_GRAPH_NAME = "e2i_semantic"

# Config file path
FALKORDB_CONFIG_PATH = Path(__file__).parent.parent / "config" / "ontology" / "falkordb_config.yaml"


# =============================================================================
# DOMAIN DATA - Same entities as e2i_causal for consistency
# =============================================================================

BRANDS = [
    {
        "id": "brand_remibrutinib",
        "name": "Remibrutinib",
        "type": "BTK_inhibitor",
        "indication": "CSU",
        "therapeutic_area": "Immunology",
        "launch_year": 2024,
    },
    {
        "id": "brand_fabhalta",
        "name": "Fabhalta",
        "type": "Factor_B_inhibitor",
        "indication": "PNH",
        "therapeutic_area": "Hematology",
        "launch_year": 2023,
    },
    {
        "id": "brand_kisqali",
        "name": "Kisqali",
        "type": "CDK4_6_inhibitor",
        "indication": "HR_HER2_breast_cancer",
        "therapeutic_area": "Oncology",
        "launch_year": 2017,
    },
]

REGIONS = [
    {"id": "region_northeast", "name": "Northeast", "states": "NY, NJ, PA, CT, MA", "market_potential": "high"},
    {"id": "region_south", "name": "South", "states": "TX, FL, GA, NC, SC", "market_potential": "high"},
    {"id": "region_midwest", "name": "Midwest", "states": "IL, OH, MI, IN, WI", "market_potential": "medium"},
    {"id": "region_west", "name": "West", "states": "CA, WA, OR, AZ, CO", "market_potential": "high"},
]

KPIS = [
    {"id": "kpi_trx", "name": "TRx", "display_name": "Total Prescriptions", "category": "volume"},
    {"id": "kpi_nrx", "name": "NRx", "display_name": "New Prescriptions", "category": "volume"},
    {"id": "kpi_market_share", "name": "Market_Share", "display_name": "Market Share", "category": "market"},
    {"id": "kpi_hcp_reach", "name": "HCP_Reach", "display_name": "HCP Reach", "category": "engagement"},
    {"id": "kpi_conversion_rate", "name": "Conversion_Rate", "display_name": "Conversion Rate", "category": "funnel"},
    {"id": "kpi_patient_retention", "name": "Patient_Retention", "display_name": "Patient Retention", "category": "loyalty"},
]

AGENTS = [
    {"id": "agent_orchestrator", "name": "orchestrator", "tier": 1, "category": "coordination"},
    {"id": "agent_causal_impact", "name": "causal_impact", "tier": 2, "category": "causal"},
    {"id": "agent_gap_analyzer", "name": "gap_analyzer", "tier": 2, "category": "causal"},
    {"id": "agent_experiment_designer", "name": "experiment_designer", "tier": 3, "category": "monitoring"},
    {"id": "agent_prediction_synthesizer", "name": "prediction_synthesizer", "tier": 4, "category": "prediction"},
    {"id": "agent_explainer", "name": "explainer", "tier": 5, "category": "self_improvement"},
]

HCPS = [
    {"id": "hcp_001", "name": "Dr. Sarah Chen", "specialty": "Oncologist", "region": "Northeast", "tier": "KOL", "npi": "1234567890"},
    {"id": "hcp_002", "name": "Dr. Michael Roberts", "specialty": "Oncologist", "region": "South", "tier": "High", "npi": "1234567891"},
    {"id": "hcp_003", "name": "Dr. Emily Watson", "specialty": "Hematologist", "region": "West", "tier": "KOL", "npi": "1234567892"},
    {"id": "hcp_004", "name": "Dr. James Miller", "specialty": "Dermatologist", "region": "Northeast", "tier": "High", "npi": "1234567893"},
    {"id": "hcp_005", "name": "Dr. Lisa Park", "specialty": "Nephrologist", "region": "Midwest", "tier": "Medium", "npi": "1234567894"},
    {"id": "hcp_006", "name": "Dr. David Kim", "specialty": "Hematologist", "region": "South", "tier": "High", "npi": "1234567895"},
]

PATIENTS = [
    {"id": "patient_001", "name": "Patient_A001", "condition": "HR_HER2_breast_cancer", "journey_stage": "first_line", "region": "Northeast"},
    {"id": "patient_002", "name": "Patient_A002", "condition": "HR_HER2_breast_cancer", "journey_stage": "second_line", "region": "South"},
    {"id": "patient_003", "name": "Patient_B001", "condition": "PNH", "journey_stage": "maintenance", "region": "West"},
    {"id": "patient_004", "name": "Patient_B002", "condition": "PNH", "journey_stage": "first_line", "region": "Midwest"},
    {"id": "patient_005", "name": "Patient_C001", "condition": "CSU", "journey_stage": "first_line", "region": "Northeast"},
    {"id": "patient_006", "name": "Patient_C002", "condition": "CSU", "journey_stage": "maintenance", "region": "South"},
]

TREATMENTS = [
    {"id": "treatment_kisqali_regimen", "name": "Kisqali_Regimen", "brand": "Kisqali", "line": "first_line", "duration_weeks": 24},
    {"id": "treatment_kisqali_maint", "name": "Kisqali_Maintenance", "brand": "Kisqali", "line": "maintenance", "duration_weeks": 52},
    {"id": "treatment_fabhalta", "name": "Fabhalta_Therapy", "brand": "Fabhalta", "line": "first_line", "duration_weeks": 26},
    {"id": "treatment_remibrutinib", "name": "Remibrutinib_Treatment", "brand": "Remibrutinib", "line": "first_line", "duration_weeks": 12},
]

CAUSAL_PATHS = [
    {"id": "causal_path_001", "name": "HCP_Engagement_to_NRx", "description": "HCP engagement drives new prescriptions", "effect_size": 0.35, "confidence": 0.92, "method": "DoWhy"},
    {"id": "causal_path_002", "name": "NRx_to_Market_Share", "description": "New prescriptions increase market share", "effect_size": 0.45, "confidence": 0.88, "method": "EconML"},
    {"id": "causal_path_003", "name": "Retention_to_TRx", "description": "Patient retention sustains total prescriptions", "effect_size": 0.52, "confidence": 0.95, "method": "DoWhy"},
    {"id": "causal_path_004", "name": "Regional_Access_to_Conversion", "description": "Regional formulary access improves conversion", "effect_size": 0.28, "confidence": 0.85, "method": "EconML"},
]

TRIGGERS = [
    {"id": "trigger_001", "name": "Peer_Influence_Trigger", "trigger_type": "peer_influence", "priority": "high", "status": "active"},
    {"id": "trigger_002", "name": "Profile_Match_Trigger", "trigger_type": "patient_profile_match", "priority": "medium", "status": "active"},
    {"id": "trigger_003", "name": "Switching_Opportunity", "trigger_type": "switching_opportunity", "priority": "high", "status": "active"},
    {"id": "trigger_004", "name": "Adherence_Risk_Trigger", "trigger_type": "adherence_risk", "priority": "medium", "status": "pending"},
]


# =============================================================================
# RELATIONSHIPS
# =============================================================================

RELATIONSHIPS = [
    # === HCP -> Brand (PRESCRIBES) ===
    {"from_type": "HCP", "from_id": "hcp_001", "to_type": "Brand", "to_id": "brand_kisqali", "rel_type": "PRESCRIBES", "weight": 0.9},
    {"from_type": "HCP", "from_id": "hcp_002", "to_type": "Brand", "to_id": "brand_kisqali", "rel_type": "PRESCRIBES", "weight": 0.85},
    {"from_type": "HCP", "from_id": "hcp_003", "to_type": "Brand", "to_id": "brand_fabhalta", "rel_type": "PRESCRIBES", "weight": 0.88},
    {"from_type": "HCP", "from_id": "hcp_003", "to_type": "Brand", "to_id": "brand_kisqali", "rel_type": "PRESCRIBES", "weight": 0.7},
    {"from_type": "HCP", "from_id": "hcp_004", "to_type": "Brand", "to_id": "brand_remibrutinib", "rel_type": "PRESCRIBES", "weight": 0.92},
    {"from_type": "HCP", "from_id": "hcp_005", "to_type": "Brand", "to_id": "brand_fabhalta", "rel_type": "PRESCRIBES", "weight": 0.8},
    {"from_type": "HCP", "from_id": "hcp_006", "to_type": "Brand", "to_id": "brand_fabhalta", "rel_type": "PRESCRIBES", "weight": 0.85},
    {"from_type": "HCP", "from_id": "hcp_006", "to_type": "Brand", "to_id": "brand_kisqali", "rel_type": "PRESCRIBES", "weight": 0.75},

    # === HCP -> Region (PRACTICES_IN) ===
    {"from_type": "HCP", "from_id": "hcp_001", "to_type": "Region", "to_id": "region_northeast", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_id": "hcp_002", "to_type": "Region", "to_id": "region_south", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_id": "hcp_003", "to_type": "Region", "to_id": "region_west", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_id": "hcp_004", "to_type": "Region", "to_id": "region_northeast", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_id": "hcp_005", "to_type": "Region", "to_id": "region_midwest", "rel_type": "PRACTICES_IN", "weight": 1.0},
    {"from_type": "HCP", "from_id": "hcp_006", "to_type": "Region", "to_id": "region_south", "rel_type": "PRACTICES_IN", "weight": 1.0},

    # === Patient -> HCP (TREATED_BY) ===
    {"from_type": "Patient", "from_id": "patient_001", "to_type": "HCP", "to_id": "hcp_001", "rel_type": "TREATED_BY", "visit_count": 8},
    {"from_type": "Patient", "from_id": "patient_002", "to_type": "HCP", "to_id": "hcp_002", "rel_type": "TREATED_BY", "visit_count": 5},
    {"from_type": "Patient", "from_id": "patient_003", "to_type": "HCP", "to_id": "hcp_003", "rel_type": "TREATED_BY", "visit_count": 6},
    {"from_type": "Patient", "from_id": "patient_004", "to_type": "HCP", "to_id": "hcp_005", "rel_type": "TREATED_BY", "visit_count": 4},
    {"from_type": "Patient", "from_id": "patient_005", "to_type": "HCP", "to_id": "hcp_004", "rel_type": "TREATED_BY", "visit_count": 3},
    {"from_type": "Patient", "from_id": "patient_006", "to_type": "HCP", "to_id": "hcp_006", "rel_type": "TREATED_BY", "visit_count": 7},

    # === Patient -> Treatment (RECEIVES) ===
    {"from_type": "Patient", "from_id": "patient_001", "to_type": "Treatment", "to_id": "treatment_kisqali_regimen", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_002", "to_type": "Treatment", "to_id": "treatment_kisqali_maint", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_003", "to_type": "Treatment", "to_id": "treatment_fabhalta", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_004", "to_type": "Treatment", "to_id": "treatment_fabhalta", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_005", "to_type": "Treatment", "to_id": "treatment_remibrutinib", "rel_type": "RECEIVES", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_006", "to_type": "Treatment", "to_id": "treatment_remibrutinib", "rel_type": "RECEIVES", "weight": 1.0},

    # === Patient -> Region (LOCATED_IN) ===
    {"from_type": "Patient", "from_id": "patient_001", "to_type": "Region", "to_id": "region_northeast", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_002", "to_type": "Region", "to_id": "region_south", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_003", "to_type": "Region", "to_id": "region_west", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_004", "to_type": "Region", "to_id": "region_midwest", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_005", "to_type": "Region", "to_id": "region_northeast", "rel_type": "LOCATED_IN", "weight": 1.0},
    {"from_type": "Patient", "from_id": "patient_006", "to_type": "Region", "to_id": "region_south", "rel_type": "LOCATED_IN", "weight": 1.0},

    # === Treatment -> Brand (USES) ===
    {"from_type": "Treatment", "from_id": "treatment_kisqali_regimen", "to_type": "Brand", "to_id": "brand_kisqali", "rel_type": "USES", "weight": 1.0},
    {"from_type": "Treatment", "from_id": "treatment_kisqali_maint", "to_type": "Brand", "to_id": "brand_kisqali", "rel_type": "USES", "weight": 1.0},
    {"from_type": "Treatment", "from_id": "treatment_fabhalta", "to_type": "Brand", "to_id": "brand_fabhalta", "rel_type": "USES", "weight": 1.0},
    {"from_type": "Treatment", "from_id": "treatment_remibrutinib", "to_type": "Brand", "to_id": "brand_remibrutinib", "rel_type": "USES", "weight": 1.0},

    # === Brand -> KPI (TRACKS) ===
    {"from_type": "Brand", "from_id": "brand_kisqali", "to_type": "KPI", "to_id": "kpi_trx", "rel_type": "TRACKS", "weight": 0.95},
    {"from_type": "Brand", "from_id": "brand_kisqali", "to_type": "KPI", "to_id": "kpi_market_share", "rel_type": "TRACKS", "weight": 0.9},
    {"from_type": "Brand", "from_id": "brand_fabhalta", "to_type": "KPI", "to_id": "kpi_trx", "rel_type": "TRACKS", "weight": 0.9},
    {"from_type": "Brand", "from_id": "brand_fabhalta", "to_type": "KPI", "to_id": "kpi_nrx", "rel_type": "TRACKS", "weight": 0.85},
    {"from_type": "Brand", "from_id": "brand_remibrutinib", "to_type": "KPI", "to_id": "kpi_nrx", "rel_type": "TRACKS", "weight": 0.92},
    {"from_type": "Brand", "from_id": "brand_remibrutinib", "to_type": "KPI", "to_id": "kpi_conversion_rate", "rel_type": "TRACKS", "weight": 0.88},

    # === KPI -> KPI (CAUSES) - Causal Chain ===
    {"from_type": "KPI", "from_id": "kpi_hcp_reach", "to_type": "KPI", "to_id": "kpi_nrx", "rel_type": "CAUSES", "weight": 0.75},
    {"from_type": "KPI", "from_id": "kpi_nrx", "to_type": "KPI", "to_id": "kpi_trx", "rel_type": "CAUSES", "weight": 0.9},
    {"from_type": "KPI", "from_id": "kpi_trx", "to_type": "KPI", "to_id": "kpi_market_share", "rel_type": "CAUSES", "weight": 0.85},
    {"from_type": "KPI", "from_id": "kpi_conversion_rate", "to_type": "KPI", "to_id": "kpi_nrx", "rel_type": "CAUSES", "weight": 0.8},
    {"from_type": "KPI", "from_id": "kpi_patient_retention", "to_type": "KPI", "to_id": "kpi_trx", "rel_type": "AFFECTS", "weight": 0.7},

    # === Region -> KPI (INFLUENCES) ===
    {"from_type": "Region", "from_id": "region_northeast", "to_type": "KPI", "to_id": "kpi_market_share", "rel_type": "INFLUENCES", "weight": 0.6},
    {"from_type": "Region", "from_id": "region_south", "to_type": "KPI", "to_id": "kpi_hcp_reach", "rel_type": "INFLUENCES", "weight": 0.55},
    {"from_type": "Region", "from_id": "region_west", "to_type": "KPI", "to_id": "kpi_conversion_rate", "rel_type": "INFLUENCES", "weight": 0.5},
    {"from_type": "Region", "from_id": "region_midwest", "to_type": "KPI", "to_id": "kpi_patient_retention", "rel_type": "INFLUENCES", "weight": 0.45},

    # === CausalPath -> KPI (IMPACTS) ===
    {"from_type": "CausalPath", "from_id": "causal_path_001", "to_type": "KPI", "to_id": "kpi_nrx", "rel_type": "IMPACTS", "confidence": 0.92},
    {"from_type": "CausalPath", "from_id": "causal_path_002", "to_type": "KPI", "to_id": "kpi_market_share", "rel_type": "IMPACTS", "confidence": 0.88},
    {"from_type": "CausalPath", "from_id": "causal_path_003", "to_type": "KPI", "to_id": "kpi_trx", "rel_type": "IMPACTS", "confidence": 0.95},
    {"from_type": "CausalPath", "from_id": "causal_path_004", "to_type": "KPI", "to_id": "kpi_conversion_rate", "rel_type": "IMPACTS", "confidence": 0.85},

    # === Agent -> KPI (ANALYZES) ===
    {"from_type": "Agent", "from_id": "agent_causal_impact", "to_type": "KPI", "to_id": "kpi_trx", "rel_type": "ANALYZES", "weight": 0.95},
    {"from_type": "Agent", "from_id": "agent_causal_impact", "to_type": "KPI", "to_id": "kpi_market_share", "rel_type": "ANALYZES", "weight": 0.9},
    {"from_type": "Agent", "from_id": "agent_gap_analyzer", "to_type": "KPI", "to_id": "kpi_conversion_rate", "rel_type": "ANALYZES", "weight": 0.88},
    {"from_type": "Agent", "from_id": "agent_gap_analyzer", "to_type": "KPI", "to_id": "kpi_hcp_reach", "rel_type": "ANALYZES", "weight": 0.85},
    {"from_type": "Agent", "from_id": "agent_prediction_synthesizer", "to_type": "KPI", "to_id": "kpi_nrx", "rel_type": "PREDICTS", "weight": 0.92},
    {"from_type": "Agent", "from_id": "agent_prediction_synthesizer", "to_type": "KPI", "to_id": "kpi_patient_retention", "rel_type": "PREDICTS", "weight": 0.87},

    # === Agent -> CausalPath (DISCOVERED) ===
    {"from_type": "Agent", "from_id": "agent_causal_impact", "to_type": "CausalPath", "to_id": "causal_path_001", "rel_type": "DISCOVERED", "weight": 1.0},
    {"from_type": "Agent", "from_id": "agent_causal_impact", "to_type": "CausalPath", "to_id": "causal_path_002", "rel_type": "DISCOVERED", "weight": 1.0},
    {"from_type": "Agent", "from_id": "agent_causal_impact", "to_type": "CausalPath", "to_id": "causal_path_003", "rel_type": "DISCOVERED", "weight": 1.0},
    {"from_type": "Agent", "from_id": "agent_gap_analyzer", "to_type": "CausalPath", "to_id": "causal_path_004", "rel_type": "DISCOVERED", "weight": 1.0},

    # === Agent -> Brand (MONITORS) ===
    {"from_type": "Agent", "from_id": "agent_orchestrator", "to_type": "Brand", "to_id": "brand_kisqali", "rel_type": "MONITORS", "weight": 1.0},
    {"from_type": "Agent", "from_id": "agent_orchestrator", "to_type": "Brand", "to_id": "brand_fabhalta", "rel_type": "MONITORS", "weight": 1.0},
    {"from_type": "Agent", "from_id": "agent_orchestrator", "to_type": "Brand", "to_id": "brand_remibrutinib", "rel_type": "MONITORS", "weight": 1.0},

    # === HCP -> Trigger (RECEIVED) ===
    {"from_type": "HCP", "from_id": "hcp_001", "to_type": "Trigger", "to_id": "trigger_001", "rel_type": "RECEIVED", "accepted": True},
    {"from_type": "HCP", "from_id": "hcp_002", "to_type": "Trigger", "to_id": "trigger_002", "rel_type": "RECEIVED", "accepted": True},
    {"from_type": "HCP", "from_id": "hcp_003", "to_type": "Trigger", "to_id": "trigger_003", "rel_type": "RECEIVED", "accepted": False},
]


# =============================================================================
# HELPERS
# =============================================================================

def get_timestamp() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def load_falkordb_schema() -> Dict[str, Any]:
    """Load FalkorDB schema configuration from YAML."""
    if not FALKORDB_CONFIG_PATH.exists():
        logger.warning(f"Schema config not found: {FALKORDB_CONFIG_PATH}")
        return {}

    with open(FALKORDB_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    return config.get("falkordb_config", {})


# =============================================================================
# SEMANTIC GRAPH SEEDER
# =============================================================================

class SemanticGraphSeeder:
    """Seeds the FalkorDB e2i_semantic graph with E2I domain data."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.client = None
        self.graph = None
        self.graph_name = SEMANTIC_GRAPH_NAME
        self.schema_config = load_falkordb_schema()

    def connect(self) -> bool:
        """Connect to FalkorDB."""
        if self.dry_run:
            logger.info("[DRY RUN] Would connect to FalkorDB graph: %s", self.graph_name)
            return True

        try:
            from falkordb import FalkorDB

            host = os.getenv("FALKORDB_HOST", "localhost")
            port = int(os.getenv("FALKORDB_PORT", "6381"))
            password = os.getenv("FALKORDB_PASSWORD", None)

            self.client = FalkorDB(host=host, port=port, password=password)
            self.graph = self.client.select_graph(self.graph_name)
            logger.info("Connected to FalkorDB graph: %s at %s:%d", self.graph_name, host, port)
            return True
        except Exception as e:
            logger.error("Failed to connect to FalkorDB: %s", e)
            return False

    def execute_query(self, query: str, description: str = "", params: Optional[Dict] = None) -> bool:
        """Execute a single Cypher query."""
        query = query.strip()
        if not query:
            return True

        if self.dry_run:
            logger.info("[DRY RUN] %s:\n%s", description, query[:300])
            return True

        try:
            if params:
                self.graph.query(query, params)
            else:
                self.graph.query(query)
            if description:
                logger.debug("Executed: %s", description)
            return True
        except Exception as e:
            # Some errors are expected (e.g., constraint already exists)
            if "already exists" in str(e).lower() or "already indexed" in str(e).lower():
                logger.debug("Skipped (already exists): %s", description)
                return True
            logger.error("Query failed: %s\nError: %s", description, e)
            return False

    # =========================================================================
    # SCHEMA OPERATIONS
    # =========================================================================

    def clear_graph(self) -> bool:
        """Clear all data from the graph."""
        logger.info("Clearing existing graph data...")
        return self.execute_query("MATCH (n) DETACH DELETE n", "Clear graph")

    def apply_schema(self) -> bool:
        """Apply constraints and indexes from schema configuration."""
        logger.info("Applying schema constraints and indexes...")
        success = True

        # Apply unique constraints
        constraints = self.schema_config.get("constraints", {}).get("unique_properties", [])
        for constraint in constraints:
            node_type = constraint.get("node")
            prop = constraint.get("property")
            if node_type and prop:
                # FalkorDB constraint syntax
                query = f"CREATE CONSTRAINT ON (n:{node_type}) ASSERT n.{prop} IS UNIQUE"
                success = self.execute_query(query, f"Constraint: {node_type}.{prop}") and success

        # Apply indexes
        indexes = self.schema_config.get("indexes", {})
        for node_type, properties in indexes.items():
            for prop in properties:
                query = f"CREATE INDEX ON :{node_type}({prop})"
                success = self.execute_query(query, f"Index: {node_type}.{prop}") and success

        return success

    # =========================================================================
    # ENTITY SEEDING
    # =========================================================================

    def _generate_entity_query(self, label: str, entity: Dict[str, Any]) -> str:
        """Generate MERGE query for an entity."""
        entity_id = entity.get("id")
        props = {k: v for k, v in entity.items()}
        props["created_at"] = get_timestamp()
        props["updated_at"] = get_timestamp()

        # Build property string
        prop_items = []
        for k, v in props.items():
            if isinstance(v, str):
                # Escape single quotes
                v_escaped = v.replace("'", "\\'")
                prop_items.append(f"{k}: '{v_escaped}'")
            elif isinstance(v, bool):
                prop_items.append(f"{k}: {str(v).lower()}")
            elif v is not None:
                prop_items.append(f"{k}: {v}")

        prop_string = ", ".join(prop_items)
        return f"MERGE (n:{label} {{id: '{entity_id}'}}) ON CREATE SET n += {{{prop_string}}} ON MATCH SET n.updated_at = '{get_timestamp()}'"

    def seed_brands(self) -> bool:
        """Seed Brand entities."""
        logger.info("Seeding %d brands...", len(BRANDS))
        success = True
        for brand in BRANDS:
            query = self._generate_entity_query("Brand", brand)
            success = self.execute_query(query, f"Brand: {brand['name']}") and success
        return success

    def seed_regions(self) -> bool:
        """Seed Region entities."""
        logger.info("Seeding %d regions...", len(REGIONS))
        success = True
        for region in REGIONS:
            query = self._generate_entity_query("Region", region)
            success = self.execute_query(query, f"Region: {region['name']}") and success
        return success

    def seed_kpis(self) -> bool:
        """Seed KPI entities."""
        logger.info("Seeding %d KPIs...", len(KPIS))
        success = True
        for kpi in KPIS:
            query = self._generate_entity_query("KPI", kpi)
            success = self.execute_query(query, f"KPI: {kpi['name']}") and success
        return success

    def seed_agents(self) -> bool:
        """Seed Agent entities."""
        logger.info("Seeding %d agents...", len(AGENTS))
        success = True
        for agent in AGENTS:
            query = self._generate_entity_query("Agent", agent)
            success = self.execute_query(query, f"Agent: {agent['name']}") and success
        return success

    def seed_hcps(self) -> bool:
        """Seed HCP entities."""
        logger.info("Seeding %d HCPs...", len(HCPS))
        success = True
        for hcp in HCPS:
            query = self._generate_entity_query("HCP", hcp)
            success = self.execute_query(query, f"HCP: {hcp['name']}") and success
        return success

    def seed_patients(self) -> bool:
        """Seed Patient entities."""
        logger.info("Seeding %d patients...", len(PATIENTS))
        success = True
        for patient in PATIENTS:
            query = self._generate_entity_query("Patient", patient)
            success = self.execute_query(query, f"Patient: {patient['name']}") and success
        return success

    def seed_treatments(self) -> bool:
        """Seed Treatment entities."""
        logger.info("Seeding %d treatments...", len(TREATMENTS))
        success = True
        for treatment in TREATMENTS:
            query = self._generate_entity_query("Treatment", treatment)
            success = self.execute_query(query, f"Treatment: {treatment['name']}") and success
        return success

    def seed_causal_paths(self) -> bool:
        """Seed CausalPath entities."""
        logger.info("Seeding %d causal paths...", len(CAUSAL_PATHS))
        success = True
        for path in CAUSAL_PATHS:
            query = self._generate_entity_query("CausalPath", path)
            success = self.execute_query(query, f"CausalPath: {path['name']}") and success
        return success

    def seed_triggers(self) -> bool:
        """Seed Trigger entities."""
        logger.info("Seeding %d triggers...", len(TRIGGERS))
        success = True
        for trigger in TRIGGERS:
            query = self._generate_entity_query("Trigger", trigger)
            success = self.execute_query(query, f"Trigger: {trigger['name']}") and success
        return success

    def seed_all_entities(self) -> bool:
        """Seed all entity types."""
        success = True
        success = self.seed_brands() and success
        success = self.seed_regions() and success
        success = self.seed_kpis() and success
        success = self.seed_agents() and success
        success = self.seed_hcps() and success
        success = self.seed_patients() and success
        success = self.seed_treatments() and success
        success = self.seed_causal_paths() and success
        success = self.seed_triggers() and success
        return success

    # =========================================================================
    # RELATIONSHIP SEEDING
    # =========================================================================

    def seed_relationships(self) -> bool:
        """Seed all relationships."""
        logger.info("Seeding %d relationships...", len(RELATIONSHIPS))
        success = True
        ts = get_timestamp()

        for rel in RELATIONSHIPS:
            from_type = rel["from_type"]
            from_id = rel["from_id"]
            to_type = rel["to_type"]
            to_id = rel["to_id"]
            rel_type = rel["rel_type"]

            # Build properties
            props = {k: v for k, v in rel.items() if k not in ["from_type", "from_id", "to_type", "to_id", "rel_type"]}
            props["created_at"] = ts

            prop_items = []
            for k, v in props.items():
                if isinstance(v, str):
                    v_escaped = v.replace("'", "\\'")
                    prop_items.append(f"{k}: '{v_escaped}'")
                elif isinstance(v, bool):
                    prop_items.append(f"{k}: {str(v).lower()}")
                elif v is not None:
                    prop_items.append(f"{k}: {v}")

            prop_string = ", ".join(prop_items) if prop_items else ""

            if prop_string:
                query = f"""
                    MATCH (a:{from_type} {{id: '{from_id}'}}), (b:{to_type} {{id: '{to_id}'}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    ON CREATE SET r += {{{prop_string}}}
                """
            else:
                query = f"""
                    MATCH (a:{from_type} {{id: '{from_id}'}}), (b:{to_type} {{id: '{to_id}'}})
                    MERGE (a)-[r:{rel_type}]->(b)
                """

            success = self.execute_query(query, f"{from_id} -[{rel_type}]-> {to_id}") and success

        return success

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def verify(self) -> Dict[str, Any]:
        """Verify the seeded data."""
        if self.dry_run:
            logger.info("[DRY RUN] Would verify seeded data")
            return {"dry_run": True}

        results = {}

        # Node counts by type
        try:
            result = self.graph.query("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            results["node_counts"] = {row[0]: row[1] for row in result.result_set}
            total_nodes = sum(results["node_counts"].values())
            logger.info("Total nodes: %d", total_nodes)
            for label, count in results["node_counts"].items():
                logger.info("  %s: %d", label, count)
        except Exception as e:
            logger.error("Node count query failed: %s", e)

        # Relationship counts by type
        try:
            result = self.graph.query("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
            """)
            results["relationship_counts"] = {row[0]: row[1] for row in result.result_set}
            total_rels = sum(results["relationship_counts"].values())
            logger.info("Total relationships: %d", total_rels)
            for rel_type, count in results["relationship_counts"].items():
                logger.info("  %s: %d", rel_type, count)
        except Exception as e:
            logger.error("Relationship count query failed: %s", e)

        # Sample causal chain traversal
        try:
            result = self.graph.query("""
                MATCH path = (a:KPI)-[:CAUSES|AFFECTS*1..3]->(b:KPI)
                RETURN a.name as from_kpi, b.name as to_kpi, length(path) as path_length
                LIMIT 5
            """)
            results["causal_paths"] = [
                {"from": row[0], "to": row[1], "length": row[2]}
                for row in result.result_set
            ]
            logger.info("Sample causal paths: %d found", len(results["causal_paths"]))
        except Exception as e:
            logger.error("Causal path query failed: %s", e)

        return results

    # =========================================================================
    # MAIN SEED METHOD
    # =========================================================================

    def seed(self, clear_first: bool = False, skip_schema: bool = False) -> bool:
        """Run the complete seeding process."""
        logger.info("=" * 60)
        logger.info("E2I Semantic Graph Seeding")
        logger.info("Graph: %s", self.graph_name)
        logger.info("=" * 60)

        if not self.connect():
            return False

        if clear_first:
            if not self.clear_graph():
                return False

        if not skip_schema:
            if not self.apply_schema():
                logger.warning("Some schema operations may have failed")

        if not self.seed_all_entities():
            logger.error("Entity seeding had failures")
            return False

        if not self.seed_relationships():
            logger.error("Relationship seeding had failures")
            return False

        # Verify
        verification = self.verify()

        logger.info("=" * 60)
        logger.info("Seeding complete!")
        logger.info("=" * 60)

        return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed FalkorDB e2i_semantic graph with E2I domain data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print queries without executing",
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear existing data before seeding",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip constraint and index creation",
    )

    args = parser.parse_args()

    seeder = SemanticGraphSeeder(dry_run=args.dry_run)

    if seeder.seed(clear_first=args.clear_first, skip_schema=args.skip_schema):
        logger.info("✓ Semantic graph seeding completed successfully")
        sys.exit(0)
    else:
        logger.error("✗ Semantic graph seeding failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
