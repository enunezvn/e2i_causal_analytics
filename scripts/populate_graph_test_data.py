#!/usr/bin/env python3
"""
Populate FalkorDB graph with test data for E2I Causal Analytics.

Creates representative entities and relationships for:
- HCPs (Healthcare Professionals)
- Patients
- Brands (Remibrutinib, Fabhalta, Kisqali)
- KPIs (Key Performance Indicators)
- Causal Paths
- Triggers
- Agent Activities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.semantic_memory import FalkorDBSemanticMemory
from src.memory.episodic_memory import E2IEntityType


def populate_test_data():
    """Populate the graph with test data."""
    print("=" * 60)
    print("E2I Graph Test Data Population")
    print("=" * 60)

    # Initialize semantic memory
    memory = FalkorDBSemanticMemory()
    # Access the graph to trigger initialization
    _ = memory.graph
    print(f"\nConnected to FalkorDB graph")

    # ========================================================================
    # BRANDS
    # ========================================================================
    print("\n--- Adding Brands ---")
    brands = [
        {
            "id": "brand_remibrutinib",
            "name": "Remibrutinib",
            "therapeutic_area": "Immunology",
            "indication": "Chronic Spontaneous Urticaria (CSU)",
            "drug_class": "BTK Inhibitor"
        },
        {
            "id": "brand_fabhalta",
            "name": "Fabhalta",
            "therapeutic_area": "Hematology",
            "indication": "Paroxysmal Nocturnal Hemoglobinuria (PNH)",
            "drug_class": "Factor B Inhibitor"
        },
        {
            "id": "brand_kisqali",
            "name": "Kisqali",
            "therapeutic_area": "Oncology",
            "indication": "HR+/HER2- Breast Cancer",
            "drug_class": "CDK4/6 Inhibitor"
        }
    ]

    for brand in brands:
        memory.add_e2i_entity(
            E2IEntityType.TREATMENT,  # Using TREATMENT for brands
            brand["id"],
            {"name": brand["name"], **{k: v for k, v in brand.items() if k != "id"}}
        )
        print(f"  Added: {brand['name']}")

    # ========================================================================
    # HCPs (Healthcare Professionals)
    # ========================================================================
    print("\n--- Adding HCPs ---")
    hcps = [
        {
            "id": "hcp_001",
            "name": "Dr. Sarah Chen",
            "specialty": "Oncology",
            "sub_specialty": "Breast Cancer",
            "region": "Northeast",
            "priority_tier": 1,
            "decile": 10,
            "adoption_category": "early_adopter"
        },
        {
            "id": "hcp_002",
            "name": "Dr. Michael Roberts",
            "specialty": "Hematology",
            "sub_specialty": "Rare Blood Disorders",
            "region": "West",
            "priority_tier": 1,
            "decile": 9,
            "adoption_category": "early_adopter"
        },
        {
            "id": "hcp_003",
            "name": "Dr. Emily Thompson",
            "specialty": "Dermatology",
            "sub_specialty": "Chronic Urticaria",
            "region": "South",
            "priority_tier": 2,
            "decile": 8,
            "adoption_category": "mainstream"
        },
        {
            "id": "hcp_004",
            "name": "Dr. James Wilson",
            "specialty": "Oncology",
            "sub_specialty": "Medical Oncology",
            "region": "Midwest",
            "priority_tier": 2,
            "decile": 7,
            "adoption_category": "mainstream"
        },
        {
            "id": "hcp_005",
            "name": "Dr. Lisa Martinez",
            "specialty": "Immunology",
            "sub_specialty": "Allergy",
            "region": "Northeast",
            "priority_tier": 3,
            "decile": 6,
            "adoption_category": "late_adopter"
        }
    ]

    for hcp in hcps:
        memory.add_e2i_entity(
            E2IEntityType.HCP,
            hcp["id"],
            {k: v for k, v in hcp.items() if k != "id"}
        )
        print(f"  Added: {hcp['name']} ({hcp['specialty']})")

    # ========================================================================
    # PATIENTS
    # ========================================================================
    print("\n--- Adding Patients ---")
    patients = [
        {
            "id": "patient_001",
            "journey_stage": "adoption",
            "region": "Northeast",
            "risk_score": 0.25,
            "age_group": "55-64",
            "insurance_type": "Commercial"
        },
        {
            "id": "patient_002",
            "journey_stage": "trial",
            "region": "West",
            "risk_score": 0.45,
            "age_group": "45-54",
            "insurance_type": "Medicare"
        },
        {
            "id": "patient_003",
            "journey_stage": "consideration",
            "region": "South",
            "risk_score": 0.60,
            "age_group": "35-44",
            "insurance_type": "Commercial"
        },
        {
            "id": "patient_004",
            "journey_stage": "adherence",
            "region": "Midwest",
            "risk_score": 0.15,
            "age_group": "65+",
            "insurance_type": "Medicare"
        },
        {
            "id": "patient_005",
            "journey_stage": "awareness",
            "region": "Northeast",
            "risk_score": 0.70,
            "age_group": "25-34",
            "insurance_type": "Commercial"
        }
    ]

    for patient in patients:
        memory.add_e2i_entity(
            E2IEntityType.PATIENT,
            patient["id"],
            {k: v for k, v in patient.items() if k != "id"}
        )
        print(f"  Added: {patient['id']} ({patient['journey_stage']}, {patient['region']})")

    # ========================================================================
    # KPIs
    # ========================================================================
    print("\n--- Adding KPIs ---")
    kpis = [
        {
            "id": "kpi_trx",
            "name": "TRx Volume",
            "category": "business",
            "workstream": "WS1",
            "target_value": 1000,
            "unit": "prescriptions"
        },
        {
            "id": "kpi_nrx",
            "name": "NRx Volume",
            "category": "business",
            "workstream": "WS1",
            "target_value": 250,
            "unit": "prescriptions"
        },
        {
            "id": "kpi_conversion",
            "name": "Conversion Rate",
            "category": "engagement",
            "workstream": "WS2",
            "target_value": 0.35,
            "unit": "ratio"
        },
        {
            "id": "kpi_trigger_acceptance",
            "name": "Trigger Acceptance Rate",
            "category": "engagement",
            "workstream": "WS3",
            "target_value": 0.40,
            "unit": "ratio"
        },
        {
            "id": "kpi_market_share",
            "name": "Market Share",
            "category": "business",
            "workstream": "WS1",
            "target_value": 0.25,
            "unit": "percentage"
        }
    ]

    for kpi in kpis:
        memory.add_e2i_entity(
            E2IEntityType.PREDICTION,  # Using PREDICTION as a proxy for KPI
            kpi["id"],
            {"entity_subtype": "KPI", **{k: v for k, v in kpi.items() if k != "id"}}
        )
        print(f"  Added: {kpi['name']} ({kpi['category']})")

    # ========================================================================
    # TRIGGERS
    # ========================================================================
    print("\n--- Adding Triggers ---")
    triggers = [
        {
            "id": "trigger_001",
            "type": "peer_influence",
            "category": "engagement",
            "priority": "high",
            "precision_score": 0.85,
            "expected_impact": 0.25,
            "description": "Dr. Chen influenced by peer network success with Kisqali"
        },
        {
            "id": "trigger_002",
            "type": "patient_profile_match",
            "category": "clinical",
            "priority": "medium",
            "precision_score": 0.78,
            "expected_impact": 0.20,
            "description": "Patient profile matches Fabhalta responder criteria"
        },
        {
            "id": "trigger_003",
            "type": "switching_opportunity",
            "category": "business",
            "priority": "high",
            "precision_score": 0.82,
            "expected_impact": 0.30,
            "description": "Competitor discontinuation creates switching opportunity"
        }
    ]

    for trigger in triggers:
        memory.add_e2i_entity(
            E2IEntityType.TRIGGER,
            trigger["id"],
            {k: v for k, v in trigger.items() if k != "id"}
        )
        print(f"  Added: {trigger['type']} (precision: {trigger['precision_score']})")

    # ========================================================================
    # CAUSAL PATHS
    # ========================================================================
    print("\n--- Adding Causal Paths ---")
    causal_paths = [
        {
            "id": "causal_path_001",
            "description": "Peer influence → HCP engagement → Brand adoption → TRx increase",
            "effect_size": 0.32,
            "confidence": 0.89,
            "method": "DoWhy",
            "validation_status": "validated"
        },
        {
            "id": "causal_path_002",
            "description": "Digital engagement → Trigger acceptance → Prescription behavior change",
            "effect_size": 0.28,
            "confidence": 0.85,
            "method": "EconML",
            "validation_status": "validated"
        },
        {
            "id": "causal_path_003",
            "description": "Patient risk score → HCP prioritization → Intervention timing → Outcome improvement",
            "effect_size": 0.41,
            "confidence": 0.92,
            "method": "CausalML",
            "validation_status": "validated"
        }
    ]

    for path in causal_paths:
        memory.add_e2i_entity(
            E2IEntityType.CAUSAL_PATH,
            path["id"],
            {k: v for k, v in path.items() if k != "id"}
        )
        print(f"  Added: {path['description'][:50]}...")

    # ========================================================================
    # AGENT ACTIVITIES
    # ========================================================================
    print("\n--- Adding Agent Activities ---")
    activities = [
        {
            "id": "activity_001",
            "agent_name": "causal_impact",
            "tier": 2,
            "activity_type": "causal_chain_trace",
            "query": "What drives TRx for Kisqali?",
            "confidence": 0.88
        },
        {
            "id": "activity_002",
            "agent_name": "gap_analyzer",
            "tier": 2,
            "activity_type": "roi_opportunity",
            "query": "Where are the highest ROI opportunities in Northeast?",
            "confidence": 0.91
        },
        {
            "id": "activity_003",
            "agent_name": "orchestrator",
            "tier": 1,
            "activity_type": "query_routing",
            "query": "Analyze Remibrutinib adoption patterns",
            "confidence": 0.95
        }
    ]

    for activity in activities:
        memory.add_e2i_entity(
            E2IEntityType.AGENT_ACTIVITY,
            activity["id"],
            {k: v for k, v in activity.items() if k != "id"}
        )
        print(f"  Added: {activity['agent_name']} - {activity['activity_type']}")

    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================
    print("\n--- Adding Relationships ---")

    # HCPs prescribe Brands
    hcp_prescriptions = [
        ("hcp_001", "brand_kisqali", {"volume_monthly": 45, "market_share": 0.35}),
        ("hcp_002", "brand_fabhalta", {"volume_monthly": 12, "market_share": 0.28}),
        ("hcp_003", "brand_remibrutinib", {"volume_monthly": 30, "market_share": 0.22}),
        ("hcp_004", "brand_kisqali", {"volume_monthly": 25, "market_share": 0.20}),
        ("hcp_005", "brand_remibrutinib", {"volume_monthly": 18, "market_share": 0.15}),
    ]

    for hcp_id, brand_id, props in hcp_prescriptions:
        memory.add_e2i_relationship(
            E2IEntityType.HCP, hcp_id,
            E2IEntityType.TREATMENT, brand_id,
            "PRESCRIBES", props
        )
        print(f"  {hcp_id} PRESCRIBES {brand_id}")

    # Patients treated by HCPs
    patient_treatments = [
        ("patient_001", "hcp_001", {"visit_count": 8, "is_primary": True}),
        ("patient_002", "hcp_002", {"visit_count": 5, "is_primary": True}),
        ("patient_003", "hcp_003", {"visit_count": 3, "is_primary": True}),
        ("patient_004", "hcp_004", {"visit_count": 12, "is_primary": True}),
        ("patient_005", "hcp_005", {"visit_count": 2, "is_primary": False}),
    ]

    for patient_id, hcp_id, props in patient_treatments:
        memory.add_e2i_relationship(
            E2IEntityType.PATIENT, patient_id,
            E2IEntityType.HCP, hcp_id,
            "TREATED_BY", props
        )
        print(f"  {patient_id} TREATED_BY {hcp_id}")

    # Patients prescribed brands
    patient_prescriptions = [
        ("patient_001", "brand_kisqali", {"is_first_line": True, "outcome": "positive"}),
        ("patient_002", "brand_fabhalta", {"is_first_line": True, "outcome": "ongoing"}),
        ("patient_003", "brand_remibrutinib", {"is_first_line": False, "outcome": "pending"}),
        ("patient_004", "brand_kisqali", {"is_first_line": True, "outcome": "positive"}),
    ]

    for patient_id, brand_id, props in patient_prescriptions:
        memory.add_e2i_relationship(
            E2IEntityType.PATIENT, patient_id,
            E2IEntityType.TREATMENT, brand_id,
            "PRESCRIBED", props
        )
        print(f"  {patient_id} PRESCRIBED {brand_id}")

    # HCPs received triggers
    hcp_triggers = [
        ("hcp_001", "trigger_001", {"accepted": True, "action_taken": True, "channel": "digital"}),
        ("hcp_002", "trigger_002", {"accepted": True, "action_taken": True, "channel": "rep_visit"}),
        ("hcp_003", "trigger_003", {"accepted": False, "action_taken": False, "channel": "email"}),
    ]

    for hcp_id, trigger_id, props in hcp_triggers:
        memory.add_e2i_relationship(
            E2IEntityType.HCP, hcp_id,
            E2IEntityType.TRIGGER, trigger_id,
            "RECEIVED", props
        )
        print(f"  {hcp_id} RECEIVED {trigger_id}")

    # Causal paths impact KPIs
    path_impacts = [
        ("causal_path_001", "kpi_trx", {"impact_magnitude": 0.32, "direction": "positive"}),
        ("causal_path_001", "kpi_market_share", {"impact_magnitude": 0.18, "direction": "positive"}),
        ("causal_path_002", "kpi_trigger_acceptance", {"impact_magnitude": 0.28, "direction": "positive"}),
        ("causal_path_003", "kpi_conversion", {"impact_magnitude": 0.41, "direction": "positive"}),
    ]

    for path_id, kpi_id, props in path_impacts:
        memory.add_e2i_relationship(
            E2IEntityType.CAUSAL_PATH, path_id,
            E2IEntityType.PREDICTION, kpi_id,  # Using PREDICTION as proxy for KPI
            "IMPACTS", props
        )
        print(f"  {path_id} IMPACTS {kpi_id}")

    # Agent activities discovered causal paths
    agent_discoveries = [
        ("activity_001", "causal_path_001", {"method": "DoWhy"}),
        ("activity_002", "causal_path_002", {"method": "EconML"}),
    ]

    for activity_id, path_id, props in agent_discoveries:
        memory.add_e2i_relationship(
            E2IEntityType.AGENT_ACTIVITY, activity_id,
            E2IEntityType.CAUSAL_PATH, path_id,
            "DISCOVERED", props
        )
        print(f"  {activity_id} DISCOVERED {path_id}")

    # HCP influence network
    hcp_influences = [
        ("hcp_001", "hcp_004", {"influence_strength": 0.72, "network_type": "academic"}),
        ("hcp_002", "hcp_005", {"influence_strength": 0.45, "network_type": "referral"}),
    ]

    for source_id, target_id, props in hcp_influences:
        memory.add_e2i_relationship(
            E2IEntityType.HCP, source_id,
            E2IEntityType.HCP, target_id,
            "INFLUENCES", props
        )
        print(f"  {source_id} INFLUENCES {target_id}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST DATA POPULATION COMPLETE")
    print("=" * 60)

    # Get stats
    stats = memory.get_graph_stats()
    print(f"\nGraph Statistics:")
    print(f"  Total Nodes: {stats.get('total_nodes', 0)}")
    print(f"  Total Edges: {stats.get('total_edges', 0)}")
    print(f"  Nodes by Type: {stats.get('nodes_by_type', {})}")
    print(f"  Edges by Type: {stats.get('edges_by_type', {})}")

    return True


if __name__ == "__main__":
    try:
        success = populate_test_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
