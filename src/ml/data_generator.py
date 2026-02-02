#!/usr/bin/env python3
"""
E2I Causal Analytics - Complete ML-Compliant Data Generator V3.0

Generates synthetic data for all tables including NEW KPI gap tables:
- user_sessions (WS3 Active Users)
- data_source_tracking (WS1 Cross-source Match, Stacking Lift)
- ml_annotations (WS1 Label Quality/IAA)
- etl_pipeline_metrics (WS1 Time-to-Release)
- hcp_intent_surveys (Brand Intent-to-Prescribe Δ)

Features:
- Chronological train/validation/test/holdout splits
- Patient-level isolation (no patient in multiple splits)
- Complete field population for 100% KPI calculability
- 11-agent tiered architecture support
"""

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List

# Try to import optional packages
try:
    from faker import Faker

    fake = Faker()
    Faker.seed(42)
except ImportError:
    fake = None
    print("Warning: Faker not installed. Using basic random generation.")

import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class SplitConfig:
    """ML split configuration matching database schema."""

    config_name: str = "e2i_pilot_v3"
    config_version: str = "3.0.0"
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    test_ratio: float = 0.15
    holdout_ratio: float = 0.05
    data_start_date: date = date(2025, 1, 1)
    data_end_date: date = date(2025, 10, 31)
    train_end_date: date = date(2025, 6, 30)
    validation_end_date: date = date(2025, 8, 31)
    test_end_date: date = date(2025, 9, 30)
    temporal_gap_days: int = 7


# Volume configuration
NUM_PATIENTS = 200
NUM_HCPS = 50
NUM_USERS = 30  # Dashboard users
EVENTS_PER_PATIENT_RANGE = (3, 12)
PREDICTIONS_PER_PATIENT_RANGE = (1, 5)
TRIGGERS_PER_HCP_RANGE = (5, 20)

# Reference data
BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali", "competitor", "other"]
TARGET_BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali"]
REGIONS = ["northeast", "south", "midwest", "west"]
SPECIALTIES = [
    "Allergy/Immunology",
    "Hematology",
    "Oncology",
    "Dermatology",
    "Internal Medicine",
    "Rheumatology",
]
EVENT_TYPES = [
    "diagnosis",
    "prescription",
    "lab_test",
    "procedure",
    "consultation",
    "hospitalization",
]
PREDICTION_TYPES = ["trigger", "propensity", "risk", "churn", "next_best_action"]
JOURNEY_STAGES = [
    "diagnosis",
    "initial_treatment",
    "treatment_optimization",
    "maintenance",
    "treatment_switch",
]
JOURNEY_STATUSES = ["active", "stable", "transitioning", "completed"]
PRIORITY_LEVELS = ["critical", "high", "medium", "low"]
DATA_SOURCES = ["IQVIA_APLD", "IQVIA_LAAD", "HealthVerity", "Komodo", "Veeva"]
AGENT_NAMES = [
    "orchestrator",
    "causal_impact",
    "gap_analyzer",
    "heterogeneous_optimizer",
    "drift_monitor",
    "experiment_designer",
    "health_score",
    "prediction_synthesizer",
    "resource_optimizer",
    "explainer",
    "feedback_learner",
]
AGENT_TIERS = {
    "orchestrator": "coordination",
    "causal_impact": "causal_analytics",
    "gap_analyzer": "causal_analytics",
    "heterogeneous_optimizer": "causal_analytics",
    "drift_monitor": "monitoring",
    "experiment_designer": "monitoring",
    "health_score": "monitoring",
    "prediction_synthesizer": "ml_predictions",
    "resource_optimizer": "ml_predictions",
    "explainer": "self_improvement",
    "feedback_learner": "self_improvement",
}
WORKSTREAMS = ["WS1", "WS2", "WS3"]
STATES_BY_REGION = {
    "northeast": ["NY", "MA", "PA", "NJ", "CT"],
    "south": ["TX", "FL", "GA", "NC", "VA"],
    "midwest": ["IL", "OH", "MI", "IN", "WI"],
    "west": ["CA", "WA", "AZ", "CO", "OR"],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_id(prefix: str, index: int) -> str:
    """Generate a consistent ID."""
    return f"{prefix}_{index:06d}"


def random_date_in_range(start: date, end: date) -> date:
    """Generate random date in range."""
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def random_datetime_in_range(start: date, end: date) -> datetime:
    """Generate random datetime in range."""
    d = random_date_in_range(start, end)
    return datetime.combine(d, datetime.min.time()) + timedelta(
        hours=random.randint(0, 23), minutes=random.randint(0, 59)
    )


def assign_split(journey_date: date, config: SplitConfig) -> str:
    """Assign data split based on date."""
    if journey_date <= config.train_end_date:
        return "train"
    elif journey_date <= config.validation_end_date:
        return "validation"
    elif journey_date <= config.test_end_date:
        return "test"
    else:
        return "holdout"


def generate_hash(value: str) -> str:
    """Generate consistent hash for anonymization."""
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def random_jsonb(keys: List[str], value_type: str = "float") -> Dict:
    """Generate random JSONB object."""
    result = {}
    for key in keys:
        if value_type == "float":
            result[key] = round(random.uniform(0, 1), 4)
        elif value_type == "int":
            result[key] = random.randint(1, 100)
    return result


def json_serial(obj):
    """JSON serializer for dates and other objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# =============================================================================
# DATA GENERATORS
# =============================================================================


class E2IDataGenerator:
    """Generates complete E2I dataset with all KPI gaps filled."""

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()
        self.split_config_id = str(uuid.uuid4())

        # Data containers
        self.hcp_profiles = []
        self.patient_journeys = []
        self.treatment_events = []
        self.ml_predictions = []
        self.triggers = []
        self.agent_activities = []
        self.business_metrics = []
        self.causal_paths = []

        # New KPI gap tables
        self.user_sessions = []
        self.data_source_tracking = []
        self.ml_annotations = []
        self.etl_pipeline_metrics = []
        self.hcp_intent_surveys = []
        self.reference_universe = []

        # Tracking
        self.patient_splits = {}  # patient_id -> split

    def generate_all(self):
        """Generate complete dataset."""
        print("Generating E2I Complete Dataset V3.0...")
        print(f"Split config ID: {self.split_config_id}")

        self._generate_reference_universe()
        self._generate_hcp_profiles()
        self._generate_patient_journeys()
        self._generate_treatment_events()
        self._generate_ml_predictions()
        self._generate_triggers()
        self._generate_agent_activities()
        self._generate_business_metrics()
        self._generate_causal_paths()

        # New KPI gap tables
        self._generate_user_sessions()
        self._generate_data_source_tracking()
        self._generate_ml_annotations()
        self._generate_etl_pipeline_metrics()
        self._generate_hcp_intent_surveys()

        self._generate_preprocessing_metadata()

        print("\nGeneration complete!")
        self._print_summary()

    def _generate_reference_universe(self):
        """Generate reference universe for coverage calculations."""
        print("  - Generating reference universe...")

        for brand in TARGET_BRANDS:
            for region in REGIONS:
                for specialty in SPECIALTIES:
                    self.reference_universe.append(
                        {
                            "universe_id": str(uuid.uuid4()),
                            "universe_type": "hcp",
                            "brand": brand,
                            "region": region,
                            "specialty": specialty,
                            "total_count": random.randint(500, 2000),
                            "target_count": random.randint(100, 500),
                            "effective_date": self.config.data_start_date.isoformat(),
                            "expiration_date": None,
                            "data_source": "IQVIA_APLD",
                            "methodology_notes": "Derived from prescribing data",
                            "created_at": datetime.now().isoformat(),
                        }
                    )

        # Patient universe
        for brand in TARGET_BRANDS:
            for region in REGIONS:
                self.reference_universe.append(
                    {
                        "universe_id": str(uuid.uuid4()),
                        "universe_type": "patient",
                        "brand": brand,
                        "region": region,
                        "specialty": None,
                        "total_count": random.randint(50000, 200000),
                        "target_count": random.randint(10000, 50000),
                        "effective_date": self.config.data_start_date.isoformat(),
                        "expiration_date": None,
                        "data_source": "HealthVerity",
                        "methodology_notes": "Claims-based patient identification",
                        "created_at": datetime.now().isoformat(),
                    }
                )

    def _generate_hcp_profiles(self):
        """Generate HCP profiles."""
        print(f"  - Generating {NUM_HCPS} HCP profiles...")

        for i in range(NUM_HCPS):
            region = random.choice(REGIONS)
            state = random.choice(STATES_BY_REGION[region])
            specialty = random.choice(SPECIALTIES)

            self.hcp_profiles.append(
                {
                    "hcp_id": generate_id("HCP", i),
                    "npi": f"{random.randint(1000000000, 9999999999)}",
                    "first_name": fake.first_name() if fake else f"Doctor{i}",
                    "last_name": fake.last_name() if fake else f"Smith{i}",
                    "specialty": specialty,
                    "sub_specialty": random.choice([None, f"{specialty} - Subspecialty"]),
                    "practice_type": random.choice(
                        ["Academic", "Private", "Hospital", "Community"]
                    ),
                    "practice_size": random.choice(["Solo", "Small", "Medium", "Large"]),
                    "geographic_region": region,
                    "state": state,
                    "city": fake.city() if fake else f"City{i}",
                    "zip_code": f"{random.randint(10000, 99999)}",
                    "priority_tier": random.randint(1, 5),
                    "decile": random.randint(1, 10),
                    "total_patient_volume": random.randint(100, 5000),
                    "target_patient_volume": random.randint(20, 500),
                    "prescribing_volume": random.randint(50, 2000),
                    "years_experience": random.randint(3, 35),
                    "affiliation_primary": fake.company() if fake else f"Hospital{i}",
                    "affiliation_secondary": [
                        fake.company() if fake else f"Clinic{j}"
                        for j in range(random.randint(0, 2))
                    ],
                    "digital_engagement_score": round(random.uniform(0.1, 0.99), 2),
                    "preferred_channel": random.choice(["email", "app", "phone", "in_person"]),
                    "last_interaction_date": random_date_in_range(
                        self.config.data_start_date, self.config.data_end_date
                    ).isoformat(),
                    "interaction_frequency": round(random.uniform(0.5, 10.0), 1),
                    "influence_network_size": random.randint(5, 100),
                    "peer_influence_score": round(random.uniform(0.2, 0.95), 2),
                    "adoption_category": random.choice(
                        ["early_adopter", "mainstream", "late_adopter"]
                    ),
                    "coverage_status": random.random() > 0.1,
                    "territory_id": f"TER_{random.randint(1, 20):03d}",
                    "sales_rep_id": f"REP_{random.randint(1, 50):03d}",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
            )

    def _generate_patient_journeys(self):
        """Generate patient journeys with source tracking fields."""
        print(f"  - Generating {NUM_PATIENTS} patient journeys...")

        for i in range(NUM_PATIENTS):
            # Determine journey start date first (for split assignment)
            journey_start = random_date_in_range(
                self.config.data_start_date, self.config.data_end_date - timedelta(days=30)
            )

            # Assign split based on date
            data_split = assign_split(journey_start, self.config)
            patient_id = generate_id("PAT", i)
            self.patient_splits[patient_id] = data_split

            region = random.choice(REGIONS)
            state = random.choice(STATES_BY_REGION[region])
            journey_end = journey_start + timedelta(days=random.randint(30, 180))

            # Source tracking (NEW for WS1 gaps)
            primary_source = random.choice(DATA_SOURCES)
            num_sources_matched = random.randint(1, 3)
            sources_matched = random.sample(
                DATA_SOURCES, min(num_sources_matched, len(DATA_SOURCES))
            )
            if primary_source not in sources_matched:
                sources_matched[0] = primary_source

            source_timestamp = datetime.combine(journey_start, datetime.min.time()) - timedelta(
                hours=random.randint(24, 336)  # 1-14 days before ingestion
            )
            ingestion_timestamp = datetime.combine(journey_start, datetime.min.time())
            data_lag_hours = int((ingestion_timestamp - source_timestamp).total_seconds() / 3600)

            self.patient_journeys.append(
                {
                    "patient_journey_id": generate_id("PJ", i),
                    "patient_id": patient_id,
                    "patient_hash": generate_hash(patient_id),
                    "journey_start_date": journey_start.isoformat(),
                    "journey_end_date": journey_end.isoformat() if random.random() > 0.3 else None,
                    "journey_duration_days": (
                        (journey_end - journey_start).days if random.random() > 0.3 else None
                    ),
                    "journey_stage": random.choice(JOURNEY_STAGES),
                    "journey_status": random.choice(JOURNEY_STATUSES),
                    "primary_diagnosis_code": f"ICD10_{random.choice(['C50', 'D89', 'L40', 'M05'])}",
                    "primary_diagnosis_desc": random.choice(
                        [
                            "Breast Cancer",
                            "Autoimmune Disorder",
                            "Psoriasis",
                            "Rheumatoid Arthritis",
                        ]
                    ),
                    "secondary_diagnosis_codes": [
                        f"ICD10_{random.randint(100, 999)}" for _ in range(random.randint(0, 3))
                    ],
                    "brand": random.choice(TARGET_BRANDS),
                    "age_group": random.choice(["18-34", "35-49", "50-64", "65+"]),
                    "gender": random.choice(["M", "F"]),
                    "geographic_region": region,
                    "state": state,
                    "zip_code": f"{random.randint(10000, 99999)}",
                    "insurance_type": random.choice(
                        ["Commercial", "Medicare", "Medicaid", "Self-Pay"]
                    ),
                    "data_quality_score": round(random.uniform(0.7, 0.99), 2),
                    "comorbidities": random.sample(
                        ["Diabetes", "Hypertension", "Obesity", "Heart Disease"],
                        random.randint(0, 3),
                    ),
                    "risk_score": round(random.uniform(0.1, 0.9), 2),
                    # NEW: Source tracking fields (WS1 Cross-source Match, Stacking Lift, Data Lag)
                    "data_source": primary_source,
                    "data_sources_matched": sources_matched,
                    "source_match_confidence": round(random.uniform(0.8, 0.99), 2),
                    "source_stacking_flag": len(sources_matched) > 1,
                    "source_combination_method": (
                        "deterministic_match" if len(sources_matched) > 1 else None
                    ),
                    "source_timestamp": source_timestamp.isoformat(),
                    "ingestion_timestamp": ingestion_timestamp.isoformat(),
                    "data_lag_hours": data_lag_hours,
                    # ML Split tracking
                    "data_split": data_split,
                    "split_config_id": self.split_config_id,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
            )

    def _generate_treatment_events(self):
        """Generate treatment events for each patient."""
        print("  - Generating treatment events...")

        event_count = 0
        for pj in self.patient_journeys:
            num_events = random.randint(*EVENTS_PER_PATIENT_RANGE)
            journey_start = date.fromisoformat(pj["journey_start_date"])

            for j in range(num_events):
                event_date = journey_start + timedelta(days=random.randint(0, 90))

                self.treatment_events.append(
                    {
                        "treatment_event_id": generate_id("TE", event_count),
                        "patient_journey_id": pj["patient_journey_id"],
                        "patient_id": pj["patient_id"],
                        "hcp_id": random.choice(self.hcp_profiles)["hcp_id"],
                        "event_date": event_date.isoformat(),
                        "event_type": random.choice(EVENT_TYPES),
                        "event_subtype": f"subtype_{random.randint(1, 5)}",
                        "brand": pj["brand"],
                        "drug_ndc": f"NDC{random.randint(10000, 99999)}",
                        "drug_name": f"{pj['brand']}_Drug",
                        "drug_class": random.choice(
                            ["BTK Inhibitor", "Complement Inhibitor", "CDK4/6 Inhibitor"]
                        ),
                        "dosage": f"{random.choice([50, 100, 200, 400])}mg",
                        "duration_days": random.randint(7, 90),
                        "icd_codes": [
                            f"ICD10_{random.randint(100, 999)}" for _ in range(random.randint(1, 3))
                        ],
                        "cpt_codes": [
                            f"CPT{random.randint(10000, 99999)}"
                            for _ in range(random.randint(0, 2))
                        ],
                        "loinc_codes": [
                            f"LOINC{random.randint(10000, 99999)}"
                            for _ in range(random.randint(0, 2))
                        ],
                        "lab_values": (
                            {"value": round(random.uniform(0, 100), 2), "unit": "mg/dL"}
                            if random.random() > 0.5
                            else {}
                        ),
                        "location_type": random.choice(["Office", "Hospital", "Clinic", "Home"]),
                        "facility_id": f"FAC_{random.randint(1, 100):03d}",
                        "cost": round(random.uniform(100, 5000), 2),
                        "outcome_indicator": random.choice(
                            ["improved", "stable", "declined", None]
                        ),
                        "adverse_event_flag": random.random() < 0.05,
                        "discontinuation_flag": random.random() < 0.1,
                        "discontinuation_reason": random.choice(
                            ["Side effects", "Lack of efficacy", "Cost", None]
                        ),
                        "sequence_number": j + 1,
                        "days_from_diagnosis": random.randint(0, 365),
                        "previous_treatment": (
                            f"PrevDrug_{random.randint(1, 10)}" if j > 0 else None
                        ),
                        "next_treatment": (
                            f"NextDrug_{random.randint(1, 10)}" if random.random() > 0.5 else None
                        ),
                        # Source tracking
                        "data_source": pj["data_source"],
                        "source_timestamp": pj["source_timestamp"],
                        # ML Split tracking
                        "data_split": pj["data_split"],
                        "split_config_id": self.split_config_id,
                        "created_at": datetime.now().isoformat(),
                    }
                )
                event_count += 1

    def _generate_ml_predictions(self):
        """Generate ML predictions with ALL metrics including gaps."""
        print("  - Generating ML predictions...")

        pred_count = 0
        for pj in self.patient_journeys:
            num_preds = random.randint(*PREDICTIONS_PER_PATIENT_RANGE)
            journey_start = date.fromisoformat(pj["journey_start_date"])

            for _j in range(num_preds):
                pred_timestamp = datetime.combine(
                    journey_start + timedelta(days=random.randint(1, 60)), datetime.min.time()
                ) + timedelta(hours=random.randint(0, 23))

                # Generate comprehensive metrics including GAP FILLS
                auc = round(random.uniform(0.65, 0.95), 3)
                precision = round(random.uniform(0.6, 0.9), 3)
                recall = round(random.uniform(0.6, 0.9), 3)

                self.ml_predictions.append(
                    {
                        "prediction_id": generate_id("PRED", pred_count),
                        "model_version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                        "model_type": random.choice(
                            ["XGBoost", "LightGBM", "CausalForest", "DoWhy"]
                        ),
                        "prediction_timestamp": pred_timestamp.isoformat(),
                        "patient_id": pj["patient_id"],
                        "hcp_id": random.choice(self.hcp_profiles)["hcp_id"],
                        "prediction_type": random.choice(PREDICTION_TYPES),
                        "prediction_value": round(random.uniform(0, 1), 4),
                        "prediction_class": random.choice(["high_risk", "medium_risk", "low_risk"]),
                        "confidence_score": round(random.uniform(0.5, 0.99), 4),
                        "probability_scores": {
                            "class_0": round(random.uniform(0, 0.5), 4),
                            "class_1": round(random.uniform(0.5, 1), 4),
                        },
                        "feature_importance": random_jsonb(
                            ["age", "comorbidities", "prior_rx", "hcp_tier"], "float"
                        ),
                        "shap_values": random_jsonb(
                            ["age", "comorbidities", "prior_rx", "hcp_tier"], "float"
                        ),
                        "top_features": [
                            {"name": "prior_rx", "importance": round(random.uniform(0.1, 0.3), 4)},
                            {
                                "name": "comorbidities",
                                "importance": round(random.uniform(0.05, 0.2), 4),
                            },
                        ],
                        # Standard metrics
                        "model_auc": auc,
                        "model_precision": precision,
                        "model_recall": recall,
                        "calibration_score": round(random.uniform(0.8, 0.99), 3),
                        # NEW: Gap-filling metrics (WS1 Model Performance)
                        "model_pr_auc": round(random.uniform(0.5, 0.85), 3),  # PR-AUC gap
                        "rank_metrics": {  # Recall@Top-K gap
                            "recall_at_5": round(random.uniform(0.7, 0.95), 3),
                            "recall_at_10": round(random.uniform(0.8, 0.97), 3),
                            "recall_at_20": round(random.uniform(0.85, 0.99), 3),
                            "precision_at_5": round(random.uniform(0.6, 0.9), 3),
                            "precision_at_10": round(random.uniform(0.5, 0.85), 3),
                        },
                        "brier_score": round(random.uniform(0.05, 0.25), 4),  # Brier Score gap
                        "fairness_metrics": {
                            "demographic_parity": round(random.uniform(0.9, 1.0), 3),
                            "equalized_odds": round(random.uniform(0.85, 1.0), 3),
                        },
                        "explanation_text": f"Patient shows {random.choice(['elevated', 'moderate', 'low'])} risk based on {random.choice(['prior treatment history', 'comorbidity profile', 'HCP engagement patterns'])}.",
                        "treatment_effect_estimate": round(random.uniform(0.05, 0.3), 3),
                        "heterogeneous_effect": round(random.uniform(-0.1, 0.2), 3),
                        "segment_assignment": f"segment_{random.randint(1, 5)}",
                        "causal_confidence": round(random.uniform(0.6, 0.95), 3),
                        "counterfactual_outcome": round(random.uniform(0.2, 0.8), 3),
                        "features_available_at_prediction": {
                            "feature_count": random.randint(20, 50),
                            "feature_names": [
                                "age",
                                "gender",
                                "region",
                                "prior_rx_count",
                                "days_since_dx",
                            ],
                        },
                        # ML Split tracking
                        "data_split": pj["data_split"],
                        "split_config_id": self.split_config_id,
                        "created_at": datetime.now().isoformat(),
                    }
                )
                pred_count += 1

    def _generate_triggers(self):
        """Generate triggers with change tracking for CFR."""
        print("  - Generating triggers...")

        trigger_count = 0
        for hcp in self.hcp_profiles:
            num_triggers = random.randint(*TRIGGERS_PER_HCP_RANGE)

            previous_trigger_id = None
            for j in range(num_triggers):
                trigger_date = random_datetime_in_range(
                    self.config.data_start_date, self.config.data_end_date
                )

                # Assign split based on date
                data_split = assign_split(trigger_date.date(), self.config)

                # Change tracking (NEW for WS2 Change-Fail Rate)
                is_change = j > 0 and random.random() > 0.7
                change_type = (
                    random.choice(["update", "escalation", "downgrade"]) if is_change else "new"
                )
                change_failed = random.random() < 0.15 if is_change else False

                trigger_id = generate_id("TRG", trigger_count)

                # Determine patient (ensure same split)
                eligible_patients = [
                    p for p in self.patient_journeys if p["data_split"] == data_split
                ]
                if not eligible_patients:
                    eligible_patients = self.patient_journeys
                patient = random.choice(eligible_patients)

                delivery_timestamp = trigger_date + timedelta(hours=random.randint(1, 24))
                view_timestamp = (
                    delivery_timestamp + timedelta(hours=random.randint(1, 48))
                    if random.random() > 0.3
                    else None
                )

                self.triggers.append(
                    {
                        "trigger_id": trigger_id,
                        "patient_id": patient["patient_id"],
                        "hcp_id": hcp["hcp_id"],
                        "trigger_timestamp": trigger_date.isoformat(),
                        "trigger_type": random.choice(
                            ["alert", "recommendation", "insight", "next_best_action"]
                        ),
                        "priority": random.choice(PRIORITY_LEVELS),
                        "confidence_score": round(random.uniform(0.6, 0.99), 3),
                        "lead_time_days": random.randint(7, 60),
                        "expiration_date": (
                            trigger_date.date() + timedelta(days=random.randint(14, 90))
                        ).isoformat(),
                        "delivery_channel": random.choice(["email", "app", "crm", "phone"]),
                        "delivery_status": random.choice(
                            ["pending", "delivered", "viewed", "expired"]
                        ),
                        "delivery_timestamp": delivery_timestamp.isoformat(),
                        "view_timestamp": view_timestamp.isoformat() if view_timestamp else None,
                        "acceptance_status": random.choice(
                            ["accepted", "rejected", "pending", None]
                        ),
                        "acceptance_timestamp": (
                            (view_timestamp + timedelta(hours=random.randint(1, 72))).isoformat()
                            if view_timestamp and random.random() > 0.4
                            else None
                        ),
                        "action_taken": random.choice(
                            ["called_patient", "scheduled_visit", "sent_info", None]
                        ),
                        "action_timestamp": (
                            (view_timestamp + timedelta(days=random.randint(1, 7))).isoformat()
                            if view_timestamp and random.random() > 0.5
                            else None
                        ),
                        "false_positive_flag": random.random() < 0.1,
                        "trigger_reason": f"Based on {random.choice(['prescription pattern', 'lab trend', 'engagement signal', 'competitive activity'])}",
                        "causal_chain": {
                            "nodes": ["HCP_Engagement", "Patient_Adherence", "Outcome"],
                            "effects": [round(random.uniform(0.1, 0.5), 3) for _ in range(2)],
                        },
                        "supporting_evidence": {
                            "confidence": round(random.uniform(0.7, 0.95), 3),
                            "sources": random.sample(DATA_SOURCES, 2),
                        },
                        "recommended_action": random.choice(
                            [
                                "Schedule follow-up",
                                "Review patient file",
                                "Contact for adherence check",
                            ]
                        ),
                        "outcome_tracked": random.random() > 0.3,
                        "outcome_value": (
                            round(random.uniform(0.4, 0.9), 3) if random.random() > 0.3 else None
                        ),
                        # NEW: Change tracking (WS2 Change-Fail Rate gap)
                        "previous_trigger_id": previous_trigger_id if is_change else None,
                        "change_type": change_type,
                        "change_reason": (
                            f"Updated based on {random.choice(['new data', 'model refresh', 'user feedback'])}"
                            if is_change
                            else None
                        ),
                        "change_timestamp": trigger_date.isoformat() if is_change else None,
                        "change_failed": change_failed,
                        "change_outcome_delta": (
                            round(random.uniform(-0.2, 0.3), 3) if is_change else None
                        ),
                        # ML Split tracking
                        "data_split": data_split,
                        "split_config_id": self.split_config_id,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    }
                )

                previous_trigger_id = trigger_id
                trigger_count += 1

    def _generate_agent_activities(self):
        """Generate agent activities with tier support."""
        print("  - Generating agent activities...")

        activity_count = 0
        current_date = self.config.data_start_date

        while current_date <= self.config.data_end_date:
            # Generate activities for each day
            for agent_name in AGENT_NAMES:
                if random.random() > 0.3:  # Not all agents active every day
                    activity_timestamp = datetime.combine(
                        current_date, datetime.min.time()
                    ) + timedelta(hours=random.randint(6, 22), minutes=random.randint(0, 59))

                    data_split = assign_split(current_date, self.config)

                    self.agent_activities.append(
                        {
                            "activity_id": generate_id("AA", activity_count),
                            "agent_name": agent_name,
                            "agent_tier": AGENT_TIERS[agent_name],
                            "activity_timestamp": activity_timestamp.isoformat(),
                            "activity_type": random.choice(
                                ["analysis", "recommendation", "action", "alert", "experiment"]
                            ),
                            "workstream": random.choice(WORKSTREAMS),
                            "processing_duration_ms": random.randint(100, 30000),
                            "input_data": {
                                "patient_count": random.randint(10, 1000),
                                "hcp_count": random.randint(5, 100),
                            },
                            "records_processed": random.randint(100, 10000),
                            "time_window": random.choice(["1h", "6h", "24h", "7d"]),
                            "analysis_results": {
                                "insights_found": random.randint(1, 20),
                                "anomalies_detected": random.randint(0, 5),
                            },
                            "causal_paths_analyzed": random.randint(0, 50),
                            "confidence_level": round(random.uniform(0.7, 0.99), 3),
                            "recommendations": [
                                {
                                    "action": f"action_{i}",
                                    "priority": random.choice(PRIORITY_LEVELS),
                                }
                                for i in range(random.randint(0, 3))
                            ],
                            "actions_initiated": [
                                {"type": f"action_type_{i}", "status": "initiated"}
                                for i in range(random.randint(0, 2))
                            ],
                            "impact_estimate": round(random.uniform(1000, 500000), 2),
                            "roi_estimate": round(random.uniform(1.5, 10.0), 2),
                            "status": random.choice(
                                ["completed", "completed", "completed", "failed", "pending"]
                            ),
                            "error_message": None if random.random() > 0.05 else "Timeout error",
                            "resource_usage": {
                                "cpu_seconds": round(random.uniform(0.1, 10.0), 2),
                                "memory_mb": random.randint(50, 500),
                            },
                            # ML Split tracking
                            "data_split": data_split,
                            "split_config_id": self.split_config_id,
                            "created_at": datetime.now().isoformat(),
                        }
                    )
                    activity_count += 1

            current_date += timedelta(days=1)

    def _generate_business_metrics(self):
        """Generate business metrics."""
        print("  - Generating business metrics...")

        metric_count = 0
        metric_types = ["revenue", "cost", "efficiency", "adoption", "engagement"]
        metric_names = [
            "TRx",
            "NBRx",
            "HCP_Coverage",
            "Patient_Touch_Rate",
            "Conversion_Rate",
            "ROI",
        ]

        current_date = self.config.data_start_date
        while current_date <= self.config.data_end_date:
            for brand in TARGET_BRANDS:
                for region in REGIONS:
                    for metric_name in metric_names:
                        data_split = assign_split(current_date, self.config)

                        value = round(random.uniform(10000, 1000000), 2)
                        target = value * random.uniform(0.9, 1.2)

                        self.business_metrics.append(
                            {
                                "metric_id": f"BM_{current_date.isoformat()}_{brand}_{region}_{metric_name}",
                                "metric_date": current_date.isoformat(),
                                "metric_type": random.choice(metric_types),
                                "metric_name": metric_name,
                                "brand": brand,
                                "region": region,
                                "value": value,
                                "target": round(target, 2),
                                "achievement_rate": (
                                    round(value / target, 3) if target > 0 else None
                                ),
                                "year_over_year_change": round(random.uniform(-0.2, 0.3), 3),
                                "month_over_month_change": round(random.uniform(-0.1, 0.15), 3),
                                "roi": round(random.uniform(1.5, 8.0), 2),
                                "statistical_significance": round(random.uniform(0.8, 0.99), 3),
                                "confidence_interval_lower": round(value * 0.9, 2),
                                "confidence_interval_upper": round(value * 1.1, 2),
                                "sample_size": random.randint(100, 5000),
                                # ML Split tracking
                                "data_split": data_split,
                                "split_config_id": self.split_config_id,
                                "created_at": datetime.now().isoformat(),
                            }
                        )
                        metric_count += 1

            current_date += timedelta(days=7)  # Weekly metrics

    def _generate_causal_paths(self):
        """Generate discovered causal paths."""
        print("  - Generating causal paths...")

        path_nodes = [
            "HCP_Engagement",
            "Patient_Adherence",
            "Treatment_Response",
            "Outcome_Improvement",
            "Cost_Reduction",
            "Market_Share",
        ]
        methods = ["DoWhy", "EconML", "CausalML", "NetworkX", "Granger"]

        for i in range(50):
            discovery_date = random_date_in_range(
                self.config.data_start_date, self.config.data_end_date
            )
            data_split = assign_split(discovery_date, self.config)

            path_length = random.randint(2, 5)
            nodes = random.sample(path_nodes, path_length)

            self.causal_paths.append(
                {
                    "path_id": generate_id("CP", i),
                    "discovery_date": discovery_date.isoformat(),
                    "causal_chain": {
                        "nodes": nodes,
                        "edges": [
                            {
                                "from": nodes[j],
                                "to": nodes[j + 1],
                                "effect": round(random.uniform(0.1, 0.5), 3),
                            }
                            for j in range(len(nodes) - 1)
                        ],
                    },
                    "start_node": nodes[0],
                    "end_node": nodes[-1],
                    "intermediate_nodes": nodes[1:-1],
                    "path_length": path_length,
                    "causal_effect_size": round(random.uniform(0.05, 0.4), 3),
                    "confidence_level": round(random.uniform(0.7, 0.95), 3),
                    "method_used": random.choice(methods),
                    "confounders_controlled": random.sample(
                        ["age", "region", "comorbidities", "prior_treatment"], random.randint(1, 3)
                    ),
                    "mediators_identified": (
                        [nodes[i] for i in range(1, len(nodes) - 1)] if len(nodes) > 2 else []
                    ),
                    "interaction_effects": {
                        "interactions": [
                            {"vars": ["A", "B"], "effect": round(random.uniform(-0.1, 0.2), 3)}
                        ]
                    },
                    "time_lag_days": random.randint(7, 90),
                    "validation_status": random.choice(["validated", "pending", "needs_review"]),
                    "business_impact_estimate": round(random.uniform(50000, 2000000), 2),
                    # ML Split tracking
                    "data_split": data_split,
                    "split_config_id": self.split_config_id,
                    "created_at": datetime.now().isoformat(),
                }
            )

    def _generate_user_sessions(self):
        """Generate user sessions for Active Users KPI."""
        print("  - Generating user sessions...")

        session_count = 0
        user_ids = [f"USER_{i:03d}" for i in range(NUM_USERS)]
        user_roles = ["analyst", "manager", "director", "rep", "admin"]

        current_date = self.config.data_start_date
        while current_date <= self.config.data_end_date:
            # Each user may have 0-3 sessions per day
            for user_id in user_ids:
                if random.random() > 0.6:  # 40% chance of session
                    num_sessions = random.randint(1, 3)
                    for _ in range(num_sessions):
                        session_start = datetime.combine(
                            current_date, datetime.min.time()
                        ) + timedelta(hours=random.randint(6, 20), minutes=random.randint(0, 59))
                        session_duration = random.randint(60, 7200)  # 1 min to 2 hours
                        session_end = session_start + timedelta(seconds=session_duration)

                        self.user_sessions.append(
                            {
                                "session_id": str(uuid.uuid4()),
                                "user_id": user_id,
                                "user_email": f"{user_id.lower()}@company.com",
                                "user_role": random.choice(user_roles),
                                "user_region": random.choice(REGIONS),
                                "user_territory_id": f"TER_{random.randint(1, 20):03d}",
                                "session_start": session_start.isoformat(),
                                "session_end": session_end.isoformat(),
                                "session_duration_seconds": session_duration,
                                "page_views": random.randint(3, 50),
                                "queries_executed": random.randint(1, 20),
                                "triggers_viewed": random.randint(0, 15),
                                "actions_taken": random.randint(0, 5),
                                "exports_downloaded": random.randint(0, 3),
                                "device_type": random.choice(["desktop", "laptop", "tablet"]),
                                "browser": random.choice(["Chrome", "Safari", "Firefox", "Edge"]),
                                "ip_hash": generate_hash(f"{user_id}_{current_date}"),
                                "engagement_score": round(random.uniform(0.3, 0.99), 2),
                                "created_at": datetime.now().isoformat(),
                            }
                        )
                        session_count += 1

            current_date += timedelta(days=1)

    def _generate_data_source_tracking(self):
        """Generate data source tracking for Cross-source Match and Stacking Lift."""
        print("  - Generating data source tracking...")

        current_date = self.config.data_start_date
        while current_date <= self.config.data_end_date:
            for source in DATA_SOURCES:
                records_received = random.randint(10000, 100000)
                records_matched = int(records_received * random.uniform(0.85, 0.98))
                stacking_eligible = int(records_received * random.uniform(0.3, 0.6))
                stacking_applied = int(stacking_eligible * random.uniform(0.7, 0.95))

                self.data_source_tracking.append(
                    {
                        "tracking_id": str(uuid.uuid4()),
                        "tracking_date": current_date.isoformat(),
                        "source_name": source,
                        "source_type": (
                            "claims" if "IQVIA" in source else random.choice(["lab", "emr", "crm"])
                        ),
                        "records_received": records_received,
                        "records_matched": records_matched,
                        "records_unique": records_received - random.randint(100, 1000),
                        "match_rate_vs_iqvia": (
                            round(random.uniform(0.8, 0.95), 3) if source != "IQVIA_APLD" else 1.0
                        ),
                        "match_rate_vs_healthverity": (
                            round(random.uniform(0.75, 0.92), 3)
                            if source != "HealthVerity"
                            else 1.0
                        ),
                        "match_rate_vs_komodo": (
                            round(random.uniform(0.78, 0.93), 3) if source != "Komodo" else 1.0
                        ),
                        "match_rate_vs_veeva": (
                            round(random.uniform(0.7, 0.88), 3) if source != "Veeva" else 1.0
                        ),
                        "stacking_eligible_records": stacking_eligible,
                        "stacking_applied_records": stacking_applied,
                        "stacking_lift_percentage": round(random.uniform(5, 25), 2),
                        "source_combination_flags": {
                            "IQVIA+HV": random.randint(1000, 10000),
                            "IQVIA+Komodo": random.randint(500, 5000),
                            "HV+Veeva": random.randint(200, 2000),
                        },
                        "data_quality_score": round(random.uniform(0.85, 0.99), 2),
                        "created_at": datetime.now().isoformat(),
                    }
                )

            current_date += timedelta(days=1)

    def _generate_ml_annotations(self):
        """Generate ML annotations for Label Quality/IAA."""
        print("  - Generating ML annotations...")

        annotation_types = [
            "diagnosis_validation",
            "outcome_label",
            "treatment_response",
            "adverse_event",
        ]
        annotator_roles = ["physician", "data_scientist", "domain_expert"]

        # Create IAA groups (multiple annotators per item)
        annotation_count = 0
        for pj in self.patient_journeys[:50]:  # Annotate subset
            iaa_group_id = str(uuid.uuid4())
            num_annotators = random.randint(2, 4)

            for annotator_idx in range(num_annotators):
                annotation_type = random.choice(annotation_types)

                self.ml_annotations.append(
                    {
                        "annotation_id": str(uuid.uuid4()),
                        "entity_type": "patient_journey",
                        "entity_id": pj["patient_journey_id"],
                        "annotation_type": annotation_type,
                        "annotator_id": f"ANN_{annotator_idx:03d}",
                        "annotator_role": random.choice(annotator_roles),
                        "annotation_value": {
                            "label": random.choice(["positive", "negative", "uncertain"]),
                            "confidence": round(random.uniform(0.7, 0.99), 2),
                            "notes": f"Annotation notes for {annotation_type}",
                        },
                        "annotation_confidence": round(random.uniform(0.7, 0.99), 2),
                        "annotation_timestamp": datetime.now().isoformat(),
                        "annotation_duration_seconds": random.randint(30, 300),
                        "is_adjudicated": annotator_idx == num_annotators - 1
                        and random.random() > 0.3,
                        "adjudication_result": (
                            {"final_label": random.choice(["positive", "negative"])}
                            if annotator_idx == num_annotators - 1
                            else None
                        ),
                        "adjudicated_by": (
                            f"ADJ_{random.randint(1, 5):03d}"
                            if annotator_idx == num_annotators - 1
                            else None
                        ),
                        "adjudicated_at": (
                            datetime.now().isoformat()
                            if annotator_idx == num_annotators - 1
                            else None
                        ),
                        "iaa_group_id": iaa_group_id,
                        "created_at": datetime.now().isoformat(),
                    }
                )
                annotation_count += 1

    def _generate_etl_pipeline_metrics(self):
        """Generate ETL pipeline metrics for Time-to-Release."""
        print("  - Generating ETL pipeline metrics...")

        pipeline_names = [
            "patient_journey_etl",
            "treatment_event_etl",
            "hcp_profile_etl",
            "claims_integration",
        ]

        current_date = self.config.data_start_date
        while current_date <= self.config.data_end_date:
            for pipeline_name in pipeline_names:
                run_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=2)
                duration_seconds = random.randint(300, 7200)  # 5 min to 2 hours
                run_end = run_start + timedelta(seconds=duration_seconds)

                # TTR: Time from source data availability to dashboard
                source_data_timestamp = run_start - timedelta(
                    hours=random.randint(12, 168)
                )  # 0.5 to 7 days
                ttr_hours = (run_end - source_data_timestamp).total_seconds() / 3600

                self.etl_pipeline_metrics.append(
                    {
                        "pipeline_run_id": str(uuid.uuid4()),
                        "pipeline_name": pipeline_name,
                        "pipeline_version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}",
                        "run_start": run_start.isoformat(),
                        "run_end": run_end.isoformat(),
                        "duration_seconds": duration_seconds,
                        "source_data_date": (
                            current_date - timedelta(days=random.randint(1, 7))
                        ).isoformat(),
                        "source_data_timestamp": source_data_timestamp.isoformat(),
                        "time_to_release_hours": round(ttr_hours, 2),
                        "stage_timings": {
                            "extract": random.randint(60, 600),
                            "transform": random.randint(120, 3600),
                            "load": random.randint(60, 600),
                            "validate": random.randint(30, 300),
                        },
                        "records_processed": random.randint(10000, 500000),
                        "records_failed": random.randint(0, 100),
                        "status": random.choice(
                            ["success", "success", "success", "partial", "failed"]
                        ),
                        "error_details": (
                            None if random.random() > 0.1 else "Timeout on large batch"
                        ),
                        "quality_checks_passed": random.randint(8, 10),
                        "quality_checks_failed": random.randint(0, 2),
                        "quality_check_details": {
                            "null_check": "passed",
                            "range_check": "passed",
                            "consistency_check": random.choice(["passed", "warning"]),
                        },
                        "created_at": datetime.now().isoformat(),
                    }
                )

            current_date += timedelta(days=1)

    def _generate_hcp_intent_surveys(self):
        """Generate HCP intent surveys for Intent-to-Prescribe Δ."""
        print("  - Generating HCP intent surveys...")

        survey_types = ["market_research", "detail_followup", "conference", "advisory_board"]

        for hcp in self.hcp_profiles:
            # Each HCP may have 1-4 surveys over the period
            num_surveys = random.randint(1, 4)
            previous_survey_id = None
            previous_intent = None

            for survey_idx in range(num_surveys):
                survey_date = random_date_in_range(
                    self.config.data_start_date + timedelta(days=30 * survey_idx),
                    self.config.data_start_date + timedelta(days=30 * (survey_idx + 1)),
                )

                for brand in TARGET_BRANDS:
                    survey_id = str(uuid.uuid4())
                    intent_score = random.randint(1, 7)
                    intent_change = intent_score - previous_intent if previous_intent else 0

                    self.hcp_intent_surveys.append(
                        {
                            "survey_id": survey_id,
                            "hcp_id": hcp["hcp_id"],
                            "survey_date": survey_date.isoformat(),
                            "survey_type": random.choice(survey_types),
                            "brand": brand,
                            "intent_to_prescribe_score": intent_score,
                            "intent_to_prescribe_change": intent_change,
                            "awareness_score": random.randint(3, 7),
                            "favorability_score": random.randint(2, 7),
                            "previous_survey_id": previous_survey_id,
                            "days_since_last_survey": (
                                random.randint(30, 90) if previous_survey_id else None
                            ),
                            "interventions_since_last": [
                                {
                                    "type": random.choice(["detail", "email", "webinar", "sample"]),
                                    "date": (
                                        survey_date - timedelta(days=random.randint(1, 30))
                                    ).isoformat(),
                                }
                                for _ in range(random.randint(0, 3))
                            ],
                            "survey_source": random.choice(["Veeva", "Internal", "Third_Party"]),
                            "response_quality_flag": random.random() > 0.05,
                            "created_at": datetime.now().isoformat(),
                        }
                    )

                    previous_survey_id = survey_id
                    previous_intent = intent_score

    def _generate_preprocessing_metadata(self):
        """Generate preprocessing metadata computed on training data."""
        print("  - Generating preprocessing metadata...")

        train_journeys = [pj for pj in self.patient_journeys if pj["data_split"] == "train"]

        # Compute stats on training data only
        feature_list = ["age", "comorbidity_count", "risk_score", "prior_rx_count", "days_since_dx"]

        self.preprocessing_metadata = {
            "metadata_id": str(uuid.uuid4()),
            "split_config_id": self.split_config_id,
            "computed_on_split": "train",
            "computed_at": datetime.now().isoformat(),
            "feature_means": {f: round(random.uniform(0.3, 0.7), 4) for f in feature_list},
            "feature_stds": {f: round(random.uniform(0.1, 0.3), 4) for f in feature_list},
            "feature_mins": {f: round(random.uniform(0, 0.2), 4) for f in feature_list},
            "feature_maxs": {f: round(random.uniform(0.8, 1.0), 4) for f in feature_list},
            "categorical_encodings": {
                "region": {"northeast": 0, "south": 1, "midwest": 2, "west": 3},
                "brand": {"Remibrutinib": 0, "Fabhalta": 1, "Kisqali": 2},
                "journey_stage": {s: i for i, s in enumerate(JOURNEY_STAGES)},
            },
            "feature_distributions": {
                f: {
                    "mean": round(random.uniform(0.3, 0.7), 4),
                    "std": round(random.uniform(0.1, 0.3), 4),
                }
                for f in feature_list
            },
            "num_training_samples": len(train_journeys),
            "feature_list": feature_list,
            "preprocessing_pipeline_version": "3.0.0",
        }

    def _print_summary(self):
        """Print generation summary."""
        # Count by split
        split_counts = {}
        for pj in self.patient_journeys:
            split = pj["data_split"]
            split_counts[split] = split_counts.get(split, 0) + 1

        print("\n" + "=" * 60)
        print("DATA GENERATION SUMMARY")
        print("=" * 60)
        print("\nCore Tables:")
        print(f"  - HCP Profiles:        {len(self.hcp_profiles):,}")
        print(f"  - Patient Journeys:    {len(self.patient_journeys):,}")
        print(f"  - Treatment Events:    {len(self.treatment_events):,}")
        print(f"  - ML Predictions:      {len(self.ml_predictions):,}")
        print(f"  - Triggers:            {len(self.triggers):,}")
        print(f"  - Agent Activities:    {len(self.agent_activities):,}")
        print(f"  - Business Metrics:    {len(self.business_metrics):,}")
        print(f"  - Causal Paths:        {len(self.causal_paths):,}")

        print("\nNEW KPI Gap Tables:")
        print(f"  - User Sessions:       {len(self.user_sessions):,} (WS3 Active Users)")
        print(
            f"  - Data Source Tracking:{len(self.data_source_tracking):,} (WS1 Cross-source/Stacking)"
        )
        print(f"  - ML Annotations:      {len(self.ml_annotations):,} (WS1 Label Quality/IAA)")
        print(f"  - ETL Pipeline Metrics:{len(self.etl_pipeline_metrics):,} (WS1 Time-to-Release)")
        print(
            f"  - HCP Intent Surveys:  {len(self.hcp_intent_surveys):,} (Brand Intent-to-Prescribe)"
        )
        print(f"  - Reference Universe:  {len(self.reference_universe):,} (Coverage calculations)")

        print("\nSplit Distribution (Patient Journeys):")
        for split in ["train", "validation", "test", "holdout"]:
            count = split_counts.get(split, 0)
            pct = count / len(self.patient_journeys) * 100
            print(f"  - {split:12}: {count:,} ({pct:.1f}%)")

    def export_to_json(self, output_dir: str = "."):
        """Export all data to JSON files."""
        print(f"\nExporting data to {output_dir}...")

        # Split registry
        split_registry = {
            "split_config_id": self.split_config_id,
            "config_name": self.config.config_name,
            "config_version": self.config.config_version,
            "train_ratio": self.config.train_ratio,
            "validation_ratio": self.config.validation_ratio,
            "test_ratio": self.config.test_ratio,
            "holdout_ratio": self.config.holdout_ratio,
            "data_start_date": self.config.data_start_date.isoformat(),
            "data_end_date": self.config.data_end_date.isoformat(),
            "train_end_date": self.config.train_end_date.isoformat(),
            "validation_end_date": self.config.validation_end_date.isoformat(),
            "test_end_date": self.config.test_end_date.isoformat(),
            "temporal_gap_days": self.config.temporal_gap_days,
            "patient_level_isolation": True,
            "split_strategy": "chronological",
            "is_active": True,
            "created_at": datetime.now().isoformat(),
        }

        # Export files
        exports = {
            "e2i_ml_v3_split_registry.json": [split_registry],
            "e2i_ml_v3_preprocessing_metadata.json": [self.preprocessing_metadata],
            "e2i_ml_v3_reference_universe.json": self.reference_universe,
            "e2i_ml_v3_hcp_profiles.json": self.hcp_profiles,
            "e2i_ml_v3_patient_journeys.json": self.patient_journeys,
            "e2i_ml_v3_treatment_events.json": self.treatment_events,
            "e2i_ml_v3_ml_predictions.json": self.ml_predictions,
            "e2i_ml_v3_triggers.json": self.triggers,
            "e2i_ml_v3_agent_activities.json": self.agent_activities,
            "e2i_ml_v3_business_metrics.json": self.business_metrics,
            "e2i_ml_v3_causal_paths.json": self.causal_paths,
            "e2i_ml_v3_user_sessions.json": self.user_sessions,
            "e2i_ml_v3_data_source_tracking.json": self.data_source_tracking,
            "e2i_ml_v3_ml_annotations.json": self.ml_annotations,
            "e2i_ml_v3_etl_pipeline_metrics.json": self.etl_pipeline_metrics,
            "e2i_ml_v3_hcp_intent_surveys.json": self.hcp_intent_surveys,
        }

        # Split data by train/validation/test
        train_data = {
            "patient_journeys": [p for p in self.patient_journeys if p["data_split"] == "train"],
            "treatment_events": [e for e in self.treatment_events if e["data_split"] == "train"],
            "ml_predictions": [p for p in self.ml_predictions if p["data_split"] == "train"],
            "triggers": [t for t in self.triggers if t["data_split"] == "train"],
            "agent_activities": [a for a in self.agent_activities if a["data_split"] == "train"],
            "business_metrics": [m for m in self.business_metrics if m["data_split"] == "train"],
            "causal_paths": [c for c in self.causal_paths if c["data_split"] == "train"],
        }

        validation_data = {
            "patient_journeys": [
                p for p in self.patient_journeys if p["data_split"] == "validation"
            ],
            "treatment_events": [
                e for e in self.treatment_events if e["data_split"] == "validation"
            ],
            "ml_predictions": [p for p in self.ml_predictions if p["data_split"] == "validation"],
            "triggers": [t for t in self.triggers if t["data_split"] == "validation"],
            "agent_activities": [
                a for a in self.agent_activities if a["data_split"] == "validation"
            ],
            "business_metrics": [
                m for m in self.business_metrics if m["data_split"] == "validation"
            ],
            "causal_paths": [c for c in self.causal_paths if c["data_split"] == "validation"],
        }

        test_data = {
            "patient_journeys": [p for p in self.patient_journeys if p["data_split"] == "test"],
            "treatment_events": [e for e in self.treatment_events if e["data_split"] == "test"],
            "ml_predictions": [p for p in self.ml_predictions if p["data_split"] == "test"],
            "triggers": [t for t in self.triggers if t["data_split"] == "test"],
            "agent_activities": [a for a in self.agent_activities if a["data_split"] == "test"],
            "business_metrics": [m for m in self.business_metrics if m["data_split"] == "test"],
            "causal_paths": [c for c in self.causal_paths if c["data_split"] == "test"],
        }

        exports["e2i_ml_v3_train.json"] = train_data
        exports["e2i_ml_v3_validation.json"] = validation_data
        exports["e2i_ml_v3_test.json"] = test_data

        for filename, data in exports.items():
            filepath = f"{output_dir}/{filename}"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=json_serial)
            print(f"  - Exported: {filename}")

        print("\nExport complete!")
        return list(exports.keys())


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os

    # Create output directory
    output_dir = "/home/claude/e2i_ml_complete_v3_data"
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    generator = E2IDataGenerator()
    generator.generate_all()

    # Export to JSON
    files = generator.export_to_json(output_dir)

    print(f"\n{'=' * 60}")
    print("FILES GENERATED:")
    print("=" * 60)
    for f in files:
        print(f"  - {output_dir}/{f}")
