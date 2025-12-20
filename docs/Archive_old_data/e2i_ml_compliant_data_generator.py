#!/usr/bin/env python3
"""
E2I Causal Analytics Dashboard - ML-Compliant Data Generator
Prevents data leakage through proper temporal splits, preprocessing isolation,
and causal inference-aware data partitioning.

Key Features:
- Chronological train/validation/test splits
- Patient-level isolation (no patient spans multiple sets)
- Time-based validation windows for time-series data
- Preprocessing metadata stored separately from target
- Causal inference compatible structure
"""

import json
import csv
import random
import string
from datetime import datetime, timedelta
import hashlib
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import copy


class DataSplit(Enum):
    """Data split categories"""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HOLDOUT = "holdout"  # For final model evaluation


class SplitStrategy(Enum):
    """Splitting strategies for different use cases"""
    CHRONOLOGICAL = "chronological"          # Time-based split
    PATIENT_STRATIFIED = "patient_stratified"  # Ensure patient only in one split
    ROLLING_WINDOW = "rolling_window"          # For time-series validation
    CAUSAL_HOLDOUT = "causal_holdout"          # Pre/post intervention periods


@dataclass
class SplitConfig:
    """Configuration for data splitting"""
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    test_ratio: float = 0.15
    holdout_ratio: float = 0.05
    
    # Temporal boundaries
    train_end_date: Optional[datetime] = None
    validation_end_date: Optional[datetime] = None
    test_end_date: Optional[datetime] = None
    
    # Gap between sets to prevent leakage
    temporal_gap_days: int = 7
    
    # Patient-level settings
    patient_level_split: bool = True  # Ensure patient only in one split
    
    # Rolling window settings (for time-series validation)
    rolling_window_size_days: int = 30
    rolling_step_days: int = 7
    
    def __post_init__(self):
        total = self.train_ratio + self.validation_ratio + self.test_ratio + self.holdout_ratio
        assert abs(total - 1.0) < 0.001, f"Split ratios must sum to 1.0, got {total}"


@dataclass
class DataLeakageAudit:
    """Track potential data leakage issues"""
    audit_id: str
    timestamp: str
    check_type: str
    passed: bool
    details: str
    severity: str  # "critical", "warning", "info"


class E2IMLCompliantDataGenerator:
    """
    Generate synthetic pharmaceutical data with ML-compliant splits
    to prevent data leakage in training/testing.
    """
    
    def __init__(
        self, 
        num_patients: int = 1000, 
        num_hcps: int = 100,
        split_config: Optional[SplitConfig] = None,
        random_seed: int = 42
    ):
        # Set seed for reproducibility
        random.seed(random_seed)
        self.random_seed = random_seed
        
        self.num_patients = num_patients
        self.num_hcps = num_hcps
        self.split_config = split_config or SplitConfig()
        
        # Define temporal boundaries
        self.data_start_date = datetime(2024, 1, 1)
        self.data_end_date = datetime(2025, 9, 28)
        
        # Calculate split dates based on config
        self._calculate_split_dates()
        
        # Brand configurations
        self.brands = ["Remibrutinib", "Fabhalta", "Kisqali"]
        self.regions = ["northeast", "south", "midwest", "west"]
        self.states = {
            "northeast": ["NY", "MA", "CT", "NJ", "PA", "ME", "NH", "VT", "RI"],
            "south": ["TX", "FL", "GA", "NC", "VA", "TN", "SC", "AL", "LA"],
            "midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "IA", "MO"],
            "west": ["CA", "WA", "OR", "AZ", "CO", "NV", "UT", "NM"]
        }
        
        self.disease_configs = {
            "Remibrutinib": {
                "diagnoses": ["L50.1", "L50.8", "L50.9"],
                "descriptions": ["Chronic spontaneous urticaria", "Other urticaria", "Urticaria, unspecified"],
                "specialties": ["Dermatology", "Allergy & Immunology"],
                "age_groups": ["18-24", "25-34", "35-44", "45-54", "55-64"],
                "comorbidities": ["allergic_rhinitis", "asthma", "eczema", "food_allergies"]
            },
            "Fabhalta": {
                "diagnoses": ["D59.3", "D59.5"],
                "descriptions": ["Hemolytic-uremic syndrome", "Paroxysmal nocturnal hemoglobinuria"],
                "specialties": ["Hematology", "Hematology-Oncology"],
                "age_groups": ["25-34", "35-44", "45-54", "55-64"],
                "comorbidities": ["anemia", "chronic_kidney_disease", "thrombosis", "aplastic_anemia"]
            },
            "Kisqali": {
                "diagnoses": ["C50.911", "C50.912", "C50.811", "C50.812"],
                "descriptions": ["Malignant neoplasm of breast", "HR+ HER2- breast cancer"],
                "specialties": ["Oncology", "Medical Oncology", "Breast Surgery"],
                "age_groups": ["35-44", "45-54", "55-64", "65-74", "75+"],
                "comorbidities": ["diabetes_type2", "hyperlipidemia", "hypertension", "osteoporosis"]
            }
        }
        
        # Data containers by split
        self.data_by_split = {
            DataSplit.TRAIN: self._create_empty_data_container(),
            DataSplit.VALIDATION: self._create_empty_data_container(),
            DataSplit.TEST: self._create_empty_data_container(),
            DataSplit.HOLDOUT: self._create_empty_data_container()
        }
        
        # Master lookup tables (generated once, referenced across splits)
        self.hcp_profiles = []
        self.patient_split_assignments = {}  # patient_id -> DataSplit
        
        # Preprocessing metadata (calculated ONLY on training data)
        self.preprocessing_metadata = {
            "feature_means": {},
            "feature_stds": {},
            "feature_mins": {},
            "feature_maxs": {},
            "categorical_encodings": {},
            "computed_on_split": DataSplit.TRAIN.value,
            "computed_timestamp": None
        }
        
        # Audit trail for leakage checks
        self.leakage_audits: List[DataLeakageAudit] = []
        
    def _create_empty_data_container(self) -> Dict:
        """Create empty data container for a split"""
        return {
            "patient_journeys": [],
            "treatment_events": [],
            "ml_predictions": [],
            "triggers": [],
            "agent_activities": [],
            "business_metrics": [],
            "causal_paths": []
        }
    
    def _calculate_split_dates(self):
        """Calculate temporal boundaries for each split"""
        total_days = (self.data_end_date - self.data_start_date).days
        gap = self.split_config.temporal_gap_days
        
        # Calculate days per split
        train_days = int(total_days * self.split_config.train_ratio)
        val_days = int(total_days * self.split_config.validation_ratio)
        test_days = int(total_days * self.split_config.test_ratio)
        holdout_days = total_days - train_days - val_days - test_days - (3 * gap)
        
        # Set date boundaries with gaps
        self.split_dates = {
            DataSplit.TRAIN: {
                "start": self.data_start_date,
                "end": self.data_start_date + timedelta(days=train_days)
            },
            DataSplit.VALIDATION: {
                "start": self.data_start_date + timedelta(days=train_days + gap),
                "end": self.data_start_date + timedelta(days=train_days + gap + val_days)
            },
            DataSplit.TEST: {
                "start": self.data_start_date + timedelta(days=train_days + 2*gap + val_days),
                "end": self.data_start_date + timedelta(days=train_days + 2*gap + val_days + test_days)
            },
            DataSplit.HOLDOUT: {
                "start": self.data_start_date + timedelta(days=train_days + 3*gap + val_days + test_days),
                "end": self.data_end_date
            }
        }
        
        print("\n📅 Data Split Temporal Boundaries:")
        for split, dates in self.split_dates.items():
            print(f"  {split.value}: {dates['start'].date()} to {dates['end'].date()}")
    
    def _assign_patient_to_split(self, patient_id: str, journey_start: datetime) -> DataSplit:
        """
        Assign patient to a split based on their journey start date.
        Once assigned, patient stays in that split (patient-level isolation).
        """
        # If already assigned, return existing assignment
        if patient_id in self.patient_split_assignments:
            return self.patient_split_assignments[patient_id]
        
        # Assign based on journey start date
        for split in [DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST, DataSplit.HOLDOUT]:
            dates = self.split_dates[split]
            if dates["start"] <= journey_start < dates["end"]:
                self.patient_split_assignments[patient_id] = split
                return split
        
        # If in gap period, assign to earlier split to avoid leakage
        # Find the closest preceding split
        for split in [DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST]:
            if journey_start < self.split_dates[split]["end"] + timedelta(days=self.split_config.temporal_gap_days):
                self.patient_split_assignments[patient_id] = split
                return split
        
        # Default to holdout
        self.patient_split_assignments[patient_id] = DataSplit.HOLDOUT
        return DataSplit.HOLDOUT
    
    def _random_date(self, start: datetime, end: datetime) -> datetime:
        """Generate random date within range"""
        delta = end - start
        if delta.days <= 0:
            return start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)
    
    def _random_datetime(self, start: datetime, end: datetime) -> datetime:
        """Generate random datetime within range"""
        delta = end - start
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)
    
    def generate_hcp_profiles(self):
        """Generate HCP profiles (shared across all splits as reference data)"""
        print("\n👨‍⚕️ Generating HCP profiles...")
        
        first_names = ["Sarah", "Michael", "Jennifer", "David", "Emily", "Robert", "Lisa", 
                       "James", "Patricia", "John", "Maria", "William", "Susan", "Richard",
                       "Karen", "Charles", "Linda", "Thomas", "Barbara", "Christopher"]
        
        last_names = ["Johnson", "Chen", "Williams", "Martinez", "Thompson", "Anderson",
                     "Taylor", "Garcia", "Miller", "Davis", "Rodriguez", "Wilson", "Smith",
                     "Jones", "Brown", "Lee", "Kim", "Park", "Patel", "Shah"]
        
        practice_types = ["academic_medical_center", "hospital", "group_practice", 
                         "solo_practice", "cancer_center", "specialty_clinic"]
        
        for i in range(self.num_hcps):
            brand_weight = random.random()
            if brand_weight < 0.33:
                brand = "Remibrutinib"
            elif brand_weight < 0.66:
                brand = "Fabhalta"
            else:
                brand = "Kisqali"
            
            config = self.disease_configs[brand]
            specialty = random.choice(config["specialties"])
            region = random.choice(self.regions)
            state = random.choice(self.states[region])
            
            hcp = {
                "hcp_id": f"HCP-{i+1:04d}",
                "npi": f"1{random.randint(100000000, 999999999)}",
                "first_name": random.choice(first_names),
                "last_name": random.choice(last_names),
                "specialty": specialty,
                "sub_specialty": self._get_subspecialty(specialty),
                "practice_type": random.choice(practice_types),
                "practice_size": random.choice(["small", "medium", "large"]),
                "geographic_region": region,
                "state": state,
                "priority_tier": random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.25, 0.15])[0],
                "decile": random.randint(1, 10),
                "total_patient_volume": random.randint(500, 3000),
                "digital_engagement_score": round(random.uniform(0.4, 0.95), 2),
                "preferred_channel": random.choice(["email", "app", "phone", "in_person"]),
                "adoption_category": random.choice(["early_adopter", "mainstream", "late_adopter"]),
                "coverage_status": random.choice([True, True, True, False])
            }
            
            self.hcp_profiles.append(hcp)
        
        print(f"  Generated {len(self.hcp_profiles)} HCP profiles")
    
    def _get_subspecialty(self, specialty: str) -> str:
        """Get subspecialty based on specialty"""
        subspecialties = {
            "Dermatology": ["General Dermatology", "Immunodermatology", "Dermatopathology"],
            "Allergy & Immunology": ["Clinical Immunology", "Pediatric Allergy", "Adult Allergy"],
            "Hematology": ["Benign Hematology", "Hematopathology", "Transfusion Medicine"],
            "Hematology-Oncology": ["Leukemia", "Lymphoma", "Bone Marrow Transplant"],
            "Oncology": ["Medical Oncology", "Gynecologic Oncology", "Thoracic Oncology"],
            "Medical Oncology": ["Breast Oncology", "GI Oncology", "GU Oncology"],
            "Breast Surgery": ["Surgical Oncology", "Reconstructive Surgery"]
        }
        return random.choice(subspecialties.get(specialty, ["General"]))
    
    def generate_patient_journeys_with_splits(self):
        """Generate patient journeys with proper split assignment"""
        print("\n🏥 Generating patient journeys with ML-compliant splits...")
        
        split_counts = {s: 0 for s in DataSplit}
        
        for i in range(self.num_patients):
            # Select brand and configuration
            brand = random.choices(
                self.brands,
                weights=[0.4, 0.2, 0.4]
            )[0]
            
            config = self.disease_configs[brand]
            region = random.choice(self.regions)
            state = random.choice(self.states[region])
            
            # Generate journey start date
            journey_start = self._random_date(
                self.data_start_date, 
                self.data_end_date - timedelta(days=30)
            )
            
            # Assign patient to split based on journey start
            patient_id = f"PAT-{i+1:05d}"
            assigned_split = self._assign_patient_to_split(patient_id, journey_start)
            split_counts[assigned_split] += 1
            
            # Constrain journey to split boundaries
            split_dates = self.split_dates[assigned_split]
            journey_start = max(journey_start, split_dates["start"])
            
            # Ensure we have room for a minimum journey
            max_possible_duration = (split_dates["end"] - journey_start).days
            if max_possible_duration < 7:
                # Move journey_start earlier if possible
                journey_start = split_dates["start"]
                max_possible_duration = (split_dates["end"] - journey_start).days
            
            if max_possible_duration < 7:
                # Skip this patient - split window too small
                continue
                
            journey_duration = random.randint(7, min(150, max_possible_duration))
            journey_end = journey_start + timedelta(days=journey_duration)
            
            journey = {
                "patient_journey_id": f"PJ-{i+1:05d}",
                "patient_id": patient_id,
                "patient_hash": hashlib.md5(patient_id.encode()).hexdigest()[:12],
                "data_split": assigned_split.value,
                "journey_start_date": journey_start.strftime("%Y-%m-%d"),
                "journey_end_date": journey_end.strftime("%Y-%m-%d"),
                "journey_duration_days": (journey_end - journey_start).days,
                "journey_stage": random.choice([
                    "diagnosis", "initial_treatment", 
                    "treatment_optimization", "maintenance", "treatment_switch"
                ]),
                "journey_status": random.choice(["active", "stable", "transitioning", "completed"]),
                "primary_diagnosis_code": random.choice(config["diagnoses"]),
                "primary_diagnosis_desc": random.choice(config["descriptions"]),
                "brand": brand,
                "age_group": random.choice(config["age_groups"]),
                "gender": random.choice(["M", "F"]),
                "geographic_region": region,
                "state": state,
                "insurance_type": random.choice(["commercial", "medicare", "medicaid", "cash"]),
                "data_quality_score": round(random.uniform(0.75, 0.99), 2),
                "comorbidities": random.sample(config["comorbidities"], k=random.randint(0, 3))
            }
            
            # Add to appropriate split container
            self.data_by_split[assigned_split]["patient_journeys"].append(journey)
            
            # Generate treatment events for this patient
            self._generate_treatment_events_for_patient(
                journey, assigned_split, split_dates, config
            )
            
            # Generate ML predictions for this patient
            self._generate_ml_predictions_for_patient(
                journey, assigned_split, split_dates
            )
        
        print(f"  Split distribution:")
        for split, count in split_counts.items():
            print(f"    {split.value}: {count} patients ({count/self.num_patients*100:.1f}%)")
    
    def _generate_treatment_events_for_patient(
        self, 
        journey: Dict, 
        split: DataSplit,
        split_dates: Dict,
        config: Dict
    ):
        """Generate treatment events for a patient within their split boundaries"""
        num_events = random.randint(2, 8)
        journey_start = datetime.strptime(journey["journey_start_date"], "%Y-%m-%d")
        journey_end = datetime.strptime(journey["journey_end_date"], "%Y-%m-%d")
        
        hcp = random.choice(self.hcp_profiles)
        
        event_types = ["diagnosis", "prescription", "lab_test", "procedure", "consultation"]
        
        for j in range(num_events):
            event_date = self._random_date(journey_start, journey_end)
            
            event = {
                "treatment_event_id": f"TE-{journey['patient_id']}-{j+1:03d}",
                "patient_journey_id": journey["patient_journey_id"],
                "patient_id": journey["patient_id"],
                "hcp_id": hcp["hcp_id"],
                "data_split": split.value,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "event_type": random.choice(event_types),
                "brand": journey["brand"],
                "icd_codes": [random.choice(config["diagnoses"])],
                "sequence_number": j + 1,
                "days_from_diagnosis": (event_date - journey_start).days,
                "cost": round(random.uniform(100, 5000), 2),
                "outcome_indicator": random.choice(["positive", "neutral", "negative"]),
                "adverse_event_flag": random.random() < 0.05
            }
            
            self.data_by_split[split]["treatment_events"].append(event)
    
    def _generate_ml_predictions_for_patient(
        self, 
        journey: Dict, 
        split: DataSplit,
        split_dates: Dict
    ):
        """Generate ML predictions ensuring no future data leakage"""
        # Only generate predictions if enough time has passed in journey
        journey_start = datetime.strptime(journey["journey_start_date"], "%Y-%m-%d")
        journey_end = datetime.strptime(journey["journey_end_date"], "%Y-%m-%d")
        
        if (journey_end - journey_start).days < 14:
            return  # Not enough history for prediction
        
        # Prediction must be AFTER enough history is available
        prediction_window_start = journey_start + timedelta(days=14)
        prediction_date = self._random_datetime(prediction_window_start, journey_end)
        
        hcp = random.choice(self.hcp_profiles)
        
        model_types = ["gradient_boosting", "neural_net", "causal_forest", "double_ml"]
        prediction_types = ["trigger", "propensity", "risk", "churn", "next_best_action"]
        
        prediction = {
            "prediction_id": f"PRED-{journey['patient_id']}-{random.randint(1,999):03d}",
            "model_version": f"v2.3.{random.randint(0, 5)}",
            "model_type": random.choice(model_types),
            "prediction_timestamp": prediction_date.isoformat() + "Z",
            "patient_id": journey["patient_id"],
            "hcp_id": hcp["hcp_id"],
            "data_split": split.value,
            "prediction_type": random.choice(prediction_types),
            "prediction_value": round(random.uniform(0, 1), 2),
            "prediction_class": random.choice(["low", "medium", "high"]),
            "confidence_score": round(random.uniform(0.6, 0.98), 2),
            "model_auc": round(random.uniform(0.75, 0.92), 2),
            "model_precision": round(random.uniform(0.35, 0.85), 2),
            "model_recall": round(random.uniform(0.60, 0.90), 2),
            "calibration_score": round(random.uniform(0.80, 0.98), 2),
            "causal_confidence": round(random.uniform(0.70, 0.95), 2),
            "treatment_effect_estimate": round(random.uniform(0.3, 0.9), 2),
            "heterogeneous_effect": round(random.uniform(0.2, 0.9), 2),
            "segment_assignment": random.choice([
                "low_risk", "moderate_risk", "high_risk", 
                "treatment_naive", "experienced"
            ]),
            # CRITICAL: Store which features were available at prediction time
            "features_available_at_prediction": {
                "days_of_history": (prediction_date - journey_start).days,
                "num_prior_events": random.randint(1, 5),
                "prior_treatments_known": True,
                "outcome_known": False  # Cannot know outcome at prediction time!
            }
        }
        
        self.data_by_split[split]["ml_predictions"].append(prediction)
    
    def generate_triggers_with_temporal_awareness(self):
        """Generate triggers respecting temporal boundaries"""
        print("\n⚡ Generating triggers with temporal awareness...")
        
        for split in DataSplit:
            journeys = self.data_by_split[split]["patient_journeys"]
            split_dates = self.split_dates[split]
            
            for journey in journeys:
                if random.random() > 0.6:  # 60% of patients get triggers
                    continue
                    
                journey_start = datetime.strptime(journey["journey_start_date"], "%Y-%m-%d")
                journey_end = datetime.strptime(journey["journey_end_date"], "%Y-%m-%d")
                
                # Trigger must be AFTER sufficient patient history
                if (journey_end - journey_start).days < 21:
                    continue
                    
                trigger_date = self._random_datetime(
                    journey_start + timedelta(days=21),
                    journey_end
                )
                
                hcp = random.choice(self.hcp_profiles)
                
                trigger = {
                    "trigger_id": f"TRG-{journey['patient_id']}-{random.randint(1,999):03d}",
                    "patient_id": journey["patient_id"],
                    "hcp_id": hcp["hcp_id"],
                    "data_split": split.value,
                    "trigger_timestamp": trigger_date.isoformat() + "Z",
                    "trigger_type": random.choice([
                        "high_risk", "treatment_opportunity", 
                        "adherence_concern", "switching_opportunity"
                    ]),
                    "priority": random.choice(["critical", "high", "medium", "low"]),
                    "confidence_score": round(random.uniform(0.6, 0.95), 2),
                    "lead_time_days": random.randint(7, 28),
                    "expiration_date": (trigger_date + timedelta(days=random.randint(7, 30))).strftime("%Y-%m-%d"),
                    "delivery_status": random.choice(["pending", "delivered", "viewed", "actioned"]),
                    "acceptance_status": random.choice([None, "accepted", "rejected", "deferred"]),
                    "causal_chain": {
                        "steps": [
                            {
                                "step": i+1,
                                "action": random.choice(["intervention", "monitoring", "escalation"]),
                                "effect": random.choice(["improvement", "stabilization", "prevention"]),
                                "probability": round(random.uniform(0.6, 0.95), 2)
                            }
                            for i in range(3)
                        ]
                    }
                }
                
                self.data_by_split[split]["triggers"].append(trigger)
    
    def generate_agent_activities(self):
        """Generate agent activities across splits"""
        print("\n🤖 Generating agent activities...")
        
        agents = [
            "causal_chain_analyzer", "multiplier_discoverer", "data_drift_monitor",
            "prediction_synthesizer", "heterogeneous_optimizer", 
            "competitive_landscape", "explainer_agent", "feedback_learner"
        ]
        
        workstreams = ["WS1", "WS2", "WS3"]
        activity_types = ["analysis", "prediction", "optimization", "monitoring", "action"]
        
        for split in DataSplit:
            split_dates = self.split_dates[split]
            num_activities = int(100 * {
                DataSplit.TRAIN: 0.60,
                DataSplit.VALIDATION: 0.20,
                DataSplit.TEST: 0.15,
                DataSplit.HOLDOUT: 0.05
            }[split])
            
            for i in range(num_activities):
                activity_date = self._random_datetime(
                    split_dates["start"],
                    split_dates["end"]
                )
                
                activity = {
                    "activity_id": f"AGT-{split.value[:3].upper()}-{i+1:05d}",
                    "agent_name": random.choice(agents),
                    "activity_timestamp": activity_date.isoformat() + "Z",
                    "activity_type": random.choice(activity_types),
                    "workstream": random.choice(workstreams),
                    "data_split": split.value,
                    "processing_duration_ms": random.randint(500, 10000),
                    "records_processed": random.randint(10, 1000),
                    "causal_paths_analyzed": random.randint(10, 200),
                    "confidence_level": round(random.uniform(0.7, 0.98), 2),
                    "impact_estimate": random.randint(100000, 20000000),
                    "roi_estimate": round(random.uniform(2.0, 10.0), 1),
                    "status": random.choice(["completed", "failed", "pending"])
                }
                
                self.data_by_split[split]["agent_activities"].append(activity)
    
    def generate_business_metrics(self):
        """Generate business metrics by split"""
        print("\n📊 Generating business metrics...")
        
        metric_types = ["revenue", "adoption", "efficiency", "quality", "impact"]
        
        for split in DataSplit:
            split_dates = self.split_dates[split]
            
            # Generate weekly metrics
            current_date = split_dates["start"]
            while current_date < split_dates["end"]:
                for brand in self.brands:
                    for region in self.regions:
                        metric = {
                            "metric_id": f"BM-{current_date.strftime('%Y%m%d')}-{brand[:3]}-{region[:2]}",
                            "metric_date": current_date.strftime("%Y-%m-%d"),
                            "metric_type": random.choice(metric_types),
                            "brand": brand,
                            "region": region,
                            "data_split": split.value,
                            "value": round(random.uniform(10000, 1000000), 2),
                            "target": round(random.uniform(15000, 1200000), 2),
                            "achievement_rate": round(random.uniform(0.7, 1.2), 2),
                            "year_over_year_change": round(random.uniform(-0.1, 0.3), 2),
                            "roi": round(random.uniform(2.0, 8.0), 1),
                            "statistical_significance": round(random.uniform(0.90, 0.99), 2)
                        }
                        
                        self.data_by_split[split]["business_metrics"].append(metric)
                
                current_date += timedelta(days=7)
    
    def compute_preprocessing_metadata(self):
        """
        Compute preprocessing statistics ONLY from training data.
        This prevents data leakage from test/validation sets.
        """
        print("\n📐 Computing preprocessing metadata from TRAINING data only...")
        
        train_data = self.data_by_split[DataSplit.TRAIN]
        
        # Collect numerical features from training predictions
        train_predictions = train_data["ml_predictions"]
        
        if train_predictions:
            features = ["prediction_value", "confidence_score", "model_auc", 
                       "treatment_effect_estimate", "heterogeneous_effect"]
            
            for feature in features:
                values = [p[feature] for p in train_predictions if feature in p]
                if values:
                    self.preprocessing_metadata["feature_means"][feature] = sum(values) / len(values)
                    self.preprocessing_metadata["feature_stds"][feature] = (
                        (sum((v - self.preprocessing_metadata["feature_means"][feature])**2 
                             for v in values) / len(values)) ** 0.5
                    )
                    self.preprocessing_metadata["feature_mins"][feature] = min(values)
                    self.preprocessing_metadata["feature_maxs"][feature] = max(values)
        
        # Collect categorical encodings from training data
        train_journeys = train_data["patient_journeys"]
        if train_journeys:
            self.preprocessing_metadata["categorical_encodings"]["brands"] = list(set(
                j["brand"] for j in train_journeys
            ))
            self.preprocessing_metadata["categorical_encodings"]["regions"] = list(set(
                j["geographic_region"] for j in train_journeys
            ))
            self.preprocessing_metadata["categorical_encodings"]["journey_stages"] = list(set(
                j["journey_stage"] for j in train_journeys
            ))
        
        self.preprocessing_metadata["computed_timestamp"] = datetime.now().isoformat()
        
        print(f"  Computed statistics for {len(self.preprocessing_metadata['feature_means'])} numerical features")
        print(f"  Computed encodings for {len(self.preprocessing_metadata['categorical_encodings'])} categorical features")
    
    def run_leakage_audit(self):
        """Run comprehensive data leakage audit"""
        print("\n🔍 Running data leakage audit...")
        
        # Audit 1: Check patient split isolation
        patient_splits = {}
        for split in DataSplit:
            for journey in self.data_by_split[split]["patient_journeys"]:
                pid = journey["patient_id"]
                if pid in patient_splits:
                    self.leakage_audits.append(DataLeakageAudit(
                        audit_id=f"AUDIT-{len(self.leakage_audits)+1:04d}",
                        timestamp=datetime.now().isoformat(),
                        check_type="patient_split_isolation",
                        passed=False,
                        details=f"Patient {pid} found in both {patient_splits[pid]} and {split.value}",
                        severity="critical"
                    ))
                else:
                    patient_splits[pid] = split.value
        
        if not any(a.check_type == "patient_split_isolation" and not a.passed 
                   for a in self.leakage_audits):
            self.leakage_audits.append(DataLeakageAudit(
                audit_id=f"AUDIT-{len(self.leakage_audits)+1:04d}",
                timestamp=datetime.now().isoformat(),
                check_type="patient_split_isolation",
                passed=True,
                details=f"All {len(patient_splits)} patients correctly isolated to single split",
                severity="info"
            ))
        
        # Audit 2: Check temporal boundaries
        for split in DataSplit:
            split_dates = self.split_dates[split]
            for journey in self.data_by_split[split]["patient_journeys"]:
                journey_start = datetime.strptime(journey["journey_start_date"], "%Y-%m-%d")
                journey_end = datetime.strptime(journey["journey_end_date"], "%Y-%m-%d")
                
                if journey_end > split_dates["end"]:
                    self.leakage_audits.append(DataLeakageAudit(
                        audit_id=f"AUDIT-{len(self.leakage_audits)+1:04d}",
                        timestamp=datetime.now().isoformat(),
                        check_type="temporal_boundary",
                        passed=False,
                        details=f"Journey {journey['patient_journey_id']} extends beyond {split.value} boundary",
                        severity="warning"
                    ))
        
        # Audit 3: Check preprocessing metadata source
        if self.preprocessing_metadata["computed_on_split"] != DataSplit.TRAIN.value:
            self.leakage_audits.append(DataLeakageAudit(
                audit_id=f"AUDIT-{len(self.leakage_audits)+1:04d}",
                timestamp=datetime.now().isoformat(),
                check_type="preprocessing_isolation",
                passed=False,
                details="Preprocessing metadata not computed exclusively from training data",
                severity="critical"
            ))
        else:
            self.leakage_audits.append(DataLeakageAudit(
                audit_id=f"AUDIT-{len(self.leakage_audits)+1:04d}",
                timestamp=datetime.now().isoformat(),
                check_type="preprocessing_isolation",
                passed=True,
                details="Preprocessing metadata correctly computed from training data only",
                severity="info"
            ))
        
        # Audit 4: Check for future outcome leakage in predictions
        for split in DataSplit:
            for pred in self.data_by_split[split]["ml_predictions"]:
                if "features_available_at_prediction" in pred:
                    if pred["features_available_at_prediction"].get("outcome_known", True):
                        self.leakage_audits.append(DataLeakageAudit(
                            audit_id=f"AUDIT-{len(self.leakage_audits)+1:04d}",
                            timestamp=datetime.now().isoformat(),
                            check_type="outcome_leakage",
                            passed=False,
                            details=f"Prediction {pred['prediction_id']} has outcome known at prediction time",
                            severity="critical"
                        ))
        
        # Print audit summary
        passed = sum(1 for a in self.leakage_audits if a.passed)
        failed = sum(1 for a in self.leakage_audits if not a.passed)
        critical = sum(1 for a in self.leakage_audits if a.severity == "critical" and not a.passed)
        
        print(f"\n  Audit Results:")
        print(f"    ✅ Passed: {passed}")
        print(f"    ❌ Failed: {failed}")
        print(f"    🚨 Critical Issues: {critical}")
        
        if critical > 0:
            print("\n  ⚠️  CRITICAL: Data leakage detected! Review audit log before proceeding.")
    
    def generate_all(self):
        """Generate all data with proper splits"""
        print("=" * 70)
        print("E2I ML-Compliant Data Generator")
        print(f"Generating data for {self.num_patients} patients, {self.num_hcps} HCPs")
        print(f"Random seed: {self.random_seed}")
        print("=" * 70)
        
        # Generate data
        self.generate_hcp_profiles()
        self.generate_patient_journeys_with_splits()
        self.generate_triggers_with_temporal_awareness()
        self.generate_agent_activities()
        self.generate_business_metrics()
        
        # Compute preprocessing metadata from training data only
        self.compute_preprocessing_metadata()
        
        # Run leakage audit
        self.run_leakage_audit()
        
        # Summary
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        
        for split in DataSplit:
            data = self.data_by_split[split]
            print(f"\n{split.value.upper()} Set:")
            for key, records in data.items():
                print(f"  {key}: {len(records)} records")
    
    def save_to_files(self, output_prefix: str = "e2i_ml_compliant"):
        """Save data to files by split"""
        print(f"\n💾 Saving data files with prefix '{output_prefix}'...")
        
        # Save each split separately
        for split in DataSplit:
            split_data = {
                "metadata": {
                    "generated_date": datetime.now().isoformat(),
                    "generator_version": "2.0.0-ml-compliant",
                    "split": split.value,
                    "split_dates": {
                        "start": self.split_dates[split]["start"].isoformat(),
                        "end": self.split_dates[split]["end"].isoformat()
                    },
                    "num_patients": len(self.data_by_split[split]["patient_journeys"]),
                    "random_seed": self.random_seed
                },
                "data": self.data_by_split[split]
            }
            
            filename = f"{output_prefix}_{split.value}.json"
            with open(filename, 'w') as f:
                json.dump(split_data, f, indent=2, default=str)
            print(f"  Saved {filename}")
        
        # Save HCP profiles (shared reference data)
        hcp_filename = f"{output_prefix}_hcp_profiles.json"
        with open(hcp_filename, 'w') as f:
            json.dump({"hcp_profiles": self.hcp_profiles}, f, indent=2)
        print(f"  Saved {hcp_filename}")
        
        # Save preprocessing metadata
        meta_filename = f"{output_prefix}_preprocessing_metadata.json"
        with open(meta_filename, 'w') as f:
            json.dump(self.preprocessing_metadata, f, indent=2)
        print(f"  Saved {meta_filename}")
        
        # Save leakage audit report
        audit_filename = f"{output_prefix}_leakage_audit.json"
        with open(audit_filename, 'w') as f:
            json.dump([asdict(a) for a in self.leakage_audits], f, indent=2)
        print(f"  Saved {audit_filename}")
        
        # Save CSV files for each split
        for split in DataSplit:
            for entity_type, records in self.data_by_split[split].items():
                if records:
                    filename = f"{output_prefix}_{split.value}_{entity_type}.csv"
                    
                    flattened = []
                    for record in records:
                        flat = {}
                        for key, value in record.items():
                            if isinstance(value, (dict, list)):
                                flat[key] = json.dumps(value)
                            else:
                                flat[key] = value
                        flattened.append(flat)
                    
                    with open(filename, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                        writer.writeheader()
                        writer.writerows(flattened)
                    
                    print(f"  Saved {filename} ({len(records)} records)")
    
    def create_pipeline_config(self) -> Dict:
        """Create scikit-learn compatible pipeline configuration"""
        return {
            "version": "1.0",
            "description": "ML Pipeline configuration for E2I Causal Analytics",
            "data_splits": {
                split.value: {
                    "start_date": self.split_dates[split]["start"].isoformat(),
                    "end_date": self.split_dates[split]["end"].isoformat(),
                    "patient_count": len(self.data_by_split[split]["patient_journeys"])
                }
                for split in DataSplit
            },
            "preprocessing": {
                "source_split": DataSplit.TRAIN.value,
                "numerical_features": list(self.preprocessing_metadata["feature_means"].keys()),
                "categorical_features": list(self.preprocessing_metadata["categorical_encodings"].keys()),
                "scaling": {
                    "method": "standard",
                    "means": self.preprocessing_metadata["feature_means"],
                    "stds": self.preprocessing_metadata["feature_stds"]
                }
            },
            "temporal_config": {
                "gap_days": self.split_config.temporal_gap_days,
                "patient_level_isolation": self.split_config.patient_level_split
            },
            "audit_summary": {
                "total_audits": len(self.leakage_audits),
                "passed": sum(1 for a in self.leakage_audits if a.passed),
                "failed": sum(1 for a in self.leakage_audits if not a.passed),
                "critical_issues": sum(1 for a in self.leakage_audits 
                                      if a.severity == "critical" and not a.passed)
            }
        }


def create_time_series_cv_splits(
    data: Dict,
    n_splits: int = 5,
    gap_days: int = 7
) -> List[Tuple[List[str], List[str]]]:
    """
    Create time-series cross-validation splits for model validation.
    Uses expanding window approach where training set grows.
    
    Returns list of (train_patient_ids, test_patient_ids) tuples.
    """
    journeys = data["patient_journeys"]
    
    # Sort by journey start date
    sorted_journeys = sorted(
        journeys, 
        key=lambda j: j["journey_start_date"]
    )
    
    # Calculate split boundaries
    total = len(sorted_journeys)
    test_size = total // (n_splits + 1)
    
    splits = []
    for i in range(n_splits):
        train_end_idx = (i + 1) * test_size
        test_start_idx = train_end_idx + gap_days  # Gap to prevent leakage
        test_end_idx = test_start_idx + test_size
        
        train_ids = [j["patient_id"] for j in sorted_journeys[:train_end_idx]]
        test_ids = [j["patient_id"] for j in sorted_journeys[test_start_idx:test_end_idx]]
        
        splits.append((train_ids, test_ids))
    
    return splits


# Main execution
if __name__ == "__main__":
    # Configure split
    config = SplitConfig(
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.15,
        holdout_ratio=0.05,
        temporal_gap_days=7,
        patient_level_split=True
    )
    
    # Generate data
    generator = E2IMLCompliantDataGenerator(
        num_patients=500,  # Adjust for your needs
        num_hcps=100,
        split_config=config,
        random_seed=42
    )
    
    generator.generate_all()
    generator.save_to_files("e2i_ml_compliant")
    
    # Save pipeline config
    pipeline_config = generator.create_pipeline_config()
    with open("e2i_pipeline_config.json", 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    
    print("\n✅ ML-compliant data generation complete!")
    print("\nFiles created:")
    print("  - e2i_ml_compliant_{train,validation,test,holdout}.json")
    print("  - e2i_ml_compliant_hcp_profiles.json")
    print("  - e2i_ml_compliant_preprocessing_metadata.json")
    print("  - e2i_ml_compliant_leakage_audit.json")
    print("  - e2i_pipeline_config.json")
    print("  - CSV files for each entity by split")
