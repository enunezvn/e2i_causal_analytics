#!/usr/bin/env python3
"""
E2I Causal Analytics - Supabase Data Loader with ML Split Support

This script loads the ML-compliant generated data into Supabase,
maintaining proper train/validation/test/holdout split tracking.

Features:
- Creates split configuration in registry
- Assigns patients to splits with isolation
- Stores preprocessing metadata
- Runs leakage audit after load
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import supabase, provide helpful message if not installed
try:
    from supabase import create_client, Client
except ImportError:
    logger.error("Supabase client not installed. Run: pip install supabase")
    raise


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection"""
    url: str
    key: str
    schema: str = "public"


class E2ISupabaseLoader:
    """
    Load ML-compliant E2I data into Supabase with proper split tracking.
    """

    def __init__(self, config: SupabaseConfig, data_prefix: str = "e2i_ml_compliant", data_dir: str = "."):
        self.config = config
        self.data_prefix = data_prefix
        self.data_dir = data_dir
        self.client: Client = create_client(config.url, config.key)
        self.split_config_id: Optional[str] = None

        # Data containers
        self.split_data = {}
        self.hcp_profiles = []
        self.preprocessing_meta = {}
        self.leakage_audits = []
        self.pipeline_config = {}
        
    def load_local_data(self):
        """Load all generated data files from the data directory"""
        logger.info(f"Loading local data files from: {self.data_dir}")

        # Load split data
        for split in ['train', 'validation', 'test', 'holdout']:
            filename = os.path.join(self.data_dir, f"{self.data_prefix}_{split}.json")
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.split_data[split] = json.load(f)
                logger.info(f"  Loaded {split} data")
            else:
                logger.warning(f"  Missing {split} data file: {filename}")

        # Load HCP profiles
        hcp_file = os.path.join(self.data_dir, f"{self.data_prefix}_hcp_profiles.json")
        if os.path.exists(hcp_file):
            with open(hcp_file, 'r') as f:
                self.hcp_profiles = json.load(f)['hcp_profiles']
            logger.info(f"  Loaded {len(self.hcp_profiles)} HCP profiles")
        else:
            logger.warning(f"  Missing HCP profiles file: {hcp_file}")

        # Load preprocessing metadata
        meta_file = os.path.join(self.data_dir, f"{self.data_prefix}_preprocessing_metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                self.preprocessing_meta = json.load(f)
            logger.info("  Loaded preprocessing metadata")
        else:
            logger.warning(f"  Missing preprocessing metadata file: {meta_file}")

        # Load leakage audit
        audit_file = os.path.join(self.data_dir, f"{self.data_prefix}_leakage_audit.json")
        if os.path.exists(audit_file):
            with open(audit_file, 'r') as f:
                self.leakage_audits = json.load(f)
            logger.info(f"  Loaded {len(self.leakage_audits)} audit records")
        else:
            logger.warning(f"  Missing leakage audit file: {audit_file}")

        # Load pipeline config
        config_file = os.path.join(self.data_dir, "e2i_pipeline_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.pipeline_config = json.load(f)
            logger.info("  Loaded pipeline config")
        else:
            logger.warning(f"  Missing pipeline config file: {config_file}")
    
    def create_split_configuration(self, config_name: str = "e2i_pilot_v1"):
        """Create or get split configuration in registry"""
        logger.info(f"Creating split configuration: {config_name}")
        
        # Extract dates from pipeline config
        splits = self.pipeline_config.get('data_splits', {})
        
        # Prepare configuration record
        config_record = {
            "config_name": config_name,
            "config_version": "1.0.0",
            "train_ratio": 0.60,
            "validation_ratio": 0.20,
            "test_ratio": 0.15,
            "holdout_ratio": 0.05,
            "data_start_date": splits.get('train', {}).get('start_date', '2024-01-01')[:10],
            "data_end_date": splits.get('holdout', {}).get('end_date', '2025-09-28')[:10],
            "train_end_date": splits.get('train', {}).get('end_date', '2025-01-16')[:10],
            "validation_end_date": splits.get('validation', {}).get('end_date', '2025-05-30')[:10],
            "test_end_date": splits.get('test', {}).get('end_date', '2025-09-09')[:10],
            "temporal_gap_days": self.pipeline_config.get('temporal_config', {}).get('gap_days', 7),
            "patient_level_isolation": True,
            "split_strategy": "chronological",
            "random_seed": 42,
            "is_active": True,
            "created_by": "e2i_loader",
            "notes": "Auto-generated from ML-compliant data generator"
        }
        
        # Check if config exists
        result = self.client.table('ml_split_registry').select('split_config_id').eq('config_name', config_name).execute()
        
        if result.data:
            self.split_config_id = result.data[0]['split_config_id']
            logger.info(f"  Using existing config: {self.split_config_id}")
        else:
            # Insert new config
            result = self.client.table('ml_split_registry').insert(config_record).execute()
            self.split_config_id = result.data[0]['split_config_id']
            logger.info(f"  Created new config: {self.split_config_id}")
        
        return self.split_config_id
    
    def upload_hcp_profiles(self):
        """Upload HCP profiles to Supabase"""
        logger.info(f"Uploading {len(self.hcp_profiles)} HCP profiles...")
        
        # HCPs are reference data - not split-specific
        batch_size = 100
        for i in range(0, len(self.hcp_profiles), batch_size):
            batch = self.hcp_profiles[i:i+batch_size]
            
            # Prepare records
            records = []
            for hcp in batch:
                record = {
                    "hcp_id": hcp['hcp_id'],
                    "npi": hcp.get('npi'),
                    "first_name": hcp.get('first_name'),
                    "last_name": hcp.get('last_name'),
                    "specialty": hcp.get('specialty'),
                    "sub_specialty": hcp.get('sub_specialty'),
                    "practice_type": hcp.get('practice_type'),
                    "practice_size": hcp.get('practice_size'),
                    "geographic_region": hcp.get('geographic_region'),
                    "state": hcp.get('state'),
                    "priority_tier": hcp.get('priority_tier'),
                    "decile": hcp.get('decile'),
                    "total_patient_volume": hcp.get('total_patient_volume'),
                    "digital_engagement_score": hcp.get('digital_engagement_score'),
                    "preferred_channel": hcp.get('preferred_channel'),
                    "adoption_category": hcp.get('adoption_category'),
                    "coverage_status": hcp.get('coverage_status')
                }
                records.append(record)
            
            # Upsert to handle duplicates
            self.client.table('hcp_profiles').upsert(
                records, 
                on_conflict='hcp_id'
            ).execute()
        
        logger.info("  HCP profiles uploaded")
    
    def upload_patient_journeys(self):
        """Upload patient journeys with split assignments"""
        logger.info("Uploading patient journeys...")
        
        total = 0
        for split_name, split_data in self.split_data.items():
            journeys = split_data.get('data', {}).get('patient_journeys', [])
            if not journeys:
                continue
            
            logger.info(f"  Processing {split_name}: {len(journeys)} journeys")
            
            batch_size = 100
            for i in range(0, len(journeys), batch_size):
                batch = journeys[i:i+batch_size]
                
                records = []
                for journey in batch:
                    record = {
                        "patient_journey_id": journey['patient_journey_id'],
                        "patient_id": journey['patient_id'],
                        "patient_hash": journey.get('patient_hash'),
                        "journey_start_date": journey.get('journey_start_date'),
                        "journey_end_date": journey.get('journey_end_date'),
                        "journey_duration_days": journey.get('journey_duration_days'),
                        "journey_stage": journey.get('journey_stage'),
                        "journey_status": journey.get('journey_status'),
                        "primary_diagnosis_code": journey.get('primary_diagnosis_code'),
                        "primary_diagnosis_desc": journey.get('primary_diagnosis_desc'),
                        "brand": journey.get('brand'),
                        "age_group": journey.get('age_group'),
                        "gender": journey.get('gender'),
                        "geographic_region": journey.get('geographic_region'),
                        "state": journey.get('state'),
                        "insurance_type": journey.get('insurance_type'),
                        "data_quality_score": journey.get('data_quality_score'),
                        "comorbidities": journey.get('comorbidities'),
                        # Split tracking
                        "data_split": split_name,
                        "split_config_id": self.split_config_id
                    }
                    records.append(record)
                
                self.client.table('patient_journeys').upsert(
                    records,
                    on_conflict='patient_journey_id'
                ).execute()
                
                total += len(batch)
        
        logger.info(f"  Uploaded {total} patient journeys")
        
        # Record patient split assignments
        self._record_patient_assignments()
    
    def _record_patient_assignments(self):
        """Record patient split assignments in assignment table"""
        logger.info("Recording patient split assignments...")
        
        assignments = []
        for split_name, split_data in self.split_data.items():
            journeys = split_data.get('data', {}).get('patient_journeys', [])
            for journey in journeys:
                assignments.append({
                    "split_config_id": self.split_config_id,
                    "patient_id": journey['patient_id'],
                    "assigned_split": split_name,
                    "assignment_reason": "chronological"
                })
        
        if assignments:
            batch_size = 500
            for i in range(0, len(assignments), batch_size):
                batch = assignments[i:i+batch_size]
                self.client.table('ml_patient_split_assignments').upsert(
                    batch,
                    on_conflict='split_config_id,patient_id'
                ).execute()
        
        logger.info(f"  Recorded {len(assignments)} patient assignments")
    
    def upload_treatment_events(self):
        """Upload treatment events with split tracking"""
        logger.info("Uploading treatment events...")
        
        total = 0
        for split_name, split_data in self.split_data.items():
            events = split_data.get('data', {}).get('treatment_events', [])
            if not events:
                continue
            
            batch_size = 200
            for i in range(0, len(events), batch_size):
                batch = events[i:i+batch_size]
                
                records = []
                for event in batch:
                    record = {
                        "treatment_event_id": event['treatment_event_id'],
                        "patient_journey_id": event.get('patient_journey_id'),
                        "patient_id": event.get('patient_id'),
                        "hcp_id": event.get('hcp_id'),
                        "event_date": event.get('event_date'),
                        "event_type": event.get('event_type'),
                        "brand": event.get('brand'),
                        "icd_codes": event.get('icd_codes'),
                        "sequence_number": event.get('sequence_number'),
                        "days_from_diagnosis": event.get('days_from_diagnosis'),
                        "cost": event.get('cost'),
                        "outcome_indicator": event.get('outcome_indicator'),
                        "adverse_event_flag": event.get('adverse_event_flag'),
                        # Split tracking
                        "data_split": split_name,
                        "split_config_id": self.split_config_id
                    }
                    records.append(record)
                
                self.client.table('treatment_events').upsert(
                    records,
                    on_conflict='treatment_event_id'
                ).execute()
                
                total += len(batch)
        
        logger.info(f"  Uploaded {total} treatment events")
    
    def upload_ml_predictions(self):
        """Upload ML predictions with split tracking"""
        logger.info("Uploading ML predictions...")
        
        total = 0
        for split_name, split_data in self.split_data.items():
            predictions = split_data.get('data', {}).get('ml_predictions', [])
            if not predictions:
                continue
            
            batch_size = 200
            for i in range(0, len(predictions), batch_size):
                batch = predictions[i:i+batch_size]
                
                records = []
                for pred in batch:
                    record = {
                        "prediction_id": pred['prediction_id'],
                        "model_version": pred.get('model_version'),
                        "model_type": pred.get('model_type'),
                        "prediction_timestamp": pred.get('prediction_timestamp'),
                        "patient_id": pred.get('patient_id'),
                        "hcp_id": pred.get('hcp_id'),
                        "prediction_type": pred.get('prediction_type'),
                        "prediction_value": pred.get('prediction_value'),
                        "prediction_class": pred.get('prediction_class'),
                        "confidence_score": pred.get('confidence_score'),
                        "model_auc": pred.get('model_auc'),
                        "model_precision": pred.get('model_precision'),
                        "model_recall": pred.get('model_recall'),
                        "calibration_score": pred.get('calibration_score'),
                        "causal_confidence": pred.get('causal_confidence'),
                        "treatment_effect_estimate": pred.get('treatment_effect_estimate'),
                        "heterogeneous_effect": pred.get('heterogeneous_effect'),
                        "segment_assignment": pred.get('segment_assignment'),
                        "features_available_at_prediction": pred.get('features_available_at_prediction', {}),
                        # Split tracking
                        "data_split": split_name,
                        "split_config_id": self.split_config_id
                    }
                    records.append(record)
                
                self.client.table('ml_predictions').upsert(
                    records,
                    on_conflict='prediction_id'
                ).execute()
                
                total += len(batch)
        
        logger.info(f"  Uploaded {total} ML predictions")
    
    def upload_triggers(self):
        """Upload triggers with split tracking"""
        logger.info("Uploading triggers...")
        
        total = 0
        for split_name, split_data in self.split_data.items():
            triggers = split_data.get('data', {}).get('triggers', [])
            if not triggers:
                continue
            
            batch_size = 200
            for i in range(0, len(triggers), batch_size):
                batch = triggers[i:i+batch_size]
                
                records = []
                for trigger in batch:
                    record = {
                        "trigger_id": trigger['trigger_id'],
                        "patient_id": trigger.get('patient_id'),
                        "hcp_id": trigger.get('hcp_id'),
                        "trigger_timestamp": trigger.get('trigger_timestamp'),
                        "trigger_type": trigger.get('trigger_type'),
                        "priority": trigger.get('priority'),
                        "confidence_score": trigger.get('confidence_score'),
                        "lead_time_days": trigger.get('lead_time_days'),
                        "expiration_date": trigger.get('expiration_date'),
                        "delivery_status": trigger.get('delivery_status'),
                        "acceptance_status": trigger.get('acceptance_status'),
                        "causal_chain": trigger.get('causal_chain'),
                        # Split tracking
                        "data_split": split_name,
                        "split_config_id": self.split_config_id
                    }
                    records.append(record)
                
                self.client.table('triggers').upsert(
                    records,
                    on_conflict='trigger_id'
                ).execute()
                
                total += len(batch)
        
        logger.info(f"  Uploaded {total} triggers")
    
    def upload_agent_activities(self):
        """Upload agent activities with split tracking"""
        logger.info("Uploading agent activities...")
        
        total = 0
        for split_name, split_data in self.split_data.items():
            activities = split_data.get('data', {}).get('agent_activities', [])
            if not activities:
                continue
            
            batch_size = 200
            for i in range(0, len(activities), batch_size):
                batch = activities[i:i+batch_size]
                
                records = []
                for activity in batch:
                    record = {
                        "activity_id": activity['activity_id'],
                        "agent_name": activity.get('agent_name'),
                        "activity_timestamp": activity.get('activity_timestamp'),
                        "activity_type": activity.get('activity_type'),
                        "workstream": activity.get('workstream'),
                        "processing_duration_ms": activity.get('processing_duration_ms'),
                        "records_processed": activity.get('records_processed'),
                        "causal_paths_analyzed": activity.get('causal_paths_analyzed'),
                        "confidence_level": activity.get('confidence_level'),
                        "impact_estimate": activity.get('impact_estimate'),
                        "roi_estimate": activity.get('roi_estimate'),
                        "status": activity.get('status'),
                        # Split tracking
                        "data_split": split_name,
                        "split_config_id": self.split_config_id
                    }
                    records.append(record)
                
                self.client.table('agent_activities').upsert(
                    records,
                    on_conflict='activity_id'
                ).execute()
                
                total += len(batch)
        
        logger.info(f"  Uploaded {total} agent activities")
    
    def upload_business_metrics(self):
        """Upload business metrics with split tracking"""
        logger.info("Uploading business metrics...")
        
        total = 0
        for split_name, split_data in self.split_data.items():
            metrics = split_data.get('data', {}).get('business_metrics', [])
            if not metrics:
                continue
            
            batch_size = 200
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i:i+batch_size]
                
                records = []
                for metric in batch:
                    record = {
                        "metric_id": metric['metric_id'],
                        "metric_date": metric.get('metric_date'),
                        "metric_type": metric.get('metric_type'),
                        "brand": metric.get('brand'),
                        "region": metric.get('region'),
                        "value": metric.get('value'),
                        "target": metric.get('target'),
                        "achievement_rate": metric.get('achievement_rate'),
                        "year_over_year_change": metric.get('year_over_year_change'),
                        "roi": metric.get('roi'),
                        "statistical_significance": metric.get('statistical_significance'),
                        # Split tracking
                        "data_split": split_name,
                        "split_config_id": self.split_config_id
                    }
                    records.append(record)
                
                self.client.table('business_metrics').upsert(
                    records,
                    on_conflict='metric_id'
                ).execute()
                
                total += len(batch)
        
        logger.info(f"  Uploaded {total} business metrics")
    
    def upload_preprocessing_metadata(self):
        """Upload preprocessing metadata to Supabase"""
        logger.info("Uploading preprocessing metadata...")
        
        if not self.preprocessing_meta:
            logger.warning("  No preprocessing metadata to upload")
            return
        
        record = {
            "split_config_id": self.split_config_id,
            "computed_on_split": self.preprocessing_meta.get('computed_on_split', 'train'),
            "computed_at": self.preprocessing_meta.get('computed_timestamp', datetime.now().isoformat()),
            "feature_means": self.preprocessing_meta.get('feature_means', {}),
            "feature_stds": self.preprocessing_meta.get('feature_stds', {}),
            "feature_mins": self.preprocessing_meta.get('feature_mins', {}),
            "feature_maxs": self.preprocessing_meta.get('feature_maxs', {}),
            "categorical_encodings": self.preprocessing_meta.get('categorical_encodings', {}),
            "feature_list": list(self.preprocessing_meta.get('feature_means', {}).keys()),
            "preprocessing_pipeline_version": "2.0.0"
        }
        
        # Upsert (one metadata record per config)
        self.client.table('ml_preprocessing_metadata').upsert(
            record,
            on_conflict='split_config_id'
        ).execute()
        
        logger.info("  Preprocessing metadata uploaded")
    
    def upload_leakage_audit(self):
        """Upload leakage audit results"""
        logger.info("Uploading leakage audit results...")
        
        if not self.leakage_audits:
            logger.warning("  No audit results to upload")
            return
        
        records = []
        for audit in self.leakage_audits:
            record = {
                "split_config_id": self.split_config_id,
                "audit_timestamp": audit.get('timestamp', datetime.now().isoformat()),
                "check_type": audit.get('check_type'),
                "passed": audit.get('passed'),
                "severity": audit.get('severity', 'info'),
                "details": audit.get('details'),
                "audited_by": "e2i_loader"
            }
            records.append(record)
        
        self.client.table('ml_leakage_audit').insert(records).execute()
        
        logger.info(f"  Uploaded {len(records)} audit records")
    
    def run_full_load(self, config_name: str = "e2i_pilot_v1"):
        """Run the complete data load process"""
        logger.info("="*70)
        logger.info("E2I Supabase Data Loader - Starting Full Load")
        logger.info("="*70)
        
        # Step 1: Load local data
        self.load_local_data()
        
        # Step 2: Create/get split configuration
        self.create_split_configuration(config_name)
        
        # Step 3: Upload reference data
        self.upload_hcp_profiles()
        
        # Step 4: Upload split-tracked data
        self.upload_patient_journeys()
        self.upload_treatment_events()
        self.upload_ml_predictions()
        self.upload_triggers()
        self.upload_agent_activities()
        self.upload_business_metrics()
        
        # Step 5: Upload ML pipeline metadata
        self.upload_preprocessing_metadata()
        self.upload_leakage_audit()
        
        logger.info("="*70)
        logger.info("Data load complete!")
        logger.info("="*70)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print load summary"""
        logger.info("\nLoad Summary:")
        logger.info(f"  Split Config ID: {self.split_config_id}")
        
        for split_name, split_data in self.split_data.items():
            data = split_data.get('data', {})
            logger.info(f"\n  {split_name.upper()}:")
            logger.info(f"    Patient Journeys:  {len(data.get('patient_journeys', []))}")
            logger.info(f"    Treatment Events:  {len(data.get('treatment_events', []))}")
            logger.info(f"    ML Predictions:    {len(data.get('ml_predictions', []))}")
            logger.info(f"    Triggers:          {len(data.get('triggers', []))}")


def main():
    """Main entry point"""
    # Get Supabase credentials from environment or config
    supabase_url = os.environ.get('SUPABASE_URL', 'your-project-url')
    supabase_key = os.environ.get('SUPABASE_KEY', 'your-anon-key')

    if supabase_url == 'your-project-url':
        logger.error("Please set SUPABASE_URL and SUPABASE_KEY environment variables")
        logger.info("Example:")
        logger.info("  export SUPABASE_URL='https://xxx.supabase.co'")
        logger.info("  export SUPABASE_KEY='your-anon-key'")
        return

    # Create config
    config = SupabaseConfig(
        url=supabase_url,
        key=supabase_key
    )

    # Determine data directory - use the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create loader and run
    loader = E2ISupabaseLoader(
        config,
        data_prefix="e2i_ml_compliant",
        data_dir=script_dir
    )
    loader.run_full_load(config_name="e2i_pilot_v1")


if __name__ == "__main__":
    main()
