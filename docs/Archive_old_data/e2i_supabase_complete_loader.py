#!/usr/bin/env python3
"""
E2I Causal Analytics - Complete Supabase Data Loader

This script loads ALL ML-compliant generated data into a fresh Supabase database.
Run this after executing e2i_supabase_complete_schema.sql in your Supabase SQL Editor.

Usage:
    Simply run the script - it loads credentials from .env in the root directory:
       python e2i_supabase_complete_loader.py

Features:
    - Creates split configuration in registry
    - Uploads all entity types with proper split tracking
    - Records patient split assignments
    - Stores preprocessing metadata
    - Runs and records leakage audit
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import time
from pathlib import Path

# Load environment variables from .env file in root directory
from dotenv import load_dotenv

# Find the root directory (parent of e2i_ml_compliant_package)
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection"""
    url: str
    key: str


class E2ICompleteLoader:
    """
    Complete data loader for E2I Causal Analytics Dashboard.
    Loads all generated data into Supabase with proper ML split tracking.
    """
    
    def __init__(
        self, 
        supabase_url: str,
        supabase_key: str,
        data_prefix: str = "e2i_ml_compliant"
    ):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.data_prefix = data_prefix
        self.client = None
        self.split_config_id: Optional[str] = None
        
        # Data containers
        self.split_data = {}
        self.hcp_profiles = []
        self.preprocessing_meta = {}
        self.leakage_audits = []
        self.pipeline_config = {}
        
        # Statistics
        self.stats = {
            'hcp_profiles': 0,
            'patient_journeys': 0,
            'treatment_events': 0,
            'ml_predictions': 0,
            'triggers': 0,
            'agent_activities': 0,
            'business_metrics': 0
        }
        
        # Initialize Supabase client
        self._init_client()
    
    def _init_client(self):
        """Initialize Supabase client"""
        try:
            from supabase import create_client, Client
            self.client = create_client(self.supabase_url, self.supabase_key)
            logger.info("✅ Supabase client initialized")
        except ImportError:
            logger.error("❌ Supabase client not installed.")
            logger.error("   Run: pip install supabase")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            sys.exit(1)
    
    def load_local_data(self):
        """Load all generated data files from local directory"""
        logger.info("\n📂 Loading local data files...")
        
        # Load split data files
        for split in ['train', 'validation', 'test', 'holdout']:
            filename = f"{self.data_prefix}_{split}.json"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.split_data[split] = json.load(f)
                patient_count = len(self.split_data[split].get('data', {}).get('patient_journeys', []))
                logger.info(f"   ✓ Loaded {split}: {patient_count} patients")
            else:
                logger.warning(f"   ⚠ File not found: {filename}")
        
        # Load HCP profiles
        hcp_file = f"{self.data_prefix}_hcp_profiles.json"
        if os.path.exists(hcp_file):
            with open(hcp_file, 'r') as f:
                self.hcp_profiles = json.load(f).get('hcp_profiles', [])
            logger.info(f"   ✓ Loaded {len(self.hcp_profiles)} HCP profiles")
        else:
            logger.warning(f"   ⚠ HCP profiles not found: {hcp_file}")
        
        # Load preprocessing metadata
        meta_file = f"{self.data_prefix}_preprocessing_metadata.json"
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                self.preprocessing_meta = json.load(f)
            logger.info("   ✓ Loaded preprocessing metadata")
        
        # Load leakage audit
        audit_file = f"{self.data_prefix}_leakage_audit.json"
        if os.path.exists(audit_file):
            with open(audit_file, 'r') as f:
                self.leakage_audits = json.load(f)
            logger.info(f"   ✓ Loaded {len(self.leakage_audits)} audit records")
        
        # Load pipeline config
        config_file = "e2i_pipeline_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.pipeline_config = json.load(f)
            logger.info("   ✓ Loaded pipeline config")
        
        if not self.split_data:
            logger.error("❌ No data files found. Run e2i_ml_compliant_data_generator.py first.")
            sys.exit(1)
    
    def create_split_configuration(self, config_name: str = "e2i_pilot_v1"):
        """Create split configuration in the registry"""
        logger.info(f"\n⚙️  Creating split configuration: {config_name}")
        
        # Extract dates from pipeline config or use defaults
        splits = self.pipeline_config.get('data_splits', {})
        
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
            "created_by": "e2i_complete_loader",
            "notes": "Auto-generated from ML-compliant data generator"
        }
        
        try:
            # Check if config already exists
            result = self.client.table('ml_split_registry')\
                .select('split_config_id')\
                .eq('config_name', config_name)\
                .execute()
            
            if result.data:
                self.split_config_id = result.data[0]['split_config_id']
                logger.info(f"   Using existing config: {self.split_config_id[:8]}...")
            else:
                # Insert new config
                result = self.client.table('ml_split_registry')\
                    .insert(config_record)\
                    .execute()
                self.split_config_id = result.data[0]['split_config_id']
                logger.info(f"   Created new config: {self.split_config_id[:8]}...")
            
            return self.split_config_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create split config: {e}")
            raise
    
    def _batch_upsert(
        self, 
        table_name: str, 
        records: List[Dict], 
        conflict_column: str,
        batch_size: int = 100
    ) -> int:
        """Batch upsert records to a table"""
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            try:
                self.client.table(table_name).upsert(
                    batch,
                    on_conflict=conflict_column
                ).execute()
                total += len(batch)
            except Exception as e:
                logger.error(f"   Error upserting to {table_name}: {e}")
                # Continue with next batch
        return total
    
    def upload_hcp_profiles(self):
        """Upload HCP profiles"""
        logger.info("\n👨‍⚕️ Uploading HCP profiles...")
        
        if not self.hcp_profiles:
            logger.warning("   No HCP profiles to upload")
            return
        
        records = []
        for hcp in self.hcp_profiles:
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
                "coverage_status": hcp.get('coverage_status', True)
            }
            records.append(record)
        
        count = self._batch_upsert('hcp_profiles', records, 'hcp_id')
        self.stats['hcp_profiles'] = count
        logger.info(f"   ✓ Uploaded {count} HCP profiles")
    
    def upload_patient_journeys(self):
        """Upload patient journeys with split assignments"""
        logger.info("\n🏥 Uploading patient journeys...")
        
        all_records = []
        assignments = []
        
        for split_name, split_data in self.split_data.items():
            journeys = split_data.get('data', {}).get('patient_journeys', [])
            
            for journey in journeys:
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
                    "comorbidities": journey.get('comorbidities', []),
                    "data_split": split_name,
                    "split_config_id": self.split_config_id
                }
                all_records.append(record)
                
                # Track assignment
                assignments.append({
                    "split_config_id": self.split_config_id,
                    "patient_id": journey['patient_id'],
                    "assigned_split": split_name,
                    "assignment_reason": "chronological"
                })
        
        # Upload journeys
        count = self._batch_upsert('patient_journeys', all_records, 'patient_journey_id')
        self.stats['patient_journeys'] = count
        logger.info(f"   ✓ Uploaded {count} patient journeys")
        
        # Upload assignments
        if assignments:
            assign_count = self._batch_upsert(
                'ml_patient_split_assignments', 
                assignments, 
                'split_config_id,patient_id',
                batch_size=500
            )
            logger.info(f"   ✓ Recorded {assign_count} patient split assignments")
    
    def upload_treatment_events(self):
        """Upload treatment events"""
        logger.info("\n💊 Uploading treatment events...")
        
        all_records = []
        
        for split_name, split_data in self.split_data.items():
            events = split_data.get('data', {}).get('treatment_events', [])
            
            for event in events:
                record = {
                    "treatment_event_id": event['treatment_event_id'],
                    "patient_journey_id": event.get('patient_journey_id'),
                    "patient_id": event.get('patient_id'),
                    "hcp_id": event.get('hcp_id'),
                    "event_date": event.get('event_date'),
                    "event_type": event.get('event_type'),
                    "brand": event.get('brand'),
                    "icd_codes": event.get('icd_codes', []),
                    "sequence_number": event.get('sequence_number'),
                    "days_from_diagnosis": event.get('days_from_diagnosis'),
                    "cost": event.get('cost'),
                    "outcome_indicator": event.get('outcome_indicator'),
                    "adverse_event_flag": event.get('adverse_event_flag', False),
                    "data_split": split_name,
                    "split_config_id": self.split_config_id
                }
                all_records.append(record)
        
        count = self._batch_upsert('treatment_events', all_records, 'treatment_event_id', batch_size=200)
        self.stats['treatment_events'] = count
        logger.info(f"   ✓ Uploaded {count} treatment events")
    
    def upload_ml_predictions(self):
        """Upload ML predictions"""
        logger.info("\n🤖 Uploading ML predictions...")
        
        all_records = []
        
        for split_name, split_data in self.split_data.items():
            predictions = split_data.get('data', {}).get('ml_predictions', [])
            
            for pred in predictions:
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
                    "data_split": split_name,
                    "split_config_id": self.split_config_id
                }
                all_records.append(record)
        
        count = self._batch_upsert('ml_predictions', all_records, 'prediction_id', batch_size=200)
        self.stats['ml_predictions'] = count
        logger.info(f"   ✓ Uploaded {count} ML predictions")
    
    def upload_triggers(self):
        """Upload triggers"""
        logger.info("\n⚡ Uploading triggers...")
        
        all_records = []
        
        for split_name, split_data in self.split_data.items():
            triggers = split_data.get('data', {}).get('triggers', [])
            
            for trigger in triggers:
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
                    "causal_chain": trigger.get('causal_chain', {}),
                    "data_split": split_name,
                    "split_config_id": self.split_config_id
                }
                all_records.append(record)
        
        count = self._batch_upsert('triggers', all_records, 'trigger_id', batch_size=200)
        self.stats['triggers'] = count
        logger.info(f"   ✓ Uploaded {count} triggers")
    
    def upload_agent_activities(self):
        """Upload agent activities"""
        logger.info("\n🔧 Uploading agent activities...")
        
        all_records = []
        
        for split_name, split_data in self.split_data.items():
            activities = split_data.get('data', {}).get('agent_activities', [])
            
            for activity in activities:
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
                    "data_split": split_name,
                    "split_config_id": self.split_config_id
                }
                all_records.append(record)
        
        count = self._batch_upsert('agent_activities', all_records, 'activity_id', batch_size=200)
        self.stats['agent_activities'] = count
        logger.info(f"   ✓ Uploaded {count} agent activities")
    
    def upload_business_metrics(self):
        """Upload business metrics"""
        logger.info("\n📊 Uploading business metrics...")
        
        all_records = []
        
        for split_name, split_data in self.split_data.items():
            metrics = split_data.get('data', {}).get('business_metrics', [])
            
            for metric in metrics:
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
                    "data_split": split_name,
                    "split_config_id": self.split_config_id
                }
                all_records.append(record)
        
        count = self._batch_upsert('business_metrics', all_records, 'metric_id', batch_size=200)
        self.stats['business_metrics'] = count
        logger.info(f"   ✓ Uploaded {count} business metrics")
    
    def upload_preprocessing_metadata(self):
        """Upload preprocessing metadata (computed on training data only)"""
        logger.info("\n📐 Uploading preprocessing metadata...")
        
        if not self.preprocessing_meta:
            logger.warning("   No preprocessing metadata to upload")
            return
        
        record = {
            "split_config_id": self.split_config_id,
            "computed_on_split": "train",  # MUST be train
            "computed_at": self.preprocessing_meta.get('computed_timestamp', datetime.now().isoformat()),
            "feature_means": self.preprocessing_meta.get('feature_means', {}),
            "feature_stds": self.preprocessing_meta.get('feature_stds', {}),
            "feature_mins": self.preprocessing_meta.get('feature_mins', {}),
            "feature_maxs": self.preprocessing_meta.get('feature_maxs', {}),
            "categorical_encodings": self.preprocessing_meta.get('categorical_encodings', {}),
            "feature_list": list(self.preprocessing_meta.get('feature_means', {}).keys()),
            "preprocessing_pipeline_version": "2.0.0"
        }
        
        try:
            self.client.table('ml_preprocessing_metadata').upsert(
                record,
                on_conflict='split_config_id'
            ).execute()
            logger.info("   ✓ Preprocessing metadata uploaded")
        except Exception as e:
            logger.error(f"   Error uploading preprocessing metadata: {e}")
    
    def upload_leakage_audit(self):
        """Upload leakage audit results"""
        logger.info("\n🔍 Uploading leakage audit results...")
        
        if not self.leakage_audits:
            logger.warning("   No audit results to upload")
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
                "audited_by": "e2i_complete_loader"
            }
            records.append(record)
        
        try:
            self.client.table('ml_leakage_audit').insert(records).execute()
            logger.info(f"   ✓ Uploaded {len(records)} audit records")
        except Exception as e:
            logger.error(f"   Error uploading audit records: {e}")
    
    def run_database_audit(self):
        """Run leakage audit using database function"""
        logger.info("\n🔒 Running database leakage audit...")
        
        try:
            # Call the database function
            result = self.client.rpc(
                'run_leakage_audit',
                {'p_split_config_id': self.split_config_id}
            ).execute()
            
            if result.data:
                for check in result.data:
                    status = "✅ PASS" if check['passed'] else "❌ FAIL"
                    logger.info(f"   {status} {check['check_type']}: {check['details']}")
            
        except Exception as e:
            logger.warning(f"   Could not run database audit: {e}")
    
    def print_summary(self):
        """Print upload summary"""
        logger.info("\n" + "="*60)
        logger.info("📋 UPLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Split Config ID: {self.split_config_id}")
        logger.info("")
        logger.info("Records uploaded:")
        for table, count in self.stats.items():
            logger.info(f"   {table}: {count:,}")
        
        logger.info("")
        logger.info("Data by split:")
        for split_name, split_data in self.split_data.items():
            data = split_data.get('data', {})
            patients = len(data.get('patient_journeys', []))
            events = len(data.get('treatment_events', []))
            logger.info(f"   {split_name}: {patients} patients, {events} events")
        
        logger.info("")
        logger.info("="*60)
        logger.info("✅ Data upload complete!")
        logger.info("="*60)
    
    def run_full_load(self, config_name: str = "e2i_pilot_v1"):
        """Run the complete data load process"""
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("🚀 E2I Causal Analytics - Complete Data Loader")
        logger.info("="*60)
        logger.info(f"Supabase URL: {self.supabase_url[:40]}...")
        logger.info(f"Data prefix: {self.data_prefix}")
        
        # Step 1: Load local data
        self.load_local_data()
        
        # Step 2: Create split configuration
        self.create_split_configuration(config_name)
        
        # Step 3: Upload all data
        self.upload_hcp_profiles()
        self.upload_patient_journeys()
        self.upload_treatment_events()
        self.upload_ml_predictions()
        self.upload_triggers()
        self.upload_agent_activities()
        self.upload_business_metrics()
        
        # Step 4: Upload ML metadata
        self.upload_preprocessing_metadata()
        self.upload_leakage_audit()
        
        # Step 5: Run database audit
        self.run_database_audit()
        
        # Print summary
        elapsed = time.time() - start_time
        self.print_summary()
        logger.info(f"\nTotal time: {elapsed:.1f} seconds")


def main():
    """Main entry point"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     E2I Causal Analytics - Supabase Data Loader               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Get credentials from environment (loaded from .env file)
    supabase_url = os.environ.get('SUPABASE_URL')
    # Support both SUPABASE_SERVICE_KEY and SUPABASE_KEY for flexibility
    supabase_key = os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials!")
        print("")
        print(f"Looked for .env file at: {ENV_PATH}")
        print("")
        print("Please ensure .env file contains:")
        print("  SUPABASE_URL=https://your-project.supabase.co")
        print("  SUPABASE_SERVICE_KEY=your-service-key")
        print("")
        print("Or run with arguments:")
        print("  python e2i_supabase_complete_loader.py <url> <key>")
        print("")

        # Check for command line args
        if len(sys.argv) >= 3:
            supabase_url = sys.argv[1]
            supabase_key = sys.argv[2]
        else:
            sys.exit(1)
    
    # Confirm before proceeding
    print(f"Supabase URL: {supabase_url[:50]}...")
    print("")
    
    response = input("Proceed with data upload? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Create loader and run
    loader = E2ICompleteLoader(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        data_prefix="e2i_ml_compliant"
    )
    
    loader.run_full_load(config_name="e2i_pilot_v1")


if __name__ == "__main__":
    main()
