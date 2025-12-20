#!/usr/bin/env python3
"""
E2I Causal Analytics - Example ML Pipeline with Leakage Prevention

This script demonstrates how to properly use the ML-compliant data
with scikit-learn pipelines to prevent data leakage.

Key Principles:
1. Fit preprocessors ONLY on training data
2. Use pipelines to encapsulate transformations
3. Respect temporal ordering for time-series
4. Validate with proper hold-out strategy
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score


class E2ILeakageFreeMLPipeline:
    """
    ML Pipeline for E2I Causal Analytics that prevents data leakage.
    
    This class demonstrates:
    - Loading data with split awareness
    - Building preprocessing pipelines
    - Training models on correct splits
    - Evaluating without contamination
    """
    
    def __init__(self, data_prefix: str = "e2i_ml_compliant"):
        self.data_prefix = data_prefix
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.holdout_data = None
        self.hcp_profiles = None
        self.preprocessing_meta = None
        self.pipeline = None
        self.label_encoders = {}
        
    def load_all_data(self):
        """Load all split data files"""
        print("📂 Loading data splits...")
        
        # Load each split
        self.train_data = self._load_split("train")
        self.val_data = self._load_split("validation")
        self.test_data = self._load_split("test")
        self.holdout_data = self._load_split("holdout")
        
        # Load HCP profiles
        with open(f'{self.data_prefix}_hcp_profiles.json', 'r') as f:
            self.hcp_profiles = json.load(f)['hcp_profiles']
        
        # Load preprocessing metadata
        with open(f'{self.data_prefix}_preprocessing_metadata.json', 'r') as f:
            self.preprocessing_meta = json.load(f)
        
        # Verify preprocessing was computed on training data
        assert self.preprocessing_meta['computed_on_split'] == 'train', \
            "CRITICAL: Preprocessing metadata must be computed on training data!"
        
        self._print_data_summary()
    
    def _load_split(self, split_name: str) -> Dict:
        """Load a specific data split"""
        with open(f'{self.data_prefix}_{split_name}.json', 'r') as f:
            return json.load(f)
    
    def _print_data_summary(self):
        """Print summary of loaded data"""
        print("\n📊 Data Summary:")
        print(f"  Training:   {len(self.train_data['data']['patient_journeys']):,} patients")
        print(f"  Validation: {len(self.val_data['data']['patient_journeys']):,} patients")
        print(f"  Test:       {len(self.test_data['data']['patient_journeys']):,} patients")
        print(f"  Holdout:    {len(self.holdout_data['data']['patient_journeys']):,} patients")
        print(f"  HCPs:       {len(self.hcp_profiles):,} profiles")
    
    def prepare_prediction_dataset(
        self, 
        split: str = "train"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML prediction.
        
        For this example, we'll predict trigger acceptance (binary classification).
        """
        # Select appropriate data
        data = {
            "train": self.train_data,
            "validation": self.val_data,
            "test": self.test_data,
            "holdout": self.holdout_data
        }[split]
        
        # Get triggers with acceptance status
        triggers = pd.DataFrame(data['data']['triggers'])
        
        if triggers.empty:
            return pd.DataFrame(), pd.Series(dtype='float64')
        
        # Filter to triggers with known outcomes
        triggers = triggers[triggers['acceptance_status'].notna()]
        
        if triggers.empty:
            return pd.DataFrame(), pd.Series(dtype='float64')
        
        # Get patient journey info
        journeys = pd.DataFrame(data['data']['patient_journeys'])
        journeys_lookup = journeys.set_index('patient_id')
        
        # Feature engineering (using ONLY information available at trigger time)
        features = []
        for _, trigger in triggers.iterrows():
            patient_id = trigger['patient_id']
            
            # Get patient info (available at trigger time)
            if patient_id in journeys_lookup.index:
                journey = journeys_lookup.loc[patient_id]
                patient_features = {
                    'trigger_type': trigger['trigger_type'],
                    'priority': trigger['priority'],
                    'confidence_score': trigger['confidence_score'],
                    'lead_time_days': trigger['lead_time_days'],
                    'brand': journey['brand'],
                    'geographic_region': journey['geographic_region'],
                    'journey_stage': journey['journey_stage'],
                    'insurance_type': journey['insurance_type'],
                    'data_quality_score': journey['data_quality_score'],
                    'journey_duration_days': journey['journey_duration_days']
                }
                features.append(patient_features)
        
        if not features:
            return pd.DataFrame(), pd.Series(dtype='float64')
        
        X = pd.DataFrame(features)
        
        # Create binary target (accepted = 1, not accepted = 0)
        y = (triggers['acceptance_status'] == 'accepted').astype(int)[:len(X)]
        
        return X, y
    
    def build_pipeline(self):
        """
        Build sklearn pipeline with preprocessing.
        
        CRITICAL: This pipeline ensures preprocessing is learned from 
        training data and applied consistently to all splits.
        """
        print("\n🔧 Building ML Pipeline...")
        
        # Define feature columns
        numerical_features = [
            'confidence_score', 
            'lead_time_days', 
            'data_quality_score',
            'journey_duration_days'
        ]
        
        categorical_features = [
            'trigger_type',
            'priority', 
            'brand', 
            'geographic_region',
            'journey_stage',
            'insurance_type'
        ]
        
        # Create column transformer
        # IMPORTANT: These transformers will be FIT only on training data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(
                    handle_unknown='ignore',  # Handle unseen categories in test
                    sparse_output=False
                ), categorical_features)
            ],
            remainder='drop'
        )
        
        # Create full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        
        print("  ✅ Pipeline created with:")
        print(f"     - {len(numerical_features)} numerical features")
        print(f"     - {len(categorical_features)} categorical features")
        print("     - StandardScaler for numerical")
        print("     - OneHotEncoder for categorical")
        print("     - GradientBoostingClassifier")
    
    def train(self):
        """
        Train the pipeline on TRAINING data only.
        
        This is where preprocessing statistics are learned.
        """
        print("\n🎯 Training Model...")
        print("  CRITICAL: Fitting ONLY on training data")
        
        X_train, y_train = self.prepare_prediction_dataset("train")
        
        if X_train.empty:
            print("  ⚠️ No training data available with outcomes")
            return False
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Positive class: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        
        # FIT pipeline - this learns preprocessing from training data
        self.pipeline.fit(X_train, y_train)
        
        # Training metrics
        train_pred = self.pipeline.predict(X_train)
        train_proba = self.pipeline.predict_proba(X_train)[:, 1]
        
        print(f"\n  Training Metrics:")
        print(f"    Accuracy:  {accuracy_score(y_train, train_pred):.3f}")
        print(f"    Precision: {precision_score(y_train, train_pred, zero_division=0):.3f}")
        print(f"    Recall:    {recall_score(y_train, train_pred, zero_division=0):.3f}")
        print(f"    F1 Score:  {f1_score(y_train, train_pred, zero_division=0):.3f}")
        if len(np.unique(y_train)) > 1:
            print(f"    ROC-AUC:   {roc_auc_score(y_train, train_proba):.3f}")
        
        return True
    
    def validate(self):
        """
        Evaluate on VALIDATION data (for hyperparameter tuning).
        
        Preprocessing transformations are APPLIED (not re-learned).
        """
        print("\n📋 Validating Model...")
        print("  Using validation split (for hyperparameter tuning)")
        
        X_val, y_val = self.prepare_prediction_dataset("validation")
        
        if X_val.empty:
            print("  ⚠️ No validation data available")
            return None
        
        # TRANSFORM (not fit!) validation data using training statistics
        val_pred = self.pipeline.predict(X_val)
        val_proba = self.pipeline.predict_proba(X_val)[:, 1]
        
        print(f"  Validation samples: {len(X_val)}")
        print(f"\n  Validation Metrics:")
        print(f"    Accuracy:  {accuracy_score(y_val, val_pred):.3f}")
        print(f"    Precision: {precision_score(y_val, val_pred, zero_division=0):.3f}")
        print(f"    Recall:    {recall_score(y_val, val_pred, zero_division=0):.3f}")
        print(f"    F1 Score:  {f1_score(y_val, val_pred, zero_division=0):.3f}")
        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, val_proba)
            print(f"    ROC-AUC:   {auc:.3f}")
            return auc
        
        return f1_score(y_val, val_pred, zero_division=0)
    
    def test(self):
        """
        Final evaluation on TEST data.
        
        ONLY use this for final model evaluation, not for tuning!
        """
        print("\n🧪 Testing Model...")
        print("  Using test split (FINAL evaluation only)")
        
        X_test, y_test = self.prepare_prediction_dataset("test")
        
        if X_test.empty:
            print("  ⚠️ No test data available")
            return
        
        # TRANSFORM test data using training statistics
        test_pred = self.pipeline.predict(X_test)
        test_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        print(f"  Test samples: {len(X_test)}")
        print(f"\n  Test Metrics (FINAL):")
        print(f"    Accuracy:  {accuracy_score(y_test, test_pred):.3f}")
        print(f"    Precision: {precision_score(y_test, test_pred, zero_division=0):.3f}")
        print(f"    Recall:    {recall_score(y_test, test_pred, zero_division=0):.3f}")
        print(f"    F1 Score:  {f1_score(y_test, test_pred, zero_division=0):.3f}")
        if len(np.unique(y_test)) > 1:
            print(f"    ROC-AUC:   {roc_auc_score(y_test, test_proba):.3f}")
        
        print("\n  Classification Report:")
        print(classification_report(y_test, test_pred, zero_division=0))
    
    def demonstrate_leakage_prevention(self):
        """
        Demonstrate what happens with and without leakage prevention.
        """
        print("\n" + "="*70)
        print("DEMONSTRATION: Data Leakage Impact")
        print("="*70)
        
        # Prepare all data
        X_train, y_train = self.prepare_prediction_dataset("train")
        X_val, y_val = self.prepare_prediction_dataset("validation")
        X_test, y_test = self.prepare_prediction_dataset("test")
        
        if X_train.empty or X_test.empty:
            print("Insufficient data for demonstration")
            return
        
        # Combine all data (simulating leakage scenario)
        X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)
        y_all = pd.concat([y_train, y_val, y_test], ignore_index=True)
        
        print("\n❌ WRONG APPROACH (with leakage):")
        print("   Fitting scaler on ALL data before splitting...")
        
        # Wrong: Fit on all data
        numerical_features = ['confidence_score', 'lead_time_days', 
                              'data_quality_score', 'journey_duration_days']
        
        wrong_scaler = StandardScaler()
        wrong_scaler.fit(X_all[numerical_features])  # LEAKAGE!
        
        print(f"   Scaler means: {wrong_scaler.mean_[:2]}")
        print("   This includes information from test set!")
        
        print("\n✅ CORRECT APPROACH (no leakage):")
        print("   Fitting scaler on TRAINING data only...")
        
        # Correct: Fit only on training
        correct_scaler = StandardScaler()
        correct_scaler.fit(X_train[numerical_features])  # No leakage
        
        print(f"   Scaler means: {correct_scaler.mean_[:2]}")
        print("   Uses only training data statistics")
        
        # Show difference
        mean_diff = np.abs(wrong_scaler.mean_ - correct_scaler.mean_).mean()
        print(f"\n   Average difference in means: {mean_diff:.4f}")
        print("   This difference can lead to optimistic performance estimates!")
    
    def save_pipeline_config(self, filename: str = "trained_pipeline_config.json"):
        """Save pipeline configuration for reproducibility"""
        config = {
            "timestamp": datetime.now().isoformat(),
            "data_prefix": self.data_prefix,
            "preprocessing_source": self.preprocessing_meta['computed_on_split'],
            "pipeline_steps": [
                step[0] for step in self.pipeline.steps
            ],
            "training_samples": len(self.train_data['data']['triggers']),
            "notes": [
                "Preprocessing fitted on training data only",
                "Validation used for hyperparameter tuning",
                "Test used for final evaluation only"
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Pipeline config saved to {filename}")


def run_full_pipeline():
    """Run the complete ML pipeline demonstration"""
    print("="*70)
    print("E2I Causal Analytics - Leakage-Free ML Pipeline Demo")
    print("="*70)
    
    # Initialize pipeline
    pipeline = E2ILeakageFreeMLPipeline()
    
    # Load data
    pipeline.load_all_data()
    
    # Build pipeline
    pipeline.build_pipeline()
    
    # Train (on training data ONLY)
    if pipeline.train():
        # Validate (for hyperparameter tuning)
        pipeline.validate()
        
        # Test (final evaluation)
        pipeline.test()
        
        # Demonstrate leakage impact
        pipeline.demonstrate_leakage_prevention()
        
        # Save configuration
        pipeline.save_pipeline_config()
    
    print("\n" + "="*70)
    print("Pipeline demonstration complete!")
    print("="*70)


if __name__ == "__main__":
    run_full_pipeline()
