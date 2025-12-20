#!/usr/bin/env python3
"""
E2I FastText Model Training and Testing Suite
=============================================

This script trains a domain-specific fastText model on the E2I corpus
and provides testing utilities to validate typo handling.

Usage:
    python e2i_fasttext_trainer.py train      # Train the model
    python e2i_fasttext_trainer.py test       # Run test suite
    python e2i_fasttext_trainer.py interactive # Interactive testing
    python e2i_fasttext_trainer.py all        # Train + test
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

# Check for fasttext installation
try:
    import fasttext
except ImportError:
    print("ERROR: fasttext not installed. Run: pip install fasttext")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

CORPUS_PATH = "e2i_corpus.txt"
MODEL_PATH = "e2i_fasttext.bin"

# Training hyperparameters optimized for typo detection
TRAINING_CONFIG = {
    "model": "skipgram",       # skipgram better for rare words (typos)
    "dim": 100,                # Embedding dimension
    "epoch": 50,               # Training epochs
    "lr": 0.05,                # Learning rate
    "wordNgrams": 3,           # Character n-gram max (critical for typos)
    "minCount": 1,             # Include all words (even rare typos)
    "minn": 2,                 # Min character n-gram length
    "maxn": 5,                 # Max character n-gram length
    "bucket": 200000,          # Hash buckets for n-grams
    "thread": 4,               # Training threads
}

# E2I canonical vocabulary for testing
CANONICAL_VOCABULARY = {
    "brands": [
        "Remibrutinib", "Fabhalta", "Kisqali", "ribociclib"
    ],
    "regions": [
        "northeast", "southeast", "south", "midwest", "west"
    ],
    "agents": [
        "orchestrator", "causal_impact", "gap_analyzer",
        "heterogeneous_optimizer", "drift_monitor", "experiment_designer",
        "health_score", "prediction_synthesizer", "resource_optimizer",
        "explainer", "feedback_learner"
    ],
    "kpis": [
        "TRx", "NRx", "NBRx", "conversion_rate", "engagement_score",
        "treatment_effect", "causal_impact", "HCP_coverage",
        "time_to_therapy", "ROI", "AUC", "acceptance_rate"
    ],
    "journey_stages": [
        "diagnosis", "initial_treatment", "treatment_optimization",
        "maintenance", "treatment_switch"
    ],
    "workstreams": [
        "WS1", "WS2", "WS3"
    ]
}

# Test cases: (typo, expected_match, category)
TEST_CASES = [
    # Brand typos
    ("Remibrutanib", "Remibrutinib", "brands"),
    ("remibritunib", "Remibrutinib", "brands"),
    ("remiburtinib", "Remibrutinib", "brands"),
    ("remi", "Remibrutinib", "brands"),
    ("fabhlta", "Fabhalta", "brands"),
    ("fabahalta", "Fabhalta", "brands"),
    ("fab", "Fabhalta", "brands"),
    ("kisquali", "Kisqali", "brands"),
    ("kisqalli", "Kisqali", "brands"),
    ("kiqali", "Kisqali", "brands"),
    
    # Region typos
    ("northest", "northeast", "regions"),
    ("norhteast", "northeast", "regions"),
    ("midwset", "midwest", "regions"),
    ("midwets", "midwest", "regions"),
    ("MW", "midwest", "regions"),
    ("NE", "northeast", "regions"),
    ("souhteast", "south", "regions"),
    
    # Agent typos
    ("orchstrator", "orchestrator", "agents"),
    ("orchestrtor", "orchestrator", "agents"),
    ("causal_impcat", "causal_impact", "agents"),
    ("gap_anlyzer", "gap_analyzer", "agents"),
    ("drift_moniter", "drift_monitor", "agents"),
    ("experiment_desinger", "experiment_designer", "agents"),
    ("health_scroe", "health_score", "agents"),
    ("explainr", "explainer", "agents"),
    ("feedback_leaner", "feedback_learner", "agents"),
    
    # KPI typos
    ("conversoin", "conversion_rate", "kpis"),
    ("engagment", "engagement_score", "kpis"),
    ("treatmnet_effect", "treatment_effect", "kpis"),
    ("time_to_therpy", "time_to_therapy", "kpis"),
    ("hcp_coverge", "HCP_coverage", "kpis"),
    
    # Journey stage typos
    ("diagnossi", "diagnosis", "journey_stages"),
    ("maintenence", "maintenance", "journey_stages"),
    ("treatement_switch", "treatment_switch", "journey_stages"),
    
    # Workstream typos
    ("ws1", "WS1", "workstreams"),
    ("workstream1", "WS1", "workstreams"),
    
    # Abbreviation expansions
    ("trx", "TRx", "kpis"),
    ("nrx", "NRx", "kpis"),
    ("roi", "ROI", "kpis"),
    ("hcp", "HCP_coverage", "kpis"),  # Should associate with HCP terms
]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def preprocess_corpus(input_path: str, output_path: str) -> None:
    """
    Preprocess corpus for fastText training.
    - Remove comments and section headers
    - Lowercase everything (fastText default behavior)
    - Clean up whitespace
    """
    print(f"Preprocessing corpus: {input_path} -> {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        # Skip comments and empty lines
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('='):
            continue
        
        # Keep the line as-is (fastText handles case internally)
        processed_lines.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    print(f"Processed {len(processed_lines)} training lines")


def train_model(corpus_path: str, model_path: str) -> fasttext.FastText._FastText:
    """
    Train fastText model on E2I corpus.
    """
    print("\n" + "="*60)
    print("TRAINING E2I FASTTEXT MODEL")
    print("="*60)
    
    # Preprocess corpus
    processed_path = corpus_path.replace('.txt', '_processed.txt')
    preprocess_corpus(corpus_path, processed_path)
    
    print(f"\nTraining configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\nTraining model...")
    model = fasttext.train_unsupervised(
        processed_path,
        **TRAINING_CONFIG
    )
    
    print(f"Saving model to: {model_path}")
    model.save_model(model_path)
    
    # Cleanup
    os.remove(processed_path)
    
    print(f"\nModel trained successfully!")
    print(f"  Vocabulary size: {len(model.words)}")
    print(f"  Embedding dimension: {model.get_dimension()}")
    
    return model


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def find_best_match(
    model: fasttext.FastText._FastText,
    query: str,
    candidates: List[str],
    threshold: float = 0.5
) -> Tuple[Optional[str], float]:
    """
    Find the best matching canonical term for a query.
    
    Returns: (best_match, similarity_score)
    """
    query_vec = model.get_word_vector(query.lower())
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        candidate_vec = model.get_word_vector(candidate.lower())
        score = cosine_similarity(query_vec, candidate_vec)
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score < threshold:
        return None, best_score
    
    return best_match, best_score


def run_test_suite(model: fasttext.FastText._FastText) -> Dict:
    """
    Run comprehensive test suite against the trained model.
    """
    print("\n" + "="*60)
    print("RUNNING E2I FASTTEXT TEST SUITE")
    print("="*60)
    
    results = {
        "passed": 0,
        "failed": 0,
        "total": len(TEST_CASES),
        "by_category": {},
        "failures": []
    }
    
    # Build flat candidate list for each category
    all_candidates = []
    for category, terms in CANONICAL_VOCABULARY.items():
        all_candidates.extend(terms)
    
    for typo, expected, category in TEST_CASES:
        # Get candidates for this category
        candidates = CANONICAL_VOCABULARY.get(category, all_candidates)
        
        # Find best match
        match, score = find_best_match(model, typo, candidates, threshold=0.3)
        
        # Check if match is correct (case-insensitive comparison)
        passed = match and match.lower() == expected.lower()
        
        # Track results
        if category not in results["by_category"]:
            results["by_category"][category] = {"passed": 0, "failed": 0}
        
        if passed:
            results["passed"] += 1
            results["by_category"][category]["passed"] += 1
            status = "✓ PASS"
        else:
            results["failed"] += 1
            results["by_category"][category]["failed"] += 1
            status = "✗ FAIL"
            results["failures"].append({
                "typo": typo,
                "expected": expected,
                "got": match,
                "score": score,
                "category": category
            })
        
        print(f"  {status} | '{typo}' -> '{match}' (expected: '{expected}') [score: {score:.3f}]")
    
    # Print summary
    print("\n" + "-"*60)
    print("TEST SUMMARY")
    print("-"*60)
    
    accuracy = (results["passed"] / results["total"]) * 100
    print(f"\nOverall: {results['passed']}/{results['total']} passed ({accuracy:.1f}%)")
    
    print("\nBy Category:")
    for category, stats in results["by_category"].items():
        cat_total = stats["passed"] + stats["failed"]
        cat_acc = (stats["passed"] / cat_total) * 100 if cat_total > 0 else 0
        print(f"  {category}: {stats['passed']}/{cat_total} ({cat_acc:.1f}%)")
    
    if results["failures"]:
        print("\nFailures to investigate:")
        for f in results["failures"][:5]:  # Show first 5 failures
            print(f"  '{f['typo']}' -> got '{f['got']}' instead of '{f['expected']}'")
    
    return results


def interactive_test(model: fasttext.FastText._FastText) -> None:
    """
    Interactive testing mode for manual exploration.
    """
    print("\n" + "="*60)
    print("INTERACTIVE E2I FASTTEXT TESTING")
    print("="*60)
    print("\nEnter a term to find its nearest canonical matches.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Build all candidates
    all_candidates = []
    for terms in CANONICAL_VOCABULARY.values():
        all_candidates.extend(terms)
    
    while True:
        try:
            query = input("Query> ").strip()
        except EOFError:
            break
        
        if not query or query.lower() in ('quit', 'exit', 'q'):
            break
        
        print(f"\nSearching for matches to '{query}'...")
        
        # Get word vector
        query_vec = model.get_word_vector(query.lower())
        
        # Calculate similarity to all candidates
        similarities = []
        for candidate in all_candidates:
            candidate_vec = model.get_word_vector(candidate.lower())
            score = cosine_similarity(query_vec, candidate_vec)
            similarities.append((candidate, score))
        
        # Sort by score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 10
        print("\nTop 10 matches:")
        for i, (term, score) in enumerate(similarities[:10], 1):
            print(f"  {i}. {term}: {score:.4f}")
        
        # Also show fastText's built-in nearest neighbors
        print("\nFastText nearest neighbors (by vocabulary):")
        neighbors = model.get_nearest_neighbors(query.lower(), k=10)
        for score, word in neighbors:
            print(f"  {word}: {score:.4f}")
        
        print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "train":
        if not os.path.exists(CORPUS_PATH):
            print(f"ERROR: Corpus file not found: {CORPUS_PATH}")
            sys.exit(1)
        train_model(CORPUS_PATH, MODEL_PATH)
    
    elif command == "test":
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model not found: {MODEL_PATH}")
            print("Run 'python e2i_fasttext_trainer.py train' first.")
            sys.exit(1)
        
        print(f"Loading model from: {MODEL_PATH}")
        model = fasttext.load_model(MODEL_PATH)
        run_test_suite(model)
    
    elif command == "interactive":
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model not found: {MODEL_PATH}")
            print("Run 'python e2i_fasttext_trainer.py train' first.")
            sys.exit(1)
        
        print(f"Loading model from: {MODEL_PATH}")
        model = fasttext.load_model(MODEL_PATH)
        interactive_test(model)
    
    elif command == "all":
        if not os.path.exists(CORPUS_PATH):
            print(f"ERROR: Corpus file not found: {CORPUS_PATH}")
            sys.exit(1)
        
        model = train_model(CORPUS_PATH, MODEL_PATH)
        run_test_suite(model)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
