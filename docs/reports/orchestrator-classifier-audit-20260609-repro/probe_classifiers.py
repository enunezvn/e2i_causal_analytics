"""Cheapest-disproof probe for the Orchestrator routing engines.

Runs, on REAL code (no mocks), two classifiers over the same pharma query
battery and prints their routing decisions side by side:

  A) The 4-stage ``ClassificationPipeline`` (classifier/ package) — the module
     the audit task calls "the decision engine that routes every query".
     Run OFFLINE/deterministic (enable_llm_layer=False, no Anthropic client),
     which is exactly the rule-based path.

  B) The LIVE engine actually wired into create_orchestrator_graph():
     IntentClassifierNode._pattern_classify (deterministic regex layer) +
     RouterNode (pure-logic intent->agent dispatch).

It also directly probes the DomainMapper to test the hypothesis that
MONITORING (base=0.4) and EXPLANATION (base=0.3) clear the 0.3 inclusion
threshold for EVERY query -> every query is forced multi-domain.

Run:
    .venv/bin/python docs/reports/orchestrator-classifier-audit-20260609-repro/probe_classifiers.py
"""

import asyncio
import sys

# --- 4-stage classifier (the audited package) ---
from src.agents.orchestrator.classifier.domain_mapper import DomainMapper
from src.agents.orchestrator.classifier.feature_extractor import FeatureExtractor
from src.agents.orchestrator.classifier.pipeline import ClassificationPipeline

# --- live engine (what create_orchestrator_graph actually wires) ---
from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
from src.agents.orchestrator.nodes.router import RouterNode

BATTERY = [
    ("causal", "What was the impact of the Q3 Kisqali campaign on prescriptions?"),
    ("gap", "Where are we underperforming in the Northeast region?"),
    ("segment", "Which HCP segments responded best to the messaging?"),
    ("experiment", "Design an A/B test for the new rep cadence."),
    ("prediction", "Predict next quarter's TRx for Fabhalta."),
    ("monitoring", "Is there any data drift in the model inputs?"),
    ("explanation", "Explain why the ATE dropped last month."),
    ("cohort", "Build a cohort of CSU patients eligible for Remibrutinib."),
    (
        "multi_dependent",
        "What drove the Kisqali uplift, and which segments responded best, "
        "then design a test to confirm it?",
    ),
    ("ambiguous", "Show me the numbers."),
    ("exploration", "How many oncologists are in the West territory?"),
    ("comparison", "Compare Kisqali vs Fabhalta adoption across regions."),
    ("greeting", "hello"),
]


def run_4stage(query: str):
    """Deterministic rule-based path of the 4-stage pipeline."""
    pipe = ClassificationPipeline(llm_client=None, enable_llm_layer=False)
    result = asyncio.run(pipe.classify(query))
    fe = FeatureExtractor()
    dm = DomainMapper()
    mapping = dm.map_domains(fe.extract(query))
    detected = [(m.domain.value, m.confidence) for m in mapping.domains_detected]
    return result, detected


def run_live(query: str):
    """Deterministic regex layer + pure-logic router (the wired engine)."""
    clf = IntentClassifierNode.__new__(IntentClassifierNode)  # skip LLM ctor
    intent = clf._pattern_classify(query.lower())
    router = RouterNode()
    state = {"query": query, "intent": dict(intent)}
    routed = asyncio.run(router.execute(state))
    agents = [d["agent_name"] for d in routed.get("dispatch_plan", [])]
    return intent, agents


def main():
    print("=" * 100)
    print("PROBE 1 — 4-stage ClassificationPipeline (classifier/ package) — the AUDITED module")
    print("=" * 100)
    all_multi = True
    monitoring_always = True
    explanation_always = True
    crashes = []
    clarification_count = 0
    for label, q in BATTERY:
        print(f"\n[{label}] {q!r}")
        try:
            result, detected = run_4stage(q)
        except Exception as e:  # faithfully record crashes
            crashes.append((label, q, f"{type(e).__name__}: {str(e)[:80]}"))
            print(f"   *** CRASH *** {type(e).__name__}: {str(e).splitlines()[0][:90]}")
            continue
        domains = {d for d, _ in detected}
        if len(detected) < 2:
            all_multi = False
        if "MONITORING" not in domains:
            monitoring_always = False
        if "EXPLANATION" not in domains:
            explanation_always = False
        if result.routing_pattern.value == "CLARIFICATION_NEEDED":
            clarification_count += 1
        print(f"   routing_pattern : {result.routing_pattern.value}")
        print(f"   target_agents   : {result.target_agents}")
        print(f"   primary/conf    : {detected[0] if detected else None}")
        print(f"   domains_detected: {detected}")
        print(f"   used_llm_layer  : {result.used_llm_layer}")
        print(f"   reasoning       : {result.reasoning}")

    print("\n" + "-" * 100)
    n = len(BATTERY)
    print(f"HYPOTHESIS  every (non-crashing) query is multi-domain (>=2)  : {all_multi}")
    print(f"HYPOTHESIS  MONITORING present in EVERY query                : {monitoring_always}")
    print(f"HYPOTHESIS  EXPLANATION present in EVERY query               : {explanation_always}")
    print(f"OUTCOME     CLARIFICATION_NEEDED count                       : {clarification_count}/{n}")
    print(f"OUTCOME     CRASH count                                      : {len(crashes)}/{n}")
    for label, q, err in crashes:
        print(f"            CRASH [{label}] {err}")
    print("-" * 100)

    print("\n\n" + "=" * 100)
    print("PROBE 2 — LIVE engine wired into create_orchestrator_graph()")
    print("          IntentClassifierNode._pattern_classify + RouterNode")
    print("=" * 100)
    for label, q in BATTERY:
        intent, agents = run_live(q)
        print(f"\n[{label}] {q!r}")
        print(f"   primary_intent  : {intent['primary_intent']}  (conf={intent['confidence']})")
        print(f"   secondary       : {intent['secondary_intents']}")
        print(f"   multi_agent     : {intent['requires_multi_agent']}")
        print(f"   -> dispatched   : {agents}")

    print("\n\n" + "=" * 100)
    print("PROBE 3 — DomainMapper on the EMPTY query and a pure greeting (base-score floor test)")
    print("=" * 100)
    fe = FeatureExtractor()
    dm = DomainMapper()
    for q in ["", "hello", "asdf qwerty zzz"]:
        mapping = dm.map_domains(fe.extract(q))
        detected = [(m.domain.value, m.confidence) for m in mapping.domains_detected]
        print(f"\n   query={q!r}")
        print(f"   domains_detected: {detected}")
        print(f"   is_multi_domain : {mapping.is_multi_domain}")
        print(f"   primary_domain  : {mapping.primary_domain.value if mapping.primary_domain else None}")


if __name__ == "__main__":
    sys.exit(main())
