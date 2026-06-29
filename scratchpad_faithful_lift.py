import os, json, asyncio
from dotenv import load_dotenv
load_dotenv()
import redis
r = redis.from_url(os.environ["REDIS_URL"], password=os.environ.get("REDIS_PASSWORD"), decode_responses=True)
d = json.loads(r.get("segments:analysis:seg_34ffa2905dc6"))

from src.agents.heterogeneous_optimizer.nodes.segment_analyzer import SegmentAnalyzerNode
from src.agents.heterogeneous_optimizer.nodes.policy_learner import PolicyLearnerNode
from src.agents.heterogeneous_optimizer.agent import calculate_confidence

state = {
    "cate_by_segment": d["cate_by_segment"],
    "overall_ate": d["overall_ate"],
    "heterogeneity_score": d["heterogeneity_score"],
    "top_segments_count": 10,
    "errors": [], "warnings": [], "status": "optimizing",
    "estimation_latency_ms": 0, "analysis_latency_ms": 0, "total_latency_ms": 0,
}

async def main():
    s1 = await SegmentAnalyzerNode().execute(state)  # produces high/low responder tiers
    s2 = await PolicyLearnerNode().execute(s1)        # NEW significance-gated relative policy
    print("OLD persisted: expected_total_lift =", d["expected_total_lift"], "| confidence =", d["confidence"])
    print("NEW pipeline:  expected_total_lift =", round(s2["expected_total_lift"], 2))
    print("NEW confidence (route SSOT) =", round(calculate_confidence(s2), 4))
    print("high responders:", len(s1.get("high_responders") or []), "| low:", len(s1.get("low_responders") or []))
    recs = s2["policy_recommendations"]
    inc = [x for x in recs if x["recommended_treatment_rate"] > x["current_treatment_rate"]]
    dec = [x for x in recs if x["recommended_treatment_rate"] < x["current_treatment_rate"]]
    print(f"recs={len(recs)} increase={len(inc)} decrease={len(dec)}")
    for x in sorted(recs, key=lambda z:-z["expected_incremental_outcome"])[:6]:
        print("  ", x["segment"], "rate", x["current_treatment_rate"], "->", x["recommended_treatment_rate"],
              "lift", round(x["expected_incremental_outcome"],2))

asyncio.run(main())
