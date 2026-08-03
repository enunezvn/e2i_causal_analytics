"""#1407 follow-on probes.

B1 — RESUME: answer A.5's pending ask-back in its own session. A clarification
     that cannot resume is worse than no clarification, so this is the half of
     the feature the first probe did not exercise.
B2 — NARRATIVE SUPPRESSION (live): seed a session with 5.3 (names "Kisqali"),
     then ask 5.7. Predicted: prior referent suppresses the gate and 5.7 runs a
     real analysis instead of clarifying. Measured, not reasoned — the cold-turn
     intent prediction for 4.3 was already proven wrong.
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path("/home/enunez/Projects/e2i_causal_analytics/scripts/demos")))

from copilot_chat_perf_runner import mint_token, jwt_sub, stream_one  # noqa: E402

API_BASE = os.environ.get("E2I_API_BASE", "https://eznomics.site/api")
A5_SESSION = "f0ff4927-1038-449d-a3f1-ee260dd2cfa5"  # pending from probe 1
OUT = Path(
    "/tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/"
    "d1fee615-2671-49d0-8499-c176468a70bf/scratchpad/clarify_probe2_results.json"
)

qs = json.loads(
    Path(
        "/home/enunez/Projects/e2i_causal_analytics/scripts/demos/copilot_demo_questions.json"
    ).read_text()
)["questions"]
by_id = {q["question_id"]: q for q in qs}

token = mint_token()
user_id = jwt_sub(token)
results = []


def run(label, qid, text, session_id):
    q = dict(by_id.get(qid, {"question_id": qid, "session": "probe", "tier": "-"}))
    q["text"] = text
    q["question_id"] = label
    print(f"--- {label} (sess {session_id[:8]}): {text[:66]}")
    rec = stream_one(
        api_base=API_BASE, token=token, user_id=user_id, question=q,
        session_id=session_id, label="clarifyprobe2", timeout=240,
    )
    rec["_session_id"] = session_id
    di = rec.get("dispatch_info") or {}
    print(f"    total={rec.get('total_ms',0)/1000:.1f}s  orch={di.get('orchestrator_used')} "
          f"intent={di.get('intent')} agents={di.get('agents_dispatched')}")
    print(f"    answer[:260]: {(rec.get('response_text') or '')[:260]!r}\n")
    results.append(rec)
    time.sleep(3)


# B1 — resume A.5 with a slot answer
run("A.5-resume", "A.5", "Kisqali TRx", A5_SESSION)

# B2 — narrative suppression: seed with 5.3 (names Kisqali), then 5.7
nsess = str(uuid.uuid4())
run("5.3-seed", "5.3", by_id["5.3"]["text"], nsess)
run("5.7-after-seed", "5.7", by_id["5.7"]["text"], nsess)

OUT.write_text(json.dumps(results, indent=2, default=str) + "\n")
print("wrote", OUT)
print("NARRATIVE SESSION:", nsess)
