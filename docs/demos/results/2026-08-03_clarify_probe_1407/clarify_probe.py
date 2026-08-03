"""#1407 clarify-gate probe — COLD condition on /chat/stream.

Runs 4.3, 5.5, 5.7, A.5 each as a fresh single-turn session (no prior referent),
which is the worst case for over-abstention. Narrative-order suppression was
already proven deterministically by _has_analytical_referent, so this probe
isolates the remaining unknown: what the live DSPy intent classifier does.

Ground truth for "did it clarify" is the persisted pending_clarification state,
not string-matching the prose.
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
TARGETS = ["4.3", "5.5", "5.7", "A.5"]
OUT = Path(
    "/tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/"
    "d1fee615-2671-49d0-8499-c176468a70bf/scratchpad/clarify_probe_results.json"
)

qs = json.loads(
    Path(
        "/home/enunez/Projects/e2i_causal_analytics/scripts/demos/copilot_demo_questions.json"
    ).read_text()
)["questions"]
by_id = {q["question_id"]: q for q in qs}

token = mint_token()
user_id = jwt_sub(token)
print(f"api_base={API_BASE}  user={user_id}\n")

results = []
for qid in TARGETS:
    q = by_id[qid]
    session_id = str(uuid.uuid4())  # fresh -> guaranteed cold turn
    print(f"--- {qid} (session {session_id[:8]}) : {q['text'][:70]}")
    t0 = time.time()
    rec = stream_one(
        api_base=API_BASE,
        token=token,
        user_id=user_id,
        question=q,
        session_id=session_id,
        label="clarifyprobe",
        timeout=180,
    )
    rec["_session_id"] = session_id
    rec["_gold_pattern"] = q.get("gold_pattern")
    rec["_clarify_watch"] = q.get("clarify_watch")
    results.append(rec)
    ans = (rec.get("answer") or rec.get("response") or "")
    print(f"    total={time.time()-t0:.1f}s  err={rec.get('error')}")
    print(f"    intent_actual={rec.get('intent_actual')} routing={rec.get('routing_pattern')} "
          f"agents={rec.get('agents_dispatched')}")
    print(f"    answer[:220]: {ans[:220]!r}\n")
    time.sleep(3)

OUT.write_text(json.dumps(results, indent=2, default=str) + "\n")
print(f"wrote {OUT}")
print("SESSION IDS:", json.dumps({r['question_id']: r['_session_id'] for r in results}))
