"""Lane B cheap disproofs, captured (run from the worktree root:
``python docs/demos/results/2026-09-22_lane_b_structural_author_scaffold/disproofs.py``
writes ``disproofs.txt`` next to itself). Each probe answers one assumption
named in README.md; the numbers there cite this file's output by line."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(
    {
        "SUPABASE_URL": "http://127.0.0.1:1",
        "SUPABASE_KEY": "test-key",
        "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        "SUPABASE_SERVICE_KEY": "test-key",
        "SUPABASE_ANON_KEY": "test-key",
    }
)
out: list[str] = []
import src  # noqa: E402

assert ".worktrees/lane-b-structural-author-scaffold" in src.__file__, src.__file__
out.append(f"commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()}")

# 1. DummyLM drives typed list / bool outputs.
import dspy  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402


class _Sig(dspy.Signature):
    """Draw a DAG."""

    feature_name: str = dspy.InputField()
    edges: list[list[str]] = dspy.OutputField()
    ambiguous: bool = dspy.OutputField()


with dspy.context(lm=DummyLM([{"edges": '[["f","T"],["f","Y"],["T","Y"]]', "ambiguous": "false"}])):
    pred = dspy.Predict(_Sig)(feature_name="f")
out.append(
    f"probe1 dummylm typed outputs: dspy={dspy.__version__} edges={pred.edges!r} "
    f"type={type(pred.edges).__name__} ambiguous={pred.ambiguous!r} type={type(pred.ambiguous).__name__}"
)

# 2. Importing the graph builder costs the whole agent package.
t0 = time.time()
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode  # noqa: E402

out.append(f"probe2 graph_builder import seconds: {time.time() - t0:.1f}")
t0 = time.time()
import src.ml.causal_role_dgp.backdoor as _bd  # noqa: E402,F401

out.append(f"probe2 backdoor module import seconds (after the above): {time.time() - t0:.3f}")

# 3. CitationResolver takes injectable clients.
from src.data.kg.citation_resolver import CitationResolver  # noqa: E402

out.append(f"probe3 CitationResolver.__init__ signature: {inspect.signature(CitationResolver.__init__)}")

# 4. Repo under the dead-Supabase pin.
import asyncio  # noqa: E402

from src.repositories.expert_review import ExpertReviewRepository  # noqa: E402


async def _probe4():
    repo = ExpertReviewRepository()
    t = time.time()
    res = await repo.create_review(reviewer_id="probe", review_type="initial_dag")
    return repo.client, res, time.time() - t


client, res, dt = asyncio.run(_probe4())
out.append(f"probe4 dead pin: repo.client={client!r} create_review->{res!r} in {dt:.1f}s")

# 5. docs/ is excluded from the image.
for lineno, line in enumerate(Path(".dockerignore").read_text().splitlines(), start=1):
    if line.strip() == "docs/":
        out.append(f"probe5 .dockerignore:{lineno} == {line.strip()!r}")

# 6. dspy cleandoc drops only the trailing newline of the guide.
from src.data.kg.structural_author import (  # noqa: E402
    GUIDE_SECTIONS_0_TO_6,
    StructuralAttestationSignature,
)

instr = StructuralAttestationSignature.instructions
out.append(
    f"probe6 instructions chars={len(instr)} guide chars={len(GUIDE_SECTIONS_0_TO_6)} "
    f"equal_stripped={instr == GUIDE_SECTIONS_0_TO_6.strip()}"
)

# 7. Node line count vs the ratchet pin.
n = len(Path("src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py").read_text().splitlines())
out.append(f"probe7 adaptive_validity_check.py lines: {n}")

# 8. Run (b) fake diff: the disagreements are exactly the manifest instruments.
import json  # noqa: E402

from src.data.manifests.optum_feature_manifest import _OPTUM_INSTRUMENT_FEATURES  # noqa: E402

diff = json.loads(
    (HERE / "author_fake/optum_biologic_initiation_initiated_biologic_180d/manifest_diff.json").read_text()
)
out.append(
    f"probe8 fake run(b) disagreements={len(diff['disagreements'])} manifest instruments="
    f"{len(_OPTUM_INSTRUMENT_FEATURES)} equal={sorted(diff['disagreements']) == sorted(_OPTUM_INSTRUMENT_FEATURES)}"
)

# 9. Cost ratios of the real runs vs the 91-brief estimate.
cost = json.loads((HERE / "measure_real_refused/cost_estimate.json").read_text())
per_brief = cost["prompt_chars_per_brief_mean"]
out.append(
    f"probe9 cost: 91 briefs usd={cost['usd_estimate']:.2f}; run(a) 64 briefs ratio={64/91:.2f}; "
    f"run(b) 110 briefs ratio={110/91:.2f}; prompt chars/brief={per_brief:.0f}"
)

(HERE / "disproofs.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
