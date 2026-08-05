"""Contract test: the fixture RAGAS job must not present itself as a quality gate.

``ragas-evaluation.yml`` runs ``scripts/run_ragas_eval.py``, which calls
``run_evaluation()`` with no ``rag_pipeline`` — so ``_generate_answers`` is
skipped and the frozen gpt-4o judge scores the golden set's HARDCODED answers
over ``retrieved_contexts`` byte-identical to the reference ``contexts``.
Context precision/recall are 1.0-by-construction and faithfulness/
answer_relevancy score the fixture author's prose.

That is still a useful signal — it detects drift in the judge stack on frozen
input — but it was labelled "RAG Quality Evaluation", which is what let a
0.804 answer-relevancy reading stand while the real pipeline sat at 0.401
(#1485). These tests pin the honest labelling and keep the real gate
discoverable from here.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ragas-evaluation.yml"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_ragas_eval.py"
REAL_GATE_SCRIPT = REPO_ROOT / "scripts" / "run_real_pipeline_ragas.py"


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as fh:
        return yaml.safe_load(fh)


def _triggers(workflow: dict) -> dict:
    # PyYAML parses the bare mapping key ``on:`` as the boolean ``True``.
    return workflow.get("on", workflow.get(True, {})) or {}


def test_workflow_name_identifies_it_as_a_fixture_check():
    name = _load_workflow()["name"]
    assert "Fixture" in name, f"workflow name {name!r} does not say it scores a fixture"


def test_no_job_claims_to_evaluate_production_rag_quality():
    """ "RAG Quality Evaluation" is the label that made 0.804 look like production."""
    jobs = _load_workflow()["jobs"]
    for job_id, job in jobs.items():
        display = job.get("name", job_id)
        assert display != "RAG Quality Evaluation", (
            f"job {job_id} still claims to evaluate RAG quality; it scores a static fixture"
        )
        assert "Fixture" in display, (
            f"job {job_id} display name {display!r} must say it is a fixture check"
        )


def test_workflow_stays_manual_only():
    """#504: this eval is OpenAI-throughput-bound and must never run per-PR."""
    triggers = _triggers(_load_workflow())
    assert "workflow_dispatch" in triggers
    assert "push" not in triggers, "a paid gpt-4o eval must not run on push (#504)"
    assert "pull_request" not in triggers, "a paid gpt-4o eval must not run per-PR (#504)"


def test_workflow_documents_the_frozen_input_limitation():
    text = WORKFLOW_PATH.read_text()
    assert "1485" in text, "the workflow must reference #1485"
    for phrase in ("fixture", "judge"):
        assert phrase in text.lower(), f"workflow does not document {phrase!r}"
    assert "run_real_pipeline_ragas.py" in text, (
        "the workflow must point readers at the real-pipeline gate"
    )


def test_fixture_script_docstring_disclaims_production_quality():
    docstring = EVAL_SCRIPT.read_text().split('"""')[1]
    lowered = docstring.lower()
    assert "fixture" in lowered, "run_ragas_eval.py must say it scores a fixture"
    assert "run_real_pipeline_ragas.py" in docstring, (
        "run_ragas_eval.py must point at the real-pipeline gate"
    )
    assert "1485" in docstring


def test_the_real_pipeline_gate_exists():
    assert REAL_GATE_SCRIPT.exists(), "the real-pipeline gate this workflow defers to is missing"


def test_dormant_regression_job_is_preserved():
    """Not vestigial: the header documents restoring push/PR triggers.

    The job is gated ``if: github.event_name == 'pull_request'`` and the
    workflow is dispatch-only, so it never runs today. It comes back to life
    if the triggers are restored, which the header explicitly contemplates —
    deleting it would remove working code on a pattern-match.
    """
    jobs = _load_workflow()["jobs"]
    assert "ragas-regression" in jobs
    assert jobs["ragas-regression"]["if"] == "github.event_name == 'pull_request'"


# ---------------------------------------------------------------------------
# Rename blast radius (codex iter-1 HIGH)
# ---------------------------------------------------------------------------
#
# Renaming the workflow is only half the fix: docs that still describe it as a
# "RAG quality eval" keep the wrong mental model alive, which is the actual
# defect in #1485 (a 0.804 fixture number read as production quality).

# Every project doc, not a hardcoded list: a NEW docs/*.md describing the
# fixture eval as a quality gate would escape a fixed set entirely, and the
# wrong mental model is the actual defect in #1485.
#
# EXACTLY ONE exclusion, matched by prefix rather than by directory name:
# docs/reports/ holds dated records of what was believed and measured at the
# time (e.g. dspy_lane_ab_20260718.md), and rewriting a historical report to
# match today's naming would falsify the record.
#
# The earlier part-name set {"reports", "archive", "node_modules"} silently
# over-excluded. Measured on this tree: `docs/archive/ragas.md` and
# `docs/plans/reports/x.md` were both dropped from the scan, while the repo's
# real directory is docs/Archive (capital A) — so the lowercase entry matched
# nothing here yet WOULD have hidden docs/Archive on a case-insensitive
# checkout. A guard whose coverage depends on the developer's filesystem is
# not a guard.
#
# docs/Archive is deliberately NOT excluded. It holds 1 .md with zero RAGAS
# mentions, so excluding it buys nothing and costs coverage; exclusions should
# be earned by a demonstrated conflict, not added pre-emptively. If an archived
# doc ever legitimately needs the same historical-record rationale, add it then
# — spelled `docs/Archive`, exactly. node_modules is likewise dropped: the glob
# only matches *.md and there is no node_modules under docs/.
DOC_SCAN_EXCLUDED_PREFIX = ("docs", "reports")


def _is_scanned_doc(path: Path) -> bool:
    """Whether a doc is in scope for the drift scan.

    Extracted so the exclusion rule can be exercised over CONSTRUCTED paths —
    the over-exclusion above was invisible to a test that only asserted which
    real files were included.
    """
    return not path.is_relative_to(REPO_ROOT.joinpath(*DOC_SCAN_EXCLUDED_PREFIX))


def _project_docs() -> list:
    paths = [REPO_ROOT / "README.md", *(REPO_ROOT / "docs").rglob("*.md")]
    return [p for p in paths if p.exists() and _is_scanned_doc(p)]


def test_doc_scan_actually_covers_the_known_ragas_docs():
    """Guard the guard: a glob that silently matches nothing proves nothing."""
    covered = {p.name for p in _project_docs()}
    for expected in ("README.md", "ONBOARDING.md", "LLM_CONFIGURATION.md"):
        assert expected in covered, f"doc scan missed {expected}"


def test_only_docs_reports_is_excluded_from_the_scan():
    """Over-exclusion is the failure a coverage-only test cannot see."""
    assert _is_scanned_doc(REPO_ROOT / "docs" / "archive" / "ragas.md"), (
        "a lowercase docs/archive path must be scanned — excluding it by bare "
        "directory name is a case-sensitivity trap"
    )
    assert _is_scanned_doc(REPO_ROOT / "docs" / "Archive" / "ragas.md")
    assert _is_scanned_doc(REPO_ROOT / "docs" / "plans" / "reports" / "x.md"), (
        "only docs/reports/ is a historical record; a nested reports/ dir is not"
    )
    assert _is_scanned_doc(REPO_ROOT / "README.md")
    assert not _is_scanned_doc(REPO_ROOT / "docs" / "reports" / "dspy_lane_ab_20260718.md")


def test_nothing_outside_docs_reports_is_excluded_in_practice():
    """Applies the rule to the REAL tree, so a future glob typo shows up here."""
    everything = [REPO_ROOT / "README.md", *(REPO_ROOT / "docs").rglob("*.md")]
    scanned = set(_project_docs())
    over_excluded = [
        str(p.relative_to(REPO_ROOT))
        for p in everything
        if p.exists() and p not in scanned and _is_scanned_doc(p)
    ]
    assert not over_excluded, f"docs dropped from the scan for no stated reason: {over_excluded}"


def test_no_doc_still_calls_the_fixture_job_a_rag_quality_eval():
    stale = []
    for path in _project_docs():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            lowered = line.lower()
            if "ragas" not in lowered and "run_ragas_eval" not in lowered:
                continue
            if "rag quality eval" in lowered or "quality gate" in lowered:
                stale.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not stale, "docs still present the fixture eval as a quality gate:\n" + "\n".join(stale)


def test_docs_point_at_the_real_pipeline_gate():
    """A reader who finds the fixture eval must be able to find the real one."""
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "ONBOARDING.md"):
        text = path.read_text()
        assert "run_real_pipeline_ragas.py" in text, (
            f"{path.name} describes the RAGAS stack without pointing at the real-pipeline gate"
        )


def test_smoke_workflow_header_uses_the_current_name():
    """ragas-smoke.yml's comments name its counterpart; behaviour untouched."""
    text = (REPO_ROOT / ".github" / "workflows" / "ragas-smoke.yml").read_text()
    assert "RAGAS Evaluation" not in text, "stale workflow name in ragas-smoke.yml comments"
