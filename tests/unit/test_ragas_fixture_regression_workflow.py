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


# ---------------------------------------------------------------------------
# Rename blast radius, part 2: STEPS and RUN blocks (codex iter-6 HIGH)
# ---------------------------------------------------------------------------
#
# test_no_job_claims_to_evaluate_production_rag_quality reads only job display
# names, so four strings inside the dormant ragas-regression job survived the
# rename: a step name, an echo, and two ::error::/::notice:: lines all claiming
# "RAG quality". On the restore path the header contemplates (re-adding the
# push/pull_request triggers), PR logs would report RAG-quality pass/fail from
# the FIXTURE evaluator — the 0.804-reads-as-production defect #1485 exists to
# fix, printed straight into CI output.
#
# Parsed YAML, not a raw-text regex: the header comments legitimately say
# "DOES NOT MEASURE PRODUCTION RAG QUALITY", and parsing drops comments for
# free. No framing exemption — these strings land in CI logs where a comment
# three screens up is not visible, so "RAG quality regression detected!" is
# wrong regardless of what the file says elsewhere.

_QUALITY_CLAIMS = ("rag quality", "quality regression")


def _job_step_strings() -> list:
    """(job_id, where, text) for every step name and run block in the workflow."""
    out = []
    for job_id, job in (_load_workflow().get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            if step.get("name"):
                out.append((job_id, f"step[{index}].name", step["name"]))
            if step.get("run"):
                out.append((job_id, f"step[{index}].run", step["run"]))
    return out


def test_no_step_or_run_block_claims_rag_quality():
    offenders = []
    for job_id, where, text in _job_step_strings():
        for lineno, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            if any(claim in lowered for claim in _QUALITY_CLAIMS):
                offenders.append(f"{job_id}:{where}:{lineno}: {line.strip()}")
    assert not offenders, (
        "workflow steps still claim RAG quality; this job scores a static fixture "
        "and these strings reach CI logs:\n" + "\n".join(offenders)
    )


def test_step_scan_actually_reaches_the_dormant_job():
    """Guard the guard: the strings at issue live in ragas-regression's run block."""
    scanned = {job_id for job_id, _, _ in _job_step_strings()}
    assert "ragas-regression" in scanned, "step scan never reached the dormant job"
    assert "ragas-evaluation" in scanned


# ---------------------------------------------------------------------------
# Dormant job's threshold loop must fail closed (codex iter-6, LATENT)
# ---------------------------------------------------------------------------
#
# `if score is not None and score < threshold` silently SKIPS a missing metric,
# then prints "All ... thresholds passed". The all-zero degenerate check only
# catches `== 0.0`, the OLD fabrication mode. Sibling lane #1488 makes unjudged
# metrics None/missing rather than 0.0, so post-merge a degenerate report would
# dodge both and report success on an unverifiable report.
#
# UNREACHABLE TODAY (verified, not assumed): the workflow is dispatch-only, the
# job needs ragas-evaluation, GitHub applies implicit success() to its `if:`,
# and the eval step runs --fail-on-threshold which post-#1488 fails closed on
# unmeasured metrics — so any report reaching this job has all four metrics
# measured. Fixed anyway because it is the same latent-fail-open-goes-live
# class as check_thresholds, whose dead guards went live when safe_score died.


def _dormant_run_block() -> str:
    job = _load_workflow()["jobs"]["ragas-regression"]
    return "\n".join(step.get("run", "") for step in job.get("steps") or [])


def test_dormant_regression_job_fails_closed_on_an_unmeasured_metric():
    run = _dormant_run_block()
    # Comment lines are stripped before the pattern check: the fix documents
    # the old `score is not None and ...` skip by quoting it, and a guard that
    # cannot tell a comment from live code would forbid explaining the defect
    # it exists to prevent.
    code = "\n".join(line for line in run.splitlines() if not line.strip().startswith("#"))
    assert "is not None and" not in code, (
        "the threshold loop still skips a missing metric instead of blocking on it"
    )
    lowered = run.lower()
    assert "unverifiable" in lowered or "unmeasured" in lowered, (
        "the run block must name an unmeasured metric as unverifiable, mirroring "
        "check_thresholds' fail-closed contract"
    )


# ---------------------------------------------------------------------------
# Script surface: user-visible strings (codex iter-7 HIGH)
# ---------------------------------------------------------------------------
#
# The module docstring says "RAGAS FIXTURE regression — judge-drift sentinel",
# but the argparse description and the run banner still announced a RAG
# PIPELINE evaluation — and the banner is what CI logs print. Same class as the
# iter-6 workflow-string fix, one layer down.
#
# Its own claim vocabulary, deliberately: the workflow scan looks for
# "rag quality"/"quality regression", which would never match "RAG pipeline".
# Force-fitting one vocabulary onto both surfaces is how the first gap
# survived.
#
# AST over call arguments, not a raw-text scan: the module docstring
# legitimately explains that the script "never invokes the RAG pipeline", and
# banning that phrase file-wide would forbid the disclaimer. Docstrings are
# Expr nodes, so walking only Call arguments excludes them structurally.

_SCRIPT_CLAIMS = ("rag pipeline", "rag evaluation pipeline")


def _script_user_visible_strings() -> list:
    """(lineno, text) for every string literal passed as a call argument."""
    import ast

    tree = ast.parse(EVAL_SCRIPT.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append((arg.lineno, arg.value))
    return out


def test_eval_script_user_visible_strings_do_not_claim_pipeline_evaluation():
    offenders = [
        f"run_ragas_eval.py:{lineno}: {text!r}"
        for lineno, text in _script_user_visible_strings()
        if any(claim in text.lower() for claim in _SCRIPT_CLAIMS)
    ]
    assert not offenders, (
        "the fixture script still presents itself as a RAG pipeline evaluation "
        "in user-visible output:\n" + "\n".join(offenders)
    )


def test_eval_script_string_scan_is_not_vacuous():
    """Guard the guard: the scan must actually reach the banner and argparse."""
    texts = [t for _, t in _script_user_visible_strings()]
    assert any("Fixture" in t or "fixture" in t for t in texts), (
        "no fixture-framed user-visible string found — the AST scan found nothing"
    )
    assert len(texts) > 10, f"AST scan returned only {len(texts)} strings; parse likely broke"


# ---------------------------------------------------------------------------
# Workflow header comment (codex iter-7 HIGH)
# ---------------------------------------------------------------------------


def test_workflow_header_does_not_instruct_a_rag_quality_check():
    """A PIN, not a general scan — deliberately.

    Comments are invisible to yaml.safe_load, so the parsed-YAML step scan
    cannot see this line. A general raw-text comment scan would false-fail on
    the header's own legitimate disclaimers ("DOES NOT MEASURE PRODUCTION RAG
    QUALITY"), so this pins the one instruction that told a reader to run the
    fixture job when they want a quality check.
    """
    text = WORKFLOW_PATH.read_text()
    assert "RAG-quality check" not in text, (
        "the header still tells readers to run this job for a RAG-quality check; "
        "it scores a fixture"
    )
    assert "fixture judge-drift check" in text


# ---------------------------------------------------------------------------
# Dormant job's inline python: BEHAVIORAL fail-closed test (codex iter-7 LOW)
# ---------------------------------------------------------------------------
#
# The textual `"is not None and" not in code` guard above is formatting-brittle:
# it misses a wrapped `if (score is not None\n    and score < threshold):`.
# These run the extracted heredoc for real. The textual guard stays — it
# documents intent and costs nothing; this is what closes the formatting hole.


def _dormant_regression_python() -> str:
    """Extract the heredoc python from the dormant job's run block.

    After YAML block-scalar dedent the body sits at column 0 between the
    ``<< 'PYEOF'`` line and the terminating ``PYEOF``.
    """
    lines = _dormant_run_block().splitlines()
    start = next(i for i, line in enumerate(lines) if "<< 'PYEOF'" in line)
    end = next(i for i, line in enumerate(lines) if i > start and line.strip() == "PYEOF")
    return "\n".join(lines[start + 1 : end])


def _run_dormant_python(tmp_path, report: dict):
    import json
    import subprocess
    import sys

    results = tmp_path / "ragas-results"
    results.mkdir(exist_ok=True)
    (results / "evaluation-report.json").write_text(json.dumps(report))
    return subprocess.run(
        [sys.executable, "-c", _dormant_regression_python()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


_HEALTHY_REPORT = {
    "avg_faithfulness": 0.90,
    "avg_answer_relevancy": 0.90,
    "avg_context_precision": 0.90,
    "avg_context_recall": 0.90,
    "total_samples": 30,
    "passed_samples": 30,
}


def test_dormant_python_blocks_on_an_unmeasured_metric(tmp_path):
    """A metric we could not measure makes its threshold unverifiable."""
    report = dict(_HEALTHY_REPORT, avg_faithfulness=None)
    proc = _run_dormant_python(tmp_path, report)
    assert proc.returncode == 1, f"expected block, got {proc.returncode}: {proc.stdout}"
    assert "unmeasured" in proc.stdout and "unverifiable" in proc.stdout


def test_dormant_python_blocks_on_a_missing_metric_key(tmp_path):
    report = {k: v for k, v in _HEALTHY_REPORT.items() if k != "avg_context_recall"}
    proc = _run_dormant_python(tmp_path, report)
    assert proc.returncode == 1
    assert "context_recall" in proc.stdout


def test_dormant_python_passes_a_healthy_report(tmp_path):
    proc = _run_dormant_python(tmp_path, _HEALTHY_REPORT)
    assert proc.returncode == 0, f"healthy report must pass: {proc.stdout} {proc.stderr}"
    assert "thresholds passed" in proc.stdout


def test_dormant_python_warns_on_the_all_zero_degenerate_report(tmp_path):
    """The pre-existing 0.0 fabrication mode still warns rather than failing."""
    report = dict.fromkeys(
        ("avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall"),
        0.0,
    )
    report.update(total_samples=30, passed_samples=0)
    proc = _run_dormant_python(tmp_path, report)
    assert proc.returncode == 0
    assert "::warning::" in proc.stdout


def test_dormant_python_blocks_a_genuine_threshold_regression(tmp_path):
    proc = _run_dormant_python(tmp_path, dict(_HEALTHY_REPORT, avg_faithfulness=0.10))
    assert proc.returncode == 1
    assert "0.100" in proc.stdout
