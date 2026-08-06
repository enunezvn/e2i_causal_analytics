"""The cognitive-RAG runtime loads its optimized synthesis prompt (issue #1486).

Rounds 1-2 of #1486 established that `src/rag/` had NO artifact-consumption seam
at all: repo-wide, `load_optimized_module(` had exactly one real call site
(`pattern_analyzer.py:372`, hardcoded `feedback_learner_pattern`), and
`install_all_prompt_bundles` only iterates `RECIPIENT_FACTORIES`. So a
GEPA-optimized RAG prompt had nowhere to land — optimizing one would have been
a second dead end.

These tests pin the consumer half of the chain. The producer half (the nightly
leg) is pinned in tests/unit/test_tasks/test_dspy_rag_optimization_leg_1486.py,
and the two are tied together by the shared-constant test there: a save name and
a load name that are separate string literals is exactly how the artifact
written on 2026-06-08 sat unshipped for six weeks
(docs/reports/dspy_lane_ab_20260718.md section 7).
"""

import json
from pathlib import Path
from typing import Any

import pytest

# DSPy import safety — keep this module on one xdist worker (repo convention).
pytestmark = pytest.mark.xdist_group(name="gepa_metrics")


def _write_artifact(
    root: Path, agent_name: str, instructions: str, version_id: str | None = None
) -> Path:
    """Write an artifact in the exact shape save_optimized_module produces.

    Built by calling the real saver rather than hand-rolling JSON, so this test
    breaks if the on-disk contract drifts.
    """
    import dspy

    from src.optimization.gepa.versioning import save_optimized_module
    from src.rag.cognitive_rag_dspy import EvidenceSynthesisSignature

    module = dspy.ChainOfThought(EvidenceSynthesisSignature.with_instructions(instructions))
    info = save_optimized_module(
        module=module,
        agent_name=agent_name,
        version_id=version_id,
        output_dir=str(root),
        metadata={"source": "test"},
    )
    return Path(info["path"])


class TestOptimizedSynthesisLoader:
    def test_returns_none_when_no_artifact_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An intentional miss is None, not an exception."""
        from src.rag.cognitive_rag_dspy import load_optimized_synthesis_module

        monkeypatch.chdir(tmp_path)
        assert load_optimized_synthesis_module(reset=True) is None

    def test_loads_the_artifact_the_saver_wrote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real save -> real load round trip under the shared agent name."""
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        _write_artifact(
            tmp_path / "optimized_modules",
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            "Ground every claim in the retrieved evidence.",
        )

        module = load_optimized_synthesis_module(reset=True)
        assert module is not None

    def test_miss_is_cached_but_transient_failure_is_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mirrors pattern_analyzer semantics: cache the miss, retry the error.

        A cached transient error would strand the runtime on the base prompt
        until the process restarted.
        """
        import src.rag.cognitive_rag_dspy as module_under_test
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        # An artifact must EXIST for a load to be attempted at all — with none on
        # disk the loader short-circuits on the signature probe (F3).
        _write_artifact(tmp_path / "optimized_modules", OPTIMIZED_SYNTHESIS_AGENT_NAME, "present")
        calls = {"n": 0}

        def _boom(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            raise RuntimeError("corrupt read")

        monkeypatch.setattr(module_under_test, "_load_optimized_module", _boom, raising=False)
        assert load_optimized_synthesis_module(reset=True) is None
        assert load_optimized_synthesis_module() is None
        assert calls["n"] == 2, "a transient error was cached; a later cycle cannot retry"

    def test_a_first_artifact_is_picked_up_without_a_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex iter-1 F3 — supersedes the earlier cache-the-miss-forever contract.

        A worker that probed before the first nightly success would otherwise
        serve the base prompt until the process restarted. Nothing invalidates
        the cache: docker-compose mounts optimized_modules read-only into api
        (:709) and writable into worker_medium (:843), with no signalling
        between them. Caching keyed on the artifact's existence+mtime keeps the
        parse cached while still noticing a new file.
        """
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        assert load_optimized_synthesis_module(reset=True) is None

        _write_artifact(tmp_path / "optimized_modules", OPTIMIZED_SYNTHESIS_AGENT_NAME, "later")
        assert load_optimized_synthesis_module() is not None, (
            "a worker that cached the miss serves the base prompt until restart"
        )

    def test_a_newer_version_supersedes_a_cached_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Holding version N while N+1 sits on disk is the same staleness bug."""
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        root = tmp_path / "optimized_modules"
        _write_artifact(root, OPTIMIZED_SYNTHESIS_AGENT_NAME, "VERSION-N")
        first = load_optimized_synthesis_module(reset=True)
        assert first is not None

        _write_artifact(root, OPTIMIZED_SYNTHESIS_AGENT_NAME, "VERSION-N-PLUS-1")
        reloaded = load_optimized_synthesis_module()
        instructions = [
            getattr(getattr(p, "signature", None), "instructions", "")
            for p in reloaded.predictors()
        ]
        assert any("VERSION-N-PLUS-1" in (i or "") for i in instructions), instructions

    def test_an_unchanged_artifact_is_not_reparsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache must still hold; only existence+mtime is re-checked."""
        import src.rag.cognitive_rag_dspy as module_under_test
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        _write_artifact(tmp_path / "optimized_modules", OPTIMIZED_SYNTHESIS_AGENT_NAME, "stable")
        load_optimized_synthesis_module(reset=True)

        calls = {"n": 0}
        real = module_under_test._load_optimized_module

        def _counting(*a: Any, **k: Any) -> Any:
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(module_under_test, "_load_optimized_module", _counting, raising=True)
        for _ in range(3):
            assert load_optimized_synthesis_module() is not None
        assert calls["n"] == 0, "an unchanged artifact was re-parsed on every call"

    def test_miss_logs_at_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Prod ran 6 weeks on a silent fallback; the miss must be visible."""
        from src.rag.cognitive_rag_dspy import load_optimized_synthesis_module

        monkeypatch.chdir(tmp_path)
        with caplog.at_level("INFO", logger="src.rag.cognitive_rag_dspy"):
            load_optimized_synthesis_module(reset=True)

        assert any(
            "optimized" in r.message.lower() and r.levelname == "INFO" for r in caplog.records
        ), [r.message for r in caplog.records]


class TestAgentModuleUsesTheLoadedPrompt:
    """The behavioural half: the runtime must USE it, not merely load it."""

    def test_agent_module_synthesize_carries_optimized_instructions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AgentModule.synthesize must be built from the artifact's instructions.

        Asserting on the instruction TEXT reaching the predictor's signature is
        what makes this a use-test rather than a load-test: a loader whose result
        is dropped on the floor still passes an is-not-None check.
        """
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            AgentModule,
            load_optimized_synthesis_module,
        )

        optimized_text = "OPTIMIZED-1486: cite each retrieved passage explicitly."
        monkeypatch.chdir(tmp_path)
        _write_artifact(
            tmp_path / "optimized_modules", OPTIMIZED_SYNTHESIS_AGENT_NAME, optimized_text
        )
        load_optimized_synthesis_module(reset=True)

        agent = AgentModule(agent_registry={})
        instructions = [
            getattr(getattr(p, "signature", None), "instructions", "")
            for p in agent.synthesize.predictors()
        ]
        assert any(optimized_text in (i or "") for i in instructions), instructions

    def test_agent_module_falls_back_to_base_signature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No artifact -> the base ChainOfThought, and construction still works."""
        from src.rag.cognitive_rag_dspy import AgentModule, load_optimized_synthesis_module

        monkeypatch.chdir(tmp_path)
        load_optimized_synthesis_module(reset=True)

        agent = AgentModule(agent_registry={})
        assert agent.synthesize is not None
        assert hasattr(agent.synthesize, "predictors")

    def test_a_corrupt_artifact_never_breaks_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The RAG runtime is user-facing; a bad artifact must degrade, not 500."""
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            AgentModule,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        artifact_dir = tmp_path / "optimized_modules" / OPTIMIZED_SYNTHESIS_AGENT_NAME
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "gepa_v1_broken.json").write_text("{not valid json")

        load_optimized_synthesis_module(reset=True)
        agent = AgentModule(agent_registry={})
        assert agent.synthesize is not None


class TestNewestResolutionIsNumeric1496:
    """Newest-artifact resolution must sort the gepa_v<n> suffix numerically (#1496).

    Two independent sites resolve "newest": versioning.load_optimized_module
    (what gets parsed) and _artifact_signature here (what keys the module-reload
    cache). Lexicographic name order inverts at v10 ("gepa_v10..." < "gepa_v2...",
    with "gepa_v9..." greatest of all), pinning both to a stale artifact after
    the 10th save. Because the two sites are consistent-together today, fixing
    only ONE desynchronizes the reload cache from what the loader parses —
    worse than both being wrong together — so the agreement test below is a
    desync guard: it goes red if the two ever rank the same directory
    differently.
    """

    def _write_versions(self, tmp_path: Path, upto: int = 10) -> None:
        from src.rag.cognitive_rag_dspy import OPTIMIZED_SYNTHESIS_AGENT_NAME

        root = tmp_path / "optimized_modules"
        for n in range(1, upto + 1):
            _write_artifact(
                root,
                OPTIMIZED_SYNTHESIS_AGENT_NAME,
                f"VERSION-{n}",
                version_id=(f"gepa_v{n}_{OPTIMIZED_SYNTHESIS_AGENT_NAME}_202512{n:02d}_100000"),
            )

    def test_signature_probe_resolves_v10_not_v9(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache key must track the artifact a numeric sort calls newest."""
        from src.rag.cognitive_rag_dspy import _artifact_signature

        monkeypatch.chdir(tmp_path)
        self._write_versions(tmp_path)

        signature = _artifact_signature()
        assert signature is not None
        assert Path(signature[0]).name.startswith("gepa_v10_"), signature[0]

    def test_loader_and_signature_probe_agree_on_newest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DESYNC GUARD: both resolution sites must rank the same directory
        identically. Exercised on an adversarial set (the v9/v10 boundary plus
        an unversioned straggler) so that changing the ordering in only one of
        versioning.py / cognitive_rag_dspy.py fails here."""
        import dspy

        from src.optimization.gepa.versioning import load_optimized_module
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            EvidenceSynthesisSignature,
            _artifact_signature,
        )

        monkeypatch.chdir(tmp_path)
        self._write_versions(tmp_path)
        # An unversioned name that lexicographically outranks every real
        # version; valid JSON so a wrong selection fails the assert, not earlier.
        straggler = (
            tmp_path
            / "optimized_modules"
            / OPTIMIZED_SYNTHESIS_AGENT_NAME
            / "gepa_zzz_manual_copy.json"
        )
        straggler.write_text(
            json.dumps(
                {
                    "version_id": "gepa_zzz_manual_copy",
                    "created_at": "2025-12-03T10:00:00",
                    "instruction_hash": "0" * 64,
                    "module_state": {},
                    "metadata": {},
                }
            )
        )

        signature = _artifact_signature()
        assert signature is not None

        _, metadata = load_optimized_module(
            lambda: dspy.ChainOfThought(EvidenceSynthesisSignature),
            agent_name=OPTIMIZED_SYNTHESIS_AGENT_NAME,
        )
        assert metadata["source_path"] == signature[0], (
            "the reload-cache key and the loader disagree on which artifact is "
            "newest — the cache will either thrash or pin a stale module"
        )

    def test_v10_supersedes_a_cached_v9_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: a worker holding v9 must serve v10 once it lands.

        Under the lexicographic sort the probe keeps resolving v9 as newest
        (path and mtime unchanged), so the cache never invalidates and the
        stale module is served forever — the #1496 failure mode.
        """
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        self._write_versions(tmp_path, upto=9)
        assert load_optimized_synthesis_module(reset=True) is not None

        _write_artifact(
            tmp_path / "optimized_modules",
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            "VERSION-10-NEWEST",
            version_id=(f"gepa_v10_{OPTIMIZED_SYNTHESIS_AGENT_NAME}_20251210_100000"),
        )
        reloaded = load_optimized_synthesis_module()
        assert reloaded is not None
        instructions = [
            getattr(getattr(p, "signature", None), "instructions", "")
            for p in reloaded.predictors()
        ]
        assert any("VERSION-10-NEWEST" in (i or "") for i in instructions), instructions


class TestArtifactRootMatchesComposeWiring:
    def test_loader_reads_the_root_compose_mounts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The artifact must land under optimized_modules/ (a declared volume).

        tests/integration/test_optimized_artifacts_compose_wiring.py pins
        /app/optimized_modules as a named volume shared between the worker that
        writes and the api that reads. A loader pointed anywhere else would be
        invisible across containers.
        """
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        path = _write_artifact(tmp_path / "optimized_modules", OPTIMIZED_SYNTHESIS_AGENT_NAME, "x")

        assert path.parent.parent.name == "optimized_modules"
        assert json.loads(path.read_text())["agent_name"] == OPTIMIZED_SYNTHESIS_AGENT_NAME
        assert load_optimized_synthesis_module(reset=True) is not None
