"""#1475 WI3: guard the unguarded DSPy configure in create_production_cognitive_workflow.

Under dspy 3.1.0 ownership semantics the FIRST ``dspy.configure`` permanently
binds an owner thread/task, and any later ``configure`` from a different
thread raises RuntimeError. Every request-reachable configure site guards on
``settings.lm is not None`` (canonical pattern:
src/api/routes/chatbot_dspy.py:62); ``create_production_cognitive_workflow``
at src/rag/cognitive_rag_dspy.py was the one site without it — a future
non-owner-thread caller would crash even though an LM is already configured.
"""

from unittest.mock import MagicMock, patch

import src.rag.cognitive_rag_dspy as crd


def _run_factory(**kwargs):
    """Call the factory with the heavy graph build stubbed out.

    The unit under test is ONLY the configure guard; the adapters are cheap
    real constructions, but ``create_dspy_cognitive_workflow`` builds DSPy
    modules + a compiled LangGraph, which is not what these tests pin.
    """
    with patch.object(crd, "create_dspy_cognitive_workflow", return_value=MagicMock()):
        return crd.create_production_cognitive_workflow(**kwargs)


class TestConfigureGuard:
    def test_configure_skipped_when_lm_already_configured(self):
        """The missing guard: with settings.lm already set, configure must NOT
        be called again (dspy 3.1.0 would raise from a non-owner thread)."""
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = object()  # an LM is already configured

        with patch.object(crd, "dspy", mock_dspy):
            _run_factory(configure_dspy=True)

        mock_dspy.configure.assert_not_called()
        mock_dspy.LM.assert_not_called()

    def test_configure_runs_when_lm_not_configured(self):
        """Preserved behavior: unconfigured process still gets the env-resolved
        model configured."""
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None

        with (
            patch.object(crd, "dspy", mock_dspy),
            patch(
                "src.optimization.dspy_lm.get_default_dspy_model",
                return_value="openai/test-model",
            ),
        ):
            _run_factory(configure_dspy=True)

        mock_dspy.LM.assert_called_once_with("openai/test-model")
        mock_dspy.configure.assert_called_once_with(lm=mock_dspy.LM.return_value)

    def test_explicit_lm_model_still_used_when_unconfigured(self):
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None

        with patch.object(crd, "dspy", mock_dspy):
            _run_factory(configure_dspy=True, lm_model="anthropic/some-model")

        mock_dspy.LM.assert_called_once_with("anthropic/some-model")
        mock_dspy.configure.assert_called_once_with(lm=mock_dspy.LM.return_value)

    def test_configure_dspy_false_never_configures(self):
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None

        with patch.object(crd, "dspy", mock_dspy):
            _run_factory(configure_dspy=False)

        mock_dspy.configure.assert_not_called()

    def test_guard_tolerates_settings_without_lm_attribute(self):
        """Mirror the canonical hasattr() defensiveness: a settings object with
        no ``lm`` attribute means 'not configured' — configure proceeds."""

        class _NoLmSettings:
            pass

        mock_dspy = MagicMock()
        mock_dspy.settings = _NoLmSettings()

        with (
            patch.object(crd, "dspy", mock_dspy),
            patch(
                "src.optimization.dspy_lm.get_default_dspy_model",
                return_value="openai/test-model",
            ),
        ):
            _run_factory(configure_dspy=True)

        mock_dspy.configure.assert_called_once_with(lm=mock_dspy.LM.return_value)
