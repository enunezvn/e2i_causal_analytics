"""#1419: the causal_effect dispatch SLA is raised on a MEASURED completion.

Router convention (#1351): raise past 120 s only with a measured completion
time. Measured 2026-07-31 on the live 37,371-row conversion frame:

    estimation (tournament 36 s + full-frame refit 34 s + prep) ~93 s
    + subsampled reconstruction (5,000 rows)                     ~12 s
    + 1-sim calibration                                          ~2 s
    + placebo 30 sims x 2.13 s                                   ~64 s
    + random_common_cause 20 sims x 2.59 s                       ~52 s
    + analytic e-value                                           ~0 s
    ------------------------------------------------------------------
    critical-gates chat turn                                    ~223 s

300 s x _CAUSAL_DEADLINE_FRACTION (0.8) = 240 s cooperative deadline covers
the measured 223 s critical path with margin; the non-critical bootstrap
(50 inference-bearing sims x ~11.7 s ~ 585 s) degrades to an honest SKIPPED
result under the #1419 skip policy + heavy-cost gate. 300 s also aligns with the host-nginx proxy_read_timeout
ceiling — the dispatch stays the binding constraint end-to-end.
"""

from src.agents.orchestrator.nodes.router import RouterNode


class TestCausalDispatchMeasuredSla:
    def test_causal_effect_timeout_is_the_measured_300s(self):
        dispatches = RouterNode.INTENT_TO_AGENTS["causal_effect"]
        causal = next(d for d in dispatches if d["agent_name"] == "causal_impact")
        assert causal["timeout_ms"] == 300_000
