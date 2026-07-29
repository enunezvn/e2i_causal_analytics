# UI Pass — Copilot Chat demo beats (chrome-devtools, 2026-07-29 ~04:40–04:55 UTC)

Browser session against https://eznomics.site (headless Chrome, admin login, dashboard page).
UI session threadId / DB session_id: `9252519b-5664-4417-bd65-fef39597b097`.
Screenshots in `screenshots/`.

## Beat results

| # | Beat | Result | Evidence |
|---|------|--------|----------|
| 1 | Opener + suggestion pills at chat open | **PASS** — 4 contextual opener pills ("Chart the TRx trend", "Remibrutinib market share", "Executive summary", "Biggest KPI movers") | `01_opener_pills.jpg` |
| 2 | T1 answer + per-turn pill refresh (1.1) | **PASS** — answer identical to API pass (13,242 TRx, source + window caveat); pills refreshed to TRx-specific follow-ups | `02_t1_answer_pill_refresh.jpg` |
| 3 | T3 progress renderer (1.4) | **PASS** — "Working… 25% → 75%" with step text ("Processing your query…", "Synthesizing tool results…"); pills disabled + Send→Stop during flight | `03_t3_progress_renderer.jpg` |
| 4 | T3 streaming answer with live progress | **PASS** — causal-driver table (6 drivers, effect sizes, confidence, $ impact) streamed while the card showed 75% | `04_t3_streaming_with_progress.jpg` |
| 5 | T4 decomposition (6.1) | **PASS** — progress card showed "Executing 5 tool(s): kpi_calculate_tool ×3, causal_analysis_tool, e2i_data_query_tool" (parallel fan-out visible to the user) | `05_t4_tool_decomposition.jpg` |
| 6 | T4 answer quality (6.1) | **PASS** — honest scope framing (portfolio-internal share, explicitly declines to attribute vs. untracked external competitors = T6 no-hallucination), share table, drivers, prioritized rep plan (Northeast → West → maintain) | `06_t4_61_answer.jpg` |
| 7 | A.7 concurrent/interrupt | **PASS** — second question typed+sent mid-stream; both turns completed sequentially, no corruption; second answer used episodic memory ("This is the same question I analyzed earlier — here's a recap") | `07_a7_interrupt_both_answered.jpg` |
| 8 | Immortal-"Working…" defect watch | **PARTIAL** — no infinite spinner (spinner + Stop clear on completion), but the completed progress card persists in the transcript with a stale "Working…"/100% header that flips between "Working…" and "Processing Query" across re-renders. Cosmetic; a user scrolling up sees "Working… 100% Response complete". | evaluate_script probes |
| 9 | "Why this agent?" routing chip | Observed — answers carry an expandable "Why this agent? 74%" affordance (agent-routing transparency) | `06_t4_61_answer.jpg` |

## Defect observations (view layer)

- **UI-D1 (capture of the od9uob3 freeze signature)**: one 6.1 send (issued via programmatic
  click during automation) produced a run whose POST /api/copilotkit was killed client-side
  (`net::ERR_ABORTED` right after response headers). The user message stayed in the chat with
  **no progress card, no answer, and no error message — a silent dead turn**. The interrupted
  request's resent `state.messages` was also missing the immediately-preceding completed turn
  (1.4), i.e. the client state snapshot raced the stream completion. A normal re-send of the
  same question worked. This matches the 2026-07-24 "copilot freeze" diagnosis (backend fine,
  view/interaction layer drops the run) — first time captured with the aborted request in hand.
- **UI-D2 (stale progress-card header)**: see beat 8. The card's terminal state is inconsistent
  ("Working…" vs "Processing Query" header on identical 100%/"Response complete" content).
- **Automation note**: the garbled duplicated text visible in the 6.1 user bubble
  ("…differeCompare TRx…quarternce…") is an artifact of this automation (typed text merged
  with leftover programmatic textarea state), not a product bug. The model handled the
  duplicated question correctly regardless.

## Network timings (browser, /api/copilotkit agent-run POSTs)

Server responds with headers in ~340–360 ms (`server-timing: total;dur≈337–358`); all content
streams over `text/event-stream` after that. End-to-end turn times observed in-browser match
the AG-UI API pass (≈8–20 s simple/medium turns, ≈45–80 s for T4 composites) — full per-turn
figures come from `measurements_agui.csv` + `chatbot_analytics` server metrics.

## Suggestion-pill mechanism

Pills are generated per turn by `POST /api/chat/suggestions` (fires after every answer).
Refreshes observed after every completed turn; relevance was consistently high
(e.g. post-1.4: "Chart Kisqali TRx trend: Northeast Q1 vs. prior quarters",
"Compare Northeast vs. other regions", "Dig into competitor activity detail",
"Test detailing frequency impact"). `suggestion_pills_relevant = yes` for all observed turns.
