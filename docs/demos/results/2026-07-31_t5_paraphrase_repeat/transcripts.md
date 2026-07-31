# T5 Paraphrase-Repeat Probe — Transcripts (2026-07-31, #1339)

Raw per-turn Q&A from `scripts/demos/t5_semantic_repeat_probe.py` against the real UI brain
(`POST /api/copilotkit/agent/default`). Tool list, latency, and full answer text per turn.
See `RESULTS.md` for the analysis and verdict; `raw_t5probe.jsonl` for machine-readable events.

---

### t5p1-cold  ·  cold
**Q:** What is TRx for Kisqali?
*ttfb 10109.4ms · total 10111.0ms · tools: kpi_calculate_tool*

> **Kisqali TRx: 13,185 prescriptions**
>
> - Source: WS3-BI-005 (Total Prescriptions), synthetic data source
> - Window: no specific period was requested, so this reflects the engine's default reporting window — the **most recent 30 days** of prescription data, through **2026-07-30**
>
> Let me know if you'd like this broken down by patient segment, line of therapy, or region, or plotted as a trend over time.

---

### t5p1-mid  ·  intervening
**Q:** And what is NRx for Fabhalta?
*ttfb 7554.9ms · total 7556.5ms · tools: kpi_calculate_tool*

> **Fabhalta NRx: 3,298 new prescriptions**
>
> - Source: WS3-BI-006 (New Prescriptions), synthetic data source
> - Window: no specific period was requested, so this reflects the engine's default reporting window — the **most recent 30 days** of prescription data, through **2026-07-30**
>
> Let me know if you'd like this broken down by patient segment/severity, line of therapy, or plotted as a trend.

---

### t5p1-para  ·  paraphrase
**Q:** Remind me — what was Kisqali's total prescription count again?
*ttfb 5362.8ms · total 5364.7ms · tools: none*

> Yes — **Kisqali TRx (Total Prescriptions): 13,185**, from earlier in our conversation.
>
> - Source: WS3-BI-005 (Total Prescriptions), synthetic data source
> - Window: the engine's default reporting window — the **most recent 30 days** of prescription data, through **2026-07-30** (no specific period was requested)

---

### t5p2-cold  ·  cold
**Q:** What is the TRx share for Fabhalta?
*ttfb 5453.8ms · total 5454.8ms · tools: kpi_calculate_tool*

> **Fabhalta TRx Share: 35.9%** (data through 2026-07-30, most recent 30 days of prescription data).
>
> This represents Fabhalta's share of the tracked portfolio's prescriptions — i.e., Fabhalta's TRx volume relative to the combined total of Fabhalta + Kisqali + Remibrutinib (cross-indication tracking), not market share against external competitors like Soliris/Empaveli. No comparison period was requested, so this reflects the engine's default 30-day reporting window only.

---

### t5p2-mid  ·  intervening
**Q:** What about Remibrutinib's NRx?
*ttfb 5637.9ms · total 5643.4ms · tools: kpi_calculate_tool*

> **Remibrutinib NRx: 3,249 new prescriptions** (data through 2026-07-30, most recent 30 days).
>
> No specific time period was requested, so this reflects the engine's default 30-day reporting window — same window basis as the Fabhalta TRx share figure above.

---

### t5p2-para  ·  paraphrase
**Q:** Can you tell me again what portion of total scripts Fabhalta holds?
*ttfb 3471.7ms · total 3473.6ms · tools: none*

> **Fabhalta's TRx share is 35.9%** of the tracked portfolio's total prescriptions (Fabhalta + Kisqali + Remibrutinib), based on data through 2026-07-30 (most recent 30-day window). This is the same figure I reported earlier — it reflects Fabhalta's share of scripts *within Novartis's tracked brand set*, not versus external competitors in the PNH market.

---

### t5p3-cold  ·  cold
**Q:** Why did Kisqali TRx drop in Q1 in the northeast region?
*ttfb 15286.3ms · total 15293.1ms · tools: causal_analysis_tool*

> **Kisqali TRx – Northeast, Q1 2025 (Jan 1–Apr 1, 2025): 142 total prescriptions.** *(Source: kpi_calculate_tool, WS3-BI-005)*
>
> I don't have a non-overlapping prior-period Northeast figure to compare against, so I can't quantify the size of a "drop" from this data alone — but the causal and trigger data point to the likely drivers behind softening TRx in that region:
>
> **Top validated causal drivers of Kisqali TRx (brand-wide, not Northeast-specific):**
> | Driver | Effect size | Confidence | Est. business impact |
> |---|---|---|---|
> | Persistent 180-day adherence → refill continuity | +0.285 | 0.892 | $127.5K |
> | Formulary status → patient access | +0.223 | 0.805 | $100.7K |
> | Rep detailing frequency → HCP engagement | +0.166 | 0.796 | $38.2K |
> | HCP coverage → prescriber base | +0.09 (on share) | 0.899 | $29.5K |
> | Copay support → adherence | +0.088 | 0.794 | $17.0K |
> | **Competitor activity → share of voice** | **-0.073** | 0.793 | **-$24.2K** |
>
> The **negative driver** is competitor share-of-voice pressure — increased competitor activity erodes Kisqali's trx_market_share with ~70-day lag. Combined with any softening in formulary access or rep detailing intensity, this is consistent with a regional dip.
>
> **Northeast-specific signal from triggers:** Kisqali HCPs in the region show recurring **"engagement_gap"** triggers (e.g., trigger `scvtrg_08087`, confidence 0.796; `scvtrg_15957`, confidence 0.754) flagging below-average HCP engagement vs. peers — several were left unactioned or expired, meaning reduced touchpoint frequency likely contributed to the volume softness (aligns with the rep-detailing-frequency causal path above).
>
> **Bottom line:** The drop is most plausibly explained by a combination of (1) reduced rep engagement/detailing in the Northeast (evidenced by unresolved engagement-gap triggers) and (2) competitive share-of-voice pressure, with formulary/access frictions as a secondary factor. I'd need a prior comparable Northeast quarter's TRx to confirm magnitude — want me to pull Q4 2024 Northeast TRx for a clean quarter-over-quarter comparison?

---

### t5p3-mid  ·  intervening
**Q:** What is TRx for Remibrutinib?
*ttfb 4842.0ms · total 4844.2ms · tools: kpi_calculate_tool*

> **Remibrutinib TRx: 14,199 total prescriptions**, covering the engine's default reporting window — the **most recent 30 days** of data (through 2026-07-30). *(Source: kpi_calculate_tool, WS3-BI-005)*
>
> No specific period was requested for this query, so this reflects the standard 30-day snapshot rather than a custom window. Let me know if you'd like it broken down by severity tier, biologic status, IgE tier, or region.

---

### t5p3-para  ·  paraphrase
**Q:** Circle back to Kisqali — what was driving that Northeast decline again?
*ttfb 8689.0ms · total 8692.1ms · tools: none*

> Recapping what I found for the Kisqali Northeast TRx softness:
>
> **Key drivers identified:**
>
> | Driver | Effect size | Confidence | Est. business impact |
> |---|---|---|---|
> | Persistent 180-day adherence → refill continuity | +0.285 | 0.892 | $127.5K |
> | Formulary status → patient access | +0.223 | 0.805 | $100.7K |
> | Rep detailing frequency → HCP engagement | +0.166 | 0.796 | $38.2K |
> | HCP coverage → prescriber base | +0.09 (on share) | 0.899 | $29.5K |
> | Copay support → adherence | +0.088 | 0.794 | $17.0K |
> | **Competitor activity → share of voice** | **-0.073** | 0.793 | **-$24.2K** |
>
> **The two most likely contributors to the Northeast decline specifically:**
>
> 1. **Reduced rep engagement/detailing** — Northeast Kisqali HCPs showed recurring **"engagement_gap"** triggers (e.g., `scvtrg_08087` at 0.796 confidence, `scvtrg_15957` at 0.754), several of which went unactioned or expired. Since rep detailing frequency is a positive causal driver of HCP engagement (+0.166), a gap here directly suppresses TRx.
> 2. **Competitor share-of-voice pressure** — the only *negative* driver in the causal registry (-0.073 effect on trx_market_share, ~70-day lag), meaning increased competitor activity erodes Kisqali's share over time.
>
> Secondary factors (formulary access, adherence/persistence) are important brand-wide levers but I don't have Northeast-specific data confirming they moved — the trigger evidence is strongest for the engagement gap.
>
> **Caveat still stands:** I haven't been able to quantify the actual size of the "decline" since I don't have a prior comparable Northeast TRx figure to compare against. Want me to pull a prior quarter's Northeast TRx now to size the drop?

---

### t5b1-baseline  ·  baseline
**Q:** Remind me — what was Kisqali's total prescription count again?
*ttfb 4937.5ms · total 4938.8ms · tools: kpi_calculate_tool*

> Kisqali's **Total Prescriptions (TRx)** was **13,185** — this figure covers the engine's default reporting window (most recent 30 days of data, through 2026-07-30), not a specific period you requested earlier. Source: WS3-BI-005 (synthetic data).

---

### t5b2-baseline  ·  baseline
**Q:** Can you tell me again what portion of total scripts Fabhalta holds?
*ttfb 5631.7ms · total 5633.2ms · tools: kpi_calculate_tool*

> **Fabhalta's TRx Share is 35.9%**, based on data through the most recent 30-day reporting window (data through **2026-07-30**).
>
> This means Fabhalta accounts for roughly **36% of scripts across the tracked portfolio** (Fabhalta + Kisqali + Remibrutinib combined) — it is **not** a measure of market share against external competitors like Uptravi or other PNH treatments, since those aren't in this data model.

---

### t5b3-baseline  ·  baseline
**Q:** Circle back to Kisqali — what was driving that Northeast decline again?
*ttfb 14798.3ms · total 14804.4ms · tools: causal_analysis_tool*

> Here's what's driving the **Kisqali Northeast** softness, per the causal-path registry (validated chains, min confidence 0.5):
>
> **Top validated causal drivers (TRx, Northeast region):**
>
> | Driver | Effect Size | Confidence | Lag | Est. $ Impact |
> |---|---|---|---|---|
> | Persistent 180d (refill continuity) | +0.285 | 0.892 | 24 days | +$127.5K |
> | Formulary status (patient access) | +0.223 | 0.805 | 78 days | +$100.7K |
> | Rep detailing frequency (HCP engagement) | +0.166 | 0.796 | 77 days | +$38.2K |
> | HCP coverage (prescriber base) | +0.09 | 0.899 | 55 days | +$29.5K |
> | Copay support program (adherence) | +0.088 | 0.794 | 64 days | +$17.0K |
> | **Competitor activity (share of voice)** | **-0.073** | **0.793** | **70 days** | **-$24.2K** |
>
> The one clearly *negative* lever is **competitor activity eroding share of voice**, which directly pulls down `trx_market_share` (-$24.2K estimated impact). Everything else in the list is a positive lever whose *underperformance* — not intrinsic negative effect — is likely compounding the decline: weaker HCP coverage/prescriber base, softer rep detailing cadence, formulary/access friction, and refill continuity gaps all have strong positive effect sizes, meaning if any of these degraded in the Northeast, they'd translate directly into lost TRx.
>
> **Supporting signal from triggers:** Northeast Kisqali HCPs are generating repeated `engagement_gap` triggers (e.g., trigger `scvtrg_08087`, confidence 0.796, "below-average engagement compared to similar HCPs," recommending increased touchpoint frequency) — consistent with the rep-detailing and HCP-coverage drivers above showing up as real-world flags, several of which expired unactioned or were rejected/overridden.
>
> **Bottom line:** The decline looks like a combination of **competitive share-of-voice pressure** plus **underdelivery on the positive-impact levers** (detailing frequency, prescriber coverage, formulary access) — the engagement-gap triggers suggest reps are under-covering Northeast HCPs relative to peers, which is likely the most actionable lever.
>
> Want me to pull the specific Northeast HCP list tied to those unactioned engagement-gap triggers for targeting?
