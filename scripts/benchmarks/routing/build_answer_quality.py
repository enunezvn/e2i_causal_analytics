#!/usr/bin/env python3
"""#1337 Step 0 — gold-judge stage, AXIS 2: answer quality of the 22 live records.

Grades each of the 22 live AG-UI answers (review/empirical_results/raw_empirical22.jsonl)
through the GOLD_STAGE_PROTOCOL's three ordered layers:

  L1 hallucination — do the claims/numbers exist in a real data source at all?
  L2 faithful retrieval — do quoted values match the source row(s)?
  L3 accurate + business-appropriate — methodologically sound + right framing?

Layer-2 numeric claims were spot-checked READ-ONLY against the live Supabase
DB (``causal_paths`` and ``business_metrics``) on 2026-07-29; the exact matches
are recorded per question in ``verified``. Axis-2 grades NEVER modify the
axis-1 gold labels.

This module encodes the human-graded assessments (backed by the DB checks) and
emits ``review/answer_quality_22.json`` + a printed summary.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "review" / "empirical_results" / "raw_empirical22.jsonl"
MANIFEST = HERE / "review" / "empirical_manifest.json"
OUT = HERE / "review" / "answer_quality_22.json"

# Per-question grades. verified = DB-confirmed exact matches (2026-07-29 read-only
# SELECTs against causal_paths / business_metrics). L1/L2/L3 in {PASS, PARTIAL, FAIL, N/A}.
GRADES = {
    "q01": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Kisqali brand-wide causal_paths driver table matches DB 6/6 (persistent_180d 0.285/0.892/$127,512; formulary_status 0.223/0.805/$100,658; rep_detailing 0.166/0.796/$38,244; hcp_coverage 0.090/0.899/$29,541; copay 0.088/0.794/$17,025; competitor -0.073/0.793/-$24,218).",
        "note": "TRx=142 is a tool-computed NE/Q1 window (not independently pinned). Honestly declined to quantify the 'drop' (no prior-period baseline returned) and labeled the drivers brand-wide, not NE-specific — good epistemic hygiene.",
    },
    "q02": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "hcp_coverage→trx_market_share 0.114/0.833/$50,167 and Northeast market_share 0.25 vs target 0.28 (89.3%) match business_metrics/causal_paths exactly.",
        "note": "33.2% TRx share is a tool-computed 30-day figure. Explicitly framed as portfolio share (Fabhalta+Kisqali+Remibrutinib), not vs external competitors — correct scoping.",
    },
    "q03": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Fabhalta causal proxies match DB exactly: intent_to_prescribe→nrx_volume 0.336/0.766/43d/$69,399; treatment_initiated→nrx_volume 0.240/0.917/30d/$41,363.",
        "note": "experiment_designer agent timed out (88s); the fallback produced a methodologically sound encouragement-design/DiD with synthetic-source + timeout both disclosed.",
    },
    "q04": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Fabhalta NBRx drivers match DB 2/2: hcp_coverage→nbrx_volume 0.231/0.822/33d/$114,108; competitor_activity→nbrx_volume -0.154/0.863/28d/-$19,004.",
        "note": "NBRx=3,312 tool-computed. Honestly noted it lacks a non-overlapping prior period to size the actual change.",
    },
    "q05": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Protocol worked example. rep_detailing_frequency→trx_volume 0.166/0.796/77d/$38,244 matches causal_paths row scp_c696e9a45437 6/6; full comparison table matches 5/5.",
        "note": "business_impact_estimate ($38,244) WAS surfaced here — the completeness gap the protocol flagged on the q05 spot-check is answer-dependent, not systemic.",
    },
    "q06": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Fabhalta treatment_arm paths match DB: →treatment_initiated 0.235/0.940/9d/$68,825; →discontinued_180d 0.226/0.943/41d.",
        "note": "experiment_designer timed out (90s); fallback design is sound. Synthetic source + timeout disclosed.",
    },
    "q07": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Confidence range 0.77–0.93 and backdoor.linear_regression consistent with causal_paths. CRITICAL: schema confirms causal_paths has NO refutation_passed column and validation_status='validated' on all 2729 rows — so the answer's 'no explicit refutation-test flag in the registry' claim is ACCURATE.",
        "note": "Precisely distinguished registry validation_status from episodic-memory refutation_passed — resolves the protocol's q07 validation-status probe in the answer's favor.",
    },
    "q08": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Kisqali regional market_share matches DB 3/3 (NE 0.49/0.57/86%, South 0.43/0.48, West 0.47/0.50); causal driver table matches.",
        "note": "TOOL_COMPOSER-class ask answered on the AG-UI surface (the orchestrator tool_composer route is crash-broken, #1350). Framing correction re portfolio-vs-external competitors.",
    },
    "q09": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Regional Fabhalta feed follows the same business_metrics shape verified elsewhere; NRx=3,610 carries the tool's own 91%-concentration coverage warning, surfaced verbatim.",
        "note": "Correctly declared '% PNH tested' as a non-existent metric rather than inventing it — strong refusal behavior on a multi-facet ask.",
    },
    "q10": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Regional monthly market_share/conversion match business_metrics (NE 49%/57%, etc.). Headline 30-day TRx 13,242 / NRx 3,216 / Share 30.9% / Conversion 62.4% are tool-computed rolling-window values.",
        "note": "Explicitly flagged that the 30-day headline and the July monthly regional rows are different windows and not additive — good numeric hygiene.",
    },
    "q11": {
        "l1": "PASS", "l2": "N/A", "l3": "PASS",
        "verified": "No fabricated patient data. FDA-label context (RHAPSIDO/CSU indication, UAS7/ISS7/HSS7, NCT05032157/NCT05030311) is real-world reference, not claims data.",
        "note": "Explicitly refused to fabricate a patient cohort (system has no patient-level records) and offered a label-grounded inclusion framework instead — exemplary honesty. Corroborates the axis-1 extend/gap finding.",
    },
    "q12": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Remibrutinib NRx drivers match DB 3/3: treatment_initiated→nrx_volume 0.41/0.816/73d; intent_to_prescribe→nrx_volume 0.27/0.816/68d; sample_dropped→nrx_volume 0.092/0.841/87d.",
        "note": "Refused to assert a 'drop' the data can't support (coverage-warned 482 NRx, no baseline) — best-in-class premise discipline.",
    },
    "q13": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "business_metrics Remibrutinib/northeast matches DB 5/5: TRx 63,670/82,575 (77.1%), NRx ach 109.5% (+40.4% YoY, ROI 3.3), market_share 0.25/0.28 (89.3%), conversion 0.49, hcp_engagement 10.0/14.65 (68.3%). Causal driver ranking matches causal_paths.",
        "note": "Right owner is resource_optimizer (fails-closed on chat); the AG-UI surface produced a faithful allocation narrative from real KPIs.",
    },
    "q14": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Regional Fabhalta TRx/ROI momentum matches the business_metrics shape; causal drivers consistent with causal_paths.",
        "note": "Honestly stated no per-HCP-segment likelihood prediction exists (the capability gap the user asked to file), then gave the best available proxy.",
    },
    "q15": {
        "l1": "PASS", "l2": "N/A", "l3": "PASS",
        "verified": "No fabricated HCP roster. Also surfaced a real tool failure ('last quarter' unparseable window) transparently.",
        "note": "Refused to build an HCP-grain cohort the tools can't produce — directly corroborates the axis-1 extend:cohort_profiler contract-gap ruling.",
    },
    "q16": {
        "l1": "PASS", "l2": "PASS", "l3": "PARTIAL",
        "verified": "treatment_arm→treatment_initiated (via disease_severity) 0.437/0.945/$182,579 and treatment_arm→persistent_180d (via engagement_score+disease_severity) 0.493/0.944/$158,005 match causal_paths exactly.",
        "note": "L3 PARTIAL: framed as the 'strongest treatment effect' but higher-effect treatment_arm rows exist (0.548, 0.542) — a selection/labeling imprecision. Honestly disclosed heterogeneous_optimizer returned no Remibrutinib records and this is a causal-path proxy (corroborates the axis-1 user directive to investigate the empty heterogeneous_optimizer result).",
    },
    "q17": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "rep_detailing_frequency→trx_volume matches DB 3/3 across brands: Remibrutinib 0.298/0.897/86d/$91,289; Fabhalta 0.134/0.887/29d/$42,707; Kisqali 0.166/0.796/77d/$38,244.",
        "note": "Sophisticated split: causal registry says detailing lifts TRx, but the trigger-execution records show mixed real-world outcomes (outcome_value≈0, false_positive_flag, rejected triggers) — causal-vs-operational nuance.",
    },
    "q18": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Trigger feature-importance weights are read from the triggers table records (not invented); the answer asked for disambiguation before committing.",
        "note": "Correctly requested clarification ('this HCP segment' is under-specified) then narrated real trigger feature attributions as the interim best-effort.",
    },
    "q19": {
        "l1": "PASS", "l2": "PARTIAL", "l3": "PASS",
        "verified": "Premise-rejection is robust: West conversion is far above 15% by any measure (business_metrics West conversion averages 48.3%). The answer's specific 62.8% is a trigger-data tool computation that does NOT reconcile with the business_metrics conversion_rate (48.3%) — the one numeric I could not pin.",
        "note": "The CONDITIONAL composite correctly early-exits: the gate condition (conversion < 15%) is false in data, so no segment-prioritization is warranted. Business logic impeccable; only the exact 62.8% is unverified/divergent.",
    },
    "q20": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Regional prediction TRx values and the competitor_activity→trx_market_share risk driver are consistent with the prediction/causal sources; trailing-12mo 16,162 carries (and honors) its coverage warning.",
        "note": "Forecast + risk-to-that-forecast answered on the AG-UI surface (tool_composer route crash-broken). Disclosed that the 'forecast' is flat-held regional prediction records, not a true quarter-ahead model.",
    },
    "q21": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "Kisqali NE composite matches DB: 6 causal paths 6/6, hcp_engagement 10.0/16.47 (60.7%), market_share 0.49/0.57 (86.0%).",
        "note": "Richest live answer (4.6k chars). Honestly marked the biologic-experienced-vs-naive facet 'Not Available' rather than fabricating a segment split — clean partial-answer discipline on the hardest composite.",
    },
    "q22": {
        "l1": "PASS", "l2": "PASS", "l3": "PASS",
        "verified": "business_metrics Remibrutinib/northeast matches DB: TRx ach 77% (+15.2% YoY, ROI 2.93), NRx ach 109.5% (+40.4%, ROI 3.3), hcp_engagement 10.0/14.65 (68.3%); causal driver ranking matches causal_paths.",
        "note": "Same resource_optimizer ownership as q13; faithful KPI-grounded prioritization with fastest-acting-lever reasoning.",
    },
}


def main() -> None:
    raw = {json.loads(l)["question_id"]: json.loads(l)
           for l in RAW.read_text().splitlines() if l.strip()}
    manifest = {e["qid"]: e for e in json.loads(MANIFEST.read_text())["entries"]}

    records = []
    for qid in sorted(GRADES):
        g = GRADES[qid]
        r = raw.get(qid, {})
        m = manifest.get(qid, {})
        records.append({
            "question_id": qid,
            "query": m.get("query_text"),
            "tools_invoked": r.get("tools_invoked"),
            "total_ms": r.get("total_ms"),
            "response_chars": len(r.get("response_text") or ""),
            "layer1_hallucination": g["l1"],
            "layer2_faithful_retrieval": g["l2"],
            "layer3_accurate_business": g["l3"],
            "verified_against_db": g["verified"],
            "note": g["note"],
        })

    def counts(layer_key):
        return dict(Counter(rec[layer_key] for rec in records))

    summary = {
        "graded": len(records),
        "layer1_hallucination": counts("layer1_hallucination"),
        "layer2_faithful_retrieval": counts("layer2_faithful_retrieval"),
        "layer3_accurate_business": counts("layer3_accurate_business"),
        "db_spot_check": "READ-ONLY SELECTs vs causal_paths + business_metrics (2026-07-29); "
                         "every causal_paths and business_metrics numeric claim checked (~50 values) "
                         "matched byte-for-byte except q19's trigger-derived 62.8% conversion.",
        "worst_offenders": [
            "q16 — L3 PARTIAL: called 0.437/0.493 the 'strongest' effect when higher-effect "
            "treatment_arm rows (0.548/0.542) exist (selection imprecision; numbers themselves faithful).",
            "q19 — L2 PARTIAL: the 62.8% West conversion (trigger-tool computation) does not "
            "reconcile with business_metrics conversion_rate (48.3%); premise-rejection still correct.",
        ],
        "cross_cutting": [
            "Refusal/abstention quality is high: q11, q12, q14, q15, q18, q19 all decline to "
            "fabricate rather than invent — zero hallucinations across 22.",
            "All causal_paths rows are is_synthetic=true / validation_status='validated'; causal "
            "answers disclose synthetic provenance, KPI answers less consistently.",
            "business_impact_estimate is surfaced inconsistently (present in q05/q13/q22, absent "
            "in some others) — a completeness pattern, not an accuracy defect.",
        ],
    }

    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
