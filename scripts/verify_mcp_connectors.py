#!/usr/bin/env python3
"""MCP Connector Verification Script.

Phase 4 of Skills & MCP Implementation Plan.
Tests whether Anthropic's hosted MCP connectors (ChEMBL, ClinicalTrials.gov, PubMed)
are accessible via the API.

Usage:
    python scripts/verify_mcp_connectors.py
"""

import json
import os
import time
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
    exit(1)


def verify_connectors() -> dict:
    """Test MCP connector availability via Anthropic API.

    Returns:
        Dictionary with test results for each connector.
    """
    # Initialize client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        return {"error": "ANTHROPIC_API_KEY not set"}

    client = anthropic.Anthropic(api_key=api_key)

    # Define connector test cases
    # Each prompt explicitly requests use of the named connector
    connectors = [
        (
            "ChEMBL",
            "Using the ChEMBL connector, search for ribociclib (Kisqali) and return its molecular properties including molecular weight, SMILES structure, and any available pharmacological data.",
        ),
        (
            "ClinicalTrials.gov",
            "Using the ClinicalTrials.gov connector, find active clinical trials for HR+/HER2- breast cancer. List the trial identifiers (NCT numbers), titles, and current status.",
        ),
        (
            "PubMed",
            "Using the PubMed connector, search for recent papers on causal inference methods in pharmaceutical marketing analytics. Return PMIDs, titles, and publication dates.",
        ),
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "claude-sonnet-4-20250514",
        "connectors": {},
    }

    print("=" * 60)
    print("MCP Connector Verification")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Model: {results['model']}")
    print("=" * 60)

    for name, prompt in connectors:
        print(f"\nTesting {name} connector...")
        print("-" * 40)

        try:
            start = time.time()
            response = client.messages.create(
                model=results["model"],
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed_ms = (time.time() - start) * 1000

            response_text = response.content[0].text if response.content else ""

            # Store results
            results["connectors"][name] = {
                "status": "success",
                "latency_ms": round(elapsed_ms, 2),
                "response_length": len(response_text),
                "response_preview": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "full_response": response_text,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

            print(f"Status: SUCCESS")
            print(f"Latency: {elapsed_ms:.2f}ms")
            print(f"Response length: {len(response_text)} chars")
            print(f"Preview: {response_text[:200]}...")

        except anthropic.APIError as e:
            results["connectors"][name] = {
                "status": "api_error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            print(f"Status: API ERROR - {e}")

        except Exception as e:
            results["connectors"][name] = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            print(f"Status: ERROR - {e}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    for name, data in results["connectors"].items():
        print(f"\n{name}:")
        if data["status"] == "success":
            response = data["full_response"].lower()

            # Check for indicators of real external data access
            indicators = {
                "ChEMBL": [
                    "chembl" in response,
                    "molecular weight" in response and any(c.isdigit() for c in response),
                    "smiles" in response,
                ],
                "ClinicalTrials.gov": [
                    "nct" in response,
                    any(f"nct0{i}" in response for i in range(10)),
                    "clinicaltrials.gov" in response,
                ],
                "PubMed": [
                    "pmid" in response or "pubmed" in response,
                    any(f"202{i}" in response for i in range(0, 7)),  # Recent years
                    "doi" in response,
                ],
            }

            connector_indicators = indicators.get(name, [])
            matches = sum(connector_indicators)
            total = len(connector_indicators)

            # Check if response indicates connector unavailability
            unavailable_indicators = [
                "don't have access" in response,
                "cannot access" in response,
                "not able to" in response,
                "no direct access" in response,
                "unable to search" in response,
                "i don't have" in response,
                "i cannot" in response,
            ]

            if any(unavailable_indicators):
                data["connector_available"] = False
                data["analysis"] = "Response indicates connector is NOT available"
                print(f"  Connector Available: NO (response indicates no access)")
            elif matches >= total // 2 + 1:
                data["connector_available"] = "LIKELY"
                data["analysis"] = f"Response contains {matches}/{total} expected data indicators"
                print(f"  Connector Available: LIKELY ({matches}/{total} indicators)")
            else:
                data["connector_available"] = "UNCLEAR"
                data["analysis"] = f"Only {matches}/{total} indicators - may be Claude's knowledge"
                print(f"  Connector Available: UNCLEAR ({matches}/{total} indicators)")
        else:
            data["connector_available"] = False
            print(f"  Connector Available: NO (error occurred)")

    return results


def save_results(results: dict, filepath: str = "scripts/mcp_connector_results.json"):
    """Save results to JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


def main():
    results = verify_connectors()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    available = []
    unavailable = []
    unclear = []

    for name, data in results.get("connectors", {}).items():
        status = data.get("connector_available", False)
        if status is True or status == "LIKELY":
            available.append(name)
        elif status == "UNCLEAR":
            unclear.append(name)
        else:
            unavailable.append(name)

    print(f"Available/Likely: {available or 'None'}")
    print(f"Unavailable: {unavailable or 'None'}")
    print(f"Unclear: {unclear or 'None'}")

    # Recommendation
    print("\n" + "-" * 40)
    print("RECOMMENDATION:")
    if available:
        print(f"  Proceed with MCP Gateway (Phase 5) for: {available}")
    if unavailable:
        print(f"  Consider alternatives for: {unavailable}")
        print("  Options: Direct API integration, Skills-only approach")
    if unclear:
        print(f"  Needs manual verification: {unclear}")

    # Save results
    save_results(results)

    return results


if __name__ == "__main__":
    main()
