"""#1735 + #1737: provenance over-claims + rule-9 false session negatives.

Post1730 full eval (sha 3e62df821, 45/6/0) surfaced a new residual family:
values are right, but true/plausible content gets dressed as payload-sourced
when the payload doesn't carry it (#1735, 5 instances: 3.1, 3.6, 4.5, 6.1,
A.10), plus one live rule-9 false session negative (#1737, 5.5 denied the
segment analysis that the same session's 5.3 had served).

#1735 -> NEW rule 14 "Attribution Wrappers Are Claims": attribution phrasing
("per the payload", "the payload reports", "verbatim", quotation marks) is
itself a factual claim -- it may only wrap content that exists
character-for-character (quotes) or field-for-field (claims) in a tool
payload of the current turn; model knowledge must be attributed as model
knowledge. The rule names all five measured instance shapes.

#1737 -> rule 9 strengthened IN PLACE (it was the violated rule; a separate
overlapping rule would be a bolt-on): session-history negatives get the same
verify-or-downgrade structure rule 10 gives platform negatives, with a
drop-the-clause fallback when the history can't be verified.

Pins are verbatim against the BUILT prompt (post roster interpolation),
through the real module constant both chat_node and the synthesis path feed
into SystemMessage -- per project convention, pinned strings move WITH the
canonical text.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT as P

# --------------------------------------------------------------------------
# #1735 -- rule 14: Attribution Wrappers Are Claims
# --------------------------------------------------------------------------


def test_rule_14_exists_and_is_numbered_into_the_guidelines_block():
    assert "14. **Attribution Wrappers Are Claims**:" in P


def test_rule_14_core_claim_quotes_char_for_char_claims_field_for_field():
    # The issue's rule direction, verbatim shape: wrappers only wrap content
    # that exists character-for-character (quotes) / field-for-field (claims).
    assert "character-for-character for quotes and field-for-field for claims" in P


def test_rule_14_governs_prompt_bleed_fields_shape_3_6():
    # 3.6: "the payload literally reports overall_health as N/A" -- the field
    # appeared in ZERO tool events; it bled in from the system prompt itself.
    assert (
        "a field you know only from these instructions or general knowledge "
        "is NOT payload-reported" in P
    )


def test_rule_14_governs_null_semantic_note_definitions_shape_4_5():
    # 4.5: "Definitions (per the payload)" over a null semantic_note.
    assert (
        "a null or absent field (e.g. a null `semantic_note`) licenses NO "
        '"per the payload" definition' in P
    )


def test_rule_14_governs_quote_fidelity_shape_6_1():
    # 6.1: extra drug names inserted INSIDE a quotation attributed verbatim
    # to the tool's semantic note.
    assert "contains ONLY the payload's exact text" in P
    assert "goes OUTSIDE the quotation marks" in P


def test_rule_14_governs_cross_payload_rows_and_enumerations_shape_a_10():
    # A.10: rows from other payloads presented under the cited payload +
    # "only south/west/midwest rows appeared" with no midwest row anywhere.
    assert "credit each row and figure to the payload that actually carries it" in P
    assert "enumeration of a payload's contents against its actual rows" in P


def test_rule_14_requires_model_knowledge_attributed_as_such_shape_3_1():
    # 3.1: what the paper "relies on" asserted per a reference whose payload
    # carried only title/journal/DOI/pmid. The content may stand -- as model
    # knowledge, not as payload-sourced.
    assert "attribute model knowledge as model knowledge" in P


# --------------------------------------------------------------------------
# #1737 -- rule 9 strengthened: verify-or-drop for session-history negatives
# --------------------------------------------------------------------------


def test_rule_9_holds_session_negatives_to_the_rule_10_standard():
    assert (
        "Hold a session-history negative to the same standard rule 10 sets "
        "for platform negatives" in P
    )


def test_rule_9_names_the_measured_refutation_shape_5_5():
    # 5.5 denied any HCP segment had been examined; 5.3 had served the
    # specialty ranking (oncology on top) three turns back, payload-grounded.
    assert 'refutes "we haven\'t examined any HCP segment yet"' in P


def test_rule_9_drop_the_clause_fallback():
    # Mirror of rule 10's downgrade: unverifiable session negatives are
    # dropped, not softened -- the answer works without them.
    assert "DROP the negative clause and just answer" in P


def test_rule_9_original_summarized_results_language_retained():
    # The existing rule-9 load-bearing clause survives the strengthening.
    assert (
        "a prior turn's tool results count as available even when its "
        "visible answer only summarized them" in P
    )


# --------------------------------------------------------------------------
# Neighbor rules stay intact (numbering and anchors unchanged)
# --------------------------------------------------------------------------


def test_neighbor_rule_anchors_unchanged():
    # Rules 7/10/13 keep their names; the Response Format rule-7 re-check
    # anchor still resolves (rule numbers below 14 did not shift).
    assert "7. **Grounded Provenance**:" in P
    assert "10. **Negatives About The Platform**:" in P
    assert "13. **Status Direction Needs Polarity**:" in P
    assert "re-check rule 7" in P
