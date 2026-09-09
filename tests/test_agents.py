"""Unit tests for the Phase 4 agent plumbing (no LLM, no model).

The agents themselves are LLM calls, but the code around them — query
parsing, JSON extraction, claim collection, citation auditing, verdict
counting, report rendering — is pure and testable. These tests pin that
plumbing so prompt or layout changes can't silently break it.

Run with:
    python -m pytest tests/test_agents.py -v
"""

from market_intel.agents import (
    audit_citations,
    audit_claims,
    collect_claims,
    count_verdicts,
    extract_json,
    parse_research_queries,
    render_report,
)

COMPETITORS = ["shopify", "woocommerce", "bigcommerce"]


# --------------------------------------------------------------------------
# Research queries (line-based parsing, unchanged from Phase 4)
# --------------------------------------------------------------------------

def test_parse_research_queries_keeps_valid_lines():
    reply = (
        "shopify payout cuts for developers\n"
        "shopify checkout fees increase\n"   # third shopify line -> dropped (cap 2)
        "woocommerce migration plugins\n"
        "bigcommerce enterprise pricing\n"
        "some line without a brand\n"        # no competitor -> dropped
        "shopify and woocommerce together\n"  # two brands -> dropped
    )
    parsed = parse_research_queries(reply, COMPETITORS, max_per=2)
    assert parsed == {
        "shopify": ["shopify payout cuts for developers", "shopify checkout fees increase"],
        "woocommerce": ["woocommerce migration plugins"],
        "bigcommerce": ["bigcommerce enterprise pricing"],
    }


def test_parse_research_queries_handles_messy_reply():
    # Bullets, numbered markers, stray quotes, empty lines — only clean
    # lines survive, with markers and quotes stripped.
    reply = (
        '- "woocommerce" plugin fees\n'
        "shopify \"developer fees\n"          # unbalanced quote gets stripped
        "## Notes\n"
        "1. bigcommerce pricing\n"
        "\n"
    )
    parsed = parse_research_queries(reply, COMPETITORS, max_per=2)
    assert parsed["shopify"] == ["shopify developer fees"]
    assert parsed["woocommerce"] == ["woocommerce plugin fees"]
    assert parsed["bigcommerce"] == ["bigcommerce pricing"]


def test_parse_research_queries_empty_reply_gives_empty_buckets():
    assert parse_research_queries("", COMPETITORS) == {c: [] for c in COMPETITORS}


# --------------------------------------------------------------------------
# Tolerant JSON extraction
# --------------------------------------------------------------------------

def test_extract_json_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fences_and_trailing_prose():
    reply = (
        "Here is my JSON:\n```json\n{\"verdicts\": [{\"claim_id\": 1, "
        "\"verdict\": \"SUPPORTED\", \"note\": \"ok\"}]}\n```\n"
        "Hope this helps!"
    )
    assert extract_json(reply) == {
        "verdicts": [{"claim_id": 1, "verdict": "SUPPORTED", "note": "ok"}]
    }


def test_extract_json_skips_preamble():
    reply = 'Sure thing!\n\n{"claims": [{"claim": "a}", "citations": [1]}]}'
    # The brace inside the string must not close the object early.
    assert extract_json(reply) == {
        "claims": [{"claim": "a}", "citations": [1]}]
    }


def test_extract_json_none_when_unusable():
    assert extract_json("no json here at all") is None
    assert extract_json('{"unterminated": [1, 2}') is None
    assert extract_json("[1, 2, 3]") is None  # not an object


# --------------------------------------------------------------------------
# Claim collection + mechanical audits
# --------------------------------------------------------------------------

def _customer(structured=True):
    return {
        "structured": structured,
        "pain_points": [{"claim": "payout cuts hurt devs", "citations": [3],
                         "out_of_range": []}],
        "praises": [],
        "switching_themes": [{"claim": "devs look at alternatives",
                              "citations": [1], "out_of_range": []}],
    } if structured else {"structured": False, "free_text": "raw text [n]"}


def _strategy(structured=True):
    return {
        "structured": structured,
        "opportunities": [
            {"title": "Payout dashboard", "claim": "Add payout transparency",
             "citations": [3], "out_of_range": [],
             "why_now": "evidence", "effort": "medium"},
            {"title": "Plugin audit", "claim": "Offer a security audit",
             "citations": [9], "out_of_range": [9],
             "why_now": "", "effort": "high"},
        ],
        "risks": [{"claim": "migration risk", "citations": [],
                   "out_of_range": []}],
        "next_step": {"claim": "prototype it", "citations": [3],
                      "out_of_range": []},
    } if structured else {"structured": False, "free_text": "raw [9]"}


def test_collect_claims_numbers_and_labels():
    claims = collect_claims(_customer(), _strategy(), n_items=5)
    # 2 customer + 2 opportunities + 1 risk + 1 next step = 6 claims.
    assert [c["id"] for c in claims] == [1, 2, 3, 4, 5, 6]
    assert claims[0]["section"] == "customer: pain points"
    assert claims[1]["section"] == "customer: switching themes"
    assert claims[2]["section"] == "strategy: opportunity"
    assert claims[3]["section"] == "strategy: opportunity"
    assert claims[4]["section"] == "strategy: risk"
    assert claims[5]["section"] == "strategy: next step"
    # Opportunity title is folded into the audited claim text.
    assert "Payout dashboard" in claims[2]["claim"]
    # Out-of-range citations are carried for the audit to flag.
    assert claims[3]["out_of_range"] == [9]


def test_collect_claims_skips_unstructured_sections():
    claims = collect_claims(_customer(structured=False), _strategy(), 5)
    # Customer contributed nothing; ids still restart at 1.
    assert [c["id"] for c in claims] == [1, 2, 3, 4]
    assert claims[0]["section"] == "strategy: opportunity"


def test_audit_claims_used_out_of_range_uncited():
    claims = [
        {"id": 1, "citations": [1, 3], "out_of_range": []},
        {"id": 2, "citations": [], "out_of_range": [9]},
    ]
    a = audit_claims(claims, n_items=5)
    assert a == {"used": [1, 3], "out_of_range": [9], "uncited_ids": [2]}


def test_audit_citations_in_range_and_out():
    a = audit_citations("fees hurt devs [1] and [3], plus [9] and [0]",
                        n_items=5)
    assert a == {"used": [1, 3], "out_of_range": [0, 9]}


def test_audit_citations_ignores_literal_n_placeholder():
    # The model once emitted the literal "[n]" — that must never count
    # as a citation (and the draft gets flagged as having none).
    a = audit_citations("merchants complain about fees [n] and [n]", n_items=5)
    assert a == {"used": [], "out_of_range": []}


def test_count_verdicts_counts_typed_verdicts_only():
    critic = {"structured": True, "verdicts": [
        {"claim_id": 1, "verdict": "SUPPORTED", "note": ""},
        {"claim_id": 2, "verdict": "SUPPORTED", "note": ""},
        {"claim_id": 3, "verdict": "PARTIAL", "note": ""},
        {"claim_id": 4, "verdict": "unsupported", "note": ""},  # case-normalized
        {"claim_id": 5, "verdict": "MAYBE", "note": ""},  # invalid -> ignored
    ]}
    assert count_verdicts(critic) == {
        "SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 1,
    }


# --------------------------------------------------------------------------
# Report rendering from typed sections
# --------------------------------------------------------------------------

def _summary(**overrides):
    s = {
        "brief": "fees and payouts",
        "competitors": COMPETITORS,
        "provider": "ollama",
        "model": "llama3.2",
        "queries": ['"shopify"', "shopify fees"],
        "evidence": 3,
        "per_competitor": {"shopify": 3, "woocommerce": 0, "bigcommerce": 0},
        "verdicts": {"SUPPORTED": 1, "PARTIAL": 0, "UNSUPPORTED": 1},
        "citation_audit": {"used": [1, 3], "out_of_range": [],
                           "uncited_ids": []},
        "unstructured_sections": [],
        "evidence_items": [
            {"n": 1, "competitor": "shopify", "title": "Fees post",
             "url": "https://example.com/1"},
            {"n": 2, "competitor": "shopify", "title": "Payouts post",
             "url": "https://example.com/2"},
            {"n": 3, "competitor": "shopify", "title": "Suitcase post",
             "url": "https://example.com/3"},
        ],
        "claims": [
            {"id": 1, "section": "customer: pain points",
             "claim": "developers complain about payout cuts",
             "citations": [3], "out_of_range": []},
            {"id": 2, "section": "strategy: opportunity",
             "claim": "Payout dashboard — add payout transparency",
             "citations": [1], "out_of_range": []},
            {"id": 3, "section": "strategy: risk",
             "claim": "migration risk", "citations": [], "out_of_range": []},
        ],
        "sections": {
            "competitor": [
                {"structured": True, "competitor": "shopify",
                 "positioning": {"claim": "positioning claim",
                                 "citations": [1], "out_of_range": []},
                 "strengths": [{"claim": "strong ecosystem", "citations": [1],
                                "out_of_range": []}],
                 "weaknesses": [{"claim": "payout cuts", "citations": [3],
                                 "out_of_range": []}],
                 "recent_changes": []},
                {"structured": True, "competitor": "woocommerce",
                 "positioning": {"claim": "insufficient evidence",
                                 "citations": [], "out_of_range": []},
                 "strengths": [], "weaknesses": [], "recent_changes": []},
                {"structured": True, "competitor": "bigcommerce",
                 "positioning": {"claim": "insufficient evidence",
                                 "citations": [], "out_of_range": []},
                 "strengths": [], "weaknesses": [], "recent_changes": []},
            ],
            "customer": {
                "structured": True,
                "pain_points": [
                    {"claim": "developers complain about payout cuts",
                     "citations": [3], "out_of_range": []}
                ],
                "praises": [],
                "switching_themes": [],
            },
            "strategy": {
                "structured": True,
                "opportunities": [
                    {"title": "Payout dashboard",
                     "claim": "add payout transparency", "citations": [1],
                     "out_of_range": [], "why_now": "recurring theme",
                     "effort": "medium"}
                ],
                "risks": [{"claim": "migration risk", "citations": [],
                           "out_of_range": []}],
                "next_step": {"claim": "prototype the dashboard",
                              "citations": [1], "out_of_range": []},
            },
            "critic": {
                "structured": True,
                "verdicts": [
                    {"claim_id": 1, "verdict": "SUPPORTED",
                     "note": "matches the sarcastic complaint"},
                    {"claim_id": 2, "verdict": "UNSUPPORTED",
                     "note": "evidence does not say that"},
                ],
            },
        },
        "generated_at": "2026-09-08T14:52:28.123456+00:00",
        "report_path": None,
    }
    s.update(overrides)
    return s


def test_render_report_structure_and_audit_lines():
    md = render_report(_summary())

    for expected in [
        "# Market Intelligence Report",
        "**Brief:** fees and payouts",
        "## 1. Research scope",
        "- `\"shopify\"`",
        "## 2. Competitor analysis",
        "### Shopify",
        "### WooCommerce",
        "- **Positioning & pricing** — positioning claim [1]",
        "- **Strengths**",
        "  - strong ecosystem [1]",
        "  - payout cuts [3]",
        "## 3. Customer & UX analysis",
        "- **Pain points & complaints**",
        "  - developers complain about payout cuts [3]",
        "## 4. Strategy & opportunities",
        "- **Opportunities**",
        "  - **Payout dashboard** — add payout transparency [1]",
        "    - why now: recurring theme",
        "    - effort: medium",
        "- **Risks**",
        "  - migration risk [—]",
        "- **Recommended next step** — prototype the dashboard [1]",
        "## 5. Fact-check (critic)",
        "- **1. [customer: pain points]** developers complain about payout cuts — cites [3]",
        "  - verdict: **SUPPORTED** — matches the sarcastic complaint",
        "  - verdict: **UNSUPPORTED** — evidence does not say that",
        "- **3. [strategy: risk]** migration risk — cites [—]",
        "  - verdict: ⚠ not audited by the critic",
        "### Mechanical citation audit",
        "- 2 distinct in-range citations used",
        "- critic verdicts (typed, code-counted): 1 supported, 0 partial, 1 unsupported; 1 not audited: [3]",
        "## Sources",
        "1. [shopify] Fees post — https://example.com/1",
        "2. [shopify] Payouts post — https://example.com/2",
    ]:
        assert expected in md, f"missing from report: {expected!r}"


def test_render_report_warns_on_out_of_range_and_uncited():
    s = _summary()
    s["citation_audit"] = {"used": [1], "out_of_range": [9],
                           "uncited_ids": [3], "uncited_competitor": 4}
    md = render_report(s)
    assert "1 out-of-range citation(s): [9]" in md
    assert "⚠ claims without any citation: [3]" in md
    assert "⚠ 4 competitor-section claim(s) have no citations" in md


def test_render_report_warns_when_no_citations():
    s = _summary()
    s["citation_audit"] = {"used": [], "out_of_range": [], "uncited_ids": [3]}
    md = render_report(s)
    assert "⚠ the draft contains NO valid [n] citations" in md


def test_render_report_renders_unstructured_fallback_with_warning():
    s = _summary()
    s["unstructured_sections"] = ["Shopify"]
    s["sections"]["competitor"][0] = {
        "structured": False, "competitor": "shopify",
        "free_text": "**Positioning** raw prose [1]",
    }
    md = render_report(s)
    assert "### Shopify (unstructured)" in md
    assert "**Positioning** raw prose [1]" in md
    assert "⚠ Shopify was NOT structured JSON — rendered as free text, claims unverified" in md