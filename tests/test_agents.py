"""Unit tests for the Phase 4 agent plumbing (no LLM, no model).

The agents themselves are LLM calls, but the code around them — query
parsing, citation auditing, verdict counting, report rendering — is
pure and testable. These tests pin that plumbing so prompt or layout
changes can't silently break it.

Run with:
    python -m pytest tests/test_agents.py -v
"""

from market_intel.agents import (
    audit_citations,
    count_verdicts,
    parse_research_queries,
    render_report,
)

COMPETITORS = ["shopify", "woocommerce", "bigcommerce"]


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


def test_audit_citations_in_range_and_out():
    a = audit_citations("fees hurt devs [1] and [3], plus [9] and [0]",
                        n_items=5)
    assert a == {"used": [1, 3], "out_of_range": [0, 9]}


def test_audit_citations_ignores_literal_n_placeholder():
    # The model once emitted the literal "[n]" — that must never count
    # as a citation (and the draft gets flagged as having none).
    a = audit_citations("merchants complain about fees [n] and [n]", n_items=5)
    assert a == {"used": [], "out_of_range": []}


def test_count_verdicts_counts_only_bracketed_verdicts():
    critic = (
        "[SUPPORTED] claim one\n"
        "[SUPPORTED] claim two\n"
        "[PARTIAL] claim three\n"
        "[UNSUPPORTED] claim four\n"
        "[unsupported] claim five\n"
        "**Verdict summary: 2 supported, 1 partial, 2 unsupported.**\n"
    )
    # The summary line must NOT double-count (it has no brackets).
    assert count_verdicts(critic) == {
        "SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 2,
    }


def test_render_report_structure_and_audit_lines():
    summary = {
        "brief": "fees and payouts",
        "competitors": COMPETITORS,
        "provider": "ollama",
        "model": "llama3.2",
        "queries": ['"shopify"', "shopify fees"],
        "evidence": 3,
        "per_competitor": {"shopify": 3, "woocommerce": 0, "bigcommerce": 0},
        "verdicts": {"SUPPORTED": 1, "PARTIAL": 0, "UNSUPPORTED": 1},
        "citation_audit": {"used": [1, 3], "out_of_range": []},
        "evidence_items": [
            {"n": 1, "competitor": "shopify", "title": "Fees post",
             "url": "https://example.com/1"},
            {"n": 2, "competitor": "shopify", "title": "Payouts post",
             "url": "https://example.com/2"},
        ],
        "sections": {
            "competitor": [
                {"competitor": "shopify", "text": "**Positioning** bullet [1]"},
                {"competitor": "woocommerce", "text": "insufficient evidence"},
                {"competitor": "bigcommerce", "text": "insufficient evidence"},
            ],
            "customer": "complaints [1]",
            "strategy": "opportunity [1]",
            "critic": "[SUPPORTED] x",
        },
        "generated_at": "2026-09-08T14:52:28.123456+00:00",
        "report_path": None,
    }
    md = render_report(summary)

    for expected in [
        "# Market Intelligence Report",
        "**Brief:** fees and payouts",
        "## 1. Research scope",
        "- `\"shopify\"`",
        "## 2. Competitor analysis",
        "### Shopify",
        "### WooCommerce",
        "## 3. Customer & UX analysis",
        "## 4. Strategy & opportunities",
        "## 5. Fact-check (critic)",
        "### Mechanical citation audit",
        "- 2 distinct in-range citations used",
        "- critic verdicts by bracket count (code-counted, not the LLM's summary line): 1 supported, 0 partial, 1 unsupported",
        "## Sources",
        "1. [shopify] Fees post — https://example.com/1",
        "2. [shopify] Payouts post — https://example.com/2",
    ]:
        assert expected in md, f"missing from report: {expected!r}"


def test_render_report_warns_when_no_citations():
    summary = {
        "brief": "b", "competitors": ["shopify"],
        "provider": "ollama", "model": "llama3.2",
        "queries": ['"shopify"'], "evidence": 1,
        "per_competitor": {"shopify": 1},
        "verdicts": {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 1},
        "citation_audit": {"used": [], "out_of_range": []},
        "evidence_items": [
            {"n": 1, "competitor": "shopify", "title": "T",
             "url": "https://example.com"}
        ],
        "sections": {
            "competitor": [{"competitor": "shopify", "text": "x"}],
            "customer": "no citations here [n]",
            "strategy": "also none",
            "critic": "nothing",
        },
        "generated_at": "2026-09-08T00:00:00+00:00",
        "report_path": None,
    }
    md = render_report(summary)
    assert "⚠ the draft contains NO valid [n] citations" in md