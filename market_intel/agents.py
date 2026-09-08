"""Multi-agent research layer (Phase 4) over the RAG knowledge base.

Five roles, one job each, all working from the vector index:

  research    — turns the brief into search queries, gathers numbered
                evidence from the index
  competitor  — analyzes one competitor: positioning/pricing, strengths,
                weaknesses, recent changes
  customer    — the community's complaints, pain points, and praise
  strategy    — ranked product opportunities grounded in the evidence
  critic      — fact-checks the draft against the evidence and flags
                unsupported claims and tone misreads (e.g. sarcasm)

Each agent is a focused LLM call with its own system prompt; the code
around them supplies the numbered evidence and mechanically audits the
citations afterwards — the same layered defense as Phase 3, now at
report scale. ``build_report`` assembles the whole pipeline and returns
a markdown report plus a console-friendly summary.
"""

import re
from datetime import datetime, timezone

from market_intel import config
from market_intel.answer import _clean as clean_text
from market_intel.llm import chat
from market_intel.vector import search

# --------------------------------------------------------------------------
# Research agent: plan queries, gather evidence
# --------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """\
You are the research agent in a market-intelligence pipeline that tracks
e-commerce platforms through public discussions. Your job is to plan
search queries for a vector index of those discussions.

Given the brief below, output ONLY search queries — one per line, no
numbering, no commentary. Every query MUST contain the brand name of
exactly one competitor and target an angle from the brief (fees,
migration, checkout, downtime, support, pricing, payouts, ...). Max
{max_per} queries per competitor.

Competitors: {competitors}
Brief: "{brief}"
"""


def parse_research_queries(
    text: str, competitors: list[str], max_per: int = 2
) -> dict[str, list[str]]:
    """Extract valid research queries from the LLM's reply.

    One query per line; only lines mentioning exactly one competitor
    are kept, capped at ``max_per`` per competitor. List markers and
    stray quotes are stripped (queries are embedded, so quotes are
    noise). Anything else is dropped — a messy model reply must never
    derail the pipeline.
    """
    out: dict[str, list[str]] = {c: [] for c in competitors}
    for line in text.splitlines():
        q = line.strip()
        q = re.sub(r"^[-*•]\s*", "", q)      # bullet markers
        q = re.sub(r"^\d+[.)]\s*", "", q)    # numbered list markers
        q = q.replace('"', "").strip()
        if len(q) < 3:
            continue
        matches = [c for c in competitors if c in q.lower()]
        if len(matches) == 1 and len(out[matches[0]]) < max_per:
            out[matches[0]].append(q)
    return out


def plan_queries(
    brief: str, competitors: list[str], provider: str
) -> list[str]:
    """Research agent: propose queries; guarantee a baseline per brand.

    The quoted brand name alone (e.g. `"shopify"`) is always included
    as a broad baseline — the corpus is already brand-filtered, so it
    can only help. LLM-proposed topical queries are added on top, and
    any competitor the model skipped gets just the baseline.
    """
    messages = [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT.format(
            max_per=config.QUERIES_PER_COMPETITOR,
            competitors=", ".join(competitors),
            brief=brief,
        )},
        {"role": "user", "content": "Output the search queries now."},
    ]
    reply = chat(messages, provider=provider,
                 max_tokens=config.RESEARCH_MAX_TOKENS)
    proposed = parse_research_queries(reply, competitors)

    queries: list[str] = []
    for comp in competitors:
        queries.append(f'"{comp}"')  # baseline
        queries.extend(proposed[comp])
    return queries


def gather_evidence(
    conn, queries: list[str], per_query: int = 4, max_per_competitor: int = 8
) -> list[dict]:
    """Run the research queries, dedupe by post, assign global numbers.

    Returns one list of evidence items across all competitors; each item
    carries a global ``n`` used as its [n] citation number everywhere in
    the report (analysts cite the same numbers the Sources section maps).
    """
    items: list[dict] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for q in queries:
        for r in search(conn, q, top_k=per_query):
            comp = r["competitor"]
            if r["post_id"] in seen or counts.get(comp, 0) >= max_per_competitor:
                continue
            seen.add(r["post_id"])
            counts[comp] = counts.get(comp, 0) + 1
            items.append(
                {
                    "competitor": comp,
                    "post_id": r["post_id"],
                    "title": r["title"],
                    "url": r["url"],
                    "text": r["text"],
                    "score": r["score"],
                }
            )
    for n, item in enumerate(items, 1):
        item["n"] = n
    return items


def format_evidence(items: list[dict], max_chars: int = 450) -> str:
    """Render evidence items as the numbered list prompts see."""
    blocks = []
    for it in items:
        excerpt = clean_text(it["text"])
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars] + "…"
        blocks.append(
            f"[{it['n']}] [{it['competitor']}] {it['title']}\n"
            f"    url: {it['url']}\n    {excerpt}"
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Analyst agents: one focused LLM call each
# --------------------------------------------------------------------------

COMPETITOR_SYSTEM_PROMPT = """\
You are the competitor analyst in a market-intelligence pipeline tracking
e-commerce platforms (Shopify, WooCommerce, BigCommerce) through public
discussions. Analyze ONE competitor from the numbered evidence ([n]).

Write a markdown section with these subsections:
- **Positioning & pricing** — how the community sees the product and price
- **Strengths** — what users praise, with evidence
- **Weaknesses / complaints** — concrete gripes, with evidence
- **Recent changes** — anything new, disputed, or under discussion

Rules:
- Base every claim on the evidence; cite inline with the evidence
  number, e.g. "merchants complain about checkout fees [3]". Always a
  REAL number from the evidence list — never the literal text "[n]".
- Never invent facts. If the evidence is thin, say "insufficient
  evidence" instead of padding.
- Watch tone: a sarcastic or quoted statement in the evidence is NOT
  endorsement — say so when the community is being ironic.
- Be concise: 3-6 bullets per subsection.
"""

CUSTOMER_SYSTEM_PROMPT = """\
You are the customer / UX analyst in a market-intelligence pipeline
tracking e-commerce platforms (Shopify, WooCommerce, BigCommerce)
through public discussions. Synthesize what MERCHANTS and DEVELOPERS
say across all the numbered evidence ([n]).

Write a markdown section:
- **Pain points & complaints** — the recurring frustrations, each cited
- **Praises** — what users genuinely like, each cited
- **UX & platform-switching themes** — friction points that push users
  toward or away from a platform, each cited

Rules:
- Every claim must cite the evidence number, e.g. "developers complain
  about payout cuts [3]". Always a REAL number from the evidence list
  — never the literal text "[n]".
- Never invent facts.
- Distinguish a direct complaint from a sarcastic/quoted remark; do not
  report irony as praise or anger as a feature request.
- Group recurring themes; note where sources disagree.
"""

STRATEGY_SYSTEM_PROMPT = """\
You are the strategy agent in a market-intelligence pipeline. Below are
competitor analyses and a customer analysis, all citing numbered
evidence ([n]) from public discussions. Synthesize them into product
opportunities for a team building e-commerce tooling.

Write a markdown section:
- **Opportunities** — 3-5, each as one bold title line followed by:
  - evidence: cite the number(s), e.g. [3] (a REAL number from the
    evidence list — never the literal text "[n]")
  - why now: the evidence-backed reason
  - effort: rough (low / medium / high)
  Rank them by evidence strength and impact.
- **Risks** — 2-4 things that could go wrong, cited where possible
- **Recommended next step** — one concrete, evidence-backed action

Rules:
- Only propose opportunities the cited evidence supports.
- Flag contradictions in the evidence instead of smoothing them over.
"""

CRITIC_SYSTEM_PROMPT = """\
You are the critic / fact-checker in a market-intelligence pipeline.
Your only job is to catch hallucinations. Review the draft report
against the numbered evidence ([n]).

For EVERY claim you can identify in the draft, output one line:
- [SUPPORTED] <short claim> — cites [3]
- [PARTIAL] <short claim> — cites [3] — what is overstated
- [UNSUPPORTED] <short claim> — why the cited evidence does not support it

(Use REAL evidence numbers like [3], never the literal text "[n]".)

Also:
- Any claim with NO citation is [UNSUPPORTED] by definition.
- Flag tone misreads: evidence that is sarcastic, quoted, or an
  opinion presented as fact. IMPORTANT: sarcasm is still a signal —
  a sarcastic complaint is still a complaint. Report what the
  community MEANS, not the literal words.
- Flag claims that cite evidence which actually contradicts them.
- A claim that accurately reports what a source said is SUPPORTED even
  if the source itself is subjective or unverified: the claim is about
  what the community says, not about objective truth. Only mark
  UNSUPPORTED when the cited evidence does not actually say what the
  claim asserts (or the claim has no citation).

End with a summary line:
**Verdict summary: X supported, Y partial, Z unsupported.**
Be harsh — better to flag a borderline claim than to let it through.
"""


def _messages(system_prompt: str, content: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def run_competitor_analysis(
    items: list[dict], competitor: str, provider: str
) -> str:
    if not items:
        return (
            f"**Insufficient evidence** for {competitor} — no retrieved "
            "posts matched. Consider fetching more discussions."
        )
    content = (
        f"Competitor: {competitor}\n\n"
        f"Evidence:\n{format_evidence(items)}\n\n"
        "Write the analysis, citing evidence by number, e.g. [3]."
    )
    return chat(_messages(COMPETITOR_SYSTEM_PROMPT, content),
                provider=provider, max_tokens=config.AGENT_MAX_TOKENS)


def run_customer_analysis(items: list[dict], provider: str) -> str:
    if not items:
        return "**Insufficient evidence** — the index returned no posts."
    content = (
        f"Evidence:\n{format_evidence(items)}\n\n"
        "Write the customer analysis, citing evidence by number, e.g. [3]."
    )
    return chat(_messages(CUSTOMER_SYSTEM_PROMPT, content),
                provider=provider, max_tokens=config.AGENT_MAX_TOKENS)


def run_strategy(
    items: list[dict],
    competitor_sections: str,
    customer_section: str,
    provider: str,
) -> str:
    if not items:
        return "**Insufficient evidence** — nothing to strategize from."
    content = (
        f"Competitor analyses:\n{competitor_sections[:2000]}\n\n"
        f"Customer analysis:\n{customer_section[:2000]}\n\n"
        f"Full evidence list:\n{format_evidence(items, max_chars=300)}\n\n"
        "Write the strategy section, citing evidence by number, e.g. [3]."
    )
    return chat(_messages(STRATEGY_SYSTEM_PROMPT, content),
                provider=provider, max_tokens=config.AGENT_MAX_TOKENS)


def run_critic(draft: str, items: list[dict], provider: str) -> str:
    if not items:
        return "**Verdict summary: 0 supported, 0 partial, 0 unsupported.**"
    content = (
        f"Draft report:\n{draft[:3000]}\n\n"
        f"Numbered evidence:\n{format_evidence(items, max_chars=300)}\n\n"
        "Output the fact-check lines and the verdict summary now."
    )
    return chat(_messages(CRITIC_SYSTEM_PROMPT, content),
                provider=provider, max_tokens=config.CRITIC_MAX_TOKENS)


# --------------------------------------------------------------------------
# Mechanical audits (code, not LLM — the last line of defense)
# --------------------------------------------------------------------------

CITATION_RE = re.compile(r"\[(\d+)\]")


def audit_citations(text: str, n_items: int) -> dict:
    """Every [n] in a draft must point at real evidence (1..n_items).

    The LLM layers judge *support*; this mechanical pass guarantees a
    citation can never reference evidence that was not retrieved — the
    same post-validation Phase 3 applies to single answers.
    """
    used = {int(m) for m in CITATION_RE.findall(text)}
    return {
        "used": sorted(n for n in used if 1 <= n <= n_items),
        "out_of_range": sorted(n for n in used if not 1 <= n <= n_items),
    }


VERDICT_RE = re.compile(r"\[(SUPPORTED|PARTIAL|UNSUPPORTED)\]")


def count_verdicts(critic_text: str) -> dict:
    """Count the critic's per-claim verdicts.

    Only the bracketed verdict form [SUPPORTED]/[PARTIAL]/[UNSUPPORTED]
    that the critic is instructed to emit counts — the summary line
    ("Verdict summary: X supported ...") must not double-count.
    """
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0}
    for v in VERDICT_RE.findall(critic_text.upper()):
        if v in counts:
            counts[v] += 1
    return counts


# --------------------------------------------------------------------------
# The pipeline: research -> analysts -> strategy -> critic -> report
# --------------------------------------------------------------------------

def build_report(
    conn,
    brief: str,
    competitors: list[str],
    provider: str,
    per_query: int = 4,
) -> dict:
    """Run all five agents and assemble the market report."""
    queries = plan_queries(brief, competitors, provider)
    items = gather_evidence(conn, queries, per_query=per_query)

    competitor_sections: list[dict] = []
    for comp in competitors:
        sub = [it for it in items if it["competitor"] == comp]
        competitor_sections.append(
            {
                "competitor": comp,
                "text": run_competitor_analysis(sub, comp, provider),
            }
        )
    competitor_text = "\n\n".join(
        f"### {sec['competitor'].title()}\n\n{sec['text']}"
        for sec in competitor_sections
    )

    customer_section = run_customer_analysis(items, provider)
    strategy_section = run_strategy(
        items, competitor_text, customer_section, provider
    )

    # The critic fact-checks the claim-bearing parts of the draft.
    draft = f"{customer_section}\n\n{strategy_section}"
    critic_section = run_critic(draft, items, provider)
    audit = audit_citations(draft, len(items))

    now = datetime.now(timezone.utc)
    from market_intel.llm import model_for

    summary = {
        "brief": brief,
        "competitors": competitors,
        "provider": provider,
        "model": model_for(provider),
        "queries": queries,
        "evidence": len(items),
        "per_competitor": {
            c: sum(1 for it in items if it["competitor"] == c)
            for c in competitors
        },
        "verdicts": count_verdicts(critic_section),
        "citation_audit": audit,
        "evidence_items": items,
        "sections": {
            "competitor": competitor_sections,
            "customer": customer_section,
            "strategy": strategy_section,
            "critic": critic_section,
        },
        "generated_at": now.isoformat(),
        "report_path": None,  # filled by the caller after writing
    }
    summary["markdown"] = render_report(summary)
    return summary


def render_report(s: dict) -> str:
    """Assemble the markdown report from a build_report summary.

    Pure function (no LLM, no DB): given the summary dict, produce the
    report. Kept separate from build_report so the layout is testable
    and a report can be re-rendered from cached sections.
    """
    audit = s["citation_audit"]
    audit_lines = [f"- {len(audit['used'])} distinct in-range citations used"]
    if not audit["used"]:
        audit_lines.append(
            "- ⚠ the draft contains NO valid [n] citations — claims are "
            "unverifiable and the critic cannot audit them"
        )
    elif audit["out_of_range"]:
        audit_lines.append(
            f"- {len(audit['out_of_range'])} out-of-range citation(s): "
            f"{audit['out_of_range']}"
        )
    else:
        audit_lines.append(
            "- no out-of-range citations — every [n] points at retrieved evidence"
        )
    v = s["verdicts"]
    audit_lines.append(
        "- critic verdicts by bracket count (code-counted, not the LLM's "
        f"summary line): {v['SUPPORTED']} supported, {v['PARTIAL']} partial, "
        f"{v['UNSUPPORTED']} unsupported"
    )

    per_comp = ", ".join(
        f"{config.COMPETITOR_LABELS.get(c, c)}: {s['per_competitor'][c]}"
        for c in s["competitors"]
    )
    lines = [
        "# Market Intelligence Report",
        "",
        f"**Brief:** {s['brief']}",
        f"**Generated:** {s['generated_at'][:19].replace('T', ' ')} UTC",
        f"**Model:** {s['model']} (provider: {s['provider']})",
        f"**Evidence:** {s['evidence']} posts retrieved ({per_comp})",
        "",
        "## 1. Research scope",
        "",
        "Queries run against the vector index:",
        "",
        *[f"- `{q}`" for q in s["queries"]],
        "",
        "## 2. Competitor analysis",
        "",
        "\n\n".join(
            f"### {config.COMPETITOR_LABELS.get(sec['competitor'], sec['competitor'].title())}"
            f"\n\n{sec['text']}"
            for sec in s["sections"]["competitor"]
        ),
        "",
        "## 3. Customer & UX analysis",
        "",
        s["sections"]["customer"],
        "",
        "## 4. Strategy & opportunities",
        "",
        s["sections"]["strategy"],
        "",
        "## 5. Fact-check (critic)",
        "",
        s["sections"]["critic"],
        "",
        "### Mechanical citation audit",
        "",
        *audit_lines,
        "",
        "## Sources",
        "",
        *[
            f"{it['n']}. [{it['competitor']}] {it['title']} — {it['url']}"
            for it in s["evidence_items"]
        ],
        "",
    ]
    return "\n".join(lines)