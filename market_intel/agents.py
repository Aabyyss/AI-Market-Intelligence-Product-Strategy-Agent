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

The analysts and the critic reply in **structured JSON** (claims with
citation numbers; verdicts keyed by claim id), so the report renders
from typed data — no parsing free-text markdown, no trusting the
model's arithmetic. The code around them is deliberately defensive:

  - every reply is parsed with a tolerant JSON extractor,
  - claims/citations/verdicts are validated and normalized,
  - a section whose reply is not usable JSON falls back to free text
    and is flagged in the report (the pipeline never hard-crashes on
    a bad model reply),
  - citations are mechanically audited afterwards — the same layered
    defense as Phase 3, now at report scale.
"""

import json
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
    """Render evidence items as the numbered list prompts see.

    Uses each item's GLOBAL number (assigned by ``gather_evidence``), so
    every prompt — including per-competitor sub-lists — speaks the same
    numbering the report's Sources section uses. Renumbering per
    sub-list made the model cite numbers outside its visible list, which
    the mechanical audit then dropped.
    """
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
# Tolerant JSON parsing + normalization (the defensive layer)
# --------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} object out of a model reply.

    Model replies are unreliable: markdown fences, preamble sentences,
    trailing commentary. This scans for the first '{' and tracks braces
    (skipping quoted strings) until the object closes, then parses it.
    Returns None when no usable JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
                    return obj if isinstance(obj, dict) else None
    return None


def _norm_citations(raw, n_items: int) -> tuple[list[int], list[int]]:
    """Coerce a citations field to (in-range, out-of-range) int lists.

    Models write citations as [1, 3] or ["1","3"] or even "1,3" — all
    are accepted; anything non-numeric is dropped. ``n_items`` is the
    number of evidence items, so out-of-range citations are surfaced
    (the report warns about them) instead of silently discarded.
    """
    if not isinstance(raw, list):
        return [], []
    in_range, out = [], []
    for c in raw:
        if isinstance(c, bool):
            continue
        if isinstance(c, int):
            n = c
        elif isinstance(c, str) and c.strip().isdigit():
            n = int(c)
        else:
            continue
        (in_range if 1 <= n <= n_items else out).append(n)
    return sorted(set(in_range)), sorted(set(out))


def _norm_claim_list(raw, n_items: int) -> list[dict]:
    """Normalize a list of {claim, citations} items from the model.

    Drops items without a claim text; keeps the citation split
    (in-range vs out-of-range) so the audit can warn about the latter.
    """
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        cites, oor = _norm_citations(item.get("citations"), n_items)
        out.append({"claim": claim, "citations": cites, "out_of_range": oor})
    return out


def _norm_effort(raw) -> str:
    v = str(raw or "").strip().lower()
    return v if v in ("low", "medium", "high") else "?"


def _norm_verdict(raw) -> str | None:
    v = str(raw or "").strip().upper()
    return v if v in ("SUPPORTED", "PARTIAL", "UNSUPPORTED") else None


def _has_content(*parts) -> bool:
    """True if any of the parts contains at least one real claim."""
    for part in parts:
        if isinstance(part, list) and part:
            return True
        if isinstance(part, dict) and part.get("claim"):
            return True
    return False


# --------------------------------------------------------------------------
# Analyst agents: one focused LLM call each, replying in JSON
# --------------------------------------------------------------------------

COMPETITOR_SYSTEM_PROMPT = """\
You are the competitor analyst in a market-intelligence pipeline tracking
e-commerce platforms (Shopify, WooCommerce, BigCommerce) through public
discussions. Analyze ONE competitor from the numbered evidence ([n]).

Reply with ONLY a JSON object of this exact shape (no markdown, no
commentary):

{
  "positioning": {"claim": "one sentence on how the community sees the product and price", "citations": [1]},
  "strengths": [{"claim": "what users praise", "citations": [2]}],
  "weaknesses": [{"claim": "a concrete gripe", "citations": [3]}],
  "recent_changes": [{"claim": "anything new, disputed, or under discussion", "citations": [4]}]
}

Rules:
- Every claim must cite REAL evidence numbers (e.g. [3]) from the list.
- Never invent facts. If the evidence is thin, say "insufficient
  evidence" in the claim instead of padding.
- Watch tone: a sarcastic or quoted statement in the evidence is NOT
  endorsement — say so when the community is being ironic.
- 3-6 items per list. Use "citations": [] only when a claim has no
  direct support (it will be flagged in the audit).
"""

CUSTOMER_SYSTEM_PROMPT = """\
You are the customer / UX analyst in a market-intelligence pipeline
tracking e-commerce platforms (Shopify, WooCommerce, BigCommerce)
through public discussions. Synthesize what MERCHANTS and DEVELOPERS
say across all the numbered evidence ([n]).

Reply with ONLY a JSON object of this exact shape (no markdown, no
commentary):

{
  "pain_points": [{"claim": "a recurring frustration", "citations": [1, 2]}],
  "praises": [{"claim": "what users genuinely like", "citations": [3]}],
  "switching_themes": [{"claim": "a friction point that pushes users toward or away from a platform", "citations": [4]}]
}

Rules:
- Every claim must cite REAL evidence numbers (e.g. [3]).
- Never invent facts.
- Distinguish a direct complaint from a sarcastic/quoted remark; do not
  report irony as praise or anger as a feature request. Sarcasm is still
  a signal — a sarcastic complaint is still a complaint.
- Group recurring themes; note where sources disagree. 3-6 items per
  list.
"""

STRATEGY_SYSTEM_PROMPT = """\
You are the strategy agent in a market-intelligence pipeline. Below are
competitor analyses and a customer analysis, all citing numbered
evidence ([n]) from public discussions. Synthesize them into product
opportunities for a team building e-commerce tooling.

Reply with ONLY a JSON object of this exact shape (no markdown, no
commentary):

{
  "opportunities": [
    {"title": "short title", "claim": "one sentence describing the opportunity",
     "citations": [1], "why_now": "the evidence-backed reason", "effort": "low"}
  ],
  "risks": [{"claim": "something that could go wrong", "citations": [2]}],
  "next_step": {"claim": "one concrete, evidence-backed action", "citations": [3]}
}

Rules:
- 3-5 opportunities, ranked by evidence strength and impact; effort is
  exactly "low", "medium", or "high".
- Only propose opportunities the cited evidence supports. Cite REAL
  evidence numbers (e.g. [3]).
- Flag contradictions in the evidence instead of smoothing them over.
"""

CRITIC_SYSTEM_PROMPT = """\
You are the critic / fact-checker in a market-intelligence pipeline.
Your only job is to catch hallucinations. You receive a list of numbered
claims (each with its cited evidence numbers) and the evidence list.

For EVERY claim, decide whether the cited evidence supports it and reply
with ONLY a JSON object of this exact shape (no markdown, no
commentary):

{
  "verdicts": [
    {"claim_id": 1, "verdict": "SUPPORTED", "note": "one short sentence"},
    {"claim_id": 2, "verdict": "PARTIAL", "note": "what is overstated"},
    {"claim_id": 3, "verdict": "UNSUPPORTED", "note": "why the cited evidence does not support it"}
  ]
}

Verdict rules:
- "SUPPORTED": the cited evidence says what the claim asserts.
- "PARTIAL": the evidence supports part of it, but something is
  overstated or only partially backed.
- "UNSUPPORTED": the evidence contradicts it, does not cover it, or the
  claim has no citation.
- IMPORTANT: sarcasm is still a signal — a sarcastic complaint is still
  a complaint. Report what the community MEANS, not the literal words.
- A claim that accurately reports what a source said is SUPPORTED even
  if the source itself is subjective or unverified: the claim is about
  what the community says, not about objective truth.
- Cover EVERY claim_id in the list. Be harsh — better to flag a
  borderline claim than to let it through.
"""


def _messages(system_prompt: str, content: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _chat_json(
    system_prompt: str, content: str, provider: str, max_tokens: int
) -> tuple[str, dict | None]:
    """One json_mode LLM call; retries once when the reply is not valid JSON.

    Small local models produce malformed JSON surprisingly often (a
    missing comma or bracket). One targeted retry usually fixes it; if
    the second attempt is still unusable, the caller falls back to free
    text rather than crash the pipeline.
    """
    reply = chat(_messages(system_prompt, content), provider=provider,
                 max_tokens=max_tokens, json_mode=True)
    data = extract_json(reply)
    if data is not None:
        return reply, data
    retry = chat(
        _messages(
            system_prompt,
            content + "\n\n(Your previous reply was not valid JSON. "
            "Reply with ONLY a valid JSON object of the required shape.)",
        ),
        provider=provider, max_tokens=max_tokens, json_mode=True,
    )
    return retry, extract_json(retry)


def run_competitor_analysis(
    items: list[dict], n_items: int, competitor: str, provider: str
) -> dict:
    """Competitor analyst: typed JSON section (or unstructured fallback).

    ``items`` is this competitor's evidence subset; ``n_items`` is the
    total evidence count, so citations are validated against the global
    numbering the report uses.
    """
    if not items:
        return {
            "structured": True,
            "competitor": competitor,
            "positioning": {
                "claim": "insufficient evidence — no retrieved posts matched.",
                "citations": [],
            },
            "strengths": [],
            "weaknesses": [],
            "recent_changes": [],
        }
    content = (
        f"Competitor: {competitor}\n\n"
        f"Evidence:\n{format_evidence(items)}\n\n"
        'Reply with the JSON object now, e.g. {"positioning": {...}, '
        '"strengths": [...], "weaknesses": [...], "recent_changes": [...]}.'
    )
    reply, data = _chat_json(COMPETITOR_SYSTEM_PROMPT, content, provider,
                             config.AGENT_MAX_TOKENS)
    n = n_items
    if data is None:
        return {"structured": False, "competitor": competitor,
                "free_text": reply}

    positioning = data.get("positioning")
    if isinstance(positioning, dict):
        claim = str(positioning.get("claim", "")).strip() or \
            "positioning summary missing"
        cites, oor = _norm_citations(positioning.get("citations"), n)
        positioning = {"claim": claim, "citations": cites,
                       "out_of_range": oor}
    else:
        positioning = {"claim": "", "citations": [], "out_of_range": []}
    strengths = _norm_claim_list(data.get("strengths"), n)
    weaknesses = _norm_claim_list(data.get("weaknesses"), n)
    recent = _norm_claim_list(data.get("recent_changes"), n)

    if not _has_content(positioning, strengths, weaknesses, recent):
        return {"structured": False, "competitor": competitor,
                "free_text": reply}
    return {
        "structured": True,
        "competitor": competitor,
        "positioning": positioning,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recent_changes": recent,
    }


def run_customer_analysis(items: list[dict], provider: str) -> dict:
    """Customer/UX analyst: typed JSON section (or unstructured fallback)."""
    if not items:
        return {"structured": True, "pain_points": [], "praises": [],
                "switching_themes": []}
    content = (
        f"Evidence:\n{format_evidence(items)}\n\n"
        'Reply with the JSON object now, e.g. {"pain_points": [...], '
        '"praises": [...], "switching_themes": [...]}.'
    )
    reply, data = _chat_json(CUSTOMER_SYSTEM_PROMPT, content, provider,
                             config.AGENT_MAX_TOKENS)
    n = len(items)
    if data is None:
        return {"structured": False, "free_text": reply}

    pain = _norm_claim_list(data.get("pain_points"), n)
    praises = _norm_claim_list(data.get("praises"), n)
    switching = _norm_claim_list(data.get("switching_themes"), n)

    if not _has_content(pain, praises, switching):
        return {"structured": False, "free_text": reply}
    return {"structured": True, "pain_points": pain, "praises": praises,
            "switching_themes": switching}


def run_strategy(
    items: list[dict],
    competitor_sections: list[dict],
    customer_section: dict,
    provider: str,
) -> dict:
    """Strategy agent: typed JSON section (or unstructured fallback)."""
    if not items:
        return {"structured": True, "opportunities": [], "risks": [],
                "next_step": {"claim": "", "citations": []}}
    content = (
        f"Competitor analyses:\n{_render_sections_for_prompt(competitor_sections)[:2000]}\n\n"
        f"Customer analysis:\n{_render_claims_for_prompt(customer_section)[:2000]}\n\n"
        f"Full evidence list:\n{format_evidence(items, max_chars=300)}\n\n"
        'Reply with the JSON object now, e.g. {"opportunities": [...], '
        '"risks": [...], "next_step": {...}}.'
    )
    reply, data = _chat_json(STRATEGY_SYSTEM_PROMPT, content, provider,
                             config.AGENT_MAX_TOKENS)
    n = len(items)
    if data is None:
        return {"structured": False, "free_text": reply}

    opportunities: list[dict] = []
    if isinstance(data.get("opportunities"), list):
        for o in data["opportunities"]:
            if not isinstance(o, dict):
                continue
            title = str(o.get("title", "")).strip()
            claim = str(o.get("claim", "")).strip()
            if not claim:
                continue
            cites, oor = _norm_citations(o.get("citations"), n)
            opportunities.append({
                "title": title or "(untitled opportunity)",
                "claim": claim,
                "citations": cites,
                "out_of_range": oor,
                "why_now": str(o.get("why_now", "")).strip(),
                "effort": _norm_effort(o.get("effort")),
            })
    risks = _norm_claim_list(data.get("risks"), n)
    nxt = data.get("next_step")
    next_step = {"claim": "", "citations": [], "out_of_range": []}
    if isinstance(nxt, dict):
        claim = str(nxt.get("claim", "")).strip()
        if claim:
            cites, oor = _norm_citations(nxt.get("citations"), n)
            next_step = {"claim": claim, "citations": cites,
                         "out_of_range": oor}

    if not _has_content(opportunities, risks, next_step):
        return {"structured": False, "free_text": reply}
    return {"structured": True, "opportunities": opportunities,
            "risks": risks, "next_step": next_step}


def run_critic(claims: list[dict], items: list[dict], provider: str) -> dict:
    """Critic: typed verdicts keyed by claim id (or unstructured fallback).

    ``claims`` is the numbered claim list (see ``collect_claims``); the
    critic audits exactly those claims and refers to them by claim_id,
    so verdicts stay aligned with the draft mechanically — no free-text
    matching, no trusting the model's arithmetic.
    """
    if not items:
        return {"structured": True, "verdicts": []}
    claims_json = json.dumps(
        [
            {"claim_id": c["id"], "section": c["section"],
             "claim": c["claim"], "citations": c["citations"]}
            for c in claims
        ],
        indent=1,
    )
    content = (
        f"Draft claims to audit:\n{claims_json}\n\n"
        f"Numbered evidence:\n{format_evidence(items, max_chars=300)}\n\n"
        'Reply with the JSON object now, e.g. {"verdicts": [{"claim_id": 1, '
        '"verdict": "SUPPORTED", "note": "..."}]}.'
    )
    reply, data = _chat_json(CRITIC_SYSTEM_PROMPT, content, provider,
                             config.CRITIC_MAX_TOKENS)
    if data is None:
        return {"structured": False, "verdicts": [], "free_text": reply}

    verdicts: list[dict] = []
    raw = data.get("verdicts")
    if isinstance(raw, list):
        for v in raw:
            if not isinstance(v, dict):
                continue
            cid = v.get("claim_id")
            if isinstance(cid, bool):
                continue
            if isinstance(cid, str):
                if cid.strip().isdigit():
                    cid = int(cid)
                else:
                    continue
            if not isinstance(cid, int):
                continue
            verdict = _norm_verdict(v.get("verdict"))
            if verdict is None:
                continue
            verdicts.append({
                "claim_id": cid,
                "verdict": verdict,
                "note": str(v.get("note", "")).strip(),
            })
    if not verdicts:
        return {"structured": False, "verdicts": [], "free_text": reply}
    return {"structured": True, "verdicts": verdicts}


# --------------------------------------------------------------------------
# Mechanical audits (code, not LLM — the last line of defense)
# --------------------------------------------------------------------------

CITATION_RE = re.compile(r"\[(\d+)\]")


def audit_citations(text: str, n_items: int) -> dict:
    """Every [n] in free text must point at real evidence (1..n_items).

    Used for unstructured (fallback) sections; typed claims go through
    ``audit_claims`` instead. The LLM layers judge *support*; this
    mechanical pass guarantees a citation can never reference evidence
    that was not retrieved.
    """
    used = {int(m) for m in CITATION_RE.findall(text)}
    return {
        "used": sorted(n for n in used if 1 <= n <= n_items),
        "out_of_range": sorted(n for n in used if not 1 <= n <= n_items),
    }


def collect_claims(customer: dict, strategy: dict, n_items: int) -> list[dict]:
    """Flatten typed analyst sections into numbered, auditable claims.

    Each claim gets a stable id used by the critic (claim_id) and by
    the report. Citations are validated against ``n_items`` evidence
    items, so out-of-range ones are surfaced here rather than silently
    rendered.
    """
    claims: list[dict] = []

    def add(section: str, c: dict) -> None:
        claims.append({
            "id": len(claims) + 1,
            "section": section,
            "claim": c["claim"],
            "citations": c["citations"],
            "out_of_range": c["out_of_range"],
        })

    if customer.get("structured"):
        for c in customer["pain_points"]:
            add("customer: pain points", c)
        for c in customer["praises"]:
            add("customer: praises", c)
        for c in customer["switching_themes"]:
            add("customer: switching themes", c)
    if strategy.get("structured"):
        for o in strategy["opportunities"]:
            claims.append({
                "id": len(claims) + 1,
                "section": "strategy: opportunity",
                "claim": f"{o['title']} — {o['claim']}",
                "citations": o["citations"],
                "out_of_range": o["out_of_range"],
            })
        for c in strategy["risks"]:
            add("strategy: risk", c)
        if strategy["next_step"]["claim"]:
            add("strategy: next step", strategy["next_step"])
    return claims


def audit_claims(claims: list[dict], n_items: int) -> dict:
    """Mechanical audit of typed claims: citations used, out-of-range, uncited."""
    used: set[int] = set()
    out: set[int] = set()
    uncited: list[int] = []
    for c in claims:
        used.update(c["citations"])
        out.update(c["out_of_range"])
        if not c["citations"]:
            uncited.append(c["id"])
    return {
        "used": sorted(used),
        "out_of_range": sorted(out),
        "uncited_ids": uncited,
    }


def count_verdicts(critic: dict) -> dict:
    """Count the critic's typed verdicts.

    Verdicts are typed data now (each with claim_id + verdict), so the
    count is exact — the old failure mode, where the model's prose
    summary line contradicted its own bracket lines, cannot happen.
    """
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0}
    for v in critic.get("verdicts", []):
        verdict = str(v["verdict"]).upper()
        if verdict in counts:
            counts[verdict] += 1
    return counts


# --------------------------------------------------------------------------
# The pipeline: research -> analysts -> strategy -> critic -> report
# --------------------------------------------------------------------------

def _render_sections_for_prompt(sections: list[dict]) -> str:
    """Compact text form of typed competitor sections, for LLM prompts."""
    parts = []
    for sec in sections:
        label = config.COMPETITOR_LABELS.get(sec["competitor"], sec["competitor"])
        if not sec.get("structured"):
            parts.append(f"### {label} (unstructured)\n{sec.get('free_text', '')}")
            continue
        parts.append(f"### {label}\n{render_claims_block(sec)}")
    return "\n\n".join(parts)


def _render_claims_for_prompt(section: dict) -> str:
    """Compact text form of a typed analyst section, for LLM prompts."""
    if not section.get("structured"):
        return section.get("free_text", "")
    return render_claims_block(section)


def _fmt_cites(citations: list[int]) -> str:
    """Render citation ints as [1, 3] — empty list renders as [—]."""
    if not citations:
        return "[—]"
    return "[" + ", ".join(str(c) for c in citations) + "]"


def render_claims_block(section: dict) -> str:
    """Render any typed analyst section dict as claim bullets with cites."""
    lines: list[str] = []

    def bullets(title: str, items: list[dict]) -> None:
        if items:
            lines.append(f"- **{title}**")
            for it in items:
                lines.append(
                    f"  - {it['claim']} {_fmt_cites(it['citations'])}"
                )

    if "positioning" in section:
        p = section["positioning"]
        lines.append(f"- **Positioning & pricing** — {p['claim']} {_fmt_cites(p['citations'])}")
        bullets("Strengths", section["strengths"])
        bullets("Weaknesses / complaints", section["weaknesses"])
        bullets("Recent changes", section["recent_changes"])
    elif "pain_points" in section:
        bullets("Pain points & complaints", section["pain_points"])
        bullets("Praises", section["praises"])
        bullets("UX & platform-switching themes", section["switching_themes"])
    return "\n".join(lines)


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
            run_competitor_analysis(sub, len(items), comp, provider)
        )

    customer_section = run_customer_analysis(items, provider)
    strategy_section = run_strategy(
        items, competitor_sections, customer_section, provider
    )

    # The critic fact-checks the typed claims from the customer and
    # strategy sections, keyed by mechanically assigned ids.
    claims = collect_claims(customer_section, strategy_section, len(items))
    critic_section = run_critic(claims, items, provider)

    citation_audit = audit_claims(claims, len(items))
    citation_audit["uncited_competitor"] = 0
    unstructured: list[str] = []
    for sec in competitor_sections + [customer_section, strategy_section,
                                      critic_section]:
        if not sec.get("structured"):
            unstructured.append(
                config.COMPETITOR_LABELS.get(sec.get("competitor", ""),
                                             sec.get("competitor", "section"))
                or "section"
            )
            # A free-text fallback still gets the regex citation audit.
            ft = audit_citations(sec.get("free_text", ""), len(items))
            citation_audit["used"] = sorted(
                set(citation_audit["used"]) | set(ft["used"])
            )
            citation_audit["out_of_range"] = sorted(
                set(citation_audit["out_of_range"]) | set(ft["out_of_range"])
            )

    # Structured competitor sections also contribute to the citation
    # audit (their claims are not critic-audited, but their citations
    # must still point at real evidence).
    for sec in competitor_sections:
        if not sec.get("structured"):
            continue
        parts: list[dict] = []
        for key in ("positioning", "strengths", "weaknesses",
                    "recent_changes"):
            part = sec.get(key)
            if isinstance(part, dict):
                parts.append(part)
            elif isinstance(part, list):
                parts.extend(p for p in part if isinstance(p, dict))
        for p in parts:
            if not p.get("claim"):
                continue
            if not p["citations"]:
                citation_audit["uncited_competitor"] += 1
            citation_audit["used"] = sorted(
                set(citation_audit["used"]) | set(p["citations"])
            )
            citation_audit["out_of_range"] = sorted(
                set(citation_audit["out_of_range"]) | set(p["out_of_range"])
            )

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
        "citation_audit": citation_audit,
        "unstructured_sections": unstructured,
        "evidence_items": items,
        "claims": claims,
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
    if audit["out_of_range"]:
        audit_lines.append(
            f"- {len(audit['out_of_range'])} out-of-range citation(s): "
            f"{audit['out_of_range']}"
        )
    if audit["uncited_ids"]:
        audit_lines.append(
            f"- ⚠ claims without any citation: {audit['uncited_ids']}"
        )
    if audit.get("uncited_competitor"):
        audit_lines.append(
            f"- ⚠ {audit['uncited_competitor']} competitor-section claim(s) "
            "have no citations — they are analyst assertions, not evidence"
        )
    for name in s.get("unstructured_sections", []):
        audit_lines.append(
            f"- ⚠ {name} was NOT structured JSON — rendered as free text, "
            "claims unverified"
        )

    # Verdict coverage: every claim should have a critic verdict.
    verdict_by_id = {
        v["claim_id"]: v
        for v in s["sections"]["critic"].get("verdicts", [])
    }
    unverified = [
        c["id"] for c in s["claims"]
        if c["id"] not in verdict_by_id
    ]
    v = s["verdicts"]
    audit_lines.append(
        "- critic verdicts (typed, code-counted): "
        f"{v['SUPPORTED']} supported, {v['PARTIAL']} partial, "
        f"{v['UNSUPPORTED']} unsupported"
        + (f"; {len(unverified)} not audited: {unverified}" if unverified else "")
    )

    per_comp = ", ".join(
        f"{config.COMPETITOR_LABELS.get(c, c)}: {s['per_competitor'][c]}"
        for c in s["competitors"]
    )

    # --- competitor sections ---
    comp_blocks = []
    for sec in s["sections"]["competitor"]:
        label = config.COMPETITOR_LABELS.get(sec["competitor"], sec["competitor"])
        if not sec.get("structured"):
            comp_blocks.append(f"### {label} (unstructured)\n\n{sec['free_text']}")
            continue
        comp_blocks.append(f"### {label}\n\n{render_claims_block(sec)}")

    # --- customer section ---
    cust = s["sections"]["customer"]
    customer_block = (
        render_claims_block(cust)
        if cust.get("structured")
        else cust.get("free_text", "")
    )

    # --- strategy section ---
    strat = s["sections"]["strategy"]
    if strat.get("structured"):
        strat_lines: list[str] = []
        strat_lines.append("- **Opportunities**")
        for o in strat["opportunities"]:
            strat_lines.append(
                f"  - **{o['title']}** — {o['claim']} {_fmt_cites(o['citations'])}"
            )
            if o["why_now"]:
                strat_lines.append(f"    - why now: {o['why_now']}")
            strat_lines.append(f"    - effort: {o['effort']}")
        if strat["risks"]:
            strat_lines.append("- **Risks**")
            for r in strat["risks"]:
                strat_lines.append(
                    f"  - {r['claim']} {_fmt_cites(r['citations'])}"
                )
        if strat["next_step"]["claim"]:
            ns = strat["next_step"]
            strat_lines.append(
                f"- **Recommended next step** — {ns['claim']} "
                f"{_fmt_cites(ns['citations'])}"
            )
        strategy_block = "\n".join(strat_lines)
    else:
        strategy_block = strat.get("free_text", "")

    # --- critic section ---
    critic = s["sections"]["critic"]
    if critic.get("structured"):
        critic_lines: list[str] = []
        for c in s["claims"]:
            verdict = verdict_by_id.get(c["id"])
            cites = _fmt_cites(c["citations"])
            if verdict:
                critic_lines.append(
                    f"- **{c['id']}. [{c['section']}]** {c['claim']} — cites "
                    f"{cites}\n"
                    f"  - verdict: **{verdict['verdict']}**"
                    + (f" — {verdict['note']}" if verdict["note"] else "")
                )
            else:
                critic_lines.append(
                    f"- **{c['id']}. [{c['section']}]** {c['claim']} — cites "
                    f"{cites}\n  - verdict: ⚠ not audited by the critic"
                )
        critic_block = "\n\n".join(critic_lines)
    else:
        critic_block = critic.get("free_text", "")

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
        "\n\n".join(comp_blocks),
        "",
        "## 3. Customer & UX analysis",
        "",
        customer_block,
        "",
        "## 4. Strategy & opportunities",
        "",
        strategy_block,
        "",
        "## 5. Fact-check (critic)",
        "",
        critic_block,
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