"""Phase 4: the multi-agent research layer — research, competitor analyst,
customer/UX analyst, strategy, critic — producing a market report.

Flow (all agents work from the vector index, never raw text):
    research    -> plan queries, gather numbered evidence
    competitor  -> per-competitor analysis (positioning, strengths, ...)
    customer    -> complaints, pain points, praise from the community
    strategy    -> ranked product opportunities with citations
    critic      -> fact-checks the draft against the evidence

Usage:
    python run_report.py "fees and developer payouts"
    python run_report.py "why merchants switch platforms" --competitors shopify woocommerce
    python run_report.py "checkout friction" --out my_report.md

The report is written to data/reports/ (or --out); a short summary is
printed. Provider auto-detection: OPENAI_API_KEY if set, else local
Ollama. Override with --provider or the LLM_PROVIDER env var.
"""
import argparse
import sys
from pathlib import Path

from market_intel import config
from market_intel.agents import build_report
from market_intel.llm import LLMError, model_for, resolve_provider
from market_intel.store import connect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = "data/market_intel.db"


def print_summary(s: dict, out_path: Path) -> None:
    print(f"\nbrief      : {s['brief']}")
    print(f"model      : {s['model']} (provider: {s['provider']})")
    print(f"evidence   : {s['evidence']} posts retrieved")
    for c in s["competitors"]:
        print(f"  {c:<12}: {s['per_competitor'][c]}")
    print("queries    :")
    for q in s["queries"]:
        print(f"  - {q}")
    verdicts = s["verdicts"]
    print(
        "critic     : "
        f"{verdicts['SUPPORTED']} supported, {verdicts['PARTIAL']} partial, "
        f"{verdicts['UNSUPPORTED']} unsupported"
    )
    audit = s["citation_audit"]
    print(
        "citations  : "
        f"{len(audit['used'])} in-range"
        + (f", {len(audit['out_of_range'])} OUT-OF-RANGE {audit['out_of_range']}"
           if audit["out_of_range"] else ", none out of range")
    )
    print(f"report     : {out_path}")


def main(brief: str, competitors: list[str], provider: str | None,
         per_query: int, out: str | None) -> None:
    try:
        provider = provider or resolve_provider()
    except LLMError as exc:
        print(f"error: {exc}")
        sys.exit(1)
    print(f"provider: {provider} / {model_for(provider)}\n")

    conn = connect(DB_PATH)

    print("research -> competitor x3 -> customer -> strategy -> critic")
    summary = build_report(conn, brief, competitors, provider,
                           per_query=per_query)
    conn.close()

    report_dir = Path(config.REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else (
        report_dir / f"market_report_{summary['generated_at'][:19].replace(':', '').replace('T', '_')}.md"
    )
    out_path.write_text(summary["markdown"], encoding="utf-8")
    summary["report_path"] = str(out_path)

    print_summary(summary, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brief", help="what to research, e.g. "
                                      "\"fees and developer payouts\"")
    parser.add_argument("--competitors", nargs="+", default=config.COMPETITORS,
                        help="competitors to analyze (default: all)")
    parser.add_argument("--top", type=int, default=4,
                        help="evidence chunks retrieved per query (default: 4)")
    parser.add_argument("--out", default=None,
                        help="write the report to this path instead of data/reports/")
    parser.add_argument("--provider", default=None,
                        help="openai | ollama | custom (default: auto-detect)")
    args = parser.parse_args()
    main(args.brief, args.competitors, args.provider, args.top, args.out)