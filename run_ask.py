"""Phase 3: evidence-backed Q&A over the competitor corpus.

Retrieves the most relevant chunks from the vector index, asks an LLM
to answer from that evidence alone, and prints the sources it cited.

Usage:
    python run_ask.py "Is Shopify raising fees for app developers?"
    python run_ask.py "Why do people leave shopify?" --top 5 --competitor shopify
    python run_ask.py            # interactive REPL

Provider auto-detection: OPENAI_API_KEY if set, else a local Ollama
server. Override with --provider or the LLM_PROVIDER env var.
"""
import argparse
import sys

from market_intel.answer import answer_question
from market_intel.llm import LLMError, model_for, resolve_provider
from market_intel.store import connect

# Windows consoles default to a legacy code page; force UTF-8 so
# Unicode titles/bodies survive display.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = "data/market_intel.db"


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")

    sources = result["sources"]
    if not sources:
        return
    cited = set(result["cited"])
    print("Sources:")
    for i, r in enumerate(sources, 1):
        mark = "[cited] " if i in cited else "         "
        print(f"  {mark}{i}. {r['title']} — {r['url']} (score {r['score']:.2f})")


def main(question: str | None, top_k: int, competitor: str | None,
         provider: str | None) -> None:
    # Resolve the provider up front so setup problems surface before
    # the (slow) retrieval step.
    try:
        provider = provider or resolve_provider()
    except LLMError as exc:
        print(f"error: {exc}")
        sys.exit(1)
    print(f"provider: {provider} / {model_for(provider)}")

    conn = connect(DB_PATH)

    if question:
        print(f"\nq: {question}" + (f"  (competitor={competitor})" if competitor else ""))
        print_result(answer_question(
            conn, question, top_k=top_k, competitor=competitor, provider=provider
        ))
        conn.close()
        return

    if not sys.stdin.isatty():
        print("no question — pass one as an argument or run interactively")
        sys.exit(1)

    print("evidence-backed Q&A. Ctrl+C / Ctrl+D to exit.\n")
    try:
        while True:
            q = input("question> ").strip()
            if not q:
                continue
            try:
                print_result(answer_question(
                    conn, q, top_k=top_k, competitor=competitor, provider=provider
                ))
                print()
            except LLMError as exc:
                print(f"error: {exc}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=None,
                        help="question to answer (omit for interactive mode)")
    parser.add_argument("--top", type=int, default=5,
                        help="evidence chunks to retrieve (default: 5)")
    parser.add_argument("--competitor", default=None,
                        help="restrict evidence to one competitor (e.g. shopify)")
    parser.add_argument("--provider", default=None,
                        help="openai | ollama | custom (default: auto-detect)")
    args = parser.parse_args()
    main(args.question, args.top, args.competitor, args.provider)