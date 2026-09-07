"""Phase 2b: semantic search over the indexed posts.

Usage:
    python run_search.py "checkout fees on shopify" --top 5
    python run_search.py "migrating away from shopify" --competitor shopify
    python run_search.py            # interactive REPL (one query per line)

Results show the cosine similarity score, the post title, and the
source URL — the citation back to the discussion.
"""
import argparse
import sys

from market_intel.store import connect
from market_intel.vector import search

# Windows consoles default to a legacy code page (e.g. cp1252), which
# mangles non-ASCII text (smart quotes, em dashes) in titles/bodies.
# Reconfigure stdout to UTF-8 so the data survives display.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = "data/market_intel.db"


def print_results(results: list[dict]) -> None:
    if not results:
        print("  no matches (did you run `python run_index.py` first?)")
        return
    for i, r in enumerate(results, 1):
        print(f"\n{i}. score {r['score']:.3f}  [{r['competitor']}]")
        print(f"   {r['title']}")
        print(f"   {r['url']}")
        snippet = r["text"] if len(r["text"]) <= 400 else r["text"][:400] + "…"
        print(f"   > {snippet}")


def main(query: str | None, top_k: int, competitor: str | None) -> None:
    conn = connect(DB_PATH)

    if query:
        print(f"query: {query}" + (f"  (competitor={competitor})" if competitor else ""))
        print_results(search(conn, query, top_k=top_k, competitor=competitor))
        conn.close()
        return

    # No query given: interactive REPL (only makes sense on a terminal).
    if not sys.stdin.isatty():
        print("no query — pass --query or run interactively")
        sys.exit(1)
    print("semantic search over competitor discussions. Ctrl+C / Ctrl+D to exit.\n")
    try:
        while True:
            q = input("query> ").strip()
            if not q:
                continue
            print_results(search(conn, q, top_k=top_k, competitor=competitor))
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=None,
                        help="search query (omit for interactive mode)")
    parser.add_argument("--top", type=int, default=5, help="results to show")
    parser.add_argument("--competitor", default=None,
                        help="filter to one competitor (e.g. shopify)")
    args = parser.parse_args()
    main(args.query, args.top, args.competitor)