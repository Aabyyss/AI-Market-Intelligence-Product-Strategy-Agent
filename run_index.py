"""Phase 2a: build the vector index — chunk posts, embed, store in SQLite.

Usage:
    python run_index.py

Reads every post from data/market_intel.db, chunks it, embeds the
chunks with the local bge model, and (re)builds the ``chunks`` table.
"""
from pathlib import Path

from market_intel.store import connect
from market_intel.vector import build_index


def main() -> None:
    db_path = Path("data/market_intel.db")
    print("== Building vector index ==")
    conn = connect(str(db_path))
    build_index(conn)
    conn.close()


if __name__ == "__main__":
    main()