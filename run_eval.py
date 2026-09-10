"""Evaluation harness (Phase 6 preview): score the RAG pipeline.

Scores two layers against hand-labeled questions in
tests/fixtures/eval_questions.json (each labels which posts are relevant):

  retrieval  — precision@k, recall@k, MRR, nDCG@k of the vector index
  answer     — citation precision/recall of LLM answers (--with-answers):
               of the posts an answer cites, how many are relevant, and
               how many relevant posts got cited at all

Usage:
    python run_eval.py                       # retrieval metrics only
    python run_eval.py --with-answers        # + LLM answers and citation scores
    python run_eval.py --with-answers --limit 4
    python run_eval.py --top 8 --out data/reports/eval_report.md

The report lands in data/reports/ (or --out). Retrieval-only runs need
no LLM at all, so they are fast and CI-safe.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from market_intel import config
from market_intel.answer import answer_question, dedupe_by_post
from market_intel.eval import (
    citation_precision,
    citation_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from market_intel.llm import LLMError, model_for, resolve_provider
from market_intel.store import connect
from market_intel.vector import search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = "data/market_intel.db"
DEFAULT_QUESTIONS = "tests/fixtures/eval_questions.json"


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise SystemExit(f"{path}: expected a JSON list of questions")
    return questions


def retrieve(conn, question: dict, top_k: int) -> list[str]:
    """Top-k distinct source posts for a question (post_ids, best first)."""
    results = dedupe_by_post(
        search(conn, question["question"], top_k=top_k * 3,
               competitor=question.get("competitor"))
    )[:top_k]
    return [r["post_id"] for r in results]


def score_retrieval(conn, question: dict, top_k: int) -> dict:
    retrieved = retrieve(conn, question, top_k)
    relevant = question["relevant"]
    return {
        "question": question["question"],
        "competitor": question.get("competitor"),
        "retrieved": retrieved,
        "relevant": relevant,
        "precision": precision_at_k(retrieved, relevant, top_k),
        "recall": recall_at_k(retrieved, relevant, top_k),
        "mrr": mrr(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevant, top_k),
    }


def score_answer(conn, row: dict, top_k: int, provider: str) -> dict:
    q = row["question"]
    ans = answer_question(conn, q, top_k=top_k,
                          competitor=row.get("competitor"), provider=provider)
    cited_posts = [ans["sources"][i - 1]["post_id"] for i in ans["cited"]]
    relevant = row["relevant"]
    row["answer"] = ans["answer"]
    row["cited_posts"] = cited_posts
    row["cit_precision"] = citation_precision(cited_posts, relevant)
    row["cit_recall"] = citation_recall(cited_posts, relevant)
    return row


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def render_report(rows: list[dict], top_k: int, provider: str | None,
                  model: str | None, corpus: int, generated_at: str) -> str:
    p = mean([r["precision"] for r in rows])
    rec = mean([r["recall"] for r in rows])
    m = mean([r["mrr"] for r in rows])
    n = mean([r["ndcg"] for r in rows])

    lines = [
        "# Evaluation Report — retrieval & answer quality",
        "",
        f"**Generated:** {generated_at[:19].replace('T', ' ')} UTC",
        f"**Corpus:** {corpus} posts in the index",
        f"**top_k:** {top_k}",
        f"**Labeled questions:** {len(rows)}",
    ]
    if provider:
        lines += [
            f"**Model:** {model} (provider: {provider})",
        ]
    lines += [
        "",
        "## Methodology",
        "",
        "- Each question carries hand-labeled relevant posts (by HN post "
        "id) in `tests/fixtures/eval_questions.json`.",
        "- Retrieval metrics score the vector index's top-k: precision@k "
        "(relevant hits / k), recall@k (relevant hits / all relevant), "
        "MRR (inverse rank of the first relevant hit), nDCG@k "
        "(rank-weighted gain vs the ideal ordering).",
        "- Answer metrics (`--with-answers`) score the citations of the "
        "LLM answer: citation precision (of the posts cited, how many "
        "are relevant) and citation recall (of the relevant posts, how "
        "many got cited).",
        "- All metrics are 0.0 when nothing relevant was retrieved or "
        "cited — a conservative floor for the averages.",
        "",
        "## Retrieval metrics",
        "",
        "| # | question | competitor | P@{k} | R@{k} | MRR | nDCG@{k} |".format(k=top_k),
        "|---|----------|------------|------|------|-----|---------|",
    ]
    for i, r in enumerate(rows, 1):
        comp = r["competitor"] or "all"
        lines.append(
            f"| {i} | {r['question'][:70]}{'…' if len(r['question']) > 70 else ''} "
            f"| {comp} | {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['mrr']:.3f} | {r['ndcg']:.3f} |"
        )
    lines += [
        f"| **avg** | | | **{p:.3f}** | **{rec:.3f}** | **{m:.3f}** | **{n:.3f}** |",
        "",
    ]

    if any("cit_precision" in r for r in rows):
        cp = mean([r["cit_precision"] for r in rows if "cit_precision" in r])
        cr = mean([r["cit_recall"] for r in rows if "cit_recall" in r])
        lines += [
            "## Answer quality (citations)",
            "",
            "| # | question | cit-precision | cit-recall | cited posts |",
            "|---|----------|---------------|------------|-------------|",
        ]
        for i, r in enumerate(rows, 1):
            if "cit_precision" not in r:
                lines.append(f"| {i} | {r['question'][:60]} | — | — | (answers skipped) |")
                continue
            lines.append(
                f"| {i} | {r['question'][:60]}{'…' if len(r['question']) > 60 else ''} "
                f"| {r['cit_precision']:.3f} | {r['cit_recall']:.3f} "
                f"| {', '.join(p.rsplit('_', 1)[-1] for p in r['cited_posts'])[:60]} |"
            )
        lines += [
            f"| **avg** | | **{cp:.3f}** | **{cr:.3f}** | |",
            "",
            "## Answers",
            "",
        ]
        for r in rows:
            if "answer" not in r:
                continue
            lines += [
                f"### {r['question']}",
                "",
                r["answer"],
                "",
            ]
    lines += [
        "## Notes & honest limits",
        "",
        "- Labels are a small hand-made set (one person, quick judgment); "
        "they measure the index, not ground truth about the market.",
        "- The corpus is HN discussions only — labels reflect what HN "
        "covers, not all of e-commerce.",
        "- A small local model's answers are weaker than a frontier "
        "model's; rerun with `--provider openai` (or a bigger Ollama "
        "model) and compare the citation scores.",
        "- Retrieval-only runs (`python run_eval.py`) need no LLM and can "
        "run in CI on every push.",
        "",
    ]
    return "\n".join(lines)


def main(questions_path: str, top_k: int, with_answers: bool,
         provider: str | None, limit: int | None, out: str | None) -> None:
    questions = load_questions(questions_path)
    if limit:
        questions = questions[:limit]
        print(f"note: --limit {limit} — evaluating a subset\n")

    if with_answers:
        try:
            provider = provider or resolve_provider()
        except LLMError as exc:
            print(f"error: {exc}")
            sys.exit(1)
    model = model_for(provider) if provider else None

    conn = connect(DB_PATH)
    corpus = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    # Sanity-check the labels against the corpus (label drift warning).
    known = {r[0] for r in conn.execute("SELECT id FROM posts").fetchall()}
    missing = sorted(
        {p for q in questions for p in q["relevant"] if p not in known}
    )
    if missing:
        print(f"warning: {len(missing)} labeled post(s) not in the DB "
              f"(stale labels?): {missing[:5]}{'...' if len(missing) > 5 else ''}\n")

    rows = []
    for question in questions:
        row = score_retrieval(conn, question, top_k)
        if with_answers:
            row = score_answer(conn, row, top_k, provider)
        rows.append(row)
        print(
            f"[{len(rows)}/{len(questions)}] P@{top_k}={row['precision']:.3f} "
            f"R@{top_k}={row['recall']:.3f} MRR={row['mrr']:.3f} "
            f"nDCG={row['ndcg']:.3f}"
            + (f" citP={row['cit_precision']:.3f} citR={row['cit_recall']:.3f}"
               if with_answers else "")
            + f"  {row['question'][:60]}"
        )
    conn.close()

    generated_at = datetime.now(timezone.utc).isoformat()
    md = render_report(rows, top_k, provider, model, corpus, generated_at)

    report_dir = Path(config.REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else (
        report_dir / f"eval_report_{generated_at[:19].replace(':', '').replace('T', '_')}.md"
    )
    out_path.write_text(md, encoding="utf-8")

    print("\naverages:")
    print(f"  retrieval : P@{top_k}={mean([r['precision'] for r in rows]):.3f} "
          f"R@{top_k}={mean([r['recall'] for r in rows]):.3f} "
          f"MRR={mean([r['mrr'] for r in rows]):.3f} "
          f"nDCG@{top_k}={mean([r['ndcg'] for r in rows]):.3f}")
    if with_answers:
        print(f"  answers   : cit-precision={mean([r['cit_precision'] for r in rows]):.3f} "
              f"cit-recall={mean([r['cit_recall'] for r in rows]):.3f}")
    print(f"report     : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS,
                        help="labeled questions JSON (default: tests/fixtures/eval_questions.json)")
    parser.add_argument("--top", type=int, default=5,
                        help="k for retrieval and answer evidence (default: 5)")
    parser.add_argument("--with-answers", action="store_true",
                        help="also run LLM answers and score their citations")
    parser.add_argument("--provider", default=None,
                        help="openai | ollama | custom (default: auto-detect)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N questions (quick runs)")
    parser.add_argument("--out", default=None,
                        help="write the eval report to this path")
    args = parser.parse_args()
    main(args.questions, args.top, args.with_answers, args.provider,
         args.limit, args.out)