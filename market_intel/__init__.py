"""AI Market Intelligence & Product Strategy Agent.

Phase 1: data pipeline — fetch, clean, store competitor discussions.
Phase 2: RAG foundation — chunk the posts, embed them locally, and
search them with cosine similarity over SQLite.
Phase 3: grounded Q&A — retrieve evidence, answer from it with an LLM,
and cite the source posts.
Phase 4: multi-agent layer — research, competitor, customer, strategy,
and critic agents produce evidence-backed market reports with typed
JSON sections (claims + citations + verdicts).
Phase 5: n8n orchestration — scheduled workflows drive the API, poll the
jobs they start, and notify Slack (n8n/).
Phase 6: service — FastAPI over the same code (search, ask, report and
refresh jobs, evaluation), containerised, configured from the
environment.

Evaluation runs through all of it: labeled questions score retrieval and
answer citation quality (run_eval.py), and gate CI on every push.
"""

__version__ = "1.0.0"