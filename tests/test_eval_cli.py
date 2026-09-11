"""Unit tests for the eval CLI's gate + CI reporting plumbing.

The metric math lives in market_intel/eval.py and has its own tests; this
file covers what run_eval.py itself owns — deciding whether retrieval
passes the quality gate, and publishing the report to the GitHub Actions
job summary. No DB, no LLM, no embeddings.

Run with:
    python -m pytest tests/test_eval_cli.py -v
"""

import run_eval


# --- quality gate -----------------------------------------------------------

def test_gate_passes_above_thresholds():
    gate = run_eval.gate_result(5, avg_mrr=0.9, avg_recall=0.8,
                                min_mrr=0.8, min_recall=0.7)
    assert gate["passed"] is True
    assert gate["failures"] == []
    # Both thresholds become checks, reported under their own labels.
    assert [c["label"] for c in gate["checks"]] == ["MRR", "recall@5"]


def test_gate_fails_only_the_threshold_that_was_missed():
    gate = run_eval.gate_result(5, avg_mrr=0.9, avg_recall=0.6,
                                min_mrr=0.8, min_recall=0.7)
    assert gate["passed"] is False
    assert [c["label"] for c in gate["failures"]] == ["recall@5"]


def test_gate_uses_the_configured_k_in_the_label():
    gate = run_eval.gate_result(8, 0.0, 0.0, None, 0.7)
    assert [c["label"] for c in gate["checks"]] == ["recall@8"]


def test_gate_is_a_noop_without_thresholds():
    # Retrieval-only runs without gates must not "fail" — there is nothing
    # to compare against, and CI relies on this remaining a pass.
    gate = run_eval.gate_result(5, avg_mrr=0.0, avg_recall=0.0, min_mrr=None,
                                min_recall=None)
    assert gate["checks"] == []
    assert gate["passed"] is True


def test_gate_threshold_is_inclusive():
    # Exactly on the threshold counts as passing (matches the >= in render).
    assert run_eval.gate_result(5, 0.8, 0.7, 0.8, 0.7)["passed"] is True


# --- job summary rendering --------------------------------------------------

def test_render_gate_reports_each_check_and_marks_the_outcome():
    gate = run_eval.gate_result(5, avg_mrr=0.9, avg_recall=0.6,
                                min_mrr=0.8, min_recall=0.7)
    md = run_eval.render_gate(gate)
    assert "### CI gate" in md
    assert "- `MRR`: 0.900 (threshold 0.800) — **PASS**" in md
    assert "- `recall@5`: 0.600 (threshold 0.700) — **FAIL**" in md


def test_render_gate_explains_a_run_without_thresholds():
    md = run_eval.render_gate(run_eval.gate_result(5, 0.5, 0.5, None, None))
    assert "No thresholds requested" in md


# --- step summary writer ----------------------------------------------------

def test_write_step_summary_appends_to_the_env_path(tmp_path, monkeypatch):
    target = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))

    assert run_eval.write_step_summary("# report") == str(target)
    assert target.read_text(encoding="utf-8") == "# report\n"

    # A second write appends (the runner may have its own content), and a
    # newline is added when the caller's markdown lacks one.
    run_eval.write_step_summary("# gate\n")
    assert target.read_text(encoding="utf-8") == "# report\n# gate\n"


def test_write_step_summary_is_a_noop_outside_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert run_eval.write_step_summary("# report\n") is None
