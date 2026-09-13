"""Pytest plugin: simulate a runner with no LLM provider reachable.

Why this exists
---------------
``/ask`` and ``/reports`` resolve the LLM provider *per request* by probing
localhost:11434 (and checking ``OPENAI_API_KEY``). A developer box with
Ollama running therefore takes a different code path than CI, which has
neither — so a test can pass locally and 503 on GitHub. That is exactly how
the first CI run failed (7 tests in tests/test_api.py), and it was invisible
without a way to reproduce the CI condition.

Loading this plugin hides every ambient provider: nothing is listening on
the LLM port and no key is set, so anything that resolves a provider for
real fails loudly instead of silently succeeding on your machine.

Usage
-----
    PYTHONPATH=tests python -m pytest tests/ -p ci_sim_plugin -q

It is never loaded implicitly, so a normal ``pytest tests/`` is unaffected.
"""

import os


def pytest_configure(config):
    # No key, and "auto" so the reachability probe is what decides.
    os.environ.pop("OPENAI_API_KEY", None)

    import market_intel.config as config_mod
    import market_intel.llm as llm

    config_mod.LLM_PROVIDER = "auto"
    llm._reachable = lambda *a, **k: False
