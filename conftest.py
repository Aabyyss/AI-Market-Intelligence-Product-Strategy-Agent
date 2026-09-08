"""Pytest bootstrap.

An empty conftest at the project root makes pytest put the root
directory on sys.path, so ``import market_intel`` resolves no matter
how pytest is invoked (``python -m pytest`` or bare ``pytest``).
"""