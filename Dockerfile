# --- Phase 6: deploy the service in a container -------------------------
#
#   docker build -t market-intel .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" market-intel
#
# Two things are baked in at build time on purpose:
#
#   * the embedding model (~130 MB), so a cold container answers its
#     first request immediately instead of blocking on a download;
#   * a sqlite-vec smoke test, because loading the extension depends on
#     the host SQLite build — if that is broken we want the *build* to
#     fail, not the first search at runtime.
#
# The image has no LLM in it. Point it at Ollama or OpenAI with env vars
# (see docker-compose.yml and .env.example) — credentials never get
# copied into a layer.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Where the embedding model lives (baked in below).
    FASTEMBED_CACHE_DIR=/opt/fastembed \
    # Inside the container the API must listen on all interfaces.
    HOST=0.0.0.0 \
    PORT=8000 \
    MARKET_INTEL_DB=/app/data/market_intel.db \
    MARKET_INTEL_REPORT_DIR=/app/data/reports

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY market_intel/ ./market_intel/
COPY run_api.py run_report.py run_ask.py run_search.py run_index.py \
     run_pipeline.py run_eval.py run_corpus.py ./

# Fail the build if the vector extension cannot load in this image.
RUN python -c "import sqlite3, sqlite_vec; \
c = sqlite3.connect(':memory:'); c.enable_load_extension(True); \
sqlite_vec.load(c); print('sqlite-vec loads OK')"

# Bake the embedding model into the image (one download, at build time).
RUN python -c "from market_intel.embed import get_embedder; \
e = get_embedder(); print('embedder ready:', e.model_name, e.dim, 'dims')"

# Run as a non-root user; it needs to own the data dir it writes to.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data/reports \
    && chown -R app:app /app
USER app

EXPOSE 8000

# No curl in slim images — use the stdlib. /health never 500s on a
# missing dependency, so it is a true liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; \
urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"

CMD ["uvicorn", "market_intel.api:app", "--host", "0.0.0.0", "--port", "8000"]
