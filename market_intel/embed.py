"""Embedding wrapper around a local model (fastembed / ONNX Runtime).

Why local? No API key, no per-call cost, works offline once the model
is downloaded — a good fit for a portfolio project and for learning
how embeddings actually behave.

Model: BAAI/bge-small-en-v1.5 — small (384 dims) and strong for
English retrieval. bge models are trained with an asymmetric
convention: *queries* get an instruction prefix, *documents* do not.
So ``embed_query`` prefixes the instruction and ``embed_documents``
does not. Getting this asymmetry right is a classic RAG gotcha.
"""

import numpy as np

from market_intel import config

# Instruction bge expects on the query side (see BAAI/bge docs).
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Lazily loads the model on first use (downloads it once)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBED_MODEL
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=config.EMBED_CACHE_DIR or None,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed many document texts -> (n_texts, dim) float32 matrix."""
        model = self._load()
        vectors = list(model.embed(texts))  # generator -> list of vectors
        matrix = np.vstack(vectors).astype(np.float32)
        self._dim = matrix.shape[1]
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query (with the bge instruction prefix)."""
        model = self._load()
        vector = next(model.embed([f"{_QUERY_INSTRUCTION}{text}"]))
        return np.asarray(vector, dtype=np.float32)

    @property
    def dim(self) -> int:
        """Vector dimensions; triggers a load if not yet known."""
        if self._dim is None:
            self.embed_documents([""])
        return self._dim


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Module-level singleton: the model is loaded once per process."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER