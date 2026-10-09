from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - environment-specific fallback
    SentenceTransformer = None


class LocalFallbackEmbeddingModel:
    """Simple local embedding fallback for restricted environments without sentence-transformers."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"\b\w+\b", text) if token.strip()]

    def _vector_for_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vector = np.zeros(self.dimension, dtype=np.float32)
        if not tokens:
            return vector

        for index, token in enumerate(tokens[: self.dimension]):
            value = 0.0
            for char in token:
                value += ord(char)
            vector[index % self.dimension] = value / max(len(token), 1)

        return vector

    def encode(self, texts: Iterable[str], convert_to_numpy: bool = True, normalize_embeddings: bool = True):
        vector_list = [self._vector_for_text(text) for text in texts]
        matrix = np.vstack(vector_list).astype(np.float32) if vector_list else np.empty((0, self.dimension), dtype=np.float32)
        if normalize_embeddings and matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            matrix = matrix / norms
        return matrix if convert_to_numpy else [row.tolist() for row in matrix]


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is not None:
        try:
            return SentenceTransformer(model_name)
        except Exception:
            pass
    return LocalFallbackEmbeddingModel()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def embed_texts(texts: list[str], model) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str, model) -> np.ndarray:
    encoded = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(encoded, dtype=np.float32)[0]
