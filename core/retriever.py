from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=np.float32)

    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    embedding_matrix = np.asarray(embeddings, dtype=np.float32)

    query_norm = np.linalg.norm(query_vector)
    embedding_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_norms[embedding_norms == 0] = 1.0

    if query_norm == 0:
        return np.zeros(embedding_matrix.shape[0], dtype=np.float32)

    query_norm = query_vector / query_norm
    normalized_embeddings = embedding_matrix / embedding_norms
    return (normalized_embeddings @ query_norm.T).ravel().astype(np.float32)


def retrieve_top_k(
    chunks: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int = 4,
    min_similarity: float = 0.15,
    allow_fallback: bool = True,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    embeddings = np.vstack([chunk["embedding"] for chunk in chunks if "embedding" in chunk])
    if embeddings.size == 0:
        return []

    scores = cosine_similarity(query_embedding, embeddings)
    ranked_results = []

    for chunk, score in zip(chunks, scores):
        result = dict(chunk)
        result["score"] = float(score)
        ranked_results.append(result)

    ranked_results.sort(key=lambda item: item["score"], reverse=True)

    qualifying_results = [item for item in ranked_results if item["score"] >= min_similarity]
    if qualifying_results:
        return qualifying_results[:top_k]

    if allow_fallback:
        return ranked_results[:top_k]
    return []
