import numpy as np

from core.retriever import retrieve_top_k


def test_retrieval_orders_by_similarity():
    chunks = [
        {"text": "Python is a programming language", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
        {"text": "The moon is bright tonight", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
        {"text": "Python for machine learning", "embedding": np.array([0.9, 0.1], dtype=np.float32)},
    ]

    query = np.array([1.0, 0.1], dtype=np.float32)
    results = retrieve_top_k(chunks, query, top_k=2, min_similarity=0.0)

    assert results[0]["text"] == "Python for machine learning"
    assert len(results) == 2


def test_retrieval_falls_back_to_best_matches_when_threshold_is_tight():
    chunks = [
        {"text": "Soil texture is determined by sand, silt, and clay proportions.", "embedding": np.array([0.8, 0.2], dtype=np.float32)},
        {"text": "The weather is sunny and warm today.", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
    ]

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = retrieve_top_k(chunks, query, top_k=2, min_similarity=0.99, allow_fallback=True)

    assert len(results) == 2
    assert results[0]["text"] == "Soil texture is determined by sand, silt, and clay proportions."
    assert results[0]["score"] < 0.99
