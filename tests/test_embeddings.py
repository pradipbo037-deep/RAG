import numpy as np

from core.embeddings import normalize_embeddings


def test_normalize_embeddings_maintains_unit_length():
    values = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(values)

    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.allclose(normalized[1], np.array([0.0, 0.0], dtype=np.float32))
