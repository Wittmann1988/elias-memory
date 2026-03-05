import numpy as np
from elias_memory.store.vec import VectorIndex, NumpyVectorIndex

def test_vector_index_is_abc():
    import pytest
    with pytest.raises(TypeError):
        VectorIndex()

def test_numpy_index_add_and_search():
    idx = NumpyVectorIndex(dim=4)
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
    idx.add("c", np.array([0.9, 0.1, 0, 0], dtype=np.float32))
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
    assert len(results) == 2
    assert results[0][0] == "a"
    assert results[1][0] == "c"

def test_numpy_index_delete():
    idx = NumpyVectorIndex(dim=4)
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.delete("a")
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
    assert len(results) == 0

def test_numpy_index_empty_search():
    idx = NumpyVectorIndex(dim=4)
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
    assert results == []
