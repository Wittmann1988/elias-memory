import numpy as np
from elias_memory.embeddings.base import Embedder
from elias_memory.embeddings.fallback import HashEmbedder

def test_hash_embedder_returns_correct_dim():
    emb = HashEmbedder(dim=384)
    vec = emb.embed("hello world")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

def test_hash_embedder_deterministic():
    emb = HashEmbedder(dim=384)
    v1 = emb.embed("same text")
    v2 = emb.embed("same text")
    np.testing.assert_array_equal(v1, v2)

def test_hash_embedder_different_texts_differ():
    emb = HashEmbedder(dim=384)
    v1 = emb.embed("text one")
    v2 = emb.embed("text two")
    assert not np.array_equal(v1, v2)

def test_embedder_is_abc():
    import pytest
    with pytest.raises(TypeError):
        Embedder()

def test_hash_embedder_dim_property():
    emb = HashEmbedder(dim=128)
    assert emb.dim == 128
