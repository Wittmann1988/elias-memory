import numpy as np
from elias_memory.retrieval import VectorRetriever
from elias_memory.store.vec import NumpyVectorIndex
from elias_memory.embeddings.fallback import HashEmbedder
from elias_memory.decay import ExponentialDecay
from elias_memory.types import MemoryRecord

def test_retriever_basic():
    embedder = HashEmbedder(dim=64)
    index = NumpyVectorIndex(dim=64)
    decay = ExponentialDecay(half_life_days=7)
    records = {}
    for text in ["Python is great", "Java is verbose", "Python for data science"]:
        rec = MemoryRecord(content=text, type="semantic", importance=0.8)
        vec = embedder.embed(text)
        index.add(rec.id, vec)
        records[rec.id] = rec
    retriever = VectorRetriever(index=index, embedder=embedder, decay=decay)
    results = retriever.search("Python programming", records, top_k=2)
    assert len(results) == 2
    assert all(hasattr(r, "content") for r in results)

def test_retriever_empty_index():
    embedder = HashEmbedder(dim=64)
    index = NumpyVectorIndex(dim=64)
    decay = ExponentialDecay(half_life_days=7)
    retriever = VectorRetriever(index=index, embedder=embedder, decay=decay)
    results = retriever.search("anything", {}, top_k=5)
    assert results == []
