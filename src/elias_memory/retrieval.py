from __future__ import annotations
from elias_memory.decay import DecayStrategy
from elias_memory.embeddings.base import Embedder
from elias_memory.store.vec import VectorIndex
from elias_memory.types import MemoryRecord

class VectorRetriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, decay: DecayStrategy) -> None:
        self._index = index
        self._embedder = embedder
        self._decay = decay

    def search(self, query: str, records: dict[str, MemoryRecord], top_k: int = 5) -> list[MemoryRecord]:
        if not records:
            return []
        query_vec = self._embedder.embed(query)
        candidates = self._index.search(query_vec, top_k=top_k * 3)
        scored = []
        for mem_id, similarity in candidates:
            rec = records.get(mem_id)
            if rec is None:
                continue
            decay_score = self._decay.compute(rec)
            combined = 0.7 * similarity + 0.3 * decay_score
            scored.append((rec, combined))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in scored[:top_k]]
