"""Memory facade — high-level API combining all components."""
from __future__ import annotations

from typing import Any

from elias_memory.decay import ExponentialDecay
from elias_memory.embeddings.fallback import HashEmbedder
from elias_memory.export import export_sft
from elias_memory.retrieval import VectorRetriever
from elias_memory.store.db import Database
from elias_memory.store.vec import NumpyVectorIndex
from elias_memory.types import MemoryRecord


class Memory:
    """High-level facade for the memory framework."""

    def __init__(
        self,
        db_path: str,
        *,
        embedder: object | None = None,
        decay_half_life: float = 7.0,
        embedding_dim: int = 384,
    ) -> None:
        self._db = Database(db_path)
        self._embedder = embedder or HashEmbedder(dim=embedding_dim)
        self._decay = ExponentialDecay(half_life_days=decay_half_life)
        self._vec_index = NumpyVectorIndex(dim=embedding_dim)
        self._retriever = VectorRetriever(
            index=self._vec_index,
            embedder=self._embedder,
            decay=self._decay,
        )
        self._records: dict[str, MemoryRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for rec in self._db.list_all():
            self._records[rec.id] = rec
            vec = self._embedder.embed(rec.content)
            self._vec_index.add(rec.id, vec)

    def add(
        self,
        content: str,
        *,
        type: str,
        importance: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        rec = MemoryRecord(
            content=content,
            type=type,
            importance=importance,
            metadata=metadata or {},
        )
        vec = self._embedder.embed(content)
        rec.embedding = vec.tobytes()
        self._db.insert(rec)
        self._vec_index.add(rec.id, vec)
        self._records[rec.id] = rec
        return rec.id

    def recall(self, query: str, *, top_k: int = 5) -> list[MemoryRecord]:
        return self._retriever.search(query, self._records, top_k=top_k)

    def reinforce(self, memory_id: str) -> None:
        self._db.update_access(memory_id)
        rec = self._records.get(memory_id)
        if rec:
            rec.access_count += 1

    def decay_cycle(self) -> None:
        for rec in self._records.values():
            new_importance = self._decay.compute(rec)
            if abs(new_importance - rec.importance) > 0.001:
                self._db.update_importance(rec.id, new_importance)
                rec.importance = new_importance

    def export_sft(self, path: str) -> None:
        export_sft(list(self._records.values()), path)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
