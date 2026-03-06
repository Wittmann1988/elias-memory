"""Memory facade — high-level API combining all components.

v0.3.0: Knowledge Graph, namespaced access control, entity extraction.
"""
from __future__ import annotations

from typing import Any

from elias_memory.consolidation import (
    consolidate_cluster,
    create_semantic_from_cluster,
    find_clusters,
)
from elias_memory.decay import ExponentialDecay
from elias_memory.embeddings.base import Embedder
from elias_memory.embeddings.fallback import HashEmbedder
from elias_memory.export import export_sft
from elias_memory.gaps import KnowledgeGap, detect_gaps, detect_retrieval_gaps
from elias_memory.graph import KnowledgeGraph, Relation, Entity
from elias_memory.retrieval import VectorRetriever
from elias_memory.store.db import Database
from elias_memory.store.vec import NumpyVectorIndex, VectorIndex
from elias_memory.types import MemoryRecord


def _make_embedder(profile: str, dim: int) -> Embedder:
    if profile == "desktop":
        try:
            from elias_memory.embeddings.nvidia import NvidiaEmbedder
            return NvidiaEmbedder()
        except Exception:
            pass
    return HashEmbedder(dim=dim)


def _make_vec_index(profile: str, dim: int) -> VectorIndex:
    if profile == "desktop":
        try:
            import faiss  # noqa: F401
            from elias_memory.store.faiss_index import FaissIndex
            return FaissIndex(dim=dim)
        except ImportError:
            pass
    return NumpyVectorIndex(dim=dim)


class Memory:
    """High-level facade for the memory framework.

    Profiles:
        - "desktop": NVIDIA embeddings, FAISS index, full features
        - "mobile": Hash embeddings, numpy index, basic features
        - "auto": Detect based on available libraries

    Namespaces:
        Each Memory instance can be scoped to specific namespaces.
        Only memories in those namespaces are loaded and searchable.
        - "global": Shared knowledge (rules, facts, preferences)
        - "project/X": Project-specific memories
        - "agent/X": Agent-private memories
        - "session/X": Temporary session buffer

    Scopes control visibility:
        - "shared": Visible to all namespaces
        - "project": Visible within same project
        - "agent": Visible only to owning agent
        - "session": Visible only in current session
    """

    def __init__(
        self,
        db_path: str,
        *,
        profile: str = "auto",
        namespace: str = "global",
        namespaces: list[str] | None = None,
        embedder: Embedder | None = None,
        vec_index: VectorIndex | None = None,
        decay_half_life: float = 7.0,
        embedding_dim: int = 384,
    ) -> None:
        """Initialize memory.

        Args:
            namespace: Default namespace for new memories.
            namespaces: List of namespaces to load. If None, loads all.
                        Example: ["global", "project/way2agi", "agent/elias"]
        """
        if profile == "auto":
            try:
                import faiss  # noqa: F401
                profile = "desktop"
            except ImportError:
                profile = "mobile"

        self.profile = profile
        self.namespace = namespace
        self._namespaces = namespaces
        self._db = Database(db_path)
        self._embedder = embedder or _make_embedder(profile, embedding_dim)
        self._decay = ExponentialDecay(half_life_days=decay_half_life)
        self._vec_index = vec_index or _make_vec_index(profile, embedding_dim)
        self._retriever = VectorRetriever(
            index=self._vec_index,
            embedder=self._embedder,
            decay=self._decay,
        )
        self._graph = KnowledgeGraph(self._db)
        self._records: dict[str, MemoryRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for rec in self._db.list_all(namespaces=self._namespaces):
            self._records[rec.id] = rec
            vec = self._embedder.embed(rec.content)
            self._vec_index.add(rec.id, vec)

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    def add(
        self,
        content: str,
        *,
        type: str,
        importance: float,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        scope: str = "shared",
        extract_entities: bool = True,
    ) -> str:
        rec = MemoryRecord(
            content=content,
            type=type,
            importance=importance,
            metadata=metadata or {},
            namespace=namespace or self.namespace,
            scope=scope,
        )
        vec = self._embedder.embed(content)
        rec.embedding = vec.tobytes()
        self._db.insert(rec)
        self._vec_index.add(rec.id, vec)
        self._records[rec.id] = rec

        if extract_entities:
            self._graph.extract_and_link(rec.id, content)

        return rec.id

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        graph_expand: bool = False,
    ) -> list[MemoryRecord]:
        """Recall memories matching a query.

        Only searches memories in loaded namespaces.
        If graph_expand=True, also includes graph-connected memories.
        """
        results = self._retriever.search(query, self._records, top_k=top_k)

        if graph_expand and results:
            result_ids = {r.id for r in results}
            extra_ids: set[str] = set()
            for rec in results[:3]:
                neighbors = self._graph.neighbors(rec.id, max_depth=1)
                extra_ids.update(neighbors - result_ids)

            for eid in list(extra_ids)[:top_k]:
                if eid in self._records:
                    results.append(self._records[eid])

        return results

    def reinforce(self, memory_id: str) -> None:
        self._db.update_access(memory_id)
        rec = self._records.get(memory_id)
        if rec:
            rec.access_count += 1

    def delete(self, memory_id: str) -> None:
        self._db.delete(memory_id)
        self._vec_index.delete(memory_id)
        self._records.pop(memory_id, None)

    def decay_cycle(self) -> int:
        pruned = 0
        to_delete = []
        for rec in self._records.values():
            new_importance = self._decay.compute(rec)
            if abs(new_importance - rec.importance) > 0.001:
                self._db.update_importance(rec.id, new_importance)
                rec.importance = new_importance
            if new_importance <= 0.01:
                to_delete.append(rec.id)
        for mid in to_delete:
            self.delete(mid)
            pruned += 1
        return pruned

    # --- Consolidation ---

    def consolidate(
        self,
        similarity_threshold: float = 0.6,
        min_cluster_size: int = 3,
        summarizer: Any | None = None,
    ) -> list[str]:
        episodes = [
            r for r in self._records.values()
            if r.type == "episodic" and not r.metadata.get("consolidated")
        ]
        if len(episodes) < min_cluster_size:
            return []

        clusters = find_clusters(
            episodes, self._vec_index, self._embedder,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )

        new_ids = []
        for cluster in clusters:
            if summarizer:
                summary = summarizer(cluster)
            else:
                summary = consolidate_cluster(cluster)

            semantic = create_semantic_from_cluster(cluster, summary)
            semantic.namespace = self.namespace
            vec = self._embedder.embed(semantic.content)
            semantic.embedding = vec.tobytes()
            self._db.insert(semantic)
            self._vec_index.add(semantic.id, vec)
            self._records[semantic.id] = semantic
            new_ids.append(semantic.id)

            for ep in cluster:
                ep.metadata["consolidated"] = True
                ep.metadata["consolidated_into"] = semantic.id
                self._db.execute(
                    "UPDATE memories SET metadata = ? WHERE id = ?",
                    (self._db._serialize_metadata(ep.metadata), ep.id),
                )

        return new_ids

    # --- Knowledge Gap Detection ---

    def knowledge_gaps(self, min_coverage: float = 0.3) -> list[KnowledgeGap]:
        return detect_gaps(self._records, min_coverage_threshold=min_coverage)

    def has_gap(self, query: str, threshold: float = 0.3) -> bool:
        query_vec = self._embedder.embed(query)
        results = self._vec_index.search(query_vec, top_k=3)
        return detect_retrieval_gaps(query, results, threshold=threshold)

    # --- Export ---

    def export_sft(self, path: str, *, min_importance: float = 0.0) -> None:
        records = [
            r for r in self._records.values()
            if r.importance >= min_importance
        ]
        export_sft(records, path)

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        ns_counts: dict[str, int] = {}
        total_importance = 0.0
        for rec in self._records.values():
            type_counts[rec.type] = type_counts.get(rec.type, 0) + 1
            ns_counts[rec.namespace] = ns_counts.get(rec.namespace, 0) + 1
            total_importance += rec.importance
        total = len(self._records)
        return {
            "total": total,
            "by_type": type_counts,
            "by_namespace": ns_counts,
            "avg_importance": round(total_importance / total, 3) if total > 0 else 0,
            "profile": self.profile,
            "active_namespaces": self._namespaces or ["all"],
            "graph": self._graph.stats(),
        }

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
