"""Memory consolidation — converts episodic memories into semantic knowledge.

Inspired by biological memory consolidation (hippocampus → neocortex):
- Groups similar episodic memories by vector proximity
- Extracts common patterns into semantic memories
- Marks source episodes as consolidated
"""
from __future__ import annotations

import numpy as np
from datetime import datetime, timezone
from typing import Any

from elias_memory.types import MemoryRecord
from elias_memory.store.vec import VectorIndex


def find_clusters(
    records: list[MemoryRecord],
    index: VectorIndex,
    embedder: Any,
    similarity_threshold: float = 0.6,
    min_cluster_size: int = 3,
) -> list[list[MemoryRecord]]:
    """Find clusters of similar episodic memories.

    Uses two strategies:
    1. Topic-based: group by metadata["topic"] (fast, works with any embedder)
    2. Vector-based: cosine similarity (better with real embeddings)

    On mobile (hash embedder), topic-based dominates.
    On desktop (NVIDIA embedder), vector-based gives better results.
    """
    if len(records) < min_cluster_size:
        return []

    # Strategy 1: Topic-based clustering
    by_topic: dict[str, list[MemoryRecord]] = {}
    no_topic = []
    for r in records:
        topic = r.metadata.get("topic")
        if topic:
            by_topic.setdefault(topic, []).append(r)
        else:
            no_topic.append(r)

    clusters = []
    for topic, group in by_topic.items():
        if len(group) >= min_cluster_size:
            clusters.append(group)

    # Strategy 2: Vector-based for records without topics
    if len(no_topic) >= min_cluster_size:
        vecs = {r.id: embedder.embed(r.content) for r in no_topic}
        used = set()

        for rec in sorted(no_topic, key=lambda r: r.importance, reverse=True):
            if rec.id in used:
                continue
            cluster = [rec]
            used.add(rec.id)
            query_vec = vecs[rec.id].astype(np.float32)
            norm_q = np.linalg.norm(query_vec)
            if norm_q == 0:
                continue

            for other in no_topic:
                if other.id in used:
                    continue
                other_vec = vecs[other.id].astype(np.float32)
                norm_o = np.linalg.norm(other_vec)
                if norm_o == 0:
                    continue
                sim = float(np.dot(query_vec, other_vec) / (norm_q * norm_o))
                if sim >= similarity_threshold:
                    cluster.append(other)
                    used.add(other.id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

    return clusters


def consolidate_cluster(cluster: list[MemoryRecord]) -> str:
    """Create a consolidated summary from a cluster of episodic memories.

    Returns a combined content string. For LLM-based summarization,
    override this in the Desktop version.
    """
    contents = [r.content for r in cluster]
    # Simple merge: deduplicate and combine
    unique = list(dict.fromkeys(contents))
    if len(unique) == 1:
        return unique[0]
    return f"Consolidated from {len(cluster)} episodes: " + " | ".join(
        c[:200] for c in unique[:5]
    )


def create_semantic_from_cluster(
    cluster: list[MemoryRecord],
    summary: str | None = None,
) -> MemoryRecord:
    """Create a new semantic memory from a cluster of episodic ones."""
    content = summary or consolidate_cluster(cluster)
    avg_importance = sum(r.importance for r in cluster) / len(cluster)
    # Consolidated memories are more important than their sources
    boosted = min(1.0, avg_importance * 1.3)

    source_ids = [r.id for r in cluster]
    topics = set()
    for r in cluster:
        if "topic" in r.metadata:
            topics.add(r.metadata["topic"])

    return MemoryRecord(
        content=content,
        type="semantic",
        importance=boosted,
        metadata={
            "consolidated_from": source_ids,
            "consolidated_at": datetime.now(timezone.utc).isoformat(),
            "source_count": len(cluster),
            "topics": list(topics),
        },
    )
