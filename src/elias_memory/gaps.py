"""Knowledge Gap Detection — identifies what the agent doesn't know.

Feeds the Curiosity Drive in Way2AGI's cognitive architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from elias_memory.types import MemoryRecord


@dataclass
class KnowledgeGap:
    topic: str
    coverage: float  # 0.0 = no knowledge, 1.0 = well covered
    memory_count: int
    avg_importance: float


def detect_gaps(
    records: dict[str, MemoryRecord],
    min_coverage_threshold: float = 0.3,
) -> list[KnowledgeGap]:
    """Detect topics with low coverage.

    Returns gaps sorted by coverage (lowest first = biggest gaps).
    """
    topics: dict[str, list[MemoryRecord]] = {}
    for rec in records.values():
        topic = rec.metadata.get("topic", "general")
        topics.setdefault(topic, []).append(rec)

    if not topics:
        return [KnowledgeGap(topic="everything", coverage=0.0, memory_count=0, avg_importance=0.0)]

    max_count = max(len(mems) for mems in topics.values())
    gaps = []

    for topic, mems in topics.items():
        coverage = min(1.0, len(mems) / max_count) if max_count > 0 else 0.0
        avg_imp = sum(r.importance for r in mems) / len(mems)
        gaps.append(KnowledgeGap(
            topic=topic,
            coverage=coverage,
            memory_count=len(mems),
            avg_importance=round(avg_imp, 3),
        ))

    gaps.sort(key=lambda g: g.coverage)
    return gaps


def detect_retrieval_gaps(
    query: str,
    results: list[tuple[str, float]],
    threshold: float = 0.3,
) -> bool:
    """Check if a recall query hit a knowledge gap.

    Returns True if the best similarity score is below threshold,
    meaning the agent has no good knowledge about this topic.
    """
    if not results:
        return True
    best_score = max(score for _, score in results)
    return best_score < threshold
