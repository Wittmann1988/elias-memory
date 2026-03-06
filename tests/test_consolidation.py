"""Tests for memory consolidation."""
import tempfile
import os
from elias_memory import Memory


def test_consolidate_creates_semantic():
    """Consolidation should merge similar episodic memories into semantic."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        # Add similar episodic memories
        for i in range(5):
            mem.add(
                f"Python debugging session #{i}: found null pointer exception in line {100+i}",
                type="episodic", importance=0.6,
                metadata={"topic": "debugging"},
            )

        # Add unrelated memory
        mem.add("The weather is nice today", type="episodic", importance=0.3)

        # Hash embeddings have low similarity — use very low threshold for testing
        new_ids = mem.consolidate(similarity_threshold=0.05, min_cluster_size=3)
        assert len(new_ids) >= 1

        # Check new semantic memory exists
        for mid in new_ids:
            rec = mem._records[mid]
            assert rec.type == "semantic"
            assert "consolidated_from" in rec.metadata
            assert rec.metadata["source_count"] >= 3

        mem.close()


def test_consolidate_marks_episodes():
    """Consolidated episodes should be marked."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        ids = []
        for i in range(4):
            mid = mem.add(
                f"API timeout error in service {i}, retry helped",
                type="episodic", importance=0.5,
                metadata={"topic": "api-errors"},
            )
            ids.append(mid)

        mem.consolidate(similarity_threshold=0.05, min_cluster_size=3)

        # Check some episodes are marked as consolidated
        consolidated_count = sum(
            1 for r in mem._records.values()
            if r.type == "episodic" and r.metadata.get("consolidated")
        )
        assert consolidated_count >= 3
        mem.close()


def test_consolidate_too_few_episodes():
    """Should not consolidate if fewer than min_cluster_size episodes."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")
        mem.add("single episode", type="episodic", importance=0.5)
        result = mem.consolidate(min_cluster_size=3)
        assert result == []
        mem.close()
