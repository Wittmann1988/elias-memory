"""Tests for knowledge gap detection."""
import tempfile
import os
from elias_memory import Memory


def test_detect_gaps_basic():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        # Add many Python memories, few Rust memories
        for i in range(10):
            mem.add(f"Python tip #{i}", type="semantic", importance=0.7, metadata={"topic": "python"})
        mem.add("Rust ownership basics", type="semantic", importance=0.5, metadata={"topic": "rust"})

        gaps = mem.knowledge_gaps()
        assert len(gaps) >= 2
        # Rust should be the bigger gap
        rust_gap = next(g for g in gaps if g.topic == "rust")
        python_gap = next(g for g in gaps if g.topic == "python")
        assert rust_gap.coverage < python_gap.coverage
        assert rust_gap.memory_count == 1
        mem.close()


def test_detect_gaps_empty():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")
        gaps = mem.knowledge_gaps()
        assert len(gaps) == 1
        assert gaps[0].topic == "everything"
        assert gaps[0].coverage == 0.0
        mem.close()


def test_has_gap():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")
        mem.add("Python is great", type="semantic", importance=0.8)
        # With hash embeddings, unrelated queries should have low similarity
        # This test is embedder-dependent
        result = mem.has_gap("completely unrelated quantum physics topic")
        # Just test it doesn't crash; actual gap detection quality depends on embedder
        assert isinstance(result, bool)
        mem.close()
