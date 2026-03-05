from elias_memory.types import MemoryRecord, VALID_TYPES

def test_memory_record_creation():
    rec = MemoryRecord(content="Test fact", type="semantic", importance=0.8)
    assert rec.content == "Test fact"
    assert rec.type == "semantic"
    assert rec.importance == 0.8
    assert rec.id
    assert rec.access_count == 0
    assert rec.embedding is None

def test_memory_record_invalid_type():
    import pytest
    with pytest.raises(ValueError, match="Invalid type"):
        MemoryRecord(content="x", type="invalid", importance=0.5)

def test_memory_record_importance_clamped():
    rec = MemoryRecord(content="x", type="semantic", importance=0.0)
    assert rec.importance == 0.01
    rec2 = MemoryRecord(content="x", type="semantic", importance=1.5)
    assert rec2.importance == 1.0

def test_valid_types():
    assert "semantic" in VALID_TYPES
    assert "episodic" in VALID_TYPES
