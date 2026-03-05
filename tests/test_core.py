import json
from elias_memory import Memory, MemoryRecord


def test_memory_context_manager(tmp_db):
    with Memory(tmp_db) as mem:
        assert mem is not None


def test_add_and_recall(tmp_db):
    with Memory(tmp_db) as mem:
        mem.add("Python is great", type="semantic", importance=0.9)
        mem.add("Fixed a bug", type="episodic", importance=0.5)
        results = mem.recall("Python", top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, MemoryRecord) for r in results)


def test_reinforce(tmp_db):
    with Memory(tmp_db) as mem:
        mid = mem.add("important fact", type="semantic", importance=0.8)
        mem.reinforce(mid)
        rec = mem._db.get(mid)
        assert rec.access_count == 1


def test_decay_cycle(tmp_db):
    with Memory(tmp_db) as mem:
        mid = mem.add("will decay", type="semantic", importance=0.5)
        mem.decay_cycle()
        rec = mem._db.get(mid)
        assert rec.importance >= 0.01


def test_export_sft(tmp_db, tmp_path):
    path = str(tmp_path / "export.jsonl")
    with Memory(tmp_db) as mem:
        mem.add("fact one", type="semantic", importance=0.8)
        mem.add("event one", type="episodic", importance=0.6)
        mem.export_sft(path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "messages" in first


def test_add_returns_id(tmp_db):
    with Memory(tmp_db) as mem:
        mid = mem.add("test", type="semantic", importance=0.5)
        assert isinstance(mid, str)
        assert len(mid) == 36


def test_persistence(tmp_db):
    """Memories survive close and reopen."""
    with Memory(tmp_db) as mem:
        mem.add("persistent fact", type="semantic", importance=0.9)

    with Memory(tmp_db) as mem2:
        results = mem2.recall("persistent", top_k=1)
        assert len(results) == 1
        assert results[0].content == "persistent fact"
