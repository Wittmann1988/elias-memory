from elias_memory.store.db import Database
from elias_memory.types import MemoryRecord

def test_db_init_creates_tables(tmp_db):
    db = Database(tmp_db)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "memories" in table_names
    db.close()

def test_db_wal_mode(tmp_db):
    db = Database(tmp_db)
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    db.close()

def test_insert_and_get(tmp_db):
    db = Database(tmp_db)
    rec = MemoryRecord(content="test fact", type="semantic", importance=0.8)
    db.insert(rec)
    loaded = db.get(rec.id)
    assert loaded is not None
    assert loaded.content == "test fact"
    assert loaded.type == "semantic"
    assert abs(loaded.importance - 0.8) < 0.001
    db.close()

def test_delete(tmp_db):
    db = Database(tmp_db)
    rec = MemoryRecord(content="to delete", type="episodic", importance=0.5)
    db.insert(rec)
    db.delete(rec.id)
    assert db.get(rec.id) is None
    db.close()

def test_update_access(tmp_db):
    db = Database(tmp_db)
    rec = MemoryRecord(content="accessed", type="semantic", importance=0.7)
    db.insert(rec)
    db.update_access(rec.id)
    loaded = db.get(rec.id)
    assert loaded.access_count == 1
    db.close()

def test_list_all(tmp_db):
    db = Database(tmp_db)
    db.insert(MemoryRecord(content="a", type="semantic", importance=0.5))
    db.insert(MemoryRecord(content="b", type="episodic", importance=0.6))
    all_recs = db.list_all()
    assert len(all_recs) == 2
    db.close()
