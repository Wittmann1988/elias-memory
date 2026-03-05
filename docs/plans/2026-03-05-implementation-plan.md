# elias-memory v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean-room Python memory framework with vector retrieval, decay, reinforcement, and SFT export.

**Architecture:** SQLite+WAL as single-file store, pluggable embeddings (NVIDIA NIM 384-dim + hash fallback), exponential decay (7d half-life), vector-only retrieval in v1. ABCs for all swappable components.

**Tech Stack:** Python 3.12, SQLite 3.51, numpy, httpx, pytest. Optional: sqlite-vec.

---

### Task 1: Project Scaffolding + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `src/elias_memory/__init__.py`
- Create: `src/elias_memory/core.py` (empty placeholder)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "elias-memory"
version = "0.1.0"
description = "Persistent memory framework for AI agents"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "httpx>=0.24",
]

[project.optional-dependencies]
vec = ["sqlite-vec>=0.1"]
full = ["sqlite-vec>=0.1", "rank-bm25>=0.2"]
dev = ["pytest>=8.0", "pytest-cov>=4.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create src/elias_memory/__init__.py**

```python
"""elias-memory: Persistent memory framework for AI agents."""

__version__ = "0.1.0"
```

**Step 3: Create tests/conftest.py**

```python
import os
import tempfile
import pytest

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path."""
    return str(tmp_path / "test_memory.db")
```

**Step 4: Verify pytest discovers tests**

Run: `cd ~/repos/elias-memory && python -m pytest --co -q`
Expected: "no tests ran" (but no import errors)

**Step 5: Commit**

```bash
git init && git add -A && git commit -m "chore: project scaffolding with pyproject.toml and test setup"
```

---

### Task 2: MemoryRecord Dataclass + Types

**Files:**
- Create: `src/elias_memory/types.py`
- Create: `tests/test_types.py`

**Step 1: Write failing test**

```python
# tests/test_types.py
from elias_memory.types import MemoryRecord, VALID_TYPES
from datetime import datetime, timezone

def test_memory_record_creation():
    rec = MemoryRecord(
        content="Test fact",
        type="semantic",
        importance=0.8,
    )
    assert rec.content == "Test fact"
    assert rec.type == "semantic"
    assert rec.importance == 0.8
    assert rec.id  # UUID generated
    assert rec.access_count == 0
    assert rec.embedding is None

def test_memory_record_invalid_type():
    import pytest
    with pytest.raises(ValueError, match="Invalid type"):
        MemoryRecord(content="x", type="invalid", importance=0.5)

def test_memory_record_importance_clamped():
    rec = MemoryRecord(content="x", type="semantic", importance=0.0)
    assert rec.importance == 0.01  # floor

    rec2 = MemoryRecord(content="x", type="semantic", importance=1.5)
    assert rec2.importance == 1.0  # ceiling

def test_valid_types():
    assert "semantic" in VALID_TYPES
    assert "episodic" in VALID_TYPES
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write implementation**

```python
# src/elias_memory/types.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

VALID_TYPES = frozenset({"semantic", "episodic"})

@dataclass
class MemoryRecord:
    content: str
    type: str
    importance: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: bytes | None = None

    def __post_init__(self):
        if self.type not in VALID_TYPES:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of: {VALID_TYPES}")
        self.importance = max(0.01, min(1.0, self.importance))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_types.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/elias_memory/types.py tests/test_types.py
git commit -m "feat: MemoryRecord dataclass with validation and type checking"
```

---

### Task 3: SQLite Store + Schema

**Files:**
- Create: `src/elias_memory/store/__init__.py`
- Create: `src/elias_memory/store/db.py`
- Create: `src/elias_memory/store/schema.sql`
- Create: `tests/test_store.py`

**Step 1: Write failing test**

```python
# tests/test_store.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL

**Step 3: Create schema.sql**

```sql
-- src/elias_memory/store/schema.sql
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('semantic','episodic')),
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
```

**Step 4: Write Database implementation**

```python
# src/elias_memory/store/db.py
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from elias_memory.types import MemoryRecord

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        schema = _SCHEMA_PATH.read_text()
        self._conn.executescript(schema)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def insert(self, rec: MemoryRecord) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO memories
                   (id, content, type, importance, created_at, accessed_at,
                    access_count, metadata, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.id,
                    rec.content,
                    rec.type,
                    rec.importance,
                    rec.created_at.isoformat(),
                    rec.accessed_at.isoformat(),
                    rec.access_count,
                    json.dumps(rec.metadata),
                    rec.embedding,
                ),
            )
            self._conn.commit()

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete(self, memory_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()

    def update_access(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """UPDATE memories
                   SET access_count = access_count + 1, accessed_at = ?
                   WHERE id = ?""",
                (now, memory_id),
            )
            self._conn.commit()

    def update_importance(self, memory_id: str, importance: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (max(0.01, min(1.0, importance)), memory_id),
            )
            self._conn.commit()

    def list_all(self) -> list[MemoryRecord]:
        rows = self.execute("SELECT * FROM memories").fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: tuple) -> MemoryRecord:
        return MemoryRecord(
            id=row[0],
            content=row[1],
            type=row[2],
            importance=row[3],
            created_at=datetime.fromisoformat(row[4]),
            accessed_at=datetime.fromisoformat(row[5]),
            access_count=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
            embedding=row[8],
        )

    def close(self) -> None:
        self._conn.close()
```

**Step 5: Create store/__init__.py**

```python
# src/elias_memory/store/__init__.py
from .db import Database

__all__ = ["Database"]
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_store.py -v`
Expected: 6 passed

**Step 7: Commit**

```bash
git add src/elias_memory/store/ tests/test_store.py
git commit -m "feat: SQLite store with WAL mode, CRUD operations, thread safety"
```

---

### Task 4: Embedder ABC + Hash Fallback

**Files:**
- Create: `src/elias_memory/embeddings/__init__.py`
- Create: `src/elias_memory/embeddings/base.py`
- Create: `src/elias_memory/embeddings/fallback.py`
- Create: `tests/test_embeddings.py`

**Step 1: Write failing test**

```python
# tests/test_embeddings.py
import numpy as np
from elias_memory.embeddings.base import Embedder
from elias_memory.embeddings.fallback import HashEmbedder

def test_hash_embedder_returns_correct_dim():
    emb = HashEmbedder(dim=384)
    vec = emb.embed("hello world")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

def test_hash_embedder_deterministic():
    emb = HashEmbedder(dim=384)
    v1 = emb.embed("same text")
    v2 = emb.embed("same text")
    np.testing.assert_array_equal(v1, v2)

def test_hash_embedder_different_texts_differ():
    emb = HashEmbedder(dim=384)
    v1 = emb.embed("text one")
    v2 = emb.embed("text two")
    assert not np.array_equal(v1, v2)

def test_embedder_is_abc():
    import pytest
    with pytest.raises(TypeError):
        Embedder()  # Can't instantiate ABC

def test_hash_embedder_dim_property():
    emb = HashEmbedder(dim=128)
    assert emb.dim == 128
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: FAIL

**Step 3: Write Embedder ABC**

```python
# src/elias_memory/embeddings/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a float32 vector for the given text."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...
```

**Step 4: Write HashEmbedder fallback**

```python
# src/elias_memory/embeddings/fallback.py
from __future__ import annotations

import hashlib
import numpy as np

from .base import Embedder


class HashEmbedder(Embedder):
    """Deterministic hash-based embedder. Zero dependencies beyond numpy.
    Not semantically meaningful, but provides consistent vectors for testing
    and environments without a real embedding model."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        # Use SHA-512 repeatedly to fill the vector
        h = hashlib.sha512(text.encode()).digest()
        # Extend hash bytes to cover dim * 4 bytes (float32)
        needed = self._dim * 4
        data = h
        while len(data) < needed:
            data += hashlib.sha512(data).digest()
        raw = np.frombuffer(data[:needed], dtype=np.float32).copy()
        # Normalize to unit vector
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw /= norm
        return raw
```

**Step 5: Create embeddings/__init__.py**

```python
# src/elias_memory/embeddings/__init__.py
from .base import Embedder
from .fallback import HashEmbedder

__all__ = ["Embedder", "HashEmbedder"]
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: 5 passed

**Step 7: Commit**

```bash
git add src/elias_memory/embeddings/ tests/test_embeddings.py
git commit -m "feat: Embedder ABC + deterministic hash fallback"
```

---

### Task 5: NVIDIA NIM Embedder

**Files:**
- Create: `src/elias_memory/embeddings/nvidia.py`
- Create: `tests/test_nvidia_embedder.py`

**Step 1: Write failing test (mocked — no real API call)**

```python
# tests/test_nvidia_embedder.py
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from elias_memory.embeddings.nvidia import NvidiaEmbedder

def test_nvidia_embedder_dim():
    emb = NvidiaEmbedder(api_key="fake", dim=384)
    assert emb.dim == 384

def test_nvidia_embedder_no_key_raises():
    with pytest.raises(ValueError, match="API key"):
        NvidiaEmbedder(api_key="", dim=384)

@patch("elias_memory.embeddings.nvidia.httpx")
def test_nvidia_embedder_calls_api(mock_httpx):
    fake_vec = np.random.randn(384).astype(np.float32).tolist()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": fake_vec}]
    }
    mock_httpx.post.return_value = mock_response

    emb = NvidiaEmbedder(api_key="test-key", dim=384)
    result = emb.embed("hello")
    assert result.shape == (384,)
    assert result.dtype == np.float32
    mock_httpx.post.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nvidia_embedder.py -v`
Expected: FAIL

**Step 3: Write NvidiaEmbedder**

```python
# src/elias_memory/embeddings/nvidia.py
from __future__ import annotations

import httpx
import numpy as np

from .base import Embedder

DEFAULT_URL = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"


class NvidiaEmbedder(Embedder):
    """NVIDIA NIM embedding API client."""

    def __init__(
        self,
        api_key: str,
        dim: int = 384,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("API key required for NVIDIA NIM embedder")
        self._api_key = api_key
        self._dim = dim
        self._url = url
        self._model = model

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        response = httpx.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": [text],
                "model": self._model,
                "encoding_format": "float",
                "input_type": "query",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        # Truncate or pad to target dim
        if len(vec) > self._dim:
            vec = vec[: self._dim]
        elif len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))
        return vec
```

**Step 4: Update embeddings/__init__.py**

```python
# src/elias_memory/embeddings/__init__.py
from .base import Embedder
from .fallback import HashEmbedder
from .nvidia import NvidiaEmbedder

__all__ = ["Embedder", "HashEmbedder", "NvidiaEmbedder"]
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_nvidia_embedder.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
git add src/elias_memory/embeddings/ tests/test_nvidia_embedder.py
git commit -m "feat: NVIDIA NIM embedder with API client"
```

---

### Task 6: VectorIndex ABC + Numpy Fallback

**Files:**
- Create: `src/elias_memory/store/vec.py`
- Update: `src/elias_memory/store/__init__.py`
- Create: `tests/test_vec.py`

**Step 1: Write failing test**

```python
# tests/test_vec.py
import numpy as np
from elias_memory.store.vec import VectorIndex, NumpyVectorIndex

def test_vector_index_is_abc():
    import pytest
    with pytest.raises(TypeError):
        VectorIndex()

def test_numpy_index_add_and_search():
    idx = NumpyVectorIndex(dim=4)
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
    idx.add("c", np.array([0.9, 0.1, 0, 0], dtype=np.float32))

    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
    assert len(results) == 2
    assert results[0][0] == "a"  # most similar
    assert results[1][0] == "c"  # second most similar

def test_numpy_index_delete():
    idx = NumpyVectorIndex(dim=4)
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.delete("a")
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
    assert len(results) == 0

def test_numpy_index_empty_search():
    idx = NumpyVectorIndex(dim=4)
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
    assert results == []
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vec.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/elias_memory/store/vec.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    @abstractmethod
    def add(self, id: str, vec: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...


class NumpyVectorIndex(VectorIndex):
    """Brute-force cosine similarity using numpy. Works everywhere."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []

    def add(self, id: str, vec: np.ndarray) -> None:
        norm = np.linalg.norm(vec)
        normalized = vec / norm if norm > 0 else vec
        self._ids.append(id)
        self._vecs.append(normalized)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self._vecs:
            return []
        norm = np.linalg.norm(query_vec)
        q = query_vec / norm if norm > 0 else query_vec
        matrix = np.stack(self._vecs)
        similarities = matrix @ q
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self._ids[i], float(similarities[i])) for i in top_indices]

    def delete(self, id: str) -> None:
        try:
            idx = self._ids.index(id)
            self._ids.pop(idx)
            self._vecs.pop(idx)
        except ValueError:
            pass
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_vec.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/elias_memory/store/vec.py tests/test_vec.py
git commit -m "feat: VectorIndex ABC + numpy brute-force fallback"
```

---

### Task 7: Decay Strategy

**Files:**
- Create: `src/elias_memory/decay.py`
- Create: `tests/test_decay.py`

**Step 1: Write failing test**

```python
# tests/test_decay.py
import math
from datetime import datetime, timezone, timedelta
from elias_memory.decay import DecayStrategy, ExponentialDecay
from elias_memory.types import MemoryRecord

def test_decay_strategy_is_abc():
    import pytest
    with pytest.raises(TypeError):
        DecayStrategy()

def test_exponential_decay_fresh_memory():
    decay = ExponentialDecay(half_life_days=7)
    rec = MemoryRecord(content="fresh", type="semantic", importance=1.0)
    score = decay.compute(rec)
    assert 0.99 < score <= 1.0  # almost no decay

def test_exponential_decay_after_half_life():
    decay = ExponentialDecay(half_life_days=7)
    rec = MemoryRecord(content="old", type="semantic", importance=1.0)
    rec.accessed_at = datetime.now(timezone.utc) - timedelta(days=7)
    score = decay.compute(rec)
    assert 0.45 < score < 0.55  # roughly halved

def test_exponential_decay_floor():
    decay = ExponentialDecay(half_life_days=7)
    rec = MemoryRecord(content="ancient", type="semantic", importance=0.1)
    rec.accessed_at = datetime.now(timezone.utc) - timedelta(days=365)
    score = decay.compute(rec)
    assert score == 0.01  # floor

def test_exponential_decay_access_boost():
    decay = ExponentialDecay(half_life_days=7)
    rec = MemoryRecord(content="popular", type="semantic", importance=0.5)
    rec.access_count = 10
    rec.accessed_at = datetime.now(timezone.utc) - timedelta(days=7)
    score = decay.compute(rec)
    # access_weight = 1 + log(11) ≈ 3.4, so score should be higher than 0.5 * 0.5
    assert score > 0.5 * 0.5
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decay.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/elias_memory/decay.py
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from elias_memory.types import MemoryRecord

FLOOR = 0.01


class DecayStrategy(ABC):
    @abstractmethod
    def compute(self, record: MemoryRecord) -> float:
        """Return decayed importance score (>= FLOOR)."""
        ...


class ExponentialDecay(DecayStrategy):
    def __init__(self, half_life_days: float = 7.0) -> None:
        self._lambda = math.log(2) / half_life_days

    def compute(self, record: MemoryRecord) -> float:
        now = datetime.now(timezone.utc)
        days = (now - record.accessed_at).total_seconds() / 86400
        access_weight = 1.0 + math.log(record.access_count + 1)
        raw = record.importance * access_weight * math.exp(-self._lambda * days)
        return max(FLOOR, min(1.0, raw))
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_decay.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/elias_memory/decay.py tests/test_decay.py
git commit -m "feat: ExponentialDecay with 7-day half-life and importance floor"
```

---

### Task 8: Retrieval (Vector-Only v1)

**Files:**
- Create: `src/elias_memory/retrieval.py`
- Create: `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
# tests/test_retrieval.py
import numpy as np
from elias_memory.retrieval import VectorRetriever
from elias_memory.store.vec import NumpyVectorIndex
from elias_memory.embeddings.fallback import HashEmbedder
from elias_memory.decay import ExponentialDecay
from elias_memory.types import MemoryRecord

def test_retriever_basic():
    embedder = HashEmbedder(dim=64)
    index = NumpyVectorIndex(dim=64)
    decay = ExponentialDecay(half_life_days=7)

    # Add some memories to index
    records = {}
    for text in ["Python is great", "Java is verbose", "Python for data science"]:
        rec = MemoryRecord(content=text, type="semantic", importance=0.8)
        vec = embedder.embed(text)
        index.add(rec.id, vec)
        records[rec.id] = rec

    retriever = VectorRetriever(index=index, embedder=embedder, decay=decay)
    results = retriever.search("Python programming", records, top_k=2)
    assert len(results) == 2
    assert all(hasattr(r, "content") for r in results)

def test_retriever_empty_index():
    embedder = HashEmbedder(dim=64)
    index = NumpyVectorIndex(dim=64)
    decay = ExponentialDecay(half_life_days=7)
    retriever = VectorRetriever(index=index, embedder=embedder, decay=decay)
    results = retriever.search("anything", {}, top_k=5)
    assert results == []
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/elias_memory/retrieval.py
from __future__ import annotations

from elias_memory.decay import DecayStrategy
from elias_memory.embeddings.base import Embedder
from elias_memory.store.vec import VectorIndex
from elias_memory.types import MemoryRecord


class VectorRetriever:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        decay: DecayStrategy,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._decay = decay

    def search(
        self,
        query: str,
        records: dict[str, MemoryRecord],
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        if not records:
            return []
        query_vec = self._embedder.embed(query)
        # Get more candidates than needed, then re-rank with decay
        candidates = self._index.search(query_vec, top_k=top_k * 3)

        scored = []
        for mem_id, similarity in candidates:
            rec = records.get(mem_id)
            if rec is None:
                continue
            decay_score = self._decay.compute(rec)
            # Combined score: 70% vector similarity + 30% decay-weighted importance
            combined = 0.7 * similarity + 0.3 * decay_score
            scored.append((rec, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in scored[:top_k]]
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/elias_memory/retrieval.py tests/test_retrieval.py
git commit -m "feat: VectorRetriever with decay-weighted scoring"
```

---

### Task 9: SFT Export

**Files:**
- Create: `src/elias_memory/export.py`
- Create: `tests/test_export.py`

**Step 1: Write failing test**

```python
# tests/test_export.py
import json
from elias_memory.export import export_sft
from elias_memory.types import MemoryRecord

def test_export_sft_creates_jsonl(tmp_path):
    path = tmp_path / "output.jsonl"
    records = [
        MemoryRecord(content="Python is best", type="semantic", importance=0.9),
        MemoryRecord(content="Fixed bug today", type="episodic", importance=0.7),
    ]
    export_sft(records, str(path))
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "messages" in first
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["role"] == "user"
    assert first["messages"][2]["role"] == "assistant"

def test_export_sft_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    export_sft([], str(path))
    assert path.read_text() == ""
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/elias_memory/export.py
from __future__ import annotations

import json
from typing import Sequence

from elias_memory.types import MemoryRecord


def export_sft(records: Sequence[MemoryRecord], path: str) -> None:
    """Export memories as HuggingFace Chat-format JSONL for SFT training."""
    with open(path, "w") as f:
        for rec in records:
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI memory manager. Store, retrieve, and maintain memories efficiently.",
                    },
                    {
                        "role": "user",
                        "content": f"Remember this {rec.type} memory with importance {rec.importance}: {rec.content}",
                    },
                    {
                        "role": "assistant",
                        "content": f"Stored {rec.type} memory (importance={rec.importance}): {rec.content}",
                    },
                ],
                "metadata": {
                    "type": rec.type,
                    "importance": rec.importance,
                    "created_at": rec.created_at.isoformat(),
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_export.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/elias_memory/export.py tests/test_export.py
git commit -m "feat: SFT export in HuggingFace Chat JSONL format"
```

---

### Task 10: Memory Facade (core.py) — Alles zusammen

**Files:**
- Create: `src/elias_memory/core.py`
- Update: `src/elias_memory/__init__.py`
- Create: `tests/test_core.py`

**Step 1: Write failing test**

```python
# tests/test_core.py
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
        # After decay on a fresh memory, importance should stay near original
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
        assert len(mid) == 36  # UUID format
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core.py -v`
Expected: FAIL

**Step 3: Write core.py**

```python
# src/elias_memory/core.py
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
        """Load all memories from DB into vector index."""
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
```

**Step 4: Update __init__.py**

```python
# src/elias_memory/__init__.py
"""elias-memory: Persistent memory framework for AI agents."""

from elias_memory.core import Memory
from elias_memory.types import MemoryRecord

__version__ = "0.1.0"
__all__ = ["Memory", "MemoryRecord"]
```

**Step 5: Run ALL tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass (20+ tests)

**Step 6: Commit**

```bash
git add src/elias_memory/core.py src/elias_memory/__init__.py tests/test_core.py
git commit -m "feat: Memory facade — complete v1 API with add, recall, reinforce, decay, export"
```

---

### Task 11: HuggingFace Dataset Upload + First SFT Training

**Files:**
- Create: `scripts/upload_dataset.py`
- Create: `scripts/train_sft.py`

**Step 1: Create dataset upload script**

```python
# scripts/upload_dataset.py
"""Upload memory traces to HuggingFace as SFT dataset."""
import subprocess
import sys

DATASET_ID = "erik1988/elias-memory-traces-v1"
TRACES_DIR = "../repos/SelfEvolvingFramework/data/traces/"
EXPORT_FILE = "data/sft_traces.jsonl"

def main():
    # Export existing traces
    print(f"Exporting traces from {TRACES_DIR}...")

    # Use HF CLI to create dataset repo and upload
    subprocess.run(["hf", "repo", "create", DATASET_ID, "--type", "dataset", "-y"],
                   capture_output=True)
    subprocess.run(["hf", "upload", DATASET_ID, EXPORT_FILE, "--repo-type", "dataset"],
                   check=True)
    print(f"Dataset uploaded to https://huggingface.co/datasets/{DATASET_ID}")

if __name__ == "__main__":
    main()
```

**Step 2: Create SFT training script (HF Jobs)**

```python
# scripts/train_sft.py
"""Launch SFT training on HuggingFace Jobs.

Usage: python scripts/train_sft.py
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12",
#     "transformers>=4.45",
#     "datasets>=3.0",
#     "torch>=2.4",
#     "peft>=0.13",
#     "accelerate>=1.0",
# ]
# ///

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

DATASET_ID = "erik1988/elias-memory-traces-v1"
BASE_MODEL = "Qwen/Qwen2.5-3B"
OUTPUT_MODEL = "erik1988/elias-memory-agent-v1"

def main():
    dataset = load_dataset(DATASET_ID, split="train")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.push_to_hub()
    print(f"Model pushed to https://huggingface.co/models/{OUTPUT_MODEL}")

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add scripts/
git commit -m "feat: HF dataset upload + SFT training scripts"
```

---

### Task 12: Final — Push to GitHub + Start Training

**Step 1: Create GitHub repo**

```bash
cd ~/repos/elias-memory
gh repo create Wittmann1988/elias-memory --public --source=. --push
```

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests pass

**Step 3: Upload dataset to HuggingFace**

```bash
python scripts/upload_dataset.py
```

**Step 4: Launch SFT training via HF Jobs**

Use the `hugging-face-model-trainer` skill to launch training on cloud GPU.

**Step 5: Final commit + push**

```bash
git add -A && git commit -m "chore: v0.1.0 release — memory framework + training pipeline"
git push
```
