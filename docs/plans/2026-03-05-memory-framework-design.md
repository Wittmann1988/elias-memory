# Design: elias-memory — Clean-Room Python Memory Framework

**Datum:** 2026-03-05
**Status:** Multi-Modell-Optimiert (4 Modelle: Qwen-Coder, Step-Flash, Nemotron, Kimi-K2)

## Vision
Das beste Memory Framework fuer AI-Agenten. Erst fuer Elias, dann Open-Source.

## Multi-Modell-Konsens (Korrekturen am Originaldesign)

| Original | Korrektur | Quelle |
|----------|-----------|--------|
| NVIDIA NIM 2048-dim | **384-dim** (all-MiniLM-L6-v2 oder NVIDIA) — schneller, reicht | Kimi + Nemotron |
| 30d Halbwertzeit | **7d Start**, spaeter tunable | Kimi |
| 4 Memory-Typen sofort | **v1: 2 Typen** (semantic + episodic), v2: +procedural +meta | Kimi |
| Flat module layout | **Sub-packages mit ABCs** fuer Swappability | Qwen-Coder |
| Kein Context Manager | **with-Statement + close()** | Qwen-Coder |
| Keine Concurrency | **WAL-Mode + threading.Lock** | Qwen-Coder + Step-Flash |
| Kein Decay-Floor | **min importance = 0.01** (nie komplett vergessen) | Step-Flash |
| Hybrid sofort | **v1: Vector-only**, v2: +BM25 hybrid | Kimi |

## Architektur (v1 — vereinfacht nach Konsens)

```
elias-memory/
├── src/elias_memory/
│   ├── __init__.py           # Public API: Memory, MemoryRecord
│   ├── core.py               # Memory Fassade (add, recall, reinforce, decay_cycle, export)
│   ├── store/
│   │   ├── __init__.py
│   │   ├── db.py             # SQLite + WAL + schema init
│   │   ├── schema.sql        # CREATE TABLE statements
│   │   └── vec.py            # VectorIndex ABC + sqlite-vec impl + numpy fallback
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py           # Embedder ABC
│   │   ├── nvidia.py         # NVIDIA NIM (384-dim)
│   │   └── fallback.py       # Hash-basiert (zero-dependency)
│   ├── decay.py              # DecayStrategy ABC + ExponentialDecay (7d default)
│   ├── retrieval.py          # v1: vector-only, v2: +BM25 hybrid
│   └── export.py             # HF SFT/DPO JSONL export
├── tests/
│   ├── test_core.py
│   ├── test_store.py
│   ├── test_embeddings.py
│   ├── test_decay.py
│   └── test_retrieval.py
├── pyproject.toml
└── README.md
```

## Kern-API (nach Konsens)

```python
from elias_memory import Memory, MemoryRecord

# Context Manager
with Memory("memory.db") as mem:
    # Speichern (v1: semantic + episodic)
    mem.add("Erik bevorzugt Python", type="semantic", importance=0.9)
    mem.add("Sidekick v2.1 deployed", type="episodic", metadata={"date": "2026-03-05"})

    # Abrufen (v1: vector-only, scored)
    results: list[MemoryRecord] = mem.recall("Python Praeferenzen", top_k=5)

    # Reinforcement
    mem.reinforce(results[0].id)

    # Decay (automatisch oder manuell)
    mem.decay_cycle()

    # Export
    mem.export_sft("traces.jsonl")
```

## Datenmodell

```python
@dataclass
class MemoryRecord:
    id: str                    # UUID4
    content: str
    type: str                  # "semantic" | "episodic" (v1)
    importance: float          # 0.01 - 1.0 (floor bei 0.01)
    created_at: datetime
    accessed_at: datetime
    access_count: int
    metadata: dict[str, Any]
    embedding: bytes | None    # 384-dim float32
```

## Schema (SQLite + WAL)

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

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

## Decay-Formel (korrigiert)

```python
score = max(0.01, importance * access_weight * exp(-lambda * days))
lambda = ln(2) / half_life  # default: 7 Tage
access_weight = 1 + log(access_count + 1)
```

## ABCs fuer Swappability

```python
class VectorIndex(ABC):
    def add(self, id: str, vec: ndarray) -> None: ...
    def search(self, query_vec: ndarray, top_k: int) -> list[tuple[str, float]]: ...
    def delete(self, id: str) -> None: ...

class Embedder(ABC):
    def embed(self, text: str) -> ndarray: ...
    @property
    def dim(self) -> int: ...

class DecayStrategy(ABC):
    def compute(self, record: MemoryRecord) -> float: ...
```

## Abhaengigkeiten (minimal)

```toml
[project]
name = "elias-memory"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "httpx>=0.24",
]

[project.optional-dependencies]
vec = ["sqlite-vec>=0.1"]
full = ["sqlite-vec>=0.1", "rank-bm25>=0.2"]
```

## Training-Pipeline (parallel)

1. 24 Traces aus SelfEvolvingFramework -> HF Dataset
2. SFT auf Memory-Operationen (Qwen2.5-3B base)
3. Benchmark gegen Baseline (80/100)
4. GGUF -> Ollama

## v1 Scope (YAGNI)

**Drin:** Memory CRUD, vector retrieval, decay, reinforcement, SFT export, 2 Typen
**Nicht drin (v2):** BM25 hybrid, procedural/meta Typen, async API, learning-to-rank, metrics
