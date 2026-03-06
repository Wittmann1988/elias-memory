# BEFORE YOU DO ANYTHING: RUN `elias-mem check "your action"`

## THIS IS NOT OPTIONAL. THIS IS A HARD RULE.

Every agent, every session, every action — check your goals FIRST.
If the check returns BLOCKED, you do NOT proceed. Fix the violation first.

```bash
# Before implementing anything:
elias-mem check "Implement new feature X"

# Before designing anything:
elias-mem check "Design new architecture for Y"

# See all active goals:
elias-mem goals
```

**If you skip this check, you are violating the fundamental rules.**

---

# elias-memory

Persistent memory framework for AI agents. Pure Python, pure SQLite, works everywhere.

## Features (v0.3.0)

| Feature | Description |
|---------|-------------|
| **4-Tier Memory** | Buffer, Episodic, Semantic, Procedural |
| **Knowledge Graph** | Relations between memories (causes, supports, contradicts) |
| **Causal Chains** | Follow cause-effect relationships across memories |
| **Entity Extraction** | Auto-extract entities from content on add() |
| **Namespaced Access** | Each agent only sees what it needs (global/project/agent/session) |
| **Goal Guard** | Enforced goal-checking before every action |
| **Consolidation** | Merge episodic memories into semantic knowledge |
| **Knowledge Gaps** | Detect topics with low coverage |
| **Exponential Decay** | Memories fade unless reinforced |
| **SFT Export** | Export memories as training data |

## Install

```bash
pip install -e .          # Basic (mobile)
pip install -e ".[full]"  # Desktop (FAISS + sqlite-vec)
```

## Quick Start

```python
from elias_memory import Memory

mem = Memory("my.db", namespace="agent/my-agent",
             namespaces=["global", "agent/my-agent"])

# ALWAYS check goals first
result = mem.guard.check("Add new memories")
print(result.format())

# Add with auto entity extraction
mid = mem.add("Python and SQLite are fast",
              type="semantic", importance=0.8)

# Create relations
mid2 = mem.add("SQLite uses WAL mode for concurrency",
               type="semantic", importance=0.7)
mem.graph.link(mid, mid2, "relates_to")

# Recall with graph expansion
results = mem.recall("database performance", graph_expand=True)

# Causal chain
mem.graph.causal_chain(mid, direction="backward")

# Entity search
mem.graph.get_memories_by_entity("python")
```

## Architecture

```
No server needed. No GPU needed. No internet needed.
Just SQLite + numpy. Works on any device.

Desktop (optional):  NVIDIA embeddings + FAISS
Mobile (default):    Hash embeddings + numpy
Jetson (bonus):      Permanent inference server
```

## CLI

```bash
elias-mem store "Python is great" --importance 0.9
elias-mem query "Python" --expand
elias-mem check "Build new feature"
elias-mem goals
elias-mem stats
elias-mem consolidate
elias-mem gaps
```

## Tests

```bash
pytest tests/ -v  # 68 tests
```

## License

MIT
