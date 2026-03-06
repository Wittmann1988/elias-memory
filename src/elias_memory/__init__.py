"""elias-memory: Persistent memory framework for AI agents.

Desktop/Mobile 2-tier architecture:
- Desktop: NVIDIA embeddings, FAISS index, full consolidation
- Mobile: Hash embeddings, numpy brute-force, basic features
"""

from elias_memory.core import Memory
from elias_memory.types import MemoryRecord
from elias_memory.gaps import KnowledgeGap

__version__ = "0.2.0"
__all__ = ["Memory", "MemoryRecord", "KnowledgeGap"]
