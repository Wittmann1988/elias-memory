from __future__ import annotations
import json
from typing import Sequence
from elias_memory.types import MemoryRecord

def export_sft(records: Sequence[MemoryRecord], path: str) -> None:
    with open(path, "w") as f:
        for rec in records:
            entry = {
                "messages": [
                    {"role": "system", "content": "You are an AI memory manager. Store, retrieve, and maintain memories efficiently."},
                    {"role": "user", "content": f"Remember this {rec.type} memory with importance {rec.importance}: {rec.content}"},
                    {"role": "assistant", "content": f"Stored {rec.type} memory (importance={rec.importance}): {rec.content}"},
                ],
                "metadata": {"type": rec.type, "importance": rec.importance, "created_at": rec.created_at.isoformat()},
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
