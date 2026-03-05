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
