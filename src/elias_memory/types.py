from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

VALID_TYPES = frozenset({"semantic", "episodic"})
VALID_SCOPES = frozenset({"shared", "project", "agent", "session"})

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
    namespace: str = "global"
    scope: str = "shared"

    def __post_init__(self):
        if self.type not in VALID_TYPES:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of: {VALID_TYPES}")
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"Invalid scope '{self.scope}'. Must be one of: {VALID_SCOPES}")
        self.importance = max(0.01, min(1.0, self.importance))
