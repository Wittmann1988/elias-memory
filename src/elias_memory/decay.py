from __future__ import annotations
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from elias_memory.types import MemoryRecord

FLOOR = 0.01

class DecayStrategy(ABC):
    @abstractmethod
    def compute(self, record: MemoryRecord) -> float: ...

class ExponentialDecay(DecayStrategy):
    def __init__(self, half_life_days: float = 7.0) -> None:
        self._lambda = math.log(2) / half_life_days

    def compute(self, record: MemoryRecord) -> float:
        now = datetime.now(timezone.utc)
        days = (now - record.accessed_at).total_seconds() / 86400
        access_weight = 1.0 + math.log(record.access_count + 1)
        raw = record.importance * access_weight * math.exp(-self._lambda * days)
        return max(FLOOR, min(1.0, raw))
