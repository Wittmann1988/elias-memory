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
    assert 0.99 < score <= 1.0

def test_exponential_decay_after_half_life():
    decay = ExponentialDecay(half_life_days=7)
    rec = MemoryRecord(content="old", type="semantic", importance=1.0)
    rec.accessed_at = datetime.now(timezone.utc) - timedelta(days=7)
    score = decay.compute(rec)
    assert 0.45 < score < 0.55

def test_exponential_decay_floor():
    decay = ExponentialDecay(half_life_days=7)
    rec = MemoryRecord(content="ancient", type="semantic", importance=0.1)
    rec.accessed_at = datetime.now(timezone.utc) - timedelta(days=365)
    score = decay.compute(rec)
    assert score == 0.01

def test_exponential_decay_access_boost():
    decay = ExponentialDecay(half_life_days=7)
    rec = MemoryRecord(content="popular", type="semantic", importance=0.5)
    rec.access_count = 10
    rec.accessed_at = datetime.now(timezone.utc) - timedelta(days=7)
    score = decay.compute(rec)
    assert score > 0.5 * 0.5
